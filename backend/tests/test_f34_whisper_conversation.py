"""F34 — reuse local Whisper for active conversation.

Acceptance (audit report): "duplicates/reordering/contradiction/cross-turn
windows never authorize unstable actions; local-only failure transmits nothing
externally."

The baseline defects pinned here:
  * ``TranscriptWindow.overlaps`` returned True when timestamps were missing,
    so unverifiable windows counted as corroboration;
  * agreement counted any prior agreeing window (duplicates included, and
    non-consecutive agreement across a contradiction);
  * a local engine choice did not establish a local-only privacy policy — the
    conversation path still fell through to cloud STT on local failure.

The audit's FIRST acceptance clause ("partials arrive before the utterance
ends") is deliberately NOT pinned here any more: the partial-window layer was
removed in the 2026-10 latency pass (tag ``simple-listening``), so one utterance
now costs exactly one transcription of the complete audio. The stabilizer rules
below are unchanged and still govern the final commit.

No microphone, TTS engine, wake engine or subprocess is opened here.
"""

import os
import unittest
from unittest.mock import patch

import speech_recognition as sr

from backend.services import listener
from backend.services.transcript_stabilizer import (
    TranscriptStabilizer,
    TranscriptWindow,
)


class StabilizerRuleTests(unittest.TestCase):
    """Acceptance: duplicates/reordering/contradiction/cross-turn windows
    never authorize unstable actions."""

    def setUp(self):
        self.s = TranscriptStabilizer()

    def _push(self, wid, text, start, end, final=False, turn=None):
        return self.s.push(TranscriptWindow(
            wid, text, final=final, start_ms=start, end_ms=end, turn=turn))

    def test_consecutive_agreeing_windows_commit(self):
        self._push("p1", "open chrome", 0, 1000)
        stable = self._push("p2", "open chrome", 100, 1100)
        self.assertIsNotNone(stable)
        self.assertEqual(self.s.committed(), "open chrome")

    def test_duplicate_window_id_is_not_a_second_opinion(self):
        self._push("p1", "open chrome", 0, 1000)
        self.assertIsNone(self._push("p1", "open chrome", 0, 1000))
        self.assertEqual(self.s.committed(), "")
        self.assertGreaterEqual(self.s.counters()["duplicates"], 1)

    def test_identical_audio_range_is_not_independent_evidence(self):
        self._push("p1", "open chrome", 0, 1000)
        self.assertIsNone(self._push("p2", "open chrome", 0, 1000))
        self.assertEqual(self.s.committed(), "")

    def test_missing_timestamps_are_not_an_overlap(self):
        self._push("p1", "open chrome", None, None)
        self.assertIsNone(self._push("p2", "open chrome", None, None))
        self.assertEqual(self.s.committed(), "")

    def test_contradiction_breaks_consecutive_agreement(self):
        self._push("p1", "open chrome", 0, 1000)
        self._push("p2", "open browser", 100, 1100)
        self.assertIsNone(self._push("p3", "open chrome", 200, 1200))
        self.assertEqual(self.s.committed(), "")

    def test_late_reordered_window_is_stale(self):
        self._push("p1", "open chrome", 0, 2000)
        self._push("p2", "open chrome", 100, 2100)
        # A window that ends before the newest accepted window is a late
        # arrival, not new corroboration.
        self.assertIsNone(self._push("p0", "open chrome", 0, 500))
        self.assertGreaterEqual(self.s.counters()["stale"], 1)

    def test_windows_from_another_turn_never_combine(self):
        self._push("p1", "open chrome", 0, 1000)  # turn 0
        self.s.begin_turn()                       # turn 1
        # A straggler window from the PREVIOUS turn (a late transcription of
        # the previous utterance) can never corroborate this turn.
        self.assertIsNone(self._push("p1b", "open chrome", 0, 1000, turn=0))
        self.assertEqual(self.s.committed(), "")
        self.assertGreaterEqual(self.s.counters()["cross_turn"], 1)

    def test_begin_turn_drops_the_previous_commitment(self):
        self._push("f1", "open chrome", 0, 1000, final=True)
        self.assertEqual(self.s.committed(), "open chrome")
        self.s.begin_turn()
        self.assertEqual(self.s.committed(), "")
        self.assertFalse(self.s.is_committed("open chrome"))

    def test_a_final_window_commits_exactly_once(self):
        first = self._push("f1", "open chrome", 0, 1000, final=True)
        self.assertIsNotNone(first)
        self.assertIsNone(self._push("f1", "open chrome", 0, 1000, final=True))
        self.assertEqual(self.s.commit_count(), 1)
        self.assertEqual(self.s.committed(), "open chrome")

    def test_non_overlapping_windows_never_agree(self):
        self._push("p1", "open chrome", 0, 1000)
        self.assertIsNone(self._push("p2", "open chrome", 5000, 6000))
        self.assertEqual(self.s.committed(), "")

    def test_reset_clears_every_rule_state(self):
        self._push("f1", "open chrome", 0, 1000, final=True)
        self.s.reset()
        self.assertEqual(self.s.committed(), "")
        self.assertEqual(self.s.commit_count(), 0)
        self.assertEqual(self.s.counters()["duplicates"], 0)


class CloudPolicyTests(unittest.TestCase):
    """Acceptance: local-only failure transmits nothing externally."""

    def setUp(self):
        self.inworld = patch.object(listener, "recognize_inworld")
        self.local = patch.object(listener, "recognize_local_whisper")
        self.online = patch.object(listener, "recognize_google_or_groq")
        self.mock_inworld = self.inworld.start()
        self.mock_local = self.local.start()
        self.mock_online = self.online.start()
        for p in (self.inworld, self.local, self.online):
            self.addCleanup(p.stop)
        os.environ.pop("JARVIS_STT_CLOUD_POLICY", None)
        os.environ.pop("JARVIS_STT_LOCAL_ONLY", None)
        self.addCleanup(os.environ.pop, "JARVIS_STT_CLOUD_POLICY", None)
        self.addCleanup(os.environ.pop, "JARVIS_STT_LOCAL_ONLY", None)

    def test_local_only_failure_makes_no_external_call(self):
        os.environ["JARVIS_STT_CLOUD_POLICY"] = "off"
        self.mock_local.side_effect = sr.RequestError("local whisper down")
        result = listener.recognize_multilingual(object())
        self.assertEqual(result, (None, None, None))
        self.mock_inworld.assert_not_called()
        self.mock_online.assert_not_called()

    def test_local_only_success_uses_the_local_engine_only(self):
        os.environ["JARVIS_STT_CLOUD_POLICY"] = "local-only"
        self.mock_local.return_value = ("Open C:\\Temp\\A.TXT", "en")
        raw, normalized, language = listener.recognize_multilingual(object())
        self.assertEqual(raw, "Open C:\\Temp\\A.TXT")
        self.assertEqual(normalized, "open c:\\temp\\a.txt")
        self.assertEqual(language, "en")
        self.mock_inworld.assert_not_called()
        self.mock_online.assert_not_called()

    def test_default_policy_runs_only_the_selected_engine(self):
        """[P0-03] The Google/Groq rung of the ladder is gone.

        The old assertion ("default policy still reaches cloud engines") required
        a fall THROUGH Inworld and local whisper into the per-language
        Google/Groq loop — the serial path whose worst case was 117s. With one
        engine per turn, an Inworld failure is the end of the turn.
        """
        self.mock_local.side_effect = sr.RequestError("local whisper down")
        self.mock_inworld.side_effect = sr.RequestError("no inworld key")
        self.mock_online.return_value = "Create File Hello.txt"
        with patch.object(listener.model_registry, "get_model_for_role",
                          return_value={"provider": "inworld"}):
            raw, normalized, language = listener.recognize_multilingual(object())
        self.assertEqual((raw, normalized, language), (None, None, None))
        self.mock_online.assert_not_called()
        self.mock_local.assert_not_called()
        self.assertEqual(listener.LAST_STT_FAILURE[0], "inworld")


if __name__ == "__main__":
    unittest.main()
