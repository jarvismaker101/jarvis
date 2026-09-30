"""F34 — reuse local Whisper for active conversation.

Acceptance (audit report): "Partials arrive before utterance end;
duplicates/reordering/contradiction/cross-turn windows never authorize
unstable actions; local-only failure transmits nothing externally."

The baseline defects pinned here:
  * capture completed before the FIRST (and only) transcription, and the only
    stabilizer window ever produced was the final one — there were no real
    overlapping partial windows at all;
  * ``TranscriptWindow.overlaps`` returned True when timestamps were missing,
    so unverifiable windows counted as corroboration;
  * agreement counted any prior agreeing window (duplicates included, and
    non-consecutive agreement across a contradiction);
  * a local engine choice did not establish a local-only privacy policy — the
    conversation path still fell through to cloud STT on local failure.

No microphone, TTS engine, wake engine or subprocess is opened here.
"""

import os
import unittest
from unittest.mock import patch

import speech_recognition as sr

from backend.services import listener
from backend.services import transcript_stabilizer
from backend.services.transcript_stabilizer import (
    TranscriptStabilizer,
    TranscriptWindow,
)


def _chunk(seconds=0.5, rate=16000):
    return sr.AudioData(b"\x00" * int(rate * seconds) * 2, rate, 2)


class _ChunkStream:
    """Iterable capture stream that records how many chunks were consumed."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.consumed = 0

    def __iter__(self):
        for chunk in self.chunks:
            self.consumed += 1
            yield chunk


class PartialWindowTests(unittest.TestCase):
    """Acceptance: partials arrive BEFORE the utterance ends."""

    def setUp(self):
        listener._turn_stabilizer.reset()
        self.stream = _ChunkStream([_chunk() for _ in range(4)])
        self.observed = []
        self._observer = None

    def tearDown(self):
        if self._observer is not None:
            listener.unregister_partial_observer(self._observer)
        listener._turn_stabilizer.reset()

    def _run_capture(self, transcript="open chrome"):
        def observer(window):
            # TranscriptWindow is __slots__-based: record the observation
            # around it rather than decorating the window itself.
            self.observed.append({
                "window": window,
                "at_chunk": self.stream.consumed,
                "committed_then": listener._turn_stabilizer.committed(),
            })

        listener.register_partial_observer(observer)
        self._observer = observer

        if isinstance(transcript, (list, tuple)):
            remaining = list(transcript)

            def local_whisper(_audio, timeout=None):
                # [P0-04] The partial engine is called WITH a deadline: the
                # no-deadline TypeError fallback was removed so an engine that
                # cannot express one is never handed work it can hang on. The
                # assertions in this file are unchanged.
                text = remaining.pop(0) if len(remaining) > 1 else remaining[0]
                return text, "en"
        else:
            def local_whisper(_audio, timeout=None):
                return transcript, "en"

        class _Source:
            stream = object()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        patches = [
            patch.object(listener, "_get_microphone_source",
                         return_value=_Source()),
            patch.object(listener.recognizer, "listen",
                         return_value=self.stream),
            patch.object(listener, "_aec_filter_chunk",
                         side_effect=lambda chunk, fid, t: (chunk, False,
                                                            False)),
            patch.object(listener, "_should_confirm_speech_start",
                         return_value=True),
            patch.object(listener, "barge_in_on_speech_onset"),
            patch.object(listener, "play_capture_complete_earcon"),
            patch.object(listener, "_recalibrate_listener"),
            patch.object(listener, "is_human_voice", return_value=True),
            patch.object(listener, "PARTIAL_TRANSCRIBE_MIN_SECONDS", 0.5),
            patch.object(listener, "recognize_local_whisper",
                         side_effect=local_whisper),
        ]
        for p in patches:
            p.start()
        try:
            return listener._capture_audio()
        finally:
            for p in reversed(patches):
                p.stop()

    @property
    def windows(self):
        return [item["window"] for item in self.observed]

    def test_partial_windows_are_produced_during_capture(self):
        self._run_capture()
        self.assertGreaterEqual(len(self.observed), 2,
                                "capture produced no real partial windows")
        self.assertTrue(all(not w.final for w in self.windows))

    def test_the_first_partial_arrives_before_the_utterance_ends(self):
        self._run_capture()
        total = len(self.stream.chunks)
        self.assertLess(self.observed[0]["at_chunk"], total,
                        "the first partial only appeared at utterance end")

    def test_partial_windows_are_identified_timestamped_and_turn_scoped(self):
        self._run_capture()
        windows = self.windows
        wids = [w.wid for w in windows]
        self.assertEqual(len(wids), len(set(wids)), "window ids are not unique")
        ends = [w.end_ms for w in windows]
        self.assertEqual(ends, sorted(ends), "window ranges do not advance")
        self.assertEqual(len({w.turn for w in windows}), 1,
                         "partial windows span more than one turn")
        self.assertIsNotNone(windows[0].turn)

    def test_agreeing_partials_commit_before_the_utterance_ends(self):
        self._run_capture(transcript="open chrome")
        self.assertTrue(self.observed)
        self.assertEqual(self.observed[-1]["committed_then"], "open chrome",
                         "local agreement did not commit during capture")

    def test_disagreeing_partials_never_commit_the_contradicted_text(self):
        self._run_capture(["open chrome", "open browser", "play music",
                           "play music"])
        self.assertEqual(listener._turn_stabilizer.committed(), "play music")
        committed_seen = [item["committed_then"] for item in self.observed]
        self.assertNotIn("open chrome", committed_seen)

    def test_partial_only_text_never_becomes_committed_text(self):
        # A single partial (no corroboration) can never authorize an action.
        listener._turn_stabilizer.begin_turn()
        listener._turn_stabilizer.push(TranscriptWindow(
            "p1", "delete everything", final=False, start_ms=0, end_ms=1000))
        self.assertEqual(listener._turn_stabilizer.committed(), "")
        self.assertFalse(listener._turn_stabilizer.is_committed(
            "delete everything"))
        self.assertEqual(listener._turn_stabilizer.unstable(),
                         "delete everything")


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
