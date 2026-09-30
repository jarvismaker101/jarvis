"""[P1-09] Every prefetch resolves the selected engine first.

Two call sites in ``voice.py`` called ``prefetch_fish_audio`` directly while a
third used the engine-aware ``_prefetch_tts_audio``. With Google selected, the
direct calls spent metered Fish API credits AND left the selected engine cold.

These tests pin the contract of the shared helper:

* the engine is resolved once per call and only that engine is warmed;
* an engine with no prefetch implementation is a silent no-op — never a crash
  and never a Fish call;
* the P0-05 rule (never prefetch the sentence about to play while idle) is
  applied inside the helper, so all call sites inherit it;
* no call site in ``voice.py`` bypasses the helper.
"""

import os
import re
import unittest
from unittest.mock import patch

from backend.services import voice as voice_mod

VOICE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "services", "voice.py")


class EngineResolutionTests(unittest.TestCase):
    """Only the SELECTED engine may be warmed."""

    def test_a_non_fish_provider_never_issues_a_fish_prefetch(self):
        """The metered-provider concern: Fish credits are spent only on request."""
        with patch.object(voice_mod, "_resolve_tts_provider", return_value="gtts"), \
                patch.object(voice_mod, "prefetch_fish_audio") as fish_prefetch, \
                patch.object(voice_mod, "prefetch_google_tts") as google_prefetch:
            voice_mod._prefetch_tts_audio("next sentence")

        fish_prefetch.assert_not_called()
        google_prefetch.assert_called_once_with("next sentence")

    def test_fish_selected_still_prefetches_through_the_same_helper(self):
        with patch.object(voice_mod, "_resolve_tts_provider", return_value="fish"), \
                patch.object(voice_mod, "prefetch_fish_audio") as fish_prefetch, \
                patch.object(voice_mod, "prefetch_google_tts") as google_prefetch:
            voice_mod._prefetch_tts_audio("next sentence")

        fish_prefetch.assert_called_once_with("next sentence")
        google_prefetch.assert_not_called()

    def test_an_engine_without_a_prefetch_implementation_is_a_silent_no_op(self):
        """Unknown engine: do nothing. Never fall back to a different provider."""
        for provider in ("elevenlabs", "sapi5", "", None):
            with self.subTest(provider=provider):
                with patch.object(voice_mod, "_resolve_tts_provider",
                                  return_value=provider), \
                        patch.object(voice_mod, "prefetch_fish_audio") as fish_prefetch, \
                        patch.object(voice_mod, "prefetch_google_tts") as google_prefetch:
                    voice_mod._prefetch_tts_audio("next sentence")

                fish_prefetch.assert_not_called()
                google_prefetch.assert_not_called()

    def test_the_engine_is_resolved_exactly_once_per_call(self):
        """One call, one decision: a settings change mid-call cannot split it."""
        with patch.object(voice_mod, "_resolve_tts_provider",
                          return_value="fish") as resolve, \
                patch.object(voice_mod, "prefetch_fish_audio"):
            voice_mod._prefetch_tts_audio("next sentence")

        self.assertEqual(resolve.call_count, 1)

    def test_the_engines_with_a_prefetch_implementation_are_fish_and_gtts(self):
        """A new provider must be added here deliberately, not by accident."""
        self.assertEqual(set(voice_mod._prefetch_implementations()),
                         {"fish", "gtts"})

    def test_the_map_is_resolved_at_call_time_not_captured_at_import(self):
        """Established convention in this module (see `_cloud_tts_ladder`).

        The whole suite patches ``voice.prefetch_*``; a captured reference would
        silently defeat every one of those patches — and, worse, would make a
        "no Fish call happened" test pass while a real one ran.
        """
        with patch.object(voice_mod, "_resolve_tts_provider", return_value="fish"), \
                patch.object(voice_mod, "prefetch_fish_audio") as replacement:
            voice_mod._prefetch_tts_audio("next sentence")

        replacement.assert_called_once_with("next sentence")

    def test_blank_text_is_ignored_before_any_resolution(self):
        with patch.object(voice_mod, "_resolve_tts_provider") as resolve, \
                patch.object(voice_mod, "prefetch_fish_audio") as fish_prefetch:
            for text in ("", "   ", None):
                voice_mod._prefetch_tts_audio(text)

        resolve.assert_not_called()
        fish_prefetch.assert_not_called()



class SharedRuleTests(unittest.TestCase):
    """The P0-05 timing rule is applied by the helper for EVERY engine."""

    def test_a_sentence_about_to_play_is_not_prefetched_while_idle(self):
        for provider, target in (("fish", "prefetch_fish_audio"),
                                 ("gtts", "prefetch_google_tts")):
            with self.subTest(provider=provider):
                with patch.object(voice_mod, "_resolve_tts_provider",
                                  return_value=provider), \
                        patch.object(voice_mod, "is_speaking", False), \
                        patch.object(voice_mod, target) as warm:
                    voice_mod._prefetch_tts_audio("about to play", plays_next=True)

                warm.assert_not_called()

    def test_the_rule_yields_when_something_is_already_playing(self):
        """While sentence A plays, warming sentence B is the whole point."""
        for provider, target in (("fish", "prefetch_fish_audio"),
                                 ("gtts", "prefetch_google_tts")):
            with self.subTest(provider=provider):
                with patch.object(voice_mod, "_resolve_tts_provider",
                                  return_value=provider), \
                        patch.object(voice_mod, "is_speaking", True), \
                        patch.object(voice_mod, target) as warm:
                    voice_mod._prefetch_tts_audio("plays next", plays_next=True)

                warm.assert_called_once_with("plays next")

    def test_a_sentence_that_cannot_play_next_is_always_warmed(self):
        with patch.object(voice_mod, "_resolve_tts_provider", return_value="fish"), \
                patch.object(voice_mod, "is_speaking", False), \
                patch.object(voice_mod, "prefetch_fish_audio") as warm:
            voice_mod._prefetch_tts_audio("later sentence")

        warm.assert_called_once_with("later sentence")


class NoBypassTests(unittest.TestCase):
    """Every call site goes through the helper (the actual P1-09 defect)."""

    def _source_without_the_dispatcher(self):
        with open(VOICE_PATH, encoding="utf-8") as handle:
            source = handle.read()
        # The dispatcher is the ONE place allowed to name an engine prefetch.
        match = re.search(r"def _prefetch_tts_audio\(.*?(?=\ndef |\nclass )",
                          source, re.S)
        self.assertIsNotNone(match, "the dispatcher must exist")
        return source[:match.start()] + source[match.end():]

    def test_the_dispatcher_still_maps_both_engines(self):
        with open(VOICE_PATH, encoding="utf-8") as handle:
            source = handle.read()
        match = re.search(r"def _prefetch_implementations\(.*?\n\n\n", source, re.S)
        self.assertIsNotNone(match, "the implementation map must exist")
        body = match.group(0)
        self.assertIn('"fish": prefetch_fish_audio', body)
        self.assertIn('"gtts": prefetch_google_tts', body)

    def test_voice_py_never_calls_an_engine_prefetch_directly(self):
        body = self._source_without_the_dispatcher()
        offenders = re.findall(r"prefetch_(?:fish|google)[a-z_]*\s*\(", body)
        self.assertEqual(offenders, [],
                         "direct engine prefetch calls bypass the engine-aware "
                         "helper: %r" % offenders)

    def test_the_queue_call_sites_pass_the_plays_next_fact(self):
        """The rule needs one fact only the queue owner has."""
        with open(VOICE_PATH, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        calls = [line for line in lines
                 if "_prefetch_tts_audio(" in line
                 and not line.strip().startswith("def ")]
        self.assertEqual(len(calls), 3, "expected three call sites: %r" % calls)
        gated = [line for line in calls if "plays_next=(pending_ahead == 0)" in line]
        self.assertEqual(len(gated), 2,
                         "both _enqueue call sites must pass plays_next: %r" % calls)
        # The speak() loop warms the chunk AFTER the one it is about to speak,
        # so it is never the next to play and needs no gate.
        ungated = [line for line in calls if line not in gated]
        self.assertIn("chunks[index + 1]", ungated[0])


if __name__ == "__main__":
    unittest.main()
