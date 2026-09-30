"""P1-08 — the settings file must be parsed once per CHANGE, not per audio call.

``_resolve_role`` read and parsed ``data/jarvis_settings.json`` under a lock on
every resolve, and the audio path resolves roles several times PER CHUNK. A read
error also returned ``{}``, which silently flipped every role to its env default
— the user's whole configuration disappeared with only a log line.
"""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.services import model_registry
from backend.services import voice as voice_mod


def _forget_cache():
    """Drop the registry's parsed-settings cache if it has one.

    Tolerant on purpose: on the pre-fix tree there is nothing to drop, and the
    tests still have to reach their real assertion.
    """
    reset = getattr(model_registry, "_forget_cached_settings_unlocked", None)
    if reset is not None:
        reset()


class _SettingsFileTestCase(unittest.TestCase):
    """Points the registry at a scratch settings file it fully controls."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "jarvis_settings.json"
        self._orig = model_registry.SETTINGS_FILE
        model_registry.SETTINGS_FILE = self.path
        _forget_cache()
        self.addCleanup(self._restore)

    def _restore(self):
        model_registry.SETTINGS_FILE = self._orig
        _forget_cache()

    def write(self, payload):
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        # The cache key is (mtime_ns, size): give every write its own, strictly
        # increasing timestamp, so a same-size rewrite still looks like a
        # different file exactly as a real save would. (time.time() alone is too
        # coarse on Windows: two writes inside one tick share a timestamp.)
        self._writes = getattr(self, "_writes", 0) + 1
        stamp = time.time() + self._writes
        os.utime(self.path, (stamp, stamp))
        return payload


class ParseCountTests(_SettingsFileTestCase):
    """Stat is cheap; parse is not. Only a CHANGE may re-parse."""

    def test_repeated_resolves_parse_the_file_exactly_once(self):
        self.write({"tts_model": {"provider": "fish", "model": "s1"}})
        parses = []
        real_load = json.load

        def counting_load(stream):
            parses.append(1)
            return real_load(stream)

        with patch.object(model_registry.json, "load", side_effect=counting_load):
            for _ in range(25):
                model_registry._load_settings()
                model_registry.get_model_for_role("tts")

        self.assertEqual(len(parses), 1,
                         "the settings file was re-parsed on every call")

    def test_a_changed_file_is_re_parsed_on_the_very_next_resolve(self):
        """The live-switching guarantee: a UI change applies immediately."""
        self.write({"tts_model": {"provider": "fish", "model": "s1"}})
        self.assertEqual(
            model_registry.get_model_for_role("tts").get("model"), "s1")

        self.write({"tts_model": {"provider": "fish", "model": "s2"}})

        self.assertEqual(
            model_registry.get_model_for_role("tts").get("model"), "s2",
            "a settings change did not take effect without a restart")

    def test_a_same_size_rewrite_is_still_noticed(self):
        """mtime_ns is part of the key, so same-length values still switch."""
        self.write({"tts_model": {"provider": "fish", "model": "aa"}})
        self.assertEqual(model_registry.get_model_for_role("tts")["model"], "aa")

        self.write({"tts_model": {"provider": "fish", "model": "bb"}})

        self.assertEqual(model_registry.get_model_for_role("tts")["model"], "bb")

    def test_callers_get_their_own_copy_of_the_settings(self):
        """A read-modify-write caller must not corrupt the cache."""
        self.write({"tts_model": {"provider": "fish", "model": "s1"}})

        settings = model_registry._load_settings()
        settings["tts_model"] = {"provider": "gtts"}

        self.assertEqual(
            model_registry.get_model_for_role("tts").get("provider"), "fish",
            "a caller's mutation leaked into the cached settings")

    def test_a_missing_file_is_still_an_empty_configuration(self):
        self.assertFalse(self.path.exists())

        self.assertEqual(model_registry._load_settings(), {})


class LastGoodCopyTests(_SettingsFileTestCase):
    """[P1-08] A read failure must not silently revert the user's settings."""

    def test_a_corrupt_file_falls_back_to_the_last_good_copy(self):
        """THE regression test for the silent-defaults bug."""
        self.write({"tts_model": {"provider": "fish", "model": "s1"}})
        self.assertEqual(model_registry.get_model_for_role("tts")["model"], "s1")

        self.path.write_text("{not json", encoding="utf-8")

        self.assertEqual(
            model_registry.get_model_for_role("tts").get("model"), "s1",
            "a corrupt read silently discarded the user's settings")

    def test_the_fallback_is_logged(self):
        self.write({"tts_model": {"provider": "fish", "model": "s1"}})
        model_registry._load_settings()
        self.path.write_text("{not json", encoding="utf-8")

        with self.assertLogs(level="WARNING") as captured:
            model_registry._load_settings()

        self.assertTrue(any("settings" in line.lower()
                            for line in captured.output),
                        "the fallback must be visible in the log")

    def test_a_first_ever_failure_returns_empty_and_warns(self):
        _forget_cache()
        self.path.write_text("{not json", encoding="utf-8")

        with self.assertLogs(level="WARNING") as captured:
            self.assertEqual(model_registry._load_settings(), {})

        self.assertTrue(captured.output,
                        "the first-ever fallback must be logged")

    def test_an_unreadable_file_keeps_the_last_good_copy(self):
        self.write({"tts_model": {"provider": "fish", "model": "s1"}})
        model_registry._load_settings()
        self.path.unlink()
        self.path.mkdir()          # a directory in place of the file: read fails

        try:
            with self.assertLogs(level="WARNING"):
                settings = model_registry._load_settings()
        finally:
            self.path.rmdir()

        self.assertEqual(settings.get("tts_model", {}).get("model"), "s1")

    def test_a_different_file_never_lends_its_copy(self):
        """The last good copy is kept PER PATH, not globally."""
        self.write({"tts_model": {"provider": "fish", "model": "s1"}})
        model_registry._load_settings()

        other = Path(self._tmp.name) / "other_settings.json"
        other.write_text("{not json", encoding="utf-8")
        model_registry.SETTINGS_FILE = other

        with self.assertLogs(level="WARNING"):
            self.assertEqual(model_registry._load_settings(), {})

    def test_writing_settings_adopts_them_without_a_reparse(self):
        model_registry.set_model_for_role("tts", "fish", "s-written")

        parses = []
        real_load = json.load

        def counting_load(stream):
            parses.append(1)
            return real_load(stream)

        with patch.object(model_registry.json, "load", side_effect=counting_load):
            self.assertEqual(
                model_registry.get_model_for_role("tts").get("model"),
                "s-written")

        self.assertEqual(parses, [],
                         "our own write was immediately re-parsed")


class ReplySessionProviderTests(unittest.TestCase):
    """[P1-08] The engine is resolved ONCE per reply, not per chunk."""

    def test_a_speaker_resolves_the_engine_only_once(self):
        speaker = voice_mod.StreamSpeaker()
        self.addCleanup(speaker.close)

        with patch.object(voice_mod, "_resolve_tts_provider",
                          return_value="gtts") as resolve, \
             patch.object(voice_mod, "_prefetch_tts_audio"), \
             patch.object(voice_mod, "_speak_chunk"):
            for _ in range(6):
                speaker.feed("One two three four five six. ")

        self.assertEqual(resolve.call_count, 1,
                         "the engine was resolved again mid-reply")

    def test_a_new_reply_session_sees_the_new_selection(self):
        """Resolved once per session, but sessions still see changes."""
        first = voice_mod.StreamSpeaker()
        self.addCleanup(first.close)
        with patch.object(voice_mod, "_resolve_tts_provider",
                          return_value="fish"):
            self.assertEqual(first._tts_engine(), "fish")

        second = voice_mod.StreamSpeaker()
        self.addCleanup(second.close)
        with patch.object(voice_mod, "_resolve_tts_provider",
                          return_value="gtts"):
            self.assertEqual(second._tts_engine(), "gtts",
                             "a new reply kept the previous reply's engine")

    def test_the_ladder_and_the_chunk_limit_share_one_provider(self):
        with patch.object(voice_mod, "_resolve_tts_provider",
                          return_value="gtts") as resolve:
            ladder = voice_mod._cloud_tts_ladder("gtts")
            limit = voice_mod._tts_char_limit("gtts")

        self.assertEqual(ladder[0][0], "google")
        self.assertEqual(limit, voice_mod.GOOGLE_TTS_CHAR_LIMIT)
        self.assertEqual(resolve.call_count, 0,
                         "a session provider must not be re-resolved")

    def test_an_unknown_provider_resolves_lazily(self):
        """A bare call still resolves: the cache is the SESSION, not a global."""
        with patch.object(voice_mod, "_resolve_tts_provider",
                          return_value="gtts") as resolve:
            self.assertEqual(voice_mod._cloud_tts_ladder()[0][0], "google")
            self.assertEqual(voice_mod._tts_char_limit(),
                             voice_mod.GOOGLE_TTS_CHAR_LIMIT)

        self.assertEqual(resolve.call_count, 2,
                         "each unresolved call must ask the registry")


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
