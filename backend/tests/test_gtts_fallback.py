"""Google Translate TTS — the free, key-less tts fallback.

Acceptance: the user can pick a zero-cost voice engine from the sidebar's
VOICE MODEL (TTS) section and have it actually speak, without disturbing the
F32 "one playback owner" invariant that the audit established.

What is pinned here:
  * the registry offers ``gtts`` for the tts role ONLY, lists its voices
    without any credential, and resolves a selection with no API key;
  * the endpoint's ~200-character limit is respected by the splitter, which
    must not lose or reorder text;
  * the engine returns False (never raises) when it cannot produce audio, so
    the ladder can fall through instead of going silent;
  * playback goes through the SAME ``audio_actor`` owner Fish uses;
  * ``voice._speak_chunk`` tries the user's selected engine first and the
    other engine second — an explicit choice is honoured, and a dead engine
    degrades rather than silencing the reply.
"""

import math
import os
import sys
import unittest
from unittest.mock import patch

from backend.services import audio_actor
from backend.services import google_tts as gtts
from backend.services import model_registry
from backend.services import voice


class GoogleTtsRegistryTests(unittest.TestCase):
    """The provider must be selectable, key-less and role-scoped."""

    def test_offered_for_tts_only(self):
        allowed = model_registry.get_allowed_providers_for_role("tts")
        self.assertIn("gtts", allowed)
        self.assertIn("fish", allowed)
        for role in ("chat", "vision", "browser_tool", "listening", "planner"):
            self.assertNotIn("gtts", model_registry.get_allowed_providers_for_role(role),
                             "%s must not accept the voice engine" % role)

    def test_voices_list_without_any_key(self):
        # No credential exists for this engine — listing must still work, or
        # the fallback would be invisible exactly when it is needed.
        models = model_registry.list_provider_models("gtts")
        ids = [m["id"] for m in models]
        self.assertIn("en", ids)
        self.assertIn("hi", ids)
        self.assertTrue(all(m["display"] for m in models))

    def test_resolves_with_no_credentials(self):
        snapshot = model_registry._snapshot_from_settings(
            "tts", {}, override=("gtts", "en"))
        self.assertEqual(snapshot["provider"], "gtts")
        self.assertEqual(snapshot["model"], "en")
        self.assertFalse(snapshot["endpoint"]["has_credentials"],
                         "a key-less engine reports has_credentials=False")
        self.assertIn("audio_output", snapshot["capabilities"])

    def test_declares_the_audio_output_capability(self):
        providers = {p["id"]: p for p in model_registry.list_providers()}
        self.assertIn("gtts", providers)
        self.assertEqual(providers["gtts"]["capabilities"], ["audio_output"])
        self.assertFalse(providers["gtts"]["has_key"])

    def test_voice_list_has_one_source_of_truth(self):
        listed = [m["id"] for m in model_registry.list_provider_models("gtts")]
        self.assertEqual(listed, list(gtts.LANGUAGES))


class GoogleTtsSplittingTests(unittest.TestCase):
    """The endpoint 400s over ~200 chars, so the splitter is load-bearing."""

    def test_short_text_is_one_piece(self):
        self.assertEqual(gtts.split_for_google("Hello, sir."), ["Hello, sir."])

    def test_long_text_is_split_within_the_limit(self):
        text = ("This sentence is deliberately long so the splitter has to "
                "break it, and it keeps going well past the two hundred "
                "character limit the endpoint enforces for a single request. "
                "More words follow to be certain there is a second piece.")
        parts = gtts.split_for_google(text)
        self.assertGreater(len(parts), 1)
        for piece in parts:
            self.assertLessEqual(len(piece), gtts.MAX_CHARS)

    def test_split_preserves_the_text(self):
        text = ("One two three four five six seven eight nine ten eleven "
                "twelve thirteen fourteen fifteen sixteen seventeen eighteen "
                "nineteen twenty twentyone twentytwo twentythree twentyfour "
                "twentyfive twentysix twentyseven twentyeight twentynine "
                "thirty thirtyone thirtytwo thirtythree thirtyfour.")
        parts = gtts.split_for_google(text)
        self.assertEqual(" ".join(parts), " ".join(text.split()))

    def test_blank_text_yields_nothing(self):
        self.assertEqual(gtts.split_for_google("   \n  "), [])


class GoogleTtsEngineTests(unittest.TestCase):
    """Failure must be a False, never an exception and never silence."""

    def test_returns_false_when_the_fetch_fails(self):
        with patch.object(gtts, "_resolve_language", return_value="en"), \
             patch.object(gtts, "_fetch_mp3", return_value=None):
            self.assertFalse(gtts.speak_google_tts("Hello there."))

    def test_returns_false_when_the_request_raises(self):
        with patch.object(gtts, "_resolve_language", return_value="en"), \
             patch.object(gtts._session, "get",
                          side_effect=RuntimeError("network down")):
            self.assertFalse(gtts.speak_google_tts("Hello there."))

    def test_returns_false_for_empty_text(self):
        self.assertFalse(gtts.speak_google_tts("   "))

    def test_plays_through_the_shared_playback_owner(self):
        # A short burst of real-looking s16 PCM.
        pcm = b"\x00\x10" * 2048
        handle = type("H", (), {"stopped": False})()

        # Asserted at the seam rather than by driving the module-level actor:
        # the singleton is global, so a producer thread left running by an
        # earlier test can still feed it and would corrupt a byte-level
        # assertion. What matters here is the F32 invariant — Google's audio
        # goes through the SAME playback owner Fish uses, never a stream of
        # its own — and that is exactly what these three calls express.
        with patch.object(gtts, "_resolve_language", return_value="en"), \
             patch.object(gtts, "synthesise_pcm", return_value=pcm), \
             patch.object(gtts._fish, "_register_sounddevice_playback",
                          return_value=handle) as registered, \
             patch.object(gtts._fish, "_clear_sounddevice_playback") as cleared, \
             patch.object(gtts._fish, "_play_pcm_through_actor",
                          return_value=True) as player:
            ok = gtts.speak_google_tts("Hello there.")

        self.assertTrue(ok)
        registered.assert_called_once()
        cleared.assert_called_once_with(handle)
        player.assert_called_once()

        args, kwargs = player.call_args
        self.assertEqual(args[0], pcm,
                         "the decoded PCM must reach the owner intact")
        self.assertEqual(args[1], handle)
        self.assertEqual(kwargs.get("rate"), 44100)
        self.assertEqual(kwargs.get("channels"), 1)
        self.assertEqual(kwargs.get("ck"), gtts._cache_key("Hello there.", "en"),
                         "the stream must carry the engine-scoped identity key")

    def test_audio_cache_is_keyed_by_engine_and_language(self):
        # Same sentence, different language -> different cache identity, so a
        # language switch can never replay the previous voice.
        self.assertNotEqual(gtts._cache_key("hello", "en"),
                            gtts._cache_key("hello", "hi"))
        # ...and the identity names the engine, so a sentence spoken by Fish
        # can never be served from the Google slot.
        self.assertIn("google-tts", gtts._cache_key("hello", "en"))
        self.assertNotIn("google-tts",
                         audio_actor.audio_cache_key("fish", "ref", "pcm", "hello"))

    def test_stop_never_raises(self):
        gtts.stop_google_tts()


class GoogleTtsTempoTests(unittest.TestCase):
    """The free voice is time-stretched so it does not read as slow motion.

    The stretch must be pitch-preserving: a resample would have been simpler
    but would move a 440 Hz tone to 660 Hz, i.e. a chipmunk, which is not what
    "speak 1.5x faster" means.
    """

    @staticmethod
    def _sine(seconds=1.0, freq=440.0, rate=44100):
        import numpy as np

        n = int(rate * seconds)
        t = np.arange(n, dtype=np.float64) / rate
        return (6000 * np.sin(2 * math.pi * freq * t)).astype(np.int16).tobytes()

    @staticmethod
    def _dominant_hz(pcm, rate=44100):
        import numpy as np

        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
        spectrum = np.abs(np.fft.rfft(arr * np.hanning(arr.size)))
        freqs = np.fft.rfftfreq(arr.size, 1.0 / rate)
        return float(freqs[int(np.argmax(spectrum))])

    @staticmethod
    def _segment(pcm):
        from pydub import AudioSegment

        return AudioSegment(data=pcm, sample_width=2, frame_rate=44100,
                            channels=1)

    def test_the_default_speed_is_one_and_a_half(self):
        saved = os.environ.pop("JARVIS_GOOGLE_TTS_SPEED", None)
        try:
            self.assertEqual(gtts.DEFAULT_SPEED, 1.5)
            self.assertEqual(gtts._resolve_speed(), 1.5)
        finally:
            if saved is not None:
                os.environ["JARVIS_GOOGLE_TTS_SPEED"] = saved

    def test_a_bad_speed_value_falls_back_and_a_valid_one_is_clamped(self):
        for raw in ("", "   ", "abc", "0", "-3", "nan", "inf"):
            with patch.dict(os.environ, {"JARVIS_GOOGLE_TTS_SPEED": raw}):
                self.assertEqual(gtts._resolve_speed(), gtts.DEFAULT_SPEED,
                                 "%r must fall back to the default" % raw)
        for raw, want in (("99", 2.0), ("0.1", 0.5), ("1.2", 1.2)):
            with patch.dict(os.environ, {"JARVIS_GOOGLE_TTS_SPEED": raw}):
                self.assertAlmostEqual(gtts._resolve_speed(), want, places=4)

    def test_the_stretch_cuts_the_duration_by_the_factor(self):
        pcm = self._sine(seconds=2.0)
        out = gtts._speed_up(pcm, 1.5)
        self.assertAlmostEqual(len(pcm) / len(out), 1.5, delta=0.05,
                               msg="1.5x must shorten the audio by a third")

    def test_the_stretch_preserves_pitch(self):
        # The reason atempo is used instead of a resample: a resample would
        # land the tone on 660 Hz here.
        pcm = self._sine(seconds=1.5, freq=440.0)
        self.assertAlmostEqual(self._dominant_hz(pcm), 440.0, delta=20.0)
        stretched = gtts._speed_up(pcm, 1.5)
        self.assertAlmostEqual(self._dominant_hz(stretched), 440.0, delta=25.0,
                               msg="pitch must not ride up with the tempo")

    def test_speed_one_is_a_no_op(self):
        pcm = self._sine(seconds=0.2)
        self.assertIs(gtts._speed_up(pcm, 1.0), pcm,
                      "1.0 must not pay for a filter pass")

    def test_a_failed_stretch_returns_the_original_not_silence(self):
        pcm = self._sine(seconds=0.2)
        with patch.dict(sys.modules, {"av": None}):
            self.assertEqual(gtts._speed_up(pcm, 1.5), pcm)

    def test_decode_applies_the_stretch(self):
        # _decode_pcm must hand the owner the FASTER audio, not merely decode.
        pcm = self._sine(seconds=2.0)
        with patch.object(gtts._fish, "_decode_audio",
                          return_value=self._segment(pcm)):
            fast = gtts._decode_pcm(b"fake-mp3")
            plain = gtts._decode_pcm(b"fake-mp3", speed=1.0)
        self.assertAlmostEqual(len(pcm) / len(fast), 1.5, delta=0.05)
        self.assertEqual(plain, pcm, "speed=1.0 must be byte-identical")

    def test_the_cache_key_tracks_the_speed(self):
        self.assertNotEqual(gtts._cache_key("hello", "en", 1.5),
                            gtts._cache_key("hello", "en", 1.2),
                            "a tempo change must not replay the cached take")
        self.assertEqual(gtts._cache_key("hello", "en", 1.5),
                         gtts._cache_key("hello", "en", 1.5))
        self.assertIn("google-tts", gtts._cache_key("hello", "en", 1.5))


class TtsLadderTests(unittest.TestCase):
    """The selected engine goes first; the other is the safety net."""

    def _run_chunk(self, provider, google_ok, fish_ok):
        calls = []

        def fake_google(text, before_playback=None):
            calls.append("google")
            return google_ok

        def fake_fish(text, before_playback=None):
            calls.append("fish")
            return fish_ok

        with patch.object(voice, "_resolve_tts_provider", return_value=provider), \
             patch.object(voice, "speak_google_tts", side_effect=fake_google), \
             patch.object(voice, "speak_fish_audio", side_effect=fake_fish), \
             patch.object(voice, "_is_current_generation", return_value=True), \
             patch.object(voice, "_speak_local", return_value=True):
            result = voice._speak_chunk("Systems nominal, sir.", 1)
        return calls, result

    def test_google_selected_goes_first_and_wins(self):
        calls, result = self._run_chunk("gtts", google_ok=True, fish_ok=True)
        self.assertEqual(calls, ["google"],
                         "the selected engine must run, and the fallback must not")
        self.assertTrue(result)

    def test_fish_selected_goes_first_and_wins(self):
        calls, result = self._run_chunk("fish", google_ok=True, fish_ok=True)
        self.assertEqual(calls, ["fish"])
        self.assertTrue(result)

    def test_dead_google_engine_falls_through_to_fish(self):
        calls, result = self._run_chunk("gtts", google_ok=False, fish_ok=True)
        self.assertEqual(calls, ["google", "fish"],
                         "a dead free engine must degrade, not go silent")
        self.assertTrue(result)

    def test_dead_fish_engine_falls_through_to_google(self):
        calls, result = self._run_chunk("fish", google_ok=True, fish_ok=False)
        self.assertEqual(calls, ["fish", "google"])
        self.assertTrue(result)

    def test_prefetch_targets_the_selected_engine(self):
        with patch.object(voice, "_resolve_tts_provider", return_value="gtts"), \
             patch.object(voice, "prefetch_google_tts") as g, \
             patch.object(voice, "prefetch_fish_audio") as f:
            voice._prefetch_tts_audio("next sentence")
        self.assertTrue(g.called, "the free engine must be warmed when selected")
        self.assertFalse(f.called, "warming the wrong engine wastes a synthesis")

        with patch.object(voice, "_resolve_tts_provider", return_value="fish"), \
             patch.object(voice, "prefetch_google_tts") as g2, \
             patch.object(voice, "prefetch_fish_audio") as f2:
            voice._prefetch_tts_audio("next sentence")
        self.assertTrue(f2.called)
        self.assertFalse(g2.called)


if __name__ == "__main__":
    unittest.main()
