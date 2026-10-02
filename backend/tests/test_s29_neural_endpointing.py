"""S29 - neural-VAD-driven utterance endpointing.

Acceptance: the END of the user's utterance is decided by a small neural
VAD over the echo-cancelled frames, with hysteresis (voice must persist
~100 ms to count; quiet must persist ~200 ms to end), instead of by the
fixed energy threshold that a fan/AC outruns and a soft voice falls under.

  * the hysteresis is a pure state machine (scripted judges here);
  * the Silero judge is a thin ONNX wrapper (smoke-tested, no torch);
  * webrtcvad is the fallback judge with the same hysteresis;
  * the listener breaks its capture loop on the endpoint's "ended" event,
    and an endpoint error or absence changes nothing.
"""

import unittest
from unittest.mock import patch

import speech_recognition as sr

from backend.services import listener, neural_vad


class _ScriptedJudge:
    """Deterministic stand-in for a VAD judge: scripted (prob, seconds)."""

    name = "scripted"

    def __init__(self, script):
        self._script = list(script)
        self._index = 0

    def reset(self):
        self._index = 0

    def frames(self, pcm_bytes=None, sample_rate=16000, sample_width=2):
        while self._index < len(self._script):
            item = self._script[self._index]
            self._index += 1
            yield item


VOICE = (1.0, 0.032)
QUIET = (0.0, 0.032)


class HysteresisTests(unittest.TestCase):
    """The state machine: ~100 ms of voice starts, ~200 ms of quiet ends."""

    def _endpoint(self, script, **kw):
        return neural_vad.SpeechEndpoint(_ScriptedJudge(script), **kw)

    def test_quiet_alone_never_ends(self):
        endpoint = self._endpoint([QUIET] * 20)
        self.assertIsNone(endpoint.feed(b""))
        self.assertFalse(endpoint.has_voiced)

    def test_speech_needs_the_full_on_run(self):
        endpoint = self._endpoint([VOICE] * 3)
        self.assertIsNone(endpoint.feed(b""))
        self.assertFalse(endpoint.has_voiced)

    def test_end_fires_after_the_full_off_run(self):
        script = [VOICE] * 4 + [QUIET] * 6
        endpoint = self._endpoint(script)
        self.assertIsNone(endpoint.feed(b""))  # 0.32s total: 160ms quiet < 200ms
        endpoint.reset()
        script = [VOICE] * 4 + [QUIET] * 7  # 224 ms of quiet
        endpoint = self._endpoint(script)
        self.assertEqual(endpoint.feed(b""), "ended")
        self.assertFalse(endpoint.speaking)

    def test_a_voice_break_resets_the_quiet_run(self):
        script = [VOICE] * 4 + [QUIET] * 5 + [VOICE] + [QUIET] * 7
        endpoint = self._endpoint(script)
        self.assertEqual(endpoint.feed(b""), "ended")
        self.assertLess(endpoint._voice_run, 0.032)

    def test_end_fires_exactly_once(self):
        script = [VOICE] * 4 + [QUIET] * 7 + [QUIET] * 7
        endpoint = self._endpoint(script)
        self.assertEqual(endpoint.feed(b""), "ended")
        self.assertIsNone(endpoint.feed(b""))

    def test_no_judge_is_inert(self):
        endpoint = neural_vad.SpeechEndpoint(None)
        self.assertFalse(endpoint.available)
        self.assertIsNone(endpoint.feed(b""))

    def test_hysteresis_windows_are_env_tunable(self):
        endpoint = self._endpoint([VOICE] * 2 + [QUIET] * 2,
                                  on_seconds=0.06, off_seconds=0.06)
        self.assertEqual(endpoint.feed(b""), "ended")


class SileroJudgeTests(unittest.TestCase):
    """The bundled ONNX model runs on onnxruntime (no torch required)."""

    def setUp(self):
        if neural_vad._load_silero_session() is None:
            self.skipTest("Silero model/session not available")

    def test_the_session_loads_and_runs_without_torch(self):
        endpoint = neural_vad.make_speech_endpoint()
        self.assertEqual(endpoint.judge_name, "silero")
        silence = b"\x00\x00" * 1024
        first = next(endpoint._judge.frames(silence), None)
        self.assertIsNotNone(first)
        self.assertLess(first[0], 0.5)
        self.assertAlmostEqual(first[1], 0.032)

    def test_silence_stays_silence_across_many_frames(self):
        endpoint = neural_vad.make_speech_endpoint()
        silence = b"\x00\x00" * (1024 * 4)
        probs = [p for p, _ in endpoint._judge.frames(silence)]
        self.assertGreaterEqual(len(probs), 8)
        self.assertTrue(all(p < 0.5 for p in probs))


class WebRtcFallbackTests(unittest.TestCase):
    """Without Silero the same hysteresis runs on webrtcvad verdicts."""

    def test_the_fallback_judge_is_webrtcvad(self):
        with patch.object(neural_vad, "_load_silero_session",
                          return_value=None):
            endpoint = neural_vad.make_speech_endpoint()
        self.assertEqual(endpoint.judge_name, "webrtcvad")
        silence = b"\x00\x00" * 1024
        probs = [p for p, _ in endpoint._judge.frames(silence)]
        self.assertTrue(all(p == 0.0 for p in probs))
        self.assertIsNone(endpoint.feed(silence))


def _audio(seconds=0.5, rate=16000):
    return sr.AudioData(b"\x10\x00" * int(rate * seconds), rate, 2)


class _Stream:
    """Iterable capture stream that records how many chunks were consumed."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.consumed = 0

    def __iter__(self):
        for chunk in self.chunks:
            self.consumed += 1
            yield chunk


class _FakeEndpoint:
    def __init__(self, results):
        self._results = list(results)
        self.feeds = 0

    def feed(self, *args, **kwargs):
        self.feeds += 1
        if not self._results:
            return None
        return self._results.pop(0)


class _CaptureHarness(unittest.TestCase):
    def setUp(self):
        listener._turn_stabilizer.reset()
        self.addCleanup(listener._turn_stabilizer.reset)

    def _run_capture(self, chunks=6, endpoint=None, onset=True):
        stream = _Stream([_audio(0.2) for _ in range(chunks)])
        patches = [
            patch.object(listener, "_get_microphone_source",
                         return_value=object()),
            patch.object(listener.recognizer, "listen", return_value=stream),
            patch.object(listener, "_aec_filter_chunk",
                         side_effect=lambda chunk, fid, t: (chunk, False,
                                                            False)),
            patch.object(listener, "_should_confirm_speech_start",
                         return_value=onset),
            patch.object(listener, "barge_in_on_speech_onset"),
            patch.object(listener, "play_capture_complete_earcon"),
            patch.object(listener, "_recalibrate_listener"),
            patch.object(listener, "is_human_voice", return_value=True),
            patch.object(listener, "make_speech_endpoint",
                         return_value=endpoint),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        return stream, listener._capture_audio()


class ListenerWiringTests(_CaptureHarness):
    """The capture loop breaks on trusted silence, and only that."""

    def test_the_capture_ends_on_trusted_silence(self):
        endpoint = _FakeEndpoint([None, None, None, "ended"])
        stream, audio = self._run_capture(chunks=6, endpoint=endpoint)
        self.assertEqual(stream.consumed, 4,
                         "the capture must end on the endpoint's event")
        self.assertEqual(endpoint.feeds, 4)
        self.assertIsNotNone(audio)

    def test_a_neural_end_without_speech_does_not_end_the_capture(self):
        # speech_started stays False in-loop: the endpoint event is ignored.
        endpoint = _FakeEndpoint(["ended"])
        stream, _audio_out = self._run_capture(chunks=4, endpoint=endpoint,
                                               onset=False)
        self.assertEqual(stream.consumed, 4)

    def test_no_endpoint_changes_nothing(self):
        stream, audio = self._run_capture(chunks=4, endpoint=None)
        self.assertEqual(stream.consumed, 4)
        self.assertIsNotNone(audio)

    def test_an_endpoint_error_never_breaks_the_capture(self):
        class _Broken:
            def feed(self, *args, **kwargs):
                raise RuntimeError("boom")

        stream, audio = self._run_capture(chunks=4, endpoint=_Broken())
        self.assertEqual(stream.consumed, 4)
        self.assertIsNotNone(audio)


class DisabledFeatureTests(unittest.TestCase):
    """The kill-switch restores the previous behaviour exactly."""

    def test_disabled_by_env_makes_no_endpoint(self):
        with patch.object(neural_vad, "NEURAL_VAD_ENABLED", False):
            self.assertIsNone(neural_vad.make_speech_endpoint())


if __name__ == "__main__":
    unittest.main()
