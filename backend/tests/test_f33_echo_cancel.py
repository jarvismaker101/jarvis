"""F33 — echo-cancelled full-duplex listening.

Acceptance (audit report): "Assistant-only playback never becomes committed
user speech, including final fallback; test arbitrary chunk boundaries,
buffers, generations, and actual double-talk recordings."

The baseline defects pinned here:
  * 44.1 kHz playback was "converted" with an integer stride
    (``int(round(16000/44100)) == 0`` -> 1), so the AEC was fed 44.1 k samples
    as if they were 16 k: an effective ~14.7 kHz reference;
  * the reference was selected by "newest suffix", which is not timestamp
    alignment;
  * the listener's sliding VAD window re-fed overlapping audio to the AEC;
  * the final VAD / human-voice gate / STT input used UNFILTERED audio, so the
    final fallback could still commit the assistant's own voice;
  * the reference singleton is process-local while the API voices replies and
    the voice process owns the microphone, and nothing joined them.
"""

import base64
import json
import math
import random
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from backend.services import echo_cancel
from backend.services import listener


def _tone(seconds, freq, rate, amplitude=9000.0, channels=1):
    t = np.arange(int(rate * seconds)) / float(rate)
    mono = (np.sin(2 * math.pi * freq * t) * amplitude).astype(np.int16)
    if channels > 1:
        return np.repeat(mono[:, None], channels, axis=1).reshape(-1).tobytes()
    return mono.tobytes()


class ChunkBoundaryTests(unittest.TestCase):
    """Arbitrary chunk boundaries must not change the converted signal."""

    def test_one_second_of_device_audio_becomes_one_second_of_reference(self):
        pcm = _tone(1.0, 440.0, 44100)
        resampler = echo_cancel.StatefulResampler(44100, 16000)
        out = np.frombuffer(resampler.resample(pcm), dtype=np.int16)
        # The old stride conversion produced 44100 "16k" samples (14.7 kHz).
        self.assertAlmostEqual(out.size / 16000.0, 1.0, delta=0.01)
        # A 440 Hz tone has 2 sign changes per period (use signbit so exact
        # zero samples do not double-count).
        sign = np.signbit(out)
        crossings = int(np.count_nonzero(np.diff(sign)))
        self.assertAlmostEqual(crossings / 2.0, 440.0, delta=10.0)

    def test_chunked_input_matches_whole_input_exactly(self):
        pcm = _tone(0.7, 300.0, 44100)
        whole = echo_cancel.StatefulResampler(44100, 16000)
        reference = whole.resample(pcm)

        rng = random.Random(7)
        chunked = echo_cancel.StatefulResampler(44100, 16000)
        pieces = []
        index = 0
        while index < len(pcm):
            size = rng.randint(1, 3000)
            pieces.append(chunked.resample(pcm[index:index + size]))
            index += size
        joined = b"".join(pieces)
        self.assertEqual(joined, reference,
                         "resampling must be continuous across chunk borders")
        # Every produced frame is a complete s16 sample.
        self.assertEqual(len(joined) % 2, 0)

    def test_stereo_device_output_becomes_mono_reference(self):
        pcm = _tone(0.5, 500.0, 48000, channels=2)
        resampler = echo_cancel.StatefulResampler(48000, 16000, channels=2)
        out = resampler.resample(pcm)
        self.assertAlmostEqual(len(out) / 2 / 16000.0, 0.5, delta=0.02)

    def test_the_buffer_resamples_across_feeds(self):
        pcm = _tone(0.4, 220.0, 44100)
        buffer = echo_cancel.ReferencePcmBuffer(max_seconds=5)
        for start in range(0, len(pcm), 999):
            buffer.feed(pcm[start:start + 999], sample_rate=44100,
                        sample_width=2)
        joined = buffer.aligned_reference(10 ** 6)
        self.assertAlmostEqual(len(joined) / 2 / 16000.0, 0.4, delta=0.02)


class ReferenceAlignmentTests(unittest.TestCase):
    """Buffers and generations: alignment is by TIME, not "newest suffix"."""

    def setUp(self):
        self.buffer = echo_cancel.ReferencePcmBuffer(max_seconds=10)

    def test_a_window_is_cancelled_against_the_span_that_overlapped_it(self):
        # 0.5 s of playback, then the mic window ending NOW.
        self.buffer.feed(_tone(0.5, 300.0, 16000), sample_rate=16000)
        now = time.monotonic()
        span = self.buffer.aligned_reference(8000, mic_t_end=now)  # 0.25 s
        self.assertEqual(len(span), 8000)

    def test_playback_that_did_not_overlap_reports_no_reference(self):
        self.buffer.feed(_tone(0.5, 300.0, 16000), sample_rate=16000,
                         t_end=time.monotonic() - 30.0)
        self.assertEqual(self.buffer.aligned_reference(8000), b"",
                         "a 30 s old reference is not this window's echo")

    def test_a_generation_change_does_not_reuse_the_old_utterance(self):
        old = _tone(0.4, 400.0, 16000)
        self.buffer.feed(old, sample_rate=16000,
                         t_end=time.monotonic() - 5.0)
        new = _tone(0.4, 900.0, 16000)
        self.buffer.feed(new, sample_rate=16000, t_end=time.monotonic())
        span = self.buffer.aligned_reference(4000)
        self.assertEqual(span[-2000:], new[-2000:],
                         "the aligned window must be the CURRENT playback")
        self.assertNotIn(old[-2000:], span)

    def test_the_ring_never_grows_without_bound(self):
        buffer = echo_cancel.ReferencePcmBuffer(max_seconds=1)
        for _ in range(50):
            buffer.feed(_tone(0.1, 200.0, 16000), sample_rate=16000)
        self.assertLessEqual(len(buffer), 16000 * 2)


class _RecordingCanceller(echo_cancel.EchoCanceller):
    """A non-degraded canceller stand-in, so ``cancel`` is actually called."""

    name = "recording"
    degraded = False

    def __init__(self):
        self.calls = 0

    def cancel(self, mic_pcm, ref_pcm):
        self.calls += 1
        return mic_pcm


class OnceOnlyProcessingTests(unittest.TestCase):
    def test_the_same_capture_frame_is_never_cancelled_twice(self):
        canceller = _RecordingCanceller()
        path = echo_cancel.AecSignalPath(canceller=canceller, transport=None)
        path.feed_reference(_tone(1.0, 300.0, 16000), sample_rate=16000)
        mic = _tone(0.1, 300.0, 16000)

        first = path.cancelled_mic_window(mic, frame_id=11, mic_t_end=time.monotonic())
        second = path.cancelled_mic_window(mic, frame_id=11, mic_t_end=time.monotonic())
        self.assertEqual(first, second)
        self.assertEqual(canceller.calls, 1,
                         "the sliding VAD window must not re-feed the AEC")
        self.assertEqual(path.stats["replayed_frames"], 1)
        self.assertEqual(path.stats["frames_processed"], 1)

    def test_distinct_frames_are_each_processed(self):
        canceller = _RecordingCanceller()
        path = echo_cancel.AecSignalPath(canceller=canceller, transport=None)
        path.feed_reference(_tone(1.0, 300.0, 16000), sample_rate=16000)
        mic = _tone(0.1, 300.0, 16000)
        for frame_id in range(3):
            path.cancelled_mic_window(mic, frame_id=frame_id,
                                      mic_t_end=time.monotonic())
        self.assertEqual(canceller.calls, 3)

    def test_state_reports_the_degraded_path_explicitly(self):
        path = echo_cancel.AecSignalPath(
            canceller=echo_cancel.NoOpEchoCanceller(), transport=None)
        state = path.state()
        self.assertTrue(state["degraded"])
        self.assertEqual(state["mode"], "noop")
        self.assertIn("unfiltered", state["reason"])
        path.feed_reference(_tone(0.2, 300.0, 16000), sample_rate=16000)
        self.assertLess(path.state()["reference_age_seconds"], 1.0)


class CrossProcessTransportTests(unittest.TestCase):
    """The API renders playback; the voice process owns the microphone."""

    def _payload(self, pcm, age):
        return json.dumps({
            "pcm_b64": base64.b64encode(pcm).decode("ascii"),
            "sample_rate": 16000,
            "sample_width": 2,
            "bytes": len(pcm),
            "age_seconds": age,
        }).encode("utf-8")

    def test_the_reference_crosses_the_process_boundary(self):
        pcm = _tone(0.3, 300.0, 16000)

        class _Response:
            def __init__(self, body):
                self._body = body

            def read(self):
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(request, timeout=None):
            self.assertIn("/aec/reference", request.full_url)
            self.assertIn("seconds=0.100", request.full_url)
            return _Response(self._payload(pcm, 0.05))

        path = echo_cancel.AecSignalPath(
            canceller=echo_cancel.NoOpEchoCanceller(),
            transport=echo_cancel.RemoteAecTransport())
        with patch.object(echo_cancel, "urlopen", side_effect=fake_urlopen):
            filtered, had_reference = path.cancelled_mic_window(
                _tone(0.1, 300.0, 16000), mic_t_end=time.monotonic())
        self.assertTrue(had_reference,
                        "the API's playback is this window's echo")
        self.assertEqual(path.stats["reference_sources"]["remote"], 1)

    def test_a_stale_remote_span_is_not_used(self):
        pcm = _tone(0.3, 300.0, 16000)

        class _Response:
            def read(self):
                return json.dumps({
                    "pcm_b64": base64.b64encode(pcm).decode("ascii"),
                    "age_seconds": 9.0,
                }).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        path = echo_cancel.AecSignalPath(
            canceller=echo_cancel.NoOpEchoCanceller(),
            transport=echo_cancel.RemoteAecTransport())
        with patch.object(echo_cancel, "urlopen",
                          side_effect=lambda *a, **k: _Response()):
            _filtered, had_reference = path.cancelled_mic_window(
                _tone(0.1, 300.0, 16000), mic_t_end=time.monotonic())
        self.assertFalse(had_reference)

    def test_a_transport_failure_is_degraded_not_fatal(self):
        path = echo_cancel.AecSignalPath(
            canceller=echo_cancel.NoOpEchoCanceller(),
            transport=echo_cancel.RemoteAecTransport())
        with patch.object(echo_cancel, "urlopen",
                          side_effect=OSError("backend down")):
            filtered, had_reference = path.cancelled_mic_window(
                _tone(0.1, 300.0, 16000), mic_t_end=time.monotonic())
        self.assertFalse(had_reference)
        self.assertEqual(filtered, _tone(0.1, 300.0, 16000),
                         "no reference means unfiltered passthrough, not silence")
        self.assertGreaterEqual(path.transport.stats["errors"], 1)


class _FakeChunk:
    def __init__(self, data, rate=16000, width=2):
        self.frame_data = data
        self.sample_rate = rate
        self.sample_width = width


class EchoGateTests(unittest.TestCase):
    """Without a real AEC model, playback must still be recognised."""

    def setUp(self):
        self.gate = echo_cancel.ReferenceEchoGate()

    def test_assistant_only_playback_is_recognised_as_echo(self):
        ref = _tone(0.2, 250.0, 16000, amplitude=12000.0)
        # The mic hears the playback through the speaker (louder + scaled).
        mic = np.frombuffer(ref, dtype=np.int16).astype(np.float64) * 1.6
        is_echo, residual, metrics = self.gate.analyse(
            np.clip(mic, -32768, 32767).astype(np.int16).tobytes(), ref)
        self.assertTrue(is_echo, metrics)
        self.assertGreater(metrics["correlation"], 0.9)
        self.assertGreater(metrics["suppression_db"], 15.0)
        self.assertLess(len(residual), len(ref) + 2)

    def test_double_talk_is_not_recognised_as_echo(self):
        ref = _tone(0.2, 250.0, 16000, amplitude=8000.0)
        user = _tone(0.2, 190.0, 16000, amplitude=8000.0)
        mic = (np.frombuffer(ref, dtype=np.int16).astype(np.float64) +
               np.frombuffer(user, dtype=np.int16).astype(np.float64))
        is_echo, residual, metrics = self.gate.analyse(
            np.clip(mic, -32768, 32767).astype(np.int16).tobytes(), ref)
        self.assertFalse(is_echo, metrics)
        # The residual must be handed back untouched for a non-echo window.
        self.assertGreater(len(residual), 0)

    def test_user_speech_alone_is_not_echo(self):
        ref = _tone(0.2, 250.0, 16000, amplitude=8000.0)
        user = _tone(0.2, 300.0, 16000, amplitude=9000.0)
        is_echo, _residual, metrics = self.gate.analyse(user, ref)
        self.assertFalse(is_echo, metrics)

    def test_the_path_marks_suppressed_frames(self):
        path = echo_cancel.AecSignalPath(
            canceller=echo_cancel.NoOpEchoCanceller(), transport=None)
        ref = _tone(0.2, 250.0, 16000, amplitude=12000.0)
        path.feed_reference(ref, sample_rate=16000, t_end=time.monotonic())
        mic = (np.frombuffer(ref, dtype=np.int16).astype(np.float64) * 1.6)
        frame = path.cancelled_capture_frame(
            np.clip(mic, -32768, 32767).astype(np.int16).tobytes(),
            mic_t_end=time.monotonic())
        self.assertTrue(frame.had_reference)
        self.assertTrue(frame.suppressed)
        self.assertEqual(path.stats["echo_suppressed"], 1)
        # ...and the two-tuple API still reports the same reference state.
        pcm, had_reference = path.cancelled_mic_window(
            _tone(0.1, 250.0, 16000), mic_t_end=time.monotonic())
        self.assertTrue(had_reference)
        self.assertIsInstance(pcm, (bytes, bytearray))


class ListenerCommitTests(unittest.TestCase):
    """Assistant-only playback must never become committed user speech."""

    def setUp(self):
        listener.listener_state.mark_user_speaking(False)
        listener.empty_listen_count = 0

    def _capture(self, chunks, aec):
        """Drive _capture_audio over *chunks* with *aec* as the cancel path."""
        def fake_listen(source, timeout=None, phrase_time_limit=None,
                        stream=False):
            return iter(chunks)

        patches = [
            patch.object(listener, "_get_microphone_source",
                         return_value=MagicMock()),
            patch.object(listener.recognizer, "listen",
                         side_effect=fake_listen),
            patch.object(listener, "_aec_capture_frame", side_effect=aec),
            patch.object(listener, "barge_in_on_speech_onset"),
            patch.object(listener, "play_capture_complete_earcon"),
            patch.object(listener, "_recalibrate_listener"),
            patch.object(listener, "_reset_microphone_source"),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        return listener._capture_audio()

    def _real_path(self, playback):
        """A signal path fed with *playback*, using the degraded no-op AEC.

        The reference is the rendered playback (what the API's TTS wrote to
        the device); the mic is that playback coming back through the
        speaker.
        """
        path = echo_cancel.AecSignalPath(
            canceller=echo_cancel.NoOpEchoCanceller(), transport=None)
        path.feed_reference(playback, sample_rate=16000,
                            t_end=time.monotonic())
        return path

    def test_assistant_only_playback_is_not_committed_as_speech(self):
        playback = _tone(0.4, 250.0, 16000, amplitude=12000.0)
        mic = (np.frombuffer(playback, dtype=np.int16).astype(np.float64) * 1.5)
        mic = np.clip(mic, -32768, 32767).astype(np.int16).tobytes()
        chunks = [_FakeChunk(mic[index:index + 3200])
                  for index in range(0, len(mic), 3200)]
        raw_audio = listener.sr.AudioData(mic, 16000, 2)
        self.assertTrue(listener.is_human_voice(raw_audio),
                        "the fixture must look like speech before filtering")

        self.assertTrue(len(chunks) >= 2)

        def aec(pcm, **kwargs):
            return echo_cancel.CaptureFrame(
                b"\x00" * len(pcm), True, True, 16000)

        result = self._capture(chunks, aec)
        self.assertIsNone(result,
                          "assistant-only playback became user speech")
        listener.barge_in_on_speech_onset.assert_not_called()

    def test_the_real_path_suppresses_assistant_only_playback(self):
        """End-to-end through the actual gate (the double-talk stand-in for
        the recordings the audit asks for)."""
        playback = _tone(0.4, 250.0, 16000, amplitude=12000.0)
        mic = np.clip(np.frombuffer(playback, dtype=np.int16).astype(np.float64)
                      * 1.5, -32768, 32767).astype(np.int16).tobytes()
        chunks = [_FakeChunk(mic[index:index + 3200])
                  for index in range(0, len(mic), 3200)]
        path = self._real_path(playback)

        def aec(pcm, **kwargs):
            return path.cancelled_capture_frame(pcm, **kwargs)

        result = self._capture(chunks, aec)
        self.assertIsNone(result, "the echo gate let playback through")
        listener.barge_in_on_speech_onset.assert_not_called()

    def test_double_talk_keeps_the_user_speech_and_barges_in(self):
        """Perfect cancellation of the playback leaves the user's words."""
        playback = _tone(0.4, 250.0, 16000, amplitude=12000.0)
        user = _tone(0.4, 180.0, 16000, amplitude=15000.0)
        mic = (np.frombuffer(playback, dtype=np.int16).astype(np.float64) +
               np.frombuffer(user, dtype=np.int16).astype(np.float64))
        mic = np.clip(mic, -32768, 32767).astype(np.int16).tobytes()
        chunks = [_FakeChunk(mic[index:index + 3200])
                  for index in range(0, len(mic), 3200)]
        path = self._real_path(playback)

        def aec(pcm, **kwargs):
            return path.cancelled_capture_frame(pcm, **kwargs)

        result = self._capture(chunks, aec)
        self.assertIsNotNone(result, "real user speech must still be captured")
        listener.barge_in_on_speech_onset.assert_called()

    def test_every_frame_is_cancelled_once_with_its_own_timestamp(self):
        loud = _tone(0.3, 250.0, 16000, amplitude=15000.0)
        chunks = [_FakeChunk(loud[index:index + 3200])
                  for index in range(0, 9600, 3200)]
        seen = []

        def aec(pcm, **kwargs):
            seen.append((kwargs.get("frame_id"), kwargs.get("mic_t_end"),
                         kwargs.get("sample_rate")))
            return echo_cancel.CaptureFrame(pcm, False, False, 16000)

        self._capture(chunks, aec)
        self.assertEqual([row[0] for row in seen], [0, 1, 2])
        stamps = [row[1] for row in seen]
        self.assertTrue(all(isinstance(s, float) for s in stamps))
        self.assertEqual(stamps, sorted(stamps))
        self.assertTrue(all(row[2] == 16000 for row in seen))

    def test_final_fallback_does_not_commit_a_suppressed_window(self):
        """Onset VAD never sees the suppressed frames, and when every frame
        was our own playback the capture is dropped instead of being handed
        to the final human-voice gate."""
        loud = _tone(0.3, 250.0, 16000, amplitude=15000.0)
        chunks = [_FakeChunk(loud[index:index + 3200])
                  for index in range(0, 9600, 3200)]
        seen = []
        orig_confirm = listener._should_confirm_speech_start

        def confirm_spy(window):
            seen.append(len(window))
            return orig_confirm(window)

        def aec(pcm, **kwargs):
            return echo_cancel.CaptureFrame(b"\x00" * len(pcm), True, True,
                                            16000)

        with patch.object(listener, "_should_confirm_speech_start",
                          side_effect=confirm_spy):
            result = self._capture(chunks, aec)
        self.assertIsNone(result,
                          "assistant-only playback reached the final gate")
        self.assertEqual(set(seen), {0},
                         "suppressed frames must not enter the onset window")
        listener.barge_in_on_speech_onset.assert_not_called()

    def test_a_mixed_capture_is_still_committed(self):
        loud = _tone(0.3, 250.0, 16000, amplitude=15000.0)
        chunks = [_FakeChunk(loud[index:index + 3200])
                  for index in range(0, 9600, 3200)]

        def aec(pcm, **kwargs):
            return echo_cancel.CaptureFrame(pcm, False, False, 16000)

        result = self._capture(chunks, aec)
        self.assertIsNotNone(result)


class DegradedReportingTests(unittest.TestCase):
    def test_the_listener_reports_a_degraded_aec_once(self):
        during = {"frames": 2, "reported": False}
        with patch.object(listener, "_aec_state",
                          return_value={"degraded": True, "mode": "noop",
                                        "reason": "no AEC installed"}):
            with patch("builtins.print") as fake_print:
                listener._report_aec_degraded_once(during)
                listener._report_aec_degraded_once(during)
        self.assertTrue(during["reported"])
        printed = " ".join(str(call) for call in fake_print.call_args_list)
        self.assertIn("degraded", printed.lower())

    def test_nothing_is_reported_when_no_playback_overlapped(self):
        during = {"frames": 0, "reported": False}
        with patch("builtins.print") as fake_print:
            listener._report_aec_degraded_once(during)
        fake_print.assert_not_called()


if __name__ == "__main__":
    unittest.main()
