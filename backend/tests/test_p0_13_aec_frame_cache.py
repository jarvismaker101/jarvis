"""P0-13 — the AEC frame cache must never replay one capture into the next.

The defect: ``listener._capture_audio`` restarted ``frame_id`` at 0 on every
capture, while ``AecSignalPath._frame_cache`` is instance state on a
process-wide singleton and is only ever evicted by LRU. So capture N+1's frame
0 hit capture N's entry and the listener analysed the PREVIOUS utterance's PCM
(as well as its ``had_reference``/``suppressed`` flags) as if it were the new
one. Audibly: Jarvis talking over itself from the second turn onwards.

Two things must hold after the fix, and the tests pin both:

  * identity — a frame id is unique for the whole process, because ids are
    scoped by the token ``begin_capture()`` hands out;
  * eviction — ``begin_capture()`` also drops the previous capture's entries,
    bounding memory over a long session.

Clearing alone would satisfy the second and break the first: the cache exists so
the listener's sliding VAD window can re-ask for a frame it already analysed
without feeding the AEC twice. That once-only contract is pinned here too.
"""

import math
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from backend.services import echo_cancel
from backend.services import listener


def _tone(seconds, freq, rate, amplitude=9000.0):
    t = np.arange(int(rate * seconds)) / float(rate)
    return (np.sin(2 * math.pi * freq * t) * amplitude).astype(np.int16).tobytes()


class _FakeChunk:
    """Minimal stand-in for a speech_recognition AudioData chunk."""

    def __init__(self, frame_data, sample_rate=16000):
        self.frame_data = frame_data
        self.sample_rate = sample_rate
        self.sample_width = 2

    def __len__(self):
        return len(self.frame_data) // 2


class _CountingCanceller:
    """Canceller that records every call, so 'fed the AEC twice' is visible."""

    name = "counting"
    degraded = False

    def __init__(self):
        self.calls = 0
        self.seen = []

    def cancel(self, mic_pcm, ref_pcm):
        self.calls += 1
        self.seen.append(mic_pcm)
        return mic_pcm


class _CaptureHarness(unittest.TestCase):
    """Drives the real ``_capture_audio`` against a real ``AecSignalPath``."""

    def setUp(self):
        listener.listener_state.mark_user_speaking(False)
        listener.empty_listen_count = 0

    def _path(self, canceller=None):
        return echo_cancel.AecSignalPath(
            canceller=canceller or echo_cancel.NoOpEchoCanceller(),
            transport=None)

    def _capture(self, chunks):
        """One full ``_capture_audio()`` over *chunks* (AEC path left real)."""

        def fake_listen(source, timeout=None, phrase_time_limit=None,
                        stream=False):
            return iter(chunks)

        patches = [
            patch.object(listener, "_get_microphone_source",
                         return_value=MagicMock()),
            patch.object(listener.recognizer, "listen", side_effect=fake_listen),
            patch.object(listener, "barge_in_on_speech_onset"),
            patch.object(listener, "play_capture_complete_earcon"),
            patch.object(listener, "_recalibrate_listener"),
            patch.object(listener, "_reset_microphone_source"),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        return listener._capture_audio()

    def _double_talk(self, playback, user_freq, seconds=0.4):
        """Playback + the user's own voice, chunked like the mic delivers it.

        The reference is the rendered playback and the mic is that playback
        coming back through the speaker plus the user, which is the situation
        ``had_reference`` is True for — and the only one where a cached frame is
        actually used as the STT input.
        """
        mic = (np.frombuffer(playback, dtype=np.int16).astype(np.float64) * 1.5 +
               np.frombuffer(_tone(seconds, user_freq, 16000, 15000.0),
                             dtype=np.int16).astype(np.float64))
        mic = np.clip(mic, -32768, 32767).astype(np.int16).tobytes()
        return [_FakeChunk(mic[index:index + 3200])
                for index in range(0, len(mic), 3200)]


class CrossCaptureReplayTests(_CaptureHarness):
    """The regression: turn 2 must never hear turn 1."""

    def test_two_captures_with_different_audio_produce_different_output(self):
        playback = _tone(0.4, 250.0, 16000, amplitude=12000.0)
        first_chunks = self._double_talk(playback, user_freq=180.0)
        second_chunks = self._double_talk(playback, user_freq=900.0)
        self.assertNotEqual(b"".join(c.frame_data for c in first_chunks),
                            b"".join(c.frame_data for c in second_chunks))
        path = self._path()
        path.feed_reference(playback, sample_rate=16000, t_end=time.monotonic())

        # The listener reaches the AEC through the module-level singleton, so
        # redirecting it here exercises the real wiring end to end.
        with patch.object(echo_cancel, "signal_path", path):
            first = self._capture(first_chunks)
            second = self._capture(second_chunks)

        self.assertIsNotNone(first, "the first capture must yield audio")
        self.assertIsNotNone(second, "the second capture must yield audio")
        first_pcm = bytes(first.get_raw_data())
        second_pcm = bytes(second.get_raw_data())
        self.assertNotEqual(
            first_pcm, second_pcm,
            "the second capture replayed the first capture's cached frames")
        self.assertEqual(path.stats["replayed_frames"], 0,
                         "replayed_frames must count only within-capture "
                         "re-asks, never cross-capture collisions")

    def test_the_second_capture_is_actually_processed_by_the_aec(self):
        """Distinct output is not enough — the frames must be re-analysed."""
        playback = _tone(0.4, 250.0, 16000, amplitude=12000.0)
        path = self._path()
        path.feed_reference(playback, sample_rate=16000, t_end=time.monotonic())
        with patch.object(echo_cancel, "signal_path", path):
            self._capture(self._double_talk(playback, user_freq=180.0))
            after_first = path.stats["frames_processed"]
            self._capture(self._double_talk(playback, user_freq=900.0))
        self.assertEqual(path.stats["frames_processed"], after_first * 2,
                         "the second capture reused frames instead of "
                         "processing its own")
        self.assertEqual(path.stats["captures_started"], 2)

    def test_an_identical_capture_is_still_reproducible(self):
        """Two identical captures are byte-identical (no flakiness)."""
        playback = _tone(0.4, 250.0, 16000, amplitude=12000.0)
        chunks = self._double_talk(playback, user_freq=180.0)
        path = self._path()
        path.feed_reference(playback, sample_rate=16000, t_end=time.monotonic())
        with patch.object(echo_cancel, "signal_path", path):
            first = self._capture(chunks)
            second = self._capture(chunks)


class WithinCaptureOnceOnlyTests(_CaptureHarness):
    """F33's once-only contract must survive the fix."""

    def test_the_same_frame_twice_hits_the_cache_and_skips_the_aec(self):
        canceller = _CountingCanceller()
        path = self._path(canceller)
        path.feed_reference(_tone(1.0, 300.0, 16000), sample_rate=16000,
                            t_end=time.monotonic())
        token = path.begin_capture(turn_id=7)
        mic = _tone(0.1, 300.0, 16000)
        frame_id = path.frame_id(0, token)

        first = path.cancelled_capture_frame(mic, frame_id=frame_id,
                                             mic_t_end=time.monotonic())
        second = path.cancelled_capture_frame(mic, frame_id=frame_id,
                                              mic_t_end=time.monotonic())
        self.assertEqual(first.pcm, second.pcm)
        self.assertEqual(first.had_reference, second.had_reference)
        self.assertEqual(first.suppressed, second.suppressed)
        self.assertEqual(canceller.calls, 1,
                         "the sliding VAD window must not re-feed the AEC")
        self.assertEqual(path.stats["replayed_frames"], 1)
        self.assertEqual(path.stats["frames_processed"], 1)

    def test_begin_capture_empties_the_cache(self):
        canceller = _CountingCanceller()
        path = self._path(canceller)
        path.feed_reference(_tone(1.0, 300.0, 16000), sample_rate=16000,
                            t_end=time.monotonic())
        token = path.begin_capture(turn_id=1)
        for index in range(3):
            path.cancelled_capture_frame(_tone(0.1, 300.0, 16000),
                                         frame_id=path.frame_id(index, token),
                                         mic_t_end=time.monotonic())
        self.assertEqual(len(path._frame_cache), 3)

        new_token = path.begin_capture(turn_id=2)
        self.assertEqual(len(path._frame_cache), 0)
        self.assertEqual(len(path._frame_order), 0)
        self.assertIsNone(path._last_frame_id)

        # ...and the within-capture de-duplication still works afterwards.
        frame_id = path.frame_id(0, new_token)
        path.cancelled_capture_frame(_tone(0.1, 300.0, 16000),
                                     frame_id=frame_id,
                                     mic_t_end=time.monotonic())
        path.cancelled_capture_frame(_tone(0.1, 300.0, 16000),
                                     frame_id=frame_id,
                                     mic_t_end=time.monotonic())
        self.assertEqual(canceller.calls, 4,
                         "3 frames, then 1 deduplicated after the reset")
        self.assertEqual(path.stats["replayed_frames"], 1)

    def test_frame_ids_never_repeat_across_captures(self):
        """The token is unique for the process, not just per turn."""
        path = self._path()
        path.feed_reference(_tone(1.0, 300.0, 16000), sample_rate=16000,
                            t_end=time.monotonic())
        ids = []
        for turn in range(4):
            token = path.begin_capture(turn_id=turn)
            ids.extend(path.frame_id(index, token) for index in range(5))
        self.assertEqual(len(ids), len(set(ids)),
                         "frame ids must be unique across the whole process")

    def test_a_constant_turn_id_still_yields_unique_ids(self):
        """Defence in depth: uniqueness must not depend on begin_turn()."""
        path = self._path()
        path.feed_reference(_tone(1.0, 300.0, 16000), sample_rate=16000,
                            t_end=time.monotonic())
        first = path.begin_capture(turn_id=99)
        second = path.begin_capture(turn_id=99)
        self.assertNotEqual(first, second)
        self.assertNotEqual(path.frame_id(0, first), path.frame_id(0, second))

    def test_the_legacy_signature_still_accepts_a_plain_id(self):
        """Callers outside the listener (and older tests) pass bare ids."""
        canceller = _CountingCanceller()
        path = self._path(canceller)
        path.feed_reference(_tone(1.0, 300.0, 16000), sample_rate=16000,
                            t_end=time.monotonic())
        mic = _tone(0.1, 300.0, 16000)
        first = path.cancelled_mic_window(mic, frame_id=11,
                                          mic_t_end=time.monotonic())
        second = path.cancelled_mic_window(mic, frame_id=11,
                                           mic_t_end=time.monotonic())
        self.assertEqual(first, second)
        self.assertEqual(canceller.calls, 1)


class StatsSemanticsTests(unittest.TestCase):
    def test_the_cache_stays_bounded_within_a_capture(self):
        canceller = _CountingCanceller()
        path = echo_cancel.AecSignalPath(canceller=canceller, transport=None)
        path.feed_reference(_tone(2.0, 300.0, 16000), sample_rate=16000,
                            t_end=time.monotonic())
        token = path.begin_capture(turn_id=1)
        total = path.FRAME_CACHE_LIMIT + 50
        for index in range(total):
            path.cancelled_mic_window(_tone(0.02, 300.0, 16000),
                                      frame_id=path.frame_id(index, token),
                                      mic_t_end=time.monotonic())
        self.assertLessEqual(len(path._frame_cache), path.FRAME_CACHE_LIMIT)
        self.assertEqual(path.stats["frames_processed"], total)
        self.assertEqual(path.stats["replayed_frames"], 0)


if __name__ == "__main__":
    unittest.main()
