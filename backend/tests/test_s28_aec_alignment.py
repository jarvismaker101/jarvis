"""S28 - "Jarvis's own voice vs. yours" (AEC alignment).

Acceptance: the degraded echo-gate path must recognise Jarvis's own
playback reliably, not only when the timing happens to line up.

  * The reference timestamp is recorded when the actor's blocking write()
    returns; the sound actually leaves the speaker later (output buffer +
    room). A zero-lag, single-frame comparison cannot survive that, so the
    path now (a) estimates the lag once per session by cross-correlation
    and applies it to every later reference fetch, (b) fits the gate over
    the last few frames jointly (~100-200 ms), and (c) fetches the remote
    reference from a background daemon so the capture loop stops blocking.
"""

import json
import os
import time
import unittest
from unittest.mock import patch

import numpy as np

from backend.services import echo_cancel, local_auth


def _pcm(samples):
    return np.asarray(samples, dtype="<i2").tobytes()


def _arr(n, lo=-2000, hi=2000, seed=7):
    rng = np.random.default_rng(seed)
    return rng.integers(lo, hi, size=n, dtype=np.int16)


def _noise_pcm(n, lo=-2000, hi=2000, seed=7):
    return _pcm(_arr(n, lo=lo, hi=hi, seed=seed))


RATE = echo_cancel.AEC_SAMPLE_RATE
WIDTH = echo_cancel.AEC_SAMPLE_WIDTH
FRAME_SAMPLES = 512  # ~32 ms
FRAME_BYTES = FRAME_SAMPLES * WIDTH


class DelayEstimatorTests(unittest.TestCase):
    """Acceptance: the lag is measured, not assumed."""

    def test_a_planted_lag_is_recovered(self):
        est = echo_cancel.DelayEstimator()
        ref = _arr(FRAME_SAMPLES + 1920)  # window + 120 ms of search
        lag = 1920
        mic = ref[ref.size - FRAME_SAMPLES - lag:ref.size - lag]
        best = est.estimate(mic.tobytes(), ref.tobytes())
        self.assertIsNotNone(best)
        self.assertAlmostEqual(best, lag / RATE, places=3)

    def test_uncorrelated_noise_casts_no_vote(self):
        est = echo_cancel.DelayEstimator()
        ref = _arr(FRAME_SAMPLES + 1920, seed=1)
        mic = _arr(FRAME_SAMPLES, seed=2)
        self.assertIsNone(est.estimate(mic.tobytes(), ref.tobytes()))
        self.assertEqual(est.votes, [])

    def test_lock_takes_consistent_votes(self):
        est = echo_cancel.DelayEstimator()
        ref = _arr(FRAME_SAMPLES + 1920)
        lag = 1920
        mic = ref[ref.size - FRAME_SAMPLES - lag:ref.size - lag]
        for _ in range(echo_cancel.AEC_LAG_CONFIRMATIONS - 1):
            self.assertFalse(est.locked)
            est.estimate(mic.tobytes(), ref.tobytes())
        est.estimate(mic.tobytes(), ref.tobytes())
        self.assertTrue(est.locked)
        self.assertAlmostEqual(est.locked_lag_seconds, lag / RATE, places=3)

    def test_a_locked_estimator_stops_estimating(self):
        est = echo_cancel.DelayEstimator()
        ref = _arr(FRAME_SAMPLES + 1920)
        mic = ref[ref.size - FRAME_SAMPLES - 1920:ref.size - 1920]
        for _ in range(echo_cancel.AEC_LAG_CONFIRMATIONS):
            est.estimate(mic.tobytes(), ref.tobytes())
        self.assertTrue(est.locked)
        self.assertIsNone(est.estimate(mic.tobytes(), ref.tobytes()))

    def test_disagreeing_votes_do_not_lock(self):
        est = echo_cancel.DelayEstimator()
        ref = _arr(FRAME_SAMPLES + 1920)
        first = ref[ref.size - FRAME_SAMPLES - 192:ref.size - 192]
        second = ref[ref.size - FRAME_SAMPLES - 1728:ref.size - 1728]
        for i in range(10):
            est.estimate(first.tobytes() if i % 2 == 0
                         else second.tobytes(), ref.tobytes())
        self.assertFalse(est.locked)


class ReferenceEchoGateJointWindowTests(unittest.TestCase):
    """Acceptance: the compare window is ~100-200 ms, not one frame."""

    def test_a_pure_echo_stretch_is_suppressed_jointly(self):
        gate = echo_cancel.ReferenceEchoGate()
        ref = _arr(FRAME_SAMPLES, seed=10)
        mic = ref.copy()  # exact playback copy
        history = _arr(FRAME_SAMPLES, seed=11)  # history was also pure echo
        suppressed, residual, metrics = gate.analyse(
            mic.tobytes(), ref.tobytes(),
            history_mic=history.tobytes(),
            history_ref=history.tobytes())
        self.assertTrue(suppressed)
        self.assertEqual(len(residual), FRAME_BYTES)

    def test_the_residual_covers_only_the_current_frame(self):
        gate = echo_cancel.ReferenceEchoGate()
        ref = _arr(FRAME_SAMPLES, seed=21)
        mic = ref.copy()
        history = _arr(FRAME_SAMPLES, seed=22)
        suppressed, residual, _ = gate.analyse(
            mic.tobytes(), ref.tobytes(),
            history_mic=history.tobytes(),
            history_ref=history.tobytes())
        self.assertTrue(suppressed)
        self.assertEqual(len(residual), FRAME_BYTES,
                         "history must not leak into the returned residual")

    def test_double_talk_is_not_suppressed(self):
        gate = echo_cancel.ReferenceEchoGate()
        ref = _arr(FRAME_SAMPLES, seed=31)
        mic = (ref.astype(np.int64) + _arr(FRAME_SAMPLES, seed=32,
                                           lo=3000, hi=6000)).astype(np.int16)
        history = _arr(FRAME_SAMPLES, seed=33)
        suppressed, residual, _ = gate.analyse(
            mic.tobytes(), ref.tobytes(),
            history_mic=history.tobytes(),
            history_ref=history.tobytes())
        self.assertFalse(suppressed)


class LagCorrectedCancellationTests(unittest.TestCase):
    """Acceptance: end-to-end, a delayed echo is suppressed after lock."""

    LAG_SECONDS = 0.12
    REF_START = 8.0
    REF_END = 10.0

    def _reference_chunks(self):
        """Feed the reference the way the actor does: small stamped chunks."""
        total = int((self.REF_END - self.REF_START) * RATE)
        pcm = _noise_pcm(total, seed=99)
        chunks = []
        step = FRAME_SAMPLES
        for start in range(0, total, step):
            end = min(start + step, total)
            t_end = self.REF_START + end / RATE
            chunks.append((pcm[start * WIDTH:end * WIDTH], t_end))
        return pcm, chunks

    def _echo_window(self, ref_pcm, t_end):
        """What the mic hears during a window ending at *t_end*."""
        end_index = int((t_end - self.LAG_SECONDS - self.REF_START) * RATE)
        start_index = end_index - FRAME_SAMPLES
        return ref_pcm[start_index * WIDTH:end_index * WIDTH]

    def test_delayed_playback_is_suppressed_after_the_lag_locks(self):
        ref_pcm, chunks = self._reference_chunks()
        path = echo_cancel.AecSignalPath(
            canceller=echo_cancel.NoOpEchoCanceller(),
            reference_buffer=echo_cancel.ReferencePcmBuffer(),
            transport=None)
        for pcm_bytes, t_end in chunks:
            path.feed_reference(pcm_bytes, sample_rate=RATE,
                                sample_width=WIDTH, t_end=t_end,
                                channels=1)
        token = path.begin_capture()
        suppressed_frames = 0
        for index in range(8):
            t_end = 9.700 + index * (FRAME_SAMPLES / RATE)
            mic = self._echo_window(ref_pcm, t_end)
            frame = path.cancelled_capture_frame(
                mic, duration_seconds=FRAME_SAMPLES / RATE,
                mic_t_end=t_end, frame_id=path.frame_id(index, token))
            if frame.suppressed:
                suppressed_frames += 1
        # The first frames estimate; the rest are suppressed.
        self.assertGreaterEqual(suppressed_frames, 4)
        self.assertIsNotNone(path.stats["lag_locked_seconds"])
        self.assertAlmostEqual(path.stats["lag_locked_seconds"],
                               self.LAG_SECONDS, places=2)

    def test_state_reports_the_alignment(self):
        ref_pcm, chunks = self._reference_chunks()
        path = echo_cancel.AecSignalPath(
            canceller=echo_cancel.NoOpEchoCanceller(),
            reference_buffer=echo_cancel.ReferencePcmBuffer(),
            transport=None)
        for pcm_bytes, t_end in chunks:
            path.feed_reference(pcm_bytes, sample_rate=RATE,
                                sample_width=WIDTH, t_end=t_end,
                                channels=1)
        token = path.begin_capture()
        for index in range(6):
            t_end = 9.700 + index * (FRAME_SAMPLES / RATE)
            mic = self._echo_window(ref_pcm, t_end)
            path.cancelled_capture_frame(
                mic, duration_seconds=FRAME_SAMPLES / RATE,
                mic_t_end=t_end, frame_id=path.frame_id(index, token))
        lag_state = path.state()["lag"]
        self.assertTrue(lag_state["locked"])
        self.assertGreaterEqual(lag_state["votes"],
                                echo_cancel.AEC_LAG_CONFIRMATIONS)


class _Resp:
    """Minimal urlopen context manager returning a JSON payload."""

    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TransportPrefetchTests(unittest.TestCase):
    """Acceptance: the capture loop stops paying the fetch itself."""

    def test_a_live_capture_keeps_the_cache_fresh_off_thread(self):
        calls = []

        def _fake_urlopen(request, timeout=None):
            calls.append(time.monotonic())
            return _Resp({"pcm_b64": "QUJD", "age_seconds": 0.0})

        transport = echo_cancel.RemoteAecTransport(
            base_url="http://127.0.0.1:1", cache_seconds=0.1)
        try:
            with patch.dict(os.environ, {local_auth._ENV_VAR: ""}), \
                    patch.object(echo_cancel, "urlopen", _fake_urlopen):
                transport.note_capture_activity(0.032)
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    if transport.stats["prefetch_fetches"] >= 1:
                        break
                    time.sleep(0.02)
                self.assertGreaterEqual(transport.stats["prefetch_fetches"],
                                        1,
                                        "the daemon never fetched")
                before = len(calls)
                pcm, _age = transport.fetch_reference(0.032)
                self.assertEqual(pcm, b"ABC")
                self.assertEqual(len(calls), before,
                                 "the capture loop dialed the network "
                                 "although the daemon had a fresh span")
        finally:
            transport._prefetch_until = 0.0

    def test_no_activity_starts_no_thread(self):
        transport = echo_cancel.RemoteAecTransport(
            base_url="http://127.0.0.1:1", cache_seconds=0.1)
        self.assertIsNone(transport._prefetch_thread)
        self.assertEqual(transport.stats["prefetch_polls"], 0)


if __name__ == "__main__":
    unittest.main()
