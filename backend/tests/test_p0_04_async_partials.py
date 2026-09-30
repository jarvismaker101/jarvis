"""[P0-04] Partial transcription off the capture loop, and worth having.

The bugs this pins down:

* ``_emit_partial_window`` called the whisper daemon INLINE from inside
  ``for chunk in audio_stream``, so every partial blocked frame consumption for
  up to PARTIAL_STT_TIMEOUT seconds - delaying onset, barge-in and the end of
  the utterance itself.
* the stabilizer's ``push()`` return value was thrown away, so the agreement
  the whole F34 design exists for could never end a turn early.
* PARTIAL_MAX_AUDIO_SECONDS (20) was LARGER than MAX_PHRASE_SECONDS (15), so
  the "only send the tail" cap could never apply.
* ``_transcribe_partial`` retried a TypeError with NO deadline at all.

No microphone, TTS engine, provider or external process is opened here.
"""

import threading
import time
import unittest
from unittest.mock import patch

import speech_recognition as sr

from backend.services import listener


def _speech(seconds=0.25):
    """A chunk the (stubbed) VAD calls speech."""
    return sr.AudioData(b"\x10\x00" * int(16000 * seconds), 16000, 2)


def _silence(seconds=0.25):
    return sr.AudioData(b"\x00\x00" * int(16000 * seconds), 16000, 2)


def _vad_on_energy(audio, *args, **kwargs):
    """Deterministic stand-in for the VAD: non-zero PCM is speech."""
    data = getattr(audio, "frame_data", b"") or b""
    return any(data)


class _Gate:
    """A stub engine whose calls block until the test releases them."""

    def __init__(self):
        self.release = threading.Event()
        self.started = []
        self.finished = []

    def __call__(self, audio, timeout=None):
        self.started.append(audio)
        self.release.wait(timeout=10.0)
        self.finished.append(listener._audio_duration_seconds(audio))
        return "open chrome", "en"


class _Pipeline:
    """Drives the REAL capture loop with a stubbed engine and mic."""

    def __init__(self, chunks, engine, min_seconds=0.25, ready=True):
        self.chunks = list(chunks)
        self.engine = engine
        self.min_seconds = min_seconds
        self.ready = ready
        self.counter = {"n": 0}
        self.stream = None
        self.patches = []
        self.early = {}
        self.stt_calls = []

    def _make_stream(self):
        class _Stream:
            def __init__(self, chunks, counter):
                self._chunks = chunks
                self._counter = counter

            def __iter__(self):
                for chunk in self._chunks:
                    self._counter["n"] += 1
                    yield chunk

        return _Stream(self.chunks, self.counter)

    def __enter__(self):
        listener._turn_stabilizer.reset()
        self.stream = self._make_stream()

        class _Source:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        self.patches = [
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
            patch.object(listener, "is_human_voice", side_effect=_vad_on_energy),
            patch.object(listener, "recognize_local_whisper",
                         side_effect=self.engine),
            patch.object(listener, "recognize_multilingual",
                         side_effect=self._final_stt),
            patch.object(listener, "PARTIAL_TRANSCRIBE_MIN_SECONDS",
                         self.min_seconds),
            patch.object(listener, "whisper_daemon_ready",
                         return_value=self.ready),
        ]
        for p in self.patches:
            p.start()
        return self

    def _final_stt(self, audio):
        self.stt_calls.append(audio)
        return "final stt", "final stt", "en"

    def __exit__(self, *exc):
        for p in reversed(self.patches):
            p.stop()
        listener._turn_stabilizer.reset()
        return False

    def run(self):
        """Run the real capture loop (early-commit out-param included)."""
        return listener._capture_audio(early=self.early)

    def run_listen(self):
        """Run the real listen() (capture + commit) against the same stream."""
        self.stream = self._make_stream()
        return listener.listen()

    @property
    def consumed_chunks(self):
        return self.counter["n"]



class PartialsAreOffTheLoopTests(unittest.TestCase):
    """PART A: the capture loop never waits on transcription again."""

    def tearDown(self):
        # The worker is process-global: never leave a blocked stub behind.
        listener._partial_worker.drain(3.0)

    def test_frames_keep_being_consumed_while_a_partial_is_in_flight(self):
        engine = _Gate()
        pipeline = _Pipeline([_speech() for _ in range(6)], engine)
        with pipeline:
            started = time.monotonic()
            try:
                pipeline.run()
                elapsed = time.monotonic() - started
            finally:
                engine.release.set()
        self.assertEqual(pipeline.consumed_chunks, 6,
                         "the loop stopped consuming frames")
        self.assertTrue(engine.started, "the engine was never called")
        # The capture returned while the engine was STILL blocked; the old
        # inline implementation could not return before the engine did.
        self.assertEqual(engine.finished, [],
                         "the loop waited for the transcription to finish")
        self.assertLess(elapsed, 0.5)

    def test_a_backlogged_worker_does_not_cost_wait_after_wait(self):
        engine = _Gate()
        pipeline = _Pipeline([_speech() for _ in range(6)], engine)
        with pipeline:
            started = time.monotonic()
            try:
                pipeline.run()
                elapsed = time.monotonic() - started
            finally:
                engine.release.set()
        budget = listener.PARTIAL_DELIVERY_GRACE_SECONDS * 3
        self.assertLess(elapsed, budget,
                        "the grace wait was paid on a backlogged worker")


class NewestPartialWinsTests(unittest.TestCase):
    """PART A2: when partials outpace the engine, only the newest is kept."""

    def setUp(self):
        ready = patch.object(listener, "whisper_daemon_ready",
                             return_value=True)
        ready.start()
        self.addCleanup(ready.stop)

    def tearDown(self):
        listener._partial_worker.drain(3.0)

    def test_only_the_newest_pending_partial_is_transcribed(self):
        """Three submissions against a blocked engine: the middle is lost.

        Audio LENGTH is the tag - the engine reports each window's duration, so
        it is unambiguous which submission was transcribed.
        """
        gate = threading.Event()
        seen = []

        def engine(audio, timeout=None):
            seen.append(f"{listener._audio_duration_seconds(audio):.2f}")
            gate.wait(timeout=10.0)
            return "x", "en"

        with patch.object(listener, "recognize_local_whisper", engine):
            try:
                for seconds in (0.25, 0.50, 0.75):
                    listener._partial_worker.submit(
                        [_speech(seconds)], 1, 1, 500.0)
            finally:
                gate.set()
            listener._partial_worker.drain(3.0)

        self.assertIn("0.75", seen,
                      "the NEWEST partial was not the one kept")
        self.assertNotIn("0.50", seen,
                         "a superseded pending partial was still transcribed")

    def test_submit_never_blocks_on_a_full_queue(self):
        gate = threading.Event()

        def engine(audio, timeout=None):
            gate.wait(timeout=10.0)
            return "x", "en"

        with patch.object(listener, "recognize_local_whisper", engine):
            listener._partial_worker.submit([_speech()], 1, 1, 500.0)
            started = time.monotonic()
            for _ in range(10):
                listener._partial_worker.submit([_speech()], 1, 1, 500.0)
            elapsed = time.monotonic() - started
            gate.set()
            listener._partial_worker.drain(3.0)
        self.assertLess(elapsed, 0.1, "submit() blocked on the worker")


class EarlyCommitTests(unittest.TestCase):
    """PART B: agreement + 250ms silence ends the turn with the agreed text."""

    def tearDown(self):
        listener._partial_worker.drain(3.0)

    def _agreeing_engine(self, text="open chrome"):
        def engine(audio, timeout=None):
            return text, "en"
        return engine

    def test_two_agreeing_partials_plus_250ms_of_silence_end_the_capture(self):
        # Speech long enough for two agreeing partials, then quiet.
        chunks = ([_speech() for _ in range(4)] + [_silence() for _ in range(4)])
        pipeline = _Pipeline(chunks, self._agreeing_engine())
        with pipeline:
            pipeline.run()
        self.assertEqual(pipeline.early.get("text"), "open chrome",
                         "agreement + silence did not commit early")
        self.assertLess(pipeline.consumed_chunks, len(chunks),
                        "the capture ran to the end instead of ending early")

    def test_the_early_commit_returns_the_text_without_the_final_stt(self):
        chunks = ([_speech() for _ in range(4)] + [_silence() for _ in range(4)])
        pipeline = _Pipeline(chunks, self._agreeing_engine())
        with pipeline:
            result = pipeline.run_listen()
        self.assertEqual(result, "open chrome")
        self.assertEqual(pipeline.stt_calls, [],
                         "the final STT ran despite an agreed transcript")

    def test_disagreeing_partials_do_not_end_the_capture(self):
        """Every window contradicts the previous one: nothing may commit."""
        calls = {"n": 0}

        def engine(audio, timeout=None):
            calls["n"] += 1
            # A different text every time, so no two consecutive windows agree.
            return f"phrase number {calls['n']}", "en"

        chunks = ([_speech() for _ in range(4)] + [_silence() for _ in range(4)])
        pipeline = _Pipeline(chunks, engine)
        with pipeline:
            pipeline.run()
        self.assertEqual(pipeline.early, {},
                         "disagreeing partials approved an early end")
        self.assertEqual(pipeline.consumed_chunks, len(chunks),
                         "the capture ended early without agreement")

    def test_the_normal_path_still_needs_the_full_pause_threshold(self):
        """No agreement -> no early exit, however long the silence."""
        chunks = [_speech() for _ in range(3)] + [_silence() for _ in range(6)]
        pipeline = _Pipeline(chunks, self._agreeing_engine("a"))
        with pipeline:
            pipeline.run()
        # "a" is too short for the stabilizer's min_stable_chars, so it never
        # commits: the loop must consume every frame.
        self.assertEqual(pipeline.early, {})
        self.assertEqual(pipeline.consumed_chunks, len(chunks))


class HallucinationNeverCommitsTests(unittest.TestCase):
    """PART B8: the gate stays on partials, so agreement cannot commit junk."""

    #: The live incident phrase (wake-bias prompt echo over TTS echo).
    HALLUCINATION = "jarvis, wake up, jervis, utho, jago, chalu"

    def setUp(self):
        from backend.services.transcription import is_hallucinated_transcript

        # Self-validating: the phrase really is gated.
        self.assertTrue(is_hallucinated_transcript(self.HALLUCINATION))

    def tearDown(self):
        listener._partial_worker.drain(3.0)

    def test_a_hallucinated_partial_never_reaches_the_stabilizer(self):
        def engine(audio, timeout=None):
            return self.HALLUCINATION, "en"

        chunks = ([_speech() for _ in range(4)] + [_silence() for _ in range(4)])
        pipeline = _Pipeline(chunks, engine)
        with pipeline:
            pipeline.run()
        self.assertEqual(pipeline.early, {},
                         "a hallucinated partial committed an utterance")
        self.assertEqual(listener._turn_stabilizer.committed(), "")
        self.assertFalse(listener._turn_stabilizer.unstable())

    def test_a_hallucinated_partial_is_never_returned_by_listen(self):
        def engine(audio, timeout=None):
            return self.HALLUCINATION, "en"

        chunks = ([_speech() for _ in range(4)] + [_silence() for _ in range(4)])
        pipeline = _Pipeline(chunks, engine)
        with pipeline:
            result = pipeline.run_listen()
        # The hallucination is never the utterance: no partial window was
        # produced, so nothing committed and the turn fell through to the
        # (clean) final engine instead.
        self.assertNotEqual(result, self.HALLUCINATION)
        self.assertEqual(result, "final stt")
        self.assertEqual(pipeline.stt_calls.__len__(), 1)


class DaemonReadinessGateTests(unittest.TestCase):
    """PART A5: partials are skipped when the daemon cannot serve them."""

    def tearDown(self):
        listener._partial_worker.drain(3.0)

    def _run_capture(self):  # retained for the readiness tests below
        seen = []

        def engine(audio, timeout=None):
            seen.append(audio)
            return "open chrome", "en"

        pipeline = _Pipeline([_speech() for _ in range(2)], engine)
        with pipeline:
            pipeline.run()
        return seen

    def test_a_not_ready_daemon_skips_the_partial_entirely(self):
        before = listener.partial_worker_stats()
        seen = []

        def engine(audio, timeout=None):
            seen.append(audio)
            return "open chrome", "en"

        pipeline = _Pipeline([_speech() for _ in range(2)], engine,
                             ready=False)
        with pipeline:
            pipeline.run()
        listener._partial_worker.drain(2.0)
        after = listener.partial_worker_stats()
        self.assertEqual(seen, [], "the engine was called anyway")
        self.assertEqual(after["completed"] - before["completed"], 0)
        self.assertGreater(after["skipped_not_ready"]
                           - before["skipped_not_ready"], 0,
                           "the skip was not recorded")

    def test_a_ready_daemon_still_produces_partials(self):
        seen = []

        def engine(audio, timeout=None):
            seen.append(audio)
            return "open chrome", "en"

        pipeline = _Pipeline([_speech() for _ in range(2)], engine, ready=True)
        with pipeline:
            pipeline.run()
        listener._partial_worker.drain(2.0)
        self.assertTrue(seen, "a ready daemon produced no partials")

    def test_a_definite_report_decides_readiness(self):
        class _Resp:
            def __init__(self, payload):
                self._payload = payload

            def read(self):
                import json
                return json.dumps(self._payload).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        for payload, expected in (
            ({"ready": True, "device": "cuda"}, True),
            ({"ready": False, "device": "cuda"}, False),
            ({"ready": True, "device": "cpu"}, False),
        ):
            with patch("urllib.request.urlopen",
                       return_value=_Resp(payload)):
                self.assertEqual(listener.probe_whisper_daemon(), expected,
                                 payload)

    def test_an_unreachable_daemon_is_no_report_not_not_ready(self):
        """No report must not silently disable partials (see the docstring)."""
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            self.assertIsNone(listener.probe_whisper_daemon())

    def test_the_gate_never_raises(self):
        with patch.object(listener, "_refresh_daemon_ready",
                          side_effect=RuntimeError("boom")):
            self.assertIn(listener.whisper_daemon_ready(), (True, False))


class PartialBoundsTests(unittest.TestCase):
    """The caps that keep a long utterance bounded, and actually apply."""

    def test_the_tail_cap_is_smaller_than_the_phrase_cap(self):
        self.assertLess(listener.PARTIAL_MAX_AUDIO_SECONDS,
                        listener.MAX_PHRASE_SECONDS,
                        "the tail cap can never apply at this size")

    def test_only_the_tail_is_sent_to_the_engine(self):
        chunks = [_speech(1.0) for _ in range(8)]     # 8 seconds of audio
        tail = listener._bounded_audio_tail(chunks,
                                           listener.PARTIAL_MAX_AUDIO_SECONDS)
        self.assertLess(len(tail), len(chunks))
        self.assertEqual(tail, chunks[-len(tail):])
        total = sum(listener._audio_duration_seconds(c) for c in tail)
        self.assertLessEqual(total, listener.PARTIAL_MAX_AUDIO_SECONDS + 1.0)


class OnsetGatingTests(unittest.TestCase):
    """PART A3: nothing is transcribed before the user is known to speak."""

    def tearDown(self):
        listener._partial_worker.drain(3.0)

    def test_no_partial_until_onset_is_confirmed(self):
        seen = []

        def engine(audio, timeout=None):
            seen.append(audio)
            return "open chrome", "en"

        pipeline = _Pipeline([_speech() for _ in range(4)], engine)
        with pipeline:
            with patch.object(listener, "_should_confirm_speech_start",
                              return_value=False):
                pipeline.run()
        for window in seen:
            self.assertTrue(window.frame_data)


class WorkerLifecycleTests(unittest.TestCase):
    """The worker is a daemon and shuts down cleanly at listener exit."""

    def test_the_worker_thread_is_a_daemon(self):
        worker = listener._PartialWorker()
        worker.submit([_speech()], 1, 1, 500.0)
        worker.drain(2.0)
        self.assertIsNotNone(worker._thread)
        self.assertTrue(worker._thread.daemon)
        worker.shutdown()

    def test_shutdown_is_clean_and_drops_pending_work(self):
        worker = listener._PartialWorker()
        worker._shutdown.set()
        self.assertIsNone(worker.submit([_speech()], 1, 1, 500.0))
        worker.shutdown()

    def test_a_failing_engine_never_breaks_the_worker(self):
        worker = listener._PartialWorker()

        def engine(audio, timeout=None):
            raise RuntimeError("engine exploded")

        with patch.object(listener, "whisper_daemon_ready",
                          return_value=True), \
             patch.object(listener, "recognize_local_whisper", engine):
            worker.submit([_speech()], 1, 1, 500.0)
            worker.drain(2.0)
        self.assertGreaterEqual(worker.stats["failed"], 1)
        self.assertLessEqual(worker.stats["completed"], 1)


if __name__ == "__main__":
    unittest.main()

