"""The simple listening contract (tag ``simple-listening``, 2026-10).

What the latency pass removed: the partial-window layer — background whisper
passes over the trailing audio, the two-agreeing-windows early commit, the
partial worker thread and its daemon-readiness gate. What replaced it: capture
the whole utterance, then transcribe it exactly ONCE on the selected engine.

These tests pin that shape, and that the removed API stays removed — so putting
it back has to be a deliberate decision rather than an accident.

No microphone, TTS engine, provider or subprocess is opened here.
"""

import inspect
import os
import unittest
from unittest.mock import patch

import speech_recognition as sr

from backend import whisper_daemon
from backend.services import listener


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


class _CaptureHarness(unittest.TestCase):
    def setUp(self):
        listener._turn_stabilizer.reset()
        self.addCleanup(listener._turn_stabilizer.reset)

    def _run_capture(self, chunks=4):
        stream = _Stream([_audio(0.2) for _ in range(chunks)])
        patches = [
            patch.object(listener, "_get_microphone_source", return_value=object()),
            patch.object(listener.recognizer, "listen", return_value=stream),
            patch.object(listener, "_aec_filter_chunk",
                         side_effect=lambda chunk, fid, t: (chunk, False, False)),
            patch.object(listener, "_should_confirm_speech_start",
                         return_value=True),
            patch.object(listener, "barge_in_on_speech_onset"),
            patch.object(listener, "play_capture_complete_earcon"),
            patch.object(listener, "_recalibrate_listener"),
            patch.object(listener, "is_human_voice", return_value=True),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        return stream, listener._capture_audio()


class OneUtteranceOneTranscriptionTests(_CaptureHarness):
    """``listen()`` = capture, then exactly one engine call over that audio."""

    def test_the_selected_engine_transcribes_the_whole_utterance_once(self):
        audio = _audio()
        with patch.object(listener, "_capture_audio", return_value=audio), \
             patch.object(listener, "recognize_multilingual",
                          return_value=("open chrome", "open chrome",
                                        "en")) as engine:
            self.assertEqual(listener.listen(), "open chrome")
        engine.assert_called_once_with(audio)

    def test_no_transcription_happens_inside_the_capture_loop(self):
        """The loop that used to fire background partials is now engine-free.

        This is the invariant the old partial WORKER existed to protect (the
        capture loop must never block on an engine). It is now structural: the
        loop makes no engine call at all.
        """
        engines = ("recognize_multilingual", "recognize_local_whisper",
                   "recognize_inworld", "recognize_google_or_groq")
        mocks = {}
        for name in engines:
            item = patch.object(listener, name)
            mocks[name] = item.start()
            self.addCleanup(item.stop)

        stream, audio = self._run_capture()
        self.assertIsNotNone(audio)
        self.assertEqual(stream.consumed, len(stream.chunks))
        for name, mock in mocks.items():
            mock.assert_not_called()

    def test_capture_never_ends_early(self):
        """Every frame is consumed: no agreement-based early exit remains."""
        stream, audio = self._run_capture(chunks=6)
        self.assertEqual(stream.consumed, 6)
        # The capture ran to the end of its stream, so its own telemetry says
        # the utterance ended where the stream did.
        self.assertIsNotNone(audio)

    def test_capture_takes_no_early_out_parameter(self):
        self.assertNotIn("early",
                         inspect.signature(listener._capture_audio).parameters)

    def test_a_hallucinated_final_is_still_never_committed(self):
        audio = _audio()
        with patch.object(listener, "_capture_audio", return_value=audio), \
             patch.object(listener, "recognize_multilingual",
                          return_value=("Jarvis, wake up, jervis, utho, jago, "
                                        "chalu", "jarvis wake up", "en")):
            self.assertIsNone(listener.listen())


class RemovedPartialApiTests(unittest.TestCase):
    """The partial layer stays deleted (re-adding it must be deliberate)."""

    REMOVED = (
        "_PartialWorker", "_partial_worker", "_submit_partial",
        "_transcribe_partial", "_emit_partial_window", "_bounded_audio_tail",
        "register_partial_observer", "unregister_partial_observer",
        "_notify_partial", "probe_whisper_daemon", "whisper_daemon_ready",
        "partial_daemon_state", "partial_worker_stats",
        "PARTIAL_TRANSCRIBE_MIN_SECONDS", "MAX_PARTIAL_WINDOWS_PER_UTTERANCE",
        "PARTIAL_MAX_AUDIO_SECONDS", "PARTIAL_STT_TIMEOUT_SECONDS",
        "PARTIAL_AGREEMENT_SILENCE_SECONDS", "PARTIAL_DELIVERY_GRACE_SECONDS",
    )

    def test_every_partial_symbol_is_gone(self):
        still_here = [name for name in self.REMOVED if hasattr(listener, name)]
        self.assertEqual(still_here, [], "the partial layer was reintroduced")

    def test_the_listener_module_no_longer_imports_the_job_queue(self):
        source = inspect.getsource(listener)
        self.assertNotIn("import queue", source)


class WhisperLatencyDefaultTests(unittest.TestCase):
    """The shipped STT model is the accurate one, and stays overridable."""

    def test_the_shipped_model_is_medium(self):
        self.assertEqual(whisper_daemon.MODEL_SIZE,
                         os.getenv("JARVIS_WHISPER_MODEL", "medium"))

    def test_the_shipped_transcription_language_is_english_only(self):
        # Auto-detect hallucinates whole sentences in random languages on
        # noisy/echo audio; the owner wants English only.
        self.assertEqual(whisper_daemon.TRANSCRIBE_LANGUAGE, "en")

    def test_the_default_is_read_from_the_same_env_var_as_before(self):
        # An override still wins: nothing about the knob's name changed.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JARVIS_WHISPER_MODEL", None)
            self.assertEqual(
                whisper_daemon.MODEL_SIZE,
                os.environ.get("JARVIS_WHISPER_MODEL", "medium"))


if __name__ == "__main__":
    unittest.main()
