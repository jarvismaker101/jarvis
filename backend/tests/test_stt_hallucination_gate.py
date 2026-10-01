"""STT hallucination gate — junk transcripts never become user speech.

Found live on 2026-09-23: with the AEC degraded (headset playback captured
into the mic), local Whisper transcribed noise/TTS-echo into its memorised
filler and the brain ANSWERED it:

  [HEARD:local-whisper] jarvis, a ver si te acuerdas de esto
  [HEARD:local-whisper] jervis, wake up, jervis, utho, jago, chalu
  (and "chalu, chalu, chalu, ..." loops in the partial windows)

Two root causes, both pinned here:

  1. whisper_daemon applied the wake-bias ``INITIAL_PROMPT`` to EVERY
     transcription. On non-speech audio the model echoes the prompt tokens
     verbatim — the "hallucinated words" WERE the prompt. The prompt is now
     opt-in (``X-Jarvis-Purpose: wake``, sent only by the watcher).
  2. Nothing vetoed degenerate transcripts at the commit boundary, so
     hallucinations reached /ask as user speech (and is_wake_word matched the
     prompt echo, which could false-launch the stack).

``is_hallucinated_transcript`` is the shared deterministic gate; the positive
cases pin that it NEVER drops a real command, wake phrase, literal payload,
or repeated safety word.

No microphone, TTS engine, model download or outbound request is made here.
"""

import json
import threading
import unittest
import urllib.request
from types import SimpleNamespace
from unittest.mock import patch

import speech_recognition as sr

from backend import watcher
from backend import whisper_daemon
from backend.services import listener
from backend.services import wake_engine
from backend.services.transcription import is_hallucinated_transcript

LIVE_PROMPT_ECHO = "jarvis, wake up, jervis, utho, jago, chalu"
LIVE_PROMPT_ECHO_VARIANT = "jervis, wake up, jervis, utho, jago, chalu"
LIVE_LOOP = "chalu, " * 12 + "chalu."
LIVE_SILENCE_PHRASE = "Jarvis, a ver si te acuerdas de esto"
LIVE_EIGHT_TIMES = " ".join([LIVE_PROMPT_ECHO_VARIANT] * 8)


class RejectsObservedHallucinations(unittest.TestCase):
    """Acceptance: every junk shape observed live is rejected."""

    def test_wake_bias_prompt_echo_is_rejected(self):
        self.assertTrue(is_hallucinated_transcript(LIVE_PROMPT_ECHO))
        self.assertTrue(is_hallucinated_transcript(LIVE_PROMPT_ECHO_VARIANT))

    def test_prompt_echo_repeated_eight_times_is_rejected(self):
        self.assertTrue(is_hallucinated_transcript(LIVE_EIGHT_TIMES))

    def test_looped_single_token_is_rejected(self):
        self.assertTrue(is_hallucinated_transcript(LIVE_LOOP))
        self.assertTrue(is_hallucinated_transcript("chalu chalu chalu"))

    def test_memorised_silence_phrase_is_rejected(self):
        self.assertTrue(is_hallucinated_transcript(LIVE_SILENCE_PHRASE))
        self.assertTrue(is_hallucinated_transcript("a ver si te acuerdas de esto"))
        self.assertTrue(is_hallucinated_transcript("thanks for watching"))

    def test_filler_only_utterances_are_rejected(self):
        self.assertTrue(is_hallucinated_transcript("you"))
        self.assertTrue(is_hallucinated_transcript("um uh"))

    def test_one_token_dominating_babble_is_rejected(self):
        self.assertTrue(
            is_hallucinated_transcript("chalu chalu wake chalu chalu jarvis chalu"))

    def test_looped_phrase_is_rejected(self):
        self.assertTrue(
            is_hallucinated_transcript(" ".join(["play music"] * 5)))


class AcceptsRealSpeech(unittest.TestCase):
    """Acceptance: real commands, wake phrases and payloads always pass."""

    def test_real_commands_pass(self):
        for text in (
            "open chrome",
            "what's on my screen",
            "Create File Q4 Report.TXT",
            "search for the latest gpu prices",
            "jarvis chalu karo youtube",
        ):
            self.assertFalse(is_hallucinated_transcript(text), text)

    def test_short_wake_phrases_pass(self):
        for text in (
            "jarvis",
            "jarvis wake up",
            "wake up jarvis",
            "utho jarvis",
            "jarvis utho jago chalu",  # 4 tokens: below the 5-token echo rule
            "Jarvis Utho",
        ):
            self.assertFalse(is_hallucinated_transcript(text), text)

    def test_repeated_safety_words_pass(self):
        for text in ("stop stop stop", "no no no no", "cancel cancel"):
            self.assertFalse(is_hallucinated_transcript(text), text)

    def test_a_command_repeated_three_times_still_passes(self):
        self.assertFalse(
            is_hallucinated_transcript(
                "what time is it what time is it what time is it"))

    def test_literal_payloads_pass(self):
        for text in (
            'copy "Q4 Report.PDF" to C:\\Users\\Me\\Final',
            "run deploy {{step0.files.0}}.ps1 --Wait True",
            "fetch https://ex.test/a?X-Amz-Signature=AbC123&Expires=99",
        ):
            self.assertFalse(is_hallucinated_transcript(text), text)

    def test_a_sentence_about_a_silence_phrase_pass(self):
        self.assertFalse(
            is_hallucinated_transcript(
                "can you translate a ver si te acuerdas de esto to english please"))

    def test_empty_text_is_not_a_hallucination(self):
        self.assertFalse(is_hallucinated_transcript(""))
        self.assertFalse(is_hallucinated_transcript(None))


class WhisperDaemonWakeBiasTests(unittest.TestCase):
    """The wake-bias prompt is opt-in: conversation transcription is unbiased."""

    class _CaptureModel:
        def __init__(self):
            self.kwargs = None

        def transcribe(self, _audio, **kwargs):
            self.kwargs = kwargs
            return [SimpleNamespace(text=" hello")], SimpleNamespace(
                language="en", language_probability=0.9
            )

    def setUp(self):
        self._saved = {
            "model": whisper_daemon.model,
            "device": whisper_daemon.device,
            "_model_error": whisper_daemon._model_error,
            "_model_load_finished": whisper_daemon._model_load_finished,
            "_model_load_started": whisper_daemon._model_load_started,
        }
        self.model = self._CaptureModel()
        whisper_daemon.model = self.model
        whisper_daemon.device = "cpu"
        whisper_daemon._model_error = None
        whisper_daemon._model_load_finished = threading.Event()
        whisper_daemon._model_load_finished.set()
        whisper_daemon._model_load_started = True
        self.server = whisper_daemon.build_server(0, bind_retry_seconds=0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop)

    def _stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        for name, value in self._saved.items():
            setattr(whisper_daemon, name, value)

    def _post(self, headers=None):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/transcribe",
            data=b"RIFFxxxx",
            method="POST",
            headers=headers or {},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    def test_conversation_transcription_carries_no_wake_bias(self):
        payload = self._post()
        self.assertTrue(payload["ok"])
        self.assertIsNone(self.model.kwargs["initial_prompt"])

    def test_wake_purpose_requests_the_bias_prompt(self):
        payload = self._post({"X-Jarvis-Purpose": "wake"})
        self.assertTrue(payload["ok"])
        self.assertEqual(
            self.model.kwargs["initial_prompt"], whisper_daemon.INITIAL_PROMPT
        )

    def test_conversation_transcription_decodes_greedily(self):
        """[PERF] One draft instead of a beam of five.

        The latency pass turned beam search off (temperature 0 keeps greedy
        decode deterministic). Pinned here because a silent revert would cost
        ~1.5-2x the decode time on every turn again.
        """
        payload = self._post()
        self.assertTrue(payload["ok"])
        self.assertEqual(self.model.kwargs["beam_size"], 1)
        self.assertEqual(self.model.kwargs["temperature"], 0.0)


class ListenerCommitGateTests(unittest.TestCase):
    """The conversation listener never commits a hallucination."""

    def setUp(self):
        listener._turn_stabilizer.reset()
        self.addCleanup(listener._turn_stabilizer.reset)

    def test_listen_returns_none_for_a_hallucinated_final(self):
        audio = sr.AudioData(b"\x00" * 3200, 16000, 2)
        with patch.object(listener, "_capture_audio", return_value=audio), \
             patch.object(listener, "recognize_multilingual",
                          return_value=(LIVE_PROMPT_ECHO,
                                        LIVE_PROMPT_ECHO.lower(), "en")):
            self.assertIsNone(listener.listen())

    def test_listen_still_commits_real_speech(self):
        audio = sr.AudioData(b"\x00" * 3200, 16000, 2)
        with patch.object(listener, "_capture_audio", return_value=audio), \
             patch.object(listener, "recognize_multilingual",
                          return_value=("open chrome", "open chrome", "en")):
            self.assertEqual(listener.listen(), "open chrome")

    def test_a_whisper_hallucination_is_dropped_with_no_second_engine(self):
        """[P0-03] The gate rejects the transcript; the ladder is gone.

        This used to assert that the hallucinated whisper output "falls through
        to the next engine" (Inworld). With ONE engine per turn there is no next
        engine: a hallucination is a failed turn, never a prompt for another
        network call.
        """
        with patch.object(listener.model_registry, "get_model_for_role",
                          return_value={"provider": "whisper"}), \
             patch.object(listener, "recognize_local_whisper",
                          return_value=(LIVE_PROMPT_ECHO, "en")) as whisper, \
             patch.object(listener, "recognize_inworld",
                          return_value="open chrome") as inworld, \
             patch.object(listener, "recognize_google_or_groq") as google:
            raw, normalized, language = listener.recognize_multilingual(object())
        self.assertEqual((raw, normalized, language), (None, None, None))
        whisper.assert_called_once()
        inworld.assert_not_called()
        google.assert_not_called()
        self.assertEqual(listener.LAST_STT_FAILURE, ("local-whisper", "hallucination"))


class WatcherWakeGateTests(unittest.TestCase):
    """A prompt-echo hallucination can never wake or enter candidates."""

    def test_prompt_echo_never_matches_a_wake_word(self):
        self.assertFalse(watcher.is_wake_word(LIVE_PROMPT_ECHO))
        self.assertFalse(watcher.is_wake_word(LIVE_PROMPT_ECHO_VARIANT))
        self.assertFalse(watcher.is_wake_word(LIVE_EIGHT_TIMES))
        self.assertFalse(watcher.is_wake_word(LIVE_LOOP))

    def test_real_wake_phrases_still_match(self):
        self.assertTrue(watcher.is_wake_word("Jarvis, open Chrome"))
        self.assertTrue(watcher.is_wake_word("wake up jarvis"))

    def test_candidates_drop_hallucinations(self):
        candidates = watcher._add_candidates(
            [], set(), ["open chrome", LIVE_PROMPT_ECHO, LIVE_LOOP])
        self.assertEqual(candidates, ["open chrome"])


class VerificationGateTests(unittest.TestCase):
    """Wake verification (F36) fails closed on a prompt echo."""

    def setUp(self):
        wake_engine.reset_pre_roll()
        self.addCleanup(wake_engine.reset_pre_roll)
        verify = patch.object(wake_engine, "WAKE_ONLINE_VERIFY", True)
        verify.start()
        self.addCleanup(verify.stop)
        wake_engine.pre_roll().feed(sr.AudioData(
            b"\x00" * int(16000 * 0.6) * 2, 16000, 2))

    def test_wake_verification_rejects_a_prompt_echo(self):
        with patch.object(watcher, "_transcribe_with_daemon",
                          return_value=(LIVE_PROMPT_ECHO, {})):
            self.assertFalse(wake_engine.online_verify())

    def test_wake_verification_accepts_a_real_wake_phrase(self):
        with patch.object(watcher, "_transcribe_with_daemon",
                          return_value=("wake up jarvis", {})):
            self.assertTrue(wake_engine.online_verify())


if __name__ == "__main__":
    unittest.main()