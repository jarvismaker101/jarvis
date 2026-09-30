import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import URLError

import requests
import speech_recognition as sr

from backend.services import transcription


def _make_audio(sample_rate=16000, sample_width=2):
    return sr.AudioData(b"\x01\x00" * 160, sample_rate, sample_width)


def _inworld_response(transcript="hello jarvis", status_code=200):
    response = Mock()
    response.status_code = status_code
    response.json.return_value = {
        "transcription": {
            "transcript": transcript,
            "isFinal": True,
            "wordTimestamps": [],
        },
        "usage": {},
    }
    return response


class InworldRequestShapeTests(unittest.TestCase):
    """recognize_inworld builds the documented API request exactly."""

    def setUp(self):
        self.audio = _make_audio()

    def test_posts_documented_payload_with_basic_auth(self):
        with patch.object(transcription, "INWORLD_STT_API_KEY", "test-inworld-key"), \
             patch.object(
                 transcription._session, "post",
                 return_value=_inworld_response("hello jarvis"),
             ) as post:
            result = transcription.recognize_inworld(self.audio)

        self.assertEqual(result, "hello jarvis")
        post.assert_called_once()
        args, kwargs = post.call_args
        self.assertEqual(args[0], transcription.INWORLD_STT_URL)
        self.assertEqual(
            kwargs["headers"]["Authorization"], "Basic test-inworld-key"
        )
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")
        self.assertEqual(kwargs["timeout"], transcription.INWORLD_STT_TIMEOUT)
        # [P0-03] With one engine per turn this IS the turn's worst case, so it
        # is tightened from the old anonymous (5, 15).
        connect, read = transcription.INWORLD_STT_TIMEOUT
        self.assertLessEqual(connect, 5)
        self.assertLessEqual(read, 15)

        body = kwargs["json"]
        config = body["transcribeConfig"]
        self.assertEqual(config["modelId"], transcription.INWORLD_STT_MODEL)
        self.assertEqual(config["audioEncoding"], "AUTO_DETECT")
        self.assertEqual(config["sampleRateHertz"], self.audio.sample_rate)
        self.assertEqual(config["numberOfChannels"], 1)
        self.assertNotIn("language", config)
        self.assertNotIn("prompts", config)

        expected_content = base64.b64encode(
            self.audio.get_wav_data()
        ).decode("ascii")
        self.assertEqual(body["audioData"]["content"], expected_content)

    def test_language_is_reduced_to_iso639_base(self):
        with patch.object(transcription, "INWORLD_STT_API_KEY", "k"), \
             patch.object(
                 transcription._session, "post",
                 return_value=_inworld_response(),
             ) as post:
            transcription.recognize_inworld(self.audio, language="en-IN")
            transcription.recognize_inworld(self.audio, language="hi-IN")

        first_config = post.call_args_list[0][1]["json"]["transcribeConfig"]
        second_config = post.call_args_list[1][1]["json"]["transcribeConfig"]
        self.assertEqual(first_config["language"], "en")
        self.assertEqual(second_config["language"], "hi")

    def test_prompts_passed_when_given(self):
        prompts = ["Jarvis", "jervis", "utho", "jago", "chalu"]
        with patch.object(transcription, "INWORLD_STT_API_KEY", "k"), \
             patch.object(
                 transcription._session, "post",
                 return_value=_inworld_response(),
             ) as post:
            transcription.recognize_inworld(self.audio, prompts=prompts)

        config = post.call_args[1]["json"]["transcribeConfig"]
        self.assertEqual(config["prompts"], prompts)


class InworldFailureTests(unittest.TestCase):
    def setUp(self):
        self.audio = _make_audio()

    def test_missing_key_raises_request_error_without_network(self):
        post = Mock()
        with patch.object(transcription, "INWORLD_STT_API_KEY", ""), \
             patch.object(transcription._session, "post", post):
            with self.assertRaises(sr.RequestError) as ctx:
                transcription.recognize_inworld(self.audio)
        self.assertIn("INWORLD_STT_API_KEY", str(ctx.exception))
        post.assert_not_called()

    def test_empty_transcript_raises_unknown_value(self):
        with patch.object(transcription, "INWORLD_STT_API_KEY", "k"), \
             patch.object(
                 transcription._session, "post",
                 return_value=_inworld_response("   "),
             ):
            with self.assertRaises(sr.UnknownValueError):
                transcription.recognize_inworld(self.audio)

    def test_grpc_error_bodies_map_to_request_error(self):
        for grpc_code, http_status, message in (
            (3, 400, "Invalid argument"),
            (8, 429, "Resource exhausted"),
            (16, 401, "Unauthenticated"),
        ):
            body = {"code": grpc_code, "message": message}
            response = Mock()
            response.status_code = http_status
            response.text = json.dumps(body)
            response.json.return_value = body
            with patch.object(transcription, "INWORLD_STT_API_KEY", "k"), \
                 patch.object(
                     transcription._session, "post", return_value=response,
                 ):
                with self.assertRaises(sr.RequestError) as ctx:
                    transcription.recognize_inworld(self.audio)
            self.assertIn(str(http_status), str(ctx.exception))
            self.assertIn(message, str(ctx.exception))

    def test_request_exception_maps_to_request_error(self):
        with patch.object(transcription, "INWORLD_STT_API_KEY", "k"), \
             patch.object(
                 transcription._session, "post",
                 side_effect=requests.RequestException("connection reset"),
             ):
            with self.assertRaises(sr.RequestError):
                transcription.recognize_inworld(self.audio)

    def test_invalid_json_body_raises_request_error(self):
        response = Mock()
        response.status_code = 200
        response.json.side_effect = ValueError("bad json")
        with patch.object(transcription, "INWORLD_STT_API_KEY", "k"), \
             patch.object(transcription._session, "post", return_value=response):
            with self.assertRaises(sr.RequestError):
                transcription.recognize_inworld(self.audio)


class _FakeDaemonResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class LocalWhisperTests(unittest.TestCase):
    def setUp(self):
        self.audio = _make_audio()

    def _patch_urlopen(self, payload):
        response = _FakeDaemonResponse(json.dumps(payload).encode("utf-8"))
        return patch.object(transcription, "urlopen", return_value=response)

    def test_ok_with_text_returns_transcript_and_language(self):
        with self._patch_urlopen(
            {"ok": True, "text": "jarvis utho", "language": "hi",
             "language_probability": 0.98}
        ) as urlopen:
            text, language = transcription.recognize_local_whisper(self.audio)

        self.assertEqual(text, "jarvis utho")
        self.assertEqual(language, "hi")

        request = urlopen.call_args[0][0]
        self.assertEqual(
            request.full_url, f"{transcription.LOCAL_WHISPER_URL}/transcribe"
        )
        self.assertEqual(
            request.get_header("Content-type"), "application/octet-stream"
        )
        self.assertEqual(request.data, self.audio.get_wav_data())
        self.assertEqual(urlopen.call_args[1]["timeout"], 15.0)

    def test_missing_language_returns_none(self):
        with self._patch_urlopen({"ok": True, "text": "hello"}):
            text, language = transcription.recognize_local_whisper(self.audio)
        self.assertEqual(text, "hello")
        self.assertIsNone(language)

    def test_ok_with_empty_text_raises_unknown_value(self):
        with self._patch_urlopen({"ok": True, "text": "   ", "language": "en"}):
            with self.assertRaises(sr.UnknownValueError):
                transcription.recognize_local_whisper(self.audio)

    def test_not_ok_payload_raises_request_error(self):
        with self._patch_urlopen({"ok": False, "error": "model exploded"}):
            with self.assertRaises(sr.RequestError):
                transcription.recognize_local_whisper(self.audio)

    def test_connection_refused_raises_request_error(self):
        with patch.object(
            transcription, "urlopen",
            side_effect=URLError("[WinError 10061] No connection could be made"),
        ):
            with self.assertRaises(sr.RequestError):
                transcription.recognize_local_whisper(self.audio)


class ListenerSttChainTests(unittest.TestCase):
    """recognize_multilingual: [P0-03] exactly ONE engine per turn.

    The engine is resolved per call from the listening role in settings. This
    used to describe a CHAIN (Inworld primary, local whisper fallback and a
    per-language Google->Groq loop as the final net) whose serial worst case was
    117s; the ladder is gone, so these tests now pin that a failure of the
    selected engine ENDS the turn â€” the other engines are never contacted.
    """

    def setUp(self):
        from backend.services import listener as listener_mod
        from backend.services import model_registry
        self.listener_mod = listener_mod
        self.model_registry = model_registry
        self.audio = object()
        self._tmp = tempfile.TemporaryDirectory()
        self._settings_path = Path(self._tmp.name) / "jarvis_settings.json"
        self._orig_settings_file = model_registry.SETTINGS_FILE
        model_registry.SETTINGS_FILE = self._settings_path

    def tearDown(self):
        self.model_registry.SETTINGS_FILE = self._orig_settings_file
        self._tmp.cleanup()

    def _select_listening(self, provider, model):
        self.model_registry.set_model_for_role("listening", provider, model)

    def _assert_one_engine(self, inworld, whisper, google):
        calls = (inworld.call_count, whisper.call_count, google.call_count)
        self.assertEqual(sum(1 for count in calls if count), 1,
                         f"more than one engine ran for one turn: {calls}")

    def test_inworld_success_is_the_only_engine_contacted(self):
        listener_mod = self.listener_mod
        with patch.object(
            listener_mod, "recognize_inworld", return_value="Jarvis Utho",
        ) as inworld, patch.object(
            listener_mod, "recognize_local_whisper",
        ) as whisper, patch.object(
            listener_mod, "recognize_google_or_groq",
        ) as google:
            raw, normalized, language = listener_mod.recognize_multilingual(self.audio)

        self.assertEqual(raw, "Jarvis Utho")
        self.assertEqual(normalized, "jarvis utho")
        self.assertEqual(language, "auto")
        inworld.assert_called_once_with(self.audio, language="en")
        whisper.assert_not_called()
        google.assert_not_called()

    def test_inworld_request_shape_carries_english_language_hint(self):
        listener_mod = self.listener_mod
        with patch.object(transcription, "INWORLD_STT_API_KEY", "k"), \
             patch.object(
                 transcription._session, "post",
                 return_value=_inworld_response("jarvis utho"),
             ) as post, patch.object(
                 listener_mod, "recognize_local_whisper",
             ) as whisper, patch.object(
                 listener_mod, "recognize_google_or_groq",
             ) as google:
            raw, normalized, language = listener_mod.recognize_multilingual(
                _make_audio()
            )

        self.assertEqual(
            (raw, normalized, language), ("jarvis utho", "jarvis utho", "auto")
        )
        post.assert_called_once()
        config = post.call_args[1]["json"]["transcribeConfig"]
        self.assertEqual(config["language"], "en")
        whisper.assert_not_called()
        google.assert_not_called()

    def test_inworld_devanagari_ends_the_turn_with_no_second_engine(self):
        """[P0-03] The English hint was ignored: a failed turn, not a retry."""
        listener_mod = self.listener_mod
        with patch.object(
            listener_mod, "recognize_inworld",
            return_value="\u092e\u0948\u0902 \u0924\u094b hi",
        ) as inworld, patch.object(
            listener_mod, "recognize_local_whisper",
            return_value=("main toh hi", "hi"),
        ) as whisper, patch.object(
            listener_mod, "recognize_google_or_groq", return_value="main toh hi",
        ) as google:
            result = listener_mod.recognize_multilingual(self.audio)

        self.assertEqual(result, (None, None, None))
        inworld.assert_called_once_with(self.audio, language="en")
        whisper.assert_not_called()
        google.assert_not_called()
        self.assertEqual(listener_mod.LAST_STT_FAILURE,
                         ("inworld", "non-english-script"))

    def test_inworld_no_speech_ends_the_turn(self):
        listener_mod = self.listener_mod
        with patch.object(
            listener_mod, "recognize_inworld", side_effect=sr.UnknownValueError(),
        ) as inworld, patch.object(
            listener_mod, "recognize_local_whisper",
            return_value=("namaste", "hi"),
        ) as whisper, patch.object(
            listener_mod, "recognize_google_or_groq",
        ) as google:
            result = listener_mod.recognize_multilingual(self.audio)

        self.assertEqual(result, (None, None, None))
        inworld.assert_called_once_with(self.audio, language="en")
        whisper.assert_not_called()
        google.assert_not_called()
        self.assertEqual(listener_mod.LAST_STT_FAILURE, ("inworld", "no-speech"))

    def test_inworld_request_error_ends_the_turn(self):
        listener_mod = self.listener_mod
        with patch.object(
            listener_mod, "recognize_inworld", side_effect=sr.RequestError("down"),
        ) as inworld, patch.object(
            listener_mod, "recognize_local_whisper",
            return_value=("hello there", "en"),
        ) as whisper, patch.object(listener_mod, "recognize_google_or_groq") as google:
            result = listener_mod.recognize_multilingual(self.audio)

        self.assertEqual(result, (None, None, None))
        inworld.assert_called_once_with(self.audio, language="en")
        whisper.assert_not_called()
        google.assert_not_called()
        self.assertEqual(listener_mod.LAST_STT_FAILURE, ("inworld", "down"))

    def test_inworld_whitespace_ends_the_turn(self):
        listener_mod = self.listener_mod
        with patch.object(
            listener_mod, "recognize_inworld", return_value="   ",
        ) as inworld, patch.object(
            listener_mod, "recognize_local_whisper",
            return_value=("fallback", None),
        ) as whisper, patch.object(listener_mod, "recognize_google_or_groq") as google:
            result = listener_mod.recognize_multilingual(self.audio)

        self.assertEqual(result, (None, None, None))
        inworld.assert_called_once_with(self.audio, language="en")
        whisper.assert_not_called()
        google.assert_not_called()
        self.assertEqual(listener_mod.LAST_STT_FAILURE,
                         ("inworld", "empty transcript"))

    def test_the_google_groq_loop_is_never_a_fallback(self):
        """The per-language loop was the third and fourth serial attempt."""
        listener_mod = self.listener_mod
        with patch.object(
            listener_mod, "recognize_inworld", side_effect=sr.RequestError("down"),
        ) as inworld, patch.object(
            listener_mod, "recognize_local_whisper",
            side_effect=sr.RequestError("no daemon"),
        ) as whisper, patch.object(
            listener_mod, "recognize_google_or_groq", return_value="hey there",
        ) as google:
            result = listener_mod.recognize_multilingual(self.audio)

        self.assertEqual(result, (None, None, None))
        inworld.assert_called_once_with(self.audio, language="en")
        whisper.assert_not_called()
        google.assert_not_called()

    def test_the_single_engine_failing_returns_the_empty_tuple(self):
        listener_mod = self.listener_mod
        with patch.object(
            listener_mod, "recognize_inworld", side_effect=sr.UnknownValueError(),
        ) as inworld, patch.object(
            listener_mod, "recognize_local_whisper",
            side_effect=sr.RequestError("no daemon"),
        ) as whisper, patch.object(
            listener_mod, "recognize_google_or_groq",
            side_effect=sr.UnknownValueError(),
        ) as google:
            result = listener_mod.recognize_multilingual(self.audio)

        self.assertEqual(result, (None, None, None))
        inworld.assert_called_once_with(self.audio, language="en")
        whisper.assert_not_called()
        google.assert_not_called()

    def test_whisper_selected_runs_only_local_whisper(self):
        self._select_listening("whisper", "whisper-local")
        listener_mod = self.listener_mod
        with patch.object(
            listener_mod, "recognize_local_whisper",
            return_value=("hello sir", "en"),
        ) as whisper, patch.object(
            listener_mod, "recognize_inworld",
        ) as inworld, patch.object(
            listener_mod, "recognize_google_or_groq",
        ) as google:
            raw, normalized, language = listener_mod.recognize_multilingual(self.audio)

        self.assertEqual(
            (raw, normalized, language), ("hello sir", "hello sir", "en")
        )
        whisper.assert_called_once()
        inworld.assert_not_called()
        google.assert_not_called()

    def test_whisper_selected_failure_never_reaches_another_engine(self):
        self._select_listening("whisper", "whisper-local")
        listener_mod = self.listener_mod
        with patch.object(
            listener_mod, "recognize_local_whisper",
            side_effect=sr.RequestError("no daemon"),
        ) as whisper, patch.object(
            listener_mod, "recognize_inworld",
            return_value="an inworld second opinion",
        ) as inworld, patch.object(
            listener_mod, "recognize_google_or_groq", return_value="hey there",
        ) as google:
            result = listener_mod.recognize_multilingual(self.audio)

        self.assertEqual(result, (None, None, None))
        whisper.assert_called_once()
        inworld.assert_not_called()
        google.assert_not_called()
        self.assertEqual(listener_mod.LAST_STT_FAILURE[0], "local-whisper")

    def test_inworld_selected_runs_only_inworld(self):
        self._select_listening("inworld", "inworld/inworld-stt-1")
        listener_mod = self.listener_mod
        with patch.object(
            listener_mod, "recognize_inworld", return_value="jarvis utho",
        ) as inworld, patch.object(
            listener_mod, "recognize_local_whisper",
        ) as whisper, patch.object(
            listener_mod, "recognize_google_or_groq",
        ) as google:
            raw, normalized, language = listener_mod.recognize_multilingual(self.audio)

        self.assertEqual(
            (raw, normalized, language), ("jarvis utho", "jarvis utho", "auto")
        )
        inworld.assert_called_once_with(self.audio, language="en")
        whisper.assert_not_called()
        google.assert_not_called()

    def test_registry_read_failure_defaults_to_inworld(self):
        listener_mod = self.listener_mod
        with patch.object(
            self.model_registry, "get_model_for_role",
            side_effect=RuntimeError("registry unavailable"),
        ), patch.object(
            listener_mod, "recognize_inworld", return_value="jarvis utho",
        ) as inworld, patch.object(
            listener_mod, "recognize_local_whisper",
        ) as whisper:
            raw, normalized, language = listener_mod.recognize_multilingual(self.audio)

        self.assertEqual(normalized, "jarvis utho")
        inworld.assert_called_once()
        whisper.assert_not_called()

    def test_unknown_listening_provider_falls_back_to_inworld(self):
        self._settings_path.write_text(
            json.dumps({"listening_model": {"provider": "bogus", "model": "x"}}),
            encoding="utf-8",
        )
        listener_mod = self.listener_mod
        with patch.object(
            listener_mod, "recognize_inworld", return_value="jarvis utho",
        ) as inworld, patch.object(
            listener_mod, "recognize_local_whisper",
        ) as whisper, patch.object(
            listener_mod, "recognize_google_or_groq",
        ) as google:
            raw, normalized, language = listener_mod.recognize_multilingual(self.audio)

        self.assertEqual(normalized, "jarvis utho")
        inworld.assert_called_once()
        whisper.assert_not_called()
        google.assert_not_called()


if __name__ == "__main__":
    unittest.main()
