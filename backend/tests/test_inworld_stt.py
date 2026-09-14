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
                 transcription.requests, "post",
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
        self.assertEqual(kwargs["timeout"], (5, 15))

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
                 transcription.requests, "post",
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
                 transcription.requests, "post",
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
             patch.object(transcription.requests, "post", post):
            with self.assertRaises(sr.RequestError) as ctx:
                transcription.recognize_inworld(self.audio)
        self.assertIn("INWORLD_STT_API_KEY", str(ctx.exception))
        post.assert_not_called()

    def test_empty_transcript_raises_unknown_value(self):
        with patch.object(transcription, "INWORLD_STT_API_KEY", "k"), \
             patch.object(
                 transcription.requests, "post",
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
                     transcription.requests, "post", return_value=response,
                 ):
                with self.assertRaises(sr.RequestError) as ctx:
                    transcription.recognize_inworld(self.audio)
            self.assertIn(str(http_status), str(ctx.exception))
            self.assertIn(message, str(ctx.exception))

    def test_request_exception_maps_to_request_error(self):
        with patch.object(transcription, "INWORLD_STT_API_KEY", "k"), \
             patch.object(
                 transcription.requests, "post",
                 side_effect=requests.RequestException("connection reset"),
             ):
            with self.assertRaises(sr.RequestError):
                transcription.recognize_inworld(self.audio)

    def test_invalid_json_body_raises_request_error(self):
        response = Mock()
        response.status_code = 200
        response.json.side_effect = ValueError("bad json")
        with patch.object(transcription, "INWORLD_STT_API_KEY", "k"), \
             patch.object(transcription.requests, "post", return_value=response):
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
    """recognize_multilingual: Inworld primary (English-hinted, roman-script
    only), local whisper fallback, Google->Groq loop unchanged as the final
    net. The listening-role switch (whisper <-> inworld) is resolved per
    call from the registry; these tests isolate the settings file so the
    default (inworld-first) is deterministic."""

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

    def test_inworld_success_short_circuits_other_engines(self):
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
                 transcription.requests, "post",
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

    def test_inworld_devanagari_transcript_falls_to_local_whisper(self):
        listener_mod = self.listener_mod
        with patch.object(
            listener_mod, "recognize_inworld",
            return_value="\u092e\u0948\u0902 \u0924\u094b hi",
        ) as inworld, patch.object(
            listener_mod, "recognize_local_whisper",
            return_value=("main toh hi", "hi"),
        ) as whisper, patch.object(
            listener_mod, "recognize_google_or_groq",
        ) as google:
            raw, normalized, language = listener_mod.recognize_multilingual(self.audio)

        self.assertEqual(
            (raw, normalized, language), ("main toh hi", "main toh hi", "hi")
        )
        inworld.assert_called_once_with(self.audio, language="en")
        whisper.assert_called_once()
        google.assert_not_called()

    def test_inworld_devanagari_reaches_google_only_if_whisper_fails(self):
        listener_mod = self.listener_mod
        with patch.object(
            listener_mod, "recognize_inworld",
            return_value="\u092e\u0948\u0902 \u0924\u094b hi",
        ), patch.object(
            listener_mod, "recognize_local_whisper",
            side_effect=sr.RequestError("no daemon"),
        ) as whisper, patch.object(
            listener_mod, "recognize_google_or_groq", return_value="main toh hi",
        ) as google:
            raw, normalized, language = listener_mod.recognize_multilingual(self.audio)

        whisper.assert_called_once()
        google.assert_called()
        self.assertEqual(raw, "main toh hi")
        self.assertEqual(language, listener_mod.RECOGNITION_LANGUAGES[0])

    def test_inworld_no_speech_falls_to_local_whisper(self):
        listener_mod = self.listener_mod
        with patch.object(
            listener_mod, "recognize_inworld", side_effect=sr.UnknownValueError(),
        ) as inworld, patch.object(
            listener_mod, "recognize_local_whisper",
            return_value=("namaste", "hi"),
        ) as whisper, patch.object(
            listener_mod, "recognize_google_or_groq",
        ) as google:
            raw, normalized, language = listener_mod.recognize_multilingual(self.audio)

        self.assertEqual((raw, normalized, language), ("namaste", "namaste", "hi"))
        inworld.assert_called_once_with(self.audio, language="en")
        whisper.assert_called_once()
        google.assert_not_called()

    def test_inworld_request_error_falls_to_local_whisper(self):
        listener_mod = self.listener_mod
        with patch.object(
            listener_mod, "recognize_inworld", side_effect=sr.RequestError("down"),
        ) as inworld, patch.object(
            listener_mod, "recognize_local_whisper",
            return_value=("hello there", "en"),
        ), patch.object(listener_mod, "recognize_google_or_groq") as google:
            raw, normalized, language = listener_mod.recognize_multilingual(self.audio)

        self.assertEqual(
            (raw, normalized, language), ("hello there", "hello there", "en")
        )
        inworld.assert_called_once_with(self.audio, language="en")
        google.assert_not_called()

    def test_inworld_whitespace_falls_through(self):
        listener_mod = self.listener_mod
        with patch.object(
            listener_mod, "recognize_inworld", return_value="   ",
        ) as inworld, patch.object(
            listener_mod, "recognize_local_whisper",
            return_value=("fallback", None),
        ), patch.object(listener_mod, "recognize_google_or_groq") as google:
            raw, normalized, language = listener_mod.recognize_multilingual(self.audio)

        self.assertEqual((raw, normalized, language), ("fallback", "fallback", None))
        inworld.assert_called_once_with(self.audio, language="en")
        google.assert_not_called()

    def test_new_steps_fail_google_groq_loop_still_runs(self):
        listener_mod = self.listener_mod
        with patch.object(
            listener_mod, "recognize_inworld", side_effect=sr.RequestError("down"),
        ) as inworld, patch.object(
            listener_mod, "recognize_local_whisper",
            side_effect=sr.RequestError("no daemon"),
        ), patch.object(
            listener_mod, "recognize_google_or_groq", return_value="hey there",
        ) as google:
            raw, normalized, language = listener_mod.recognize_multilingual(self.audio)

        self.assertEqual(raw, "hey there")
        self.assertEqual(normalized, "hey there")
        self.assertEqual(language, listener_mod.RECOGNITION_LANGUAGES[0])
        inworld.assert_called_once_with(self.audio, language="en")
        google.assert_called()

    def test_all_engines_fail_returns_none_tuple(self):
        listener_mod = self.listener_mod
        with patch.object(
            listener_mod, "recognize_inworld", side_effect=sr.UnknownValueError(),
        ) as inworld, patch.object(
            listener_mod, "recognize_local_whisper",
            side_effect=sr.RequestError("no daemon"),
        ), patch.object(
            listener_mod, "recognize_google_or_groq",
            side_effect=sr.UnknownValueError(),
        ):
            result = listener_mod.recognize_multilingual(self.audio)

        self.assertEqual(result, (None, None, None))
        inworld.assert_called_once_with(self.audio, language="en")

    def test_whisper_selected_calls_local_whisper_first_and_skips_inworld(self):
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

    def test_whisper_selected_falls_to_inworld_then_google(self):
        self._select_listening("whisper", "whisper-local")
        listener_mod = self.listener_mod
        with patch.object(
            listener_mod, "recognize_local_whisper",
            side_effect=sr.RequestError("no daemon"),
        ) as whisper, patch.object(
            listener_mod, "recognize_inworld",
            side_effect=sr.RequestError("offline in tests"),
        ) as inworld, patch.object(
            listener_mod, "recognize_google_or_groq", return_value="hey there",
        ) as google:
            raw, normalized, language = listener_mod.recognize_multilingual(self.audio)

        self.assertEqual(raw, "hey there")
        self.assertEqual(language, listener_mod.RECOGNITION_LANGUAGES[0])
        whisper.assert_called_once()
        inworld.assert_called_once()
        google.assert_called()

    def test_inworld_selected_keeps_current_chain(self):
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

    def test_registry_read_failure_defaults_to_inworld_first(self):
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

    def test_unknown_listening_provider_falls_back_to_inworld_first(self):
        self._settings_path.write_text(
            json.dumps({"listening_model": {"provider": "bogus", "model": "x"}}),
            encoding="utf-8",
        )
        listener_mod = self.listener_mod
        with patch.object(
            listener_mod, "recognize_inworld", return_value="jarvis utho",
        ) as inworld, patch.object(
            listener_mod, "recognize_local_whisper",
        ) as whisper:
            raw, normalized, language = listener_mod.recognize_multilingual(self.audio)

        self.assertEqual(normalized, "jarvis utho")
        inworld.assert_called_once()
        whisper.assert_not_called()


if __name__ == "__main__":
    unittest.main()
