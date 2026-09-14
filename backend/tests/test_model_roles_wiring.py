"""Wiring tests: TTS / Vision / Browser-tool call sites must read the
runtime registry per call and degrade to env defaults on failure.

Zero network, zero real keys — all HTTP mocked.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.services import model_registry
from backend.services.gemini_client import GEMINI_MODEL, GEMINI_CHAT_MODEL
from backend.config import FISH_MODEL, BROWSER_AGENT_MODEL, BROWSER_AGENT_PROVIDER


class WiringTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._settings_path = Path(self._tmp.name) / "jarvis_settings.json"
        self._orig_settings_file = model_registry.SETTINGS_FILE
        self._orig_gemini_key = model_registry.GEMINI_API_KEY
        self._orig_fireworks_key = model_registry.FIREWORKS_API_KEY
        self._orig_fish_key = model_registry.FISH_API_KEY
        self._orig_openrouter_key = model_registry.OPENROUTER_API_KEY
        model_registry.SETTINGS_FILE = self._settings_path
        model_registry.GEMINI_API_KEY = "test-gemini-key"
        model_registry.FIREWORKS_API_KEY = "test-fireworks-key"
        model_registry.FISH_API_KEY = "test-fish-key"
        model_registry.OPENROUTER_API_KEY = "test-openrouter-key"

    def tearDown(self):
        model_registry.SETTINGS_FILE = self._orig_settings_file
        model_registry.GEMINI_API_KEY = self._orig_gemini_key
        model_registry.FIREWORKS_API_KEY = self._orig_fireworks_key
        model_registry.FISH_API_KEY = self._orig_fish_key
        model_registry.OPENROUTER_API_KEY = self._orig_openrouter_key
        self._tmp.cleanup()


class TTSWiringTests(WiringTestBase):
    def test_resolve_tts_model_uses_registry(self):
        from backend.services.fish_voice import _resolve_tts_model
        model_registry.set_model_for_role("tts", "fish", "s1")
        self.assertEqual(_resolve_tts_model(), "s1")
        # fallback to env default when registry corrupt -> still returns something
        self._settings_path.write_text("{not json", encoding="utf-8")
        self.assertEqual(_resolve_tts_model(), FISH_MODEL)

    def test_request_audio_payload_uses_registry_model(self):
        from backend.services import fish_voice
        model_registry.set_model_for_role("tts", "fish", "fish-speech-1.5")
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.content = b"ID3_fake_mp3_content_for_test_payload"
        fake_resp.headers = {"Content-Type": "audio/mpeg"}
        with patch.object(fish_voice._session, "post", return_value=fake_resp) as mock_post:
            fish_voice._request_audio("hello sir")
            # payload and headers must carry the registry model, not the import-time constant
            _, kwargs = mock_post.call_args
            payload = kwargs.get("json") or mock_post.call_args.args[1] if len(mock_post.call_args.args) > 1 else {}
            # The post is called with json=payload and headers containing model
            sent_payload = kwargs.get("json", {})
            sent_headers = kwargs.get("headers", {})
            self.assertEqual(sent_payload.get("model"), "fish-speech-1.5")
            self.assertEqual(sent_headers.get("model"), "fish-speech-1.5")

    def test_do_pcm_stream_payload_uses_registry_model(self):
        from backend.services import fish_voice
        model_registry.set_model_for_role("tts", "fish", "s2")
        # Mock the streaming post + sounddevice to avoid audio hardware
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.iter_content.return_value = [b"\x01\x02" * 2048]
        with patch.object(fish_voice._session, "post", return_value=fake_resp) as mock_post, \
             patch.object(fish_voice, "FISH_API_KEY", "test-fish-key"), \
             patch("sounddevice.OutputStream") as mock_stream, \
             patch.object(fish_voice, "_register_sounddevice_playback") as mock_handle, \
             patch.object(fish_voice, "_get_cached_device", return_value=None):
            handle = MagicMock()
            handle.stopped = False
            mock_handle.return_value = handle
            stream_inst = MagicMock()
            mock_stream.return_value.__enter__ = MagicMock(return_value=stream_inst)
            mock_stream.return_value.__exit__ = MagicMock(return_value=False)
            # Use play=False to collect without actually playing
            ok = fish_voice._do_pcm_stream("hello", play=False)
            self.assertTrue(ok)
            payload = mock_post.call_args.kwargs.get("json", {})
            self.assertEqual(payload.get("model"), "s2")


class VisionWiringTests(WiringTestBase):
    def test_screen_analyzer_uses_gemini_registry_model(self):
        from backend.services.screen_analyzer import _ask_screen_vision_cascade, _extract_result_content
        model_registry.set_model_for_role("vision", "gemini", "gemini-2.5-pro")
        fake_result = {"choices": [{"message": {"content": '{"tip":"ok"}'}}], "grounding_links": []}
        with patch("backend.services.gemini_client.ask_gemini_vision", return_value=fake_result) as mock_gem, \
             patch("backend.services.grok_client.ask_groq_vision", return_value={}) as mock_groq:
            # Ensure gemini available
            with patch("backend.services.gemini_client.is_available", return_value=True):
                result = _ask_screen_vision_cascade("prompt", "data:image/png;base64,abc")
            mock_gem.assert_called_once()
            self.assertEqual(mock_gem.call_args.kwargs.get("model"), "gemini-2.5-pro")
            self.assertEqual(result, fake_result)

    def test_screen_analyzer_uses_openrouter_when_selected(self):
        from backend.services.screen_analyzer import _ask_screen_vision_cascade
        model_registry.set_model_for_role("vision", "openrouter", "google/gemma-4-31b-it:free")
        fake_result = {"choices": [{"message": {"content": '{"tip":"ok"}'}}]}
        with patch("backend.services.openrouter_client.ask_openrouter_vision", return_value=fake_result) as mock_or, \
             patch("backend.services.gemini_client.is_available", return_value=True), \
             patch("backend.services.gemini_client.ask_gemini_vision", return_value={}) as mock_gem:
            result = _ask_screen_vision_cascade("prompt", "data:image/png;base64,abc")
            mock_or.assert_called_once()
            self.assertEqual(mock_or.call_args.kwargs.get("model"), "google/gemma-4-31b-it:free")
            # Should not fall through to gemini when openrouter succeeds
            mock_gem.assert_not_called()

    def test_vision_degrades_to_env_default_on_registry_failure(self):
        from backend.services.screen_analyzer import _ask_screen_vision_cascade
        # Corrupt the registry file — _resolve_vision_model should fall back
        self._settings_path.write_text("{not json", encoding="utf-8")
        fake_result = {"choices": [{"message": {"content": '{"tip":"fallback"}'}}]}
        with patch("backend.services.gemini_client.ask_gemini_vision", return_value=fake_result) as mock_gem, \
             patch("backend.services.grok_client.ask_groq_vision", return_value={}), \
             patch("backend.services.gemini_client.is_available", return_value=True):
            result = _ask_screen_vision_cascade("prompt", "data:image/png;base64,abc")
            self.assertEqual(result, fake_result)
            mock_gem.assert_called_once()
            # Degraded -> called with env default model (GEMINI_MODEL) or None (which defaults inside client)
            got = mock_gem.call_args.kwargs.get("model")
            self.assertIn(got, [GEMINI_MODEL, None])

    def test_screen_control_cascade_uses_registry(self):
        from backend.services import screen_control
        model_registry.set_model_for_role("vision", "gemini", "gemini-2.5-flash")
        fake_result = {"choices": [{"message": {"content": '{"ok":true}'}}]}
        with patch("backend.services.gemini_client.ask_gemini_vision", return_value=fake_result) as mock_gem, \
             patch("backend.services.grok_client.ask_groq_vision", return_value={}), \
             patch("backend.services.gemini_client.is_available", return_value=True):
            result = screen_control._ask_vision_cascade("prompt", "data:image/png;base64,abc")
            mock_gem.assert_called_once()
            self.assertEqual(mock_gem.call_args.kwargs.get("model"), "gemini-2.5-flash")


class BrowserToolWiringTests(WiringTestBase):
    def test_model_turn_uses_registry_provider_and_model(self):
        from backend.services import browser_agent
        from backend import config
        model_registry.set_model_for_role("browser_tool", "fireworks", "accounts/fireworks/models/qwen3p7-plus")
        with patch.object(config, "FIREWORKS_API_KEY", "test-fireworks-key"), \
             patch.object(browser_agent, "_call_openai_compatible", return_value=(None, [])) as compat, \
             patch.object(browser_agent, "_call_gemini", return_value=(None, [])) as gem:
            browser_agent._model_turn([], [])
            compat.assert_called_once()
            # Check that the model passed is the registry model
            self.assertEqual(compat.call_args.kwargs.get("model") or compat.call_args.args[4] if len(compat.call_args.args) > 4 else compat.call_args.kwargs.get("model"), "accounts/fireworks/models/qwen3p7-plus")
            gem.assert_not_called()

    def test_model_turn_gemini_when_registry_says_gemini(self):
        from backend.services import browser_agent
        from backend import config
        model_registry.set_model_for_role("browser_tool", "gemini", "gemini-2.0-flash")
        with patch.object(browser_agent, "_call_gemini", return_value=(None, [])) as gem, \
             patch.object(browser_agent, "_call_openai_compatible", return_value=(None, [])) as compat:
            browser_agent._model_turn([], [])
            gem.assert_called_once()
            self.assertEqual(gem.call_args.kwargs.get("model") or (gem.call_args.args[2] if len(gem.call_args.args) > 2 else None), "gemini-2.0-flash")
            compat.assert_not_called()

    def test_model_turn_degrades_to_env_default_on_registry_corruption(self):
        from backend.services import browser_agent
        from backend import config
        self._settings_path.write_text("{not json", encoding="utf-8")
        with patch.object(config, "BROWSER_AGENT_PROVIDER", "fireworks"), \
             patch.object(config, "BROWSER_AGENT_MODEL", "accounts/fireworks/models/qwen3p7-plus"), \
             patch.object(config, "FIREWORKS_API_KEY", "k"), \
             patch.object(browser_agent, "_call_openai_compatible", return_value=(None, [])) as compat:
            browser_agent._model_turn([], [])
            compat.assert_called_once()
            # Should have been called with env default model (degraded)
            passed_model = compat.call_args.kwargs.get("model") or (compat.call_args.args[4] if len(compat.call_args.args) > 4 else None)
            self.assertEqual(passed_model, "accounts/fireworks/models/qwen3p7-plus")

    def test_browser_custom_provider_via_registry(self):
        from backend.services import browser_agent
        from backend import config
        with patch.object(model_registry, "_list_openai_compat_models", return_value=[{"id": "m", "display": "m"}]):
            model_registry.add_custom_provider("acme", "Acme", "sk-acme", "https://acme.example/v1")
        model_registry.set_model_for_role("browser_tool", "acme", "acme-browser-model")
        with patch.object(browser_agent, "_call_openai_compatible", return_value=(None, [])) as compat:
            browser_agent._model_turn([], [])
            compat.assert_called_once()
            # Should have used custom base_url
            self.assertIn("acme.example", compat.call_args.args[0])
            # model should be the registry model
            passed_model = compat.call_args.kwargs.get("model") or (compat.call_args.args[4] if len(compat.call_args.args) > 4 else None)
            self.assertEqual(passed_model, "acme-browser-model")
