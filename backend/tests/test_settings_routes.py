import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from backend.api import routes
from backend.services import model_registry


class SettingsRoutesTestBase(unittest.TestCase):
    """Isolate the registry behind the routes: temp settings file +
    deterministic env keys, so no test ever touches real keys or network."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._settings_path = Path(self._tmp.name) / "jarvis_settings.json"
        self._orig_settings_file = model_registry.SETTINGS_FILE
        self._orig_gemini_key = model_registry.GEMINI_API_KEY
        self._orig_fireworks_key = model_registry.FIREWORKS_API_KEY
        self._orig_groq_key = model_registry.GROQ_API_KEY
        self._orig_fish_key = model_registry.FISH_API_KEY
        self._orig_openrouter_key = model_registry.OPENROUTER_API_KEY
        model_registry.SETTINGS_FILE = self._settings_path
        model_registry.GEMINI_API_KEY = "test-gemini-key"
        model_registry.FIREWORKS_API_KEY = "test-fireworks-key"
        model_registry.GROQ_API_KEY = "test-groq-key"
        model_registry.FISH_API_KEY = "test-fish-key"
        model_registry.OPENROUTER_API_KEY = "test-openrouter-key"

    def tearDown(self):
        model_registry.SETTINGS_FILE = self._orig_settings_file
        model_registry.GEMINI_API_KEY = self._orig_gemini_key
        model_registry.FIREWORKS_API_KEY = self._orig_fireworks_key
        model_registry.GROQ_API_KEY = self._orig_groq_key
        model_registry.FISH_API_KEY = self._orig_fish_key
        model_registry.OPENROUTER_API_KEY = self._orig_openrouter_key
        self._tmp.cleanup()


class SettingsRoutesTests(SettingsRoutesTestBase):
    def test_get_settings_defaults_and_masks_keys(self):
        resp = routes.get_settings()
        self.assertEqual(resp["chat_model"], model_registry.get_default_chat_model())
        self.assertEqual(
            [p["id"] for p in resp["providers"]],
            ["gemini", "fireworks", "groq", "fish", "gtts", "openrouter",
             "whisper", "inworld", "ollama"],
        )
        self.assertIn("role_allowed", resp)
        self.assertIn("last_fallback", resp)
        # new roles present and match registry defaults
        self.assertEqual(resp["tts_model"], model_registry.get_model_for_role("tts"))
        self.assertEqual(resp["vision_model"], model_registry.get_model_for_role("vision"))
        self.assertEqual(resp["browser_tool_model"], model_registry.get_model_for_role("browser_tool"))
        self.assertEqual(resp["listening_model"], model_registry.get_model_for_role("listening"))
        # listening role exposed in the allowed map with both STT engines
        self.assertEqual(
            sorted(resp["role_allowed"]["listening"]), ["inworld", "whisper"]
        )
        blob = json.dumps(resp)
        self.assertNotIn("test-gemini-key", blob)
        self.assertNotIn("test-fireworks-key", blob)
        self.assertNotIn("api_key", blob)

    def test_post_chat_model_roundtrip(self):
        payload = routes.ChatModelUpdate(provider="gemini", model="gemini-2.5-pro")
        resp = routes.set_chat_model(payload)
        self.assertTrue(resp["ok"])
        self.assertEqual(
            resp["chat_model"], {"provider": "gemini", "model": "gemini-2.5-pro"}
        )
        # reflected by GET /settings and the registry (immediate effect)
        self.assertEqual(
            routes.get_settings()["chat_model"]["model"], "gemini-2.5-pro"
        )
        self.assertEqual(
            model_registry.get_default_chat_model()["model"], "gemini-2.5-pro"
        )

    def test_post_chat_model_invalid_provider_400(self):
        payload = routes.ChatModelUpdate(provider="grok", model="x")
        with self.assertRaises(HTTPException) as ctx:
            routes.set_chat_model(payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("grok", ctx.exception.detail)

    def test_post_chat_model_custom_provider_after_add(self):
        with patch.object(
            model_registry,
            "_list_openai_compat_models",
            return_value=[{"id": "m", "display": "m"}],
        ):
            routes.add_provider(
                routes.ProviderAddRequest(
                    id="acme", name="Acme", api_key="sk-1",
                    base_url="https://acme.example/v1",
                )
            )
        resp = routes.set_chat_model(
            routes.ChatModelUpdate(provider="acme", model="acme-chat")
        )
        self.assertTrue(resp["ok"])
        self.assertEqual(
            model_registry.get_default_chat_model()["provider"], "acme"
        )


class ProviderModelsRouteTests(SettingsRoutesTestBase):
    @staticmethod
    def _fake_response(payload, status=200):
        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = payload
        resp.text = json.dumps(payload)
        return resp

    def test_models_endpoint_with_mocked_http(self):
        payload = {
            "models": [
                {
                    "name": "models/gemini-2.5-pro",
                    "displayName": "Gemini 2.5 Pro",
                    "supportedGenerationMethods": ["generateContent"],
                },
            ]
        }
        with patch.object(model_registry, "_session") as sess:
            sess.get.return_value = self._fake_response(payload)
            resp = routes.get_provider_models("gemini")
        self.assertEqual(
            resp["models"],
            [{"id": "gemini-2.5-pro", "display": "Gemini 2.5 Pro"}],
        )

    def test_models_endpoint_unknown_provider_400(self):
        with self.assertRaises(HTTPException) as ctx:
            routes.get_provider_models("nope")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_models_endpoint_missing_key_400(self):
        model_registry.GEMINI_API_KEY = None
        with self.assertRaises(HTTPException) as ctx:
            routes.get_provider_models("gemini")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Gemini API key", ctx.exception.detail)


class AddProviderRouteTests(SettingsRoutesTestBase):
    def test_add_provider_success_and_appears_in_settings(self):
        with patch.object(
            model_registry,
            "_list_openai_compat_models",
            return_value=[{"id": "m", "display": "m"}],
        ) as probe:
            resp = routes.add_provider(
                routes.ProviderAddRequest(
                    id="acme", name="Acme", api_key="sk-live-1",
                    base_url="https://acme.example/v1",
                )
            )
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["provider"]["id"], "acme")
        self.assertTrue(resp["provider"]["has_key"])
        self.assertNotIn("api_key", json.dumps(resp))
        self.assertNotIn("sk-live-1", json.dumps(resp))
        probe.assert_called_once_with("https://acme.example/v1", "sk-live-1")
        # auto-listed by GET /settings
        ids = [p["id"] for p in routes.get_settings()["providers"]]
        self.assertIn("acme", ids)

    def test_add_provider_invalid_key_400_and_not_stored(self):
        def boom(url, api_key):
            raise model_registry.ModelRegistryError(
                "model list request failed (HTTP 401)"
            )

        with patch.object(
            model_registry, "_list_openai_compat_models", side_effect=boom
        ):
            with self.assertRaises(HTTPException) as ctx:
                routes.add_provider(
                    routes.ProviderAddRequest(
                        id="acme", name="Acme", api_key="sk-bad",
                        base_url="https://acme.example/v1",
                    )
                )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("401", ctx.exception.detail)
        # nothing persisted
        self.assertEqual(model_registry.list_custom_providers(), [])
        ids = [p["id"] for p in routes.get_settings()["providers"]]
        self.assertNotIn("acme", ids)

    def test_add_provider_duplicate_400(self):
        with patch.object(
            model_registry,
            "_list_openai_compat_models",
            return_value=[{"id": "m", "display": "m"}],
        ):
            routes.add_provider(
                routes.ProviderAddRequest(
                    id="acme", name="Acme", api_key="sk-1",
                    base_url="https://acme.example/v1",
                )
            )
        with self.assertRaises(HTTPException) as ctx:
            routes.add_provider(
                routes.ProviderAddRequest(
                    id="acme", name="Acme Again", api_key="sk-2",
                    base_url="https://acme.example/v1",
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("already exists", ctx.exception.detail)


class RoleRoutesTests(SettingsRoutesTestBase):
    def test_generic_model_roundtrip_per_role(self):
        for role in ["tts", "vision", "browser_tool", "listening"]:
            # pick a valid provider for each role (fish for tts, gemini for vision, fireworks for browser, whisper for listening)
            prov = {"tts": "fish", "vision": "gemini", "browser_tool": "fireworks", "listening": "whisper"}[role]
            model = f"test-{role}-model-1"
            payload = routes.ModelUpdate(role=role, provider=prov, model=model)
            resp = routes.set_model(payload)
            self.assertTrue(resp["ok"])
            self.assertEqual(resp["role"], role)
            # reflected in GET /settings
            settings = routes.get_settings()
            key = {"tts": "tts_model", "vision": "vision_model", "browser_tool": "browser_tool_model", "listening": "listening_model"}[role]
            self.assertEqual(settings[key], {"provider": prov, "model": model})
            self.assertEqual(model_registry.get_model_for_role(role), {"provider": prov, "model": model})

    def test_listening_model_post_returns_role_key_and_masks_inworld_key(self):
        with patch.dict(os.environ, {"INWORLD_STT_API_KEY": "test-inworld-key"}):
            payload = routes.ModelUpdate(
                role="listening", provider="inworld", model="inworld/inworld-stt-1"
            )
            resp = routes.set_model(payload)
            self.assertTrue(resp["ok"])
            self.assertEqual(resp["role"], "listening")
            self.assertEqual(
                resp["listening_model"],
                {"provider": "inworld", "model": "inworld/inworld-stt-1"},
            )
            blob = json.dumps(routes.get_settings())
        self.assertNotIn("test-inworld-key", blob)
        self.assertNotIn("api_key", blob)
        self.assertEqual(
            model_registry.get_model_for_role("listening")["provider"], "inworld"
        )

    def test_generic_model_unknown_role_400(self):
        payload = routes.ModelUpdate(role="unknown_role", provider="gemini", model="m")
        with self.assertRaises(HTTPException) as ctx:
            routes.set_model(payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("unknown role", ctx.exception.detail.lower())

    def test_generic_model_unknown_provider_400(self):
        payload = routes.ModelUpdate(role="tts", provider="nope_provider", model="m")
        with self.assertRaises(HTTPException) as ctx:
            routes.set_model(payload)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("unknown provider", ctx.exception.detail.lower())

    def test_generic_model_masks_keys(self):
        payload = routes.ModelUpdate(role="vision", provider="gemini", model="gemini-2.5-flash")
        routes.set_model(payload)
        blob = json.dumps(routes.get_settings())
        self.assertNotIn("test-gemini-key", blob)
        self.assertNotIn("test-fish-key", blob)
        self.assertNotIn("api_key", blob)

    def test_fish_models_endpoint_via_routes(self):
        resp = routes.get_provider_models("fish")
        self.assertIn("models", resp)
        ids = [m["id"] for m in resp["models"]]
        self.assertIn("s2.1-pro-free", ids)

    def test_openrouter_models_endpoint_via_routes(self):
        payload = {"data": [{"id": "google/gemma-4-31b-it:free", "architecture": {"input_modalities": ["text", "image"]}}]}
        fake = MagicMock()
        fake.status_code = 200
        fake.json.return_value = payload
        fake.text = json.dumps(payload)
        with patch.object(model_registry, "_session") as sess:
            sess.get.return_value = fake
            resp = routes.get_provider_models("openrouter")
        self.assertEqual(resp["models"][0]["id"], "google/gemma-4-31b-it:free")


if __name__ == "__main__":
    unittest.main()
