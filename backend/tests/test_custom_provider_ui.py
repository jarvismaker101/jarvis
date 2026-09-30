"""Custom-provider UI feature: TEST endpoint, per-functionality roles and
the generic OpenAI-compatible vision adapter.

The feature the tests pin:
  * POST /settings/provider/test live-checks a (base_url, api_key) pair and
    persists NOTHING — the add-provider form's TEST button must never store
    a key that has not been saved deliberately;
  * a user-added OpenAI-compatible provider may serve every LLM-backed
    functionality (chat / vision / browser_tool / planner) but never a voice
    role (tts / listening drive dedicated audio engines);
  * a custom provider participates in the F37 vision cascade through ONE
    generic image-part adapter, at every vision call site.

No network and no real keys anywhere: the registry sits behind a temp
settings file and the model-list probe is mocked.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from backend.api import routes
from backend.services import model_registry
from backend.services import openai_compat_client
from backend.services import screen_analyzer
from backend.services import screen_control
from backend.services import vision_cascade


def _add_custom_provider(pid="acme", name="Acme", api_key="sk-test-1",
                         base_url="https://acme.example/v1"):
    with patch.object(
        model_registry,
        "_list_openai_compat_models",
        return_value=[{"id": "m", "display": "m"}],
    ):
        return routes.add_provider(
            routes.ProviderAddRequest(
                id=pid, name=name, api_key=api_key, base_url=base_url,
            )
        )


class CustomProviderTestBase(unittest.TestCase):
    """Temp settings file + deterministic env keys (mirrors
    test_settings_routes): no test touches real keys or the network."""

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


class ProviderTestEndpointTests(CustomProviderTestBase):
    """POST /settings/provider/test — the TEST button."""

    def test_success_lists_models_and_persists_nothing(self):
        with patch.object(
            model_registry,
            "_list_openai_compat_models",
            return_value=[{"id": "m1", "display": "Model One"},
                          {"id": "m2", "display": "Model Two"}],
        ) as probe:
            resp = routes.test_provider(
                routes.ProviderTestRequest(
                    api_key="sk-candidate", base_url="https://gw.example/v1")
            )
        probe.assert_called_once_with("https://gw.example/v1", "sk-candidate")
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["count"], 2)
        self.assertEqual(resp["models"][0]["id"], "m1")
        # a TEST must never write: no provider record, no settings file
        self.assertEqual(model_registry.list_custom_providers(), [])
        self.assertFalse(self._settings_path.exists())

    def test_failure_is_scrubbed_400_and_persists_nothing(self):
        def boom(url, api_key):
            raise model_registry.ModelRegistryError(
                "model list request failed (HTTP 401) api_key=sk-secret-123456"
            )

        with patch.object(
            model_registry, "_list_openai_compat_models", side_effect=boom
        ):
            with self.assertRaises(HTTPException) as ctx:
                routes.test_provider(
                    routes.ProviderTestRequest(
                        api_key="sk-secret-123456",
                        base_url="https://gw.example/v1",
                    )
                )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertNotIn("sk-secret-123456", ctx.exception.detail)
        self.assertIn("REDACTED", ctx.exception.detail)
        self.assertEqual(model_registry.list_custom_providers(), [])
        self.assertFalse(self._settings_path.exists())

    def test_missing_key_or_bad_url_is_rejected(self):
        for api_key, base_url in (
            ("", "https://gw.example/v1"),
            ("sk-x", "ftp://gw.example/v1"),
            ("sk-x", ""),
        ):
            with self.assertRaises(HTTPException) as ctx:
                routes.test_provider(
                    routes.ProviderTestRequest(
                        api_key=api_key, base_url=base_url)
                )
            self.assertEqual(ctx.exception.status_code, 400)

    def test_no_models_is_an_explicit_error(self):
        with patch.object(
            model_registry, "_list_openai_compat_models", return_value=[]
        ):
            with self.assertRaises(HTTPException) as ctx:
                routes.test_provider(
                    routes.ProviderTestRequest(
                        api_key="sk-x", base_url="https://gw.example/v1")
                )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("no models", ctx.exception.detail)


class CustomProviderRoleScopeTests(CustomProviderTestBase):
    """Which functionalities accept a user-added provider."""

    def test_roles_allowing_custom_covers_the_llm_functions(self):
        self.assertEqual(
            model_registry.roles_allowing_custom(),
            ["browser_tool", "chat", "planner", "vision"],
        )
        resp = routes.get_settings()
        self.assertEqual(
            resp["custom_provider_roles"],
            ["browser_tool", "chat", "planner", "vision"],
        )

    def test_custom_provider_serves_vision_and_planner(self):
        _add_custom_provider()
        for role in ("vision", "planner"):
            resp = routes.set_model(
                routes.ModelUpdate(
                    role=role, provider="acme", model="acme-vision-1")
            )
            self.assertTrue(resp["ok"])
            self.assertEqual(
                model_registry.get_model_for_role(role)["provider"], "acme")

    def test_custom_provider_still_serves_chat_and_browser_tool(self):
        _add_custom_provider()
        for role in ("chat", "browser_tool"):
            resp = routes.set_model(
                routes.ModelUpdate(role=role, provider="acme", model="m")
            )
            self.assertTrue(resp["ok"])

    def test_custom_provider_is_refused_for_voice_roles(self):
        _add_custom_provider()
        for role in ("tts", "listening"):
            with self.assertRaises(HTTPException) as ctx:
                routes.set_model(
                    routes.ModelUpdate(role=role, provider="acme", model="m")
                )
            self.assertEqual(ctx.exception.status_code, 400)
            self.assertIn("not allowed", ctx.exception.detail)


class OpenAICompatVisionAdapterTests(CustomProviderTestBase):
    """ask_openai_compat_vision — the generic image-part adapter."""

    @staticmethod
    def _fake_response(payload, status=200):
        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = payload
        resp.text = json.dumps(payload)
        return resp

    def test_payload_carries_the_image_part_and_auth(self):
        payload = {"choices": [{"message": {"content": "{}"}}]}
        with patch.object(openai_compat_client, "_session") as sess:
            sess.post.return_value = self._fake_response(payload)
            result = openai_compat_client.ask_openai_compat_vision(
                "what is on screen", "data:image/png;base64,AAAA",
                "acme-vision-1", "https://gw.example/v1", "sk-x",
                max_completion_tokens=800,
            )
        self.assertEqual(result, payload)
        args, kwargs = sess.post.call_args
        self.assertEqual(
            args[0], "https://gw.example/v1/chat/completions")
        self.assertEqual(
            kwargs["headers"]["Authorization"], "Bearer sk-x")
        body = kwargs["json"]
        self.assertEqual(body["model"], "acme-vision-1")
        self.assertEqual(body["max_tokens"], 800)
        content = body["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text",
                                      "text": "what is on screen"})
        self.assertEqual(
            content[1], {"type": "image_url",
                         "image_url": {"url": "data:image/png;base64,AAAA"}})
        # no response_format passed -> the gateway never sees the field
        self.assertNotIn("response_format", body)

    def test_response_format_rides_only_when_given(self):
        with patch.object(openai_compat_client, "_session") as sess:
            sess.post.return_value = self._fake_response({})
            openai_compat_client.ask_openai_compat_vision(
                "p", "data:image/png;base64,AAAA", "m",
                "https://gw.example/v1", "sk-x",
                response_format={"type": "json_object"},
            )
        body = sess.post.call_args[1]["json"]
        self.assertEqual(
            body["response_format"], {"type": "json_object"})


class CustomVisionCascadeTests(CustomProviderTestBase):
    """A custom provider is one more F37 cascade dispatcher."""

    def test_dispatchers_cover_registered_custom_providers(self):
        self.assertEqual(vision_cascade.custom_vision_dispatchers(), {})
        _add_custom_provider()
        dispatchers = vision_cascade.custom_vision_dispatchers()
        self.assertIn("acme", dispatchers)
        # call sites merge the same map
        self.assertIn("acme", screen_control._vision_dispatchers())
        self.assertIn("acme", screen_analyzer._vision_dispatchers())

    def test_dispatcher_resolves_credentials_and_calls_the_adapter(self):
        _add_custom_provider()
        dispatch = vision_cascade.custom_vision_dispatchers()["acme"]
        with patch.object(
            model_registry, "get_provider_credentials",
            return_value=("sk-stored", "https://acme.example/v1"),
        ), patch.object(
            openai_compat_client, "ask_openai_compat_vision",
            return_value={"choices": []},
        ) as adapter:
            result = dispatch("describe", "data:image/png;base64,AAAA",
                              "acme-vision-1", 800, {"type": "json_object"})
        self.assertEqual(result, {"choices": []})
        adapter.assert_called_once_with(
            "describe", "data:image/png;base64,AAAA",
            "acme-vision-1", "https://acme.example/v1", "sk-stored",
            max_completion_tokens=800,
            response_format={"type": "json_object"},
        )

    def test_dispatcher_without_credentials_dispatches_nothing(self):
        _add_custom_provider()
        dispatch = vision_cascade.custom_vision_dispatchers()["acme"]
        with patch.object(
            model_registry, "get_provider_credentials",
            return_value=(None, None),
        ), patch.object(
            openai_compat_client, "ask_openai_compat_vision"
        ) as adapter:
            self.assertEqual(
                dispatch("describe", "data:image/png;base64,AAAA",
                         "acme-vision-1", 800, None),
                {})
        adapter.assert_not_called()

    def test_custom_provider_is_eligible_when_selected(self):
        _add_custom_provider()
        self.assertTrue(vision_cascade.provider_available("acme"))
        ok, reason = vision_cascade.provider_eligible("acme")
        self.assertTrue(ok, reason)
        eligible, skipped = vision_cascade.provider_candidates(
            selected={"provider": "acme", "model": "acme-vision-1"})
        self.assertEqual(eligible[0]["provider"], "acme")
        self.assertEqual(eligible[0]["model"], "acme-vision-1")
        self.assertFalse(any(s["provider"] == "acme" for s in skipped))

    def test_unregistered_provider_is_never_available(self):
        self.assertFalse(vision_cascade.provider_available("acme"))
        ok, reason = vision_cascade.provider_eligible("acme")
        self.assertFalse(ok)
        self.assertIn("not allowed", reason)


if __name__ == "__main__":
    unittest.main()
