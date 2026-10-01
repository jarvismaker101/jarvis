"""F56 — LOCAL models as first-class selections (chat + intent classifier).

What this locks down:

* the local Ollama server is a registered, KEY-LESS env provider whose calls
  ride the generic OpenAI-compatible adapter (so brain.py needed no new
  branch), and it is allowed for exactly the roles it can serve — chat and
  the intent classifier. It is NOT allowed for vision/audio/planner.
* a local TEXT model can never be selected to answer a screen question: the
  provider floor carries no vision_input and no model-name rule grants it.
* thinking is switched OFF for a local endpoint inside the client, because
  Ollama's OpenAI-compatible surface ignores the native ``think`` field and
  a local thinking model otherwise spends seconds of wall clock before its
  first answer token. A model that rejects the control is replayed without it.
* the intent classifier's hop 1 is the UI's selection and the shipped cloud
  chain stays as the fallback — including "do not retry the same provider
  twice inside one budget", which is what keeps an untouched install
  behaviourally identical to before this feature existed.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from backend.api import routes
from backend.services import intent as intent_mod
from backend.services import model_registry
from backend.services import openai_compat_client as occ

LOCAL_BASE = model_registry.OLLAMA_OPENAI_BASE_URL


def _json_response(payload, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    resp.text = json.dumps(payload)
    return resp


def _chat_completion(content, reasoning=""):
    message = {"content": content}
    if reasoning:
        message["reasoning"] = reasoning
    return {"choices": [{"message": message}]}


class _RegistryIsolation(unittest.TestCase):
    """Isolate the registry behind a temp settings file and fake env keys."""

    KEYS = ("GEMINI_API_KEY", "FIREWORKS_API_KEY", "GROQ_API_KEY",
            "FISH_API_KEY", "OPENROUTER_API_KEY")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_settings_file = model_registry.SETTINGS_FILE
        model_registry.SETTINGS_FILE = Path(self._tmp.name) / "jarvis_settings.json"
        self._orig_keys = {k: getattr(model_registry, k) for k in self.KEYS}
        for key in self.KEYS:
            setattr(model_registry, key, "test-%s" % key.lower())
        model_registry._forget_cached_settings_unlocked()

    def tearDown(self):
        model_registry.SETTINGS_FILE = self._orig_settings_file
        for key, value in self._orig_keys.items():
            setattr(model_registry, key, value)
        model_registry._forget_cached_settings_unlocked()
        self._tmp.cleanup()

    def _stored_settings(self):
        with open(model_registry.SETTINGS_FILE, encoding="utf-8") as fh:
            return json.load(fh)


class OllamaProviderTests(_RegistryIsolation):
    def test_local_server_is_registered_keyless(self):
        self.assertIn("ollama", model_registry.ENV_PROVIDERS)
        api_key, base_url = model_registry.get_provider_credentials("ollama")
        self.assertTrue(api_key, "the generic adapter needs a truthy key")
        self.assertEqual(base_url, LOCAL_BASE)
        self.assertTrue(base_url.endswith(":11434/v1"))
        by_id = {p["id"]: p for p in model_registry.list_providers()}
        self.assertIn("ollama", by_id)
        self.assertEqual(by_id["ollama"]["kind"], "env")
        self.assertEqual(by_id["ollama"]["capabilities_source"], "env_adapter")

    def test_placeholder_key_is_not_a_secret_and_never_masked_out(self):
        # A placeholder that IS the credential would be a liability; this one
        # is a constant the adapter needs, so it must appear in no API surface.
        blob = json.dumps(routes.get_settings())
        self.assertNotIn("api_key", blob)
        self.assertEqual(
            model_registry.get_provider_credentials("ollama")[0],
            "ollama-local",
        )

    def test_sidecar_capabilities_have_no_vision(self):
        caps = model_registry.model_capabilities_for("ollama", "llama3.2:latest")
        self.assertIn("streaming", caps)
        self.assertIn("structured_output", caps)
        self.assertNotIn(
            "vision_input", caps,
            "a local text model must never be selectable for a screen question")

    def test_reasoning_control_is_declared_as_thinking_off(self):
        reasoning = model_registry._reasoning_for(
            "ollama", "qwen3:1.7b", frozenset())
        self.assertTrue(reasoning["supported"])
        self.assertEqual(reasoning["param"], "reasoning_effort")
        self.assertEqual(reasoning["effort"], "none")


class OllamaRoleTests(_RegistryIsolation):
    def test_chat_accepts_a_local_model(self):
        self.assertEqual(
            model_registry.set_model_for_role(
                "chat", "ollama", "llama3.2:latest"),
            {"provider": "ollama", "model": "llama3.2:latest"},
        )
        self.assertEqual(
            model_registry.get_model_for_role("chat"),
            {"provider": "ollama", "model": "llama3.2:latest"},
        )
        self.assertEqual(
            self._stored_settings()["chat_model"],
            {"provider": "ollama", "model": "llama3.2:latest"},
        )

    def test_intent_role_is_selectable_and_persisted(self):
        self.assertIn("intent", model_registry.VALID_ROLES)
        self.assertEqual(
            model_registry._ROLE_STORAGE_KEY["intent"], "intent_model")
        self.assertEqual(
            model_registry.set_model_for_role("intent", "ollama", "qwen3:1.7b"),
            {"provider": "ollama", "model": "qwen3:1.7b"},
        )
        self.assertEqual(
            self._stored_settings()["intent_model"],
            {"provider": "ollama", "model": "qwen3:1.7b"},
        )
        self.assertEqual(
            model_registry.get_model_for_role("intent"),
            {"provider": "ollama", "model": "qwen3:1.7b"},
        )

    def test_intent_requires_structured_output(self):
        self.assertEqual(
            model_registry.ROLE_CAPABILITIES["intent"],
            frozenset(("structured_output",)),
        )

    def test_intent_default_is_the_shipped_first_hop(self):
        # Nothing persisted -> the classifier must start where it always did,
        # otherwise this feature would silently reroute every message.
        self.assertEqual(
            model_registry._env_default_for_role("intent"),
            {"provider": "openrouter",
             "model": intent_mod.DEFAULT_OPENROUTER_MODEL},
        )
        self.assertEqual(
            model_registry.get_model_for_role("intent")["provider"],
            "openrouter",
        )

    def test_local_models_are_refused_for_every_other_role(self):
        for role in ("vision", "tts", "listening", "planner", "browser_tool"):
            with self.subTest(role=role):
                self.assertNotIn(
                    "ollama", model_registry.get_allowed_providers_for_role(role))
                with self.assertRaises(model_registry.ModelRegistryError):
                    model_registry.set_model_for_role(
                        role, "ollama", "llama3.2:latest")

    def test_local_models_are_not_custom_provider_capable(self):
        self.assertNotIn("intent", model_registry.roles_allowing_custom())

    def test_an_uninstalled_model_is_accepted_by_the_registry(self):
        # The registry validates CAPABILITIES, not the local disk: a model the
        # daemon does not have must fail at call time and fall back, never
        # block the selection UI (the daemon may load it a moment later).
        self.assertEqual(
            model_registry.set_model_for_role("intent", "ollama", "not-pulled"),
            {"provider": "ollama", "model": "not-pulled"},
        )


class OllamaModelListingTests(_RegistryIsolation):
    def test_listing_returns_installed_models_and_drops_an_embedder(self):
        payload = {"models": [
            {"name": "llama3.2:latest"},
            {"name": "qwen3:1.7b"},
            {"name": "nomic-embed-text:latest"},
        ]}
        with patch.object(model_registry, "_http_get_json",
                          return_value=payload) as get:
            models = model_registry.list_provider_models("ollama")
        self.assertEqual(
            [m["id"] for m in models], ["llama3.2:latest", "qwen3:1.7b"])
        self.assertIn("/api/tags", get.call_args.args[0])

    def test_listing_with_no_usable_model_is_a_clean_error(self):
        with patch.object(model_registry, "_http_get_json",
                          return_value={"models": [{"name": "nomic-embed-text"}]}):
            with self.assertRaises(model_registry.ModelRegistryError):
                model_registry.list_provider_models("ollama")

    def test_stopped_daemon_surfaces_as_a_400_not_an_empty_list(self):
        with patch.object(
            model_registry, "_http_get_json",
            side_effect=model_registry.ModelRegistryError(
                "model list request failed (network error)"),
        ):
            with self.assertRaises(HTTPException) as ctx:
                routes.get_provider_models("ollama")
        self.assertEqual(ctx.exception.status_code, 400)


class LocalThinkingControlTests(unittest.TestCase):
    def test_only_the_local_ollama_host_counts_as_local(self):
        for base in ("http://localhost:11434/v1",
                     "http://127.0.0.1:11434/v1",
                     "http://[::1]:11434/v1"):
            self.assertTrue(occ._is_local_thinking_host(base), base)
        for base in ("https://openrouter.ai/api/v1",
                     "https://api.groq.com/openai/v1",
                     "http://localhost:11435/v1",
                     "http://evil.example/localhost:11434/v1",
                     "", None):
            self.assertFalse(occ._is_local_thinking_host(base), base)

    def test_local_call_disables_thinking(self):
        with patch.object(occ, "_session") as session:
            session.post.return_value = _json_response(_chat_completion("hi"))
            result = occ.ask_openai_compat(
                [{"role": "user", "content": "hi"}], model="qwen3:1.7b",
                base_url=LOCAL_BASE, api_key="ollama-local")
        self.assertTrue(result.get("choices"))
        body = session.post.call_args.kwargs["json"]
        self.assertEqual(body["reasoning_effort"], "none")

    def test_cloud_call_carries_no_reasoning_control(self):
        with patch.object(occ, "_session") as session:
            session.post.return_value = _json_response(_chat_completion("hi"))
            occ.ask_openai_compat(
                [{"role": "user", "content": "hi"}], model="m",
                base_url="https://openrouter.ai/api/v1", api_key="k")
        body = session.post.call_args.kwargs["json"]
        self.assertNotIn("reasoning_effort", body)

    def test_an_explicit_caller_value_is_never_overridden(self):
        with patch.object(occ, "_session") as session:
            session.post.return_value = _json_response(_chat_completion("hi"))
            occ.ask_openai_compat(
                [{"role": "user", "content": "hi"}], model="qwen3:1.7b",
                base_url=LOCAL_BASE, api_key="k", reasoning_effort="high")
        body = session.post.call_args.kwargs["json"]
        self.assertEqual(body["reasoning_effort"], "high")

    def test_a_rejected_reasoning_control_is_replayed_without_it(self):
        rejected = _json_response(
            {"error": {"message": 'invalid reasoning value: "none"'}}, status=400)
        ok = _json_response(_chat_completion('{"intent":"chat"}'))
        with patch.object(occ, "_session") as session:
            session.post.side_effect = [rejected, ok]
            result = occ.ask_openai_compat(
                [{"role": "user", "content": "hi"}], model="qwen3:1.7b",
                base_url=LOCAL_BASE, api_key="k")
        self.assertTrue(result.get("choices"))
        self.assertEqual(session.post.call_count, 2)
        self.assertNotIn("reasoning_effort", session.post.call_args.kwargs["json"])

    def test_a_rejection_unrelated_to_reasoning_is_not_replayed(self):
        rejected = _json_response({"error": {"message": "model not found"}},
                                  status=404)
        with patch.object(occ, "_session") as session:
            session.post.return_value = rejected
            result = occ.ask_openai_compat(
                [{"role": "user", "content": "hi"}], model="nope",
                base_url=LOCAL_BASE, api_key="k")
        self.assertEqual(result, {})
        self.assertEqual(session.post.call_count, 1)

    def test_local_stream_disables_thinking(self):
        class _Stream:
            status_code = 200
            encoding = "utf-8"

            def iter_lines(self, decode_unicode=True):
                yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
                yield "data: [DONE]"

            def close(self):
                pass

        with patch.object(occ, "_session") as session:
            session.post.return_value = _Stream()
            deltas = list(occ.ask_openai_compat_stream(
                [{"role": "user", "content": "hi"}], model="qwen3:1.7b",
                base_url=LOCAL_BASE, api_key="k"))
        self.assertEqual(deltas, ["hi"])
        body = session.post.call_args.kwargs["json"]
        self.assertEqual(body["reasoning_effort"], "none")

    def test_local_stream_replays_without_the_rejected_control(self):
        class _Stream:
            status_code = 200
            encoding = "utf-8"

            def iter_lines(self, decode_unicode=True):
                yield 'data: {"choices":[{"delta":{"content":"ok"}}]}'
                yield "data: [DONE]"

            def close(self):
                pass

        rejected = MagicMock()
        rejected.status_code = 400
        rejected.text = 'invalid reasoning value: "none"'
        with patch.object(occ, "_session") as session:
            session.post.side_effect = [rejected, _Stream()]
            deltas = list(occ.ask_openai_compat_stream(
                [{"role": "user", "content": "hi"}], model="qwen3:1.7b",
                base_url=LOCAL_BASE, api_key="k"))
        self.assertEqual(deltas, ["ok"])
        self.assertEqual(session.post.call_count, 2)
        self.assertNotIn("reasoning_effort",
                         session.post.call_args.kwargs["json"])


class IntentClassifierSelectionTests(_RegistryIsolation):
    def setUp(self):
        super().setUp()
        # The classifier's own module reads the same patched settings file.
        self._orig_intent_settings = intent_mod.model_registry.SETTINGS_FILE
        intent_mod.model_registry.SETTINGS_FILE = model_registry.SETTINGS_FILE

    def tearDown(self):
        intent_mod.model_registry.SETTINGS_FILE = self._orig_intent_settings
        super().tearDown()

    def test_selected_local_model_is_hop_one_and_names_the_source(self):
        model_registry.set_model_for_role("intent", "ollama", "qwen3:1.7b")
        verdict = '{"intent": "tool", "steps": [{"action": "open_website", "input": "youtube.com"}]}'
        with patch.object(intent_mod, "_classify_with_openai_compat",
                          return_value=verdict) as local, \
             patch.object(intent_mod, "_classify_with_openrouter") as orr, \
             patch.object(intent_mod, "_classify_with_gemini") as gem, \
             patch.object(intent_mod, "_classify_with_groq") as groq:
            result = intent_mod.classify_intent("open youtube")
        self.assertEqual(result["intent"], "tool")
        self.assertEqual(result["_source"], "ollama")
        local.assert_called_once()
        orr.assert_not_called()
        gem.assert_not_called()
        groq.assert_not_called()

    def test_a_failing_local_hop_falls_back_to_the_shipped_chain(self):
        model_registry.set_model_for_role("intent", "ollama", "qwen3:1.7b")
        with patch.object(intent_mod, "_classify_with_openai_compat",
                          return_value="") as local, \
             patch.object(intent_mod, "_classify_with_openrouter",
                          return_value='{"intent": "research", "query": "x"}') as orr:
            result = intent_mod.classify_intent("deepseek kya hai, dhundho")
        self.assertEqual(result["intent"], "research")
        self.assertEqual(result["_source"], "openrouter")
        local.assert_called_once()
        orr.assert_called_once()

    def test_selecting_a_provider_the_chain_also_uses_does_not_retry_it(self):
        model_registry.set_model_for_role(
            "intent", "openrouter", "google/gemini-2.5-flash-lite")
        with patch.object(intent_mod, "_classify_with_openrouter",
                          return_value="") as orr, \
             patch.object(intent_mod, "_classify_with_gemini",
                          return_value='{"intent": "chat"}') as gem:
            result = intent_mod.classify_intent("hello")
        self.assertEqual(result["_source"], "gemini")
        orr.assert_called_once()
        gem.assert_called_once()

    def test_no_selection_keeps_the_shipped_order(self):
        with patch.object(intent_mod, "_classify_with_openrouter",
                          return_value='{"intent": "chat"}') as orr, \
             patch.object(intent_mod, "_classify_with_gemini") as gem:
            result = intent_mod.classify_intent("hello")
        self.assertEqual(result["_source"], "openrouter")
        orr.assert_called_once()
        gem.assert_not_called()

    def test_an_unusable_selection_never_breaks_routing(self):
        # A persisted selection whose provider is unknown/unregistered must be
        # dropped by the registry, not swallowed into a chat verdict.
        with open(model_registry.SETTINGS_FILE, "w", encoding="utf-8") as fh:
            json.dump({"intent_model": {"provider": "ghost", "model": "x"}}, fh)
        model_registry._forget_cached_settings_unlocked()
        self.assertIsNone(intent_mod._selected_intent_model())
        with patch.object(intent_mod, "_classify_with_openrouter",
                          return_value='{"intent": "screen"}'):
            self.assertEqual(
                intent_mod.classify_intent("what is on my screen")["intent"],
                "screen")

    def test_the_local_hop_asks_for_json_and_uses_the_local_endpoint(self):
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            return _json_response(_chat_completion('{"intent":"chat"}'))

        with patch.object(occ, "_single_attempt_session") as session:
            session.post.side_effect = fake_post
            content = intent_mod._classify_with_openai_compat(
                "ollama", "qwen3:1.7b", "hello", (1, 1))
        self.assertIn("intent", content)
        self.assertEqual(captured["url"], LOCAL_BASE + "/chat/completions")
        self.assertEqual(captured["json"]["response_format"],
                         {"type": "json_object"})
        self.assertEqual(captured["json"]["reasoning_effort"], "none")

    def test_the_local_hop_uses_the_single_attempt_session(self):
        # A classifier hop owns ONE slice of a shared deadline: a read timeout
        # must end the attempt instead of being replayed under a spent budget.
        with patch.object(occ, "_single_attempt_session") as once, \
             patch.object(occ, "_session") as retrying:
            once.post.return_value = _json_response(
                _chat_completion('{"intent":"chat"}'))
            intent_mod._classify_with_openai_compat(
                "ollama", "qwen3:1.7b", "hello", (1, 1))
        once.post.assert_called_once()
        retrying.post.assert_not_called()

    def test_the_budget_hop_never_replays_a_read_timeout(self):
        import requests as _requests

        with patch.object(occ, "_single_attempt_session") as once:
            once.post.side_effect = _requests.exceptions.ReadTimeout("late")
            result = occ.ask_openai_compat(
                [{"role": "user", "content": "hi"}], model="m",
                base_url=LOCAL_BASE, api_key="k", timeout=(1, 1),
                single_attempt=True)
        self.assertEqual(result, {})
        self.assertEqual(once.post.call_count, 1)

    def test_default_callers_keep_the_retrying_session(self):
        with patch.object(occ, "_single_attempt_session") as once, \
             patch.object(occ, "_session") as retrying:
            retrying.post.return_value = _json_response(_chat_completion("hi"))
            occ.ask_openai_compat(
                [{"role": "user", "content": "hi"}], model="m",
                base_url="https://gateway.test/v1", api_key="k")
        retrying.post.assert_called_once()
        once.post.assert_not_called()


class IntentRoleRouteTests(_RegistryIsolation):
    def test_get_settings_exposes_the_intent_role(self):
        resp = routes.get_settings()
        self.assertIn("intent_model", resp)
        self.assertEqual(resp["intent_model"],
                         model_registry.get_model_for_role("intent"))
        self.assertIn("ollama", resp["role_allowed"]["intent"])
        self.assertNotIn("ollama", resp["role_allowed"]["vision"])

    def test_post_settings_model_switches_the_classifier(self):
        resp = routes.set_model(
            routes.ModelUpdate(role="intent", provider="ollama",
                               model="qwen3:1.7b"))
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["intent_model"],
                         {"provider": "ollama", "model": "qwen3:1.7b"})
        self.assertEqual(
            routes.get_settings()["intent_model"],
            {"provider": "ollama", "model": "qwen3:1.7b"})

    def test_a_role_incompatible_provider_is_refused(self):
        with self.assertRaises(HTTPException) as ctx:
            routes.set_model(routes.ModelUpdate(
                role="intent", provider="fish", model="s1"))
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
