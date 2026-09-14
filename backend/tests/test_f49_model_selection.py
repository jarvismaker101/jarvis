"""F49 — make model selection capability-aware.

Acceptance (audit report): "Text-only browser/non-tool planner selections
fail; invalid persisted settings cannot run; missing private credentials send
nothing elsewhere; in-flight configuration remains consistent."

The registry is the single source of truth for which (provider, model) may
serve a role. These tests exercise the real code paths: the shipped
capability tables and model rules, the provider-metadata observation written
by list_provider_models, use-time validation of persisted settings, the
atomic per-call snapshot, and the orchestrator's planner call. Only true
externals (HTTP model lists, the OpenAI-compatible client) are mocked.
"""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend import config
from backend.services import model_registry
from backend.services import orchestrator
from backend.services.gemini_client import GEMINI_MODEL

FIREWORKS_VISION_MODEL = "accounts/fireworks/models/deepseek-v4-flash-vision-exp"
TEXT_ONLY_LLAMA = "accounts/fireworks/models/llama-v3p1-8b-instruct"
GUARD_MODEL = "accounts/fireworks/models/llama-guard-3-8b"
EMBEDDING_MODEL = "accounts/fireworks/models/text-embedding-3-large"
DEFAULT_BROWSER_MODEL = "accounts/fireworks/models/qwen3p7-plus"


class RegistryTestBase(unittest.TestCase):
    """Isolate the registry: temp settings file + deterministic env keys."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._settings_path = Path(self._tmp.name) / "jarvis_settings.json"
        self._saved = {
            name: getattr(model_registry, name) for name in (
                "SETTINGS_FILE", "GEMINI_API_KEY", "FIREWORKS_API_KEY",
                "GROQ_API_KEY", "FISH_API_KEY", "OPENROUTER_API_KEY")
        }
        model_registry.SETTINGS_FILE = self._settings_path
        model_registry.GEMINI_API_KEY = "test-gemini-key"
        model_registry.FIREWORKS_API_KEY = "test-fireworks-key"
        model_registry.GROQ_API_KEY = "test-groq-key"
        model_registry.FISH_API_KEY = "test-fish-key"
        model_registry.OPENROUTER_API_KEY = "test-openrouter-key"

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(model_registry, name, value)
        self._tmp.cleanup()

    # ── helpers ────────────────────────────────────────────────────────────
    def _write_settings(self, data):
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        self._settings_path.write_text(json.dumps(data), encoding="utf-8")

    def _read_settings(self):
        with open(self._settings_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _add_provider(self, pid="acme", capabilities=None,
                      base_url="https://acme.example/v1", api_key="sk-acme"):
        with patch.object(
            model_registry, "_list_openai_compat_models",
            return_value=[{"id": "m", "display": "m"}],
        ):
            return model_registry.add_custom_provider(
                pid, pid.upper(), api_key, base_url, capabilities=capabilities)

    def _list_provider(self, provider_id, payload):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = payload
        response.text = json.dumps(payload)
        with patch.object(model_registry, "_session") as session:
            session.get.return_value = response
            return model_registry.list_provider_models(provider_id)


class BrowserVisionRequirementTests(RegistryTestBase):
    """A text-only model must not be selectable for the browser role."""

    def test_browser_role_requires_vision(self):
        self.assertIn("vision_input",
                      model_registry.ROLE_CAPABILITIES["browser_tool"])

    def test_text_only_model_rejected_for_browser_selection(self):
        with self.assertRaises(model_registry.ModelRegistryError) as ctx:
            model_registry.set_model_for_role(
                "browser_tool", "fireworks", TEXT_ONLY_LLAMA)
        self.assertIn("vision_input", str(ctx.exception))
        # Nothing was persisted: the rejected selection cannot linger.
        self.assertFalse(self._settings_path.exists())

    def test_text_only_model_fails_use_time_validation_too(self):
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.validate_role_capabilities(
                "browser_tool", "fireworks", TEXT_ONLY_LLAMA)
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.validate_role_capabilities(
                "vision", "fireworks", "accounts/fireworks/models/llama-v3p1-70b-instruct")

    def test_vision_capable_models_still_selectable_for_browser(self):
        for model in (FIREWORKS_VISION_MODEL, DEFAULT_BROWSER_MODEL):
            model_registry.set_model_for_role("browser_tool", "fireworks", model)
            self.assertEqual(
                model_registry.get_model_for_role("browser_tool")["model"], model)
        snapshot = model_registry.get_model_config("browser_tool")
        self.assertTrue(snapshot["vision"])
        self.assertTrue(snapshot["tools"])
        self.assertTrue(snapshot["schema"])


class PlannerCapabilityTests(RegistryTestBase):
    """A non-tool planner model must fail; tools/schema/streaming are pinned."""

    def test_planner_default_is_capability_validated(self):
        snapshot = model_registry.get_model_config("planner")
        self.assertEqual(snapshot["provider"], "fireworks")
        self.assertIn("tool_calling", snapshot["required"])
        self.assertTrue(snapshot["tools"])
        self.assertTrue(snapshot["schema"])
        self.assertTrue(snapshot["streaming"])
        self.assertEqual(snapshot["adapter"], "openai_compatible")

    def test_non_tool_models_rejected_for_planner(self):
        for model in (GUARD_MODEL, EMBEDDING_MODEL):
            with self.assertRaises(model_registry.ModelRegistryError) as ctx:
                model_registry.set_model_for_role("planner", "fireworks", model)
            self.assertIn("tool_calling", str(ctx.exception))
        # ...and the same models are not chat models either (no streaming).
        for model in (GUARD_MODEL, EMBEDDING_MODEL):
            with self.assertRaises(model_registry.ModelRegistryError):
                model_registry.set_model_for_role("chat", "fireworks", model)

    def test_novel_model_needs_an_explicit_capability_declaration(self):
        novel = "accounts/fireworks/models/novel-1"
        model_registry.declare_model_capabilities(
            "fireworks", novel, ["streaming"])
        with self.assertRaises(model_registry.ModelRegistryError) as ctx:
            model_registry.set_model_for_role("planner", "fireworks", novel)
        self.assertIn("tool_calling", str(ctx.exception))
        # The explicit declaration is persisted and reviewable.
        stored = self._read_settings()["model_capabilities"][
            "fireworks/%s" % novel]
        self.assertEqual(stored["capabilities"], ["streaming"])
        # Widening it (an explicit authorization) makes the selection legal.
        model_registry.declare_model_capabilities(
            "fireworks", novel,
            ["streaming", "tool_calling", "structured_output"])
        self.assertEqual(
            model_registry.set_model_for_role("planner", "fireworks", novel),
            {"provider": "fireworks", "model": novel})
        self.assertEqual(
            model_registry.get_model_for_role("planner")["model"], novel)

    def test_declaration_rejects_unknown_capability_names(self):
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.declare_model_capabilities(
                "fireworks", "accounts/fireworks/models/x", ["telepathy"])
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.declare_model_capabilities("ghost", "m", ["streaming"])


class UnknownProviderFailClosedTests(RegistryTestBase):
    """F49: unknown providers inherit nothing; registered ones are declared."""

    def test_unknown_provider_has_no_capabilities(self):
        self.assertEqual(model_registry.model_capabilities_for("ghost"),
                         frozenset())
        for role in ("chat", "planner", "browser_tool", "vision"):
            with self.assertRaises(model_registry.ModelRegistryError) as ctx:
                model_registry.validate_role_capabilities(role, "ghost", "m")
            self.assertIn("unknown provider", str(ctx.exception))

    def test_registered_custom_provider_uses_its_declared_set(self):
        self._add_provider("acme", capabilities=["streaming"])
        # chat is authorized by the explicit declaration...
        model_registry.set_model_for_role("chat", "acme", "acme-chat")
        self.assertEqual(model_registry.get_model_for_role("chat")["provider"],
                         "acme")
        # ...browser/vision/planner are NOT (no tools, no vision declared).
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.set_model_for_role("browser_tool", "acme", "m")
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.validate_role_capabilities("browser_tool", "acme", "m")

    def test_custom_provider_declaration_can_authorize_browser(self):
        self._add_provider(
            "acme", capabilities=["streaming", "tool_calling",
                                  "structured_output", "vision_input"])
        model_registry.set_model_for_role("browser_tool", "acme", "acme-vision")
        snapshot = model_registry.get_model_config("browser_tool")
        self.assertEqual(snapshot["provider"], "acme")
        self.assertIn("custom_declared", snapshot["capability_sources"])
        self.assertTrue(snapshot["vision"])

    def test_provider_record_exposes_declared_capabilities(self):
        masked = self._add_provider("acme", capabilities=["streaming"])
        self.assertEqual(masked["capabilities"], ["streaming"])
        self.assertEqual(masked["capabilities_source"], "custom_declared")
        listed = [p for p in model_registry.list_providers() if p["id"] == "acme"][0]
        self.assertEqual(listed["capabilities"], ["streaming"])

    def test_default_custom_provider_declaration_is_explicit_and_stored(self):
        self._add_provider("acme")
        stored = self._read_settings()["custom_providers"][0]
        self.assertEqual(stored["capabilities_source"], "custom_default")
        self.assertIn("tool_calling", stored["capabilities"])
        self.assertIn("vision_input", stored["capabilities"])

    def test_env_providers_report_adapter_capabilities(self):
        providers = {p["id"]: p for p in model_registry.list_providers()}
        self.assertIn("vision_input", providers["fireworks"]["capabilities"])
        self.assertEqual(providers["fireworks"]["capabilities_source"],
                         "env_adapter")
        self.assertEqual(providers["fish"]["capabilities"], ["audio_output"])


class ObservedProviderMetadataTests(RegistryTestBase):
    """Actual provider metadata narrows a model, not just its provider."""

    def test_text_only_model_from_provider_metadata_is_refused(self):
        payload = {"data": [
            {"id": "vendor/text-only-1",
             "architecture": {"input_modalities": ["text"]},
             "supported_parameters": ["temperature"],
             "context_length": 8192,
             "top_provider": {"max_completion_tokens": 1024}},
            {"id": "vendor/vision-tools-1",
             "architecture": {"input_modalities": ["text", "image"]},
             "supported_parameters": ["tools", "response_format"],
             "context_length": 32768},
        ]}
        listed = self._list_provider("openrouter", payload)
        self.assertEqual([m["id"] for m in listed], ["vendor/vision-tools-1"])

        capabilities = model_registry.model_capabilities_for(
            "openrouter", "vendor/text-only-1")
        self.assertNotIn("tool_calling", capabilities)
        self.assertNotIn("vision_input", capabilities)
        self.assertNotIn("structured_output", capabilities)

        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.set_model_for_role(
                "browser_tool", "openrouter", "vendor/text-only-1")
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.set_model_for_role(
                "vision", "openrouter", "vendor/text-only-1")

        # The model the provider described as image + tools capable passes.
        model_registry.set_model_for_role(
            "browser_tool", "openrouter", "vendor/vision-tools-1")
        self.assertEqual(
            model_registry.get_model_for_role("browser_tool")["provider"],
            "openrouter")

    def test_observed_metadata_is_persisted_for_later_resolutions(self):
        payload = {"data": [{
            "id": "vendor/plain-1",
            "architecture": {"input_modalities": ["text"]},
            "supported_parameters": [],
        }]}
        self._list_provider("openrouter", payload)
        stored = self._read_settings()["observed_capabilities"]
        entry = stored["openrouter/vendor/plain-1"]
        self.assertFalse(entry["capabilities"]["vision_input"])
        self.assertFalse(entry["capabilities"]["tool_calling"])

    def test_gemini_metadata_records_streaming_and_limits(self):
        payload = {"models": [{
            "name": "models/gemini-2.5-flash",
            "displayName": "Gemini 2.5 Flash",
            "supportedGenerationMethods": ["generateContent"],
            "inputTokenLimit": 1048576,
            "outputTokenLimit": 65536,
        }]}
        self._list_provider("gemini", payload)
        self.assertIn("streaming", model_registry.model_capabilities_for(
            "gemini", "gemini-2.5-flash"))
        model_registry.set_model_for_role("vision", "gemini", "gemini-2.5-flash")
        snapshot = model_registry.get_model_config("vision")
        self.assertEqual(snapshot["limits"],
                         {"max_input_tokens": 1048576,
                          "max_output_tokens": 65536})


class PersistedSettingsValidationTests(RegistryTestBase):
    """An invalid persisted selection can never run."""

    def test_persisted_text_only_browser_selection_cannot_run(self):
        self._write_settings({"browser_tool_model": {
            "provider": "fireworks", "model": TEXT_ONLY_LLAMA}})
        with self.assertRaises(model_registry.ModelRegistryError) as ctx:
            model_registry.get_model_config("browser_tool")
        self.assertIn("vision_input", str(ctx.exception))
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.resolve_call_config("browser_tool")
        # The lenient read resolves the role's authorized default instead —
        # the persisted (invalid) model is never the one that runs.
        resolved = model_registry.get_model_for_role("browser_tool")
        self.assertNotEqual(resolved["model"], TEXT_ONLY_LLAMA)
        self.assertEqual(resolved["provider"], config.BROWSER_AGENT_PROVIDER)

    def test_persisted_disallowed_provider_never_runs(self):
        # gemini is capability-compatible but NOT allowed for the planner; a
        # hand-edited file must not be able to route planner work anywhere.
        self._write_settings({"planner_model": {
            "provider": "gemini", "model": GEMINI_MODEL}})
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.resolve_call_config("planner")
        resolved = model_registry.get_model_for_role("planner")
        self.assertEqual(resolved["provider"], "fireworks")
        self.assertEqual(resolved["model"], config.BROWSER_AGENT_MODEL)

    def test_persisted_unknown_provider_is_never_substituted(self):
        self._write_settings({"chat_model": {
            "provider": "ghost_private", "model": "m"}})
        with self.assertRaises(model_registry.ModelRegistryError) as ctx:
            model_registry.get_model_for_role("chat")
        self.assertIn("ghost_private", str(ctx.exception))
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.get_model_config("chat")

    def test_incomplete_persisted_selection_degrades_to_env_default(self):
        self._write_settings({"chat_model": {"provider": "fireworks"}})
        self.assertEqual(model_registry.get_default_chat_model()["provider"],
                         "gemini")

    def test_authorized_fallback_only_for_env_providers(self):
        self._add_provider("acme")
        self.assertEqual(
            model_registry.authorized_fallback_for("chat", "gemini")["provider"],
            "gemini")
        # A private provider's work is never rerouted.
        self.assertIsNone(model_registry.authorized_fallback_for("chat", "acme"))
        self.assertIsNone(
            model_registry.authorized_fallback_for("chat", "ghost_private"))


class PrivateCredentialTests(RegistryTestBase):
    """Missing private credentials send nothing elsewhere."""

    def test_private_provider_without_key_fails_closed(self):
        self._add_provider("acme")
        model_registry.set_model_for_role("chat", "acme", "acme-chat")
        settings = self._read_settings()
        settings["custom_providers"][0].pop("api_key")
        self._write_settings(settings)

        for resolve in (model_registry.get_model_for_role,
                        model_registry.get_model_config,
                        model_registry.resolve_call_config):
            with self.assertRaises(model_registry.ModelRegistryError) as ctx:
                resolve("chat")
            self.assertIn("acme", str(ctx.exception))
            self.assertIn("credentials", str(ctx.exception))
        # No other provider's credentials are offered in its place.
        key, _base = model_registry.get_provider_credentials("acme")
        self.assertIsNone(key)

    def test_private_provider_without_base_url_fails_closed(self):
        self._add_provider("acme")
        model_registry.set_model_for_role("browser_tool", "acme", "acme-vision")
        settings = self._read_settings()
        settings["custom_providers"][0]["base_url"] = ""
        self._write_settings(settings)
        with self.assertRaises(model_registry.ModelRegistryError) as ctx:
            model_registry.resolve_call_config("browser_tool")
        self.assertIn("credentials", str(ctx.exception))

    def test_private_endpoint_is_used_verbatim_for_the_call(self):
        self._add_provider("acme")
        model_registry.set_model_for_role("chat", "acme", "acme-chat")
        snapshot = model_registry.resolve_call_config("chat")
        with patch.object(orchestrator, "ask_openai_compat") as client:
            client.return_value = {"choices": []}
            orchestrator._chat([{"role": "user", "content": "hi"}], snapshot)
        self.assertEqual(client.call_count, 1)
        self.assertEqual(client.call_args.kwargs["base_url"],
                         "https://acme.example/v1")
        self.assertEqual(client.call_args.kwargs["api_key"], "sk-acme")
        self.assertNotIn("fireworks", json.dumps(client.call_args.kwargs))

    def test_chat_snapshot_without_endpoint_sends_nothing(self):
        snapshot = {
            "provider": "acme", "model": "acme-chat",
            "adapter": "openai_compatible",
            "endpoint": {"base_url": None, "has_credentials": False},
        }
        with patch.object(orchestrator, "ask_openai_compat") as client:
            self.assertEqual(orchestrator._chat([], snapshot), {})
        client.assert_not_called()

    def test_native_adapter_snapshot_is_never_sent_to_fireworks(self):
        model_registry.set_model_for_role("vision", "gemini", "gemini-2.5-pro")
        snapshot = model_registry.resolve_call_config("vision")
        self.assertEqual(snapshot["adapter"], "gemini_native")
        with patch.object(orchestrator, "ask_openai_compat") as client:
            self.assertEqual(orchestrator._chat([], snapshot), {})
        client.assert_not_called()

    def test_orchestrator_declines_without_calling_any_provider(self):
        """A persisted planner override that cannot be validated must not be
        turned into a Fireworks request (the old silent substitution)."""
        self._write_settings({"planner_model": {
            "provider": "gemini", "model": GEMINI_MODEL}})
        with patch.object(orchestrator, "orchestrator_mode", return_value=True), \
             patch.object(orchestrator, "ask_openai_compat") as client, \
             patch.object(orchestrator, "_chat") as chat:
            self.assertIsNone(
                orchestrator.handle_message("hello", screen_question=False))
        client.assert_not_called()
        chat.assert_not_called()


class AtomicSnapshotTests(RegistryTestBase):
    """One locked read per resolution; a snapshot travels with its call."""

    def test_resolution_reads_settings_exactly_once(self):
        self._add_provider("acme")
        model_registry.set_model_for_role("browser_tool", "acme", "acme-vision")
        real_load = model_registry._load_unlocked
        for role in ("chat", "planner", "browser_tool", "tts", "listening"):
            calls = []

            def counting(role=role, calls=calls):
                calls.append(1)
                return real_load()

            with patch.object(model_registry, "_load_unlocked",
                              side_effect=counting):
                model_registry.get_model_config(role)
            self.assertEqual(len(calls), 1, role)

    def test_snapshot_travels_with_the_call(self):
        self._add_provider("acme")
        model_registry.set_model_for_role("chat", "acme", "acme-chat")
        first = model_registry.resolve_call_config("chat")
        model_registry.set_model_for_role("chat", "fireworks", FIREWORKS_VISION_MODEL)
        # The in-flight snapshot is unchanged by the later switch.
        self.assertEqual(first["provider"], "acme")
        self.assertEqual(first["model"], "acme-chat")
        self.assertEqual(first["endpoint"]["base_url"], "https://acme.example/v1")
        self.assertEqual(first["endpoint"]["api_key"], "sk-acme")
        second = model_registry.resolve_call_config("chat")
        self.assertEqual(second["provider"], "fireworks")
        self.assertEqual(second["model"], FIREWORKS_VISION_MODEL)
        self.assertNotEqual(first["settings_revision"], second["settings_revision"])
        self.assertGreater(second["settings_revision"],
                           first["settings_revision"])

    def test_concurrent_model_switch_never_tears_a_snapshot(self):
        pair_a = ("gemini", "gemini-2.5-pro")
        pair_b = ("fireworks", FIREWORKS_VISION_MODEL)
        model_registry.set_model_for_role("chat", *pair_a)
        stop = threading.Event()
        errors = []

        def writer():
            toggle = False
            while not stop.is_set():
                toggle = not toggle
                try:
                    model_registry.set_model_for_role(
                        "chat", *(pair_b if toggle else pair_a))
                except Exception as exc:  # pragma: no cover - surfaced below
                    errors.append(exc)

        thread = threading.Thread(target=writer)
        thread.start()
        try:
            for _ in range(150):
                snapshot = model_registry.get_model_config("chat")
                pair = (snapshot["provider"], snapshot["model"])
                self.assertIn(pair, (pair_a, pair_b))
                # Capabilities, adapter and endpoint must belong to the SAME
                # provider that the snapshot names.
                self.assertTrue(snapshot["streaming"])
                self.assertEqual(snapshot["adapter"],
                                 "gemini_native" if pair[0] == "gemini"
                                 else "openai_compatible")
                self.assertEqual(snapshot["endpoint"]["credential_source"],
                                 "environment")
                self.assertIsInstance(snapshot["settings_revision"], int)
        finally:
            stop.set()
            thread.join()
        self.assertEqual(errors, [])

    def test_every_mutation_bumps_the_settings_revision(self):
        self._write_settings({"chat_model": {
            "provider": "gemini", "model": "gemini-2.5-pro"}})
        before = model_registry.get_model_config("chat")["settings_revision"]
        model_registry.set_model_for_role("chat", "gemini", "gemini-2.5-flash")
        after = model_registry.get_model_config("chat")["settings_revision"]
        self.assertGreater(after, before)


class LimitsAndReasoningTests(RegistryTestBase):
    """Limits and reasoning controls come from the validated snapshot."""

    def test_observed_limits_are_enforced(self):
        payload = {"data": [{
            "id": "vendor/vision-tools-1",
            "architecture": {"input_modalities": ["text", "image"]},
            "supported_parameters": ["tools", "response_format"],
            "context_length": 8192,
            "top_provider": {"max_completion_tokens": 1024},
        }]}
        self._list_provider("openrouter", payload)
        model_registry.set_model_for_role(
            "browser_tool", "openrouter", "vendor/vision-tools-1")
        snapshot = model_registry.get_model_config("browser_tool")
        self.assertEqual(snapshot["limits"],
                         {"max_input_tokens": 8192, "max_output_tokens": 1024})
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.validate_request_limits(snapshot, max_tokens=4096)
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.validate_request_limits(
                snapshot, prompt_tokens=8000, max_tokens=512)
        self.assertTrue(
            model_registry.validate_request_limits(snapshot, max_tokens=512))

    def test_orchestrator_refuses_a_request_over_the_model_limits(self):
        snapshot = {
            "role": "planner", "provider": "fireworks", "model": "m",
            "adapter": "openai_compatible",
            "endpoint": {"base_url": "https://api.fireworks.ai/inference/v1",
                         "api_key": "k"},
            "limits": {"max_input_tokens": 4096, "max_output_tokens": 256},
            "reasoning": {"supported": False, "effort": None},
        }
        with patch.object(orchestrator, "ask_openai_compat") as client:
            self.assertEqual(orchestrator._chat([], snapshot), {})
        client.assert_not_called()

    def test_reasoning_control_only_when_the_model_takes_it(self):
        snapshot = model_registry.resolve_call_config("planner")
        self.assertTrue(snapshot["reasoning"]["supported"])
        self.assertEqual(snapshot["reasoning"]["param"], "reasoning_effort")
        with patch.object(config, "BROWSER_AGENT_REASONING_EFFORT", "medium"), \
             patch.object(orchestrator, "ask_openai_compat") as client:
            client.return_value = {"choices": []}
            orchestrator._chat([], model_registry.resolve_call_config("planner"))
        self.assertEqual(client.call_args.kwargs["reasoning_effort"], "medium")

        # A model that runs its own default reasoning never receives it.
        model_registry.set_model_for_role(
            "planner", "fireworks", "accounts/fireworks/models/minimax-m2")
        minimax = model_registry.resolve_call_config("planner")
        self.assertFalse(minimax["reasoning"]["supported"])
        with patch.object(config, "BROWSER_AGENT_REASONING_EFFORT", "medium"), \
             patch.object(orchestrator, "ask_openai_compat") as client:
            client.return_value = {"choices": []}
            orchestrator._chat([], minimax)
        self.assertNotIn("reasoning_effort", client.call_args.kwargs)

    def test_default_planner_call_uses_its_own_endpoint(self):
        snapshot = model_registry.resolve_call_config("planner")
        self.assertEqual(snapshot["endpoint"]["base_url"],
                         "https://api.fireworks.ai/inference/v1")
        self.assertEqual(snapshot["endpoint"]["api_key"], "test-fireworks-key")
        with patch.object(orchestrator, "ask_openai_compat") as client:
            client.return_value = {"choices": []}
            orchestrator._chat([{"role": "user", "content": "hi"}], snapshot)
        self.assertEqual(client.call_args.kwargs["base_url"],
                         "https://api.fireworks.ai/inference/v1")
        self.assertEqual(client.call_args.kwargs["model"],
                         config.BROWSER_AGENT_MODEL)


if __name__ == "__main__":
    unittest.main()
