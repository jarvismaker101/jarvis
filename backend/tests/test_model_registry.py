import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.services import model_registry
from backend.services.gemini_client import GEMINI_CHAT_MODEL


class ModelRegistryTestBase(unittest.TestCase):
    """Isolate the registry: temp settings file + deterministic env keys."""

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


class DefaultChatModelTests(ModelRegistryTestBase):
    def test_env_default_when_no_settings(self):
        self.assertEqual(
            model_registry.get_default_chat_model(),
            {"provider": "gemini", "model": GEMINI_CHAT_MODEL},
        )

    def test_settings_override_wins(self):
        model_registry.set_default_chat_model("gemini", "gemini-2.5-pro")
        self.assertEqual(
            model_registry.get_default_chat_model(),
            {"provider": "gemini", "model": "gemini-2.5-pro"},
        )

    def test_persistence_roundtrip(self):
        model_registry.set_default_chat_model(
            "fireworks", "accounts/fireworks/models/qwen3p7-plus"
        )
        self.assertTrue(self._settings_path.exists())
        with open(self._settings_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["chat_model"]["provider"], "fireworks")
        self.assertEqual(
            data["chat_model"]["model"], "accounts/fireworks/models/qwen3p7-plus"
        )
        # fresh read reflects the override
        self.assertEqual(
            model_registry.get_default_chat_model(),
            {"provider": "fireworks", "model": "accounts/fireworks/models/qwen3p7-plus"},
        )

    def test_set_rejects_unknown_provider(self):
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.set_default_chat_model("grok", "x-latest")
        self.assertFalse(self._settings_path.exists())
        self.assertEqual(model_registry.get_default_chat_model()["provider"], "gemini")

    def test_set_rejects_empty_model(self):
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.set_default_chat_model("gemini", "   ")

    def test_corrupt_settings_falls_back_to_env_default(self):
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        self._settings_path.write_text("{not json", encoding="utf-8")
        self.assertEqual(model_registry.get_default_chat_model()["provider"], "gemini")


class CustomProviderTests(ModelRegistryTestBase):
    def setUp(self):
        super().setUp()
        # default live-validation stub for every test in this class: never
        # touch the network (specific tests override with their own patch)
        patcher = patch.object(
            model_registry,
            "_list_openai_compat_models",
            return_value=[{"id": "m", "display": "m"}],
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _add(self, pid="acme", name="Acme", key="sk-acme-123",
             base="https://acme.example/v1"):
        return model_registry.add_custom_provider(pid, name, key, base)

    def test_add_stores_and_masks(self):
        masked = self._add()
        self.assertEqual(masked["id"], "acme")
        self.assertTrue(masked["has_key"])
        self.assertNotIn("api_key", masked)
        listed = model_registry.list_custom_providers()
        self.assertEqual([p["id"] for p in listed], ["acme"])
        self.assertNotIn("api_key", json.dumps(listed))

    def test_add_requires_live_validation(self):
        with patch.object(model_registry, "_list_openai_compat_models") as probe:
            self._add()
        probe.assert_called_once_with("https://acme.example/v1", "sk-acme-123")

    def test_invalid_key_rejected_and_not_stored(self):
        with patch.object(
            model_registry,
            "_list_openai_compat_models",
            side_effect=model_registry.ModelRegistryError(
                "model list request failed (HTTP 401)"
            ),
        ):
            with self.assertRaises(model_registry.ModelRegistryError):
                self._add(key="sk-bad")
        self.assertEqual(model_registry.list_custom_providers(), [])
        self.assertFalse(self._settings_path.exists())

    def test_empty_model_list_rejected(self):
        with patch.object(
            model_registry, "_list_openai_compat_models", return_value=[]
        ):
            with self.assertRaises(model_registry.ModelRegistryError):
                self._add()
        self.assertEqual(model_registry.list_custom_providers(), [])

    def test_duplicate_id_rejected_without_network(self):
        self._add()
        with patch.object(model_registry, "_list_openai_compat_models") as probe:
            with self.assertRaises(model_registry.ModelRegistryError):
                self._add(pid="acme", key="sk-other")
        probe.assert_not_called()  # duplicate check fires before live validation

    def test_reserved_and_malformed_ids_rejected(self):
        for bad in ("gemini", "fireworks", "Bad Id", "", "-x"):
            with self.assertRaises(model_registry.ModelRegistryError):
                self._add(pid=bad)
        self.assertEqual(model_registry.list_custom_providers(), [])

    def test_custom_provider_selectable_as_default(self):
        self._add()
        model_registry.set_default_chat_model("acme", "acme-chat-1")
        self.assertEqual(
            model_registry.get_default_chat_model(),
            {"provider": "acme", "model": "acme-chat-1"},
        )


class ProviderCredentialsTests(ModelRegistryTestBase):
    def test_env_providers_resolve_env_keys(self):
        key, base = model_registry.get_provider_credentials("gemini")
        self.assertEqual(key, "test-gemini-key")
        self.assertIsNone(base)
        key, base = model_registry.get_provider_credentials("fireworks")
        self.assertEqual(key, "test-fireworks-key")
        self.assertIsNone(base)

    def test_custom_provider_credentials_and_url_normalization(self):
        with patch.object(
            model_registry,
            "_list_openai_compat_models",
            return_value=[{"id": "m", "display": "m"}],
        ):
            model_registry.add_custom_provider(
                "acme", "Acme", "sk-acme-123", "https://acme.example/v1/"
            )
        key, base = model_registry.get_provider_credentials("acme")
        self.assertEqual(key, "sk-acme-123")
        self.assertEqual(base, "https://acme.example/v1")  # trailing slash stripped

    def test_unknown_provider_has_no_credentials(self):
        key, base = model_registry.get_provider_credentials("nope")
        self.assertIsNone(key)
        self.assertIsNone(base)


class ListProvidersTests(ModelRegistryTestBase):
    def test_list_providers_masks_all_keys(self):
        with patch.object(
            model_registry,
            "_list_openai_compat_models",
            return_value=[{"id": "m", "display": "m"}],
        ):
            model_registry.add_custom_provider(
                "acme", "Acme", "sk-secret-1", "https://acme.example/v1"
            )
        providers = model_registry.list_providers()
        blob = json.dumps(providers)
        self.assertNotIn("sk-secret-1", blob)
        self.assertNotIn("api_key", blob)
        self.assertEqual([p["id"] for p in providers], ["gemini", "fireworks", "groq", "fish", "gtts", "openrouter", "whisper", "inworld", "acme"])
        by_id = {p["id"]: p for p in providers}
        self.assertTrue(by_id["gemini"]["has_key"])
        self.assertTrue(by_id["fireworks"]["has_key"])
        self.assertIn("has_key", by_id["fish"])
        self.assertIn("has_key", by_id["openrouter"])
        self.assertEqual(by_id["gemini"]["source"], "env")
        self.assertEqual(by_id["acme"]["source"], "custom")
        self.assertEqual(by_id["acme"]["base_url"], "https://acme.example/v1")


class ProviderModelListTests(ModelRegistryTestBase):
    @staticmethod
    def _fake_response(payload, status=200):
        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = payload
        resp.text = json.dumps(payload)
        return resp

    def test_gemini_models_filtered_and_prefixed_stripped(self):
        payload = {
            "models": [
                {
                    "name": "models/gemini-2.5-pro",
                    "displayName": "Gemini 2.5 Pro",
                    "supportedGenerationMethods": ["generateContent", "countTokens"],
                },
                {
                    "name": "models/embedding-001",
                    "displayName": "Embedding",
                    "supportedGenerationMethods": ["embedContent"],
                },
                {
                    "name": "models/gemini-2.5-flash",
                    "displayName": "Gemini 2.5 Flash",
                    "supportedGenerationMethods": ["generateContent"],
                },
            ]
        }
        with patch.object(model_registry, "_session") as sess:
            sess.get.return_value = self._fake_response(payload)
            models = model_registry.list_provider_models("gemini")
        self.assertEqual(
            models,
            [
                {"id": "gemini-2.5-pro", "display": "Gemini 2.5 Pro"},
                {"id": "gemini-2.5-flash", "display": "Gemini 2.5 Flash"},
            ],
        )

    def test_gemini_pagination_followed(self):
        page1 = {
            "models": [
                {
                    "name": "models/m1",
                    "displayName": "M1",
                    "supportedGenerationMethods": ["generateContent"],
                }
            ],
            "nextPageToken": "tok",
        }
        page2 = {
            "models": [
                {
                    "name": "models/m2",
                    "displayName": "M2",
                    "supportedGenerationMethods": ["generateContent"],
                }
            ]
        }
        with patch.object(model_registry, "_session") as sess:
            sess.get.side_effect = [
                self._fake_response(page1),
                self._fake_response(page2),
            ]
            models = model_registry.list_provider_models("gemini")
        self.assertEqual([m["id"] for m in models], ["m1", "m2"])
        self.assertEqual(sess.get.call_count, 2)

    def test_gemini_key_never_in_raised_error_text(self):
        with patch.object(model_registry, "_session") as sess:
            sess.get.side_effect = OSError("boom https://x?key=test-gemini-key")
            with self.assertRaises(model_registry.ModelRegistryError) as ctx:
                model_registry.list_provider_models("gemini")
        self.assertNotIn("test-gemini-key", str(ctx.exception))

    def test_fireworks_ids_kept_full(self):
        payload = {
            "data": [
                {"id": "accounts/fireworks/models/qwen3p7-plus"},
                {"id": "accounts/fireworks/models/deepseek-v4-pro"},
            ]
        }
        with patch.object(model_registry, "_session") as sess:
            sess.get.return_value = self._fake_response(payload)
            models = model_registry.list_provider_models("fireworks")
        self.assertEqual(
            models[0]["id"], "accounts/fireworks/models/qwen3p7-plus"
        )
        self.assertEqual(models[0]["display"], "qwen3p7-plus")

    def test_custom_provider_models_openai_shape(self):
        with patch.object(
            model_registry,
            "_list_openai_compat_models",
            return_value=[{"id": "m", "display": "m"}],
        ):
            model_registry.add_custom_provider(
                "acme", "Acme", "sk-1", "https://acme.example/v1"
            )
        with patch.object(model_registry, "_session") as sess:
            sess.get.return_value = self._fake_response(
                {"data": [{"id": "acme-chat"}, {"id": "acme-fast"}]}
            )
            models = model_registry.list_provider_models("acme")
        self.assertEqual([m["id"] for m in models], ["acme-chat", "acme-fast"])
        # request went to {base_url}/models with a bearer header
        url = sess.get.call_args.args[0]
        self.assertEqual(url, "https://acme.example/v1/models")
        headers = sess.get.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer sk-1")

    def test_unknown_provider_models_error(self):
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.list_provider_models("nope")

    def test_missing_env_key_models_error(self):
        model_registry.GEMINI_API_KEY = None
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.list_provider_models("gemini")

    def test_http_error_becomes_registry_error(self):
        with patch.object(model_registry, "_session") as sess:
            sess.get.return_value = self._fake_response({}, status=403)
            with self.assertRaises(model_registry.ModelRegistryError) as ctx:
                model_registry.list_provider_models("fireworks")
        self.assertIn("403", str(ctx.exception))


class ConcurrencyTests(ModelRegistryTestBase):
    def test_concurrent_set_writes_leave_valid_json(self):
        def worker(n):
            for i in range(10):
                model_registry.set_default_chat_model(
                    "gemini", f"gemini-test-{n}-{i}"
                )

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        with open(self._settings_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("chat_model", data)
        self.assertEqual(
            model_registry.get_default_chat_model()["provider"], "gemini"
        )


class LostUpdateRaceTests(ModelRegistryTestBase):
    """Round-5 hardening: mutations must hold _lock across the ENTIRE
    read-modify-write, or two writers on different keys can interleave and
    the later full-file write silently clobbers the earlier change."""

    def _read_file(self):
        with open(self._settings_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _assert_both_changes_present(self):
        data = self._read_file()
        self.assertEqual(
            data.get("chat_model"),
            {"provider": "gemini", "model": "gemini-2.5-pro"},
        )
        self.assertEqual(
            [p["id"] for p in data.get("custom_providers", [])], ["acme"]
        )

    def test_barrier_synchronized_set_and_add_both_survive(self):
        barrier = threading.Barrier(2)

        def set_chat_model():
            barrier.wait()
            model_registry.set_default_chat_model("gemini", "gemini-2.5-pro")

        def add_provider():
            barrier.wait()
            with patch.object(
                model_registry,
                "_list_openai_compat_models",
                return_value=[{"id": "m", "display": "m"}],
            ):
                model_registry.add_custom_provider(
                    "acme", "Acme", "sk-1", "https://acme.example/v1"
                )

        t1 = threading.Thread(target=set_chat_model)
        t2 = threading.Thread(target=add_provider)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self._assert_both_changes_present()

    def test_forced_interleave_set_during_add_probe_keeps_both(self):
        # Deterministic version of the race: the probe fires after add's
        # first settings read and before its write — exactly the window
        # where the pre-fix code held stale state. The chat-model change
        # committed inside that window must survive the add's full-file
        # write (the add re-reads under the lock instead of clobbering).
        def probe(url, api_key):
            model_registry.set_default_chat_model("gemini", "gemini-2.5-pro")
            return [{"id": "m", "display": "m"}]

        with patch.object(
            model_registry, "_list_openai_compat_models", side_effect=probe
        ):
            masked = model_registry.add_custom_provider(
                "acme", "Acme", "sk-1", "https://acme.example/v1"
            )
        self.assertEqual(masked["id"], "acme")
        self._assert_both_changes_present()


class ErrorScrubTests(ModelRegistryTestBase):
    """Round-5 hardening: provider error bodies must be scrubbed before the
    text can reach a UI-facing ModelRegistryError detail."""

    def test_non_200_body_key_material_never_reaches_error_detail(self):
        body = (
            "request denied: url https://gw/v1/models?key=SECRETPATTERN&x=1 "
            "Authorization: Bearer SECRETPATTERN header, raw run "
            "abcdefghijklmnopqrstuvwxyz012345"
        )
        resp = MagicMock()
        resp.status_code = 403
        resp.text = body
        with patch.object(model_registry, "_session") as sess:
            sess.get.return_value = resp
            with self.assertRaises(model_registry.ModelRegistryError) as ctx:
                model_registry.list_provider_models("fireworks")
        detail = str(ctx.exception)
        self.assertNotIn("SECRETPATTERN", detail)
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz012345", detail)
        self.assertIn("403", detail)

    def test_masked_base_url_hides_query_string(self):
        with patch.object(
            model_registry,
            "_list_openai_compat_models",
            return_value=[{"id": "m", "display": "m"}],
        ):
            masked = model_registry.add_custom_provider(
                "acme", "Acme", "sk-1",
                "https://acme.example/v1?apikey=sk-supersecret",
            )
        self.assertEqual(masked["base_url"], "https://acme.example/v1")
        blob = json.dumps(masked) + json.dumps(model_registry.list_providers())
        self.assertNotIn("sk-supersecret", blob)


class BrainChatModelResolutionTests(ModelRegistryTestBase):
    """brain.py must resolve the registry per message and pass model=."""

    @staticmethod
    def _msg():
        return [{"role": "user", "content": "x"}]

    def test_stream_uses_registry_model(self):
        from backend.core import brain

        model_registry.set_default_chat_model("gemini", "gemini-2.5-pro")
        captured = {}

        def fake_gemini_stream(messages, temperature, max_tokens, model=None,
                             cancel=None):
            captured["model"] = model
            yield "hi"

        with patch.object(brain, "ask_gemini_chat_stream",
                          side_effect=fake_gemini_stream):
            deltas = list(brain._stream_chat_deltas(self._msg(), 0.7, 100))
        self.assertEqual(deltas, ["hi"])
        self.assertEqual(captured["model"], "gemini-2.5-pro")

    def test_stream_fireworks_primary(self):
        from backend.core import brain

        model_registry.set_default_chat_model(
            "fireworks", "accounts/fireworks/models/qwen3p7-plus"
        )
        calls = []

        def fake_fw_stream(messages, temperature, max_tokens, model=None,
                         cancel=None):
            calls.append(("fireworks", model))
            yield "fw"

        def fake_gemini_stream(messages, temperature, max_tokens, model=None,
                             cancel=None):
            calls.append(("gemini", model))
            yield "should not be reached"

        with patch.object(brain, "ask_fireworks_stream", side_effect=fake_fw_stream), \
             patch.object(brain, "ask_gemini_chat_stream",
                          side_effect=fake_gemini_stream):
            deltas = list(brain._stream_chat_deltas(self._msg(), 0.7, 100))
        self.assertEqual(deltas, ["fw"])
        self.assertEqual(
            calls, [("fireworks", "accounts/fireworks/models/qwen3p7-plus")]
        )

    def test_stream_custom_provider_primary(self):
        from backend.core import brain

        with patch.object(
            model_registry,
            "_list_openai_compat_models",
            return_value=[{"id": "m", "display": "m"}],
        ):
            model_registry.add_custom_provider(
                "acme", "Acme", "sk-1", "https://acme.example/v1"
            )
        model_registry.set_default_chat_model("acme", "acme-chat")
        captured = {}

        def fake_compat_stream(messages, model, base_url, api_key,
                               temperature, max_tokens, cancel=None):
            captured.update(model=model, base_url=base_url, api_key=api_key)
            yield "acme"

        with patch.object(brain, "ask_openai_compat_stream",
                          side_effect=fake_compat_stream):
            deltas = list(brain._stream_chat_deltas(self._msg(), 0.7, 100))
        self.assertEqual(deltas, ["acme"])
        self.assertEqual(captured["model"], "acme-chat")
        self.assertEqual(captured["base_url"], "https://acme.example/v1")
        self.assertEqual(captured["api_key"], "sk-1")

    def test_nonstream_uses_registry_model(self):
        from backend.core import brain

        model_registry.set_default_chat_model("gemini", "gemini-2.5-pro")
        captured = {}

        def fake_gemini(messages, temperature, max_tokens, model=None):
            captured["model"] = model
            return {"choices": [{"message": {"content": "ok"}}]}

        with patch.object(brain, "ask_gemini_chat", side_effect=fake_gemini):
            result = brain._ask_chat_nonstream(self._msg(), 0.7, 100)
        self.assertEqual(result["choices"][0]["message"]["content"], "ok")
        self.assertEqual(captured["model"], "gemini-2.5-pro")


class RoleModelTests(ModelRegistryTestBase):
    def test_per_role_env_defaults(self):
        # No settings file -> env defaults per role
        self.assertEqual(model_registry.get_model_for_role("chat")["provider"], "gemini")
        from backend.services.gemini_client import GEMINI_CHAT_MODEL as _CM
        self.assertEqual(model_registry.get_model_for_role("chat")["model"], _CM)
        self.assertEqual(model_registry.get_model_for_role("tts")["provider"], "fish")
        from backend.config import FISH_MODEL as _FM
        self.assertEqual(model_registry.get_model_for_role("tts")["model"], _FM)
        self.assertEqual(model_registry.get_model_for_role("vision")["provider"], "gemini")
        from backend.services.gemini_client import GEMINI_MODEL as _VM
        self.assertEqual(model_registry.get_model_for_role("vision")["model"], _VM)
        bt = model_registry.get_model_for_role("browser_tool")
        from backend.config import BROWSER_AGENT_PROVIDER as _BP, BROWSER_AGENT_MODEL as _BM
        self.assertEqual(bt["provider"], _BP)
        self.assertEqual(bt["model"], _BM)

    def test_role_roundtrip_persists_independently(self):
        model_registry.set_model_for_role("tts", "fish", "s1")
        model_registry.set_model_for_role("vision", "gemini", "gemini-2.5-flash")
        model_registry.set_model_for_role("browser_tool", "fireworks", "accounts/fireworks/models/qwen3p7-plus")
        self.assertEqual(model_registry.get_model_for_role("tts"), {"provider": "fish", "model": "s1"})
        self.assertEqual(model_registry.get_model_for_role("vision"), {"provider": "gemini", "model": "gemini-2.5-flash"})
        self.assertEqual(model_registry.get_model_for_role("browser_tool"), {"provider": "fireworks", "model": "accounts/fireworks/models/qwen3p7-plus"})
        # chat remains env default
        from backend.services.gemini_client import GEMINI_CHAT_MODEL as _CM
        self.assertEqual(model_registry.get_model_for_role("chat"), {"provider": "gemini", "model": _CM})
        # chat set does not clobber other roles
        model_registry.set_model_for_role("chat", "gemini", "gemini-2.5-pro")
        self.assertEqual(model_registry.get_model_for_role("tts")["model"], "s1")
        self.assertEqual(model_registry.get_model_for_role("chat")["model"], "gemini-2.5-pro")

    def test_role_whitelist_rejects_unknown(self):
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.get_model_for_role("unknown_role")
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.set_model_for_role("invalid", "gemini", "m")
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.set_model_for_role("chat", "unknown_provider_xyz", "m")

    def test_chat_backward_compat_delegates(self):
        model_registry.set_default_chat_model("gemini", "gemini-2.5-pro")
        self.assertEqual(model_registry.get_default_chat_model(), {"provider": "gemini", "model": "gemini-2.5-pro"})
        self.assertEqual(model_registry.get_model_for_role("chat"), {"provider": "gemini", "model": "gemini-2.5-pro"})
        model_registry.set_model_for_role("chat", "fireworks", "accounts/fireworks/models/qwen3p7-plus")
        self.assertEqual(model_registry.get_default_chat_model()["model"], "accounts/fireworks/models/qwen3p7-plus")

    def test_custom_provider_selectable_for_any_role(self):
        with patch.object(model_registry, "_list_openai_compat_models", return_value=[{"id": "m", "display": "m"}]):
            model_registry.add_custom_provider("acme", "Acme", "sk-1", "https://acme.example/v1")
        # custom allowed only for chat and browser_tool per allowlist
        model_registry.set_model_for_role("chat", "acme", "acme-chat")
        model_registry.set_model_for_role("browser_tool", "acme", "acme-browser")
        self.assertEqual(model_registry.get_model_for_role("chat")["provider"], "acme")
        self.assertEqual(model_registry.get_model_for_role("browser_tool")["model"], "acme-browser")
        # tts, vision and listening must reject custom
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.set_model_for_role("tts", "acme", "acme-tts")
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.set_model_for_role("vision", "acme", "acme-vision")
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.set_model_for_role("listening", "acme", "acme-stt")


class ListeningRoleTests(ModelRegistryTestBase):
    """Listening (conversation STT) role: whisper <-> inworld switching,
    mirroring the existing role machinery."""

    def test_listening_in_valid_roles_and_maps(self):
        self.assertIn("listening", model_registry.VALID_ROLES)
        self.assertEqual(
            model_registry._ROLE_STORAGE_KEY["listening"], "listening_model"
        )
        self.assertEqual(
            model_registry.get_allowed_providers_for_role("listening"),
            ["inworld", "whisper"],
        )
        self.assertEqual(
            model_registry.get_role_allowed_map()["listening"],
            ["inworld", "whisper"],
        )

    def test_listening_default_is_inworld_when_unset(self):
        with patch.dict(os.environ, {}):
            os.environ.pop("INWORLD_STT_MODEL", None)
            self.assertEqual(
                model_registry.get_model_for_role("listening"),
                {"provider": "inworld", "model": "inworld/inworld-stt-1"},
            )

    def test_listening_default_model_follows_env(self):
        with patch.dict(os.environ, {"INWORLD_STT_MODEL": "inworld/custom-stt-9"}):
            self.assertEqual(
                model_registry.get_model_for_role("listening"),
                {"provider": "inworld", "model": "inworld/custom-stt-9"},
            )

    def test_listening_roundtrip_and_storage_key(self):
        model_registry.set_model_for_role("listening", "whisper", "whisper-local")
        self.assertEqual(
            model_registry.get_model_for_role("listening"),
            {"provider": "whisper", "model": "whisper-local"},
        )
        with open(self._settings_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(
            data["listening_model"],
            {"provider": "whisper", "model": "whisper-local"},
        )
        model_registry.set_model_for_role(
            "listening", "inworld", "inworld/inworld-stt-1"
        )
        self.assertEqual(
            model_registry.get_model_for_role("listening"),
            {"provider": "inworld", "model": "inworld/inworld-stt-1"},
        )

    def test_listening_rejects_disallowed_env_providers(self):
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.set_model_for_role(
                "listening", "gemini", "gemini-2.5-pro"
            )
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.set_model_for_role("listening", "fish", "s1")

    def test_listening_credentials_masked_shape(self):
        self.assertEqual(
            model_registry.get_provider_credentials("whisper"), (None, None)
        )
        with patch.dict(os.environ, {"INWORLD_STT_API_KEY": "test-inworld-key"}):
            key, base = model_registry.get_provider_credentials("inworld")
            self.assertEqual(key, "test-inworld-key")
            self.assertIsNone(base)
        with patch.dict(os.environ, {}):
            os.environ.pop("INWORLD_STT_API_KEY", None)
            self.assertEqual(
                model_registry.get_provider_credentials("inworld"), (None, None)
            )

    def test_list_providers_reports_listening_providers_masked(self):
        with patch.dict(os.environ, {"INWORLD_STT_API_KEY": "test-inworld-key"}):
            providers = model_registry.list_providers()
        by_id = {p["id"]: p for p in providers}
        self.assertIn("whisper", by_id)
        self.assertIn("inworld", by_id)
        self.assertEqual(by_id["whisper"]["name"], "Local Whisper")
        self.assertEqual(by_id["inworld"]["name"], "Inworld STT")
        self.assertFalse(by_id["whisper"]["has_key"])
        self.assertTrue(by_id["inworld"]["has_key"])
        blob = json.dumps(providers)
        self.assertNotIn("test-inworld-key", blob)
        self.assertNotIn("api_key", blob)
        with patch.dict(os.environ, {}):
            os.environ.pop("INWORLD_STT_API_KEY", None)
            providers = model_registry.list_providers()
        by_id = {p["id"]: p for p in providers}
        self.assertFalse(by_id["inworld"]["has_key"])

    def test_listening_model_lists_are_static_and_offline(self):
        self.assertEqual(
            model_registry.list_provider_models("whisper"),
            [{"id": "whisper-local", "display": "whisper-local"}],
        )
        with patch.dict(os.environ, {}):
            os.environ.pop("INWORLD_STT_MODEL", None)
            models = model_registry.list_provider_models("inworld")
        self.assertEqual(
            models,
            [{"id": "inworld/inworld-stt-1", "display": "inworld/inworld-stt-1"}],
        )


class FishOpenRouterListTests(ModelRegistryTestBase):
    @staticmethod
    def _fake_response(payload, status=200):
        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = payload
        resp.text = json.dumps(payload)
        return resp

    def test_fish_static_list_requires_key(self):
        model_registry.FISH_API_KEY = None
        with self.assertRaises(model_registry.ModelRegistryError) as ctx:
            model_registry.list_provider_models("fish")
        self.assertIn("Fish Audio", str(ctx.exception))
        # with key, returns static list containing default
        model_registry.FISH_API_KEY = "fish-key-123"
        models = model_registry.list_provider_models("fish")
        ids = [m["id"] for m in models]
        self.assertIn("s2.1-pro-free", ids)
        self.assertIn("fish-speech-1.5", ids)
        for m in models:
            self.assertEqual(m["id"], m["display"])

    def test_fish_has_key_in_provider_list(self):
        model_registry.FISH_API_KEY = "k"
        providers = model_registry.list_providers()
        fish = [p for p in providers if p["id"] == "fish"][0]
        self.assertTrue(fish["has_key"])
        model_registry.FISH_API_KEY = None
        providers = model_registry.list_providers()
        fish = [p for p in providers if p["id"] == "fish"][0]
        self.assertFalse(fish["has_key"])

    def test_openrouter_list_public_without_key(self):
        payload = {"data": [{"id": "google/gemma-4-31b-it:free", "architecture": {"input_modalities": ["text", "image"]}}]}
        with patch.object(model_registry, "_session") as sess:
            sess.get.return_value = self._fake_response(payload)
            model_registry.OPENROUTER_API_KEY = None
            models = model_registry.list_provider_models("openrouter")
        self.assertEqual(models[0]["id"], "google/gemma-4-31b-it:free")
        # header should not contain Bearer
        args, kwargs = sess.get.call_args
        self.assertNotIn("Bearer", str(kwargs.get("headers") or ""))

    def test_openrouter_list_filters_image_capable(self):
        payload = {
            "data": [
                {"id": "m1", "architecture": {"input_modalities": ["text"]}},
                {"id": "m2", "architecture": {"input_modalities": ["text", "image"]}},
                {"id": "m3", "architecture": {"modality": "text->text"}},
                {"id": "m4", "architecture": {"modality": "text+image->text"}},
                {"id": "m5"},  # no architecture -> keep
            ]
        }
        with patch.object(model_registry, "_session") as sess:
            sess.get.return_value = self._fake_response(payload)
            model_registry.OPENROUTER_API_KEY = "k-or-123"
            models = model_registry.list_provider_models("openrouter")
        ids = [m["id"] for m in models]
        self.assertNotIn("m1", ids)
        self.assertIn("m2", ids)
        self.assertNotIn("m3", ids)
        self.assertIn("m4", ids)
        self.assertIn("m5", ids)

    def test_openrouter_has_key_in_provider_list(self):
        model_registry.OPENROUTER_API_KEY = "k-or"
        providers = model_registry.list_providers()
        ora = [p for p in providers if p["id"] == "openrouter"][0]
        self.assertTrue(ora["has_key"])
        model_registry.OPENROUTER_API_KEY = None
        providers = model_registry.list_providers()
        ora = [p for p in providers if p["id"] == "openrouter"][0]
        self.assertFalse(ora["has_key"])

    def test_openrouter_http_error_scrubbed(self):
        with patch.object(model_registry, "_session") as sess:
            resp = MagicMock()
            resp.status_code = 401
            resp.text = "invalid key Bearer openrouter-sk-super-secret-token-1234567890abc"
            sess.get.return_value = resp
            model_registry.OPENROUTER_API_KEY = "k-or"
            with self.assertRaises(model_registry.ModelRegistryError) as ctx:
                model_registry.list_provider_models("openrouter")
        self.assertNotIn("openrouter-sk-super-secret", str(ctx.exception))
        self.assertIn("401", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
