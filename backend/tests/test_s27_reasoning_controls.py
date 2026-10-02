"""S27 - explicit reasoning controls on every voice-path model call.

Acceptance: a model that can think must be TOLD whether to think on the
voice path. Thought text is stripped downstream (F31), but the time it
costs is not - a voice reply waits for it just the same.

  * Gemini (chat + classifier ride the same body builder): thinkingConfig
    by model family - flash-lite 0, bigger Flash a small budget, Pro its
    API floor (thinking cannot be disabled), unknown family nothing.
  * Groq (the classifier's qwen3 fallback): reasoning_effort=none, gated
    to the qwen family so a non-reasoning model never sees the field.
  * OpenRouter (the generic openai-compat chat path): the registry
    snapshot decides the effort, and brain passes it through - the same
    F49/F56 contract the local and Fireworks paths already honour.
"""

import os
import unittest
from unittest.mock import patch

from backend.services import gemini_client, grok_client, model_registry
from backend.core import brain


class GeminiThinkingConfigTests(unittest.TestCase):
    """Acceptance: the thinking budget is explicit, by model family."""

    def test_the_deployed_flash_lite_thinks_zero(self):
        self.assertEqual(gemini_client._thinking_config(),
                         {"thinkingBudget": 0})

    def test_a_bigger_flash_model_gets_a_small_budget(self):
        self.assertEqual(gemini_client._thinking_config("gemini-3.5-flash"),
                         {"thinkingBudget": 512})

    def test_pro_models_get_their_api_floor(self):
        # Pro thinking cannot be disabled: the API floor is 128.
        self.assertEqual(gemini_client._thinking_config("gemini-3.5-pro"),
                         {"thinkingBudget": 128})

    def test_an_unknown_family_sends_nothing(self):
        self.assertIsNone(gemini_client._thinking_config("some-unknown-model"))

    def test_an_env_override_rides_any_model(self):
        with patch.object(gemini_client, "GEMINI_THINKING_BUDGET", "2048"):
            self.assertEqual(
                gemini_client._thinking_config("gemini-3.5-flash-lite"),
                {"thinkingBudget": 2048})

    def test_opting_out_sends_nothing_at_all(self):
        with patch.object(gemini_client, "GEMINI_THINKING_BUDGET", ""):
            self.assertIsNone(gemini_client._thinking_config())

    def test_an_unparseable_env_value_sends_nothing(self):
        with patch.object(gemini_client, "GEMINI_THINKING_BUDGET", "soon"):
            self.assertIsNone(gemini_client._thinking_config())

    def test_the_chat_body_carries_the_config_for_flash_lite(self):
        body = gemini_client._build_chat_body(
            [{"role": "user", "content": "hi"}], 0.7, None,
            "gemini-3.5-flash-lite")
        self.assertEqual(body["generationConfig"]["thinkingConfig"],
                         {"thinkingBudget": 0})

    def test_the_chat_body_stays_clean_for_an_unknown_model(self):
        body = gemini_client._build_chat_body(
            [{"role": "user", "content": "hi"}], 0.7, None, "custom-x")
        self.assertNotIn("thinkingConfig", body["generationConfig"])


class GroqReasoningTests(unittest.TestCase):
    """Acceptance: the qwen fallback never spends its budget thinking."""

    def test_the_default_qwen_model_gets_thinking_none(self):
        self.assertEqual(grok_client._reasoning_effort(), "none")

    def test_an_explicit_qwen_model_gets_thinking_none(self):
        self.assertEqual(grok_client._reasoning_effort("qwen/qwen3.6-27b"),
                         "none")

    def test_a_non_reasoning_model_never_sees_the_field(self):
        self.assertIsNone(grok_client._reasoning_effort("llama-3.3-70b"))

    def test_the_control_can_be_disabled_by_env(self):
        with patch.object(grok_client, "GROQ_REASONING_EFFORT", ""):
            self.assertIsNone(grok_client._reasoning_effort())

    def test_the_ask_grok_payload_carries_the_control(self):
        captured = {}

        def fake_post(data, timeout=None):
            captured["data"] = data
            return {}

        with patch.object(grok_client, "_post_chat_completion",
                          side_effect=fake_post):
            grok_client.ask_grok([{"role": "user", "content": "hi"}])
        self.assertEqual(captured["data"]["reasoning_effort"], "none")

    def test_the_ask_grok_payload_stays_clean_for_llama(self):
        captured = {}

        def fake_post(data, timeout=None):
            captured["data"] = data
            return {}

        with patch.object(grok_client, "_post_chat_completion",
                          side_effect=fake_post):
            grok_client.ask_grok([{"role": "user", "content": "hi"}],
                                 model="llama-3.3-70b")
        self.assertNotIn("reasoning_effort", captured["data"])


class RegistryReasoningSnapshotTests(unittest.TestCase):
    """Acceptance: the snapshot tells the truth about every voice provider."""

    def test_groq_declares_thinking_none(self):
        result = model_registry._reasoning_for("groq", "qwen/qwen3.6-27b",
                                               set())
        self.assertTrue(result["supported"])
        self.assertEqual(result["effort"], "none")

    def test_openrouter_declares_low_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENROUTER_REASONING_EFFORT", None)
            result = model_registry._reasoning_for("openrouter", "x/y", set())
        self.assertTrue(result["supported"])
        self.assertEqual(result["effort"], "low")

    def test_openrouter_effort_follows_the_env(self):
        with patch.dict(os.environ,
                        {"OPENROUTER_REASONING_EFFORT": "medium"}):
            result = model_registry._reasoning_for("openrouter", "x/y", set())
        self.assertTrue(result["supported"])
        self.assertEqual(result["effort"], "medium")

    def test_openrouter_can_opt_out_entirely(self):
        with patch.dict(os.environ, {"OPENROUTER_REASONING_EFFORT": ""}):
            result = model_registry._reasoning_for("openrouter", "x/y", set())
        self.assertFalse(result["supported"])

    def test_the_fireworks_own_default_carve_out_survives(self):
        result = model_registry._reasoning_for("fireworks", "glm-x", set())
        self.assertFalse(result["supported"])

    def test_an_unknown_provider_declares_no_control(self):
        result = model_registry._reasoning_for("mystery", "m", set())
        self.assertFalse(result["supported"])


class BrainChatReasoningWiringTests(unittest.TestCase):
    """Acceptance: the chat stream sends exactly what the snapshot allows."""

    def test_a_supported_snapshot_effort_is_sent(self):
        snapshot = {"reasoning": {"supported": True,
                                  "param": "reasoning_effort",
                                  "effort": "low"}}
        with patch.object(model_registry, "get_model_config",
                          return_value=snapshot):
            self.assertEqual(brain._reasoning_effort_for_chat(), "low")

    def test_an_unsupported_snapshot_sends_nothing(self):
        snapshot = {"reasoning": {"supported": False, "param": None,
                                  "effort": None}}
        with patch.object(model_registry, "get_model_config",
                          return_value=snapshot):
            self.assertIsNone(brain._reasoning_effort_for_chat())

    def test_a_raising_registry_sends_nothing(self):
        def boom(role):
            raise model_registry.ModelRegistryError("invalid")

        with patch.object(model_registry, "get_model_config", side_effect=boom):
            self.assertIsNone(brain._reasoning_effort_for_chat())

    def test_a_supported_snapshot_without_effort_sends_nothing(self):
        snapshot = {"reasoning": {"supported": True,
                                  "param": "reasoning_effort",
                                  "effort": None}}
        with patch.object(model_registry, "get_model_config",
                          return_value=snapshot):
            self.assertIsNone(brain._reasoning_effort_for_chat())


if __name__ == "__main__":
    unittest.main()
