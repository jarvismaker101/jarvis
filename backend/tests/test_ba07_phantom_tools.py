"""I-2 / BA-07 — phantom-tool prose and advertise-then-forbid.

Verified 2026-10-04: ``_SYSTEM_PROMPT`` spent ~12% of its characters on
three tools the model cannot call (``understand_page``, ``click_element``,
``fill_element``) while simultaneously advertising ``evaluate``'s full
schema AND forbidding it in prose. Now: the phantom sentences are gone,
``evaluate`` left ``_TOOL_ALLOWLIST`` (its schema never reaches the
model), it stays in ``_TOOL_DISPATCH_ALLOWLIST`` for the internal
handlers, and an import-time guard (``_PHANTOM_TOOL_MENTIONS``) fails any
future drift the way the F13 guard does.
"""
import json
import unittest
from unittest.mock import Mock, patch

from backend.services import browser_agent


class PhantomProseAbsentTests(unittest.TestCase):
    def test_no_phantom_tool_named_in_prompt(self):
        prompt = browser_agent._SYSTEM_PROMPT
        for phantom in ("understand_page", "click_element", "fill_element",
                        "evaluate"):
            self.assertNotIn(phantom, prompt)

    def test_no_prohibition_for_unmentioned_tools(self):
        prompt = browser_agent._SYSTEM_PROMPT
        self.assertNotIn("NOT available", prompt)
        self.assertNotIn("never ask for it", prompt)
        self.assertNotIn("no longer exposed to the model", prompt)

    def test_surviving_guidance_names_only_real_tools(self):
        prompt = browser_agent._SYSTEM_PROMPT
        self.assertIn("Use click_mark / click_point / click_text for all clicks",
                      prompt)
        self.assertIn("fill_mark / fill for all fills", prompt)


class PromptSizeTests(unittest.TestCase):
    def test_prompt_shrunk(self):
        # Was 5,648 chars with ~12.5% phantom/prohibition prose; the four
        # deleted blocks remove 677 of prompt characters.
        self.assertLess(len(browser_agent._SYSTEM_PROMPT), 5000)


class ImportGuardTests(unittest.TestCase):
    def test_guard_set_is_empty(self):
        self.assertEqual(browser_agent._PHANTOM_TOOL_MENTIONS, set())

    def test_guard_scans_the_live_prompt(self):
        self.assertIn("click_mark", browser_agent._PROMPT_TOOL_MENTIONS)
        for phantom in ("understand_page", "click_element", "fill_element",
                        "evaluate"):
            self.assertNotIn(phantom, browser_agent._PROMPT_TOOL_MENTIONS)


class AdvertisementTests(unittest.TestCase):
    def test_evaluate_hidden_from_model_but_dispatchable_inside(self):
        self.assertNotIn("evaluate", browser_agent._TOOL_ALLOWLIST)
        self.assertIn("evaluate", browser_agent._TOOL_DISPATCH_ALLOWLIST)
        self.assertNotIn("evaluate",
                         browser_agent._MODEL_DISPATCH_ALLOWLIST)

    def test_model_tool_list_carries_no_evaluate_schema(self):
        seen = {}

        def fake_model_turn(history, tools):
            seen["tools"] = tools
            return ("done", [])

        class ListingClient:
            def list_tools(self):
                return [{"name": "navigate", "description": "n",
                         "input_schema": {"type": "object"}},
                        {"name": "evaluate", "description": "e",
                         "input_schema": {"type": "object"}}]

            def call_tool(self, name, arguments):
                return "ok"

            def connect(self):
                pass

            def close(self):
                pass

        with patch.object(browser_agent, "ensure_brave_mcp_daemon",
                          return_value=True), \
             patch.object(browser_agent, "BraveMcpClient",
                          return_value=ListingClient()), \
             patch.object(browser_agent, "_model_turn",
                          side_effect=fake_model_turn), \
             patch.object(browser_agent, "append_activity_line"), \
             patch.object(browser_agent, "narrate_activity"), \
             patch.object(browser_agent, "truncate_activity_log"):
            browser_agent.run_browser_task("task")
        names = {t["name"] for t in seen["tools"]}
        self.assertIn("navigate", names)
        self.assertNotIn("evaluate", names)


class ModelEvaluateRefusedTests(unittest.TestCase):
    def _run_model_evaluate(self, session):
        history = []
        client = Mock()
        call = {"id": "c1", "name": "evaluate",
                "arguments": {"expression": "document.title"}}
        with patch.object(browser_agent, "append_activity_line"), \
             patch.object(browser_agent, "narrate_activity"):
            browser_agent._run_one_tool(client, history, call, session)
        return client, history

    def test_model_evaluate_refused_without_grant(self):
        client, history = self._run_model_evaluate({})
        client.call_tool.assert_not_called()
        self.assertIn("blocked by policy", history[0]["content"])

    def test_model_evaluate_refused_even_with_grant(self):
        client, history = self._run_model_evaluate(
            {"grants": {"privileged_js"}})
        client.call_tool.assert_not_called()
        self.assertIn("not in the dispatch allowlist",
                      history[0]["content"])


class InternalHandlersWorkTests(unittest.TestCase):
    def test_internal_probes_still_read_the_page(self):
        class ProbeClient:
            def call_tool(self, name, arguments=None, timeout=None):
                expression = (arguments or {}).get("expression", "")
                if expression.strip() == "location.href":
                    return "https://example.com/page"
                return json.dumps({"doc": 111, "dpr": 1,
                                   "url": "https://example.com/page",
                                   "title": "T"})

        client = ProbeClient()
        self.assertEqual(browser_agent._page_url(client),
                         "https://example.com/page")
        state = browser_agent._capture_state(client)
        self.assertEqual(state["url"], "https://example.com/page")


if __name__ == "__main__":
    unittest.main()
