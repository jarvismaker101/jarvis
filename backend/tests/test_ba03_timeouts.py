"""L-15 / BA-03 — split the single shared timeout; adopt core/deadline.py.

Verified 2026-10-03: one 120 s constant did double duty as the MCP wire
timeout AND the LLM HTTP timeout, every MCP tool shared it, and the task
loop rolled its own monotonic comparison while core/deadline.py sat unused.
These tests prove each tool class now carries its own wire budget, the model
has a separate (larger) budget, every budget is sliced to the bound task
deadline, a spent budget sends nothing, and the loop's timeout is the
deadline itself — with unbound callers behaving exactly as configured.
"""
import os
import time
import unittest
from unittest.mock import patch

import requests

from backend import config
from backend.core import deadline as budget
from backend.services import brave_mcp_client
from backend.services import browser_agent
from backend.services.brave_mcp_client import BraveMcpClient


class _WireResp:
    def __init__(self, payload):
        self._payload = payload
        self.headers = {}
        self.status_code = 200
        self.text = ""

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Wire:
    """Fake requests.Session: answers tools/call, records every timeout."""

    def __init__(self):
        self.timeouts = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.timeouts.append(timeout)
        body = dict(json or {})
        method = body.get("method")
        if method == "initialize":
            resp = _WireResp({"jsonrpc": "2.0", "id": body.get("id"),
                              "result": {"protocolVersion": "2025-03-26",
                                         "capabilities": {}}})
            resp.headers = {"mcp-session-id": "sess-1"}
            return resp
        if method == "tools/list":
            return _WireResp({"jsonrpc": "2.0", "id": body.get("id"),
                              "result": {"tools": []}})
        return _WireResp({"jsonrpc": "2.0", "id": body.get("id"),
                          "result": {"content": [{"type": "text",
                                                 "text": "daemon-ok"}]}})

    def delete(self, url, headers=None, timeout=None):
        return _WireResp({})

    def close(self):
        pass


def _client(timeout=120):
    client = BraveMcpClient(base_url="http://x/mcp", token="tok",
                            timeout=timeout)
    client.session = _Wire()
    return client


class ToolClassBudgetTests(unittest.TestCase):
    def test_each_class_carries_its_own_budget(self):
        expected = {
            # probes
            "evaluate": 8, "list_tabs": 8,
            # real-input actions
            "click_locator": 15, "fill_locator": 15, "scroll": 15,
            "screenshot": 15,
            # navigation
            "navigate": 30,
            # file transfer
            "download": 45, "upload_file": 45,
        }
        for name, want in expected.items():
            with self.subTest(tool=name):
                client = _client()
                client.call_tool(name, {})
                self.assertEqual(client.session.timeouts[-1], want)

    def test_unlisted_tool_falls_back_to_client_timeout(self):
        client = _client(timeout=120)
        client.call_tool("some_future_tool", {})
        self.assertEqual(client.session.timeouts[-1], 120)

    def test_explicit_timeout_wins_over_class_budget(self):
        client = _client()
        client.call_tool("evaluate", {}, timeout=3)
        self.assertEqual(client.session.timeouts[-1], 3)
        client.list_tools(timeout=7)
        self.assertEqual(client.session.timeouts[-1], 7)


class DeadlineSlicingTests(unittest.TestCase):
    def test_spent_budget_sends_nothing(self):
        client = _client()
        with budget.bound(budget.Deadline.at(time.monotonic() - 1)):
            with self.assertRaises(budget.BudgetExhausted):
                client.call_tool("evaluate", {})
        self.assertEqual(client.session.timeouts, [])

    def test_class_budget_is_capped_to_what_is_left(self):
        client = _client()
        with budget.bound(budget.Deadline.after(3)):
            client.call_tool("evaluate", {})
        sent = client.session.timeouts[-1]
        self.assertLessEqual(sent, 3.0)
        self.assertGreater(sent, 2.0)

    def test_healthy_budget_leaves_class_budget_whole(self):
        client = _client()
        with budget.bound(budget.Deadline.after(480)):
            client.call_tool("navigate", {})
        self.assertEqual(client.session.timeouts[-1], 30)


class ModelBudgetTests(unittest.TestCase):
    def _ok_post(self, seen, body):
        def post(url, headers=None, params=None, json=None, timeout=None):
            seen["timeout"] = timeout
            resp = _WireResp(body)
            resp.json = lambda: body
            return resp
        return post

    def test_openai_path_uses_the_split_model_budget(self):
        seen = {}
        body = {"choices": [{"message": {"content": "hi",
                                         "tool_calls": []}}]}
        with patch.object(browser_agent._MODEL_SESSION, "post",
                          side_effect=self._ok_post(seen, body)):
            browser_agent._call_openai_compatible(
                "http://x", "k", [], [], model="m", provider="p")
        self.assertEqual(seen["timeout"], config.BROWSER_AGENT_MODEL_TIMEOUT)

    def test_gemini_path_uses_the_split_model_budget(self):
        seen = {}
        with patch.object(browser_agent._MODEL_SESSION, "post",
                          side_effect=self._ok_post(seen, {"candidates": []})):
            browser_agent._call_gemini([], [])
        self.assertEqual(seen["timeout"], config.BROWSER_AGENT_MODEL_TIMEOUT)

    def test_split_budgets_exist_with_expected_defaults(self):
        self.assertEqual(config.BROWSER_AGENT_TOOL_TIMEOUT,
                         int(os.getenv("JARVIS_BROWSER_AGENT_TOOL_TIMEOUT",
                                       "30")))
        self.assertEqual(config.BROWSER_AGENT_MODEL_TIMEOUT,
                         int(os.getenv("JARVIS_BROWSER_AGENT_MODEL_TIMEOUT",
                                       "90")))

    def test_spent_budget_sends_no_model_request(self):
        seen = {}
        with patch.object(browser_agent._MODEL_SESSION, "post",
                          side_effect=self._ok_post(seen, {})):
            with budget.bound(budget.Deadline.at(time.monotonic() - 1)):
                with self.assertRaises(budget.BudgetExhausted):
                    browser_agent._call_openai_compatible(
                        "http://x", "k", [], [], model="m", provider="p")
        self.assertEqual(seen, {})

    def test_model_retries_stop_once_the_budget_is_spent(self):
        attempts = []

        def boom(messages, tools):
            attempts.append(1)
            raise requests.ConnectionError("down")

        with patch.object(browser_agent, "_model_turn", side_effect=boom):
            with budget.bound(budget.Deadline.at(time.monotonic() - 1)):
                with self.assertRaises(RuntimeError):
                    browser_agent._model_turn_with_retries([], [])
        # One attempt, not three: no slice of a spent budget can succeed.
        self.assertEqual(len(attempts), 1)


class TaskDeadlineTests(unittest.TestCase):
    def setUp(self):
        try:
            browser_agent.reset_mcp_pool()
        except Exception:
            pass

    def tearDown(self):
        try:
            browser_agent.reset_mcp_pool()
        except Exception:
            pass

    def test_loop_timeout_is_the_deadline(self):
        expired = budget.Deadline.at(time.monotonic() - 1)
        with patch.object(browser_agent, "_model_turn_with_retries",
                          side_effect=AssertionError("must not be called")):
            result = browser_agent._agent_loop_inner(
                None, "task", time.monotonic(), browser_agent._SwStats(),
                cached_tools=[], deadline=expired)
        self.assertIn("timed out", str(result).lower())

    def test_loop_builds_a_deadline_from_started_when_omitted(self):
        with patch.object(browser_agent, "_model_turn_with_retries",
                          side_effect=AssertionError("must not be called")):
            result = browser_agent._agent_loop_inner(
                None, "task", time.monotonic() - 1000,
                browser_agent._SwStats(), cached_tools=[])
        self.assertIn("timed out", str(result).lower())

    def test_task_wire_calls_are_sliced_to_the_task_budget(self):
        wire = _Wire()
        tool_call = {"id": "call_1", "name": "navigate",
                     "arguments": {"url": "https://example.com"}}
        with patch.dict(os.environ, {"JARVIS_MCP_POOL": "0"}), \
             patch.object(brave_mcp_client.requests, "Session",
                          return_value=wire), \
             patch.object(browser_agent, "ensure_brave_mcp_daemon",
                          return_value=True), \
             patch.object(browser_agent, "_model_turn",
                          side_effect=[(None, [tool_call]),
                                       ("Done.", [])]), \
             patch.object(browser_agent, "append_activity_line"), \
             patch.object(browser_agent, "narrate_activity"), \
             patch.object(browser_agent, "truncate_activity_log"):
            result = browser_agent.run_browser_task("open the page")
        self.assertEqual(result.status, "completed")
        # navigate's class budget (30 s) with ~480 s left: sent whole.
        self.assertIn(30, wire.timeouts)


if __name__ == "__main__":
    unittest.main()
