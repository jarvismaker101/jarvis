import importlib
import json
import os
import re
import time
import unittest
from unittest.mock import Mock, patch

from backend import config
from backend.api import routes
from backend.core import brain
from backend.services import brave_mcp_client
from backend.services import browser_agent
from backend.services import opencode_client
from backend.services.task_result import TaskResult


def _wait_until(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class FakeResponse:
    """requests.post/delete stand-in: fixed payload + headers."""

    def __init__(self, payload=None, headers=None, status=200, text=""):
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}
        self.status_code = status
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("HTTP %s" % self.status_code)
        return None

    def json(self):
        return self._payload


_INIT_RESULT = {"jsonrpc": "2.0", "id": 1, "result": {
    "protocolVersion": "2025-03-26",
    "capabilities": {},
    "serverInfo": {"name": "brave-control", "version": "1.0.0"},
}}


class FakeClient:
    """Happy-path MCP client stand-in used by the agent-loop tests."""

    def __init__(self):
        self.tool_calls = []
        self.closed = False

    def connect(self):
        return None

    def close(self):
        self.closed = True

    def list_tools(self):
        return [{"name": "navigate", "description": "d", "input_schema": {"type": "object"}}]

    def call_tool(self, name, arguments):
        self.tool_calls.append((name, arguments))
        return "Navigated to https://example.com"


class FailingClient(FakeClient):
    """Client whose call_tool fails the first *errors* attempts."""

    def __init__(self, errors=3):
        super().__init__()
        self._errors = errors
        self.attempts = 0

    def call_tool(self, name, arguments):
        self.attempts += 1
        if self.attempts <= self._errors:
            raise RuntimeError("browser died")
        return "recovered ok"


class BraveMcpClientTests(unittest.TestCase):
    """Raw JSON-RPC client: handshake, session echo, tool parsing."""

    def _session_with(self, responses):
        fake_session = Mock()
        fake_session.post.side_effect = responses
        return fake_session

    def test_connect_captures_session_header_and_initializes(self):
        fake_session = self._session_with([
            FakeResponse(_INIT_RESULT, headers={"mcp-session-id": "sess-abc"}),
            FakeResponse({}),
        ])
        with patch.object(brave_mcp_client.requests, "Session",
                          return_value=fake_session):
            client = brave_mcp_client.BraveMcpClient(
                base_url="http://x/mcp", token="tok", timeout=5
            )
            client.connect()

        self.assertEqual(client._session_id, "sess-abc")
        self.assertEqual(fake_session.post.call_count, 2)
        init_call, notif_call = fake_session.post.call_args_list
        self.assertEqual(init_call.kwargs["headers"]["Authorization"], "Bearer tok")
        self.assertEqual(init_call.kwargs["headers"]["Accept"],
                         "application/json, text/event-stream")
        self.assertNotIn("mcp-session-id", init_call.kwargs["headers"])
        self.assertEqual(init_call.kwargs["json"]["method"], "initialize")
        self.assertEqual(init_call.kwargs["json"]["params"]["protocolVersion"],
                         "2025-03-26")
        self.assertEqual(init_call.kwargs["json"]["params"]["clientInfo"]["name"],
                         "jarvis-browser-agent")
        # Notification: no id, session header echoed.
        self.assertEqual(notif_call.kwargs["json"]["method"],
                         "notifications/initialized")
        self.assertNotIn("id", notif_call.kwargs["json"])
        self.assertEqual(notif_call.kwargs["headers"]["mcp-session-id"], "sess-abc")
        self.assertEqual(notif_call.kwargs["headers"]["Authorization"], "Bearer tok")

    def test_call_tool_echoes_session_and_concatenates_blocks(self):
        fake_session = self._session_with([
            FakeResponse(_INIT_RESULT, headers={"mcp-session-id": "s1"}),
            FakeResponse({}),
            FakeResponse({"jsonrpc": "2.0", "id": 2, "result": {"content": [
                {"type": "text", "text": "part one"},
                {"type": "image", "data": "..."},
                {"type": "text", "text": "part two"},
            ]}}),
        ])
        with patch.object(brave_mcp_client.requests, "Session",
                          return_value=fake_session):
            client = brave_mcp_client.BraveMcpClient(
                base_url="http://x/mcp", token="tok"
            )
            client.connect()
            text = client.call_tool("understand_page", {})

        call = fake_session.post.call_args_list[2]
        self.assertEqual(call.kwargs["headers"]["mcp-session-id"], "s1")
        self.assertEqual(call.kwargs["headers"]["Authorization"], "Bearer tok")
        self.assertEqual(call.kwargs["json"]["method"], "tools/call")
        self.assertEqual(call.kwargs["json"]["params"]["name"], "understand_page")
        self.assertEqual(text, "part one\n[image omitted]\npart two")

    def test_jsonrpc_error_raises_runtime_error(self):
        fake_session = self._session_with([
            FakeResponse(_INIT_RESULT, headers={"mcp-session-id": "s1"}),
            FakeResponse({}),
            FakeResponse({"jsonrpc": "2.0", "id": 2,
                          "error": {"code": -32602, "message": "invalid params"}}),
        ])
        with patch.object(brave_mcp_client.requests, "Session",
                          return_value=fake_session):
            client = brave_mcp_client.BraveMcpClient(
                base_url="http://x/mcp", token="tok"
            )
            client.connect()
            with self.assertRaises(RuntimeError) as ctx:
                client.call_tool("click_element", {})
        self.assertIn("invalid params", str(ctx.exception))

    def test_list_tools_parses_descriptors(self):
        fake_session = self._session_with([
            FakeResponse(_INIT_RESULT, headers={"mcp-session-id": "s1"}),
            FakeResponse({}),
            FakeResponse({"jsonrpc": "2.0", "id": 2, "result": {"tools": [
                {"name": "open_brave", "description": "Launch a visible Brave browser",
                 "inputSchema": {"type": "object", "properties": {}}},
                {"name": "navigate", "description": "Navigate",
                 "inputSchema": {"type": "object",
                                 "properties": {"url": {"type": "string"}}}},
            ]}}),
        ])
        with patch.object(brave_mcp_client.requests, "Session",
                          return_value=fake_session):
            client = brave_mcp_client.BraveMcpClient(
                base_url="http://x/mcp", token="tok"
            )
            client.connect()
            tools = client.list_tools()

        self.assertEqual(len(tools), 2)
        self.assertEqual(tools[0]["name"], "open_brave")
        self.assertEqual(tools[1]["name"], "navigate")
        self.assertEqual(tools[1]["input_schema"]["properties"]["url"]["type"],
                         "string")

    def test_sse_response_is_parsed_for_matching_id(self):
        sse_body = (
            "event: message\n"
            'data: {"jsonrpc":"2.0","method":"notifications/progress",'
            '"params":{}}\n\n'
            "event: message\n"
            'data: {"jsonrpc":"2.0","id":2,"result":{"content":'
            '[{"type":"text","text":"sse part"}]}}\n\n'
        )
        fake_session = self._session_with([
            FakeResponse(_INIT_RESULT, headers={"mcp-session-id": "s1"}),
            FakeResponse({}),
            FakeResponse(None, headers={"Content-Type": "text/event-stream"},
                         text=sse_body),
        ])
        with patch.object(brave_mcp_client.requests, "Session",
                          return_value=fake_session):
            client = brave_mcp_client.BraveMcpClient(
                base_url="http://x/mcp", token="tok"
            )
            client.connect()
            text = client.call_tool("understand_page", {})
        self.assertEqual(text, "sse part")

    def test_sse_response_without_matching_id_raises(self):
        sse_body = 'data: {"jsonrpc":"2.0","id":99,"result":{}}\n\n'
        fake_session = self._session_with([
            FakeResponse(_INIT_RESULT, headers={"mcp-session-id": "s1"}),
            FakeResponse({}),
            FakeResponse(None, headers={"Content-Type": "text/event-stream"},
                         text=sse_body),
        ])
        with patch.object(brave_mcp_client.requests, "Session",
                          return_value=fake_session):
            client = brave_mcp_client.BraveMcpClient(
                base_url="http://x/mcp", token="tok"
            )
            client.connect()
            with self.assertRaises(RuntimeError) as ctx:
                client.call_tool("click_element", {})
        self.assertIn("no message", str(ctx.exception))

    def test_sse_error_with_null_id_still_surfaces(self):
        sse_body = ('data: {"jsonrpc":"2.0","id":null,'
                    '"error":{"code":-32000,"message":"bad session"}}\n\n')
        fake_session = self._session_with([
            FakeResponse(_INIT_RESULT, headers={"mcp-session-id": "s1"}),
            FakeResponse({}),
            FakeResponse(None, headers={"Content-Type": "text/event-stream"},
                         text=sse_body),
        ])
        with patch.object(brave_mcp_client.requests, "Session",
                          return_value=fake_session):
            client = brave_mcp_client.BraveMcpClient(
                base_url="http://x/mcp", token="tok"
            )
            client.connect()
            with self.assertRaises(RuntimeError) as ctx:
                client.call_tool("click_element", {})
        self.assertIn("bad session", str(ctx.exception))

    def test_close_deletes_with_session_header(self):
        fake_session = self._session_with([
            FakeResponse(_INIT_RESULT, headers={"mcp-session-id": "s1"}),
            FakeResponse({}),
        ])
        with patch.object(brave_mcp_client.requests, "Session",
                          return_value=fake_session):
            client = brave_mcp_client.BraveMcpClient(
                base_url="http://x/mcp", token="tok"
            )
            client.connect()
            client.close()
        delete_call = fake_session.delete.call_args
        self.assertEqual(delete_call.kwargs["headers"]["mcp-session-id"], "s1")
        self.assertEqual(delete_call.kwargs["headers"]["Authorization"], "Bearer tok")


class SchemaSanitizerTests(unittest.TestCase):
    """Gemini rejects several JSON-Schema keys; the sanitizer drops them."""

    def test_sanitizer_keeps_only_allowed_keys_recursively(self):
        raw = {
            "type": "object",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "additionalProperties": False,
            "description": "Args for navigate",
            "properties": {
                "url": {"type": "string", "description": "Full URL to navigate to"},
            },
            "required": ["url"],
            "enum": ["a", "b"],
        }
        clean = browser_agent._sanitize_schema(raw)
        self.assertEqual(set(clean.keys()),
                         {"type", "description", "properties", "required", "enum"})
        self.assertEqual(clean["properties"]["url"],
                         {"type": "string", "description": "Full URL to navigate to"})

    def test_sanitizer_handles_items(self):
        raw = {"type": "array", "items": {"type": "string", "$schema": "x"}}
        clean = browser_agent._sanitize_schema(raw)
        self.assertEqual(clean["items"], {"type": "string"})


class AgentLoopTests(unittest.TestCase):
    """run_browser_task: happy path, retry policy, caps, timeout, failures."""

    def _run(self, task, client=None, model_turns=None):
        client = client or FakeClient()
        patchers = [
            patch.object(browser_agent, "ensure_brave_mcp_daemon",
                         return_value=True),
            patch.object(browser_agent, "BraveMcpClient", return_value=client),
            patch.object(browser_agent, "_model_turn",
                         side_effect=model_turns),
            patch.object(browser_agent, "append_activity_line"),
            patch.object(browser_agent, "narrate_activity"),
            patch.object(browser_agent, "truncate_activity_log"),
        ]
        with patchers[0], patchers[1], patchers[2], patchers[3], \
             patchers[4], patchers[5]:
            result = browser_agent.run_browser_task(task)
        return result, client

    def _capture(self, task, client=None, model_turns=None):
        client = client or FakeClient()
        with patch.object(browser_agent, "ensure_brave_mcp_daemon",
                          return_value=True), \
             patch.object(browser_agent, "BraveMcpClient",
                          return_value=client), \
             patch.object(browser_agent, "_model_turn",
                          side_effect=model_turns), \
             patch.object(browser_agent, "append_activity_line") as log_line, \
             patch.object(browser_agent, "narrate_activity") as narrate:
            result = browser_agent.run_browser_task(task)
        return result, client, log_line, narrate

    def test_happy_path_returns_final_text_and_logs(self):
        tool_call = {"id": "call_1", "name": "navigate",
                     "arguments": {"url": "https://example.com"}}
        result, client, log_line, narrate = self._capture(
            "find the price",
            model_turns=[
                (None, [tool_call]),
                ("Done: found the price.", []),
            ],
        )
        self.assertEqual(result, "Done: found the price.")
        self.assertEqual(client.tool_calls,
                         [("navigate", {"url": "https://example.com"})])
        self.assertTrue(client.closed)
        lines = [call.args[0] for call in log_line.call_args_list]
        self.assertTrue(any(line.lstrip().startswith("=== ") for line in lines))
        self.assertTrue(any(line.startswith("TOOL navigate ")
                            and "https://example.com" in line for line in lines))
        self.assertTrue(any(line.startswith("RESULT ok: ") for line in lines))
        narrate.assert_called_once_with("Navigating")

    def test_tool_fails_three_times_error_becomes_result_and_loop_continues(self):
        client = FailingClient(errors=3)
        # F17: use an ADVERTISED read tool. A hidden daemon primitive such as
        # read_file is no longer dispatchable by the model at all.
        tool_call = {"id": "call_1", "name": "list_tabs", "arguments": {}}
        captured = []

        def fake_model_turn(history, tools):
            captured.append([dict(m) for m in history])
            if len(captured) == 1:
                return None, [tool_call]
            return "Done without the list.", []

        with patch.object(browser_agent, "ensure_brave_mcp_daemon",
                          return_value=True), \
             patch.object(browser_agent, "BraveMcpClient",
                          return_value=client), \
             patch.object(browser_agent, "_model_turn",
                          side_effect=fake_model_turn), \
             patch.object(browser_agent.time, "sleep"), \
             patch.object(browser_agent, "append_activity_line") as log_line, \
             patch.object(browser_agent, "narrate_activity"):
            result = browser_agent.run_browser_task("list the tabs")
        # The optional tool failed 3x, but the task CONTINUES and the model
        # adapts - the error was delivered AS the tool result message.
        # The task is partial, not completed: a tool failed along the way.
        self.assertEqual(result, "Done without the list.")
        self.assertEqual(result.status, "partial")
        self.assertTrue(any("list_tabs" in record for record in result.evidence))
        self.assertEqual(client.attempts, 3)
        self.assertTrue(client.closed)
        tool_msgs = [m for m in captured[1] if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("browser died", tool_msgs[0]["content"])
        self.assertIn("tool failed after retries", tool_msgs[0]["content"])
        self.assertIn("list_tabs", tool_msgs[0]["content"])
        self.assertTrue(any(call.args[0].startswith("RESULT list_tabs: FAILED ")
                            for call in log_line.call_args_list))

    def test_hidden_daemon_primitive_is_not_dispatchable(self):
        """F17: naming a tool the model was never shown must not dispatch it."""
        client = FailingClient(errors=1)
        tool_call = {"id": "call_1", "name": "read_file",
                     "arguments": {"path": "x.png"}}
        captured = []

        def fake_model_turn(history, tools):
            captured.append([dict(m) for m in history])
            if len(captured) == 1:
                return None, [tool_call]
            return "Done.", []

        with patch.object(browser_agent, "ensure_brave_mcp_daemon",
                          return_value=True), \
             patch.object(browser_agent, "BraveMcpClient",
                          return_value=client), \
             patch.object(browser_agent, "_model_turn",
                          side_effect=fake_model_turn), \
             patch.object(browser_agent.time, "sleep"), \
             patch.object(browser_agent, "append_activity_line"), \
             patch.object(browser_agent, "narrate_activity"):
            browser_agent.run_browser_task("read a file")
        tool_msgs = [m for m in captured[1] if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("blocked by policy", tool_msgs[0]["content"])
        # The primitive never ran, so the transport was never attempted.
        self.assertEqual(client.attempts, 0)

    def test_tool_fails_once_then_recovers(self):
        # navigate is a MUTATION tool: single attempt, no auto-retry. The
        # model still adapts on the next turn, but the task is partial.
        client = FailingClient(errors=1)
        tool_call = {"id": "call_1", "name": "navigate",
                     "arguments": {"url": "https://example.com"}}
        result, client = self._run(
            "go",
            client=client,
            model_turns=[(None, [tool_call]), ("Done.", [])],
        )
        self.assertEqual(result, "Done.")
        self.assertEqual(result.status, "partial")
        self.assertTrue(any("navigate" in record for record in result.evidence))
        self.assertEqual(client.attempts, 1)

    def test_max_steps_exceeded_fails(self):
        tool_call = {"id": "c", "name": "evaluate",
                     "arguments": {"expression": "1+1"}}
        client = FakeClient()
        with patch.object(browser_agent, "ensure_brave_mcp_daemon",
                          return_value=True), \
             patch.object(browser_agent, "BraveMcpClient",
                          return_value=client), \
             patch.object(browser_agent, "_model_turn",
                          return_value=(None, [tool_call])), \
             patch.object(config, "BROWSER_AGENT_MAX_STEPS", 3), \
             patch.object(browser_agent, "append_activity_line"), \
             patch.object(browser_agent, "narrate_activity"):
            result = browser_agent.run_browser_task("loop")
        self.assertTrue(result.startswith("TASK NOT COMPLETED"))
        self.assertIn("exceeded max steps (3)", result)

    def test_timeout_fails_with_message(self):
        with patch.object(browser_agent, "ensure_brave_mcp_daemon",
                          return_value=True), \
             patch.object(browser_agent, "BraveMcpClient",
                          return_value=FakeClient()), \
             patch.object(browser_agent, "_model_turn",
                          return_value=(None, [{"id": "c", "name": "evaluate",
                                                "arguments": {}}])), \
             patch.object(browser_agent.time, "monotonic",
                          side_effect=[0.0, 999.0]), \
             patch.object(config, "BROWSER_AGENT_TIMEOUT", 180), \
             patch.object(browser_agent, "append_activity_line"), \
             patch.object(browser_agent, "narrate_activity"):
            result = browser_agent.run_browser_task("slow")
        self.assertTrue(result.startswith("TASK NOT COMPLETED"))
        self.assertIn("timed out after 180s", result)

    def test_daemon_failure_returns_not_completed_without_model_calls(self):
        with patch.object(browser_agent, "ensure_brave_mcp_daemon",
                          return_value=False), \
             patch.object(browser_agent, "BraveMcpClient") as client_cls, \
             patch.object(browser_agent, "_model_turn") as model:
            result = browser_agent.run_browser_task("anything")
        self.assertTrue(result.startswith("TASK NOT COMPLETED"))
        self.assertIn("daemon", result)
        client_cls.assert_not_called()
        model.assert_not_called()

    def test_connect_failure_returns_not_completed_without_model_calls(self):
        with patch.object(browser_agent, "ensure_brave_mcp_daemon",
                          return_value=True), \
             patch.object(browser_agent, "BraveMcpClient",
                          side_effect=RuntimeError("daemon unreachable")), \
             patch.object(browser_agent, "_model_turn") as model:
            result = browser_agent.run_browser_task("anything")
        self.assertTrue(result.startswith("TASK NOT COMPLETED"))
        self.assertIn("daemon unreachable", result)
        model.assert_not_called()

    def test_list_tools_failure_returns_not_completed_without_model_calls(self):
        client = FakeClient()
        client.list_tools = lambda: (_ for _ in ()).throw(
            RuntimeError("tools/list exploded"))
        with patch.object(browser_agent, "ensure_brave_mcp_daemon",
                          return_value=True), \
             patch.object(browser_agent, "BraveMcpClient",
                          return_value=client), \
             patch.object(browser_agent, "_model_turn") as model, \
             patch.object(browser_agent, "append_activity_line"), \
             patch.object(browser_agent, "narrate_activity"):
            result = browser_agent.run_browser_task("anything")
        self.assertTrue(result.startswith("TASK NOT COMPLETED"))
        self.assertIn("tools/list exploded", result)
        model.assert_not_called()


class TaskResultLoopTests(unittest.TestCase):
    """G1 Round 1 (F03/F05): completed/partial statuses, read-vs-mutation
    retry policy, and the identical-action failure cap."""

    def _run(self, task, client=None, model_turns=None):
        client = client or FakeClient()
        with patch.object(browser_agent, "ensure_brave_mcp_daemon",
                          return_value=True), \
             patch.object(browser_agent, "BraveMcpClient", return_value=client), \
             patch.object(browser_agent, "_model_turn", side_effect=model_turns), \
             patch.object(browser_agent, "append_activity_line"), \
             patch.object(browser_agent, "narrate_activity"), \
             patch.object(browser_agent, "truncate_activity_log"):
            result = browser_agent.run_browser_task(task)
        return result, client

    def test_completed_when_no_tool_failures(self):
        tool_call = {"id": "call_1", "name": "navigate",
                     "arguments": {"url": "https://example.com"}}
        result, _client = self._run(
            "find the price",
            model_turns=[(None, [tool_call]), ("Done: found the price.", [])],
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result, "Done: found the price.")
        self.assertEqual(result.evidence, [])
        self.assertTrue(result)

    def test_mutation_tool_not_retried_on_transport_error(self):
        client = FailingClient(errors=99)
        history = []
        call = {"id": "c1", "name": "navigate",
                "arguments": {"url": "https://example.com"}}
        with patch.object(browser_agent.time, "sleep") as sleep, \
             patch.object(browser_agent, "append_activity_line"), \
             patch.object(browser_agent, "narrate_activity"):
            browser_agent._run_one_tool(client, history, call, {})
        self.assertEqual(client.attempts, 1)
        sleep.assert_not_called()
        self.assertIn("tool failed:", history[0]["content"])
        self.assertNotIn("after retries", history[0]["content"])
        self.assertIn("navigate", history[0]["content"])

    def test_read_tool_still_retried_three_times(self):
        client = FailingClient(errors=99)
        history = []
        call = {"id": "c1", "name": "list_tabs", "arguments": {}}
        with patch.object(browser_agent.time, "sleep"), \
             patch.object(browser_agent, "append_activity_line"), \
             patch.object(browser_agent, "narrate_activity"):
            browser_agent._run_one_tool(client, history, call, {})
        self.assertEqual(client.attempts, 3)
        self.assertIn("tool failed after retries", history[0]["content"])
        self.assertIn("list_tabs", history[0]["content"])

    def test_second_identical_failure_appends_note(self):
        client = FailingClient(errors=99)
        session = {}
        call = {"id": "c1", "name": "navigate",
                "arguments": {"url": "https://example.com"}}
        histories = []
        with patch.object(browser_agent.time, "sleep"), \
             patch.object(browser_agent, "append_activity_line"), \
             patch.object(browser_agent, "narrate_activity"):
            for _ in range(2):
                history = []
                browser_agent._run_one_tool(client, history, call, session)
                histories.append(history)
        self.assertNotIn("NOTE", histories[0][0]["content"])
        self.assertIn("NOTE: this action has now failed 2 times",
                      histories[1][0]["content"])

    def test_third_identical_failure_blocked_without_executing(self):
        client = FailingClient(errors=99)
        session = {}
        call = {"id": "c1", "name": "navigate",
                "arguments": {"url": "https://example.com"}}
        with patch.object(browser_agent.time, "sleep"), \
             patch.object(browser_agent, "append_activity_line"), \
             patch.object(browser_agent, "narrate_activity"):
            browser_agent._run_one_tool(client, [], call, session)
            browser_agent._run_one_tool(client, [], call, session)
            history = []
            blocked = browser_agent._run_one_tool(client, history, call, session)
        # The 3rd identical action never executed: only 2 real attempts.
        self.assertEqual(client.attempts, 2)
        self.assertEqual(blocked, "blocked: repeated failure of navigate")
        self.assertIn("action blocked after 3 identical failures",
                      history[0]["content"])

    def test_success_resets_failure_counter(self):
        client = FailingClient(errors=1)
        session = {}
        call = {"id": "c1", "name": "navigate",
                "arguments": {"url": "https://example.com"}}
        histories = []
        with patch.object(browser_agent.time, "sleep"), \
             patch.object(browser_agent, "append_activity_line"), \
             patch.object(browser_agent, "narrate_activity"):
            for _ in range(3):
                history = []
                browser_agent._run_one_tool(client, history, call, session)
                histories.append(history)
        # fail, succeed (resets), fail again as a first failure: no NOTE.
        self.assertEqual(client.attempts, 3)
        self.assertNotIn("NOTE", histories[2][0]["content"])

    def test_third_identical_failure_blocked_and_task_ends_partial(self):
        client = FailingClient(errors=99)
        tool_call = {"id": "c1", "name": "navigate",
                     "arguments": {"url": "https://example.com"}}
        histories = []

        def fake_model_turn(history, tools):
            histories.append([dict(m) for m in history])
            return None, [dict(tool_call)]

        with patch.object(browser_agent, "ensure_brave_mcp_daemon",
                          return_value=True), \
             patch.object(browser_agent, "BraveMcpClient", return_value=client), \
             patch.object(browser_agent, "_model_turn",
                          side_effect=fake_model_turn), \
             patch.object(browser_agent.time, "sleep"), \
             patch.object(browser_agent, "append_activity_line") as log_line, \
             patch.object(browser_agent, "narrate_activity"), \
             patch.object(browser_agent, "truncate_activity_log"):
            result = browser_agent.run_browser_task("open the page")

        # Only the first two identical actions executed; the 3rd was blocked.
        self.assertEqual(client.attempts, 2)
        self.assertEqual(result.status, "partial")
        self.assertIn("blocked: repeated failure of navigate", result.evidence)
        lines = [call.args[0] for call in log_line.call_args_list]
        self.assertTrue(any(line.startswith("RESULT partial: ")
                            for line in lines))

    def test_stop_status_and_byte_identical_string(self):
        browser_agent._STOP_REQUESTED.set()
        client = FakeClient()
        model = Mock()
        try:
            with patch.object(browser_agent, "_model_turn", side_effect=model), \
                 patch.object(browser_agent, "append_activity_line"), \
                 patch.object(browser_agent, "narrate_activity") as narrate:
                result = browser_agent._agent_loop(client, "task", time.monotonic())
        finally:
            browser_agent._STOP_REQUESTED.clear()
        self.assertEqual(result.status, "stopped")
        self.assertEqual(str(result), "Stopped per your request.")
        self.assertTrue(result)
        model.assert_not_called()

    def test_timeout_status_and_byte_identical_string(self):
        with patch.object(browser_agent, "ensure_brave_mcp_daemon",
                          return_value=True), \
             patch.object(browser_agent, "BraveMcpClient",
                          return_value=FakeClient()), \
             patch.object(browser_agent, "_model_turn",
                          return_value=(None, [{"id": "c", "name": "evaluate",
                                                "arguments": {}}])), \
             patch.object(browser_agent.time, "monotonic",
                          side_effect=[0.0, 999.0]), \
             patch.object(config, "BROWSER_AGENT_TIMEOUT", 180), \
             patch.object(browser_agent, "append_activity_line"), \
             patch.object(browser_agent, "narrate_activity"):
            result = browser_agent.run_browser_task("slow")
        self.assertEqual(result.status, "failed")
        self.assertFalse(result)
        self.assertEqual(
            str(result),
            "TASK NOT COMPLETED. Error: task timed out after 180s")

    def test_budget_exhaustion_with_completion_claim_stays_completed(self):
        tool_call = {"id": "c", "name": "evaluate",
                     "arguments": {"expression": "1+1"}}
        seen = {"budget_note": False}

        def fake_model(history, tools):
            for message in history:
                if message.get("role") == "user" and "budget" in (message.get("content") or ""):
                    seen["budget_note"] = True
            if seen["budget_note"]:
                return ("Done: opened the page.", [])
            return (None, [tool_call])

        with patch.object(browser_agent, "ensure_brave_mcp_daemon",
                          return_value=True), \
             patch.object(browser_agent, "BraveMcpClient",
                          return_value=FakeClient()), \
             patch.object(browser_agent, "_model_turn", side_effect=fake_model), \
             patch.object(config, "BROWSER_AGENT_MAX_STEPS", 2), \
             patch.object(browser_agent, "append_activity_line") as log_line, \
             patch.object(browser_agent, "narrate_activity"), \
             patch.object(browser_agent, "truncate_activity_log"):
            result = browser_agent.run_browser_task("loop")
        self.assertTrue(seen["budget_note"])
        self.assertEqual(result.status, "completed")
        self.assertEqual(result, "Done: opened the page.")
        lines = [call.args[0] for call in log_line.call_args_list]
        self.assertTrue(any(line.startswith("RESULT ok: ") for line in lines))

    def test_budget_exhaustion_with_negated_claim_stays_partial(self):
        # Audit LOW flag: 'success was not achieved' matches the prefix but
        # the immediate negation cancels it - partial, not completed.
        tool_call = {"id": "c", "name": "evaluate",
                     "arguments": {"expression": "1+1"}}
        seen = {"budget_note": False}

        def fake_model(history, tools):
            for message in history:
                if message.get("role") == "user" and "budget" in (message.get("content") or ""):
                    seen["budget_note"] = True
            if seen["budget_note"]:
                return ("Success was not achieved, the page did not load.", [])
            return (None, [tool_call])

        with patch.object(browser_agent, "ensure_brave_mcp_daemon",
                          return_value=True), \
             patch.object(browser_agent, "BraveMcpClient",
                          return_value=FakeClient()), \
             patch.object(browser_agent, "_model_turn", side_effect=fake_model), \
             patch.object(config, "BROWSER_AGENT_MAX_STEPS", 2), \
             patch.object(browser_agent, "append_activity_line") as log_line, \
             patch.object(browser_agent, "narrate_activity"), \
             patch.object(browser_agent, "truncate_activity_log"):
            result = browser_agent.run_browser_task("loop")
        self.assertTrue(seen["budget_note"])
        self.assertEqual(result.status, "partial")
        self.assertEqual(result, "Success was not achieved, the page did not load.")
        self.assertIn("step budget exhausted", result.evidence)
        lines = [call.args[0] for call in log_line.call_args_list]
        self.assertTrue(any(line.startswith("RESULT partial: ") for line in lines))

    def test_states_completion_rejects_negations_accepts_genuine(self):
        self.assertTrue(browser_agent._states_completion("Done: opened the page."))
        self.assertTrue(browser_agent._states_completion("done — the file is created."))
        self.assertTrue(browser_agent._states_completion("Completed the task"))
        self.assertTrue(browser_agent._states_completion("success, the form was submitted"))
        self.assertTrue(browser_agent._states_completion("Finished all steps."))
        self.assertFalse(browser_agent._states_completion("Success was not achieved."))
        self.assertFalse(browser_agent._states_completion("Done: nothing could be saved, unable to write."))
        self.assertFalse(browser_agent._states_completion("Completed nothing - failed at login."))
        self.assertFalse(browser_agent._states_completion("Partial: I opened the page."))
        self.assertFalse(browser_agent._states_completion(""))

    def test_clip_result_marks_truncation(self):
        clipped = browser_agent._clip_result("a" * 20000)
        self.assertTrue(clipped.endswith("...[truncated]"))
        self.assertEqual(len(clipped),
                         15000 + len("...[truncated]"))
        self.assertEqual(browser_agent._clip_result("short"), "short")


class ModelAdapterTests(unittest.TestCase):
    """Provider adapters post the right payloads and parse responses."""

    def test_fireworks_adapter_posts_chat_completions_with_tools(self):
        fake_resp = FakeResponse({"choices": [{"message": {
            "content": "hi",
            "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "navigate",
                "arguments": '{"url": "https://example.com"}',
            }}],
        }}]})
        messages = [{"role": "user", "content": "go"}]
        tools = [{"name": "navigate", "description": "Navigate",
                  "input_schema": {"type": "object",
                                   "properties": {"url": {"type": "string"}}}}]
        with patch.object(browser_agent._MODEL_SESSION, "post",
                          return_value=fake_resp) as post, \
              patch.object(config, "BROWSER_AGENT_MODEL", "m/pro"), \
              patch.object(config, "BROWSER_AGENT_PROVIDER", "fireworks"), \
              patch.object(config, "BROWSER_AGENT_REASONING_EFFORT", "medium"):
            text, calls = browser_agent._call_openai_compatible(
                "http://fw/v1/chat/completions", "k-fw", messages, tools
            )
        self.assertEqual(text, "hi")
        self.assertEqual(calls[0]["name"], "navigate")
        self.assertEqual(calls[0]["arguments"], {"url": "https://example.com"})
        kwargs = post.call_args.kwargs
        self.assertIn("chat/completions", post.call_args.args[0])
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer k-fw")
        self.assertEqual(kwargs["json"]["model"], "m/pro")
        self.assertEqual(kwargs["json"]["reasoning_effort"], "medium")
        self.assertEqual(kwargs["json"]["tools"][0]["type"], "function")
        self.assertEqual(kwargs["json"]["tools"][0]["function"]["name"],
                         "navigate")
        self.assertEqual(kwargs["json"]["tool_choice"], "auto")

    def test_reasoning_effort_only_sent_for_fireworks(self):
        fake_resp = FakeResponse({"choices": [{"message": {"content": "hi"}}]})
        messages = [{"role": "user", "content": "go"}]
        tools = [{"name": "navigate", "description": "Navigate",
                  "input_schema": {"type": "object"}}]
        with patch.object(browser_agent._MODEL_SESSION, "post",
                          return_value=fake_resp) as post, \
              patch.object(config, "BROWSER_AGENT_PROVIDER", "groq"), \
              patch.object(config, "BROWSER_AGENT_REASONING_EFFORT", "medium"):
            browser_agent._call_openai_compatible(
                "http://groq/v1/chat/completions", "k-g", messages, tools
            )
        self.assertNotIn("reasoning_effort", post.call_args.kwargs["json"])

    def test_reasoning_effort_never_sent_for_minimax_models(self):
        fake_resp = FakeResponse({"choices": [{"message": {"content": "hi"}}]})
        messages = [{"role": "user", "content": "go"}]
        tools = [{"name": "navigate", "description": "Navigate",
                  "input_schema": {"type": "object"}}]
        with patch.object(browser_agent._MODEL_SESSION, "post",
                          return_value=fake_resp) as post, \
              patch.object(config, "BROWSER_AGENT_PROVIDER", "fireworks"), \
              patch.object(config, "BROWSER_AGENT_MODEL",
                           "accounts/fireworks/models/minimax-m3"), \
              patch.object(config, "BROWSER_AGENT_REASONING_EFFORT", "medium"):
            browser_agent._call_openai_compatible(
                "https://api.fireworks.ai/inference/v1/chat/completions",
                "k-fw", messages, tools
            )
        self.assertNotIn("reasoning_effort", post.call_args.kwargs["json"])

    def test_reasoning_effort_sent_for_fireworks_non_minimax(self):
        fake_resp = FakeResponse({"choices": [{"message": {"content": "hi"}}]})
        messages = [{"role": "user", "content": "go"}]
        tools = [{"name": "navigate", "description": "Navigate",
                  "input_schema": {"type": "object"}}]
        with patch.object(browser_agent._MODEL_SESSION, "post",
                          return_value=fake_resp) as post, \
              patch.object(config, "BROWSER_AGENT_PROVIDER", "fireworks"), \
              patch.object(config, "BROWSER_AGENT_MODEL",
                            "accounts/fireworks/models/deepseek-v4-flash-0731"), \
              patch.object(config, "BROWSER_AGENT_REASONING_EFFORT", "medium"):
              browser_agent._call_openai_compatible(
                  "https://api.fireworks.ai/inference/v1/chat/completions",
                  "k-fw", messages, tools
              )
        self.assertEqual(
            post.call_args.kwargs["json"]["reasoning_effort"], "medium"
        )

    def test_reasoning_effort_never_sent_for_glm_models(self):
        fake_resp = FakeResponse({"choices": [{"message": {"content": "hi"}}]})
        messages = [{"role": "user", "content": "go"}]
        tools = [{"name": "navigate", "description": "Navigate",
                  "input_schema": {"type": "object"}}]
        with patch.object(browser_agent._MODEL_SESSION, "post",
                          return_value=fake_resp) as post, \
              patch.object(config, "BROWSER_AGENT_PROVIDER", "fireworks"), \
              patch.object(config, "BROWSER_AGENT_MODEL",
                            "z-ai/glm-5.3-flash"), \
              patch.object(config, "BROWSER_AGENT_REASONING_EFFORT", "medium"):
            browser_agent._call_openai_compatible(
                "https://api.fireworks.ai/inference/v1/chat/completions",
                "k-fw", messages, tools
            )
        self.assertNotIn("reasoning_effort", post.call_args.kwargs["json"])

    def test_cline_data_wrapper_response_is_parsed(self):
        fake_resp = FakeResponse({"data": {"choices": [{"message": {
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "navigate",
                "arguments": '{"url": "https://example.com"}',
            }}],
        }}]}})
        messages = [{"role": "user", "content": "go"}]
        tools = [{"name": "navigate", "description": "Navigate",
                  "input_schema": {"type": "object"}}]
        with patch.object(browser_agent._MODEL_SESSION, "post",
                          return_value=fake_resp):
            text, calls = browser_agent._call_openai_compatible(
                "https://api.cline.bot/api/v1/chat/completions", "k-cline",
                messages, tools
            )
        self.assertIsNone(text)
        self.assertEqual(calls[0]["name"], "navigate")
        self.assertEqual(calls[0]["arguments"], {"url": "https://example.com"})

    def test_model_turn_dispatches_cline_provider(self):
        with patch.object(browser_agent, "_resolve_browser_tool_model", return_value=("cline", "cline-model")), \
             patch.object(config, "CLINE_API_KEY", "k-cline"), \
             patch.object(browser_agent, "_call_openai_compatible",
                          return_value=(None, [])) as compat:
            browser_agent._model_turn([], [])
        # browser_tool model is resolved per-turn from registry; call now
        # includes model and provider kwargs — verify the primary 4 args.
        compat.assert_called_once()
        self.assertEqual(compat.call_args.args[0], config.CLINE_API_URL)
        self.assertEqual(compat.call_args.args[1], "k-cline")
        self.assertEqual(compat.call_args.args[2], [])
        self.assertEqual(compat.call_args.args[3], [])

    def test_gemini_adapter_posts_function_declarations(self):
        fake_resp = FakeResponse({"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "navigate",
                              "args": {"url": "https://example.com"}}},
        ]}}]})
        messages = [{"role": "user", "content": "go"}]
        tools = [{"name": "navigate", "description": "Navigate",
                  "input_schema": {"type": "object", "$schema": "http://x",
                                   "additionalProperties": False,
                                   "properties": {"url": {"type": "string"}}}}]
        with patch.object(browser_agent._MODEL_SESSION, "post",
                          return_value=fake_resp) as post, \
              patch.object(config, "GEMINI_API_KEY", "k-gem"), \
              patch.object(config, "BROWSER_AGENT_MODEL", "gemini-2.0-flash"):
            text, calls = browser_agent._call_gemini(messages, tools)
        self.assertIsNone(text)
        self.assertEqual(calls[0]["name"], "navigate")
        self.assertEqual(calls[0]["arguments"], {"url": "https://example.com"})
        kwargs = post.call_args.kwargs
        self.assertIn(":generateContent", post.call_args.args[0])
        self.assertEqual(kwargs["params"]["key"], "k-gem")
        decls = kwargs["json"]["tools"][0]["functionDeclarations"]
        self.assertEqual(decls[0]["name"], "navigate")
        self.assertNotIn("$schema", decls[0]["parameters"])
        self.assertNotIn("additionalProperties", decls[0]["parameters"])
        self.assertEqual(decls[0]["parameters"]["properties"]["url"]["type"],
                         "string")

    def test_gemini_adapter_parses_text_answer(self):
        fake_resp = FakeResponse({"candidates": [{"content": {"parts": [
            {"text": "Done."},
        ]}}]})
        with patch.object(browser_agent._MODEL_SESSION, "post",
                          return_value=fake_resp), \
              patch.object(config, "GEMINI_API_KEY", "k"), \
              patch.object(config, "BROWSER_AGENT_MODEL", "gemini-2.0-flash"):
            text, calls = browser_agent._call_gemini(
                [{"role": "user", "content": "go"}], []
            )
        self.assertEqual(text, "Done.")
        self.assertEqual(calls, [])


class BrainRoutingTests(unittest.TestCase):
    """TASK_ENGINE routing in _execute_deferred_opencode."""

    def tearDown(self):
        brain.set_opencode_task_running(False)
        brain._pending_opencode_task = None
        browser_agent._STOP_REQUESTED.clear()

    def _run_browser_engine(self, run_result):
        with patch.object(config, "TASK_ENGINE", "browser_agent"), \
             patch.object(brain, "run_browser_task", return_value=run_result), \
             patch.object(brain, "_notify_async_reply") as notify:
            reply = brain._execute_deferred_opencode("t", "t")
            self.assertTrue(_wait_until(lambda: notify.called))
        return reply, notify

    def test_browser_engine_runs_browser_task_and_reports_success(self):
        seen = {}

        def fake_run(_task):
            seen["flag"] = brain.opencode_task_in_progress()
            seen["narration"] = opencode_client._narration_enabled
            return "Done: found the price."

        with patch.object(config, "TASK_ENGINE", "browser_agent"), \
             patch.object(brain, "run_browser_task", side_effect=fake_run), \
             patch.object(brain, "_notify_async_reply") as notify:
            reply = brain._execute_deferred_opencode(
                "find the price", "find the price"
            )
            self.assertTrue(_wait_until(lambda: notify.called))
        self.assertEqual(reply, brain.BROWSER_AGENT_START_PHRASE)
        expected = brain._summarize_browser_output(
            "Done: found the price.", "find the price")
        self.assertEqual(notify.call_args[0][0], expected)
        self.assertEqual(notify.call_args[1]["spoken"], expected)
        # Flags set before the worker starts, reset in finally.
        self.assertTrue(seen["flag"])
        self.assertTrue(seen["narration"])
        self.assertFalse(brain.opencode_task_in_progress())
        self.assertFalse(opencode_client._narration_enabled)

    def test_browser_success_speaks_clipped_long_answer(self):
        long_output = ("The desktop contains 42 items including report.docx, budget.xlsx, and photos. " * 10)
        with patch.object(config, "TASK_ENGINE", "browser_agent"), \
             patch.object(brain, "run_browser_task", return_value=long_output), \
             patch.object(brain, "_notify_async_reply") as notify:
            brain._execute_deferred_opencode("list desktop", "list desktop")
            self.assertTrue(_wait_until(lambda: notify.called))
        spoken = notify.call_args[1]["spoken"]
        text = notify.call_args[0][0]
        self.assertEqual(spoken, brain._summarize_browser_output(long_output, "list desktop"))
        self.assertEqual(text, spoken)
        self.assertLessEqual(len(spoken), 300)
        self.assertLessEqual(len(spoken.splitlines()), 3)
        self.assertNotEqual(spoken, "Sir, the task has been completed.")

    def test_browser_engine_failure_speaks_could_not_complete(self):
        with patch.object(config, "TASK_ENGINE", "browser_agent"), \
             patch.object(brain, "run_browser_task",
                          return_value="TASK NOT COMPLETED. Error: nope (tool: navigate)"), \
             patch.object(brain, "_notify_async_reply") as notify:
            brain._execute_deferred_opencode("find the price", "find the price")
            self.assertTrue(_wait_until(lambda: notify.called))
        self.assertIn("could not be completed",
                        notify.call_args[1]["spoken"].lower())
        self.assertEqual(notify.call_args[0][0], notify.call_args[1]["spoken"])
        self.assertLessEqual(len(notify.call_args[0][0]), 300)
        # Full error stays reachable in the detail store.
        self.assertIn("TASK NOT COMPLETED", brain.get_last_browser_full_output())

    def test_browser_engine_returns_browser_start_phrase(self):
        with patch.object(config, "TASK_ENGINE", "browser_agent"), \
             patch.object(brain, "run_browser_task", return_value="ok"), \
             patch.object(brain, "_notify_async_reply") as notify:
            reply = brain._execute_deferred_opencode("t", "t")
            # Wait for THIS test's worker thread to finish. Without the wait
            # its late notification landed inside the next test's mock and
            # failed an unrelated assertion (test isolation, not behaviour).
            self.assertTrue(_wait_until(
                lambda: notify.called and not brain.opencode_task_in_progress()))
        self.assertEqual(reply, brain.BROWSER_AGENT_START_PHRASE)

    def test_browser_stop_message_resets_flag_and_is_spoken(self):
        reply, notify = self._run_browser_engine("Stopped per your request.")
        self.assertEqual(reply, brain.BROWSER_AGENT_START_PHRASE)
        # The task-running flag resets so the UI stop button hides, and the
        # stop line reaches the user verbatim — never with a success prefix.
        self.assertFalse(brain.opencode_task_in_progress())
        self.assertEqual(notify.call_args[0][0], "Stopped per your request.")
        self.assertEqual(notify.call_args[1]["spoken"],
                         "Stopped per your request.")
        self.assertNotIn("done", notify.call_args[0][0].lower())

    def test_browser_clarifying_question_has_no_done_prefix(self):
        question = "Which site should I open for the price check?"
        with patch.object(config, "TASK_ENGINE", "browser_agent"), \
             patch.object(brain, "run_browser_task", return_value=question), \
             patch.object(brain, "_notify_async_reply") as notify:
            try:
                brain._execute_deferred_opencode("find the price", "find the price")
                self.assertTrue(_wait_until(lambda: notify.called))
            finally:
                brain._pending_browser_clarification = None
        text = notify.call_args[0][0]
        spoken = notify.call_args[1]["spoken"]
        self.assertEqual(text, spoken)
        self.assertNotIn("done", text.lower())
        self.assertIn("quick question", text.lower())
        self.assertIn(question, text)
        self.assertLessEqual(len(text), 300)

    def test_opencode_engine_keeps_open_code_start_phrase(self):
        with patch.object(config, "TASK_ENGINE", "opencode"), \
             patch.object(brain, "is_opencode_available", return_value=True), \
             patch.object(brain, "run_opencode_task", return_value="ok"), \
             patch.object(brain, "_notify_async_reply") as notify:
            reply = brain._execute_deferred_opencode("t", "t")
            # Same isolation wait as the browser case above.
            self.assertTrue(_wait_until(
                lambda: notify.called and not brain.opencode_task_in_progress()))
        self.assertEqual(reply, brain.OPENCODE_START_PHRASE)


class G1BrainHonestyTests(unittest.TestCase):
    """G1 Round 1 (F03): partial never says 'Sir, done', failed keeps the
    honest prefix, engine reasons surface in the reply."""

    def tearDown(self):
        brain.set_opencode_task_running(False)
        brain._pending_browser_clarification = None
        browser_agent._STOP_REQUESTED.clear()

    def test_partial_browser_result_never_says_done(self):
        partial = TaskResult.partial(
            "Opened the page but the price was missing.",
            detail="Opened the page but the price was missing.",
            evidence=["navigate: timeout", "evaluate: boom"],
        )
        with patch.object(config, "TASK_ENGINE", "browser_agent"), \
             patch.object(brain, "run_browser_task", return_value=partial), \
             patch.object(brain, "_notify_async_reply") as notify:
            brain._execute_deferred_opencode("find the price", "find the price")
            self.assertTrue(_wait_until(lambda: notify.called))
        text = notify.call_args[0][0]
        self.assertIn("only partly done", text.lower())
        self.assertNotIn("Sir, done", text)
        self.assertIn("2 step(s) failed", text)
        self.assertLessEqual(len(text), 300)
        self.assertIn("Opened the page", brain.get_last_browser_full_output())

    def test_failed_browser_result_keeps_could_not_complete(self):
        failed = TaskResult.failed("nope")
        with patch.object(config, "TASK_ENGINE", "browser_agent"), \
             patch.object(brain, "run_browser_task", return_value=failed), \
             patch.object(brain, "_notify_async_reply") as notify:
            brain._execute_deferred_opencode("find the price", "find the price")
            self.assertTrue(_wait_until(lambda: notify.called))
        self.assertIn("could not be completed",
                      notify.call_args[1]["spoken"].lower())
        self.assertNotIn("Sir, done", notify.call_args[0][0])

    def test_opencode_failed_includes_engine_reason(self):
        failed = TaskResult.failed("opencode timed out after 5s")
        with patch.object(config, "TASK_ENGINE", "opencode"), \
             patch.object(brain, "is_opencode_available", return_value=True), \
             patch.object(brain, "run_opencode_task", return_value=failed), \
             patch.object(brain, "_notify_async_reply") as notify:
            brain._execute_deferred_opencode("build it", "build it")
            self.assertTrue(_wait_until(lambda: notify.called))
        text = notify.call_args[0][0]
        spoken = notify.call_args[1]["spoken"]
        self.assertIn("there was a problem", text.lower())
        self.assertIn("timed out after 5s", text)
        self.assertEqual(spoken, brain._voice_clip(text))

    def test_opencode_failed_never_infers_success_from_created(self):
        summary = brain._summarize_opencode_output(
            "Created folder 'badmoss'.", status="failed",
            error="opencode exited with code 2")
        self.assertIn("there was a problem", summary.lower())
        self.assertIn("exited with code 2", summary)
        self.assertNotIn("has been created", summary.lower())

    def test_opencode_completed_empty_says_something_sensible(self):
        completed = TaskResult.completed("", detail="")
        with patch.object(config, "TASK_ENGINE", "opencode"), \
             patch.object(brain, "is_opencode_available", return_value=True), \
             patch.object(brain, "run_opencode_task", return_value=completed), \
             patch.object(brain, "_notify_async_reply") as notify:
            brain._execute_deferred_opencode("build it", "build it")
            self.assertTrue(_wait_until(lambda: notify.called))
        text = notify.call_args[0][0]
        self.assertIn("done", text.lower())
        self.assertNotIn("couldn't", text.lower())

    def test_opencode_partial_does_not_claim_success(self):
        partial = TaskResult.partial(
            "Created half the files.", detail="Created half the files.",
            evidence=["write failed"],
        )
        with patch.object(config, "TASK_ENGINE", "opencode"), \
             patch.object(brain, "is_opencode_available", return_value=True), \
             patch.object(brain, "run_opencode_task", return_value=partial), \
             patch.object(brain, "_notify_async_reply") as notify:
            brain._execute_deferred_opencode("build it", "build it")
            self.assertTrue(_wait_until(lambda: notify.called))
        text = notify.call_args[0][0]
        spoken = notify.call_args[1]["spoken"]
        self.assertIn("partly done", text.lower())
        self.assertNotIn("has been created", text.lower())
        self.assertNotIn("that has been done", text.lower())
        self.assertEqual(spoken, brain._voice_clip(text))

    def test_opencode_stopped_says_stopped(self):
        with patch.object(config, "TASK_ENGINE", "opencode"), \
             patch.object(brain, "is_opencode_available", return_value=True), \
             patch.object(brain, "run_opencode_task",
                          return_value=TaskResult.stopped()), \
             patch.object(brain, "_notify_async_reply") as notify:
            brain._execute_deferred_opencode("build it", "build it")
            self.assertTrue(_wait_until(lambda: notify.called))
        self.assertEqual(notify.call_args[0][0], "Sir, the task was stopped.")

    def test_opencode_needs_input_surfaces_question(self):
        question = TaskResult.needs_input("Which folder should I use?")
        with patch.object(config, "TASK_ENGINE", "opencode"), \
             patch.object(brain, "is_opencode_available", return_value=True), \
             patch.object(brain, "run_opencode_task", return_value=question), \
             patch.object(brain, "_notify_async_reply") as notify:
            brain._execute_deferred_opencode("build it", "build it")
            self.assertTrue(_wait_until(lambda: notify.called))
        text = notify.call_args[0][0]
        self.assertIn("quick question", text.lower())
        self.assertIn("Which folder should I use?", text)

    def test_summarize_opencode_partial_never_infers_success(self):
        summary = brain._summarize_opencode_output(
            "Created folder 'badmoss'.", status="partial")
        self.assertIn("partly done", summary.lower())
        self.assertNotIn("has been created", summary.lower())

    def test_summarize_partial_never_sir_done(self):
        summary = brain._summarize_browser_output(
            "Did most things.", "tidy the tabs",
            status="partial", evidence=["navigate: timeout"])
        self.assertIn("only partly done", summary.lower())
        self.assertNotIn("Sir, done", summary)
        self.assertIn("1 step(s) failed", summary)
        self.assertLessEqual(len(summary), 300)

    def test_summarize_failed_without_prefix_stays_honest(self):
        summary = brain._summarize_browser_output(
            "Created the file.", "make a file", status="failed")
        self.assertIn("could not be completed", summary.lower())
        self.assertNotIn("Sir, done", summary)


class ConfigDefaultsTests(unittest.TestCase):
    """Cline GLM 5.3 Flash defaults + dotenv-override regression."""

    _RELOAD_ENV_VARS = (
        "FIREWORKS_API_KEY",
        "JARVIS_BROWSER_AGENT_PROVIDER",
        "JARVIS_BROWSER_AGENT_MODEL",
        "JARVIS_BROWSER_AGENT_MAX_STEPS",
        "JARVIS_BROWSER_AGENT_TIMEOUT",
    )

    def _snapshot_env(self):
        saved = {}
        for name in self._RELOAD_ENV_VARS:
            saved[name] = os.environ.get(name)
        return saved

    def _restore_env(self, saved):
        for name in self._RELOAD_ENV_VARS:
            value = saved[name]
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        importlib.reload(config)

    def test_dotenv_override_beats_stale_process_env(self):
        # The contract is ".env wins over a stale process value". That needs a
        # .env which actually defines the key — the repo's .env is a
        # developer-specific, untracked file and may legitimately omit it, in
        # which case there is nothing to override with and the test is not
        # applicable (it used to fail on such a machine rather than skip).
        env_text = ""
        try:
            if config.ENV_PATH.exists():
                env_text = config.ENV_PATH.read_text(encoding="utf-8",
                                                     errors="replace")
        except Exception:
            env_text = ""
        # A present-but-EMPTY assignment is not applicable either: load_dotenv
        # (override=True) would win with "" quite correctly, so the assertion
        # below could never hold and the bare name alone proves nothing.
        # [ \t] (not \s) around the "=" so the value group cannot reach the next
        # line, and no trailing $ so a CRLF .env (Notepad/VS Code on Windows)
        # still counts as configured.
        assigned = re.search(r"(?m)^[ \t]*FIREWORKS_API_KEY[ \t]*=[ \t]*(\S[^\r\n]*)",
                             env_text)
        if not assigned or not assigned.group(1).strip().strip("\"'"):
            self.skipTest("repo .env defines no non-empty FIREWORKS_API_KEY — "
                          "tested only where the key is configured")
        saved = self._snapshot_env()
        self.addCleanup(self._restore_env, saved)
        os.environ["FIREWORKS_API_KEY"] = "stale-test-key"
        importlib.reload(config)
        self.assertTrue(config.FIREWORKS_API_KEY)
        self.assertNotEqual(config.FIREWORKS_API_KEY, "stale-test-key")

    def test_browser_agent_defaults_point_to_fireworks_qwen(self):
        saved = self._snapshot_env()
        self.addCleanup(self._restore_env, saved)
        for name in ("JARVIS_BROWSER_AGENT_PROVIDER", "JARVIS_BROWSER_AGENT_MODEL",
                     "JARVIS_BROWSER_AGENT_MAX_STEPS", "JARVIS_BROWSER_AGENT_TIMEOUT"):
            os.environ.pop(name, None)
        importlib.reload(config)
        self.assertEqual(config.BROWSER_AGENT_PROVIDER, "fireworks")
        self.assertEqual(config.BROWSER_AGENT_MODEL,
                         "accounts/fireworks/models/qwen3p7-plus")
        self.assertEqual(config.BROWSER_AGENT_MAX_STEPS, 50)
        self.assertEqual(config.BROWSER_AGENT_TIMEOUT, 480)

    # Compatibility alias: old name now asserts fireworks/qwen defaults
    def test_browser_agent_defaults_point_to_fireworks_minimax(self):
        saved = self._snapshot_env()
        self.addCleanup(self._restore_env, saved)
        for name in ("JARVIS_BROWSER_AGENT_PROVIDER", "JARVIS_BROWSER_AGENT_MODEL",
                     "JARVIS_BROWSER_AGENT_MAX_STEPS", "JARVIS_BROWSER_AGENT_TIMEOUT"):
            os.environ.pop(name, None)
        importlib.reload(config)
        self.assertEqual(config.BROWSER_AGENT_PROVIDER, "fireworks")
        self.assertEqual(config.BROWSER_AGENT_MODEL,
                         "accounts/fireworks/models/qwen3p7-plus")
        self.assertEqual(config.BROWSER_AGENT_MAX_STEPS, 50)
        self.assertEqual(config.BROWSER_AGENT_TIMEOUT, 480)


class _CleanStopStateTestCase(unittest.TestCase):
    """[P1-11] Stop assertions are about GLOBAL state, so establish it.

    ``browser_agent.stop_requested()`` is the shared flag OR any live browser
    job wanting to stop, and the job registry is process-wide. Another module's
    background handoff (e.g. test_brain_gate's failed-local-execution path) can
    still own a live browser job when this file runs, which silently breaks
    every "nothing is stopping" assertion here — and, worse, makes the first
    /task/stop CANCEL that unrelated job. Retire those jobs first.
    """

    def setUp(self):
        from backend.services import jobs as job_registry

        for job in job_registry.live_jobs(kind="browser"):
            try:
                job.finish()
            except Exception:
                pass
        browser_agent._STOP_REQUESTED.clear()


class StopControlTests(_CleanStopStateTestCase):
    """User stop: request_stop flips the cancel event; the loop bails out
    gracefully at the next step boundary; runs clear a stale stop."""

    def tearDown(self):
        browser_agent._STOP_REQUESTED.clear()

    def _run(self, task, model_turns=None, model=None):
        client = FakeClient()
        model_patch = (patch.object(browser_agent, "_model_turn",
                                    side_effect=model_turns)
                       if model is None
                       else patch.object(browser_agent, "_model_turn",
                                         side_effect=model))
        with patch.object(browser_agent, "ensure_brave_mcp_daemon",
                          return_value=True), \
             patch.object(browser_agent, "BraveMcpClient", return_value=client), \
             model_patch, \
             patch.object(browser_agent, "append_activity_line") as log_line, \
             patch.object(browser_agent, "narrate_activity"), \
             patch.object(browser_agent, "truncate_activity_log"):
            result = browser_agent.run_browser_task(task)
        return result, client, log_line

    def test_request_stop_exposes_event(self):
        self.assertFalse(browser_agent.stop_requested())
        browser_agent.request_stop()
        self.assertTrue(browser_agent.stop_requested())
        browser_agent._STOP_REQUESTED.clear()
        self.assertFalse(browser_agent.stop_requested())

    def test_agent_loop_checks_stop_at_step_top(self):
        # The loop bails at the top of the very first step without calling
        # the model. (run_browser_task clears the event at start - a stop
        # can only take effect once a run is in flight.)
        browser_agent._STOP_REQUESTED.set()
        client = FakeClient()
        model = Mock()
        with patch.object(browser_agent, "_model_turn", side_effect=model), \
             patch.object(browser_agent, "append_activity_line") as log_line, \
             patch.object(browser_agent, "narrate_activity") as narrate:
            result = browser_agent._agent_loop(client, "task", time.monotonic())
        self.assertEqual(result, "Stopped per your request.")
        model.assert_not_called()
        self.assertEqual(client.tool_calls, [])
        lines = [call.args[0] for call in log_line.call_args_list]
        self.assertTrue(any(line.startswith("stopped by user")
                            for line in lines))
        narrate.assert_called_once_with("Stopping")

    def test_stop_mid_loop_stops_at_next_step_boundary(self):
        class StoppingClient(FakeClient):
            def call_tool(self, name, arguments):
                result = super().call_tool(name, arguments)
                browser_agent.request_stop()
                return result

        client = StoppingClient()
        tool_call = {"id": "c1", "name": "navigate",
                     "arguments": {"url": "https://example.com"}}
        with patch.object(browser_agent, "ensure_brave_mcp_daemon",
                          return_value=True), \
             patch.object(browser_agent, "BraveMcpClient", return_value=client), \
             patch.object(browser_agent, "_model_turn",
                          return_value=(None, [tool_call])), \
             patch.object(browser_agent, "append_activity_line") as log_line, \
             patch.object(browser_agent, "narrate_activity"), \
             patch.object(browser_agent, "truncate_activity_log"):
            result = browser_agent.run_browser_task("task")
        self.assertEqual(result, "Stopped per your request.")
        # Only the in-flight tool ran; the next step never called the model
        # or another tool.
        self.assertEqual(client.tool_calls,
                         [("navigate", {"url": "https://example.com"})])
        lines = [call.args[0] for call in log_line.call_args_list]
        self.assertTrue(any(line.startswith("stopped by user")
                            for line in lines))

    def test_stale_stop_is_cleared_by_the_next_run(self):
        browser_agent.request_stop()
        self.assertTrue(browser_agent.stop_requested())
        result, _client, _log = self._run(
            "task", model_turns=[("Done.", [])])
        self.assertEqual(result, "Done.")
        self.assertFalse(browser_agent.stop_requested())


class TaskStopRouteTests(_CleanStopStateTestCase):
    """/task/stop endpoint + task_running exposure in /ui-state."""

    def tearDown(self):
        browser_agent._STOP_REQUESTED.clear()
        brain.set_opencode_task_running(False)
        from backend.services.research_service import _STOP_REQUESTED as rs_stop
        rs_stop.clear()

    def test_ui_state_exposes_task_running(self):
        brain.set_opencode_task_running(True)
        self.assertTrue(routes.get_ui_state()["task_running"])
        brain.set_opencode_task_running(False)
        self.assertFalse(routes.get_ui_state()["task_running"])

    def test_stop_task_endpoint_sets_cancel_event_and_returns_ok(self):
        # F20: with no live job the endpoint falls back to the engines' legacy
        # flags, and it reports which jobs (if any) it cancelled.
        response = routes.stop_task()
        self.assertTrue(response["ok"])
        self.assertEqual(response["cancelled"], [])
        self.assertTrue(browser_agent.stop_requested())
        browser_agent._STOP_REQUESTED.clear()
        self.assertFalse(browser_agent.stop_requested())

    def test_stop_task_endpoint_cancels_one_identified_job(self):
        from backend.services import jobs as job_registry

        job_a = job_registry.new_job(kind="browser", label="A")
        job_b = job_registry.new_job(kind="browser", label="B")
        try:
            response = routes.stop_task(job_id=job_a.job_id)
            self.assertEqual(response["cancelled"], [job_a.job_id])
            self.assertTrue(job_a.cancelled)
            self.assertFalse(job_b.cancelled)
        finally:
            job_a.finish()
            job_b.finish()


class VirtualToolTests(unittest.TestCase):
    """Vision grounding and batch/wait/direct-action composite tools."""

    def test_virtual_tool_defs_include_all_composite_tools(self):
        names = {t["name"] for t in browser_agent._VIRTUAL_TOOL_DEFS}
        for expected in ("look", "click_mark", "fill_mark", "verify_playing",
                         "batch_probe", "wait_for", "fill", "click_text", "click_point"):
            self.assertIn(expected, names)
        # MCP tools still appear untouched via list_tools merging
        self.assertIn("look", browser_agent._VIRTUAL_TOOL_NAMES)
        # truth-in-advertising: DOM tools declare the top-document limit
        defs = {t["name"]: t["description"] for t in browser_agent._VIRTUAL_TOOL_DEFS}
        self.assertIn("top document", defs["wait_for"].lower())
        self.assertIn("top document", defs["batch_probe"].lower())
        self.assertIn("cross-origin", defs["click_point"].lower())
        self.assertIn("look-image", defs["click_point"].lower())

    def test_tool_defs_merge_passes_virtual_tools_to_model(self):
        # Fake client with one MCP tool
        class MergeClient:
            def list_tools(self):
                return [{"name": "navigate", "description": "d", "input_schema": {"type": "object"}}]
            def call_tool(self, n, a):
                return "ok"
            def connect(self): pass
            def close(self): pass
        seen = {}
        def fake_model_turn(history, tools):
            seen["tools"] = tools
            return ("done", [])
        with patch.object(browser_agent, "ensure_brave_mcp_daemon", return_value=True), \
             patch.object(browser_agent, "BraveMcpClient", return_value=MergeClient()), \
             patch.object(browser_agent, "_model_turn", side_effect=fake_model_turn), \
             patch.object(browser_agent, "append_activity_line"), \
             patch.object(browser_agent, "narrate_activity"), \
             patch.object(browser_agent, "truncate_activity_log"):
            result = browser_agent.run_browser_task("task")
        self.assertEqual(result, "done")
        tool_names = {t["name"] for t in seen["tools"]}
        self.assertIn("navigate", tool_names)
        self.assertIn("look", tool_names)
        self.assertIn("click_mark", tool_names)
        self.assertIn("batch_probe", tool_names)

    def test_daemon_tools_filtered_to_allowlist(self):
        # The daemon exposes ~16 tools; the model must see ONLY the
        # curated allowlist (+ virtual tools) - file tools and internal
        # tools are hidden so the model cannot improvise
        # screenshot+read_file chases that can never deliver an image.
        daemon_tools = [
            "open_brave", "navigate", "list_tabs", "switch_tab", "new_tab",
            "understand_page", "click_element", "fill_element", "evaluate",
            "read_file", "list_dir", "screenshot", "ask_chat",
            "copy_code_block", "extract_code_blocks", "close_brave",
        ]

        class NoisyClient:
            def list_tools(self):
                return [{"name": n, "description": "d",
                         "input_schema": {"type": "object"}}
                        for n in daemon_tools]
            def call_tool(self, n, a):
                return "ok"
            def connect(self): pass
            def close(self): pass

        seen = {}

        def fake_model_turn(history, tools):
            seen["tools"] = tools
            return ("done", [])

        with patch.object(browser_agent, "ensure_brave_mcp_daemon",
                          return_value=True), \
             patch.object(browser_agent, "BraveMcpClient",
                          return_value=NoisyClient()), \
             patch.object(browser_agent, "_model_turn",
                          side_effect=fake_model_turn), \
             patch.object(browser_agent, "append_activity_line"), \
             patch.object(browser_agent, "narrate_activity"), \
             patch.object(browser_agent, "truncate_activity_log"):
            result = browser_agent.run_browser_task("task")
        self.assertEqual(result, "done")
        tool_names = {t["name"] for t in seen["tools"]}
        for allowed in ("open_brave", "navigate", "list_tabs", "switch_tab",
                        "new_tab", "evaluate"):
            self.assertIn(allowed, tool_names)
        for hidden in ("read_file", "list_dir", "screenshot", "ask_chat",
                       "copy_code_block", "extract_code_blocks",
                       "close_brave", "understand_page", "click_element",
                       "fill_element"):
            self.assertNotIn(hidden, tool_names)
        # virtual tools are still merged in
        for virtual in ("look", "click_mark", "fill_mark", "verify_playing",
                        "wait_for", "batch_probe", "fill", "click_text",
                        "click_point"):
            self.assertIn(virtual, tool_names)

    def test_routing_virtual_intercepted_mcp_not_called(self):
        # look is virtual: handler should be called, MCP call_tool for look should NOT be called directly
        history = []
        session = {"marks": {}}
        # client that records calls
        client = Mock()
        # For look we need special handling: our _handle_look will call screenshot/evaluate internally
        # Use a client that handles screenshot/evaluate
        def side_effect(name, args):
            if name == "screenshot":
                # create tiny PNG at path
                from PIL import Image
                import os
                path = args["path"]
                img = Image.new("RGB", (50, 50), color=(10, 10, 10))
                img.save(path, format="PNG")
                return "ok"
            if name == "evaluate":
                return json.dumps({"elements": [], "url": "https://x", "title": "t"})
            return "ok"
        client.call_tool.side_effect = side_effect
        call = {"id": "c1", "name": "look", "arguments": {}}
        browser_agent._run_one_tool(client, history, call, session)
        self.assertEqual(history[0]["name"], "look")
        self.assertIn("image_b64", history[0])
        # Ensure the original virtual name was not forwarded as MCP tool named "look" beyond the composite handler's internal calls
        # The handler internally called screenshot/evaluate, but never call_tool with name "look" itself
        for n, _ in client.call_tool.call_args_list:
            self.assertNotEqual(n[0] if isinstance(n, tuple) else n, "look")  # internal calls are screenshot/evaluate

    def test_routing_unknown_goes_to_mcp(self):
        history = []
        client = Mock()
        client.call_tool.return_value = "mcp result"
        call = {"id": "c2", "name": "navigate", "arguments": {"url": "https://example.com"}}
        browser_agent._run_one_tool(client, history, call, {})
        client.call_tool.assert_called_once_with("navigate", {"url": "https://example.com"})
        self.assertEqual(history[0]["content"], "mcp result")

    def test_look_handler_produces_image_and_marks_and_deletes_temp(self):
        from PIL import Image
        import tempfile, os, json
        class LookClient:
            def __init__(self):
                self.seen_path = None
            def call_tool(self, name, args):
                if name == "screenshot":
                    path = args["path"]
                    self.seen_path = path
                    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
                    img.save(path, format="PNG")
                    return "saved"
                if name == "evaluate":
                    data = {"elements": [{"x": 10, "y": 20, "w": 30, "h": 20, "tag": "a", "label": "Home"}, {"x": 50, "y": 60, "w": 100, "h": 30, "tag": "button", "label": "Submit"}], "url": "https://example.com", "title": "Test"}
                    return json.dumps(data)
                return "ok"
        client = LookClient()
        session = {}
        text, b64 = browser_agent._handle_look(client, session)
        self.assertIsNotNone(b64)
        self.assertIn("mark | tag", text.lower())
        self.assertIn("Home", text)
        self.assertEqual(session["marks"][1]["cx"], 25)
        self.assertEqual(session["marks"][1]["cy"], 30)
        # Coordinate frame statement + 1:1 scale stored for click_point
        self.assertIn("coordinate frame", text.lower())
        self.assertIn("look-image", text.lower())
        self.assertEqual(session["look_scale"], 1.0)
        self.assertEqual(session["viewport_size"], (100, 100))
        # temp file deleted
        self.assertFalse(os.path.exists(client.seen_path))
        # image is valid base64 JPEG (starts with JPEG header)
        import base64
        raw = base64.b64decode(b64)
        self.assertTrue(raw.startswith(b"\xff\xd8"))

    def test_look_stores_scale_and_reports_image_frame_when_downscaled(self):
        from PIL import Image
        class LookClient:
            def call_tool(self, name, args):
                if name == "screenshot":
                    Image.new("RGB", (1600, 900), (0, 128, 0)).save(
                        args["path"], format="PNG")
                    return "saved"
                if name == "evaluate":
                    return json.dumps({"elements": [
                        {"x": 400, "y": 100, "w": 200, "h": 50, "tag": "input",
                         "label": "Search Here..."},
                    ], "url": "https://example.com", "title": "Test"})
                return "ok"
        session = {}
        with patch.object(config, "BROWSER_AGENT_LOOK_WIDTH", 1280):
            text, b64 = browser_agent._handle_look(LookClient(), session)
        # scale = 1280/1600 = 0.8; session keeps original viewport coords
        self.assertAlmostEqual(session["look_scale"], 0.8)
        self.assertEqual(session["viewport_size"], (1600, 900))
        self.assertEqual(session["marks"][1]["cx"], 500)
        # the model-facing table is in look-image pixels (500 * 0.8 = 400)
        self.assertIn("400", text)
        # and the frame statement explains the relationship
        self.assertIn("look image is 1280x", text.lower())
        self.assertIn("the real viewport is 1600x900", text.lower())
        self.assertIn("never scale coordinates yourself", text.lower())

    def test_look_inventory_prioritizes_form_fields(self):
        from PIL import Image
        class LookClient:
            def __init__(self):
                self.expressions = []
            def call_tool(self, name, args):
                if name == "screenshot":
                    Image.new("RGB", (100, 100), (0, 0, 0)).save(
                        args["path"], format="PNG")
                    return "saved"
                if name == "evaluate":
                    self.expressions.append(args["expression"])
                    return json.dumps({"elements": [], "url": "u", "title": "t"})
                return "ok"
        client = LookClient()
        browser_agent._handle_look(client, {})
        # F39: the continuity probe runs after the screenshot, so the
        # inventory is not necessarily the LAST expression seen.
        seen_expr = "\n".join(client.expressions)
        self.assertIn("formEls", seen_expr)
        self.assertIn("mediaEls", seen_expr)
        self.assertIn("formEls.concat(mediaEls, inVp, offVp)", seen_expr)
        self.assertIn("iframe", seen_expr)
        self.assertIn("out.length >= 40", seen_expr)
        self.assertIn("slice(0,25)", seen_expr)

    def test_look_media_elements_appear_right_after_form_fields(self):
        from PIL import Image
        elements = [
            {"x": 50, "y": 20, "w": 200, "h": 30, "tag": "input", "label": "Search"},
            {"x": 100, "y": 100, "w": 640, "h": 360, "tag": "video", "label": ""},
            {"x": 100, "y": 100, "w": 640, "h": 360, "tag": "iframe", "label": ""},
            {"x": 10, "y": 500, "w": 80, "h": 20, "tag": "a", "label": "Home"},
            {"x": 10, "y": 530, "w": 80, "h": 20, "tag": "button", "label": "Next"},
        ]
        class LookClient:
            def call_tool(self, name, args):
                if name == "screenshot":
                    Image.new("RGB", (100, 100), (0, 0, 0)).save(args["path"], format="PNG")
                    return "saved"
                if name == "evaluate":
                    return json.dumps({"elements": elements, "url": "https://watch.example/embed", "title": "Watch"})
                return "ok"
        session = {}
        with patch.object(config, "BROWSER_AGENT_LOOK_WIDTH", 1280):
            text, b64 = browser_agent._handle_look(LookClient(), session)
        self.assertEqual(session["marks"][1]["tag"], "input")
        self.assertEqual(session["marks"][2]["tag"], "video")
        self.assertEqual(session["marks"][3]["tag"], "iframe")
        self.assertEqual(session["marks"][4]["tag"], "a")
        lines = [l for l in text.splitlines() if l.startswith("2 |") or l.startswith("3 |")]
        self.assertTrue(any("video" in l for l in lines))
        self.assertTrue(any("iframe" in l for l in lines))

    def test_evaluate_syntax_error_guidance_and_click_element_stale(self):
        # F17: raw page JavaScript is privileged, so it only runs for a job
        # that holds the privileged_js grant (JARVIS_BROWSER_PRIVILEGED_JS).
        privileged = {"grants": {"privileged_js"}}
        history = []
        client = Mock()
        client.call_tool.return_value = "SyntaxError: Unexpected token ';'"
        call = {"id": "c1", "name": "evaluate", "arguments": {"expression": "a; b;"}}
        with patch.object(browser_agent, "append_activity_line"), patch.object(browser_agent, "narrate_activity"):
            browser_agent._run_one_tool(client, history, call, privileged)
        content = history[0]["content"]
        self.assertIn("SyntaxError", content)
        self.assertIn("IIFE", content)
        self.assertIn("trailing semicolons", content.lower())
        history2 = []
        client2 = Mock()
        client2.call_tool.return_value = "ok result"
        call2 = {"id": "c2", "name": "evaluate", "arguments": {"expression": "document.title"}}
        with patch.object(browser_agent, "append_activity_line"), patch.object(browser_agent, "narrate_activity"):
            browser_agent._run_one_tool(client2, history2, call2, privileged)
        self.assertNotIn("IIFE", history2[0]["content"])
        # click_element is not exposed to the model at all, so a direct call
        # is refused at the dispatch boundary instead of reaching the daemon.
        history3 = []
        client3 = Mock()
        client3.call_tool.return_value = "click_element error: index 5 is stale - please re-look"
        call3 = {"id": "c3", "name": "click_element", "arguments": {"index": 5}}
        with patch.object(browser_agent, "append_activity_line"), patch.object(browser_agent, "narrate_activity"):
            browser_agent._run_one_tool(client3, history3, call3, {})
        client3.call_tool.assert_not_called()
        self.assertIn("blocked by policy", history3[0]["content"])

    def test_evaluate_without_grant_is_refused_at_dispatch(self):
        """F17: privileged page execution needs an explicit grant."""
        history = []
        client = Mock()
        call = {"id": "c1", "name": "evaluate",
                "arguments": {"expression": "document.cookie"}}
        with patch.object(browser_agent, "append_activity_line"), patch.object(browser_agent, "narrate_activity"):
            browser_agent._run_one_tool(client, history, call, {})
        client.call_tool.assert_not_called()
        content = history[0]["content"]
        self.assertIn("blocked by policy", content)
        self.assertIn("privileged", content)

    def test_batch_probe_rejects_mutating_expressions(self):
        """F17: a probe must be a single typed read-only expression."""
        history = []
        client = Mock()
        call = {"id": "c1", "name": "batch_probe",
                "arguments": {"expressions": ["document.querySelector('#a').click()"]}}
        with patch.object(browser_agent, "append_activity_line"), patch.object(browser_agent, "narrate_activity"):
            browser_agent._run_one_tool(client, history, call, {})
        client.call_tool.assert_not_called()
        self.assertIn("blocked by policy", history[0]["content"])

    def test_tool_result_and_arguments_are_scrubbed(self):
        """F21: secrets never reach the log or the model."""
        history = []
        client = Mock()
        client.call_tool.return_value = "signed in with token sk-abcdefghijklmnopqrstuv"
        call = {"id": "c1", "name": "navigate",
                "arguments": {"url": "https://example.com"}}
        with patch.object(browser_agent, "append_activity_line") as log_line, \
             patch.object(browser_agent, "narrate_activity"):
            browser_agent._run_one_tool(client, history, call, {})
        self.assertNotIn("sk-abcdefghijklmnopqrstuv", history[0]["content"])
        self.assertIn("masked", history[0]["content"].lower())
        logged = " ".join(c.args[0] for c in log_line.call_args_list)
        self.assertNotIn("sk-abcdefghijklmnopqrstuv", logged)

    def test_system_prompt_contains_watch_page_doctrine(self):
        prompt = browser_agent._SYSTEM_PROMPT
        lower = prompt.lower()
        self.assertIn("for click_mark and fill_mark only", lower)
        self.assertIn("different inventory", lower)
        self.assertIn("understand_page indices", lower)
        self.assertIn("re-look and use click_mark", lower)
        self.assertIn("play affordance", lower)
        self.assertIn("verify_playing", lower)
        self.assertIn("do not re-click the same coordinates", lower)
        self.assertIn("look again and pick a different target", lower)
        self.assertIn("popup", lower)
        self.assertIn("switch server", lower)
        self.assertIn("dismiss it with one targeted click", lower)
        self.assertIn("do not spend multiple probes", lower)
        self.assertIn("iframe only after", lower)
        self.assertIn("click play first", lower)

    def test_click_mark_with_fresh_marks_calls_evaluate_with_coords(self):
        session = {"marks": {1: {"cx": 25, "cy": 30, "tag": "a", "label": "Home"}}}
        client = Mock()
        client.call_tool.return_value = json.dumps({"clicked": True, "tag": "a", "label": "Home", "url": "https://example.com"})
        with patch.object(browser_agent.time, "sleep"):
            result = browser_agent._handle_click_mark(client, session, {"index": 1})
        self.assertIn("clicked", result.lower())
        expr = client.call_tool.call_args_list[0][0][1]["expression"]
        self.assertIn("elementFromPoint", expr)
        self.assertIn("25", expr)
        self.assertIn("30", expr)
        # sync click evaluate, then the post-settle location.href read
        self.assertEqual(
            client.call_tool.call_args_list[1][0][1]["expression"], "location.href")

    def test_click_mark_missing_returns_error(self):
        client = Mock()
        result = browser_agent._handle_click_mark(client, {}, {"index": 5})
        self.assertIn("look", result.lower())
        client.call_tool.assert_not_called()

    def test_batch_probe_js_contains_all_expressions(self):
        client = Mock()
        client.call_tool.return_value = json.dumps([{"expr": "a", "ok": 1}])
        exprs = ["document.title", "location.href", "1+1"]
        result = browser_agent._handle_batch_probe(client, {"expressions": exprs})
        expr = client.call_tool.call_args[0][1]["expression"]
        for e in exprs:
            self.assertIn(e, expr)
        self.assertIsNotNone(result)

    def test_batch_probe_rejects_too_many(self):
        client = Mock()
        exprs = ["a"] * 11
        result = browser_agent._handle_batch_probe(client, {"expressions": exprs})
        self.assertIn("too many", result.lower())
        client.call_tool.assert_not_called()

    def test_batch_probe_rejects_long_expression(self):
        client = Mock()
        long_expr = "a" * 501
        result = browser_agent._handle_batch_probe(client, {"expressions": [long_expr]})
        self.assertIn("too long", result.lower())
        client.call_tool.assert_not_called()

    def test_wait_for_polls_synchronously_until_found(self):
        client = Mock()
        client.call_tool.side_effect = [
            json.dumps({"found": False, "url": "https://x/before", "title": "t"}),
            json.dumps({"found": True, "url": "https://x/after", "title": "t2"}),
        ]
        with patch.object(browser_agent.time, "sleep") as sleep_mock, \
             patch.object(browser_agent.time, "monotonic",
                          side_effect=[0.0, 0.0, 0.1, 0.1, 0.4, 0.4]):
            # BA-00: the wait.polls span reads the clock twice more (start +
            # done); the started/elapsed/check values are unchanged.
            result = browser_agent._handle_wait_for(
                client, {"selector": "#foo", "text": "hello", "timeout_ms": 5000})
        payload = json.loads(result)
        self.assertTrue(payload["found"])
        self.assertEqual(payload["url"], "https://x/after")
        self.assertIn("elapsed_ms", payload)
        self.assertEqual(payload["elapsed_ms"], 400)
        # one synchronous IIFE per poll, 200ms Python-side gap, no Promise
        expr = client.call_tool.call_args_list[0][0][1]["expression"]
        self.assertIn("#foo", expr)
        self.assertIn("hello", expr)
        self.assertIn("(() => {", expr)
        self.assertNotIn("Promise", expr)
        sleep_mock.assert_called_once_with(0.2)

    def test_wait_for_timeout_capped_at_10s_and_reports_not_found(self):
        client = Mock()
        client.call_tool.return_value = json.dumps(
            {"found": False, "url": "https://x", "title": "t"})
        with patch.object(browser_agent.time, "sleep"), \
             patch.object(browser_agent.time, "monotonic",
                          side_effect=[0.0, 0.0, 10.5, 10.5, 10.5]):
            # BA-00: +2 clock reads for the wait.polls span (see above).
            result = browser_agent._handle_wait_for(
                client, {"selector": "#foo", "timeout_ms": 15000})
        payload = json.loads(result)
        self.assertFalse(payload["found"])
        # capped to 10000ms: 10.5s is already past the deadline - exactly
        # ONE poll ran (with the requested 15000ms it would have polled
        # again), and found:false + elapsed_ms come back.
        self.assertEqual(client.call_tool.call_count, 1)
        self.assertEqual(payload["elapsed_ms"], 10500)

    def test_wait_for_default_timeout_is_5s(self):
        client = Mock()
        client.call_tool.return_value = json.dumps(
            {"found": False, "url": "https://x", "title": "t"})
        with patch.object(browser_agent.time, "sleep"), \
             patch.object(browser_agent.time, "monotonic",
                          side_effect=[0.0, 0.0, 6.0, 6.0, 6.0]):
            # BA-00: +2 clock reads for the wait.polls span (see above).
            result = browser_agent._handle_wait_for(client, {"selector": "#foo"})
        payload = json.loads(result)
        self.assertFalse(payload["found"])
        # 6s is past the default 5000ms deadline: one poll, then give up
        self.assertEqual(client.call_tool.call_count, 1)

    def test_wait_for_rejects_an_empty_condition(self):
        """F41: an empty wait used to return found=true without observing."""
        client = Mock()
        result = browser_agent._handle_wait_for(client, {})
        self.assertIn("verifies nothing", result)
        client.call_tool.assert_not_called()

    def test_wait_for_rejects_blank_strings(self):
        client = Mock()
        result = browser_agent._handle_wait_for(
            client, {"selector": "   ", "text": ""})
        self.assertIn("verifies nothing", result)
        client.call_tool.assert_not_called()

    def test_fill_resolves_then_uses_real_keyboard_input(self):
        """F40: fill resolves the target, then types with fill_locator."""
        client = Mock()
        client.call_tool.side_effect = [
            json.dumps({"found": True, "tag": "input", "cssPath": "#inp",
                        "frame": "", "disabled": False}),
            "Filled via real input (top, submit=enter).\nurl=https://x navigated=no",
        ]
        result = browser_agent._handle_fill(
            client, {"selector": "#inp", "value": "hello", "press_enter": True})
        payload = json.loads(result)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["via"], "real-input")
        self.assertEqual(payload["css"], "#inp")
        names = [c[0][0] for c in client.call_tool.call_args_list]
        self.assertEqual(names, ["evaluate", "fill_locator"])
        expr = client.call_tool.call_args_list[0][0][1]["expression"]
        self.assertIn("jarvisIdent", expr)
        self.assertIn("placeholder", expr)
        self.assertIn("visible_inputs", expr)
        self.assertIn("score", expr)
        # F40: ONE submission mechanism — the daemon's explicit submit flag.
        # Enter must not also be dispatched by page JavaScript.
        self.assertNotIn("requestSubmit", expr)
        args = client.call_tool.call_args_list[1][0][1]
        self.assertEqual(args["css"], "#inp")
        self.assertEqual(args["value"], "hello")
        self.assertEqual(args["submit"], "enter")

    def test_fill_without_press_enter_asks_for_no_submit(self):
        client = Mock()
        client.call_tool.side_effect = [
            json.dumps({"found": True, "tag": "input", "cssPath": "#inp",
                        "frame": "", "disabled": False}),
            "Filled via real input (top, submit=none).\nurl=https://x navigated=no",
        ]
        browser_agent._handle_fill(client, {"selector": "#inp", "value": "hi"})
        args = client.call_tool.call_args_list[1][0][1]
        self.assertEqual(args["submit"], "none")
        self.assertNotIn("requestSubmit", json.dumps(args))

    def test_fill_js_scans_visible_inputs_and_reports_diagnostics(self):
        client = Mock()
        client.call_tool.return_value = json.dumps({"ok": True})
        browser_agent._handle_fill(client, {"selector": "Search Here", "value": "silo", "press_enter": True})
        expr = client.call_tool.call_args_list[0][0][1]["expression"]
        # placeholder-scoring fallback exists for a selector miss
        self.assertIn("placeholder", expr)
        self.assertIn("aria-label", expr)
        self.assertIn("visible_inputs", expr)
        self.assertIn("score", expr)

    def test_fill_total_miss_returns_placeholder_diagnostics(self):
        client = Mock()
        client.call_tool.return_value = json.dumps({
            "ok": False,
            "error": "no visible text input matches zzz",
            "visible_inputs": "<input placeholder='Search Here...'>",
        })
        result = browser_agent._handle_fill(client, {"selector": "zzz", "value": "v"})
        self.assertIn("Search Here...", result)
        self.assertIn("visible_inputs", result)

    def test_fill_mark_success_uses_mark_point_and_native_setter(self):
        session = {"marks": {3: {"cx": 210, "cy": 40, "tag": "input",
                                 "label": "Search Here..."}}}
        client = Mock()
        client.call_tool.return_value = json.dumps(
            {"ok": True, "tag": "input", "value": "silo", "visible": True})
        result = browser_agent._handle_fill_mark(
            client, session, {"index": 3, "value": "silo", "press_enter": True})
        self.assertIn("ok", result)
        expr = client.call_tool.call_args_list[0][0][1]["expression"]
        self.assertIn("elementFromPoint", expr)
        self.assertIn("210", expr)
        self.assertIn("40", expr)
        self.assertIn("silo", expr)
        self.assertIn("Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype", expr)
        self.assertIn("el.focus()", expr)
        self.assertIn("keydown", expr)
        self.assertIn("keyup", expr)
        # F40: ONE submission mechanism. requestSubmit() plus a synthetic
        # Enter double-submitted Enter-handled forms, so the legacy path no
        # longer calls it either.
        self.assertNotIn("requestSubmit", expr)
        # after-state evaluate ran (press_enter=True triggered it)
        self.assertGreaterEqual(len(client.call_tool.call_args_list), 2)

    def test_fill_mark_missing_mark_tells_model_to_look(self):
        client = Mock()
        result = browser_agent._handle_fill_mark(client, {}, {"index": 7, "value": "x"})
        self.assertIn("look", result.lower())
        client.call_tool.assert_not_called()

    def test_fill_mark_wrong_type_error_names_tag(self):
        session = {"marks": {1: {"cx": 5, "cy": 5, "tag": "a", "label": "Home"}}}
        client = Mock()
        # the JS returns this error when the marked element is not an input
        client.call_tool.return_value = json.dumps({
            "ok": False,
            "tag": "a",
            "error": "mark 1 is not a text input - the element there is a <a>",
        })
        result = browser_agent._handle_fill_mark(client, session,
                                                 {"index": 1, "value": "x"})
        self.assertIn("not a text input", result)
        self.assertIn("<a>", result)
        # and the JS carries the tag-naming recovery branch
        expr = client.call_tool.call_args[0][1]["expression"]
        self.assertIn("is not a text input", expr)

    def test_click_text_resolves_then_uses_real_input(self):
        client = Mock()
        client.call_tool.side_effect = [
            json.dumps({"found": True, "tag": "a", "label": "Movies",
                        "cssPath": "#movies", "frame": "", "disabled": False}),
            "Clicked via real input (top).\nurl=https://x/after navigated=yes",
        ]
        result = browser_agent._handle_click_text(client, {"text": "Movies"})
        payload = json.loads(result)
        self.assertTrue(payload["clicked"])
        self.assertEqual(payload["via"], "real-input")
        self.assertEqual(payload["css"], "#movies")
        self.assertEqual(payload["url"], "https://x/after")
        self.assertTrue(payload["navigated"])
        names = [c[0][0] for c in client.call_tool.call_args_list]
        self.assertEqual(names, ["evaluate", "click_locator"])
        expr = client.call_tool.call_args_list[0][0][1]["expression"]
        self.assertIn("Movies", expr)
        self.assertIn("jarvisIdent", expr)
        self.assertIn("click", expr.lower())
        # synchronous IIFE only - the daemon stringifies synchronously,
        # a Promise would come back as literal '{}'
        self.assertIn("(() => {", expr)
        self.assertNotIn("Promise", expr)
        self.assertNotIn(".click()", expr)

    def test_click_text_miss_never_dispatches_a_click(self):
        client = Mock()
        client.call_tool.return_value = json.dumps(
            {"found": False, "clicked": False, "error": "not found",
             "text": "Missing", "href": "https://x"})
        result = browser_agent._handle_click_text(client, {"text": "Missing"})
        # miss: no click primitive is dispatched at all
        self.assertEqual([c[0][0] for c in client.call_tool.call_args_list],
                         ["evaluate"])
        payload = json.loads(result)
        self.assertFalse(payload["clicked"])

    def test_click_text_refuses_a_disabled_target(self):
        client = Mock()
        client.call_tool.return_value = json.dumps(
            {"found": True, "tag": "button", "label": "Buy", "cssPath": "#buy",
             "frame": "", "disabled": True})
        result = browser_agent._handle_click_text(client, {"text": "Buy"})
        self.assertIn("disabled", result)
        self.assertEqual([c[0][0] for c in client.call_tool.call_args_list],
                         ["evaluate"])

    def test_click_mark_js_confirms_navigation(self):
        session = {"marks": {1: {"cx": 25, "cy": 30, "tag": "a", "label": "Home"}}}
        client = Mock()
        client.call_tool.side_effect = [
            json.dumps({"clicked": True, "tag": "a", "label": "Home",
                        "href": "https://x/before"}),
            '"https://x/after"',
            json.dumps({"title": "Page", "url": "https://x/after", "items": []}),
        ]
        with patch.object(browser_agent.time, "sleep"):
            result = browser_agent._handle_click_mark(client, session,
                                                      {"index": 1})
        payload = json.loads(result)
        self.assertTrue(payload["clicked"])
        self.assertEqual(payload["href_before"], "https://x/before")
        self.assertEqual(payload["url"], "https://x/after")
        self.assertTrue(payload["navigated"])
        expr = client.call_tool.call_args_list[0][0][1]["expression"]
        self.assertIn("elementFromPoint", expr)
        self.assertIn("25", expr)
        self.assertIn("30", expr)
        self.assertIn("(() => {", expr)
        self.assertNotIn("Promise", expr)
        self.assertEqual(
            client.call_tool.call_args_list[1][0][1]["expression"], "location.href")

    def test_click_mark_second_evaluate_failure_degrades_gracefully(self):
        session = {"marks": {1: {"cx": 25, "cy": 30, "tag": "a", "label": "Home"}}}
        client = Mock()
        client.call_tool.side_effect = [
            json.dumps({"clicked": True, "tag": "a", "href": "https://x/before"}),
            RuntimeError("browser closed"),
            json.dumps({"title": "Page", "url": "https://x/after", "items": []}),
        ]
        with patch.object(browser_agent.time, "sleep"):
            result = browser_agent._handle_click_mark(client, session,
                                                      {"index": 1})
        payload = json.loads(result)
        self.assertTrue(payload["clicked"])
        self.assertIn("unknown", payload["url"])
        self.assertFalse(payload["navigated"])

    def _look_capture_session(self, **overrides):
        capture = {"url": "https://x", "doc": "111", "mut": "7", "dpr": "1",
                   "tab_id": "t1", "capture_verified": True}
        capture.update(overrides)
        return {"look_scale": 1.0, "viewport_size": (1600, 900),
                "look_capture": capture}

    def _capture_probe(self):
        return json.dumps({"doc": 111, "mut": 7, "dpr": 1, "url": "https://x",
                           "title": "Page"})

    def test_click_point_js_construction(self):
        session = self._look_capture_session()
        client = Mock()
        client.call_tool.side_effect = [
            self._capture_probe(),
            json.dumps({"found": True, "tag": "button", "label": "Go",
                        "cssPath": "#go", "frame": "", "disabled": False,
                        "nonInteractive": False}),
            "Clicked via real input (top).\nurl=https://x navigated=no",
        ]
        result = browser_agent._handle_click_point(client, session,
                                                   {"x": 123, "y": 456})
        expr = client.call_tool.call_args_list[1][0][1]["expression"]
        self.assertIn("elementFromPoint", expr)
        self.assertIn("123", expr)
        self.assertIn("456", expr)
        self.assertIn("(() => {", expr)
        self.assertNotIn("Promise", expr)
        # F40: the click goes through the daemon's real mouse input on the
        # resolved identity, never element.click().
        names = [c[0][0] for c in client.call_tool.call_args_list]
        self.assertEqual(names, ["evaluate", "evaluate", "click_locator"])
        self.assertEqual(client.call_tool.call_args_list[2][0][1]["css"], "#go")
        payload = json.loads(result)
        self.assertTrue(payload["clicked"])
        self.assertEqual(payload["via"], "real-input")

    def test_click_point_refuses_coordinates_without_a_verified_look(self):
        """F39: a raw point must not bypass mark validation."""
        client = Mock()
        result = browser_agent._handle_click_point(client, {}, {"x": 10, "y": 10})
        self.assertIn("refused", result)
        client.call_tool.assert_not_called()

    def test_click_point_converts_look_image_coords_to_viewport(self):
        session = self._look_capture_session()
        session["look_scale"] = 0.8
        client = Mock()
        client.call_tool.side_effect = [
            self._capture_probe(),
            json.dumps({"found": True, "tag": "button", "label": "Go",
                        "cssPath": "#go", "frame": "", "disabled": False,
                        "nonInteractive": False}),
            "Clicked via real input (top).\nurl=https://x navigated=no",
        ]
        browser_agent._handle_click_point(client, session, {"x": 640, "y": 360})
        expr = client.call_tool.call_args_list[1][0][1]["expression"]
        # 640/0.8 = 800, 360/0.8 = 450 - converted server-side
        self.assertIn("800", expr)
        self.assertIn("450", expr)
        self.assertNotIn("640", expr)
        self.assertNotIn("360", expr)

    def test_verify_playing_static_on_identical_frames(self):
        from PIL import Image
        frames = [Image.new("RGB", (400, 300), (10, 10, 10)),
                  Image.new("RGB", (400, 300), (10, 10, 10))]
        client, paths = self._verify_client(frames)
        with patch.object(browser_agent.time, "sleep"):
            result = browser_agent._handle_verify_playing(client)
        self.assertIn("STATIC", result)
        self.assertNotIn("PLAYING", result)
        self.assertFalse(any(os.path.exists(p) for p in paths))

    def test_verify_playing_motion_alone_never_proves_playback(self):
        """F41: ads / spinners animating must not be reported as PLAYING."""
        from PIL import Image
        frames = [Image.new("RGB", (400, 300), (10, 10, 10)),
                  Image.new("RGB", (400, 300), (200, 200, 200))]
        client, paths = self._verify_client(frames)
        with patch.object(browser_agent.time, "sleep"):
            result = browser_agent._handle_verify_playing(client)
        self.assertIn("UNCERTAIN", result)
        self.assertNotIn("PLAYING", result)
        self.assertFalse(any(os.path.exists(p) for p in paths))

    def _media_client(self, probes):
        """Client whose evaluate returns the given media probes in order."""
        pending = list(probes)

        class MediaClient:
            def call_tool(self, name, args):
                if name != "evaluate":
                    raise AssertionError("only evaluate is expected here")
                return pending.pop(0) if pending else "{}"

        return MediaClient()

    def _player(self, current_time, paused=False, ended=False, ready=4,
                duration=120.0):
        return {
            "players": [{
                "tag": "video", "paused": paused, "ended": ended,
                "muted": False, "volume": 1.0, "currentTime": current_time,
                "duration": duration, "readyState": ready, "hasVideo": True,
                "src": "https://cdn.example/v.mp4",
            }],
            "url": "https://example/watch", "title": "t",
        }

    def test_verify_playing_reports_playing_from_advancing_media_time(self):
        client = self._media_client([
            json.dumps(self._player(3.0)),
            json.dumps(self._player(4.4)),
        ])
        with patch.object(browser_agent.time, "sleep"):
            result = browser_agent._handle_verify_playing(client)
        self.assertIn("PLAYING", result)
        self.assertIn("media time advanced", result)

    def test_verify_playing_reports_paused_from_the_element_state(self):
        client = self._media_client([
            json.dumps(self._player(3.0, paused=True)),
            json.dumps(self._player(3.0, paused=True)),
        ])
        with patch.object(browser_agent.time, "sleep"):
            result = browser_agent._handle_verify_playing(client)
        self.assertIn("PAUSED", result)
        self.assertNotIn("PLAYING", result)

    def test_verify_playing_reports_ended(self):
        client = self._media_client([
            json.dumps(self._player(120.0, paused=True, ended=True)),
            json.dumps(self._player(120.0, paused=True, ended=True)),
        ])
        with patch.object(browser_agent.time, "sleep"):
            result = browser_agent._handle_verify_playing(client)
        self.assertIn("ENDED", result)

    def test_verify_playing_is_uncertain_when_media_time_stalls(self):
        client = self._media_client([
            json.dumps(self._player(3.0, paused=False)),
            json.dumps(self._player(3.0, paused=False)),
        ])
        with patch.object(browser_agent.time, "sleep"):
            result = browser_agent._handle_verify_playing(client)
        self.assertIn("UNCERTAIN", result)
        self.assertNotIn("PLAYING", result)

    def test_verify_playing_reports_not_ready_before_metadata(self):
        client = self._media_client([
            json.dumps(self._player(0.0, paused=False, ready=0)),
            json.dumps(self._player(0.0, paused=False, ready=0)),
        ])
        with patch.object(browser_agent.time, "sleep"):
            result = browser_agent._handle_verify_playing(client)
        self.assertIn("NOT READY", result)


    def test_verify_playing_screenshot_failure_is_graceful(self):
        class BrokenClient:
            def call_tool(self, name, args):
                raise RuntimeError("browser closed")
        with patch.object(browser_agent.time, "sleep"):
            result = browser_agent._handle_verify_playing(BrokenClient())
        self.assertIn("verify_playing failed", result)
        self.assertIn("browser closed", result)

    def _verify_client(self, frames):
        from PIL import Image
        paths = []
        pending = list(frames)
        class VerifyClient:
            def call_tool(self, name, args):
                if name == "screenshot":
                    if pending:
                        pending.pop(0).save(args["path"], format="PNG")
                    paths.append(args["path"])
                    return "saved"
                return "ok"
        return VerifyClient(), paths

    def test_budget_exhaustion_forces_final_summary_turn(self):
        tool_call = {"id": "c", "name": "evaluate",
                     "arguments": {"expression": "1+1"}}
        seen = {"tools": None, "budget_note": False}
        client = FakeClient()

        def fake_model(history, tools):
            for message in history:
                if message.get("role") == "user" and "budget" in (message.get("content") or ""):
                    seen["budget_note"] = True
            if seen["budget_note"]:
                seen["tools"] = tools
                return ("Partial: I opened the page.", [])
            return (None, [tool_call])

        with patch.object(browser_agent, "ensure_brave_mcp_daemon",
                          return_value=True), \
             patch.object(browser_agent, "BraveMcpClient", return_value=client), \
             patch.object(browser_agent, "_model_turn", side_effect=fake_model), \
             patch.object(config, "BROWSER_AGENT_MAX_STEPS", 2), \
             patch.object(browser_agent, "append_activity_line") as log_line, \
             patch.object(browser_agent, "narrate_activity"), \
             patch.object(browser_agent, "truncate_activity_log"):
            result = browser_agent.run_browser_task("loop")
        self.assertEqual(result, "Partial: I opened the page.")
        # The step budget was spent and the forced summary claims no
        # completion: partial with the budget note as evidence.
        self.assertEqual(result.status, "partial")
        self.assertIn("step budget exhausted", result.evidence)
        # the final turn ran with tools withheld
        self.assertTrue(seen["budget_note"])
        self.assertEqual(seen["tools"], [])
        lines = [call.args[0] for call in log_line.call_args_list]
        self.assertTrue(any(line.startswith("RESULT partial: ") for line in lines))

    def test_tool_results_logged_truncated_and_never_base64(self):
        history = []
        client = Mock()
        client.call_tool.return_value = "x" * 500
        call = {"id": "c", "name": "evaluate", "arguments": {}}
        with patch.object(browser_agent, "append_activity_line") as log_line:
            browser_agent._run_one_tool(client, history, call, {})
        result_lines = [c.args[0] for c in log_line.call_args_list
                       if c.args[0].startswith("RESULT evaluate: ")]
        self.assertEqual(len(result_lines), 1)
        # cap is 200 chars + "..." marker + trailing newline
        self.assertLessEqual(len(result_lines[0]),
                             len("RESULT evaluate: ") + 200 + 3 + 1)
        # fill results stay at full length (short JSON)
        fill_result = json.dumps({"ok": False, "visible_inputs": "y" * 600})
        client.call_tool.return_value = fill_result
        fill_call = {"id": "f", "name": "fill",
                     "arguments": {"selector": "s", "value": "v"}}
        with patch.object(browser_agent, "append_activity_line") as log_line2:
            browser_agent._run_one_tool(client, history, fill_call, {})
        fill_lines = [c.args[0] for c in log_line2.call_args_list
                     if c.args[0].startswith("RESULT fill: ")]
        self.assertEqual(len(fill_lines), 1)
        self.assertIn("y" * 600, fill_lines[0])

    def test_look_result_logged_without_image_data(self):
        from PIL import Image
        def side_effect(name, args):
            if name == "screenshot":
                Image.new("RGB", (60, 60), (5, 5, 5)).save(
                    args["path"], format="PNG")
                return "saved"
            if name == "evaluate":
                return json.dumps({"elements": [], "url": "u", "title": "t"})
            return "ok"
        client = Mock()
        client.call_tool.side_effect = side_effect
        call = {"id": "l", "name": "look", "arguments": {}}
        with patch.object(browser_agent, "append_activity_line") as log_line:
            browser_agent._run_one_tool(client, [], call, {})
        logged = " ".join(str(c.args[0]) for c in log_line.call_args_list)
        # the RESULT line carries the text table only - never base64 image
        # data (a JPEG payload would show up as /9j/ in the log)
        self.assertNotIn("image_b64", logged)
        self.assertNotIn("/9j/", logged)
        self.assertTrue(any(str(c.args[0]).startswith("RESULT look: ")
                           for c in log_line.call_args_list))

    def test_image_plumbing_openai_and_gemini(self):
        # OpenAI parts array
        msgs = [{"role": "tool", "tool_call_id": "c1", "name": "look", "content": "hello", "image_b64": "abc123"}]
        out = browser_agent._to_openai_messages(msgs)
        self.assertIsInstance(out[0]["content"], list)
        self.assertEqual(out[0]["content"][0]["type"], "text")
        self.assertEqual(out[0]["content"][1]["type"], "image_url")
        self.assertIn("data:image/jpeg;base64,abc123", out[0]["content"][1]["image_url"]["url"])
        # text-only unchanged
        msgs2 = [{"role": "tool", "tool_call_id": "c2", "name": "navigate", "content": "done"}]
        out2 = browser_agent._to_openai_messages(msgs2)
        self.assertIsInstance(out2[0]["content"], str)
        # Gemini: the functionResponse part carries the JPEG inside ITS OWN
        # response (functionResponse.parts) and the id of the call it answers.
        msgs3 = [{"role": "tool", "tool_call_id": "c1", "name": "look", "content": "hello", "image_b64": "abc123"}]
        gout = browser_agent._to_gemini_contents([{"role": "system", "content": "sys"}, msgs3[0]])
        self.assertEqual(len(gout), 1)
        response = gout[0]["parts"][0]["functionResponse"]
        self.assertEqual(response["response"]["result"], "hello")
        self.assertEqual(response["id"], "c1")
        # The image is attached to the response it belongs to - no separate
        # message that could drift out of order.
        self.assertEqual(response["parts"][1]["inlineData"]["mimeType"], "image/jpeg")
        self.assertEqual(response["parts"][1]["inlineData"]["data"], "abc123")

    def test_gemini_text_only_tool_has_no_observation(self):
        msgs = [{"role": "tool", "tool_call_id": "c2", "name": "navigate", "content": "done"}]
        gout = browser_agent._to_gemini_contents(msgs)
        self.assertEqual(len(gout), 1)
        self.assertEqual(gout[0]["parts"][0]["functionResponse"]["response"]["result"], "done")

    def test_gemini_tool_image_ordering_intact(self):
        # assistant tool-call, look result with image, then a second tool-call:
        # model(functionCall) / user(functionResponse with image) / model(...)
        history = [
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "name": "look", "arguments": {}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "look",
             "content": "page", "image_b64": "img1"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c2", "name": "click_mark", "arguments": {"mark": 1}}]},
        ]
        gout = browser_agent._to_gemini_contents(history)
        self.assertEqual(len(gout), 3)
        self.assertIn("functionCall", gout[0]["parts"][0])
        self.assertEqual(gout[0]["parts"][0]["functionCall"]["id"], "c1")
        response = gout[1]["parts"][0]["functionResponse"]
        self.assertEqual(response["id"], "c1")
        self.assertEqual(response["parts"][1]["inlineData"]["data"], "img1")
        self.assertIn("functionCall", gout[2]["parts"][0])
        self.assertEqual(gout[2]["parts"][0]["functionCall"]["name"], "click_mark")

    def test_gemini_groups_parallel_results_with_ids(self):
        """F38: two results for one turn are ONE user content, id-correlated."""
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "name": "look", "arguments": {}},
                {"id": "c2", "name": "look", "arguments": {}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "name": "look", "content": "one"},
            {"role": "tool", "tool_call_id": "c2", "name": "look", "content": "two"},
        ]
        gout = browser_agent._to_gemini_contents(history)
        self.assertEqual(len(gout), 2)
        responses = [p["functionResponse"] for p in gout[1]["parts"]]
        self.assertEqual([r["id"] for r in responses], ["c1", "c2"])
        self.assertEqual(
            [r["response"]["result"] for r in responses], ["one", "two"])

    def test_gemini_preserves_thought_signature(self):
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "name": "look", "arguments": {},
                 "thought_signature": "sig-abc"},
            ]},
        ]
        gout = browser_agent._to_gemini_contents(history)
        call = gout[0]["parts"][0]["functionCall"]
        self.assertEqual(call["id"], "c1")
        self.assertEqual(gout[0]["parts"][0]["thoughtSignature"], "sig-abc")


    def test_adapter_capability_known_providers(self):
        for provider in ("fireworks", "groq", "openrouter", "gemini"):
            self.assertTrue(
                browser_agent._adapter_supports_tools_and_vision(provider),
                provider,
            )
        # F38: an unknown provider no longer inherits tool+vision support.
        self.assertFalse(
            browser_agent._adapter_supports_tools_and_vision("my-custom"))
        self.assertFalse(browser_agent._adapter_supports_tools_and_vision(""))
        # A REGISTERED openai-compatible provider that declares both
        # capabilities is still accepted.
        with patch("backend.services.model_registry.model_capabilities_for",
                   return_value={"tool_calling", "vision_input"}):
            self.assertTrue(
                browser_agent._adapter_supports_tools_and_vision(
                    "my-custom", "custom-vl-1"))
        # ... but one that cannot see images is rejected (fail closed).
        with patch("backend.services.model_registry.model_capabilities_for",
                   return_value={"tool_calling"}):
            self.assertFalse(
                browser_agent._adapter_supports_tools_and_vision(
                    "my-custom", "text-only-1"))

    def test_model_turn_rejects_incapable_adapter(self):
        with patch.dict(browser_agent._ADAPTER_TOOLS_AND_VISION, {"gemini": False}), \
             patch.object(browser_agent, "_resolve_browser_tool_model",
                          return_value=("gemini", "gemini-3.5-flash-lite")), \
             patch.object(browser_agent, "_call_gemini") as cg:
            with self.assertRaises(RuntimeError) as ctx:
                browser_agent._model_turn([], [])
        cg.assert_not_called()
        self.assertIn("selection rejected", str(ctx.exception))

    def test_model_turn_gemini_proceeds_when_capable(self):
        with patch.object(browser_agent, "_resolve_browser_tool_model",
                          return_value=("gemini", "gemini-3.5-flash-lite")), \
             patch.object(browser_agent, "_call_gemini",
                          return_value=("done", [])) as cg:
            text, tool_calls = browser_agent._model_turn([], [])
        cg.assert_called_once()
        self.assertEqual(text, "done")
        self.assertEqual(tool_calls, [])

    def test_clip_result_exempts_image(self):
        long_text = "a" * 20000
        clipped = browser_agent._clip_result(long_text)
        self.assertTrue(clipped.endswith("...[truncated]"))
        # image_b64 not clipped via _clip_result - ensure _to_openai_messages keeps full b64
        big_b64 = "a" * 20000
        msgs = [{"role": "tool", "tool_call_id": "c1", "name": "look", "content": "short", "image_b64": big_b64}]
        out = browser_agent._to_openai_messages(msgs)
        self.assertEqual(out[0]["content"][1]["image_url"]["url"], "data:image/jpeg;base64,%s" % big_b64)

    def test_prompt_contains_key_guidance(self):
        prompt = browser_agent._SYSTEM_PROMPT.lower()
        for kw in ("look first", "batch_probe", "wait_for", "click_mark"):
            self.assertIn(kw, prompt)
        # also check spoken-friendly summary guidance retained
        self.assertIn("spoken-friendly", prompt)

    def test_prompt_contains_completion_doctrine_and_vision_first_rules(self):
        prompt = browser_agent._SYSTEM_PROMPT.lower()
        # completion doctrine: goal-first every turn, stop when achieved
        self.assertIn("stop calling tools", prompt)
        self.assertIn("never keep improving a finished task", prompt)
        # vision-first: visual actions are the default loop
        self.assertIn("fill_mark", prompt)
        self.assertIn("look -> act visually", prompt)
        # DOM tools are fallback only; no understand_page when marks exist
        self.assertIn("fallback only", prompt)
        # F17: the model is told raw page JavaScript is unavailable, so it
        # never wastes a turn asking for evaluate.
        self.assertIn("evaluate) is not available", prompt)
        self.assertIn("typed read-only expressions", prompt)
        self.assertIn("never", prompt)
        # mark numbers are not understand_page indices
        self.assertIn("not understand_page indices", prompt)
        # cross-origin player doctrine + verify_playing
        self.assertIn("cross-origin", prompt)
        self.assertIn("verify_playing", prompt)
        # look is the ONLY image source; file tools never display images
        self.assertIn("the only tool that shows you an image", prompt)
        self.assertIn("file tools never", prompt)
        self.assertIn("never judge playback from a saved file", prompt)
        # file-tool advertising removed
        self.assertNotIn("read_file", prompt)
        self.assertNotIn("list_dir", prompt)

    def test_narration_phrases_include_virtual_tools(self):
        for name in ("look", "click_mark", "fill_mark", "verify_playing",
                     "batch_probe", "wait_for", "fill", "click_text", "click_point"):
            self.assertIn(name, browser_agent._NARRATION_PHRASES)

    def test_config_max_steps_default_50(self):
        saved = {k: os.environ.get(k) for k in ("JARVIS_BROWSER_AGENT_MAX_STEPS",)}
        try:
            os.environ.pop("JARVIS_BROWSER_AGENT_MAX_STEPS", None)
            importlib.reload(config)
            self.assertEqual(config.BROWSER_AGENT_MAX_STEPS, 50)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            importlib.reload(config)


class HistoryTrimTests(unittest.TestCase):
    """WI1: history image stripping - only latest K images survive."""

    def _make_history(self):
        # 3 look tool messages with image_b64, plus one non-look
        return [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "tool", "tool_call_id": "c1", "name": "look",
             "content": "Page: https://a.example\nTitle: A\n\nMarks 2\n1 | a | Home | 10,10", "image_b64": "aaa"},
            {"role": "tool", "tool_call_id": "c2", "name": "click_mark",
             "content": '{"clicked": true}'},
            {"role": "tool", "tool_call_id": "c3", "name": "look",
             "content": "Page: https://b.example\nTitle: B\n\nMarks 1\n1 | button | Go | 20,20", "image_b64": "bbb"},
            {"role": "tool", "tool_call_id": "c4", "name": "look",
             "content": "Page: https://c.example\nTitle: C\n\nMarks 0\nNo elements", "image_b64": "ccc"},
        ]

    def test_trim_keep_last_one(self):
        history = self._make_history()
        with patch.object(config, "BROWSER_AGENT_KEEP_LAST_IMAGES", 1):
            browser_agent._trim_look_history(history)
        # only last look keeps image
        self.assertNotIn("image_b64", history[2])
        self.assertNotIn("image_b64", history[4])
        self.assertIn("image_b64", history[5])
        self.assertEqual(history[5]["image_b64"], "ccc")
        # older contain tombstone and keep Page/Title
        self.assertIn("[earlier look image and marks omitted - use the latest look]", history[2]["content"])
        self.assertIn("Page: https://a.example", history[2]["content"])
        self.assertIn("Title: A", history[2]["content"])
        self.assertNotIn("Marks 2", history[2]["content"])
        self.assertIn("[earlier look image and marks omitted", history[4]["content"])
        self.assertIn("Page: https://b.example", history[4]["content"])
        # non-look untouched
        self.assertEqual(history[3]["content"], '{"clicked": true}')
        self.assertNotIn("image_b64", history[3])

    def test_trim_keep_last_two(self):
        history = self._make_history()
        with patch.object(config, "BROWSER_AGENT_KEEP_LAST_IMAGES", 2):
            browser_agent._trim_look_history(history)
        self.assertNotIn("image_b64", history[2])
        self.assertIn("image_b64", history[4])
        self.assertIn("image_b64", history[5])

    def test_trim_idempotent(self):
        history = self._make_history()
        with patch.object(config, "BROWSER_AGENT_KEEP_LAST_IMAGES", 1):
            browser_agent._trim_look_history(history)
            first = [dict(m) for m in history]
            browser_agent._trim_look_history(history)
            second = [dict(m) for m in history]
        self.assertEqual(first, second)

    def test_trim_leaves_latest_full_marks(self):
        history = self._make_history()
        with patch.object(config, "BROWSER_AGENT_KEEP_LAST_IMAGES", 1):
            browser_agent._trim_look_history(history)
        # latest keeps full content including marks table
        self.assertIn("No elements", history[5]["content"])
        self.assertIn("image_b64", history[5])

    def test_agent_loop_trims_before_model_turn(self):
        # Integration: _agent_loop calls trim before each model turn
        client = FakeClient()
        seen_histories = []

        def fake_model_turn(history, tools):
            # capture copy of history as seen by model (after trim)
            seen_histories.append([dict(m) for m in history])
            if len(seen_histories) == 1:
                # first turn: model returns a look tool call
                return None, [{"id": "c1", "name": "look", "arguments": {}}]
            if len(seen_histories) == 2:
                return None, [{"id": "c2", "name": "look", "arguments": {}}]
            return ("done", [])

        # need to mock look handler to return images so history grows
        original_handle_look = browser_agent._handle_look

        def fake_handle_look(client, session, stats=None):
            text = "Page: https://x%d.example\nTitle: T%d\n\nMarks 1\n1 | a | X | 10,10" % (len(seen_histories), len(seen_histories))
            b64 = "img%d" % len(seen_histories)
            session["marks"] = {1: {"cx": 10, "cy": 10, "tag": "a", "label": "X"}}
            session["look_scale"] = 1.0
            session["viewport_size"] = (100, 100)
            return text, b64

        with patch.object(browser_agent, "_model_turn", side_effect=fake_model_turn), \
             patch.object(browser_agent, "_handle_look", side_effect=fake_handle_look), \
             patch.object(browser_agent, "ensure_brave_mcp_daemon", return_value=True), \
             patch.object(browser_agent, "BraveMcpClient", return_value=client), \
             patch.object(browser_agent, "append_activity_line"), \
             patch.object(browser_agent, "narrate_activity"), \
             patch.object(browser_agent, "truncate_activity_log"), \
             patch.object(config, "BROWSER_AGENT_KEEP_LAST_IMAGES", 1), \
             patch.object(config, "BROWSER_AGENT_MAX_STEPS", 4):
            # also need to patch _MODEL_SESSION if needed but not for this test
            result = browser_agent.run_browser_task("task")
        # second model turn should have only 1 image in its history
        # first history had 0 images, second had 1, third should have trimmed to 1
        if len(seen_histories) >= 3:
            img_count = sum(1 for m in seen_histories[2] if m.get("image_b64"))
            self.assertEqual(img_count, 1)


class PayloadKnobTests(unittest.TestCase):
    """WI2: payload knobs width/quality with clamping."""

    def test_config_defaults_1280_70_1(self):
        # Isolate from the repo .env WITHOUT touching it: neutralise the dotenv
        # loader for the duration of the reload so only os.environ decides the
        # values. The old version deleted the REAL .env and restored it in a
        # `finally`; if the run was ever interrupted between the unlink and the
        # restore (crash, hard timeout, Ctrl+C, kill -9, power loss) the
        # developer's API keys were gone for good. Never operate on a secrets
        # file that the test did not create.
        saved = {k: os.environ.get(k) for k in (
            "JARVIS_BROWSER_AGENT_KEEP_LAST_IMAGES",
            "JARVIS_BROWSER_AGENT_LOOK_WIDTH",
            "JARVIS_BROWSER_AGENT_JPEG_QUALITY",
        )}
        try:
            for k in saved:
                os.environ.pop(k, None)
            with patch("dotenv.load_dotenv"):  # reload reads no .env file
                importlib.reload(config)
                self.assertEqual(config.BROWSER_AGENT_KEEP_LAST_IMAGES, 1)
                self.assertEqual(config.BROWSER_AGENT_LOOK_WIDTH, 1280)
                self.assertEqual(config.BROWSER_AGENT_JPEG_QUALITY, 70)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            importlib.reload(config)

    def test_config_clamps_width_and_quality(self):
        # Same non-destructive isolation as test_config_defaults_1280_70_1:
        # neutralise dotenv instead of deleting the real .env.
        saved = {k: os.environ.get(k) for k in (
            "JARVIS_BROWSER_AGENT_LOOK_WIDTH",
            "JARVIS_BROWSER_AGENT_JPEG_QUALITY",
        )}
        try:
            with patch("dotenv.load_dotenv"):
                os.environ["JARVIS_BROWSER_AGENT_LOOK_WIDTH"] = "300"
                os.environ["JARVIS_BROWSER_AGENT_JPEG_QUALITY"] = "10"
                importlib.reload(config)
                self.assertEqual(config.BROWSER_AGENT_LOOK_WIDTH, 640)
                self.assertEqual(config.BROWSER_AGENT_JPEG_QUALITY, 40)
                os.environ["JARVIS_BROWSER_AGENT_LOOK_WIDTH"] = "5000"
                os.environ["JARVIS_BROWSER_AGENT_JPEG_QUALITY"] = "200"
                importlib.reload(config)
                self.assertEqual(config.BROWSER_AGENT_LOOK_WIDTH, 1920)
                self.assertEqual(config.BROWSER_AGENT_JPEG_QUALITY, 95)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            importlib.reload(config)

    def test_width_1024_produces_resized_image_and_actual_frame_text(self):
        from PIL import Image
        class LookClient:
            def call_tool(self, name, args):
                if name == "evaluate":
                    return json.dumps({"elements": [{"x": 400, "y": 100, "w": 200, "h": 50, "tag": "input", "label": "Search"}], "url": "https://example.com", "title": "Test"})
                if name == "screenshot":
                    Image.new("RGB", (1600, 900), (0, 128, 0)).save(args["path"], format="PNG")
                    return "saved"
                return "ok"
        session = {}
        with patch.object(config, "BROWSER_AGENT_LOOK_WIDTH", 1024), \
             patch.object(config, "BROWSER_AGENT_JPEG_QUALITY", 70):
            text, b64 = browser_agent._handle_look(LookClient(), session)
        # width 1024 should downscale 1600 -> 1024 (not 1280)
        self.assertAlmostEqual(session["look_scale"], 1024 / 1600)
        self.assertEqual(session["viewport_size"], (1600, 900))
        self.assertIn("look image is 1024x", text.lower())
        self.assertIn("the real viewport is 1600x900", text.lower())
        self.assertNotIn("1280x", text.lower())
        # image actually 1024 wide
        import base64, io
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw))
        self.assertEqual(img.size[0], 1024)
        # marks in ORIGINAL viewport px (500) still, but table shows look-image px (320)
        self.assertEqual(session["marks"][1]["cx"], 500)
        self.assertIn("320", text)  # 500 * 0.64 =320

    def test_clamp_via_handler_width_300_and_quality_10(self):
        from PIL import Image
        import io, base64
        class LookClient:
            def call_tool(self, name, args):
                if name == "evaluate":
                    return json.dumps({"elements": [], "url": "u", "title": "t"})
                if name == "screenshot":
                    Image.new("RGB", (2000, 1000), (1, 1, 1)).save(args["path"], format="PNG")
                    return "saved"
                return "ok"
        session = {}
        # patch config to out-of-range values, handler should clamp
        with patch.object(config, "BROWSER_AGENT_LOOK_WIDTH", 300), \
             patch.object(config, "BROWSER_AGENT_JPEG_QUALITY", 10):
            text, b64 = browser_agent._handle_look(LookClient(), session)
        # 300 clamped to 640, so scale 0.32, image width 640
        self.assertAlmostEqual(session["look_scale"], 640 / 2000)
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw))
        self.assertEqual(img.size[0], 640)
        # quality clamped to 40 still produces valid JPEG (just check header)
        self.assertTrue(raw.startswith(b"\xff\xd8"))


class LookPipelineTests(unittest.TestCase):
    """WI3: look pipeline reorder - evaluate before screenshot."""

    def test_evaluate_called_before_screenshot(self):
        from PIL import Image
        call_order = []
        class OrderedClient:
            def call_tool(self, name, args):
                call_order.append(name)
                if name == "evaluate":
                    return json.dumps({"elements": [], "url": "https://example.com", "title": "Test"})
                if name == "screenshot":
                    Image.new("RGB", (100, 100), (0, 0, 0)).save(args["path"], format="PNG")
                    return "saved"
                return "ok"
        browser_agent._handle_look(OrderedClient(), {})
        self.assertIn("evaluate", call_order)
        self.assertIn("screenshot", call_order)
        self.assertLess(call_order.index("evaluate"), call_order.index("screenshot"))


class ModelSessionTests(unittest.TestCase):
    """WI4: model calls reuse single requests.Session."""

    def test_call_openai_compatible_posts_via_shared_session(self):
        fake_resp = FakeResponse({"choices": [{"message": {"content": "hi"}}]})
        messages = [{"role": "user", "content": "go"}]
        tools = []
        with patch.object(browser_agent._MODEL_SESSION, "post", return_value=fake_resp) as mock_post, \
             patch.object(config, "BROWSER_AGENT_MODEL", "m/pro"), \
             patch.object(config, "BROWSER_AGENT_PROVIDER", "fireworks"), \
             patch.object(config, "BROWSER_AGENT_REASONING_EFFORT", "medium"):
            text, calls = browser_agent._call_openai_compatible(
                "http://fw/v1/chat/completions", "k-fw", messages, tools
            )
        mock_post.assert_called_once()
        self.assertEqual(text, "hi")
        # also verify session object exists and is a Session
        import requests
        self.assertIsInstance(browser_agent._MODEL_SESSION, requests.Session)

    def test_call_gemini_posts_via_shared_session(self):
        fake_resp = FakeResponse({"candidates": [{"content": {"parts": [{"text": "Done."}]}}]})
        with patch.object(browser_agent._MODEL_SESSION, "post", return_value=fake_resp) as mock_post, \
             patch.object(config, "GEMINI_API_KEY", "k"), \
             patch.object(config, "BROWSER_AGENT_MODEL", "gemini-2.0-flash"):
            text, calls = browser_agent._call_gemini([{"role": "user", "content": "go"}], [])
        mock_post.assert_called_once()
        self.assertEqual(text, "Done.")


class StopwatchInstrumentationTests(unittest.TestCase):
    """STOPWATCH lines: tool timing, model timing, retries, end summary."""

    _MODEL_RE = re.compile(
        r"^STOPWATCH model step=(\d+) start=\d{2}:\d{2}:\d{2}\.\d{3} "
        r"dur_ms=\d+ ok=(true|false) req_bytes=\d+ resp_bytes=\d+$"
    )
    _TOOL_RE = re.compile(
        r"^STOPWATCH tool=(\S+) start=\d{2}:\d{2}:\d{2}\.\d{3} dur_ms=\d+$"
    )

    def _lines(self, log_line):
        return [c.args[0] for c in log_line.call_args_list]

    def test_tool_stopwatch_line_logged_and_interleaved(self):
        client = FakeClient()
        history = []
        call = {"id": "c1", "name": "navigate",
                "arguments": {"url": "https://example.com"}}
        with patch.object(browser_agent, "append_activity_line") as log_line, \
             patch.object(browser_agent, "narrate_activity"):
            browser_agent._run_one_tool(client, history, call, {})
        lines = self._lines(log_line)
        tool_idx = next(i for i, l in enumerate(lines)
                        if l.startswith("TOOL navigate "))
        sw_idx = next(i for i, l in enumerate(lines)
                      if l.startswith("STOPWATCH tool=navigate "))
        result_idx = next(i for i, l in enumerate(lines)
                          if l.startswith("RESULT navigate: "))
        self.assertTrue(tool_idx < sw_idx < result_idx)
        self.assertIsNotNone(self._TOOL_RE.match(lines[sw_idx].strip()))
        self.assertEqual(len(history), 1)

    def test_model_stopwatch_line_logged(self):
        with patch.object(browser_agent, "_model_turn",
                          return_value=("done", [])), \
             patch.object(browser_agent, "append_activity_line") as log_line:
            text, calls = browser_agent._model_turn_with_retries(
                [{"role": "user", "content": "go"}], [], step=0)
        self.assertEqual((text, calls), ("done", []))
        lines = self._lines(log_line)
        model_line = next(l for l in lines
                          if l.startswith("STOPWATCH model step=0 "))
        m = self._MODEL_RE.match(model_line.strip())
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "0")
        self.assertEqual(m.group(2), "true")
        self.assertGreater(
            int(m.group(0).split("req_bytes=")[1].split()[0]), 0)

    def test_model_retry_line_logged_when_retry_wrapper_fires(self):
        def flaky(history, tools):
            flaky.calls += 1
            if flaky.calls == 1:
                raise RuntimeError("connection reset")
            return "done", []
        flaky.calls = 0
        with patch.object(browser_agent, "_model_turn", side_effect=flaky), \
             patch.object(browser_agent, "append_activity_line") as log_line, \
             patch.object(browser_agent.time, "sleep"):
            text, calls = browser_agent._model_turn_with_retries(
                [], [], step=2)
        self.assertEqual((text, calls), ("done", []))
        lines = self._lines(log_line)
        retry_line = next(l for l in lines
                          if l.startswith("STOPWATCH model retry step=2 "))
        self.assertIsNotNone(re.match(
            r"^STOPWATCH model retry step=2 attempt=0 dur_ms=\d+$",
            retry_line.strip()))
        self.assertTrue(any(l.startswith("STOPWATCH model step=2 ")
                            and "ok=true" in l for l in lines))

    def test_model_total_failure_logs_ok_false_and_raises(self):
        with patch.object(browser_agent, "_model_turn",
                          side_effect=RuntimeError("boom")), \
             patch.object(browser_agent, "append_activity_line") as log_line, \
             patch.object(browser_agent.time, "sleep"):
            with self.assertRaises(RuntimeError):
                browser_agent._model_turn_with_retries([], [], step=1)
        lines = self._lines(log_line)
        retries = [l for l in lines
                   if l.startswith("STOPWATCH model retry step=1 ")]
        self.assertEqual(len(retries), 2)
        fail_line = next(l for l in lines
                         if l.startswith("STOPWATCH model step=1 "))
        m = self._MODEL_RE.match(fail_line.strip())
        self.assertIsNotNone(m)
        self.assertEqual(m.group(2), "false")

    def test_summary_lines_on_task_end(self):
        tool_call = {"id": "c1", "name": "navigate",
                     "arguments": {"url": "https://example.com"}}
        with patch.object(browser_agent, "ensure_brave_mcp_daemon",
                          return_value=True), \
             patch.object(browser_agent, "BraveMcpClient",
                          return_value=FakeClient()), \
             patch.object(browser_agent, "_model_turn",
                          side_effect=[(None, [tool_call]), ("Done.", [])]), \
             patch.object(browser_agent, "append_activity_line") as log_line, \
             patch.object(browser_agent, "narrate_activity"), \
             patch.object(browser_agent, "truncate_activity_log"):
            result = browser_agent.run_browser_task("task")
        self.assertEqual(result, "Done.")
        lines = self._lines(log_line)
        summary = next(l for l in lines
                       if l.startswith("STOPWATCH summary "))
        self.assertIsNotNone(re.match(
            r"^STOPWATCH summary total_ms=\d+ model_calls=2 "
            r"total_model_ms=\d+ tool_calls=1 total_tool_ms=\d+$",
            summary.strip()))
        self.assertTrue(any(l.startswith("STOPWATCH slowest model step=")
                            for l in lines))
        self.assertTrue(any(l.startswith("STOPWATCH slowest tool navigate ")
                            for l in lines))
        self.assertFalse(any(l.startswith("STOPWATCH overshoot_ms=")
                             for l in lines))

    def test_summary_lines_on_timeout(self):
        def fake_monotonic():
            fake_monotonic.calls += 1
            return 0.0 if fake_monotonic.calls == 1 else 481.0
        fake_monotonic.calls = 0
        with patch.object(browser_agent, "ensure_brave_mcp_daemon",
                          return_value=True), \
             patch.object(browser_agent, "BraveMcpClient",
                          return_value=FakeClient()), \
             patch.object(browser_agent, "_model_turn",
                          return_value=(None, [])), \
             patch.object(browser_agent.time, "monotonic",
                          side_effect=fake_monotonic), \
             patch.object(config, "BROWSER_AGENT_TIMEOUT", 180), \
             patch.object(browser_agent, "append_activity_line") as log_line, \
             patch.object(browser_agent, "narrate_activity"), \
             patch.object(browser_agent, "truncate_activity_log"):
            result = browser_agent.run_browser_task("slow")
        self.assertTrue(result.startswith("TASK NOT COMPLETED"))
        self.assertIn("timed out after 180s", result)
        lines = self._lines(log_line)
        summary = next(l for l in lines
                       if l.startswith("STOPWATCH summary "))
        self.assertIsNotNone(re.match(
            r"^STOPWATCH summary total_ms=\d+ model_calls=0 "
            r"total_model_ms=0 tool_calls=0 total_tool_ms=0$",
            summary.strip()))
        self.assertTrue(any(l.startswith("STOPWATCH overshoot_ms=")
                            for l in lines))


class TurnChurnReductionTests(unittest.TestCase):
    """WI1-RESULT-TRUST doctrine, WI2-off-viewport marks, WI3-honest clicks."""

    def test_result_trust_doctrine_appended_to_system_prompt(self):
        prompt = browser_agent._SYSTEM_PROMPT
        self.assertIn("RESULT-TRUST DOCTRINE", prompt)
        self.assertIn("trust the result and plan the next action", prompt)
        self.assertIn("wait_for on expected text or url instead of a look", prompt)
        self.assertIn("marks only become stale after a navigation", prompt)
        self.assertIn("never call look just to double-check", prompt)
        # Append-only: existing doctrine text must still be present
        self.assertIn("COMPLETION DOCTRINE", prompt)
        self.assertIn("Default browser loop: look -> act visually", prompt)
        self.assertIn("If click_element says an index is stale", prompt)

    def test_off_screen_annotation_in_marks_table(self):
        from PIL import Image
        def call_tool(name, args):
            if name == "screenshot":
                path = args["path"]
                img = Image.new("RGB", (100, 100), color=(255, 0, 0))
                img.save(path, format="PNG")
                return "saved"
            if name == "evaluate":
                return json.dumps({"elements": [
                    {"x": 10, "y": 20, "w": 30, "h": 20, "tag": "a", "label": "Home", "inView": True},
                    {"x": 50, "y": 6000, "w": 100, "h": 30, "tag": "button", "label": "Below", "inView": False},
                ], "url": "https://x", "title": "t"})
            return "ok"
        client = Mock()
        client.call_tool.side_effect = call_tool
        text, b64 = browser_agent._handle_look(client, {})
        self.assertIn("[off-screen]", text)
        self.assertIn("Below", text)
        self.assertIn("Home", text)

    def test_click_mark_guard_on_off_screen_mark(self):
        session = {"marks": {1: {"cx": 25, "cy": 30, "tag": "a", "label": "Home", "inView": True},
                             2: {"cx": 50, "cy": 6000, "tag": "button", "label": "Below", "inView": False}}}
        # Off-screen mark: guard returns early without calling evaluate
        client = Mock()
        result_guard = browser_agent._handle_click_mark(
            client, session, {"index": 2})
        self.assertIn("off-screen", result_guard.lower())
        self.assertIn("scroll it into view", result_guard.lower())
        client.call_tool.assert_not_called()

    def test_click_point_non_interactive_feedback(self):
        session = {"look_scale": 1.0, "viewport_size": (1600, 900),
                   "look_capture": {"url": "https://x/before", "doc": "111",
                                    "mut": "7", "dpr": "1"}}
        client = Mock()
        # Simulate a click on a non-interactive div (ancestor walk finds nothing)
        client.call_tool.side_effect = [
            json.dumps({"doc": 111, "mut": 7, "dpr": 1,
                        "url": "https://x/before", "title": "Page"}),
            json.dumps({"found": True, "tag": "div", "label": "",
                        "cssPath": "div.wrap", "frame": "", "disabled": False,
                        "nonInteractive": True}),
            "Clicked via real input (top).\nurl=https://x/before navigated=no",
        ]
        result = browser_agent._handle_click_point(client, session, {"x": 100, "y": 200})
        payload = json.loads(result)
        self.assertTrue(payload["clicked"])
        self.assertIn("feedback", payload)
        self.assertIn("non-interactive", payload["feedback"])
        self.assertIn("<div>", payload["feedback"])
        self.assertIn("probably no effect", payload["feedback"])

    def test_click_mark_non_interactive_feedback_also_emitted(self):
        client = Mock()
        session = {"marks": {1: {"cx": 25, "cy": 30, "tag": "div", "label": "", "inView": True}}}
        client.call_tool.side_effect = [
            json.dumps({"clicked": True, "tag": "div", "label": "",
                        "href": "https://x/before", "nonInteractive": True}),
            '"https://x/before"',
            json.dumps({"title": "Page", "url": "https://x/before", "items": []}),
        ]
        with patch.object(browser_agent.time, "sleep"):
            result = browser_agent._handle_click_mark(client, session, {"index": 1})
        payload = json.loads(result)
        self.assertTrue(payload.get("nonInteractive"))
        self.assertIn("feedback", payload)
        self.assertIn("non-interactive", payload["feedback"])

    def test_after_state_present_on_click_mark_result(self):
        session = {"marks": {1: {"cx": 25, "cy": 30, "tag": "a", "label": "Home"}}}
        client = Mock()
        client.call_tool.side_effect = [
            json.dumps({"clicked": True, "tag": "a", "label": "Home",
                        "href": "https://x/before"}),
            '"https://x/after"',
            json.dumps({"title": "New Page", "url": "https://x/after",
                        "items": [{"tag": "a", "label": "Home"},
                                  {"tag": "button", "label": "Go"}]}),
        ]
        with patch.object(browser_agent.time, "sleep"):
            result = browser_agent._handle_click_mark(client, session,
                                                      {"index": 1})
        payload = json.loads(result)
        self.assertIn("after", payload)
        self.assertIn("Home", payload["after"])
        self.assertIn("button", payload["after"])
        self.assertIn("New Page", payload["after"])

    def test_after_state_absent_when_evaluate_fails(self):
        session = {"marks": {1: {"cx": 25, "cy": 30, "tag": "a", "label": "Home"}}}
        client = Mock()
        client.call_tool.side_effect = [
            json.dumps({"clicked": True, "tag": "a", "label": "Home",
                        "href": "https://x/before"}),
            '"https://x/after"',
            RuntimeError("browser closed for after-state"),
        ]
        with patch.object(browser_agent.time, "sleep"):
            result = browser_agent._handle_click_mark(client, session,
                                                      {"index": 1})
        payload = json.loads(result)
        self.assertTrue(payload["clicked"])
        self.assertNotIn("after", payload)

    def test_daemon_tools_pruned_from_allowlist(self):
        self.assertNotIn("click_element", browser_agent._TOOL_ALLOWLIST)
        self.assertNotIn("fill_element", browser_agent._TOOL_ALLOWLIST)
        self.assertNotIn("understand_page", browser_agent._TOOL_ALLOWLIST)
        self.assertIn("evaluate", browser_agent._TOOL_ALLOWLIST)
        self.assertIn("navigate", browser_agent._TOOL_ALLOWLIST)
        self.assertIn("open_brave", browser_agent._TOOL_ALLOWLIST)

    def test_daemon_tool_pruning_prompt_sentence_appended(self):
        prompt = browser_agent._SYSTEM_PROMPT
        self.assertIn("daemon inventory tools (click_element, fill_element, understand_page)", prompt)
        self.assertIn("no longer exposed to the model", prompt)
        self.assertIn("mark numbers from the look table are the only index space", prompt)
        self.assertIn("Use click_mark / click_point / click_text for all clicks", prompt)
