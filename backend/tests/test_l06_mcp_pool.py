"""L-6: pooled MCP session + tool-list cache + confirmation-gap prewarm.

Verified 2026-10-03: every task paid initialize + notifications/initialized
plus a tools/list round trip on a throwaway session while the daemon itself
stays warm. These tests prove the pool pays that handshake once per process
lifetime (per daemon generation), the cache kills the per-task tools/list,
a restarted daemon heals exactly once, and every mock/patched path behaves
exactly as before (never pooled, always closed).
"""
import os
import threading
import time
import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch

import requests

from backend.core import brain
from backend.services import brave_mcp_client
from backend.services import browser_agent
from backend.services.brave_mcp_client import BraveMcpClient


_INIT = {"jsonrpc": "2.0", "id": 1, "result": {
    "protocolVersion": "2025-03-26", "capabilities": {},
    "serverInfo": {"name": "brave-control", "version": "1.0.0"},
}}

_DEFS = [
    {"name": "navigate", "description": "d", "inputSchema": {"type": "object"}},
    {"name": "evaluate", "description": "e", "inputSchema": {"type": "object"}},
]


class FakeHttpResponse:
    def __init__(self, payload=None, headers=None, status=200):
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}
        self.status_code = status
        self.text = ""

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError("HTTP %d" % self.status_code)
            err.response = self
            raise err
        return None

    def json(self):
        return self._payload


def _http_404():
    err = requests.HTTPError("404 Client Error: Not Found")
    err.response = FakeHttpResponse({}, status=404)
    return err


class FakeDaemonTransport:
    """Fake requests.Session routing MCP methods; counts every call."""

    def __init__(self, tools=None, fail_first_404=False,
                 fail_first_error=None):
        self.posts = []
        self.tools = list(tools if tools is not None else _DEFS)
        self._fail_404 = fail_first_404
        self._fail_error = fail_first_error
        self.closed = 0

    def _resp(self, payload, session=None):
        headers = {"mcp-session-id": session} if session else {}
        return FakeHttpResponse(payload, headers=headers)

    def _maybe_fail(self):
        if self._fail_404:
            self._fail_404 = False
            raise _http_404()
        if self._fail_error is not None:
            message, self._fail_error = self._fail_error, None
            return {"jsonrpc": "2.0", "id": -1,
                    "error": {"code": -32000, "message": message}}
        return None

    def post(self, url, headers=None, json=None, timeout=None):
        body = dict(json or {})
        method = body.get("method")
        self.posts.append(method)
        if method == "initialize":
            return self._resp(dict(_INIT), session="sess-1")
        if method == "notifications/initialized":
            return self._resp({})
        if method == "tools/list":
            failed = self._maybe_fail()
            if failed is not None:
                return self._resp(failed)
            return self._resp({"jsonrpc": "2.0", "id": body.get("id"),
                               "result": {"tools": self.tools}})
        if method == "tools/call":
            failed = self._maybe_fail()
            if failed is not None:
                return self._resp(failed)
            name = (body.get("params") or {}).get("name")
            return self._resp({"jsonrpc": "2.0", "id": body.get("id"),
                               "result": {"content": [{"type": "text",
                                                       "text": "daemon-ok:%s"
                                                               % name}]}})
        raise AssertionError("unexpected MCP method %r" % (method,))

    def delete(self, url, headers=None, timeout=None):
        return self._resp({})

    def close(self):
        self.closed += 1

    def count(self, method):
        return sum(1 for m in self.posts if m == method)


def _real_client(daemon, **kw):
    client = BraveMcpClient(base_url="http://x/mcp", token="tok",
                            timeout=5, **kw)
    client.connect()
    return client


@contextmanager
def _fake_transport(daemon):
    """Hold the Session patch open for the WHOLE test: reconnect() builds
    a fresh Session mid-test, and it must get the fake too — otherwise the
    heal path dials the real network."""
    with patch.object(brave_mcp_client.requests, "Session",
                      return_value=daemon):
        yield daemon


class SessionDeathTests(unittest.TestCase):
    def test_plain_404_is_session_death(self):
        self.assertTrue(brave_mcp_client._is_session_death(_http_404()))

    def test_invalid_session_message_is_session_death(self):
        self.assertTrue(brave_mcp_client._is_session_death(
            RuntimeError("Invalid session deadbeef")))
        self.assertTrue(brave_mcp_client._is_session_death(
            RuntimeError("unknown session id")))

    def test_other_failures_are_not_session_death(self):
        self.assertFalse(brave_mcp_client._is_session_death(
            RuntimeError("HTTP 500")))
        self.assertFalse(brave_mcp_client._is_session_death(
            RuntimeError("tool not found: frobnicate")))
        self.assertFalse(brave_mcp_client._is_session_death(
            requests.Timeout("timed out")))
        self.assertFalse(brave_mcp_client._is_session_death(
            RuntimeError("connection refused")))

    def test_404_heals_once_then_replays(self):
        daemon = FakeDaemonTransport(fail_first_404=True)
        with _fake_transport(daemon):
            client = _real_client(daemon)
            text = client.call_tool("evaluate", {"expression": "1+1"})
        self.assertEqual(text, "daemon-ok:evaluate")
        self.assertEqual(client.reconnects, 1)
        # connect(2) + failed call(1) + reconnect handshake(2) + replay(1).
        self.assertEqual(daemon.count("initialize"), 2)
        self.assertEqual(daemon.count("tools/call"), 2)

    def test_invalid_session_error_body_heals(self):
        daemon = FakeDaemonTransport(
            fail_first_error="Invalid session deadbeef")
        with _fake_transport(daemon):
            client = _real_client(daemon)
            text = client.call_tool("list_tabs", {})
        self.assertEqual(text, "daemon-ok:list_tabs")
        self.assertEqual(client.reconnects, 1)

    def test_second_session_death_propagates(self):
        daemon = FakeDaemonTransport()
        with _fake_transport(daemon):
            client = _real_client(daemon)
            calls = {"n": 0}
            orig_post = daemon.post

            def always_404(url, headers=None, json=None, timeout=None):
                if (json or {}).get("method") not in (
                        "initialize", "notifications/initialized"):
                    calls["n"] += 1
                    raise _http_404()
                return orig_post(url, headers=headers, json=json,
                                 timeout=timeout)

            daemon.post = always_404
            with self.assertRaises(requests.HTTPError):
                client.call_tool("evaluate", {})
        self.assertEqual(client.reconnects, 1)
        self.assertEqual(calls["n"], 2)  # original + exactly one replay

    def test_tool_errors_and_500s_never_heal(self):
        daemon = FakeDaemonTransport()
        with _fake_transport(daemon):
            client = _real_client(daemon)

            def error_body(url, headers=None, json=None, timeout=None):
                if (json or {}).get("method") == "tools/call":
                    return FakeHttpResponse(
                        {"jsonrpc": "2.0", "id": 9,
                         "error": {"code": -32602,
                                   "message": "tool not found"}})
                return FakeDaemonTransport.post(
                    daemon, url, headers=headers, json=json, timeout=timeout)

            daemon.post = error_body
            with self.assertRaises(RuntimeError) as ctx:
                client.call_tool("frobnicate", {})
        self.assertIn("tool not found", str(ctx.exception))
        self.assertEqual(client.reconnects, 0)
        self.assertEqual(daemon.count("initialize"), 1)


class PoolTests(unittest.TestCase):
    def setUp(self):
        self._env = dict(os.environ)
        os.environ.pop("JARVIS_MCP_POOL", None)
        os.environ.pop("JARVIS_MCP_PREWARM", None)
        browser_agent.reset_mcp_pool()

    def tearDown(self):
        browser_agent.reset_mcp_pool()
        os.environ.clear()
        os.environ.update(self._env)

    def _borrow(self, daemon, timeout=5):
        with patch.object(brave_mcp_client.requests, "Session",
                          return_value=daemon):
            return browser_agent._borrow_mcp_client(timeout)

    def test_sequential_borrows_reuse_one_session(self):
        daemon = FakeDaemonTransport()
        c1, cached1, pooled1 = self._borrow(daemon)
        self.assertTrue(pooled1)
        self.assertIsNone(cached1)
        browser_agent._note_pooled_tools(c1, [{"name": "navigate"}])
        browser_agent._return_mcp_client(c1, pooled1)
        self.assertEqual(daemon.closed, 0)  # returned, NOT closed

        c2, cached2, pooled2 = self._borrow(daemon)
        try:
            self.assertTrue(pooled2)
            self.assertIs(c2, c1)
            self.assertEqual(cached2, [{"name": "navigate"}])
        finally:
            browser_agent._return_mcp_client(c2, pooled2)
        self.assertEqual(daemon.count("initialize"), 1)
        self.assertEqual(daemon.closed, 0)

    def test_concurrent_holder_forces_private_client(self):
        d1, d2 = FakeDaemonTransport(), FakeDaemonTransport()
        with patch.object(brave_mcp_client.requests, "Session",
                          side_effect=[d1, d2]):
            c1, _, p1 = browser_agent._borrow_mcp_client(5)
            c2, _, p2 = browser_agent._borrow_mcp_client(5)
        self.assertTrue(p1)
        self.assertFalse(p2)
        self.assertIsNot(c1, c2)
        browser_agent._return_mcp_client(c1, p1)
        browser_agent._return_mcp_client(c2, p2)
        self.assertEqual(d1.closed, 0)  # pooled holder goes back to the slot
        self.assertEqual(d2.closed, 1)  # private client closed as before
        self.assertIs(browser_agent._POOLED_CLIENT, c1)

    def test_ttl_expiry_evicts_and_closes(self):
        d1, d2 = FakeDaemonTransport(), FakeDaemonTransport()
        with patch.object(brave_mcp_client.requests, "Session",
                          side_effect=[d1, d2]):
            c1, _, p1 = browser_agent._borrow_mcp_client(5)
            browser_agent._return_mcp_client(c1, p1)
            browser_agent._POOLED_LAST_USE = 0.0  # age past the TTL
            c2, cached2, p2 = browser_agent._borrow_mcp_client(5)
        try:
            self.assertIsNot(c2, c1)
            self.assertIsNone(cached2)  # eviction drops the tool cache too
        finally:
            browser_agent._return_mcp_client(c2, p2)
        self.assertEqual(d1.closed, 1)

    def test_patched_factory_takes_legacy_path(self):
        mock_client = Mock()
        mock_client.base_url = "http://x/mcp"
        with patch.object(browser_agent, "BraveMcpClient",
                          return_value=mock_client):
            client, cached, pooled = browser_agent._borrow_mcp_client(5)
            self.assertIs(client, mock_client)
            self.assertFalse(pooled)
            self.assertIsNone(cached)
            browser_agent._return_mcp_client(client, pooled)
        mock_client.close.assert_called_once_with()
        self.assertIsNone(browser_agent._POOLED_CLIENT)

    def test_pool_disabled_flag_means_private_clients(self):
        daemon = FakeDaemonTransport()
        with patch.dict(os.environ, {"JARVIS_MCP_POOL": "0"}):
            with patch.object(brave_mcp_client.requests, "Session",
                              return_value=daemon):
                c1, _, p1 = browser_agent._borrow_mcp_client(5)
                self.assertFalse(p1)
                browser_agent._return_mcp_client(c1, p1)
        self.assertEqual(daemon.closed, 1)
        self.assertIsNone(browser_agent._POOLED_CLIENT)

    def test_dead_session_dropped_never_repooled(self):
        daemon = FakeDaemonTransport()
        c1, _, p1 = self._borrow(daemon)
        c1._session_id = None  # reconnect failed mid-task
        browser_agent._return_mcp_client(c1, p1)
        self.assertIsNone(browser_agent._POOLED_CLIENT)
        self.assertEqual(daemon.closed, 1)

    def test_cache_dropped_when_session_reborn(self):
        daemon = FakeDaemonTransport()
        c1, _, p1 = self._borrow(daemon)
        browser_agent._note_pooled_tools(c1, [{"name": "navigate"}])
        browser_agent._return_mcp_client(c1, p1)
        c1.reconnects += 1  # daemon restarted/upgraded underneath us
        c2, cached2, p2 = self._borrow(daemon)
        try:
            self.assertIs(c2, c1)
            self.assertIsNone(cached2)
        finally:
            browser_agent._return_mcp_client(c2, p2)

    def test_note_ignores_foreign_clients(self):
        daemon = FakeDaemonTransport()
        c1, _, p1 = self._borrow(daemon)
        try:
            browser_agent._note_pooled_tools(Mock(), [{"name": "x"}])
            c2, cached2, _ = self._borrow(daemon)
            # c2 IS c1 (pool was never returned, so cold private path):
            # either way the foreign note must not surface.
            self.assertTrue(cached2 is None or cached2 != [{"name": "x"}])
            browser_agent._return_mcp_client(c2, False)
        finally:
            browser_agent._return_mcp_client(c1, p1)

    def test_prewarm_populates_pool_and_cache(self):
        daemon = FakeDaemonTransport()
        with patch.object(brave_mcp_client.requests, "Session",
                          return_value=daemon):
            browser_agent.prewarm_mcp_pool(timeout=5)
            posts_after_first = list(daemon.posts)
            browser_agent.prewarm_mcp_pool(timeout=5)  # second: no-op
        self.assertIn("initialize", posts_after_first)
        self.assertIn("tools/list", posts_after_first)
        self.assertEqual(list(daemon.posts), posts_after_first)
        self.assertIsNotNone(browser_agent._POOLED_CLIENT)
        self.assertIsNotNone(browser_agent._CACHED_TOOL_DEFS)

    def test_prewarm_never_raises_when_daemon_down(self):
        dead = Mock()
        dead.post.side_effect = requests.ConnectionError("refused")
        with patch.object(brave_mcp_client.requests, "Session",
                          return_value=dead):
            browser_agent.prewarm_mcp_pool(timeout=5)  # must not raise
        self.assertIsNone(browser_agent._POOLED_CLIENT)

    def test_prewarm_disabled_flag_is_silent(self):
        daemon = FakeDaemonTransport()
        with patch.dict(os.environ, {"JARVIS_MCP_PREWARM": "0"}):
            with patch.object(brave_mcp_client.requests, "Session",
                              return_value=daemon):
                browser_agent.prewarm_mcp_pool(timeout=5)
        self.assertEqual(daemon.posts, [])
        self.assertIsNone(browser_agent._POOLED_CLIENT)

    def test_two_tasks_share_one_handshake_and_one_list(self):
        daemon = FakeDaemonTransport()
        tool_call = {"id": "call_1", "name": "navigate",
                     "arguments": {"url": "https://example.com"}}
        turns = [[(None, [tool_call]), ("Done: found the price.", [])],
                 [(None, [tool_call]), ("Done: found the price.", [])]]
        results = []
        with patch.object(brave_mcp_client.requests, "Session",
                          return_value=daemon), \
             patch.object(browser_agent, "ensure_brave_mcp_daemon",
                          return_value=True), \
             patch.object(browser_agent, "_model_turn",
                          side_effect=[t for pair in turns for t in pair]), \
             patch.object(browser_agent, "append_activity_line"), \
             patch.object(browser_agent, "narrate_activity"), \
             patch.object(browser_agent, "truncate_activity_log"):
            for _ in range(2):
                results.append(browser_agent.run_browser_task("find price"))
        self.assertEqual([r.status for r in results],
                         ["completed", "completed"])
        self.assertEqual(daemon.count("initialize"), 1)
        self.assertEqual(daemon.count("tools/list"), 1)
        self.assertEqual(daemon.closed, 0)

    def test_pool_off_restores_per_task_handshake(self):
        daemon = FakeDaemonTransport()
        tool_call = {"id": "call_1", "name": "navigate",
                     "arguments": {"url": "https://example.com"}}
        with patch.dict(os.environ, {"JARVIS_MCP_POOL": "0"}), \
             patch.object(brave_mcp_client.requests, "Session",
                          return_value=daemon), \
             patch.object(browser_agent, "ensure_brave_mcp_daemon",
                          return_value=True), \
             patch.object(browser_agent, "_model_turn",
                          side_effect=[(None, [tool_call]),
                                       ("Done: found the price.", [])]), \
             patch.object(browser_agent, "append_activity_line"), \
             patch.object(browser_agent, "narrate_activity"), \
             patch.object(browser_agent, "truncate_activity_log"):
            result = browser_agent.run_browser_task("find price")
        self.assertEqual(result.status, "completed")
        self.assertEqual(daemon.count("initialize"), 1)
        self.assertEqual(daemon.closed, 1)


class PrewarmGateTests(unittest.TestCase):
    def test_due_when_no_contract_and_browser_engine(self):
        with patch.object(brain.config, "TASK_ENGINE", "browser_agent"):
            self.assertTrue(brain._browser_prewarm_due(None))

    def test_not_due_when_no_contract_and_other_engine(self):
        with patch.object(brain.config, "TASK_ENGINE", "opencode"):
            self.assertFalse(brain._browser_prewarm_due(None))

    def test_contract_executor_is_authoritative(self):
        browser_contract = Mock()
        browser_contract.executor = "browser_agent"
        other_contract = Mock()
        other_contract.executor = "opencode"
        with patch.object(brain.config, "TASK_ENGINE", "opencode"):
            self.assertTrue(brain._browser_prewarm_due(browser_contract))
        with patch.object(brain.config, "TASK_ENGINE", "browser_agent"):
            self.assertFalse(brain._browser_prewarm_due(other_contract))

    def test_handoff_spawns_prewarm_thread_for_browser(self):
        with patch.object(brain.config, "TASK_ENGINE", "browser_agent"), \
             patch.object(threading, "Thread") as thread_cls:
            brain.handle_opencode_task("open example", original_message="m")
        (call,) = thread_cls.call_args_list
        self.assertIs(call.kwargs["target"], browser_agent.prewarm_mcp_pool)
        self.assertTrue(call.kwargs["daemon"])
        thread_cls.return_value.start.assert_called_once_with()

    def test_handoff_spawns_nothing_for_other_engine(self):
        contract = Mock()
        contract.executor = "opencode"
        with patch.object(threading, "Thread") as thread_cls:
            brain.handle_opencode_task("open example", original_message="m",
                                       contract=contract)
        thread_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
