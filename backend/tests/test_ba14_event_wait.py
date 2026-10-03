"""L-5 / BA-14 — event-driven wait_for (requires daemon).

Verified 2026-10-03: the virtual wait_for polled evaluate from Python every
200 ms (15 round trips for a 3 s wait, each re-shipping the same JS, up to
200 ms of avoidable latency on every success) because the daemon's
synchronous stringify made a Promise waiter impossible. The daemon now owns
a real event-driven `wait_for` tool (waitForSelector/waitForFunction
equivalent, one round trip) plus opt-in `awaitPromise` on evaluate; the
agent prefers it whenever the task's daemon tool list advertises it and
keeps the polling loop as the capability-gated fallback for old daemons.
"""
import json
import unittest
from unittest.mock import patch

from backend.services import browser_agent


class FakeDaemon:
    """Answers evaluate (polling) and/or the event-driven wait_for tool."""

    def __init__(self, polls=None, event=None, event_error=None):
        self.polls = list(polls or [])
        self.event = event
        self.event_error = event_error
        self.calls = []

    def call_tool(self, name, arguments=None):
        arguments = dict(arguments or {})
        self.calls.append((name, arguments))
        if name == "evaluate":
            if not self.polls:
                raise AssertionError("polling path taken with no polls left")
            answer = self.polls.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        if name == "wait_for":
            if isinstance(self.event_error, Exception):
                raise self.event_error
            if self.event is not None:
                return self.event() if callable(self.event) else self.event
            raise AssertionError("event path taken with no event answer")
        raise AssertionError("unexpected daemon call: %s" % name)

    def names(self):
        return [name for name, _arguments in self.calls]


def _session_with_wait():
    return {"daemon_tools": {"navigate", "evaluate", "wait_for"}}


def _session_without_wait():
    return {"daemon_tools": {"navigate", "evaluate"}}


class EventPathTests(unittest.TestCase):
    def test_one_round_trip_for_the_whole_wait(self):
        daemon = FakeDaemon(
            event=json.dumps({"found": True, "elapsed_ms": 1234,
                              "url": "https://x/after", "title": "t2"}))
        with patch.object(browser_agent.time, "sleep",
                          side_effect=AssertionError("must not poll")):
            result = browser_agent._handle_wait_for(
                daemon, {"selector": "#foo", "timeout_ms": 5000},
                session=_session_with_wait())
        payload = json.loads(result)
        self.assertTrue(payload["found"])
        self.assertEqual(payload["elapsed_ms"], 1234)
        self.assertEqual(payload["url"], "https://x/after")
        self.assertEqual(daemon.names(), ["wait_for"])

    def test_event_wait_forwards_selector_text_and_timeout(self):
        daemon = FakeDaemon(
            event=json.dumps({"found": False, "elapsed_ms": 5000,
                              "url": "https://x", "title": "t"}))
        browser_agent._handle_wait_for(
            daemon, {"selector": "#foo", "text": "hello", "timeout_ms": 9000},
            session=_session_with_wait())
        _name, args = daemon.calls[0]
        self.assertEqual(args["selector"], "#foo")
        self.assertEqual(args["text"], "hello")
        self.assertEqual(args["timeout_ms"], 9000)

    def test_event_timeout_reports_not_found(self):
        daemon = FakeDaemon(
            event=json.dumps({"found": False, "elapsed_ms": 5000,
                              "url": "https://x", "title": "t"}))
        result = browser_agent._handle_wait_for(
            daemon, {"selector": "#foo"}, session=_session_with_wait())
        payload = json.loads(result)
        self.assertFalse(payload["found"])
        self.assertEqual(daemon.names(), ["wait_for"])

    def test_daemon_garbage_fails_loudly_without_polling(self):
        daemon = FakeDaemon(event="not-json{{{")
        result = browser_agent._handle_wait_for(
            daemon, {"selector": "#foo"}, session=_session_with_wait())
        self.assertIn("unparsable", result)
        self.assertEqual(daemon.names(), ["wait_for"])

    def test_daemon_error_fails_without_polling(self):
        daemon = FakeDaemon(event_error=RuntimeError("CDP blew up"))
        result = browser_agent._handle_wait_for(
            daemon, {"selector": "#foo"}, session=_session_with_wait())
        self.assertIn("daemon waiter error", result)
        self.assertEqual(daemon.names(), ["wait_for"])

    def test_empty_wait_still_refused_before_any_call(self):
        daemon = FakeDaemon(event=json.dumps({"found": True}))
        result = browser_agent._handle_wait_for(daemon, {},
                                               session=_session_with_wait())
        self.assertIn("verifies nothing", result)
        self.assertEqual(daemon.names(), [])


class FallbackPathTests(unittest.TestCase):
    def test_old_daemon_keeps_polling(self):
        daemon = FakeDaemon(polls=[
            json.dumps({"found": False, "url": "https://x", "title": "t"}),
            json.dumps({"found": True, "url": "https://x/after",
                        "title": "t2"}),
        ])
        with patch.object(browser_agent.time, "sleep") as sleep_mock, \
             patch.object(browser_agent.time, "monotonic",
                          side_effect=[0.0, 0.0, 0.1, 0.1, 0.4, 0.4]):
            result = browser_agent._handle_wait_for(
                daemon, {"selector": "#foo", "timeout_ms": 5000},
                session=_session_without_wait())
        payload = json.loads(result)
        self.assertTrue(payload["found"])
        self.assertEqual(daemon.names(), ["evaluate", "evaluate"])
        sleep_mock.assert_called_once_with(0.2)

    def test_missing_capability_means_polling(self):
        # Handler unit-test sessions (plain {}) predate the capability and
        # must behave exactly as before: poll, never touch wait_for.
        daemon = FakeDaemon(polls=[
            json.dumps({"found": True, "url": "https://x", "title": "t"}),
        ])
        with patch.object(browser_agent.time, "sleep"), \
             patch.object(browser_agent.time, "monotonic",
                          side_effect=[0.0, 0.0, 0.0, 0.0]):
            result = browser_agent._handle_wait_for(
                daemon, {"selector": "#foo"}, session={})
        self.assertTrue(json.loads(result)["found"])
        self.assertEqual(daemon.names(), ["evaluate"])

    def test_unknown_tool_error_falls_back_to_polling(self):
        daemon = FakeDaemon(
            polls=[json.dumps({"found": True, "url": "https://x",
                               "title": "t"})],
            event_error=RuntimeError("MCP error -32602: Unknown tool: wait_for"))
        with patch.object(browser_agent.time, "sleep"), \
             patch.object(browser_agent.time, "monotonic",
                          side_effect=[0.0, 0.0, 0.0, 0.0]):
            result = browser_agent._handle_wait_for(
                daemon, {"selector": "#foo"}, session=_session_with_wait())
        self.assertTrue(json.loads(result)["found"])
        self.assertEqual(daemon.names(), ["wait_for", "evaluate"])


class PlumbingTests(unittest.TestCase):
    def test_loop_stashes_daemon_tool_names_in_session(self):
        seen = {}

        def spy(client, name, arguments, session=None, stats=None):
            seen["daemon_tools"] = set((session or {}).get("daemon_tools")
                                       or set())
            return (json.dumps({"found": True, "url": "https://x",
                                "title": "t"}), None)

        tool_call = {"id": "call_1", "name": "wait_for",
                     "arguments": {"selector": "#foo"}}
        with patch.object(browser_agent, "_run_virtual_tool",
                          side_effect=spy) as virtual, \
             patch.object(browser_agent, "_model_turn",
                          side_effect=[(None, [tool_call]),
                                       ("Done.", [])]), \
             patch.object(browser_agent, "append_activity_line"), \
             patch.object(browser_agent, "narrate_activity"), \
             patch.object(browser_agent, "truncate_activity_log"):
            from unittest.mock import Mock
            result = browser_agent._agent_loop_inner(
                Mock(), "task", __import__("time").monotonic(),
                browser_agent._SwStats(),
                cached_tools=[{"name": "navigate"},
                              {"name": "evaluate"},
                              {"name": "wait_for"}])
        self.assertEqual(str(result.status), "completed")
        self.assertIn("wait_for", seen["daemon_tools"])
        virtual.assert_called_once()
        positional, _kwargs = virtual.call_args
        self.assertIn("wait_for", positional[3].get("daemon_tools"))

    def test_loop_without_wait_for_leaves_it_out(self):
        seen = {}

        def spy(client, name, arguments, session=None, stats=None):
            seen["daemon_tools"] = set((session or {}).get("daemon_tools")
                                       or set())
            return (json.dumps({"found": True, "url": "https://x",
                                "title": "t"}), None)

        tool_call = {"id": "call_1", "name": "wait_for",
                     "arguments": {"selector": "#foo"}}
        with patch.object(browser_agent, "_run_virtual_tool",
                          side_effect=spy), \
             patch.object(browser_agent, "_model_turn",
                          side_effect=[(None, [tool_call]),
                                       ("Done.", [])]), \
             patch.object(browser_agent, "append_activity_line"), \
             patch.object(browser_agent, "narrate_activity"), \
             patch.object(browser_agent, "truncate_activity_log"):
            from unittest.mock import Mock
            import time as _time
            browser_agent._agent_loop_inner(
                Mock(), "task", _time.monotonic(), browser_agent._SwStats(),
                cached_tools=[{"name": "navigate"}, {"name": "evaluate"}])
        self.assertNotIn("wait_for", seen["daemon_tools"])


if __name__ == "__main__":
    unittest.main()
