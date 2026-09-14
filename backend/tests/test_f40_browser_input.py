"""F40 — Use real browser input and frame control (acceptance pins).

Acceptance clause under test:

    "Enter-handled forms submit once; disabled/throwing clicks fail; real
     permitted iframe input works; error text containing URL/navigation fields
     cannot become success. Daemon primitives remain unverified."

The last sentence is a LIMIT: these tests pin the agent's dispatch contract
(one submit channel, refusals, frame forwarding, literal outcome parsing)
against a fake daemon. The daemon's own primitive implementations are not
verified here.
"""

import json
import unittest
from unittest.mock import Mock, patch

from backend.services import browser_agent


class FakeDaemon:
    """Fake daemon; records every (name, arguments) pair."""

    def __init__(self, evaluate=None, **tools):
        self.evaluate = evaluate
        self.tools = tools
        self.calls = []

    def call_tool(self, name, arguments=None):
        arguments = dict(arguments or {})
        self.calls.append((name, arguments))
        if name == "evaluate":
            value = self.evaluate
            if callable(value):
                return value(arguments)
            return value if value is not None else json.dumps({})
        if name not in self.tools:
            raise AssertionError("unexpected daemon call: %s %r" % (name, arguments))
        value = self.tools[name]
        return value if isinstance(value, str) else json.dumps(value)

    def names(self):
        return [name for name, _arguments in self.calls]

    def args_for(self, name):
        return [arguments for call_name, arguments in self.calls if call_name == name]


class SingleSubmitChannelTests(unittest.TestCase):
    """F40: "Enter-handled forms submit once"."""

    def test_the_core_fill_javascript_has_exactly_one_submit_channel(self):
        core = browser_agent._FILL_JS_CORE
        self.assertIn("KeyboardEvent('keydown'", core)
        self.assertIn("KeyboardEvent('keyup'", core)
        self.assertEqual(core.count("KeyboardEvent('keydown'"), 1)
        # form.requestSubmit() alongside the Enter dispatch submitted
        # Enter-handled forms twice.
        self.assertNotIn("requestSubmit", core)
        self.assertNotIn(".form.submit()", core)

    def test_fill_asks_the_daemon_for_one_submit(self):
        daemon = FakeDaemon(evaluate=json.dumps(
            {"found": True, "tag": "input", "cssPath": "#q", "frame": "",
             "disabled": False}),
            fill_locator=("Filled via real input (top, submit=enter).\n"
                          "url=https://x navigated=no"))
        browser_agent._handle_fill(
            daemon, {"selector": "#q", "value": "silo", "press_enter": True})
        args = daemon.args_for("fill_locator")
        self.assertEqual(len(args), 1)
        self.assertEqual(args[0]["submit"], "enter")
        # nothing else may submit: the resolve probe must not submit either
        expr = daemon.args_for("evaluate")[0]["expression"]
        self.assertNotIn("requestSubmit", expr)
        self.assertNotIn("KeyboardEvent", expr)

    def test_fill_without_press_enter_never_submits(self):
        daemon = FakeDaemon(evaluate=json.dumps(
            {"found": True, "tag": "input", "cssPath": "#q", "frame": "",
             "disabled": False}),
            fill_locator="Filled via real input (top, submit=none).")
        browser_agent._handle_fill(daemon, {"selector": "#q", "value": "silo"})
        self.assertEqual(daemon.args_for("fill_locator")[0]["submit"], "none")

    def test_the_legacy_fill_mark_path_submits_once_too(self):
        session = {"marks": {2: {"cx": 10, "cy": 20, "tag": "input",
                                 "label": "Search", "inView": True}}}
        daemon = FakeDaemon(evaluate=json.dumps({"ok": True}))
        browser_agent._handle_fill_mark(
            daemon, session, {"index": 2, "value": "silo", "press_enter": True})
        expr = daemon.args_for("evaluate")[0]["expression"]
        self.assertNotIn("requestSubmit", expr)
        self.assertEqual(expr.count("KeyboardEvent('keydown'"), 1)


class DisabledAndThrowingClickTests(unittest.TestCase):
    """F40: "disabled/throwing clicks fail"."""

    def _resolved(self, **overrides):
        payload = {"found": True, "tag": "button", "label": "Buy",
                   "cssPath": "#buy", "frame": "", "disabled": False,
                   "nonInteractive": False}
        payload.update(overrides)
        return json.dumps(payload)

    def test_a_disabled_target_is_refused_before_any_click(self):
        daemon = FakeDaemon(evaluate=self._resolved(disabled=True),
                            click_locator="Clicked via real input (top).")
        result = browser_agent._handle_click_text(daemon, {"text": "Buy"})
        self.assertIn("disabled", result)
        self.assertEqual(daemon.names(), ["evaluate"])

    def test_a_disabled_target_at_a_point_is_refused(self):
        session = {"look_capture": {"url": "https://x", "doc": "1", "dpr": "1"}}
        daemon = FakeDaemon(evaluate=self._resolved(disabled=True),
                            click_locator="Clicked via real input (top).")
        with patch.object(browser_agent, "_capture_state",
                          return_value={"doc": 1, "dpr": 1, "url": "https://x"}):
            result = browser_agent._handle_click_point(
                daemon, session, {"x": 5, "y": 5})
        self.assertIn("disabled", result)
        self.assertEqual(daemon.names(), ["evaluate"])

    def test_a_throwing_click_is_a_failure(self):
        parsed = browser_agent._parse_locator_outcome(
            "click_locator: click threw: TypeError: el.click is not a function "
            "(url=https://x navigated=no)")
        self.assertFalse(parsed["ok"])

    def test_a_daemon_exception_is_a_failure_not_a_click(self):
        daemon = FakeDaemon(evaluate=self._resolved())

        def boom(name, arguments=None):
            raise RuntimeError("mouse input unavailable")

        daemon.call_tool = boom
        result = browser_agent._handle_click_text(daemon, {"text": "Buy"})
        self.assertIn("failed", result)

    def test_the_legacy_coordinate_click_reports_a_throwing_click(self):
        session = {"marks": {3: {"cx": 10, "cy": 20, "tag": "button",
                                 "label": "Buy", "inView": True}}}
        daemon = FakeDaemon(evaluate=json.dumps({"ok": True}))
        browser_agent._handle_click_mark(daemon, session, {"index": 3})
        expr = daemon.args_for("evaluate")[0]["expression"]
        # A click that throws is reported as NOT clicked, and a disabled
        # element is never clicked at all.
        self.assertIn("click threw", expr)
        self.assertIn("disabled", expr)
        self.assertIn("clicked: false", expr)


class FrameInputTests(unittest.TestCase):
    """F40: "real permitted iframe input works"."""

    def test_click_inside_a_same_origin_frame_uses_the_locator_with_its_frame(self):
        daemon = FakeDaemon(
            evaluate=json.dumps({"found": True, "tag": "button",
                                 "label": "Accept", "cssPath": "button#accept",
                                 "frame": "iframe#consent", "disabled": False}),
            click_locator=("Clicked via real input (iframe#consent).\n"
                           "url=https://x navigated=no"))
        result = browser_agent._handle_click_text(daemon, {"text": "Accept"})
        args = daemon.args_for("click_locator")[0]
        self.assertEqual(args["css"], "button#accept")
        self.assertEqual(args["frame"], "iframe#consent")
        body = json.loads(result)
        self.assertTrue(body["clicked"])
        self.assertEqual(body["via"], "real-input")

    def test_fill_inside_a_same_origin_frame_forwards_the_frame(self):
        daemon = FakeDaemon(
            evaluate=json.dumps({"found": True, "tag": "input",
                                 "cssPath": "input#card", "frame": "iframe#pay",
                                 "disabled": False}),
            fill_locator=("Filled via real input (iframe#pay, submit=none).\n"
                          "url=https://x navigated=no"))
        result = browser_agent._handle_fill(
            daemon, {"selector": "#card", "value": "4111"})
        args = daemon.args_for("fill_locator")[0]
        self.assertEqual(args["frame"], "iframe#pay")
        self.assertTrue(json.loads(result)["ok"])

    def test_the_resolve_probe_reports_the_owning_frame(self):
        daemon = FakeDaemon(evaluate=json.dumps(
            {"found": True, "tag": "button", "label": "Go", "cssPath": "#go",
             "frame": "", "disabled": False}))
        browser_agent._handle_click_text(daemon, {"text": "Go"})
        expr = daemon.args_for("evaluate")[0]["expression"]
        self.assertIn("jarvisFrame", expr)
        self.assertIn("ownerDocument", expr)

    def test_a_top_document_target_carries_no_frame_argument(self):
        daemon = FakeDaemon(
            evaluate=json.dumps({"found": True, "tag": "button", "label": "Go",
                                 "cssPath": "#go", "frame": "",
                                 "disabled": False}),
            click_locator="Clicked via real input (top).\nurl=https://x navigated=no")
        browser_agent._handle_click_text(daemon, {"text": "Go"})
        self.assertNotIn("frame", daemon.args_for("click_locator")[0])


class OutcomeParsingTests(unittest.TestCase):
    """F40: "error text containing URL/navigation fields cannot become
    success" and literal preservation."""

    def test_error_text_with_url_and_navigation_fields_is_not_success(self):
        for text in (
            "error: navigate failed url=https://x navigated=yes",
            "could not click the element (url=https://x navigated=true)",
            "no element matches #go url=https://x navigated=yes",
            "click_locator: timed out url=https://x navigated=yes",
            "refused: the element is not clickable url=https://x navigated=1",
            "the download is missing url=https://x navigated=yes",
        ):
            parsed = browser_agent._parse_locator_outcome(text)
            self.assertFalse(parsed["ok"], text)

    def test_an_explicit_positive_marker_is_success(self):
        for text in (
            "Clicked via real input (top).\nurl=https://x navigated=yes",
            "Filled via real input (top, submit=enter).\nurl=https://x navigated=no",
            "ok: selected",
            "selection succeeded (url=https://x)",
        ):
            parsed = browser_agent._parse_locator_outcome(text)
            self.assertTrue(parsed["ok"], text)

    def test_a_bare_url_field_is_not_success(self):
        parsed = browser_agent._parse_locator_outcome("url=https://x navigated=yes")
        self.assertFalse(parsed["ok"])

    def test_the_url_keeps_its_literal_case(self):
        parsed = browser_agent._parse_locator_outcome(
            "Clicked via real input (top).\n"
            "url=https://Example.COM/CaseSensitive?Token=AbC navigated=yes")
        self.assertEqual(parsed["url"],
                         "https://Example.COM/CaseSensitive?Token=AbC")
        self.assertTrue(parsed["navigated"])

    def test_failure_wins_over_a_success_marker(self):
        parsed = browser_agent._parse_locator_outcome(
            "clicked=false: the element was not found (url=https://x)")
        self.assertFalse(parsed["ok"])

    def test_a_failed_fill_is_reported_as_a_failure(self):
        daemon = FakeDaemon(
            evaluate=json.dumps({"found": True, "tag": "input",
                                 "cssPath": "#q", "frame": "",
                                 "disabled": False}),
            fill_locator=("fill_locator: failed to type (url=https://x "
                          "navigated=yes)"))
        result = browser_agent._handle_fill(
            daemon, {"selector": "#q", "value": "silo", "press_enter": True})
        self.assertIn("fill failed", result)


if __name__ == "__main__":
    unittest.main()
