"""F13 — Expose complete browser interactions (acceptance pins).

Acceptance clause under test:

    "Malformed calls do nothing; readable files cannot upload to unapproved
     sites; downloads require real artifacts; state changes are read back.
     Daemon-side implementation remains unverified."

The last sentence is a LIMIT, not a claim: these tests pin the agent-side
contract (schemas, refusal, read-back, artifact checks) against a fake
daemon. Whether the daemon's own primitives really upload/download/select is
NOT asserted here and is NOT verified by this module.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from backend.services import browser_agent


class FakeDaemon:
    """Fake MCP client that answers by tool name and by evaluate expression."""

    def __init__(self, url="https://app.example/page", state=None, probe=None,
                 evaluate=None, **tools):
        self.url = url
        self.state = state
        self.probe = probe
        self.evaluate = evaluate
        self.tools = tools
        self.calls = []

    def call_tool(self, name, arguments=None):
        arguments = dict(arguments or {})
        self.calls.append((name, arguments))
        if name == "evaluate":
            expr = str(arguments.get("expression") or "")
            if expr == "location.href":
                return json.dumps(self.url)
            if "selectedOptions" in expr:
                return json.dumps(self.state if self.state is not None else {})
            if "__jarvisEpoch" in expr:
                return json.dumps(self.probe if self.probe is not None else {})
            if "devicePixelRatio" in expr:
                return json.dumps({"doc": 111, "dpr": 1, "url": self.url})
            return json.dumps(self.evaluate if self.evaluate is not None else {})
        if name not in self.tools:
            raise AssertionError("unexpected daemon call: %s %r" % (name, arguments))
        value = self.tools[name]
        return value if isinstance(value, str) else json.dumps(value)

    def names(self):
        return [name for name, _arguments in self.calls]


class NeverCalledDaemon:
    """A daemon that fails the test the moment anything is dispatched."""

    def call_tool(self, name, arguments=None):
        raise AssertionError(
            "nothing may be dispatched, got %s %r" % (name, arguments))


class MalformedCallTests(unittest.TestCase):
    """F13: "Malformed calls do nothing"."""

    def test_unparseable_arguments_dispatch_nothing(self):
        history = []
        blocked = browser_agent._run_one_tool(
            NeverCalledDaemon(), history,
            {"id": "c1", "name": "scroll", "arguments": {},
             "parse_error": "Expecting value: line 1 column 1 (char 0)"},
            session={})
        self.assertIsNone(blocked)
        self.assertEqual(len(history), 1)
        content = history[-1]["content"]
        self.assertIn("not valid JSON", content)
        self.assertIn("NOTHING was executed", content)
        self.assertEqual(history[-1]["tool_call_id"], "c1")

    def test_unparseable_arguments_of_a_virtual_tool_dispatch_nothing(self):
        history = []
        browser_agent._run_one_tool(
            NeverCalledDaemon(), history,
            {"id": "c2", "name": "select_option", "arguments": {},
             "parse_error": "Unterminated string"},
            session={})
        self.assertIn("not valid JSON", history[-1]["content"])

    def test_arguments_that_are_not_an_object_dispatch_nothing(self):
        history = []
        browser_agent._run_one_tool(
            NeverCalledDaemon(), history,
            {"id": "c3", "name": "click_text", "arguments": "not-an-object"},
            session={})
        self.assertEqual(len(history), 1)
        self.assertIn("must be a JSON object", history[-1]["content"])

    def test_valid_arguments_are_still_dispatched(self):
        daemon = FakeDaemon(scroll="scrolled down (url=https://app.example/x)")
        history = []
        browser_agent._run_one_tool(
            daemon, history,
            {"id": "c4", "name": "scroll", "arguments": {"direction": "down"}},
            session={})
        self.assertEqual(daemon.names(), ["scroll"])


class SchemaBindingTests(unittest.TestCase):
    """F13: schemas/validation are generated TOGETHER."""

    def test_advertised_schema_is_the_validated_object(self):
        spec = browser_agent._VIRTUAL_TOOL_SPECS_BY_NAME["select_option"]
        self.assertIs(browser_agent._virtual_schema("select_option"),
                      spec["input_schema"])
        self.assertIsNone(browser_agent._virtual_schema("no-such-tool"))

    def test_every_advertised_tool_has_a_schema_and_a_handler(self):
        advertised = {tool["name"] for tool in browser_agent._VIRTUAL_TOOL_DEFS}
        self.assertEqual(advertised, browser_agent._VIRTUAL_TOOL_NAMES)
        for name in sorted(advertised):
            self.assertIsInstance(browser_agent._virtual_schema(name), dict)

    def test_missing_required_argument_is_refused_without_dispatching(self):
        history = []
        browser_agent._run_one_tool(
            NeverCalledDaemon(), history,
            {"id": "c1", "name": "select_option", "arguments": {"css": "#s"}},
            session={})
        self.assertEqual(len(history), 1)
        self.assertIn("tool blocked by policy", history[-1]["content"])
        self.assertIn("value", history[-1]["content"])

    def test_cross_field_rule_is_enforced_at_execution(self):
        history = []
        browser_agent._run_one_tool(
            NeverCalledDaemon(), history,
            {"id": "c2", "name": "wait_for", "arguments": {}}, session={})
        self.assertIn("wait_for requires one of selector or text",
                      history[-1]["content"])

    def test_download_requires_a_target_and_drag_drop_requires_two(self):
        self.assertIn("download requires one of index or css",
                      browser_agent._virtual_argument_error("download", {}))
        self.assertEqual(
            browser_agent._virtual_argument_error("download", {"css": "#d"}), "")
        self.assertIn("drag_drop requires a drop target",
                      browser_agent._virtual_argument_error(
                          "drag_drop", {"index": 1}))
        self.assertEqual(
            browser_agent._virtual_argument_error(
                "drag_drop", {"index": 1, "target_index": 2}), "")


class UploadOriginTests(unittest.TestCase):
    """F13: "readable files cannot upload to unapproved sites"."""

    def setUp(self):
        handle, self.path = tempfile.mkstemp(prefix="jarvis_f13_", suffix=".txt")
        os.close(handle)

    def tearDown(self):
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def _grant(self, allowed=True):
        return patch("backend.services.code_grants.grant_check",
                     return_value=(allowed, self.path, "ok"))

    def test_unapproved_origin_refuses_before_any_dispatch(self):
        daemon = FakeDaemon(url="https://evil.example/collect",
                            upload_file="uploaded (url=https://evil.example)")
        with self._grant():
            result = browser_agent._handle_upload_file(
                daemon, {}, {"path": self.path, "css": "#file"})
        self.assertIn("refused", result)
        self.assertIn("evil.example", result)
        self.assertNotIn("upload_file", daemon.names())
        # only the origin probe ran
        self.assertEqual(daemon.names(), ["evaluate"])

    def test_unknown_destination_origin_refuses(self):
        daemon = FakeDaemon(url="", upload_file="uploaded")
        with self._grant():
            result = browser_agent._handle_upload_file(
                daemon, {}, {"path": self.path, "css": "#file"})
        self.assertIn("could not be established", result)
        self.assertNotIn("upload_file", daemon.names())

    def test_approved_origin_uploads(self):
        daemon = FakeDaemon(
            url="https://app.example/upload",
            upload_file="Uploaded via real input (top).\nurl=https://app.example/upload navigated=no")
        session = {"upload_origins": ["https://app.example"]}
        with self._grant():
            result = browser_agent._handle_upload_file(
                daemon, session, {"path": self.path, "css": "#file"})
        self.assertIn("uploaded", result.lower())
        self.assertIn("upload_file", daemon.names())

    def test_a_grant_can_approve_the_origin(self):
        daemon = FakeDaemon(
            url="https://good.example/form",
            upload_file="Uploaded via real input (top).\nurl=https://good.example/form navigated=no")
        session = {"grants": ["upload_origin:https://good.example"]}
        with self._grant():
            result = browser_agent._handle_upload_file(
                daemon, session, {"path": self.path, "css": "#file"})
        self.assertIn("uploaded", result.lower())

    def test_approved_origin_does_not_approve_another_site(self):
        daemon = FakeDaemon(url="https://other.example/form", upload_file="uploaded")
        session = {"upload_origins": ["https://good.example"]}
        with self._grant():
            result = browser_agent._handle_upload_file(
                daemon, session, {"path": self.path, "css": "#file"})
        self.assertIn("refused", result)
        self.assertNotIn("upload_file", daemon.names())

    def test_unapproved_path_is_refused_even_for_an_approved_origin(self):
        daemon = FakeDaemon(url="https://app.example/upload", upload_file="uploaded")
        with self._grant(allowed=False):
            result = browser_agent._handle_upload_file(
                daemon, {"upload_origins": ["https://app.example"]},
                {"path": self.path, "css": "#file"})
        self.assertIn("refused", result)
        self.assertEqual(daemon.calls, [])


class DownloadArtifactTests(unittest.TestCase):
    """F13: "downloads require real artifacts"."""

    def setUp(self):
        handle, self.path = tempfile.mkstemp(prefix="jarvis_f13_dl_", suffix=".zip")
        os.close(handle)

    def tearDown(self):
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def test_completion_without_an_artifact_is_a_failure(self):
        daemon = FakeDaemon(
            download="Downloaded via real input (top).\nurl=https://app.example/file")
        result = browser_agent._handle_download(daemon, {}, {"css": "#dl"})
        self.assertIn("download failed", result)
        self.assertIn("named no artifact", result)

    def test_missing_on_disk_artifact_is_a_failure(self):
        missing = os.path.join(tempfile.gettempdir(), "jarvis_f13_absent_artifact.zip")
        daemon = FakeDaemon(
            download="Downloaded via real input (top).\nartifact=%s\nurl=https://app.example/f" % missing)
        result = browser_agent._handle_download(daemon, {}, {"css": "#dl"})
        self.assertIn("download failed", result)
        self.assertIn("does not exist on disk", result)

    def test_real_artifact_is_reported_with_its_path(self):
        daemon = FakeDaemon(
            download="Downloaded via real input (top).\nsaved=%s\nurl=https://app.example/f" % self.path)
        result = browser_agent._handle_download(daemon, {}, {"css": "#dl"})
        self.assertIn("download complete", result)
        self.assertIn(self.path, result)

    def test_a_daemon_error_is_not_a_download(self):
        daemon = FakeDaemon(
            download="no download event was observed (url=https://app.example/f)")
        result = browser_agent._handle_download(daemon, {}, {"css": "#dl"})
        self.assertIn("download failed", result)
        self.assertNotIn("download complete", result)


class ReadBackTests(unittest.TestCase):
    """F13: "state changes are read back"."""

    def test_select_option_reports_the_state_actually_in_effect(self):
        daemon = FakeDaemon(
            state={"found": True, "value": "b", "label": "Bravo",
                   "url": "https://app.example/page"},
            select_option="Selected via real input (top).\nurl=https://app.example/page navigated=no")
        result = browser_agent._handle_select_option(
            daemon, {}, {"css": "#pick", "value": "Bravo"})
        body = json.loads(result)
        self.assertTrue(body["verified"])
        self.assertEqual(body["value"], "b")
        self.assertEqual(daemon.names(), ["select_option", "evaluate"])
        args = daemon.calls[0][1]
        self.assertEqual(args["value"], "Bravo")
        self.assertEqual(args["css"], "#pick")

    def test_select_option_that_did_not_take_effect_is_a_failure(self):
        daemon = FakeDaemon(
            state={"found": True, "value": "a", "label": "Alpha",
                   "url": "https://app.example/page"},
            select_option="Selected via real input (top).\nurl=https://app.example/page navigated=no")
        result = browser_agent._handle_select_option(
            daemon, {}, {"css": "#pick", "value": "Bravo"})
        self.assertIn("select_option failed", result)
        self.assertIn("did not take effect", result)

    def test_select_option_without_a_readable_state_is_a_failure(self):
        daemon = FakeDaemon(
            state={"found": False},
            select_option="Selected via real input (top).\nurl=https://app.example/page navigated=no")
        result = browser_agent._handle_select_option(
            daemon, {}, {"css": "#pick", "value": "Bravo"})
        self.assertIn("could not be read back", result)

    def test_set_checked_reports_the_state_actually_in_effect(self):
        daemon = FakeDaemon(
            state={"found": True, "checked": True, "url": "https://app.example/page"},
            set_checked="Checked via real input (top).\nurl=https://app.example/page navigated=no")
        body = json.loads(browser_agent._handle_set_checked(
            daemon, {}, {"css": "#agree", "checked": True}))
        self.assertTrue(body["verified"])
        self.assertTrue(body["checked"])

    def test_set_checked_that_did_not_take_effect_is_a_failure(self):
        daemon = FakeDaemon(
            state={"found": True, "checked": False, "url": "https://app.example/page"},
            set_checked="Checked via real input (top).\nurl=https://app.example/page navigated=no")
        result = browser_agent._handle_set_checked(
            daemon, {}, {"css": "#agree", "checked": True})
        self.assertIn("set_checked failed", result)
        self.assertIn("did not take effect", result)

    def test_set_checked_without_a_readable_state_is_a_failure(self):
        daemon = FakeDaemon(
            state={"found": True, "value": "x"},
            set_checked="Checked via real input (top).\nurl=https://app.example/page navigated=no")
        result = browser_agent._handle_set_checked(
            daemon, {}, {"css": "#agree", "checked": True})
        self.assertIn("could not be read back", result)


class FramePreservationTests(unittest.TestCase):
    """F13: "preserve frames" — a target inside a frame stays in its frame."""

    def _mark(self, frame):
        return {"cssPath": "button#go", "frame": frame, "epoch_doc": "111",
                "epoch_mut": "7", "dpr": "1", "url": "https://app.example/page",
                "rect": {"x": 10, "y": 10, "w": 40, "h": 20}, "inView": True}

    def test_resolve_returns_the_frame_with_the_selector(self):
        daemon = FakeDaemon(probe={
            "checked": True, "exists": True, "visible": True,
            "epoch": {"doc": 111, "mut": 7}, "dpr": 1,
            "url": "https://app.example/page",
            "rect": {"x": 10, "y": 10, "w": 40, "h": 20},
            "frame": "iframe#embed"})
        session = {"marks": {1: self._mark("iframe#embed")}}
        css, frame, error = browser_agent._resolve_mark_or_css(
            daemon, session, "select_option", 1, None)
        self.assertEqual(error, "")
        self.assertEqual(css, "button#go")
        self.assertEqual(frame, "iframe#embed")

    def test_dispatch_carries_the_frame_to_the_daemon(self):
        daemon = FakeDaemon(
            probe={
                "checked": True, "exists": True, "visible": True,
                "epoch": {"doc": 111, "mut": 7}, "dpr": 1,
                "url": "https://app.example/page",
                "rect": {"x": 10, "y": 10, "w": 40, "h": 20},
                "frame": "iframe#embed"},
            state={"found": True, "checked": True, "url": "https://app.example/page"},
            set_checked="Checked via real input (iframe#embed).\nurl=https://app.example/page navigated=no")
        session = {"marks": {1: self._mark("iframe#embed")}}
        browser_agent._handle_set_checked(daemon, session, {"index": 1, "checked": True})
        set_args = [args for name, args in daemon.calls if name == "set_checked"]
        self.assertEqual(len(set_args), 1)
        self.assertEqual(set_args[0]["frame"], "iframe#embed")
        self.assertEqual(set_args[0]["css"], "button#go")

    def test_a_frame_change_is_refused(self):
        daemon = FakeDaemon(probe={
            "checked": True, "exists": True, "visible": True,
            "epoch": {"doc": 111, "mut": 7}, "dpr": 1,
            "url": "https://app.example/page",
            "rect": {"x": 10, "y": 10, "w": 40, "h": 20},
            "frame": ""})
        session = {"marks": {1: self._mark("iframe#embed")}}
        css, frame, error = browser_agent._resolve_mark_or_css(
            daemon, session, "set_checked", 1, None)
        self.assertIsNone(css)
        self.assertIn("different frame", error)


if __name__ == "__main__":
    unittest.main()
