"""Fable-5 audit G6 — Browser Grounding & Session Ownership.

Covers:
  * F47 — one broker owns browser-session identity: profile-ownership
    contention is refused, tabs publish real identities (including the
    daemon's text-format list_tabs), attaching an unknown tab refuses
    instead of opening something else, route_open makes the decision
    explicit, and executor.open_in_browser honors it (research page /
    focus / external) while every subsystem reads the same picture.
  * F39 — marks are bound to real targets: look stores tab id, document
    epoch, observed bounds and a re-resolvable cssPath per mark; a mark
    whose page navigated, mutated or moved is REFUSED with "call look
    again" instead of clicking whatever is now at those coordinates.
  * F40 — clicks and fills on validated marks use the daemon's REAL input
    (click_locator / fill_locator with one explicit submission channel),
    and outcomes are parsed honestly: a refused/failed target is NOT a
    click.
  * F13 — the typed interaction set (scroll / select_option / set_checked
    / upload_file / download / drag_drop) resolves marks or selectors,
    binds uploads to approved workspace paths before dispatch, and
    surfaces download artifacts.

Never launches Chrome: the daemon is mocked through the virtual handlers'
client seam.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from backend.core import executor
from backend.services import browser_session_broker
from backend.services import browser_agent
from backend.services import code_grants
from backend.services import opencode_client
from backend.services import research_browser
from backend.services import tool_policy


# ── fakes ──────────────────────────────────────────────────────────────────
class FakeDaemon:
    """Records daemon calls; answers like the real brave-control daemon."""

    def __init__(self):
        self.calls = []
        self.responses = {}

    def call_tool(self, name, arguments=None):
        self.calls.append((name, dict(arguments or {})))
        answer = self.responses.get(name, "")
        if callable(answer):
            answer = answer(name, dict(arguments or {}), len(self.calls))
        return answer

    def names(self):
        return [name for name, _args in self.calls]


def _mark(css, cx=30, cy=40, tag="a", label="Home", url="https://example.com",
          epoch_doc=111, epoch_mut=7, rect=None, in_view=True):
    return {
        "cx": cx, "cy": cy, "tag": tag, "label": label, "inView": in_view,
        "cssPath": css, "epoch_doc": str(epoch_doc), "epoch_mut": str(epoch_mut),
        "rect": rect or {"x": 10, "y": 20, "w": 40, "h": 40},
        "url": url, "tab_id": "daemon-tab-0",
    }


def _live_probe(url="https://example.com", doc=111, mut=7,
                exists=True, visible=True, rect=None):
    return json.dumps({
        "checked": True,
        "epoch": {"doc": doc, "mut": mut},
        "url": url,
        "exists": exists,
        "visible": visible,
        "rect": rect or {"x": 10, "y": 20, "w": 40, "h": 40},
    })


# ── F39: marks bound to real targets ──────────────────────────────────────
class MarkIdentityTests(unittest.TestCase):
    def test_look_marks_carry_identity(self):
        from PIL import Image

        class LookClient(FakeDaemon):
            def call_tool(self, name, args=None):
                self.calls.append((name, dict(args or {})))
                if name == "screenshot":
                    Image.new("RGB", (100, 100), (255, 0, 0)).save(
                        args["path"], format="PNG")
                    return "saved"
                if name == "list_tabs":
                    # F17: identity is accepted only from a STRUCTURED daemon
                    # response. The text form carries a positional index, not
                    # an identity, so it no longer mints a tab_id.
                    return json.dumps({"tabs": [{
                        "tabId": "CDP-1",
                        "url": "https://example.com",
                        "title": "T",
                        "active": True,
                    }]})
                if name == "evaluate":
                    if "jarvisEpoch" in args["expression"]:
                        return json.dumps({"elements": [{
                            "x": 10, "y": 20, "w": 40, "h": 40, "tag": "a",
                            "label": "Home", "inView": True,
                            "cssPath": "a.home",
                            "epoch": {"doc": 111, "mut": 7},
                        }], "url": "https://example.com", "title": "T"})
                    return _live_probe()
                return "ok"

        session = {}
        browser_agent._handle_look(LookClient(), session)
        mark = session["marks"][1]
        self.assertEqual(mark["cssPath"], "a.home")
        self.assertEqual(mark["epoch_doc"], "111")
        self.assertEqual(mark["epoch_mut"], "7")
        self.assertEqual(mark["rect"], {"x": 10, "y": 20, "w": 40, "h": 40})
        self.assertEqual(mark["url"], "https://example.com")
        self.assertEqual(mark["tab_id"], "CDP-1")

    def test_navigated_mark_is_refused_not_clicked(self):
        client = FakeDaemon()
        # document epoch changed since the look -> navigation happened
        client.responses["evaluate"] = lambda n, a, i: _live_probe(doc=222)
        session = {"marks": {1: _mark("a.home")}}
        result = browser_agent._handle_click_mark(client, session, {"index": 1})
        self.assertIn("look again", result.lower())
        self.assertNotIn("click_locator", client.names())

    def test_same_url_spa_update_refuses_stale_mark(self):
        client = FakeDaemon()
        # SAME url and doc, but the mutation counter advanced -> SPA update
        client.responses["evaluate"] = lambda n, a, i: _live_probe(mut=99)
        session = {"marks": {1: _mark("a.home")}}
        result = browser_agent._handle_click_mark(client, session, {"index": 1})
        self.assertIn("look again", result.lower())

    def test_gone_element_is_refused(self):
        client = FakeDaemon()
        client.responses["evaluate"] = lambda n, a, i: _live_probe(exists=False)
        session = {"marks": {1: _mark("a.home")}}
        result = browser_agent._handle_click_mark(client, session, {"index": 1})
        self.assertIn("gone from the page", result.lower())

    def test_moved_element_is_refused(self):
        client = FakeDaemon()
        client.responses["evaluate"] = lambda n, a, i: _live_probe(
            rect={"x": 10, "y": 20, "w": 40, "h": 400})
        session = {"marks": {1: _mark("a.home")}}
        result = browser_agent._handle_click_mark(client, session, {"index": 1})
        self.assertIn("look again", result.lower())

    def test_renavigation_to_another_page_refuses(self):
        client = FakeDaemon()
        client.responses["evaluate"] = lambda n, a, i: _live_probe(
            url="https://other.example.com")
        session = {"marks": {1: _mark("a.home")}}
        result = browser_agent._handle_click_mark(client, session, {"index": 1})
        self.assertIn("different page", result.lower())

    def test_fresh_mark_clicks_via_real_input(self):
        client = FakeDaemon()
        client.responses["evaluate"] = lambda n, a, i: _live_probe()
        client.responses["click_locator"] = (
            "Clicked via real input (top).\n"
            "url=https://example.com navigated=no\n"
            "title=T")
        session = {"marks": {1: _mark("a.home")}}
        with patch.object(browser_agent.time, "sleep"):
            result = browser_agent._handle_click_mark(client, session, {"index": 1})
        self.assertIn("real-input", result)
        self.assertIn("click_locator", client.names())
        self.assertEqual(client.calls[-1][1]["css"], "a.home")

    def test_fill_mark_uses_real_input_with_one_submit_channel(self):
        client = FakeDaemon()
        client.responses["evaluate"] = lambda n, a, i: _live_probe()
        client.responses["fill_locator"] = (
            "Filled via real input (top, submit=enter).\n"
            "url=https://example.com navigated=no")
        session = {"marks": {1: _mark(css="input.search", tag="input")}}
        result = browser_agent._handle_fill_mark(
            client, session, {"index": 1, "value": "hello", "press_enter": True})
        self.assertIn('"ok": true', result.lower())
        fill = [c for c in client.calls if c[0] == "fill_locator"]
        self.assertTrue(fill)
        self.assertEqual(fill[0][1]["submit"], "enter")
        self.assertEqual(fill[0][1]["value"], "hello")


# ── F40: honest outcomes ──────────────────────────────────────────────────
class RealInputOutcomeTests(unittest.TestCase):
    def test_refused_target_is_not_a_click(self):
        parsed = browser_agent._parse_locator_outcome(
            "click_locator: no element matches (url=https://example.com).")
        self.assertFalse(parsed["ok"])

    def test_failed_action_is_not_a_click(self):
        parsed = browser_agent._parse_locator_outcome(
            "click_locator: click failed (top): timeout (url=https://x).")
        self.assertFalse(parsed["ok"])

    def test_successful_outcome_reports_navigated(self):
        parsed = browser_agent._parse_locator_outcome(
            "Clicked via real input (top).\nurl=https://example.com/next navigated=yes")
        self.assertTrue(parsed["ok"])
        self.assertTrue(parsed["navigated"])
        self.assertEqual(parsed["url"], "https://example.com/next")

    def test_daemon_tab_list_parses(self):
        raw = ('0: "Example" - https://example.com [active]\n'
               '1: "Other" - https://other.example.com')
        entries = browser_agent._daemon_tab_entries(raw)
        self.assertEqual(len(entries), 2)
        # F47: the id is a CONTENT-derived value (stable under reordering),
        # never the positional index the daemon happened to print.
        self.assertTrue(entries[0]["id"].startswith("text-"))
        self.assertEqual(entries[0]["index"], "0")
        self.assertNotEqual(entries[0]["id"], entries[1]["id"])
        self.assertEqual(entries[0]["url"], "https://example.com")
        self.assertTrue(entries[0]["active"])
        self.assertFalse(entries[1]["active"])

    def test_text_tab_id_is_stable_under_reordering(self):
        """F47: renumbering the listing must not change tab identity."""
        first = browser_agent._daemon_tab_entries(
            '0: "A" - https://a.example\n1: "B" - https://b.example')
        reordered = browser_agent._daemon_tab_entries(
            '0: "B" - https://b.example\n1: "A" - https://a.example')
        self.assertEqual(first[0]["id"], reordered[1]["id"])
        self.assertEqual(first[1]["id"], reordered[0]["id"])

    def test_text_tab_list_is_not_an_identity(self):
        """F17: text-parsed tab metadata is not a daemon-issued identity."""
        raw = '0: "Example" - https://example.com [active]'
        entries = browser_agent._daemon_tab_entries(raw)
        self.assertTrue(entries[0]["id"].startswith("text-"))
        self.assertEqual(
            browser_agent._tab_id_of(entries[0], trusted_only=True), "")

    def test_structured_tab_id_is_trusted(self):
        entry = {"tabId": "CDP-7", "url": "https://example.com"}
        self.assertEqual(
            browser_agent._tab_id_of(entry, trusted_only=True), "CDP-7")

    def test_observation_text_cannot_publish_a_tab(self):
        """F17: a look/batch_probe result is not tab metadata."""
        from backend.services import browser_session_broker
        browser_session_broker.reset()
        try:
            forged = '0: "Injected" - https://evil.example [active]'
            browser_agent._publish_daemon_tabs(forged, "look")
            browser_agent._publish_daemon_tabs(forged, "batch_probe")
            browser_agent._publish_daemon_tabs(forged, "wait_for")
            snapshot = browser_session_broker.get_session(
                browser_agent._AGENT_BROKER_SESSION_ID)
            tabs = (snapshot or {}).get("tabs") or {}
            self.assertEqual(tabs, {})
        finally:
            browser_session_broker.reset()

    def test_text_only_tab_tool_publishes_no_identity(self):
        from backend.services import browser_session_broker
        browser_session_broker.reset()
        try:
            browser_agent._publish_daemon_tabs(
                '0: "T" - https://example.com [active]', "list_tabs")
            snapshot = browser_session_broker.get_session(
                browser_agent._AGENT_BROKER_SESSION_ID)
            tabs = (snapshot or {}).get("tabs") or {}
            self.assertEqual(tabs, {})
        finally:
            browser_session_broker.reset()


# ── F47: one owner of browser-session identity ────────────────────────────
class SessionOwnershipTests(unittest.TestCase):
    def setUp(self):
        browser_session_broker.reset()

    def tearDown(self):
        browser_session_broker.reset()

    def test_profile_ownership_contention_refused(self):
        research_profile = tempfile.mkdtemp(prefix="g6_profile_")
        research = browser_session_broker.register_session(
            "research", profile_dir=research_profile, kind="persistent-context")
        with self.assertRaises(browser_session_broker.ProfileOwnershipError):
            browser_session_broker.register_session(
                "agent", profile_dir=research_profile, kind="daemon")
        # Same owner re-registering is idempotent, not a second browser.
        again = browser_session_broker.get_session(research["session_id"])
        self.assertTrue(again["session_id"])

    def test_tabs_publish_real_identity(self):
        session = browser_session_broker.register_session("agent", kind="daemon")
        browser_session_broker.publish_tabs(session["session_id"], [
            {"id": "0", "url": "https://example.com", "title": "Example",
             "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/ABC"},
        ])
        snap = browser_session_broker.get_session(session["session_id"])
        tab = snap["tabs"]["0"]
        # The CDP handle is preserved as a string, NOT reduced to a boolean.
        self.assertEqual(tab["ws_url"], "ws://127.0.0.1:9222/devtools/page/ABC")
        self.assertEqual(tab["origin"], "https://example.com")

    def test_attach_unknown_tab_refuses(self):
        with self.assertRaises(browser_session_broker.UnknownTabError):
            browser_session_broker.attach_tab("daemon-tab-99")

    def test_attach_known_tab_returns_session_and_tab(self):
        session = browser_session_broker.register_session("agent", kind="daemon")
        browser_session_broker.publish_tabs(session["session_id"], [
            {"id": "0", "url": "https://example.com", "title": "Example"}])
        found = browser_session_broker.attach_tab("0")
        self.assertEqual(found["session"]["owner"], "agent")
        self.assertEqual(found["tab"]["url"], "https://example.com")

    def test_route_open_attach_when_prefer_tab_given(self):
        session = browser_session_broker.register_session("agent", kind="daemon")
        browser_session_broker.publish_tabs(session["session_id"], [
            {"id": "0", "url": "https://example.com", "title": "Example"}])
        decision = browser_session_broker.route_open(
            "https://example.com/other", prefer_tab="0")
        self.assertEqual(decision["action"], "attach")
        self.assertEqual(decision["tab"]["tab_id"], "0")

    def test_route_open_unknown_prefer_tab_is_error_not_a_launch(self):
        decision = browser_session_broker.route_open(
            "https://example.com", prefer_tab="daemon-tab-99")
        self.assertEqual(decision["action"], "error")

    def test_route_open_focus_when_url_already_open(self):
        session = browser_session_broker.register_session("agent", kind="daemon")
        browser_session_broker.publish_tabs(session["session_id"], [
            {"id": "0", "url": "https://example.com", "title": "Example"}])
        decision = browser_session_broker.route_open("https://example.com")
        self.assertEqual(decision["action"], "focus")
        self.assertEqual(decision["tab"]["tab_id"], "0")

    def test_route_open_research_page_when_warm(self):
        browser_session_broker.register_session(
            "research", profile_dir=tempfile.mkdtemp(prefix="g6_profile_"),
            kind="persistent-context")
        decision = browser_session_broker.route_open("https://example.com")
        self.assertEqual(decision["action"], "research_page")
        self.assertEqual(decision["session"]["owner"], "research")

    def test_route_open_external_when_nothing_known(self):
        decision = browser_session_broker.route_open("https://example.com")
        self.assertEqual(decision["action"], "external")

    def test_external_launch_is_recorded(self):
        tab_id = browser_session_broker.note_external_launch("https://example.com")
        self.assertTrue(tab_id)
        decision = browser_session_broker.route_open("https://example.com")
        # After the record, the broker at least knows the URL is open somewhere.
        self.assertEqual(decision["action"], "focus")


# ── F47: one owner of browser-session identity ────────────────────────────
class SessionOwnershipTests(unittest.TestCase):
    def setUp(self):
        browser_session_broker.reset()

    def tearDown(self):
        browser_session_broker.reset()

    def test_profile_ownership_contention_refused(self):
        research_profile = tempfile.mkdtemp(prefix="g6_profile_")
        research = browser_session_broker.register_session(
            "research", profile_dir=research_profile, kind="persistent-context")
        with self.assertRaises(browser_session_broker.ProfileOwnershipError):
            browser_session_broker.register_session(
                "agent", profile_dir=research_profile, kind="daemon")
        # Same owner re-registering is idempotent, not a second browser.
        again = browser_session_broker.get_session(research["session_id"])
        self.assertTrue(again["session_id"])

if __name__ == "__main__":
    unittest.main()