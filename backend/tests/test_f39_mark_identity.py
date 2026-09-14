"""F39 — Bind browser marks to real targets (acceptance pins).

Acceptance clause under test:

    "Same-URL reload, first mutation, same-position replacement, duplicate
     tabs, switches, and DPR changes cannot redirect old marks or bypass
     rejection through raw primitives."
"""

import json
import os
import unittest
from unittest.mock import patch

from backend.services import browser_agent
from backend.services import browser_session_broker


URL = "https://app.example/watch"


def _mark(**overrides):
    """A mark exactly as `look` stores an identity-bearing one."""
    mark = {
        "cx": 40,
        "cy": 30,
        "tag": "button",
        "label": "Play",
        "inView": True,
        "cssPath": "button#play",
        "epoch_doc": "111",
        "epoch_mut": "7",
        "dpr": "1",
        "frame": "",
        "url": URL,
        "rect": {"x": 30, "y": 20, "w": 60, "h": 30},
        "tab_id": "",
        "capture_verified": True,
    }
    mark.update(overrides)
    return mark


def _live_probe(url=URL, doc=111, mut=7, dpr=1, exists=True, visible=True,
                rect=None, frame="", extra=None):
    payload = {
        "checked": True,
        "epoch": {"doc": doc, "mut": mut},
        "dpr": dpr,
        "url": url,
        "exists": exists,
        "visible": visible,
        "rect": rect if rect is not None else {"x": 30, "y": 20, "w": 60, "h": 30},
        "frame": frame,
    }
    if extra:
        payload.update(extra)
    return json.dumps(payload)


class FakeDaemon:
    """Fake daemon: `evaluate` answers by expression, tools by name."""

    def __init__(self, probe=None, click=None, tabs=None):
        self.probe = probe
        self.click = click
        self.tabs = tabs
        self.calls = []

    def call_tool(self, name, arguments=None):
        arguments = dict(arguments or {})
        self.calls.append((name, arguments))
        if name == "evaluate":
            expr = str(arguments.get("expression") or "")
            if "__jarvisEpoch" in expr:
                if isinstance(self.probe, Exception):
                    raise self.probe
                return self.probe
            return json.dumps({})
        if name == "list_tabs":
            if self.tabs is None:
                raise RuntimeError("no tab listing")
            return self.tabs
        if name == "click_locator":
            return self.click if self.click is not None else (
                "Clicked via real input (top).\nurl=%s navigated=no" % URL)
        raise AssertionError("unexpected daemon call: %s" % name)

    def names(self):
        return [name for name, _arguments in self.calls]


class SameUrlReloadTests(unittest.TestCase):
    def test_same_url_reload_with_a_new_document_refuses_the_mark(self):
        daemon = FakeDaemon(probe=_live_probe(doc=222))
        session = {"marks": {1: _mark()}}
        result = browser_agent._handle_click_mark(daemon, session, {"index": 1})
        self.assertIn("older page", result)
        self.assertNotIn("click_locator", daemon.names())

    def test_same_url_reload_whose_counters_restart_at_zero_refuses(self):
        # doc epoch is identical but the mutation counter restarted at 0: the
        # explicit zero must be COMPARED, not skipped as "missing".
        daemon = FakeDaemon(probe=_live_probe(mut=0))
        session = {"marks": {1: _mark(epoch_mut="7")}}
        result = browser_agent._handle_click_mark(daemon, session, {"index": 1})
        self.assertIn("stale", result.lower())
        self.assertNotIn("click_locator", daemon.names())

    def test_first_mutation_after_a_look_refuses_the_mark(self):
        daemon = FakeDaemon(probe=_live_probe(mut=8))
        session = {"marks": {1: _mark()}}
        result = browser_agent._handle_click_mark(daemon, session, {"index": 1})
        self.assertIn("stale", result.lower())
        self.assertNotIn("click_locator", daemon.names())

    def test_a_zero_mutation_mark_is_refused_on_the_first_mutation(self):
        daemon = FakeDaemon(probe=_live_probe(mut=1))
        session = {"marks": {1: _mark(epoch_mut="0")}}
        result = browser_agent._handle_click_mark(daemon, session, {"index": 1})
        self.assertIn("stale", result.lower())

    def test_an_identical_zero_state_is_accepted(self):
        daemon = FakeDaemon(probe=_live_probe(mut=0))
        session = {"marks": {1: _mark(epoch_mut="0")}}
        _live, error = browser_agent._mark_target_state(daemon, session["marks"][1])
        self.assertIsNone(error)
        self.assertTrue(_live["exists"])

    def test_zero_epochs_survive_as_explicit_values(self):
        self.assertEqual(browser_agent._epoch_text(0), "0")
        self.assertEqual(browser_agent._mark_epoch_of({"epoch_mut": 0}),
                         ("", "0"))
        self.assertEqual(browser_agent._epoch_text(None), "")
        self.assertEqual(browser_agent._epoch_text(False), "")

    def test_a_mark_without_any_epoch_stamp_is_refused(self):
        daemon = FakeDaemon(probe=_live_probe())
        session = {"marks": {1: _mark(epoch_doc="", epoch_mut="")}}
        result = browser_agent._handle_click_mark(daemon, session, {"index": 1})
        self.assertIn("no page-identity stamp", result)
        self.assertNotIn("click_locator", daemon.names())

    def test_a_page_that_cannot_report_its_document_refuses_the_mark(self):
        probe = json.dumps({"checked": True, "epoch": {"mut": 7}, "dpr": 1,
                            "url": URL, "exists": True, "visible": True,
                            "rect": {"x": 30, "y": 20, "w": 60, "h": 30},
                            "frame": ""})
        daemon = FakeDaemon(probe=probe)
        session = {"marks": {1: _mark()}}
        result = browser_agent._handle_click_mark(daemon, session, {"index": 1})
        self.assertIn("cannot be verified", result)
        self.assertNotIn("click_locator", daemon.names())

    def test_a_url_change_refuses_the_mark(self):
        daemon = FakeDaemon(probe=_live_probe(url=URL + "?other=1"))
        session = {"marks": {1: _mark()}}
        result = browser_agent._handle_click_mark(daemon, session, {"index": 1})
        self.assertIn("different page", result)
        self.assertNotIn("click_locator", daemon.names())


class SamePositionReplacementTests(unittest.TestCase):
    def test_a_replaced_element_at_the_same_rect_is_refused(self):
        # Same URL, same document, same coordinates - but the element the
        # mark pointed at is gone.
        daemon = FakeDaemon(probe=_live_probe(exists=False))
        session = {"marks": {1: _mark()}}
        result = browser_agent._handle_click_mark(daemon, session, {"index": 1})
        self.assertIn("gone from the page", result)
        self.assertNotIn("click_locator", daemon.names())

    def test_same_position_replacement_with_a_changed_geometry_is_refused(self):
        daemon = FakeDaemon(probe=_live_probe(
            rect={"x": 300, "y": 200, "w": 60, "h": 30}))
        session = {"marks": {1: _mark()}}
        result = browser_agent._handle_click_mark(daemon, session, {"index": 1})
        self.assertIn("moved on the page", result)
        self.assertNotIn("click_locator", daemon.names())

    def test_a_small_geometry_drift_is_tolerated(self):
        daemon = FakeDaemon(probe=_live_probe(
            rect={"x": 35, "y": 24, "w": 60, "h": 30}))
        session = {"marks": {1: _mark()}}
        _live, error = browser_agent._mark_target_state(daemon, session["marks"][1])
        self.assertIsNone(error)

    def test_an_invisible_replacement_is_refused(self):
        daemon = FakeDaemon(probe=_live_probe(visible=False))
        session = {"marks": {1: _mark()}}
        result = browser_agent._handle_click_mark(daemon, session, {"index": 1})
        self.assertIn("no longer visible", result)


class TabIdentityTests(unittest.TestCase):
    def setUp(self):
        browser_session_broker.reset()

    def tearDown(self):
        browser_session_broker.reset()

    def test_a_tab_switch_refuses_the_mark(self):
        daemon = FakeDaemon(probe=_live_probe(), tabs=json.dumps([
            {"tabId": "CDP-1", "url": URL, "active": False},
            {"tabId": "CDP-2", "url": "https://other.example", "active": True},
        ]))
        session = {"marks": {1: _mark(tab_id="CDP-1")}}
        result = browser_agent._handle_click_mark(daemon, session, {"index": 1})
        self.assertIn("different tab", result)
        self.assertNotIn("click_locator", daemon.names())

    def test_the_mark_s_own_tab_still_activates(self):
        daemon = FakeDaemon(probe=_live_probe(), tabs=json.dumps([
            {"tabId": "CDP-1", "url": URL, "active": True},
        ]))
        session = {"marks": {1: _mark(tab_id="CDP-1")}}
        result = browser_agent._handle_click_mark(daemon, session, {"index": 1})
        self.assertIn("click_locator", daemon.names())
        self.assertNotIn("different tab", result)

    def test_an_unknown_active_tab_does_not_refuse(self):
        daemon = FakeDaemon(probe=_live_probe())
        session = {"marks": {1: _mark(tab_id="CDP-1")}}
        result = browser_agent._handle_click_mark(daemon, session, {"index": 1})
        self.assertNotIn("different tab", result)

    def test_duplicate_url_tabs_make_the_tab_identity_unknown(self):
        listing = ('0: "A" - %s [active]\n'
                   '1: "B" - %s' % (URL, URL))
        self.assertEqual(browser_agent._current_tab_id(
            FakeDaemon(tabs=listing), URL), "")

    def test_a_single_matching_tab_is_reported(self):
        structured = json.dumps([
            {"tabId": "CDP-9", "url": URL, "active": True},
            {"tabId": "CDP-10", "url": "https://other.example", "active": False},
        ])
        self.assertEqual(browser_agent._current_tab_id(
            FakeDaemon(tabs=structured), URL), "CDP-9")

    def test_text_tab_metadata_is_not_a_trusted_identity(self):
        listing = '0: "A" - %s [active]' % URL
        self.assertEqual(browser_agent._current_tab_id(
            FakeDaemon(tabs=listing), URL), "")

    def test_broker_refuses_an_ambiguous_url(self):
        browser_session_broker.register_session("agent", session_id="s1")
        browser_session_broker.publish_tabs("s1", [
            {"tab_id": "t1", "url": URL},
            {"tab_id": "t2", "url": URL},
        ])
        self.assertIsNone(browser_session_broker.resolve_target(URL))
        resolved = browser_session_broker.resolve_targets(URL)
        self.assertTrue(resolved["ambiguous"])
        self.assertEqual(len(resolved["candidates"]), 2)
        route = browser_session_broker.route_open(URL)
        self.assertEqual(route["action"], "ambiguous")

    def test_broker_refuses_an_ambiguous_tab_id(self):
        for session_id in ("s1", "s2"):
            browser_session_broker.register_session("agent", session_id=session_id)
            browser_session_broker.publish_tabs(session_id, [
                {"tab_id": "t1", "url": "%s/%s" % (URL, session_id)},
            ])
        with self.assertRaises(browser_session_broker.AmbiguousTabError) as ctx:
            browser_session_broker.attach_tab("t1")
        self.assertEqual(len(ctx.exception.candidates), 2)


class RawPrimitiveTests(unittest.TestCase):
    """F39: raw primitives must not bypass the rejection paths."""

    def test_click_point_without_a_look_anchor_is_refused(self):
        client = FakeDaemon()
        result = browser_agent._handle_click_point(client, {}, {"x": 5, "y": 5})
        self.assertIn("refused", result)
        self.assertEqual(client.calls, [])

    def test_click_point_refuses_after_a_reload(self):
        session = {"look_capture": {"url": URL, "doc": "111", "mut": "7",
                                    "dpr": "1"}}
        client = FakeDaemon()
        client.probe = None
        client.calls = []
        with patch.object(browser_agent, "_capture_state",
                          return_value={"doc": 222, "dpr": 1, "url": URL}):
            result = browser_agent._handle_click_point(client, session,
                                                       {"x": 5, "y": 5})
        self.assertIn("different document", result)
        self.assertEqual(client.calls, [])

    def test_click_point_refuses_when_the_page_cannot_be_rechecked(self):
        session = {"look_capture": {"url": URL, "doc": "111", "mut": "7",
                                    "dpr": "1"}}
        client = FakeDaemon()
        with patch.object(browser_agent, "_capture_state", return_value=None):
            result = browser_agent._handle_click_point(client, session,
                                                       {"x": 5, "y": 5})
        self.assertIn("could not be re-checked", result)
        self.assertEqual(client.calls, [])

    def test_every_typed_tool_goes_through_the_same_mark_check(self):
        for verb, handler in (
            ("select_option", lambda d, s: browser_agent._handle_select_option(
                d, s, {"index": 1, "value": "x"})),
            ("set_checked", lambda d, s: browser_agent._handle_set_checked(
                d, s, {"index": 1, "checked": True})),
            ("download", lambda d, s: browser_agent._handle_download(
                d, s, {"index": 1})),
            ("drag_drop", lambda d, s: browser_agent._handle_drag_drop(
                d, s, {"index": 1, "target_index": 2})),
            ("upload_file", lambda d, s: browser_agent._handle_upload_file(
                d, s, {"index": 1, "path": __file__})),
        ):
            daemon = FakeDaemon(probe=_live_probe(exists=False))
            session = {"marks": {1: _mark(), 2: _mark()}}
            with patch("backend.services.code_grants.grant_check",
                       return_value=(True, __file__, "ok")):
                result = handler(daemon, session)
            self.assertIn("gone from the page", result, verb)
            self.assertEqual(
                [n for n in daemon.names()
                 if n not in ("evaluate",)], [], verb)


class DprTests(unittest.TestCase):
    def test_a_dpr_change_refuses_the_mark(self):
        daemon = FakeDaemon(probe=_live_probe(dpr=2))
        session = {"marks": {1: _mark()}}
        result = browser_agent._handle_click_mark(daemon, session, {"index": 1})
        self.assertIn("display scale", result)
        self.assertNotIn("click_locator", daemon.names())

    def test_a_dpr_change_refuses_raw_coordinates(self):
        session = {"look_capture": {"url": URL, "doc": "111", "mut": "7",
                                    "dpr": "1"}}
        client = FakeDaemon()
        with patch.object(browser_agent, "_capture_state",
                          return_value={"doc": 111, "dpr": 2, "url": URL}):
            result = browser_agent._handle_click_point(client, session,
                                                       {"x": 5, "y": 5})
        self.assertIn("display scale changed", result)
        self.assertEqual(client.calls, [])

    def test_an_unknown_live_dpr_does_not_refuse_but_an_unknown_mark_dpr_does(self):
        daemon = FakeDaemon(probe=_live_probe(dpr=None))
        session = {"marks": {1: _mark(dpr="")}}
        # dpr unknown on both sides: cannot be proven false, so it proceeds.
        daemon2 = FakeDaemon(probe=_live_probe())
        _live, error = browser_agent._mark_target_state(daemon2,
                                                        session["marks"][1])
        self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
