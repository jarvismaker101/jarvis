"""L-16 / BA-15 — scroll-into-view-and-click in one call.

Verified 2026-10-03: the look inventory deliberately advertises off-screen
marks as numbered clickable targets, then click_mark refused them outright
— refuse -> scroll -> look -> click, four turns for a button already in the
table. The daemon's click_locator already scrolls into view inside the same
real-input call (scrollIntoViewIfNeeded; fill_locator gained parity with
this change), so an off-screen mark WITH an addressable identity now flows
straight through, reporting scrolled=true on success. Only identity-less
(coordinate) marks keep the refusal: no selector means nothing to scroll to.
"""
import json
import unittest

from backend.services import browser_agent


URL = "https://app.example/fold"


def _mark(**overrides):
    mark = {
        "cx": 40,
        "cy": 30,
        "tag": "button",
        "label": "Below",
        "inView": False,
        "cssPath": "button#below",
        "epoch_doc": "111",
        "epoch_mut": "7",
        "dpr": "1",
        "frame": "",
        "url": URL,
        "rect": {"x": 30, "y": 2000, "w": 60, "h": 30},
        "tab_id": "",
        "capture_verified": True,
    }
    mark.update(overrides)
    return mark


def _live_probe(url=URL, doc=111, mut=7, dpr=1, exists=True, visible=True):
    return json.dumps({
        "checked": True,
        "epoch": {"doc": doc, "mut": mut},
        "dpr": dpr,
        "url": url,
        "exists": exists,
        "visible": visible,
        "rect": {"x": 30, "y": 2000, "w": 60, "h": 30},
        "frame": "",
    })


_CLICK_OK = ("Clicked via real input (top).\nurl=%s navigated=no" % URL)
_FILL_OK = ("Filled via real input (top, submit=none).\nurl=%s navigated=no"
            % URL)


class FakeDaemon:
    def __init__(self, probe=None, click=_CLICK_OK, fill=_FILL_OK):
        self.probe = probe
        self.click = click
        self.fill = fill
        self.calls = []

    def call_tool(self, name, arguments=None):
        arguments = dict(arguments or {})
        self.calls.append((name, arguments))
        if name == "evaluate":
            return self.probe
        if name == "click_locator":
            return self.click
        if name == "fill_locator":
            return self.fill
        raise AssertionError("unexpected daemon call: %s" % name)

    def names(self):
        return [name for name, _arguments in self.calls]


class OffscreenClickTests(unittest.TestCase):
    def test_offscreen_mark_with_identity_clicks_and_reports_scrolled(self):
        daemon = FakeDaemon(probe=_live_probe())
        session = {"marks": {1: _mark()}}
        result = browser_agent._handle_click_mark(daemon, session, {"index": 1})
        payload = json.loads(result)
        self.assertTrue(payload["clicked"])
        self.assertTrue(payload.get("scrolled"))
        self.assertIn("click_locator", daemon.names())

    def test_offscreen_mark_without_identity_still_refuses(self):
        daemon = FakeDaemon(probe=_live_probe())
        mark = _mark()
        del mark["cssPath"]
        session = {"marks": {1: mark}}
        result = browser_agent._handle_click_mark(daemon, session, {"index": 1})
        self.assertIn("off-screen", result.lower())
        self.assertIn("scroll it into view", result.lower())
        self.assertEqual(daemon.names(), [])

    def test_inview_mark_reports_no_scroll(self):
        daemon = FakeDaemon(probe=_live_probe())
        session = {"marks": {1: _mark(inView=True)}}
        result = browser_agent._handle_click_mark(daemon, session, {"index": 1})
        payload = json.loads(result)
        self.assertTrue(payload["clicked"])
        self.assertNotIn("scrolled", payload)

    def test_failed_click_reports_no_scroll(self):
        daemon = FakeDaemon(
            probe=_live_probe(),
            click="click_locator: no element matches (url=%s)." % URL)
        session = {"marks": {1: _mark()}}
        result = browser_agent._handle_click_mark(daemon, session, {"index": 1})
        payload = json.loads(result)
        self.assertFalse(payload["clicked"])
        self.assertNotIn("scrolled", payload)

    def test_stale_offscreen_mark_still_refuses(self):
        daemon = FakeDaemon(probe=_live_probe(doc=222))
        session = {"marks": {1: _mark()}}
        result = browser_agent._handle_click_mark(daemon, session, {"index": 1})
        self.assertIn("look again", result.lower())
        self.assertNotIn("click_locator", daemon.names())
        self.assertNotIn("scrolled", result)


class OffscreenFillTests(unittest.TestCase):
    def test_offscreen_fill_reports_scrolled(self):
        daemon = FakeDaemon(probe=_live_probe())
        session = {"marks": {1: _mark(tag="input")}}
        result = browser_agent._handle_fill_mark(
            daemon, session, {"index": 1, "value": "hello"})
        payload = json.loads(result)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload.get("scrolled"))

    def test_inview_fill_reports_no_scroll(self):
        daemon = FakeDaemon(probe=_live_probe())
        session = {"marks": {1: _mark(tag="input", inView=True)}}
        result = browser_agent._handle_fill_mark(
            daemon, session, {"index": 1, "value": "hello"})
        payload = json.loads(result)
        self.assertTrue(payload["ok"])
        self.assertNotIn("scrolled", payload)

    def test_failed_fill_reports_no_scroll(self):
        daemon = FakeDaemon(
            probe=_live_probe(),
            fill="fill_locator: fill failed (top): timeout (url=%s)." % URL)
        session = {"marks": {1: _mark(tag="input")}}
        result = browser_agent._handle_fill_mark(
            daemon, session, {"index": 1, "value": "hello"})
        payload = json.loads(result)
        self.assertFalse(payload["ok"])
        self.assertNotIn("scrolled", payload)


if __name__ == "__main__":
    unittest.main()
