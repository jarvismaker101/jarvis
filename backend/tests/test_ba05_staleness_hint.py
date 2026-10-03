"""L-1 / BA-05: the DOM-mutation epoch is a hint, not a refusal.

A drifted mutation counter no longer refuses the click — the element is
re-queried by cssPath and the click proceeds when it still exists, is
visible and sits within tolerance (3x widened on drift). Document, URL,
DPR, frame, tab, exists and visible refusals are UNCHANGED: relaxing `mut`
must never relax identity.
"""
import inspect
import json
import os
import re
import unittest
from unittest.mock import patch

from backend.services import browser_agent


URL = "https://app.example/watch"


def _mark(**overrides):
    mark = {
        "cx": 40, "cy": 30, "tag": "button", "label": "Play", "inView": True,
        "cssPath": "button#play", "epoch_doc": "111", "epoch_mut": "7",
        "dpr": "1", "frame": "", "url": URL,
        "rect": {"x": 30, "y": 20, "w": 60, "h": 30},
        "tab_id": "", "capture_verified": True,
    }
    mark.update(overrides)
    return mark


def _live_probe(url=URL, doc=111, mut=7, dpr=1, exists=True, visible=True,
                rect=None, frame=""):
    return json.dumps({
        "checked": True, "epoch": {"doc": doc, "mut": mut}, "dpr": dpr,
        "url": url, "exists": exists, "visible": visible,
        "rect": rect if rect is not None else {"x": 30, "y": 20, "w": 60,
                                               "h": 30},
        "frame": frame,
    })


class FakeDaemon:
    def __init__(self, probe):
        self.probe = probe
        self.calls = []

    def call_tool(self, name, arguments=None):
        self.calls.append(name)
        if name == "evaluate":
            return self.probe
        if name == "click_locator":
            return "Clicked via real input (top).\nurl=%s navigated=no" % URL
        raise AssertionError("unexpected daemon call: %s" % name)


def _check(mark, probe):
    daemon = FakeDaemon(probe=probe)
    live, error = browser_agent._mark_target_state(daemon, mark)
    return live, error, daemon


class HintPolicyTests(unittest.TestCase):
    def setUp(self):
        self._env = dict(os.environ)
        os.environ.pop("JARVIS_BROWSER_STALE_ON_MUTATION", None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def test_drift_with_unmoved_element_proceeds(self):
        live, error, _daemon = _check(_mark(), _live_probe(mut=999))
        self.assertIsNone(error)
        self.assertTrue(live["exists"])

    def test_drift_within_widened_tolerance_proceeds(self):
        # 30px moved: over the base 12px tolerance, inside the drifted 36px.
        probe = _live_probe(mut=8, rect={"x": 60, "y": 20, "w": 60, "h": 30})
        _live, error, _daemon = _check(_mark(), probe)
        self.assertIsNone(error)

    def test_drift_beyond_widened_tolerance_still_refuses(self):
        probe = _live_probe(mut=8, rect={"x": 100, "y": 20, "w": 60, "h": 30})
        _live, error, _daemon = _check(_mark(), probe)
        self.assertIn("moved on the page", error)

    def test_no_drift_keeps_the_tight_tolerance(self):
        probe = _live_probe(rect={"x": 60, "y": 20, "w": 60, "h": 30})
        _live, error, _daemon = _check(_mark(), probe)
        self.assertIn("moved on the page", error)

    def test_small_drift_free_move_still_tolerated(self):
        probe = _live_probe(rect={"x": 35, "y": 24, "w": 60, "h": 30})
        _live, error, _daemon = _check(_mark(), probe)
        self.assertIsNone(error)

    def test_legacy_flag_restores_refusal(self):
        with patch.dict(os.environ, {"JARVIS_BROWSER_STALE_ON_MUTATION": "1"}):
            _live, error, _daemon = _check(_mark(), _live_probe(mut=8))
        self.assertIn("stale (the page content changed", error)

    def test_flag_helper_defaults_off(self):
        self.assertFalse(browser_agent._stale_on_mutation())
        with patch.dict(os.environ, {"JARVIS_BROWSER_STALE_ON_MUTATION": "1"}):
            self.assertTrue(browser_agent._stale_on_mutation())


class IdentityRefusalsIntactTests(unittest.TestCase):
    """Relaxing `mut` must not relax anything else — one case per guard."""

    def test_document_change_refuses(self):
        _live, error, _d = _check(_mark(), _live_probe(doc=222))
        self.assertIn("older page", error)

    def test_url_change_refuses(self):
        _live, error, _d = _check(_mark(), _live_probe(url=URL + "?x=1"))
        self.assertIn("different page", error)

    def test_dpr_change_refuses(self):
        _live, error, _d = _check(_mark(), _live_probe(dpr=2))
        self.assertIn("display scale", error)

    def test_frame_change_refuses(self):
        _live, error, _d = _check(_mark(frame="iframe#a"),
                                  _live_probe(frame="iframe#b"))
        self.assertIn("different frame", error)

    def test_gone_element_refuses(self):
        _live, error, _d = _check(_mark(), _live_probe(exists=False))
        self.assertIn("gone from the page", error)

    def test_invisible_element_refuses(self):
        _live, error, _d = _check(_mark(), _live_probe(visible=False))
        self.assertIn("no longer visible", error)

    def test_unstamped_mark_refuses(self):
        _live, error, _d = _check(_mark(epoch_doc="", epoch_mut=""),
                                  _live_probe())
        self.assertIn("no page-identity stamp", error)

    def test_identity_refusals_ignore_the_legacy_flag(self):
        # The flag only restores the MUT refusal; it must not weaken the
        # identity guards (here: a drifted counter PLUS a moved rect still
        # refuses even though the flag is off).
        probe = _live_probe(mut=8, rect={"x": 300, "y": 200, "w": 60,
                                         "h": 30})
        _live, error, _d = _check(_mark(), probe)
        self.assertIn("moved on the page", error)


class ObserverScopingTests(unittest.TestCase):
    def test_observer_watches_structure_only(self):
        src = inspect.getsource(browser_agent._handle_look)
        match = re.search(
            r"\.observe\(document\.documentElement,\s*\{([^}]*)\}\)", src)
        self.assertIsNotNone(match, "look must install the epoch observer")
        options = match.group(1)
        self.assertIn("childList", options)
        self.assertNotIn("subtree", options)
        self.assertNotIn("attributes", options)


if __name__ == "__main__":
    unittest.main()
