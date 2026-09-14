"""F47 — Unify browser session identity (acceptance pins).

Acceptance clause under test:

    "All tabs remain addressable; reordering preserves identity; ambiguity
     prompts resolution; dead sessions authorize nothing; explicit attachment
     never silently launches elsewhere."
"""

import json
import os
import unittest
from unittest.mock import patch

from backend.services import browser_agent
from backend.services import browser_session_broker
from backend.services.task_agent.connectors import browser_cdp


URL_A = "https://app.example/alpha"
URL_B = "https://app.example/beta"


class BrokerIdentityTests(unittest.TestCase):
    def setUp(self):
        browser_session_broker.reset()

    def tearDown(self):
        browser_session_broker.reset()

    def test_tab_ids_are_instance_qualified(self):
        first = browser_session_broker.register_session("agent", session_id="s1")
        second = browser_session_broker.register_session("agent", session_id="s2")
        one = browser_session_broker.qualified_tab_id(first["instance"], "0")
        two = browser_session_broker.qualified_tab_id(second["instance"], "0")
        self.assertNotEqual(one, two)
        self.assertTrue(one.endswith(":0"))
        self.assertIn(":", one)

    def test_all_published_tabs_stay_addressable(self):
        browser_session_broker.register_session("agent", session_id="s1")
        tabs = [{"tab_id": "t%d" % index,
                 "url": "https://app.example/p%d" % index,
                 "title": "P%d" % index} for index in range(12)]
        browser_session_broker.publish_tabs("s1", tabs)
        snapshot = browser_session_broker.get_session("s1")
        self.assertEqual(len(snapshot["tabs"]), 12)
        # every single one resolves, including the ones past a UI limit
        for index in range(12):
            target = browser_session_broker.resolve_target("t%d" % index)
            self.assertIsNotNone(target, index)
            self.assertEqual(target["tab"]["url"],
                             "https://app.example/p%d" % index)

    def test_reordering_preserves_identity(self):
        browser_session_broker.register_session("agent", session_id="s1")
        original = [{"tab_id": "t1", "url": URL_A},
                    {"tab_id": "t2", "url": URL_B}]
        browser_session_broker.publish_tabs("s1", original)
        before = {tab_id: browser_session_broker.resolve_target(tab_id)["tab"]
                  ["qualified_id"] for tab_id in ("t1", "t2")}
        browser_session_broker.publish_tabs("s1", list(reversed(original)))
        after = {tab_id: browser_session_broker.resolve_target(tab_id)["tab"]
                 ["qualified_id"] for tab_id in ("t1", "t2")}
        self.assertEqual(before, after)
        self.assertEqual(
            browser_session_broker.resolve_target("t1")["tab"]["url"], URL_A)

    def test_an_ambiguous_url_prompts_resolution_instead_of_guessing(self):
        browser_session_broker.register_session("agent", session_id="s1")
        browser_session_broker.publish_tabs("s1", [
            {"tab_id": "t1", "url": URL_A, "title": "Alpha one"},
            {"tab_id": "t2", "url": URL_A, "title": "Alpha two"},
        ])
        route = browser_session_broker.route_open(URL_A)
        self.assertEqual(route["action"], "ambiguous")
        self.assertEqual(len(route["candidates"]), 2)
        self.assertEqual(sorted(c["tab_id"] for c in route["candidates"]),
                         ["t1", "t2"])

    def test_dead_sessions_authorize_nothing(self):
        session = browser_session_broker.register_session(
            "agent", session_id="s1", pid=os.getpid() + 100000)
        browser_session_broker.publish_tabs("s1", [
            {"tab_id": "t1", "url": URL_A}])
        self.assertFalse(browser_session_broker.is_live("s1"))
        self.assertFalse(browser_session_broker.get_session("s1")["live"])
        self.assertIsNone(browser_session_broker.resolve_target("t1"))
        with self.assertRaises(browser_session_broker.UnknownTabError):
            browser_session_broker.attach_tab("t1")
        route = browser_session_broker.route_open(URL_A)
        self.assertNotEqual(route["action"], "focus")
        self.assertEqual(session["session_id"], "s1")

    def test_a_retired_session_authorizes_nothing(self):
        browser_session_broker.register_session("agent", session_id="s1")
        browser_session_broker.publish_tabs("s1", [
            {"tab_id": "t1", "url": URL_A}])
        self.assertIsNotNone(browser_session_broker.resolve_target("t1"))
        browser_session_broker.mark_dead("s1")
        self.assertFalse(browser_session_broker.is_live("s1"))
        self.assertIsNone(browser_session_broker.resolve_target("t1"))
        with self.assertRaises(browser_session_broker.UnknownTabError):
            browser_session_broker.attach_tab("t1")

    def test_a_live_profile_claim_is_exclusive_and_a_dead_one_is_reclaimed(self):
        profile = os.path.join(os.path.dirname(__file__), "f47-profile")
        browser_session_broker.register_session(
            "agent", profile_dir=profile, kind="daemon", session_id="holder")
        with self.assertRaises(browser_session_broker.ProfileOwnershipError):
            browser_session_broker.register_session(
                "other", profile_dir=profile, kind="daemon", session_id="live")
        browser_session_broker.mark_dead("holder")
        session = browser_session_broker.register_session(
            "other", profile_dir=profile, kind="daemon", session_id="live")
        self.assertEqual(session["session_id"], "live")

    def test_a_claim_held_by_a_foreign_process_is_not_ownership(self):
        profile = os.path.join(os.path.dirname(__file__), "f47-foreign")
        browser_session_broker.register_session(
            "agent", profile_dir=profile, kind="daemon", session_id="foreign",
            pid=os.getpid() + 100000)
        # A record whose owning process is not this one proves nothing, so the
        # profile is free for a live session of another owner.
        session = browser_session_broker.register_session(
            "other", profile_dir=profile, kind="daemon", session_id="live2")
        self.assertEqual(session["session_id"], "live2")

    def test_the_ws_handle_is_never_a_stringified_boolean(self):
        browser_session_broker.register_session("agent", session_id="s1")
        browser_session_broker.publish_tabs("s1", [
            {"tab_id": "t1", "url": URL_A, "ws_url": True},
        ])
        tab = browser_session_broker.get_session("s1")["tabs"]["t1"]
        self.assertEqual(tab["ws_url"], "")
        real = "ws://127.0.0.1:9222/devtools/page/ABC"
        browser_session_broker.note_tab("s1", "t2", url=URL_B, ws_url=real)
        note = browser_session_broker.get_session("s1")["tabs"]["t2"]
        self.assertEqual(note["ws_url"], real)
        self.assertEqual(browser_session_broker.canonical_ws_url(True), "")
        self.assertEqual(browser_session_broker.canonical_ws_url("True"), "")
        self.assertEqual(browser_session_broker.canonical_ws_url(real), real)

    def test_origins_are_canonical(self):
        browser_session_broker.register_session("agent", session_id="s1")
        browser_session_broker.publish_tabs("s1", [
            {"tab_id": "t1", "url": "https://App.Example/Path?x=1"},
        ])
        tab = browser_session_broker.get_session("s1")["tabs"]["t1"]
        self.assertEqual(tab["origin"], "https://app.example")
        self.assertEqual(browser_session_broker.origin_of("not a url"), "")


class ExplicitAttachmentTests(unittest.TestCase):
    """F47: "explicit attachment never silently launches elsewhere"."""

    def setUp(self):
        browser_session_broker.reset()
        self.launched = []

    def tearDown(self):
        browser_session_broker.reset()

    def _launch(self, url, browser=None):
        self.launched.append(url)

    def test_an_unknown_explicit_tab_is_refused_not_launched(self):
        with patch.object(browser_cdp, "open_in_browser", self._launch):
            result = browser_cdp.open_url(URL_A, prefer_tab="tab-that-never-was")
        self.assertEqual(self.launched, [])
        self.assertIn("not opening a different browser", result)

    def test_an_unreachable_explicit_tab_is_refused_not_launched(self):
        browser_session_broker.register_session("agent", session_id="s1")
        browser_session_broker.publish_tabs("s1", [
            {"tab_id": "t1", "url": URL_A, "title": "Alpha"}])
        with patch.object(browser_cdp, "activate_tab",
                          return_value={"activated": False, "reason": "no"}):
            with patch.object(browser_cdp, "open_in_browser", self._launch):
                result = browser_cdp.open_url(URL_B, prefer_tab="t1")
        self.assertEqual(self.launched, [])
        self.assertIn("not opening a different browser", result)

    def test_an_ambiguous_explicit_target_is_refused_not_launched(self):
        for session_id in ("s1", "s2"):
            browser_session_broker.register_session("agent", session_id=session_id)
            browser_session_broker.publish_tabs(session_id, [
                {"tab_id": "dup", "url": "%s/%s" % (URL_A, session_id),
                 "title": "Dup %s" % session_id}])
        with patch.object(browser_cdp, "open_in_browser", self._launch):
            result = browser_cdp.open_url(URL_B, prefer_tab="dup")
        self.assertEqual(self.launched, [])
        self.assertIn("not guessing", result)

    def test_an_explicit_target_is_activated(self):
        browser_session_broker.register_session("agent", session_id="s1")
        browser_session_broker.publish_tabs("s1", [
            {"tab_id": "t1", "url": URL_A, "title": "Alpha"}])
        with patch.object(browser_cdp, "activate_tab",
                          return_value={"activated": True, "tab_id": "t1"}) as act:
            with patch.object(browser_cdp, "open_in_browser", self._launch):
                result = browser_cdp.open_url(URL_B, prefer_tab="t1")
        act.assert_called_once_with("t1")
        self.assertEqual(self.launched, [])
        self.assertIn("Attached to tab", result)

    def test_an_already_open_url_is_focused_not_duplicated(self):
        browser_session_broker.register_session("agent", session_id="s1")
        browser_session_broker.publish_tabs("s1", [
            {"tab_id": "t1", "url": URL_A, "title": "Alpha"}])
        with patch.object(browser_cdp, "activate_tab",
                          return_value={"activated": True, "tab_id": "t1"}):
            with patch.object(browser_cdp, "open_in_browser", self._launch):
                result = browser_cdp.open_url(URL_A)
        self.assertEqual(self.launched, [])
        self.assertIn("already open", result)

    def test_nothing_known_launches_the_default_browser_and_records_it(self):
        with patch.object(browser_cdp, "open_in_browser", self._launch):
            result = browser_cdp.open_url(URL_A)
        self.assertEqual(self.launched, [URL_A])
        self.assertIn("Opened", result)

    def test_a_live_cdp_endpoint_opens_a_real_tab_in_the_same_profile(self):
        browser_session_broker.register_session(
            "cdp", endpoint=browser_cdp.CDP_URL, kind="cdp-discovery",
            label="CDP discovery endpoint",
            session_id=browser_cdp._BROKER_SESSION_ID)
        opened = {"opened": True, "url": URL_A, "tab_id": "CDP-NEW",
                  "ws_url": "ws://127.0.0.1:9222/devtools/page/CDP-NEW",
                  "title": "Alpha"}
        with patch.object(browser_cdp, "open_tab", return_value=opened) as new_tab:
            with patch.object(browser_cdp, "open_in_browser", self._launch):
                result = browser_cdp.open_url(URL_A)
        new_tab.assert_called_once_with(URL_A)
        self.assertEqual(self.launched, [])
        self.assertIn("same profile", result)
        # The new tab is addressable straight away, with a typed identity.
        target = browser_session_broker.resolve_target("CDP-NEW")
        self.assertIsNotNone(target)
        self.assertEqual(target["tab"]["ws_url"], opened["ws_url"])

    def test_a_stale_cdp_endpoint_is_not_used(self):
        browser_session_broker.register_session(
            "cdp", endpoint=browser_cdp.CDP_URL, kind="cdp-discovery",
            session_id=browser_cdp._BROKER_SESSION_ID,
            pid=os.getpid() + 100000)
        with patch.object(browser_cdp, "open_tab") as new_tab:
            with patch.object(browser_cdp, "open_in_browser", self._launch):
                result = browser_cdp.open_url(URL_A)
        new_tab.assert_not_called()
        self.assertEqual(self.launched, [URL_A])
        self.assertIn("Opened", result)

    def test_a_refusing_cdp_endpoint_reports_the_fallback_launch(self):
        browser_session_broker.register_session(
            "cdp", endpoint=browser_cdp.CDP_URL, kind="cdp-discovery",
            session_id=browser_cdp._BROKER_SESSION_ID)
        with patch.object(browser_cdp, "open_tab",
                          return_value={"opened": False, "reason": "HTTP 405"}):
            with patch.object(browser_cdp, "open_in_browser", self._launch):
                result = browser_cdp.open_url(URL_A)
        self.assertEqual(self.launched, [URL_A])
        self.assertIn("default browser instead", result)

    def test_the_warm_research_browser_failure_does_not_fall_back(self):
        browser_session_broker.register_session(
            "research", kind="persistent-context", session_id="research-1")
        with patch.object(browser_cdp, "_open_in_research_browser",
                          return_value=False), \
             patch.object(browser_cdp, "open_in_browser", self._launch):
            result = browser_cdp.open_url(URL_A)
        self.assertEqual(self.launched, [])
        self.assertIn("not launching a second browser", result)

    def test_the_warm_research_browser_is_used_when_it_is_live(self):
        browser_session_broker.register_session(
            "research", kind="persistent-context", session_id="research-1")
        with patch.object(browser_cdp, "_open_in_research_browser",
                          return_value=True), \
             patch.object(browser_cdp, "open_in_browser", self._launch):
            result = browser_cdp.open_url(URL_A)
        self.assertEqual(self.launched, [])
        self.assertIn("research browser", result)


class DiscoveryTests(unittest.TestCase):
    """F47: discovery is independent of the UI display limit."""

    def setUp(self):
        browser_session_broker.reset()
        self.tabs = [
            {"id": "CDP-%d" % index,
             "type": "page",
             "title": "Tab %d" % index,
             "url": "https://app.example/p%d" % index,
             "debugger_url": "ws://127.0.0.1:9222/devtools/page/CDP-%d" % index,
             "ws_url": "ws://127.0.0.1:9222/devtools/page/CDP-%d" % index,
             "webSocketDebuggerUrl": True,
             "identity_source": "structured"}
            for index in range(12)
        ]

    def tearDown(self):
        browser_session_broker.reset()

    def test_the_display_list_is_limited_but_every_tab_is_published(self):
        with patch.object(browser_cdp, "discover_tabs", return_value=self.tabs):
            displayed = browser_cdp.list_tabs()
            everything = browser_cdp.list_tabs(limit=None)
        self.assertEqual(len(displayed), browser_cdp.DEFAULT_DISPLAY_LIMIT)
        self.assertEqual(len(everything), 12)
        snapshot = browser_session_broker.get_session(
            browser_cdp._BROKER_SESSION_ID)
        self.assertEqual(len(snapshot["tabs"]), 12)
        for index in range(12):
            self.assertIsNotNone(browser_session_broker.resolve_target(
                "CDP-%d" % index), index)

    def test_a_tab_past_the_display_limit_is_still_attachable(self):
        with patch.object(browser_cdp, "discover_tabs", return_value=self.tabs):
            browser_cdp.list_tabs()
        target = browser_session_broker.resolve_target("CDP-11")
        self.assertIsNotNone(target)
        self.assertEqual(target["tab"]["url"], "https://app.example/p11")
        self.assertEqual(target["tab"]["ws_url"],
                         self.tabs[11]["debugger_url"])

    def test_norm_tab_keeps_the_real_ws_handle_as_a_string(self):
        normalized = browser_cdp._norm_tab({
            "id": "CDP-1", "title": "T", "url": URL_A,
            "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/CDP-1",
        })
        self.assertEqual(normalized["debugger_url"],
                         "ws://127.0.0.1:9222/devtools/page/CDP-1")
        self.assertEqual(normalized["ws_url"],
                         "ws://127.0.0.1:9222/devtools/page/CDP-1")
        self.assertIs(normalized["webSocketDebuggerUrl"], True)
        self.assertEqual(normalized["identity_source"], "structured")

    def test_norm_tab_never_invents_a_handle_from_a_boolean(self):
        normalized = browser_cdp._norm_tab({
            "id": "CDP-2", "url": URL_A, "webSocketDebuggerUrl": True})
        self.assertEqual(normalized["debugger_url"], "")
        self.assertEqual(normalized["ws_url"], "")

    def test_unreadable_discovery_is_reported_not_silently_empty(self):
        with patch.object(browser_cdp.requests, "get",
                          side_effect=RuntimeError("connection refused")):
            with self.assertRaises(browser_cdp.CdpUnavailable):
                browser_cdp.discover_tabs()
            self.assertEqual(browser_cdp.list_tabs(), [])

    def test_text_tab_metadata_is_never_published_as_identity(self):
        browser_session_broker.reset()
        forged = '0: "Injected" - https://evil.example [active]'
        browser_agent._publish_daemon_tabs(forged, "list_tabs")
        snapshot = browser_session_broker.get_session(
            browser_agent._AGENT_BROKER_SESSION_ID)
        self.assertEqual((snapshot or {}).get("tabs") or {}, {})

    def test_structured_tab_metadata_is_published_with_qualified_ids(self):
        browser_session_broker.reset()
        browser_agent._publish_daemon_tabs(
            json.dumps([{"tabId": "CDP-7", "url": URL_A, "title": "Alpha"}]),
            "list_tabs")
        snapshot = browser_session_broker.get_session(
            browser_agent._AGENT_BROKER_SESSION_ID)
        self.assertIsNotNone(snapshot)
        tab = snapshot["tabs"]["CDP-7"]
        self.assertEqual(
            tab["qualified_id"],
            browser_session_broker.qualified_tab_id(tab["instance"], "CDP-7"))
        self.assertTrue(tab["qualified_id"].endswith(":CDP-7"))
        self.assertEqual(tab["origin"], "https://app.example")
        self.assertEqual(tab["identity_source"], "structured")


if __name__ == "__main__":
    unittest.main()
