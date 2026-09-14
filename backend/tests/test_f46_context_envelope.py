"""F46 — Build One Bounded Context Envelope.

Acceptance clauses pinned by this module:

  * "Fix this using that tab resolves consistently despite focus changes" — the
    envelope is an IMMUTABLE request snapshot: the concrete browser target
    identity (instance/tab id + url + title) is captured once and every
    consumer reads the same identity, even after the caller's connector dicts
    change underneath it.
  * "no truncated JSON" — bounded serialization is always VALID JSON; a slice
    of serialized JSON (the old ``json.dumps(context)[:12000]`` planner path)
    is never produced.
  * "omitted content is retrievable" — every omission is a record with the
    tool/arguments that retrieves the missing content.
  * "long keys/escaping cannot exceed the total bound" — ``render()`` and
    ``to_json()`` stay within TOTAL_BUDGET for pathological keys and
    escape-heavy values.
"""

import json
import unittest
from unittest.mock import patch

from backend.services import context_envelope as envelope_mod
from backend.services.context_envelope import (
    TOTAL_BUDGET,
    ContextEnvelope,
    budgeted_json,
    build_envelope,
)


def _connectors(active_tab_id="tab-7"):
    tabs = [
        {"id": "tab-7", "url": "https://docs.example/a", "title": "Docs",
         "active": active_tab_id == "tab-7"},
        {"id": "tab-9", "url": "https://mail.example/b", "title": "Inbox",
         "active": active_tab_id == "tab-9"},
    ]
    return {
        "windows": {"active_window": {"title": "VS Code", "hwnd": 4242,
                                      "process_id": 111}},
        "editor": {"available": True, "state": {
            "activeFile": {"path": "C:/x/a.py", "version": 7},
            "selectedText": "return total",
            "workspaceFolders": ["C:/x"],
        }},
        "browser": {"available": True, "instance_id": "browser-1", "tabs": tabs},
    }


class ImmutableSnapshotTests(unittest.TestCase):
    """One immutable snapshot — not a live view of mutable connector dicts."""

    def test_envelope_attributes_cannot_be_reassigned(self):
        env = ContextEnvelope(utterance="hi")
        with self.assertRaises(AttributeError):
            env.fields = {"utterance": "hijacked"}
        with self.assertRaises(AttributeError):
            env.identities = {}
        with self.assertRaises(AttributeError):
            env.screen_question = True

    def test_identity_snapshot_is_deep_frozen(self):
        env = build_envelope("fix this", connectors=_connectors())
        with self.assertRaises(TypeError):
            env.identities["browser_target"] = {}
        with self.assertRaises(TypeError):
            env.identities["browser_target"]["tab_id"] = "hijacked"

    def test_caller_mutation_after_build_cannot_change_the_snapshot(self):
        connectors = _connectors()
        capture = {"capture_mode": "active_window", "hwnd": 4242,
                   "region": {"left": 0, "top": 0, "width": 10, "height": 10}}
        env = build_envelope("use that tab", connectors=connectors,
                             capture=capture)

        # Focus moves to the other tab; the caller keeps mutating its dicts.
        connectors["browser"]["tabs"][0]["active"] = False
        connectors["browser"]["tabs"][1]["active"] = True
        connectors["browser"]["tabs"] = [{"id": "tab-9", "url": "https://other"}]
        connectors["windows"]["active_window"]["title"] = "Notepad"
        capture["hwnd"] = 9999
        capture["region"]["left"] = 500

        self.assertEqual(env.identities["browser_target"]["tab_id"], "tab-7")
        self.assertEqual(env.fields["references"]["active_window"], "VS Code")
        self.assertEqual(env.identities["active_window"]["hwnd"], 4242)
        self.assertEqual(env.identities["screen"]["hwnd"], 4242)
        self.assertEqual(
            env.identities["screen"]["region"],
            {"left": 0, "top": 0, "width": 10, "height": 10})

    def test_render_records_omissions_without_changing_the_snapshot(self):
        env = ContextEnvelope(utterance="look", goal_state="x " * 1500,
                              memory_hints="m " * 1500, editor_text="y " * 1500,
                              capture_identity="c " * 1500)
        before = dict(env.fields)
        rendered = env.render()
        self.assertLessEqual(len(rendered), TOTAL_BUDGET)
        self.assertTrue(env.omissions)
        # The snapshot still holds every field: omissions are a VIEW decision.
        self.assertEqual(dict(env.fields), before)


class TabIdentityResolutionTests(unittest.TestCase):
    """Acceptance: "use that tab" resolves consistently despite focus changes."""

    def test_that_tab_keeps_resolving_to_the_snapshot_tab(self):
        connectors = _connectors(active_tab_id="tab-7")
        env = build_envelope("summarise that tab", connectors=connectors)
        self.assertEqual(env.identities["browser_target"]["tab_id"], "tab-7")
        self.assertEqual(env.identities["browser_target"]["url"],
                         "https://docs.example/a")

        # The window focus changes to a different tab (and the connector
        # snapshot is rebuilt with the new active tab).
        focused = _connectors(active_tab_id="tab-9")
        resolved = env.resolve_reference("that tab")
        self.assertEqual(resolved["kind"], "browser_target")
        self.assertEqual(resolved["identity"]["tab_id"], "tab-7")
        # A NEW request snapshot sees the new tab — identity is per request,
        # not a stale process global.
        fresh = build_envelope("summarise that tab", connectors=focused)
        self.assertEqual(fresh.identities["browser_target"]["tab_id"], "tab-9")

    def test_every_consumer_reads_the_same_identity(self):
        env = build_envelope("fix this error", connectors=_connectors())
        snapshot_identity = env.snapshot()["identities"]["browser_target"]
        for consumer in ("planner", "chat", "screen", "research"):
            view = env.for_consumer(consumer)
            self.assertEqual(view["identities"]["browser_target"],
                             snapshot_identity)
            self.assertEqual(view["identities"]["editor"]["version"], 7)
        self.assertIn("tab-7", env.browser_target)

    def test_consumer_field_selection_is_explicit(self):
        env = build_envelope("fix this error", connectors=_connectors(),
                             selected_text="print(total)")
        screen = env.for_consumer("screen")
        self.assertIn("capture", screen)
        self.assertNotIn("editor_text", screen["fields"])
        chat = env.for_consumer("chat")
        self.assertNotIn("editor_text", chat["fields"])
        planner = env.for_consumer("planner")
        self.assertIn("editor_text", planner["fields"])
        self.assertEqual(planner["fields"]["editor_text"], "print(total)")

    def test_duplicate_titles_still_resolve_by_tab_id(self):
        connectors = _connectors()
        for tab in connectors["browser"]["tabs"]:
            tab["title"] = "Documents"
        env = build_envelope("use that tab", connectors=connectors)
        identity = env.identities["browser_target"]
        self.assertEqual(identity["title"], "Documents")
        self.assertEqual(identity["tab_id"], "tab-7")
        self.assertEqual(env.fields["references"]["browser_tabs"],
                         ["Documents", "Documents"])

    def test_resolution_falls_back_to_window_then_editor(self):
        env = build_envelope("fix this", connectors={
            "windows": {"active_window": {"title": "Notepad", "hwnd": 7}},
        })
        resolved = env.resolve_reference("this")
        self.assertEqual(resolved["kind"], "active_window")
        self.assertEqual(resolved["identity"]["hwnd"], 7)

        editor_only = build_envelope("fix this", connectors={
            "editor": {"available": True, "state": {
                "activeFile": {"path": "C:/x/a.py", "version": 3}}},
        })
        resolved = editor_only.resolve_reference("this")
        self.assertEqual(resolved["kind"], "editor")
        self.assertEqual(resolved["identity"]["version"], 3)

    def test_capture_identity_names_the_concrete_target(self):
        env = build_envelope(
            "what is this", connectors=_connectors(),
            capture={"capture_mode": "region", "hwnd": 4242, "monitor": 2,
                     "observed_at": "2026-01-01T00:00:00+00:00", "epoch": 11})
        identity = env.identities["screen"]
        self.assertEqual(identity["capture_mode"], "region")
        self.assertEqual(identity["monitor"], 2)
        self.assertEqual(identity["epoch"], 11)
        self.assertIn("capture_mode=region", env.capture_identity)


class OmissionRetrievalTests(unittest.TestCase):
    """Acceptance: omitted content is retrievable."""

    def test_every_omission_carries_a_retrieval_hint(self):
        env = ContextEnvelope(utterance="look", goal_state="g " * 1500,
                              memory_hints="m " * 1500, editor_text="e " * 1500,
                              browser_target="b " * 1500,
                              capture_identity="c " * 1500)
        rendered = env.render()
        self.assertTrue(env.omissions)
        self.assertIn("[omitted:", rendered)
        for record in env.omission_records:
            self.assertIn("field", record)
            self.assertIn("reason", record)
            self.assertTrue(record["retrieve"],
                            "omission %s has no retrieval path" % record["field"])
            self.assertIn(record["field"], rendered)
            self.assertIn(record["retrieve"][:20], rendered)
        hints = env.retrieval_hints
        for field in env.omissions:
            self.assertTrue(hints[field], "omitted %s has no hint" % field)
        # The hint table names the real retrieval tool for every field, so an
        # omitted field is always recoverable through a tool call.
        table = envelope_mod.RETRIEVAL_HINTS
        self.assertIn("memory.recall", table["memory_hints"])
        self.assertIn("read_buffer", table["editor_text"])
        self.assertIn("inspect_tabs", table["browser_target"])
        self.assertIn("screen.observe", table["capture_identity"])

    def test_a_dropped_memory_field_still_names_its_retrieval_tool(self):
        env = ContextEnvelope(
            utterance="look",
            goal_state="g " * 1500,
            references={"k%d" % i: "v" * 24 for i in range(60)},
            memory_hints="m " * 1500,
            editor_text="e " * 1500,
            browser_target="b " * 1500,
            capture_identity="c " * 1500)
        rendered = env.render()
        self.assertLessEqual(len(rendered), TOTAL_BUDGET)
        records = {record["field"]: record for record in env.omission_records}
        self.assertIn("memory_hints", records)
        self.assertIn("memory.recall", records["memory_hints"]["retrieve"])
        self.assertIn("memory.recall", rendered)

    def test_omissions_are_reported_in_to_json_too(self):
        env = ContextEnvelope(utterance="look", goal_state="g " * 1500,
                              memory_hints="m " * 1500, editor_text="e " * 1500,
                              capture_identity="c " * 1500)
        payload = json.loads(env.to_json())
        self.assertTrue(payload["omissions"])
        for record in payload["omissions"]:
            self.assertTrue(record["retrieve"])

    def test_summarized_references_explain_their_retrieval(self):
        env = ContextEnvelope(
            utterance="hi",
            references={"tab%d" % i: "x" * 50 for i in range(200)})
        rendered = env.render()
        self.assertLessEqual(len(rendered), TOTAL_BUDGET)
        self.assertIn("_note", json.dumps(env.fields["references"]))
        # The references summary is itself bounded and marked as an omission
        # with the tools that can fetch the real content back.
        self.assertTrue(
            any("references" in record["field"] for record in env.omission_records)
            or env.omissions == [])


class BoundedSerializationTests(unittest.TestCase):
    """Acceptance: no truncated JSON; long keys/escaping cannot exceed bound."""

    def test_bounded_serialization_is_valid_json(self):
        payload = {"tabs": [{"id": "t%d" % i, "text": "y" * 200}
                            for i in range(500)]}
        text = budgeted_json(payload, 12000)
        self.assertLessEqual(len(text), 12000)
        json.loads(text)  # must not raise: a sliced prefix would

    def test_old_planner_slice_is_what_budgeted_json_replaces(self):
        """The planner path (task_agent/_build_planner_prompt) used
        ``json.dumps(context)[:12000]``; this pins the bounded replacement the
        remaining call site must switch to."""
        context = {"windows": {"visible_controls": [
            {"name": "control-%d" % i, "control_type": "Button",
             "metadata": "z" * 40} for i in range(2000)]}}
        sliced = json.dumps(context, ensure_ascii=True)[:12000]
        with self.assertRaises(json.JSONDecodeError):
            json.loads(sliced)
        bounded = budgeted_json(context, 12000)
        self.assertLessEqual(len(bounded), 12000)
        parsed = json.loads(bounded)
        self.assertIn("_note", parsed)

    def test_small_values_round_trip_unchanged(self):
        value = {"a": 1, "b": ["x", "y"]}
        self.assertEqual(json.loads(budgeted_json(value, 12000)), value)

    def test_long_keys_and_escaping_cannot_exceed_the_bound(self):
        pathological = {
            "k" * 5000: "v" * 100000,
            "quote\"and\\slash\nnewline": "\"" * 20000,
            "nested": {"deep" * 500: ["\n" * 3000, {"x": "y" * 50000}]},
        }
        env = ContextEnvelope(utterance="q" * 50000,
                              references=pathological,
                              goal_state="g" * 50000,
                              memory_hints="m" * 50000,
                              editor_text="e" * 50000,
                              browser_target="b" * 50000,
                              capture_identity="c" * 50000)
        rendered = env.render()
        self.assertLessEqual(len(rendered), TOTAL_BUDGET)
        text = env.to_json()
        self.assertLessEqual(len(text), TOTAL_BUDGET)
        json.loads(text)
        # Summarized keys are themselves bounded.
        for key in (env.fields["references"] or {}).get("keys", []):
            self.assertLessEqual(len(key), envelope_mod.MAX_KEY_CHARS)

    def test_render_bound_holds_for_a_battery_of_hostile_inputs(self):
        hostile = [
            {"utterance": "\x00\x01" * 4000},
            {"utterance": "😀" * 8000, "memory_hints": "\t" * 9000},
            {"utterance": "a", "references": {"x": "\"" * 20000}},
            {"utterance": "a", "references": ["y" * 30000]},
            {"utterance": "a", "references": "z" * 30000},
            {"utterance": "a", "goal_state": "\n" * 30000},
        ]
        for kwargs in hostile:
            env = ContextEnvelope(**kwargs)
            self.assertLessEqual(len(env.render()), TOTAL_BUDGET, kwargs)
            self.assertLessEqual(len(env.to_json()), TOTAL_BUDGET, kwargs)
            json.loads(env.to_json())

    def test_no_truncated_json_marker_in_the_json_view(self):
        env = ContextEnvelope(utterance="u" * 9000, goal_state="g" * 9000,
                              references={"k%d" % i: "v" * 100
                                          for i in range(300)})
        text = env.to_json()
        self.assertLessEqual(len(text), TOTAL_BUDGET)
        self.assertNotIn(envelope_mod._TRUNC_MARK, text)
        json.loads(text)


class SingleSnapshotBuilderTests(unittest.TestCase):
    """The builder hands ONE snapshot to every consumer."""

    def test_build_envelope_fills_every_identity(self):
        env = build_envelope("fix this error", connectors=_connectors(),
                             capture={"capture_mode": "active_window",
                                      "hwnd": 4242, "epoch": 3})
        self.assertEqual(env.fields["references"]["editor_file"], "C:/x/a.py")
        self.assertEqual(env.fields["references"]["editor_version"], 7)
        self.assertEqual(env.fields["references"]["editor_selection"],
                         "return total")
        self.assertEqual(env.identities["editor"]["selected_text"],
                         "return total")
        self.assertEqual(env.selected_text, "return total")
        self.assertEqual(env.identities["active_window"]["process_id"], 111)
        self.assertEqual(env.identities["browser_target"]["instance_id"],
                         "browser-1")

    def test_build_envelope_accepts_history_and_memory_hints(self):
        env = build_envelope(
            "continue",
            history=[{"user": "open the file", "assistant": "opened it"}],
            connectors=_connectors(),
            memory_hints="prefers dark mode",
            screen_question=False)
        self.assertIn("user: open the file", env.goal_state)
        self.assertIn("jarvis: opened it", env.goal_state)
        self.assertEqual(env.memory_hints, "prefers dark mode")

    def test_missing_context_leaves_identities_empty_not_invented(self):
        env = build_envelope("hello")
        self.assertEqual(dict(env.identities), {})
        self.assertEqual(env.fields["references"], {})
        self.assertEqual(env.resolve_reference("this")["kind"], None)
        self.assertTrue(env.resolve_reference("this")["retrieve"])


class OrchestratorEnvelopeWiringTests(unittest.TestCase):
    """The orchestrator builds the ONE snapshot (and only once) per request."""

    def test_build_envelope_gathers_identities_once(self):
        from backend.services import orchestrator
        with patch.object(orchestrator, "_gather_connectors",
                          return_value=_connectors()) as gather, \
             patch.object(orchestrator, "_capture_snapshot",
                          return_value={"capture_mode": "active_window",
                                        "hwnd": 4242, "epoch": 5}) as capture:
            env = orchestrator._build_envelope(
                "use that tab", history=[{"user": "hi"}], screen_question=False)
        gather.assert_called_once()
        capture.assert_called_once()
        self.assertEqual(env.identities["browser_target"]["tab_id"], "tab-7")
        self.assertEqual(env.identities["screen"]["epoch"], 5)

    def test_supplied_connectors_are_not_gathered_again(self):
        from backend.services import orchestrator
        with patch.object(orchestrator, "_gather_connectors") as gather, \
             patch.object(orchestrator, "_capture_snapshot", return_value=None):
            env = orchestrator._build_envelope("hello", connectors=_connectors())
        gather.assert_not_called()
        self.assertEqual(env.identities["active_window"]["hwnd"], 4242)

    def test_missing_context_still_yields_a_usable_envelope(self):
        from backend.services import orchestrator
        with patch.object(orchestrator, "_gather_connectors",
                          return_value=None), \
             patch.object(orchestrator, "_capture_snapshot", return_value=None):
            env = orchestrator._build_envelope("hello")
        self.assertEqual(env.utterance, "hello")
        self.assertEqual(dict(env.identities), {})
        self.assertLessEqual(len(env.render()), TOTAL_BUDGET)

    def test_connector_gather_is_time_bounded(self):
        import time
        from backend.services import orchestrator
        from backend.services.task_agent import agent as task_agent

        def _slow():
            time.sleep(5)
            return {}

        with patch.object(task_agent, "gather_context", side_effect=_slow):
            started = time.time()
            result = orchestrator._gather_connectors(timeout=0.05)
            elapsed = time.time() - started
        self.assertIsNone(result)
        self.assertLess(elapsed, 2.0)

    def test_connector_gather_failure_is_not_fatal(self):
        from backend.services import orchestrator
        from backend.services.task_agent import agent as task_agent
        with patch.object(task_agent, "gather_context",
                          side_effect=RuntimeError("connector down")):
            self.assertIsNone(orchestrator._gather_connectors(timeout=1.0))


if __name__ == "__main__":
    unittest.main()
