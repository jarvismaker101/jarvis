"""F15 — the editor bridge as a coding interface (Python-side contract).

Acceptance (audit report): "Same-position content changes, focus switches,
missing state, and stale members of multi-file edits cause no unintended edit.
Extension implementation/authentication must be supplied for closure."

The audited defects pinned here:

  * the active-selection check compared uri/selection only — a same-position
    content change (version bump) and a missing version passed;
  * ``edit_active_selection`` POSTed ``{"text": ...}`` alone, so a check/use
    race applied the edit to whatever was focused, and the agent failed OPEN
    when the state re-check raised;
  * ``apply_workspace_edit``'s expected versions were optional and a single
    top-level version stood in for every member of a multi-file edit;
  * structured inspection (diagnostics, versions, buffers) became clipped
    prose before the planner saw it.

Everything here mocks the HTTP layer (``_get_json``/``_post_json``); no editor,
browser or model is touched. The extension-side check/apply itself lives in
``integrations/jarvis-editor-bridge/extension.js`` (not in this module's
ownership) and is reported as the remaining closure gap.
"""

import json
import unittest
from unittest.mock import patch

from backend.services.task_agent import agent
from backend.services.task_agent.connectors import editor_bridge

URI = "file:///c:/proj/app.py"
PATH = "C:\\proj\\app.py"
SELECTION = {"start": {"line": 1, "character": 0},
             "end": {"line": 1, "character": 4}}
DIAGNOSTICS = [{"file": PATH, "severity": 0, "message": "undefined name 'x'",
                "code": "F821",
                "range": {"start": {"line": 11, "character": 2}}}]


def live_state(uri=URI, version=7, selection=None):
    return {"workspaceFolders": ["C:\\proj"],
            "activeFile": {"uri": uri, "path": PATH, "fileName": "app.py",
                           "languageId": "python", "lineCount": 120,
                           "version": version,
                           "selection": selection or SELECTION},
            "diagnostics": list(DIAGNOSTICS)}


def editor_context(state=None):
    return {"windows": {"active_window": {"title": "code.exe"},
                        "visible_controls": []},
            "editor": {"available": True, "state": state or {}},
            "browser": {"available": False, "tabs": []}}


class _PostSpy:
    """Stands in for the bridge's HTTP POST and records what was sent."""

    def __init__(self, response=None):
        self.calls = []
        self.response = response or {"ok": True, "message": "Applied 1 edit(s)."}

    def __call__(self, path, payload, timeout=2.0):
        self.calls.append((path, payload))
        return dict(self.response)

    @property
    def edits(self):
        out = []
        for _path, payload in self.calls:
            out.extend(payload.get("edits") or [])
        return out


# ── 1. the selection edit needs uri + version + range ─────────────────────
class SelectionEditPreconditionTests(unittest.TestCase):

    def test_a_text_only_edit_is_refused_without_any_request(self):
        spy = _PostSpy()
        with patch.object(editor_bridge, "_post_json", side_effect=spy):
            result = editor_bridge.edit_active_selection("replacement text")
        self.assertFalse(result["ok"])
        self.assertEqual(spy.calls, [],
                         "a text-only edit must never be POSTed")
        self.assertIn("uri", result["error"])

    def test_a_version_less_or_rangeless_edit_is_refused(self):
        spy = _PostSpy()
        with patch.object(editor_bridge, "_post_json", side_effect=spy):
            no_version = editor_bridge.edit_active_selection(
                "text", uri=URI, selection=SELECTION)
            no_range = editor_bridge.edit_active_selection(
                "text", uri=URI, version=7)
        self.assertFalse(no_version["ok"])
        self.assertIn("version", no_version["error"])
        self.assertFalse(no_range["ok"])
        self.assertIn("range", no_range["error"])
        self.assertEqual(spy.calls, [])

    def test_a_versioned_edit_uses_the_version_checked_endpoint(self):
        spy = _PostSpy()
        with patch.object(editor_bridge, "_get_json",
                          return_value=live_state()), \
             patch.object(editor_bridge, "_post_json", side_effect=spy):
            result = editor_bridge.edit_active_selection(
                "new text", uri=URI, version=7, selection=SELECTION)
        self.assertTrue(result["ok"], result)
        self.assertEqual([path for path, _ in spy.calls], ["/apply-edit"],
                         "the unversioned /edit-active-selection POST is "
                         "never used")
        edit = spy.edits[0]
        self.assertEqual(edit["expectedVersion"], 7)
        self.assertEqual(edit["newText"], "new text")
        self.assertEqual(edit["range"], SELECTION)
        self.assertEqual(spy.calls[0][1]["expectedVersion"], 7)


# ── 2. drift / missing state causes no edit ───────────────────────────────
class SelectionDriftTests(unittest.TestCase):

    def _attempt(self, state):
        spy = _PostSpy()
        with patch.object(editor_bridge, "_get_json", return_value=state), \
             patch.object(editor_bridge, "_post_json", side_effect=spy):
            text = agent._execute_step(
                {"tool": "editor.edit_active_selection",
                 "args": {"text": "new"}},
                editor_context(live_state()))
        return text, spy

    def test_a_same_position_content_change_stale_version_causes_no_edit(self):
        # The selection is byte-identical, but the document moved to version 8
        # (the user typed). The plan's version 7 is stale: no edit.
        text, spy = self._attempt(live_state(version=8))
        self.assertIn("editor.edit_active_selection failed", text)
        self.assertIn("selection changed", text)
        self.assertEqual(spy.calls, [])

    def test_a_focus_switch_causes_no_edit(self):
        text, spy = self._attempt(
            live_state(uri="file:///c:/proj/other.py"))
        self.assertIn("editor.edit_active_selection failed", text)
        self.assertEqual(spy.calls, [])

    def test_missing_state_causes_no_edit(self):
        text, spy = self._attempt({})
        self.assertIn("editor.edit_active_selection failed", text)
        self.assertIn("selection changed", text)
        self.assertEqual(spy.calls, [])

    def test_a_bridge_failure_does_not_fail_open(self):
        # The re-check used to swallow an exception and edit anyway.
        spy = _PostSpy()
        with patch.object(editor_bridge, "active_selection_matches",
                          side_effect=RuntimeError("bridge gone")), \
             patch.object(editor_bridge, "_post_json", side_effect=spy), \
             self.assertLogs(level="WARNING"):
            text = agent._execute_step(
                {"tool": "editor.edit_active_selection",
                 "args": {"text": "new"}},
                editor_context(live_state()))
        self.assertIn("editor.edit_active_selection failed", text)
        self.assertIn("could not be verified", text)
        self.assertEqual(spy.calls, [])

    def test_missing_plan_time_version_causes_no_edit(self):
        # The plan-time snapshot has a selection but no version: it cannot be
        # pinned, so nothing is attempted.
        state = {"activeFile": {"uri": URI, "selection": SELECTION}}
        spy = _PostSpy()
        with patch.object(editor_bridge, "_post_json", side_effect=spy), \
             patch.object(editor_bridge, "_get_json",
                          return_value=live_state()) as fake_get:
            text = agent._execute_step(
                {"tool": "editor.edit_active_selection",
                 "args": {"text": "new"}}, editor_context(state))
        self.assertIn("editor.edit_active_selection failed", text)
        self.assertEqual(spy.calls, [])
        fake_get.assert_not_called()

    def test_a_matching_live_target_does_edit(self):
        spy = _PostSpy()
        with patch.object(editor_bridge, "_get_json",
                          return_value=live_state()), \
             patch.object(editor_bridge, "_post_json", side_effect=spy):
            text = agent._execute_step(
                {"tool": "editor.edit_active_selection",
                 "args": {"text": "new"}},
                editor_context(live_state()))
        self.assertEqual([path for path, _ in spy.calls], ["/apply-edit"])
        self.assertEqual(spy.edits[0]["newText"], "new")

    def test_active_selection_matches_compares_the_version(self):
        with patch.object(editor_bridge, "_get_json",
                          return_value=live_state(version=8)):
            self.assertFalse(editor_bridge.active_selection_matches(
                expected_uri=URI, expected_selection=SELECTION,
                expected_version=7))
            self.assertTrue(editor_bridge.active_selection_matches(
                expected_uri=URI, expected_selection=SELECTION,
                expected_version=8))
        with patch.object(editor_bridge, "_get_json", return_value={}):
            self.assertFalse(editor_bridge.active_selection_matches(
                expected_uri=URI, expected_version=7))
        with patch.object(editor_bridge, "_get_json", return_value={
                "activeFile": {"uri": URI, "selection": SELECTION}}):
            self.assertFalse(editor_bridge.active_selection_matches(
                expected_uri=URI, expected_selection=SELECTION,
                expected_version=7))


# ── 3. per-document preconditions for WorkspaceEdits ─────────────────────
class WorkspaceEditPreconditionTests(unittest.TestCase):

    def _edits(self):
        return [
            {"path": "C:\\proj\\a.py",
             "range": {"start": {"line": 0, "character": 0},
                       "end": {"line": 0, "character": 0}},
             "new_text": "a", "version": 7},
            {"path": "C:\\proj\\b.py",
             "range": {"start": {"line": 0, "character": 0},
                       "end": {"line": 0, "character": 0}},
             "new_text": "b", "version": 9},
        ]

    def test_a_multi_file_edit_needs_a_version_for_every_member(self):
        spy = _PostSpy()
        edits = self._edits()
        edits[1].pop("version")
        with patch.object(editor_bridge, "_post_json", side_effect=spy):
            result = editor_bridge.apply_workspace_edit(
                edits, expected_version=7)
        self.assertFalse(result["ok"])
        self.assertIn("per-document", result["error"])
        self.assertEqual(spy.calls, [])

    def test_one_top_level_version_cannot_stand_for_several_documents(self):
        spy = _PostSpy()
        edits = self._edits()
        for edit in edits:
            edit.pop("version")
        with patch.object(editor_bridge, "_post_json", side_effect=spy):
            result = editor_bridge.apply_workspace_edit(
                edits, expected_version=7)
        self.assertFalse(result["ok"])
        self.assertEqual(spy.calls, [])

    def test_a_single_document_edit_may_use_the_declared_version(self):
        spy = _PostSpy()
        with patch.object(editor_bridge, "_post_json", side_effect=spy):
            result = editor_bridge.apply_workspace_edit(
                [{"path": "C:\\proj\\a.py", "new_text": "x"}],
                expected_version=7)
        self.assertTrue(result["ok"], result)
        self.assertEqual(spy.edits[0]["expectedVersion"], 7)

    def test_contradicting_versions_are_refused(self):
        spy = _PostSpy()
        with patch.object(editor_bridge, "_post_json", side_effect=spy):
            result = editor_bridge.apply_workspace_edit(
                self._edits(), expected_version=7)
        self.assertFalse(result["ok"])
        self.assertEqual(spy.calls, [])

    def test_an_edit_without_a_target_or_text_is_refused(self):
        spy = _PostSpy()
        with patch.object(editor_bridge, "_post_json", side_effect=spy):
            no_target = editor_bridge.apply_workspace_edit(
                [{"new_text": "x", "version": 7}])
            no_text = editor_bridge.apply_workspace_edit(
                [{"path": "C:\\proj\\a.py", "version": 7}])
        self.assertFalse(no_target["ok"])
        self.assertFalse(no_text["ok"])
        self.assertEqual(spy.calls, [])

    def test_a_stale_member_rejects_the_whole_edit(self):
        """The extension validates every member before applying any of them."""
        live = {"C:\\proj\\a.py": 7, "C:\\proj\\b.py": 9}
        applied = []

        def fake_post(path, payload, timeout=2.0):
            for edit in payload.get("edits") or []:
                key = edit.get("path")
                if live.get(key) != edit.get("expectedVersion"):
                    return {"ok": False, "status": 409,
                            "error": ("Version mismatch for %s: document is "
                                      "at version %s, expected %s."
                                      % (key, live.get(key),
                                         edit.get("expectedVersion"))),
                            "expectedVersion": edit.get("expectedVersion"),
                            "actualVersion": live.get(key)}
            applied.extend(payload.get("edits") or [])
            return {"ok": True, "message": "Applied 2 edit(s)."}

        edits = self._edits()
        edits[1]["version"] = 8  # stale: the live document is at 9
        with patch.object(editor_bridge, "_post_json", side_effect=fake_post):
            result = editor_bridge.apply_workspace_edit(edits)
        self.assertFalse(result["ok"])
        self.assertEqual(result.get("status"), 409)
        self.assertEqual(applied, [], "a stale member applied nothing")

        # Through the agent the failure is honest and points at a re-read.
        context = editor_context(live_state())
        with patch.object(editor_bridge, "_post_json", side_effect=fake_post):
            text = agent._execute_step(
                {"tool": "editor.apply_workspace_edit",
                 "args": {"edits": edits}}, context)
        self.assertIn("editor.apply_workspace_edit failed", text)
        self.assertIn("editor.read_buffer", text)
        self.assertEqual(applied, [])

    def test_the_agent_sends_every_per_document_precondition(self):
        spy = _PostSpy(response={"ok": True, "message": "Applied 2 edit(s)."})
        edits = self._edits()
        with patch.object(editor_bridge, "_post_json", side_effect=spy):
            text = agent._execute_step(
                {"tool": "editor.apply_workspace_edit",
                 "args": {"edits": edits}}, editor_context(live_state()))
        self.assertIn("Applied 2 edit(s).", text)
        self.assertEqual([edit["expectedVersion"] for edit in spy.edits],
                         [7, 9])


# ── 4. structured inspection survives into what the planner sees ─────────
class StructuredInspectionTests(unittest.TestCase):

    def test_the_step_result_keeps_the_structured_diagnostics(self):
        with patch.object(agent.editor_bridge, "diagnostics",
                          return_value={"ok": True,
                                        "diagnostics": list(DIAGNOSTICS)}):
            text, structured = agent._execute_step_structured(
                {"tool": "editor.diagnostics", "args": {}},
                editor_context(live_state()))
        self.assertIn("F821", text)
        self.assertTrue(structured["ok"])
        self.assertEqual(structured["diagnostics"][0]["code"], "F821")
        self.assertEqual(structured["diagnostics"][0]["severity"], 0)

    def test_structured_diagnostics_reach_a_later_step_argument(self):
        recorded = []

        def fake_call(tool, args, grants=None):
            recorded.append((tool, dict(args)))
            return {"ok": True, "content": "1 match", "matches": [],
                    "has_more": False}

        plan = agent._normalize_plan(
            {"ok": True,
             "steps": [{"tool": "editor.inspect_workspace", "args": {},
                        "risk": "safe", "reason": "inspect"},
                       {"tool": "code.search",
                        "args": {"pattern": "{{step0.diagnostics.0.code}}",
                                 "path": "{{step0.diagnostics.0.file}}"},
                        "risk": "safe", "reason": "search"}]},
            "find the broken code")
        self.assertFalse(plan["requires_confirmation"])
        with patch.object(agent.code_tools, "call_tool", side_effect=fake_call):
            agent.execute_plan(plan, editor_context(live_state()),
                               confirmed=True)
        self.assertEqual(recorded[0][1]["pattern"], "F821",
                         "the diagnostic's code never reached the next step")
        self.assertEqual(recorded[0][1]["path"], PATH,
                         "the diagnostic's file never reached the next step")

    def test_the_planner_prompt_carries_structured_diagnostics(self):
        context = editor_context(live_state())
        prompt = agent._build_planner_prompt("fix the diagnostics", context)
        self.assertIn('"code": "F821"', prompt)
        self.assertIn('"severity": 0', prompt)
        self.assertIn("undefined name", prompt)
        # F46: the context section must be PARSEABLE, bounded JSON — a sliced
        # serialization would silently lose every structured diagnostic.
        block = prompt.split("CONNECTOR CONTEXT:\n", 1)[1].strip()
        self.assertLessEqual(len(block), 12000)
        self.assertIn('"diagnostics"', json.dumps(json.loads(block)))

        observations = {0: {"tool": "editor.diagnostics", "status": "ok",
                            "diagnostics": list(DIAGNOSTICS), "version": 7}}
        prompt = agent._build_planner_prompt(
            "fix the diagnostics", context, observations)
        self.assertIn("OBSERVED STRUCTURED DATA", prompt)
        self.assertIn('"code": "F821"', prompt)
        self.assertIn('"version": 7', prompt)

    def test_an_oversized_context_is_still_valid_bounded_json(self):
        state = live_state()
        state["noise"] = ["x" * 200] * 200  # serializes well past 12000 chars
        prompt = agent._build_planner_prompt("fix it", editor_context(state))
        block = prompt.split("CONNECTOR CONTEXT:\n", 1)[1].strip()
        self.assertLessEqual(len(block), 12000)
        self.assertIsInstance(json.loads(block), dict)

    def test_read_buffer_keeps_its_version_and_lines_structured(self):
        with patch.object(agent.editor_bridge, "read_buffer",
                          return_value={"ok": True, "path": PATH, "uri": URI,
                                        "version": 3, "lineCount": 40,
                                        "startLine": 1, "endLine": 2,
                                        "text": "def foo():\n    return 1",
                                        "textTruncated": False}):
            _text, structured = agent._execute_step_structured(
                {"tool": "editor.read_buffer", "args": {"path": "app.py"}},
                editor_context(live_state()))
        self.assertEqual(structured["version"], 3)
        self.assertEqual(structured["text"], "def foo():\n    return 1")


if __name__ == "__main__":
    unittest.main()
