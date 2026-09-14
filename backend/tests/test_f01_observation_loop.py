"""F01 — observation-driven, dependency-aware task execution.

Acceptance (audit report): "Failed prerequisites block dependents; discovered
filenames inform later arguments; ten intended steps cannot be completed by
executing eight and silently omitting two."

Baseline defects pinned here:
  * ``_normalize_plan`` DROPPED ``depends_on``, so dependency enforcement was
    decorative;
  * execution was a precomputed list — a filename discovered by step 2 could
    never be used by step 3;
  * the 8-step cap was recorded in the summary but the result was still
    ``completed``, so 8 of 10 intended steps read as a finished task.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from backend.services.task_agent import agent
from backend.services.task_result import TaskResult


class DependencyPreservationTests(unittest.TestCase):
    def test_depends_on_survives_normalization(self):
        plan = {
            "ok": True,
            "steps": [
                {"tool": "code.create_folder", "args": {"path": "C:/tmp/x"}},
                {"tool": "code.write_file",
                 "args": {"path": "C:/tmp/x/a.txt", "content": "hi"},
                 "depends_on": [0]},
            ],
        }
        normalized = agent._normalize_plan(plan, "make a file")
        self.assertEqual(normalized["steps"][1]["depends_on"], [0])

    def test_forward_and_self_dependencies_are_refused(self):
        plan = {
            "ok": True,
            "steps": [
                {"tool": "code.list_directory", "args": {"path": "C:/tmp"},
                 "depends_on": [1]},
                {"tool": "code.list_directory", "args": {"path": "C:/tmp"},
                 "depends_on": [1]},
                {"tool": "code.list_directory", "args": {"path": "C:/tmp"},
                 "depends_on": [True]},
            ],
        }
        normalized = agent._normalize_plan(plan, "look")
        self.assertEqual(normalized["steps"][0]["depends_on"], [])
        self.assertIn("unprovable_dependency", normalized["steps"][0])
        # A forward reference and a bool are both unprovable.
        self.assertIn("unprovable_dependency", normalized["steps"][1])
        self.assertIn("unprovable_dependency", normalized["steps"][2])
        self.assertTrue(normalized["dependency_errors"])
        # An unprovable dependency can never run without asking.
        self.assertTrue(normalized["requires_confirmation"])

    def test_failed_prerequisite_blocks_its_dependent(self):
        calls = []

        def fake_execute(step, context):
            calls.append(step["tool"])
            if step["tool"] == "code.create_folder":
                return ("code.create_folder failed: denied", {"ok": False,
                                                              "error": "denied"})
            return ("did the thing", {"ok": True})

        plan = {
            "ok": True,
            "steps": [
                {"tool": "code.create_folder", "args": {"path": "C:/tmp/nope"}},
                {"tool": "code.write_file",
                 "args": {"path": "C:/tmp/nope/a.txt", "content": "x"},
                 "depends_on": [0]},
            ],
        }
        normalized = agent._normalize_plan(plan, "make it")
        with patch.object(agent, "_execute_step_structured",
                          side_effect=fake_execute):
            result = agent.execute_plan(normalized, {}, confirmed=True)
        self.assertNotIn("code.write_file", calls,
                         "a dependent step must not run after its prerequisite")
        self.assertEqual(result.status, "partial")
        self.assertTrue(any("prerequisite failed" in item
                            for item in result.evidence))

    def test_unprovable_dependency_never_executes(self):
        calls = []

        def fake_execute(step, context):
            calls.append(step["tool"])
            return ("ok", {"ok": True})

        plan = {
            "ok": True,
            "steps": [
                {"tool": "code.list_directory", "args": {"path": "C:/tmp"},
                 "depends_on": [5]},
            ],
        }
        normalized = agent._normalize_plan(plan, "look")
        with patch.object(agent, "_execute_step_structured",
                          side_effect=fake_execute):
            result = agent.execute_plan(normalized, {}, confirmed=True)
        self.assertEqual(calls, [])
        self.assertEqual(result.status, "partial")
        self.assertTrue(any("unprovable dependency" in item
                            for item in result.evidence))


class ObservationDrivenPlanningTests(unittest.TestCase):
    def test_discovered_filename_informs_a_later_argument(self):
        """A real file found by step 0 is what step 1 opens."""
        directory = tempfile.mkdtemp(prefix="f01_")
        discovered = os.path.join(directory, "found-report.txt")
        with open(discovered, "w", encoding="utf-8") as handle:
            handle.write("hello from the discovered file")

        steps = [
            {"tool": "code.list_directory", "args": {"path": directory}},
            {"tool": "code.read_file", "args": {"path": "{{step0.paths.0}}"}},
        ]
        executed = []

        def fake_execute(step, context):
            executed.append(dict(step["args"]))
            if step["tool"] == "code.list_directory":
                return ("1 entry", {"ok": True, "path": directory,
                                    "paths": [discovered]})
            path = step["args"].get("path")
            if path == discovered:
                return ("hello from the discovered file", {"ok": True,
                                                            "path": path})
            return ("code.read_file failed: not found", {"ok": False,
                                                         "error": "not found"})

        plan = agent._normalize_plan({"ok": True, "steps": steps}, "read it")
        with patch.object(agent, "_execute_step_structured",
                          side_effect=fake_execute):
            result = agent.execute_plan(plan, {}, confirmed=True)
        self.assertEqual(executed[1]["path"], discovered,
                         "the discovered filename must reach the next step")
        self.assertEqual(result.status, "completed")

    def test_unresolvable_placeholder_is_left_alone(self):
        resolved = agent._resolve_step_args(
            {"path": "{{step9.path}}", "content": "{{last.text}}"},
            {0: {"path": "C:/tmp/a.txt", "text": "body"}})
        self.assertEqual(resolved["path"], "{{step9.path}}")
        self.assertEqual(resolved["content"], "body")

    def test_artifacts_are_kept_with_the_result(self):
        executed = []

        def fake_execute(step, context):
            executed.append(step)
            return ("wrote it", {"ok": True, "path": "C:/tmp/out.txt"})

        plan = agent._normalize_plan({
            "ok": True,
            "steps": [{"tool": "code.write_file",
                       "args": {"path": "C:/tmp/out.txt", "content": "x"}}],
        }, "write")
        with patch.object(agent, "_execute_step_structured",
                          side_effect=fake_execute):
            result = agent.execute_plan(plan, {}, confirmed=True)
        self.assertEqual(result.artifacts,
                         [{"step": 0, "path": "C:/tmp/out.txt"}])

    def test_changed_effect_after_approval_is_reapproved_not_run(self):
        """F01: a resolved argument is a changed effect — consent re-asked."""
        steps = [
            {"tool": "code.list_directory", "args": {"path": "C:/tmp"}},
            {"tool": "code.write_file",
             "args": {"path": "{{step0.path}}", "content": "x"}},
        ]

        def fake_execute(step, context):
            return ("1 entry", {"ok": True, "path": "C:/tmp/other.txt"})

        plan = agent._normalize_plan({"ok": True, "steps": steps}, "write")
        plan["requires_confirmation"] = True
        written = []
        with patch.object(agent, "_execute_step_structured",
                          side_effect=fake_execute), \
             patch.object(agent, "_arm_plan_confirmation",
                          return_value=None) as armed:
            result = agent.execute_plan(plan, {}, confirmed=True)
        self.assertEqual(written, [])
        self.assertEqual(result.status, "needs_input")
        armed.assert_called_once()


class StepBudgetTests(unittest.TestCase):
    def _ten_step_plan(self):
        steps = [{"tool": "code.list_directory",
                  "args": {"path": "C:/tmp/step%d" % i}}
                 for i in range(10)]
        return agent._normalize_plan({"ok": True, "steps": steps}, "list them")

    def test_eight_of_ten_is_not_completion(self):
        plan = self._ten_step_plan()
        with patch.object(agent, "TASK_MAX_STEPS", 8):
            bounded = agent._normalize_plan(
                {"ok": True, "steps": plan["steps"]}, "list them")
        self.assertTrue(bounded["truncated"])
        self.assertEqual(bounded["intended_steps"], 10)
        self.assertEqual(len(bounded["steps"]), 8)
        self.assertEqual(len(bounded["omitted_steps"]), 2)

        def fake_execute(step, context):
            return ("ok", {"ok": True})

        with patch.object(agent, "_execute_step_structured",
                          side_effect=fake_execute):
            result = agent.execute_plan(bounded, {}, confirmed=True)
        self.assertNotEqual(result.status, "completed")
        self.assertEqual(result.status, "partial")
        self.assertTrue(any("intended steps" in item
                            for item in result.evidence))
        self.assertIn("could not be completed", result.summary)
        self.assertFalse(str(result).startswith("Done, sir."))

    def test_a_complete_plan_still_completes(self):
        plan = agent._normalize_plan({
            "ok": True,
            "steps": [{"tool": "code.list_directory", "args": {"path": "C:/tmp"}}],
        }, "list")

        def fake_execute(step, context):
            return ("ok", {"ok": True})

        with patch.object(agent, "_execute_step_structured",
                          side_effect=fake_execute):
            result = agent.execute_plan(plan, {}, confirmed=True)
        self.assertEqual(result.status, "completed")
        self.assertTrue(str(result).startswith("Done, sir."))

    def test_budget_is_configurable(self):
        self.assertGreaterEqual(agent.TASK_MAX_STEPS, 1)

    def test_truncation_note_names_the_budget(self):
        note = agent._with_truncation_note("Working", True)
        self.assertIn(str(agent.TASK_MAX_STEPS), note)


class ObservationHelperTests(unittest.TestCase):
    def test_nested_observation_lookup(self):
        observations = {0: {"files": [{"path": "C:/tmp/a"}, {"path": "C:/tmp/b"}]}}
        resolved = agent._resolve_step_args(
            {"path": "{{step0.files.1.path}}"}, observations)
        self.assertEqual(resolved["path"], "C:/tmp/b")

    def test_json_payload_is_serialized_for_text_args(self):
        observations = {0: {"matches": ["a", "b"]}}
        resolved = agent._resolve_step_args(
            {"pattern": "{{last.matches}}"}, observations)
        self.assertEqual(json.loads(resolved["pattern"]), ["a", "b"])

    def test_lists_and_nested_dicts_are_walked(self):
        observations = {0: {"path": "C:/tmp/x"}}
        resolved = agent._resolve_step_args(
            {"items": [{"p": "{{step0.path}}"}], "n": 3}, observations)
        self.assertEqual(resolved, {"items": [{"p": "C:/tmp/x"}], "n": 3})


if __name__ == "__main__":
    unittest.main()
