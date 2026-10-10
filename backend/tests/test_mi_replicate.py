"""LIVE FIX 12 — the on-screen folder structure can be replicated.

Live log: "look at my screen there is a project folder structure visible
i want you to see it and replicate it exactly on my desktop" was answered
"Which folder should I check, sir? Please say the folder name." — the
replicate clause was no task kind, the chain gate rejected the sentence,
and the folder-inspect net asked the user to name a folder that was
visible on their screen. These tests cover the parser, the replicate task
step and its dispatch.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from backend.core import brain
from backend.services.task_agent import agent as task_agent


class FolderTreeParseTests(unittest.TestCase):
    """_parse_folder_tree turns vision entries into relative paths."""

    def test_depth_builds_relative_paths_parents_first(self):
        entries = [
            {"name": "proj", "depth": 0},
            {"name": "src", "depth": 1},
            {"name": "tests", "depth": 1},
            {"name": "deep", "depth": 5},
        ]
        self.assertEqual(
            brain._parse_folder_tree(entries),
            ["proj", os.path.join("proj", "src"),
             os.path.join("proj", "tests"),
             os.path.join("proj", "tests", "deep")])

    def test_dict_wrapper_and_junk_entries_are_handled(self):
        self.assertEqual(
            brain._parse_folder_tree({"folders": [
                {"name": "ok", "depth": 0},
                {"name": "  ..  ", "depth": 1},
                {"name": "", "depth": 0},
                "not-a-dict",
                {"name": "bad<na>me:?", "depth": 0},
            ]}),
            ["ok", "badname"])
        self.assertEqual(brain._parse_folder_tree(None), [])
        self.assertEqual(brain._parse_folder_tree({"folders": "x"}), [])

    def test_duplicates_collapse_and_depth_is_clamped(self):
        entries = [
            {"name": "a", "depth": 0},
            {"name": "a", "depth": 0},
            {"name": "b", "depth": 9},
        ]
        self.assertEqual(brain._parse_folder_tree(entries),
                         ["a", os.path.join("a", "b")])

    def test_paths_inside_a_name_keep_only_the_last_component(self):
        self.assertEqual(
            brain._parse_folder_tree([{"name": "x/y/z", "depth": 0}]),
            ["z"])


class ReplicateTaskStepTests(unittest.TestCase):
    """_mi_replicate_task_step arms ONE folder plan, or fails honestly."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        task_agent._pending_task_action = None

    def _run(self, tree, results=None, consumes=None):
        captured = {}

        def fake_arm(plan, context, task_text=""):
            captured["plan"] = plan
            return object()

        step = {"kind": "task", "text": "replicate it exactly on my desktop",
                "consumes": [0] if consumes is None else consumes}
        if results is None:
            results = [{"kind": "screen", "status": "ok",
                        "output_obs_id": "obs1"}]
        with patch.object(brain, "_mi_extract_folder_tree",
                          return_value=tree), \
             patch.object(task_agent, "_known_folders",
                          return_value={"desktop": self._tmp.name}), \
             patch.object(task_agent, "_arm_plan_confirmation",
                          side_effect=fake_arm), \
             patch.object(task_agent, "confirmation_prompt",
                          return_value="PROMPT"), \
             patch.object(task_agent, "has_pending_task_confirmation",
                          return_value=False):
            result = brain._mi_replicate_task_step(
                "replicate it exactly on my desktop", step, results)
        return result, captured.get("plan")

    def test_armed_plan_creates_every_folder_parents_first(self):
        tree = ["proj", os.path.join("proj", "src"),
                os.path.join("proj", "tests")]
        result, plan = self._run(tree)
        self.assertEqual(result["status"], "armed")
        self.assertEqual(result["prompt"], "PROMPT")
        self.assertTrue(plan["requires_confirmation"])
        self.assertEqual([s["tool"] for s in plan["steps"]],
                         ["code.create_folder"] * 3)
        self.assertEqual([s["args"]["path"] for s in plan["steps"]],
                         [os.path.join(self._tmp.name, "proj"),
                          os.path.join(self._tmp.name, "proj", "src"),
                          os.path.join(self._tmp.name, "proj", "tests")])

    def test_an_existing_root_is_never_overwritten(self):
        os.makedirs(os.path.join(self._tmp.name, "proj"))
        result, plan = self._run(["proj"])
        self.assertEqual(result["status"], "armed")
        self.assertEqual(plan["steps"][0]["args"]["path"],
                         os.path.join(self._tmp.name, "proj (2)"))

    def test_unreadable_screen_fails_honestly(self):
        result, plan = self._run([])
        self.assertEqual(result["status"], "failed")
        self.assertIn("couldn't read a folder structure", result["fragment"])
        self.assertIsNone(plan)

    def test_consumes_and_fallback_both_find_the_screen_observation(self):
        for consumes in ([0], []):
            with self.subTest(consumes=consumes):
                result, _plan = self._run(["proj"], consumes=consumes)
                self.assertEqual(result["status"], "armed")

    def test_dispatch_routes_replicate_clauses_here(self):
        with patch.object(brain, "_mi_replicate_task_step",
                          return_value={"kind": "task",
                                        "status": "armed"}) as fake:
            brain._mi_task_step(
                "msg", {"kind": "task",
                        "text": "replicate it exactly on my desktop",
                        "consumes": [0]}, [])
        self.assertTrue(fake.called)

    def test_dispatch_keeps_plain_folder_tasks_on_the_folder_path(self):
        with patch.object(brain, "_mi_folder_task_step",
                          return_value={"kind": "task",
                                        "status": "armed"}) as fake:
            brain._mi_task_step(
                "msg", {"kind": "task",
                        "text": "create a folder named history",
                        "consumes": []}, [])
        self.assertTrue(fake.called)


if __name__ == "__main__":
    unittest.main()
