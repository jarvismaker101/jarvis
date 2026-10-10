"""LIVE FIX 12/13 — the on-screen folder structure can be replicated.

Live log: "look at my screen there is a project folder structure visible
i want you to see it and replicate it exactly on my desktop" was answered
"Which folder should I check, sir? Please say the folder name." — the
replicate clause was no task kind, the chain gate rejected the sentence,
and the folder-inspect net asked the user to name a folder that was
visible on their screen. LIVE FIX 13 extends it: the reader now also gets
the file names, retries once, falls back to a fresh capture, remembers the
structure, and answers follow-up turns ("ok so create it", "create the
files as well") natively instead of handing them to the browser agent.
"""

import os
import tempfile
import time
import unittest
from unittest.mock import patch

from backend.core import brain
from backend.services.task_agent import agent as task_agent


class _FakeObs:
    def __init__(self, image="data:image/png;base64,AAAA", obs_id="obs1"):
        self.image_data_url = image
        self.id = obs_id


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


class FileEntryParseTests(unittest.TestCase):
    """_parse_file_entries resolves vision file entries into paths."""

    TREE = ["proj", os.path.join("proj", "src")]

    def test_exact_folder_path_places_the_file(self):
        self.assertEqual(
            brain._parse_file_entries(
                [{"name": "index.js", "folder": "proj/src"}], self.TREE),
            [os.path.join("proj", "src", "index.js")])

    def test_bare_folder_name_matches_a_unique_suffix(self):
        self.assertEqual(
            brain._parse_file_entries(
                [{"name": "app.js", "folder": "src"}], self.TREE),
            [os.path.join("proj", "src", "app.js")])

    def test_top_level_file_uses_the_single_root(self):
        self.assertEqual(
            brain._parse_file_entries(
                [{"name": ".gitignore", "folder": ""}], ["proj"]),
            [os.path.join("proj", ".gitignore")])

    def test_unknown_parent_is_never_invented(self):
        self.assertEqual(
            brain._parse_file_entries(
                [{"name": "x.js", "folder": "nowhere"}], ["proj"]),
            [])

    def test_ambiguous_top_level_file_is_skipped(self):
        self.assertEqual(
            brain._parse_file_entries(
                [{"name": "x.js", "folder": ""}], ["a", "b"]),
            [])

    def test_junk_names_are_sanitized_or_dropped(self):
        self.assertEqual(
            brain._parse_file_entries(
                [{"name": "  ", "folder": "proj"},
                 {"name": "..", "folder": "proj"},
                 {"name": "a/b.js", "folder": "proj"},
                 {"name": "bad<>:name?.py", "folder": "proj"}],
                ["proj"]),
            [os.path.join("proj", "b.js"),
             os.path.join("proj", "badname.py")])

    def test_names_dedupe_case_insensitively(self):
        self.assertEqual(
            brain._parse_file_entries(
                [{"name": "Index.js", "folder": "proj"},
                 {"name": "index.js", "folder": "proj"}], ["proj"]),
            [os.path.join("proj", "Index.js")])

    def test_caps_the_file_count(self):
        entries = [{"name": "f%03d.txt" % i, "folder": "proj"}
                   for i in range(brain._MI_TREE_MAX_FILES + 5)]
        self.assertEqual(len(brain._parse_file_entries(entries, ["proj"])),
                         brain._MI_TREE_MAX_FILES)


class ProjectTreeLadderTests(unittest.TestCase):
    """_mi_extract_project_tree: stored read -> retry -> fresh capture."""

    def test_first_empty_read_retries_the_same_image(self):
        obs = _FakeObs()
        entries = ([{"name": "proj", "depth": 0}],
                   [{"name": "a.js", "folder": "proj"}])
        with patch.object(brain, "get_observation", return_value=obs), \
             patch.object(brain, "extract_project_tree",
                          side_effect=[([], []), entries]) as fake:
            paths, files = brain._mi_extract_project_tree("obs1")
        self.assertEqual(fake.call_count, 2)
        self.assertEqual(paths, ["proj"])
        self.assertEqual(files, [os.path.join("proj", "a.js")])

    def test_missing_stored_image_falls_back_to_a_fresh_capture(self):
        entries = ([{"name": "proj", "depth": 0}], [])
        with patch.object(brain, "get_observation", return_value=None), \
             patch.object(brain, "capture_stored_observation",
                          return_value=_FakeObs()) as cap, \
             patch.object(brain, "extract_project_tree",
                          side_effect=[entries]):
            paths, _files = brain._mi_extract_project_tree("obs1")
        self.assertTrue(cap.called)
        self.assertEqual(paths, ["proj"])

    def test_honest_failure_after_every_read(self):
        obs = _FakeObs()
        with patch.object(brain, "get_observation", return_value=obs), \
             patch.object(brain, "capture_stored_observation",
                          return_value=_FakeObs(image="")), \
             patch.object(brain, "extract_project_tree",
                          side_effect=[([], []), ([], [])]):
            paths, files = brain._mi_extract_project_tree("obs1")
        self.assertEqual((paths, files), ([], []))


class FollowupMatcherTests(unittest.TestCase):
    """The follow-up shapes match; plain creates never do."""

    def test_structure_followups_match(self):
        for msg in ("ok so create it", "create it",
                    "can you create them now", "go ahead and create it",
                    "create the folders", "okay so now create it",
                    "and then create it", "create them too"):
            with self.subTest(msg=msg):
                self.assertTrue(brain._MI_FOLLOWUP_RE.search(msg))

    def test_plain_creates_do_not_match(self):
        for msg in ("create a folder named demo", "create a file",
                    "open the folder", "what did you create"):
            with self.subTest(msg=msg):
                self.assertFalse(brain._MI_FOLLOWUP_RE.search(msg))

    def test_file_followups_match(self):
        for msg in ("create the files as well", "create files too",
                    "ok so create the files as well",
                    "create the files in those folders",
                    "create a file in those folders",
                    "write the files as well"):
            with self.subTest(msg=msg):
                self.assertTrue(brain._MI_FOLLOWUP_FILES_RE.search(msg))

    def test_file_followups_need_the_anaphora(self):
        for msg in ("create a file", "write a file called notes",
                    "create the file"):
            with self.subTest(msg=msg):
                self.assertFalse(brain._MI_FOLLOWUP_FILES_RE.search(msg))


class StructureFollowupTests(unittest.TestCase):
    """_mi_replicate_followup_reply resolves the anaphoric creates."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        task_agent._pending_task_action = None
        brain._mi_structure_state.update(
            {"tree": [], "files": [], "base": "", "roots": {},
             "obs_id": "", "at": 0.0})

    def _remember(self, tree, files=(), roots=None, obs_id="obs1"):
        brain._mi_remember_structure(list(tree), list(files),
                                     self._tmp.name, roots, obs_id)

    def _reply(self, msg):
        captured = {}

        def fake_arm(plan, context, task_text=""):
            captured["plan"] = plan
            return object()

        with patch.object(task_agent, "_known_folders",
                          return_value={"desktop": self._tmp.name}), \
             patch.object(task_agent, "_arm_plan_confirmation",
                          side_effect=fake_arm), \
             patch.object(task_agent, "confirmation_prompt",
                          return_value="PROMPT"), \
             patch.object(task_agent, "has_pending_task_confirmation",
                          return_value=False):
            reply = brain._mi_replicate_followup_reply(msg)
        return reply, captured.get("plan")

    def test_create_it_rearms_missing_folders_from_memory(self):
        self._remember(["proj", os.path.join("proj", "src")])
        reply, plan = self._reply("ok so create it")
        self.assertEqual(reply, "PROMPT")
        self.assertEqual([s["tool"] for s in plan["steps"]],
                         ["code.create_folder"] * 2)

    def test_create_it_only_arms_what_is_missing(self):
        os.makedirs(os.path.join(self._tmp.name, "proj"))
        self._remember(["proj", os.path.join("proj", "src")])
        reply, plan = self._reply("create it")
        self.assertEqual(reply, "PROMPT")
        self.assertEqual([s["args"]["path"] for s in plan["steps"]],
                         [os.path.join(self._tmp.name, "proj", "src")])

    def test_create_it_when_everything_exists_nudges_the_files(self):
        os.makedirs(os.path.join(self._tmp.name, "proj", "src"))
        self._remember(["proj", os.path.join("proj", "src")])
        reply, plan = self._reply("create it")
        self.assertIsNone(plan)
        self.assertIn("already there", reply)
        self.assertIn("create the files", reply)

    def test_create_the_files_arms_writes_under_the_existing_root(self):
        os.makedirs(os.path.join(self._tmp.name, "proj", "src"))
        files = [os.path.join("proj", "src", "index.js"),
                 os.path.join("proj", ".gitignore")]
        self._remember(["proj", os.path.join("proj", "src")], files)
        reply, plan = self._reply("create the files as well")
        self.assertEqual(reply, "PROMPT")
        self.assertEqual([s["tool"] for s in plan["steps"]],
                         ["code.write_file", "code.write_file"])
        self.assertEqual(
            sorted(s["args"]["path"] for s in plan["steps"]),
            sorted([os.path.join(self._tmp.name, "proj", "src", "index.js"),
                    os.path.join(self._tmp.name, "proj", ".gitignore")]))
        self.assertTrue(all(s["args"]["create_only"] for s in plan["steps"]))

    def test_create_the_files_when_files_exist_is_honest(self):
        os.makedirs(os.path.join(self._tmp.name, "proj"))
        with open(os.path.join(self._tmp.name, "proj", "a.js"), "w"):
            pass
        self._remember(["proj"], [os.path.join("proj", "a.js")])
        reply, plan = self._reply("create the files as well")
        self.assertIsNone(plan)
        self.assertIn("already there", reply)

    def test_create_the_files_with_no_names_is_honest(self):
        os.makedirs(os.path.join(self._tmp.name, "proj"))
        self._remember(["proj"], [])
        reply, plan = self._reply("create the files as well")
        self.assertIsNone(plan)
        self.assertIn("no file names", reply)

    def test_stale_memory_is_reread_from_the_screen(self):
        self._remember(["old"])
        brain._mi_structure_state["at"] = time.time() - 99999
        with patch.object(brain, "_mi_extract_project_tree",
                          return_value=(["proj"], [])):
            reply, plan = self._reply("ok so create it")
        self.assertEqual(reply, "PROMPT")
        self.assertEqual([s["args"]["path"] for s in plan["steps"]],
                         [os.path.join(self._tmp.name, "proj")])

    def test_empty_memory_and_unreadable_screen_fails_honestly(self):
        with patch.object(brain, "_mi_extract_project_tree",
                          return_value=([], [])):
            reply, plan = self._reply("ok so create it")
        self.assertIsNone(plan)
        self.assertIn("couldn't read a folder structure", reply)

    def test_plain_creates_are_not_structure_followups(self):
        self._remember(["proj"])
        for msg in ("create a folder named demo",
                    "create a file called notes",
                    "make a folder on my desktop"):
            with self.subTest(msg=msg):
                reply, plan = self._reply(msg)
                self.assertIsNone(reply)
                self.assertIsNone(plan)


class ReplicateIntentGateTests(unittest.TestCase):
    """The original sentence routes by the user's OWN words (live fix).

    Live log: "there is a project structure visible at my screen , create a
    exact replica of this on my desktop" was classified as ONE task with a
    rewritten description, armed as a generic opencode confirmation, and
    the confirmed handoff executed the description — which the capability
    resolver sent to the browser agent. This gate reads the user's words,
    reads the structure, and arms the same folder plan the chain step
    would. It must also never hijack a plain create or a screen question.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        task_agent._pending_task_action = None
        brain._mi_structure_state.update(
            {"tree": [], "files": [], "base": "", "roots": {},
             "obs_id": "", "at": 0.0})

    def _reply(self, msg, tree=("proj",), files=("proj/index.js",)):
        captured = {}

        def fake_arm(plan, context, task_text=""):
            captured["plan"] = plan
            return object()

        with patch.object(brain, "capture_stored_observation",
                          return_value=_FakeObs(obs_id="obs9")), \
             patch.object(brain, "_mi_extract_project_tree",
                          return_value=(list(tree), list(files))), \
             patch.object(task_agent, "_known_folders",
                          return_value={"desktop": self._tmp.name}), \
             patch.object(task_agent, "_arm_plan_confirmation",
                          side_effect=fake_arm), \
             patch.object(task_agent, "confirmation_prompt",
                          return_value="PROMPT"), \
             patch.object(task_agent, "has_pending_task_confirmation",
                          return_value=False):
            reply = brain._mi_replicate_intent_reply(msg)
        return reply, captured.get("plan")

    def test_the_original_sentence_arms_the_folder_plan(self):
        msg = ("there is a project structure visible at my screen , "
               "create a exact replica of this on my desktop")
        reply, plan = self._reply(msg)
        self.assertEqual(reply, "PROMPT")
        self.assertIsNotNone(plan)
        self.assertTrue(plan["requires_confirmation"])
        self.assertEqual([s["tool"] for s in plan["steps"]],
                         ["code.create_folder"])
        self.assertEqual(plan["steps"][0]["args"]["path"],
                         os.path.join(self._tmp.name, "proj"))
        self.assertEqual(brain._mi_structure_state["obs_id"], "obs9")

    def test_the_plan_stays_folders_only_and_files_are_remembered(self):
        reply, plan = self._reply(
            "create a exact replica of the structure visible at my screen")
        self.assertEqual(reply, "PROMPT")
        self.assertEqual([s["tool"] for s in plan["steps"]],
                         ["code.create_folder"])
        self.assertEqual(brain._mi_structure_state["files"],
                         ["proj/index.js"])

    def test_an_existing_root_is_never_overwritten(self):
        os.makedirs(os.path.join(self._tmp.name, "proj"))
        reply, plan = self._reply(
            "create a exact replica of the structure visible at my screen")
        self.assertEqual(reply, "PROMPT")
        self.assertEqual(plan["steps"][0]["args"]["path"],
                         os.path.join(self._tmp.name, "proj (2)"))

    def test_a_replica_without_a_screen_reference_returns_none(self):
        reply, plan = self._reply(
            "create a exact replica of this on my desktop")
        self.assertIsNone(reply)
        self.assertIsNone(plan)

    def test_non_replicate_messages_return_none(self):
        for msg in ("create a folder named demo",
                    "what's on my screen",
                    "open chrome and scrape x"):
            with self.subTest(msg=msg):
                reply, plan = self._reply(msg)
                self.assertIsNone(reply)
                self.assertIsNone(plan)

    def test_unreadable_screen_fails_honestly(self):
        reply, plan = self._reply(
            "create a exact replica of the structure visible at my screen",
            tree=())
        self.assertIsNone(plan)
        self.assertIn("couldn't read a folder structure", reply)


class ReplicateTaskStepTests(unittest.TestCase):
    """_mi_replicate_task_step arms ONE folder plan, or fails honestly."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        task_agent._pending_task_action = None
        brain._mi_structure_state.update(
            {"tree": [], "files": [], "base": "", "roots": {},
             "obs_id": "", "at": 0.0})

    def _run(self, tree, files=(), results=None, consumes=None):
        captured = {}

        def fake_arm(plan, context, task_text=""):
            captured["plan"] = plan
            return object()

        step = {"kind": "task", "text": "replicate it exactly on my desktop",
                "consumes": [0] if consumes is None else consumes}
        if results is None:
            results = [{"kind": "screen", "status": "ok",
                        "output_obs_id": "obs1"}]
        with patch.object(brain, "_mi_extract_project_tree",
                          return_value=(tree, list(files))), \
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

    def test_the_folder_plan_stays_folders_only(self):
        files = [os.path.join("proj", "index.js")]
        result, plan = self._run(["proj"], files=files)
        self.assertEqual(result["status"], "armed")
        self.assertEqual([s["tool"] for s in plan["steps"]],
                         ["code.create_folder"])

    def test_a_successful_arm_remembers_the_structure_and_files(self):
        files = [os.path.join("proj", "index.js")]
        result, _plan = self._run(["proj"], files=files)
        self.assertEqual(result["status"], "armed")
        self.assertEqual(brain._mi_structure_state["tree"], ["proj"])
        self.assertEqual(brain._mi_structure_state["files"], files)
        self.assertEqual(brain._mi_structure_state["obs_id"], "obs1")

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
