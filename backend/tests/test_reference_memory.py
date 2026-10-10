"""R20 — the tool/task side of the conversation is real memory.

Live diagnosis (history dump): chain/tool requests, the folder-name
answers and the "confirm" answers never entered history; the chain's
step findings (the identified player, the research summary) were never
written anywhere; and the task path that must resolve "that folder"
read no history at all. A follow-up ("did you put the information
inside that folder?") was answered from a conversation the model
literally could not see.
"""

import os
import tempfile
import time
import unittest
from unittest.mock import patch

from backend.core import brain
from backend.core import memory
from backend.core.memory import add_message, clear_history, get_history
from backend.services import multi_intent
from backend.services.task_agent import agent as task_agent


def _history_texts():
    return [str(m.get("content") or "") for m in get_history()]


class HistoryCapTests(unittest.TestCase):
    def setUp(self):
        clear_history()

    def tearDown(self):
        clear_history()

    def test_history_cap_is_40(self):
        self.assertEqual(memory.MAX_HISTORY, 40)
        for i in range(50):
            add_message("user", "m%d" % i)
        self.assertEqual(len(get_history()), 40)
        self.assertEqual(_history_texts()[-1], "m49")


class ChainTurnCommitTests(unittest.TestCase):
    """A chain request is a real turn: both halves land in history."""

    def setUp(self):
        clear_history()
        brain._pending_folder_name = {"text": "", "at": 0.0}

    def tearDown(self):
        clear_history()
        brain._pending_folder_name = {"text": "", "at": 0.0}

    def test_chain_gate_commits_user_and_ack(self):
        plan = {"ok": True, "source": "s", "command_text": "s",
                "steps": [{"kind": "screen", "text": "t", "index": 0,
                           "consumes": []}]}
        with patch.object(brain.multi_intent, "build_chain",
                          return_value=plan), \
             patch.object(brain, "handle_multi_intent",
                          return_value="Sir, I'll do this in 2 steps."):
            reply = brain.process_message(
                "look at my screen and research it online",
                sync_voice=False)
        self.assertIn("2 steps", reply)
        hist = get_history()
        self.assertEqual(
            [m["role"] for m in hist][-2:], ["user", "assistant"])
        self.assertEqual(hist[-2]["content"],
                         "look at my screen and research it online")
        self.assertIn("2 steps", hist[-1]["content"])

    def test_task_gate_commits_the_user_turn(self):
        with patch.object(brain, "is_explicit_task_request",
                          return_value=True), \
             patch.object(brain, "handle_task_message",
                          return_value="Done, sir."), \
             patch.object(brain, "_record_native_task_outcome"):
            brain.process_message("create a folder named demo",
                                   sync_voice=False)
        hist = get_history()
        self.assertTrue(hist)
        self.assertEqual(hist[-1]["role"], "user")
        self.assertEqual(hist[-1]["content"],
                         "create a folder named demo")


class ChainStepNotesTests(unittest.TestCase):
    """Every finished chain step leaves its specifics in history."""

    def setUp(self):
        clear_history()
        brain._set_chain_findings("")

    def tearDown(self):
        clear_history()
        brain._set_chain_findings("")

    def test_step_notes_and_findings_land_in_history(self):
        plan = {"ok": True, "source": "s", "command_text": "s", "steps": [
            {"kind": "screen", "text": "look at my screen",
             "index": 0, "consumes": []},
            {"kind": "research", "text": "research Yao Ming career online",
             "index": 1, "consumes": [0]},
        ]}
        with patch.object(brain, "analyze_screen", return_value={
                "tip": "A Yao Ming interview video is playing.",
                "topic": "Yao Ming interview",
                "creator": ""}), \
             patch.object(brain, "run_quick_search",
                          return_value={"query": "Yao Ming career",
                                        "spoken_summary":
                                        "Yao Ming is a retired center."}):
            brain._run_multi_intent_chain("look at my screen", plan)
        texts = " || ".join(_history_texts())
        self.assertIn("[screen]", texts)
        self.assertIn("Yao Ming interview", texts)
        self.assertIn("[research]", texts)
        self.assertIn('searched "Yao Ming career"', texts)
        self.assertIn("retired center", texts)
        # The research findings survive the worker thread's death.
        self.assertIn("retired center", brain._get_chain_findings())

    def test_lapsed_findings_return_empty(self):
        brain._set_chain_findings("old findings")
        brain._chain_last_findings["at"] = time.time() - 400.0
        self.assertEqual(brain._get_chain_findings(), "")


class FolderNameTests(unittest.TestCase):
    def test_by_the_name_resolves_the_folder(self):
        self.assertEqual(
            brain._mi_folder_name(
                "create a folder on desktop by the name information and "
                "inside that folder create a txt file by the name random"),
            "information")

    def test_the_file_half_never_names_the_folder(self):
        # LIVE_FOLDER_FILE: the folder phrase is a typo ("byt"), so the
        # file's "by the name info" must not become the folder name.
        self.assertEqual(
            brain._mi_folder_name(
                "create a folder on my desktop byt the name of this website "
                "and inside that folder create a txt file by the name info"),
            "")

    def test_pointer_names_stay_with_the_findings(self):
        self.assertEqual(
            brain._mi_folder_name(
                "create a folder by the name of this website"),
            "")


class FileRequestTests(unittest.TestCase):
    def test_pointer_content_uses_the_findings(self):
        name, content = brain._mi_file_request(
            "create a txt file by the name random and write the "
            "information you found about this player inside that txt file",
            "Yao Ming is a retired center, 7 ft 6 in tall.")
        self.assertEqual(name, "random.txt")
        self.assertIn("retired center", content)
        self.assertNotIn("information you found", content)

    def test_literal_content_beats_the_findings(self):
        name, content = brain._mi_file_request(
            "create a txt file by the name info and write hello from "
            "jarvis inside that txt file",
            "findings that must not appear")
        self.assertEqual(name, "info.txt")
        self.assertIn("hello from jarvis", content)

    def test_a_folder_only_request_never_conjures_a_file(self):
        self.assertEqual(
            brain._mi_file_request("create a folder named demo",
                                   "findings that must not appear"),
            ("", ""))


class FolderNameAnswerTests(unittest.TestCase):
    """The "name it X" answer completes the WHOLE original request."""

    def setUp(self):
        clear_history()
        brain._set_chain_findings("")
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        clear_history()
        brain._set_chain_findings("")
        brain._pending_folder_name = {"text": "", "at": 0.0}
        self._tmp.cleanup()

    def test_answer_arms_folder_and_file_with_findings(self):
        captured = {}

        def fake_arm(plan, context, task_text=""):
            captured["plan"] = plan
            return object()

        brain._set_pending_folder_name(
            "create a folder on desktop and inside that folder create a "
            "txt file by the name random and write the information you "
            "found about this player inside that txt file")
        brain._set_chain_findings(
            "Yao Ming is a retired center, 7 ft 6 in tall.")
        with patch.object(task_agent, "_known_folders",
                          return_value={"desktop": self._tmp.name}), \
             patch.object(task_agent, "_arm_plan_confirmation",
                          side_effect=fake_arm):
            reply = brain.consume_pending_folder_name("name it information")
        self.assertIsNotNone(reply)
        steps = captured["plan"]["steps"]
        self.assertEqual([s["tool"] for s in steps],
                         ["code.create_folder", "code.write_file"])
        self.assertTrue(steps[0]["args"]["path"].endswith("information"))
        self.assertTrue(steps[1]["args"]["path"].endswith("random.txt"))
        self.assertIn("retired center", steps[1]["args"]["content"])
        self.assertFalse(brain._pending_folder_name.get("text"))

    def test_answer_commits_both_turns_to_history(self):
        with patch.object(task_agent, "_known_folders",
                          return_value={"desktop": self._tmp.name}), \
             patch.object(task_agent, "_arm_plan_confirmation",
                          return_value=object()):
            brain._set_pending_folder_name("create a folder")
            brain.consume_pending_folder_name("name it james dark")
        hist = get_history()
        roles = [m["role"] for m in hist]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)
        self.assertEqual(hist[-2]["content"], "name it james dark")
        self.assertTrue(hist[-1]["content"])


class SplitterExpansionTests(unittest.TestCase):
    def test_screen_clause_with_research_markers_expands(self):
        plan = multi_intent.build_chain(
            "find out who this Basketball player on my screen is by "
            "researching on the internet, and create a folder on desktop "
            "by the name information and inside that folder create a txt "
            "file by the name random")
        self.assertIsNotNone(plan)
        self.assertEqual([s["kind"] for s in plan["steps"]],
                         ["screen", "research", "task"])

    def test_expansion_waits_when_research_is_its_own_clause(self):
        plan = multi_intent.build_chain(
            "look at my screen and find out about this website, and then "
            "research its benefits on the internet")
        self.assertIsNotNone(plan)
        self.assertEqual([s["kind"] for s in plan["steps"]],
                         ["screen", "research"])

    def test_single_screen_research_clause_still_splits(self):
        plan = multi_intent.build_chain(
            "research about this verse on my screen from bhagwat gita and "
            "tell me what does it actually say")
        self.assertIsNotNone(plan)
        self.assertEqual([s["kind"] for s in plan["steps"]],
                         ["screen", "research"])
        self.assertEqual(plan["steps"][1]["consumes"], [0])


class TaskAgentReferenceTests(unittest.TestCase):
    """The task path resolves "that folder" from the conversation."""

    def setUp(self):
        clear_history()
        self._tmp = tempfile.TemporaryDirectory()
        self._folder = os.path.join(self._tmp.name, "information")
        os.makedirs(self._folder, exist_ok=True)

    def tearDown(self):
        clear_history()
        self._tmp.cleanup()

    def _seed(self):
        add_message(
            "assistant",
            "[task] name it information — Done, sir. Folder ready: %s."
            % self._folder)

    def test_that_folder_resolves_to_the_created_path(self):
        self._seed()
        resolved = task_agent._resolve_folder_references(
            "create a txt file by the name notes in that folder and write "
            "hello")
        self.assertIn(self._folder, resolved)
        self.assertNotIn("that folder", resolved)

    def test_no_history_leaves_the_command_untouched(self):
        resolved = task_agent._resolve_folder_references(
            "create a file in that folder")
        self.assertEqual(resolved, "create a file in that folder")

    def test_gather_context_carries_history(self):
        self._seed()
        add_message("user", "hello jarvis")
        context = task_agent.gather_context()
        self.assertTrue(context.get("history"))
        self.assertEqual(context["history"][-1]["content"], "hello jarvis")

    def test_planner_prompt_carries_the_recent_conversation(self):
        self._seed()
        prompt = task_agent._build_planner_prompt(
            "create a file in that folder",
            {"history": [
                {"role": "assistant",
                 "content": "[task] Folder ready: %s." % self._folder}]})
        self.assertIn("RECENT CONVERSATION", prompt)
        self.assertIn(self._folder, prompt)


if __name__ == "__main__":
    unittest.main()
