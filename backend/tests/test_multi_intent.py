import os
import tempfile
import unittest
from unittest.mock import patch

from backend.core import brain
from backend.core import entity_ledger
from backend.services import multi_intent
from backend.services.task_agent import agent as task_agent

HEADLINE = (
    "jarvis, look at my screen at this video, and do a deep research "
    "about it on internet, and whatever you find create a txt file on "
    "desktop and write your report in it"
)


class ChainSplitTests(unittest.TestCase):
    """Rank 2: one utterance, many jobs — split by meaning, chain the rest."""

    def test_headline_becomes_screen_research_task(self):
        plan = multi_intent.build_chain(HEADLINE)
        self.assertIsNotNone(plan)
        self.assertEqual([s["kind"] for s in plan["steps"]],
                         ["screen", "research", "task"])
        self.assertEqual([s["consumes"] for s in plan["steps"]],
                         [[], [0], [1]])

    def test_headline_without_commas_also_chains(self):
        text = ("look at my screen at this video and do a deep research "
                "about it on internet and whatever you find create a txt "
                "file on desktop and write your report in it")
        plan = multi_intent.build_chain(text)
        self.assertIsNotNone(plan)
        self.assertEqual([s["kind"] for s in plan["steps"]],
                         ["screen", "research", "task"])

    def test_fragment_merges_into_single_screen_job(self):
        self.assertIsNone(multi_intent.build_chain(
            "look at my screen and tell me what you see"))

    def test_same_kind_clauses_merge_to_single_task(self):
        text = ("create a folder named history and inside that folder "
                "create a file hello.txt and write hello in it")
        self.assertIsNone(multi_intent.build_chain(text))

    def test_tool_clause_blocks_the_chain(self):
        self.assertIsNone(multi_intent.build_chain(
            "open youtube and research the news"))

    def test_task_before_the_end_blocks_the_chain(self):
        text = ("create a file called x.txt with hello and research "
                "the news")
        self.assertIsNone(multi_intent.build_chain(text))

    def test_research_then_write_chain_with_anaphora(self):
        text = ("research the new iphone and create a file called "
                "iphone.txt with whatever you find")
        plan = multi_intent.build_chain(text)
        self.assertIsNotNone(plan)
        self.assertEqual([s["kind"] for s in plan["steps"]],
                         ["research", "task"])
        self.assertEqual(plan["steps"][1]["consumes"], [0])

    def test_save_it_on_desktop_consumes_research(self):
        text = "research the Zephyr 900 drone and save it on desktop"
        plan = multi_intent.build_chain(text)
        self.assertIsNotNone(plan)
        self.assertEqual([s["kind"] for s in plan["steps"]],
                         ["research", "task"])
        self.assertEqual(plan["steps"][1]["consumes"], [0])

    def test_future_talk_is_not_a_chain(self):
        self.assertIsNone(multi_intent.build_chain(
            "i will look at my screen later and then research space"))

    def test_ack_is_one_narrative(self):
        ack = multi_intent.render_ack(multi_intent.build_chain(HEADLINE))
        self.assertIn("3 steps", ack)
        self.assertIn("ask", ack.lower())
        self.assertIn("screen", ack.lower())

    def test_ack_says_folder_for_a_folder_task(self):
        plan = multi_intent.build_chain(
            "research the strongest current ai model and then create a "
            "folder on my desktop by that model name which you found")
        self.assertIsNotNone(plan)
        ack = multi_intent.render_ack(plan)
        self.assertIn("folder", ack.lower())
        self.assertNotIn("file", ack.lower())
        self.assertIn("creating it", ack.lower())


class ChainExecutorTests(unittest.TestCase):
    """Rank 2: the worker runs the steps and arms ONE file-write approval."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        brain._mi_chain_active = False
        brain._pending_confirmation = None
        brain._pending_opencode_task = None
        brain._pending_browser_clarification = None
        brain._held_redirect = None
        brain._proactive_research_fired = False
        task_agent._pending_task_action = None
        entity_ledger.reset()
        self.addCleanup(entity_ledger.reset)
        self.messages = []
        self._orig_cb = brain._async_reply_callback

        def capture(text, spoken=None):
            self.messages.append(text)

        brain.set_async_reply_callback(capture)
        self.addCleanup(brain.set_async_reply_callback, self._orig_cb)

    def _run_chain(self, text):
        plan = multi_intent.build_chain(text)
        self.assertIsNotNone(plan)
        brain._run_multi_intent_chain(text, plan)
        return plan

    def test_screen_research_write_chain_arms_one_file_write(self):
        captured = {}

        def fake_research(query, pinned_overview=None, **kwargs):
            captured["query"] = query
            return {
                "query": query,
                "spoken_summary": "Interstellar is a 2014 science fiction film.",
                "detailed_markdown": "# Interstellar\n\nA 2014 film.",
                "related_videos": [],
                "report_path": "r.md",
                "visited_count": 3,
                "failed_count": 0,
            }

        def fake_arm(plan, context, task_text=""):
            captured["plan"] = plan
            return object()

        with patch.object(brain, "analyze_screen", return_value={
                "tip": "This is the movie 'Interstellar'.",
                "topic": "Interstellar movie"}), \
             patch.object(brain, "run_research", side_effect=fake_research), \
             patch.object(brain, "fetch_ai_overview_text",
                          return_value="overview"), \
             patch.object(brain, "push_research_result"), \
             patch.object(task_agent, "_known_folders",
                          return_value={"desktop": self._tmp.name}), \
             patch.object(task_agent, "_arm_plan_confirmation",
                          side_effect=fake_arm):
            self._run_chain(HEADLINE)

        self.assertEqual(captured.get("query"), "Interstellar movie")
        step = captured["plan"]["steps"][0]
        self.assertEqual(step["tool"], "code.write_file")
        self.assertTrue(step["args"]["create_only"])
        self.assertEqual(step["args"]["path"],
                         os.path.join(self._tmp.name, "jarvis_report.txt"))
        self.assertIn("Interstellar", step["args"]["content"])
        self.assertTrue(self.messages)
        self.assertIn("confirm task", self.messages[-1].lower())

    def test_folder_chain_arms_folder_creation(self):
        text = ("research the Zephyr 900 drone and create a folder called "
                "zephyr_files on my desktop")
        captured = {}

        def fake_quick(query):
            return {"query": query, "spoken_summary": "The Zephyr 900."}

        def fake_arm(plan, context, task_text=""):
            captured["plan"] = plan
            return object()

        with patch.object(brain, "run_quick_search", side_effect=fake_quick), \
             patch.object(task_agent, "_known_folders",
                          return_value={"desktop": self._tmp.name}), \
             patch.object(task_agent, "_arm_plan_confirmation",
                          side_effect=fake_arm):
            self._run_chain(text)
        step = captured["plan"]["steps"][0]
        self.assertEqual(step["tool"], "code.create_folder")
        self.assertTrue(step["args"]["path"].endswith("zephyr_files"))

    def test_unnamed_folder_chain_asks_instead_of_writing_a_file(self):
        text = ("research the strongest current ai model and then create a "
                "folder on my desktop by that model name which you found")

        def fake_quick(query):
            return {"query": query, "spoken_summary": "The strongest is X."}

        with patch.object(brain, "run_quick_search", side_effect=fake_quick), \
             patch.object(task_agent, "_known_folders",
                          return_value={"desktop": self._tmp.name}), \
             patch.object(task_agent, "_arm_plan_confirmation") as arm:
            self._run_chain(text)
        arm.assert_not_called()
        self.assertTrue(self.messages)
        self.assertIn("name the folder", self.messages[-1].lower())

    def test_screen_failure_skips_dependent_steps_and_arms_nothing(self):
        with patch.object(brain, "analyze_screen", return_value={
                "tip": "I couldn't analyse the screen, sir."}), \
             patch.object(brain, "run_research") as research, \
             patch.object(brain, "run_quick_search") as quick, \
             patch.object(task_agent, "_arm_plan_confirmation") as arm:
            self._run_chain(HEADLINE)
        research.assert_not_called()
        quick.assert_not_called()
        arm.assert_not_called()
        self.assertTrue(self.messages)
        self.assertIn("skipped", self.messages[-1].lower())
        self.assertIn("screen", self.messages[-1].lower())

    def test_quick_chain_uses_named_file_and_research_content(self):
        text = ("research the Zephyr 900 drone and create a file called "
                "zephyr.txt with whatever you find")
        captured = {}

        def fake_quick(query):
            return {"query": query,
                    "spoken_summary": "The Zephyr 900 costs $999."}

        def fake_arm(plan, context, task_text=""):
            captured["plan"] = plan
            return object()

        with patch.object(brain, "run_quick_search", side_effect=fake_quick), \
             patch.object(task_agent, "_known_folders",
                          return_value={"desktop": self._tmp.name}), \
             patch.object(task_agent, "_arm_plan_confirmation",
                          side_effect=fake_arm):
            self._run_chain(text)
        step = captured["plan"]["steps"][0]
        self.assertTrue(step["args"]["path"].endswith("zephyr.txt"))
        self.assertIn("999", step["args"]["content"])

    def test_pending_approval_blocks_arming(self):
        text = ("research the Zephyr 900 drone and create a file called "
                "zephyr.txt with whatever you find")
        with patch.object(brain, "run_quick_search", return_value={
                "query": "q", "spoken_summary": "s"}), \
             patch.object(task_agent, "has_pending_task_confirmation",
                          return_value=True), \
             patch.object(task_agent, "_arm_plan_confirmation") as arm:
            self._run_chain(text)
        arm.assert_not_called()
        self.assertIn("already waiting", self.messages[-1])

    def test_concurrent_chain_is_refused(self):
        brain._mi_chain_active = True
        try:
            reply = brain.handle_multi_intent(
                "x", {"steps": [{"kind": "screen"}, {"kind": "research"}]})
        finally:
            brain._mi_chain_active = False
        self.assertIn("already running", reply)


class ChainGateTests(unittest.TestCase):
    """Rank 2: the brain gate routes a compound turn to the chain handler."""

    def setUp(self):
        brain._mi_chain_active = False
        brain._pending_confirmation = None
        brain._pending_opencode_task = None
        brain._pending_browser_clarification = None
        brain._held_redirect = None
        brain._proactive_research_fired = False
        task_agent._pending_task_action = None

    def test_gate_routes_compound_to_chain_ack(self):
        with patch.object(brain, "handle_multi_intent",
                          return_value="ACK") as handler, \
             patch.object(brain, "classify_intent") as classifier:
            response = brain.process_message(HEADLINE, sync_voice=False)
        self.assertEqual(response, "ACK")
        handler.assert_called_once()
        classifier.assert_not_called()

    def test_gate_leaves_single_jobs_alone(self):
        with patch.object(brain, "handle_multi_intent") as handler, \
             patch.object(brain, "classify_intent",
                          return_value={"intent": "chat", "reply": "hi",
                                        "_source": "test"}), \
             patch.object(brain, "handle_chat", return_value="hi"):
            brain.process_message("look at my screen and tell me what you see",
                                  sync_voice=False)
        handler.assert_not_called()


if __name__ == "__main__":
    unittest.main()
