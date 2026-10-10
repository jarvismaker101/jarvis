import os
import shutil
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
LIVE_FOLDER_FILE = (
    "look at my screen and find out about this website im currently on , "
    "tell me what are its benefits and then create a folder on my desktop "
    "byt the name of this website and inside that folder create a txt file "
    "by the name info and inside that txt file write hello from jarvis"
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

    def test_open_ai_is_a_company_not_a_tool_verb(self):
        # Live transcript: "…talking about anthropic and open ai, find out
        # about it on internet" — the splitter read "open" as a browser
        # verb and rejected the chain, so the message became a literal
        # research of the whole sentence.
        text = ("jarvis what is this video on my screen talking about "
                "anthropic and open ai , find out about it on internet")
        plan = multi_intent.build_chain(text)
        self.assertIsNotNone(plan)
        self.assertEqual([s["kind"] for s in plan["steps"]],
                         ["screen", "research"])
        self.assertIsNone(multi_intent.classify_clause("open ai"))
        self.assertIsNone(multi_intent.classify_clause("openai"))
        self.assertEqual(multi_intent.classify_clause("open youtube"),
                         "tool")
        self.assertEqual(multi_intent.classify_clause("visit openai.com"),
                         "tool")

    def test_screen_deictic_research_splits_into_screen_then_research(self):
        # Live transcript: "research about this verse on my screen from
        # bhagwat gita and tell me what does it actually say" — one clause
        # that both points at the screen and asks to research it. It was
        # typed into the web verbatim; it must look first, then research.
        text = ("research about this verse on my screen from bhagwat gita "
                "and tell me what does it actually say")
        plan = multi_intent.build_chain(text)
        self.assertIsNotNone(plan)
        self.assertEqual([s["kind"] for s in plan["steps"]],
                         ["screen", "research"])
        self.assertEqual([s["consumes"] for s in plan["steps"]],
                         [[], [0]])
        # A screen-only single job still never becomes a chain.
        self.assertIsNone(multi_intent.build_chain(
            "look at my screen to see what verse im talking about"))

    def test_screen_structure_replication_chains_screen_then_task(self):
        # LIVE FIX 12, live log: this exact sentence was answered "Which
        # folder should I check, sir? Please say the folder name." — the
        # replicate half was no task kind, the chain died, and the folder
        # inspect net asked the user to name a folder visible on screen.
        text = ("look at my screen there is a project folder structure "
                "visible i want you to see it and replicate it exactly on "
                "my desktop")
        plan = multi_intent.build_chain(text)
        self.assertIsNotNone(plan)
        self.assertEqual([s["kind"] for s in plan["steps"]],
                         ["screen", "task"])
        self.assertEqual(plan["steps"][1]["consumes"], [0])

    def test_screen_structure_replication_without_a_conjunction(self):
        # Run-on variant: the replicate fragment has no conjunction to
        # split on, so the screen+task expansion has to split them.
        text = ("look at my screen there is a project folder structure "
                "replicate it on my desktop")
        plan = multi_intent.build_chain(text)
        self.assertIsNotNone(plan)
        self.assertEqual([s["kind"] for s in plan["steps"]],
                         ["screen", "task"])
        self.assertEqual(plan["steps"][1]["consumes"], [0])

    def test_screen_and_replicate_the_structure_chains(self):
        text = ("look at my screen and replicate the folder structure on "
                "my desktop")
        plan = multi_intent.build_chain(text)
        self.assertIsNotNone(plan)
        self.assertEqual([s["kind"] for s in plan["steps"]],
                         ["screen", "task"])

    def test_replication_alone_is_a_single_job_not_a_chain(self):
        self.assertIsNone(multi_intent.build_chain(
            "replicate the folder structure on my desktop"))

    def test_replica_noun_form_chains_screen_then_task(self):
        # Live log: "there is a project structure visible at my screen ,
        # create a exact replica of this on my desktop" — the NOUN form
        # ("create a exact replica") matched no replicate verb, and "at my
        # screen" matched no screen clause, so the chain died, the
        # classifier collapsed the sentence into one rewritten task and the
        # confirmed handoff executed THAT description — in the browser.
        text = ("there is a project structure visible at my screen , "
                "create a exact replica of this on my desktop")
        plan = multi_intent.build_chain(text)
        self.assertIsNotNone(plan)
        self.assertEqual([s["kind"] for s in plan["steps"]],
                         ["screen", "task"])
        self.assertEqual(plan["steps"][1]["consumes"], [0])

    def test_visible_at_my_screen_is_a_screen_clause(self):
        self.assertEqual(
            multi_intent.classify_clause(
                "there is a project structure visible at my screen"),
            "screen")

    def test_ack_describes_structure_replication_not_a_file(self):
        text = ("look at my screen there is a project folder structure "
                "visible i want you to see it and replicate it exactly on "
                "my desktop")
        ack = multi_intent.render_ack(multi_intent.build_chain(text))
        self.assertIn("structure", ack.lower())
        self.assertNotIn("file", ack.lower())
        self.assertIn("creating it", ack.lower())

    def test_screen_research_single_clause_without_a_tell_me_fragment(self):
        # Live transcript: "research about these image generation models on
        # my screen" — a ONE-clause message. The chain gate used to require
        # at least two raw clauses, so this fell to the search net and the
        # phrase was searched near-verbatim.
        text = "research about these image generation models on my screen"
        plan = multi_intent.build_chain(text)
        self.assertIsNotNone(plan)
        self.assertEqual([s["kind"] for s in plan["steps"]],
                         ["screen", "research"])
        self.assertEqual([s["consumes"] for s in plan["steps"]],
                         [[], [0]])

    def test_screen_typo_still_splits_the_chain(self):
        # Live transcript: "on my scrren" (STT typo) dropped the screen
        # clause, so the whole sentence went to the search net and was
        # searched near-verbatim. The screen word tolerates its typos.
        text = ("what is this secret message short video on my scrren , "
                "research about it i want to know the secret message")
        plan = multi_intent.build_chain(text)
        self.assertIsNotNone(plan)
        self.assertEqual([s["kind"] for s in plan["steps"]],
                         ["screen", "research"])
        self.assertEqual(plan["steps"][1]["consumes"], [0])

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

    def test_ack_mentions_the_file_inside_the_folder(self):
        plan = multi_intent.build_chain(LIVE_FOLDER_FILE)
        self.assertIsNotNone(plan)
        ack = multi_intent.render_ack(plan)
        self.assertIn("with a file inside", ack.lower())


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
        brain._pending_folder_name = {"text": "", "at": 0.0}
        brain._pending_file_name = {"text": "", "at": 0.0}
        task_agent._pending_task_action = None
        entity_ledger.reset()
        self.addCleanup(entity_ledger.reset)
        self.addCleanup(setattr, brain, "_pending_confirmation", None)
        self.addCleanup(setattr, brain, "_pending_screen_clarify", None)
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
             patch.object(brain, "_extract_name_from_findings",
                          return_value=""), \
             patch.object(task_agent, "_known_folders",
                          return_value={"desktop": self._tmp.name}), \
             patch.object(task_agent, "_arm_plan_confirmation") as arm:
            self._run_chain(text)
        arm.assert_not_called()
        self.assertTrue(self.messages)
        self.assertIn("name the folder", self.messages[-1].lower())
        self.assertTrue(brain._pending_folder_name.get("text"))

    def test_folder_chain_names_the_folder_from_the_findings(self):
        text = ("search on the internet about the creator of monalisa and "
                "whichever name you find create a folder by that name on my "
                "desktop")
        captured = {}

        def fake_quick(query):
            return {"query": query,
                    "spoken_summary": ("The Mona Lisa was painted by "
                                       "Leonardo da Vinci.")}

        def fake_arm(plan, context, task_text=""):
            captured["plan"] = plan
            return object()

        plan = multi_intent.build_chain(text)
        self.assertIsNotNone(plan)
        self.assertEqual([s["kind"] for s in plan["steps"]],
                         ["research", "task"])
        self.assertEqual(plan["steps"][1]["consumes"], [0])
        with patch.object(brain, "run_quick_search", side_effect=fake_quick), \
             patch.object(brain, "_extract_name_from_findings",
                          return_value="Leonardo da Vinci"), \
             patch.object(brain, "classify_intent",
                          return_value={"intent": "research",
                                        "query": "creator of Mona Lisa"}), \
             patch.object(task_agent, "_known_folders",
                          return_value={"desktop": self._tmp.name}), \
             patch.object(task_agent, "_arm_plan_confirmation",
                          side_effect=fake_arm):
            brain._run_multi_intent_chain(text, plan)
        step = captured["plan"]["steps"][0]
        self.assertEqual(step["tool"], "code.create_folder")
        self.assertTrue(step["args"]["path"].endswith("Leonardo da Vinci"))

    def test_folder_chain_with_a_file_inside_arms_both_steps(self):
        captured = {}

        def fake_quick(query):
            return {"query": query,
                    "spoken_summary": "Assesly is an AI assessment platform."}

        def fake_arm(plan, context, task_text=""):
            captured["plan"] = plan
            return object()

        plan = multi_intent.build_chain(LIVE_FOLDER_FILE)
        self.assertIsNotNone(plan)
        self.assertEqual([s["kind"] for s in plan["steps"]],
                         ["screen", "research", "task"])
        with patch.object(brain, "analyze_screen", return_value={
                "tip": "An AI assessment dashboard.",
                "topic": "AI assessment platform dashboard",
                "creator": ""}), \
             patch.object(brain, "run_quick_search", side_effect=fake_quick), \
             patch.object(brain, "_extract_name_from_findings",
                          return_value="Assesly"), \
             patch.object(task_agent, "_known_folders",
                          return_value={"desktop": self._tmp.name}), \
             patch.object(task_agent, "_arm_plan_confirmation",
                          side_effect=fake_arm):
            brain._run_multi_intent_chain(LIVE_FOLDER_FILE, plan)
        steps = captured["plan"]["steps"]
        self.assertEqual([s["tool"] for s in steps],
                         ["code.create_folder", "code.write_file"])
        self.assertTrue(steps[0]["args"]["path"].endswith("Assesly"))
        self.assertTrue(steps[1]["args"]["path"].endswith("info.txt"))
        self.assertIn("hello from jarvis", steps[1]["args"]["content"])
        self.assertIn("info.txt", captured["plan"]["summary"])

    def test_pending_folder_name_answer_arms_the_folder(self):
        captured = {}

        def fake_arm(plan, context, task_text=""):
            captured["plan"] = plan
            return object()

        brain._set_pending_folder_name("create a folder by that name")
        with patch.object(task_agent, "_known_folders",
                          return_value={"desktop": self._tmp.name}), \
             patch.object(task_agent, "_arm_plan_confirmation",
                          side_effect=fake_arm):
            reply = brain.consume_pending_folder_name("name it james dark")
        self.assertIsNotNone(reply)
        self.assertIn("confirm", reply.lower())
        step = captured["plan"]["steps"][0]
        self.assertEqual(step["tool"], "code.create_folder")
        self.assertTrue(step["args"]["path"].endswith("james dark"))
        self.assertFalse(brain._pending_folder_name.get("text"))

    def test_name_it_answer_never_falls_into_chat(self):
        brain._set_pending_folder_name("create a folder by that name")
        with patch.object(task_agent, "_known_folders",
                          return_value={"desktop": self._tmp.name}), \
             patch.object(task_agent, "_arm_plan_confirmation",
                          return_value=object()), \
             patch.object(brain, "classify_intent") as classifier:
            reply = brain.process_message("name it james dark",
                                          sync_voice=False)
        self.assertIn("confirm", reply.lower())
        classifier.assert_not_called()

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

    def test_screen_deictic_research_chain_searches_what_was_seen(self):
        text = ("research about this verse on my screen from bhagwat gita "
                "and tell me what does it actually say")
        searches = []

        def fake_quick(query, *args, **kwargs):
            searches.append(query)
            return {"query": query,
                    "spoken_summary": "The verse speaks of knowledge."}

        with patch.object(brain, "analyze_screen", return_value={
                "tip": "A YouTube Short about Bhagavad Gita Chapter 4 "
                       "verse 5.",
                "topic": "Bhagavad Gita Chapter 4 verse 5",
                "creator": ""}), \
             patch.object(brain, "run_quick_search", side_effect=fake_quick):
            self._run_chain(text)
        self.assertEqual(searches, ["Bhagavad Gita Chapter 4 verse 5"])

    SECRET = ("what is this secret message they are talking about on my "
              "screen research on the internet about it and tell me")

    def _fake_quick(self, searches):
        def fake(query, *args, **kwargs):
            searches.append(query)
            return {"query": query, "spoken_summary": "summary"}
        return fake

    def test_generic_screen_topic_asks_before_searching(self):
        # Live transcript: the screen report's topic was the generic
        # "YouTube live chat message" and it was researched as-is — the
        # generic guess must become ONE candidate question instead.
        searches = []
        with patch.object(brain, "analyze_screen", return_value={
                "tip": "A YouTube live chat discussing a hidden promo code.",
                "topic": "YouTube live chat message", "creator": ""}), \
             patch.object(brain, "_mi_extract_screen_query",
                          return_value=("YouTube live chat hidden promo "
                                        "code", False)), \
             patch.object(brain, "run_quick_search",
                          side_effect=self._fake_quick(searches)):
            self._run_chain(self.SECRET)
        self.assertEqual(searches, [])
        pending = brain._pending_confirmation or {}
        self.assertEqual(pending.get("query"),
                         "YouTube live chat hidden promo code")
        self.assertIn("is this what you mean", self.messages[-1])

    def test_generic_screen_topic_searches_the_extracted_subject(self):
        searches = []
        with patch.object(brain, "analyze_screen", return_value={
                "tip": "A YouTube live chat discussing a hidden promo code.",
                "topic": "YouTube live chat message", "creator": ""}), \
             patch.object(brain, "_mi_extract_screen_query",
                          return_value=("YouTube live chat hidden promo "
                                        "code", True)), \
             patch.object(brain, "run_quick_search",
                          side_effect=self._fake_quick(searches)):
            self._run_chain(self.SECRET)
        self.assertEqual(searches, ["YouTube live chat hidden promo code"])

    def test_generic_screen_topic_asks_once_when_not_extractable(self):
        searches = []
        with patch.object(brain, "analyze_screen", return_value={
                "tip": "A YouTube live chat discussing a hidden promo code.",
                "topic": "YouTube live chat message", "creator": ""}), \
             patch.object(brain, "_mi_extract_screen_query",
                          return_value=("", False)), \
             patch.object(brain, "run_quick_search",
                          side_effect=self._fake_quick(searches)):
            self._run_chain(self.SECRET)
        self.assertEqual(searches, [])
        self.assertIn("couldn't tell exactly", self.messages[-1])
        self.assertIsNotNone(brain._get_pending_screen_clarify())

    def test_vision_call_identifies_the_screen_subject(self):
        # The focused vision call is the PRIMARY resolver: the screen step
        # is analysed, then the vision model is asked what exact thing the
        # user means. The chat-side extractor (which guessed "QTI The
        # Secret Betr" live) must not be consulted when vision answers.
        searches = []
        reports = [
            {"tip": "A YouTube Shorts video is playing.",
             "topic": "YouTube live chat message", "creator": ""},
            {"tip": "Karna Vs Arjun Ko Secret Message",
             "topic": "Karna Vs Arjun", "creator": ""},
        ]
        with patch.object(brain, "analyze_screen",
                          side_effect=reports), \
             patch.object(brain, "_mi_extract_screen_query",
                          side_effect=AssertionError("chat fallback used")), \
             patch.object(brain, "run_quick_search",
                          side_effect=self._fake_quick(searches)):
            self._run_chain(self.SECRET)
        self.assertEqual(searches, ["Karna Vs Arjun Ko Secret Message"])

    def test_vision_unavailable_falls_back_to_the_chat_extractor(self):
        searches = []
        with patch.object(brain, "analyze_screen", return_value={
                "tip": "A YouTube live chat discussing a hidden promo code.",
                "topic": "YouTube live chat message", "creator": ""}), \
             patch.object(brain, "_mi_extract_screen_query",
                          return_value=("YouTube live chat hidden promo "
                                        "code", True)), \
             patch.object(brain, "run_quick_search",
                          side_effect=self._fake_quick(searches)):
            self._run_chain(self.SECRET)
        self.assertEqual(searches, ["YouTube live chat hidden promo code"])

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


class FolderReferenceTests(unittest.TestCase):
    """Live fix: after the chain creates a folder, "that folder" resolves."""

    def setUp(self):
        brain._notebook_entities.clear()
        self.addCleanup(brain._notebook_entities.clear)

    def _seed_folder(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        brain.notebook_record_entity("assesly", tmp, kind="folder",
                                     source="created")
        return tmp

    def test_that_folder_resolves_to_the_chain_created_folder(self):
        tmp = self._seed_folder()
        hint = task_agent._located_write_folder(
            "create a txt file inside that folder and write hello from "
            "jarvis")
        self.assertEqual(task_agent._resolve_folder_hint(hint), tmp)

    def test_specified_folder_also_resolves(self):
        tmp = self._seed_folder()
        hint = task_agent._located_write_folder(
            "create a text file inside the specified folder and write "
            "hello from jarvis")
        self.assertEqual(task_agent._resolve_folder_hint(hint), tmp)


class NameExtractionTests(unittest.TestCase):
    """Live fix: the folder name comes from the findings, or not at all."""

    def test_parses_a_name_from_findings(self):
        reply = {"choices": [{"message": {
            "content": '{"name": "Leonardo da Vinci"}'}}]}
        with patch.object(brain, "_ask_chat_nonstream", return_value=reply):
            self.assertEqual(
                brain._extract_name_from_findings(
                    "The Mona Lisa was painted by Leonardo da Vinci "
                    "around 1503 in Florence."),
                "Leonardo da Vinci")

    def test_refuses_meta_or_overlong_names(self):
        for payload in ('{"name": ""}',
                        '{"name": "the name previously searched"}',
                        '{"name": "one two three four five six words"}'):
            reply = {"choices": [{"message": {"content": payload}}]}
            with patch.object(brain, "_ask_chat_nonstream",
                              return_value=reply):
                self.assertEqual(
                    brain._extract_name_from_findings(
                        "long findings text about a creator of something "
                        "clearly identified here"), "")


if __name__ == "__main__":
    unittest.main()
