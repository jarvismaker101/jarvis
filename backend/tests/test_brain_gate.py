import time
import unittest
from unittest.mock import patch

from backend import config
from backend.core import brain
from backend.services import browser_agent
from backend.services.task_agent import agent as task_agent


class OpencodeHandoffGateTests(unittest.TestCase):
    """The opencode handoff must never execute before the user confirms."""

    def setUp(self):
        brain._pending_opencode_task = None
        brain._pending_confirmation = None
        brain._proactive_research_fired = False

    def tearDown(self):
        brain._pending_opencode_task = None
        brain._pending_confirmation = None

    def _wait_until(self, predicate, timeout=2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.05)
        return predicate()

    def test_handle_opencode_task_arms_and_does_not_execute(self):
        with patch.object(brain, "run_opencode_task") as fake:
            response = brain.handle_opencode_task(
                "create a folder named badmoss",
                original_message="create a folder named badmoss",
            )
        fake.assert_not_called()
        self.assertIn("this is what i understood", response.lower())
        self.assertIn("go ahead and execute it", response.lower())
        self.assertIsNotNone(brain._pending_opencode_task)

    def test_voice_compact_long_task_keeps_suffix_within_180(self):
        task = "this task description is deliberately verbose " * 15
        self.assertGreater(len(task), 200)
        response = brain.handle_opencode_task(
            task, original_message="m", from_voice=True, voice_compact=True
        )
        self.assertLessEqual(len(response), 180)
        self.assertTrue(
            response.endswith("Do you want me to go ahead and execute it?")
        )
        self.assertIn("...", response)
        self.assertNotIn(task, response)

    def test_voice_compact_short_task_returns_full_question(self):
        response = brain.handle_opencode_task(
            "create folder x",
            original_message="m",
            from_voice=True,
            voice_compact=True,
        )
        self.assertIn("create folder x", response)
        self.assertGreater(len(response), 30)
        self.assertTrue(
            response.endswith("Do you want me to go ahead and execute it?")
        )

    def test_non_voice_compact_long_task_not_truncated(self):
        task = "verbose task description " * 25
        response = brain.handle_opencode_task(task, original_message="m")
        self.assertIn(task, response)
        self.assertTrue(
            response.endswith("Do you want me to go ahead and execute it?")
        )

    def test_voice_clip_short_passthrough(self):
        short = "Done, sir."
        self.assertEqual(brain._voice_clip(short), short)

    def test_voice_clip_sentence_cut(self):
        # terminator at >=100 should cut cleanly without ellipsis
        long_text = ("word " * 30) + "Done. " + ("extra " * 50)
        # first sentence ends well beyond 100 chars
        clipped = brain._voice_clip(long_text, limit=180)
        self.assertTrue(clipped.endswith("."))
        self.assertNotIn("...", clipped)
        self.assertLessEqual(len(clipped), 180)

    def test_voice_clip_word_boundary_with_ellipsis(self):
        long_text = "word " * 100
        clipped = brain._voice_clip(long_text, limit=180)
        self.assertLessEqual(len(clipped), 183)
        self.assertTrue(clipped.endswith("..."))
        # should cut at word boundary, not mid-word: last word before ellipsis is intact
        before = clipped[:-3].rstrip()
        self.assertTrue(before.endswith("word"))

    def test_consume_yes_executes_deferred_task(self):
        # The patches must stay installed until the hand-off has actually run:
        # the hand-off is backgrounded (F50 runs it as a typed effect), so
        # leaving the patch context before the wait let the real engine run.
        with patch.object(config, "TASK_ENGINE", "opencode"), \
             patch.object(brain, "run_opencode_task", return_value="Created folder.") as fake, \
             patch.object(brain, "is_opencode_available", return_value=True):
            brain.handle_opencode_task("create folder x", original_message="m")
            response = brain._consume_opencode_confirmation("yes")
            self.assertEqual(response, "Handing the task to opencode, sir.")
            self.assertTrue(self._wait_until(lambda: fake.called, timeout=10.0))
        # F16: the frozen execution contract travels with the call, so the
        # command is still the first positional argument.
        self.assertEqual(fake.call_args.args[0], "create folder x")
        self.assertIsNotNone(fake.call_args.kwargs.get("contract"))
        self.assertIsNone(brain._pending_opencode_task)

    def test_consume_no_skips(self):
        with patch.object(brain, "run_opencode_task") as fake, \
             patch.object(brain, "is_opencode_available", return_value=True):
            brain.handle_opencode_task("create folder x", original_message="m")
            response = brain._consume_opencode_confirmation("cancel")
        time.sleep(0.2)
        fake.assert_not_called()
        self.assertEqual(response, "As you wish, sir. I will skip that.")

    def test_consume_not_sure_never_confirms(self):
        # "not sure" family resolves to NO (never executes) — same as the task gate.
        with patch.object(brain, "run_opencode_task") as fake:
            brain.handle_opencode_task("create folder x", original_message="m")
            response = brain._consume_opencode_confirmation("not sure")
        time.sleep(0.2)
        self.assertEqual(response, "As you wish, sir. I will skip that.")
        fake.assert_not_called()
        self.assertIsNone(brain._pending_opencode_task)

    def test_consume_no_pending_returns_none(self):
        self.assertIsNone(brain._consume_opencode_confirmation("yes"))

    def test_consume_expired_returns_none(self):
        brain.handle_opencode_task("create folder x", original_message="m")
        brain._pending_opencode_task["expires"] = time.time() - 1
        self.assertIsNone(brain._consume_opencode_confirmation("yes"))
        self.assertIsNone(brain._pending_opencode_task)

    def test_handle_tool_intent_empty_steps_gates_opencode(self):
        """No-local-steps tool intents must arm the gate, never run opencode."""
        with patch.object(brain, "run_opencode_task") as fake:
            response = brain.handle_tool_intent(
                [], original_message="create a folder named badmoss"
            )
        fake.assert_not_called()
        self.assertIn("this is what i understood", response.lower())
        self.assertIn("go ahead and execute it", response.lower())
        self.assertIsNotNone(brain._pending_opencode_task)

    def test_tool_intent_failed_local_execution_gates_opencode(self):
        """Failed local execution must arm the gate, never run opencode."""
        with patch.object(brain, "execute_multiple", return_value=None), \
             patch.object(brain, "is_opencode_available", return_value=True), \
             patch.object(brain, "run_opencode_task") as fake, \
             patch.object(brain, "_notify_async_reply") as notify:
            brain.handle_tool_intent(
                [{"action": "open", "target": "x"}],
                original_message="create a folder named badmoss",
            )
            # This gate arms from a background thread, and other tests in this
            # module leave their own notify threads running: assert on THIS
            # call appearing in the list, never on it being the last one.
            self.assertTrue(
                self._wait_until(
                    lambda: any(
                        "this is what i understood" in str(call[0][0]).lower()
                        for call in notify.call_args_list if call[0]
                    )),
                "the gated handoff question must be delivered: %r"
                % (notify.call_args_list,),
            )
        fake.assert_not_called()
        self.assertIsNotNone(brain._pending_opencode_task)

    def test_task_gate_arming_disarms_opencode_gate(self):
        brain.handle_opencode_task("create folder x", original_message="m")
        self.assertIsNotNone(brain._pending_opencode_task)
        context = {
            "windows": {"active_window": {"title": "x"}, "visible_controls": []},
            "editor": {"available": False},
            "browser": {"available": False, "tabs": []},
        }
        plan = task_agent.plan_task("run pip list", context)
        task_agent.execute_plan(plan, context)
        try:
            brain._disarm_other_gates_if_task_gate_armed()
            self.assertIsNone(brain._pending_opencode_task)
        finally:
            task_agent._pending_task_action = None

    def test_opencode_arming_disarms_research_gate(self):
        brain._pending_confirmation = {
            "message": "look up python",
            "expires": time.time() + 30,
        }
        brain.handle_opencode_task("create folder x", original_message="m")
        self.assertIsNone(brain._pending_confirmation)
        self.assertIsNotNone(brain._pending_opencode_task)

    def test_process_message_consumes_confirmation_before_routing(self):
        brain.handle_opencode_task("create folder x", original_message="m")
        with patch.object(config, "TASK_ENGINE", "opencode"), \
             patch.object(brain, "_consume_confirmation", return_value=None), \
             patch.object(brain, "consume_task_confirmation", return_value=None), \
             patch.object(brain, "classify_intent", side_effect=AssertionError("must not route")), \
             patch.object(brain, "run_opencode_task", return_value="ok"), \
             patch.object(brain, "is_opencode_available", return_value=True):
            response = brain.process_message("yes go ahead", sync_voice=False)
        self.assertEqual(response, "Handing the task to opencode, sir.")
        self.assertIsNone(brain._pending_opencode_task)


class BrowserClarificationTests(unittest.TestCase):
    def setUp(self):
        brain._pending_browser_clarification = None
        brain._pending_opencode_task = None
        brain._pending_confirmation = None

    def tearDown(self):
        brain._pending_browser_clarification = None
        brain._pending_opencode_task = None
        brain._pending_confirmation = None
        brain.set_opencode_task_running(False)

    def _wait_until(self, predicate, timeout=2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.02)
        return predicate()

    def test_browser_question_arms_pending(self):
        with patch.object(config, "TASK_ENGINE", "browser_agent"), \
             patch.object(brain, "run_browser_task", return_value="Which site should I open?"), \
             patch.object(brain, "_notify_async_reply"):
            brain._execute_deferred_opencode("open something", "open something")
            self.assertTrue(self._wait_until(lambda: brain._pending_browser_clarification is not None))
        pending = brain._pending_browser_clarification
        self.assertIsNotNone(pending)
        self.assertEqual(pending["task_description"], "open something")
        self.assertIn("Which site", pending["question"])

    def test_followup_continues_without_confirmation_gate(self):
        captured = {}

        def fake_run(desc, resume_from=None):
            captured["desc"] = desc
            captured["resume_from"] = resume_from
            return "Done."

        with patch.object(config, "TASK_ENGINE", "browser_agent"), \
             patch.object(brain, "run_browser_task", side_effect=fake_run), \
             patch.object(brain, "_notify_async_reply"):
            # arm by simulating a clarifying question
            brain._execute_deferred_opencode("find price on amazon", "find price on amazon")
            # F50: that first run is a backgrounded typed effect — wait for it
            # to finish before arming the follow-up, so the captured call is
            # the FOLLOW-UP's (the first run could otherwise land after the
            # captured.clear() below and mask the resume_from assertion).
            self.assertTrue(self._wait_until(lambda: "desc" in captured,
                                             timeout=10.0))
            # F08: a suspended run carries a CHECKPOINT of its verified
            # progress; the follow-up resumes it instead of re-running the
            # original description as a fresh task.
            checkpoint_id = browser_agent.new_checkpoint_id("find price on amazon")
            browser_agent.suspend_checkpoint(
                checkpoint_id, {}, question="Which product?")
            brain._pending_browser_clarification = {
                "task_description": "find price on amazon",
                "question": "Which product?",
                "checkpoint_id": checkpoint_id,
                "expires": time.time() + 90,
            }
            # now send follow-up via process_message path (consume_browser_followup)
            with patch.object(brain, "classify_intent", side_effect=AssertionError("should not classify")):
                # need to ensure run_browser_task for continuation is captured
                with patch.object(brain, "run_browser_task", side_effect=fake_run) as mock_run:
                    # clear previous captured
                    captured.clear()
                    try:
                        resp = brain.process_message("the one on amazon prime", sync_voice=False)
                        # process_message should return the deferred start phrase
                        self.assertEqual(resp, brain.BROWSER_AGENT_START_PHRASE)
                        # wait for background thread to call run_browser_task with augmented desc
                        self.assertTrue(self._wait_until(lambda: "desc" in captured))
                        self.assertEqual(captured["resume_from"], checkpoint_id,
                                         "the follow-up must RESUME the checkpoint")
                        self.assertNotIn("find price on amazon", captured["desc"],
                                         "the old description must not be replayed")
                        self.assertIn("User follow-up answering your last question",
                                      captured["desc"])
                        self.assertIn("the one on amazon prime", captured["desc"])
                        # no new confirmation gate should be armed
                        self.assertIsNone(brain._pending_opencode_task)
                        self.assertIsNone(brain._pending_browser_clarification)
                    finally:
                        browser_agent.drop_checkpoint(checkpoint_id)

    def test_followup_expiry_not_consumed(self):
        brain._pending_browser_clarification = {
            "task_description": "open something",
            "question": "Which site?",
            "expires": time.time() - 1,
        }
        # should return None and clear expired
        result = brain._consume_browser_followup("my answer")
        self.assertIsNone(result)
        self.assertIsNone(brain._pending_browser_clarification)

    def test_screen_targeted_msg_not_consumed(self):
        brain._pending_browser_clarification = {
            "task_description": "open something",
            "question": "Which site?",
            "expires": time.time() + 90,
        }
        # screen-targeted message must fall through, not be consumed
        result = brain._consume_browser_followup("just click at the video on my screen")
        self.assertIsNone(result)
        # pending should remain (not consumed)
        self.assertIsNotNone(brain._pending_browser_clarification)
        # via process_message, it should go to screen control, not browser followup
        with patch.object(brain, "maybe_handle_screen_control_message", return_value="Screen controls are off. Say turn on screen controls first.") as mock_screen:
            resp = brain.process_message("just click at the video on my screen", sync_voice=False)
            mock_screen.assert_called_once()
            self.assertIn("Screen controls", resp)
            # after screen handling, pending should be cleared per spec
            self.assertIsNone(brain._pending_browser_clarification)

    def test_browser_clarification_does_not_block_task_gate(self):
        # arm both: browser clarification + task gate? task gate takes precedence
        brain._pending_browser_clarification = {
            "task_description": "browse something",
            "question": "Which site?",
            "expires": time.time() + 90,
        }
        with patch.object(brain, "consume_task_confirmation", return_value="Handing off, sir.") as mock_task:
            # process_message checks task gate before browser followup
            with patch.object(brain, "_consume_opencode_confirmation", return_value=None), \
                 patch.object(brain, "_consume_confirmation", return_value=None):
                resp = brain.process_message("yes go ahead", sync_voice=False)
                mock_task.assert_called_once()
                self.assertEqual(resp, "Handing off, sir.")
                # browser pending should be cleared when new task confirmed
                self.assertIsNone(brain._pending_browser_clarification)

    def test_non_question_does_not_arm(self):
        with patch.object(config, "TASK_ENGINE", "browser_agent"), \
             patch.object(brain, "run_browser_task", return_value="Done: found price."), \
             patch.object(brain, "_notify_async_reply"):
            brain._execute_deferred_opencode("find price", "find price")
            time.sleep(0.2)
        self.assertIsNone(brain._pending_browser_clarification)


class FreshInfoAndWebRoutingTests(unittest.TestCase):
    """Round 15: stale-answer auto-search + web-task browser-agent net."""

    def setUp(self):
        brain._pending_opencode_task = None
        brain._pending_confirmation = None
        brain._pending_browser_clarification = None
        brain._proactive_research_fired = False
        task_agent._pending_task_action = None

    def tearDown(self):
        brain._pending_opencode_task = None
        brain._pending_confirmation = None
        brain._pending_browser_clarification = None
        task_agent._pending_task_action = None

    @staticmethod
    def _chat_verdict():
        return {"intent": "chat", "steps": [], "task_description": "", "query": ""}

    def test_should_search_matches_pricing(self):
        self.assertTrue(
            brain.should_search("whats the pricing for claude fable 5.1 model")
        )

    def test_should_search_ignores_substring_false_positives(self):
        # S1 — whole-word matching. These must stay plain chat, not research.
        for msg in (
            "what do you know about the roman empire",
            "how do you feel today",
            "i feel tired",
            "do you know me",
            "is my costume ready",
            "he scored a goal",
            "it matches the description",
        ):
            self.assertFalse(brain.should_search(msg), msg)

    def test_should_search_weak_recency_requires_fact_noun(self):
        self.assertFalse(brain.should_search("what do you know now"))
        self.assertFalse(brain.should_search("aaj kya karu"))
        self.assertTrue(brain.should_search("whats the news today"))
        self.assertTrue(brain.should_search("whats the score now"))
        self.assertTrue(brain.should_search("aaj ka mausam"))

    def test_should_search_strong_whole_words(self):
        for msg in (
            "whats the latest on the mars rover",
            "whats the weather in delhi",
            "how much does a ps5 cost",
            "tell me the score",
        ):
            self.assertTrue(brain.should_search(msg), msg)

    def test_fresh_info_reroutes_chat_to_research(self):
        msg = "whats the pricing for claude fable 5.1 model"
        with patch.object(brain, "classify_intent", return_value=self._chat_verdict()), \
             patch.object(brain, "maybe_handle_screen_control_message", return_value=None), \
             patch.object(brain, "handle_research_intent", return_value="researched") as research, \
             patch.object(brain, "_ChatRacer") as racer_cls:
            response = brain.process_message(
                msg, sync_voice=False, stream_reply=object()
            )
        self.assertEqual(response, "researched")
        research.assert_called_once()
        self.assertEqual(research.call_args[0][0], msg)
        self.assertTrue(research.call_args[1].get("derived"))
        racer_cls.return_value.cancel.assert_called_once()

    def test_greeting_guard_stays_chat(self):
        with patch.object(brain, "classify_intent", return_value=self._chat_verdict()), \
             patch.object(brain, "maybe_handle_screen_control_message", return_value=None), \
             patch.object(brain, "handle_research_intent") as research, \
             patch.object(brain, "handle_chat", return_value="hello") as chat:
            response = brain.process_message("how are you today", sync_voice=False)
        research.assert_not_called()
        chat.assert_called_once()
        self.assertEqual(response, "hello")

    def test_web_shaped_task_routes_to_browser_agent(self):
        msg = "Go to 1hd.to website, search for One Piece Movie Red, and play it"
        with patch.object(config, "TASK_ENGINE", "browser_agent"), \
             patch.object(brain, "classify_intent", return_value=self._chat_verdict()), \
             patch.object(brain, "maybe_handle_screen_control_message", return_value=None), \
             patch.object(brain, "handle_opencode_task", return_value="confirm?") as handoff, \
             patch.object(brain, "handle_task_message", return_value="task") as task_msg:
            response = brain.process_message(msg, sync_voice=False)
        handoff.assert_called_once_with(
            msg, original_message=msg, from_voice=False, voice_compact=False
        )
        task_msg.assert_not_called()
        self.assertEqual(response, "confirm?")

    def test_local_task_stays_on_task_message(self):
        msg = "create a folder called test in my workspace"
        with patch.object(config, "TASK_ENGINE", "browser_agent"), \
             patch.object(brain, "classify_intent", return_value=self._chat_verdict()), \
             patch.object(brain, "maybe_handle_screen_control_message", return_value=None), \
             patch.object(brain, "handle_opencode_task") as handoff, \
             patch.object(brain, "handle_task_message", return_value="task") as task_msg:
            response = brain.process_message(msg, sync_voice=False)
        task_msg.assert_called_once()
        handoff.assert_not_called()
        self.assertEqual(response, "task")


class ScreenQuestionNetTests(unittest.TestCase):
    """Round 16: deterministic screen-question net fires when the cloud
    classifier falls back to chat (throttle/timeout)."""

    def setUp(self):
        brain._pending_opencode_task = None
        brain._pending_confirmation = None
        brain._pending_browser_clarification = None
        brain._proactive_research_fired = False
        task_agent._pending_task_action = None

    def tearDown(self):
        brain._pending_opencode_task = None
        brain._pending_confirmation = None
        brain._pending_browser_clarification = None
        task_agent._pending_task_action = None

    @staticmethod
    def _chat_verdict():
        return {"intent": "chat", "steps": [], "task_description": "", "query": ""}

    @staticmethod
    def _screen_result():
        return {
            "tip": "You are looking at a code editor.",
            "evidence": [],
            "topic": "",
            "grounding_links": [],
            "show_images": False,
            "region": None,
        }

    def _run_screen(self, msg, verdict=None):
        verdict = verdict if verdict is not None else self._chat_verdict()
        with patch.object(brain, "classify_intent", return_value=verdict), \
             patch.object(brain, "maybe_handle_screen_control_message", return_value=None), \
             patch.object(brain, "analyze_screen", return_value=self._screen_result()) as analyze, \
             patch.object(brain, "push_screen_answer") as push:
            response = brain.process_message(msg, sync_voice=False)
        return response, analyze, push

    def test_exact_user_message_routes_via_net(self):
        response, analyze, push = self._run_screen("what's on my screen jarvis")
        analyze.assert_called_once()
        push.assert_called_once()
        self.assertEqual(response, "You are looking at a code editor.")

    def test_region_variant_routes_via_net(self):
        response, analyze, push = self._run_screen(
            "what does this highlighted area on my screen mean"
        )
        analyze.assert_called_once()
        self.assertEqual(response, "You are looking at a code editor.")

    def test_hinglish_variant_routes_via_net(self):
        response, analyze, push = self._run_screen("screen pe kya dikh raha hai")
        analyze.assert_called_once()
        self.assertEqual(response, "You are looking at a code editor.")

    def test_control_verb_stays_chat(self):
        with patch.object(brain, "classify_intent", return_value=self._chat_verdict()), \
             patch.object(brain, "maybe_handle_screen_control_message", return_value=None), \
             patch.object(brain, "is_task_request", return_value=False), \
             patch.object(brain, "analyze_screen") as analyze, \
             patch.object(brain, "handle_chat", return_value="chat reply") as chat:
            response = brain.process_message(
                "click on the play button on my screen", sync_voice=False
            )
        analyze.assert_not_called()
        chat.assert_called_once()
        self.assertEqual(response, "chat reply")

    def test_non_screen_chat_unaffected(self):
        with patch.object(brain, "classify_intent", return_value=self._chat_verdict()), \
             patch.object(brain, "maybe_handle_screen_control_message", return_value=None), \
             patch.object(brain, "is_task_request", return_value=False), \
             patch.object(brain, "analyze_screen") as analyze, \
             patch.object(brain, "handle_chat", return_value="hello sir") as chat:
            response = brain.process_message("hello jarvis", sync_voice=False)
        analyze.assert_not_called()
        chat.assert_called_once()
        self.assertEqual(response, "hello sir")

    def test_direct_screen_verdict_not_disturbed(self):
        verdict = {"intent": "screen", "steps": [], "task_description": "", "query": ""}
        response, analyze, push = self._run_screen(
            "what's on my screen jarvis", verdict=verdict
        )
        analyze.assert_called_once()
        push.assert_called_once()
        self.assertEqual(response, "You are looking at a code editor.")

    def test_research_misclassification_routes_via_net(self):
        verdict = {"intent": "research", "steps": [], "task_description": "", "query": "q"}
        response, analyze, push = self._run_screen(
            "what's on my screen jarvis", verdict=verdict
        )
        analyze.assert_called_once()
        self.assertEqual(response, "You are looking at a code editor.")

    def test_tool_verdict_steps_preserved(self):
        steps = [{"action": "open_website", "input": "site.com"}]
        verdict = {"intent": "tool", "steps": steps, "task_description": "", "query": ""}
        with patch.object(brain, "classify_intent", return_value=verdict), \
             patch.object(brain, "maybe_handle_screen_control_message", return_value=None), \
             patch.object(brain, "analyze_screen") as analyze, \
             patch.object(brain, "handle_tool_intent", return_value="opened") as tool:
            response = brain.process_message(
                "open the website showing on my screen", sync_voice=False
            )
        analyze.assert_not_called()
        tool.assert_called_once()
        self.assertEqual(tool.call_args[0][0], steps)
        self.assertEqual(response, "opened")


if __name__ == "__main__":
    unittest.main()
