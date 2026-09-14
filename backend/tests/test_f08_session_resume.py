"""F08 — resume the actual agent session, not a fresh replay.

Acceptance (audit report): "Clarification does not replay committed actions;
query-string URLs do not suspend; cancel does not restart; expanded scope
requires approval."

Baseline defects pinned here:
  * a clarifying question was detected with `"?" in text`, so an ANSWER
    containing a URL query string armed a bogus continuation;
  * the follow-up appended the answer to the description and started a NEW run
    (fresh history/marks/failure state), replaying committed actions;
  * a negative or a materially different follow-up was swallowed as an answer.
"""

import os
import unittest
from unittest.mock import patch

from backend.core import brain
from backend.services import browser_agent


class QuestionDetectionTests(unittest.TestCase):
    def test_real_questions_are_detected(self):
        for text in (
            "Which site should I open for the price check?",
            "Do you want the cheapest option or the fastest?",
            "Should I book the 6pm slot?",
            "What name should I use on the form? Please confirm.",
            "I need to know your preferred airport? ",
        ):
            self.assertTrue(browser_agent._is_clarifying_question(text), text)

    def test_query_string_urls_do_not_suspend(self):
        for text in (
            "Opened https://example.com/?q=price+check and read the table.",
            "The page www.shop.com/?ref=abc shows 799 dollars.",
            "Found it at https://x.test/search?q=1",
        ):
            self.assertFalse(browser_agent._is_clarifying_question(text), text)

    def test_plain_statements_are_not_questions(self):
        for text in ("Done.", "Created the file.", "", None,
                     "The price is 799 dollars for the base model."):
            self.assertFalse(browser_agent._is_clarifying_question(text), text)


class CheckpointTests(unittest.TestCase):
    def tearDown(self):
        browser_agent._SUSPENDED_CHECKPOINTS.clear()
        brain._pending_browser_clarification = None

    def _session(self):
        return {
            "marks": {"1": "Save button"},
            "failures": browser_agent.FailureTracker(),
            "tool_failures": ["look: timeout"],
            "completed_actions": ["click_locator selector=#pay"],
            "completed_effects": ['["click_locator", "{\\"selector\\": \\"#pay\\"}"]'],
            "url": "https://shop.test/checkout",
            "tab": "tab-3",
        }

    def test_checkpoint_round_trip_keeps_progress(self):
        cid = browser_agent.new_checkpoint_id("buy the thing")
        browser_agent.suspend_checkpoint(cid, self._session(),
                                         question="Which card?",
                                         identity={"url": "https://shop.test/checkout"})
        entry = browser_agent.peek_checkpoint(cid)
        self.assertEqual(entry["question"], "Which card?")
        self.assertEqual(entry["completed"], ["click_locator selector=#pay"])
        self.assertEqual(entry["identity"]["url"], "https://shop.test/checkout")
        self.assertEqual(entry["tool_failures"], ["look: timeout"])
        self.assertTrue(entry["completed_effects"])
        # resume pops it: a checkpoint can be resumed exactly once.
        self.assertIsNotNone(browser_agent.resume_checkpoint(cid))
        self.assertIsNone(browser_agent.resume_checkpoint(cid))

    def test_drop_is_final(self):
        cid = browser_agent.new_checkpoint_id("x")
        browser_agent.suspend_checkpoint(cid, self._session(), question="q")
        self.assertTrue(browser_agent.drop_checkpoint(cid))
        self.assertIsNone(browser_agent.resume_checkpoint(cid))
        self.assertFalse(browser_agent.drop_checkpoint(cid))

    def test_expired_checkpoints_are_pruned(self):
        cid = browser_agent.new_checkpoint_id("old")
        browser_agent.suspend_checkpoint(cid, self._session(), question="q")
        with patch.object(browser_agent, "_SUSPENDED_TTL", -1):
            browser_agent._prune_suspended()
        self.assertIsNone(browser_agent.peek_checkpoint(cid))

    def test_already_committed_effects_are_detected(self):
        # Only effects restored from a RESUMED checkpoint block a repeat.
        session = {"resumed_effects": set()}
        browser_agent._note_committed(session, "click_locator",
                                      {"selector": "#pay"})
        self.assertFalse(browser_agent._already_committed(
            session, "click_locator", {"selector": "#pay"}),
            "a live run may repeat its own action")
        session["resumed_effects"] = set(session["completed_effects"])
        self.assertTrue(browser_agent._already_committed(
            session, "click_locator", {"selector": "#pay"}))
        self.assertFalse(browser_agent._already_committed(
            session, "click_locator", {"selector": "#other"}))
        self.assertIn("click_locator", session["completed_actions"][0])

    def test_resumed_history_tells_the_model_not_to_repeat(self):
        cid = browser_agent.new_checkpoint_id("t")
        entry = browser_agent.suspend_checkpoint(
            cid, self._session(), question="Which card?")
        history = browser_agent._checkpoint_history(entry, "use the saved card")
        self.assertIn("RESUMING", history)
        self.assertIn("do NOT repeat", history)
        self.assertIn("click_locator selector=#pay", history)
        self.assertIn("Which card?", history)
        self.assertIn("use the saved card", history)


class BrainContinuationTests(unittest.TestCase):
    def tearDown(self):
        browser_agent._SUSPENDED_CHECKPOINTS.clear()
        brain._pending_browser_clarification = None

    def _arm(self, checkpoint_id="cp-1"):
        browser_agent.suspend_checkpoint(checkpoint_id, {}, question="Which site?")
        with brain._browser_clarification_lock:
            brain._pending_browser_clarification = {
                "task_description": "find the price",
                "question": "Which site should I open?",
                "checkpoint_id": checkpoint_id,
                "expires": brain.time.time() + 60,
            }
        return checkpoint_id

    def test_answer_resumes_the_checkpoint(self):
        cid = self._arm()
        calls = []

        def fake_execute(description, original, resume_from=None):
            calls.append((description, resume_from))
            return "running"

        with patch.object(brain, "_execute_deferred_opencode",
                          side_effect=fake_execute):
            reply = brain._consume_browser_followup("use amazon.in")
        self.assertEqual(reply, "running")
        self.assertEqual(len(calls), 1)
        description, resume_from = calls[0]
        self.assertEqual(resume_from, cid,
                         "the follow-up must RESUME, not restart")
        self.assertIn("use amazon.in", description)
        self.assertNotIn("find the price", description,
                         "the old description must not be re-run as a new task")

    def test_cancel_drops_the_checkpoint_and_falls_through(self):
        cid = self._arm()
        with patch.object(brain, "_execute_deferred_opencode") as fake:
            reply = brain._consume_browser_followup("cancel")
        self.assertIsNone(reply)
        fake.assert_not_called()
        self.assertIsNone(browser_agent.peek_checkpoint(cid),
                          "a cancelled clarification must not restart later")

    def test_negative_answer_does_not_restart(self):
        cid = self._arm()
        with patch.object(brain, "_execute_deferred_opencode") as fake:
            self.assertIsNone(brain._consume_browser_followup("no, don't"))
        fake.assert_not_called()
        self.assertIsNone(browser_agent.peek_checkpoint(cid))

    def test_expanded_scope_falls_through_to_the_normal_path(self):
        cid = self._arm()
        with patch.object(brain, "_execute_deferred_opencode") as fake:
            reply = brain._consume_browser_followup(
                "open gmail and send the report to my manager instead")
        self.assertIsNone(reply,
                          "new scope must go through the normal (approved) path")
        fake.assert_not_called()
        self.assertIsNone(browser_agent.peek_checkpoint(cid))

    def test_a_short_answer_is_not_new_scope(self):
        self._arm()
        with patch.object(brain, "_execute_deferred_opencode",
                          return_value="ok") as fake:
            self.assertEqual(brain._consume_browser_followup("amazon.in"), "ok")
        fake.assert_called_once()

    def test_missing_checkpoint_never_replays_the_description(self):
        with brain._browser_clarification_lock:
            brain._pending_browser_clarification = {
                "task_description": "find the price",
                "question": "Which site?",
                "checkpoint_id": None,
                "expires": brain.time.time() + 60,
            }
        with patch.object(brain, "_execute_deferred_opencode") as fake:
            reply = brain._consume_browser_followup("amazon.in")
        fake.assert_not_called()
        self.assertIn("no longer suspended", reply)

    def test_expiry_drops_the_checkpoint(self):
        cid = self._arm()
        with brain._browser_clarification_lock:
            brain._pending_browser_clarification["expires"] = brain.time.time() - 1
        with patch.object(brain, "_execute_deferred_opencode") as fake:
            self.assertIsNone(brain._consume_browser_followup("amazon.in"))
        fake.assert_not_called()
        self.assertIsNone(browser_agent.peek_checkpoint(cid))


class ResumeWiringTests(unittest.TestCase):
    def test_run_browser_task_accepts_resume_from(self):
        import inspect

        params = inspect.signature(browser_agent.run_browser_task).parameters
        self.assertIn("resume_from", params)

    def test_deferred_execution_passes_resume_from(self):
        import inspect

        code = inspect.getsource(brain._execute_deferred_opencode)
        self.assertIn("resume_from=resume_from", code)

    def test_agent_loop_restores_the_checkpoint(self):
        import inspect

        code = inspect.getsource(browser_agent._agent_loop_inner)
        self.assertIn("completed_effects", code)
        self.assertIn("_checkpoint_history", code)


if __name__ == "__main__":
    unittest.main()
