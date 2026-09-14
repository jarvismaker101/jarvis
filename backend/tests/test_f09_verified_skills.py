"""F09 — successful runs become VERIFIED skills (audit correction).

Acceptance pinned here:
  * Failed/unvalidated runs never become trusted.
  * Recapture preserves the working (promoted) version.
  * Changed selectors and verified replay failures invalidate applicability.

Every test runs against a per-test tmp SQLite file (memory_store.configure);
the user's real memory DB is never opened.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from backend.core import memory_store  # noqa: E402
from backend.services.task_agent import agent  # noqa: E402


class F09TestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        memory_store.configure(os.path.join(self._tmp.name, "mem.db"))
        memory_store.MEMORY_ENABLED = True

    def tearDown(self):
        memory_store.close()
        self._tmp.cleanup()

    def _row(self, skill_id):
        return memory_store._conn().execute(
            "SELECT * FROM skills WHERE id = ?", (skill_id,)).fetchone()

    def _promote(self, skill_id, version=1, evidence="replayed and verified"):
        return memory_store.promote_skill(
            skill_id, approved_version=version, replay_evidence=evidence)

    def _promoted_skill(self, name="pay the invoice",
                        goal="pay the invoice online",
                        steps=("click_locator: selector=#pay",
                               "type_text: selector=#card, text=4111"),
                        postconditions=("payment form open",)):
        skill_id = memory_store.record_skill_candidate(
            name, "browser", goal, steps=list(steps),
            postconditions=list(postconditions))
        self.assertTrue(self._promote(skill_id))
        return skill_id


class UnvalidatedRunsNeverBecomeTrustedTests(F09TestBase):
    """Acceptance clause 1: failed/unvalidated runs never become trusted."""

    def test_failed_run_never_captures_a_candidate(self):
        memory_store.record_task_outcome(
            "browser_agent", "failed", "pay the invoice",
            summary="payment declined",
            trace=[{"tool": "click_locator", "args": {"selector": "#pay"},
                    "observation": "declined", "ok": False}])
        self.assertEqual(
            memory_store.find_skills("pay the invoice", status="candidate"),
            [])
        self.assertEqual(memory_store.find_skills("pay the invoice"), [])

    def test_partial_run_never_captures_a_candidate(self):
        memory_store.record_task_outcome(
            "browser_agent", "partial", "pay the invoice",
            summary="only the form opened",
            trace=[{"tool": "click_locator", "args": {"selector": "#pay"},
                    "observation": "form open", "ok": True}])
        self.assertEqual(
            memory_store.find_skills("pay the invoice", status="candidate"),
            [])

    def test_promotion_requires_the_candidate_version(self):
        skill_id = memory_store.record_skill_candidate(
            "pay the invoice", "browser", "pay the invoice online")
        # No approved version at all: refused.
        self.assertFalse(memory_store.promote_skill(
            skill_id, replay_evidence="replay verified"))
        # A version from a DIFFERENT capture: refused even with evidence.
        self.assertFalse(memory_store.promote_skill(
            skill_id, approved_version=2, replay_evidence="replay verified"))
        self.assertEqual(
            memory_store.find_skills("pay the invoice", status="promoted"), [])
        row = self._row(skill_id)
        self.assertEqual(row["status"], "candidate")
        self.assertIsNone(row["approved_at"])

    def test_promotion_requires_a_verified_replay(self):
        skill_id = memory_store.record_skill_candidate(
            "pay the invoice", "browser", "pay the invoice online")
        # Right version, explicit approval, but NOTHING verified it.
        self.assertFalse(memory_store.promote_skill(skill_id,
                                                    approved_version=1))
        self.assertEqual(
            memory_store.find_skills("pay the invoice", status="promoted"), [])
        # A successful replay validates it; only then does approval promote.
        memory_store.note_skill_replay(skill_id, True,
                                       reason="replayed and verified")
        self.assertIsNotNone(self._row(skill_id)["validated_at"])
        self.assertTrue(memory_store.promote_skill(skill_id,
                                                   approved_version=1))
        self.assertEqual(len(memory_store.find_skills(
            "pay the invoice", status="promoted")), 1)
        self.assertEqual(self._row(skill_id)["replay_evidence"],
                         "verified replay (validated)")

    def test_approval_phrase_cannot_trust_an_unvalidated_candidate(self):
        memory_store.record_skill_candidate(
            "pay the invoice", "browser", "pay the invoice online")
        reply = memory_store.handle_memory_phrase(
            "approve the pay the invoice skill")
        self.assertIn("has not been replayed", reply)
        self.assertEqual(
            memory_store.find_skills("pay the invoice", status="promoted"), [])

    def test_traceless_completion_has_no_procedure_to_capture(self):
        memory_store.record_task_outcome(
            "browser_agent", "completed", "pay the invoice",
            summary="paid the invoice")
        self.assertEqual(
            memory_store.find_skills("pay the invoice", status="candidate"),
            [])


class RecapturePreservesTheWorkingVersionTests(F09TestBase):
    """Acceptance clause 2: recapture preserves the working version."""

    def test_recapture_does_not_retire_the_promoted_version(self):
        first = self._promoted_skill(steps=["click_locator: selector=#pay"])
        second = memory_store.record_skill_candidate(
            "pay the invoice", "browser", "pay the invoice online",
            steps=["click_locator: selector=#pay-now"])
        self.assertEqual(second and self._row(second)["version"], 2)
        self.assertEqual(self._row(second)["status"], "candidate")
        # The trusted version keeps working — and stays retrievable WITH its
        # procedure — while the replacement is only a candidate.
        old = self._row(first)
        self.assertEqual(old["status"], "promoted")
        self.assertIsNone(old["retired_at"])
        promoted = memory_store.find_skills("pay the invoice",
                                            status="promoted")
        self.assertEqual([r["id"] for r in promoted], [first])
        block = memory_store.recall_skills_for("pay the invoice online")
        self.assertIn("selector=#pay", block)

    def test_promoting_the_replacement_retires_the_old_version(self):
        first = self._promoted_skill(steps=["click_locator: selector=#pay"])
        second = memory_store.record_skill_candidate(
            "pay the invoice", "browser", "pay the invoice online",
            steps=["click_locator: selector=#pay-now"])
        self.assertTrue(self._promote(second, version=2))
        self.assertEqual(self._row(first)["status"], "retired")
        self.assertIn("replaced by promoted v2",
                      self._row(first)["retire_reason"])
        promoted = memory_store.find_skills("pay the invoice",
                                            status="promoted")
        self.assertEqual([r["id"] for r in promoted], [second])
        block = memory_store.recall_skills_for("pay the invoice online")
        self.assertIn("selector=#pay-now", block)

    def test_a_stale_version_needs_fresh_approval(self):
        self._promoted_skill()
        second = memory_store.record_skill_candidate(
            "pay the invoice", "browser", "pay the invoice online",
            steps=["click_locator: selector=#pay-now"])
        # Approving the PREVIOUS version must not promote the recapture.
        self.assertFalse(self._promote(second, version=1))
        self.assertEqual(self._row(second)["status"], "candidate")


class ApplicabilityInvalidationTests(F09TestBase):
    """Acceptance clause 3: changed selectors / replay failures invalidate."""

    def test_changed_selector_invalidates_on_the_first_failure(self):
        skill_id = self._promoted_skill()
        reason = memory_store.note_skill_replay(
            skill_id, False,
            reason="browser.click_locator: failed: selector #pay not found",
            observed="the payment button is gone")
        self.assertTrue(reason)
        self.assertIn("#pay", reason)
        row = self._row(skill_id)
        self.assertEqual(row["status"], "invalidated")
        self.assertEqual(row["failure_count"], 1)
        self.assertIn("#pay", row["retire_reason"])
        # No longer offered to the planner.
        self.assertEqual(
            memory_store.find_skills("pay the invoice", status="promoted"), [])

    def test_a_generic_failure_needs_repeats(self):
        skill_id = self._promoted_skill()
        memory_store.note_skill_replay(skill_id, False,
                                       reason="the network is slow")
        self.assertEqual(self._row(skill_id)["status"], "promoted")
        for _ in range(memory_store._SKILL_INVALIDATE_FAILURES - 1):
            memory_store.note_skill_replay(skill_id, False,
                                           reason="the network is slow")
        row = self._row(skill_id)
        self.assertEqual(row["status"], "invalidated")
        self.assertIn("invalidated after",
                      row["retire_reason"])

    def test_observation_contradicting_a_postcondition_invalidates(self):
        skill_id = self._promoted_skill(
            steps=["click_locator: selector=#pay"],
            postconditions=("payment form open",))
        reason = memory_store.note_skill_replay(
            skill_id, False, reason="verification failed",
            observed="postcondition failed: payment form open "
                     "does not match the current page")
        self.assertTrue(reason)
        self.assertEqual(self._row(skill_id)["status"], "invalidated")

    def test_successful_replay_records_the_evidence(self):
        skill_id = self._promoted_skill()
        memory_store.note_skill_replay(skill_id, True,
                                       reason="paid the invoice again")
        row = self._row(skill_id)
        self.assertEqual(row["success_count"], 1)
        self.assertIsNotNone(row["validated_at"])
        self.assertEqual(row["replay_evidence"], "paid the invoice again")
        self.assertEqual(row["status"], "promoted")

    def test_note_skill_outcome_stays_back_compatible(self):
        skill_id = self._promoted_skill()
        memory_store.note_skill_outcome(skill_id, success=True)
        self.assertEqual(self._row(skill_id)["success_count"], 1)
        memory_store.note_skill_outcome(skill_id, success=False)
        self.assertEqual(self._row(skill_id)["failure_count"], 1)


class ProcedureRetrievalTests(F09TestBase):
    """Acceptance: retrieval carries the ACTUAL captured procedure."""

    def test_trace_becomes_real_steps_postconditions_and_params(self):
        memory_store.record_task_outcome(
            "browser_agent", "completed", "pay the invoice",
            summary="paid the invoice",
            trace=[
                {"tool": "browser.click_locator",
                 "args": {"selector": "#pay"}, "observation": "form open",
                 "ok": True},
                {"tool": "browser.type_text",
                 "args": {"selector": "#card", "text": "4111"},
                 "observation": "card entered", "ok": True},
                {"tool": "browser.click_locator",
                 "args": {"selector": "#broken"}, "observation": "nope",
                 "ok": False},
            ],
            verification=["payment confirmed"])
        candidates = memory_store.find_skills("pay the invoice",
                                             status="candidate")
        self.assertEqual(len(candidates), 1)
        row = candidates[0]
        steps = json.loads(row["steps"])
        # Real committed steps — not the goal sentence, and no failed step.
        self.assertEqual(steps[0], "browser.click_locator: selector=#pay")
        self.assertIn("browser.type_text: selector=#card, text=4111", steps)
        self.assertNotIn("browser.click_locator: selector=#broken", steps)
        posts = json.loads(row["postconditions"])
        self.assertIn("form open", posts)
        self.assertIn("payment confirmed", posts)
        schema = json.loads(row["param_schema"])
        self.assertEqual(schema["type"], "object")
        self.assertIn("selector", schema["properties"])
        self.assertIn("text", schema["properties"])

    def test_retrieval_block_contains_the_procedure(self):
        skill_id = self._promoted_skill()
        block = memory_store.recall_skills_for("pay the invoice online")
        self.assertIn("Approved skill", block)
        self.assertIn("selector=#pay", block)
        self.assertIn("payment form open", block)
        self.assertLessEqual(len(block), memory_store.RECALL_BUDGET_CHARS + 40)
        rendered = memory_store.skill_procedure(self._row(skill_id))
        self.assertIn("Steps:", rendered)
        self.assertIn("Postconditions:", rendered)

    def test_only_the_exact_status_field_counts_as_verified(self):
        memory_store.record_event(
            "task_result",
            "[browser_agent/not completed] the invoice was completed nothing "
            "of the sort")
        memory_store.record_event(
            "task_result",
            "[browser_agent/partial] the invoice lookup completed nothing")
        block = memory_store.recall_skills_for("invoice lookup")
        self.assertNotIn("Verified prior outcome", block)
        memory_store.record_event(
            "task_result",
            "[browser_agent/completed] the invoice lookup finished")
        block = memory_store.recall_skills_for("invoice lookup")
        self.assertIn("Verified prior outcome", block)
        self.assertEqual(memory_store._event_status(
            "[browser_agent/not completed] x"), "not completed")
        self.assertEqual(
            memory_store._event_status("[x/partial] completed nothing"),
            "partial")


class AgentReplayFeedbackTests(F09TestBase):
    """Acceptance: the native engine reports the real replay outcome."""

    def _plan(self, skill_id):
        return {
            "ok": True,
            "confidence": 0.9,
            "summary": "pay the invoice",
            "requires_confirmation": False,
            "steps": [{"tool": "browser.click_locator",
                       "args": {"selector": "#pay"},
                       "risk": "safe", "reason": "click pay"}],
            "memory_skills": [{"id": skill_id, "name": "pay-the-invoice",
                               "version": 1}],
        }

    def test_verified_completion_records_a_successful_replay(self):
        skill_id = self._promoted_skill()
        with patch.object(agent, "_execute_step_structured",
                          return_value=("clicked #pay", {"ok": True})):
            result = agent.execute_plan(self._plan(skill_id), {})
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.trace[0]["tool"], "browser.click_locator")
        self.assertTrue(result.trace[0]["ok"])
        self.assertEqual(result.trace[0]["args"], {"selector": "#pay"})
        self.assertIn("clicked #pay", result.verification[0])
        row = self._row(skill_id)
        self.assertEqual(row["success_count"], 1)
        self.assertIsNotNone(row["validated_at"])
        self.assertEqual(row["status"], "promoted")

    def test_failed_run_invalidates_the_changed_selector(self):
        skill_id = self._promoted_skill()
        failure = ("browser.click_locator failed: selector #pay not found",
                   {"ok": False, "error": "selector #pay not found"})
        with patch.object(agent, "_execute_step_structured",
                          return_value=failure):
            result = agent.execute_plan(self._plan(skill_id), {})
        self.assertNotEqual(result.status, "completed")
        self.assertFalse(result.trace[0]["ok"])
        self.assertEqual(self._row(skill_id)["status"], "invalidated")
        self.assertIn("#pay", self._row(skill_id)["retire_reason"])
        self.assertEqual(
            memory_store.find_skills("pay the invoice", status="promoted"), [])

    def test_unverified_completion_is_not_a_successful_replay(self):
        skill_id = self._promoted_skill()
        # Every step answers, none leaves an observable effect: the run is
        # partial, so the offered skill must not be marked as replayed.
        with patch.object(agent, "_execute_step_structured",
                          return_value=("(no output)", {"ok": True})):
            result = agent.execute_plan(self._plan(skill_id), {})
        self.assertEqual(result.status, "partial")
        self.assertEqual(self._row(skill_id)["success_count"], 0)
        self.assertIsNone(self._row(skill_id)["validated_at"])

    def test_no_skill_means_no_replay_bookkeeping(self):
        plan = self._plan(1)
        plan.pop("memory_skills")
        with patch.object(agent, "_execute_step_structured",
                          return_value=("clicked", {"ok": True})), \
             patch.object(memory_store, "note_skill_replay") as replay:
            agent.execute_plan(plan, {})
        replay.assert_not_called()

    def test_normalize_plan_carries_the_offered_skill_identity(self):
        normalized = agent._normalize_plan(
            {"ok": True, "steps": [{"tool": "code.read_file",
                                    "args": {"path": "a.txt"}}],
             "memory_skills": [{"id": 7, "name": "pay-the-invoice",
                                "version": 2}]},
            "pay the invoice")
        self.assertEqual(normalized["memory_skills"],
                         [{"id": 7, "name": "pay-the-invoice",
                           "version": 2}])


if __name__ == "__main__":
    unittest.main()
