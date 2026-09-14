"""F18 — bind consent to the entire plan.

Acceptance (audit report): "Every target/effect appears in the full preview;
confirmations cannot cross-authorize; model output cannot override no;
changed/expired plans execute nothing."
"""

import time
import unittest
from unittest.mock import patch

from backend.services import approvals, screen_state, tool_policy
from backend.services.task_agent import agent


def _native_plan():
    return {
        "ok": True,
        "requires_confirmation": True,
        "command_text": "run the build and write the log",
        "steps": [
            {"tool": "code.run_command", "args": {"command": "npm run build"},
             "risk": "confirm"},
            {"tool": "code.write_file",
             "args": {"path": "out/build.txt", "content": "done"},
             "risk": "confirm"},
            {"tool": "code.create_folder", "args": {"path": "out/extra"},
             "risk": "safe"},
        ],
    }


class PreviewCoverageTests(unittest.TestCase):
    """Every target and effect is present in what the user approves."""

    def tearDown(self):
        approvals.clear()
        agent._pending_task_action = None

    def test_describe_plan_reads_native_args(self):
        effects = tool_policy.describe_plan(
            {"steps": [{"tool": "code.write_file",
                        "args": {"path": "a.txt", "content": "x"}}]})
        self.assertEqual(len(effects), 1)
        self.assertIn("a.txt", effects[0])
        self.assertIn("write_file", effects[0])

    def test_describe_plan_carries_the_external_effect_preview(self):
        # F14: an effect step (send mail / commit event) must show the SAME
        # scrubbed draft preview in the approval record that the task agent
        # showed in the spoken confirmation — tool+args alone hide what is
        # actually being sent.
        preview = ("Ready to proceed after your approval: Send mail from "
                   "<acct> to <to> — subject 'Quarterly report'")
        effects = tool_policy.describe_plan(
            {"steps": [{"tool": "mail.send_draft",
                        "args": {"draft_id": "d1"},
                        "effect_preview": preview}]})
        self.assertEqual(effects, [preview])

    def test_describe_plan_masks_secrets_in_an_effect_preview(self):
        effects = tool_policy.describe_plan(
            {"steps": [{"tool": "mail.send_draft", "args": {},
                        "effect_preview": "Send mail with token=abcd1234"}]})
        self.assertNotIn("abcd1234", effects[0])

    def test_full_preview_lists_every_effect_and_target(self):
        record = approvals.build_record(_native_plan(), "build it",
                                        scope="task")
        self.assertEqual(len(record.effects), 3)
        self.assertIn("out/build.txt", record.preview)
        self.assertIn("out/extra", record.preview)
        self.assertIn("npm run build", record.preview)
        for effect in record.effects:
            self.assertIn(effect, record.preview)

    def test_spoken_preview_counts_the_rest_of_the_plan(self):
        preview = agent._confirmation_preview(_native_plan())
        # The lead effect is described, and the remaining work is counted.
        self.assertIn("npm run build", preview)
        self.assertIn("file write", preview)
        self.assertIn("folder creation", preview)

    def test_spoken_preview_mentions_a_single_effect_plainly(self):
        plan = {"steps": [{"tool": "code.run_command",
                           "args": {"command": "pip list"},
                           "risk": "confirm"}]}
        preview = agent._confirmation_preview(plan)
        self.assertIn("pip list", preview)
        self.assertNotIn("Plus", preview)

    def test_confirmation_arms_a_plan_bound_record(self):
        agent.execute_plan(_native_plan(), {})
        record = approvals.pending()
        self.assertIsNotNone(record)
        self.assertEqual(record.scope, "task")
        self.assertEqual(record.plan_hash,
                         approvals.plan_hash(_native_plan(),
                                             "run the build and write the log"))
        self.assertGreater(record.expires_at, time.time())


class CrossAuthorizationTests(unittest.TestCase):
    def tearDown(self):
        approvals.clear()
        agent._pending_task_action = None

    def test_a_replaced_record_executes_nothing(self):
        agent.execute_plan(_native_plan(), {})
        original_id = agent._pending_task_action["approval_id"]
        # Another request arms its own approval in the meantime.
        other = approvals.arm({"steps": [{"tool": "code.run_command",
                                          "args": {"command": "rm -rf /"}}]},
                              "something else", scope="screen")
        self.assertNotEqual(original_id, other.id)
        with patch.object(agent, "execute_plan") as executor:
            reply = agent.consume_task_confirmation("confirm task")
        executor.assert_not_called()
        self.assertIn("replaced", reply)

    def test_changed_plan_executes_nothing(self):
        plan = _native_plan()
        agent.execute_plan(plan, {})
        # The plan materially changes AFTER it was approved.
        plan["steps"][0]["args"]["command"] = "npm run deploy --force"
        with patch.object(agent, "execute_plan") as executor:
            reply = agent.consume_task_confirmation("confirm task")
        executor.assert_not_called()
        self.assertIn("changed since it was approved", reply)

    def test_expired_window_executes_nothing(self):
        agent.execute_plan(_native_plan(), {})
        agent._pending_task_action["expires"] = time.time() - 1
        with patch.object(agent, "execute_plan") as executor:
            reply = agent.consume_task_confirmation("confirm task")
        executor.assert_not_called()
        self.assertIsNone(reply)
        self.assertIsNone(approvals.pending())

    def test_explicit_negative_consumes_without_executing(self):
        agent.execute_plan(_native_plan(), {})
        with patch.object(agent, "execute_plan") as executor:
            reply = agent.consume_task_confirmation("no, don't do it")
        executor.assert_not_called()
        self.assertIn("skip", reply)
        self.assertIsNone(approvals.pending())

    def test_a_confirmed_plan_runs_the_approved_snapshot(self):
        agent.execute_plan(_native_plan(), {})
        with patch.object(agent, "execute_plan",
                          return_value="ran it") as executor:
            reply = agent.consume_task_confirmation("confirm task")
        executor.assert_called_once()
        self.assertEqual(reply, "ran it")
        self.assertIsNone(approvals.pending())

    def test_has_pending_confirmation_follows_the_shared_record(self):
        agent.execute_plan(_native_plan(), {})
        self.assertTrue(agent.has_pending_task_confirmation())
        approvals.cancel("test")
        self.assertFalse(agent.has_pending_task_confirmation())


class NegativeFirstTests(unittest.TestCase):
    """A deterministic negative is never interpreted by the model."""

    def _brain(self):
        from backend.core import brain
        return brain

    def test_explicit_no_is_not_sent_to_the_model(self):
        brain = self._brain()
        brain._arm_confirmation("what is the price of gold")
        with patch.object(brain, "_llm_resolve_confirmation",
                          return_value=("yes", "gold price")) as model, \
             patch.object(brain, "handle_research_intent") as research:
            reply = brain._consume_confirmation("no, don't do it")
        model.assert_not_called()
        research.assert_not_called()
        self.assertIn("what I know", reply)
        self.assertIsNone(brain._pending_confirmation)

    def test_explicit_yes_is_confirmed_without_a_verdict_from_the_model(self):
        brain = self._brain()
        brain._arm_confirmation("what is the price of gold")
        with patch.object(brain, "_llm_resolve_confirmation",
                          return_value=("no", None)) as model, \
             patch.object(brain, "handle_research_intent") as research:
            reply = brain._consume_confirmation("yes")
        model.assert_called_once()
        research.assert_not_called()
        self.assertIn("what I know", reply)

    def test_inconclusive_answer_falls_through_to_the_model(self):
        brain = self._brain()
        brain._arm_confirmation("what is the price of gold")
        with patch.object(brain, "_llm_resolve_confirmation",
                          return_value=("yes", "gold price")) as model, \
             patch.object(brain, "handle_research_intent",
                          return_value="researching") as research:
            reply = brain._consume_confirmation("han kar do yaar")
        model.assert_called_once()
        research.assert_called_once()
        self.assertEqual(reply, "researching")


class ScreenConsumptionTests(unittest.TestCase):
    """Verification and consumption are one atomic step."""

    def tearDown(self):
        screen_state.set_enabled(False)
        screen_state.clear_pending_plan()
        approvals.clear()

    def test_consume_returns_an_immutable_snapshot_with_its_record(self):
        screen_state.set_enabled(True)
        plan = {"steps": [{"action": "click", "x": 10, "y": 20}],
                "command_text": "click it", "summary": "done"}
        screen_state.set_pending_plan(plan, "click it")
        got_plan, record = screen_state.consume_pending_plan()
        self.assertIsNotNone(record)
        self.assertEqual(got_plan["steps"][0]["x"], 10)
        # The snapshot is a copy: mutating it cannot change what was approved.
        got_plan["steps"][0]["x"] = 999
        self.assertIsNone(screen_state.get_pending_plan())
        ok, why = approvals.verify(got_plan, record, "click it")
        self.assertFalse(ok)
        self.assertIn("changed since it was approved", why)

    def test_consume_drops_the_record_so_it_cannot_be_used_twice(self):
        screen_state.set_enabled(True)
        screen_state.set_pending_plan(
            {"steps": [{"action": "click", "x": 1, "y": 2}],
             "command_text": "click"}, "click")
        screen_state.consume_pending_plan()
        self.assertIsNone(approvals.pending())
        plan, record = screen_state.consume_pending_plan()
        self.assertIsNone(plan)
        self.assertIsNone(record)


if __name__ == "__main__":
    unittest.main()
