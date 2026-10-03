"""F02 — route goals, not single categories.

Acceptance (audit report): "One screen/research/edit goal retains evidence,
arms one identified approval, executes only after consent, and never repeats
started work during fallback. Keep safeguards until parity."

What the baseline got wrong, pinned here:
  * ``_coerce_screen_output`` did not exist at all (NameError on every
    screen.observe);
  * declared argument TYPES were not enforced;
  * a falsey argument reached ``envelope.utterance``, which an envelope does
    not always carry;
  * the brain read only the proposal's REPLY — the plan never reached the
    confirmation gate;
  * two proposals could overwrite each other;
  * legacy speculation started BEFORE the route was chosen.
"""

import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from backend.services import approvals
from backend.services import orchestrator
from backend.services import research_service
from backend.services.task_agent import agent as task_agent


def _tool_response(tool, args, content=""):
    return {
        "choices": [{
            "message": {
                "content": content,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": tool, "arguments": json.dumps(args)},
                }],
            }
        }]
    }


def _answer(text):
    return {"choices": [{"message": {"content": text, "tool_calls": []}}]}


class _EnvelopeWithoutUtterance:
    """An envelope shape the handlers must survive (no ``.utterance``)."""

    def to_system_message(self):
        return {"role": "system", "content": "system"}


PLAN = {
    "ok": True,
    "summary": "Create the demo folder?",
    "requires_confirmation": True,
    "command_text": "create a folder named demo",
    "steps": [{"tool": "code.create_folder", "args": {"path": "C:/x/demo"},
               "risk": "safe", "reason": "r"}],
}


class ScreenOutputCoercionTests(unittest.TestCase):
    """The missing helper: every screen.observe used to raise NameError."""

    def test_dict_output_yields_its_answer(self):
        self.assertEqual(
            orchestrator._coerce_screen_output({"answer": "VS Code is open."}),
            "VS Code is open.")

    def test_string_output_passes_through(self):
        self.assertEqual(orchestrator._coerce_screen_output(" editor "), "editor")

    def test_list_and_none_outputs_are_handled(self):
        self.assertEqual(
            orchestrator._coerce_screen_output([{"text": "a"}, "b"]), "a\nb")
        self.assertEqual(orchestrator._coerce_screen_output(None), "")

    def test_object_output_reads_common_attributes(self):
        class Result:
            summary = "two windows"

        self.assertEqual(orchestrator._coerce_screen_output(Result()), "two windows")

    def test_screen_observe_records_evidence_instead_of_erroring(self):
        outcome = orchestrator.new_outcome()
        with patch("backend.services.screen_analyzer.analyze_screen",
                   return_value={"answer": "A code editor is open."}):
            text = orchestrator._tool_screen_observe(
                {"question": "what's on screen"}, outcome, _EnvelopeWithoutUtterance())
        self.assertEqual(text, "A code editor is open.")
        self.assertTrue(any("screen.observe" in e for e in outcome["evidence"]))


class EnvelopeFallbackTests(unittest.TestCase):
    """Falsey arguments must never reach a missing attribute."""

    def test_missing_utterance_is_empty_not_an_attribute_error(self):
        self.assertEqual(orchestrator.envelope_utterance(_EnvelopeWithoutUtterance()), "")

    def test_research_lookup_with_a_falsey_query_uses_the_envelope_text(self):
        class Env(_EnvelopeWithoutUtterance):
            utterance = "the real question"

        with patch("backend.services.quick_search.run_quick_search",
                   return_value="answer") as lookup:
            orchestrator._tool_research_lookup(
                {"query": ""}, orchestrator.new_outcome(), Env())
        lookup.assert_called_once_with("the real question")

    def test_research_lookup_with_no_envelope_text_still_runs(self):
        with patch("backend.services.quick_search.run_quick_search",
                   return_value="answer"):
            text = orchestrator._tool_research_lookup(
                {"query": None}, orchestrator.new_outcome(),
                _EnvelopeWithoutUtterance())
        self.assertEqual(text, "answer")


class ArgumentTypeValidationTests(unittest.TestCase):
    """F02: wrong argument types were accepted."""

    def test_a_number_where_a_string_is_declared_is_rejected(self):
        args, error = orchestrator.validate_tool_call("screen.observe", '{"question": 5}')
        self.assertIsNone(args)
        self.assertIn("must be string", error)

    def test_a_boolean_where_a_string_is_declared_is_rejected(self):
        args, error = orchestrator.validate_tool_call("research.lookup", '{"query": true}')
        self.assertIsNone(args)
        self.assertIn("must be string", error)

    def test_a_string_argument_still_validates(self):
        args, error = orchestrator.validate_tool_call(
            "research.lookup", '{"query": "price of gold"}')
        self.assertIsNone(error)
        self.assertEqual(args, {"query": "price of gold"})

    def test_a_wrong_typed_call_is_fed_back_never_executed(self):
        env = _EnvelopeWithoutUtterance()
        handler = MagicMock()
        with patch.object(orchestrator, "_chat",
                          MagicMock(side_effect=[
                              _tool_response("research.lookup", {"query": 12}),
                              _answer("Let me ask again.")])), \
             patch.dict(orchestrator._TOOL_HANDLERS,
                        {"research.lookup": handler}, clear=False):
            outcome = orchestrator.run_orchestrator("look it up", env)
        handler.assert_not_called()
        self.assertTrue(any("tool rejected" in e for e in outcome["evidence"]))
        self.assertEqual(outcome["reply"], "Let me ask again.")


class OutcomeContractTests(unittest.TestCase):
    """F02: explicit answer / proposal / suspension / error contracts."""

    def test_every_status_is_in_the_contract(self):
        self.assertEqual(orchestrator.STATUSES,
                         {"answered", "proposal", "suspension", "error"})

    def test_unusable_output_with_no_actions_is_suspension_and_declines(self):
        env = _EnvelopeWithoutUtterance()
        with patch.object(orchestrator, "_chat", return_value=_answer("")):
            outcome = orchestrator.run_orchestrator("something", env)
        self.assertIsNone(outcome)

    def test_a_started_tool_survives_a_later_planner_failure(self):
        """Never repeat started work: partial evidence is returned, not None."""
        env = _EnvelopeWithoutUtterance()
        observe = MagicMock(return_value="A code editor is open.")
        with patch.object(orchestrator, "_chat",
                          MagicMock(side_effect=[
                              _tool_response("screen.observe", {"question": "?"}),
                              {}])), \
             patch.object(orchestrator, "_tool_screen_observe", observe):
            outcome = orchestrator.run_orchestrator("whats on my screen", env)
        self.assertIsNotNone(outcome, "started work must not be discarded")
        self.assertEqual(outcome["status"], "suspension")
        self.assertEqual(outcome["actions"][0]["tool"], "screen.observe")

    def test_a_proposal_status_is_proposal_not_needs_input(self):
        env = _EnvelopeWithoutUtterance()
        with patch.object(orchestrator, "_chat",
                          MagicMock(side_effect=[_tool_response(
                              "task.propose", {"description": "create demo"})])), \
             patch.object(task_agent, "plan_task", return_value=PLAN):
            outcome = orchestrator.run_orchestrator("create demo", env)
        self.assertEqual(outcome["status"], "proposal")
        self.assertEqual(len(outcome["proposals"]), 1)
        self.assertTrue(outcome["proposals"][0]["approval_id"])


class ProposalApprovalTests(unittest.TestCase):
    """One goal → one identified approval → execution only after consent."""

    def setUp(self):
        approvals.cancel("test setup")
        task_agent._pending_task_action = None
        research_service.clear_stop_request()

    def tearDown(self):
        approvals.cancel("test teardown")
        task_agent._pending_task_action = None

    def test_a_proposal_arms_the_shared_confirmation_gate(self):
        env = _EnvelopeWithoutUtterance()
        with patch.object(orchestrator, "_chat",
                          MagicMock(side_effect=[_tool_response(
                              "task.propose", {"description": "create demo"})])), \
             patch.object(task_agent, "plan_task", return_value=PLAN):
            outcome = orchestrator.run_orchestrator("create demo", env)
        record = approvals.pending()
        self.assertIsNotNone(record, "the proposal must be identified")
        self.assertEqual(record.id, outcome["proposals"][0]["approval_id"])
        self.assertEqual(task_agent._pending_task_action["plan"], PLAN)
        self.assertEqual(task_agent._pending_task_action["approval_id"], record.id)

    def test_evidence_is_retained_through_the_proposal(self):
        env = _EnvelopeWithoutUtterance()
        observe = MagicMock(return_value="A code editor is open.")
        with patch.object(orchestrator, "_chat", MagicMock(side_effect=[
                _tool_response("screen.observe", {"question": "what is open"}),
                _tool_response("task.propose", {"description": "close the editor"})])), \
             patch.object(orchestrator, "_tool_screen_observe", observe), \
             patch.object(task_agent, "plan_task", return_value=PLAN):
            outcome = orchestrator.run_orchestrator("close the editor", env)
        self.assertEqual(outcome["status"], "proposal")
        self.assertTrue(any("screen.observe" in e for e in outcome["evidence"]),
                        "the goal keeps the evidence collected before it")

    def test_consent_executes_the_very_plan_that_was_proposed(self):
        env = _EnvelopeWithoutUtterance()
        with patch.object(orchestrator, "_chat",
                          MagicMock(side_effect=[_tool_response(
                              "task.propose", {"description": "create demo"})])), \
             patch.object(task_agent, "plan_task", return_value=PLAN):
            orchestrator.run_orchestrator("create demo", env)
        executed = {}

        def fake_execute(plan, context, task_text="", confirmed=False,
                         approval=None):
            executed["plan"] = plan
            executed["confirmed"] = confirmed or approval is not None
            from backend.services.task_result import TaskResult

            return TaskResult("completed", "Created the demo folder, sir.")

        with patch.object(task_agent, "execute_plan", side_effect=fake_execute):
            reply = task_agent.consume_task_confirmation("confirm task")
        self.assertTrue(executed.get("confirmed"),
                        "nothing may execute before consent")
        self.assertEqual(executed["plan"], PLAN)
        self.assertIn("Created the demo folder", reply)

    def test_a_second_proposal_in_one_turn_is_refused_not_armed(self):
        """At most ONE proposal is armed; a second cannot replace it."""
        outcome = orchestrator.new_outcome()
        env = _EnvelopeWithoutUtterance()
        with patch.object(task_agent, "plan_task", return_value=PLAN):
            orchestrator._tool_task_propose({"description": "create demo"},
                                            outcome, env)
            armed = outcome["proposals"][0]["approval_id"]
            second = {
                "ok": True, "summary": "Delete everything?",
                "requires_confirmation": True, "command_text": "delete x",
                "steps": [{"tool": "code.delete", "args": {"path": "C:/x"},
                           "risk": "dangerous", "reason": "r"}],
            }
            with patch.object(task_agent, "plan_task", return_value=second):
                reply = orchestrator._tool_task_propose(
                    {"description": "delete x"}, outcome, env)
        self.assertIn("already awaiting confirmation", reply)
        self.assertEqual(len(outcome["proposals"]), 1,
                         "the refused proposal must not be recorded as armed")
        self.assertEqual(approvals.pending().id, armed)
        self.assertEqual(approvals.pending().plan_hash,
                         approvals.plan_hash(PLAN, "create a folder named demo"))

    def test_a_different_task_already_pending_is_not_replaced(self):
        """Across requests too: an armed approval is never silently swapped."""
        other = {
            "ok": True, "summary": "Delete everything?",
            "requires_confirmation": True, "command_text": "delete x",
            "steps": [{"tool": "code.delete", "args": {"path": "C:/x"},
                       "risk": "dangerous", "reason": "r"}],
        }
        record, reason = task_agent.register_proposal(
            other, task_text="delete x")
        self.assertIsNotNone(record)
        again, reason = task_agent.register_proposal(
            PLAN, task_text="create demo")
        self.assertIsNone(again)
        self.assertIn("already waiting", reason)
        self.assertEqual(approvals.pending().id, record.id)

    def test_re_proposing_the_same_plan_reuses_the_same_approval(self):
        first, _ = task_agent.register_proposal(PLAN, task_text="create demo")
        second, reason = task_agent.register_proposal(
            PLAN, task_text="create demo")
        self.assertEqual(first.id, second.id)
        self.assertEqual(reason, "")


class RouteSelectionTests(unittest.TestCase):
    """F02: select the route BEFORE speculating."""

    def test_legacy_when_the_flag_is_off(self):
        from backend import config
        with patch.object(config, "ORCHESTRATOR_MODE", "legacy"):
            self.assertEqual(orchestrator.select_route("what is on my screen"),
                             "legacy")

    def test_orchestrator_when_the_flag_is_on(self):
        from backend import config
        with patch.object(config, "ORCHESTRATOR_MODE", "orchestrator"):
            self.assertEqual(orchestrator.select_route("find the price of gold"),
                             "orchestrator")

    def test_the_command_prefix_belongs_to_the_legacy_task_path(self):
        from backend import config
        with patch.object(config, "ORCHESTRATOR_MODE", "orchestrator"):
            self.assertEqual(orchestrator.select_route("command run pip list"),
                             "legacy")

    def test_empty_messages_are_legacy(self):
        from backend import config
        with patch.object(config, "ORCHESTRATOR_MODE", "orchestrator"):
            self.assertEqual(orchestrator.select_route("   "), "legacy")


class BrainRouteBeforeSpeculationTests(unittest.TestCase):
    """F02: legacy speculation must not start on an orchestrator route.

    ``process_message`` is the real entry point, so these tests exercise the
    actual ordering: route selection, then speculation, then the model call.
    """

    def setUp(self):
        research_service.clear_stop_request()
        approvals.cancel("test setup")
        task_agent._pending_task_action = None

    def tearDown(self):
        approvals.cancel("test teardown")
        task_agent._pending_task_action = None

    def _run(self, message, orchestrator_reply=None, reraise=None, **overrides):
        from backend import config
        from backend.core import brain

        started = []
        patches = {
            "ORCHESTRATOR_MODE": "orchestrator",
            "get_history": MagicMock(return_value=[]),
            "classify_intent": MagicMock(return_value={"intent": "chat"}),
            "handle_chat": MagicMock(return_value="Hello, sir."),
            "should_search": MagicMock(return_value=False),
            "is_screen_question": MagicMock(return_value=False),
        }
        patches.update(overrides)
        racer = MagicMock(side_effect=lambda *a, **k: started.append(a))
        handle = patches.pop("orchestrator_handle_message",
                             MagicMock(return_value=orchestrator_reply))
        with patch.object(config, "ORCHESTRATOR_MODE", patches.pop("ORCHESTRATOR_MODE")), \
             patch.object(brain, "_ChatRacer", racer), \
             patch.object(brain, "orchestrator_handle_message", handle), \
             patch.object(brain, "get_history", patches.pop("get_history")), \
             patch.object(brain, "classify_intent", patches.pop("classify_intent")), \
             patch.object(brain, "handle_chat", patches.pop("handle_chat")), \
             patch.object(brain, "should_search", patches.pop("should_search")), \
             patch.object(brain, "is_screen_question",
                          patches.pop("is_screen_question")):
            reply = brain.process_message(
                message, from_voice=False, sync_voice=False,
                commit_response=False)
        return reply, started, handle

    def test_speculation_never_starts_when_the_orchestrator_owns_the_route(self):
        reply, started, _handle = self._run(
            "what is on my screen",
            orchestrator_reply={"status": "answered", "reply": "Fine, sir.",
                                "plan": None})
        self.assertEqual(started, [],
                         "no chat work may start on an orchestrator route")
        self.assertEqual(reply, "Fine, sir.")

    def test_the_proposal_reply_is_the_armed_confirmation_prompt(self):
        with patch.object(task_agent, "confirmation_prompt",
                          return_value="Create the demo folder? Say confirm task "
                                       "to proceed, or cancel."):
            reply, started, _handle = self._run(
                "create a demo folder",
                orchestrator_reply={"status": "proposal",
                                    "reply": "plan summary", "plan": PLAN})
        self.assertEqual(started, [])
        self.assertIn("confirm task", reply)

    def test_a_declined_orchestrator_route_falls_back_without_a_racer(self):
        reply, started, _handle = self._run("tell me about tea",
                                            orchestrator_reply=None)
        self.assertEqual(started, [],
                         "a goal-shaped route must not speculate, even on fallback")
        self.assertEqual(reply, "Hello, sir.")

    def test_the_legacy_route_still_speculates(self):
        from backend import config
        from backend.core import brain

        with patch.object(config, "ORCHESTRATOR_MODE", "legacy"), \
             patch.object(brain, "_ChatRacer") as racer, \
             patch.object(brain, "get_history", return_value=[]), \
             patch.object(brain, "classify_intent",
                          return_value={"intent": "chat"}), \
             patch.object(brain, "handle_chat", return_value="Hello, sir."), \
             patch.object(brain, "should_search", return_value=False):
            brain.process_message("tell me about tea", from_voice=False,
                                  sync_voice=False, commit_response=False,
                                  stream_reply=lambda chunk: None)
        racer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
