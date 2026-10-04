"""F07/F09, brain side — the native engine's structured result is persisted.

The audit's context gap: ``handle_task_message`` speaks a string, so the
verified ``TaskResult`` (trace + verification) was thrown away. A verified
native run could therefore never become a skill, and the work log only ever
saw the opencode engine.

Pinned here: the agent keeps the last structured result, brain records it with
trace/verification, a turn that ran nothing records nothing, and a stale
result can never be attributed to a later turn.

R19 adds the history half: the terminal native run also leaves one compact
``[task]`` note in the conversation, so "what file did you create?" is
recallable from history rather than only from work events.
"""

import unittest
from unittest import mock

from backend.core import brain
from backend.core import memory
from backend.services.task_agent import agent


class RememberResultTests(unittest.TestCase):
    def tearDown(self):
        agent._remember_task_result(None)

    def test_handle_task_message_remembers_the_structured_result(self):
        result = mock.Mock(status="completed", summary="done",
                           detail="", evidence=[], trace=[{"tool": "x"}],
                           verification=["observed"])
        with mock.patch.object(agent, "gather_context", return_value={}), \
             mock.patch.object(agent, "plan_task", return_value={"ok": True,
                                                                 "steps": []}), \
             mock.patch.object(agent, "execute_plan", return_value=result):
            agent.handle_task_message("open the file")
        got, text = agent.last_task_result()
        self.assertIs(got, result)
        self.assertEqual(text, "open the file")

    def test_a_new_run_never_inherits_the_previous_result(self):
        agent._remember_task_result("stale", "old")
        with mock.patch.object(agent, "gather_context", return_value={}), \
             mock.patch.object(agent, "plan_task", return_value={"ok": False,
                                                                 "response": "no"}), \
             mock.patch.object(agent, "execute_plan",
                               return_value=type("R", (), {"__str__": lambda s: "no"})()):
            agent.handle_task_message("something")
        got, _ = agent.last_task_result()
        self.assertIsNot(got, "stale")

    def test_a_declined_confirmation_records_nothing(self):
        agent._remember_task_result("stale", "old")
        with mock.patch.object(agent, "_pending_task_action",
                               {"plan": {"summary": "s"}, "context": {},
                                "expires": 9e9, "approval_id": "a"}), \
             mock.patch.object(agent.approvals, "cancel"):
            reply = agent.consume_task_confirmation("no thanks")
        self.assertIn("skip", reply)
        got, _ = agent.last_task_result()
        self.assertIsNone(got, "declining a plan must not replay an old result")

    def test_last_task_result_is_none_before_anything_runs(self):
        agent._remember_task_result(None)
        self.assertEqual(agent.last_task_result(), (None, ""))


class BrainNativeOutcomeTests(unittest.TestCase):
    def setUp(self):
        memory.clear_history()
        self.addCleanup(memory.clear_history)

    def tearDown(self):
        agent._remember_task_result(None)

    def test_records_trace_and_verification(self):
        result = mock.Mock(status="completed", summary="created the folder",
                           detail="", evidence=[{"source": "fs"}],
                           trace=[{"tool": "code.create_folder", "ok": True}],
                           verification=["folder exists"])
        agent._remember_task_result(result, "create demo folder")
        with mock.patch.object(brain.memory_store, "record_task_outcome") as record:
            brain._record_native_task_outcome("fallback")
        self.assertTrue(record.called)
        kwargs = record.call_args.kwargs
        args = record.call_args.args
        engine, status, text = args[0], args[1], args[2]
        self.assertEqual(engine, "task_agent")
        self.assertEqual(status, "completed")
        self.assertEqual(text, "create demo folder")
        self.assertEqual(kwargs.get("trace"),
                         [{"tool": "code.create_folder", "ok": True}])
        self.assertEqual(kwargs.get("verification"), ["folder exists"])

    def test_records_nothing_when_no_run_happened(self):
        agent._remember_task_result(None)
        with mock.patch.object(brain.memory_store, "record_task_outcome") as record:
            brain._record_native_task_outcome("nothing ran")
        record.assert_not_called()

    def test_unknown_status_is_not_recorded(self):
        agent._remember_task_result(mock.Mock(status="thinking", summary="",
                                              detail="", evidence=[],
                                              trace=[], verification=[]), "x")
        with mock.patch.object(brain.memory_store, "record_task_outcome") as record:
            brain._record_native_task_outcome("x")
        record.assert_not_called()

    def test_failed_run_is_recorded_but_captures_no_procedure(self):
        result = mock.Mock(status="failed", summary="it broke", detail="",
                           evidence=[], trace=[{"tool": "x", "ok": False}],
                           verification=[])
        agent._remember_task_result(result, "do the thing")
        with mock.patch.object(brain.memory_store, "record_task_outcome") as record:
            brain._record_native_task_outcome("do the thing")
        self.assertEqual(record.call_args.args[1], "failed")
        self.assertEqual(record.call_args.kwargs.get("trace"),
                         [{"tool": "x", "ok": False}])

    def test_recording_never_raises_into_the_turn(self):
        agent._remember_task_result(mock.Mock(status="completed", summary="",
                                              detail="", evidence=[],
                                              trace=[], verification=[]), "x")
        with mock.patch.object(brain.memory_store, "record_task_outcome",
                               side_effect=RuntimeError("db gone")):
            brain._record_native_task_outcome("x")  # must not raise


class ToolHistoryTests(unittest.TestCase):
    """R19 - a completed tool run leaves one compact note in the history."""

    def setUp(self):
        memory.clear_history()
        self.addCleanup(memory.clear_history)

    def tearDown(self):
        agent._remember_task_result(None)

    def _run(self, status, summary,
             task_text="create a file called notes.txt"):
        result = mock.Mock(status=status, summary=summary, detail="",
                           evidence=[], trace=[], verification=[])
        agent._remember_task_result(result, task_text)
        with mock.patch.object(brain.memory_store, "record_task_outcome"):
            brain._record_native_task_outcome(task_text)

    def test_a_completed_native_run_joins_the_history(self):
        self._run("completed", "Created notes.txt at C:\\Users\\me\\notes.txt")
        self.assertEqual(
            memory.get_history()[-1],
            {"role": "assistant",
             "content": "[task] create a file called notes.txt — "
                        "Created notes.txt at C:\\Users\\me\\notes.txt"})

    def test_a_needs_input_turn_is_recallable(self):
        self._run("needs_input", "Ready to create notes.txt — shall I?")
        self.assertIn(
            "[task] create a file called notes.txt — Ready to create",
            memory.get_history()[-1]["content"])

    def test_the_note_is_bounded_and_keeps_the_outcome(self):
        self._run("completed", "Created the file. " + ("more detail " * 40),
                  task_text="x" * 300)
        content = memory.get_history()[-1]["content"]
        self.assertLessEqual(
            len(content), len("[task] ") + brain._TOOL_SUMMARY_MAX + 1)
        self.assertIn("Created the file.", content)

    def test_a_repeat_does_not_stack(self):
        self._run("completed", "Created notes.txt")
        self._run("completed", "Created notes.txt")
        entries = [m for m in memory.get_history()
                   if m["content"].startswith("[task]")]
        self.assertEqual(len(entries), 1)

    def test_a_later_different_run_still_joins(self):
        self._run("completed", "Created notes.txt")
        self._run("completed", "Created todo.txt", task_text="create todo.txt")
        self.assertEqual(len(memory.get_history()), 2)
        self.assertIn("todo.txt", memory.get_history()[-1]["content"])

    def test_a_turn_that_ran_nothing_records_nothing(self):
        agent._remember_task_result(None)
        brain._record_native_task_outcome("nothing ran")
        self.assertEqual(memory.get_history(), [])

    def test_an_unknown_status_records_nothing(self):
        self._run("thinking", "still going")
        self.assertEqual(memory.get_history(), [])

    def test_the_marker_names_the_activity_kind(self):
        brain._remember_tool_summary("research", "Found two sources.")
        self.assertEqual(memory.get_history()[-1]["content"],
                         "[research] Found two sources.")


if __name__ == "__main__":
    unittest.main()
