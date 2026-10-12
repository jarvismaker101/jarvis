"""LIVE FIX 22: repair the five live-test bugs (B9-B13 + minimize exclusion).

Wave 1 (B9): element-level control chains classify ("click … on my
screen" is a control, backspace/type-content/ordinal-play match, bare
"search" after a control step submits) and plan instantly.
Wave 2 (minimize exclusion): qualified window commands build a
deterministic per-window plan honoring "except X".
Wave 3 (B10): complaints never route to research; unfinished chains
resume.
Wave 4 (B11): requests register at dispatch, so "Again." retries the
last request even when it failed.
Wave 5 (B12/B13): last-action recall answers from the registry; "request"
is a recall noun.
"""

import unittest
from unittest.mock import patch

from backend.core import brain
from backend.services import multi_intent
from backend.services import screen_control
from backend.services import screen_state


def _windows():
    return [
        {"hwnd": 11, "title": "New Tab - Brave", "iconic": False},
        {"hwnd": 22, "title": "Untitled - Notepad", "iconic": False},
        {"hwnd": 33, "title": "Terminal - Ubuntu", "iconic": False},
    ]


class B9ChainClassificationTests(unittest.TestCase):
    def test_live_utterance_is_a_five_step_control_chain(self):
        plan = multi_intent.build_chain(
            "click in the search bar on my screen , backspace whatever "
            "is written in it , type  mr.beast and then search and then "
            "play the first viideo")
        self.assertIsNotNone(plan)
        kinds = [s["kind"] for s in plan["steps"]]
        self.assertEqual(kinds, ["screen_control"] * 5)
        texts = [s["text"] for s in plan["steps"]]
        self.assertEqual(texts[0], "click in the search bar on my screen")
        self.assertEqual(texts[3], "press enter")

    def test_control_on_my_screen_is_not_a_look(self):
        self.assertEqual(
            multi_intent.classify_clause(
                "click in the search bar on my screen"), "screen_control")

    def test_look_clauses_still_look(self):
        self.assertEqual(
            multi_intent.classify_clause("look at my screen"), "screen")
        self.assertEqual(
            multi_intent.classify_clause(
                "what is running on my screen right now"), "screen")

    def test_task_and_tool_shapes_are_not_stolen(self):
        self.assertEqual(
            multi_intent.classify_clause("delete the folder named x"),
            "task")
        self.assertEqual(multi_intent.classify_clause("copy it"), "task")
        self.assertEqual(
            multi_intent.classify_clause("type the report in the file"),
            None)
        self.assertEqual(
            multi_intent.classify_clause("play despacito"), "tool")
        self.assertEqual(
            multi_intent.classify_clause("play the first video"),
            "screen_control")

    def test_ack_narrates_each_step(self):
        plan = multi_intent.build_chain(
            "click in the search bar on my screen , backspace whatever "
            "is written in it , type  mr.beast and then search and then "
            "play the first viideo")
        ack = multi_intent.render_ack(plan, voice_compact=True)
        self.assertIn("type mr.beast", ack)
        self.assertNotIn("control your screen, control your screen", ack)


class B9DirectPlanTests(unittest.TestCase):
    def setUp(self):
        self._was_enabled = screen_state.is_enabled()
        screen_state.set_enabled(True)

    def tearDown(self):
        screen_state.set_enabled(self._was_enabled)
        screen_state.clear_pending_plan()

    def test_bare_type_plans_instantly(self):
        plan = screen_control.plan_command("type mr.beast")
        self.assertTrue(plan.get("ok"))
        self.assertEqual(plan["steps"], [{"action": "type",
                                          "text": "mr.beast"}])

    def test_backspace_content_clears_the_field(self):
        plan = screen_control.plan_command(
            "backspace whatever is written in it")
        self.assertTrue(plan.get("ok"))
        self.assertEqual(plan["steps"],
                         [{"action": "press", "keys": ["ctrl", "a"]},
                          {"action": "press", "keys": ["delete"]}])

    def test_bare_backspace_is_one_keypress(self):
        plan = screen_control.plan_command("backspace")
        self.assertTrue(plan.get("ok"))
        self.assertEqual(plan["steps"],
                         [{"action": "press", "keys": ["backspace"]}])

    def test_chat_type_shapes_stay_out(self):
        self.assertFalse(
            screen_control._looks_like_screen_command("write to me a poem"))

    def test_play_ordinal_step_rephrases_to_click(self):
        with patch.object(brain, "plan_command",
                          return_value={"ok": True}) as planned, \
             patch.object(brain, "_execute_plan_tagged",
                          return_value=("executed", "Clicked it, sir.")):
            rec = brain._mi_screen_control_step(
                {"kind": "screen_control", "text": "play the first video"})
        self.assertEqual(rec.get("status"), "ok")
        self.assertTrue(
            planned.call_args[0][0].lower().startswith("click"),
            planned.call_args[0][0])


class MinimizeExclusionTests(unittest.TestCase):
    def test_except_brave_leaves_brave(self):
        with patch.object(screen_control, "_list_managable_windows",
                          return_value=_windows()):
            plan = screen_control._build_direct_plan(
                "minimize all the windows on my screen except for the "
                "brave browser")
        self.assertTrue(plan.get("ok"))
        hwnds = sorted(s["hwnd"] for s in plan["steps"])
        self.assertEqual(hwnds, [22, 33])
        for step in plan["steps"]:
            self.assertEqual(step["action"], "minimize_hwnd")
        self.assertIn("Brave", plan["summary"])

    def test_app_filter_selects_only_matching_windows(self):
        with patch.object(screen_control, "_list_managable_windows",
                          return_value=_windows()):
            plan = screen_control._build_direct_plan(
                "minimize all the terminal windows")
        self.assertTrue(plan.get("ok"))
        self.assertEqual([s["hwnd"] for s in plan["steps"]], [33])

    def test_empty_except_clause_is_honest(self):
        with patch.object(screen_control, "_list_managable_windows",
                          return_value=_windows()):
            plan = screen_control._build_direct_plan(
                "minimize all the windows except for the")
        self.assertFalse(plan.get("ok"))
        self.assertIn("which app", plan.get("reason", ""))

    def test_everything_excluded_is_honest_success(self):
        with patch.object(screen_control, "_list_managable_windows",
                          return_value=[_windows()[0]]):
            plan = screen_control._build_direct_plan(
                "minimize all the windows except brave")
        self.assertTrue(plan.get("ok"))
        self.assertEqual(plan["steps"], [])

    def test_bare_minimize_still_targets_foreground(self):
        plan = screen_control._parse_window_command(
            "minimize the window on my screen")
        self.assertEqual(plan["steps"], [{"action": "minimize_window"}])

    def test_dead_hwnd_is_skipped_not_fatal(self):
        from backend.services import screen_executor as executor
        with patch.object(executor, "user32") as fake_u32:
            fake_u32.IsWindow.return_value = 0
            self.assertFalse(
                executor.set_window_state_by_hwnd(999999, 6))
        with self.assertRaises(RuntimeError):
            executor.set_window_state_by_hwnd(0, 6)


class ComplaintRoutingTests(unittest.TestCase):
    def test_complaint_is_not_search_shaped(self):
        self.assertFalse(brain.is_search_shaped_message(
            "you only stopped after clicking in the search bar but "
            "didnt do the next steps that i told"))

    def test_real_searches_still_route(self):
        self.assertTrue(brain.is_search_shaped_message(
            "can you search about this stream"))
        self.assertTrue(brain.is_search_shaped_message(
            "research quantum computing news"))

    def test_resume_with_no_chain_is_honest(self):
        brain._remember_chain_plan(None, None)
        self.assertEqual(
            brain._resume_chain_unfinished(
                "you only stopped after clicking, do the rest"),
            "Sir, I don't have any unfinished steps from that — say it "
            "once more and I'll run it from the top.")
        self.assertIsNone(
            brain._resume_chain_unfinished("continue"))

    def test_resume_reruns_only_the_unfinished_tail(self):
        plan = {"ok": True, "source": "click the search bar then type hi",
                "command_text": "click the search bar then type hi",
                "steps": [{"kind": "screen_control",
                           "text": "click the search bar"},
                          {"kind": "screen_control", "text": "type hi"}]}
        brain._remember_chain_plan(plan, [
            {"kind": "screen_control", "status": "ok",
             "fragment": "Clicked it, sir."},
            {"kind": "screen_control", "status": "failed",
             "fragment": "That screen action failed."},
        ])
        try:
            with patch.object(brain, "_run_multi_intent_chain") as worker:
                reply = brain._resume_chain_unfinished(
                    "you didn't do the next steps", voice_compact=True)
            self.assertIn("continuing where it stopped", reply)
            args = worker.call_args[0]
            # Prior results carry only the finished step; work restarts
            # at the failed one.
            self.assertEqual(len(args[2]), 1)
            self.assertEqual(args[3], 1)
            # Repeat points at the original request, not the complaint.
            self.assertEqual(
                brain.get_last_action()["command"],
                "click the search bar then type hi")
        finally:
            brain._remember_chain_plan(None, None)
            with brain._MI_CHAIN_LOCK:
                brain._mi_chain_active = False


class RepeatIntegrityTests(unittest.TestCase):
    def setUp(self):
        self._saved = dict(brain.get_last_action())

    def tearDown(self):
        try:
            with brain._last_action_lock:
                brain._last_action.update(self._saved)
        except Exception:
            pass

    def test_request_registers_before_the_outcome(self):
        brain.note_executed_action(
            "screen_control",
            "click on the instagram tab at the top of my screen", "")
        last = brain.get_last_action()
        self.assertEqual(
            last["command"],
            "click on the instagram tab at the top of my screen")
        self.assertFalse(last["done"])
        self.assertEqual(
            brain._repeatable_last_command(),
            "click on the instagram tab at the top of my screen")

    def test_finished_request_marks_done(self):
        brain.note_executed_action(
            "screen_control", "click at home button on my screen",
            "Clicked it, sir.")
        self.assertTrue(brain.get_last_action()["done"])


class LastActionRecallTests(unittest.TestCase):
    def setUp(self):
        self._saved = dict(brain.get_last_action())

    def tearDown(self):
        try:
            with brain._last_action_lock:
                brain._last_action.update(self._saved)
        except Exception:
            pass

    def test_finished_action_reports_request_and_result(self):
        brain.note_executed_action(
            "screen_control", "click at home button on my screen",
            "Clicked it, sir.")
        reply = brain._answer_last_action_question(
            "what was my last screen control request to you?")
        self.assertIn("click at home button on my screen", reply)
        self.assertIn("Clicked it, sir.", reply)

    def test_unfinished_action_reports_honestly(self):
        brain.note_executed_action(
            "screen_control",
            "click on the instagram tab at the top of my screen", "")
        reply = brain._answer_last_action_question(
            "what was my last screen control request to you?")
        self.assertIn("click on the instagram tab", reply)
        self.assertIn("again", reply)

    def test_empty_registry_is_honest(self):
        try:
            with brain._last_action_lock:
                brain._last_action.update(
                    {"kind": "", "command": "", "summary": "",
                     "done": False, "seq": 0, "at": 0.0})
        except Exception:
            pass
        with patch(
                "backend.services.screen_state.get_recent_interactions",
                return_value=[]):
            reply = brain._answer_last_action_question(
                "what was my last screen control request to you?")
        self.assertIn("don't have", reply)

    def test_non_recall_gets_no_answer(self):
        self.assertIsNone(
            brain._answer_last_action_question("what is the weather like?"))

    def test_request_is_a_recall_noun(self):
        self.assertTrue(brain._memory_recall_request(
            "what was my 4th last request to you?"))
        self.assertTrue(brain._memory_recall_request(
            "tell me what I asked in the second last reply"))


if __name__ == "__main__":
    unittest.main()
