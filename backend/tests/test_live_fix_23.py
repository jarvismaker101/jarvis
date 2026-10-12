"""LIVE FIX 23: repair the four re-test bugs (B14-B17).

B14: "click search" (control verb + submit target) is a screen click —
never a research step with the raw clause as its query.
B15: "stop the task" routes to the real stop handler (not chat), and a
running multi-intent chain actually aborts between steps with an honest
"Stopped by your request" summary.
B16: cross-thread minimize uses SW_FORCEMINIMIZE and verifies IsIconic;
a window that refuses its state lands in an honest partial report, and
the kept-window ack names the app ("Leaving Brave.").
B17: a chain with screen_control work refuses honestly at ack time when
screen controls are off — no worker starts.
"""

import unittest
from unittest.mock import patch

from backend.core import brain
from backend.services import multi_intent
from backend.services import screen_control
from backend.services import screen_state


def _windows():
    return [
        {"hwnd": 11, "title": "(22) YouTube - Brave", "iconic": False},
        {"hwnd": 22, "title": "Untitled - Notepad", "iconic": False},
    ]


class B14ClickSearchTests(unittest.TestCase):
    def test_live_utterance_has_no_research_step(self):
        plan = multi_intent.build_chain(
            "click in the youtube search bar on my screen , then type "
            "mr.beast and click search , then play the first video "
            "that appears")
        self.assertIsNotNone(plan)
        kinds = [s["kind"] for s in plan["steps"]]
        self.assertEqual(kinds, ["screen_control"] * 4)
        texts = [s["text"] for s in plan["steps"]]
        self.assertIn("click search", texts)

    def test_click_search_shapes_are_controls(self):
        for clause in ("click search", "press search",
                       "tap the submit button", "click the search"):
            self.assertEqual(
                multi_intent.classify_clause(clause), "screen_control",
                clause)

    def test_real_lookups_still_research(self):
        for clause in ("search the internet for flights",
                       "google it", "research quantum computing news"):
            self.assertEqual(
                multi_intent.classify_clause(clause), "research", clause)
        self.assertEqual(
            multi_intent.classify_clause("play despacito"), "tool")

    def test_bare_submit_after_control_unchanged(self):
        plan = multi_intent.build_chain("type mr.beast and then search")
        self.assertIsNotNone(plan)
        self.assertEqual(
            [(s["kind"], s["text"]) for s in plan["steps"]],
            [("screen_control", "type mr.beast"),
             ("screen_control", "press enter")])

    def test_no_bogus_research_sibling_for_click_search(self):
        plan = multi_intent.build_chain("type mr.beast and click search")
        self.assertIsNotNone(plan)
        self.assertEqual(
            [s["kind"] for s in plan["steps"]],
            ["screen_control", "screen_control"])

    def test_real_lookup_sibling_survives(self):
        plan = multi_intent.build_chain(
            "click the search bar to search for flights")
        self.assertIsNotNone(plan)
        self.assertIn("research", [s["kind"] for s in plan["steps"]])

    def test_click_search_passes_command_gate(self):
        was_enabled = screen_state.is_enabled()
        screen_state.set_enabled(True)
        try:
            self.assertTrue(
                screen_control._looks_like_screen_command("click search"))
        finally:
            screen_state.set_enabled(was_enabled)
            screen_state.clear_pending_plan()


class B15StopTaskTests(unittest.TestCase):
    def test_stop_task_shapes_match(self):
        for phrase in ("stop the task", "Stop the task.", "stop the chain",
                       "cancel the task", "stop working", "stop this",
                       "jarvis stop the task please", "halt the job"):
            self.assertTrue(brain.is_stop_task_target(phrase), phrase)

    def test_stop_task_shapes_do_not_steal(self):
        for phrase in ("stop the research", "stop the browser task",
                       "don't stop the task", "stop the task and open youtube",
                       "stop searching for that", "stop the task manager",
                       "what task"):
            self.assertFalse(brain.is_stop_task_target(phrase), phrase)

    def test_stopped_chain_runs_nothing(self):
        plan = {"ok": True, "source": "do three things",
                "command_text": "do three things",
                "steps": [{"kind": "screen_control", "text": "click x"},
                          {"kind": "research", "text": "look up y"},
                          {"kind": "screen_control", "text": "type z"}]}
        brain._mi_chain_stop.set()
        try:
            with patch.object(brain, "_mi_screen_control_step") as sc, \
                    patch.object(brain, "_mi_research_step") as rs, \
                    patch.object(brain, "_notify_async_reply") as notify:
                sc.side_effect = AssertionError("must not run")
                rs.side_effect = AssertionError("must not run")
                brain._run_multi_intent_chain("do three things", plan)
            sc.assert_not_called()
            rs.assert_not_called()
            notify.assert_called_once()
            self.assertIn("Stopped by your request", notify.call_args[0][0])
        finally:
            brain._mi_chain_stop.clear()
            with brain._MI_CHAIN_LOCK:
                brain._mi_chain_active = False

    def test_research_step_honors_stop(self):
        brain._mi_chain_stop.set()
        try:
            with patch.object(brain, "run_quick_search") as qs:
                qs.side_effect = AssertionError("must not search")
                result = brain._mi_research_step(
                    {"text": "click search", "consumes": []}, [])
            qs.assert_not_called()
            self.assertEqual(result["status"], "stopped")
        finally:
            brain._mi_chain_stop.clear()

    def test_stop_message_cancels_active_chain(self):
        with brain._MI_CHAIN_LOCK:
            brain._mi_chain_active = True
        brain._mi_chain_stop.clear()
        try:
            with patch.object(brain, "opencode_task_in_progress",
                              return_value=False):
                reply = brain.handle_stop_message("stop the task")
            self.assertEqual(reply, "Stopping, sir.")
            self.assertTrue(brain._mi_chain_stop.is_set())
        finally:
            with brain._MI_CHAIN_LOCK:
                brain._mi_chain_active = False
            brain._mi_chain_stop.clear()


class B16MinimizeVerifyTests(unittest.TestCase):
    def test_minimize_forces_and_verifies(self):
        from backend.services import screen_executor as executor
        with patch.object(executor, "user32") as fake_u32:
            fake_u32.IsWindow.return_value = 1
            fake_u32.IsIconic.return_value = 1
            self.assertTrue(
                executor.set_window_state_by_hwnd(1234, 6))
            args = fake_u32.ShowWindow.call_args[0]
            self.assertEqual(tuple(args), (1234, 11))

    def test_stubborn_window_is_reported_not_claimed(self):
        from backend.services import screen_executor as executor
        with patch.object(executor, "user32") as fake_u32, \
                patch.object(executor.time, "sleep"):
            fake_u32.IsWindow.return_value = 1
            fake_u32.IsIconic.return_value = 0
            failed = []
            self.assertFalse(
                executor.set_window_state_by_hwnd(1234, 6, failed=failed))
            self.assertEqual(failed, [1234])

    def test_dead_hwnd_stays_silent_skip(self):
        from backend.services import screen_executor as executor
        with patch.object(executor, "user32") as fake_u32:
            fake_u32.IsWindow.return_value = 0
            failed = []
            self.assertFalse(
                executor.set_window_state_by_hwnd(999999, 6,
                                                  failed=failed))
            self.assertEqual(failed, [])

    def test_partial_minimize_reports_honestly(self):
        from backend.services import screen_executor as executor
        steps = [{"action": "minimize_hwnd", "hwnd": 11, "title": "App A"},
                 {"action": "minimize_hwnd", "hwnd": 22, "title": "App B"}]
        with patch.object(executor, "user32") as fake_u32:
            fake_u32.IsWindow.return_value = 1
            fake_u32.IsIconic.side_effect = lambda h: 1 if h == 11 else 0
            with patch.object(executor.time, "sleep"):
                with self.assertRaises(RuntimeError) as ctx:
                    executor.execute_steps(steps)
        self.assertIn("1 of 2", str(ctx.exception))
        self.assertIn("App B", str(ctx.exception))

    def test_hwnd_state_postcondition(self):
        with patch("ctypes.windll") as mock_windll:
            mock_windll.user32.IsWindow.return_value = 1
            mock_windll.user32.IsIconic.side_effect = lambda h: h == 11
            steps = [{"action": "minimize_hwnd", "hwnd": 11}]
            self.assertTrue(
                screen_control._verify_hwnd_state_steps(steps))
            steps = [{"action": "minimize_hwnd", "hwnd": 22}]
            self.assertFalse(
                screen_control._verify_hwnd_state_steps(steps))
            self.assertIsNone(
                screen_control._verify_hwnd_state_steps(
                    [{"action": "click", "x": 1, "y": 2}]))

    def test_leaving_names_the_app(self):
        with patch.object(screen_control, "_list_managable_windows",
                          return_value=_windows()):
            plan = screen_control._build_direct_plan(
                "minimize all the windows on my screen except brave window")
        self.assertTrue(plan.get("ok"))
        self.assertEqual(len(plan["steps"]), 1)
        self.assertIn("Leaving Brave.", plan["summary"])
        self.assertNotIn("(22)", plan["summary"])


class B17ChainOffStateTests(unittest.TestCase):
    def setUp(self):
        self._was_enabled = screen_state.is_enabled()

    def tearDown(self):
        screen_state.set_enabled(self._was_enabled)
        screen_state.clear_pending_plan()
        brain._remember_chain_plan(None, None)
        # A mocked worker never exits, so it never resets the flag —
        # clear it here so no test observes another test's chain.
        with brain._MI_CHAIN_LOCK:
            brain._mi_chain_active = False
        brain._mi_chain_stop.clear()

    def _plan(self):
        return {"ok": True, "source": "click x then type y",
                "command_text": "click x then type y",
                "steps": [{"kind": "screen_control", "text": "click x"},
                          {"kind": "screen_control", "text": "type y"}]}

    def test_chain_ack_refuses_when_controls_off(self):
        screen_state.set_enabled(False)
        reply = brain.handle_multi_intent("click x then type y",
                                          self._plan())
        self.assertIn("screen controls are off", reply.lower())
        with brain._MI_CHAIN_LOCK:
            self.assertFalse(brain._mi_chain_active)

    def test_chain_ack_normal_when_controls_on(self):
        screen_state.set_enabled(True)
        with patch.object(brain, "_run_multi_intent_chain"):
            reply = brain.handle_multi_intent("click x then type y",
                                              self._plan())
        self.assertIn("2 steps", reply)

    def test_non_control_chain_unaffected_by_toggle(self):
        screen_state.set_enabled(False)
        plan = {"ok": True, "source": "look then research",
                "command_text": "look then research",
                "steps": [{"kind": "screen", "text": "look at my screen"},
                          {"kind": "research", "text": "research this"}]}
        with patch.object(brain, "_run_multi_intent_chain"):
            reply = brain.handle_multi_intent("look then research", plan)
        self.assertNotIn("screen controls are off", reply.lower())

    def test_resume_refuses_when_controls_off(self):
        screen_state.set_enabled(False)
        brain._remember_chain_plan(self._plan(), [
            {"kind": "screen_control", "status": "ok",
             "fragment": "Clicked it, sir."},
            {"kind": "screen_control", "status": "failed",
             "fragment": "That screen action failed."},
        ])
        reply = brain._resume_chain_unfinished("do the rest")
        self.assertIsNotNone(reply)
        self.assertIn("screen controls are off", reply.lower())


if __name__ == "__main__":
    unittest.main()
