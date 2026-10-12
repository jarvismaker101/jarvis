"""LIVE FIX 24: repair the re-test bugs (B18-B24).

B18: counted/app-named window targets ("minimize the two terminal
windows", "minimize the brave window") build deterministic per-window
plans — the bare active-window plan answers bare objects only.
B19: positional references ("minimize the first two that you mentioned")
resolve against the stored enumeration, honestly refusing when the list
is gone.
B20: "not the active window, the terminal window" re-targets the previous
window action instead of reaching chat.
B21: chain-armed approvals live longer; a late "confirm" gets the honest
expiry reply; confirming a chain step resumes the tail; chat can no
longer narrate screen actions ("Clicking ...").
B22: "search what you typed" submits (press enter), a leading "not X"
fragment merges instead of researching, "search icon" is a UI noun, and
a restated submit is one step.
B23: a chain research step registers with the research-stop mechanism, so
a stop aborts it mid-flight and the result reports "stopped".
B24: a question ABOUT an action ("can you identify the windows ... that
you are trying to minimize") routes to screen Q&A, never the planner.
"""

import threading
import time
import unittest
from unittest.mock import patch

from backend.core import brain
from backend.services import approvals
from backend.services import multi_intent
from backend.services import screen_control
from backend.services import screen_state
from backend.services.screen_analyzer import is_screen_question


def _windows():
    return [
        {"hwnd": 11, "title": "New Tab - Brave", "iconic": False},
        {"hwnd": 22, "title": "Untitled - Notepad", "iconic": False},
        {"hwnd": 33, "title": "Terminal - Ubuntu", "iconic": False},
        {"hwnd": 44, "title": "Terminal - PowerShell", "iconic": False},
    ]


class B18CountedWindowTargetsTests(unittest.TestCase):
    def setUp(self):
        screen_control._clear_window_enumeration()

    def tearDown(self):
        screen_control._clear_window_enumeration()

    def test_bare_minimize_still_active_window(self):
        plan = screen_control._parse_window_command(
            "minimize the window on my screen")
        self.assertEqual(plan["steps"], [{"action": "minimize_window"}])

    def test_counted_target_defers_from_active_window(self):
        self.assertIsNone(screen_control._parse_window_command(
            "minimize the two terminal windows on my screen"))

    def test_two_terminal_windows_builds_two_steps(self):
        with patch.object(screen_control, "_list_managable_windows",
                          return_value=_windows()):
            plan = screen_control._build_direct_plan(
                "minimize the two terminal windows on my screen")
        self.assertTrue(plan.get("ok"))
        self.assertEqual([s["hwnd"] for s in plan["steps"]], [33, 44])
        for step in plan["steps"]:
            self.assertEqual(step["action"], "minimize_hwnd")

    def test_count_exceeding_matches_is_honest(self):
        with patch.object(screen_control, "_list_managable_windows",
                          return_value=_windows()[:3]):
            plan = screen_control._build_direct_plan(
                "minimize the two terminal windows on my screen")
        self.assertFalse(plan.get("ok"))
        self.assertIn("only 1", plan.get("reason", ""))

    def test_app_named_singular_takes_the_first_match(self):
        with patch.object(screen_control, "_list_managable_windows",
                          return_value=_windows()):
            plan = screen_control._build_direct_plan(
                "minimize the brave window")
        self.assertTrue(plan.get("ok"))
        self.assertEqual([s["hwnd"] for s in plan["steps"]], [11])

    def test_app_named_plural_takes_all_matches(self):
        with patch.object(screen_control, "_list_managable_windows",
                          return_value=_windows()):
            plan = screen_control._build_direct_plan(
                "minimize the terminal windows")
        self.assertTrue(plan.get("ok"))
        self.assertEqual([s["hwnd"] for s in plan["steps"]], [33, 44])

    def test_unknown_app_is_honest(self):
        with patch.object(screen_control, "_list_managable_windows",
                          return_value=_windows()):
            plan = screen_control._build_direct_plan(
                "minimize the calculator window")
        self.assertFalse(plan.get("ok"))
        self.assertIn("calculator", plan.get("reason", ""))


class B19ReferencedWindowTargetsTests(unittest.TestCase):
    def setUp(self):
        screen_control._clear_window_enumeration()

    def tearDown(self):
        screen_control._clear_window_enumeration()

    def test_first_two_resolves_from_the_stored_list(self):
        screen_control.remember_window_enumeration(_windows())
        with patch.object(screen_control, "_list_managable_windows",
                          return_value=_windows()):
            plan = screen_control._build_direct_plan(
                "minimize the first two that you mentioned")
        self.assertTrue(plan.get("ok"), plan.get("reason"))
        self.assertEqual([s["hwnd"] for s in plan["steps"]], [11, 22])

    def test_reference_with_app_filter(self):
        screen_control.remember_window_enumeration(_windows())
        plan = screen_control._build_direct_plan(
            "minimize the first terminal that you mentioned")
        self.assertTrue(plan.get("ok"), plan.get("reason"))
        self.assertEqual([s["hwnd"] for s in plan["steps"]], [33])

    def test_missing_list_is_honest(self):
        plan = screen_control._build_direct_plan(
            "minimize the first two that you mentioned")
        self.assertFalse(plan.get("ok"))
        self.assertIn("window list", plan.get("reason", ""))


class B20WindowTargetCorrectionTests(unittest.TestCase):
    def setUp(self):
        self._saved = brain.get_last_action()

    def tearDown(self):
        with brain._last_action_lock:
            brain._last_action.update(dict(self._saved))

    def test_not_the_active_window_retargets(self):
        brain.note_executed_action(
            "screen_control",
            "minimize the two terminal windows on my screen",
            "Minimizing the active window, sir.")
        self.assertEqual(
            brain._screen_target_correction_command(
                "not the active window, the terminal window"),
            "minimize the terminal window")

    def test_no_previous_window_action_is_not_a_correction(self):
        with brain._last_action_lock:
            brain._last_action.update(
                {"kind": "", "command": "", "summary": "",
                 "done": False, "seq": 0, "at": 0.0})
        with patch("backend.services.screen_state.get_recent_interactions",
                    return_value=[]):
            self.assertIsNone(brain._screen_target_correction_command(
                "not the active window, the terminal window"))

    def test_recall_turn_is_never_a_correction(self):
        brain.note_executed_action(
            "screen_control", "minimize the window on my screen",
            "Minimizing the active window, sir.")
        self.assertIsNone(brain._screen_target_correction_command(
            "no about zed new ai ide , tell me exactly what you just told"))


class B21ConfirmPathTests(unittest.TestCase):
    def setUp(self):
        self._was_enabled = screen_state.is_enabled()
        screen_state.set_enabled(True)

    def tearDown(self):
        screen_state.clear_pending_plan()
        screen_state.set_enabled(self._was_enabled)
        approvals.clear()
        try:
            approvals._last_expired.update({"command": "", "at": 0.0})
        except Exception:
            pass
        brain._remember_chain_plan(None, None)
        with brain._MI_CHAIN_LOCK:
            brain._mi_chain_active = False
        brain._mi_chain_stop.clear()

    def test_chain_armed_approval_outlives_single_turn(self):
        plan = {"ok": True, "confidence": 0.6, "summary": "Clicking, sir.",
                "needs_confirmation": False, "reason": "",
                "steps": [{"action": "click", "x": 1, "y": 1}],
                "screen_generation": screen_state.generation()}
        step = {"kind": "screen_control", "text": "click the icon",
                "index": 0, "consumes": []}
        with patch.object(brain, "plan_command", return_value=plan):
            out = brain._mi_screen_control_step(
                step, chain_plan={"source": "x", "command_text": "x",
                                  "steps": [step]},
                chain_index=0)
        self.assertEqual(out["status"], "armed")
        record = approvals.pending()
        self.assertIsNotNone(record)
        self.assertGreater(record.remaining(), 200)

    def test_late_confirm_gets_the_honest_expiry_reply(self):
        screen_state.clear_pending_plan()
        approvals.arm({"steps": []}, "click the icon", ttl=-1)
        self.assertIsNone(approvals.pending())
        reply = screen_control.maybe_handle_screen_control_message("confirm")
        self.assertIn("confirmation has closed", reply)

    def test_chat_cannot_narrate_a_click(self):
        out = brain._strip_unverified_action_claims(
            "Clicking 'Tab search', sir.", "chat")
        # LIVE FIX 27 / B50: the fallback is neutral (it must never assert
        # nothing was started) — the gate still speaks no claim.
        self.assertEqual(out, brain._CHAT_SCREEN_FALLBACK)

    def test_confirmed_chain_step_resumes_the_tail(self):
        plan = {"ok": True, "source": "click then type",
                "command_text": "click then type",
                "steps": [{"kind": "screen_control",
                            "text": "click the search bar",
                            "index": 0, "consumes": []},
                           {"kind": "screen_control", "text": "type hi",
                            "index": 1, "consumes": []}]}
        brain._remember_chain_plan(plan, [
            {"kind": "screen_control", "status": "armed",
             "fragment": "Say confirm screen action to proceed.",
             "prompt": "Say confirm screen action to proceed."},
        ])
        started = threading.Event()
        calls = {}

        def fake_worker(msg, _plan, prior, tail_from=0):
            calls["prior"] = prior
            calls["tail_from"] = tail_from
            calls["plan"] = _plan
            started.set()

        with patch.object(brain, "_run_multi_intent_chain",
                           side_effect=fake_worker):
            brain._on_chain_plan_confirmed(
                {"signature": brain._chain_signature(plan),
                 "step_index": 0},
                "Clicked it, sir.")
            self.assertTrue(started.wait(2.0))
        self.assertEqual(len(calls["prior"]), 1)
        self.assertEqual(calls["prior"][0]["status"], "ok")
        self.assertEqual(calls["tail_from"], 1)
        self.assertEqual(calls["plan"]["source"], "click then type")


class B22SubmitCorrectionTests(unittest.TestCase):
    def test_search_what_you_typed_is_a_control(self):
        self.assertEqual(
            multi_intent.classify_clause(
                "i simply want you to search what you typed"),
            "screen_control")

    def test_live_correction_becomes_one_press_enter(self):
        plan = multi_intent.build_chain(
            "not tab search , i simply want you to search what you typed , "
            "you can just do it by clicking enter or the search icon next "
            "to search bar")
        self.assertIsNone(plan)
        direct = screen_control._build_direct_plan("press enter")
        self.assertTrue(direct.get("ok"))
        self.assertEqual(direct["steps"],
                         [{"action": "press", "keys": ["enter"]}])

    def test_search_icon_grows_no_research_sibling(self):
        plan = multi_intent.build_chain(
            "click the search icon and then type hello")
        self.assertIsNotNone(plan)
        self.assertEqual([s["kind"] for s in plan["steps"]],
                         ["screen_control", "screen_control"])


class B23ChainResearchStopTests(unittest.TestCase):
    def tearDown(self):
        brain._mi_chain_stop.clear()
        with brain._MI_CHAIN_LOCK:
            brain._mi_chain_active = False
        try:
            from backend.services import research_service as _rs
            _rs.clear_stop_request()
        except Exception:
            pass

    def test_midflight_lookup_reports_stopped(self):
        seen = {}

        def fake_search(query):
            seen["running"] = brain._research_running
            brain._mi_chain_stop.set()
            return {"query": query, "spoken_summary": "Sir, hi",
                    "overview_text": "hi"}

        with patch.object(brain, "run_quick_search",
                           side_effect=fake_search), \
             patch.object(brain, "_resolve_search_query",
                          return_value="mr.beast"):
            result = brain._mi_research_step(
                {"text": "look up x", "consumes": []}, [])
        self.assertTrue(seen.get("running"))
        self.assertFalse(brain._research_running)
        self.assertEqual(result["status"], "stopped")

    def test_stop_message_signals_chain_research(self):
        from backend.services import research_service as _rs
        with brain._MI_CHAIN_LOCK:
            brain._mi_chain_active = True
        brain._mi_chain_stop.clear()
        try:
            with patch.object(brain, "opencode_task_in_progress",
                               return_value=False):
                brain.handle_stop_message("stop the task")
            self.assertTrue(_rs.stop_requested())
        finally:
            with brain._MI_CHAIN_LOCK:
                brain._mi_chain_active = False
            brain._mi_chain_stop.clear()


class B24QuestionAboutActionTests(unittest.TestCase):
    def setUp(self):
        self._was_enabled = screen_state.is_enabled()
        screen_state.set_enabled(True)

    def tearDown(self):
        screen_state.set_enabled(self._was_enabled)
        screen_state.clear_pending_plan()

    def test_identify_windows_is_a_question_not_a_command(self):
        msg = ("can you identify the three windows other than brave that "
               "you are trying to minimize")
        self.assertFalse(screen_control._looks_like_screen_command(msg))
        self.assertTrue(is_screen_question(msg))

    def test_real_window_commands_still_route(self):
        self.assertTrue(screen_control._looks_like_screen_command(
            "could you close the settings tab"))
        self.assertTrue(screen_control._looks_like_screen_command(
            "minimize the window on my screen"))


if __name__ == "__main__":
    unittest.main()
