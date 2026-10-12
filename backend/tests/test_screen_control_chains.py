"""LIVE FIX 21: chains of screen controls, just like task chains.

A multi-clause screen-control request ("minimize all the terminal
windows then click the search bar") becomes an ordered chain whose
steps each plan fresh and execute in order with the usual gates; a
step that needs consent stops the chain honestly; and a follow-up
("then type hello") extends the last finished chain instead of
starting over.
"""

import time
import unittest
from unittest.mock import patch

from backend.core import brain
from backend.services import multi_intent
from backend.services import screen_control
from backend.services import screen_state


class ClassifyControlClausesTests(unittest.TestCase):
    def test_control_shapes_classify_screen_control(self):
        for text in ("minimize all the terminal windows",
                     "click the search bar",
                     "type hello into the search box",
                     "press enter",
                     "scroll down",
                     "close the window and then close it".split(" and ")[0]):
            with self.subTest(text=text):
                self.assertEqual(
                    multi_intent.classify_clause(text), "screen_control",
                    text)

    def test_look_task_tool_clauses_are_not_stolen(self):
        self.assertEqual(
            multi_intent.classify_clause("look at my screen"), "screen")
        self.assertEqual(
            multi_intent.classify_clause("write the report in the file"),
            "task")
        self.assertEqual(
            multi_intent.classify_clause("create a folder named x"), "task")
        self.assertEqual(
            multi_intent.classify_clause("open youtube"), "tool")
        self.assertEqual(
            multi_intent.classify_clause("search the web for cats"),
            "research")

    def test_search_bar_is_a_target_not_a_lookup(self):
        # "click the search bar" contains the word "search" but acts on a
        # UI target — it must not route to web research.
        self.assertEqual(
            multi_intent.classify_clause("click the search bar"),
            "screen_control")


class BuildControlChainsTests(unittest.TestCase):
    def test_two_control_clauses_do_not_merge(self):
        plan = multi_intent.build_chain(
            "minimize all the terminal windows then click the search bar")
        self.assertIsNotNone(plan)
        self.assertEqual([s["kind"] for s in plan["steps"]],
                         ["screen_control", "screen_control"])

    def test_pronoun_step_consumes_the_previous_control_step(self):
        plan = multi_intent.build_chain(
            "minimize the terminal window and then close it")
        self.assertIsNotNone(plan)
        self.assertEqual([s["kind"] for s in plan["steps"]],
                         ["screen_control", "screen_control"])
        self.assertEqual(plan["steps"][1]["consumes"], [0])

    def test_control_clause_with_a_lookup_grows_a_research_sibling(self):
        plan = multi_intent.build_chain(
            "click the search bar to search for flights then tell me")
        self.assertIsNotNone(plan)
        self.assertEqual([s["kind"] for s in plan["steps"]][:2],
                         ["screen_control", "research"])

    def test_tool_clause_still_kills_the_chain(self):
        self.assertIsNone(multi_intent.build_chain(
            "minimize this window then open youtube"))

    def test_task_still_last(self):
        plan = multi_intent.build_chain(
            "minimize all the terminal windows then create a folder "
            "called x on my desktop")
        self.assertIsNotNone(plan)
        self.assertEqual([s["kind"] for s in plan["steps"]][-1], "task")

    def test_single_control_job_is_not_a_chain(self):
        self.assertIsNone(multi_intent.build_chain(
            "minimize the window on my screen"))


class PlanCommandTests(unittest.TestCase):
    def setUp(self):
        self._was_enabled = screen_state.is_enabled()

    def tearDown(self):
        screen_state.set_enabled(self._was_enabled)
        screen_state.clear_pending_plan()

    def test_non_command_returns_none(self):
        self.assertIsNone(
            screen_control.plan_command("what is on my screen"))
        self.assertIsNone(
            screen_control.plan_command("tell me a joke"))

    def test_bare_minimize_plans_instantly(self):
        plan = screen_control.plan_command(
            "minimize the window on my screen")
        self.assertIsNotNone(plan)
        self.assertTrue(plan.get("ok"))
        self.assertEqual(plan["steps"], [{"action": "minimize_window"}])
        self.assertEqual(plan.get("command_text"),
                         "minimize the window on my screen")

    def test_controls_off_gives_a_failed_plan_not_an_exception(self):
        screen_state.set_enabled(False)
        plan = screen_control.plan_command("click the search bar")
        self.assertIsNotNone(plan)
        self.assertFalse(plan.get("ok"))
        self.assertIn("controls are off", plan.get("reason", ""))


class TaggedExecuteTests(unittest.TestCase):
    def setUp(self):
        self._was_enabled = screen_state.is_enabled()

    def tearDown(self):
        screen_state.clear_pending_plan()
        screen_state.set_enabled(self._was_enabled)

    def _direct_minimize(self):
        # Built AFTER enabling (set_enabled bumps the generation, so a
        # plan stamped before enabling reads as revoked).
        return {"ok": True, "confidence": 0.99,
                "summary": "Minimizing the active window, sir.",
                "needs_confirmation": False, "reason": "",
                "steps": [{"action": "minimize_window"}],
                "screen_generation": screen_state.generation()}

    def test_wrapper_keeps_the_exact_strings(self):
        screen_state.set_enabled(True)
        plan = self._direct_minimize()
        with patch.object(screen_control, "execute_steps",
                          return_value=None), \
             patch.object(screen_control, "_verify_action_with_steps",
                          return_value=True):
            status, text = screen_control._execute_plan_tagged(
                dict(plan), "minimize the window")
            legacy = screen_control._execute_or_queue(
                dict(plan), "minimize the window")
        self.assertEqual(status, "executed")
        self.assertEqual(text, legacy)

    def test_low_confidence_plan_queues_and_arms(self):
        screen_state.set_enabled(True)
        plan = self._direct_minimize()
        plan["confidence"] = 0.6
        status, text = screen_control._execute_plan_tagged(
            dict(plan), "click the small icon")
        armed = screen_state.has_pending_plan()
        self.assertEqual(status, "queued")
        self.assertTrue(armed)
        self.assertIn("confirm", text.lower())

    def test_failed_plan_is_blocked(self):
        screen_state.set_enabled(True)
        status, text = screen_control._execute_plan_tagged(
            {"ok": False, "confidence": 0.0, "summary": "",
             "needs_confirmation": False, "reason": "nope",
             "steps": [],
             "screen_generation": screen_state.generation()}, "do it")
        self.assertEqual(status, "blocked")
        self.assertEqual(text, "nope")


class ChainStepTests(unittest.TestCase):
    def test_executed_step_is_ok(self):
        step = {"kind": "screen_control",
                "text": "minimize the window on my screen", "index": 0,
                "consumes": []}
        plan = {"ok": True, "confidence": 0.99,
                "summary": "Minimizing the active window, sir.",
                "needs_confirmation": False, "steps": []}
        with patch.object(brain, "plan_command", return_value=plan), \
             patch.object(brain, "_execute_plan_tagged",
                          return_value=("executed", plan["summary"])):
            out = brain._mi_screen_control_step(step)
        self.assertEqual(out["status"], "ok")
        self.assertIn("Minimizing", out["fragment"])

    def test_queued_step_is_armed(self):
        step = {"kind": "screen_control", "text": "click the icon",
                "index": 0, "consumes": []}
        plan = {"ok": True, "confidence": 0.6, "summary": "",
                "needs_confirmation": False, "steps": []}
        with patch.object(brain, "plan_command", return_value=plan), \
             patch.object(brain, "_execute_plan_tagged",
                          return_value=("queued", "Say confirm screen "
                                                 "action to proceed.")):
            out = brain._mi_screen_control_step(step)
        self.assertEqual(out["status"], "armed")
        self.assertIn("confirm", out["fragment"].lower())

    def test_unplannable_step_fails_honestly(self):
        step = {"kind": "screen_control", "text": "do the thing",
                "index": 0, "consumes": []}
        with patch.object(brain, "plan_command", return_value=None):
            out = brain._mi_screen_control_step(step)
        self.assertEqual(out["status"], "failed")

    def test_armed_step_stops_the_chain(self):
        from backend.core import memory as _mem
        saved = _mem.get_history()
        plan = {"ok": True, "steps": [
            {"kind": "screen_control", "text": "minimize the window",
             "index": 0, "consumes": []},
            {"kind": "screen_control", "text": "click the search bar",
             "index": 1, "consumes": []},
            {"kind": "screen_control", "text": "type hello",
             "index": 2, "consumes": []},
        ], "source": "x", "command_text": "x"}
        calls = []

        def fake_step(step):
            calls.append(step["text"])
            if step["index"] == 1:
                return {"kind": "screen_control", "status": "armed",
                        "fragment": "Say confirm screen action to proceed.",
                        "prompt": "Say confirm screen action to proceed."}
            return {"kind": "screen_control", "status": "ok",
                    "fragment": "done", "output_content": "done"}

        captured = {}
        try:
            with patch.object(brain, "_mi_screen_control_step",
                              side_effect=fake_step), \
                 patch.object(brain, "_notify_async_reply",
                              side_effect=lambda t, spoken=None:
                              captured.setdefault("reply", t)):
                brain._run_multi_intent_chain(
                    "minimize then click then type", plan)
        finally:
            _mem.conversation_history.clear()
            _mem.conversation_history.extend(saved)
        # Step 3 never ran; the reply carries the pending confirmation.
        self.assertEqual(calls, ["minimize the window",
                                 "click the search bar"])
        self.assertIn("confirm", captured.get("reply", "").lower())


class ChainContinuationTests(unittest.TestCase):
    def _seed(self, statuses=("ok",)):
        plan = {"ok": True, "steps": [
            {"kind": "screen_control", "text": "minimize the window",
             "index": 0, "consumes": []},
        ], "source": "minimize the window", "command_text": "x"}
        results = [{"kind": "screen_control", "status": s,
                    "fragment": "done"} for s in statuses]
        brain._remember_chain_plan(plan, results)
        return plan

    def tearDown(self):
        try:
            with brain._chain_followup_lock:
                brain._last_chain.update(
                    {"plan": None, "results": None, "at": 0.0})
        except Exception:
            pass
        try:
            with brain._MI_CHAIN_LOCK:
                brain._mi_chain_active = False
        except Exception:
            pass

    def test_followup_runs_only_the_new_step(self):
        self._seed()
        ran = []
        # A control follow-up needs controls on (LIVE FIX 23 / B17 refuses
        # honestly when they are off, instead of acking doomed steps).
        was_enabled = screen_state.is_enabled()
        screen_state.set_enabled(True)
        try:
            def fake_run(msg, plan, prior=None, tail=0):
                ran.append(([s["text"] for s in plan["steps"]], prior, tail))

            class SyncThread:
                def __init__(self, target=None, args=(), **kwargs):
                    self._target = target
                    self._args = args
                def start(self):
                    self._target(*self._args)

            with patch.object(brain, "_run_multi_intent_chain",
                              side_effect=fake_run), \
                 patch.object(brain.threading, "Thread", SyncThread):
                ack = brain._continue_chain(
                    "then click the search bar", "click the search bar")
        finally:
            screen_state.set_enabled(was_enabled)
            screen_state.clear_pending_plan()
        self.assertIsNotNone(ack)
        self.assertIn("continuing", ack.lower())
        # The worker got old steps + 1 new step, with the old result
        # pre-seeded so only the new step executes.
        self.assertEqual(len(ran), 1)
        texts, prior, tail = ran[0]
        self.assertEqual(texts, ["minimize the window",
                                 "click the search bar"])
        self.assertEqual(len(prior), 1)
        self.assertEqual(tail, 1)

    def test_no_chain_means_no_continuation(self):
        self.assertIsNone(
            brain._continue_chain("then click the search bar",
                                  "click the search bar"))

    def test_stale_chain_means_no_continuation(self):
        self._seed()
        with brain._chain_followup_lock:
            brain._last_chain["at"] = time.time() - 9999
        self.assertIsNone(
            brain._continue_chain("then click the search bar",
                                  "click the search bar"))

    def test_tool_followup_is_not_a_continuation(self):
        self._seed()
        self.assertIsNone(
            brain._continue_chain("then open youtube", "open youtube"))


if __name__ == "__main__":
    unittest.main()
