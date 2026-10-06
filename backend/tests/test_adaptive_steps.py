"""Rank 3 — the adaptive step loop: diagnose, retry DIFFERENTLY, report honestly.

Covers the pure decision layer (adaptive_steps) and the loop wired into the
task agent (do -> check -> diagnose -> different retry, with hard caps).
"""

import os
import tempfile
import unittest

from unittest.mock import patch

from backend.services import adaptive_steps
from backend.services import approvals
from backend.services.task_agent import agent


class DiagnosisTests(unittest.TestCase):
    def test_timeout_is_transient_and_retryable_for_a_search(self):
        verdict = adaptive_steps.diagnose(
            "browser.search_web",
            "browser.search_web failed: timed out")
        self.assertEqual(verdict["class"], adaptive_steps.TRANSIENT)
        self.assertTrue(verdict["retryable"])
        tactics = adaptive_steps.tactics_for("browser.search_web",
                                             verdict["class"])
        self.assertIn("resettle_wait", tactics)
        self.assertIn("refresh_evidence", tactics)

    def test_popup_evidence_is_an_overlay(self):
        verdict = adaptive_steps.diagnose(
            "windows.screen_action",
            "windows.screen_action failed: the button was covered by a popup")
        self.assertEqual(verdict["class"], adaptive_steps.OVERLAY)
        self.assertIn("dismiss_overlay",
                      adaptive_steps.tactics_for("windows.screen_action",
                                                 verdict["class"]))

    def test_login_wall_is_never_retried(self):
        verdict = adaptive_steps.diagnose(
            "browser.open_url",
            "the page shows a sign in form and asks for a password")
        self.assertEqual(verdict["class"], adaptive_steps.LOGIN)
        self.assertFalse(verdict["retryable"])
        self.assertEqual(
            adaptive_steps.tactics_for("browser.open_url", verdict["class"]),
            ())

    def test_existing_target_is_never_retried(self):
        verdict = adaptive_steps.diagnose(
            "code.write_file", "code.write_file failed: refusal",
            {"ok": False, "error": "the file already exists"})
        self.assertEqual(verdict["class"], adaptive_steps.EXISTS)
        self.assertFalse(verdict["retryable"])

    def test_permission_denial_is_never_retried(self):
        verdict = adaptive_steps.diagnose(
            "windows.screen_action",
            "windows.screen_action failed: access denied by the app")
        self.assertEqual(verdict["class"], adaptive_steps.PERMISSION)
        self.assertFalse(verdict["retryable"])

    def test_missing_parent_is_a_replan_not_a_retry(self):
        verdict = adaptive_steps.diagnose(
            "code.write_file",
            "code.write_file failed: [WinError 3] The system cannot find "
            "the path specified")
        self.assertEqual(verdict["class"], adaptive_steps.PREREQ)
        self.assertFalse(verdict["retryable"])

    def test_read_file_not_found_is_not_retried(self):
        verdict = adaptive_steps.diagnose(
            "code.read_file", "code.read_file failed: File not found: a.txt")
        self.assertFalse(verdict["retryable"])

    def test_external_effects_are_never_retried(self):
        verdict = adaptive_steps.diagnose(
            "mail.send", "mail.send failed: timed out", external_effect=True)
        self.assertFalse(verdict["retryable"])

    def test_tactic_order_prefers_the_smarter_route_for_searches(self):
        tactics = adaptive_steps.tactics_for("browser.search_web",
                                             adaptive_steps.MISSING)
        self.assertEqual(tactics[0], "alternate_route")

    def test_simplify_query_changes_a_complex_query(self):
        original = "please could you look up the best coffee machine reviews"
        simplified = adaptive_steps.simplify_query(original)
        self.assertTrue(simplified)
        self.assertNotEqual(simplified.lower(), original.lower())
        self.assertNotIn("please", simplified.lower().split())

    def test_simplify_query_refuses_a_no_change_retry(self):
        self.assertEqual(adaptive_steps.simplify_query("weather"), "")

    def test_attempt_signature_distinguishes_tactics(self):
        first = adaptive_steps.attempt_signature(
            "browser.search_web", {"query": "x"}, "initial")
        second = adaptive_steps.attempt_signature(
            "browser.search_web", {"query": "x"}, "resettle_wait")
        self.assertNotEqual(first, second)

    def test_replan_budget_is_bounded(self):
        self.assertTrue(adaptive_steps.can_replan(0))
        self.assertTrue(adaptive_steps.can_replan(1))
        self.assertFalse(adaptive_steps.can_replan(2))


class AdaptiveLoopTests(unittest.TestCase):
    def test_a_transient_failure_is_retried_differently_and_can_win(self):
        step = {"tool": "browser.search_web",
                "args": {"query": "best coffee machine reviews"}}
        responses = [
            ("browser.search_web failed: timed out", None),
            ("Opened a web search.", {"ok": True}),
        ]
        with patch.object(agent, "_execute_step_structured",
                          side_effect=responses), \
             patch.object(agent, "_run_recovery_tactic",
                          return_value=True) as tactic:
            text, structured, attempts, failed, reason, diagnosis = \
                agent._execute_step_adaptive(step, {})
        self.assertFalse(failed)
        self.assertEqual([item["tactic"] for item in attempts],
                         ["initial", "resettle_wait"])
        tactic.assert_called_once()

    def test_identical_retries_are_banned_and_the_cap_is_hard(self):
        step = {"tool": "windows.screen_action",
                "args": {"command": "click play"}}
        failure = ("windows.screen_action failed: the target was covered "
                   "by a popup", None)
        with patch.object(agent, "_execute_step_structured",
                          side_effect=[failure] * 5), \
             patch.object(agent, "_run_recovery_tactic",
                          return_value=True) as tactic:
            text, structured, attempts, failed, reason, diagnosis = \
                agent._execute_step_adaptive(step, {})
        self.assertTrue(failed)
        self.assertEqual(len(attempts), adaptive_steps.MAX_TRIES_PER_STEP)
        tactics_used = [item["tactic"] for item in attempts]
        self.assertEqual(len(set(tactics_used)), len(tactics_used),
                         "the same tactic must never run twice for one step")
        self.assertEqual(tactic.call_count, adaptive_steps.MAX_TRIES_PER_STEP - 1)

    def test_a_login_wall_stops_immediately(self):
        step = {"tool": "browser.open_url",
                "args": {"url": "https://mail.example.com"}}
        with patch.object(
                agent, "_execute_step_structured",
                return_value=("browser.open_url failed: please sign in to "
                              "continue", None)), \
             patch.object(agent, "_run_recovery_tactic") as tactic:
            text, structured, attempts, failed, reason, diagnosis = \
                agent._execute_step_adaptive(step, {})
        self.assertTrue(failed)
        self.assertEqual(len(attempts), 1)
        tactic.assert_not_called()

    def test_a_recovery_that_cannot_run_stops_the_loop(self):
        step = {"tool": "browser.search_web", "args": {"query": "x"}}
        with patch.object(agent, "_execute_step_structured",
                          return_value=("browser.search_web failed: timed "
                                        "out", None)), \
             patch.object(agent, "_run_recovery_tactic",
                          return_value=False):
            text, structured, attempts, failed, reason, diagnosis = \
                agent._execute_step_adaptive(step, {})
        self.assertTrue(failed)
        self.assertEqual(len(attempts), 1)

    def test_alternate_route_simplifies_a_failed_search(self):
        original = "please look up the best coffee machine for me"
        step = {"tool": "browser.search_web", "args": {"query": original}}
        with patch.object(agent, "_refresh_context_evidence"):
            self.assertTrue(
                agent._recovery_alternate_route(step, {}, 1))
        self.assertNotEqual(step["args"]["query"], original)

    def test_alternate_route_refuses_when_nothing_can_change(self):
        step = {"tool": "browser.search_web", "args": {"query": "weather"}}
        self.assertFalse(agent._recovery_alternate_route(step, {}, 1))

    def test_alternate_route_activates_an_existing_tab(self):
        step = {"tool": "browser.open_url",
                "args": {"url": "https://example.com/page"}}
        context = {"browser": {"tabs": [
            {"id": "42", "url": "https://example.com/other"}]}}
        with patch.object(agent.browser_cdp, "activate_tab",
                          return_value={"activated": True}) as activate:
            self.assertTrue(
                agent._recovery_alternate_route(step, context, 1))
        activate.assert_called_once_with("42")

    def test_dismiss_overlay_only_fires_while_the_window_is_unchanged(self):
        step = {"tool": "windows.screen_action",
                "args": {"command": "click play"}}
        context = {"windows": {"active_window": {"hwnd": 7}}}
        with patch.object(agent.windows_connector, "get_active_window",
                          return_value={"hwnd": 9}):
            self.assertFalse(
                agent._recovery_dismiss_overlay(step, context, 1))
        with patch.object(agent.windows_connector, "get_active_window",
                          return_value={"hwnd": 7}), \
             patch("backend.services.screen_executor.press_keys") as press, \
             patch.object(agent, "_refresh_context_evidence"):
            self.assertTrue(
                agent._recovery_dismiss_overlay(step, context, 1))
        press.assert_called_once_with(["esc"])


class ReplanTests(unittest.TestCase):
    def setUp(self):
        approvals.cancel("rank3 test setup")

    def tearDown(self):
        approvals.cancel("rank3 test teardown")

    def _plan(self, target):
        plan = agent._normalize_plan({
            "ok": True,
            "command_text": "write the report",
            "steps": [{"tool": "code.write_file",
                       "args": {"path": target, "content": "x"}}],
        }, "write the report")
        plan["requires_confirmation"] = True
        return plan

    def test_missing_parent_write_rearms_with_a_folder_step(self):
        directory = tempfile.mkdtemp(prefix="rank3_")
        missing_parent = os.path.join(directory, "not-there")
        target = os.path.join(missing_parent, "report.txt")
        failure = (
            "code.write_file failed: [WinError 3] The system cannot find "
            "the path specified",
            {"ok": False,
             "error": "[WinError 3] The system cannot find the path "
                      "specified"})
        armed = {}

        def fake_arm(updated, context, task_text=""):
            armed["plan"] = updated
            return None

        with patch.object(agent, "_execute_step_structured",
                          return_value=failure), \
             patch.object(agent, "_arm_plan_confirmation",
                          side_effect=fake_arm):
            result = agent.execute_plan(self._plan(target), {},
                                        confirmed=True)
        self.assertEqual(result.status, "needs_input")
        self.assertIn("create it first", str(result))
        tools = [step["tool"] for step in armed["plan"]["steps"]]
        self.assertEqual(tools, ["code.create_folder", "code.write_file"])
        self.assertEqual(armed["plan"]["steps"][0]["args"]["path"],
                         missing_parent)
        self.assertEqual(armed["plan"]["_adaptive_replans"], 1)

    def test_replan_cap_is_hard(self):
        directory = tempfile.mkdtemp(prefix="rank3_")
        target = os.path.join(directory, "gone", "report.txt")
        plan = self._plan(target)
        plan["_adaptive_replans"] = adaptive_steps.MAX_REPLANS_PER_JOB
        failure = (
            "code.write_file failed: [WinError 3] The system cannot find "
            "the path specified",
            {"ok": False, "error": "The system cannot find the path "
                                   "specified"})
        with patch.object(agent, "_execute_step_structured",
                          return_value=failure), \
             patch.object(agent, "_arm_plan_confirmation") as armed:
            result = agent.execute_plan(plan, {}, confirmed=True)
        armed.assert_not_called()
        self.assertNotEqual(result.status, "needs_input")
        self.assertEqual(result.status, "partial")

    def test_a_failed_step_reports_how_many_tries(self):
        plan = agent._normalize_plan({
            "ok": True,
            "steps": [{"tool": "browser.search_web",
                       "args": {"query": "best coffee machine"}}],
        }, "search it")
        failure = ("browser.search_web failed: timed out", None)
        with patch.object(agent, "_execute_step_structured",
                          return_value=failure), \
             patch.object(agent, "_run_recovery_tactic",
                          return_value=True):
            result = agent.execute_plan(plan, {}, confirmed=True)
        self.assertEqual(result.status, "partial")
        evidence = " ".join(result.evidence or [])
        self.assertIn("after 3 tries", evidence)


if __name__ == "__main__":
    unittest.main()
