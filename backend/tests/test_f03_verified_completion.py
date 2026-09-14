"""F03 — completion is an evidence-backed status.

Acceptance (audit report): "Refusals, no-action Done, failed tools, zero-exit
no-ops, unknown exit status, and failed output containing created cannot
become verified completion."

Baseline defects pinned here:
  * ``brain._coerce_task_result`` promoted ANY legacy string to 'completed',
    including refusals and bare "Done.";
  * a report whose own words carried a failure ("failed: nothing was created")
    was still read as success because the word 'created' appeared;
  * ``opencode_client`` accepted ``returncode is None`` (the wait failed, so
    the status was never observed) as a verified completion;
  * a run made only of zero-exit no-ops reported "Done, sir.".
"""

import unittest
from unittest.mock import MagicMock, patch

from backend.core import brain
from backend.services import opencode_client
from backend.services.task_agent import agent
from backend.services.task_result import (
    COMPLETED,
    KNOWN_FAILURE,
    NO_ACTION,
    PARTIAL,
    REFUSED,
    UNVERIFIED,
    VERIFIED,
    TaskResult,
    classify_reported_text,
    result_from_reported_text,
)


class ClassifierTests(unittest.TestCase):
    def test_refusals_are_not_verified(self):
        for text in (
            "I can't do that.",
            "I cannot access that file.",
            "I'm unable to comply.",
            "I won't delete that.",
            "Unable to complete the request.",
            "I did not change anything.",
            "No action was taken.",
        ):
            verdict = classify_reported_text(text)
            self.assertIn(verdict, (REFUSED, NO_ACTION), text)
            self.assertNotEqual(verdict, VERIFIED, text)

    def test_bare_done_is_not_verified(self):
        for text in ("Done.", "done", "Completed.", "OK", "Finished."):
            self.assertIn(classify_reported_text(text), (NO_ACTION, UNVERIFIED),
                          text)

    def test_failed_wording_beats_success_wording(self):
        """A failure report that mentions 'created' is still a failure."""
        for text in (
            "code.write_file failed: the file was not created",
            "Error: could not create the folder",
            "the request failed after the file was written",
            "Permission denied while the file was being created",
            "opencode timed out before anything was created",
        ):
            self.assertEqual(classify_reported_text(text), KNOWN_FAILURE, text)

    def test_real_effects_are_verified(self):
        for text in (
            "Created the folder and wrote hello.txt",
            "Saved the report to report.md",
            "Opened https://example.com and clicked the login button",
        ):
            self.assertEqual(classify_reported_text(text), VERIFIED, text)

    def test_plain_claims_are_unverified(self):
        for text in (
            "I think that should be fine.",
            "The task is probably complete now.",
            "Let me know if you need more.",
        ):
            self.assertEqual(classify_reported_text(text), UNVERIFIED, text)


class ResultBuilderTests(unittest.TestCase):
    def test_failure_report_becomes_failed(self):
        result = result_from_reported_text("failed: could not create the file")
        self.assertEqual(result.status, "failed")

    def test_refusal_becomes_partial_with_evidence(self):
        result = result_from_reported_text("I can't do that.")
        self.assertEqual(result.status, PARTIAL)
        self.assertTrue(result.evidence)

    def test_no_action_done_is_partial(self):
        result = result_from_reported_text("Done.")
        self.assertEqual(result.status, PARTIAL)
        self.assertNotEqual(result.status, COMPLETED)

    def test_verified_effect_can_complete(self):
        result = result_from_reported_text("Created hello.txt")
        self.assertEqual(result.status, COMPLETED)

    def test_no_evidence_is_added_for_unverified_claims(self):
        result = result_from_reported_text("Everything should be fine.",
                                           trust_answer=False)
        self.assertEqual(result.status, PARTIAL)
        self.assertIn("completion not evidenced", result.evidence)

    def test_a_question_is_a_suspension_not_a_completion(self):
        result = result_from_reported_text(
            "Which site should I open for the price check?")
        self.assertEqual(result.status, "needs_input")
        self.assertEqual(classify_reported_text(
            "Which site should I open for the price check?"), "question")

    def test_a_substantive_answer_completes_by_default(self):
        answer = ("The Pixel 9 costs about 799 dollars on the official store, "
                  "and the Pro model starts at 999.")
        result = result_from_reported_text(answer)
        self.assertEqual(result.status, COMPLETED)

    def test_verified_builder_requires_evidence(self):
        with self.assertRaises(ValueError):
            TaskResult.verified("done", [])
        result = TaskResult.verified("Created x", ["x exists on disk"])
        self.assertEqual(result.status, COMPLETED)
        self.assertEqual(result.evidence, ["x exists on disk"])

    def test_unmet_goals_force_partial(self):
        result = TaskResult.completed("Done, sir.", unmet_goals=["step 9"])
        self.assertEqual(result.status, PARTIAL)

    def test_artifacts_survive_the_builders(self):
        result = TaskResult.partial(
            "partly", artifacts=[{"step": 0, "path": "C:/tmp/a.txt"}])
        self.assertEqual(result.artifacts, [{"step": 0, "path": "C:/tmp/a.txt"}])


class BrainCoercionTests(unittest.TestCase):
    def test_legacy_string_refusal_is_not_completed(self):
        result = brain._coerce_task_result("I'm unable to do that.")
        self.assertNotEqual(result.status, COMPLETED)

    def test_legacy_string_bare_done_is_not_completed(self):
        result = brain._coerce_task_result("Done.")
        self.assertNotEqual(result.status, COMPLETED)

    def test_legacy_string_failure_still_failed(self):
        result = brain._coerce_task_result(
            "TASK NOT COMPLETED. Error: no such file")
        self.assertEqual(result.status, "failed")

    def test_legacy_string_with_real_effect_completes(self):
        result = brain._coerce_task_result("Created the folder C:/tmp/x")
        self.assertEqual(result.status, COMPLETED)

    def test_task_result_passes_through(self):
        original = TaskResult.partial("partly")
        self.assertIs(brain._coerce_task_result(original), original)


class OpencodeStatusTests(unittest.TestCase):
    def _run_with_returncode(self, returncode, output="Created the file."):
        requested = {"code": returncode}

        class FakeStdout:
            def __init__(self, lines):
                self._lines = list(lines)

            def readline(self):
                return self._lines.pop(0) if self._lines else ""

        class FakeProc:
            def __init__(self):
                self.returncode = None
                self.stdout = FakeStdout([output] if output else [])

            def wait(self, timeout=None):
                self.returncode = requested["code"]
                return self.returncode

        fake_job = MagicMock()
        fake_job.should_stop.return_value = False
        fake_job.finish.return_value = None
        fake_job.terminate_processes.return_value = None

        with patch.object(opencode_client, "_opencode_engine_enabled",
                          return_value=True), \
             patch.object(opencode_client, "_resolve_opencode",
                          return_value="opencode"), \
             patch.object(opencode_client, "_reset_narration_state"), \
             patch.object(opencode_client, "_truncate_activity_log"), \
             patch.object(opencode_client, "_append_activity"), \
             patch.object(opencode_client, "_narrate_line"), \
             patch.object(opencode_client, "_build_command",
                          return_value=["opencode", "run"]), \
             patch.object(opencode_client.subprocess, "Popen",
                          return_value=FakeProc()), \
             patch("backend.services.jobs.new_job", return_value=fake_job):
            return opencode_client.run_opencode_task("do it", timeout=5)

    def test_unknown_exit_status_is_not_completion(self):
        result = self._run_with_returncode(None)
        self.assertNotEqual(result.status, COMPLETED)
        self.assertEqual(result.status, PARTIAL)
        self.assertTrue(result.evidence)

    def test_nonzero_exit_is_failed(self):
        result = self._run_with_returncode(3)
        self.assertEqual(result.status, "failed")

    def test_zero_exit_with_effect_completes(self):
        result = self._run_with_returncode(0)
        self.assertEqual(result.status, COMPLETED)


class ZeroEffectRunTests(unittest.TestCase):
    def test_a_run_of_no_ops_is_not_completed(self):
        def fake_execute(step, context):
            return ("(no output)", {"ok": True, "content": "(no output)",
                                    "exit_code": 0})

        plan = agent._normalize_plan({
            "ok": True,
            "steps": [{"tool": "code.run_command", "args": {"command": "rem"}}],
        }, "run it")
        with patch.object(agent, "_execute_step_structured",
                          side_effect=fake_execute):
            result = agent.execute_plan(plan, {}, confirmed=True)
        self.assertEqual(result.status, PARTIAL)
        self.assertIn("no step produced an observable effect", result.evidence)

    def test_a_real_observation_still_completes(self):
        def fake_execute(step, context):
            return ("1 entry: report.md", {"ok": True, "entries": ["report.md"]})

        plan = agent._normalize_plan({
            "ok": True,
            "steps": [{"tool": "code.list_directory", "args": {"path": "C:/tmp"}}],
        }, "list")
        with patch.object(agent, "_execute_step_structured",
                          side_effect=fake_execute):
            result = agent.execute_plan(plan, {}, confirmed=True)
        self.assertEqual(result.status, COMPLETED)


class BrowserRunOwnershipTests(unittest.TestCase):
    """F03/F26: a superseded browser run publishes nothing.

    A deferred run executes in its own thread; before this, ANY finished run
    could notify, arm a clarification and speak, so an older run's result — or
    its question — could land in a later request.
    """

    def tearDown(self):
        brain.set_opencode_task_running(False)
        brain._pending_browser_clarification = None

    def test_a_new_run_supersedes_the_previous_one(self):
        first = brain._new_browser_run()
        self.assertTrue(brain._browser_run_is_current(first))
        second = brain._new_browser_run()
        self.assertFalse(brain._browser_run_is_current(first))
        self.assertTrue(brain._browser_run_is_current(second))
        brain.invalidate_browser_runs("test")
        self.assertFalse(brain._browser_run_is_current(second))

    def test_superseded_run_never_notifies(self):
        import threading
        import time as _time

        from backend import config

        release = threading.Event()
        calls = {"n": 0}

        def fake_run(task):
            calls["n"] += 1
            if task == "first":
                release.wait(5)
                return "Created the first file."
            return "Which site should I open for the price check?"

        with patch.object(config, "TASK_ENGINE", "browser_agent"), \
             patch.object(brain, "run_browser_task", side_effect=fake_run), \
             patch.object(brain, "_notify_async_reply") as notify, \
             patch.object(brain, "set_narration_enabled"):
            brain._execute_deferred_opencode("first", "first")
            _time.sleep(0.05)
            brain._execute_deferred_opencode("second", "second")
            deadline = _time.time() + 3
            while _time.time() < deadline and notify.call_count < 1:
                _time.sleep(0.02)
            self.assertEqual(notify.call_count, 1,
                             "only the newest run may notify")
            release.set()
            _time.sleep(0.4)
        self.assertEqual(notify.call_count, 1,
                         "the superseded run must stay silent")
        self.assertIn("price check", notify.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
