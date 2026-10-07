"""Rank 5 — STOP MEANS STOP.

A bare "stop" cuts speech and cancels exactly the watched work job;
"stop everything" cancels every work job; "Stopped" is spoken only when the
cancelled work is observably still; and a stopped browser run reports what
it had already done.
"""

import unittest
from unittest.mock import patch

from backend.core import brain
from backend.services import browser_agent
from backend.services import jobs


class StopPhraseTests(unittest.TestCase):
    def test_bare_stop_forms(self):
        for text in ("stop", "stop it", "stop now", "please stop",
                     "jarvis, stop!", "ruko", "stop it please"):
            self.assertTrue(brain.is_bare_stop(text), text)

    def test_negated_stop_is_not_a_stop(self):
        for text in ("don't stop", "do not stop", "never stop", "mat roko"):
            self.assertFalse(brain.is_bare_stop(text), text)
            self.assertFalse(brain.is_stop_everything(text), text)

    def test_dedicated_phrases_are_not_bare_stop(self):
        # "stop the research" / "stop everything" keep their own paths.
        self.assertFalse(brain.is_bare_stop("stop the research"))
        self.assertFalse(brain.is_bare_stop("stop everything"))
        self.assertFalse(brain.is_bare_stop("stop searching for that"))

    def test_stop_everything_forms(self):
        for text in ("stop everything", "stop everything now",
                     "stop all tasks", "stop it all", "stop all work"):
            self.assertTrue(brain.is_stop_everything(text), text)


class StopHandlerTests(unittest.TestCase):
    def _quiet(self):
        return [
            patch.object(brain, "opencode_task_in_progress",
                         return_value=False),
            patch.object(brain, "_browser_quiescent", return_value=True),
            patch.object(brain, "request_browser_task_stop"),
            patch.object(brain, "invalidate_browser_runs"),
            patch.object(brain, "set_narration_enabled"),
            patch.object(jobs, "live_jobs", return_value=[]),
            patch.object(jobs, "request_stop", return_value=[]),
            patch("backend.services.voice.stop_speaking"),
        ]

    def test_stop_with_nothing_running_is_stopped_now(self):
        browser_agent.consume_stop_report()
        patches = self._quiet()
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.assertEqual(brain.handle_stop_message("stop"), "Stopped, sir.")

    def test_stop_everything_cancels_every_work_job(self):
        job_a = jobs.new_job(kind="browser", label="a")
        job_b = jobs.new_job(kind="background", label="b")
        request_job = jobs.new_job(kind="request", label="chat")
        self.addCleanup(job_a.finish)
        self.addCleanup(job_b.finish)
        self.addCleanup(request_job.finish)
        patches = self._quiet()
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        # live_jobs/request_stop are not mocked here: the real registry runs.
        with patch.object(jobs, "live_jobs",
                          side_effect=lambda kind=None, exclude_kinds=None: [
                              j for j in (job_a, job_b, request_job)
                              if not exclude_kinds
                              or j.kind not in exclude_kinds]):
            brain.handle_stop_message("stop everything")
        self.assertTrue(job_a.cancelled)
        self.assertTrue(job_b.cancelled)
        self.assertFalse(request_job.cancelled,
                         "the chat request is never stopped")

    def test_stop_while_watched_work_runs_says_stopping(self):
        patches = self._quiet()
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        with patch.object(brain, "opencode_task_in_progress",
                          return_value=True), \
                patch.object(brain, "_finish_stop_reply") as finisher, \
                patch.object(brain, "threading") as thread_mod:
            reply = brain.handle_stop_message("stop")
        self.assertEqual(reply, "Stopping, sir.")
        self.assertEqual(thread_mod.Thread.call_count, 1)
        self.assertEqual(finisher.call_count, 0)

    def test_finisher_reports_stopped_when_still(self):
        messages = []
        with patch.object(brain, "_stop_targets_still", return_value=True), \
                patch.object(brain, "_notify_async_reply",
                             side_effect=messages.append), \
                patch.object(brain, "consume_browser_stop_report",
                             return_value="clicked Play; opened youtube.com"):
            brain._finish_stop_reply([], False, False)
        self.assertEqual(len(messages), 1)
        self.assertIn("Stopped, sir.", messages[0])
        self.assertIn("clicked Play", messages[0])

    def test_finisher_is_honest_when_still_moving(self):
        messages = []
        with patch.object(brain, "_stop_targets_still", return_value=False), \
                patch.object(brain, "_notify_async_reply",
                             side_effect=messages.append), \
                patch.object(brain, "_STOP_QUIESCE_TIMEOUT_S", 0.01):
            brain._finish_stop_reply([], False, False)
        self.assertEqual(len(messages), 1)
        self.assertNotIn("Stopped, sir.", messages[0])
        self.assertIn("still moving", messages[0])


class StopGateRoutingTests(unittest.TestCase):
    def setUp(self):
        brain._held_redirect = None
        browser_agent.consume_stop_report()

    def test_typed_stop_reaches_the_general_stop(self):
        with patch.object(brain, "opencode_task_in_progress",
                          return_value=False), \
                patch.object(brain, "_browser_quiescent", return_value=True), \
                patch.object(brain, "request_browser_task_stop"), \
                patch.object(brain, "invalidate_browser_runs"), \
                patch.object(brain, "set_narration_enabled"), \
                patch.object(jobs, "live_jobs", return_value=[]), \
                patch.object(jobs, "request_stop", return_value=[]), \
                patch("backend.services.voice.stop_speaking"), \
                patch.object(brain, "classify_intent") as clf:
            reply = brain.process_message("stop", sync_voice=False)
        self.assertEqual(reply, "Stopped, sir.")
        clf.assert_not_called()

    def test_typed_stop_everything_reaches_the_general_stop(self):
        with patch.object(brain, "opencode_task_in_progress",
                          return_value=False), \
                patch.object(brain, "_browser_quiescent", return_value=True), \
                patch.object(brain, "request_browser_task_stop"), \
                patch.object(brain, "invalidate_browser_runs"), \
                patch.object(brain, "set_narration_enabled"), \
                patch.object(jobs, "live_jobs", return_value=[]), \
                patch.object(jobs, "cancel_job", return_value=[]) as cancel, \
                patch("backend.services.voice.stop_speaking"), \
                patch.object(brain, "classify_intent") as clf:
            reply = brain.process_message("stop everything", sync_voice=False)
        self.assertEqual(reply, "Stopped, sir.")
        clf.assert_not_called()


class BrowserStopReportTests(unittest.TestCase):
    def setUp(self):
        browser_agent.consume_stop_report()

    def test_publish_and_consume_roundtrip(self):
        browser_agent._publish_stop_report(
            {"completed_actions": ["open_url youtube.com", "click_mark Play"]})
        report = browser_agent.consume_stop_report()
        self.assertIn("opened youtube.com", report)
        self.assertIn("clicked Play", report)
        self.assertEqual(browser_agent.consume_stop_report(), "")

    def test_empty_session_publishes_nothing(self):
        browser_agent._publish_stop_report({})
        self.assertEqual(browser_agent.consume_stop_report(), "")

    def test_report_is_bounded(self):
        actions = ["click_mark button-%d" % i for i in range(30)]
        browser_agent._publish_stop_report({"completed_actions": actions})
        report = browser_agent.consume_stop_report()
        self.assertLessEqual(len(report), 240)


class VoiceStopAllTests(unittest.TestCase):
    def test_stop_everything_is_its_own_voice_control(self):
        from backend import voice_mode as vm
        self.assertEqual(vm.classify_control("stop everything"),
                         "task_stop_all")
        self.assertEqual(vm.classify_control("stop all tasks"),
                         "task_stop_all")
        self.assertEqual(vm.classify_control("stop it"), "task_stop")
        self.assertEqual(vm.control_owner("task_stop_all"), "task")

    def test_dispatch_stop_all_posts_scope_all(self):
        from backend import voice_mode as vm
        posted = []

        def fake_post(path, payload, timeout=2.5):
            posted.append(path)
            return True, {}

        with patch.object(vm, "stop_speaking"), \
                patch.object(vm, "_post_backend", side_effect=fake_post):
            self.assertTrue(vm.dispatch_control("task_stop_all"))
        self.assertEqual(posted, ["/task/stop?scope=all"])


if __name__ == "__main__":
    unittest.main()
