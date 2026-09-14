"""F20 — cancel jobs, not global flags.

Acceptance (audit report): "Stop A without affecting B; idle stop leaves new
work untouched; cancellation before click prevents it; paused expiry cannot
resume effects; interruption never later becomes completion/speech."
"""

import threading
import time
import unittest
import uuid
from unittest.mock import patch

from backend.services import jobs as job_registry


class StopOneJobTests(unittest.TestCase):
    def setUp(self):
        self.jobs = []
        self._saved_current = job_registry._current_id

    def tearDown(self):
        for job in self.jobs:
            job.finish()
        job_registry._current_id = self._saved_current

    def _new(self, kind="browser", label=""):
        job = job_registry.new_job(kind=kind, label=label)
        self.jobs.append(job)
        return job

    def test_stop_a_does_not_affect_b(self):
        job_a = self._new(label="A")
        job_b = self._new(label="B")
        cancelled = job_registry.request_stop(job_a.job_id)
        self.assertEqual(cancelled, [job_a.job_id])
        self.assertTrue(job_a.cancelled)
        self.assertFalse(job_b.cancelled)
        self.assertTrue(job_a.should_stop())
        self.assertFalse(job_b.should_stop())

    def test_unaddressed_stop_hits_exactly_one_job(self):
        self._new(label="older")
        newer = self._new(label="newer")
        cancelled = job_registry.request_stop()
        self.assertEqual(cancelled, [newer.job_id])
        live_names = [j.label for j in job_registry.live_jobs() if j.cancelled]
        self.assertEqual(live_names, ["newer"])

    def test_unaddressed_stop_is_kind_scoped(self):
        browser = self._new(kind="browser", label="browser")
        research = self._new(kind="research", label="research")
        cancelled = job_registry.request_stop(kinds=("research",))
        self.assertEqual(cancelled, [research.job_id])
        self.assertFalse(browser.cancelled)

    def test_idle_stop_leaves_new_work_untouched(self):
        self.assertEqual(job_registry.request_stop(), [])
        fresh = self._new(label="fresh")
        self.assertFalse(fresh.cancelled)
        self.assertFalse(fresh.should_stop())
        # The new job is not poisoned by the earlier idle stop.
        self.assertTrue(fresh.checkpoint())

    def test_stopping_an_unknown_id_is_a_noop(self):
        live = self._new(label="live")
        self.assertEqual(job_registry.request_stop("job-does-not-exist"), [])
        self.assertFalse(live.cancelled)


class PausedDeadlineTests(unittest.TestCase):
    def test_paused_expiry_cannot_resume_effects(self):
        job = job_registry.new_job(kind="test", timeout=0.05)
        try:
            job.pause()
            time.sleep(0.12)
            # Waking up must NOT clear the job for more work.
            with self.assertRaises(job_registry.Cancelled):
                job.checkpoint()
            self.assertTrue(job.expired())
        finally:
            job.finish()

    def test_pause_within_the_deadline_still_resumes(self):
        job = job_registry.new_job(kind="test", timeout=30.0)
        try:
            job.pause()

            def _resume():
                time.sleep(0.05)
                job.resume()

            threading.Thread(target=_resume, daemon=True).start()
            self.assertTrue(job.checkpoint())
        finally:
            job.finish()


class TurnCheckpointTests(unittest.TestCase):
    def tearDown(self):
        job_registry.unbind_turn_job(None)

    def test_jobless_turn_is_never_cancelled(self):
        job_registry.bind_turn_job(None)
        self.assertTrue(job_registry.checkpoint_turn())
        self.assertFalse(job_registry.turn_cancelled())

    def test_a_cancelled_turn_stops_the_next_effect(self):
        job = job_registry.new_job(kind="request")
        try:
            job_registry.bind_turn_job(job)
            self.assertTrue(job_registry.checkpoint_turn())
            job.cancel("stopped by user")
            with self.assertRaises(job_registry.TurnCancelled):
                job_registry.checkpoint_turn("synthesis")
            self.assertTrue(job_registry.turn_cancelled())
        finally:
            job.finish()
            job_registry.unbind_turn_job(None)

    def test_two_threads_own_different_turns(self):
        job_a = job_registry.new_job(kind="request", label="A")
        job_b = job_registry.new_job(kind="request", label="B")
        seen = {}

        def _worker(name, job):
            job_registry.bind_turn_job(job)
            seen[name] = job_registry.current_turn_job().job_id
            time.sleep(0.05)
            seen[name + "_after"] = job_registry.current_turn_job().job_id
            job_registry.unbind_turn_job(None)

        try:
            threads = [
                threading.Thread(target=_worker, args=("a", job_a)),
                threading.Thread(target=_worker, args=("b", job_b)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(seen["a"], job_a.job_id)
            self.assertEqual(seen["a_after"], job_a.job_id)
            self.assertEqual(seen["b"], job_b.job_id)
            self.assertEqual(seen["b_after"], job_b.job_id)
            self.assertNotEqual(seen["a"], seen["b"])
        finally:
            job_a.finish()
            job_b.finish()
            job_registry.unbind_turn_job(None)


class InterruptionIsFinalTests(unittest.TestCase):
    def _state(self):
        """A fresh, isolated request state per test call."""
        from backend.services import request_registry as req_registry
        state = req_registry.RequestState(
            "req-f20-" + uuid.uuid4().hex[:8], "hi")
        return state

    def test_late_completion_cannot_overwrite_an_interruption(self):
        state = self._state()
        state.interrupt("stopped by user")
        state.complete("late answer from the abandoned worker")
        types = [frame["type"] for _seq, frame in state.events]
        self.assertEqual(types[-1], "interrupted")
        self.assertNotIn("completed", types)
        self.assertIsNone(state.reply)

    def test_late_delta_cannot_speak_after_an_interruption(self):
        state = self._state()
        state.interrupt("stopped by user")
        state.delta("this must never be spoken")
        state.replace("nor this")
        state.progress("still working")
        types = [frame["type"] for _seq, frame in state.events]
        self.assertEqual(types, ["interrupted"])

    def test_interrupting_a_request_cancels_its_worker_job(self):
        state = self._state()
        job = job_registry.new_job(kind="request")
        try:
            state.attach_job(job)
            state.interrupt("stopped by user")
            self.assertTrue(job.cancelled)
        finally:
            job.finish()

    def test_a_job_attached_after_completion_is_still_stopped(self):
        state = self._state()
        state.complete("done")
        job = job_registry.new_job(kind="request")
        try:
            state.attach_job(job)
            self.assertTrue(job.cancelled)
        finally:
            job.finish()

    def test_completion_still_works_without_an_interruption(self):
        state = self._state()
        state.delta("hello ")
        state.complete("hello sir")
        types = [frame["type"] for _seq, frame in state.events]
        self.assertEqual(types, ["delta", "completed"])
        self.assertEqual(state.reply, "hello sir")


class TaskStepCheckpointTests(unittest.TestCase):
    """A cancellation lands BEFORE the next effect, not after it."""

    def tearDown(self):
        job_registry.unbind_turn_job(None)

    def test_cancelled_turn_runs_no_later_step(self):
        from backend.services.task_agent import agent

        job = job_registry.new_job(kind="request")
        plan = {"ok": True, "steps": [
            {"tool": "code.run_command", "args": {"command": "echo one"}},
            {"tool": "code.run_command", "args": {"command": "echo two"}},
        ]}
        calls = []

        def _fake_execute(step, context):
            calls.append(step.get("args", {}).get("command"))
            job.cancel("stopped by user")
            return "one", {"ok": True}

        try:
            job_registry.bind_turn_job(job)
            with patch.object(agent, "_execute_step_structured",
                              side_effect=_fake_execute):
                result = agent.execute_plan(plan, {}, task_text="run two things")
            self.assertEqual(calls, ["echo one"])
            # The stopped turn reports itself as stopped, not merely failed,
            # and the evidence names the cancellation.
            self.assertIn("Stopped per your request.", str(result))
            evidence = " ".join(getattr(result, "evidence", None) or [])
            self.assertIn("cancelled", evidence)
        finally:
            job.finish()
            job_registry.unbind_turn_job(None)


if __name__ == "__main__":
    unittest.main()
