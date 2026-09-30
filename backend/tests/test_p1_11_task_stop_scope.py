"""P1-11 â€” /task/stop cancels precisely what was asked for, and nothing else.

The audit: with no job specified, the endpoint took the newest job of ANY kind
(which may be a chat request rather than the task the user meant to stop) and
then interrupted EVERY live request, so stopping a browser job silently killed
an unrelated chat reply. P0-08 made that materially worse by making concurrent
work the normal case.

These tests pin the three halves of the fix: an unaddressed stop excludes
``request`` jobs, only the cancelled job's own streams are interrupted, and a
stop with no target is a safe, informative no-op.
"""

import threading
import uuid
import unittest

from backend.api import routes
from backend.services import browser_agent, jobs as job_registry
from backend.services import request_registry
from backend.services import research_service


class _StopTestCase(unittest.TestCase):
    """Clean job/request registries and engine flags around every test."""

    def setUp(self):
        self._jobs = list(job_registry.live_jobs())
        self._requests = list(getattr(request_registry.REGISTRY, "_requests", {})
                              .values())
        browser_agent._STOP_REQUESTED.clear()
        research_service._STOP_REQUESTED.clear()

    def tearDown(self):
        for job in job_registry.live_jobs():
            if job not in self._jobs:
                job.finish()
        browser_agent._STOP_REQUESTED.clear()
        research_service._STOP_REQUESTED.clear()
        for state in self._requests:
            try:
                if not state.done:
                    state.complete("")
            except Exception:
                pass

    def _request(self, name, message="hi", job=None):
        """Admit a request under a per-test-unique id.

        The registry keeps a finished request until it is reaped, so a reused
        id would hand back the PREVIOUS test's state (with its own job already
        attached) and the test would assert against a stale object.
        """
        request_id = "%s-%s" % (name, uuid.uuid4().hex[:8])
        state = request_registry.REGISTRY.admit(request_id, message)[0]
        self.assertIsNotNone(state, "the registry refused to admit a request")
        if job is not None:
            state.attach_job(job)
        return state


class UnaddressedStopTests(_StopTestCase):
    """Acceptance: a bare stop never kills an in-flight chat request."""

    def test_a_bare_stop_does_not_cancel_an_active_chat_request(self):
        chat_job = job_registry.new_job(kind="request", label="chat")
        self._request("req-chat", "chat", chat_job)

        response = routes.stop_task()

        self.assertTrue(response["ok"])
        self.assertEqual(response["cancelled"], [],
                         "the chat request's own job was cancelled")
        self.assertFalse(chat_job.cancelled)

    def test_a_bare_stop_cancels_the_newest_tool_job(self):
        older = job_registry.new_job(kind="browser", label="older")
        newer = job_registry.new_job(kind="opencode", label="newer")

        response = routes.stop_task()

        self.assertEqual(response["cancelled"], [newer.job_id])
        self.assertTrue(newer.cancelled)
        self.assertFalse(older.cancelled, "a stop cancelled more than one job")

    def test_a_newer_chat_request_does_not_shadow_the_running_task(self):
        """The reported bug: the newest job of ANY kind won, chat included."""
        task = job_registry.new_job(kind="browser", label="the task")
        chat_job = job_registry.new_job(kind="request", label="chat")
        self._request("req-chat", "chat", chat_job)

        response = routes.stop_task()

        self.assertEqual(response["cancelled"], [task.job_id])
        self.assertTrue(task.cancelled)
        self.assertFalse(chat_job.cancelled)

    def test_an_explicit_request_kind_can_still_stop_a_chat(self):
        """Excluding the kind by DEFAULT must not remove the capability."""
        chat_job = job_registry.new_job(kind="request", label="chat")

        response = routes.stop_task(kind="request")

        self.assertEqual(response["cancelled"], [chat_job.job_id])
        self.assertTrue(chat_job.cancelled)

    def test_an_explicit_job_id_can_still_stop_a_chat(self):
        chat_job = job_registry.new_job(kind="request", label="chat")

        response = routes.stop_task(job_id=chat_job.job_id)

        self.assertEqual(response["cancelled"], [chat_job.job_id])
        self.assertTrue(chat_job.cancelled)


class PreciseInterruptionTests(_StopTestCase):
    """The regression test: stopping one job leaves unrelated work running."""

    def test_stopping_one_job_leaves_a_concurrent_request_running(self):
        task = job_registry.new_job(kind="browser", label="task")
        other_job = job_registry.new_job(kind="request", label="other")
        other = self._request("req-other", "other question", other_job)

        response = routes.stop_task(job_id=task.job_id)

        self.assertEqual(response["cancelled"], [task.job_id])
        self.assertEqual(response["interrupted"], [],
                         "an unrelated request was interrupted")
        self.assertFalse(other.done, "the concurrent request was terminated")

    def test_the_cancelled_jobs_own_stream_is_interrupted(self):
        chat_job = job_registry.new_job(kind="request", label="chat")
        state = self._request("req-chat", "stop me", chat_job)

        response = routes.stop_task(job_id=chat_job.job_id)

        self.assertEqual(response["interrupted"], [state.request_id])
        self.assertTrue(state.done)
        self.assertTrue(state.interrupted)

    def test_a_tool_job_stop_leaves_an_unrelated_stream_untouched(self):
        """The audit's exact scenario: a browser stop must not kill a chat."""
        browser_job = job_registry.new_job(kind="browser", label="browser")
        chat_job = job_registry.new_job(kind="request", label="chat")
        chat = self._request("req-chat", "long answer", chat_job)
        self._request("req-unrelated", "another one")

        response = routes.stop_task(job_id=browser_job.job_id)

        self.assertEqual(response["cancelled"], [browser_job.job_id])
        self.assertEqual(response["interrupted"], [])
        self.assertFalse(chat.done)
        self.assertEqual(request_registry.REGISTRY.live_count(), 2)


class IdempotentStopTests(_StopTestCase):
    """Acceptance: a stop with no valid target reports 'stopped nothing'."""

    def test_a_stop_on_a_finished_job_is_a_safe_no_op(self):
        job = job_registry.new_job(kind="browser", label="done")
        job.finish()

        first = routes.stop_task(job_id=job.job_id)
        second = routes.stop_task(job_id=job.job_id)

        for response in (first, second):
            self.assertTrue(response["ok"])
            self.assertEqual(response["cancelled"], [])
            self.assertEqual(response["stopped"], "")
            self.assertEqual(response["interrupted"], [])

    def test_a_stop_with_no_target_reports_stopped_nothing(self):
        response = routes.stop_task()

        self.assertTrue(response["ok"])
        self.assertEqual(response["cancelled"], [])
        self.assertEqual(response["stopped"], "",
                         "an idle stop must not claim it stopped something")

    def test_a_bare_stop_still_arms_the_legacy_engine_flags(self):
        """No registered job means a pre-registration run is still stoppable."""
        response = routes.stop_task()

        self.assertEqual(response["cancelled"], [])
        self.assertTrue(browser_agent.stop_requested())
        self.assertTrue(research_service.stop_requested())

    def test_the_response_names_the_job_it_stopped(self):
        task = job_registry.new_job(kind="browser", label="task")

        response = routes.stop_task()

        self.assertEqual(response["stopped"], task.job_id)
        self.assertEqual(response["cancelled"], [task.job_id])

    def test_a_stop_never_raises_when_the_registry_fails(self):
        from unittest.mock import patch

        with patch.object(job_registry, "request_stop",
                          side_effect=RuntimeError("registry down")), \
             patch.object(request_registry, "interrupt_job",
                          side_effect=RuntimeError("registry down")):
            response = routes.stop_task(job_id="job-anything")

        self.assertTrue(response["ok"])
        self.assertEqual(response["cancelled"], [])


class RequestScopedCancelTests(_StopTestCase):
    """The turn manager uses the request-scoped cancel, not /task/stop."""

    def test_the_voice_turn_manager_cancels_by_request_id(self):
        import backend.voice_mode as voice_mode

        self.assertIn("/ask/cancel/", voice_mode._cancel_backend_request.__doc__
                      or "")

    def test_ask_cancel_is_request_scoped_and_idempotent(self):
        first_job = job_registry.new_job(kind="request", label="first")
        state = self._request("req-one", "one", first_job)
        other_job = job_registry.new_job(kind="request", label="other")
        other = self._request("req-two", "two", other_job)

        result = routes.cancel_request(state.request_id, "barge-in")

        self.assertTrue(result["cancelled"])
        self.assertTrue(state.done)
        self.assertFalse(other.done, "a request-scoped cancel hit another one")
        # Idempotent: the same call again reports why it did nothing.
        again = routes.cancel_request(state.request_id, "barge-in")
        self.assertFalse(again["cancelled"])
        self.assertIn(again["reason"], ("already_interrupted", "already_finished"))
        # And an unknown id is a safe no-op.
        unknown = routes.cancel_request("req-missing")
        self.assertEqual(unknown["reason"], "unknown_request")

    def test_a_cancelled_request_cancels_its_own_worker_job(self):
        job = job_registry.new_job(kind="request", label="worker")
        state = self._request("req-worker", "worker", job)

        routes.cancel_request(state.request_id, "barge-in")

        self.assertTrue(job.cancelled, "the worker kept running after a cancel")


class JobRegistrySelectionTests(unittest.TestCase):
    """The selection primitives the route relies on."""

    def tearDown(self):
        for job in job_registry.live_jobs():
            job.finish()

    def test_exclude_kinds_filters_the_candidate_set(self):
        chat = job_registry.new_job(kind="request", label="chat")
        task = job_registry.new_job(kind="browser", label="task")

        newest = job_registry.newest_job(exclude_kinds=("request",))

        self.assertIs(newest, task)
        self.assertIsNot(newest, chat)

    def test_excluding_everything_yields_no_target(self):
        job_registry.new_job(kind="request", label="chat")

        self.assertIsNone(job_registry.newest_job(exclude_kinds=("request",)))
        self.assertEqual(
            job_registry.request_stop(exclude_kinds=("request",)), [])

    def test_request_stop_keeps_its_own_default_behaviour(self):
        chat = job_registry.new_job(kind="request", label="chat")

        self.assertEqual(job_registry.request_stop(), [chat.job_id])


class InterruptJobTests(_StopTestCase):
    """interrupt_job is the precise replacement for interrupt_active."""

    def test_interrupt_job_matches_by_worker_job_id(self):
        job = job_registry.new_job(kind="request", label="a")
        state = self._request("req-a", "a", job)
        self._request("req-b", "b")

        interrupted = request_registry.interrupt_job([job.job_id], "stopped")

        self.assertEqual(interrupted, [state.request_id])
        self.assertTrue(state.done)

    def test_interrupt_job_with_no_ids_touches_nothing(self):
        state = self._request("req-c", "c")

        self.assertEqual(request_registry.interrupt_job([], "stopped"), [])
        self.assertEqual(request_registry.interrupt_job(None, "stopped"), [])
        self.assertFalse(state.done)

    def test_interrupt_active_is_still_available_for_worker_replacement(self):
        self._request("req-d", "d")

        self.assertEqual(
            request_registry.interrupt_active("supervisor replacement"), 1)


if __name__ == "__main__":
    unittest.main()
