"""P1-10 — a paused job must not burn CPU, and a kill must not hold the lock.

``cancel_job`` used to run ``taskkill`` (up to 5s) and ``proc.wait(2)`` WHILE
HOLDING the job registry lock, so the next request's ``new_job`` queued behind
an OS call that had nothing to do with it. The pause wait also re-checked its
flags on a fixed 250ms poll, so a cancel could take a quarter second to reach a
parked job.
"""

import threading
import time
import unittest
from unittest.mock import patch

from backend.services import jobs


class _SlowProc:
    """A Popen stand-in whose kill takes real time (never exits on its own)."""

    def __init__(self, pid=4242, kill_seconds=0.6):
        self.pid = pid
        self.kill_seconds = kill_seconds
        self.terminated = threading.Event()
        self.killed = threading.Event()
        self.waited = 0

    def poll(self):
        return None                      # looks alive until terminate() lands

    def terminate(self):
        self.terminated.set()

    def kill(self):
        self.killed.set()

    def wait(self, timeout=None):
        self.waited += 1
        # The kill thread is parked here: this is the window in which the old
        # code held the registry lock.
        time.sleep(self.kill_seconds)
        return 0


class _JobTestCase(unittest.TestCase):
    def setUp(self):
        self._launched = []
        self.addCleanup(self._drain)

    def _drain(self):
        for job in list(jobs._jobs.values()):
            try:
                job.finish()
            except Exception:
                pass

    def make_job(self, **kwargs):
        job = jobs.new_job(kind=kwargs.pop("kind", "unit"), **kwargs)
        self.addCleanup(job.finish)
        return job


class PauseGateTests(_JobTestCase):
    """[P1-10] A paused job parks on an event: no spin, prompt wake-up."""

    def test_a_paused_job_consumes_no_cpu_while_parked(self):
        job = self.make_job()
        job.pause()
        started = threading.Event()
        finished = threading.Event()
        cpu = {}

        def waiter():
            started.set()
            before = time.process_time()
            job.wait_if_paused()
            cpu["used"] = time.process_time() - before
            finished.set()

        thread = threading.Thread(target=waiter, daemon=True)
        thread.start()
        self.assertTrue(started.wait(1.0))
        time.sleep(0.4)                   # a spin would burn ~0.4s of CPU here

        self.assertFalse(finished.is_set(), "the waiter must still be parked")
        job.resume()
        self.assertTrue(finished.wait(2.0))
        thread.join(2.0)

        self.assertLess(cpu.get("used", 1.0), 0.05,
                        "a paused job was spinning on the CPU")

    def test_resume_starts_the_job_promptly(self):
        job = self.make_job()
        job.pause()
        parked = threading.Event()
        result = {}

        def waiter():
            parked.set()
            started = time.monotonic()
            result["ok"] = job.wait_if_paused()
            result["elapsed"] = time.monotonic() - started

        thread = threading.Thread(target=waiter, daemon=True)
        thread.start()
        self.assertTrue(parked.wait(1.0))
        time.sleep(0.05)                  # let it reach the gate

        job.resume()
        thread.join(2.0)

        self.assertTrue(result.get("ok"))
        self.assertLess(result.get("elapsed", 5.0), 0.15,
                        "resume did not wake the paused job promptly")

    def test_cancel_wakes_a_paused_job_promptly(self):
        """The old fixed poll made a cancel wait out the rest of its interval."""
        job = self.make_job()
        job.pause()
        parked = threading.Event()
        result = {}

        def waiter():
            parked.set()
            started = time.monotonic()
            result["ok"] = job.wait_if_paused()
            result["elapsed"] = time.monotonic() - started

        thread = threading.Thread(target=waiter, daemon=True)
        thread.start()
        self.assertTrue(parked.wait(1.0))
        time.sleep(0.05)

        job.cancel("stopped")
        thread.join(2.0)

        self.assertFalse(result.get("ok"), "a cancelled wait must report stop")
        self.assertLess(result.get("elapsed", 5.0), 0.15,
                        "cancel did not wake the paused job promptly")

    def test_a_paused_job_past_its_deadline_is_released(self):
        job = self.make_job(timeout=0.3)
        job.pause()

        self.assertFalse(job.wait_if_paused(), "the deadline must release it")
        self.assertFalse(job.paused)
        self.assertTrue(job.expired())

    def test_resume_does_not_resurrect_a_cancelled_job(self):
        job = self.make_job()
        job.cancel("stopped")

        job.resume()                      # must be a no-op for the token

        self.assertTrue(job.cancelled)
        self.assertFalse(job.paused)
        with self.assertRaises(jobs.Cancelled):
            job.checkpoint()


class AsyncKillTests(_JobTestCase):
    """[P1-10] The kill is an OS call: it must not run under the registry lock."""

    def _attach_slow_proc(self, job, kill_seconds=0.6, kill_tree=None):
        proc = _SlowProc(kill_seconds=kill_seconds)
        job.attach(proc)
        return proc

    def test_new_job_succeeds_while_another_job_is_being_killed(self):
        """THE regression test: new_job must not queue behind taskkill."""
        victim = self.make_job()
        with patch.object(jobs, "_kill_tree", return_value=None) as kill_tree:
            self._attach_slow_proc(victim)

            cancelling = threading.Event()

            def cancel():
                cancelling.set()
                jobs.cancel_job(victim.job_id, "stopping")

            thread = threading.Thread(target=cancel, daemon=True)
            thread.start()
            self.assertTrue(cancelling.wait(1.0))
            time.sleep(0.1)               # the kill thread is now inside wait()

            started = time.monotonic()
            fresh = self.make_job(kind="unit-new")
            elapsed = time.monotonic() - started

            self.assertLess(elapsed, 0.25,
                            "new_job blocked behind an in-flight process kill")
            self.assertEqual(fresh.job_id, jobs.get_job(fresh.job_id).job_id)
            self.assertTrue(victim.wait_for_kill(5.0),
                            "the kill never completed")
            self.assertTrue(kill_tree.called, "the child tree was not killed")
            thread.join(2.0)

    def test_cancelling_is_reported_while_the_kill_is_in_flight(self):
        victim = self.make_job()
        with patch.object(jobs, "_kill_tree", return_value=None):
            self._attach_slow_proc(victim)
            jobs.cancel_job(victim.job_id, "stopping")

            snapshot = victim.to_dict()
            self.assertTrue(snapshot["cancelled"])
            # …and the kill is reported rather than waited on.
            self.assertIn("cancelling", snapshot)
            self.assertIn("state", snapshot)
            self.assertTrue(victim.wait_for_kill(5.0))

    def test_cancel_still_kills_the_child_process(self):
        victim = self.make_job()
        with patch.object(jobs, "_kill_tree", return_value=None) as kill_tree:
            proc = self._attach_slow_proc(victim, kill_seconds=0.05)
            cancelled = jobs.cancel_job(victim.job_id, "stopping")

            self.assertEqual(cancelled, [victim.job_id])
            self.assertTrue(victim.wait_for_kill(5.0))

        self.assertEqual(kill_tree.call_args.args[0], proc.pid)
        self.assertTrue(proc.terminated.is_set(),
                        "the child survived the cancellation")

    def test_an_unattached_job_cancel_does_not_create_a_kill_thread(self):
        victim = self.make_job()

        jobs.cancel_job(victim.job_id, "stopping")

        self.assertFalse(victim.killing)
        self.assertTrue(victim.wait_for_kill(1.0))

    def test_attach_after_cancel_still_kills_the_late_process(self):
        """A process claimed by a cancelled job must not be orphaned."""
        victim = self.make_job()
        victim.cancel("stopping")
        proc = _SlowProc(kill_seconds=0.0)

        with patch.object(jobs, "_kill_tree", return_value=None):
            victim.attach(proc)

        self.assertTrue(proc.terminated.is_set(),
                        "attach() let a cancelled job keep a live child")

    def test_resume_during_a_cancel_neither_deadlocks_nor_resurrects(self):
        victim = self.make_job()
        victim.pause()
        with patch.object(jobs, "_kill_tree", return_value=None):
            self._attach_slow_proc(victim)

            started = threading.Event()

            def cancel():
                started.set()
                jobs.cancel_job(victim.job_id, "stopping")

            thread = threading.Thread(target=cancel, daemon=True)
            thread.start()
            self.assertTrue(started.wait(1.0))
            time.sleep(0.05)

            began = time.monotonic()
            victim.resume()               # must not block on the kill
            resume_elapsed = time.monotonic() - began

            began = time.monotonic()
            fresh = self.make_job(kind="unit-after")   # must not block either
            registry_elapsed = time.monotonic() - began

            self.assertTrue(victim.wait_for_kill(5.0))
            thread.join(2.0)

        self.assertLess(resume_elapsed, 0.1, "resume deadlocked on the kill")
        self.assertLess(registry_elapsed, 0.25,
                        "the registry deadlocked behind the kill")
        self.assertTrue(victim.cancelled)
        self.assertFalse(victim.paused)
        self.assertIsNotNone(fresh.job_id)
        with self.assertRaises(jobs.Cancelled):
            victim.checkpoint()

    def test_the_registry_lock_is_free_during_the_kill(self):
        victim = self.make_job()
        with patch.object(jobs, "_kill_tree", return_value=None):
            self._attach_slow_proc(victim, kill_seconds=0.5)
            jobs.cancel_job(victim.job_id, "stopping")

            began = time.monotonic()
            acquired = jobs._lock.acquire(timeout=0.2)
            waited = time.monotonic() - began
            if acquired:
                jobs._lock.release()

            self.assertTrue(acquired,
                            "the registry lock was held across the kill")
            self.assertLess(waited, 0.2)
            self.assertTrue(victim.wait_for_kill(5.0))


class ContractTests(_JobTestCase):
    """The F20 cancellation contract and the reported shape are unchanged."""

    def test_the_state_values_keep_their_meaning(self):
        job = self.make_job()
        self.assertEqual(job.state, "running")

        job.pause()
        self.assertEqual(job.state, "paused")
        self.assertTrue(job.to_dict()["paused"])

        job.resume()
        self.assertEqual(job.state, "running")

        job.cancel()
        self.assertEqual(job.state, "cancelled")
        self.assertTrue(job.to_dict()["cancelled"])

    def test_to_dict_keeps_every_existing_key(self):
        job = self.make_job(label="unit label")
        snapshot = job.to_dict()

        for key in ("job_id", "kind", "label", "generation", "cancelled",
                    "paused", "elapsed", "remaining"):
            self.assertIn(key, snapshot, "an existing job field vanished")
        self.assertEqual(snapshot["label"], "unit label")

    def test_an_unaddressed_stop_with_nothing_running_is_still_a_no_op(self):
        self.assertEqual(jobs.cancel_job(), [])


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
