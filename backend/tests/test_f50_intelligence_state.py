"""F50 — Give the Backend Sole Ownership of Intelligence State.

Pins the finding's Acceptance clauses against real code paths:

1. **Typed/voice/background work shares authority/history.** Every effect
   surface runs through ONE job runtime and records into ONE sequenced
   journal (no per-surface gates or private logs).
2. **Setup creates backend jobs.** A local setup command is submitted as a
   typed ``setup`` backend job instead of running in the caller's process.
3. **New worker state supersedes old generations immediately.** A restarted
   worker's generation wins at once; a publish from the superseded
   incarnation/generation is rejected instead of overwriting newer truth,
   and expired incarnations are not served at all.
4. **Only designated owners schedule/write durable state.** A second live
   writer is refused; only the designated owner may run an effect that
   writes a durable resource.

F20 containment is exercised too: effect jobs are individually cancellable
through the same registry.
"""

import threading
import unittest

from backend.services import intelligence_state as istate
from backend.services import jobs


class _ResetMixin:
    def setUp(self):
        istate.reset()
        for job in jobs.live_jobs():
            job.finish()

    def tearDown(self):
        for job in jobs.live_jobs():
            job.finish()
        istate.reset()


class SharedAuthorityTests(_ResetMixin, unittest.TestCase):
    """Acceptance 1 — typed/voice/background share runtime and history."""

    def test_every_surface_runs_through_the_one_job_runtime(self):
        seen = {}

        def make_handler(kind, value):
            def handler(job):
                seen[kind] = {
                    "job_kind": job.kind,
                    "live": [j.job_id for j in jobs.live_jobs(kind)],
                    "turn_job": jobs.current_turn_job(),
                }
                return value
            return handler

        outcomes = []
        for kind in (jobs.EFFECT_TYPED, jobs.EFFECT_VOICE,
                     jobs.EFFECT_BACKGROUND):
            outcomes.append(istate.run_effect(
                kind, "%s-effect" % kind, make_handler(kind, kind),
                owner="backend"))

        for outcome, kind in zip(outcomes, (jobs.EFFECT_TYPED,
                                            jobs.EFFECT_VOICE,
                                            jobs.EFFECT_BACKGROUND)):
            self.assertEqual(outcome.status, "completed")
            self.assertEqual(outcome.result, kind)
            self.assertEqual(seen[kind]["job_kind"], kind)
            self.assertIn(outcome.job_id, seen[kind]["live"])
            self.assertEqual(seen[kind]["turn_job"].job_id, outcome.job_id)

    def test_one_sequenced_history_holds_every_surface(self):
        for kind in (jobs.EFFECT_TYPED, jobs.EFFECT_VOICE,
                     jobs.EFFECT_BACKGROUND):
            istate.run_effect(kind, "%s-effect" % kind, lambda job: None,
                              owner="backend")
        events = istate.journal.events()
        self.assertEqual({e.kind for e in events},
                         {jobs.EFFECT_TYPED, jobs.EFFECT_VOICE,
                          jobs.EFFECT_BACKGROUND})
        for kind in (jobs.EFFECT_TYPED, jobs.EFFECT_VOICE,
                     jobs.EFFECT_BACKGROUND):
            phases = [e.phase for e in istate.journal.events(kind=kind)]
            self.assertEqual(phases, ["submitted", "started", "finished"])
        seqs = [e.seq for e in events]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(set(seqs)), len(seqs))
        # Authority is recorded, not implied.
        self.assertTrue(all(e.authority for e in events))

    def test_finished_effect_is_released_from_the_registry(self):
        outcome = istate.run_effect(jobs.EFFECT_TYPED, "quick",
                                    lambda job: "ok")
        self.assertIsNone(jobs.get_job(outcome.job_id))
        self.assertEqual(jobs.live_jobs(jobs.EFFECT_TYPED), [])

    def test_untyped_effect_kind_is_refused(self):
        with self.assertRaises(jobs.UnknownEffectKind):
            istate.run_effect("banana", "untyped", lambda job: None)
        with self.assertRaises(jobs.UnknownEffectKind):
            jobs.new_effect_job("banana")
        self.assertEqual(istate.journal.events(), [])

    def test_cancelled_effect_stops_without_the_handler_doing_more(self):
        handled = []

        def handler(job):
            handled.append(True)
            job.cancel("user said stop")
            job.checkpoint()   # F20: raises Cancelled for a cancelled job
            handled.append("should not happen")

        outcome = istate.run_effect(jobs.EFFECT_VOICE, "voice-cancel",
                                    handler)
        self.assertEqual(outcome.status, "stopped")
        self.assertEqual(handled, [True])
        self.assertIn("stopped", [e.phase for e in istate.journal.events()])

    def test_failing_effect_is_journalled_and_propagated(self):
        def handler(job):
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            istate.run_effect(jobs.EFFECT_BACKGROUND, "bad", handler)
        failed = istate.journal.events(phase="failed")
        self.assertEqual(len(failed), 1)
        self.assertIn("boom", failed[0].detail.get("error", ""))
        self.assertTrue(jobs.live_jobs(jobs.EFFECT_BACKGROUND) == [])


class SetupOwnershipTests(_ResetMixin, unittest.TestCase):
    """Acceptance 2 — setup creates backend jobs."""

    def test_setup_runs_as_a_typed_backend_job(self):
        observed = {}
        done = threading.Event()

        def handler(job):
            observed["job_kind"] = job.kind
            observed["job_id"] = job.job_id
            observed["in_runtime"] = jobs.current_turn_job() is job
            observed["live_setup_jobs"] = [
                j.job_id for j in jobs.live_jobs(jobs.EFFECT_SETUP)]
            done.set()

        job = istate.submit_setup_effect("normal-setup", handler)
        self.assertTrue(done.wait(timeout=5.0), "setup effect never ran")
        self.assertEqual(observed["job_kind"], jobs.EFFECT_SETUP)
        self.assertEqual(observed["job_id"], job.job_id)
        self.assertIn(job.job_id, observed["live_setup_jobs"])
        self.assertTrue(observed["in_runtime"])
        # The backend job is addressable while it runs (F20).
        self.assertEqual(job.kind, jobs.EFFECT_SETUP)
        job.finish()

    def test_setup_history_is_recorded_with_its_job_id(self):
        done = threading.Event()

        def handler(job):
            done.set()

        job = istate.submit_setup_effect("normal-setup", handler)
        self.assertTrue(done.wait(timeout=5.0))
        for _ in range(50):
            phases = [e.phase for e in istate.journal.events(
                kind=jobs.EFFECT_SETUP)]
            if "finished" in phases:
                break
            threading.Event().wait(0.02)
        events = istate.journal.events(kind=jobs.EFFECT_SETUP)
        self.assertEqual([e.phase for e in events][0], "submitted")
        self.assertIn("finished", [e.phase for e in events])
        self.assertTrue(all(e.job_id == job.job_id for e in events))

    def test_setup_effect_can_be_stopped_by_job_id(self):
        release = threading.Event()
        cancelled = []

        def handler(job):
            release.wait(timeout=5.0)
            try:
                job.checkpoint()
            except jobs.Cancelled:
                cancelled.append(True)

        job = istate.submit_setup_effect("normal-setup", handler)
        stopped = jobs.request_stop(job.job_id, reason="user said stop")
        self.assertEqual(stopped, [job.job_id])
        release.set()
        for _ in range(100):
            if cancelled:
                break
            threading.Event().wait(0.02)
        self.assertEqual(cancelled, [True])
        job.finish()


class WorkerGenerationTests(_ResetMixin, unittest.TestCase):
    """Acceptance 3 — new worker state supersedes old generations at once."""

    def test_supersede_drops_the_old_incarnations_state_immediately(self):
        registry = istate.worker_states
        first = registry.supersede(istate.ROLE_VOICE, "voice@100", pid=100,
                                   state={"status": "listening"})
        self.assertEqual(first.generation, 1)
        self.assertEqual(registry.current(istate.ROLE_VOICE).pid, 100)

        registry.supersede(istate.ROLE_VOICE, "voice@200", pid=200)
        # The old worker's truth is gone the moment the new one exists.
        self.assertIsNone(registry.current(istate.ROLE_VOICE))
        self.assertEqual(registry.generation(istate.ROLE_VOICE), 2)

    def test_stale_incarnation_publish_is_rejected(self):
        registry = istate.worker_states
        registry.supersede(istate.ROLE_VOICE, "voice@100", pid=100)
        registry.supersede(istate.ROLE_VOICE, "voice@200", pid=200)

        outcome = registry.publish(istate.ROLE_VOICE, "voice@100",
                                   {"status": "listening"})
        self.assertFalse(outcome.accepted)
        self.assertTrue(outcome.stale)
        stale_generation = registry.publish(
            istate.ROLE_VOICE, "voice@200", {"status": "listening"},
            generation=1)
        self.assertFalse(stale_generation.accepted)
        self.assertTrue(stale_generation.stale)

        fresh = registry.publish(istate.ROLE_VOICE, "voice@200",
                                 {"status": "hearing"}, pid=200)
        self.assertTrue(fresh.accepted)
        self.assertEqual(fresh.generation, 2)
        self.assertEqual(registry.current(istate.ROLE_VOICE).state,
                         {"status": "hearing"})

    def test_expired_incarnation_state_is_not_served(self):
        registry = istate.worker_states
        registry.supersede(istate.ROLE_VOICE, "voice@100", pid=100)
        registry.publish(istate.ROLE_VOICE, "voice@100",
                         {"status": "listening"}, ttl=0.0)
        self.assertIsNone(registry.current(istate.ROLE_VOICE))

    def test_voice_publication_is_journalled_with_its_generation(self):
        registry = istate.worker_states
        registry.supersede(istate.ROLE_VOICE, "voice@100", pid=100)
        outcome = istate.note_voice_state(istate.ROLE_VOICE, "voice@100",
                                          {"status": "speaking"})
        self.assertTrue(outcome.accepted)
        published = [e for e in istate.journal.events()
                     if e.effect == "voice-state"]
        self.assertEqual(len(published), 1)
        self.assertEqual(published[0].detail["generation"], 1)
        self.assertEqual(published[0].authority, "voice@100")


class DurableOwnershipTests(_ResetMixin, unittest.TestCase):
    """Acceptance 4 — only designated owners write durable state."""

    def test_second_live_writer_is_refused(self):
        istate.durable.claim(istate.RESOURCE_MEMORY, "backend")
        self.assertEqual(istate.durable.owner_of(istate.RESOURCE_MEMORY),
                         "backend")
        with self.assertRaises(istate.NotDurableOwner):
            istate.durable.claim(istate.RESOURCE_MEMORY, "voice-io")
        self.assertEqual(istate.durable.owner_of(istate.RESOURCE_MEMORY),
                         "backend")

    def test_non_owner_effect_on_a_durable_resource_is_refused(self):
        istate.durable.claim(istate.RESOURCE_MEMORY, "backend")
        ran = []
        with self.assertRaises(istate.NotDurableOwner):
            istate.run_effect(
                jobs.EFFECT_VOICE, "memory-write", lambda job: ran.append(1),
                owner="voice-io", resource=istate.RESOURCE_MEMORY)
        self.assertEqual(ran, [])
        self.assertEqual(istate.journal.events(), [],
                         "a refused effect must not enter the shared history")

    def test_designated_owner_effect_runs(self):
        istate.durable.claim(istate.RESOURCE_MEMORY, "backend")
        outcome = istate.run_effect(
            jobs.EFFECT_BACKGROUND, "memory-write", lambda job: "written",
            owner="backend", resource=istate.RESOURCE_MEMORY)
        self.assertEqual(outcome.status, "completed")
        self.assertEqual(outcome.result, "written")

    def test_expired_lease_can_be_taken_over(self):
        istate.durable.claim(istate.RESOURCE_CHECKPOINTS, "voice-io", ttl=0.0)
        self.assertEqual(istate.durable.owner_of(istate.RESOURCE_CHECKPOINTS),
                         "")
        self.assertTrue(
            istate.durable.claim(istate.RESOURCE_CHECKPOINTS, "backend"))

    def test_released_lease_can_be_reclaimed(self):
        istate.durable.claim(istate.RESOURCE_APPROVALS, "voice-io")
        self.assertTrue(istate.durable.release(istate.RESOURCE_APPROVALS,
                                               "voice-io"))
        self.assertTrue(istate.durable.claim(istate.RESOURCE_APPROVALS,
                                             "backend"))

    def test_one_playback_owner_at_a_time(self):
        istate.designate_playback_owner("backend")
        self.assertEqual(istate.playback_owner(), "backend")
        with self.assertRaises(istate.NotDurableOwner):
            istate.designate_playback_owner("voice-io")
        self.assertEqual(istate.playback_owner(), "backend")
        istate.release_playback_owner("backend")
        istate.designate_playback_owner("voice-io")
        self.assertEqual(istate.playback_owner(), "voice-io")


class JournalTransactionTests(_ResetMixin, unittest.TestCase):
    """Centralized TRANSACTIONAL events (F50 correction)."""

    def test_committed_transaction_appends_all_events(self):
        with istate.journal.transaction():
            istate.journal.append("typed", "checkpoint", "step-1")
            istate.journal.append("typed", "checkpoint", "step-2")
        self.assertEqual([e.effect for e in istate.journal.events()],
                         ["step-1", "step-2"])

    def test_failed_transaction_appends_nothing(self):
        with self.assertRaises(RuntimeError):
            with istate.journal.transaction():
                istate.journal.append("typed", "checkpoint", "step-1")
                raise RuntimeError("rollback")
        self.assertEqual(istate.journal.events(), [])

    def test_checkpoint_and_approval_notes_share_the_history(self):
        istate.note_checkpoint("suspended-run", job_id="job-9",
                               authority="backend")
        istate.note_approval("screen-plan", job_id="job-9",
                             authority="backend")
        phases = [e.phase for e in istate.journal.events()]
        self.assertEqual(phases, ["checkpoint", "approval"])


if __name__ == "__main__":
    unittest.main()
