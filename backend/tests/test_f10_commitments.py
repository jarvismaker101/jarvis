"""F10 — persistent commitments and proactivity (the acknowledged outbox).

Acceptance (audit report): "Restart delivers existing reminders; callback
failure retries; concurrent ticks notify once; cancelling A preserves B;
tomorrow at 9 resolves correctly."

Baseline defects pinned here:
  * the scheduler only started as a side effect of ADDING a reminder, so a
    restart with stored reminders never delivered them;
  * delivery was marked complete BEFORE the callback ran, so a failed
    notification was lost forever;
  * two concurrent ticks could both deliver the same reminder;
  * naming a reminder cancelled whichever one was newest;
  * "tomorrow at 9" ignored the day qualifier whenever a clock time was given.
"""

import os
import tempfile
import threading
import time
import unittest
import datetime

from backend.core import memory_store


class CommitmentTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        memory_store.configure(os.path.join(self._tmp.name, "mem.db"))
        memory_store.MEMORY_ENABLED = True
        memory_store._delivery_cb = None
        memory_store.stop_scheduler()

    def tearDown(self):
        memory_store.set_commitment_delivery(None)
        memory_store.stop_scheduler()
        memory_store.close()
        self._tmp.cleanup()

    def _row(self, cid):
        return memory_store.get_commitment(cid) or {}


class RestartDeliveryTests(CommitmentTestBase):
    def test_stored_reminder_is_delivered_after_a_restart(self):
        due = time.time() - 60
        cid = memory_store.add_commitment(
            "call the dentist", trigger_kind="deadline", due_at=due,
            expires_at=due + 86400)
        # Simulate a full restart: close the store, reopen the SAME file.
        memory_store.close()
        memory_store.configure(os.path.join(self._tmp.name, "mem.db"))
        delivered = []
        memory_store.set_commitment_delivery(delivered.append)

        self.assertTrue(memory_store.start_scheduler(),
                        "the backend must own the scheduler at startup")
        self.assertTrue(memory_store.scheduler_running())

        ids = memory_store.commitment_tick()
        self.assertEqual(ids, [cid])
        self.assertEqual(len(delivered), 1)
        self.assertEqual(self._row(cid)["status"], "delivered")

    def test_scheduler_start_is_idempotent(self):
        memory_store.start_scheduler()
        first = memory_store._scheduler_thread
        memory_store.start_scheduler()
        self.assertIs(memory_store._scheduler_thread, first)


class AcknowledgedDeliveryTests(CommitmentTestBase):
    def _due_now(self, text="stand up"):
        due = time.time() - 5
        return memory_store.add_commitment(
            text, trigger_kind="deadline", due_at=due, expires_at=due + 86400)

    def test_callback_failure_is_retried_not_lost(self):
        cid = self._due_now()
        calls = []

        def flaky(commitment):
            calls.append(commitment["id"])
            if len(calls) == 1:
                raise RuntimeError("tts busy")
            return True

        memory_store.set_commitment_delivery(flaky)
        self.assertEqual(memory_store.commitment_tick(), [],
                         "a failed delivery must not be reported as delivered")
        row = self._row(cid)
        self.assertEqual(row["status"], "armed")
        self.assertEqual(row["attempts"], 1)
        self.assertGreater(row["next_attempt_at"], time.time())
        self.assertIn("tts busy", row["last_error"] or "")

        # Not due again yet — the backoff holds it back.
        self.assertEqual(memory_store.commitment_tick(), [])
        self.assertEqual(len(calls), 1)

        # After the backoff window the retry succeeds and is acknowledged.
        later = row["next_attempt_at"] + 1
        self.assertEqual(memory_store.commitment_tick(later), [cid])
        self.assertEqual(self._row(cid)["status"], "delivered")
        self.assertEqual(len(calls), 2)

    def test_false_return_from_callback_counts_as_failure(self):
        cid = self._due_now()
        memory_store.set_commitment_delivery(lambda c: False)
        memory_store.commitment_tick()
        self.assertEqual(self._row(cid)["status"], "armed")
        self.assertEqual(self._row(cid)["attempts"], 1)

    def test_attempts_are_bounded_and_failure_becomes_visible(self):
        cid = self._due_now()
        memory_store.set_commitment_delivery(
            lambda c: (_ for _ in ()).throw(RuntimeError("nope")))
        now = time.time()
        for _ in range(memory_store.COMMITMENT_MAX_ATTEMPTS):
            memory_store.commitment_tick(now)
            now += 10_000
        row = self._row(cid)
        self.assertEqual(row["status"], "failed")
        self.assertGreaterEqual(row["attempts"],
                                memory_store.COMMITMENT_MAX_ATTEMPTS)

    def test_no_callback_leaves_the_reminder_armed(self):
        cid = self._due_now()
        memory_store.set_commitment_delivery(None)
        self.assertEqual(memory_store.commitment_tick(), [])
        self.assertEqual(self._row(cid)["status"], "armed")

    def test_stale_claim_is_recovered_after_a_crash(self):
        cid = self._due_now()
        self.assertTrue(memory_store.claim_commitment(cid))
        self.assertEqual(self._row(cid)["status"], "delivering")
        self.assertEqual(memory_store.reclaim_stale_claims(time.time()), 0,
                         "a fresh claim must not be stolen")
        stolen = memory_store.reclaim_stale_claims(
            time.time() + memory_store.COMMITMENT_CLAIM_TIMEOUT + 1)
        self.assertEqual(stolen, 1)
        self.assertEqual(self._row(cid)["status"], "armed")


class ConcurrentTickTests(CommitmentTestBase):
    def test_concurrent_ticks_notify_exactly_once(self):
        due = time.time() - 5
        cid = memory_store.add_commitment(
            "drink water", trigger_kind="deadline", due_at=due,
            expires_at=due + 86400)
        seen = []
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def cb(commitment):
            with lock:
                seen.append(commitment["id"])
            time.sleep(0.02)
            return True

        memory_store.set_commitment_delivery(cb)
        results = []

        def tick():
            barrier.wait()
            results.append(memory_store.commitment_tick())

        threads = [threading.Thread(target=tick) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)

        self.assertEqual(seen, [cid], "a reminder must be announced once")
        self.assertEqual(len([r for r in results if r]), 1,
                         "exactly one tick may win the claim")
        self.assertEqual(self._row(cid)["status"], "delivered")


class NamedCancellationTests(CommitmentTestBase):
    def _arm(self, text):
        due = time.time() - 5
        return memory_store.add_commitment(
            text, trigger_kind="deadline", due_at=due, expires_at=due + 86400)

    def test_cancelling_a_preserves_b(self):
        dentist = self._arm("remind me to call the dentist")
        taxes = self._arm("remind me about the tax deadline")
        cancelled = memory_store.cancel_commitment_by_text(
            "cancel the dentist reminder")
        self.assertEqual(cancelled, 1)
        self.assertEqual(self._row(dentist)["status"], "cancelled")
        self.assertEqual(self._row(taxes)["status"], "armed")

    def test_resolve_identifies_by_content_not_recency(self):
        self._arm("remind me to call the dentist")
        taxes = self._arm("remind me about the tax deadline")
        found = memory_store.resolve_commitment("the tax deadline")
        self.assertIsNotNone(found)
        self.assertEqual(found["id"], taxes)

    def test_unknown_name_cancels_nothing(self):
        self._arm("remind me to call the dentist")
        self.assertEqual(
            memory_store.cancel_commitment_by_text("cancel the rocket launch"),
            0)
        self.assertEqual(
            memory_store.list_commitments(status="armed")[0]["status"],
            "armed")

    def test_cancelling_by_id_is_still_exact(self):
        a = self._arm("first reminder about alpha")
        b = self._arm("second reminder about beta")
        self.assertEqual(memory_store.cancel_commitment(b), 1)
        self.assertEqual(self._row(a)["status"], "armed")
        self.assertEqual(self._row(b)["status"], "cancelled")


class TomorrowParsingTests(CommitmentTestBase):
    def _at(self, y, mo, d, h, mi):
        return datetime.datetime(y, mo, d, h, mi).timestamp()

    def test_tomorrow_at_9_is_tomorrow_at_9(self):
        now = self._at(2026, 3, 10, 15, 0)  # 3pm
        due, _expires, cleaned = memory_store.parse_deadline(
            "remind me to call mum tomorrow at 9", now=now)
        resolved = datetime.datetime.fromtimestamp(due)
        self.assertEqual((resolved.year, resolved.month, resolved.day),
                         (2026, 3, 11))
        self.assertEqual((resolved.hour, resolved.minute), (9, 0))
        self.assertIn("call mum", cleaned)
        self.assertNotIn("tomorrow", cleaned.lower())

    def test_tomorrow_at_9_late_at_night_is_still_tomorrow(self):
        now = self._at(2026, 3, 10, 23, 50)
        due, _e, _c = memory_store.parse_deadline(
            "tomorrow at 9 stand up", now=now)
        resolved = datetime.datetime.fromtimestamp(due)
        self.assertEqual((resolved.month, resolved.day, resolved.hour),
                         (3, 11, 9))

    def test_plain_at_9_after_nine_means_tomorrow_morning(self):
        now = self._at(2026, 3, 10, 22, 0)
        due, _e, _c = memory_store.parse_deadline("at 9 call mum", now=now)
        resolved = datetime.datetime.fromtimestamp(due)
        self.assertEqual((resolved.day, resolved.hour), (11, 9))

    def test_plain_at_9_before_nine_means_today(self):
        now = self._at(2026, 3, 10, 7, 0)
        due, _e, _c = memory_store.parse_deadline("at 9 call mum", now=now)
        resolved = datetime.datetime.fromtimestamp(due)
        self.assertEqual((resolved.day, resolved.hour), (10, 9))

    def test_bare_tomorrow_keeps_its_24h_meaning(self):
        now = self._at(2026, 3, 10, 23, 50)
        due, _e, _c = memory_store.parse_deadline("tomorrow buy tickets", now=now)
        self.assertAlmostEqual(due - now, 86400, delta=2)

    def test_in_duration_still_wins(self):
        now = self._at(2026, 3, 10, 9, 0)
        due, _e, _c = memory_store.parse_deadline("in 30 minutes stretch", now=now)
        self.assertAlmostEqual(due, now + 1800, places=3)


if __name__ == "__main__":
    unittest.main()
