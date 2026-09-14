"""Fable-5 audit G9 tests — Memory & Continuity (F06/F07/F10/F09).

The store runs against a per-test tmp SQLite file (memory_store.configure),
the scheduler is stopped and the delivery callback cleared between tests so
nothing leaks across the suite.
"""

import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from backend.core import memory_store  # noqa: E402


class G9MemoryTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        memory_store.configure(os.path.join(self._tmp.name, "mem.db"))
        memory_store.MEMORY_ENABLED = True
        memory_store._delivery_cb = None
        memory_store.stop_scheduler()
        self.delivered = []
        memory_store.set_commitment_delivery(self.delivered.append)

    def tearDown(self):
        memory_store.set_commitment_delivery(None)
        memory_store.stop_scheduler()
        memory_store.close()
        self._tmp.cleanup()


class FactsTests(G9MemoryTestBase):
    def test_remember_supersedes_never_overwrites(self):
        memory_store.remember("project", "phoenix")
        new_id = memory_store.remember("project", "atlas")
        rows = memory_store.relevant_facts("project")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value"], "atlas")
        self.assertEqual(rows[0]["id"], new_id)
        # Revision chain preserved: the old fact points at the successor.
        old = memory_store._conn().execute(
            "SELECT * FROM facts WHERE value = 'phoenix'").fetchone()
        self.assertEqual(old["superseded_by"], new_id)

    def test_remember_masks_secrets(self):
        memory_store.remember("deployment", "api_key=sk-supersecret123")
        rows = memory_store.relevant_facts("deployment")
        self.assertEqual(len(rows), 1)
        self.assertNotIn("sk-supersecret123", rows[0]["value"])

    def test_remember_rejects_empty(self):
        self.assertIsNone(memory_store.remember("", "value"))
        self.assertIsNone(memory_store.remember("subject", ""))

    def test_forget_is_scoped_and_blanket_proof(self):
        memory_store.remember("alpha", "one")
        memory_store.remember("beta", "two")
        self.assertEqual(memory_store.forget(subject="alpha"), 1)
        self.assertEqual(memory_store.relevant_facts("alpha"), [])
        rows = memory_store.relevant_facts("beta")
        self.assertEqual(len(rows), 1)
        # No criterion -> forgets nothing.
        self.assertEqual(memory_store.forget(), 0)
        self.assertEqual(len(memory_store.relevant_facts("beta")), 1)

    def test_memory_context_empty_store_is_empty(self):
        self.assertEqual(memory_store.memory_context("anything"), "")

    def test_memory_context_bounded_block(self):
        for i in range(6):
            memory_store.remember("topic%d" % i, "detail value %d" % i)
        block = memory_store.memory_context("topic0 topic1 detail")
        self.assertTrue(block.startswith("Known context from memory:"))
        self.assertLessEqual(len(block), memory_store.CONTEXT_BUDGET_CHARS)
        self.assertIn("topic0", block)

    def test_entities_upsert_and_lookup(self):
        memory_store.upsert_entity("phoenix", kind="project",
                                   display_name="Phoenix")
        ent = memory_store.lookup_entity("PHOENIX")
        self.assertEqual(ent["kind"], "project")
        self.assertEqual(ent["display_name"], "Phoenix")


class EventsTests(G9MemoryTestBase):
    def test_event_masks_secrets_and_bounds_summary(self):
        eid = memory_store.record_event(
            "research", "report ready token=abc123supersecret",
            detail={"query": "x", "api_key": "zzz-secret-zzz"},
            refs=["https://example.com/source"])
        row = memory_store._conn().execute(
            "SELECT * FROM events WHERE id = ?", (eid,)).fetchone()
        self.assertNotIn("abc123supersecret", row["summary"])
        self.assertNotIn("zzz-secret-zzz", row["detail"])
        refs = json.loads(row["refs"])
        self.assertEqual(refs, ["https://example.com/source"])

    def test_recent_events_newest_first(self):
        memory_store.record_event("research", "first finding")
        memory_store.record_event("research", "second finding")
        rows = memory_store.recent_events(limit=2)
        self.assertEqual(rows[0]["summary"], "second finding")
        self.assertEqual(rows[1]["summary"], "first finding")

    def test_find_events_matches_summary(self):
        memory_store.record_event("research", "quarterly pricing report done")
        memory_store.record_event("task_result", "[browser_agent/failed] oops")
        rows = memory_store.find_events("pricing report")
        self.assertEqual(len(rows), 1)
        self.assertIn("pricing", rows[0]["summary"])

    def test_record_task_outcome_captures_candidate_on_completed(self):
        # F09: capture is the PROCEDURE derived from the verified trace — a
        # completion with no committed steps is not capturable at all.
        memory_store.record_task_outcome(
            "browser_agent", "completed", "order groceries on amazon",
            summary="ordered groceries on amazon", evidence=["cart ok"],
            trace=[{"tool": "click_locator", "args": {"selector": "#cart"},
                    "observation": "cart page open", "ok": True}],
            verification=["cart page open"])
        evs = memory_store.find_events("groceries", kind="task_result")
        self.assertEqual(len(evs), 1)
        cands = memory_store.find_skills("groceries", status="candidate")
        self.assertEqual(len(cands), 1)
        self.assertEqual(json.loads(cands[0]["steps"]),
                         ["click_locator: selector=#cart"])
        # Not trusted as a procedure until approved — but the VERIFIED
        # outcome still grounds future plans (F09/F07): never an
        # "Approved skill" block, always the evidence.
        block = memory_store.recall_skills_for("groceries")
        self.assertNotIn("Approved skill", block)
        self.assertIn("Verified prior outcome", block)

    def test_record_task_outcome_traceless_completion_is_not_captured(self):
        memory_store.record_task_outcome(
            "browser_agent", "completed", "summarise the news",
            summary="summarised the news")
        self.assertEqual(
            memory_store.find_skills("summarise the news", status="candidate"),
            [])

    def test_record_task_outcome_failure_no_skill(self):
        memory_store.record_task_outcome(
            "opencode", "failed", "build the wheel", summary="crashed")
        self.assertEqual(memory_store.find_skills("wheel"), [])


class SkillsTests(G9MemoryTestBase):
    def test_promote_requires_candidate(self):
        sid = memory_store.record_skill_candidate(
            "check order status", "browser", "check order status on amazon")
        # F09: promotion is version-bound AND replay-validated.
        self.assertTrue(memory_store.promote_skill(
            sid, approved_version=1, replay_evidence="replay verified"))
        self.assertFalse(memory_store.promote_skill(
            sid, approved_version=1, replay_evidence="replay verified"))

    def test_repeated_failures_invalidate(self):
        sid = memory_store.record_skill_candidate(
            "export ledger", "shell", "export the ledger to csv")
        memory_store.promote_skill(sid, approved_version=1,
                                   replay_evidence="first run verified")
        for _ in range(memory_store._SKILL_INVALIDATE_FAILURES - 1):
            memory_store.note_skill_outcome(sid, success=False)
        self.assertEqual(
            len(memory_store.find_skills("export ledger", status="promoted")),
            1)  # 2 failures: still trusted
        memory_store.note_skill_outcome(sid, success=False)
        self.assertEqual(memory_store.find_skills("export ledger"), [])
        skills = memory_store.list_skills(limit=10)
        self.assertTrue(all(s["status"] == "invalidated"
                            for s in skills if s["id"] == sid))

    def test_success_validates(self):
        sid = memory_store.record_skill_candidate(
            "flag post", "browser", "flag the weekly post")
        memory_store.promote_skill(sid, approved_version=1,
                                   replay_evidence="verified run")
        memory_store.note_skill_outcome(sid, success=True)
        row = memory_store.find_skills("flag post", status="promoted")[0]
        self.assertEqual(row["success_count"], 1)
        self.assertIsNotNone(row["validated_at"])

    def test_recapture_supersedes_with_version_bump(self):
        memory_store.record_skill_candidate(
            "order tea", "browser", "order chamomile tea")
        memory_store.record_skill_candidate(
            "order tea", "browser", "order green tea")
        rows = memory_store.find_skills("order tea", status="candidate")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["version"], 2)
        old = memory_store._conn().execute(
            "SELECT * FROM skills WHERE version = 1").fetchone()
        self.assertEqual(old["status"], "retired")

    def test_recall_grounds_promoted_skills(self):
        sid = memory_store.record_skill_candidate(
            "order groceries", "browser", "order groceries on amazon",
            steps=["click_locator: selector=#checkout"],
            postconditions=["cart shows 3 items"])
        memory_store.promote_skill(sid, approved_version=1,
                                   replay_evidence="verified run")
        block = memory_store.recall_skills_for("order groceries on amazon")
        self.assertIn("Approved skill", block)
        self.assertIn("order groceries", block)
        # F09: retrieval carries the ACTUAL procedure, not just the goal.
        self.assertIn("selector=#checkout", block)
        self.assertLessEqual(len(block), memory_store.RECALL_BUDGET_CHARS
                             + 40)


class CommitmentsTests(G9MemoryTestBase):
    def test_rejects_unsupported_trigger(self):
        with self.assertRaises(ValueError):
            memory_store.add_commitment(
                "check calendar", trigger_kind="calendar_change")

    def test_tick_delivers_once(self):
        now = time.time()
        cid = memory_store.add_commitment(
            "call the bank", trigger_kind="deadline",
            due_at=now - 1, expires_at=now + 3600)
        ids = memory_store.commitment_tick()
        self.assertEqual(ids, [cid])
        self.assertEqual(len(self.delivered), 1)
        self.assertEqual(self.delivered[0]["id"], cid)
        # A second tick must not repeat the delivery.
        self.assertEqual(memory_store.commitment_tick(), [])
        self.assertEqual(len(self.delivered), 1)

    def test_not_yet_due_stays_armed(self):
        now = time.time()
        memory_store.add_commitment(
            "water the plants", trigger_kind="deadline",
            due_at=now + 600, expires_at=now + 7200)
        self.assertEqual(memory_store.commitment_tick(), [])
        self.assertEqual(len(self.delivered), 0)
        armed = memory_store.list_commitments(status="armed")
        self.assertEqual(len(armed), 1)

    def test_expiry_lapses_without_delivery(self):
        now = time.time()
        memory_store.add_commitment(
            "expired thing", trigger_kind="deadline",
            due_at=now - 7200, expires_at=now - 1)
        self.assertEqual(memory_store.commitment_tick(), [])
        self.assertEqual(len(self.delivered), 0)
        self.assertEqual(len(memory_store.list_commitments(
            status="expired")), 1)

    def test_cancel_commitment(self):
        memory_store.add_commitment(
            "send invoice in 1 minute", trigger_kind="deadline",
            due_at=time.time() + 60)
        self.assertEqual(memory_store.cancel_commitment(), 1)
        self.assertEqual(memory_store.cancel_commitment(), 0)
        row = memory_store.list_commitments()[0]
        self.assertEqual(row["status"], "cancelled")


class DeadlineParseTests(G9MemoryTestBase):
    def test_in_duration(self):
        now = time.time()
        due, expires, cleaned = memory_store.parse_deadline(
            "call mom in 5 minutes", now=now)
        self.assertAlmostEqual(due - now, 300, delta=2)
        self.assertEqual(cleaned, "call mom")
        self.assertEqual(expires, due + 86400)

    def test_tomorrow(self):
        now = time.time()
        due, _expires, cleaned = memory_store.parse_deadline(
            "buy tickets tomorrow", now=now)
        self.assertAlmostEqual(due - now, 86400, delta=2)
        self.assertEqual(cleaned, "buy tickets")

    def test_no_deadline_returns_none(self):
        self.assertIsNone(memory_store.parse_deadline("call mom sometime"))


class MemoryPhraseTests(G9MemoryTestBase):
    def test_remember_phrase_stores_scoped_fact(self):
        reply = memory_store.handle_memory_phrase(
            "remember that my project is phoenix")
        self.assertIn("Noted", reply)
        rows = memory_store.relevant_facts("project")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value"], "phoenix")
        self.assertEqual(rows[0]["provenance"], "user")

    def test_remember_freeform_clause(self):
        reply = memory_store.handle_memory_phrase(
            "note that the garage door code changes on monday")
        self.assertIn("Noted", reply)
        self.assertNotEqual(memory_store.relevant_facts("garage"), [])

    def test_recall_phrase(self):
        memory_store.remember("project", "phoenix")
        reply = memory_store.handle_memory_phrase(
            "what do you remember about project")
        self.assertIn("phoenix", reply)
        reply2 = memory_store.handle_memory_phrase(
            "what do you remember about unicorns")
        self.assertIn("don't have anything", reply2)

    def test_forget_phrase(self):
        memory_store.remember("project", "phoenix")
        self.assertIn("Forgotten", memory_store.handle_memory_phrase(
            "forget my project"))
        self.assertEqual(memory_store.relevant_facts("project"), [])
        self.assertIn("don't have that stored",
                      memory_store.handle_memory_phrase("forget my project"))

    def test_forget_never_touches_screen_ops(self):
        # The screen-control confirmation grammar keeps priority: a
        # screen-flavoured "forget ..." is NOT a memory op.
        self.assertIsNone(memory_store.handle_memory_phrase(
            "forget about the browser action"))

    def test_negation_idioms_are_not_memory_ops(self):
        self.assertIsNone(memory_store.handle_memory_phrase("forget it"))
        self.assertIsNone(memory_store.handle_memory_phrase("never mind"))

    def test_remind_me_arms_commitment(self):
        reply = memory_store.handle_memory_phrase(
            "remind me to stretch in 2 minutes")
        self.assertIn("Understood", reply)
        armed = memory_store.list_commitments(status="armed")
        self.assertEqual(len(armed), 1)
        self.assertGreater(armed[0]["due_at"], time.time() + 60)

    def test_remind_me_without_when_asks(self):
        self.assertIn("When should I remind you",
                      memory_store.handle_memory_phrase("remind me to nap"))

    def test_reminders_list_and_cancel(self):
        memory_store.handle_memory_phrase("remind me to stretch in 2 minutes")
        listing = memory_store.handle_memory_phrase("what reminders are set")
        self.assertIn("stretch", listing)
        self.assertIn("cancelled", memory_store.handle_memory_phrase(
            "cancel the reminder"))
        self.assertIn("No armed reminder", memory_store.handle_memory_phrase(
            "cancel the reminder"))


class SkillPhraseTests(G9MemoryTestBase):
    def test_approve_promote_list_retire(self):
        sid = memory_store.record_skill_candidate(
            "order groceries", "browser", "order groceries on amazon")
        # F09: the approval phrase is version-bound and replay-validated — an
        # unvalidated candidate is refused until a replay verifies it.
        self.assertIn("has not been replayed",
                      memory_store.handle_memory_phrase(
                          "approve the order groceries skill"))
        memory_store.note_skill_replay(sid, True,
                                       reason="replayed and verified")
        self.assertIn("Approved", memory_store.handle_memory_phrase(
            "approve the order groceries skill"))
        listing = memory_store.handle_memory_phrase("what skills do you know")
        self.assertIn("order-groceries", listing)
        self.assertIn("Retired", memory_store.handle_memory_phrase(
            "retire the order groceries skill"))
        self.assertEqual(memory_store.recall_skills_for("order groceries"),
                         "")

    def test_approve_unknown_skill(self):
        self.assertIn("don't have a pending skill",
                      memory_store.handle_memory_phrase(
                          "approve the flying-car skill"))


class DisabledModeTests(G9MemoryTestBase):
    def test_disabled_store_is_a_noop(self):
        memory_store.MEMORY_ENABLED = False
        self.assertIsNone(memory_store.remember("a", "b"))
        self.assertEqual(memory_store.relevant_facts("a"), [])
        self.assertEqual(memory_store.memory_context("a"), "")
        self.assertEqual(memory_store.recent_events(), [])
        self.assertIsNone(memory_store.handle_memory_phrase(
            "remember that x is y"))
        self.assertEqual(memory_store.commitment_tick(), [])


class BrainIntegrationTests(G9MemoryTestBase):
    """Wired into the real brain routing (guarded import is live)."""

    @classmethod
    def setUpClass(cls):
        from backend.core import brain
        cls.brain = brain

    def test_build_chat_messages_injects_memory(self):
        memory_store.remember("project", "phoenix")
        with patch.object(self.brain, "should_search", return_value=False):
            built = self.brain._build_chat_messages(
                "what is my project name", history=[])
        system = built["messages"][0]["content"]
        self.assertIn("Known context from memory", system)
        self.assertIn("phoenix", system)

    def test_build_chat_messages_empty_store_unchanged(self):
        # Zero prompt change until a fact exists (regression guard).
        with patch.object(self.brain, "should_search", return_value=False):
            built = self.brain._build_chat_messages(
                "what is the weather", history=[])
        self.assertNotIn("Known context from memory",
                         built["messages"][0]["content"])

    def test_process_message_remembers(self):
        reply = self.brain.process_message(
            "remember that my project is phoenix", from_voice=False)
        self.assertIn("Noted", reply)
        rows = memory_store.relevant_facts("project")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value"], "phoenix")

    def test_process_message_recalls(self):
        memory_store.remember("project", "phoenix")
        reply = self.brain.process_message(
            "what do you remember about project", from_voice=False)
        self.assertIn("phoenix", reply)

    def test_process_message_forget(self):
        memory_store.remember("project", "phoenix")
        reply = self.brain.process_message(
            "forget my project", from_voice=False)
        self.assertIn("Forgotten", reply)
        self.assertEqual(memory_store.relevant_facts("project"), [])

    def test_notify_async_reply_records_event(self):
        self.brain._notify_async_reply("Sir, the report is ready.")
        rows = memory_store.recent_events(kind="async_result")
        self.assertEqual(len(rows), 1)
        self.assertIn("report is ready", rows[0]["summary"])

    def test_commitment_delivery_rides_async_path(self):
        # The brain-registered delivery hook speaks through the async
        # reply callback — captured here to prove notification-only delivery.
        from backend.core import brain as _b
        memory_store.set_commitment_delivery(_b._deliver_commitment)
        captured = []
        old = _b._async_reply_callback
        _b._async_reply_callback = lambda text, spoken=None: captured.append(text)
        try:
            memory_store.add_commitment(
                "water the plants", trigger_kind="deadline",
                due_at=time.time() - 1, expires_at=time.time() + 3600)
            memory_store.commitment_tick()
        finally:
            _b._async_reply_callback = old
            memory_store.set_commitment_delivery(self.delivered.append)
        self.assertTrue(any("water the plants" in c for c in captured))





