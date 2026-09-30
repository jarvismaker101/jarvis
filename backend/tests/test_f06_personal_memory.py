"""F06 — Build retrievable personal memory (audit correction/acceptance).

Covers the corrected behaviour:
  * independent facts (incl. free-form notes) get their own stable key, so
    unrelated notes coexist instead of superseding each other;
  * a correction affects ONE intended fact (resolved by key/alias/subject)
    and leaves every other fact alone;
  * the latest revision of a fact stays retrievable, even when the query
    matches an older revision's text;
  * aliases resolve for retrieval and correction;
  * retrieval returns metadata (key, revision, source, active flag);
  * a literal '%' (or '_') forgetting criterion is literal and cannot wipe
    the store;
  * a model proposal is stored for review and is NOT applied until an
    explicit approval call, which then routes through the same
    independent-fact/correction path.

The store runs against a per-test tmp SQLite file (memory_store.configure) —
the user's real DB is never touched.
"""

import os
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from backend.core import memory_store  # noqa: E402


class F06MemoryTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        memory_store.configure(os.path.join(self._tmp.name, "mem.db"))
        memory_store.MEMORY_ENABLED = True
        memory_store.stop_scheduler()

    def tearDown(self):
        memory_store.stop_scheduler()
        memory_store.close()
        self._tmp.cleanup()

    # -- helpers ---------------------------------------------------------
    def _active(self):
        return [dict(r) for r in memory_store._conn().execute(
            "SELECT * FROM facts WHERE forgotten = 0 AND superseded_by IS NULL "
            "ORDER BY id").fetchall()]

    def _values(self, query):
        return sorted(r["value"] for r in memory_store.relevant_facts(query))


class IndependentFactsTests(F06MemoryTestBase):
    def test_unrelated_notes_coexist_and_are_retrievable(self):
        r1 = memory_store.handle_memory_phrase(
            "note that the garage door code changes on monday")
        r2 = memory_store.handle_memory_phrase(
            "note that the piano needs tuning")
        self.assertIn("Noted", r1)
        self.assertIn("Noted", r2)
        # The second note must NOT tombstone the first.
        self.assertEqual(len(self._active()), 2)
        self.assertNotEqual(memory_store.relevant_facts("garage"), [])
        self.assertNotEqual(memory_store.relevant_facts("piano"), [])
        # Distinct, stable per-fact keys (content-addressed for notes).
        keys = sorted(r["key"] for r in self._active())
        self.assertEqual(len(set(keys)), 2)
        self.assertTrue(all(k.startswith("note:") for k in keys), keys)

    def test_repeating_a_note_does_not_supersede_other_notes(self):
        memory_store.remember("user note", "buy milk", predicate="about")
        memory_store.remember("user note", "buy milk", predicate="about")
        memory_store.remember("user note", "walk the dog", predicate="about")
        active = self._active()
        values = sorted(r["value"] for r in active)
        self.assertIn("walk the dog", values)
        for r in active:
            self.assertTrue(r["superseded_by"] is None)

    def test_same_fact_subject_still_revises(self):
        # The existing contract is preserved: the SAME independent fact
        # (subject+attribute) is revised, not duplicated.
        memory_store.remember("project", "phoenix")
        new_id = memory_store.remember("project", "atlas")
        rows = memory_store.relevant_facts("project")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value"], "atlas")
        old = memory_store._conn().execute(
            "SELECT * FROM facts WHERE value = 'phoenix'").fetchone()
        self.assertEqual(old["superseded_by"], new_id)

    def test_independent_attributes_of_one_subject_coexist(self):
        memory_store.remember("project", "phoenix", predicate="is")
        memory_store.remember("project", "atlas", predicate="codename")
        self.assertEqual(len(self._active()), 2)


class CorrectionTests(F06MemoryTestBase):
    def test_correction_changes_one_fact_and_leaves_the_other_alone(self):
        memory_store.remember("project", "phoenix")
        car_id = memory_store.remember("car", "tesla")
        new_id = memory_store.correct_fact("project", value="atlas")
        self.assertTrue(new_id)
        self.assertEqual(self._values("project"), ["atlas"])
        car = memory_store.resolve_fact("car")
        self.assertEqual(car["id"], car_id)
        self.assertEqual(car["value"], "tesla")
        self.assertTrue(car["active"])
        self.assertEqual(len(self._active()), 2)
        # The correction keeps its own provenance and supersedes only the
        # revision of the fact it was about.
        corrected = memory_store.resolve_fact("project")
        self.assertEqual(corrected["provenance"], "user_correction")
        old = memory_store._conn().execute(
            "SELECT * FROM facts WHERE value = 'phoenix'").fetchone()
        self.assertEqual(old["superseded_by"], corrected["id"])

    def test_correction_does_not_touch_another_note(self):
        memory_store.remember("user note", "garage code is 1234",
                              predicate="about")
        memory_store.remember("user note", "piano needs tuning",
                              predicate="about")
        # Correct the garage note by resolving its fact key.
        target = memory_store.relevant_facts("garage")[0]
        self.assertTrue(memory_store.correct_fact(
            target["key"], value="garage code is 9999"))
        values = sorted(r["value"] for r in self._active())
        self.assertIn("garage code is 9999", values)
        self.assertIn("piano needs tuning", values)
        # The explicit key resolves for history too (2 revisions of one note).
        history = memory_store.fact_revisions(target["key"])
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["key"], target["key"])

    def test_correction_with_nothing_stored_is_a_noop(self):
        self.assertIsNone(memory_store.correct_fact("unicorn", value="x"))
        self.assertEqual(len(self._active()), 0)

    def test_legacy_positional_shape_still_works(self):
        memory_store.remember("project", "phoenix")
        new_id = memory_store.correct_fact("project", "is", "atlas")
        self.assertTrue(new_id)
        self.assertEqual(self._values("project"), ["atlas"])

    def test_phrase_correction_affects_one_fact(self):
        memory_store.remember("project", "phoenix")
        memory_store.remember("car", "tesla")
        reply = memory_store.handle_memory_phrase("correct my project to atlas")
        self.assertIn("Corrected", reply)
        self.assertEqual(self._values("project"), ["atlas"])
        self.assertEqual(self._values("car"), ["tesla"])

    def test_phrase_correction_with_nothing_stored_does_not_hijack(self):
        # "fix the header to blue" is a task, not a memory op: the store
        # must return None so normal routing continues.
        self.assertIsNone(
            memory_store.handle_memory_phrase("fix the header to blue"))
        self.assertEqual(self._active(), [])


class RevisionTests(F06MemoryTestBase):
    def test_latest_revision_stays_retrievable(self):
        memory_store.remember("project", "phoenix")
        memory_store.correct_fact("project", value="atlas")
        rows = memory_store.relevant_facts("project")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value"], "atlas")
        self.assertEqual(rows[0]["revision"], 2)
        self.assertTrue(rows[0]["active"])
        # History preserved, nothing dropped.
        history = memory_store.fact_revisions("project")
        self.assertEqual([h["value"] for h in history], ["atlas", "phoenix"])
        self.assertEqual([h["active"] for h in history], [True, False])

    def test_query_matching_an_old_revision_returns_the_active_one(self):
        memory_store.remember("project", "phoenix")
        memory_store.correct_fact("project", value="atlas")
        # "phoenix" is only in the superseded revision; FTS matching only
        # inactive rows must still surface the latest ACTIVE revision.
        rows = memory_store.relevant_facts("phoenix")
        self.assertTrue(rows, "the latest revision must stay retrievable")
        self.assertEqual(rows[0]["value"], "atlas")

    def test_retrieval_falls_back_when_fts_is_unavailable(self):
        memory_store.remember("deployment", "shipped on friday")
        with patch.object(memory_store, "_fts_ok", False):
            rows = memory_store.relevant_facts("shipped on friday")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value"], "shipped on friday")


class AliasTests(F06MemoryTestBase):
    def test_alias_resolves_for_retrieval(self):
        memory_store.remember("project", "phoenix")
        self.assertEqual(memory_store.add_alias("the big one", "my project"),
                         "big-one")
        self.assertEqual(memory_store.resolve_alias("the big one"),
                         "project:is")
        rows = memory_store.relevant_facts("what about the big one")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value"], "phoenix")

    def test_alias_resolves_for_correction(self):
        memory_store.remember("project", "phoenix")
        memory_store.remember("car", "tesla")
        memory_store.add_alias("phoenix", "project")
        # Storing/correcting through the alias lands on the SAME fact key.
        memory_store.correct_fact("phoenix", value="atlas")
        self.assertEqual(self._values("project"), ["atlas"])
        self.assertEqual(self._values("car"), ["tesla"])

    def test_alias_phrase(self):
        memory_store.remember("project", "phoenix")
        reply = memory_store.handle_memory_phrase(
            "add alias the big one for my project")
        self.assertIn("big one", reply)
        rows = memory_store.relevant_facts("the big one")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value"], "phoenix")

    def test_alias_known_as_phrase(self):
        memory_store.remember("project", "phoenix")
        # [P1-16] An alias is only written on EXPLICIT memory wording. The bare
        # sentence "X is also known as Y" is ordinary speech, and firing on it
        # let casual conversation bind a false alias (memory corruption).
        memory_store.handle_memory_phrase("project is also known as phoenix")
        self.assertIsNone(memory_store.resolve_alias("phoenix"))

        memory_store.handle_memory_phrase(
            "remember that project is also known as phoenix")
        self.assertEqual(memory_store.resolve_alias("phoenix"), "project:is")


class ForgetEscapingTests(F06MemoryTestBase):
    def test_literal_percent_forgets_nothing(self):
        memory_store.remember("project", "phoenix")
        memory_store.remember("car", "tesla")
        self.assertEqual(memory_store.forget(subject="%"), 0)
        self.assertEqual(len(self._active()), 2)
        self.assertEqual(self._values("project"), ["phoenix"])

    def test_literal_percent_only_matches_a_literal_percent(self):
        memory_store.remember("project", "phoenix")
        memory_store.remember("progress", "100% complete")
        # Escaped: the criterion matches only the row literally containing %.
        self.assertEqual(memory_store.forget(subject="%"), 1)
        self.assertEqual(self._values("project"), ["phoenix"])

    def test_literal_underscore_is_not_a_wildcard(self):
        memory_store.remember("axb", "one")
        memory_store.remember("a_b", "two")
        # Escaped: only the row literally containing '_' matches.
        self.assertEqual(memory_store.forget(subject="_"), 1)
        self.assertEqual(self._values("axb"), ["one"])

    def test_blanket_forget_still_forgets_nothing(self):
        memory_store.remember("project", "phoenix")
        self.assertEqual(memory_store.forget(), 0)
        self.assertEqual(len(self._active()), 1)

    def test_key_criterion_is_literal(self):
        memory_store.remember("project", "phoenix")
        self.assertEqual(memory_store.forget(key="%"), 0)
        self.assertEqual(len(self._active()), 1)


class MetadataTests(F06MemoryTestBase):
    def test_retrieval_returns_metadata(self):
        fid = memory_store.remember("project", "phoenix", source="chat")
        rows = memory_store.relevant_facts("project")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        for field in ("key", "subject", "predicate", "value", "source",
                      "created_at", "revision", "active", "provenance",
                      "forgotten"):
            self.assertIn(field, row)
        self.assertEqual(row["id"], fid)
        self.assertEqual(row["key"], "project:is")
        self.assertEqual(row["source"], "chat")
        self.assertEqual(row["revision"], 1)
        self.assertTrue(row["active"])
        self.assertFalse(row["forgotten"])
        self.assertGreater(row["created_at"], 0)

    def test_superseded_rows_are_not_active_but_resolvable(self):
        memory_store.remember("project", "phoenix")
        memory_store.correct_fact("project", value="atlas")
        self.assertEqual([r["value"] for r in memory_store.relevant_facts(
            "project")], ["atlas"])
        resolved = memory_store.resolve_fact("my project")
        self.assertEqual(resolved["value"], "atlas")
        self.assertEqual(resolved["revision"], 2)
        self.assertTrue(resolved["active"])


class ProposalTests(F06MemoryTestBase):
    def test_proposal_is_not_applied_until_approved(self):
        memory_store.remember("project", "phoenix")
        pid = memory_store.propose_fact(
            "project", "atlas", rationale="the user mentioned atlas")
        self.assertTrue(pid)
        pending = memory_store.list_fact_proposals(status="pending")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["review_state"], "pending")
        # NOT applied: the stored fact is untouched.
        self.assertEqual(self._values("project"), ["phoenix"])
        self.assertEqual(len(self._active()), 1)

        applied = memory_store.approve_fact_proposal(pid)
        self.assertIsNotNone(applied)
        self.assertEqual(applied["applied_fact_id"],
                         memory_store.resolve_fact("project")["id"])
        self.assertEqual(self._values("project"), ["atlas"])
        self.assertEqual(memory_store.resolve_fact("project")["provenance"],
                         "user_approved_proposal")
        self.assertEqual(memory_store.list_fact_proposals(status="pending"), [])
        self.assertEqual(
            len(memory_store.list_fact_proposals(status="applied")), 1)
        # Approving twice is a no-op (never re-applied).
        self.assertIsNone(memory_store.approve_fact_proposal(pid))

    def test_rejected_proposal_is_never_applied(self):
        memory_store.remember("project", "phoenix")
        pid = memory_store.propose_fact("project", "atlas")
        self.assertTrue(memory_store.reject_fact_proposal(pid, reason="wrong"))
        self.assertEqual(self._values("project"), ["phoenix"])
        self.assertIsNone(memory_store.approve_fact_proposal(pid))
        self.assertEqual(self._values("project"), ["phoenix"])

    def test_approved_proposal_only_touches_its_own_fact(self):
        memory_store.remember("project", "phoenix")
        memory_store.remember("car", "tesla")
        pid = memory_store.propose_fact("project", "atlas")
        memory_store.approve_fact_proposal(pid)
        self.assertEqual(self._values("project"), ["atlas"])
        self.assertEqual(self._values("car"), ["tesla"])

    def test_proposal_phrases_require_explicit_approval(self):
        memory_store.remember("project", "phoenix")
        memory_store.propose_fact("project", "atlas")
        listing = memory_store.handle_memory_phrase(
            "what memory proposals are pending")
        self.assertIn("atlas", listing)
        self.assertEqual(self._values("project"), ["phoenix"])
        reply = memory_store.handle_memory_phrase("approve the memory proposal")
        self.assertIn("Approved", reply)
        self.assertEqual(self._values("project"), ["atlas"])

    def test_episodic_derivation_only_proposes(self):
        memory_store.record_event(
            "task_result", "[browser_agent/completed] ordered groceries",
            detail={"task": "order groceries on amazon", "status": "completed",
                    "summary": "ordered groceries on amazon"})
        ids = memory_store.propose_facts_from_events()
        self.assertEqual(len(ids), 1)
        # Derived, but NOT stored: review first.
        self.assertEqual(self._active(), [])
        self.assertEqual(len(memory_store.list_fact_proposals()), 1)


_LEGACY_SCHEMA = """
CREATE TABLE facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL DEFAULT 'is',
    value TEXT NOT NULL,
    provenance TEXT NOT NULL DEFAULT 'user',
    source TEXT NOT NULL DEFAULT 'chat',
    confidence REAL NOT NULL DEFAULT 1.0,
    sensitivity TEXT NOT NULL DEFAULT 'normal',
    created_at REAL NOT NULL,
    superseded_by INTEGER,
    forgotten INTEGER NOT NULL DEFAULT 0,
    forgotten_at REAL
);
"""


class LegacyMigrationTests(F06MemoryTestBase):
    """An existing DB (facts without the F06 key column) keeps its rows and
    gains per-fact keys — never a wipe, never a rewrite of the user's file."""

    def _make_legacy_db(self, path):
        conn = sqlite3.connect(path)
        conn.executescript(_LEGACY_SCHEMA)
        conn.execute(
            "INSERT INTO facts (subject, predicate, value, created_at) "
            "VALUES (?, ?, ?, ?)", ("user note", "about",
                                    "the garage door code is 1234", 1.0))
        conn.execute(
            "INSERT INTO facts (subject, predicate, value, created_at) "
            "VALUES (?, ?, ?, ?)", ("project", "is", "phoenix", 2.0))
        conn.commit()
        conn.close()

    def test_legacy_rows_are_backfilled_and_retrievable(self):
        path = os.path.join(self._tmp.name, "legacy.db")
        self._make_legacy_db(path)
        memory_store.configure(path)
        rows = memory_store.relevant_facts("garage")
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["key"].startswith("note:"), rows[0]["key"])
        self.assertTrue(rows[0]["active"])
        project = memory_store.relevant_facts("project")
        self.assertEqual(project[0]["key"], "project:is")
        # Corrections and independent notes work on the migrated DB.
        memory_store.correct_fact("project", value="atlas")
        self.assertEqual(self._values("project"), ["atlas"])
        self.assertEqual(self._values("garage"), ["the garage door code is 1234"])

    def test_legacy_note_is_correctable_by_key(self):
        path = os.path.join(self._tmp.name, "legacy2.db")
        self._make_legacy_db(path)
        memory_store.configure(path)
        key = memory_store.relevant_facts("garage")[0]["key"]
        self.assertTrue(memory_store.correct_fact(key, value="code is 9999"))
        values = sorted(r["value"] for r in self._active())
        self.assertIn("code is 9999", values)
        self.assertIn("phoenix", values)


class BrainWiringTests(F06MemoryTestBase):
    """Corrections/aliases route through the real brain chat call path."""
    """Corrections/aliases route through the real brain chat call path."""

    @classmethod
    def setUpClass(cls):
        from backend.core import brain
        cls.brain = brain

    def test_brain_process_message_corrects_one_fact(self):
        memory_store.remember("project", "phoenix")
        memory_store.remember("car", "tesla")
        reply = self.brain.process_message(
            "correct my project to atlas", from_voice=False)
        self.assertIn("Corrected", reply)
        self.assertEqual(self._values("project"), ["atlas"])
        self.assertEqual(self._values("car"), ["tesla"])

    def test_brain_recall_uses_alias(self):
        memory_store.remember("project", "phoenix")
        memory_store.add_alias("the big one", "project")
        reply = self.brain.process_message(
            "what do you remember about the big one", from_voice=False)
        self.assertIn("phoenix", reply)


if __name__ == "__main__":
    unittest.main()
