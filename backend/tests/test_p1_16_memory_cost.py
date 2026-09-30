"""[P1-16] Memory retrieval cost, retention, and conversation safety.

The audit found that personal memory had become expensive and unreliable:

A. RETRIEVAL — ``relevant_facts`` ran up to ~100 queries per turn (one FTS
   query per term per column, then a per-token LIKE scan) over stopwords, and
   injected loosely-related facts into the prompt. It is now ONE FTS query over
   the CONTENT words, with a relevance gate ("an empty context is better than a
   wrong one") and a write-generation cache.
B. MAINTENANCE — the events table was never pruned and the whole FTS index was
   rebuilt on EVERY process start, so startup cost grew without bound. The
   rebuild is now gated on ``PRAGMA user_version`` and old events are pruned
   with their FTS rows; the scheduler sleeps until the next due item.
C. PHRASES — "X is also known as Y" and "fix X to Y" are ordinary sentences and
   used to WRITE memory; "Forget it" guessed a target (by matching a fact's
   VALUE) and could delete the wrong memory. Writes now require explicit memory
   wording, and forgetting resolves to exactly one fact or asks.
"""

import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import memory_store  # noqa: E402


def _active_rows():
    return [dict(r) for r in memory_store._conn().execute(
        "SELECT * FROM facts WHERE forgotten = 0 AND superseded_by IS NULL "
        "ORDER BY id").fetchall()]


def _alias_rows():
    return [dict(r) for r in memory_store._conn().execute(
        "SELECT * FROM fact_aliases").fetchall()]


def _counting_connection(recorded):
    """A connection subclass that records every statement it executes."""

    class _Counting(memory_store._MemoryConnection):
        def execute(self, sql, *args, **kwargs):
            recorded.append(sql)
            return super().execute(sql, *args, **kwargs)

    return _Counting


class P116MemoryCostTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._path = os.path.join(self._tmp.name, "mem.db")
        memory_store.MEMORY_ENABLED = True
        memory_store.configure(self._path)
        memory_store.stop_scheduler()
        memory_store.set_commitment_delivery(None)

    def tearDown(self):
        memory_store.set_commitment_delivery(None)
        memory_store.stop_scheduler()
        memory_store.close()
        self._tmp.cleanup()


# ── A. retrieval ─────────────────────────────────────────────────────────
class RetrievalCostTests(P116MemoryCostTestBase):
    def test_one_lookup_is_one_fts_query_over_content_words_only(self):
        for i in range(5):
            memory_store.remember("project%d" % i, "phoenix%d" % i)
        memory_store.flush_writes()
        calls = []
        real = memory_store._fts_match_any

        def _spy(conn, table, terms, limit):
            calls.append(list(terms))
            return real(conn, table, terms, limit)

        with patch.object(memory_store, "_fts_match_any", side_effect=_spy):
            rows = memory_store.relevant_facts(
                "what do you know about the project2 that we said")
        # ONE query for the whole lookup — pre-fix it was one per term.
        self.assertEqual(len(calls), 1, "expected exactly one FTS query")
        # …and it searches the CONTENT words, never the stopwords.
        self.assertEqual(calls[0], ["project2"])
        self.assertEqual([r["value"] for r in rows], ["phoenix2"])

    def test_the_query_count_of_a_lookup_is_bounded(self):
        query = "what about the one that we said about the project"

        def _count_selects():
            conn = memory_store._conn()
            seen = []
            conn.set_trace_callback(seen.append)
            try:
                memory_store.relevant_facts(query)
            finally:
                conn.set_trace_callback(None)
            return [s for s in seen if s.lstrip().upper().startswith("SELECT")]

        for i in range(5):
            memory_store.remember(
                "note%d" % i, "the one that we said about the project %d" % i)
        memory_store.flush_writes()
        small = _count_selects()
        for i in range(35):
            memory_store.remember(
                "note%d" % (i + 5),
                "the one that we said about the project %d" % (i + 5))
        memory_store.flush_writes()
        large = _count_selects()
        self.assertLessEqual(len([s for s in small if " MATCH " in s]), 1)
        self.assertLessEqual(len(small), 5,
                             "a lookup must not fan out into one query per term")
        # …and the cost must not GROW with the number of stored facts.
        self.assertEqual(len(small), len(large),
                         "query count scaled with the number of facts")

    def test_stopwords_are_never_search_terms(self):
        self.assertEqual(
            memory_store._retrieval_terms("what about the one that we said"),
            [])
        self.assertEqual(
            memory_store._retrieval_terms("What is my deployment target?"),
            ["deployment", "target"])

    def test_a_stopword_only_question_injects_nothing(self):
        memory_store.remember("user note", "the one that we said about the one",
                              predicate="about")
        memory_store.flush_writes()
        self.assertEqual(
            memory_store.relevant_facts("what about the one that we said"), [])
        self.assertEqual(
            memory_store.memory_context("what about the one that we said"), "")

    def test_letters_inside_an_unrelated_word_are_not_relevance(self):
        memory_store.remember("user note", "the martian start report",
                              predicate="about")
        memory_store.flush_writes()
        # "art" only occurs INSIDE "martian"/"start": pre-fix the LIKE scan
        # matched it and injected the note anyway.
        self.assertEqual(memory_store.relevant_facts("art"), [])
        # The genuine match survives.
        self.assertEqual(
            [r["subject"] for r in memory_store.relevant_facts("martian")],
            ["user note"])

    def test_a_genuine_match_is_still_returned(self):
        memory_store.remember("deployment target", "phoenix cluster")
        memory_store.flush_writes()
        rows = memory_store.relevant_facts("What is my deployment target?")
        self.assertEqual([r["subject"] for r in rows], ["deployment target"])

    def test_a_write_invalidates_the_cached_retrieval(self):
        memory_store.remember("project", "phoenix")
        memory_store.flush_writes()
        self.assertEqual([r["value"] for r in memory_store.relevant_facts("project")],
                         ["phoenix"])
        # A queued write…
        memory_store.remember("project", "atlas")
        memory_store.flush_writes()
        self.assertEqual([r["value"] for r in memory_store.relevant_facts("project")],
                         ["atlas"])
        # …and a DIRECT write that bypasses the queue (proposal approval).
        proposal = memory_store.propose_fact("project", "nebula")
        memory_store.approve_fact_proposal(proposal)
        self.assertEqual([r["value"] for r in memory_store.relevant_facts("project")],
                         ["nebula"])

    def test_memory_context_keeps_the_delimited_contract(self):
        memory_store.remember("project", "phoenix")
        memory_store.flush_writes()
        context = memory_store.memory_context("what is my project")
        self.assertTrue(context.startswith("Known context from memory:"))
        self.assertIn("- project is: phoenix", context)
        # Nothing genuinely relevant -> no block at all (zero prompt change).
        self.assertEqual(memory_store.memory_context("what about the weather"),
                         "")

    def test_retrieval_failures_are_logged_and_swallowed(self):
        with patch.object(memory_store, "_conn",
                          side_effect=RuntimeError("database is gone")):
            with self.assertLogs(level="WARNING") as logs:
                self.assertEqual(memory_store.relevant_facts("project"), [])
        self.assertTrue(any("relevant_facts" in line for line in logs.output))


# ── B. maintenance ───────────────────────────────────────────────────────
class MaintenanceTests(P116MemoryCostTestBase):
    def _user_version(self, conn):
        return int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)

    def test_the_fts_rebuild_is_gated_on_the_schema_version(self):
        conn = memory_store._conn()
        self.assertEqual(self._user_version(conn),
                         memory_store.MEMORY_SCHEMA_VERSION)
        recorded = []
        with patch.object(memory_store, "_MemoryConnection",
                          _counting_connection(recorded)):
            memory_store.configure(self._path)   # a fresh process opening it
            memory_store._conn()
        rebuilds = [s for s in recorded if "rebuild" in s]
        self.assertEqual(rebuilds, [],
                         "an up-to-date database must not rebuild its index")

    def test_an_old_database_is_migrated_and_rebuilt_once(self):
        conn = memory_store._conn()
        conn.execute("PRAGMA user_version = 0")
        conn.commit()
        recorded = []
        with patch.object(memory_store, "_MemoryConnection",
                          _counting_connection(recorded)):
            memory_store.configure(self._path)
            reopened = memory_store._conn()
        self.assertEqual(len([s for s in recorded if "rebuild" in s]), 3)
        self.assertEqual(self._user_version(reopened),
                         memory_store.MEMORY_SCHEMA_VERSION)
        # …and the NEXT open rebuilds nothing at all.
        recorded2 = []
        with patch.object(memory_store, "_MemoryConnection",
                          _counting_connection(recorded2)):
            memory_store.configure(self._path)
            memory_store._conn()
        self.assertEqual([s for s in recorded2 if "rebuild" in s], [])

    def test_retention_prunes_old_events_and_their_fts_rows(self):
        conn = memory_store._conn()
        for i in range(3):
            memory_store.record_event("task_result", "ancient event %d" % i)
        memory_store.flush_writes()
        conn.execute("UPDATE events SET ts = ?", (time.time() - 90 * 86400,))
        conn.commit()
        memory_store.record_event("task_result", "fresh event")
        memory_store.flush_writes()
        self.assertEqual(memory_store.prune_events(), 3)
        self.assertEqual(
            conn.execute("SELECT COUNT(*) FROM events").fetchone()[0], 1)
        # The index shrank with the table: a pruned row is not searchable.
        self.assertEqual(memory_store.find_events("ancient"), [])
        self.assertEqual(len(memory_store.find_events("fresh")), 1)

    def test_retention_also_caps_the_row_count(self):
        conn = memory_store._conn()
        for i in range(6):
            memory_store.record_event("task_result", "capped event %d" % i)
        memory_store.flush_writes()
        with patch.object(memory_store, "MEMORY_EVENT_MAX_ROWS", 2):
            self.assertEqual(memory_store.prune_events(keep_days=10 ** 6), 4)
        self.assertEqual(
            conn.execute("SELECT COUNT(*) FROM events").fetchone()[0], 2)

    def test_one_commit_per_event_row(self):
        commits = []

        class _Counting(memory_store._MemoryConnection):
            def commit(self):
                commits.append(1)
                return super().commit()

        with patch.object(memory_store, "_MemoryConnection", _Counting):
            memory_store.configure(self._path)
            memory_store._conn()
            # Warm up so the writer thread and its connection already exist.
            memory_store.record_event("task_result", "warm up event")
            memory_store.flush_writes()
            before = len(commits)
            memory_store.record_event("task_result", "single commit event")
            memory_store.flush_writes()
        self.assertEqual(len(commits) - before, 1,
                         "the row and its FTS mirror share ONE commit")

    def test_the_store_writes_with_synchronous_normal(self):
        value = memory_store._conn().execute("PRAGMA synchronous").fetchone()[0]
        self.assertEqual(int(value), 1)          # 1 == NORMAL

    def test_the_scheduler_sleeps_until_the_next_due_item(self):
        # Nothing armed: sleep the (bounded) maximum instead of polling.
        self.assertEqual(memory_store._next_commitment_delay(),
                         memory_store.SCHEDULER_MAX_WAIT)
        self.assertGreater(memory_store.SCHEDULER_MAX_WAIT, 5.0)
        # Far in the future: still bounded.
        memory_store.add_commitment("stretch", due_at=time.time() + 600)
        self.assertEqual(memory_store._next_commitment_delay(),
                         memory_store.SCHEDULER_MAX_WAIT)
        # Soon: sleep exactly until it is due.
        memory_store.add_commitment("nap", due_at=time.time() + 10)
        delay = memory_store._next_commitment_delay()
        self.assertGreater(delay, 5.0)
        self.assertLessEqual(delay, 11.0)
        # Already due: wake immediately.
        memory_store.add_commitment("now", due_at=time.time() - 5)
        self.assertLessEqual(memory_store._next_commitment_delay(), 0.06)

    def test_arming_a_commitment_wakes_a_running_scheduler(self):
        # A scheduler that is not running needs no wake (it reads the due times
        # when it starts), so start it first.
        memory_store._start_scheduler()
        memory_store._scheduler_wake.clear()
        memory_store.add_commitment("wake me", due_at=time.time() + 30)
        self.assertTrue(memory_store._scheduler_wake.is_set())

    def test_the_scheduler_thread_stops_when_asked(self):
        memory_store._start_scheduler()
        thread = memory_store._scheduler_thread
        self.assertIsNotNone(thread)
        self.assertTrue(thread.is_alive())
        memory_store.stop_scheduler()
        self.assertFalse(thread.is_alive(),
                         "a lingering tick thread keeps the DB file locked")


# ── C. conversation safety ───────────────────────────────────────────────
class PhraseSafetyTests(P116MemoryCostTestBase):
    def test_ordinary_sentences_never_write_memory(self):
        for sentence in (
            "the sky is also known as the heavens",
            "python is also known as a scripting language",
            "fix the header to blue",
            "actually the sky is green",
        ):
            self.assertIsNone(memory_store.handle_memory_phrase(sentence),
                              "ordinary sentence was treated as a memory op")
        self.assertEqual(_alias_rows(), [])
        self.assertEqual(_active_rows(), [])

    def test_explicit_alias_wording_still_works(self):
        memory_store.remember("project", "phoenix")
        memory_store.flush_writes()
        reply = memory_store.handle_memory_phrase(
            "remember that project is also known as phoenix")
        self.assertIn("another name", reply)
        self.assertEqual(memory_store.resolve_alias("phoenix"), "project:is")

    def test_an_alias_for_an_unknown_target_is_never_written(self):
        self.assertIsNone(memory_store.handle_memory_phrase(
            "remember that unicorn is also known as horse"))
        self.assertEqual(_alias_rows(), [])
        self.assertIsNone(memory_store.resolve_alias("horse"))

    def test_explicit_correction_wording_still_works(self):
        memory_store.remember("project", "phoenix")
        memory_store.flush_writes()
        reply = memory_store.handle_memory_phrase("correct my project to atlas")
        self.assertIn("Corrected", reply)
        self.assertEqual(
            [r["value"] for r in memory_store.relevant_facts("project")],
            ["atlas"])

    def test_forget_it_asks_instead_of_deleting(self):
        memory_store.remember("project", "phoenix")
        memory_store.flush_writes()
        for phrase in ("forget it", "Forget it.", "forget that"):
            reply = memory_store.handle_memory_phrase(phrase)
            self.assertIn("Which memory", reply)
        # Nothing was deleted, and "never mind" stays a pure idiom.
        self.assertEqual(len(_active_rows()), 1)
        self.assertIsNone(memory_store.handle_memory_phrase("never mind"))

    def test_forget_never_matches_a_fact_value(self):
        memory_store.remember("project", "phoenix")
        memory_store.flush_writes()
        reply = memory_store.handle_memory_phrase("forget phoenix")
        self.assertIn("don't have that stored", reply)
        self.assertEqual(
            [r["value"] for r in memory_store.relevant_facts("project")],
            ["phoenix"])

    def test_forget_resolves_to_exactly_one_fact_or_asks(self):
        memory_store.remember("project alpha", "phoenix")
        memory_store.remember("project beta", "atlas")
        memory_store.remember("car", "tesla")
        memory_store.flush_writes()
        # Two candidates -> ask, delete nothing.
        reply = memory_store.handle_memory_phrase("forget project")
        self.assertIn("more than one", reply)
        self.assertEqual(len(memory_store.relevant_facts("project")), 2)
        # Exactly one -> forget exactly that one (punctuation normalised too).
        reply = memory_store.handle_memory_phrase("forget my car.")
        self.assertIn("Forgotten", reply)
        self.assertEqual(memory_store.relevant_facts("car"), [])
        self.assertEqual(len(memory_store.relevant_facts("project")), 2)
        # Nothing left -> say so, never guess.
        reply = memory_store.handle_memory_phrase("forget my car")
        self.assertIn("don't have that stored", reply)

    def test_a_broken_store_never_raises_into_the_conversation(self):
        with patch.object(memory_store, "_conn",
                          side_effect=RuntimeError("database is gone")):
            self.assertEqual(memory_store.relevant_facts("project"), [])
            reply = memory_store.handle_memory_phrase("forget my project")
        self.assertIn("don't have that stored", reply)


if __name__ == "__main__":
    unittest.main()
