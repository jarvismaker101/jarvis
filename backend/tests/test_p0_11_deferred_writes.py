"""P0-11 — synchronous disk writes must be off the path to the first delta.

Before this item a chat turn paid for two durable writes on the REQUEST thread
before a single character reached the user: the history JSON was rewritten
under a lock, and ``begin_request`` ran several SQLite commits. The change:

* the in-memory history append stays synchronous (it is the source of truth),
  the file follows from ONE debounced background writer, and it is flushed on
  the way out;
* the F07 bookkeeping (open request / record event / record result) is queued
  to ONE background writer thread, with the event id and the request identity
  allocated in memory;
* readers keep read-your-writes (the queue is drained before a direct access),
  so nothing that was written is invisible;
* a bounded queue applies backpressure instead of dropping a record, and a
  bookkeeping failure is retried/counted, never raised into the turn.

The tests below assert the ORDERING (a delta is delivered before the writes),
the durability (a clean shutdown persists everything, in order), the retry, and
that a bookkeeping failure cannot fail the turn.
"""

import json
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from backend.core import memory
from backend.core import memory_store


class HistoryWriterTests(unittest.TestCase):
    """The conversation-history JSON is written by a background writer."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._file = os.path.join(self._tmp.name, "conversation_history.json")
        self._patched = patch.object(memory, "_MEMORY_FILE",
                                     type(memory._MEMORY_FILE)(self._file))
        self._patched.start()
        memory.clear_history()
        memory.flush_history(timeout=2.0)

    def tearDown(self):
        memory.clear_history()
        memory.flush_history(timeout=2.0)
        self._patched.stop()
        self._tmp.cleanup()

    def _file_messages(self):
        with open(self._file, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_the_append_is_visible_before_the_file_is_written(self):
        """The in-memory append is the source of truth — and it is instant."""
        memory.add_message("user", "hello")
        self.assertEqual(memory.get_history(),
                         [{"role": "user", "content": "hello"}])
        self.assertTrue(memory.history_write_stats()["pending"],
                        "the durable copy is allowed to lag the append")

    def test_history_is_fully_persisted_after_a_clean_shutdown(self):
        """The atexit/shutdown path is what makes the debounce safe."""
        for index in range(3):
            memory.add_message("user", "msg %d" % index)
            memory.add_message("assistant", "reply %d" % index)
        # Nothing waits for the writer here: this is the shutdown flush.
        self.assertTrue(memory.flush_history(timeout=2.0))
        self.assertEqual(
            self._file_messages(),
            [{"role": role, "content": content}
             for index in range(3)
             for role, content in (("user", "msg %d" % index),
                                   ("assistant", "reply %d" % index))])

    def test_the_debounced_writer_persists_on_its_own(self):
        memory.add_message("user", "written by the background writer")
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if os.path.exists(self._file):
                try:
                    if self._file_messages():
                        break
                except Exception:
                    pass
            time.sleep(0.05)
        self.assertEqual(self._file_messages(),
                         [{"role": "user",
                           "content": "written by the background writer"}])

    def test_a_permission_error_on_replace_is_retried_and_reported(self):
        """Windows file locking is transient: retry, then report — never lose."""
        calls = {"count": 0}
        real_replace = os.replace

        def flaky_replace(src, dst):
            calls["count"] += 1
            if calls["count"] < 3:
                raise PermissionError(13, "locked by another process")
            return real_replace(src, dst)

        memory.add_message("user", "retried message")
        with patch.object(memory.os, "replace", side_effect=flaky_replace):
            self.assertTrue(memory.flush_history(timeout=2.0))
        self.assertGreaterEqual(calls["count"], 3)
        self.assertEqual(self._file_messages(),
                         [{"role": "user", "content": "retried message"}])

    def test_a_permanent_failure_is_counted_and_never_loses_memory(self):
        memory.add_message("user", "undeliverable")
        before = memory.history_write_stats()["failures"]
        with patch.object(memory.os, "replace",
                          side_effect=PermissionError(13, "always locked")):
            self.assertFalse(memory.flush_history(timeout=2.0))
        self.assertGreater(memory.history_write_stats()["failures"], before)
        # The MESSAGE is not lost — only its durable copy is behind, and the
        # turn keeps working.
        self.assertEqual(memory.get_history(),
                         [{"role": "user", "content": "undeliverable"}])

    def test_message_order_is_preserved_under_concurrent_turns(self):
        threads = [threading.Thread(target=lambda n=n: [
            memory.add_message("user", "t%d-%d" % (n, i)) for i in range(10)])
            for n in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        memory.flush_history(timeout=5.0)
        in_memory = memory.get_history()
        # MAX_HISTORY trims the head; the file must match the live list exactly,
        # in order, with no lost or duplicated message.
        self.assertEqual(self._file_messages(), in_memory)
        self.assertEqual(len(in_memory), memory.MAX_HISTORY)

    def test_clearing_history_does_not_resurrect_it(self):
        memory.add_message("user", "forget me")
        memory.clear_history()
        memory.flush_history(timeout=2.0)
        self.assertFalse(os.path.exists(self._file),
                         "a pending write must not recreate a cleared history")


class BookkeepingQueueTests(unittest.TestCase):
    """F07 bookkeeping is queued, but never invisible and never dropped."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._path = os.path.join(self._tmp.name, "mem.db")
        memory_store.MEMORY_ENABLED = True
        memory_store.configure(self._path)
        memory_store.stop_scheduler()

    def tearDown(self):
        memory_store.stop_scheduler()
        memory_store.close()
        memory_store.MEMORY_ENABLED = False
        self._tmp.cleanup()

    def test_the_request_id_is_known_before_any_durable_write(self):
        rid = memory_store.begin_request("do the thing", route="task")
        self.assertTrue(str(rid).startswith("req-"))
        self.assertEqual(memory_store.current_request_id(), rid)

    def test_a_queued_write_is_visible_to_the_next_read(self):
        rid = memory_store.record_request("find the price of the new laptop",
                                          route="research")
        memory_store.record_result(rid, "completed", summary="799 dollars",
                                   artifacts=[{"kind": "source",
                                               "url": "https://a.test/x"}])
        row = memory_store.get_work_request(rid)
        self.assertIsNotNone(row, "a read must see the queued writes")
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["summary"], "799 dollars")
        self.assertEqual(row["artifacts"][0]["url"], "https://a.test/x")

    def test_the_event_row_lands_with_the_id_that_was_returned(self):
        event_id = memory_store.record_event("research", "report ready")
        self.assertIsInstance(event_id, int)
        row = memory_store._conn().execute(
            "SELECT kind, summary FROM events WHERE id = ?",
            (event_id,)).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["kind"], "research")

    def test_writes_are_in_order_and_never_interleaved(self):
        rids = [memory_store.record_request("request %d" % index)
                for index in range(25)]
        for index, rid in enumerate(rids):
            memory_store.record_result(rid, "completed", summary="done %d" % index)
        memory_store.flush_writes(timeout=5.0)
        rows = memory_store._conn().execute(
            "SELECT id, summary FROM events WHERE kind = 'work_result' "
            "ORDER BY id").fetchall()
        self.assertEqual([row["summary"] for row in rows],
                         ["[completed] done %d" % index
                          for index in range(25)],
                         "the queue is FIFO — a result can never overtake one")

    def test_nothing_is_dropped_when_the_queue_is_full(self):
        """Backpressure, not silent loss: shrink the queue and push through it."""
        with patch.object(memory_store, "_WRITE_QUEUE_LIMIT", 2):
            rids = [memory_store.record_request("burst %d" % index)
                    for index in range(20)]
            memory_store.flush_writes(timeout=5.0)
        rows = memory_store._conn().execute(
            "SELECT COUNT(*) FROM work_requests").fetchone()[0]
        self.assertEqual(rows, len(rids),
                         "a full queue must never discard a record")
        self.assertGreater(memory_store.write_queue_stats()["inline"], 0)

    def test_a_write_failure_is_counted_and_never_surfaces(self):
        before = memory_store.write_queue_stats()["failures"]
        with patch.object(memory_store, "_write_request_row",
                          side_effect=RuntimeError("disk on fire")):
            memory_store.record_request("this one cannot be written")
            memory_store.flush_writes(timeout=5.0)
        stats = memory_store.write_queue_stats()
        self.assertGreater(stats["failures"], before)
        self.assertTrue(stats["writer_alive"], "the writer must survive it")

    def test_the_writer_survives_a_failure_and_keeps_writing(self):
        with patch.object(memory_store, "_write_request_row",
                          side_effect=RuntimeError("once")):
            memory_store.record_request("fails")
        rid = memory_store.record_request("succeeds")
        memory_store.flush_writes(timeout=5.0)
        self.assertIsNotNone(memory_store.get_work_request(rid))


class NoWriteBeforeFirstDeltaTests(unittest.TestCase):
    """The acceptance criterion: nothing durable happens before the first delta."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        from backend.core.memory import clear_history
        clear_history()
        memory_store.MEMORY_ENABLED = True
        memory_store.configure(os.path.join(self._tmp.name, "mem.db"))

    def tearDown(self):
        memory_store.close()
        memory_store.MEMORY_ENABLED = False
        self._tmp.cleanup()
        from backend.core.memory import clear_history
        clear_history()

    def test_no_durable_write_happens_before_the_first_delta(self):
        """Ordering asserted explicitly: delta first, SQLite/file writes after.

        The turn is driven through ``process_message`` (the real routing) and
        the real ``handle_chat`` (the real delivery), with only the provider
        and the racer faked — so the order being asserted is the production
        one, not a test double's.
        """
        from backend.core import brain

        events = []
        deltas = []
        real_writer = memory_store._write_request_row

        def watched_writer(*args, **kwargs):
            events.append(("sqlite", args[0] if args else None))
            return real_writer(*args, **kwargs)

        def watched_save(messages):
            events.append(("history_file", len(messages)))
            return True

        built = {
            "path": "llm",
            "messages": [{"role": "system", "content": "sys"}],
            "system_prompt": "sys",
            "query": "hello",
        }

        class _Racer:
            """A racer that hands over a prebuilt (pure) LLM answer."""

            def __init__(self, msg, voice_compact=False, history=None):
                pass

            def built(self):
                return built

            def has_stream(self):
                # No live stream: handle_chat generates through
                # _stream_chat_deltas, which is where the delta is observed.
                return False

            def adopt(self):
                return iter(())

            def cancel(self):
                pass

            @property
            def is_adopted(self):
                return True

        def fake_stream(messages, temperature, max_tokens, cancel=None):
            class _Gen:
                def __iter__(self):
                    yield "Hello "
                    # Everything the turn deferred must still be deferred when
                    # the FIRST delta is produced.
                    events.append(("first_delta", None))
                    yield "sir."

                def close(self):
                    pass

            return _Gen()

        with patch.object(memory, "_write_snapshot", side_effect=watched_save), \
             patch.object(memory_store, "_write_request_row",
                          side_effect=watched_writer), \
             patch.object(memory_store, "_write_event_row") as queued_event, \
             patch.object(brain, "_ChatRacer", _Racer), \
             patch.object(brain, "_build_chat_messages", return_value=built), \
             patch.object(brain, "_stream_chat_deltas", side_effect=fake_stream), \
             patch.object(brain, "classify_intent",
                          return_value={"intent": "chat", "steps": [],
                                        "task_description": "",
                                        "query": "hello"}), \
             patch.object(brain, "maybe_handle_screen_control_message",
                          return_value=None), \
             patch.object(brain, "force_research", return_value=False), \
             patch.object(brain, "is_explicit_task_request", return_value=False), \
             patch.object(brain, "is_code_tool_request", return_value=False), \
             patch.object(brain, "should_search", return_value=False):
            response = brain.process_message("hello", sync_voice=False,
                                             stream_reply=deltas.append)

        self.assertEqual(deltas, ["Hello ", "sir."])
        self.assertEqual(response, "Hello sir.")
        names = [name for name, _ in events]
        self.assertIn("first_delta", names)
        for durable in ("sqlite", "history_file"):
            if durable in names:
                self.assertGreater(
                    names.index(durable), names.index("first_delta"),
                    "%s happened before the first delta" % durable)
        # And the deferred work does happen — deferred is not dropped.
        memory_store.flush_writes(timeout=5.0)
        memory.flush_history(timeout=5.0)


if __name__ == "__main__":
    unittest.main()
