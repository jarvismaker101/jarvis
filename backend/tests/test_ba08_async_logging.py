"""L-18 / BA-08 — async, non-blocking activity logging.

Verified 2026-10-04: every browser-agent tool call paid a blocking open +
redact + write + flush per log line (4+ lines per step on the critical
path). Now the request path only enqueues: one daemon writer thread in
``opencode_client`` drains the bounded queue in batches (ONE open per
batch, redaction on the writer), drops on overflow, and never raises into
the request path (rule 19). Narration gets the same treatment — the
throttle slot is claimed on the caller (nanoseconds), the ``speak()``
itself runs on the writer.
"""
import os
import queue as queue_mod
import tempfile
import unittest
from unittest.mock import patch

from backend.services import opencode_client


def _use_temp_log(test):
    """Point the activity log at a scratch file for one test."""
    tmpdir = tempfile.mkdtemp(prefix="activity_test_")
    path = os.path.join(tmpdir, "opencode_activity.log")
    patcher = patch.object(opencode_client, "_activity_log_path",
                           return_value=path)
    patcher.start()
    test.addCleanup(patcher.stop)
    return path


class NoIoOnRequestPathTests(unittest.TestCase):
    def test_append_never_opens_or_blocks(self):
        with patch.object(opencode_client, "_write_activity_blob") as blob, \
             patch.object(opencode_client, "_ensure_activity_log") as ensure:
            for i in range(5):
                self.assertIsNone(
                    opencode_client.append_activity_line("TOOL x %d\n" % i))
            # The request path enqueued only — nothing reached the file.
            blob.assert_not_called()
            ensure.assert_not_called()
            # Drain inside the mock context so no stray lines leak into
            # other tests' temp files.
            self.assertTrue(opencode_client.flush_activity_log(timeout=5.0))

    def test_append_never_raises_even_when_queue_broken(self):
        with patch.object(opencode_client, "_ACTIVITY_QUEUE") as broken, \
             patch.object(opencode_client, "_ACTIVITY_WRITER_STARTED", True):
            broken.put_nowait.side_effect = RuntimeError("queue exploded")
            self.assertIsNone(
                opencode_client.append_activity_line("TOOL x\n"))

    def test_truncate_never_raises(self):
        with patch.object(opencode_client, "flush_activity_log",
                          side_effect=RuntimeError("flush exploded")), \
             patch.object(opencode_client, "_truncate_activity_log") as trunc:
            self.assertIsNone(opencode_client.truncate_activity_log())
            trunc.assert_called_once_with()


class LinesStillAppearTests(unittest.TestCase):
    def test_queued_lines_land_in_order_after_flush(self):
        path = _use_temp_log(self)
        opencode_client.append_activity_line("TOOL navigate {}\n")
        opencode_client.append_activity_line("RESULT ok: done\n")
        opencode_client.append_activity_line("STOPWATCH tool=3\n")
        self.assertTrue(opencode_client.flush_activity_log(timeout=5.0))
        with open(path, encoding="utf-8") as handle:
            content = handle.read()
        self.assertLess(content.index("TOOL navigate"),
                        content.index("RESULT ok:"))
        self.assertLess(content.index("RESULT ok:"),
                        content.index("STOPWATCH tool=3"))

    def test_one_open_per_batch(self):
        # Deterministic at the unit level: the batch writer opens once no
        # matter how many lines the batch holds (the threaded path feeds
        # it the same way).
        with patch.object(opencode_client, "_write_activity_blob") as blob:
            opencode_client._write_lines_batch(
                ["line %d\n" % i for i in range(5)])
        self.assertEqual(blob.call_count, 1)
        blob_text = blob.call_args[0][0]
        for i in range(5):
            self.assertIn("line %d" % i, blob_text)

    def test_redaction_still_applies(self):
        path = _use_temp_log(self)
        opencode_client.append_activity_line(
            "RESULT ok: signed in with token sk-abcdefghijklmnopqrstuv\n")
        self.assertTrue(opencode_client.flush_activity_log(timeout=5.0))
        with open(path, encoding="utf-8") as handle:
            content = handle.read()
        self.assertNotIn("sk-abcdefghijklmnopqrstuv", content)

    def test_truncate_drains_first_then_empties(self):
        path = _use_temp_log(self)
        opencode_client.append_activity_line("TOOL stale\n")
        opencode_client.truncate_activity_log()
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "")


class OverflowDropsTests(unittest.TestCase):
    def test_full_queue_drops_and_counts(self):
        fresh = queue_mod.Queue(maxsize=2)
        saved_queue = opencode_client._ACTIVITY_QUEUE
        saved_started = opencode_client._ACTIVITY_WRITER_STARTED
        saved_dropped = opencode_client._activity_dropped
        opencode_client._ACTIVITY_QUEUE = fresh
        # Pretend a writer owns `fresh` so none spawns; nothing drains it.
        opencode_client._ACTIVITY_WRITER_STARTED = True
        try:
            self.assertTrue(opencode_client._enqueue_activity(("line", "a\n")))
            self.assertTrue(opencode_client._enqueue_activity(("line", "b\n")))
            self.assertFalse(opencode_client._enqueue_activity(("line", "c\n")))
            self.assertFalse(opencode_client._enqueue_activity(("line", "d\n")))
            dropped = (opencode_client.activity_queue_stats()["dropped"]
                       - saved_dropped)
            self.assertEqual(dropped, 2)
        finally:
            opencode_client._ACTIVITY_QUEUE = saved_queue
            opencode_client._ACTIVITY_WRITER_STARTED = saved_started

    def test_stats_shape(self):
        stats = opencode_client.activity_queue_stats()
        self.assertIn("queued", stats)
        self.assertIn("dropped", stats)
        self.assertIn("maxsize", stats)


class NarrationAsyncTests(unittest.TestCase):
    def setUp(self):
        opencode_client._reset_narration_state()
        opencode_client.set_narration_enabled(True)
        self.addCleanup(opencode_client.set_narration_enabled, False)
        self.addCleanup(opencode_client._reset_narration_state)

    def test_narration_speaks_on_writer_not_caller(self):
        with patch.object(opencode_client, "speak") as speak:
            opencode_client.narrate_activity("Opening the page, sir.")
            speak.assert_not_called()
            self.assertTrue(opencode_client.flush_activity_log(timeout=5.0))
        speak.assert_called_once_with("Opening the page, sir.")

    def test_throttle_still_claims_on_caller(self):
        with patch.object(opencode_client, "speak") as speak:
            opencode_client.narrate_activity("Clicking it, sir.")
            opencode_client.narrate_activity("Clicking it, sir.")
            self.assertTrue(opencode_client.flush_activity_log(timeout=5.0))
        speak.assert_called_once_with("Clicking it, sir.")

    def test_disabled_narration_enqueues_nothing(self):
        opencode_client.set_narration_enabled(False)
        before = opencode_client.activity_queue_stats()["queued"]
        opencode_client.narrate_activity("Opening the page, sir.")
        # Nothing claimed, nothing queued (the live writer may drain
        # earlier lines meanwhile — only assert no growth here).
        self.assertLessEqual(
            opencode_client.activity_queue_stats()["queued"], before)


if __name__ == "__main__":
    unittest.main()
