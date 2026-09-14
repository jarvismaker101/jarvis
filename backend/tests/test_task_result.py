"""G1 Round 1 (F03/F05): TaskResult wire-string/bool semantics, builders,
evidence lists, and the FailureTracker identical-action cap."""

import unittest

from backend.services.task_result import (
    FailureTracker,
    TaskResult,
    is_failure_text,
)


class TaskResultStringSemanticsTests(unittest.TestCase):
    """str() reproduces the legacy wire strings byte-for-byte."""

    def test_failed_str_matches_legacy_fail_format(self):
        result = TaskResult.failed("boom")
        self.assertEqual(str(result), "TASK NOT COMPLETED. Error: boom")

    def test_failed_str_falls_back_to_summary_without_error(self):
        result = TaskResult(status="failed", summary="fell over")
        self.assertEqual(str(result), "TASK NOT COMPLETED. Error: fell over")

    def test_failed_plain_returns_bare_message(self):
        result = TaskResult.failed("I could not plan that task.", plain=True)
        self.assertEqual(str(result), "I could not plan that task.")
        self.assertEqual(result.status, "failed")
        self.assertFalse(result)
        self.assertNotIn("TASK NOT COMPLETED", str(result))

    def test_failed_without_plain_keeps_legacy_prefix(self):
        result = TaskResult.failed("I could not plan that task.")
        self.assertEqual(
            str(result),
            "TASK NOT COMPLETED. Error: I could not plan that task.")

    def test_stopped_str_is_byte_identical(self):
        self.assertEqual(str(TaskResult.stopped()), "Stopped per your request.")

    def test_completed_str_is_the_summary(self):
        result = TaskResult.completed("Done: found the price.", detail="full")
        self.assertEqual(str(result), "Done: found the price.")

    def test_partial_str_is_the_summary(self):
        result = TaskResult.partial("Half done.", evidence=["x failed"])
        self.assertEqual(str(result), "Half done.")

    def test_needs_input_str_is_the_question(self):
        result = TaskResult.needs_input("Which site?")
        self.assertEqual(str(result), "Which site?")

    def test_startswith_and_contains_delegate_to_str(self):
        result = TaskResult.failed("nope (tool: navigate)")
        self.assertTrue(result.startswith("TASK NOT COMPLETED"))
        self.assertIn("nope (tool: navigate)", result)

    def test_eq_against_plain_strings(self):
        self.assertEqual(TaskResult.completed("Done."), "Done.")
        self.assertEqual(TaskResult.stopped(), "Stopped per your request.")
        self.assertNotEqual(TaskResult.completed("Done."), "Other.")


class TaskResultBoolSemanticsTests(unittest.TestCase):
    """Only 'failed' is falsy, so `if output:` keeps meaning usable output."""

    def test_failed_is_falsy(self):
        self.assertFalse(TaskResult.failed("boom"))
        self.assertFalse(TaskResult.failed("opencode timed out after 5s"))

    def test_all_other_statuses_are_truthy(self):
        self.assertTrue(TaskResult.completed("ok"))
        self.assertTrue(TaskResult.completed(""))
        self.assertTrue(TaskResult.partial("half"))
        self.assertTrue(TaskResult.stopped())
        self.assertTrue(TaskResult.needs_input("Which site?"))


class TaskResultBuilderTests(unittest.TestCase):
    def test_builders_set_fields(self):
        completed = TaskResult.completed("s", detail="d", evidence=["e1"])
        self.assertEqual(
            (completed.status, completed.summary, completed.detail,
             completed.evidence, completed.error),
            ("completed", "s", "d", ["e1"], None),
        )
        partial = TaskResult.partial("s", detail="d", evidence=["e1"],
                                     error="e")
        self.assertEqual(partial.status, "partial")
        self.assertEqual(partial.evidence, ["e1"])
        self.assertEqual(partial.error, "e")
        failed = TaskResult.failed("bad", detail="raw transcript")
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.detail, "raw transcript")
        self.assertFalse(failed)
        stopped = TaskResult.stopped()
        self.assertEqual(stopped.status, "stopped")
        question = TaskResult.needs_input("Which site?", detail="raw")
        self.assertEqual(
            (question.status, question.summary, question.detail),
            ("needs_input", "Which site?", "raw"),
        )

    def test_evidence_lists_do_not_alias(self):
        seed = ["a"]
        first = TaskResult.completed("s", evidence=seed)
        seed.append("b")
        first.evidence.append("c")
        second = TaskResult.completed("s", evidence=["a"])
        self.assertEqual(first.evidence, ["a", "c"])
        self.assertEqual(second.evidence, ["a"])

    def test_invalid_status_rejected(self):
        with self.assertRaises(ValueError):
            TaskResult(status="done-ish", summary="x")

    def test_is_failure_text(self):
        self.assertTrue(is_failure_text("TASK NOT COMPLETED. Error: x"))
        self.assertTrue(is_failure_text(TaskResult.failed("x")))
        self.assertFalse(is_failure_text("Done."))
        self.assertFalse(is_failure_text(TaskResult.completed("Done.")))
        self.assertFalse(is_failure_text(""))
        self.assertFalse(is_failure_text(None))


class FailureTrackerTests(unittest.TestCase):
    def test_key_is_canonical_across_argument_order(self):
        first = FailureTracker.key("navigate", {"b": 1, "a": 2})
        second = FailureTracker.key("navigate", {"a": 2, "b": 1})
        self.assertEqual(first, second)
        self.assertNotEqual(
            first, FailureTracker.key("navigate", {"a": 2, "b": 3}))
        self.assertNotEqual(
            first, FailureTracker.key("evaluate", {"a": 2, "b": 1}))

    def test_success_resets_counter(self):
        tracker = FailureTracker()
        key = FailureTracker.key("navigate", {"url": "https://x"})
        self.assertEqual(tracker.record_failure(key), 1)
        tracker.record_success(key)
        self.assertEqual(tracker.failures_for(key), 0)
        self.assertEqual(tracker.record_failure(key), 1)

    def test_thresholds_match_policy(self):
        self.assertEqual(FailureTracker.NOTE_AFTER, 2)
        self.assertEqual(FailureTracker.BLOCK_AFTER, 3)
        self.assertIn("2 times", FailureTracker.REPEAT_NOTE)


if __name__ == "__main__":
    unittest.main()
