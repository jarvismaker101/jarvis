"""F05 — recovery limits across model turns, with outcome-unknown mutations.

Acceptance (audit report): "Virtual/native failures share the same attempt
allowance; a lost response after submission cannot cause a second submission
without independent evidence of noncommitment."

Baseline defects pinned here:
  * virtual (composite) tools returned BEFORE the tracker block, so they were
    never counted and could fail forever;
  * a tool that reported its failure in the RETURN TEXT reset the
    identical-action counter every turn ("textual failure can reset it");
  * a mutation whose transport died after submission was retried on the next
    model turn as if nothing had happened.
"""

import unittest

from backend.services import browser_agent
from backend.services.task_result import FailureTracker


class TrackerStructuredOutcomeTests(unittest.TestCase):
    def test_identical_failures_are_counted_per_action(self):
        tracker = FailureTracker()
        key = FailureTracker.key("click_locator", {"selector": "#save"})
        for expected in (1, 2, 3):
            self.assertEqual(tracker.record_failure(key), expected)
        self.assertEqual(tracker.failures_for(key), 3)
        tracker.record_success(key)
        self.assertEqual(tracker.failures_for(key), 0)

    def test_ambiguous_errors_are_recognised(self):
        for error in ("read timeout", "connection reset by peer",
                      "socket closed", "broken pipe", "EOF from daemon"):
            self.assertTrue(FailureTracker.is_ambiguous_error(error), error)
        for error in ("element not found", "invalid selector",
                      "permission denied"):
            self.assertFalse(FailureTracker.is_ambiguous_error(error), error)

    def test_unknown_outcome_blocks_replay_until_observed(self):
        tracker = FailureTracker()
        args = {"selector": "#pay"}
        key = FailureTracker.key("click_locator", args)
        target = FailureTracker.target_key(args)
        self.assertTrue(tracker.replay_decision(key, target)[0])
        tracker.mark_outcome_unknown(key, target, "socket closed")
        allowed, reason = tracker.replay_decision(key, target)
        self.assertFalse(allowed, "a lost response must block a second submit")
        self.assertIn("socket closed", reason)
        # An independent observation of the SAME target, afterwards, is the
        # evidence that permits a replay.
        tracker.note_observation(target, "look")
        allowed, _ = tracker.replay_decision(key, target)
        self.assertTrue(allowed)
        # ...and the action's failure count is untouched by reconciliation.
        self.assertEqual(tracker.failures_for(key), 0)

    def test_an_observation_of_a_different_target_does_not_reconcile(self):
        tracker = FailureTracker()
        args = {"selector": "#pay"}
        key = FailureTracker.key("click_locator", args)
        target = FailureTracker.target_key(args)
        tracker.mark_outcome_unknown(key, target, "timeout")
        tracker.note_observation(FailureTracker.target_key({"selector": "#other"}))
        self.assertFalse(tracker.replay_decision(key, target)[0])

    def test_target_key_is_stable_for_the_same_target(self):
        one = FailureTracker.target_key({"selector": "#Save", "url": "HTTPS://X"})
        two = FailureTracker.target_key({"url": "https://x", "selector": "#save"})
        self.assertEqual(one, two)
        self.assertEqual(FailureTracker.target_key({}), "")


class ToolTextFailureTests(unittest.TestCase):
    def test_failure_text_is_a_failure(self):
        for text in ("tool failed: timeout", "Error: no such element",
                     "virtual tool look failed: boom", "blocked after 3",
                     "not replayed: outcome unknown", "Failed to click",
                     '{"ok": false, "error": "nope"}'):
            self.assertTrue(browser_agent._tool_text_reports_failure(text), text)

    def test_real_content_is_not_a_failure(self):
        for text in ("Button 'Save' clicked.", "[3] Edit  [4] Delete",
                     '{"ok": true, "value": 3}'):
            self.assertFalse(browser_agent._tool_text_reports_failure(text), text)

    def test_empty_text_is_treated_as_failure(self):
        self.assertTrue(browser_agent._tool_text_reports_failure(""))


class SharedAllowanceTests(unittest.TestCase):
    def test_virtual_and_native_tools_share_one_allowance(self):
        # Both sets resolve to the SAME tracker + key policy.
        session = {}
        native = browser_agent._tracker_for_tool(
            session, "click_locator", {"selector": "#a"})
        virtual = browser_agent._tracker_for_tool(
            session, "click_mark", {"name": "3"})
        self.assertIs(native[0], virtual[0],
                      "one task must own exactly one tracker")

    def test_virtual_mutations_are_in_the_all_mutation_set(self):
        for name in ("click_mark", "click_point", "fill", "upload_file"):
            self.assertIn(name, browser_agent._ALL_MUTATION_TOOLS, name)
            self.assertIn(name, browser_agent._VIRTUAL_MUTATION_TOOLS, name)

    def test_observation_tools_match_the_real_virtual_names(self):
        for name in ("look", "batch_probe", "wait_for", "verify_playing"):
            self.assertIn(name, browser_agent._OBSERVATION_TOOLS, name)
            self.assertIn(name, browser_agent._VIRTUAL_TOOL_NAMES, name)
        for name in browser_agent._OBSERVATION_TOOLS:
            self.assertNotIn(name, browser_agent._ALL_MUTATION_TOOLS, name)


class TrackerWiringTests(unittest.TestCase):
    """The tracker must be consulted in the real dispatch path."""

    def test_virtual_branch_checks_the_block_threshold(self):
        source = browser_agent._run_one_tool.__doc__ or ""
        import inspect

        code = inspect.getsource(browser_agent._run_one_tool)
        virtual_start = code.index("if name in _VIRTUAL_TOOL_NAMES:")
        # NOTE: the native call appears LAST — the virtual branch calls the
        # same helper, so a plain index() would cut the slice too early.
        native_start = code.rindex("_tracker_for_tool(session, name, arguments)")
        virtual_block = code[virtual_start:native_start]
        self.assertIn("FailureTracker.BLOCK_AFTER", virtual_block,
                      "virtual tools must share the attempt allowance")
        self.assertIn("_ALL_MUTATION_TOOLS", virtual_block,
                      "virtual mutations need the same replay rule")

    def test_native_mutation_replay_is_gated(self):
        import inspect

        code = inspect.getsource(browser_agent._run_one_tool)
        self.assertIn("replay_decision", code)
        self.assertIn("mark_outcome_unknown", code)
        self.assertIn("_tool_text_reports_failure", code)


if __name__ == "__main__":
    unittest.main()
