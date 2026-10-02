"""S18 - conversation continues while a task runs; actions queue.

Acceptance: chat-only voice turns are submitted and answered during a
running task (the old full mute is gone); task-class requests made mid-task
are HELD in a bounded action queue and drained - in order, with their reply
spoken for voice turns - when the task releases the machinery.
"""

import time
import unittest
from unittest.mock import patch

from backend.core import brain


def _wait_until(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class _ActionQueueIsolation(unittest.TestCase):
    def setUp(self):
        brain.set_opencode_task_running(False)
        with brain._action_queue_lock:
            brain._pending_action_requests.clear()

    def tearDown(self):
        brain.set_opencode_task_running(False)
        with brain._action_queue_lock:
            brain._pending_action_requests.clear()


class QueueActionRequestTests(_ActionQueueIsolation):
    def test_no_hold_when_no_task_is_running(self):
        self.assertIsNone(brain._queue_action_request("open chrome", True))
        self.assertEqual(brain._pending_action_requests, [])

    def test_action_request_is_held_while_a_task_runs(self):
        brain.set_opencode_task_running(True)
        reply = brain._queue_action_request("open chrome and scrape x", True)
        self.assertEqual(reply, brain.ACTION_QUEUED_REPLY)
        self.assertEqual(len(brain._pending_action_requests), 1)
        self.assertEqual(brain._pending_action_requests[0],
                         {"message": "open chrome and scrape x",
                          "from_voice": True})

    def test_the_queue_is_bounded(self):
        brain.set_opencode_task_running(True)
        for _ in range(brain.MAX_QUEUED_ACTIONS):
            brain._queue_action_request("do a thing", False)
        reply = brain._queue_action_request("one more", False)
        self.assertEqual(reply, brain.ACTION_QUEUE_FULL_REPLY)
        self.assertEqual(len(brain._pending_action_requests),
                         brain.MAX_QUEUED_ACTIONS)


class DrainTests(_ActionQueueIsolation):
    def _fill(self, *items):
        brain.set_opencode_task_running(True)
        for msg, voice in items:
            self.assertIsNotNone(brain._queue_action_request(msg, voice))

    def test_drain_runs_queued_actions_in_order_and_notifies_voice(self):
        self._fill(("first queued task", True), ("second queued task", False))
        with patch.object(brain, "process_message") as pm, \
                patch.object(brain, "_notify_async_reply") as notify:
            pm.side_effect = ["first done, sir", "second done"]
            brain.set_opencode_task_running(False)
            self.assertTrue(_wait_until(lambda: pm.call_count == 2))
            self.assertTrue(_wait_until(lambda: notify.called))
        self.assertEqual(
            [(c.args[0], c.kwargs.get("from_voice")) for c in pm.call_args_list],
            [("first queued task", True), ("second queued task", False)])
        # Only the VOICE turn's reply rides the delivery surface.
        notify.assert_called_once_with("first done, sir")
        self.assertEqual(brain._pending_action_requests, [])

    def test_a_failing_queued_action_does_not_block_the_rest(self):
        self._fill(("boom task", True), ("safe task", True))
        with patch.object(brain, "process_message") as pm, \
                patch.object(brain, "_notify_async_reply") as notify:
            pm.side_effect = [RuntimeError("engine exploded"), "safe done"]
            brain.set_opencode_task_running(False)
            self.assertTrue(_wait_until(lambda: pm.call_count == 2))
        notify.assert_called_once_with("safe done")

    def test_empty_queue_drain_is_a_quiet_noop(self):
        # Every tearDown/stray flag flip drains; an empty queue must not
        # spawn work or touch process_message.
        with patch.object(brain, "process_message") as pm:
            brain.set_opencode_task_running(False)
            time.sleep(0.05)
        pm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
