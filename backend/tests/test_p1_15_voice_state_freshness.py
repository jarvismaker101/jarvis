"""P1-15 — voice-state must survive a worker restart and update immediately.

The publish handler rejected any update whose ``state_seq`` was not HIGHER than
the last one it saw — but that counter lives in the voice worker and restarts at
1 on every launch. So after a restart (or a crash-loop), every update from the
new worker was discarded as "old" and the UI showed a FROZEN voice state until
the counter climbed back past the old high-water mark. Publishing also ran only
once per second, so a transition could take a second to become visible.
"""

import threading
import time
import unittest
from unittest.mock import patch

from backend import listener_state
from backend.api import routes
from backend.services import request_registry


def _reset_published():
    routes._published_voice.clear()
    routes._published_voice_seq = 0
    routes._published_voice_publisher = ""


class _PublishTestCase(unittest.TestCase):
    def setUp(self):
        _reset_published()
        self.addCleanup(_reset_published)

    def publish(self, **payload):
        payload.setdefault("status", "listening")
        return routes.publish_voice_state(dict(payload))

    def served(self):
        return routes.get_published_voice_state()


class PublisherIdentityTests(_PublishTestCase):
    """Ordering is scoped to ONE publisher; a new one resets the baseline."""

    def test_a_new_publisher_with_a_low_seq_is_accepted(self):
        """THE regression test: a restarted worker starts counting at 1."""
        self.publish(publisher_id="launch-A", publisher_pid=111,
                     state_seq=500, status="speaking")

        result = self.publish(publisher_id="launch-B", publisher_pid=222,
                              state_seq=1, status="listening")

        self.assertFalse(result.get("stale"),
                         "a restarted worker's state was discarded as old")
        self.assertEqual(self.served().get("status"), "listening")
        self.assertEqual(routes._published_voice_seq, 1)

    def test_a_restart_is_visible_in_ui_state(self):
        self.publish(publisher_id="launch-A", publisher_pid=111,
                     state_seq=900, status="speaking")
        self.assertEqual(routes.get_ui_state()["state"].get("status"), "speaking")

        self.publish(publisher_id="launch-B", publisher_pid=222,
                     state_seq=1, status="listening")
        self.assertEqual(routes.get_ui_state()["state"].get("status"), "listening")

        # …and the new publisher's LATER updates keep flowing.
        self.publish(publisher_id="launch-B", publisher_pid=222,
                     state_seq=2, status="thinking")
        self.assertEqual(routes.get_ui_state()["state"].get("status"), "thinking")

    def test_an_out_of_order_update_from_the_same_publisher_is_rejected(self):
        self.publish(publisher_id="launch-A", state_seq=10, status="speaking")

        result = self.publish(publisher_id="launch-A", state_seq=4,
                              status="listening")

        self.assertTrue(result.get("stale"),
                        "a slow update overwrote a newer one")
        self.assertEqual(self.served().get("status"), "speaking")
        self.assertEqual(routes._published_voice_seq, 10)

    def test_a_repeated_seq_from_the_same_publisher_is_rejected(self):
        self.publish(publisher_id="launch-A", state_seq=7, status="thinking")

        result = self.publish(publisher_id="launch-A", state_seq=7,
                              status="listening")

        self.assertTrue(result.get("stale"))
        self.assertEqual(self.served().get("status"), "thinking")

    def test_the_pid_is_the_identity_when_there_is_no_launch_id(self):
        self.publish(publisher_pid=111, state_seq=400, status="speaking")

        result = self.publish(publisher_pid=222, state_seq=1,
                              status="listening")

        self.assertFalse(result.get("stale"))
        self.assertEqual(self.served().get("status"), "listening")

    def test_a_publisher_that_identifies_itself_not_at_all_keeps_legacy_order(
            self):
        """Legacy publishers still get the old single-counter protection."""
        self.publish(state_seq=10, status="speaking")

        result = self.publish(state_seq=3, status="listening")

        self.assertTrue(result.get("stale"))
        self.assertEqual(self.served().get("status"), "speaking")

    def test_a_seq_of_zero_is_never_treated_as_older(self):
        """A publisher that does not number its frames still gets through."""
        self.publish(publisher_id="launch-A", state_seq=9, status="speaking")

        self.publish(publisher_id="launch-A", status="listening")

        self.assertEqual(self.served().get("status"), "listening")

    def test_the_ui_state_shape_is_unchanged(self):
        """The frontend polls a fixed contract: add, never rename."""
        self.publish(publisher_id="launch-A", state_seq=1)
        state = routes.get_ui_state()["state"]

        for key in ("status", "assistant_speaking", "user_speaking",
                    "thinking", "threshold", "voice_input_enabled"):
            self.assertIn(key, state, "an existing /ui-state field vanished")
        # Additive only: the identity of the publisher is new information.
        self.assertIn("publisher_id", state)


class _LoopHarness(unittest.TestCase):
    """Runs the real publisher loop against a recording transport."""

    def setUp(self):
        self.posts = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.addCleanup(_reset_published)
        self.addCleanup(self.stop.set)
        self.addCleanup(listener_state.set_thinking, False)
        self.addCleanup(listener_state.set_speaking, False)
        from backend import voice_mode

        self.voice = voice_mode

        def record(path, payload, timeout=1.0):
            with self.lock:
                self.posts.append((time.monotonic(), dict(payload)))
            return True, {}

        self._patch = patch.object(voice_mode, "_post_backend",
                                   side_effect=record)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.thread = threading.Thread(
            target=voice_mode._publish_voice_state_loop,
            args=(self.stop,), daemon=True)
        self.thread.start()

    def wait_for(self, predicate, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                for stamp, payload in self.posts:
                    if predicate(payload, stamp):
                        return stamp, payload
            time.sleep(0.01)
        raise AssertionError("no matching publish within %.1fs" % timeout)


class ImmediateTransitionTests(_LoopHarness):
    def test_a_transition_is_published_without_waiting_for_the_tick(self):
        self.wait_for(lambda payload, _t: payload.get("state_seq"))

        started = time.monotonic()
        listener_state.set_thinking(True)
        stamp, payload = self.wait_for(
            lambda payload, _t: payload.get("thinking") is True)

        self.assertLess(stamp - started, 0.5,
                        "the transition waited for the 1s heartbeat")
        self.assertEqual(payload.get("status"), "thinking")

    def test_speaking_and_hearing_are_published_immediately_too(self):
        self.wait_for(lambda payload, _t: payload.get("state_seq"))

        listener_state.set_speaking(True)
        self.wait_for(lambda payload, _t: payload.get("status") == "speaking")

        listener_state.mark_user_speaking(True)
        self.wait_for(lambda payload, _t: payload.get("status") == "hearing")

        listener_state.mark_user_speaking(False)
        listener_state.set_speaking(False)
        self.wait_for(lambda payload, _t: payload.get("status") == "listening")

    def test_the_heartbeat_still_publishes_when_nothing_changes(self):
        """A dead worker must stay detectable: no change, still a tick."""
        self.wait_for(lambda payload, _t: payload.get("state_seq"))

        _, second = self.wait_for(
            lambda payload, _t: payload.get("state_seq", 0) >= 2, timeout=3.0)

        self.assertGreaterEqual(second.get("state_seq"), 2)

    def test_the_publish_carries_a_per_launch_identity(self):
        _, payload = self.wait_for(lambda payload, _t: payload.get("state_seq"))

        self.assertEqual(payload.get("publisher_id"),
                         self.voice._VOICE_LAUNCH_ID)
        self.assertTrue(payload.get("publisher_id"))
        self.assertTrue(payload.get("publisher_pid"))

    def test_a_failing_publish_never_kills_the_loop(self):
        self.wait_for(lambda payload, _t: payload.get("state_seq"))

        with patch.object(self.voice, "_post_backend",
                          side_effect=RuntimeError("backend down")):
            listener_state.set_thinking(True)
            time.sleep(0.2)

        listener_state.set_thinking(False)
        _, payload = self.wait_for(
            lambda payload, _t: payload.get("state_seq", 0) >= 3, timeout=3.0)

        self.assertGreaterEqual(payload.get("state_seq"), 3,
                                "the publisher stopped after a failed POST")


class StateHookTests(unittest.TestCase):
    """The listener's hooks must be cheap, safe and non-raising."""

    def setUp(self):
        self.addCleanup(listener_state.set_thinking, False)
        self.addCleanup(listener_state.set_speaking, False)
        self.addCleanup(listener_state.set_voice_input_enabled, True)

    def test_a_raising_hook_never_reaches_the_caller(self):
        calls = []

        def boom():
            calls.append("called")
            raise RuntimeError("hook blew up")

        listener_state.register_state_hook(boom)
        self.addCleanup(listener_state.unregister_state_hook, boom)

        listener_state.set_thinking(True)      # must not raise
        listener_state.set_speaking(True)
        listener_state.mark_user_speaking(True)

        self.assertEqual(len(calls), 3)

    def test_a_hook_is_called_outside_the_state_lock(self):
        """A hook that touches the state must not deadlock the setter."""
        seen = []

        def hook():
            seen.append(listener_state.get_voice_state()["status"])

        listener_state.register_state_hook(hook)
        self.addCleanup(listener_state.unregister_state_hook, hook)

        listener_state.set_thinking(True)

        self.assertEqual(seen, ["thinking"])

    def test_a_hook_registers_once(self):
        listener_state.register_state_hook(self._noop)
        listener_state.register_state_hook(self._noop)
        self.addCleanup(listener_state.unregister_state_hook, self._noop)

        self.assertEqual(listener_state._state_hooks.count(self._noop), 1)

    def test_unregistering_stops_the_notifications(self):
        calls = []
        hook = lambda: calls.append(1)          # noqa: E731
        listener_state.register_state_hook(hook)
        listener_state.unregister_state_hook(hook)

        listener_state.set_thinking(True)

        self.assertEqual(calls, [])

    @staticmethod
    def _noop():
        pass


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
