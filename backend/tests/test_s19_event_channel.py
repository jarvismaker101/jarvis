"""S19 — push state over one event channel instead of polling.

Three things used to be ASKED on a 1s-cached HTTP poll: "is a task running?"
and "is voice enabled?" from the voice worker, and the UI's /ui-state. S19
pushes each change the moment it happens over ``GET /events`` so the answer is
instant instead of up to a second stale. Polling stays as the fallback.

Covers the event bus, the three emitters (task running, voice enabled, voice
activity), the voice worker's cache updates on push, and the SSE endpoint.
"""

import json
import os
import unittest

from backend.services import event_bus
from backend.services import local_auth
from backend.listener_state import set_voice_input_enabled
from backend.core.brain import set_opencode_task_running
from backend.voice_mode import (
    _apply_pushed_state,
    _task_running_last_known,
    _task_running_checked_at,
    _voice_flag_last_known,
    _voice_flag_checked_at,
)


def _drain(q):
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except Exception:
            return out


class EventBusTests(unittest.TestCase):
    def setUp(self):
        event_bus.reset()

    def tearDown(self):
        event_bus.reset()

    def test_publish_reaches_every_subscriber(self):
        a = event_bus.subscribe()
        b = event_bus.subscribe()
        event_bus.publish("task_running", {"task_running": True})
        self.assertEqual(_drain(a), [{"type": "task_running", "task_running": True}])
        self.assertEqual(_drain(b), [{"type": "task_running", "task_running": True}])

    def test_snapshot_accumulates_and_replays_to_a_late_subscriber(self):
        event_bus.publish("task_running", {"task_running": True})
        event_bus.publish("voice_enabled", {"voice_input_enabled": False})
        snap = event_bus.snapshot()
        self.assertEqual(snap["task_running"], True)
        self.assertEqual(snap["voice_input_enabled"], False)

    def test_wakeup_is_called_after_an_event_is_queued(self):
        woke = []
        q = event_bus.subscribe(wakeup=lambda: woke.append(1))
        event_bus.publish("task_running", {"task_running": True})
        self.assertEqual(woke, [1])
        self.assertEqual(_drain(q)[0]["task_running"], True)

    def test_unsubscribe_stops_delivery(self):
        q = event_bus.subscribe()
        event_bus.unsubscribe(q)
        event_bus.publish("task_running", {"task_running": True})
        self.assertEqual(_drain(q), [])
        self.assertEqual(event_bus.subscriber_count(), 0)

    def test_a_wedged_reader_drops_oldest_not_grows(self):
        q = event_bus.subscribe()
        for i in range(200):
            event_bus.publish("task_running", {"task_running": bool(i % 2)})
        events = _drain(q)
        self.assertLessEqual(len(events), 64, "queue must stay bounded")
        # The NEWEST state survives even when the reader never drained.
        self.assertEqual(events[-1]["task_running"], bool(199 % 2))

    def test_a_broken_wakeup_never_breaks_the_emitter(self):
        def _boom():
            raise RuntimeError("bad wake")

        event_bus.subscribe(wakeup=_boom)
        event_bus.publish("task_running", {"task_running": True})  # must not raise


class TaskRunningEmitterTests(unittest.TestCase):
    def setUp(self):
        event_bus.reset()

    def tearDown(self):
        set_opencode_task_running(False)  # leave the module flag reset
        event_bus.reset()

    def test_flipping_the_task_flag_pushes_an_event(self):
        q = event_bus.subscribe()
        set_opencode_task_running(True)
        self.assertEqual(_drain(q), [{"type": "task_running", "task_running": True}])

    def test_setting_the_same_value_is_silent(self):
        set_opencode_task_running(True)
        q = event_bus.subscribe()
        _drain(q)
        set_opencode_task_running(True)  # no change -> no event
        self.assertEqual(_drain(q), [])


class VoiceEnabledEmitterTests(unittest.TestCase):
    def setUp(self):
        event_bus.reset()

    def tearDown(self):
        set_voice_input_enabled(True)  # restore the default
        event_bus.reset()

    def test_toggling_voice_input_pushes_an_event(self):
        q = event_bus.subscribe()
        set_voice_input_enabled(False)
        self.assertEqual(_drain(q),
                         [{"type": "voice_enabled", "voice_input_enabled": False}])

    def test_unchanged_toggle_is_silent(self):
        set_voice_input_enabled(True)  # already True (default) -> no event
        q = event_bus.subscribe()
        _drain(q)
        set_voice_input_enabled(True)
        self.assertEqual(_drain(q), [])


class PushApplyTests(unittest.TestCase):
    """The voice worker folds a pushed event into its two poll caches."""

    def setUp(self):
        import backend.voice_mode as vm
        self.vm = vm
        vm._task_running_last_known = False
        vm._task_running_checked_at = 0.0
        vm._voice_flag_last_known = True
        vm._voice_flag_checked_at = 0.0

    def test_a_pushed_task_flip_updates_the_cache_instantly(self):
        before = self.vm._task_running_checked_at
        _apply_pushed_state({"type": "task_running", "task_running": True})
        self.assertTrue(self.vm._task_running_last_known)
        self.assertGreater(self.vm._task_running_checked_at, before,
                           "pushed truth must refresh the poll TTL")

    def test_a_pushed_voice_flip_updates_the_cache(self):
        _apply_pushed_state({"type": "voice_enabled", "voice_input_enabled": False})
        self.assertFalse(self.vm._voice_flag_last_known)

    def test_a_snapshot_carries_both_fields(self):
        _apply_pushed_state({"type": "snapshot", "task_running": True,
                             "voice_input_enabled": False})
        self.assertTrue(self.vm._task_running_last_known)
        self.assertFalse(self.vm._voice_flag_last_known)

    def test_unrelated_events_are_ignored(self):
        _apply_pushed_state({"type": "voice_state", "speaking": True})
        self.assertFalse(self.vm._task_running_last_known)  # unchanged default
        self.assertTrue(self.vm._voice_flag_last_known)


class _AuthTestCase(unittest.TestCase):
    def setUp(self):
        os.environ.pop("JARVIS_DEV_MODE", None)
        self.token = local_auth.mint_token()
        local_auth.configure(self.token)
        event_bus.reset()
        self.addCleanup(self._disarm)

    def _disarm(self):
        local_auth.configure("")
        event_bus.reset()
        os.environ.pop("JARVIS_LOCAL_TOKEN", None)
        os.environ.pop("JARVIS_DEV_MODE", None)

    def headers(self, token=None):
        return {local_auth.HEADER: self.token if token is None else token}


class EventsEndpointTests(_AuthTestCase):
    def _client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        return TestClient(app)

    def test_events_is_a_private_read(self):
        client = self._client()
        # Unauthenticated: refused by the middleware BEFORE the stream starts,
        # so this is a finite response and cannot hang.
        self.assertEqual(client.get("/events").status_code, 401)

    def test_sse_frame_is_well_formed(self):
        from backend.api.routes import _sse
        line = _sse({"type": "task_running", "task_running": True})
        self.assertTrue(line.startswith("data: "))
        self.assertTrue(line.endswith("\n\n"))
        self.assertEqual(json.loads(line[6:].strip()),
                         {"type": "task_running", "task_running": True})

    def test_stream_delivers_snapshot_then_pushed_changes(self):
        import asyncio

        async def _take():
            from backend.api.routes import events_stream
            resp = await events_stream()          # StreamingResponse
            gen = resp.body_iterator              # the SSE async generator
            frames = []
            # 1) the current snapshot is the FIRST frame, no waiting.
            frames.append(await asyncio.wait_for(gen.asend(None), timeout=2))
            # 2) a pushed change arrives next, already queued by publish().
            event_bus.publish("task_running", {"task_running": True})
            frames.append(await asyncio.wait_for(gen.asend(None), timeout=2))
            await gen.aclose()
            return frames

        snap, change = asyncio.run(_take())
        self.assertIn("snapshot", snap)
        self.assertIn('"task_running"', change)
        self.assertEqual(json.loads(change[6:].strip())["task_running"], True)


if __name__ == "__main__":
    unittest.main()
