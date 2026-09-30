"""P1-14 — SSE must not hold a thread per stream, and barge-in must survive.

Before this item ``/ask/stream`` streamed through a SYNC generator, so every
open stream occupied a thread from the same 40-thread pool that serves
/speak/stop, /task/stop, /ask/cancel and /voice-state. A handful of open chat
streams could starve the very endpoints needed to stop them. Frames were also
written one at a time, and ``RequestState.delta`` updated the text accumulator
and appended the frame under two separate lock acquisitions, so two producers
could interleave and leave the accumulated text disagreeing with the order of
the frames a resuming client rebuilds from.
"""

import asyncio
import json
import threading
import time
import unittest
from unittest.mock import patch

import anyio
from fastapi import HTTPException

from backend import watcher
from backend.api import routes
from backend.services import request_registry

EVENT_LIMIT = request_registry.EVENT_LIMIT


def _state(message="hi"):
    return request_registry.RequestState(
        request_registry.new_request_id(), message)


async def _collect(agen):
    """Flatten an astream generator into ``(batches, frames)``."""
    batches = []
    async for batch in agen:
        batches.append(list(batch))
    return batches, [frame for batch in batches for frame in batch]


class AsyncStreamTests(unittest.IsolatedAsyncioTestCase):
    """[P1-14] One async consumer, batched delivery, exact sequence numbers."""

    async def test_a_burst_arrives_as_one_batch_with_exact_seqs(self):
        state = _state()
        for chunk in ("a", "b", "c", "d"):
            state.delta(chunk)
        state.complete("abcd")

        batches, frames = await _collect(state.astream(last_seq=-1))

        self.assertEqual([f["seq"] for f in frames], [0, 1, 2, 3, 4])
        self.assertEqual([f["type"] for f in frames],
                         ["delta"] * 4 + ["completed"])
        self.assertEqual(len(batches), 1,
                         "a burst that is already buffered is ONE write")

    async def test_seq_numbers_are_never_renumbered_or_dropped(self):
        state = _state()
        for index in range(25):
            state.delta(str(index))
        state.complete("done")

        _, frames = await _collect(state.astream(last_seq=-1))

        seqs = [f["seq"] for f in frames]
        self.assertEqual(seqs, list(range(26)))
        self.assertEqual(len(set(seqs)), len(seqs), "a seq was duplicated")

    async def test_a_producer_thread_wakes_a_parked_consumer(self):
        """The frame is appended on a WORKER THREAD, never on the loop."""
        state = _state()
        agen = state.astream(last_seq=-1, heartbeat=30.0)
        first = asyncio.ensure_future(agen.__anext__())
        await asyncio.sleep(0.05)
        self.assertFalse(first.done(), "an empty stream must park, not spin")

        threading.Thread(target=lambda: state.delta("from a worker"),
                         daemon=True).start()

        batch = await asyncio.wait_for(first, timeout=2.0)
        self.assertEqual([f["text"] for f in batch], ["from a worker"])
        await agen.aclose()

    async def test_a_pending_batch_is_flushed_on_the_coalescing_timer(self):
        """A partial utterance must not wait for the next one."""
        state = _state()
        agen = state.astream(last_seq=-1, heartbeat=30.0, flush=0.01)
        started = time.monotonic()
        first = asyncio.ensure_future(agen.__anext__())
        await asyncio.sleep(0.02)
        state.delta("partial")

        batch = await asyncio.wait_for(first, timeout=2.0)
        elapsed = time.monotonic() - started

        self.assertEqual([f["text"] for f in batch], ["partial"])
        self.assertLess(elapsed, 1.0, "the flush window was not bounded")
        await agen.aclose()

    async def test_resume_from_last_event_id_delivers_exactly_the_rest(self):
        state = _state()
        for chunk in ("one", "two", "three"):
            state.delta(chunk)
        state.complete("onetwothree")

        _, frames = await _collect(state.astream(last_seq=1))

        self.assertEqual([f["seq"] for f in frames], [2, 3])
        self.assertEqual([f.get("text") for f in frames], ["three", None])
        self.assertFalse(state.started,
                         "a resume must never re-execute the request")

    async def test_a_client_that_fell_behind_gets_a_snapshot_first(self):
        """A slow client is bounded by EVENT_LIMIT, never by memory."""
        state = _state()
        for index in range(EVENT_LIMIT + 40):
            state.delta("x")
        state.complete("done")

        self.assertLessEqual(len(state.events), EVENT_LIMIT)

        _, frames = await _collect(state.astream(last_seq=0))

        self.assertTrue(frames[0].get("snapshot"),
                        "text the client already rendered must be restored")
        self.assertIsNone(frames[0]["seq"],
                          "a snapshot is not a numbered event")
        self.assertEqual(frames[-1]["type"], "completed")

    async def test_a_stop_event_flushes_what_is_pending_then_ends(self):
        stop = threading.Event()
        state = _state()
        state.delta("already rendered")
        stop.set()

        batches, frames = await _collect(
            state.astream(last_seq=-1, stop_event=stop))

        self.assertEqual([f["text"] for f in frames], ["already rendered"])
        self.assertEqual(len(batches), 1)

    async def test_a_quiet_stream_emits_unnumbered_heartbeats(self):
        state = _state()
        agen = state.astream(last_seq=-1, heartbeat=0.05)

        batch = await asyncio.wait_for(agen.__anext__(), timeout=2.0)

        self.assertEqual(batch[0]["type"], "progress")
        self.assertTrue(batch[0].get("heartbeat"))
        self.assertIsNone(batch[0]["seq"],
                          "heartbeats must never advance a resume cursor")
        await agen.aclose()

    async def test_a_closed_consumer_is_unregistered(self):
        state = _state()
        agen = state.astream(last_seq=-1, heartbeat=30.0)
        pending = asyncio.ensure_future(agen.__anext__())
        await asyncio.sleep(0.05)
        self.assertEqual(len(state._async_waiters), 1)

        pending.cancel()
        try:
            await pending
        except asyncio.CancelledError:
            pass
        state.delta("after the client left")   # must not raise

        self.assertEqual(state._async_waiters, [])

    async def test_a_dead_loop_never_breaks_the_producer(self):
        """A client that disconnected cannot break the worker thread."""
        state = _state()

        class _DeadLoop:
            def call_soon_threadsafe(self, *args):
                raise RuntimeError("event loop is closed")

        state.add_async_waiter(_DeadLoop(), asyncio.Event())

        state.delta("still works")             # must not raise

        self.assertEqual(state.text, "still works")
        self.assertEqual(state._async_waiters, [],
                         "the dead waiter must be dropped")


class DeltaAtomicityTests(unittest.TestCase):
    """[P1-14] The text accumulator and the frame are ONE critical section."""

    class _CountingCondition:
        """Counts lock acquisitions so atomicity is observable, not inferred."""

        def __init__(self, inner):
            self.inner = inner
            self.entries = 0

        def __enter__(self):
            self.entries += 1
            return self.inner.__enter__()

        def __exit__(self, *exc):
            return self.inner.__exit__(*exc)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    def test_delta_takes_the_lock_exactly_once(self):
        state = _state()
        counter = self._CountingCondition(state.cond)
        state.cond = counter

        state.delta("single")

        self.assertEqual(counter.entries, 1,
                         "text and frame were updated under two locks")

    def test_replace_takes_the_lock_exactly_once(self):
        state = _state()
        counter = self._CountingCondition(state.cond)
        state.cond = counter

        state.replace("whole answer")

        self.assertEqual(counter.entries, 1)

    def test_concurrent_producers_keep_text_in_frame_order(self):
        state = _state()
        producers = 4
        per_producer = 100        # < EVENT_LIMIT, so no frame is trimmed

        def produce(marker):
            for _ in range(per_producer):
                state.delta(marker)

        threads = [threading.Thread(target=produce, args=("abcd"[i],))
                   for i in range(producers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)

        frames = [frame for _seq, frame in state.events]
        rebuilt = "".join(frame["text"] for frame in frames)
        self.assertEqual(len(frames), producers * per_producer)
        self.assertEqual(state.text, rebuilt,
                         "accumulated text disagrees with the frame order")


class ControlCapacityTests(unittest.IsolatedAsyncioTestCase):
    """[P1-14] Barge-in capacity is RESERVED; an open stream holds no thread."""

    async def test_the_control_plane_has_its_own_reserved_capacity(self):
        self.assertEqual(routes.CONTROL_LIMITER.total_tokens, 4)
        self.assertIsNot(routes.CONTROL_LIMITER,
                         anyio.to_thread.current_default_thread_limiter(),
                         "control work must not share the bulk limiter")

    async def test_only_the_reserved_number_of_control_calls_run_at_once(self):
        release = threading.Event()
        started = []
        lock = threading.Lock()

        def blocking():
            with lock:
                started.append(1)
            release.wait(5.0)
            return "ok"

        tasks = [asyncio.ensure_future(routes._on_control_plane(blocking))
                 for _ in range(7)]
        await asyncio.sleep(0.3)
        try:
            self.assertEqual(len(started), 4,
                             "the reserved control capacity was not honoured")
        finally:
            release.set()
        results = await asyncio.gather(*tasks)
        self.assertEqual(results, ["ok"] * 7)

    async def test_a_control_call_answers_with_many_streams_open(self):
        """The regression test: open streams must not starve barge-in."""
        streams = []
        for _ in range(45):
            state = _state()
            state.delta("first token")
            agen = state.astream(last_seq=-1, heartbeat=30.0)
            await agen.__anext__()          # the backlog batch
            parked = asyncio.ensure_future(agen.__anext__())
            streams.append((agen, parked))
        await asyncio.sleep(0.05)
        try:
            self.assertEqual(
                anyio.to_thread.current_default_thread_limiter().borrowed_tokens,
                0,
                "an open stream is holding a thread-pool thread")
            started = time.monotonic()
            result = await asyncio.wait_for(
                routes._on_control_plane(lambda: "stopped"), timeout=2.0)
            elapsed = time.monotonic() - started
        finally:
            for agen, parked in streams:
                parked.cancel()
                try:
                    await parked
                except asyncio.CancelledError:
                    pass

        self.assertEqual(result, "stopped")
        self.assertLess(elapsed, 0.5,
                        "a control call queued behind open streams")

    async def test_the_sse_body_is_async_not_a_thread_per_stream(self):
        """A SYNC body iterator is what made every stream hold a thread."""
        request_id = request_registry.new_request_id()
        # Admit the request and consume the execute-once guard, so building the
        # response cannot start a real worker (this test is about the transport).
        state, _created, _conflict = request_registry.admit(request_id, "hi")
        self.assertTrue(request_registry.try_start(state))
        self.addCleanup(state.complete, "")
        try:
            response = routes.ask_stream(
                routes.Query(message="hi", request_id=request_id))

            body = response.body_iterator
            self.assertTrue(hasattr(body, "__anext__"),
                            "the SSE body must be an async iterator (P1-14)")
            first = await asyncio.wait_for(body.__anext__(), timeout=2.0)
            self.assertIn("attached", first)
            await body.aclose()
        finally:
            if not state.done:
                state.complete("")


class ControlRouteWiringTests(unittest.TestCase):
    """The HTTP surface must use the reserved capacity, not the bulk pool."""

    def _handler(self, method, path):
        for route in routes.router.routes:
            if route.path == path and method in route.methods:
                return route.endpoint
        raise AssertionError("no route for %s %s" % (method, path))

    def test_the_control_routes_are_async_wrappers(self):
        import inspect

        for method, path in (("POST", "/speak/stop"),
                             ("POST", "/task/stop"),
                             ("POST", "/ask/cancel/{request_id}"),
                             ("GET", "/voice-state")):
            endpoint = self._handler(method, path)
            self.assertTrue(
                inspect.iscoroutinefunction(endpoint),
                "%s %s still runs on the shared pool" % (method, path))

    def test_the_sync_implementations_stay_directly_callable(self):
        """Existing in-tree callers keep a plain synchronous function."""
        import inspect

        for name in ("stop_speech", "stop_task", "cancel_request",
                     "get_voice_state"):
            self.assertFalse(inspect.iscoroutinefunction(getattr(routes, name)))


class LiveSseTests(unittest.TestCase):
    """End-to-end over the ASGI app: the real cross-thread wake-up path."""

    def setUp(self):
        import os

        from backend.services import local_auth

        os.environ.pop("JARVIS_DEV_MODE", None)
        self.token = local_auth.mint_token()
        local_auth.configure(self.token)
        self.addCleanup(self._disarm)

    def _disarm(self):
        import os

        from backend.services import local_auth

        local_auth.configure("")
        os.environ.pop("JARVIS_LOCAL_TOKEN", None)
        os.environ.pop("JARVIS_DEV_MODE", None)

    def _headers(self):
        from backend.services import local_auth

        return {local_auth.HEADER: self.token}

    def _fake_worker(self, state, **_kwargs):
        """A worker thread that streams exactly like the real one does."""
        state.progress("started", stage="start")
        state.delta("Hel")
        time.sleep(0.02)
        state.delta("lo")
        state.complete("Hello")

    def _client(self):
        from fastapi.testclient import TestClient
        from backend.main import app

        return TestClient(app)

    def test_a_live_stream_delivers_numbered_frames_and_stops_for_control(self):
        import uuid

        request_id = "req-p14-" + uuid.uuid4().hex[:8]
        frames = []
        control = {}

        def press_stop():
            started = time.monotonic()
            try:
                response = client.post("/speak/stop", headers=self._headers())
                control["status"] = response.status_code
            except Exception as exc:                    # pragma: no cover
                control["error"] = exc
            control["elapsed"] = time.monotonic() - started

        with patch.object(routes, "_run_request_worker",
                          side_effect=self._fake_worker):
            with self._client() as client:
                with client.stream(
                        "POST", "/ask/stream",
                        json={"message": "hello", "request_id": request_id,
                              "speak": False},
                        headers=self._headers()) as response:
                    self.assertEqual(response.status_code, 200)
                    self.assertIn("text/event-stream",
                                  response.headers["content-type"])
                    # The stream is genuinely open and parked on the event.
                    for line in response.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        frames.append(json.loads(line[5:].strip()))
                        if len(frames) == 1:
                            thread = threading.Thread(target=press_stop,
                                                      daemon=True)
                            thread.start()
                            thread.join(5.0)
                            self.assertFalse(
                                thread.is_alive(),
                                "the control endpoint was starved by an open "
                                "stream")
                        if frames[-1]["type"] in ("completed", "interrupted",
                                                  "error"):
                            break

        self.assertEqual(control.get("status"), 200)
        self.assertLess(control.get("elapsed", 5.0), 2.0)
        numbered = [f["seq"] for f in frames if isinstance(f.get("seq"), int)]
        self.assertEqual(numbered, list(range(len(numbered))),
                         "frames must stay in exact seq order")
        self.assertEqual(frames[-1]["type"], "completed")
        self.assertEqual(frames[-1]["reply"], "Hello")
        self.assertEqual([f.get("text") for f in frames
                          if f["type"] == "delta"], ["Hel", "lo"])


class LauncherTests(unittest.TestCase):
    """[P1-14] The access log is synchronous I/O on the event loop."""

    class _FakeProc:
        pid = 4242

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    def test_the_backend_is_launched_without_an_access_log(self):
        with patch.object(watcher.subprocess, "Popen",
                          return_value=self._FakeProc()) as popen, \
             patch.object(watcher, "wait_for_backend_ready", return_value=True):
            watcher._spawn_backend({})

        argv = popen.call_args.args[0]
        self.assertIn("uvicorn", argv)
        self.assertIn("--no-access-log", argv)

    def test_the_electron_launcher_disables_the_access_log(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2]
        source = (root / "main.js").read_text(encoding="utf-8")
        backend_args = source.split("const backendArgs = () => [[", 1)[1]
        backend_args = backend_args.split("], {", 1)[0]
        self.assertIn('"uvicorn"', backend_args)
        self.assertIn('"--no-access-log"', backend_args)


class AuthMiddlewareTests(unittest.TestCase):
    """The audit flagged local_auth's async purity as an ASSUMPTION."""

    def test_local_auth_is_pure_asgi_middleware(self):
        import inspect

        from backend.services import local_auth

        self.assertTrue(inspect.iscoroutinefunction(
            local_auth.LocalTokenMiddleware.__call__))
        source = inspect.getsource(local_auth)
        for blocking in ("urllib.request", "requests.", "time.sleep",
                         "BaseHTTPMiddleware", "to_thread"):
            self.assertNotIn(blocking, source,
                             "local_auth does blocking work per request")


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
