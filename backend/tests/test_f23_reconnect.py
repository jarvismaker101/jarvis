"""F23 — reconnect without reexecuting requests.

Acceptance (audit report): "More than 1,000 events stay ordered; concurrent
transports execute once; conflicts return 409; saturation never forgets live
work; reconnect restores text without repeated effects/speech."
"""

import json
import threading
import unittest
import uuid

from backend.services import request_registry as req_registry


def _fresh_state(message="hello"):
    return req_registry.RequestState(
        "req-" + uuid.uuid4().hex[:10], message)


class LongStreamTests(unittest.TestCase):
    def test_more_than_a_thousand_events_stay_ordered(self):
        state = _fresh_state()
        for index in range(1500):
            state.progress("step %d" % index)
        seqs = [seq for seq, _frame in state.events]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(set(seqs)), len(seqs))
        # The newest frame keeps the true global sequence, not the buffer
        # length (the old bug reused numbers after trimming).
        self.assertEqual(seqs[-1], 1499)

    def test_the_terminal_frame_survives_trimming_with_its_own_sequence(self):
        state = _fresh_state()
        for index in range(900):
            state.progress("step %d" % index)
        state.complete("done")
        self.assertTrue(state.done)
        self.assertEqual(state.events[-1][1]["type"], "completed")
        self.assertEqual(state.events[-1][0], 900)

    def test_resume_after_trimming_never_replays_old_frames(self):
        state = _fresh_state()
        for index in range(900):
            state.progress("step %d" % index)
        state.complete("done")
        cursor = state.events[0][0]
        delivered = [frame for _seq, frame in state.events
                     if frame["seq"] > cursor]
        self.assertTrue(delivered)
        self.assertTrue(all(f["seq"] > cursor for f in delivered))
        self.assertEqual(delivered[-1]["type"], "completed")


class AdmissionTests(unittest.TestCase):
    def _registry(self):
        return req_registry._Registry()

    def test_execute_once_across_concurrent_transports(self):
        registry = self._registry()
        state, created, conflict = registry.admit("req-shared", "same message")
        self.assertTrue(created)
        self.assertFalse(conflict)
        self.assertTrue(registry.try_start(state))
        results = []

        def _other_transport():
            _state, created_here, conflict_here = registry.admit(
                "req-shared", "same message")
            results.append((created_here, conflict_here,
                            registry.try_start(_state)))

        thread = threading.Thread(target=_other_transport)
        thread.start()
        thread.join()
        self.assertEqual(results, [(False, False, False)])

    def test_a_different_payload_under_the_same_id_is_a_conflict(self):
        registry = self._registry()
        registry.admit("req-x", "first message")
        _state, created, conflict = registry.admit("req-x", "a different ask")
        self.assertFalse(created)
        self.assertTrue(conflict)

    def test_whitespace_only_differences_are_not_a_conflict(self):
        registry = self._registry()
        registry.admit("req-y", "same message")
        _state, _created, conflict = registry.admit("req-y", "  same message ")
        self.assertFalse(conflict)

    def test_saturation_never_evicts_live_work(self):
        registry = self._registry()
        live = []
        for index in range(req_registry.MAX_REQUESTS + 5):
            state, created, _conflict = registry.admit(
                "req-live-%d" % index, "m%d" % index)
            if state is None:
                break
            state.started = True  # a live worker
            live.append(state)
        self.assertTrue(live)
        # Every admitted live request is still reachable, and none was reaped.
        for state in live:
            self.assertIs(registry.get(state.request_id), state)
        self.assertGreaterEqual(registry.live_count(), len(live))

    def test_finished_states_are_evicted_before_live_ones(self):
        registry = self._registry()
        finished = []
        for index in range(req_registry.MAX_REQUESTS):
            state, _created, _conflict = registry.admit(
                "req-fin-%d" % index, "m%d" % index)
            state.complete("done")
            finished.append(state)
        state, created, _conflict = registry.admit("req-new", "new work")
        self.assertTrue(created)
        self.assertIsNotNone(state)
        # A finished state made room; the new request is live and present.
        self.assertIs(registry.get("req-new"), state)

    def test_admission_refuses_when_every_slot_is_live(self):
        registry = self._registry()
        for index in range(req_registry.MAX_REQUESTS):
            state, created, _conflict = registry.admit(
                "req-full-%d" % index, "m%d" % index)
            self.assertTrue(created)
            state.started = True
        state, created, _conflict = registry.admit("req-overflow", "over")
        self.assertIsNone(state)
        self.assertFalse(created)


class TerminalImmutabilityTests(unittest.TestCase):
    def test_first_terminal_wins(self):
        state = _fresh_state()
        state.complete("the real answer")
        state.complete("a late answer")
        types = [frame["type"] for _seq, frame in state.events]
        self.assertEqual(types, ["completed"])
        self.assertEqual(state.reply, "the real answer")

    def test_a_late_error_cannot_replace_a_completion(self):
        state = _fresh_state()
        state.complete("done")
        state.error("late failure")
        types = [frame["type"] for _seq, frame in state.events]
        self.assertEqual(types, ["completed"])

    def test_interrupting_a_finished_request_is_a_noop_frame(self):
        state = _fresh_state()
        state.complete("done")
        state.interrupt("stopped by user")
        types = [frame["type"] for _seq, frame in state.events]
        self.assertEqual(types, ["completed"])


class SnapshotTests(unittest.TestCase):
    def test_a_client_behind_the_retention_window_gets_a_snapshot(self):
        state = _fresh_state()
        for index in range(900):
            state.delta("word%d " % index)
        state.complete("final text")
        frames = list(state.stream(last_seq=0, heartbeat=0.01))
        self.assertTrue(frames[0].get("snapshot"))
        self.assertEqual(frames[0]["text"], "final text")
        self.assertTrue(any(f["type"] == "completed" for f in frames))

    def test_an_up_to_date_client_gets_no_snapshot(self):
        state = _fresh_state()
        state.delta("hi ")
        state.complete("hi sir")
        frames = list(state.stream(last_seq=-1, heartbeat=0.01))
        self.assertFalse(any(f.get("snapshot") for f in frames))

    def test_snapshot_carries_the_accumulated_text(self):
        state = _fresh_state()
        state.delta("Hello ")
        state.delta("sir")
        frame = state.snapshot_frame()
        self.assertEqual(frame["text"], "Hello sir")

    def test_a_replacement_updates_the_snapshot_text(self):
        state = _fresh_state()
        state.delta("wrong text")
        state.replace("right text")
        self.assertEqual(state.snapshot_frame()["text"], "right text")


class RouteConflictTests(unittest.TestCase):
    def test_ask_returns_409_for_a_reused_id_with_a_new_message(self):
        from fastapi import HTTPException
        from backend.api import routes

        request_id = "req-409-" + uuid.uuid4().hex[:8]
        first = routes.Query(message="first message", request_id=request_id)
        routes.ask(first)
        with self.assertRaises(HTTPException) as caught:
            routes.ask(routes.Query(message="second message",
                                    request_id=request_id))
        self.assertEqual(caught.exception.status_code, 409)

    def test_ask_stream_returns_409_for_a_reused_id_with_a_new_message(self):
        from fastapi import HTTPException
        from backend.api import routes

        request_id = "req-409s-" + uuid.uuid4().hex[:8]
        first = routes.Query(message="first message", request_id=request_id)
        routes.ask(first)
        with self.assertRaises(HTTPException) as caught:
            routes.ask_stream(routes.Query(message="second message",
                                           request_id=request_id))
        self.assertEqual(caught.exception.status_code, 409)

    def test_a_retried_identical_request_is_not_reexecuted(self):
        from backend.api import routes

        request_id = "req-retry-" + uuid.uuid4().hex[:8]
        query = routes.Query(message="same message", request_id=request_id)
        state, created, conflict = req_registry.admit(request_id, "same message")
        self.assertTrue(created)
        self.assertFalse(conflict)
        # The worker already owns this request; the reply is authoritative.
        self.assertTrue(req_registry.try_start(state))
        state.complete("the one answer")
        first = routes.ask(query)
        second = routes.ask(query)
        self.assertEqual(first["reply"], "the one answer")
        self.assertEqual(second["reply"], "the one answer")


class VoiceResumeTests(unittest.TestCase):
    """The voice transport must resume, not replay, a streamed reply."""

    def _frames(self, payloads):
        return [("data: " + json.dumps(p) + "\n\n").encode("utf-8")
                for p in payloads]

    def _run(self, streams, sink, **kwargs):
        from backend import voice_mode

        calls = {"n": 0}
        captured = []

        class _Response:
            def __init__(self, lines):
                self._lines = lines

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def __iter__(self):
                for line in self._lines:
                    yield line

        def _urlopen(request, timeout=None):
            captured.append(json.loads(request.data.decode("utf-8")))
            index = calls["n"]
            calls["n"] += 1
            if index >= len(streams):
                raise OSError("stream dropped")
            return _Response(streams[index])

        import unittest.mock as mock
        with mock.patch.object(voice_mode, "urlopen", side_effect=_urlopen):
            return voice_mode._ask_backend(
                "hi", "req-voice", stream_sink=sink, **kwargs), captured

    def test_a_dropped_stream_resumes_from_the_cursor(self):
        spoken = []
        reply, captured = self._run([
            self._frames([
                {"type": "delta", "text": "Hello ", "seq": 0},
                {"type": "delta", "text": "sir", "seq": 1},
            ]),
            self._frames([
                {"type": "completed", "reply": "Hello sir, done.", "seq": 2},
            ]),
        ], spoken.append)
        self.assertEqual(spoken, ["Hello ", "sir"])   # no repetition
        self.assertEqual(captured[1]["last_event_id"], 1)
        self.assertEqual(reply, "Hello sir, done.")

    def test_a_resumed_snapshot_only_speaks_the_missing_tail(self):
        spoken = []
        reply, _captured = self._run([
            self._frames([{"type": "delta", "text": "Hello ", "seq": 0}]),
            self._frames([
                {"type": "replace", "text": "Hello sir", "seq": 1,
                 "snapshot": True},
                {"type": "completed", "reply": "Hello sir", "seq": 2},
            ]),
        ], spoken.append)
        self.assertEqual("".join(spoken), "Hello sir")
        self.assertEqual(reply, "Hello sir")

    def test_a_rewritten_answer_is_never_respoken(self):
        spoken = []
        replaced = []
        reply, _captured = self._run([
            self._frames([
                {"type": "delta", "text": "wrong answer", "seq": 0},
                {"type": "replace", "text": "right answer", "seq": 1},
                {"type": "completed", "reply": "right answer", "seq": 2},
            ]),
        ], spoken.append, replace_sink=replaced.append)
        # Audio already played is not repeated; the caller is told the truth.
        self.assertEqual(spoken, ["wrong answer"])
        self.assertEqual(replaced, ["right answer"])
        self.assertEqual(reply, "right answer")

    def test_voice_sends_a_cursor_on_the_first_attempt(self):
        _reply, captured = self._run([
            self._frames([{"type": "completed", "reply": "ok", "seq": 0}]),
        ], None)
        self.assertEqual(captured[0]["last_event_id"], -1)


if __name__ == "__main__":
    unittest.main()
