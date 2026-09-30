"""P1-13 — a streamed provider call is bounded, cancellable and unambiguous.

Before this item the OpenAI-compatible stream had NO deadline at all, its read
could not be interrupted (cancellation was only consulted between lines), a
``requests`` failure escaped the generator as a bare traceback, an in-stream
``{"error": ...}`` event was silently DROPPED (which is how a failed turn looked
like an empty one), and a response without a charset handed the loop BYTES.

These tests use a REAL local HTTP server for the parts that only a socket can
prove: a provider that accepts the connection and then goes silent, non-ASCII
text split across a chunk boundary, and a barge-in that must release a blocked
read by CLOSING the socket.
"""

import json
import logging
import uuid
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import requests

from backend.services import jobs as job_registry
from backend.services import openai_compat_client as client
from backend.core.deadline import Deadline


def _frame(payload):
    # ensure_ascii=False: the UTF-8 split test needs REAL multibyte bytes.
    return ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode("utf-8")


def _delta(text, finish_reason=None):
    return {"choices": [{"delta": {"content": text},
                         "finish_reason": finish_reason}]}


class _Server:
    """A real HTTP server that streams a scripted SSE body, then closes.

    * ``hang_after`` — sleep this many seconds after the Nth chunk, i.e. a
      provider that goes silent mid-stream (the blocked-read case).
    * ``split_at`` — write the body in two byte slices at this offset, i.e. a
      UTF-8 character split across a TCP write.
    """

    def __init__(self, chunks, hang_after=None, hang_seconds=30.0,
                 silent_seconds=None):
        self.chunks = list(chunks)
        self.hang_after = hang_after
        self.hang_seconds = hang_seconds
        #: Keep the connection OPEN with no body at all: the provider that
        #: accepted the request and then never produced a single token.
        self.silent_seconds = silent_seconds
        self.requests_seen = []
        self.first_write = threading.Event()
        self.finished = threading.Event()
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):  # noqa: N802 - stdlib naming
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                outer.requests_seen.append(self.path)
                self.send_response(200)
                # Deliberately NO charset: this is the shape that used to make
                # requests fall back to undecoded BYTES.
                self.send_header("Content-Type", "text/event-stream")
                # Real SSE is chunked, and chunked is what lets a reader take
                # each frame as it lands instead of waiting for a full buffer.
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                try:
                    if outer.silent_seconds:
                        time.sleep(outer.silent_seconds)
                    for index, chunk in enumerate(outer.chunks):
                        if outer.hang_after is not None and index == outer.hang_after:
                            time.sleep(outer.hang_seconds)
                        self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                        self.wfile.flush()
                        outer.first_write.set()
                    if not outer.silent_seconds:
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                except Exception:
                    return
                finally:
                    outer.finished.set()

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self.thread = threading.Thread(target=self._server.serve_forever,
                                       daemon=True)
        self.thread.start()

    @property
    def base_url(self):
        host, port = self._server.server_address[:2]
        return "http://%s:%s" % (host, port)

    def close(self):
        try:
            self._server.shutdown()
        except Exception:
            pass
        try:
            self._server.server_close()
        except Exception:
            pass


class _CancelHandle:
    """A minimal P0-08-style handle: an Event plus closers."""

    def __init__(self):
        self._event = threading.Event()
        self._closers = []
        self.closed = []

    def is_set(self):
        return self._event.is_set()

    def register_closer(self, closer):
        self._closers.append(closer)
        return closer

    def unregister_closer(self, closer):
        try:
            self._closers.remove(closer)
        except ValueError:
            pass

    def cancel(self):
        self._event.set()
        for closer in list(self._closers):
            self.closed.append(closer)
            closer()


class _StreamResult:
    """Collect a call's deltas + outcome so the assertions stay readable."""

    def __init__(self, **kwargs):
        self.outcome = client.new_stream_outcome()
        self.kwargs = kwargs

    def run(self, server, **overrides):
        kwargs = dict(self.kwargs)
        kwargs.update(overrides)
        self.deltas = list(client.ask_openai_compat_stream(
            [{"role": "user", "content": "hi"}],
            model="test-model",
            base_url=server.base_url,
            api_key="k",
            outcome=self.outcome,
            **kwargs,
        ))
        return self.deltas


class RealSocketTests(unittest.TestCase):
    """What only a real socket can prove."""

    def test_a_silent_provider_hits_the_first_token_budget(self):
        """A provider that accepts the connection then never speaks.

        The OLD code blocked in the read for up to 120s per chunk with no
        ceiling worth the name; now the time-to-first-token budget ends it with
        a clean terminal error instead of a hang.
        """
        server = _Server([], silent_seconds=30.0)   # headers, then silence
        self.addCleanup(server.close)
        result = _StreamResult(first_token_timeout=0.3)

        started = time.monotonic()
        deltas = result.run(server)
        elapsed = time.monotonic() - started

        self.assertEqual(deltas, [])
        self.assertEqual(result.outcome["status"], client.STREAM_ERRORED)
        self.assertTrue(result.outcome["error"],
                        "an errored stream must say WHY")
        self.assertLess(elapsed, 15.0,
                        "a silent provider held the read far too long")

    def test_no_read_can_block_the_first_token_forever(self):
        """The read budget IS the first-token budget (about 4s by default)."""
        server = _Server([], silent_seconds=30.0)
        self.addCleanup(server.close)
        with mock.patch.object(client._session, "post",
                               wraps=client._session.post) as post:
            _StreamResult(first_token_timeout=0.3).run(server)

        timeout = post.call_args.kwargs["timeout"]
        self.assertEqual(timeout[1], 0.3,
                         "the read budget must be the first-token budget")

    def test_non_ascii_survives_a_chunk_boundary(self):
        """A UTF-8 character split across two writes must not become mojibake.

        The response carries NO charset, which is exactly the case where
        requests hands back undecoded bytes — the old loop then compared bytes
        to str and raised.
        """
        body = _frame(_delta("héllo — 日本語 ✓")) + _frame(
            {"choices": [{"delta": {}, "finish_reason": "stop"}]})
        split = body.index("é".encode("utf-8")) + 1        # inside the char
        server = _Server([body[:split], body[split:]])
        self.addCleanup(server.close)

        deltas = _StreamResult().run(server)

        self.assertEqual("".join(deltas), "héllo — 日本語 ✓")

    def test_cancelling_closes_the_socket_and_frees_the_reader(self):
        """The audit's marquee case: cancel must release a BLOCKED read."""
        server = _Server([_frame(_delta("first")), _frame(_delta("second"))],
                         hang_after=1, hang_seconds=30.0)
        self.addCleanup(server.close)
        handle = _CancelHandle()
        seen = []
        outcome = client.new_stream_outcome()
        finished = threading.Event()

        def reader():
            try:
                for delta in client.ask_openai_compat_stream(
                    [{"role": "user", "content": "hi"}], model="m",
                    base_url=server.base_url, api_key="k",
                    cancel=handle, outcome=outcome,
                ):
                    seen.append(delta)
                    # The barge-in arrives while the NEXT read is blocked.
                    handle.cancel()
            finally:
                finished.set()

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        self.assertTrue(server.first_write.wait(5.0), "server never wrote")

        self.assertTrue(finished.wait(5.0),
                        "the reader thread was still parked after cancel")
        self.assertEqual(seen, ["first"],
                         "cancel must stop the stream at the next read")
        self.assertEqual(outcome["status"], client.STREAM_CANCELLED)
        self.assertTrue(handle.closed,
                        "cancelling must CLOSE the registered response")

    def test_the_socket_is_closed_even_when_the_consumer_stops_early(self):
        """Walking away mid-answer must not leak the connection."""
        closed = []

        class _Response:
            status_code = 200
            text = ""
            encoding = None

            def iter_lines(self, decode_unicode=False):
                yield 'data: {"choices":[{"delta":{"content":"one"}}]}'
                yield 'data: {"choices":[{"delta":{"content":"two"}}]}'

            def close(self):
                closed.append("closed")

        with mock.patch.object(client._session, "post", return_value=_Response()):
            stream = client.ask_openai_compat_stream(
                [{"role": "user", "content": "hi"}], model="m",
                base_url="http://127.0.0.1:1", api_key="k")
            self.assertEqual(next(iter(stream)), "one")
            stream.close()          # consumer walks away: the socket goes too

        self.assertEqual(closed, ["closed"])


class SilentFailureTests(unittest.TestCase):
    """Provider-side failures surface as terminal states, never as silence."""

    def _response(self, lines, raises=None):
        class _Response:
            status_code = 200
            encoding = None
            text = ""

            def iter_lines(self, decode_unicode=False):
                for line in lines:
                    yield line
                if raises is not None:
                    raise raises

            def close(self):
                pass

        return _Response()

    def run_stream(self, response, **kwargs):
        outcome = client.new_stream_outcome()
        with mock.patch.object(client._session, "post",
                               return_value=response):
            deltas = list(client.ask_openai_compat_stream(
                [{"role": "user", "content": "hi"}], model="m",
                base_url="http://127.0.0.1:1", api_key="k",
                outcome=outcome, **kwargs))
        return deltas, outcome

    def test_a_mid_stream_network_failure_is_terminal_not_a_traceback(self):
        response = self._response(
            ['data: {"choices":[{"delta":{"content":"par"}}]}'],
            raises=requests.ConnectionError("connection reset by peer"))
        deltas, outcome = self.run_stream(response)

        self.assertEqual(deltas, ["par"],
                         "text already yielded must survive")
        self.assertEqual(outcome["status"], client.STREAM_ERRORED)
        self.assertIn("reset", outcome["error"])

    def test_a_first_token_timeout_is_a_terminal_error(self):
        deltas, outcome = self.run_stream(
            self._response([], raises=requests.Timeout("read timed out")))

        self.assertEqual(deltas, [])
        self.assertEqual(outcome["status"], client.STREAM_ERRORED)

    def test_an_in_stream_error_event_is_logged_and_surfaces(self):
        """A provider error frame used to be dropped without a word."""
        response = self._response([
            'data: {"error":{"message":"model overloaded","type":"server"}}',
        ])
        with self.assertLogs(level="WARNING") as captured:
            deltas, outcome = self.run_stream(response)

        self.assertTrue(any("model overloaded" in line
                            for line in captured.output),
                        "the provider error must be LOGGED")
        self.assertEqual(outcome["status"], client.STREAM_ERRORED)
        self.assertIn("model overloaded", outcome["error"])

    def test_a_short_finish_reason_is_logged(self):
        response = self._response([
            'data: {"choices":[{"delta":{"content":"trunc"},"finish_reason":"length"}]}',
            "data: [DONE]",
        ])
        with self.assertLogs(level="WARNING") as captured:
            deltas, outcome = self.run_stream(response)

        self.assertEqual(deltas, ["trunc"])
        self.assertTrue(any("finish_reason=length" in line
                            for line in captured.output))
        self.assertEqual(outcome["finish_reason"], "length")

    def test_a_normal_stream_is_unambiguously_completed(self):
        response = self._response([
            'data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ])
        deltas, outcome = self.run_stream(response)

        self.assertEqual(deltas, ["done"])
        self.assertEqual(outcome["status"], client.STREAM_COMPLETED)
        self.assertEqual(outcome["chunks"], 1)
        self.assertTrue(outcome["first_token"])

    def test_a_stream_that_just_ends_is_completed_not_silent(self):
        response = self._response([
            'data: {"choices":[{"delta":{"content":"tail"}}]}',
        ])
        deltas, outcome = self.run_stream(response)

        self.assertEqual(deltas, ["tail"])
        self.assertEqual(outcome["status"], client.STREAM_COMPLETED)

    def test_the_response_encoding_is_pinned_to_utf8(self):
        """Without this, requests yields BYTES and the old loop raised."""
        response = self._response(['data: {"choices":[{"delta":{"content":"x"}}]}'])
        self.run_stream(response)
        self.assertEqual(response.encoding, "utf-8")

    def test_an_http_error_is_a_terminal_state(self):
        response = self._response([])
        response.status_code = 401
        response.text = "unauthorized"
        deltas, outcome = self.run_stream(response)

        self.assertEqual(deltas, [])
        self.assertEqual(outcome["status"], client.STREAM_ERRORED)
        self.assertIn("401", outcome["error"])

    def test_a_connection_failure_before_any_byte_is_terminal(self):
        outcome = client.new_stream_outcome()
        with mock.patch.object(client._session, "post",
                               side_effect=requests.ConnectionError("refused")):
            deltas = list(client.ask_openai_compat_stream(
                [{"role": "user", "content": "hi"}], model="m",
                base_url="http://127.0.0.1:1", api_key="k", outcome=outcome))

        self.assertEqual(deltas, [])
        self.assertEqual(outcome["status"], client.STREAM_ERRORED)

    def test_a_cancel_handle_without_closers_still_stops_the_loop(self):
        """A bare Event keeps its old meaning: stop at the next chunk."""
        cancel = threading.Event()
        response = self._response([
            'data: {"choices":[{"delta":{"content":"a"}}]}',
            'data: {"choices":[{"delta":{"content":"b"}}]}',
        ])
        outcome = client.new_stream_outcome()

        def post(*_args, **_kwargs):
            cancel.set()
            return response

        with mock.patch.object(client._session, "post", side_effect=post):
            deltas = list(client.ask_openai_compat_stream(
                [{"role": "user", "content": "hi"}], model="m",
                base_url="http://127.0.0.1:1", api_key="k",
                cancel=cancel, outcome=outcome))

        self.assertEqual(deltas, [])
        self.assertEqual(outcome["status"], client.STREAM_CANCELLED)


class DeadlineTests(unittest.TestCase):
    """F24's budget still owns the FIRST token."""

    def test_a_spent_budget_sends_no_request_at_all(self):
        outcome = client.new_stream_outcome()
        handle = Deadline.after(0.001)
        time.sleep(0.01)
        with mock.patch.object(client._session, "post") as post:
            deltas = list(client.ask_openai_compat_stream(
                [{"role": "user", "content": "hi"}], model="m",
                base_url="http://127.0.0.1:1", api_key="k",
                deadline=handle, outcome=outcome))

        post.assert_not_called()
        self.assertEqual(deltas, [])
        self.assertEqual(outcome["status"], client.STREAM_DEADLINE)

    def test_the_read_budget_is_sliced_to_the_budget_left(self):
        """A 3s turn budget outranks a 900s first-token budget."""
        handle = Deadline.after(3.0)
        response = type("R", (), {
            "status_code": 200, "text": "", "encoding": None,
            "iter_lines": lambda self, decode_unicode=False: iter([]),
            "close": lambda self: None,
        })()
        with mock.patch.object(client._session, "post",
                               return_value=response) as post:
            list(client.ask_openai_compat_stream(
                [{"role": "user", "content": "hi"}], model="m",
                base_url="http://127.0.0.1:1", api_key="k",
                deadline=handle, first_token_timeout=900.0))

        timeout = post.call_args.kwargs["timeout"]
        self.assertLessEqual(timeout[1], 3.0,
                             "the caller's budget must cap the first token")
        self.assertGreater(timeout[1], 0.0)

    def test_the_loop_stops_between_chunks_when_the_budget_runs_out(self):
        response = type("R", (), {
            "status_code": 200, "text": "", "encoding": None,
            "close": lambda self: None,
            "iter_lines": lambda self, decode_unicode=False: iter([
                'data: {"choices":[{"delta":{"content":"a"}}]}',
                'data: {"choices":[{"delta":{"content":"b"}}]}',
            ]),
        })()
        handle = Deadline.after(0.05)
        outcome = client.new_stream_outcome()

        def slow(*_args, **_kwargs):
            time.sleep(0.06)          # the budget expires mid-stream
            return response

        with mock.patch.object(client._session, "post", side_effect=slow):
            deltas = list(client.ask_openai_compat_stream(
                [{"role": "user", "content": "hi"}], model="m",
                base_url="http://127.0.0.1:1", api_key="k",
                deadline=handle, outcome=outcome))

        self.assertEqual(deltas, [])
        self.assertEqual(outcome["status"], client.STREAM_DEADLINE)


class VoiceStreamTests(unittest.TestCase):
    """The voice worker's OWN read of the backend stream is bounded too."""

    def setUp(self):
        from backend import voice_mode
        self.voice = voice_mode
        self.addCleanup(self.voice.TURNS.cancel_current, "test cleanup")

    def _response(self, frames, on_first=None):
        """A stand-in for the backend's SSE response object."""

        class _Response:
            def __init__(self):
                self.closed = False

            def __iter__(self):
                for index, frame in enumerate(frames):
                    if index == 1 and on_first is not None:
                        on_first()
                    yield ("data: " + json.dumps(frame) + "\n\n").encode("utf-8")

            def close(self):
                self.closed = True

        return _Response()

    def _new_id(self):
        return "req-" + uuid.uuid4().hex[:8]

    def _turn(self, request_id):
        """Start a turn owned by *request_id*, and always reset the manager."""
        self.voice.TURNS.start(request_id, None)
        self.addCleanup(self.voice.TURNS.clear, request_id)

    def _run(self, response, request_id=None, sink=None):
        request_id = request_id or self._new_id()
        captured = {}

        def _urlopen(request, timeout=None):
            captured["timeout"] = timeout
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return response

        with mock.patch.object(self.voice, "urlopen", side_effect=_urlopen), \
             mock.patch.object(self.voice, "_cancel_backend_request_async"):
            reply = self.voice._ask_backend(
                "hi", request_id=request_id, stream_sink=sink,
                timeout=600)
        return reply, captured

    def test_the_read_budget_is_the_idle_budget_not_the_turn_budget(self):
        """600s of turn budget must not become 600s of blocked read."""
        response = self._response([{"type": "completed", "reply": "answer"}])
        reply, captured = self._run(response)

        self.assertEqual(reply, "answer")
        self.assertEqual(captured["timeout"],
                         self.voice._STREAM_READ_TIMEOUT_SECONDS)
        self.assertLess(captured["timeout"], 60.0)

    def test_a_barge_in_stops_the_read_at_the_next_line(self):
        """The OLD loop only noticed a barge-in at the next delta."""
        request_id = self._new_id()
        spoken = []
        response = self._response(
            [{"type": "delta", "text": "one", "seq": 0},
             {"type": "delta", "text": "two", "seq": 1},
             {"type": "completed", "reply": "one two", "seq": 2}],
            on_first=lambda: self.voice.TURNS.cancel_current("barge-in"),
        )
        self._turn(request_id)

        self._run(response, request_id=request_id, sink=spoken.append)

        self.assertEqual(spoken, ["one"],
                         "audio after the barge-in must not be spoken")

    def test_cancelling_the_turn_closes_the_reading_socket(self):
        """A barge-in must release a read that is already blocked."""
        request_id = self._new_id()
        response = self._response([{"type": "delta", "text": "x", "seq": 0}])
        self._turn(request_id)
        self.assertTrue(self.voice.TURNS.register_stream(request_id, response),
                        "the response should be adopted by its own turn")

        self.voice.TURNS.cancel_current("barge-in")

        self.assertTrue(response.closed,
                        "the blocked read was never released")

    def test_a_stream_registered_after_the_barge_in_is_closed_at_once(self):
        request_id = self._new_id()
        response = self._response([{"type": "delta", "text": "late", "seq": 0}])
        self._turn(request_id)
        self.voice.TURNS.cancel_current("barge-in")

        kept = self.voice.TURNS.register_stream(request_id, response)

        self.assertFalse(kept, "a cancelled turn must not adopt a stream")
        self.assertTrue(response.closed)

    def test_a_new_turn_releases_the_previous_turns_stream(self):
        first_id, second_id = self._new_id(), self._new_id()
        first = self._response([{"type": "delta", "text": "old", "seq": 0}])
        self._turn(first_id)
        self.voice.TURNS.register_stream(first_id, first)

        self._turn(second_id)   # pre-empts first_id

        self.assertTrue(first.closed,
                        "the replaced turn's socket was left open")

    def test_the_reader_closes_its_own_stream_when_it_finishes(self):
        response = self._response([{"type": "completed", "reply": "done"}])
        self._run(response)

        self.assertTrue(response.closed,
                        "a finished read must close its response")

    def test_a_loopback_call_outside_a_turn_still_reads(self):
        """A direct caller the turn manager never saw must not be cut off."""
        spoken = []
        response = self._response([{"type": "delta", "text": "hi", "seq": 0},
                                   {"type": "completed", "reply": "hi"}])

        reply, _ = self._run(response, request_id=self._new_id(),
                             sink=spoken.append)

        self.assertEqual(spoken, ["hi"])
        self.assertEqual(reply, "hi")

    def test_a_wedged_read_becomes_a_dropped_connection_not_an_exception(self):
        """A read timeout is a reconnect signal, not a traceback."""
        class _Wedged:
            def __iter__(self):
                yield b'data: {"type":"delta","text":"part","seq":0}\n\n'
                raise OSError("read timed out")

            def close(self):
                pass

        attempts = {"n": 0}

        def _urlopen(request, timeout=None):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return _Wedged()
            # The resumed attempt: the backend does not resend seq 0.
            return self._response([{"type": "completed", "reply": "part"}])

        spoken = []
        with mock.patch.object(self.voice, "urlopen", side_effect=_urlopen), \
             mock.patch.object(self.voice, "_cancel_backend_request_async"):
            reply = self.voice._ask_backend("hi", request_id=self._new_id(),
                                            stream_sink=spoken.append)

        self.assertEqual(spoken, ["part"],
                         "what was already spoken must not repeat")
        self.assertEqual(reply, "part",
                         "the resume must deliver the authoritative reply")
        self.assertEqual(attempts["n"], 2)


class JobTokenCloserTests(unittest.TestCase):
    """The F20 job token is a P0-08 cancel handle: it closes what it cancels."""

    def test_cancelling_a_job_closes_the_registered_stream(self):
        job = job_registry.new_job(kind="request", label="turn")
        self.addCleanup(job.finish)
        closed = []

        job.register_closer(lambda: closed.append("closed"))
        job.cancel("barge-in")

        self.assertEqual(closed, ["closed"])
        self.assertTrue(job.cancelled)

    def test_a_second_cancel_does_not_re_close(self):
        job = job_registry.new_job(kind="request", label="turn")
        self.addCleanup(job.finish)
        closed = []

        job.register_closer(lambda: closed.append("closed"))
        job.cancel()
        job.cancel()

        self.assertEqual(closed, ["closed"])

    def test_a_late_registration_is_closed_immediately(self):
        """A stream registered after the cancel must not outlive it."""
        job = job_registry.new_job(kind="request", label="turn")
        self.addCleanup(job.finish)
        job.cancel()

        closed = []
        job.register_closer(lambda: closed.append("closed"))

        self.assertEqual(closed, ["closed"])

    def test_a_failing_closer_never_breaks_cancellation(self):
        job = job_registry.new_job(kind="request", label="turn")
        self.addCleanup(job.finish)
        survived = []

        def boom():
            raise RuntimeError("closer blew up")

        job.register_closer(boom)
        job.register_closer(lambda: survived.append("ok"))
        job.cancel()          # must not raise

        self.assertEqual(survived, ["ok"])

    def test_unregistering_leaves_nothing_to_close(self):
        job = job_registry.new_job(kind="request", label="turn")
        self.addCleanup(job.finish)
        closed = []

        def closer():
            closed.append("closed")

        job.register_closer(closer)
        job.unregister_closer(closer)
        job.cancel()

        self.assertEqual(closed, [])

    def test_the_client_registers_through_the_handle(self):
        """The client asks the handle to close its response on cancel."""
        handle = _CancelHandle()
        response = type("R", (), {
            "status_code": 200, "text": "", "encoding": None,
            "iter_lines": lambda self, decode_unicode=False: iter([]),
            "close": lambda self: handle.__setattr__("response_closed", True),
        })()
        with mock.patch.object(client._session, "post", return_value=response):
            list(client.ask_openai_compat_stream(
                [{"role": "user", "content": "hi"}], model="m",
                base_url="http://127.0.0.1:1", api_key="k", cancel=handle))

        # Registered during the call and unregistered on the way out, so a
        # finished stream is never closed a second time by a later cancel.
        self.assertEqual(handle._closers, [])


if __name__ == "__main__":   # pragma: no cover
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
