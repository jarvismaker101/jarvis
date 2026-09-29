"""Execute-once request registry with reconnect-resume event buffers.

Fable-5 audit G3:
  F23 — reconnect without re-executing. The client generates a request id
        and shares it between /ask and /ask/stream. The server executes the
        message once, appends numbered events to that request's buffer, and
        lets a reconnecting client resume after its last event instead of
        running the work again (no duplicate actions, duplicate reasoning or
        competing spoken replies).
  F26 — one event protocol. Every frame is one of
        ``delta`` / ``replace`` / ``progress`` / ``completed`` /
        ``interrupted`` / ``error`` and carries ``seq`` + ``request_id``.
        ``completed`` is the terminal authority: it carries the final reply
        the UI must accept verbatim.

Pure data module: stdlib only, no backend imports (no cycles).
"""

import threading
import time
import uuid
from typing import Dict, Generator, List, Optional, Tuple

# Event type names (F26 — the single protocol).
DELTA = "delta"
REPLACE = "replace"
PROGRESS = "progress"
COMPLETED = "completed"
INTERRUPTED = "interrupted"
ERROR = "error"

TERMINAL_TYPES = frozenset((COMPLETED, INTERRUPTED, ERROR))

#: How long a finished request stays resumable.
REQUEST_TTL = 300.0
#: Per-request buffer bound — a disconnected client can never grow memory
#: unboundedly; old frames are dropped (the terminal frame is always kept).
EVENT_LIMIT = 500
#: Total registry bound: oldest finished requests are reaped first.
MAX_REQUESTS = 50
#: When the stream is quiet this long, a heartbeat frame is emitted so the
#: client (and any proxies) can tell a slow task from a dead connection.
HEARTBEAT_SECONDS = 4.0


def new_request_id():
    return "req-" + uuid.uuid4().hex[:12]


def _payload_signature(message):
    """A stable fingerprint of a request payload (F23).

    Two transports that share a request id must be running the SAME message.
    A retried POST with a different message under a live id used to silently
    attach to the other request's result, so the user got the answer to a
    question they never asked (and the first message was never run).
    """
    import hashlib

    return hashlib.sha256(
        (message or "").strip().encode("utf-8", "replace")).hexdigest()


class RequestState:
    """One executed (or in-flight) request and its numbered event buffer."""

    def __init__(self, request_id, message):
        self.request_id = request_id
        self.message = message
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.events: List[Tuple[int, dict]] = []
        self.cond = threading.Condition()
        self.started = False          # execute-once guard
        self.done = False             # a terminal frame has been appended
        self.reply = None             # terminal reply text (completed)
        self._dropped = 0
        #: F20: set once the request has been interrupted. An interruption is
        #: final — nothing a late worker produces may reach the client.
        self._interrupted = False
        #: F23: an INDEPENDENT monotonic sequence. Deriving the next seq from
        #: ``len(self.events)`` stopped advancing once the buffer trimmed, so
        #: post-trim frames reused sequence numbers and a reconnecting client
        #: could be handed the wrong frame (or none).
        self._next_seq = 0
        #: F23: the accumulated final text, so a client whose events were
        #: trimmed away can be handed a reconstructive snapshot instead of
        #: silently losing the reply it had already started rendering.
        self.text = ""
        #: F23: the payload this id was admitted with; a different payload
        #: under the same id is a conflict, not a resume.
        self.payload_signature = _payload_signature(message)
        #: F20: the job producing this stream, so interrupting the request also
        #: cancels the worker — otherwise the client saw INTERRUPTED while the
        #: abandoned worker kept burning model calls and tool effects.
        self.job = None

    # ── F20: worker ownership ─────────────────────────────────────────────
    def attach_job(self, job):
        """Bind the job that produces this stream; cancel it on interrupt."""
        with self.cond:
            self.job = job
            already_done = self.done
        if job is not None and already_done:
            # The request finished while the job was being attached: do not
            # leave an uncancellable worker behind.
            try:
                job.cancel("request already finished")
            except Exception:
                pass
        return job

    def cancel_worker(self, reason="stopped by user"):
        """Cancel the job behind this request. Returns True when one was hit."""
        with self.cond:
            job = self.job
        if job is None:
            return False
        try:
            job.cancel(reason)
            return True
        except Exception:
            return False

    # ── producer side ─────────────────────────────────────────────────────
    def append(self, payload):
        """Append one event; returns its sequence number.

        The sequence is monotonic for the lifetime of the request: trimming the
        buffer no longer resets it, so a reconnecting client's cursor can never
        alias a different frame (F23). Returns -1 when the frame was dropped.
        """
        with self.cond:
            frame = dict(payload)
            kind = frame.get("type")
            if self._interrupted and kind != INTERRUPTED:
                # F20: an interruption is terminal for good. A worker that
                # finishes late must not turn "stopped" back into "completed",
                # and must not speak another delta after the user stopped it.
                return -1
            if self.done and kind in TERMINAL_TYPES:
                # F23: terminals are immutable — the first one wins, so a late
                # completion/error cannot rewrite what the client already saw.
                return -1
            seq = self._next_seq
            self._next_seq += 1
            frame["seq"] = seq
            frame["request_id"] = self.request_id
            self.events.append((seq, frame))
            self.updated_at = time.time()
            if kind in TERMINAL_TYPES:
                self.done = True
                if kind == INTERRUPTED:
                    self._interrupted = True
                elif kind == COMPLETED:
                    self.reply = frame.get("reply")
                    if frame.get("reply"):
                        self.text = frame["reply"]
            if len(self.events) > EVENT_LIMIT:
                # The terminal frame is always the newest when one exists,
                # so trimming from the front can never lose it.
                keep = self.events[-EVENT_LIMIT:]
                self._dropped += len(self.events) - len(keep)
                self.events = keep
            self.cond.notify_all()
            return seq

    def delta(self, text):
        if text:
            with self.cond:
                self.text += text
            self.append({"type": DELTA, "text": text})

    def replace(self, text):
        """F26 — publish a full replacement for everything streamed so far.

        Used when the final answer supersedes the streamed text (the client
        must discard its accumulated deltas and show this instead).
        """
        text = text or ""
        with self.cond:
            self.text = text
        self.append({"type": REPLACE, "text": text})

    def snapshot_frame(self):
        """F23 — a reconstructive snapshot for a client whose events were
        trimmed away, so a reconnect restores the text it was rendering
        instead of leaving the reply silently truncated."""
        with self.cond:
            return {
                "type": REPLACE,
                "text": self.text or self.reply or "",
                "snapshot": True,
                "request_id": self.request_id,
                "seq": None,
                "done": self.done,
                "dropped": self._dropped,
            }

    def progress(self, message, **extra):
        payload = {"type": PROGRESS, "message": message}
        payload.update(extra)
        self.append(payload)

    def complete(self, reply):
        self.append({"type": COMPLETED, "reply": reply or ""})

    def interrupt(self, reason=""):
        # A request that already reached a terminal state keeps it: the
        # terminal is immutable, so a late "stop" cannot overwrite a reply the
        # user has already been given.
        with self.cond:
            already_done = self.done
        if not already_done:
            self.append({"type": INTERRUPTED, "reason": reason})
        # F20: the terminal frame alone does not stop the work. Cancel the job
        # that owns this stream so the abandoned worker stops calling models
        # and tools, and so "interrupted" cannot later become "completed".
        self.cancel_worker(reason or "stopped by user")

    def error(self, message):
        self.append({"type": ERROR, "error": str(message)[:500]})

    # ── consumer side ─────────────────────────────────────────────────────
    def wait_done(self, timeout):
        """Block until this request reaches a terminal frame (or *timeout*).

        [PERF] The non-streaming ``POST /ask`` path used to poll ``done`` every
        0.25s, so a reply that landed 10ms after the last check still waited
        almost a quarter second. This is an event-driven wait on the SAME
        condition the frames already notify, so a terminal frame is observed
        the moment it is appended.
        """
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self.cond:
            while not self.done:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.cond.wait(timeout=min(remaining, HEARTBEAT_SECONDS))
            return self.done

    def events_after(self, last_seq):
        """Snapshot of events with seq > last_seq (drop-aware)."""
        with self.cond:
            return [(seq, dict(frame)) for seq, frame in self.events
                    if seq > last_seq]

    def latest_seq(self):
        with self.cond:
            return self.events[-1][0] if self.events else -1

    def first_buffered_seq(self):
        with self.cond:
            return self.events[0][0] if self.events else 0

    def stream(self, last_seq=-1, heartbeat=HEARTBEAT_SECONDS, stop_event=None):
        """Yield frames newer than *last_seq*, blocking for new ones.

        Emits a lightweight ``progress`` heartbeat when the request is quiet
        for *heartbeat* seconds so stalled clients can distinguish a slow
        task from a dead connection. Ends after the terminal frame.
        """
        cursor = last_seq
        # A client that was so far behind its frames got dropped is handed a
        # reconstructive snapshot first (F23), then resumes from the oldest
        # surviving frame — text already rendered is never silently lost.
        first = self.first_buffered_seq()
        if self.events and cursor >= 0 and cursor < first - 1:
            yield self.snapshot_frame()
            cursor = first - 1
        while True:
            with self.cond:
                # Deliver what is already buffered *before* waiting — a
                # reattaching client must not sit on a heartbeat delay while
                # its backlog is sitting in the buffer.
                newer = [(seq, dict(frame)) for seq, frame in self.events
                         if seq > cursor]
                if not newer and not self.done:
                    self.cond.wait(timeout=heartbeat)
                    newer = [(seq, dict(frame)) for seq, frame in self.events
                             if seq > cursor]
                done = self.done
            if newer:
                for seq, frame in newer:
                    cursor = seq
                    yield frame
                if done:
                    return
                continue
            if done:
                return
            if stop_event is not None and stop_event.is_set():
                return
            # heartbeat: keep the wire alive during non-chat work
            yield {
                "type": PROGRESS,
                "seq": None,
                "request_id": self.request_id,
                "message": "working",
                "elapsed": round(time.time() - self.created_at, 1),
                "heartbeat": True,
            }


class _Registry:
    def __init__(self):
        self._lock = threading.Lock()
        self._requests: Dict[str, RequestState] = {}

    def _reap(self):
        now = time.time()
        stale = [rid for rid, st in self._requests.items()
                 if st.done and now - st.updated_at > REQUEST_TTL]
        for rid in stale:
            self._requests.pop(rid, None)
        while len(self._requests) > MAX_REQUESTS:
            finished = sorted(
                (st for st in self._requests.values() if st.done),
                key=lambda st: st.updated_at)
            if not finished:
                # F23: saturation must never forget LIVE work. The oldest
                # state used to be evicted regardless of whether its worker
                # was still running: the worker kept going with no way to
                # reach it, and a reconnect re-ran the message under the same
                # id. Live states are kept; the bound is a soft one and
                # ``admit`` refuses new work once every slot is live.
                break
            self._requests.pop(finished[0].request_id, None)

    def get_or_create(self, request_id, message=""):
        """Return ``(state, created)`` — created True means caller executes.

        Kept for callers that do not need conflict reporting; a payload that
        clashes with the id is still caught by :meth:`admit`.
        """
        state, created, _conflict = self.admit(request_id, message)
        return state, created

    def admit(self, request_id, message=""):
        """Atomically admit a request payload (F23).

        Returns ``(state, created, conflict)``. ``conflict`` is a state whose
        payload differs from the one this id was admitted with — the caller
        must answer 409 instead of executing or attaching.
        """
        request_id = (request_id or "").strip() or new_request_id()
        signature = _payload_signature(message)
        with self._lock:
            self._reap()
            state = self._requests.get(request_id)
            if state is not None:
                if not getattr(state, "payload_signature", signature) == signature:
                    return state, False, True
                return state, False, False
            if len(self._requests) >= MAX_REQUESTS and not any(
                    st.done for st in self._requests.values()):
                return None, False, False
            state = RequestState(request_id, message)
            self._requests[request_id] = state
            return state, True, False

    def live_count(self):
        with self._lock:
            return sum(1 for st in self._requests.values() if not st.done)

    def get(self, request_id):
        with self._lock:
            return self._requests.get((request_id or "").strip())

    def try_start(self, state):
        """Execute-once guard: True for exactly one caller per request."""
        with self._lock:
            if state.started:
                return False
            state.started = True
            return True

    def state_snapshot(self, request_id):
        state = self.get(request_id)
        if state is None:
            return None
        with state.cond:
            return {
                "request_id": state.request_id,
                "started": state.started,
                "done": state.done,
                "reply": state.reply,
                "events": len(state.events),
                "elapsed": round(time.time() - state.created_at, 1),
            }

    def interrupt_active(self, reason="stopped by user"):
        """Append the INTERRUPTED terminal frame to every unfinished request.

        Called from /task/stop so a client still attached to a cancelled job
        gets a terminal frame instead of hanging on the wire.
        """
        interrupted = 0
        with self._lock:
            states = [st for st in self._requests.values() if not st.done]
        for state in states:
            state.interrupt(reason)
            interrupted += 1
        return interrupted


REGISTRY = _Registry()

#: Convenience module-level aliases for the routes layer.
get_or_create = REGISTRY.get_or_create
admit = REGISTRY.admit
live_count = REGISTRY.live_count
try_start = REGISTRY.try_start
get = REGISTRY.get
state_snapshot = REGISTRY.state_snapshot
interrupt_active = REGISTRY.interrupt_active
