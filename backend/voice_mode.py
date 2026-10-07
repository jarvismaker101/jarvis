import random
import datetime
import subprocess
import time
import threading
import queue
import os
import re
import glob
import json
import itertools
import uuid
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from backend import config as _config  # noqa: F401 - loads .env before service imports
from backend.config import BACKEND_PORT
from backend import listener_state
from backend.services import latency as _latency
from backend.services import listener as _listener_module
from backend.services.listener import listen, _close_microphone_source
from backend.services import local_auth
from backend.services.fish_voice import warm_up_fish_tts
from backend.services.voice import (
    StreamSpeaker,
    get_active_stream,
    pause_speaking,
    resume_local_playback,
    set_active_stream,
    speak,
    stop_speaking,
)

# ─────────────────────────────────────────────────────────────────────────────
# G11 / F50 — the voice process is an I/O WORKER, not a second brain.
#
# Every utterance is submitted to the ONE backend task runtime over the
# authenticated local HTTP contract; the backend owns conversation events,
# approvals, task checkpoints, model decisions and job cancellation. This
# process only captures audio, transcribes, submits, and speaks — and it
# PUBLISHES its real listening state to the backend (POST /voice-state/publish)
# instead of letting the UI read this process's empty listener_state copy.
# Deliberately NO import of backend.core.brain: importing it here would give
# the voice process a second, never-authoritative copy of the intelligence
# state (the exact module-copy bug the audit flagged).
# ─────────────────────────────────────────────────────────────────────────────


def _local_token():
    """Per-launch local command token injected by the supervisor."""
    return os.getenv("JARVIS_LOCAL_TOKEN", "")


#: [P1-13] Per-read (per-chunk) IDLE budget for the backend SSE stream.
#: The backend heartbeats every ~4s while a turn is quiet, so a reading that
#: goes silent for this long means the stream is dead and the reader must give
#: up rather than park forever. The caller's ``timeout`` still caps the whole
#: attempt, and a barge-in closes the socket outright.
_STREAM_READ_TIMEOUT_SECONDS = 15.0


def _backend_headers():
    """The ONE authenticated header set for backend calls (F51).

    Delegates to ``local_auth.auth_headers`` so the header name/shape lives
    in exactly one place; auth now FAILS CLOSED outside explicit dev mode, so
    every non-public backend call from this worker must carry it.
    """
    return local_auth.auth_headers()


def _post_backend(path, payload, timeout=2.5):
    """POST a control command to the backend; returns (ok, reply)."""
    try:
        request = Request(
            f"http://127.0.0.1:{BACKEND_PORT}{path}",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers=_backend_headers(),
        )
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
        try:
            return True, json.loads(body)
        except Exception:
            return True, {}
    except Exception as exc:
        print(f"[VOICE→BACKEND] {path} failed: {exc}")
        return False, None


def _get_backend(path, timeout=1.0):
    """GET a backend endpoint, authenticated (F51).

    Non-public reads are behind the launch token now: the task-mute poll
    (``/ui-state``) silently returned None without it, which made
    ``backend_task_running`` permanently False.
    """
    try:
        request = Request(
            f"http://127.0.0.1:{BACKEND_PORT}{path}",
            method="GET",
            headers=_backend_headers(),
        )
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


# ── Published task state (replaces the dead module-copy flag) ───────────────
_TASK_STATE_POLL_S = 1.0
_task_running_last_known = False
_task_running_checked_at = 0.0


def backend_task_running():
    """Is a browser/task job running — as the BACKEND reports it?

    The voice process must never read a module copy of this flag: the task
    runs in the backend process. Polled (1s TTL) from the authoritative
    /ui-state; on request failure the last known value is kept.
    """
    global _task_running_last_known, _task_running_checked_at
    now = time.monotonic()
    if now - _task_running_checked_at >= _TASK_STATE_POLL_S:
        _task_running_checked_at = now
        data = _get_backend("/ui-state")
        if isinstance(data, dict) and "task_running" in data:
            _task_running_last_known = bool(data.get("task_running"))
    return _task_running_last_known


#: [S19] The push channel's read timeout must exceed the server heartbeat
#: (15s) so a quiet-but-alive stream is never mistaken for a dead one.
_EVENTS_READ_TIMEOUT_S = 30.0


def _apply_pushed_state(evt):
    """[S19] Fold a pushed state event into the two poll caches.

    The event is the backend's authoritative truth the moment it flips, so the
    caches are updated AND their poll TTLs refreshed — the next 1s poll would
    only re-read the same value. If the push channel is down, the polls keep
    working as before; nothing breaks.
    """
    global _task_running_last_known, _task_running_checked_at
    global _voice_flag_last_known, _voice_flag_checked_at
    if not isinstance(evt, dict):
        return
    if "task_running" in evt:
        _task_running_last_known = bool(evt.get("task_running"))
        _task_running_checked_at = time.monotonic()
    if "voice_input_enabled" in evt:
        _voice_flag_last_known = bool(evt.get("voice_input_enabled", True))
        _voice_flag_checked_at = time.monotonic()


def _task_state_push_loop(stop_event=None):
    """[S19] Subscribe to the backend's /events channel and stay subscribed.

    Turns the task-mute and voice-input polls from "up to a second stale" into
    instant. Reconnects with a short backoff on any error so a dropped channel
    never wedges the worker (the polls remain the fallback).
    """
    backoff = 0.5
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        try:
            request = Request(
                f"http://127.0.0.1:{BACKEND_PORT}/events",
                method="GET",
                headers=_backend_headers(),
            )
            with urlopen(request, timeout=_EVENTS_READ_TIMEOUT_S) as response:
                backoff = 0.5
                for raw in response:
                    if stop_event is not None and stop_event.is_set():
                        return
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        evt = json.loads(line[5:].strip())
                    except Exception:
                        continue
                    _apply_pushed_state(evt)
        except Exception:
            pass
        if stop_event is not None and stop_event.is_set():
            return
        time.sleep(backoff)
        backoff = min(backoff * 2, 8.0)


# ── Voice-state publisher (F50: publish, don't expose a module copy) ────────
_voice_state_seq = 0

#: [P1-15] A per-LAUNCH identity for this worker's state stream. The sequence
#: counter restarts at 1 on every launch, so the backend cannot compare a new
#: worker's numbers with the previous worker's high-water mark — it keyed on a
#: bare integer, discarded everything from a restarted worker as "old", and the
#: UI showed a frozen voice state until the counter climbed back past it. A
#: fresh id makes the backend reset the baseline instead. (A pid alone is not
#: enough: pids are recycled.)
_VOICE_LAUNCH_ID = uuid.uuid4().hex

#: [P1-15] Set whenever the listening state CHANGES (see listener_state's state
#: hooks). The publisher waits on it, so a transition reaches the backend in
#: milliseconds instead of up to a second — and the hook itself only sets an
#: event, so nothing blocks the capture thread.
_VOICE_STATE_DIRTY = threading.Event()
#: The keep-alive cadence: still published every second when nothing changes, so
#: a dead worker remains detectable.
_VOICE_STATE_HEARTBEAT_SECONDS = 1.0

#: [PERF] P1-19 — the voice turn whose marks are still being shipped to the
#: backend. The submission carries the capture/STT marks; the playback
#: boundaries land AFTER it (playback is still running when the request
#: returns), so they are shipped on the publisher's existing ~1s cadence.
_latency_ship = {"turn": None, "request_id": "", "until": 0.0}
#: How long a finished turn may keep collecting late marks. Bounded, so a turn
#: can never hold the shipping slot forever (a barge-in may mean
#: ``playback_started`` never arrives at all).
LATENCY_TURN_GRACE_SECONDS = 8.0


def _flush_turn_marks(turn, request_id):
    """[PERF] P1-19 — ship *turn*'s not-yet-sent marks. True when drained."""
    pending = turn.pending()
    if not pending:
        return True
    if not request_id:
        return False
    ok, _ = _post_backend("/latency/client", {
        "request_id": request_id,
        "marks": pending,
        "client_now_ns": _latency.local_now_ns(),
    }, timeout=1.5)
    if ok:
        turn.ack(len(pending))
    return not turn.pending()


def _watch_turn_for_shipping(turn, request_id):
    """[PERF] P1-19 — hand *turn* to the publisher for staggered shipping.

    Any previous turn gets one last flush first: its playback marks are worth
    keeping even when a new utterance starts before its grace period ended.
    """
    previous = _latency_ship["turn"]
    if previous is not None and previous is not turn:
        try:
            _flush_turn_marks(previous, _latency_ship["request_id"])
        except Exception:
            pass
    _latency_ship["turn"] = turn
    _latency_ship["request_id"] = request_id
    _latency_ship["until"] = time.monotonic() + LATENCY_TURN_GRACE_SECONDS


def _ship_turn_marks():
    """[PERF] P1-19 — ship the current turn's marks and then let it go.

    Runs on the state publisher's cadence, so the voice worker needs no thread
    of its own for telemetry. The FIRST batch rode the /ask/stream submission
    under the same ``request_id`` — never a second id, which would describe a
    different turn.
    """
    turn = _latency_ship["turn"]
    if turn is None:
        return
    drained = _flush_turn_marks(turn, _latency_ship["request_id"])
    if drained and time.monotonic() >= _latency_ship["until"]:
        _latency_ship["turn"] = None
        _latency_ship["request_id"] = ""
        if _latency.local_turn() is turn:
            _latency.set_local_turn(None)


def _on_voice_state_change():
    """[P1-15] listener_state hook: wake the publisher NOW.

    Runs on whatever thread changed the state — usually the capture thread —
    so it must never do more than set an event. Publishing happens on the
    publisher thread (see :func:`_publish_voice_state_loop`), which keeps the
    HTTP round-trip off the audio path.
    """
    try:
        _VOICE_STATE_DIRTY.set()
    except Exception:
        pass


def _publish_voice_state_loop(stop_event=None):
    """Publish this worker's real listening state to the backend.

    [P1-15] A state CHANGE is published immediately (the hook above wakes this
    loop), and the 1s tick remains as a keep-alive so a dead worker is still
    detectable. Transitions used to wait for the tick, so the UI indicator
    lagged real state by up to a second.

    [PERF] P1-19: the same cadence ships the current turn's late latency marks,
    and the snapshot carries the listener's AEC error count — that counter lives
    in THIS process (which owns the microphone), so the backend's /latency
    endpoint reads it from here instead of importing the listener module.

    *stop_event* exists for tests (production runs it forever).
    """
    global _voice_state_seq
    listener_state.register_state_hook(_on_voice_state_change)
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        # Clear BEFORE publishing: a change that lands during the POST then
        # wakes the next iteration immediately instead of being lost.
        _VOICE_STATE_DIRTY.clear()
        try:
            snapshot = listener_state.get_voice_state()
            _voice_state_seq += 1
            snapshot["state_seq"] = _voice_state_seq
            snapshot["publisher_pid"] = os.getpid()
            # [P1-15] per-launch identity, so a RESTARTED worker is not judged
            # against the previous worker's high-water mark.
            snapshot["publisher_id"] = _VOICE_LAUNCH_ID
            snapshot["aec_errors"] = int(
                getattr(_listener_module, "_aec_error_count", 0) or 0)
            _post_backend("/voice-state/publish", snapshot, timeout=1.0)
            _ship_turn_marks()
        except Exception:
            pass
        _VOICE_STATE_DIRTY.wait(_VOICE_STATE_HEARTBEAT_SECONDS)


# ── Backend submission (F50: the utterance goes to the ONE runtime) ─────────
def _ask_backend(text, request_id, stream_sink=None, replace_sink=None,
                 timeout=600, client_marks=None):
    """Submit one utterance to the backend task runtime; return the reply.

    Prefers the SSE /ask/stream contract (F23/F26): deltas are fed to
    ``stream_sink`` (the voice speaker) exactly like the old in-process
    ``process_message(stream_reply=...)`` did, and the terminal frame
    carries the authoritative reply. ``speak=False`` keeps the backend
    silent — playback ownership stays with this I/O worker.

    F23: the worker carries a cursor (``last_event_id``) and resumes from it
    when a stream drops, so a reconnect restores the rest of the reply WITHOUT
    replaying deltas it already spoke — the old code always re-attached from
    the beginning, so every network hiccup repeated the answer out loud.
    ``replace``/snapshot frames are honoured: only text that EXTENDS what has
    already been spoken is fed to the sink (audio that has been played cannot
    be un-played), and ``replace_sink`` is told about the authoritative text
    when the caller can re-render it.

    [PERF] P1-19: ``client_marks`` are this process's already-stamped turn
    marks (``[name, perf_counter_ns, meta]``). They ride THIS submission, under
    the SAME ``request_id``, so the backend's record and the voice process's
    marks describe one turn instead of two disconnected halves.
    """
    payload = {
        "message": text,
        "request_id": request_id,
        "speak": False,
        "origin": "voice",
    }
    headers = _backend_headers()
    marks_batch = [list(mark) for mark in (client_marks or ())]

    def _attempt_stream():
        """One stream attempt starting at the current cursor.

        Returns ``(reply, finished, cursor)``: *finished* True means the
        terminal frame was reached (or the request failed permanently).

        [P1-13] Two bounds replace what used to be an uninterruptible read:
        the socket read timeout is a small per-chunk IDLE budget (the backend
        heartbeats every ~4s, so 15s of silence means the stream is dead), and
        the whole attempt is capped by the caller's ``timeout``. The response is
        registered with the turn manager, so a barge-in closes it and the read
        raises at once instead of waiting out either bound.
        """
        cursor_local = payload.get("last_event_id", -1)
        spoken_local = payload.get("_spoken", "")
        request = Request(
            f"http://127.0.0.1:{BACKEND_PORT}/ask/stream",
            data=json.dumps({
                "message": text,
                "request_id": request_id,
                "speak": False,
                "origin": "voice",
                "last_event_id": cursor_local,
                # [PERF] P1-19 — the alignment reference is sampled HERE, at
                # send time, so a retried attempt re-aligns instead of reusing a
                # stale clock.
                "client_marks": marks_batch,
                "client_now_ns": _latency.local_now_ns(),
            }).encode("utf-8"),
            method="POST",
            headers=headers,
        )
        terminal_reply = ""
        attempt_deadline = time.monotonic() + float(timeout or 0)
        response = urlopen(request, timeout=_STREAM_READ_TIMEOUT_SECONDS)
        registered = False
        try:
            registered = TURNS.register_stream(request_id, response)
            if not registered and TURNS.superseded(request_id):
                # This turn was already replaced/cancelled while the request
                # was being sent: there is nothing left to read.
                return terminal_reply, True, cursor_local, spoken_local
            for raw_line in response:
                # [P1-13] cancellation is checked on EVERY line, so a coalesced
                # batch or a late reattach still ends the moment the turn is
                # gone — the old loop only noticed at the next delta.
                if TURNS.superseded(request_id):
                    return terminal_reply, True, cursor_local, spoken_local
                if time.monotonic() >= attempt_deadline:
                    print("[VOICE-BACKEND] stream attempt hit its budget")
                    return terminal_reply, True, cursor_local, spoken_local
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    frame = json.loads(line[5:].strip())
                except Exception:
                    continue
                ftype = frame.get("type")
                seq = frame.get("seq")
                if isinstance(seq, int) and seq > cursor_local:
                    cursor_local = seq
                # F31: only FINAL-answer frames are spoken or counted as
                # spoken. A reasoning/thinking frame (explicit ``channel``, or
                # a legacy ``thought``/``reasoning`` flag) is preserved by the
                # transport but must never reach TTS or the spoken history.
                channel = frame.get("channel")
                if channel is None and (frame.get("thought")
                                        or frame.get("reasoning")):
                    channel = "reasoning"
                if ftype == "delta":
                    if channel not in (None, "final"):
                        continue
                    text_delta = frame.get("text") or ""
                    if text_delta:
                        spoken_local += text_delta
                        if stream_sink is not None:
                            try:
                                stream_sink(text_delta)
                            except Exception:
                                pass
                elif ftype == "reasoning":
                    # A dedicated reasoning frame: never spoken.
                    continue
                elif ftype == "replace":
                    if channel not in (None, "final"):
                        continue
                    full = frame.get("text") or ""
                    if replace_sink is not None:
                        try:
                            replace_sink(full)
                        except Exception:
                            pass
                    elif stream_sink is not None and full.startswith(spoken_local):
                        # A pure extension: speak only the new tail. Anything
                        # else would repeat audio the user already heard.
                        tail = full[len(spoken_local):]
                        if tail:
                            try:
                                stream_sink(tail)
                            except Exception:
                                pass
                    spoken_local = full or spoken_local
                elif ftype == "completed":
                    terminal_reply = frame.get("reply") or terminal_reply
                    return terminal_reply, True, cursor_local, spoken_local
                elif ftype == "interrupted":
                    return frame.get("reply") or terminal_reply, True, \
                        cursor_local, spoken_local
                elif ftype == "error":
                    return ("Error: %s" % (frame.get("error") or "unknown error"),
                            True, cursor_local, spoken_local)
            # The stream ended without a terminal frame: a dropped connection.
            return terminal_reply, False, cursor_local, spoken_local
        except Exception as exc:
            # [P1-13] a socket timeout / reset is a DROPPED CONNECTION, not an
            # exception through the caller: the retry loop above decides whether
            # to reconnect from the cursor or fall back to /ask.
            print("[VOICE-BACKEND] stream read failed: %s" % exc)
            return terminal_reply, False, cursor_local, spoken_local
        finally:
            TURNS.unregister_stream(request_id, response)
            _close_quietly(response)

    last_error = None
    for attempt in range(3):
        try:
            reply, finished, cursor, spoken = _attempt_stream()
        except Exception as exc:
            last_error = exc
            print("[VOICE→BACKEND] stream submit failed: %s" % exc)
            if attempt == 0:
                # Nothing has been spoken yet: this is a plain connection
                # failure, so fall through to the non-stream endpoint.
                break
            time.sleep(0.2)
            payload["last_event_id"] = payload.get("last_event_id", -1)
            continue
        payload["last_event_id"] = cursor
        payload["_spoken"] = spoken
        if finished:
            return reply or None
        # Reconnect from the cursor: the resumed stream skips everything the
        # speaker already heard.
        time.sleep(0.2)

    # Non-stream fallback: same runtime, one JSON round-trip.
    try:
        request = Request(
            f"http://127.0.0.1:{BACKEND_PORT}/ask",
            data=json.dumps({**payload,
                             "client_marks": marks_batch,
                             "client_now_ns": _latency.local_now_ns()
                             }).encode("utf-8"),
            method="POST",
            headers=headers,
        )
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
        return data.get("reply") or None
    except Exception as exc2:
        print("[VOICE→BACKEND] ask fallback failed: %s" % exc2)
        return None

# ─────────────────────────────────────────
# P0-08 — PRE-EMPTIVE TURN MANAGER
# ─────────────────────────────────────────
# The voice process stays a strict I/O worker (F50): it speaks, the backend owns
# the turn. But every interruption used to WAIT for the old generation to finish,
# because nothing cancelled it — "actually, what about X?" said in the middle of
# a reply queued behind the very answer it was interrupting (typically 1-5s, far
# longer for a tool, screen or research turn), and a non-streamed reply even
# voiced the OLD answer first.
#
# This manager owns the ONE active voice turn. A new utterance — or a barge-in
# onset — tears the previous turn down pre-emptively: its local audio is closed
# at once, its backend request gets an INTERRUPTED terminal frame (so the SSE
# reader exits instead of blocking on a dead stream) and its worker job is
# cancelled. The new turn is submitted without waiting for any of that to unwind.
#
# What it deliberately does NOT do: rewrite history. The interrupted turn's FULL
# generated reply is still committed by the backend (brain.handle_chat), because
# the user wants to read what they missed. Cancelling stops the *audio* and the
# *pending tool work* (F20 checkpoints); the stored reply text is untouched.
_TURN_CANCEL_TIMEOUT = 1.0

#: P0-08 — per-process turn counter. The previous id was
#: ``voice-<epoch-ms>-<pid>``, which COLLIDES when two utterances are submitted
#: inside the same millisecond. That was mostly theoretical before, but
#: pre-emptive dispatch submits the interrupting utterance immediately, and a
#: reused id is not merely cosmetic: the backend request registry treats a
#: reused id carrying a different message as a 409 conflict, so the second turn
#: would have been refused outright. The counter makes the id unique for the
#: life of the process.
_TURN_SEQ = itertools.count(1)


def _new_turn_request_id():
    """A request id that is unique even for turns submitted back to back."""
    return "voice-%s-%s-%s" % (int(time.time() * 1000), os.getpid(),
                               next(_TURN_SEQ))


def _cancel_backend_request(request_id, reason="cancelled by voice barge-in"):
    """POST /ask/cancel/{request_id}. Best-effort, never raises.

    Returns True when the backend reported that it actually cancelled something.
    Idempotent by construction: the endpoint answers ``cancelled: False`` with a
    reason (``unknown_request`` / ``already_finished`` / ``already_interrupted``)
    for every case where there is nothing left to cancel, so this caller never
    has to check first and can never flip a completed turn to interrupted.
    """
    if not request_id:
        return False
    url = ("http://127.0.0.1:%s/ask/cancel/%s?reason=%s"
           % (BACKEND_PORT, quote(str(request_id), safe=""),
              quote(str(reason), safe="")))
    try:
        request = Request(url, method="POST", headers=_backend_headers())
        with urlopen(request, timeout=_TURN_CANCEL_TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8", errors="replace"))
        return bool(body.get("cancelled"))
    except Exception as exc:
        print("[VOICE→BACKEND] cancel failed for %s: %s" % (request_id, exc))
        return False


def _cancel_backend_request_async(request_id, reason):
    """Fire the cancel on a daemon thread.

    Barge-in onset runs on the real-time capture thread, so the network call
    must never happen inline (P1-03 took the blocking calls out of that loop and
    this must not put one back). Only the thread spawn is synchronous; it costs
    a few microseconds.
    """
    try:
        threading.Thread(
            target=_cancel_backend_request,
            args=(request_id, reason),
            name="voice-turn-cancel",
            daemon=True,
        ).start()
        return True
    except Exception:
        return False


def _close_quietly(stream):
    """[P1-13] Close an HTTP response/stream. Never raises.

    Closing is what releases a reader blocked in ``recv``: a cancellation flag
    is only ever consulted BETWEEN lines, so the socket itself has to be
    closed from the cancelling thread.
    """
    if stream is None:
        return False
    close = getattr(stream, "close", None)
    if not callable(close):
        return False
    try:
        close()
        return True
    except Exception:
        return False


class _TurnManager:
    """One active voice turn, replaceable pre-emptively (P0-08).

    Invariants this class owns:

    * at most ONE current turn, so two turns can never speak at once;
    * a cancelled/replaced turn is no longer *current*, so its late reply is
      dropped instead of being voiced over the newer answer;
    * cancellation is request-scoped, idempotent and non-raising;
    * nothing here blocks the caller on the network.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._request_id = ""
        self._speaker = None
        self._started_at = 0.0
        #: [P1-13] the in-flight backend SSE response of the current turn.
        #: Cancelling CLOSES it, so a barge-in releases a blocked socket read
        #: instead of leaving the reader thread parked until the read timeout.
        self._stream = None
        #: The id of the turn most recently cancelled, remembered so a stream
        #: registered AFTER the barge-in is still refused (and closed).
        self._cancelled_id = ""
        self.stats = {
            "started": 0,
            "preempted": 0,
            "barge_in_cancels": 0,
            "stale_replies_dropped": 0,
        }

    # -- lifecycle ---------------------------------------------------------

    def start(self, request_id, speaker):
        """Register a new turn, pre-empting the previous one. Returns the id."""
        with self._lock:
            previous = self._request_id
            previous_speaker = self._speaker
            previous_stream = self._stream
            self._request_id = request_id
            self._speaker = speaker
            self._stream = None
            # A new turn clears the "recently cancelled" mark: a fresh turn may
            # legitimately reuse a request id.
            self._cancelled_id = ""
            self._started_at = time.monotonic()
            self.stats["started"] += 1
            # Pre-empt on a NEW SPEAKER as well as a new id: the identity that
            # matters is the one that owns playback, so an id collision (or a
            # caller reusing an id) can never leave two turns registered.
            replaced = bool(previous) and (
                previous != request_id or previous_speaker is not speaker)
            if replaced:
                self.stats["preempted"] += 1
        if replaced:
            # [P1-13] The replaced turn's reader is still parked on its own
            # socket: closing it here is what actually frees that thread.
            _close_quietly(previous_stream)
        if replaced:
            # Outside the lock: this closes a speaker and fires a network call.
            self._teardown(previous, previous_speaker,
                           "replaced by a newer utterance")
        return request_id

    def is_current(self, request_id):
        """True while *request_id* is still the turn that owns playback."""
        with self._lock:
            return bool(request_id) and request_id == self._request_id

    def superseded(self, request_id):
        """True when another turn took over (or cancelled) *request_id*.

        [P1-13] A caller the manager has never seen (a direct ``_ask_backend``
        call outside a voice turn) is NOT superseded: nothing could have
        replaced it, so its stream must keep reading.
        """
        with self._lock:
            if request_id and request_id == self._cancelled_id:
                return True
            return bool(self._request_id) and self._request_id != request_id

    def clear(self, request_id):
        """Release the turn if it is still current. Returns True when it was."""
        with self._lock:
            if request_id and request_id == self._cancelled_id:
                # [P1-13] The turn has fully ended, so its cancellation no
                # longer needs to be remembered: a later turn may legitimately
                # reuse the id.
                self._cancelled_id = ""
            if self._request_id != request_id:
                return False
            self._request_id = ""
            self._speaker = None
            self._stream = None
            return True

    # -- [P1-13] the turn's in-flight stream ---------------------------------

    def register_stream(self, request_id, stream):
        """Attach the SSE response this turn is reading. Returns True if kept.

        A stream belonging to a CANCELLED or superseded turn is closed
        immediately and never adopted, so a late registration cannot outlive
        the barge-in that cancelled the turn it belongs to.
        """
        if stream is None:
            return False
        with self._lock:
            if request_id and request_id == self._cancelled_id:
                stale = True
            elif self._request_id and self._request_id != request_id:
                stale = True
            else:
                self._stream = stream
                stale = False
        if stale:
            _close_quietly(stream)
            return False
        return True

    def unregister_stream(self, request_id, stream):
        with self._lock:
            if self._stream is stream:
                self._stream = None

    def close_stream(self):
        """Close the current turn's stream, if any. Never raises."""
        with self._lock:
            stream = self._stream
            self._stream = None
        _close_quietly(stream)

    def cancel_current(self, reason="barge-in"):
        """Cancel the active turn. Never raises, never blocks.

        Called from the barge-in observer, i.e. on the capture thread, so the
        local teardown is synchronous (audio must die NOW) and the remote cancel
        is fire-and-forget.

        [P1-13] The turn's SSE response is closed here too: the backend's
        INTERRUPTED frame is the normal exit, but if the socket is wedged the
        blocked read must be released by US rather than by waiting out its
        timeout.
        """
        with self._lock:
            request_id = self._request_id
            speaker = self._speaker
            stream = self._stream
            if request_id:
                self._request_id = ""
                self._speaker = None
                self._stream = None
                self._cancelled_id = request_id
                self.stats["barge_in_cancels"] += 1
        if not request_id:
            return False
        _close_quietly(stream)
        self._teardown(request_id, speaker, reason)
        return True

    def note_stale_reply(self):
        with self._lock:
            self.stats["stale_replies_dropped"] += 1

    def active_request_id(self):
        with self._lock:
            return self._request_id

    def snapshot(self):
        with self._lock:
            state = dict(self.stats)
            state["active_request_id"] = self._request_id
        return state

    # -- internals ---------------------------------------------------------

    def _teardown(self, request_id, speaker, reason):
        """Silence the old turn locally, then cancel it remotely."""
        # (a) local audio dies immediately — this is what the user hears
        if speaker is not None:
            try:
                speaker.close()
            except Exception:
                pass
        # (b)+(c) the backend publishes INTERRUPTED (its SSE reader exits) and
        # cancels that request's worker job. Request-scoped: nothing else is hit.
        _cancel_backend_request_async(request_id, reason)


#: The process-wide turn manager.
TURNS = _TurnManager()


def _on_barge_in():
    """[P0-08] Barge-in onset observer — cancel the turn being interrupted."""
    try:
        TURNS.cancel_current("barge-in")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# P0-12 — PRE-WARM ON VAD ONSET
# ─────────────────────────────────────────────────────────────────────────────
# VAD onset gives this process a head start of several hundred milliseconds
# before the transcript exists. The provider handshake (chat + classifier + STT)
# used to be paid at the START of the turn, i.e. on the user's clock; the
# backend's /prewarm opens it during that head start instead.
#
# Two rules, both non-negotiable:
#   * NEVER inline in the capture loop. This hook runs on the real-time capture
#     thread, so it only (maybe) spawns a daemon thread — the same shape as the
#     P0-08 cancel the onset path already fires.
#   * Rate-limited here AND in the backend. The local timestamp stops this
#     process from spawning a thread on every onset of a long utterance.
_PREWARM_REQUEST_INTERVAL_SECONDS = 20.0
_prewarm_lock = threading.Lock()
_last_prewarm_request_at = 0.0


def _request_backend_prewarm():
    """POST /prewarm once. Authenticated, best-effort, never raises."""
    url = "http://127.0.0.1:%s/prewarm" % BACKEND_PORT
    try:
        request = Request(url, data=b"{}", method="POST",
                          headers=_backend_headers())
        with urlopen(request, timeout=_TURN_CANCEL_TIMEOUT) as response:
            # Read the body fully: a half-read response holds the socket and
            # would turn the warm into a leak.
            response.read()
        return True
    except Exception as exc:
        # Advisory only: a provider being down (or the backend not up yet) is
        # not an error the user should ever see. ASCII-only so this line can
        # never itself fail on a cp1252 console.
        print("[PREWARM] skipped: %s" % exc)
        return False


def _prewarm_backend_async():
    """[P0-12] Warm the provider connections while the user is still talking.

    Called from the speech-onset hook, which runs on the real-time capture
    thread: the only synchronous work here is a timestamp check and (at most) a
    thread spawn. Never raises, never blocks.
    """
    global _last_prewarm_request_at
    try:
        now = time.monotonic()
        with _prewarm_lock:
            if (now - _last_prewarm_request_at) < _PREWARM_REQUEST_INTERVAL_SECONDS:
                return False
            _last_prewarm_request_at = now
        threading.Thread(
            target=_request_backend_prewarm,
            name="voice-prewarm",
            daemon=True,
        ).start()
        return True
    except Exception:
        return False


def _on_speech_onset():
    """[P0-12] Onset observer — the connection warm rides the barge-in signal."""
    _prewarm_backend_async()


def _interrupt_active_turn(reason="interrupted by a new utterance"):
    """Cut the active turn because the user committed a new utterance.

    [P1-06] The listener used to DISCARD a committed transcript whenever the
    "speaking" flag was set, so the user could talk during a reply and be
    ignored with no error, no log and no interruption. Speaking during a reply
    means "change the subject", i.e. an interruption: the turn manager cancels
    the turn that owns playback and the utterance is dispatched as the new
    turn.

    Idempotent and non-raising by construction — the barge-in onset hook has
    usually already cancelled the turn, in which case this simply reports that
    there was nothing active. Never blocks: the remote cancel is fire-and-forget.
    """
    try:
        return TURNS.cancel_current(reason)
    except Exception:
        return False


try:  # the listener owns WHEN a barge-in happens; this process owns the turn
    from backend.services.listener import register_barge_in_hook

    register_barge_in_hook(_on_barge_in)
    # [P0-12] the same onset signal is the head start the warm uses
    register_barge_in_hook(_on_speech_onset)
except Exception:
    pass


BOOT_RESPONSES = [
    "Good to see you back, sir. What's on your mind today?",
    "I'm here, sir. How can I assist you?",
    "At your service. What would you like to do?",
    "Ready when you are. Tell me your command.",
    "Welcome back, sir. What are we working on today?",
    "Always listening. Go ahead.",
    "Nice to have you back. What's the plan?"
]

DESKTOP = Path(os.getenv("JARVIS_DESKTOP_PATH", str(Path.home() / "Desktop")))

command_queue = queue.Queue()

# ─────────────────────────────────────────
# STOP TALKING — English + Hindi variants
# ─────────────────────────────────────────
STOP_TALKING_EN = [
    "stop speaking", "stop talking", "stop voice",
    "be quiet", "shut up", "silence"
]

STOP_TALKING_HI = [
    # chup = quiet/silent
    "chup ho jao", "chup ho ja", "chup raho",
    "chup ho", "chup kar", "chup karo",
    # band = stop/close (for voice only, not shutdown)
    "bolna band karo", "bolna band karo",
    "bol mat", "bas karo",
    # mishears
    "cup ho jao", "choop", "chup jao",
    "cheap ho", "cheap raho",
]

# ─────────────────────────────────────────
# SHUTDOWN — English + Hindi variants
# ─────────────────────────────────────────
SHUTDOWN_EN = [
    ("jarvis", "stop listening"),
    ("jarvis", "shutdown"),
    ("jarvis", "shut down"),
    ("jarvis", "stop", "listening"),
]

SHUTDOWN_HI_PHRASES = [
    # band ho = shut down
    "band ho jao", "band ho ja", "band ho",
    "band karo", "band kar do", "bandh karo",
    "sab band", "chalo band",
    # bund = mishear of band
    "bund ho", "bund karo",
    # sunna band karo = stop listening
    "sunna band", "sunna band karo",
    "mat suno", "sun mat",
]

JARVIS_VARIANTS = [
    "jarvis", "jervis", "jarvish", "harvey",
    "jarvas", "jarbus", "garvis", "jarwis",
]

# ─────────────────────────────────────────
# CONTINUE VARIANTS
# ─────────────────────────────────────────
CONTINUE_PHRASES = [
    "continue", "carry on", "go on", "keep going",
    "aage bolo", "aage boliye", "jaari rakho",
    "bolo aage", "continue karo",
]

# F35 — one exact, target-aware multilingual grammar -------------------------
# Every control below is matched as a WHOLE normalized token sequence (never a
# substring) and routed to exactly ONE owner. The old code mixed grammars:
# Hindi shutdown was substring-based ("music band karo" matched "band karo"),
# negation was handled only for the research phrases, "don't continue" still
# resumed, pause had no handler at all, speech stop depended on this process's
# own speaking flag, and shutdown called an UNAUTHENTICATED warm-stop.
#
# Targets:
#   speech   -> the active narration/TTS (backend /speak/*)
#   task     -> the running job                (backend /task/stop)
#   approval -> the armed consent gate         (backend /approvals/reset)
#   pause    -> narration, but resumable       (backend /speak/pause)
#   continue -> resume the paused narration    (backend /speak/resume)
#   sleep    -> warm sleep, UI down, backend warm (watcher /stop, authed)
#   shutdown -> full shutdown of the whole stack  (watcher /shutdown, authed)
SPEECH_STOP_EN = (
    "stop speaking", "stop the speaking", "stop talking", "stop the talking",
    "stop voice", "stop your voice", "stop the voice", "be quiet", "shut up",
    "silence", "quiet", "stop the speech", "stop speech", "enough talking",
)

SPEECH_STOP_HI = (
    "chup ho jao", "chup ho ja", "chup raho", "chup ho", "chup kar",
    "chup karo", "bolna band karo", "bolna band", "bolna chup", "bol mat",
    "bas karo", "bas", "awaz band karo", "awaaz band karo",
    # mishears of the same command
    "cup ho jao", "choop", "chup jao", "cheap ho", "cheap raho",
)

TASK_STOP_EN = (
    "stop task", "stop the task", "stop this task", "cancel task",
    "cancel the task", "abort task", "abort the task", "kill task",
    "kill the task", "stop it", "stop doing that", "stop working on it",
)

TASK_STOP_HI = (
    "kaam band karo", "kaam rok do", "kaam roko", "task band karo",
    "task rok do", "task cancel karo", "kaam cancel karo", "kaam band",
)

#: Rank 5 — the broad stop: every work job, never the chat request. A
#: separate control kind so "stop it" (ONE job) and "stop everything" (ALL
#: work jobs) can never be confused.
TASK_STOP_ALL_EN = (
    "stop everything", "stop it all", "stop all tasks", "stop all jobs",
    "stop everything now", "stop all the tasks", "stop all work",
)

APPROVAL_CANCEL_EN = (
    "cancel approval", "cancel the approval", "cancel that", "cancel that action",
    "cancel screen action", "cancel the screen action", "don't do that",
    "dont do that", "do not do that", "no don't do it", "cancel the action",
)

APPROVAL_CANCEL_HI = (
    "approval cancel karo", "manzoori cancel karo", "manjuri cancel karo",
    "wo mat karo", "ye mat karo", "aisa mat karo",
)

PAUSE_EN = (
    "pause", "pause it", "pause that", "pause the speech", "pause speaking",
    "hold on", "hold that", "wait", "wait a moment", "give me a moment",
)

PAUSE_HI = (
    "ruk jao", "ruko", "ruko zara", "pause karo", "thoda ruko", "ek minute",
    "ek second", "ruk", "zara ruko",
)

CONTINUE_HI = (
    "aage bolo", "aage boliye", "jaari rakho", "bolo aage", "continue karo",
    "aage chalo", "wapas bolo",
)

SLEEP_EN = (
    "sleep", "go to sleep", "sleep mode", "warm sleep", "standby",
    "go to standby", "take a rest",
)

SLEEP_HI = (
    "so jao", "sone jao", "aaram karo", "sleep karo", "so ja",
)

SHUTDOWN_EN_PHRASES = (
    "stop listening", "shutdown", "shut down", "shut it down", "power off",
    "turn off", "turn yourself off", "exit jarvis", "close jarvis", "quit jarvis",
)

SHUTDOWN_HI_EXACT = (
    "band ho jao", "band ho ja", "band ho", "band karo", "band kar do",
    "bandh karo", "bandh kar do", "sab band karo", "sab band", "chalo band",
    "bund ho", "bund karo", "sunna band karo", "sunna band", "mat suno",
    "sun mat", "sunna band kar do",
)

#: Negation anywhere in the phrase, or in the run-up to it, cancels the match
#: ("don't stop the research", "mat band karo", "band mat karo"). Both
#: apostrophe forms normalize to the same token because normalize strips them.
NEGATION_TOKENS = frozenset((
    "dont", "not", "never", "mat", "nahi", "nahin", "nako", "without",
))
NEGATION_WINDOW = 3

#: Trailing politeness/adverbs that do not change a control's meaning.
COMMAND_TAIL_TOKENS = frozenset((
    "now", "please", "sir", "maam", "madam", "yaar", "yarr", "zara", "ab",
    "abhi", "quickly", "immediately", "fast", "jaldi", "thoda", "just",
))

#: Classification precedence: the NON-shutdown controls win, so a phrase like
#: "bolna band karo" (stop talking) or "kaam band karo" (stop the task) can
#: never be read as shutdown just because it contains "band karo".
CONTROL_GRAMMAR = (
    ("speech_stop", (SPEECH_STOP_EN, SPEECH_STOP_HI)),
    ("task_stop", (TASK_STOP_EN, TASK_STOP_HI)),
    ("task_stop_all", (TASK_STOP_ALL_EN, ())),
    ("approval_cancel", (APPROVAL_CANCEL_EN, APPROVAL_CANCEL_HI)),
    ("pause", (PAUSE_EN, PAUSE_HI)),
    ("continue", (tuple(CONTINUE_PHRASES), CONTINUE_HI)),
    ("sleep", (SLEEP_EN, SLEEP_HI)),
    ("shutdown", (SHUTDOWN_EN_PHRASES, SHUTDOWN_HI_EXACT)),
)


#: Articles carry no meaning in a control phrase ("stop the task" == "stop task").
ARTICLE_TOKENS = frozenset(("the", "a", "an"))

#: Tokens that may remain in the utterance around a matched control phrase
#: without making it a different command.
CONTROL_NEUTRAL_TOKENS = frozenset((
    "the", "a", "an", "it", "that", "this", "my", "your", "please", "now",
    "jarvis",
))


def normalize_control_text(text):
    """Normalize an utterance for the control grammar.

    Apostrophes are removed (both ``'`` and ``’``) so "don't" and "dont" are
    the same token, the Jarvis name and other fillers are dropped, articles
    are dropped, and trailing politeness/adverbs are ignored.
    """
    cleaned = (text or "").lower().replace("\u2019", "").replace("'", "")
    words = [word for word in re.findall(r"[a-z0-9]+", cleaned)
             if word not in JARVIS_VARIANTS]
    filler = {"please", "sir", "hey", "ok", "okay", "oh"}
    while words and words[0] in filler:
        words.pop(0)
    while words and (words[-1] in filler or words[-1] in COMMAND_TAIL_TOKENS):
        words.pop()
    return [word for word in words if word not in ARTICLE_TOKENS]


def _phrase_tokens(phrase):
    return [word for word in re.findall(r"[a-z0-9]+", phrase)
            if word not in JARVIS_VARIANTS and word not in ARTICLE_TOKENS]


def _negated(tokens, start, length):
    """Is the phrase at *start* negated by the words right before it?

    Only the run-up is inspected: a negator INSIDE the matched run is part of
    the phrase's own vocabulary ("don't do that" is an approval cancel, not a
    negated cancel), while a negator before it turns the phrase into its
    opposite ("don't stop the research", "mat band karo").
    """
    return any(token in NEGATION_TOKENS
               for token in tokens[max(0, start - NEGATION_WINDOW):start])


def _match_tokens(tokens, phrases):
    """Whole-utterance match, negation-aware and target-aware.

    The phrase must appear as a contiguous token run AND everything else in
    the utterance must be neutral (a particle, not another target word): this
    is what stops "music band karo" from reading as "shut down" and
    "don't stop the research" from reading as a stop request.
    """
    for phrase in phrases:
        wanted = _phrase_tokens(phrase)
        if not wanted or len(wanted) > len(tokens):
            continue
        for start in range(0, len(tokens) - len(wanted) + 1):
            if tokens[start:start + len(wanted)] != wanted:
                continue
            if _negated(tokens, start, len(wanted)):
                continue
            rest = tokens[:start] + tokens[start + len(wanted):]
            if all(token in NEGATION_TOKENS or token in COMMAND_TAIL_TOKENS
                   or token in CONTROL_NEUTRAL_TOKENS for token in rest):
                return phrase
    return None


def classify_control(text):
    """Classify an utterance as exactly one control command (or None).

    This is the ONE grammar the voice runtime consults before any mute: an
    exact, target-aware, negation-aware multilingual match. Returns one of
    ``speech_stop``, ``task_stop``, ``approval_cancel``, ``pause``,
    ``continue``, ``sleep``, ``shutdown`` or ``None``.
    """
    tokens = normalize_control_text(text)
    if not tokens:
        return None
    for name, phrase_groups in CONTROL_GRAMMAR:
        for phrases in phrase_groups:
            if _match_tokens(tokens, phrases):
                return name
    return None


def control_owner(command):
    """Which runtime owns *command* — the audit's target-aware routing."""
    return {
        "speech_stop": "speech",
        "pause": "speech",
        "continue": "speech",
        "task_stop": "task",
        "task_stop_all": "task",
        "approval_cancel": "approval",
        "stop_research": "task",
        "sleep": "supervisor",
        "shutdown": "supervisor",
    }.get(command)

# ─────────────────────────────────────────
# NORMAL SETUP VARIANTS
# ─────────────────────────────────────────
SETUP_PHRASES = [
    "normal setup", "put my setup", "my normal setup",
    "mera setup", "setup chalu karo", "setup kholo",
]

#: Verbs a setup request may carry beside the phrase vocabulary ("put my
#: normal setup", "mera setup karo"). Stripped ONLY when the utterance does
#: not already match a setup phrase, and never added to the shared
#: neutral-token sets the F35 grammar depends on.
SETUP_VERB_TOKENS = frozenset((
    "put", "set", "lagao", "chalao", "karo", "kar", "kro", "kardo", "chalu",
    "kholo", "launch", "open", "start", "activate", "run",
))


# ─────────────────────────────────────────
# MATCHING FUNCTIONS
# ─────────────────────────────────────────
def normalize_spoken_command(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    filler_words = {"jarvis", "jervis", "jarvish", "please", "sir", "hey", "ok", "okay"}

    while words and words[0] in filler_words:
        words.pop(0)
    while words and words[-1] in filler_words:
        words.pop()

    return " ".join(words)


def is_stop_talking(text):
    return classify_control(text) == "speech_stop"


def is_pause(text):
    return classify_control(text) == "pause"


def is_sleep(text):
    return classify_control(text) == "sleep"


#: Kept as frozensets because other modules/tests import them by name; the
#: authoritative grammar is CONTROL_GRAMMAR above (one matcher, one order).
STOP_TASK_EN = frozenset(TASK_STOP_EN)
CANCEL_APPROVAL_EN = frozenset(APPROVAL_CANCEL_EN)


def is_stop_task(text):
    return classify_control(text) == "task_stop"


def is_cancel_approval(text):
    return classify_control(text) == "approval_cancel"


def _deliver_stop_task(all_jobs=False):
    """Deliver 'stop task' to the BACKEND (the one runtime that owns jobs).

    G11 / F50 — the old version called browser_agent/research/jobs module
    copies INSIDE the voice process, which never own the running task: the
    stop was a silent no-op against the real backend job. The authoritative
    kill switch is POST /task/stop (authed), which stops the browser agent,
    the research flow, interrupts live registered requests and cuts TTS.
    Rank 5: "stop everything" posts scope=all, which cancels EVERY work job
    (still never the chat request).
    """
    try:
        stop_speaking()
    except Exception:
        pass
    path = "/task/stop?scope=all" if all_jobs else "/task/stop"
    ok, _ = _post_backend(path, {})
    print("🛑 Stop task delivered to backend: %s" % ("ok" if ok else "FAILED"))
    return bool(ok)


def _deliver_cancel_approval():
    """Deliver 'cancel approval' to the consent gate on the backend.

    G11 / F50 — the pending approval lives in the backend process; the voice
    process's module copy is empty, so cancelling locally was a no-op.
    POST /approvals/reset drops the pending plan (and the pending screen
    preview) authoritatively.
    """
    try:
        stop_speaking()
    except Exception:
        pass
    ok, reply = _post_backend("/approvals/reset", {})
    dropped = bool(isinstance(reply, dict) and reply.get("dropped"))
    print("🛑 Cancel approval delivered (%s): %s"
          % ("ok" if ok else "FAILED", "dropped pending plan" if dropped else "nothing pending"))


# ── Explicit websearch stop ("stop the research") ──
# F35: matched with the SAME normalization + negation rules as every other
# control (both apostrophe forms, a negation anywhere in the phrase or its
# run-up), so "don't stop the research" is never a stop request.
STOP_RESEARCH_PHRASES = (
    "stop the research", "stop the search", "stop researching",
    "stop the deepsearch", "stop deepsearch", "stop research", "stop search",
)


def is_stop_research(text):
    tokens = normalize_control_text(text)
    if not tokens:
        return False
    return bool(_match_tokens(tokens, STOP_RESEARCH_PHRASES))


def _deliver_stop_research():
    """Stop BOTH narration and the task — on the backend (F50).

    Mirrors brain.handle_stop_research_request(from_voice=True): the voice
    process cannot reach the backend's internals, so the control plane is
    the authed HTTP contract: /task/stop (stops research + browser task +
    live requests) and /speak/stop (cuts TTS + narration).
    """
    try:
        stop_speaking()
    except Exception:
        pass
    _post_backend("/task/stop", {})
    _post_backend("/speak/stop", {})
    print("🛑 Stop research delivered to backend")

# Control phrases that must NEVER classify as shutdown, with or without the
# Jarvis name: they have their own handlers (stop speaking / pause / cancel)
# or fall through to the normal command path. Checked BEFORE the shutdown
# vocabulary so 'jarvis stop speaking' can never terminate the stack.
_NON_SHUTDOWN_PHRASES = tuple(STOP_TALKING_EN) + tuple(STOP_TALKING_HI) + (
    "stop task", "stop the task", "cancel approval", "pause",
)

# Exact English shutdown commands, derived from the SHUTDOWN_EN vocabulary
# with the jarvis-name token removed (normalize_spoken_command strips it).
_SHUTDOWN_EN_EXACT = frozenset(
    " ".join(word for word in combo if word not in JARVIS_VARIANTS)
    for combo in SHUTDOWN_EN
)


def is_shutdown(text):
    """True ONLY for explicit shutdown phrases.

    F35: one exact, token-based, negation-aware match for BOTH languages — no
    substring matching, so "music band karo" is not shutdown and "bolna band
    karo" (stop talking) is handled by the speech target first. Control
    phrases with their own owner (speech/task/approval/pause/continue/sleep)
    are classified before shutdown and therefore never terminate the stack.
    """
    return classify_control(text) == "shutdown"


def is_continue(text):
    """F35: exact and negation-aware — "don't continue" never resumes."""
    return classify_control(text) == "continue"


def is_normal_setup(text):
    """True when the utterance asks for the normal setup (F50/F35).

    The old ``any(phrase in t)`` substring test fired on any utterance that
    merely CONTAINED a setup phrase — including "do not put my normal setup".
    Setup now uses the same exact, negation-aware token grammar as every other
    control, with two deliberate safeguards:

      * a setup launch is a state-changing effect, so ANY negator in the
        utterance refuses it (F16's fail-closed rule) rather than relying on
        where the negator sits;
      * an imperative verb ("put my normal setup", "mera setup karo") is
        stripped and the match retried — instead of widening the shared
        neutral-token sets, which the F35 grammar's negatives depend on.
    """
    tokens = normalize_control_text(text)
    if not tokens:
        return False
    if any(token in NEGATION_TOKENS for token in tokens):
        return False
    if _match_tokens(tokens, SETUP_PHRASES):
        return True
    reduced = [token for token in tokens if token not in SETUP_VERB_TOKENS]
    return bool(reduced) and bool(_match_tokens(reduced, SETUP_PHRASES))


# ─────────────────────────────────────────
# F35 — ONE delivery action per control owner
# ─────────────────────────────────────────
def _deliver_speech_stop():
    """Stop the SPEECH owner — never gated on this process's own flag.

    F35: the old path only honoured "stop speaking" while THIS process thought
    it was speaking. The backend voices replies and narrates tasks, so its
    local flag is False exactly when the user most needs the stop: the request
    must always reach the backend's /speak/stop (authed).
    """
    try:
        stop_speaking()
    except Exception:
        pass
    ok, _ = _post_backend("/speak/stop", {})
    print("🛑 Speech stop delivered to backend: %s" % ("ok" if ok else "FAILED"))
    return ok


def _deliver_pause():
    """Pause narration but KEEP the remainder resumable (F35 / P1-07).

    A VOICE reply's audio lives in THIS process, so the pause has to happen
    here. The old implementation called the local ``stop_speaking()`` FIRST —
    which discards the queue and bumps the generation — and only then posted
    ``/speak/pause`` to the backend, whose active stream is empty for a voice
    turn. Net effect: pause was exactly a stop and "continue" had nothing to
    resume.

    Pause is now a LOCAL, non-destructive control (``pause_speaking`` parks the
    play loop at the byte it reached and keeps the queue). The backend endpoint
    is still used when nothing was playing here, because the BACKEND process is
    the one narrating typed-UI replies and task announcements.
    """
    resumable = False
    try:
        resumable = bool(pause_speaking())
    except Exception as exc:
        print("[VOICE] local pause failed: %s" % exc)
        resumable = False
    if resumable:
        print("⏸️ Speech paused locally — resumable")
        return True
    ok, reply = _post_backend("/speak/pause", {})
    resumable = bool(isinstance(reply, dict) and reply.get("resumable"))
    print("⏸️ Speech paused (%s): %s"
          % ("ok" if ok else "FAILED",
             "resumable" if resumable else "nothing to resume"))
    return bool(ok and resumable)


def _deliver_continue():
    """Resume what a pause left unplayed (F35 / P1-07).

    A pause armed in THIS process is resumed in THIS process: the actor's parked
    play loop replays the unplayed tail from its cursor and the stream speaker's
    queue carries on. Only when nothing is paused here does this fall back to
    the backend, which owns the narration the BACKEND process speaks.

    Deliberately does NOT re-speak a text snapshot from here: the old
    ``listener_state.pop_remaining()`` + ``speak()`` fallback made this process a
    second playback owner, which F50 forbids.
    """
    try:
        if resume_local_playback():
            print("▶️ Resumed the paused audio locally")
            return True
    except Exception as exc:
        print("[VOICE] local resume failed: %s" % exc)
    ok, reply = _post_backend("/speak/resume", {})
    resumed = bool(isinstance(reply, dict) and reply.get("resumed"))
    print("▶️ Resume delivered to backend: %s"
          % ("ok" if ok and resumed else "nothing to resume"))
    return bool(ok and resumed)


def _request_watcher(path, timeout=3.0):
    """POST to the supervisor's control plane WITH the per-launch token.

    F35: the old shutdown path sent an unauthenticated request to
    ``/stop`` — which the watcher rejects with 401 (and which, even when it
    came from a trusted caller, is WARM SLEEP: it leaves the backend running).
    Both actions now carry the token the watcher injected into this process
    and target the endpoint that matches what the user asked for.
    """
    port = os.getenv("JARVIS_WATCHER_CONTROL_PORT")
    if not port:
        return False, None
    try:
        request = Request(
            f"http://127.0.0.1:{port}{path}",
            data=b"{}",
            method="POST",
            headers=_backend_headers(),
        )
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
        try:
            return True, json.loads(body)
        except Exception:
            return True, {}
    except Exception as exc:
        print(f"[VOICE→WATCHER] {path} failed: {exc}")
        return False, None


def _deliver_sleep():
    """Warm sleep: UI + voice down, backend kept warm (F35/F52)."""
    ok, reply = _request_watcher("/stop")
    mode = (reply or {}).get("mode") if isinstance(reply, dict) else None
    print("😴 Warm sleep requested: %s" % (mode or ("ok" if ok else "FAILED")))
    return ok


def _deliver_shutdown():
    """FULL shutdown of the whole stack, authenticated (F35).

    Target: watcher ``/shutdown`` (stops electron AND the warm backend AND the
    daemons). If the supervisor is unreachable we fall back to killing the
    tracked children ourselves instead of silently leaving half the stack up.
    """
    ok, reply = _request_watcher("/shutdown")
    if ok:
        mode = (reply or {}).get("mode") if isinstance(reply, dict) else None
        print("🔴 Full shutdown requested: %s" % (mode or "ok"))
        return True
    print("🔴 Supervisor unreachable — shutting down tracked processes directly")
    _taskkill_pid(os.getenv("JARVIS_BACKEND_PID"))
    _taskkill_pid(os.getenv("JARVIS_ELECTRON_PID"))
    _stop_backend_port_if_jarvis()
    return False


def dispatch_control(command):
    """Route ONE classified control to exactly its owner (F35).

    Returns True when the command was delivered. Every control has a single
    target, so a stop can never take down unrelated work and a shutdown can
    never be confused with "stop talking".
    """
    owner = control_owner(command)
    if owner is None:
        return False
    if command == "speech_stop":
        return _deliver_speech_stop()
    if command == "pause":
        return _deliver_pause()
    if command == "continue":
        return _deliver_continue()
    if command == "task_stop":
        return _deliver_stop_task()
    if command == "task_stop_all":
        return _deliver_stop_task(all_jobs=True)
    if command == "approval_cancel":
        return _deliver_cancel_approval()
    if command == "stop_research":
        return _deliver_stop_research()
    if command == "sleep":
        return _deliver_sleep()
    if command == "shutdown":
        return _deliver_shutdown()
    return False


# ─────────────────────────────────────────
# SETUP COMMAND
# ─────────────────────────────────────────
def launch_normal_setup():
    """Ask the BACKEND to launch the user's normal setup (F50).

    A setup launch opens applications and mutates shared session state, so it
    belongs to the process that owns intelligence state — this worker POSTs a
    typed request to ``/voice-setup/launch`` and never calls ``os.startfile``
    (or spawns anything) itself: one owner, one place to see what ran.

    Returns True when the backend accepted the launch.
    """
    print("🖥️ Requesting normal setup from the backend...")
    ok, reply = _post_backend("/voice-setup/launch", {"setup": "normal"})
    if not ok:
        print("❌ Normal setup request failed — the backend owns setup launches")
        return False
    launched = reply.get("launched") if isinstance(reply, dict) else None
    if isinstance(launched, list) and launched:
        print("✅ Normal setup launched by backend: %s" % ", ".join(
            str(item) for item in launched))
    else:
        print("✅ Normal setup accepted by the backend")
    return True


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def get_time_based_greeting():
    hour = datetime.datetime.now().hour
    if hour < 12:
        return "Good morning, sir."
    elif hour < 18:
        return "Good afternoon, sir."
    else:
        return "Good evening, sir."


def _taskkill_pid(pid):
    if not pid:
        return False
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
        )
        return True
    except Exception:
        return False


def _pids_on_port(port):
    pids = set()
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
        )
    except Exception:
        return pids

    marker = f":{port}"
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        if marker not in parts[1] or parts[-2].upper() != "LISTENING":
            continue
        try:
            pids.add(int(parts[-1]))
        except ValueError:
            continue
    return pids


def _is_jarvis_backend():
    try:
        with urlopen(f"http://127.0.0.1:{BACKEND_PORT}/health", timeout=0.8) as response:
            return b"jarvis-backend" in response.read()
    except Exception:
        return False


def _stop_backend_port_if_jarvis():
    if not _is_jarvis_backend():
        return False
    stopped = False
    for pid in _pids_on_port(BACKEND_PORT):
        stopped = _taskkill_pid(pid) or stopped
    return stopped


def _request_watcher_stop():
    """Warm sleep (kept for callers that explicitly want it)."""
    ok, _ = _request_watcher("/stop")
    return ok


def _terminate_self():
    """Exit this voice worker once the stack has been asked to stop.

    Separated so callers (and tests) can observe the shutdown request without
    the process disappearing: the SIGTERM is the LAST step, after the
    authenticated supervisor request.
    """
    try:
        import signal

        os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        os._exit(0)


def shutdown_everything():
    """Full shutdown: authenticated supervisor request, then this process.

    F35: this used to fire an UNAUTHENTICATED warm-stop at the watcher and
    call it done — the backend kept running. The full-shutdown request is
    authenticated, and if the supervisor cannot be reached the tracked
    children are killed directly before this process exits.
    """
    stop_speaking()
    try:
        speak("Goodbye sir. Shutting everything down.")
        time.sleep(3)
    except Exception:
        pass

    _deliver_shutdown()
    time.sleep(1)
    _terminate_self()


# ─────────────────────────────────────────
# THREAD 1 — LISTENER
# ─────────────────────────────────────────
# TEXT MODE FLAG — consumed over HTTP, never via module import.
# This voice process is a SEPARATE OS process from the API (spawned by
# the watcher): its imported copy of listener_state can never see the
# API-side toggles, so the flag is polled from the backend
# (GET /voice-mode) on a short cadence and cached. On request failure
# the last known value is kept.
_VOICE_FLAG_POLL_S = 1.0
_voice_flag_last_known = True
_voice_flag_checked_at = 0.0


def _fetch_voice_flag():
    """Current voice-input flag as seen by THIS process (HTTP-polled).

    F51: ``/voice-mode`` is a private read (only ``/health`` is public), so
    the poll authenticates with the launch token — an unauthenticated 401
    looked identical to "no answer" and froze the voice-input switch at its
    last known value.
    """
    global _voice_flag_last_known, _voice_flag_checked_at
    now = time.monotonic()
    if now - _voice_flag_checked_at < _VOICE_FLAG_POLL_S:
        return _voice_flag_last_known
    _voice_flag_checked_at = now
    try:
        request = Request(
            f"http://127.0.0.1:{BACKEND_PORT}/voice-mode",
            method="GET",
            headers=_backend_headers(),
        )
        with urlopen(request, timeout=0.5) as response:
            data = json.loads(response.read().decode("utf-8"))
        _voice_flag_last_known = bool(data.get("voice_input_enabled", True))
    except Exception:
        pass  # keep the last known value on request failure
    return _voice_flag_last_known


def voice_input_enabled():
    return _fetch_voice_flag()


def listener_thread():
    while True:
        try:
            if not voice_input_enabled():
                try:
                    _close_microphone_source()
                except Exception:
                    pass
                time.sleep(0.25)
                continue
            # [PERF] P1-19 — one mark timeline per capture. The listener stamps
            # this utterance's capture/STT boundaries on it and it travels with
            # the text to the brain thread, which ships it to the backend under
            # the request_id it submits the utterance with.
            turn = _latency.new_local_turn()
            text = listen(marks=turn)
            if not text:
                continue

            print("👤:", text)

            # ── F35: ONE exact control grammar runs BEFORE any mute. ──
            # Every control is classified here (one match, one owner) and
            # delivered while narration plays and while a long task runs:
            # supervision can never be locked out by Jarvis's own speaking or
            # working state, and a stop never depends on this process's own
            # (never-authoritative) speaking flag.
            control = classify_control(text)
            if control == "shutdown":
                print("🔴 Shutdown command received")
                shutdown_everything()
                break
            if control in ("speech_stop", "task_stop", "task_stop_all",
                           "approval_cancel", "pause", "continue", "sleep"):
                print(f"🎛️ Control command: {control}")
                dispatch_control(control)
                continue
            if is_stop_research(text):
                print("🛑 Stop research command")
                _deliver_stop_research()
                continue

            # [S18] The task no longer mutes conversation: committed
            # utterances are submitted while a task runs and the BACKEND
            # decides — chat turns answer normally, action requests queue
            # until the task releases the machinery. Echo safety is the
            # AEC gate's job now (S28): the task's own narration reaching
            # the mic is suppressed as echo, not by dropping everything.

            # [P1-06] A COMMITTED transcript is never silently discarded. It has
            # already passed the VAD, the human-voice gate and the hallucination
            # gate, so throwing it away wastes all that work and loses intent.
            #
            # If Jarvis is mid-reply, the user speaking means "change the
            # subject": that is an INTERRUPTION, so cut the current turn now
            # (audio dies, the old backend request is cancelled) and hand the
            # utterance to the P0-08 turn manager, which dispatches it
            # pre-emptively. The old `if listener_state.is_speaking(): continue`
            # dropped the transcript with no handling, no log and no
            # interruption — and because the speaking flag used to flicker off
            # between sentences, the drop was unpredictable as well. The flag is
            # now scoped to the whole reply session (voice.StreamSpeaker), so
            # this branch is the reliable "the user talked over the reply"
            # signal. Onset barge-in still cut the audio immediately in the
            # listener's own barge-in path; this is the commit-time belt.
            if listener_state.is_speaking():
                interrupted = _interrupt_active_turn()
                print("[LISTENER] Utterance during a reply — interrupting: %s"
                      % ("yes" if interrupted else "no active turn"))

            command_queue.put((text, turn))

        except Exception as e:
            print(f"Listener thread error: {e}")
            time.sleep(0.5)


def _clear_active_stream_if(speaker):
    """Clear the module-level active stream only while it is still ours.

    With per-turn workers a stale turn must not de-register the NEWER turn's
    speaker: that would leave background announcements with nowhere to go.
    """
    try:
        if get_active_stream() is speaker:
            set_active_stream(None)
    except Exception:
        pass


def _respond_to_utterance(text, turn=None):
    """Dispatch one queued voice utterance as its own pre-emptive TURN.

    G11 / F50: this process no longer executes its own ``process_message``
    copy. The utterance is submitted to the ONE backend task runtime
    (authenticated /ask/stream, ``speak=False``); streamed deltas feed the
    same StreamSpeaker the in-process path used, and the terminal frame's
    reply is authoritative. [S18] This runs while a backend task is in
    progress too: the utterance is submitted either way and the backend
    queues action-shaped requests while answering chat normally.

    [PERF] P1-19: *turn* is this utterance's mark timeline from the capture
    thread. Its marks ride the submission under the request_id below and are
    merged into the backend's record for the SAME turn; the playback
    boundaries it collects afterwards are shipped by the state publisher.

    [P0-08] This function now DISPATCHES: it registers the turn with the turn
    manager and hands the backend round trip to its own worker thread. Waiting
    here (as it used to) meant an interruption could not start until the reply
    it was interrupting had finished generating and speaking — the brain thread
    is a dispatcher, not a waiter. Registering before the thread starts is what
    makes a barge-in that lands mid-submission able to cancel this turn, and
    starting the turn is what pre-empts the previous one.
    """
    if turn is None:
        turn = _latency.new_local_turn()
    request_id = _new_turn_request_id()
    previous = get_active_stream()
    if previous is not None:
        previous.close()
    speaker = StreamSpeaker()
    set_active_stream(speaker)
    # [PERF] P1-19 — from here on, untagged marks (the audio actor's
    # tts_first_byte / playback_started) belong to THIS turn. Set at SUBMIT
    # time so submission ORDER decides ownership deterministically, even though
    # the turns themselves now run concurrently.
    _latency.set_local_turn(turn)
    early_marks = turn.marks()
    # [P0-08] Register BEFORE the worker starts, so a barge-in that arrives
    # while this turn is still being submitted can already cancel it — and so
    # this turn pre-empts whatever was active.
    TURNS.start(request_id, speaker)
    threading.Thread(
        target=_run_turn,
        name="voice-turn",
        daemon=True,
        args=(text, request_id, speaker, turn, early_marks),
    ).start()


def _safe_print(*args):
    """Print without letting an encoding failure derail the caller.

    The reply text is model output and the console is often cp1252, so echoing
    it can raise UnicodeEncodeError. On the brain thread that used to be
    swallowed by the loop's handler; a turn now runs on its own thread, so the
    echo must not be able to kill it.
    """
    try:
        print(*args)
    except Exception:
        try:
            print(*(str(arg).encode("ascii", "replace").decode("ascii")
                    for arg in args))
        except Exception:
            pass


def _run_turn(text, request_id, speaker, turn, early_marks):
    """Run one turn, guaranteeing no exception escapes the worker thread."""
    try:
        _run_turn_inner(text, request_id, speaker, turn, early_marks)
    except Exception as exc:
        print("[VOICE] turn error: %s" % exc)
        try:
            listener_state.set_thinking(False)
        except Exception:
            pass
        try:
            speaker.close()
        except Exception:
            pass
        TURNS.clear(request_id)


def _run_turn_inner(text, request_id, speaker, turn, early_marks):
    """One turn's backend round trip, on its own worker thread (P0-08).

    Every exit path releases the turn so the next utterance can start cleanly.
    """
    def _sink(delta):
        # A superseded turn must not feed the actor — the turn manager owns
        # "one voice at a time" and this is the belt to its braces for the
        # streaming path.
        if TURNS.is_current(request_id):
            try:
                speaker.feed(delta)
            except Exception:
                pass

    listener_state.set_thinking(True)
    response = None
    try:
        response = _ask_backend(
            text,
            request_id=request_id,
            stream_sink=_sink,
            client_marks=early_marks,
        )
    except Exception as exc:
        # Nothing reached the backend, so this turn has no record to join:
        # release it instead of letting its marks bleed into the next
        # utterance's shipping slot.
        print("[VOICE→BACKEND] turn failed: %s" % exc)
        _latency.set_local_turn(None)
    finally:
        listener_state.set_thinking(False)
        # Those marks have been handed over (inline, under this request_id);
        # everything the turn records from now on is shipped later.
        try:
            turn.ack(len(early_marks))
        except Exception:
            pass
    _safe_print("🤖:", response)

    # [P0-08] A superseded turn must never speak: its audio would land on top
    # of the answer that replaced it. The reply itself is already committed to
    # history by the backend (brain.handle_chat), so nothing the user wanted to
    # read is lost — only the out-of-date narration is dropped.
    if not TURNS.is_current(request_id):
        TURNS.note_stale_reply()
        try:
            speaker.close()
        except Exception:
            pass
        _clear_active_stream_if(speaker)
        return

    if not response:
        # The one runtime was unreachable — never leave the user in silence.
        speaker.close()
        _clear_active_stream_if(speaker)
        speak("I couldn't reach the backend, sir.")
        _watch_turn_for_shipping(turn, request_id)
        TURNS.clear(request_id)
        return

    if speaker.spoken_any:
        # The reply was voiced sentence-by-sentence as it streamed;
        # flush the remainder and let the worker take announcements.
        speaker.finish()
    else:
        # Task/tool/screen branches return a full reply without
        # streaming — speak it the normal way.
        speaker.close()
        _clear_active_stream_if(speaker)
        speak(response)
    # [PERF] P1-19 — hand the turn to the publisher: the playback boundaries
    # land while the audio is still playing, so they are shipped on its cadence.
    _watch_turn_for_shipping(turn, request_id)
    TURNS.clear(request_id)


# ─────────────────────────────────────────
# THREAD 2 — BRAIN
# ─────────────────────────────────────────
def handle_queued_item(text, turn=None):
    """Dispatch ONE queued utterance. Returns True when the loop must stop.

    Extracted from :func:`brain_thread` so the control-vs-utterance decision is
    directly testable. [P0-08] A control phrase is dispatched as a CONTROL and
    never becomes a turn: it must not be submitted to the backend as an
    interrupting utterance, which is what would otherwise make "stop" queue a
    second generation behind the reply it was meant to silence.
    """
    # ── F35: the SAME one grammar, so a queued utterance is handled
    # exactly like a live one (a control can never be re-interpreted
    # as a chat message after it was classified). ──
    control = classify_control(text)
    if control == "shutdown":
        print("🔴 Shutdown command received")
        shutdown_everything()
        return True
    if control == "continue":
        _deliver_continue()
        return False
    if control == "pause":
        _deliver_pause()
        return False
    if control in ("speech_stop", "task_stop", "task_stop_all",
                   "approval_cancel", "sleep"):
        print(f"🎛️ Control command: {control}")
        dispatch_control(control)
        return False

    # [S18] Queued utterances no longer die at task start: they are
    # submitted and the backend answers chat / queues actions. (Shutdown
    # stays available above as a deliberate kill switch.)

    # ── NORMAL SETUP ──
    if is_normal_setup(text):
        speak("Opening your normal setup, sir.")
        threading.Thread(target=launch_normal_setup, daemon=True).start()
        return False

    # ── PROCESS ──
    _respond_to_utterance(text, turn)
    return False


def brain_thread():
    while True:
        try:
            item = command_queue.get(timeout=1)
            # [PERF] P1-19 — the capture thread pairs the utterance with its
            # mark timeline; a plain string (a test, a legacy producer) keeps
            # working with no timeline.
            text, turn = item if isinstance(item, tuple) else (item, None)
            # [P0-08] Submit and return: this thread dispatches, it does not
            # wait for a generation. Waiting here is what made every
            # interruption queue behind the reply it was interrupting.
            if handle_queued_item(text, turn):
                break
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Brain thread error: {e}")
            time.sleep(0.5)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def start_voice_mode():
    greeting = get_time_based_greeting()
    reply    = f"{greeting} {random.choice(BOOT_RESPONSES)}"
    print("🟢 Jarvis Active")
    print("🤖:", reply)

    # Warm the Fish TTS model + connection in the background so the first
    # real reply isn't slowed by a cold synthesis request.
    threading.Thread(target=warm_up_fish_tts, daemon=True).start()

    speak(reply)

    # G11 / F50 — publish this worker's real listening state to the backend
    # so /voice-state and /ui-state show the TRUTH instead of the backend's
    # empty listener_state module copy.
    threading.Thread(target=_publish_voice_state_loop, daemon=True).start()

    # [S19] Subscribe to the backend's /events push channel so the task-mute
    # and voice-input flags update the moment they flip, instead of on the1s
    # poll. The polls stay as the fallback if this channel ever drops.
    threading.Thread(target=_task_state_push_loop, daemon=True).start()

    # Async replies (task completions, screen Q&A) are SPOKEN BY THE BACKEND
    # process where those jobs actually run — the old in-process callback
    # registrations here were dead code against module copies that never run
    # anything (F50 removes exactly that pattern).

    t_listener = threading.Thread(target=listener_thread, daemon=True)
    t_listener.start()

    try:
        brain_thread()
    except KeyboardInterrupt:
        print("\n🔴 Stopped manually")
        stop_speaking()


if __name__ == "__main__":
    start_voice_mode()
