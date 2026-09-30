import functools
import json
import logging
import os
import threading
import time
from typing import List, Optional

import anyio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend import listener_state
from backend.core.brain import (
    process_message,
    register_voice_log_sink,
    set_async_reply_callback,
    opencode_task_in_progress,
    OPENCODE_START_PHRASE,
    BROWSER_AGENT_START_PHRASE,
)
from backend.services import browser_agent
from backend.services import latency as _latency
from backend.services import local_auth
from backend.services import model_registry
from backend.services import request_registry as req_registry
from backend.services import research_service
from backend.services import runtime_identity
from backend.services.approvals import invalidate as invalidate_approval
from backend.services.model_registry import ModelRegistryError
from backend.services.opencode_client import set_narration_enabled
from backend.services import screen_state
from backend.services.screen_control import set_response_callback
from backend.services.voice import speak, stop_speaking


router = APIRouter()

#: [P1-11] The kind a chat turn's own job carries (see _run_request_worker).
#: A bare /task/stop excludes it: stopping the conversation is request-scoped.
REQUEST_JOB_KIND = "request"

# ── P1-14 — RESERVED capacity for the control plane ─────────────────────────
# Barge-in is a safety-critical path: /speak/stop, /task/stop, /ask/cancel and
# /voice-state must answer while chat streams are open. They used to run on the
# SAME 40-thread pool the streams occupied for their whole lives, so enough open
# streams starved the very endpoints needed to stop them.
#
# Two changes fix that: streams no longer hold a pool thread at all (they are
# async generators parked on an anyio event — see RequestState.astream), and
# these four endpoints now run on their OWN limiter, so control work can never
# queue behind bulk work whatever else happens to the shared pool.
CONTROL_LIMITER = anyio.CapacityLimiter(4)


async def _on_control_plane(handler, *args, **kwargs):
    """Run a blocking control handler on the RESERVED control capacity.

    ``limiter=`` is anyio's own reservation mechanism: the call takes one of the
    four control tokens (and anyio uses that same limiter as its thread-pool
    limit for the call), so a burst of chat traffic cannot consume the capacity
    barge-in needs.
    """
    return await anyio.to_thread.run_sync(
        functools.partial(handler, *args, **kwargs),
        limiter=CONTROL_LIMITER,
    )

last_voice_message = ""
last_voice_response = ""
last_voice_log_id = 0
last_request = {"message": None, "time": 0}

# ── P0-08 — barge-in visibility, NOT a history rewrite ──
# The user's decision stands: an interrupted turn still commits its FULL
# generated reply to history (brain.handle_chat, unchanged), because they want
# to read what they missed. Nothing here touches that text — no "(interrupted)"
# suffix, no replacement with the spoken prefix. This is a separate, additive
# marker so the UI CAN indicate "this reply was cut off" later. It carries no
# reply text at all, and clearing it is a matter of dropping this block.
_last_interrupt_lock = threading.Lock()
_last_interrupt = {
    "interrupted": False,
    "request_id": "",
    "reason": "",
    "at": 0.0,
}


def _note_reply_interrupted(request_id, reason=""):
    """Record that the turn for *request_id* was cut off mid-reply."""
    with _last_interrupt_lock:
        _last_interrupt.update(
            interrupted=True,
            request_id=str(request_id or ""),
            reason=str(reason or ""),
            at=time.time(),
        )


def _note_reply_completed():
    """A reply finished normally, so the last interruption no longer describes
    the newest turn."""
    with _last_interrupt_lock:
        _last_interrupt.update(interrupted=False, at=time.time())


def get_last_reply_interrupted():
    """Snapshot of the barge-in marker for /ui-state."""
    with _last_interrupt_lock:
        return dict(_last_interrupt)


# G11 / F50 — the voice I/O worker publishes its real listening state here
# (one snapshot, newest state_seq wins); /voice-state and /ui-state serve it
# instead of this process's unrelated listener_state module copy.
_published_voice: dict = {}
#: [P1-15] The high-water mark of the publisher named by
#: ``_published_voice_publisher``. A bare counter cannot be compared across
#: publishers: the voice worker's sequence restarts at 1 on every launch, so a
#: restarted worker's updates were all discarded as "old" and the UI stayed
#: frozen. The mark is reset whenever the publisher identity changes.
_published_voice_seq = 0
_published_voice_publisher = ""
_published_voice_lock = threading.Lock()

# ── Screen answer state (for overlay) ──────────────
_screen_answer_id = 0
_screen_answer_data = {
    "id": 0, "tip": "", "evidence": [], "links": [], "images": [],
    "region": {}, "request_id": "", "capture_id": "",
    "revision": 0, "enriched": False, "timestamp": 0,
}
#: F30 — capture id of the newest published answer. An enrichment patch that
#: belongs to an older capture is rejected instead of overwriting it.
_screen_answer_capture = ""
#: F30 — the newest capture GENERATION, and the lock that makes the
#: publish/patch decision atomic (two concurrent pushes used to interleave
#: their read-merge-write and lose one of the decorations).
_screen_answer_seq = 0
_screen_answer_lock = threading.Lock()


class Query(BaseModel):
    message: str
    #: F23 — client-generated request id, shared by /ask and /ask/stream.
    request_id: str = ""
    #: F23 — reconnect: resume after this event sequence instead of
    #: re-executing the message.
    last_event_id: int = -1
    #: G11 / F50 — when False the BACKEND does not speak the reply: the
    #: submitting client (the voice I/O worker) owns playback itself. Typed
    #: UI requests keep the default (True) and are spoken here as before.
    speak: bool = True
    #: G11 / F50 — submission origin ("ui" | "voice").
    #:
    #: [PERF] This used to be diagnostics only, which left the voice path
    #: running the FULL chat profile: no compact prompt, no 300-token cap and
    #: no spoken filler-word stripping — so a spoken turn produced a longer
    #: answer (later first audio, more TTS chunks, later completion) than the
    #: voice path was designed for. The field is now authoritative for two
    #: decisions: the compact reply profile and ``from_voice`` in the brain.
    origin: str = ""
    #: [PERF] P1-19 — the voice worker's already-captured turn marks
    #: (speech_end / capture_end / stt_start / stt_done …), shipped WITH the
    #: submission so one waterfall can span both processes. Each entry is
    #: ``[name, perf_counter_ns, meta]`` — absolute marks, never durations.
    client_marks: List[list] = []
    #: [PERF] P1-19 — the client's own clock sampled at send time; it is the
    #: single reference used to align ``client_marks`` onto this process's
    #: clock. Absent means "assume a shared clock".
    client_now_ns: Optional[int] = None


def _is_voice_submission(query) -> bool:
    """True when this request came from the voice I/O worker.

    ``speak=False`` is the F50 contract the voice worker uses to own playback
    itself, and ``origin`` is the explicit marker. Either one is accepted so a
    submission that predates the ``origin`` field still gets the right profile.
    """
    origin = (getattr(query, "origin", "") or "").strip().lower()
    if origin == "voice":
        return True
    return not bool(getattr(query, "speak", True))


def _begin_turn_clock(request_id, query, route):
    """[PERF] P1-19 — start this turn's waterfall at the request boundary.

    The clock used to start inside the worker thread, so everything the request
    spent before the worker was scheduled was invisible. ``begin`` is
    idempotent, so an F23 retry/reconnect re-attaches instead of restarting.

    Best-effort by contract: a telemetry failure must never touch the request
    path, so every call here swallows its own errors.
    """
    try:
        _latency.begin(request_id,
                       origin="voice" if _is_voice_submission(query) else "ui",
                       label=(getattr(query, "message", "") or "")[:60])
        _latency.mark(request_id, "http_in", meta={"route": route})
    except Exception:
        pass


def _merge_client_marks(request_id, query):
    """[PERF] P1-19 — merge the voice worker's marks into THIS turn.

    The worker and this backend are separate OS processes (F50), so its
    capture/STT marks — and, later, its playback marks — are shipped here under
    the SAME ``request_id``. An unknown turn simply drops them.
    """
    try:
        marks = getattr(query, "client_marks", None)
        if not marks:
            return 0
        return _latency.merge_client_marks(
            request_id, marks,
            client_now_ns=getattr(query, "client_now_ns", None))
    except Exception:
        return 0


class VoiceLog(BaseModel):
    message: str
    response: str


def _publish_voice_log(message: str, response: str):
    global last_voice_message, last_voice_response, last_voice_log_id
    last_voice_message = message
    last_voice_response = response
    last_voice_log_id += 1


def _voice_log_message(message: str) -> str:
    """The user text the UI mirror should show for a voice turn.

    P1-12: same normalisation the brain applied before publishing — the spoken
    "command ..." prefix is an addressing form, not part of what the user asked,
    so the log shows the request itself.
    """
    text = str(message or "").strip()
    if text.lower().startswith("command"):
        return text[len("command"):].strip()
    return text


#: P1-12 — this module owns the voice-log state, so the brain updates it by
#: calling this function DIRECTLY instead of POSTing to /update-voice-log (a
#: request that carried no token and therefore 401'd on every voice turn).
register_voice_log_sink(_publish_voice_log)


def _publish_async_screen_reply(response: str):
    reply = (response or "").strip()
    if not reply:
        return
    _publish_voice_log("", reply)
    threading.Thread(target=speak, args=(reply,), daemon=True).start()


set_response_callback(_publish_async_screen_reply)


def _publish_async_opencode_reply(reply: str, spoken: str = None):
    """Speak + mirror background task results (opencode completion, etc.).

    `spoken` overrides what is said when it differs from the chat text: the
    opencode completion speaks the fixed 'Sir, the task has been completed.'
    while the full summary stays in the UI log. While an opencode task runs,
    only opencode speaks — the text still goes to the UI, the speech is
    suppressed (the completion phrase is unaffected: the flag is already
    False by the time it fires).
    """
    _publish_voice_log("", reply)
    if opencode_task_in_progress():
        print("[API] Async reply while opencode task runs — speech suppressed.")
        return
    threading.Thread(target=speak, args=(spoken or reply,), daemon=True).start()


def _maybe_speak(reply: str):
    """Speak a UI reply unless an opencode task is running.

    The handoff start phrases are exempt — the mandatory announcement made
    exactly at handoff, which the mute would otherwise swallow once the task
    flag is already True.
    """
    if not opencode_task_in_progress() or reply in (
        OPENCODE_START_PHRASE,
        BROWSER_AGENT_START_PHRASE,
    ):
        threading.Thread(target=speak, args=(reply,), daemon=True).start()


set_async_reply_callback(_publish_async_opencode_reply)


def _run_request_worker(state, from_voice=False, speak_stream=False,
                        speak_terminal=True):
    """Execute a registered request once, publishing numbered events.

    F26: streamed final-channel deltas also feed one StreamSpeaker, so typed
    replies start speaking before generation completes — the same contract
    voice replies already had. When nothing streams (tool acks, screen
    answers, confirmations) the terminal reply is spoken via _maybe_speak.

    G11 / F50: ``speak_stream=False`` AND ``speak_terminal=False`` suppress
    ALL backend speech for this request — the voice I/O worker submits its
    utterances with both flags off and owns playback itself, so the backend
    never double-speaks a voice reply.
    """
    speaker = None
    # F20: this stream owns a cancellable job. Interrupting the request (stop
    # button, voice "stop", /task/stop, or a replaced worker) now cancels the
    # worker itself instead of only telling the client it stopped.
    request_job = None
    try:
        try:
            from backend.services import jobs as job_registry

            request_job = job_registry.new_job(
                kind=REQUEST_JOB_KIND,
                label=(state.message or "")[:80],
            )
            state.attach_job(request_job)
        except Exception as exc:
            print("[API] job registration failed:", exc)
            request_job = None
        # Barge-in: stop in-progress TTS before the new query runs.
        try:
            stop_speaking()
            set_narration_enabled(False)
        except Exception:
            pass
        if speak_stream and not opencode_task_in_progress():
            try:
                from backend.services.voice import StreamSpeaker, set_active_stream
                speaker = StreamSpeaker()
                set_active_stream(speaker)
            except Exception as exc:
                print("[API] stream speaker unavailable:", exc)
                speaker = None

        def on_delta(text):
            if not text:
                return
            # [PERF] First streamed token: the first moment the user (or the
            # voice speaker) can perceive anything at all.
            _latency.mark(state.request_id, "first_token")
            state.delta(text)
            if speaker is not None:
                try:
                    speaker.feed(text)
                except Exception:
                    pass

        def on_progress(message, **extra):
            # F30/F23 — phase updates (vision running, enrichment, …) so a
            # silent-but-working request is never mistaken for a dead one.
            try:
                state.progress(message, **extra)
            except Exception:
                pass

        # [PERF] P1-19 — the turn clock is begun by the ROUTE handler (it sees
        # `http_in`, where the request actually arrived); this is the
        # idempotent fallback for a caller that reaches the worker directly,
        # and `finish` closes the record. Recording is best-effort and can
        # never fail a request: both helpers swallow their own errors.
        _latency.begin(state.request_id, origin="voice" if from_voice else "ui",
                       label=(state.message or "")[:60])
        try:
            reply = process_message(
                state.message,
                from_voice=from_voice,
                # [P1-12] The brain no longer maintains the voice log on this
                # path. It used to POST to our own /update-voice-log on EVERY
                # voice turn — without the token, so it 401'd every time, after
                # spawning a thread and a round trip. The publish happens once
                # below, in-process, where the reply is final.
                sync_voice=False,
                # [PERF] A spoken turn uses the compact profile: the reply is
                # short (under 35 words / 300 max tokens) which is what makes
                # the first audio arrive sooner and the turn finish sooner.
                voice_compact=from_voice,
                stream_reply=on_delta,
                progress=on_progress,
                request_id=state.request_id,
                job=request_job,
            )
        finally:
            _latency.finish(state.request_id)
        if reply is None:
            reply = "I didn't get a response. Please try again."

        if request_job is not None and request_job.cancelled:
            # F20: the worker was stopped while it ran. Do NOT publish a
            # completed frame — an interruption must never become a completion
            # (nor be spoken) after the user asked for silence.
            state.interrupt(request_job.cancel_reason or "stopped by user")
            print("[API] Request cancelled; no completion published.")
            return

        # Terminal authority: one completed frame carries the final reply.
        state.complete(reply)
        # [P1-12] The voice log is updated IN PROCESS, once, here — the reply is
        # final at this point. Same update logic the endpoint uses (this IS
        # `_publish_voice_log`), no HTTP, no thread, and a failure can never
        # touch the reply because it is swallowed and the turn is already
        # complete. Only voice turns publish: the log mirrors the last spoken
        # exchange for the UI, exactly as the brain's `sync_voice_log` did.
        if from_voice:
            try:
                _publish_voice_log(_voice_log_message(state.message), reply)
            except Exception as exc:
                logging.debug("[VOICE-LOG] in-process publish failed: %s", exc)
        # P0-08: this turn finished normally, so the barge-in marker no longer
        # describes the newest reply. It never touched the reply text itself.
        _note_reply_completed()
        print("[API] Stream done, speaking:", reply[:80])
        if not speak_terminal:
            # Voice I/O worker owns playback — the backend stays silent.
            return
        if speaker is not None:
            try:
                speaker.finish()
            except Exception:
                pass
            if not speaker.spoken_any:
                _maybe_speak(reply)
        else:
            _maybe_speak(reply)
    except Exception as exc:
        print("[API] Stream error:", exc)
        state.error(exc)
    finally:
        if request_job is not None:
            try:
                request_job.finish()
            except Exception:
                pass
        if speaker is not None:
            try:
                speaker.close()
            except Exception:
                pass


@router.post("/ask")
def ask(query: Query):
    global last_request

    # F23: when the client carries a request id we already executed (or are
    # executing), answer from the registry instead of re-running the message
    # — a retried POST must never trigger the same action twice. A DIFFERENT
    # message under the same id is a conflict: attaching it to the other
    # request's result would answer a question the user never asked.
    state, _created, conflict = req_registry.admit(
        query.request_id, query.message)
    if conflict:
        raise HTTPException(
            status_code=409,
            detail="request_id is already bound to a different message",
        )
    if state is None:
        raise HTTPException(
            status_code=503,
            detail="request registry saturated; retry shortly",
        )
    if not req_registry.try_start(state):
        # [PERF] Event-driven wait on the request's own condition instead of a
        # 0.25s poll, so a terminal frame is observed as soon as it lands.
        state.wait_done(timeout=60.0)
        if state.reply is not None:
            return {"reply": state.reply, "request_id": state.request_id}
        if state.done:
            return {"reply": "", "request_id": state.request_id}
        return {"reply": "Request still processing, sir.",
                "request_id": state.request_id}

    # [PERF] P1-19 — start this turn's waterfall here: the request has arrived
    # and has an identity, and everything the request spends before the worker
    # thread is scheduled is part of the turn's latency.
    _begin_turn_clock(state.request_id, query, "/ask")
    _merge_client_marks(state.request_id, query)

    current_time = time.time()
    if (
        query.message == last_request["message"]
        and (current_time - last_request["time"]) < 1
        and not query.request_id
    ):
        print("[API] Duplicate request blocked")
        state.complete("Duplicate ignored")
        return {"reply": "Duplicate ignored", "request_id": state.request_id}

    last_request["message"] = query.message
    last_request["time"] = current_time

    print("[API] Processing:", query.message)
    # Barge-in: a typed query cuts any in-progress TTS BEFORE the new query
    # is processed (generation bump + fish PCM flush). The new reply then
    # speaks normally via _maybe_speak below.
    try:
        stop_speaking()
        set_narration_enabled(False)
    except Exception:
        pass
    try:
        # [PERF] Voice submissions (speak=False / origin=voice) get the compact
        # profile and the from_voice paths, exactly like /ask/stream.
        from_voice = _is_voice_submission(query)
        reply = process_message(query.message, from_voice=from_voice,
                                voice_compact=from_voice,
                                # [PERF] P1-19 — the transport identity, so this
                                # turn's marks (racer_start, classify_done,
                                # first_token) and its screen answer are
                                # attributed to THIS request like they are on
                                # the streaming path.
                                request_id=state.request_id)
        if reply is None:
            reply = "I didn't get a response. Please try again."
        state.complete(reply)
    except Exception as exc:
        state.error(exc)
        raise
    finally:
        # [PERF] P1-19 — close this turn's waterfall (marks `end`).
        _latency.finish(state.request_id)

    print("[API] Sending reply:", reply)
    if query.speak:
        _maybe_speak(reply)
    return {"reply": reply, "request_id": state.request_id}


@router.post("/ask/stream")
def ask_stream(query: Query):
    """SSE streaming endpoint with request identity (F23) and one event
    protocol (F26).

    Every frame is ``{type, seq, request_id, ...}`` where type is
    ``delta`` | ``replace`` | ``progress`` | ``completed`` | ``interrupted``
    | ``error``. The client supplies ``request_id``; a reconnecting client
    passes ``last_event_id`` and resumes after it — the message is never
    re-executed. Quiet phases emit bounded progress heartbeats so the client
    can tell a slow task from a dead connection.
    """
    message = (query.message or "").strip()
    # F23: atomic admission. A reused id carrying a DIFFERENT message is a
    # conflict (409), never a silent reattach to another request's stream.
    state, _created, conflict = req_registry.admit(query.request_id, message)
    if conflict:
        raise HTTPException(
            status_code=409,
            detail="request_id is already bound to a different message",
        )
    if state is None:
        raise HTTPException(
            status_code=503,
            detail="request registry saturated; retry shortly",
        )

    # [PERF] P1-19 — the waterfall starts at the request boundary, not inside
    # the worker thread (which made admission and thread start invisible), and
    # the voice worker's capture/STT marks are merged in under this SAME
    # request_id. Both calls are best-effort.
    _begin_turn_clock(state.request_id, query, "/ask/stream")
    _merge_client_marks(state.request_id, query)

    # Execute-once: only the first attach starts the worker; reconnects just
    # reattach to the same numbered event buffer.
    if req_registry.try_start(state):
        state.progress("started", stage="start")
        # G11 / F50 — a voice-worker submission (speak=False) runs the very
        # same request runtime but stays silent: the I/O worker speaks it.
        speak = bool(query.speak)
        # [PERF] A spoken turn gets the compact profile (short answer, 300
        # max tokens, spoken filler words stripped in command mode) and is
        # marked from_voice so the brain's voice-specific paths actually run.
        from_voice = _is_voice_submission(query)
        threading.Thread(
            target=_run_request_worker,
            args=(state,),
            kwargs={
                "from_voice": from_voice,
                "speak_stream": speak,
                "speak_terminal": speak,
            },
            daemon=True,
        ).start()

    def event(payload: dict):
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def generate():
        # [P1-14] This used to be a SYNC generator, so every open stream held a
        # thread-pool thread for its whole life — the same 40-thread pool the
        # control endpoints (barge-in) need. An async generator parked on an
        # asyncio.Event holds NO thread, and each batch is written as ONE
        # chunk instead of one socket write per delta.
        yield event({
            "type": "progress",
            "message": "attached",
            "stage": "attach",
            "request_id": state.request_id,
            "resume_from": state.latest_seq(),
            "seq": None,
        })
        async for batch in state.astream(last_seq=query.last_event_id):
            # One write per batch; the per-frame seq values are untouched.
            yield "".join(event(frame) for frame in batch)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/ask/status/{request_id}")
def ask_status(request_id: str):
    """F23: status lookup for a request — reconnects check before resuming."""
    snapshot = req_registry.state_snapshot(request_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="unknown request_id")
    return snapshot


@router.post("/ask/cancel/{request_id}")
async def cancel_request_route(
        request_id: str,
        reason: str = "cancelled by voice barge-in"):
    """P0-08 — cancel ONE in-flight request, on the RESERVED control capacity
    (P1-14), so a barge-in is never queued behind open chat streams."""
    return await _on_control_plane(cancel_request, request_id, reason)


def cancel_request(request_id: str, reason: str = "cancelled by voice barge-in"):
    """P0-08 — cancel ONE in-flight request, addressed by its request id.

    The voice I/O worker calls this the moment the user barges in: without it,
    an interruption only stopped the *audio*, while the backend kept generating
    the old reply and the next utterance waited behind it. This cancels that one
    request. [P1-11] It is also the ONLY way to stop a chat turn: a bare
    ``/task/stop`` deliberately leaves ``request`` jobs alone, so request-scoped
    cancellation is this route (and ``state.attach_job``'s job id for the worker
    it owns), never the task-shaped stop.

    Contract:

    * **Request-scoped.** Only *request_id* is touched; a concurrent unrelated
      request keeps running.
    * **Never cancels a finished turn.** A request that already reached a
      terminal state keeps its terminal frame — a ``completed`` reply stays
      completed, which is exactly the F23/F20 immutability rule.
    * **Idempotent and safe when the request is gone.** An unknown id, an
      already-finished id and an already-interrupted id are all ``ok: True``
      no-ops with a distinct ``reason``, so a caller never has to check first.
    * **Never raises.** Any internal failure is reported in the body.
    * **Does not touch history.** Cancelling publishes the INTERRUPTED terminal
      frame and cancels the worker; the full generated reply has already been
      committed by ``brain.handle_chat`` and stays readable (user decision).
    """
    rid = (request_id or "").strip()
    try:
        state = req_registry.get(rid)
    except Exception as exc:                      # never raise into the caller
        return {"ok": True, "cancelled": False, "reason": "lookup_failed",
                "error": str(exc)[:200]}
    if state is None:
        return {"ok": True, "cancelled": False, "reason": "unknown_request",
                "request_id": rid}
    try:
        with state.cond:
            already_done = state.done
        if already_done:
            # A terminal frame is immutable: report which one it was rather
            # than pretending the cancel did something.
            return {
                "ok": True,
                "cancelled": False,
                "reason": "already_interrupted" if state.interrupted
                          else "already_finished",
                "request_id": rid,
            }
        state.interrupt(reason)
        _note_reply_interrupted(rid, reason)
        return {"ok": True, "cancelled": True, "reason": reason,
                "request_id": rid}
    except Exception as exc:
        return {"ok": True, "cancelled": False, "reason": "error",
                "request_id": rid, "error": str(exc)[:200]}


@router.post("/update-voice-log")
def update_voice_log(data: VoiceLog):
    _publish_voice_log(data.message, data.response)
    return {"ok": True}


@router.get("/voice-log")
def get_voice_log():
    return {
        "id": last_voice_log_id,
        "message": last_voice_message,
        "response": last_voice_response,
    }


class VoiceModeUpdate(BaseModel):
    enabled: bool


@router.get("/voice-mode")
def get_voice_mode():
    return {"voice_input_enabled": listener_state.is_voice_input_enabled()}


@router.post("/voice-mode")
def set_voice_mode(payload: VoiceModeUpdate):
    enabled = listener_state.set_voice_input_enabled(payload.enabled)
    return {"voice_input_enabled": enabled}


#: F50 — the normal setup the *backend* launches. The voice worker used to
#: call os.startfile() itself, which made a second writer of shared session
#: state; it now asks for a typed backend job (see voice_mode.launch_normal_setup).
NORMAL_SETUP_APPS = ("brave", "vscode", "whatsapp", "edge")


@router.post("/voice-setup/launch")
def launch_voice_setup(payload: dict):
    """Run the user's normal setup AS a backend job (F50).

    Returns the job id immediately (the launches themselves are journalled in
    the shared history), so the caller can report or stop them.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="setup payload must be an object")
    setup = str(payload.get("setup") or "normal").strip().lower()
    if setup not in ("normal", "default", ""):
        raise HTTPException(status_code=400, detail="unknown setup '%s'" % setup)

    launched = []

    def _handler(job):
        from backend.core.executor import launch_app

        for app in NORMAL_SETUP_APPS:
            try:
                # F20: a cancelled setup job stops before the next launch.
                job.checkpoint()
            except Exception:
                break
            if launch_app(app):
                launched.append(app)

    try:
        from backend.services import intelligence_state as istate

        job = istate.submit_setup_effect(
            "normal-setup", _handler, label="normal setup",
            owner=istate.ROLE_BACKEND)
        job_id = getattr(job, "job_id", "")
    except Exception as exc:
        logging.warning("[SETUP] backend setup launch failed: %s", exc)
        raise HTTPException(status_code=503,
                            detail="could not start the setup job")
    return {"ok": True, "job_id": job_id, "launched": launched}


@router.get("/voice-state")
async def get_voice_state_route():
    """The UI's voice-state read, on the RESERVED control capacity (P1-14).

    The UI polls this while a reply is streaming; it must not wait behind the
    streams themselves.
    """
    return await _on_control_plane(get_voice_state)


def get_voice_state():
    state = listener_state.get_voice_state()
    # G11 / F50 — the real listening state lives in the voice I/O worker (a
    # separate OS process); when it publishes, its truth overrides this
    # process's empty module copy. State published by a SUPERSEDED worker
    # generation (or an expired one) is never served.
    published = get_published_voice_state()
    if published:
        state.update(published)
    state.update(screen_state.get_state())
    return state


def _voice_state_publisher(payload: dict) -> str:
    """[P1-15] Who is publishing this state stream?

    The per-launch ``publisher_id`` is authoritative (a pid can be recycled);
    ``owner`` (the F50 incarnation) and the pid are the fallbacks, and a
    publisher that identifies itself in NO way keeps the legacy behaviour of
    one global counter.
    """
    for key in ("publisher_id", "launch_id", "incarnation", "owner"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    pid = payload.get("publisher_pid") or payload.get("pid")
    try:
        if int(pid or 0):
            return "pid:%d" % int(pid)
    except (TypeError, ValueError):
        pass
    return ""


@router.post("/voice-state/publish")
def publish_voice_state(payload: dict):
    """G11 / F50 — the voice I/O worker publishes its real listening state.

    The backend serves /voice-state and /ui-state to the UI from THIS
    published snapshot instead of reading its own unrelated listener_state
    module copy. Publishing goes through the ONE intelligence-state registry
    (F50): the payload must name its owner incarnation, a stale generation's
    publish is REJECTED (never merged), and an expired snapshot is not served.

    [P1-15] The out-of-order guard is scoped to ONE publisher: a slow update
    from the same publisher is still refused, but a NEW publisher (a restarted
    worker, whose counter starts again at 1) is accepted and resets the
    baseline instead of being ignored forever.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="state payload must be an object")
    global _published_voice_seq, _published_voice_publisher
    owner = str(payload.get("owner") or payload.get("incarnation") or "").strip()
    generation = int(payload.get("generation") or 0)
    try:
        from backend.services import intelligence_state as istate

        role = str(payload.get("role") or istate.ROLE_VOICE)
        outcome = istate.worker_states.publish(
            role, owner, {k: v for k, v in payload.items()
                          if k not in ("owner", "incarnation", "role",
                                       "generation", "state_seq")},
            pid=int(payload.get("pid") or payload.get("publisher_pid") or 0),
            generation=generation or None,
        )
        if getattr(outcome, "accepted", True) is False:
            return {"ok": True, "stale": True,
                    "reason": getattr(outcome, "reason", "")}
    except Exception as exc:
        logging.debug("[VOICE-STATE] registry publish skipped: %s", exc)
    with _published_voice_lock:
        seq = int(payload.get("state_seq") or 0)
        publisher = _voice_state_publisher(payload)
        if publisher and publisher != _published_voice_publisher:
            # A different publisher: its counters mean nothing here, so accept
            # the update and start the ordering guard over from ITS baseline.
            _published_voice_seq = 0
            _published_voice_publisher = publisher
        elif seq and seq <= _published_voice_seq:
            # Same publisher, older (or repeated) frame: a slow update must
            # never overwrite a newer one.
            return {"ok": True, "stale": True}
        _published_voice.clear()
        for key, value in payload.items():
            if key in ("state_seq", "owner", "incarnation", "role",
                       "generation"):
                continue
            _published_voice[str(key)] = value
        if seq:
            _published_voice_seq = seq
        if owner:
            _published_voice["owner"] = owner
        if publisher:
            _published_voice["publisher_id"] = publisher
    return {"ok": True}


def get_published_voice_state():
    with _published_voice_lock:
        snapshot = dict(_published_voice)
    # F50: when the publishing worker identified itself, its snapshot is only
    # truth while that incarnation is the live one — an EXPIRED or superseded
    # worker's state is not served. Publishers that do not identify themselves
    # keep the legacy behaviour (nothing to expire).
    if not snapshot.get("owner"):
        return snapshot
    try:
        from backend.services import intelligence_state as istate

        if istate.worker_states.current(istate.ROLE_VOICE) is None:
            return {}
    except Exception:
        pass
    return snapshot


@router.get("/ui-state")
def get_ui_state():
    """Fused endpoint: voice-state + latest voice-log in one round-trip.

    Lets the renderer poll a single URL instead of two, halving request
    volume and eliminating state/log skew between polls.
    """
    state = listener_state.get_voice_state()
    published = get_published_voice_state()
    if published:
        state.update(published)
    state.update(screen_state.get_state())
    return {
        "state": state,
        "voice_log": {
            "id": last_voice_log_id,
            "message": last_voice_message,
            "response": last_voice_response,
        },
        "voice_input_enabled": state.get("voice_input_enabled", True),
        "task_running": opencode_task_in_progress(),
        # P0-08: additive barge-in marker. The interrupted reply is still in
        # `voice_log`/history in full — this only lets the UI say it was cut off.
        "last_reply_interrupted": get_last_reply_interrupted(),
    }


@router.post("/speak/stop")
async def stop_speech_route():
    """Instant barge-in stop for typed queries (and the voice process echo).

    [P1-14] Runs on the RESERVED control capacity so stopping playback is
    never queued behind open chat streams.
    """
    return await _on_control_plane(stop_speech)


def stop_speech():
    """Instant barge-in stop for typed queries (and the voice process echo).

    Cuts any in-progress TTS mid-word: generation token bump + fish PCM
    buffer flush via stop_speaking. Also mutes the running narration so
    the next task phrase does not resume playing. Idempotent, never raises.
    """
    try:
        stop_speaking()
        set_narration_enabled(False)
    except Exception:
        pass
    return {"ok": True}


@router.post("/speak/pause")
def pause_speech():
    """Pause narration and KEEP the remainder for a resume (F35).

    "Stop speaking" and "pause" are distinct controls: stop discards the rest
    of the sentence stream, pause keeps it so "continue" has something real to
    resume instead of depending on leftover stream state.
    """
    try:
        from backend.services.voice import pause_speaking

        resumable = pause_speaking()
        set_narration_enabled(False)
        return {"ok": True, "resumable": bool(resumable),
                "remaining_chars": len(listener_state.get_remaining() or "")}
    except Exception as exc:
        logging.warning("[SPEAK] pause failed: %s", exc)
        return {"ok": False, "error": str(exc)}


@router.post("/speak/resume")
def resume_speech():
    """Speak the text an interruption left unplayed (F35)."""
    try:
        from backend.services.voice import resume_speaking

        resumed = resume_speaking()
        return {"ok": True, "resumed_chars": len(resumed or ""),
                "resumed": bool(resumed)}
    except Exception as exc:
        logging.warning("[SPEAK] resume failed: %s", exc)
        return {"ok": False, "error": str(exc)}


@router.get("/speak/remaining")
def speech_remaining():
    """Unplayed speech this process still holds (F35 pause/continue).

    The voice process is a separate OS process: its ``listener_state`` copy
    can never see what the API paused, so the remainder is published here.
    """
    remaining = listener_state.get_remaining() or ""
    return {"remaining": remaining, "has_remaining": bool(remaining),
            "chars": len(remaining)}


@router.get("/aec/state")
def aec_state():
    """Explicit AEC state (F33): mode, degradation reason, reference stats.

    A silent no-op is not acceptable instrumentation, so the voice process (and
    the UI) can always see whether echo cancellation is real, degraded, or off,
    how much reference PCM has been rendered, and how fresh it is.
    """
    try:
        from backend.services.echo_cancel import aec_state as _state

        return _state()
    except Exception as exc:  # never let diagnostics 500
        return {"mode": "unknown", "degraded": True, "reason": str(exc)}


@router.get("/aec/reference")
def aec_reference(seconds: float = 0.5):
    """Rendered playback PCM overlapping the caller's mic window (F33).

    This is the cross-process AEC transport: the API process renders replies
    while the voice process owns the microphone, so the listener asks for the
    reference span here instead of reading a process-local buffer that is
    always empty. ``age_seconds`` is measured on this process's clock and is
    what establishes the shared render/capture timeline.
    """
    try:
        from backend.services import echo_cancel as _echo

        pcm, age = _echo.signal_path.reference_span(seconds)
        import base64

        return {
            "pcm_b64": base64.b64encode(pcm).decode("ascii") if pcm else "",
            "sample_rate": _echo.AEC_SAMPLE_RATE,
            "sample_width": _echo.AEC_SAMPLE_WIDTH,
            "bytes": len(pcm),
            "age_seconds": age,
            "mode": _echo.signal_path.state().get("mode"),
            "degraded": _echo.signal_path.state().get("degraded"),
        }
    except Exception as exc:
        logging.warning("[AEC] reference fetch failed: %s", exc)
        return {"pcm_b64": "", "bytes": 0, "age_seconds": None,
                "error": str(exc)}


@router.post("/task/stop")
async def stop_task_route(job_id: str = "", kind: str = ""):
    """Stop ONE identified job at its next checkpoint (F20).

    [P1-14] Runs on the RESERVED control capacity (see CONTROL_LIMITER), so a
    stop is never queued behind open chat streams.
    """
    return await _on_control_plane(stop_task, job_id, kind)


def stop_task(job_id: str = "", kind: str = ""):
    """Stop ONE identified job at its next checkpoint (F20).

    *job_id* addresses a specific job; without one the NEWEST still-running job
    of the requested kind is stopped. [P1-11] A BARE stop (no id, no kind)
    deliberately EXCLUDES ``request`` jobs: with the P0-08 turn manager there is
    genuinely more than one thing running, and the newest job of any kind is
    frequently the chat request the user is reading — "stop the browser task"
    must never kill the answer to their question. Stopping a chat turn is
    request-scoped (``POST /ask/cancel/{request_id}``), which is what the turn
    manager already uses.

    [P1-11] The stop also interrupts ONLY the streams the cancelled job owned.
    It used to call ``interrupt_active``, so cancelling one job published an
    INTERRUPTED terminal frame to every live request; a concurrent unrelated
    turn died with it. A worker whose own job was cancelled still terminates its
    stream itself (``_run_request_worker`` checks ``job.cancelled``), so nothing
    is left hanging on the wire.

    Idempotent and engine-safe: the owning token is cancelled, its registered
    stop handler arms the engine's legacy flag, the loop exits gracefully, the
    brain's finally resets the task-running flag, and the UI stop button hides
    on the next /ui-state poll. A stop with nothing to stop reports "stopped
    nothing" instead of touching unrelated work. Also cuts TTS and mutes
    narration so the user gets silence immediately.
    """
    from backend.services import jobs as job_registry

    cancelled = []
    try:
        if kind:
            # An explicit kind is authoritative — including kind="request".
            cancelled = job_registry.request_stop(
                job_id or None, kinds=(kind,))
        elif job_id:
            cancelled = job_registry.request_stop(job_id)
        else:
            cancelled = job_registry.request_stop(
                None, exclude_kinds=(REQUEST_JOB_KIND,))
    except Exception as exc:  # never let a stop request 500
        logging.warning("[STOP] job cancellation failed: %s", exc)
    # [P1-11] Interrupt exactly the requests the cancelled job produced.
    interrupted = []
    try:
        interrupted = req_registry.interrupt_job(cancelled, "stopped by user")
    except Exception as exc:
        logging.warning("[STOP] request interruption failed: %s", exc)
    # F50: the checkpoint where work halted is part of the shared history, so a
    # replaced/restarted worker can see WHICH job stopped and why.
    if cancelled:
        try:
            from backend.services import intelligence_state as istate

            for stopped_id in cancelled:
                istate.note_checkpoint(
                    "stopped", job_id=str(stopped_id), authority="user",
                    detail={"kind": kind or "any", "requested": job_id or ""})
        except Exception:
            pass
    # Legacy engines that are driven by their module flag rather than a job
    # (a run started before job registration) still need to hear the stop.
    if not cancelled and not job_id:
        try:
            browser_agent.request_stop()
            research_service.request_stop()
        except Exception:
            pass
    try:
        stop_speaking()
        set_narration_enabled(False)
    except Exception:
        pass
    # "Stopped nothing" is a valid and useful answer, so say which it was.
    return {
        "ok": True,
        "cancelled": cancelled,
        "stopped": cancelled[0] if cancelled else "",
        "interrupted": interrupted,
    }


@router.post("/approvals/reset")
def approvals_reset():
    """G11 / F52 — supervisor + voice control entry for consent reset.

    Before a backend worker is torn down or replaced, the supervisor calls
    this so a pending screen-plan approval can never survive into a NEW
    worker and be consumed there ("invalidate stale approvals before any
    safe recovery"). The voice worker's 'cancel approval' phrase routes here
    too, because the pending approval lives in THIS process. Also clears the
    pending screen plan so the UI preview cannot re-arm a dropped consent,
    and interrupts live registered requests so a replaced worker cannot
    resume a half-finished job. Idempotent; returns the dropped approval.
    """
    record = invalidate_approval("supervisor reset")
    # F50: consent drops belong in the ONE shared history, so the next worker
    # can see that a stale approval was invalidated rather than guessing.
    try:
        from backend.services import intelligence_state as istate

        istate.note_approval(
            "approval-dropped", authority="supervisor reset",
            detail={"dropped": (record.to_dict() if record else None)})
    except Exception:
        pass
    try:
        from backend.services import screen_state
        screen_state.clear_pending_plan()
    except Exception:
        pass
    try:
        req_registry.interrupt_active("supervisor replacement")
    except Exception:
        pass
    return {"ok": True, "dropped": record.to_dict() if record else None}


@router.get("/latency")
def get_latency(limit: int = 50):
    """[PERF] Per-turn latency telemetry — the numbers the latency work needs.

    P1-19: every mark in a record is an ABSOLUTE ``perf_counter_ns`` boundary;
    durations are DERIVED at read time by differencing consecutive marks, so a
    duration can no longer contradict the offset it is stored next to, and the
    step deltas of a turn sum to its ``total_ms``. ``summary`` keeps the
    pre-P1-19 keys (``samples``, ``spans``, ``total`` and their
    ``median_ms`` / ``p90_ms`` / ``max_ms``) and adds the ``waterfall`` rows
    (p50 / p90 / max per step over the window, ordered by median offset) plus
    ``p50_ms`` and the ``offset_*`` numbers.

    The ``listener`` module is deliberately NOT imported here (P1-19,
    requirement 7): this process owns no microphone, so reading the listener's
    AEC counter from here always reported 0 anyway — the voice worker
    publishes it with its state snapshot instead.

    Authed like every other private read (F51): it exposes what users said.
    """
    limit = max(1, min(int(limit or 50), 200))
    try:
        return {
            "summary": _latency.summary(limit),
            "recent": _latency.recent(limit),
            "aec": {
                "remote": _aec_remote_stats(),
                "listener_aec_errors": _published_voice_aec_errors(),
            },
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


class ClientMarks(BaseModel):
    """[PERF] P1-19 — one batch of the voice worker's already-stamped marks."""

    #: The SAME request id the utterance was submitted with.
    request_id: str = ""
    #: ``[[name, perf_counter_ns, meta], ...]`` — absolute marks only.
    marks: List[list] = []
    #: The client's clock at send time (the alignment reference).
    client_now_ns: Optional[int] = None


@router.post("/latency/client")
def post_latency_client(payload: ClientMarks):
    """[PERF] P1-19 — merge another process's marks into one turn's waterfall.

    The voice worker and this backend are separate OS processes (F50), so a
    mark it records is invisible here until it is merged. The worker ships its
    capture/STT marks inline with the submission and the later ones (the
    playback boundaries) through this endpoint, always under the request_id it
    submitted with — never a second id, or the two halves would describe
    different turns.

    Authenticated like every other private endpoint (F51, via the installed
    middleware). Telemetry is never allowed to fail a turn, so a mark that
    cannot be merged is dropped and the answer says how many were accepted.
    """
    try:
        merged = _latency.merge_client_marks(
            payload.request_id, payload.marks,
            client_now_ns=payload.client_now_ns)
    except Exception:
        merged = 0
    return {"ok": True, "merged": merged}


def _published_voice_aec_errors():
    """The voice worker's AEC error count, as IT published it (F50 / P1-19).

    The count lives in the listener, which runs in the VOICE process (the
    process that owns the microphone). Reading it here used to mean importing
    the listener module into the API process, where the counter is always 0.
    """
    try:
        with _published_voice_lock:
            return int(_published_voice.get("aec_errors") or 0)
    except Exception:
        return 0


def _aec_remote_stats():
    """AEC transport counters, including the [PERF] cache/idle-skip hits."""
    try:
        from backend.services import echo_cancel as _aec
        transport = getattr(_aec.signal_path, "transport", None)
        return dict(getattr(transport, "stats", {}) or {})
    except Exception:
        return {}


@router.get("/health")
def health():
    """Liveness + G11 / F52 identity surface.

    ``instance_id``/``protocol``/``build`` let the supervisor verify WHICH
    runtime is listening instead of trusting a port; ``pid`` lets it
    attribute the listening socket to a process it owns (so a foreign
    listener sharing the port is never killed); ``auth`` reports the
    local-token fingerprint (or "off") so a supervisor can tell whether a
    warm backend still holds THIS launch's token.
    """
    return {
        "ok": True,
        "service": "jarvis-backend",
        "instance_id": runtime_identity.instance_id(),
        "pid": os.getpid(),
        "protocol": runtime_identity.protocol_version(),
        "build": runtime_identity.build_id(),
        "auth": local_auth.token_fingerprint(),
    }


# ── Screen answer endpoints (for overlay) ──────────
class EvidenceItem(BaseModel):
    # F48: provenance is NOT decoration — the overlay is where the user decides
    # whether to trust a claim, so the observation/verification metadata has to
    # survive this model. Declaring only source/title/snippet silently DROPPED
    # spans/provenance/uncertainty/lookup on the way to the UI.
    source: str = ""
    title: str = ""
    snippet: str = ""
    #: verbatim quotes that support the claim (claim-level spans)
    spans: List[dict] = []
    #: observed / inferred / externally_checked …
    provenance: str = ""
    #: corroboration level + accounting (independent sources vs mirrors)
    corroboration: dict = {}
    uncertainty: str = ""
    observed_at: str = ""
    #: the lookup this claim came from (query + reference), when there was one
    lookup: dict = {}
    #: the claim this evidence is evidence FOR
    claim: str = ""
    query: str = ""
    url: str = ""
    #: F48: anything a producer adds later survives too, instead of being
    #: silently discarded by the response model.
    model_config = {"extra": "allow"}


class LinkItem(BaseModel):
    label: str = ""
    url: str = ""
    icon: str = "🔗"


class ImageItem(BaseModel):
    url: str = ""
    title: str = ""
    caption: str = ""
    width: int = 480
    height: int = 360


class ScreenAnswer(BaseModel):
    tip: str
    evidence: List[EvidenceItem] = []
    links: List[LinkItem] = []
    images: List[ImageItem] = []
    region: dict = {}
    #: F30 — set on an enrichment patch to update an already-published answer
    #: in place instead of allocating a new (and therefore newer-looking) id.
    id: int = 0
    #: F30 — the F23 request this answer belongs to (traceability only).
    request_id: str = ""
    #: F30 — the capture/question this result came from. A push whose capture
    #: is no longer the newest is rejected so a late answer cannot overwrite a
    #: newer question.
    capture_id: str = ""
    #: F30 — the capture GENERATION, registered by the backend BEFORE the slow
    #: vision call. A late initial post from an older generation is refused
    #: instead of replacing the newer answer that is already on screen.
    capture_seq: int = 0


def _model_to_dict(item):
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return item.dict()


@router.post("/screen-answer")
def post_screen_answer(data: ScreenAnswer):
    """Publish a screen answer, or patch the decorations of an existing one.

    F30 — the tip/evidence is published as soon as vision returns; optional
    exploration links and topic images arrive later in a bounded enrichment
    phase that updates **the same answer id** (``id`` + ``revision``). The
    overlay re-renders on a revision bump, so decorations appear without the
    answer ever being duplicated or reordered.

    Every push carries the ``capture_id`` of the capture it came from. A
    patch whose id is not the newest, or whose capture has been superseded by
    a newer question, is rejected with 409 instead of clobbering the newer
    answer.
    """
    global _screen_answer_id, _screen_answer_data, _screen_answer_capture
    global _screen_answer_seq
    update_id = data.id or 0

    # F30 — ONE atomic decision. The read-merge-write below is a critical
    # section: an enrichment patch arriving while a newer answer publishes
    # must either land on its own answer or be refused, never merge into a
    # different one.
    with _screen_answer_lock:
        if update_id:
            if update_id != _screen_answer_id:
                raise HTTPException(status_code=409, detail="stale screen answer")
            if data.capture_seq and data.capture_seq < _screen_answer_seq:
                raise HTTPException(status_code=409,
                                    detail="stale screen capture generation")
            if (
                data.capture_id
                and _screen_answer_capture
                and data.capture_id != _screen_answer_capture
            ):
                raise HTTPException(status_code=409, detail="stale screen capture")
            merged = dict(_screen_answer_data)
            # F30 — decorations only, and only additively. A patch may never
            # replace the authoritative tip/evidence with different text: the
            # answer the user is already reading is not a decoration. Blank /
            # identical values stay a no-op.
            if data.tip and data.tip != merged.get("tip"):
                raise HTTPException(
                    status_code=409,
                    detail="enrichment may not replace the answer text")
            if data.evidence:
                incoming = [_model_to_dict(e) for e in data.evidence]
                if incoming != list(merged.get("evidence") or []):
                    raise HTTPException(
                        status_code=409,
                        detail="enrichment may not replace the evidence")
            if data.links:
                merged["links"] = [_model_to_dict(l) for l in data.links]
            if data.images:
                merged["images"] = [_model_to_dict(i) for i in data.images]
            if data.region and not merged.get("region"):
                merged["region"] = data.region
            merged["revision"] = int(merged.get("revision") or 0) + 1
            merged["enriched"] = True
            _screen_answer_data = merged
            return {"ok": True, "id": update_id, "updated": True,
                    "revision": merged["revision"]}

        # A NEW answer. A late INITIAL post from an older capture generation
        # must not replace the answer already on screen — that was the audit's
        # "older A finishing after B replaces B".
        if data.capture_seq and data.capture_seq < _screen_answer_seq:
            raise HTTPException(status_code=409,
                                detail="stale screen capture generation")
        _screen_answer_id += 1
        _screen_answer_capture = data.capture_id or ""
        if data.capture_seq:
            _screen_answer_seq = max(_screen_answer_seq, int(data.capture_seq))
        _screen_answer_data = {
            "id": _screen_answer_id,
            "tip": data.tip,
            "evidence": [_model_to_dict(e) for e in data.evidence],
            "links": [_model_to_dict(l) for l in data.links],
            "images": [_model_to_dict(i) for i in data.images],
            "region": data.region or {},
            "request_id": data.request_id or "",
            "capture_id": data.capture_id or "",
            "capture_seq": int(data.capture_seq or 0),
            "revision": 0,
            "enriched": False,
            "timestamp": time.time(),
        }
        return {"ok": True, "id": _screen_answer_id, "updated": False}


@router.get("/screen-answer")
def get_screen_answer():
    return _screen_answer_data


# ── Research report endpoints (glass overlay) ─────────
_research_answer_id = 0
_research_answer_data = {"id": 0, "query": "", "markdown": "", "videos": [], "report_path": ""}


class ResearchResult(BaseModel):
    id: int = 0
    query: str = ""
    markdown: str = ""
    videos: List[dict] = []
    report_path: str = ""
    visited_count: int = 0
    failed_count: int = 0


@router.post("/research-result")
def post_research_result(data: ResearchResult):
    global _research_answer_id, _research_answer_data
    # Respect the sender's timestamp-id so the backend copy and the file
    # mirror dedupe to the same id on the overlay side.
    _research_answer_id = data.id or (_research_answer_id + 1)
    _research_answer_data = {
        "id": _research_answer_id,
        "query": data.query,
        "markdown": data.markdown,
        "videos": [dict(v) for v in data.videos],
        "report_path": data.report_path,
        "visited_count": data.visited_count,
        "failed_count": data.failed_count,
        "timestamp": time.time(),
    }
    return {"ok": True, "id": _research_answer_id}


@router.get("/research-result")
def get_research_result():
    return _research_answer_data


# ── Research progress + incremental evidence (F28) ─────────
# A deep research run outlives the chat request that started it, so its
# progress cannot ride the SSE stream (that stream is already closed). Same
# shape as /research-result: brain POSTs, anything interested GETs.
_research_progress_data = {"id": 0, "query": "", "message": "", "evidence": []}


class ResearchProgress(BaseModel):
    id: int = 0
    query: str = ""
    message: str = ""
    evidence: List[dict] = []


@router.post("/research-progress")
def post_research_progress(data: ResearchProgress):
    global _research_progress_data
    _research_progress_data = {
        "id": data.id or int(time.time() * 1000),
        "query": data.query,
        "message": data.message,
        "evidence": [dict(e) for e in data.evidence],
        "timestamp": time.time(),
    }
    return {"ok": True, "id": _research_progress_data["id"]}


@router.get("/research-progress")
def get_research_progress():
    return _research_progress_data


# ── Model settings (UI model switcher) ─────────────
class ChatModelUpdate(BaseModel):
    provider: str
    model: str


class ModelUpdate(BaseModel):
    role: str
    provider: str
    model: str


class ProviderAddRequest(BaseModel):
    id: str
    name: str
    api_key: str
    base_url: str
    # F49: the capability set explicitly authorized for this provider. Omitted
    # -> the shipped OpenAI-compatible default is recorded with the provider.
    capabilities: Optional[List[str]] = None


class ProviderTestRequest(BaseModel):
    """Candidate credentials for the add-provider form's TEST button.

    Deliberately carries NO id/name: this endpoint validates a (base_url,
    api_key) pair live and persists nothing.
    """
    api_key: str
    base_url: str


@router.get("/settings")
def get_settings():
    """Current models per role + selectable providers (masked: has_key only,
    raw API keys never leave the registry)."""
    try:
        from backend.core.brain import get_last_chat_fallback
        last_fb = get_last_chat_fallback()
        # Scrub and limit age for UI: only expose if exists, frontend checks recency
        if last_fb:
            # ensure no key material in error
            last_fb = dict(last_fb)
            last_fb["error"] = model_registry._scrub_secrets(str(last_fb.get("error", "")))[:200]
    except Exception:
        last_fb = None
    try:
        role_allowed = model_registry.get_role_allowed_map()
    except Exception:
        role_allowed = {}

    def role_model(role):
        """F49: a persisted selection that no longer validates must be
        visible (and fixable) from the UI instead of breaking this endpoint.
        The rejection reason is surfaced scrubbed; the role resolves to None.
        """
        try:
            return model_registry.get_model_for_role(role), None
        except Exception as exc:
            return None, model_registry._scrub_secrets(str(exc))[:200]

    resolved = {}
    errors = {}
    # F49: every role the registry can resolve is reported, so a feature whose
    # model is selectable in the registry is also selectable from the UI. The
    # planner role drives the native tool-use orchestrator; omitting it here
    # left its model with no way to be chosen.
    for role in ("chat", "tts", "vision", "browser_tool", "listening",
                 "planner"):
        resolved[role], errors[role] = role_model(role)
    return {
        "chat_model": resolved["chat"],
        "tts_model": resolved["tts"],
        "vision_model": resolved["vision"],
        "browser_tool_model": resolved["browser_tool"],
        "listening_model": resolved["listening"],
        "planner_model": resolved["planner"],
        "model_errors": {r: e for r, e in errors.items() if e},
        "providers": model_registry.list_providers(),
        "role_allowed": role_allowed,
        # Which functionality sections accept a user-added OpenAI-compatible
        # provider (so the UI shows "add custom provider" exactly there).
        "custom_provider_roles": model_registry.roles_allowing_custom(),
        "last_fallback": last_fb,
    }


@router.post("/settings/chat-model")
def set_chat_model(payload: ChatModelUpdate):
    """Make (provider, model) the default for text chat responses.

    Effect is immediate — brain.py resolves the registry per message — so
    the very next reply uses the new model, no restart needed.
    Kept for backward compat; delegates to the generic role setter.
    """
    try:
        chat_model = model_registry.set_model_for_role(
            "chat", payload.provider, payload.model
        )
    except ModelRegistryError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "chat_model": chat_model}


@router.post("/settings/model")
def set_model(payload: ModelUpdate):
    """Generic per-role model switch: chat/tts/vision/browser_tool/listening."""
    try:
        result = model_registry.set_model_for_role(
            payload.role, payload.provider, payload.model
        )
    except ModelRegistryError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # Return role-specific key plus generic for convenience
    key = {
        "chat": "chat_model",
        "tts": "tts_model",
        "vision": "vision_model",
        "browser_tool": "browser_tool_model",
        "listening": "listening_model",
        "planner": "planner_model",
    }.get(str(payload.role).strip(), "model")
    return {"ok": True, key: result, "role": payload.role}


@router.get("/providers/{provider_id}/models")
def get_provider_models(provider_id: str):
    """Live model list for a provider (proxied server-side so the API key
    stays on the backend)."""
    try:
        models = model_registry.list_provider_models(provider_id)
    except ModelRegistryError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"models": models}


@router.post("/prewarm")
def prewarm(force: bool = False):
    """[P0-12] Open the pooled provider connections BEFORE the turn needs them.

    Called by the voice worker on VAD onset — several hundred milliseconds
    before a transcript exists — so the chat / classifier / STT handshake is
    already done when the turn starts. Advisory in every direction:

    * authenticated like every other control endpoint (auth fails closed here,
      so an unauthenticated pre-warm would be a new hole);
    * rate-limited per process to one warm per ``WARM_INTERVAL_SECONDS``, and
      skipped outright while a warm is already in flight — never queued, so it
      can never become a bottleneck of its own;
    * silent but COUNTED on failure. A provider being down returns
      ``ok: True`` with a ``degraded`` list: it is an optimisation, not a
      precondition, and the system must work exactly as before without it.
    """
    from backend.services import prewarm as prewarm_service

    try:
        return prewarm_service.warm(force=force)
    except Exception as exc:  # a warm must never 500
        logging.debug("[PREWARM] refused: %s", exc)
        return {"ok": True, "warmed": [], "degraded": ["warm"]}


@router.get("/prewarm/stats")
def prewarm_stats():
    """[P0-12] What the pre-warm has actually done (counters, last result)."""
    from backend.services import prewarm as prewarm_service

    try:
        return prewarm_service.stats()
    except Exception:
        return {"warms": 0, "skipped": 0, "degraded": 0, "errors": 0,
                "targets": {}}


@router.post("/settings/provider")
def add_provider(payload: ProviderAddRequest):
    """Add a custom OpenAI-compatible provider. The key is validated with a
    live model-list call before anything is persisted; invalid keys get a
    clear 4xx and are never stored. F49: the capability set the caller
    explicitly authorizes is stored with the provider (omitted -> the shipped
    OpenAI-compatible default)."""
    try:
        provider = model_registry.add_custom_provider(
            payload.id, payload.name, payload.api_key, payload.base_url,
            capabilities=payload.capabilities,
        )
    except ModelRegistryError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "provider": provider}


@router.post("/settings/provider/test")
def test_provider(payload: ProviderTestRequest):
    """TEST button of the add-provider form: live-check (base_url, api_key)
    with one model-list call and store NOTHING — the user sees whether the
    provider works before saving. Success returns the discovered models
    (ids/displays only) so the caller can show what a save would offer."""
    try:
        models = model_registry.test_custom_provider(
            payload.base_url, payload.api_key
        )
    except ModelRegistryError as exc:
        raise HTTPException(
            status_code=400,
            detail=model_registry._scrub_secrets(str(exc))[:200],
        )
    return {"ok": True, "count": len(models), "models": models}
