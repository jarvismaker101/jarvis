import json
import logging
import os
import threading
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend import listener_state
from backend.core.brain import (
    process_message,
    set_async_reply_callback,
    opencode_task_in_progress,
    OPENCODE_START_PHRASE,
    BROWSER_AGENT_START_PHRASE,
)
from backend.services import browser_agent
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

last_voice_message = ""
last_voice_response = ""
last_voice_log_id = 0
last_request = {"message": None, "time": 0}

# G11 / F50 — the voice I/O worker publishes its real listening state here
# (one snapshot, newest state_seq wins); /voice-state and /ui-state serve it
# instead of this process's unrelated listener_state module copy.
_published_voice: dict = {}
_published_voice_seq = 0
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
    #: G11 / F50 — submission origin ("ui" | "voice"); diagnostics only.
    origin: str = ""


class VoiceLog(BaseModel):
    message: str
    response: str


def _publish_voice_log(message: str, response: str):
    global last_voice_message, last_voice_response, last_voice_log_id
    last_voice_message = message
    last_voice_response = response
    last_voice_log_id += 1


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
                kind="request",
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

        reply = process_message(
            state.message,
            from_voice=from_voice,
            stream_reply=on_delta,
            progress=on_progress,
            request_id=state.request_id,
            job=request_job,
        )
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
        deadline = time.time() + 60.0
        while not state.done and time.time() < deadline:
            time.sleep(0.25)
        if state.reply is not None:
            return {"reply": state.reply, "request_id": state.request_id}
        if state.done:
            return {"reply": "", "request_id": state.request_id}
        return {"reply": "Request still processing, sir.",
                "request_id": state.request_id}

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
        reply = process_message(query.message, from_voice=False)
        if reply is None:
            reply = "I didn't get a response. Please try again."
        state.complete(reply)
    except Exception as exc:
        state.error(exc)
        raise

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

    # Execute-once: only the first attach starts the worker; reconnects just
    # reattach to the same numbered event buffer.
    if req_registry.try_start(state):
        state.progress("started", stage="start")
        # G11 / F50 — a voice-worker submission (speak=False) runs the very
        # same request runtime but stays silent: the I/O worker speaks it.
        speak = bool(query.speak)
        threading.Thread(
            target=_run_request_worker,
            args=(state,),
            kwargs={
                "from_voice": False,
                "speak_stream": speak,
                "speak_terminal": speak,
            },
            daemon=True,
        ).start()

    def event(payload: dict):
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def generate():
        yield event({
            "type": "progress",
            "message": "attached",
            "stage": "attach",
            "request_id": state.request_id,
            "resume_from": state.latest_seq(),
            "seq": None,
        })
        for frame in state.stream(last_seq=query.last_event_id):
            yield event(frame)

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


@router.post("/voice-state/publish")
def publish_voice_state(payload: dict):
    """G11 / F50 — the voice I/O worker publishes its real listening state.

    The backend serves /voice-state and /ui-state to the UI from THIS
    published snapshot instead of reading its own unrelated listener_state
    module copy. Publishing goes through the ONE intelligence-state registry
    (F50): the payload must name its owner incarnation, a stale generation's
    publish is REJECTED (never merged), and an expired snapshot is not served.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="state payload must be an object")
    global _published_voice_seq
    owner = str(payload.get("owner") or payload.get("incarnation") or "").strip()
    generation = int(payload.get("generation") or 0)
    try:
        from backend.services import intelligence_state as istate

        role = str(payload.get("role") or istate.ROLE_VOICE)
        outcome = istate.worker_states.publish(
            role, owner, {k: v for k, v in payload.items()
                          if k not in ("owner", "incarnation", "role",
                                       "generation", "state_seq")},
            pid=int(payload.get("pid") or 0),
            generation=generation or None,
        )
        if getattr(outcome, "accepted", True) is False:
            return {"ok": True, "stale": True,
                    "reason": getattr(outcome, "reason", "")}
    except Exception as exc:
        logging.debug("[VOICE-STATE] registry publish skipped: %s", exc)
    with _published_voice_lock:
        seq = int(payload.get("state_seq") or 0)
        if seq and seq <= _published_voice_seq:
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
    }


@router.post("/speak/stop")
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
def stop_task(job_id: str = "", kind: str = ""):
    """Stop ONE identified job at its next checkpoint (F20).

    *job_id* addresses a specific job; without one the NEWEST still-running job
    is stopped — never every registered request, which is how "stop A" used to
    take unrelated work down with it. An idle stop cancels nothing: it leaves
    no flag behind for the next job to trip over.

    Idempotent and engine-safe: the owning token is cancelled, its registered
    stop handler arms the engine's legacy flag, the loop exits gracefully, the
    brain's finally resets the task-running flag, and the UI stop button hides
    on the next /ui-state poll. Any client still attached to a stream gets an
    INTERRUPTED terminal frame (F26). Also cuts TTS and mutes narration so the
    user gets silence immediately.
    """
    from backend.services import jobs as job_registry

    cancelled = []
    try:
        cancelled = job_registry.request_stop(
            job_id or None, kinds=(kind,) if kind else None)
    except Exception as exc:  # never let a stop request 500
        logging.warning("[STOP] job cancellation failed: %s", exc)
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
        req_registry.interrupt_active("stopped by user")
    except Exception:
        pass
    try:
        stop_speaking()
        set_narration_enabled(False)
    except Exception:
        pass
    return {"ok": True, "cancelled": cancelled}


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
