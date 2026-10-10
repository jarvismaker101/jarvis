import contextlib
import json
import itertools
import logging
import os
import queue
import random
import re
import threading
import time
import uuid

import requests

from backend.core import deadline as budget
from backend.core import entity_ledger
from backend.core.executor import execute_multiple
from backend.core.memory import add_message, clear_history, get_history
# G9: guarded import — the memory store is optional and import-safe
# (pure data module; every call is a no-op when disabled).
try:
    from backend.core import memory_store
except Exception:
    memory_store = None

from backend.services.screen_control import (
    maybe_handle_screen_control_message,
    set_response_callback,
)
from backend.services.task_agent import (
    consume_task_confirmation,
    handle_task_message,
    has_pending_task_confirmation,
    is_code_tool_request,
    is_explicit_task_request,
    is_task_request,
)
from backend.services.task_agent.agent import (
    _TASK_CONFIRM_NO_RE,
    _TASK_CONFIRM_YES_RE,
    last_task_result as _last_task_result,
)
from backend.config import BACKEND_PORT
from backend.services.fireworks_client import ask_fireworks, ask_fireworks_stream
from backend.services.gemini_client import (
    ask_gemini_chat,
    ask_gemini_chat_stream,
)
from backend.services import jobs
from backend.services import model_registry
from backend.services import multi_intent
from backend.services.task_agent import agent as task_agent_module
from backend.services.openai_compat_client import (
    ask_openai_compat,
    ask_openai_compat_stream,
)
from backend.services.intent import classify_intent
from backend.services.web_task_routing import is_web_shaped_task
from backend.services.orchestrator import handle_message as orchestrator_handle_message
from backend.services.orchestrator import select_route as orchestrator_select_route
from backend.services.opencode_client import run_opencode_task, is_opencode_available, set_narration_enabled
from backend.services.browser_agent import run_browser_task, request_stop as request_browser_task_stop, consume_stop_report as consume_browser_stop_report
from backend.services import browser_agent
from backend.services import event_bus
from backend.services.task_result import (
    TaskResult,
    is_failure_text,
    result_from_reported_text,
)
from backend import config
from backend.services.research_service import run_research, request_stop as request_research_stop, clear_stop_request
from backend.services.quick_search import run_quick_search, fetch_ai_overview_text
from backend.services.screen_analyzer import analyze_screen, is_screen_question, is_region_question
from backend.services.screen_analyzer import identify_on_screen, get_observation
from backend.services.screen_analyzer import compose_search_query
from backend.services.image_fetcher import build_explore_links, fetch_topic_images


#: P1-12 — where the voice log actually lives. The API layer owns that state
#: (``routes._publish_voice_log``) and registers its updater here.
#:
#: The brain used to POST to its OWN ``/update-voice-log`` with no token. Auth
#: fails closed in this project, so that request 401'd on EVERY voice turn: a
#: wasted round trip, a background thread and a misleading error in the log,
#: every time. The backend is already in the same process as the voice-log
#: state, so there is nothing to call over HTTP. Registering a sink instead of
#: importing the route module also keeps the voice I/O worker (which imports
#: this module but has no route state of its own) free of the whole API layer.
_voice_log_sink = None


def register_voice_log_sink(sink):
    """Install the in-process voice-log updater (the API layer calls this)."""
    global _voice_log_sink
    _voice_log_sink = sink
    return sink


def sync_voice_log(message, response):
    """P1-12 — update the voice log IN PROCESS, and never over HTTP.

    Bookkeeping only: the UI's voice indicator reads this state, so it must
    keep working, and a failure must never touch the reply. There is no thread
    here any more either — publishing is now a couple of assignments on the
    calling thread instead of a POST that had to be pushed off it.
    """
    sink = _voice_log_sink
    if sink is None:
        return False
    try:
        sink(message, response)
        return True
    except Exception:
        return False


#: F30 — a decoration phase is bounded. Exploration links and topic images are
#: optional polish; the answer itself is never held back for them.
SCREEN_ENRICH_TIMEOUT = 8.0
#: F30 — at most this many decoration phases run at once. Each one is a
#: network-bound background thread; a hanging enrichment must not be able to
#: create unbounded work while the user keeps asking about the screen.
SCREEN_ENRICH_MAX_THREADS = 2
_screen_capture_seq = itertools.count(1)
_screen_enrich_lock = threading.Lock()
_screen_enrich_active = 0


def begin_screen_capture(request_id=None):
    """F30 — register the capture GENERATION *before* the slow vision call.

    The generation is what orders two overlapping screen questions: a late
    initial answer from generation N can never replace generation N+1's
    answer, because the backend already knows N+1 was asked first.
    """
    seq = next(_screen_capture_seq)
    return {
        "seq": seq,
        "capture_id": "cap-%d-%s" % (seq, uuid.uuid4().hex[:8]),
        "request_id": request_id or "",
        "at": time.time(),
    }


def _start_screen_enrichment(answer_id, capture, tip, evidence, links, topic,
                             show_images, region):
    """F30 — start ONE bounded decoration phase, or skip it.

    Returns True when a phase was started. When the cap is reached the
    decorations are simply dropped: the answer is already on screen, and the
    slow optional polish must never delay it or pile up.
    """
    global _screen_enrich_active
    with _screen_enrich_lock:
        if _screen_enrich_active >= SCREEN_ENRICH_MAX_THREADS:
            logging.debug("[SCREEN] enrichment already running — skipping")
            return False
        _screen_enrich_active += 1

    def _worker():
        global _screen_enrich_active
        try:
            _enrich_screen_answer(answer_id, capture["capture_id"],
                                  capture["request_id"], tip, evidence, links,
                                  topic, show_images, region,
                                  capture_seq=capture["seq"])
        finally:
            with _screen_enrich_lock:
                _screen_enrich_active = max(0, _screen_enrich_active - 1)

    try:
        threading.Thread(target=_worker, daemon=True).start()
        return True
    except Exception as exc:
        with _screen_enrich_lock:
            _screen_enrich_active = max(0, _screen_enrich_active - 1)
        logging.debug("[SCREEN] enrichment thread failed: %s", exc)
        return False


def push_screen_answer(tip, evidence, links=None, images=None, region=None,
                       answer_id=None, request_id=None, capture_id=None,
                       capture_seq=None):
    """Push a screen analysis result to the overlay API endpoint.

    F30 — the answer and its decorations are delivered separately:

      * the first call publishes tip + evidence the moment vision returns and
        allocates an answer id;
      * a later call carrying that *answer_id* is an enrichment patch: it adds
        links/images to the **same** answer instead of publishing a second,
        apparently newer one.

    *capture_id* ties every push to the capture/question it came from, so a
    slow enrichment belonging to an old question can never overwrite the
    answer to a newer one (the backend rejects it with 409).

    Returns the answer id, or None when the push failed or was rejected as
    stale.
    """
    try:
        resp = requests.post(
            f"http://127.0.0.1:{BACKEND_PORT}/screen-answer",
            json={
                "tip": tip,
                "evidence": evidence,
                "links": links or [],
                "images": images or [],
                "region": region or {},
                "id": answer_id or 0,
                "request_id": request_id or "",
                "capture_id": capture_id or "",
                "capture_seq": int(capture_seq or 0),
            },
            timeout=2,
        )
        if resp.status_code == 409:
            logging.debug("[SCREEN] answer update rejected as stale")
            return None
        try:
            return (resp.json() or {}).get("id")
        except Exception:
            return None
    except Exception:
        logging.debug("Could not push screen answer to overlay API.")
        return None


def _enrich_screen_answer(answer_id, capture_id, request_id, tip, evidence,
                          links, topic, show_images, region, capture_seq=None):
    """F30 — bounded decoration phase for an already-published screen answer.

    Runs on a background thread *after* the tip is on screen, so exploration
    links and topic images (the slow, optional parts) can never delay the
    answer. Everything it produces is patched into the same *answer_id* and
    tagged with the same *capture_id*, so it enriches its own answer or is
    rejected — it can never land on a newer one.
    """
    started = time.time()
    try:
        enriched = list(links)
        if topic:
            for fl in build_explore_links(topic):
                if len(enriched) >= 5:
                    break
                if not any(l.get("url") == fl.get("url") for l in enriched):
                    enriched.append(fl)
            if time.time() - started > SCREEN_ENRICH_TIMEOUT:
                # Out of budget: ship the links we already have and skip the
                # slow image phase rather than discarding the work.
                logging.debug("[SCREEN] enrichment deadline — links only")
                show_images = False
        images = []
        if show_images and topic:
            try:
                images = fetch_topic_images(topic, max_images=2) or []
            except Exception as exc:
                logging.debug("[SCREEN] image enrichment failed: %s", exc)
                images = []
        if len(enriched) == len(links) and not images:
            return  # nothing new to add
        push_screen_answer(
            tip, evidence, enriched, images, region=region,
            answer_id=answer_id, request_id=request_id, capture_id=capture_id,
            capture_seq=capture_seq,
        )
    except Exception as exc:
        logging.debug("[SCREEN] enrichment failed: %s", exc)


# ── Screen Q&A background thread support ──
_screen_qa_callback = None
_screen_qa_busy = threading.Lock()


def set_screen_qa_callback(cb):
    """Register a callback(text) invoked when a background screen Q&A completes."""
    global _screen_qa_callback
    _screen_qa_callback = cb


# ── R13 speech gateway: action claims need a verified event ──────────────
# Strict mode (user-approved): a sentence claiming Jarvis DID / WILL / IS
# doing a real-world action may only be spoken when it is backed by an
# authoritative operational event — a verified TaskResult, a worker
# acknowledgement, a durable queue acceptance, or a delivered preview.
# Tool-less chat has ZERO action authority: it may chit-chat, explain, or
# ask for detail, but never promise/claim an action. Anything it emits that
# looks like an action claim is replaced with an honest redirect.
#
# Narrator roles (Astra §5): preview proposes+asks, scheduler announces
# queue-after-ack, runner announces started-after-ack, result announces
# done-after-verification, status describes a live snapshot. Chat is none
# of these, so chat-shaped text can never carry those sentences.
_ACTION_CLAIM_RE = re.compile(
    r"\b("
    r"i\s+will\s+(get|have|create|make|check|do|run|start|open|send|fetch|look)|"
    r"i(?:'m| am)\s+(?:right\s+)?on\s+it|"
    r"(?:it(?:'s| is)\s+)?(?:done|created|ready|finished|taken\s+care\s+of)|"
    r"consider\s+it\s+done|"
    r"right\s+away|"
    r"has\s+been\s+(created|deleted|removed|completed|finished|started)|"
    r"have\s+been\s+(created|deleted|removed)|"
    r"file\s+(?:has\s+been\s+)?created|"
    r"folder\s+(?:has\s+been\s+)?created|"
    r"(?:started|starting|running|executing|working\s+on)\s+(?:the|that|your|this)?\s*(task|file|folder|browser|job|request)|"
    r"(?:queued|in\s+the\s+queue)|"
    r"taking\s+over\s+the\s+browser\s+task|"
    r"handing\s+the\s+task\s+to|"
    # R14: optimistic pre-execution acks spoken BEFORE any worker ack —
    # "On it", "Playing now", "Opening/Checking/Searching/Looking ... now",
    # "Navigating to ...", "I've opened ... for you".
    r"on\s+it|"
    r"(?:playing|opening|checking|searching|looking|running|navigating)\s+.*\bnow\b|"
    r"(?:playing|opening|checking|searching|looking|running|navigating)\b|"
    r"back\s+in\s+a\s+moment|"
    r"(?:i(?:'ve| have)\s+)?(?:opened|started)\s+.*\bfor\s+you\b"
    r")\b",
    re.IGNORECASE,
)

# Sentences that merely offer to act or describe capability are NOT claims:
# "I can create files", "Do you want me to...", "Say the word and I will".
_ACTION_OFFER_RE = re.compile(
    r"\b(i\s+can\s+(create|make|check|list|open|run|help|do)|"
    r"do\s+you\s+want\s+me\s+to|"
    r"shall\s+i\s+|"
    r"say\s+the\s+word|"
    r"let\s+me\s+know|"
    r"what\s+(specific|exactly)|"
    r"which\s+(folder|file|folder)|"
    r"ask\s+for\s+the\s+missing|"
    r"nothing\s+was\s+started|"
    r"what\s+would\s+you\s+like)\b",
    re.IGNORECASE,
)

# R13 chat-safe fallback: states what happened (nothing) + what is needed.
# Never an action sentence — verified by _ACTION_CLAIM_RE below at def time.
_CHAT_NO_ACTION_FALLBACK = (
    "Understood, sir. Nothing was started — "
    "which exact folder and file name should I use?"
)


def _strip_unverified_action_claims(text, role):
    """R13 gate: drop action-claim sentences `role` has no authority to speak.

    *role* is one of preview/scheduler/runner/result/status/chat.
    Chat has no action authority at all: any claim-shaped sentence is CUT
    and replaced with the honest fallback. The other narrators keep their
    own authoritative sentences (they are constructed from verified events
    upstream) — this gate only strips recognised action claims that leaked
    in from free-form model prose.
    """
    if not text:
        return text
    if role != "chat":
        return text
    sentences = re.split(r"(?<=[.!?])\s+", str(text).strip())
    kept = [
        s for s in sentences
        if not _ACTION_CLAIM_RE.search(s) or _ACTION_OFFER_RE.search(s)
    ]
    # If every sentence was a claim (the classic "I will get that created
    # right away, sir." turn), say the honest fallback instead of silence.
    if not kept or not any(s.strip() for s in kept):
        logging.warning("[R13] chat action claim blocked: %r", text[:160])
        return _CHAT_NO_ACTION_FALLBACK
    if len(kept) != len(sentences):
        logging.warning("[R13] chat action claim stripped: %r", text[:160])
    return " ".join(kept).strip()


# ── Generic async-reply callback (opencode fallback results, etc.) ──
_async_reply_callback = None


def set_async_reply_callback(cb):
    """Register callback(text) invoked when a background task finishes with a final reply."""
    global _async_reply_callback
    _async_reply_callback = cb


def _notify_async_reply(text, spoken=None):
    """Deliver a background-task final reply to the registered callback.

    `spoken` overrides what is said when it differs from the UI text — the
    opencode completion speaks a fixed phrase while the full summary stays
    in the chat log.

    F10: returns True only when the reply actually reached a live surface.
    The commitment outbox uses this as its acknowledgement, so "no callback
    registered" and "the callback blew up" are both failures to deliver —
    a reminder that nobody heard must not be recorded as delivered.
    """
    # G9 (F07): every background terminal reply lands in the bounded,
    # secret-masked work-event store so future questions can ground in it.
    if not text:
        return False
    try:
        if memory_store is not None:
            memory_store.record_event("async_result", text)
    except Exception:
        pass
    if _async_reply_callback is None:
        return False
    try:
        _async_reply_callback(text, spoken)
    except Exception as exc:
        logging.warning("[NOTIFY] async reply callback failed: %s", exc)
        return False
    # [S13] A delivered background result is now part of the conversation:
    # follow-ups like "what was the second point?" or "open that" ground in
    # history instead of pointing at something the model never saw. The
    # marker rides in-band (the prompt shows it as a background result, not
    # a turn answer); only DELIVERED results are remembered. A repeat of the
    # immediately-preceding entry (same text, e.g. a completion re-firing)
    # is NOT appended again — seven identical "folder has been created"
    # turns once crowded real conversation out of the model's window and
    # broke "what did we talk about in the last six messages" counting.
    try:
        entry = "[background result] %s" % text
        try:
            recent = get_history()[-1:]
        except Exception:
            recent = []
        if not (recent and recent[0].get("content") == entry):
            add_message("assistant", entry)
    except Exception as exc:
        logging.warning("[NOTIFY] background result history write failed: %s",
                        exc)
    return True


# G9 (F10): commitment delivery rides the SAME event/UI/speech path as
# background task results — the interruption-aware suppression guards in
# routes.py apply, and delivery is a notification only (never an action).
def _deliver_commitment(commitment):
    """F10 — the outbox acknowledges delivery ONLY when this returns True.

    Raising (or returning False) leaves the reminder armed with backoff, so a
    transient failure is retried instead of being recorded as delivered.
    """
    if not commitment:
        raise ValueError("no commitment to deliver")
    text = "Sir, reminder: %s" % (commitment.get("text") or "your commitment")
    if not _notify_async_reply(text):
        raise RuntimeError("notification surface refused the reminder")
    return True


try:
    if memory_store is not None:
        memory_store.set_commitment_delivery(_deliver_commitment)
        # F10: the scheduler belongs to the BACKEND, not to the act of adding
        # a reminder — this is what makes a restart deliver stored reminders.
        memory_store.start_scheduler()
except Exception:
    pass



def _coerce_task_result(value):
    """Accept a TaskResult or a legacy plain-string engine output.

    F03: a bare string can no longer be promoted to 'completed'. Only text
    that names an actual effect is verified; a refusal, a bare "Done.", a
    report whose own wording carries a failure, and anything else without
    evidence become 'partial' (or 'failed'/'stopped' where the wording is
    unambiguous). TaskResult values pass through untouched.
    """
    if isinstance(value, TaskResult):
        return value
    text = value if isinstance(value, str) else ("" if value is None else str(value))
    if not text:
        return TaskResult.failed("internal crash")
    if text.strip() == "Stopped per your request.":
        return TaskResult.stopped()
    if is_failure_text(text):
        err = text.split("Error:", 1)[1].strip() if "Error:" in text else text
        return TaskResult.failed(err or text, detail=text)
    return result_from_reported_text(text, detail=text)


def _summarize_opencode_output(output, status=None, error=None):
    """Turn raw opencode CLI output into a short, spoken-style confirmation.

    opencode prints things like `Created 'C:\...\folder'.` or
    `Error: access denied` — not ideal to read verbatim. This keeps the
    confirmation natural ("Sir, the folder has been created.") while
    preserving the key detail as a short quote when useful.

    status/error (F03): when the engine reports 'failed', success is NEVER
    inferred from 'created'/'done' substrings — the reply stays an honest
    problem report, including the short engine reason when available.
    status=None preserves the historical behavior for other callers.
    """
    out = (output or "").strip()
    if status == "failed":
        err = re.sub(r"\s+", " ", (error or "").strip())
        if err:
            if len(err) <= 160:
                return "Sir, there was a problem completing that task: %s" % err
            return "Sir, there was a problem completing that task."
        if not out:
            return "Sir, I couldn't complete that task."
        return "Sir, there was a problem completing that task."
    if status == "partial":
        # Future-proof honesty branch (G1 re-audit): a partial opencode
        # result must never take the success-inference path below.
        if not out:
            return "Sir, the task is only partly done."
        if len(out) <= 160:
            return "Sir, the task is only partly done: %s" % out
        return "Sir, the task is only partly done: %s..." % out[:140]
    if not out:
        if status == "completed":
            # Empty-but-successful engine output: still confirm completion.
            return "Sir, that has been done."
        return "Sir, I couldn't complete that task."

    lowered = out.lower()
    if "created" in lowered or "done" in lowered or "success" in lowered:
        if "folder" in lowered or "directory" in lowered:
            return "Sir, the folder has been created."
        if "file" in lowered:
            return "Sir, the file has been created."
        return "Sir, that has been done."
    if any(word in lowered for word in ("deleted", "deleting", "removed", "removing", "remove-item", "delete")):
        if "folder" in lowered or "directory" in lowered:
            return "Sir, the folder has been deleted."
        if "file" in lowered:
            return "Sir, the file has been deleted."
        return "Sir, that has been removed."
    if "new-item" in lowered or "directory" in lowered:
        return "Sir, the folder has been created."
    if any(word in lowered for word in ("error", "failed", "couldn't", "cannot", "denied")):
        return "Sir, there was a problem completing that task."
    if len(out) <= 160:
        return f"Sir, here's what happened: {out}"
    return f"Sir, here's what happened: {out[:140]}..."


# ── Browser-task short summary (Feature A) ──
# The browser agent's final summary can run to thousands of chars. Speaking
# it (and dumping it as the headline chat bubble) is the "REALLY long reply"
# complaint — mirror the opencode pattern: a 2-3 line / <300 char headline
# for speech + chat, full output kept in the detail store below.
_last_browser_full_output = ""
_browser_detail_lock = threading.Lock()


def _set_last_browser_detail(output):
    global _last_browser_full_output
    with _browser_detail_lock:
        _last_browser_full_output = output or ""


def get_last_browser_full_output():
    """Full untrimmed browser-task output (detail path behind the summary)."""
    with _browser_detail_lock:
        return _last_browser_full_output


def _summarize_browser_output(output, task_description="", status=None, evidence=None):
    """Deterministic heuristic: 2-3 line, <=300 char spoken/headline summary.

    Outcome (success/fail) + what was done (task site/action) + key result
    (first meaningful sentence of the agent output). No LLM call, no new
    provider dependency — prompt JSON + downstream parsing already handle
    the detail; this is only the headline.

    status (F03): drives the outcome prefix — 'partial' NEVER yields
    'Sir, done' (honest partly-done headline instead), 'failed' NEVER
    yields a success headline. status=None preserves the historical
    behavior for other callers.
    """
    out = (output or "").strip()
    if status == "partial":
        flat = re.sub(r"\s+", " ", out)
        key = ""
        for part in re.split(r"(?<=[.!?])\s+", flat):
            if len(part.strip()) >= 10:
                key = part.strip()
                break
        if not key:
            key = flat or "no summary was produced"
        task = re.sub(r"\s+", " ", task_description or "").strip()
        if len(task) > 90:
            task = _truncate_at_word(task, 90) + "..."
        failed_bits = ""
        try:
            count = len(evidence or [])
        except Exception:
            count = 0
        if count:
            failed_bits = " %d step(s) failed." % count
        if task:
            base = "Sir, the task is only partly done — %s. %s%s" % (task, key, failed_bits)
        else:
            base = "Sir, the task is only partly done. %s%s" % (key, failed_bits)
        if len(base) > 300:
            budget = max(40, 300 - len(base) + len(key) - 3)
            key = _truncate_at_word(key, budget) + "..."
            if task:
                base = "Sir, the task is only partly done — %s. %s%s" % (task, key, failed_bits)
            else:
                base = "Sir, the task is only partly done. %s%s" % (key, failed_bits)
            if len(base) > 300:
                base = base[:297] + "..."
        lines = base.splitlines()
        if len(lines) > 3:
            base = " ".join(lines[:3])
        return base
    if not out:
        return "Sir, the task could not be completed."
    if status == "failed" and not out.startswith("TASK NOT COMPLETED"):
        reason = re.sub(r"\s+", " ", out)
        if len(reason) > 180:
            reason = _truncate_at_word(reason, 180) + "..."
        summary = "Sir, the task could not be completed. %s" % reason
        return summary if len(summary) <= 300 else summary[:297] + "..."
    if out.startswith("TASK NOT COMPLETED"):
        reason = out.split("Error:", 1)[1].strip() if "Error:" in out else out
        reason = re.sub(r"\s+", " ", reason)
        if len(reason) > 180:
            reason = _truncate_at_word(reason, 180) + "..."
        summary = "Sir, the task could not be completed. %s" % reason
        return summary if len(summary) <= 300 else summary[:297] + "..."
    flat = re.sub(r"\s+", " ", out)
    key = ""
    for part in re.split(r"(?<=[.!?])\s+", flat):
        if len(part.strip()) >= 10:
            key = part.strip()
            break
    if not key:
        key = flat
    task = re.sub(r"\s+", " ", task_description or "").strip()
    if len(task) > 90:
        task = _truncate_at_word(task, 90) + "..."
    base = "Sir, done — %s. %s" % (task, key) if task else "Sir, done. %s" % key
    if len(base) > 300:
        budget = max(40, 300 - len(base) + len(key) - 3)
        key = _truncate_at_word(key, budget) + "..."
        base = "Sir, done — %s. %s" % (task, key) if task else "Sir, done. %s" % key
        if len(base) > 300:
            base = base[:297] + "..."
    lines = base.splitlines()
    if len(lines) > 3:
        base = " ".join(lines[:3])
    return base


def detect_language(text):
    if any("\u0900" <= char <= "\u097F" for char in text):
        return "hindi"

    hindi_words = [
        "kya",
        "hai",
        "hain",
        "mujhe",
        "mera",
        "meri",
        "aap",
        "tum",
        "karo",
        "karna",
        "bolo",
        "batao",
        "samjho",
        "dekho",
        "suno",
        "kyun",
        "kab",
        "kaise",
        "kaun",
        "kitna",
        "yahan",
        "wahan",
        "theek",
        "accha",
        "haan",
        "nahi",
        "bilkul",
        "zaroor",
        "abhi",
        "baad",
        "pehle",
        "aaj",
        "kal",
        "subah",
        "shaam",
    ]
    words = text.lower().split()
    hindi_count = sum(1 for word in words if word in hindi_words)

    hindi_phrases = [
        "kya hai",
        "kaise ho",
        "batao",
        "search karo",
        "open karo",
        "chalu karo",
        "band karo",
        "aaj ka",
        "abhi ka",
    ]
    has_hindi_phrase = any(phrase in text.lower() for phrase in hindi_phrases)
    return "hindi" if hindi_count >= 2 or has_hindi_phrase else "english"


# Fresh-info detection. Whole-word matching only: the old substring test
# wrongly routed normal conversation to web research ("now" in "know", "fee"
# in "feel", "cost" in "costume", "match" in "matches", "score" in "scored").
_SEARCH_WORDS = frozenset({
    "latest",
    "current",
    "price",
    "pricing",
    "priced",
    "cost",
    "costs",
    "fee",
    "fees",
    "subscription",
    "score",
    "match",
    "news",
    "weather",
    "taaza",
    "taza",
    "mausam",
    "khabar",
})

# Recency words are *weak*: a bare "today"/"now" is not enough, so "how do you
# feel today?", "what do you know now?" and "aaj kya karu" stay plain chat.
_SEARCH_WEAK_WORDS = frozenset({"today", "now", "aaj", "abhi"})

# Nouns that turn a weak recency word into a genuine lookup ("news today").
_SEARCH_FACT_NOUNS = frozenset({
    "news",
    "price",
    "pricing",
    "cost",
    "costs",
    "fee",
    "fees",
    "subscription",
    "score",
    "match",
    "weather",
    "release",
    "releases",
    "rate",
    "rates",
    "stock",
    "market",
    "result",
    "results",
    "update",
    "updates",
    "version",
    "schedule",
    "forecast",
    "temperature",
    "headline",
    "headlines",
    "mausam",
    "khabar",
    "taaza",
    "taza",
})

_SEARCH_HOW_MUCH_RE = re.compile(r"\bhow much (?:does|is)\b")
_SEARCH_WORD_RE = re.compile(r"[a-z0-9']+")


def should_search(query):
    """True when the query asks for fresh, current world information.

    Matching is whole-word, and recency words only count when a world-fact
    noun is also present, so conversational questions ("what do you know
    about X", "how do you feel today?", "I feel tired", "do you know me")
    are never mistaken for a request to search the web.
    """
    q = query.lower()
    words = set(_SEARCH_WORD_RE.findall(q))
    if words & _SEARCH_WORDS:
        return True
    if _SEARCH_HOW_MUCH_RE.search(q):
        return True
    return bool(words & _SEARCH_WEAK_WORDS and words & _SEARCH_FACT_NOUNS)


# Greeting-like chat that must never auto-reroute to research.
_GREETING_LIKE_RE = re.compile(
    r"how are you|how.?s it going|what.?s up|kaise ho|kya haal|"
    r"good (morning|afternoon|evening|night)",
    re.IGNORECASE,
)

# Question-shaped chat (optional hey/ok/hi/hello/jarvis fillers, then an
# interrogative — "whats" counts as "what"). Non-question chat stays chat.
_QUESTION_SHAPED_RE = re.compile(
    r"^(?:(?:hey|ok|hi|hello|jarvis)[,.\s]+)*"
    r"(what|who|when|where|which|whose|why|how|is|are|do|does|can|could|"
    r"will|would|tell|kya|kaun|kab|kitna|kitne|kaisa)(?:'s|s)?(?=\s|$|[?])",
    re.IGNORECASE,
)


def force_search(query):
    q = query.lower()
    triggers = [
        "look it up",
        "search it",
        "find this",
        "dhundho",
        "search karo",
        "khojo",
    ]
    return any(trigger in q for trigger in triggers)


_RESEARCH_TRIGGERS = [
    "look this up", "look that up", "look it up", "look up",
    "find out about", "find out more", "find about",
    "search this on the internet", "search that on the internet",
    "search it on the internet", "search the internet", "search the web",
    "search internet", "search on internet", "internet search",
    "research this", "research that", "research it",
    "google it", "internet pe search", "internet par search",
    "dhundh ke batao", "dhundh ke dikhao", "khoj ke batao", "search kar ke",
    # Tiered search: the explicit deepsearch keyword always counts as an
    # explicit research request (handled at brain level, never in intent.py).
    "deepsearch",
]
_RESEARCH_PREFIXES = ("research ", "deepsearch ")

# Common speech-to-text mishearings of "search" (e.g. "sardi internet").
_RESEARCH_SEARCH_ALIASES = [
    "sardi", "sardhi", "sarch", "serch", "surch", "seach",
    "searcha", "sarsho", "sarchi", "sherch", "sarech",
]


def force_research(query):
    """True when the user explicitly asked Jarvis to look something up online."""
    q = (" " + query.strip().lower() + " ")
    for alias in _RESEARCH_SEARCH_ALIASES:
        q = q.replace(" " + alias + " ", " search ")
    if any(trigger in q for trigger in _RESEARCH_TRIGGERS):
        return True
    stripped = query.strip().lower()
    if any(stripped.startswith(p) for p in _RESEARCH_PREFIXES):
        return True
    return False


# ── Live fix: search-shaped requests + offered-search follow-ups ─────────────
# "can you search about this stream", "find anything about that stream",
# "don't ask any questions, just search it" and "execute it" used to fall to
# chat, which could only keep asking. A search-shaped message is a search;
# "execute it / just do it" consumes the search Jarvis last offered.

_SEARCH_SHAPED_VERB_RE = re.compile(
    r"\b(search|searching|look\s+(?:it|this|that|them)\s+up|look\s+up|"
    r"google|research|find\s+(?:anything|something|it|this|that|out))\b",
    re.IGNORECASE,
)
_SEARCH_TOOL_GUARD_RE = re.compile(
    r"\b(open|launch|play|pause|close|go\s+to|navigate|visit|click|type|"
    r"press|scroll|switch)\b",
    re.IGNORECASE,
)
_SEARCH_APP_GUARD_RE = re.compile(
    r"\b(?:in|on|using|with)\s+(?:chrome|edge|brave|the\s+browser)\b",
    re.IGNORECASE,
)
_SEARCH_HOWTO_GUARD_RE = re.compile(
    r"^\s*(?:how\s+(?:do|to|can|would|should|does)\b|"
    r"what\s+is\s+the\s+best\s+way\b|can\s+you\s+(?:teach|show|tell)\b)",
    re.IGNORECASE,
)
_SEARCH_PAST_GUARD_RE = re.compile(r"^\s*(?:did|have|has|had|why\s+did)\b",
                                   re.IGNORECASE)
_SEARCH_FUTURE_GUARD_RE = re.compile(
    r"^\s*(?:i\s+(?:will|'ll|would|might|could|may)|maybe\s+i|"
    r"we\s+(?:will|'ll))\b",
    re.IGNORECASE,
)


def is_search_shaped_message(msg):
    """True when the message asks for a web search, however it is phrased.

    Deliberately conservative: how-to questions, past/future talk, and
    tool-style phrases ("open youtube and search…") keep their normal route.
    """
    t = str(msg or "").strip()
    if not t:
        return False
    if (_SEARCH_HOWTO_GUARD_RE.match(t) or _SEARCH_PAST_GUARD_RE.match(t)
            or _SEARCH_FUTURE_GUARD_RE.match(t)):
        return False
    if _SEARCH_TOOL_GUARD_RE.search(t) or _SEARCH_APP_GUARD_RE.search(t):
        return False
    return bool(_SEARCH_SHAPED_VERB_RE.search(t))


_NO_ASK_RE = re.compile(
    r"\b(?:don'?t|dont|do\s+not|no)\s+(?:ask|questions?)\b"
    r"|\bno\s+questions?\b|\bwithout\s+asking\b",
    re.IGNORECASE,
)

_EXECUTE_IT_RE = re.compile(
    r"^\s*(?:ok(?:ay)?\s*[,.]?\s*)?(?:now\s+|just\s+|pl[sz]\s+)?"
    r"(?:execute(?:\s+it)?|do\s+it|run\s+it|go\s+ahead|"
    r"(?:just\s+)?search\s+(?:it|for\s+it)|kar\s*do|kar\s*de|"
    r"chala\s*do)\b",
    re.IGNORECASE,
)

#: A chat reply that OFFERS a search ("I can search that…", "shall I look
#: it up?"). Matches both word orders.
_OFFER_RE = re.compile(
    r"\b(?:shall|should|would\s+you\s+like|do\s+you\s+want|want\s+me\s+to|"
    r"i\s+can|i\s+could|let\s+me\s+know\s+if)\b[^.!?]{0,90}"
    r"\b(?:search|searched|research|google|find|look\s+up)\b"
    r"|\b(?:search|searched|research|google|find|look\s+up)\b"
    r"[^.!?]{0,90}"
    r"\b(?:shall|would\s+you\s+like|want\s+me\s+to|do\s+you\s+want)\b",
    re.IGNORECASE,
)

_offer_lock = threading.Lock()
_last_offer = None
_OFFER_TTL = 180.0


def _search_request_query(msg):
    """The concrete web query for a search-shaped message, or "".

    Deictic messages ("just search it", "about this stream") resolve from
    the screen/entity ledger; concrete messages go through the normal
    derivation. The pointer text is never returned as the query.
    """
    cleaned = _NO_ASK_RE.sub(" ", str(msg or "")).strip(" ,.").strip()
    return _resolve_search_query(cleaned or msg)


def _handle_search_shaped(msg, from_voice=False, voice_compact=False):
    """Route a search-shaped request to research with a real query."""
    query = _search_request_query(msg)
    if not query:
        return ("Sir, I could not tell what to search for — name it once "
                "and I will search immediately.")
    print("[SEARCH] Shaped request -> research:", query)
    return handle_research_intent(query, from_voice=from_voice,
                                  voice_compact=voice_compact, derived=True)


def _remember_chat_offer(user_msg, reply):
    """Remember a search Jarvis OFFERED but has not started. Never raises."""
    global _last_offer
    try:
        text = str(reply or "")
        if not text or not _OFFER_RE.search(text):
            return
        query = _resolve_reference_query(user_msg)
        with _offer_lock:
            _last_offer = {
                "kind": "search",
                "text": str(user_msg or ""),
                "query": query,
                "at": time.time(),
                "expires": time.time() + _OFFER_TTL,
            }
        print("[OFFER] Search offered — awaiting go-ahead")
    except Exception:
        pass


def _consume_offered_action(msg):
    """Run the last offered search when the user says "execute it"."""
    global _last_offer
    try:
        t = str(msg or "")
        wants_execute = bool(_EXECUTE_IT_RE.search(t))
        no_ask_search = bool(_NO_ASK_RE.search(t)
                             and _SEARCH_SHAPED_VERB_RE.search(t))
        if not (wants_execute or no_ask_search):
            return None
        with _offer_lock:
            offer = dict(_last_offer) if _last_offer else None
        if not offer:
            return None
        if time.time() > float(offer.get("expires") or 0.0):
            with _offer_lock:
                _last_offer = None
            return None
        with _offer_lock:
            _last_offer = None
        if offer.get("kind") != "search":
            return None
        query = str(offer.get("query") or "")
        if not query:
            query = _search_request_query(str(offer.get("text") or ""))
        if not query:
            return ("Sir, I still need the name — say it once and I will "
                    "search immediately.")
        print("[OFFER] Executing offered search:", query)
        return handle_research_intent(query, derived=True)
    except Exception as exc:
        logging.warning("[OFFER] consume failed: %s", exc)
        return None


# ── [PERF] deterministic "definitely plain chat" fast path ───────────────────
# The intent classifier is a cloud round trip (one 3-hop call with a 3.5s
# budget) that sits in front of EVERY message, and nothing it produces is
# consumed until it returns — so it also gates the first streamed token even
# when the speculative chat stream is already producing text.
#
# For the large class of messages that are obviously conversation, the verdict
# is already `chat`, and every deterministic net that can UPGRADE a chat
# verdict (screen question, fresh-info search, tool steps, research, task,
# web-shaped task) still runs afterwards. Skipping the classifier for those
# messages therefore removes a network round trip from the hot path without
# removing a single routing decision.
#
# This is deliberately conservative: anything with a plausible signal for
# another route returns False and still goes to the classifier.

#: Verbs that mean "do something", not "talk to me".
_CHAT_ACTION_RE = re.compile(
    r"\b(open|launch|run|execute|create|make|delete|remove|install|download|"
    r"search|browse|go to|visit|type|click|press|play|pause|stop|close|"
    r"kill|rename|move|copy|write|read|edit|fix|build|deploy|send|email|"
    r"kholo|chalu|banao|chalao|band|khatam)\b",
    re.IGNORECASE,
)

#: Nouns that point at the world outside a conversation.
_CHAT_OBJECT_HINT_RE = re.compile(
    r"(https?://|www\.|\.com\b|\.in\b|\.org\b|\.io\b|"
    r"\b(file|folder|directory|app|application|website|browser|chrome|edge|"
    r"brave|youtube|gmail|notepad|vscode|terminal|cmd|powershell|excel|"
    r"word|calculator|spotify|netflix)\b)",
    re.IGNORECASE,
)

#: First-person / second-person framing, the strongest chat signal there is.
_CHAT_PERSON_RE = re.compile(
    r"\b(i|me|my|mine|you|your|yours|we|us|our|am i|do i|did i|have i|"
    r"should i|who are you|who r u|are you|can you|could you|would you|"
    r"will you|thank|thanks|hello|hi|hey|goodbye|bye|jarvis|sir|"
    r"good (?:morning|afternoon|evening|night)|kaise ho|kya haal|"
    r"kya hua|batao|how.?s it going|how is it going|how are you doing|"
    r"sup\b|namaste)\b",
    re.IGNORECASE,
)

#: Utterances longer than this stop being "obviously chat" — a long message is
#: far more likely to be a task description or a research question.
_CHAT_FASTPATH_MAX_WORDS = 12

#: Deterministic phrase routes owned by the G9 memory store. These are resolved
#: before the classifier is ever reached, so the fast path must never apply to
#: them — the guard is belt-and-braces, not the primary mechanism.
_CHAT_MEMORY_RE = re.compile(
    r"\b(remember|forget|remind|reminder|recall|unforget|approve the|"
    r"retire the|skill)\b",
    re.IGNORECASE,
)


#: [PERF] Classifier budget per origin. classify_intent degrades to a `chat`
#: verdict when the budget expires, and the deterministic nets still run, so a
#: smaller spoken budget costs nothing but bounds the worst case.
INTENT_BUDGET_MS = int(os.getenv("JARVIS_INTENT_BUDGET_MS", "3500"))
INTENT_BUDGET_VOICE_MS = int(os.getenv("JARVIS_INTENT_BUDGET_VOICE_MS", "1200"))


def _fastpath_chat_enabled():
    """Kill switch for the deterministic chat fast path.

    Read per call so ``JARVIS_CHAT_FASTPATH=0`` disables it without a restart,
    the same escape hatch the routing mode flag uses. Any misroute found in
    testing can be turned off immediately while a fix is written.
    """
    return os.getenv("JARVIS_CHAT_FASTPATH", "1").strip() not in ("0", "false",
                                                                 "off", "no")


def _mark_latency(request_id, name, meta=None):
    """[PERF] P1-19 — record one ABSOLUTE boundary mark for *request_id*.

    Deliberately not a duration: a pre-computed duration stored next to
    offsets is what made the telemetry ring's numbers un-addable. The boundary
    is marked here; the step's cost is derived from consecutive marks when the
    waterfall is read. No-op when the turn is unknown, and never raises.
    """
    if not request_id:
        return
    try:
        from backend.services import latency as _lat
        _lat.mark(request_id, name, meta=meta)
    except Exception:
        pass


def _mark_latency_duration(request_id, name, started_ns, meta=None):
    """[PERF] P1-19 / [P0-10] — close a PRE-ROUTE PREDICATE step.

    The waterfall derives every step from consecutive ABSOLUTE marks — the ring
    deliberately stores no step durations (see ``backend/services/latency.py``)
    — so what the timeline needs is the predicate's END boundary, which is what
    is recorded here. The predicate's measured cost rides along as
    ``predicate_ms`` metadata so the numbers P0-10 asked to report are readable
    straight off the record instead of having to be re-derived from deltas.

    No-op when the turn is unknown, and never raises: telemetry must not be able
    to break routing.
    """
    if not request_id:
        return
    try:
        elapsed_ms = (time.perf_counter_ns() - int(started_ns)) / 1_000_000.0
        payload = {"predicate": name, "predicate_ms": round(elapsed_ms, 3)}
        if meta:
            payload.update(meta)
        from backend.services import latency as _lat
        _lat.mark(request_id, name, meta=payload)
    except Exception:
        pass


def _bind_latency_request(request_id):
    """[PERF] P1-19 — name the turn that untagged marks belong to.

    The provider stream clients (``provider_headers``) and the audio actor
    (``tts_first_byte`` / ``playback_started``) observe their boundary from a
    worker thread that carries no transport identity, so the turn is named
    here for the duration of the turn.
    """
    try:
        from backend.services import latency as _lat
        _lat.set_active_request(request_id)
    except Exception:
        pass


def _release_latency_request(request_id):
    """[PERF] P1-19 — stop attributing untagged marks to a finished turn."""
    try:
        from backend.services import latency as _lat
        if _lat.active_request() == str(request_id or ""):
            _lat.set_active_request("")
    except Exception:
        pass


#: [P0-10] The speculative racer started for the turn running on THIS thread.
#: ``process_message`` owns its lifetime in a ``finally``, so an early return
#: anywhere in the routing chain (memory phrase, a confirmation answer, a task
#: or screen-control verdict, research, …) cannot leak a background stream.
_turn_racer = threading.local()


def _register_turn_racer(racer):
    """[P0-10] Hand *racer* to this turn's cleanup. Never raises."""
    try:
        _turn_racer.racer = racer
    except Exception:
        pass


def _cancel_orphan_turn_racer():
    """[P0-10] Cancel a speculation that no route adopted.

    Runs from ``process_message``'s ``finally``, i.e. exactly once per finished
    turn, on the thread that ran it. Idempotent: an already-cancelled racer is
    simply cancelled again, and an ADOPTED one is left alone because it is the
    reply being streamed. Never raises.
    """
    racer = getattr(_turn_racer, "racer", None)
    try:
        _turn_racer.racer = None
    except Exception:
        pass
    if racer is None:
        return
    try:
        if getattr(racer, "is_adopted", False):
            return
    except Exception:
        pass
    try:
        racer.cancel()
    except Exception:
        pass



def is_definitely_plain_chat(msg):
    """True when *msg* is plain conversation, so the classifier can be skipped.

    Returns a `chat` verdict from :func:`classify_intent`'s shape, which the
    caller treats exactly like a classifier answer: every deterministic net
    that can upgrade it still runs. This function may only ever say "skip the
    network call, it would have said chat" — never "skip the nets".
    """
    if not msg or not msg.strip():
        return False
    text = msg.strip()
    if text.lower().startswith("command"):
        return False
    # Deterministic phrase routes (memory / commitment / skill ops) resolve
    # before this point, but the fast path must never be *why* one of them was
    # skipped, so those verbs always take the classifier.
    if _CHAT_MEMORY_RE.search(text):
        return False
    # Any signal another route could claim wins over the fast path.
    try:
        if is_screen_question(msg) or is_region_question(msg):
            return False
    except Exception:
        return False
    try:
        if force_research(msg) or force_search(msg) or should_search(msg):
            return False
    except Exception:
        return False
    try:
        if is_explicit_task_request(msg) or is_code_tool_request(msg):
            return False
    except Exception:
        return False
    if _CHAT_ACTION_RE.search(text) or _CHAT_OBJECT_HINT_RE.search(text):
        return False
    if len(text.split()) > _CHAT_FASTPATH_MAX_WORDS:
        return False
    if not _CHAT_PERSON_RE.search(text):
        return False
    # "who/where/when" as an OPENING word usually asks about the world ("who
    # won", "where is Berlin", "when is the match") and belongs to the
    # classifier. The exception is a question aimed at the assistant itself
    # ("who are you"), which the person regex already matched.
    if re.match(r"^\s*(?:who|where|when)\b", text, re.IGNORECASE) \
            and not re.match(r"^\s*who\s+(?:are|r)\s+you\b", text,
                             re.IGNORECASE):
        return False
    return True


def derive_research_query(raw_query):
    """Turn the user's raw sentence into the actual web query.

    The full utterance ("jarvis i am about to study a .net course at my company
    in the training sessions so i want you to find out about that course") must
    never be typed verbatim into the browser — the LLM intent router pulls out
    the true subject ("find out about ... that course" -> ".net course training
    session"). A local heuristic is the fallback when the router is offline.
    """
    if not raw_query or not raw_query.strip():
        return raw_query
    # Live fix: "you searched a different man by that same name, search the
    # youtube content creator" refines the last researched NAME.
    refined = _refine_last_name_query(raw_query)
    if refined:
        print(f"[RESEARCH] Refined to last name: {refined!r} <- {raw_query!r}")
        return refined
    reaction = _reaction_query(raw_query)
    if reaction:
        return reaction
    query = None
    meta_candidate = False
    try:
        intent = classify_intent(raw_query, timeout_ms=3500)
        if intent.get("intent") in ("research", "tools"):
            candidate = str(intent.get("query") or "").strip()
            if candidate and candidate != raw_query.strip():
                if _META_QUERY_RE.search(candidate):
                    # Self-referential text is not a subject: asking beats
                    # searching "the name previously searched".
                    meta_candidate = True
                    print(f"[RESEARCH] Rejected meta query {candidate!r}")
                else:
                    query = candidate[:220]
                    print(f"[RESEARCH] Derived query: {query!r} <- {raw_query!r}")
    except Exception as exc:
        logging.warning("[RESEARCH] Query derivation failed: %s", exc)
    if not query:
        if meta_candidate:
            print(f"[RESEARCH] Meta-only request has no concrete subject: "
                  f"{raw_query!r}")
            return ""
        query = _heuristic_research_query(raw_query)
    if _is_reference_only(query) and _last_research_topic:
        print(
            f"[RESEARCH] Bare reference ({query!r}) - "
            f"carrying over last topic {_last_research_topic!r}"
        )
        return _last_research_topic
    if _is_deictic_query(query):
        resolved = _resolve_reference_query(query)
        if resolved:
            print(f"[RESEARCH] Deictic query ({query!r}) -> {resolved!r}")
            return resolved
        print(f"[RESEARCH] Deictic query ({query!r}) has no referent")
        return ""
    return _carry_context_subject(raw_query, query)


# Last successfully researched topic, so a Hinglish consent reply such as
# "my internet search karke batao iske bare mein" or "just type what i said"
# can fall back to the topic from the original message instead of being
# searched verbatim.
_last_research_topic = None


def _set_last_research_topic(topic):
    """Remember the last researched subject. A self-referential meta query
    ("the name previously searched") is never a subject and is not stored,
    so later "by that name" refinements cannot chain off garbage."""
    global _last_research_topic
    text = str(topic or "").strip()
    if not text or _META_QUERY_RE.search(text):
        return
    _last_research_topic = topic

_REFERENCE_ONLY_TOKENS = {
    "bare", "baare", "mein", "batao", "bata", "kar", "karke", "karkar", "karo",
    "ke", "ka", "ki", "ko", "se", "to", "pe", "per", "is", "us", "it", "its",
    "this", "that", "these", "those", "the", "a", "an", "about", "after",
    "also", "and", "any", "are", "at", "be", "been", "but", "by", "can",
    "concerning", "did", "do", "if", "into", "me", "my", "myself",
        "internet", "web", "of", "off", "or",
        "out", "please", "regarding", "related", "search", "deepsearch", "so", "some", "than",
    "their", "through", "told", "up", "want", "was", "we", "were", "what",
    "when", "which", "who", "will", "with", "would", "ye", "you", "iske",
    "uske", "unki", "unke", "inme", "unme", "isme", "ismein", "usme",
    "usmein", "inmein", "unmein", "isko", "usko", "inko", "unko", "is",
    "us", "bani", "baat", "cheez", "wo", "ye", "voh",
}


def _is_reference_only(query):
    """True when the text is a dangling pointer ("iske bare mein") that only
    makes sense relative to the previous topic we just researched."""
    if not query or not query.strip():
        return True
    tokens = [
        token for token in re.split(r"[\W_]+", query.lower())
        if token
    ]
    if not tokens or len(tokens) > 10:
        return False
    leftovers = [
        token for token in tokens
        if token not in _REFERENCE_ONLY_TOKENS
    ]
    return not leftovers


# ── Live fix: deictic search queries ("this stream", "that creator") ─────────
# A search must never be handed a bare pointer. "this stream" resolves to the
# thing last visible on screen (or the matching ledger entity); when nothing
# concrete exists, the caller asks ONE short question instead of searching
# the pointer text.

#: Generic media nouns that carry no subject on their own.
_GENERIC_MEDIA_TOKENS = {
    "stream", "streamer", "streaming", "livestream", "creator", "youtuber",
    "youtube", "yt", "channel", "video", "videos", "song", "movie",
    "content", "clip", "series", "guy", "person", "dude", "thing", "name",
    "title", "topic", "subject", "anything", "something", "use", "using",
    "reaction", "reactions", "response", "responses", "feedback",
    "thoughts", "opinions", "release", "releases", "announcement",
    "announcements", "update", "updates", "launch", "verse",
    "screen", "monitor", "display",
    "live", "chat", "message", "messages", "comment", "comments",
    "post", "posts",
}

#: Words that carry no referent by themselves.
_DEREF_FILLERS = _REFERENCE_ONLY_TOKENS | _GENERIC_MEDIA_TOKENS | {
    "on", "in", "to", "for", "from", "with", "the", "a", "an", "just",
    "now", "find", "found", "searching", "looking", "look", "up", "please",
    "my", "your", "this", "that", "it", "them", "these", "those", "about",
    "do", "don", "t", "dont", "not", "ask", "asking", "question",
    "questions", "pls", "plz", "execute", "executing", "run", "running",
    # Live fix: a correction is still a pointer — "not the video, i wanted
    # you to search about the creator of this video" names no subject of
    # its own and must resolve (never be searched verbatim).
    "i", "wanted", "meant", "asked",
}

#: The reference asks for an ATTRIBUTE of the on-screen media (its maker),
#: not for the media itself — "search about this creator".
_CREATOR_ATTR_RE = re.compile(
    r"\b(?:content\s+)?(?:creator|channel|uploader|youtuber|streamer|"
    r"author|maker)\b"
    r"|\bwho\s+(?:made|created|owns|runs|posted|is\s+behind)\b",
    re.IGNORECASE,
)

#: Attribute-talk words that are still pointers in this context ("the
#: channel that MADE this video").
_CREATOR_POINTER_EXTRA = {"made", "created", "posted", "owns", "runs",
                          "behind", "no", "tell"}

#: A correction pointing back at the last researched name ("you searched a
#: different man by that same name, search the youtube creator").
_NAME_REFINE_RE = re.compile(
    r"\b(?:by|with)\s+(?:that|the\s+same|this)\s+name\b"
    r"|\bsame\s+name\b"
    r"|\bdifferent\s+(?:man|person|guy)\b",
    re.IGNORECASE,
)

#: Self-referential meta text a classifier can emit instead of a real
#: subject ("...the name previously searched"). Never searched, never
#: stored as the last research topic.
_META_QUERY_RE = re.compile(
    r"\b(previously|earlier|mentioned|same\s+name|that\s+name|"
    r"the\s+name\s+(?:you|previously|just)|different\s+(?:man|person|guy))\b",
    re.IGNORECASE,
)

_screen_topic_lock = threading.Lock()
_last_screen_topic = {"text": "", "at": 0.0}
_SCREEN_TOPIC_TTL = 900.0


def _set_last_screen_topic(topic):
    """Remember what the last screen analysis saw. Never raises."""
    global _last_screen_topic
    try:
        text = str(topic or "").strip()
        if not text:
            return
        with _screen_topic_lock:
            _last_screen_topic = {"text": text, "at": time.time()}
    except Exception:
        pass


def _get_last_screen_topic():
    """The last screen topic inside its TTL, else ""."""
    try:
        with _screen_topic_lock:
            state = dict(_last_screen_topic)
        if state.get("text") and (time.time() - float(state.get("at") or 0.0)
                                  < _SCREEN_TOPIC_TTL):
            return str(state["text"])
    except Exception:
        pass
    return ""


_last_screen_creator = {"text": "", "at": 0.0}


def _set_last_screen_creator(creator):
    """Remember the media maker the last screen analysis saw. Never raises."""
    global _last_screen_creator
    try:
        text = str(creator or "").strip()
        if not text:
            return
        with _screen_topic_lock:
            _last_screen_creator = {"text": text, "at": time.time()}
    except Exception:
        pass


def _get_last_screen_creator():
    """The last screen creator inside its TTL, else ""."""
    try:
        with _screen_topic_lock:
            state = dict(_last_screen_creator)
        if state.get("text") and (time.time() - float(state.get("at") or 0.0)
                                  < _SCREEN_TOPIC_TTL):
            return str(state["text"])
    except Exception:
        pass
    return ""


_last_screen_report = {"topic": "", "tip": "", "creator": "", "at": 0.0}


def _set_last_screen_report(topic="", tip="", creator=""):
    """Remember the FULL last screen observation, so a later "research that
    secret message" can resolve to what was actually identified on screen —
    the title in the tip, not the 2-5-word topic label. Never raises."""
    global _last_screen_report
    try:
        topic = str(topic or "").strip()
        tip = str(tip or "").strip()
        creator = str(creator or "").strip()
        if not (topic or tip or creator):
            return
        with _screen_topic_lock:
            _last_screen_report = {
                "topic": topic, "tip": tip, "creator": creator,
                "at": time.time(),
            }
    except Exception:
        pass


def _get_last_screen_report():
    """The last screen report inside its TTL, else None."""
    try:
        with _screen_topic_lock:
            state = dict(_last_screen_report)
        if state and (time.time() - float(state.get("at") or 0.0)
                      < _SCREEN_TOPIC_TTL):
            return state
    except Exception:
        pass
    return None


def _screen_subject_candidates():
    """Concrete subject strings from the last screen observation, best
    first: a quoted title from the tip ("titled "Karna Vs Arjun Ko Secret
    Message"") beats the short topic label."""
    report = _get_last_screen_report()
    if not report:
        return []
    out = []
    tip = str(report.get("tip") or "")
    for match in re.finditer(r"[\"\u201c\u2018']([^\"\u201d\u2019']{3,140})"
                             r"[\"\u201d\u2019']", tip):
        quoted = match.group(1).strip(" ,.;:-")
        if quoted and not _is_deictic_query(quoted):
            out.append(quoted)
    topic = str(report.get("topic") or "").strip()
    if topic and not _is_deictic_query(topic) and topic not in out:
        out.append(topic)
    return out


def _carry_context_subject(raw, query):
    """Bind a deictic research request to what was last seen on screen.

    "i want to know that secret message research about it" must research
    the TITLE the user is pointing at ("Karna Vs Arjun Ko Secret Message"),
    not the generic fragment the classifier extracted ("secret message").
    Only fires when the request contains a pointer word AND every content
    word of the derived query already appears in a screen subject — so a
    genuinely new subject ("the news", "laptop prices") passes through.
    """
    q = str(query or "").strip()
    if not q:
        return q
    if not re.search(r"\b(?:that|this|it|these|those)\b", str(raw or ""),
                     re.IGNORECASE):
        return q
    qt = [t for t in re.split(r"[\W_]+", q.lower())
          if t and t not in _DEREF_FILLERS]
    if not qt or len(qt) > 8:
        return q
    for subject in _screen_subject_candidates():
        st = set(t for t in re.split(r"[\W_]+", subject.lower()) if t)
        if set(qt) <= st and len(subject) > len(q) + 2:
            print(f"[RESEARCH] Context carry: {q!r} -> {subject!r}")
            return subject[:220]
    return q


def _is_deictic_query(query):
    """True when the text is only a pointer + generic media noun
    ("that youtube creator name", "this stream") with no real subject."""
    if not query or not query.strip():
        return True
    tokens = [t for t in re.split(r"[\W_]+", query.lower()) if t]
    if not tokens or len(tokens) > 12:
        return False
    return not [t for t in tokens if t not in _DEREF_FILLERS]


def _creator_reference_query(text):
    """The on-screen creator name when *text* is a pure pointer at the
    maker ("search about this creator", "not the video, the creator of
    this one") — else "". A clause with its own subject ("creator of
    monalisa") is NOT a pointer and is derived normally."""
    raw = str(text or "")
    if not _CREATOR_ATTR_RE.search(raw):
        return ""
    creator = _get_last_screen_creator()
    if not creator or _is_deictic_query(creator):
        return ""
    tokens = [t for t in re.split(r"[\W_]+", raw.lower()) if t]
    allowed = _DEREF_FILLERS | _CREATOR_POINTER_EXTRA
    if [t for t in tokens if t not in allowed]:
        return ""
    return creator


def _refine_last_name_query(text):
    """A correction pointing at the last researched name refines the search
    to that name's YouTube presence instead of searching the meta sentence
    ("you searched a different man by that same name..."). Returns "" when
    the turn is not such a correction."""
    raw = str(text or "")
    if not raw or not _last_research_topic:
        return ""
    if not _NAME_REFINE_RE.search(raw):
        return ""
    low = raw.lower()
    if not re.search(r"\b(youtube|youtuber|channel|content\s+creator|"
                     r"creator|streamer)\b", low):
        return ""
    topic = str(_last_research_topic).strip()
    if not topic or _META_QUERY_RE.search(topic):
        return ""
    if re.search(r"\b(youtube|youtuber|channel)\b", low):
        return ("%s youtube channel" % topic)[:220]
    return ("%s creator" % topic)[:220]


def _resolve_reference_query(text):
    """Bind "this stream" / "that creator" to a concrete subject.

    Deterministic, never a guess: the entity ledger (recorded screen topics,
    videos, sites), then the last screen topic, then the most recent topic
    entity, then the last researched topic. Returns "" when nothing real is
    available — callers must ask once instead of searching the deictic text.
    """
    raw = str(text or "").strip()
    if not raw:
        return ""
    # Live fix: "search about this creator" / "not the video, the creator
    # of this one" asks for the MAKER seen on screen — never the video topic.
    creator = _creator_reference_query(raw)
    if creator:
        return creator
    media_kinds = {"topic", "video", "movie", "website", "url", "site"}
    try:
        verdict, ent = entity_ledger.resolve_mention(raw)
        if verdict == "bound" and isinstance(ent, dict):
            kind = str(ent.get("kind") or "")
            name = str(ent.get("display_name")
                       or ent.get("canon") or "").strip()
            if kind in media_kinds and name and not _is_deictic_query(name):
                return name
    except Exception:
        pass
    topic = _get_last_screen_topic()
    if topic and not _is_deictic_query(topic):
        return topic
    try:
        ent = entity_ledger.focus_head(kind="topic")
        if ent:
            name = str(ent.get("display_name")
                       or ent.get("canon") or "").strip()
            if name and not _is_deictic_query(name):
                return name
    except Exception:
        pass
    if _last_research_topic and not _is_deictic_query(_last_research_topic):
        return str(_last_research_topic)
    return ""


# ── Live fix: reactions/opinions about the last researched subject ──────────
# "find out how people are reacting to that release from openai" and the
# follow-up correction "no i meant the reactions to that release you just
# researched for me" must both search the SAME concrete subject the earlier
# research produced — not the sentence around it, and never a bare pointer.

#: "find out how people are reacting to X", "search for what users are
#: saying about X".
_REACTION_LEAD_RE = re.compile(
    r"\b(?:find\s+out|search|google|look\s+up|research)\b"
    r"(?:\s+(?:about|for|on))?\s+"
    r"(?:(?:how|what)\s+)?"
    r"(?:people|users|everyone|folks|netizens|the\s+internet|the\s+community|"
    r"twitter|x|reddit)?\s*(?:are|is|'s)?\s*"
    r"(?:reacting|responding|saying|feeling|thinking|thoughts|opinions?|"
    r"reactions?)\s+(?:to|about|on|regarding)\s+(.+)$",
    re.IGNORECASE,
)

#: "the reactions to X" (a correction after the fact, no lead verb).
_REACTION_SHORT_RE = re.compile(
    r"\b(?:reactions?|responses?|feedback|thoughts|opinions?)\s+"
    r"(?:to|about|on|regarding)\s+(.+)$",
    re.IGNORECASE,
)

#: A deictic media noun ("that release", "this announcement") refers to the
#: last researched/screen subject; a named subject is used as-is.
_DEICTIC_MEDIA_RE = re.compile(
    r"^(?:that|this|the|its)\s+"
    r"(?:release|announcement|update|news|video|launch|event|topic|model|"
    r"thing|it|one)\b",
    re.IGNORECASE,
)

#: "…you just researched for me" — the trailing back-reference is not part
#: of the subject.
_REFERENT_TAIL_RE = re.compile(
    r"\s+(?:you|u)\s+(?:just\s+)?"
    r"(?:researched|searched|found|looked\s+up|mentioned|showed|saw|told)"
    r"(?:\s+(?:me|about\s+it))?(?:\s+(?:for\s+me|earlier|before))?\s*$",
    re.IGNORECASE,
)


def _reaction_referent():
    """The concrete subject last researched/screened, or "".

    A stored sentence that is itself a search instruction ("find out how
    people are reacting…") is not a real subject and is skipped.
    """
    for candidate in (globals().get("_last_research_topic"),
                      _get_last_screen_topic()):
        text = str(candidate or "").strip()
        if not text or _META_QUERY_RE.search(text):
            continue
        if _REACTION_LEAD_RE.search(text):
            continue
        if _is_deictic_query(text):
            continue
        return text
    return ""


def _reaction_query(text):
    """Rewrite an opinion/reaction request into a concrete web query.

    "find out how people are reacting to that release from openai" ->
    "reactions to <last researched subject>". Returns "" when the message
    is not a reaction request or no real referent exists — callers fall
    back to the normal derivation (and, for a bare pointer, ask once).
    """
    raw = str(text or "").strip()
    if not raw:
        return ""
    match = _REACTION_LEAD_RE.search(raw) or _REACTION_SHORT_RE.search(raw)
    if not match:
        return ""
    subject = _REFERENT_TAIL_RE.sub("", match.group(1)).strip(" ,.!?")
    if not subject:
        return ""
    if _DEICTIC_MEDIA_RE.match(subject):
        referent = _reaction_referent()
        if not referent:
            print(f"[RESEARCH] Reaction pointer {subject!r} has no referent")
            return ""
        # The stored subject may already be an opinion query ("reactions to
        # X") — never stack "reactions to reactions to X".
        if re.match(r"(?i)^(?:reactions?|responses?|feedback|thoughts|"
                    r"opinions?)\b", referent):
            print(f"[RESEARCH] Reaction query reused: {referent!r}")
            return referent[:220]
        query = ("reactions to %s" % referent)[:220]
        print(f"[RESEARCH] Reaction query: {query!r} <- {raw!r}")
        return query
    query = ("reactions to %s" % subject)[:220]
    print(f"[RESEARCH] Reaction query: {query!r} <- {raw!r}")
    return query


def _meant_refinement_query(text):
    """A correction ("no i meant the reactions to that release you just
    researched") that restates the PREVIOUS request in clearer words is
    that request — not chat. Returns the concrete query, or "".

    Only fires when the sentence carries a reaction/opinion verb and the
    rewrite resolves to a real subject; anything else keeps its route.
    """
    raw = str(text or "").strip()
    if not raw:
        return ""
    if not re.match(r"(?i)^\s*(?:no[,.! ]+|actually[,.! ]+|sorry[,.! ]+)?"
                    r"(?:i|what)\s+(?:meant|said|was\s+asking|am\s+asking)\b",
                    raw):
        return ""
    return _reaction_query(raw)


#: Consent/affirmation lead-ins that open a follow-up to something shown
#: or researched ("yes, that is the verse I wanted you to research about").
_CONSENT_LEAD_RE = re.compile(
    r"^\s*(?:yes|yeah|yep|ya|yup|correct|right|exactly|perfect|sure|"
    r"ok(?:ay)?|that(?:'s| is) (?:right|correct|it))\b[,.! ]*",
    re.IGNORECASE,
)

#: Extra grammar words allowed in a pure back-reference (beyond the global
#: pointer fillers) so "that is the verse I wanted you to research about"
#: leaves no real subject behind.
_BACKREF_EXTRA = {
    "is", "was", "were", "are", "did", "do", "have", "had",
    "yes", "yeah", "yep", "ya", "yup", "sure", "okay", "ok",
    "wanted", "want", "asked", "told", "meant", "mean", "said",
    "research", "searched", "search", "find", "found", "looking", "look",
    "google", "deepsearch", "up", "out", "about", "for", "on", "in",
    "to", "of", "from", "with", "the", "a", "an", "it", "its", "this",
    "that", "these", "those", "i", "you", "me", "we", "one", "same",
    "thing", "there",
}


def _is_backreference_text(text):
    """True when *text* is a consent + pure back-reference shape ("yes
    that is the verse i wanted you to research about") — all checks
    except the resolution itself."""
    raw = str(text or "").strip()
    if not raw:
        return False
    low = _CONSENT_LEAD_RE.sub("", raw).strip(" ,.!?")
    if not low:
        return False
    if not re.search(r"\b(research|search|google|looking\s+up|look\s+up|"
                     r"find\s+out|deepsearch)\b", low, re.IGNORECASE):
        return False
    if not re.search(r"\b(that|this|it|these|those)\b", low, re.IGNORECASE):
        return False
    tokens = [tok for tok in re.split(r"[\W_]+", low.lower()) if tok]
    if not tokens or len(tokens) > 24:
        return False
    allowed = _DEREF_FILLERS | _BACKREF_EXTRA | _GENERIC_MEDIA_TOKENS
    return not [tok for tok in tokens if tok not in allowed]


def _backreference_query(text):
    """Resolve a consent + pure back-reference ("yes that is the verse i
    wanted you to research about") to what was last seen/researched.

    Only when every content word is a pointer/generic noun; a sentence
    that names its own subject keeps its normal route. Returns "" when
    the message is not such a back-reference or nothing concrete exists.
    """
    raw = str(text or "").strip()
    if not _is_backreference_text(raw):
        return ""
    resolved = _resolve_reference_query(raw)
    if resolved:
        print(f"[RESEARCH] Back-reference resolved: {resolved!r} <- {raw!r}")
    return resolved


def _resolve_search_query(text):
    """The best concrete web query for *text*, or "" when only a pointer
    exists. Concrete text goes through the normal derivation; deictic text
    is resolved from the ledger/screen and never searched verbatim."""
    t = str(text or "").strip()
    if not t:
        return ""
    # Live fix: a correction pointing at the last researched name refines
    # to that name's YouTube presence, never the meta sentence.
    refined = _refine_last_name_query(t)
    if refined:
        return refined
    # Live fix: "find out how people are reacting to that release from
    # openai" / "the reactions to that release you just researched" — the
    # opinion is about the last researched subject, never about the
    # sentence itself.
    reaction = _reaction_query(t)
    if reaction:
        return reaction
    # Live fix: "yes that is the verse i wanted you to research about" —
    # a consent + pure back-reference names no subject of its own and must
    # resolve to what was last seen/researched, never reach the classifier
    # (which answered "verse research" and searched lab suppliers). With
    # no referent the caller asks once instead of searching the sentence.
    backref = _backreference_query(t)
    if backref:
        return backref
    if _is_backreference_text(t):
        print(f"[RESEARCH] Back-reference has no referent: {t!r}")
        return ""
    # Live fix: an attribute request about the on-screen media ("...the
    # creator of this video") resolves to the creator name when it is a
    # PURE pointer. A clause with its own subject ("creator of monalisa")
    # falls through and is derived on its own terms.
    creator_hit = _creator_reference_query(t)
    if creator_hit:
        return creator_hit
    local = _heuristic_research_query(t)
    if local and not _is_deictic_query(local):
        q = derive_research_query(t)
        return _carry_context_subject(
            t, q if (q and not _is_deictic_query(q)) else local)
    resolved = _resolve_reference_query(t)
    if resolved:
        return resolved
    q = derive_research_query(t)
    if q and not _is_deictic_query(q):
        return _carry_context_subject(t, q)
    return ""


#: Mid-sentence instruction chatter ("... , research about it i want to know
#: the secret message") is not a subject. When the classifier is down the
#: local heuristic used to leave it in, so the whole sentence was searched.
_INSTRUCTION_RESIDUE_RE = re.compile(
    r"\b(?:research|search|google|look\s+up|find\s+out)\b"
    r"[^.?!,;]{0,24}?\b(?:it|this|that|them)\b"
    r"|\bi\s+(?:want|wanted|would\s+like|need)\s+(?:to\s+)?know\b"
    r"|\bi\s+want\s+you\s+to\b|\btell\s+me\b|\bplease\s+tell\b",
    re.IGNORECASE,
)


def _heuristic_research_query(raw_query):
    """Local fallback: strip wake words, instruction loops, and chasers."""
    low = raw_query.strip().lower()
    low = re.sub(r"^(hey |ok |okay |hello )*(jarvis|jervis|jarvish)[\s.,!]*", "", low).strip()
    # cut at trailing instruction ("so i want you to find out about that course")
    cut = re.split(
        r"\bso\b|\bi want you to\b|\bi need you to\b|\bcan you\b|\bplease\b|\byou should\b",
        low, maxsplit=1,
    )
    left = cut[0].strip(" ,.!?")
    if left:
        low = left
    while True:
        matched = False
        for phrase in (
            "i want to know ", "i want to learn ", "i am trying to remember ",
            "i was wondering about ", "i am about to know ", "i would like to know ",
            "i am planning to ", "i am about to ", "i am going to ", "i have to ",
            "i want you to ", "can you ", "could you ", "would you ",
            "find out about ", "find out more about ", "find about ", "look this up ",
            "look that up ", "look it up ", "look up ", "search for ", "search about ",
            "search on the internet about ", "search on the internet for ",
            "search on internet about ", "search the internet about ",
            "search the internet for ", "search the web for ", "research about ",
            "research on ", "research ",
            "google it ", "tell me about ", "tell me more about ", "what can you find about ",
            "what do you know about ", "what is ", "what are ", "who is ",
            "do a deepsearch on ", "do a deepsearch about ", "do a deepsearch for ",
            "deepsearch about ", "deepsearch on ", "deepsearch for ", "deepsearch ",
            "about ", "on the topic of ",
        ):
            if low.startswith(phrase):
                low = low[len(phrase):].strip(" ,.!?")
                matched = True
                break
        if not matched:
            break
    # Live fix: an instruction that trails the pointing phrase ("this secret
    # message short video on my screen , research about it i want to know
    # the secret message") is scaffolding, not subject matter — cut it off
    # and keep the phrase the user pointed at. Runs after the prefix strip,
    # so a leading "i want to know" cannot shadow the trailing instruction.
    residue = _INSTRUCTION_RESIDUE_RE.search(low)
    if residue and residue.start() > 0:
        left = low[:residue.start()].strip(" ,.!?")
        if left:
            low = left
    low = low.strip(" ,.!?")
    if not low:
        return raw_query
    # The pointing phrase keeps its noun but sheds the bare pointer and the
    # screen mention ("this ... short video on my scrren" -> "secret message
    # short video") — a search never needs "this" or "on my screen".
    low = re.sub(r"^\s*(?:this|that|the)\s+(?=\S)", "", low).strip(" ,.!?")
    low = re.sub(r"\b(?:on|from)\s+(?:my|the)\s+scr+[ae]*n\b", " ", low)
    low = re.sub(r"\s+", " ", low).strip(" ,.!?")
    if not low:
        return raw_query
    return low[:220]


#: F24 / [PERF] The legacy DDGS lookup runs inline in the chat build, BEFORE the
#: first streamed token. It had no timeout and no budget, so a slow or hung
#: lookup could stall a turn for as long as the underlying transport allowed.
#: It is now bounded, and an expired budget sends no request at all.
SEARCH_TIMEOUT = 8.0


def search_internet(query, deadline=None):
    """Fetch a few live snippets for *query* (best-effort, never blocking long).

    *deadline* (F24) is the shared turn budget: when it is already spent no
    lookup is attempted, and the caller's remaining window bounds the wait.
    """
    handle = budget.resolve(deadline)
    if handle is not None and handle.stopped():
        print("[SEARCH] lookup skipped — budget exhausted")
        return None
    request_timeout = budget.seconds_for(handle, SEARCH_TIMEOUT)
    if request_timeout is None:
        return None
    try:
        from ddgs import DDGS
        results = DDGS(timeout=request_timeout).text(query, max_results=3)
        if not results:
            return None

        info_lines = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            info_lines.append(f"- {title}: {body}")

        return "\n".join(info_lines)
    except Exception as e:
        print("[SEARCH] Error:", e)
        return None


#: [PERF] Window in which one message's memory-context reads are reused.
#: Long enough to cover the speculative build and the selected-route build of
#: the SAME turn (which are milliseconds apart), short enough that a fact
#: stored in between is picked up by the next turn.
_MEMORY_CONTEXT_TTL = 5.0
_memory_context_cache = {}
_memory_context_lock = threading.Lock()


def _memory_context_cached(user_message):
    """``(memory_block, work_block)`` for *user_message*, memoised briefly.

    Both reads are pure and keyed only on the message, so the speculative and
    the committed build of one turn compute the same thing. Failures degrade
    to empty strings exactly as the uncached version did.
    """
    if memory_store is None:
        return "", ""
    now = time.monotonic()
    key = (user_message or "").strip()
    with _memory_context_lock:
        hit = _memory_context_cache.get(key)
        if hit is not None and (now - hit[0]) < _MEMORY_CONTEXT_TTL:
            return hit[1], hit[2]
    # Read OUTSIDE the lock: these are SQLite queries and must not serialise
    # two concurrent turns behind one another.
    #
    # [PERF] P0-11 — building the prompt happens on the way to the first delta,
    # so it must not wait for the store's queued BOOKKEEPING writes (the open
    # work_request row for THIS turn). A prompt one turn behind on bookkeeping
    # is correct; a SQLite write between "message received" and "first delta"
    # is not. Everything else keeps read-your-writes. A store stand-in without
    # the guard (tests) simply reads normally.
    _guard = getattr(memory_store, "read_without_barrier", None)
    with (_guard() if callable(_guard) else contextlib.nullcontext()):
        try:
            mem_block = memory_store.memory_context(user_message)
        except Exception:
            mem_block = ""
        # F07: a follow-up about EARLIER WORK gets the identified request's real
        # outcome plus its artifact paths and sources — read from the persisted
        # work-event store, so it survives a restart.
        try:
            work_block = memory_store.work_context(user_message)
        except Exception:
            work_block = ""
    with _memory_context_lock:
        if len(_memory_context_cache) > 64:
            _memory_context_cache.clear()
        _memory_context_cache[key] = (now, mem_block, work_block)
    return mem_block, work_block


def _current_time_note():
    """Spoken-friendly date/time line at minute precision (S12).

    The model otherwise has no idea what "today" is, so "what day is it"
    needed a web lookup. Rounded to the minute because the seconds are noise.
    """
    try:
        return time.strftime(
            "Current date and time: %A, %d %B %Y, %I:%M %p (local).",
            time.localtime())
    except Exception:
        return ""


def _append_time_note(text):
    note = _current_time_note()
    return f"{text}\n\n{note}" if note else text


def _build_chat_messages(user_message, voice_compact=False, speculative=False, history=None):
    """Shared message-construction for chat (used by both streaming and plain paths).

    F25: with *speculative* the build is pure — no web search, no external
    effect of any kind — because it races the intent router and may be
    cancelled before anyone sees it. A speculative build that *would* have
    searched returns ``path="needs_search"`` instead: the lookup is acquired
    later, in the selected route (see :func:`handle_chat`), so a cancelled
    speculation never leaves a stray DDGS call or a duplicate search behind.

    *history* is an immutable context snapshot. The racer passes the history
    it captured before the user turn was committed; when it is omitted the
    live history is read, which is only correct once the turn is committed.
    """
    lang = detect_language(user_message)
    print(f"[CHAT] Detected language: {lang}")

    if lang == "hindi":
        system_prompt = (
            "You are Jarvis, a smart and efficient personal AI assistant. "
            "The user is speaking in Hindi or Hinglish (mixed Hindi-English). "
            "Understand the user's Hindi or Hinglish meaning, but ALWAYS reply only in English. "
            "Do not answer in Hindi, Hinglish, or Devanagari unless the user is asking for translation examples. "
            "Be concise, helpful, and address the user as 'sir' occasionally. "
            "Keep responses short unless asked for detail. "
            "If this is a spoken conversation, sound natural and answer in one or two short sentences unless more detail is requested."
        )
    else:
        system_prompt = (
            "You are Jarvis, a sharp and efficient AI personal assistant. "
            "Always respond only in English, even if the user speaks another language. "
            "You have memory of this conversation. Be concise and helpful. "
            "Address the user as 'sir' occasionally for character. "
            # Capability grounding: this chat route has no tools, so it must
            # NEVER claim an inability ("I cannot create files/folders",
            # "I am unable to browse websites"). Real actions — file/folder
            # ops, code, shell, browser automation — run through task routes;
            # a request phrased as an action belongs there, not in chat.
            # Honesty rule: this reply performs NOTHING. Never promise an
            # action ("I will get that created", "done", "it is ready") —
            # either the turn routed to a task path (which speaks its own
            # verified result), or ask for the missing detail instead.
            # R13 (strict): the deterministic gate in _finalize_chat_reply
            # strips unverified action claims, but do not rely on it — NEVER
            # emit claim-shaped sentences ("I will...", "I am on it", "right
            # away", "has been created", "taking over...") from this route.
            # Say capability ("I can create files..."), ask for the missing
            # detail, or say plainly that nothing was started.
            # When asked to recall the conversation, answer ONLY from the
            # turns above; never invent topics, and never agree with a
            # premise ("I do recall that") unless the turns show it.
            "You are the voice interface, not the hands: file, folder, code, "
            "shell and browser actions are performed by task routes, never "
            "by this chat reply, so never say you cannot do them. Name "
            "capability as capability ('I can create files...') or ask for "
            "the missing detail — never narrate an action as happening. "
            "This reply itself performs nothing: never claim an action is "
            "done, in progress, queued, or promised — if the request needs "
            "an action, ask for the missing detail instead. "
            "When asked what was discussed, report only what the turns "
            "above show; if a claimed topic is absent, say so plainly "
            "instead of agreeing. "
            "If this is a spoken conversation, sound natural and answer in one or two short sentences unless more detail is requested."
        )

    if voice_compact:
        # [S12] A spoken reply must sound like speech, not like rendered text.
        system_prompt += (
            " This reply is SPOKEN aloud: write only what should be read out."
            " Never use markdown, headings, bullet or numbered lists, tables,"
            " emojis, code blocks, or URLs."
            " Say numbers, units, dates and times the way a person says them"
            " (for example 'four thirty PM', 'twenty percent')."
            " Vary how you open a reply; do not start every reply with 'Sir'."
            " Prioritize a fast spoken reply over a detailed one."
            " Keep it natural, under 35 words, and within two short sentences"
            " unless detail is requested."
        )

    # G9 (F06): bounded scoped-memory injection — a pure read (speculation
    # safe, F25), silent on an empty store, budget-capped, provenance from
    # the facts table only (never raw chat history).
    #
    # [PERF] These are two SQLite reads (FTS5 lookup + work-event lookup) and
    # they are keyed only on the user message, but this function runs TWICE per
    # turn: once inside the speculative racer and once in the selected route.
    # The result is memoised per message for a short window so the second build
    # reuses the first one's reads. A TTL (not a permanent cache) keeps it
    # honest: a fact stored between two builds of the same message is picked
    # up, and a later turn always re-reads.
    mem_block, work_block = _memory_context_cached(user_message)
    if mem_block:
        system_prompt = system_prompt + "\n\n" + mem_block
    if work_block:
        system_prompt = system_prompt + "\n\n" + work_block

    search_info = None

    if force_search(user_message) or should_search(user_message):
        if speculative:
            # F25 — speculation stays pure. No network call is made here; the
            # selected route performs the lookup if it still wants one.
            print("[CHAT] Search needed but build is speculative — deferring lookup")
            return {
                "path": "needs_search",
                "system_prompt": system_prompt,
                "query": user_message,
                "search_info": None,
            }
        print("[CHAT] Search triggered")
        # F24/[PERF]: the legacy DDGS lookup runs inline here, before the first
        # streamed token. It is bounded by this turn's job so a slow lookup can
        # no longer stall the reply for an unbounded time.
        search_info = search_internet(user_message, deadline=current_turn_job())

        if not search_info or len(search_info) < 20:
            return {
                "path": "browser_search",
                "system_prompt": system_prompt,
                "query": user_message,
                "search_info": None,
            }

    if history is None:
        history = get_history()
    if voice_compact:
        history = history[-6:]
    if search_info:
        # ``history[:-1]`` assumes the new user turn is already the last
        # entry. That holds in the committed (selected) route but NOT in a
        # speculative build, which runs before the turn is committed — slicing
        # there would silently drop a real message (F25).
        tail = history if speculative else history[:-1]
        messages = [
            {"role": "system", "content": system_prompt},
            *tail,
            {"role": "user", "content": _append_time_note(
                f"{user_message}\n\nSearch info: {search_info}")},
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
        ]
        # [S12] Put the current date/time in the LATEST user turn — never the
        # system prompt — so the cacheable prompt prefix is unchanged minute to
        # minute (a timestamp at the top would bust prompt caching every turn).
        # Only the current turn qualifies here; in a speculative build the new
        # turn is appended later by the racer, which adds the note itself.
        if (messages[-1].get("role") == "user"
                and messages[-1].get("content") == user_message):
            messages[-1] = dict(messages[-1],
                                content=_append_time_note(user_message))
    return {
        "path": "llm",
        "system_prompt": system_prompt,
        "query": user_message,
        "messages": messages,
    }


def _commit_chat(role, text):
    if text:
        add_message(role, text)


#: R19 — how much of a tool-call summary one history entry carries. Bounded
#: for the same reason S13 bounds background results: history is the model's
#: conversation window, not a work log.
_TOOL_SUMMARY_MAX = 240


def _remember_tool_summary(kind, summary):
    """R19: a completed tool call leaves one compact entry in history.

    Delivered browser/opencode/research terminal results already join the
    history as ``[background result]`` (S13). This covers the native engine,
    which spoke a string and persisted only work events: local file/folder
    work was never recallable ("what file did you just create?"). The
    ``[<kind>]`` marker rides in-band so the model reads it as a work note,
    not a conversational turn, and a repeat of the immediately-preceding
    entry is not appended again (the S13 stacking guard).
    """
    text = re.sub(r"\s+", " ", str(summary or "")).strip()
    if not text:
        return
    if len(text) > _TOOL_SUMMARY_MAX:
        text = text[:_TOOL_SUMMARY_MAX].rstrip()
        cut = text.rfind(" ")
        if cut > 0:
            text = text[:cut].rstrip(".,;:")
        text += "…"
    entry = "[%s] %s" % (kind, text)
    try:
        recent = get_history()[-1:]
    except Exception:
        recent = []
    if recent and recent[0].get("content") == entry:
        return
    try:
        add_message("assistant", entry)
    except Exception as exc:
        logging.warning("[R19] tool summary history write failed: %s", exc)


def _record_native_task_outcome(fallback_text=""):
    """F07/F09 — persist the NATIVE engine's terminal TaskResult.

    ``handle_task_message`` speaks a string, so the structured result used to
    be thrown away: a verified native run could never become a skill and the
    work log only saw the opencode engine. The agent keeps the last result;
    this records it (trace and verification included) exactly once per turn.

    R19: the same outcome also leaves a compact ``[task]`` note in the
    conversation history, so a later turn can recall what was done instead
    of the exchange living only in work events.
    """
    try:
        result, task_text = _last_task_result()
    except Exception:
        return
    if result is None:
        return
    status = str(getattr(result, "status", "") or "")
    if status not in ("completed", "partial", "failed", "stopped",
                      "needs_input", "no_action", "refused", "known_failure"):
        return
    summary = (getattr(result, "summary", "")
               or getattr(result, "detail", "") or "")[:200]
    # R19: the request is bounded tightly and the outcome headline gets the
    # larger share, so the entry's cap can never truncate away what happened.
    label = re.sub(r"\s+", " ", str(task_text or fallback_text or "")).strip()
    outcome = re.sub(r"\s+", " ", str(summary or "")).strip()
    if len(label) > 100:
        label = label[:100].rsplit(" ", 1)[0] + "…"
    if len(outcome) > 160:
        outcome = outcome[:160].rsplit(" ", 1)[0] + "…"
    _remember_tool_summary("task", " — ".join(
        part for part in (label, outcome) if part))
    if memory_store is None:
        return
    try:
        memory_store.record_task_outcome(
            "task_agent", status, task_text or fallback_text,
            summary=summary,
            evidence=list(getattr(result, "evidence", None) or []),
            trace=list(getattr(result, "trace", None) or []),
            verification=list(getattr(result, "verification", None) or []),
        )
    except Exception as exc:
        logging.debug("[TASK] native outcome not recorded: %s", exc)


def _resolve_chat_model():
    """Runtime (provider, model) selection for text chat replies.

    The registry reads data/jarvis_settings.json per message, so a UI model
    switch takes effect on the very next reply — no restart.

    F49: a registry failure is NOT permission to use someone else's provider.
    ``get_model_for_role`` already replaces an invalid ENV selection with the
    role's authorized default; it raises only when the selection touches a
    private/unknown provider or is otherwise unauthorized. That case fails
    CLOSED here (no provider) instead of quietly posting the message to Gemini.
    """
    try:
        return model_registry.get_default_chat_model()
    except Exception as exc:
        logging.warning("[CHAT] Model registry resolution failed: %s", exc)
        return {"provider": None, "model": None, "refused": str(exc)[:200]}


def _chat_refusal_text(reason):
    """F49 — an honest, secret-free reply when no provider is authorized."""
    detail = re.sub(r"\s+", " ", str(reason or "")).strip()
    if len(detail) > 160:
        detail = detail[:157] + "…"
    return ("Sir, the chat model you selected isn't authorized to run"
            + (" (%s)" % detail if detail else "")
            + " — I sent nothing anywhere. Please pick a valid model in "
              "settings.")


def _turn_budget():
    """F24 — the shared deadline/cancellation handle for this turn, if any."""
    try:
        from backend.services import jobs as _jobs
        return _jobs.turn_budget()
    except Exception:
        return None


def _terminal_failure_text(provider, result):
    """F24 — an honest reply for a non-retryable provider rejection."""
    detail = ""
    if isinstance(result, dict):
        failure = result.get("failure")
        if isinstance(failure, dict):
            detail = str(failure.get("detail") or failure.get("message") or "")
        else:
            detail = str(getattr(failure, "detail", "") or
                         getattr(failure, "message", "") or "")
        detail = detail or str(result.get("detail") or result.get("error") or "")
    detail = re.sub(r"\s+", " ", detail).strip()
    if len(detail) > 140:
        detail = detail[:137] + "…"
    return ("Sir, %s rejected the request and it isn't something a retry can "
            "fix%s. I did not ask another provider — please check the "
            "credentials or the model name."
            % (provider, " (%s)" % detail if detail else ""))


def _failure_is_terminal(result):
    """F24 — True when a client result reports a NON-retryable failure.

    The clients classify centrally (auth/404/validation are terminal;
    rate-limit/server/timeout/connection are retryable). A terminal failure
    must not be "retried" by silently asking a different provider: the
    credential or the request is wrong, and that is the user's to fix.
    """
    if not isinstance(result, dict):
        return False
    failure = result.get("failure")
    if failure is None:
        return False
    terminal = getattr(failure, "terminal", None)
    if terminal is None and isinstance(failure, dict):
        terminal = failure.get("terminal")
    return bool(terminal)


_last_chat_fallback = None
_last_chat_fallback_lock = threading.Lock()


def _record_chat_fallback(provider, model, error_detail, fallback_provider="gemini"):
    """Record a provider fallback for UI warning and diagnostics.

    Called when the user-selected provider/model failed and the reply was
    served by a different provider. Stores provider/model/error/timestamp
    and logs one line to stdout. Keys are scrubbed — never log raw secrets.
    """
    global _last_chat_fallback
    try:
        with _last_chat_fallback_lock:
            _last_chat_fallback = {
                "provider": str(provider or ""),
                "model": str(model or ""),
                "error": str(error_detail or "stream empty")[:200],
                "fallback_provider": str(fallback_provider or ""),
                "ts": time.time(),
            }
        print(f"[CHAT] Fallback: {provider}/{model} -> {fallback_provider} error: {error_detail}")
    except Exception:
        pass


def get_last_chat_fallback():
    """Return the last fallback event dict or None. Thread-safe copy."""
    with _last_chat_fallback_lock:
        return dict(_last_chat_fallback) if _last_chat_fallback else None


def _reasoning_effort_for_chat():
    """S27 — the registry-validated reasoning control for the chat role.

    A thinking model behind a generic gateway must be TOLD not to think on
    the voice path: thought text is stripped, but the time it costs is not.
    The snapshot decides which effort (if any) is safe; the client replays
    once without the field if the model rejects it (F56).
    """
    try:
        _reasoning = (model_registry.get_model_config("chat")
                      or {}).get("reasoning") or {}
    except Exception:
        _reasoning = {}
    return _reasoning.get("effort") if _reasoning.get("supported") else None


def _stream_chat_deltas(messages, temperature, max_tokens, cancel=None):
    """Yield streaming deltas; the runtime-selected model is primary, the
    env-configured chain (Gemini → Fireworks) is the fallback.

    F25: *cancel* is an optional ``threading.Event``. Every provider client
    checks it between chunks and closes its HTTP response when it fires, so a
    cancelled speculation stops reading (and stops holding a socket) instead
    of running to completion. It is also checked here between providers so a
    cancelled race never starts a fallback stream.
    """
    selection = _resolve_chat_model()
    provider = selection.get("provider")
    model = selection.get("model")
    budget = _turn_budget()
    # F24 — only pass the handle when a turn actually has one, so a client
    # without budget support keeps its default behaviour.
    _budget_kw = {"deadline": budget} if budget is not None else {}

    def _stopped():
        return cancel is not None and cancel.is_set()

    if not provider:
        # F49 — no authorized provider: fail closed, send nothing.
        yield _chat_refusal_text(selection.get("refused"))
        return

    def _gemini_then_fireworks():
        streamed = False
        for delta in ask_gemini_chat_stream(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model if provider == "gemini" else None,
            cancel=cancel,
            **_budget_kw,
        ):
            if delta:
                streamed = True
                yield delta
            if _stopped():
                return
        if not streamed and not _stopped():
            print("[CHAT] Gemini stream empty — falling back to Fireworks.")
            for delta in ask_fireworks_stream(
                messages, temperature=temperature, max_tokens=max_tokens,
                cancel=cancel, **_budget_kw,
            ):
                if delta:
                    yield delta
                if _stopped():
                    return

    if provider == "fireworks":
        streamed = False
        for delta in ask_fireworks_stream(
            messages, temperature=temperature, max_tokens=max_tokens, model=model,
            cancel=cancel, **_budget_kw,
        ):
            if delta:
                streamed = True
                yield delta
            if _stopped():
                return
        if not streamed and not _stopped():
            # Stream failed (400/404 or empty) — retry SAME model non-stream
            # before any provider fallback. Reuses the non-stream path which
            # already handles the broadened reasoning_effort retry.
            print(f"[CHAT] Fireworks stream empty for {model} — trying non-stream fallback before Gemini.")
            try:
                result = ask_fireworks(
                    messages, temperature=temperature, max_tokens=max_tokens, model=model,
                    **_budget_kw,
                )
                if result and result.get("choices"):
                    content = result["choices"][0].get("message", {}).get("content", "")
                    if content and content.strip():
                        yield content
                        return
                # F24 — an auth/permission/validation rejection is terminal:
                # asking Gemini instead would hide a fixable configuration
                # error and answer the user with a different model than the
                # one they chose.
                if _failure_is_terminal(result):
                    _record_chat_fallback(provider, model, "terminal failure", "none")
                    yield _terminal_failure_text(provider, result)
                    return
            except Exception as exc:
                logging.warning("[CHAT] Fireworks non-stream fallback error: %s", exc)
            if _stopped():
                return
            # Non-stream also failed — record provider fallback and go to Gemini
            _record_chat_fallback(provider, model, "stream and non-stream failed", "gemini")
            print("[CHAT] Fireworks stream empty — falling back to Gemini chain.")
            yield from _gemini_then_fireworks()
        return

    if provider and provider != "gemini" and model:
        api_key, base_url = model_registry.get_provider_credentials(provider)
        if api_key and base_url:
            streamed = False
            reasoning_effort = _reasoning_effort_for_chat()
            for delta in ask_openai_compat_stream(
                messages,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                cancel=cancel,
                # [P1-13] this client was the one stream with NO deadline: a
                # provider that accepted the connection and went silent held
                # the thread until the turn ended. Same F24 handle as the
                # gemini/fireworks streams.
                **_budget_kw,
            ):
                if delta:
                    streamed = True
                    yield delta
                if _stopped():
                    return
            if streamed:
                return
            if _stopped():
                return
            # Stream empty — try same-model non-stream before provider fallback
            print(f"[CHAT] Custom provider {provider}/{model} stream empty — trying non-stream fallback.")
            try:
                from backend.services.openai_compat_client import ask_openai_compat
                result = ask_openai_compat(
                    messages,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if result and result.get("choices"):
                    content = result["choices"][0].get("message", {}).get("content", "")
                    if content and content.strip():
                        yield content
                        return
            except Exception as exc:
                logging.warning("[CHAT] Custom provider non-stream fallback error: %s", exc)
            if _stopped():
                return
            if _failure_is_terminal(result):
                # F24 — a rejected key/endpoint is terminal: don't mail the
                # conversation to Gemini on top of it.
                _record_chat_fallback(provider, model, "terminal failure", "none")
                yield _terminal_failure_text(provider, result)
                return
            _record_chat_fallback(provider, model, "custom stream and non-stream failed", "gemini")
            print("[CHAT] Custom provider stream empty — falling back to Gemini chain.")
        else:
            # F49 — a private provider with no usable credentials has NO
            # authorized fallback: nothing is sent anywhere else.
            _record_chat_fallback(provider, model, "missing credentials", "none")
            print("[CHAT] Custom provider missing credentials — failing closed.")
            yield _chat_refusal_text(
                "%s has no usable credentials" % provider)
            return

    if not _stopped():
        yield from _gemini_then_fireworks()


def _ask_chat_nonstream(messages, temperature, max_tokens):
    """Non-stream chat ask against the runtime-selected model; the
    env-configured chain (Gemini → Fireworks) is the fallback.

    F24/F49: a terminal failure (auth/permission/validation) or an
    unauthorized selection ends the chain instead of being replayed against
    another provider.
    """
    selection = _resolve_chat_model()
    provider = selection.get("provider")
    model = selection.get("model")
    budget = _turn_budget()
    _budget_kw = {"deadline": budget} if budget is not None else {}

    if not provider:
        return {"choices": [], "refused": selection.get("refused"),
                "detail": _chat_refusal_text(selection.get("refused"))}

    if provider == "fireworks":
        result = ask_fireworks(
            messages, temperature=temperature, max_tokens=max_tokens, model=model,
            **_budget_kw,
        )
        if result and result.get("choices"):
            return result
        if _failure_is_terminal(result):
            _record_chat_fallback(provider, model, "terminal failure", "none")
            return result
        _record_chat_fallback(provider, model, str(result.get("detail") or result.get("error") or "fireworks empty") if isinstance(result, dict) else "fireworks empty", "gemini")
        print("[CHAT] Fireworks returned nothing — falling back to Gemini.")

    elif provider != "gemini" and model:
        api_key, base_url = model_registry.get_provider_credentials(provider)
        if api_key and base_url:
            result = ask_openai_compat(
                messages,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if result and result.get("choices"):
                return result
            if _failure_is_terminal(result):
                _record_chat_fallback(provider, model, "terminal failure", "none")
                return result
            _record_chat_fallback(provider, model, str(result.get("detail") or result.get("error") or "custom empty") if isinstance(result, dict) else "custom empty", "gemini")
            print("[CHAT] Custom provider returned nothing — falling back to Gemini.")
        else:
            # F49 — no authorized fallback for a private provider.
            _record_chat_fallback(provider, model, "missing credentials", "none")
            print("[CHAT] Custom provider missing credentials — failing closed.")
            return {"choices": [], "detail": _chat_refusal_text(
                "%s has no usable credentials" % provider)}

    result = ask_gemini_chat(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model if provider == "gemini" else None,
        **_budget_kw,
    )
    if not result or not result.get("choices"):
        if _failure_is_terminal(result):
            return result
        print("[CHAT] Gemini returned nothing — falling back to Fireworks.")
        result = ask_fireworks(
            messages, temperature=temperature, max_tokens=max_tokens,
            **_budget_kw,
        )
    return result


#: F25 — a speculative stream is abandoned rather than allowed to buffer
#: without bound. If nobody consumes this many deltas, the race is over and
#: the producer stops (and closes its HTTP response).
_RACER_QUEUE_LIMIT = 256

#: Hard ceiling on ONE speculative chat stream.
#:
#: The speculation runs on its own thread, and the turn budget (F24) is a
#: ``threading.local`` job handle — so the racer thread inherits NO budget and
#: its provider stream is bounded only by per-socket read timeouts. A stream
#: that trickles (or a provider that goes quiet without closing) therefore ran
#: forever, and because ``handle_chat`` drains it synchronously, the whole
#: request never reached a terminal frame: the user's turn was committed, no
#: reply was stored, and the UI stayed locked. This deadline is the backstop
#: that makes the speculation finite no matter what the provider does.
_RACER_DEADLINE_SECONDS = 30.0

#: How long a consumer waits for the speculation before it abandons the race.
#: Deliberately longer than :data:`_RACER_DEADLINE_SECONDS` so the normal
#: end-of-stream sentinel always wins; this only fires if the producer thread
#: itself is wedged in a blocking read that cancellation cannot interrupt.
_RACER_DRAIN_TIMEOUT_SECONDS = 45.0


def _orchestrator_reply(outcome):
    """The spoken reply for an orchestrator outcome (F02).

    The orchestrator's contract has four statuses; anything else is treated as
    an error rather than answered with a half-empty string. A PROPOSAL replies
    with the SAME confirmation preview the legacy task path uses, because the
    approval the preview describes is the one already armed.
    """
    from backend.services.orchestrator import ANSWERED, PROPOSAL, STATUSES, SUSPENSION

    if not isinstance(outcome, dict):
        return ""
    status = outcome.get("status")
    reply = str(outcome.get("reply") or "")
    if status not in STATUSES:
        print("[ORCHESTRATOR] unknown status %r — treating as suspension" % (status,))
        status = SUSPENSION
    if status == PROPOSAL:
        plan = outcome.get("plan") or {}
        try:
            from backend.services.task_agent import agent as task_agent
            return task_agent.confirmation_prompt(plan)
        except Exception as exc:
            logging.warning("[ORCHESTRATOR] confirmation preview failed: %s", exc)
            return reply
    if status == ANSWERED:
        # R13: the orchestrator's free-form answer travels the chat channel,
        # so it carries chat authority only — strip unverified action claims.
        return _strip_unverified_action_claims(reply, "chat")
    if status == PROPOSAL:
        # R17: a proposal goes nowhere unless the SAME shared confirmation
        # gate was actually armed for it — both routes run the one gate.
        try:
            from backend.services.task_agent import agent as task_agent
            if not task_agent.has_pending_task_confirmation():
                return ("Sir, I could not arm that action for approval — "
                        "nothing was started. Please ask again.")
        except Exception:
            pass
    # R14: suspension/error planner text travels the chat channel too — a
    # "navigating to..." planner line must never reach speech unexamined.
    return _strip_unverified_action_claims(reply, "chat")


class _ChatRacer:
    """Race a *pure* chat speculation against the intent router.

    F25: the speculation does nothing but generate an answer. It performs no
    search and has no other external effect, it builds from an immutable
    context snapshot taken once at construction, its queue is bounded, and
    :meth:`cancel` closes the underlying HTTP response instead of merely
    setting a flag that is only noticed when the next delta arrives.

    [P0-10] It now starts BEFORE the pre-route predicate chain, which is what
    makes it necessary for the object to know whether a route ADOPTED it: the
    turn's cleanup cancels a speculation no route wanted, and cancelling an
    adopted one would inject the end-of-stream sentinel and truncate the reply
    that is being streamed from it.
    """

    def __init__(self, msg, voice_compact, history=None):
        self._msg = msg
        self._voice_compact = voice_compact
        # Immutable request/context snapshot: taken once, before the caller
        # commits the new user turn, so a concurrent commit cannot mutate the
        # messages this race is generating from.
        self._history = list(get_history() if history is None else history)
        self._queue = queue.Queue(maxsize=_RACER_QUEUE_LIMIT)
        self._sentinel = object()
        self._built = None
        self._has_stream = False
        self._cancelled = False
        #: [P0-10] True once a route has taken ownership of the speculation.
        self._adopted = False
        self._cancel = threading.Event()
        #: [P1-13] resources (a streaming provider response) to close the moment
        #: this speculation is cancelled, so a blocked read is released.
        self._closers = []
        self._closer_lock = threading.Lock()
        self._done = threading.Event()
        self._built_ready = threading.Event()
        # Backstop: this thread has no turn budget (see
        # _RACER_DEADLINE_SECONDS), so bound the speculation explicitly.
        self._deadline_timer = threading.Timer(
            _RACER_DEADLINE_SECONDS, self._expire)
        self._deadline_timer.daemon = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._deadline_timer.start()
        self._thread.start()

    def _expire(self):
        """Abandon a speculation that outlived its deadline.

        Runs on the timer thread. ``cancel`` unblocks a consumer parked on
        :meth:`adopt` by delivering the end-of-stream sentinel, so a provider
        that goes silent can no longer hold the request open.
        """
        if self._done.is_set():
            return
        logging.warning(
            "[CHAT] speculation exceeded %.0fs — abandoning it so the reply "
            "can still be produced", _RACER_DEADLINE_SECONDS)
        self.cancel()

    def _run(self):
        try:
            built = _build_chat_messages(
                self._msg,
                voice_compact=self._voice_compact,
                speculative=True,
                history=self._history,
            )
            self._built = built
            self._built_ready.set()
            if built.get("path") != "llm":
                # "browser_search" (no usable lookup) and F25's "needs_search"
                # both defer to the selected route — no speculation is emitted.
                self._has_stream = False
                self._put_sentinel()
                return
            self._has_stream = True
            # The racer builds messages BEFORE handle_chat adds the new user
            # message to history, so the user turn must be appended here.
            msgs = built["messages"]
            last = msgs[-1] if msgs else None
            if not (
                last
                and last.get("role") == "user"
                and str(last.get("content", "")).startswith(self._msg)
            ):
                msgs.append({"role": "user",
                             "content": _append_time_note(self._msg)})
            temp = 0.45 if self._voice_compact else 0.7
            mx = 300 if self._voice_compact else 1400
            gen = _stream_chat_deltas(built["messages"], temp, mx,
                                      cancel=self)
            cancelled = False
            try:
                for delta in gen:
                    if self._cancelled:
                        cancelled = True
                        break
                    if delta and not self._put(delta):
                        # Bounded queue full: nobody is consuming this race.
                        # Abandon the speculation instead of buffering further
                        # (F25) — dropping mid-stream deltas would corrupt the
                        # text, so the whole speculation is discarded.
                        cancelled = True
                        self._cancelled = True
                        break
                    if self._cancelled:
                        cancelled = True
                        break
                if cancelled or self._cancelled:
                    try:
                        gen.close()
                    except Exception:
                        pass
            except Exception as exc:
                logging.warning("[CHAT] Racer stream error: %s", exc)
            # always end with sentinel
            self._put_sentinel()
        except Exception as exc:
            logging.warning("[CHAT] Racer build error: %s", exc)
            self._put_sentinel()
            try:
                self._built_ready.set()
            except Exception:
                pass
        finally:
            try:
                self._built_ready.set()
            except Exception:
                pass
            self._done.set()

    def _put(self, delta):
        """Offer one delta to the bounded queue.

        Returns False when the queue is full, which means the consumer has
        stopped draining — the caller abandons the speculation.
        """
        try:
            self._queue.put_nowait(delta)
            return True
        except queue.Full:
            return False

    def _put_sentinel(self):
        """Always deliver the end-of-stream marker, even on a full queue."""
        while True:
            try:
                self._queue.put_nowait(self._sentinel)
                return
            except queue.Full:
                # Make room: this race is being abandoned anyway.
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    return

    def built(self):
        # Wait briefly for _build to finish (runs in parallel with router)
        if self._built is None and not self._built_ready.is_set():
            self._built_ready.wait(timeout=5)
        return self._built

    def adopt(self):
        # [P0-10] From here the speculation IS the reply: the turn's cleanup
        # must not cancel it (cancel() injects the sentinel, which would cut
        # the drained stream short).
        #
        # R14: the live deltas were generated BEFORE the route was known, so
        # they may carry action sentences chat has no authority to speak.
        # Gate at SENTENCE boundaries: deltas pass through byte-identical
        # (whitespace intact) until a terminator completes a sentence, then
        # the whole sentence is gated. Gating each delta alone would strip
        # trailing spaces and merge words ("Hello"+"sir." -> "Hellosir.").
        # The final content is still re-gated in _finalize_chat_reply, so a
        # claim split ACROSS the stream end can never slip through — the
        # boundary gate only protects what the user hears live.
        self._adopted = True

        def _drain():
            buf = ""
            # R14 follow-up: characters of `buf` already yielded live. The
            # sentence flush emits only the not-yet-spoken tail, so a delta
            # that completes a buffered sentence can never replay the words
            # the user already heard ("Loud" + "Loud and clear, sir."
            # duplicated the first word).
            emitted = 0
            while True:
                item = self._queue.get()
                if item is self._sentinel:
                    if buf:
                        try:
                            gated = _strip_unverified_action_claims(buf, "chat")
                        except Exception:
                            gated = buf
                        if (gated and str(gated).strip()
                                and len(str(gated).split()) != len(buf.split())):
                            # A cut claim ("I will get that created" ->
                            # fallback) is redelivered in gated form: live
                            # speech pauses rather than lies, and the prefix
                            # already spoken cannot be unsaid.
                            yield gated
                        else:
                            tail = buf[emitted:]
                            if tail.strip():
                                yield tail
                    break
                piece = str(item or "")
                buf += piece
                # Emit complete sentences; hold the tail (maybe a claim cut
                # mid-sentence) for the next delta or the sentinel flush.
                parts = re.split(r"(?<=[.!?])(\s+)", buf)
                if len(parts) > 1:
                    head = "".join(parts[:-1])
                    buf = parts[-1]
                    try:
                        gated = _strip_unverified_action_claims(head, "chat")
                    except Exception:
                        gated = head
                    # Byte-identical passthrough when nothing was cut (the
                    # overwhelming case — preserves "Hello "+"sir." spacing):
                    # only the part not already streamed is emitted. A cut
                    # sentence replaces the head and is HELD in buf for the
                    # next flush, which redelivers it in gated form.
                    if gated and str(gated).strip():
                        if len(str(gated).split()) == len(head.split()):
                            tail = head[emitted:]
                            if tail.strip():
                                yield tail
                        else:
                            buf = str(gated) + (" " if buf[:1].isspace() else "") + buf
                    emitted = 0
                else:
                    # Byte-identical passthrough of everything not yet
                    # emitted (normally just this piece; after a cut claim,
                    # the held gated replacement goes out here, in order).
                    tail = buf[emitted:]
                    if tail:
                        yield tail
                    emitted = len(buf)
        return _drain()

    def cancel(self):
        """Stop the speculation now, not at the next delta.

        F25: sets the shared cancellation event — every provider client polls
        it between chunks and closes its HTTP response when it fires — and
        unblocks a consumer parked on :meth:`adopt` so an abandoned race is
        released immediately instead of waiting for the socket to drain.

        [P1-13] It also CLOSES the resources registered by the running stream
        (see :meth:`register_closer`): polling an event cannot interrupt a read
        that is already in progress, and this racer has its own deadline, so a
        blocked read must be released here rather than at the next chunk.
        """
        self._cancelled = True
        self._cancel.set()
        self.close_resources()
        self._put_sentinel()

    # ── [P1-13] cancellation handle protocol ────────────────────────────────
    def is_set(self):
        """``threading.Event``-compatible view of the cancellation state."""
        return self._cancel.is_set()

    def register_closer(self, closer):
        """Register ``closer()`` (a response's ``close``) to run on cancel."""
        if closer is None:
            return None
        with self._closer_lock:
            if not self._cancel.is_set():
                self._closers.append(closer)
                return closer
        try:
            closer()
        except Exception:
            pass
        return closer

    def unregister_closer(self, closer):
        with self._closer_lock:
            try:
                self._closers.remove(closer)
            except ValueError:
                pass

    def close_resources(self):
        """Close every registered resource. Never raises."""
        with self._closer_lock:
            closers = list(self._closers)
            self._closers = []
        for closer in closers:
            try:
                closer()
            except Exception:
                pass

    @property
    def is_done(self):
        return self._done.is_set()

    @property
    def is_adopted(self):
        """[P0-10] True once a route took ownership of this speculation."""
        return self._adopted

    def join(self, timeout=None):
        self._thread.join(timeout=timeout)
        return self.is_done

    def has_stream(self):
        return self._has_stream


def _finalize_chat_reply(user_message, content, pieces, stream,
                         commit_response, request_id):
    """The shared tail of every chat reply: sanity floor, F26 uncertainty
    handling, history commit, work-event record. Returns the final text.

    Split out of :func:`handle_chat` so a reply the intent router already
    wrote ([S6] one call for classification AND answer) goes through exactly
    the same commit/stream/memory path as a streamed chat completion - it is
    the same kind of answer, just produced earlier and by the router.
    """
    if not content:
        content = "I'm having trouble connecting. Please try again."
    # R13: tool-less chat has zero action authority. A free-form model reply
    # claiming Jarvis did/will/is doing an action ("I will get that created",
    # "I am on it", "it is ready") is an unverified claim — strip it and say
    # the honest fallback instead. Offers ("I can create files", "do you
    # want me to...") and detail questions pass through untouched.
    content = _strip_unverified_action_claims(content, "chat")
    # F26 - the uncertainty/clarification decision must happen BEFORE speech.
    # With a stream consumer attached every delta has already been spoken, so
    # swapping the reply here would speak one answer and store a different
    # permission question. When the stream produced nothing, nothing has been
    # spoken yet and the rewrite is still safe.
    if stream is not None and pieces:
        if re.search(_UNSURE_RE, content):
            print("[CHAT] Answer looked unsure - already spoken, keeping it verbatim.")
    elif re.search(_UNSURE_RE, content):
        print("[CHAT] Answer looked unsure - asking before researching.")
        if maybe_proactive_research(user_message):
            content = _confirmation_question()
    if commit_response:
        _commit_chat("assistant", content)
    print("[CHAT] Reply:", content)
    # G9 (F07): the chat exchange lands in the work-event store (bounded,
    # masked) - cross-restart continuity beyond the 20-message window.
    if memory_store is not None:
        try:
            memory_store.record_event(
                "chat_exchange",
                "user: %s" % user_message,
                request_id=request_id,
                detail={"reply": content[:300]},
            )
            # F07: the terminal event LINKED to the identified request. Chat is
            # a projection of the work-event store, not a parallel record.
            if request_id:
                memory_store.record_result(request_id, "completed",
                                           summary=content)
        except Exception:
            pass
    return content


def handle_chat(user_message, voice_compact=False, commit_response=True, stream=None, prebuilt=None, live_stream=None, answered=None):
    """Handle a chat message.

    If *stream* is a callable(delta_text), the reply is delivered token by
    token as it is generated (live typewriter feel) instead of waiting for
    the full response. *_commit_response* controls whether the final text is
    stored to conversation memory.

    [S6] *answered* is a reply the intent router already produced in the same
    call that classified the turn. When it is set the chat model is NOT called
    at all - one LLM call served both the route and the answer - and the text
    still travels the normal commit/stream/memory path.
    """
    print("\n[CHAT] Processing chat...")

    add_message("user", user_message)

    # F07: every handled request is IDENTIFIED before any work happens, so its
    # outcome, artifacts and provenance can be linked back to it.
    request_id = None
    if memory_store is not None:
        try:
            request_id = memory_store.begin_request(user_message, route="chat")
        except Exception:
            request_id = None

    if answered:
        # [S6] The router already wrote this answer. Emit it as a single delta
        # so a streaming consumer (voice/UI) sees exactly one reply, and skip
        # message building and the chat model entirely.
        content = re.sub(r"\s+", " ", str(answered)).strip()
        pieces = []
        if stream:
            stream(content)
            pieces = [content]
        return _finalize_chat_reply(
            user_message, content, pieces, stream, commit_response, request_id)

    built = prebuilt if prebuilt is not None else _build_chat_messages(user_message, voice_compact=voice_compact)

    if built["path"] == "needs_search":
        # F25 — the speculative build refused to search (it may still be
        # cancelled). We are now the selected route, so acquire the lookup
        # here and rebuild against the committed history.
        built = _build_chat_messages(user_message, voice_compact=voice_compact)

    if built["path"] == "browser_search":
        execute_multiple([{"action": "search", "input": user_message}])
        # R14: no unverified "I've opened a search for you" — the lookup
        # ran synchronously above, so report the fact, not a promise.
        response = "Sir, I ran a search for that."
        if commit_response:
            _commit_chat("assistant", response)
        if stream:
            stream(response)
        if memory_store is not None and request_id:
            try:
                memory_store.record_result(request_id, "completed",
                                           summary=response)
            except Exception:
                pass
        return response

    messages = built["messages"]
    pieces = []
    if stream:
        if live_stream is not None:
            for delta in live_stream:
                if delta:
                    pieces.append(delta)
                    stream(delta)
        else:
            for delta in _stream_chat_deltas(
                messages, 0.45 if voice_compact else 0.7, 300 if voice_compact else 1400
            ):
                if delta:
                    pieces.append(delta)
                    stream(delta)
        content = re.sub(r"\s+", " ", "".join(pieces)).strip()
    else:
        result = _ask_chat_nonstream(
            messages,
            temperature=0.45 if voice_compact else 0.7,
            max_tokens=300 if voice_compact else 1400,
        )
        if not result or not result.get("choices"):
            response = "I'm having trouble connecting. Please try again."
            if commit_response:
                _commit_chat("assistant", response)
            return response
        content = re.sub(r"\s+", " ", result["choices"][0]["message"]["content"]).strip()

    return _finalize_chat_reply(
        user_message, content, pieces, stream, commit_response, request_id)



def generate_command_response(actions):
    """R14: describe the REQUEST, never claim the outcome.

    The old lines ("Consider it done. Now playing", "On it") spoke action
    sentences before the worker thread proved anything. These lines only
    name what was asked — completion is announced by the result narrator
    after verified execution, never here. "Playing <song>" alone names the
    request; the gate distinguishes it from the claim "Playing ... now".
    """
    play_lines = [
        "{song}, sir.",
    ]
    open_lines = [
        "{site}, sir.",
    ]
    search_lines = [
        "Searching for {query}, sir.",
    ]

    responses = []
    for action in actions:
        if action["action"] == "youtube_play":
            song = action["input"].title()
            responses.append(random.choice(play_lines).format(song=song))
        elif action["action"] == "open_website":
            site = action["input"].replace(".com", "").title()
            responses.append(random.choice(open_lines).format(site=site))
        elif action["action"] == "launch_app":
            app = action["input"].replace(".com", "").title()
            responses.append(random.choice(open_lines).format(site=app))
        elif action["action"] == "search":
            responses.append(random.choice(search_lines).format(query=action["input"]))
    return " ".join(responses)


# ── Tool-intent execution with opencode fallback ──
def _execution_failed(steps, results):
    """Heuristic: an action failed when the executor reported so or results are empty."""
    if not steps:
        return True
    if results is None:
        return True
    if isinstance(results, str):
        return False
    if len(results) != len(steps):
        return True
    return any(not results[i] for i in range(len(steps)))


def handle_tool_intent(steps, original_message, from_voice=False, voice_compact=False):
    """Execute structured tool steps, acknowledge like a human, and route
    any opencode handoff through the confirmation gate.

    R14: the immediate ack only NAMES the request (see
    generate_command_response) — it never claims done/started/queued. The
    real outcome is announced after verified execution (or via the gated
    handoff's own narrators). If execution fails, the opencode handoff is
    armed for spoken confirmation and the real result is delivered async
    via the async-reply callback after the user confirms.
    """
    if not steps:
        # No local steps — route through the gated opencode handoff instead
        # of running opencode directly (never execute unconfirmed).
        return handle_opencode_task(
            original_message, original_message, voice_compact=voice_compact
        )

    ack = generate_command_response(steps)
    if voice_compact and len(ack) > 120:
        ack = "On it, sir."

    def _run():
        try:
            results = execute_multiple(steps)
        except Exception as exc:
            logging.warning("[TOOL] Execution error: %s", exc)
            results = None

        if not _execution_failed(steps, results):
            return

        # Local execution didn't work — route the handoff through the
        # confirmation gate instead of running it unconfirmed.
        # F16 (G8): engine AVAILABILITY is separate from PERMISSION. The
        # browser agent recovers local-execution failures WITHOUT the
        # opencode-installation prerequisite; opencode itself remains
        # opt-in (only when the user explicitly asked for the coding agent).
        # The decision is FROZEN into an immutable contract here and carried
        # through the confirmation, so it cannot drift before execution.
        from backend.services import capability_contract as _contracts
        from backend.services.capability_resolver import recovery_after_failure
        decision = recovery_after_failure(
            original_message,
            availability={"opencode": is_opencode_available(), "editor": True},
            task_engine=config.TASK_ENGINE,
        )
        contract = None
        try:
            contract = _contracts.contract_for(
                decision, original_message, grant="tool-intent-recovery")
        except Exception as exc:
            logging.warning("[TOOL] no execution contract for recovery: %s", exc)
        print("[TOOL] Local execution failed — arming gated %s handoff (%s)." % (
            decision["engine"], decision["reason"]))
        _notify_async_reply(handle_opencode_task(
            original_message, original_message, contract=contract))

    threading.Thread(target=_run, daemon=True).start()
    return ack


def _truncate_at_word(text, budget):
    """Cut `text` at the last word boundary within `budget` chars."""
    if len(text) <= budget:
        return text
    cut = text.rfind(" ", 0, budget + 1)
    if cut <= 0:
        cut = budget
    return text[:cut].rstrip()


def _voice_clip(text, limit=180):
    """Sentence-aware clip for spoken replies: prefer a clean sentence end."""
    if not text or len(text) <= limit:
        return text
    kept = text[: limit - 3].rstrip()
    term_idx = max(kept.rfind(t) for t in (".", "!", "?"))
    if term_idx >= 100:
        return kept[: term_idx + 1]
    word_idx = kept.rfind(" ")
    if word_idx > 0:
        kept = kept[:word_idx].rstrip()
    return kept + "..."


def _browser_prewarm_due(contract):
    """True when the armed handoff will run the browser agent (L-6).

    A frozen contract naming another engine is authoritative (never prewarm
    for it); without a contract the configured engine decides. Warming the
    wrong pool is harmless (idle TTL reaps it, no side effects), so this
    errs toward warming.
    """
    try:
        if contract is not None:
            return str(getattr(contract, "executor", "") or "") == "browser_agent"
        return config.TASK_ENGINE == "browser_agent"
    except Exception:
        return False


def handle_opencode_task(task_description, original_message,
                         from_voice=False, voice_compact=False, contract=None):
    """Route a complex command to the opencode agent.

    Never executes on the first turn. Arms a spoken confirmation — "sir,
    this is what I understood - <task>. Do you want me to go ahead and
    execute it?" — and only hands off to opencode after the user confirms
    on the next turn (see _consume_opencode_confirmation).

    F16: *contract* (when supplied) is the already-frozen execution decision;
    it is stored with the armed confirmation so execution cannot re-decide.
    """
    _arm_opencode_confirmation(task_description, original_message,
                               contract=contract)
    # L-6: while the user reads the confirmation prompt, warm the MCP
    # session + tool cache on a daemon thread — connect + tools/list then
    # happen inside the approval gap instead of after "yes". Best effort:
    # never delays the question, never navigates, never spawns the daemon.
    if _browser_prewarm_due(contract):
        try:
            threading.Thread(target=browser_agent.prewarm_mcp_pool,
                             name="mcp-prewarm", daemon=True).start()
        except Exception:
            pass
    suffix = "Do you want me to go ahead and execute it?"
    task_text = task_description or original_message
    prefix = "Sir, this is what I understood — "
    question = f"{prefix}{task_text}. {suffix}"
    if voice_compact and len(question) > 180:
        # Never truncate the trailing question: budget the task text only.
        budget = 180 - len(prefix) - len("... ") - len(suffix)
        question = f"{prefix}{_truncate_at_word(task_text, budget)}... {suffix}"
    return question


# ── R12 status grounding: answer "are you doing X?" from live state ────
# A status question must NEVER be answered from chat recall ("that topic has
# not been raised"). The speaker reads a live snapshot — armed confirmations,
# running jobs, queue entries, last verified result — and answers from that.
# Astra §4 utterance table; every reply below is a fixed controlled form.
_STATUS_QUESTION_RE = re.compile(
    r"\b("
    r"are\s+you\s+(doing|working\s+on|running|executing)|"
    r"is\s+(it|that|the\s+\w+\s+task)\s+(done|finished|complete|completed|running|started)|"
    r"did\s+you\s+(finish|complete|do|start|stop|cancel)|"
    r"have\s+you\s+(finished|completed|started|done)|"
    r"what(?:'s|\s+is)\s+(?:the\s+)?(?:status|queued|running|happening)|"
    r"anything\s+(queued|running|pending)|"
    r"are\s+you\s+done|"
    r"is\s+it\s+done"
    r")\b",
    re.IGNORECASE,
)

# "cued task" is STT noise for "queued task" in status language (transcript
# turn: "are you doing the cued task right now"). Repair ONLY here — never in
# filenames or task descriptions.
_CUED_TASK_RE = re.compile(r"\bcued\s+tasks?\b", re.IGNORECASE)
_Q_TEST_RE = re.compile(r"\bq[\s-]?test\b", re.IGNORECASE)


def is_status_question(msg):
    """R12: True when the turn asks about live work, not new work."""
    if not msg:
        return False
    text = str(msg)
    if _CUED_TASK_RE.search(text):
        return True
    return bool(_STATUS_QUESTION_RE.search(text))


#: R10 — pronouns resolve from the notebook, not the nearest noun.
#: "that folder" = focus-head folder (active task); "there" = a PLACE
#: (folder/directory/location), never a file or a browser page; bare "it"
#: = the focus-head folder when one exists, AMBIGUOUS when a file and a
#: folder both fit (caller asks once).
_THAT_FOLDER_RE = re.compile(
    r"\bthat\s+(?:folder|directory)\b|\bthe\s+same\s+(?:folder|directory)\b",
    re.IGNORECASE,
)
_THERE_RE = re.compile(r"\bthere\b", re.IGNORECASE)
_BARE_IT_RE = re.compile(r"\bit\b", re.IGNORECASE)

#: "queued task" names a REAL queue entry: the S18 action queue, a held R6
#: redirect, or an armed confirmation awaiting yes. Anything else is "nothing
#: is queued" — never invented.
_QUEUED_TASK_RE = re.compile(
    r"\bqueued\s+task\b|\bqueue\b",
    re.IGNORECASE,
)


def resolve_that_folder():
    """R10: "that folder" from the notebook focus head, or None.

    One resolver, used by the folder-hint path and any future pronoun
    route — never the nearest noun in the current sentence.
    """
    try:
        return notebook_focus_folder()
    except Exception:
        return None


def resolve_there():
    """R10: "there" is always a place — the focus-head folder, or None.

    Never a file, never a browser page: when no folder is in focus the
    caller asks once instead of guessing.
    """
    return resolve_that_folder()


def resolve_bare_it(text):
    """R10: bare "it" from notebook focus — folder path, "ask", or None.

    Returns (kind, value): ("folder", path) when the focus head decides;
    ("ask", "it ...") when a file entity and a folder entity both fit the
    job (caller asks once); (None, "") when no "it" is present at all.
    """
    if not _BARE_IT_RE.search(text or ""):
        return None, ""
    folder = resolve_that_folder()
    files = []
    try:
        with _notebook_lock:
            entries = list(_notebook_entities)
        files = [e for e in entries if e.get("kind") == "file"
                 and e.get("path")]
    except Exception:
        files = []
    if folder and files:
        # File-vs-folder both fit "it" — Astra: ask once, never guess.
        return "ask", ("Sir, by 'it' do you mean the folder %s or the file "
                       "%s?" % (folder, files[-1].get("path")))
    if folder:
        return "folder", folder
    return None, ""


def resolve_last_command():
    """R10: "last command" = last REAL work from the notebook, or None.

    Skips own messages, bare confirmations, stop controls, and status
    questions — the same skip rules as _record_last_work_request, but read
    from the notebook ledger so it survives restarts of the globals.
    """
    try:
        with _notebook_lock:
            entries = list(_notebook_requests)
    except Exception:
        return None
    for entry in reversed(entries):
        state = str(entry.get("state") or "")
        if state in ("superseded",):
            continue
        return str(entry.get("text") or "")
    return None


def resolve_queued_task():
    """R10: "queued task" from the real queue entries, or "nothing queued".

    Returns (kind, value): ("action", text) for the oldest S18 entry,
    ("held", text) for the R6 held redirect, ("approval", text) for an
    armed confirmation, ("none", "Nothing is queued, sir.") otherwise.
    """
    try:
        with _action_queue_lock:
            queued = list(_pending_action_requests)
        if queued:
            return "action", str(queued[0].get("message") or "")
    except Exception:
        pass
    try:
        with _held_redirect_lock:
            held = _held_redirect
        if held and held.get("text"):
            return "held", str(held.get("text"))
    except Exception:
        pass
    try:
        from backend.services.task_agent import agent as _ta
        if _ta.has_pending_task_confirmation():
            return "approval", "a file task awaiting your approval"
    except Exception:
        pass
    try:
        with _opencode_confirm_lock:
            if _pending_opencode_task:
                return "approval", "a browser task awaiting your approval"
    except Exception:
        pass
    return "none", "Nothing is queued, sir."


def _status_snapshot():
    """R12: read the live operational state in one consistent snapshot.

    Sources: armed opencode/native confirmations, the running flag, the
    S18 action queue, the pending browser clarification, and the last
    verified native result. Returns a plain dict — the speaker below only
    reads this, never live globals.
    """
    try:
        with _opencode_confirm_lock:
            pending_opencode = dict(_pending_opencode_task or {})
    except Exception:
        pending_opencode = {}
    try:
        from backend.services.task_agent import agent as _ta
        pending_native = bool(_ta.has_pending_task_confirmation())
    except Exception:
        pending_native = False
    try:
        with _action_queue_lock:
            queued = list(_pending_action_requests)
    except Exception:
        queued = []
    try:
        with _browser_clarification_lock:
            clarification = dict(_pending_browser_clarification or {})
    except Exception:
        clarification = {}
    running = bool(_opencode_task_running)
    researching = bool(_research_running)
    last_status, last_summary = "", ""
    try:
        result, task_text = _last_task_result()
        if result is not None:
            last_status = str(getattr(result, "status", "") or "")
            last_summary = str(
                getattr(result, "summary", "")
                or getattr(result, "detail", "") or task_text or "")
    except Exception:
        pass
    return {
        "pending_opencode": bool(pending_opencode),
        "pending_native": bool(pending_native),
        "awaiting_approval": bool(pending_opencode) or pending_native,
        "running": running,
        "researching": researching,
        "queued": len(queued),
        "clarification": bool(clarification),
        "last_status": last_status,
        "last_summary": re.sub(r"\s+", " ", last_summary).strip()[:160],
    }


def answer_status_question(msg):
    """R12: the grounded status reply for *msg* (Astra §4 table).

    Reads _status_snapshot() and returns a controlled-form sentence. Never
    consults chat history or the LLM. Never invents monitoring processes.
    Uncertain states say so plainly ("I cannot verify...").
    """
    snap = _status_snapshot()
    text = _CUED_TASK_RE.sub("queued task", str(msg or ""))
    lowered = text.lower()
    asks_queue = bool(re.search(r"\bqueu", lowered))

    # Queue-specific questions get queue-grounded answers first — R10 reads
    # the REAL entries (S18 queue, held redirect, armed approval), never an
    # invented count.
    if asks_queue:
        kind, value = resolve_queued_task()
        if kind == "none":
            if snap["running"]:
                return "Sir, a task is running. Nothing is queued."
            return "Nothing is queued, sir."
        if snap["running"]:
            return ("Sir, the browser task is running. "
                    "Queued: %s — not started." % value[:120])
        return "Not yet, sir. Queued: %s." % value[:120]

    # "Q test" was never a tracked task: ask once, never assert absence.
    if _Q_TEST_RE.search(text):
        if snap["running"] or snap["queued"] or snap["awaiting_approval"]:
            return ("Sir, I have a task in motion, but I have no task "
                    "matching that description. Do you mean the file task "
                    "or the browser task?")
        return ("Sir, I have no task matching that description. "
                "What would you like me to work on?")

    # Precedence: running > queued > awaiting approval > clarification >
    # last verified result > idle. Combined states name both facts.
    if snap["running"] and snap["queued"]:
        return ("Sir, the browser task is running. "
                "The other task is queued, not started.")
    if snap["running"]:
        return "Yes, sir. A task is running right now."
    if snap["queued"]:
        return "Not yet, sir. The task is queued."
    if snap["awaiting_approval"]:
        return "No, sir. The task is waiting for your approval."
    if snap["researching"]:
        return "Yes, sir. A search is running right now."
    if snap["clarification"]:
        return "Sir, quick question is waiting on you before I continue."
    status = snap["last_status"]
    if status == "completed":
        if snap["last_summary"]:
            return "Yes, sir. Done — %s." % _voice_clip(
                snap["last_summary"], 140)
        return "Yes, sir. The last task finished."
    if status == "partial":
        return ("Partly, sir. The last task is only partly done"
                + (" — %s." % _voice_clip(snap["last_summary"], 120)
                   if snap["last_summary"] else "."))
    if status in ("failed", "known_failure"):
        return "No, sir. The last attempt failed."
    if status == "stopped":
        return "Sir, the task was stopped. No further changes were verified."
    if status == "needs_input":
        return "Sir, I asked a question before continuing."
    return "Sir, nothing is running right now."


# ── Dual-voice mute: while an opencode task runs, only opencode speaks ──
_opencode_task_running = False

# Whether a research/quick-search is currently running in the background
# (handle_research_intent daemon thread).  Used to gate the stop-research
# event so an idle stop utterance never poisons the next search.
_research_running = False

# Mandatory announcement made exactly at handoff — exempt from the mute so
# the UI path never swallows it (the flag is already True by then).
# R14: these name the HANDOFF (a real event — the worker thread just
# started), never the outcome. "Started/done/queued" are spoken only by
# the result narrator after verification, never here.
OPENCODE_START_PHRASE = "Handing the task to opencode, sir."
# The browser-agent engine makes its own announcement with the same status.
BROWSER_AGENT_START_PHRASE = "Taking over the browser task, sir."


def opencode_task_in_progress():
    """True while the opencode subprocess is executing.

    Consumers (voice_mode, routes) mute Jarvis's own replies for the whole
    duration so the opencode agent's narration is never echoed back or
    talked over.
    """
    return _opencode_task_running


def set_opencode_task_running(running):
    global _opencode_task_running
    running = bool(running)
    changed = _opencode_task_running != running
    _opencode_task_running = running
    if changed:
        # [S19] Push the flip to event subscribers so the voice worker and UI
        # mute/unmute instantly instead of on their next /ui-state poll.
        event_bus.publish("task_running", {"task_running": running})
    # [S18] The tools are free again — anything the user asked for mid-task
    # that needed them is run now, in order, on its own thread.
    if not _opencode_task_running:
        _drain_queued_actions()


# ── [S18] chat-through-task: actions queue, conversation continues ─────────
# While a task runs the user may keep TALKING (chat turns answer normally);
# only requests that need the executor/task machinery are held until the
# running task releases it. Echo safety is the AEC gate's job (S28), not a
# full-channel mute.
_action_queue_lock = threading.Lock()
_pending_action_requests = []
MAX_QUEUED_ACTIONS = 3

ACTION_QUEUE_FULL_REPLY = ("Sir, the action queue is full — ask me again "
                           "when the current task finishes.")
ACTION_QUEUED_REPLY = ("Sir, I'll run that as soon as the current task "
                       "finishes — it's queued.")


def _queue_action_request(msg, from_voice):
    """Hold an action request made while a task runs.

    Returns the polite hold reply when the request was queued (or the queue
    is full), or None when no task is running and the caller must proceed
    with the normal dispatch.
    """
    if not _opencode_task_running:
        return None
    with _action_queue_lock:
        if len(_pending_action_requests) >= MAX_QUEUED_ACTIONS:
            return ACTION_QUEUE_FULL_REPLY
        _pending_action_requests.append(
            {"message": msg, "from_voice": bool(from_voice)})
    return ACTION_QUEUED_REPLY


def _drain_queued_actions():
    with _action_queue_lock:
        pending = list(_pending_action_requests)
        _pending_action_requests.clear()
    if not pending:
        return

    def _run():
        for item in pending:
            try:
                reply = process_message(item["message"],
                                        from_voice=item["from_voice"])
            except Exception as exc:
                logging.warning("[TASK] queued action failed: %s", exc)
                continue
            if item["from_voice"] and reply:
                # F10 surface: UI log + speech (+ [S13] history).
                _notify_async_reply(reply)

    threading.Thread(target=_run, name="queued-action-drain",
                     daemon=True).start()


def _execute_deferred_opencode(task_description, original_message,
                               resume_from=None, contract=None):
    """Run the deferred task and report the result asynchronously.

    F16: the EXECUTOR is frozen before anything runs. The old code re-derived
    the engine from ``config.TASK_ENGINE`` at execution time, so a settings
    change between consent and execution could silently hand the work to a
    different engine. *contract* is the immutable, digest-checked execution
    contract frozen at consent (or frozen here when the caller had none);
    ``BLOCKED`` means nothing runs and the user is told why.

    The speech contract is the same for both engines — exactly the start
    phrase here, then only the agent speaks, then the real answer is spoken
    voice-clipped by _voice_clip (the full summary still goes to the chat UI).

    The task-running flag flips BEFORE the worker thread starts so the mute
    is deterministic from the moment of handoff; callers exempt the start
    phrase from the mute so it is never swallowed by the guard.

    *resume_from* (F08) continues a SUSPENDED browser run from its checkpoint
    instead of starting a fresh one, so already-committed actions are not
    replayed.
    """
    from backend.services import capability_contract as _contracts
    from backend.services import capability_resolver as _resolver

    if contract is None:
        contract = _resolver.begin_dispatch(
            task_description,
            availability={"opencode": is_opencode_available(), "editor": True},
            task_engine=config.TASK_ENGINE,
            grant="user-confirmed-handoff",
        )
    executor = getattr(contract, "executor", None) or config.TASK_ENGINE
    # F26: this handoff owns its own identity. A run that is superseded (a new
    # handoff, or a stop) can no longer deliver its result, its question or its
    # speech into the newer request.
    run_id = _new_browser_run()
    # F07: the handed-off task is an IDENTIFIED request in its own right, so
    # its suspension, artifacts and terminal outcome all link to one id.
    work_request_id = None
    if memory_store is not None:
        try:
            work_request_id = memory_store.begin_request(
                task_description, route=executor,
                provenance="resume" if resume_from else "handoff")
        except Exception:
            work_request_id = None

    def _run():
        global _pending_browser_clarification
        print("[TASK] Handing off to %s:" % executor, task_description)
        if executor == _resolver.BLOCKED:
            # F16: no compatible executor — fail closed, run NOTHING, and say
            # so instead of quietly degrading to a different engine.
            set_opencode_task_running(False)
            set_narration_enabled(False)
            if memory_store is not None and work_request_id:
                try:
                    memory_store.record_result(
                        work_request_id, "refused",
                        summary="no compatible executor for this task",
                        engine="blocked")
                except Exception:
                    pass
            _notify_async_reply(
                "I can't do that one, sir — no executor I'm allowed to use "
                "can handle it. Nothing was started.")
            return
        try:
            contract.verify()
        except Exception as exc:
            # The frozen decision changed underneath us: refuse rather than
            # run something other than what the user approved.
            set_opencode_task_running(False)
            set_narration_enabled(False)
            _notify_async_reply(
                "I stopped that one, sir — the approval no longer matches "
                "what would run (%s). Nothing was started." % exc)
            return
        if executor == _resolver.BROWSER_AGENT or executor == "browser_agent":
            try:
                if resume_from:
                    output = run_browser_task(task_description,
                                              resume_from=resume_from)
                else:
                    output = run_browser_task(task_description)
            except Exception as exc:
                logging.warning("[TASK] browser task crashed: %s", exc)
                output = None
            finally:
                set_opencode_task_running(False)
                set_narration_enabled(False)
            # F03: branch on the TaskResult status, never on substrings.
            result = _coerce_task_result(output)
            if not _browser_run_is_current(run_id):
                logging.info(
                    "[TASK] browser run %s finished after being superseded; "
                    "result dropped.", run_id)
                return
            detail_text = result.detail or str(result)
            # G9 (F07/F09): terminal result -> bounded event; a VERIFIED
            # completion also captures a candidate skill (approval-gated).
            if memory_store is not None:
                try:
                    memory_store.record_task_outcome(
                        "browser_agent", result.status, task_description,
                        summary=result.summary or detail_text[:200],
                        evidence=list(result.evidence or []),
                        request_id=work_request_id,
                        trace=list(getattr(result, "trace", None) or []),
                        verification=list(
                            getattr(result, "verification", None) or []),
                    )
                    # F07: the artifacts the run actually produced (report
                    # paths, downloaded files, visited sources) are stored as
                    # STRUCTURE against the same request.
                    memory_store.record_result(
                        work_request_id, result.status,
                        summary=result.summary or detail_text[:200],
                        artifacts=_task_result_artifacts(result),
                        evidence=list(result.evidence or []),
                        engine="browser_agent",
                    )
                except Exception:
                    pass

            if result.status == "needs_input" or (
                result.status == "completed"
                and _is_browser_clarifying_question(result.summary)
            ):
                # F08: the suspended run keeps its identity. The answer later
                # RESUMES this checkpoint (verified progress + committed
                # effects) instead of starting a fresh run that replays them.
                checkpoint_id = getattr(result, "checkpoint", None)
                if not checkpoint_id:
                    checkpoint_id = _suspend_browser_checkpoint(
                        task_description, result.summary)
                with _browser_clarification_lock:
                    _pending_browser_clarification = {
                        "task_description": task_description,
                        "question": result.summary,
                        "checkpoint_id": checkpoint_id,
                        "request_id": work_request_id,
                        "expires": time.time() + _BROWSER_CLARIFICATION_TTL,
                    }
                # F07: the SUSPENSION is an event of the same identified
                # request — the request stays open, waiting for the user.
                if memory_store is not None and work_request_id:
                    try:
                        memory_store.record_suspension(
                            work_request_id, result.summary, checkpoint_id)
                    except Exception:
                        pass
                # Pending question, not a result — no success prefix.
                _set_last_browser_detail(detail_text)
                question = re.sub(r"\s+", " ", result.summary).strip()
                browser_summary = "Sir, quick question — %s" % _voice_clip(question)
                _notify_async_reply(
                    browser_summary,
                    spoken=browser_summary,
                )
            elif result.status == "stopped":
                # User abort — no success prefix.
                _set_last_browser_detail(detail_text)
                browser_summary = str(result).strip()
                _notify_async_reply(
                    browser_summary,
                    spoken=browser_summary,
                )
            elif result.status == "partial":
                # Honest partly-done headline — never 'Sir, done'.
                _set_last_browser_detail(detail_text)
                browser_summary = _summarize_browser_output(
                    result.summary or detail_text, task_description,
                    status="partial", evidence=result.evidence,
                )
                _notify_async_reply(
                    browser_summary,
                    spoken=browser_summary,
                )
            elif result.status == "completed":
                # Short headline for speech + chat; full output stays in
                # the detail store (activity log holds RESULT ok too).
                _set_last_browser_detail(detail_text)
                browser_summary = _summarize_browser_output(
                    result.summary, task_description, status="completed")
                _notify_async_reply(
                    browser_summary,
                    spoken=browser_summary,
                )
            else:
                _set_last_browser_detail(detail_text)
                browser_summary = _summarize_browser_output(
                    str(result), task_description, status="failed")
                _notify_async_reply(
                    browser_summary,
                    spoken=browser_summary,
                )
            return
        if not is_opencode_available():
            set_opencode_task_running(False)
            set_narration_enabled(False)
            _notify_async_reply(
                "I can't do that right now, sir — opencode isn't installed."
            )
            return
        try:
            output = run_opencode_task(task_description, contract=contract)
        except Exception as exc:
            logging.warning("[TASK] opencode task crashed: %s", exc)
            output = None
        finally:
            set_opencode_task_running(False)
            set_narration_enabled(False)
        # F03: branch on the explicit status — only 'completed' may ever
        # claim success (future-proof: run_opencode_task returns
        # completed|failed today, but a truthy non-completed result must
        # never take the success path).
        result = _coerce_task_result(output)
        # G9 (F07/F09): terminal result -> bounded event; a VERIFIED
        # completion also captures a candidate skill (approval-gated).
        if memory_store is not None:
            try:
                memory_store.record_task_outcome(
                    "opencode", result.status, task_description,
                    summary=result.summary or (result.detail or "")[:200],
                    evidence=list(result.evidence or []),
                    trace=list(getattr(result, "trace", None) or []),
                    verification=list(
                        getattr(result, "verification", None) or []),
                )
            except Exception:
                pass
        if result.status == "completed":

            summary = _summarize_opencode_output(
                result.summary, status="completed")
            _notify_async_reply(
                summary,
                spoken=_voice_clip(summary),
            )
        elif result.status in ("partial", "stopped", "needs_input"):
            if result.status == "stopped":
                summary = "Sir, the task was stopped."
            elif result.status == "needs_input":
                question = re.sub(r"\s+", " ", result.summary or "").strip()
                summary = "Sir, quick question — %s" % _voice_clip(question)
            else:
                summary = _summarize_opencode_output(
                    result.summary or result.detail, status="partial")
            _notify_async_reply(
                summary,
                spoken=_voice_clip(summary),
            )
        elif result.error == "internal crash":
            # Engine raised (old output=None path): keep the legacy snag line.
            # R14: no "I started the task" — the worker never proved start.
            _notify_async_reply("Sir, the task could not start.")
        else:
            summary = _summarize_opencode_output(
                result.detail, status="failed", error=result.error)
            _notify_async_reply(
                summary,
                spoken=_voice_clip(summary),
            )

    set_opencode_task_running(True)
    set_narration_enabled(True)
    _handed_off = []

    def _hand_off_once():
        # F50: the hand-off body runs exactly once, whichever runtime reached it.
        if _handed_off:
            return
        _handed_off.append(True)
        _run()

    def _typed_hand_off():
        """Run the hand-off as ONE typed effect in the shared job runtime.

        F50: a hand-off is typed work, so it gets the same turn-job
        cancellation, checkpoints and journalled history as voice and
        background work instead of being an untracked daemon thread.
        """
        try:
            from backend.core import jobs as _jobs
            from backend.services import intelligence_state as _istate
        except Exception:
            _hand_off_once()
            return
        try:
            _istate.run_effect(
                _jobs.EFFECT_TYPED, "deferred-handoff",
                lambda job: _hand_off_once(),
                label=(task_description or "hand-off")[:60],
                owner=_istate.ROLE_TYPED)
        except Exception as _effect_exc:
            logging.debug("[TASK] typed effect failed (%s); running the "
                          "hand-off directly", _effect_exc)
            _hand_off_once()

    try:
        threading.Thread(target=_typed_hand_off, daemon=True).start()
    except Exception:
        set_opencode_task_running(False)
        set_narration_enabled(False)
        raise
    if config.TASK_ENGINE == "browser_agent":
        return BROWSER_AGENT_START_PHRASE
    return OPENCODE_START_PHRASE


# ── Deep web research (headed browser on the user's real profile) ──
def push_research_result(payload):
    """Push a finished research report to the overlay.

    Primary path: the backend API. Mirror path: a `latest.json` file next to
    the saved report — the Electron overlay polls BOTH, so a stale backend
    (old uvicorn without /research-result) can never again leave the overlay
    stuck on "waiting for report".
    """
    result = dict(payload)
    result["id"] = int(time.time() * 1000)
    try:
        resp = requests.post(
            f"http://127.0.0.1:{BACKEND_PORT}/research-result",
            json=result,
            timeout=2,
        )
        if resp.status_code != 200:
            logging.warning(
                "[RESEARCH] Backend push returned HTTP %s — overlay falls back to file.",
                resp.status_code,
            )
    except Exception:
        logging.warning("[RESEARCH] Backend push failed — overlay falls back to file.")

    try:
        reports_dir = os.path.dirname(payload.get("report_path") or "")
        if not reports_dir:
            reports_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data", "research_reports",
            )
        os.makedirs(reports_dir, exist_ok=True)
        with open(
            os.path.join(reports_dir, "latest.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logging.warning("[RESEARCH] Could not write latest.json: %s", exc)


def push_research_progress(query, message, evidence=None):
    """Publish live progress + the evidence gathered so far (F28).

    A deep research run happens on a background thread long after the chat
    request that started it has finished streaming, so progress cannot ride
    the SSE stream — it gets its own endpoint. Best-effort by design: a
    research run must never fail because the overlay was not listening.
    """
    try:
        requests.post(
            f"http://127.0.0.1:{BACKEND_PORT}/research-progress",
            json={
                "id": int(time.time() * 1000),
                "query": query,
                "message": message,
                "evidence": [dict(e) for e in (evidence or [])],
            },
            timeout=1.5,
        )
    except Exception:
        pass


# ── Explicit websearch stop ("stop the research") ──
# Substring matching, consistent with is_continue/is_normal_setup routing,
# with a negation gate: "don't stop the research" is NOT a stop request.
STOP_RESEARCH_PHRASES = (
    "stop the research", "stop the search", "stop researching",
    "stop the deepsearch", "stop deepsearch",
    # R6: the typed/voiced browser-task stop had NO brain text route — only
    # /task/stop HTTP and the 5 research phrases above. "Stop the browser
    # task" must signal the worker through the same control path.
    "stop the browser task", "stop browser task", "stop the browser",
    "stop browser",
)
_STOP_RESEARCH_NEGATION_RE = re.compile(
    r"(?:dont|don't|do not|never)\s+(?:.{0,30})?(?:"
    + "|".join(re.escape(p) for p in STOP_RESEARCH_PHRASES if "browser" not in p)
    + r")"
)


#: R8 — one turn can be TWO jobs, never one queued blob. A question +
#: a request ("are you doing X? also create Y", "is it done — and make Z")
#: answers the status from live state FIRST, then routes the work half as a
#: fresh turn (its own preview/approval, never inheriting anything). Stop +
#: redirect compounds keep their dedicated control-first path (R6); this is
#: for status+work pairs. Pure status ("did you stop it?") never splits —
#: the work half must be non-trivial text.
_TURN_SPLIT_RE = re.compile(
    r"^(?P<first>.+?)\s*(?:[?!.…,;—–-]+\s*|\s+and\s+|\s+also\s+|\s+then\s+)"
    r"(?P<second>(?:create|make|write|save|check|verify|list|inspect|see|"
    r"show|search|find|look(?:\s+up)?|open|run|execute|stop)\b.+)$",
    re.IGNORECASE | re.DOTALL,
)


def split_compound_turn(msg):
    """R8: split a status+work turn into (status_half, work_half).

    Returns (None, None) when the turn is NOT such a compound: the first
    half must be a real status question, the second a non-trivial work
    request. Corrections, confirmations, and bare status never split — they
    have their own gates.
    """
    text = (msg or "").strip()
    if not text or len(text) < 12:
        return None, None
    try:
        if is_correction(text):
            return None, None
    except Exception:
        pass
    match = _TURN_SPLIT_RE.match(text)
    if not match:
        return None, None
    first = (match.group("first") or "").strip()
    second = (match.group("second") or "").strip()
    if len(second) < 8:
        return None, None
    try:
        if not is_status_question(first):
            return None, None
        if is_status_question(second):
            return None, None
    except Exception:
        return None, None
    return first, second


#: R6 — "stop X and do Y" is TWO jobs (control + redirect), never one queued
#: blob. Splitter runs before the stop handler: the stop half goes through
#: the control path NOW, the redirect half is held behind the browser's
#: quiescence and never queued behind a task we claimed to stop.
_STOP_AND_REDIRECT_RE = re.compile(
    r"^(?P<stop>.*?stop\s+(?:the\s+)?(?:browser(?:\s+task)?|research|search|it|that|everything).*?)"
    r"\s+(?:and|then)\s+(?P<redirect>.+)$",
    re.IGNORECASE | re.DOTALL,
)

#: Redirect references that need the last real work request ("execute the
#: last command", "do the last thing", "run that again"). Excludes the
#: assistant's own messages, confirmations, status turns, and stop controls.
_LAST_COMMAND_RE = re.compile(
    r"\b(?:last\s+(?:command|task|thing|request)|that\s+again|do\s+it\s+again|"
    r"run\s+that\s+again|execute\s+(?:it|that)\s+again)\b",
    re.IGNORECASE,
)

#: R6 — a held redirect waits for browser quiescence before it may run.
#: Keys: text (the redirect utterance, resolved lazily at release so folder
#: resolution sees fresh state), armed_at. Guarded: stop+redirect can arrive
#: on any thread.
_held_redirect_lock = threading.Lock()
_held_redirect = None


def _record_last_work_request(msg):
    """R6: remember the last SUBSTANTIVE user work request.

    Status questions, bare confirmations ("yes"/"no"), stop controls, and
    "command"-prefixed lines are not work — the redirect resolver must skip
    them so "execute the last command" means the last real thing asked.
    Also mirrors into the R2 notebook (focus stack + entity table).
    """
    text = (msg or "").strip()
    if not text or text.lower().startswith("command"):
        return
    try:
        if is_status_question(text):
            return
    except Exception:
        pass
    lowered = text.lower()
    if re.match(r"^(yes|yeah|yep|yup|no|nope|nah|sure|ok(ay)?|alright|"
                r"haan|ha|kar\s*do|karo|proceed|continue|do\s+it)\b",
                lowered):
        return
    if re.search(r"\bstop\s+(the\s+)?(browser|research|search|it|that|everything)\b",
                 lowered):
        return
    global _last_user_work_request
    _last_user_work_request = text
    try:
        notebook_record_request(text)
    except Exception:
        pass


#: The last substantive user work request (see _record_last_work_request).
_last_user_work_request = ""


# ── R2 notebook: one small authoritative ledger across turns ──────────
# Scattered globals used to own cross-turn truth separately (pending
# opencode preview, pending native plan, S18 queue, held redirect, last
# result, last work text). The notebook does NOT replace them — it is the
# one READ model over them: request records (turn, kind, state), entity
# table (folders/files with canonical paths + focus order), approval and
# result mirrors. Resolvers ("that folder", status speaker) read the
# notebook; writers update it at the same points that arm/clear the
# underlying globals, so it can never drift from live state.
_notebook_lock = threading.Lock()
_notebook_requests = []
_NOTEBOOK_REQUESTS_MAX = 20
_notebook_entities = []
_NOTEBOOK_ENTITIES_MAX = 12
_notebook_seq = 0

#: Request kinds the notebook tracks (Astra §2 task states, compressed).
_NOTEBOOK_KIND_RE = re.compile(
    r"\b(create|make|write|save|check|verify|list|inspect|see|show|"
    r"search|find|look\s+up|open|run|execute|stop)\b",
    re.IGNORECASE,
)


def notebook_record_request(text):
    """R2: append one request record; returns its id (req-N)."""
    global _notebook_seq
    with _notebook_lock:
        _notebook_seq += 1
        rid = "req-%d" % _notebook_seq
        kind = "work"
        try:
            kind_m = _NOTEBOOK_KIND_RE.search(text or "")
            kind = kind_m.group(1).lower() if kind_m else "work"
        except Exception:
            pass
        _notebook_requests.append({
            "id": rid, "text": str(text or "")[:300], "kind": kind,
            "state": "seen", "turn": _notebook_seq,
            "at": time.time(),
        })
        while len(_notebook_requests) > _NOTEBOOK_REQUESTS_MAX:
            _notebook_requests.pop(0)
        return rid


def notebook_mark_state(rid, state):
    """R2: move a request record to a new task state (no-op if unknown)."""
    if not rid:
        return
    with _notebook_lock:
        for entry in _notebook_requests:
            if entry.get("id") == rid:
                entry["state"] = str(state)
                entry["at"] = time.time()
                return


def notebook_last_request(kind=None):
    """R2: the most recent request record (optionally of one kind)."""
    with _notebook_lock:
        entries = list(_notebook_requests)
    for entry in reversed(entries):
        if kind and entry.get("kind") != kind:
            continue
        return dict(entry)
    return None


def notebook_record_entity(name, path, kind="folder", source="observed"):
    """R2: upsert a folder/file entity with its canonical path + focus.

    Most-recently-touched entity is the focus head: "that folder" resolves
    here first, before the last-native-artifacts fallback.
    """
    if not path:
        return
    canon = os.path.normpath(str(path))
    with _notebook_lock:
        _notebook_entities[:] = [
            e for e in _notebook_entities
            if os.path.normpath(str(e.get("path") or "")) != canon]
        _notebook_entities.append({
            "name": str(name or ""), "path": canon, "kind": str(kind),
            "source": str(source), "at": time.time(),
        })
        while len(_notebook_entities) > _NOTEBOOK_ENTITIES_MAX:
            _notebook_entities.pop(0)
    # Rank 1: every notebook entity also lands in the entity ledger (the
    # richer store the reference resolver scores) — one funnel, no drift.
    try:
        entity_ledger.record_entity(name, canon, kind=str(kind),
                                    source=str(source),
                                    identifiers={"path": canon})
    except Exception:
        pass


def notebook_focus_folder():
    """R2: the focus-head folder entity's path, or None."""
    with _notebook_lock:
        entries = list(_notebook_entities)
    for entry in reversed(entries):
        if entry.get("kind") == "folder" and entry.get("path"):
            return entry["path"]
    return None


def notebook_snapshot():
    """R2: one consistent read for resolvers and the status speaker."""
    with _notebook_lock:
        requests = [dict(r) for r in _notebook_requests]
        entities = [dict(e) for e in _notebook_entities]
    live = {}
    try:
        live = _status_snapshot()
    except Exception:
        live = {}
    return {"requests": requests, "entities": entities, "live": live}


def _resolve_last_command():
    """R6: the last real work the user asked, or "" when none is known."""
    try:
        return str(globals().get("_last_user_work_request") or "")
    except Exception:
        return ""


def split_stop_and_redirect(msg):
    """R6: split "stop X and do Y" into (stop_half, redirect_half).

    Returns (None, None) when the turn is not a compound. The stop half must
    contain a real stop phrase; the redirect half must be non-trivial text.
    "Did you stop it?" (status) never splits — it has no redirect half.
    """
    if not msg or not msg.strip():
        return None, None
    match = _STOP_AND_REDIRECT_RE.match(msg.strip())
    if not match:
        return None, None
    stop_half = (match.group("stop") or "").strip()
    redirect = (match.group("redirect") or "").strip()
    if len(redirect) < 3:
        return None, None
    if is_status_question(msg):
        return None, None
    return stop_half, redirect


#: R7 — a correction REVISES the pending/armed request instead of adding a
#: new one. "That was meant to be a check (whether folder Malik exists)",
#: "I meant check, not create", "actually just look" — the old create idea
#: is thrown away, its yes dies with it, and the replacement runs its own
#: route (read-only checks need no approval).
_CORRECTION_RE = re.compile(
    r"\bthat\s+was\s+(?:meant\s+to\s+be|supposed\s+to\s+be)\b"
    r"|\bi\s+meant\s+(?:to\s+)?(?:check|look|see|verify|inspect|list|"
    r"create|make|write)\b"
    r"|\bactually\s+(?:just\s+)?(?:check|look|see|verify|inspect|list)\b"
    r"|\bno\s*,?\s*i\s+meant\s+(?:check|look|see)\b"
    r"|\bcorrection\s*:",
    re.IGNORECASE,
)

#: R7 — the replacement KIND carried by the correction: check-family words
#: mean a read-only inspect; create-family words mean a write.
_CORRECTION_CHECK_RE = re.compile(
    r"\bcheck\b|\bverify\b|\bconfirm\b|\bsee\b|\bshow\b|\blist\b|"
    r"\binspect\b|\blook\b|\bexist\w*\b|\bquick\s+look\b",
    re.IGNORECASE,
)
_CORRECTION_CREATE_RE = re.compile(
    r"\bcreate\b|\bmake\b|\bwrite\b|\bsave\b",
    re.IGNORECASE,
)


def is_correction(msg):
    """R7: True when the turn revises the previous request, not a new one."""
    return bool(_CORRECTION_RE.search(msg or ""))


def _cancel_armed_gates(reason):
    """R7: kill every armed approval + held clarification so the old yes dies.

    Returns True when SOMETHING was actually armed (a create was pending),
    False when there was nothing to cancel.
    """
    global _pending_opencode_task, _pending_confirmation
    cancelled = False
    try:
        from backend.services.task_agent import agent as _ta
        if _ta.has_pending_task_confirmation():
            cancelled = True
    except Exception:
        pass
    try:
        with _opencode_confirm_lock:
            if _pending_opencode_task:
                cancelled = True
    except Exception:
        pass
    try:
        with _confirmation_lock:
            if _pending_confirmation:
                cancelled = True
    except Exception:
        pass
    try:
        from backend.services import approvals as _ap
        if _ap.pending() is not None:
            cancelled = True
    except Exception:
        pass
    if not cancelled:
        return False
    try:
        from backend.services.task_agent import agent as _ta2
        _ta2.cancel_pending_task_confirmation(
            reason or "revised by correction")
    except Exception:
        pass
    try:
        with _opencode_confirm_lock:
            _pending_opencode_task = None
    except Exception:
        pass
    try:
        with _confirmation_lock:
            _pending_confirmation = None
    except Exception:
        pass
    try:
        from backend.services import approvals as _ap2
        _ap2.cancel(reason or "revised by correction")
    except Exception:
        pass
    try:
        _clear_browser_clarification()
    except Exception:
        pass
    return True


def answer_exact_vs_candidate(name, resolved):
    """R15: "is there a folder named Malik" names exact vs candidate.

    *resolved* is the ("kind", value) from
    task_agent.resolve_folder_name. Exact → plain confirmation with the
    canonical spelling; candidates → "no exact …, but … exists" with the
    real on-disk names; none → plain absence. Never a bare yes/no that
    hides the alias.
    """
    spoken = (name or "").strip() or "that"
    if not resolved:
        return "Sir, I could not verify that folder."
    kind, value = resolved
    if kind == "exact":
        return "Yes, sir. The folder %s exists." % value
    if kind == "candidates" and value:
        names = ", ".join(value[:3]) if isinstance(value, list) else value
        return ("Sir, no exact folder named %s — but this exists: %s. "
                "Shall I use it?" % (spoken, names))
    return "No, sir. No folder named %s was found." % spoken


def handle_correction(msg, from_voice=False, voice_compact=False):
    """R7: replace the old request with the corrected one — never add.

    Cancels every armed gate (the old yes dies), marks the superseded
    notebook record, then routes the CORRECTION text itself: a check-shaped
    correction runs read-only immediately; anything else falls through to
    normal routing. A correction with nothing pending is just its own
    request — also routed, never refused.
    """
    text = (msg or "").strip()
    was_pending = _cancel_armed_gates("revised by correction: %s" % text[:120])
    try:
        snap = notebook_snapshot()
        reqs = snap.get("requests") or []
        if reqs:
            notebook_mark_state(reqs[-1].get("id"), "superseded")
    except Exception:
        pass
    if _CORRECTION_CHECK_RE.search(text) and not _CORRECTION_CREATE_RE.search(text):
        try:
            from backend.services.task_agent import agent as _ta
            plan = _ta.plan_task(text, _ta.gather_context())
            steps = plan.get("steps") or [] if isinstance(plan, dict) else []
            if steps and all(
                    str(s.get("tool") or "") in (
                        "code.list_directory", "code.read_file")
                    for s in steps if isinstance(s, dict)):
                result = _ta.execute_plan(plan, _ta.gather_context())
                reply = str(result or "")
                return ("Sir, understood — checking instead. %s" % reply
                        if reply else
                        "Sir, understood — checking instead. Nothing to show.")
        except Exception as exc:
            logging.warning("[CORRECTION] Replacement check failed: %s", exc)
    if was_pending:
        return ("Sir, understood — I dropped the earlier request and will "
                "take this one instead. %s" % text)
    return None


#: Rank 1 — a reference correction revises the last BINDING, never the
#: request. "No, not that one" / "I meant the other one" / "wrong file":
#: the rejected entity becomes negative evidence and the same mention is
#: re-resolved. Deliberately excludes R7's check/create verbs ("no, I meant
#: check...") — those stay on the R7 correction path.
_REFERENCE_CORRECTION_RE = re.compile(
    r"\bnot\s+that\s+(one|file|folder|video|movie|thing)\b"
    r"|\bi\s+meant\s+the\s+other\s+one\b"
    r"|\bno\s*,?\s*not\s+that\b"
    r"|\bwrong\s+(one|file|folder)\b",
    re.IGNORECASE,
)


def handle_reference_correction(msg):
    """Rank 1: reject the last-bound entity and re-resolve the same mention.

    Returns the confirm/ask reply, or None when the turn is not a
    reference correction or no binding exists to revise (caller falls
    through to normal routing). Armed gates are cancelled — the old yes
    dies with the rejected binding.
    """
    text = (msg or "").strip()
    if not _REFERENCE_CORRECTION_RE.search(text):
        return None
    try:
        binding = entity_ledger.get_last_binding()
    except Exception:
        return None
    if not binding or not binding.get("entity_id"):
        return None
    _cancel_armed_gates("revised by reference correction: %s" % text[:120])
    try:
        snap = notebook_snapshot()
        reqs = snap.get("requests") or []
        if reqs:
            notebook_mark_state(reqs[-1].get("id"), "superseded")
    except Exception:
        pass
    entity_ledger.reject_entity(binding.get("entity_id"))
    status, payload = entity_ledger.resolve_mention(
        binding.get("text") or binding.get("mention") or "",
        expected_kind=binding.get("expected_kind"),
        constraints=binding.get("constraints"))
    if status == "bound" and isinstance(payload, dict):
        name = (payload.get("display_name") or
                os.path.basename(str(payload.get("canon") or "")) or "that")
        return "Sir, understood — I will use '%s' instead." % name
    if status == "ask" and isinstance(payload, dict):
        options = payload.get("options") or []
        question = str(payload.get("question") or "").strip()
        if options and question:
            return "Sir, understood — not that one. %s" % question
    return ("Sir, understood — not that one. "
            "Could you tell me the exact name?")
    """R6: split "stop X and do Y" into (stop_half, redirect_half).

    Returns (None, None) when the turn is not a compound. The stop half must
    contain a real stop phrase; the redirect half must be non-trivial text.
    "Did you stop it?" (status) never splits — it has no redirect half.
    """
    if not msg or not msg.strip():
        return None, None
    match = _STOP_AND_REDIRECT_RE.match(msg.strip())
    if not match:
        return None, None
    stop_half = (match.group("stop") or "").strip()
    redirect = (match.group("redirect") or "").strip()
    if len(redirect) < 3:
        return None, None
    if is_status_question(msg):
        return None, None
    return stop_half, redirect


def _browser_quiescent():
    """R6: True when no browser work is observably in flight.

    No job running, no armed opencode/browser confirmation about to start
    one, no pending clarification holding a checkpoint. Best-effort ack:
    the worker loop honours should_stop() before every tool call and
    reports a terminal stopped result; until then the redirect stays held.
    """
    try:
        if opencode_task_in_progress():
            return False
    except Exception:
        pass
    try:
        with _opencode_confirm_lock:
            if _pending_opencode_task:
                return False
    except Exception:
        pass
    try:
        from backend.services.task_agent import agent as _ta
        if _ta.has_pending_task_confirmation():
            return False
    except Exception:
        pass
    try:
        with _browser_clarification_lock:
            if _pending_browser_clarification:
                return False
    except Exception:
        pass
    return True


def _release_held_redirect():
    """R6: run the held redirect once the browser is quiescent.

    Returns the reply string, or None when nothing is held or the browser
    is still busy. Runs on the caller's thread — the caller (process_message
    entry) decides threading; this only executes the already-held text.
    """
    global _held_redirect
    with _held_redirect_lock:
        held = _held_redirect
        _held_redirect = None
    if not held:
        return None
    if not _browser_quiescent():
        with _held_redirect_lock:
            _held_redirect = held
        return None
    redirect = held.get("text") or ""
    if _LAST_COMMAND_RE.search(redirect):
        resolved = _resolve_last_command()
        if not resolved:
            return ("Sir, the browser task is stopped. "
                    "I have no earlier command to run — "
                    "what would you like me to do?")
        redirect = resolved
    print("[STOP] Releasing held redirect:", redirect)
    try:
        return process_message(
            redirect, from_voice=held.get("from_voice", False),
            sync_voice=False, voice_compact=held.get("voice_compact", False),
            commit_response=True)
    except Exception as exc:
        logging.warning("[STOP] Held redirect failed: %s", exc)
        return ("Sir, the browser task is stopped, but I could not "
                "start the next part.")


def handle_stop_then_redirect(stop_half, redirect, from_voice=False,
                              voice_compact=False):
    """R6: stop FIRST, hold the redirect behind browser quiescence.

    1) Signals the worker through the control path NOW (never queues the
    redirect behind the job being stopped). 2) When nothing was running,
    says so honestly (Trace A — no false "stopping" claim) and runs the
    redirect's exact preview path immediately. 3) When something WAS
    running, holds the redirect and says so (Trace B). The held redirect
    needs NO new approval to resolve — but starting it still needs valid
    authority (fresh preview/approval), never a resurrected stale yes.
    """
    was_running = False
    try:
        was_running = bool(opencode_task_in_progress())
    except Exception:
        pass
    if not was_running:
        try:
            with _browser_clarification_lock:
                was_running = bool(_pending_browser_clarification)
        except Exception:
            pass
    try:
        request_browser_task_stop()
    except Exception:
        pass
    try:
        invalidate_browser_runs("browser stop requested")
    except Exception:
        pass
    if _research_running:
        try:
            request_research_stop()
        except Exception:
            pass
    try:
        set_narration_enabled(False)
    except Exception:
        pass
    if not was_running:
        # Trace A: no browser job exists — no cancellation is falsely
        # recorded, no compound queued. Resolve "last command" to the
        # original user file request (never our own wording) and run the
        # redirect through the normal path so it gets its FIRST valid
        # approval opportunity, not a second approval after execution.
        text = redirect
        if _LAST_COMMAND_RE.search(redirect):
            resolved = _resolve_last_command()
            if not resolved:
                return ("Sir, no browser task is running. "
                        "I have no earlier command to run — "
                        "what would you like me to do?")
            text = resolved
        reply = process_message(
            text, from_voice=from_voice, sync_voice=False,
            voice_compact=voice_compact, commit_response=True)
        return "Sir, no browser task is running. " + str(reply or "")
    # Trace B: a job was in flight — hold the redirect, do NOT claim
    # stopping succeeded while an in-flight action may still continue.
    global _held_redirect
    with _held_redirect_lock:
        _held_redirect = {"text": redirect, "from_voice": from_voice,
                          "voice_compact": voice_compact,
                          "armed_at": time.time()}
    return ("Stop requested for the browser task, sir. "
            "The next part is held until it stops.")


def is_stop_research(text):
    t = (text or "").strip().lower()
    if _STOP_RESEARCH_NEGATION_RE.search(t):
        return False
    return any(phrase in t for phrase in STOP_RESEARCH_PHRASES)


def handle_stop_research_request(from_voice=False):
    """Explicit 'stop the research': stop BOTH narration and the task.

    Engine- and process-agnostic: cancels the browser-agent task and the
    research flow (both idempotent no-ops when nothing runs), cuts TTS
    via the caller, and disables further narration. From the voice
    process (from_voice=True) an API-side task/narration can only be
    reached over HTTP, so best-effort POSTs cover a typed-started
    websearch. Random sounds/queries never reach here — only the
    explicit phrase set does (task isolation holds everywhere else).
    """
    had_task = opencode_task_in_progress()
    try:
        request_browser_task_stop()
    except Exception:
        pass
    # F26: the stopped run must never publish its result or its question into
    # whatever request comes next.
    invalidate_browser_runs("browser stop requested")
    # Gate the research-stop event on a genuinely running research: an idle
    # stop utterance must never set the event, or the next quick search
    # would brick into a spurious 'Stopped the research' reply.
    if _research_running:
        try:
            request_research_stop()
        except Exception:
            pass
    set_narration_enabled(False)
    if from_voice:
        def _post(path):
            try:
                requests.post(
                    f"http://127.0.0.1:{BACKEND_PORT}{path}",
                    json={}, timeout=1.5,
                )
            except Exception:
                pass
        for path in ("/task/stop", "/speak/stop"):
            try:
                threading.Thread(target=_post, args=(path,), daemon=True).start()
            except Exception:
                pass
    if had_task:
        return "Stopping the research, sir."
    return "Stopped, sir."


# ── Rank 5: STOP MEANS STOP ─────────────────────────────────────────────────
# "stop the browser task" / "stop the research" keep their dedicated phrases
# above. What reaches here is the GENERAL stop: a bare "stop" / "stop it" /
# "ruko" and the broad "stop everything". A plain stop cuts speech and stops
# the ONE job the user is watching (the newest live work job - never the chat
# request, never older background work); "stop everything" stops every work
# job. "Stopped" is only spoken once the cancelled work is observably still;
# otherwise the user hears "Stopping" now and the honest report when it lands.
_STOP_EVERYTHING_RE = re.compile(
    r"\bstop\s+(?:everything|it\s+all|all(?:\s+(?:tasks?|jobs?|work))?|"
    r"everything\s+now)\b", re.IGNORECASE)
_BARE_STOP_RE = re.compile(
    r"^(?:jarvis[,\s]*)?(?:please\s+|just\s+|now\s+|abhi\s+)*"
    r"(?:stop|ruko)(?:[\s,]+(?:it|now|please|sir|jarvis|karo|kar|abhi))*"
    r"[.!]*$", re.IGNORECASE)
_STOP_NEGATION_RE = re.compile(
    r"\b(?:dont|don't|do not|never|mat)\b", re.IGNORECASE)

#: Rank 5 — how long the stop finisher waits for cancelled work to actually
#: go quiet before it reports honestly that something is still moving.
_STOP_QUIESCE_TIMEOUT_S = 8.0


def is_stop_everything(text):
    """True for the broad stop: "stop everything", "stop all tasks"."""
    t = (text or "").strip()
    return bool(_STOP_EVERYTHING_RE.search(t)) and \
        not _STOP_NEGATION_RE.search(t)


def is_bare_stop(text):
    """True when the WHOLE utterance is just a stop control ("stop it")."""
    t = (text or "").strip()
    if not t or _STOP_NEGATION_RE.search(t):
        return False
    return bool(_BARE_STOP_RE.match(t))


def _stop_targets_still(cancelled_ids, had_browser, research_was):
    """Rank 5: are the things we stopped observably still?

    Background jobs we did NOT touch never block the "Stopped" report — a
    plain stop only promises silence for what it stopped.
    """
    try:
        if research_was and _research_running:
            return False
    except Exception:
        pass
    if had_browser:
        try:
            if opencode_task_in_progress() or not _browser_quiescent():
                return False
        except Exception:
            pass
    try:
        from backend.services import jobs as _jobs

        for job_id in cancelled_ids or ():
            job = _jobs.get_job(job_id)
            if job is None:
                continue
            if job.killing and not job.wait_for_kill(0.5):
                return False
            if not job.cancelled:
                return False
    except Exception:
        pass
    return True


def _stop_report_text():
    """Rank 5: "Stopped, sir." plus what the stopped run had already done."""
    report = ""
    try:
        report = consume_browser_stop_report()
    except Exception:
        report = ""
    if report:
        return ("Stopped, sir. I had already %s. Nothing is moving now."
                % report)
    return "Stopped, sir."


def _finish_stop_reply(cancelled_ids, had_browser, research_was):
    """Wait (bounded) for the stopped work to go quiet, then report once."""
    deadline = time.time() + _STOP_QUIESCE_TIMEOUT_S
    while time.time() < deadline:
        if _stop_targets_still(cancelled_ids, had_browser, research_was):
            break
        time.sleep(0.2)
    if _stop_targets_still(cancelled_ids, had_browser, research_was):
        _notify_async_reply(_stop_report_text())
    else:
        _notify_async_reply(
            "Sir, I asked it to stop, but something is still moving. Give "
            "it a moment — if it stays stuck, tell me and I will cut it "
            "hard.")


def handle_stop_message(msg, from_voice=False):
    """Rank 5 — the unified stop.

    Order: cut speech NOW, cancel the right job(s) NOW, then say "Stopped"
    only when the cancelled work is truly still (synchronously when it
    already is, otherwise "Stopping" now and the report lands async).
    """
    want_all = is_stop_everything(msg)
    cancelled = []
    try:
        from backend.services import jobs as _jobs

        if want_all:
            for job in list(_jobs.live_jobs(exclude_kinds=("request",))):
                cancelled.extend(
                    _jobs.cancel_job(job.job_id,
                                     reason="stop everything requested"))
        else:
            cancelled = _jobs.request_stop(
                None, reason="stop requested",
                exclude_kinds=("request",))
    except Exception as exc:
        logging.warning("[STOP] job cancellation failed: %s", exc)
    had_browser = False
    try:
        had_browser = bool(opencode_task_in_progress())
    except Exception:
        pass
    research_was = False
    try:
        research_was = bool(_research_running)
    except Exception:
        pass
    if had_browser or want_all:
        try:
            request_browser_task_stop()
            invalidate_browser_runs("stop requested")
        except Exception:
            pass
    if research_was:
        try:
            request_research_stop()
        except Exception:
            pass
    try:
        set_narration_enabled(False)
    except Exception:
        pass
    try:
        from backend.services.voice import stop_speaking

        stop_speaking()
    except Exception:
        pass
    if from_voice:
        def _post(path):
            try:
                requests.post(
                    f"http://127.0.0.1:{BACKEND_PORT}{path}",
                    json={}, timeout=1.5,
                )
            except Exception:
                pass
        for path in ("/task/stop", "/speak/stop"):
            try:
                threading.Thread(target=_post, args=(path,), daemon=True).start()
            except Exception:
                pass
    if _stop_targets_still(cancelled, had_browser, research_was):
        # Everything is already still: "Stopped" is true right now.
        return _stop_report_text()
    try:
        threading.Thread(
            target=_finish_stop_reply,
            args=(list(cancelled), had_browser, research_was),
            daemon=True).start()
    except Exception:
        pass
    return "Stopping, sir."


def is_deepsearch_request(text):
    """True when the user explicitly asked for the deep multi-site mode.

    Tiered-search keyword detection lives HERE (websearch entry/brain
    parsing) — intent.py stays untouched by standing rule.
    """
    return "deepsearch" in (text or "").lower()


def handle_research_intent(research_query, from_voice=False, voice_compact=False, derived=False, deep=False):
    """Ack immediately, then run the websearch in the background.

    Tiered search:
      * default (deep=False) — Google AI Overview first: quick_search
        reads the answer box, no website scraping, short spoken+text
        summary, light snippet fallback when Google shows no overview.
      * deepsearch (deep=True) — the AI Overview PLUS the existing
        multi-site research flow (run_research with the overview pinned
        in as a first note).

    The full detailed report is pushed to the glass overlay (deep mode
    only); only a short spoken summary is announced when it's done (never
    the huge report).

    *derived* marks queries that already came from the LLM intent router
    (its "query" field) — raw user sentences are re-derived to the real
    subject so we never type the user's sentence verbatim into the browser.
    """
    global _research_running
    # R14: the ack only NAMES the request — "On it / running now" claims
    # execution before the worker thread below proves anything. Completion
    # is announced by the result path when the research actually lands.
    if deep:
        ack = ("Deepsearch for that, sir — the full report comes up on "
               "your screen, and I will sum it up when it is ready.")
    else:
        ack = "Quick lookup for that, sir."

    if voice_compact and len(ack) > 110:
        ack = ("Deepsearch running, sir — full report on your screen, "
               "summary when it's ready." if deep
               else "Quick lookup running, sir.")

    def _run():
        global _research_running
        try:
            try:
                query = research_query if derived else derive_research_query(research_query)
                if not query:
                    # Live fix: never type a bare pointer ("this stream")
                    # into the web — ask once, with nothing to guess at.
                    _notify_async_reply(
                        "Sir, I could not tell what to search for — name it "
                        "once and I will look it up immediately.")
                    return
                _set_last_research_topic(query)
                if deep:
                    # F28: progress and evidence are wired immediately — the
                    # overlay shows sources as they land instead of going dark
                    # for the whole run.
                    evidence_seen = []

                    def _progress(message):
                        print(f"[RESEARCH] {message}")
                        push_research_progress(query, message, evidence_seen)

                    def _evidence(item):
                        evidence_seen.append(item)
                        push_research_progress(
                            query, f"Collected {len(evidence_seen)} source(s) so far.",
                            evidence_seen)

                    _progress("Fetching the search engine overview…")
                    overview = fetch_ai_overview_text(query)
                    result = run_research(
                        query, pinned_overview=overview,
                        on_progress=_progress, on_evidence=_evidence,
                    )
                    _progress(f"Finished — {len(evidence_seen)} source(s).")
                else:
                    result = run_quick_search(query)
            except Exception as exc:
                logging.warning("[RESEARCH] failed: %s", exc)
                _notify_async_reply(
                    "I hit a snag while researching that, sir. The browser may need a moment — please try again."
                )
                return
            if result.get("stopped"):
                if deep:
                    push_research_progress(query, "Stopped the research.", [])
                _notify_async_reply(result["spoken_summary"])
                return
            if deep:
                push_research_result({
                    "query": result["query"],
                    "markdown": result["detailed_markdown"],
                    "videos": result["related_videos"],
                    "report_path": result["report_path"],
                    "visited_count": result["visited_count"],
                    "failed_count": result["failed_count"],
                })
                print(f"[RESEARCH] done: {result['report_path']}")
            else:
                print(f"[QUICKSEARCH] done: {result['query']}")
            # F07: the report's REAL path and its sources are recorded as
            # STRUCTURE against this identified request, so a follow-up after a
            # restart can retrieve where the report is and what it cited.
            if memory_store is not None:
                try:
                    artifacts = []
                    if result.get("report_path"):
                        artifacts.append({"kind": "report",
                                          "path": result["report_path"],
                                          "title": result.get("query") or query})
                    for item in list(result.get("evidence") or
                                     result.get("sources") or [])[:20]:
                        if isinstance(item, dict):
                            artifacts.append({
                                "kind": "source",
                                "url": item.get("url") or item.get("link"),
                                "title": item.get("title"),
                                "source": item.get("site") or item.get("source"),
                            })
                    rid = memory_store.begin_request(
                        query, route="research" if deep else "quick_search")
                    memory_store.record_result(
                        rid, "completed",
                        summary=result.get("spoken_summary")
                        or result.get("overview_text") or query,
                        artifacts=artifacts, engine="research")
                except Exception as exc:
                    logging.debug("[RESEARCH] work-event record failed: %s", exc)
            ui_text = result.get("overview_text") or result["spoken_summary"]
            _notify_async_reply(text=ui_text, spoken=result["spoken_summary"])
        finally:
            _research_running = False
            try:
                clear_stop_request()
            except Exception:
                pass

    _research_running = True
    try:
        threading.Thread(target=_run, daemon=True).start()
    except Exception:
        _research_running = False
        try:
            clear_stop_request()
        except Exception:
            pass
        raise
    return ack


# Guards proactive (unasked) research so a single reply only triggers it once.
_proactive_research_lock = threading.Lock()
_proactive_research_fired = False

_pending_confirmation = None
_confirmation_lock = threading.Lock()
CONFIRM_WINDOW_SECONDS = 45.0
_confirmation_cooldown_until = 0.0

# Pending opencode-handoff confirmation gate (mirrors the research gate).
_pending_opencode_task = None
_opencode_confirm_lock = threading.Lock()

# Browser follow-up continuity: when browser agent asks a clarifying question
_pending_browser_clarification = None
_browser_clarification_lock = threading.Lock()
_BROWSER_CLARIFICATION_TTL = 90.0

# F26/F03 — request-owned delivery for deferred browser runs.
#
# A deferred run executes in its own thread; the old code let ANY finished run
# notify, arm a clarification and speak, so a superseded run could post its
# result — or its question — into a LATER request. Each run now carries an
# identity, and only the newest run may publish anything.
_browser_run_lock = threading.Lock()
_browser_run_generation = 0


def _new_browser_run():
    """Start a new browser run generation; returns its identity."""
    global _browser_run_generation
    with _browser_run_lock:
        _browser_run_generation += 1
        return _browser_run_generation


def invalidate_browser_runs(reason=""):
    """F26: make every in-flight browser run stale (stop/interrupt)."""
    global _browser_run_generation
    with _browser_run_lock:
        _browser_run_generation += 1
        generation = _browser_run_generation
    if reason:
        logging.info("[TASK] browser runs invalidated: %s", reason)
    return generation


def _browser_run_is_current(run_id):
    with _browser_run_lock:
        return run_id == _browser_run_generation


def _publish_browser_run(run_id, publish):
    """Run *publish* only if *run_id* is still the newest run (F26)."""
    if not _browser_run_is_current(run_id):
        logging.info("[TASK] superseded browser run %s produced no delivery",
                     run_id)
        return False
    publish()
    return True


def _can_ask_confirmation():
    """False for a while after a confirmation was asked / answered unclear,
    so a skipped answer can never instantly re-arm the same question."""
    return time.time() >= _confirmation_cooldown_until


def _set_confirmation_cooldown():
    global _confirmation_cooldown_until
    _confirmation_cooldown_until = time.time() + CONFIRM_WINDOW_SECONDS

_CONFIRM_YES_RE = re.compile(
    r"\b(yes|yeah|yep|yup|sure|okay|ok|alright|go ahead|do it|please do|"
    r"haan|ha|hmm|of course|absolutely)\b|"
    r"\b(kar sakte|kar sakta|kar sakti|kar do|karo)\b|"
    r"\b(dhundho|dhundh|khoj ke|khoj|search kar|search karo|"
    r"search karke|karke batao|kar ke batao|karke batana)\b|"
    r"\b(internet pe|internet per)\b|"
    r"looking it up|look it up|look that up|search it|find it|find out",
    flags=re.IGNORECASE,
)
_CONFIRM_NO_RE = re.compile(
    r"\b(no|nah|nope|nahi|nahin|naheen|nai|never|skip|ignore|not needed|no thanks|"
    r"don.?.?t (bother|need|worry)|leave it|as you like|whatever|"
    r"koi baat nahi|zarurat nahi|bina matlab)\b",
    flags=re.IGNORECASE,
)

_UNSURE_RE = re.compile(
    r"\b(i don'?t (know|have)|i'?m not (sure|certain)|i can'?t (say|answer|tell|find)|"
    r"i have no (idea|information)|i couldn'?t find|not in my knowledge base|"
    r"i'm sorry, i (don'?t|can'?t)|i do not have information)\b",
    flags=re.IGNORECASE,
)


def _confirmation_question():
    return ("I'm not entirely sure about that, sir. Should I look it up on the "
            "web — or would you like me to just answer from what I know?")


def _confirmation_verdict(answer):
    if not answer or not answer.strip():
        return None
    if _CONFIRM_NO_RE.search(answer):
        return "no"
    if _CONFIRM_YES_RE.search(answer):
        return "yes"
    return None


def _llm_resolve_confirmation(original_message, answer):
    """Let the brain itself judge the confirmation.

    The brain has the whole conversation in front of it, so a Hinglish reply
    like "ha, internet pe search kar ke batao iske bare mein" is read as a
    yes — and the exact search query is taken from the *original* message
    ("you already told him what to search"), never from the answer phrase.

    Returns (verdict, query):
      verdict -> "yes" | "no" | "unclear"
      query   -> the web query to run when verdict is "yes", else None.
      On any LLM failure returns (None, None) so the regex path can fall back.
    """
    try:
        history = get_history()[-10:]
        context_lines = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in history
        )
        prompt = (
            "You are Jarvis's permission gate. A user was asked whether to "
            "search the web. Just answer with the decision, using the whole "
            "conversation for context.\n\n"
            f"===== RECENT CONVERSATION =====\n{context_lines}\n"
            f"===== ORIGINAL QUESTION =====\n{original_message}\n"
            f"===== USER'S REPLY =====\n{answer}\n\n"
            "Does the reply consent to searching the web, decline it, or is "
            "it unclear/off-topic? Reply STRICT JSON only:\n"
            '{"verdict": "yes" or "no" or "unclear", '
            '"query": "the exact web search query — the subject the user '
            'meant to research, from the ORIGINAL QUESTION unless the reply '
            'clearly names a better subject, or null when verdict is not yes"}'
        )
        messages = [
            {
                "role": "system",
                "content": "Return strict JSON only. No markdown, no extra text.",
            },
            {"role": "user", "content": prompt},
        ]
        result = _ask_chat_nonstream(messages, temperature=0.0, max_tokens=200)
        if not result or not result.get("choices"):
            return None, None
        raw = result["choices"][0].get("message", {}).get("content", "")
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return None, None
        parsed = json.loads(match.group(0))
        verdict = str(parsed.get("verdict", "")).strip().lower()
        if verdict not in ("yes", "no", "unclear"):
            return None, None
        query = parsed.get("query")
        if not isinstance(query, str):
            query = None
        query = query.strip() if query else None
        print(
            f"[RESEARCH] Brain confirmation: verdict={verdict!r} "
            f"query={query!r} <- '{answer}'"
        )
        return verdict, query
    except Exception as exc:
        logging.warning("[RESEARCH] LLM confirmation failed: %s", exc)
        return None, None


def _arm_confirmation(word, query=None, question=None):
    with _confirmation_lock:
        global _pending_confirmation
        _pending_confirmation = {
            "message": word,
            "expires": time.time() + CONFIRM_WINDOW_SECONDS,
            "query": str(query or "").strip() or None,
            "question": str(question or "").strip() or None,
        }


# ── Live fix: screen-research confirmation ("is this what you mean?") ───────
# When a screen step cannot pin down exactly what the user pointed at, Jarvis
# asks ONE candidate question instead of researching a generic guess. The
# ask can be confirmed ("yes"), declined ("no"), or corrected ("no not that,
# the image to the left") — the correction is re-read against the SAME screen
# observation, so no second screen analysis is needed.
_pending_screen_clarify = None
_SCREEN_CLARIFY_TTL = 240.0


def _set_pending_screen_clarify(clause, topic, content, obs_id=""):
    global _pending_screen_clarify
    try:
        with _confirmation_lock:
            _pending_screen_clarify = {
                "clause": str(clause or ""),
                "topic": str(topic or ""),
                "content": str(content or ""),
                # RANK 1: the observation a correction re-asks on (RAM only;
                # an expired observation simply falls back to a fresh look).
                "obs_id": str(obs_id or ""),
                "at": time.time(),
            }
    except Exception:
        pass


def _clear_pending_screen_clarify():
    global _pending_screen_clarify
    try:
        with _confirmation_lock:
            _pending_screen_clarify = None
    except Exception:
        pass


def _get_pending_screen_clarify():
    try:
        with _confirmation_lock:
            state = dict(_pending_screen_clarify) if _pending_screen_clarify \
                else None
    except Exception:
        return None
    if not state:
        return None
    if time.time() - float(state.get("at") or 0.0) > _SCREEN_CLARIFY_TTL:
        return None
    return state


def _mi_extract_screen_query(clause, topic, content, correction=""):
    """Read the SPECIFIC thing the user pointed at from the screen report.

    Returns (query, confident). Both empty/False when the chat model is
    unavailable or cannot name a concrete subject — callers then ask once
    instead of researching a generic guess.
    """
    try:
        screen_text = _mi_clip("%s %s" % (topic, content), 1400)
        prompt = (
            "A user asked Jarvis to research something shown on their "
            "screen.\n"
            f"===== WHAT THE SCREEN SHOWS =====\n{screen_text}\n"
            f"===== WHAT THE USER SAID =====\n{clause}\n"
        )
        if correction:
            prompt += f"===== USER'S CORRECTION =====\n{correction}\n"
        prompt += (
            "\nReturn STRICT JSON only:\n"
            '{"query": "the precise web search query for the SPECIFIC thing '
            'the user pointed at — use the actual names, words or text '
            'visible on the screen, never a generic description", '
            '"confident": true or false}\n'
            "Use \"query\": null when the screen does not show what the "
            "user means. Set \"confident\" to false when you are guessing."
        )
        messages = [
            {"role": "system",
             "content": "Return strict JSON only. No markdown, no extra "
                        "text."},
            {"role": "user", "content": prompt},
        ]
        result = _ask_chat_nonstream(messages, temperature=0.0,
                                     max_tokens=160)
        if not result or not result.get("choices"):
            return "", False
        raw = result["choices"][0].get("message", {}).get("content", "")
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return "", False
        parsed = json.loads(match.group(0))
        query = parsed.get("query")
        confident = bool(parsed.get("confident"))
        if not isinstance(query, str):
            return "", False
        query = query.strip().strip("\"'")[:220]
        if (not query or _META_QUERY_RE.search(query)
                or _is_deictic_query(query)):
            return "", False
        if query.lower() == str(clause or "").strip().lower():
            return "", False
        print(f"[CHAIN] Screen subject extracted ({'high' if confident else 'low'} "
              f"confidence): {query!r}")
        return query, confident
    except Exception as exc:
        logging.warning("[CHAIN] Screen subject extraction failed: %s", exc)
        return "", False


def _clean_screen_subject(text):
    """A vision answer to a search query — the exact name, nothing else."""
    t = str(text or "").strip()
    if not t:
        return ""
    quoted = re.search(r"[\"\u201c\u2018']([^\"\u201d\u2019']{3,140})"
                       r"[\"\u201d\u2019']", t)
    if quoted:
        t = quoted.group(1).strip()
    else:
        # A multi-word sentence ending in a period is a description, not a
        # name/title — the focused prompt asks for names only, so this is
        # the model not complying; the caller falls back and asks.
        head = t.rstrip()
        if head.endswith(".") and len(head.split()) > 8:
            return ""
    t = t.strip(" \t\"'\u201c\u201d\u2018\u2019`.,;:-\u2014")[:160].strip()
    if len(t) < 3 or len(t.split()) > 14:
        return ""
    if re.search(r"(?i)\b(?:couldn'?t|cannot|can'?t|not\s+clear|unclear|"
                 r"no\s+idea|nothing|doesn'?t\s+show)\b", t):
        return ""
    if _META_QUERY_RE.search(t) or _is_deictic_query(t):
        return ""
    return t


def _grade_confident(grade):
    """Exact, single-fit, non-ambiguous vision bindings search silently."""
    return bool(grade and grade.get("match") == "exact"
                and not grade.get("ambiguous")
                and (grade.get("n_fit") or 0) <= 1)


def _mi_vision_screen_target(clause, correction="", obs_id=""):
    """(label, grade): the vision model resolves the pointed-at screen item.

    RANK 1: when the screen step stored an observation, its SAME image is
    re-used — no re-capture, so a Shorts feed that advanced between the
    two calls cannot swap the subject — and the vision model resolves the
    user's pointer against the item inventory. `grade` carries the match
    quality so the caller can ask instead of guessing. `grade is None`
    means the legacy focused-vision path answered (no stored
    observation); bool(label) keeps the old confidence semantics there.
    Never raises.
    """
    if not _screen_qa_busy.acquire(timeout=15):
        return "", None
    try:
        ask = _mi_clip(str(clause or "").strip(), 200)
        if not ask:
            return "", None
        target = ask
        if correction:
            target = ('%s (the user corrected themselves: "%s")'
                      % (ask, _mi_clip(str(correction), 120)))
        obs = get_observation(obs_id) if obs_id else None
        if obs is not None and obs.image_data_url:
            capture = {"image_data_url": obs.image_data_url,
                       "region": obs.region,
                       "mode": obs.mode or "full",
                       "img_hash": obs.img_hash}
            fresh, grade = identify_on_screen(
                ask, target_text=target, image=capture,
                mode=obs.mode or "full", rejected=obs.rejected)
            label = ""
            if fresh is not None and grade.get("ids"):
                by_id = {item.id: item for item in fresh.items}
                parts = [by_id[i].label for i in grade["ids"] if i in by_id]
                label = _mi_clip(" ".join(p for p in parts if p), 160)
            if (label and label.lower() != ask.lower()
                    and not _META_QUERY_RE.search(label)
                    and not _is_deictic_query(label)):
                print("[CHAIN] Vision screen subject: %r (match=%s, n_fit=%s)"
                      % (label, grade.get("match"), grade.get("n_fit")))
                return label, grade
            return "", grade
        # Legacy path (no stored observation): focused vision question.
        question = (
            "Search-query task: the user wants me to research ONE specific "
            "item visible on this screen. They are pointing at it with: "
            '"%s". Identify that exact item and put ONLY its exact displayed '
            'name or title into the "tip" field — no sentences, no quotes. '
            "If the screen does not clearly show the item they mean, return "
            'an empty "tip".' % target
        )
        result = analyze_screen(question) or {}
        for candidate in (result.get("tip"), result.get("topic")):
            cleaned = _clean_screen_subject(candidate)
            if cleaned and cleaned.lower() != target.lower():
                print(f"[CHAIN] Vision screen subject: {cleaned!r}")
                return cleaned, None
        return "", None
    except Exception as exc:
        logging.warning("[CHAIN] Vision screen subject failed: %s", exc)
        return "", None
    finally:
        try:
            _screen_qa_busy.release()
        except Exception:
            pass


def _mi_vision_screen_query(clause, correction="", obs_id=""):
    """String-only view of _mi_vision_screen_target (compat callers/tests)."""
    return _mi_vision_screen_target(clause, correction=correction,
                                    obs_id=obs_id)[0]


#: "no not that, the image to the left" — a redirect, not a refusal.
_SCREEN_REDIRECT_RE = re.compile(
    r"\b(?:not\s+(?:that|this|those|these|it|the\s+\w+)|instead|"
    r"other\s+one|different|to\s+the\s+(?:left|right)|that\s+one|"
    r"this\s+one|below|above|next\s+to|the\s+(?:image|video|item|one|"
    r"thing)\s+(?:to\s+the\s+)?(?:left|right|above|below))\b",
    re.IGNORECASE,
)


def _research_from_screen_correction(state, correction):
    """Re-read the SAME screen observation with the user's corrected pointer
    ("no not that, the image to the left") and research the result. The
    focused vision call is primary — the correction points at pixels."""
    clause = str(state.get("clause") or "")
    query = _mi_vision_screen_query(clause, correction=correction,
                                    obs_id=str(state.get("obs_id") or ""))
    if not query:
        query, _ = _mi_extract_screen_query(
            clause, str(state.get("topic") or ""),
            str(state.get("content") or ""), correction=correction)
    if not query:
        return None
    print("[RESEARCH] Corrected screen subject:", query)
    return handle_research_intent(query, derived=True)


def consume_screen_research_clarify(msg):
    """The user answers the open screen-research ask ("which part?").

    Only pointer-like replies are consumed; a fresh request keeps its
    normal route (the pending state simply expires).
    """
    state = _get_pending_screen_clarify()
    if not state:
        return None
    text = str(msg or "").strip()
    if not text:
        return None
    if _confirmation_verdict(text) is not None:
        return None  # yes/no is owned by the confirmation gate
    if _SEARCH_SHAPED_VERB_RE.search(text) or _CHAT_ACTION_RE.search(text):
        return None  # a new request, not a pointer correction
    _clear_pending_screen_clarify()
    print("[CHAIN] Screen clarification answered:", text)
    return _research_from_screen_correction(state, text)


# ── Rank 2: multi-intent chains (screen → research → file) ─────────────
# One utterance can be several jobs. multi_intent.build_chain parses the
# compound into ordered steps; this executor runs them on a worker thread,
# injects each step's output into the next (screen observation → research
# query → file content), and lands the single approval pause at the end:
# the file write is only ARMED here — nothing is written until the user's
# confirmation, which flows through the normal task gate.
_MI_CHAIN_LOCK = threading.Lock()
_mi_chain_active = False

#: Live fix: remember the last finished chain so the SAME request arriving
#: seconds later answers from that run instead of silently re-running it.
_chain_memory_lock = threading.Lock()
_last_chain_run = None

#: R20 — the last chain step findings (the screen step's vision answer, the
#: research step's summary). The folder-name answer arrives on a LATER turn
#: when the chain worker is long gone, so its file write needs the findings
#: the chain already produced — stored here, TTL'd, never persisted.
_CHAIN_FINDINGS_TTL = 300.0
_chain_last_findings = {"text": "", "at": 0.0}


def _set_chain_findings(text):
    """Remember the latest chain step output for a later follow-up."""
    try:
        with _chain_memory_lock:
            _chain_last_findings["text"] = str(text or "")[:4000]
            _chain_last_findings["at"] = time.time()
    except Exception:
        pass


def _get_chain_findings():
    """The recent chain findings, or "" when the window has lapsed."""
    try:
        with _chain_memory_lock:
            state = dict(_chain_last_findings)
    except Exception:
        return ""
    if time.time() - float(state.get("at") or 0.0) > _CHAIN_FINDINGS_TTL:
        return ""
    return str(state.get("text") or "")


def _remember_user_turn(text):
    """R20: a consumed action turn is a real user turn in the history.

    The chat path always committed its halves, but chain/task/tool turns
    never did — history held danging assistant notes with no request above
    them, so a follow-up ("did you put it in that folder?") was answered
    from a conversation the model literally could not see.
    """
    try:
        if str(text or "").strip():
            add_message("user", str(text))
    except Exception:
        pass


def _remember_chain_step(result):
    """R20: a finished chain step leaves its specifics in history.

    The final [background result] report is a summary; the per-step notes
    carry the identified label, the searched query and the armed file so a
    later turn can resolve its referent against what actually happened.
    """
    try:
        kind = str(result.get("kind") or "")
        status = str(result.get("status") or "")
        frag = str(result.get("fragment") or "").strip()
        if not kind or not frag:
            return
        if kind == "screen" and status == "ok":
            label = str(result.get("output_query") or "").strip()
            note = ("I looked at your screen: %s" % (label or frag)) \
                if label else frag
        elif kind == "research" and status == "ok":
            q = str(result.get("output_query") or "").strip()
            note = ('I searched "%s" — %s' % (q, frag)) if q else frag
        else:
            note = frag
        _remember_tool_summary(kind, note)
    except Exception:
        pass


def _mi_clip(text, limit=600):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) > limit:
        cut = text.rfind(" ", 0, limit - 3)
        text = (text[:cut] if cut > 0 else text[:limit - 3]).rstrip() + "..."
    return text


def _is_generic_screen_topic(topic):
    """True when a screen report's topic is a type description ("YouTube
    live chat message"), not a concrete name — researching it as-is would
    search 'whatever was seen' instead of the thing the user pointed at."""
    text = str(topic or "").strip()
    if not text or _is_deictic_query(text):
        return True
    tokens = [t for t in re.split(r"[\W_]+", text.lower()) if t]
    if len(tokens) > 6:
        return False
    return not [t for t in tokens
                if t not in _DEREF_FILLERS and t not in _GENERIC_MEDIA_TOKENS]


#: Grammar words that survive instruction stripping but name nothing.
_MI_CLAUSE_STOPWORDS = {
    "does", "do", "did", "actually", "say", "says", "said", "mean",
    "means", "tell", "what", "and", "or", "they", "are", "talking",
    "explain", "understand", "summarise", "summarize", "please",
    # Effort adjectives ("do a deep research about it") are not subjects.
    "deep", "deeper", "brief", "quick", "short", "detailed", "full",
    "proper", "little", "bit",
    # Filler adverbs/quantifiers ("very", "some movie") name nothing.
    "very", "really", "basically", "literally", "probably", "maybe",
    "simply", "just", "much", "even", "still", "also", "then", "now",
    "here", "there", "again", "ever", "quite", "some", "any", "few",
    "many", "lot", "lots",
}


def _mi_clause_specific_tokens(clause):
    """Content words the user named in a research clause ("secret",
    "image generation models") after instruction words are stripped."""
    text = str(clause or "").lower()
    text = re.sub(
        r"\bresearch\w*\b|\bsearch\w*\b|\blook\s+up\b|\bfind\s+out\b"
        r"|\bgoogle\w*\b|\bdeepsearch\b",
        " ", text)
    text = re.sub(r"\bon\s+(?:my|the)\s+screen\b", " ", text)
    text = re.sub(r"\b(?:on|from|over)\s+(?:the\s+)?internet\b", " ", text)
    text = re.sub(r"\bin\s+detail\b", " ", text)
    tokens = [t for t in re.split(r"[\W_]+", text) if t]
    return [t for t in tokens
            if t not in _DEREF_FILLERS and t not in _GENERIC_MEDIA_TOKENS
            and t not in _MI_CLAUSE_STOPWORDS]


def _mi_needs_screen_extraction(clause, topic):
    """True when the generic screen topic is not enough to search: either
    the topic names nothing by itself, or the user named something the
    topic does not mention."""
    if _is_generic_screen_topic(topic):
        return True
    specific = _mi_clause_specific_tokens(clause)
    if not specific:
        return False
    topic_low = str(topic or "").lower()
    return bool([t for t in specific if t[:4] not in topic_low])


#: Possessive back-references ("research about THEIR release dates") — the
#: subject is the item the screen step just identified; what follows is an
#: ASPECT of it. The second, stateless vision call has no antecedent for the
#: possessive (it lived in the previous call), so it cannot bind and asking
#: "which part?" would be wrong — the identification already happened.
_POSSESSIVE_BACKREF_RE = re.compile(
    r"\b(?:their|its|his|her)\b"
    r"|\bof\s+(?:these|those|them|it)\b"
    r"|\b(?:uska|uski|uske|unka|unki|inka|inki|iska|iske)\b",
    re.IGNORECASE,
)

#: Instruction tails that are not aspect words ("... and tell me what you find").
_ASPECT_TAIL_RE = re.compile(
    r"\b(?:and\s+)?(?:tell|show|give|send|let)\s+me\b.*$"
    r"|\bwhat\s+you\s+(?:find|found|think|get|see)\b.*$",
    re.IGNORECASE,
)

#: Grammar words that can never be an aspect of the subject.
_ASPECT_STOP_TOKENS = _MI_CLAUSE_STOPWORDS | {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "you", "me", "i", "we", "they", "them", "their", "its", "his", "her",
    "it", "this", "that", "these", "those", "about", "of", "for", "with",
    "to", "on", "in", "from", "by", "and", "or", "please", "sir",
    "video", "videos", "clip", "content", "thing", "name", "title",
    "screen", "monitor", "display", "stream", "channel", "youtube", "yt",
    "post", "posts",
    # Pointer nouns and medium words ask for nothing by themselves.
    "internet", "web", "online",
    "creator", "streamer", "uploader", "youtuber", "maker", "author",
    "guy", "person", "dude",
    # Look/touch verbs and determiners ("look at my screen") are not aspects.
    "look", "at", "see", "check", "watch", "view", "examine", "scan",
    "read", "analyse", "analyze", "my", "your", "our",
}


def _mi_aspect_tokens(clause, subject):
    """Content words the user added ABOUT the subject ("release dates")."""
    text = _ASPECT_TAIL_RE.sub(" ", str(clause or "").lower())
    text = re.sub(
        r"\bresearch\w*\b|\bsearch\w*\b|\blook\s+up\b|\bfind\s+out\b"
        r"|\bgoogle\w*\b|\bdeepsearch\b",
        " ", text)
    text = re.sub(r"\bon\s+(?:my|the)\s+screen\b", " ", text)
    text = re.sub(r"\b(?:on|from|over)\s+(?:the\s+)?internet\b", " ", text)
    subject_low = str(subject or "").lower()
    tokens = []
    for token in re.split(r"[\W_]+", text):
        if not token or token in _ASPECT_STOP_TOKENS:
            continue
        if len(token) >= 4 and token[:4] in subject_low:
            continue
        tokens.append(token)
    return tokens


def _mi_subject_aspect_query(clause, obs_id):
    """Compose "subject + aspect" for a possessive back-reference.

    "research about their release dates" after a screen step names the
    subject ("their") in the observation the step just stored; only the
    ASPECT words ("release dates") are new. Returns "" when anything is
    unclear (no possessive, no stored primary label, no aspect words) so
    the caller keeps the existing vision/ask flow.
    """
    if not obs_id or not _POSSESSIVE_BACKREF_RE.search(str(clause or "")):
        return ""
    obs = get_observation(obs_id)
    if obs is None or not obs.items:
        return ""
    item = next((candidate for candidate in obs.items if candidate.primary),
                obs.items[0])
    if not item.label or not item.label_is_text or item.truncated:
        return ""
    if _is_generic_screen_topic(item.label):
        return ""
    aspect = _mi_aspect_tokens(clause, item.label)
    if not aspect:
        return ""
    subject = _clean_screen_subject(item.label) or item.label
    return " ".join([subject] + aspect)[:200]


def _mi_aspect_suffix(clause, subject, query, obs_id):
    """Aspect words to append to an already-bound screen subject.

    The aspect lives in the research clause ("their release dates") or, for
    a purely deictic request ("research about it and tell me"), in the
    observation's own utterance ("what is this video about dimensions").
    Returns [] when nothing new asks a question about the subject — the
    query then stays exactly the identified label, no invented words.
    Tokens already present in *query* are dropped, so a composed query is
    never doubled.
    """
    tokens = _mi_aspect_tokens(clause, subject)
    if not tokens and obs_id:
        obs = get_observation(obs_id)
        if obs is not None:
            tokens = _mi_aspect_tokens(
                str(getattr(obs, "utterance", "") or ""), subject)
    query_low = str(query or "").lower()
    return [token for token in tokens if token[:4] not in query_low]


def _mi_screen_step(step):
    """Run one screen-analysis step; returns its result record."""
    if not _screen_qa_busy.acquire(timeout=30):
        return {"kind": "screen", "status": "failed",
                "fragment": "I couldn't look at the screen — another "
                            "screen analysis is running."}
    try:
        result = analyze_screen(step.get("text") or "") or {}
    except Exception as exc:
        return {"kind": "screen", "status": "failed",
                "fragment": "I couldn't analyse the screen (%s)."
                            % _mi_clip(exc, 80)}
    finally:
        _screen_qa_busy.release()
    tip = str(result.get("tip") or "").strip()
    topic = str(result.get("topic") or "").strip()
    creator = str(result.get("creator") or "").strip()
    if not tip or "couldn't analyse the screen" in tip.lower():
        return {"kind": "screen", "status": "failed",
                "fragment": "I couldn't make sense of the screen."}
    try:
        if topic:
            entity_ledger.record_entity(topic, topic, kind="topic",
                                        source="screen")
            _set_last_screen_topic(topic)
        if creator:
            _set_last_screen_creator(creator)
        _set_last_screen_report(topic, tip, creator)
    except Exception:
        pass
    return {"kind": "screen", "status": "ok",
            "fragment": "I looked at your screen.",
            "output_query": topic or tip[:200],
            "output_creator": creator,
            "output_content": tip,
            # RANK 1: the stored observation the research step re-reads
            # (same image, item inventory) instead of re-capturing.
            "output_obs_id": str(result.get("observation_id") or "")}


def _mi_research_step(step, results):
    """Run one research step synchronously on the chain worker thread."""
    prior = results[step["consumes"][0]] if step.get("consumes") else None
    obs_id = str(prior.get("output_obs_id") or "") if prior else ""
    creator_override = False
    used_text_extractor = False
    aspect_composed = False
    base = ""
    if prior and prior.get("status") == "ok":
        base = str(prior.get("output_query")
                   or prior.get("output_content") or "").strip()
    clause = step.get("text") or ""
    # Live fix: a clause that asks for the creator is about the MAKER read
    # on screen — search that name, never the video topic again. A clause
    # with its own subject ("the creator of monalisa") keeps its subject.
    if prior is not None and _CREATOR_ATTR_RE.search(clause):
        creator = str(prior.get("output_creator") or "").strip() \
            or _get_last_screen_creator()
        if creator and not _is_deictic_query(creator):
            tokens = [t for t in re.split(r"[\W_]+", clause.lower()) if t]
            allowed = _DEREF_FILLERS | _CREATOR_POINTER_EXTRA
            if not [t for t in tokens if t not in allowed]:
                base = creator
                creator_override = True
    # Live fix: research the SPECIFIC thing the user pointed at, read from
    # the screen report — not "whatever was seen". When the chat model can
    # only guess, ask "is this what you mean?" instead of searching a
    # generic topic; the user can confirm, decline, or correct the target.
    if (prior is not None and prior.get("status") == "ok" and base
            and not _CREATOR_ATTR_RE.search(clause)
            and _mi_needs_screen_extraction(clause, base)):
        screen_content = str(prior.get("output_content") or "")
        # A possessive back-reference ("research about their release dates")
        # only ADDS an aspect to the subject the screen step just identified;
        # compose it in code — the stateless second vision call has no
        # antecedent for "their", and asking "which part?" would be wrong
        # because the identification already happened.
        composed = _mi_subject_aspect_query(clause, obs_id)
        if composed:
            print("[CHAIN] Subject + aspect query:", composed)
            base = composed
            aspect_composed = True
        else:
            # Primary: the vision model identifies the exact thing the user
            # pointed at — against the SAME stored observation the screen
            # step captured (RANK 1), so a screen that changed in between
            # cannot swap the subject. The chat-side extraction is only the
            # fallback when no stored observation served the call.
            extracted, vgrade = _mi_vision_screen_target(clause, obs_id=obs_id)
            if vgrade is not None:
                confident = _grade_confident(vgrade)
            else:
                confident = bool(extracted)
            if not extracted and vgrade is None:
                extracted, confident = _mi_extract_screen_query(
                    clause, base, screen_content)
                used_text_extractor = bool(extracted)
            if extracted and confident:
                base = extracted
            elif extracted:
                question = ('Sir, is this what you mean — "%s"? Say yes and '
                            "I'll research it, or correct me."
                            % _mi_clip(extracted, 110))
                _set_pending_screen_clarify(clause, base, screen_content,
                                            obs_id=obs_id)
                _arm_confirmation(clause, query=extracted, question=question)
                print("[CHAIN] Screen subject unsure — asking before searching:",
                      extracted)
                return {"kind": "research", "status": "asked",
                        "fragment": question}
            else:
                hint = _mi_clip(base or screen_content, 70)
                question = ("Sir, I can see %s, but I couldn't tell exactly "
                            "what to search for — tell me which part, and I'll "
                            "research it." % hint)
                _set_pending_screen_clarify(clause, base, screen_content,
                                            obs_id=obs_id)
                print("[CHAIN] Screen subject not extractable — asking once.")
                return {"kind": "research", "status": "asked",
                        "fragment": question}
    try:
        if base and not _is_deictic_query(base):
            query = base[:220]
            # Deterministic fallback: the aspect the user asked about
            # ("...about dimensions") appended to the identified subject.
            if not creator_override and not used_text_extractor \
                    and not aspect_composed:
                aspect = _mi_aspect_suffix(clause, base, query, obs_id)
                if aspect:
                    query = _mi_clip(" ".join([query] + aspect), 220)
                    print("[CHAIN] Subject + aspect query:", query)
            # RANK 10: the MODEL composes the final search phrase — it gets
            # the target description from the observation (exact on-screen
            # text, kinds, creators, the vision description) and the user's
            # request. The deterministic query above is its input hint and
            # the fallback when the model is unavailable or breaks a rule.
            if obs_id and not creator_override:
                obs = get_observation(obs_id)
                if obs is not None and getattr(obs, "items", None):
                    composed_q = compose_search_query(obs, clause, query)
                    if composed_q and composed_q != query:
                        print("[CHAIN] Model-composed query:", composed_q)
                    query = composed_q or query
        else:
            query = _resolve_search_query(clause)
        if not query:
            return {"kind": "research", "status": "failed",
                    "fragment": "I could not tell which streamer or item you "
                                "meant — name it and I will search at once."}
        deep = bool(re.search(r"\bdeep\s*(?:research|search)|deepsearch\b",
                              clause, re.IGNORECASE))
        _set_last_research_topic(query)
        if deep:
            overview = fetch_ai_overview_text(query)
            result = run_research(query, pinned_overview=overview)
        else:
            result = run_quick_search(query)
    except Exception as exc:
        return {"kind": "research", "status": "failed",
                "fragment": "I couldn't finish the research (%s)."
                            % _mi_clip(exc, 100)}
    if not isinstance(result, dict):
        return {"kind": "research", "status": "failed",
                "fragment": "The research came back empty."}
    if result.get("stopped"):
        return {"kind": "research", "status": "failed",
                "fragment": "The research was stopped."}
    summary = str(result.get("spoken_summary")
                  or result.get("overview_text") or "").strip()
    content = str(result.get("detailed_markdown") or summary or "").strip()
    if not content:
        return {"kind": "research", "status": "failed",
                "fragment": "The research came back empty."}
    if len(content) > 60000:
        content = content[:60000]
    if deep:
        try:
            push_research_result({
                "query": result.get("query") or query,
                "markdown": result.get("detailed_markdown") or "",
                "videos": result.get("related_videos") or [],
                "report_path": result.get("report_path") or "",
                "visited_count": result.get("visited_count") or 0,
                "failed_count": result.get("failed_count") or 0,
            })
        except Exception:
            pass
    return {"kind": "research", "status": "ok",
            "fragment": _mi_clip(summary, 900) or "Research done.",
            "output_query": str(result.get("query") or query),
            "output_content": content,
            "spoken_summary": summary}


def _mi_report_filename(clause):
    """A safe .txt report name for the write step, from the clause."""
    name = ""
    match = re.search(
        r"\b(?:named|called|by\s+the\s+name(?:\s+of)?|name\s+it)\s+"
        r"([A-Za-z0-9_.\- ]{1,60})", clause or "", re.IGNORECASE)
    if match:
        name = match.group(1).strip().strip("\"'")
        name = re.split(
            r"\s+(?:with|and|in|on|for|to|about|containing|that|which)\b",
            name, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    if not name:
        name = "jarvis_report.txt"
    name = name.replace("\\", "/").split("/")[-1].strip()
    if not re.search(r"\.\w{1,6}$", name):
        name += ".txt"
    return name or "jarvis_report.txt"


def _mi_free_path(path):
    """Never overwrite: "name.txt" -> "name (2).txt" when it exists."""
    try:
        if not os.path.exists(path):
            return path
        base, ext = os.path.splitext(path)
        for n in range(2, 51):
            candidate = "%s (%d)%s" % (base, n, ext)
            if not os.path.exists(candidate):
                return candidate
    except Exception:
        pass
    return path


#: A chain task clause that creates a FOLDER, not a file.
_MI_FOLDER_RE = re.compile(r"\b(folder|directory)\b", re.IGNORECASE)


def _mi_safe_folder_name(name):
    """One filesystem-safe folder name (single component, bounded)."""
    name = str(name or "").strip().strip("\"'")
    name = re.sub(r"\s+", " ", name)
    name = name.replace("\\", "/").split("/")[-1].strip(" .")
    return name[:60]


#: A folder name that is a POINTER ("by the name of this website", "the
#: player you found") — not a literal. The findings extraction should name
#: it, not the folder.
_FOLDER_NAME_POINTER_RE = re.compile(
    r"\b(?:this|that|these|those|it|whatever|website|site|player|song|"
    r"video|movie|model|channel|creator)\b"
    r"|\byou\s+(?:find|found|see|get|learn)\b",
    re.IGNORECASE)


def _mi_folder_name_phrase(clause):
    """The "named X" folder name in the FOLDER half, or "" when absent."""
    # R20: the FILE half of the clause ("…txt file by the name info…")
    # names the FILE, not the folder — only the folder half (before any
    # file mention) may name the folder.
    head = re.split(
        r"\b(?:txt|text|file|document|note|report)\b",
        str(clause or ""), maxsplit=1, flags=re.IGNORECASE)[0]
    match = re.search(
        r"\b(?:named|called|name\s+it|by\s+the\s+name(?:\s+of)?|"
        r"with\s+the\s+name(?:\s+of)?|in\s+the\s+name(?:\s+of)?)\s+"
        r"([A-Za-z0-9_.\- ]{1,60})",
        head, re.IGNORECASE)
    if not match:
        return ""
    name = match.group(1).strip().strip("\"'")
    name = re.split(
        r"\s+(?:with|and|in|on|for|to|by|about|containing|that|which)\b",
        name, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    # R20: a pointer name ("by the name of this website") is not a literal —
    # leave it to the findings extraction instead of naming the folder
    # "of this website".
    if _FOLDER_NAME_POINTER_RE.search(name):
        return ""
    return _mi_safe_folder_name(name)


#: A folder named by POSITION ("inside information folder on my desktop") —
#: the name rides immediately before the folder word, even after the file
#: mention. An article ("the folder"), a pointer ("that folder") or a
#: grammar word names nothing.
_FOLDER_POSITIONAL_RE = re.compile(
    r"\b(?:in|inside|into|on|at|under)\s+(?:the\s+|my\s+|our\s+)?"
    r"([A-Za-z0-9_.\-]+(?:\s+[A-Za-z0-9_.\-]+){0,2})\s+"
    r"(?:folder|directory)\b", re.IGNORECASE)

_FOLDER_POSITIONAL_STOPWORDS = {
    "a", "an", "the", "my", "our", "his", "her", "their", "its",
    "this", "that", "these", "those", "new", "same", "one", "some",
    "any", "every", "following", "above", "below", "txt", "text",
    "file", "document", "note", "report", "folder", "directory",
    "name", "named", "called", "in", "on", "at", "by", "with",
    "under", "inside", "into", "and", "or", "of", "for", "to",
}


def _mi_folder_name_positional(clause):
    """The "<name> folder" positional folder name, or "" when absent."""
    for match in _FOLDER_POSITIONAL_RE.finditer(str(clause or "")):
        name = match.group(1).strip().strip("\"'").strip(" .")
        words = [w for w in re.split(r"\s+", name.lower()) if w]
        if not words or len(name) > 40:
            continue
        if any(w in _FOLDER_POSITIONAL_STOPWORDS for w in words):
            continue
        if _FOLDER_NAME_POINTER_RE.search(name):
            continue
        return _mi_safe_folder_name(name)
    return ""


def _mi_folder_name(clause):
    """The explicit folder name in the clause, or "" when it is unnamed.

    Live fix: a clause can name the folder by position ("store it in a
    txt file inside information folder on my desktop") — the name rides
    before the folder word, past the file mention where the phrase
    search ("named X") never looks.
    """
    return _mi_folder_name_phrase(clause) \
        or _mi_folder_name_positional(clause)


def _extract_name_from_findings(content):
    """The single name the findings identify (person/channel), or "".

    Bounded LLM extraction with strict JSON — a name or nothing, never an
    invented guess. Returns "" on any failure so the caller can ask.
    """
    text = str(content or "").strip()
    if len(text) < 20:
        return ""
    prompt = (
        "From the research findings below, return the ONE person, artist, "
        "creator or channel name that answers the request. Use the exact "
        "name as written. If no single name is clearly identified, return "
        "an empty string.\n\n"
        "Reply STRICT JSON only: {\"name\": \"...\"}\n\n"
        "===== FINDINGS =====\n" + text[:4000]
    )
    try:
        result = _ask_chat_nonstream(
            [{"role": "system",
              "content": "Return strict JSON only. No markdown, no extra text."},
             {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=60)
        if not result or not result.get("choices"):
            return ""
        raw = result["choices"][0].get("message", {}).get("content", "")
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return ""
        name = str(json.loads(match.group(0)).get("name") or "").strip()
    except Exception as exc:
        logging.warning("[CHAIN] Name extraction failed: %s", exc)
        return ""
    if not name or len(name) > 60 or len(name.split()) > 5:
        return ""
    if _META_QUERY_RE.search(name) or not re.search(r"[A-Za-z]", name):
        return ""
    return _mi_safe_folder_name(name)


_PENDING_FOLDER_TTL = 300.0
_pending_folder_name = {"text": "", "at": 0.0}


def _set_pending_folder_name(clause):
    """Remember the folder ask so "name it X" can answer it. Never raises."""
    global _pending_folder_name
    try:
        with _screen_topic_lock:
            _pending_folder_name = {"text": str(clause or "")[:400],
                                    "at": time.time()}
    except Exception:
        pass


def _clear_pending_folder_name():
    global _pending_folder_name
    try:
        with _screen_topic_lock:
            _pending_folder_name = {"text": "", "at": 0.0}
    except Exception:
        pass


#: Live fix: the FILE-name ask's remembered request. The folder task asked
#: WHICH name to give the txt file inside the (existing) folder — the
#: answer ("name it random.txt") arrives on a LATER turn, so the clause
#: (and its folder half) must survive the chain worker's death.
_pending_file_name = {"text": "", "at": 0.0}


def _set_pending_file_name(clause):
    """Remember the file-name ask so "name it X" can answer it."""
    global _pending_file_name
    try:
        with _screen_topic_lock:
            _pending_file_name = {"text": str(clause or "")[:400],
                                   "at": time.time()}
    except Exception:
        pass


def _clear_pending_file_name():
    global _pending_file_name
    try:
        with _screen_topic_lock:
            _pending_file_name = {"text": "", "at": 0.0}
    except Exception:
        pass


#: File content that is a POINTER to the findings ("the information you
#: found", "your report") — the real findings must be substituted, not the
#: pointer words written into the file.
_FILE_CONTENT_POINTER_RE = re.compile(
    r"\b(?:whatever|what)\s+you\s+(?:find|found|see|get|learn)\b"
    r"|\byou\s+(?:found|find)\b"
    r"|\byour\s+(?:report|findings|research|results|summary)\b"
    r"|\bthe\s+(?:information|info|findings|results|report|research|"
    r"summary|answer)\b",
    re.IGNORECASE)


#: The explicit file name inside a folder request ("txt file by the
#: name random") — the lookahead stops the name at the next clause
#: boundary so it never swallows the rest of the request.
_FILE_NAME_RE = re.compile(
    r"\bfile\s+(?:named\s+|called\s+|by\s+the\s+name(?:\s+of)?\s+)"
    r"([A-Za-z0-9_ .\-]{1,60}?)(?=\s*(?:,|;|\band\b|\bon\b|\bin\b|"
    r"\binside\b|\binto\b|\bat\b|\bwith\b|\bwrite\b|$))", re.IGNORECASE)


def _mi_file_name(clause):
    """The explicit "<file> named X" file name, or "" when it is unnamed."""
    match = _FILE_NAME_RE.search(str(clause or ""))
    if not match:
        return ""
    name = match.group(1).strip().strip("\"'")
    if not name:
        return ""
    if not re.search(r"\.\w{1,5}$", name):
        name += ".txt"
    return _mi_safe_folder_name(name)


def _mi_file_request(clause, fallback_content=""):
    """(file_name, content) for a file the clause asks for INSIDE the folder.

    "…create a txt file by the name info and inside that txt file write
    hello from jarvis" -> ("info.txt", "hello from jarvis"). Empty when the
    clause asks for no file.

    R20: when the clause asks for a file but its content is a POINTER to the
    findings ("write the information you found inside that txt file"), the
    *fallback_content* (the chain's real findings) is written instead of
    the pointer words — never conjured when the clause asks for no file.
    """
    text = str(clause or "")
    match = _FILE_NAME_RE.search(text)
    rest = text[match.end():] if match else text
    try:
        content = str(task_agent_module._located_write_content(rest, "")
                      or "").strip()
    except Exception:
        content = ""
    if content and _FILE_CONTENT_POINTER_RE.search(content):
        content = ""
    if not content:
        asks_file = bool(re.search(
            r"\b(?:txt|text|file|document|note|report)\b",
            text, re.IGNORECASE))
        if asks_file and str(fallback_content or "").strip():
            content = str(fallback_content).strip()
    content = re.sub(
        r"\s+(?:inside|in|into)\s+(?:the|that|this)\s+"
        r"(?:txt\s+|text\s+)?file\.?$", "", content,
        flags=re.IGNORECASE).strip()
    if not content:
        return "", ""
    name = _mi_file_name(text) \
        or ("info.txt" if re.search(r"\b(?:txt|text)\b", text,
                                    re.IGNORECASE) else "notes.txt")
    if not name:
        return "", ""
    return name, content


#: The clause says the folder ALREADY EXISTS ("the folder is already there
#: on the desktop by the name information") — it is a LOCATION for the
#: file, never a new folder to create.
_FOLDER_EXISTS_RE = re.compile(
    r"\balready\s+(?:there|exists?|existing|created|present|made)\b"
    r"|\b(?:pre[-\s]?existing|existing)\s+(?:folder|directory)\b",
    re.IGNORECASE)


def _mi_folder_task_step(msg, clause, step=None, results=None):
    """Arm a folder creation; an unnamed folder asks instead of guessing.

    Live fix: "whichever name you find … by that name" is named from the
    research findings (one bounded LLM extraction); when no name can be
    extracted the ask is remembered so "name it X" completes it.
    Live fix 2: "…and inside that folder create a txt file by the name
    info and write hello" arms BOTH steps in the one plan — the folder and
    the file inside it — so the follow-up file work is never dropped.
    Live fix 3: "the folder is already there by the name information" —
    the folder is a LOCATION for the file: it is reused as-is (never a
    duplicate "information (2)"), and when the findings file is unnamed
    it ASKS for the file name (remembered) so "name it random.txt" names
    the FILE instead of birthing a "random" folder.
    """
    name = _mi_folder_name(clause)
    findings = ""
    for idx in (step or {}).get("consumes") or []:
        if 0 <= idx < len(results or []):
            prior = results[idx]
            if prior.get("status") == "ok":
                findings = str(prior.get("output_content") or "")
                if findings:
                    break
    if not findings:
        # R20: the folder-name answer arrives on a LATER turn, after the
        # chain worker is gone — the findings it produced are stored.
        findings = _get_chain_findings()
    if not name and results:
        content = ""
        for idx in (step or {}).get("consumes") or []:
            if 0 <= idx < len(results):
                prior = results[idx]
                if prior.get("status") == "ok":
                    content = str(prior.get("output_content") or "")
                    if content:
                        break
        if content:
            name = _extract_name_from_findings(content)
    if not name:
        _set_pending_folder_name(clause)
        return {"kind": "task", "status": "failed",
                "fragment": "I have the findings, but I couldn't tell what "
                            "to name the folder — tell me the exact name "
                            "and I'll create it."}
    try:
        folders = task_agent_module._known_folders() or {}
    except Exception:
        folders = {}
    base = folders.get("desktop") or folders.get("home") or ""
    if not base:
        return {"kind": "task", "status": "failed",
                "fragment": "I couldn't find your Desktop folder."}
    try:
        if task_agent_module.has_pending_task_confirmation():
            return {"kind": "task", "status": "failed",
                    "fragment": "Another task is already waiting for your "
                                "approval, so I left the folder uncreated — "
                                "ask me again once that is settled."}
    except Exception:
        pass
    direct = os.path.join(base, name)
    try:
        folder_exists = os.path.isdir(direct)
    except Exception:
        folder_exists = False
    file_name, file_content = _mi_file_request(clause, findings)
    # Live fix 3: the folder the user says is already there, with nothing
    # new to put inside it — never re-create it.
    if folder_exists and _FOLDER_EXISTS_RE.search(clause) \
            and not (file_name and file_content):
        return {"kind": "task", "status": "failed",
                "fragment": "The folder %s is already there — tell me what "
                            "to create inside it and I'll do it." % name}
    # Live fix 3: the findings will be written but the file is unnamed —
    # ASK for the file name, never guess "info.txt"; the ask is remembered
    # so "name it random.txt" names the FILE, not a second folder.
    if (file_name and file_content and findings
            and not _mi_file_name(clause)
            and str(file_content).strip() == str(findings).strip()):
        _set_pending_file_name(clause)
        _clear_pending_folder_name()
        return {"kind": "task", "status": "failed",
                "fragment": "I have the findings for the folder %s — what "
                            "name should I give the txt file? Say 'name it "
                            "<name>' and I'll prepare the write." % name}
    # Live fix 3: an existing folder hosting the file is reused, never
    # duplicated — the file is the deliverable, the folder the location.
    reuse = bool(folder_exists and file_name and file_content)
    path = direct if reuse else _mi_free_path(direct)
    plan = {
        "ok": True,
        "confidence": 0.9,
        "summary": ("Writing %s inside the existing folder %s."
                    % (file_name, name)) if reuse else
                   ("Creating the folder %s." % name),
        "requires_confirmation": True,
        "command_text": msg,
        "steps": [],
    }
    if not reuse:
        plan["steps"].append({
            "tool": "code.create_folder",
            "args": {"path": path},
            "risk": "safe",
            "reason": "Creating the folder %s." % name,
        })
    if file_name and file_content:
        file_path = os.path.join(path, file_name)
        if not reuse:
            plan["summary"] = ("Creating the folder %s and writing %s "
                               "inside it." % (name, file_name))
        plan["steps"].append({
            "tool": "code.write_file",
            "args": {"path": file_path, "content": file_content,
                     "create_only": True},
            "risk": "safe",
            "reason": "Writing %s inside it with: %s"
                      % (file_name, _mi_clip(file_content, 80)),
        })
    try:
        task_agent_module._arm_plan_confirmation(plan, {}, task_text=msg)
        prompt = task_agent_module.confirmation_prompt(plan)
    except Exception as exc:
        return {"kind": "task", "status": "failed",
                "fragment": "I couldn't prepare the folder creation (%s)."
                            % _mi_clip(exc, 80)}
    _clear_pending_folder_name()
    _clear_pending_file_name()
    return {"kind": "task", "status": "armed", "fragment": prompt,
            "prompt": prompt, "path": path}


def consume_pending_folder_name(msg):
    """The user answers the folder-name ask ("name it james dark").

    Returns the confirmation prompt when the message names the folder for
    a remembered, unfulfilled folder request; None otherwise.
    """
    text = str(msg or "").strip()
    if not text:
        return None
    try:
        with _screen_topic_lock:
            state = dict(_pending_folder_name)
    except Exception:
        return None
    if not state.get("text"):
        return None
    if time.time() - float(state.get("at") or 0.0) > _PENDING_FOLDER_TTL:
        return None
    match = re.match(
        r"^(?:ok(?:ay)?\s*[,.]?\s*)?(?:name\s+(?:it|the\s+folder)|"
        r"call\s+(?:it|the\s+folder)|folder\s+name\s+(?:is|:))\s+(.+)$",
        text, re.IGNORECASE)
    clause = ""
    if match:
        name = _mi_safe_folder_name(match.group(1))
        if not name:
            return None
        print("[CHAIN] Folder name answer:", name)
        # R20: answer against the ORIGINAL clause, not a bare "create a
        # folder named X" — the original also carries the file request
        # ("…and inside that folder create a txt file by the name
        # random…"), which a bare clause dropped, so the file half of the
        # request silently vanished.
        clause = str(state.get("text") or "").strip() or "create a folder"
        if not _mi_folder_name(clause):
            # The name rides on the FIRST folder mention so the folder half
            # keeps it — an appended tail lands past the file mention,
            # where the folder-name search never looks.
            clause = re.sub(r"\b(folder|directory)\b",
                            "folder named %s" % name,
                            clause, count=1, flags=re.IGNORECASE)
        if not _mi_folder_name(clause):
            clause = "create a folder named %s. %s" % (name, clause)
    else:
        # Live fix: the answer arrives as a full clarification ("the
        # folder is already there on the desktop by the name information,
        # create a txt file inside that folder…") — it names the folder
        # itself and re-carries the file request, so it completes the ask
        # directly instead of falling into chat and leaving the ask stale.
        inline = _mi_folder_name_phrase(text)
        if not inline or not _MI_FOLDER_RE.search(text):
            return None
        name = inline
        print("[CHAIN] Folder name answer (inline):", name)
        clause = text
    result = _mi_folder_task_step(text, clause)
    # R20: the "name it X" answer is a real turn — commit both halves so
    # the armed folder is recallable later.
    _remember_user_turn(text)
    try:
        add_message(
            "assistant",
            str(result.get("prompt") or result.get("fragment") or ""))
    except Exception:
        pass
    if result.get("status") == "armed":
        return result.get("prompt")
    return result.get("fragment")


def consume_pending_file_name(msg):
    """The user answers the FILE-name ask ("name it random.txt").

    Returns the confirmation prompt when the message names the FILE for a
    remembered, unfulfilled folder+file request; None otherwise. The
    answer names the file INSIDE the resolved folder — never a second
    folder.
    """
    text = str(msg or "").strip()
    if not text:
        return None
    try:
        with _screen_topic_lock:
            state = dict(_pending_file_name)
    except Exception:
        return None
    if not state.get("text"):
        return None
    if time.time() - float(state.get("at") or 0.0) > _PENDING_FOLDER_TTL:
        return None
    match = re.match(
        r"^(?:ok(?:ay)?\s*[,.]?\s*)?(?:name\s+(?:it|the\s+(?:txt\s+|"
        r"text\s+)?file)|call\s+(?:it|the\s+(?:txt\s+|text\s+)?file)|"
        r"(?:txt\s+|text\s+)?file\s+name\s+(?:is|:))\s+(.+)$",
        text, re.IGNORECASE)
    clause = ""
    if match:
        name = _mi_file_name("file named %s" % match.group(1))
        if not name:
            return None
        print("[CHAIN] File name answer:", name)
        clause = str(state.get("text") or "").strip()
    else:
        # The answer arrives as a full request ("create a txt file named
        # random.txt inside that folder…") — it names the file itself.
        inline = _mi_file_name(text)
        if not inline:
            return None
        name = inline
        print("[CHAIN] File name answer (inline):", name)
        clause = text
    if not _mi_file_name(clause):
        # The name rides on the FIRST file mention so the file half keeps
        # it — never past a clause boundary.
        clause = re.sub(r"\bfile\b", "file named %s," % name,
                        clause, count=1, flags=re.IGNORECASE)
    if not _mi_file_name(clause):
        clause = "create a txt file named %s. %s" % (name, clause)
    result = _mi_folder_task_step(text, clause)
    # The answer is a real turn — commit both halves so the written file
    # is recallable later.
    _remember_user_turn(text)
    try:
        add_message(
            "assistant",
            str(result.get("prompt") or result.get("fragment") or ""))
    except Exception:
        pass
    if result.get("status") == "armed":
        _clear_pending_file_name()
        return result.get("prompt")
    return result.get("fragment")


def _mi_task_step(msg, step, results):
    """Compose the file write and arm the ONE approval; nothing runs now."""
    clause = step.get("text") or ""
    if _MI_FOLDER_RE.search(clause):
        return _mi_folder_task_step(msg, clause, step, results)
    content = ""
    for idx in step.get("consumes") or []:
        prior = results[idx]
        if prior.get("status") == "ok" and prior.get("output_content"):
            content = str(prior["output_content"])
            break
    if not content:
        try:
            content = str(task_agent_module._located_write_content(
                clause, "") or "").strip()
        except Exception:
            content = ""
    if not content:
        return {"kind": "task", "status": "failed",
                "fragment": "I don't have anything to put in that file yet."}
    try:
        folders = task_agent_module._known_folders() or {}
    except Exception:
        folders = {}
    base = folders.get("desktop") or folders.get("home") or ""
    if not base:
        return {"kind": "task", "status": "failed",
                "fragment": "I couldn't find your Desktop folder."}
    path = _mi_free_path(os.path.join(base, _mi_report_filename(clause)))
    try:
        if task_agent_module.has_pending_task_confirmation():
            return {"kind": "task", "status": "failed",
                    "fragment": "Another task is already waiting for your "
                                "approval, so I left the file uncreated — "
                                "ask me again once that is settled."}
    except Exception:
        pass
    plan = {
        "ok": True,
        "confidence": 0.9,
        "summary": "Creating %s with the findings."
                   % os.path.basename(path),
        "requires_confirmation": True,
        "command_text": msg,
        "steps": [{
            "tool": "code.write_file",
            "args": {"path": path, "content": content, "create_only": True},
            "risk": "safe",
            "reason": "Saving the findings to %s." % os.path.basename(path),
        }],
    }
    try:
        task_agent_module._arm_plan_confirmation(plan, {}, task_text=msg)
        prompt = task_agent_module.confirmation_prompt(plan)
    except Exception as exc:
        return {"kind": "task", "status": "failed",
                "fragment": "I couldn't prepare the file write (%s)."
                            % _mi_clip(exc, 80)}
    return {"kind": "task", "status": "armed", "fragment": prompt,
            "prompt": prompt, "path": path}


def _run_multi_intent_chain(msg, plan):
    """Execute the chain in order, then deliver ONE honest summary."""
    global _mi_chain_active
    results = []
    # Live fix: the chain ends by STORING the findings in a file — the
    # research summary is then never spoken aloud (the user asked to save
    # it, not to hear it); the full findings still reach the file write,
    # the follow-up findings store and a one-line history note.
    stores_findings = any(
        str(s.get("kind") or "") == "task" and re.search(
            r"\b(?:txt|text|file|document|note|report)\b",
            str(s.get("text") or ""), re.IGNORECASE)
        for s in plan.get("steps") or [])
    try:
        for step in plan.get("steps") or []:
            blocked = [j for j in (step.get("consumes") or [])
                       if j >= len(results)
                       or results[j].get("status") != "ok"]
            if blocked:
                which = ", ".join(
                    (plan["steps"][j].get("kind")
                     if j < len(plan["steps"]) else "earlier")
                    for j in blocked)
                results.append({
                    "kind": step.get("kind"), "status": "skipped",
                    "fragment": "I skipped the %s step because the %s "
                                "step didn't finish."
                                % (step.get("kind"), which)})
                continue
            if step.get("kind") == "screen":
                results.append(_mi_screen_step(step))
            elif step.get("kind") == "research":
                results.append(_mi_research_step(step, results))
            elif step.get("kind") == "task":
                results.append(_mi_task_step(msg, step, results))
            else:
                results.append({"kind": step.get("kind"),
                                "status": "skipped",
                                "fragment": "I skipped an unknown step."})
            if (stores_findings and results[-1].get("kind") == "research"
                    and results[-1].get("status") == "ok"):
                # The findings are destined for the file — say THAT, not
                # the summary the user never asked to hear.
                results[-1]["fragment"] = \
                    "The findings are ready for your file."
            # R20: every finished step leaves its specifics in history, and
            # the latest real output is stored for a later follow-up turn
            # (the folder-name answer arrives after the worker is gone).
            _remember_chain_step(results[-1])
            if results[-1].get("status") == "ok" \
                    and results[-1].get("output_content"):
                _set_chain_findings(results[-1]["output_content"])
    finally:
        try:
            _finish_multi_intent(plan, results)
        finally:
            with _MI_CHAIN_LOCK:
                _mi_chain_active = False


def _chain_signature(plan):
    """Stable identity for a chain request (for repeat detection)."""
    try:
        source = str(plan.get("source") or plan.get("command_text") or "")
        return re.sub(r"\s+", " ", source.lower()).strip()
    except Exception:
        return ""


def _record_chain_run(plan, reply):
    """Remember the last finished chain so a repeat answers from it."""
    global _last_chain_run
    try:
        sig = _chain_signature(plan)
        if not sig:
            return
        with _chain_memory_lock:
            _last_chain_run = {"sig": sig, "at": time.time(),
                               "reply": str(reply or "")}
    except Exception:
        pass


def _finish_multi_intent(plan, results):
    armed = None
    ok_bits = []
    bad_bits = []
    asked_bits = []
    for r in results:
        status = r.get("status")
        if status == "armed":
            armed = r
        elif status == "asked":
            asked_bits.append(str(r.get("fragment") or ""))
        elif status == "ok":
            frag = str(r.get("fragment") or "")
            # Honesty: say what was actually searched, so a resolved query
            # (or a bad one) is visible in the report instead of hidden.
            if r.get("kind") == "research" and r.get("output_query"):
                q = _mi_clip(str(r["output_query"]), 120)
                if q and q.lower() not in frag.lower():
                    frag = 'I searched "%s" — %s' % (q, frag)
            if frag:
                ok_bits.append(frag)
        else:
            bad_bits.append(str(r.get("fragment") or ""))
    if armed:
        lead = " ".join(ok_bits).strip()
        reply = ("Sir, " + (lead + " " if lead else "")
                 + str(armed.get("prompt") or ""))
    elif asked_bits:
        lead = " ".join(ok_bits).strip()
        reply = ((lead + " ") if lead else "") + " ".join(asked_bits)
    elif bad_bits:
        reply = ("Sir, here is where it stands. "
                 + " ".join(ok_bits + bad_bits))
    else:
        reply = "Sir, done. " + " ".join(ok_bits)
    reply = _mi_clip(reply, 1400)
    _record_chain_run(plan, reply)
    _notify_async_reply(reply)


def handle_multi_intent(msg, plan, from_voice=False, voice_compact=False):
    """Ack the chain now, run it on a worker; returns the single ack line."""
    global _mi_chain_active
    sig = _chain_signature(plan)
    with _MI_CHAIN_LOCK:
        if _mi_chain_active:
            with _chain_memory_lock:
                last = dict(_last_chain_run) if _last_chain_run else None
            if last and sig and last.get("sig") == sig:
                return ("Sir, already on that one — I will report the "
                        "moment it is done.")
            return ("Sir, another chained task is already running — "
                    "one moment.")
        with _chain_memory_lock:
            last = dict(_last_chain_run) if _last_chain_run else None
        if (last and sig and last.get("sig") == sig
                and time.time() - float(last.get("at") or 0.0) < 150.0):
            return _mi_clip("Sir, I did that a moment ago — "
                            + str(last.get("reply") or ""), 600)
        _mi_chain_active = True
    try:
        threading.Thread(
            target=_run_multi_intent_chain, args=(msg, plan),
            daemon=True).start()
    except Exception as exc:
        with _MI_CHAIN_LOCK:
            _mi_chain_active = False
        logging.warning("[CHAIN] worker start failed: %s", exc)
        return None
    return multi_intent.render_ack(plan, voice_compact=voice_compact)


def _consume_confirmation(answer):
    """User answered a pending 'shall I look it up?' question.

    Returns the spoken reply when confirmed (or politely declined), else None
    when there was no pending question or the window lapsed.
    """
    with _confirmation_lock:
        global _pending_confirmation
        pending, _pending_confirmation = _pending_confirmation, None
    if not pending:
        return None
    if time.time() > pending["expires"]:
        print("[RESEARCH] Confirmation window expired — research skipped.")
        _set_confirmation_cooldown()
        return None
    # F18: deterministic negatives resolve FIRST and locally. The model used
    # to be asked to interpret every reply — including an explicit "no, don't
    # do it" — so a model that read it as consent could launch research the
    # user had just refused. A refusal is never sent to the model at all.
    verdict = _confirmation_verdict(answer)
    llm_query = None
    if verdict == "no":
        # Live fix: "no not that, the image to the left" is a REDIRECT, not
        # a refusal — re-read the same screen report with the correction.
        clarify = _get_pending_screen_clarify()
        if clarify and _SCREEN_REDIRECT_RE.search(str(answer or "")):
            _clear_pending_screen_clarify()
            print("[RESEARCH] User corrected the target — re-reading the "
                  "screen report.")
            reply = _research_from_screen_correction(clarify, answer)
            if reply:
                return reply
        _clear_pending_screen_clarify()
        print("[RESEARCH] User declined research (explicit negative, no model consult).")
        return "As you wish, sir. I'll just answer from what I know."
    if verdict == "yes":
        pending_query = str(pending.get("query") or "").strip()
        if pending_query:
            # The question named exactly what will be searched ("is this
            # what you mean?") — a yes confirms THAT subject.
            _clear_pending_screen_clarify()
            print("[RESEARCH] Confirmed by user — researching the confirmed "
                  "subject:", pending_query)
            return handle_research_intent(pending_query, derived=True)
        # The model may still refine the QUERY for a locally-affirmative
        # answer, but its verdict can only downgrade to a decline — it can
        # never manufacture consent, and never override the local "no" above.
        llm_verdict, llm_query = _llm_resolve_confirmation(
            pending["message"], answer)
        if llm_verdict == "no":
            print("[RESEARCH] User declined research (model read a refusal).")
            return "As you wish, sir. I'll just answer from what I know."
        print("[RESEARCH] Confirmed by user — launching research.")
        orig_msg = pending["message"]
        if llm_query:
            return handle_research_intent(llm_query, derived=True,
                                          deep=is_deepsearch_request(orig_msg))
        return handle_research_intent(orig_msg,
                                      deep=is_deepsearch_request(orig_msg))
    # Inconclusive locally: this is the only case the model gets to judge.
    llm_verdict, llm_query = _llm_resolve_confirmation(pending["message"], answer)
    if llm_verdict == "no":
        print("[RESEARCH] User declined research.")
        return "As you wish, sir. I'll just answer from what I know."
    if llm_verdict == "yes":
        print("[RESEARCH] Confirmed by user — launching research.")
        orig_msg = pending["message"]
        if llm_query:
            return handle_research_intent(llm_query, derived=True,
                                          deep=is_deepsearch_request(orig_msg))
        return handle_research_intent(orig_msg,
                                      deep=is_deepsearch_request(orig_msg))
    # Not a yes/no — the user rephrased or moved on. Drop the pending question
    # and block re-asking for a window so we don't nag them repeatedly. An
    # open screen-clarify ask is left for its own gate (the reply may be a
    # corrected pointer, not an answer here).
    print("[RESEARCH] Confirmation answered with something else — research skipped.")
    _set_confirmation_cooldown()
    return None


def _disarm_other_gates_if_task_gate_armed():
    """Only one gate may be armed at a time — a task-agent confirmation takes
    precedence over the research and opencode ones. Arming is synchronous, so
    a wiped gate can at worst skip one question."""
    global _pending_confirmation, _pending_opencode_task
    if not has_pending_task_confirmation():
        return
    with _confirmation_lock:
        _pending_confirmation = None
    with _opencode_confirm_lock:
        _pending_opencode_task = None


def _arm_opencode_confirmation(task_description, original_message,
                               contract=None):
    """Arm the 'shall I execute this?' gate before any opencode handoff.

    F16: the execution contract is frozen HERE (at consent time) and travels
    with the armed confirmation, so the engine the user is consenting to is
    the engine that runs — a settings change in between cannot swap it.

    Also wipes the research gate so only one gate stays armed at a time.
    """
    global _pending_opencode_task, _pending_confirmation
    if contract is None:
        try:
            from backend.services import capability_resolver as _resolver

            contract = _resolver.begin_dispatch(
                task_description,
                availability={"opencode": is_opencode_available(),
                              "editor": True},
                task_engine=config.TASK_ENGINE,
                grant="user-confirmed-handoff",
            )
        except Exception:
            contract = None
    with _confirmation_lock:
        _pending_confirmation = None
    with _opencode_confirm_lock:
        _pending_opencode_task = {
            "task_description": task_description,
            "original_message": original_message,
            "expires": time.time() + CONFIRM_WINDOW_SECONDS,
            "contract": contract,
        }


def _consume_opencode_confirmation(answer):
    """Resolve a pending opencode-handoff confirmation (mirrors the task gate).

    R4: the WHOLE sentence decides (shared classify_confirmation): a clean
    "yes" runs the exact deferred effect; "yes, but call it X" re-previews
    under the new name; "yes, a quick look" holds and asks once
    ("create, or only check?"); "yes, don't create it" declines. Extra
    actions never inherit the old yes.

    Returns the spoken reply when the answer resolves the gate, else None
    when nothing is pending, the window lapsed, or the answer is unclear
    (unclear DISCARDS the pending preview so the message falls through to
    normal chat — same as before R4).
    """
    with _opencode_confirm_lock:
        global _pending_opencode_task
        pending, _pending_opencode_task = _pending_opencode_task, None
    if not pending:
        return None
    if time.time() > pending["expires"]:
        print("[TASK] opencode confirmation window expired — skipped.")
        return None
    try:
        from backend.services.task_agent import agent as _ta
        verdict = _ta.classify_confirmation(answer)
    except Exception:
        verdict = None
    if verdict is None:
        if _TASK_CONFIRM_NO_RE.search(answer):
            verdict = "no"
        elif _TASK_CONFIRM_YES_RE.search(answer):
            verdict = "yes"
        else:
            verdict = "unclear"
    if verdict == "no":
        print("[TASK] User declined the opencode handoff.")
        return "As you wish, sir. I will skip that."
    if verdict == "unclear":
        print("[TASK] opencode confirmation answered with something else — skipped.")
        return None
    if verdict == "inspect":
        # Re-arm: the write is HELD, not discarded — a clear "create" next
        # turn both disambiguates and approves the exact displayed effect.
        with _opencode_confirm_lock:
            _pending_opencode_task = pending
        return ("Create it, sir, or only check? "
                "Say create to proceed, or check for a read-only look.")
    if verdict.startswith("rename:"):
        new_name = verdict[len("rename:"):].strip()
        if not new_name:
            with _opencode_confirm_lock:
                _pending_opencode_task = pending
            return ("Sir, what name should I use instead? "
                    "Nothing was started.")
        pending = dict(pending)
        pending["task_description"] = re.sub(
            r"(?:named|called)\s+\S+",
            "named %s" % new_name, pending.get("task_description") or "",
            count=1, flags=re.IGNORECASE) or pending.get("task_description")
        pending["original_message"] = pending.get("task_description")
        with _opencode_confirm_lock:
            _pending_opencode_task = pending
        return ("Sir, noted — %s. Say yes to proceed with that name, "
                "or no to skip." % new_name)
    if verdict == "extra":
        print("[TASK] Confirmed by user — handing off to opencode.")
        reply = _execute_deferred_opencode(
            pending["task_description"], pending["original_message"],
            contract=pending.get("contract"),
        )
        return str(reply or "") + (" Sir, I only did what was previewed — "
                                   "please ask the extra part separately.")
    print("[TASK] Confirmed by user — handing off to opencode.")
    return _execute_deferred_opencode(
        pending["task_description"], pending["original_message"],
        contract=pending.get("contract"),
    )


def _clear_browser_clarification(drop_checkpoint=True):
    """Forget a pending clarification (F08).

    Dropping the checkpoint too is what makes cancel/expiry FINAL: a later
    utterance can never restart the abandoned run from stale progress.
    """
    global _pending_browser_clarification
    with _browser_clarification_lock:
        pending = _pending_browser_clarification
        _pending_browser_clarification = None
    if drop_checkpoint and pending:
        checkpoint_id = pending.get("checkpoint_id")
        if checkpoint_id:
            try:
                browser_agent.drop_checkpoint(checkpoint_id)
            except Exception:
                pass


def _task_result_artifacts(result):
    """F07: the STRUCTURED artifacts a terminal TaskResult reported.

    The audit found task references carrying only a description, so a follow-up
    could not reach the file the run actually produced. These are the real
    paths/URLs, kept as records (never as a clipped blob).
    """
    artifacts = []
    try:
        for item in list(getattr(result, "artifacts", None) or []):
            if isinstance(item, dict):
                artifacts.append(dict(item))
            elif isinstance(item, str) and item.strip():
                artifacts.append({"kind": "ref", "value": item.strip()})
        for entry in list(getattr(result, "evidence", None) or []):
            text = str(entry or "").strip()
            if not text:
                continue
            match = re.search(r"[A-Za-z]:\\[^\s\"']+|/[\w./-]{4,}|"
                              r"https?://\S+", text)
            if match:
                found = match.group(0).rstrip(".,;)")
                kind = "url" if found.startswith("http") else "path"
                artifacts.append({"kind": kind,
                                  "path" if kind == "path" else "url": found})
    except Exception:
        pass
    return artifacts[:24]


def _suspend_browser_checkpoint(task_description, question):
    """F08: persist a checkpoint for a question the agent reported as text."""
    try:
        same_scope = browser_agent.new_checkpoint_id(task_description)
        browser_agent.suspend_checkpoint(
            same_scope, {}, question=question or "")
        return same_scope
    except Exception:
        return None


def _is_browser_clarifying_question(text):
    """F08: ask the browser agent's own STRICT question detector.

    The old check was `"?" in text`, so ANY report carrying a URL with a query
    string ("https://x.com/?q=1") armed a continuation that replayed the run.
    Kept as the single source of truth for every caller in this module.
    """
    try:
        from backend.services.browser_agent import _is_clarifying_question
        return _is_clarifying_question(text)
    except Exception:
        return False


#: F08: a follow-up that is really a NEW request must not be swallowed as an
#: answer to the pending question — it goes through the normal path, where the
#: usual routing and approval gates apply.
_NEW_SCOPE_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"actually|instead|forget (?:it|that)|never ?mind|new task|"
    r"now (?:do|open|search|find|write|create)|"
    r"also (?:do|open|search|find|write|create)|"
    r"and then (?:do|open|search|find|write|create)"
    r")\b", re.IGNORECASE)
_NEW_SCOPE_ACTION_RE = re.compile(
    r"\b(?:open|search|find|book|buy|order|send|delete|write|create|install|"
    r"download|upload|navigate|go to|refactor|edit)\b", re.IGNORECASE)


def _followup_expands_scope(pending, msg):
    """True when the follow-up is a materially different request (F08)."""
    text = (msg or "").strip()
    if not text:
        return True
    if _NEW_SCOPE_PREFIX_RE.match(text):
        return True
    # A long imperative request that names other work is new scope, not an
    # answer: answers to a clarifying question are short and answer-shaped.
    question = (pending or {}).get("question") or ""
    if _NEW_SCOPE_ACTION_RE.search(text) and len(text.split()) > 6:
        try:
            from backend.services.browser_agent import _is_clarifying_question
            if not _is_clarifying_question(text):
                return True
        except Exception:
            return True
    return False


def _consume_browser_followup(msg):
    """If a browser clarifying question is pending, treat the next utterance as the answer.

    Screen-targeted messages must fall through to screen control, not be consumed here.
    The continuation runs through the same deferred machinery (background thread, mute, voice_clip).
    """
    global _pending_browser_clarification
    checkpoint_id = None
    with _browser_clarification_lock:
        pending = _pending_browser_clarification
        if not pending:
            return None
        checkpoint_id = pending.get("checkpoint_id")
        if time.time() > pending["expires"]:
            _pending_browser_clarification = None
            _drop_checkpoint(checkpoint_id)
            return None
        # Screen-targeted messages must not be consumed - let them reach screen_control
        try:
            from backend.services.screen_control import _looks_like_screen_command
            if _looks_like_screen_command(msg):
                return None
        except Exception:
            pass
        # F08 — "cancel does not restart" and "expanded scope requires
        # approval": a negative or a materially different request ENDS the
        # suspended run (checkpoint dropped) and falls through to the normal
        # path, where the usual routing and approval gates apply.
        if _BROWSER_FOLLOWUP_CANCEL_RE.search(msg or "") or \
                _followup_expands_scope(pending, msg):
            _pending_browser_clarification = None
            _drop_checkpoint(checkpoint_id)
            return None
        _pending_browser_clarification = None
    if not checkpoint_id:
        # No checkpoint survived: do NOT re-run the whole description as if it
        # were a fresh task (that replay is exactly what the audit found).
        return ("That earlier task is no longer suspended, sir — its progress "
                "was already cleared. Please tell me the task again.")
    # Build continuation description: the ANSWER only. The checkpoint carries
    # the verified progress and the do-not-repeat list (F08).
    augmented = ("User follow-up answering your last question - continue the "
                 "same task where you left off: %s" % msg)
    # Run through the same deferred machinery, RESUMING the checkpoint.
    return _execute_deferred_opencode(augmented, augmented,
                                      resume_from=checkpoint_id)


#: F08: explicit negatives end a suspended run — never restart it.
_BROWSER_FOLLOWUP_CANCEL_RE = re.compile(
    r"^\s*(?:no|nope|nah|cancel|stop|never ?mind|forget it|quit|abort|"
    r"don'?t|do not|leave it|drop it)\b", re.IGNORECASE)


def _drop_checkpoint(checkpoint_id):
    if not checkpoint_id:
        return False
    try:
        return browser_agent.drop_checkpoint(checkpoint_id)
    except Exception:
        return False


def maybe_proactive_research(user_message):
    """Arm the 'shall I look it up?' gate when a factual reply looked unsure.

    No research starts until the user answers the confirmation question in
    their next message — so Jarvis never silently grabs the browser on its
    own. The caller speaks the confirmation question as its reply.
    """
    if not user_message or not user_message.strip():
        return False
    if not _can_ask_confirmation():
        return False
    with _proactive_research_lock:
        global _proactive_research_fired
        if _proactive_research_fired:
            return False
        _proactive_research_fired = True
    _arm_confirmation(user_message)
    _set_confirmation_cooldown()
    return True


def extract_targets_with_counts(text):
    items = ["youtube", "gmail", "google"]
    targets = []
    for item in items:
        if item in text:
            targets.append((item, 1))
    return targets


def clean_website(value):
    if "youtube" in value:
        return "youtube.com"
    if "gmail" in value:
        return "gmail.com"
    if "google" in value:
        return "google.com"
    return value


def extract_count(text):
    numbers = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    for word, number in numbers.items():
        if word in text:
            return number
    match = re.search(r"\d+", text)
    if match:
        return int(match.group())
    return 1


#: F12 — filler words that may be stripped from the EDGES of a spoken command.
#: Only these exact words are removed, and only from the ends: the payload in
#: between keeps its original case, punctuation, quotes, paths, URLs, flags and
#: newlines.
_VOICE_FILLER_WORDS = frozenset(
    ("jarvis", "jervis", "jarvish", "please", "sir", "hey", "ok", "okay"))
_VOICE_EDGE_RE = re.compile(r"\s*([A-Za-z']+)[\s,.:;!?]*")
_VOICE_TAIL_RE = re.compile(
    r"(?P<pre>.*?)(?P<word>[A-Za-z']+)(?P<post>[\s,.:;!?]*)$", re.DOTALL)


def strip_voice_filler_words(text):
    """F12 — strip leading/trailing filler WORDS, never the payload.

    The old implementation rebuilt the sentence from `[a-z0-9]+` tokens, so a
    spoken command lost its case, punctuation, quotes, URLs, paths, flags and
    newlines before anything could act on it ("open C:\\Users\\Me\\Q4 Report.PDF"
    became "open c users me q4 report pdf"). Now the original string is
    returned with only the filler words removed from its ends.
    """
    if not isinstance(text, str) or not text:
        return text
    original = text
    start, end = 0, len(original)
    # Leading fillers ("jarvis, please open …").
    while start < end:
        match = _VOICE_EDGE_RE.match(original, start)
        if not match or match.group(1).lower() not in _VOICE_FILLER_WORDS:
            break
        start = match.end()
    # Trailing fillers ("… play it please sir").
    while end > start:
        match = _VOICE_TAIL_RE.match(original[start:end])
        if not match or match.group("word").lower() not in _VOICE_FILLER_WORDS:
            break
        end = start + match.start("word")
    # Separators consumed by the loops are dropped; the payload itself is
    # returned byte-identical.
    return original[start:end].rstrip(" \t,.;:!?")


# ── F20: the job that owns the turn currently being processed ──────────────
# The thread-local token lives in backend/services/jobs.py (stdlib-only) so
# leaf modules like task_agent can checkpoint without importing this one.
#: Raised at a phase checkpoint when the owning job was cancelled.
TurnCancelled = jobs.TurnCancelled


def _bind_turn_job(job):
    """Park *job* as this thread's turn job; returns the previous value."""
    return jobs.bind_turn_job(job)


def _unbind_turn_job(previous):
    jobs.unbind_turn_job(previous)


def current_turn_job():
    """The job owning this thread's turn, or None (legacy jobless callers)."""
    return jobs.current_turn_job()


def checkpoint_turn(stage="phase"):
    """F20 checkpoint for composite/synthesis phases.

    A phase must ask the OWNING JOB whether to continue, not a global flag —
    and a jobless caller (background work with no transport) is never
    cancelled here. Raises :class:`TurnCancelled` so the caller can return the
    "stopped" reply without running the next effect.
    """
    return jobs.checkpoint_turn(stage)


def turn_cancelled():
    """True when the owning job was cancelled (non-raising form)."""
    return jobs.turn_cancelled()


def process_message(
    user_message,
    from_voice=False,
    sync_voice=True,
    voice_compact=False,
    commit_response=True,
    stream_reply=None,
    progress=None,
    request_id=None,
    job=None,
):
    """Route one user message.

    *stream_reply* receives final-answer text deltas (F26). *progress*
    receives ``(message, **kw)`` phase updates — used by F30 to say "still
    analysing" while vision runs, so a silent request is never mistaken for a
    dead one. *request_id* is the F23 transport identity this message is
    being executed under; screen answers carry it so a late result can be
    matched to the request that asked for it. *job* is the F20 cancellation
    token owned by the transport: it is parked for the duration of the turn so
    the composite/synthesis phases can checkpoint against a real job instead of
    a global flag.
    """
    print("\n==============================")
    print("[USER]:", user_message)

    token = _bind_turn_job(job)
    # [PERF] P1-19 — marks recorded deep inside the turn (the provider stream
    # clients' `provider_headers`, the audio actor's playback boundaries) have
    # no transport identity of their own, so this turn is named for their
    # duration. Cleared only if it still points at THIS request.
    _bind_latency_request(request_id)
    try:
        return _process_message_inner(
            user_message,
            from_voice=from_voice,
            sync_voice=sync_voice,
            voice_compact=voice_compact,
            commit_response=commit_response,
            stream_reply=stream_reply,
            progress=progress,
            request_id=request_id,
        )
    finally:
        # [P0-10] The speculation started for this turn is cancelled here unless
        # a route ADOPTED it, whatever path the routing took to get back to us.
        _cancel_orphan_turn_racer()
        _unbind_turn_job(token)
        _release_latency_request(request_id)


def _process_message_inner(
    user_message,
    from_voice=False,
    sync_voice=True,
    voice_compact=False,
    commit_response=True,
    stream_reply=None,
    progress=None,
    request_id=None,
):

    msg = user_message.strip()
    voice_log_message = msg
    if from_voice and msg.lower().startswith("command"):
        voice_log_message = msg[len("command"):].strip()

    # R6: every turn refreshes the last-work pointer FIRST, so "execute the
    # last command" on the NEXT turn resolves to this turn's real request.
    # (Status/confirm/stop turns are filtered inside.)
    try:
        _record_last_work_request(msg)
    except Exception:
        pass

    global _proactive_research_fired
    _proactive_research_fired = False

    clear_triggers = [
        "clear memory",
        "forget everything",
        "reset memory",
        "memory clear karo",
        "sab bhool jao",
        "memory reset karo",
    ]
    if msg.lower() in clear_triggers:
        clear_history()
        response = "Memory cleared, sir. Starting fresh."
        if from_voice and sync_voice:
            sync_voice_log(voice_log_message, response)
        return response

    # ── G9 scoped memory / commitments / skills (F06/F07/F09/F10) ──
    # Deterministic explicit-phrase ops — BEFORE any other routing, so a
    # "remember that X" is never misclassified as chat or a task. Returns
    # None for non-memory messages; the store never blanket-forgets and
    # never executes computer control.
    if memory_store is not None:
        memory_reply = None
        _t = time.perf_counter_ns()
        try:
            memory_reply = memory_store.handle_memory_phrase(msg)
        except Exception as exc:
            logging.debug("[MEMORY] phrase op failed: %s", exc)
        _mark_latency_duration(request_id, "preroute_memory_phrase", _t)
        if memory_reply is not None:
            if from_voice and sync_voice:
                sync_voice_log(voice_log_message, memory_reply)
            return memory_reply

    # ── R6: release a held redirect once the browser is quiescent ──
    # A "stop X and do Y" turn held Y behind the stop ack; when a later turn
    # arrives with the browser quiet, the held part runs now. Runs before
    # the confirmation gates so a held write still gets its approval path.
    try:
        if not is_status_question(msg):
            held_reply = _release_held_redirect()
            if held_reply is not None:
                if from_voice and sync_voice:
                    sync_voice_log(voice_log_message, held_reply)
                return held_reply
    except Exception as exc:
        logging.warning("[STOP] Held redirect release failed: %s", exc)

    # ── Rank 1: reference correction revises the last binding ──
    # "No, not that one" / "I meant the other one" rejects the bound entity
    # and re-resolves the same mention — never a new request, never a guess.
    # Runs before R7 (different phrases, no overlap) and before the
    # confirmation gates so it is never eaten as a yes/no answer.
    try:
        reference_reply = handle_reference_correction(msg)
        if reference_reply is not None:
            print("[REFERENCE] Binding revised:", msg)
            if from_voice and sync_voice:
                sync_voice_log(voice_log_message, reference_reply)
            return reference_reply
    except Exception as exc:
        logging.warning("[REFERENCE] Revision handling failed: %s", exc)

    # ── R7: correction replaces, never adds ──
    # "That was meant to be a check" kills the armed create preview (its yes
    # dies with it) and routes the correction as the ONE live request. Runs
    # before the confirmation gates so the old preview can never eat it as
    # an "unclear" answer — and before status so "that was meant to be..."
    # is never misread as a status question.
    try:
        if is_correction(msg):
            print("[CORRECTION] Revision replaces pending request:", msg)
            correction_reply = handle_correction(
                msg, from_voice=from_voice, voice_compact=voice_compact)
            if correction_reply is not None:
                if from_voice and sync_voice:
                    sync_voice_log(voice_log_message, correction_reply)
                return correction_reply
    except Exception as exc:
        logging.warning("[CORRECTION] Revision handling failed: %s", exc)

    # ── Live fix: "name it X" answers a pending folder-name ask ──
    # The chain folder step asked WHICH name to use; the next turn names it
    # instead of falling into chat ("I can certainly go by James Dark").
    try:
        folder_reply = consume_pending_folder_name(msg)
        if folder_reply is not None:
            if from_voice and sync_voice:
                sync_voice_log(voice_log_message, folder_reply)
            return folder_reply
    except Exception as exc:
        logging.warning("[CHAIN] Folder-name answer failed: %s", exc)

    # ── Live fix: "name it X" answers a pending FILE-name ask ──
    # The folder task asked WHICH name to give the txt file inside the
    # (existing) folder; the answer names the FILE, never a second folder.
    try:
        file_reply = consume_pending_file_name(msg)
        if file_reply is not None:
            if from_voice and sync_voice:
                sync_voice_log(voice_log_message, file_reply)
            return file_reply
    except Exception as exc:
        logging.warning("[CHAIN] File-name answer failed: %s", exc)

    # ── R8: one turn, two jobs (status + work) ──
    # "Are you doing the queued task? Also create a folder named x" is NOT
    # one queued blob: the question is answered from live state NOW, and the
    # work half routes as a FRESH turn through the normal path (its own
    # preview/approval, never inheriting the answered question). Runs after
    # held-redirect release and correction (their own compounds) but BEFORE
    # the confirmation gates — neither half may be mistaken for a yes/no.
    try:
        status_half, work_half = split_compound_turn(msg)
    except Exception:
        status_half, work_half = None, None
    if status_half and work_half:
        print("[COMPOUND] Status+work split:", status_half, "||", work_half)
        try:
            status_reply = answer_status_question(status_half)
        except Exception as exc:
            logging.warning("[COMPOUND] Status half failed: %s", exc)
            status_reply = "Sir, nothing is running right now."
        try:
            work_reply = _process_message_inner(
                work_half, from_voice=from_voice, sync_voice=False,
                voice_compact=voice_compact, commit_response=commit_response,
                request_id=request_id)
        except Exception as exc:
            logging.warning("[COMPOUND] Work half failed: %s", exc)
            work_reply = "Sir, I could not start the second part."
        response = "%s %s" % (status_reply, work_reply)
        if from_voice and sync_voice:
            sync_voice_log(voice_log_message, response)
        return response

    # ── R6: "stop X and do Y" splits into control + held redirect ──
    # The stop half signals the worker NOW through the control path; the
    # redirect half is NEVER queued behind the job being stopped. Runs
    # before the research-stop handler so the "and do Y" half survives
    # (the old handler returned early and discarded it).
    try:
        stop_half, redirect = split_stop_and_redirect(msg)
    except Exception:
        stop_half, redirect = None, None
    if stop_half and redirect:
        print("[STOP] Stop-then-redirect:", stop_half, "||", redirect)
        response = handle_stop_then_redirect(
            stop_half, redirect, from_voice=from_voice,
            voice_compact=voice_compact)
        if from_voice and sync_voice:
            sync_voice_log(voice_log_message, response)
        return response

    # ── Explicit websearch stop — before any other routing ──
    _t = time.perf_counter_ns()
    _stop_research = is_stop_research(msg)
    _mark_latency_duration(request_id, "preroute_stop_research", _t)
    if _stop_research:
        print("[RESEARCH] Explicit stop request")
        response = handle_stop_research_request(from_voice=from_voice)
        if from_voice and sync_voice:
            sync_voice_log(voice_log_message, response)
        return response

    # ── Rank 5: bare "stop" / "stop everything" — the general stop ──
    # Runs AFTER the dedicated research/browser phrases (they keep their
    # exact path) and BEFORE the confirmation gates, so a stop is never
    # eaten as an unclear answer to an armed preview.
    try:
        if is_bare_stop(msg) or is_stop_everything(msg):
            print("[STOP] General stop:", msg)
            response = handle_stop_message(msg, from_voice=from_voice)
            if from_voice and sync_voice:
                sync_voice_log(voice_log_message, response)
            return response
    except Exception as exc:
        logging.warning("[STOP] General stop failed: %s", exc)

    # ── Pending 'shall I look it up?' answer — consume before any routing ──
    # R12: a STATUS question ("is it done?", "are you doing X?") is never a
    # confirmation answer — it must reach the grounded status path, not be
    # swallowed here as an unclear answer that DISCARDS the pending preview.
    _t = time.perf_counter_ns()
    confirmed = None if is_status_question(msg) else _consume_confirmation(msg)
    _mark_latency_duration(request_id, "preroute_confirmation", _t)
    if confirmed is not None:
        # R20: the confirmation answer is a real turn of the conversation.
        _remember_user_turn(msg)
        if from_voice and sync_voice:
            sync_voice_log(voice_log_message, confirmed)
        return confirmed

    # ── Pending task-action confirmation answer — consume before routing ──
    # R12: same guard — a status question leaves the armed task preview
    # intact so the later answer can report "waiting for your approval".
    _t = time.perf_counter_ns()
    task_confirmed = (None if is_status_question(msg)
                      else consume_task_confirmation(msg))
    _mark_latency_duration(request_id, "preroute_task_confirmation", _t)
    if task_confirmed is not None:
        # R20: the "confirm" answer is a real turn of the conversation.
        _remember_user_turn(msg)
        _record_native_task_outcome(msg)
        _clear_browser_clarification()
        if from_voice and sync_voice:
            sync_voice_log(voice_log_message, task_confirmed)
        return task_confirmed

    # ── Pending opencode-handoff confirmation answer — consume before routing ──
    # R12: same status guard — "is it done?" must not discard the preview it
    # is asking about.
    _t = time.perf_counter_ns()
    opencode_confirmed = (None if is_status_question(msg)
                          else _consume_opencode_confirmation(msg))
    _mark_latency_duration(request_id, "preroute_opencode_confirmation", _t)
    if opencode_confirmed is not None:
        _clear_browser_clarification()
        if from_voice and sync_voice:
            sync_voice_log(voice_log_message, opencode_confirmed)
        return opencode_confirmed

    # ── Pending browser clarification follow-up — continue same task ──
    _t = time.perf_counter_ns()
    browser_followup = _consume_browser_followup(msg)
    _mark_latency_duration(request_id, "preroute_browser_followup", _t)
    if browser_followup is not None:
        if from_voice and sync_voice:
            sync_voice_log(voice_log_message, browser_followup)
        return browser_followup

    # ── R12 status grounding — answer from live state, not chat recall ──
    # "Are you doing the queued task?", "is it done?", "what's queued" are
    # state queries, not new work: the speaker reads the live snapshot
    # (armed confirmations, running flag, action queue, last verified
    # result) and answers a controlled form. Runs BEFORE the intent
    # classifier so a status turn can never be answered from chat memory
    # ("that topic has not been raised"). Confirmation-shaped status ("yes,
    # check it" answering a pending preview) is NOT status — the gates above
    # already consumed it.
    if not msg.lower().startswith("command"):
        try:
            if is_status_question(msg):
                print("[STATUS] Grounded status answer from live state:", msg)
                status_reply = answer_status_question(msg)
                if from_voice and sync_voice:
                    sync_voice_log(voice_log_message, status_reply)
                return status_reply
        except Exception as exc:
            logging.warning("[STATUS] Grounded answer failed: %s", exc)

    # ── Rank 2: one utterance, many jobs (screen → research → file) ──
    # A compound request whose pieces classify to distinct action kinds
    # becomes an ordered chain: ack once here, run the steps on a worker,
    # and let the file write ask for its approval after the earlier steps
    # finish. Single jobs never reach this gate (build_chain returns None),
    # and the gates above keep priority: confirmations, corrections, stops
    # and status answers are all consumed before a chain is considered.
    if not msg.lower().startswith("command"):
        try:
            _chain = multi_intent.build_chain(msg)
        except Exception as exc:
            _chain = None
            logging.warning("[CHAIN] split failed: %s", exc)
        if _chain is not None:
            chain_reply = handle_multi_intent(
                msg, _chain, from_voice=from_voice,
                voice_compact=voice_compact)
            if chain_reply is not None:
                print("[CHAIN] multi-intent chain:",
                      [s.get("kind") for s in _chain.get("steps") or []])
                # R20: the chain request is a real turn — commit both halves
                # so a follow-up ("put the findings in that folder") has its
                # referent instead of dangling assistant notes.
                _remember_user_turn(msg)
                try:
                    add_message("assistant", chain_reply)
                except Exception:
                    pass
                if from_voice and sync_voice:
                    sync_voice_log(voice_log_message, chain_reply)
                return chain_reply

    # ── Live fix: offered search + search-shaped requests ──
    # "execute it / just do it / don't ask questions" consumes the search
    # Jarvis last offered (instead of falling into chat); a search-shaped
    # request ("can you search about this stream", "find anything about
    # that stream") routes to research with its referent resolved from the
    # screen/entity ledger. Runs after the chain gate (compound turns keep
    # priority) and before the classifier, so the ask-loop cannot start.
    if not msg.lower().startswith("command"):
        try:
            offered_reply = _consume_offered_action(msg)
        except Exception as exc:
            logging.warning("[OFFER] consume gate failed: %s", exc)
            offered_reply = None
        if offered_reply is not None:
            if from_voice and sync_voice:
                sync_voice_log(voice_log_message, offered_reply)
            return offered_reply
        # A correction that restates the previous request ("no i meant the
        # reactions to that release you just researched") is that request.
        # Without this it fell to chat, which answered from stale context.
        try:
            meant_query = _meant_refinement_query(msg)
        except Exception as exc:
            meant_query = ""
            logging.warning("[MEANT] refinement gate failed: %s", exc)
        if meant_query:
            print("[MEANT] Correction restates the request -> research:",
                  meant_query)
            meant_reply = handle_research_intent(
                meant_query, from_voice=from_voice,
                voice_compact=voice_compact, derived=True)
            if from_voice and sync_voice:
                sync_voice_log(voice_log_message, meant_reply)
            return meant_reply
        # Answer to the open screen-research ask ("which part?") — a pointer
        # correction is re-read against the SAME screen report.
        try:
            clarify_reply = consume_screen_research_clarify(msg)
        except Exception as exc:
            clarify_reply = None
            logging.warning("[CHAIN] screen clarify gate failed: %s", exc)
        if clarify_reply is not None:
            if from_voice and sync_voice:
                sync_voice_log(voice_log_message, clarify_reply)
            return clarify_reply
        try:
            # The cheap pure-regex check runs first (P0-10 ordering: a
            # non-search message must not run task predicates before the
            # racer). Explicit task/code-tool requests keep the task gate
            # priority: "task search python decorators" is a task.
            _search_shaped = False
            if is_search_shaped_message(msg):
                _search_shaped = not (
                    is_explicit_task_request(msg)
                    or is_code_tool_request(msg)
                    or is_task_request(msg))
        except Exception:
            _search_shaped = False
        if _search_shaped:
            search_reply = _handle_search_shaped(
                msg, from_voice=from_voice, voice_compact=voice_compact)
            if search_reply is not None:
                if from_voice and sync_voice:
                    sync_voice_log(voice_log_message, search_reply)
                return search_reply

    # ── [P0-10] Speculative chat racer — started HERE ────────────────────
    # This used to start after the whole predicate chain below (task request,
    # code tools, screen control, explicit research, route selection), and every
    # predicate in that chain is dead air between "transcript ready" and "first
    # token". It is safe to start early because the speculation is PURE and
    # CANCELLABLE (F25): it performs no search and no other external effect, it
    # builds from an immutable context snapshot, and its queue is bounded.
    #
    # It starts only when a reply stream exists to feed, for a non-`command`
    # message, and NOT on an orchestrator route: F02 gives that route to the
    # orchestrator, which must not pay for a chat stream it would throw away
    # (the route is deterministic and free to compute — no model call, no I/O).
    #
    # The gates ABOVE deliberately stay in front of it: a pending confirmation
    # or clarification answer must never race a speculative answer (test:
    # test_brain_gate.py), and those consumers return before this point.
    #
    # Cancellation is guaranteed by the turn's own cleanup
    # (``_cancel_orphan_turn_racer`` in ``process_message``'s finally), so no
    # early return below can leak the background stream. It is only skipped for
    # a racer a route ADOPTED, which is the reply being streamed.
    is_explicit_command = msg.lower().startswith("command")
    _t = time.perf_counter_ns()
    route = ("legacy" if is_explicit_command
             else orchestrator_select_route(
                 msg, screen_question=is_screen_question(msg)))
    _mark_latency_duration(request_id, "preroute_route", _t)
    racer = None
    if (stream_reply is not None and not is_explicit_command
            and route != "orchestrator"):
        _mark_latency(request_id, "racer_start")
        try:
            racer = _ChatRacer(msg, voice_compact)
        except Exception as exc:
            logging.warning("[CHAT] Racer start failed: %s", exc)
            racer = None
        _register_turn_racer(racer)

    _t = time.perf_counter_ns()
    _explicit_task_request = (
        (not is_explicit_command) and is_explicit_task_request(msg))
    _mark_latency_duration(request_id, "preroute_task_request", _t)
    if _explicit_task_request:
        _remember_user_turn(msg)
        response = handle_task_message(msg, voice_compact=voice_compact)
        _record_native_task_outcome(msg)
        _disarm_other_gates_if_task_gate_armed()
        if from_voice and sync_voice:
            sync_voice_log(voice_log_message, response)
        return response

    # ── Native code tools — plain read/write/run requests route here before
    # the LLM intent router can send them to opencode. ──
    _t = time.perf_counter_ns()
    _code_tool_request = (
        (not is_explicit_command) and is_code_tool_request(msg))
    _mark_latency_duration(request_id, "preroute_code_tool", _t)
    if _code_tool_request:
        _remember_user_turn(msg)
        response = handle_task_message(msg, voice_compact=voice_compact)
        _record_native_task_outcome(msg)
        _disarm_other_gates_if_task_gate_armed()
        if from_voice and sync_voice:
            sync_voice_log(voice_log_message, response)
        return response

    if not is_explicit_command:
        _t = time.perf_counter_ns()
        screen_control_response = maybe_handle_screen_control_message(msg)
        _mark_latency_duration(request_id, "preroute_screen_control", _t)
        if screen_control_response is not None:
            _clear_browser_clarification()
            if from_voice and sync_voice:
                sync_voice_log(voice_log_message, screen_control_response)
            return screen_control_response

    # ── Deep research — explicit phrases ("look this up", "find out about X") ──
    _t = time.perf_counter_ns()
    _force_research = (not is_explicit_command) and force_research(msg)
    _mark_latency_duration(request_id, "preroute_force_research", _t)
    if _force_research:
        print("[RESEARCH] Explicit research request:", msg)
        response = handle_research_intent(
            msg,
            from_voice=from_voice,
            voice_compact=voice_compact,
            deep=is_deepsearch_request(msg),
        )
        if from_voice and sync_voice:
            sync_voice_log(voice_log_message, response)
        return response

    # ── Tool-intent routing (natural language, no "command" prefix) ──
    # "jarvis open youtube in chrome", "play this song", "open whatsapp" —
    # the LLM intent router decides whether this needs tool calling. If it
    # does, we acknowledge like a human ("On it, sir") and execute; if local
    # execution fails, opencode takes over.
    #
    # [P0-10] `racer` is NOT re-initialised here any more: the speculation for a
    # non-`command` message was started at the top of the turn (and is None only
    # when it must not run), so resetting it would drop a live stream on the
    # floor.
    if not is_explicit_command:
        # ── G8 orchestrator migration (F02) — behind the mode flag ──
        # F02: the ROUTE is selected before any work starts. Speculative chat
        # used to begin first, so an orchestrator-owned goal paid for a chat
        # stream that was thrown away — and on fallback the speculative work
        # had already started, which is exactly the "repeats started work"
        # failure. Speculation now runs only on the legacy chat route.
        # [P0-10] `route` was selected at the TOP of this turn, before the
        # predicate chain — the racer could not be started before it without
        # knowing the orchestrator does not own this message. The F02 rule is
        # unchanged: only the legacy route may speculate.
        if route == "orchestrator":
            orchestrator_outcome = orchestrator_handle_message(
                msg,
                history=list(get_history()),
                screen_question=is_screen_question(msg),
            )
        else:
            orchestrator_outcome = None
        if orchestrator_outcome is not None:
            if racer is not None:
                try:
                    racer.cancel()
                except Exception:
                    pass
            # F02: a proposal must reach the confirmation gate, not just be
            # read for its reply. The gate was armed inside the orchestrator
            # against the SAME plan, so the pending confirmation and the
            # spoken preview cannot disagree.
            reply = _orchestrator_reply(orchestrator_outcome)
            if from_voice and sync_voice:
                sync_voice_log(voice_log_message, reply)
            return reply
        if route == "orchestrator":
            # The orchestrator owned this route and declined (planner
            # unreachable / nothing usable): fall through to the legacy
            # routing WITHOUT a speculative stream — no chat work was
            # started for a goal, and none is silently re-used from here.
            logging.info("[ORCHESTRATOR] declined after selection; legacy routing")
        # [PERF] Deterministic "definitely plain chat" fast path. When the
        # message is obviously conversation, the classifier would only return
        # "chat" — and it is a cloud round trip that gates EVERYTHING below,
        # including the first streamed token. Skipping it here removes that
        # round trip from the hot path. Crucially this yields a *chat verdict*,
        # so every net that follows (screen question, fresh-info search, tool
        # steps, research, task, web-shaped task) still runs and can still
        # upgrade the route. Nothing is skipped except the network call.
        # [PERF] A spoken turn cannot afford the full classifier budget. A
        # timeout here does not break the turn: classify_intent degrades to a
        # `chat` verdict, and the deterministic nets below (screen question,
        # fresh-info, task, web-shaped) still correct any misroute. The typed
        # UI keeps the full window because a human is waiting on a screen and
        # can absorb it.
        if _fastpath_chat_enabled() and is_definitely_plain_chat(msg):
            intent = {
                "intent": "chat",
                "steps": [],
                "task_description": "",
                "query": msg,
                "_source": "fastpath",
            }
            # [PERF] P1-19 — the hop is part of the mark now: "fastpath" is a
            # ~0ms classify where a cloud classifier round trip used to be.
            _mark_latency(request_id, "classify_done",
                          {"hop": "fastpath", "intent": "chat",
                           "classifier": False})
            print("[INTENT] Deterministic chat fast path (classifier skipped)")
        else:
            intent = classify_intent(msg, timeout_ms=INTENT_BUDGET_VOICE_MS
                                     if from_voice else INTENT_BUDGET_MS)
            # [PERF] P1-19 — WHICH hop answered (openrouter | gemini | groq |
            # none) is recorded with the boundary: a throttled primary hides
            # completely inside a single "classify" duration.
            _mark_latency(request_id, "classify_done", {
                "hop": intent.get("_source") or "unknown",
                "intent": intent.get("intent"),
            })
        # Screen-question safety net — deterministic: routes screen Q&A even
        # when the cloud classifier misfires (chat fallback on throttle, or a
        # research misread). tool/task verdicts are exempt: they carry
        # structured steps/task_description the upgrade would discard.
        if intent.get("intent") in ("chat", "research") and is_screen_question(msg):
            intent["intent"] = "region" if is_region_question(msg) else "screen"
            print("[INTENT] Screen-question net ->", intent["intent"])
        # Holdback: non-chat verdicts cancel the speculative chat stream
        if racer is not None and intent.get("intent") != "chat":
            try:
                racer.cancel()
            except Exception:
                pass
        # ── Fresh-info auto-search — chat verdicts that need current world
        # facts (pricing, latest, news…) auto-route to tiered quick-search so
        # a stale chat answer is never served. Greetings and non-question
        # chat are exempt. ──
        if (
            intent.get("intent") == "chat"
            and should_search(msg)
            and not _GREETING_LIKE_RE.search(msg)
            and _QUESTION_SHAPED_RE.match(msg)
        ):
            if racer is not None:
                try:
                    racer.cancel()
                except Exception:
                    pass
            print("[INTENT] Fresh-info chat query -> auto research")
            response = handle_research_intent(
                msg,
                from_voice=from_voice,
                voice_compact=voice_compact,
                derived=True,
                deep=is_deepsearch_request(msg),
            )
            if from_voice and sync_voice:
                sync_voice_log(voice_log_message, response)
            return response
        if intent.get("intent") == "tool" and intent.get("steps"):
            steps = intent["steps"]
            # Web searches ("search X", "news in india"…) belong to the deep
            # research workflow — not the legacy Google-opener "search" step.
            if steps and all(
                isinstance(s, dict) and s.get("action") == "search"
                for s in steps
            ):
                search_q = steps[0].get("input") or msg
                print("[RESEARCH] Tool-search intent rerouted ->", search_q)
                response = handle_research_intent(
                    search_q,
                    from_voice=from_voice,
                    voice_compact=voice_compact,
                    derived=True,
                    deep=is_deepsearch_request(msg),
                )
                if from_voice and sync_voice:
                    sync_voice_log(voice_log_message, response)
                return response
            print("[INTENT] Tool steps:", steps)
            response = handle_tool_intent(
                intent["steps"],
                original_message=msg,
                from_voice=from_voice,
                voice_compact=voice_compact,
            )
            if from_voice and sync_voice:
                sync_voice_log(voice_log_message, response)
            return response

        # ── Screen / region Q&A — captured from the LLM intent router ──
        intent_name = intent.get("intent")

        if intent_name == "research":
            research_q = intent.get("query") or intent.get("task_description") or msg
            # Live fix: a name-refining correction keeps the last researched
            # name; a self-referential classifier query is replaced by a
            # concrete resolution or an honest ask — never searched.
            refined = _refine_last_name_query(msg)
            if refined:
                print("[RESEARCH] Refined to last name:", refined)
                research_q = refined
            elif _META_QUERY_RE.search(str(research_q)):
                concrete = _resolve_search_query(msg)
                if not concrete:
                    response = ("Sir, I could not tell what to search for — "
                                "name it once and I will search immediately.")
                    if from_voice and sync_voice:
                        sync_voice_log(voice_log_message, response)
                    return response
                print("[RESEARCH] Meta query replaced:", concrete)
                research_q = concrete
            print("[INTENT] research intent ->", research_q)
            response = handle_research_intent(
                research_q,
                from_voice=from_voice,
                voice_compact=voice_compact,
                derived=True,
                deep=is_deepsearch_request(msg),
            )
            if from_voice and sync_voice:
                sync_voice_log(voice_log_message, response)
            return response

        if intent_name in ("screen", "region"):
            print(f"[INTENT] {intent_name} intent — analysing screen.")
            if not _screen_qa_busy.acquire(blocking=False):
                return "Still analysing your screen, sir. One moment."
            try:
                # Conversation continuity: a screen Q&A is a real exchange.
                # Both halves land in history like any chat turn, so a
                # follow-up ("research the release date of this volume") can
                # resolve its referent instead of asking what it refers to.
                add_message("user", msg)
                # F30 — register the capture generation BEFORE the slow vision
                # call, so a late answer from an earlier question can be
                # rejected rather than replacing this one.
                capture = begin_screen_capture(request_id)
                # F30 — say where we are: vision is the slow part and the
                # user is waiting on it.
                if progress:
                    try:
                        progress("analysing screen", stage="vision")
                    except Exception:
                        pass
                result = analyze_screen(msg)
                tip = result.get("tip") or "I couldn't analyse the screen, sir."
                # The spoken tip is the assistant half of the exchange; it
                # rides the normal commit rule (commit_response).
                if commit_response:
                    _commit_chat("assistant", tip)
                evidence = result.get("evidence", [])
                topic = result.get("topic", "")
                creator = result.get("creator", "")
                # Live fix: the topic of every screen answer becomes a real
                # entity, so a later "search this stream" resolves to the
                # streamer/video actually seen instead of asking again. The
                # creator name is remembered separately, so "search about
                # this creator" searches the MAKER, not the video.
                if topic:
                    try:
                        entity_ledger.record_entity(topic, topic, kind="topic",
                                                    source="screen")
                        _set_last_screen_topic(topic)
                    except Exception:
                        pass
                if creator:
                    try:
                        _set_last_screen_creator(creator)
                    except Exception:
                        pass
                _set_last_screen_report(topic, tip, creator)
                grounding_links = result.get("grounding_links", [])
                show_images = result.get("show_images", False)
                region = result.get("region")

                # Grounding links come back with the vision result, so they
                # are part of the answer, not a decoration.
                links = []
                for gl in grounding_links[:4]:
                    links.append({
                        "label": gl.get("title", "Source")[:40],
                        "url": gl["url"],
                        "icon": "🔗",
                    })

                # F30 — publish the answer NOW, tagged with the generation
                # registered before analysis. Exploration links and topic
                # images are fetched afterwards, in a bounded phase that
                # patches this same answer id.
                answer_id = push_screen_answer(
                    tip, evidence, links, [], region=region,
                    capture_id=capture["capture_id"], request_id=request_id,
                    capture_seq=capture["seq"],
                )
                if sync_voice:
                    sync_voice_log(voice_log_message, tip)
                # Exploration links and images both need a topic; without one
                # there is nothing to enrich. The phase is bounded in TIME
                # (SCREEN_ENRICH_TIMEOUT) and in CONCURRENCY
                # (SCREEN_ENRICH_MAX_THREADS), and runs strictly after the
                # answer is on screen.
                if answer_id and topic:
                    _start_screen_enrichment(
                        answer_id, capture, tip, evidence, links, topic,
                        show_images, region)
                return tip
            finally:
                _screen_qa_busy.release()

        # ── Complex task — hand off to opencode agent ──
        if intent_name == "task":
            # [S18] a task-class request made WHILE a task runs cannot take
            # the machinery — it waits in the action queue instead.
            held = _queue_action_request(msg, from_voice)
            if held is not None:
                if from_voice and sync_voice:
                    sync_voice_log(voice_log_message, held)
                return held
            description = intent.get("task_description") or msg
            # Live fix: a file/folder task ("create a text file inside that
            # folder and write …") is NATIVE work — it goes to the code-tool
            # planner (where "that folder" resolves from the notebook), never
            # to the browser handoff default.
            try:
                code_like = (is_code_tool_request(msg)
                             and not is_web_shaped_task(msg))
            except Exception:
                code_like = False
            if code_like:
                if racer is not None:
                    try:
                        racer.cancel()
                    except Exception:
                        pass
                print("[TASK] Code-shaped task intent -> native task path:", msg)
                _remember_user_turn(msg)
                response = handle_task_message(msg, voice_compact=voice_compact)
                _record_native_task_outcome(msg)
                _disarm_other_gates_if_task_gate_armed()
                if from_voice and sync_voice:
                    sync_voice_log(voice_log_message, response)
                return response
            print("[INTENT] task intent:", description)
            response = handle_opencode_task(
                description,
                original_message=msg,
                from_voice=from_voice,
                voice_compact=voice_compact,
            )
            if from_voice and sync_voice:
                sync_voice_log(voice_log_message, response)
            return response

    # Implicit task-mode requests ("use browser to login", "open chrome and
    # automate") — the broad connector+action heuristic runs only AFTER tool
    # routing so simple commands like "open youtube in chrome" are executed
    # by the executor, not swallowed by the task agent's planning reply.
    if not msg.lower().startswith("command") and is_task_request(msg):
        # [S18] same hold as the intent-task branch above: task-shaped
        # requests queue while the machinery is busy.
        held = _queue_action_request(msg, from_voice)
        if held is not None:
            if racer is not None:
                try:
                    racer.cancel()
                except Exception:
                    pass
            if from_voice and sync_voice:
                sync_voice_log(voice_log_message, held)
            return held
        if racer is not None:
            try:
                racer.cancel()
            except Exception:
                pass
        if config.TASK_ENGINE == "browser_agent" and is_web_shaped_task(msg):
            print("[TASK] Web-shaped task -> browser-agent handoff:", msg)
            response = handle_opencode_task(
                msg,
                original_message=msg,
                from_voice=from_voice,
                voice_compact=voice_compact,
            )
            if from_voice and sync_voice:
                sync_voice_log(voice_log_message, response)
            return response
        _remember_user_turn(msg)
        response = handle_task_message(msg, voice_compact=voice_compact)
        _record_native_task_outcome(msg)
        _disarm_other_gates_if_task_gate_armed()
        if from_voice and sync_voice:
            sync_voice_log(voice_log_message, response)
        return response

    if not msg.lower().startswith("command"):
        # Honesty guard: a file/folder action the pre-route gates recognise
        # must NEVER be answered by chat prose. A defensive re-check runs
        # here (after the classifier, before any chat reply is produced) so
        # a misrouted action still reaches the task path instead of a model
        # promising work no tool ever does (live bug: a "create a text file
        # ..." turn was answered with "I will get that created" and no file
        # appeared). Unresolvable phrasing (no locatable target) stays chat
        # and asks for the missing detail.
        try:
            if is_code_tool_request(msg):
                if racer is not None:
                    try:
                        racer.cancel()
                    except Exception:
                        pass
                print("[TASK] Code-tool safety net -> task path:", msg)
                _remember_user_turn(msg)
                response = handle_task_message(msg, voice_compact=voice_compact)
                _record_native_task_outcome(msg)
                _disarm_other_gates_if_task_gate_armed()
                if from_voice and sync_voice:
                    sync_voice_log(voice_log_message, response)
                return response
        except Exception as exc:
            logging.warning("[TASK] Code-tool safety net failed: %s", exc)
        # [S6] ONE call, two jobs: the intent router classified this turn AND
        # wrote the answer, so a plain conversational turn no longer pays for a
        # second chat completion. The reply is used ONLY here - after every
        # deterministic net (screen question, fresh-info search, task shape) had
        # its chance to upgrade the route - so a misrouted chat verdict still
        # becomes research/task/screen exactly as before. An empty reply (the
        # fast path, or a router that returned no text) falls through to the
        # normal chat model below.
        router_reply = str((intent or {}).get("reply") or "").strip()
        if router_reply:
            if racer is not None:
                # The answer already exists, so the speculative chat stream is
                # pure waste - same cancellation every non-chat route does.
                try:
                    racer.cancel()
                except Exception:
                    pass
            print("[INTENT] Chat answered by the router (no second LLM call)")
            response = handle_chat(
                msg,
                voice_compact=voice_compact,
                commit_response=commit_response,
                stream=stream_reply,
                answered=router_reply,
            )
            _remember_chat_offer(msg, response)
            if from_voice and sync_voice:
                sync_voice_log(voice_log_message, response)
            return response
        if racer is not None:
            prebuilt = racer.built()
            live_stream = None
            try:
                has_it = racer.has_stream() if hasattr(racer, "has_stream") else getattr(racer, "_has_stream", False)
                if callable(has_it):
                    has_it = has_it()
                if has_it and prebuilt is not None and prebuilt.get("path") == "llm":
                    live_stream = racer.adopt()
                elif has_it and prebuilt is None:
                    # built not ready yet but stream exists - still adopt
                    live_stream = racer.adopt()
            except Exception:
                live_stream = None
            # browser_search path must not pass a live_stream
            if prebuilt is not None and prebuilt.get("path") != "llm":
                live_stream = None
            response = handle_chat(
                msg,
                voice_compact=voice_compact,
                commit_response=commit_response,
                stream=stream_reply,
                prebuilt=prebuilt,
                live_stream=live_stream,
            )
            _remember_chat_offer(msg, response)
        else:
            response = handle_chat(
                msg,
                voice_compact=voice_compact,
                commit_response=commit_response,
                stream=stream_reply,
            )
            _remember_chat_offer(msg, response)
        if from_voice and sync_voice:
            sync_voice_log(voice_log_message, response)
        return response

    command_text = msg[len("command"):].strip()
    if from_voice:
        command_text = strip_voice_filler_words(command_text)
    print("[CMD] Command mode:", command_text)

    browser = None
    if "chrome" in command_text:
        browser = "chrome"
    elif "edge" in command_text:
        browser = "edge"
    elif "brave" in command_text:
        browser = "brave"

    final = []
    parts = re.split(r"\band\b", command_text)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        count = extract_count(part)
        targets = extract_targets_with_counts(part)

        for name, _ in targets:
            for _ in range(count):
                final.append(
                    {"action": "open_website", "input": clean_website(name), "browser": browser}
                )

        if "play" in part:
            query = re.sub(
                r"\b(play|command|open|youtube|and|times|time|one|two|three|four|five|in|on|chrome|edge|brave|jarvis)\b",
                "",
                part,
            )
            query = re.sub(r"\d+", "", query).strip() or "music"
            final.append({"action": "youtube_play", "input": query, "browser": browser})

        if part.startswith(("search ", "google ", "find ")):
            query = re.sub(r"^(search|google|find)\s+", "", part).strip()
            if query:
                final.append({"action": "search", "input": query, "browser": browser})

        if not targets and ("open" in part or "launch" in part):
            words = part.split()
            for verb in ("open", "launch"):
                if verb in words:
                    idx = words.index(verb)
                    if idx + 1 < len(words):
                        target = " ".join(words[idx + 1:]).strip().rstrip(".").rstrip(",")
                        if target:
                            for _ in range(count):
                                final.append(
                                    {
                                        "action": "launch_app",
                                        "input": target,
                                        "browser": browser,
                                    }
                                )
                    break

    print("[CMD] Final actions:", final)

    if not final:
        # F16: the empty-steps handoff no longer depends on the opencode CLI
        # being installed. The resolver decides the executor and fails closed
        # (BLOCKED) for work no permitted engine can do — the browser path
        # must never be gated on a CLI it does not use.
        response = handle_tool_intent(
            [],
            original_message=command_text,
            from_voice=from_voice,
            voice_compact=voice_compact,
        )
        if from_voice and sync_voice:
            sync_voice_log(voice_log_message, response)
        return response

    execute_multiple(final)
    print("[CMD] Execution complete")

    response = generate_command_response(final)
    if from_voice and sync_voice:
        sync_voice_log(voice_log_message, response)
    return response
