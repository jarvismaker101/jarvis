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
from backend.services.openai_compat_client import (
    ask_openai_compat,
    ask_openai_compat_stream,
)
from backend.services.intent import classify_intent
from backend.services.web_task_routing import is_web_shaped_task
from backend.services.orchestrator import handle_message as orchestrator_handle_message
from backend.services.orchestrator import select_route as orchestrator_select_route
from backend.services.opencode_client import run_opencode_task, is_opencode_available, set_narration_enabled
from backend.services.browser_agent import run_browser_task, request_stop as request_browser_task_stop
from backend.services import browser_agent
from backend.services.task_result import (
    TaskResult,
    is_failure_text,
    result_from_reported_text,
)
from backend import config
from backend.services.research_service import run_research, request_stop as request_research_stop, clear_stop_request
from backend.services.quick_search import run_quick_search, fetch_ai_overview_text
from backend.services.screen_analyzer import analyze_screen, is_screen_question, is_region_question
from backend.services.image_fetcher import build_explore_links, fetch_topic_images


def sync_voice_log(message, response):
    def _do():
        try:
            requests.post(
                f"http://127.0.0.1:{BACKEND_PORT}/update-voice-log",
                json={"message": message, "response": response},
                timeout=2,
            )
        except Exception:
            pass
    try:
        threading.Thread(target=_do, daemon=True).start()
    except Exception:
        pass


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


def should_search(query):
    q = query.lower()
    keywords = [
        "today",
        "now",
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
        "how much does",
        "how much is",
        "score",
        "match",
        "news",
        "weather",
        "aaj",
        "abhi",
        "taaza",
        "taza",
        "mausam",
        "khabar",
    ]
    return any(keyword in q for keyword in keywords)


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


def _mark_latency(request_id, name, value_ms):
    """[PERF] Record an already-measured span for *request_id* (no-op if none)."""
    if not request_id:
        return
    try:
        from backend.services import latency as _lat
        _lat.mark_ms(request_id, name, value_ms)
    except Exception:
        pass


def _mark_latency_duration(request_id, name, started_at):
    """[PERF] Record one call's own cost as a span for *request_id*."""
    if not request_id:
        return
    try:
        from backend.services import latency as _lat
        _lat.mark_duration(request_id, name, started_at)
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
    query = None
    try:
        intent = classify_intent(raw_query, timeout_ms=3500)
        if intent.get("intent") in ("research", "tools"):
            candidate = str(intent.get("query") or "").strip()
            if candidate and candidate != raw_query.strip():
                query = candidate[:220]
                print(f"[RESEARCH] Derived query: {query!r} <- {raw_query!r}")
    except Exception as exc:
        logging.warning("[RESEARCH] Query derivation failed: %s", exc)
    if not query:
        query = _heuristic_research_query(raw_query)
    if _is_reference_only(query) and _last_research_topic:
        print(
            f"[RESEARCH] Bare reference ({query!r}) - "
            f"carrying over last topic {_last_research_topic!r}"
        )
        return _last_research_topic
    return query


# Last successfully researched topic, so a Hinglish consent reply such as
# "my internet search karke batao iske bare mein" or "just type what i said"
# can fall back to the topic from the original message instead of being
# searched verbatim.
_last_research_topic = None


def _set_last_research_topic(topic):
    global _last_research_topic
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
            "search the web for ", "research about ", "research on ", "research ",
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
    low = low.strip(" ,.!?")
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
            "If this is a spoken conversation, sound natural and answer in one or two short sentences unless more detail is requested."
        )

    if voice_compact:
        system_prompt += (
            " Prioritize a fast spoken reply over a detailed one."
            " Keep it natural, under 35 words, and within two short sentences unless detail is requested."
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
            {"role": "user", "content": f"{user_message}\n\nSearch info: {search_info}"},
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
        ]
    return {
        "path": "llm",
        "system_prompt": system_prompt,
        "query": user_message,
        "messages": messages,
    }


def _commit_chat(role, text):
    if text:
        add_message(role, text)


def _record_native_task_outcome(fallback_text=""):
    """F07/F09 — persist the NATIVE engine's terminal TaskResult.

    ``handle_task_message`` speaks a string, so the structured result used to
    be thrown away: a verified native run could never become a skill and the
    work log only saw the opencode engine. The agent keeps the last result;
    this records it (trace and verification included) exactly once per turn.
    """
    if memory_store is None:
        return
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
    try:
        memory_store.record_task_outcome(
            "task_agent", status, task_text or fallback_text,
            summary=(getattr(result, "summary", "")
                     or getattr(result, "detail", "") or "")[:200],
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
            for delta in ask_openai_compat_stream(
                messages,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                cancel=cancel,
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
        return reply
    # suspension/error: hand back whatever was produced, or nothing at all.
    return reply


class _ChatRacer:
    """Race a *pure* chat speculation against the intent router.

    F25: the speculation does nothing but generate an answer. It performs no
    search and has no other external effect, it builds from an immutable
    context snapshot taken once at construction, its queue is bounded, and
    :meth:`cancel` closes the underlying HTTP response instead of merely
    setting a flag that is only noticed when the next delta arrives.
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
        self._cancel = threading.Event()
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
                msgs.append({"role": "user", "content": self._msg})
            temp = 0.45 if self._voice_compact else 0.7
            mx = 300 if self._voice_compact else 1400
            gen = _stream_chat_deltas(built["messages"], temp, mx,
                                      cancel=self._cancel)
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
        def _drain():
            while True:
                item = self._queue.get()
                if item is self._sentinel:
                    break
                yield item
        return _drain()

    def cancel(self):
        """Stop the speculation now, not at the next delta.

        F25: sets the shared cancellation event — every provider client polls
        it between chunks and closes its HTTP response when it fires — and
        unblocks a consumer parked on :meth:`adopt` so an abandoned race is
        released immediately instead of waiting for the socket to drain.
        """
        self._cancelled = True
        self._cancel.set()
        self._put_sentinel()

    @property
    def is_done(self):
        return self._done.is_set()

    def join(self, timeout=None):
        self._thread.join(timeout=timeout)
        return self.is_done

    def has_stream(self):
        return self._has_stream


def handle_chat(user_message, voice_compact=False, commit_response=True, stream=None, prebuilt=None, live_stream=None):
    """Handle a chat message.

    If *stream* is a callable(delta_text), the reply is delivered token by
    token as it is generated (live typewriter feel) instead of waiting for
    the full response. *_commit_response* controls whether the final text is
    stored to conversation memory.
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

    built = prebuilt if prebuilt is not None else _build_chat_messages(user_message, voice_compact=voice_compact)

    if built["path"] == "needs_search":
        # F25 — the speculative build refused to search (it may still be
        # cancelled). We are now the selected route, so acquire the lookup
        # here and rebuild against the committed history.
        built = _build_chat_messages(user_message, voice_compact=voice_compact)

    if built["path"] == "browser_search":
        execute_multiple([{"action": "search", "input": user_message}])
        response = "Sir, I couldn't access that information directly, so I've opened a search for you."
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

    if not content:
        content = "I'm having trouble connecting. Please try again."
    # F26 — the uncertainty→clarification decision must happen BEFORE speech.
    # With a stream consumer attached every delta has already been spoken, so
    # swapping the reply here would speak one answer and store a different
    # permission question. When the stream produced nothing, nothing has been
    # spoken yet and the rewrite is still safe.
    if stream is not None and pieces:
        if re.search(_UNSURE_RE, content):
            print("[CHAT] Answer looked unsure — already spoken, keeping it verbatim.")
    elif re.search(_UNSURE_RE, content):
        print("[CHAT] Answer looked unsure — asking before researching.")
        if maybe_proactive_research(user_message):
            content = _confirmation_question()
    if commit_response:
        _commit_chat("assistant", content)
    print("[CHAT] Reply:", content)
    # G9 (F07): the chat exchange lands in the work-event store (bounded,
    # masked) — cross-restart continuity beyond the 20-message window.
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



def generate_command_response(actions):
    play_lines = [
        "On it, sir. Playing {song}.",
        "Right away, sir. Enjoy {song}.",
        "Consider it done. Now playing {song}.",
    ]
    open_lines = [
        "Opening {site}, sir.",
        "Launching {site}.",
        "Accessing {site}.",
    ]
    search_lines = [
        "Searching for {query}, sir.",
        "Looking that up now.",
        "Opening search results for {query}.",
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

    Returns an immediate ack text ("On it, sir."). If execution fails, the
    opencode handoff is armed for spoken confirmation and the real result is
    delivered async via the async-reply callback after the user confirms.
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
    suffix = "Do you want me to go ahead and execute it?"
    task_text = task_description or original_message
    prefix = "Sir, this is what I understood — "
    question = f"{prefix}{task_text}. {suffix}"
    if voice_compact and len(question) > 180:
        # Never truncate the trailing question: budget the task text only.
        budget = 180 - len(prefix) - len("... ") - len(suffix)
        question = f"{prefix}{_truncate_at_word(task_text, budget)}... {suffix}"
    return question


# ── Dual-voice mute: while an opencode task runs, only opencode speaks ──
_opencode_task_running = False

# Whether a research/quick-search is currently running in the background
# (handle_research_intent daemon thread).  Used to gate the stop-research
# event so an idle stop utterance never poisons the next search.
_research_running = False

# Mandatory announcement made exactly at handoff — exempt from the mute so
# the UI path never swallows it (the flag is already True by then).
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
    _opencode_task_running = bool(running)


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
            _notify_async_reply("I started the task, sir, but my agent hit a snag.")
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
)
_STOP_RESEARCH_NEGATION_RE = re.compile(
    r"(?:dont|don't|do not|never)\s+(?:.{0,30})?(?:"
    + "|".join(re.escape(p) for p in STOP_RESEARCH_PHRASES)
    + r")"
)


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
    if deep:
        ack_chosen = random.choice([
            "On it, sir. Running a deepsearch now.",
            "Deepsearch it is, sir. Digging through the sites now.",
        ])
        ack = (ack_chosen +
               " The full report is coming up on your screen — and I'll sum it up for you when it's ready.")
    else:
        ack_chosen = random.choice([
            "On it, sir. Quick look coming up.",
            "Let me check that on Google, sir.",
        ])
        ack = ack_chosen + " Back in a moment with the short version."

    if voice_compact and len(ack) > 110:
        ack = ("Deepsearch running, sir — full report on your screen, "
               "summary when it's ready." if deep
               else "Quick lookup running, sir.")

    def _run():
        global _research_running
        try:
            try:
                query = research_query if derived else derive_research_query(research_query)
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


def _arm_confirmation(word):
    with _confirmation_lock:
        global _pending_confirmation
        _pending_confirmation = {
            "message": word,
            "expires": time.time() + CONFIRM_WINDOW_SECONDS,
        }


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
        print("[RESEARCH] User declined research (explicit negative, no model consult).")
        return "As you wish, sir. I'll just answer from what I know."
    if verdict == "yes":
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
    # and block re-asking for a window so we don't nag them repeatedly.
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

    Returns the spoken reply when the user confirms or declines, else None
    when nothing is pending, the window lapsed, or the answer is unclear
    (pending is discarded so the message falls through to normal chat).
    """
    with _opencode_confirm_lock:
        global _pending_opencode_task
        pending, _pending_opencode_task = _pending_opencode_task, None
    if not pending:
        return None
    if time.time() > pending["expires"]:
        print("[TASK] opencode confirmation window expired — skipped.")
        return None
    if _TASK_CONFIRM_NO_RE.search(answer):
        print("[TASK] User declined the opencode handoff.")
        return "As you wish, sir. I will skip that."
    if not _TASK_CONFIRM_YES_RE.search(answer):
        print("[TASK] opencode confirmation answered with something else — skipped.")
        return None
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
        _unbind_turn_job(token)


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
        try:
            memory_reply = memory_store.handle_memory_phrase(msg)
        except Exception as exc:
            logging.debug("[MEMORY] phrase op failed: %s", exc)
        if memory_reply is not None:
            if from_voice and sync_voice:
                sync_voice_log(voice_log_message, memory_reply)
            return memory_reply

    # ── Explicit websearch stop — before any other routing ──
    if is_stop_research(msg):
        print("[RESEARCH] Explicit stop request")
        response = handle_stop_research_request(from_voice=from_voice)
        if from_voice and sync_voice:
            sync_voice_log(voice_log_message, response)
        return response

    # ── Pending 'shall I look it up?' answer — consume before any routing ──
    confirmed = _consume_confirmation(msg)
    if confirmed is not None:
        if from_voice and sync_voice:
            sync_voice_log(voice_log_message, confirmed)
        return confirmed

    # ── Pending task-action confirmation answer — consume before routing ──
    task_confirmed = consume_task_confirmation(msg)
    if task_confirmed is not None:
        _record_native_task_outcome(msg)
        _clear_browser_clarification()
        if from_voice and sync_voice:
            sync_voice_log(voice_log_message, task_confirmed)
        return task_confirmed

    # ── Pending opencode-handoff confirmation answer — consume before routing ──
    opencode_confirmed = _consume_opencode_confirmation(msg)
    if opencode_confirmed is not None:
        _clear_browser_clarification()
        if from_voice and sync_voice:
            sync_voice_log(voice_log_message, opencode_confirmed)
        return opencode_confirmed

    # ── Pending browser clarification follow-up — continue same task ──
    browser_followup = _consume_browser_followup(msg)
    if browser_followup is not None:
        if from_voice and sync_voice:
            sync_voice_log(voice_log_message, browser_followup)
        return browser_followup

    # ── Screen Q&A — "what's on my screen?" ──
    is_explicit_command = msg.lower().startswith("command")
    if not is_explicit_command and is_explicit_task_request(msg):
        response = handle_task_message(msg, voice_compact=voice_compact)
        _record_native_task_outcome(msg)
        _disarm_other_gates_if_task_gate_armed()
        if from_voice and sync_voice:
            sync_voice_log(voice_log_message, response)
        return response

    # ── Native code tools — plain read/write/run requests route here before
    # the LLM intent router can send them to opencode. ──
    if not is_explicit_command and is_code_tool_request(msg):
        response = handle_task_message(msg, voice_compact=voice_compact)
        _record_native_task_outcome(msg)
        _disarm_other_gates_if_task_gate_armed()
        if from_voice and sync_voice:
            sync_voice_log(voice_log_message, response)
        return response

    if not msg.lower().startswith("command"):
        screen_control_response = maybe_handle_screen_control_message(msg)
        if screen_control_response is not None:
            _clear_browser_clarification()
            if from_voice and sync_voice:
                sync_voice_log(voice_log_message, screen_control_response)
            return screen_control_response

    # ── Deep research — explicit phrases ("look this up", "find out about X") ──
    if not msg.lower().startswith("command") and force_research(msg):
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
    racer = None
    if not msg.lower().startswith("command"):
        # ── G8 orchestrator migration (F02) — behind the mode flag ──
        # F02: the ROUTE is selected before any work starts. Speculative chat
        # used to begin first, so an orchestrator-owned goal paid for a chat
        # stream that was thrown away — and on fallback the speculative work
        # had already started, which is exactly the "repeats started work"
        # failure. Speculation now runs only on the legacy chat route.
        route = orchestrator_select_route(msg, screen_question=is_screen_question(msg))
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
        elif stream_reply is not None:
            try:
                racer = _ChatRacer(msg, voice_compact)
            except Exception as exc:
                logging.warning("[CHAT] Racer start failed: %s", exc)
                racer = None
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
            # [PERF] The fast path's whole point: this mark is ~0ms here, where
            # it used to be the classifier's full round trip.
            _mark_latency(request_id, "classify", 0.0)
            print("[INTENT] Deterministic chat fast path (classifier skipped)")
        else:
            _classify_started = time.monotonic()
            intent = classify_intent(msg, timeout_ms=INTENT_BUDGET_VOICE_MS
                                     if from_voice else INTENT_BUDGET_MS)
            # [PERF] Record the classifier's own cost against this turn.
            _mark_latency_duration(request_id, "classify", _classify_started)
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
                evidence = result.get("evidence", [])
                topic = result.get("topic", "")
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
            description = intent.get("task_description") or msg
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
        response = handle_task_message(msg, voice_compact=voice_compact)
        _record_native_task_outcome(msg)
        _disarm_other_gates_if_task_gate_armed()
        if from_voice and sync_voice:
            sync_voice_log(voice_log_message, response)
        return response

    if not msg.lower().startswith("command"):
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
        else:
            response = handle_chat(
                msg,
                voice_compact=voice_compact,
                commit_response=commit_response,
                stream=stream_reply,
            )
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
