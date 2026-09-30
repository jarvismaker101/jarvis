"""Google Gemini client — vision (screen grounding) and the Jarvis text brain.

Vision uses ``gemini-3.5-flash-lite`` (fast, free tier, excellent at GUI
understanding) as the primary vision provider for screen Q&A and action
plans, with Groq (Qwen 3.6 27B) as the fallback.  Returns responses in the
same OpenAI-shaped format so the cascade code can use it as a drop-in.

Supports optional Google Search grounding for enriched answers with real
source URLs (requires Google AI Pro or equivalent API access).
"""

import json
import logging
import os
import re
import time

import requests
from requests.adapters import HTTPAdapter

from backend.config import GEMINI_API_KEY
from backend.core.deadline import (
    BudgetedRetry,
    bound,
    classify_exception,
    resolve,
    retry_eligible,
)
# F31: the one typed channel contract (final vs reasoning) every streaming
# adapter speaks. Gemini marks thinking parts with ``"thought": true``; they
# are preserved in the reasoning channel and can never be narrated as answer.
from backend.services.openai_compat_client import (
    FINAL_CHANNEL,
    REASONING_CHANNEL,
    StreamDelta,
    _mark_headers,
)


GEMINI_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-3.5-flash-lite")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_BRAIN_MODEL", "gemini-3.5-flash-lite")

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

# Standing directive: keys NEVER land in logs. urllib3 retry warnings and
# manual-loop exception logs otherwise print the full request URL including
# the ?key=... query param.
_REDACT_RE = re.compile(r"key=[A-Za-z0-9_\-]+")


def _redact(text):
    """Strip API keys out of any log line — 'key=<value>' -> 'key=<redacted>'."""
    return _REDACT_RE.sub("key=<redacted>", str(text))


# ── F31: Gemini thought parts are a separate channel ────────────────────────
def _is_thought_part(part):
    """True when a Gemini content part is the model's thinking.

    Gemini's thinking-capable models mark reasoning parts with
    ``"thought": true``. Such a part is NEVER answer text: it must not be
    concatenated into a chat reply, spoken, or written to memory.
    """
    return bool(isinstance(part, dict) and part.get("thought"))


def _final_parts_text(parts):
    """Concatenate only the FINAL-answer text parts of one Gemini candidate."""
    return "".join(
        part.get("text", "")
        for part in (parts or ())
        if isinstance(part, dict) and "text" in part and not _is_thought_part(part)
    )


def _reasoning_parts_text(parts):
    """Concatenate the thinking-channel text parts (preserved, never spoken)."""
    return "".join(
        part.get("text", "")
        for part in (parts or ())
        if isinstance(part, dict) and "text" in part and _is_thought_part(part)
    )


# urllib3 logs 'Retrying ... /v1beta/...?key=<actual key>' at WARNING level —
# silence that logger entirely so no key can ever leak through it.
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

# Persistent session with automatic retries on low-level connection errors.
# This avoids creating a new TCP connection for every call and handles
# transient WiFi / socket issues that are common on Windows.
#
# F24: the strategy is deadline-aware. Hidden adapter retries used to sleep
# and re-POST with no knowledge of the caller's budget; BudgetedRetry refuses
# a retry whose budget is spent and caps every backoff to the time left.
_MAX_RETRIES = 2
_session = requests.Session()
_retry_strategy = BudgetedRetry(
    total=_MAX_RETRIES,
    backoff_factor=0.5,          # 0s, 0.5s between retries
    status_forcelist=[429, 502, 503, 504],
    allowed_methods=["POST"],    # retries for POST requests
)
_session.mount("https://", HTTPAdapter(max_retries=_retry_strategy))


def is_available():
    """Return True if a Gemini API key is configured."""
    return bool(GEMINI_API_KEY)


def ask_gemini_vision(
    prompt,
    image_data_url,
    max_completion_tokens=800,
    response_format=None,
    use_google_search=False,
    model=None,
    deadline=None,
):
    """Send a vision request to Gemini and return an OpenAI-shaped response.

    *image_data_url* is a ``data:image/png;base64,...`` string.  The prefix
    is stripped automatically to extract the raw base64 payload.

    When *use_google_search* is True the request includes the
    ``google_search`` tool so Gemini can ground its answer with live web
    results.  Grounding links are returned under the ``grounding_links``
    key in the response dict.

    *deadline* (F24) is the shared budget handle: a
    :class:`backend.core.deadline.Deadline`, an absolute ``time.monotonic()``
    timestamp, or a ``jobs.JobToken``.  A spent budget sends NO request at
    all, and every attempt gets whatever slice of the window is left.
    """
    if not GEMINI_API_KEY:
        return {}

    handle = resolve(deadline)

    # Strip the data-URL prefix to get raw base64.
    if "," in image_data_url:
        raw_b64 = image_data_url.split(",", 1)[1]
    else:
        raw_b64 = image_data_url

    # Detect mime type from the data URL header.
    if image_data_url.startswith("data:image/png"):
        mime = "image/png"
    elif image_data_url.startswith("data:image/jpeg"):
        mime = "image/jpeg"
    else:
        mime = "image/png"

    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": mime, "data": raw_b64}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": max_completion_tokens,
        },
    }

    # If caller wants JSON output, tell Gemini to produce JSON.
    # NOTE: responseMimeType is NOT set when google_search is active
    # because the grounding tool may conflict with forced JSON mode.
    if response_format and response_format.get("type") == "json_object":
        if not use_google_search:
            body["generationConfig"]["responseMimeType"] = "application/json"

    # Google Search grounding — lets Gemini search the web and return
    # real source URLs alongside its answer.
    if use_google_search:
        body["tools"] = [{"google_search": {}}]

    url = _GEMINI_URL.format(model=model or GEMINI_MODEL)

    # Manual retry loop for connection-aborted / write-timeout errors that
    # sometimes slip past urllib3's retry strategy (e.g. mid-write TCP resets
    # common on Windows WiFi). *handle* (F24) bounds the whole loop: no
    # attempt is sent once the budget is spent, an attempt's own (connect,
    # read) timeout is sliced to what is left, and only transient failures
    # (classified centrally) are ever replayed.
    last_exc = None
    for attempt in range(_MAX_RETRIES + 1):
        if handle is not None and handle.stopped():
            break
        request_timeout = handle.timeout((8, 45)) if handle is not None else (8, 45)
        if request_timeout is None:
            break
        try:
            if handle is None:
                resp = _session.post(
                    url,
                    params={"key": GEMINI_API_KEY},
                    json=body,
                    timeout=request_timeout,
                )
            else:
                # Bind the budget to this thread so even the adapter's hidden
                # retries obey it.
                with bound(handle):
                    resp = _session.post(
                        url,
                        params={"key": GEMINI_API_KEY},
                        json=body,
                        timeout=request_timeout,
                    )

            if resp.status_code != 200:
                logging.warning(
                    "[GEMINI] %d: %s", resp.status_code, _redact(resp.text[:300])
                )
                return {}

            data = resp.json()

            # Check for API-level errors inside a 200 response.
            if "error" in data:
                logging.warning("[GEMINI] API error: %s", _redact(str(data["error"])[:300]))
                return {}

            # Convert Gemini response to OpenAI-compatible shape so the
            # cascade code doesn't need to know which provider answered.
            candidates = data.get("candidates", [])
            if not candidates:
                return {}

            candidate = candidates[0]

            parts = candidate.get("content", {}).get("parts", [])
            # F31: only final-answer parts become the reply text; thinking
            # parts stay in the reasoning channel and are never returned as
            # the answer (they would otherwise become chat text / memory).
            text = _final_parts_text(parts)

            if not text.strip():
                return {}

            # Extract grounding links from Google Search metadata.
            grounding_links = []
            grounding = candidate.get("groundingMetadata", {})
            for chunk in grounding.get("groundingChunks", []):
                web = chunk.get("web", {})
                if web.get("uri"):
                    grounding_links.append({
                        "url": web["uri"],
                        "title": web.get("title", "").strip(),
                    })

            return {
                "choices": [
                    {
                        "message": {"content": text},
                    }
                ],
                "model": model or GEMINI_MODEL,
                "grounding_links": grounding_links,
            }

        except Exception as exc:
            last_exc = exc
            failure = classify_exception(exc)
            if not retry_eligible(failure, handle, attempt + 1, _MAX_RETRIES + 1):
                if failure.retryable:
                    logging.warning(
                        "[GEMINI] Request failed after %d attempt(s), no budget to replay: %s",
                        attempt + 1, _redact(exc),
                    )
                else:
                    logging.warning("[GEMINI] Request error: %s", _redact(exc))
                break
            wait = 1.0 * (attempt + 1)
            logging.info(
                "[GEMINI] Connection error (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1, _MAX_RETRIES + 1, wait, _redact(exc),
            )
            if handle is not None:
                if not handle.sleep(wait):
                    break
            else:
                time.sleep(wait)
            continue

    # Should not reach here, but just in case.
    if last_exc is not None:
        logging.warning("[GEMINI] Exhausted retries: %s", _redact(last_exc))
    return {}


# ---------------------------------------------------------------------------
# Text chat (Jarvis brain) — gemini-3.5-flash-lite
# ---------------------------------------------------------------------------

def _build_chat_body(messages, temperature, max_tokens):
    """Convert OpenAI-style messages to a Gemini generateContent body.

    The first ``system`` message becomes ``systemInstruction``; user and
    assistant messages become ``contents`` with Gemini role names.
    """
    system_text = ""
    contents = []
    for msg in messages or []:
        role = msg.get("role", "user")
        text = msg.get("content", "") or ""
        if role == "system":
            system_text += text + "\n"
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": text}]})

    body = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
        },
    }
    if system_text.strip():
        body["systemInstruction"] = {"parts": [{"text": system_text.strip()}]}
    if max_tokens is not None:
        body["generationConfig"]["maxOutputTokens"] = max_tokens
    return body


def ask_gemini_chat(messages, temperature=0.7, max_tokens=None, model=None, timeout=None, no_retry=False, deadline=None):
    """Send a text chat to Gemini and return an OpenAI-shaped response.

    *timeout* overrides the default (connect=8s, read=45s) when provided.
    *no_retry* uses a single plain POST (no _session retry, no manual loop)
    so the caller gets a fast fail — ideal for the intent classifier.
    *deadline* (F24) is the shared budget handle (Deadline / absolute
    monotonic timestamp / JobToken): a spent budget sends no request, an
    explicit timeout is sliced to what is left, and only transient failures
    are replayed.
    """
    if not GEMINI_API_KEY:
        return {}

    body = _build_chat_body(messages, temperature, max_tokens)
    url = _GEMINI_URL.format(model=model or GEMINI_CHAT_MODEL)
    handle = resolve(deadline)
    _timeout = handle.timeout(timeout or (8, 45)) if handle is not None else (timeout or (8, 45))
    if _timeout is None:
        return {}

    if no_retry:
        try:
            resp = requests.post(
                url,
                params={"key": GEMINI_API_KEY},
                json=body,
                timeout=_timeout,
            )
        except Exception:
            return {}
        if resp.status_code != 200:
            return {}
        try:
            data = resp.json()
        except Exception:
            return {}
        if "error" in data:
            return {}
        candidates = data.get("candidates", [])
        if not candidates:
            return {}
        parts = candidates[0].get("content", {}).get("parts", [])
        # F31: a thought part is not answer text (never chat/memory/TTS).
        text = _final_parts_text(parts)
        if not text.strip():
            return {}
        return {
            "choices": [{"message": {"content": text}}],
            "model": model or GEMINI_CHAT_MODEL,
        }

    for attempt in range(_MAX_RETRIES + 1):
        if handle is not None and handle.stopped():
            break
        attempt_timeout = handle.timeout(timeout or (8, 45)) if handle is not None else _timeout
        if attempt_timeout is None:
            break
        try:
            if handle is None:
                resp = _session.post(
                    url,
                    params={"key": GEMINI_API_KEY},
                    json=body,
                    timeout=attempt_timeout,
                )
            else:
                with bound(handle):
                    resp = _session.post(
                        url,
                        params={"key": GEMINI_API_KEY},
                        json=body,
                        timeout=attempt_timeout,
                    )

            if resp.status_code != 200:
                logging.warning(
                    "[GEMINI CHAT] %d: %s", resp.status_code, _redact(resp.text[:300])
                )
                return {}

            data = resp.json()
            if "error" in data:
                logging.warning("[GEMINI CHAT] API error: %s", _redact(str(data["error"])[:300]))
                return {}

            candidates = data.get("candidates", [])
            if not candidates:
                return {}

            parts = candidates[0].get("content", {}).get("parts", [])
            # F31: final-answer parts only — thinking parts never become the
            # reply that is spoken or written to conversational memory.
            text = _final_parts_text(parts)

            if not text.strip():
                return {}

            return {
                "choices": [{"message": {"content": text}}],
                "model": model or GEMINI_CHAT_MODEL,
            }

        except Exception as exc:
            failure = classify_exception(exc)
            if not retry_eligible(failure, handle, attempt + 1, _MAX_RETRIES + 1):
                logging.warning("[GEMINI CHAT] Request error: %s", _redact(exc))
                break
            wait = 1.0 * (attempt + 1)
            if handle is not None:
                if not handle.sleep(wait):
                    break
            else:
                time.sleep(wait)
            continue

    return {}


def ask_gemini_chat_stream(messages, temperature=0.7, max_tokens=None, model=None, cancel=None, deadline=None, include_reasoning=False, typed=False):
    """Stream Gemini chat deltas (SSE) so the UI gets a live typewriter reply.

    *cancel* is an optional ``threading.Event`` (F25): when it is set the
    loop stops reading and the HTTP response is closed immediately.

    *deadline* (F24) is the shared budget handle.  A spent budget sends no
    request, the request's own read timeout is sliced to the time left, and
    the loop stops between chunks — so a trickling stream cannot extend the
    deadline.  Text already yielded stays yielded: partial results survive.

    F31 channels: parts marked ``"thought": true`` are the model's thinking,
    never the answer. The default stream yields only final-answer text as
    plain strings (thinking is excluded, so a mixed thought/final response
    can never narrate reasoning). Pass *include_reasoning* (or *typed*) to
    receive :class:`StreamDelta` events instead, with thinking parts tagged
    ``reasoning`` in their own preserved channel.
    """
    typed_output = bool(typed or include_reasoning)
    if not GEMINI_API_KEY:
        return

    handle = resolve(deadline)
    if handle is not None and handle.stopped():
        return

    body = _build_chat_body(messages, temperature, max_tokens)
    m = model or GEMINI_CHAT_MODEL
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{m}:streamGenerateContent"
    )
    request_timeout = handle.timeout((8, 18)) if handle is not None else (8, 18)
    if request_timeout is None:
        return

    try:
        if handle is not None:
            with bound(handle):
                resp = _session.post(
                    url,
                    params={"key": GEMINI_API_KEY, "alt": "sse"},
                    json=body,
                    stream=True,
                    timeout=request_timeout,
                )
        else:
            resp = _session.post(
                url,
                params={"key": GEMINI_API_KEY, "alt": "sse"},
                json=body,
                stream=True,
                timeout=request_timeout,
            )
    except (requests.ConnectionError, requests.Timeout, OSError) as exc:
        logging.warning("[GEMINI CHAT] Stream request error: %s", _redact(exc))
        return

    if resp.status_code != 200:
        logging.warning(
            "[GEMINI CHAT] Stream error: %d %s", resp.status_code, _redact(resp.text[:300])
        )
        return

    # [PERF] P1-19 — response headers in: from here on the delay is generation.
    _mark_headers("gemini", m)

    try:
        for raw_line in resp.iter_lines(decode_unicode=True):
            if cancel is not None and cancel.is_set():
                break
            # F24: checked between chunks, before the next chunk is
            # consumed — a stream that trickles in forever still ends here.
            if handle is not None and handle.stopped():
                break
            if not raw_line:
                continue
            line = raw_line.strip()
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
            else:
                payload = line
            if not payload or payload == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except Exception:
                continue
            candidates = chunk.get("candidates") or []
            if not candidates:
                continue
            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if not text:
                    continue
                if _is_thought_part(part):
                    # F31: thinking is a separate channel — preserved, typed,
                    # and never narrated as the answer.
                    if typed_output:
                        yield StreamDelta(text, REASONING_CHANNEL)
                    continue
                yield (StreamDelta(text, FINAL_CHANNEL)
                       if typed_output else text)
    finally:
        # F25 — never leave the socket open behind a cancelled speculation.
        try:
            resp.close()
        except Exception:
            pass
