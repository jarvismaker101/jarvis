import os

import requests

from backend.config import ENV_PATH, FIREWORKS_API_KEY, FIREWORKS_MODEL
from backend.core.deadline import (
    BUDGET,
    Failure,
    bound,
    budget_allows,
    classify_exception,
    classify_status,
    failure_of,
    resolve,
    retry_eligible,
)
# F31: the one typed channel contract (final vs reasoning) every streaming
# adapter speaks, so reasoning text can never reach chat/TTS/memory as answer.
from backend.services.openai_compat_client import (
    FINAL_CHANNEL,
    REASONING_CHANNEL,
    StreamDelta,
    _mark_headers,
    reasoning_text as _reasoning_text,
)

API_KEY = FIREWORKS_API_KEY
DEFAULT_MODEL = FIREWORKS_MODEL
API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
REASONING_EFFORT = os.getenv("FIREWORKS_REASONING_EFFORT", "none")

#: Statuses worth one retry: rate limits, timeouts and server-side hiccups.
#: Auth failures (401/403), missing models (404) and other validation errors
#: are never replayed. F24: the list is derived from the shared classifier
#: (backend.core.deadline) so the two can never drift apart.
_TRANSIENT_STATUSES = frozenset(
    status for status in (408, 425, 429, 500, 502, 503, 504)
    if classify_status(status).retryable
)

#: One retry, never more: a manual loop on top of the classifier.
_MAX_ATTEMPTS = 2

print("LOADING ENV FROM:", ENV_PATH)
print("FIREWORKS API KEY FOUND:", API_KEY is not None)


def _exhausted(detail="budget exhausted before the request"):
    """The result of asking Fireworks with no budget left: no request sent."""
    failure = Failure(BUDGET, detail=detail)
    return {"error": "budget_exhausted", "detail": detail, "failure": failure}



def _mentions_reasoning(text):
    """True when an error body references the reasoning/thinking setting —
    the signature of a thinking-only model (e.g. GLM) rejecting
    reasoning_effort. Deliberately substring-based: no model-name list to
    maintain, and a false positive only costs one extra attempt.

    Broadened to also catch fireworks validation messages like
    'unsupported parameter', 'extra fields' or 'unknown field' that may
    accompany reasoning_effort rejections on some models.
    """
    lowered = (text or "").lower()
    return any(w in lowered for w in (
        "reasoning", "thinking", "effort",
        "unsupported", "extra fields", "extra field", "unknown field",
        "invalid reasoning", "invalid parameter",
    ))


def _post_chat_completion(data, timeout=(5.05, 60), deadline=None):
    """One HTTP attempt. *deadline* (F24) is bound for the call so a hidden
    adapter retry cannot outlive the caller's budget."""
    if not API_KEY:
        print("[FIREWORKS] API KEY MISSING")
        return {}

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        if deadline is not None:
            with bound(deadline):
                response = requests.post(API_URL, headers=headers, json=data, timeout=timeout)
        else:
            response = requests.post(API_URL, headers=headers, json=data, timeout=timeout)

        print("[FIREWORKS] Status:", response.status_code)

        if response.status_code != 200:
            print("[FIREWORKS] Error:", response.text[:500])
            detail = response.text[:500]
            return {
                "error": response.status_code,
                "detail": detail,
                "failure": classify_status(response.status_code, detail),
            }

        return response.json()
    except Exception as exc:
        print("REQUEST ERROR:", exc)
        detail = str(exc)[:200]
        return {"detail": detail, "failure": classify_exception(exc)}


def ask_fireworks(messages, temperature=0.7, max_tokens=None, model=None,
                  timeout=None, deadline=None):
    """Non-stream chat completion.

    *deadline* (F24) is the shared budget handle (a Deadline, an absolute
    monotonic timestamp, or a jobs.JobToken).  A spent budget sends no
    request; the attempt's own timeout is sliced to the time left; and a
    failure is replayed only when the shared classifier says it is
    transient AND the budget still allows it.  Auth, not-found and
    validation failures are returned as-is to be replayed by nobody.
    """
    data = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    if REASONING_EFFORT:
        data["reasoning_effort"] = REASONING_EFFORT

    handle = resolve(deadline)
    base_timeout = timeout or (5.05, 60)
    result = {}
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        attempt_timeout = handle.timeout(base_timeout) if handle is not None else base_timeout
        if attempt_timeout is None:
            return _exhausted()
        result = _post_chat_completion(data, timeout=attempt_timeout, deadline=handle)
        if isinstance(result, dict) and result.get("choices"):
            return result

        # The one deliberate replay: a model that rejects the reasoning
        # setting (e.g. a thinking-only GLM) is asked again without it — but
        # only while there is budget to spend.
        if (
            isinstance(result, dict)
            and result.get("error") == 400
            and "reasoning_effort" in data
            and _mentions_reasoning(result.get("detail"))
        ):
            print("[FIREWORKS] Retrying without reasoning_effort (model rejected the reasoning setting)")
            data.pop("reasoning_effort", None)
            if budget_allows(handle, attempt, _MAX_ATTEMPTS):
                continue
            return result

        failure = failure_of(result)
        if failure is not None and retry_eligible(failure, handle, attempt, _MAX_ATTEMPTS):
            print(f"[FIREWORKS] Transient HTTP {failure.status} — retrying once")
            continue
        return result
    return result


def ask_fireworks_vision(prompt, image_data_url, max_completion_tokens=800, model=None, response_format=None, deadline=None):
    """Vision request to Fireworks (openai-compatible image_url) — for screen Q&A.

    Uses the same chat completions endpoint with a user message containing
    text + image_url parts. Non-stream, temperature 0. Vision models like
    deepseek-v4-flash-vision-exp are supported.

    *deadline* (F24): a spent budget sends no request at all, and the
    request's own timeout is sliced to the time that is left.
    """
    if not API_KEY:
        print("[FIREWORKS VISION] API KEY MISSING")
        return {}
    handle = resolve(deadline)
    if handle is not None and handle.stopped():
        print("[FIREWORKS VISION] budget exhausted before the request")
        return {}
    request_timeout = handle.timeout((5, 30)) if handle is not None else (5, 30)
    if request_timeout is None:
        return {}
    effective_model = model or DEFAULT_MODEL
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ],
    }
    data = {
        "model": effective_model,
        "messages": [message],
        "temperature": 0.0,
        "max_tokens": max_completion_tokens,
    }
    if response_format is not None:
        data["response_format"] = response_format
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        if handle is not None:
            with bound(handle):
                resp = requests.post(API_URL, headers=headers, json=data, timeout=request_timeout)
        else:
            resp = requests.post(API_URL, headers=headers, json=data, timeout=request_timeout)
        if resp.status_code != 200:
            print(f"[FIREWORKS VISION] {effective_model} {resp.status_code} {resp.text[:300]}")
            return {}
        result = resp.json()
        if result.get("error"):
            print(f"[FIREWORKS VISION] API error: {result['error']}")
            return {}
        return result
    except Exception as exc:
        print("[FIREWORKS VISION] Request error:", exc)
        return {}


def ask_fireworks_stream(messages, temperature=0.7, max_tokens=None, model=None, include_reasoning=False, cancel=None, deadline=None, typed=False):
    """Stream chat-completion text deltas from Fireworks (SSE).

    Yields partial text chunks as they arrive so the UI can render a
    live typewriter reply instead of waiting for the full response.

    F31 channels: the default stream yields ONLY final-answer content as
    plain strings — reasoning deltas never reach chat, TTS or conversational
    memory, and "hel" + "lo" stays "hello" (the text is forwarded exactly as
    the model produced it, never re-joined or re-spaced). Pass
    *include_reasoning* (or *typed*) to receive :class:`StreamDelta` events
    instead, each tagged ``final`` or ``reasoning`` — reasoning is preserved
    verbatim in its own channel and can never be mistaken for an answer.

    *cancel* is an optional ``threading.Event`` (F25). When it is set the
    loop stops reading and the HTTP response is closed immediately, so a
    cancelled speculative stream releases its socket instead of running to
    completion.

    *deadline* (F24) is the shared budget handle. A spent budget sends no
    request, the read timeout is sliced to the time left, and the read loop
    stops between chunks — so a trickling stream cannot extend the deadline
    and the chunks already yielded survive as a partial result.
    """
    typed_output = bool(typed or include_reasoning)
    if not API_KEY:
        print("[FIREWORKS] API KEY MISSING (stream)")
        return

    handle = resolve(deadline)
    if handle is not None and handle.stopped():
        print("[FIREWORKS] budget exhausted before the request (stream)")
        return

    data = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    if REASONING_EFFORT:
        data["reasoning_effort"] = REASONING_EFFORT

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    request_timeout = handle.timeout((5.05, 120)) if handle is not None else (5.05, 120)
    if request_timeout is None:
        return
    try:
        if handle is not None:
            with bound(handle):
                response = requests.post(
                    API_URL,
                    headers=headers,
                    json=data,
                    stream=True,
                    timeout=request_timeout,
                )
        else:
            response = requests.post(
                API_URL,
                headers=headers,
                json=data,
                stream=True,
                timeout=request_timeout,
            )
    except Exception as exc:
        print("[FIREWORKS] Stream request error:", exc)
        return

    if response.status_code != 200:
        print("[FIREWORKS] Stream error:", response.status_code, response.text[:300])
        # Thinking-only models (e.g. GLM) 400 on reasoning_effort; retry
        # ONCE without it so they answer with their default thinking
        # behavior instead of erroring into the Gemini fallback. F24: the
        # replay happens only while there is budget left to spend on it.
        if (
            response.status_code == 400
            and "reasoning_effort" in data
            and _mentions_reasoning(response.text)
            and budget_allows(handle, 1, _MAX_ATTEMPTS)
        ):
            print("[FIREWORKS] Retrying stream without reasoning_effort (model rejected the reasoning setting)")
            data.pop("reasoning_effort", None)
            retry_timeout = handle.timeout((5.05, 120)) if handle is not None else (5.05, 120)
            if retry_timeout is None:
                return
            try:
                if handle is not None:
                    with bound(handle):
                        response = requests.post(
                            API_URL,
                            headers=headers,
                            json=data,
                            stream=True,
                            timeout=retry_timeout,
                        )
                else:
                    response = requests.post(
                        API_URL,
                        headers=headers,
                        json=data,
                        stream=True,
                        timeout=retry_timeout,
                    )
            except Exception as exc:
                print("[FIREWORKS] Stream request error:", exc)
                return
            if response.status_code != 200:
                print("[FIREWORKS] Stream error:", response.status_code, response.text[:300])
                return
        else:
            return

    # [PERF] P1-19 — response headers in: from here on the delay is generation.
    # After the optional reasoning_effort replay, so it marks the response the
    # stream actually reads.
    _mark_headers("fireworks", data.get("model"))

    try:
        for line in response.iter_lines(decode_unicode=True):
            if cancel is not None and cancel.is_set():
                break
            # F24: checked between chunks — a trickling stream cannot extend
            # the deadline, and what was already yielded stays delivered.
            if handle is not None and handle.stopped():
                break
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if not payload or payload == "[DONE]":
                break
            try:
                import json
                chunk = json.loads(payload)
            except Exception:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                # F31: exactly the model's text — no re-joining, no
                # re-spacing — so 'hel' + 'lo' remains 'hello'.
                yield (StreamDelta(content, FINAL_CHANNEL)
                       if typed_output else content)
            if include_reasoning or typed:
                # F31: the thinking channel is always typed. It can be read
                # (typed consumers), but it can never be appended to answer
                # text by a consumer that only understands strings.
                reasoning = _reasoning_text(delta)
                if reasoning:
                    yield StreamDelta(reasoning, REASONING_CHANNEL)
    finally:
        # F25 — never leave the socket open behind a cancelled speculation.
        try:
            response.close()
        except Exception:
            pass
