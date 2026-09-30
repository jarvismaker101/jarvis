"""OpenAI-compatible chat client for user-added providers.

Custom providers added from the UI (OpenRouter, Groq, any /v1-compatible
gateway) speak the OpenAI /chat/completions dialect. Responses are shaped
exactly like fireworks_client's ({"choices": [...]} on success, {} on
failure) so brain.py can treat this as a drop-in member of the fallback
chain. The API key travels in the Authorization header only.
"""

import json
import logging

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend.core.deadline import resolve

# Persistent session with automatic retries on low-level connection errors
# (same pattern as gemini_client / fireworks_client).
_MAX_RETRIES = 2

# ── F31: typed answer channels ───────────────────────────────────────────────
# Every streaming adapter speaks ONE channel contract, defined here once:
#
#   * ``FINAL_CHANNEL`` text is answer text. Only it may reach chat, TTS or
#     conversational memory.
#   * ``REASONING_CHANNEL`` text is the model's thinking. It is preserved
#     verbatim (never discarded) but it is NEVER answer text.
#
# A default stream yields plain ``str`` final-answer deltas — the contract
# every existing consumer relies on, so "hel" + "lo" still concatenates to
# "hello". The moment a caller asks for the reasoning channel
# (``include_reasoning=True``) or for typed output (``typed=True``), the
# stream yields :class:`StreamDelta` objects instead: a reasoning delta can
# then never be mistaken for — or spliced into — answer text by a consumer
# that could not otherwise tell the two apart.
FINAL_CHANNEL = "final"
REASONING_CHANNEL = "reasoning"

#: Delta field names that carry reasoning/thinking text in the wild
#: (Fireworks ``reasoning_content``, OpenRouter ``reasoning``, custom
#: gateways ``thinking``/``thought``).
REASONING_FIELDS = ("reasoning_content", "reasoning", "thinking", "thought")


class StreamDelta:
    """One typed streamed text delta: ``(channel, text)``.

    ``channel`` is :data:`FINAL_CHANNEL` or :data:`REASONING_CHANNEL`.
    """

    __slots__ = ("text", "channel")

    def __init__(self, text, channel=FINAL_CHANNEL):
        self.text = str(text or "")
        self.channel = channel

    @property
    def is_final(self):
        return self.channel == FINAL_CHANNEL

    def __eq__(self, other):
        if isinstance(other, StreamDelta):
            return (self.channel, self.text) == (other.channel, other.text)
        return NotImplemented

    def __hash__(self):
        return hash((self.channel, self.text))

    def __repr__(self):
        return "StreamDelta(%r, %r)" % (self.channel, self.text)


def reasoning_text(delta):
    """The reasoning text of one SSE delta object ('' when none).

    Reads the known reasoning field names, including list-valued shapes such
    as Gemini's ``[{"text": ...}]`` content blocks. Never raises.
    """
    if not isinstance(delta, dict):
        return ""
    parts = []
    for field in REASONING_FIELDS:
        value = delta.get(field)
        if isinstance(value, str) and value:
            parts.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)
    return "".join(parts)


def final_text(delta):
    """The final-answer text of one SSE delta object ('' when none).

    A content block explicitly marked as thinking is NOT answer text.
    """
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and not item.get("thought"):
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "".join(parts)
    return ""


def final_only(deltas):
    """Filter a (possibly typed) delta stream down to plain final-answer text.

    The one gate that keeps reasoning out of chat/TTS/memory for a consumer
    that can only handle text: plain strings pass through, typed
    ``StreamDelta`` events pass only when their channel is final.
    """
    for delta in deltas or ():
        if isinstance(delta, StreamDelta):
            if delta.is_final and delta.text:
                yield delta.text
        elif delta:
            yield delta


_session = requests.Session()
_retry_strategy = Retry(
    total=_MAX_RETRIES,
    backoff_factor=0.5,
    status_forcelist=[429, 502, 503, 504],
    allowed_methods=["POST"],
)
# P0-12: keepalive on the pool, so a socket held for the next turn is not
# dropped by a VPN / NAT in between (see backend/services/prewarm.py).
try:
    from backend.services.prewarm import KeepAliveAdapter as _KeepAliveAdapter

    _session.mount("https://", _KeepAliveAdapter(max_retries=_retry_strategy))
except Exception:  # pragma: no cover - never lose the plain adapter
    _session.mount("https://", HTTPAdapter(max_retries=_retry_strategy))


def _mark_headers(provider, model=None):
    """[PERF] P1-19 — mark the model provider's HTTP response boundary.

    The split that matters on a chat turn: everything before this mark is
    connect + queue + time-to-headers, everything after it is generation. One
    "time to first token" number cannot tell a slow provider apart from a slow
    prompt. It lives in this module because every streaming adapter
    (openai-compat, Gemini, Fireworks) already imports from here. Telemetry
    only: it swallows its own errors and marks nothing when no turn is active.
    """
    try:
        from backend.services import latency as _lat
        _lat.mark_provider_headers(provider, model)
    except Exception:
        pass


def _chat_url(base_url):
    return str(base_url or "").rstrip("/") + "/chat/completions"


# ── P1-13 — streaming budgets, cancellation and terminal state ──────────────
# The stream used to be unbounded and uninterruptible: a provider that accepted
# the connection and then went silent left the read blocking forever (holding
# the thread), a `requests` exception escaped the generator as a bare
# traceback, and the consumer could not tell "finished" from "gave up".
#: Time-to-FIRST-token budget. Once tokens are flowing, a per-chunk IDLE
#: timeout is the right shape — a total cap would kill a long, healthy answer.
DEFAULT_FIRST_TOKEN_TIMEOUT = 4.0
DEFAULT_STREAM_IDLE_TIMEOUT = 20.0
STREAM_CONNECT_TIMEOUT = 5.05

#: Terminal states a streaming call can end in. "pending" is only ever seen if
#: a consumer inspects the holder before iterating.
STREAM_PENDING = "pending"
STREAM_COMPLETED = "completed"
STREAM_CANCELLED = "cancelled"
STREAM_ERRORED = "errored"
STREAM_DEADLINE = "deadline"


def new_stream_outcome():
    """[P1-13] The unambiguous terminal state of ONE streaming call.

    The generator's yielded values keep the schema they always had (plain text
    or :class:`StreamDelta`), so existing consumers are untouched. Pass this
    holder as ``outcome=`` to also learn HOW the stream ended — a cancelled
    stream, an errored stream and a completed stream are now distinguishable,
    which is what the turn manager needs to tell "finished" from "gave up".
    """
    return {
        "status": STREAM_PENDING,
        "error": "",
        "finish_reason": "",
        "chunks": 0,
        "first_token": False,
    }


def _finish_outcome(outcome, status, error="", finish_reason=""):
    if not isinstance(outcome, dict):
        return
    outcome["status"] = status
    if error:
        outcome["error"] = str(error)[:300]
    if finish_reason:
        outcome["finish_reason"] = str(finish_reason)


def _register_cancel_closer(cancel, closer):
    """Ask *cancel* to CLOSE *closer* when it fires. Returns the registered
    closer, or None when the handle cannot do it (a bare Event cannot).

    The P0-08 handles (a ``jobs.JobToken``, the chat racer, the voice turn
    manager) expose ``register_closer``; a plain ``threading.Event`` does not,
    and for those the bounded read timeout is what unblocks the loop.
    """
    register = getattr(cancel, "register_closer", None)
    if not callable(register) or closer is None:
        return None
    try:
        register(closer)
        return closer
    except Exception:
        return None


def _unregister_cancel_closer(cancel, closer):
    if closer is None:
        return
    unregister = getattr(cancel, "unregister_closer", None)
    if not callable(unregister):
        return
    try:
        unregister(closer)
    except Exception:
        pass


def _set_socket_timeout(response, seconds):
    """Best-effort: relax the per-read timeout once streaming has started.

    ``requests`` fixes the timeout for the whole response, so the tight
    first-token budget would otherwise stay in force for a long generation.
    Every known path to the socket is tried; if none works the request's own
    (already bounded) timeout stands, which is still not a hang. Never raises.
    """
    if not seconds or seconds <= 0:
        return False
    try:
        raw = getattr(response, "raw", None)
        sock = None
        connection = getattr(raw, "_connection", None)
        sock = getattr(connection, "sock", None)
        if sock is None:
            fp = getattr(raw, "_fp", None)
            sock = getattr(fp, "fp", None)
            raw_sock = getattr(fp, "raw", None)
            sock = getattr(raw_sock, "_sock", None) or sock
        if sock is None:
            return False
        sock.settimeout(seconds)
        return True
    except Exception:
        return False


def _headers(api_key):
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def ask_openai_compat(
    messages, model, base_url, api_key, temperature=0.7, max_tokens=None,
    tools=None, tool_choice=None, reasoning_effort=None, timeout=None,
    response_format=None,
):
    """Non-stream chat completion; {} on any failure (fireworks pattern).

    *tools*/*tool_choice* (G8/F02): optional native tool-use definitions
    (OpenAI dialect). They ride on the same wire shape; when tools are given
    the response may carry ``tool_calls`` instead of plain content.

    *reasoning_effort* (F49): sent only when the caller's validated model
    snapshot says the model takes it — a model that runs its own default
    reasoning must never receive the field.

    *response_format*: optional structured-output directive (OpenAI dialect,
    e.g. ``{"type": "json_object"}``). Sent only when the caller passes one,
    so gateways that reject the field never see it.

    *timeout*: optional (connect, read) override — latency-critical callers
    (the intent classifier) pass a tight slice of their own deadline.
    """
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    if tools:
        data["tools"] = tools
        if tool_choice is not None:
            data["tool_choice"] = tool_choice
    if reasoning_effort:
        data["reasoning_effort"] = reasoning_effort
    if response_format:
        data["response_format"] = response_format
    try:
        response = _session.post(
            _chat_url(base_url),
            headers=_headers(api_key),
            json=data,
            timeout=timeout or (5.05, 60),
        )
    except Exception as exc:
        print("[OPENAI-COMPAT] Request error:", exc)
        return {}
    if response.status_code != 200:
        print("[OPENAI-COMPAT] Error:", response.status_code, response.text[:300])
        return {}
    try:
        return response.json()
    except Exception:
        return {}


def ask_openai_compat_vision(
    prompt, image_data_url, model, base_url, api_key,
    max_completion_tokens=None, response_format=None,
):
    """One non-stream vision completion over an OpenAI-compatible gateway.

    The screenshot rides the standard ``image_url`` content part; the return
    is the raw OpenAI-shaped result dict (``{}`` on any failure) — the same
    shape every dedicated vision adapter returns, so the F37 vision cascade
    can treat a user-added provider as one more dispatcher instead of a
    special case. The API key never leaves this call: errors from
    ask_openai_compat are already key-safe.
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": str(prompt or "")},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ],
    }]
    return ask_openai_compat(
        messages,
        model,
        base_url,
        api_key,
        temperature=0.2,
        max_tokens=max_completion_tokens,
        response_format=response_format,
    )


def ask_openai_compat_stream(
    messages, model, base_url, api_key, temperature=0.7, max_tokens=None,
    cancel=None, include_reasoning=False, typed=False, timeout=None,
    deadline=None, first_token_timeout=DEFAULT_FIRST_TOKEN_TIMEOUT,
    idle_timeout=DEFAULT_STREAM_IDLE_TIMEOUT, outcome=None,
):
    """Stream chat-completion text deltas (SSE) — mirrors fireworks_client.

    *cancel* is an optional cancellation handle (F25): when it is set the loop
    stops reading. [P1-13] A handle that supports ``register_closer`` (the F20
    job token, the chat racer, the voice turn manager) is ALSO given the HTTP
    response, so cancelling closes the socket and a read that is already in
    progress raises instead of holding the thread until it times out.

    *timeout* overrides the ``(connect, read)`` budget. By default the READ
    budget is the time-to-FIRST-token budget (:data:`DEFAULT_FIRST_TOKEN_TIMEOUT`,
    about 4s) and is relaxed to the per-chunk IDLE budget
    (:data:`DEFAULT_STREAM_IDLE_TIMEOUT`) once tokens are flowing — a total cap
    would kill a long, healthy answer, while no cap at all lets a silent
    provider hold the thread forever.

    *deadline* (F24) is the shared budget handle: a spent budget sends no
    request, the read timeout is sliced to the time left, and the loop stops
    between chunks. *outcome* (P1-13) is an optional
    :func:`new_stream_outcome` holder; it is how a consumer learns whether the
    stream completed, was cancelled, errored or ran out of budget WITHOUT any
    change to the yielded values (plain text / :class:`StreamDelta`).

    F31 channels: by default only FINAL-answer content is yielded, as plain
    strings. A gateway that streams ``reasoning_content``/``reasoning``/
    ``thinking`` deltas never gets that text into the answer channel. Pass
    *include_reasoning* (or *typed*) to receive :class:`StreamDelta` events
    instead, each tagged ``final`` or ``reasoning`` — reasoning is preserved
    but can never be mistaken for an answer by an untyped consumer.
    """
    typed_output = bool(typed or include_reasoning)
    handle = resolve(deadline)
    if handle is not None and handle.stopped():
        _finish_outcome(outcome, STREAM_DEADLINE, "budget already spent")
        return
    if cancel is not None and cancel.is_set():
        _finish_outcome(outcome, STREAM_CANCELLED)
        return
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    if timeout is None:
        timeout = (STREAM_CONNECT_TIMEOUT,
                   max(float(first_token_timeout or 0), 0.1))
    if handle is not None:
        # F24: the caller's budget, never the provider's, decides how long the
        # FIRST token may take to arrive.
        timeout = handle.timeout(timeout)
        if timeout is None:
            _finish_outcome(outcome, STREAM_DEADLINE, "budget ran out")
            return
    try:
        response = _session.post(
            _chat_url(base_url),
            headers=_headers(api_key),
            json=data,
            stream=True,
            timeout=timeout,
        )
    except Exception as exc:
        logging.warning("[OPENAI-COMPAT] Stream request error: %s", exc)
        _finish_outcome(outcome, STREAM_ERRORED, exc)
        return
    if response.status_code != 200:
        detail = ""
        try:
            detail = response.text[:300]
        except Exception:
            pass
        logging.warning("[OPENAI-COMPAT] Stream error: %s %s",
                        response.status_code, detail)
        _finish_outcome(outcome, STREAM_ERRORED,
                        "HTTP %s %s" % (response.status_code, detail))
        try:
            response.close()
        except Exception:
            pass
        return
    # [PERF] P1-19 — the provider's HTTP response headers have arrived: the
    # queueing/TTFT part of the call is over and only generation is left.
    _mark_headers("openai-compat", model)
    # P1-13: a half-read response with no charset used to hand back BYTES
    # (``line.startswith("data:")`` then raised on the comparison) and a
    # wrong charset mangled non-ASCII. Pin it, explicitly.
    try:
        response.encoding = "utf-8"
    except Exception:
        pass
    registered = _register_cancel_closer(cancel, getattr(response, "close", None))
    relaxed = False
    try:
        for line in response.iter_lines(decode_unicode=True):
            if cancel is not None and cancel.is_set():
                _finish_outcome(outcome, STREAM_CANCELLED)
                break
            # F24: checked between chunks, before the next chunk is consumed —
            # a stream that trickles in forever still ends here.
            if handle is not None and handle.stopped():
                _finish_outcome(outcome, STREAM_DEADLINE, "budget ran out")
                break
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if not payload or payload == "[DONE]":
                if isinstance(outcome, dict):
                    _finish_outcome(outcome, STREAM_COMPLETED,
                                    finish_reason=outcome.get("finish_reason"))
                break
            try:
                chunk = json.loads(payload)
            except Exception:
                continue
            # P1-13: a provider-side error frame used to be silently DROPPED,
            # which is exactly how a failed turn looked like an empty one.
            error = chunk.get("error")
            if error:
                detail = (error.get("message") if isinstance(error, dict)
                          else str(error))
                logging.warning("[OPENAI-COMPAT] in-stream provider error: %s",
                                detail)
                _finish_outcome(outcome, STREAM_ERRORED, detail or "provider error")
                break
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0] or {}
            reason = choice.get("finish_reason")
            if reason:
                if isinstance(outcome, dict):
                    outcome["finish_reason"] = str(reason)
                if str(reason) not in ("stop", "null", ""):
                    logging.warning(
                        "[OPENAI-COMPAT] stream finished early: finish_reason=%s",
                        reason)
            delta = choice.get("delta") or {}
            content = final_text(delta)
            if content:
                if not relaxed:
                    # Once the first token is here, the tight first-token
                    # budget becomes the generous per-chunk idle budget.
                    relaxed = True
                    _set_socket_timeout(response, idle_timeout)
                if isinstance(outcome, dict):
                    outcome["chunks"] = outcome.get("chunks", 0) + 1
                    outcome["first_token"] = True
                yield (StreamDelta(content, FINAL_CHANNEL)
                       if typed_output else content)
            if typed_output:
                # F31: reasoning is only ever delivered typed, never as a
                # bare string that a caller could append to an answer.
                thinking = reasoning_text(delta)
                if thinking:
                    yield StreamDelta(thinking, REASONING_CHANNEL)
        else:
            # The iterator ended without an explicit terminal: the provider
            # closed a completed stream.
            if isinstance(outcome, dict) and outcome.get("status") == STREAM_PENDING:
                _finish_outcome(outcome, STREAM_COMPLETED)
    except requests.RequestException as exc:
        # P1-13: a network failure is a TERMINAL ERROR EVENT, never a bare
        # traceback out of the generator. The consumer decides what to say.
        logging.warning("[OPENAI-COMPAT] stream read failed: %s", exc)
        _finish_outcome(outcome, STREAM_ERRORED, exc)
    except Exception as exc:  # pragma: no cover - defensive, still terminal
        logging.warning("[OPENAI-COMPAT] stream aborted: %s", exc)
        _finish_outcome(outcome, STREAM_ERRORED, exc)
    finally:
        _unregister_cancel_closer(cancel, registered)
        # F25 — never leave the socket open behind a cancelled speculation.
        try:
            response.close()
        except Exception:
            pass
