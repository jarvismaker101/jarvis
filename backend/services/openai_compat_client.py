"""OpenAI-compatible chat client for user-added providers.

Custom providers added from the UI (OpenRouter, Groq, any /v1-compatible
gateway) speak the OpenAI /chat/completions dialect. Responses are shaped
exactly like fireworks_client's ({"choices": [...]} on success, {} on
failure) so brain.py can treat this as a drop-in member of the fallback
chain. The API key travels in the Authorization header only.
"""

import json

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
_session.mount("https://", HTTPAdapter(max_retries=_retry_strategy))


def _chat_url(base_url):
    return str(base_url or "").rstrip("/") + "/chat/completions"


def _headers(api_key):
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def ask_openai_compat(
    messages, model, base_url, api_key, temperature=0.7, max_tokens=None,
    tools=None, tool_choice=None, reasoning_effort=None, timeout=None,
):
    """Non-stream chat completion; {} on any failure (fireworks pattern).

    *tools*/*tool_choice* (G8/F02): optional native tool-use definitions
    (OpenAI dialect). They ride on the same wire shape; when tools are given
    the response may carry ``tool_calls`` instead of plain content.

    *reasoning_effort* (F49): sent only when the caller's validated model
    snapshot says the model takes it — a model that runs its own default
    reasoning must never receive the field.

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


def ask_openai_compat_stream(
    messages, model, base_url, api_key, temperature=0.7, max_tokens=None,
    cancel=None, include_reasoning=False, typed=False,
):
    """Stream chat-completion text deltas (SSE) — mirrors fireworks_client.

    *cancel* is an optional ``threading.Event`` (F25): when it is set the
    loop stops reading and the HTTP response is closed immediately.

    F31 channels: by default only FINAL-answer content is yielded, as plain
    strings. A gateway that streams ``reasoning_content``/``reasoning``/
    ``thinking`` deltas never gets that text into the answer channel. Pass
    *include_reasoning* (or *typed*) to receive :class:`StreamDelta` events
    instead, each tagged ``final`` or ``reasoning`` — reasoning is preserved
    but can never be mistaken for an answer by an untyped consumer.
    """
    typed_output = bool(typed or include_reasoning)
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    try:
        response = _session.post(
            _chat_url(base_url),
            headers=_headers(api_key),
            json=data,
            stream=True,
            timeout=(5.05, 120),
        )
    except Exception as exc:
        print("[OPENAI-COMPAT] Stream request error:", exc)
        return
    if response.status_code != 200:
        print(
            "[OPENAI-COMPAT] Stream error:",
            response.status_code,
            response.text[:300],
        )
        return
    try:
        for line in response.iter_lines(decode_unicode=True):
            if cancel is not None and cancel.is_set():
                break
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if not payload or payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except Exception:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = final_text(delta)
            if content:
                yield (StreamDelta(content, FINAL_CHANNEL)
                       if typed_output else content)
            if typed_output:
                # F31: reasoning is only ever delivered typed, never as a
                # bare string that a caller could append to an answer.
                thinking = reasoning_text(delta)
                if thinking:
                    yield StreamDelta(thinking, REASONING_CHANNEL)
    finally:
        # F25 — never leave the socket open behind a cancelled speculation.
        try:
            response.close()
        except Exception:
            pass
