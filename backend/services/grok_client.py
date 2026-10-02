import os
import re

import requests

from backend.config import ENV_PATH, GROQ_API_KEY


API_KEY = GROQ_API_KEY
# H8: "llama-3.3-70b-versatile" is RETIRED at Groq (404) - it was still the
# fallback classifier default here. Default to the same Qwen 3.6 27B family
# this module already uses for vision (and that intent.py documents).
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
VISION_MODEL = os.getenv(
    "GROQ_VISION_MODEL",
    "qwen/qwen3.6-27b",
)

print("LOADING ENV FROM:", ENV_PATH)
print("API KEY FOUND:", API_KEY is not None)

# S26 — pooled keepalive session: this is the classifier's Groq fallback, and a
# bare ``requests.post`` paid a fresh TCP + TLS handshake on every classifier
# call. No hidden adapter retries are added — the plain fast-fail semantics
# (return {} on any error) are unchanged; only the connection is reused.
try:
    from backend.services.prewarm import pooled_session as _pooled_session
except Exception:  # pragma: no cover - the plain session still works
    _pooled_session = None
try:
    _session = _pooled_session()
except Exception:  # pragma: no cover
    _session = requests.Session()

# S27 — thinking OFF for the qwen fallback. Gated to the qwen family: Groq
# honours reasoning_effort on qwen3 models, and "none" is the value that
# disables it; other Groq models must never see the field.
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "none")


def _reasoning_effort(model=None):
    """The reasoning control for one call, or None (send nothing)."""
    if not GROQ_REASONING_EFFORT:
        return None
    if "qwen" not in str(model or DEFAULT_MODEL).lower():
        return None
    return GROQ_REASONING_EFFORT


def _post_chat_completion(data, timeout=(3.05, 18)):
    if not API_KEY:
        print("[GROQ] API KEY MISSING")
        return {}

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        response = _session.post(url, headers=headers, json=data, timeout=timeout)

        print("[GROQ] Status:", response.status_code)

        if response.status_code != 200:
            print("[GROQ] Error:", response.text[:500])
            return {}

        return response.json()
    except Exception as exc:
        print("REQUEST ERROR:", exc)
        return {}


def ask_grok(messages, temperature=0.7, max_tokens=None, model=None, timeout=None):
    data = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    # S27: the deployed Groq model (qwen3) writes a thinking block before the
    # answer. The text is stripped afterwards (_strip_think_blocks), but the
    # TIME is not — and inside the classifier's tight budget that thinking
    # time is the difference between an answer and a timeout. "none"
    # disables thinking on Groq; gated by model family so a non-reasoning
    # Groq model never sees the field.
    effort = _reasoning_effort(model)
    if effort:
        data["reasoning_effort"] = effort

    return _post_chat_completion(data, timeout=timeout or (3.05, 18))


def _strip_think_blocks(content):
    """Remove <think>…</think> reasoning blocks some models (Qwen) prepend.

    If the model spent the whole budget thinking (no closing tag, or nothing
    after the block), return "" so callers treat it as "no usable content"
    and fall back to another provider.
    """
    if not content:
        return content
    if "<think>" in content:
        if "</think>" not in content:
            return ""
        stripped = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return stripped
    return content.strip()


def ask_groq_vision(
    prompt,
    image_data_url,
    max_completion_tokens=650,
    model=None,
    response_format=None,
):
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ],
    }
    data = {
        "model": model or VISION_MODEL,
        "messages": [message],
        "temperature": 0.0,
        "max_completion_tokens": max_completion_tokens,
    }
    if response_format is not None:
        data["response_format"] = response_format

    result = _post_chat_completion(data, timeout=(3.05, 30))
    # Qwen reasoning models wrap output in <think>…</think>; strip it so
    # callers get clean JSON/text (and Groq's json_object validation works).
    if result and result.get("choices"):
        msg = result["choices"][0].get("message")
        if msg and msg.get("content"):
            msg["content"] = _strip_think_blocks(msg["content"])
    return result
