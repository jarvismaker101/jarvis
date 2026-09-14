"""OpenRouter vision API client for screen-control grounding.

Provides access to free vision models (Gemma 4, Nemotron) via OpenRouter
that are significantly better at GUI grounding than the default Groq model.
Falls back gracefully if OPENROUTER_API_KEY is not set.

Includes a rate-limit cooldown cache so 429'd models are skipped instantly
instead of wasting seconds on failed network calls.
"""

import logging
import time

import requests

from backend.config import OPENROUTER_API_KEY

# Free vision models ranked by GUI-grounding ability.
# Gemma 4 31B is Google's latest & largest free vision model.
# Gemma 4 26B is slightly smaller but equally capable.
# Nemotron Nano VL is NVIDIA's multimodal model, smaller but fast.
OPENROUTER_VISION_MODELS = [
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
]

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ---- rate-limit cooldown cache ----
# When a model returns 429, we record the timestamp and skip it for
# _COOLDOWN_SECONDS without wasting a network round-trip.
_COOLDOWN_SECONDS = 120
_rate_limited_until = {}  # model_id -> timestamp when cooldown expires


def _is_on_cooldown(model_id):
    """Return True if the model was recently rate-limited."""
    expires = _rate_limited_until.get(model_id, 0)
    return time.time() < expires


def _mark_rate_limited(model_id):
    """Record that this model just returned 429."""
    _rate_limited_until[model_id] = time.time() + _COOLDOWN_SECONDS
    logging.info(
        "[OPENROUTER] %s on cooldown for %ds", model_id, _COOLDOWN_SECONDS
    )


def is_available():
    """Return True if an OpenRouter API key is configured."""
    return bool(OPENROUTER_API_KEY)


def any_model_available():
    """Return True if at least one model is not on cooldown."""
    if not OPENROUTER_API_KEY:
        return False
    return any(not _is_on_cooldown(m) for m in OPENROUTER_VISION_MODELS)


def ask_openrouter_vision(
    prompt,
    image_data_url,
    max_completion_tokens=800,
    model=None,
    response_format=None,
):
    """Send a vision request to OpenRouter and return the raw API response.

    Returns an empty dict on failure so callers can fall through to the
    next provider.  Skips models on cooldown instantly.
    """
    if not OPENROUTER_API_KEY:
        return {}

    effective_model = model or OPENROUTER_VISION_MODELS[0]

    # Skip immediately if this model was recently rate-limited.
    if _is_on_cooldown(effective_model):
        return {}

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
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://jarvis-assistant.local",
        "X-Title": "Jarvis Assistant",
    }

    try:
        # Shorter timeout: 2s connect, 25s read.  Free tiers can be slow
        # but we don't want to block voice interaction for 45s.
        response = requests.post(
            _OPENROUTER_URL, headers=headers, json=data, timeout=(2, 25)
        )

        if response.status_code == 429:
            _mark_rate_limited(effective_model)
            return {}

        if response.status_code != 200:
            logging.warning(
                "[OPENROUTER] %s returned %d: %s",
                effective_model,
                response.status_code,
                response.text[:300],
            )
            return {}

        result = response.json()

        # OpenRouter wraps some errors inside a 200 response.
        if result.get("error"):
            err_str = str(result["error"])
            if "429" in err_str or "rate" in err_str.lower():
                _mark_rate_limited(effective_model)
            logging.warning(
                "[OPENROUTER] %s API error: %s",
                effective_model,
                err_str[:300],
            )
            return {}

        return result

    except Exception as exc:
        logging.warning("[OPENROUTER] Request error for %s: %s", effective_model, exc)
        return {}
