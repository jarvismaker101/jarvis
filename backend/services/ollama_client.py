"""Local Ollama API client for native accessibility agent logic."""

import json
import logging
import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_MODEL = "llama3.2:latest"
OLLAMA_CONNECT_TIMEOUT_SECONDS = 2
OLLAMA_READ_TIMEOUT_SECONDS = 15
OLLAMA_WARMUP_TIMEOUT_SECONDS = int(
    os.getenv("JARVIS_OLLAMA_WARMUP_TIMEOUT", "10")
)
# Keep the model resident in the Ollama daemon indefinitely (-1) so a
# re-wake after Jarvis closes doesn't have to reload llama3.2 into VRAM.
OLLAMA_KEEP_ALIVE = os.getenv("JARVIS_OLLAMA_KEEP_ALIVE", "-1")

_session = requests.Session()
_retry_strategy = Retry(
    total=1,
    connect=1,
    read=0,
    status=1,
    backoff_factor=1.0,
    status_forcelist=(408, 429, 500, 502, 503, 504),
    allowed_methods=frozenset({"GET", "POST"}),
)
_adapter = HTTPAdapter(max_retries=_retry_strategy)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)


def is_available():
    """Return True if Ollama is running and accessible."""
    try:
        resp = _session.get(
            f"{OLLAMA_BASE_URL}/",
            timeout=(OLLAMA_CONNECT_TIMEOUT_SECONDS, 1),
        )
        return resp.status_code == 200
    except Exception:
        return False


def _ollama_timeout(read_timeout):
    return (OLLAMA_CONNECT_TIMEOUT_SECONDS, read_timeout)


def _stream_response_text(resp):
    chunks = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            logging.debug("[OLLAMA] Skipping malformed stream line: %s", line[:120])
            continue
        chunks.append(payload.get("response", ""))
        if payload.get("done"):
            break
    return "".join(chunks)


def ask_ollama(
    prompt,
    system_prompt="",
    response_format=None,
    max_tokens=800,
    timeout=OLLAMA_READ_TIMEOUT_SECONDS,
    stream=False,
    quiet=False,
):
    """Send a request to local Ollama and return an OpenAI-shaped response."""
    body = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "stream": bool(stream),
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.0,
        },
    }

    if response_format and response_format.get("type") == "json_object":
        body["format"] = "json"

    try:
        resp = _session.post(
            OLLAMA_URL,
            json=body,
            timeout=_ollama_timeout(timeout),
            stream=bool(stream),
        )

        if resp.status_code != 200:
            if quiet:
                logging.debug("[OLLAMA] %d: %s", resp.status_code, resp.text[:300])
            else:
                logging.warning("[OLLAMA] %d: %s", resp.status_code, resp.text[:300])
            return {}

        if stream:
            text = _stream_response_text(resp)
        else:
            data = resp.json()
            text = data.get("response", "")

        if not text.strip():
            return {}

        return {
            "choices": [
                {
                    "message": {"content": text},
                }
            ],
            "model": OLLAMA_MODEL,
        }
    except Exception as exc:
        if quiet:
            logging.debug("[OLLAMA] Request error: %s", exc)
        else:
            logging.warning("[OLLAMA] Request error: %s", exc)
        return {}


def _model_installed(model):
    """Return True if the model exists locally in Ollama."""
    try:
        resp = _session.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=(OLLAMA_CONNECT_TIMEOUT_SECONDS, 5),
        )
        if resp.status_code != 200:
            return False
        return any(
            tag.get("name") == model or tag.get("name", "").startswith(model.split(":")[0])
            for tag in resp.json().get("models", [])
        )
    except Exception:
        return False


def warm_up_ollama():
    """Pre-load the configured model so the first real screen command is faster."""
    if not is_available():
        logging.debug("[OLLAMA] Not available; skipping warm-up.")
        return False
    if not _model_installed(OLLAMA_MODEL):
        logging.debug("[OLLAMA] Model %s not installed; skipping warm-up.", OLLAMA_MODEL)
        return False

    result = ask_ollama(
        'Return exactly {"ok": true}.',
        system_prompt="You are a JSON-only health check.",
        response_format={"type": "json_object"},
        max_tokens=8,
        timeout=OLLAMA_WARMUP_TIMEOUT_SECONDS,
        stream=False,
        quiet=True,
    )
    warmed = bool(result.get("choices"))
    if warmed:
        logging.info("[OLLAMA] Warm-up complete for %s", OLLAMA_MODEL)
    return warmed
