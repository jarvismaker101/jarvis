import os
import re

import requests

from backend.config import ENV_PATH, GROQ_API_KEY


API_KEY = GROQ_API_KEY
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
VISION_MODEL = os.getenv(
    "GROQ_VISION_MODEL",
    "qwen/qwen3.6-27b",
)

print("LOADING ENV FROM:", ENV_PATH)
print("API KEY FOUND:", API_KEY is not None)


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
        response = requests.post(url, headers=headers, json=data, timeout=timeout)

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
