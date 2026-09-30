"""Meaning-based intent routing for Jarvis via Gemini (with a Groq Qwen fallback).

Every user message flows through a fast cloud LLM (Gemini 3.5 Flash Lite by
default) that *understands* the utterance — across English, Hindi and
Hinglish — and routes it to exactly one handler:

    chat     -> Gemini chat + TTS (pure conversation)
    tool     -> built-in executor (open site/app, play media, web search)
    screen   -> Gemini Vision analysis of the whole screen
    region   -> Gemini Vision analysis of the area around the cursor
    research -> deep web research on the user's real browser profile
                (headed, so captchas are solvable) with a summarized report
    task     -> opencode agent (file/folder ops, code, shell, automation…)

Gemini (Flash Lite) classifies first; if it is unavailable, the same Lite
model over OpenRouter is tried next, then Qwen on Groq. Any hard failure just
routes to chat rather than stalling.
"""

import json
import logging
import os
import re
import time

from backend.config import GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY
from backend.services.gemini_client import ask_gemini_chat
from backend.services.grok_client import _strip_think_blocks, ask_grok

_INTENT_PROMPT = (
    "You are Jarvis intent router. Understand MEANING across English, Hindi and Hinglish, not keywords.\n\n"
    "Intents (one only):\n"
    ' chat - conversation/general knowledge, no visual/action\n'
    ' tool - simple direct actions: open app/site, play media, web search (ONLY 4 actions below)\n'
     ' screen - LOOK AT whole screen (e.g. "what is on my screen", "what\'s on my screen jarvis", "screen pe kya dikh raha hai")\n'
    ' region - SPECIFIC small area/cursor/highlight (e.g. "what does this highlighted area mean", "cursor pe kya aaya")\n'
    ' research - LOOK UP on internet and report; ANY question needing CURRENT world facts: pricing/price/cost of any product, service or AI model (even obscure or possibly-new names - e.g. "whats the pricing for claude fable 5.1 model" -> research), latest/newest versions, releases, current events, news, scores, weather, dates, today/now/latest questions; when unsure between chat and research for a specific named product/model/version/price/date, choose research\n'
    ' task - ANY real computer manipulation not covered by 4 tool actions: files/folders, code, shell, install, settings, automation, multi-step web/browser interaction (e.g. "Go to 1hd.to website, search for One Piece Movie Red, and play it" -> task with task_description)\n\n'
    "Allowed tool actions (ONLY these -> else task):\n"
    ' {"action": "open_website", "input": "site.com"} {"action": "launch_app", "input": "app"}'
    ' {"action": "youtube_play", "input": "song"} {"action": "search", "input": "query"}'
    ' (+ browser chrome|edge|brave)\n\n'
    'Rules: "how are you" -> chat NOT research. screen/region no steps. task has task_description. tool has steps. research has query.'
    ' Hinglish e.g. "deepseek kya hai, dhundho" -> research.\n\n'
    'Reply ONLY JSON: {"intent": "chat"|"tool"|"screen"|"region"|"research"|"task", "steps": [], "query": "...", "task_description": "...", "original": "..."}\n'
    "User message: __MESSAGE__"
)


def _extract_json(content):
    if not content:
        return {}
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _parse_intent_json(content, message):
    """Normalise any model output into the canonical router result dict."""
    result = {
        "intent": "chat",
        "steps": [],
        "task_description": "",
        "query": message,
    }

    parsed = _extract_json(content)
    intent = parsed.get("intent")

    if intent == "research":
        research_q = str(parsed.get("query") or parsed.get("original") or message).strip()
        if research_q:
            result["intent"] = "research"
            result["query"] = research_q
            result["task_description"] = research_q
    elif intent == "task":
        description = str(parsed.get("task_description") or message).strip()
        if description:
            result["intent"] = "task"
            result["task_description"] = description
    elif intent == "screen":
        result["intent"] = "screen"
    elif intent == "region":
        result["intent"] = "region"
    elif intent == "tool":
        steps = parsed.get("steps")
        if isinstance(steps, list) and steps:
            cleaned = []
            for step in steps[:6]:
                if not isinstance(step, dict):
                    continue
                action = str(step.get("action") or "").strip()
                if action not in (
                    "open_website", "launch_app", "youtube_play", "search",
                ):
                    continue
                cleaned.append({
                    "action": action,
                    "input": str(step.get("input") or "").strip(),
                    "browser": str(step.get("browser") or "").strip() or None,
                })
            if cleaned:
                result["intent"] = "tool"
                result["steps"] = cleaned

    return result


def _classify_with_groq(message, timeout=(3, 3)):
    """Ask Qwen via the Groq API. Returns raw JSON text or "" on any failure."""
    if not GROQ_API_KEY:
        return ""
    response = ask_grok(
        [
            {
                "role": "system",
                "content": "Return strict JSON only. No markdown, no extra text.",
            },
            {
                "role": "user",
                "content": _INTENT_PROMPT.replace("__MESSAGE__", message),
            },
        ],
        temperature=0.0,
        max_tokens=320,
        model=os.getenv(
            "GROQ_INTENT_MODEL",
            os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b"),
        ),
        timeout=timeout,
    )
    if not response or not response.get("choices"):
        return ""
    content = response["choices"][0].get("message", {}).get("content", "")
    # Qwen reasoning models wrap output in thinking… response blocks — strip
    # so the JSON extractor sees clean text (and "" if the budget was eaten).
    return _strip_think_blocks(content)


def _classify_with_gemini(message, timeout=(3, 3)):
    """Ask Gemini Flash Lite. Returns raw JSON text or "" on any failure.

    Uses a short timeout + no_retry so a slow/hanging Gemini falls back to
    Groq fast — no urllib3 retry multiplication, no manual retry loop.
    """
    if not GEMINI_API_KEY:
        return ""
    response = ask_gemini_chat(
        [
            {
                "role": "system",
                "content": "Return strict JSON only. No markdown, no extra text.",
            },
            {
                "role": "user",
                "content": _INTENT_PROMPT.replace("__MESSAGE__", message),
            },
        ],
        temperature=0.0,
        max_tokens=300,
        model=os.getenv(
            "GEMINI_INTENT_MODEL",
            os.getenv("GEMINI_BRAIN_MODEL", "gemini-3.5-flash-lite"),
        ),
        timeout=timeout,
        no_retry=True,
    )
    if not response or not response.get("choices"):
        return ""
    return response["choices"][0].get("message", {}).get("content", "")


def _budget_timeout(remaining):
    """(connect, read) timeouts allocated from the remaining classification
    budget. The budget is a TOTAL deadline across primary + fallback — each
    call gets whatever is left, so the advertised timeout_ms is real."""
    if remaining <= 0:
        return None
    connect = min(3.0, remaining)
    read = max(0.1, remaining)
    return (connect, read)


def _classify_with_openrouter(message, timeout=(2, 2.5)):
    """Ask the Lite brain model over OpenRouter. Returns raw JSON text or "".

    First hop since the 2026-09-23 incident: the Cloudflare-fronted openrouter
    endpoint answers in ~1.4s even while the direct Gemini API degrades to
    7-45s+ behind a VPN relay — so the router keeps routing within its budget
    instead of silently landing every message on the chat fallback.
    """
    if not OPENROUTER_API_KEY:
        return ""
    from backend.services.openai_compat_client import ask_openai_compat
    response = ask_openai_compat(
        [
            {
                "role": "system",
                "content": "Return strict JSON only. No markdown, no extra text.",
            },
            {
                "role": "user",
                "content": _INTENT_PROMPT.replace("__MESSAGE__", message),
            },
        ],
        model=os.getenv(
            "INTENT_OPENROUTER_MODEL", "google/gemini-2.5-flash-lite"),
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0.0,
        max_tokens=500,
        timeout=timeout,
    )
    if not response or not response.get("choices"):
        return ""
    return response["choices"][0].get("message", {}).get("content", "")


def classify_intent(message: str, timeout_ms: int = 3500) -> dict:
    """Route *message* to chat/tool/screen/region/task.

    Every message is classified — no keyword pre-check — fastest cloud
    classifier first (the Lite brain model over OpenRouter), then Gemini
    direct, then Qwen on Groq. Any remaining failure just lands on chat.

    A single monotonic deadline (timeout_ms) spans the whole classification:
    primary and fallbacks share the remaining budget, so fast-fail never
    exceeds the advertised window.

    [PERF] P1-19: the verdict carries ``_source`` — the hop that answered
    (openrouter | gemini | groq | none). Which hop won is invisible in a single
    "classify" duration, and it is the first thing to check when a turn is
    slow.
    """
    fallback = {
        "intent": "chat",
        "steps": [],
        "task_description": "",
        "query": message,
        # [PERF] P1-19 — which hop produced a verdict travels with the verdict,
        # so the latency waterfall can say whether OpenRouter, Gemini or Groq
        # answered (and "none" when every hop failed).
        "_source": "none",
    }
    if not message or not message.strip():
        return fallback

    deadline = time.monotonic() + max(0, timeout_ms) / 1000.0

    # 1) Fastest hop first (2026-09-23): OpenRouter stays ~1.4s even when the
    #    direct Gemini API degrades behind a VPN relay, so routing survives.
    timeout = _budget_timeout(deadline - time.monotonic())
    if timeout:
        try:
            content = _classify_with_openrouter(message, timeout=timeout)
            result = _parse_intent_json(content, message)
            if result["intent"] != "chat" or content:
                # Even a chat verdict from the model is a deliberate answer.
                result["_source"] = "openrouter"   # [PERF] P1-19 (hop label)
                return result
        except Exception as exc:
            logging.warning("[INTENT] OpenRouter classifier unavailable: %s", exc)

    # 2) Cloud classifier (Gemini 3.5 Flash Lite direct).
    timeout = _budget_timeout(deadline - time.monotonic())
    if timeout:
        try:
            content = _classify_with_gemini(message, timeout=timeout)
            result = _parse_intent_json(content, message)
            if result["intent"] != "chat" or content:
                # Even a chat verdict from the model is a deliberate answer.
                result["_source"] = "gemini"       # [PERF] P1-19 (hop label)
                return result
        except Exception as exc:
            logging.warning("[INTENT] Gemini classifier unavailable: %s", exc)

    # 3) Single fallback: Qwen 3.6 27B on Groq — only with budget left.
    timeout = _budget_timeout(deadline - time.monotonic())
    if timeout:
        try:
            content = _classify_with_groq(message, timeout=timeout)
            result = _parse_intent_json(content, message)
            if result["intent"] != "chat" or content:
                result["_source"] = "groq"         # [PERF] P1-19 (hop label)
                return result
        except Exception as exc:
            logging.warning("[INTENT] Groq Qwen classifier unavailable: %s", exc)

    # 3) Never break chat.
    return fallback
