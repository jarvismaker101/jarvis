"""Native browser-automation agent for Jarvis.

The default task engine (config.TASK_ENGINE == 'browser_agent'): a strong
model with function calling drives the brave-control MCP daemon directly —
warm persistent browser with DOM inventory and self-healing, no opencode
CLI involved. The loop is deliberately conservative: at most 2 retries per
failing action, then the task is reported as not completed with the error
shown (user hard requirement: never flail).

Model providers are adapters over one neutral message list
(role user/assistant/tool, content, tool_calls [{'id','name','arguments'}],
tool_call_id, name), converted per provider at request time.
"""

import base64
import datetime
import hashlib
import io
import json
import logging
import os
import re
import tempfile
import threading
import time
from urllib.parse import urlsplit

import requests

from PIL import Image, ImageChops, ImageDraw, ImageFont

from backend import config
from backend.core import deadline as budget
from backend.services.brave_mcp_client import BraveMcpClient
from backend.services.opencode_client import (
    append_activity_line,
    ensure_brave_mcp_daemon,
    narrate_activity,
    truncate_activity_log,
)
from backend.services.task_result import FailureTracker, TaskResult
from backend.services import jobs as job_registry
from backend.services import tool_policy

# Reused for all model calls (avoids per-turn TCP/TLS handshake)
_MODEL_SESSION = requests.Session()


def _default_grants():
    """Authorities a browser job starts with (F17).

    Unrestricted page JavaScript is PRIVILEGED: it can mutate the DOM, submit
    forms and read cookies, so it is NOT granted by default — the model is
    told to use the typed tools instead. Set JARVIS_BROWSER_PRIVILEGED_JS=1
    to grant it (the pre-audit behaviour).
    """
    grants = set()
    flag = str(os.getenv("JARVIS_BROWSER_PRIVILEGED_JS", "0")).strip().lower()
    if flag in ("1", "true", "yes", "on"):
        grants.add("privileged_js")
    return grants


def _grants_for(session):
    """Grants in force for a run (F17).

    An explicitly EMPTY grant set means exactly that: no authority. The old
    ``set(grants) if grants else _default_grants()`` treated "holds nothing"
    as "holds nothing recorded" and silently revived the environment default
    (including privileged page JavaScript when JARVIS_BROWSER_PRIVILEGED_JS
    was set). Only a session that never recorded grants falls back.
    """
    try:
        recorded = (session or {})["grants"]
    except Exception:
        return _default_grants()
    if recorded is None:
        return _default_grants()
    if isinstance(recorded, str):
        recorded = [recorded] if recorded else []
    try:
        return set(recorded)
    except TypeError:
        return set()

_SYSTEM_PROMPT = (
    "You are a browser automation agent. "
    "COMPLETION DOCTRINE: at the start of EVERY turn ask: is the goal already "
    "achieved? If yes, STOP calling tools and give the final spoken-friendly "
    "summary NOW. Verify success at most once (verify_playing for media "
    "playback), then finish - never keep improving a finished task. "
    "Default browser loop: look -> act visually (click_mark / fill_mark / "
    "click_text / click_point) -> wait_for -> look again. "
    "After open_brave or navigate, call look first - it returns an annotated "
    "screenshot with numbered marks plus a table; act on the marks you see. "
    "Marks are from the most recent look and become stale after navigation "
    "or page changes - after any navigation or missed click, look again. "
    "For text inputs use fill_mark on the input's mark number (press_enter "
    "true for search boxes); use fill only when the input has no mark. "
    "DOM tools (batch_probe / wait_for) are FALLBACK ONLY: "
    "use them after a visual attempt failed, when a needed mark is missing, "
    "or when the task needs extracted data (prices, lists, titles). NEVER "
    "call them to discover what is clickable when a fresh look "
    "exists. "
    "Use batch_probe to answer several DOM questions in one step - it takes "
    "typed read-only expressions such as document.querySelector('#price')"
    ".innerText or document.querySelectorAll('.row').length. "
    "After actions that trigger async updates (search, form submit, SPA "
    "navigation), call wait_for instead of sleeping or re-polling. "
    "Video players / cross-origin embeds: DOM tools (wait_for, batch_probe) "
    "and synthetic clicks cannot see or act inside cross-origin "
    "iframes - do not hunt an embedded player's internal buttons; click the "
    "page's own overlay at most, look again after each player click and "
    "re-assess, judge playback from the look screenshot or one verify_playing "
    "call, then finish. "
    "When the user references content on their screen or a tab that should "
    "already be open, first call list_tabs to enumerate open tabs, then "
    "switch_tab with url_contains to the matching tab, then look; only ask "
    "the user for a URL after enumerating tabs. "
    "look is the ONLY tool that shows you an image - file tools never "
    "display images, so never try to view screenshots or images through "
    "file tools. To confirm media playback, verify_playing is the ONLY "
    "reliable way - never judge playback from a saved file. "
    "Complete the task in as few steps as possible. When done, reply with "
    "ONLY a short spoken-friendly summary of the result (include counts and "
    "notable items, not just 'it worked'). If the task is genuinely "
    "impossible, say exactly what failed. "
    "Indices in the look marks table are for click_mark and fill_mark ONLY. "
    "On a watch, embed or player page, click the play affordance (player "
    "mark, video overlay, or a labeled play button) ONCE, then IMMEDIATELY "
    "call verify_playing - it is the designated playback check. Do not "
    "re-click the same coordinates without a visible state change; if a click "
    "does nothing, look again and pick a different target. "
    "If a popup or overlay appears (close buttons, 'switch server' notices, "
    "cookie banners), dismiss it with one targeted click (find the X via "
    "marks or elementFromPoint), then continue - do not spend multiple probes. "
    "Video players often load their real content in an iframe only AFTER "
    "clicking play - so click play first, THEN wait or look; never wait for "
    "an iframe before clicking. "
    "RESULT-TRUST DOCTRINE: When a click or action result reports clicked=true "
    "with the resulting url and title, do NOT issue a look just to confirm "
    "the navigation - trust the result and plan the next action directly. "
    "After a click that navigates to a new page, confirm arrival with "
    "wait_for on expected text or url instead of a look. "
    "When a click result shows the url UNCHANGED (no navigation), continue "
    "using the existing marks table - marks only become stale after a "
    "navigation, not after a click that stayed on the same page. "
    "look is for discovering a genuinely unknown page, finding an element "
    "that has no mark, or locating an unlabeled target - never call look "
    "just to double-check what the tool result already told you. "
    "Use click_mark / click_point / click_text for all clicks and "
    "fill_mark / fill for all fills. "
    "G6 MARK-TRUTH: marks are bound to the element they were seen on - a "
    "click or fill on a mark whose page navigated, reloaded or reflowed is "
    "REFUSED with 'call look again'. When refused, look again; never retry "
    "the same stale mark. "
    "G6 TYPED INTERACTIONS: use the typed tools instead of improvised "
    "JavaScript - scroll (real mouse wheel; then look again to refresh "
    "marks), select_option (dropdowns, by mark or css; the value is read back "
    "from the control and only a verified value is reported), set_checked "
    "(checkboxes/radios; read back too), upload_file (file inputs; the path "
    "must exist and be inside a granted workspace AND the destination site "
    "must be approved for uploads, otherwise it is refused), download (saves "
    "the file and returns its on-disk artifact path; a click with no real "
    "artifact is a failure), drag_drop (source -> target). "
)

_NARRATION_PHRASES = {
    "open_brave": "Opening the browser",
    "navigate": "Navigating",
    "screenshot": "Taking a screenshot",
    "new_tab": "Opening a new tab",
    "evaluate": "Checking the page",
    "look": "Looking at the page",
    "click_mark": "Clicking",
    "batch_probe": "Checking the page",
    "wait_for": "Waiting for the page",
    "fill": "Typing",
    "fill_mark": "Typing",
    "click_text": "Clicking",
    "click_point": "Clicking",
    "verify_playing": "Checking playback",
    "scroll": "Scrolling",
    "select_option": "Selecting",
    "set_checked": "Checking",
    "upload_file": "Uploading",
    "download": "Downloading",
    "drag_drop": "Dragging",
}

_MAX_RESULT_CHARS = 15000

# ── Stop control ────────────────────────────────────────────────────────────
# F20: the source of truth is a per-job JobToken (see backend/services/jobs.py).
# This module-level event is kept as a LEGACY BRIDGE only: request_stop() now
# cancels the identified job, and cancelling a browser job sets this event so
# older callers that still poll it keep working. It is no longer cleared
# blindly at the start of every run (that used to disarm a stop the user had
# just issued for the previous job).
_STOP_REQUESTED = threading.Event()

#: Job ids owned by a run through run_browser_task below. The job-cancelled
#: stop handler (registered above) arms _STOP_REQUESTED ONLY for these — a
#: foreign browser job (a direct registry test, another engine's stop) must
#: not leak "user pressed STOP" into the next unrelated loop.
_OWNED_JOB_IDS = set()

_STOP_MESSAGE = "Stopped per your request."


def _on_job_cancelled(job):
    # The stop handler fires for EVERY browser cancellation — including jobs
    # this module never started (e.g. a /task/stop test that creates and
    # cancels jobs directly). Arming the module-global flag for a foreign
    # job leaks "user pressed STOP" into the next unrelated loop, which is
    # exactly the order-dependent failure the BA-14/BA-03 plumbing tests hit
    # after the websearch stop suite. Only jobs owned by a run through
    # run_browser_task (tracked in _OWNED_JOB_IDS) may arm it.
    try:
        if job is not None and getattr(job, "job_id", None) in _OWNED_JOB_IDS:
            _STOP_REQUESTED.set()
    except Exception:
        pass


job_registry.register_stop_handler("browser", _on_job_cancelled)


def request_stop(job_id=None):
    """Ask a browser task to stop (F20).

    With *job_id* only that job is cancelled; without one every live browser
    job is. The legacy flag is armed either way so callers that poll
    stop_requested() (routes, voice, the UI) observe the request even when it
    arrived between runs; the next run clears it, but ONLY when no other
    browser job is still live — that is what stops run B from disarming a
    stop the user issued for run A.
    """
    cancelled = job_registry.request_stop(job_id, kinds=("browser",))
    _STOP_REQUESTED.set()
    return cancelled


# ── Rank 5: what a stopped run had already done ──
# A stopped TaskResult drops the session's committed effects, so the user
# never hears what the run did before the stop ("Stop" that leaves silent
# changes behind is exactly what "stop means stop" must not do). The loop
# publishes a short report the moment it honours a stop; the brain's stop
# finisher consumes it for its "Stopped" reply.
_STOP_REPORT_LOCK = threading.Lock()
_last_stop_report = ""


def _publish_stop_report(session):
    """Rank 5: remember the committed actions of a run being stopped."""
    global _last_stop_report
    try:
        actions = [str(item) for item in
                   ((session or {}).get("completed_actions") or [])]
    except Exception:
        actions = []
    if not actions:
        return
    summary = "; ".join(item[:60] for item in actions[-5:])
    if len(summary) > 240:
        summary = summary[:237].rstrip() + "..."
    with _STOP_REPORT_LOCK:
        _last_stop_report = summary


def consume_stop_report():
    """Take (and clear) the last stopped run's already-done report."""
    global _last_stop_report
    with _STOP_REPORT_LOCK:
        report = _last_stop_report
        _last_stop_report = ""
    return report


def stop_requested():
    """Legacy poll: the shared flag OR any live browser job being cancelled."""
    if _STOP_REQUESTED.is_set():
        return True
    return any(job.should_stop()
               for job in job_registry.live_jobs(kind="browser"))

# Keys Gemini's functionDeclarations accepts; everything else is dropped.
_GEMINI_SCHEMA_KEYS = ("type", "description", "items", "properties", "required", "enum")


def _fail(message):
    return "TASK NOT COMPLETED. Error: %s" % message


def _fail_and_log(message):
    append_activity_line("RESULT failed: %s\n" % message)
    return TaskResult.failed(message)


def _clip_result(text):
    """Cap a tool result before it enters the model history (token safety)."""
    text = text or ""
    if len(text) > _MAX_RESULT_CHARS:
        return text[:_MAX_RESULT_CHARS] + "...[truncated]"
    return text


def _trim_look_history(history):
    """Strip old look images from history, keeping only the latest K.

    Collects tool messages carrying image_b64, keeps the newest
    BROWSER_AGENT_KEEP_LAST_IMAGES intact, and for every older one deletes
    image_b64 and replaces content with Page/Title header + tombstone.
    Idempotent: already-trimmed messages have no image_b64.
    Non-look tool results (no image_b64) are untouched.
    """
    try:
        keep = int(getattr(config, "BROWSER_AGENT_KEEP_LAST_IMAGES", 1))
    except Exception:
        keep = 1
    if keep < 0:
        keep = 0
    image_indices = [
        i for i, m in enumerate(history)
        if m.get("role") == "tool" and m.get("image_b64")
    ]
    if len(image_indices) <= keep:
        return
    to_trim = image_indices[:-keep] if keep > 0 else image_indices
    for idx in to_trim:
        msg = history[idx]
        # Already trimmed? skip (idempotent)
        if "image_b64" not in msg:
            continue
        msg.pop("image_b64", None)
        content = msg.get("content") or ""
        lines = content.splitlines()
        header_lines = []
        for line in lines:
            if line.startswith("Page:") or line.startswith("Title:"):
                header_lines.append(line)
                if len(header_lines) == 2:
                    break
        # fallback: if headers not found, keep whatever Page/Title exists
        if len(header_lines) < 2:
            # try first 4 lines scan
            header_lines = [l for l in lines[:6] if l.startswith("Page:") or l.startswith("Title:")]
            if not header_lines and lines:
                # keep first line as header if no Page/Title found
                header_lines = lines[:1]
        new_content = "\n".join(header_lines)
        if new_content:
            new_content += "\n"
        new_content += "[earlier look image and marks omitted - use the latest look]"
        msg["content"] = new_content


def _sanitize_schema(schema):
    """Strip JSON-Schema keys Gemini rejects, recursively."""
    if not isinstance(schema, dict):
        return schema
    out = {}
    for key, value in schema.items():
        if key not in _GEMINI_SCHEMA_KEYS:
            continue
        if key == "properties":
            out[key] = {
                name: _sanitize_schema(sub) for name, sub in value.items()
            }
        elif key == "items":
            out[key] = _sanitize_schema(value)
        else:
            out[key] = value
    return out


def _to_openai_messages(messages):
    out = []
    for message in messages:
        if message["role"] == "tool":
            content = message.get("content") or ""
            b64 = message.get("image_b64")
            if b64:
                # Multimodal tool result: text + image
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": message["tool_call_id"],
                        "content": [
                            {"type": "text", "text": content},
                            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,%s" % b64}},
                        ],
                    }
                )
            else:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": message["tool_call_id"],
                        "content": content,
                    }
                )
        elif message["role"] == "assistant":
            item = {"role": "assistant", "content": message.get("content")}
            calls = message.get("tool_calls") or []
            if calls:
                item["tool_calls"] = [
                    {
                        # F38: never lose the call id (a Gemini-style call
                        # routed here has no id) — mint a stable one instead
                        # of raising on the OpenAI payload build.
                        "id": str(call.get("id") or "call_%d" % position),
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call.get("arguments") or {}),
                        },
                    }
                    for position, call in enumerate(calls)
                ]
            out.append(item)
        elif message["role"] == "system":
            out.append({"role": "system", "content": message.get("content") or ""})
        else:
            out.append({"role": "user", "content": message.get("content") or ""})
    return out


def _tool_call_id(message, index=0):
    """Stable id for a tool result Gemini must correlate (F38)."""
    return str(message.get("tool_call_id") or message.get("id")
               or "call_%s_%d" % (message.get("name") or "tool", index))


def _to_gemini_contents(messages):
    """Neutral history -> Gemini ``contents`` (F38).

    The deployed API's grouped format is used: every tool result belonging to
    one model turn is a ``functionResponse`` part INSIDE ONE user content,
    and each part carries the ``id`` (and ``name``) of the ``functionCall`` it
    answers. The previous shape emitted one user message per result and
    matched them by NAME only, so two parallel calls to the same tool (or two
    tools that share a name) could not be told apart, and a look image
    "interposed" as its own message between outstanding responses. The image
    now rides INSIDE the functionResponse it belongs to (``functionResponse.
    parts``), which is the supported way to attach media to a response, so
    call/response/observation stay one group and stay correlated.

    ``thoughtSignature`` (the continuation metadata Gemini 2.5-class thinking
    models require echoed back on the functionCall part) is preserved when the
    adapter captured it.

    Ids are ALWAYS emitted on both sides: a call the upstream adapter did not
    stamp with an id gets a deterministic one (``call_<name>_<position>``) and
    the matching result reuses it, so a repeated tool name cannot make the
    pairing ambiguous even for id-less upstreams.
    """
    contents = []
    index = 0
    # Results of the assistant turn currently being answered: (name, id).
    pending = []
    while index < len(messages):
        message = messages[index]
        if message["role"] == "system":
            index += 1
            continue
        if message["role"] == "tool":
            # Group every consecutive tool result into ONE user content.
            parts = []
            cursor = index
            while cursor < len(messages) and messages[cursor]["role"] == "tool":
                tool_message = messages[cursor]
                name = tool_message.get("name")
                call_id = tool_message.get("tool_call_id") or tool_message.get("id")
                if not call_id and pending:
                    for position, (pending_name, pending_id) in enumerate(pending):
                        if pending_name == name:
                            call_id = pending_id
                            pending.pop(position)
                            break
                response = {
                    "name": name,
                    "response": {"result": tool_message.get("content") or ""},
                    "id": str(call_id) if call_id else _tool_call_id(tool_message,
                                                                    cursor),
                }
                part = {"functionResponse": response}
                b64 = tool_message.get("image_b64")
                if b64:
                    # Media attached to the response it belongs to, not to a
                    # separate message that could drift out of order.
                    response["parts"] = [
                        {"text": "Visual observation for the %s tool result above:"
                         % name},
                        {"inlineData": {"mimeType": "image/jpeg", "data": b64}},
                    ]
                parts.append(part)
                cursor += 1
            contents.append({"role": "user", "parts": parts})
            pending = []
            index = cursor
            continue
        if message["role"] == "assistant":
            parts = []
            pending = []
            if message.get("content"):
                parts.append({"text": message["content"]})
            for position, call in enumerate(message.get("tool_calls") or []):
                name = call.get("name")
                # Echo (or mint) the id so the matching functionResponse is
                # unambiguous even with repeated tool names.
                call_id = str(call.get("id")
                              or "call_%s_%d" % (name or "tool", position))
                function_call = {
                    "name": name,
                    "args": call.get("arguments") or {},
                    "id": call_id,
                }
                pending.append((name, call_id))
                part = {"functionCall": function_call}
                signature = call.get("thought_signature") or call.get("thoughtSignature")
                if signature:
                    part["thoughtSignature"] = signature
                parts.append(part)
            contents.append({"role": "model", "parts": parts})
            index += 1
            continue
        contents.append(
            {"role": "user", "parts": [{"text": message.get("content") or ""}]}
        )
        index += 1
    return contents


def _resolve_browser_tool_model():
    """Browser tool model per task/turn: registry browser_tool_model else env default.

    Reads registry per call (per-turn) so a UI switch applies on the next model
    turn. If registry is unavailable or corrupt, degrades to the env-default
    (fireworks qwen3p7-plus or whatever BROWSER_AGENT_* is in .env).
    """
    try:
        from backend.services import model_registry
        sel = model_registry.get_model_for_role("browser_tool")
        prov = str(sel.get("provider") or "").strip()
        mod = str(sel.get("model") or "").strip()
        if prov and mod:
            return prov, mod
    except Exception:
        pass
    return config.BROWSER_AGENT_PROVIDER, config.BROWSER_AGENT_MODEL


# Adapters that can carry BOTH tool calls and visual (JPEG) observations in
# one conversation. A browser model selection resolving to anything else is
# rejected outright at selection time — silently degrading would drop every
# look screenshot and leave the model acting blind.
_ADAPTER_TOOLS_AND_VISION = {
    "fireworks": True,   # OpenAI-compatible: tool calls + image_url parts
    "groq": True,        # OpenAI-compatible: tool calls + image_url parts
    "openrouter": True,  # OpenAI-compatible: tool calls + image_url parts
    "cline": True,       # OpenAI-compatible: tool calls + image_url parts
    "gemini": True,      # functionCall/functionResponse + inlineData observation
}


def _adapter_supports_tools_and_vision(provider, model=None):
    """True when the (provider, model) adapter can carry tools AND images.

    F38: an UNKNOWN provider used to be accepted outright ("custom providers
    ride the OpenAI-compatible adapter"), so a typo, a removed provider id or
    an arbitrary string inherited tool+vision support and the browser agent
    would proceed — then drop every screenshot. Now a provider that is not one
    of the shipped adapters must be a REGISTERED provider whose declared
    capabilities include tool calling and vision; anything else fails closed.
    """
    pid = str(provider or "").strip()
    if pid in _ADAPTER_TOOLS_AND_VISION:
        return bool(_ADAPTER_TOOLS_AND_VISION[pid])
    if not pid:
        return False
    try:
        from backend.services import model_registry

        capabilities = set(model_registry.model_capabilities_for(pid, model))
    except Exception:
        return False
    if not capabilities:
        return False
    return {"tool_calling", "vision_input"} <= capabilities


def _model_wire_timeout():
    """Wire timeout for one model generation (BA-03).

    The model's own budget (``BROWSER_AGENT_MODEL_TIMEOUT``, ~90 s) sliced
    to the bound task deadline — a generation never gets fresh time past
    the task's end. Raises :class:`BudgetExhausted` instead of sending when
    the budget is already spent; unbound callers (tests, one-off probes)
    get the plain configured budget.
    """
    sliced = budget.seconds_for(None, config.BROWSER_AGENT_MODEL_TIMEOUT)
    if sliced is None:
        raise budget.BudgetExhausted(
            "browser task budget exhausted - model call not sent")
    return sliced


def _call_openai_compatible(url, api_key, messages, tools, model=None, provider=None):
    """One turn against an OpenAI-compatible chat completions endpoint."""
    effective_model = model or config.BROWSER_AGENT_MODEL
    effective_provider = provider or config.BROWSER_AGENT_PROVIDER
    payload = {
        "model": effective_model,
        "messages": _to_openai_messages(messages),
    }
    if tools:
        # Withheld-tools turns (forced final summary) omit the keys entirely.
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object"}),
                },
            }
            for tool in tools
        ]
        payload["tool_choice"] = "auto"
    # MiniMax and GLM models always use their own default reasoning: never send
    # reasoning_effort for them, whatever the provider.
    if effective_provider == "fireworks" and (
        config.BROWSER_AGENT_REASONING_EFFORT
    ) and "minimax" not in (effective_model or "").lower() and "glm" not in (effective_model or "").lower():
        payload["reasoning_effort"] = config.BROWSER_AGENT_REASONING_EFFORT
    response = _MODEL_SESSION.post(
        url,
        headers={"Authorization": "Bearer %s" % api_key},
        json=payload,
        timeout=_model_wire_timeout(),
    )
    response.raise_for_status()
    body = response.json()
    # Cline wraps the OpenAI payload under a top-level "data" key.
    if isinstance(body.get("data"), dict) and "choices" in body["data"]:
        body = body["data"]
    message = body["choices"][0]["message"]
    text = message.get("content")
    tool_calls = []
    for index, call in enumerate(message.get("tool_calls") or []):
        function = call.get("function", {})
        raw_arguments = function.get("arguments")
        parse_error = ""
        try:
            arguments = json.loads(raw_arguments or "{}")
        except (ValueError, TypeError) as exc:
            # F13: this used to become `{}` — an empty call that still ran.
            # The call is carried with its parse error so `_run_one_tool` can
            # refuse it without dispatching anything.
            arguments = {}
            parse_error = str(exc)
        if not isinstance(arguments, dict):
            parse_error = parse_error or "arguments were not a JSON object"
            arguments = {}
        entry = {
            "id": call.get("id") or "call_%d" % index,
            "name": function.get("name", ""),
            "arguments": arguments,
        }
        if parse_error:
            entry["parse_error"] = parse_error
        tool_calls.append(entry)
    return text, tool_calls


def _call_gemini(messages, tools, model=None):
    """One turn against the Gemini generateContent endpoint."""
    effective_model = model or config.BROWSER_AGENT_MODEL
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + effective_model
        + ":generateContent"
    )
    payload = {
        "system_instruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
        "contents": _to_gemini_contents(messages),
    }
    if tools:
        payload["tools"] = [
            {
                "functionDeclarations": [
                    {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": _sanitize_schema(
                            tool.get("input_schema", {"type": "object"})
                        ),
                    }
                    for tool in tools
                ]
            }
        ]
    response = _MODEL_SESSION.post(
        url,
        params={"key": config.GEMINI_API_KEY},
        json=payload,
        timeout=_model_wire_timeout(),
    )
    response.raise_for_status()
    body = response.json()
    candidates = body.get("candidates") or []
    if not candidates:
        return None, []
    candidate = candidates[0]
    parts = candidate.get("content", {}).get("parts") or []
    text = None
    tool_calls = []
    for part in parts:
        function_call = part.get("functionCall")
        if function_call:
            raw_args = function_call.get("args")
            parse_error = ""
            if isinstance(raw_args, str):
                # The deployed contract returns an object; a string only
                # appears from a non-conforming proxy and must not silently
                # become {} (F13: malformed calls do nothing).
                try:
                    raw_args = json.loads(raw_args or "{}")
                except ValueError as exc:
                    raw_args = {}
                    parse_error = str(exc)
            if raw_args is None:
                raw_args = {}
            if not isinstance(raw_args, dict):
                parse_error = parse_error or "arguments were not a JSON object"
                raw_args = {}
            call = {
                # F38: prefer the id the API issued so a response can be
                # correlated with THIS call; only mint one when absent.
                "id": str(function_call.get("id")
                          or "gemini_%d_%d" % (candidate.get("index") or 0,
                                               len(tool_calls))),
                "name": function_call.get("name", ""),
                "arguments": raw_args,
            }
            if parse_error:
                call["parse_error"] = parse_error
            # F38: required continuation metadata for thinking models.
            signature = part.get("thoughtSignature") or part.get("thought_signature")
            if signature:
                call["thought_signature"] = signature
            tool_calls.append(call)
        elif "text" in part and text is None:
            text = part["text"]
    return text, tool_calls


def _model_turn(messages, tools):
    """One model call for the configured provider; returns (text, tool_calls).

    Resolved per-turn from the runtime registry (browser_tool_model) so a UI
    switch applies on the next turn with no restart. Degrades to the env
    default on registry failure. Supports custom providers as openai_compat.
    Raises on transport/HTTP errors and missing API keys.
    """
    provider, model = _resolve_browser_tool_model()
    if not _adapter_supports_tools_and_vision(provider, model):
        raise RuntimeError(
            "browser model provider '%s' cannot carry both tools and visual "
            "observations — selection rejected" % provider
        )
    if provider == "gemini":
        return _call_gemini(messages, tools, model=model)
    endpoints = {
        "fireworks": (config.FIREWORKS_API_URL, config.FIREWORKS_API_KEY),
        "groq": (config.GROQ_API_URL, config.GROQ_API_KEY),
        "openrouter": (config.OPENROUTER_API_URL, config.OPENROUTER_API_KEY),
        "cline": (config.CLINE_API_URL, config.CLINE_API_KEY),
    }
    if provider in endpoints:
        url, api_key = endpoints[provider]
    else:
        # Custom provider (openai_compat) — resolve via registry credentials
        try:
            from backend.services import model_registry
            api_key, base_url = model_registry.get_provider_credentials(provider)
            if api_key and base_url:
                url = str(base_url).rstrip("/") + "/chat/completions"
            else:
                url, api_key = endpoints.get(
                    provider, (config.FIREWORKS_API_URL, config.FIREWORKS_API_KEY)
                )
        except Exception:
            url, api_key = endpoints.get(
                provider, (config.FIREWORKS_API_URL, config.FIREWORKS_API_KEY)
            )
    if not api_key:
        raise RuntimeError(
            "no API key configured for provider '%s'" % provider
        )
    return _call_openai_compatible(url, api_key, messages, tools, model=model, provider=provider)


def _model_turn_with_retries(history, tools, step=None, stats=None):
    """Model call with at most 2 retries on transport/HTTP errors.

    L-8: only RETRYABLE failures loop (transient HTTP statuses, transport
    errors, anything without a provider response). A deterministic
    rejection — an HTTP error whose status is outside _RETRYABLE_STATUS —
    breaks out after the first attempt with the provider's own message.

    Timing is observational only: every attempt is timed and logged as a
    STOPWATCH line, but the retry policy, the raised error and the model
    call itself are never affected by a timing hiccup.
    """
    last_error = None
    req_bytes = 0
    # BA-00: serializing the history IS the upload-size measurement — the
    # payload is dominated by base64 look images, so its byte count is the
    # "what did this turn cost to send" number.
    _ser = _Span("model.serialize")
    try:
        req_bytes = len(json.dumps(history, default=str))
    except Exception:
        req_bytes = 0
    _ser.done(stats, req_bytes=req_bytes)
    if stats is not None:
        try:
            stats.upload_bytes_total += req_bytes
        except Exception:
            pass
    total_t0 = None
    first_start = ""
    try:
        total_t0 = time.monotonic()
        first_start = _sw_timestamp()
    except Exception:
        pass
    for attempt in range(3):
        t0 = None
        start = ""
        try:
            t0 = time.monotonic()
            start = _sw_timestamp()
        except Exception:
            pass
        # BA-00: the whole attempt as one span. NOTE — there is deliberately
        # no `model.ttfb` span: the provider calls are NON-streaming POSTs
        # (headers and body arrive together), so time-to-first-byte is not
        # separable from the total. True TTFB needs streaming (a later item).
        _total = _Span("model.total")
        try:
            text, tool_calls = _model_turn(history, tools)
        except Exception as exc:
            last_error = exc
            dur_ms = _sw_elapsed_ms(t0)
            _total.done(stats, ok=False)
            # BA-03: a spent task budget is never retried — no slice of it
            # can succeed, so further attempts only burn wall-clock past
            # the deadline. Falls through to the final raise below.
            try:
                _handle = budget.resolve(None)
                if _handle is not None and _handle.stopped():
                    break
            except Exception:
                pass
            # L-8: a deterministic rejection (same payload fails identically)
            # breaks out immediately instead of burning two more full-image
            # uploads plus a second of sleeping.
            retryable = _is_retryable(exc)
            try:
                if attempt < 2:
                    append_activity_line(
                        "STOPWATCH model retry step=%s attempt=%d dur_ms=%d\n"
                        % (step, attempt, dur_ms)
                    )
                    if stats is not None:
                        stats.record_model(step, dur_ms, ok=False, retry=True)
            except Exception:
                pass
            if not retryable:
                try:
                    append_activity_line(
                        "STOPWATCH model deterministic step=%s attempt=%d "
                        "dur_ms=%d error=%s\n"
                        % (step, attempt, dur_ms,
                           _provider_message(exc)[:200] or type(exc).__name__)
                    )
                except Exception:
                    pass
                break
            if attempt < 2:
                # L-8: the provider's Retry-After wins over the flat gap.
                # BA-03: the wait is capped to the remaining task budget —
                # a spent budget stops retrying instead of sleeping past
                # the deadline.
                try:
                    _gap = _retry_delay_s(exc)
                except Exception:
                    _gap = 0.0
                try:
                    if not budget.wait(None, _gap):
                        break
                except Exception:
                    pass
            continue
        dur_ms = _sw_elapsed_ms(t0)
        _total.done(stats, ok=True)
        resp_bytes = 0
        try:
            resp_bytes = len(json.dumps(text or "", default=str)) + len(
                json.dumps(tool_calls or [], default=str))
            append_activity_line(
                "STOPWATCH model step=%s start=%s dur_ms=%d ok=true "
                "req_bytes=%d resp_bytes=%d\n"
                % (step, start, dur_ms, req_bytes, resp_bytes)
            )
            if stats is not None:
                stats.record_model(step, dur_ms, ok=True)
        except Exception:
            pass
        return text, tool_calls
    dur_total = _sw_elapsed_ms(total_t0)
    try:
        append_activity_line(
            "STOPWATCH model step=%s start=%s dur_ms=%d ok=false "
            "req_bytes=%d resp_bytes=0\n"
            % (step, first_start, dur_total, req_bytes)
        )
        if stats is not None:
            stats.record_model(step, dur_total, ok=False)
    except Exception:
        pass
    raise RuntimeError(_model_failure_message(last_error))


# ── L-8 / BA-02: retryable vs. deterministic model errors ──────────────────
# _model_turn_with_retries used to catch bare Exception and loop exactly 3
# attempts for EVERYTHING. A deterministic 400 (bad schema, context overflow,
# unsupported reasoning_effort, oversize image) therefore cost 3 full image
# uploads + 1.0 s of sleeping, and the operator saw only "model call failed".

#: HTTP statuses worth another attempt. Anything ELSE that arrives with a
#: provider response — notably 400 invalid_request — fails identically on
#: the same payload, so retrying only burns uploads.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: Longest a provider's Retry-After is honoured (seconds). A broken
#: `Retry-After: 3600` must not park the task for an hour.
_RETRY_AFTER_CAP_S = 10.0

#: Fallback gap between attempts when the provider names no wait.
_RETRY_GAP_S = 0.5


def _is_retryable(exc):
    """True when another attempt could plausibly succeed.

    Fail-fast applies ONLY to known-deterministic failures: an error
    carrying an HTTP response whose status is outside _RETRYABLE_STATUS.
    Anything without a response (transport errors, RuntimeErrors from the
    adapters) keeps today's retry behaviour — failing fast on unknowns
    would trade a measured waste for unmeasured fragility, and the suite
    pins bare RuntimeErrors as retried.
    """
    try:
        resp = getattr(exc, "response", None)
        if resp is None:
            return True
        return int(getattr(resp, "status_code", 0)) in _RETRYABLE_STATUS
    except Exception:
        return True


def _retry_after_s(exc):
    """Seconds the provider asked us to wait (Retry-After), capped at
    _RETRY_AFTER_CAP_S. 0 when absent, unparsable, or past (HTTP-date)."""
    try:
        resp = getattr(exc, "response", None)
        headers = getattr(resp, "headers", None) if resp is not None else None
        raw = headers.get("Retry-After") if headers else None
        if raw is None:
            return 0.0
        raw = str(raw).strip()
        try:
            wait = float(raw)
        except ValueError:
            # HTTP-date form: wait until then, never backwards.
            try:
                from email.utils import parsedate_to_datetime
                target = parsedate_to_datetime(raw).timestamp()
                wait = target - time.time()
            except Exception:
                return 0.0
        if wait <= 0:
            return 0.0
        return min(wait, _RETRY_AFTER_CAP_S)
    except Exception:
        return 0.0


def _provider_message(exc):
    """The provider's own error text, best-effort (truncated).

    `raise_for_status()` keeps only status+URL in str(exc); the body — the
    part that says WHICH schema field or limit broke — is what the operator
    needs, and it used to be discarded by the retry loop's RuntimeError.
    """
    try:
        resp = getattr(exc, "response", None)
        if resp is None:
            return ""
        try:
            body = resp.json()
        except Exception:
            body = None
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict) and err.get("message"):
                text = str(err["message"])
            elif isinstance(body.get("message"), str):
                text = body["message"]
            else:
                text = json.dumps(body, default=str)
        elif body is not None:
            text = str(body)
        else:
            try:
                text = resp.text or ""
            except Exception:
                text = ""
        text = " ".join(str(text).split())
        return text[:500]
    except Exception:
        return ""


def _model_failure_message(exc):
    """What the task raises with: the error plus the provider's own words."""
    try:
        base = str(exc) if exc is not None else "unknown model error"
    except Exception:
        base = "unknown model error"
    detail = _provider_message(exc)
    if detail and detail not in base:
        return "%s | provider said: %s" % (base, detail)
    return base


def _retry_delay_s(exc):
    """Gap before the next attempt: the provider's Retry-After when it
    names one (429/503), else the flat historical 0.5 s."""
    wait = _retry_after_s(exc)
    return wait if wait > 0 else _RETRY_GAP_S


# ── BA-00: turn-level instrumentation ───────────────────────────────────────
# The audit's central claim ("the staleness check wastes most turns") was an
# ESTIMATE with nothing in the codebase able to confirm or refute it. These
# spans and counters turn it into a measured number. They are observational
# only: a timing hiccup can never change a task's outcome, and every recorder
# below is exception-swallowing by construction.

#: Bounded per-task span map: at most this many distinct span names.
_SPAN_LIMIT = 64


class _Span:
    """One timed stage. ``done()`` records into the task stats and returns
    the elapsed ms so callers can use it without a second clock read."""

    __slots__ = ("name", "t0")

    def __init__(self, name):
        self.name = name
        self.t0 = time.monotonic()

    def done(self, stats=None, **fields):
        try:
            ms = int((time.monotonic() - self.t0) * 1000)
        except Exception:
            return 0
        if stats is not None:
            try:
                stats.record_span(self.name, ms, fields)
            except Exception:
                pass
        return ms


class _SwStats:
    """Per-task STOPWATCH state: totals plus the samples the end summary
    ranks. Created per task and threaded through the loop only - never
    module-global, so tasks cannot corrupt each other's numbers.

    BA-00 adds per-stage spans and the wasted-turn census on top of the
    original totals; every pre-existing key keeps its exact meaning."""

    def __init__(self):
        self.model_calls = 0
        self.total_model_ms = 0
        self.tool_calls = 0
        self.total_tool_ms = 0
        self.model_samples = []
        self.tool_samples = []
        # ── BA-00 census ──
        #: ``{span_name: [total_ms, count]}`` - bounded by _SPAN_LIMIT.
        self.spans = {}
        #: Model turns consumed (a step whose every attempt failed counts).
        self.model_turns = 0
        #: Turns whose ONLY tool outcomes were refusals / stale marks /
        #: policy blocks - the "wasted turn" census the audit lacked.
        self.wasted_turns = 0
        #: Stale-mark refusals attributable specifically to `mut` drift.
        self.stale_refusals = 0
        #: Stale-mark refusals for every OTHER reason (doc/dpr/url/frame/...),
        #: so `stale_refusals` can be read against its total.
        self.stale_refusals_other = 0
        #: Off-screen mark refusals.
        self.offscreen_refusals = 0
        #: Composite looks performed.
        self.look_count = 0
        #: Image bytes uploaded to the model this task.
        self.bytes_uploaded = 0
        #: Estimated image tokens uploaded (what those bytes actually bill).
        self.image_tokens_est = 0
        #: Full request-payload bytes across all model turns (history JSON,
        #: dominated by the same base64 images — the "what did turns cost to
        #: send" number, distinct from the image-only counters above).
        self.upload_bytes_total = 0
        # Per-turn wasted-outcome accounting, reset by begin_turn().
        self._turn_outcomes = 0
        self._turn_wasted = 0

    def record_model(self, step, dur_ms, ok, retry=False):
        """Accumulate one model attempt. Retried attempts add wall time but
        do not count as extra model calls (the step's line is the call)."""
        self.total_model_ms += dur_ms
        if retry:
            return
        self.model_calls += 1
        self.model_turns += 1
        self.model_samples.append((step, dur_ms, ok))

    def record_tool(self, name, dur_ms):
        self.tool_calls += 1
        self.total_tool_ms += dur_ms
        self.tool_samples.append((name, dur_ms))

    def record_span(self, name, dur_ms, fields=None):
        """Accumulate one stage timing. Bounded by ``_SPAN_LIMIT`` names so a
        pathological tool name cannot grow the dict without limit."""
        entry = self.spans.get(name)
        if entry is None:
            if len(self.spans) >= _SPAN_LIMIT:
                return
            entry = [0, 0]
            self.spans[name] = entry
        entry[0] += dur_ms
        entry[1] += 1

    def note_image_upload(self, n_bytes, tokens_est=0):
        try:
            self.bytes_uploaded += max(0, int(n_bytes))
            self.image_tokens_est += max(0, int(tokens_est))
        except Exception:
            pass

    # ── wasted-turn census ──
    # A turn is "wasted" when it advanced nothing: every tool it dispatched
    # came back refused. begin_turn() / record_outcome() / end_turn() bracket
    # one model step's dispatch loop.

    def begin_turn(self):
        self._turn_outcomes = 0
        self._turn_wasted = 0

    def record_outcome(self, wasted):
        self._turn_outcomes += 1
        if wasted:
            self._turn_wasted += 1

    def end_turn(self):
        if self._turn_outcomes and self._turn_wasted == self._turn_outcomes:
            self.wasted_turns += 1
        self._turn_outcomes = 0
        self._turn_wasted = 0

    def span_table(self):
        """``[(name, total_ms, count)]`` sorted by total ms, descending."""
        out = [(name, int(v[0]), int(v[1])) for name, v in self.spans.items()]
        out.sort(key=lambda r: r[1], reverse=True)
        return out


# ── BA-00: cross-task model-turn percentile reservoir ───────────────────────
# The audit needs p50/p90 model-turn latency ACROSS tasks (required later for
# hedged requests). Per-task samples cannot answer that, so a bounded,
# lock-guarded reservoir of each task's MEAN model turn is kept here.

_PERF_MAX_SAMPLES = 512
_perf_lock = threading.Lock()
_perf_turn_ms = []
_perf_tasks = 0


def _perf_record_task(stats):
    """Fold one finished task's numbers into the cross-task reservoir."""
    global _perf_tasks
    try:
        if stats is None:
            return
        turns = int(getattr(stats, "model_turns", 0) or 0)
        total = int(getattr(stats, "total_model_ms", 0) or 0)
        with _perf_lock:
            _perf_tasks += 1
            if turns <= 0:
                return
            # One sample per task: its MEAN model turn. A task's single
            # slowest turn would bias the percentile toward heavy tasks.
            _perf_turn_ms.append(total // turns)
            if len(_perf_turn_ms) > _PERF_MAX_SAMPLES:
                del _perf_turn_ms[:len(_perf_turn_ms) - _PERF_MAX_SAMPLES]
    except Exception:
        pass


def _percentile(sorted_values, pct):
    """Nearest-rank percentile over an already-sorted list. 0 for empty."""
    if not sorted_values:
        return 0
    try:
        idx = int(round((pct / 100.0) * (len(sorted_values) - 1)))
    except Exception:
        return 0
    idx = max(0, min(len(sorted_values) - 1, idx))
    return int(sorted_values[idx])


def browser_agent_perf_snapshot():
    """Cross-task browser-agent performance, for ``GET /latency``.

    Read-only and self-contained: it returns plain JSON-safe types and never
    raises, so a broken reservoir can never take down the latency endpoint.
    """
    try:
        with _perf_lock:
            samples = sorted(int(v) for v in _perf_turn_ms)
            tasks = int(_perf_tasks)
        return {
            "tasks_observed": tasks,
            "turn_samples": len(samples),
            "mean_turn_ms_p50": _percentile(samples, 50),
            "mean_turn_ms_p90": _percentile(samples, 90),
            "mean_turn_ms_min": samples[0] if samples else 0,
            "mean_turn_ms_max": samples[-1] if samples else 0,
        }
    except Exception:
        return {"tasks_observed": 0, "turn_samples": 0}


def _sw_timestamp():
    """Wall-clock HH:MM:SS.mmm start stamp for a STOPWATCH line."""
    try:
        now = datetime.datetime.now()
        return now.strftime("%H:%M:%S.") + "%03d" % (now.microsecond // 1000)
    except Exception:
        return ""


def _sw_elapsed_ms(t0):
    """Monotonic ms since t0; 0 when t0 is missing or timing fails."""
    if t0 is None:
        return 0
    try:
        return int((time.monotonic() - t0) * 1000)
    except Exception:
        return 0


def _sw_log_tool(name, start, t0, stats=None):
    """Emit one STOPWATCH tool line (before the RESULT line) and accumulate
    the totals. Never raises; a timing hiccup must not break the tool call."""
    try:
        dur_ms = _sw_elapsed_ms(t0)
        append_activity_line(
            "STOPWATCH tool=%s start=%s dur_ms=%d\n" % (name, start, dur_ms)
        )
        if stats is not None:
            stats.record_tool(name, dur_ms)
    except Exception:
        pass


def _emit_summary(stats, started):
    """End-of-task STOPWATCH block: totals, slowest model steps, slowest
    tools, the BA-00 census, and any deadline overshoot. Never raises and
    never changes the task's outcome."""
    try:
        total_ms = max(0, int((time.monotonic() - started) * 1000))
        append_activity_line(
            "STOPWATCH summary total_ms=%d model_calls=%d total_model_ms=%d "
            "tool_calls=%d total_tool_ms=%d\n"
            % (total_ms, stats.model_calls, stats.total_model_ms,
               stats.tool_calls, stats.total_tool_ms)
        )
        # BA-00: the census the audit needed and could not get. Every number
        # here is MEASURED, never estimated.
        stale_total = stats.stale_refusals + stats.stale_refusals_other
        append_activity_line(
            "STOPWATCH census model_turns=%d wasted_turns=%d "
            "wasted_pct=%d look_count=%d stale_refusals=%d "
            "stale_refusals_other=%d offscreen_refusals=%d\n"
            % (stats.model_turns, stats.wasted_turns,
               _wasted_pct(stats), stats.look_count, stats.stale_refusals,
               stats.stale_refusals_other, stats.offscreen_refusals)
        )
        append_activity_line(
            "STOPWATCH upload bytes_uploaded=%d image_tokens_est=%d "
            "upload_bytes_total=%d\n"
            % (stats.bytes_uploaded, stats.image_tokens_est,
               stats.upload_bytes_total)
        )
        for name, span_ms, count in stats.span_table()[:12]:
            append_activity_line(
                "STOPWATCH span %s total_ms=%d count=%d\n"
                % (name, span_ms, count))
        for step, dur_ms, _ok in sorted(
                stats.model_samples, key=lambda s: s[1], reverse=True)[:3]:
            append_activity_line(
                "STOPWATCH slowest model step=%s dur_ms=%d\n"
                % (step, dur_ms))
        for name, dur_ms in sorted(
                stats.tool_samples, key=lambda s: s[1], reverse=True)[:3]:
            append_activity_line(
                "STOPWATCH slowest tool %s dur_ms=%d\n"
                % (name, dur_ms))
        overshoot_ms = total_ms - config.BROWSER_AGENT_TIMEOUT * 1000
        if overshoot_ms > 0:
            append_activity_line("STOPWATCH overshoot_ms=%d\n" % overshoot_ms)
        _perf_record_task(stats)
    except Exception:
        pass


def _wasted_pct(stats):
    """Percentage of consumed turns that advanced nothing; 0 when no turns."""
    try:
        if not stats.model_turns:
            return 0
        return int(round(100.0 * stats.wasted_turns / stats.model_turns))
    except Exception:
        return 0


# ── Virtual (composite) tools ──────────────────────────────────────────────
# These are Python handlers that the model sees as tools but that
# browser_agent.py executes itself (one model step = several MCP calls).
#
# F13: the ADVERTISEMENT and the EXECUTION validation are generated from one
# declaration. Before this, the schema the model was shown and the checks the
# handler really applied were separate code paths that drifted: arguments the
# schema required were optional in practice, arguments it declared as one type
# were coerced from another, and a cross-field requirement (download needs a
# target, drag_drop needs two) existed nowhere. Now every spec carries its
# schema AND its own cross-field rule; `_run_one_tool` validates against the
# very schema object the model was shown.

def _spec(name, description, schema, check=None):
    return {"name": name, "description": description, "input_schema": schema,
            "check": check}


def _needs_one_of(verb, *field_names):
    """Cross-field rule: at least one of *field_names* must be supplied."""
    def _check(arguments):
        for field_name in field_names:
            value = arguments.get(field_name)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return ""
        return ("%s requires one of %s" % (verb, " or ".join(field_names)))
    return _check


def _drag_drop_check(arguments):
    """drag_drop needs a source AND a drop target (F13)."""
    error = _needs_one_of("drag_drop", "index", "css")(arguments)
    if error:
        return error
    error = _needs_one_of("drag_drop", "target_index", "target_css")(arguments)
    if error:
        return "drag_drop requires a drop target (%s)" % error
    return ""


#: F13: the cross-field rules the JSON schema cannot express. They live next
#: to the schemas (and are checked against them at import) so advertisement
#: and enforcement cannot drift apart again.
_VIRTUAL_ARG_CHECKS = {
    "wait_for": _needs_one_of("wait_for", "selector", "text"),
    "download": _needs_one_of("download", "index", "css"),
    "drag_drop": _drag_drop_check,
}


_VIRTUAL_TOOL_DEFS = [
    _spec(
        "look",
        "Take an annotated screenshot with numbered marks for every interactive element (form fields in view first). Returns a JPEG image (downscaled to width <=1280 when the viewport is wider) plus a table mark | tag | label | center x,y in LOOK-IMAGE pixels - the same frame click_point uses. Marks refer to the most recent look and become stale after navigation - call look again after any page change. Use click_mark / fill_mark on the mark numbers.",
        {"type": "object", "properties": {}, "required": []},
    ),
    _spec(
        "click_mark",
        "Click a numbered visual mark from the most recent look with REAL mouse input (click_locator) after re-resolving the mark's target. Input index is the 1-based mark number. If the mark is missing or stale (page navigated, reloaded, reflowed or SPA-updated since the look), the tool returns an error telling you to call look again. An off-screen mark is scrolled into view automatically inside the same call (reported as scrolled: true) - never scroll manually first. Returns {clicked, via, url, href_before, navigated} - navigated true means the click changed the page, so look again before acting. Synthetic clicks cannot click inside cross-origin iframes/embeds - if the mark sits over an embedded player, click_point on the player area or judge playback from the screenshot instead. After navigation or a missed click, call look again.",
        {
            "type": "object",
            "properties": {"index": {"type": "integer", "description": "1-based mark number from the most recent look"}},
            "required": ["index"],
        },
    ),
    _spec(
        "fill_mark",
        "Fill the text input at a numbered visual mark from the most recent look with REAL keyboard input (fill_locator, one explicit submit channel: press_enter true sends Enter, omit it for no submit), in one step. Prefer this over fill whenever the input appears in the look marks. If the mark is missing or stale (page navigated, reloaded or updated since the look), returns an error telling you to call look again. An off-screen input is scrolled into view automatically (reported as scrolled: true). If the mark is not a text input, returns an error naming the element's tag so you can recover in one step. Returns {ok, via, css, url, navigated} on the real-input path.",
        {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "1-based mark number of the input from the most recent look"},
                "value": {"type": "string", "description": "Value to fill"},
                "press_enter": {"type": "boolean", "description": "Whether to press Enter after filling"},
            },
            "required": ["index", "value"],
        },
    ),
    _spec(
        "batch_probe",
        "Run several read-only DOM probes in ONE step instead of many evaluate calls. Probes run in the TOP DOCUMENT ONLY - they cannot read inside iframes (especially cross-origin players). Input expressions is an array of typed DOM read expressions (max 10, each <=400 chars): document.querySelector(...)/getElementById(...)/location.* and reading innerText, textContent, value, href, src, getAttribute(...), length, etc. No assignment, no function calls beyond the read-only getter allowlist, no fetch/eval/cookies. Returns an array of {expr, ok: result} or {expr, error} as JSON.",
        {
            "type": "object",
            "properties": {
                "expressions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "JS expression strings to evaluate",
                }
            },
            "required": ["expressions"],
        },
    ),
    _spec(
        "wait_for",
        "Wait for async page updates (search results, SPA loads) in one step. Checks the TOP DOCUMENT ONLY - it cannot see text or selectors inside iframes (a cross-origin player's timecode will never appear here). Event-driven on current daemons (one round trip, returns the instant the condition holds; Python polling fallback on older ones). Waits up to timeout_ms (default 5000, hard cap 10000) for selector (querySelector) or text (body innerText contains) to appear. Supply at least one of selector/text: an empty wait verifies nothing and is refused. Returns {found: true/false, url, title}.",
        {
            "type": "object",
            "properties": {
                "selector": {"type": "string", "description": "CSS selector to wait for"},
                "text": {"type": "string", "description": "Text substring to wait for"},
                "timeout_ms": {"type": "integer", "description": "Timeout in ms, default 5000, max 10000"},
            },
            "required": [],
        },
    ),
    _spec(
        "fill",
        "Fill a text input by CSS selector or placeholder text with REAL keyboard input (fill_locator) and optionally submit, in one step. Among querySelectorAll matches picks the first VISIBLE one; if the selector misses, scans every visible text input and picks the best case-insensitive placeholder/name/aria-label/id match; if nothing matches, the error lists every visible input and its placeholder so you can recover in one step. press_enter true uses the daemon's ONE explicit submit channel (Enter); nothing else submits the form. A disabled input is refused instead of silently doing nothing. Prefer fill_mark when the input has a look mark. Returns {ok, via, css, url, navigated}.",
        {
            "type": "object",
            "properties": {
                "selector": {"type": "string", "description": "CSS selector or placeholder text"},
                "value": {"type": "string", "description": "Value to fill"},
                "press_enter": {"type": "boolean", "description": "Whether to press Enter after filling"},
            },
            "required": ["selector", "value"],
        },
    ),
    _spec(
        "click_text",
        "Click a button/link by its visible text without first discovering its index. Finds visible clickable elements (a, button, [role=button], [onclick]) whose trimmed text matches exactly, else contains case-insensitive; resolves the first match to a real element identity and clicks it with REAL mouse input (click_locator), including inside a same-origin iframe. A disabled element is refused, never silently clicked. Returns {clicked, via, url, href_before, navigated} - navigated true means the click changed the page, so look again before acting.",
        {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Visible text of the button/link to click"}},
            "required": ["text"],
        },
    ),
    _spec(
        "click_point",
        "Click at coordinates in the LOOK-IMAGE pixel frame - the same frame as the annotated look screenshot and its mark table. The agent converts them to viewport coordinates automatically, so never scale coordinates yourself. The coordinates are anchored to the page state of the last look: if the page navigated, reloaded, changed document epoch or the display scale changed since then, the click is REFUSED and you must look again (this is what stops stale coordinates from hitting whatever now sits there). The element at the point is then clicked with REAL mouse input (click_locator), including inside a same-origin iframe; a disabled element or a throwing click is reported as a failure, never as a click. Synthetic clicks cannot reach inside a cross-origin iframe/embed: if the target is an embedded player, do not hunt its internal buttons - click the page's own overlay at most, look again after each player click and re-assess, then judge playback from the look screenshot or verify_playing. Returns {clicked, via, url, href_before, navigated}.",
        {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "Look-image x coordinate"},
                "y": {"type": "integer", "description": "Look-image y coordinate"},
            },
            "required": ["x", "y"],
        },
    ),
    _spec(
        "verify_playing",
        "Verify that video/media is actually playing when the player is inside a cross-origin iframe the DOM cannot read. Takes two screenshots ~1.2s apart and compares pixels: returns PLAYING when the content is animating, STATIC when there is no motion. Call at most ONCE when the task goal is media playback, then finish with your summary.",
        {"type": "object", "properties": {}, "required": []},
    ),
    _spec(
        "scroll",
        "Scroll the page (or a scrollable element) with REAL mouse-wheel input. Direction up|down|left|right, amount in px (default 600). Use when the target is below the fold instead of guessing coordinates; after scrolling call look again to refresh marks.",
        {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "description": "up, down (default), left or right"},
                "amount": {"type": "integer", "description": "Wheel delta in px, default 600"},
                "css": {"type": "string", "description": "CSS selector of a scrollable element; omit for the page"},
            },
            "required": [],
        },
    ),
    _spec(
        "select_option",
        "Select an <option> in a <select> dropdown with REAL input — by mark index (look table) or CSS selector. The value is matched against the option value and label. The selection is READ BACK from the control afterwards: the reported value is the one actually in effect, and a selection that did not take effect is a failure, never a success. Returns the verified value and the real after-state (url/navigated).",
        {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "Mark number of the <select> from look"},
                "css": {"type": "string", "description": "CSS selector of the <select> (used when index is omitted)"},
                "value": {"type": "string", "description": "Option value or label to select"},
            },
            "required": ["value"],
        },
    ),
    _spec(
        "set_checked",
        "Check or uncheck a checkbox/radio with REAL input — by mark index (look table) or CSS selector. The control's checked state is READ BACK afterwards: the result states the state actually in effect, and a change that did not take effect is a failure, never a success. Returns the verified checked state and the real after-state.",
        {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "Mark number of the checkbox/radio from look"},
                "css": {"type": "string", "description": "CSS selector (used when index is omitted)"},
                "checked": {"type": "boolean", "description": "true to check, false to uncheck"},
            },
            "required": ["checked"],
        },
    ),
    _spec(
        "upload_file",
        "Upload file(s) through a file <input> with REAL input — by mark index (look table) or CSS selector. BOTH the file and the destination must be approved: the path MUST already exist and be inside a granted workspace (repo, cwd or temp), AND the page's current origin must be approved for uploads in this run (session upload_origins, an `upload_origin:<origin>` grant, or JARVIS_BROWSER_UPLOAD_ORIGINS). An unapproved destination is refused BEFORE anything is dispatched, so a readable file cannot be disclosed to an arbitrary site. Returns the uploaded filenames.",
        {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "Mark number of the file input from look"},
                "css": {"type": "string", "description": "CSS selector (used when index is omitted)"},
                "path": {"type": "string", "description": "Local file path to upload (single file)"},
            },
            "required": ["path"],
        },
    ),
    _spec(
        "download",
        "Click a download affordance with REAL input and SAVE the file locally — supply a mark index or a CSS selector. Waits up to 30s for the browser download event. A download is only reported as complete when a REAL artifact exists on disk: the daemon's saved path is required and is checked, so a click that produced no file is reported as a failure with no artifact. Returns the on-disk artifact path and filename so you can cite it.",
        {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "Mark number of the download link/button from look"},
                "css": {"type": "string", "description": "CSS selector (used when index is omitted)"},
            },
            "required": [],
        },
    ),
    _spec(
        "drag_drop",
        "Drag one element onto another with REAL mouse input — by mark indices (look table) or CSS selectors. Use for sliders, kanban cards, file-drop zones and range handles.",
        {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "Mark number of the element to drag"},
                "target_index": {"type": "integer", "description": "Mark number of the drop target"},
                "css": {"type": "string", "description": "CSS selector of the source (used when index is omitted)"},
                "target_css": {"type": "string", "description": "CSS selector of the drop target"},
            },
            "required": [],
        },
    ),
]

# F13: ONE declaration drives both the advertisement the model sees and the
# validation applied when the call comes back. `schema` is the very object
# handed to the provider, so a handler can never accept something the model
# was not told about, and a required argument can never be "optional in
# practice".
_VIRTUAL_TOOL_SPECS = [
    dict(entry, check=_VIRTUAL_ARG_CHECKS.get(entry["name"]))
    for entry in _VIRTUAL_TOOL_DEFS
]
_VIRTUAL_TOOL_SPECS_BY_NAME = {entry["name"]: entry for entry in _VIRTUAL_TOOL_SPECS}
unknown_checks = sorted(set(_VIRTUAL_ARG_CHECKS) - set(_VIRTUAL_TOOL_SPECS_BY_NAME))
if unknown_checks:  # pragma: no cover - import-time guard, never in a good build
    raise RuntimeError("argument rules for unknown virtual tools: %s"
                       % ", ".join(unknown_checks))

_VIRTUAL_TOOL_NAMES = set(_VIRTUAL_TOOL_SPECS_BY_NAME)

# BA-07: the system prompt must never name a tool the model was not
# offered. Naming a phantom tool reads like an escape hatch (tried, then
# policy-blocked — a burned turn); advertising-then-forbidding is worse
# still. Mirrors the F13 import-time guard above.
# NOTE: "screenshot" is deliberately NOT in this set — the prompt uses it
# as plain English ("annotated screenshot"), and it was never advertised
# as a callable tool, so matching the bare word would false-positive.
_PROMPT_TOOL_MENTIONS = set(
    re.findall(r"\b([a-z][a-z_]{3,24})\b", _SYSTEM_PROMPT))
_PHANTOM_TOOL_MENTIONS = {"understand_page", "click_element", "fill_element",
                          "evaluate"} & _PROMPT_TOOL_MENTIONS
if _PHANTOM_TOOL_MENTIONS:  # pragma: no cover - import-time guard, never in a good build
    raise RuntimeError("system prompt names non-advertised tools: %s"
                       % sorted(_PHANTOM_TOOL_MENTIONS))


def _virtual_schema(name):
    """The advertised schema of one virtual tool (same object, not a copy)."""
    entry = _VIRTUAL_TOOL_SPECS_BY_NAME.get(str(name or ""))
    return entry["input_schema"] if entry else None


def _virtual_argument_error(name, arguments):
    """Cross-field argument rule for one virtual tool ("" when acceptable)."""
    entry = _VIRTUAL_TOOL_SPECS_BY_NAME.get(str(name or ""))
    check = entry.get("check") if entry else None
    if check is None:
        return ""
    try:
        return str(check(arguments if isinstance(arguments, dict) else {}) or "")
    except Exception:
        # A broken rule must not open the gate: refuse the call instead.
        return "argument validation failed for %s" % name


def _screenshot_has_content(path):
    """True when the screenshot file exists AND actually holds bytes.

    ``tempfile.mkstemp`` already created an EMPTY file, so existence alone
    never proved the daemon wrote anything (F38).
    """
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def _write_inline_screenshot(client, path):
    """F38: materialise an inline screenshot the daemon returned (True/False).

    The MCP client preserves non-text content in ``last_images``; the older
    behavior reduced an image block to the literal marker "[image omitted]",
    so an inline capture was silently unusable.
    """
    images = getattr(client, "last_images", None)
    if not images:
        return False
    import base64 as _base64

    for image in images:
        data = (image or {}).get("data")
        if not isinstance(data, str) or not data.strip():
            continue
        try:
            raw = _base64.b64decode(data, validate=False)
        except Exception:
            continue
        if not raw:
            continue
        try:
            with open(path, "wb") as handle:
                handle.write(raw)
        except Exception:
            return False
        return True
    return False


def _probe(client, expression, attempts=2, timeout=8.0):
    """BA-06: read-only evaluate with a short timeout and one fast retry.

    The virtual handlers used to call ``client.call_tool("evaluate", ...)``
    directly, so one dropped connection turned a read-only probe into a
    failed look / failed target check — costing a full model turn to
    recover. Retrying is safe here because these are read-only probes
    (``tool_policy.probe_expression_error`` already guarantees no
    mutation), so this never violates the single-attempt-for-mutations
    invariant that keeps model-issued ``evaluate`` at ``max_attempts = 1``.

    The ``timeout`` kwarg needs the L-15 per-call client; older or test
    clients without it fall back to the client's default timeout.
    """
    last = None
    for i in range(attempts):
        try:
            try:
                return client.call_tool("evaluate",
                                        {"expression": expression},
                                        timeout=timeout)
            except TypeError:
                return client.call_tool("evaluate",
                                        {"expression": expression})
        except Exception as exc:
            last = exc
            if i + 1 < attempts:
                time.sleep(0.05)
    raise last


def _handle_look(client, session, stats=None):
    """Composite look: evaluate -> screenshot -> downscale -> annotate -> JPEG (reordered, no threads)."""
    tmp_path = None
    # BA-00: every stage below is timed separately so a slow look can be
    # attributed (daemon round trip vs local image work) instead of guessed.
    if stats is not None:
        try:
            stats.look_count += 1
        except Exception:
            pass
    try:
        fd, tmp_path = tempfile.mkstemp(prefix="jarvis_", suffix=".png")
        os.close(fd)
        # 1. evaluate for elements + url/title in ONE call (before screenshot - no image needed).
        # Inventory is prioritized: in-viewport form fields first, then other
        # in-viewport elements, then off-screen ones - a nav-heavy header
        # can no longer crowd the 40-mark cap before the search box is seen.
        js = (
            "(() => {"
            "const selectors = 'a, button, input, select, textarea, [role=\"button\"], [onclick], [contenteditable], video, iframe, [tabindex]';"
            # F39: a document epoch — timeOrigin changes on ANY navigation
            # (including same-URL reloads); the MutationObserver counter
            # changes on structural DOM updates. Together they catch what URL
            # equality cannot. cssPath gives every mark a real element
            # identity to re-resolve against later.
            # BA-05: the observer watches direct-child structure ONLY (no
            # subtree, no attributes) — class toggles, aria-live updates,
            # lazy images and spinner animations were pure counter noise for
            # a signal that is now just a tolerance hint, not a refusal.
            "if (!window.__jarvisEpoch) { window.__jarvisEpoch = {mut: 0};"
            " try { new MutationObserver(ms => { window.__jarvisEpoch.mut += ms.length; })"
            " .observe(document.documentElement, {childList: true}); } catch(e) {} }"
            "const epochDoc = (performance && performance.timeOrigin) ? Math.round(performance.timeOrigin) : 0;"
            # F39: 0 is a REAL epoch value (the first mutation count, a
            # timeOrigin that rounds to zero), so it is reported explicitly
            # instead of being coerced into "unknown" by `0 || ""`.
            "const epochMut = (window.__jarvisEpoch && typeof window.__jarvisEpoch.mut === 'number') ? window.__jarvisEpoch.mut : 0;"
            # F39: the display scale the screenshot geometry depends on. A
            # DPR change after the look invalidates both the mark rects and
            # the click_point conversion, so it is part of the identity.
            "const dpr = (typeof devicePixelRatio === 'number' && devicePixelRatio > 0) ? devicePixelRatio : 1;"
            "const cssPath = (el) => { try {"
            " let path = ''; let n = el;"
            " while (n && n.nodeType === 1 && path.length < 220) {"
            "  let seg = n.tagName.toLowerCase();"
            "  if (n.id) { path = '#' + n.id + (path ? '>' + path : ''); break; }"
            "  let i = 1; let sib = n;"
            "  while ((sib = sib.previousElementSibling)) { if (sib.tagName === n.tagName) i++; }"
            "  seg += ':nth-of-type(' + i + ')';"
            "  path = seg + (path ? '>' + path : '');"
            "  n = n.parentElement; }"
            " return path; } catch(e) { return ''; } };"
            "const els = Array.from(document.querySelectorAll(selectors));"
            "const formEls = []; const mediaEls = []; const inVp = []; const offVp = [];"
            "for (const el of els) {"
            "  const rect = el.getBoundingClientRect();"
            "  if (rect.width < 12 || rect.height < 12) continue;"
            "  if (el.getClientRects().length === 0) continue;"
            "  if (el.offsetParent === null) {"
            "    const st = window.getComputedStyle(el);"
            "    if (st.position !== 'fixed' && st.position !== 'sticky') continue;"
            "  }"
            "  const inView = !(rect.bottom < 0 || rect.top > window.innerHeight || rect.right < 0 || rect.left > window.innerWidth);"
            "  if (!inView) { offVp.push(el); continue; }"
            "  const tag = el.tagName.toLowerCase();"
            "  if (tag === 'input' || tag === 'textarea' || tag === 'select' || el.hasAttribute('contenteditable')) formEls.push(el);"
            "  else if (tag === 'video' || tag === 'iframe') mediaEls.push(el);"
            "  else inVp.push(el);"
            "}"
            "const out = [];"
            "for (const el of formEls.concat(mediaEls, inVp, offVp)) {"
            "  if (out.length >= 40) break;"
            "  const rect = el.getBoundingClientRect();"
            "  let label = (el.innerText || '').trim();"
            "  if (!label) label = (el.getAttribute('aria-label') || '').trim();"
            "  if (!label) label = (el.getAttribute('placeholder') || '').trim();"
            "  if (!label) label = (el.getAttribute('title') || '').trim();"
            "  label = label.slice(0,25);"
            "  out.push({x: Math.round(rect.left), y: Math.round(rect.top), w: Math.round(rect.width), h: Math.round(rect.height), tag: el.tagName.toLowerCase(), label, inView: !(rect.bottom < 0 || rect.top > window.innerHeight || rect.right < 0 || rect.left > window.innerWidth),"
            "   cssPath: cssPath(el), epoch: {doc: epochDoc, mut: epochMut}, dpr: dpr});"
            "}"
            "return JSON.stringify({elements: out, url: location.href, title: document.title, epoch: {doc: epochDoc, mut: epochMut}, dpr: dpr});"
            "})()"
        )
        try:
            _sp = _Span("look.evaluate")
            raw = _probe(client, js)
            _sp.done(stats, marks=len(js))
        except Exception as exc:
            return _clip_result("look failed: evaluate error: %s" % exc), None
        # parse
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(data, str):
                data = json.loads(data)
        except Exception:
            # try to find JSON substring
            try:
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1:
                    data = json.loads(raw[start:end+1])
                else:
                    raise ValueError("no JSON")
            except Exception as exc:
                return _clip_result("look failed: cannot parse elements: %s" % (raw[:500] if isinstance(raw, str) else str(raw))), None
        elements = data.get("elements", []) if isinstance(data, dict) else []
        url = data.get("url", "") if isinstance(data, dict) else ""
        title = data.get("title", "") if isinstance(data, dict) else ""
        elements = elements[:40]
        # F39: the epoch/DPR the INVENTORY was taken at. Explicitly stringified
        # so a legitimate zero ("0") stays distinguishable from "unknown".
        observed_doc, observed_mut = _mark_epoch_of(data if isinstance(data, dict) else {})
        observed_dpr = _epoch_text(data.get("dpr")) if isinstance(data, dict) else ""
        # G6 (F39 + F47): correlate the observed page with a real daemon tab
        # id, best-effort — unknown tabs keep "" and no identity is guessed.
        # The same id lands on every mark (see step 4).
        look_tab_id = _current_tab_id(client, url, stats)
        _publish_look_tab(url, title, look_tab_id)
        # 2. screenshot (after evaluate - evaluate needs no image)
        try:
            _sp = _Span("look.screenshot")
            client.call_tool("screenshot", {"path": tmp_path})
            _sp.done(stats)
        except Exception as exc:
            return _clip_result("look failed: screenshot error: %s" % exc), None
        if not _screenshot_has_content(tmp_path):
            # F38: a daemon that returns the capture INLINE instead of writing
            # `path` used to make every visual step fail here ("did not create
            # file") even though the image had arrived. Use the preserved
            # inline image instead of dropping it on the floor.
            if not _write_inline_screenshot(client, tmp_path):
                return _clip_result("look failed: screenshot did not create file"), None
        # 2b. F39: the inventory and the capture are two separate daemon calls.
        # Confirm the page did not navigate/reload or change its display scale
        # in between, or the marks would describe a page the image no longer
        # shows (and click_point's coordinate frame would be wrong).
        _sp = _Span("look.capture_state")
        capture = _capture_state(client)
        _sp.done(stats)
        capture_verified = capture is not None
        if capture is not None:
            capture_doc = _epoch_text(capture.get("doc"))
            capture_dpr = _epoch_text(capture.get("dpr"))
            changed = (
                (observed_doc and capture_doc and observed_doc != capture_doc)
                or (observed_dpr and capture_dpr and observed_dpr != capture_dpr)
                or (url and capture.get("url") and str(capture.get("url")) != url)
            )
            if changed:
                return _clip_result(
                    "look failed: the page changed while the screenshot was "
                    "being taken, so the marks and the image would not match - "
                    "call look again."), None
        # 3. read PNG and downscale using config width
        _sp = _Span("look.pil")
        try:
            img = Image.open(tmp_path).convert("RGB")
        except Exception as exc:
            return _clip_result("look failed: cannot read screenshot: %s" % exc), None
        orig_w, orig_h = img.size
        try:
            look_width = int(getattr(config, "BROWSER_AGENT_LOOK_WIDTH", 1280))
        except Exception:
            look_width = 1280
        look_width = max(640, min(1920, look_width))
        scale = 1.0
        if orig_w > look_width:
            scale = look_width / float(orig_w)
            new_w = look_width
            new_h = max(1, int(round(orig_h * scale)))
            try:
                img_small = img.resize((new_w, new_h), Image.LANCZOS)
            except Exception:
                img_small = img.resize((new_w, new_h))
        else:
            img_small = img
        # 4. session marks in ORIGINAL viewport pixels (used internally by
        # click_mark / fill_mark) + scale/size so click_point can convert
        # look-image coordinates to viewport coordinates server-side.
        marks = {}
        for idx, el in enumerate(elements, start=1):
            try:
                x = int(el.get("x", 0)); y = int(el.get("y", 0)); w = int(el.get("w", 0)); h = int(el.get("h", 0))
            except Exception:
                x = y = w = h = 0
            cx = int(x + w // 2) if w else int(x)
            cy = int(y + h // 2) if h else int(y)
            marks[idx] = {"cx": cx, "cy": cy, "tag": el.get("tag", ""), "label": el.get("label", ""),
                          "inView": el.get("inView", True),
                          # G6 / F39: real target identity bound to the mark —
                          # the re-resolvable cssPath, the document epoch at
                          # observation time, the display scale, the observed
                          # bounds, the page URL, the tab this page belongs to
                          # and the frame that owns the element ("" = top
                          # document). click_mark / fill_mark / the typed tools
                          # validate all of these before acting.
                          "cssPath": (el.get("cssPath") or ""),
                          "epoch_doc": _epoch_text((el.get("epoch") or {}).get("doc")),
                          "epoch_mut": _epoch_text((el.get("epoch") or {}).get("mut")),
                          "dpr": _epoch_text(el.get("dpr")) or observed_dpr,
                          "frame": (el.get("frame") or ""),
                          "rect": {"x": x, "y": y, "w": w, "h": h},
                          "url": url,
                          "tab_id": look_tab_id,
                          "capture_verified": capture_verified}
        session["marks"] = marks
        session["look_scale"] = scale
        session["viewport_size"] = (orig_w, orig_h)
        # F39: the observation this look anchored — click_point checks the page
        # against it before converting its coordinates, so stale coordinates
        # cannot be aimed at a reloaded/renavigated page.
        session["look_capture"] = {
            "url": url,
            "doc": observed_doc,
            "mut": observed_mut,
            "dpr": observed_dpr,
            "tab_id": look_tab_id,
            "capture_verified": capture_verified,
        }
        # 5. draw overlay: green outline + filled badge
        try:
            # ensure we can draw with alpha even on RGB
            if img_small.mode != "RGBA":
                # use RGB drawing; colors without alpha
                draw = ImageDraw.Draw(img_small)
                outline_color = (0, 255, 0)
                badge_fill = (0, 200, 0)
                text_fill = (255, 255, 255)
            else:
                draw = ImageDraw.Draw(img_small, "RGBA")
                outline_color = (0, 255, 0, 255)
                badge_fill = (0, 200, 0, 255)
                text_fill = (255, 255, 255, 255)
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except Exception:
                font = ImageFont.load_default()
            for idx, el in enumerate(elements, start=1):
                x = int(el.get("x", 0)); y = int(el.get("y", 0)); w = int(el.get("w", 0)); h = int(el.get("h", 0))
                x_s = int(round(x * scale))
                y_s = int(round(y * scale))
                w_s = int(round(w * scale))
                h_s = int(round(h * scale))
                # clamp
                if w_s < 1:
                    w_s = 1
                if h_s < 1:
                    h_s = 1
                draw.rectangle([x_s, y_s, x_s + w_s, y_s + h_s], outline=outline_color, width=2)
                badge_text = str(idx)
                try:
                    bbox = font.getbbox(badge_text)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                except AttributeError:
                    tw, th = font.getsize(badge_text)  # type: ignore
                pad = 2
                # badge background
                draw.rectangle([x_s, y_s, x_s + tw + pad * 2, y_s + th + pad * 2], fill=badge_fill)
                draw.text((x_s + pad, y_s + pad), badge_text, fill=text_fill, font=font)
        except Exception:
            pass
        # 6. JPEG quality from config
        _sp.done(stats, out_w=img_small.size[0], out_h=img_small.size[1],
                 scaled=scale < 1.0)
        _sp = _Span("look.jpeg")
        if img_small.mode == "RGBA":
            img_small = img_small.convert("RGB")
        try:
            jpeg_quality = int(getattr(config, "BROWSER_AGENT_JPEG_QUALITY", 70))
        except Exception:
            jpeg_quality = 70
        jpeg_quality = max(40, min(95, jpeg_quality))
        buf = io.BytesIO()
        img_small.save(buf, format="JPEG", quality=jpeg_quality)
        jpeg_bytes = len(buf.getvalue())
        _sp.done(stats, jpeg_bytes=jpeg_bytes)
        _sp = _Span("look.b64")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        _sp.done(stats, b64_chars=len(b64))
        # BA-00: what this observation actually costs to send. Tokens are the
        # standard 750px-per-token estimate for a downscaled screenshot.
        if stats is not None:
            try:
                px = max(1, img_small.size[0] * img_small.size[1])
                stats.note_image_upload(len(b64), px // 750)
            except Exception:
                pass
        # 7. text part
        img_w, img_h = img_small.size
        lines = []
        lines.append("Page: %s" % url)
        lines.append("Title: %s" % title)
        lines.append("")
        if scale < 1.0:
            lines.append(
                "Coordinate frame: this look image is %dx%d px, the real viewport is %dx%d px. "
                % (img_w, img_h, orig_w, orig_h)
            )
            lines.append(
                "All x,y below are LOOK-IMAGE pixels. click_point also takes look-image pixels - "
                "the agent converts them to viewport coordinates automatically, so NEVER scale coordinates yourself."
            )
        else:
            lines.append(
                "Coordinate frame: this look image matches the viewport exactly (1:1). "
                "All x,y below are look-image pixels, which are also viewport pixels; click_point takes the same frame."
            )
        lines.append("")
        lines.append("Marks from most recent look (%d). Marks become stale after navigation - call look again after any page change. Mark clicks/fills are re-validated against the live page; a stale mark is refused." % len(elements))
        lines.append("")
        if not elements:
            lines.append("No interactive elements found.")
        else:
            lines.append("mark | tag | label | center x,y (look-image pixels)")
            for idx, el in enumerate(elements, start=1):
                x = int(el.get("x", 0)); y = int(el.get("y", 0)); w = int(el.get("w", 0)); h = int(el.get("h", 0))
                cx = int(x + w // 2) if w else int(x)
                cy = int(y + h // 2) if h else int(y)
                cx_img = int(round(cx * scale))
                cy_img = int(round(cy * scale))
                label = (el.get("label", "") or "")[:25]
                tag = el.get("tag", "") or ""
                # escape pipe in label
                label = label.replace("|", "/")
                suffix = "" if el.get("inView", True) else " [off-screen]"
                lines.append("%d | %s | %s | %d,%d%s" % (idx, tag, label, cx_img, cy_img, suffix))
        text = "\n".join(lines)
        text = _clip_result(text)
        return text, b64
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# ── G6 / F39: mark identity ───────────────────────────────────────────────
# Every mark carries a real target identity — tab id, document epoch,
# observed bounds, frame identity and a re-resolvable locator — rather than
# bare pixels. _handle_look stores all five; _mark_target_state() validates
# them before ANY click or fill. Marks without a cssPath (fallback parsers,
# old sessions) keep the legacy coordinate path.
_MARK_RECT_TOLERANCE = 12  # px: reflow slop before a mark counts as moved


def _stale_on_mutation():
    """Legacy kill-switch (BA-05): JARVIS_BROWSER_STALE_ON_MUTATION=1 restores
    the pre-BA-05 refusal on ANY mutation-epoch drift. Default (unset) is the
    hint policy: drift only widens the rect tolerance."""
    return str(os.getenv("JARVIS_BROWSER_STALE_ON_MUTATION", "0")).strip().lower() in (
        "1", "true", "yes", "on")


def _epoch_text(value):
    """One epoch/DPR component as an EXPLICIT string (F39).

    ``0`` is a real value — the first mutation count after a look, or a
    ``timeOrigin`` that rounds to zero — but ``value or ""`` turned it into
    "missing", and a missing component silently SKIPPED the staleness
    comparison. That is how "the first mutation since the look" (and a
    same-URL reload into a fresh document whose counter restarted at 0) went
    undetected. Only None/"" mean unknown now.
    """
    if value is None or value == "" or isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    if number == int(number):
        return str(int(number))
    return ("%f" % number).rstrip("0").rstrip(".")


def _mark_epoch_of(stored):
    """(doc_epoch, mut_epoch) of a stored mark or a live ``epoch`` dict.

    The look inventory stores the pieces flat (``epoch_doc`` / ``epoch_mut``)
    while the page probes report them nested (``epoch: {doc, mut}``) — accept
    both, plus the daemon's own ``doc_epoch`` / ``mut_epoch`` aliases. Every
    component goes through :func:`_epoch_text`, so a zero survives.
    """
    if not isinstance(stored, dict):
        return "", ""
    epoch = stored.get("epoch") if isinstance(stored.get("epoch"), dict) else None
    if epoch is not None:
        return _epoch_text(epoch.get("doc")), _epoch_text(epoch.get("mut"))
    doc = stored.get("epoch_doc", "")
    mut = stored.get("epoch_mut", "")
    if doc in ("", None):
        doc = stored.get("doc_epoch", stored.get("doc", ""))
    if mut in ("", None):
        mut = stored.get("mut_epoch", stored.get("mut", ""))
    return _epoch_text(doc), _epoch_text(mut)


#: F39: one synchronous read of the state the screenshot geometry depends on.
#: Deliberately does NOT touch the mutation-observer global (a probe must be
#: side-effect free) — document identity, URL and display scale are what can
#: invalidate a capture.
_CAPTURE_STATE_JS = (
    "(() => {"
    "const doc = (typeof performance !== 'undefined' && performance.timeOrigin)"
    " ? Math.round(performance.timeOrigin) : null;"
    "const dpr = (typeof devicePixelRatio === 'number' && devicePixelRatio > 0)"
    " ? devicePixelRatio : null;"
    "return JSON.stringify({doc: doc, dpr: dpr, url: location.href,"
    " title: document.title});"
    "})()"
)


def _capture_state(client):
    """F39: page identity/scale right now, or None when unreadable."""
    try:
        raw = _probe(client, _CAPTURE_STATE_JS)
    except Exception:
        return None
    payload = _parse_json_result(raw)
    if not isinstance(payload, dict):
        return None
    if not payload.get("url") and payload.get("doc") is None:
        return None
    return payload


#: F39: re-resolve a mark's real target. The probe searches the top document
#: and every SAME-ORIGIN frame (a cross-origin frame is unreadable by design)
#: and reports which frame owns the element, so a mark can be refused when it
#: would act inside a different frame than the one it was observed in.
_VALIDATE_TARGET_JS = (
    "(() => {"
    "const css = %s;"
    "const doc = (performance && performance.timeOrigin) ? Math.round(performance.timeOrigin) : 0;"
    "const mut = (window.__jarvisEpoch && typeof window.__jarvisEpoch.mut === 'number')"
    " ? window.__jarvisEpoch.mut : 0;"
    "const dpr = (typeof devicePixelRatio === 'number' && devicePixelRatio > 0)"
    " ? devicePixelRatio : null;"
    "const path = (el) => { try {"
    " let p = ''; let n = el;"
    " while (n && n.nodeType === 1 && p.length < 220) {"
    "  let seg = n.tagName.toLowerCase();"
    "  if (n.id) { p = '#' + n.id + (p ? '>' + p : ''); break; }"
    "  let i = 1; let sib = n;"
    "  while ((sib = sib.previousElementSibling)) { if (sib.tagName === n.tagName) i++; }"
    "  seg += ':nth-of-type(' + i + ')';"
    "  p = seg + (p ? '>' + p : '');"
    "  n = n.parentElement; }"
    " return p; } catch(e) { return ''; } };"
    "const out = {checked: true, epoch: {doc: doc, mut: mut}, dpr: dpr,"
    " url: location.href, exists: false, frame: ''};"
    "const docs = [{doc: document, frame: ''}];"
    "try { for (const f of Array.from(document.querySelectorAll('iframe'))) {"
    "  try { if (f.contentDocument) docs.push({doc: f.contentDocument, frame: path(f)}); } catch(e) {}"
    " } } catch(e) {}"
    "for (const candidate of docs) {"
    "  let el = null;"
    "  try { el = css ? candidate.doc.querySelector(css) : null; }"
    "  catch (e) { out.bad = true; return JSON.stringify(out); }"
    "  if (!el) continue;"
    "  const rect = el.getBoundingClientRect();"
    "  const style = candidate.doc.defaultView.getComputedStyle(el);"
    "  out.exists = true;"
    "  out.frame = candidate.frame;"
    "  out.rect = {x: Math.round(rect.left), y: Math.round(rect.top),"
    "   w: Math.round(rect.width), h: Math.round(rect.height)};"
    "  out.visible = rect.width > 0 && rect.height > 0"
    "   && style.display !== 'none' && style.visibility !== 'hidden';"
    "  break;"
    "}"
    "return JSON.stringify(out);"
    "})()"
)


def _parse_mark_idx(idx_raw):
    try:
        return int(idx_raw)
    except Exception:
        return None


def _missing_mark_error(verb, idx, marks):
    avail = sorted(marks.keys()) if marks else []
    if avail:
        return ("%s error: mark %d not found. Available marks: %s. Call look again to refresh marks."
                % (verb, idx, avail))
    return "%s error: mark %d not found. No marks available - call look first." % (verb, idx)


def _active_tab_id(client):
    """The daemon's ACTIVE tab id, or "" when it cannot be established (F39)."""
    try:
        raw = client.call_tool("list_tabs", {})
    except Exception:
        return ""
    for entry in _daemon_tab_entries(raw):
        if not isinstance(entry, dict):
            continue
        if not entry.get("active"):
            continue
        tab_id = _tab_id_of(entry, trusted_only=True)
        if tab_id:
            return tab_id
    return ""


def _mark_target_state(client, stored, stats=None):
    """Re-resolve a stored mark against the live page (F39).

    Returns ``(state, error)``: live ``epoch``/``url``/``exists``/
    ``visible``/``rect``/``frame`` when the mark's cssPath could be
    re-checked, or a model-facing staleness sentence when it cannot. Marks
    with no cssPath return ``(None, None)`` — legacy coordinate marks keep
    their old path.

    F39 rules, all fail-closed on an identity-bearing mark:
      * a mark with no epoch stamp at all is REFUSED (it cannot be proven
        fresh) rather than compared against nothing;
      * zero epochs compare like any other value, so the FIRST mutation after
        a look, and a same-URL reload whose counters restart, are detected;
      * a document/url/DPR/frame/tab change is a refusal, so a mark cannot be
        redirected onto whatever now occupies the same coordinates.
      * BA-05: a MUTATION-counter drift is NOT a refusal (the counter fires
        on unrelated DOM noise) — it only widens the rect tolerance 3x,
        unless JARVIS_BROWSER_STALE_ON_MUTATION=1 restores the old refusal.
    """
    css = (stored or {}).get("cssPath") or ""
    if not css:
        return None, None
    want_doc, want_mut = _mark_epoch_of(stored)
    want_dpr = _epoch_text((stored or {}).get("dpr"))
    if not want_doc and not want_mut:
        return None, ("That mark carries no page-identity stamp, so it cannot "
                      "be proven fresh. Call look again.")
    try:
        js = _VALIDATE_TARGET_JS % json.dumps(css)
    except Exception:
        return None, None
    try:
        # BA-00: this probe is the "validation" half of every guarded
        # click/fill — timed separately so its cost is visible per task.
        _val = _Span("act.validate")
        raw = _probe(client, js)
        _val.done(stats)
    except Exception as exc:
        return None, "Target check failed (%s) - call look again." % exc
    live = _parse_json_result(raw)
    if not isinstance(live, dict):
        return None, "Target check returned no readable result - call look again."
    have_doc, have_mut = _mark_epoch_of(live.get("epoch") or {})
    if not have_doc:
        return None, ("The page did not report which document it is, so this "
                      "mark cannot be verified. Call look again.")
    if want_doc and have_doc != want_doc:
        return None, ("That mark is from an older page (the document changed since the look). "
                      "Call look again.")
    # BA-05: mut drift is NO LONGER a refusal — the global counter fires on
    # any unrelated DOM activity (class toggles, aria-live updates,
    # lazy images, spinners), so equality on it invalidated every mark at
    # once. A drifted counter only widens the rect tolerance: the precise
    # question ("is this still this element, in the same place?") is
    # answered by the exists/visible/live-rect checks below, which re-query
    # the cssPath against the live page.
    mut_drifted = bool(want_mut and have_mut and have_mut != want_mut)
    if mut_drifted and _stale_on_mutation():
        return None, ("That mark is stale (the page content changed since the look). "
                      "Call look again.")
    have_dpr = _epoch_text(live.get("dpr"))
    if want_dpr and have_dpr and want_dpr != have_dpr:
        return None, ("That mark was observed at a different display scale (device "
                      "pixel ratio %s, now %s), so its geometry no longer applies. "
                      "Call look again." % (want_dpr, have_dpr))
    live_url = live.get("url") or ""
    observed_url = (stored or {}).get("url") or ""
    if observed_url and live_url and observed_url != live_url:
        return None, ("That mark belongs to a different page (%s, now at %s). "
                      "Call look again." % (observed_url, live_url))
    want_frame = str((stored or {}).get("frame") or "")
    have_frame = str(live.get("frame") or "")
    if want_frame != have_frame:
        return None, ("That mark belongs to a different frame (%s, now %s). "
                      "Call look again." % (want_frame or "top document",
                                            have_frame or "top document"))
    if not live.get("exists"):
        return None, "That mark's element is gone from the page. Call look again."
    if not live.get("visible", True):
        return None, "That mark's element is no longer visible. Call look again to refresh."
    # F39: the tab this mark was observed in must still be the page's tab.
    # An unknown active tab proves nothing (so it is not a refusal), but a
    # KNOWN different one means a tab switch redirected the mark.
    stored_tab = str((stored or {}).get("tab_id") or "")
    if stored_tab:
        active_tab = _active_tab_id(client)
        if active_tab and active_tab != stored_tab:
            return None, ("That mark belongs to a different tab (%s; the active "
                          "tab is now %s). Call look again." % (stored_tab, active_tab))
    rect = live.get("rect") or {}
    observed = (stored or {}).get("rect") or {}
    # BA-05: a drifted mutation counter widens the tolerance 3x — the page
    # may have reflowed around an element that is still the same element.
    tolerance = _MARK_RECT_TOLERANCE * (3 if mut_drifted else 1)
    for key in ("x", "y", "w", "h"):
        try:
            moved = abs(int(rect.get(key, 0)) - int(observed.get(key, 0)))
        except Exception:
            moved = 0
        if moved > tolerance:
            return None, "That mark moved on the page since the look. Call look again."
    return live, None
# stringifies to literal '{}' and the resolve value is discarded forever.
# Every JS expression sent to evaluate MUST therefore be a synchronous
# IIFE / plain expression (like fill / batch_probe / look) - NEVER a
# Promise. Asynchronous outcomes (click navigation, wait_for) are handled
# from Python with sleeps between synchronous evaluates.
_CLICK_SETTLE_S = 0.5
_WAIT_POLL_S = 0.2

_AFTER_STATE_JS = (
    "(()=>{"
    "const s='a,button,input,select,textarea,[role=\"button\"],[onclick],[contenteditable],video,iframe,[tabindex]';"
    "const all=Array.from(document.querySelectorAll(s));"
    "const f=[];const m=[];const g=[];"
    "for(const e of all){"
    "const r=e.getBoundingClientRect();"
    "if(r.width<12||r.height<12)continue;"
    "const t=e.tagName.toLowerCase();"
    "if(t==='input'||t==='textarea'||t==='select'||e.hasAttribute('contenteditable'))f.push(e);"
    "else if(t==='video'||t==='iframe')m.push(e);"
    "else g.push(e);"
    "}"
    "const items=[];"
    "for(const e of f.concat(m,g).slice(0,12)){"
    "let l=(e.innerText||'').trim();"
    "if(!l)l=(e.getAttribute('aria-label')||'').trim();"
    "if(!l)l=(e.getAttribute('placeholder')||'').trim();"
    "l=l.slice(0,25).replace(/\\|/g,'/');"
    "items.push({tag:e.tagName.toLowerCase(),label:l||'(no label)'});"
    "}"
    "return JSON.stringify({title:document.title,url:location.href,items:items});"
    "})()"
)


def _parse_json_result(raw):
    """Best-effort parse of one evaluate result into a dict.

    Tolerates the daemon's double-stringify (a JS string comes back as a
    JSON-quoted string) and JSON embedded in surrounding text.
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    try:
        value = json.loads(text)
    except ValueError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            value = json.loads(text[start:end + 1])
        except ValueError:
            return None
    if isinstance(value, str):
        return _parse_json_result(value)
    return value if isinstance(value, dict) else None


# ── F47: browser-session identity ─────────────────────────────────────────
# The daemon owns its browser, but its tab tools tell us the identities as
# they change. Publishing them to the broker means executor / CDP / research
# / agent all share ONE picture of "which browser, which tab" instead of each
# subsystem guessing independently (the exact complaint in F47).

_AGENT_BROKER_SESSION_ID = "agent-daemon"
#: daemon tools whose results speak about tabs — publish what they said.
_TAB_PUBLISH_TOOLS = frozenset((
    "list_tabs", "switch_tab", "new_tab", "navigate", "open_brave",
))


def _parse_json_value(raw):
    """Like _parse_json_result, but keeps arrays too (tab lists)."""
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    try:
        value = json.loads(text)
    except ValueError:
        start = text.find("[")
        brace = text.find("{")
        if start == -1 or (brace != -1 and brace < start):
            start = brace
        end = text.rfind("]")
        if end == -1 or start == -1 or end <= start:
            return None
        try:
            value = json.loads(text[start:end + 1])
        except ValueError:
            return None
    if isinstance(value, str):
        return _parse_json_value(value)
    return value if isinstance(value, (dict, list)) else None


def _daemon_tab_entries(raw):
    """Tab dicts out of a daemon result.

    Accepts the shapes a tab-speaking tool may return: a list, a key-nested
    dict, a single dict — AND the daemon's actual ``list_tabs`` text format
    (``lib/tab_tools.mjs formatTabs``): ``0: "Title" - https://url [active]``.
    The text form is what the real daemon prints, so without it every
    identity lookup silently returned nothing.
    """
    if isinstance(raw, str):
        payload = _parse_json_value(raw)
        if payload is None:
            payload = raw
    else:
        payload = raw
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("tabs", "pages", "targets", "result", "data"):
            if isinstance(payload.get(key), list):
                return payload[key]
        if any(k in payload for k in ("id", "tabId", "tab_id", "targetId")):
            return [payload]
    if isinstance(payload, str):
        entries = []
        for line in payload.splitlines():
            entry = _daemon_tab_line(line)
            if entry:
                entries.append(entry)
        if entries:
            return entries
    return []


_DAEMON_TAB_LINE_RE = re.compile(
    r"^\s*(\d+):\s*\"(.*)\"\s+-\s+(\S+?)(?:\s+\[active\])?\s*$"
)


def _daemon_tab_line(line):
    """One formatTabs line -> {id, title, url, active} (None otherwise).

    F47: the daemon's TEXT listing exposes no CDP id, and the old parser
    minted ``daemon-tab-<positional index>`` — an UNSTABLE id: any switch,
    new tab or close renumbered every later tab, so a previously published
    identity pointed at a different tab. The derived id is now a content hash
    of (url, title), which survives reordering; the positional index is kept
    only as the ``index`` display hint. It is still NOT a trusted identity
    (F17): ``identity_source: "text"`` makes every identity lookup refuse it,
    so page-visible text can never mint tab identity, while URL/origin
    resolution keeps the tab addressable.
    """
    match = _DAEMON_TAB_LINE_RE.match(line or "")
    if not match:
        return None
    url = match.group(3)
    if not url.startswith(("http://", "https://", "file://", "about:")):
        return None
    title = match.group(2).strip()
    digest = hashlib.sha1(
        ("%s|%s" % (url, title)).encode("utf-8", "replace")).hexdigest()[:12]
    return {
        "id": "text-%s" % digest,
        "index": match.group(1),
        "title": title,
        "url": url,
        "active": "[active]" in (line or ""),
        "identity_source": "text",
    }


#: F17: fields a daemon must set to assert a real tab identity. A generic
#: ``id`` and a text-parsed positional index are display hints, not identity.
_STRUCTURED_TAB_ID_FIELDS = ("tabId", "tab_id", "targetId")


def _tab_id_of(entry, trusted_only=False):
    """The tab identity in *entry*, or "".

    F17: with *trusted_only* an identity is accepted only from a structured
    daemon response — page-visible text (a tab title, a look result) must
    never mint a tab identity.
    """
    if not isinstance(entry, dict):
        return ""
    source = str(entry.get("identity_source") or "structured")
    if trusted_only and source != "structured":
        return ""
    if trusted_only:
        for field_name in _STRUCTURED_TAB_ID_FIELDS:
            value = str(entry.get(field_name) or "").strip()
            if value:
                return value
        return ""
    return str(
        entry.get("tabId") or entry.get("tab_id") or entry.get("id")
        or entry.get("targetId") or ""
    ).strip()


#: F17: only a TAB-SPEAKING tool's result may be read as tab metadata. A page
#: observation (look / batch_probe / wait_for) can contain text that looks
#: exactly like the daemon's tab listing, and page-authored text must never
#: become trusted tab identity.
_TAB_SPEAKING_TOOLS = frozenset((
    "list_tabs", "switch_tab", "new_tab", "open_brave", "navigate",
))


def _may_publish_tabs(tool_name):
    return str(tool_name or "").strip().lower() in _TAB_SPEAKING_TOOLS


def _publish_daemon_tabs(result_text, tool_name=""):
    """F47: keep the broker's picture of the daemon browser current.

    Best-effort by design: an unparseable result just leaves the previous
    picture in place — identity tracking must never break a tool call.

    F17: only a tab-speaking tool's result is read, and only a structured
    daemon identity is published. Observation text is not evidence of a tab.
    """
    if tool_name and not _may_publish_tabs(tool_name):
        return
    try:
        from backend.services import browser_session_broker

        browser_session_broker.register_session(
            owner="agent", kind="daemon", session_id=_AGENT_BROKER_SESSION_ID,
            label="MCP browser-agent daemon browser")
        tabs = []
        for entry in _daemon_tab_entries(result_text):
            tab_id = _tab_id_of(entry, trusted_only=True)
            if not tab_id:
                continue
            tabs.append({
                "tab_id": tab_id,
                "url": str(entry.get("url") or ""),
                "title": str(entry.get("title") or ""),
                "ws_url": str(entry.get("webSocketDebuggerUrl")
                              or entry.get("wsUrl") or ""),
            })
        if tabs:
            browser_session_broker.publish_tabs(_AGENT_BROKER_SESSION_ID, tabs)
    except Exception:
        pass


def _publish_look_tab(url, title, tab_id):
    """F47: one look anchors the observed page to its daemon tab identity."""
    if not tab_id:
        return
    try:
        from backend.services import browser_session_broker

        browser_session_broker.register_session(
            owner="agent", kind="daemon", session_id=_AGENT_BROKER_SESSION_ID,
            label="MCP browser-agent daemon browser")
        browser_session_broker.note_tab(
            _AGENT_BROKER_SESSION_ID, tab_id, url=url, title=title)
    except Exception:
        pass


def _current_tab_id(client, url, stats=None):
    """F39/F47: which daemon tab is the page at *url*?

    Correlates the page the agent is looking at with a real tab id by asking
    the daemon. Returns "" when the daemon does not report a match — the
    caller treats tab identity as unknown, never guessed.

    F47: a URL that matches SEVERAL tabs is ambiguous, and guessing the first
    match is exactly how a request lands in the wrong tab. Ambiguity returns
    "" (unknown) — resolved by the caller's refusal paths — instead of
    silently picking one.
    """
    if not url:
        return ""
    try:
        # BA-00: this is the round trip BA-01 proposes deleting; measuring it
        # per look is what makes that deletion provable rather than asserted.
        _sp = _Span("look.list_tabs")
        raw = client.call_tool("list_tabs", {})
        _sp.done(stats)
    except Exception:
        return ""
    matches = []
    for entry in _daemon_tab_entries(raw):
        tab_url = str(entry.get("url") or "") if isinstance(entry, dict) else ""
        if tab_url and tab_url.rstrip("/") == str(url).rstrip("/"):
            # F17: identity comes from a structured daemon response only.
            tab_id = _tab_id_of(entry, trusted_only=True)
            if tab_id:
                matches.append(tab_id)
    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0]
    return ""


def _after_state(client):
    """Best-effort evaluate returning compact after-state field string.
    
    Returns a string like 'Page Title | 1 a Watch now | 2 input Search'
    or empty string '' on any failure. Never raises.
    """
    try:
        raw = _probe(client, _AFTER_STATE_JS)
        data = _parse_json_result(raw)
        if not isinstance(data, dict):
            return ""
        title = data.get("title", "")
        items = data.get("items", [])
        parts = [title] if title else []
        for i, item in enumerate(items, 1):
            tag = item.get("tag", "el")
            label = item.get("label", "")
            parts.append("%d %s %s" % (i, tag, label))
        return " | ".join(parts)[:250]
    except Exception:
        return ""


def _click_and_confirm(client, name, click_js):
    """Run ONE synchronous click evaluate, settle, then confirm navigation.

    The first evaluate performs the click and returns
    {clicked, tag, label, href} synchronously (href captured before the
    click's navigation can start - the stringify is synchronous). Python
    then sleeps the settle gap, runs a SECOND synchronous evaluate reading
    location.href, and assembles url / href_before / navigated itself.
    Misses (clicked false) skip the settle and the second evaluate.
    """
    try:
        raw = client.call_tool("evaluate", {"expression": click_js})
    except Exception as exc:
        return _clip_result("%s failed: evaluate error: %s" % (name, exc))
    payload = _parse_json_result(raw)
    if payload is None:
        return _clip_result(
            "%s error: could not parse click result: %s" % (name, (raw or "")[:500])
        )
    if not payload.get("clicked"):
        return _clip_result(json.dumps(payload))
    href_before = payload.get("href")
    time.sleep(_CLICK_SETTLE_S)
    url = href_before
    url_known = True
    try:
        raw_url = _probe(client, "location.href")
        if isinstance(raw_url, str):
            text_url = raw_url.strip()
            try:
                parsed_url = json.loads(text_url)
            except ValueError:
                parsed_url = None
            if isinstance(parsed_url, str):
                url = parsed_url
            elif text_url:
                url = text_url
            else:
                url_known = False
        elif raw_url is not None:
            url = str(raw_url)
        else:
            url_known = False
    except Exception:
        url = "unknown (post-click evaluate failed)"
        url_known = False
    payload["url"] = url
    payload["href_before"] = href_before
    payload["navigated"] = bool(url_known and url != href_before)
    if payload.get("nonInteractive"):
        tag = payload.get("tag", "element")
        payload["feedback"] = "clicked a non-interactive <%s> - probably no effect, look for a real control" % tag
    after = _after_state(client)
    if after:
        payload["after"] = after
    return _clip_result(json.dumps(payload))


def _click_confirm_locator_result(client, name, raw, scrolled=False):
    """Fold a daemon *_locator result into the agent's click-result shape.

    The daemon already settles and reports after-state; Python keeps the
    model's contract (url / href_before / navigated / after) by parsing the
    daemon's navigated/url lines rather than trusting a bare clicked flag.
    No further daemon round-trip happens here — the daemon is the source of
    truth for the after-state, and callers rely on the locator call being
    the last tool invocation on this path.     ``client`` is kept in the
    signature for symmetry with the coordinate click path (and to allow a
    future enriched after-state without touching every call site).

    *scrolled* (BA-15) marks a click on a target the look found off-screen:
    the daemon scrolled it into view inside the same real-input call, so the
    viewport moved — the model must look again before trusting coordinates.
    Reported only on success; a failed click reports no scroll.
    """
    _ = client
    text = (raw or "") if isinstance(raw, str) else str(raw or "")
    parsed = _parse_locator_outcome(text)
    payload = {"clicked": parsed["ok"], "via": "real-input"}
    if parsed["ok"] and scrolled:
        payload["scrolled"] = True
    if parsed.get("reason"):
        payload["reason"] = parsed["reason"]
    payload["url"] = parsed.get("url") or ""
    payload["navigated"] = bool(parsed.get("navigated"))
    payload["href_before"] = ""
    return _clip_result(json.dumps(payload))


def _parse_locator_outcome(text):
    """Outcome lines the daemon's real-input tools print (F40).

    Two defects are fixed here:

      * the whole result was lowercased before ``url=`` was extracted, so a
        literal URL came back folded to lower case — destroying
        case-sensitive paths, query values and fragments. The URL is now read
        from the ORIGINAL text;
      * success was inferred from incidental words ("real", any url= field,
        absence of the word "error"), so an error sentence that happened to
        quote a URL and a ``navigated=`` field counted as a successful click.
        Success now requires an EXPLICIT positive marker and no failure marker,
        and failure markers win over everything else.
    """
    line = (text or "").strip()
    lowered = line.lower()

    url = ""
    for part in line.split():
        if part.lower().startswith("url="):
            # Literal-preserving: the value keeps its case; only the KEY was
            # matched case-insensitively.
            url = part[len("url="):].strip().rstrip(",;")
    if not url:
        match = re.search(r"\burl=(\S+)", line)
        if match:
            url = match.group(1).strip().rstrip(",;")

    navigated = bool(re.search(r"\bnavigated=(?:yes|true|1)\b", lowered))

    failed = bool(_LOCATOR_FAILURE_RE.search(lowered))
    succeeded = bool(_LOCATOR_SUCCESS_RE.search(lowered))

    if failed:
        ok = False
    elif succeeded:
        ok = True
    else:
        ok = False
    # BA-05 fallout: more clicks reach this parser now that mut drift no
    # longer refuses up front — an empty daemon reply must fail gracefully,
    # never IndexError on splitlines()[0].
    reason = line.splitlines()[0][:300] if (not ok and line) else ""
    return {"ok": ok, "url": url, "navigated": navigated, "reason": reason}


#: F40: wording that means the primitive did NOT do the thing. Failure always
#: wins, so an error that also prints url=/navigated= cannot read as success.
_LOCATOR_FAILURE_RE = re.compile(
    r"no element matches|no element|not found|was not found|does not exist"
    r"|failed|failure|could not|couldn't|cannot|can't|unable|timed out|timeout"
    r"|refused|denied|disabled|not clickable|not visible|not saved"
    r"|no download event|nothing was saved|missing|blocked|error",
    re.IGNORECASE)

#: F40: explicit positive markers. Producing no marker at all is NOT success.
_LOCATOR_SUCCESS_RE = re.compile(
    r"via real input|\bok\s*[:=]\s*(?:true|yes|1|ok|done)"
    r"|\b(?:clicked|filled|typed|selected|checked|unchecked|uploaded|downloaded"
    r"|dragged|dropped|scrolled|activated|saved|succeeded|success)\b",
    re.IGNORECASE)


def _page_url(client):
    """The live page's URL ("" when it cannot be read)."""
    try:
        raw = _probe(client, "location.href")
    except Exception:
        return ""
    if raw is None:
        return ""
    if isinstance(raw, str):
        text = raw.strip()
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        if isinstance(parsed, str):
            return parsed
        return text
    return str(raw)


# ── F40: real, frame-aware primitives ─────────────────────────────────────
# Every model-facing interaction resolves a REAL target identity first and
# then dispatches the daemon's real-input primitive (click_locator /
# fill_locator) with it. A synthetic element.click()/KeyboardEvent can be
# ignored by a framework, cannot move the mouse and cannot reach into a
# same-origin iframe; the locator calls can. The resolve step also reports
# the frame that owns the element, so the primitive acts in the same frame
# the element was found in.

_CSS_PATH_JS = (
    "const jarvisPath = (el) => { try {"
    " let p = ''; let n = el;"
    " while (n && n.nodeType === 1 && p.length < 220) {"
    "  let seg = n.tagName.toLowerCase();"
    "  if (n.id) { p = '#' + n.id + (p ? '>' + p : ''); break; }"
    "  let i = 1; let sib = n;"
    "  while ((sib = sib.previousElementSibling)) { if (sib.tagName === n.tagName) i++; }"
    "  seg += ':nth-of-type(' + i + ')';"
    "  p = seg + (p ? '>' + p : '');"
    "  n = n.parentElement; }"
    " return p; } catch(e) { return ''; } };"
    # The owning frame of an element: '' for the top document, else the
    # cssPath of the same-origin iframe that contains it.
    "const jarvisFrame = (el) => { try {"
    " const w = el.ownerDocument && el.ownerDocument.defaultView;"
    " if (w && w !== window && w.frameElement) return jarvisPath(w.frameElement);"
    " } catch(e) {}"
    " return ''; };"
    # (cssPath, frame) for an element — the identity the locator takes.
    "const jarvisIdent = (el) => ({cssPath: jarvisPath(el), frame: jarvisFrame(el)});"
    "const jarvisDisabled = (el) => { try {"
    " return !!(el.disabled || el.getAttribute('aria-disabled') === 'true');"
    " } catch(e) { return false; } };"
)


def _locator_target(arguments, css, frame):
    """Daemon arguments for a locator call, frame included when known."""
    payload = dict(arguments or {})
    payload["css"] = css
    if frame:
        payload["frame"] = frame
    return payload


def _real_locate(client, name, resolve_js):
    """One synchronous resolve probe -> (identity payload, error text)."""
    try:
        raw = _probe(client, resolve_js)
    except Exception as exc:
        return None, "%s failed: target resolution error: %s" % (name, exc)
    payload = _parse_json_result(raw)
    if payload is None:
        return None, ("%s error: could not parse the target resolution result: %s"
                      % (name, (raw or "")[:300] if isinstance(raw, str) else str(raw)))
    return payload, None


def _real_click_target(client, name, payload):
    """Click a resolved target with the daemon's REAL input primitive."""
    css = str(payload.get("cssPath") or "")
    if not css:
        return _clip_result(
            "%s failed: the matched element has no addressable identity, so it "
            "cannot be clicked with real input." % name)
    if payload.get("disabled"):
        return _clip_result(
            "%s failed: the matched element is disabled, so a click cannot do "
            "anything." % name)
    args = _locator_target({}, css, payload.get("frame") or "")
    try:
        raw = client.call_tool("click_locator", args)
    except Exception as exc:
        return _clip_result("%s failed: real click error: %s" % (name, exc))
    result = _click_confirm_locator_result(client, name, raw)
    parsed = _parse_locator_outcome(raw if isinstance(raw, str) else str(raw or ""))
    try:
        body = json.loads(result)
    except ValueError:
        return result
    # F40: the click happened through the element's REAL identity — say which
    # one, so a later stale-page click is obvious in the trace.
    body["css"] = css
    if parsed.get("reason") and not body.get("clicked"):
        body["reason"] = parsed["reason"]
    return _clip_result(json.dumps(body))


#: F13: read a control's ACTUAL state back after a state change. Searches the
#: top document and same-origin frames, so a permitted iframe control is
#: verified too; anything else reports "could not verify" instead of guessing.
_READ_CONTROL_JS = (
    "(() => {"
    "const css = %s;"
    "const out = {found: false, url: location.href, frame: ''};"
    "const docs = [{doc: document, frame: ''}];"
    "try { for (const f of Array.from(document.querySelectorAll('iframe'))) {"
    "  try { if (f.contentDocument) docs.push({doc: f.contentDocument, frame: ''}); } catch(e) {}"
    " } } catch(e) {}"
    "for (const candidate of docs) {"
    "  let el = null;"
    "  try { el = css ? candidate.doc.querySelector(css) : null; } catch(e) { el = null; }"
    "  if (!el) continue;"
    "  out.found = true;"
    "  out.frame = candidate.frame;"
    "  out.tag = el.tagName.toLowerCase();"
    "  if ('checked' in el) out.checked = !!el.checked;"
    "  if ('value' in el) out.value = String(el.value);"
    "  try { if (el.selectedOptions && el.selectedOptions[0]) out.label = (el.selectedOptions[0].text || '').trim(); } catch(e) {}"
    "  break;"
    "}"
    "return JSON.stringify(out);"
    "})()"
)


def _read_control_state(client, css):
    """(state dict | None) — the control's real state, None when unreadable."""
    if not css:
        return None
    try:
        js = _READ_CONTROL_JS % json.dumps(css)
    except Exception:
        return None
    try:
        raw = _probe(client, js)
    except Exception:
        return None
    state = _parse_json_result(raw)
    return state if isinstance(state, dict) else None


#: F13: destination approval for a file disclosure. An upload moves a LOCAL
#: FILE into a REMOTE page, so being allowed to READ the file is not the same
#: authority as being allowed to SEND it somewhere: the (file, origin) pair
#: needs its own grant.
_UPLOAD_ORIGINS_ENV = "JARVIS_BROWSER_UPLOAD_ORIGINS"


def _origin_of(url):
    try:
        parts = urlsplit(str(url or "").strip())
    except Exception:
        return ""
    if not parts.netloc:
        return ""
    return "%s://%s" % ((parts.scheme or "http").lower(), parts.netloc.lower())


def _approved_upload_origins(session):
    """Origins this run may disclose a file to (F13)."""
    origins = set()
    recorded = None
    try:
        recorded = (session or {}).get("upload_origins")
    except Exception:
        recorded = None
    if recorded is None:
        recorded = [part.strip()
                    for part in os.getenv(_UPLOAD_ORIGINS_ENV, "").split(",")]
    if isinstance(recorded, str):
        recorded = [recorded]
    for item in recorded or []:
        origin = _origin_of(item)
        if origin:
            origins.add(origin)
    for grant in _grants_for(session):
        if isinstance(grant, str) and grant.startswith("upload_origin:"):
            origin = _origin_of(grant.split(":", 1)[1])
            if origin:
                origins.add(origin)
    return origins


def _upload_origin_error(session, client):
    """Refusal text when the CURRENT page origin may not receive a file."""
    page_url = _page_url(client)
    origin = _origin_of(page_url)
    if not origin:
        return ("upload_file refused: the destination page's origin could not be "
                "established, so the file cannot be shown to be approved for it. "
                "Nothing was dispatched.")
    approved = _approved_upload_origins(session)
    if origin in approved:
        return ""
    return ("upload_file refused: %s is not approved to receive local files in "
            "this run (approved: %s). Nothing was dispatched - approve the origin "
            "with a 'upload_origin:%s' grant, session upload_origins, or "
            "%s." % (origin, ", ".join(sorted(approved)) or "none", origin,
                     _UPLOAD_ORIGINS_ENV))


def _handle_click_mark(client, session, arguments, stats=None):
    idx = _parse_mark_idx(arguments.get("index"))
    if idx is None:
        return _clip_result("click_mark error: index must be integer, got %r" % (arguments.get("index"),))
    marks = session.get("marks") or {}
    if idx not in marks:
        return _clip_result(_missing_mark_error("click_mark", idx, marks))
    info = marks[idx]
    cx = info["cx"]
    cy = info["cy"]
    # BA-15: an off-screen mark with an addressable identity is NO LONGER
    # refused — click_locator scrolls it into view inside the same real-input
    # call (daemon scrollIntoViewIfNeeded), so the four-turn
    # refuse -> scroll -> look -> click collapses to one. Only marks WITHOUT
    # a cssPath keep the refusal: coordinate clicks cannot scroll to a
    # target the daemon cannot address.
    was_offscreen = not info.get("inView", True)
    if was_offscreen and not info.get("cssPath"):
        return _clip_result(
            "click_mark error: mark %d is off-screen (outside the current viewport) "
            "- scroll it into view first" % idx
        )
    # G6 / F39: marks with a stored cssPath are re-resolved BEFORE any
    # click — navigation, same-URL reloads and SPA DOM updates invalidate
    # them, and the model is told to look again rather than clicking stale
    # coordinates. A validated mark is then clicked with REAL daemon input
    # (G6 / F40: click_locator) instead of coordinate JavaScript.
    if info.get("cssPath"):
        _state, stale = _mark_target_state(client, info, stats)
        if stale:
            return _clip_result("click_mark error: %s" % stale)
        try:
            frame = (info.get("frame") or "").strip() if isinstance(info, dict) else ""
            args = {"css": info["cssPath"]}
            if frame:
                args["frame"] = frame
            # The last daemon call on this path is the real input itself —
            # no post-click evaluate: the daemon already settled and reported
            # the after-state inside its result text (see
            # _click_confirm_locator_result).
            # BA-00: the "locator" half of the guarded click — the real
            # input round trip, timed apart from the validation probe above.
            _loc = _Span("act.locator")
            raw = client.call_tool("click_locator", args)
            _loc.done(stats)
        except Exception as exc:
            return _clip_result("click_mark failed: real click error: %s" % exc)
        return _click_confirm_locator_result(client, "click_mark", raw,
                                             scrolled=was_offscreen)
    # Synchronous IIFE: elementFromPoint with ancestor clickable check up
    # to 3 levels. href is captured at click time; Python confirms the
    # navigation after a settle gap - SPA navigations report
    # navigated: true, which kills duplicate-click loops (the model knows
    # to look again instead).
    js = (
        "(() => {"
        "const x = %d, y = %d;"
        "const el = document.elementFromPoint(x, y);"
        "if (!el) { return JSON.stringify({clicked: false, error: 'no element at point', x: x, y: y, href: location.href}); }"
        "let target = el;"
        "for (let i = 0; i < 3; i++) {"
        "  if (!target) break;"
        "  const tag = target.tagName ? target.tagName.toLowerCase() : '';"
        "  const isClickable = tag==='a' || tag==='button' || tag==='input' || tag==='select' || tag==='textarea' || target.hasAttribute('onclick') || target.getAttribute('role')==='button' || target.hasAttribute('contenteditable') || tag==='video' || target.hasAttribute('tabindex');"
        "  if (isClickable) {"
        # F40: a click that throws or hits a disabled control is a FAILURE,
        # not a silent no-op — swallowing the exception made both look
        # identical to a successful click.
        "   if (target.disabled || target.getAttribute('aria-disabled') === 'true') { return JSON.stringify({clicked: false, error: 'element is disabled', tag: tag, href: location.href}); }"
        "   try { target.click(); } catch(e) { return JSON.stringify({clicked: false, error: 'click threw: ' + String(e), tag: tag, href: location.href}); }"
        "   return JSON.stringify({clicked: true, tag: tag, label: (target.innerText||target.getAttribute('aria-label')||'').trim().slice(0,50), href: location.href}); }"
        "  target = target.parentElement;"
        "}"
        "if (el.disabled || el.getAttribute('aria-disabled') === 'true') { return JSON.stringify({clicked: false, error: 'element is disabled', tag: el.tagName.toLowerCase(), href: location.href}); }"
        "try { el.click(); } catch(e) { return JSON.stringify({clicked: false, error: 'click threw: ' + String(e), tag: el.tagName.toLowerCase(), href: location.href}); }"
        "return JSON.stringify({clicked: true, tag: el.tagName.toLowerCase(), label: (el.innerText||'').trim().slice(0,50), href: location.href, nonInteractive: !['a','button','input','select','textarea','video'].includes(el.tagName.toLowerCase()) && !el.hasAttribute('onclick') && el.getAttribute('role')!=='button' && !el.hasAttribute('contenteditable') && !el.hasAttribute('tabindex')});"
        "})()" % (int(cx), int(cy))
    )
    return _click_and_confirm(client, "click_mark", js)


def _handle_batch_probe(client, arguments):
    exprs = arguments.get("expressions")
    if not isinstance(exprs, list):
        return _clip_result("batch_probe error: expressions must be an array")
    if len(exprs) > 10:
        return _clip_result("batch_probe error: too many expressions (max 10, got %d)" % len(exprs))
    for e in exprs:
        if not isinstance(e, str):
            return _clip_result("batch_probe error: each expression must be a string")
        # F17: this tool advertises itself as read-only but used to eval
        # whatever the model wrote. Every expression must now be a typed DOM
        # read expression — no assignment, no network, no cookies, no calls
        # outside the getter allowlist.
        reason = tool_policy.probe_expression_error(e)
        if reason:
            return _clip_result(
                "batch_probe rejected (read-only probe policy): %s — use a typed "
                "DOM read such as document.querySelector('#price').innerText" % reason)
    # Build JS that evaluates each via try/catch
    exprs_json = json.dumps(exprs)
    js = (
        "(() => {"
        "const exprs = %s;"
        "const results = [];"
        "for (const e of exprs) {"
        "  try { results.push({expr: e, ok: eval(e)}); } catch(err) { results.push({expr: e, error: String(err)}); }"
        "}"
        "return JSON.stringify(results);"
        "})()" % exprs_json
    )
    try:
        raw = _probe(client, js)
    except Exception as exc:
        return _clip_result("batch_probe failed: evaluate error: %s" % exc)
    return _clip_result(raw)


def _handle_wait_for(client, arguments, stats=None, session=None):
    selector = arguments.get("selector")
    text = arguments.get("text")
    # F41: a wait with neither a selector nor text used to return found=true
    # immediately ("!selector && !text -> found"), so an empty wait "succeeded"
    # without observing anything. There is no goal to verify, so refuse it.
    has_selector = isinstance(selector, str) and selector.strip() != ""
    has_text = isinstance(text, str) and text.strip() != ""
    if not has_selector and not has_text:
        return _clip_result(
            "wait_for failed: provide a selector, text, or both - an empty wait "
            "verifies nothing."
        )
    if not has_selector:
        selector = None
    if not has_text:
        text = None
    timeout_ms = arguments.get("timeout_ms", 5000)
    try:
        timeout_ms = int(timeout_ms)
    except Exception:
        timeout_ms = 5000
    if timeout_ms > 10000:
        timeout_ms = 10000
    if timeout_ms < 0:
        timeout_ms = 0
    # BA-14: a daemon that advertises its own event-driven `wait_for` gets
    # ONE round trip instead of N polls. The capability comes from the task's
    # daemon tool list (see _agent_loop_inner) — never error-sniffed — so an
    # old daemon transparently keeps the polling path below.
    daemon_names = session.get("daemon_tools") if isinstance(session, dict) else None
    if daemon_names and "wait_for" in daemon_names:
        return _handle_wait_for_event(client, selector, text, timeout_ms,
                                       stats=stats, session=session)
    return _handle_wait_for_poll(client, selector, text, timeout_ms,
                                 stats=stats)


#: Markers proving the daemon predates the event-driven `wait_for` tool
#: (BA-14): only THESE fall back to polling. Any other daemon error is a
#: real failure — silently converting it to N polls would burn round trips
#: hiding a breakage.
_UNKNOWN_TOOL_MARKERS = (
    "unknown tool",
    "tool not found",
    "method not found",
    "-32601",
)


def _is_unknown_tool_error(exc):
    try:
        text = str(exc or "").lower()
    except Exception:
        return False
    return any(marker in text for marker in _UNKNOWN_TOOL_MARKERS)


def _handle_wait_for_event(client, selector, text, timeout_ms, stats=None,
                           session=None):
    """One daemon round trip for the whole wait (BA-14).

    The daemon waits inside its own process and returns the instant the
    condition holds; the BA-00 span still closes with polls=1 so a slow wait
    attributes to one daemon round trip, not many cheap polls. An old daemon
    that somehow lacks the tool (capability raced a downgrade) falls back to
    polling; a daemon that HAS it but answers garbage fails loudly — that is
    a contract violation, not a missing capability.
    """
    payload = {"timeout_ms": timeout_ms}
    if selector:
        payload["selector"] = selector
    if text:
        payload["text"] = text
    _wait = _Span("wait.polls")
    try:
        raw = client.call_tool("wait_for", payload)
    except Exception as exc:
        if _is_unknown_tool_error(exc):
            return _handle_wait_for_poll(client, selector, text, timeout_ms,
                                         stats=stats)
        _wait.done(stats, polls=1, found=False)
        return _clip_result("wait_for failed: daemon waiter error: %s" % exc)
    data = _parse_json_result(raw)
    if not isinstance(data, dict) or "found" not in data:
        _wait.done(stats, polls=1, found=False)
        return _clip_result(
            "wait_for error: daemon waiter returned an unparsable result: %s"
            % ((raw or "")[:500] if isinstance(raw, str) else str(raw or "")))
    try:
        found = bool(data.get("found"))
    except Exception:
        found = False
    try:
        elapsed_ms = int(data.get("elapsed_ms", 0))
    except Exception:
        elapsed_ms = 0
    out = {"found": found, "elapsed_ms": elapsed_ms,
           "url": data.get("url") or "", "title": data.get("title") or ""}
    if data.get("error"):
        out["error"] = str(data.get("error"))[:300]
    _wait.done(stats, polls=1, found=found)
    return _clip_result(json.dumps(out))


def _handle_wait_for_poll(client, selector, text, timeout_ms, stats=None):
    """The pre-BA-14 polling path, kept as the capability-gated fallback."""
    sel_json = json.dumps(selector) if isinstance(selector, str) else "null"
    text_json = json.dumps(text) if isinstance(text, str) else "null"
    # Polling fallback for daemons predating the event-driven `wait_for`
    # tool: the old daemon stringify is synchronous, so a Promise-based
    # waiter would stringify to '{}' and hang. Each poll is ONE synchronous
    # IIFE checking the page; Python sleeps between polls and stops at the
    # timeout, reporting found + elapsed_ms.
    js = (
        "(() => {"
        "const selector = %s;"
        "const text = %s;"
        "let found = false;"
        "if (selector) { try { if (document.querySelector(selector)) found = true; } catch(e){} }"
        "if (text) { if (document.body && document.body.innerText && document.body.innerText.includes(text)) found = true; }"
        "return JSON.stringify({found: found, url: location.href, title: document.title});"
        "})()" % (sel_json, text_json)
    )
    started = time.monotonic()
    deadline = started + timeout_ms / 1000.0
    # BA-00: count the polls so a slow wait_for can be attributed (many cheap
    # polls vs one slow daemon round trip) instead of guessed.
    _wait = _Span("wait.polls")
    polls = 0
    while True:
        try:
            raw = _probe(client, js)
        except Exception as exc:
            _wait.done(stats, polls=polls, found=False)
            return _clip_result("wait_for failed: evaluate error: %s" % exc)
        payload = _parse_json_result(raw)
        if payload is None:
            _wait.done(stats, polls=polls, found=False)
            return _clip_result(
                "wait_for error: could not parse page check: %s" % (raw or "")[:500]
            )
        polls += 1
        found = bool(payload.get("found"))
        elapsed_ms = int(round((time.monotonic() - started) * 1000))
        if found or time.monotonic() >= deadline:
            payload["found"] = found
            payload["elapsed_ms"] = elapsed_ms
            _wait.done(stats, polls=polls, found=found)
            return _clip_result(json.dumps(payload))
        time.sleep(_WAIT_POLL_S)


# Shared JS tail for fill / fill_mark: focus -> native setter -> input
# event -> optional Enter keydown+keyup + form.requestSubmit. The synchronous
# change event is deliberately dropped (it broke search arming on some
# sites); frameworks react to the input event. Returns enough context
# (tag/placeholder/visible/rect/value) for the model to self-correct.
# F21: the fill result used to return the field's raw `value`, so a password
# or token typed by the model came straight back into the transcript (and was
# logged in full — _log_tool_result allowed 5,000 chars for fill). The JS now
# classifies the target and returns a masked acknowledgment instead.
_FILL_JS_SENSITIVE = (
    "const isSensitive = (el) => {"
    "  const t = (el.getAttribute('type') || '').toLowerCase();"
    "  if (t === 'password') return true;"
    "  const auto = (el.getAttribute('autocomplete') || '').toLowerCase();"
    "  if (auto.includes('password') || auto.includes('cc-') || auto.includes('otp')) return true;"
    "  const hay = [el.getAttribute('name'), el.id, el.getAttribute('placeholder'),"
    "               el.getAttribute('aria-label'), el.getAttribute('autocomplete')]"
    "               .filter(Boolean).join(' ').toLowerCase();"
    "  return /pass|pwd|secret|token|otp|cvv|cvc|pin|card|credit|ssn/.test(hay);"
    "};"
    "const echoValue = (el, v) => isSensitive(el)"
    "  ? ('<masked:' + (v ? String(v.length) : 0) + ' chars>')"
    "  : v;"
    ";"
)

_FILL_JS_CORE = (
    "const fillTarget = (el) => {"
    "el.focus();"
    "const isTextarea = el.tagName.toLowerCase() === 'textarea';"
    "const desc = isTextarea ? Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value') : Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');"
    "if (desc && desc.set) desc.set.call(el, val); else el.value = val;"
    "el.dispatchEvent(new Event('input', {bubbles: true}));"
    "const rect = el.getBoundingClientRect();"
    "const visible = rect.width > 2 && rect.height > 2 && el.getClientRects().length > 0 && !(rect.bottom < 0 || rect.top > window.innerHeight || rect.right < 0 || rect.left > window.innerWidth);"
    # F40: ONE submission mechanism. Dispatching Enter AND calling
    # form.requestSubmit() submitted Enter-handled forms twice.
    "if (pressEnter) {"
    "  el.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true}));"
    "  el.dispatchEvent(new KeyboardEvent('keyup', {key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true}));"
    "}"
    "return JSON.stringify({ok: true, tag: el.tagName.toLowerCase(), placeholder: el.getAttribute('placeholder') || '', name: el.getAttribute('name') || '', visible: visible, rect: {x: Math.round(rect.left), y: Math.round(rect.top), w: Math.round(rect.width), h: Math.round(rect.height)}, value: echoValue(el, el.value)});"
    "};"
)

_FILL_JS_VISIBLE = (
    "const isVisible = (el) => {"
    "  const rect = el.getBoundingClientRect();"
    "  if (rect.width < 2 || rect.height < 2) return false;"
    "  if (el.getClientRects().length === 0) return false;"
    "  const st = window.getComputedStyle(el);"
    "  if (st.visibility === 'hidden' || st.display === 'none' || st.opacity === '0') return false;"
    "  if (rect.bottom < 0 || rect.top > window.innerHeight || rect.right < 0 || rect.left > window.innerWidth) return false;"
    "  return true;"
    "};"
)


def _handle_fill(client, arguments):
    selector = arguments.get("selector")
    value = arguments.get("value")
    press_enter = arguments.get("press_enter", False)
    if not isinstance(selector, str) or not isinstance(value, str):
        return _clip_result("fill error: selector and value must be strings")
    # Normalize press_enter to bool
    press_enter = bool(press_enter)
    sel_json = json.dumps(selector)
    # F40: the match step only RESOLVES the real target (identity + frame);
    # the value is typed by the daemon's real keyboard primitive, never by
    # page JavaScript, and Enter is the single submission channel.
    js_template = (
        "(() => {"
        "const sel = %s;"
        + _CSS_PATH_JS +
        _FILL_JS_VISIBLE +
        "let matches = [];"
        "try { matches = Array.from(document.querySelectorAll(sel)); } catch (e) { matches = []; }"
        "let el = matches.find(isVisible);"
        "if (!el) {"
        "  const inputs = Array.from(document.querySelectorAll('input, textarea')).filter(isVisible).filter(i => {"
        "    if (i.tagName.toLowerCase() === 'textarea') return true;"
        "    const t = (i.getAttribute('type') || 'text').toLowerCase();"
        "    return t === 'text' || t === 'search' || t === 'email' || t === 'url' || t === 'password' || t === 'tel' || t === 'number' || t === '';"
        "  });"
        "  const low = sel.toLowerCase();"
        "  const score = (i) => {"
        "    const ph = (i.getAttribute('placeholder') || '').toLowerCase();"
        "    const nm = (i.getAttribute('name') || '').toLowerCase();"
        "    const al = (i.getAttribute('aria-label') || '').toLowerCase();"
        "    const id = (i.id || '').toLowerCase();"
        "    if (ph === low || nm === low || al === low || id === low) return 3;"
        "    if (ph.includes(low) || nm.includes(low) || al.includes(low) || id.includes(low)) return 2;"
        "    return 0;"
        "  };"
        "  let best = null; let bestScore = 0;"
        "  for (const i of inputs) { const s = score(i); if (s > bestScore) { best = i; bestScore = s; } }"
        "  if (best) el = best;"
        "  else {"
        "    const list = inputs.map(i => '<' + i.tagName.toLowerCase() + (i.getAttribute('placeholder') ? \" placeholder='\" + i.getAttribute('placeholder') + \"'\" : '') + (i.id ? \" id='\" + i.id + \"'\" : '') + (i.getAttribute('name') ? \" name='\" + i.getAttribute('name') + \"'\" : '') + '>').join('; ');"
        "    return JSON.stringify({ok: false, error: 'no visible text input matches ' + sel, visible_inputs: list || 'none'});"
        "  }"
        "}"
        "const ident = jarvisIdent(el);"
        "return JSON.stringify({found: true, tag: el.tagName.toLowerCase(),"
        " placeholder: el.getAttribute('placeholder') || '',"
        " visible: isVisible(el), cssPath: ident.cssPath, frame: ident.frame,"
        " disabled: jarvisDisabled(el)});"
        "})()"
    )
    js = js_template % (sel_json,)
    payload, error = _real_locate(client, "fill", js)
    if error:
        return _clip_result(error)
    if not payload.get("found"):
        # The miss diagnostics (visible_inputs / error) are the model's
        # recovery path and are passed through unchanged.
        return _clip_result(json.dumps(payload))
    if payload.get("disabled"):
        return _clip_result("fill failed: the matched input is disabled, so it "
                            "cannot receive text.")
    css = str(payload.get("cssPath") or "")
    if not css:
        return _clip_result("fill failed: the matched input has no addressable "
                            "identity, so real keyboard input cannot target it.")
    args = _locator_target({"value": value,
                            "submit": "enter" if press_enter else "none"},
                           css, payload.get("frame") or "")
    try:
        raw = client.call_tool("fill_locator", args)
    except Exception as exc:
        return _clip_result("fill failed: real fill error: %s" % exc)
    parsed = _parse_locator_outcome(raw if isinstance(raw, str) else str(raw or ""))
    if not parsed["ok"]:
        return _clip_result("fill failed: %s" % (parsed.get("reason") or raw))
    return _clip_result(json.dumps({
        "ok": True, "via": "real-input", "css": css,
        "url": parsed.get("url") or "", "navigated": bool(parsed.get("navigated")),
        "tag": payload.get("tag") or ""}))


def _handle_fill_mark(client, session, arguments, stats=None):
    idx = _parse_mark_idx(arguments.get("index"))
    if idx is None:
        return _clip_result("fill_mark error: index must be integer, got %r" % (arguments.get("index"),))
    value = arguments.get("value")
    if not isinstance(value, str):
        return _clip_result("fill_mark error: value must be a string")
    press_enter = bool(arguments.get("press_enter", False))
    marks = session.get("marks") or {}
    if idx not in marks:
        return _clip_result(_missing_mark_error("fill_mark", idx, marks))
    info = marks[idx]
    cx = info["cx"]
    cy = info["cy"]
    # G6 / F39 + F40: re-resolve marks with identity before filling; a
    # validated mark is filled with REAL daemon input (fill_locator) with
    # ONE explicit submission channel instead of synthetic KeyboardEvents +
    # requestSubmit improvisation. Legacy coordinate marks keep the old JS.
    if info.get("cssPath"):
        _state, stale = _mark_target_state(client, info, stats)
        if stale:
            return _clip_result("fill_mark error: %s" % stale)
        submit = "enter" if press_enter else "none"
        frame = (info.get("frame") or "").strip() if isinstance(info, dict) else ""
        args = {"css": info["cssPath"], "value": value, "submit": submit}
        if frame:
            args["frame"] = frame
        try:
            # BA-00: the "locator" half of the guarded fill.
            _loc = _Span("act.locator")
            raw = client.call_tool("fill_locator", args)
            _loc.done(stats)
        except Exception as exc:
            return _clip_result("fill_mark failed: real fill error: %s" % exc)
        parsed = _parse_locator_outcome(raw if isinstance(raw, str) else str(raw or ""))
        if not parsed["ok"]:
            payload = {"ok": False, "via": "real-input",
                       "reason": parsed.get("reason") or raw,
                       "url": parsed.get("url") or ""}
            return _clip_result(json.dumps(payload))
        # The daemon is the source of truth for the after-state (already
        # folded into parsed) — the locator call stays the last daemon
        # round-trip, exactly like the click_mark path.
        result_obj = {"ok": True, "via": "real-input", "css": info["cssPath"],
                      "url": parsed.get("url") or "",
                      "navigated": bool(parsed.get("navigated"))}
        # BA-15: fill_locator scrolls an off-screen target into view inside
        # the same call (daemon parity with click_locator) — report it so
        # the model knows the viewport moved. Success-only, like clicks.
        if not info.get("inView", True):
            result_obj["scrolled"] = True
        return _clip_result(json.dumps(result_obj))
    val_json = json.dumps(value)
    press_json = "true" if press_enter else "false"
    js_template = (
        "(() => {"
        "const x = %d, y = %d;"
        "const val = %s;"
        "const pressEnter = %s;"
        + _FILL_JS_SENSITIVE +
        _FILL_JS_CORE +
        "const el = document.elementFromPoint(x, y);"
        "if (!el) return JSON.stringify({ok: false, error: 'no element at mark point', x, y});"
        "let target = null; let node = el;"
        "for (let i = 0; i < 4; i++) {"
        "  if (!node) break;"
        "  const tag = node.tagName ? node.tagName.toLowerCase() : '';"
        "  if (tag === 'input' || tag === 'textarea') { target = node; break; }"
        "  node = node.parentElement;"
        "}"
        "if (!target) return JSON.stringify({ok: false, error: 'mark %d is not a text input - the element there is a <' + el.tagName.toLowerCase() + '>', tag: el.tagName.toLowerCase()});"
        "return fillTarget(target);"
        "})()"
    )
    js = js_template % (int(cx), int(cy), val_json, press_json, idx)
    try:
        raw = client.call_tool("evaluate", {"expression": js})
    except Exception as exc:
        return _clip_result("fill_mark failed: evaluate error: %s" % exc)
    if press_enter:
        try:
            result_obj = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(result_obj, dict) and result_obj.get("ok"):
                after = _after_state(client)
                if after:
                    result_obj["after"] = after
                    raw = json.dumps(result_obj)
        except Exception:
            pass
    return _clip_result(raw)


def _handle_click_text(client, arguments):
    text = arguments.get("text")
    if not isinstance(text, str):
        return _clip_result("click_text error: text must be a string")
    text_json = json.dumps(text)
    # F40: resolve the visible element whose text matches, then click it with
    # the daemon's REAL mouse input through its resolved identity (and inside
    # its same-origin frame). A disabled element is refused; a JS click is no
    # longer used, so a framework that ignores synthetic events is not an
    # invisible failure any more.
    js = (
        "(() => {"
        "const needle = %s;"
        + _CSS_PATH_JS +
        "const candidates = Array.from(document.querySelectorAll('a, button, [role=\"button\"], [onclick]'));"        "const visible = candidates.filter(el => {"
        "  const rect = el.getBoundingClientRect();"
        "  if (rect.width < 3 || rect.height < 3) return false;"
        "  if (el.getClientRects().length === 0) return false;"
        "  const st = window.getComputedStyle(el);"
        "  if (st.visibility === 'hidden' || st.display === 'none' || st.opacity === '0') return false;"
        "  return true;"
        "});"
        "const norm = s => (s||'').trim().toLowerCase();"
        "let target = visible.find(el => (el.innerText||'').trim() === needle);"
        "if (!target) {"
        "  const low = needle.toLowerCase();"
        "  target = visible.find(el => norm(el.innerText).includes(low));"
        "}"
        "if (!target) { return JSON.stringify({found: false, clicked: false, error: 'not found', text: needle, href: location.href}); }"
        "const ident = jarvisIdent(target);"
        "return JSON.stringify({found: true, tag: target.tagName.toLowerCase(),"
        " label: (target.innerText||'').trim().slice(0,50),"
        " cssPath: ident.cssPath, frame: ident.frame,"
        " disabled: jarvisDisabled(target)});"
        "})()"
    ) % text_json
    payload, error = _real_locate(client, "click_text", js)
    if error:
        return _clip_result(error)
    if not payload.get("found"):
        return _clip_result(json.dumps(payload))
    return _real_click_target(client, "click_text", payload)


def _click_point_identity_error(client, session):
    """F39: coordinates are only meaningful inside the observation that
    produced them. Returns a refusal when the page no longer matches the look
    that anchored them (unknown state is a refusal too — a raw point must not
    be able to bypass mark validation)."""
    observation = (session or {}).get("look_capture")
    if not isinstance(observation, dict) or not observation.get("url"):
        return ("click_point refused: these coordinates are not anchored to a "
                "verified look of the current page. Call look first.")
    live = _capture_state(client)
    if live is None:
        return ("click_point refused: the page could not be re-checked, so the "
                "coordinates cannot be proven to belong to it. Call look again.")
    live_doc = _epoch_text(live.get("doc"))
    want_doc = _epoch_text(observation.get("doc"))
    live_dpr = _epoch_text(live.get("dpr"))
    want_dpr = _epoch_text(observation.get("dpr"))
    live_url = str(live.get("url") or "")
    if want_doc and live_doc and want_doc != live_doc:
        return ("click_point refused: the page is a different document than the "
                "look these coordinates came from. Call look again.")
    if want_dpr and live_dpr and want_dpr != live_dpr:
        return ("click_point refused: the display scale changed since the look "
                "(%s -> %s), so those coordinates no longer map to the same "
                "pixels. Call look again." % (want_dpr, live_dpr))
    if live_url and live_url != observation.get("url"):
        return ("click_point refused: the page moved to %s since the look (%s). "
                "Call look again." % (live_url, observation.get("url")))
    return ""


def _handle_click_point(client, session, arguments):
    x = arguments.get("x")
    y = arguments.get("y")
    try:
        x = int(x)
        y = int(y)
    except Exception:
        return _clip_result("click_point error: x and y must be integers")
    identity_error = _click_point_identity_error(client, session)
    if identity_error:
        return _clip_result(identity_error)
    # The model sees and reasons about the LOOK IMAGE; convert its
    # coordinates to original viewport pixels here so it never has to do
    # the scaling math itself.
    scale = 1.0
    try:
        scale = float(session.get("look_scale") or 1.0)
    except Exception:
        scale = 1.0
    if scale and scale != 1.0:
        x = int(round(x / scale))
        y = int(round(y / scale))
    size = session.get("viewport_size")
    if size:
        try:
            vw = int(size[0])
            vh = int(size[1])
            x = max(0, min(x, max(0, vw - 1)))
            y = max(0, min(y, max(0, vh - 1)))
        except Exception:
            pass
    # F40: elementFromPoint still decides WHAT sits at those pixels, but the
    # click itself goes through the daemon's real mouse input on the resolved
    # element identity — including a same-origin iframe target.
    js = (
        "(() => {"
        "const x = %d, y = %d;"
        + _CSS_PATH_JS +
        "const el = document.elementFromPoint(x, y);"
        "if (!el) { return JSON.stringify({found: false, clicked: false, error: 'no element at point', x: x, y: y, href: location.href}); }"
        "let target = el;"
        "for (let i = 0; i < 3; i++) {"
        "  if (!target) break;"
        "  const tag = target.tagName ? target.tagName.toLowerCase() : '';"
        "  const isClickable = tag==='a' || tag==='button' || target.hasAttribute('onclick') || target.getAttribute('role')==='button' || tag==='input' || tag==='video' || target.hasAttribute('contenteditable') || target.hasAttribute('tabindex');"
        "  if (isClickable) break;"
        "  target = target.parentElement;"
        "}"
        "if (!target) target = el;"
        "const ident = jarvisIdent(target);"
        "return JSON.stringify({found: true, tag: target.tagName.toLowerCase(),"
        " label: (target.innerText||'').trim().slice(0,20),"
        " cssPath: ident.cssPath, frame: ident.frame,"
        " disabled: jarvisDisabled(target),"
        " nonInteractive: !['a','button','input','video'].includes(target.tagName.toLowerCase()) && !target.hasAttribute('onclick') && target.getAttribute('role')!=='button' && !target.hasAttribute('contenteditable') && !target.hasAttribute('tabindex')});"
        "})()"
    ) % (int(x), int(y))
    payload, error = _real_locate(client, "click_point", js)
    if error:
        return _clip_result(error)
    if not payload.get("found"):
        return _clip_result(json.dumps(payload))
    result = _real_click_target(client, "click_point", payload)
    if payload.get("nonInteractive"):
        try:
            body = json.loads(result)
        except ValueError:
            return result
        body["feedback"] = ("clicked a non-interactive <%s> - probably no effect, "
                            "look for a real control" % payload.get("tag", "element"))
        return _clip_result(json.dumps(body))
    return result


_VERIFY_PLAYING_GAP_S = 1.2
_VERIFY_PLAYING_DIFF_THRESHOLD = 0.02


# ── G6 / F13: mark-or-css target resolution ──────────────────────────────
# scroll / select_option / set_checked / upload_file / download / drag_drop
# all resolve their target exactly like click_mark and fill_mark: a mark
# index re-resolves the mark's stored identity (F39); an explicit css
# dispatches straight to the daemon primitive (the daemon still probes the
# target before acting, so a gone element is reported honestly either way).
def _resolve_mark_or_css(client, session, verb, index_raw, css, need_visible=True, stats=None):
    """Resolve one typed interaction target to ``(css, frame, error)``.

    F13/F39/F40: the FRAME identity travels with the target, not just the CSS
    path. Dropping it sent the daemon primitive to the top document, where the
    same selector either matched a different element or nothing at all.
    """
    if index_raw is not None:
        idx = _parse_mark_idx(index_raw)
        if idx is None:
            return None, "", "%s error: index must be integer, got %r" % (verb, index_raw)
        marks = session.get("marks") or {}
        if idx not in marks:
            return None, "", _missing_mark_error(verb, idx, marks)
        info = marks[idx]
        if need_visible and not info.get("inView", True):
            return None, "", ("%s error: mark %d is off-screen (outside the current viewport) "
                              "- scroll it into view first" % (verb, idx))
        if info.get("cssPath"):
            _state, stale = _mark_target_state(client, info, stats)
            if stale:
                return None, "", "%s error: %s" % (verb, stale)
            return info["cssPath"], (info.get("frame") or ""), ""
        css = ""
    css = (css or "").strip()
    if not css:
        return None, "", "%s error: supply a mark index or a CSS selector." % verb
    return css, "", ""


def _locator_call(client, name, arguments, stats=None):
    """Call a daemon primitive with the resolved frame included (F40)."""
    # BA-00: the "locator" half of every typed-tool action.
    _loc = _Span("act.locator")
    try:
        return client.call_tool(name, arguments), None
    except Exception as exc:
        return None, "%s failed: %s" % (name, exc)
    finally:
        _loc.done(stats)


def _handle_scroll(client, arguments, stats=None):
    direction = (arguments.get("direction") or "down")
    try:
        amount = int(arguments.get("amount", 600))
    except Exception:
        amount = 600
    css = (arguments.get("css") or "").strip()
    payload = {"direction": direction, "amount": max(1, min(amount, 5000))}
    if css:
        payload["css"] = css
    raw, error = _locator_call(client, "scroll", payload, stats)
    if error:
        return _clip_result(error)
    parsed = _parse_locator_outcome(raw if isinstance(raw, str) else str(raw or ""))
    if not parsed["ok"]:
        return _clip_result("scroll failed: %s" % (parsed.get("reason") or raw))
    return _clip_result("scrolled %s %dpx (url=%s, navigated=%s)" % (
        direction, payload["amount"], parsed.get("url") or "",
        "yes" if parsed.get("navigated") else "no"))


def _control_readback_missing(verb, css):
    return ("%s failed: the daemon reported success but the control's state "
            "could not be read back from %s, so the change is unverified."
            % (verb, css))


def _handle_select_option(client, session, arguments, stats=None):
    value = arguments.get("value")
    if not isinstance(value, str) or not value.strip():
        return _clip_result("select_option error: value must be a non-empty string")
    css, frame, error = _resolve_mark_or_css(client, session, "select_option",
                                             arguments.get("index"),
                                             arguments.get("css"),
                                             stats=stats)
    if error:
        return _clip_result(error)
    raw, error = _locator_call(client, "select_option",
                               _locator_target({"value": value}, css, frame),
                               stats)
    if error:
        return _clip_result(error)
    parsed = _parse_locator_outcome(raw if isinstance(raw, str) else str(raw or ""))
    if not parsed["ok"]:
        return _clip_result("select_option failed: %s" % (parsed.get("reason") or raw))
    # F13: a state change is only real when it is READ BACK. The daemon's
    # "did it" line is not evidence of the value now in effect.
    state = _read_control_state(client, css)
    if not state or not state.get("found"):
        return _clip_result(_control_readback_missing("select_option", css))
    actual = str(state.get("value") or "")
    label = str(state.get("label") or "")
    wanted = value.strip()
    if wanted.lower() not in (actual.strip().lower(), label.strip().lower()):
        return _clip_result(
            "select_option failed: the control is now %r (label %r), not the "
            "requested %r - the selection did not take effect."
            % (actual, label, value))
    return _clip_result(json.dumps({
        "ok": True, "verified": True, "value": actual, "label": label,
        "url": state.get("url") or parsed.get("url") or "",
        "navigated": bool(parsed.get("navigated"))}))


def _handle_set_checked(client, session, arguments, stats=None):
    checked = arguments.get("checked")
    if not isinstance(checked, bool):
        return _clip_result("set_checked error: checked must be true or false")
    css, frame, error = _resolve_mark_or_css(client, session, "set_checked",
                                             arguments.get("index"),
                                             arguments.get("css"),
                                             stats=stats)
    if error:
        return _clip_result(error)
    raw, error = _locator_call(client, "set_checked",
                               _locator_target({"checked": checked}, css, frame),
                               stats)
    if error:
        return _clip_result(error)
    parsed = _parse_locator_outcome(raw if isinstance(raw, str) else str(raw or ""))
    if not parsed["ok"]:
        return _clip_result("set_checked failed: %s" % (parsed.get("reason") or raw))
    # F13: read the control's checked state back instead of inferring it.
    state = _read_control_state(client, css)
    if not state or not state.get("found") or "checked" not in state:
        return _clip_result(_control_readback_missing("set_checked", css))
    actual = bool(state.get("checked"))
    if actual != checked:
        return _clip_result(
            "set_checked failed: the control is now checked=%s, not %s - the "
            "change did not take effect."
            % ("true" if actual else "false", "true" if checked else "false"))
    return _clip_result(json.dumps({
        "ok": True, "verified": True, "checked": actual,
        "url": state.get("url") or parsed.get("url") or "",
        "navigated": bool(parsed.get("navigated"))}))


def _handle_upload_file(client, session, arguments, stats=None):
    upload_path = arguments.get("path")
    if not isinstance(upload_path, str) or not upload_path.strip():
        return _clip_result("upload_file error: path must be a non-empty string")
    # F13: bind uploads to approved file paths BEFORE dispatch — the scoped
    # grants from G2/F22, not a new policy surface.
    try:
        from backend.services import code_grants

        allowed, resolved, reason = code_grants.grant_check(upload_path)
    except Exception as exc:
        return _clip_result("upload_file error: grant check failed: %s" % exc)
    if not allowed:
        return _clip_result("upload_file refused: %s" % reason)
    if not os.path.isfile(resolved):
        return _clip_result("upload_file error: file does not exist: %s" % resolved)
    css, frame, error = _resolve_mark_or_css(client, session, "upload_file",
                                             arguments.get("index"),
                                             arguments.get("css"),
                                             stats=stats)
    if error:
        return _clip_result(error)
    # F13: reading the file is NOT authority to send it to any site. The
    # (file, destination origin) pair needs its own approval, checked against
    # the page the file would actually land in — fail closed when the
    # destination cannot be established.
    origin_error = _upload_origin_error(session, client)
    if origin_error:
        if frame:
            origin_error += " (target frame: %s)" % frame
        return _clip_result(origin_error)
    raw, error = _locator_call(
        client, "upload_file",
        _locator_target({"paths": [resolved]}, css, frame), stats)
    if error:
        return _clip_result(error)
    parsed = _parse_locator_outcome(raw if isinstance(raw, str) else str(raw or ""))
    if not parsed["ok"]:
        return _clip_result("upload_file failed: %s" % (parsed.get("reason") or raw))
    return _clip_result("uploaded %s (url=%s, navigated=%s)" % (
        os.path.basename(resolved), parsed.get("url") or "",
        "yes" if parsed.get("navigated") else "no"))


def _download_artifact(text):
    """The on-disk artifact a download result names (literal case kept)."""
    for line in (text or "").splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        for key in ("artifact=", "saved=", "path=", "file="):
            if lowered.startswith(key):
                return stripped[len(key):].strip().strip('"')
    match = re.search(r"(?:artifact|saved|path|file)=([^\s]+)", text or "",
                      re.IGNORECASE)
    return match.group(1).strip().strip('"') if match else ""


def _handle_download(client, session, arguments, stats=None):
    css, frame, error = _resolve_mark_or_css(client, session, "download",
                                             arguments.get("index"),
                                             arguments.get("css"),
                                             stats=stats)
    if error:
        return _clip_result(error)
    raw, error = _locator_call(client, "download", _locator_target({}, css, frame),
                               stats)
    if error:
        return _clip_result(error)
    text = raw if isinstance(raw, str) else str(raw or "")
    parsed = _parse_locator_outcome(text)
    if not parsed["ok"]:
        return _clip_result("download failed: %s" % (parsed.get("reason") or text))
    # F13: a "download complete" event with no artifact is not a download. The
    # named file must exist on disk; otherwise the result is a failure so the
    # task can never claim a file it does not have.
    artifact = _download_artifact(text)
    if not artifact:
        return _clip_result(
            "download failed: the daemon reported completion but named no "
            "artifact, so no file can be shown to exist (url=%s)"
            % (parsed.get("url") or ""))
    if not os.path.isfile(artifact):
        return _clip_result(
            "download failed: the reported artifact %s does not exist on disk "
            "(url=%s)" % (artifact, parsed.get("url") or ""))
    return _clip_result("download complete: %s (url=%s)" % (
        artifact, parsed.get("url") or ""))


def _handle_drag_drop(client, session, arguments, stats=None):
    src, src_frame, error = _resolve_mark_or_css(client, session, "drag_drop",
                                                 arguments.get("index"),
                                                 arguments.get("css"),
                                                 stats=stats)
    if error:
        return _clip_result(error)
    dst, dst_frame, target_error = _resolve_mark_or_css(
        client, session, "drag_drop",
        arguments.get("target_index"), arguments.get("target_css"),
        stats=stats)
    if target_error:
        return _clip_result(target_error)
    payload = {"css": src, "target_css": dst}
    if src_frame:
        payload["frame"] = src_frame
    if dst_frame:
        payload["target_frame"] = dst_frame
    raw, error = _locator_call(client, "drag_drop", payload, stats)
    if error:
        return _clip_result(error)
    parsed = _parse_locator_outcome(raw if isinstance(raw, str) else str(raw or ""))
    if not parsed["ok"]:
        return _clip_result("drag_drop failed: %s" % (parsed.get("reason") or raw))
    return _clip_result("dragged onto target (url=%s, navigated=%s)" % (
        parsed.get("url") or "", "yes" if parsed.get("navigated") else "no"))


def _handle_verify_playing(client):
    """Report REAL media state; pixel motion alone never proves playback.

    F41: whole-page animation used to be the evidence — animated ads, spinners
    and carousels made a paused video look "PLAYING". The postcondition is now
    the media element's own state and advancing media time, read twice across
    the same gap. Pixel motion is kept only as *weak* evidence and is reported
    as uncertain, never as PLAYING.
    """
    probe = _probe_media_state(client)
    if probe is not None:
        players = probe.get("players") or []
        if players:
            first = _media_signature(players)
            time.sleep(_VERIFY_PLAYING_GAP_S)
            second_probe = _probe_media_state(client)
            if second_probe is not None:
                second = _media_signature(second_probe.get("players") or [])
                verdict = _media_verdict(first, second)
                if verdict:
                    return _clip_result(verdict)

    return _verify_playing_by_motion(client, probe)


#: F41: real media state, including same-origin iframe players. Cross-origin
#: players stay invisible to script, which is exactly why their absence must
#: be reported as uncertainty rather than papered over with page motion.
_MEDIA_STATE_JS = (
    "(() => {"
    "const out = [];"
    "const collect = (doc) => {"
    "  let els = [];"
    "  try { els = Array.from(doc.querySelectorAll('video,audio')); } catch(e) { return; }"
    "  for (const el of els) {"
    "    out.push({"
    "      tag: el.tagName.toLowerCase(),"
    "      paused: !!el.paused,"
    "      ended: !!el.ended,"
    "      muted: !!el.muted,"
    "      volume: (typeof el.volume === 'number') ? el.volume : null,"
    "      currentTime: (typeof el.currentTime === 'number') ? el.currentTime : null,"
    "      duration: (typeof el.duration === 'number' && isFinite(el.duration)) ? el.duration : null,"
    "      readyState: el.readyState,"
    "      hasVideo: el.tagName.toLowerCase() === 'video' && el.videoWidth > 0,"
    "      src: (el.currentSrc || el.src || '').slice(0, 200)"
    "    });"
    "  }"
    "  try {"
    "    for (const f of Array.from(doc.querySelectorAll('iframe'))) {"
    "      try { if (f.contentDocument) collect(f.contentDocument); } catch(e) {}"
    "    }"
    "  } catch(e) {}"
    "};"
    "collect(document);"
    "return JSON.stringify({players: out, url: location.href, title: document.title});"
    "})()"
)


def _probe_media_state(client):
    """F41: read the page's media elements, or None when unreadable."""
    try:
        raw = _probe(client, _MEDIA_STATE_JS)
    except Exception as exc:
        logging.debug("verify_playing media probe failed: %s", exc)
        return None
    payload = _parse_json_result(raw)
    if not isinstance(payload, dict):
        return None
    return payload


def _media_signature(players):
    """Comparable snapshot of one media reading."""
    return [
        {
            "tag": str(p.get("tag") or ""),
            "paused": bool(p.get("paused")),
            "ended": bool(p.get("ended")),
            "currentTime": p.get("currentTime"),
            "duration": p.get("duration"),
            "readyState": p.get("readyState"),
            "src": str(p.get("src") or ""),
        }
        for p in players
        if isinstance(p, dict)
    ]


def _media_verdict(first, second):
    """F41: verdict from real media state, or None when evidence is absent."""
    if not first or not second:
        return None
    comparable = min(len(first), len(second))
    if comparable <= 0:
        return None

    advanced = []
    for index in range(comparable):
        a, b = first[index], second[index]
        t_a, t_b = a.get("currentTime"), b.get("currentTime")
        if (
            isinstance(t_a, (int, float)) and isinstance(t_b, (int, float))
            and t_b > t_a + 0.05
        ):
            advanced.append((index, t_a, t_b, b))

    if advanced:
        index, t_a, t_b, latest = advanced[0]
        duration = latest.get("duration")
        duration_text = (" of %.1fs" % duration) if isinstance(duration, (int, float)) else ""
        return (
            "verify_playing: PLAYING - media time advanced %.2fs -> %.2fs%s over "
            "%.1fs (player %d, paused=%s)."
            % (t_a, t_b, duration_text, _VERIFY_PLAYING_GAP_S, index,
               latest.get("paused"))
        )

    # No media time moved. Say why, using the element's own state.
    first_player = first[0]
    latest = second[0]
    if latest.get("ended"):
        return (
            "verify_playing: ENDED - the media element reports ended=true "
            "(media time %.2fs)." % (latest.get("currentTime") or 0.0)
        )
    if latest.get("paused") and not first_player.get("paused"):
        return "verify_playing: PAUSED - the media element is paused."
    if latest.get("paused"):
        return (
            "verify_playing: PAUSED - the media element reports paused=true "
            "(media time %.2fs of %s)."
            % (latest.get("currentTime") or 0.0,
               ("%.1fs" % latest["duration"])
               if isinstance(latest.get("duration"), (int, float)) else "unknown")
        )
    if latest.get("readyState") in (0, 1):
        return (
            "verify_playing: NOT READY - the media element reports readyState=%s "
            "and its media time is not advancing." % latest.get("readyState")
        )
    return (
        "verify_playing: UNCERTAIN - the media element is not paused but its "
        "media time did not advance over %.1fs, so I cannot call this playing."
        % _VERIFY_PLAYING_GAP_S
    )


def _verify_playing_by_motion(client, probe):
    """Weak evidence only: page motion is NOT proof that media is playing."""
    tmp_paths = []
    try:
        frames = []
        for _ in range(2):
            fd, tmp_path = tempfile.mkstemp(prefix="jarvis_vp_", suffix=".png")
            os.close(fd)
            tmp_paths.append(tmp_path)
            try:
                client.call_tool("screenshot", {"path": tmp_path})
                frames.append(Image.open(tmp_path).convert("L"))
            except Exception as exc:
                return _clip_result(
                    "verify_playing failed: screenshot error: %s" % exc)
            if len(frames) == 1:
                time.sleep(_VERIFY_PLAYING_GAP_S)
        target_w = 160

        def _small(img):
            w, h = img.size
            if w <= target_w:
                return img
            new_h = max(1, int(round(h * target_w / float(w))))
            try:
                return img.resize((target_w, new_h), Image.LANCZOS)
            except Exception:
                return img.resize((target_w, new_h))

        frame_a = _small(frames[0])
        frame_b = _small(frames[1])
        if frame_a.size != frame_b.size:
            try:
                frame_b = frame_b.resize(frame_a.size)
            except Exception:
                pass
        try:
            delta = ImageChops.difference(frame_a, frame_b)
            data = list(delta.getdata())
            ratio = float(sum(data)) / (255.0 * max(1, len(data)))
        except Exception as exc:
            return _clip_result(
                "verify_playing failed: could not compare frames: %s" % exc)

        reason = (
            "no media element was readable (a cross-origin player is the usual "
            "reason)"
            if probe is not None
            else "the page could not be inspected for media elements"
        )
        if ratio >= _VERIFY_PLAYING_DIFF_THRESHOLD:
            return _clip_result(
                "verify_playing: UNCERTAIN - %s, so I can only report that the "
                "page is animating (frame diff %.4f >= %.2f over %.1fs). Motion "
                "from ads, spinners or carousels does NOT prove playback."
                % (reason, ratio, _VERIFY_PLAYING_DIFF_THRESHOLD,
                   _VERIFY_PLAYING_GAP_S)
            )
        return _clip_result(
            "verify_playing: STATIC - %s and no page motion was detected "
            "(frame diff %.4f < %.2f over %.1fs). If you expected video, the "
            "player may be paused."
            % (reason, ratio, _VERIFY_PLAYING_DIFF_THRESHOLD,
               _VERIFY_PLAYING_GAP_S)
        )
    finally:
        for tmp_path in tmp_paths:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass


# ── Virtual tool routing ────────────────────────────────────────────────────

def _log_tool_result(name, result_text):
    """Append a per-tool RESULT line to the activity log for post-mortems.

    Text only - NEVER base64/image data. Most tools are capped at 200
    chars; fill/fill_mark results stay at full length (short JSON
    diagnostics the model needs to self-correct with).
    """
    if result_text is None:
        return
    text = result_text if isinstance(result_text, str) else str(result_text)
    text = " ".join(text.split())
    cap = 200
    if name in ("fill", "fill_mark"):
        cap = 5000
    # F21: a filled field can echo back a password or token, and a page can
    # echo one back inside an otherwise innocent result. Scrub before the
    # line is persisted, whatever the tool.
    text = tool_policy.redact_for_egress(text)
    if len(text) > cap:
        text = text[:cap] + "..."
    append_activity_line("RESULT %s: %s\n" % (name, text))


def _effect_signature(name, arguments):
    """F08: stable identity of one effect (tool + resolved arguments)."""
    return FailureTracker.key(name, arguments)[1]


def _already_committed(session, name, arguments):
    """F08: was this exact effect committed by the SUSPENDED run we resumed?

    Only checkpoint-restored effects count. Within one live run the model is
    allowed to repeat an action (that is its own business); the audit's rule is
    that a RESUME must not replay what the suspended run already did.
    """
    try:
        done = session.get("resumed_effects")
    except Exception:
        return False
    if not done:
        return False
    return _effect_signature(name, arguments) in done


def _note_committed(session, name, arguments, result_text=""):
    """F08: record a committed effect for checkpointing / do-not-repeat."""
    try:
        signature = _effect_signature(name, arguments)
        session.setdefault("completed_effects", set()).add(signature)
        label = "%s %s" % (name, FailureTracker.target_key(arguments) or
                           signature[:60])
        actions = session.setdefault("completed_actions", [])
        if label not in actions:
            actions.append(label)
        if len(actions) > 50:
            del actions[:-50]
    except Exception:
        pass


def _note_trace(session, name, arguments, result_text="", ok=True):
    """F09: record one executed step of the verified run trace.

    Bounded (40 steps, 300 chars of observation each) and masked: a trace is
    persisted as a candidate procedure, so it crosses the same redaction
    boundary as every other stored field.
    """
    try:
        entries = session.setdefault("trace", [])
        if len(entries) >= 40:
            return
        observation = tool_policy.mask_secrets(
            _clip_result(str(result_text or "")))[:300]
        entries.append({
            "tool": str(name)[:60],
            "args": _trace_args(arguments),
            "observation": observation,
            "ok": bool(ok),
        })
    except Exception:
        pass


def _trace_args(arguments):
    """The arguments of one traced step, masked and field-capped."""
    try:
        if not isinstance(arguments, dict):
            return {}
        safe = {}
        for key, value in list(arguments.items())[:12]:
            if isinstance(value, (list, tuple)):
                safe[str(key)[:40]] = [
                    tool_policy.mask_secrets(str(v))[:200]
                    for v in list(value)[:8]]
            else:
                safe[str(key)[:40]] = tool_policy.mask_secrets(
                    str(value))[:200]
        return safe
    except Exception:
        return {}


def _tracker_for_tool(session, name, arguments):
    """F05: (tracker, action_key, target_key) for one tool call.

    Every tool — native or virtual — goes through the same structured
    bookkeeping, so the attempt allowance is shared instead of the virtual
    path silently escaping it.
    """
    tracker = (session or {}).get("failures")
    if not isinstance(tracker, FailureTracker):
        tracker = FailureTracker()
        try:
            session["failures"] = tracker
        except Exception:
            pass
    return (tracker,
            FailureTracker.key(name, arguments),
            FailureTracker.target_key(arguments))


#: F05: tool results whose TEXT reports a failure, so a returned string is not
#: mistaken for a successful execution.
#:
#: F13/F40: the typed interaction handlers report failures as
#: "<verb> failed: ..." ("select_option failed: ...", "download failed: ...",
#: "click_text failed: ..."). Those texts carry a verb prefix, so an anchored
#: match on "failed" alone missed them and a failed interaction could still be
#: recorded as a successful effect. The verb-prefixed form is part of the
#: pattern now (still anchored, so a success sentence that merely mentions a
#: past failure is unaffected).
_TOOL_FAILURE_TEXT_RE = re.compile(
    r"^\s*(?:"
    r"[a-z][a-z_]{1,30} failed\b|"
    r"tool failed|failed|error|err\b|exception|traceback|blocked|"
    r"not replayed|unknown tool|permission denied|timed out|timeout|"
    r"no such element|element not found|not found|unable to|"
    r"virtual tool \w+ failed|could not|couldn't"
    r")", re.IGNORECASE)


def _tool_text_reports_failure(result_text):
    """True when a tool's own result text says the action failed (F05)."""
    if not isinstance(result_text, str):
        return False
    body = result_text.strip()
    if not body:
        return True
    if _TOOL_FAILURE_TEXT_RE.match(body):
        return True
    # A structured JSON result with ok=false is a failure whatever it says.
    if body.startswith("{"):
        try:
            payload = json.loads(body)
        except Exception:
            return False
        if isinstance(payload, dict) and payload.get("ok") is False:
            return True
    return False


#: F05: read-only tools whose success counts as an independent observation of
#: their target (the evidence that reconciles an unknown-outcome mutation).
_OBSERVATION_TOOLS = frozenset((
    "look", "batch_probe", "wait_for", "verify_playing", "screenshot",
    "list_tabs", "inspect_tabs", "read_file", "read_locator", "get_text",
))


def _run_virtual_tool(client, name, arguments, session, stats=None):
    """Dispatch a virtual tool; returns (result_text, image_b64|None).

    BA-00: the handlers that own measurable stages (look, click_mark,
    fill_mark, wait_for) receive the task stats; every other handler keeps
    its exact signature."""
    if name == "look":
        return _handle_look(client, session, stats)
    if name == "click_mark":
        return _handle_click_mark(client, session, arguments,
                                 stats=stats), None
    if name == "fill_mark":
        return _handle_fill_mark(client, session, arguments,
                                stats=stats), None
    if name == "verify_playing":
        return _handle_verify_playing(client), None
    if name == "batch_probe":
        return _handle_batch_probe(client, arguments), None
    if name == "wait_for":
        return _handle_wait_for(client, arguments, stats=stats,
                                session=session), None
    if name == "fill":
        return _handle_fill(client, arguments), None
    if name == "click_text":
        return _handle_click_text(client, arguments), None
    if name == "click_point":
        return _handle_click_point(client, session, arguments), None
    if name == "scroll":
        return _handle_scroll(client, arguments, stats=stats), None
    if name == "select_option":
        return _handle_select_option(client, session, arguments,
                                     stats=stats), None
    if name == "set_checked":
        return _handle_set_checked(client, session, arguments,
                                   stats=stats), None
    if name == "upload_file":
        return _handle_upload_file(client, session, arguments,
                                   stats=stats), None
    if name == "download":
        return _handle_download(client, session, arguments,
                                stats=stats), None
    if name == "drag_drop":
        return _handle_drag_drop(client, session, arguments,
                                 stats=stats), None
    return None


def _run_one_tool(client, history, call, session=None, stats=None):
    """Execute one tool call.

    Reads (and unknown daemon tools) keep the 3-attempt transport retry;
    MUTATION tools get a single attempt (never auto-replay a possibly
    committed mutation after an ambiguous transport failure). A tool
    failure is delivered AS the tool result so the model can adapt - the
    task continues. The 3rd identical failure is blocked WITHOUT
    executing and ends the task as partial (F05).

    Returns None normally, or the block-evidence string
    ('blocked: repeated failure of <tool>') when the call was blocked.
    """
    if session is None:
        session = {}
    name = call.get("name", "")
    arguments = call.get("arguments") or {}
    # F13: a provider response whose tool arguments were not valid JSON used
    # to become `{}` — an EMPTY call that still executed, with whatever the
    # handler defaults happened to be. Nothing may be dispatched now: the
    # model gets the parse error back and re-issues the call.
    parse_error = call.get("parse_error")
    if parse_error:
        parse_text = _clip_result(
            "tool call refused: the arguments for %s were not valid JSON (%s), "
            "so NOTHING was executed. Re-issue the call with valid JSON."
            % (name, parse_error))
        history.append({
            "role": "tool",
            "tool_call_id": call.get("id") or "call_%s" % name,
            "name": name,
            "content": parse_text,
        })
        append_activity_line("REFUSED %s: malformed arguments (%s)\n"
                            % (name, parse_error))
        _log_tool_result(name, "REFUSED " + parse_text)
        _ba00_note_result(stats, parse_text)
        return None
    # F21: arguments cross the redacted egress boundary before they reach the
    # activity log, so a password/token typed into a form never lands in a
    # persisted trace (any depth, any key).
    try:
        log_args = json.dumps(
            tool_policy.redact_for_egress(arguments), default=str)
    except Exception:
        log_args = "{}"
    if len(log_args) > 200:
        log_args = log_args[:200] + "..."
    append_activity_line("TOOL %s %s\n" % (name, log_args))
    phrase = _NARRATION_PHRASES.get(name)
    if phrase:
        narrate_activity(phrase)

    # F17: policy at the EXECUTION boundary. The tool list the model *sees*
    # was already filtered; this validates the name it actually returned, its
    # arguments and the authority the call needs. A MODEL-authored call may
    # only dispatch a tool the model was offered; the hidden daemon primitives
    # require an explicit internal origin, so naming one is not enough.
    internal = bool(session.get("_internal_tool_call"))
    if internal:
        allowlist = _TOOL_DISPATCH_ALLOWLIST
        origin = "internal"
    else:
        allowlist = _MODEL_DISPATCH_ALLOWLIST
        origin = "model"
    if not internal and arguments and not isinstance(arguments, dict):
        malformed_text = _clip_result(
            "tool call refused: %s arguments must be a JSON object, got %s - "
            "NOTHING was executed." % (name, type(arguments).__name__))
        history.append({
            "role": "tool",
            "tool_call_id": call.get("id") or "call_%s" % name,
            "name": name,
            "content": malformed_text,
        })
        append_activity_line("REFUSED %s: arguments are not an object\n" % name)
        _log_tool_result(name, "REFUSED " + malformed_text)
        _ba00_note_result(stats, malformed_text)
        return None
    # F13: the schema validated here is the SAME object the model was shown
    # (see _VIRTUAL_TOOL_SPECS), so the advertisement cannot drift from what
    # the handler will really accept.
    decision = tool_policy.validate_dispatch(
        name, arguments, allowlist, schema=_virtual_schema(name),
        grants=_grants_for(session), origin=origin)
    if decision.allowed and name in _VIRTUAL_TOOL_NAMES:
        rule_error = _virtual_argument_error(name, arguments)
        if rule_error:
            decision = tool_policy.Decision(
                False, name, tool_policy.classify_operation(name, arguments),
                rule_error, {}, [])
    if not decision.allowed:
        blocked_text = _clip_result(
            "tool blocked by policy: %s" % decision.reason)
        history.append(
            {
                "role": "tool",
                "tool_call_id": call.get("id") or "call_%s" % name,
                "name": name,
                "content": blocked_text,
            }
        )
        append_activity_line("POLICY BLOCK %s: %s\n" % (name, decision.reason))
        _log_tool_result(name, "BLOCKED " + blocked_text)
        _ba00_note_result(stats, blocked_text)
        return None
    try:
        _sw_t0 = time.monotonic()
        _sw_start = _sw_timestamp()
    except Exception:
        _sw_t0 = None
        _sw_start = ""
    # Intercept virtual tools
    if name in _VIRTUAL_TOOL_NAMES:
        # F05: virtual tools share the SAME attempt allowance and the same
        # structured-outcome bookkeeping as native tools. They used to return
        # before the tracker block, so a virtual tool could fail forever.
        vtracker, vkey, vtarget = _tracker_for_tool(session, name, arguments)
        if vtracker is not None and vtracker.failures_for(vkey) >= \
                FailureTracker.BLOCK_AFTER - 1:
            blocked_text = _clip_result(
                "action blocked after 3 identical failures (tool: %s) - "
                "the task will finish without it" % name)
            history.append({
                "role": "tool",
                "tool_call_id": call.get("id") or "call_%s" % name,
                "name": name,
                "content": blocked_text,
            })
            _sw_log_tool(name, _sw_start, _sw_t0, stats)
            _log_tool_result(name, "BLOCKED " + blocked_text)
            try:
                session["task_blocked"] = "blocked: repeated failure of %s" % name
            except Exception:
                pass
            return "blocked: repeated failure of %s" % name
        try:
            if name in _ALL_MUTATION_TOOLS:
                # F08: this effect may already have been committed by the
                # suspended run this one resumed — never make it twice.
                if _already_committed(session, name, arguments):
                    committed_text = _clip_result(
                        "already done earlier in this task (tool: %s) - the "
                        "effect was NOT repeated; continue with the next "
                        "step or finish." % name)
                    history.append({
                        "role": "tool",
                        "tool_call_id": call.get("id") or "call_%s" % name,
                        "name": name,
                        "content": committed_text,
                    })
                    _sw_log_tool(name, _sw_start, _sw_t0, stats)
                    _log_tool_result(name, "ALREADY-COMMITTED " + committed_text)
                    return None
                allowed, unknown_reason = vtracker.replay_decision(vkey, vtarget)
                if not allowed:
                    reconcile_text = _clip_result(
                        "not replayed: the previous %s attempt's outcome is "
                        "unknown (%s) - it may already have been applied. "
                        "Observe the current state with a read-only tool "
                        "(look/probe) first, then retry only if the change is "
                        "really missing." % (name, unknown_reason))
                    history.append({
                        "role": "tool",
                        "tool_call_id": call.get("id") or "call_%s" % name,
                        "name": name,
                        "content": reconcile_text,
                    })
                    _sw_log_tool(name, _sw_start, _sw_t0, stats)
                    _log_tool_result(name, "OUTCOME-UNKNOWN " + reconcile_text)
                    _ba00_note_result(stats, reconcile_text)
                    return None
            outcome = _run_virtual_tool(client, name, arguments, session)
            if outcome is None:
                result_text = _clip_result(
                    "virtual tool %s failed: no handler" % name)
                image_b64 = None
            else:
                result_text, image_b64 = outcome
            if name == "verify_playing":
                # RANK 4: keep the playback verdict - it is this task's media
                # proof, or the honest record that no proof exists.
                try:
                    session["_media_verdict"] = result_text or ""
                except Exception:
                    pass
        except Exception as exc:
            # virtual handler unexpected failure -> return as tool error text
            result_text = _clip_result("virtual tool %s failed: %s" % (name, exc))
            image_b64 = None
            # F05: a virtual mutation that died on the transport may have
            # committed — same outcome-unknown rule as the native path.
            if name in _ALL_MUTATION_TOOLS and \
                    FailureTracker.is_ambiguous_error(exc):
                vtracker.mark_outcome_unknown(
                    vkey, vtarget, "%s: %s" % (name, str(exc)[:120]))
        # F05: a textual failure is still a FAILURE. Recording every returned
        # text as a success let an always-failing action run forever.
        if vtracker is not None:
            if _tool_text_reports_failure(result_text):
                vtracker.record_failure(vkey)
            else:
                vtracker.record_success(vkey)
                if name in _OBSERVATION_TOOLS:
                    vtracker.note_observation(vtarget, name)
                elif name in _ALL_MUTATION_TOOLS:
                    # F08: remember the committed effect (checkpoint resume).
                    _note_committed(session, name, arguments, result_text)
            # F09: every executed step joins the verified trace.
            _note_trace(session, name, arguments, result_text,
                        ok=not _tool_text_reports_failure(result_text))
        # F21: a virtual result is page text (a mark table, a probe value, a
        # typed field) and it travels on to the MODEL as well as the log, so it
        # crosses the same redacted egress boundary. This used to be the one
        # result path that reached the transcript unmasked.
        if isinstance(result_text, str):
            result_text = tool_policy.redact_for_egress(result_text)
        _sw_log_tool(name, _sw_start, _sw_t0, stats)
        _log_tool_result(name, result_text)
        _ba00_note_result(stats, result_text)
        # G6 / F47: tab-speaking results keep the broker current — a
        # navigation, tab switch or new tab changes "which tab is which", and
        # every subsystem must read the same picture (not its own guess).
        # F17: a virtual observation (look / batch_probe / wait_for) is NOT
        # tab-speaking — its text can contain a page-authored tab listing.
        _publish_daemon_tabs(result_text, name)
        message = {
            "role": "tool",
            "tool_call_id": call.get("id") or "call_%s" % name,
            "name": name,
            "content": result_text,
        }
        if image_b64:
            message["image_b64"] = image_b64
        history.append(message)
        return None

    # F05: task-level identical-action cap — the SAME allowance for native and
    # virtual tools now that both pass through here.
    tracker, action_key, target = _tracker_for_tool(session, name, arguments)
    if tracker is None:
        tracker = FailureTracker()
    if tracker.failures_for(action_key) >= FailureTracker.BLOCK_AFTER - 1:
        blocked_text = _clip_result(
            "action blocked after 3 identical failures (tool: %s) - "
            "the task will finish without it" % name
        )
        history.append(
            {
                "role": "tool",
                "tool_call_id": call.get("id") or "call_%s" % name,
                "name": name,
                "content": blocked_text,
            }
        )
        _sw_log_tool(name, _sw_start, _sw_t0, stats)
        _log_tool_result(name, "BLOCKED " + blocked_text)
        blocked_evidence = "blocked: repeated failure of %s" % name
        _ba00_note_result(stats, blocked_text)
        try:
            session["task_blocked"] = blocked_evidence
        except Exception:
            pass
        return blocked_evidence

    if name in _MUTATION_TOOLS:
        # F08: never repeat an effect this same task already committed (the
        # resumed-run case) — the audit's "clarification does not replay
        # committed actions".
        if _already_committed(session, name, arguments):
            committed_text = _clip_result(
                "already done earlier in this task (tool: %s) - the effect was "
                "NOT repeated; continue with the next step or finish." % name)
            history.append({
                "role": "tool",
                "tool_call_id": call.get("id") or "call_%s" % name,
                "name": name,
                "content": committed_text,
            })
            _sw_log_tool(name, _sw_start, _sw_t0, stats)
            _log_tool_result(name, "ALREADY-COMMITTED " + committed_text)
            _ba00_note_result(stats, committed_text)
            return None
        # F05: a mutation whose EARLIER attempt ended with a lost response may
        # already have committed. Replaying the identical action without
        # independent evidence of non-commitment is exactly the double-submit
        # the audit found, across model turns — so it is refused until an
        # observation of the same target proves the current state.
        allowed, unknown_reason = tracker.replay_decision(action_key, target)
        if not allowed:
            reconcile_text = _clip_result(
                "not replayed: the previous %s attempt's outcome is unknown "
                "(%s) - it may already have been applied. Observe the current "
                "state with a read-only tool (look/probe/read) first, then "
                "retry only if the change is really missing."
                % (name, unknown_reason))
            history.append({
                "role": "tool",
                "tool_call_id": call.get("id") or "call_%s" % name,
                "name": name,
                "content": reconcile_text,
            })
            _sw_log_tool(name, _sw_start, _sw_t0, stats)
            _log_tool_result(name, "OUTCOME-UNKNOWN " + reconcile_text)
            _ba00_note_result(stats, reconcile_text)
            return None
        # Single attempt: never auto-replay a possibly committed mutation
        # after an ambiguous transport failure.
        max_attempts = 1
    else:
        # Reads (and any other daemon tool) keep the 3-attempt retry.
        max_attempts = 3
    result_text = None
    last_error = None
    for attempt in range(max_attempts):
        try:
            result_text = client.call_tool(name, arguments)
            break
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts - 1:
                # BA-03: capped to the remaining task budget (plain 0.5 s
                # sleep when unbound). A spent budget makes the NEXT
                # attempt fail fast on its sliced timeout instead of
                # granting fresh time here.
                try:
                    budget.wait(None, 0.5)
                except Exception:
                    pass
    if result_text is None:
        # One optional tool failing must NOT kill the whole run (a single
        # read_file silence once aborted the entire task): deliver the
        # error text AS the tool result so the model can adapt and carry
        # on. Task abort is reserved for model-call failures ONLY.
        # The failure is also recorded as task evidence (F03) and in the
        # identical-action tracker (F05).
        fail_count = tracker.record_failure(action_key)
        # F05: an ambiguous transport failure on a MUTATION leaves the effect
        # undetermined — record it as outcome-unknown so a later identical
        # attempt must first reconcile against observed state.
        if name in _MUTATION_TOOLS and FailureTracker.is_ambiguous_error(last_error):
            tracker.mark_outcome_unknown(
                action_key, target,
                "%s: %s" % (name, str(last_error)[:120]))
        try:
            session.setdefault("tool_failures", []).append(
                "%s: %s" % (name, str(last_error)[:120])
            )
        except Exception:
            pass
        if max_attempts > 1:
            result_text = _clip_result(
                "tool failed after retries: %s (tool: %s) - continue with "
                "other tools or finish the task without it"
                % (last_error, name)
            )
        else:
            result_text = _clip_result(
                "tool failed: %s (tool: %s) - continue with "
                "other tools or finish the task without it"
                % (last_error, name)
            )
        if fail_count >= FailureTracker.NOTE_AFTER:
            result_text = _clip_result(
                result_text + "\n" + FailureTracker.REPEAT_NOTE)
        if name == "evaluate" and "syntaxerror" in str(last_error).lower():
            result_text = _clip_result(result_text + "\nHint: wrap multi-statement JS in an IIFE like (function(){ ... })() and avoid trailing semicolons.")
        # F21: mask at the boundary so a credential that surfaced in an error
        # never reaches the model, memory or the UI.
        result_text = tool_policy.mask_secrets(_clip_result(result_text))
        history.append(
            {
                "role": "tool",
                "tool_call_id": call.get("id") or "call_%s" % name,
                "name": name,
                "content": result_text,
            }
        )
        _sw_log_tool(name, _sw_start, _sw_t0, stats)
        _log_tool_result(name, "FAILED " + result_text)
        _ba00_note_result(stats, result_text)
        return None
    # WI3: make evaluate SyntaxError actionable (do not auto-retry)
    if name == "evaluate" and isinstance(result_text, str) and "syntaxerror" in result_text.lower():
        result_text = result_text + "\nHint: wrap multi-statement JS in an IIFE like (function(){ ... })() and avoid trailing semicolons."
    # F21: a token, cookie or password echoed by a page must not travel on to
    # the model, the activity log or the UI.
    if isinstance(result_text, str):
        result_text = tool_policy.mask_secrets(result_text)
    # F05: a returned TEXT is not automatically a success. A tool that reports
    # its failure in the result string used to reset the identical-action
    # counter every turn, so the same broken action could run forever.
    if _tool_text_reports_failure(result_text):
        fail_count = tracker.record_failure(action_key)
        if fail_count >= FailureTracker.NOTE_AFTER:
            result_text = _clip_result(
                result_text + "\n" + FailureTracker.REPEAT_NOTE)
    else:
        # A successful execution resets this action's failure counter (F05).
        tracker.record_success(action_key)
        if name in _OBSERVATION_TOOLS:
            # F05: an independent observation of a target is what reconciles an
            # unknown-outcome mutation of that same target.
            tracker.note_observation(target, name)
        elif name in _ALL_MUTATION_TOOLS:
            # F08: remember the committed effect so a resumed run never
            # repeats it.
            _note_committed(session, name, arguments, result_text)
    # F09: a step that reported its failure is traced as a failure.
    _note_trace(session, name, arguments, result_text,
                ok=not _tool_text_reports_failure(result_text))
    _sw_log_tool(name, _sw_start, _sw_t0, stats)
    _log_tool_result(name, result_text)
    _ba00_note_result(stats, result_text)
    # G6 / F47: native daemon tab tools speak about tabs too — publish what
    # they said (previously only dead code, never called).
    if name in _TAB_PUBLISH_TOOLS:
        _publish_daemon_tabs(result_text, name)
    history.append(
        {
            "role": "tool",
            "tool_call_id": call.get("id") or "call_%s" % name,
            "name": name,
            "content": _clip_result(result_text),
        }
    )
    return None


# ── BA-00: outcome classification ───────────────────────────────────────────
# One choke point for the wasted-turn census. Every tool result — virtual or
# native — is classified here, so no handler signature needs a stats param.
# Classification is by stable, distinctive F39/F05 refusal substrings, never
# by full-text equality (callers prepend "click_mark error: " etc.).

#: The one text _mark_target_state returns ONLY for `mut` drift.
_MUT_STALE_MARKER = "That mark is stale (the page content changed since the look)"

#: Every OTHER staleness refusal _mark_target_state / the missing-mark path
#: can produce. Kept in the same order as _mark_target_state's checks.
_OTHER_STALE_MARKERS = (
    "carries no page-identity stamp",
    "did not report which document it is",
    "from an older page (the document changed",
    "observed at a different display scale",
    "belongs to a different page",
    "belongs to a different frame",
    "element is gone from the page",
    "Target check failed",
    "Target check returned no readable result",
    "not found. Available marks:",
    "No marks available - call look first",
)

_OFFSCREEN_MARKER = "is off-screen (outside the current viewport)"

#: Refusal / policy-block markers that are NOT staleness: the model was told
#: no without anything being attempted. These are the exact virtual-tool
#: error prefixes (enumerated from every ``_clip_result("<tool> error"``
#: call site), so native daemon results carrying page text can never match.
_REFUSAL_MARKERS = (
    "tool call refused:",
    "REFUSED ",
    "refused:",
    "not replayed: the previous",
    "action blocked after",
    "NOT repeated; continue with",
    "upload_file refused:",
    "wait_for failed: provide a selector",
    "look failed:",
    "batch_probe error",
    "batch_probe failed",
    "click_mark error",
    "click_mark failed",
    "click_point error",
    "click_text error",
    "download failed",
    "drag_drop failed",
    "fill error",
    "fill failed",
    "fill_mark error",
    "fill_mark failed",
    "scroll failed",
    "select_option error",
    "select_option failed",
    "set_checked error",
    "set_checked failed",
    "upload_file error",
    "upload_file failed",
    "wait_for failed",
    "tool failed after retries:",
    "tool failed:",
    "tool blocked by policy:",
    "virtual tool ",
)


def _ba00_classify_result(text):
    """Classify one tool result for the BA-00 census.

    Returns ``(wasted, stale_kind)`` where ``stale_kind`` is ``"mut"``,
    ``"other"``, ``"offscreen"`` or ``""``. Never raises; unrecognized text
    is simply not wasted.
    """
    try:
        text = str(text or "")
    except Exception:
        return False, ""
    if not text:
        return False, ""
    if _MUT_STALE_MARKER in text:
        return True, "mut"
    if _OFFSCREEN_MARKER in text:
        return True, "offscreen"
    for marker in _OTHER_STALE_MARKERS:
        if marker in text:
            return True, "other"
    for marker in _REFUSAL_MARKERS:
        if marker in text:
            return True, ""
    return False, ""


def _ba00_note_result(stats, text):
    """Fold one tool result into the BA-00 census. Never raises."""
    if stats is None:
        return
    try:
        wasted, stale_kind = _ba00_classify_result(text)
        stats.record_outcome(wasted)
        if stale_kind == "mut":
            stats.stale_refusals += 1
        elif stale_kind == "other":
            stats.stale_refusals_other += 1
        elif stale_kind == "offscreen":
            stats.offscreen_refusals += 1
    except Exception:
        pass


# Daemon tools the model is allowed to see. Everything else the daemon
# exposes (read_file, list_dir, screenshot, ask_chat, copy_code_block,
# extract_code_blocks, ...) is hidden from the model: the file tools only
# ever return text (no image can reach the model that way, and the model
# improvised screenshot+read_file chasing one), and the internal tools are
# invoked by the virtual handlers themselves.
# BA-07: raw page JavaScript (evaluate) is NOT advertised anymore — showing
# its schema while forbidding it in prose was the worst of both worlds
# (schema tokens + prohibition tokens + a wasted turn when tried anyway).
# An unmentioned tool needs no warning, so the prohibition went with it.
_TOOL_ALLOWLIST = frozenset((
    "open_brave", "navigate", "list_tabs", "switch_tab", "new_tab",
))

# F17: the set the DISPATCH gate validates against. It is deliberately
# broader than the advertised list — advertising is a prompt-budget choice,
# dispatch is the authority boundary. A hidden but provably side-effect-free
# daemon helper (a file read, a directory listing) stays callable; what the
# gate must refuse is anything that mutates, navigates or executes without
# the grant for it. G6: the real-input F13/F40 daemon tools are dispatchable
# too — the virtual handlers call them on behalf of the model.
# BA-07: evaluate STAYS here so the internal handlers keep working through
# the internal-origin path — only the MODEL population lost it (see
# _MODEL_DISPATCH_ALLOWLIST below, which no longer includes it).
_TOOL_DISPATCH_ALLOWLIST = _TOOL_ALLOWLIST | frozenset((
    "evaluate",
    "read_file", "list_dir",
    "click_locator", "fill_locator", "scroll", "select_option", "set_checked",
    "upload_file", "download", "drag_drop",
))

#: F17: what a MODEL-authored call may dispatch — exactly what the model was
#: offered (the advertised daemon tools plus the virtual tools). The hidden
#: daemon primitives above stay available to our own handlers, but a name the
#: model was never shown must not become callable merely by naming it; that
#: was the "hidden primitives remain dispatchable" bypass.
_MODEL_DISPATCH_ALLOWLIST = frozenset(_TOOL_ALLOWLIST) | frozenset(_VIRTUAL_TOOL_NAMES)

#: F17: primitives only our own code may reach. Kept explicit so the
#: separation between model tools and internal primitives is reviewable.
_INTERNAL_ONLY_TOOLS = _TOOL_DISPATCH_ALLOWLIST - _MODEL_DISPATCH_ALLOWLIST

# F05 read vs mutation classification for the transport retry policy.
# READ tools are side-effect free: safe to auto-retry. MUTATION tools may
# commit a change before the transport fails: single attempt, never
# auto-replayed. Virtual tools (look/click/fill/...) bypass _run_one_tool's
# MCP path entirely and are NOT part of the tracker this round.
_READ_TOOLS = frozenset(("list_tabs",))
_MUTATION_TOOLS = frozenset((
    "open_brave", "navigate", "switch_tab", "new_tab", "evaluate",
    "click_locator", "fill_locator", "scroll", "select_option", "set_checked",
    "upload_file", "download", "drag_drop",
))
#: F05: the virtual (composite) mutations. They carry the same single-attempt,
#: outcome-unknown policy as the native ones above.
_VIRTUAL_MUTATION_TOOLS = frozenset((
    "click_mark", "click_point", "click_text", "fill", "fill_mark", "scroll",
    "select_option", "set_checked", "upload_file", "download", "drag_drop",
))
_ALL_MUTATION_TOOLS = _MUTATION_TOOLS | _VIRTUAL_MUTATION_TOOLS

# Conservative completion-phrase check for the budget-exhaustion UNLESS
# clause: after the step budget is spent the task is partial by default;
# only a no-failure run whose forced final text explicitly claims
# completion keeps the completed status (a bare model claim is NOT
# verification - default to partial).
_COMPLETION_PREFIXES = ("done", "completed", "task complete", "finished", "success")

# Audit LOW flag (G1 re-audit): a prefix match cancelled by an immediate
# negation ('success was not achieved') must NOT verify completion.
_COMPLETION_NEGATION_RE = re.compile(
    r"\b(not|n't|never|unable|failed|cannot|couldn't|didn't|wasn't|isn't|"
    r"aren't|weren't|hasn't|haven't|hadn't|won't|wouldn't|don't|doesn't|"
    r"can't|no success|not achieved)\b"
)


def _states_completion(text):
    """True when the forced final summary explicitly claims completion."""
    lowered = (text or "").strip().lower()
    if not lowered.startswith(_COMPLETION_PREFIXES):
        return False
    return not _COMPLETION_NEGATION_RE.search(lowered[:40])


#: RANK 4 — task descriptions whose goal is media PLAYBACK, which may only be
#: called done with a PLAYING verdict from the verify_playing media probe.
_MEDIA_GOAL_RE = re.compile(
    r"\b(play|plays|playing|playback|watch|watching)\b", re.IGNORECASE)


def _unproven_media_verdict(task_description, session):
    """RANK 4 — (hedge, evidence) when a media task ends without playback proof.

    Returns None when the goal is not media playback or when the task's own
    verify_playing observation proved PLAYING. Anything else - a static,
    paused, uncertain or never-taken probe - downgrades the completion to a
    partial result instead of a false "done".
    """
    if not _MEDIA_GOAL_RE.search(task_description or ""):
        return None
    verdict = str((session or {}).get("_media_verdict") or "").strip()
    if verdict.startswith("verify_playing: PLAYING"):
        return None
    if verdict:
        return ("I could not confirm it started playing (%s)." % verdict,
                verdict)
    return ("I pressed play, but I could not confirm it started - ask me to "
            "verify playback if you want me to check again.",
            "no playback observation was made")


# ── F08: suspended checkpoints ────────────────────────────────────────────
# A clarifying question SUSPENDS a run. The audit found the old continuation
# simply appended the answer to the description and started a NEW run, so
# committed actions were replayed. A checkpoint keeps the verified progress —
# the identity of the browser/tab the run had reached, the effects already
# committed and the pending question — and a resumed run restores it and is
# told explicitly not to repeat what already happened.
_SUSPENDED_LOCK = threading.Lock()
_SUSPENDED_CHECKPOINTS = {}
_SUSPENDED_TTL = 15 * 60.0
_SUSPENDED_LIMIT = 16


def _prune_suspended(now=None):
    now = now if now is not None else time.time()
    with _SUSPENDED_LOCK:
        for key in [key for key, entry in _SUSPENDED_CHECKPOINTS.items()
                    if now - entry.get("at", 0) > _SUSPENDED_TTL]:
            _SUSPENDED_CHECKPOINTS.pop(key, None)
        while len(_SUSPENDED_CHECKPOINTS) > _SUSPENDED_LIMIT:
            oldest = min(_SUSPENDED_CHECKPOINTS,
                         key=lambda k: _SUSPENDED_CHECKPOINTS[k].get("at", 0))
            _SUSPENDED_CHECKPOINTS.pop(oldest, None)


def new_checkpoint_id(task_description=""):
    """Mint a stable identity for one suspended run (F08)."""
    digest = hashlib.sha1(
        ("%s|%s|%s" % (task_description or "", os.getpid(),
                       time.monotonic())).encode("utf-8", "replace")).hexdigest()
    return "bp-%s" % digest[:16]


def suspend_checkpoint(checkpoint_id, session, question="", identity=None,
                       authorization=None):
    """Persist the verified progress of a suspended run (F08)."""
    if not checkpoint_id:
        return None
    identity = identity or {}
    entry = {
        "id": checkpoint_id,
        "question": question or "",
        "at": time.time(),
        # Verified progress: what this run already did, so a resume never
        # replays a committed effect.
        "completed": list((session or {}).get("completed_actions") or []),
        "completed_effects": sorted(
            str(item) for item in ((session or {}).get("completed_effects") or [])),
        "marks": dict((session or {}).get("marks") or {}),
        "tool_failures": list((session or {}).get("tool_failures") or []),
        "identity": {
            "url": identity.get("url") or (session or {}).get("url") or "",
            "tab": identity.get("tab") or (session or {}).get("tab") or "",
            "frame": identity.get("frame") or "",
            "session": identity.get("session") or "",
        },
        # The authority the user already granted for this run; a resumed run
        # keeps it, a CHANGED effect still needs fresh consent.
        "authorization": authorization,
    }
    with _SUSPENDED_LOCK:
        _SUSPENDED_CHECKPOINTS[checkpoint_id] = entry
    _prune_suspended()
    return entry


def peek_checkpoint(checkpoint_id):
    if not checkpoint_id:
        return None
    _prune_suspended()
    with _SUSPENDED_LOCK:
        entry = _SUSPENDED_CHECKPOINTS.get(checkpoint_id)
        return dict(entry) if entry else None


def resume_checkpoint(checkpoint_id):
    """Pop a suspended checkpoint so it can be resumed EXACTLY once (F08)."""
    if not checkpoint_id:
        return None
    _prune_suspended()
    with _SUSPENDED_LOCK:
        entry = _SUSPENDED_CHECKPOINTS.pop(checkpoint_id, None)
    return dict(entry) if entry else None


def drop_checkpoint(checkpoint_id):
    """Forget a suspended run (cancel/expiry): it can never restart later."""
    if not checkpoint_id:
        return False
    with _SUSPENDED_LOCK:
        return _SUSPENDED_CHECKPOINTS.pop(checkpoint_id, None) is not None


def _checkpoint_history(checkpoint, task_description):
    """Seed messages for a resumed run (F08): progress + don't-repeat list."""
    lines = [
        "RESUMING a suspended run. This is NOT a new task and its earlier "
        "verified progress still stands.",
    ]
    completed = list(checkpoint.get("completed") or [])
    if completed:
        lines.append("Already done — do NOT repeat these actions:")
        lines.extend("  - %s" % item for item in completed[-20:])
    identity = checkpoint.get("identity") or {}
    location = identity.get("url") or identity.get("tab")
    if location:
        lines.append("You were working here: %s" % location)
    question = checkpoint.get("question")
    if question:
        lines.append("You had asked the user: %s" % question)
    lines.append("User answer: %s" % task_description)
    return "\n".join(lines)


#: F08: a report is a CLARIFYING QUESTION only when it really asks the user
#: something. The old check was `"?" in text`, so any answer carrying a URL
#: with a query string ("https://x.com/?q=1") armed a bogus continuation.
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_QUESTION_START_RE = re.compile(
    r"^\s*(?:"
    r"which|what|where|who|whom|whose|when|why|how|"
    r"do you|did you|would you|should i|shall i|can you|could you|"
    r"are you|is it|are there|is there|any preference|please (?:choose|pick|"
    r"confirm|clarify|specify)|kindly (?:choose|confirm|specify)|"
    r"let me know|i need to know|to proceed"
    r")\b", re.IGNORECASE)


def _is_clarifying_question(text):
    """True when the run is ASKING THE USER something (F08).

    A question mark inside a URL/query string is not a question, and neither
    is a statement that merely ends with one.
    """
    if not text or not isinstance(text, str):
        return False
    stripped = _URL_RE.sub(" ", text).strip()
    if "?" not in stripped:
        return False
    # The interrogative must be the model ASKING, not quoting a page.
    for sentence in re.split(r"(?<=[?])\s+", stripped):
        if not sentence.rstrip().endswith("?"):
            continue
        if _QUESTION_START_RE.match(sentence.strip()):
            return True
    return False


def _agent_loop(client, task_description, started, stats=None, job=None,
                checkpoint=None, cached_tools=None, deadline=None):
    """Run the model/tool loop; returns the final TaskResult.

    The end-of-task STOPWATCH summary is emitted on EVERY exit path -
    success, failure, user stop and the 480s timeout - before the result
    is returned, so the activity log always explains where the time went.

    *checkpoint* (F08) is the suspended run being resumed: its verified
    progress is restored into the session and its committed actions are handed
    to the model as a do-not-repeat list.

    *cached_tools* (L-6) are raw tools/list defs from the pool cache: the
    inner loop advertises them without another wire round trip.

    *deadline* (BA-03) is the task's absolute budget, derived from the same
    *started* origin as the legacy timeout check so the two can never
    disagree. When omitted one is built here, so direct callers and tests
    keep working unchanged.
    """
    if stats is None:
        stats = _SwStats()
    if deadline is None:
        deadline = budget.Deadline.at(
            started + config.BROWSER_AGENT_TIMEOUT)
    # F09: the verified trace of what this run actually did (one entry per
    # executed step) — the raw material for capturing a PROCEDURE, not just a
    # goal sentence, when the run is verified complete.
    trace = []
    try:
        result = _agent_loop_inner(client, task_description, started, stats,
                                   job=job, checkpoint=checkpoint,
                                   trace=trace, cached_tools=cached_tools,
                                   deadline=deadline)
    finally:
        _emit_summary(stats, started)
    if result is not None:
        try:
            result.trace = list(trace)
        except Exception:
            pass
    return result


def _agent_loop_inner(client, task_description, started, stats, job=None,
                      checkpoint=None, trace=None, cached_tools=None,
                      deadline=None):
    if deadline is None:
        # Same origin as the task timeout: one budget, never two that
        # can disagree (BA-03 replaces the ad-hoc monotonic comparison).
        deadline = budget.Deadline.at(
            started + config.BROWSER_AGENT_TIMEOUT)
    if cached_tools is not None:
        # L-6: the pool already knows the daemon's tool list (code-static),
        # so the task starts without a tools/list round trip.
        tool_defs = list(cached_tools)
    else:
        try:
            tool_defs = client.list_tools()
        except Exception as exc:
            return _fail_and_log("tools/list failed: %s" % exc)
        _note_pooled_tools(client, tool_defs)
    # BA-14: the daemon's own tool names, captured BEFORE allowlisting — the
    # capability is about what the daemon HAS (event-driven `wait_for`?),
    # not what the model may SEE. Virtual handlers read session["daemon_tools"]
    # to prefer the event-driven path with a polling fallback on old daemons.
    try:
        daemon_tool_names = {t.get("name") for t in tool_defs
                             if isinstance(t, dict)}
    except Exception:
        daemon_tool_names = set()
    # Allowlist the daemon tools BEFORE the model ever sees them.
    tool_defs = [
        tool for tool in tool_defs
        if isinstance(tool, dict) and tool.get("name") in _TOOL_ALLOWLIST
    ]
    # Merge virtual tools (model sees them, MCP does not)
    try:
        tool_defs = list(tool_defs) + list(_VIRTUAL_TOOL_DEFS)
    except Exception:
        tool_defs = list(_VIRTUAL_TOOL_DEFS)
    session = {"marks": {}, "failures": FailureTracker(), "tool_failures": [],
               "job": job, "completed_actions": [], "url": "", "tab": "",
               "daemon_tools": daemon_tool_names,
               "trace": trace if trace is not None else []}
    history = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": task_description},
    ]
    if checkpoint:
        # F08: restore the suspended run's verified progress. The committed
        # actions are restored into the tracker as "already done" so an
        # identical mutation is not replayed, and the model is told plainly.
        session["checkpoint_id"] = checkpoint.get("id") or ""
        session["completed_actions"] = list(checkpoint.get("completed") or [])
        session["tool_failures"] = list(checkpoint.get("tool_failures") or [])
        try:
            session["marks"] = dict(checkpoint.get("marks") or {})
        except Exception:
            pass
        identity = checkpoint.get("identity") or {}
        session["url"] = identity.get("url") or ""
        session["tab"] = identity.get("tab") or ""
        # F08: the committed EFFECTS are restored as a do-not-repeat set, so a
        # resumed run cannot replay an action the suspended run already made.
        # (Effects the RESUMED run commits itself go to completed_effects only.)
        session["completed_effects"] = set(
            checkpoint.get("completed_effects") or [])
        session["resumed_effects"] = set(session["completed_effects"])
        history.append({
            "role": "user",
            "content": _checkpoint_history(checkpoint, task_description),
        })
    for step in range(config.BROWSER_AGENT_MAX_STEPS):
        # F20: per-job cancellation. The job token is honoured alongside the
        # legacy module flag; a pause blocks here instead of burning steps.
        if job is not None:
            try:
                job.checkpoint()
            except job_registry.Cancelled:
                append_activity_line("stopped by user\n")
                narrate_activity("Stopping")
                _publish_stop_report(session)
                return TaskResult.stopped()
        if _STOP_REQUESTED.is_set():
            # User pressed STOP: bail out gracefully - the browser stays
            # open, no tool error, the state resets via the caller.
            append_activity_line("stopped by user\n")
            narrate_activity("Stopping")
            _publish_stop_report(session)
            return TaskResult.stopped()
        if deadline.expired():
            return _fail_and_log(
                "task timed out after %ds" % config.BROWSER_AGENT_TIMEOUT
            )
        _trim_look_history(history)
        try:
            text, tool_calls = _model_turn_with_retries(
                history, tool_defs, step, stats)
        except Exception as exc:
            return _fail_and_log("model call failed: %s" % exc)
        if not tool_calls:
            # F03: the model stopped calling tools. Completed ONLY when no
            # tool failed during the task; any recorded tool failure makes
            # it partial (the evidence list carries the failures - the
            # summary itself is the model's text, byte-identical to before).
            failures = list(session.get("tool_failures") or [])
            # F08: a CLARIFYING QUESTION suspends the run with a checkpoint of
            # its verified progress, so the answer continues this same run
            # instead of starting a fresh one that replays committed actions.
            if _is_clarifying_question(text):
                checkpoint_id = new_checkpoint_id(task_description)
                suspend_checkpoint(
                    checkpoint_id, session, question=(text or "").strip(),
                    identity={"url": session.get("url") or "",
                              "tab": session.get("tab") or ""})
                append_activity_line("RESULT question: %s\n" % (text or ""))
                return TaskResult.needs_input(text or "",
                                              checkpoint=checkpoint_id)
            if failures:
                append_activity_line("RESULT partial: %s\n" % (text or ""))
                return TaskResult.partial(text or "", detail=text or "",
                                          evidence=failures)
            # RANK 4: a media-playback goal is only 'done' when the task's
            # own verify_playing probe proved PLAYING. Anything less becomes
            # a partial with the honest hedge instead of a false "done".
            unproven = _unproven_media_verdict(task_description, session)
            if unproven:
                hedge, proof_note = unproven
                summary = re.sub(r"\s+", " ", ("%s %s" % (text or "", hedge))
                                 ).strip()
                append_activity_line("RESULT partial: %s\n" % summary)
                return TaskResult.partial(summary, detail=text or "",
                                          evidence=[proof_note])
            append_activity_line("RESULT ok: %s\n" % (text or ""))
            return TaskResult.completed(text or "", detail=text or "")
        history.append(
            {"role": "assistant", "content": text, "tool_calls": tool_calls}
        )
        # BA-00: bracket one model step's dispatch so the census can tell a
        # turn that advanced nothing (every tool refused) from a mixed turn.
        if stats is not None:
            try:
                stats.begin_turn()
            except Exception:
                pass
        for call in tool_calls:
            # Cancellation is checked before EVERY tool call, not just at
            # the outer step boundary.
            if job is not None and job.should_stop():
                append_activity_line("stopped by user\n")
                narrate_activity("Stopping")
                _publish_stop_report(session)
                return TaskResult.stopped()
            if _STOP_REQUESTED.is_set():
                append_activity_line("stopped by user\n")
                narrate_activity("Stopping")
                _publish_stop_report(session)
                return TaskResult.stopped()
            # Tool failures are delivered as tool results and the loop
            # CONTINUES - only model-call failures abort the task. A
            # blocked repeat-failure (F05) ends the task as partial.
            blocked = _run_one_tool(client, history, call, session, stats)
            if blocked:
                evidence = list(session.get("tool_failures") or []) + [blocked]
                summary = ("I stopped the task after %s failed 3 times "
                           "in a row." % (call.get("name") or "the action"))
                append_activity_line("RESULT partial: %s\n" % summary)
                if stats is not None:
                    try:
                        stats.end_turn()
                    except Exception:
                        pass
                return TaskResult.partial(summary, detail=summary,
                                          evidence=evidence)
        if stats is not None:
            try:
                stats.end_turn()
            except Exception:
                pass
    # Budget exhausted: force ONE final summary turn with tools withheld
    # instead of returning a bare error - whatever the model achieved so
    # far is reported to the user, never flailed for more steps.
    history.append(
        {
            "role": "user",
            "content": (
                "Step budget exhausted - give your final summary now, "
                "no more tool calls."
            ),
        }
    )
    _trim_look_history(history)
    try:
        text, tool_calls = _model_turn_with_retries(
            history, [], config.BROWSER_AGENT_MAX_STEPS, stats)
    except Exception as exc:
        return _fail_and_log(
            "exceeded max steps (%d); final summary turn failed: %s"
            % (config.BROWSER_AGENT_MAX_STEPS, exc)
        )
    if text and not tool_calls:
        # F03 honesty rule: the step budget is spent, so this is partial
        # by default (a model claim without verification is NOT completion).
        # UNLESS the run had zero tool failures AND the forced final text
        # explicitly claims completion, the evidence carries the budget note.
        failures = list(session.get("tool_failures") or [])
        if not failures and _states_completion(text):
            append_activity_line("RESULT ok: %s\n" % text)
            return TaskResult.completed(text, detail=text)
        evidence = failures + ["step budget exhausted"]
        append_activity_line("RESULT partial: %s\n" % text)
        return TaskResult.partial(text, detail=text, evidence=evidence)
    return _fail_and_log("exceeded max steps (%d)" % config.BROWSER_AGENT_MAX_STEPS)


# ── L-6: pooled MCP session + tool-list cache ─────────────────────────────
# Verified 2026-10-03 against current code: every task built a fresh
# BraveMcpClient, paid initialize + notifications/initialized (2 RTTs) plus a
# tools/list round trip, then closed the session — while the daemon itself
# stays warm across tasks (ensure_brave_mcp_daemon is idempotent, the
# watcher boots it at startup) and the tool list is code-static (server.mjs
# defines tools via server.tool(); the client allowlist is a frozenset).
# So the pool keeps ONE connected session across sequential tasks and
# remembers the raw tool defs; the per-task ensure_daemon call STAYS (it is
# what restarts a dead daemon — the pool never skips it).
#
# Deliberate deviations from the audit sketch, all safety-driven:
# - No borrow-time health-check RTT. Reuse is optimistic; a restarted daemon
#   surfaces as a session-death signal that BraveMcpClient heals itself
#   (reconnect + exactly one replay of the proven-never-dispatched request).
# - Concurrent tasks never share a session (request ids would collide):
#   the pool hands out ONE client at a time (checkout flag); a second live
#   task takes the legacy private-client path (connect + close, as before).
# - Only REAL clients are pooled. Tests patch browser_agent.BraveMcpClient,
#   so the factory-identity check below routes every mock down the legacy
#   path and the existing suite behaves byte-identically to before.
# - The tool cache is dropped whenever the session is reborn (reconnects
#   generation check): a daemon upgrade may have changed the tool list.
# - Speculative navigation (navigating while the model drafts turn 0) is
#   OUT of scope: it changes first-turn task semantics and needs benchmark
#   proof first.
# Kill switches (default on): JARVIS_MCP_POOL=0 disables pooling,
# JARVIS_MCP_PREWARM=0 disables the confirmation-gap prewarm.
_TRUE_CLIENT_CLS = BraveMcpClient
_POOL_LOCK = threading.Lock()
_POOLED_CLIENT = None
_POOLED_KEY = None
_POOLED_LAST_USE = 0.0
_POOLED_IN_USE = False
_POOL_IDLE_TTL_S = 180.0
_CACHED_TOOL_DEFS = None  # (pool key, reconnects generation, [raw defs])


def _pool_enabled():
    return str(os.getenv("JARVIS_MCP_POOL", "1")).strip().lower() not in (
        "0", "false", "no", "off")


def _prewarm_enabled():
    return str(os.getenv("JARVIS_MCP_PREWARM", "1")).strip().lower() not in (
        "0", "false", "no", "off")


def _borrow_mcp_client(timeout):
    """Check out an MCP client: (client, cached_tools_or_None, pooled).

    Pooled clients are returned to the slot via _return_mcp_client (never
    closed); legacy clients (patched factory in tests, pool disabled, or a
    second concurrent task) are private and must be closed by the caller.
    NEVER raises pool bookkeeping errors — a broken pool degrades to a
    private client, never to a failed task.
    """
    factory = BraveMcpClient
    if not _pool_enabled() or factory is not _TRUE_CLIENT_CLS:
        client = factory(timeout=timeout)
        client.connect()
        return client, None, False
    global _POOLED_CLIENT, _POOLED_KEY, _POOLED_LAST_USE, _POOLED_IN_USE
    global _CACHED_TOOL_DEFS
    now = time.monotonic()
    stale = None
    with _POOL_LOCK:
        if _POOLED_CLIENT is not None and not _POOLED_IN_USE:
            if now - _POOLED_LAST_USE <= _POOL_IDLE_TTL_S:
                _POOLED_IN_USE = True
                client = _POOLED_CLIENT
                cached = _CACHED_TOOL_DEFS
                if (cached is None or cached[0] != _POOLED_KEY
                        or cached[1] != client.reconnects):
                    cached = None
                try:
                    client.timeout = timeout
                except Exception:
                    pass
                return client, (cached[2] if cached else None), True
            stale = _POOLED_CLIENT
            _POOLED_CLIENT = None
            _POOLED_KEY = None
            _CACHED_TOOL_DEFS = None
    if stale is not None:
        try:
            stale.close()
        except Exception:
            pass
    client = factory(timeout=timeout)
    client.connect()
    try:
        key = (client.base_url, client.token)
    except Exception:
        return client, None, False
    with _POOL_LOCK:
        if _POOLED_CLIENT is None:
            _POOLED_CLIENT = client
            _POOLED_KEY = key
            _POOLED_LAST_USE = time.monotonic()
            _POOLED_IN_USE = True
            return client, None, True
    return client, None, False


def _return_mcp_client(client, pooled):
    """Return a borrowed client: pooled ones go back to the slot, the rest
    are closed exactly as before. A client whose session died unrecoverably
    (_session_id None after a failed reconnect) is dropped, never re-pooled.
    """
    if client is None:
        return
    if not pooled:
        try:
            client.close()
        except Exception:
            pass
        return
    global _POOLED_CLIENT, _POOLED_KEY, _POOLED_IN_USE, _POOLED_LAST_USE
    global _CACHED_TOOL_DEFS
    drop = True
    with _POOL_LOCK:
        if client is _POOLED_CLIENT:
            try:
                alive = client._session_id is not None
            except Exception:
                alive = False
            if alive:
                _POOLED_IN_USE = False
                _POOLED_LAST_USE = time.monotonic()
                drop = False
            else:
                _POOLED_CLIENT = None
                _POOLED_KEY = None
                _CACHED_TOOL_DEFS = None
    if drop:
        try:
            client.close()
        except Exception:
            pass


def _note_pooled_tools(client, raw_defs):
    """Remember a fresh tools/list for the pooled client (L-6 cache fill)."""
    global _CACHED_TOOL_DEFS
    try:
        with _POOL_LOCK:
            if (client is not None and client is _POOLED_CLIENT
                    and _POOLED_KEY is not None):
                _CACHED_TOOL_DEFS = (
                    _POOLED_KEY, client.reconnects, list(raw_defs))
    except Exception:
        pass


def reset_mcp_pool():
    """Close and forget the pooled session (tests; never used by tasks)."""
    global _POOLED_CLIENT, _POOLED_KEY, _POOLED_LAST_USE, _POOLED_IN_USE
    global _CACHED_TOOL_DEFS
    with _POOL_LOCK:
        pooled, _POOLED_CLIENT = _POOLED_CLIENT, None
        _POOLED_KEY = None
        _CACHED_TOOL_DEFS = None
        _POOLED_IN_USE = False
        _POOLED_LAST_USE = 0.0
    if pooled is not None:
        try:
            pooled.close()
        except Exception:
            pass


def prewarm_mcp_pool(timeout=None):
    """Warm the pooled session + tool cache on the CALLER's thread (L-6).

    Meant to run on a daemon thread while the user reads the confirmation
    prompt, so connect + tools/list happen inside the approval gap instead
    of after "yes". Best effort and side-effect free: no navigation, no task
    state, never spawns the daemon (a down daemon just fails fast here and
    the real task path ensures + connects as before). NEVER raises.
    """
    try:
        if not _prewarm_enabled():
            return
        tmo = (timeout if timeout is not None
               else config.BROWSER_AGENT_TOOL_TIMEOUT)
        with _POOL_LOCK:
            if (_POOLED_CLIENT is not None and time.monotonic()
                    - _POOLED_LAST_USE <= _POOL_IDLE_TTL_S):
                return  # already warm (or held by a live task — warm enough)
        client, cached, pooled = _borrow_mcp_client(tmo)
        try:
            if cached is None:
                _note_pooled_tools(client, client.list_tools())
        finally:
            _return_mcp_client(client, pooled)
    except Exception:
        pass


def run_browser_task(task_description, job=None, resume_from=None):
    """Run one browser-automation task end to end. NEVER raises: returns a
    TaskResult (str-compatible: str() is the model's final summary, or
    'TASK NOT COMPLETED. Error: <msg>').

    *job* is an optional jobs.JobToken; when omitted a fresh one is created so
    this run can be cancelled on its own without disturbing another job.

    *resume_from* is a suspended-checkpoint id (F08): instead of starting a
    fresh history, the run restores the verified progress of the suspended
    run — the browser/tab identity it had reached and the effects it already
    committed — so answering a clarifying question continues the SAME task
    instead of replaying committed actions.
    """
    started = time.monotonic()
    stats = _SwStats()
    # BA-03: the ONE absolute budget for this task, from the same `started`
    # origin the timeout message reports. Bound to this thread so every
    # model POST and MCP call below slices its own timeout to what is
    # actually left (see core/deadline); passed explicitly to the loop so
    # the step-top check and the wire timeouts can never disagree.
    task_deadline = budget.Deadline.at(
        started + config.BROWSER_AGENT_TIMEOUT)
    checkpoint = resume_checkpoint(resume_from) if resume_from else None
    # F20: the legacy flag stays armed while ANOTHER live browser job is
    # already cancelled — that is what stops run B from disarming the stop the
    # user just issued for run A. A merely-live job (no cancellation pending)
    # must not keep it armed, or one slow run would swallow a later stop.
    if not any(j.should_stop() for j in job_registry.live_jobs(kind="browser")):
        _STOP_REQUESTED.clear()
    owns_job = job is None
    client = None
    pooled = False
    cached_tools = None
    try:
        try:
            if not ensure_brave_mcp_daemon():
                return TaskResult.failed("brave MCP daemon could not be started")
            # L-6: sequential tasks reuse one pooled session + tool cache
            # (initialize/handshake/list_tools paid once); the ensure above
            # still runs every task so a dead daemon is restarted first.
            client, cached_tools, pooled = _borrow_mcp_client(
                timeout=config.BROWSER_AGENT_TOOL_TIMEOUT)
        except Exception as exc:
            return TaskResult.failed(str(exc))
        if owns_job:
            # Registered only once the run is actually under way: a task that
            # never gets a daemon must not leave a cancellable job behind.
            # BA-03: the job carries no second budget of its own — the
            # task_deadline above is the single authority; the loop checks
            # it and every wire call slices to it.
            job = job_registry.new_job(
                kind="browser", label=(task_description or "")[:80])
            try:
                _OWNED_JOB_IDS.add(job.job_id)
            except Exception:
                pass
        truncate_activity_log()
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        append_activity_line("\n=== %s === %s ===\n" % (stamp, task_description))
        with budget.bound(task_deadline):
            return _agent_loop(client, task_description, started, stats,
                               job=job, checkpoint=checkpoint,
                               cached_tools=cached_tools,
                               deadline=task_deadline)
    finally:
        # L-6: pooled clients go back to the slot; private ones are closed
        # exactly as before.
        _return_mcp_client(client, pooled)
        if owns_job and job is not None:
            job.finish()
            try:
                _OWNED_JOB_IDS.discard(job.job_id)
            except Exception:
                pass
