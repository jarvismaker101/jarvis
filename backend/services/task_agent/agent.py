"""Connector-first task execution brain for Jarvis.

This layer is deliberately separate from chat and legacy command parsing. It
tries to reason from structured app state first: editor bridge, browser CDP,
and Windows UI Automation. Screenshot/OCR screen control remains a fallback.
"""

import json
import logging
import getpass
import os
import re
import threading
import time

from backend.services.fireworks_client import ask_fireworks
from backend.services.task_agent.connectors import browser_cdp, editor_bridge, windows_connector
from backend.services import approvals
from backend.services import code_tools
from backend.services import productivity_connector
from backend.services import tool_policy
from backend.services.context_envelope import budgeted_json
from backend.services.task_result import TaskResult


TASK_PREFIX_RE = re.compile(
    r"^\s*(?:jarvis\s+)?(?:"
    r"execute task|do task|use task brain|task brain|computer task|agent mode|agent|task|take over"
    r")\b[\s:,-]*",
    re.IGNORECASE,
)

ACTION_TERMS = {
    "automate",
    "click",
    "configure",
    "create",
    "debug",
    "execute",
    "find",
    "fix",
    "install",
    "login",
    "logout",
    "open",
    "run",
    "search",
    "setup",
    "change",
    "update",
}

CONNECTOR_HINTS = {
    "antigravity",
    "browser",
    "chrome",
    "brave",
    "edge",
    "editor",
    "site",
    "website",
    "webpage",
    "workspace",
    "vs code",
    "vscode",
    "cursor",
    "windsurf",
}

#: F01 — how many steps one plan may execute. This is a real budget: the plan
#: records the steps it INTENDED and reports every omitted one as an unmet
#: goal, so a cut-short plan can never be published as completed work.
TASK_MAX_STEPS = max(1, int(os.getenv("JARVIS_TASK_MAX_STEPS", "24") or 24))

SAFE_TOOLS = {
    "browser.search_web",
    "browser.open_url",
    "browser.inspect_tabs",
    "editor.inspect_workspace",
    "windows.inspect_active_window",
    "windows.screen_action",
    # Native code-agent tools (file/command/script) — Jarvis's own hands.
    "code.read_file",
    "code.write_file",
    "code.list_directory",
    "code.run_command",
    "code.run_script",
    # F11 (G4): read-only inspection tools are safe.
    "code.search",
    "code.read_range",
    "code.inspect_diff",
}

# F14: the connector's read/draft operations are least privilege — the
# connector itself enforces the delegated scope for every one of them, so they
# do not additionally need the whole-plan confirmation. Send/create operations
# are deliberately NOT here: they always need the confirmation AND their own
# per-effect approval (see _PRODUCTIVITY_EFFECT_TOOLS).
SAFE_TOOLS |= {
    name for name, spec in productivity_connector.OPERATIONS.items()
    if spec["operation"] not in productivity_connector.EXTERNAL_EFFECTS
}

#: F14 — connector operations that change the world outside this machine. A
#: plan step using one never carries its own authority: it must present the
#: approval id of a separate per-draft approval (request_effect_approval) that
#: the connector re-verifies immediately before the send/create.
_PRODUCTIVITY_EFFECT_TOOLS = {
    name: spec["effect"]
    for name, spec in productivity_connector.OPERATIONS.items()
    if spec["operation"] in productivity_connector.EXTERNAL_EFFECTS
}

# Tools whose args should be surfaced verbatim in a confirmation preview so the
# user can see exactly what would run (esp. shell commands and writes).
CONFIRM_SENSITIVE_PREVIEW = {"code.run_command", "code.write_file", "code.run_script", "code.create_folder", "code.apply_patch",
                             # F15 (G4): an editor WorkspaceEdit is a real
                             # edit — show which files it touches.
                             "editor.apply_workspace_edit",
                             # F14: a send/create shows the connector's own
                             # scrubbed preview of the draft's content.
                             "mail.send_draft", "calendar.commit_event"}

# Code tools that always require a spoken confirmation before execution.
CONFIRM_TOOLS = {
    "code.write_file",
    "code.run_command",
    "code.run_script",
    "code.create_folder",
    "code.apply_patch",
    "code.run_checks",
    "editor.edit_active_selection",
    "editor.apply_workspace_edit",
    "editor.test_results",
    # F14: an externally visible send/create is never implicit.
    "mail.send_draft",
    "calendar.commit_event",
}

# Pending task-action confirmation gate (mirrors the brain's research gate).
_pending_task_action = None
_task_confirm_lock = threading.Lock()
TASK_CONFIRM_WINDOW_SECONDS = 45.0

_TASK_CONFIRM_YES_RE = re.compile(
    r"\b(yes|yeah|yep|yup|sure|okay|ok|alright|go ahead|do it|please do|"
    r"confirm|proceed|haan|ha|kar do|karo|kar de|continue)\b",
    flags=re.IGNORECASE,
)
_TASK_CONFIRM_NO_RE = re.compile(
    r"\b(no|nah|nope|nahi|nahin|nai|never|skip|cancel|leave it|"
    r"don'?t bother|mat karo|mat kar|"
    r"(?:don'?t|do not)|stop|abort|"
    r"not sure|not ready|not yet|well no|hmm no)\b",
    flags=re.IGNORECASE,
)

#: R4 — whole-sentence yes check. A bare "yes"/"ok"/"kar do" approves the
#: exact preview; a QUALIFIED yes changes the effect and needs a new preview:
#: "yes, but call it X" (rename), "yes, don't create it" (negates the
#: effect), "yes, a quick look" (inspection tail vs creation preview —
#: ambiguous, hold the write and ask once). Extra actions ("yes, and also
#: ...") never inherit the old yes. Courtesy tails ("please", "thanks")
#: are harmless and stay approvals.
_TAIL_RENAME_RE = re.compile(
    r"\b(?:but|instead|rather)\b.{0,40}?\b(?:call|name|rename|use)\b"
    r"|\bcall\s+it\b|\bname\s+it\b",
    re.IGNORECASE,
)
_TAIL_INSPECT_RE = re.compile(
    r"\bquick\s+look\b|\bjust\s+(?:look|check|checking|see|seeing)\b"
    r"|\bcheck\s+(?:it|that|this|the\s+folder)\s+first\b"
    r"|\bonly\s+(?:check|look|see)\b",
    re.IGNORECASE,
)
_TAIL_EXTRA_ACTION_RE = re.compile(
    r"\b(?:and\s+also|also\s+(?:create|make|delete|remove|move|send)|"
    r"then\s+(?:delete|remove|send|create|make))\b",
    re.IGNORECASE,
)
_TAIL_NEGATE_EFFECT_RE = re.compile(
    r"\b(?:don'?t|do not|never)\s+(?:create|make|write|delete|remove|"
    r"send|run|execute|do)\b|\bnot\s+(?:that|the)\s+(?:folder|file)\b",
    re.IGNORECASE,
)
_HARMLESS_TAIL_RE = re.compile(
    r"^(?:please|thanks?|thank\s+you|sir|ji|ok)[\s,!.]*$", re.IGNORECASE)


def classify_confirmation(answer):
    """R4: the whole-sentence verdict on a pending-preview answer.

    Returns one of:
    - "yes": clean assent to the unchanged effect ("yes", "yes please").
    - "no": decline, including effect-negating tails ("yes, don't create").
    - "rename:<name>": assent with a changed name ("yes, but call it X").
    - "inspect": assent with an inspection tail vs a write preview
      ("yes, a quick look") — ambiguous, hold the write, ask once.
    - "extra": assent plus an extra action — the extra never inherits.
    - "unclear": no assent at all ("did I say yes?", "not yet").

    The caller decides the reply; NOTHING here executes or discards state.
    """
    text = (answer or "").strip()
    if not text:
        return "unclear"
    lowered = text.lower()
    # R4: a QUESTION about saying yes is a question, never assent.
    if "?" in text and re.search(r"\bdid\s+i\s+say\b", lowered):
        return "unclear"
    has_yes = bool(_TASK_CONFIRM_YES_RE.search(text))
    # Negation of the EFFECT ("don't create it") is a decline even when a
    # "yes" word rides along — NO-first, whole sentence.
    if _TAIL_NEGATE_EFFECT_RE.search(text):
        return "no"
    if _TASK_CONFIRM_NO_RE.search(text):
        return "no"
    if not has_yes:
        return "unclear"
    # A yes-word is present and nothing declined: check the tail.
    if _TAIL_RENAME_RE.search(text):
        name = re.search(
            r"(?:call|name)\s+it\s+([A-Za-z0-9][\w\- ]{0,60}?)(?:\s+please)?[\s.,!]*$",
            text, re.IGNORECASE)
        rename = (name.group(1).strip().rstrip(".,! ") if name else "")
        return "rename:%s" % rename if rename else "rename:"
    if _TAIL_INSPECT_RE.search(text):
        return "inspect"
    if _TAIL_EXTRA_ACTION_RE.search(text):
        return "extra"
    return "yes"


def _normalize(text):
    return " ".join((text or "").strip().lower().split())


def is_task_request(text):
    normalized = _normalize(text)
    if not normalized:
        return False
    if TASK_PREFIX_RE.match(text or ""):
        return True

    has_connector_hint = any(hint in normalized for hint in CONNECTOR_HINTS)
    has_action = any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in ACTION_TERMS)
    return has_connector_hint and has_action


def is_explicit_task_request(text):
    """True only when the user explicitly invoked task mode via a prefix
    ("use task brain ...", "execute task ...", "take over ..."). Plain
    natural commands like "open youtube in chrome" are NOT considered task
    requests here — they belong to the tool-intent router instead.
    """
    return bool(TASK_PREFIX_RE.match(text or ""))


# Words that indicate a web/browser target rather than a local file/command.
_CODE_TOOL_WEB_HINTS = (
    "browser", "website", "webpage", ".com", ".org",
    "youtube", "gmail", "chrome", "edge", "brave",
)

# URL-shaped targets (word-boundary http, or an http(s):// prefix) belong to
# the web — unless a run-family command token leads ("run curl http://...").
_CODE_TOOL_URL_RE = re.compile(r"\bhttp\b|https?://", re.IGNORECASE)

# Known command tokens that can start a shell command after the run verb.
_CODE_TOOL_COMMAND_TOKENS = (
    "pip", "python", "py", "npm", "node", "npx", "git", "docker",
    "dir", "echo", "cd", "ls", "where", "tasklist", "ping", "curl",
    "java", "mvn", "gradle", "python3",
)

# Leading conversational fillers a spoken request may carry before the verb
# ("now create ...", "please just make ..."). Routing strips these before
# the anchored verb matches so the filler can never push a real action into
# chat (live bug: "now create a text file ..." fell through to chat and the
# model promised work no tool ever did).
_CODE_TOOL_FILLER_RE = re.compile(
    r"^(?:(?:now|ok|okay|so|then|please|just|actually|well|hey|hi|hello|"
    r"jarvis|sir)\b[\s,.-]*)+",
    re.IGNORECASE,
)

# "text file" / "txt file" / "notepad file" spoken adjectives: the noun is
# still "file", so routing/planning must see through them.
_CODE_TOOL_FILE_ADJECTIVES = (
    "text", "txt", "notepad", "new", "empty", "blank", "simple", "small",
)

# Targets starting with these pronouns are conversational, not commands.
# "that" is deliberately NOT here: "create a file inside that folder" is a
# write with a location, not small talk (see _CODE_TOOL_LOCATION_RE).
_CODE_TOOL_PRONOUN_STARTS = ("me", "us", "him", "her", "them", "it")

# Location phrases that carry the write destination when no file name is
# named ("create a text file inside that folder ...").
_CODE_TOOL_LOCATION_RE = re.compile(
    r"\b(?:inside|into|in|within|under)\s+"
    r"(?:(?:the|this|that|my|our)\s+)?"
    r"(folder|directory)(?:\s+(?:named|called)\s+([a-z0-9_ .\-]+))?",
    re.IGNORECASE,
)

# A located write naming the folder BARE ("create a file in mayankmalik and
# write hello") — no "folder" word at all. Requires the "and <verb>" shape
# so content phrases ("write hello in english") never match.
_CODE_TOOL_BARE_FOLDER_RE = re.compile(
    r"\bin\s+([a-z0-9_ .\-]+?)\s+and\s+"
    r"(?:inside(?:\s+(?:that|this|the))?(?:\s+(?:text\s+)?file)?\s+)?"
    r"(?:just\s+)?(?:write|containing|with|as|saying|say|of)\b",
    re.IGNORECASE,
)

# Path-like target: name with a dot-extension at the end.
_CODE_TOOL_PATH_RE = re.compile(
    r"^[a-z0-9_ .\-]+\.(?:[a-z0-9]{1,8})$",
    re.IGNORECASE,
)


def _is_path_like(target):
    if not target:
        return False
    if "\\" in target or "/" in target:
        return True
    return bool(_CODE_TOOL_PATH_RE.match(target))


def is_code_tool_request(text):
    """True when *text* looks like a direct file/command/script operation that
    the native code tools can handle deterministically (no LLM needed):
    read a file, create/write a file, inspect a local folder, or run a
    command / script.

    Deliberately conservative: targets must look path-like or start with a
    known command token, and anything that smells like a web/browser target
    is excluded so those keep flowing through the normal tool/executor path.

    R5: local folder inspection ("check whether folder X exists", "have a
    quick look at that folder", "see what is inside") is a LOCAL read via
    code.list_directory — never a browser job. "Look"/"navigate" alone never
    implies the browser; only a web URL / web target does.
    """
    if not text or not text.strip():
        return False
    normalized = _normalize(text)
    if any(hint in normalized for hint in _CODE_TOOL_WEB_HINTS):
        return False
    # Spoken fillers ride in front of the verb ("now create ..."); strip
    # them so the anchored verb matches still apply.
    routed = _CODE_TOOL_FILLER_RE.sub("", normalized).strip() or normalized

    # R5: local inspect/existence shape — checked FIRST, before the web
    # "look up" branch in _heuristic_plan can claim it. A bare local folder
    # mention with an inspect verb is a list_directory read, never browser.
    if _local_inspect_folder(routed) is not None:
        return True

    read_match = re.match(r"^(?:read|cat|open file)\s+(.+)$", routed)
    show_match = re.match(r"^show\s+(.+)$", routed)
    write_match = re.match(r"^(?:create|make|write|save|update)\b.*?\bfile\b\s*(.*)$", routed)
    folder_match = re.match(
        r"^(?:create|make)\s+(?:a\s+|an\s+)?(?:folder|directory)\b(?:\s+(.+))?$",
        routed,
    )
    multi_match = re.match(
        r"^(?:create|make)\s+(?:(?:\d+|(?:one|two|three|four|five|six|seven|eight|nine|ten))\s+)?files?\b(?:\s*[:,-]?\s*(.+))?$",
        routed,
    )
    run_match = re.match(r"^(?:run|execute)\s+(.+)$", routed)

    if _CODE_TOOL_URL_RE.search(routed) and not run_match:
        return False

    if read_match:
        return _is_path_like(read_match.group(1))

    if show_match and _is_path_like(show_match.group(1)):
        return True

    if write_match:
        tail = write_match.group(1)
        if re.search(r"\b(?:with|containing|as)\b", tail):
            return True
        tail_target = re.sub(r"^(?:called|named|the|a|an)\s+", "", tail.strip())
        if _is_path_like(tail_target):
            return True
        # A located write names no file ("create a text file inside that
        # folder and write hello inside it"): the location + the content
        # are the intent, so it routes; planning resolves the folder and
        # asks for (or defaults) the name.
        if _located_write_folder(routed) is not None:
            return True
        if _CODE_TOOL_LOCATION_RE.search(tail):
            return True
        if re.search(r"\b(?:inside|into|within)\b.*\b(?:write|containing|content|text)\b", tail):
            return True
        return False

    # "create/make a folder named X", "create N files: a.txt, b.py, ..." —
    # route into the gated task-agent path. The folder tail must carry a
    # naming/location cue ("named badmoss", "name it demo", "in temp") so a
    # bare noun phrase like "create a directory listing" stays conversational.
    if folder_match and folder_match.group(1) and _FOLDER_TAIL_HINT_RE.search(folder_match.group(1)):
        return True
    if multi_match and multi_match.group(1):
        return True

    if run_match:
        target = run_match.group(1).strip()
        first = target.split(" ", 1)[0].strip().lower()
        if first in _CODE_TOOL_PRONOUN_STARTS:
            return False
        if _is_path_like(target):
            return True
        return first in _CODE_TOOL_COMMAND_TOKENS

    return False



def _strip_task_prefix(text):
    return TASK_PREFIX_RE.sub("", text or "").strip() or (text or "").strip()


def _extract_json_object(content):
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


def _active_window_title(context):
    return (
        context.get("windows", {})
        .get("active_window", {})
        .get("title", "")
    )


def gather_context():
    windows = windows_connector.snapshot()
    title = windows.get("active_window", {}).get("title", "")
    return {
        "windows": windows,
        "editor": editor_bridge.snapshot(active_window_title=title),
        "browser": browser_cdp.snapshot(),
    }


def _summarize_context(context):
    title = _active_window_title(context) or "Unknown window"
    editor = context.get("editor", {})
    browser = context.get("browser", {})
    controls = context.get("windows", {}).get("visible_controls", [])

    lines = [f"Active window: {title}."]
    if editor.get("active_window_looks_like_editor"):
        if editor.get("available"):
            state = editor.get("state", {})
            active_file = (state.get("activeFile") or {}).get("path", "")
            workspace = state.get("workspaceFolders") or []
            if active_file:
                lines.append(f"Editor bridge is connected. Active file: {active_file}.")
            if workspace:
                lines.append(f"Workspace folders: {', '.join(workspace[:3])}.")
        else:
            lines.append("This looks like an editor, but the Jarvis editor bridge is not connected yet.")
    if browser.get("available"):
        tabs = browser.get("tabs", [])
        lines.append(f"Browser CDP is connected with {len(tabs)} visible tab(s).")
    else:
        lines.append("Browser DOM control is not connected; browser actions can still open/search pages.")
    if controls:
        labels = [
            (item.get("name") or item.get("control_type") or "").strip()
            for item in controls[:6]
        ]
        labels = [label for label in labels if label]
        if labels:
            lines.append("Visible controls include: " + ", ".join(labels) + ".")
    return " ".join(lines)


def _heuristic_plan(command, context):
    # Raw utterance (case, newlines, quoted content intact) plus a separate
    # normalized copy for case-insensitive verb detection. Every payload
    # extraction below runs on the RAW spans so paths, flags, URLs and file
    # content survive the planner/tool boundary unchanged.
    raw = (command or "").strip()
    normalized = _normalize(command)

    if any(phrase in normalized for phrase in ("what can you do", "what is possible", "understand this window", "where am i")):
        return {
            "ok": True,
            "confidence": 0.95,
            "summary": "Inspecting the active workspace.",
            "requires_confirmation": False,
            "steps": [{"tool": "windows.inspect_active_window", "args": {}, "risk": "safe"}],
            "response": _summarize_context(context),
        }

    search_match = re.match(r"^(?:search|google|find|look up)\s+(.+)$", raw, re.IGNORECASE | re.DOTALL)
    if search_match and " in " not in normalized:
        # R5: a LOCAL folder target keeps the turn local even when the verb
        # looks web-ish ("find malik folder", "look up what's in there").
        # Tool choice comes from the target, never from the verb alone.
        if _local_inspect_folder(_CODE_TOOL_FILLER_RE.sub("", normalized).strip() or normalized) is None:
            return {
                "ok": True,
                "confidence": 0.9,
                "summary": "Opening a web search.",
                "requires_confirmation": False,
                "steps": [
                    {
                        "tool": "browser.search_web",
                        "args": {"query": search_match.group(1).strip()},
                        "risk": "safe",
                    }
                ],
            }

    open_match = re.match(r"^(?:open|go to|navigate to)\s+(.+)$", raw, re.IGNORECASE | re.DOTALL)
    if open_match and "navigate to" not in normalized:
        # R5 repair: bare "navigate to <folder>" is NOT a browser open. The
        # old rule only fired on URL tokens, so this branch keeps that —
        # and a non-URL navigate target falls through to the local-inspect
        # shape below instead of becoming a vague browser plan.
        if any(token in open_match.group(1).lower() for token in (".com", ".org", ".net", "http")):
            return {
                "ok": True,
                "confidence": 0.9,
                "summary": "Opening the requested URL.",
                "requires_confirmation": False,
                "steps": [
                    {
                        "tool": "browser.open_url",
                        "args": {"url": open_match.group(1).strip()},
                        "risk": "safe",
                    }
                ],
            }

    # ── R5 local inspect/existence (read-only; no approval needed) ──
    # "check whether folder Malik exists", "have a quick look at that
    # folder", "see what is inside" -> one code.list_directory read. This
    # branch runs BEFORE any model plan so a local folder can never become
    # a "Navigate to ..." browser job. "Look"/"navigate" never imply the
    # browser — the local target picks the local tool.
    inspect_hint = _local_inspect_folder(
        _CODE_TOOL_FILLER_RE.sub("", normalized).strip() or normalized)
    if inspect_hint is not None:
        resolved = _resolve_folder_hint(inspect_hint)
        if resolved is None:
            return {
                "ok": True,
                "confidence": 0.9,
                "summary": "Need the folder.",
                "requires_confirmation": False,
                "steps": [],
                "response": _folder_hint_clarification(inspect_hint),
            }
        return _code_tool_plan(
            "code.list_directory", {"path": resolved},
            "Checking the folder.",
        )

    # ── Native code-tools heuristics (short-circuit; no LLM needed) ──
    # "read <file>", "show me <file>", "what's in <file>" -> read_file
    read_match = re.match(r"^(?:read|show|display|open file|cat)\s+(.+)$", raw, re.IGNORECASE | re.DOTALL)
    if read_match and not any(t in normalized for t in ("browser", "website", "http", ".com")):
        return _code_tool_plan("code.read_file", {"path": read_match.group(1).strip()}, "Reading the file.")

    # "create/write/save <file> containing/with/to ..." -> write_file
    split = _split_write_request(_strip_code_tool_filler(raw))
    if split is not None:
        path, content_after = split
        folder_hint = _located_write_folder(raw)
        if folder_hint is not None and not _is_path_like(path):
            resolved = _resolve_folder_hint(folder_hint)
            if resolved is None:
                return {
                    "ok": True,
                    "confidence": 0.9,
                    "summary": "Need the folder.",
                    "requires_confirmation": False,
                    "steps": [],
                    "response": _folder_hint_clarification(folder_hint),
                }
            name = _located_write_name(raw, content_after)
            content = _located_write_content(raw, content_after)
            return _code_tool_plan(
                "code.write_file",
                {"path": os.path.join(resolved, name), "content": content},
                "Writing the file.",
            )
        return _code_tool_plan(
            "code.write_file",
            {"path": path, "content": content_after},
            "Writing the file.",
        )

    # "create/make a folder/directory named X ..." and "create N files: ..."
    # -> code.create_folder + one code.write_file per named file. One plan,
    # one confirmation gate covering the whole thing.
    folder_match = re.match(
        r"^(?:create|make)\s+(?:a\s+|an\s+)?(?:folder|directory)\b", raw, re.IGNORECASE
    )
    multi_match = re.match(
        r"^(?:create|make)\s+(?:(?:\d+|(?:one|two|three|four|five|six|seven|eight|nine|ten))\s+)?files?\b",
        raw,
        re.IGNORECASE,
    )
    if folder_match or multi_match:
        folder_path = _folder_plan_path(raw)
        file_names = _multi_file_names(raw)
        if folder_path or file_names:
            steps = []
            if folder_path:
                steps.append({
                    "tool": "code.create_folder",
                    "args": {"path": folder_path},
                    "risk": "safe",
                    "reason": "Creating the folder.",
                })
            for name in file_names:
                path = os.path.join(folder_path, name) if folder_path else name
                steps.append({
                    "tool": "code.write_file",
                    "args": {"path": path, "content": ""},
                    "risk": "safe",
                    "reason": f"Creating {name}.",
                })
            if folder_path and file_names:
                summary = f"Creating folder {folder_path} with {len(file_names)} files."
            elif folder_path:
                summary = f"Creating folder {folder_path}."
            else:
                summary = f"Creating {len(file_names)} files."
            return {
                "ok": True,
                "confidence": 0.9,
                "summary": summary,
                "requires_confirmation": True,
                "steps": steps,
            }

    # "run/execute <command> or <script>"
    run_match = re.match(r"^(?:run|execute)\s+(.+)$", raw, re.IGNORECASE | re.DOTALL)
    if run_match:
        target = run_match.group(1).strip()
        if re.search(r"\.(py|bat|cmd)$", target, re.IGNORECASE):
            return _code_tool_plan("code.run_script", {"path": target}, "Running the script.")
        return _code_tool_plan(
            "code.run_command", {"command": target}, "Running that command.",
        )

    return None


def _code_tool_plan(tool, args, summary, risk="safe"):
    """Build a single-step plan invoking a native code tool."""
    return {
        "ok": True,
        "confidence": 0.9,
        "summary": summary,
        "requires_confirmation": tool in CONFIRM_TOOLS,
        "steps": [
            {
                "tool": tool,
                "args": args,
                "risk": risk,
                "reason": summary,
            }
        ],
    }


#: F12 — the words that introduce file content in a write request. They are
#: only delimiters when what precedes them is a COMPLETE file name; a file
#: called "Q4 Report With Care.TXT" must not be truncated at its own "With".
#: A file-type adjective ("text", "txt", "notepad") may sit between "a" and
#: "file" ("create a text file X with Y").
_WRITE_REQUEST_RE = re.compile(
    r"^(?:create|make|write|save|update)\s+(?:a\s+)?"
    r"(?:(?:text|txt|notepad|new|empty|blank|simple|small)\s+)?file\s+"
    r"(?:called\s+|named\s+)?",
    re.IGNORECASE,
)
_WRITE_DELIM_RE = re.compile(r"\s+(?:with|containing|as)\s+", re.IGNORECASE)
_FILE_EXT_TAIL_RE = re.compile(r"\.[A-Za-z0-9]{1,8}$")


def _looks_like_complete_file_name(name):
    """True when *name* can stand alone as a file name (F12).

    Either it ends in an extension, carries a path separator/drive, or is a
    single bare token. Deliberately conservative: anything else is treated as
    "file name + more words", so a real delimiter is still found.
    """
    candidate = str(name or "").strip().strip("\"'")
    if not candidate or re.search(r"[\r\n]", candidate):
        return False
    if _FILE_EXT_TAIL_RE.search(candidate):
        return True
    if "\\" in candidate or "/" in candidate:
        return True
    return not re.search(r"\s", candidate)


def _split_write_request(raw):
    """(path, content) out of 'create file X with Y', or None (F12).

    F12: content delimiters must not match inside a file name. Only a split
    whose left-hand side is a complete file name counts (the LAST such split,
    so "a.txt with b.txt as the name" keeps "a.txt" + "b.txt as the name"),
    and a tail that is itself one complete file name carries no content at
    all. Falls back to the first delimiter when neither test is decisive, so
    the historical phrasing keeps working.
    """
    match = _WRITE_REQUEST_RE.match(raw or "")
    if not match:
        return None
    tail = (raw or "")[match.end():].strip()
    splits = list(_WRITE_DELIM_RE.finditer(tail))
    chosen = None
    for split in splits:
        name = tail[:split.start()].strip()
        if _looks_like_complete_file_name(name):
            chosen = (name, tail[split.end():].strip())
    if chosen is not None:
        return chosen
    if splits and _looks_like_complete_file_name(tail):
        return tail, ""
    if splits:
        first = splits[0]
        return tail[:first.start()].strip(), tail[first.end():].strip()
    return tail, ""


def _extract_write_content(raw):
    """Pull the '... with <content>' / 'containing <content>' tail of a write
    request so the planner can pass it as file content. Returns '' if absent.
    Runs on the RAW utterance so case, newlines and quoted content survive.
    F12: a delimiter word inside the file name is part of the name, not a
    delimiter (see :func:`_split_write_request`)."""
    split = _split_write_request(raw)
    if split:
        return _located_write_content(raw, split[1])
    return ""


#: Pronouns that refer to the folder of the previous turn ("that folder",
#: "inside it", "in there") — resolved from the last native run's artifacts.
_FOLDER_PRONOUN_RE = re.compile(
    r"\b(?:that|this|the same|same)\s+(?:folder|directory)\b"
    r"|\binside\s+(?:it|there|them)\b"
    r"|\bin\s+(?:it|there|them)\b"
    r"|\bthere\b",
    re.IGNORECASE,
)

#: Words that introduce the file content AFTER the location ("... folder and
#: write hello", "... folder containing hello", "... and inside just hello").
_LOCATED_CONTENT_RE = re.compile(
    r"\s+(?:and\s+)?(?:inside(?:\s+(?:that|this|the))?(?:\s+(?:text\s+)?file)?\s+)?"
    r"(?:just\s+)?(?:write|containing|with|as|saying|say|of)\s+(.+)$",
    re.IGNORECASE | re.DOTALL,
)

#: Default name for a nameless file request ("create a text file ..."), spoken
#: verbatim in the confirmation preview so the user's "yes" authorizes the
#: exact effect.
_DEFAULT_TEXT_FILE_NAME = "hello.txt"


#: R5 — inspect/existence verbs that make a turn a LOCAL folder read, never a
#: browser job. "Look" and "navigate" are deliberately ABSENT: "have a quick
#: look" inspects, but does not say with what tool — tool choice comes from
#: the target (local folder = local tool), never from "look"/"navigate".
_INSPECT_VERB_RE = re.compile(
    r"\b(?:check|verify|confirm|see|show|list|inspect|look\s+at|"
    r"have\s+a\s+(?:quick\s+)?look|tell\s+me\s+(?:what(?:'s| is))?|"
    r"what(?:'s|\s+is)\s+(?:in(?:side)?|there))"
    r"|\b(?:is\s+there|does\s+\w+\s+exist|exists?)\b",
    re.IGNORECASE,
)

#: R5 — a local folder target: the folder word, a pronoun, or a bare/named
#: local location. Web targets are excluded upstream by _CODE_TOOL_WEB_HINTS
#: and the URL guard, so this only ever names filesystem places.
_INSPECT_FOLDER_RE = re.compile(
    r"\b(?:folder|directory)\b"
    r"|\bthat\s+(?:folder|directory)\b"
    r"|\binside\s+(?:it|there|them)\b"
    r"|\bin\s+(?:it|there|them)\b",
    re.IGNORECASE,
)


def _local_inspect_folder(routed):
    """R5: the folder target of a local inspect/existence request, or None.

    "check whether folder Malik exists", "have a quick look at that folder",
    "see what is inside", "list that folder" -> ("named"|"pronoun", detail).
    Returns None for web-shaped targets (URL already excluded upstream) and
    for turns with no inspect verb at all, so plain chit-chat never routes.

    "Create a directory listing" is NOT an inspect request — "listing" there
    is the THING being created, not the act of listing. The inspect verb
    must not sit inside a create/make/write shaped turn.
    """
    if not routed or not _INSPECT_VERB_RE.search(routed):
        return None
    if re.match(r"^(?:create|make|write|save|update)\b", routed or ""):
        return None
    hint = _located_write_folder(routed)
    if hint is not None:
        return hint
    if _INSPECT_FOLDER_RE.search(routed):
        return ("pronoun", "")
    # Named folder without the folder word: "check whether malik exists",
    # "see what is inside mayankmalik".
    named = re.search(
        r"(?:whether|if|named\s+(?:folder\s+)?|called\s+(?:folder\s+)?"
        r"|folder\s+(?:named\s+|called\s+)?|inside\s+|in\s+folder\s+)"
        r"([A-Za-z][\w\- ]{1,60}?)\s*(?:exists?|is\s+there|folder|directory|$)",
        routed,
        re.IGNORECASE,
    )
    if named:
        name = named.group(1).strip().strip("\"'")
        if name and not _is_path_like(name):
            return ("named", name)
    return None


def _strip_code_tool_filler(raw):
    """Remove leading spoken fillers ("now", "please just", ...) so the
    anchored write/folder/run matches apply to the verb, not the filler."""
    stripped = _CODE_TOOL_FILLER_RE.sub("", (raw or "").strip())
    return stripped.strip() or (raw or "").strip()


def _located_write_folder(raw):
    """The destination hint of a located write, or None when the request
    names no folder ("inside that folder", "in mayankmalik", ...).

    Returns (kind, detail): ("pronoun", "") for "that folder"/"inside it"
    (resolved from the last run's artifacts), ("named", name) for an
    explicitly named folder. A bare "in <word>" is only a folder hint when
    the word is not a file name or content delimiter tail.
    """
    match = _CODE_TOOL_LOCATION_RE.search(raw or "")
    if match:
        named = (match.group(2) or "").strip()
        if named and _is_path_like(named):
            pass  # an extension-looking name is a FILE, not a folder
        elif named:
            return ("named", named.strip().strip("\"'"))
        else:
            # "in X" with no determiner and no folder word is too weak on its
            # own ("write hello in english" is content, not a location) —
            # only a determiner ("that folder", "the folder") or
            # inside/into/within/under counts as a location cue.
            preposition = (match.group(0).split() or [""])[0].lower()
            determiner = re.search(r"\b(?:the|this|that|my|our)\b",
                                   match.group(0), re.IGNORECASE)
            if (match.group(1).lower() in ("folder", "directory")
                    and (preposition in ("inside", "into", "within", "under")
                         or determiner)):
                return ("pronoun", "")
    # Bare folder name, no "folder" word: "create a file in mayankmalik
    # and write hello". The "and <verb>" shape keeps content phrases
    # ("write hello in english") from matching.
    bare = _CODE_TOOL_BARE_FOLDER_RE.search(raw or "")
    if bare:
        name = (bare.group(1) or "").strip().strip("\"'")
        if name and not _is_path_like(name):
            return ("named", name)
    return None


def _resolve_folder_hint(hint):
    """Turn a located-write folder hint into an absolute folder path.

    "that folder" resolves from the last native run's artifacts (the folder
    just created/confirmed); a named folder resolves against the desktop
    when unqualified. Returns None when unresolvable — the planner then
    asks which folder instead of guessing.
    """
    if not hint:
        return None
    kind, detail = hint
    if kind == "named":
        name = (detail or "").strip()
        if not name:
            return None
        lowered = name.lower()
        try:
            folders = _known_folders()
        except Exception:
            folders = {}
        for key in ("desktop", "documents", "downloads", "home"):
            base = folders.get(key) or ""
            if base and _normalize_fs_path(name) == _normalize_fs_path(base):
                return base
        desktop = folders.get("desktop") or ""
        candidate = name if os.path.isabs(name) else (
            os.path.join(desktop, name) if desktop else name)
        if os.path.isdir(candidate):
            return candidate
        return os.path.normpath(candidate)
    # Pronoun: the folder of the previous turn. Prefer the most recent
    # create_folder artifact; fall back to the parent dir of the most recent
    # write_file artifact.
    try:
        result, _task_text = last_task_result()
    except Exception:
        return None
    artifacts = list(getattr(result, "artifacts", None) or []) if result else []
    if not artifacts and result is not None:
        # A folder-creation run whose artifacts list is empty (older runs
        # recorded only step observations): recover the folder dir from the
        # most recent folder/file trace entry. Never guess — only real,
        # existing directories count.
        try:
            trace = list(getattr(result, "trace", None) or [])
        except Exception:
            trace = []
        for entry in reversed(trace):
            if not isinstance(entry, dict):
                continue
            tool = str(entry.get("tool") or "")
            if tool not in ("code.create_folder", "code.write_file"):
                continue
            raw_path = ""
            try:
                raw_path = str((entry.get("args") or {}).get("path") or "")
            except Exception:
                raw_path = ""
            if not raw_path:
                continue
            candidate = (raw_path if tool == "code.create_folder"
                         else os.path.dirname(raw_path))
            if candidate and os.path.isdir(candidate):
                return candidate
        return None
    for entry in reversed(artifacts):
        path = entry.get("path") if isinstance(entry, dict) else None
        if path and os.path.isdir(str(path)):
            return str(path)
    for entry in reversed(artifacts):
        path = entry.get("path") if isinstance(entry, dict) else None
        if path:
            parent = os.path.dirname(str(path))
            if parent and os.path.isdir(parent):
                return parent
    return None


def _folder_hint_clarification(hint):
    """Ask which folder a located write means (never guess the location)."""
    if hint and hint[0] == "named":
        return ("Which folder should I use, sir — I could not find "
                "'%s'. Please say the full folder name." % hint[1])
    return ("Which folder should I create that file in, sir? "
            "Please say the folder name.")


def _located_write_name(raw, content_after):
    """File name for a located write: an explicit name wins, else the
    predictable default (spoken verbatim in the confirmation preview)."""
    split = _split_write_request(_strip_code_tool_filler(raw))
    if split:
        candidate = (split[0] or "").strip().strip("\"'")
        if candidate and _is_path_like(candidate):
            return os.path.basename(candidate)
    tail = (content_after or "").strip().strip("\"'")
    if tail and _is_path_like(tail) and not re.search(r"\s", tail):
        return os.path.basename(tail)
    return _DEFAULT_TEXT_FILE_NAME


def _located_write_content(raw, content_after):
    """Content for a located write ("... folder and write hello")."""
    match = _LOCATED_CONTENT_RE.search(raw or "")
    if match:
        content = (match.group(1) or "").strip().strip("\"'")
        content = re.sub(r"^(?:it\s+)?(?:as\s+)?", "", content,
                         flags=re.IGNORECASE).strip()
        if content:
            return content
    return (content_after or "").strip()


# Extensions recognized when a request lists file types instead of names
# ("create 4 files .txt .py .js .html" -> file1.txt, file2.py, ...).
_MULTI_FILE_EXT_RE = re.compile(
    r"\.(txt|py|js|jsx|ts|tsx|html|css|json|csv|md|xml|yaml|yml|sh|bat|cmd|ini|cfg|log|env)\b",
    re.IGNORECASE,
)

# Naming/location cues that make a folder tail a real folder request
# ("named X", "by the name X", "on the desktop", "in temp") rather than a
# bare noun phrase like "a directory listing". "by the name" covers the
# "... folder on the desktop by the name X" phrasing a bare "named" misses.
_FOLDER_TAIL_HINT_RE = re.compile(
    r"\b(?:named|called|call it|name it|name it as|call it as|by the name|"
    r"by name|at|on|in|for)\b",
    re.IGNORECASE,
)

_COUNT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


# Windows known-folder GUIDs for SHGetKnownFolderPath (Desktop is often
# OneDrive-redirected, so ~/Desktop must never be assumed).
_KNOWN_FOLDER_IDS = {
    "desktop": "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}",
    "documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
    "downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
}

_KNOWN_FOLDERS_CACHE = None


def _known_folder_windows(folder_id):
    """Resolve one Windows known-folder GUID, or None on any failure."""
    try:
        import ctypes
        import uuid
        from ctypes import wintypes
        shell32 = ctypes.windll.shell32
        ole32 = ctypes.windll.ole32
        shell32.SHGetKnownFolderPath.argtypes = [
            ctypes.c_void_p, wintypes.DWORD, wintypes.HANDLE,
            ctypes.POINTER(wintypes.LPWSTR),
        ]
        shell32.SHGetKnownFolderPath.restype = wintypes.LONG
        ole32.CoTaskMemFree.argtypes = [wintypes.LPVOID]
        guid_bytes = uuid.UUID(folder_id).bytes_le
        guid = (ctypes.c_ubyte * 16).from_buffer_copy(guid_bytes)
        path_ptr = wintypes.LPWSTR()
        if shell32.SHGetKnownFolderPath(guid, 0, None, ctypes.byref(path_ptr)) != 0:
            return None
        path = path_ptr.value
        try:
            ole32.CoTaskMemFree(path_ptr)
        except Exception:
            pass
        return path or None
    except Exception:
        return None


def _known_folders():
    """Real per-machine folders: username, home, desktop, documents,
    downloads. Resolved lazily and cached at module level (patchable in
    tests). Falls back per-folder to ~/Desktop|Documents|Downloads."""
    global _KNOWN_FOLDERS_CACHE
    if _KNOWN_FOLDERS_CACHE is not None:
        return _KNOWN_FOLDERS_CACHE
    try:
        home = os.path.expanduser("~")
    except Exception:
        home = ""
    username = os.path.basename(os.path.normpath(home or "")) if home else ""
    if not username:
        try:
            username = getpass.getuser()
        except Exception:
            username = ""
    folders = {"username": username or "", "home": home or ""}
    for key, leaf in (("desktop", "Desktop"), ("documents", "Documents"),
                      ("downloads", "Downloads")):
        resolved = None
        if os.name == "nt":
            resolved = _known_folder_windows(_KNOWN_FOLDER_IDS[key])
        folders[key] = resolved or os.path.join(home, leaf) if home else (resolved or "")
    _KNOWN_FOLDERS_CACHE = folders
    return folders


def _sanitize_fs_text(text):
    """Substitute user/home placeholders in a path or command string.

    Case-insensitive and repeated: <username>/{username}/%username% ->
    real username; %userprofile%/%home% -> home; a leading ~ or ~/ ->
    home. Returns the input unchanged when it is empty or the folders
    cannot be resolved."""
    if not text:
        return text
    try:
        folders = _known_folders()
    except Exception:
        return text
    result = str(text)
    username = folders.get("username") or ""
    home = folders.get("home") or ""
    if username:
        for token in ("<username>", "{username}", "%username%"):
            result = re.sub(re.escape(token), lambda _m: username,
                            result, flags=re.IGNORECASE)
    if home:
        for token in ("%userprofile%", "%home%"):
            result = re.sub(re.escape(token), lambda _m: home,
                            result, flags=re.IGNORECASE)
        result = re.sub(r"^~(?=[/\\]|$)", lambda _m: home, result)
    return result


def _redirect_desktop(path):
    """Rewrite a path under the fallback ~/Desktop to the REAL desktop
    dir when this machine redirects it (e.g. OneDrive). No-op when the
    real desktop equals ~/Desktop, or when the path is not under it."""
    if not path:
        return path
    try:
        folders = _known_folders()
    except Exception:
        return path
    home = folders.get("home") or ""
    real = folders.get("desktop") or ""
    if not home or not real:
        return path
    local = os.path.join(home, "Desktop")
    if _normalize_fs_path(real) == _normalize_fs_path(local):
        return path
    if _normalize_fs_path(path) == _normalize_fs_path(local):
        return real
    if _path_is_inside(path, local):
        suffix = _normalize_fs_path(path)[len(_normalize_fs_path(local)):].lstrip(os.sep)
        return os.path.join(real, suffix) if suffix else real
    return path


_INVALID_PATH_RE = re.compile(r"""[<>|?*"]""")


def _path_has_invalid_chars(path):
    """True when a resolved path still contains wildcard/redirection
    characters (the drive colon is fine). Such steps must never execute."""
    try:
        return bool(_INVALID_PATH_RE.search(path or ""))
    except Exception:
        return False


# Tools whose args.path is a filesystem path (full sanitize treatment).
_PATH_ARG_TOOLS = (
    "code.write_file", "code.create_folder", "code.read_file",
    "code.list_directory", "code.run_script", "editor.open_file",
    "code.search", "code.read_range", "code.apply_patch",
    "code.inspect_diff", "code.run_checks",
)


def _folder_plan_path(raw):
    """Folder name out of 'create/make a folder/directory ...' phrasing,
    prefixed with the Desktop path when one is mentioned. Returns None when
    no deterministic name can be found (the LLM planner takes over then).
    Runs on the RAW utterance so folder names keep their original case."""
    name = None
    named = re.search(
        r"(?:named|called|call it|name it|and name it|name it as|call it as|"
        r"by the name(?: of)?|by name)"
        r"\s+([a-z0-9_ .\-]+?)(?:\s+(?:on|in|at|inside|under)\s+.*)?$",
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    if named:
        name = named.group(1).strip()
        # A location phrase that rode along with the name ("X on the
        # desktop") is a path cue, not part of the folder name.
        name = re.split(
            r"\s+(?:on|in|at|inside|under)\s+(?:the\s+)?(?:desktop|"
            r"documents|downloads|workspace|home|folder|directory)\b",
            name,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
    if not name:
        bare = re.match(
            r"^(?:create|make)\s+(?:a\s+|an\s+)?(?:folder|directory)\s+"
            r"(?:named|called|at|in|for)\s+([a-z0-9_ .\-]+)$",
            raw,
            re.IGNORECASE,
        )
        if bare:
            candidate = bare.group(1).strip()
            if candidate and not re.search(r"\s", candidate):
                name = candidate
    if not name:
        return None
    if re.search(r"\bdesktop\b", raw, re.IGNORECASE):
        return os.path.join(_known_folders()["desktop"], name)
    return name


def _multi_file_count(raw):
    """Spoken file count in 'create N files ...' phrasing, or None."""
    count_m = re.search(
        r"(?:create|make)\s+(?:(\d+)|(one|two|three|four|five|six|seven|eight|nine|ten))\s+files?\b",
        raw,
        re.IGNORECASE,
    )
    if not count_m:
        return None
    if count_m.group(1):
        return int(count_m.group(1))
    return _COUNT_WORDS.get(count_m.group(2).lower())


def _multi_file_names(raw):
    """File names out of 'create N files: a.txt, b.py, ...' phrasing.

    Named comma-separated files (path-like) win; otherwise the extensions in
    the tail become file1<ext>, file2<ext>, ... in order, padded to the
    spoken count (cycling the listed extensions) when fewer are listed.
    Runs on the RAW utterance so file names keep their original case.
    """
    files_m = re.search(r"\bfiles?\b(?:\s*(?:named|called|:|-)\s*|\s+)(.+)$", raw, re.IGNORECASE | re.DOTALL)
    if not files_m:
        return []
    tail = files_m.group(1)
    candidates = [t.strip().rstrip(".,;") for t in re.split(r"[,;]", tail)]
    names = [t for t in candidates if t and not re.search(r"\s", t) and _is_path_like(t)]
    if names:
        return names
    exts = _MULTI_FILE_EXT_RE.findall(tail)
    if exts:
        count = _multi_file_count(raw)
        if count and count > len(exts):
            names = [f"file{i + 1}.{ext}" for i, ext in enumerate(exts)]
            for j in range(count - len(exts)):
                pad_ext = exts[(len(exts) + j) % len(exts)]
                names.append(f"file{len(names) + 1}.{pad_ext}")
            return names
        return [f"file{i + 1}.{ext}" for i, ext in enumerate(exts)]
    return []


def _build_planner_prompt(command, context, observations=None):
    # F46: one BOUNDED envelope serialization — a prefix slice of serialized
    # JSON can be unparseable, so an oversized context becomes an explicit,
    # itself-bounded summary instead of a truncated document.
    compact_context = budgeted_json(context, 12000)
    try:
        folders = _known_folders()
    except Exception:
        folders = {}
    fs_context = (
        "FILESYSTEM CONTEXT (this machine - use these EXACT absolute paths "
        "in args.path and inside command strings; NEVER invent paths or use "
        "placeholders like <username>):\n"
        "- username: %s\n- home: %s\n- desktop: %s\n- documents: %s\n"
        "- downloads: %s\n" % (
            folders.get("username", ""), folders.get("home", ""),
            folders.get("desktop", ""), folders.get("documents", ""),
            folders.get("downloads", ""),
        )
    )
    # F14: the productivity connector advertises itself from its own typed
    # registry (planner_tool_lines), so the planner can never be shown a
    # calendar/mail operation the connector would not accept.
    try:
        productivity_lines = "\n".join(
            productivity_connector.planner_tool_lines())
    except Exception:  # noqa: BLE001 - a broken advertisement is not fatal
        productivity_lines = ""
    # F15: structured observations from steps that already ran (diagnostics,
    # versions, hashes, cursors...) travel to the planner as data, not as the
    # clipped prose that was spoken.
    observations_block = ""
    if observations:
        try:
            observations_block = (
                "\nOBSERVED STRUCTURED DATA (what earlier steps actually "
                "reported this run — plan against these real values):\n"
                "%s\n" % json.dumps(observations, ensure_ascii=True,
                                    default=str)[:6000])
        except Exception:  # noqa: BLE001 - never fail planning over this
            observations_block = ""
    return (
        "You are Jarvis Task Brain, a connector-first computer control planner.\n"
        "You are not watching a video feed. You are inside the app through structured connectors.\n"
        "Prefer direct structured tools over screen/OCR fallback.\n"
        "Return only a JSON object with keys: ok, confidence, summary, requires_confirmation, steps, response.\n"
        "Each step must be {tool, args, risk, reason}. Allowed tools:\n"
        "- editor.inspect_workspace: read editor bridge state (active file, diagnostics, workspace).\n"
        "- editor.execute_command: run a VS Code/Antigravity command by command id, only if the bridge is available.\n"
        "- editor.open_file: open a file path through the editor bridge.\n"
        "- editor.read_buffer: read editor buffer text for args.path (optional args.start_line/end_line); returns version for checked edits.\n"
        "- editor.diagnostics: list editor diagnostics (optional args.path filter).\n"
        "- editor.symbols: list document symbols for args.path.\n"
        "- editor.references: find references at args.path, args.line, args.character.\n"
        "- editor.workspace_search: text-search the editor workspace (args.query, optional args.limit).\n"
        "- editor.test_results: run the workspace test command and report structured results, risky.\n"
        "- editor.apply_workspace_edit: version-checked WorkspaceEdit (args.edits = [{path|uri, range, version, new_text}] — every edit carries the document version it was read at and, for multi-file edits, ONE VERSION PER DOCUMENT; a stale member rejects the whole edit; optional args.expected_version for a single-document edit), risky.\n"
        "- editor.edit_active_selection: replace one explicit document range (args.text, args.uri, args.version, args.selection = {start:{line,character}, end:{line,character}}); requires the version and range captured by editor.inspect_workspace/editor.read_buffer and is refused when the editor moved on, risky.\n"
        "- browser.inspect_tabs: inspect connected browser tabs.\n"
        "- browser.search_web: open a web search for args.query.\n"
        "- browser.open_url: open args.url.\n"
        "- windows.inspect_active_window: inspect current native window/UIA state.\n"
        "- windows.screen_action: fallback to UI Automation/OCR screen action using args.command.\n"
        "- code.read_file: read a text file (args.path).\n"
        "- code.write_file: create/overwrite a text file (args.path, args.content).\n"
        "- code.list_directory: list a folder (args.path).\n"
        "- code.create_folder: create a folder and parents (args.path).\n"
        "- code.run_command: run a shell/cmd command and capture output (args.command) — ONLY for explicit shell work, never as a substitute for the structured code tools below.\n"
        "- code.run_script: run a Python/batch script file (args.path, optional args.args).\n"
        "- code.search: search file contents (args.pattern, optional args.path, args.glob_pattern, args.max_results, args.cursor, args.regex); returns structured matches with a continuation cursor (next_cursor) that resumes at the first match the page did not show — keep paging rather than assuming the page is everything.\n"
        "- code.read_range: read a bounded line range of a file (args.path, args.start_line, optional args.end_line, args.max_lines); streams the file and returns structured lines ({line, text}), next_start for the first unreturned line and content_hash.\n"
        "- code.apply_patch: apply a validated unified-diff patch to a file (args.path, args.patch, optional args.dry_run, args.expected_hash); the patch must name args.path and its hunks must match their declared counts exactly, otherwise nothing is written.\n"
        "- code.inspect_diff: diff a file against its last restore point (optional args.path).\n"
        "- code.run_checks: run project checks (args.kind = auto/pytest/py_compile/node_check, optional args.path).\n"
        + (productivity_lines + "\n" if productivity_lines else "")
        + "F01 — plan against REAL observations, not guesses: a step may list "
        "\"depends_on\": [earlier step indices] and may reference what an "
        "earlier step will report with placeholders such as "
        "{{step0.path}}, {{step0.matches.0.path}}, {{step0.lines.0.text}} or "
        "{{last.files.0}}; unresolved placeholders are left "
        "as written and the step fails honestly. Every intended step must be "
        "included: the runtime executes a bounded number of steps and reports "
        "the rest as unmet goals, so never omit a needed step.\n"
        "Set requires_confirmation true for login/logout, destructive edits, purchases, sending, deleting, closing, settings changes, installs, or unknown-risk multi-step changes.\n"
        "If a required connector is unavailable, produce the best safe first step and explain the missing connector in response.\n\n"
        f"{fs_context}\n"
        f"{observations_block}"
        f"USER TASK:\n{command}\n\n"
        f"CONNECTOR CONTEXT:\n{compact_context}\n"
    )


def _memory_skill_offer(command):
    """F09: the trusted skills offered to the planner for *command*.

    Returns ``(block, offered)`` — the bounded grounding block and the
    identity of every promoted skill that went into the prompt, so the run
    that follows can report a verified replay (or a failure) back to the
    store. Best effort: a memory failure never blocks planning.
    """
    try:
        from backend.core import memory_store
        block = memory_store.recall_skills_for(command)
        offered = []
        for row in memory_store.find_skills(command, status="promoted",
                                            limit=2):
            offered.append({"id": row["id"], "name": row["name"],
                            "version": row["version"]})
        return block, offered
    except Exception:
        return "", []


def _with_memory_skills(plan, offered):
    """Attach the offered skill identity to the plan the executor receives."""
    if not offered:
        return plan
    if isinstance(plan, dict):
        plan = dict(plan)
        plan["memory_skills"] = [dict(entry) for entry in offered]
    return plan


def _model_plan(command, context, observations=None):
    prompt = _build_planner_prompt(command, context, observations)
    # G9 (F09): user-approved skills — WITH their procedures — and VERIFIED
    # prior outcomes for this kind of task ground the planner before it
    # improvises. Empty block (no prompt change) when nothing matches.
    memory_block, offered = _memory_skill_offer(command)
    if memory_block:
        prompt = prompt + "\n\n" + memory_block
    messages = [
        {
            "role": "system",
            "content": "Return strict JSON only. Do not include markdown or explanations outside JSON.",
        },
        {"role": "user", "content": prompt},
    ]
    # F49 (G8): planner model selection is resolved — and capability-
    # validated — through the registry, and the model/adapter snapshot
    # travels with the call. A registry failure degrades to the default
    # Fireworks call exactly as before (never a second registry).
    try:
        from backend.services import model_registry as _registry
        snapshot = _registry.get_model_config("planner")
    except Exception:
        result = ask_fireworks(messages, temperature=0.1, max_tokens=650)
        return _with_memory_skills(_parse_model_plan(result), offered)
    provider = snapshot.get("provider")
    model = snapshot.get("model")
    api_key, base_url = _registry.get_provider_credentials(provider)
    if provider == "fireworks" or not base_url:
        result = ask_fireworks(messages, temperature=0.1, max_tokens=650, model=model or None)
    else:
        from backend.services.openai_compat_client import ask_openai_compat
        result = ask_openai_compat(
            messages,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.1,
            max_tokens=650,
        )
    return _with_memory_skills(_parse_model_plan(result), offered)


def _parse_model_plan(result):
    if not result or not result.get("choices"):
        return None
    content = result["choices"][0].get("message", {}).get("content", "")
    parsed = _extract_json_object(content)
    return parsed if parsed else None


def _productivity_effect_precondition(tool, args):
    """F14 — obtain the separate per-effect approval a send/create needs.

    The planner's step holds no authority of its own: the connector records an
    approval bound to the draft's content hash and the current grant epoch,
    and refuses outright when the elevated scope was never delegated. Returns
    ``(args, preview, error)``; the approval id is embedded in the step so the
    confirmed plan executes exactly the effect the user heard described, and
    the connector re-verifies it (unconsumed, unrevoked, unexpired, unchanged
    draft, same epoch) immediately before the send/create.
    """
    if tool not in _PRODUCTIVITY_EFFECT_TOOLS:
        return args, "", ""
    if str(args.get("approval_id") or "").strip():
        return args, "", ""
    draft_id = str(args.get("draft_id") or "").strip()
    if not draft_id:
        return args, "", "no draft_id was given, so nothing was sent or created"
    try:
        approval = productivity_connector.request_effect_approval(draft_id)
    except Exception as exc:  # noqa: BLE001 - fail closed, never crash a plan
        return args, "", "the effect approval could not be requested: %s" % exc
    if not approval.get("ok"):
        return args, "", (approval.get("error")
                          or "the effect approval was refused")
    updated = dict(args)
    updated["approval_id"] = approval.get("approval_id")
    return updated, str(approval.get("preview") or ""), ""


def _normalize_plan(plan, command):
    if not isinstance(plan, dict):
        return {
            "ok": False,
            "confidence": 0.0,
            "summary": "",
            "requires_confirmation": False,
            "steps": [],
            "response": "I could not build a task plan.",
        }

    steps = plan.get("steps") or []
    if not isinstance(steps, list):
        steps = []
    # F01: the cap is a real budget, not a silent omission. The plan records
    # how many steps were INTENDED, and every step beyond the budget becomes an
    # explicit unmet goal, so a 10-step intent can never be reported as done
    # after executing 8.
    intended_steps = len([s for s in steps if isinstance(s, dict)])
    truncated = intended_steps > TASK_MAX_STEPS
    omitted = [s for s in steps[TASK_MAX_STEPS:] if isinstance(s, dict)]

    clean_steps = []
    requires_confirmation = bool(plan.get("requires_confirmation"))
    dependency_errors = []
    for index, raw in enumerate(steps[:TASK_MAX_STEPS]):
        if not isinstance(raw, dict):
            continue
        tool = str(raw.get("tool") or "").strip()
        if not tool:
            continue
        risk = str(raw.get("risk") or "safe").strip().lower()
        args = raw.get("args") if isinstance(raw.get("args"), dict) else {}
        # PATH SANITIZER (single pass every plan goes through, before the
        # confirmation preview is built): substitute placeholders and
        # redirect the fallback ~/Desktop to the real desktop dir so the
        # user approves the REAL paths. F12: this applies to designated
        # path/file fields only — a command line, script body or free-text
        # argument is payload and keeps every byte it was given (a command's
        # own shell placeholders, e.g. %USERPROFILE%, are the shell's to
        # expand, not ours to rewrite).
        args = dict(args)
        if tool in _PATH_ARG_TOOLS and isinstance(args.get("path"), str):
            args["path"] = os.path.normpath(
                _redirect_desktop(_sanitize_fs_text(args["path"])))
        if risk != "safe" or tool not in SAFE_TOOLS or tool in CONFIRM_TOOLS:
            requires_confirmation = True
        # F01: PRESERVE the dependency list and validate it. The old
        # normalizer dropped depends_on entirely, so a plan that declared a
        # prerequisite executed its dependents unconditionally. Only earlier
        # integer indices are provable; anything else (self, forward, bool,
        # negative, non-int) makes the step unprovable and it is refused
        # rather than run on a dependency nobody can check.
        depends_on, invalid = _validate_dependencies(raw.get("depends_on"), index)
        if invalid:
            dependency_errors.append(
                "step %d: %s" % (index, "; ".join(invalid)))
        clean_step = {
            "tool": tool,
            "args": args,
            "risk": risk,
            "reason": str(raw.get("reason") or "").strip(),
            "depends_on": depends_on,
        }
        if invalid:
            clean_step["unprovable_dependency"] = "; ".join(invalid)
            requires_confirmation = True
        # F14: a send/create is externally visible, so it is never implicit
        # and never self-authorised. The step gets the id (and the scrubbed
        # preview) of a separate per-draft approval; when that approval cannot
        # be requested — no draft, no delegated elevated scope — the step
        # carries an honest precondition failure and executes nothing.
        if tool in _PRODUCTIVITY_EFFECT_TOOLS:
            requires_confirmation = True
            args, effect_preview, precondition_error = \
                _productivity_effect_precondition(tool, args)
            clean_step["args"] = args
            if effect_preview:
                clean_step["effect_preview"] = effect_preview
            if precondition_error:
                clean_step["precondition_error"] = precondition_error
        clean_steps.append(clean_step)

    if not clean_steps and not plan.get("response"):
        clean_steps.append(
            {
                "tool": "windows.screen_action",
                "args": {"command": command},
                "risk": "safe",
                "reason": "Fallback to native UI control.",
            }
        )
        # Unstructured vision improvisation with the raw command must never
        # run unconfirmed.
        requires_confirmation = True

    return {
        "ok": bool(plan.get("ok", True)),
        "confidence": float(plan.get("confidence") or 0.0),
        "summary": _with_truncation_note(
            str(plan.get("summary") or "Working on the task.").strip(), truncated),
        "requires_confirmation": requires_confirmation,
        "steps": clean_steps,
        "response": _with_truncation_note(
            str(plan.get("response") or "").strip(), truncated),
        "truncated": truncated,
        # F01: what the planner INTENDED, and the goals that the step budget
        # did not admit. The executor keeps these as unmet goals, so a plan
        # that was cut short is partial — never completed.
        "intended_steps": intended_steps,
        "omitted_steps": omitted,
        "dependency_errors": dependency_errors,
        # F09: the trusted skills whose procedures were offered to the
        # planner — the executor reports the replay outcome back against
        # exactly these identities.
        "memory_skills": [
            {"id": entry["id"], "name": str(entry.get("name") or "")[:80],
             "version": entry.get("version")}
            for entry in (plan.get("memory_skills") or [])
            if isinstance(entry, dict) and entry.get("id") is not None
        ][:2],
    }


def _validate_dependencies(raw_depends_on, index):
    """F01: return (provable dependency indices, invalid-entry reasons).

    A dependency is only provable when it points at an EARLIER step in the
    same plan: any other value (self, forward reference, bool, negative,
    non-integer) cannot be checked, so the step that declares it is refused
    instead of executing on an unverifiable prerequisite.
    """
    if raw_depends_on is None:
        return [], []
    if not isinstance(raw_depends_on, (list, tuple, set)):
        return [], ["depends_on must be a list of earlier step indices"]
    valid = []
    invalid = []
    for dep in raw_depends_on:
        if isinstance(dep, bool) or not isinstance(dep, int):
            invalid.append("dependency %r is not a step index" % (dep,))
            continue
        if dep < 0:
            invalid.append("dependency %d is negative" % dep)
            continue
        if dep >= index:
            invalid.append(
                "dependency %d is not an earlier step" % dep)
            continue
        valid.append(dep)
    return sorted(set(valid)), invalid


def _with_truncation_note(text, truncated):
    """Make the step-budget cap explicit wherever the plan surfaces."""
    if not truncated:
        return text
    note = ("Plan limited to the first %d steps (the rest stay unmet)."
            % TASK_MAX_STEPS)
    return ("%s %s" % (text, note)).strip() if text else note


def plan_task(command, context=None, observations=None):
    context = context or gather_context()
    plan = _heuristic_plan(command, context)
    if plan is None:
        plan = _model_plan(command, context, observations)
    if plan is None:
        plan = {
            "ok": True,
            "confidence": 0.5,
            "summary": "Using the Windows control fallback.",
            "requires_confirmation": True,
            "steps": [
                {
                    "tool": "windows.screen_action",
                    "args": {"command": command},
                    "risk": "safe",
                    "reason": "No structured connector plan was available.",
                }
            ],
        }
    return _normalize_plan(plan, command)


# ── F15 (G4): structured editor inspection renderers ──────────────────────
#
# The bridge used to collapse every editor question into "Editor state is
# available through the bridge." These renderers turn the structured
# responses of integrations/jarvis-editor-bridge/extension.js into real,
# bounded inspection data the planning loop can act on.

#: vscode.DiagnosticSeverity (the bridge forwards the numeric enum).
_SEVERITY_LABELS = {0: "error", 1: "warning", 2: "info", 3: "hint"}

#: vscode.SymbolKind (the bridge forwards the numeric enum).
_SYMBOL_KINDS = {
    0: "file", 1: "module", 2: "namespace", 3: "package", 4: "class",
    5: "method", 6: "property", 7: "field", 8: "constructor", 9: "enum",
    10: "interface", 11: "function", 12: "variable", 13: "constant",
    14: "string", 15: "number", 16: "boolean", 17: "array", 18: "object",
    19: "key", 20: "null", 21: "enum member", 22: "struct", 23: "event",
    24: "operator", 25: "type parameter",
}

#: How many structured items one renderer lists before summarising the tail.
_EDITOR_ITEM_LIMIT = 20

#: Spoken/step-result text bound for a single tool call (shared by the F11
#: code-tool renderer and the F15 editor renderers below).
_STEP_TEXT_LIMIT = 800


def _editor_clip(text, limit=_STEP_TEXT_LIMIT):
    """Bound renderer output so a huge buffer can't flood the loop."""
    text = (text or "").rstrip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _editor_basename(value):
    """Short display name for a filesystem path *or* a file:// URI."""
    text = str(value or "").strip()
    if not text:
        return ""
    name = os.path.basename(text.replace("\\", "/").rstrip("/"))
    return name or text


def _editor_pos(point):
    """Render a 0-based ``{line, character}`` position as 1-based ``l:c``."""
    if not isinstance(point, dict):
        return "?"
    try:
        line = int(point.get("line") or 0) + 1
    except (TypeError, ValueError):
        line = "?"
    try:
        char = int(point.get("character") or 0) + 1
    except (TypeError, ValueError):
        char = "?"
    return "%s:%s" % (line, char)


def _editor_span(rng):
    """Render a 0-based range as ``start-end`` in 1-based coordinates."""
    if not isinstance(rng, dict):
        return "?"
    return "%s-%s" % (_editor_pos(rng.get("start")),
                      _editor_pos(rng.get("end")))


def _severity_label(value):
    """Diagnostic severity -> 'error' / 'warning' / 'info' / 'hint'."""
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    try:
        return _SEVERITY_LABELS.get(int(value), "info")
    except (TypeError, ValueError):
        return "info"


def _symbol_kind(value):
    """Numeric vscode.SymbolKind -> a readable word ('' when unknown)."""
    try:
        return _SYMBOL_KINDS.get(int(value), "")
    except (TypeError, ValueError):
        return str(value or "").strip().lower()


def _editor_state_text(state):
    """F15 — the actual structured editor snapshot.

    Replaces the old stub ("Editor state is available through the bridge.")
    with the real buffer identity, selection, workspace folders and
    diagnostics so the planning loop can target a document by URI/version
    instead of guessing at "the current file".
    """
    if not isinstance(state, dict) or not state:
        return "The editor bridge returned no state."

    lines = []
    active = state.get("activeFile")
    if isinstance(active, dict) and active:
        name = (active.get("fileName")
                or _editor_basename(active.get("path") or active.get("uri") or "")
                or "(untitled)")
        bits = []
        if active.get("languageId"):
            bits.append(str(active["languageId"]))
        if active.get("version") is not None:
            bits.append("version %s" % active["version"])
        if active.get("lineCount") is not None:
            bits.append("%s lines" % active["lineCount"])
        if active.get("isDirty"):
            bits.append("unsaved")
        if active.get("isUntitled"):
            bits.append("untitled")
        lines.append("Active file: %s%s." % (
            name, " (%s)" % ", ".join(bits) if bits else ""))
        if active.get("path"):
            lines.append("Path: %s." % active["path"])
        selection = active.get("selection")
        if isinstance(selection, dict) and (selection.get("start")
                                            or selection.get("end")):
            lines.append("Selection: %s." % _editor_span(selection))
        if active.get("textTruncated"):
            lines.append("Buffer text was truncated by the bridge.")
    else:
        lines.append("No file is open in the editor.")

    folders = [str(f) for f in (state.get("workspaceFolders") or []) if f]
    if folders:
        lines.append("Workspace folders: %s." % ", ".join(folders[:3]))

    diagnostics = state.get("diagnostics") or []
    if diagnostics:
        lines.append(_editor_diagnostics_text(diagnostics))
    return _editor_clip(" ".join(lines))


def _editor_buffer_text(result):
    """F15 — structured buffer read: identity + version + numbered lines."""
    name = _editor_basename(result.get("path") or result.get("uri") or "")
    header = "Read %s" % (name or "the buffer")
    start = result.get("startLine")
    end = result.get("endLine")
    try:
        start = max(1, int(start)) if start is not None else 1
    except (TypeError, ValueError):
        start = 1
    if start is not None and end is not None:
        header += " lines %s-%s" % (start, end)
    if result.get("lineCount") is not None:
        header += " of %s" % result["lineCount"]
    if result.get("version") is not None:
        header += " (version %s)" % result["version"]

    text = result.get("text") or ""
    if not text.strip():
        return "%s. (empty)" % header
    numbered = "\n".join(
        "%6d | %s" % (start + offset, line)
        for offset, line in enumerate(text.splitlines())
    )
    out = "%s.\n%s" % (header, numbered)
    if result.get("textTruncated"):
        out += ("\n...[buffer truncated by the bridge — read the rest with "
                "start_line/end_line]")
    return _editor_clip(out)


def _editor_diagnostics_text(diagnostics):
    """F15 — diagnostics as `{file}:{line}:{col} severity message` items."""
    items = [item for item in (diagnostics or []) if isinstance(item, dict)]
    if not items:
        return "No diagnostics reported."

    counts = {}
    for item in items:
        label = _severity_label(item.get("severity"))
        counts[label] = counts.get(label, 0) + 1
    breakdown = ", ".join(
        "%d %s%s" % (count, label, "" if count == 1 else "s")
        for label, count in sorted(counts.items())
    )
    lines = ["%d diagnostic%s (%s)." % (
        len(items), "" if len(items) == 1 else "s", breakdown)]
    for item in items[:_EDITOR_ITEM_LIMIT]:
        where = _editor_basename(item.get("file") or item.get("path")
                                 or item.get("uri") or "") or "?"
        pos = _editor_pos((item.get("range") or {}).get("start"))
        label = _severity_label(item.get("severity"))
        code = str(item.get("code") or "").strip()
        message = re.sub(r"\s+", " ", str(item.get("message") or "")).strip()
        prefix = "%s:%s %s%s" % (where, pos, label,
                                 " %s" % code if code else "")
        lines.append("- %s: %s" % (prefix, message[:200]))
    if len(items) > _EDITOR_ITEM_LIMIT:
        lines.append("- ...[%d more]" % (len(items) - _EDITOR_ITEM_LIMIT))
    return _editor_clip("\n".join(lines))


def _editor_symbols_text(symbols):
    """F15 — document symbols as `kind name at line:col` items."""
    items = [item for item in (symbols or []) if isinstance(item, dict)]
    if not items:
        return "No symbols reported."
    lines = ["%d symbol%s." % (len(items), "" if len(items) == 1 else "s")]
    for item in items[:_EDITOR_ITEM_LIMIT]:
        kind = _symbol_kind(item.get("kind"))
        name = str(item.get("name") or "?").strip()
        label = "%s %s" % (kind, name) if kind else name
        container = str(item.get("containerName") or "").strip()
        if container:
            label += " (in %s)" % container
        pos = _editor_pos((item.get("range") or {}).get("start"))
        lines.append("- %s at %s" % (label, pos))
    if len(items) > _EDITOR_ITEM_LIMIT:
        lines.append("- ...[%d more]" % (len(items) - _EDITOR_ITEM_LIMIT))
    return _editor_clip("\n".join(lines))


def _editor_references_text(references):
    """F15 — references as `{file}:{line}:{col}` locations."""
    items = [item for item in (references or []) if isinstance(item, dict)]
    if not items:
        return "No references found."
    lines = ["%d reference%s." % (len(items), "" if len(items) == 1 else "s")]
    for item in items[:_EDITOR_ITEM_LIMIT]:
        where = _editor_basename(item.get("path") or item.get("uri") or "") or "?"
        pos = _editor_pos((item.get("range") or {}).get("start"))
        lines.append("- %s:%s" % (where, pos))
    if len(items) > _EDITOR_ITEM_LIMIT:
        lines.append("- ...[%d more]" % (len(items) - _EDITOR_ITEM_LIMIT))
    return _editor_clip("\n".join(lines))


def _editor_search_text(result):
    """F15 — workspace search matches (1-based lines, like code.search)."""
    matches = [item for item in (result.get("matches") or [])
               if isinstance(item, dict)]
    query = str(result.get("query") or "").strip()
    if not matches:
        return 'No workspace matches for "%s".' % query
    lines = ['%d match%s for "%s" (scanned %s file%s).' % (
        len(matches), "" if len(matches) == 1 else "es", query,
        result.get("scannedFiles") or 0,
        "" if (result.get("scannedFiles") or 0) == 1 else "s")]
    for item in matches[:_EDITOR_ITEM_LIMIT]:
        where = _editor_basename(item.get("path") or "") or "?"
        text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
        lines.append("- %s:%s: %s" % (where, item.get("line") or "?",
                                      text[:160]))
    if len(matches) > _EDITOR_ITEM_LIMIT or result.get("truncated"):
        lines.append("- ...[more matches — narrow the query or raise the "
                     "limit]")
    return _editor_clip("\n".join(lines))


def _editor_tests_text(result):
    """F15 — structured test run: verdict, exit code and bounded output."""
    command = str(result.get("command") or "").strip() or "the workspace tests"
    verdict = "Tests passed" if result.get("ok") else "Tests failed"
    if result.get("exitCode") is not None:
        verdict += " (exit %s)" % result["exitCode"]
    lines = ["%s: %s" % (verdict, command)]
    if result.get("cwd"):
        lines.append("cwd: %s" % result["cwd"])
    output = str(result.get("output") or "").strip()
    if output:
        lines.append(_editor_clip(output))
    return _editor_clip("\n".join(lines))


def _editor_tool_text(tool, result, render):
    """F15 — render one editor-bridge response as step text.

    A rejected or errored response becomes ``'<tool> failed: <reason>'`` so
    the closed loop's ``failed_prefix`` check records an honest failure
    instead of mistaking an error message for a successful result. *render*
    turns a successful payload into compact, structured inspection text.
    """
    if not isinstance(result, dict) or not result:
        return "%s failed: the editor bridge did not respond." % tool
    if result.get("ok") is False:
        reason = re.sub(r"\s+", " ", str(
            result.get("error") or "the editor bridge rejected the request")
        ).strip()
        message = "%s failed: %s" % (tool, reason)
        # F15: keep the structured body when there is one. A failing test
        # run is still a real result with output the loop needs — the
        # 'failed:' prefix above is what marks the step as failed.
        try:
            detail = (render(result) or "").strip()
        except Exception:  # noqa: BLE001 - a bad shape just stays unrendered
            detail = ""
        return "%s\n%s" % (message, detail) if detail else message
    try:
        text = render(result)
    except Exception as exc:  # noqa: BLE001 - a bad shape is a failure, not a crash
        logging.warning("Editor bridge render failed: %s", exc)
        return "%s failed: could not read the editor bridge response." % tool
    return (text or "").strip() or "(no data from the editor bridge)"


#: F15 — every editor tool. They all share one implementation so the prose
#: path (_execute_step) and the structured path (_execute_step_structured)
#: can never drift apart.
_EDITOR_TOOLS = (
    "editor.inspect_workspace", "editor.execute_command", "editor.open_file",
    "editor.edit_active_selection", "editor.read_buffer", "editor.diagnostics",
    "editor.symbols", "editor.references", "editor.workspace_search",
    "editor.test_results", "editor.apply_workspace_edit",
)


def _editor_structured(result):
    """A bridge payload with an explicit ``ok``, or None (F15).

    The structured body is what later planning steps resolve against, so it
    is returned untouched; only a missing ``ok`` is filled in (from ``error``)
    so the executor's verdict stays honest.
    """
    if not isinstance(result, dict) or not result:
        return None
    payload = dict(result)
    if "ok" not in payload:
        payload["ok"] = not bool(payload.get("error"))
    return payload


def _execute_editor_step(tool, args, context):
    """(text, structured|None) for one editor tool (F15).

    F15 — the structured response travels back UNTOUCHED next to the bounded
    spoken rendering: the loop records real diagnostics, document versions,
    buffers and hashes, so later planning works from data instead of from a
    clipped sentence. ``None`` means the bridge never answered at all.
    """
    editor = context.get("editor", {})
    available = bool(editor.get("available"))
    if tool == "editor.inspect_workspace":
        if not available:
            return (editor.get("setup_hint")
                    or "Editor bridge is not connected."), None
        # F15 — return the actual structured inspection data to the planning
        # loop, not a stub saying the data exists somewhere.
        state = editor.get("state") or {}
        text = _editor_state_text(state)
        structured = dict(state) if isinstance(state, dict) else {}
        structured["ok"] = True
        return text, structured
    if tool == "editor.execute_command":
        if not available:
            return "Editor bridge is not connected, so I cannot run editor commands yet.", None
        result = editor_bridge.execute_command(args.get("command") or "",
                                              args.get("args") or [])
        text = (result.get("message") or result.get("error")
                or "Editor command executed.")
        return text, _editor_structured(result)
    if tool == "editor.open_file":
        if not available:
            return "Editor bridge is not connected, so I cannot open files inside the editor yet.", None
        result = editor_bridge.open_file(args.get("path") or "")
        text = result.get("message") or result.get("error") or "Opened the file."
        return text, _editor_structured(result)
    if tool == "editor.edit_active_selection":
        if not available:
            return "Editor bridge is not connected, so I cannot edit the selection yet.", None
        # F15 — the active selection is NOT a stable edit target: between
        # the plan/confirmation and this call the user may have clicked
        # elsewhere, typed at the same position, or closed the document. The
        # edit needs the URI, the document version and the explicit range
        # captured at plan time, is re-validated against the LIVE state, and
        # is refused — without touching the editor — when any of that is
        # missing or has drifted. A bridge failure is a refusal too: this
        # path used to fail OPEN on a hiccup.
        planned = editor.get("state") or {}
        active = planned.get("activeFile") if isinstance(planned, dict) else {}
        active = active if isinstance(active, dict) else {}
        uri = str(args.get("uri") or active.get("uri")
                  or active.get("path") or "").strip()
        version = (args.get("version") if args.get("version") is not None
                   else active.get("version"))
        selection = (args.get("selection")
                     if isinstance(args.get("selection"), dict)
                     else active.get("selection"))
        if not uri or version is None or not isinstance(selection, dict):
            return ("editor.edit_active_selection failed: no document uri, "
                    "version and explicit range were captured for this edit, "
                    "so nothing was changed — re-run "
                    "editor.inspect_workspace and target the document and "
                    "range explicitly."), None
        try:
            still_there = editor_bridge.active_selection_matches(
                expected_uri=uri, expected_selection=selection,
                expected_version=version)
        except Exception as exc:  # noqa: BLE001 - fail closed, never edit
            logging.warning("Editor state check failed: %s", exc)
            return ("editor.edit_active_selection failed: the editor state "
                    "could not be verified, so nothing was changed — check "
                    "the editor bridge and re-run editor.inspect_workspace."), None
        if not still_there:
            return ("editor.edit_active_selection failed: the selection "
                    "changed since this was planned — re-run "
                    "editor.inspect_workspace and target the new "
                    "selection explicitly."), None
        result = editor_bridge.edit_active_selection(
            args.get("text") or "", uri=uri, path=active.get("path") or "",
            version=version, selection=selection)
        text = (result.get("message") or result.get("error")
                or "Edited the active selection.")
        return text, _editor_structured(result)
    # ── F15 (G4): structured coding interface ────────────────────────────
    if tool == "editor.read_buffer":
        if not available:
            return "Editor bridge is not connected, so I cannot read editor buffers yet.", None
        result = editor_bridge.read_buffer(
            path=args.get("path") or "", uri=args.get("uri") or "",
            start_line=args.get("start_line"), end_line=args.get("end_line"))
        return (_editor_tool_text(tool, result, _editor_buffer_text),
                _editor_structured(result))
    if tool == "editor.diagnostics":
        if not available:
            return "Editor bridge is not connected, so I cannot read diagnostics yet.", None
        result = editor_bridge.diagnostics(path=args.get("path") or "")
        return (_editor_tool_text(
            tool, result, lambda r: _editor_diagnostics_text(
                r.get("diagnostics") or [])), _editor_structured(result))
    if tool == "editor.symbols":
        if not available:
            return "Editor bridge is not connected, so I cannot read symbols yet.", None
        result = editor_bridge.symbols(path=args.get("path") or "")
        return (_editor_tool_text(
            tool, result, lambda r: _editor_symbols_text(r.get("symbols") or [])),
            _editor_structured(result))
    if tool == "editor.references":
        if not available:
            return "Editor bridge is not connected, so I cannot find references yet.", None
        result = editor_bridge.references(
            path=args.get("path") or "", line=args.get("line"),
            character=args.get("character"))
        return (_editor_tool_text(
            tool, result, lambda r: _editor_references_text(
                r.get("references") or [])), _editor_structured(result))
    if tool == "editor.workspace_search":
        if not available:
            return "Editor bridge is not connected, so I cannot search the workspace yet.", None
        result = editor_bridge.workspace_search(
            query=args.get("query") or "", limit=args.get("limit") or 50)
        return (_editor_tool_text(tool, result, _editor_search_text),
                _editor_structured(result))
    if tool == "editor.test_results":
        if not available:
            return "Editor bridge is not connected, so I cannot run workspace tests yet.", None
        result = editor_bridge.test_results(
            command=args.get("command") or None, cwd=args.get("cwd") or None)
        return (_editor_tool_text(tool, result, _editor_tests_text),
                _editor_structured(result))
    if tool == "editor.apply_workspace_edit":
        if not available:
            return "Editor bridge is not connected, so I cannot apply workspace edits yet.", None
        # F15 — the edits carry their per-document version/range; the bridge
        # refuses locally when a precondition is missing and the extension
        # refuses the WHOLE edit when any member is stale.
        edits = args.get("edits") if isinstance(args.get("edits"), list) else []
        result = editor_bridge.apply_workspace_edit(
            edits, expected_version=args.get("expected_version"))
        if result.get("status") == 409 or "mismatch" in str(
                result.get("error", "")).lower():
            return (("editor.apply_workspace_edit failed: %s — the document "
                     "changed since it was read; re-read with "
                     "editor.read_buffer and retry."
                     % (result.get("error") or "version mismatch")),
                    _editor_structured(result))
        return (_editor_tool_text(tool, result, lambda r: r.get("message")
                                  or "Workspace edit applied."),
                _editor_structured(result))
    return "Unknown task tool: %s." % tool, None


def _productivity_operation(tool, args):
    """F14 — the connector is the only entry point for its typed operations.

    An externally visible send/create is refused outright when the step does
    not carry the separate approval id the user's confirmation produced; the
    connector then re-verifies that id (unconsumed, unrevoked, unexpired,
    unchanged draft, same grant epoch) immediately before the effect.
    """
    if tool in _PRODUCTIVITY_EFFECT_TOOLS and not str(
            (args or {}).get("approval_id") or "").strip():
        return {
            "ok": False, "tool": tool, "external_effect": False,
            "error": ("no separate approval was given for this externally "
                      "visible effect, so nothing was sent or created"),
        }
    try:
        return productivity_connector.run_operation(tool, args)
    except Exception as exc:  # noqa: BLE001 - an exception is not permission
        logging.warning("Productivity operation %s failed: %s", tool, exc)
        return {"ok": False, "tool": tool, "error": str(exc),
                "external_effect": False}


def _productivity_step_text(tool, result):
    """Honest step text for one connector result (F14)."""
    if not isinstance(result, dict):
        return "%s failed: the productivity connector did not respond." % tool
    if result.get("ok") is False:
        reason = (result.get("error") or result.get("reason")
                  or "the connector refused the request")
        return "%s failed: %s" % (tool, reason)
    content = str(result.get("content") or "").strip()
    return content or "(the connector returned no content)"


def _execute_step(step, context):
    tool = step.get("tool")
    args = step.get("args") or {}

    # F14: a connector effect whose separate approval could not be obtained
    # carries an honest precondition failure — it executes nothing.
    precondition_error = str(step.get("precondition_error") or "").strip()
    if precondition_error:
        return f"{tool} failed: {precondition_error}"

    if tool == "browser.search_web":
        return browser_cdp.search_web(args.get("query") or "")
    if tool == "browser.open_url":
        return browser_cdp.open_url(args.get("url") or "")
    if tool == "browser.inspect_tabs":
        tabs = context.get("browser", {}).get("tabs", [])
        if not tabs:
            return "No browser tabs are available through CDP."
        titles = [tab.get("title") or tab.get("url") for tab in tabs[:5]]
        return "Open browser tabs: " + "; ".join(title for title in titles if title)
    if tool in _EDITOR_TOOLS:
        return _execute_editor_step(tool, args, context)[0]
    if tool == "windows.inspect_active_window":
        return _summarize_context(context)
    if tool == "windows.screen_action":
        return windows_connector.perform_screen_action(args.get("command") or "")
    if tool in _CODE_TOOLS:
        # F17: our own task path declares the authority it holds; the typed
        # boundary in code_tools refuses anything undeclared.
        result = code_tools.call_tool(tool, args,
                                      grants=code_tools.agent_grants())
        if result.get("ok"):
            # Keep responses compact/spoken-friendly, but never lose the
            # continuation cursor (F11).
            return _code_tool_text(result)
        err = result.get("error") or "the tool reported an error"
        return f"{tool} failed: {err}"
    # F14: semantic productivity connectors (mail/calendar/contacts) — one
    # typed dispatch through the connector's own registry.
    if productivity_connector.is_productivity_tool(tool):
        return _productivity_step_text(tool, _productivity_operation(tool, args))
    return f"Unknown task tool: {tool}."


_CODE_TOOLS = (
    "code.read_file", "code.write_file", "code.list_directory",
    "code.create_folder", "code.run_command", "code.run_script",
    # F11 (G4): the inspection/edit loop.
    "code.search", "code.read_range", "code.apply_patch",
    "code.inspect_diff", "code.run_checks",
)

def _continuation_note(result):
    """F11 — the continuation cursor, in one short sentence.

    Structured tools (search/read_range) return ``has_more`` + a cursor so
    the rest of the file/result set is never silently lost; this renders the
    cursor so the planning loop (and the user) can continue exactly where
    the call stopped.
    """
    if not isinstance(result, dict) or not result.get("has_more"):
        return ""
    if result.get("next_start") is not None:
        return "more available — continue with start_line=%s" % result["next_start"]
    if result.get("next_cursor") is not None:
        return "more available — continue with cursor=%s" % result["next_cursor"]
    return "more available"


def _code_tool_text(result):
    """Render a code-tool result as step text: compact, but with the
    continuation cursor preserved even when the content is clipped (F11).

    F11 — this is the HUMAN/SPOKEN rendering only. The structured payload
    (``path``/``content_hash``/``lines``/``matches``/``next_cursor``/...)
    travels separately: :func:`_execute_step_structured` hands the untouched
    result dict to :func:`_record_observation`, so later planning steps can
    still resolve ``{{step0.matches.0.path}}`` or ``{{step0.next_cursor}}``
    even when this text was clipped. The code tools themselves already page
    their ``content`` to fit, so clipping here is a safety net; when it does
    fire it never cuts mid-line and never hides that the page was cut short.
    """
    content = (result.get("content") or "").strip()
    note = _continuation_note(result)
    limit = _STEP_TEXT_LIMIT
    if len(content) <= limit:
        return (content + ("\n" + note if note else "")).strip() or "(ok)"
    budget = max(0, limit - len(note) - 40) if note else limit
    clipped = content[:budget]
    cut = clipped.rfind("\n")
    if cut > 0:
        clipped = clipped[:cut]
    clipped = clipped.rstrip()
    if note:
        return (clipped + "\n...[page clipped for speech]\n" + note).strip()
    return (clipped + "\n...[page clipped for speech]").strip()


def _execute_step_structured(step, context):
    """(text, tool_result_dict|None) variant of _execute_step.

    _execute_step itself is UNCHANGED (same signature/behavior for its
    callers); this internal variant reuses code_tools.call_tool directly
    for code tools so the closed loop can apply deterministic
    postconditions on the resolved path / exit code, and returns the
    structured editor/connector payloads (F15/F14) so later planning sees
    data instead of only the spoken rendering. Never raises: exceptions
    become '<tool> failed: ...' texts like _execute_step's.
    """
    tool = step.get("tool") if isinstance(step, dict) else ""
    args = step.get("args") if isinstance(step, dict) else {}
    if not isinstance(args, dict):
        args = {}
    if tool in _CODE_TOOLS:
        try:
            result = code_tools.call_tool(tool, args,
                                          grants=code_tools.agent_grants())
        except Exception as exc:
            logging.warning("Task step failed: %s", exc)
            return f"{tool} failed: {exc}", {"ok": False, "error": str(exc)}
        if not isinstance(result, dict):
            result = {"ok": False, "error": "the tool reported an error"}
        if result.get("ok"):
            return _code_tool_text(result), result
        err = result.get("error") or "the tool reported an error"
        return f"{tool} failed: {err}", result
    if tool in _EDITOR_TOOLS:
        # F15: the structured inspection body (diagnostics, versions, buffer
        # lines, hashes) is recorded for later planning, not clipped to prose.
        try:
            return _execute_editor_step(tool, args, context)
        except Exception as exc:
            logging.warning("Task step failed: %s", exc)
            return f"{tool} failed: {exc}", {"ok": False, "error": str(exc)}
    if productivity_connector.is_productivity_tool(tool):
        # F14: one dispatch, and the typed payload is recorded for later
        # planning as well (the text is only the spoken rendering).
        precondition_error = str(step.get("precondition_error") or "").strip()
        if precondition_error:
            return (f"{tool} failed: {precondition_error}",
                    {"ok": False, "error": precondition_error})
        try:
            result = _productivity_operation(tool, args)
        except Exception as exc:
            logging.warning("Task step failed: %s", exc)
            return f"{tool} failed: {exc}", {"ok": False, "error": str(exc)}
        return _productivity_step_text(tool, result), result
    try:
        return _execute_step(step, context), None
    except Exception as exc:
        logging.warning("Task step failed: %s", exc)
        return f"{tool} failed: {exc}", None


def _normalize_fs_path(path):
    """Normalize for dependency prefix matching (case-insensitive on
    Windows via normcase, separator-aware via normpath)."""
    try:
        return os.path.normcase(os.path.normpath(os.path.expanduser(str(path or ""))))
    except Exception:
        return str(path or "")


def _path_is_inside(child, folder):
    """True when `child` is strictly inside `folder` (prefix match on the
    normalized forms; equality does not count)."""
    child_n = _normalize_fs_path(child)
    folder_n = _normalize_fs_path(folder)
    if not child_n or not folder_n or child_n == folder_n:
        return False
    return child_n.startswith(folder_n + os.sep)


def _step_target(step):
    """Short human target for summaries: path, command, query or url."""
    args = step.get("args") or {}
    if not isinstance(args, dict):
        return ""
    # F11: code.search targets a *pattern* (its path is only the root).
    return str(args.get("path") or args.get("command")
               or args.get("query") or args.get("url")
               or args.get("pattern") or "").strip()


def _ok_fragment(step, text):
    """One short honest phrase for a successful step: the actual short
    result text when it fits, else a compact action phrase."""
    flat = re.sub(r"\s+", " ", (text or "")).strip()
    if flat and len(flat) <= 120:
        return flat if flat.endswith((".", "!", "?")) else flat + "."
    tool = step.get("tool") or ""
    target = _step_target(step)
    if tool == "code.create_folder":
        return f"Created folder {target}." if target else "Created the folder."
    if tool == "code.write_file":
        return f"Wrote file {target}." if target else "Wrote the file."
    if tool == "code.read_file":
        return f"Read file {target}." if target else "Read the file."
    if tool == "code.list_directory":
        return f"Listed {target}." if target else "Listed the folder."
    if tool == "code.run_command":
        short = (target[:60] + "...") if len(target) > 60 else target
        return f"Ran {short}." if short else "Ran the command."
    if tool == "code.run_script":
        return f"Ran script {target}." if target else "Ran the script."
    if tool == "code.search":
        return f"Searched {target or 'the workspace'}."
    if tool == "code.read_range":
        return f"Read part of {target}." if target else "Read part of the file."
    if tool == "code.apply_patch":
        return f"Patched {target}." if target else "Applied the patch."
    if tool == "code.inspect_diff":
        return f"Inspected changes to {target}." if target else "Inspected recent changes."
    if tool == "code.run_checks":
        return "Ran the checks."
    # ── F15 (G4): the structured editor tools ───────────────────────────
    if tool == "editor.read_buffer":
        return f"Read buffer {target}." if target else "Read the editor buffer."
    if tool == "editor.diagnostics":
        return "Checked the diagnostics."
    if tool == "editor.symbols":
        return f"Listed symbols in {target}." if target else "Listed the symbols."
    if tool == "editor.references":
        return f"Found references from {target}." if target else "Found the references."
    if tool == "editor.workspace_search":
        return f"Searched the workspace for {target}." if target else "Searched the workspace."
    if tool == "editor.test_results":
        return "Ran the workspace tests."
    if tool == "editor.apply_workspace_edit":
        return "Applied the workspace edit."
    if tool == "browser.search_web":
        return "Searched the web."
    if tool == "browser.open_url":
        return f"Opened {target}." if target else "Opened the page."
    short = tool.split(".")[-1].replace("_", " ") if tool else "step"
    return f"Finished {short}."


def _fail_fragment(step, reason):
    """One short honest phrase for a failed step: what + why."""
    tool = step.get("tool") or ""
    target = _step_target(step)
    reason = re.sub(r"\s+", " ", (reason or "unknown error")).strip()
    if len(reason) > 140:
        reason = reason[:137].rstrip() + "..."
    if tool == "code.write_file" and target:
        return f"File {target} failed: {reason}."
    if tool == "code.create_folder" and target:
        return f"Folder {target} failed: {reason}."
    if target:
        return f"{target} failed: {reason}."
    return f"{tool or 'The step'} failed: {reason}."


def _summarize_outcomes(outcomes, plan):
    """Spoken-friendly 1-3 line summary reflecting ACTUAL step outcomes.

    All ok -> 'Done, sir. ...'. Any failed/skipped -> 'Partly done, sir.
    ...' with each failure/skip named. Never the pre-authored response.
    """
    done = [item for item in outcomes if item["status"] == "ok"]
    bad = [item for item in outcomes if item["status"] != "ok"]
    if not outcomes:
        return plan.get("summary") or "Done, sir."
    if any(item["status"] == "cancelled" for item in outcomes):
        # F20: a cancelled turn says it was stopped — never "partly done",
        # which would read as if the remaining steps had merely failed.
        bits = [item["fragment"] for item in done if item["status"] == "ok"]
        bits += [item["fragment"] for item in bad
                 if item["status"] == "cancelled"]
        return re.sub(r"\s+", " ", " ".join(bits)).strip() or (
            "Stopped per your request.")
    if not bad:
        summary = "Done, sir. " + " ".join(item["fragment"] for item in done)
    else:
        bits = [item["fragment"] for item in done]
        bits += [item["fragment"] for item in bad]
        summary = "Partly done, sir. " + " ".join(bits)
    summary = re.sub(r"\s+", " ", summary).strip()
    if len(summary) > 300:
        cut = summary.rfind(" ", 0, 297)
        summary = (summary[:cut] if cut > 0 else summary[:297]).rstrip() + "..."
    return summary



def _clip_preview(value, limit=120):
    value = (value or "").strip()
    if len(value) > limit:
        return value[: limit - 3].rstrip() + "..."
    return value


def _confirmation_preview(plan):
    """Concise spoken preview of the WHOLE plan (F18).

    The old version described only the FIRST sensitive step, so a plan whose
    later steps ran commands or overwrote files was approved without the user
    ever hearing about them. The lead effect is still described in full; every
    remaining effect is then counted, and the complete ordered list (with
    targets and expiry) lives on the shared approval record the UI reads.
    """
    steps = [s for s in (plan.get("steps") or []) if isinstance(s, dict)]
    sensitive = [
        s for s in steps
        if s.get("tool") in CONFIRM_TOOLS or s.get("risk") != "safe"
    ]
    if not sensitive:
        sensitive = steps[:1]
    if not sensitive:
        return "Ready to proceed."
    lead = _describe_native_step(sensitive[0])
    rest = sensitive[1:]
    if not rest:
        return lead
    counts = tool_policy.effect_counts(
        tool_policy.describe_plan({"steps": rest}))
    return "%s Plus %s." % (lead, counts)


def _describe_native_step(step):
    """One sensitive native step, described with its real target."""
    tool = step.get("tool")
    args = step.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    if tool in CONFIRM_SENSITIVE_PREVIEW:
        if tool == "code.run_command":
            return f"Ready to run: {_clip_preview(args.get('command') or '(command)')}."
        if tool == "code.run_script":
            return f"Ready to run: {_clip_preview(args.get('path') or '(script)')}."
        if tool == "code.write_file":
            content = (args.get("content") or "").strip()[:80]
            path = args.get("path") or "(file)"
            if content:
                return f"Ready to write file {path} with: {content}."
            return f"Ready to write file {path}."
        if tool == "code.create_folder":
            return f"Ready to create folder: {_clip_preview(args.get('path') or '(folder)')}."
        if tool == "code.apply_patch":
            path = args.get("path") or "(file)"
            hunks = len(re.findall(r"^@@ ", args.get("patch") or "", re.MULTILINE))
            if hunks:
                return f"Ready to patch {path} ({hunks} hunk{'s' if hunks != 1 else ''})."
            return f"Ready to patch {path}."
        if tool == "editor.apply_workspace_edit":
            edits = args.get("edits") if isinstance(args.get("edits"), list) else []
            names = []
            for edit in edits[:3]:
                if not isinstance(edit, dict):
                    continue
                name = edit.get("path") or edit.get("uri") or ""
                if name:
                    names.append(os.path.basename(
                        str(name).replace("\\", "/").rstrip("/")))
            if names:
                shown = ", ".join(names)
                if len(edits) > len(names):
                    shown += " and %d more" % (len(edits) - len(names))
                versions = [str(e.get("version") or e.get("expectedVersion"))
                            for e in edits if isinstance(e, dict)]
                versions = [v for v in versions if v and v != "None"]
                suffix = (" at version%s %s" % ("" if len(set(versions)) == 1
                                                else "s", ", ".join(versions))
                          if versions else "")
                return f"Ready to edit {shown}{suffix} in the editor."
            return "Ready to apply an editor workspace edit."
    if tool in _PRODUCTIVITY_EFFECT_TOOLS:
        # F14 — the user hears the connector's own scrubbed preview of the
        # draft being sent/created, and that this needs their approval.
        preview = str(step.get("effect_preview") or "").strip()
        if preview:
            return "Ready to proceed after your approval: %s" % preview
        return "Ready to %s - %s." % (
            "send a message" if tool == "mail.send_draft"
            else "create a calendar event",
            step.get("reason") or "externally visible action")
    return f"{tool} - {step.get('reason') or 'planned action'}."


def _arm_plan_confirmation(plan, context, task_text=""):
    """Arm the ONE identified approval for *plan* (F02/F18).

    Shared by the legacy task path (``execute_plan``) and by the
    orchestrator's ``task.propose`` tool, so both routes ask about — and later
    run — the same plan through the same record.
    """
    global _pending_task_action
    # F02/F18: the plan's OWN command identity comes first — the approval is
    # verified against exactly this later, so arming with a different text
    # (the orchestrator's paraphrase, say) would invalidate its own record.
    command_text = plan.get("command_text") or task_text or ""
    record = approvals.arm(
        plan, command_text=command_text,
        scope="task", ttl=TASK_CONFIRM_WINDOW_SECONDS)
    with _task_confirm_lock:
        _pending_task_action = {
            "plan": plan,
            "context": context,
            "expires": record.expires_at,
            "approval_id": record.id,
            "plan_hash": record.plan_hash,
        }
    return record


def arm_task_confirmation(plan, context, task_text="", preview=""):
    """R4 public wrapper: arm the ONE identified approval for *plan*.

    Used when a qualified yes ("yes, but call it X") re-previews under a
    changed effect: the caller already cancelled the old approval, and this
    mints the new exact-effect record the next yes must match.
    """
    record = _arm_plan_confirmation(plan, context, task_text=task_text)
    with _task_confirm_lock:
        if _pending_task_action is not None:
            _pending_task_action["preview"] = preview
    return record


def confirmation_prompt(plan):
    """The byte-identical confirmation preview for *plan* (F02).

    Shared by ``execute_plan`` and the orchestrator so both routes ask the
    user about a plan in exactly the same words.
    """
    return (
        f"{_confirmation_preview(plan)} "
        "Say confirm task to proceed, or cancel."
    )


def register_proposal(plan, context=None, task_text=""):
    """F02: hand a plan to the confirmation gate from outside execute_plan.

    The orchestrator uses this so a proposed goal keeps its evidence and its
    identity: the armed record's id/hash is what the later "confirm" verifies
    against.

    Returns ``(record, reason)``:
      * ``(record, "")``            — armed; this is the plan to ask about;
      * ``(None, "nothing to confirm")`` — the plan needs no confirmation;
      * ``(None, "…already waiting")``   — a DIFFERENT plan is already armed
        for confirmation. Only one proposal may await consent at a time, so
        the armed one is never silently overwritten by a newer one.
    """
    if not isinstance(plan, dict) or not plan.get("ok"):
        return None, "the plan could not be built"
    if not plan.get("requires_confirmation"):
        return None, "nothing to confirm"
    global _pending_task_action
    # Same identity rule as _arm_plan_confirmation: the plan's own command
    # text first, so the hash computed here matches the hash that gets armed
    # and later verified.
    command_text = plan.get("command_text") or task_text or ""
    existing = approvals.pending()
    if existing is not None and existing.scope == "task":
        wanted = approvals.plan_hash(plan, command_text)
        if existing.plan_hash != wanted:
            return None, "another task is already waiting for confirmation"
        # Same plan proposed again: hand back the SAME identified record
        # rather than minting a second, indistinguishable approval.
        with _task_confirm_lock:
            pending = _pending_task_action
            if not pending or pending.get("approval_id") != existing.id:
                _pending_task_action = {
                    "plan": plan,
                    "context": context,
                    "expires": existing.expires_at,
                    "approval_id": existing.id,
                    "plan_hash": existing.plan_hash,
                }
        return existing, ""
    return _arm_plan_confirmation(plan, context, task_text=task_text), ""


#: F09: bounds on the verified run trace / verification the native engine
#: carries, matching the browser engine's trace contract.
_TRACE_STEPS_MAX = 40
_TRACE_OBSERVATION_MAX = 300
_VERIFICATION_MAX = 8
_TRACE_ARG_MAX = 12


def _trace_args(args):
    """One traced step's arguments, masked and field-capped."""
    safe = {}
    if not isinstance(args, dict):
        return safe
    for key, value in list(args.items())[:_TRACE_ARG_MAX]:
        if isinstance(value, (list, tuple)):
            safe[str(key)[:40]] = [str(v)[:200] for v in list(value)[:8]]
        else:
            safe[str(key)[:40]] = str(value)[:200]
    return safe


def _trace_step(trace, tool, args, observation, ok):
    """F09: record one EXECUTED step of the verified run trace."""
    try:
        if len(trace) >= _TRACE_STEPS_MAX:
            return
        trace.append({
            "tool": str(tool or "")[:60],
            "args": _trace_args(args),
            "observation": str(observation or "")[:_TRACE_OBSERVATION_MAX],
            "ok": bool(ok),
        })
    except Exception:
        pass


def _note_skill_replays(plan, result, verification):
    """F09: report this run's replay outcome for every offered skill.

    A run that used an offered procedure and finished VERIFIED complete is a
    successful replay; a run that failed — or did not verify completion — is
    reported with its real reason, so a changed target can invalidate the
    skill's applicability instead of it being retried forever.
    """
    offered = plan.get("memory_skills") if isinstance(plan, dict) else None
    if not offered or not plan.get("steps"):
        return
    try:
        from backend.core import memory_store
    except Exception:
        return
    completed = result.status == "completed"
    reasons = [str(item)[:200] for item in (result.evidence or [])]
    reason = str(result.error or (reasons[0] if reasons else "")
                 or result.status)
    for entry in offered:
        skill_id = entry.get("id") if isinstance(entry, dict) else entry
        if skill_id is None:
            continue
        try:
            if completed and verification:
                memory_store.note_skill_replay(
                    skill_id, True,
                    reason="verified replay: %s" % verification[0][:160],
                    observed=list(verification))
            else:
                memory_store.note_skill_replay(
                    skill_id, False, reason=reason,
                    observed=list(verification))
        except Exception:
            continue


def _finish_run(plan, result, trace, verification):
    """Attach the verified trace/verification and close the replay loop."""
    try:
        result.trace = list(trace)
        result.verification = list(verification)
    except Exception:
        pass
    _note_skill_replays(plan, result, verification)
    return result


def execute_plan(plan, context, task_text="", confirmed=False):
    """Run a normalized plan as a closed loop; returns a TaskResult.

    Callers that speak to the user (consume_task_confirmation,
    handle_task_message) wrap this with str(). Early paths keep today's
    spoken text byte-identical: plan-not-ok returns a plain failed result
    whose str() is exactly the plan response wording (default 'I could
    not plan that task.'), the unconfirmed gate returns the byte-identical
    confirmation preview (needs_input), and response-without-steps returns
    plan['response'].
    After steps run, the summary ALWAYS reflects actual outcomes — the
    pre-authored response override is gone.
    """
    if not isinstance(plan, dict) or not plan.get("ok"):
        message = (plan.get("response") if isinstance(plan, dict) else "") \
            or "I could not plan that task."
        return TaskResult.failed(message, plain=True)

    if plan.get("requires_confirmation") and not confirmed:
        # F18: consent is bound to the WHOLE plan through a shared approval
        # record — plan hash, every effect, every target, expiry and scope —
        # instead of a bare dict carrying only the first step's description.
        _arm_plan_confirmation(plan, context, task_text=task_text)
        return TaskResult.needs_input(confirmation_prompt(plan))

    if plan.get("response") and not plan.get("steps"):
        return TaskResult.completed(plan["response"], detail=plan["response"])

    steps = plan.get("steps", []) or []
    outcomes = []
    failed_folders = []  # (original path, normalized path) of failed mkdirs
    # F01 — OBSERVATION-DRIVEN STEP EXECUTION.
    #
    # The old loop ran a precomputed list: whatever the planner guessed in
    # advance was what ran, so a filename discovered by step 2 could never be
    # used by step 3, and a step whose prerequisite failed still ran. Here each
    # executed step publishes its STRUCTURED result into `observations`, which
    # every later step is re-resolved against before it runs.
    observations = {}
    #: Args that were covered by the user's approval, so a change introduced by
    #: observation resolution can be detected and re-approved instead of run.
    approved_args = [_canonical_step_args(step)
                     for step in steps if isinstance(step, dict)]
    unmet_goals = []
    #: F03 — how many steps produced something an observer could point at.
    #: A plan whose every step reported a silent "ok" proves no effect.
    observed_effects = 0
    #: F09 — the verified trace of what this run actually did (one entry per
    #: executed step) and the observations that verified the goal. Both travel
    #: on the TaskResult so a completed run becomes a PROCEDURE, not a slogan.
    trace = []
    verification = []

    for index, raw_step in enumerate(steps):
        step = raw_step if isinstance(raw_step, dict) else {}
        tool = step.get("tool") or ""
        args = step.get("args") if isinstance(step.get("args"), dict) else {}

        # F20: ask the job that OWNS this turn whether to keep going, before
        # every effect. A global flag could not tell two tasks apart, and a
        # cancelled turn must not run the next step (or speak its fragment).
        try:
            from backend.services import jobs as _jobs

            _jobs.checkpoint_turn("task step %d" % index)
        except Exception as exc:
            outcomes.append({
                "tool": tool,
                "status": "cancelled",
                "result": "",
                "reason": str(exc) or "cancelled",
                "goal": _goal_of(step),
                "fragment": "Stopped per your request.",
            })
            break

        # F01: a step that declared a dependency nobody can prove does not
        # run — an unverifiable prerequisite can never authorize an effect.
        unprovable = str(step.get("unprovable_dependency") or "").strip()
        if unprovable:
            outcomes.append({
                "tool": tool,
                "status": "skipped",
                "result": "",
                "reason": "unprovable dependency: %s" % unprovable,
                "goal": _goal_of(step),
                "fragment": "Skipped %s." % (_step_target(step) or tool or "the step"),
            })
            continue

        # DEPENDENCY SKIPPING: an explicit depends_on list, plus the
        # implicit folder->file rule (a write inside a folder whose
        # creation failed). Skipped steps never execute.
        skip_reason = ""
        depends_on = step.get("depends_on") or []
        if isinstance(depends_on, (list, tuple)):
            for dep in depends_on:
                if isinstance(dep, bool) or not isinstance(dep, int):
                    continue
                if 0 <= dep < len(outcomes) and outcomes[dep]["status"] != "ok":
                    skip_reason = "prerequisite failed: step %d" % dep
                    break
        if not skip_reason and tool == "code.write_file":
            write_path = args.get("path") or ""
            for original, _normalized in failed_folders:
                if write_path and _path_is_inside(write_path, original):
                    skip_reason = "prerequisite failed: folder %s" % original
                    break
        if skip_reason:
            outcomes.append({
                "tool": tool,
                "status": "skipped",
                "result": "",
                "reason": skip_reason,
                "goal": _goal_of(step),
                "fragment": "Skipped %s." % (_step_target(step) or tool or "the step"),
            })
            continue

        # F01: resolve arguments from what EARLIER STEPS ACTUALLY OBSERVED
        # (discovered paths, filenames, search hits) before this step runs.
        # A precomputed list could not do this: the planner's guess was final.
        if observations:
            resolved = _resolve_step_args(args, observations)
            if resolved != args:
                step = dict(step)
                step["args"] = resolved
                args = resolved
                # F01/reapproval: a resolved effect is a CHANGED effect. When
                # the user approved a specific effect set, consent does not
                # carry over to different arguments — re-arm the gate with the
                # real values. A plan that needed no confirmation has no
                # approved arguments to invalidate, so it proceeds.
                if confirmed and plan.get("requires_confirmation") \
                        and index < len(approved_args) \
                        and _canonical_step_args(step) != approved_args[index]:
                    updated = dict(plan)
                    updated["steps"] = list(steps)
                    updated["steps"][index] = step
                    _arm_plan_confirmation(updated, context, task_text=task_text)
                    return TaskResult.needs_input(
                        "The task reached a step whose details changed since "
                        "you approved it (%s). %s"
                        % (_step_target(step) or tool or "a step",
                           confirmation_prompt(updated)))

        # DETERMINISTIC INVALID-PATH REJECTION (paths only): a resolved
        # path that still carries wildcard/redirection characters never
        # executes - honest failure instead of a platform error.
        if tool in _PATH_ARG_TOOLS:
            suspect = args.get("path") if isinstance(args.get("path"), str) else ""
            if suspect and _path_has_invalid_chars(suspect):
                outcomes.append({
                    "tool": tool,
                    "status": "failed",
                    "result": "",
                    "reason": "invalid characters in path",
                    "fragment": _fail_fragment(step, "invalid characters in path"),
                })
                continue

        text, structured = _execute_step_structured(step, context)
        failed_prefix = f"{tool} failed:" if tool else "failed:"
        reason = ""
        failed = False
        if structured is not None and not structured.get("ok"):
            reason = str(structured.get("error") or "the tool reported an error")
            failed = True
        elif (text or "").startswith(failed_prefix):
            reason = text[len(failed_prefix):].strip() or "the tool reported an error"
            failed = True
        elif (text or "").startswith("Unknown task tool:"):
            # Nothing ran: honest failure, never a silent ok.
            reason = "unknown tool"
            failed = True

        # DETERMINISTIC POSTCONDITIONS (code tools only): a reported ok
        # for write_file/create_folder/apply_patch must leave the path
        # behind; for run_command/run_script the ok/exit_code already is
        # the verdict.
        if not failed and structured is not None and structured.get("ok") \
                and tool in ("code.write_file", "code.create_folder",
                             # F11: a patch is a write — it must leave the
                             # patched file behind (dry_run writes nothing,
                             # so the file simply still exists).
                             "code.apply_patch"):
            resolved = structured.get("path") or args.get("path") or ""
            if resolved and not os.path.exists(resolved):
                failed = True
                reason = "postcondition failed: %s missing" % resolved

        if failed:
            if tool == "code.create_folder":
                folder_path = args.get("path") or ""
                if folder_path:
                    failed_folders.append(
                        (folder_path, _normalize_fs_path(folder_path)))
            outcomes.append({
                "tool": tool,
                "status": "failed",
                "result": text or "",
                "reason": reason,
                "goal": _goal_of(step),
                "fragment": _fail_fragment(step, reason),
            })
            _record_observation(observations, index, step, structured, text,
                                "failed")
            # F09: a step that reported its failure is traced as a failure.
            _trace_step(trace, tool, args, reason or text, False)
            continue

        outcomes.append({
            "tool": tool,
            "status": "ok",
            "result": text or "",
            "reason": "",
            "goal": _goal_of(step),
            "fragment": _ok_fragment(step, text),
        })
        # F09: every executed step joins the verified trace; an ok step that
        # left something an observer can point at is VERIFICATION.
        _trace_step(trace, tool, args, text, True)
        if _step_observed_effect(text, structured):
            observed_effects += 1
            if len(verification) < _VERIFICATION_MAX:
                observation = (str(text or "").strip()
                               or str((structured or {}).get("effect") or ""))
                if observation:
                    verification.append(
                        "%s: %s" % (tool or "step",
                                    observation[:_TRACE_OBSERVATION_MAX]))
        _record_observation(observations, index, step, structured, text, "ok")

    detail = "\n".join(
        "[%s] %s" % (item["tool"] or "step", item["result"])
        for item in outcomes if item["result"]
    )
    summary = _summarize_outcomes(outcomes, plan)
    bad = [item for item in outcomes if item["status"] != "ok"]
    # F01: goals the step budget (or a cancellation) never admitted are UNMET.
    # Executing 8 of an intended 10 must never be published as completion.
    for step in (plan.get("omitted_steps") or []):
        unmet_goals.append(_goal_of(step))
    for item in bad:
        unmet_goals.append(
            item.get("goal") or item["reason"] or item["fragment"])
    unmet_goals = [goal for goal in _dedupe(unmet_goals) if goal]
    artifacts = _task_artifacts(plan, observations)
    evidence = [
        "%s: %s: %s" % (
            item["tool"] or "step", item["status"],
            item["reason"] or item["fragment"],
        )
        for item in bad
    ]
    if plan.get("truncated"):
        evidence.append(
            "%d of %d intended steps ran (budget %d)"
            % (len(outcomes), int(plan.get("intended_steps") or 0),
               TASK_MAX_STEPS))
    if unmet_goals:
        evidence.append("unmet goals: " + "; ".join(unmet_goals))
    if not bad and not unmet_goals and steps and observed_effects == 0:
        # F03: every step answered, and not one of them left anything an
        # observer could point at — a zero-effect run cannot be verified
        # completion, even though no step reported a failure.
        return _finish_run(plan, TaskResult.partial(
            summary, detail=detail,
            evidence=["no step produced an observable effect"],
            unmet_goals=["produce a verifiable result"]), trace, verification)
    if not bad and not unmet_goals:
        return _finish_run(
            plan, TaskResult.completed(summary, detail=detail), trace,
            verification)
    if unmet_goals:
        summary = _with_unmet_note(summary, unmet_goals)
    result = TaskResult.partial(summary, detail=detail, evidence=evidence)
    result.artifacts = artifacts
    return _finish_run(plan, result, trace, verification)


#: F03: text that records a successful command which did NOTHING. A zero-exit
#: no-op is not an observed effect, so a plan made only of these is partial.
NO_OUTPUT_MARKERS = frozenset((
    "(no output)", "(error, no output)", "no output", "(empty)",
))


def _step_observed_effect(text, structured):
    """F03: did this step leave anything an observer could point at?"""
    body = (text or "").strip()
    if body and body.lower() not in NO_OUTPUT_MARKERS:
        return True
    if not isinstance(structured, dict):
        return False
    if structured.get("verified_effect") or structured.get("effect"):
        return True
    content = str(structured.get("content") or "").strip()
    if content and content.lower() not in NO_OUTPUT_MARKERS:
        return True
    for key in ("path", "file", "files", "paths", "artifacts", "matches",
                "entries", "items", "results"):
        value = structured.get(key)
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, (list, tuple, dict)) and value:
            return True
    return False


def _dedupe(items):
    """Order-preserving dedupe (goals must stay readable in plan order)."""
    seen = set()
    out = []
    for item in items:
        key = str(item)
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _with_unmet_note(summary, unmet_goals):
    """Say plainly that part of the request did not run (F01).

    "Done, sir." is the one headline a cut-short plan must never keep: every
    step that ran succeeded, but the request itself is unfinished.
    """
    count = len(unmet_goals)
    note = "%d step%s could not be completed." % (
        count, "" if count == 1 else "s")
    summary = summary or ""
    if summary.startswith("Done, sir."):
        summary = "Partly done, sir." + summary[len("Done, sir."):]
    if "could not be completed" in summary:
        return summary
    return re.sub(r"\s+", " ", "%s %s" % (summary, note)).strip()


def _goal_of(step):
    """The human goal of one step, used for unmet-goal reporting."""
    if not isinstance(step, dict):
        return ""
    reason = str(step.get("reason") or "").strip()
    target = _step_target(step)
    if target and reason:
        return "%s (%s)" % (target, reason)
    return target or reason or str(step.get("tool") or "").strip()


def _canonical_step_args(step):
    """Stable signature of a step's EFFECT (tool + args) for re-approval."""
    if not isinstance(step, dict):
        return ""
    try:
        return "%s|%s" % (step.get("tool") or "",
                          json.dumps(step.get("args") or {},
                                     sort_keys=True, default=str))
    except Exception:
        return "%s|%r" % (step.get("tool") or "", step.get("args"))


def _record_observation(observations, index, step, structured, text, status):
    """Publish one step's STRUCTURED result for later steps to plan against.

    F11 — this is the structured-preserving path that runs alongside the
    clipped spoken rendering: every field the tool returned (``path``,
    ``content_hash``, ``lines``, ``matches``, ``next_cursor``/``next_start``,
    ``diff``, ...) is recorded untouched, while ``text`` is only the bounded
    human rendering. ``{{step0.lines.0.text}}`` and ``{{step0.next_cursor}}``
    therefore resolve against real artifacts, not against a clip.
    """
    payload = {
        "index": index,
        "tool": (step or {}).get("tool") or "",
        "status": status,
        "text": text or "",
    }
    if isinstance(structured, dict):
        # `ok` is bookkeeping, not an observation: keep the payload fields.
        payload.update({k: v for k, v in structured.items() if k != "ok"})
    payload.setdefault("path", ((step or {}).get("args") or {}).get("path"))
    observations[index] = payload


#: Keys a later step may reference from an earlier step's observation, either
#: as {{step0.key}} or {{last.key}}.
_OBSERVATION_PLACEHOLDER_RE = re.compile(
    r"\{\{\s*(step\s*(?P<index>\d+)|last)\s*(?:\.\s*(?P<key>[\w.\[\]-]+))?\s*\}\}")


#: F12 — the ONLY argument fields where a placeholder may be substituted
#: inside a larger value. These designate a path/file/document; every other
#: field (a command line, a script body, a quoted literal, free text) is
#: payload whose bytes must survive exactly as they were spoken or planned.
_PLACEHOLDER_PATH_FIELDS = frozenset((
    "path", "paths", "file", "files", "filepath", "file_path", "filename",
    "filenames", "folder", "folders", "directory", "directory_path", "dir",
    "uri", "uris", "artifact", "artifacts", "source", "source_path", "src",
    "destination", "destination_path", "dest", "target", "target_path",
    "output", "output_path", "out_path", "input", "input_path", "cwd",
    "old_path", "new_path", "expected_path", "workspace_folder",
))

#: A value that IS one placeholder reference, with nothing else around it:
#: an explicit template reference (the F01 convention), never literal payload.
_FULL_PLACEHOLDER_RE = re.compile(
    r"^\s*\{\{\s*(?:step\s*\d+|last)\s*(?:\.[\w.\[\]-]+)?\s*\}\}\s*$")


def _argument_accepts_placeholders(field, value):
    """F12 — may this argument's placeholders be substituted?

    Two cases are templates, everything else is payload:

    * a designated path/file field, where a placeholder may be embedded in a
      larger value ("{{step0.dir}}\\report.txt");
    * any field whose ENTIRE value is a single placeholder reference.

    A command line, script body, quoted literal or free-text argument that
    merely CONTAINS a placeholder is never rewritten — the replacement used to
    mangle quoted code and paths that were never templates.
    """
    if not isinstance(value, str) or not _OBSERVATION_PLACEHOLDER_RE.search(value):
        return False
    if str(field or "").strip().lower() in _PLACEHOLDER_PATH_FIELDS:
        return True
    return bool(_FULL_PLACEHOLDER_RE.match(value))


def _resolve_step_args(args, observations, field=None):
    """Fill step arguments from what previous steps actually observed (F01).

    ``{{step0.path}}`` becomes the path step 0 reported; ``{{last.files.0}}``
    becomes the first discovered filename. Unresolvable placeholders are left
    untouched so the step fails honestly instead of running with a guess.

    F12 — substitution is restricted to designated path/file fields (and to
    values that are exactly one placeholder reference). A command line, script
    body, quoted literal or free text keeps every byte it was given, so a
    payload like ``patch "Q4 Report.TXT" --Dry-Run`` can never be rewritten by
    an observation from an earlier step.
    """
    if isinstance(args, dict):
        return {key: _resolve_step_args(value, observations, key)
                for key, value in args.items()}
    if isinstance(args, list):
        return [_resolve_step_args(value, observations, field)
                for value in args]
    if not isinstance(args, str):
        return args
    if not _argument_accepts_placeholders(field, args):
        return args
    return _OBSERVATION_PLACEHOLDER_RE.sub(
        lambda match: _lookup_observation(match, observations), args)


def _lookup_observation(match, observations):
    index_token = match.group("index")
    if index_token:
        index = int(re.sub(r"\D", "", index_token) or -1)
    else:
        index = max(observations) if observations else -1
    payload = observations.get(index)
    if payload is None:
        return match.group(0)
    key = match.group("key")
    if not key:
        for candidate in ("path", "text", "result"):
            value = payload.get(candidate)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return match.group(0)
    value = payload
    for part in key.split("."):
        if isinstance(value, dict):
            if part not in value:
                return match.group(0)
            value = value[part]
        elif isinstance(value, (list, tuple)):
            try:
                value = value[int(part)]
            except (ValueError, IndexError):
                return match.group(0)
        else:
            return match.group(0)
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if value is None:
        return match.group(0)
    return str(value)


def _task_artifacts(plan, observations):
    """Real artifacts produced by this run, kept with the result (F01)."""
    artifacts = []
    for index in sorted(observations):
        payload = observations[index]
        for key in ("path", "file", "artifact"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                artifacts.append({"step": index, "path": value.strip()})
                break
        for key in ("paths", "files", "artifacts"):
            value = payload.get(key)
            if isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        artifacts.append({"step": index, "path": item.strip()})
                    elif isinstance(item, dict) and item.get("path"):
                        artifacts.append({"step": index,
                                          "path": str(item["path"])})
    return artifacts


def has_pending_task_confirmation():
    """True while a task-action confirmation is armed and still in window."""
    with _task_confirm_lock:
        pending = _pending_task_action
        if pending is None:
            return False
    # F18: the shared record owns expiry, so both gates agree on the deadline.
    record = approvals.pending()
    return record is not None and record.id == pending.get("approval_id")


def consume_task_confirmation(answer):
    """Resolve a pending task-action confirmation.

    R4: the WHOLE sentence decides. A clean "yes"/"yes please" runs the exact
    previewed effect. A renamed yes ("yes, but call it X") re-previews under
    the new name — the old yes never authorizes the changed plan. An
    inspection tail ("yes, a quick look") HOLDS the write and asks once
    ("create, or only check?"). An effect-negating tail ("yes, don't create
    it") declines. An extra action ("yes, and also delete...") runs the
    approved effect only; the extra needs its own turn.

    Returns a response string when the answer resolves the gate, or None
    when nothing is pending or the window expired (pending is discarded in
    that case so the message falls through to normal chat).

    F18: a confirmation cannot cross-authorize. The shared approval record is
    consumed FIRST (so it can never be used twice), then verified against the
    plan that is about to run: a changed plan, an expired window or a record
    armed for a different request executes nothing and asks again.
    """
    global _pending_task_action
    with _task_confirm_lock:
        pending = _pending_task_action
        if pending is None:
            return None
        # F09/F07: this turn is about to speak a string; whatever the outcome,
        # it must not inherit the previous run's structured result.
        _remember_task_result(None)
        if time.time() >= pending["expires"]:
            _pending_task_action = None
            approvals.cancel("task confirmation window expired")
            return None
        if not answer or not answer.strip():
            _pending_task_action = None
            approvals.cancel("task confirmation abandoned")
            return None
        verdict = classify_confirmation(answer)
        plan = pending["plan"]
        context = pending["context"]
        # Clear BEFORE executing so a crash can never re-trigger the gate.
        _pending_task_action = None

    if verdict == "no":
        # F18: an explicit negative is authoritative — it consumes the record
        # without authorising anything, and no later model turn can revive it.
        approvals.cancel("declined by the user")
        return "As you wish, sir. I will skip that."

    if verdict == "unclear":
        _pending_task_action = None
        approvals.cancel("task confirmation not understood")
        return None

    if verdict == "inspect":
        # Astra Trace A: "Yes, a quick look" against a CREATION preview is
        # ambiguous — never run the write, never infer a browser task. Hold
        # the write, keep ONE clarification, and let a clear "create" answer
        # both disambiguate and approve the exact displayed effect.
        approvals.cancel("held for inspection-vs-creation disambiguation")
        preview = pending.get("preview") or confirmation_prompt(plan) or ""
        return ("Create the file, sir, or only check the folder? "
                "%s" % preview if preview else
                "Create the file, sir, or only check the folder?")

    if verdict.startswith("rename:"):
        new_name = verdict[len("rename:"):].strip()
        approvals.cancel("renamed after preview — new preview required")
        if not new_name:
            return ("Sir, what name should I use instead? "
                    "Nothing was started.")
        # Re-plan under the new name so the user approves the EXACT changed
        # effect; the old yes authorizes nothing.
        return _repreview_with_name(plan, context, new_name)

    if verdict == "extra":
        # The approved effect may run; the EXTRA action never inherits the
        # yes — it needs its own turn. Fall through to normal execution of
        # the exact previewed plan, then name the leftover.
        extra_note = (" Sir, I only did what was previewed — "
                      "please ask the extra part separately.")
    else:
        extra_note = ""

    # verdict == "yes": consume, then verify: consumption is atomic, and a
    # failure here means NOTHING runs (not a partially authorised plan).
    record = approvals.take()
    if record is None:
        return ("That approval is no longer available, sir. Please ask again.")
    if record.id != pending.get("approval_id"):
        return ("That approval was replaced, sir. Please ask again.")
    command_text = plan.get("command_text") or record.command or ""
    ok, why = approvals.verify(plan, record, command_text=command_text)
    if not ok:
        _remember_task_result(None)
        return ("I cannot proceed, sir — %s. Please ask again." % why)
    result = execute_plan(plan, context, confirmed=True)
    _remember_task_result(result, command_text or plan.get("summary") or "")
    return str(result) + extra_note


def _repreview_with_name(plan, context, new_name):
    """R4: re-preview a write plan under a renamed file, nothing executed.

    Swaps the file name inside the pending plan's write/create steps, re-arms
    the confirmation gate, and returns the new exact preview. The caller
    already cancelled the old approval, so the old yes is dead.
    """
    from copy import deepcopy
    plan = deepcopy(plan)
    steps = plan.get("steps") or []
    changed = False
    for step in steps:
        tool = str(step.get("tool") or "")
        if tool not in ("code.write_file", "code.create_folder"):
            continue
        args = dict(step.get("args") or {})
        path = str(args.get("path") or "")
        if not path:
            continue
        parent = os.path.dirname(path)
        safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", new_name).strip()
        if not safe:
            continue
        if tool == "code.write_file" and not os.path.splitext(safe)[1]:
            safe += ".txt"
        args["path"] = os.path.join(parent, safe) if parent else safe
        step["args"] = args
        changed = True
    if not changed:
        return ("Sir, I could not apply that name to the preview. "
                "Nothing was started.")
    preview = confirmation_prompt(plan) or ""
    arm_task_confirmation(plan, context, preview=preview)
    return preview


#: F09/F07 — the structured result of the most recent native run, so the
#: caller (brain) can persist a VERIFIED trace instead of only the spoken
#: string. Guarded because tasks can run on more than one thread.
_last_task_result = None
_last_task_result_lock = threading.Lock()


def _remember_task_result(result, task_text=""):
    """Keep the structured terminal result of the run that just finished."""
    global _last_task_result
    try:
        with _last_task_result_lock:
            _last_task_result = (result, str(task_text or ""))
    except Exception:
        pass


def last_task_result():
    """(TaskResult, task text) of the most recent native run, or (None, "")."""
    with _last_task_result_lock:
        if _last_task_result is None:
            return None, ""
        return _last_task_result


def handle_task_message(text, voice_compact=False):
    command = _strip_task_prefix(text)
    # A fresh run must not look like the previous one's result.
    _remember_task_result(None)
    context = gather_context()
    plan = plan_task(command, context)
    result = execute_plan(plan, context, task_text=command)
    # F09/F07: the STRUCTURED terminal result must survive the string
    # conversion below — a verified trace is what lets a completed run become a
    # skill, and what the work log persists.
    _remember_task_result(result, command)
    response = str(result)
    if voice_compact and len(response) > 180:
        kept = response[:177].rstrip()
        # Prefer a clean sentence: cut at the last terminator if it is at
        # least 100 chars in (no ellipsis needed). Otherwise cut at the last
        # word boundary and append "...". Never exceed 180 or end mid-word.
        term_idx = max(kept.rfind(t) for t in (".", "!", "?"))
        if term_idx >= 100:
            return kept[: term_idx + 1]
        word_idx = kept.rfind(" ")
        if word_idx > 0:
            kept = kept[:word_idx].rstrip()
        return kept + "..."
    return response
