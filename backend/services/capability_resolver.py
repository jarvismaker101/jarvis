"""Dispatch by capability, not one global engine (Fable-5 audit G8: F16).

CURRENT (audit F16): ``brain.py::_execute_deferred_opencode`` chooses every
deferred task using ``config.TASK_ENGINE``; ``handle_tool_intent`` checks
``is_opencode_available()`` even before a handoff that may use the browser
engine — so a complex local task can be handed to a browser-only agent, and
browser recovery depends on an unrelated CLI installation.

CHANGE: a capability resolver selects native code tools, editor tools,
browser tools, or an EXPLICITLY enabled coding agent. Engine AVAILABILITY is
separated from the PERMISSION to use that engine. The opencode-installation
prerequisite for browser recovery is removed. The current opt-in boundary is
preserved: opencode is never started merely because another engine failed.

Audit F16 (this revision) adds four failure modes the first implementation
still had:

* **negated opt-in** — "don't use opencode" matched the positive opt-in
  spelling, so a refusal could ENABLE the coding agent;
* **local coding → browser work** — a local refactor fell through to the
  configured default and became browser-agent work;
* **incompatible fallback** — an explicit coding-agent request with the CLI
  missing silently became browser work instead of failing closed;
* **decision ignored by execution** — the resolver's answer was logged but
  execution re-read live configuration. :func:`begin_dispatch` freezes the
  decision into an immutable
  :class:`~backend.services.capability_contract.ExecutionContract` that the
  executor verifies, so a configuration change after consent cannot change
  the executor.

Pure decision module: stdlib only. Availability facts are passed IN
(booleans) so this stays free of connector imports and trivially testable.
"""

import re

from backend.services import capability_contract

# ── Engines (ordered by increasing authority) ──────────────────────────────
CODE_TOOLS = "code_tools"          # deterministic file/command/script tools
EDITOR_TOOLS = "editor_tools"      # structured editor-bridge operations
BROWSER_TOOLS = "browser_tools"    # executor steps (open/search/play) — the tool intent
BROWSER_AGENT = "browser_agent"    # brave-control MCP agent (web-shaped multi-step)
OPENCODE = "opencode"              # local coding CLI — OPT-IN ONLY, never a fallback
BLOCKED = capability_contract.BLOCKED   # fail closed: no compatible executor

# ── Required capabilities (what the request NEEDS, not which engine wins) ──
CAPABILITY_CODING = "coding"
CAPABILITY_EDITOR = "editor"
CAPABILITY_BROWSER = "browser"
CAPABILITY_GENERIC = "generic"

#: Which executors can actually deliver which capability. A fallback that is
#: not listed here is INCOMPATIBLE and must fail closed (F16) — this is what
#: stops a local refactor from being handed to a browser-only engine.
COMPATIBLE_EXECUTORS = {
    CAPABILITY_CODING: (CODE_TOOLS, EDITOR_TOOLS, OPENCODE),
    CAPABILITY_EDITOR: (EDITOR_TOOLS, CODE_TOOLS, OPENCODE),
    CAPABILITY_BROWSER: (BROWSER_AGENT, BROWSER_TOOLS, OPENCODE),
    CAPABILITY_GENERIC: (CODE_TOOLS, EDITOR_TOOLS, BROWSER_AGENT,
                         BROWSER_TOOLS, OPENCODE),
}

# Explicit opt-in spellings for the coding agent (mirrors task_agent's
# TASK_PREFIX_RE boundary: plain natural commands must NOT reach it).
_OPENCODE_OPT_IN_RE = re.compile(
    r"\b(opencode|coding agent|use agent|agent mode|take over|execute task|use task brain|task brain)\b",
    re.IGNORECASE,
)

# Negation-aware opt-in (F16): a negator in the run-up to the opt-in phrase
# turns "use opencode" into "do NOT use opencode". A negator INSIDE the
# matched phrase is part of its own vocabulary ("no" in "no agent mode" is
# the run-up, "take over" carries none), so only the run-up is inspected.
_NEGATION_RE = re.compile(
    r"\b(no|not|never|without|dont|doesnt|wont|cant|avoid|skip|instead|"
    r"rather|nahi|nahin|mat|stop|quit)\b",
    re.IGNORECASE,
)
#: Clause boundaries: a negator anywhere in the PHRASE'S OWN CLAUSE negates
#: it ("do not take over with the coding agent" negates both spellings),
#: while a negator in an earlier clause does not ("don't stop; use opencode"
#: still requests opencode).
_CLAUSE_BOUNDARY_RE = re.compile(
    r"[,;.!?]|\b(?:and|but|then|so|also|however|plus)\b", re.IGNORECASE)

# Web/browser targets belong to the browser agent, not the local CLI.
_WEB_HINT_RE = re.compile(
    r"\b(browser|website|webpage|web|online|internet|\.com\b|\.org\b|"
    r"youtube|gmail|chrome|edge|brave|tab)\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"\bhttp\b|https?://", re.IGNORECASE)

# Deterministic local file/command shapes the native code tools handle
# without any LLM (path-like target or a known command token; the same
# conservative boundary as task_agent.is_code_tool_request).
_COMMAND_TOKENS = frozenset((
    "pip", "python", "py", "npm", "node", "npx", "git", "docker", "dir",
    "echo", "cd", "ls", "where", "tasklist", "ping", "curl", "java",
    "mvn", "gradle", "python3",
))
_PATH_LIKE_RE = re.compile(r"^[a-z0-9_ .\-]+\.(?:[a-z0-9]{1,8})$", re.IGNORECASE)
_VERB_RE = re.compile(
    r"^(read|cat|open file|write|create|run|execute|install|search|find|debug|edit|test)\b",
    re.IGNORECASE,
)

# Editor/IDE entities are handled by the structured editor bridge.
_EDITOR_HINT_RE = re.compile(
    r"\b(editor|vs ?code|vscode|cursor|windsurf|antigravity|unsaved|selection|diagnostics)\b",
    re.IGNORECASE,
)

# Local coding/refactoring shapes (F16). These REQUIRE a coding capability:
# they must never be delegated to the browser agent, and when no coding
# executor is available the dispatch fails closed instead of substituting an
# incompatible engine.
_CODE_VERB_RE = re.compile(
    r"\b(refactor|rename|extract|inline|reimplement|implement|debug|patch|"
    r"lint|format|type-?check|compile|optimize|optimi[sz]e|clean ?up|review)\b",
    re.IGNORECASE,
)
#: Verbs that are local-coding on their own (no explicit source entity needed).
_LOCAL_ONLY_VERB_RE = re.compile(
    r"\b(refactor|rename|lint|format|type-?check|compile|unit tests?|"
    r"test suite|clean ?up the code|fix the build|fix the bug|fix this bug|"
    r"add a (unit )?test|write a (unit )?test|fix the (type|import) error)\b",
    re.IGNORECASE,
)
#: Source-code entities/paths that make a generic code verb a LOCAL task.
_LOCAL_SOURCE_RE = re.compile(
    r"\.(?:py|js|ts|tsx|jsx|java|cs|cpp|c|h|go|rs|rb|php|sql)\b|"
    r"\b(function|method|class|module|package|import|docstring|repo|"
    r"repository|codebase|source file|unit test|test file|script)\b",
    re.IGNORECASE,
)


def _normalized(command):
    return " ".join(str(command or "").strip().lower().split())


def _opt_in_requested(normalized):
    """True when the coding agent was POSITIVELY requested.

    F16 acceptance: a negated opencode mention ("don't use opencode",
    "without opencode", "no coding agent", "do not take over with the coding
    agent") must NEVER enable it. Every match of an opt-in spelling is
    inspected; the request counts as opt-in only when at least one match has
    no negator in its own clause's run-up.
    """
    for match in _OPENCODE_OPT_IN_RE.finditer(normalized):
        run_up = _clause_run_up(normalized, match.start())
        # Apostrophes are dropped so "don't" and "dont" negate identically.
        if not _NEGATION_RE.search(run_up.replace("'", "")):
            return True
    return False


def _clause_run_up(normalized, index):
    """The text from the start of the match's clause up to the match."""
    preceding = normalized[:index]
    last = None
    for boundary in _CLAUSE_BOUNDARY_RE.finditer(preceding):
        last = boundary
    return preceding[last.end():] if last is not None else preceding


def _looks_like_local_file_op(command):
    """Conservative: verb + path-like target / known command token."""
    normalized = _normalized(command)
    if not normalized:
        return False
    if _WEB_HINT_RE.search(normalized) or _URL_RE.search(normalized):
        return False
    if not _VERB_RE.match(normalized):
        return False
    target = _VERB_RE.sub("", normalized, count=1).strip()
    if "\\" in target or "/" in target:
        return True
    if any(target == token or target.startswith(token + " ") for token in _COMMAND_TOKENS):
        return True
    return bool(_PATH_LIKE_RE.match(target))


def _looks_like_editor_op(command):
    return bool(_EDITOR_HINT_RE.search(_normalized(command)))


def _looks_like_local_coding(command):
    """True for local refactoring / code work with no web target (F16).

    Such a request needs a CODING capability. It must never be delegated to
    the browser agent, and it must fail closed when no coding executor is
    available.
    """
    normalized = _normalized(command)
    if not normalized:
        return False
    if _WEB_HINT_RE.search(normalized) or _URL_RE.search(normalized):
        return False
    if _LOCAL_ONLY_VERB_RE.search(normalized):
        return True
    return bool(_CODE_VERB_RE.search(normalized)
                and _LOCAL_SOURCE_RE.search(normalized))


def _availability(availability):
    facts = dict(availability or {})
    # Native code tools are in-process and always available unless a caller
    # explicitly declares them unavailable (tests, degraded modes).
    facts.setdefault("code_tools", True)
    return facts


def _decision(engine, capability, reason, opt_in, availability, blocked=False):
    compatible = engine in COMPATIBLE_EXECUTORS.get(capability, ())
    return {
        "engine": engine,
        "reason": reason,
        "opt_in": bool(opt_in),
        "capability": capability,
        "availability": dict(availability),
        "compatible": bool(compatible) and not blocked,
        "blocked": bool(blocked),
    }


def _blocked(capability, reason, availability, opt_in=False):
    """Fail closed: no compatible executor exists for the capability."""
    return _decision(BLOCKED, capability, reason, opt_in, availability,
                     blocked=True)


def resolve_engine(command, context=None, availability=None, task_engine=None):
    """Select the engine for one deferred request.

    Returns ``{"engine", "reason", "opt_in", "capability", "availability",
    "compatible", "blocked"}``.

    AVAILABILITY vs PERMISSION (audit F16):
      * a declared-but-unavailable engine is skipped with its reason;
      * opencode is chosen ONLY on explicit (non-negated) opt-in AND
        availability — never because another engine failed;
      * browser recovery NEVER requires opencode availability;
      * a request whose capability cannot be served by the fallback FAILS
        CLOSED (:data:`BLOCKED`) — an incompatible engine is never a fallback.
    """
    availability = _availability(availability)
    normalized = _normalized(command)
    opt_in = _opt_in_requested(normalized)

    # 1) Deterministic native code tools — no LLM, no engine handoff.
    if _looks_like_local_file_op(command):
        if availability.get("code_tools"):
            return _decision(
                CODE_TOOLS, CAPABILITY_CODING,
                "deterministic file/command operation", False, availability)
        return _blocked(CAPABILITY_CODING,
                        "deterministic code tools are unavailable",
                        availability)

    # 2) Structured editor operations when the bridge is connected.
    if _looks_like_editor_op(command) and availability.get("editor"):
        return _decision(EDITOR_TOOLS, CAPABILITY_EDITOR,
                         "editor bridge connected", False, availability)

    # 3) Local coding/refactoring work: a CODING capability. Explicit
    #    opencode opt-in is honoured when available; otherwise a local
    #    coding executor (editor bridge, native tools) serves it. The
    #    browser engine is NOT a coding executor, so this can never become
    #    browser work — when no coding executor exists the dispatch fails
    #    closed.
    if _looks_like_local_coding(command):
        if opt_in:
            if availability.get("opencode"):
                return _decision(OPENCODE, CAPABILITY_CODING,
                                 "explicitly requested and available", True,
                                 availability)
            return _blocked(
                CAPABILITY_CODING,
                "coding agent explicitly requested but unavailable; refusing "
                "an incompatible browser fallback", availability, opt_in=True)
        if availability.get("editor"):
            return _decision(EDITOR_TOOLS, CAPABILITY_EDITOR,
                             "local coding work served by the editor bridge",
                             False, availability)
        if availability.get("code_tools"):
            return _decision(CODE_TOOLS, CAPABILITY_CODING,
                             "local coding work", False, availability)
        return _blocked(CAPABILITY_CODING,
                        "local coding work with no compatible local executor",
                        availability)

    # 4) Explicit coding-agent opt-in: opencode when BOTH requested and
    #    available. The browser agent is a DIFFERENT capability, so a
    #    missing CLI fails closed instead of silently becoming browser work.
    if opt_in:
        if availability.get("opencode"):
            return _decision(OPENCODE, CAPABILITY_CODING,
                             "explicitly requested and available", True,
                             availability)
        return _blocked(
            CAPABILITY_CODING,
            "coding agent requested but opencode unavailable; refusing an "
            "incompatible browser fallback", availability, opt_in=True)

    # 5) Web-shaped / browser multi-step work → the brave-control agent.
    if _WEB_HINT_RE.search(normalized) or _URL_RE.search(normalized):
        return _decision(BROWSER_AGENT, CAPABILITY_BROWSER,
                         "web/browser target", False, availability)

    # 6) Nothing matched a specific capability: defer to the configured
    #    default handoff engine (config.TASK_ENGINE), without touching the
    #    opt-in boundary. The capability is GENERIC, so the browser agent is
    #    a compatible fallback here (unlike for coding work in step 3).
    default = BROWSER_AGENT if (task_engine or "browser_agent") == "browser_agent" else OPENCODE
    if default == OPENCODE:
        if availability.get("opencode"):
            return _decision(OPENCODE, CAPABILITY_GENERIC,
                             "configured default coding engine", False,
                             availability)
        return _decision(BROWSER_AGENT, CAPABILITY_GENERIC,
                         "configured engine unavailable; browser agent recovery",
                         False, availability)
    return _decision(BROWSER_AGENT, CAPABILITY_GENERIC,
                     "configured default handoff engine", False, availability)


def begin_dispatch(command, availability=None, task_engine=None, grant=""):
    """Resolve AND freeze the decision into an immutable contract (F16).

    Call this at the moment consent is requested. The returned
    :class:`~backend.services.capability_contract.ExecutionContract` carries
    the required capability, the selected executor, the availability facts
    and (once granted) the explicit user grant. Execution must verify THIS
    contract rather than re-reading ``config.TASK_ENGINE``, so a
    configuration change after consent cannot change the executor.
    """
    decision = resolve_engine(command, availability=availability,
                              task_engine=task_engine)
    return capability_contract.contract_for(decision, command, grant=grant)


def recovery_after_failure(command, availability=None, task_engine=None):
    """Engine choice after local executor steps FAILED.

    Audit F16: the opencode-installation prerequisite for browser recovery
    is REMOVED — when the configured engine is the browser agent, recovery
    is armed even when the CLI is not installed. opencode itself stays
    opt-in: it is only chosen when the user explicitly asked for it (and it
    is available). Availability never escalates permission, and recovery
    NEVER substitutes an incompatible engine: a coding-capability decision
    (opencode or fail-closed) is returned unchanged.
    """
    decision = resolve_engine(command, availability=availability,
                              task_engine=task_engine)
    if decision["engine"] in (OPENCODE, BLOCKED):
        return decision
    recovered = dict(decision)
    recovered["reason"] = (decision["reason"]
                           + " (recovery after local execution failed)")
    return recovered
