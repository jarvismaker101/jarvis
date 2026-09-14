"""Dispatch-bound tool policy and secret hygiene.

Fable-5 audit G2:
  F17 — enforce tool policy at dispatch. The browser agent filters the tools
        the model is *shown*, but the dispatch path trusted whatever name the
        model returned. This module validates name + JSON schema + operation +
        target + authorization at the moment of execution, replaces the
        model-authored "read-only" ``eval`` with typed DOM probes, and
        classifies unrestricted ``evaluate`` as privileged, mutation-capable
        execution that needs an explicit grant.
  F21 — keep user secrets out of tool traces. Field/argument names and
        free-form result text are scrubbed before anything reaches the
        activity log, the model, memory or the UI.

Pure data module: stdlib only, no backend imports (no cycles).
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Operation classes ──────────────────────────────────────────────────────
# Ordered by increasing authority. A dispatcher should require a grant for
# anything at or above MUTATE unless the call is a typed internal handler.
READ = "read"
NAVIGATE = "navigate"
MUTATE = "mutate"
PRIVILEGED = "privileged"

_OPERATION_RANK = {READ: 0, NAVIGATE: 1, MUTATE: 2, PRIVILEGED: 3}

#: Operations that may commit a change even when the transport fails, so they
#: must never be auto-replayed by a retry wrapper.
SIDE_EFFECTING = frozenset((NAVIGATE, MUTATE, PRIVILEGED))


def rank(operation):
    return _OPERATION_RANK.get(operation, _OPERATION_RANK[PRIVILEGED])


def classify_operation(tool, arguments=None):
    """Classify one tool call by the authority it needs.

    ``evaluate`` is PRIVILEGED: arbitrary JavaScript in the page can mutate
    the DOM, submit forms, read cookies and exfiltrate whatever the page can
    reach. It is not "read-only" because a prompt said so.
    """
    name = str(tool or "").strip().lower()
    arguments = arguments or {}
    if name in ("evaluate", "run_js", "exec_js"):
        return PRIVILEGED
    if name in ("click", "click_mark", "click_text", "click_point", "fill",
                "fill_mark", "type", "set_checked", "upload_file",
                "submit", "press", "key", "scroll", "drag_drop",
                "close_tab", "new_tab", "switch_tab", "download",
                "click_locator", "fill_locator", "select_option"):
        return MUTATE
    if name in ("open_brave", "navigate", "open_url", "goto", "open_website",
                "launch_app", "open_in_browser", "youtube_play", "search"):
        return NAVIGATE
    if name in ("look", "screenshot", "list_tabs", "wait_for", "batch_probe",
                "read_file", "list_dir", "list_directory", "probe",
                "verify_playing", "get_text", "dom_query"):
        return READ
    # F17: the native code/editor tools are classified here too, so their
    # dispatch goes through the same authority check as the browser's. Reading
    # is free; writing the workspace is a mutation; RUNNING a program is
    # privileged, because the model does not have to have written it.
    if name in ("code.read_file", "code.list_directory", "code.search",
                "code.read_range", "code.inspect_diff",
                "editor.read_buffer", "editor.workspace_search",
                "editor.test_results", "windows.inspect_active_window"):
        return READ
    if name in ("code.write_file", "code.create_folder", "code.apply_patch",
                "editor.apply_workspace_edit", "editor.open_file"):
        return MUTATE
    if name in ("code.run_command", "code.run_script", "code.run_checks",
                "windows.screen_action"):
        return PRIVILEGED
    # Unknown tools are treated as mutating: failing closed is the only safe
    # default for a name the policy does not recognise.
    return MUTATE


# ── Secret detection ───────────────────────────────────────────────────────
#: Argument / field names whose value must never be logged or returned.
SENSITIVE_KEY_RE = re.compile(
    r"(pass(word|wd|phrase)?|secret|token|api[_-]?key|apikey|auth|"
    r"authorization|credential|otp|one[_-]?time|cvv|cvc|pin|ssn|"
    r"(credit|debit|card)[_-]?(number|num|no)?|session|cookie|"
    r"private[_-]?key|access[_-]?key|refresh[_-]?token|bearer)",
    re.IGNORECASE,
)

#: Free-text shapes that look like a credential even without a telling key.
_SECRET_SHAPE_RES = (
    # sk- / sk-ant- / ghp_ / gsk_ style provider tokens
    re.compile(r"\b(?:sk|pk|rk|ghp|gho|ghu|ghs|gsk|glpat|xox[baprs])[-_][A-Za-z0-9_\-]{16,}"),
    # AWS access key id
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    # JWT
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{4,}"),
    # bearer <token>
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-\.=]{12,}"),
    # key=value / "key": "value" for a sensitive key
    re.compile(
        r"(?i)(?:%s)\s*[\"']?\s*[:=]\s*[\"']?([^\s\"',&}\]]{4,})"
        % SENSITIVE_KEY_RE.pattern
    ),
)

#: Field-level markers used by the browser fill handlers: a value typed into
#: one of these must come back masked, never echoed.
SENSITIVE_FIELD_RE = re.compile(
    r"(pass|pwd|secret|token|otp|cvv|cvc|pin|card|credit|ssn)",
    re.IGNORECASE,
)

MASK = "***masked***"


def mask_value(value):
    """Return a masked stand-in that keeps just enough shape to debug."""
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    if not text:
        return text
    if len(text) <= 2:
        return "*" * len(text)
    return "%s%s%s" % (text[0], "*" * min(6, max(3, len(text) - 2)), text[-1])


def is_sensitive_key(key):
    return bool(key) and bool(SENSITIVE_KEY_RE.search(str(key)))


def is_sensitive_field(*parts):
    """True when an input's identity (type/name/id/placeholder) is sensitive."""
    haystack = " ".join(str(p or "") for p in parts)
    return bool(haystack) and bool(SENSITIVE_FIELD_RE.search(haystack))


#: Recursion guard for the scrub walk. Past this depth the subtree is masked
#: wholesale rather than passed through.
_MAX_SCRUB_DEPTH = 32

#: F21: keys that carry typed input inside a screen/task step. A step can be
#: marked sensitive explicitly, or its own target/label can name a sensitive
#: field ("Password", "OTP"); either way the value it would type is a secret
#: even though the key is the innocuous word "text".
_STEP_INPUT_KEYS = ("text", "input", "value", "keys", "content", "secret_value")
_STEP_MARKER_KEYS = ("sensitive", "secret", "is_password", "isPassword")
_STEP_LABEL_KEYS = ("label", "target", "description", "name", "placeholder",
                    "automation_id")


def _step_is_sensitive(step):
    """True when a step declares (or names) a sensitive input."""
    for marker in _STEP_MARKER_KEYS:
        if step.get(marker) is True:
            return True
    return is_sensitive_field(*(step.get(key) for key in _STEP_LABEL_KEYS))


def scrub_mapping(obj, _depth=0, _seen=None):
    """Deep-copy a mapping/list, masking values whose key looks sensitive.

    F21: the walk used to stop at depth 8 and return the deep subtree
    UNCHANGED, so a nested ``{"a": {"b": ... {"password": "hunter2"}}}``
    reached the trace verbatim. Depth is no longer a licence to skip: past the
    guard the whole subtree is masked, and reference cycles are handled so a
    self-referential payload cannot spin the walk.
    """
    if _seen is None:
        _seen = set()
    if isinstance(obj, dict):
        if _depth >= _MAX_SCRUB_DEPTH or id(obj) in _seen:
            return MASK
        _seen.add(id(obj))
        try:
            step_sensitive = _step_is_sensitive(obj)
            out = {}
            for key, value in obj.items():
                if is_sensitive_key(key) or (
                        step_sensitive and key in _STEP_INPUT_KEYS):
                    out[key] = MASK
                else:
                    out[key] = scrub_mapping(value, _depth + 1, _seen)
            return out
        finally:
            _seen.discard(id(obj))
    if isinstance(obj, (list, tuple, set, frozenset)):
        if _depth >= _MAX_SCRUB_DEPTH or id(obj) in _seen:
            return MASK
        _seen.add(id(obj))
        try:
            return [scrub_mapping(v, _depth + 1, _seen) for v in obj]
        finally:
            _seen.discard(id(obj))
    if isinstance(obj, str):
        # A credential-shaped string is masked wherever it appears, even under
        # an innocent key ("value", "text", "result").
        return mask_secrets(obj)
    return obj


def scrub_text_deep(obj, _depth=0, _seen=None):
    """``scrub_mapping``'s text pass alone: mask credential shapes in every
    string leaf of an arbitrary JSON-ish structure."""
    if _seen is None:
        _seen = set()
    if isinstance(obj, str):
        return mask_secrets(obj)
    if isinstance(obj, dict):
        if _depth >= _MAX_SCRUB_DEPTH or id(obj) in _seen:
            return MASK
        _seen.add(id(obj))
        try:
            return {key: scrub_text_deep(value, _depth + 1, _seen)
                    for key, value in obj.items()}
        finally:
            _seen.discard(id(obj))
    if isinstance(obj, (list, tuple)):
        if _depth >= _MAX_SCRUB_DEPTH or id(obj) in _seen:
            return MASK
        _seen.add(id(obj))
        try:
            return [scrub_text_deep(v, _depth + 1, _seen) for v in obj]
        finally:
            _seen.discard(id(obj))
    return obj


def redact_for_egress(payload):
    """F21: THE redacted egress boundary.

    One function used by every path that can carry tool arguments or results
    out of the process — the activity log, the model message history, the UI,
    persisted artifacts and memory ingestion. It masks by key AND by shape, at
    any depth, so a credential cannot escape just because it arrived inside an
    innocently-named field.
    """
    try:
        if isinstance(payload, str):
            return mask_secrets(payload)
        if isinstance(payload, dict):
            return scrub_text_deep(scrub_mapping(payload))
        if isinstance(payload, (list, tuple)):
            return scrub_text_deep(scrub_mapping(payload))
        if payload is None or isinstance(payload, (int, float, bool)):
            return payload
        return mask_secrets(str(payload))
    except Exception:
        # Never let redaction failure leak the payload.
        return MASK


def mask_secrets(text, limit=None):
    """Scrub credential-shaped substrings out of free-form text.

    Used on every tool result, activity-log line and screen-command log so a
    password or token that reaches a result string still never reaches disk
    or the model.
    """
    if not text:
        return text if isinstance(text, str) else ""
    out = text if isinstance(text, str) else str(text)
    for pattern in _SECRET_SHAPE_RES:
        try:
            out = pattern.sub(lambda m: _mask_match(m), out)
        except Exception:
            continue
    if limit and len(out) > limit:
        out = out[:limit] + "..."
    return out


def _mask_match(match):
    whole = match.group(0)
    if match.lastindex and match.group(match.lastindex) != whole:
        # key=value form: keep the key visible, mask only the value.
        captured = match.group(match.lastindex)
        return whole.replace(captured, MASK)
    return MASK


# ── Typed DOM probe validation (replaces model-authored eval) ───────────────
_MAX_PROBE_CHARS = 400

#: Roots an expression may start from.
_ALLOWED_ROOTS = frozenset(("document", "window", "location", "Array", "JSON",
                            "navigator", "performance", "screen"))

#: Methods a probe may call.
_ALLOWED_CALLS = frozenset((
    "querySelector", "querySelectorAll", "getElementById", "getElementsByTagName",
    "getElementsByClassName", "getElementsByName", "closest", "matches",
    "getAttribute", "hasAttribute", "getAttributeNames", "contains", "indexOf",
    "includes", "slice", "map", "filter", "join", "trim", "toLowerCase",
    "toUpperCase", "replace", "split", "toString", "valueOf", "keys", "values",
    "isArray", "stringify", "parse", "from", "some", "every", "find",
    "getBoundingClientRect", "hasChildNodes", "getComputedStyle",
))

#: Properties a probe may read.
_ALLOWED_PROPS = frozenset((
    "innerText", "textContent", "innerHtmlLength", "value", "checked", "href",
    "src", "alt", "title", "id", "name", "className", "classList", "length",
    "tagName", "nodeName", "nodeType", "children", "childElementCount",
    "parentElement", "firstElementChild", "lastElementChild", "nextElementSibling",
    "previousElementSibling", "disabled", "readOnly", "placeholder", "type",
    "selectedIndex", "options", "text", "url", "pathname", "search", "hash",
    "host", "hostname", "origin", "protocol", "userAgent", "language",
    "width", "height", "top", "left", "right", "bottom", "x", "y", "count",
    "body", "documentElement", "head", "forms", "links", "images", "scripts",
    "currentTime", "paused", "ended", "duration", "readyState", "visibilityState",
    "scrollTop", "scrollHeight", "clientHeight", "clientWidth", "offsetTop",
    "offsetHeight", "offsetWidth", "dataset", "attributes", "labels", "files",
))

#: Anything containing one of these is rejected outright, whatever the shape.
_FORBIDDEN_TOKENS = frozenset((
    "eval", "function", "constructor", "prototype", "import", "require",
    "fetch", "xmlhttprequest", "websocket", "sendbeacon", "localstorage",
    "sessionstorage", "cookie", "indexeddb", "postmessage", "alert", "prompt",
    "confirm", "open", "close", "print", "write", "writeln", "submit",
    "requestsubmit", "click", "focus", "blur", "dispatchevent", "setattribute",
    "removeattribute", "insertadjacenthtml", "appendchild", "removechild",
    "remove", "replacewith", "settimeout", "setinterval", "top", "parent",
    "self", "globalthis", "process", "exec", "spawn", "child_process",
))

_STRING_LITERAL_RE = re.compile(r"'[^']*'|\"[^\"]*\"|`[^`]*`")
_TOKEN_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
#: F17: a bracketed STRING is a way to name a member without writing it as an
#: identifier, so ``document.querySelector('button')['click']()`` ran a
#: mutation while every identifier in the expression looked allowed. Any
#: bracket whose content is a string literal must name an allowed member.
_BRACKET_STRING_RE = re.compile(
    r"\[\s*(?:'([^']*)'|\"([^\"]*)\")\s*\]"
)
#: F17: adjacent/empty string literals are how a computed identifier is built
#: without an operator (``'cl' 'ick'`` / ``'''' ``).
_ADJACENT_LITERAL_RE = re.compile(r"''\s*''|'\s*'\s*'")


def probe_expression_error(expression):
    """Return None when *expression* is a safe typed DOM probe, else a reason.

    The point is to keep the useful part of ``batch_probe`` (asking the page
    several read-only questions in one step) while removing the part that
    made its "read-only" label untrue: the model was handing us arbitrary
    JavaScript that ran with the page's full authority.

    F17: identifier scanning alone was not enough. A member named through a
    *string* — ``document.querySelector('button')['click']()`` — never appears
    as an identifier, so the computed call is checked explicitly here.
    """
    if not isinstance(expression, str) or not expression.strip():
        return "expression must be a non-empty string"
    expr = expression.strip()
    if len(expr) > _MAX_PROBE_CHARS:
        return "expression too long (>%d chars)" % _MAX_PROBE_CHARS
    if ";" in expr or "{" in expr or "}" in expr:
        return "expression must be a single expression (no statements or blocks)"
    # F17: a template literal can carry ${...} code that stripping would hide.
    if "`" in expr:
        return "template literals are not allowed in a read-only probe"
    for op in ("=", "+=", "-=", "*=", "/=", "%=", "++", "--", "=>", "?", ":"):
        if op in expr:
            return "operator %r is not allowed in a read-only probe" % op

    # F17: computed member access. A bracketed string must name an allowed
    # member, and a bracket must not be used to build a name at all.
    for match in _BRACKET_STRING_RE.finditer(expr):
        member = match.group(1) or match.group(2) or ""
        if member not in _ALLOWED_PROPS and member not in _ALLOWED_CALLS:
            return (
                "computed member access [%r] is not allowed in a read-only "
                "probe" % member
            )
    if _ADJACENT_LITERAL_RE.search(expr):
        return (
            "adjacent string literals are not allowed in a read-only probe"
        )

    # Strip string literals so their contents are never scanned as code.
    skeleton = _STRING_LITERAL_RE.sub("''", expr)
    if "[" in skeleton:
        # Anything still bracketed is a computed access whose content was not
        # a plain string literal (a variable, a call, a template): reject it.
        # `''` is the stand-in left for a literal, already vetted above.
        for chunk in re.findall(r"\[([^\]]*)\]", skeleton):
            if chunk.strip() not in ("", "''", '"' + '"', "0", "1", "2", "3",
                                     "4", "5", "6", "7", "8", "9", "-1"):
                return (
                    "computed member access is not allowed in a read-only "
                    "probe"
                )

    lowered = skeleton.lower()
    for token in _FORBIDDEN_TOKENS:
        if re.search(r"(?<![A-Za-z0-9_$.])%s(?![A-Za-z0-9_$])" % re.escape(token), lowered):
            return "forbidden token %r in probe expression" % token
    tokens = list(_TOKEN_RE.finditer(skeleton))
    for match in tokens:
        token = match.group(0)
        after = skeleton[match.end():match.end() + 1]
        before = skeleton[match.start() - 1:match.start()]
        if after == "(":
            if token not in _ALLOWED_CALLS:
                return "call to %r is not an allowed read-only probe" % token
        elif before == ".":
            if token not in _ALLOWED_PROPS and token not in _ALLOWED_CALLS:
                return "property %r is not allowed in a read-only probe" % token
        else:
            if token not in _ALLOWED_ROOTS:
                return "identifier %r is not allowed in a read-only probe" % token
    return None


# ── Dispatch validation ────────────────────────────────────────────────────
@dataclass
class Decision:
    """The verdict of one dispatch check."""

    allowed: bool
    tool: str = ""
    operation: str = READ
    reason: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    #: Names that were masked before logging (for the caller to report).
    masked: List[str] = field(default_factory=list)
    #: F22: the authority this call was actually admitted with, so a downstream
    #: privileged tool consults the validated grants instead of re-deriving
    #: them from caller-supplied arguments.
    effective_grants: frozenset = frozenset()

    def __bool__(self):
        return self.allowed


#: F17: the grant each operation class needs, when the caller uses the
#: default. The browser path needs ``privileged_js`` only; a caller with a
#: different authority vocabulary (the code tools) passes ``required_grants``.
_DEFAULT_REQUIRED_GRANTS = {PRIVILEGED: "privileged_js"}


def validate_dispatch(name, arguments=None, allowlist=None, schema=None,
                      grants=None, max_argument_chars=8000, origin="model",
                      required_grants=None):
    """Validate one tool call at the execution boundary.

    *allowlist* is the set of names the caller may dispatch; *grants* is the
    set of authorities the current job holds — a PRIVILEGED operation needs
    ``privileged_js`` in it. Returns a Decision; ``decision.arguments`` is a
    scrubbed copy safe to log.

    F17: *origin* separates the two dispatch populations. ``"model"`` means a
    call the language model authored, and it fails CLOSED when no allowlist is
    supplied — "no list configured" must never mean "anything goes". Only
    ``"internal"`` (our own handlers invoking a primitive) skips the name gate,
    and even then the authority gate still applies.

    *required_grants* maps an operation class to the grant it needs, so every
    dispatcher shares one authority check instead of inventing its own.
    """
    arguments = arguments if isinstance(arguments, dict) else {}
    name = str(name or "").strip()
    operation = classify_operation(name, arguments)
    grants = set(grants or ())
    safe_args = scrub_mapping(arguments)
    masked = sorted(
        key for key in arguments
        if is_sensitive_key(key) and str(arguments.get(key)) not in ("", None)
    )

    if not name:
        return Decision(False, name, operation, "empty tool name", safe_args, masked)

    is_model_call = str(origin or "model").strip().lower() != "internal"
    if is_model_call:
        if allowlist is None:
            return Decision(
                False, name, operation,
                "no dispatch allowlist is configured, so this call is refused",
                safe_args, masked)
        if name not in allowlist:
            return Decision(
                False, name, operation,
                "tool %r is not in the dispatch allowlist" % name,
                safe_args, masked)
    elif allowlist is not None and name not in allowlist:
        return Decision(
            False, name, operation,
            "tool %r is not in the dispatch allowlist" % name, safe_args, masked)

    # Structural validation: arguments must be JSON-serialisable and bounded.
    try:
        encoded = json.dumps(arguments, default=str)
    except Exception as exc:
        return Decision(False, name, operation,
                        "arguments are not JSON-serialisable: %s" % exc,
                        safe_args, masked)
    if len(encoded) > max_argument_chars:
        return Decision(False, name, operation,
                        "arguments too large (%d chars)" % len(encoded),
                        safe_args, masked)

    if schema:
        ok, reason = _validate_schema(arguments, schema)
        if not ok:
            return Decision(False, name, operation, reason, safe_args, masked)

    # Authority gate: the operation's class needs an explicit grant.
    required = dict(_DEFAULT_REQUIRED_GRANTS if required_grants is None
                    else required_grants)
    needed = required.get(operation)
    if needed and needed not in grants:
        if operation == PRIVILEGED and name in ("evaluate", "run_js", "exec_js"):
            return Decision(
                False, name, operation,
                "privileged page execution is not granted for this job; use "
                "the typed tools (look / click_mark / fill_mark / wait_for / "
                "batch_probe) instead of raw JavaScript",
                safe_args, masked)
        return Decision(
            False, name, operation,
            "%s execution is not granted for this job (needs the %r grant)"
            % (operation, needed), safe_args, masked)

    # Read-only probes must actually be read-only.
    if name == "batch_probe":
        exprs = arguments.get("expressions")
        if not isinstance(exprs, list):
            return Decision(False, name, operation,
                            "batch_probe requires an expressions array",
                            safe_args, masked)
        for expr in exprs:
            reason = probe_expression_error(expr)
            if reason:
                return Decision(False, name, operation,
                                "batch_probe rejected: %s" % reason,
                                safe_args, masked)

    return Decision(True, name, operation, "", safe_args, masked,
                    effective_grants=frozenset(grants))


def _validate_schema(arguments, schema):
    """Minimal JSON-Schema check: type, required, enum, properties."""
    if not isinstance(schema, dict):
        return True, ""
    expected = schema.get("type")
    if expected == "object" and not isinstance(arguments, dict):
        return False, "arguments must be an object"
    if expected == "array" and not isinstance(arguments, list):
        return False, "arguments must be an array"
    for key in schema.get("required", []) or []:
        if isinstance(arguments, dict) and key not in arguments:
            return False, "missing required argument %r" % key
    properties = schema.get("properties") or {}
    if isinstance(arguments, dict):
        for key, value in arguments.items():
            sub = properties.get(key)
            if not isinstance(sub, dict):
                continue
            kind = sub.get("type")
            if kind and not _matches_type(value, kind):
                return False, "argument %r must be %s" % (key, kind)
            enum = sub.get("enum")
            if enum and value not in enum:
                return False, "argument %r must be one of %s" % (key, enum)
    return True, ""


def _matches_type(value, kind):
    # F11: a declared type may be a list of alternatives (JSON-Schema style).
    # A structured cursor is an object on the way out of a tool and a JSON
    # string on the way back in via an observation placeholder, so pinning it
    # to one type would make the two halves of the same loop disagree.
    if isinstance(kind, (list, tuple)):
        return any(_matches_type(value, item) for item in kind)
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    if kind == "null":
        return value is None
    return True


# ── Human-readable effect summaries (for approval records) ─────────────────
def describe_call(name, arguments=None):
    """One-line effect description used in approval previews and logs."""
    arguments = arguments or {}
    operation = classify_operation(name, arguments)
    target = (arguments.get("selector") or arguments.get("text")
              or arguments.get("url") or arguments.get("index")
              or arguments.get("path") or arguments.get("command"))
    target = mask_secrets(str(target)) if target is not None else ""
    if target:
        target = str(target)[:80]
        return "%s (%s) on %s" % (name, operation, target)
    return "%s (%s)" % (name, operation)


def describe_plan(plan):
    """Ordered effect list for an arbitrary plan dict (screen or task).

    F18: a native task step carries its tool arguments under ``args`` while a
    screen/vision step carries them under ``arguments``. Reading only
    ``arguments`` made the shared preview describe the step *wrapper* instead
    of the effect it would have ("code.write_file (mutate)" with no target),
    so consent was requested without the target ever appearing in it.
    """
    effects = []
    for step in (plan or {}).get("steps") or []:
        if not isinstance(step, dict):
            continue
        if step.get("tool"):
            # F14/F18: an EXTERNAL-effect step (a mail send, a calendar commit)
            # carries the scrubbed draft preview the task agent built when it
            # requested the separate effect approval. The approval record the
            # UI reads must show that same preview — tool+args alone hid the
            # actual message/event being sent.
            preview = step.get("effect_preview")
            if preview:
                effects.append(mask_secrets(str(preview))[:200])
                continue
            arguments = step.get("arguments")
            if not isinstance(arguments, dict):
                arguments = step.get("args")
            if not isinstance(arguments, dict):
                arguments = step
            effects.append(describe_call(step.get("tool"), arguments))
            continue
        action = step.get("action") or "step"
        bits = [str(action)]
        for key in ("text", "input", "keys", "description", "label"):
            if step.get(key):
                bits.append(str(step[key])[:60])
                break
        if "x" in step and "y" in step:
            bits.append("at (%s,%s)" % (step.get("x"), step.get("y")))
        effects.append(mask_secrets(" ".join(bits)))
    return effects


#: F18: a tool name read aloud is friendlier as a noun phrase ("2 write files").
_EFFECT_HEAD_ALIASES = {
    "write_file": "file write",
    "read_file": "file read",
    "create_folder": "folder creation",
    "list_directory": "directory listing",
    "run_command": "command run",
    "run_script": "script run",
    "run_checks": "check run",
    "apply_patch": "patch",
    "inspect_diff": "diff inspection",
    "read_range": "file range read",
}


def effect_counts(effects):
    """Concise spoken summary: '3 clicks, 1 typed text'."""
    counts = {}
    for effect in effects or []:
        head = str(effect).split(" ")[0].strip("(")
        # F18: "code.write_file" -> "file write", so the count reads aloud.
        head = head.rsplit(".", 1)[-1].replace("_", " ")
        head = _EFFECT_HEAD_ALIASES.get(head.replace(" ", "_"), head)
        counts[head] = counts.get(head, 0) + 1
    if not counts:
        return "no actions"
    return ", ".join(
        "%d %s%s" % (n, name, "" if n == 1 or name.endswith("s") else "s")
        for name, n in counts.items()
    )
