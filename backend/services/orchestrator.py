"""Contextual native tool-use orchestrator (Fable-5 audit G8: F02) — the
migration's core verdict: "Route goals, not single categories."

CURRENT (audit F02): ``intent.py`` requires "one only"; brain.py classifies
and then rewrites the category; malformed classifier output silently falls
back to chat.

CHANGE: one native-tool loop over the planner model (Fireworks qwen3p7-plus,
resolved through the capability-validated registry — F49). A single request
can inspect, research and act in sequence. Bounded domain tools:

  * screen.observe  (READ)   — the same analyze_screen the screen path uses
  * research.lookup (READ)   — the same quick search tier the research path uses
  * memory.recall   (READ)   — recent conversation turns
  * web.search      (READ)   — a web search summary
  * task.propose    (MUTATE) — surface a task plan through the EXISTING
                               confirmation gate; the orchestrator itself
                               never executes mutations

Rules enforced here (audit §5 "Keep deterministic"):
  * tool arguments are validated against the JSON schema before any handler
    runs; invalid output is fed back as an explicit tool error — it is
    NEVER mistaken for a deliberate conversational answer;
  * a plain-content turn IS the conversational answer (no classification
    round trip for ordinary chat);
  * every mutation is a PROPOSAL: authorization happens at dispatch (G2's
    approval records), never from a model saying so;
  * the deterministic safety nets (screen questions, explicit stops) replay
    deterministically BEFORE any model turn (see handle_message).
"""

import json
import logging
import threading

from backend.services import model_registry
from backend.services.openai_compat_client import ask_openai_compat

#: The loop is bounded: at most this many model turns (F01's budget thinking:
#: budget exhaustion returns partial, never a silent flail).
MAX_ORCHESTRATOR_TURNS = 4

#: Per-turn completion budget for the planner model. F49: checked against the
#: resolved model's own limits before the request is sent.
ORCHESTRATOR_MAX_TOKENS = 1200

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "screen.observe",
            "description": "Look at the user's screen (vision analysis). Use for 'what's on my screen' or to inspect the current screen state before acting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "What to look for on the screen.",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research.lookup",
            "description": "Look up CURRENT world facts (prices, versions, news, weather) and return a cited summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The lookup query."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory.recall",
            "description": "Recall recent conversation turns for context.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web.search",
            "description": "Run a general web search and return the top result text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task.propose",
            "description": "Propose a real computer task (files/shell/browser automation). Returns a plan the user must CONFIRM — never executes by itself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "The task as the user meant it."},
                },
                "required": ["description"],
            },
        },
    },
]

_TOOL_SCHEMA_BY_NAME = {
    t["function"]["name"]: t["function"]["parameters"]
    for t in TOOL_DEFINITIONS
}

_ORCHESTRATOR_SYSTEM = (
    "You are Jarvis, a Windows desktop assistant. Answer conversationally when the "
    "request is conversational. For goals that need facts, inspection or action, "
    "CALL the provided tools in sequence — one request may inspect, research and "
    "act. Keep spoken replies short, warm and natural (sir is fine). For real "
    "computer tasks, use task.propose — the user confirms before anything runs."
)

#: Evidence cap per turn count of executed READ tools in the outcome.
_EVIDENCE_LIMIT = 8


# ── Tool argument validation (audit F02: "validate tool arguments") ───────
#: JSON-schema type → the Python types that satisfy it (F02: "wrong argument
#: types are accepted" — ``screen.observe {"question": 5}`` used to run).
_JSON_TYPES = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
}


def _type_error(key, value, declared):
    expected = _JSON_TYPES.get(declared)
    if expected is None:
        return None
    if declared in ("integer", "number") and isinstance(value, bool):
        # bool is an int subclass in Python; "true" is not the number 1.
        return "argument '%s' must be %s, got boolean" % (key, declared)
    if not isinstance(value, expected):
        return "argument '%s' must be %s, got %s" % (
            key, declared, type(value).__name__)
    return None


def validate_tool_call(name, raw_arguments):
    """Validate one native tool call against its schema.

    Returns ``(args, None)`` when valid, ``(None, reason)`` when invalid —
    invalid model output is an explicit error, never a silent reinterpret.
    """
    if name not in _TOOL_SCHEMA_BY_NAME:
        return None, "unknown tool '%s'" % (name,)
    schema = _TOOL_SCHEMA_BY_NAME[name]
    try:
        parsed = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except Exception as exc:
        return None, "arguments are not valid JSON: %s" % (exc,)
    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        return None, "arguments must be a JSON object, got %s" % (type(parsed).__name__,)
    properties = schema.get("properties") or {}
    required = schema.get("required") or []
    for key in required:
        value = parsed.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            return None, "missing required argument '%s'" % (key,)
    for key, value in parsed.items():
        if not isinstance(value, (str, int, float, bool)):
            return None, "argument '%s' must be scalar" % (key,)
        if isinstance(key, str) and key not in properties:
            return None, "unknown argument '%s'" % (key,)
        declared = (properties.get(key) or {}).get("type")
        problem = _type_error(key, value, declared)
        if problem:
            return None, problem
    return {k: v for k, v in parsed.items()}, None


# ── F02: explicit outcome contract ────────────────────────────────────────
#: The four statuses the orchestrator may report. Anything else is a bug, so
#: the strings are named once here:
#:   * ANSWERED   — a deliberate conversational answer (plain-content turn);
#:   * PROPOSAL   — a mutation was proposed; consent is armed and identified;
#:   * SUSPENSION — the loop ran out of usable output/turns (caller falls back);
#:   * ERROR      — the planner could not be reached or produced nothing.
ANSWERED = "answered"
PROPOSAL = "proposal"
SUSPENSION = "suspension"
ERROR = "error"
STATUSES = frozenset((ANSWERED, PROPOSAL, SUSPENSION, ERROR))


def new_outcome():
    """A fresh outcome in the F02 contract."""
    return {
        "status": SUSPENSION,
        "reply": "",
        "actions": [],
        "evidence": [],
        "plan": None,
        #: Every proposal this turn produced, in order. A second proposal must
        #: never silently overwrite the first (see _tool_task_propose).
        "proposals": [],
    }


def envelope_utterance(envelope):
    """The request text from *envelope*, safely.

    F02: handlers used ``args.get(...) or envelope.utterance`` — a falsey or
    missing argument reached an attribute the envelope does not always carry,
    so a valid tool call died with AttributeError instead of running.
    """
    for attr in ("utterance", "message", "text"):
        value = getattr(envelope, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(envelope, dict):
        for key in ("utterance", "message", "text"):
            value = envelope.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _coerce_screen_output(output):
    """Flatten an ``analyze_screen`` result into evidence text (F02).

    This function did not exist: every screen.observe call raised NameError
    and was reported to the model as a tool error.
    """
    if output is None:
        return ""
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, dict):
        for key in ("answer", "text", "summary", "reply", "content"):
            value = output.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return _coerce_string_output(output)
    if isinstance(output, (list, tuple)):
        return "\n".join(
            part for part in (_coerce_screen_output(item) for item in output)
            if part)
    for attr in ("answer", "text", "summary", "reply", "content"):
        value = getattr(output, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _coerce_string_output(output)


# ── Tool handlers (the audit's bounded domain tools) ───────────────────────
# READ tools execute directly; the one MUTATE tool (task.propose) only ever
# returns a proposal for the existing confirmation gates.
def _tool_screen_observe(args, outcome, envelope):
    from backend.services.screen_analyzer import analyze_screen
    question = str(args.get("question") or envelope_utterance(envelope))
    output = analyze_screen(question)
    evidence = _coerce_screen_output(output)
    outcome["evidence"].append("screen.observe: %s" % (evidence or "no usable output"))
    return evidence


def _tool_research_lookup(args, outcome, envelope):
    from backend.services.quick_search import run_quick_search
    query = str(args.get("query") or envelope_utterance(envelope))
    result = run_quick_search(query)
    text = _coerce_string_output(result)
    outcome["evidence"].append("research.lookup(%s): %d chars" % (query[:80], len(text)))
    return text


def _tool_memory_recall(args, outcome, envelope):
    from backend.core.memory import get_history
    turns = get_history() or []
    lines = []
    for turn in list(turns)[-5:]:
        if isinstance(turn, dict):
            user = str(turn.get("user") or turn.get("message") or "").strip()
            reply = str(turn.get("assistant") or turn.get("response") or "").strip()
            if user:
                lines.append("user: %s" % user[:300])
            if reply:
                lines.append("jarvis: %s" % reply[:300])
    text = "\n".join(lines) or "(no recent conversation)"
    outcome["evidence"].append("memory.recall: %d recent turns" % len(lines))
    return text


def _tool_web_search(args, outcome, envelope):
    # Function-level import: brain imports orchestrator at load time, so
    # search_internet must resolve lazily (by call time brain is complete).
    from backend.core.brain import search_internet
    query = str(args.get("query") or envelope_utterance(envelope))
    result = search_internet(query)
    text = _coerce_string_output(result)
    outcome["evidence"].append("web.search(%s): %d chars" % (query[:80], len(text)))
    return text


def _tool_task_propose(args, outcome, envelope):
    """Propose a computer task and ARM ITS OWN APPROVAL (F02).

    The orchestrator never executes a mutation. It hands the plan to the one
    confirmation gate the legacy task path already consumes
    (``task_agent.register_proposal``), so the plan the user is asked about is
    the plan that later runs — with an identified record, a hash and an
    expiry — instead of a proposal the brain threw away after reading the
    reply.

    A second proposal does NOT overwrite the first: each is recorded in
    ``outcome["proposals"]`` and the reply names the plan that is armed.
    """
    from backend.services.task_agent import agent as task_agent
    description = str(args.get("description") or envelope_utterance(envelope))
    steps = 0
    # F02: at most ONE proposal may be armed per request. A second
    # task.propose call must not replace the approval the user is about to be
    # asked about, so it is reported back to the model instead.
    for existing in outcome["proposals"]:
        if existing.get("approval_id"):
            message = ("TOOL ERROR: a task is already awaiting confirmation "
                       "(%s); it cannot be replaced in the same request."
                       % existing["approval_id"])
            outcome["evidence"].append(
                "task.propose: refused a second proposal while %s is awaiting "
                "confirmation" % existing["approval_id"])
            return message
    # F02: the plan is built with the SAME context the legacy task path uses,
    # so the plan the orchestrator proposes is executable by that path.
    try:
        context = task_agent.gather_context()
    except Exception:
        context = None
    plan = task_agent.plan_task(description, context)
    steps = len(plan.get("steps") or [])
    record, reason = task_agent.register_proposal(
        plan, context, task_text=description)
    if record is None:
        # Nothing to confirm (a read-only plan, an unbuildable plan, or a
        # different approval already pending): report it, do not pretend an
        # approval exists.
        outcome["evidence"].append(
            "task.propose: %d step(s) planned — %s" % (steps, reason))
        outcome["proposals"].append({"plan": plan, "approval_id": None})
        return plan.get("summary") or description
    outcome["evidence"].append(
        "task.propose: %d step(s) planned — awaiting user confirmation "
        "(approval %s)" % (steps, record.id))
    outcome["status"] = PROPOSAL
    outcome["plan"] = plan
    outcome["proposals"].append({
        "plan": plan,
        "approval_id": record.id,
        "plan_hash": record.plan_hash,
        "preview": record.preview,
        "expires_at": record.expires_at,
    })
    return plan.get("summary") or description


_TOOL_HANDLERS = {
    "screen.observe": _tool_screen_observe,
    "research.lookup": _tool_research_lookup,
    "memory.recall": _tool_memory_recall,
    "web.search": _tool_web_search,
    "task.propose": _tool_task_propose,
}


#: F46: bound on one tool result's serialized size. Tool output is fed back to
#: the planner verbatim; without a bound a single research/observe result could
#: blow the request budget (and the old planner path sliced serialized JSON at
#: an arbitrary character, producing unparseable JSON). Oversized output is
#: summarized/ clipped WITH an explicit marker, never silently halved.
TOOL_OUTPUT_BUDGET = 4000


def _coerce_string_output(result):
    """Best-effort, BOUNDED string coercion of a service output (F46)."""
    from backend.services.context_envelope import budgeted_json, clip_for_prompt

    if result is None:
        return ""
    if isinstance(result, str):
        return clip_for_prompt(result, TOOL_OUTPUT_BUDGET)
    try:
        return budgeted_json(result, TOOL_OUTPUT_BUDGET)
    except Exception:
        return clip_for_prompt(str(result), TOOL_OUTPUT_BUDGET)


def _chat(messages, snapshot):
    """One native-tool-call completion through the resolved planner config.

    F49: the snapshot (``model_registry.resolve_call_config``) carries the
    adapter, the endpoint AND the credentials, resolved together from a single
    settings read — so every turn of one request uses the same model and the
    same endpoint even if the model is switched mid-flight. When the snapshot
    has no usable OpenAI-compatible endpoint nothing is sent: the Fireworks
    URL is never substituted for another provider's endpoint, and no other
    provider's credentials are ever sent to Fireworks. A request that cannot
    fit the model's own limits is refused the same way.
    """
    snapshot = snapshot or {}
    endpoint = snapshot.get("endpoint") or {}
    api_key = endpoint.get("api_key")
    base_url = endpoint.get("base_url")
    if snapshot.get("adapter") != "openai_compatible" or not api_key or not base_url:
        return {}
    try:
        model_registry.validate_request_limits(
            snapshot, max_tokens=ORCHESTRATOR_MAX_TOKENS)
    except Exception as exc:
        print("[ORCHESTRATOR] planner request exceeds the model limits (%s)" % exc)
        return {}
    kwargs = {
        "model": snapshot.get("model"),
        "base_url": base_url,
        "api_key": api_key,
        "temperature": 0.4,
        "max_tokens": ORCHESTRATOR_MAX_TOKENS,
        "tools": TOOL_DEFINITIONS,
        "tool_choice": "auto",
    }
    reasoning = snapshot.get("reasoning") or {}
    if reasoning.get("supported") and reasoning.get("effort"):
        # F49: only send a reasoning control the model/adapter actually takes.
        kwargs["reasoning_effort"] = reasoning["effort"]
    return ask_openai_compat(messages, **kwargs)


def _extract_answer(result):
    choices = (result or {}).get("choices") or []
    if not choices:
        return "", []
    message = choices[0].get("message") or {}
    return (message.get("content") or "").strip(), message.get("tool_calls") or []


def _run_tool_call(call, outcome, envelope):
    """Execute one validated tool call; returns the tool message content."""
    name = str(call.get("function", {}).get("name") or "").strip()
    raw_args = call.get("function", {}).get("arguments") or "{}"
    args, error = validate_tool_call(name, raw_args)
    if error:
        # Invalid model output — fed back EXPLICITLY as a tool error so the
        # next turn can correct it (never treated as an answer, never executed).
        outcome["evidence"].append("tool rejected: %s %s" % (name, error))
        return "TOOL ERROR: %s" % (error,)
    outcome["actions"].append({"tool": name, "args": args})
    handler = _TOOL_HANDLERS[name]
    try:
        return handler(args, outcome, envelope) or "(no output)"
    except Exception as exc:
        outcome["evidence"].append("tool failed: %s (%s)" % (name, exc))
        return "TOOL ERROR: %s" % (exc,)


def run_orchestrator(msg, envelope, max_turns=MAX_ORCHESTRATOR_TURNS):
    """The native tool-use loop for one request.

    Returns the outcome dict (see ``new_outcome``):
      {"status": "answered"|"proposal"|"suspension"|"error",
       "reply": ..., "actions": [...], "evidence": [...],
       "plan": ..., "proposals": [...]}
    or None when the loop produced nothing usable (caller then falls back
    to the legacy routing).
    """
    outcome = new_outcome()

    # Deterministic replay of the screen-question safety net (G8 migration):
    # the LLM never decides whether to look — a screen question ALWAYS looks,
    # through the same analyze_screen capability the screen path uses.
    if getattr(envelope, "screen_question", False):
        text = _tool_screen_observe({"question": msg}, outcome, envelope)
        messages = [
            envelope.to_system_message(),
            {"role": "user", "content": msg},
            {"role": "user", "content": "Screen analysis result:\n%s" % text},
        ]
    else:
        messages = [envelope.to_system_message(), {"role": "user", "content": msg}]

    # F49: one validated, atomic configuration snapshot for the whole loop.
    # A configuration that cannot be validated (unknown/invalid persisted
    # selection, a private provider without credentials) fails CLOSED: the
    # orchestrator declines and the legacy routing continues — no request is
    # sent to any substitute provider.
    try:
        snapshot = model_registry.resolve_call_config("planner")
    except Exception as exc:
        print("[ORCHESTRATOR] planner configuration rejected (%s) — legacy "
              "routing continues." % (exc,))
        return None
    turns = 0
    while turns < max_turns:
        turns += 1
        result = _chat(messages, snapshot)
        if not result:
            outcome["evidence"].append("planner call failed (turn %d)" % turns)
            if not outcome["actions"]:
                outcome["status"] = ERROR
                return None
            outcome["status"] = SUSPENSION
            return outcome
        answer, tool_calls = _extract_answer(result)
        if tool_calls:
            messages.append({"role": "assistant", "content": answer, "tool_calls": tool_calls})
            for call in tool_calls:
                content = _run_tool_call(call, outcome, envelope)
                messages.append({
                    "role": "tool",
                    "tool_call_id": str(call.get("id") or "call_%d" % turns),
                    "content": content,
                })
            if outcome["status"] == PROPOSAL:
                # task.propose armed its proposal; return WITHOUT another
                # model turn — the next model turn cannot add authority to
                # a mutation (confirmation happens at dispatch, G2 style).
                plan = outcome.get("plan") or {}
                outcome["reply"] = plan.get("summary") or answer
                return outcome
            continue
        if answer:
            # A plain-content turn IS the deliberate conversational answer —
            # no classification round trip for ordinary chat (audit F02).
            outcome["status"] = ANSWERED
            outcome["reply"] = answer
            outcome["evidence"].append("answered after %d turn(s)" % turns)
            return outcome
        # Neither content nor tool calls — unusable output, explicitly
        # distinguished from a deliberate answer.
        outcome["evidence"].append("unusable model output (turn %d)" % turns)
        break

    outcome["status"] = SUSPENSION
    return outcome if outcome["actions"] else None


def orchestrator_mode():
    """Live migration-mode read: True when the flag is 'orchestrator'."""
    try:
        from backend import config
        return str(getattr(config, "ORCHESTRATOR_MODE", "legacy")).strip().lower() == "orchestrator"
    except Exception:
        return False


def select_route(msg, screen_question=False):
    """F02 — choose the route BEFORE any work starts.

    Returns ``"orchestrator"`` or ``"legacy"``. The caller uses this to decide
    whether to start speculative chat work at all: speculation used to start
    before the orchestrator was consulted, so a goal-shaped request paid for a
    chat stream that was then thrown away (and, on fallback, re-ran work that
    had already started).

    Deliberately deterministic and cheap: no model call, no I/O.
    """
    text = str(msg or "").strip()
    if not text or not orchestrator_mode():
        return "legacy"
    # "command ..." is the explicit prefix that means "run this task via the
    # task path"; the orchestrator never owns it.
    if text.lower().startswith("command"):
        return "legacy"
    return "orchestrator"


def handle_message(msg, history=None, connectors=None, screen_question=False):
    """Brain's entry point — None when the orchestrator declines.

    Declines (returns None) when the migration flag is off, or when the
    loop produced nothing usable; the legacy classifier routing (and every
    deterministic safety net) then runs exactly as before. Nothing here
    removes the protections of the existing paths.
    """
    if select_route(msg, screen_question=screen_question) != "orchestrator":
        return None
    msg = str(msg or "").strip()
    if not msg:
        return None
    envelope = _build_envelope(
        msg, history=history, connectors=connectors, screen_question=screen_question)
    try:
        return run_orchestrator(msg, envelope)
    except Exception as exc:
        print("[ORCHESTRATOR] declined (%s) — legacy routing continues." % exc)
        return None


def _build_envelope(msg, history=None, connectors=None, screen_question=False,
                    capture=None):
    """The ONE request snapshot every consumer reads (F46).

    F46: brain.py does not (yet) hand the connector snapshot in, so the
    orchestrator gathers it HERE, once, instead of every consumer probing
    separately — and records the concrete capture identity (target hwnd +
    capture epoch) alongside it. Both gathers are best-effort: a connector or
    screen probe that fails leaves the identity out (with its retrieval hint
    intact) rather than blocking the request.
    """
    from backend.services.context_envelope import build_envelope as _build
    if connectors is None:
        connectors = _gather_connectors()
    if capture is None:
        capture = _capture_snapshot()
    return _build(
        msg, history=history, connectors=connectors, capture=capture,
        screen_question=screen_question)


#: F46: how long the one-time connector gather may take before the envelope is
#: built without it (the identity fields are then reported as omitted, with
#: their retrieval hints, instead of stalling the request on a connector).
CONNECTORS_GATHER_TIMEOUT = 1.5


def _gather_connectors(timeout=CONNECTORS_GATHER_TIMEOUT):
    """Best-effort, TIME-BOUNDED connector snapshot for the envelope (F46)."""
    outcome = {}

    def _work():
        try:
            from backend.services.task_agent import agent as task_agent
            outcome["value"] = task_agent.gather_context()
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = exc

    worker = threading.Thread(target=_work, name="jarvis-envelope-context",
                              daemon=True)
    worker.start()
    worker.join(max(0.0, float(timeout)))
    if worker.is_alive():
        logging.warning("[ORCHESTRATOR] connector gather exceeded %.1fs — "
                        "envelope continues without identities", timeout)
        return None
    if "error" in outcome:
        logging.warning("[ORCHESTRATOR] context gather failed: %s",
                        outcome["error"])
        return None
    return outcome.get("value")


def _capture_snapshot():
    """Concrete capture identity for the envelope: target hwnd + epoch (F46).

    Read-only: no screenshot is taken here. The epoch is what lets a later
    consumer tell whether the screen was re-observed since this snapshot.
    """
    try:
        from backend.services import screen_capture
        target = screen_capture.last_foreground_target() or {}
        return {
            "capture_mode": "active_window" if target.get("hwnd") else "",
            "hwnd": target.get("hwnd"),
            "bounds": target.get("bounds"),
            "monitor": (target.get("bounds") or {}).get("monitor"),
            "process_id": target.get("process_id"),
            # H4: the CAPTURE's own epoch (stamped on the target at capture
            # time), not whatever happens to be current at envelope time.
            "epoch": target.get("capture_epoch")
                or screen_capture.current_capture_epoch(),
        }
    except Exception as exc:  # noqa: BLE001 - identity is optional
        logging.warning("[ORCHESTRATOR] capture identity unavailable: %s", exc)
        return None
