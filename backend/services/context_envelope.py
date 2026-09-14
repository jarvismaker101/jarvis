"""One bounded, request-scoped ContextEnvelope (Fable-5 audit G8: F46).

CURRENT (audit F46): the task planner slices serialized connector JSON at
12,000 characters, the intent classifier sends only the current utterance,
and chat/screen paths build different context views — so "fix this", "use
that tab" and cross-modal follow-ups make every subsystem guess independently.

CHANGE (round 2 — the audit's remaining defects): ONE IMMUTABLE
identity-rich request snapshot is built per request and every consumer reads
the SAME snapshot:

  * the snapshot carries CONCRETE identities (window hwnd/process, editor
    file + version + selected text, browser instance/tab id + url + title,
    capture mode + monitor/bounds + observation time), deep-copied at build
    time, so "use that tab" keeps resolving to the same tab even after focus
    changes;
  * every omission is retained as a RECORD with a retrieval hint (which tool
    to call to get the missing field back) instead of a bare field name;
  * the FINAL serialization is budgeted: ``render()``/``to_json()`` can never
    exceed TOTAL_BUDGET — not even with pathological keys or escape-heavy
    strings — and a bounded serialization is always VALID JSON (a sliced JSON
    prefix is never emitted).

Pure data module: stdlib only (no brain/task_agent imports — callers hand in
already-gathered context snapshots, keeping this free of cycles).
"""

import copy
import json
from types import MappingProxyType

#: Per-field character budgets, applied BEFORE serialization. A field that
#: exceeds its budget is clipped at a word boundary and the clipping is made
#: EXPLICIT in the rendered view (audit F46: "explicit omitted-field
#: indicators" — never a silent truncation).
FIELD_BUDGETS = {
    "utterance": 2000,
    "goal_state": 1600,
    "memory_hints": 1200,
    "references": 2400,
    "editor_text": 1200,
    "browser_target": 600,
    "capture_identity": 300,
}

#: Total serialization budget for the whole rendered envelope. The renderer
#: drops the lowest-priority optional fields (in _DROP_ORDER) until the view
#: fits, always leaving the utterance and an explicit omission list.
TOTAL_BUDGET = 4000

#: Longest key kept verbatim in a summarized view. A key is data too: one
#: 8,000-character key (or an escape-heavy one) previously blew every bound
#: that trusted the per-field budget.
MAX_KEY_CHARS = 64

#: Keys kept when an oversized mapping is summarized.
MAX_SUMMARY_KEYS = 20

#: Optional fields dropped (whole, explicitly listed) when the total view
#: exceeds TOTAL_BUDGET — in this order (capture identity first: it is the
#: smallest loss; goal state last: the loop depends on it most).
_DROP_ORDER = ("capture_identity", "browser_target", "editor_text",
               "memory_hints", "goal_state")

#: Last-resort drop for the JSON view only: after every optional field is
#: gone the utterance alone is still guaranteed to fit.
_JSON_LAST_RESORT = ("goal_state",)

#: Which fields each consumer reads from the ONE snapshot. A new consumer
#: must be named here rather than re-deriving its own context view.
CONSUMER_FIELDS = {
    "planner": tuple(FIELD_BUDGETS),
    "chat": ("utterance", "goal_state", "memory_hints", "references"),
    "screen": ("utterance", "references", "capture_identity"),
    "research": ("utterance", "goal_state", "references"),
}

#: How to get an omitted field back. The point of an omission record is that
#: the model (or the caller) can RETRIEVE the missing content through a tool
#: instead of guessing — never that the content simply vanished.
RETRIEVAL_HINTS = {
    "goal_state": "tool memory.recall (recent conversation turns)",
    "memory_hints": "tool memory.recall (personal memory)",
    "references": ("tools windows.inspect_active_window / "
                   "editor.inspect_workspace / browser.inspect_tabs"),
    "editor_text": ("tool editor.read_buffer(path=<references.editor_file>, "
                    "expected_version=<references.editor_version>)"),
    "browser_target": "tool browser.inspect_tabs",
    "capture_identity": "tool screen.observe",
    "utterance": "none — the utterance is the request itself",
}

_OMISSION_MARK = "[omitted: %s]"

_CLIP_SUFFIX = "…[clipped]"

_TRUNC_MARK = "…[truncated]"

_NOT_RENDERED = "[not rendered]"


def _clip(text, budget):
    """Word-boundary clip with an explicit continuation marker.

    The suffix space is RESERVED: a clipped field is guaranteed to be at
    most *budget* characters (a clip that overshoots its budget would break
    every downstream bound that trusted it).
    """
    text = str(text or "").strip()
    if not text:
        return ""
    if len(text) <= budget:
        return text
    room = max(1, budget - len(_CLIP_SUFFIX))
    cut = text.rfind(" ", 0, room)
    if cut <= 0:
        cut = room
    return text[:cut].rstrip() + _CLIP_SUFFIX


def _bounded_keys(mapping, limit=MAX_SUMMARY_KEYS):
    """Key list for a summarized mapping — every key clipped to MAX_KEY_CHARS."""
    return [_clip(str(key), MAX_KEY_CHARS) for key in list(mapping)[:limit]]


def _clip_json(value, budget):
    """Bounded JSON view: the value when it fits, else an explicit summary.

    Halved JSON is worse than none — an unparseable prefix can mislead a
    planner. Oversized dict/list shapes are listed explicitly instead, and the
    listed KEYS are themselves bounded (a long key is data that can overflow
    the very budget that summarized its container).
    """
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    if len(text) <= budget:
        return value
    if isinstance(value, dict):
        return {"_note": "dict too large to serialize",
                "keys": _bounded_keys(value)}
    if isinstance(value, list):
        return {"_note": "list too large to serialize", "length": len(value)}
    return _clip(text, budget)


def _freeze(value):
    """Deep-immutable copy of a JSON-ish value (dict -> read-only mapping)."""
    if isinstance(value, dict):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value):
    """Thaw a frozen value back into plain JSON-serializable shapes."""
    if isinstance(value, (MappingProxyType, dict)):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _safe_copy(value):
    """Best-effort deep copy so a caller mutating its dict cannot leak in."""
    try:
        return copy.deepcopy(value)
    except Exception:
        try:
            return json.loads(json.dumps(value, ensure_ascii=False, default=str))
        except Exception:
            return value


def _hard_clamp(text, budget):
    """Final character bound for the rendered (non-JSON) view."""
    if len(text) <= budget:
        return text
    room = max(0, budget - len(_TRUNC_MARK))
    return text[:room] + _TRUNC_MARK


def clip_for_prompt(text, budget):
    """Bound one plain-text value for a prompt, with an explicit marker.

    Public counterpart of the internal field clip: callers that must bound
    already-serialized text (tool output, research summaries) use this so the
    clipping is always visible instead of a silent cut.
    """
    return _clip(text, max(1, int(budget)))


def budgeted_json(value, budget=TOTAL_BUDGET):
    """A VALID JSON string of *value*, never longer than *budget* characters.

    F46: "no truncated JSON". A prefix slice of serialized JSON cannot be
    parsed, so an oversized value is replaced by an explicit — and itself
    bounded — summary object instead of being cut. The result is ALWAYS
    parseable and always within the budget.
    """
    budget = max(64, int(budget))
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = json.dumps(str(value))
    if len(text) <= budget:
        return text

    keys = _bounded_keys(value) if isinstance(value, dict) else []
    payload = {
        "_note": "oversized value summarized to fit the budget",
        "keys": keys,
    }
    if isinstance(value, dict) and len(value) > len(keys):
        payload["_omitted_keys"] = len(value) - len(keys)
    text = json.dumps(payload, ensure_ascii=False, default=str)
    while len(text) > budget and payload["keys"]:
        payload["keys"] = payload["keys"][:-1]
        text = json.dumps(payload, ensure_ascii=False, default=str)
    while len(text) > budget and "keys" in payload:
        del payload["keys"]
        text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) > budget:
        # Escape-heavy content can still overflow after summarization; the
        # fallback object is tiny, valid JSON and still fits any sane budget.
        text = json.dumps({"_note": "too large to serialize"},
                          ensure_ascii=False)
    return text


class ContextEnvelope:
    """One request's bounded, immutable, identity-rich context snapshot.

    Fields are the audit F46 list; every field carries its own budget and
    omissions are explicit AND retrievable. ``references`` holds RESOLVED
    entities (active window identity, editor file + version + selection,
    concrete browser tab identity) so follow-ups like "fix this" or "use that
    tab" point at concrete things instead of pronouns — and keep pointing at
    the same thing for every consumer even if focus changes afterwards.

    Immutability: the caller's dicts are deep-copied at construction, the
    identity snapshot is deep-frozen, and the envelope's own attributes cannot
    be reassigned. ``render()``/``to_json()`` only record the omissions they
    decided on — they never change the snapshot.
    """

    #: Attributes that may never be reassigned once the snapshot is built.
    _LOCKED = frozenset((
        "fields", "identities", "screen_question", "capture", "consumer",
    ))

    def __init__(self, utterance, goal_state=None, references=None,
                 memory_hints=None, editor_text=None, browser_target=None,
                 capture_identity=None, screen_question=False, identities=None,
                 capture=None):
        self.fields = {
            "utterance": _clip(utterance, FIELD_BUDGETS["utterance"]),
            "goal_state": _clip(goal_state, FIELD_BUDGETS["goal_state"]),
            "references": _clip_json(_safe_copy(references) or {},
                                     FIELD_BUDGETS["references"]),
            "memory_hints": _clip(memory_hints, FIELD_BUDGETS["memory_hints"]),
            "editor_text": _clip(editor_text, FIELD_BUDGETS["editor_text"]),
            "browser_target": _clip(browser_target, FIELD_BUDGETS["browser_target"]),
            "capture_identity": _clip(capture_identity, FIELD_BUDGETS["capture_identity"]),
        }
        #: Concrete, frozen identities behind the references (F46): window,
        #: editor, browser target and screen capture.
        self.identities = _freeze(_safe_copy(identities) or {})
        self.capture = _freeze(_safe_copy(capture) or {})
        #: Deterministic net replay flag (G8 migration: the screen-question
        #: safety net must fire identically in orchestrator mode).
        self.screen_question = bool(screen_question)
        #: Omission records (field/reason/retrieve) — set by render()/to_json().
        self.omission_records = []
        #: Backwards-compatible list of omitted field names.
        self.omissions = []
        object.__setattr__(self, "_frozen", True)

    # ── immutability ──────────────────────────────────────────────────────
    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False) and name in self._LOCKED:
            raise AttributeError(
                "ContextEnvelope is an immutable request snapshot (%s)" % (name,))
        object.__setattr__(self, name, value)

    # ── one snapshot, many readers ────────────────────────────────────────
    @property
    def utterance(self):
        return self.fields.get("utterance", "")

    @property
    def goal_state(self):
        return self.fields.get("goal_state", "")

    @property
    def references(self):
        return _plain(self.fields.get("references"))

    @property
    def memory_hints(self):
        return self.fields.get("memory_hints", "")

    @property
    def editor_text(self):
        return self.fields.get("editor_text", "")

    @property
    def browser_target(self):
        return self.fields.get("browser_target", "")

    @property
    def capture_identity(self):
        return self.fields.get("capture_identity", "")

    @property
    def selected_text(self):
        editor = self.identities.get("editor") or {}
        return editor.get("selected_text") or ""

    @property
    def retrieval_hints(self):
        """field -> how to retrieve it, for everything currently omitted."""
        return {record["field"]: record["retrieve"]
                for record in self.omission_records}

    def resolve_reference(self, phrase="this"):
        """Resolve a deictic phrase to the SNAPSHOT's concrete identity.

        F46 acceptance: "fix this / use that tab" must resolve consistently
        despite focus changes. Whatever the conversation refers to is decided
        once, at request time, and read from here afterwards.

        Returns ``{"kind", "identity", "retrieve"}`` — *identity* is empty
        when the request genuinely captured nothing (never a guess).
        """
        order = ("browser_target", "active_window", "editor", "screen")
        for kind in order:
            identity = self.identities.get(kind)
            if identity:
                return {"kind": kind, "identity": _plain(identity),
                        "retrieve": RETRIEVAL_HINTS.get(
                            "browser_target" if kind == "browser_target" else "references", "")}
        return {"kind": None, "identity": {},
                "retrieve": RETRIEVAL_HINTS["references"]}

    def for_consumer(self, consumer):
        """The bounded view *consumer* reads — built from THIS snapshot only.

        Every consumer gets the same frozen ``identities``; only the field
        selection differs, so two consumers can never disagree about which tab
        or file the request was about.
        """
        name = str(consumer or "").strip().lower()
        names = CONSUMER_FIELDS.get(name) or tuple(FIELD_BUDGETS)
        view = {field: _plain(self.fields[field])
                for field in names
                if self.fields.get(field)}
        return {
            "consumer": name or "default",
            "identities": _plain(self.identities),
            "capture": _plain(self.capture),
            "screen_question": self.screen_question,
            "fields": view,
            "retrieval_hints": dict(self.retrieval_hints),
        }

    # ── bounded serialization ─────────────────────────────────────────────
    def _omission_record(self, field, reason):
        return {
            "field": field,
            "reason": reason,
            "retrieve": RETRIEVAL_HINTS.get(
                field, RETRIEVAL_HINTS.get("references", "")),
        }

    def _compose(self, view, omissions):
        lines = ["ContextEnvelope (%s):" % (
            "screen-question" if self.screen_question else "general")]
        for name, value in view.items():
            lines.append("%s: %s" % (
                name, value if isinstance(value, str)
                else json.dumps(_plain(value), ensure_ascii=False, default=str)))
        if omissions:
            lines.append(_OMISSION_MARK % "; ".join(
                "%s — retrieve with %s" % (record["field"], record["retrieve"])
                for record in omissions))
        return "\n".join(lines)

    def _references_summary(self, references):
        if isinstance(references, (MappingProxyType, dict)):
            return {"_note": "too large to render",
                    "keys": _bounded_keys(references)}
        return _NOT_RENDERED

    def _bounded_view(self):
        """``(view, omission_records)`` — every field within TOTAL_BUDGET."""
        references = self.fields.get("references")
        view = {k: v for k, v in self.fields.items() if v}
        omissions = []
        text = self._compose(view, omissions)
        for dropped in _DROP_ORDER:
            if len(text) <= TOTAL_BUDGET:
                break
            if dropped in view:
                del view[dropped]
                omissions.append(self._omission_record(
                    dropped, "whole field dropped to fit the total budget"))
                text = self._compose(view, omissions)
        if len(text) > TOTAL_BUDGET and "references" in view:
            original = references if isinstance(references, dict) else {}
            view["references"] = self._references_summary(original)
            omissions.append(self._omission_record(
                "references.detail", "oversized references summarized"))
            text = self._compose(view, omissions)
        return view, omissions

    def render(self):
        """Bounded plain-text view for an LLM prompt.

        Applies TOTAL_BUDGET by dropping optional whole fields (explicitly
        listed in omissions, each with its retrieval hint) — never by halving
        JSON or clipping silently. The returned text is ALWAYS at most
        TOTAL_BUDGET characters, whatever the input contained.
        """
        view, omissions = self._bounded_view()
        text = self._compose(view, omissions)
        if len(text) > TOTAL_BUDGET:
            # Pathological escape/key content: the hard clamp is the final
            # guarantee (this view is line-oriented text, not JSON).
            text = _hard_clamp(text, TOTAL_BUDGET)
        self.omission_records = list(omissions)
        self.omissions = [record["field"] for record in omissions]
        return text

    def _json_text(self, view, omissions):
        return json.dumps({
            "envelope": _plain(view),
            "omissions": [dict(record) for record in omissions],
            "screen_question": self.screen_question,
        }, ensure_ascii=False, default=str)

    def to_json(self):
        """The whole snapshot as VALID, budgeted JSON (F46: "no truncated
        JSON"; "long keys/escaping cannot exceed the total bound")."""
        view, omissions = self._bounded_view()
        text = self._json_text(view, omissions)
        for dropped in _JSON_LAST_RESORT:
            if len(text) <= TOTAL_BUDGET:
                break
            if dropped in view:
                del view[dropped]
                omissions.append(self._omission_record(
                    dropped, "whole field dropped to fit the JSON budget"))
                text = self._json_text(view, omissions)
        # Last guarantee: clip the (string) utterance VALUE until the JSON
        # fits. The document stays parseable; the value carries its marker.
        guard = 0
        while len(text) > TOTAL_BUDGET and guard < 64:
            guard += 1
            current = view.get("utterance")
            if not isinstance(current, str) or not current:
                text = budgeted_json(_plain(view), TOTAL_BUDGET)
                break
            over = len(text) - TOTAL_BUDGET
            room = len(current) - over - 8
            if room <= 0:
                view["utterance"] = _TRUNC_MARK
                omissions.append(self._omission_record(
                    "utterance.detail", "utterance clipped to fit the JSON budget"))
            else:
                view["utterance"] = _clip(current, room)
            text = self._json_text(view, omissions)
        if len(text) > TOTAL_BUDGET:
            text = budgeted_json(_plain(view), TOTAL_BUDGET)
        self.omission_records = list(omissions)
        self.omissions = [record["field"] for record in omissions]
        return text

    def snapshot(self):
        """The immutable identity-rich dict every consumer serializes from."""
        return {
            "utterance": self.utterance,
            "screen_question": self.screen_question,
            "fields": _plain(self.fields),
            "identities": _plain(self.identities),
            "capture": _plain(self.capture),
        }

    def to_system_message(self):
        """The envelope as a system message for the orchestrator loop."""
        return {
            "role": "system",
            "content": (
                "You are Jarvis's orchestrator. One bounded context view follows.\n"
                "Retrieve ADDITIONAL context through tools; never guess at entities.\n\n"
                + self.render()
            ),
        }


def _active_window_reference(windows):
    """(reference scalars, identity) for the active window."""
    active = (windows or {}).get("active_window") or {}
    title = str(active.get("title") or "").strip()
    identity = {
        "title": _clip(title, 200),
        "hwnd": active.get("hwnd"),
        "process_id": active.get("process_id") or active.get("pid"),
        "process": active.get("process_name") or active.get("process"),
    }
    identity = {k: v for k, v in identity.items() if v not in (None, "")}
    return title, identity


def _editor_reference(connectors, selected_text):
    """(reference scalars, identity) for the editor bridge snapshot."""
    editor = (connectors or {}).get("editor") or {}
    if not editor.get("available"):
        return {}, {}
    state = editor.get("state") or {}
    active_file = state.get("activeFile") or {}
    path = str(active_file.get("path") or active_file.get("fileName") or "").strip()
    version = active_file.get("version")
    selection = selected_text or state.get("selectedText") or active_file.get("selectedText") or ""
    identity = {
        "path": _clip(path, 300),
        "version": version,
        "selected_text": _clip(selection, FIELD_BUDGETS["editor_text"]),
        "workspace": [
            _clip(str(folder), 160)
            for folder in (state.get("workspaceFolders") or [])[:4]
        ],
    }
    identity = {k: v for k, v in identity.items() if v not in (None, "", [])}
    reference = {}
    if path:
        reference["editor_file"] = _clip(path, 300)
    if version is not None:
        reference["editor_version"] = version
    if selection:
        reference["editor_selection"] = _clip(str(selection), 300)
    return reference, identity


def _browser_reference(browser):
    """(reference scalars, identity) for the concrete browser target.

    F46: the tab identity (instance id + tab id + url) is what a follow-up
    like "use that tab" must resolve to — a bare title list cannot survive a
    focus change, and a duplicate title is ambiguous.
    """
    browser = browser or {}
    if not browser.get("available"):
        return {}, {}
    tabs = [t for t in (browser.get("tabs") or []) if isinstance(t, dict)]
    titles = []
    for tab in tabs:
        label = str(tab.get("title") or tab.get("url") or "").strip()
        if label:
            titles.append(label)
    target = None
    for tab in tabs:
        if tab.get("active") or tab.get("isActive"):
            target = tab
            break
    if target is None and tabs:
        target = tabs[0]
    identity = {}
    if target is not None:
        identity = {
            "tab_id": target.get("id") or target.get("tab_id") or target.get("target_id"),
            "url": _clip(str(target.get("url") or ""), 400),
            "title": _clip(str(target.get("title") or ""), 200),
            "instance_id": (browser.get("instance_id")
                            or browser.get("connection_id")
                            or browser.get("session_id")),
        }
        identity = {k: v for k, v in identity.items() if v not in (None, "")}
    reference = {"browser_tabs": titles[:8]}
    if identity:
        reference["browser_target"] = identity
    return reference, identity


def _capture_identity(capture):
    """Concrete capture identity: what was looked at, when, and in which frame."""
    if not capture:
        return {}, ""
    identity = {
        "capture_mode": capture.get("capture_mode") or capture.get("mode"),
        "monitor": capture.get("monitor") or capture.get("monitor_index"),
        "hwnd": capture.get("hwnd"),
        "region": capture.get("region"),
        "bounds": capture.get("bounds"),
        "source_id": capture.get("source_id") or capture.get("target_id"),
        "observed_at": capture.get("observed_at") or capture.get("captured_at"),
        "epoch": capture.get("epoch"),
    }
    identity = {k: v for k, v in identity.items() if v not in (None, "", {}, [])}
    compact = "; ".join("%s=%s" % (k, v) for k, v in identity.items())
    return identity, _clip(compact, FIELD_BUDGETS["capture_identity"])


def build_envelope(utterance, history=None, connectors=None, capture=None,
                   memory_hints=None, screen_question=False, selected_text=None):
    """Gather one request's IMMUTABLE envelope from already-collected context.

    *history* is the recent conversation (dicts with user/assistant keys or
    plain strings); *connectors* is a task_agent-style gather_context()
    snapshot ({windows, editor, browser}) already collected by the caller;
    *capture* is the screen capture identity for cross-modal follow-ups;
    *memory_hints* are retrieved personal-memory lines; *selected_text* is the
    editor selection when the caller already has it.

    Everything is deep-copied and BUDGETED here, once, so every consumer reads
    the same picture — this never serializes raw connector dumps.
    """
    goal_state = []
    for turn in list(history or [])[-3:]:
        if isinstance(turn, dict):
            user = str(turn.get("user") or turn.get("message") or "").strip()
            reply = str(turn.get("assistant") or turn.get("response") or "").strip()
            if user:
                goal_state.append("user: %s" % _clip(user, 400))
            if reply:
                goal_state.append("jarvis: %s" % _clip(reply, 400))
        elif str(turn).strip():
            goal_state.append(_clip(str(turn), 400))

    connectors = _safe_copy(connectors) or {}
    references = {}
    identities = {}

    window_ref, window_identity = _active_window_reference(connectors.get("windows"))
    if window_ref:
        references["active_window"] = window_ref
    if window_identity:
        identities["active_window"] = window_identity

    editor_ref, editor_identity = _editor_reference(connectors, selected_text)
    references.update(editor_ref)
    if editor_identity:
        identities["editor"] = editor_identity

    browser_ref, browser_identity = _browser_reference(connectors.get("browser"))
    references.update(browser_ref)
    if browser_identity:
        identities["browser_target"] = browser_identity

    capture_identity_dict, capture_identity = _capture_identity(
        _safe_copy(capture) if capture else None)
    if capture_identity_dict:
        identities["screen"] = capture_identity_dict
        references["capture_identity"] = capture_identity_dict

    # A compact, deterministic one-line browser target so a follow-up that
    # only reads the string field still sees the same concrete tab.
    browser_target = ""
    if browser_identity:
        browser_target = "; ".join(
            "%s=%s" % (k, browser_identity[k])
            for k in ("instance_id", "tab_id", "title", "url")
            if browser_identity.get(k))

    editor_text = selected_text or (editor_identity.get("selected_text") or "")

    return ContextEnvelope(
        utterance=utterance,
        goal_state="\n".join(goal_state),
        references=references,
        memory_hints=memory_hints,
        editor_text=editor_text,
        browser_target=browser_target,
        capture_identity=capture_identity,
        screen_question=screen_question,
        identities=identities,
        capture=capture,
    )
