"""Editor bridge connector.

The bridge is a small VS Code-compatible extension that exposes editor state on
localhost. Without that bridge Jarvis can still infer the active editor window,
but it cannot read buffers, selections, diagnostics, or command IDs.

F15 (Fable-5 audit, G4): beyond the original state snapshot and command/exec
helpers, this client exposes the structured coding interface — buffer reads
with document versions, diagnostics, symbols, references, workspace search,
test results, and version-checked WorkspaceEdit operations. Edits carry the
document URI/version/range, and the active selection is never treated as a
stable edit target across a confirmation delay.
"""

import os

import requests


BRIDGE_URL = os.getenv("JARVIS_EDITOR_BRIDGE_URL", "http://127.0.0.1:8765").rstrip("/")
BRIDGE_TOKEN = os.getenv("JARVIS_EDITOR_BRIDGE_TOKEN", "")


def _headers():
    if not BRIDGE_TOKEN:
        return {}
    return {"X-Jarvis-Token": BRIDGE_TOKEN}


EDITOR_TITLE_MARKERS = (
    "antigravity",
    "visual studio code",
    "vs code",
    "cursor",
    "windsurf",
    "code.exe",
)


def looks_like_editor_window(title):
    normalized = (title or "").lower()
    return any(marker in normalized for marker in EDITOR_TITLE_MARKERS)


def is_available():
    try:
        response = requests.get(f"{BRIDGE_URL}/health", headers=_headers(), timeout=0.5)
        return response.status_code == 200
    except Exception:
        return False


def _get_json(path, timeout=0.8):
    try:
        response = requests.get(f"{BRIDGE_URL}{path}", headers=_headers(), timeout=timeout)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {}


def _post_json(path, payload, timeout=2.0):
    try:
        response = requests.post(
            f"{BRIDGE_URL}{path}",
            json=payload,
            headers=_headers(),
            timeout=timeout,
        )
        try:
            data = response.json()
        except Exception:
            data = None
        if response.status_code == 200:
            return data if isinstance(data, dict) else {"ok": True, "result": data}
        # Non-200: keep the structured body when there is one (F15 — a 409
        # version mismatch carries expected/actual versions the caller needs).
        if isinstance(data, dict):
            data.setdefault("ok", False)
            data.setdefault("error", (response.text or "")[:300])
            data["status"] = response.status_code
            return data
        return {"ok": False, "error": (response.text or "")[:300],
                "status": response.status_code}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def snapshot(active_window_title=""):
    available = is_available()
    if available:
        state = _get_json("/state", timeout=1.0)
    else:
        state = {}

    return {
        "connector": "editor_bridge",
        "available": available,
        "endpoint": BRIDGE_URL,
        "active_window_looks_like_editor": looks_like_editor_window(active_window_title),
        "state": state,
        "capabilities": [
            "editor.inspect_workspace" if available else "editor.bridge_required",
            "editor.execute_command" if available else "editor.commands_unavailable",
            "editor.open_file" if available else "editor.files_unavailable",
            # F15 (G4): structured coding interface.
            "editor.read_buffer" if available else "editor.buffers_unavailable",
            "editor.diagnostics" if available else "editor.diagnostics_unavailable",
            "editor.symbols" if available else "editor.symbols_unavailable",
            "editor.references" if available else "editor.references_unavailable",
            "editor.workspace_search" if available else "editor.search_unavailable",
            "editor.test_results" if available else "editor.tests_unavailable",
            "editor.apply_workspace_edit" if available else "editor.edits_unavailable",
            "editor.edit_active_selection" if available else "editor.edits_unavailable",
        ],
        "setup_hint": (
            ""
            if available
            else "Install and start the Jarvis editor bridge extension in Antigravity/VS Code-compatible editors."
        ),
    }


def execute_command(command, args=None):
    return _post_json("/execute-command", {"command": command, "args": args or []})


def open_file(path):
    return _post_json("/open-file", {"path": path})


# ── F15 (Fable-5 audit, G4): fail-closed edit preconditions ───────────────
#
# The audit found three ways an edit could land on the wrong bytes: the
# active-selection check omitted the document version (so a check/use race
# could replace whatever the user had focused in the meantime), a text-only
# edit POST carried no version at all, and expected versions were optional.
#
# This client therefore refuses such a request LOCALLY, before any request is
# made: an edit must name its document (path or uri), carry the document
# version it was planned against and, for a selection edit, the explicit
# range. Multi-file WorkspaceEdits additionally need a per-document version
# for every member, so one stale member rejects the whole edit. The
# extension-side check/apply remains non-atomic — see
# ``integrations/jarvis-editor-bridge/extension.js`` (not owned by this
# module): it must verify every version immediately before ``applyEdit``
# under one lock for full closure.


def _refuse_edit(tool, reason, **extra):
    """A uniform, structured refusal. Nothing was requested for this one."""
    payload = {"ok": False, "error": reason, "reason": reason,
               "tool": tool, "refused": "missing_precondition",
               "external_effect": False}
    payload.update(extra)
    return payload


def _int_or_none(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _document_key(edit):
    return str(edit.get("uri") or edit.get("path") or "").strip()


def _edit_version(edit):
    """The version a single edit was planned against (F15: never optional)."""
    for key in ("expectedVersion", "version"):
        if key in edit:
            return _int_or_none(edit.get(key))
    return None


def _edit_range(value):
    """Normalize an explicit range, or None when the edit carried none."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("range must be an object")
    out = {}
    for side in ("start", "end"):
        point = value.get(side)
        if not isinstance(point, dict):
            raise ValueError("range.%s is missing" % side)
        line = _int_or_none(point.get("line"))
        character = _int_or_none(point.get("character"))
        if line is None or character is None:
            raise ValueError("range.%s needs numeric line/character" % side)
        out[side] = {"line": line, "character": character}
    return out


def _prepare_edits(edits, expected_version=None):
    """(payload_edits, top_level_version, error). F15: fail closed.

    Every edit must name its document and the version that document was read
    at. When several different documents are edited at once, a per-document
    version is mandatory — the top-level ``expectedVersion`` is only a
    convenience for a SINGLE-document edit, never a blanket precondition for a
    multi-file one.
    """
    if not isinstance(edits, (list, tuple)) or not edits:
        return [], None, "a workspace edit needs at least one edit"
    declared = _int_or_none(expected_version)
    documents = {_document_key(edit) for edit in edits
                 if isinstance(edit, dict) and _document_key(edit)}
    same_document = len(documents) <= 1
    prepared = []
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict):
            return [], None, "edit %d is not an object" % index
        path = str(edit.get("path") or "").strip()
        uri = str(edit.get("uri") or "").strip()
        if not path and not uri:
            return [], None, ("edit %d names no document (path or uri)"
                              % index)
        if "newText" not in edit and "new_text" not in edit:
            return [], None, ("edit %d carries no newText/new_text, so it "
                              "could not change anything" % index)
        version = _edit_version(edit)
        if version is None:
            if not same_document:
                return [], None, (
                    "edit %d (%s) does not carry the document version it was "
                    "planned against; a multi-file edit needs a per-document "
                    "precondition, so nothing was changed"
                    % (index, _document_key(edit) or "unknown"))
            version = declared
        if version is None:
            return [], None, (
                "edit %d does not carry the document version it was read at "
                "(read it with the structured inspection path first); nothing "
                "was changed" % index)
        if declared is not None and declared != version:
            return [], None, (
                "edit %d expects version %s but the caller declared %s; the "
                "preconditions disagree, so nothing was changed"
                % (index, version, declared))
        try:
            rng = _edit_range(edit.get("range"))
        except ValueError as exc:
            return [], None, "edit %d: %s" % (index, exc)
        new_text = edit.get("newText")
        if new_text is None:
            new_text = edit.get("new_text")
        payload = {"path": path, "uri": uri, "newText": str(new_text),
                   "expectedVersion": version}
        if rng is not None:
            payload["range"] = rng
        prepared.append(payload)
    return prepared, declared, ""


def edit_active_selection(text, uri=None, path=None, version=None,
                          selection=None, timeout=2.5):
    """Replace one explicit, version-checked document range (F15).

    This used to POST ``{"text": ...}`` to ``/edit-active-selection``, which
    the extension applies to *whatever is focused right now*: no document
    identity, no version and no range, so a check/use race silently edited
    the wrong place. Now the caller must supply the URI/path, the document
    version captured at read time and the explicit selection range; anything
    less is refused here, without a request. A live state re-check runs
    immediately before the edit so a focus switch (or a change at the same
    position) is refused rather than applied.
    """
    text = "" if text is None else str(text)
    uri = str(uri or "").strip()
    path = str(path or "").strip()
    edit_version = _int_or_none(version)
    if not uri and not path:
        return _refuse_edit("editor.edit_active_selection",
                            "the document uri (or path) is required: an edit "
                            "is addressed to a document, never to whatever "
                            "happens to be focused")
    if edit_version is None:
        return _refuse_edit("editor.edit_active_selection",
                            "the document version is required: read the "
                            "buffer with the structured inspection path and "
                            "edit the version you actually saw")
    try:
        rng = _edit_range(selection)
    except ValueError as exc:
        return _refuse_edit("editor.edit_active_selection",
                            "an explicit selection range is required: %s" % exc)
    if rng is None:
        return _refuse_edit("editor.edit_active_selection",
                            "an explicit selection range is required: an "
                            "edit must name the exact range it replaces")

    edit = {"path": path, "uri": uri, "range": rng, "newText": text,
            "expectedVersion": edit_version}
    if not active_selection_matches(expected_uri=uri or path,
                                    expected_selection=rng,
                                    expected_version=edit_version,
                                    timeout=timeout):
        return _refuse_edit(
            "editor.edit_active_selection",
            "the editor state moved on (a focus switch, a missing version or "
            "a change at the same position): re-read the document and target "
            "the buffer you actually saw",
            expectedVersion=edit_version)
    # /apply-edit is the version-checked endpoint: the unversioned
    # /edit-active-selection POST is never used by this client.
    return _post_json("/apply-edit",
                      {"edits": [edit], "expectedVersion": edit_version},
                      timeout=timeout)


# ── F15 (Fable-5 audit, G4): structured coding interface ──────────────────
#
# The functions below expose the structured coding interface: buffer reads
# (with document versions), diagnostics, symbols, references, workspace
# search, test results, and version-checked WorkspaceEdit operations. Every
# function degrades to a uniform ``{"ok": False, ...}`` when the bridge is
# offline so the planning loop can fall back honestly.


def _unwrap(result, key_hint=""):
    """Normalize a bridge response into the ok/content convention."""
    if not isinstance(result, dict):
        return {"ok": False, "error": "no response from editor bridge"}
    if result.get("ok") is False and not result.get("error"):
        result = dict(result)
        result["error"] = "editor bridge rejected the request"
    return result


def read_buffer(path=None, uri=None, start_line=None, end_line=None,
                timeout=1.5):
    """Structured buffer read.

    Returns ``{ok, path, uri, version, lineCount, startLine, endLine, text}``
    — *version* is the document version a later edit must present to be
    accepted (see :func:`apply_workspace_edit`). When *path* is omitted the
    active editor's buffer is returned; ranges are 1-based inclusive.
    """
    payload = {"path": path or "", "uri": uri or ""}
    if start_line is not None:
        payload["startLine"] = int(start_line)
    if end_line is not None:
        payload["endLine"] = int(end_line)
    return _unwrap(_post_json("/read-buffer", payload, timeout=timeout))


def diagnostics(path=None, timeout=1.5):
    """Structured diagnostics, optionally filtered to one file."""
    return _unwrap(_post_json("/diagnostics", {"path": path or ""},
                              timeout=timeout))


def symbols(path=None, timeout=2.5):
    """Document symbols for *path* (or the active buffer)."""
    return _unwrap(_post_json("/symbols", {"path": path or ""},
                              timeout=timeout))


def references(path=None, line=None, character=None, timeout=2.5):
    """References to the symbol at (*line*, *character*) in *path*."""
    payload = {"path": path or ""}
    if line is not None:
        payload["line"] = int(line)
    if character is not None:
        payload["character"] = int(character)
    return _unwrap(_post_json("/references", payload, timeout=timeout))


def workspace_search(query, limit=50, timeout=2.5):
    """Text search across the workspace folders."""
    return _unwrap(_post_json(
        "/workspace-search",
        {"query": query or "", "limit": int(limit or 50)},
        timeout=timeout))


def test_results(command=None, cwd=None, timeout_ms=120000):
    """Run the workspace's test command through the bridge and return
    structured results (exit code + bounded output)."""
    payload = {}
    if command:
        payload["command"] = command
    if cwd:
        payload["cwd"] = cwd
    if timeout_ms:
        payload["timeoutMs"] = int(timeout_ms)
    return _unwrap(_post_json("/test-results", payload,
                              timeout=max(5.0, timeout_ms / 1000.0 + 5.0)))


def apply_workspace_edit(edits, expected_version=None, timeout=2.5):
    """Version-checked WorkspaceEdit with per-document preconditions (F15).

    *edits* is a list of ``{path|uri, range, new_text, version}`` where
    *range* is ``{start: {line, character}, end: {line, character}}`` and
    *version* (``expectedVersion`` is accepted too) is the document version
    the edit was planned against. Every member needs its own version when the
    edit spans several documents — the top-level *expected_version* is only
    accepted for a single-document edit. A precondition that is missing or
    contradicts the declaration is refused here, without a request, and a
    stale member makes the extension refuse the WHOLE edit (409) before
    anything is applied.
    """
    prepared, declared, error = _prepare_edits(edits, expected_version)
    if error:
        return _refuse_edit("editor.apply_workspace_edit", error)
    payload = {"edits": prepared}
    if declared is not None:
        payload["expectedVersion"] = declared
    return _unwrap(_post_json("/apply-edit", payload, timeout=timeout))


def active_selection_matches(expected_uri=None, expected_selection=None,
                             expected_version=None, timeout=0.8):
    """F15 — is the *plan-time* target still the live one?

    A confirmation can sit in the gate for tens of seconds; the user may
    click elsewhere, keep typing (same position, new content) or close the
    editor in the meantime. Before replacing "the current selection" the
    agent re-fetches the state and refuses when any of the identity, the
    document version or the exact range has drifted. Missing state — no
    active file, or a document that reports no version — is also a refusal:
    an edit that cannot be pinned is never applied.
    """
    state = _get_json("/state", timeout=timeout)
    active = state.get("activeFile") if isinstance(state, dict) else None
    if not isinstance(active, dict) or not active:
        return False
    if expected_uri and active.get("uri") != expected_uri \
            and active.get("path") != expected_uri:
        return False
    current_version = _int_or_none(active.get("version"))
    if current_version is None:
        return False
    if expected_version is not None \
            and current_version != _int_or_none(expected_version):
        return False
    if expected_selection:
        current = active.get("selection") or {}
        for side in ("start", "end"):
            want = (expected_selection.get(side) or {})
            got = (current.get(side) or {})
            if (want.get("line"), want.get("character")) != (
                    got.get("line"), got.get("character")):
                return False
    return True
