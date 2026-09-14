# Jarvis Editor Bridge

This VS Code-compatible extension gives Jarvis structured editor access over
`http://127.0.0.1:8765`.

It exposes:

- active file path, language, full text, selection, document version, and visible ranges
- workspace folders
- structured diagnostics (severity, code, source, range)
- buffer reads with document versions and 1-based line ranges
- document symbols, references, bounded workspace text search
- command execution through VS Code command IDs
- version-checked, document-addressed edits (`/apply-edit`, `/edit-active-selection`)
- file open and structured test runs

Jarvis uses this bridge before falling back to UI Automation, OCR, or
screenshots. In Antigravity, install this as a local/unpacked extension if the
editor supports VS Code-style extensions.

## Authentication (required, fail closed)

F15 changed authentication from "fail open" to "fail closed":

- **A token must be configured.** If `JARVIS_EDITOR_BRIDGE_TOKEN` is unset or
  empty, the bridge refuses **every** request — including `GET /health` — with
  `401`. There is no unauthenticated mode.
- **The only accepted carrier is the `X-Jarvis-Token` request header.** The
  Python client sends exactly that (see
  `backend/services/task_agent/connectors/editor_bridge.py`). The old
  `?token=` query-string path has been **removed**: a token in the URL leaks
  into shell history, access logs and proxies.
- **The comparison is constant-time** (`crypto.timingSafeEqual` over
  equal-length buffers); a length mismatch is refused without leaking bytes.
- The token is re-read from the environment on every request, so a bridge can
  also be started before the variable is set.

Both sides must share the value, and **VS Code must be launched with the
variable in its environment** (the extension host inherits the environment of
the VS Code process):

```powershell
# PowerShell, before starting VS Code / Antigravity:
$env:JARVIS_EDITOR_BRIDGE_TOKEN = "a-long-random-shared-secret"
code .
```

```bash
# .env for the Python backend (same value):
JARVIS_EDITOR_BRIDGE_TOKEN=a-long-random-shared-secret
```

If the token is missing, the extension logs a warning in the editor and
`GET /health` returns `401`, so Jarvis honestly reports the bridge as
unavailable instead of silently working unauthenticated.

## Endpoints

All endpoints require the `X-Jarvis-Token` header. `POST` bodies are JSON.

- `GET /health` → `{ ok, service, port }` (liveness; also authenticated)
- `GET /state` → active file (`path`, `uri`, `languageId`, `lineCount`,
  `version`, `selection`, `selectionText`, `text`, `visibleRanges`), workspace
  folders, diagnostics, window focus. `version` is the document version an edit
  must present to be accepted.
- `POST /execute-command` with `{ "command": "...", "args": [] }`
- `POST /open-file` with `{ "path": "C:\\path\\file.py" }`
- `POST /write-file` with `{ "path": "C:\\path\\file.py", "content": "..." }` —
  a raw, **unversioned** write straight to disk. It is not a document-addressed
  edit and no agent edit path uses it; prefer `/apply-edit`.

### Structured coding interface (F15)

- `POST /read-buffer` with `{ "path" | "uri", "startLine"?, "endLine"? }` (1-based, inclusive) →
  `{ ok, path, uri, version, lineCount, startLine, endLine, text, textTruncated }`. `version` is
  the document version an edit must present to be accepted.
- `POST /diagnostics` with `{ "path"?, "limit"? }` → `{ ok, diagnostics: [{ file, message,
  severity, range, source, code }] }` (`severity`/`code` stay structured).
- `POST /symbols` with `{ "path" | "uri" }` → flattened document symbols with ranges.
- `POST /references` with `{ "path", "line", "character" }` (1-based) → reference locations.
- `POST /workspace-search` with `{ "query", "limit"? }` → bounded text search across the workspace.
- `POST /test-results` with `{ "command"?, "cwd"?, "timeoutMs"? }` → runs the workspace test
  command (`CI=true`, bounded output) and reports exit code + output.

#### Edit protocol

An edit is always addressed to a **document**, at the **version it was read
at**, and (for a selection edit) at an **explicit range**. Nothing is inferred
from "whatever is focused right now".

`POST /apply-edit` with:

```json
{
  "edits": [
    { "path": "C:\\proj\\a.py", "uri": "file:///c%3A/proj/a.py",
      "expectedVersion": 7, "newText": "replacement",
      "range": { "start": { "line": 1, "character": 0 },
                 "end":   { "line": 1, "character": 4 } } }
  ],
  "expectedVersion": 7
}
```

- Each edit names its document (`path` and/or `uri`; the two forms are matched
  as one file, including VS Code's `%3A` drive encoding), carries `newText`
  (or `new_text`; may be empty for a deletion) and the document version it was
  planned against (`expectedVersion` or `version`).
- **A per-document version is mandatory for a multi-document edit.** The
  top-level `expectedVersion` is accepted only as a convenience for a
  single-document edit, and it must not contradict any per-edit version. Edits
  that share one document must agree on its version.
- `range` is 0-based VS Code style `{ start: { line, character }, end: { line, character } }`;
  omit it to replace the whole document.
- **Every target document must already be open in the workspace.** A version
  precondition on a file the editor is not tracking would be meaningless, so a
  closed document is refused instead of silently reloaded (`/open-file` or
  `/read-buffer` establish an open, versioned target first).
- The extension holds **one apply lock**: all documents are resolved and
  re-verified, then the whole `WorkspaceEdit` is built and handed to
  `vscode.workspace.applyEdit` with no `await` in between. Concurrent applies
  are serialized, so two requests cannot each verify version N and then both
  apply.
- Responses: `200` `{ ok: true, message, newVersions: [{ path, version }] }` on
  success; `409` `{ ok: false, status, error, path, expectedVersion,
  actualVersion }` when a member is stale or a range is out of bounds — **one
  stale member rejects the whole edit and nothing is applied**; `400` when the
  request itself is malformed (no document, no `newText`, no version, missing
  per-document versions on a multi-file edit, contradictory versions).

`POST /edit-active-selection` with:

```json
{ "path": "C:\\proj\\a.py", "expectedVersion": 7,
  "range": { "start": { "line": 1, "character": 0 },
             "end":   { "line": 1, "character": 4 } },
  "newText": "replacement" }
```

- `text` is accepted as an alias for `newText`, and `selection` as an alias for
  `range`.
- The **named** document is resolved (never the active editor) and must be
  open; the live version must equal `expectedVersion`; the range must be inside
  the document (line and character bounds are both checked).
- `409` + **zero edits** when the document is not open, the version differs, or
  the range is missing/out of bounds; `400` when the identity or version is
  missing entirely.
- The Python client uses the version-checked `/apply-edit` path for selection
  edits and additionally re-checks `/state` immediately before, so a focus
  switch, a same-position content change or a missing version results in no
  request (or a refusal) rather than a misplaced edit.

The server binds only to `127.0.0.1`.

## Tests

The extension needs the `vscode` module, which does not exist outside an
extension host, so the protocol decisions are pure exported functions and the
HTTP router is exercised directly under plain Node with a small `vscode`
stand-in:

```bash
node --test integrations/jarvis-editor-bridge/test/extension-contract.test.js
```
