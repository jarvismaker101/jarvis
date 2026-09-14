const http = require("http");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const vscode = require("vscode");

const HOST = "127.0.0.1";
const PORT = Number(process.env.JARVIS_EDITOR_BRIDGE_PORT || "8765");
const TEXT_LIMIT = Number(process.env.JARVIS_EDITOR_TEXT_LIMIT || "200000");

// ── F15: authentication (fail CLOSED) ────────────────────────────────────
//
// The bridge used to fail OPEN when JARVIS_EDITOR_BRIDGE_TOKEN was unset, also
// accepted the token from a `?token=` query string (which leaks into logs,
// shells and histories) and compared with `===`. Now: a token must be
// configured or every request is refused; the only accepted carrier is the
// X-Jarvis-Token header the Python client sends (editor_bridge.py:24-27); and
// the compare is constant-time over equal-length buffers.
const AUTH_HEADER = "x-jarvis-token";
const OK = 200;
const BAD_REQUEST = 400;
const CONFLICT = 409;

function configuredToken() {
  return String(process.env.JARVIS_EDITOR_BRIDGE_TOKEN || "");
}

function timingSafeTokenEquals(provided, expected) {
  const offered = Buffer.from(provided === null || provided === undefined ? "" : String(provided), "utf8");
  const wanted = Buffer.from(expected === null || expected === undefined ? "" : String(expected), "utf8");
  if (!wanted.length || offered.length !== wanted.length) {
    // Length is compared first because timingSafeEqual throws on a length
    // mismatch; only the length (not the bytes) is revealed.
    return false;
  }
  return crypto.timingSafeEqual(offered, wanted);
}

function authorizeRequest(headers, token) {
  if (!String(token || "")) {
    // Fail closed: an unconfigured bridge is never an unauthenticated bridge.
    return false;
  }
  const offered = headers ? headers[AUTH_HEADER] : "";
  if (typeof offered !== "string" || !offered) {
    return false;
  }
  return timingSafeTokenEquals(offered, token);
}

function isAuthorized(req) {
  return authorizeRequest(req.headers, configuredToken());
}

let server = null;

// A refusal always carries the HTTP status the router must use, so a caller
// can tell "you asked wrongly" (400) from "the editor moved on" (409).
function refusal(status, error, extra) {
  const out = { ok: false, status, error };
  if (extra) {
    Object.assign(out, extra);
  }
  return out;
}

function statusOf(result) {
  const status = Number(result && result.status);
  return Number.isInteger(status) && status >= 400 && status <= 599 ? status : OK;
}

function sendJson(res, status, value) {
  const body = JSON.stringify(value);
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(body),
  });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
      if (body.length > 1024 * 1024) {
        reject(new Error("Request body too large"));
        req.destroy();
      }
    });
    req.on("end", () => {
      if (!body.trim()) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(body));
      } catch (error) {
        reject(error);
      }
    });
    req.on("error", reject);
  });
}

// ── F15: edit preconditions (pure decisions, no `vscode` dependency) ─────
//
// Every decision an edit depends on — document identity, expected version,
// explicit range, bounds, and "is any member stale?" — is made here, by
// functions that take plain data and return a plain verdict. The async
// handlers below only resolve TextDocuments and call applyEdit. Keeping the
// decisions pure is what lets test/extension-contract.test.js exercise the
// real protocol under plain Node (the `vscode` module does not exist there).

function asInteger(value) {
  if (value === null || value === undefined || value === "" || typeof value === "boolean") {
    return null;
  }
  const number = Number(value);
  return Number.isInteger(number) ? number : null;
}

function normalizePath(value) {
  return String(value === null || value === undefined ? "" : value)
    .trim()
    .replace(/\//g, "\\")
    .replace(/\\+$/, "")
    .toLowerCase();
}

// `file:///c%3A/proj/app.py` (VS Code's Uri.toString()) and
// `file:///c:/proj/app.py` (what a caller often constructs) name one file.
function pathFromUri(uri) {
  const match = /^file:\/\/(.*)$/i.exec(String(uri || "").trim());
  if (!match) {
    return "";
  }
  let rest = match[1];
  try {
    rest = decodeURIComponent(rest);
  } catch (_error) {
    // keep the raw form
  }
  rest = rest.replace(/^\/+/, "");
  if (/^[a-zA-Z]:/.test(rest) || rest.includes("\\")) {
    return rest.replace(/\//g, "\\");
  }
  return "/" + rest;
}

function documentIdentity(entry) {
  const source = entry && typeof entry === "object" ? entry : {};
  return {
    uri: String(source.uri || "").trim(),
    path: String(source.path || "").trim(),
  };
}

function identityLabel(identity) {
  const { uri, path: filePath } = documentIdentity(identity);
  return filePath || uri || "(unnamed document)";
}

function identityMatches(one, other) {
  const a = documentIdentity(one);
  const b = documentIdentity(other);
  if ((!a.uri && !a.path) || (!b.uri && !b.path)) {
    return false;
  }
  if (a.uri && b.uri && a.uri === b.uri) {
    return true;
  }
  const aPath = a.path || pathFromUri(a.uri);
  const bPath = b.path || pathFromUri(b.uri);
  if (aPath && bPath && normalizePath(aPath) === normalizePath(bPath)) {
    return true;
  }
  return false;
}

function parseRangeJson(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return { ok: false, error: "range must be an object with start/end points." };
  }
  const out = {};
  for (const side of ["start", "end"]) {
    const point = value[side];
    if (!point || typeof point !== "object" || Array.isArray(point)) {
      return { ok: false, error: `range.${side} is missing.` };
    }
    const line = asInteger(point.line);
    const character = asInteger(point.character);
    if (line === null || line < 0) {
      return { ok: false, error: `range.${side}.line must be a non-negative integer.` };
    }
    if (character === null || character < 0) {
      return { ok: false, error: `range.${side}.character must be a non-negative integer.` };
    }
    out[side] = { line, character };
  }
  if (out.end.line < out.start.line
      || (out.end.line === out.start.line && out.end.character < out.start.character)) {
    return { ok: false, error: "range.end is before range.start." };
  }
  return { ok: true, range: out };
}

// `bounds` is `{ lineCount, lineLength(line) }` — the async layer supplies a
// TextDocument wrapper, the test a literal. Bounds are line- AND
// character-checked so a range can never point outside the bytes it named.
function rangeBoundsError(range, bounds) {
  const lineCount = asInteger(bounds && bounds.lineCount);
  if (lineCount === null || lineCount < 1) {
    return "the document has no readable lines.";
  }
  if (range.start.line >= lineCount) {
    return `range.start.line ${range.start.line} is past the last line of the document (${lineCount} lines).`;
  }
  if (range.end.line > lineCount) {
    return `range.end.line ${range.end.line} is past the end of the document (${lineCount} lines).`;
  }
  if (range.end.line === lineCount && range.end.character !== 0) {
    return "range.end must be the very start of the line after the last line.";
  }
  let length = bounds.lineLength(range.start.line);
  if (!Number.isInteger(length) || range.start.character > length) {
    return `range.start.character ${range.start.character} is past the end of line ${range.start.line}.`;
  }
  if (range.end.line < lineCount) {
    length = bounds.lineLength(range.end.line);
    if (!Number.isInteger(length) || range.end.character > length) {
      return `range.end.character ${range.end.character} is past the end of line ${range.end.line}.`;
    }
  }
  return "";
}

// `POST /edit-active-selection`: a document-addressed, versioned, ranged edit.
// A payload without identity/version is a malformed request (400); one without
// an explicit range is a stale/incomplete target (409).
function planSelectionEdit(payload) {
  const body = payload && typeof payload === "object" ? payload : {};
  const identity = documentIdentity(body);
  if (!identity.uri && !identity.path) {
    return refusal(BAD_REQUEST,
      "a document uri (or path) is required: an edit is addressed to a document, never to whatever happens to be focused.");
  }
  const expectedVersion = asInteger(
    body.expectedVersion !== undefined ? body.expectedVersion : body.version);
  if (expectedVersion === null || expectedVersion < 0) {
    return refusal(BAD_REQUEST,
      "expectedVersion is required: send the document version the edit was planned against (read the buffer first).");
  }
  if (body.newText === undefined && body.new_text === undefined && body.text === undefined) {
    return refusal(BAD_REQUEST, "newText is required, even when it is empty (a deletion).");
  }
  let newText = body.newText;
  if (newText === undefined) {
    newText = body.new_text !== undefined ? body.new_text : body.text;
  }
  const parsed = parseRangeJson(body.range !== undefined ? body.range : body.selection);
  if (!parsed.ok) {
    return refusal(CONFLICT, `an explicit range is required: ${parsed.error}`);
  }
  return {
    ok: true,
    edit: {
      identity,
      expectedVersion,
      range: parsed.range,
      newText: String(newText === null || newText === undefined ? "" : newText),
    },
  };
}

// `POST /apply-edit`: every edit names its document, carries newText and the
// version it was planned against. A per-document version is mandatory for a
// multi-document edit; the top-level expectedVersion is only a convenience for
// a single-document one. One stale member later rejects the WHOLE edit.
function planWorkspaceEdit(payload) {
  const body = payload && typeof payload === "object" ? payload : {};
  const edits = Array.isArray(body.edits) ? body.edits : [];
  if (!edits.length) {
    return refusal(BAD_REQUEST, "No edits given.");
  }
  let declared = null;
  if (body.expectedVersion !== undefined && body.expectedVersion !== null) {
    declared = asInteger(body.expectedVersion);
    if (declared === null || declared < 0) {
      return refusal(BAD_REQUEST, "expectedVersion must be a non-negative integer document version.");
    }
  }
  // Group by document (uri and path forms of one file count as one document)
  // so "does this edit span several documents?" is decided on identity, not on
  // how many edit objects were sent.
  const groups = [];
  for (const edit of edits) {
    if (!edit || typeof edit !== "object" || Array.isArray(edit)) {
      return refusal(BAD_REQUEST, "every edit must be an object.");
    }
    const identity = documentIdentity(edit);
    if (!identity.uri && !identity.path) {
      return refusal(BAD_REQUEST, "every edit must name its document (path or uri).");
    }
    let group = groups.find((candidate) => identityMatches(candidate.identity, identity));
    if (!group) {
      group = { identity, count: 0 };
      groups.push(group);
    }
    group.count += 1;
  }
  const multiDocument = groups.length > 1;
  const prepared = [];
  for (let index = 0; index < edits.length; index += 1) {
    const edit = edits[index];
    const identity = documentIdentity(edit);
    if (edit.newText === undefined && edit.new_text === undefined) {
      return refusal(BAD_REQUEST,
        `edit ${index} carries no newText/new_text, so it could not change anything.`);
    }
    const explicit = edit.expectedVersion !== undefined ? edit.expectedVersion : edit.version;
    let expected = asInteger(explicit);
    if (explicit !== undefined && explicit !== null && expected === null) {
      return refusal(BAD_REQUEST, `edit ${index} carries a non-integer version.`);
    }
    if (expected === null) {
      if (multiDocument) {
        return refusal(BAD_REQUEST,
          `edit ${index} (${identityLabel(identity)}) does not carry the document version it was planned against; a multi-file edit needs one per-document precondition, so nothing was changed.`);
      }
      expected = declared;
    }
    if (expected === null || expected < 0) {
      return refusal(BAD_REQUEST,
        `edit ${index} does not carry the document version it was read at (read the buffer first); nothing was changed.`);
    }
    if (declared !== null && declared !== expected) {
      return refusal(BAD_REQUEST,
        `edit ${index} expects version ${expected} but the caller declared ${declared}; the preconditions disagree, so nothing was changed.`);
    }
    let range = null;
    if (edit.range !== undefined && edit.range !== null) {
      const parsed = parseRangeJson(edit.range);
      if (!parsed.ok) {
        return refusal(BAD_REQUEST, `edit ${index}: ${parsed.error}`);
      }
      range = parsed.range;
    }
    const rawText = edit.newText !== undefined ? edit.newText : edit.new_text;
    prepared.push({
      identity,
      expectedVersion: expected,
      range,
      newText: String(rawText === null || rawText === undefined ? "" : rawText),
    });
  }
  // Several edits to one document are applied as one atomic edit, so they must
  // have been planned against the same version.
  for (const group of groups) {
    const versions = new Set();
    for (const entry of prepared) {
      if (identityMatches(entry.identity, group.identity)) {
        versions.add(entry.expectedVersion);
      }
    }
    if (versions.size > 1) {
      return refusal(BAD_REQUEST,
        `the edits for ${identityLabel(group.identity)} disagree about the document version (${Array.from(versions).join(", ")}); nothing was changed.`);
    }
  }
  return { ok: true, edits: prepared, multiDocument };
}

// Is this planned edit still addressing the version it was planned against?
function versionMismatch(planned, live) {
  const identity = (live && live.identity) || planned.identity;
  const label = identityLabel(identity);
  const actual = asInteger(live && live.version);
  if (actual === null) {
    return refusal(CONFLICT,
      `The document ${label} reports no version, so the edit target cannot be pinned; nothing was changed.`,
      { path: label, expectedVersion: planned.expectedVersion, actualVersion: null });
  }
  if (actual !== planned.expectedVersion) {
    // The Python client keys its "re-read and retry" guidance off "mismatch";
    // keep this sentence stable.
    return refusal(CONFLICT,
      `Version mismatch for ${label}: document is at version ${actual}, expected ${planned.expectedVersion}. Re-read the buffer and retry.`,
      { path: label, expectedVersion: planned.expectedVersion, actualVersion: actual });
  }
  return null;
}

// Pair every planned edit with its live document, or refuse the whole batch:
// a document that is not open cannot carry a meaningful version precondition.
function resolveLiveDocuments(plannedEdits, liveDocuments) {
  const members = [];
  for (const planned of plannedEdits) {
    const live = (liveDocuments || []).find(
      (candidate) => identityMatches(planned.identity, candidate.identity));
    if (!live) {
      return refusal(CONFLICT,
        `The document ${identityLabel(planned.identity)} is not open in the editor, so nothing was changed; open or read it first.`,
        { path: identityLabel(planned.identity), externalEffect: false });
    }
    members.push({ planned, live });
  }
  return { ok: true, members };
}

// Every member is checked before ANY is applied: the first stale one rejects
// the whole edit and no WorkspaceEdit is ever handed to VS Code.
function verifyAllVersions(members) {
  for (const member of members) {
    const mismatch = versionMismatch(member.planned, member.live);
    if (mismatch) {
      return mismatch;
    }
  }
  return null;
}

// One apply at a time. Two concurrent edits would otherwise interleave their
// verify/apply pairs (both verify version N, both apply), so every apply runs
// through this chain.
function createSerialQueue() {
  let tail = Promise.resolve();
  return function runExclusive(work) {
    const started = tail.then(() => work(), () => work());
    tail = started.then(() => undefined, () => undefined);
    return started;
  };
}

const applyQueue = createSerialQueue();

function rangeToJson(range) {
  return {
    start: { line: range.start.line, character: range.start.character },
    end: { line: range.end.line, character: range.end.character },
  };
}

function getDiagnostics(limit = 50) {
  const output = [];
  for (const [uri, diagnostics] of vscode.languages.getDiagnostics()) {
    for (const diagnostic of diagnostics) {
      if (output.length >= limit) {
        return output;
      }
      output.push({
        file: uri.fsPath,
        message: diagnostic.message,
        severity: diagnostic.severity,
        range: rangeToJson(diagnostic.range),
        source: diagnostic.source || "",
        code: diagnostic.code ? String(diagnostic.code) : "",
      });
    }
  }
  return output;
}

function getActiveFileState() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return null;
  }

  const doc = editor.document;
  const selectionText = doc.getText(editor.selection);
  const fullText = doc.getText();
  const visibleRanges = editor.visibleRanges.map(rangeToJson);

  return {
    path: doc.uri.fsPath,
    uri: doc.uri.toString(),
    fileName: path.basename(doc.uri.fsPath || doc.fileName || ""),
    languageId: doc.languageId,
    isDirty: doc.isDirty,
    isUntitled: doc.isUntitled,
    lineCount: doc.lineCount,
    version: doc.version,
    selection: rangeToJson(editor.selection),
    selectionText,
    visibleRanges,
    text: fullText.length <= TEXT_LIMIT ? fullText : fullText.slice(0, TEXT_LIMIT),
    textTruncated: fullText.length > TEXT_LIMIT,
  };
}

async function getState() {
  const workspaceFolders = (vscode.workspace.workspaceFolders || []).map((folder) => folder.uri.fsPath);
  return {
    ok: true,
    appName: vscode.env.appName,
    remoteName: vscode.env.remoteName || "",
    workspaceFolders,
    activeFile: getActiveFileState(),
    diagnostics: getDiagnostics(),
    windowState: {
      focused: vscode.window.state.focused,
    },
  };
}

async function executeCommand(payload) {
  const command = String(payload.command || "");
  const args = Array.isArray(payload.args) ? payload.args : [];
  if (!command) {
    return { ok: false, error: "Missing command." };
  }
  const result = await vscode.commands.executeCommand(command, ...args);
  return { ok: true, message: `Executed ${command}.`, result: result === undefined ? null : result };
}

async function openFile(payload) {
  const filePath = String(payload.path || "");
  if (!filePath) {
    return { ok: false, error: "Missing path." };
  }
  const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(filePath));
  await vscode.window.showTextDocument(doc);
  return { ok: true, message: `Opened ${filePath}.` };
}

// The document a request NAMES — never "whatever is focused right now". Only
// documents the editor is already tracking qualify: a version precondition is
// meaningless for a file that is not open, so a closed document is a refusal
// (not a silent reload that would erase the precondition).
function openDocumentDescriptors() {
  return (vscode.workspace.textDocuments || []).map((doc) => ({
    identity: { uri: doc.uri.toString(), path: doc.uri.fsPath || "" },
    doc,
  }));
}

function findOpenDocument(identity) {
  const wanted = documentIdentity(identity);
  for (const candidate of openDocumentDescriptors()) {
    if (identityMatches(wanted, candidate.identity)) {
      return candidate;
    }
  }
  return null;
}

function documentBounds(doc) {
  return {
    lineCount: doc.lineCount,
    lineLength: (line) => doc.lineAt(line).text.length,
  };
}

function buildRange(range) {
  return new vscode.Range(
    range.start.line, range.start.character,
    range.end.line, range.end.character);
}

// F15 #1: replace one explicit, version-checked document range.
//
// The old handler replaced `vscode.window.activeTextEditor.selection` with a
// bare text payload: no document identity, no version and no range, so a
// check/use race edited whatever the user had focused in the meantime. Now the
// request must name the document, its expected version and the exact range,
// the NAMED document (only if open) is resolved, and a stale target produces
// ZERO edits — applyEdit is never reached.
async function editActiveSelection(payload) {
  const plan = planSelectionEdit(payload);
  if (!plan.ok) {
    return plan;
  }
  return applyQueue(async () => {
    const found = findOpenDocument(plan.edit.identity);
    if (!found) {
      return refusal(CONFLICT,
        `The document ${identityLabel(plan.edit.identity)} is not open in the editor, so nothing was changed.`,
        { path: identityLabel(plan.edit.identity), expectedVersion: plan.edit.expectedVersion });
    }
    const label = identityLabel(found.identity);
    const mismatch = versionMismatch(plan.edit, { identity: found.identity, version: found.doc.version });
    if (mismatch) {
      return mismatch;
    }
    const boundsError = rangeBoundsError(plan.edit.range, documentBounds(found.doc));
    if (boundsError) {
      return refusal(CONFLICT,
        `The target range for ${label} is not valid: ${boundsError} Nothing was changed.`,
        { path: label, expectedVersion: plan.edit.expectedVersion, actualVersion: found.doc.version });
    }
    // No await between the checks above and this application: the version that
    // was verified is the version the edit is offered against.
    const workspaceEdit = new vscode.WorkspaceEdit();
    workspaceEdit.replace(found.doc.uri, buildRange(plan.edit.range), plan.edit.newText);
    const applied = await vscode.workspace.applyEdit(workspaceEdit);
    if (!applied) {
      return refusal(CONFLICT,
        `The editor rejected the edit for ${label} (the document changed or is read-only); nothing was applied.`,
        { path: label, expectedVersion: plan.edit.expectedVersion, actualVersion: found.doc.version });
    }
    return {
      ok: true,
      message: `Edited ${label}.`,
      path: found.doc.uri.fsPath,
      uri: found.doc.uri.toString(),
      newVersion: found.doc.version,
    };
  });
}

async function writeFile(payload) {
  const filePath = String(payload.path || "");
  const content = String(payload.content || "");
  if (!filePath) {
    return { ok: false, error: "Missing path." };
  }
  await fs.promises.mkdir(path.dirname(filePath), { recursive: true });
  await fs.promises.writeFile(filePath, content, "utf8");
  return { ok: true, message: `Wrote ${filePath}.` };
}

// ── F15: structured coding interface ─────────────────────────────────────

async function resolveDocument(payload) {
  const filePath = String(payload.path || "");
  const uriString = String(payload.uri || "");
  if (filePath) {
    return vscode.workspace.openTextDocument(vscode.Uri.file(filePath));
  }
  if (uriString) {
    return vscode.workspace.openTextDocument(vscode.Uri.parse(uriString));
  }
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    throw new Error("No active text editor and no path/uri given.");
  }
  return editor.document;
}

async function readBuffer(payload) {
  const doc = await resolveDocument(payload);
  const total = doc.lineCount;
  let startLine = Number(payload.startLine || 1);
  let endLine = Number(payload.endLine || total);
  startLine = Math.max(1, Math.min(startLine, total));
  endLine = Math.max(startLine - 1, Math.min(endLine, total));
  let text = "";
  if (endLine >= startLine) {
    const range = new vscode.Range(startLine - 1, 0, endLine - 1, Number.MAX_SAFE_INTEGER);
    text = doc.getText(range);
    // getText(range) clips at line end already; drop a trailing newline the
    // range boundary can add.
    text = text.replace(/\r?\n$/, "");
  }
  const truncated = text.length > TEXT_LIMIT;
  return {
    ok: true,
    path: doc.uri.fsPath,
    uri: doc.uri.toString(),
    version: doc.version,
    lineCount: total,
    startLine,
    endLine,
    text: truncated ? text.slice(0, TEXT_LIMIT) : text,
    textTruncated: truncated,
  };
}

function getDiagnosticsFiltered(payload) {
  const wanted = String(payload.path || "").toLowerCase();
  const all = getDiagnostics(Number(payload.limit || 200));
  if (!wanted) {
    return all;
  }
  return all.filter((d) => (d.file || "").toLowerCase().includes(wanted));
}

function flattenSymbols(symbols, container, out, limit) {
  for (const symbol of symbols || []) {
    if (out.length >= limit) {
      return;
    }
    if (symbol.location) {
      // SymbolInformation
      out.push({
        name: symbol.name,
        kind: symbol.kind,
        containerName: symbol.containerName || container || "",
        range: rangeToJson(symbol.location.range),
      });
    } else {
      // DocumentSymbol (hierarchical)
      out.push({
        name: symbol.name,
        kind: symbol.kind,
        detail: symbol.detail || "",
        containerName: container || "",
        range: rangeToJson(symbol.range),
      });
      if (symbol.children && symbol.children.length) {
        flattenSymbols(symbol.children, symbol.name, out, limit);
      }
    }
  }
}

async function getSymbols(payload) {
  const doc = await resolveDocument(payload);
  const limit = Number(payload.limit || 200);
  const raw = await vscode.commands.executeCommand(
    "vscode.executeDocumentSymbolProvider", doc.uri);
  const out = [];
  flattenSymbols(raw || [], "", out, limit);
  return {
    ok: true,
    path: doc.uri.fsPath,
    uri: doc.uri.toString(),
    version: doc.version,
    symbols: out,
  };
}

async function getReferences(payload) {
  const doc = await resolveDocument(payload);
  const line = Number(payload.line || 0);
  const character = Number(payload.character || 0);
  if (!line) {
    return { ok: false, error: "Missing line (1-based)." };
  }
  const position = new vscode.Position(line - 1, Math.max(0, character - 1));
  const locations = await vscode.commands.executeCommand(
    "vscode.executeReferenceProvider", doc.uri, position) || [];
  return {
    ok: true,
    path: doc.uri.fsPath,
    uri: doc.uri.toString(),
    references: locations.slice(0, 200).map((loc) => ({
      path: loc.uri.fsPath,
      uri: loc.uri.toString(),
      range: rangeToJson(loc.range),
    })),
  };
}

const SEARCH_SKIP_DIRS = new Set([
  ".git", ".hg", ".svn", "node_modules", "__pycache__", "venv", ".venv",
  "dist", "build", "out", ".idea", ".vscode",
]);
const SEARCH_MAX_FILE_BYTES = 1024 * 1024;
const SEARCH_MAX_SCAN_FILES = 4000;

function* walkTextFiles(root) {
  let stack = [root];
  while (stack.length) {
    const dir = stack.pop();
    let entries;
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch (_e) {
      continue;
    }
    entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (!SEARCH_SKIP_DIRS.has(entry.name)) {
          stack.push(full);
        }
      } else if (entry.isFile()) {
        try {
          if (fs.statSync(full).size <= SEARCH_MAX_FILE_BYTES) {
            yield full;
          }
        } catch (_e) {
          // skip unreadable files
        }
      }
    }
  }
}

async function workspaceSearch(payload) {
  const query = String(payload.query || "");
  if (!query) {
    return { ok: false, error: "Missing query." };
  }
  const limit = Math.max(1, Math.min(Number(payload.limit || 50), 500));
  const needle = query.toLowerCase();
  const folders = (vscode.workspace.workspaceFolders || []).map((f) => f.uri.fsPath);
  const matches = [];
  let scanned = 0;
  outer:
  for (const folder of folders) {
    for (const file of walkTextFiles(folder)) {
      if (scanned >= SEARCH_MAX_SCAN_FILES) {
        break outer;
      }
      scanned += 1;
      let content;
      try {
        content = fs.readFileSync(file, "utf8");
      } catch (_e) {
        continue;
      }
      const lines = content.split(/\r?\n/);
      for (let i = 0; i < lines.length; i += 1) {
        if (lines[i].toLowerCase().includes(needle)) {
          matches.push({ path: file, line: i + 1, text: lines[i].slice(0, 300) });
          if (matches.length >= limit) {
            break outer;
          }
        }
      }
    }
  }
  return {
    ok: true,
    query,
    matches,
    matchCount: matches.length,
    scannedFiles: scanned,
    truncated: matches.length >= limit,
  };
}

function detectTestCommand(folder) {
  try {
    const pkg = JSON.parse(fs.readFileSync(path.join(folder, "package.json"), "utf8"));
    if (pkg.scripts && pkg.scripts.test && !/no test specified/.test(pkg.scripts.test)) {
      return "npm test --silent";
    }
  } catch (_e) {
    // no package.json
  }
  for (const marker of ["pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini"]) {
    if (fs.existsSync(path.join(folder, marker))) {
      return "python -m pytest -q";
    }
  }
  return "";
}

async function testResults(payload) {
  const folders = (vscode.workspace.workspaceFolders || []).map((f) => f.uri.fsPath);
  const cwd = String(payload.cwd || "") || folders[0] || process.cwd();
  const command = String(payload.command || "") || detectTestCommand(cwd);
  if (!command) {
    return {
      ok: false,
      error: "No test command could be detected; pass one explicitly.",
      cwd,
    };
  }
  const timeout = Math.max(1000, Math.min(Number(payload.timeoutMs || 120000), 600000));
  const exec = require("child_process").exec;
  return new Promise((resolve) => {
    exec(
      command,
      {
        cwd,
        timeout,
        maxBuffer: 1024 * 1024,
        env: { ...process.env, CI: "true" },
      },
      (error, stdout, stderr) => {
        const output = (String(stdout || "") + (stderr ? "\n" + stderr : "")).trim();
        const bounded = output.length > 20000 ? output.slice(-20000) : output;
        if (error && error.killed) {
          resolve({ ok: false, error: `Test run timed out after ${timeout}ms.`, command, cwd, output: bounded, exitCode: -1 });
          return;
        }
        const exitCode = error ? (error.code || 1) : 0;
        resolve({
          ok: exitCode === 0,
          command,
          cwd,
          exitCode,
          output: bounded || (exitCode === 0 ? "(no output)" : "(error, no output)"),
          error: exitCode === 0 ? "" : `exited with code ${exitCode}`,
        });
      },
    );
  });
}

// F15 #2: a WorkspaceEdit whose version checks and application are atomic.
//
// What was wrong: expected versions were optional, one top-level version stood
// in for every member of a multi-file edit, the version check and applyEdit
// were interleaved with awaits (so a check/use race could slip a user edit
// in), and two concurrent apps could each verify version N and then both
// apply. Now: every edit carries its own precondition, all documents are
// resolved and re-verified under one lock with no await between the last check
// and applyEdit, and one stale member rejects the WHOLE edit — nothing is
// applied, so there is no partial apply to undo.
async function applyWorkspaceEdit(payload) {
  const plan = planWorkspaceEdit(payload);
  if (!plan.ok) {
    return plan;
  }
  return applyQueue(async () => {
    const resolved = resolveLiveDocuments(plan.edits, openDocumentDescriptors());
    if (!resolved.ok) {
      return resolved;
    }
    const members = resolved.members; // [{ planned, live: { identity, doc } }]
    const live = members.map((member) => ({
      planned: member.planned,
      identity: member.live.identity,
      doc: member.live.doc,
    }));
    // Pass 1: every member's version, before anything is applied.
    const stale = verifyAllVersions(live.map((member) => ({
      planned: member.planned,
      live: { identity: member.identity, version: member.doc.version },
    })));
    if (stale) {
      return stale;
    }
    // Pass 2: every member's target range.
    for (const member of live) {
      if (!member.planned.range) {
        continue;
      }
      const boundsError = rangeBoundsError(member.planned.range, documentBounds(member.doc));
      if (boundsError) {
        return refusal(CONFLICT,
          `The target range for ${identityLabel(member.identity)} is not valid: ${boundsError} Nothing was changed.`,
          {
            path: identityLabel(member.identity),
            expectedVersion: member.planned.expectedVersion,
            actualVersion: member.doc.version,
          });
      }
    }
    // Pass 3: re-read every version one last time, then build and apply with no
    // await in between — a user edit cannot interleave between the last check
    // and the application.
    for (const member of live) {
      const mismatch = versionMismatch(member.planned,
        { identity: member.identity, version: member.doc.version });
      if (mismatch) {
        return mismatch;
      }
    }
    const workspaceEdit = new vscode.WorkspaceEdit();
    for (const member of live) {
      const range = member.planned.range
        ? buildRange(member.planned.range)
        : new vscode.Range(0, 0, member.doc.lineCount, 0);
      workspaceEdit.replace(member.doc.uri, range, member.planned.newText);
    }
    const applied = await vscode.workspace.applyEdit(workspaceEdit);
    if (!applied) {
      return refusal(CONFLICT,
        "The editor rejected the workspace edit (a document changed or is read-only); nothing was applied.");
    }
    return {
      ok: true,
      message: `Applied ${live.length} edit(s).`,
      newVersions: live.map((member) => ({
        path: member.doc.uri.fsPath,
        version: member.doc.version,
      })),
    };
  });
}

async function route(req, res) {
  const url = new URL(req.url, `http://${HOST}:${PORT}`);

  if (!isAuthorized(req)) {
    sendJson(res, 401, {
      ok: false,
      status: 401,
      error: "Unauthorized: a configured bridge requires the X-Jarvis-Token header.",
    });
    return;
  }

  if (req.method === "GET" && url.pathname === "/health") {
    sendJson(res, 200, { ok: true, service: "jarvis-editor-bridge", port: PORT });
    return;
  }

  if (req.method === "GET" && url.pathname === "/state") {
    sendJson(res, 200, await getState());
    return;
  }

  if (req.method !== "POST") {
    sendJson(res, 404, { ok: false, error: "Not found." });
    return;
  }

  const payload = await readBody(req);
  // A F15 edit refusal carries the HTTP status the caller needs (400 for a
  // malformed request, 409 for "the editor moved on"); every other endpoint
  // keeps the historical 200-with-{ok:false} shape.
  const sendResult = (result) => sendJson(res, statusOf(result), result);
  if (url.pathname === "/execute-command") {
    sendJson(res, 200, await executeCommand(payload));
  } else if (url.pathname === "/open-file") {
    sendJson(res, 200, await openFile(payload));
  } else if (url.pathname === "/edit-active-selection") {
    sendResult(await editActiveSelection(payload));
  } else if (url.pathname === "/write-file") {
    sendJson(res, 200, await writeFile(payload));
  } else if (url.pathname === "/read-buffer") {
    sendJson(res, 200, await readBuffer(payload));
  } else if (url.pathname === "/diagnostics") {
    sendJson(res, 200, { ok: true, diagnostics: getDiagnosticsFiltered(payload) });
  } else if (url.pathname === "/symbols") {
    sendJson(res, 200, await getSymbols(payload));
  } else if (url.pathname === "/references") {
    sendJson(res, 200, await getReferences(payload));
  } else if (url.pathname === "/workspace-search") {
    sendJson(res, 200, await workspaceSearch(payload));
  } else if (url.pathname === "/test-results") {
    sendJson(res, 200, await testResults(payload));
  } else if (url.pathname === "/apply-edit") {
    sendResult(await applyWorkspaceEdit(payload));
  } else {
    sendJson(res, 404, { ok: false, error: "Not found." });
  }
}

function startServer() {
  if (server) {
    return;
  }
  if (!configuredToken()) {
    // Fail closed: without a token every request is refused, so say so instead
    // of silently binding a port that can never serve a client.
    vscode.window.showWarningMessage(
      "Jarvis editor bridge: JARVIS_EDITOR_BRIDGE_TOKEN is not set, so every request will be refused. Set it and restart the bridge.");
  }
  server = http.createServer((req, res) => {
    route(req, res).catch((error) => {
      sendJson(res, 500, { ok: false, error: String(error && error.message ? error.message : error) });
    });
  });
  server.listen(PORT, HOST, () => {
    vscode.window.setStatusBarMessage(`Jarvis editor bridge listening on ${HOST}:${PORT}`, 3000);
  });
}

function stopServer() {
  if (!server) {
    return;
  }
  server.close();
  server = null;
  vscode.window.setStatusBarMessage("Jarvis editor bridge stopped", 3000);
}

function activate(context) {
  context.subscriptions.push(vscode.commands.registerCommand("jarvisBridge.start", startServer));
  context.subscriptions.push(vscode.commands.registerCommand("jarvisBridge.stop", stopServer));
  startServer();
}

function deactivate() {
  stopServer();
}

module.exports = {
  activate,
  deactivate,
  // F15: the protocol decisions are exported so
  // test/extension-contract.test.js can exercise them under plain Node, where
  // the `vscode` module does not exist. VS Code only consumes activate and
  // deactivate, so the extra names change nothing about being loaded there.
  AUTH_HEADER,
  OK,
  BAD_REQUEST,
  CONFLICT,
  authorizeRequest,
  configuredToken,
  timingSafeTokenEquals,
  asInteger,
  normalizePath,
  pathFromUri,
  documentIdentity,
  identityLabel,
  identityMatches,
  parseRangeJson,
  rangeBoundsError,
  planSelectionEdit,
  planWorkspaceEdit,
  versionMismatch,
  resolveLiveDocuments,
  verifyAllVersions,
  createSerialQueue,
  findOpenDocument,
  statusOf,
  refusal,
  route,
  editActiveSelection,
  applyWorkspaceEdit,
};
