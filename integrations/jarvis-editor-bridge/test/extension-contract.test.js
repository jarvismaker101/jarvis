"use strict";

/**
 * F15 — editor-bridge extension contract.
 *
 * The audit finding ("Upgrade the Editor Bridge Into a Coding Interface") left
 * three extension-side gaps open:
 *
 *   1. /edit-active-selection replaced "whatever is focused right now" with no
 *      document identity, version or range;
 *   2. /apply-edit's version check and applyEdit were not atomic: versions were
 *      optional, one top-level version stood in for a whole multi-file edit,
 *      and concurrent applies could each verify before either applied;
 *   3. authentication failed OPEN without JARVIS_EDITOR_BRIDGE_TOKEN, accepted
 *      ?token= (which leaks into logs) and compared with `===`.
 *
 * This is a plain Node script (`node --test`) because the `vscode` module only
 * exists inside an extension host: a tiny stand-in is injected into the module
 * loader before extension.js is required, and the extension's exported protocol
 * decisions + HTTP router are then driven directly. Nothing here starts a
 * server or touches VS Code.
 */

const assert = require("node:assert/strict");
const path = require("node:path");
const Module = require("node:module");
const test = require("node:test");

// ── a minimal `vscode` stand-in ───────────────────────────────────────────

class FakeUri {
  constructor(fsPath) {
    this.fsPath = fsPath;
    this.scheme = "file";
  }

  // VS Code encodes the drive colon: file:///c%3A/proj/app.py
  toString() {
    const posix = String(this.fsPath).replace(/\\/g, "/");
    const encoded = posix.replace(/^([A-Za-z]):/, (_m, drive) => `${drive.toLowerCase()}%3A`);
    return `file:///${encoded}`;
  }

  static file(fsPath) {
    return new FakeUri(fsPath);
  }

  static parse(value) {
    return new FakeUri(String(value).replace(/^file:\/\/\//, ""));
  }
}

class FakeRange {
  constructor(startLine, startCharacter, endLine, endCharacter) {
    this.start = { line: startLine, character: startCharacter };
    this.end = { line: endLine, character: endCharacter };
  }
}

class FakePosition {
  constructor(line, character) {
    this.line = line;
    this.character = character;
  }
}

class FakeWorkspaceEdit {
  constructor() {
    this.edits = [];
  }

  replace(uri, range, newText) {
    this.edits.push({ uri, range, newText });
  }
}

function makeDocument(fsPath, text, version) {
  const lines = String(text).split("\n");
  return {
    uri: FakeUri.file(fsPath),
    fileName: fsPath,
    languageId: "python",
    isDirty: false,
    isUntitled: false,
    lineCount: lines.length,
    version,
    lineAt(line) {
      if (line < 0 || line >= lines.length) {
        throw new Error(`line out of range: ${line}`);
      }
      return { text: lines[line] };
    },
    getText() {
      return lines.join("\n");
    },
  };
}

let applyHook = null;

const vscodeStub = {
  workspace: {
    textDocuments: [],
    workspaceFolders: [],
    __applied: [],
    async applyEdit(edit) {
      vscodeStub.workspace.__applied.push(edit);
      if (applyHook) {
        return applyHook(edit);
      }
      return true;
    },
    async openTextDocument() {
      throw new Error("openTextDocument must not be used by an edit path");
    },
    async showTextDocument() {},
  },
  window: {
    activeTextEditor: undefined,
    state: { focused: true },
    setStatusBarMessage() {},
    showWarningMessage() {},
  },
  languages: { getDiagnostics: () => [] },
  commands: {
    async executeCommand() {
      return undefined;
    },
    registerCommand() {
      return { dispose() {} };
    },
  },
  env: { appName: "Test Editor", remoteName: "" },
  Uri: FakeUri,
  Range: FakeRange,
  Position: FakePosition,
  WorkspaceEdit: FakeWorkspaceEdit,
};

const originalLoad = Module._load;
Module._load = function patchedLoad(request, parent, isMain) {
  if (request === "vscode") {
    return vscodeStub;
  }
  return originalLoad.call(this, request, parent, isMain);
};
// eslint-disable-next-line import/no-dynamic-require
const bridge = require(path.join(__dirname, "..", "extension.js"));
Module._load = originalLoad;

// ── helpers ───────────────────────────────────────────────────────────────

const TOKEN = "test-token-0123456789";
const PATH_A = "C:\\proj\\a.py";
const PATH_B = "C:\\proj\\b.py";
const TEXT_A = "alpha\nbeta\ngamma\n";
const SELECTION = {
  start: { line: 1, character: 0 },
  end: { line: 1, character: 4 },
};

function resetWorld({ documents = [], active, hook = null } = {}) {
  vscodeStub.workspace.textDocuments = documents;
  vscodeStub.workspace.__applied = [];
  vscodeStub.window.activeTextEditor = active;
  applyHook = hook;
}

function fakeRequest({ method = "POST", url = "/", headers = {}, body = undefined } = {}) {
  const chunks = body === undefined ? [] : [Buffer.from(JSON.stringify(body))];
  // Node lowercases incoming header names; the bridge relies on that (as the
  // real extension host does), so the stand-in request does too.
  const normalizedHeaders = {};
  for (const [name, value] of Object.entries(headers)) {
    normalizedHeaders[String(name).toLowerCase()] = value;
  }
  return {
    method,
    url,
    headers: normalizedHeaders,
    on(event, handler) {
      if (event === "data") {
        chunks.forEach((chunk) => handler(chunk));
      } else if (event === "end") {
        handler();
      }
      return this;
    },
    destroy() {},
  };
}

function fakeResponse() {
  return {
    statusCode: null,
    headers: null,
    payload: null,
    writeHead(status, headers) {
      this.statusCode = status;
      this.headers = headers;
    },
    end(chunk) {
      const raw = String(chunk === undefined ? "" : chunk);
      try {
        this.payload = JSON.parse(raw);
      } catch (_error) {
        this.payload = null;
      }
    },
  };
}

async function callBridge(pathname, body, options = {}) {
  const res = fakeResponse();
  await bridge.route(fakeRequest({
    method: options.method || "POST",
    url: pathname,
    headers: options.headers || {},
    body,
  }), res);
  return res;
}

async function withToken(token, run) {
  const previous = process.env.JARVIS_EDITOR_BRIDGE_TOKEN;
  if (token === undefined) {
    delete process.env.JARVIS_EDITOR_BRIDGE_TOKEN;
  } else {
    process.env.JARVIS_EDITOR_BRIDGE_TOKEN = token;
  }
  try {
    return await run();
  } finally {
    if (previous === undefined) {
      delete process.env.JARVIS_EDITOR_BRIDGE_TOKEN;
    } else {
      process.env.JARVIS_EDITOR_BRIDGE_TOKEN = previous;
    }
  }
}

const authHeaders = { "x-jarvis-token": TOKEN };
const authHeadersUpper = { "X-Jarvis-Token": TOKEN };

// ── 3. authentication fails closed, header-only, constant-time ────────────

test("auth: no configured token refuses every request, header or not", async () => {
  await withToken(undefined, async () => {
    assert.equal(bridge.authorizeRequest(authHeaders, ""), false);
    assert.equal(bridge.authorizeRequest(authHeaders, undefined), false);
    assert.equal(bridge.configuredToken(), "");

    const health = await callBridge("/health", undefined, {
      method: "GET", headers: authHeadersUpper,
    });
    assert.equal(health.statusCode, 401, "an unconfigured bridge must not serve /health");
    assert.equal(health.payload.ok, false);

    resetWorld({ documents: [makeDocument(PATH_A, TEXT_A, 1)] });
    const edit = await callBridge("/edit-active-selection", {
      path: PATH_A, expectedVersion: 1, range: SELECTION, newText: "x",
    }, { headers: authHeadersUpper });
    assert.equal(edit.statusCode, 401);
    assert.equal(vscodeStub.workspace.__applied.length, 0);
  });
});

test("auth: a configured token is required on every endpoint", async () => {
  await withToken(TOKEN, async () => {
    const anonymous = await callBridge("/health", undefined, { method: "GET" });
    assert.equal(anonymous.statusCode, 401);

    const wrong = await callBridge("/health", undefined, {
      method: "GET", headers: { "x-jarvis-token": "not-the-token" },
    });
    assert.equal(wrong.statusCode, 401);

    const right = await callBridge("/health", undefined, {
      method: "GET", headers: authHeadersUpper,
    });
    assert.equal(right.statusCode, 200);
    assert.equal(right.payload.ok, true);
  });
});

test("auth: the ?token= query string is no longer accepted", async () => {
  await withToken(TOKEN, async () => {
    const viaQuery = await callBridge(`/health?token=${TOKEN}`, undefined, { method: "GET" });
    assert.equal(viaQuery.statusCode, 401,
      "a token in the URL leaks into logs and must never authenticate");
    const viaHeader = await callBridge("/health?token=nope", undefined, {
      method: "GET", headers: authHeaders,
    });
    assert.equal(viaHeader.statusCode, 200);
  });
});

test("auth: comparison is constant-time and length-safe", () => {
  assert.equal(bridge.timingSafeTokenEquals(TOKEN, TOKEN), true);
  assert.equal(bridge.timingSafeTokenEquals("short", TOKEN), false);
  assert.equal(bridge.timingSafeTokenEquals(`${TOKEN}longer`, TOKEN), false);
  assert.equal(bridge.timingSafeTokenEquals("", TOKEN), false);
  assert.equal(bridge.timingSafeTokenEquals(TOKEN, ""), false);
  assert.equal(bridge.authorizeRequest(authHeaders, TOKEN), true);
  assert.equal(bridge.authorizeRequest({ "x-jarvis-token": ["a", "b"] }, TOKEN), false,
    "a repeated header is not a token");
  assert.equal(bridge.authorizeRequest({ "x-jarvis-token": "has space" }, TOKEN), false);
});

// ── 1. /edit-active-selection needs identity + version + range ────────────

test("selection: a text-only payload is refused before anything is resolved", async () => {
  await withToken(TOKEN, async () => {
    resetWorld({ documents: [makeDocument(PATH_A, TEXT_A, 1)] });
    const res = await callBridge("/edit-active-selection", { text: "replacement" },
      { headers: authHeaders });
    assert.equal(res.statusCode, 400);
    assert.match(res.payload.error, /uri/);
    assert.equal(vscodeStub.workspace.__applied.length, 0);
  });
});

test("selection: a version-less or range-less payload is refused", async () => {
  await withToken(TOKEN, async () => {
    resetWorld({ documents: [makeDocument(PATH_A, TEXT_A, 1)] });
    const noVersion = await callBridge("/edit-active-selection",
      { path: PATH_A, range: SELECTION, newText: "x" }, { headers: authHeaders });
    assert.equal(noVersion.statusCode, 400);
    assert.match(noVersion.payload.error, /expectedVersion/);

    const noRange = await callBridge("/edit-active-selection",
      { path: PATH_A, expectedVersion: 1, newText: "x" }, { headers: authHeaders });
    assert.equal(noRange.statusCode, 409);
    assert.match(noRange.payload.error, /range/);

    assert.equal(vscodeStub.workspace.__applied.length, 0);
  });
});

test("selection: the NAMED document is edited, never the focused one", async () => {
  await withToken(TOKEN, async () => {
    const docA = makeDocument(PATH_A, TEXT_A, 5);
    const docB = makeDocument(PATH_B, "other\nfile\n", 5);
    resetWorld({ documents: [docA, docB], active: { document: docB, selection: SELECTION } });

    const res = await callBridge("/edit-active-selection",
      { path: PATH_A, expectedVersion: 5, range: SELECTION, newText: "named" },
      { headers: authHeaders });
    assert.equal(res.statusCode, 200, JSON.stringify(res.payload));
    assert.equal(vscodeStub.workspace.__applied.length, 1);
    const applied = vscodeStub.workspace.__applied[0];
    assert.equal(applied.edits.length, 1);
    assert.equal(applied.edits[0].uri.fsPath, PATH_A);
    assert.deepEqual(
      { start: applied.edits[0].range.start, end: applied.edits[0].range.end },
      SELECTION);
    assert.equal(applied.edits[0].newText, "named");
  });
});

test("selection: a named document that is not open is refused with zero edits", async () => {
  await withToken(TOKEN, async () => {
    const docB = makeDocument(PATH_B, "other\nfile\n", 5);
    resetWorld({ documents: [docB], active: { document: docB, selection: SELECTION } });
    const res = await callBridge("/edit-active-selection",
      { path: PATH_A, expectedVersion: 5, range: SELECTION, newText: "x" },
      { headers: authHeaders });
    assert.equal(res.statusCode, 409);
    assert.match(res.payload.error, /not open/);
    assert.equal(vscodeStub.workspace.__applied.length, 0);
  });
});

test("selection: a same-position content change (stale version) produces ZERO edits", async () => {
  await withToken(TOKEN, async () => {
    // The selection is byte-identical; the user typed, so the document moved
    // to version 6 while the plan was made against version 5.
    const docA = makeDocument(PATH_A, TEXT_A, 6);
    resetWorld({ documents: [docA], active: { document: docA, selection: SELECTION } });
    const res = await callBridge("/edit-active-selection",
      { path: PATH_A, expectedVersion: 5, range: SELECTION, newText: "x" },
      { headers: authHeaders });
    assert.equal(res.statusCode, 409);
    assert.equal(res.payload.expectedVersion, 5);
    assert.equal(res.payload.actualVersion, 6);
    assert.match(res.payload.error, /Version mismatch/);
    assert.equal(vscodeStub.workspace.__applied.length, 0,
      "a stale target must not reach vscode.workspace.applyEdit");
  });
});

test("selection: an out-of-bounds range is refused with zero edits", async () => {
  await withToken(TOKEN, async () => {
    const docA = makeDocument(PATH_A, TEXT_A, 5);
    resetWorld({ documents: [docA] });
    const pastEnd = await callBridge("/edit-active-selection",
      { path: PATH_A, expectedVersion: 5, newText: "x",
        range: { start: { line: 0, character: 0 }, end: { line: 99, character: 0 } } },
      { headers: authHeaders });
    assert.equal(pastEnd.statusCode, 409);
    assert.match(pastEnd.payload.error, /past the end/);

    const pastLineEnd = await callBridge("/edit-active-selection",
      { path: PATH_A, expectedVersion: 5, newText: "x",
        range: { start: { line: 0, character: 0 }, end: { line: 0, character: 40 } } },
      { headers: authHeaders });
    assert.equal(pastLineEnd.statusCode, 409);
    assert.match(pastLineEnd.payload.error, /past the end of line/);

    assert.equal(vscodeStub.workspace.__applied.length, 0);
  });
});

test("selection: a uri-only payload resolves through the uri/path equivalence", async () => {
  await withToken(TOKEN, async () => {
    const docA = makeDocument(PATH_A, TEXT_A, 5);
    resetWorld({ documents: [docA] });
    const res = await callBridge("/edit-active-selection",
      { uri: "file:///c:/proj/a.py", expectedVersion: 5,
        selection: SELECTION, text: "via-uri" },
      { headers: authHeaders });
    assert.equal(res.statusCode, 200, JSON.stringify(res.payload));
    assert.equal(vscodeStub.workspace.__applied[0].edits[0].newText, "via-uri");
  });
});

// ── 2. /apply-edit: per-document, atomic, serialized ─────────────────────

function twoFileEdit(versionA = 7, versionB = 9) {
  return [
    { path: PATH_A, newText: "a", version: versionA,
      range: { start: { line: 0, character: 0 }, end: { line: 0, character: 0 } } },
    { path: PATH_B, newText: "b", version: versionB,
      range: { start: { line: 0, character: 0 }, end: { line: 0, character: 0 } } },
  ];
}

test("apply-edit: a multi-file edit needs a version for every member", async () => {
  await withToken(TOKEN, async () => {
    resetWorld({
      documents: [makeDocument(PATH_A, TEXT_A, 7), makeDocument(PATH_B, "b\n", 9)],
    });
    const edits = twoFileEdit();
    delete edits[1].version;
    const res = await callBridge("/apply-edit", { edits, expectedVersion: 7 },
      { headers: authHeaders });
    assert.equal(res.statusCode, 400);
    assert.match(res.payload.error, /per-document/);
    assert.equal(vscodeStub.workspace.__applied.length, 0);
  });
});

test("apply-edit: one top-level version may stand for a single document only", async () => {
  await withToken(TOKEN, async () => {
    resetWorld({ documents: [makeDocument(PATH_A, TEXT_A, 7)] });
    const single = await callBridge("/apply-edit",
      { edits: [{ path: PATH_A, newText: "x" }], expectedVersion: 7 },
      { headers: authHeaders });
    assert.equal(single.statusCode, 200, JSON.stringify(single.payload));
    assert.equal(vscodeStub.workspace.__applied.length, 1);

    resetWorld({
      documents: [makeDocument(PATH_A, TEXT_A, 7), makeDocument(PATH_B, "b\n", 9)],
    });
    const blanket = twoFileEdit().map(({ path: p, newText, range }) => ({ path: p, newText, range }));
    const multi = await callBridge("/apply-edit", { edits: blanket, expectedVersion: 7 },
      { headers: authHeaders });
    assert.equal(multi.statusCode, 400);
    assert.equal(vscodeStub.workspace.__applied.length, 0);
  });
});

test("apply-edit: every member's precondition is honoured on the happy path", async () => {
  await withToken(TOKEN, async () => {
    resetWorld({
      documents: [makeDocument(PATH_A, TEXT_A, 7), makeDocument(PATH_B, "b\n", 9)],
    });
    const res = await callBridge("/apply-edit", { edits: twoFileEdit() },
      { headers: authHeaders });
    assert.equal(res.statusCode, 200, JSON.stringify(res.payload));
    assert.equal(res.payload.message, "Applied 2 edit(s).");
    assert.equal(vscodeStub.workspace.__applied.length, 1, "one atomic WorkspaceEdit");
    assert.equal(vscodeStub.workspace.__applied[0].edits.length, 2);
  });
});

test("apply-edit: one stale member rejects the WHOLE edit (no partial apply)", async () => {
  await withToken(TOKEN, async () => {
    const docA = makeDocument(PATH_A, TEXT_A, 7);
    const docB = makeDocument(PATH_B, "b\n", 9);
    resetWorld({ documents: [docA, docB] });
    const edits = twoFileEdit();
    edits[1].version = 8; // stale: the live document is at 9
    const res = await callBridge("/apply-edit", { edits }, { headers: authHeaders });
    assert.equal(res.statusCode, 409);
    assert.equal(res.payload.path, PATH_B);
    assert.equal(res.payload.expectedVersion, 8);
    assert.equal(res.payload.actualVersion, 9);
    assert.match(res.payload.error, /Version mismatch for/);
    assert.equal(vscodeStub.workspace.__applied.length, 0, "no partial apply");
    assert.equal(docA.version, 7);
    assert.equal(docB.version, 9);
  });
});

test("apply-edit: a document that is not open is refused, never silently loaded", async () => {
  await withToken(TOKEN, async () => {
    resetWorld({ documents: [makeDocument(PATH_A, TEXT_A, 7)] });
    const res = await callBridge("/apply-edit", {
      edits: [{ path: PATH_B, newText: "x", version: 9 }],
    }, { headers: authHeaders });
    assert.equal(res.statusCode, 409);
    assert.match(res.payload.error, /not open/);
    assert.equal(vscodeStub.workspace.__applied.length, 0);
  });
});

test("apply-edit: contradictory versions for one document are refused", async () => {
  await withToken(TOKEN, async () => {
    resetWorld({ documents: [makeDocument(PATH_A, TEXT_A, 7)] });
    const res = await callBridge("/apply-edit", {
      edits: [
        { path: PATH_A, newText: "a", version: 7 },
        { uri: "file:///c:/proj/a.py", newText: "b", version: 6 },
      ],
    }, { headers: authHeaders });
    assert.equal(res.statusCode, 400);
    assert.match(res.payload.error, /disagree about the document version/);
    assert.equal(vscodeStub.workspace.__applied.length, 0);
  });
});

test("apply-edit: an out-of-bounds member range is refused before applying", async () => {
  await withToken(TOKEN, async () => {
    resetWorld({
      documents: [makeDocument(PATH_A, TEXT_A, 7), makeDocument(PATH_B, "b\n", 9)],
    });
    const edits = twoFileEdit();
    edits[1].range = {
      start: { line: 0, character: 0 },
      end: { line: 12, character: 0 },
    };
    const res = await callBridge("/apply-edit", { edits }, { headers: authHeaders });
    assert.equal(res.statusCode, 409);
    assert.match(res.payload.error, /past the end/);
    assert.equal(vscodeStub.workspace.__applied.length, 0);
  });
});

test("apply-edit: concurrent applies are serialized, so the second sees the new version", async () => {
  await withToken(TOKEN, async () => {
    const docA = makeDocument(PATH_A, TEXT_A, 7);
    resetWorld({
      documents: [docA],
      hook: async () => {
        // VS Code bumps the document version when an edit lands.
        docA.version += 1;
        return true;
      },
    });
    const payload = { edits: [{ path: PATH_A, newText: "x", version: 7 }] };
    // Both calls verify version 7 unless the check/apply pair is serialized:
    // without the mutex both would observe 7 and both would apply.
    const [first, second] = await Promise.all([
      callBridge("/apply-edit", payload, { headers: authHeaders }),
      callBridge("/apply-edit", payload, { headers: authHeaders }),
    ]);
    assert.equal(first.statusCode, 200, JSON.stringify(first.payload));
    assert.equal(second.statusCode, 409, JSON.stringify(second.payload));
    assert.equal(second.payload.expectedVersion, 7);
    assert.equal(second.payload.actualVersion, 8);
    assert.equal(vscodeStub.workspace.__applied.length, 1);
  });
});

// ── the exported decision helpers themselves ─────────────────────────────

test("identityMatches treats uri and path forms of one file as one document", () => {
  assert.equal(bridge.identityMatches(
    { uri: "file:///c%3A/proj/a.py" }, { uri: "file:///c:/proj/a.py" }), true);
  assert.equal(bridge.identityMatches(
    { uri: "file:///c%3A/proj/a.py" }, { path: "C:\\proj\\a.py" }), true);
  assert.equal(bridge.identityMatches(
    { path: "c:/proj/a.py" }, { path: "C:\\PROJ\\A.PY" }), true);
  assert.equal(bridge.identityMatches(
    { uri: "file:///c:/proj/a.py" }, { path: "C:\\proj\\b.py" }), false);
  assert.equal(bridge.identityMatches({}, { path: PATH_A }), false);
});

test("parseRangeJson rejects malformed, reversed and negative ranges", () => {
  assert.deepEqual(bridge.parseRangeJson(SELECTION),
    { ok: true, range: SELECTION });
  assert.equal(bridge.parseRangeJson(null).ok, false);
  assert.equal(bridge.parseRangeJson({ start: { line: 0 } }).ok, false);
  assert.equal(bridge.parseRangeJson({
    start: { line: 2, character: 0 }, end: { line: 1, character: 0 },
  }).ok, false);
  assert.equal(bridge.parseRangeJson({
    start: { line: -1, character: 0 }, end: { line: 1, character: 0 },
  }).ok, false);
});

test("rangeBoundsError accepts the end of the document and refuses past it", () => {
  const bounds = { lineCount: 3, lineLength: (line) => ["alpha", "beta", ""][line].length };
  assert.equal(bridge.rangeBoundsError(
    { start: { line: 0, character: 0 }, end: { line: 3, character: 0 } }, bounds), "");
  assert.equal(bridge.rangeBoundsError(
    { start: { line: 2, character: 0 }, end: { line: 2, character: 0 } }, bounds), "");
  assert.match(bridge.rangeBoundsError(
    { start: { line: 3, character: 0 }, end: { line: 3, character: 0 } }, bounds),
  /past the last line/);
  assert.match(bridge.rangeBoundsError(
    { start: { line: 0, character: 0 }, end: { line: 3, character: 4 } }, bounds),
  /line after the last line/);
  assert.match(bridge.rangeBoundsError(
    { start: { line: 1, character: 9 }, end: { line: 1, character: 9 } }, bounds),
  /past the end of line 1/);
});

test("createSerialQueue runs one job at a time and survives a rejection", async () => {
  const run = bridge.createSerialQueue();
  const events = [];
  let active = 0;
  const job = (name, fail) => async () => {
    active += 1;
    assert.equal(active, 1, `${name} overlapped another job`);
    events.push(`start:${name}`);
    await new Promise((resolve) => setTimeout(resolve, 2));
    events.push(`end:${name}`);
    active -= 1;
    if (fail) {
      throw new Error(`${name} failed`);
    }
    return name;
  };
  const results = await Promise.allSettled([
    run(job("a")),
    run(job("b", true)),
    run(job("c")),
  ]);
  assert.deepEqual(results.map((r) => r.status),
    ["fulfilled", "rejected", "fulfilled"]);
  assert.deepEqual(events, [
    "start:a", "end:a", "start:b", "end:b", "start:c", "end:c",
  ]);
});

test("statusOf defaults to 200 and honours an explicit refusal status", () => {
  assert.equal(bridge.statusOf({ ok: true }), 200);
  assert.equal(bridge.statusOf({ ok: false, error: "x" }), 200);
  assert.equal(bridge.statusOf(bridge.refusal(bridge.CONFLICT, "stale")), 409);
  assert.equal(bridge.statusOf(bridge.refusal(bridge.BAD_REQUEST, "malformed")), 400);
});

test("the extension still exports the VS Code entry points", () => {
  assert.equal(typeof bridge.activate, "function");
  assert.equal(typeof bridge.deactivate, "function");
  assert.equal(bridge.AUTH_HEADER, "x-jarvis-token");
});
