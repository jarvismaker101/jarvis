// G11 / F51 — per-sender policy for the main-process backend proxy.
//
// The repo has no JS test harness (package.json "test" is a stub), so this
// uses Node's built-in runner:
//
//     node --test tests/backend-request-policy.test.js
//
// main.js is an Electron entry point, so `electron` is stubbed before it is
// required; the policy helpers are pure and need no Electron runtime.
const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("path");
const { pathToFileURL } = require("url");

const electronPath = require.resolve("electron");
const ipcHandlers = new Map();
let fakeWebContentsSeq = 900;

class FakeWebContents {
  constructor(id) {
    this.id = id;
  }
  on() {}
  send() {}
  setWindowOpenHandler() {}
  isLoading() { return false; }
}

class FakeBrowserWindow {
  constructor() {
    this.webContents = new FakeWebContents(++fakeWebContentsSeq);
    this._destroyed = false;
  }
  loadFile() {}
  loadURL() {}
  setAlwaysOnTop() {}
  setIgnoreMouseEvents() {}
  on() {}
  once() {}
  isVisible() { return false; }
  isDestroyed() { return this._destroyed; }
  show() {}
  showInactive() {}
  hide() {}
  focus() {}
  minimize() {}
  close() {}
}

const exposedBridges = [];
const ipcInvokes = [];
require.cache[electronPath] = {
  id: electronPath,
  filename: electronPath,
  loaded: true,
  exports: {
    app: { whenReady: () => ({ then: () => {} }), on: () => {} },
    BrowserWindow: FakeBrowserWindow,
    ipcMain: {
      on: (channel, handler) => ipcHandlers.set("on:" + channel, handler),
      handle: (channel, handler) => ipcHandlers.set(channel, handler),
    },
    screen: {
      getPrimaryDisplay: () => ({
        workArea: { width: 1920, height: 1080, x: 0, y: 0 },
        workAreaSize: { width: 1920, height: 1080 },
      }),
    },
    contextBridge: {
      exposeInMainWorld: (name, api) => exposedBridges.push({ name, api }),
    },
    ipcRenderer: {
      invoke: (channel, payload) => {
        ipcInvokes.push({ channel, payload });
        return Promise.resolve({ status: 200, text: "{}" });
      },
      send: () => {},
      on: () => {},
    },
  },
};

const ROOT_DIR = path.join(__dirname, "..");
const policy = require(path.join(ROOT_DIR, "main.js"));
const capsuleMain = require(path.join(ROOT_DIR, "frontend", "capsule_main.js"));
const { evaluateBackendRequestPolicy, isLocalAppFrameUrl, isReadOnlyBackendPath } = policy;
const backendHandler = ipcHandlers.get("jarvis:backend-request");
const capsuleHandler = ipcHandlers.get("jarvis:capsule-command");

const APP_FRAME = pathToFileURL(path.join(ROOT_DIR, "frontend", "index.html")).href;
const CAPSULE_FRAME = pathToFileURL(path.join(ROOT_DIR, "frontend", "capsule.html")).href;

// A real capsule BrowserWindow, registered through the production path.
const capsuleWindow = capsuleMain.createCapsuleWindow();
const CAPSULE_ID = capsuleWindow.webContents.id;
const capsuleEvent = () => ({
  sender: { id: CAPSULE_ID, getURL: () => CAPSULE_FRAME },
  senderFrame: { url: CAPSULE_FRAME },
});

function decide(overrides) {
  return evaluateBackendRequestPolicy(Object.assign({
    trusted: false,
    localFrame: true,
    method: "GET",
    path: "/ui-state",
  }, overrides));
}

// ── trusted chat window: full existing authority ────────────────────────────
test("trusted window may mutate", () => {
  for (const [method, reqPath] of [
    ["POST", "/ask"],
    ["POST", "/task/stop"],
    ["POST", "/speak/stop"],
    ["POST", "/screen-action"],
    ["POST", "/settings"],
    ["PUT", "/settings"],
    ["PATCH", "/settings"],
    ["DELETE", "/jobs/1"],
    ["GET", "/anything/private"],
  ]) {
    assert.equal(decide({ trusted: true, method, path: reqPath }).allowed, true,
      `${method} ${reqPath} should be allowed for the trusted window`);
  }
});

test("trusted window path is forwarded verbatim (query preserved)", () => {
  const decision = decide({ trusted: true, method: "GET", path: "/voice-state?full=1" });
  assert.deepEqual(decision, { allowed: true, path: "/voice-state?full=1" });
});

// ── non-trusted renderers: idempotent GET allowlist ─────────────────────────
test("non-trusted renderer may read the allowlist", () => {
  for (const reqPath of [
    "/health",
    "/ui-state",
    "/voice-state",
    "/research-result",
    "/research-result/latest",
    "/research-progress",
    "/research-progress/42",
    "/ask/status",
    "/ask/status/abc",
    "/settings",
    "/settings/voice",
    "/ui-state?t=1",
    "/ui-state/", // trailing slash is normalised away before the match
  ]) {
    assert.equal(decide({ path: reqPath }).allowed, true, `${reqPath} should be readable`);
  }
});

test("non-trusted renderer may not mutate", () => {
  for (const method of ["POST", "PUT", "PATCH", "DELETE"]) {
    for (const reqPath of ["/ask", "/speak/stop", "/task/stop", "/ui-state", "/settings"]) {
      const decision = decide({ method, path: reqPath });
      assert.deepEqual(decision, { allowed: false, error: "read-only renderer" },
        `${method} ${reqPath} must be refused`);
    }
  }
});

test("non-trusted renderer may not read off-allowlist paths", () => {
  for (const reqPath of [
    "/screen-answer",
    "/ask",
    "/task/stop",
    "/voice-log",
    "/memory",
    "/approvals",
    "/jobs",
    "/settings/voice/secret",
    "/ui-state/extra",
  ]) {
    const decision = decide({ path: reqPath });
    assert.deepEqual(decision, { allowed: false, error: "path not readable" },
      `${reqPath} must be refused`);
  }
});

test("encoded, traversing and doubling paths cannot slip past the allowlist", () => {
  for (const reqPath of [
    "/%75i-state",          // "/ui-state" after backend decoding
    "/ui-state/../ask",
    "/ui-state/./../screen-action",
    "/ui-state//../ask",
    "/UI-STATE",
    "ui-state",
    "/ui-state\\..\\ask",
  ]) {
    const decision = decide({ path: reqPath });
    assert.equal(decision.allowed, false, `${reqPath} must be refused`);
  }
});

// ── frame validation ────────────────────────────────────────────────────────
test("non-local frames get nothing, trusted sender or not", () => {
  for (const frameUrl of ["", "https://evil.example/", "http://127.0.0.1:9999/",
    "file:///etc/passwd", "about:blank", "null"]) {
    for (const trusted of [false, true]) {
      const localFrame = isLocalAppFrameUrl(frameUrl, ROOT_DIR, "");
      assert.equal(localFrame, false, `${frameUrl} must not be a local app frame`);
      const decision = evaluateBackendRequestPolicy({
        trusted, localFrame, method: trusted ? "POST" : "GET", path: trusted ? "/ask" : "/health",
      });
      assert.deepEqual(decision, { allowed: false, error: "untrusted frame" });
    }
  }
});

test("local app frames are recognised, neighbours are not", () => {
  assert.equal(isLocalAppFrameUrl(APP_FRAME, ROOT_DIR, ""), true);
  assert.equal(isLocalAppFrameUrl(pathToFileURL(path.join(ROOT_DIR, "frontend", "overlay.html")).href, ROOT_DIR, ""), true);
  assert.equal(isLocalAppFrameUrl(pathToFileURL(path.join(ROOT_DIR, "..", "elsewhere.html")).href, ROOT_DIR, ""), false);
  assert.equal(isLocalAppFrameUrl("file:///C:/Windows/System32/drivers/etc/hosts", ROOT_DIR, ""), false);
});

test("app:// frames count only for the explicitly configured origin", () => {
  assert.equal(isLocalAppFrameUrl("app://jarvis/index.html", ROOT_DIR, ""), false);
  assert.equal(isLocalAppFrameUrl("app://jarvis/index.html", ROOT_DIR, "app://jarvis"), true);
  assert.equal(isLocalAppFrameUrl("app://evil/index.html", ROOT_DIR, "app://jarvis"), false);
});

// ── method / path shape ─────────────────────────────────────────────────────
test("unsupported methods and malformed paths are refused for everyone", () => {
  assert.deepEqual(decide({ trusted: true, method: "HEAD", path: "/health" }),
    { allowed: false, error: "bad method" });
  assert.deepEqual(decide({ trusted: true, method: "OPTIONS", path: "/health" }),
    { allowed: false, error: "bad method" });
  assert.deepEqual(decide({ trusted: true, method: "GET", path: "http://evil.example/" }),
    { allowed: false, error: "bad path" });
});

// ── policy table sanity ─────────────────────────────────────────────────────
test("read-only helper agrees with the policy table", () => {
  assert.equal(isReadOnlyBackendPath("/health"), true);
  assert.equal(isReadOnlyBackendPath("/screen-answer"), false);
  assert.equal(isReadOnlyBackendPath("/ask"), false);
  assert.equal(isReadOnlyBackendPath("/%61sk"), false);
});

// ── handler wiring (the real ipcMain.handle callback) ───────────────────────
// A refused request must resolve WITHOUT reaching the backend: the network
// call (and the token header) is only built after the policy grant. To prove
// that, http.request is replaced with a recorder for the duration of a test.
const { EventEmitter } = require("events");

function installFakeHttp(respond) {
  const http = require("http");
  const original = http.request;
  const calls = [];
  http.request = (options, callback) => {
    const call = { options, body: "" };
    calls.push(call);
    const req = new EventEmitter();
    req.write = (chunk) => { call.body += chunk; };
    req.destroy = () => {};
    req.end = () => {
      const res = new EventEmitter();
      res.statusCode = respond ? respond(call) : 200;
      callback(res);
      process.nextTick(() => {
        res.emit("data", '{"ok":true}');
        res.emit("end");
      });
    };
    return req;
  };
  return { calls, restore: () => { http.request = original; } };
}

function invokeHandler(handler, event, payload) {
  const warnings = [];
  const realWarn = console.warn;
  console.warn = (...args) => warnings.push(args.join(" "));
  return Promise.resolve(handler(event, payload))
    .then((result) => ({ result, warnings }))
    .finally(() => { console.warn = realWarn; });
}

test("registered handler refuses an untrusted overlay POST /ask", async () => {
  assert.equal(typeof backendHandler, "function", "jarvis:backend-request must be registered");
  const event = { sender: { id: 4242, getURL: () => APP_FRAME }, senderFrame: { url: APP_FRAME } };
  const { result, warnings } = await invokeHandler(backendHandler, event, { path: "/ask", method: "POST", body: { message: "hi" } });
  assert.deepEqual(result, { status: 0, error: "read-only renderer" });
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /backend-request refused/);
});

test("registered handler refuses a non-local frame even for a public read", async () => {
  const event = {
    sender: { id: 1, getURL: () => "https://evil.example/" },
    senderFrame: { url: "https://evil.example/" },
  };
  const { result } = await invokeHandler(backendHandler, event, { path: "/health", method: "GET" });
  assert.deepEqual(result, { status: 0, error: "untrusted frame" });
});

test("registered handler falls back to sender.getURL when senderFrame is absent", async () => {
  const event = { sender: { id: 4242, getURL: () => "http://127.0.0.1:9999/" } };
  const { result } = await invokeHandler(backendHandler, event, { path: "/ui-state", method: "GET" });
  assert.deepEqual(result, { status: 0, error: "untrusted frame" });
});

// ── the capsule's narrow command channel ────────────────────────────────────
test("registered capsule channel maps the action enum onto fixed endpoints", async () => {
  assert.equal(typeof capsuleHandler, "function", "jarvis:capsule-command must be registered");
  const cases = [
    ["ask", "hello jarvis", "/ask", JSON.stringify({ message: "hello jarvis" })],
    ["task-stop", "", "/task/stop", "{}"],
    ["speak-stop", "", "/speak/stop", "{}"],
  ];
  for (const [action, text, expectedPath, expectedBody] of cases) {
    const fake = installFakeHttp();
    try {
      const { result } = await invokeHandler(capsuleHandler, capsuleEvent(), { action, text });
      assert.deepEqual(result, { status: 200, text: '{"ok":true}' }, `${action} should reach the backend`);
      assert.equal(fake.calls.length, 1);
      assert.equal(fake.calls[0].options.path, expectedPath);
      assert.equal(fake.calls[0].options.method, "POST");
      assert.equal(typeof fake.calls[0].options.headers["X-Jarvis-Token"], "string");
      assert.ok(fake.calls[0].options.headers["X-Jarvis-Token"].length > 0, "token must ride along");
      assert.equal(fake.calls[0].body, expectedBody);
    } finally {
      fake.restore();
    }
  }
});

test("capsule command refuses a sender that is not the registered capsule", async () => {
  const fake = installFakeHttp();
  try {
    for (const event of [
      { sender: { id: 4242, getURL: () => CAPSULE_FRAME }, senderFrame: { url: CAPSULE_FRAME } },
      { sender: { id: 31337, getURL: () => CAPSULE_FRAME }, senderFrame: { url: CAPSULE_FRAME } },
    ]) {
      const { result, warnings } = await invokeHandler(capsuleHandler, event, { action: "ask", text: "hi" });
      assert.deepEqual(result, { status: 0, error: "not a capsule window" });
      assert.match(warnings[0], /capsule-command refused/);
    }
    assert.equal(fake.calls.length, 0, "a refused sender must not touch the backend");
  } finally {
    fake.restore();
  }
});

test("capsule command refuses the capsule window on any other page", async () => {
  const fake = installFakeHttp();
  try {
    for (const frameUrl of [APP_FRAME, "https://evil.example/", ""]) {
      const event = { sender: { id: CAPSULE_ID, getURL: () => frameUrl }, senderFrame: { url: frameUrl } };
      const { result } = await invokeHandler(capsuleHandler, event, { action: "ask", text: "hi" });
      assert.deepEqual(result, { status: 0, error: "untrusted frame" });
    }
    assert.equal(fake.calls.length, 0);
  } finally {
    fake.restore();
  }
});

test("capsule command refuses unknown actions and never names a URL", async () => {
  const fake = installFakeHttp();
  try {
    for (const action of ["backend", "/ask", "exec", "ask ", "", undefined, "ASK",
      "constructor", "__proto__", "toString", "hasOwnProperty"]) {
      const { result } = await invokeHandler(capsuleHandler, capsuleEvent(), { action, text: "hi" });
      assert.deepEqual(result, { status: 0, error: "unknown action" }, `${action} must be refused`);
    }
    assert.equal(fake.calls.length, 0);
  } finally {
    fake.restore();
  }
});

test("capsule ask payload is bounded and non-empty", async () => {
  const fake = installFakeHttp();
  try {
    const tooLong = "x".repeat(300 * 1024);
    const over = await invokeHandler(capsuleHandler, capsuleEvent(), { action: "ask", text: tooLong });
    assert.deepEqual(over.result, { status: 0, error: "text too large" });
    const empty = await invokeHandler(capsuleHandler, capsuleEvent(), { action: "ask", text: "   " });
    assert.deepEqual(empty.result, { status: 0, error: "empty text" });
    const wrongType = await invokeHandler(capsuleHandler, capsuleEvent(), { action: "ask", text: { message: "hi" } });
    assert.deepEqual(wrongType.result, { status: 0, error: "empty text" });
    assert.equal(fake.calls.length, 0);
  } finally {
    fake.restore();
  }
});

test("the capsule gains no blanket proxy authority", async () => {
  const fake = installFakeHttp();
  try {
    const mutation = await invokeHandler(backendHandler, capsuleEvent(), { path: "/ask", method: "POST", body: { message: "hi" } });
    assert.deepEqual(mutation.result, { status: 0, error: "read-only renderer" });
    const otherEndpoint = await invokeHandler(backendHandler, capsuleEvent(), { path: "/task/stop", method: "POST", body: {} });
    assert.deepEqual(otherEndpoint.result, { status: 0, error: "read-only renderer" });
    assert.equal(fake.calls.length, 0);
  } finally {
    fake.restore();
  }
});

test("the capsule's /ui-state read still works through the proxy", async () => {
  const fake = installFakeHttp();
  try {
    const { result } = await invokeHandler(backendHandler, capsuleEvent(), { path: "/ui-state", method: "GET" });
    assert.deepEqual(result, { status: 200, text: '{"ok":true}' });
    assert.equal(fake.calls.length, 1);
    assert.equal(fake.calls[0].options.path, "/ui-state");
    assert.equal(fake.calls[0].options.method, "GET");
  } finally {
    fake.restore();
  }
});

// ── preload bridge gating ───────────────────────────────────────────────────
function loadPreload(extraArgv) {
  const preloadPath = path.join(ROOT_DIR, "frontend", "preload.js");
  const savedArgv = process.argv;
  process.argv = ["node", preloadPath].concat(extraArgv || []);
  delete require.cache[preloadPath];
  const before = exposedBridges.length;
  try {
    require(preloadPath);
  } finally {
    delete require.cache[preloadPath];
    process.argv = savedArgv;
  }
  return exposedBridges[before].api;
}

test("preload gate: capsuleCommand is inert without the capsule marker", async () => {
  for (const argv of [[], ["--jarvis-trusted"]]) {
    const api = loadPreload(argv);
    assert.equal(typeof api.backend, "function");
    const before = ipcInvokes.length;
    const result = await api.capsuleCommand("ask", "hi");
    assert.deepEqual(result, { status: 0, error: "not permitted" });
    assert.equal(ipcInvokes.length, before, "no IPC may leave a non-capsule window");
  }
});

test("preload gate: the capsule sends the action enum, never a path", async () => {
  const api = loadPreload(["--jarvis-capsule"]);
  const before = ipcInvokes.length;
  const result = await api.capsuleCommand("ask", "hi");
  assert.equal(result.status, 200);
  assert.deepEqual(ipcInvokes.slice(before), [{
    channel: "jarvis:capsule-command",
    payload: { action: "ask", text: "hi" },
  }]);
});
