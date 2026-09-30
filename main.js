const { app, BrowserWindow, ipcMain, screen } = require("electron");
const capsuleMain = require("./frontend/capsule_main.js");
const { spawn } = require("child_process");
const crypto = require("crypto");
const fs = require("fs");
const http = require("http");
const path = require("path");
const { fileURLToPath } = require("url");

let backendProcess = null;
let voiceProcess   = null;
let overlayWindow  = null;
let imageOverlayWindow = null;
let researchWindow = null;

const ROOT_DIR = __dirname;
const VENV_PY = path.join(ROOT_DIR, "backend", "venv", "Scripts", "python.exe");
const PYTHON_CMD = fs.existsSync(VENV_PY) ? VENV_PY : "python";
const BACKEND_PORT = process.env.JARVIS_BACKEND_PORT || "9999";
const PRELOAD_PATH = path.join(ROOT_DIR, "frontend", "preload.js");

// ── G11 / F51 — per-launch local command token ──────────────────────────────
// Watcher mode: injected via env. Direct launch: minted here and forwarded
// to every child. Renderers obtain it ONLY through the gated preload API
// (trusted chat window) or through main-proxied backend calls — overlay
// renderers (web-derived content) can never read it.
const LOCAL_TOKEN = process.env.JARVIS_LOCAL_TOKEN ||
  crypto.randomBytes(32).toString("base64url");

// ── G11 / F52 — bounded child-log sink ─────────────────────────────────────
// Child stdout/stderr are consumed (never left blocking on a full pipe) and
// mirrored into data/logs/<label>.log, capped: past the cap the oldest half
// is dropped, so a chatty worker can never grow the log unbounded.
const LOG_DIR = path.join(ROOT_DIR, "data", "logs");
const LOG_MAX_BYTES = 512 * 1024;

function boundedLogWrite(label, chunk) {
  try {
    fs.mkdirSync(LOG_DIR, { recursive: true });
    const logPath = path.join(LOG_DIR, label + ".log");
    let existing = Buffer.alloc(0);
    try { existing = fs.readFileSync(logPath); } catch (_e) {}
    let combined = Buffer.concat([existing, Buffer.from(chunk, "utf8")]);
    if (combined.length > LOG_MAX_BYTES) {
      combined = combined.slice(combined.length - LOG_MAX_BYTES / 2);
    }
    fs.writeFileSync(logPath, combined);
  } catch (_e) { /* logging must never kill the supervisor */ }
}

// ── G11 / F52 — supervised managed processes ───────────────────────────────
// Retains the process handle + creation identity, drains stdio into the
// bounded log, and restarts a required worker within a restart BUDGET when
// it dies unexpectedly (a deliberate stop never counts against the budget).
// On a backend replacement the supervisor invalidates stale approvals and
// marks jobs interrupted before the replacement starts (safe recovery).
const _managed = new Map();
const _stopping = new WeakSet();

// Best effort before a replacement backend takes over: invalidate approvals
// and interrupt live jobs on the OLD one (a dead backend needs no call —
// its state dies with it).
function invalidateStaleBackendState() {
  const req = http.request({
    hostname: "127.0.0.1",
    port: BACKEND_PORT,
    path: "/approvals/reset",
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Jarvis-Token": LOCAL_TOKEN,
    },
    timeout: 700,
  }, (res) => { res.resume(); });
  req.on("error", () => {});
  req.on("timeout", () => req.destroy());
  req.end(JSON.stringify({}));
}

function attachChildHandlers(child, entry, onExit) {
  // G11 / F52 — drain stdio into the bounded log (never a full pipe).
  if (child.stdout && typeof child.stdout.on === "function") {
    child.stdout.on("data", (chunk) => boundedLogWrite(entry.label, chunk));
    child.stderr.on("data", (chunk) => boundedLogWrite(entry.label, chunk));
  }

  child.on("exit", (code) => {
    const wasDeliberate = entry.deliberate || _stopping.has(child);
    if (onExit) onExit();
    if (!entry.restartable || wasDeliberate) return;
    if (_managed.get(entry.label) !== entry) return; // superseded
    if (entry.restarts >= entry.restartBudget) {
      console.error(`[${entry.label}] exited after ${entry.restartBudget} restarts — giving up.`);
      return;
    }
    entry.restarts += 1;
    console.log(`[${entry.label}] exited (code ${code}) — supervised restart ` +
      `${entry.restarts}/${entry.restartBudget} in 1.5s`);
    setTimeout(() => {
      if (_managed.get(entry.label) !== entry) return;
      console.log(`[${entry.label}] supervised restart...`);
      if (entry.label === "backend") invalidateStaleBackendState();
      if (entry.spawnArgs) {
        const [args, spawnOpts] = entry.spawnArgs();
        const next = spawn(args[0], args.slice(1), Object.assign({
          stdio: ["pipe", "pipe", "pipe"],
        }, spawnOpts || {}));
        // Preserve the restart count — a fresh entry would reset the budget
        // and crash-loop forever.
        const nextEntry = {
          label: entry.label, restartable: entry.restartable,
          restartBudget: entry.restartBudget, spawnArgs: entry.spawnArgs,
          restarts: entry.restarts, deliberate: false,
        };
        _managed.set(entry.label, nextEntry);
        attachChildHandlers(next, nextEntry, onExit);
        if (entry.label === "backend") backendProcess = next;
        if (entry.label === "voice") voiceProcess = next;
      }
    }, 1500);
  });

  child.on("error", (error) => {
    console.error("Managed process error:", error);
  });
}

function registerManagedProcess(child, optsOrHandler, maybeOpts) {
  if (!child) {
    return child;
  }
  // Back-compat: the old signature was (child, onExitFn).
  const opts = typeof optsOrHandler === "function" ? {} : (optsOrHandler || {});
  const onExit = typeof optsOrHandler === "function" ? optsOrHandler : null;
  const label = opts.label || "worker";
  const entry = {
    label,
    restartable: !!opts.restartable,
    restartBudget: opts.restartBudget != null ? opts.restartBudget : 2,
    spawnArgs: opts.spawnArgs || null,   // () => [argsArray, spawnOpts]
    restarts: 0,
    deliberate: false,
  };
  _managed.set(label, entry);
  attachChildHandlers(child, entry, onExit);
  return child;
}

function killManagedProcess(child) {
  if (!child || child.killed || !child.pid) {
    return;
  }
  _stopping.add(child);
  spawn("taskkill", ["/PID", String(child.pid), "/T", "/F"], {
    windowsHide: true,
    detached: false,
  });
}

// Best-effort kill of whatever listens on a local port (warm opencode serve
// daemon, brave-control MCP daemon). Direct-launch mode owns no watcher to
// do this, so a hard backend kill must still leave no zombie daemon behind.
// The external-runtime path needs no equivalent: the watcher's stop_runtime
// tears the daemons down itself.
function killPortListener(port) {
  const script = [
    "$p = Get-NetTCPConnection -LocalPort " + port + " -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique",
    "foreach ($id in $p) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }",
  ].join("; ");

  spawn("powershell", ["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script], {
    windowsHide: true,
    detached: false,
  });
}

function killOpencodeServer() {
  killPortListener(Number(process.env.JARVIS_OPENCODE_PORT || "9560"));
}

function killBraveMcpDaemon() {
  killPortListener(Number(process.env.BRAVE_MCP_PORT || "9570"));
}

function killWhisperDaemon() {
  const script = [
    "$w = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'backend.whisper_daemon' }",
    "foreach ($proc in $w) { Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue }",
  ].join("; ");
  spawn("powershell", ["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script], {
    windowsHide: true,
    detached: false,
  });
}

function requestWatcherStop(callback) {
  const port = Number(process.env.JARVIS_WATCHER_CONTROL_PORT || "0");
  if (!port) {
    callback(false);
    return;
  }

  let done = false;
  const finish = (ok) => {
    if (done) return;
    done = true;
    callback(ok);
  };

  const req = http.request(
    {
      hostname: "127.0.0.1",
      port,
      path: "/stop",
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        // G11 / F51 — the watcher control plane is a token-authed command
        // endpoint; the watcher injected this very token into our env.
        "X-Jarvis-Token": LOCAL_TOKEN,
      },
      timeout: 1500,
    },
    (res) => {
      res.resume();
      res.on("end", () => finish(res.statusCode >= 200 && res.statusCode < 300));
    }
  );
  req.on("error", () => finish(false));
  req.on("timeout", () => {
    req.destroy();
    finish(false);
  });
  req.end("{}");
}

// ── G11 / F51 — hardened window factory ─────────────────────────────────────
// Every renderer runs sandboxed + context-isolated with the minimal preload
// bridge. Renderer-initiated navigation is denied outright and window.open
// is denied; external links must go through the validated open-external IPC.
const TRUSTED_PRELOAD_ARGS = ["--jarvis-trusted"];

function secureWebPreferences(trusted) {
  return {
    contextIsolation: true,
    nodeIntegration: false,
    sandbox: true,
    preload: PRELOAD_PATH,
    additionalArguments: trusted ? TRUSTED_PRELOAD_ARGS : [],
  };
}

function hardenWindow(win) {
  try {
    win.webContents.on("will-navigate", (event, url) => {
      // Only the app's own local files may load; any renderer-initiated
      // navigation to the network is denied.
      const allowed = url.startsWith("file://") && url.includes(path.join("frontend") + path.sep);
      if (!allowed) event.preventDefault();
    });
    win.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  } catch (_e) { /* hardening must never break window creation */ }
  return win;
}

function isSafeExternalUrl(raw) {
  const value = String(raw || "").trim();
  if (!/^https?:\/\//i.test(value)) return false;
  try {
    const parsed = new URL(value);
    return (parsed.protocol === "http:" || parsed.protocol === "https:");
  } catch (_e) {
    return false;
  }
}

const _trustedWebContents = new Set();

// ── G11 / F51 — validated IPC surface for the sandboxed renderers ───────────
ipcMain.on("jarvis:open-external", (event, rawUrl) => {
  const url = String(rawUrl || "");
  if (!isSafeExternalUrl(url)) {
    console.warn("[SECURITY] open-external rejected:", url.slice(0, 120));
    return;
  }
  const { shell } = require("electron");
  shell.openExternal(url);
});

ipcMain.handle("jarvis:get-config", () => ({
  backendPort: BACKEND_PORT,
  build: process.env.JARVIS_BUILD_ID || "dev",
}));

ipcMain.handle("jarvis:get-local-secret", (event) => {
  // The per-launch token is exposed ONLY to the trusted chat window; overlay
  // windows (which render web-derived content) can never obtain it.
  if (!event.sender || !_trustedWebContents.has(event.sender.id)) {
    return "";
  }
  return LOCAL_TOKEN;
});

// ── G11 / F51 — rendering is not authority: per-sender proxy policy ─────────
// The chat window is the only renderer trusted with computer-control
// authority. Every other renderer (overlays, capsule) renders content the app
// did not author, so it may only perform the idempotent GET reads allowlisted
// below; every mutating request is refused BEFORE the token is attached.
// A frame that is not a local app frame gets nothing at all.
const MUTATING_BACKEND_METHODS = new Set(["POST", "PUT", "PATCH", "DELETE"]);
const PROXY_BACKEND_METHODS = new Set(["GET", "POST", "PUT", "PATCH", "DELETE"]);

// Idempotent read paths a NON-trusted renderer may call (path only, no query).
const READ_ONLY_BACKEND_PATHS = [
  /^\/health$/,
  /^\/ui-state$/,
  /^\/voice-state$/,
  /^\/research-result(?:\/.*)?$/,
  /^\/research-progress(?:\/.*)?$/,
  /^\/ask\/status(?:\/.*)?$/,
  /^\/settings(?:\/[A-Za-z0-9._-]+)?$/,
];

// A packaged build could serve the UI from app://; only an explicitly
// configured origin counts, so the default (env unset) grants nothing.
const APP_ORIGIN = process.env.JARVIS_APP_ORIGIN || "";

// The local app file tree is the only frame root that may ever be granted
// anything (this app loads every window with loadFile()).
function isLocalAppFrameUrl(rawUrl, rootDir, appOrigin) {
  const value = String(rawUrl || "");
  if (!value) return false;
  let parsed;
  try {
    parsed = new URL(value);
  } catch (_e) {
    return false;
  }
  if (parsed.protocol === "app:") {
    // Non-special schemes report origin "null", so compare protocol + host.
    const declared = String(appOrigin || "").trim().replace(/\/+$/, "");
    return !!declared && (parsed.protocol + "//" + parsed.host) === declared;
  }
  if (parsed.protocol !== "file:") return false;
  let framePath;
  try {
    framePath = fileURLToPath(parsed);
  } catch (_e) {
    return false;
  }
  const rel = path.relative(path.resolve(rootDir), path.resolve(framePath));
  return rel === "" || (!rel.startsWith("..") && !path.isAbsolute(rel));
}

// Percent escapes, backslashes, dot segments and control characters are
// refused rather than matched loosely: the backend decodes them, so
// "/%75i-state" would otherwise route to "/ui-state" past the allowlist.
const UNSAFE_READ_PATH = /[%\\\u0000-\u001f]/;

function normalizeReadPath(rawPath) {
  let value = String(rawPath == null ? "" : rawPath).trim();
  const cut = value.search(/[?#]/);
  if (cut >= 0) value = value.slice(0, cut);
  if (value.length > 1 && value.endsWith("/")) value = value.slice(0, -1);
  return value || "/";
}

function isCanonicalReadPath(reqPath) {
  if (!reqPath.startsWith("/")) return false;
  if (UNSAFE_READ_PATH.test(reqPath)) return false;
  if (reqPath.includes("//") || reqPath.includes("/./") || reqPath.includes("/../")) return false;
  if (reqPath.endsWith("/.") || reqPath.endsWith("/..")) return false;
  return true;
}

function isReadOnlyBackendPath(rawPath) {
  const reqPath = normalizeReadPath(rawPath);
  if (!isCanonicalReadPath(reqPath)) return false;
  return READ_ONLY_BACKEND_PATHS.some((pattern) => pattern.test(reqPath));
}

// PURE decision function (see tests/backend-request-policy.test.js): returns
// { allowed: true, path } or { allowed: false, error }. It never reads the
// token — the caller attaches it only after `allowed`.
function evaluateBackendRequestPolicy(opts) {
  const options = opts || {};
  const method = String(options.method == null ? "GET" : options.method).toUpperCase();
  if (!options.localFrame) return { allowed: false, error: "untrusted frame" };
  if (!PROXY_BACKEND_METHODS.has(method)) return { allowed: false, error: "bad method" };
  const rawPath = String(options.path == null ? "/" : options.path);
  if (!rawPath.startsWith("/")) return { allowed: false, error: "bad path" };
  // The trusted chat window keeps its full existing surface (raw path, query
  // string included) — only its webContents id is in _trustedWebContents.
  if (options.trusted) return { allowed: true, path: rawPath };
  if (MUTATING_BACKEND_METHODS.has(method)) return { allowed: false, error: "read-only renderer" };
  if (!isReadOnlyBackendPath(rawPath)) return { allowed: false, error: "path not readable" };
  return { allowed: true, path: normalizeReadPath(rawPath) };
}

// The frame that sent the IPC, not the window: a non-app subframe must never
// inherit the top frame's authority.
function senderFrameUrl(event) {
  try {
    const frame = event && event.senderFrame;
    if (frame && typeof frame.url === "string" && frame.url) return frame.url;
  } catch (_e) { /* frame already gone */ }
  try {
    const sender = event && event.sender;
    if (sender && typeof sender.getURL === "function") return sender.getURL();
  } catch (_e) { /* no sender */ }
  return "";
}

function isTrustedSender(event) {
  try {
    return !!(event && event.sender && _trustedWebContents.has(event.sender.id));
  } catch (_e) {
    return false;
  }
}

const MAX_PROXY_BODY = 256 * 1024;

// The ONE place a request leaves with the local token. Both the renderer
// proxy and the capsule command channel reach the backend through here, and
// only after their policy granted authority.
function backendRequestPromise(reqPath, method, body) {
  return new Promise((resolve) => {
    let bodyText = null;
    if (method !== "GET") {
      try {
        bodyText = JSON.stringify(body == null ? {} : body);
      } catch (_e) {
        resolve({ status: 0, error: "bad body" });
        return;
      }
      if (bodyText.length > MAX_PROXY_BODY) {
        resolve({ status: 0, error: "body too large" });
        return;
      }
    }
    const headers = { "X-Jarvis-Token": LOCAL_TOKEN };
    if (bodyText !== null) {
      headers["Content-Type"] = "application/json";
      headers["Content-Length"] = Buffer.byteLength(bodyText);
    }
    const req = http.request({
      hostname: "127.0.0.1",
      port: BACKEND_PORT,
      path: reqPath,
      method,
      headers,
      timeout: 20000,
    }, (res) => {
      let data = "";
      res.on("data", (chunk) => {
        data += chunk;
        if (data.length > 2 * 1024 * 1024) req.destroy(); // bounded read
      });
      res.on("end", () => resolve({ status: res.statusCode, text: data }));
    });
    req.on("error", (err) => resolve({ status: 0, error: String(err) }));
    req.on("timeout", () => { req.destroy(); resolve({ status: 0, error: "timeout" }); });
    if (bodyText !== null) req.write(bodyText);
    req.end();
  });
}

ipcMain.handle("jarvis:backend-request", (event, payload) => {
  const method = String((payload && payload.method) || "GET").toUpperCase();
  const requestedPath = String((payload && payload.path) || "/");
  const frameUrl = senderFrameUrl(event);
  const decision = evaluateBackendRequestPolicy({
    trusted: isTrustedSender(event),
    localFrame: isLocalAppFrameUrl(frameUrl, ROOT_DIR, APP_ORIGIN),
    method,
    path: requestedPath,
  });
  if (!decision.allowed) {
    // A refused sender never reaches the token or the backend.
    console.warn("[SECURITY] backend-request refused (" + decision.error + "):",
      method, requestedPath.slice(0, 120), frameUrl.slice(0, 120));
    return { status: 0, error: decision.error };
  }
  return backendRequestPromise(decision.path, method, (payload && payload.body) || null);
});

// ── G11 / F51 — the capsule's narrow, main-validated command channel ────────
// The capsule is Jarvis's OWN window (it renders app state, not web-derived
// content), so it needs its three legitimate controls to keep working. It
// never names a URL: it sends an ACTION from a fixed enum, main maps it to a
// backend endpoint, and the blanket proxy above stays closed to it. Only the
// webContents registered by frontend/capsule_main.js at window creation, at
// the capsule page itself, may call this.
const CAPSULE_PAGE_PATH = path.join(ROOT_DIR, "frontend", "capsule.html");
const CAPSULE_COMMANDS = {
  "ask": { method: "POST", path: "/ask" },
  "task-stop": { method: "POST", path: "/task/stop" },
  "speak-stop": { method: "POST", path: "/speak/stop" },
};

function isSameLocalFile(rawUrl, targetPath) {
  try {
    const parsed = new URL(String(rawUrl || ""));
    if (parsed.protocol !== "file:") return false;
    const rel = path.relative(path.resolve(targetPath), path.resolve(fileURLToPath(parsed)));
    return rel === "";
  } catch (_e) {
    return false;
  }
}

function isCapsuleSender(event) {
  try {
    if (!event || !event.sender) return false;
    return !!capsuleMain.isCapsuleSender && capsuleMain.isCapsuleSender(event.sender.id);
  } catch (_e) {
    return false;
  }
}

// PURE decision function (see tests/backend-request-policy.test.js): maps a
// capsule action onto exactly one backend call, or refuses. The renderer can
// never influence the request path.
function evaluateCapsuleCommandPolicy(opts) {
  const options = opts || {};
  if (!options.senderIsCapsule) return { allowed: false, error: "not a capsule window" };
  if (!options.frameIsCapsule) return { allowed: false, error: "untrusted frame" };
  const action = String(options.action == null ? "" : options.action);
  // Own-property lookup only: "constructor"/"__proto__"/"toString" must not
  // resolve to an inherited object and slip through as an allowed command.
  const command = Object.prototype.hasOwnProperty.call(CAPSULE_COMMANDS, action)
    ? CAPSULE_COMMANDS[action]
    : null;
  if (!command) return { allowed: false, error: "unknown action" };
  if (action !== "ask") {
    return { allowed: true, method: command.method, path: command.path, body: {} };
  }
  const text = typeof options.text === "string" ? options.text : "";
  if (!text.trim()) return { allowed: false, error: "empty text" };
  const body = { message: text };
  if (Buffer.byteLength(JSON.stringify(body)) > MAX_PROXY_BODY) {
    return { allowed: false, error: "text too large" };
  }
  return { allowed: true, method: command.method, path: command.path, body };
}

ipcMain.handle("jarvis:capsule-command", (event, payload) => {
  const frameUrl = senderFrameUrl(event);
  const decision = evaluateCapsuleCommandPolicy({
    senderIsCapsule: isCapsuleSender(event),
    frameIsCapsule: isSameLocalFile(frameUrl, CAPSULE_PAGE_PATH),
    action: payload && payload.action,
    text: payload && payload.text,
  });
  if (!decision.allowed) {
    console.warn("[SECURITY] capsule-command refused (" + decision.error + "):",
      String((payload && payload.action) || "").slice(0, 40), frameUrl.slice(0, 120));
    return { status: 0, error: decision.error };
  }
  return backendRequestPromise(decision.path, decision.method, decision.body);
});

ipcMain.handle("jarvis:read-report-mirror", () => {
  const latestPath = path.join(ROOT_DIR, "data", "research_reports", "latest.json");
  try {
    if (!fs.existsSync(latestPath)) return null;
    return JSON.parse(fs.readFileSync(latestPath, "utf8"));
  } catch (_e) {
    return null;
  }
});

function fallbackExternalCleanup() {
  const script = [
    "$h = $null",
    "try { $h = Invoke-RestMethod -Uri http://127.0.0.1:" + BACKEND_PORT + "/health -TimeoutSec 1 } catch {}",
    "if ($h -and $h.service -eq 'jarvis-backend') { $p = Get-NetTCPConnection -LocalPort " + BACKEND_PORT + " -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique; foreach ($id in $p) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue } }",
    "$v = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'backend.voice_mode' }",
    "foreach ($proc in $v) { Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue }",
    "$w = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'backend.whisper_daemon' }",
    "foreach ($proc in $w) { Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue }",
  ].join("; ");

  spawn("powershell", ["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script], {
    windowsHide: true,
    detached: false,
  });
}

function waitForBackendReady(onReady, attempts = 100) {
  let done = false;
  const finish = (ready) => {
    if (done) return;
    done = true;
    onReady(ready);
  };
  const retry = () => {
    if (done) return;
    if (backendProcess && backendProcess.exitCode !== null) {
      finish(false);
      return;
    }
    done = true;
    retryBackendReady(onReady, attempts);
  };
  const req = http.get("http://127.0.0.1:" + BACKEND_PORT + "/health", (res) => {
    let body = "";
    res.on("data", (chunk) => { body += chunk; });
    res.on("end", () => {
      if (res.statusCode === 200 && body.includes("jarvis-backend")) {
        finish(true);
        return;
      }
      retry();
    });
  });
  req.on("error", retry);
  req.setTimeout(800, () => {
    req.destroy();
    retry();
  });
}

function retryBackendReady(onReady, attempts) {
  if (attempts <= 1) {
    onReady(false);
    return;
  }
  setTimeout(() => waitForBackendReady(onReady, attempts - 1), 400);
}

function createWindow() {
  const win = new BrowserWindow({
    width: 900,
    height: 650,
    webPreferences: secureWebPreferences(true),
  });
  hardenWindow(win);
  // The chat window is the TRUSTED renderer: only its preload may ask for
  // the per-launch token (it needs the header on its own /ask/stream fetch).
  try {
    const senderId = win.webContents.id;
    if (senderId != null) _trustedWebContents.add(senderId);
  } catch (_e) {}

  win.loadFile(path.join(ROOT_DIR, "frontend", "index.html"));

  if (process.env.JARVIS_EXTERNAL_RUNTIME === "1") {
    console.log("Jarvis runtime already launched by watcher");
    // Overlay poll still needed when launched externally. The research
    // window itself is only shown when a fresh report arrives.
    setTimeout(() => {
      startOverlayPolling();
    }, 3000);
    return;
  }

  startJarvis();
}

app.whenReady().then(() => {
  createWindow();
  try { capsuleMain.initCapsule(); } catch (_e) {}
});


// ── START ────────────────────────────────────────
function startJarvis() {
  if (backendProcess || voiceProcess) {
    console.log("Jarvis already running");
    return;
  }

  console.log("Starting Jarvis...");

  // G11 / F51 — every child carries the per-launch local command token.
  const childEnv = Object.assign({}, process.env, {
    JARVIS_LOCAL_TOKEN: LOCAL_TOKEN,
  });

  // Backend — visible so you can see responses and logs
  // [P1-14] --no-access-log: the access log is synchronous I/O on the event
  // loop, so it adds latency to EVERY request, barge-in included.
  const backendArgs = () => [[
    PYTHON_CMD,
    "-u",
    "-m",
    "uvicorn",
    "backend.main:app",
    "--host",
    "127.0.0.1",
    "--port",
    BACKEND_PORT,
    "--no-access-log",
  ], {
    cwd: ROOT_DIR,
    env: childEnv,
    windowsHide: false,   // ✅ keep this one visible
    detached:    false,
  }];
  backendProcess = spawn(...backendArgs());

  // G11 / F52 — supervised: stdio drained to the bounded log; unexpected
  // deaths restart within a budget (deliberate stops never count).
  registerManagedProcess(backendProcess, {
    label: "backend",
    restartable: true,
    restartBudget: 2,
    spawnArgs: backendArgs,
  });

  waitForBackendReady((ready) => {
    if (!ready) {
      console.error("Backend did not become ready; voice mode not started.");
      return;
    }

    const voiceArgs = () => [[
      PYTHON_CMD,
      "-u",
      "-m",
      "backend.voice_mode",
    ], {
      cwd: ROOT_DIR,
      env: Object.assign({}, childEnv, {
        JARVIS_BACKEND_PID: backendProcess && backendProcess.pid ? String(backendProcess.pid) : "",
        JARVIS_ELECTRON_PID: String(process.pid),
      }),
      windowsHide: true,
      detached:    false,
    }];
    voiceProcess = spawn(...voiceArgs());
    registerManagedProcess(voiceProcess, {
      label: "voice",
      restartable: true,
      restartBudget: 1,
      spawnArgs: voiceArgs,
    });
    startOverlayPolling();
  });
}


ipcMain.on("start-jarvis", startJarvis);


// ── STOP ─────────────────────────────────────────
ipcMain.on("stop-jarvis", () => {
  console.log("Stopping Jarvis...");
  stopOverlayPolling();
  lastOverlayAnswerId = 0;
  lastResearchId = 0;

  if (overlayWindow && !overlayWindow.isDestroyed()) {
    overlayWindow.setIgnoreMouseEvents(true, { forward: true });
    overlayWindow.hide();
  }
  if (imageOverlayWindow && !imageOverlayWindow.isDestroyed()) {
    imageOverlayWindow.setIgnoreMouseEvents(true, { forward: true });
    imageOverlayWindow.hide();
  }
  if (researchWindow && !researchWindow.isDestroyed()) {
    researchWindow.setIgnoreMouseEvents(true, { forward: true });
    researchWindow.hide();
  }

  if (process.env.JARVIS_EXTERNAL_RUNTIME === "1") {
    requestWatcherStop((ok) => {
      if (!ok) {
        fallbackExternalCleanup();
      }
      app.quit();
    });
    return;
  }

  killManagedProcess(voiceProcess);
  killManagedProcess(backendProcess);
  backendProcess = null;
  voiceProcess   = null;

  // No watcher in direct-launch mode to tear down the warm opencode session
  // server or the brave-control MCP daemon; kill anything still listening on
  // their ports so no zombie survives.
  killOpencodeServer();
  killBraveMcpDaemon();
  killWhisperDaemon();

  // In direct-launch mode there is no watcher to resume, so quit the app.
  setTimeout(() => app.quit(), 500);
});


// ── RIGHT OVERLAY WINDOW (TIP + LINKS + EVIDENCE) ──
function createOverlayWindow() {
  if (overlayWindow && !overlayWindow.isDestroyed()) {
    overlayWindow.show();
    return overlayWindow;
  }

  const primaryDisplay = screen.getPrimaryDisplay();
  const { width, height } = primaryDisplay.workAreaSize;

  overlayWindow = new BrowserWindow({
    width: 460,
    height: height,
    x: width - 460,
    y: 0,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    skipTaskbar: true,
    resizable: false,
    hasShadow: false,
    focusable: false,
    webPreferences: secureWebPreferences(false),
  });
  hardenWindow(overlayWindow);

  overlayWindow.loadFile(path.join(ROOT_DIR, "frontend", "overlay.html"));
  overlayWindow.setIgnoreMouseEvents(true, { forward: true });

  overlayWindow.on("closed", () => {
    overlayWindow = null;
  });

  return overlayWindow;
}

// ── LEFT OVERLAY WINDOW (IMAGES) ─────────────────
function createImageOverlayWindow() {
  if (imageOverlayWindow && !imageOverlayWindow.isDestroyed()) {
    imageOverlayWindow.show();
    return imageOverlayWindow;
  }

  const primaryDisplay = screen.getPrimaryDisplay();
  const { height } = primaryDisplay.workAreaSize;

  imageOverlayWindow = new BrowserWindow({
    width: 420,
    height: height,
    x: 0,
    y: 0,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    skipTaskbar: true,
    resizable: false,
    hasShadow: false,
    focusable: false,
    webPreferences: secureWebPreferences(false),
  });
  hardenWindow(imageOverlayWindow);

  imageOverlayWindow.loadFile(path.join(ROOT_DIR, "frontend", "overlay_images.html"));
  imageOverlayWindow.setIgnoreMouseEvents(true, { forward: true });

  imageOverlayWindow.on("closed", () => {
    imageOverlayWindow = null;
  });

  return imageOverlayWindow;
}


// ── IPC: Dismiss overlays ────────────────────────
ipcMain.on("dismiss-overlay", () => {
  if (overlayWindow && !overlayWindow.isDestroyed()) {
    overlayWindow.hide();
  }
});

ipcMain.on("dismiss-image-overlay", () => {
  if (imageOverlayWindow && !imageOverlayWindow.isDestroyed()) {
    imageOverlayWindow.hide();
  }
});

// ── RESEARCH OVERLAY WINDOW (glass report) ────────
function createResearchWindow() {
  if (researchWindow && !researchWindow.isDestroyed()) {
    return researchWindow;
  }

const display = screen.getPrimaryDisplay();
  const wa = display.workAreaSize;
  const w = Math.min(1040, wa.width - 120);
  const h = Math.min(780, wa.height - 140);

  researchWindow = new BrowserWindow({
    width: w,
    height: h,
    x: Math.round((wa.width - w) / 2),
    y: Math.round((wa.height - h) / 2),
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    skipTaskbar: true,
    resizable: true,
    hasShadow: false,
    focusable: true,
    webPreferences: secureWebPreferences(false),
  });
  hardenWindow(researchWindow);

  researchWindow.loadFile(path.join(ROOT_DIR, "frontend", "research_overlay.html"));
  researchWindow.setIgnoreMouseEvents(false);

  researchWindow.on("closed", () => {
    researchWindow = null;
  });

  return researchWindow;
}

// ── IPC: dismiss research overlay ─────────────────
ipcMain.on("dismiss-research-overlay", () => {
  // F28: respect an explicit mid-run dismissal — live progress will not
  // re-open the window (only a freshly finished report will).
  researchWindowDismissed = true;
  if (researchWindow && !researchWindow.isDestroyed()) {
    researchWindow.hide();
  }
});

// ── IPC: minimize research overlay ────────────────
ipcMain.on("minimize-research-overlay", () => {
  if (researchWindow && !researchWindow.isDestroyed()) {
    researchWindow.minimize();
  }
});

// ── IPC: renderer found a new report in the mirror file ──
ipcMain.on("request-show-research-window", () => {
  const win = createResearchWindow();
  win.show();
  win.showInactive();
  win.setIgnoreMouseEvents(false);
});

// ── IPC: Mouse enter/leave for right overlay ─────
// ── IPC: Mouse enter/leave for image overlay ─────

// ── Poll backend for screen answers ──────────────
let lastOverlayAnswerId = 0;
// F30 — the answer id stays the same when decorations arrive later; the
// revision tells us the same answer gained links/images and must re-render.
let lastOverlayAnswerRevision = -1;

async function pollForScreenAnswer() {
  try {
    const http = require("http");

    return new Promise((resolve) => {
      const req = http.get({
        hostname: "127.0.0.1",
        port: BACKEND_PORT,
        path: "/screen-answer",
        headers: { "X-Jarvis-Token": LOCAL_TOKEN },
        timeout: 2500,
      }, (res) => {
        let body = "";
        res.on("data", (chunk) => { body += chunk; });
        res.on("end", () => {
          try {
            const data = JSON.parse(body);
            if (data && data.id) {
              const revision = data.revision || 0;
              const isNew = data.id !== lastOverlayAnswerId;
              // Re-render for a new answer, or when the same answer was
              // enriched (F30 publish-early / enrich-later).
              if (isNew || revision !== lastOverlayAnswerRevision) {
                lastOverlayAnswerId = data.id;
                lastOverlayAnswerRevision = revision;
                showScreenOverlay(data);
              }
            }
          } catch (_e) { /* ignore */ }
          resolve();
        });
      });
      req.on("error", () => resolve());
      req.setTimeout(2000, () => { req.destroy(); resolve(); });
    });
  } catch (_err) {
    // Silently ignore
  }
}

function showScreenOverlay(data) {
  // ── Right overlay: TIP + LINKS + EVIDENCE ──
  const win = createOverlayWindow();
  win.setIgnoreMouseEvents(false);

  if (win.webContents.isLoading()) {
    win.webContents.on("did-finish-load", () => {
      win.webContents.send("show-screen-answer", data);
    });
  } else {
    win.webContents.send("show-screen-answer", data);
  }
  win.show();

  // ── Left overlay: IMAGES (only if images present) ──
  const images = data.images || [];
  if (images.length > 0) {
    const imgWin = createImageOverlayWindow();
    imgWin.setIgnoreMouseEvents(false);

    if (imgWin.webContents.isLoading()) {
      imgWin.webContents.on("did-finish-load", () => {
        imgWin.webContents.send("show-screen-images", images);
      });
    } else {
      imgWin.webContents.send("show-screen-images", images);
    }
    imgWin.show();
  } else if (imageOverlayWindow && !imageOverlayWindow.isDestroyed()) {
    if (imageOverlayWindow.webContents.isLoading()) {
      imageOverlayWindow.webContents.on("did-finish-load", () => {
        imageOverlayWindow.webContents.send("show-screen-images", []);
      });
    } else {
      imageOverlayWindow.webContents.send("show-screen-images", []);
    }
  }
}

// ── Research report: directed-glass overlay ─────────
let lastResearchId = 0;
// F28 — the overlay surfaces DURING a deep run, not only when the report
// lands. This tracks the newest /research-progress event handled.
let lastResearchProgressId = 0;
// User dismissed the overlay mid-run — do not auto-reopen it until a final
// report (or a later run's report) arrives.
let researchWindowDismissed = false;

function sendToWindow(win, channel, data) {
  if (win.isDestroyed()) return;
  if (win.webContents.isLoading()) {
    win.webContents.on("did-finish-load", () => {
      try { win.webContents.send(channel, data); } catch (_e) {}
    });
  } else {
    try { win.webContents.send(channel, data); } catch (_e) {}
  }
}

function showResearch(data) {
  const win = createResearchWindow();
  researchWindowDismissed = false;
  win.show();
  win.showInactive();
  win.setIgnoreMouseEvents(false);
  sendToWindow(win, "show-research-result", data);
}

function readLatestResearch() {
  const latestPath = path.join(ROOT_DIR, "data", "research_reports", "latest.json");
  if (!fs.existsSync(latestPath)) return null;
  try {
    return JSON.parse(fs.readFileSync(latestPath, "utf8"));
  } catch (_e) {
    return null;
  }
}

// Set at app boot: research reports pushed before this moment belong to a
// previous session and must never pop the overlay again.
let overlayStartMs = Date.now();

function maybeShowResearch(data) {
  if (
    data && data.id &&
    data.id > overlayStartMs &&
    data.id !== lastResearchId
  ) {
    lastResearchId = data.id;
    showResearch(data);
  }
}

async function pollForResearchResult() {
  try {
    // 1) backend API (fast path)
    await new Promise((resolve) => {
      const req = http.get({
        hostname: "127.0.0.1",
        port: BACKEND_PORT,
        path: "/research-result",
        headers: { "X-Jarvis-Token": LOCAL_TOKEN },
        timeout: 2500,
      }, (res) => {
        let body = "";
        res.on("data", (chunk) => { body += chunk; });
        res.on("end", () => {
          try {
            maybeShowResearch(JSON.parse(body));
          } catch (_e) { /* stale backend → not json */ }
          resolve();
        });
      });
      req.on("error", () => resolve());
      req.setTimeout(2000, () => { req.destroy(); resolve(); });
    });

    // 2) file mirror — drives the overlay even with a stale backend
    maybeShowResearch(readLatestResearch());
  } catch (_err) {
    return;
  }
}

async function pollForResearchProgress() {
  try {
    const http = require("http");

    return await new Promise((resolve) => {
      const req = http.get("http://127.0.0.1:" + BACKEND_PORT + "/research-progress", (res) => {
        let body = "";
        res.on("data", (chunk) => { body += chunk; });
        res.on("end", () => {
          try {
            const d = JSON.parse(body);
            if (
              d && d.id &&
              d.id > overlayStartMs &&
              d.id !== lastResearchProgressId &&
              !researchWindowDismissed
            ) {
              lastResearchProgressId = d.id;
              // Only for a run whose final report has NOT landed yet — the
              // report's completion timestamp is always newer than its own
              // progress events, so this can never re-pop a finished report.
              if (!lastResearchId || d.id > lastResearchId) {
                const win = createResearchWindow();
                if (!win.isVisible()) {
                  // Renderer fills the panel from /research-progress itself.
                  win.show();
                  win.showInactive();
                  win.setIgnoreMouseEvents(false);
                }
              }
            } else if (d && d.id) {
              lastResearchProgressId = d.id;
            }
          } catch (_e) { /* stale backend → not json */ }
          resolve();
        });
      });
      req.on("error", () => resolve());
      req.setTimeout(2000, () => { req.destroy(); resolve(); });
    });
  } catch (_err) {
    return;
  }
}

let overlayPollInterval = null;

function startOverlayPolling() {
  if (overlayPollInterval) return;
  overlayPollInterval = setInterval(() => {
    pollForScreenAnswer();
    pollForResearchProgress();
    pollForResearchResult();
  }, 800);
}

function stopOverlayPolling() {
  if (!overlayPollInterval) return;
  clearInterval(overlayPollInterval);
  overlayPollInterval = null;
}

// ── G11 / F51 — test surface ────────────────────────────────────────────────
// The proxy policy above is pure, so tests/backend-request-policy.test.js can
// verify who may call what without booting Electron. Nothing in the app
// requires this entry point, so exporting it changes no runtime behaviour.
module.exports = {
  evaluateBackendRequestPolicy,
  evaluateCapsuleCommandPolicy,
  isLocalAppFrameUrl,
  isReadOnlyBackendPath,
  isSameLocalFile,
  normalizeReadPath,
  CAPSULE_PAGE_PATH,
};
