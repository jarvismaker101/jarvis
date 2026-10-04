// Model sidebar — every feature's models must be listed and selectable.
//
// Regression guard for the bug where `jarvisFetch` called ITSELF instead of
// delegating to the platform fetch, so every settings/models call died with
// "Maximum call stack size exceeded" and no model could be selected for any
// role. It also pins the role coverage: each registry role must render its
// allowlisted providers and expose clickable model options that POST
// /settings/model with the right role.
//
// renderer.js is a plain browser script (not a module), so it runs in a vm
// context over a minimal DOM stub. The repo has no JS test harness
// (package.json "test" is a stub), so this uses Node's built-in runner:
//
//     node --test tests/model-sidebar.test.js
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const RENDERER_PATH = path.join(__dirname, "..", "frontend", "renderer.js");
const LAUNCH_TOKEN = "test-launch-token-0123456789abcdef";

// ── minimal DOM stub ────────────────────────────────────────────────────────
function makeElement(tag) {
  const el = {
    tagName: String(tag || "div").toUpperCase(),
    _className: "",
    children: [],
    parentNode: null,
    dataset: {},
    style: {},
    value: "",
    textContent: "",
    placeholder: "",
    title: "",
    type: "",
    disabled: false,
    hidden: false,
    _listeners: {},
    _attrs: {},
    get className() { return this._className; },
    set className(v) { this._className = String(v == null ? "" : v); },
    get classList() {
      const self = this;
      const read = () => new Set(self._className.split(/\s+/).filter(Boolean));
      const write = (s) => { self._className = [...s].join(" "); };
      return {
        add(...c) { const s = read(); c.forEach((x) => s.add(x)); write(s); },
        remove(...c) { const s = read(); c.forEach((x) => s.delete(x)); write(s); },
        contains(c) { return read().has(c); },
        toggle(c, force) {
          const s = read();
          const on = force === undefined ? !s.has(c) : !!force;
          if (on) s.add(c); else s.delete(c);
          write(s);
          return on;
        },
      };
    },
    set innerHTML(v) {
      this._html = String(v == null ? "" : v);
      if (this._html === "") this.children = [];
    },
    get innerHTML() { return this._html || ""; },
    appendChild(c) { c.parentNode = this; this.children.push(c); return c; },
    insertBefore(c, ref) {
      c.parentNode = this;
      const i = ref ? this.children.indexOf(ref) : -1;
      if (i < 0) this.children.push(c); else this.children.splice(i, 0, c);
      return c;
    },
    remove() {
      if (this.parentNode) {
        const i = this.parentNode.children.indexOf(this);
        if (i >= 0) this.parentNode.children.splice(i, 1);
      }
      this.parentNode = null;
    },
    setAttribute(k, v) { this._attrs[k] = String(v); },
    getAttribute(k) { return this._attrs[k]; },
    addEventListener(ev, fn) { (this._listeners[ev] = this._listeners[ev] || []).push(fn); },
    removeEventListener() {},
    dispatch(ev) {
      const arg = { stopPropagation() {}, preventDefault() {} };
      (this._listeners[ev] || []).forEach((f) => f(arg));
    },
    querySelector(sel) { return this._find(sel, true); },
    querySelectorAll(sel) { return this._find(sel, false); },
    focus() {},
    _matches(sel) {
      const s = String(sel || "").trim();
      if (s.startsWith(".")) return this._className.split(/\s+/).includes(s.slice(1));
      return this.tagName === s.toUpperCase();
    },
    _find(sel, first) {
      const out = [];
      const walk = (n) => {
        for (const c of n.children) {
          if (c._matches(sel)) { out.push(c); if (first) return true; }
          if (walk(c) && first) return true;
        }
        return false;
      };
      walk(this);
      return first ? (out[0] || null) : out;
    },
    get firstChild() { return this.children[0] || null; },
    scrollTop: 0,
    scrollHeight: 0,
    insertAdjacentHTML() {},
  };
  return el;
}

function makeDocument() {
  const registry = new Map();
  return {
    registry,
    byId(id) {
      if (!registry.has(id)) {
        const el = makeElement("div");
        el.id = id;
        registry.set(id, el);
      }
      return registry.get(id);
    },
    getElementById(id) { return this.byId(id); },
    createElement(t) { return makeElement(t); },
    querySelector() { return null; },
    querySelectorAll() { return []; },
    addEventListener() {},
    body: makeElement("body"),
  };
}

// ── fake backend ────────────────────────────────────────────────────────────
const SETTINGS = {
  chat_model: { provider: "gemini", model: "gemini-2.5-flash" },
  tts_model: { provider: "fish", model: "s2.1-pro-free" },
  vision_model: { provider: "gemini", model: "gemini-2.5-flash" },
  browser_tool_model: { provider: "fireworks", model: "accounts/fireworks/models/qwen3p7-plus" },
  listening_model: { provider: "inworld", model: "inworld/inworld-stt-1" },
  planner_model: { provider: "fireworks", model: "accounts/fireworks/models/qwen3p7-plus" },
  providers: [
    { id: "gemini", name: "Google Gemini", kind: "env", has_key: true, source: "env" },
    { id: "fireworks", name: "Fireworks AI", kind: "env", has_key: true, source: "env" },
    { id: "groq", name: "Groq", kind: "env", has_key: true, source: "env" },
    { id: "fish", name: "Fish Audio", kind: "env", has_key: true, source: "env" },
    // Free, key-less fallback: has_key is FALSE and that is its normal state.
    { id: "gtts", name: "Google TTS (free)", kind: "env", has_key: false, source: "env" },
    { id: "openrouter", name: "OpenRouter", kind: "env", has_key: true, source: "env" },
    { id: "whisper", name: "Local Whisper", kind: "env", has_key: false, source: "env" },
    { id: "inworld", name: "Inworld STT", kind: "env", has_key: true, source: "env" },
    { id: "sarvam", name: "Sarvam STT", kind: "env", has_key: true, source: "env" },
  ],
  role_allowed: {
    chat: ["gemini", "fireworks"],
    tts: ["fish", "gtts"],
    vision: ["gemini", "fireworks", "groq", "openrouter"],
    browser_tool: ["gemini", "fireworks", "groq", "openrouter"],
    listening: ["inworld", "sarvam", "whisper"],
    planner: ["fireworks"],
  },
  last_fallback: null,
};

const MODELS = {
  gemini: [{ id: "gemini-2.5-flash", display: "Gemini 2.5 Flash" }],
  fireworks: [{ id: "accounts/fireworks/models/qwen3p7-plus", display: "qwen3p7-plus" }],
  groq: [{ id: "qwen/qwen3.6-27b", display: "qwen/qwen3.6-27b" }],
  fish: [{ id: "s2.1-pro-free", display: "s2.1-pro-free" }],
  gtts: [
    { id: "en", display: "en" },
    { id: "hi", display: "hi" },
    { id: "es", display: "es" },
  ],
  openrouter: [{ id: "google/gemma-4-31b-it:free", display: "gemma-4-31b-it:free" }],
  whisper: [{ id: "whisper-local", display: "whisper-local" }],
  inworld: [{ id: "inworld/inworld-stt-1", display: "inworld/inworld-stt-1" }],
  sarvam: [{ id: "saaras:v4", display: "saaras:v4" }],
};

// Every role the backend registry can resolve, with the provider/model the
// test will drive through the UI.
const ROLE_CASES = [
  { role: "chat", listId: "model-provider-list", provider: "gemini", model: "gemini-2.5-flash", allowed: ["gemini", "fireworks"] },
  { role: "tts", listId: "tts-provider-list", provider: "fish", model: "s2.1-pro-free", allowed: ["fish", "gtts"] },
  { role: "listening", listId: "listening-provider-list", provider: "inworld", model: "inworld/inworld-stt-1", allowed: ["whisper", "inworld", "sarvam"] },
  { role: "vision", listId: "vision-provider-list", provider: "gemini", model: "gemini-2.5-flash", allowed: ["gemini", "fireworks", "groq", "openrouter"] },
  { role: "browser_tool", listId: "browser-provider-list", provider: "fireworks", model: "accounts/fireworks/models/qwen3p7-plus", allowed: ["gemini", "fireworks", "groq", "openrouter"] },
  { role: "planner", listId: "planner-provider-list", provider: "fireworks", model: "accounts/fireworks/models/qwen3p7-plus", allowed: ["fireworks"] },
];

function loadSidebar() {
  const document = makeDocument();
  const calls = [];
  async function fetchImpl(url, opts) {
    const options = opts || {};
    const method = String(options.method || "GET").toUpperCase();
    const reqPath = String(url).replace(/^https?:\/\/[^/]+/, "");
    calls.push({
      url: String(url),
      path: reqPath,
      method,
      headers: options.headers || {},
      body: options.body || null,
    });
    let payload = {};
    if (reqPath === "/settings") {
      payload = SETTINGS;
    } else {
      const m = reqPath.match(/^\/providers\/([^/]+)\/models$/);
      if (m) {
        payload = { models: MODELS[decodeURIComponent(m[1])] || [] };
      } else if (reqPath === "/settings/model") {
        const sent = JSON.parse(options.body || "{}");
        payload = { ok: true, role: sent.role };
        payload[sent.role + "_model"] = { provider: sent.provider, model: sent.model };
      }
    }
    const text = JSON.stringify(payload);
    return { ok: true, status: 200, text: async () => text, json: async () => payload };
  }

  const sandbox = {
    document,
    fetch: fetchImpl,
    requestAnimationFrame: () => 0,
    // renderer.js kicks off UI polling loops when it loads. Inert timers keep
    // those from holding the test process open; no sidebar logic depends on a
    // timer firing (every step below resolves through a promise).
    setTimeout: () => 0,
    clearTimeout: () => {},
    setInterval: () => 0,
    clearInterval: () => {},
    console,
    Math, Date, JSON, Object, Array, String, Number, Boolean, Promise,
    Set, Map, URL, RegExp, Error, isNaN, parseInt, parseFloat,
    encodeURIComponent, crypto,
  };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  sandbox.jarvisAPI = {
    send() {}, on() {}, openExternal() {},
    backend: async () => ({ status: 200, text: "{}" }),
    readReportMirror: async () => null,
    getConfig: async () => ({ backendPort: "9999" }),
    getLocalSecret: async () => LAUNCH_TOKEN,
  };
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(RENDERER_PATH, "utf8"), sandbox,
    { filename: "renderer.js" });
  return { sandbox, document, calls };
}

const settle = () => new Promise((r) => setTimeout(r, 25));

// ── tests ───────────────────────────────────────────────────────────────────

test("jarvisFetch delegates to the platform fetch instead of recursing", async () => {
  const { sandbox, calls } = loadSidebar();
  await settle(); // let the preload bridge resolve the port + token
  calls.length = 0;

  const response = await sandbox.jarvisFetch("http://127.0.0.1:9999/settings");

  assert.equal(response.ok, true, "the call must resolve, not blow the stack");
  assert.equal(calls.length, 1, "exactly one request must be issued");
  assert.equal(calls[0].path, "/settings");
  assert.equal(calls[0].headers["X-Jarvis-Token"], LAUNCH_TOKEN,
    "every backend call must carry the per-launch token");
});

test("every role renders its allowlisted providers", async () => {
  const { sandbox, document } = loadSidebar();
  await settle();
  await sandbox.loadProviders();

  for (const c of ROLE_CASES) {
    const list = document.byId(c.listId);
    const ids = list.children.map((s) => s.dataset.providerId).sort();
    assert.deepEqual(ids, [...c.allowed].sort(),
      `${c.role}: provider list must contain exactly the allowlisted providers`);
  }
});

test("every role's model options are displayed and selectable", async () => {
  const { sandbox, document, calls } = loadSidebar();
  await settle();
  await sandbox.loadProviders();

  for (const c of ROLE_CASES) {
    const list = document.byId(c.listId);
    const section = list.children.find((s) => s.dataset.providerId === c.provider);
    assert.ok(section, `${c.role}: ${c.provider} must be listed for the role`);

    const head = section.querySelector(".provider-head");
    assert.ok(head, `${c.role}: the provider row must be expandable`);

    // Expand, then force a fresh load so the per-role fetch path runs even
    // when the provider's (role-independent) model list is already cached.
    head.dispatch("click");
    const refresh = section.querySelector(".provider-refresh");
    if (refresh) refresh.dispatch("click");
    await settle();

    assert.ok(
      calls.some((x) => x.path === `/providers/${c.provider}/models`),
      `${c.role}: expanding ${c.provider} must request its model list`);

    const items = section.querySelectorAll(".model-item");
    assert.ok(items.length > 0, `${c.role}: model options must be displayed`);

    const target = items.find((b) => b.title === c.model);
    assert.ok(target, `${c.role}: ${c.model} must appear as a selectable option`);

    calls.length = 0;
    target.dispatch("click");
    await settle();

    const post = calls.find((x) => x.path === "/settings/model" && x.method === "POST");
    assert.ok(post, `${c.role}: clicking a model must POST /settings/model`);
    assert.deepEqual(JSON.parse(post.body), {
      role: c.role, provider: c.provider, model: c.model,
    }, `${c.role}: the selection must name the right role, provider and model`);
  }
});

test("the browser-tool role offers a clickable model list, not only free text", async () => {
  const { sandbox, document } = loadSidebar();
  await settle();
  await sandbox.loadProviders();

  const section = document.byId("browser-provider-list")
    .children.find((s) => s.dataset.providerId === "fireworks");
  section.querySelector(".provider-head").dispatch("click");
  await settle();

  const items = section.querySelectorAll(".model-item");
  assert.ok(items.length > 0,
    "the browser-tool section must list models to click, like every other role");
});

test("the free Google TTS fallback renders as keyless and is selectable", async () => {
  const { sandbox, document, calls } = loadSidebar();
  await settle();
  await sandbox.loadProviders();

  const list = document.byId("tts-provider-list");
  const section = list.children.find((s) => s.dataset.providerId === "gtts");
  assert.ok(section, "Google TTS must be offered for the tts role");

  const head = section.querySelector(".provider-head");
  assert.ok(head, "the Google TTS row must be expandable");
  assert.equal(head.disabled, false,
    "a key-less engine must never be disabled for lack of a key");
  assert.ok(head.innerHTML.includes("no key needed"),
    "the row must say no key is needed instead of implying a misconfiguration");
  assert.ok(head.innerHTML.includes("has-key") && !head.innerHTML.includes("no-key"),
    "the status dot must render as live, not as the grey no-key state");

  head.dispatch("click");
  await settle();
  assert.ok(calls.some((x) => x.path === "/providers/gtts/models"),
    "expanding Google TTS must request its voice list");

  const items = section.querySelectorAll(".model-item");
  assert.ok(items.length > 0, "the Google TTS voices must be displayed");

  const target = items.find((b) => b.title === "hi");
  assert.ok(target, "the Hindi voice must be selectable");

  calls.length = 0;
  target.dispatch("click");
  await settle();

  const post = calls.find((x) => x.path === "/settings/model" && x.method === "POST");
  assert.ok(post, "clicking a Google TTS voice must POST /settings/model");
  assert.deepEqual(JSON.parse(post.body), { role: "tts", provider: "gtts", model: "hi" },
    "the selection must name the tts role with the gtts provider");
});
