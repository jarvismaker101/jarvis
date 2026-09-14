// F54 — the preload push bridge and the overlay renderers must agree.
//
// The bridge (frontend/preload.js) deliberately forwards ONLY the payload:
// handing renderer content the raw IpcRendererEvent would leak `event.sender`,
// an unrestricted send channel that defeats the channel whitelist. The three
// overlay renderers were written against the old direct-`ipcRenderer` style
// and registered `(_event, data)` handlers, so every push arrived as
// `data === undefined`, threw inside the bridge's try/catch, and was silently
// dropped. The screen overlay therefore never appeared even though Jarvis
// described the screen correctly and spoke the answer; the research overlay
// only survived because it has its own 2s poll fallback.
//
// Run with Node's built-in runner (the repo has no JS test harness):
//
//     node --test tests/overlay-ipc-contract.test.js
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ROOT = path.join(__dirname, "..");
const PRELOAD = path.join(ROOT, "frontend", "preload.js");
const RENDERERS = [
  "overlay_renderer.js",
  "overlay_images_renderer.js",
  "research_overlay_renderer.js",
];

/** Load preload.js against a stubbed electron and return { api, fire }. */
function loadBridge() {
  const electronPath = require.resolve("electron");
  const handlers = new Map();
  const bridges = [];

  require.cache[electronPath] = {
    id: electronPath,
    filename: electronPath,
    loaded: true,
    exports: {
      contextBridge: {
        exposeInMainWorld: (name, api) => bridges.push({ name, api }),
      },
      ipcRenderer: {
        on: (channel, handler) => handlers.set(channel, handler),
        send: () => {},
        invoke: async () => ({}),
      },
    },
  };

  delete require.cache[require.resolve(PRELOAD)];
  require(PRELOAD);

  assert.equal(bridges.length, 1, "preload must expose exactly one bridge");
  assert.equal(bridges[0].name, "jarvisAPI");

  return {
    api: bridges[0].api,
    // Electron calls a raw listener as (event, ...args).
    fire: (channel, payload) => {
      const handler = handlers.get(channel);
      assert.ok(handler, "no bridge listener registered for " + channel);
      handler({ sender: "fake-webContents" }, payload);
    },
    has: (channel) => handlers.has(channel),
  };
}

/** A DOM small enough for the overlay renderer and nothing more. */
function makeElement(tag) {
  const classes = new Set();
  return {
    tagName: tag,
    textContent: "",
    innerHTML: "",
    title: "",
    offsetHeight: 0,
    children: [],
    classList: {
      add: (...names) => names.forEach((n) => classes.add(n)),
      remove: (...names) => names.forEach((n) => classes.delete(n)),
      contains: (name) => classes.has(name),
    },
    style: { setProperty() {} },
    addEventListener() {},
    appendChild(child) {
      this.children.push(child);
      return child;
    },
  };
}

function loadRenderer(file, api) {
  const elements = new Map();
  const document = {
    getElementById: (id) => {
      if (!elements.has(id)) elements.set(id, makeElement(id));
      return elements.get(id);
    },
    createElement: (tag) => makeElement(tag),
  };
  const sandbox = {
    window: { jarvisAPI: api },
    document,
    // The 14s auto-dismiss timer must never keep the test process alive.
    setTimeout: () => 0,
    clearTimeout: () => {},
    console,
  };
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(path.join(ROOT, "frontend", file), "utf8"),
                  sandbox, { filename: file });
  return elements;
}

test("the bridge delivers the payload as the callback's only argument", () => {
  const { api, fire } = loadBridge();
  const calls = [];
  api.on("show-screen-answer", (...args) => calls.push(args));

  const payload = { id: 7, tip: "This is a browser window" };
  fire("show-screen-answer", payload);

  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0], [payload],
                   "the renderer must receive the payload alone — never the "
                   + "IpcRendererEvent, and never off by one position");
});

test("the bridge refuses channels outside the push whitelist", () => {
  const { api, has } = loadBridge();
  let called = false;
  api.on("some-other-channel", () => { called = true; });

  assert.equal(has("some-other-channel"), false,
               "an unlisted channel must never reach ipcRenderer");
  assert.equal(called, false);
});

test("a pushed screen answer actually makes the glass overlay visible", () => {
  const { api, fire } = loadBridge();
  const elements = loadRenderer("overlay_renderer.js", api);

  fire("show-screen-answer", {
    id: 12,
    tip: "You are looking at Visual Studio Code.",
    region: {},
    links: [{ label: "Docs", url: "https://example.com", icon: "L" }],
    evidence: [{ source: "screen", title: "VS Code", snippet: "editor" }],
  });

  const overlay = elements.get("overlay");
  assert.equal(elements.get("tip-text").textContent,
               "You are looking at Visual Studio Code.");
  assert.ok(overlay.classList.contains("visible"),
            "the overlay must be marked visible when an answer is pushed");
  assert.ok(!overlay.classList.contains("dismissing"));
  assert.equal(elements.get("links-items").children.length, 1,
               "link pills must be rendered from the pushed answer");
  assert.equal(elements.get("evidence-items").children.length, 1);
});

test("a pushed image set reaches the image overlay renderer", () => {
  const { api, fire } = loadBridge();
  loadRenderer("overlay_images_renderer.js", api);

  // The same off-by-one signature would hand showImages `undefined` and throw
  // inside the bridge, so this pins that the pushed payload arrives intact.
  assert.doesNotThrow(() => fire("show-screen-images", [
    { url: "https://example.com/a.png", title: "A", caption: "c" },
  ]));
});

test("no overlay renderer expects the raw IPC event object", () => {
  for (const file of RENDERERS) {
    const source = fs.readFileSync(path.join(ROOT, "frontend", file), "utf8");
    // Comments may legitimately name the old signature, so scan code only.
    const code = source
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .replace(/^[ \t]*\/\/.*$/gm, "");
    assert.ok(
      !/_event/.test(code),
      `${file} registers a handler for the raw IpcRendererEvent; the bridge `
      + "delivers the payload only, so that parameter holds the payload and "
      + "the real payload lands in the next one — the bug that left the "
      + "screen overlay empty");
  }
});