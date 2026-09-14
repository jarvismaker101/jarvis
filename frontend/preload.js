/**
 * Jarvis preload bridge (G11 / F51 — sandboxed renderers).
 *
 * Every renderer window loads with contextIsolation + sandbox enabled, so
 * none of them can touch Node/Electron APIs directly. This preload exposes
 * the MINIMAL, validated surface the UI needs:
 *
 *   jarvisAPI.send(channel, ...args)     whitelisted control IPC only
 *   jarvisAPI.on(channel, cb)            whitelisted push channels only
 *   jarvisAPI.openExternal(url)          http/https validated in main
 *   jarvisAPI.backend(path, opts)        proxied backend call; the token is
 *                                        attached by the MAIN process, never
 *                                        handed to renderer content
 *   jarvisAPI.readReportMirror()         the research mirror file, read by
 *                                        main (no fs in the renderer)
 *   jarvisAPI.getConfig()                backend port + build label
 *   jarvisAPI.getLocalSecret()           ONLY in the trusted chat window
 *                                        (additionalArguments marker): the
 *                                        token it needs for its own SSE
 *                                        /ask/stream fetch. Overlay windows
 *                                        — which render web-derived content
 *                                        — can never obtain it.
 *   jarvisAPI.capsuleCommand(action,text) ONLY in the capsule window
 *                                        (--jarvis-capsule marker): the
 *                                        capsule names an ACTION from a fixed
 *                                        enum, never a URL; main maps it to
 *                                        POST /ask, /task/stop or /speak/stop.
 *                                        Every other window is refused here
 *                                        without any IPC reaching main.
 */
const { contextBridge, ipcRenderer } = require("electron");

// Sandboxed preloads get a limited `process`; argv carries the trust marker
// the main process passed via webPreferences.additionalArguments.
const TRUSTED = (typeof process !== "undefined" &&
  Array.isArray(process.argv) &&
  process.argv.includes("--jarvis-trusted"));

// G11 / F51 — the capsule carries its own marker; only frontend/capsule_main.js
// (the one place the capsule BrowserWindow is created) sets it, and main
// re-verifies the sender against its registered capsule webContents.
const CAPSULE = (typeof process !== "undefined" &&
  Array.isArray(process.argv) &&
  process.argv.includes("--jarvis-capsule"));

const SEND_CHANNELS = new Set([
  "stop-jarvis",
  "create-capsule",
  "close-capsule",
  "hide-capsule",
  "show-capsule",
  "dismiss-overlay",
  "dismiss-image-overlay",
  "minimize-research-overlay",
  "dismiss-research-overlay",
  "request-show-research-window",
]);

const ON_CHANNELS = new Set([
  "show-screen-answer",
  "show-screen-images",
  "show-research-result",
]);

contextBridge.exposeInMainWorld("jarvisAPI", {
  send(channel, ...args) {
    if (!SEND_CHANNELS.has(channel)) return;
    try { ipcRenderer.send(channel, ...args); } catch (_e) {}
  },
  on(channel, callback) {
    if (!ON_CHANNELS.has(channel) || typeof callback !== "function") return;
    ipcRenderer.on(channel, (_event, data) => {
      try { callback(data); } catch (_e) {}
    });
  },
  openExternal(url) {
    try { ipcRenderer.send("jarvis:open-external", String(url || "")); } catch (_e) {}
  },
  backend(path, opts) {
    return ipcRenderer.invoke("jarvis:backend-request", {
      path: String(path || "/"),
      method: (opts && opts.method) || "GET",
      body: (opts && opts.body) || null,
    });
  },
  readReportMirror() {
    return ipcRenderer.invoke("jarvis:read-report-mirror");
  },
  getConfig() {
    return ipcRenderer.invoke("jarvis:get-config");
  },
  getLocalSecret() {
    if (!TRUSTED) return Promise.resolve("");
    return ipcRenderer.invoke("jarvis:get-local-secret");
  },
  // G11 / F51 — the capsule's narrow command channel. The renderer sends an
  // action name, never a path; main maps it to one fixed endpoint. Windows
  // without the capsule marker never reach main at all.
  capsuleCommand(action, text) {
    if (!CAPSULE) {
      return Promise.resolve({ status: 0, error: "not permitted" });
    }
    return ipcRenderer.invoke("jarvis:capsule-command", {
      action: String(action || ""),
      text: String(text == null ? "" : text),
    });
  },
});
