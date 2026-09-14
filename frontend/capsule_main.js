const { BrowserWindow, ipcMain, screen } = require("electron");
const path = require("path");

let capsuleWindow = null;
let handlersRegistered = false;

// G11 / F51 — the capsule window is sandboxed exactly like every other
// renderer: context-isolated, no Node in the page, minimal preload bridge.
const PRELOAD_PATH = path.join(__dirname, "preload.js");

// G11 / F51 — the capsule is Jarvis's own window, so it gets the narrow
// `jarvis:capsule-command` channel (action enum -> fixed backend endpoints).
// Main identifies it POSITIVELY by the webContents id registered here at
// window creation — never by "it is not the trusted window". The preload
// marker is what makes the bridge expose that method to this window only.
const CAPSULE_PRELOAD_ARGS = ["--jarvis-capsule"];
const capsuleWebContentsIds = new Set();

function hardenWindow(win) {
  try {
    win.webContents.on("will-navigate", (event, url) => {
      const allowed = url.startsWith("file://") && url.includes(path.join("frontend") + path.sep);
      if (!allowed) event.preventDefault();
    });
    win.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  } catch (_e) { /* hardening must never break window creation */ }
  return win;
}

function createCapsuleWindow() {
  if (capsuleWindow && !capsuleWindow.isDestroyed()) {
    capsuleWindow.show();
    return capsuleWindow;
  }

  const wa = screen.getPrimaryDisplay().workArea;
  const w = 360;
  const h = 150;
  const x = Math.round(wa.width - w - 12 + wa.x);
  const y = Math.round(wa.height - h - 12 + wa.y);

  capsuleWindow = new BrowserWindow({
    width: w,
    height: h,
    x,
    y,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    skipTaskbar: true,
    resizable: false,
    hasShadow: false,
    focusable: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      preload: PRELOAD_PATH,
      additionalArguments: CAPSULE_PRELOAD_ARGS,
    },
  });
  hardenWindow(capsuleWindow);

  // Register this window's webContents so main can positively identify the
  // capsule sender of jarvis:capsule-command (see main.js).
  const contentsId = capsuleWindow.webContents && capsuleWindow.webContents.id;
  if (contentsId != null) {
    capsuleWebContentsIds.add(contentsId);
  }

  // screen-saver level keeps it above normal windows but not above system UI
  try {
    capsuleWindow.setAlwaysOnTop(true, "screen-saver");
  } catch (_e) {}

  capsuleWindow.loadFile(path.join(__dirname, "capsule.html"));

  capsuleWindow.on("closed", () => {
    if (contentsId != null) {
      capsuleWebContentsIds.delete(contentsId);
    }
    capsuleWindow = null;
  });

  // show without stealing focus from main HUD on boot
  capsuleWindow.once("ready-to-show", () => {
    try {
      capsuleWindow.showInactive();
    } catch (_e) {
      capsuleWindow.show();
    }
  });

  // if already ready, ensure visible without focus steal
  if (!capsuleWindow.isVisible()) {
    // fallback: will be shown on ready-to-show
  }

  return capsuleWindow;
}

function initCapsule() {
  if (handlersRegistered) {
    return;
  }
  handlersRegistered = true;

  ipcMain.on("create-capsule", () => {
    const win = createCapsuleWindow();
    // focusing existing capsule when requested via button should bring it forward
    if (win && !win.isDestroyed()) {
      if (!win.isVisible()) {
        try {
          win.showInactive();
        } catch (_e) {
          win.show();
        }
      } else {
        win.focus();
      }
    }
  });

  ipcMain.on("close-capsule", () => {
    if (capsuleWindow && !capsuleWindow.isDestroyed()) {
      capsuleWindow.close();
    }
    capsuleWindow = null;
  });

  ipcMain.on("hide-capsule", () => {
    if (capsuleWindow && !capsuleWindow.isDestroyed() && capsuleWindow.isVisible()) {
      capsuleWindow.hide();
    }
  });

  ipcMain.on("show-capsule", () => {
    if (capsuleWindow && !capsuleWindow.isDestroyed()) {
      if (!capsuleWindow.isVisible()) {
        try {
          capsuleWindow.showInactive();
        } catch (_e) {
          capsuleWindow.show();
        }
      }
    } else {
      createCapsuleWindow();
    }
  });
}

module.exports = {
  initCapsule,
  createCapsuleWindow,
  // G11 / F51 — positive sender identification for jarvis:capsule-command.
  isCapsuleSender: (webContentsId) => capsuleWebContentsIds.has(webContentsId),
  _getCapsuleWindow: () => capsuleWindow,
};
