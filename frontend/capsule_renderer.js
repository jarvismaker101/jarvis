// G11 / F51 — sandboxed renderer: the ONLY privileged surface is the
// preload bridge. The capsule is not the trusted chat window, so it must NOT
// use the blanket backend proxy for mutations; its three legitimate controls
// ride the narrow `capsuleCommand` action channel (main maps the action to a
// fixed endpoint and attaches the token itself).
const IPC = window.jarvisAPI || {
  send() {}, on() {}, openExternal() {},
  backend: async () => ({ status: 0, error: "no preload bridge" }),
  capsuleCommand: async () => ({ status: 0, error: "no preload bridge" }),
  readReportMirror: async () => null,
  getConfig: async () => ({ backendPort: "9999" }),
  getLocalSecret: async () => "",
};

let BACKEND_PORT = "9999";
let BACKEND_URL = "http://127.0.0.1:" + BACKEND_PORT;
IPC.getConfig().then((cfg) => {
  if (cfg && cfg.backendPort) {
    BACKEND_PORT = String(cfg.backendPort);
    BACKEND_URL = "http://127.0.0.1:" + BACKEND_PORT;
  }
}).catch(() => {});

let isRequestInFlight = false;
let hidePollTimer = null;
let capsuleHiddenByCapture = false;

function appendMessage(text, type) {
  const log = document.getElementById("capsule-log");
  if (!log) return;
  const div = document.createElement("div");
  div.className = "capsule-msg " + (type || "bot");
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

async function sendMessage() {
  if (isRequestInFlight) return;
  const input = document.getElementById("capsule-input");
  const sendBtn = document.getElementById("capsule-send");
  if (!input) return;
  const raw = input.value.trim();
  if (!raw) return;
  input.value = "";
  input.focus();
  isRequestInFlight = true;
  if (input) input.disabled = true;
  if (sendBtn) sendBtn.disabled = true;
  // Barge-in: cut any in-progress TTS the moment the user submits. The capsule
  // names the action only; main performs POST /speak/stop with the token.
  // Fire-and-forget: a refusal must not block the ask, and it never pretends
  // the audio stopped (the playback just continues).
  try {
    Promise.resolve(IPC.capsuleCommand("speak-stop")).then((result) => {
      if (result && result.error) console.warn("capsule speak-stop refused:", result.error);
    }).catch(() => {});
  } catch (_e) {}
  appendMessage(raw, "user");
  try {
    const result = await IPC.capsuleCommand("ask", raw);
    if (!result || !result.status || result.status >= 400) {
      // Honest failure: surface the policy/backend reason, never a fake reply.
      appendMessage("Jarvis did not accept this message (" +
        ((result && result.error) || ("status " + ((result && result.status) || 0))) + ")", "error");
      return;
    }
    const data = JSON.parse(result.text || "{}");
    const reply = (data.reply || "").trim();
    if (reply) {
      appendMessage(reply, "bot");
    } else {
      appendMessage("I didn't get a response. Please try again.", "error");
    }
  } catch (err) {
    const msg = err && err.message ? err.message : String(err);
    appendMessage("Could not reach backend. Is Jarvis running? (" + msg + ")", "error");
  } finally {
    isRequestInFlight = false;
    if (input) input.disabled = false;
    if (sendBtn) sendBtn.disabled = false;
    if (input) input.focus();
  }
}

function setupStaticHandlers() {
  const input = document.getElementById("capsule-input");
  const sendBtn = document.getElementById("capsule-send");
  const closeBtn = document.getElementById("capsule-close");
  if (input) {
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !isRequestInFlight) {
        e.preventDefault();
        sendMessage();
      }
    });
  }
  if (sendBtn) {
    sendBtn.addEventListener("click", () => sendMessage());
  }
  if (closeBtn) {
    closeBtn.addEventListener("click", () => {
      IPC.send("close-capsule");
    });
  }
  const stopBtn = document.getElementById("capsule-task-stop");
  if (stopBtn) {
    stopBtn.addEventListener("click", () => stopRunningTask());
  }
}

// TASK-STOP: the capsule asks main for the `task-stop` action while a
// browser/opencode task runs. The button only renders while /ui-state reports
// task_running true.
let taskStopInFlight = false;

async function stopRunningTask() {
  if (taskStopInFlight) return;
  taskStopInFlight = true;
  const btn = document.getElementById("capsule-task-stop");
  if (btn) btn.disabled = true;
  try {
    const result = await IPC.capsuleCommand("task-stop");
    if (!result || !result.status || result.status >= 400) {
      // Honest failure: do not pretend the task stopped.
      appendMessage("Could not stop the task (" +
        ((result && result.error) || (result && result.status) || 0) + ")", "error");
    }
  } catch (err) {
    appendMessage("Could not stop the task (" +
      ((err && err.message) || String(err)) + ")", "error");
  } finally {
    taskStopInFlight = false;
    if (btn) {
      btn.disabled = false;
      btn.style.display = "none";
    }
  }
}

function updateTaskStopButton(running) {
  const btn = document.getElementById("capsule-task-stop");
  if (!btn) return;
  btn.style.display = running ? "inline-block" : "none";
}

// CAPTURE-HIDE: poll /ui-state every 500ms; hide capsule during active screen-control
// capture/execution to protect click coordinates. Signal chosen: screen_action_pending
// (screen_state.has_pending_plan) - the only reliable capture/execution indicator.
// screen_controls_enabled alone would hide permanently while enabled; thinking/speaking
// are unrelated to screen capture. When pending becomes true, hide; when it clears,
// show again. Tracking capsuleHiddenByCapture prevents redundant ipc spam.
// If screen_action_pending never appears (no reliable field), fallback would be
// screen_controls_enabled && thinking, but current backend does expose pending reliably.
async function pollCaptureState() {
  try {
    const result = await IPC.backend("/ui-state");
    if (!result || !result.status || result.status >= 400) return;
    const data = JSON.parse(result.text || "{}");
    const state = (data && data.state) || data || {};
    const pending = !!state.screen_action_pending;
    updateTaskStopButton(!!(data && data.task_running));
    if (pending && !capsuleHiddenByCapture) {
      capsuleHiddenByCapture = true;
      IPC.send("hide-capsule");
    } else if (!pending && capsuleHiddenByCapture) {
      capsuleHiddenByCapture = false;
      IPC.send("show-capsule");
    }
  } catch (_e) {
    // ignore poll errors
  }
}

function startHidePolling() {
  if (hidePollTimer) return;
  hidePollTimer = setInterval(pollCaptureState, 500);
}

setupStaticHandlers();
startHidePolling();
