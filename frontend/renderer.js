// G11 / F51 — sandboxed renderer: only the preload bridge is privileged.
const IPC = window.jarvisAPI || {
  send() {}, on() {}, openExternal() {},
  backend: async () => ({ status: 0 }),
  readReportMirror: async () => null,
  getConfig: async () => ({ backendPort: "9999" }),
  getLocalSecret: async () => "",
};
let BACKEND_PORT = "9999";
let BACKEND_URL = "http://127.0.0.1:" + BACKEND_PORT;
let _jarvisSecret = "";
let _bridgeReady = null;
async function initBridge() {
  try {
    const cfg = await IPC.getConfig();
    if (cfg && cfg.backendPort) {
      BACKEND_PORT = String(cfg.backendPort);
      BACKEND_URL = "http://127.0.0.1:" + BACKEND_PORT;
    }
    _jarvisSecret = await IPC.getLocalSecret();
  } catch (_e) {}
}
_bridgeReady = initBridge();
// Every backend call rides the per-launch token (attached in the trusted
// chat renderer ONLY); non-trusted windows get an empty secret and the
// proxy path in the preload instead.
//
// The port and the token are delivered asynchronously by the main process, so
// a call issued before the bridge resolves would go to the wrong port or out
// unauthenticated (the backend answers 401). Every call therefore waits for
// the bridge exactly once.
//
// Native fetch is used when available because /ask/stream needs a real
// streaming body — the preload proxy buffers a response and cannot stream.
// A renderer without fetch falls back to that policy-gated proxy, which
// attaches the token in main.
async function jarvisFetch(url, opts) {
  if (_bridgeReady) {
    try { await _bridgeReady; } catch (_e) {}
  }
  const options = Object.assign({}, opts || {});
  const headers = Object.assign({}, options.headers || {});
  if (_jarvisSecret) headers["X-Jarvis-Token"] = _jarvisSecret;
  options.headers = headers;
  if (typeof fetch === "function") return fetch(url, options);
  const path = String(url).replace(/^[a-z][a-z0-9+.-]*:\/\/[^/]+/i, "") || "/";
  const proxied = await IPC.backend(path, {
    method: options.method || "GET",
    body: options.body,
  });
  const status = (proxied && proxied.status) || 0;
  const text = (proxied && proxied.text) || "";
  return {
    ok: status >= 200 && status < 300,
    status,
    text: () => Promise.resolve(text),
    json: () => Promise.resolve(JSON.parse(text || "{}")),
  };
}

let currentMode = "chat";
let lastVoiceLogId = 0;
let voiceStateFailures = 0;
let voiceStatePollTimer = null;
let isRequestInFlight = false;
let voiceInputEnabled = true;

// F23 — how many times a client may re-attach to the same request before it
// gives up. Every re-attach resumes from the last event it rendered; the
// server never re-executes the message.
const MAX_RECONNECTS = 3;

// F23 — one client-generated id per request, shared by /ask and /ask/stream.
// The server uses it to execute the message once and to keep a numbered
// event buffer a reconnecting client can resume from.
function newRequestId() {
  const rand = (typeof crypto !== "undefined" && crypto.randomUUID)
    ? crypto.randomUUID().replace(/-/g, "").slice(0, 12)
    : Math.random().toString(36).slice(2, 14);
  return "req-" + Date.now().toString(36) + "-" + rand;
}

const STATUS_LABELS = {
  offline: "OFFLINE",
  reconnecting: "RECONNECTING",
  listening: "LISTENING",
  hearing: "HEARING",
  thinking: "THINKING",
  speaking: "SPEAKING",
};

const STATUS_SUBTEXT = {
  offline: "Backend disconnected",
  reconnecting: "Reconnecting to backend",
  listening: "Ready for your voice",
  hearing: "I can hear you",
  thinking: "Working on a reply, still listening",
  speaking: "Responding now",
};

function buildStatusSubtext(state, status) {
  const base = STATUS_SUBTEXT[status] || STATUS_SUBTEXT.listening;
  if (!state) return base;
  if (state.screen_action_pending) return `${base} \u2022 Awaiting screen action confirmation`;
  if (state.screen_controls_enabled) return `${base} \u2022 Screen controls on`;
  return base;
}

function updateModeIndicator() {
  const ind = document.getElementById("mode-indicator");
  if (!ind) return;
  const idx = currentMode === "chat" ? 0 : currentMode === "command" ? 1 : 0;
  // memory button doesn't move indicator — keep on chat/command
  const isMem = document.activeElement && document.activeElement.id === "tab-memory";
  if (isMem) return;
  const w = `calc((100% - 12px - 8px)/3)`;
  ind.style.width = w;
  ind.style.transform = `translateX(calc(${idx} * (100% + 4px)))`;
  // color shift
  if (currentMode === "command") {
    ind.style.background = "linear-gradient(135deg, rgba(255,140,46,0.20), rgba(255,90,120,0.14))";
    ind.style.borderColor = "rgba(255,140,46,0.34)";
    ind.style.boxShadow = "0 0 18px rgba(255,140,46,0.2), inset 0 1px 0 rgba(255,255,255,0.08)";
  } else {
    ind.style.background = "linear-gradient(135deg, rgba(0,229,255,0.18), rgba(124,92,255,0.18))";
    ind.style.borderColor = "rgba(0,229,255,0.22)";
    ind.style.boxShadow = "0 0 18px rgba(0,229,255,0.16), inset 0 1px 0 rgba(255,255,255,0.08)";
  }
}

function setMode(mode) {
  if (mode === "memory") return;
  currentMode = mode;
  document.getElementById("tab-chat").classList.toggle("active", mode === "chat");
  document.getElementById("tab-command").classList.toggle("active", mode === "command");
  const prefix = document.getElementById("mode-prefix");
  const input = document.getElementById("user-input");
  if (mode === "command") {
    prefix.textContent = "CMD";
    prefix.classList.add("cmd");
    input.placeholder = "e.g. open youtube and play lofi music on chrome...";
  } else {
    prefix.textContent = "ASK";
    prefix.classList.remove("cmd");
    input.placeholder = "Type your message...";
  }
  document.body.dataset.mode = mode;
  document.body.classList.remove('mode-chat','mode-command','mode-memory');
  if (mode) document.body.classList.add(`mode-${mode}`);
  updateModeIndicator();
  // micro tilt pulse on mode bar
  const bar = document.getElementById("mode-bar");
  if (bar) { bar.style.transform = "translateZ(12px) scale(1.005)"; setTimeout(()=> bar.style.transform="translateZ(10px)", 260); }
  input.focus();
}

function handleKey(event) {
  if (event.key === "Enter" && !isRequestInFlight) sendMessage();
}

function setInputLocked(locked) {
  const input = document.getElementById("user-input");
  const button = document.getElementById("send-button");
  isRequestInFlight = locked;
  input.disabled = locked;
  if (button) button.disabled = locked;
}

async function sendMessage() {
  if (isRequestInFlight) return;
  const input = document.getElementById("user-input");
  const chatBox = document.getElementById("chat-box");
  const raw = input.value.trim();
  if (!raw) return;
  const message = currentMode === "command" && !raw.toLowerCase().startsWith("command") ? `command ${raw}` : raw;
  input.value = ""; input.focus(); setInputLocked(true);
  // Barge-in: cut any in-progress TTS the moment the user submits
  // (fire-and-forget — the backend also stops before processing).
  try { jarvisFetch(BACKEND_URL + "/speak/stop", { method: "POST" }); } catch (_e) {}
  addBubble(chatBox, raw, "user", currentMode === "command");
  const typing = addTyping(chatBox);
  // send button pulse
  const sb = document.getElementById("send-button");
  if (sb) { sb.style.transform = "scale(0.96)"; setTimeout(()=> sb.style.transform="", 180); }
  // ── F23 ── one client-generated id per send, shared by /ask and
  // /ask/stream. The server executes the message once and appends numbered
  // events to that request's buffer, so a retry or a reconnect reattaches
  // instead of running the work again (no duplicate actions, duplicate
  // reasoning or competing spoken replies).
  const requestId = newRequestId();
  let lastEventId = -1;

  // Promote the typing indicator into a real bot bubble up front so streamed
  // deltas are visible while they arrive.
  let replyBody = typing.querySelector(".msg-text");
  if (!replyBody) {
    replyBody = document.createElement("span");
    replyBody.className = "msg-text";
    typing.appendChild(replyBody);
  }
  const textNode = document.createTextNode("");
  replyBody.appendChild(textNode);
  if (!typing.querySelector(".bot-label")) {
    const label = document.createElement("span");
    label.className = "bot-label";
    label.textContent = "JARVIS";
    typing.insertBefore(label, typing.firstChild);
  }
  typing.classList.remove("typing-only");
  typing.querySelectorAll(".dot").forEach((d) => d.remove());
  typing.classList.add("revealed");

  const scrollToEnd = () => {
    chatBox.scrollTop = chatBox.scrollHeight;
    updateScrollProgress();
  };
  const dropBubble = () => { if (typing && typing.parentNode) typing.remove(); };

  // F26 — the terminal reply is authoritative. Whatever `completed` carries
  // is what the user sees (and what the server stored), even when streamed
  // deltas already painted something else.
  function applyTerminal(text) {
    if (typeof text === "string" && text.trim().length) {
      textNode.nodeValue = text;
      scrollToEnd();
      return true;
    }
    return false;
  }

  // F26 — one event protocol: delta | replace | progress | completed |
  // interrupted | error. Returns true once a terminal frame was handled.
  function handleFrame(payload) {
    const type = payload && payload.type;
    if (!type) {
      // Backwards compatibility: an old backend emits {delta} / {done, reply}.
      if (typeof payload.delta === "string") {
        textNode.nodeValue = (textNode.nodeValue || "") + payload.delta;
        scrollToEnd();
        return false;
      }
      if (payload.done) return applyTerminal(payload.reply || "");
      return false;
    }
    // Numbered frames advance the resume cursor; heartbeats (seq null) do not.
    if (typeof payload.seq === "number") lastEventId = payload.seq;

    if (type === "delta") {
      if (typeof payload.text === "string" && payload.text) {
        textNode.nodeValue = (textNode.nodeValue || "") + payload.text;
        scrollToEnd();
      }
      return false;
    }
    if (type === "replace") {
      if (typeof payload.text === "string") {
        textNode.nodeValue = payload.text;
        scrollToEnd();
      }
      return false;
    }
    if (type === "progress") {
      // Also the heartbeat frame — proof the request is alive, so this is
      // where the stall timer is reset.
      return false;
    }
    if (type === "completed") {
      applyTerminal(payload.reply || "");
      return true;
    }
    if (type === "interrupted") {
      if (!applyTerminal(payload.reply || "")) {
        dropBubble();
        addBubble(chatBox, "Stopped, sir.", "bot");
      }
      return true;
    }
    if (type === "error") {
      dropBubble();
      addBubble(chatBox, "Error: " + (payload.error || "unknown error"), "bot");
      return true;
    }
    return false;
  }

  // F23 — attach (or re-attach) to this request's event stream, resuming
  // after the last frame we already rendered. Never re-executes the message.
  async function attach() {
    const controller = new AbortController();
    const hardTimeout = setTimeout(() => controller.abort(), 35000);
    // 9s without any frame (delta, progress or heartbeat) means stalled.
    let stallTimer = setTimeout(() => controller.abort(), 9000);
    const bump = () => {
      clearTimeout(stallTimer);
      stallTimer = setTimeout(() => controller.abort(), 9000);
    };
    let terminal = false;
    try {
      const response = await jarvisFetch(BACKEND_URL + "/ask/stream", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          request_id: requestId,
          last_event_id: lastEventId,
        }),
        signal: controller.signal,
      });
      if (!response.ok || !response.body) return { attached: false, terminal };
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        while (buffer.includes("\n\n")) {
          const idx = buffer.indexOf("\n\n");
          const frame = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 2);
          const line = frame.split("\n").find((l) => l.startsWith("data:"))?.slice(5).trim();
          if (!line) continue;
          let payload;
          try { payload = JSON.parse(line); } catch (_e) { continue; }
          bump();
          if (handleFrame(payload)) { terminal = true; break; }
        }
        if (terminal) break;
      }
      return { attached: true, terminal };
    } catch (readErr) {
      // aborted by the stall/hard timeout, or the connection dropped
      console.warn("stream attachment ended:", readErr);
      return { attached: false, terminal };
    } finally {
      clearTimeout(stallTimer);
      clearTimeout(hardTimeout);
    }
  }

  // F23 — replaces the old fallbackToAsk (which *executed the message
  // again*). Look the request up first: if the server knows it, resume from
  // its buffer. Only when it has no record at all do we POST /ask — and we
  // send the same request id so that execute-once still protects us.
  async function recover() {
    try {
      const statusRes = await jarvisFetch(BACKEND_URL + "/ask/status/" + encodeURIComponent(requestId));
      if (statusRes.ok) {
        const status = await statusRes.json();
        if (status.done && applyTerminal(status.reply || "")) return true;
        // Known but still running — reattach to the live event buffer.
        const retry = await attach();
        if (retry.terminal) return true;
        if (status.reply) return applyTerminal(status.reply);
        return false;
      }
    } catch (err) {
      console.warn("status lookup failed:", err);
    }
    try {
      const r = await jarvisFetch(BACKEND_URL + "/ask", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, request_id: requestId }),
      });
      if (!r.ok) throw new Error("ask failed " + r.status);
      const data = await r.json();
      const reply = (data.reply || "").trim();
      if (reply) return applyTerminal(reply);
      dropBubble();
      addBubble(chatBox, "I didn't get a response. Please try again.", "bot");
      return true;
    } catch (err) {
      console.error("recovery /ask failed:", err);
      return false;
    }
  }

  try {
    // First attachment. On a stall, a dropped connection or a client that
    // reconnects, every retry resumes from `lastEventId` — the server keeps
    // executing exactly once.
    let attempt = await attach();
    let reconnects = 0;
    while (!attempt.terminal && reconnects < MAX_RECONNECTS) {
      reconnects += 1;
      console.warn("stream interrupted — reconnecting to request", requestId,
                   "after event", lastEventId);
      attempt = await attach();
    }
    if (attempt.terminal) { return; }
    const recovered = await recover();
    if (recovered) { return; }
    dropBubble();
    addBubble(chatBox, "Could not reach backend. Is Jarvis running?", "bot");
  } catch (error) {
    console.error("stream fetch failed:", error);
    dropBubble();
    addBubble(
      chatBox,
      "Could not reach backend. Is Jarvis running? (" + (error.message || error) + ")",
      "bot"
    );
  } finally { setInputLocked(false); input.focus(); }
}

async function clearMemory() {
  const chatBox = document.getElementById("chat-box");
  try {
    const response = await jarvisFetch(BACKEND_URL + "/ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "clear memory" }),
    });
    const data = await response.json();
    chatBox.innerHTML = "";
    addBubble(chatBox, data.reply || "Memory cleared, sir.", "bot");
  } catch (_error) { addBubble(chatBox, "Could not clear memory.", "bot"); }
}

function stopJarvis() {
  IPC.send("stop-jarvis");
  renderVoiceState({ status: "offline" });
}

/* ── 3D hover tilt per message ── */
function attachTilt(el) {
  if (!el || el._tiltBound) return;
  el._tiltBound = true;
  el.addEventListener("mousemove", (e) => {
    const r = el.getBoundingClientRect();
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    const dx = (e.clientX - cx) / (r.width / 2);
    const dy = (e.clientY - cy) / (r.height / 2);
    const rx = (-dy * 6).toFixed(2), ry = (dx * 8).toFixed(2);
    el.style.transform = `translateY(-3px) translateZ(14px) rotateX(${rx}deg) rotateY(${ry}deg) scale(1.015)`;
    el.style.setProperty("--mx", `${((e.clientX - r.left)/r.width*100).toFixed(1)}%`);
    el.style.setProperty("--my", `${((e.clientY - r.top)/r.height*100).toFixed(1)}%`);
  });
  el.addEventListener("mouseleave", () => {
    el.style.transform = "";
  });
}

function addBubble(chatBox, text, type, isCommand = false) {
  const div = document.createElement("div");
  div.className = `message ${type}${isCommand ? " command" : ""}`;
  const label = document.createElement("span");
  label.className = type === "user" ? "user-label" : "bot-label";
  label.textContent = type === "user" ? (isCommand ? "CMD" : "YOU") : "JARVIS";
  div.appendChild(label);
  const body = document.createElement("span");
  body.className = "msg-text";
  body.textContent = text;
  div.appendChild(body);
  if (type === "bot") {
    const meta = document.createElement("span");
    meta.className = "msg-meta";
    meta.textContent = new Date().toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"}) + " \u2022 delivered";
    div.appendChild(meta);
  }
  // stagger for scroll cascade
  const count = chatBox.children.length;
  div.style.animationDelay = `${Math.min(count * 0.04, 0.22)}s`;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
  updateScrollProgress();
  // enable 3D hover after mount
  requestAnimationFrame(() => attachTilt(div));
  // entrance reveal observer
  div.classList.add("revealed");
  return div;
}

function addTyping(chatBox) {
  const div = document.createElement("div");
  div.className = "message bot typing-only";
  div.innerHTML = '<span class="bot-label">JARVIS</span><span class="dot"></span><span class="dot"></span><span class="dot"></span>';
  div.style.animationDelay = "0s";
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
  updateScrollProgress();
  requestAnimationFrame(()=> attachTilt(div));
  return div;
}

function renderVoiceState(state = {}) {
  const raw = String((state && (state.status || state.state)) || 'listening').toLowerCase();
  const valid = ['idle','listening','hearing','thinking','speaking','offline','reconnecting','error','command-armed'];
  const status = valid.includes(raw) ? raw : (STATUS_LABELS[raw] ? raw : 'listening');
  const hud = ['idle','listening','hearing','thinking','speaking','error','command-armed'].includes(status) ? status : (status==='offline'||status==='reconnecting' ? 'idle' : status);
  document.body.dataset.state = hud;
  document.body.classList.remove(...['idle','listening','hearing','thinking','speaking','error','command-armed'].map(x => `state-${x}`));
  document.body.classList.add(`state-${hud}`);
  window.__hudState = hud;
  window.__hudEnergy = ({idle:.45, listening:.72, hearing:.88, thinking:.58, speaking:.92, error:.98, 'command-armed':.66})[hud] || .45;
  const badge = document.getElementById("status-badge");
  const ring = document.getElementById("status-ring");
  const liveStatus = document.getElementById("live-status");
  const reactor = document.getElementById("reactor");
  if (badge) { badge.textContent = STATUS_LABELS[status] || STATUS_LABELS.listening || status.toUpperCase(); badge.className = `badge ${status}`; }
  if (ring) ring.className = `status-ring ${status}`;
  if (liveStatus) liveStatus.textContent = buildStatusSubtext(state, status);
  if (reactor) {
    reactor.style.filter = status === "speaking" ? "brightness(1.25) saturate(1.2)" : status === "hearing" ? "brightness(1.15)" : "none";
    reactor.style.transform = status === "thinking" ? "scale(1.06)" : "scale(1)";
    reactor.style.transition = "transform 0.5s cubic-bezier(.16,1,.3,1), filter 0.4s ease";
  }
  const header = document.getElementById("header");
  if (header) {
    header.style.boxShadow = status === "offline" ? "0 10px 36px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,59,92,0.14)" : "0 10px 36px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.07)";
  }
}

function pollDelayFor(state, failures) {
  if (failures > 0) return Math.min(3000, 500 * Math.pow(2, failures - 1));
  const status = state && state.status ? state.status : "listening";
  // [PERF] The active poll is what makes the badge/ring feel responsive, and
  // the endpoint is a fused local read (loopback, in-process) so a tighter
  // cadence is cheap: 250ms -> 120ms while something is actually happening.
  // Idle stays at 600ms — there is nothing to show.
  if (status === "thinking" || status === "speaking" || status === "hearing") return 120;
  return 600;
}

function updateVoiceToggleUI(enabled) {
  voiceInputEnabled = !!enabled;
  const btn = document.getElementById("btn-voice-toggle");
  if (btn) {
    const ico = btn.querySelector(".mode-ico");
    const label = voiceInputEnabled ? "Voice: On" : "Voice: Text";
    // keep icon, replace text after it
    if (ico) {
      btn.childNodes.forEach((n) => {
        if (n.nodeType === 3 && n.textContent.trim().length) n.remove();
      });
      btn.appendChild(document.createTextNode(" " + label));
      // ensure first text after icon is correct - rebuild inner slightly
      // simpler: set innerHTML preserving icon
      btn.innerHTML = `<span class="mode-ico">\uD83C\uD999</span> ${label}`;
    } else {
      btn.textContent = label;
    }
    btn.classList.toggle("voice-off", !voiceInputEnabled);
    btn.classList.toggle("voice-on", voiceInputEnabled);
    // reuse badge visual cue via border accent
    btn.style.borderColor = voiceInputEnabled ? "rgba(0,229,255,0.22)" : "rgba(255,255,255,0.12)";
    btn.style.opacity = voiceInputEnabled ? "1" : "0.85";
  }
}

async function toggleVoiceMode() {
  const next = !voiceInputEnabled;
  const btn = document.getElementById("btn-voice-toggle");
  if (btn) btn.disabled = true;
  try {
    const res = await jarvisFetch(BACKEND_URL + "/voice-mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: next }),
    });
    if (!res.ok) throw new Error("voice-mode POST failed " + res.status);
    const data = await res.json();
    updateVoiceToggleUI(data.voice_input_enabled);
  } catch (_e) {
    // keep current UI, will resync on next poll
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function pollUiState() {
  try {
    const response = await jarvisFetch(BACKEND_URL + "/ui-state");
    const data = await response.json();
    voiceStateFailures = 0;
    renderVoiceState(data.state || {});
    // voice/text toggle sync - top-level or inside state both work
    if (typeof data.voice_input_enabled !== "undefined") {
      updateVoiceToggleUI(data.voice_input_enabled);
    } else if (data.state && typeof data.state.voice_input_enabled !== "undefined") {
      updateVoiceToggleUI(data.state.voice_input_enabled);
    }
    const log = data.voice_log || {};
    if (log.id && log.id !== lastVoiceLogId) {
      lastVoiceLogId = log.id;
      const chatBox = document.getElementById("chat-box");
      const responseText = (log.response || "").trim();
      // Echo guard: find the newest existing bot bubble's text. Typed replies
      // stream straight into the UI, so when the same response comes back
      // through /ui-state mirroring it must not be appended a second time.
      let lastBotText = "";
      if (chatBox) {
        for (let i = chatBox.children.length - 1; i >= 0; i--) {
          const el = chatBox.children[i];
          if (el.classList && el.classList.contains("bot")) {
            const t = el.querySelector(".msg-text");
            lastBotText = ((t && t.textContent) || el.textContent || "").trim();
            break;
          }
        }
      }
      if (log.message) addBubble(chatBox, log.message, "user");
      if (responseText && lastBotText !== responseText) {
        addBubble(chatBox, responseText, "bot");
      }
    }
    voiceStatePollTimer = setTimeout(pollUiState, pollDelayFor(data.state, 0));
  } catch (_error) {
    voiceStateFailures += 1;
    if (voiceStateFailures >= 6) renderVoiceState({ status: "offline" });
    else renderVoiceState({ status: "reconnecting" });
    voiceStatePollTimer = setTimeout(pollUiState, pollDelayFor(null, voiceStateFailures));
  }
}

/* ── Scroll progress ── */
function updateScrollProgress() {
  const box = document.getElementById("chat-box");
  const bar = document.getElementById("scroll-progress");
  if (!box || !bar) return;
  const max = box.scrollHeight - box.clientHeight;
  const p = max <= 0 ? 0 : (box.scrollTop / max) * 100;
  bar.style.width = `${p.toFixed(2)}%`;
  bar.style.opacity = p > 2 ? "1" : "0.35";
}

/* ── Particle field (neural net) ── */
function initParticles() {
  const c = document.getElementById("particle-canvas");
  if (!c) return;
  const ctx = c.getContext("2d");
  let w, h, dpr, pts = [], raf = 0, mx = -9999, my = -9999;
  const COUNT = 62;
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    w = c.clientWidth = c.offsetWidth; h = c.clientHeight = c.offsetHeight;
    c.width = Math.floor(w * dpr); c.height = Math.floor(h * dpr);
    c.style.width = `${w}px`; c.style.height = `${h}px`;
    ctx.setTransform(dpr,0,0,dpr,0,0);
  }
  function spawn() {
    pts = Array.from({length: COUNT}, () => ({
      x: Math.random()*w, y: Math.random()*h,
      vx: (Math.random()-0.5)*0.34, vy: (Math.random()-0.5)*0.34,
      r: 0.9 + Math.random()*1.7, a: 0.22 + Math.random()*0.42
    }));
  }
  function tick() {
    ctx.clearRect(0,0,w,h);
    // update
    for (const p of pts) {
      p.x += p.vx; p.y += p.vy;
      if (p.x < -10) p.x = w+10; if (p.x > w+10) p.x = -10;
      if (p.y < -10) p.y = h+10; if (p.y > h+10) p.y = -10;
      // mouse gently tugs
      const dx = mx - p.x, dy = my - p.y, d = Math.hypot(dx,dy);
      if (d < 140) { p.x += dx*0.0009; p.y += dy*0.0009; }
    }
    // connections
    ctx.lineWidth = 0.9;
    for (let i=0;i<pts.length;i++) for(let j=i+1;j<pts.length;j++){
      const a=pts[i], b=pts[j], d=Math.hypot(a.x-b.x,a.y-b.y);
      if (d < 132) {
        const o = (1 - d/132) * 0.16;
        ctx.strokeStyle = `rgba(0,229,255,${o.toFixed(3)})`;
        ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
        if (d < 92) {
          ctx.strokeStyle = `rgba(124,92,255,${(o*0.55).toFixed(3)})`;
          ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
        }
      }
    }
    // dots with glow
    for (const p of pts) {
      const g = ctx.createRadialGradient(p.x,p.y,0,p.x,p.y,p.r*3.2);
      g.addColorStop(0, `rgba(0,229,255,${p.a.toFixed(2)})`);
      g.addColorStop(1, "rgba(0,229,255,0)");
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(p.x,p.y,p.r*3.2,0,Math.PI*2); ctx.fill();
      ctx.fillStyle = `rgba(210,245,255,${(p.a*0.95).toFixed(2)})`;
      ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2); ctx.fill();
    }
    raf = requestAnimationFrame(tick);
  }
  resize(); spawn(); tick();
  window.addEventListener("resize", ()=>{ resize(); spawn(); });
  window.addEventListener("mousemove", (e)=>{ mx=e.clientX; my=e.clientY; });
  window.addEventListener("mouseleave", ()=>{ mx=-9999; my=-9999; });
  document.addEventListener("visibilitychange", ()=>{ if(document.hidden) cancelAnimationFrame(raf); else tick(); });
}

/* ── Subtle 3D parallax on header / app ── */
function initParallax() {
  const app = document.getElementById("app");
  const header = document.getElementById("header");
  const inputArea = document.getElementById("input-area");
  let rx=0, ry=0, tx=0, ty=0;
  window.addEventListener("mousemove", (e)=>{
    const cx = window.innerWidth/2, cy = window.innerHeight/2;
    tx = (e.clientX - cx) / cx; ty = (e.clientY - cy) / cy;
  });
  function loop(){
    rx += (ty*1.2 - rx)*0.06; ry += (tx*1.6 - ry)*0.06;
    if (header) header.style.transform = `translateZ(18px) rotateX(${-rx*0.55}deg) rotateY(${ry*0.55}deg)`;
    if (inputArea) inputArea.style.transform = `translateZ(14px) rotateX(${-rx*0.35}deg) rotateY(${ry*0.35}deg)`;
    if (app) app.style.transform = `rotateX(${rx*0.06}deg) rotateY(${ry*0.08}deg)`;
    requestAnimationFrame(loop);
  }
  // respect reduced motion
  if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) loop();
}

/* ── Magnetic hover for buttons ── */
function initMagnetic() {
  const btns = [document.getElementById("send-button"), document.querySelector(".btn-stop")].filter(Boolean);
  for (const b of btns) {
    b.addEventListener("mousemove", (e)=>{
      const r=b.getBoundingClientRect();
      const dx=(e.clientX-(r.left+r.width/2))/ (r.width/2);
      const dy=(e.clientY-(r.top+r.height/2))/ (r.height/2);
      b.style.transform = `translate(${dx*3}px, ${dy*2}px) translateZ(8px) scale(1.02)`;
    });
    b.addEventListener("mouseleave", ()=>{ b.style.transform=""; });
  }
}

pollUiState();
updateModeIndicator();
// capsule respawn + voice toggle wiring (guard null for WIP safety)
try {
  const _capsuleBtn = document.getElementById("btn-capsule");
  if (_capsuleBtn) _capsuleBtn.addEventListener("click", () => { try { IPC.send("create-capsule"); } catch (_e) {} });
  const _voiceBtn = document.getElementById("btn-voice-toggle");
  if (_voiceBtn) _voiceBtn.addEventListener("click", () => toggleVoiceMode());
  updateVoiceToggleUI(true);
} catch (_e) {}
requestAnimationFrame(()=>{ initParticles(); initParallax(); initMagnetic(); });
// scroll progress wiring
const _box = document.getElementById("chat-box");
if (_box) { _box.addEventListener("scroll", updateScrollProgress, {passive:true}); updateScrollProgress(); }
// initial tilt for hero message
document.querySelectorAll(".message").forEach(attachTilt);
// focus glow wander
const wrapper = document.getElementById("input-wrapper");
const input = document.getElementById("user-input");
if (wrapper && input) {
  input.addEventListener("focus", ()=> wrapper.style.setProperty("--focus", "1"));
  input.addEventListener("blur", ()=> wrapper.style.setProperty("--focus", "0"));
}
// keyboard ripple
if (input) input.addEventListener("keydown", (e)=>{
  if (e.key==="Enter") {
    wrapper.animate([{transform:"scale(0.995)"},{transform:"scale(1)"}],{duration:160, easing:"cubic-bezier(.16,1,.3,1)"});
  }
});
if (!document.body.dataset.mode) document.body.dataset.mode = 'chat';
if (!document.body.dataset.state) document.body.dataset.state = 'idle';

/* ═══════════════════════════════════════════════════════
    Model sidebar — per-role switchers (chat / intent / TTS / listening / vision / browser_tool / planner).
   Hamburger opens it; providers render from GET /settings,
   models lazy-load from GET /providers/{id}/models, a click
   POSTs /settings/model (or /settings/chat-model for compat) and
   takes effect on the very next reply/turn (no restart).
   Browser tool uses free-text input instead of a list.
   ═══════════════════════════════════════════════════════ */
let sidebarOpen = false;
let providersCache = null;   // GET /settings -> .providers
let activeChatModel = null;  // GET /settings -> .chat_model
let activeTtsModel = null;
let activeVisionModel = null;
let activeBrowserToolModel = null;
let activeListeningModel = null; // GET /settings -> .listening_model
let activePlannerModel = null;   // GET /settings -> .planner_model
let activeIntentModel = null;    // GET /settings -> .intent_model (F56)
const modelsCache = {};      // providerId -> { models, ts }
const expandedProviders = new Set();
const expandedTtsProviders = new Set();
const expandedVisionProviders = new Set();
const expandedListeningProviders = new Set();
const expandedBrowserProviders = new Set();
const expandedPlannerProviders = new Set();
const expandedIntentProviders = new Set();
const MODEL_CACHE_TTL_MS = 5 * 60 * 1000;
const filterState = { chat: {}, tts: {}, vision: {}, listening: {}, browser_tool: {}, planner: {}, intent: {} };
const showAllState = { chat: {}, tts: {}, vision: {}, listening: {}, browser_tool: {}, planner: {}, intent: {} };
let roleAllowedCache = null; // {role: [provider ids]} for every registry role
let customProviderRoles = null; // roles that accept a user-added provider
let lastFallback = null;

function escapeHtml(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function slugifyProviderId(s) {
  return String(s || "").toLowerCase().trim()
    .replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 32);
}

function flashSidebarConfirm(msg, isError) {
  const bar = document.getElementById("model-sidebar");
  if (!bar) return;
  let el = document.getElementById("sidebar-confirm");
  if (!el) {
    el = document.createElement("span");
    el.id = "sidebar-confirm";
    el.className = "sidebar-confirm";
    bar.appendChild(el);
  }
  el.textContent = msg;
  el.className = "sidebar-confirm" + (isError ? " err" : "");
  el.classList.add("show");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove("show"), 2400);
}

function toggleSidebar(force) {
  const next = typeof force === "boolean" ? force : !sidebarOpen;
  sidebarOpen = next;
  document.body.classList.toggle("sidebar-open", next);
  const aside = document.getElementById("model-sidebar");
  if (aside) aside.setAttribute("aria-hidden", String(!next));
  if (next && !providersCache) loadProviders();
}

async function loadProviders() {
  const list = document.getElementById("model-provider-list");
  try {
    const res = await jarvisFetch(BACKEND_URL + "/settings");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    providersCache = data.providers || [];
    roleAllowedCache = data.role_allowed || null;
    customProviderRoles = data.custom_provider_roles || null;
    lastFallback = data.last_fallback || null;
    activeChatModel = data.chat_model || null;
    activeTtsModel = data.tts_model || null;
    activeVisionModel = data.vision_model || null;
    activeBrowserToolModel = data.browser_tool_model || null;
    activeListeningModel = data.listening_model || null;
    activePlannerModel = data.planner_model || null;
    activeIntentModel = data.intent_model || null;
    renderProviders();
    renderIntentProviders();
    renderTtsProviders();
    renderListeningProviders();
    renderVisionProviders();
    renderBrowserToolSection();
    renderPlannerProviders();
    renderChatFallbackWarning();
    _syncAddProviderVisibility();
  } catch (err) {
    if (list) list.innerHTML =
      '<span class="model-error">could not load settings (' +
      escapeHtml(String(err.message || err)) + ')</span>';
  }
}

function renderChatFallbackWarning() {
  const sec = document.getElementById("section-chat");
  if (!sec) return;
  let warn = document.getElementById("chat-fallback-warn");
  if (warn) warn.remove();
  if (!lastFallback || !lastFallback.ts) return;
  const age = Date.now() / 1000 - Number(lastFallback.ts || 0);
  if (age > 600) return; // 10 min
  const fbProvider = String(lastFallback.fallback_provider || lastFallback.provider || "gemini");
  warn = document.createElement("div");
  warn.id = "chat-fallback-warn";
  warn.className = "model-warning";
  warn.textContent = "last reply fell back to " + fbProvider + " - selected model unavailable";
  // insert after title
  const title = sec.querySelector(".sidebar-section-title");
  if (title && title.nextSibling) sec.insertBefore(warn, title.nextSibling);
  else sec.appendChild(warn);
}

// Providers that legitimately need no credential. Their has_key=false is
// the normal state, not a misconfiguration, so they render with a live dot
// and a "no key needed" label instead of a grey "no key" row. The local
// Ollama server (F56) is the same idea with the opposite polarity: it
// reports a placeholder credential, so it is listed here to be LABELLED
// "local · no key needed" rather than "key ok".
const KEYLESS_PROVIDERS_BY_ROLE = {
  listening: ["whisper"],
  tts: ["gtts"],
  chat: ["ollama"],
  intent: ["ollama"],
};

// Providers that run on THIS machine — labelled "local" instead of "free".
const LOCAL_PROVIDERS = ["whisper", "ollama"];

function _isKeylessProvider(role, providerId) {
  const ids = KEYLESS_PROVIDERS_BY_ROLE[role];
  return Array.isArray(ids) && ids.indexOf(providerId) !== -1;
}

function _renderProvidersForRole(listId, activeModel, expandedSet, role) {
  const list = document.getElementById(listId);
  if (!list || !providersCache) return;
  const cm = activeModel || {};
  list.innerHTML = "";
  let visibleProviders = providersCache;
  if (roleAllowedCache && Array.isArray(roleAllowedCache[role])) {
    const allowed = new Set(roleAllowedCache[role]);
    visibleProviders = providersCache.filter((p) => allowed.has(p.id));
  }
  if (visibleProviders.length === 0 && providersCache.length > 0) {
    list.innerHTML = '<span class="model-error">no providers available for this role</span>';
    return;
  }
  for (const p of visibleProviders) {
    // Listening role: Inworld and Sarvam need their API keys — without them
    // the option renders disabled instead of selectable. Providers that need
    // no credential at all (local whisper, the free Google TTS fallback)
    // report has_key=false as their NORMAL state, so that must never render
    // as a grey "no key" row or gate selection.
    const keyGated = role === "listening" && (p.id === "inworld" || p.id === "sarvam") && !p.has_key;
    const keyless = _isKeylessProvider(role, p.id);
    const isOpen = !keyGated && expandedSet.has(p.id);
    const sec = document.createElement("div");
    sec.className = "provider-section" + (isOpen ? " expanded" : "");
    sec.dataset.providerId = p.id;

    const row = document.createElement("div");
    row.className = "provider-row";

    const head = document.createElement("button");
    head.type = "button";
    head.className =
      "provider-head" + (cm.provider === p.id ? " active" : "") + (isOpen ? " open" : "") +
      (keyGated ? " disabled" : "");
    let metaText;
    if (keyGated) metaText = "(API key missing)";
    else if (keyless) metaText = (LOCAL_PROVIDERS.indexOf(p.id) !== -1 ? "local" : "free") + " · no key needed";
    else metaText = (p.has_key ? "key ok" : "no key") + (p.source === "custom" ? " · custom" : "");
    const dotOk = p.has_key || keyless;
    head.innerHTML =
      '<span class="provider-key-dot' + (dotOk ? " has-key" : " no-key") + '"></span>' +
      '<span class="provider-name">' + escapeHtml(p.name) + '</span>' +
      '<span class="provider-meta">' + metaText + '</span>' +
      '<span class="provider-chevron">▾</span>';
    if (keyGated) {
      head.disabled = true;
    } else {
      head.addEventListener("click", () => _toggleProviderModelsForRole(p.id, sec, expandedSet, role));
    }
    row.appendChild(head);

    if (!keyGated) {
      const refresh = document.createElement("button");
      refresh.type = "button";
      refresh.className = "provider-refresh";
      refresh.title = "Refresh model list";
      refresh.textContent = "↻";
      refresh.addEventListener("click", (ev) => {
        ev.stopPropagation();
        delete modelsCache[p.id];
        if (showAllState[role]) delete showAllState[role][p.id];
        sec.classList.add("expanded");
        expandedSet.add(p.id);
        _loadProviderModelsForRole(p.id, sec.querySelector(".provider-models"), role);
      });
      row.appendChild(refresh);
    }

    sec.appendChild(row);

    if (keyGated) {
      list.appendChild(sec);
      continue;
    }

    const body = document.createElement("div");
    body.className = "provider-models";
    sec.appendChild(body);
    list.appendChild(sec);

    if (isOpen) {
      const cached = modelsCache[p.id];
      if (cached) _renderModelsForRole(body, p.id, cached.models, role);
      else _loadProviderModelsForRole(p.id, body, role);
    }
  }
}

function renderProviders() {
  _renderProvidersForRole("model-provider-list", activeChatModel, expandedProviders, "chat");
}
function renderIntentProviders() {
  // F56 — the model that classifies every message (services/intent.py). Same
  // provider → model list as every other role; picking one here changes what
  // routes chat / tool / screen / region / research / task.
  _renderProvidersForRole("intent-provider-list", activeIntentModel, expandedIntentProviders, "intent");
}
function renderTtsProviders() {
  _renderProvidersForRole("tts-provider-list", activeTtsModel, expandedTtsProviders, "tts");
}
function renderListeningProviders() {
  _renderProvidersForRole("listening-provider-list", activeListeningModel, expandedListeningProviders, "listening");
}
function renderVisionProviders() {
  _renderProvidersForRole("vision-provider-list", activeVisionModel, expandedVisionProviders, "vision");
}
function renderPlannerProviders() {
  _renderProvidersForRole("planner-provider-list", activePlannerModel, expandedPlannerProviders, "planner");
}
function renderBrowserToolSection() {
  const input = document.getElementById("browser-model-input");
  const provInput = document.getElementById("browser-model-provider");
  if (input && activeBrowserToolModel) {
    input.value = activeBrowserToolModel.model || "";
    if (provInput) provInput.value = activeBrowserToolModel.provider || "";
    input.placeholder = activeBrowserToolModel.model ? activeBrowserToolModel.model : "e.g. accounts/fireworks/models/qwen3p7-plus";
  }
  // The browser-tool provider list uses the SAME expandable provider → model
  // list as every other role, so its models are selectable by click instead of
  // only reachable by typing an id into the free-text form below (which stays
  // as the manual override for ids a provider's list does not publish).
  _renderProvidersForRole(
    "browser-provider-list", activeBrowserToolModel, expandedBrowserProviders,
    "browser_tool");
}

function _toggleProviderModelsForRole(providerId, sectionEl, expandedSet, role) {
  if (!sectionEl) return;
  const body = sectionEl.querySelector(".provider-models");
  const head = sectionEl.querySelector(".provider-head");
  if (!body) return;
  if (sectionEl.classList.contains("expanded")) {
    sectionEl.classList.remove("expanded");
    if (head) head.classList.remove("open");
    expandedSet.delete(providerId);
    if (showAllState[role]) delete showAllState[role][providerId];
    return;
  }
  sectionEl.classList.add("expanded");
  if (head) head.classList.add("open");
  expandedSet.add(providerId);
  const cached = modelsCache[providerId];
  if (cached && Date.now() - cached.ts < MODEL_CACHE_TTL_MS) {
    _renderModelsForRole(body, providerId, cached.models, role);
    return;
  }
  _loadProviderModelsForRole(providerId, body, role);
}

function toggleProviderModels(providerId, sectionEl) {
  return _toggleProviderModelsForRole(providerId, sectionEl, expandedProviders, "chat");
}

async function _loadProviderModelsForRole(providerId, bodyEl, role) {
  if (!bodyEl) return;
  bodyEl.innerHTML = '<span class="model-loading">loading models…</span>';
  try {
    const res = await jarvisFetch(
      BACKEND_URL + "/providers/" + encodeURIComponent(providerId) + "/models"
    );
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || ("HTTP " + res.status));
    modelsCache[providerId] = { models: data.models || [], ts: Date.now() };
    _renderModelsForRole(bodyEl, providerId, data.models || [], role);
  } catch (err) {
    bodyEl.innerHTML =
      '<span class="model-error">' + escapeHtml(String(err.message || err)) + '</span>';
  }
}

async function loadProviderModels(providerId, bodyEl) {
  return _loadProviderModelsForRole(providerId, bodyEl, "chat");
}

function _renderModelsForRole(bodyEl, providerId, models, role) {
  let cm = {};
  if (role === "chat") cm = activeChatModel || {};
  else if (role === "tts") cm = activeTtsModel || {};
  else if (role === "vision") cm = activeVisionModel || {};
  else if (role === "browser_tool") cm = activeBrowserToolModel || {};
  else if (role === "listening") cm = activeListeningModel || {};
  else if (role === "planner") cm = activePlannerModel || {};
  else if (role === "intent") cm = activeIntentModel || {};
  bodyEl.innerHTML = "";
  if (!models || !models.length) {
    const label = role === "chat" ? "no chat models available" : "no models available";
    bodyEl.innerHTML = '<span class="model-error">' + escapeHtml(label) + '</span>';
    return;
  }
  if (!filterState[role]) filterState[role] = {};
  if (!showAllState[role]) showAllState[role] = {};

  const filterWrap = document.createElement("div");
  filterWrap.className = "model-filter-wrap";
  const filterInput = document.createElement("input");
  filterInput.type = "text";
  filterInput.className = "model-filter-input";
  filterInput.placeholder = "Filter models...";
  filterInput.value = filterState[role][providerId] || "";
  filterInput.setAttribute("aria-label", "Filter models");
  filterWrap.appendChild(filterInput);
  bodyEl.appendChild(filterWrap);

  const listWrap = document.createElement("div");
  listWrap.className = "model-list-items";
  bodyEl.appendChild(listWrap);

  function renderList() {
    const raw = String(filterInput.value || "");
    filterState[role][providerId] = raw;
    const q = raw.trim().toLowerCase();
    const isFiltering = q.length > 0;
    let filtered = models;
    if (isFiltering) {
      filtered = models.filter((m) => {
        const disp = String(m.display || m.id || "").toLowerCase();
        const id = String(m.id || "").toLowerCase();
        return disp.includes(q) || id.includes(q);
      });
    }
    const showAll = !!showAllState[role][providerId];
    let toRender = filtered;
    let capped = false;
    if (!isFiltering && !showAll && filtered.length > 30) {
      toRender = filtered.slice(0, 30);
      capped = true;
    }
    listWrap.innerHTML = "";
    if (!filtered.length) {
      const msg = isFiltering ? "no matches" : (role === "chat" ? "no chat models available" : "no models available");
      listWrap.innerHTML = '<span class="model-error">' + escapeHtml(msg) + '</span>';
      return;
    }
    for (const m of toRender) {
      const btn = document.createElement("button");
      btn.type = "button";
      const active = cm.provider === providerId && cm.model === m.id;
      btn.className = "model-item" + (active ? " active" : "");
      btn.title = m.id;
      btn.innerHTML =
        '<span class="model-name">' + escapeHtml(m.display || m.id) + '</span>' +
        '<span class="model-check">' + (active ? "✓" : "") + '</span>';
      btn.addEventListener("click", () => _selectModelForRole(role, providerId, m.id));
      listWrap.appendChild(btn);
    }
    if (capped) {
      const more = document.createElement("button");
      more.type = "button";
      more.className = "model-show-all";
      more.textContent = "Show all (" + filtered.length + ")";
      more.addEventListener("click", () => {
        showAllState[role][providerId] = true;
        renderList();
      });
      listWrap.appendChild(more);
    }
  }

  filterInput.addEventListener("input", renderList);
  renderList();
}

function renderModels(bodyEl, providerId, models) {
  return _renderModelsForRole(bodyEl, providerId, models, "chat");
}

async function _selectModelForRole(role, providerId, modelId) {
  try {
    const res = await jarvisFetch(BACKEND_URL + "/settings/model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role, provider: providerId, model: modelId }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || ("HTTP " + res.status));
    if (role === "chat") {
      activeChatModel = data.chat_model || data.model || { provider: providerId, model: modelId };
      renderProviders();
      flashSidebarConfirm("✓ now responding with " + (activeChatModel.model || modelId));
    } else if (role === "tts") {
      activeTtsModel = data.tts_model || data.model || { provider: providerId, model: modelId };
      renderTtsProviders();
      flashSidebarConfirm("✓ voice model: " + (activeTtsModel.model || modelId));
    } else if (role === "listening") {
      activeListeningModel = data.listening_model || data.model || { provider: providerId, model: modelId };
      renderListeningProviders();
      flashSidebarConfirm("✓ listening engine: " + (activeListeningModel.provider || providerId));
    } else if (role === "vision") {
      activeVisionModel = data.vision_model || data.model || { provider: providerId, model: modelId };
      renderVisionProviders();
      flashSidebarConfirm("✓ vision model: " + (activeVisionModel.model || modelId));
    } else if (role === "browser_tool") {
      activeBrowserToolModel = data.browser_tool_model || data.model || { provider: providerId, model: modelId };
      renderBrowserToolSection();
      flashSidebarConfirm("✓ browser model: " + (activeBrowserToolModel.model || modelId));
    } else if (role === "planner") {
      activePlannerModel = data.planner_model || data.model || { provider: providerId, model: modelId };
      renderPlannerProviders();
      flashSidebarConfirm("✓ planner model: " + (activePlannerModel.model || modelId));
    } else if (role === "intent") {
      activeIntentModel = data.intent_model || data.model || { provider: providerId, model: modelId };
      renderIntentProviders();
      flashSidebarConfirm("✓ intent classifier: " + (activeIntentModel.model || modelId));
    }
  } catch (err) {
    flashSidebarConfirm("✗ " + (err.message || err), true);
  }
}

async function selectModel(providerId, modelId) {
  // keep chat compat: use generic role endpoint, fallback to old chat-model
  try {
    await _selectModelForRole("chat", providerId, modelId);
  } catch (_e) {
    // fallback to old endpoint if generic fails (should not happen)
    try {
      const res = await jarvisFetch(BACKEND_URL + "/settings/chat-model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider: providerId, model: modelId }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || ("HTTP " + res.status));
      activeChatModel = data.chat_model || { provider: providerId, model: modelId };
      renderProviders();
      flashSidebarConfirm("✓ now responding with " + (activeChatModel.model || modelId));
    } catch (err) {
      flashSidebarConfirm("✗ " + (err.message || err), true);
    }
  }
}

async function saveBrowserToolModel() {
  const modelEl = document.getElementById("browser-model-input");
  const provEl = document.getElementById("browser-model-provider");
  const errEl = document.getElementById("browser-form-error");
  if (!modelEl || !provEl) return;
  const model = modelEl.value.trim();
  const provider = provEl.value.trim() || (activeBrowserToolModel && activeBrowserToolModel.provider) || "fireworks";
  if (errEl) errEl.textContent = "";
  if (!model || !provider) {
    if (errEl) errEl.textContent = "Provider and model are required.";
    return;
  }
  const btn = document.getElementById("btn-browser-save");
  if (btn) btn.disabled = true;
  try {
    const res = await jarvisFetch(BACKEND_URL + "/settings/model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role: "browser_tool", provider, model }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || ("HTTP " + res.status));
    activeBrowserToolModel = data.browser_tool_model || { provider, model };
    renderBrowserToolSection();
    flashSidebarConfirm("✓ browser model: " + model);
  } catch (err) {
    if (errEl) errEl.textContent = err.message || String(err);
    flashSidebarConfirm("✗ " + (err.message || err), true);
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function submitAddProvider(ev) {
  ev.preventDefault();
  const form = ev.target;
  const nameEl = document.getElementById("provider-name");
  const keyEl = document.getElementById("provider-key");
  const urlEl = document.getElementById("provider-base-url");
  const errEl = document.getElementById("provider-form-error");
  if (!form || !nameEl || !keyEl || !urlEl || !errEl) return;
  const name = nameEl.value.trim();
  const apiKey = keyEl.value.trim();
  const baseUrl = urlEl.value.trim();
  const id = slugifyProviderId(name);
  errEl.textContent = "";
  if (!name || !apiKey || !baseUrl || !id) {
    errEl.textContent = "Name, API key and base URL are all required.";
    return;
  }
  const saveBtn = form.querySelector(".btn-provider-save");
  if (saveBtn) saveBtn.disabled = true;
  try {
    const res = await jarvisFetch(BACKEND_URL + "/settings/provider", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, name, api_key: apiKey, base_url: baseUrl }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || ("HTTP " + res.status));
    form.reset();
    form.hidden = true;
    providersCache = null; // force a fresh GET /settings
    for (const k of Object.keys(modelsCache)) delete modelsCache[k];
    await loadProviders();
    flashSidebarConfirm("✓ provider added");
  } catch (err) {
    errEl.textContent = err.message || String(err);
  } finally {
    if (saveBtn) saveBtn.disabled = false;
  }
}

/* ═══════════════════════════════════════════════════════
   Per-functionality custom providers (add / test / save / choose).
   Every model section that accepts a user-added OpenAI-compatible
   provider gets its own "+ ADD CUSTOM PROVIDER" entry. The form asks
   for base URL + API key (name optional), offers TEST (live check that
   stores nothing) and SAVE, and after a save auto-fetches the models on
   that key so one can be chosen for that function right away.
   ═══════════════════════════════════════════════════════ */
function _syncAddProviderVisibility() {
  // Backend is the source of truth for which roles accept custom providers;
  // fall back to the shipped LLM-backed set when it does not say.
  const roles = Array.isArray(customProviderRoles) && customProviderRoles.length
    ? customProviderRoles
    : ["chat", "vision", "browser_tool", "planner"];
  document.querySelectorAll(".provider-add").forEach((wrap) => {
    wrap.hidden = roles.indexOf(wrap.dataset.role) === -1;
  });
}

function _hostNameFromUrl(url) {
  try { return new URL(url).hostname || ""; } catch (_e) { return ""; }
}

function _apStatus(wrap, msg, isError) {
  const el = wrap.querySelector(".ap-status");
  if (!el) return;
  el.textContent = msg || "";
  el.className = "provider-form-error ap-status" + (isError || !msg ? "" : " ok");
}

function _apReset(wrap) {
  const form = wrap.querySelector(".role-add-form");
  const pick = wrap.querySelector(".ap-pick");
  if (form) { form.reset(); form.hidden = true; }
  if (pick) pick.hidden = true;
  _apStatus(wrap, "", true);
}

async function _testRoleProvider(wrap) {
  const urlEl = wrap.querySelector(".ap-url");
  const keyEl = wrap.querySelector(".ap-key");
  const btn = wrap.querySelector(".btn-provider-test");
  const baseUrl = urlEl ? urlEl.value.trim() : "";
  const apiKey = keyEl ? keyEl.value.trim() : "";
  if (!baseUrl || !apiKey) {
    _apStatus(wrap, "Base URL and API key are required.", true);
    return;
  }
  if (btn) btn.disabled = true;
  _apStatus(wrap, "testing…", true);
  try {
    const res = await jarvisFetch(BACKEND_URL + "/settings/provider/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_url: baseUrl, api_key: apiKey }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || ("HTTP " + res.status));
    _apStatus(wrap, "✓ working — " + (data.count || 0) + " models found", false);
  } catch (err) {
    _apStatus(wrap, "✗ " + (err.message || err), true);
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function _fetchProviderModels(providerId) {
  const res = await jarvisFetch(
    BACKEND_URL + "/providers/" + encodeURIComponent(providerId) + "/models"
  );
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || ("HTTP " + res.status));
  return data.models || [];
}

function _populateModelPicker(wrap, providerId, models, roleLabel) {
  const pick = wrap.querySelector(".ap-pick");
  const select = wrap.querySelector(".ap-model");
  if (!pick || !select) return;
  select.innerHTML = "";
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.display || m.id;
    select.appendChild(opt);
  }
  pick.dataset.providerId = providerId;
  pick.hidden = false;
  _apStatus(wrap,
    "✓ saved — " + models.length + " models found; choose one for " +
    roleLabel + " (or pick later from the list)", false);
  select.focus();
}

async function _saveRoleProvider(role, wrap) {
  const nameEl = wrap.querySelector(".ap-name");
  const urlEl = wrap.querySelector(".ap-url");
  const keyEl = wrap.querySelector(".ap-key");
  const saveBtn = wrap.querySelector(".ap-actions .btn-provider-save");
  const baseUrl = urlEl ? urlEl.value.trim() : "";
  const apiKey = keyEl ? keyEl.value.trim() : "";
  if (!baseUrl || !apiKey) {
    _apStatus(wrap, "Base URL and API key are required.", true);
    return;
  }
  // Name is optional: fall back to the endpoint's host, then "custom".
  const name = (nameEl && nameEl.value.trim()) ||
    _hostNameFromUrl(baseUrl) || "custom";
  const id = slugifyProviderId(name) || "custom";
  const roleLabel = (wrap.closest(".sidebar-section")
    ? wrap.closest(".sidebar-section").querySelector(".sidebar-section-title")
    : null);
  const label = roleLabel ? roleLabel.textContent.trim().toLowerCase() : role;
  if (saveBtn) saveBtn.disabled = true;
  _apStatus(wrap, "validating & saving…", true);
  try {
    const res = await jarvisFetch(BACKEND_URL + "/settings/provider", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, name, api_key: apiKey, base_url: baseUrl }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || ("HTTP " + res.status));
    // Refresh the provider lists everywhere, then auto-fetch the models on
    // this key and let the user choose one for this function.
    providersCache = null;
    for (const k of Object.keys(modelsCache)) delete modelsCache[k];
    await loadProviders();
    _apStatus(wrap, "✓ saved — fetching models…", true);
    try {
      const models = await _fetchProviderModels(id);
      _populateModelPicker(wrap, id, models, label);
    } catch (err) {
      _apStatus(wrap, "✓ saved, but the model list failed: " +
        (err.message || err), true);
    }
    flashSidebarConfirm("✓ provider added: " + name);
  } catch (err) {
    _apStatus(wrap, "✗ " + (err.message || err), true);
  } finally {
    if (saveBtn) saveBtn.disabled = false;
  }
}

async function _useRoleProviderModel(role, wrap) {
  const pick = wrap.querySelector(".ap-pick");
  const select = wrap.querySelector(".ap-model");
  const providerId = pick ? pick.dataset.providerId : "";
  const modelId = select ? select.value : "";
  if (!providerId || !modelId) return;
  await _selectModelForRole(role, providerId, modelId);
  _apReset(wrap);
}

function initRoleAddForms() {
  document.querySelectorAll(".provider-add").forEach((wrap) => {
    const role = wrap.dataset.role;
    const btn = wrap.querySelector(".role-add-btn");
    const form = wrap.querySelector(".role-add-form");
    const testBtn = wrap.querySelector(".btn-provider-test");
    const useBtn = wrap.querySelector(".ap-use");
    if (!btn || !form) return;
    btn.addEventListener("click", () => {
      if (!form.hidden) { _apReset(wrap); return; }
      form.hidden = false;
      const nameEl = wrap.querySelector(".ap-name");
      if (nameEl) nameEl.focus();
    });
    form.addEventListener("submit", (ev) => {
      ev.preventDefault();
      _saveRoleProvider(role, wrap);
    });
    if (testBtn) testBtn.addEventListener("click", () => _testRoleProvider(wrap));
    if (useBtn) useBtn.addEventListener("click", () => _useRoleProviderModel(role, wrap));
  });
}

// sidebar wiring (guarded like the capsule/voice buttons)
try {
  const _burgerBtn = document.getElementById("btn-hamburger");
  if (_burgerBtn) _burgerBtn.addEventListener("click", () => toggleSidebar());
  const _sidebarClose = document.getElementById("sidebar-close");
  if (_sidebarClose) _sidebarClose.addEventListener("click", () => toggleSidebar(false));
  const _sidebarBackdrop = document.getElementById("sidebar-backdrop");
  if (_sidebarBackdrop) _sidebarBackdrop.addEventListener("click", () => toggleSidebar(false));
  const _addProviderBtn = document.getElementById("btn-add-provider");
  const _addProviderForm = document.getElementById("add-provider-form");
  if (_addProviderBtn && _addProviderForm) {
    _addProviderBtn.addEventListener("click", () => {
      _addProviderForm.hidden = !_addProviderForm.hidden;
      if (!_addProviderForm.hidden) {
        const _nameInput = document.getElementById("provider-name");
        if (_nameInput) _nameInput.focus();
      }
    });
    _addProviderForm.addEventListener("submit", submitAddProvider);
  }
  const _browserSaveBtn = document.getElementById("btn-browser-save");
  if (_browserSaveBtn) _browserSaveBtn.addEventListener("click", saveBrowserToolModel);
  initRoleAddForms();
  _syncAddProviderVisibility();
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && sidebarOpen) toggleSidebar(false);
  });
} catch (_e) {}
