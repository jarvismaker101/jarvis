/**
 * Jarvis Screen Q&A — Overlay Renderer
 *
 * Receives screen-answer data from the main process via IPC
 * and renders it in the glassmorphism overlay.  Auto-dismisses
 * after a timeout.
 */

// G11 / F51 — sandboxed renderer: the ONLY privileged surface is the
// preload bridge (validated channels, validated openExternal).
const IPC = window.jarvisAPI || {
  send() {}, on() {}, openExternal() {},
  backend: async () => ({ status: 0, error: "no preload bridge" }),
  readReportMirror: async () => null,
  getConfig: async () => ({ backendPort: "9999" }),
  getLocalSecret: async () => "",
};

const DISMISS_AFTER_MS = 14000;

let dismissTimer = null;
let dismissAnimationTimer = null;
let isVisible = false;

// ── DOM refs ────────────────────────────────────────
const overlay = document.getElementById("overlay");
const tipText = document.getElementById("tip-text");
const tipProgress = document.getElementById("tip-progress");
const linksCard = document.getElementById("links-card");
const linksItems = document.getElementById("links-items");
const evidenceCard = document.getElementById("evidence-card");
const evidenceItems = document.getElementById("evidence-items");
const regionBadge = document.getElementById("region-badge");

// ── Show overlay with data ──────────────────────────
function showOverlay(data) {
  // Clear previous dismiss timer
  if (dismissTimer) {
    clearTimeout(dismissTimer);
    dismissTimer = null;
  }
  if (dismissAnimationTimer) {
    clearTimeout(dismissAnimationTimer);
    dismissAnimationTimer = null;
  }

  // Region focus chip — "highlighted area" questions
  const region = data.region || {};
  if (region && (region.cursor_x != null || region.width > 0)) {
    regionBadge.textContent = "📍 Focusing on the area you pointed at";
    regionBadge.classList.remove("hidden");
  } else {
    regionBadge.classList.add("hidden");
  }

  // Populate TIP
  tipText.textContent = data.tip || "I couldn't determine what's on your screen.";

  // Populate LINKS
  const links = data.links || [];
  linksItems.innerHTML = "";

  if (links.length > 0) {
    linksCard.classList.remove("hidden");
    links.forEach((link) => {
      const pill = document.createElement("div");
      pill.className = "link-pill";
      pill.title = link.url || "";
      pill.innerHTML = `
        <span class="link-pill-icon">${escapeHtml(link.icon || "🔗")}</span>
        <span class="link-pill-label">${escapeHtml(link.label || "Link")}</span>
      `;
      pill.addEventListener("click", (e) => {
        e.stopPropagation();
        if (link.url) {
          IPC.openExternal(link.url);
        }
      });
      linksItems.appendChild(pill);
    });
  } else {
    linksCard.classList.add("hidden");
  }

  // Populate EVIDENCE
  const evidence = data.evidence || [];
  evidenceItems.innerHTML = "";

  if (evidence.length > 0) {
    evidenceCard.classList.remove("hidden");
    evidence.forEach((item) => {
      const el = document.createElement("div");
      el.className = "evidence-item";
      el.innerHTML = `
        <div class="evidence-source">${escapeHtml(item.source || "")}</div>
        <div class="evidence-title">${escapeHtml(item.title || "")}</div>
        <div class="evidence-snippet">${escapeHtml(item.snippet || "")}</div>
      `;
      evidenceItems.appendChild(el);
    });
  } else {
    evidenceCard.classList.add("hidden");
  }

  // Reset progress bar animation
  tipProgress.style.animation = "none";
  tipProgress.offsetHeight; // force reflow
  tipProgress.style.setProperty("--dismiss-duration", `${DISMISS_AFTER_MS}ms`);
  tipProgress.style.animation = "";

  // Trigger entrance
  overlay.classList.remove("dismissing");
  overlay.classList.add("visible");
  isVisible = true;

  // Auto-dismiss timer
  dismissTimer = setTimeout(() => {
    dismissOverlay();
  }, DISMISS_AFTER_MS);
}

// ── Dismiss overlay ─────────────────────────────────
function dismissOverlay() {
  if (!isVisible) return;

  overlay.classList.add("dismissing");
  isVisible = false;

  if (dismissTimer) {
    clearTimeout(dismissTimer);
    dismissTimer = null;
  }

  // After animation completes, hide and tell main process
  dismissAnimationTimer = setTimeout(() => {
    overlay.classList.remove("visible", "dismissing");
    dismissAnimationTimer = null;
    IPC.send("dismiss-overlay");
  }, 400);
}

// Make dismiss available globally (called from onclick)
window.dismissOverlay = dismissOverlay;

// ── Escape HTML for safety ──────────────────────────
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// ── Event delegation for mouse enter/leave on cards ─
// Uses event delegation so dynamically created evidence
// cards and link pills are handled without re-attaching.
// ── IPC: Main process pushes data directly ──────────
// The preload bridge delivers ONLY the payload (it never hands renderer
// content the raw IpcRendererEvent, whose sender would be an unrestricted
// send channel). Registering `(_event, data)` here therefore received
// `data === undefined` and every push was swallowed by the bridge's
// try/catch — the screen overlay stayed empty while the answer was spoken.
IPC.on("show-screen-answer", (data) => {
  showOverlay(data);
});
