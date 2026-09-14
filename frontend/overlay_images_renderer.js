/**
 * Jarvis Screen Q&A — Image Overlay Renderer (Left Side)
 *
 * Receives image data from the main process via IPC and renders
 * Wikipedia topic images in glassmorphism cards.
 */

const IPC = window.jarvisAPI || { send() {}, on() {}, openExternal() {}, backend: async () => ({ status: 0 }), readReportMirror: async () => null, getConfig: async () => ({}), getLocalSecret: async () => "" };

const DISMISS_AFTER_MS = 14000;

let dismissTimer = null;
let dismissAnimationTimer = null;
let isVisible = false;

const overlay = document.getElementById("image-overlay");

// ── Show images ─────────────────────────────────────
function showImages(images) {
  if (dismissTimer) {
    clearTimeout(dismissTimer);
    dismissTimer = null;
  }
  if (dismissAnimationTimer) {
    clearTimeout(dismissAnimationTimer);
    dismissAnimationTimer = null;
  }

  overlay.innerHTML = "";

  if (!images || images.length === 0) {
    IPC.send("dismiss-image-overlay");
    return;
  }

  images.forEach((img, idx) => {
    const card = document.createElement("div");
    card.className = "glass-card image-card";
    card.style.animationDelay = `${0.1 + idx * 0.15}s`;

    card.innerHTML = `
      <img
        src="${escapeHtml(img.url)}"
        alt="${escapeHtml(img.title || "")}"
        onerror="this.parentElement.style.display='none'"
      />
      <div class="image-card-body">
        <div class="image-card-badge">📸 WIKIPEDIA</div>
        <div class="image-card-title">${escapeHtml(img.title || "")}</div>
        <div class="image-card-caption">${escapeHtml(img.caption || "")}</div>
      </div>
    `;

    overlay.appendChild(card);
  });

  // Trigger entrance
  overlay.classList.remove("dismissing");
  overlay.classList.add("visible");
  isVisible = true;

  // Auto-dismiss
  dismissTimer = setTimeout(() => {
    dismissOverlay();
  }, DISMISS_AFTER_MS);
}

// ── Dismiss ─────────────────────────────────────────
function dismissOverlay() {
  if (!isVisible) return;

  overlay.classList.add("dismissing");
  isVisible = false;

  if (dismissTimer) {
    clearTimeout(dismissTimer);
    dismissTimer = null;
  }

  dismissAnimationTimer = setTimeout(() => {
    overlay.classList.remove("visible", "dismissing");
    dismissAnimationTimer = null;
    IPC.send("dismiss-image-overlay");
  }, 400);
}

window.dismissOverlay = dismissOverlay;

// ── Escape HTML ─────────────────────────────────────
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// ── Event delegation for mouse enter/leave ──────────
// Click cards to dismiss
overlay.addEventListener("click", (e) => {
  if (e.target.closest(".glass-card")) {
    dismissOverlay();
  }
});

// ── IPC: Receive images from main process ───────────
// Payload-only, like every bridge push (see overlay_renderer.js): the bridge
// does not forward the IpcRendererEvent.
IPC.on("show-screen-images", (images) => {
  showImages(images);
});
