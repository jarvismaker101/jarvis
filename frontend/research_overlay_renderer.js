/**
 * Jarvis Research - glass overlay renderer.
 *
 * Renders the detailed research report (markdown) from data/research_reports/
 * latest.json (self-polled, backend-independent) or via IPC from main.js.
 * Lasts until the user closes it - there is NO auto-dismiss timer.
 * YouTube "related videos" render as clickable chips (yt.url).
 */

const IPC = window.jarvisAPI || { send() {}, on() {}, openExternal() {}, backend: async () => ({ status: 0 }), readReportMirror: async () => null, getConfig: async () => ({}), getLocalSecret: async () => "" };

let shown = false;
let lastRenderedId = 0;
// Reports carry id = completion timestamp (ms). Anything generated BEFORE this
// renderer started is last-session history — never auto-show it on boot.
const bootTimeMs = Date.now();

const panel   = document.getElementById("glass-panel");
const titleEl = document.getElementById("r-title");
const queryEl = document.getElementById("r-query");
const bodyEl  = document.getElementById("r-body");
const videoBox = document.getElementById("r-videos");
const videoList = document.getElementById("r-video-list");
const statusEl = document.getElementById("r-status");
const liveBox = document.getElementById("r-live");
const liveMsg = document.getElementById("r-live-msg");
const liveEvidence = document.getElementById("r-live-evidence");

// F28 — live research progress (showSourcesSoFar) is rendered only for events
// newer than the last rendered report, so an old finished run never re-pops.
let lastProgressId = 0;

// F48 — provenance words, mirrored from backend/services/provenance.py so the
// overlay says exactly what the backend meant (observed / quoted / inferred /
// secondary / externally_checked).
const PROVENANCE_BADGES = {
  observed: "observed",
  quoted: "quoted from source",
  inferred: "inferred",
  secondary: "AI summary · secondary",
  externally_checked: "externally checked",
};

let pollCount = 0;

function setStatus(text, kind) {
  if (!statusEl) return;
  const ts = new Date().toLocaleTimeString();
  statusEl.textContent = "[" + ts + "] " + text;
  statusEl.className = "r-status" + (kind ? " " + kind : "");
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Minimal, safe markdown -> HTML (headers, bullets, paragraphs, bold, code, links)
function mdToHtml(md) {
  const lines = String(md || "").split("\n");
  const out = [];
  let inList = false;
  const closeList = () => {
    if (inList) { out.push("</ul>"); inList = false; }
  };

  for (let raw of lines) {
    const line = raw.trim();
    if (!line) { closeList(); continue; }

    // headings
    const h = line.match(/^(#{1,3})\s+(.*)$/);
    if (h) {
      closeList();
      out.push(`<h2>${inline(h[2])}</h2>`);
      continue;
    }
    // bullets
    if (line.startsWith("- ") || line.startsWith("* ")) {
      if (!inList) { out.push("<ul>"); inList = true; }
      out.push(`<li>${inline(line.slice(2))}</li>`);
      continue;
    }
    // numbered-ish
    if (/^\d+\.\s+/.test(line)) {
      closeList();
      out.push(`<p>${inline(line)}</p>`);
      continue;
    }
    closeList();
    out.push(`<p>${inline(line)}</p>`);
  }
  closeList();
  return out.join("");
}

function inline(text) {
  let t = escapeHtml(text);
  const anchors = [];
  const capture = (html) => {
    const token = "\u0000L" + anchors.length + "\u0000";
    anchors.push(html);
    return token;
  };
  // markdown links [label](url) - H9: only http(s)/mailto schemes may become
  // anchors; javascript:/data:/vbscript: hrefs render as plain label text
  // (escapeHtml does not touch schemes, so one poisoned report link could
  // otherwise run script in the overlay).
  t = t.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (m, label, url) => {
    const scheme = String(url).trim().toLowerCase();
    if (!/^(https?:\/\/|mailto:)/.test(scheme)) {
      return label;
    }
    return capture(`<a href="${escapeHtml(url)}" data-ext="1" target="_blank">${label}</a>`);
  });
  // bare URLs -> clickable (keys in "Source: https://..." / "URL:" lines)
  t = t.replace(/(https?:\/\/[^\s<>]+)/g, (m) => {
    const clean = m.replace(/[),.;!?]+$/, "");
    const href = clean.replace(/"/g, "%22"); // text is already escaped; guard the attribute
    return capture(`<a href="${href}" data-ext="1" target="_blank">${clean}</a>`);
  });
  t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  t = t.replace(/`([^`]+)`/g, "<code>$1</code>");
  t = t.replace(/\u0000L(\d+)\u0000/g, (m, i) => anchors[Number(i)]);
  return t;
}

function show(data) {
  if (!data || !data.id) return;
  const query = data.query || "research";
  lastRenderedId = data.id;
  titleEl.textContent = "Research";
  queryEl.textContent = query;
  hideLiveProgress();
  let html = "";
  if (data.markdown) {
    try {
      html = mdToHtml(data.markdown);
    } catch (e) {
      // Never leave a dead "waiting" placeholder - show raw text if md fails.
      html = '<p class="r-placeholder">' + escapeHtml(data.markdown) + "</p>";
      console.error("mdToHtml failed:", e);
    }
  } else {
    html = '<p class="r-placeholder">The report was empty.</p>';
  }
  bodyEl.innerHTML = html;
  setStatus("rendered report id=" + data.id + " | q=" + query, "ok");

  // Related videos as clickable chips
  const videos = data.videos || [];
  if (videos.length) {
    videoList.innerHTML = "";
    videos.forEach((v) => {
      const chip = document.createElement("div");
      chip.className = "video-chip";
      chip.title = v.url || "";
      chip.innerHTML = `<span class="vp">PLAY</span><span class="vt">${escapeHtml(v.title || "YouTube video")}</span>`;
      chip.addEventListener("click", () => {
        try { IPC.openExternal(v.url); } catch (_e) {}
      });
      videoList.appendChild(chip);
    });
    videoBox.classList.remove("hidden");
  } else {
    videoBox.classList.add("hidden");
  }

  panel.classList.remove("hidden");
  shown = true;
}

// click delegation for links inside the report
bodyEl.addEventListener("click", (e) => {
  const a = e.target.closest("a[data-ext]");
  if (!a) return;
  e.preventDefault();
  try { IPC.openExternal(a.getAttribute("href")); } catch (_e) {}
});

// ── F28: live progress + incremental evidence ─────────────────────────────
// brain.push_research_progress POSTs each stage + the sources gathered so far
// to /research-progress. The overlay polls it (same cadence as the report
// poll) so a deep run is observable instead of a dark "waiting" panel. The
// final report, when it lands, replaces this panel entirely (show() hides it).
function hideLiveProgress() {
  if (liveBox) liveBox.classList.add("hidden");
}

function renderResearchProgress(data) {
  if (!liveBox || !liveMsg || !liveEvidence) return;
  if (!data || !data.id) return;
  // Stale / pre-boot-relative again — only a live run renders here.
  if (data.id <= bootTimeMs || data.id <= lastRenderedId) return;
  if (data.id === lastProgressId) return;
  lastProgressId = data.id;

  if (data.message) {
    liveMsg.textContent = data.message;
  }
  const evidence = Array.isArray(data.evidence) ? data.evidence : [];
  const head = liveBox.querySelector(".r-live-head");
  if (head) head.style.display = evidence.length ? "" : "none";
  liveEvidence.innerHTML = "";
  evidence.slice(0, 12).forEach((it) => {
    if (!it || typeof it !== "object") return;
    const prov = PROVENANCE_BADGES[it.provenance] ? it.provenance : "observed";
    const badge = PROVENANCE_BADGES[prov] || "observed";
    const title = String(it.result_title || it.title || it.source || "Source").slice(0, 160);
    const snippet = String(it.summary || it.text_excerpt || it.snippet || "").slice(0, 220);
    const url = it.source_url || it.url || "";
    const item = document.createElement("div");
    item.className = "r-le-item";
    const badgeEl = document.createElement("span");
    badgeEl.className = "r-le-badge b-" + prov;
    badgeEl.textContent = badge;
    const body = document.createElement("div");
    body.className = "r-le-body";
    const itemTitle = document.createElement("span");
    itemTitle.className = "r-le-title";
    itemTitle.textContent = title;
    body.appendChild(itemTitle);
    if (url) {
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("data-ext", "1");
      link.target = "_blank";
      link.textContent = truncateUrl(url);
      body.appendChild(document.createTextNode(" — "));
      body.appendChild(link);
    }
    if (snippet) {
      const snip = document.createElement("span");
      snip.className = "r-le-snippet";
      snip.textContent = snippet;
      body.appendChild(snip);
    }
    item.appendChild(badgeEl);
    item.appendChild(body);
    liveEvidence.appendChild(item);
  });
  liveBox.classList.remove("hidden");
}

function truncateUrl(url) {
  const clean = String(url).replace(/^https?:\/\//, "");
  return clean.length > 70 ? clean.slice(0, 67) + "…" : clean;
}

async function pollResearchProgress() {
  try {
    const result = await IPC.backend("/research-progress");
    if (result && result.status && result.status < 400) {
      try {
        renderResearchProgress(JSON.parse(result.text || "{}"));
      } catch (_e) { /* stale backend / bad json */ }
    }
    // The final report may have landed while we were polling progress —
    // check that too so the swap is prompt.
    pollLatestReport();
  } catch (_e) { /* no backend */ }
}

document.getElementById("r-min").addEventListener("click", () => {
  IPC.send("minimize-research-overlay");
});

document.getElementById("r-close").addEventListener("click", () => {
  IPC.send("dismiss-research-overlay");
});

// The renderer reads the report mirror THROUGH THE PRELOAD BRIDGE (F51: no
// fs in the renderer) so the report never depends on main.js being up to
// date - a stale Electron main can no longer leave the overlay stuck on
// "waiting for research report".
async function pollLatestReport() {
  pollCount++;
  const status = [];
  let data = null;
  let fileOk = false;
  try {
    data = await IPC.readReportMirror();
    if (data != null) {
      fileOk = true;
      status.push("file:" + (data && data.id ? data.id : "empty"));
    } else {
      status.push("file:missing");
    }
  } catch (e) {
    status.push("file:ERR " + (e && e.message ? e.message.slice(0, 40) : e));
  }
  if (data && data.id && data.id > bootTimeMs && data.id !== lastRenderedId) {
    try {
      show(data);
      setStatus((fileOk ? "FOUND new report -> rendered. " : "") + status.join(" ") + " | polls=" + pollCount, "ok");
    } catch (e) {
      setStatus("render FAILED: " + (e && e.message ? e.message.slice(0, 80) : e), "err");
    }
    try { IPC.send("request-show-research-window"); } catch (_e) {}
    return;
  }
  // fallback to the backend API as a second source (proxied, token attached)
  try {
    const result = await IPC.backend("/research-result");
    if (result && result.status && result.status < 400) {
      try {
        const d = JSON.parse(result.text || "{}");
        if (d && d.id && d.id > bootTimeMs && d.id !== lastRenderedId) {
          show(d);
          setStatus("API found report id=" + d.id, "ok");
          return;
        }
        setStatus(status.join(" ") + " | api:stale/" + result.status + " | polls=" + pollCount);
      } catch (_e) {
        setStatus(status.join(" ") + " | api:badjson /" + result.status, "err");
      }
    } else {
      setStatus(status.join(" ") + " | api:unreachable", "err");
    }
  } catch (_e) {
    setStatus(status.join(" ") + " | api:ERR", "err");
  }
}
setInterval(function pollTick() {
  pollResearchProgress();
  pollLatestReport();
}, 2000);
pollResearchProgress();
pollLatestReport();
setStatus("renderer v6 loaded - polling...");

// click delegation for live-evidence source links (same open-external rule)
liveEvidence.addEventListener("click", (e) => {
  const a = e.target.closest("a[data-ext]");
  if (!a) return;
  e.preventDefault();
  try { IPC.openExternal(a.getAttribute("href")); } catch (_e) {}
});

// Freshness is judged by main.js (reports older than the session window are
// never pushed); here we only dedupe. Gating on bootTimeMs would wrongly
// reject a report that was generated right before this window mounted.
// Payload-only, like every bridge push (see overlay_renderer.js): the bridge
// does not forward the IpcRendererEvent. The 2s poll below is what kept this
// window working while the push path was silently dead.
IPC.on("show-research-result", (data) => {
  if (data && data.id && data.id !== lastRenderedId) {
    show(data);
  }
});
