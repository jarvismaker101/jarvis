// lib/dom_inventory.mjs
//
// Pure, dependency-free DOM inventory helpers for the brave-control MCP
// server.
//
// ── G6 (audit F39/F40): two complementary element-identity layers ──
//
// 1. `buildCssPath` — a human-debuggable CSS path for overnight triage
//    (same role as the browser-agent's look-side cssPath).
// 2. **Backend IDs** (`record.bid`) — the durable, cross-frame handle:
//    `DOM.getDocument` (depth -1) + `DOM.querySelectorAll` walk returns real
//    `backendNodeId`s for EVERY element that has one (including shadow/iframe
//    content). `observe( bid )` asks CDP: `describeNode` (url/frame/visible +
//    contentDocument children) + `resolveNode` + `scrollIntoViewIfNeeded`
//    + `BoxModel.getBoxModel`. All `*_element` tools FIRST re-resolve this
//    backend id and verify actionability/OCCLUSION before issuing real
//    Playwright `mouse` input (F40: no synthetic top-document events).
//
// IMPORTANT: this file is importable BOTH by Node (server.mjs) and by the
// browser-agent's Python side via JsBridge-style evaluate. Every function
// that runs inside page.evaluate must reference ONLY page globals
// (document, getComputedStyle, getBoundingClientRect) — never Node scope.
//
// `domWalk` — designed to run INSIDE page.evaluate: it references nothing
// outside the page (document, getComputedStyle, getBoundingClientRect) so it
// can be passed to evaluate() as a serialized function with no closures over
// Node scope. `formatInventory` and `buildCssPath` are plain functions the
// server side (or tests) can call directly.

export const INTERACTIVE_SELECTOR = [
  "a[href]",
  "button",
  "input",
  "select",
  "textarea",
  "[role]",
  "[onclick]",
  "details",
  "summary",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "main",
  "nav",
  "header",
  "footer",
  "aside",
].join(", ")

const LANDMARK_SELECTOR = [
  "main",
  "nav",
  "header",
  "footer",
  "aside",
  "[role=main]",
  "[role=navigation]",
  "[role=banner]",
  "[role=contentinfo]",
  "[role=complementary]",
  "[role=region]",
].join(", ")

const TEXT_CAP = 80

export function escapeCssIdent(ident) {
  // CSS.escape may be missing in evaluate contexts; escape the characters
  // that would break a querySelector (ids with word chars pass through).
  return String(ident).replace(/[^a-zA-Z0-9_-]/g, (ch) => "\\" + ch)
}

// ── Locator building (used inside the walk AND reusable server-side) ──

export function buildCssPath(el) {
  // Stable, unique-enough CSS path: climbs the tree pushing tag:nth-of-type
  // segments and stops at the first id-bearing ancestor (short + robust).
  const segments = []
  let cur = el
  while (cur && cur.nodeType === 1) {
    const tag = cur.tagName.toLowerCase()
    const id = cur.getAttribute && cur.getAttribute("id")
    if (id) {
      segments.push(tag + "#" + escapeCssIdent(id))
      break
    }
    const parent = cur.parentElement
    let nth = 1
    if (parent && parent.children) {
      for (const sib of parent.children) {
        if (sib === cur) break
        if (sib.tagName && sib.tagName.toLowerCase() === tag) nth++
      }
    }
    segments.push(tag + ":nth-of-type(" + nth + ")")
    if (!parent) break
    cur = parent
  }
  return segments.length ? segments.join(" > ") : (el.tagName || "div").toLowerCase()
}

// ── Visibility / structural helpers (page-side) ──

function isVisible(el) {
  if (el.getAttribute("hidden") != null) return false
  if (el.getAttribute("aria-hidden") === "true") return false
  if (el.tagName.toLowerCase() === "input" && el.type === "hidden") return false
  const rect = typeof el.getBoundingClientRect === "function" ? el.getBoundingClientRect() : null
  if (rect && (rect.width <= 0 || rect.height <= 0)) return false
  // getComputedStyle exists in the browser; absent in fake-DOM tests.
  if (typeof getComputedStyle === "function") {
    try {
      const style = getComputedStyle(el)
      if (style.display === "none" || style.visibility === "hidden") return false
    } catch {}
  }
  return true
}

function insideClosedDetails(el) {
  let cur = el.parentElement
  while (cur) {
    const tag = cur.tagName && cur.tagName.toLowerCase()
    if (tag === "details" && !cur.open) return true
    cur = cur.parentElement
  }
  return false
}

function inferRole(el, tag) {
  const explicit = el.getAttribute("role")
  if (explicit) return explicit
  switch (tag) {
    case "button":
      return "button"
    case "a":
      return el.getAttribute("href") ? "link" : null
    case "input":
      switch ((el.type || "text").toLowerCase()) {
        case "checkbox": return "checkbox"
        case "radio": return "radio"
        case "submit": return "button"
        case "button": return "button"
        case "search": return "searchbox"
        default: return "textbox"
      }
    case "select":
      return "combobox"
    case "textarea":
      return "textbox"
    case "details":
      return "group"
    case "summary":
      return "button"
    case "h1": case "h2": case "h3": case "h4": case "h5": case "h6":
      return "heading"
    default:
      return null
  }
}

function trimText(text, max = TEXT_CAP) {
  text = (text || "").replace(/\s+/g, " ").trim()
  if (text.length > max) return text.slice(0, max - 1) + "…"
  return text
}

function sectionOf(el) {
  const hit = el.closest(LANDMARK_SELECTOR)
  if (!hit) return null
  const tag = hit.tagName.toLowerCase()
  const label = hit.getAttribute("aria-label") || hit.getAttribute("title") || ""
  return label ? tag + " (" + label + ")" : tag
}

// ── The walk (page-side) ──
//
// IMPORTANT: this function is passed DIRECTLY to Playwright's
// page.evaluate(), which serializes it via toString() and executes it in the
// page with NO arguments. It must therefore be FULLY self-contained:
//   - `doc = document` default (evaluate calls it with zero args)
//   - every constant and helper INLINED — no references to anything in this
//     module's scope (they would be undefined after serialization)
// The module-level helpers above stay for server-side/tests; the
// sync test in test/dom_inventory.test.mjs keeps the inlined copies aligned.

export function domWalk(doc = document) {
  const INTERACTIVE_SELECTOR = [
    "a[href]",
    "button",
    "input",
    "select",
    "textarea",
    "[role]",
    "[onclick]",
    "details",
    "summary",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "main",
    "nav",
    "header",
    "footer",
    "aside",
  ].join(", ")
  const LANDMARK_SELECTOR = [
    "main",
    "nav",
    "header",
    "footer",
    "aside",
    "[role=main]",
    "[role=navigation]",
    "[role=banner]",
    "[role=contentinfo]",
    "[role=complementary]",
    "[role=region]",
  ].join(", ")
  const TEXT_CAP = 80
  const escapeCssIdent = (ident) =>
    String(ident).replace(/[^a-zA-Z0-9_-]/g, (ch) => "\\" + ch)
  const buildCssPath = (el) => {
    const segments = []
    let cur = el
    while (cur && cur.nodeType === 1) {
      const tag = cur.tagName.toLowerCase()
      const id = cur.getAttribute && cur.getAttribute("id")
      if (id) {
        segments.push(tag + "#" + escapeCssIdent(id))
        break
      }
      const parent = cur.parentElement
      let nth = 1
      if (parent && parent.children) {
        for (const sib of parent.children) {
          if (sib === cur) break
          if (sib.tagName && sib.tagName.toLowerCase() === tag) nth++
        }
      }
      segments.push(tag + ":nth-of-type(" + nth + ")")
      if (!parent) break
      cur = parent
    }
    return segments.length ? segments.join(" > ") : (el.tagName || "div").toLowerCase()
  }
  const isVisible = (el) => {
    if (el.getAttribute("hidden") != null) return false
    if (el.getAttribute("aria-hidden") === "true") return false
    if (el.tagName.toLowerCase() === "input" && el.type === "hidden") return false
    const rect = typeof el.getBoundingClientRect === "function" ? el.getBoundingClientRect() : null
    if (rect && (rect.width <= 0 || rect.height <= 0)) return false
    if (typeof getComputedStyle === "function") {
      try {
        const style = getComputedStyle(el)
        if (style.display === "none" || style.visibility === "hidden") return false
      } catch {}
    }
    return true
  }
  const insideClosedDetails = (el) => {
    let cur = el.parentElement
    while (cur) {
      const tag = cur.tagName && cur.tagName.toLowerCase()
      if (tag === "details" && !cur.open) return true
      cur = cur.parentElement
    }
    return false
  }
  const inferRole = (el, tag) => {
    const explicit = el.getAttribute("role")
    if (explicit) return explicit
    switch (tag) {
      case "button":
        return "button"
      case "a":
        return el.getAttribute("href") ? "link" : null
      case "input":
        switch ((el.type || "text").toLowerCase()) {
          case "checkbox": return "checkbox"
          case "radio": return "radio"
          case "submit": return "button"
          case "button": return "button"
          case "search": return "searchbox"
          default: return "textbox"
        }
      case "select":
        return "combobox"
      case "textarea":
        return "textbox"
      case "details":
        return "group"
      case "summary":
        return "button"
      case "h1": case "h2": case "h3": case "h4": case "h5": case "h6":
        return "heading"
      default:
        return null
    }
  }
  const trimText = (text, max = TEXT_CAP) => {
    text = (text || "").replace(/\s+/g, " ").trim()
    if (text.length > max) return text.slice(0, max - 1) + "…"
    return text
  }
  const sectionOf = (el) => {
    const hit = el.closest(LANDMARK_SELECTOR)
    if (!hit) return null
    const tag = hit.tagName.toLowerCase()
    const label = hit.getAttribute("aria-label") || hit.getAttribute("title") || ""
    return label ? tag + " (" + label + ")" : tag
  }

  const records = []
  const seen = new Set()
  for (const el of doc.querySelectorAll(INTERACTIVE_SELECTOR)) {
    if (seen.has(el)) continue
    seen.add(el)
    if (!isVisible(el)) continue
    if (insideClosedDetails(el)) continue
    const tag = el.tagName.toLowerCase()
    const role = inferRole(el, tag)
    const text = trimText(el.textContent)
    const placeholder = el.getAttribute("placeholder") || ""
    const ariaLabel = el.getAttribute("aria-label") || el.getAttribute("title") || ""
    const name = el.getAttribute("name") || ""
    const id = el.getAttribute("id") || ""
    const href = el.getAttribute("href") || ""
    const type = (el.type || "") || el.getAttribute("type") || ""
    // Skip empty records: nothing the model can act on or read.
    if (!text && !placeholder && !ariaLabel && !name && !id && !href) continue
    records.push({
      index: records.length,
      tag,
      role,
      text,
      placeholder,
      ariaLabel,
      name,
      id,
      href,
      type,
      locator: buildCssPath(el),
      section: sectionOf(el) || "page",
    })
  }
  return records
}

// ── Inventory formatting (server-side) ──

export function formatRecord(r) {
  const bits = ["[" + r.index + "] " + r.tag]
  if (r.id) bits.push("#" + r.id)
  if (r.role) bits.push(r.role)
  if (r.type) bits.push("type=" + r.type)
  const label = r.text || r.placeholder || r.ariaLabel || r.name || r.id
  if (label) bits.push('"' + label.slice(0, 60) + '"')
  if (r.href) bits.push("→ " + r.href.slice(0, 60))
  return bits.join(" ")
}

export function formatInventory(records) {
  if (!records || !records.length) {
    return "No interactive elements found on this page."
  }
  const groups = new Map()
  for (const r of records) {
    const key = r.section || "page"
    if (!groups.has(key)) groups.set(key, [])
    groups.get(key).push(r)
  }
  const parts = []
  for (const [section, items] of groups) {
    parts.push("[" + String(section).toUpperCase() + "]")
    for (const r of items) parts.push("  " + formatRecord(r))
  }
  parts.push("")
  parts.push(
    "Act by index with click_element(<index>) or fill_element(<index>, <value>). " +
    "Re-run understand_page after the page changes to refresh indices."
  )
  return parts.join("\n")
}
