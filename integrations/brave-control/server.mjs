import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js"
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js"
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js"
import { z } from "zod"
import { chromium } from "playwright-core"
import { randomUUID } from "crypto"
import fs from "fs"
import http from "http"
import os from "os"
import path from "path"
import { domWalk, formatInventory } from "./lib/dom_inventory.mjs"
import { isClosedError } from "./lib/retry.mjs"
import { listDir, readFileText } from "./lib/fs_tools.mjs"
import { formatTabs, buildTabInfos } from "./lib/tab_tools.mjs"

const BRAVE_PATH = "C:\\Users\\mayan\\AppData\\Local\\BraveSoftware\\Brave-Browser\\Application\\brave.exe"
const PROFILE_DIR = path.join(process.env.USERPROFILE || os.homedir(), ".brave-mcp-profile")

// Transport mode: stdio is the default (spawned per opencode session);
// BRAVE_MCP_MODE=http runs a persistent daemon (one browser per jarvis boot)
// that opencode reaches via a remote MCP registration instead.
const HTTP_MODE = process.env.BRAVE_MCP_MODE === "http"
const HTTP_PORT = Number(process.env.BRAVE_MCP_PORT || "9570")
const HTTP_TOKEN = process.env.BRAVE_MCP_TOKEN || ""
const IDLE_MIN = Number(process.env.BRAVE_MCP_IDLE_MIN || "30")

let browser = null
let context = null
let page = null

// understand_page cache: index -> CSS locator, replaced wholesale per call.
let domCache = null

// HTTP-mode idle browser recycle: close browser/context after IDLE_MIN of
// tool inactivity, keep the daemon + transport alive.
let idleTimer = null

function touchIdleTimer() {
  if (!HTTP_MODE) return
  if (idleTimer) clearTimeout(idleTimer)
  idleTimer = setTimeout(async () => {
    if (browser) {
      try {
        await browser.close()
      } catch {}
      browser = null
      context = null
      page = null
      domCache = null
    }
  }, IDLE_MIN * 60 * 1000)
}

function staleMessage() {
  return {
    content: [
      {
        type: "text",
        text: "That index is stale or the page changed. Run understand_page again to refresh the inventory.",
      },
    ],
  }
}

// ── G6 (audit F40): real-input primitives shared by every action tool ──
//
// F40 bans synthetic top-document JavaScript clicks. Every interaction tool
// below resolves its element to a Playwright locator (optionally inside a
// sub-frame via `frame`) and then uses the REAL input path (`click`, `fill`,
// `pressSequentially`, `mouse`, …) — with actionability checks FIRST and the
// settle race AFTER.
//
// `frame` is a CSS selector for the frame's <iframe>/<frame> element in the
// TOP document. A falsy value means "act in the top document". Sub-frame
// resolution goes through `frameLocator(frame).locator(css)` — the
// cross-origin-safe handle the old top-document JS approach could never
// touch.
// ── G6 (audit F13 + F40): real-input daemon primitives ──
// A CSS selector resolved to a Playwright locator is a REAL target handle
// (actionability checks + real mouse/keyboard), unlike a coordinate
// evaluated to a top-document elementFromPoint. Every tool takes an
// optional `frame` (CSS selector of the iframe/frame in the top document;
// empty = top document) and probes the target BEFORE acting, so a gone
// element is reported as such. Returns real after-state: url + navigated +
// title — never a bare "clicked: true" after swallowing the error.

function scoped(page, frame, css) {
  const target = (frame && String(frame).trim()) || ""
  if (!target) return { scope: page.locator(css), where: "top" }
  return { scope: page.frameLocator(target).locator(css), where: "frame " + target }
}

// One evaluate that answers everything an action tool needs to know before it
// acts: current URL, whether the element still resolves, whether it is
// visible, and its viewport rect. Frame-aware: an empty `frame` means top
// document; otherwise the target must live inside that sub-frame.
async function probeTarget(page, frame, css) {
  return page.evaluate(
    ({ cssArg, frameArg }) => {
      if (!frameArg) {
        let el = null
        try {
          el = document.querySelector(cssArg)
        } catch {
          return { url: location.href, exists: false }
        }
        if (!el) return { url: location.href, exists: false }
        const rect = el.getBoundingClientRect()
        const style = getComputedStyle(el)
        return {
          url: location.href,
          exists: true,
          visible: rect.width > 0 && rect.height > 0 &&
            style.display !== "none" && style.visibility !== "hidden",
          rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
        }
      }
      // Sub-frame: reach the frame's document through the frame element (same
      // rules as the Playwright frameLocator — cross-origin frames are
      // observable only via the automation connection, never via page JS).
      let frameEl = null
      try {
        frameEl = document.querySelector(frameArg)
      } catch {
        return { url: location.href, exists: false, frameMissing: true }
      }
      const doc = (frameEl && frameEl.contentDocument) || null
      if (!doc) return { url: location.href, exists: false, crossOrigin: true }
      let el = null
      try {
        el = doc.querySelector(cssArg)
      } catch {
        return { url: location.href, exists: false }
      }
      if (!el) return { url: location.href, exists: false }
      const rect = el.getBoundingClientRect()
      return {
        url: location.href,
        frameUrl: doc.location ? doc.location.href : "",
        exists: true,
        visible: rect.width > 0 && rect.height > 0,
        rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
      }
    },
    { cssArg: css, frameArg: (frame && String(frame).trim()) || "" }
  )
}

// After-state returned by every action tool: the URL before and after the
// action plus the page title, so the caller (and the broker) can tell whether
// the action navigated — F40 wants real outcomes, not bare "clicked: true".
function afterState(beforeUrl, page, extra) {
  const out = {
    before: beforeUrl,
    url: page.url(),
    navigated: page.url() !== beforeUrl,
  }
  if (extra && typeof extra === "object") Object.assign(out, extra)
  return out
}

function afterText(state) {
  const base = `url=${state.url} navigated=${state.navigated ? "yes" : "no"}`
  return state.title ? `${base}\ntitle=${state.title}` : base
}

// ── G6 (F39): screenshot geometry reported with every look ──
// click_point receives look-image coordinates and converts them with the
// session's look_scale; the daemon is the source of truth for the frame
// those coordinates live in (real viewport pixels), so a resized look
// overlay can never drift from the click target.
async function viewportGeometry(page) {
  try {
    return await page.evaluate(() => ({
      width: window.innerWidth || 0,
      height: window.innerHeight || 0,
      deviceScaleFactor: window.devicePixelRatio || 1,
    }))
  } catch {
    return { width: 0, height: 0, deviceScaleFactor: 1 }
  }
}

async function launchBrowser() {
  // The old handle may be dead (user closed the window, crash, OOM); closing
  // it again is harmless and best-effort, and resetting every var guarantees
  // the next getPage starts from a clean slate.
  if (browser) {
    try {
      await browser.close()
    } catch {}
  }
  browser = null
  context = null
  page = null
  domCache = null
  fs.mkdirSync(PROFILE_DIR, { recursive: true })
  browser = await chromium.launchPersistentContext(PROFILE_DIR, {
    executablePath: BRAVE_PATH,
    headless: false,
    viewport: null,
    args: ["--start-maximized"],
  })
  context = browser
  page = context.pages()[0] || await context.newPage()
  page.setDefaultTimeout(30000)
}

async function getPage() {
  // A CLOSED browser keeps the var truthy, so check the page handle itself:
  // pages report closed when the browser dies. This is the recovery signal
  // for non-recycle deaths (user closing the headed window, crash, OOM).
  if (!browser || !page || page.isClosed()) {
    await launchBrowser()
  }
  touchIdleTimer()
  return page
}

// Mid-action resilience: run *fn* against the live page; if the action dies
// because the browser closed mid-call, relaunch ONCE and retry the same fn.
async function withPage(fn) {
  const p = await getPage()
  try {
    return await fn(p)
  } catch (err) {
    if (!isClosedError(err)) throw err
    await launchBrowser()
    touchIdleTimer()
    return await fn(page)
  }
}

// Event-driven settle detection. Race three signals and resolve on the FIRST
// that fires: a main-frame navigation event OR the URL changing, DOM
// quiescence (no mutations for `debounce` ms), or a hard cap. Losing racers
// are defused with .catch(() => {}) so they never reject, and the injected
// MutationObserver is disconnected on resolve. Best effort: a settle failure
// must never fail the tool call, just return control early.
async function settlePage(page, { debounce = 200, cap = 2500 } = {}) {
  try {
    let settled = false
    const finish = () => {
      if (settled) return
      settled = true
      // Disconnect any still-active observer from the page side.
      page.evaluate(() => {
        if (window.__jarvisSettleObs) {
          try {
            window.__jarvisSettleObs.disconnect()
          } catch {}
          window.__jarvisSettleObs = null
        }
      }).catch(() => {})
    }

    // (b) DOM quiescence: an observer resolves its injected promise once no
    // mutations have arrived for `debounce` ms. Fully self-contained (only
    // window/document globals), mirroring the domWalk page-side pattern.
    const quiescent = page.evaluate((deb) => {
      return new Promise((resolve) => {
        if (window.__jarvisSettleObs) {
          try {
            window.__jarvisSettleObs.disconnect()
          } catch {}
        }
        let idle = null
        const observer = new MutationObserver(() => {
          clearTimeout(idle)
          idle = setTimeout(() => {
            observer.disconnect()
            if (window.__jarvisSettleObs === observer) {
              window.__jarvisSettleObs = null
            }
            resolve(true)
          }, deb)
        })
        // Shared page-side handle: a concurrent settlePage call would
        // disconnect the previous observer via this reference. Harmless
        // today because tool handlers run serially.
        window.__jarvisSettleObs = observer
        observer.observe(document.documentElement, {
          childList: true,
          subtree: true,
          attributes: true,
          characterData: true,
        })
        // Arm the quiet timer immediately: a page that is already quiet
        // never mutates, so without this the injected promise would hang
        // until the hard cap.
        idle = setTimeout(() => {
          observer.disconnect()
          if (window.__jarvisSettleObs === observer) {
            window.__jarvisSettleObs = null
          }
          resolve(true)
        }, deb)
      })
    }, debounce)

    // (a) navigation: resolve ONLY when the main frame navigates. Subframe
    // (iframe) navigations from ads/embeds are noise and must not end settle
    // early, so each subframe event just re-arms the wait; on timeout the
    // promise rejects and the .catch() defusal below turns it into a losing
    // racer (the hard cap still bounds the race).
    const navigated = (async () => {
      while (true) {
        const frame = await page.waitForEvent("framenavigated", { timeout: cap })
        if (frame === page.mainFrame()) {
          finish()
          return
        }
      }
    })().catch(() => {})

    // (c) hard cap: never settle later than `cap` ms.
    const capped = new Promise((r) => setTimeout(r, cap))

    await Promise.race([quiescent, navigated, capped]).catch(() => {})
    finish()
  } catch {
    // Settle is best effort — the caller proceeds regardless.
  }
}

async function findChatInput() {
  const p = await getPage()
  const selectors = [
    'textarea',
    'div[role="textbox"]',
    '[contenteditable="true"]',
    '.ProseMirror',
    'input[type="text"]',
  ]
  for (const sel of selectors) {
    const el = await p.$(sel)
    if (el) {
      const visible = await el.isVisible().catch(() => false)
      if (visible) return { sel, el }
    }
  }
  return null
}

async function clickSend(p) {
  const sendButton = await p.$('button[aria-label*="send" i], button[aria-label*="Send" i], button[data-testid*="send" i]')
  if (sendButton && await sendButton.isVisible().catch(() => false)) {
    await sendButton.click()
    return
  }
  await p.keyboard.press("Enter")
}

// One McpServer with ALL tools registered. stdio mode uses a single
// instance; http mode creates one instance per MCP session.
function createMcpServer() {
  const server = new McpServer({
    name: "brave-control",
    version: "1.0.0",
  })

server.tool(
  "open_brave",
  "Launch a visible Brave browser instance with a persistent profile (keeps logins). Safe to call repeatedly.",
  {},
  async () => withPage(async () => {
    return { content: [{ type: "text", text: "Brave launched with persistent profile." }] }
  })
)

server.tool(
  "navigate",
  "Navigate the current Brave page to a URL and wait for it to load.",
  { url: z.string().describe("Full URL to navigate to") },
  async ({ url }) => withPage(async (p) => {
    await p.goto(url, { waitUntil: "domcontentloaded", timeout: 60000 })
    await p.waitForEvent("load", { timeout: 3000 }).catch(() => {})
    await settlePage(p, { debounce: 300 })
    return { content: [{ type: "text", text: `Navigated to ${p.url()}\nTitle: ${await p.title()}` }] }
  })
)

server.tool(
  "new_tab",
  "Open a new tab in the same browser window, navigate it to a URL, and make it the target page for subsequent tools.",
  { url: z.string().describe("Full URL to open in the new tab") },
  async ({ url }) => withPage(async (p) => {
    const np = await p.context().newPage()
    await np.goto(url, { waitUntil: "domcontentloaded", timeout: 60000 })
    await np.waitForEvent("load", { timeout: 3000 }).catch(() => {})
    await settlePage(np, { debounce: 300 })
    page = np
    np.setDefaultTimeout(30000)
    return { content: [{ type: "text", text: `Opened new tab: ${np.url()}\nTitle: ${await np.title()}\nTarget switched to new tab.` }] }
  })
)

server.tool(
  "switch_tab",
  "Switch the target page to an existing tab whose URL contains the given substring (e.g. 'qwen' or 'index.html'). The tab is NOT navigated.",
  { url_contains: z.string().describe("Substring to match against tab URLs") },
  async ({ url_contains }) => withPage(async () => {
    const pages = context.pages()
    const hit = pages.find((pg) => pg.url().includes(url_contains))
    if (!hit) {
      return { content: [{ type: "text", text: `No tab found containing "${url_contains}". Tabs: ${pages.map((pg) => pg.url()).join(" | ")}` }] }
    }
    page = hit
    return { content: [{ type: "text", text: `Switched target to: ${hit.url()}\nTitle: ${await hit.title()}` }] }
  })
)

server.tool(
  "ask_chat",
  "Type a prompt into the chat input on the current AI-chat page, send it, and wait for the full response to finish streaming. Returns the conversation text and any code blocks found.",
  { prompt: z.string().describe("The prompt to send to the AI chat") },
  async ({ prompt }) => withPage(async (p) => {
    const input = await findChatInput()
    if (!input) {
      const shot = path.join(process.env.TEMP || "C:\\Users\\mayan\\AppData\\Local\\Temp", "mcp-no-input.png")
      await p.screenshot({ path: shot })
      return { content: [{ type: "text", text: `Could not find a chat input on ${p.url()}. Screenshot saved to ${shot}` }] }
    }
    await input.el.click()
    await input.el.fill(prompt).catch(async () => {
      await input.el.pressSequentially(prompt, { delay: 5 })
    })
    await clickSend(p)

    // Wait for streaming to finish: poll for the stop button every 300ms; when
    // it is no longer visible the response is done. Hard-capped at 240s.
    let done = false
    const started = Date.now()
    while (Date.now() - started < 240000) {
      const stopBtn = await p.$('button[aria-label*="stop" i], button[aria-label*="Stop" i], button[aria-label*="Stop generating" i]')
      if (stopBtn && await stopBtn.isVisible().catch(() => false)) {
        await new Promise((r) => setTimeout(r, 300))
        continue
      }
      done = true
      break
    }

    const text = await p.evaluate(() => document.body.innerText.slice(0, 12000))
    const codeBlocks = await p.evaluate(() => {
      const out = []
      const blocks = document.querySelectorAll("pre")
      blocks.forEach((pre) => {
        const code = pre.innerText || ""
        if (code.length > 10) out.push(code)
      })
      return out
    })
    return {
      content: [
        { type: "text", text: `Response complete: ${done ? "yes" : "timeout (response may still be streaming)"}` },
        { type: "text", text: `PAGE TEXT:\n${text}` },
        { type: "text", text: `CODE BLOCKS FOUND: ${codeBlocks.length}` },
        ...codeBlocks.map((c) => ({ type: "text", text: `CODE BLOCK:\n${c}` })),
      ],
    }
  })
)

server.tool(
  "get_conversation",
  "Return the current page's visible text (the full chat conversation) as plain text.",
  {},
  async () => withPage(async (p) => {
    const text = await p.evaluate(() => document.body.innerText.slice(0, 20000))
    return { content: [{ type: "text", text: text }] }
  })
)

server.tool(
  "extract_code_blocks",
  "Extract every <pre> code block currently visible on the page and return them individually.",
  {},
  async () => withPage(async (p) => {
    const codeBlocks = await p.evaluate(() => {
      const out = []
      document.querySelectorAll("pre").forEach((pre) => {
        const code = pre.innerText || ""
        if (code.length > 10) out.push(code)
      })
      return out
    })
    return {
      content: [
        { type: "text", text: `CODE BLOCKS FOUND: ${codeBlocks.length}` },
        ...codeBlocks.map((c) => ({ type: "text", text: `CODE BLOCK:\n${c}` })),
      ],
    }
  })
)

server.tool(
  "extract_full_code_blocks",
  "Extract every <pre> code block on the page using textContent (includes scrolled/virtualized content that innerText hides). Returns full block text.",
  {},
  async () => withPage(async (p) => {
    const codeBlocks = await p.evaluate(() => {
      const out = []
      document.querySelectorAll("pre").forEach((pre) => {
        const code = pre.textContent || ""
        if (code.length > 10) out.push(code)
      })
      return out
    })
    return {
      content: [
        { type: "text", text: `FULL CODE BLOCKS FOUND: ${codeBlocks.length}` },
        ...codeBlocks.map((c) => ({ type: "text", text: `FULL CODE BLOCK (${c.length} chars):\n${c}` })),
      ],
    }
  })
)

server.tool(
  "understand_page",
  "Take a complete inventory of interactive elements on the current page, numbered by stable index (one call, no observe-then-think roundtrips). Returns the compact inventory grouped by section plus usage instructions.",
  {},
  async () => withPage(async (p) => {
    const records = await p.evaluate(domWalk)
    domCache = new Map(records.map((r) => [r.index, r.locator]))
    return { content: [{ type: "text", text: formatInventory(records) }] }
  })
)

server.tool(
  "click_element",
  "Click an element by its index from understand_page. If the index is stale (page changed), run understand_page again.",
  { index: z.number().int().describe("Element index from understand_page") },
  async ({ index }) => withPage(async (p) => {
    if (!domCache || !domCache.has(index)) return staleMessage()
    const locator = domCache.get(index)
    const stillThere = await p.evaluate((css) => {
      const el = document.querySelector(css)
      if (!el) return false
      const rect = el.getBoundingClientRect()
      if (rect.width <= 0 || rect.height <= 0) return false
      const style = getComputedStyle(el)
      return style.display !== "none" && style.visibility !== "hidden"
    }, locator)
    if (!stillThere) return staleMessage()
    try {
      await p.locator(locator).click({ timeout: 5000 })
    } catch {
      return staleMessage()
    }
    await settlePage(p, { debounce: 200 })
    return { content: [{ type: "text", text: `Clicked index ${index} (${locator}).` }] }
  })
)

server.tool(
  "fill_element",
  "Fill an input element (input/textarea/select/contenteditable) by its index from understand_page. If the index is stale (page changed), run understand_page again.",
  {
    index: z.number().int().describe("Element index from understand_page"),
    value: z.string().describe("Text to type into the element"),
  },
  async ({ index, value }) => withPage(async (p) => {
    if (!domCache || !domCache.has(index)) return staleMessage()
    const locator = domCache.get(index)
    const stillThere = await p.evaluate((css) => {
      const el = document.querySelector(css)
      if (!el) return false
      const rect = el.getBoundingClientRect()
      if (rect.width <= 0 || rect.height <= 0) return false
      const style = getComputedStyle(el)
      return style.display !== "none" && style.visibility !== "hidden"
    }, locator)
    if (!stillThere) return staleMessage()
    try {
      await p.locator(locator).fill(value, { timeout: 5000 })
    } catch {
      return staleMessage()
    }
    return { content: [{ type: "text", text: `Filled index ${index} (${locator}).` }] }
  })
)

server.tool(
  "evaluate",
  "Run arbitrary JavaScript in the page and return the JSON-serialized result (up to 50000 chars).",
  { expression: z.string().describe("JavaScript expression to evaluate in the page") },
  async ({ expression }) => withPage(async (p) => {
    const result = await p.evaluate((expr) => {
      const fn = new Function(`return (${expr})`)
      const v = fn()
      if (typeof v === "string") return v
      try {
        return JSON.stringify(v)
      } catch {
        return String(v)
      }
    }, expression)
    const text = String(result).slice(0, 50000)
    return { content: [{ type: "text", text: text }] }
  })
)

server.tool(
  "copy_code_block",
  "Grant clipboard permissions, click the copy button of the last code block on the page, and return the clipboard content.",
  {},
  async () => withPage(async (p) => {
    try {
      await p.context().grantPermissions(["clipboard-read", "clipboard-write"], { origin: "https://chat.qwen.ai" })
    } catch {}
    const clicked = await p.evaluate(() => {
      const pres = document.querySelectorAll("pre")
      const pre = pres[pres.length - 1]
      if (!pre) return "NO PRE"
      let el = pre
      for (let i = 0; i < 8; i++) {
        const btns = [...el.querySelectorAll("button")]
        const hit = btns.find((b) => {
          const s = ((b.getAttribute("aria-label") || "") + " " + (b.title || "") + " " + b.className + " " + (b.getAttribute("data-testid") || "")).toLowerCase()
          return s.includes("copy") || s.includes("复制")
        })
        if (hit) {
          hit.click()
          return "CLICKED: " + (hit.getAttribute("aria-label") || hit.title || hit.className)
        }
        el = el.parentElement
      }
      return "NO COPY BTN"
    })
    // Poll the clipboard every 100ms until it has content, hard-capped at
    // 2000ms (proceed with whatever is there, matching the old fixed-sleep
    // behaviour when the clipboard is empty).
    let text = ""
    const deadline = Date.now() + 2000
    while (Date.now() < deadline) {
      text = await p.evaluate(() => navigator.clipboard.readText().catch(() => "CLIPREAD-FAIL"))
      if (text && text !== "CLIPREAD-FAIL") break
      await new Promise((r) => setTimeout(r, 100))
    }
    return {
      content: [
        { type: "text", text: clicked },
        { type: "text", text: `CLIPBOARD (${text.length} chars):\n${text}` },
      ],
    }
  })
)

server.tool(
  "screenshot",
  "Take a screenshot of the current page and save it to a path on disk.",
  { path: z.string().describe("Where to save the PNG (e.g. C:\\temp\\shot.png)") },
  async ({ path: shotPath }) => withPage(async (p) => {
    fs.mkdirSync(path.dirname(shotPath), { recursive: true })
    await p.screenshot({ path: shotPath, fullPage: false })
    // G6/F39: explicit coordinate space — the viewport geometry the image
    // was captured in, so look-image coordinates convert without guessing.
    const geo = await viewportGeometry(p)
    return { content: [{ type: "text", text: `Screenshot saved to ${shotPath} (viewport=${geo.width}x${geo.height} dsf=${geo.deviceScaleFactor} url=${p.url()})` }] }
  })
)

// ── G6 (F13 + F40): real-input interaction primitives ──
// Every tool below uses REAL Playwright input (`locator.click()`,
// `locator.fill()`, `locator.selectOption()`, `locator.setChecked()`,
// `locator.setInputFiles()`, `locator.dragTo()`, `page.mouse`) against a
// frame-aware locator (`frame` = CSS selector of the <iframe>/<frame> in the
// top document; empty = top document). Each one probes the target FIRST
// (exists? visible? where?), acts, then settles and reports real after-state
// (url / navigated / title) — F40 wants outcomes, not bare "clicked: true".

server.tool(
  "click_locator",
  "Click an element by CSS selector with REAL Playwright input (actionability checks, settles, reports real after-state). Use this instead of coordinate JavaScript clicks. Optional frame targets an element inside an <iframe>/<frame>.",
  {
    css: z.string().describe("CSS selector of the element to click"),
    frame: z.string().optional().describe("CSS selector of the containing <iframe>/<frame>; omit for the top document"),
  },
  async ({ css, frame }) => withPage(async (p) => {
    const probe = await probeTarget(p, frame, css)
    if (!probe.exists) {
      return { content: [{ type: "text", text: `click_locator: no element matches ${css} (url=${probe.url}).` }] }
    }
    const { scope, where } = scoped(p, frame, css)
    await scope.scrollIntoViewIfNeeded({ timeout: 5000 }).catch(() => {})
    try {
      await scope.click({ timeout: 5000 })
    } catch (err) {
      return { content: [{ type: "text", text: `click_locator: click failed (${where}): ${err.message || err} (url=${probe.url}).` }] }
    }
    await settlePage(p, { debounce: 200 })
    const state = afterState(probe.url, p, { tag: "real-click", css, where })
    state.title = await p.title().catch(() => "")
    return { content: [{ type: "text", text: `Clicked via real input (${where}).\n${afterText(state)}` }] }
  })
)

server.tool(
  "fill_locator",
  "Fill a text input by CSS selector with REAL Playwright input (locator.fill), then submit exactly once via the chosen channel. Use this instead of synthetic KeyboardEvents and requestSubmit improvisations.",
  {
    css: z.string().describe("CSS selector of the input/textarea/contenteditable"),
    value: z.string().describe("Text to type into the element"),
    submit: z.string().optional().describe('How to submit after filling: "enter" (real keyboard Enter), "form" (enclosing form requestSubmit), or "none" (default)'),
    frame: z.string().optional().describe("CSS selector of the containing <iframe>/<frame>; omit for the top document"),
  },
  async ({ css, value, submit, frame }) => withPage(async (p) => {
    const probe = await probeTarget(p, frame, css)
    if (!probe.exists) {
      return { content: [{ type: "text", text: `fill_locator: no element matches (url=${probe.url}).` }] }
    }
    const { scope, where } = scoped(p, frame, css)
    // BA-15 parity with click_locator: an off-screen target is scrolled
    // into view instead of failing actionability — the agent advertises
    // off-screen marks as clickable, so the primitive must make them so.
    await scope.scrollIntoViewIfNeeded({ timeout: 5000 }).catch(() => {})
    try {
      await scope.fill(value, { timeout: 5000 })
    } catch (err) {
      return { content: [{ type: "text", text: `fill_locator: fill failed (${where}): ${err.message || err} (url=${probe.url}).` }] }
    }
    const mode = String(submit || "none").toLowerCase()
    if (mode === "enter") {
      try {
        await scope.press("Enter", { timeout: 5000 })
      } catch {
        await p.keyboard.press("Enter")
      }
    } else if (mode === "form") {
      await p.evaluate(({ cssArg, frameArg }) => {
        const root = frameArg
          ? ((document.querySelector(frameArg) || {}).contentDocument || null)
          : document
        if (!root) return
        const el = root.querySelector(cssArg)
        const form = el && el.form
        if (form && typeof form.requestSubmit === "function") form.requestSubmit()
      }, { cssArg: css, frameArg: (frame && String(frame).trim()) || "" }).catch(() => {})
    }
    await settlePage(p, { debounce: 200 })
    const state = afterState(probe.url, p, { filled: "yes", where })
    state.title = await p.title().catch(() => "")
    return { content: [{ type: "text", text: `Filled via real input (${where}, submit=${mode}).\n${afterText(state)}` }] }
  })
)

server.tool(
  "scroll",
  "Scroll the page or a scrollable element with REAL mouse-wheel input (page.mouse.wheel). Direction: up|down|left|right.",
  {
    direction: z.string().optional().describe('Scroll direction: "up", "down" (default), "left" or "right"'),
    amount: z.number().int().optional().describe("Wheel delta in px (default 600, max 5000)"),
    css: z.string().optional().describe("CSS selector of the scrollable element; omit to scroll the page"),
    frame: z.string().optional().describe("CSS selector of the containing <iframe>/<frame>; omit for the top document"),
  },
  async ({ direction, amount, css, frame }) => withPage(async (p) => {
    const before = p.url()
    const dir = String(direction || "down").toLowerCase()
    const delta = Number.isFinite(amount) && amount > 0 ? Math.min(amount, 5000) : 600
    let dx = 0
    let dy = 0
    if (dir === "up") dy = -delta
    else if (dir === "down") dy = delta
    else if (dir === "left") dx = -delta
    else if (dir === "right") dx = delta
    else {
      return { content: [{ type: "text", text: `scroll: unknown direction (expected up|down|left|right).` }] }
    }
    if ((css && String(css).trim()) || (frame && String(frame).trim())) {
      // Nested container: hover it first, THEN wheel. Refuse when the target
      // does not resolve (F40: honest outcomes, not phantom scrolls).
      const probe = await probeTarget(p, frame, css || "body")
      if (!probe.exists) {
        return { content: [{ type: "text", text: `scroll: no element matches (url=${probe.url}).` }] }
      }
      const { scope } = scoped(p, frame, css || "body")
      const box = await scope.boundingBox().catch(() => null)
      if (!box) {
        return { content: [{ type: "text", text: `scroll: element has no layout box (url=${probe.url}).` }] }
      }
      await p.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
      await p.mouse.wheel(dx, dy)
    } else {
      await p.mouse.wheel(dx, dy)
    }
    await settlePage(p, { debounce: 200 })
    const state = afterState(before, p, {})
    state.title = await p.title().catch(() => "")
    return { content: [{ type: "text", text: `Scrolled ${dir} ${delta}px.\n${afterText(state)}` }] }
  })
)

server.tool(
  "select_option",
  "Select <option>(s) in a <select> with REAL Playwright input (locator.selectOption). The value is matched against option value AND label.",
  {
    css: z.string().describe("CSS selector of the <select> element"),
    value: z.string().describe("Option value or label to select"),
    frame: z.string().optional().describe("CSS selector of the containing <iframe>/<frame>; omit for the top document"),
  },
  async ({ css, value, frame }) => withPage(async (p) => {
    const probe = await probeTarget(p, frame, css)
    if (!probe.exists) {
      return { content: [{ type: "text", text: `select_option: no element matches (url=${probe.url}).` }] }
    }
    const { scope, where } = scoped(p, frame, css)
    let selected = []
    try {
      selected = await scope.selectOption({ label: value }, { timeout: 5000 })
        .catch(() => scope.selectOption(value, { timeout: 5000 }))
    } catch (err) {
      return { content: [{ type: "text", text: `select_option: failed (${where}): ${err.message || err} (url=${probe.url}).` }] }
    }
    await settlePage(p, { debounce: 200 })
    const state = afterState(probe.url, p, {})
    state.title = await p.title().catch(() => "")
    return { content: [{ type: "text", text: `Selected option (${where}).\n${afterText(state)}` }] }
  })
)

server.tool(
  "set_checked",
  "Check/uncheck a checkbox or radio with REAL Playwright input (locator.setChecked).",
  {
    css: z.string().describe("CSS selector of the checkbox/radio input"),
    checked: z.boolean().describe("true to check, false to uncheck"),
    frame: z.string().optional().describe("CSS selector of the containing <iframe>/<frame>; omit for the top document"),
  },
  async ({ css, checked, frame }) => withPage(async (p) => {
    const probe = await probeTarget(p, frame, css)
    if (!probe.exists) {
      return { content: [{ type: "text", text: `set_checked: no element matches (url=${probe.url}).` }] }
    }
    const { scope, where } = scoped(p, frame, css)
    const want = Boolean(checked)
    try {
      await scope.setChecked(want, { timeout: 5000 })
    } catch (err) {
      return { content: [{ type: "text", text: `set_checked: failed (${where}): ${err.message || err} (url=${probe.url}).` }] }
    }
    await settlePage(p, { debounce: 200 })
    const state = afterState(probe.url, p, {})
    state.title = await p.title().catch(() => "")
    return { content: [{ type: "text", text: `Set checked=${want} (${where}).\n${afterText(state)}` }] }
  })
)

server.tool(
  "upload_file",
  "Upload file(s) through a file input with REAL Playwright input (locator.setInputFiles). Paths must exist; the caller (jarvis agent) binds uploads to approved workspace paths BEFORE dispatch. Returns the chosen filenames so the outcome is verifiable.",
  {
    css: z.string().describe("CSS selector of the <input type=file> element"),
    paths: z.array(z.string()).describe("Absolute local file path(s) to upload"),
    frame: z.string().optional().describe("CSS selector of the containing <iframe>/<frame>; omit for the top document"),
  },
  async ({ css, paths, frame }) => withPage(async (p) => {
    const probe = await probeTarget(p, frame, css)
    if (!probe.exists) {
      return { content: [{ type: "text", text: `upload_file: no element matches (url=${probe.url}).` }] }
    }
    const files = (Array.isArray(paths) ? paths : [paths]).filter((v) => v && String(v).trim())
    if (!files.length) {
      return { content: [{ type: "text", text: `upload_file: no file paths supplied (url=${probe.url}).` }] }
    }
    const missing = files.filter((f) => { try { return !fs.statSync(f).isFile() } catch { return true } })
    if (missing.length) {
      return { content: [{ type: "text", text: `upload_file: not found on disk (${missing.length} path(s)).` }] }
    }
    const { scope, where } = scoped(p, frame, css)
    try {
      await scope.setInputFiles(files, { timeout: 10000 })
    } catch (err) {
      return { content: [{ type: "text", text: `upload_file: failed (${where}): ${err.message || err} (url=${probe.url}).` }] }
    }
    await settlePage(p, { debounce: 200 })
    const state = afterState(probe.url, p, {})
    state.title = await p.title().catch(() => "")
    return { content: [{ type: "text", text: `Uploaded ${files.length} file(s) (${where}).\n${afterText(state)}` }] }
  })
)

server.tool(
  "download",
  "Click an element that starts a download and SAVE the file to a local downloads directory (real Playwright download event). Returns a completion event with the on-disk artifact path and filename so the caller can cite it.",
  {
    css: z.string().describe("CSS selector of the element whose click starts the download"),
    directory: z.string().optional().describe("Directory to save into (default: %TEMP%\\jarvis_downloads)"),
    frame: z.string().optional().describe("CSS selector of the containing <iframe>/<frame>; omit for the top document"),
  },
  async ({ css, directory, frame }) => withPage(async (p) => {
    const probe = await probeTarget(p, frame, css)
    if (!probe.exists) {
      return { content: [{ type: "text", text: `download: no element matches (url=${probe.url}).` }] }
    }
    const dest = directory && String(directory).trim()
      ? String(directory).trim()
      : path.join(os.tmpdir(), "jarvis_downloads")
    fs.mkdirSync(dest, { recursive: true })
    const { scope, where } = scoped(p, frame, css)
    let dl = null
    try {
      const [download] = await Promise.all([
        p.waitForEvent("download", { timeout: 30000 }),
        scope.click({ timeout: 5000 }),
      ])
      dl = download
    } catch (err) {
      return { content: [{ type: "text", text: `download: no download event within 30s (${where}): ${err.message || err} (url=${probe.url}).` }] }
    }
    const filename = dl.suggestedFilename() || "download"
    const target = path.join(dest, path.basename(filename))
    try {
      await dl.saveAs(target)
    } catch (err) {
      return { content: [{ type: "text", text: `download: could not save artifact: ${err.message || err}.` }] }
    }
    await settlePage(p, { debounce: 200 })
    const state = afterState(probe.url, p, {})
    state.title = await p.title().catch(() => "")
    return { content: [{ type: "text", text: `Download complete.\nartifact=${target}\n${afterText(state)}` }] }
  })
)

server.tool(
  "drag_drop",
  "Drag from one element and drop onto another with REAL mouse input (locator.dragTo). Use for sliders, kanban cards, file-drop zones and range handles.",
  {
    css: z.string().describe("CSS selector of the element to drag (source)"),
    target_css: z.string().describe("CSS selector of the drop target"),
    frame: z.string().optional().describe("CSS selector of the <iframe>/<frame> containing BOTH elements; omit for the top document"),
  },
  async ({ css, target_css, frame }) => withPage(async (p) => {
    const srcProbe = await probeTarget(p, frame, css)
    if (!srcProbe.exists) {
      return { content: [{ type: "text", text: `drag_drop: no source element matches (url=${srcProbe.url}).` }] }
    }
    const dstProbe = await probeTarget(p, frame, target_css)
    if (!dstProbe.exists) {
      return { content: [{ type: "text", text: `drag_drop: no target element matches (url=${dstProbe.url}).` }] }
    }
    const { scope: src, where } = scoped(p, frame, css)
    const { scope: dst } = scoped(p, frame, target_css)
    try {
      await src.dragTo(dst, { timeout: 10000 })
    } catch (err) {
      return { content: [{ type: "text", text: `drag_drop: failed (${where}): ${err.message || err} (url=${srcProbe.url}).` }] }
    }
    await settlePage(p, { debounce: 200 })
    const state = afterState(srcProbe.url, p, {})
    state.title = await p.title().catch(() => "")
    return { content: [{ type: "text", text: `Dragged onto target (${where}).\n${afterText(state)}` }] }
  })
)

server.tool(
  "close_brave",
  {},
  async () => {
    if (browser) {
      await browser.close().catch(() => {})
      browser = null
      context = null
      page = null
      domCache = null
      return { content: [{ type: "text", text: "Brave closed." }] }
    }
    return { content: [{ type: "text", text: "No browser was open." }] }
  }
)

server.tool(
  "list_dir",
  "List files and folders in a local directory. Use this for desktop or file-system questions. Accepts absolute paths and ~; defaults to the user home directory.",
  { path: z.string().optional().describe("Directory path (absolute or ~); defaults to home") },
  async ({ path: dirPath }) => {
    const result = listDir(dirPath || os.homedir())
    const text = result.text || result.error || "Unknown error"
    return { content: [{ type: "text", text }] }
  }
)

server.tool(
  "read_file",
  "Read a local text file (absolute path or ~). Returns up to 4000 characters.",
  { path: z.string().describe("File path (absolute or ~)"), max_chars: z.number().int().optional().describe("Max characters to return (default 4000)") },
  async ({ path: filePath, max_chars }) => {
    const result = readFileText(filePath, max_chars)
    const text = result.text || result.error || "Unknown error"
    return { content: [{ type: "text", text }] }
  }
)

server.tool(
  "list_tabs",
  "List all open browser tabs with index, title, URL, and active marker. Use this first when the user references content that should already be open or their screen.",
  {},
  async () => withPage(async () => {
    const pages = context.pages()
    const infos = await buildTabInfos(pages, page)
    const text = formatTabs(infos)
    return { content: [{ type: "text", text }] }
  })
)

  return server
}

if (HTTP_MODE) {
  if (!HTTP_TOKEN) {
    console.error("[brave-mcp] BRAVE_MCP_MODE=http requires BRAVE_MCP_TOKEN. Refusing to start.")
    process.exit(1)
  }

  // One browser per jarvis boot, shared by every opencode session: the
  // daemon keeps the transport alive across requests and recycles the
  // browser when it idles (touchIdleTimer resets on every tool call).
  const sessions = new Map()

  function readBody(req) {
    return new Promise((resolve) => {
      let data = ""
      req.on("data", (chunk) => {
        data += chunk
        if (data.length > 5e6) req.destroy()
      })
      req.on("end", () => {
        if (!data) return resolve(undefined)
        try {
          resolve(JSON.parse(data))
        } catch {
          resolve(undefined)
        }
      })
      req.on("error", () => resolve(undefined))
    })
  }

  const httpServer = http.createServer(async (req, res) => {
    // Auth gate BEFORE touching any transport or session state.
    if (req.headers["authorization"] !== "Bearer " + HTTP_TOKEN) {
      res.writeHead(401, { "Content-Type": "application/json" })
      res.end(JSON.stringify({ error: "Unauthorized" }))
      return
    }
    if (req.url !== "/mcp") {
      res.writeHead(404, { "Content-Type": "application/json" })
      res.end(JSON.stringify({ error: "Not found" }))
      return
    }
    const sessionId = req.headers["mcp-session-id"]
    let transport
    if (sessionId && sessions.has(sessionId)) {
      transport = sessions.get(sessionId)
    } else {
      transport = new StreamableHTTPServerTransport({
        sessionIdGenerator: () => randomUUID(),
        onsessioninitialized: (id) => sessions.set(id, transport),
      })
      const server = createMcpServer()
      await server.connect(transport)
    }
    try {
      await transport.handleRequest(req, res, await readBody(req))
    } catch (err) {
      if (!res.headersSent) {
        res.writeHead(500, { "Content-Type": "application/json" })
        res.end(JSON.stringify({ error: String((err && err.message) || err) }))
      } else {
        res.end()
      }
    }
    if (req.method === "DELETE" && sessionId) {
      sessions.delete(sessionId)
    }
  })

  httpServer.listen(HTTP_PORT, "127.0.0.1", () => {
    console.error(`[brave-mcp] HTTP daemon listening on http://127.0.0.1:${HTTP_PORT}/mcp`)
  })
} else {
  const server = createMcpServer()
  const transport = new StdioServerTransport()
  await server.connect(transport)
}