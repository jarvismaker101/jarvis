// lib/fs_tools.mjs
//
// Local file-system helpers for the brave-control MCP server.
// These are pure Node helpers that NEVER throw — callers (MCP tool handlers)
// receive { text } on success or { error } on failure so the tool call can
// fail soft and the agent can try another approach.

import fs from "fs"
import os from "os"
import path from "path"

function expandHome(p) {
  if (!p) return p
  if (p === "~" || p.startsWith("~/") || p.startsWith("~\\")) {
    return path.join(os.homedir(), p.slice(1))
  }
  return p
}

export function listDir(dirPath) {
  try {
    const raw = dirPath || os.homedir()
    const target = expandHome(String(raw))
    let stat
    try {
      stat = fs.statSync(target)
    } catch (e) {
      return { error: `No such directory: ${target}` }
    }
    if (!stat.isDirectory()) {
      return { error: `Not a directory: ${target}` }
    }

    const dirents = fs.readdirSync(target, { withFileTypes: true })

    // Sort: dirs first, then files, each alphabetical (case-insensitive).
    dirents.sort((a, b) => {
      const aDir = a.isDirectory()
      const bDir = b.isDirectory()
      if (aDir !== bDir) return aDir ? -1 : 1
      return a.name.localeCompare(b.name, undefined, { sensitivity: "base" })
    })

    const total = dirents.length
    let dirCount = 0
    let fileCount = 0
    for (const d of dirents) {
      if (d.isDirectory()) dirCount++
      else fileCount++
    }

    const capped = total > 200
    const shown = capped ? dirents.slice(0, 200) : dirents
    const remaining = total - 200

    const lines = []
    for (const d of shown) {
      if (d.isDirectory()) {
        lines.push(`[DIR] ${d.name}`)
      } else {
        let size = 0
        try {
          const st = fs.statSync(path.join(target, d.name))
          size = st.isFile() ? st.size : 0
        } catch {
          size = 0
        }
        lines.push(`[FILE] ${d.name} (${size} bytes)`)
      }
    }

    if (capped) {
      lines.push(`... and ${remaining} more entries not shown`)
    }

    lines.push(`total: ${dirCount} folders, ${fileCount} files`)
    return { text: lines.join("\n") }
  } catch (e) {
    return { error: String((e && e.message) || e) }
  }
}

export function readFileText(filePath, maxChars = 4000) {
  try {
    if (!filePath) {
      return { error: "No path provided" }
    }
    const target = expandHome(String(filePath))
    let stat
    try {
      stat = fs.statSync(target)
    } catch (e) {
      return { error: `No such file: ${target}` }
    }
    if (stat.isDirectory()) {
      return { error: `Not a file (is a directory): ${target}` }
    }

    const content = fs.readFileSync(target, "utf8")
    const totalChars = content.length
    if (totalChars > maxChars) {
      const clipped = content.slice(0, maxChars)
      const text = clipped + `\n...[truncated, total ${totalChars} chars]`
      return { text, totalChars }
    }
    return { text: content, totalChars }
  } catch (e) {
    return { error: String((e && e.message) || e) }
  }
}
