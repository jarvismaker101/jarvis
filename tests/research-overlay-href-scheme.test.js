"use strict";
// H9 (2026-09-23 audit) - research overlay markdown links must never become
// javascript:/data:/vbscript: hrefs (escapeHtml does not touch URL schemes).
// Harness mirrors tests/backend-request-policy.test.js: run the renderer in a
// vm with minimal DOM stubs and drive its inline() markdown helper directly.
const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const SRC = fs.readFileSync(
  path.join(__dirname, "..", "frontend", "research_overlay_renderer.js"),
  "utf8");

function loadRenderer() {
  const el = () => ({
    textContent: "", innerHTML: "", style: {},
    classList: { add() {}, remove() {}, toggle() {} },
    addEventListener() {}, appendChild() {},
    querySelector() { return el(); }, querySelectorAll() { return []; },
    scrollHeight: 0, scrollTop: 0, offsetHeight: 100,
  });
  const stub = el();
  const ctx = {
    document: {
      getElementById: () => stub, addEventListener() {},
      querySelector: () => stub, querySelectorAll: () => [], body: stub,
    },
    window: { addEventListener() {}, jarvis: {} },
    console, setTimeout, clearTimeout, setInterval, clearInterval,
  };
  vm.createContext(ctx);
  vm.runInContext(SRC, ctx);
  return ctx;
}

const ctx = loadRenderer();
const inline = (text) => vm.runInContext(`inline(${JSON.stringify(text)})`, ctx);

test("javascript: markdown links render as plain label text", () => {
  const out = inline("[pwn](javascript:alert(1))");
  assert.ok(!out.includes("javascript:"), `href survived: ${out}`);
  assert.ok(!out.includes("<a "), `anchor created: ${out}`);
  assert.ok(out.includes("pwn"), `label lost: ${out}`);
});

test("data: and vbscript: schemes are also refused", () => {
  for (const url of ["data:text/html,<script>x</script>", "vbscript:msgbox(1)"]) {
    const out = inline(`[x](${url})`);
    assert.ok(!out.includes("<a "), `anchor created for ${url}: ${out}`);
  }
});

test("http(s) and mailto links stay clickable", () => {
  const out = inline(
    "[ok](https://x.example/a) [m](mailto:a@b.c) [p](http://y.example)");
  assert.ok(out.includes('href="https://x.example/a"'), out);
  assert.ok(out.includes('href="mailto:a@b.c"'), out);
  assert.ok(out.includes('href="http://y.example"'), out);
});
