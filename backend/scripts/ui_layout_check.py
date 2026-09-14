"""Headless-layout check for the chat UI scroll fix.

Loads frontend/index.html in headless Chromium, injects a long conversation,
and verifies:
  1. #chat-box actually overflows (scrollbar engages)   -> flex-shrink:0 works
  2. messages keep natural height (no vertical squeeze)
  3. forcing flex-shrink:1 reproduces the squeeze bug   -> root cause proven

Run: backend\\venv\\Scripts\\python.exe -m backend.scripts.ui_layout_check
"""

import pathlib

from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parents[2]
URL = (ROOT / "frontend" / "index.html").as_uri()

MEASURE = """(() => {
  const b = document.getElementById('chat-box');
  const hs = [...b.querySelectorAll('.message')].map((m) => m.offsetHeight);
  return {
    client: b.clientHeight,
    scroll: b.scrollHeight,
    minH: Math.min.apply(null, hs),
    maxH: Math.max.apply(null, hs),
    count: hs.length,
  };
})()"""

INJECT = """(() => {
  const box = document.getElementById('chat-box');
  for (let i = 0; i < 28; i++) {
    const d = document.createElement('div');
    d.className = 'message ' + (i % 2 ? 'bot' : 'user');
    const l = document.createElement('span');
    l.className = i % 2 ? 'bot-label' : 'user-label';
    l.textContent = i % 2 ? 'JARVIS' : 'YOU';
    const t = document.createElement('span');
    t.className = 'msg-text';
    t.textContent = 'Exchange ' + i + ': Lorem ipsum dolor sit amet consectetur adipiscing elit. '.repeat(2);
    d.appendChild(l);
    d.appendChild(t);
    box.appendChild(d);
  }
  return box.children.length;
})()"""


def main():
    with sync_playwright() as p:
        browser = None
        last_err = None
        for kwargs in ({}, {"channel": "msedge"}, {"channel": "chrome"}):
            try:
                browser = p.chromium.launch(**kwargs)
                break
            except Exception as exc:  # try bundled chromium, then system Edge/Chrome
                last_err = exc
        if browser is None:
            raise SystemExit(f"NO BROWSER AVAILABLE: {last_err}")

        page = browser.new_page(viewport={"width": 960, "height": 680})
        page.goto(URL)
        page.wait_for_timeout(400)

        n = page.evaluate(INJECT)
        fixed = page.evaluate(MEASURE)

        # Force the pre-fix behavior to prove the mechanism
        page.evaluate(
            "document.querySelectorAll('#chat-box .message')"
            ".forEach(m => m.style.setProperty('flex-shrink', '1'))"
        )
        simulated = page.evaluate(MEASURE)

        # Restore and confirm
        page.evaluate(
            "document.querySelectorAll('#chat-box .message')"
            ".forEach(m => m.style.removeProperty('flex-shrink'))"
        )
        restored = page.evaluate(MEASURE)

        browser.close()

    print("messages injected :", n)
    print("FIXED  (shrink:0) :", fixed)
    print("PRE-FIX SIM       :", simulated)
    print("RESTORED          :", restored)

    scroll_ok = fixed["scroll"] > fixed["client"]
    unsqueezed_ok = fixed["minH"] > 36
    # Squeeze = dramatic compression when shrink is re-enabled (Chromium floors
    # shrunken cards at roughly one clipped line + padding, so compare against
    # the natural size rather than expecting an exact fit of clientHeight).
    squeeze_repro = (
        simulated["minH"] < fixed["minH"] * 0.5
        and simulated["scroll"] < fixed["scroll"] * 0.5
    )
    restore_ok = (
        abs(restored["minH"] - fixed["minH"]) < 1.0
        and abs(restored["scroll"] - fixed["scroll"]) < 2.0
    )
    print("SCROLLBAR_ACTIVE_WITH_FIX =", scroll_ok)
    print("NO_VERTICAL_SQUEEZE       =", unsqueezed_ok)
    print("SQUEEZE_REPRODUCED_IN_SIM =", squeeze_repro)
    print("IDEMPOTENT_AFTER_RESTORE  =", restore_ok)
    print("OVERALL:", "PASS" if all((scroll_ok, unsqueezed_ok, squeeze_repro)) else "FAIL")


if __name__ == "__main__":
    main()
