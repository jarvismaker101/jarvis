"""Debug: dump what each search engine actually serves to headless Chrome."""

from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from playwright.sync_api import sync_playwright

QUERY = "new AI models about to be released 2026"

URLS = {
    "brave": f"https://search.brave.com/search?q={quote_plus(QUERY)}",
    "ddg-html": f"https://html.duckduckgo.com/html/?q={quote_plus(QUERY)}",
    "ddg-lite": f"https://lite.duckduckgo.com/lite/?q={quote_plus(QUERY)}",
    "bing": f"https://www.bing.com/search?q={quote_plus(QUERY)}",
}


def decode_bing_href(href):
    try:
        parsed = urlparse(href)
        if "/ck/a" in parsed.path:
            target = parse_qs(parsed.query).get("u", [None])[0]
            if target:
                return unquote(target)
    except Exception:
        pass
    return href


def print_debug(page, name, url):
    print("=" * 70)
    print(f"[{name}] {url}")
    try:
        page.goto(url, timeout=25000, wait_until="domcontentloaded")
        page.wait_for_timeout(1500)
        print("  TITLE:", (page.title() or "")[:120])
        print("  URL  :", page.url[:120])
        text = page.evaluate("() => document.body ? document.body.innerText : ''") or ""
        print("  BODY(600):", " ".join(text.split())[:600])
        anchors = page.evaluate(
            """() => Array.from(document.querySelectorAll('a[href]'))
                .map(a => ({href:a.href, text:(a.innerText||'').trim().slice(0,60)}))
                .filter(x => x.href.startsWith('http'))"""
        )
        print(f"  ANCHOR COUNT: {len(anchors)}")
        for a in anchors[:12]:
            print("    -", a["text"] or "[no-text]", "||", decode_bing_href(a["href"])[:100])
    except Exception as exc:
        print(f"  ERROR: {exc}")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(locale="en-US")
        page = context.new_page()
        for name, url in URLS.items():
            print_debug(page, name, url)
        browser.close()


if __name__ == "__main__":
    main()