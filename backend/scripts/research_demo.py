"""Browser-automation research demo: search, visit top results, extract data.

Search sources, in order of preference:
  1. HTML search (Brave -> DuckDuckGo -> Bing) - works when not anti-bot'd.
  2. Keyless public developer APIs (Hacker News Algolia + arXiv) - the
     reliable free path and a great fit for AI-model/news research.

Once URLs are found, Playwright drives real Chrome to visit each page,
extract its main text, and summarize with Gemini (raw excerpt as fallback).
"""

import json
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from playwright.sync_api import sync_playwright

QUERY = "new AI models about to be released"
MAX_RESULTS = 10
TEXT_LIMIT = 2500
REPORT_PATH = _REPO_ROOT / "data" / "research_report.md"

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}


def _clean_text(raw):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return "\n".join(lines)[:TEXT_LIMIT]


def _resolve_url(href):
    """Unwrap Brave/Bing redirect wraps to the real target URL."""
    if not href:
        return href
    parsed = urlparse(href)
    target = None
    if "redirect" in parsed.path:
        target = parse_qs(parsed.query).get("url", [None])[0]
    elif "/ck/a" in parsed.path:
        target = parse_qs(parsed.query).get("u", [None])[0]
    if target:
        return unquote(target)
    return href


# ---------------- HTML search extraction ----------------

def _extract_generic(page, blocks, link_sel, desc_sel):
    """Run a CSS-based extractor: blocks -> links -> description."""
    nodes = page.query_selector_all(blocks)
    out = []
    for node in nodes:
        link = node.query_selector(link_sel) or node.query_selector("a[href]")
        if not link:
            continue
        href = _resolve_url(link.get_attribute("href"))
        if not href or not href.startswith("http"):
            continue
        desc = ""
        dn = node.query_selector(desc_sel)
        if dn:
            desc = dn.inner_text().strip()
        title = link.inner_text().strip()
        out.append({"title": title or href, "url": href, "snippet": desc})
        if len(out) >= MAX_RESULTS:
            break
    return out


def extract_brave(page):
    try:
        page.wait_for_selector(".snippet, #results", timeout=10000)
    except Exception:
        return []
    return _extract_generic(page, ".snippet", "a.snippet-title, a.result-header, a[href]", ".snippet-description, .snippet-content, p")


def extract_ddg(page):
    try:
        page.wait_for_selector(".result", timeout=10000)
    except Exception:
        return []
    return _extract_generic(page, ".result", "a.result__a", ".result__snippet")


def extract_bing(page):
    try:
        page.wait_for_selector("li.b_algo", timeout=10000)
    except Exception:
        return []
    return [r for r in _extract_generic(page, "li.b_algo", "h2 a", "p") if r["title"] != "[no-text]"]


# ---------------- keyless API search sources ----------------

def search_hacker_news():
    url = ("https://hn.algolia.com/api/v1/search?query=" + quote_plus("new AI models released")
           + "&tags=story&hitsPerPage=15")
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode("utf-8", "ignore"))
    out = []
    for hit in data.get("hits", []):
        url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
        out.append({
            "title": hit.get("title") or url,
            "url": url,
            "snippet": f"Score {hit.get('points', '?')} - {hit.get('num_comments', 0)} comments",
        })
    return out


def search_arxiv():
    ns = {"a": "http://www.w3.org/2005/Atom"}
    q = quote_plus('cat:cs.LG AND abs:"large language model" AND abs:release')
    url = "http://export.arxiv.org/api/query?search_query=" + q + "&start=0&max_results=12&sortBy=submittedDate&sortOrder=descending"
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        xml = r.read().decode("utf-8", "ignore")
    out = []
    root = ET.fromstring(xml)
    for entry in root.findall("a:entry", ns):
        title = (entry.find("a:title", ns).text or "").strip().replace("\n", " ")
        link = entry.find("a:link", ns)
        href = link.get("href") if link is not None else ""
        summary = (entry.find("a:summary", ns).text or "").strip()[:160]
        if href:
            out.append({"title": title or href, "url": href, "snippet": summary})
    return out


# ---------------- page fetch + summary ----------------

def fetch_page(page, url):
    page.goto(url, timeout=20000, wait_until="domcontentloaded")
    page.wait_for_timeout(300)
    title = page.title() or url
    raw = page.evaluate(
        """() => {
            const clone = document.body.cloneNode(true);
            clone.querySelectorAll(
                'script,style,noscript,nav,header,footer,aside,iframe,form,button'
            ).forEach(e => e.remove());
            return clone.innerText;
        }"""
    )
    return title, _clean_text(raw or "")


def summarize_with_gemini(title, url, text):
    try:
        from backend.services.gemini_client import ask_gemini_chat

        response = ask_gemini_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You extract research data. Given a page title, URL and excerpt, "
                        "return 2-3 sentences of factual data: what this source reports, "
                        "key model names/dates/facts. No preamble."
                    ),
                },
                {"role": "user", "content": f"TITLE: {title}\nURL: {url}\nEXCERPT:\n{text}"},
            ],
            temperature=0.2,
            max_tokens=200,
        )
        if not response or not response.get("choices"):
            print("    [GEMINI] empty response")
            return None
        return response["choices"][0].get("message", {}).get("content", "").strip() or None
    except Exception as exc:
        print(f"    [GEMINI] failed: {type(exc).__name__}: {exc}")
        return None


def _gather_results(page):
    """Return (engine_name, results). Braces for anti-bot on the HTML engines."""
    page.goto(f"https://search.brave.com/search?q={quote_plus(QUERY)}", timeout=25000, wait_until="domcontentloaded")
    page.wait_for_timeout(1200)
    results = extract_brave(page)
    if len(results) >= MAX_RESULTS // 2:
        return "brave", results

    page.goto(f"https://html.duckduckgo.com/html/?q={quote_plus(QUERY)}", timeout=25000, wait_until="domcontentloaded")
    results = extract_ddg(page)
    if len(results) >= MAX_RESULTS // 2:
        return "duckduckgo", results

    page.goto(f"https://www.bing.com/search?q={quote_plus(QUERY)}", timeout=25000, wait_until="domcontentloaded")
    results = extract_bing(page)
    if len(results) >= 1:
        return "bing", results

    api_results = search_hacker_news() + search_arxiv()
    return "hackernews+arxiv", api_results[:MAX_RESULTS]


def main():
    collected = []
    failures = []

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True, args=["--disable-gpu"])
        except Exception:
            browser = p.chromium.launch(channel="chrome", headless=True, args=["--disable-gpu"])
        context = browser.new_context(locale="en-US")
        page = context.new_page()

        engine, results = _gather_results(page)
        print(f'[SEARCH] Engine: {engine} | {len(results)} results for "{QUERY}"')
        for i, r in enumerate(results[:MAX_RESULTS], 1):
            print(f"  {i}. {r['title'][:85]}\n     {r['url']}")

        for i, r in enumerate(results[:MAX_RESULTS], 1):
            print(f"\n[{i}/{min(len(results), MAX_RESULTS)}] Fetching: {r['url']}")
            try:
                title, text = fetch_page(page, r["url"])
            except Exception as exc:
                print(f"    FAILED to fetch: {type(exc).__name__}: {str(exc)[:90]}")
                failures.append({"title": r.get("title"), "url": r["url"], "error": str(exc)[:140]})
                continue
            print(f"    Page title: {title[:90]}")
            summary = summarize_with_gemini(title, r["url"], text) if text else None
            collected_entry = {
                "result_title": r.get("title", r["url"]),
                "url": r["url"],
                "snippet": r.get("snippet", ""),
                "page_title": title,
                "text_excerpt": text[:800],
                "summary": summary,
            }
            collected.append(collected_entry)
            print(f"    Summary: {summary[:240] if summary else '(none - see excerpt in report)'}")

        browser.close()

    lines = [f"# Research Report: \"{QUERY}\"", "",
             f"- Engine: {engine} | Sites visited: {len(collected)} | Failed: {len(failures)}",
             "- Tool: Playwright managed Chromium + Gemini summaries", ""]
    for e in collected:
        lines += [f"## {e['result_title']}", f"- URL: {e['url']}", f"- Page: {e['page_title']}"]
        if e["summary"]:
            lines.append(f"- Summary: {e['summary']}")
        if e["text_excerpt"]:
            lines.append(f"- Excerpt: {e['text_excerpt'][:450]}...")
        lines.append("")
    if failures:
        lines.append("## Failures")
        for f in failures:
            lines.append(f"- {f['title']} | {f['url']} | {f['error']}")
        lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[REPORT] saved to {REPORT_PATH} | visited: {len(collected)}, failed: {len(failures)}")


if __name__ == "__main__":
    main()