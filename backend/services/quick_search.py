"""Fast default websearch: AI Overview first, no site scraping.

Tiered search contract (brain.handle_research_intent routes here):

  * DEFAULT mode — search via Brave Search (same captcha-free entry point
    the deep research flow uses), read the AI Overview answer box, come
    back with a short spoken+text summary. No website scraping at all.
    If the query has no AI Overview, degrade lightly: report the top
    result's snippet and say the overview wasn't available.
  * DEEPSEARCH mode (explicit "deepsearch" keyword in the user request) —
    the AI Overview PLUS the existing multi-site research flow
    (research_service.run_research with the overview pinned in).

The browser opens the SAME headed Chrome profile the deep research flow
uses (human-looking, captcha-solvable).  The AI answer is read from the
rendered DOM via stable semantic class names (``chatllm-answer``,
``chatllm-content``) with a completion-poll that waits for the streaming
answer to finish before extraction.  ``extract_ai_overview`` / ``extract_top_snippet``
handle non-Brave search engines as fallback.
"""

import asyncio
import os
import re
import time
from pathlib import Path
from urllib.parse import quote_plus

from backend.services.research_browser import maybe_await

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_PROFILE_DIR = _REPO_ROOT / "data" / "chrome_profile_jarvis"

# Candidate selectors for the AI Overview container, most specific first.
# Brave Search renders the AI answer with stable semantic class names
# (chatllm-answer, chatllm-content).  Google selectors are kept as fallback
# for non-Brave search engines.
_AI_OVERVIEW_SELECTORS = (
    "div.chatllm-answer",
    "div.chatllm-content",
    "div[class*='ai-answer']",
    "div[class*='ai-summary']",
    "div[data-attrid='AI Overview']",
    "div[data-attrid='kc:/chat/home:dialog']",
    "div.yXr61e",
    "div[class*='WaaZC']",
)

# Organic result link selectors (title + href).
_TOP_RESULT_SELECTORS = (
    "div.snippet a",
    "div#search div[data-sokoban-container] div.yuRUbf a",
    "div#search a h3",
)

# Snippet text selectors, paired with the link selectors above.
# Brave Search puts snippet text directly inside div.snippet elements
# (no dedicated .snippet-description child).
_TOP_RESULT_SNIPPET_SELECTORS = (
    "div.snippet-description",
    "div[class*='snippet-description']",
    "div.snippet",
    "div#search div[data-sncf], div#search div.VwiC3b",
)

# Ask tab selectors — Brave Search hides the AI overview for many queries;
# clicking the Ask tab routes through the AI chat interface instead.
_ASK_TAB_SELECTORS = (
    "button.ask-button",
    "a[href*='/ask?q=']",
)

# Ask-tab answer container selectors, most specific first. Live survey shows
# the answer in <div class="message assistant llm-output ..."> — semantic
# classes holding exactly the answer text, no shell. Svelte hash suffixes
# churn across deploys, so select on the semantic portion only.
_ASK_ANSWER_CONTAINER_SELECTORS = (
    "div.message.assistant.llm-output",
    "div.message.assistant",
    "div.llm-output",
)

# AI overview selectors for the Ask-tab chat page.
# (kept for reference but currently unused — extract_ai_overview reads
# <main> with ask_mode=True instead.)


def google_search_url(query):
    """Brave Search URL (same entry point as the deep research flow —
    avoids Google CAPTCHA on headed automation)."""
    return "https://search.brave.com/search?q=" + quote_plus(query)


def _clean(text, limit=360):
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text[:limit]


def _strip_markdown(text):
    """Strip bold/italic markdown for TTS — * and ** characters only."""
    return (text or "").replace("**", "").replace("*", "")


def _trim_at_disclaimer(text):
    """Trim overview text at the disclaimer marker 'AI-generated answer.'"""
    if not text:
        return text
    idx = text.find("AI-generated answer.")
    return text[:idx].strip() if idx != -1 else text


def _clean_snippet(snippet):
    """Strip title/URL/domain breadcrumb prefix from snippet text."""
    s = (snippet or "").strip()
    # Remove leading breadcrumb chain: anything before the LAST ">" or "›"
    # when the first segment contains a domain-like pattern (word.tld).
    parts = [p.strip() for p in re.split(r'\s*[>›]\s*', s)]
    if len(parts) >= 2 and re.search(r'\.[a-z]{2,}\b', parts[0], re.I):
        s = parts[-1]
    return _clean(s, limit=220)


def _clean_title(raw_title):
    """Extract the real title from the first line of a Brave result link."""
    if not raw_title:
        return ""
    first_line = raw_title.split("\n")[0].strip()
    # If the first line itself looks like a URL prefix, fall back to raw
    if re.match(r'^[\w.-]+\.[a-z]{2,}', first_line, re.I):
        return raw_title.strip()
    return first_line


async def extract_ai_overview(page, ask_mode=False, query=None):
    """Read the AI overview text from the rendered results page.

    Small and mockable on purpose: takes a Playwright page, returns the
    overview text or None when the query has no AI Overview. Selector
    knowledge lives only in _AI_OVERVIEW_SELECTORS.

    When *ask_mode* is True (Ask-tab view), prefers the stable answer
    container (``div.message.assistant.llm-output`` — live-verified to hold
    exactly the answer, no shell) and falls back to ``main`` innerText
    surgery when Brave renames the container classes. *query* powers the
    fallback's history/echo cut.

    F27: the page belongs to the ASYNC Playwright API, so every access is
    awaited (``await maybe_await(...)`` also accepts the plain mock pages the
    tests inject). The old synchronous reads silently got coroutine objects and
    "found" nothing.
    """
    for selector in _AI_OVERVIEW_SELECTORS:
        try:
            node = await maybe_await(page.query_selector(selector))
        except Exception:
            node = None
        if node is None:
            continue
        try:
            text = ((await maybe_await(node.inner_text())) or "").strip()
        except Exception:
            text = ""
        # Skip collapsed/label-only shells; a real overview carries content.
        if len(text) >= 80:
            return _clean(text, limit=2400)

    if not ask_mode:
        return None

    # Ask-tab view: prefer the stable answer container (exact answer, no
    # shell, no surgery). Falls through to <main> surgery below when Brave
    # renames the container classes.
    try:
        for selector in _ASK_ANSWER_CONTAINER_SELECTORS:
            try:
                node = await maybe_await(page.query_selector(selector))
            except Exception:
                node = None
            if node is None:
                continue
            try:
                text = ((await maybe_await(node.inner_text())) or "").strip()
            except Exception:
                text = ""
            if len(text) >= 80:
                return _clean(text, limit=2400)
    except Exception:
        pass

    # Fallback: <main> innerText surgery. Strip known UI shell lines, kill
    # the multi-line privacy-banner paragraph via regex (exact-line shell
    # matches only catch its heading), then cut conversation history plus
    # the echoed query — everything up to the LAST query occurrence is
    # chrome (sidebar history, banner, tabs, echo).
    try:
        main = await maybe_await(page.query_selector("main"))
        if main is None:
            return None
        text = ((await maybe_await(main.inner_text())) or "").strip()
        if len(text) < 200:
            return None
        lines = [l.strip() for l in text.split("\n")]
        _UI_SHELL = {
            "New Conversation", "New conversation", "Settings", "Ask",
            "All", "Images", "News", "Videos", "Maps", "Goggles",
            "Ctrl + Shift + O", "Encrypted & Private History", "History",
            "Got it", "Finished",
        }
        significant = [l for l in lines
                       if l and l not in _UI_SHELL and not l.startswith("Ctrl")]
        joined = " ".join(significant)
        joined = re.sub(r"Your chat history is encrypted.*?Learn more",
                        "", joined, flags=re.S)
        if query:
            idx = joined.rfind(query)
            if idx != -1:
                tail = joined[idx + len(query):]
                if re.match(r"\s*(?:Finished|Got it|\+\d+)", tail):
                    # Echo case: Finished/Got-it/+N markers sit between the
                    # echoed query and the answer — strip echo + markers.
                    joined = re.sub(r"^(?:\s*(?:Finished|Got it|\+\d+))+",
                                    "", tail).strip()
                else:
                    # The answer itself opens with the query words — keep it.
                    joined = joined[idx:]
        joined = re.sub(r"\s+", " ", joined).strip()
        # One-pass trailing-chrome cut: disclaimer, follow-up labels, and
        # citation breadcrumbs (domain + '>' or '›' chevron) — first hit wins.
        joined = re.split(
            r"(?:AI-generated answer\.|\bElaborate\b|\bCopy\b|\bTry again\b|"
            r"[A-Za-z0-9.-]+\.(?:com|org|net|io|in|co)\s*[>›])",
            joined, maxsplit=1)[0].strip()
        if len(joined) >= 80:
            return _clean(joined, limit=2400)
    except Exception:
        pass

    return None


async def extract_top_snippet(page):
    """Light no-overview fallback: the first organic result's title+URL.

    Tries each _TOP_RESULT_SELECTORS entry and pairs it with the
    corresponding _TOP_RESULT_SNIPPET_SELECTORS.  Deliberately does NOT
    navigate anywhere — the snippet shown on the results page is all we
    use.  Returns a dict or None. F27: every page access is awaited.
    """
    title = None
    url = None
    for sel in _TOP_RESULT_SELECTORS:
        try:
            el = await maybe_await(page.query_selector(sel))
            if el is None:
                continue
            t = ((await maybe_await(el.inner_text())) or "").strip()
            u = ((await maybe_await(el.get_attribute("href"))) or "").strip()
            if t:
                title = _clean_title(t)
                url = u
                break
        except Exception:
            continue
    if not title:
        return None
    snippet = ""
    for sel in _TOP_RESULT_SNIPPET_SELECTORS:
        try:
            container = await maybe_await(page.query_selector(sel))
            if container is not None:
                snippet = _clean_snippet(
                    await maybe_await(container.inner_text()))
                break
        except Exception:
            continue
    return {"title": title, "url": url, "snippet": snippet}


def _summarize_overview(query, overview_text):
    """Condense the overview into the short spoken style. None on failure."""
    try:
        from backend.services.gemini_client import ask_gemini_chat

        response = ask_gemini_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Condense the AI Overview into a short spoken "
                        "answer for a voice assistant. 2-3 sentences, dense, "
                        "factual, no filler, never invent facts. Keep key "
                        "names/dates."
                    ),
                },
                {"role": "user", "content": f"QUESTION: {query}\n\nAI OVERVIEW:\n{overview_text}"},
            ],
            temperature=0.2,
            max_tokens=200,
        )
        if not response or not response.get("choices"):
            return None
        content = response["choices"][0].get("message", {}).get("content", "").strip()
        return content or None
    except Exception as exc:
        print(f"[QUICKSEARCH] overview condense failed: {type(exc).__name__}: {exc}")
        return None


async def _answer_container_text(page):
    """Text currently rendered in the Ask-tab answer container ("" if none).

    F27: completion is judged from the ANSWER container, never from the
    disclaimer — the disclaimer is chrome and its length says nothing about
    whether the answer finished streaming. Every access is awaited.
    """
    for selector in _ASK_ANSWER_CONTAINER_SELECTORS:
        try:
            node = await maybe_await(page.query_selector(selector))
        except Exception:
            node = None
        if node is None:
            continue
        try:
            text = ((await maybe_await(node.inner_text())) or "").strip()
        except Exception:
            text = ""
        if text:
            return text
    return ""


async def _wait_for_answer_update(page, timeout=25, poll=1.0, min_stable=2):
    """Bounded completion for the Ask tab (F27).

    Replaces the old ``wait_for_load_state("networkidle")`` + disclaimer poll.
    ``networkidle`` never settles on a page whose answer is still streaming,
    so it used to burn its whole 15s timeout every time. Watch the answer
    container instead and call it done once the text stops changing.

    F27: ``asyncio.sleep`` instead of ``time.sleep`` — a blocking sleep on the
    owner loop froze every other job sharing the warm browser.
    """
    from backend.services.research_service import stop_requested

    deadline = time.monotonic() + timeout
    previous = None
    stable = 0
    while time.monotonic() < deadline:
        if stop_requested():
            return False
        text = await _answer_container_text(page)
        if text and text == previous:
            stable += 1
            if stable >= min_stable:
                return True
        else:
            stable = 1 if text else 0
        previous = text
        await asyncio.sleep(poll)
    return False


async def _wait_for_ai_completion(page, timeout=10, ask_mode=False):
    """Wait for the streaming AI answer to finish rendering.

    *ask_mode* watches the Ask-tab ANSWER container instead of the SERP
    selectors (F27: never the disclaimer). Returns True if the answer appears
    stable, False on timeout or stop (caller should extract whatever is
    there). F27: non-blocking waits and awaited page reads.
    """
    from backend.services.research_service import stop_requested

    deadline = time.monotonic() + timeout
    prev_len = -1
    while time.monotonic() < deadline:
        if stop_requested():
            return False
        text = ""
        try:
            if ask_mode:
                text = await _answer_container_text(page)
            else:
                node = await maybe_await(page.query_selector("div.chatllm-answer"))
                if node is not None:
                    text = ((await maybe_await(node.inner_text())) or "").strip()
        except Exception:
            text = ""
        cur_len = len(text)
        if cur_len > 0 and cur_len == prev_len:
            return True
        prev_len = cur_len
        time.sleep(1.5)
    return False


async def _click_ask_tab(page):
    """Click the Ask tab on the Brave Search results page.

    Returns True if a click was dispatched, False if no Ask tab was found.
    The caller must wait for navigation / AI completion separately.
    F27: the async Playwright click is awaited — an un-awaited click returns a
    coroutine that never dispatches anything.
    """
    for selector in _ASK_TAB_SELECTORS:
        try:
            btn = await maybe_await(page.query_selector(selector))
            if btn is not None:
                await maybe_await(btn.click())
                return True
        except Exception:
            continue
    return False


async def _open_google(page, query):
    """Navigate to the Brave Search results page and settle the DOM.

    Uses the same Brave Search entry point as the deep research flow
    (captcha-free, headed Chrome).  Waits for the AI answer container
    or at least a snippet element to confirm SvelteKit hydration.
    F27: navigation and the hydration wait are awaited.
    """
    await maybe_await(page.goto(google_search_url(query), timeout=60000,
                               wait_until="domcontentloaded"))
    try:
        await maybe_await(page.wait_for_selector(
            "div.chatllm-answer, div.chatllm-disclaimer, .snippet", timeout=15000))
    except Exception:
        pass


def _summarize_with_fallback_snippet(query, fallback):
    if fallback and fallback.get("title"):
        bit = _strip_markdown(fallback["snippet"] or "no snippet shown on the results page")
        return f"Sir, no AI answer was found. Top result — {fallback['title']}: {bit}"
    return (
        "Sir, I couldn't get a quick answer from the web for that. "
        "Try a deepsearch if you want me to dig through websites."
    )


def _stopped_result(query):
    return {
        "query": query,
        "spoken_summary": "Stopped the research as requested, sir.",
        "overview_found": False,
        "overview_text": None,
        "fallback": None,
        "stopped": True,
    }


def _browser_submit(fn, task_id=None, timeout=None):
    """Seam to the warm browser worker (F27). Patched in tests."""
    from backend.services import research_browser

    return research_browser.submit(fn, task_id=task_id, timeout=timeout)


def _browser_run(coro_fn, task_id=None, timeout=None):
    """F27: run a REAL async job body on the warm worker's owner loop."""
    from backend.services import research_browser

    return research_browser.run(coro_fn, task_id=task_id, timeout=timeout)


async def _quick_search_job(task, query):
    """Browser half of a quick search: EXTRACT the answer, nothing else.

    Runs ON the warm worker's owner loop, so it is the only code here that
    touches Playwright — and every page call is awaited (F27). Returns
    ``(overview, fallback)``; the spoken summary is produced afterwards, off
    the browser thread, so summarisation never runs while the browser is still
    open.

    The page is opened through ``task.new_page()`` so a timeout or a cancel
    closes exactly this job's pages and leaves the shared context (and any
    other task's pages) alone.
    """
    from backend.services.research_service import stop_requested

    if stop_requested():
        return None, None
    page = await task.new_page()
    await _open_google(page, query)
    await _wait_for_ai_completion(page)
    if stop_requested():
        return None, None
    overview = await extract_ai_overview(page)
    fallback = None
    if overview is None:
        if stop_requested():
            return None, None
        if await _click_ask_tab(page):
            # F27: answer-container updates instead of networkidle +
            # disclaimer-length polling.
            await _wait_for_answer_update(page, timeout=25)
            if stop_requested():
                return None, None
            overview = await extract_ai_overview(page, ask_mode=True, query=query)
        if overview is None:
            fallback = await extract_top_snippet(page)
    if overview:
        overview = _trim_at_disclaimer(overview)
    return overview, fallback


def run_quick_search(query):
    """Default websearch: AI Overview via Brave Search, no site scraping.

    Returns {query, spoken_summary, overview_found, overview_text,
    fallback, stopped}. Browser failures are NOT swallowed here — they
    propagate up to the caller (brain.handle_research_intent wraps them in
    try/except). Check result.get('stopped') for abort.

    F27: the page comes from the one warm research browser, so a lookup no
    longer pays a Chrome cold start — and the job body is genuinely async, so
    the owner loop stays responsive while it waits.
    """
    from backend.services.research_service import stop_requested

    if stop_requested():
        return _stopped_result(query)

    task_id = "quick-%d" % int(time.time() * 1000)
    overview, fallback = _browser_run(
        lambda task: _quick_search_job(task, query), task_id=task_id)

    if stop_requested():
        return _stopped_result(query)

    if overview:
        # F27: spoken summarisation is a separate step from extraction.
        summary = _summarize_overview(query, overview) or _clean(overview, limit=340)
        spoken = f"Sir, {_strip_markdown(summary)}"
    else:
        spoken = _summarize_with_fallback_snippet(query, fallback)
    print(f"[QUICKSEARCH] overview={'yes' if overview else 'no'} for {query!r}")
    return {
        "query": query,
        "spoken_summary": spoken,
        "overview_found": overview is not None,
        "overview_text": overview,
        "fallback": fallback,
        "stopped": False,
    }


def fetch_ai_overview_text(query):
    """Deepsearch helper: just the AI Overview text (best-effort, or None).

    F27: extraction ONLY. This used to call ``run_quick_search``, which ran
    the whole pipeline — including a Gemini spoken summary that deep research
    then threw away. It also reuses the warm browser context, so the overview
    phase and the deep-search phase share one profile instead of opening and
    closing Chrome twice.
    """
    async def _overview_only(task):
        overview_text, _fallback = await _quick_search_job(task, query)
        return overview_text

    try:
        return _browser_run(
            _overview_only,
            task_id="overview-%d" % int(time.time() * 1000))
    except Exception as exc:
        print(f"[QUICKSEARCH] overview fetch failed: {type(exc).__name__}: {exc}")
        return None
