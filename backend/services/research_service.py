"""Jarvis-native web research service.

Lives beside the deep-search experiments (research_headed / research_demo) so
Jarvis can run real "look this up" / "research" requests in the background:

  1. Brave Search on the user's real Chrome profile in HEADED mode
     (looks like a human; if a captcha ever appears the user can solve it).
  2. Scrape whatever the top-N results are — no ranking/curation by us.
  3. Gemini per-site notes, then ONE deduped consolidated "juice" summary.
  4. YouTube links get special treatment: their visible description is read
     via the meta tag; if it carries no useful content they become clickable
     "related videos" instead of dead summaries.
  5. Returns BOTH a short spoken summary (for Jarvis's voice) and a long
     markdown report (for the glass overlay + a saved file).

F04: the user's ORIGINAL question is threaded, unchanged, into every note
and synthesis call; a note counts as evidence only when it really is about
the question (shared meaningful terms, or an explicit statement that it is),
and the synthesis sections are chosen from the kind of question asked —
never release/upcoming framing unless the question is about a future release.
"""

import asyncio
import itertools
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import (parse_qs, parse_qsl, quote_plus, unquote,
                          urlencode, urlparse)

from backend.services.research_browser import maybe_await
from backend.services import jobs as job_registry

# User stop: set from the API/voice side ("stop the research"); the site
# loop checks it between fetches so a running research can be cancelled
# without killing the process (mirrors browser_agent._STOP_REQUESTED).
# F28: the source of truth is the per-job JobToken (see jobs.py); this flag
# is the legacy bridge armed by that job's cancel handler.
_STOP_REQUESTED = threading.Event()

#: job_id -> research task_id, so cancelling a job closes exactly that job's
#: browser pages and leaves every other task's context untouched (F28).
_RUNNING_TASKS = {}
_RUNNING_TASKS_LOCK = threading.RLock()
_TASK_IDS = itertools.count(1)


def request_stop():
    """Ask a running research to stop at the next site boundary."""
    _STOP_REQUESTED.set()


def clear_stop_request():
    """Clear the stop event so the next search runs normally."""
    _STOP_REQUESTED.clear()


def stop_requested():
    return _STOP_REQUESTED.is_set()


def _on_research_job_cancelled(job):
    """F28 + F20: cancelling a research job stops the loop AND closes only
    that job's pages. Another research task's browser context is untouched."""
    _STOP_REQUESTED.set()
    with _RUNNING_TASKS_LOCK:
        task_id = _RUNNING_TASKS.get(job.job_id)
    if task_id:
        from backend.services import research_browser

        research_browser.close_task_pages(task_id)


job_registry.register_stop_handler("research", _on_research_job_cancelled)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DEFAULT_PROFILE_DIR = _REPO_ROOT / "data" / "chrome_profile_jarvis"
DEFAULT_QUERY = None
DEFAULT_MAX_RESULTS = 10
DEFAULT_TEXT_LIMIT = 2500
DEFAULT_REPORTS_DIR = _REPO_ROOT / "data" / "research_reports"

GEMINI_MIN_WORDS = 20  # a YouTube description shorter than this is "no info"

# ── F28: bounded parallelism, explicit limits ─────────────────────────────
#: At most this many page fetches may be in flight at once.
MAX_CONCURRENT_FETCHES = 3
#: At most this many Gemini note-generation calls may be in flight at once.
MAX_CONCURRENT_NOTES = 2
#: Wall-clock budget for one research run (also the JobToken deadline).
DEFAULT_RESEARCH_DEADLINE = 420.0


class _AggregateGate:
    """F28 — ONE ceiling for the whole process, not per invocation.

    The audit found the fetch/note semaphores were created inside a single
    invocation, so two concurrent research runs could each run the full
    allowance and double the intended load. An ``asyncio.Semaphore`` cannot be
    shared here because every run owns its own event loop, so the ceiling is a
    plain thread-safe counter that async callers poll without ever blocking a
    loop thread.
    """

    POLL_SECONDS = 0.05

    def __init__(self, limit, name="gate"):
        self.limit = max(1, int(limit))
        self.name = name
        self._lock = threading.Lock()
        self._in_use = 0
        self._peak = 0
        self._waiting = 0

    def _try_acquire(self):
        with self._lock:
            if self._in_use < self.limit:
                self._in_use += 1
                self._peak = max(self._peak, self._in_use)
                return True
            return False

    def release(self):
        with self._lock:
            self._in_use = max(0, self._in_use - 1)

    async def __aenter__(self):
        with self._lock:
            self._waiting += 1
        try:
            while not self._try_acquire():
                await asyncio.sleep(self.POLL_SECONDS)
        finally:
            with self._lock:
                self._waiting = max(0, self._waiting - 1)
        return self

    async def __aexit__(self, *_exc):
        self.release()
        return False

    def snapshot(self):
        """Instrumentation: the numbers a test can assert ceilings with."""
        with self._lock:
            return {"name": self.name, "limit": self.limit,
                    "in_use": self._in_use, "peak": self._peak,
                    "waiting": self._waiting}


#: F28 — the process-wide ceilings, shared by every concurrent run.
FETCH_GATE = _AggregateGate(MAX_CONCURRENT_FETCHES, "fetch")
NOTE_GATE = _AggregateGate(MAX_CONCURRENT_NOTES, "note")


def concurrency_snapshot():
    """F28 — instrumented ceilings for tests and diagnostics."""
    return {"fetch": FETCH_GATE.snapshot(), "note": NOTE_GATE.snapshot()}

# ── F48: provenance ───────────────────────────────────────────────────────
# One vocabulary for every path (screen, quick search, deep research) — see
# backend/services/provenance.py. Re-exported here because the research
# pipeline and its tests have always named them on this module.
from backend.services.provenance import (  # noqa: E402  (kept beside the other F48 knobs)
    PROVENANCE_EXTERNALLY_CHECKED,
    PROVENANCE_INFERRED,
    PROVENANCE_OBSERVED,
    PROVENANCE_QUOTED,
    PROVENANCE_SECONDARY,
)

#: How much page text is retained as the claim's supporting span.
SUPPORTING_SPAN_LIMIT = 600
#: Shortest verbatim run that counts as the note quoting the page.
_QUOTE_RUN = 25
#: Shortest sentence that can corroborate the same claim in another source.
_CORROBORATION_SENTENCE = 40

# Network filter / AV block pages (Sophos Web Control etc.) must NOT be
# treated as page content — the model kept reporting "restricted by network
# category filters" as if it were the site's actual content.
_BLOCK_MARKERS = (
    "blocked by sophos", "sophos web", "sophos cloud", "web control",
    "content category filter", "category filter", "this website has been blocked",
    "has been blocked", "blocked because it has been classified", "access denied",
    "blocked by your network", "your network administrator", "blocked page",
    "requires you to verify", "cannot be shown", "unavailable at this location",
)


class SiteBlockedError(Exception):
    """The page was served by a network filter/AV block page, not the site."""


def _looks_like_block_page(title, text):
    needle = f"{title}\n{text}".lower()
    return any(marker in needle for marker in _BLOCK_MARKERS)


def _clean_text(raw, limit=DEFAULT_TEXT_LIMIT):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return "\n".join(lines)[:limit]


def _resolve_url(href):
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


def _is_youtube_url(url):
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in ("youtube.com", "youtu.be"))


def extract_brave(page, max_results=10):
    out = []
    for node in page.query_selector_all(".snippet"):
        link = node.query_selector("a.snippet-title, a.result-header, a[href]")
        if not link:
            continue
        href = _resolve_url(link.get_attribute("href"))
        if not href or not href.startswith("http"):
            continue
        desc = ""
        dn = node.query_selector(".snippet-description, .snippet-content, p")
        if dn:
            desc = dn.inner_text().strip()
        out.append({
            "title": link.inner_text().strip() or href,
            "url": href,
            "snippet": desc,
        })
        if len(out) >= max_results:
            break
    return out


async def extract_brave_async(page, max_results=10):
    """Async twin of :func:`extract_brave` for the warm worker's loop.

    ``maybe_await`` lets the same body run against real async Playwright
    pages and against the plain MagicMock pages the tests inject.
    """
    out = []
    nodes = await maybe_await(page.query_selector_all(".snippet"))
    for node in nodes or []:
        link = await maybe_await(
            node.query_selector("a.snippet-title, a.result-header, a[href]"))
        if not link:
            continue
        href = _resolve_url(await maybe_await(link.get_attribute("href")))
        if not href or not href.startswith("http"):
            continue
        desc = ""
        dn = await maybe_await(
            node.query_selector(".snippet-description, .snippet-content, p"))
        if dn:
            desc = ((await maybe_await(dn.inner_text())) or "").strip()
        out.append({
            "title": ((await maybe_await(link.inner_text())) or "").strip() or href,
            "url": href,
            "snippet": desc,
        })
        if len(out) >= max_results:
            break
    return out


_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "gclsrc", "dclid", "msclkid", "mc_cid", "mc_eid",
    "igshid", "ref_src", "ref_url", "_ga", "yclid",
}


def _canonical_url(url):
    """Normalise a URL for de-duplication (F28).

    Scheme/host casing, the "www." prefix, a trailing slash, any fragment and
    tracking query params are all irrelevant — they still point at the same
    page. Everything else in the query is kept, so ``?id=1`` and ``?id=2``
    stay distinct.
    """
    if not url:
        return ""
    try:
        parsed = urlparse(url)
    except Exception:
        return (url or "").strip().lower()
    netloc = (parsed.netloc or "").lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    try:
        pairs = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
                 if k.lower() not in _TRACKING_PARAMS]
        query = urlencode(sorted(pairs))
    except Exception:
        query = parsed.query
    path = (parsed.path or "").rstrip("/") or "/"
    canonical = "%s://%s%s" % ((parsed.scheme or "https").lower(), netloc, path)
    return "%s?%s" % (canonical, query) if query else canonical


def dedupe_results(results):
    """Drop repeated URLs, keeping the FIRST occurrence (F28).

    Result order is the ranking Brave returned; de-duplication must not
    reshuffle it, so the first (best-ranked) copy of a URL wins.
    """
    seen = set()
    out = []
    dropped = 0
    for r in results or []:
        key = _canonical_url(r.get("url"))
        if not key or key in seen:
            dropped += 1 if key else 0
            continue
        seen.add(key)
        out.append(r)
    return out, dropped


_META_JS = """() => {
    const t = document.querySelector('meta[name="description"]');
    const og = document.querySelector('meta[property="og:description"]');
    return {
        title: document.title,
        videoDescription: (t && t.content) || (og && og.content) || ""
    };
}"""

_BODY_JS = """() => {
    const clone = document.body.cloneNode(true);
    clone.querySelectorAll(
        'script,style,noscript,nav,header,footer,aside,iframe,form,button'
    ).forEach(e => e.remove());
    return clone.innerText;
}"""


def _youtube_meta_hit(meta, url):
    """True when a YouTube page's meta description alone carries the content.

    Lets both fetchers skip the (expensive) body extraction for that case.
    """
    if not _is_youtube_url(url):
        return False
    desc = (meta.get("videoDescription") or "").strip()
    return bool(desc) and len(desc.split()) >= GEMINI_MIN_WORDS


def _finish_fetch(meta, raw, url):
    """Shared tail of both fetchers: title/body handling + block-page check."""
    title = meta.get("title") or url
    is_youtube = _is_youtube_url(url)
    video_desc = (meta.get("videoDescription") or "").strip()
    if is_youtube and video_desc and len(video_desc.split()) >= GEMINI_MIN_WORDS:
        # The meta description carries real content — use it as the text.
        return {
            "title": title,
            "text": _clean_text(video_desc, limit=min(1200, DEFAULT_TEXT_LIMIT)),
            "youtube": True,
            "video_desc": True,
        }
    # else: too thin to summarise -> treated as a bare video link
    text = _clean_text(raw or "")
    if _looks_like_block_page(title, text):
        raise SiteBlockedError(f"{title[:50]}: network filter block page")
    return {
        "title": title,
        "text": text,
        "youtube": is_youtube,
        "video_desc": video_desc,
    }


def fetch_page(page, url):
    """Fetch one page. Returns dict: title, text (or ""), youtube, video_desc."""
    page.goto(url, timeout=25000, wait_until="domcontentloaded")
    page.wait_for_timeout(400)
    meta = page.evaluate(_META_JS)
    raw = None if _youtube_meta_hit(meta, url) else page.evaluate(_BODY_JS)
    return _finish_fetch(meta, raw, url)


async def fetch_page_async(page, url):
    """Async twin of :func:`fetch_page` for the bounded research pipeline.

    ``maybe_await`` keeps this usable with the plain MagicMock pages the unit
    tests inject as well as with real async Playwright pages.
    """
    await maybe_await(page.goto(url, timeout=25000, wait_until="domcontentloaded"))
    await maybe_await(page.wait_for_timeout(400))
    meta = await maybe_await(page.evaluate(_META_JS))
    raw = None
    if not _youtube_meta_hit(meta, url):
        raw = await maybe_await(page.evaluate(_BODY_JS))
    return _finish_fetch(meta, raw, url)


def _youtube_should_be_video_only(fetched, summary):
    """A YouTube result becomes a clickable 'related video' (not a summary row)
    when its description carried no real info."""
    if not fetched.get("youtube"):
        return False
    if not fetched.get("video_desc"):
        return True
    if not summary:
        return True
    return _is_no_evidence_note(summary)


def summarize_with_gemini(query, title, url, text):
    # F04: the note is extracted against the ORIGINAL question, verbatim.
    question = research_question(query)
    try:
        from backend.services.gemini_client import ask_gemini_chat

        response = ask_gemini_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are extracting evidence for ONE research question.\n"
                        "From this page extract ONLY evidence relevant to the question:\n"
                        "1) Relevant claims the page makes about the question.\n"
                        "2) Supporting quotations (short, verbatim where possible).\n"
                        "3) Dates attached to those claims (released/expected/updated).\n"
                        "4) Applicability: which versions/products/regions the claims apply to.\n"
                        "5) Unanswered parts: what the question asks that this page does NOT cover.\n"
                        "If the page has no evidence relevant to the question, reply exactly "
                        "'NO RELEVANT EVIDENCE' and nothing else.\n"
                        "If the page shares no meaningful term with the question, that is "
                        "no evidence either — say 'NO RELEVANT EVIDENCE' rather than "
                        "reporting something off-topic.\n"
                        "Never invent names, dates or quotes — only what the page states. "
                        "Return 2-5 sentences total."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"QUESTION: {question}\nTITLE: {title}\nURL: {url}\nEXCERPT:\n{text}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=260,
        )
        if not response or not response.get("choices"):
            return None
        return response["choices"][0].get("message", {}).get("content", "").strip() or None
    except Exception as exc:
        print(f"    [GEMINI] site-note failed: {type(exc).__name__}: {exc}")
        return None


def _is_no_evidence_note(summary):
    """True when a site note reports nothing relevant to the question —
    such notes are skipped before consolidation instead of forcing the final
    model to recover relevance from lossy summaries."""
    lowered = (summary or "").strip().lower()
    return lowered.startswith(("no relevant evidence", "no useful info"))


# ── F04: the question drives every note, and the synthesis structure ──────
def research_question(query):
    """The user's ORIGINAL question, unchanged, for every note/synthesis call.

    F04: the notes and the synthesis must both see the real question. It is
    never truncated, slugified or rewritten to some generic topic — a note
    extracted against "upcoming releases" when the user asked what happened in
    1989 is worse than no note at all. Every call site passes the return value
    of this function straight through, so there is exactly one place that could
    ever change the text, and it changes nothing.
    """
    if query is None:
        return ""
    return query if isinstance(query, str) else str(query)


#: Words that carry no topical signal. Token overlap is measured on what is
#: left, so "what is the price of the X" and "the price of the X is $10" still
#: match on "price".
_STOPWORDS = frozenset("""
a about above after again against all also am an and any are aren't as at be
because been before being below between both but by can can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for from
further had hadn't has hasn't have haven't having he her here hers herself him
himself his how i if in into is isn't it its itself just me more most my myself
no nor not now of off on once only or other ought our ours ourselves out over
own same shall she should shouldn't so some such than that the their theirs
them themselves then there these they this those through to too under until up
us very was wasn't we were weren't what when where which while who whom why
will with won't would wouldn't you your yours yourself yourselves
""".split())

#: A note can also be evidence because it SAYS it is, even when it happens to
#: use none of the question's own words.
_EVIDENCE_MARKERS = (
    "relevant evidence for the question",
    "evidence for the question",
    "relevant to the question",
    "relevant to this question",
    "answers the question",
    "answers this question",
    "directly answers",
    "is about the question",
)

#: A marker preceded by one of these is a DENIAL of relevance, not a claim of
#: it ("the page provides no relevant evidence for the question").
_RELEVANCE_NEGATIONS = (
    "no ", "not ", "never ", "without ", "lacks ", "lack ", "nothing ",
    "none ", "isn't ", "isnt ", "doesn't ", "doesnt ", "didn't ", "didnt ",
    "cannot ", "can't ", "cant ",
)


def question_terms(text):
    """The topical terms of *text*: lowercased tokens minus stopwords (F04)."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if len(w) > 1 and w not in _STOPWORDS}


def _declares_its_own_relevance(note):
    """True when *note* states outright that it is evidence for the question."""
    low = re.sub(r"\s+", " ", (note or "").lower())
    for marker in _EVIDENCE_MARKERS:
        start = 0
        while True:
            index = low.find(marker, start)
            if index < 0:
                break
            prefix = low[max(0, index - 24):index]
            if not any(neg in prefix for neg in _RELEVANCE_NEGATIONS):
                return True
            start = index + 1
    return False


def note_is_relevant(question, note, explicit_evidence=False):
    """F04: is *note* actually evidence for *question*?

    Wording-independent: the note must share a meaningful (non-stopword) term
    with the question, or state outright that it is evidence for it. A note
    that does neither is not evidence, however it is phrased — the old check
    only recognised the literal strings "NO RELEVANT EVIDENCE"/"no useful
    info", so every other way of being off-topic slipped through.

    *explicit_evidence* lets a caller vouch for a source it already knows
    answers the question (the pinned search overview is an answer by
    construction). A question with no meaningful terms carries nothing to
    judge against, so nothing is dropped on that basis.
    """
    text = (note or "").strip()
    if not text:
        return False
    if explicit_evidence:
        return True
    terms = question_terms(question)
    if not terms:
        return True
    if terms & question_terms(text):
        return True
    return _declares_its_own_relevance(text)


def relevance_label(question, note):
    """F04: a short, checkable "why is this evidence?" label for the report."""
    matched = sorted(question_terms(question) & question_terms(note))
    if matched:
        return "shares with the question: " + ", ".join(matched[:6])
    if _declares_its_own_relevance(note):
        return "states that it is evidence for the question"
    return "no shared terms with the question"


def relevant_evidence_items(question, items):
    """F04: only the items whose notes are evidence for *question*.

    Items with no note at all are left out of synthesis (there is no claim to
    weigh), and an item the pipeline explicitly marked as evidence is kept.
    """
    keep = []
    for item in items or []:
        note = item.get("summary")
        if not note:
            continue
        if note_is_relevant(question, note,
                            explicit_evidence=bool(item.get("explicit_evidence"))):
            keep.append(item)
    return keep


# ── F04: the synthesis structure comes from the question ──────────────────
#: The kind of question asked, in the order the detector tests for them.
QUESTION_KIND_RELEASE = "future_release"
QUESTION_KIND_TROUBLESHOOTING = "troubleshooting"
QUESTION_KIND_COMPATIBILITY = "compatibility"
QUESTION_KIND_HISTORICAL = "historical"
QUESTION_KIND_REGIONAL = "regional"
QUESTION_KIND_COMPARISON = "comparison"
QUESTION_KIND_GENERAL = "general"

QUESTION_KINDS = (
    QUESTION_KIND_RELEASE, QUESTION_KIND_TROUBLESHOOTING,
    QUESTION_KIND_COMPATIBILITY, QUESTION_KIND_HISTORICAL,
    QUESTION_KIND_REGIONAL, QUESTION_KIND_COMPARISON, QUESTION_KIND_GENERAL,
)

#: The one kind allowed to use release/upcoming framing.
RELEASE_HEADINGS = ("Released / known", "Expected / upcoming", "Trackers")

#: Headings per question kind. Nothing outside RELEASE_HEADINGS may ever be
#: handed to synthesis for a question that is not about a future release.
_SYNTHESIS_HEADINGS = {
    QUESTION_KIND_RELEASE: RELEASE_HEADINGS,
    QUESTION_KIND_TROUBLESHOOTING: ("What is happening", "Likely causes",
                                    "Fixes to try"),
    QUESTION_KIND_COMPATIBILITY: ("Compatibility", "Requirements",
                                  "Known limits"),
    QUESTION_KIND_HISTORICAL: ("Background", "What happened", "Aftermath"),
    QUESTION_KIND_REGIONAL: ("Availability", "Regional differences"),
    QUESTION_KIND_COMPARISON: ("Options", "How they differ", "Which to pick"),
    QUESTION_KIND_GENERAL: ("What the sources say", "Details", "Open questions"),
}

#: Explicit future-release cues — the ONLY reason to use release framing.
_RELEASE_CUES = (
    "next release", "upcoming", "future release", "release date", "roadmap",
    "coming soon", "not yet released", "will be released", "will launch",
    "due out", "due in", "when will", "rumour", "rumor", "rumoured", "rumored",
    "leak", "leaked", "beta", "preview", "early access", "what's new",
    "whats new", "what is new", "expected in", "expected to ship",
    "scheduled for release", "launch date", "announced for", "not out yet",
)
_TROUBLESHOOTING_CUES = (
    "error", "errors", "not working", "doesn't work", "doesnt work",
    "does not work", "won't work", "wont work", "won't start", "wont start",
    "keeps failing", "keeps crashing", "keeps stopping", "fails", "failing",
    "failure", "broken", "crash", "crashes", "crashing", "fix", "fixes",
    "troubleshoot", "trouble", "issue", "problem", "why is my",
    "how do i stop", "stuck", "freezes", "freezing", "overheating",
)
_COMPATIBILITY_CUES = (
    "works with", "work with", "working with", "compatible", "compatibility",
    "does it support", "does it work", "supported on", "support", "runs on",
    "run on", "system requirements", "requirements", "will it run",
    "is it supported", "supported",
)
_HISTORICAL_CUES = (
    "when did", "when was", "who was", "who invented", "who created",
    "what happened", "history of", "historical", "originally", "back in",
    "used to be", "in the past", "early days", "old version",
)
_REGIONAL_CUES = (
    "available in", "availability in", "availability", "which countries",
    "which country", "in my country", "outside the us", "outside the uk",
    "region", "regional", "worldwide", "in the eu", "in europe",
)
_COMPARISON_CUES = (
    " vs ", "vs.", "versus", "compare", "comparison", "difference between",
    "better than", "differences between",
)
#: Places and regions named after "in" ("in Japan") make a question regional.
_PLACE_NAMES = (
    "afghanistan", "albania", "algeria", "argentina", "australia", "austria",
    "bangladesh", "belgium", "brazil", "bulgaria", "canada", "chile", "china",
    "colombia", "croatia", "czech republic", "czechia", "denmark", "egypt",
    "estonia", "finland", "france", "germany", "ghana", "greece", "hong kong",
    "hungary", "iceland", "india", "indonesia", "iran", "iraq", "ireland",
    "israel", "italy", "japan", "kenya", "korea", "kuwait", "latvia",
    "lithuania", "luxembourg", "malaysia", "mexico", "morocco", "netherlands",
    "new zealand", "nigeria", "norway", "pakistan", "peru", "philippines",
    "poland", "portugal", "qatar", "romania", "russia", "saudi arabia",
    "singapore", "slovakia", "slovenia", "south africa", "spain", "sweden",
    "switzerland", "taiwan", "thailand", "turkey", "ukraine", "united arab emirates",
    "united kingdom", "united states", "vietnam", "europe", "asia", "africa",
    "oceania",
)


def _has_cue(text, cues):
    """Word-boundary cue match (so "error" never matches inside "terror")."""
    for cue in cues:
        pattern = r"\b" + re.escape(cue)
        if cue[-1].isalnum():
            pattern += r"\b"
        if re.search(pattern, text):
            return True
    return False


def _is_regional_question(low):
    if _has_cue(low, _REGIONAL_CUES):
        return True
    for place in _PLACE_NAMES:
        if re.search(r"\bin\s+(?:the\s+)?" + re.escape(place) + r"\b", low):
            return True
    return False


def detect_question_kind(question):
    """F04: what kind of question is this? Drives the synthesis headings.

    Checks run strongest-cue-first so a troubleshooting question that happens
    to name a country still reads as troubleshooting, and a bare year only
    counts as historical when nothing more specific claims it.
    """
    low = " " + re.sub(r"\s+", " ", research_question(question).lower()).strip() + " "
    if _has_cue(low, _RELEASE_CUES):
        return QUESTION_KIND_RELEASE
    if _has_cue(low, _TROUBLESHOOTING_CUES):
        return QUESTION_KIND_TROUBLESHOOTING
    if _has_cue(low, _COMPATIBILITY_CUES):
        return QUESTION_KIND_COMPATIBILITY
    if _has_cue(low, _HISTORICAL_CUES):
        return QUESTION_KIND_HISTORICAL
    if _is_regional_question(low):
        return QUESTION_KIND_REGIONAL
    if _has_cue(low, _COMPARISON_CUES):
        return QUESTION_KIND_COMPARISON
    if re.search(r"\bin\s+(1[5-9]\d\d|20\d\d)\b", low):
        return QUESTION_KIND_HISTORICAL
    return QUESTION_KIND_GENERAL


def synthesis_sections(kind):
    """F04: the section headings synthesis must use for *kind*."""
    return tuple(_SYNTHESIS_HEADINGS.get(kind, _SYNTHESIS_HEADINGS[QUESTION_KIND_GENERAL]))


def question_sections(question):
    """F04: the section headings synthesis must use for *question*."""
    return synthesis_sections(detect_question_kind(question))


def synthesis_system_prompt(question):
    """F04: synthesis instructions whose STRUCTURE comes from the question.

    The headings are the question's own sections (troubleshooting question ->
    causes and fixes; historical question -> background and aftermath), and
    release/upcoming framing is only ever described for a question that is
    actually about a future release.
    """
    kind = detect_question_kind(question)
    headings = "\n".join("  ## %s" % section for section in synthesis_sections(kind))
    lines = [
        "You are a research synthesis engine. You receive notes from "
        "multiple websites about the same topic. Produce ONE master summary "
        "that ANSWERS the research question.",
        "- Keep every unique fact; do NOT repeat what several sites agree on "
        "(state it once).",
        "- Distinguish what is CONFIRMED vs EXPECTED/RUMORED.",
        "- Organise the answer under EXACTLY these markdown headings, in this "
        "order:",
        headings,
        "- Dense, factual, no filler.",
        "- End with a '## Bottom line' paragraph: the most important answer "
        "for someone asking this exact question.",
    ]
    if kind == QUESTION_KIND_RELEASE:
        lines.append(
            "- This question IS about an upcoming release, so the "
            "released/expected split above is the right shape for it.")
    else:
        lines.append(
            "- This question is NOT about a future or upcoming release: do NOT "
            "use release-cycle framing or next-version sections.")
    lines.append(
        "- If the notes do not carry enough evidence to answer the question, "
        "say so plainly in the relevant section instead of inventing facts or "
        "padding the structure with guesses.")
    # F48: provenance survives synthesis.
    lines.append(
        "- Respect provenance. 'observed'/'quoted' means a page we fetched "
        "actually said it. 'secondary' means an AI overview — it is NOT "
        "independent corroboration, so never cite it as a second source.")
    lines.append(
        "- Say how sure you are: mark single-source claims as unverified and "
        "say when sources disagree. Never present an AI overview as checking "
        "anything.")
    return "\n".join(lines)


# ── F48: provenance ───────────────────────────────────────────────────────
def _sentences(text):
    """Normalised sentences long enough to corroborate a claim."""
    out = []
    for chunk in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        norm = re.sub(r"[^a-z0-9 ]+", " ", chunk.lower())
        norm = re.sub(r"\s+", " ", norm).strip()
        if len(norm) >= _CORROBORATION_SENTENCE:
            out.append(norm)
    return out


def _quotes_page(note, span):
    """True when the note lifts a verbatim run out of the page (F48 quoted).

    Compares word runs, not raw substrings, so reflowed whitespace and
    repunctuation still count as the same quotation.
    """
    note_words = re.findall(r"[a-z0-9]+", (note or "").lower())
    span_words = re.findall(r"[a-z0-9]+", (span or "").lower())
    if len(note_words) < _QUOTE_RUN or len(span_words) < _QUOTE_RUN:
        return False
    span_runs = set()
    for i in range(len(span_words) - _QUOTE_RUN + 1):
        span_runs.add(" ".join(span_words[i:i + _QUOTE_RUN]))
    for i in range(len(note_words) - _QUOTE_RUN + 1):
        if " ".join(note_words[i:i + _QUOTE_RUN]) in span_runs:
            return True
    return False


def build_evidence_item(result, fetched, summary, retrieved_at=None, question=None):
    """One evidence row with explicit provenance (F48).

    Every claim keeps where it came from (``source_url``), the span of the
    page that supports it (``supporting_span``), when it was retrieved
    (``retrieved_at``) and how sure we are (``uncertainty``).

    F04: when *question* is supplied the row also carries ``relevance`` — the
    terms it shares with the question (or the fact that it declared itself
    evidence), so a reader can check the inclusion decision instead of
    trusting it.
    """
    url = result.get("url") or ""
    span = (fetched.get("text") or "")[:SUPPORTING_SPAN_LIMIT]
    note = summary or ""
    item = {
        "result_title": result.get("title", url),
        "url": url,
        "snippet": result.get("snippet", ""),
        "page_title": fetched.get("title") or "",
        "text_excerpt": (fetched.get("text") or "")[:800],
        "summary": note,
        # ── F48 provenance ──
        "provenance": (PROVENANCE_QUOTED if _quotes_page(note, span)
                       else PROVENANCE_OBSERVED),
        "source_url": url,
        "supporting_span": span,
        "retrieved_at": retrieved_at or datetime.now().isoformat(timespec="seconds"),
        "uncertainty": "unverified",
    }
    if question is not None:
        item["relevance"] = relevance_label(question, note)
    return item


def build_overview_item(query, overview):
    """The pinned AI Overview as a SECONDARY source (F48).

    The overview is an AI summary of other people's pages, so it is not
    independent corroboration of anything — and it must not be labelled as if
    it were a first-party source (it used to be reported as "Google AI
    Overview", which invented provenance it never had).
    """
    from backend.services.quick_search import google_search_url

    url = google_search_url(query)
    title = "Brave Search AI Overview (secondary source)"
    return {
        "result_title": title,
        "url": url,
        "snippet": "",
        "page_title": title,
        "text_excerpt": overview[:800],
        "summary": _clean_text(overview, limit=1200),
        "provenance": PROVENANCE_SECONDARY,
        "source_url": url,
        "supporting_span": overview[:SUPPORTING_SPAN_LIMIT],
        "retrieved_at": datetime.now().isoformat(timespec="seconds"),
        "uncertainty": ("AI-generated summary — secondary source, "
                        "not independent corroboration"),
        # F04: the search engine's overview IS an answer to the question, so
        # it is evidence by construction — it never has to pass the term
        # overlap check.
        "explicit_evidence": True,
        "relevance": "the search engine's own answer to the question",
    }


def annotate_corroboration(items):
    """Fill each item's *uncertainty* from how many OTHER sources repeat it.

    F48: an AI overview is a secondary source — it never corroborates another
    source and is never counted as corroborated itself.
    """
    primary = [it for it in items if it.get("provenance") != PROVENANCE_SECONDARY]
    sets = [set(_sentences(it.get("supporting_span") or it.get("text_excerpt") or ""))
            for it in primary]
    for i, item in enumerate(primary):
        others = sum(1 for j, other in enumerate(primary)
                     if j != i and (sets[i] & sets[j]))
        item["uncertainty"] = (
            "corroborated by %d other source%s" % (others, "" if others == 1 else "s")
            if others else "single source — not independently verified"
        )
    for item in items:
        item.setdefault("uncertainty", "unverified")
    return items


def attach_claim_provenance(items, question=""):
    """F48 — give every research item the shared CLAIM contract.

    The audit found the synthesis path carrying prose and a provenance word,
    with no way for a reader (or the overlay) to check what actually supports
    the claim. Each item is rebuilt through ``provenance.build_claim`` so it
    carries validated verbatim spans, the lookup it came from, an independent
    corroboration verdict and an aware UTC observation time — and the shared
    vocabulary is the same one the screen path and quick search use.
    """
    from backend.services.provenance import (
        build_claim, claim_to_evidence, utc_now_iso,
    )

    observed_at = utc_now_iso()
    for item in items or []:
        try:
            summary = item.get("summary") or item.get("snippet") or ""
            span_text = (item.get("supporting_span")
                         or item.get("text_excerpt") or "")
            sources = []
            if span_text:
                sources.append({
                    "url": item.get("url") or item.get("source_url") or "",
                    "quote": span_text,
                    "publisher": item.get("page_title")
                                 or item.get("result_title") or "",
                    "result_title": item.get("result_title") or "",
                    "retrieved_at": item.get("retrieved_at") or "",
                })
            others = [it for it in items
                      if it is not item and (it.get("summary") or "").strip()]
            for other in others[:4]:
                sources.append({
                    "url": other.get("url") or other.get("source_url") or "",
                    "quote": (other.get("supporting_span")
                              or other.get("text_excerpt") or ""),
                    "result_title": other.get("result_title") or "",
                    "retrieved_at": other.get("retrieved_at") or "",
                })
            lookup = {}
            if item.get("url"):
                lookup = {"query": question,
                          "url": item.get("url"),
                          "engine": "brave",
                          "retrieved_at": item.get("retrieved_at") or ""}
            claim = build_claim(
                summary,
                sources=sources,
                provenance=item.get("provenance") or PROVENANCE_OBSERVED,
                lookup=lookup,
                observed_at=observed_at,
            )
            claim_evidence = claim_to_evidence(
                claim,
                title=item.get("result_title") or item.get("page_title") or "",
                url=item.get("url") or item.get("source_url") or "")
            # Only the claim-contract fields are merged; the row keeps the
            # richer research fields (excerpts, relevance, failure data) it
            # already had.
            for key in ("spans", "lookup", "corroboration", "observed_at",
                        "uncertainty", "provenance"):
                value = claim_evidence.get(key)
                if value not in (None, "", [], {}):
                    item[key] = value
            item["claim"] = claim.text
            item["query"] = question
        except Exception as exc:  # pragma: no cover - never break a report
            logging.debug("[RESEARCH] claim provenance failed: %s", exc)
    return items


def consolidate_summaries(query, items):
    """Blend all per-site notes into ONE deduped master summary (the juice).

    F04: only notes that are actually evidence for the question reach the
    synthesis model, and the section headings come from the question itself —
    a troubleshooting question gets causes and fixes, a historical question
    gets background and aftermath, and nothing gets release/upcoming framing
    unless the question really is about a future release. If no note qualifies
    (or none carries a claim) there is not enough evidence for a synthesis and
    this returns None instead of a hollow structure.
    """
    question = research_question(query)
    # F04: the relevance gate runs here as well as in the site pipeline, so an
    # off-topic note can never reach synthesis however the items were built.
    items = relevant_evidence_items(question, items)
    # F48: provenance travels with the note into synthesis — the model must
    # see how each claim was obtained, not just the claim.
    entries = "\n\n".join(
        f"SITE {i}: {it['result_title']}\nURL: {it['url']}\n"
        f"PROVENANCE: {it.get('provenance') or PROVENANCE_OBSERVED}\n"
        f"CONFIDENCE: {it.get('uncertainty') or 'unverified'}\n"
        f"NOTE: {it['summary']}"
        for i, it in enumerate(items, 1)
        if it.get("summary")
    )
    if not entries.strip():
        return None
    try:
        from backend.services.gemini_client import ask_gemini_chat

        response = ask_gemini_chat(
            [
                {"role": "system", "content": synthesis_system_prompt(question)},
                {"role": "user", "content": f"QUESTION: {question}\n\n{entries}"},
            ],
            temperature=0.3,
            max_tokens=1300,
        )
        if not response or not response.get("choices"):
            return None
        content = response["choices"][0].get("message", {}).get("content", "").strip()
        return content or None
    except Exception as exc:
        print(f"  [GEMINI-CONSOLIDATE] failed: {type(exc).__name__}: {exc}")
        return None


def build_spoken_summary(query, consolidated, videos, sites_visited):
    """Short 3-4 sentence spoken reply. No extra LLM call — from the bottom line."""
    bottom = ""
    m = re.search(r"##\s*Bottom line[^\n]*\n(.*)", consolidated or "", flags=re.DOTALL)
    if m:
        bottom = re.sub(r"[#*`\n]+", " ", m.group(1)).strip()
    if not bottom or len(bottom) < 40:
        top = consolidated or ""
        first_para = re.split(r"\n\s*\n", top)
        bottom = re.sub(r"[#*`\n]+", " ", (first_para[0] if first_para else top)).strip()

    if len(bottom) > 380:
        bottom = bottom[:377].rsplit(" ", 1)[0] + "..."

    spoken = (
        f"Sir, I gathered information from {sites_visited} sources. "
        f"In simple words, {bottom} "
        "For the full detailed research, read the report that is now on your screen."
    )
    if videos:
        spoken += (
            f" I also found {len(videos)} related video"
            f"{'s' if len(videos) > 1 else ''} on YouTube — they are on your screen "
            "and clickable if you want to watch them."
        )
    return spoken


def build_details_markdown(query, consolidated, related_videos, items, failures, counts):
    lines = [f"# Research: {query}", "",
             f"- Time: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
             f"Sites visited: {counts['visited']} | Failed: {counts['failed']}", ""]
    if related_videos:
        lines += ["## ▶️ Related videos (click to watch)", ""]
        for v in related_videos:
            lines.append(f"- [{v['title']}]({v['url']})")
        lines.append("")
    if consolidated:
        lines += [
            "## The juice (one combined answer)", "", consolidated, "",
            "_Provenance: inferred by synthesis from the sources below. It is not "
            "itself an observation._", "",
        ]
    lines.append("## Sources, one note each")
    lines.append("")
    for i, it in enumerate(items, 1):
        lines.append(f"### {i}. {it['result_title']}")
        lines.append(f"- URL: {it['url']}")
        if it.get("page_title") and it["page_title"] != it["result_title"]:
            lines.append(f"- Page: {it['page_title']}")
        # F48: how this claim was obtained, and how sure we are.
        if it.get("provenance"):
            lines.append(f"- Provenance: {it['provenance']}")
        if it.get("uncertainty"):
            lines.append(f"- Confidence: {it['uncertainty']}")
        # F04: why this note counts as evidence for the question.
        if it.get("relevance"):
            lines.append(f"- Relevance: {it['relevance']}")
        if it.get("retrieved_at"):
            lines.append(f"- Retrieved: {it['retrieved_at']}")
        if it.get("summary"):
            lines.append(f"- Note: {it['summary']}")
        if it.get("supporting_span"):
            lines.append(f"- Supporting span: {it['supporting_span'][:300]}")
        lines.append("")
    if failures:
        lines.append("## Failures")
        for f in failures:
            lines.append(f"- {f.get('title')} | {f.get('url')} | {f.get('error', '')}")
        lines.append("")
    return "\n".join(lines)



def _emit(on_progress, message):
    """F28: progress is wired immediately — every stage reports as it starts."""
    if not on_progress:
        return
    try:
        on_progress(message)
    except Exception:
        pass


def _publish(on_evidence, item):
    """F28: publish useful evidence the moment it exists, not at the end."""
    if not on_evidence:
        return
    try:
        on_evidence(item)
    except Exception:
        pass


def _stop_now(job=None, deadline=None):
    """Cancelled, or out of time? Checked before every fetch and every note."""
    if job is not None and job.should_stop():
        return True
    if stop_requested():
        return True
    return deadline is not None and time.monotonic() >= deadline


def _cancelled_now(job=None):
    """Only an EXPLICIT cancel is 'stopped' (F28).

    A deadline ends the run early but keeps whatever was gathered — reporting
    that as "stopped by the user" would tell the user we threw their results
    away when we did not.
    """
    if job is not None and job.cancelled:
        return True
    return stop_requested()


def _browser_run(coro_fn, task_id=None, timeout=None, profile_dir=None, channel=None):
    """Seam to the warm browser worker (F27/F28). Patched in tests."""
    from backend.services import research_browser

    return research_browser.run(coro_fn, task_id=task_id, timeout=timeout,
                                profile_dir=profile_dir, channel=channel)


async def _research_job(task, query, max_results, job, deadline,
                        on_progress, on_evidence):
    """Browser phase: search once, then fetch + note with bounded parallelism.

    F28: at most MAX_CONCURRENT_FETCHES page fetches and MAX_CONCURRENT_NOTES
    note-generation calls are ever in flight. ``asyncio.gather`` returns in
    SUBMISSION order, so the final report keeps the ranking Brave gave us no
    matter how the work interleaves.
    """
    def emit(message):
        _emit(on_progress, message)

    # F04: the question the user actually asked, verbatim — every note call
    # below is made against this, never against a tidied-up topic string.
    question = research_question(query)

    collected = []
    failures = []
    related_videos = []

    page = await task.new_page()
    try:
        search_url = f"https://search.brave.com/search?q={quote_plus(question)}"
        emit(f'Searching: "{question}" on Brave')
        emit("If a captcha/challenge appears in the window, solve it — I'll continue.")
        await maybe_await(page.goto(search_url, timeout=60000,
                                    wait_until="domcontentloaded"))
        try:
            await maybe_await(page.wait_for_selector(".snippet, #results", timeout=180000))
        except Exception:
            pass
        results = await extract_brave_async(page, max_results=max_results)
    finally:
        try:
            await maybe_await(page.close())
        except Exception:
            pass

    emit(f"Got {len(results)} results from Brave")
    results, dropped = dedupe_results(results)
    if dropped:
        emit(f"Skipped {dropped} duplicate result(s)")
    if _stop_now(job, deadline):
        return {"collected": [], "failures": failures,
                "related_videos": [], "stopped": _cancelled_now(job)}

    fetch_gate = FETCH_GATE
    note_gate = NOTE_GATE
    total = len(results)

    def _remaining():
        """Seconds left in this run's deadline (F28)."""
        return max(0.0, deadline - time.monotonic())

    async def _one(index, r):
        if _stop_now(job, deadline):
            return None
        emit(f"[{index}/{total}] Fetching {r['url']}")
        fetched = None
        for attempt in (1, 2):
            if _stop_now(job, deadline):
                return None
            tab = None
            try:
                # Bounded: the PROCESS ceiling, not this invocation's.
                async with fetch_gate:
                    # F28 — recheck AFTER admission: a waiter that was
                    # cancelled (or whose deadline expired) while it queued
                    # must start no I/O at all.
                    if _stop_now(job, deadline):
                        return None
                    tab = await task.new_page()
                    # A fetch that outlives the deadline must not keep
                    # running in the background: it is bounded by the time
                    # this run actually has left.
                    fetched = await asyncio.wait_for(
                        fetch_page_async(tab, r["url"]),
                        timeout=max(1.0, _remaining()),
                    )
                break
            except asyncio.TimeoutError:
                # Out of time. No retry: the budget is the budget, and the
                # evidence already gathered survives.
                failures.append({
                    "index": index, "title": r.get("title"), "url": r["url"],
                    "error": "timed out after %.0fs" % _remaining(),
                })
                print("    FAILED to fetch: timeout")
                return None
            except Exception as exc:
                message = str(exc)
                # TLS/network blockers won't clear on retry — skip at once
                # (previously the browser kept hammering the dead domain).
                is_hard_fail = (
                    isinstance(exc, SiteBlockedError)
                    or "ERR_CERT" in message
                    or "CERT_AUTHORITY_INVALID" in message
                )
                # F28 — a hard failure runs EXACTLY once: record it and stop.
                # The old loop fell through to attempt 2 and appended the same
                # source a second time.
                if is_hard_fail:
                    print(f"    FAILED to fetch: {type(exc).__name__}: {message[:90]}")
                    failures.append({
                        "index": index, "title": r.get("title"),
                        "url": r["url"], "error": message[:140],
                    })
                    break
                if attempt == 1:
                    await asyncio.sleep(4)
                    # The wait above is where a cancel usually lands.
                    if _stop_now(job, deadline):
                        return None
                else:
                    print(f"    FAILED to fetch: {type(exc).__name__}: {message[:90]}")
                    failures.append({
                        "index": index, "title": r.get("title"),
                        "url": r["url"], "error": message[:140],
                    })
                    break
            finally:
                if tab is not None:
                    try:
                        await maybe_await(tab.close())
                    except Exception:
                        pass
        if fetched is None:
            return None

        title, text = fetched["title"], fetched["text"]

        if fetched.get("youtube") and not fetched.get("video_desc"):
            emit("  → YouTube, no usable description — listed as a related video.")
            return ("video", {"title": title, "url": r["url"]})

        summary = None
        if text:
            if _stop_now(job, deadline):
                return ("evidence", build_evidence_item(
                    r, fetched, None, question=question))
            # Bounded: the process-wide note ceiling, and rechecked after
            # admission so a cancelled note never reaches the model.
            async with note_gate:
                if _stop_now(job, deadline):
                    return ("evidence", build_evidence_item(
                        r, fetched, None, question=question))
                summary = await asyncio.to_thread(
                    summarize_with_gemini, question, title, r["url"], text)
            emit(f"  note: {(summary or '')[:90]}")

        # YouTube description that carried no real info -> related video chip
        # (user can click it if curious), instead of a dead summary row.
        if _youtube_should_be_video_only(fetched, summary):
            emit("  → YouTube note was not useful — listed as a related video.")
            return ("video", {"title": title, "url": r["url"]})

        # Notes that report no evidence relevant to the question are skipped
        # BEFORE consolidation — the synthesis model must never have to
        # recover relevance from lossy irrelevant summaries.
        if _is_no_evidence_note(summary):
            emit("  → no relevant evidence for the question — skipped.")
            return None

        # F04: the same rule for every other way of being off-topic. A note
        # has to share a meaningful term with the question, or say outright
        # that it is evidence for it — recognising only the literal
        # "NO RELEVANT EVIDENCE" phrasing let any other off-topic note through.
        if summary and not note_is_relevant(question, summary):
            emit("  → note is not about this question — skipped.")
            return None

        item = build_evidence_item(r, fetched, summary, question=question)
        _publish(on_evidence, item)
        return ("evidence", item)

    outcomes = await asyncio.gather(
        *(_one(i, r) for i, r in enumerate(results, 1)),
        return_exceptions=True,
    )
    # F28 — every source keeps an ORDERED outcome: the ranking the search
    # engine gave us survives, and a task that blew up becomes a recorded
    # failure instead of silently vanishing from the report.
    for index, (outcome, r) in enumerate(zip(outcomes, results), 1):
        if isinstance(outcome, BaseException):
            print(f"    [RESEARCH] site pipeline error: "
                  f"{type(outcome).__name__}: {outcome}")
            failures.append({
                "index": index, "title": r.get("title"), "url": r.get("url"),
                "error": "%s: %s" % (type(outcome).__name__,
                                     str(outcome)[:120]),
            })
            continue
        if not outcome:
            continue
        kind, payload = outcome
        if kind == "evidence":
            collected.append(payload)
        elif kind == "video":
            related_videos.append(payload)

    return {
        "collected": collected,
        "failures": failures,
        "related_videos": related_videos,
        # A deadline stops the run early but keeps what was gathered; only an
        # explicit cancel produces the "stopped" result.
        "stopped": _cancelled_now(job),
    }


def run_research(
    query,
    profile_dir=None,
    max_results=DEFAULT_MAX_RESULTS,
    reports_dir=None,
    on_progress=None,
    pinned_overview=None,
    on_evidence=None,
    job=None,
    timeout=None,
):
    """Run the headed-Chrome research flow for *query*.

    Returns: {query, spoken_summary, detailed_markdown, related_videos,
              report_path, visited, failed, stopped}

    *pinned_overview* (deepsearch mode): the search engine's AI Overview
    fetched beforehand — included as a SECONDARY source (F48), never as a
    first-party one and never as corroboration.

    *on_progress* fires as each stage starts (F28: it used to be supplied by
    nobody at all). *on_evidence* fires the moment a usable source note
    exists, so the UI can show partial results instead of waiting for the
    whole run.

    *job* is an optional jobs.JobToken; when omitted a fresh one is created
    with *timeout* as its deadline, so this run can be cancelled on its own.
    Cancelling closes this run's pages and nothing else (F28).
    """
    # F20: the legacy flag stays armed while ANOTHER live research job is
    # already cancelled — otherwise starting run B would disarm the stop the
    # user just issued for run A.
    if not any(j.should_stop() for j in job_registry.live_jobs(kind="research")):
        clear_stop_request()
    # F04: one question, captured once, threaded through search, notes,
    # synthesis and the report unchanged.
    question = research_question(query)
    profile_path = Path(profile_dir or os.getenv("JARVIS_RESEARCH_PROFILE") or DEFAULT_PROFILE_DIR)
    reports = Path(reports_dir or os.getenv("JARVIS_RESEARCH_REPORTS") or DEFAULT_REPORTS_DIR)
    channel = os.getenv("JARVIS_RESEARCH_CHANNEL", "chrome")

    _emit(on_progress, f"Opening Chrome on your profile to research: {question}")

    owns_job = job is None
    if owns_job:
        # The label is display-only (job registry / UI) — the question itself
        # is never shortened anywhere a note or the synthesis can see it.
        job = job_registry.new_job(
            kind="research", timeout=timeout or DEFAULT_RESEARCH_DEADLINE,
            label=question[:80])
    task_id = "research-%d" % next(_TASK_IDS)
    with _RUNNING_TASKS_LOCK:
        _RUNNING_TASKS[job.job_id] = task_id
    deadline = job.deadline or (time.monotonic() + (timeout or DEFAULT_RESEARCH_DEADLINE))

    try:
        _emit(on_progress, "Reusing the warm research browser…")
        outcome = _browser_run(
            lambda task: _research_job(task, question, max_results, job, deadline,
                                       on_progress, on_evidence),
            task_id=task_id,
            # F28 — the browser job gets exactly the time the run has left. A
            # 10s floor here let a job outlive its own deadline (the audit's
            # "timed-out work can continue").
            timeout=max(0.0, deadline - time.monotonic()),
            profile_dir=str(profile_path),
            channel=channel,
        )
    finally:
        with _RUNNING_TASKS_LOCK:
            _RUNNING_TASKS.pop(job.job_id, None)
        if owns_job:
            job.finish()

    collected = outcome["collected"]
    failures = outcome["failures"]
    related_videos = outcome["related_videos"]

    if outcome.get("stopped"):
        _emit(on_progress, "Stopped the research as requested.")
        print("[RESEARCH] stopped by user mid-scrape")
        return {
            "query": query,
            "spoken_summary": "Stopped the research as requested, sir.",
            "detailed_markdown": "",
            "related_videos": [],
            "report_path": None,
            "evidence": [],
            "visited_count": len(collected),
            "failed_count": len(failures),
            "stopped": True,
        }

    # Deepsearch: the AI Overview rides along as the first note so the
    # consolidated answer (and the report) cover it alongside the sites —
    # flagged as a secondary source, never as corroboration (F48).
    report_items = list(collected)
    if pinned_overview and pinned_overview.strip():
        report_items.insert(0, build_overview_item(question, pinned_overview))

    annotate_corroboration(report_items)
    # F48: the synthesis contract — every note becomes a CLAIM with validated
    # verbatim spans, its lookup reference, an independent-corroboration verdict
    # and an aware UTC timestamp, so the report/overlay can show what supports
    # what instead of a bare paragraph.
    attach_claim_provenance(report_items, question)

    consolidated = consolidate_summaries(question, report_items) if report_items else None
    spoken = build_spoken_summary(question, consolidated, related_videos, len(collected)) if (consolidated or related_videos) else (
        "Sir, I could not gather useful information on that. Check the window I opened — "
        "you may need to solve a challenge or I hit too many blocked sites."
    )

    detailed = build_details_markdown(
        question, consolidated, related_videos, report_items, failures,
        {"visited": len(collected), "failed": len(failures)},
    )

    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:50] or "research"
    reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_file = reports / f"{slug}-{stamp}.md"
    report_file.write_text(detailed, encoding="utf-8")

    return {
        "query": query,
        "spoken_summary": spoken,
        "detailed_markdown": detailed,
        "related_videos": related_videos,
        "report_path": str(report_file),
        "evidence": report_items,
        "visited_count": len(collected),
        "failed_count": len(failures),
        "stopped": False,
    }
