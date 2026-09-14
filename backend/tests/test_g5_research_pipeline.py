"""Fable-5 audit G5 — research pipeline.

Covers:
  * F48 — provenance (observed / quoted / inferred / secondary), source URLs,
    supporting spans, retrieval time, and "an AI overview is NOT
    corroboration".
  * F28 — bounded parallel worker: at most three fetches and two note calls
    in flight, original result order preserved, URL de-duplication, job
    deadline enforced, progress wired immediately, evidence published
    incrementally, cancellation scoped to the task's own pages.
  * F27 — one long-lived browser owner: the context is reused across jobs and
    each task's pages are closed without touching the shared context.
  * F04 — the question is threaded into every note request (landed in G0,
    pinned here so the pipeline can never regress it). The rest of F04 —
    relevance-by-meaning filtering and question-shaped synthesis headings —
    is pinned in ``test_f04_research_question.py``.

Every browser and LLM call is faked — these tests never launch Chrome.
"""

import asyncio
import inspect
import os
import threading
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services import quick_search
from backend.services import research_browser
from backend.services import research_service
from backend.services import screen_analyzer


def _await(value):
    """F27: the quick-search browser helpers are async now.

    Every page access is awaited, so a test that drives them directly has to
    run the coroutine instead of comparing it to a value.
    """
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


# ── fakes for the warm browser worker (F27) ───────────────────────────────
class FakePage:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class FakeContext:
    def __init__(self):
        self.pages = []
        self.closed = False

    async def new_page(self):
        page = FakePage()
        self.pages.append(page)
        return page

    async def close(self):
        self.closed = True


class FakeChromium:
    def __init__(self, context):
        self.context = context
        self.launches = 0

    async def launch_persistent_context(self, **kwargs):
        self.launches += 1
        return self.context


class FakePlaywright:
    def __init__(self, context):
        self.chromium = FakeChromium(context)
        self.stopped = False

    async def stop(self):
        self.stopped = True


class FakeAsyncPlaywright:
    """Stands in for ``playwright.async_api.async_playwright()``."""

    def __init__(self, context=None):
        self.context = context or FakeContext()
        self.playwright = None

    async def start(self):
        self.playwright = FakePlaywright(self.context)
        return self.playwright


class FakeAskNode:
    def __init__(self, text):
        self._text = text

    def inner_text(self):
        return self._text


class FakeAskPage:
    """Ask-tab page whose answer container returns a scripted text series."""

    def __init__(self, texts):
        self._texts = list(texts)

    def query_selector(self, selector):
        if selector != "div.message.assistant.llm-output":
            return None
        value = self._texts.pop(0) if self._texts else ""
        return FakeAskNode(value)


def fake_task():
    """Minimal ResearchTask stand-in for driving _research_job directly."""
    task = MagicMock()

    async def _new_page():
        return MagicMock()

    async def _close_pages():
        return None

    task.new_page = _new_page
    task.close_pages = _close_pages
    return task


def result(url, title=None):
    return {"url": url, "title": title or url, "snippet": "snippet for " + url}


# ── F48: provenance ───────────────────────────────────────────────────────
class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        research_service.clear_stop_request()

    def test_page_we_fetched_ourselves_is_observed(self):
        item = research_service.build_evidence_item(
            result("https://example.com/a"),
            {"title": "A", "text": "The launch date was announced in March."},
            "The launch happened in April, roughly a month later.",
        )
        self.assertEqual(item["provenance"], research_service.PROVENANCE_OBSERVED)

    def test_note_quoting_the_page_is_quoted(self):
        span = (" ".join("word%d" % i for i in range(60)))
        item = research_service.build_evidence_item(
            result("https://example.com/a"),
            {"title": "A", "text": span},
            "The page says: " + span[:200],
        )
        self.assertEqual(item["provenance"], research_service.PROVENANCE_QUOTED)

    def test_evidence_keeps_source_url_span_and_retrieval_time(self):
        item = research_service.build_evidence_item(
            result("https://example.com/a"),
            {"title": "A", "text": "x" * 5000},
            "a note",
            retrieved_at="2026-09-09T10:00:00",
        )
        self.assertEqual(item["source_url"], "https://example.com/a")
        self.assertEqual(item["retrieved_at"], "2026-09-09T10:00:00")
        self.assertTrue(item["supporting_span"])
        self.assertLessEqual(
            len(item["supporting_span"]), research_service.SUPPORTING_SPAN_LIMIT)

    def test_ai_overview_is_a_secondary_source(self):
        item = research_service.build_overview_item("best tanks", "An overview.")
        self.assertEqual(item["provenance"], research_service.PROVENANCE_SECONDARY)
        # It must not claim provenance it never had.
        self.assertNotIn("Google AI Overview", item["result_title"])
        self.assertIn("secondary", item["result_title"].lower())
        self.assertIn("not independent corroboration", item["uncertainty"])

    def test_overview_never_corroborates_a_real_source(self):
        span = ("The upgrade programme was confirmed in March and the first "
                "units are expected to arrive before the end of the year.")
        first = research_service.build_evidence_item(
            result("https://a.com"), {"title": "A", "text": span}, "note a")
        second = research_service.build_evidence_item(
            result("https://b.com"), {"title": "B", "text": span}, "note b")
        overview = research_service.build_overview_item("tanks", span)

        research_service.annotate_corroboration([first, second, overview])

        # Two real sources repeating each other: one corroboration each.
        self.assertIn("corroborated by 1 other source", first["uncertainty"])
        # The overview copies the same text but adds nothing: it is a
        # secondary restatement, so it is NOT counted.
        self.assertNotIn("corroborated by 2", first["uncertainty"])
        self.assertNotIn("corroborated", overview["uncertainty"])
        self.assertIn("not independent corroboration", overview["uncertainty"])

    def test_lone_source_is_marked_unverified(self):
        item = research_service.build_evidence_item(
            result("https://a.com"), {"title": "A", "text": "Some long enough "
                                      "sentence about a programme that ran for years."},
            "note")
        research_service.annotate_corroboration([item])
        self.assertIn("not independently verified", item["uncertainty"])

    def test_uncertainty_is_always_present(self):
        items = research_service.annotate_corroboration([{"provenance": "secondary"}])
        self.assertTrue(items[0]["uncertainty"])


# ── F28: URL de-duplication ───────────────────────────────────────────────
class DedupeTests(unittest.TestCase):
    def test_duplicates_dropped_keeping_the_best_ranked_first(self):
        results = [
            result("https://example.com/a"),
            result("https://www.example.com/a/"),          # www + trailing slash
            result("https://EXAMPLE.com/a#section"),       # host case + fragment
            result("https://example.com/a?utm_source=x"),  # tracking param
            result("https://example.com/b"),
        ]
        kept, dropped = research_service.dedupe_results(results)
        self.assertEqual(dropped, 3)
        self.assertEqual([r["url"] for r in kept],
                         ["https://example.com/a", "https://example.com/b"])

    def test_meaningful_query_params_stay_distinct(self):
        kept, dropped = research_service.dedupe_results([
            result("https://example.com/p?id=1"),
            result("https://example.com/p?id=2"),
        ])
        self.assertEqual(dropped, 0)
        self.assertEqual(len(kept), 2)


# ── F28: bounded parallel pipeline ────────────────────────────────────────
class PipelineTests(unittest.TestCase):
    def setUp(self):
        research_service.clear_stop_request()

    def _pipeline(self, urls, fetch_delay=None, note_delay=0.0,
                  deadline=None, job=None, summaries=None):
        """Drive _research_job over *urls* with faked fetch + note calls."""
        state = {
            "fetch_inflight": 0, "max_fetch": 0, "fetch_started": 0,
            "fetch_done": 0,
            "note_inflight": 0, "max_note": 0,
            "progress": [], "evidence": [], "evidence_at": [], "queries": [],
        }

        pages = {url: result(url) for url in urls}

        async def fake_fetch(page, url):
            state["fetch_inflight"] += 1
            state["fetch_started"] += 1
            state["max_fetch"] = max(state["max_fetch"], state["fetch_inflight"])
            delay = fetch_delay(url) if fetch_delay else 0.0
            if delay:
                await asyncio.sleep(delay)
            state["fetch_inflight"] -= 1
            return {"title": "T " + url, "text": "Body text for " + url}

        def fake_summarize(query, title, url, text):
            state["queries"].append(query)
            state["note_inflight"] += 1
            state["max_note"] = max(state["max_note"], state["note_inflight"])
            if note_delay:
                time.sleep(note_delay)
            state["note_inflight"] -= 1
            if summaries is not None:
                return summaries(url)
            # F04: a note has to be about the question to survive the
            # relevance gate, so the default fake note is on-topic.
            return "Best tanks: a useful note about %s." % url

        outcomes = []

        def on_progress(message):
            state["progress"].append(message)

        def on_evidence(item):
            state["evidence"].append(item)
            # How many fetches had finished when this evidence went out.
            state["evidence_at"].append(state["fetch_done"])

        async def _job(task):
            return await research_service._research_job(
                task, "best tanks", len(urls), job,
                deadline if deadline is not None else time.monotonic() + 60,
                on_progress, on_evidence)

        with patch.object(research_service, "extract_brave_async",
                          AsyncMock(return_value=[pages[u] for u in urls])), \
             patch.object(research_service, "fetch_page_async", fake_fetch), \
             patch.object(research_service, "summarize_with_gemini", fake_summarize):
            outcomes.append(asyncio.run(_job(fake_task())))

        state["outcome"] = outcomes[0]
        return state

    def test_result_order_is_preserved_under_concurrency(self):
        urls = ["https://example.com/%d" % i for i in range(6)]
        # The FIRST result is the slowest — a naive implementation would
        # publish it last.
        state = self._pipeline(urls, fetch_delay=lambda u: 0.05 if u.endswith("/0") else 0.01)
        collected = state["outcome"]["collected"]
        self.assertEqual([c["url"] for c in collected], urls)

    def test_at_most_three_fetches_in_flight(self):
        urls = ["https://example.com/%d" % i for i in range(9)]
        state = self._pipeline(urls, fetch_delay=lambda u: 0.02, note_delay=0.01)
        self.assertLessEqual(state["max_fetch"], research_service.MAX_CONCURRENT_FETCHES)
        self.assertEqual(state["max_fetch"], research_service.MAX_CONCURRENT_FETCHES)
        self.assertEqual(len(state["outcome"]["collected"]), 9)

    def test_at_most_two_note_calls_in_flight(self):
        urls = ["https://example.com/%d" % i for i in range(8)]
        state = self._pipeline(urls, fetch_delay=lambda u: 0.01, note_delay=0.03)
        self.assertLessEqual(state["max_note"], research_service.MAX_CONCURRENT_NOTES)
        self.assertEqual(len(state["outcome"]["collected"]), 8)

    def test_evidence_is_published_while_work_is_still_outstanding(self):
        urls = ["https://example.com/%d" % i for i in range(6)]
        # Slow fetches, instant notes: evidence must go out while sites are
        # still outstanding, not batched up and dumped at the end.
        state = self._pipeline(urls, fetch_delay=lambda u: 0.05)
        self.assertEqual(len(state["evidence"]), 6)
        self.assertLess(min(state["evidence_at"]), len(urls))
        for item in state["evidence"]:
            self.assertIn(item["provenance"], (
                research_service.PROVENANCE_OBSERVED,
                research_service.PROVENANCE_QUOTED,
            ))

    def test_progress_is_emitted_before_the_first_fetch(self):
        urls = ["https://example.com/%d" % i for i in range(3)]
        state = self._pipeline(urls)
        self.assertTrue(state["progress"])
        self.assertIn("Searching", state["progress"][0])
        self.assertTrue(any("Fetching" in m for m in state["progress"]))

    def test_deadline_stops_new_work_but_is_not_a_user_cancel(self):
        urls = ["https://example.com/%d" % i for i in range(4)]
        state = self._pipeline(urls, deadline=time.monotonic() - 1)
        self.assertEqual(state["fetch_started"], 0)
        self.assertEqual(state["outcome"]["collected"], [])
        # A deadline keeps what was gathered — it is not "stopped by user".
        self.assertFalse(state["outcome"]["stopped"])

    def test_explicit_cancel_is_reported_as_stopped(self):
        urls = ["https://example.com/%d" % i for i in range(4)]
        job = MagicMock()
        job.cancelled = True
        job.should_stop.return_value = True
        state = self._pipeline(urls, job=job)
        self.assertEqual(state["fetch_started"], 0)
        self.assertTrue(state["outcome"]["stopped"])

    def test_stop_flag_is_reported_as_stopped(self):
        urls = ["https://example.com/0"]
        research_service.request_stop()
        try:
            state = self._pipeline(urls)
        finally:
            research_service.clear_stop_request()
        self.assertTrue(state["outcome"]["stopped"])

    def test_question_is_threaded_into_every_note_request(self):
        urls = ["https://example.com/%d" % i for i in range(4)]
        state = self._pipeline(urls)
        self.assertEqual(len(state["queries"]), 4)
        self.assertTrue(all(q == "best tanks" for q in state["queries"]))

    def test_notes_with_no_relevant_evidence_are_skipped(self):
        urls = ["https://example.com/%d" % i for i in range(3)]
        # F04: both notes are worded as if they belong to the question; only
        # the one that reports no evidence at all is dropped by this guard.
        state = self._pipeline(
            urls,
            summaries=lambda url: ("No relevant evidence." if url.endswith("/1")
                                   else "Best tanks note for %s." % url),
        )
        collected = state["outcome"]["collected"]
        self.assertEqual([c["url"] for c in collected],
                         ["https://example.com/0", "https://example.com/2"])
        self.assertEqual(len(state["evidence"]), 2)


# ── F27: one long-lived browser owner ─────────────────────────────────────
class WarmBrowserTests(unittest.TestCase):
    def test_pages_are_scoped_to_their_task(self):
        worker = research_browser.ResearchBrowserWorker(idle_ttl=0.0)
        mine_a, mine_b, other = FakePage(), FakePage(), FakePage()
        worker._track_page("task-a", mine_a)
        worker._track_page("task-a", mine_b)
        worker._track_page("task-b", other)

        asyncio.run(worker.close_task_pages("task-a"))

        self.assertTrue(mine_a.closed)
        self.assertTrue(mine_b.closed)
        self.assertFalse(other.closed, "another task's page must survive")
        self.assertNotIn("task-a", worker._pages)
        self.assertIn("task-b", worker._pages)

    def test_context_is_reused_across_jobs_and_pages_are_closed(self):
        context = FakeContext()
        fake_pw = FakeAsyncPlaywright(context)
        worker = research_browser.ResearchBrowserWorker(idle_ttl=0.0)
        with patch("playwright.async_api.async_playwright", lambda: fake_pw):
            try:
                first = worker.submit(lambda page: page is not None)
                second = worker.submit(lambda page: True)
            finally:
                worker.shutdown()

        self.assertTrue(first)
        self.assertTrue(second)
        # ONE Chrome for both jobs — that is the whole point of the owner.
        self.assertEqual(fake_pw.playwright.chromium.launches, 1)
        self.assertEqual(len(context.pages), 2)
        self.assertTrue(all(p.closed for p in context.pages))
        self.assertTrue(context.closed)
        self.assertFalse(worker.alive)

    def test_async_job_pages_are_closed_when_the_job_ends(self):
        context = FakeContext()
        fake_pw = FakeAsyncPlaywright(context)
        worker = research_browser.ResearchBrowserWorker(idle_ttl=0.0)

        async def job(task):
            first = await task.new_page()
            second = await task.new_page()
            return (first is not None and second is not None)

        with patch("playwright.async_api.async_playwright", lambda: fake_pw):
            try:
                self.assertTrue(worker.run(job, task_id="t1"))
            finally:
                worker.shutdown()

        self.assertEqual(len(context.pages), 2)
        self.assertTrue(all(p.closed for p in context.pages))
        self.assertEqual(worker._pages.get("t1"), None)

    def test_overview_and_deep_phases_resolve_to_the_same_worker(self):
        # F27: the overview phase and the deep-search phase must reuse ONE
        # context — the profile can only host one persistent context anyway,
        # and two of them used to mean two cold starts per deepsearch.
        try:
            from backend.services import research_browser

            overview_worker = research_browser.get_worker()
            deep_worker = research_browser.get_worker(
                profile_dir=str(research_service.DEFAULT_PROFILE_DIR),
                channel=os.getenv("JARVIS_RESEARCH_CHANNEL", "chrome"),
            )
            self.assertIs(overview_worker, deep_worker)
        finally:
            research_browser.shutdown()

    def test_maybe_await_passes_plain_values_through(self):
        async def _check():
            self.assertEqual(await research_browser.maybe_await(5), 5)
        asyncio.run(_check())


# ── F27: Ask-tab completion ───────────────────────────────────────────────
class AskCompletionTests(unittest.TestCase):
    def setUp(self):
        research_service.clear_stop_request()

    def test_completion_detected_when_answer_text_stabilises(self):
        page = FakeAskPage(["", "partial answer", "partial answer",
                            "final answer", "final answer"])
        self.assertTrue(
            _await(quick_search._wait_for_answer_update(page, timeout=5, poll=0.0,
                                                 min_stable=2)))
        self.assertEqual(len(page._texts), 2, "must not poll out the script")

    def test_completion_is_bounded_by_the_deadline(self):
        # The text never stabilises — an unbounded wait would hang forever.
        page = FakeAskPage(["a", "b", "c", "d", "e", "f", "g", "h"])
        self.assertFalse(
            _await(quick_search._wait_for_answer_update(page, timeout=0.05, poll=0.0,
                                                 min_stable=2)))

    def test_stop_short_circuits_the_wait(self):
        research_service.request_stop()
        try:
            page = FakeAskPage(["steady", "steady", "steady"])
            self.assertFalse(
                _await(quick_search._wait_for_answer_update(page, timeout=5, poll=0.0)))
        finally:
            research_service.clear_stop_request()

    def test_answer_read_from_the_answer_container_not_the_disclaimer(self):
        page = FakeAskPage(["the real answer"])
        self.assertEqual(_await(quick_search._answer_container_text(page)), "the real answer")
        self.assertEqual(_await(quick_search._answer_container_text(FakeAskPage([]))), "")


# ── F48: screen-side provenance ───────────────────────────────────────────
def _vision_result(evidence):
    import json

    content = json.dumps({
        "tip": "That is a code editor.",
        "evidence": evidence,
        "topic": "editor",
        "show_images": False,
    })
    return {"choices": [{"message": {"content": content}}], "grounding_links": []}


class ScreenProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.capture = {"image_data_url": "data:image/png;base64,abc", "region": None}

    def _analyze(self, question, evidence, external=None):
        with patch.object(screen_analyzer, "capture_primary_screen",
                          return_value=self.capture), \
             patch.object(screen_analyzer, "is_region_question", return_value=False), \
             patch.object(screen_analyzer, "_ask_screen_vision_cascade",
                          return_value=_vision_result(evidence)), \
             patch.object(screen_analyzer, "_external_check",
                          return_value=external):
            return screen_analyzer.analyze_screen(question)

    def test_prompt_never_claims_the_model_can_search(self):
        self.assertNotIn("Google Search", screen_analyzer._ANALYSIS_PROMPT)
        self.assertNotIn("verify any claims", screen_analyzer._ANALYSIS_PROMPT)
        self.assertIn("provenance", screen_analyzer._ANALYSIS_PROMPT)

    def test_verification_questions_are_detected(self):
        for question in ("is this true?", "verify this claim",
                         "fact check this headline",
                         "what is the current price of this",
                         "latest version of this library"):
            self.assertTrue(screen_analyzer.needs_external_check(question), question)
        for question in ("what is on my screen", "what does this button do",
                         "summarise this paragraph"):
            self.assertFalse(screen_analyzer.needs_external_check(question), question)

    def test_screen_evidence_is_tagged_observed_or_inferred(self):
        result = self._analyze("what is on my screen", [
            {"source": "VS Code", "title": "Editor", "snippet": "A code editor.",
             "provenance": "observed"},
            {"source": "model", "title": "Guess", "snippet": "Probably Python.",
             "provenance": "inferred"},
        ])
        provenance = [e["provenance"] for e in result["evidence"]]
        self.assertEqual(provenance, ["observed", "inferred"])
        self.assertIn("seen on screen", result["evidence"][0]["uncertainty"])
        self.assertIn("inference", result["evidence"][1]["uncertainty"])
        self.assertTrue(result["observed_at"])

    def test_model_cannot_invent_a_stronger_provenance(self):
        # F48 (round 2): an invented/stronger label degrades to the
        # CONSERVATIVE default ("inferred") — it may never become "observed",
        # let alone the invented "externally_checked".
        result = self._analyze("what is on my screen", [
            {"source": "x", "title": "y", "snippet": "z",
             "provenance": "externally_checked"},
        ])
        self.assertEqual(result["evidence"][0]["provenance"], "inferred")

    def test_unlabelled_evidence_does_not_default_to_observed(self):
        # F48 (round 2): a MISSING label is not evidence that the detail was
        # seen on screen — it is the model's own inference until labelled.
        result = self._analyze("what is on my screen",
                               [{"source": "x", "title": "y", "snippet": "z"}])
        self.assertEqual(result["evidence"][0]["provenance"], "inferred")
        self.assertIn("inference", result["evidence"][0]["uncertainty"])

    def test_a_real_lookup_runs_when_verification_is_asked_for(self):
        # F48 (round 2): a deictic "is this true?" carries no checkable claim,
        # so the lookup is bound to the DISPLAYED claim (screen_analyzer now
        # passes it in) instead of searching the words "is this true".
        seen = {}

        def _overview(query):
            seen["query"] = query
            return "The claim checks out."

        with patch("backend.services.quick_search.fetch_ai_overview_text",
                   side_effect=_overview), \
             patch("backend.services.quick_search.google_search_url",
                   return_value="https://search.example/q=1"):
            item = screen_analyzer._external_check(
                "is this true?", claim="The tower is 330 m tall.")

        self.assertEqual(item["provenance"], "externally_checked")
        self.assertEqual(item["source_url"], "https://search.example/q=1")
        self.assertTrue(item["retrieved_at"])
        self.assertEqual(item["claim"], "The tower is 330 m tall.")
        self.assertIn("330 m tall", seen["query"])
        self.assertNotEqual(seen["query"].strip().lower(), "is this true?")

    def test_deictic_check_with_no_displayed_claim_runs_no_lookup(self):
        # F48: nothing on screen to check means no fabricated "verification".
        with patch("backend.services.quick_search.fetch_ai_overview_text") as fetch:
            self.assertIsNone(screen_analyzer._external_check("is this true?"))
        fetch.assert_not_called()

    def test_failed_lookup_yields_no_item_rather_than_a_fake_one(self):
        with patch("backend.services.quick_search.fetch_ai_overview_text",
                   side_effect=RuntimeError("browser busy")):
            self.assertIsNone(
                screen_analyzer._external_check("verify this", claim="x claim"))

    def test_external_check_is_appended_for_a_verification_question(self):
        # The item under test is the APPEND behaviour; give it an explicit
        # "observed" label so the assertion is about appending, not about the
        # (now conservative) default for unlabelled evidence.
        result = self._analyze(
            "verify this",
            [{"source": "x", "title": "y", "snippet": "z",
              "provenance": "observed"}],
            external={"source": "Web lookup", "title": "External check",
                      "snippet": "checked", "provenance": "externally_checked",
                      "source_url": "https://search.example", "uncertainty": "u",
                      "retrieved_at": "now"})
        self.assertEqual([e["provenance"] for e in result["evidence"]],
                         ["observed", "externally_checked"])


# ── F48: shared vocabulary ────────────────────────────────────────────────
class ProvenanceVocabularyTests(unittest.TestCase):
    def test_research_and_screen_paths_share_one_vocabulary(self):
        from backend.services import provenance

        self.assertEqual(research_service.PROVENANCE_OBSERVED,
                         provenance.PROVENANCE_OBSERVED)
        self.assertEqual(research_service.PROVENANCE_SECONDARY,
                         provenance.PROVENANCE_SECONDARY)

    def test_unknown_labels_are_rejected(self):
        from backend.services import provenance

        # F48 (round 2): an unknown/invented label degrades to the
        # uncertainty-preserving default ("inferred"), never to "observed".
        self.assertEqual(provenance.normalise_provenance("definitely true"), "inferred")
        self.assertEqual(provenance.normalise_provenance("OBSERVED"), "observed")
        self.assertEqual(provenance.normalise_provenance("quoted"), "quoted")
        self.assertEqual(
            provenance.normalise_provenance(None, default="inferred"), "inferred")


# ── F28/F27: brain supplies the hooks ─────────────────────────────────────
class ResearchWiringTests(unittest.TestCase):
    def test_run_research_accepts_progress_and_evidence_hooks(self):
        import inspect

        params = inspect.signature(research_service.run_research).parameters
        self.assertIn("on_progress", params)
        self.assertIn("on_evidence", params)
        self.assertIn("pinned_overview", params)


# ── F28: /research-progress roundtrip that the overlay consumes ───────────
class ResearchProgressEndpointTests(unittest.TestCase):
    """The brain POSTs live progress + incremental evidence; the research
    overlay polls GET /research-progress and renders it while a deep run is
    working (the overlay only swaps in the final report later). This pins the
    wire contract those two sides depend on."""

    def setUp(self):
        from backend.api import routes

        routes._research_progress_data = {
            "id": 0, "query": "", "message": "", "evidence": []}

    def test_post_then_get_roundtrip_preserves_evidence(self):
        from backend.api import routes

        item = {
            "result_title": "Example source",
            "url": "https://example.com/page",
            "provenance": research_service.PROVENANCE_OBSERVED,
            "summary": "A useful note.",
            "uncertainty": "single source — not independently verified",
            "retrieved_at": "2026-09-10T12:00:00",
        }
        posted = routes.post_research_progress(routes.ResearchProgress(
            id=9001, query="test query", message="Collected 1 source(s) so far.",
            evidence=[item]))
        self.assertEqual(posted["ok"], True)
        data = routes.get_research_progress()
        self.assertEqual(data["id"], 9001)
        self.assertEqual(data["query"], "test query")
        self.assertEqual(data["message"], "Collected 1 source(s) so far.")
        # F48 fields the overlay renders badges/links from survive the wire.
        self.assertEqual(len(data["evidence"]), 1)
        self.assertEqual(data["evidence"][0]["provenance"],
                         research_service.PROVENANCE_OBSERVED)
        self.assertEqual(data["evidence"][0]["url"], "https://example.com/page")

    def test_progress_rides_along_final_result_contract(self):
        """A finished deep run pushes the report AND a final progress message;
        the report's completion id is always newer than its own progress ids,
        so the overlay swap-over logic can never re-pop a finished report."""
        from backend.api import routes

        routes.post_research_progress(routes.ResearchProgress(
            id=1000, query="q", message="Finished — 2 source(s).", evidence=[]))
        routes.post_research_result(routes.ResearchResult(
            id=2000, query="q", markdown="# done", report_path="r.md",
            visited_count=2, failed_count=0))
        progress = routes.get_research_progress()
        result = routes.get_research_result()
        self.assertLess(progress["id"], result["id"])
        self.assertEqual(result["visited_count"], 2)


if __name__ == "__main__":
    unittest.main()
