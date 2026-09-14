"""Round-7 regression: tiered search — Google AI Overview first,
deepsearch on demand.  All browser / LLM calls are mocked."""

import asyncio
import inspect
import threading
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.core import brain
from backend.services import quick_search
from backend.services import research_service


def _await(value):
    """F27: the quick-search browser helpers are async now.

    Every page access is awaited, so a test that drives them directly has to
    run the coroutine instead of comparing it to a value.
    """
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


def run_with_fake_browser(coro_fn, task_id=None, timeout=None,
                          profile_dir=None, channel=None):
    """Run a research job coroutine against a MagicMock browser context.

    Stands in for ``research_service._browser_run`` so the pipeline can be
    exercised without launching Chrome.
    """
    ctx = MagicMock()
    ctx.new_page.side_effect = lambda: MagicMock()

    async def _new_page():
        return ctx.new_page()

    async def _close_pages():
        return None

    task = MagicMock()
    task.context = ctx
    task.task_id = task_id
    task.new_page = _new_page
    task.close_pages = _close_pages
    return asyncio.run(coro_fn(task))


def quick_submit(page=None):
    """Patch target for ``quick_search._browser_run``.

    F27: quick search submits a real ``async def`` job body to the warm worker,
    so the fake worker hands the body a task whose ``new_page()`` returns the
    mock page — the same shape ``research_browser.ResearchTask`` provides.
    """
    target = page if page is not None else MagicMock()

    def _run(coro_fn, task_id=None, timeout=None, profile_dir=None,
             channel=None):
        task = MagicMock()
        task.task_id = task_id
        task.context = MagicMock()

        async def _new_page():
            return target

        async def _close_pages():
            return None

        task.new_page = _new_page
        task.close_pages = _close_pages
        return asyncio.run(coro_fn(task))

    return _run


class DeepsearchKeywordDetectionTests(unittest.TestCase):
    """Feature 2: keyword detection lives in brain parsing (intent.py
    untouched by standing rule)."""

    def test_deepsearch_request_true(self):
        self.assertTrue(brain.is_deepsearch_request("deepsearch about upcoming models"))
        self.assertTrue(brain.is_deepsearch_request("do a deepsearch on ai"))
        self.assertTrue(brain.is_deepsearch_request("deepsearch for news"))
        self.assertTrue(brain.is_deepsearch_request("run a deepsearch about llms"))

    def test_deepsearch_request_false(self):
        self.assertFalse(brain.is_deepsearch_request("search the web for X"))
        self.assertFalse(brain.is_deepsearch_request("research about Y"))
        self.assertFalse(brain.is_deepsearch_request("what is the weather"))
        self.assertFalse(brain.is_deepsearch_request(""))

    def test_force_research_catches_deepsearch(self):
        self.assertTrue(brain.force_research("deepsearch about something"))
        self.assertTrue(brain.force_research("do a deepsearch on X"))

    def test_heuristic_strips_deepsearch_prefix(self):
        result = brain._heuristic_research_query("deepsearch about upcoming model releases")
        self.assertNotIn("deepsearch", result.lower())
        self.assertIn("model releases", result.lower())


class TieredModeRoutingTests(unittest.TestCase):
    """handle_research_intent routes to quick_search vs run_research based
    on the deep flag."""

    def _wait_for_thread(self, evt, timeout=3.0):
        self.assertTrue(evt.wait(timeout),
                        "worker thread did not call within %.1fs" % timeout)

    def test_default_mode_calls_quick_search(self):
        """Default (deep=False) runs run_quick_search, not run_research."""
        called = threading.Event()

        def _quick(*a, **kw):
            called.set()
            return {"query": a[0], "spoken_summary": "", "overview_found": False,
                    "overview_text": None, "fallback": None, "stopped": False}

        with patch.object(brain, "run_quick_search", side_effect=_quick), \
             patch.object(brain, "run_research") as mock_deep, \
             patch.object(brain, "derive_research_query",
                          side_effect=lambda q: q), \
             patch.object(brain, "_notify_async_reply") as notify:
            brain.handle_research_intent("what is new in ai", deep=False)
            self._wait_for_thread(called)
        mock_deep.assert_not_called()

    def test_deep_mode_calls_research(self):
        """Deep mode (deep=True) fetches overview and calls run_research
        with pinned_overview."""
        overview_called = threading.Event()
        research_called = threading.Event()

        def _overview(q):
            overview_called.set()
            return "AI overview text"

        def _research(q, **kw):
            research_called.set()
            # Full run_research return contract — handle_research_intent's
            # worker thread reads detailed_markdown/related_videos/report_path/
            # visited_count/failed_count once the result lands, so a partial
            # dict would crash the thread with a KeyError after the test ends.
            return {
                "query": q,
                "spoken_summary": "Research done.",
                "detailed_markdown": f"# Research: {q}\n\nDone.",
                "related_videos": [],
                "report_path": None,
                "evidence": [],
                "visited_count": 1,
                "failed_count": 0,
                "stopped": False,
            }

        # The worker thread's terminal call is _notify_async_reply — waiting on
        # research_called alone would race: fetch/run_research return, then the
        # thread still runs push_research_progress / push_research_result /
        # _notify_async_reply AFTER the with-block patches are torn down.
        notified = threading.Event()

        def _notify(**kw):
            notified.set()

        with patch.object(brain, "fetch_ai_overview_text",
                          side_effect=_overview) as mock_overview, \
             patch.object(brain, "run_research", side_effect=_research) as mock_deep, \
             patch.object(brain, "push_research_result") as mock_push, \
             patch.object(brain, "derive_research_query",
                          side_effect=lambda q: q), \
             patch.object(brain, "_notify_async_reply", side_effect=_notify) as notify:
            brain.handle_research_intent("deepsearch about ai", deep=True)
            self.assertTrue(research_called.wait(2.0), "run_research was not called")
            self.assertTrue(notified.wait(5.0), "research worker thread did not finish")
            mock_overview.assert_called_once()
            # pinned_overview must be passed through
            kwargs = mock_deep.call_args[1]
            self.assertEqual(kwargs.get("pinned_overview"), "AI overview text")
            # The finished report is pushed to the overlay and the spoken
            # summary is announced — the thread completed without crashing.
            mock_push.assert_called_once()
            notify_calls = [c.kwargs for c in notify.call_args_list if c.kwargs]
            self.assertTrue(notify_calls, "worker thread never announced the result")
            self.assertEqual(notify_calls[-1].get("spoken"), "Research done.")

    def test_deep_keyword_routed_via_force_research(self):
        """process_message routes a deepsearch phrase through
        handle_research_intent with deep=True."""
        with patch.object(brain, "handle_research_intent") as mock_handle, \
             patch.object(brain, "classify_intent",
                          return_value={"intent": "chat"}), \
             patch.object(brain, "_consume_confirmation",
                          return_value=None), \
             patch.object(brain, "consume_task_confirmation",
                          return_value=None), \
             patch.object(brain, "_consume_opencode_confirmation",
                          return_value=None), \
             patch.object(brain, "_consume_browser_followup",
                          return_value=None), \
             patch.object(brain, "is_explicit_task_request",
                          return_value=False), \
             patch.object(brain, "is_code_tool_request",
                          return_value=False), \
             patch.object(brain, "maybe_handle_screen_control_message",
                          return_value=None):
            brain.process_message("deepsearch about upcoming phones")
        mock_handle.assert_called_once()
        kwargs = mock_handle.call_args[1]
        self.assertTrue(kwargs.get("deep"),
                        "deepsearch keyword must set deep=True in handle_research_intent")

    def test_plain_search_routed_without_deep(self):
        """Plain research phrase routes with deep=False."""
        with patch.object(brain, "handle_research_intent") as mock_handle, \
             patch.object(brain, "classify_intent",
                          return_value={"intent": "chat"}), \
             patch.object(brain, "_consume_confirmation",
                          return_value=None), \
             patch.object(brain, "consume_task_confirmation",
                          return_value=None), \
             patch.object(brain, "_consume_opencode_confirmation",
                          return_value=None), \
             patch.object(brain, "_consume_browser_followup",
                          return_value=None), \
             patch.object(brain, "is_explicit_task_request",
                          return_value=False), \
             patch.object(brain, "is_code_tool_request",
                          return_value=False), \
             patch.object(brain, "maybe_handle_screen_control_message",
                          return_value=None):
            brain.process_message("search the web for cat food")
        mock_handle.assert_called_once()
        kwargs = mock_handle.call_args[1]
        self.assertFalse(kwargs.get("deep"),
                         "plain research must have deep=False")


class AiOverviewExtractionTests(unittest.TestCase):
    """The small mockable DOM function for Google AI Overview."""

    def test_extract_ai_overview_found(self):
        page = MagicMock()
        node = MagicMock()
        node.inner_text.return_value = (
            "Here is a comprehensive answer to your query with many details. "
            "This is a test AI Overview with enough text to be considered a "
            "real overview and not an empty shell."
        )
        page.query_selector.return_value = node
        result = _await(quick_search.extract_ai_overview(page))
        self.assertIsNotNone(result)
        self.assertIn("comprehensive answer", result)

    def test_extract_ai_overview_not_found(self):
        page = MagicMock()
        page.query_selector.return_value = None
        result = _await(quick_search.extract_ai_overview(page))
        self.assertIsNone(result)

    def test_extract_top_snippet_found(self):
        page = MagicMock()
        h3 = MagicMock()
        h3.inner_text.return_value = "Top Result Title"
        h3.get_attribute.return_value = "https://example.com"
        page.query_selector.return_value = h3
        result = _await(quick_search.extract_top_snippet(page))
        self.assertIsNotNone(result)
        self.assertEqual(result["title"], "Top Result Title")

    def test_extract_top_snippet_not_found(self):
        page = MagicMock()
        page.query_selector.return_value = None
        result = _await(quick_search.extract_top_snippet(page))
        self.assertIsNone(result)


class DeepsearchPinnedOverviewTests(unittest.TestCase):
    """research_service.run_research with pinned_overview."""

    def setUp(self):
        research_service._STOP_REQUESTED.clear()

    def test_pinned_overview_inserted_into_report_items(self):
        with patch.object(research_service, "consolidate_summaries") as mock_consolidate, \
             patch.object(research_service, "build_spoken_summary",
                          return_value="spoken"), \
             patch.object(research_service, "build_details_markdown") as mock_build, \
             patch.object(research_service, "extract_brave_async",
                          new_callable=AsyncMock, return_value=[]), \
             patch.object(research_service, "_browser_run",
                          side_effect=run_with_fake_browser):
            mock_consolidate.return_value = "consolidated"
            mock_build.return_value = "# details"
            result = research_service.run_research(
                "test query",
                pinned_overview="Google says this is the answer.",
            )
        self.assertFalse(result.get("stopped"))
        # build_details_markdown receives the report_items as its 3rd
        # positional arg — it must contain the pinned overview entry.
        items_arg = mock_build.call_args[0][3] if mock_build.call_args else []
        overview = [it for it in items_arg
                    if "overview" in (it.get("result_title") or "").lower()]
        self.assertTrue(overview, "report_items must include the AI Overview entry")
        # F48: the overview is a SECONDARY source, not a first-party one and
        # not corroboration — it must never be labelled "Google AI Overview".
        self.assertEqual(overview[0].get("provenance"),
                         research_service.PROVENANCE_SECONDARY)
        self.assertNotIn("Google AI Overview", overview[0].get("result_title") or "")

    def test_stop_event_mechanism(self):
        self.assertFalse(research_service.stop_requested())
        research_service.request_stop()
        self.assertTrue(research_service.stop_requested())

    def test_stop_event_cleared_on_new_run(self):
        research_service.request_stop()
        self.assertTrue(research_service.stop_requested())
        with patch.object(research_service, "extract_brave_async",
                          new_callable=AsyncMock, return_value=[]), \
             patch.object(research_service, "_browser_run",
                          side_effect=run_with_fake_browser):
            result = research_service.run_research("test clear")
        self.assertFalse(result.get("stopped"),
                         "stop event must be cleared on each run_research call")


class QuickSearchStopTests(unittest.TestCase):
    """run_quick_search stop checks (research_service stop event)."""

    def setUp(self):
        research_service._STOP_REQUESTED.clear()

    def test_quick_search_stop_before_run_returns_stopped(self):
        """Stop event set before run -> stopped result, no browser call."""
        research_service.request_stop()
        result = quick_search.run_quick_search("test stop before")
        self.assertTrue(result.get("stopped"))
        self.assertEqual(result["spoken_summary"],
                         "Stopped the research as requested, sir.")

    def test_quick_search_stop_during_goto_returns_stopped(self):
        """Stop set after the page goto (mocked) -> stopped result."""
        with patch.object(quick_search, "_browser_run",
                          side_effect=quick_submit()), \
             patch.object(quick_search, "_open_google") as mock_open, \
             patch.object(quick_search, "_wait_for_ai_completion") as mock_wait:
            mock_open.side_effect = lambda page, query: research_service.request_stop()
            result = quick_search.run_quick_search("test stop mid")
        self.assertTrue(result.get("stopped"))

    def test_quick_search_normal_run_not_stopped(self):
        """Normal run (no stop event) returns non-stopped result."""
        with patch.object(quick_search, "_browser_run",
                          side_effect=quick_submit()), \
             patch.object(quick_search, "_open_google"), \
             patch.object(quick_search, "_wait_for_ai_completion"), \
             patch.object(quick_search, "extract_ai_overview",
                          return_value=None), \
             patch.object(quick_search, "extract_top_snippet",
                          return_value=None):
            result = quick_search.run_quick_search("test normal")
        self.assertFalse(result.get("stopped"))

    def test_google_search_url_uses_brave(self):
        """Quick-search must use Brave Search URL (not Google, avoids
        CAPTCHA on headed automation)."""
        self.assertEqual(
            quick_search.google_search_url("test q"),
            "https://search.brave.com/search?q=test+q",
        )


class StaleStopEventRegressionTests(unittest.TestCase):
    """The stop event must not persist across idle calls or brick
    subsequent quick searches."""

    def setUp(self):
        research_service._STOP_REQUESTED.clear()
        brain._research_running = False

    def tearDown(self):
        research_service._STOP_REQUESTED.clear()
        brain._research_running = False

    def test_clear_stop_request_unit(self):
        research_service.request_stop()
        self.assertTrue(research_service.stop_requested())
        research_service.clear_stop_request()
        self.assertFalse(research_service.stop_requested())

    def test_idle_stop_does_not_brick_quick_search(self):
        """Empirical repro: idle stop request (via handle_stop_research_request
        with _research_running=False) must NOT set the stop event, so
        subsequent quick searches return real results."""
        with patch.object(brain, "_research_running", False), \
             patch.object(brain, "request_browser_task_stop"), \
             patch.object(brain, "set_narration_enabled"), \
             patch.object(brain, "opencode_task_in_progress",
                          return_value=False):
            brain.handle_stop_research_request(from_voice=False)
        self.assertFalse(research_service.stop_requested(),
                         "idle stop must not set the event")
        with patch.object(quick_search, "_browser_run",
                          side_effect=quick_submit()), \
             patch.object(quick_search, "_open_google"), \
             patch.object(quick_search, "_wait_for_ai_completion"):
            r1 = quick_search.run_quick_search("cats")
            r2 = quick_search.run_quick_search("weather")
        self.assertFalse(r1.get("stopped"),
                         "first quick search after idle stop must not be stopped")
        self.assertFalse(r2.get("stopped"),
                         "second quick search after idle stop must not be stopped")

    def test_mid_run_stop_consumed_cleared_by_finally(self):
        """Stop event set then cleared by handle_research_intent's _run
        finally -> next quick search works."""
        from backend.core import brain as brain_mod
        research_service.request_stop()
        # Simulate the _run finally block clearing the event
        brain_mod._research_running = False
        try:
            research_service.clear_stop_request()
        except Exception:
            pass
        with patch.object(quick_search, "_browser_run",
                          side_effect=quick_submit()), \
             patch.object(quick_search, "_open_google"), \
             patch.object(quick_search, "_wait_for_ai_completion"):
            r = quick_search.run_quick_search("after stop")
        self.assertFalse(r.get("stopped"),
                         "quick search after clearing must not be stopped")


class ResearchRunningFlagTests(unittest.TestCase):
    """The module-level _research_running flag is actually set by
    handle_research_intent (not kept local to the outer scope)."""

    def setUp(self):
        brain._research_running = False
        research_service._STOP_REQUESTED.clear()

    def tearDown(self):
        brain._research_running = False
        research_service._STOP_REQUESTED.clear()

    def test_handle_research_intent_sets_running_flag(self):
        """handle_research_intent must set _research_running=True on the
        module-level flag before the thread starts (patched to no-op)."""
        noop = MagicMock()
        with patch.object(brain, "threading") as mock_thr, \
             patch.object(brain, "derive_research_query",
                          side_effect=lambda q: q), \
             patch.object(brain, "_notify_async_reply"):
            mock_thr.Thread.return_value = noop
            brain.handle_research_intent("test query", deep=False)
        self.assertTrue(brain._research_running,
                        "flag must be True immediately after handle_research_intent returns")
        # The module-level flag was set by the outer scope, not just the closure.

    def test_spawn_failure_clears_flag_and_event(self):
        """If Thread.start() raises, the flag must be reset and the stop
        event cleared so the next search is not bricked."""
        with patch.object(brain, "derive_research_query",
                          side_effect=lambda q: q), \
             patch.object(brain, "threading") as mock_thr:
            mock_thr.Thread.side_effect = RuntimeError("spawn failed")
            with self.assertRaises(RuntimeError):
                brain.handle_research_intent("test spawn fail", deep=False)
        self.assertFalse(brain._research_running,
                         "flag must be False after spawn failure")
        self.assertFalse(research_service.stop_requested(),
                         "stop event must be cleared after spawn failure")


class CompletionWaitTests(unittest.TestCase):
    """_wait_for_ai_completion polls for a stable answer length."""

    def setUp(self):
        from backend.services.research_service import _STOP_REQUESTED
        _STOP_REQUESTED.clear()

    def test_ai_answer_present_stable_length(self):
        """When chatllm-answer exists with stable length, returns True."""
        page = MagicMock()
        node = MagicMock()
        node.inner_text.return_value = "x" * 200
        page.query_selector.return_value = node
        with patch.object(quick_search, "time") as mock_time:
            mock_time.monotonic.side_effect = [0, 1.5, 3.0]
            result = _await(quick_search._wait_for_ai_completion(page, timeout=5))
        self.assertTrue(result)

    def test_ai_answer_timeout_returns_false(self):
        """When no selector ever matches, returns False (timeout)."""
        page = MagicMock()
        page.query_selector.return_value = None
        with patch.object(quick_search, "time") as mock_time:
            mock_time.monotonic.side_effect = [0, 10]
            result = _await(quick_search._wait_for_ai_completion(page, timeout=2))
        self.assertFalse(result)

    def test_chatllm_answer_extracted_via_extract_ai_overview(self):
        """extract_ai_overview finds div.chatllm-answer with real text."""
        page = MagicMock()
        node = MagicMock()
        node.inner_text.return_value = (
            "The quick brown fox jumps over the lazy dog. " * 5
        )
        page.query_selector.return_value = node
        result = _await(quick_search.extract_ai_overview(page))
        self.assertIsNotNone(result)
        self.assertIn("quick brown fox", result)

    def test_chatllm_absent_extracts_google_fallback(self):
        """When chatllm-answer is absent, falls through to Google selectors."""
        page = MagicMock()
        # First call (chatllm-answer) -> None, second (chatllm-content) -> None,
        # third (ai-answer) -> a node
        google_node = MagicMock()
        google_node.inner_text.return_value = "Google AI overview answer text here with enough length to pass the 80-char threshold. " + "more padding here for length."
        page.query_selector.side_effect = [None, None, google_node]
        result = _await(quick_search.extract_ai_overview(page))
        self.assertIsNotNone(result)
        self.assertIn("Google AI overview", result)


class AskTabFallbackTests(unittest.TestCase):
    """Ask tab fallback: when default SERP has no AI overview, clicking the
    Ask tab routes through the AI chat interface.  Three DOM-seam scenarios:
    Ask answer arrives, no Ask tab, stop during Ask-wait."""

    def setUp(self):
        research_service._STOP_REQUESTED.clear()
        self.page = MagicMock()
        self.submit_patch = patch.object(quick_search, "_browser_run",
                                         side_effect=quick_submit(self.page))
        self.submit_patch.start()

    def tearDown(self):
        self.submit_patch.stop()
        research_service._STOP_REQUESTED.clear()

    def test_default_overview_still_skips_ask_tab(self):
        """When the default SERP already has an AI overview, the Ask tab
        fallback is NOT invoked — the overview is returned directly."""
        real_overview = (
            "Default AI overview that appears on the SERP. "
            "It has enough details to pass the 80-character threshold. "
            "This text is well beyond the minimum length requirement."
        )
        with patch.object(quick_search, "_open_google"), \
             patch.object(quick_search, "_wait_for_ai_completion"), \
             patch.object(quick_search, "extract_ai_overview",
                          return_value=real_overview), \
             patch.object(quick_search, "_click_ask_tab") as mock_click:
            result = quick_search.run_quick_search("test default overview")
        self.assertTrue(result.get("overview_found"))
        self.assertIn("Default AI overview", result["overview_text"])
        mock_click.assert_not_called()
        self.assertIsNone(result.get("fallback"))

    def test_ask_tab_provides_overview(self):
        """When default SERP has no AI overview but the Ask tab exists and
        clicking it yields an AI answer, the overview is returned."""
        ask_answer = (
            "This is the AI answer from the Ask tab. "
            "It contains enough detail for the 80-character threshold. "
            "The Ask tab successfully provided the AI overview. "
        )
        with patch.object(quick_search, "_open_google"), \
             patch.object(quick_search, "_wait_for_ai_completion"), \
             patch.object(quick_search, "extract_ai_overview",
                          side_effect=[None, ask_answer]), \
             patch.object(quick_search, "_click_ask_tab",
                          return_value=True) as mock_click, \
             patch.object(self.page, "wait_for_load_state"):
            result = quick_search.run_quick_search("test ask tab works")
        self.assertTrue(result.get("overview_found"))
        self.assertIn("Ask tab", result["overview_text"])
        mock_click.assert_called_once()
        self.assertIsNone(result.get("fallback"))

    def test_no_ask_tab_falls_back_to_snippet(self):
        """When neither the default SERP nor the Ask tab produces an AI
        overview, the snippet fallback is used."""
        fb = {"title": "Fallback", "url": "https://x.com", "snippet": "text"}
        with patch.object(quick_search, "_open_google"), \
             patch.object(quick_search, "_wait_for_ai_completion"), \
             patch.object(quick_search, "extract_ai_overview",
                          return_value=None), \
             patch.object(quick_search, "_click_ask_tab",
                          return_value=False), \
             patch.object(quick_search, "extract_top_snippet",
                          return_value=fb):
            result = quick_search.run_quick_search("test no ask")
        self.assertFalse(result.get("overview_found"))
        self.assertEqual(result["fallback"]["title"], "Fallback")
        self.assertEqual(result["spoken_summary"],
                         "Sir, no AI answer was found. Top result — Fallback: text")

    def test_stop_during_ask_wait_returns_stopped(self):
        """If stop_requested is set during the Ask-tab wait phase, the
        result is a stopped result (not a mangled partial)."""
        def _wait_hook(*a, **kw):
            research_service.request_stop()
            return True

        with patch.object(quick_search, "_open_google"), \
             patch.object(quick_search, "_wait_for_ai_completion"), \
             patch.object(quick_search, "_wait_for_answer_update",
                          side_effect=_wait_hook), \
             patch.object(quick_search, "extract_ai_overview",
                          return_value=None), \
             patch.object(quick_search, "_click_ask_tab",
                          return_value=True), \
             patch.object(self.page, "wait_for_load_state"):
            result = quick_search.run_quick_search("test stop ask")
        self.assertTrue(result.get("stopped"))
        self.assertEqual(result["spoken_summary"],
                         "Stopped the research as requested, sir.")


class AskModeGateTests(unittest.TestCase):
    """Real-seam regression: extract_ai_overview must NEVER read <main> when
    ask_mode=False. Without the gate, the <main> fallback captured SERP
    results text as a fake overview -> overview_found=True -> the Ask branch
    became dead code (re-audit-11 preemption bug). This test FAILS if the
    ask_mode gate is removed."""

    def _serp_page(self):
        page = MagicMock()
        # No chatllm/AI selector matches on a no-overview SERP.
        page.query_selector.return_value = None
        return page

    def test_serp_never_reads_main_without_ask_mode(self):
        """ask_mode=False (default): no selector hit -> returns None and
        <main> is never queried."""
        page = self._serp_page()
        result = _await(quick_search.extract_ai_overview(page))
        self.assertIsNone(result)
        queried = [c.args[0] for c in page.query_selector.call_args_list
                   if c.args]
        self.assertNotIn(
            "main", queried,
            "<main> must never be queried when ask_mode=False — SERP results "
            "text would be captured as a fake overview and the Ask branch "
            "would become dead code",
        )

    def test_ask_mode_reads_main_and_strips_shell(self):
        """ask_mode=True fallback: <main> is read, UI shell lines stripped,
        answer kept. (No query echo here — echo/history cutting is covered
        by test_ask_mode_fallback_real_layout_is_answer_only.)"""
        page = MagicMock()
        page.query_selector.return_value = None  # no SERP/container selectors
        main = MagicMock()
        main.inner_text.return_value = (
            "New Conversation\nCtrl + Shift + O\nEncrypted & Private History\n"
            "Settings\nAsk\nAll\nImages\n"
            "The actual AI answer text from the Ask view goes right here and "
            "it is deliberately long enough to pass the eighty character "
            "extraction threshold for the overview text. "
            "Finished\nGot it\n"
        )

        def _sel(sel):
            return main if sel == "main" else None

        page.query_selector.side_effect = _sel
        result = _await(quick_search.extract_ai_overview(page, ask_mode=True))
        self.assertIsNotNone(result)
        self.assertIn("actual AI answer", result)
        self.assertNotIn("New Conversation", result)
        self.assertNotIn("Encrypted & Private History", result)
        self.assertNotIn("Got it", result)

    def test_ask_mode_prefers_answer_container(self):
        """ask_mode=True: the stable answer container wins and <main> is
        never read — no shell, no surgery, no history."""
        page = MagicMock()
        page.query_selector.return_value = None  # no SERP selectors
        answer = MagicMock()
        answer.inner_text.return_value = (
            "asdfghjkl zxcvbnm qwertyuiop is just the letters of a QWERTY "
            "keyboard typed row by row — the middle home row, bottom row, "
            "and top row respectively. People type it as placeholder text."
        )
        main = MagicMock()
        main.inner_text.return_value = "New Conversation\n" + "x" * 300

        def _sel(sel):
            if sel == "div.message.assistant.llm-output":
                return answer
            if sel == "main":
                return main
            return None

        page.query_selector.side_effect = _sel
        result = _await(quick_search.extract_ai_overview(
            page, ask_mode=True, query="asdfghjkl zxcvbnm qwertyuiop"))
        self.assertIsNotNone(result)
        self.assertIn("QWERTY keyboard", result)
        queried = [c.args[0] for c in page.query_selector.call_args_list
                   if c.args]
        self.assertNotIn("main", queried,
                         "answer container must win — <main> must not be read")

    def test_ask_mode_fallback_real_layout_is_answer_only(self):
        """Fallback surgery on the REAL observed Ask layout (re-audit 13):
        sidebar history + banner paragraph + query echo + Finished/+6 +
        answer. Result must carry only the answer."""
        query = "asdfghjkl zxcvbnm qwertyuiop"
        page = MagicMock()
        page.query_selector.return_value = None  # no SERP selectors
        main = MagicMock()
        main.inner_text.return_value = (
            "New Conversation\nCtrl + Shift + O\n"
            f"{query}\n"
            "what is the weather in tokyo today\n"
            "cloud fable 5.1 pricing\ncloud fable 5.1 pricing\n"
            "cloud fable 5.1 pricing\n"
            "Encrypted & Private History\n"
            "Your chat history is encrypted and auto-deleted after 24 hours "
            "of inactivity by default. The encryption key is stored locally "
            "on your device. Brave does not retain your IP address.\n"
            "Learn more\nGot it\nSettings\nAsk\nAll\nImages\nNews\nVideos\n"
            "Maps\nGoggles\n"
            f"{query}\nFinished\n+6\n\n"
            "That string is just random keyboard filler with no hidden "
            "meaning whatsoever, and this sentence is deliberately long "
            "enough to pass every extraction threshold comfortably. "
            "tynker.com › coding for kids › projects › keyboard filler\n"
            "Elaborate\nWhat is the history of the QWERTY layout?\n"
            "Copy\nTry again\nAI-generated answer. Please verify critical facts.\n"
        )

        def _sel(sel):
            # Container renamed by Brave -> fallback path via <main>.
            return main if sel == "main" else None

        page.query_selector.side_effect = _sel
        result = _await(quick_search.extract_ai_overview(
            page, ask_mode=True, query=query))
        self.assertIsNotNone(result)
        self.assertIn("That string is just random keyboard filler", result)
        self.assertNotIn("auto-deleted", result, "banner paragraph must go")
        self.assertNotIn("Learn more", result, "banner tail must go")
        self.assertNotIn("weather in tokyo", result, "history must go")
        self.assertNotIn("cloud fable", result, "history must go")
        self.assertNotIn(query, result, "query echo must go")
        self.assertNotIn("+6", result, "+N marker must go")
        self.assertNotIn("tynker.com", result, "citation breadcrumb must go")
        self.assertNotIn("Elaborate", result, "follow-up label must go")
        self.assertNotIn("Try again", result, "follow-up label must go")
        self.assertNotIn("AI-generated answer", result, "disclaimer must go")
        self.assertNotIn("What is the history of the QWERTY layout", result,
                         "suggested prompt must go")
        self.assertNotRegex(result, r"\bCopy\b",
                            "standalone Copy label must go")


class ResearchStateClearTests(unittest.TestCase):
    """State flags are always cleared after a quick search completes or stops.

    BUG 1 regression: after a quick search (normal or stopped), the
    module-level _research_running flag and _STOP_REQUESTED event must
    not persist — every normal query pays no wait for residual research
    state.
    """

    def setUp(self):
        self.page = MagicMock()
        self.submit_patch = patch.object(quick_search, "_browser_run",
                                         side_effect=quick_submit(self.page))
        self.submit_patch.start()
        from backend.services.research_service import _STOP_REQUESTED
        _STOP_REQUESTED.clear()
        brain._research_running = False

    def tearDown(self):
        self.submit_patch.stop()
        from backend.services.research_service import _STOP_REQUESTED
        _STOP_REQUESTED.clear()

    def test_state_clear_after_quick_search_completes(self):
        """After a normal (non-stopped) quick search, _research_running
        is False and _STOP_REQUESTED is not set."""
        from backend.core import brain
        with patch.object(quick_search, "_open_google"), \
             patch.object(quick_search, "_wait_for_ai_completion"), \
             patch.object(quick_search, "extract_ai_overview",
                          return_value="Valid overview text here that passes the 80-char threshold. This is more content to make it even longer."):
            result = quick_search.run_quick_search("test state clear")
        self.assertFalse(result.get("stopped"))
        self.assertFalse(brain._research_running,
                         "_research_running must be False after completion")
        from backend.services.research_service import stop_requested
        self.assertFalse(stop_requested(),
                         "_STOP_REQUESTED must not be set after completion")

    def test_state_clear_after_stop_mid_run(self):
        """After a stopped quick search, _research_running must be False."""
        from backend.core import brain
        from backend.services.research_service import request_stop
        request_stop()
        result = quick_search.run_quick_search("test stop state")
        self.assertTrue(result.get("stopped"))
        self.assertFalse(brain._research_running,
                         "_research_running must be False after stop")

    def test_normal_chat_pays_no_research_wait(self):
        """classify_intent with no_retry + (3,3) timeout completes fast even
        when providers are slow — no live network needed in this test."""
        from backend.services import intent as _int
        with patch.object(_int, "_classify_with_gemini",
                          return_value='{"intent":"chat"}'), \
             patch.object(_int, "_classify_with_groq",
                          return_value=""):
            result = _int.classify_intent("what is 2+2")
        self.assertEqual(result.get("intent"), "chat")


class IntentDeadlineTests(unittest.TestCase):
    """F24: classify_intent threads ONE monotonic deadline through primary
    and fallback — the advertised budget is real and never exceeded."""

    def test_zero_budget_fast_fails_to_chat_without_calls(self):
        from backend.services import intent as _int
        with patch.object(_int, "_classify_with_gemini") as gem, \
             patch.object(_int, "_classify_with_groq") as groq:
            result = _int.classify_intent("hello there", timeout_ms=0)
        gem.assert_not_called()
        groq.assert_not_called()
        self.assertEqual(result["intent"], "chat")
        self.assertEqual(result["query"], "hello there")

    def test_remaining_budget_passed_as_timeout(self):
        from backend.services import intent as _int
        with patch.object(_int, "_classify_with_gemini",
                          return_value='{"intent":"chat"}') as gem, \
             patch.object(_int, "_classify_with_groq", return_value=""):
            _int.classify_intent("hello", timeout_ms=3500)
        gem.assert_called_once()
        timeout = gem.call_args.kwargs.get("timeout")
        self.assertIsNotNone(timeout)
        connect, read = timeout
        self.assertLessEqual(connect, 3.0)
        self.assertLessEqual(read, 3.5)
        self.assertGreater(read, 0)

    def test_exhausted_budget_skips_fallback(self):
        """When the primary eats the whole budget the Groq fallback is
        skipped — fast-fail preserved with a real total deadline."""
        from backend.services import intent as _int

        # Monotonic reads: deadline anchor, primary budget, fallback budget.
        # The third read is past the deadline, so Groq never runs.
        ticks = iter([100.0, 100.0, 109.9])
        with patch.object(_int.time, "monotonic",
                          side_effect=lambda: next(ticks)), \
             patch.object(_int, "_classify_with_gemini",
                          return_value="") as gem, \
             patch.object(_int, "_classify_with_groq",
                          return_value='{"intent":"tool"}') as groq:
            result = _int.classify_intent("hello", timeout_ms=3500)
        gem.assert_called_once()
        groq.assert_not_called()
        self.assertEqual(result["intent"], "chat")


class GeminiDeadlineTests(unittest.TestCase):
    """F24: gemini_client manual retry loops honor an optional deadline."""

    def test_chat_loop_stops_when_deadline_already_passed(self):
        from backend.services import gemini_client
        with patch.object(gemini_client, "GEMINI_API_KEY", "test-key"), \
             patch.object(gemini_client._session, "post") as post, \
             patch.object(gemini_client.time, "monotonic",
                          return_value=1000.0):
            result = gemini_client.ask_gemini_chat(
                [{"role": "user", "content": "hi"}], deadline=500.0
            )
        self.assertEqual(result, {})
        post.assert_not_called()

    def test_vision_loop_stops_when_deadline_already_passed(self):
        from backend.services import gemini_client
        with patch.object(gemini_client, "GEMINI_API_KEY", "test-key"), \
             patch.object(gemini_client._session, "post") as post, \
             patch.object(gemini_client.time, "monotonic",
                          return_value=1000.0):
            result = gemini_client.ask_gemini_vision(
                "describe", "data:image/png;base64,abc", deadline=500.0
            )
        self.assertEqual(result, {})
        post.assert_not_called()


class ResearchEvidenceSchemaTests(unittest.TestCase):
    """F04: site notes are extracted against the question, and notes with no
    relevant evidence are skipped before consolidation."""

    def test_summarize_receives_query_and_evidence_schema(self):
        with patch("backend.services.gemini_client.ask_gemini_chat") as gchat:
            gchat.return_value = {
                "choices": [{"message": {"content": "claim + date evidence"}}]
            }
            out = research_service.summarize_with_gemini(
                "what is the price of X", "Page Title", "http://u", "text"
            )
        self.assertEqual(out, "claim + date evidence")
        messages = gchat.call_args.args[0]
        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]
        self.assertIn("what is the price of X", user_prompt)
        self.assertIn("QUESTION:", user_prompt)
        for schema_key in ("quotations", "Dates", "Applicability",
                           "Unanswered parts"):
            self.assertIn(schema_key, system_prompt)
        self.assertIn("NO RELEVANT EVIDENCE", system_prompt)

    def test_no_evidence_note_detection(self):
        self.assertTrue(
            research_service._is_no_evidence_note("NO RELEVANT EVIDENCE")
        )
        self.assertTrue(
            research_service._is_no_evidence_note(
                "No useful info on this page about that."
            )
        )
        self.assertFalse(
            research_service._is_no_evidence_note("Real evidence here")
        )
        self.assertFalse(research_service._is_no_evidence_note(None))
        self.assertFalse(research_service._is_no_evidence_note(""))

    def test_irrelevant_notes_skipped_before_consolidation(self):
        import tempfile

        research_service._STOP_REQUESTED.clear()
        results = [
            {"title": "Relevant Page", "url": "http://r.example", "snippet": ""},
            {"title": "Irrelevant Page", "url": "http://i.example", "snippet": ""},
        ]

        async def fake_fetch(page, url):
            return {
                "title": "Relevant Page" if "r.example" in url else "Irrelevant Page",
                "text": "page text",
                "youtube": False,
            }

        def fake_summarize(query, title, url, text):
            if title == "Relevant Page":
                return "The price of X is $10 (confirmed 2026-09)."
            return "NO RELEVANT EVIDENCE"

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(research_service, "extract_brave_async",
                          new_callable=AsyncMock, return_value=results), \
             patch.object(research_service, "fetch_page_async",
                          side_effect=fake_fetch), \
             patch.object(research_service, "summarize_with_gemini",
                          side_effect=fake_summarize), \
             patch.object(research_service, "consolidate_summaries",
                          return_value="consolidated") as mock_cons, \
             patch.object(research_service, "build_spoken_summary",
                          return_value="spoken"), \
             patch.object(research_service, "build_details_markdown",
                          return_value="# details"), \
             patch.object(research_service, "_browser_run",
                          side_effect=run_with_fake_browser):
            research_service.run_research("what is the price of X",
                                          reports_dir=tmp)

        mock_cons.assert_called_once()
        items = mock_cons.call_args.args[1]
        titles = [it.get("result_title") for it in items]
        self.assertIn("Relevant Page", titles)
        self.assertNotIn("Irrelevant Page", titles)
        # The consolidation still receives the question.
        self.assertEqual(mock_cons.call_args.args[0], "what is the price of X")


if __name__ == "__main__":
    unittest.main()
