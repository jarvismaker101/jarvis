"""F27 — keep one research browser worker warm.

Acceptance (audit report): "Real async navigation/extraction is awaited;
quick/deep reuse one context; polling stays responsive; timeout closes only
owned pages; restart cannot create competing profile owners."
"""

import asyncio
import inspect
import threading
import time
import unittest
from unittest.mock import patch

from backend.services import quick_search
from backend.services import research_browser


class _FakePage:
    def __init__(self, context, name="", url=""):
        self.context = context
        self.name = name
        self._url = url
        self.closed = False
        #: Records every attribute read so a test can prove a call was made.
        self.calls = []

    async def close(self):
        self.closed = True
        self.context.closed.append(self)

    @property
    def url(self):
        return self._url

    async def goto(self, *args, **kwargs):
        self.calls.append(("goto", args, kwargs))
        if args:
            self._url = args[0]

    async def close(self):
        self.closed = True
        self.context.closed.append(self)

    async def goto(self, *args, **kwargs):
        self.calls.append(("goto", args, kwargs))

    async def wait_for_selector(self, *args, **kwargs):
        self.calls.append(("wait_for_selector", args, kwargs))

    async def query_selector(self, selector):
        self.calls.append(("query_selector", selector))
        return None


class _FakeContext:
    def __init__(self):
        self.pages = []
        self.closed = []
        # Playwright persistent contexts spawn with one default blank page.
        self.pages.append(_FakePage(self, "default-blank", url="about:blank"))

    async def new_page(self):
        page = _FakePage(self, "page-%d" % len(self.pages))
        self.pages.append(page)
        return page

    @property
    def _all_pages(self):
        return self.pages


def _start_worker():
    """A real owner loop + fake context, without launching Chrome."""
    worker = research_browser.ResearchBrowserWorker(
        "/tmp/f27-profile", "chrome", idle_ttl=0)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    worker._loop = loop
    worker._thread = thread
    worker._context = _FakeContext()
    return worker, loop, thread


def _stop_worker(worker, loop, thread):
    worker._closing = True
    loop.call_soon_threadsafe(loop.stop)
    thread.join(2)
    loop.close()


class AsyncJobTests(unittest.TestCase):
    def setUp(self):
        self.worker, self.loop, self.thread = _start_worker()

    def tearDown(self):
        _stop_worker(self.worker, self.loop, self.thread)

    def test_a_submitted_async_body_is_actually_awaited(self):
        """F27: the old synchronous seam returned an un-awaited coroutine."""
        async def body(page):
            await page.goto("https://example.com")
            return "done"

        result = self.worker.submit(body)
        self.assertEqual(result, "done")
        page = self.worker._context.pages[0]
        self.assertIn(("goto", ("https://example.com",), {}), page.calls)

    def test_a_synchronous_body_still_works(self):
        result = self.worker.submit(lambda page: "sync result")
        self.assertEqual(result, "sync result")

    def test_run_awaits_a_real_async_job_and_closes_its_pages(self):
        async def body(task):
            page = await task.new_page()
            await page.goto("https://example.com")
            return page.name

        name = self.worker.run(body, task_id="job-a")
        self.assertEqual(name, "default-blank")
        self.assertTrue(self.worker._context.pages[0].closed)

    def test_timeout_cancels_the_job_and_closes_only_its_pages(self):
        async def body(task):
            await task.new_page()
            await asyncio.sleep(30)
            return "never"

        started = time.monotonic()
        with self.assertRaises(research_browser.ResearchBrowserError):
            self.worker.run(body, task_id="slow", timeout=0.3)
        self.assertLess(time.monotonic() - started, 5)
        # The cancelled job's page was closed by its own finally.
        self.assertTrue(self.worker._context.pages[0].closed)

    def test_one_task_cancel_leaves_another_tasks_pages_alone(self):
        release = threading.Event()

        async def short(task):
            page = await task.new_page()
            await asyncio.sleep(0.2)
            return page.name

        async def long(task):
            page = await task.new_page()
            await asyncio.sleep(0.05)
            while not release.is_set():
                await asyncio.sleep(0.02)
            return page.name

        results = []
        threads = [
            threading.Thread(target=lambda: results.append(
                self.worker.run(short, task_id="keep-a"))),
            threading.Thread(target=lambda: results.append(
                self.worker.run(long, task_id="keep-b"))),
        ]
        try:
            for thread in threads:
                thread.start()
            time.sleep(0.2)
            # Close only task a's pages, from another thread, while task b is
            # still running in the same context.
            future = asyncio.run_coroutine_threadsafe(
                self.worker.close_task_pages("keep-a"), self.loop)
            future.result(5)
            closed = {page.name for page in self.worker._context.closed}
            # keep-a adopted the default blank tab; keep-b opened its own.
            self.assertIn("default-blank", closed)
            self.assertNotIn("page-1", closed)
        finally:
            release.set()
            for thread in threads:
                thread.join(5)

    def test_the_owner_loop_is_not_blocked_by_a_job(self):
        """A sleeping job must not freeze the loop for other work."""
        async def slow(task):
            await asyncio.sleep(0.4)
            return "slow"

        async def fast(task):
            return "fast"

        outcome = {}
        thread = threading.Thread(
            target=lambda: outcome.update(slow=self.worker.run(slow, task_id="s")))
        thread.start()
        time.sleep(0.05)
        started = time.monotonic()
        fast_result = self.worker.run(fast, task_id="f")
        elapsed = time.monotonic() - started
        thread.join(5)
        self.assertEqual(fast_result, "fast")
        self.assertLess(elapsed, 0.3, "the fast job waited behind the slow one")


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.worker, self.loop, self.thread = _start_worker()

    def tearDown(self):
        _stop_worker(self.worker, self.loop, self.thread)

    def test_admission_is_bounded(self):
        release = threading.Event()

        async def hog(task):
            await asyncio.sleep(0.01)
            release.wait(5)
            return "done"

        threads = []
        for index in range(research_browser.MAX_CONCURRENT_JOBS):
            thread = threading.Thread(
                target=lambda i=index: self.worker.run(hog, task_id="hog-%d" % i))
            thread.start()
            threads.append(thread)
        time.sleep(0.3)
        with patch.object(research_browser, "JOB_ADMISSION_TIMEOUT", 0.2):
            with self.assertRaises(research_browser.ResearchBrowserError) as caught:
                self.worker.run(hog, task_id="overflow")
        self.assertIn("busy", str(caught.exception))
        release.set()
        for thread in threads:
            thread.join(6)

    def test_a_slot_is_released_after_a_job_ends(self):
        async def quick(task):
            return "ok"

        for _ in range(research_browser.MAX_CONCURRENT_JOBS + 2):
            self.assertEqual(self.worker.run(quick, task_id="q"), "ok")


class ShutdownOwnershipTests(unittest.TestCase):
    def test_shutdown_retains_ownership_when_the_owner_will_not_stop(self):
        worker = research_browser.ResearchBrowserWorker("/tmp/f27-x", "chrome")
        stop = threading.Event()

        def _stubborn():
            stop.wait(5)

        worker._thread = threading.Thread(target=_stubborn, daemon=True)
        worker._thread.start()
        try:
            self.assertFalse(worker.shutdown(wait=0.1))
        finally:
            stop.set()
            worker._thread.join(3)

    def test_a_restart_cannot_create_a_second_owner_on_the_same_profile(self):
        saved_worker = research_browser._worker
        saved_key = research_browser._worker_key
        stubborn = research_browser.ResearchBrowserWorker("/tmp/f27-old", "chrome")
        research_browser._worker = stubborn
        research_browser._worker_key = ("/tmp/f27-old", "chrome")
        try:
            with patch.object(type(stubborn), "shutdown", return_value=False):
                with self.assertRaises(research_browser.ResearchBrowserError):
                    research_browser.get_worker(
                        profile_dir="/tmp/f27-new", channel="chrome")
            # The stubborn owner is still the registered one.
            self.assertIs(research_browser._worker, stubborn)
        finally:
            research_browser._worker = saved_worker
            research_browser._worker_key = saved_key

    def test_shutdown_returns_true_once_the_worker_is_gone(self):
        worker = research_browser.ResearchBrowserWorker("/tmp/f27-done", "chrome")
        stop = threading.Event()
        worker._thread = threading.Thread(
            target=lambda: stop.wait(5), daemon=True)
        worker._thread.start()
        stop.set()
        self.assertTrue(worker.shutdown(wait=2))


class BlankTabReuseTests(unittest.TestCase):
    """Live bug: every lookup opened a second tab (the persistent context's
    default about:blank was never touched) and cleanup closed only tracked
    pages — so one blank tab survived every task."""

    def setUp(self):
        self.worker, self.loop, self.thread = _start_worker()

    def tearDown(self):
        _stop_worker(self.worker, self.loop, self.thread)

    def test_new_page_reuses_the_default_blank_tab(self):
        async def body(task):
            first = await task.new_page()
            second = await task.new_page()
            return first, second

        first, second = self.worker.run(body, task_id="reuse")
        # First call adopts the default blank tab; the second has no blank
        # left, so it opens a real new page.
        self.assertEqual(first.name, "default-blank")
        self.assertEqual(second.name, "page-1")
        self.assertEqual(len(self.worker._context.pages), 2)

    def test_no_blank_tab_survives_the_job(self):
        async def body(task):
            page = await task.new_page()
            await page.goto("https://example.com")
            return "ok"

        self.assertEqual(self.worker.run(body, task_id="clean"), "ok")
        for page in self.worker._context.pages:
            self.assertTrue(page.closed, page.name)

    def test_a_real_url_page_is_never_mistaken_for_blank(self):
        busy = _FakePage(self.worker._context, "busy",
                         url="https://example.com")
        self.worker._context.pages.append(busy)

        async def body(task):
            return await task.new_page()

        page = self.worker.run(body, task_id="other")
        # The default blank is adopted (not the busy page); the busy page
        # stays open — it may belong to a concurrent task.
        self.assertEqual(page.name, "default-blank")
        self.assertFalse(busy.closed)

    def test_submit_reuses_the_blank_tab_too(self):
        seen = self.worker.submit(lambda page: page.name)
        self.assertEqual(seen, "default-blank")
        self.assertTrue(self.worker._context.pages[0].closed)


class QuickSearchAsyncTests(unittest.TestCase):
    """The quick-search job body must be a real coroutine."""

    def setUp(self):
        # A leaked stop flag from another test would short-circuit the job
        # before it opens a page (the job body checks it first, by design).
        from backend.services import research_service

        research_service.clear_stop_request()

    def tearDown(self):
        from backend.services import research_service

        research_service.clear_stop_request()

    def test_the_job_body_is_a_coroutine_function(self):
        self.assertTrue(inspect.iscoroutinefunction(quick_search._quick_search_job))

    def test_the_page_helpers_are_coroutine_functions(self):
        for name in ("extract_ai_overview", "extract_top_snippet",
                     "_open_google", "_click_ask_tab", "_wait_for_ai_completion",
                     "_answer_container_text", "_wait_for_answer_update"):
            self.assertTrue(
                inspect.iscoroutinefunction(getattr(quick_search, name)),
                "%s must be async so its page calls can be awaited" % name)

    def test_the_job_navigates_and_extracts_through_awaits(self):
        page = _FakePage(_FakeContext(), "job")

        class _Task:
            async def new_page(self):
                return page

        stop = patch.object(quick_search, "_wait_for_ai_completion",
                            new=_async_return(None))
        with stop:
            overview, fallback = asyncio.run(
                quick_search._quick_search_job(_Task(), "query"))
        self.assertIsNone(overview)
        self.assertTrue(any(call[0] == "goto" for call in page.calls))


def _async_return(value):
    async def _inner(*_args, **_kwargs):
        return value
    return _inner


class _DeadNewPageContext:
    """A context that looks alive but every new_page dies like a closed
    browser (the user closed the window without Playwright flagging it)."""

    def __init__(self):
        self.pages = []
        self.closed = []

    async def new_page(self):
        raise RuntimeError(
            "Target page, context or browser has been closed")

    async def close(self):
        pass


class RelaunchTests(unittest.TestCase):
    """Live fix: a closed browser is reopened, not reported as a snag."""

    def setUp(self):
        self.worker, self.loop, self.thread = _start_worker()
        self.worker._playwright = object()
        self.launches = []

    def tearDown(self):
        _stop_worker(self.worker, self.loop, self.thread)

    def _fake_launch(self):
        async def fake(playwright, *, profile_dir, channel, headless,
                       args=()):
            self.launches.append(channel)
            return _FakeContext(), "chrome"
        return patch.object(research_browser, "launch_persistent_context",
                            fake)

    def test_a_closed_context_is_reopened_for_the_next_job(self):
        old = _FakeContext()

        def is_closed():
            return True

        old.is_closed = is_closed
        self.worker._context = old

        async def body(task):
            page = await task.new_page()
            return page.name

        with self._fake_launch():
            name = self.worker.run(body, task_id="reopen")
        self.assertEqual(name, "default-blank")
        self.assertEqual(len(self.launches), 1)
        self.assertIsNot(self.worker._context, old)

    def test_a_mid_job_close_error_reopens_and_retries_once(self):
        self.worker._context = _DeadNewPageContext()

        async def body(task):
            page = await task.new_page()
            return page.name

        with self._fake_launch():
            name = self.worker.run(body, task_id="retry")
        self.assertEqual(name, "default-blank")
        self.assertEqual(len(self.launches), 1)
        self.assertNotIsInstance(self.worker._context, _DeadNewPageContext)


if __name__ == "__main__":
    unittest.main()
