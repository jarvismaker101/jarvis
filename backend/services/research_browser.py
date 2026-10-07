"""Fable-5 audit G5 / F27 — one long-lived research browser worker.

Before this module every lookup paid for its own Chrome:

  * ``quick_search.run_quick_search`` opened a persistent context on the
    research profile and closed it again at the end of every single search;
  * ``research_service.run_research`` did the same for the deep phase, so a
    deepsearch paid TWO cold starts (overview, then sites);
  * the profile directory can only host one persistent context at a time, so
    the two phases were forced to be strictly sequential.

This module owns the profile behind ONE long-lived worker:

  * a single daemon thread runs one asyncio event loop;
  * that loop starts ``async_playwright`` once and keeps ONE persistent
    context alive for the whole process;
  * callers submit *jobs* instead of launching browsers — ``submit`` for a
    synchronous job body, ``run`` for a coroutine that wants to drive several
    pages concurrently (F28).

Playwright objects therefore never cross an arbitrary thread boundary: sync
job bodies execute ON the owner thread, and async job bodies execute on the
owner loop. That is the "never share sync Playwright objects across arbitrary
worker threads" half of F27.

Cancellation is per task: every job gets a ``task_id`` and every page it opens
is tracked under that id, so :func:`close_task_pages` closes only that task's
pages and leaves the shared context (and any other task's pages) alone.
"""

import asyncio
import inspect
import logging
import os
import threading
import time
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_PROFILE_DIR = _REPO_ROOT / "data" / "chrome_profile_jarvis"
DEFAULT_CHANNEL = "chrome"

#: Seconds a warmed context is kept after its last job before it is retired.
DEFAULT_IDLE_TTL = 180.0

#: Hard ceiling on a single job. Callers pass their own deadline-derived value.
DEFAULT_JOB_TIMEOUT = 420.0

#: F27: bounded admission. At most this many jobs may be in flight on the one
#: owner loop, so a burst of requests cannot open an unbounded number of pages.
MAX_CONCURRENT_JOBS = 4
#: How long a caller waits for a free job slot before being refused.
JOB_ADMISSION_TIMEOUT = 30.0
#: How long a cancelled job gets to close the pages it owned before the
#: caller is handed the timeout error.
CANCEL_CLEANUP_TIMEOUT = 5.0


class _JobSlot:
    """Context manager releasing one admission slot exactly once."""

    def __init__(self, semaphore):
        self._semaphore = semaphore
        self._released = False

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        if not self._released:
            self._released = True
            try:
                self._semaphore.release()
            except ValueError:
                pass
        return False

#: Same flags the two callers used when they launched their own context.
LAUNCH_ARGS = (
    "--start-maximized",
    "--disable-extensions",
    "--disable-gpu",
    # Public WiFi often MITMs TLS with its own root cert; without this whole
    # domains fail with ERR_CERT_AUTHORITY_INVALID.
    "--ignore-certificate-errors",
)

#: Channels tried, in order, when the requested one cannot launch.
#:
#: ``channel`` names a *system* browser install, so the shipped default
#: ("chrome") hard-failed on any machine without Google Chrome: every quick
#: search and every deepsearch died with "Chromium distribution 'chrome' is
#: not found at ...chrome.exe", even when Edge (present on every Windows box)
#: and Playwright's own Chromium were installed and perfectly usable. The
#: requested channel is always tried first, then Edge, then the packaged
#: Chromium (``None``).
FALLBACK_CHANNELS = ("msedge", None)


def channel_launch_order(channel):
    """Requested channel first, then the fallbacks — without duplicates."""
    order = [channel or DEFAULT_CHANNEL]
    for fallback in FALLBACK_CHANNELS:
        if fallback not in order:
            order.append(fallback)
    return order


def _describe_channel(channel):
    return "bundled chromium" if channel is None else "channel %r" % channel


async def launch_persistent_context(playwright, *, profile_dir, channel,
                                   headless, args=()):
    """Launch the persistent context, degrading when a browser is absent.

    Returns ``(context, channel_used)``. Only the LAUNCH is retried across
    channels — a context that started is never re-launched behind the caller's
    back, so a failure later in the run is reported exactly as it happened.
    When every candidate fails, the last error is raised so the caller still
    sees a real Playwright message instead of a generic one.
    """
    order = channel_launch_order(channel)
    last_exc = None
    for index, candidate in enumerate(order):
        try:
            context = await maybe_await(
                playwright.chromium.launch_persistent_context(
                    user_data_dir=str(profile_dir),
                    channel=candidate,
                    headless=headless,
                    args=list(args),
                ))
        except Exception as exc:  # noqa: BLE001 - the next candidate may work
            last_exc = exc
            if index + 1 < len(order):
                logging.warning(
                    "[RESEARCH] %s could not launch (%s); trying %s",
                    _describe_channel(candidate), exc,
                    _describe_channel(order[index + 1]))
            continue
        if index:
            logging.warning("[RESEARCH] using fallback browser %s",
                            _describe_channel(candidate))
        return context, candidate
    raise last_exc


class ResearchBrowserError(RuntimeError):
    """The warm research browser could not start or serve a job."""


async def maybe_await(value):
    """Await *value* only when it actually is awaitable.

    Lets one job body drive a real async Playwright page AND the plain
    MagicMock pages the unit tests inject, without two code paths.
    """
    if inspect.isawaitable(value):
        return await value
    return value


def _context_is_closed(context):
    """True when the warm browser context is gone (user closed the window).

    Playwright's ``BrowserContext.is_closed()`` is authoritative when it
    exists; test fakes without the full API are assumed alive so they keep
    exercising the reuse path.
    """
    if context is None:
        return True
    checker = getattr(context, "is_closed", None)
    if not callable(checker):
        return False
    try:
        verdict = checker()
    except Exception:
        return True
    return verdict if isinstance(verdict, bool) else False


def _is_closed_error(exc):
    """The Playwright error raised when the browser was closed mid-call."""
    text = str(exc or "").lower()
    return ("has been closed" in text or "browser closed" in text
            or "context or browser" in text)


def _idle_ttl():
    raw = os.getenv("JARVIS_RESEARCH_IDLE_TTL")
    if not raw:
        return DEFAULT_IDLE_TTL
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_IDLE_TTL
    return value if value > 0 else 0.0


class ResearchTask:
    """Handle handed to an async job body.

    Job bodies must open pages through :meth:`new_page` so cancellation can
    find and close exactly the pages this job owns.
    """

    def __init__(self, worker, task_id):
        self.worker = worker
        self.task_id = task_id
        self.context = None
        self._pages = []

    async def new_page(self, _retried=False):
        if self.context is None:
            raise ResearchBrowserError("research task has no browser context")
        # A persistent context always spawns with one default about:blank
        # page. Reuse it instead of opening a second tab: every job used to
        # call new_page() (leaving the blank tab behind — cleanup only
        # closes tracked pages, so the blank survived every task).
        try:
            existing = await maybe_await(self.context.pages)
        except Exception:
            existing = None
        if existing:
            for page in list(existing):
                if page in self._pages:
                    continue
                try:
                    url = await maybe_await(page.url)
                except Exception:
                    url = ""
                if url in ("about:blank", "", None):
                    self._pages.append(page)
                    self.worker._track_page(self.task_id, page)
                    return page
        try:
            page = await maybe_await(self.context.new_page())
        except Exception as exc:
            # The window was closed between the liveness check and this call
            # (or mid-job): reopen the warm context once and retry.
            recover = getattr(self.worker, "_recover_context", None)
            if recover is None or _retried or not _is_closed_error(exc):
                raise
            self.context = await recover()
            return await self.new_page(_retried=True)
        # Tracked twice on purpose: under the task id so an outside cancel
        # can find it, and on the task itself so an unnamed job still gets
        # its pages closed when it ends.
        self._pages.append(page)
        self.worker._track_page(self.task_id, page)
        return page

    async def close_own_pages(self):
        pages, self._pages = self._pages, []
        for page in pages:
            try:
                await maybe_await(page.close())
            except Exception:
                pass

    async def close_pages(self):
        # The task's tracked pages go — and so does any leftover blank page
        # nobody ever adopted (e.g. the persistent context's default page
        # from a run that predates the new_page() reuse above, or a blank
        # opened by other means). Only about:blank is fair game: a page
        # with a real URL may belong to a concurrent task sharing the
        # context. Belts-and-braces on top of the reuse — never the only
        # mechanism, so a regression in new_page() cannot strand tabs.
        await self.worker.close_task_pages(self.task_id)
        await self.close_own_pages()
        try:
            existing = await maybe_await(self.context.pages)
        except Exception:
            return
        if not existing:
            return
        for page in list(existing):
            try:
                url = await maybe_await(page.url)
            except Exception:
                continue
            if url in ("about:blank", "", None):
                try:
                    await maybe_await(page.close())
                except Exception:
                    pass


class ResearchBrowserWorker:
    """One daemon thread + one event loop + one persistent context."""

    def __init__(self, profile_dir=None, channel=None, idle_ttl=None):
        self.profile_dir = str(profile_dir or DEFAULT_PROFILE_DIR)
        self.channel = channel or DEFAULT_CHANNEL
        #: The channel the live context actually launched with (the requested
        #: one, or the fallback that worked); None until a context exists.
        self.channel_used = None
        self.idle_ttl = _idle_ttl() if idle_ttl is None else idle_ttl
        self._thread = None
        self._loop = None
        self._ready = threading.Event()
        self._error = None
        self._closing = False
        self._last_used = time.monotonic()
        self._lock = threading.RLock()
        self._pages = {}
        self._context = None
        self._playwright = None
        self._relaunch_lock = None
        self._broker_session_id = None
        #: F27: bounded admission for concurrent jobs on the one owner loop.
        self._slots = threading.BoundedSemaphore(MAX_CONCURRENT_JOBS)

    # ── lifecycle ──────────────────────────────────────────────────────────
    @property
    def alive(self):
        return self._thread is not None and self._thread.is_alive()

    async def _ensure_context(self, playwright):
        """A live persistent context, relaunching when the window was closed.

        The user may close the browser between jobs; the worker is then left
        holding a dead BrowserContext. Instead of failing the next job with
        "Target page, context or browser has been closed", the same warm
        worker reopens the persistent context on the same profile.
        """
        context = self._context
        if context is not None and not _context_is_closed(context):
            return context
        if playwright is None:
            raise ResearchBrowserError(
                "research browser is closed and there is no Playwright "
                "handle to reopen it with")
        if self._relaunch_lock is None:
            self._relaunch_lock = asyncio.Lock()
        async with self._relaunch_lock:
            context = self._context
            if context is not None and not _context_is_closed(context):
                return context
            if context is not None:
                logging.warning(
                    "[RESEARCH] warm browser was closed; reopening it now")
                try:
                    await maybe_await(context.close())
                except Exception:
                    pass
            context, channel_used = await launch_persistent_context(
                playwright,
                profile_dir=self.profile_dir,
                channel=self.channel,
                headless=False,
                args=LAUNCH_ARGS,
            )
            self.channel_used = channel_used
            self._context = context
            return context

    async def _recover_context(self):
        """Force a fresh context after a dead-browser error mid-job."""
        self._context = None
        return await self._ensure_context(self._playwright)

    def start(self, timeout=60.0):
        with self._lock:
            if self.alive:
                return
            self._closing = False
            self._error = None
            self._ready = threading.Event()
            self._thread = threading.Thread(
                target=self._thread_main, name="research-browser", daemon=True)
            self._thread.start()
        if not self._ready.wait(timeout):
            raise ResearchBrowserError(
                "research browser worker did not start within %.0fs" % timeout)
        if self._error:
            raise ResearchBrowserError(str(self._error))

    def _thread_main(self):
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._serve())
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            try:
                asyncio.set_event_loop(None)
            except Exception:
                pass
            loop.close()
            self._loop = None

    async def _serve(self):
        from playwright.async_api import async_playwright

        playwright = None
        try:
            playwright = await async_playwright().start()
            self._playwright = playwright
            await self._ensure_context(playwright)
        except Exception as exc:  # noqa: BLE001 - surfaced to the submitting thread
            self._error = "%s: %s" % (type(exc).__name__, exc)
            self._ready.set()
            self._playwright = None
            if playwright is not None:
                try:
                    await playwright.stop()
                except Exception:
                    pass
            return
        self._ready.set()
        # F47: this worker is now the OWNER of the profile — the broker
        # refuses any other actor that tries to claim the same persistent
        # profile while we hold it.
        try:
            from backend.services import browser_session_broker

            self._broker_session_id = browser_session_broker.register_session(
                owner="research",
                profile_dir=self.profile_dir,
                channel=self.channel_used or self.channel,
                kind="persistent-context",
                label="warm research browser (F27)",
            )["session_id"]
        except Exception:
            self._broker_session_id = None
        try:
            while not self._closing:
                await asyncio.sleep(0.5)
                if self.idle_ttl and (time.monotonic() - self._last_used) > self.idle_ttl:
                    if self._busy():
                        continue
                    break
        finally:
            final_context = self._context
            self._context = None
            self._playwright = None
            if self._broker_session_id:
                try:
                    from backend.services import browser_session_broker

                    browser_session_broker.unregister_session(self._broker_session_id)
                except Exception:
                    pass
                self._broker_session_id = None
            try:
                if final_context is not None:
                    await maybe_await(final_context.close())
            except Exception:
                pass
            try:
                await playwright.stop()
            except Exception:
                pass

    def _busy(self):
        return any(pages for pages in self._pages.values())

    # ── pages ──────────────────────────────────────────────────────────────
    def _track_page(self, task_id, page):
        if not task_id:
            return
        with self._lock:
            self._pages.setdefault(task_id, []).append(page)

    async def close_task_pages(self, task_id):
        """Close every page opened by *task_id* — and nothing else.

        The shared context is never closed here: another task may be mid-run
        in it, and the whole point of this module is that the context outlives
        any single task.
        """
        if not task_id:
            return
        with self._lock:
            pages = self._pages.pop(task_id, [])
        for page in pages:
            try:
                await maybe_await(page.close())
            except Exception:
                pass

    # ── job entry points ───────────────────────────────────────────────────
    def _ensure_started(self):
        with self._lock:
            if not self.alive:
                self.start()
            return self._loop

    def _result(self, future, timeout, task_id=None, cancel=None):
        """Wait for a submitted job, cancelling it when the budget expires.

        F27: a timeout used to just stop WAITING — the submitted coroutine kept
        running on the owner loop, holding its pages open and competing with
        the next job. Now the job's own asyncio task is cancelled, so its
        ``finally`` closes exactly the pages it owned.
        """
        self._last_used = time.monotonic()
        try:
            return future.result(timeout)
        except FuturesTimeout:
            # ``Future.cancel()`` cannot stop a coroutine that has already
            # started, so cancel the asyncio task on its own loop.
            if callable(cancel):
                try:
                    cancel()
                except Exception:
                    pass
            else:
                future.cancel()
            # Give the cancelled job a bounded moment to run its own cleanup
            # (closing the pages it owned) before the caller is handed an
            # error and may move on to another job in the same context.
            try:
                future.result(CANCEL_CLEANUP_TIMEOUT)
            except Exception:
                pass
            logging.warning(
                "[RESEARCH-BROWSER] job %s exceeded %ss; cancelled",
                task_id or "(unnamed)", timeout)
            raise ResearchBrowserError(
                "research browser job timed out after %ss" % timeout)

    def run(self, coro_fn, task_id=None, timeout=None):
        """Run ``await coro_fn(task)`` on the owner loop and return its result.

        Use this for concurrent work: the coroutine may open several pages at
        once via :meth:`ResearchTask.new_page`.
        """
        loop = self._ensure_started()
        holder = {}

        async def _job():
            holder["task"] = asyncio.current_task()
            self._last_used = time.monotonic()
            try:
                self._context = await self._ensure_context(self._playwright)
            except Exception as exc:
                raise ResearchBrowserError(
                    "could not open the research browser: %s" % exc)
            task = ResearchTask(self, task_id)
            task.context = self._context
            try:
                return await coro_fn(task)
            finally:
                # Normal or cancelled: this task's pages go, the context stays.
                await task.close_pages()

        with self._job_slot(task_id):
            future = asyncio.run_coroutine_threadsafe(_job(), loop)
            return self._result(
                future, timeout or DEFAULT_JOB_TIMEOUT, task_id,
                cancel=self._canceller(loop, holder))

    def _canceller(self, loop, holder):
        """A thread-safe callable cancelling the job's own asyncio task."""
        def _cancel():
            task = holder.get("task")
            if task is not None:
                loop.call_soon_threadsafe(task.cancel)
        return _cancel

    def submit(self, fn, task_id=None, timeout=None):
        """Run ``fn(page)`` synchronously ON the owner thread.

        For job bodies that drive a single page. The page is opened (reusing
        the persistent context's default blank tab when it is still blank,
        so jobs no longer strand an about:blank tab per run) and tracked
        before *fn* runs and closed afterwards, so ``fn`` never touches a
        Playwright object from a foreign thread.

        F27: if *fn* returns an awaitable it is AWAITED here on the owner loop
        instead of being returned un-awaited. The synchronous variant used to
        hand back a coroutine object that nobody ever ran — the job silently
        did nothing while reporting success. New code should prefer
        :meth:`run` with a real ``async def`` body.
        """
        loop = self._ensure_started()
        holder = {}

        async def _job():
            holder["task"] = asyncio.current_task()
            self._last_used = time.monotonic()
            try:
                self._context = await self._ensure_context(self._playwright)
            except Exception:
                self._context = None
            task = ResearchTask(self, task_id)
            task.context = self._context
            try:
                page = await task.new_page()
            except Exception:
                page = None
            if page is None:
                # No context page available: run the body pageless rather
                # than failing the whole job on the open call.
                try:
                    return await maybe_await(fn(None))
                finally:
                    await task.close_pages()
                return
            self._track_page(task_id, page)
            try:
                result = fn(page)
                # An async body: await it here so the work actually happens.
                return await maybe_await(result)
            finally:
                if task_id:
                    await self.close_task_pages(task_id)
                else:
                    # Unnamed job: nothing tracks the page, so close it here
                    # or it leaks for the lifetime of the warm context.
                    try:
                        await maybe_await(page.close())
                    except Exception:
                        pass

        with self._job_slot(task_id):
            future = asyncio.run_coroutine_threadsafe(_job(), loop)
            return self._result(
                future, timeout or DEFAULT_JOB_TIMEOUT, task_id,
                cancel=self._canceller(loop, holder))

    # ── bounded admission (F27) ────────────────────────────────────────────
    def _job_slot(self, task_id=None):
        """Bound how many jobs may be in flight on the one owner loop.

        F27: admission was unbounded — every caller opened another page on the
        shared context at once, which is how one slow job starved the rest.
        """
        acquired = self._slots.acquire(timeout=JOB_ADMISSION_TIMEOUT)
        if not acquired:
            raise ResearchBrowserError(
                "research browser is busy (%d jobs in flight); try again shortly"
                % MAX_CONCURRENT_JOBS)
        return _JobSlot(self._slots)

    # ── shutdown ───────────────────────────────────────────────────────────
    def shutdown(self, wait=10.0):
        """Retire the context and stop the worker thread (idempotent).

        F27: ownership is retained until the thread has ACTUALLY stopped. The
        old version cleared ``_worker`` before knowing whether the shutdown
        completed, so a slow/blocked owner was forgotten while its Chrome was
        still holding the profile — and the next request started a second
        owner on the same user-data directory. Returns True when the worker is
        gone.
        """
        self._closing = True
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(wait)
        still_alive = thread is not None and thread.is_alive()
        if still_alive:
            # Do NOT report success: the caller keeps the reference so the
            # next request reuses this owner instead of starting a rival.
            logging.warning(
                "[RESEARCH-BROWSER] shutdown timed out; owner still alive")
            return False
        with self._lock:
            self._thread = None
            self._context = None
            self._pages = {}
        if self._broker_session_id:
            # Defensive: the serve loop normally releases the claim in its
            # finally — this covers a thread that died without running it.
            try:
                from backend.services import browser_session_broker

                browser_session_broker.unregister_session(self._broker_session_id)
            except Exception:
                pass
            self._broker_session_id = None
        return True


# ── module-level singleton ─────────────────────────────────────────────────
_lock = threading.RLock()
_worker = None
_worker_key = None


def _key(profile_dir, channel):
    return (str(profile_dir), channel or DEFAULT_CHANNEL)


def get_worker(profile_dir=None, channel=None):
    """The process-wide warm worker for *profile_dir*.

    A changed profile/channel retires the old worker — one Chrome profile can
    only host one persistent context, so there is nothing to reuse across a
    switch.

    F27: if the old worker refuses to stop, it is KEPT as the owner (see
    :meth:`ResearchBrowserWorker.shutdown`) rather than replaced, so a restart
    can never create a competing owner on the same profile.
    """
    global _worker, _worker_key
    profile = str(profile_dir or os.getenv("JARVIS_RESEARCH_PROFILE") or DEFAULT_PROFILE_DIR)
    chan = channel or os.getenv("JARVIS_RESEARCH_CHANNEL", DEFAULT_CHANNEL)
    key = _key(profile, chan)
    with _lock:
        if _worker is not None and _worker_key != key:
            if _worker.shutdown():
                _worker = None
            else:
                raise ResearchBrowserError(
                    "the warm research browser for %s would not stop; refusing "
                    "to start a second owner on the same profile"
                    % (_worker_key[0] if _worker_key else "the current profile"))
        if _worker is None:
            _worker = ResearchBrowserWorker(profile, chan)
            _worker_key = key
        return _worker


def run(coro_fn, task_id=None, timeout=None, profile_dir=None, channel=None):
    """Submit an async job body to the warm worker (see :meth:`run`)."""
    return get_worker(profile_dir, channel).run(coro_fn, task_id=task_id, timeout=timeout)


def is_warm(profile_dir=None, channel=None):
    """True when a warm worker for this profile is ALREADY running.

    Read-only: unlike :func:`get_worker` this never starts a worker. Callers
    that only want to *reuse* a live browser (executor.route_open) use this
    so a cold open can never spin one up as a side effect.
    """
    with _lock:
        worker = _worker
    if worker is None:
        return False
    thread = getattr(worker, "_thread", None)
    return bool(thread is not None and thread.is_alive())


def submit(fn, task_id=None, timeout=None, profile_dir=None, channel=None):
    """Submit a sync job body to the warm worker (see :meth:`submit`)."""
    return get_worker(profile_dir, channel).submit(fn, task_id=task_id, timeout=timeout)


def close_task_pages(task_id):
    """Close only *task_id*'s pages. Safe to call from any thread."""
    with _lock:
        worker = _worker
    if worker is None or not task_id:
        return
    loop = getattr(worker, "_loop", None)
    if loop is None:
        return
    try:
        asyncio.run_coroutine_threadsafe(
            worker.close_task_pages(task_id), loop).result(10)
    except Exception:
        pass


def shutdown():
    """Retire the warm worker (tests, profile switches, process exit).

    F27: the module-level reference is only dropped once the worker confirms
    it stopped, so a shutdown that times out cannot leave the profile owned by
    a forgotten worker.
    """
    global _worker, _worker_key
    with _lock:
        worker = _worker
    if worker is not None:
        if not worker.shutdown():
            return False
    with _lock:
        if _worker is worker:
            _worker = None
            _worker_key = None
    return True
