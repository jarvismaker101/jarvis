"""F53 — the research browser must not hard-depend on a system Chrome.

The shipped default channel was ``"chrome"``, which names a *system* install.
On a machine without Google Chrome every quick search and every deepsearch
died before it opened a page:

    BrowserType.launch_persistent_context: Chromium distribution 'chrome' is
    not found at C:\\Users\\...\\Google\\Chrome\\Application\\chrome.exe

...and the user only ever saw "I hit a snag while researching that, sir."
Edge ships with Windows and Playwright ships its own Chromium, so the launch
now degrades instead of failing:

* the requested channel is always tried first (no silent browser swap when
  the configuration is fine);
* then Edge, then the packaged Chromium;
* when every candidate fails the LAST real Playwright error is raised, so the
  failure still names a browser instead of being swallowed.
"""

import asyncio
import tempfile
import unittest
from unittest.mock import patch

from backend.services import research_browser


class _FakeContext:
    def __init__(self):
        self.pages = []
        self.closed = False

    async def close(self):
        self.closed = True


class _FakeChromium:
    def __init__(self, launch):
        self._launch = launch
        self.requested_channels = []

    async def launch_persistent_context(self, **kwargs):
        channel = kwargs.get("channel")
        self.requested_channels.append(channel)
        context = self._launch(channel)
        if isinstance(context, Exception):
            raise context
        return context


class _FakePlaywright:
    def __init__(self, launch):
        self.chromium = _FakeChromium(launch)
        self.stopped = False

    async def stop(self):
        self.stopped = True


class _FakeStarter:
    def __init__(self, playwright):
        self._playwright = playwright

    async def start(self):
        return self._playwright


def _missing(channel):
    return RuntimeError(
        "BrowserType.launch_persistent_context: Chromium distribution %r is "
        "not found" % channel)


class LaunchOrderTests(unittest.TestCase):
    def test_the_requested_channel_is_tried_first(self):
        self.assertEqual(research_browser.channel_launch_order("chrome"),
                         ["chrome", "msedge", None])

    def test_an_already_fallback_channel_is_not_duplicated(self):
        self.assertEqual(research_browser.channel_launch_order("msedge"),
                         ["msedge", None])
        self.assertEqual(research_browser.channel_launch_order(None),
                         ["chrome", "msedge", None])

    def test_no_channel_means_the_default_channel_not_the_bundled_one(self):
        self.assertEqual(research_browser.channel_launch_order(""),
                         ["chrome", "msedge", None])


class LaunchFallbackTests(unittest.TestCase):
    def _launch(self, launch):
        playwright = _FakePlaywright(launch)
        context, used = asyncio.run(research_browser.launch_persistent_context(
            playwright,
            profile_dir="/tmp/f53-profile",
            channel="chrome",
            headless=False,
            args=research_browser.LAUNCH_ARGS,
        ))
        return playwright, context, used

    def test_a_working_chrome_is_used_exactly_once(self):
        context = _FakeContext()
        playwright, launched, used = self._launch(lambda channel: context)
        self.assertIs(launched, context)
        self.assertEqual(used, "chrome")
        self.assertEqual(playwright.chromium.requested_channels, ["chrome"])

    def test_a_missing_chrome_falls_back_to_edge(self):
        context = _FakeContext()
        playwright, launched, used = self._launch(
            lambda channel: context if channel == "msedge" else _missing(channel))
        self.assertIs(launched, context)
        self.assertEqual(used, "msedge")
        self.assertEqual(playwright.chromium.requested_channels,
                         ["chrome", "msedge"])

    def test_edge_missing_too_falls_back_to_the_bundled_chromium(self):
        context = _FakeContext()
        playwright, launched, used = self._launch(
            lambda channel: context if channel is None else _missing(channel))
        self.assertIs(launched, context)
        self.assertIsNone(used)
        self.assertEqual(playwright.chromium.requested_channels,
                         ["chrome", "msedge", None])

    def test_every_candidate_failing_raises_the_last_real_error(self):
        playwright = _FakePlaywright(_missing)
        with self.assertRaises(RuntimeError) as caught:
            asyncio.run(research_browser.launch_persistent_context(
                playwright,
                profile_dir="/tmp/f53-profile",
                channel="chrome",
                headless=False,
                args=research_browser.LAUNCH_ARGS,
            ))
        self.assertIn("not found", str(caught.exception))
        self.assertEqual(playwright.chromium.requested_channels,
                         ["chrome", "msedge", None])


class WorkerFallbackTests(unittest.TestCase):
    """The worker must record the browser it actually got."""

    def _run_serve(self, launch):
        worker = research_browser.ResearchBrowserWorker(
            profile_dir=tempfile.mkdtemp(prefix="f53_profile_"),
            channel="chrome",
            idle_ttl=0,
        )
        # One pass through _serve: the idle loop exits immediately.
        worker._closing = True
        playwright = _FakePlaywright(launch)
        with patch("playwright.async_api.async_playwright",
                   return_value=_FakeStarter(playwright)), \
             patch("backend.services.browser_session_broker.register_session",
                   return_value={"session_id": "f53-session"}), \
             patch("backend.services.browser_session_broker.unregister_session",
                   return_value=True):
            asyncio.run(worker._serve())
        return worker, playwright

    def test_the_worker_records_the_fallback_it_used(self):
        worker, playwright = self._run_serve(
            lambda channel: _FakeContext() if channel == "msedge"
            else _missing(channel))
        self.assertEqual(worker.channel_used, "msedge")
        self.assertEqual(playwright.chromium.requested_channels,
                         ["chrome", "msedge"])
        self.assertTrue(playwright.stopped)

    def test_a_total_launch_failure_is_still_reported(self):
        worker = research_browser.ResearchBrowserWorker(
            profile_dir=tempfile.mkdtemp(prefix="f53_profile_"),
            channel="chrome",
            idle_ttl=0,
        )
        playwright = _FakePlaywright(_missing)
        with patch("playwright.async_api.async_playwright",
                   return_value=_FakeStarter(playwright)), \
             patch("backend.services.browser_session_broker.register_session",
                   return_value={"session_id": "f53-session"}):
            asyncio.run(worker._serve())
        self.assertEqual(worker._error is None, False)
        self.assertIn("not found", worker._error)
        self.assertIsNone(worker.channel_used)


if __name__ == "__main__":
    unittest.main()
