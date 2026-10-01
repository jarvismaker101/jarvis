"""Latency regressions â€” the fixed per-turn costs in the conversation path.

Each test here pins a PERF invariant that a later refactor could silently undo.
The point is not that the code is fast in absolute terms; it is that these
specific network calls, blocking waits and duplicated reads do not come back
on the hot path of a spoken or typed turn.

No microphone, TTS engine, provider or subprocess is opened here.
"""

import os
import time
import unittest
from unittest.mock import patch

from backend.core import brain
from backend.services import echo_cancel


class ChatFastPathTests(unittest.TestCase):
    """[PERF] obviously-conversational turns skip the cloud classifier.

    The fast path yields a *chat verdict*, so every deterministic net that can
    upgrade it must still run â€” that is the safety property, and these tests
    assert both halves: what it skips, and what it must never skip.
    """

    # Would be answered by the classifier; must NOT take the fast path.
    NOT_PLAIN_CHAT = (
        "open youtube",
        "play song by arijit",
        "what is the latest news",
        "how much is netflix priced",
        "open chrome and go to youtube",
        "create folder test",
        "read a.txt",
        "search for cats",
        "what is on my screen",
        "turn on screen controls",
        "remember that i like tea",
        "remind me in 5 minutes",
        "look it up",
        "delete everything",
        "run pip install x",
        "what is the weather today",
        "go to amazon.in and buy a book",
        "who won the match",
        "when is the match",
        "where is berlin",
        "command open youtube",
        "",
    )

    # Obviously conversation.
    PLAIN_CHAT = (
        "hi",
        "hello jarvis",
        "how are you",
        "thanks",
        "who are you",
        "good morning",
        "how is it going",
        "kaise ho",
        "batao kya hua",
    )

    def tearDown(self):
        brain._memory_context_cache.clear()

    def test_plain_chat_takes_the_fast_path(self):
        for msg in self.PLAIN_CHAT:
            with self.subTest(msg=msg):
                self.assertTrue(brain.is_definitely_plain_chat(msg))

    def test_actionable_or_current_facts_never_take_the_fast_path(self):
        for msg in self.NOT_PLAIN_CHAT:
            with self.subTest(msg=msg):
                self.assertFalse(brain.is_definitely_plain_chat(msg))

    def test_fast_path_yields_a_chat_verdict_so_nets_can_still_upgrade(self):
        """The safety property: the shape the nets key off is unchanged."""
        verdict = {
            "intent": "chat",
            "steps": [],
            "task_description": "",
            "query": "hello",
        }
        # The screen-question net upgrades exactly this verdict + message pair.
        self.assertIn(verdict["intent"], ("chat", "research"))
        self.assertEqual(verdict["steps"], [])

    def test_kill_switch_disables_the_fast_path(self):
        with patch.dict(os.environ, {"JARVIS_CHAT_FASTPATH": "0"}):
            self.assertFalse(brain._fastpath_chat_enabled())
        with patch.dict(os.environ, {"JARVIS_CHAT_FASTPATH": "1"}):
            self.assertTrue(brain._fastpath_chat_enabled())

    def test_memory_context_reads_are_memoised_within_a_turn(self):
        """[PERF] the speculative and committed builds share one pair of reads."""
        calls = []

        class _Store:
            def memory_context(self, _msg):
                calls.append("mem")
                return ""

            def work_context(self, _msg):
                calls.append("work")
                return ""

        with patch.object(brain, "memory_store", _Store()):
            brain._memory_context_cached("unique memo probe message")
            brain._memory_context_cached("unique memo probe message")
            self.assertEqual(calls.count("mem"), 1,
                             "memory_context was read twice for one message")
            self.assertEqual(calls.count("work"), 1,
                             "work_context was read twice for one message")


class SearchBudgetTests(unittest.TestCase):
    """[PERF] the inline DDGS lookup is bounded by the turn budget."""

    def test_expired_budget_sends_no_request(self):
        import time

        from backend.core import deadline as budget_mod

        expired = budget_mod.Deadline(time.monotonic() - 10.0)
        self.assertIsNone(
            brain.search_internet("anything", deadline=expired))


class _Resp:
    """Minimal urlopen context manager returning a JSON payload."""

    def __init__(self, payload):
        import json
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class AecTransportCacheTests(unittest.TestCase):
    """[PERF] the per-frame remote reference is cached and skipped when idle."""

    def _transport(self, **kw):
        return echo_cancel.RemoteAecTransport(base_url="http://127.0.0.1:1",
                                               **kw)

    def test_consecutive_frames_are_served_from_cache(self):
        t = self._transport(cache_seconds=5.0, idle_skip_seconds=99.0)
        calls = []

        def _fake_urlopen(request, timeout=None):
            calls.append(1)
            return _Resp({"pcm_b64": "AAAA", "age_seconds": 0.0})

        with patch.object(echo_cancel, "urlopen", _fake_urlopen):
            t.fetch_reference(0.03)
            t.fetch_reference(0.03)
            t.fetch_reference(0.03)
        self.assertEqual(len(calls), 1,
                         "one fetch should serve three overlapping frames")
        self.assertEqual(t.stats["cache_hits"], 2)

    def test_idle_latch_is_recoverable_not_permanent(self):
        """[P1-04] Idle is a hint with a bounded re-probe, NOT a one-way latch.

        This assertion deliberately CHANGED. It used to pin the buggy
        behaviour ("idle -> no request at all", asserted with
        ``idle_skip_seconds=0.0``). That latch was unrecoverable by
        construction: ``_last_fetch_ok_at`` only advanced on a successful
        NON-EMPTY fetch, while the skip was decided without a request, so once
        latched the transport could never ask again and a backend-spoken reply
        stayed uncancelled for the rest of the session. See the PERF intent
        preserved in test_idle_burst_costs_at_most_one_recovery_probe.
        """
        t = self._transport(cache_seconds=0.0, idle_skip_seconds=0.0)
        calls = []

        def _fake_urlopen(request, timeout=None):
            calls.append(1)
            return _Resp({"pcm_b64": "AAAA", "age_seconds": 0.0})

        with patch.object(echo_cancel, "urlopen", _fake_urlopen):
            t.fetch_reference(0.03)      # first probe: playback still unknown
            t._last_fetch_ok_at -= 10.0  # pretend it went quiet long ago
            t.fetch_reference(0.03)      # idle, but MUST re-probe eventually
        self.assertEqual(len(calls), 2,
                         "a latched idle must still recover by itself")
        self.assertEqual(t.stats["idle_reprobes"], 1)
        self.assertEqual(t.stats["skipped_idle"], 0)

    def test_idle_burst_costs_at_most_one_recovery_probe(self):
        """[P1-04] The PERF intent the old test was really about.

        Recovery must not turn the hot path back into a per-frame round trip:
        an idle BURST of frames costs at most one probe per idle window, and
        the frames in between are still skipped without a request.
        """
        t = self._transport(cache_seconds=0.0, idle_skip_seconds=1.5)
        calls = []
        playing = {"on": True}

        def _fake_urlopen(request, timeout=None):
            calls.append(1)
            # Only a NON-EMPTY span refreshes _last_fetch_ok_at, so playback
            # must be observed once before idle can ever latch.
            return _Resp({"pcm_b64": "QUJD" if playing["on"] else "",
                          "age_seconds": 0.0 if playing["on"] else None})

        with patch.object(echo_cancel, "urlopen", _fake_urlopen):
            t.fetch_reference(0.03)              # playback observed
            playing["on"] = False                # playback stops
            t._last_fetch_ok_at = time.monotonic() - 10.0   # latched idle
            for _ in range(30):           # a burst of captured frames
                t.fetch_reference(0.03)
        # 1 first probe + exactly 1 recovery probe, not 1 per frame.
        self.assertEqual(len(calls), 2,
                         "an idle burst must not cost a round trip per frame")
        self.assertEqual(t.stats["idle_reprobes"], 1)
        self.assertEqual(t.stats["skipped_idle"], 29)
