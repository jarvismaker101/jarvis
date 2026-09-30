"""[P1-04] RemoteAecTransport repair: auth, idle recovery, windowed cache,
circuit breaker, and the never-raise contract.

Four defects were fixed in ``backend/services/echo_cancel.py``:

* (a) IDLE LATCH - ``_last_fetch_ok_at`` advanced only on a successful
  NON-EMPTY fetch while the skip was decided WITHOUT a request, so once
  latched the transport could never ask again.
* (b) NO AUTH - ``Request(url, method="GET")`` carried no ``X-Jarvis-Token``,
  so every fetch 401'd behind the fail-closed local auth.
* (c) STALE CACHE - the TTL cache keyed on fetch time alone, ignoring
  ``mic_t_end``.
* (d) HOT-PATH FETCH - deliberately NOT threaded here (interacts with P0-06 /
  P0-04); mitigated by making the fetch strictly rarer.
"""

import json
import os
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from backend.services import echo_cancel, local_auth


class _Resp:
    """Minimal urlopen context manager returning a JSON payload."""

    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _BadResp:
    """urlopen stand-in whose body is not JSON."""

    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class RemoteTransportRepairTests(unittest.TestCase):
    """[P1-04] The four defects plus the circuit breaker."""

    def _transport(self, **kw):
        kw.setdefault("base_url", "http://127.0.0.1:1")
        return echo_cancel.RemoteAecTransport(**kw)

    # ── (b) AUTH ───────────────────────────────────────────────────────
    def test_request_carries_the_local_auth_header(self):
        """The 401 defect: the GET carried no auth header at all."""
        token = "p1-04-" + ("t" * 40)
        t = self._transport()
        seen = {}

        def _fake_urlopen(request, timeout=None):
            seen["headers"] = dict(getattr(request, "headers", {}) or {})
            return _Resp({"pcm_b64": "QUJD", "age_seconds": 0.0})

        with patch.dict(os.environ, {local_auth._ENV_VAR: token}), \
                patch.object(echo_cancel, "urlopen", _fake_urlopen):
            t.fetch_reference(0.03)
        headers = {str(k).lower(): v for k, v in seen["headers"].items()}
        self.assertIn(local_auth.HEADER.lower(), headers,
                      "remote reference fetch must be authenticated")
        self.assertEqual(headers[local_auth.HEADER.lower()], token)
        self.assertEqual(t.stats["errors"], 0)

    def test_auth_header_is_omitted_when_no_token_is_armed(self):
        """Unarmed (dev) mode: no token to send, and no error about it."""
        t = self._transport()
        seen = {}

        def _fake_urlopen(request, timeout=None):
            seen["headers"] = dict(getattr(request, "headers", {}) or {})
            return _Resp({"pcm_b64": "QUJD", "age_seconds": 0.0})

        with patch.dict(os.environ, {local_auth._ENV_VAR: ""}), \
                patch.object(echo_cancel, "urlopen", _fake_urlopen):
            pcm, _age = t.fetch_reference(0.03)
        self.assertEqual(pcm, b"ABC")
        self.assertEqual(t.stats["errors"], 0)

    def test_a_401_is_recorded_and_surfaced_not_swallowed(self):
        t = self._transport(breaker_failures=99)
        calls = []

        def _fake_urlopen(request, timeout=None):
            calls.append(1)
            raise HTTPError(request.full_url, 401, "Unauthorized", {}, None)

        with patch.object(echo_cancel, "urlopen", _fake_urlopen):
            pcm, age = t.fetch_reference(0.03)
        self.assertEqual((pcm, age), (b"", None),
                         "a 401 means no reference, never an exception")
        self.assertEqual(t.stats["auth_failures"], 1)
        self.assertEqual(t.stats["last_status"], 401)
        self.assertIn("401", t.stats["last_error"])
        self.assertEqual(t.state()["reason"], "auth_failed",
                         "state() must not pretend the reference is absent")

    # ── (a) IDLE LATCH ─────────────────────────────────────────────────
    def test_transport_recovers_after_going_idle(self):
        """The core acceptance property: idle must be escapable on its own.

        The fake models the real lifecycle: playback happens, then STOPS (the
        endpoint answers empty, which is why the idle timer ages out), then the
        backend speaks again. Only a non-empty span refreshes ``_last_fetch_ok_at``
        - exactly the condition that made the old latch unrecoverable.
        """
        t = self._transport(cache_seconds=0.0, idle_skip_seconds=0.2)
        calls = []
        playing = {"on": True}

        def _fake_urlopen(request, timeout=None):
            calls.append(1)
            if not playing["on"]:
                return _Resp({"pcm_b64": "", "age_seconds": None})
            return _Resp({"pcm_b64": "QUJD", "age_seconds": 0.0})

        with patch.object(echo_cancel, "urlopen", _fake_urlopen):
            self.assertEqual(t.fetch_reference(0.03)[0], b"ABC")   # speaking
            playing["on"] = False
            t._last_fetch_ok_at = time.monotonic() - 10.0         # quiet -> idle
            self.assertTrue(t._remote_is_idle(),
                            "playback stopped, so idle must now be reported")
            # The backend starts speaking again. Nothing but the transport's own
            # re-probe can discover this - the old latch made it unreachable.
            playing["on"] = True
            pcm, _age = t.fetch_reference(0.03)
        self.assertEqual(pcm, b"ABC",
                         "a backend-spoken reply must be fetchable again")
        self.assertGreaterEqual(t.stats["idle_reprobes"], 1)

    def test_idle_probe_is_bounded_to_one_per_window(self):
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
            t.fetch_reference(0.03)
            playing["on"] = False
            t._last_fetch_ok_at = time.monotonic() - 10.0   # latched idle
            for _ in range(200):            # 200 captured frames in one window
                t.fetch_reference(0.03)
        self.assertEqual(len(calls), 2, "one probe per idle window, not 200")
        self.assertEqual(t.stats["skipped_idle"], 199)

    # ── (c) STALE CACHE ────────────────────────────────────────────────
    def test_cache_is_not_replayed_onto_a_different_mic_window(self):
        t = self._transport(cache_seconds=30.0, idle_skip_seconds=99.0)
        calls = []

        def _fake_urlopen(request, timeout=None):
            calls.append(1)
            return _Resp({"pcm_b64": "QUJD", "age_seconds": 0.0})

        with patch.object(echo_cancel, "urlopen", _fake_urlopen):
            t.fetch_reference(0.03, mic_t_end=1000.0)
            t.fetch_reference(0.03, mic_t_end=1000.02)   # same window: cached
            before = len(calls)
            t.fetch_reference(0.03, mic_t_end=1040.0)   # elsewhere in time
        self.assertEqual(before, 1, "overlapping frames share the span")
        self.assertEqual(len(calls), 2,
                         "a span must not be served to another mic window")

    def test_cached_span_expires_past_the_drift_bound(self):
        """Age keeps growing after the fetch, so the cache must re-check it."""
        t = self._transport(cache_seconds=30.0, idle_skip_seconds=99.0)
        calls = []

        def _fake_urlopen(request, timeout=None):
            calls.append(1)
            return _Resp({"pcm_b64": "QUJD", "age_seconds": 1.9})

        with patch.object(echo_cancel, "urlopen", _fake_urlopen):
            t.fetch_reference(0.03, mic_t_end=1000.0)
            t._cache_at -= 0.2       # the span is now 2.1s old overall
            t.fetch_reference(0.03, mic_t_end=1000.0)
        self.assertEqual(
            len(calls), 2,
            "a span older than REFERENCE_MAX_DRIFT_SECONDS is stale")

    def test_cache_hit_is_still_served_for_overlapping_frames(self):
        """Constraint: the PERF cache must keep working as before."""
        t = self._transport(cache_seconds=5.0, idle_skip_seconds=99.0)
        calls = []

        def _fake_urlopen(request, timeout=None):
            calls.append(1)
            return _Resp({"pcm_b64": "QUJD", "age_seconds": 0.0})

        with patch.object(echo_cancel, "urlopen", _fake_urlopen):
            for i in range(5):
                t.fetch_reference(0.03, mic_t_end=1000.0 + (i * 0.01))
        self.assertEqual(len(calls), 1, "overlapping frames must share a fetch")
        self.assertEqual(t.stats["cache_hits"], 4)

    # ── circuit breaker ────────────────────────────────────────────────
    def test_breaker_opens_after_consecutive_failures_and_reports_it(self):
        t = self._transport(cache_seconds=0.0, idle_skip_seconds=99.0,
                            breaker_failures=3, breaker_cooldown_seconds=30.0)
        calls = []

        def _boom(request, timeout=None):
            calls.append(1)
            raise OSError("connection refused")

        with patch.object(echo_cancel, "urlopen", _boom):
            for _ in range(10):
                pcm, age = t.fetch_reference(0.03)
                self.assertEqual((pcm, age), (b"", None))
        self.assertEqual(len(calls), 3,
                         "the breaker must stop re-dialling a dead endpoint")
        state = t.state()
        self.assertTrue(state["breaker"]["open"])
        self.assertEqual(state["reason"], "circuit_open")
        self.assertGreater(state["breaker"]["cooldown_remaining_seconds"], 0.0)
        self.assertGreater(t.stats["circuit_skips"], 0)

    def test_breaker_reopens_after_cooldown_and_recovers(self):
        t = self._transport(cache_seconds=0.0, idle_skip_seconds=99.0,
                            breaker_failures=2, breaker_cooldown_seconds=0.05)

        def _boom(request, timeout=None):
            raise OSError("connection refused")

        with patch.object(echo_cancel, "urlopen", _boom):
            for _ in range(2):
                t.fetch_reference(0.03)
        self.assertTrue(t.state()["breaker"]["open"])
        time.sleep(0.08)

        def _ok(request, timeout=None):
            return _Resp({"pcm_b64": "QUJD", "age_seconds": 0.0})

        with patch.object(echo_cancel, "urlopen", _ok):
            pcm, _age = t.fetch_reference(0.03)
        self.assertEqual(pcm, b"ABC", "the transport must recover after cooldown")
        self.assertEqual(t.stats["consecutive_failures"], 0)
        self.assertFalse(t.state()["breaker"]["open"])

    def test_reachable_endpoint_clears_the_consecutive_failure_count(self):
        t = self._transport(cache_seconds=0.0, idle_skip_seconds=99.0,
                            breaker_failures=3)
        state = {"fail": True}

        def _urlopen(request, timeout=None):
            if state["fail"]:
                raise OSError("nope")
            return _Resp({"pcm_b64": "QUJD", "age_seconds": 0.0})

        with patch.object(echo_cancel, "urlopen", _urlopen):
            t.fetch_reference(0.03)
            self.assertEqual(t.stats["consecutive_failures"], 1)
            state["fail"] = False
            t.fetch_reference(0.03)
            self.assertEqual(t.stats["consecutive_failures"], 0)
            state["fail"] = True
            t.fetch_reference(0.03)
        self.assertEqual(t.stats["consecutive_failures"], 1,
                         "an empty answer is reachable, so it clears the count")
        self.assertFalse(t.state()["breaker"]["open"])

    # ── non-raising contract ───────────────────────────────────────────
    def test_transport_never_raises_on_garbage(self):
        t = self._transport(cache_seconds=0.0, idle_skip_seconds=0.0)

        def _garbage(request, timeout=None):
            return _BadResp(b"<html>not json</html>")

        with patch.object(echo_cancel, "urlopen", _garbage):
            self.assertEqual(t.fetch_reference(0.03), (b"", None))
            self.assertEqual(t.fetch_reference(None, mic_t_end="bad"),
                             (b"", None))
        self.assertGreaterEqual(t.stats["errors"], 1)

    def test_bad_base64_is_reported_not_raised(self):
        t = self._transport(cache_seconds=0.0, idle_skip_seconds=99.0)

        def _bad_b64(request, timeout=None):
            return _Resp({"pcm_b64": "!!!not-base64!!!", "age_seconds": 0.0})

        with patch.object(echo_cancel, "urlopen", _bad_b64):
            self.assertEqual(t.fetch_reference(0.03), (b"", None))
        self.assertEqual(t.state()["reason"], "bad_payload")

    # ── backward compatibility ─────────────────────────────────────────
    def test_existing_stat_keys_are_all_preserved(self):
        """Constraint: add keys, never rename or remove one."""
        t = self._transport()
        for key in ("fetches", "hits", "errors", "cache_hits", "skipped_idle",
                    "last_age_seconds", "last_error"):
            self.assertIn(key, t.stats, f"existing stat key {key!r} must survive")
            self.assertIn(key, t.state()["stats"])

    def test_state_shape_is_backward_compatible(self):
        t = self._transport()
        state = t.state()
        for key in ("url", "timeout", "stats"):
            self.assertIn(key, state)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
