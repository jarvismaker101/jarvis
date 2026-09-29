"""[PERF] Per-turn latency telemetry.

The conversation path is a chain of independent costs — endpointing, STT,
intent classification, first model token, first TTS audio — and until now none
of them was observable. Without numbers, "make it faster" is guesswork and a
regression is invisible.

This module is a tiny, dependency-free ring of completed-turn records plus a
thread-safe "in flight" span set, so a caller can mark a boundary once and have
it attributed to the turn that is currently running:

    t = latency.begin("s3f1…", origin="voice")
    t.mark("stt_done")
    ...
    t.mark("first_audio")
    t.finish()          # publishes one record

Records are bounded (oldest dropped) and cheap; nothing here raises into a
caller, and a failure to record can never fail a turn. ``GET /latency`` serves
the summary plus the recent raw records.

Pure stdlib by design: this sits on the request path.
"""

import threading
import time
from collections import deque

#: Bounded history. One voice turn is ~5s, so this is a few minutes of detail.
MAX_RECORDS = 200

#: Canonical span order. ``end`` is the sum of the measured spans so a record
#: can be read without knowing the wall clock; gaps are normal (not every turn
#: has every span).
SPAN_ORDER = (
    "endpoint",     # trailing silence before commit
    "stt",          # transcript acquired
    "classify",     # intent classification (0 on the fast path)
    "first_token",  # first model token
    "first_audio",  # first TTS chunk handed to the device
)

_lock = threading.Lock()
_records = deque(maxlen=MAX_RECORDS)
_spans = {}


def begin(request_id, origin="ui", label=""):
    """Start (or restart) the in-flight span set for *request_id*."""
    span = _Turn(request_id, origin, label)
    with _lock:
        _spans[request_id] = span
    return span


def mark(request_id, name, at=None):
    """Mark a boundary on the in-flight turn. Safe if no turn is in flight."""
    with _lock:
        span = _spans.get(request_id)
    if span is not None:
        span.mark(name, at)


def mark_ms(request_id, name, value_ms):
    """Record an already-measured span (its own cost) for *request_id*."""
    with _lock:
        span = _spans.get(request_id)
    if span is not None:
        span.mark_ms(name, value_ms)


def mark_duration(request_id, name, started_at, at=None):
    """Record a span whose duration is *now - started_at*, in milliseconds.

    Used where the cost is one call rather than a phase boundary (the intent
    classifier, for instance): the caller times just that call and hands the
    start over, so the recorded value is the call's own cost and not the
    elapsed turn time.
    """
    now = time.monotonic() if at is None else at
    with _lock:
        span = _spans.get(request_id)
    if span is not None:
        span.mark_ms(name, round((now - started_at) * 1000.0, 1))


def finish(request_id, at=None):
    """Publish the in-flight turn's record. Returns it, or None."""
    with _lock:
        span = _spans.pop(request_id, None)
    if span is None:
        return None
    record = span.finish(at)
    with _lock:
        _records.append(record)
    return record


def recent(limit=50):
    """The most recent records, newest last."""
    with _lock:
        items = list(_records)
    return items[-max(1, int(limit)):]


def summary(limit=50):
    """Per-span median / p90 / max over the recent window, in milliseconds.

    Medians over a rolling window are the number to watch: they are stable and
    they are not moved by a single slow provider call.
    """
    items = recent(limit)
    out = {"samples": len(items), "spans": {}}
    for name in SPAN_ORDER:
        values = sorted(r["ms"].get(name) for r in items
                        if r["ms"].get(name) is not None)
        if not values:
            continue
        out["spans"][name] = {
            "n": len(values),
            "median_ms": _pct(values, 50),
            "p90_ms": _pct(values, 90),
            "max_ms": values[-1],
        }
    totals = sorted(r["total_ms"] for r in items if r.get("total_ms")
                    is not None)
    if totals:
        out["total"] = {
            "n": len(totals),
            "median_ms": _pct(totals, 50),
            "p90_ms": _pct(totals, 90),
            "max_ms": totals[-1],
        }
    return out


def _pct(sorted_values, pct):
    if not sorted_values:
        return None
    index = int(round((pct / 100.0) * (len(sorted_values) - 1)))
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def _mark_unfinished():
    """Drop in-flight spans older than a minute so a crashed turn cannot leak."""
    cutoff = time.monotonic() - 60.0
    with _lock:
        for rid in [r for r, s in _spans.items() if s.started < cutoff]:
            _spans.pop(rid, None)


class _Turn:
    """One in-flight turn's span boundaries."""

    __slots__ = ("request_id", "origin", "label", "started", "_marks")

    def __init__(self, request_id, origin="ui", label=""):
        self.request_id = request_id
        self.origin = origin
        self.label = label
        self.started = time.monotonic()
        self._marks = {}

    def mark(self, name, at=None):
        now = time.monotonic() if at is None else at
        # First mark wins: a repeated boundary must not rewrite the first.
        self._marks.setdefault(name, round((now - self.started) * 1000.0, 1))

    def mark_ms(self, name, value_ms):
        """Record an already-measured span (its own cost, not elapsed turn)."""
        self._marks.setdefault(name, value_ms)

    def finish(self, at=None):
        now = time.monotonic() if at is None else at
        self.mark("end", now)
        return {
            "request_id": self.request_id,
            "origin": self.origin,
            "label": self.label,
            "started_at": round(self.started, 3),
            "total_ms": round((now - self.started) * 1000.0, 1),
            "ms": dict(self._marks),
        }
