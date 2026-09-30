"""[PERF] Per-turn latency telemetry — absolute marks, one waterfall per turn.

P1-19 (responsiveness audit, phase 0): every other item in that pack claims a
latency win, and none of them could be measured, because the ring this module
publishes mixed two different things in one ``ms`` map — offsets from the turn
start and per-call durations. A duration could contradict the offset it was
stored next to, and no single stage was identifiable.

What a record IS now
--------------------
* A record is a list of ``(name, perf_counter_ns, meta)`` tuples: ABSOLUTE
  ``time.perf_counter_ns()`` marks only, at most one per boundary name (the
  first one wins — a repeated boundary must not rewrite the first).
* A duration is NEVER stored. Durations are derived at read time by
  differencing consecutive marks, so a duration cannot contradict the offsets
  it was derived from, and a mark that arrives late (a voice turn's playback
  boundary, say) merges in without rewriting history.
* The waterfall is therefore addable: the step deltas of one turn sum to its
  ``total_ms``, and only one field per mark can be authoritative — its clock.

What produces the marks
-----------------------
Backend (this process), per turn: ``http_in`` (the route handler, before any
work is queued — the turn clock used to start inside the worker thread), then
``racer_start``, ``classify_done`` (with ``hop``: which classifier —
openrouter / gemini / groq / fastpath — answered), ``provider_headers`` (the
model provider's HTTP response arrived), ``first_token``, ``tts_first_byte`` /
``playback_started`` when this process owns playback, and ``end``.

Voice worker (a SEPARATE process — F50): ``speech_end``, ``capture_end``,
``stt_start``, ``stt_done`` (with the STT engine that produced the transcript),
``tts_first_byte``, ``playback_started``. Those marks ride the submission
itself (``client_marks`` on ``POST /ask/stream``) and afterwards ``POST
/latency/client``, under the SAME ``request_id``, and are merged onto this
process's clock with one offset per turn (:func:`merge_client_marks`). That is
what closes the holes in the waterfall: without it the two processes' records
never met.

Everything here is best-effort: a mark that cannot be recorded is dropped, and
nothing in this module raises into a caller. Pure stdlib by design — this sits
on the request path.
"""

import json
import threading
import time
from collections import deque

#: Bounded history of completed turns. One voice turn is ~5s, so this is a few
#: minutes of detail; the ring never grows past it.
MAX_RECORDS = 200

#: Hard bound on the marks ONE turn may carry while it is in flight. A runaway
#: or buggy producer can therefore not grow a record without limit.
MAX_MARKS_PER_TURN = 64

#: In-flight turns older than this are dropped so a crashed turn cannot leak.
#: Deliberately long: a browser-agent turn can legitimately run for minutes and
#: pruning it mid-flight would silently lose its record.
UNFINISHED_TTL_S = 1800.0

#: Canonical waterfall order. Reporting walks this first, then any other name a
#: producer marked, so a new mark is visible without editing this tuple.
WATERFALL_ORDER = (
    "speech_end",       # user stopped talking (voice worker)
    "capture_end",      # capture finalised, audio handed to STT
    "stt_start",        # transcript requested
    "stt_done",         # transcript committed (meta: engine)
    "http_in",          # backend received the request for this turn
    "racer_start",      # speculative chat stream started
    "classify_done",    # classifier verdict (meta: hop)
    "provider_headers", # model provider answered (meta: provider/model)
    "first_token",      # first streamed answer token reached the caller
    "tts_first_byte",   # first synthesized audio exists
    "playback_started", # first PCM written to the device
    "end",              # turn finished in the backend
)

#: Legacy span names (the pre-P1-19 ``summary()["spans"]`` keys) mapped onto the
#: mark they now describe. Reported as aliases with ``alias_of`` set so a
#: diagnostic reading the old names still finds numbers — computed from real
#: marks this time.
LEGACY_SPAN_ALIASES = {
    "endpoint": "speech_end",
    "stt": "stt_done",
    "classify": "classify_done",
    "first_audio": "playback_started",
}

#: Voice-process local turns carry the shipping cursor; a process that never
#: drains one (the watcher, say) must not accumulate marks forever. Kept at the
#: same bound as a backend turn.
MAX_LOCAL_MARKS = MAX_MARKS_PER_TURN

_lock = threading.Lock()
_records = deque(maxlen=MAX_RECORDS)
_spans = {}
_active_request = {"id": ""}
_local = {"turn": None, "lock": threading.Lock()}


def _safe_meta(meta):
    """A JSON-safe dict for one mark, whatever the caller passed."""
    try:
        if meta is None:
            return {}
        if not isinstance(meta, dict):
            meta = {"value": meta}
        return json.loads(json.dumps(meta, default=str))
    except Exception:
        return {}


def _now_ns():
    return int(time.perf_counter_ns())


class _Turn:
    """One turn's marks — in flight first, then published as the same object.

    The object stays mutable after publication on purpose: a voice turn's
    playback marks arrive AFTER the backend finished the turn (the two stamp
    their clocks in different processes at different moments), and merging them
    into the published record is the whole point of P1-19.
    """

    __slots__ = ("request_id", "origin", "label", "started_ns", "_lock",
                 "_marks", "_names", "_limit", "clock_offset_ns",
                 "finished_ns")

    def __init__(self, request_id, origin="ui", label="",
                 limit=MAX_MARKS_PER_TURN):
        self.request_id = request_id
        self.origin = origin
        self.label = label
        self.started_ns = _now_ns()
        self._lock = threading.Lock()
        #: ``[(name, perf_counter_ns, meta)]`` — ABSOLUTE marks only.
        self._marks = []
        self._names = set()
        self._limit = max(1, int(limit))
        #: Offset (ns) that maps THIS turn's marks onto another process's
        #: clock; computed once, reused for every later batch of that turn.
        self.clock_offset_ns = None
        self.finished_ns = None

    def add(self, name, at_ns=None, meta=None):
        """Append one absolute boundary mark. Returns True when accepted."""
        try:
            name = str(name or "").strip()
            if not name:
                return False
            at_ns = _now_ns() if at_ns is None else int(at_ns)
        except Exception:
            return False
        with self._lock:
            if name in self._names:
                return False      # first mark wins: never rewrite a boundary
            if len(self._marks) >= self._limit:
                return False      # bounded: a runaway producer cannot grow it
            self._names.add(name)
            self._marks.append((name, at_ns, _safe_meta(meta)))
        return True

    def add_client(self, marks, offset_ns, tag="client"):
        """Merge already-translated marks from another process."""
        added = 0
        for entry in list(marks or ()):
            try:
                name = str(entry[0])
                at_ns = int(entry[1]) + int(offset_ns)
            except Exception:
                continue      # a malformed mark is dropped, never raised
            meta = entry[2] if len(entry) > 2 else None
            if self.add(name, at_ns, _merge_clock(meta, tag)):
                added += 1
        return added

    def ordered(self):
        """The marks ordered by absolute time (first mark per name wins)."""
        with self._lock:
            marks = list(self._marks)
        marks.sort(key=lambda item: item[1])
        return marks

    def snapshot(self):
        """The JSON-safe record: absolute marks plus their DERIVED waterfall."""
        marks = self.ordered()
        started_ns = marks[0][1] if marks else self.started_ns
        steps = []
        offsets = {}
        previous = started_ns
        for index, (name, at_ns, meta) in enumerate(marks):
            offset_ms = round((at_ns - started_ns) / 1e6, 1)
            # A duration is DERIVED here and nowhere else: the first mark
            # defines the origin, so its own step is 0 by definition and the
            # deltas of the remaining steps sum to ``total_ms``.
            delta_ms = 0.0 if index == 0 else round(
                (at_ns - previous) / 1e6, 1)
            previous = at_ns
            offsets[name] = offset_ms
            steps.append({
                "name": name,
                "ns": at_ns,
                "offset_ms": offset_ms,
                "delta_ms": delta_ms,
                "meta": meta,
            })
        total_ms = round((marks[-1][1] - started_ns) / 1e6, 1) if marks else 0.0
        return {
            "request_id": self.request_id,
            "origin": self.origin,
            "label": self.label,
            # Backward-compatible keys (pre-P1-19 readers).
            "started_at": round(started_ns / 1e9, 3),
            "total_ms": total_ms,
            "ms": offsets,
            # New keys.
            "started_ns": started_ns,
            "finished_ns": self.finished_ns,
            "marks": [[name, at_ns, meta] for (name, at_ns, meta) in marks],
            "steps": steps,
            "clock_offset_ms": (None if self.clock_offset_ns is None else
                                round(self.clock_offset_ns / 1e6, 3)),
        }


def _merge_clock(meta, tag):
    """Copy one mark's meta with the clock it was stamped on recorded."""
    out = _safe_meta(meta)
    out.setdefault("clock", tag)
    return out


# ── In-flight turns (backend process) ───────────────────────────────────────


def _prune_unfinished():
    """Drop in-flight turns older than the TTL so a crashed turn cannot leak."""
    cutoff = _now_ns() - int(UNFINISHED_TTL_S * 1e9)
    with _lock:
        for request_id in [rid for rid, turn in _spans.items()
                           if turn.started_ns < cutoff]:
            _spans.pop(request_id, None)


def begin(request_id, origin="ui", label=""):
    """Ensure an in-flight turn for *request_id* and return it (or None).

    Deliberately IDEMPOTENT: an F23 retry or a reconnected stream re-enters the
    route handler for the same request, and it must continue the SAME turn
    instead of restarting the clock and losing the ``http_in`` mark that was
    already recorded.
    """
    if not request_id:
        return None
    try:
        _prune_unfinished()
        with _lock:
            turn = _spans.get(request_id)
            if turn is None:
                turn = _Turn(request_id, origin, label)
                _spans[request_id] = turn
            else:
                # A later attach can know more about the turn than the first.
                if origin and not turn.origin:
                    turn.origin = origin
                if label and not turn.label:
                    turn.label = label
        return turn
    except Exception:
        return None


def _span(request_id):
    """The in-flight turn for *request_id*, if any."""
    if not request_id:
        return None
    with _lock:
        return _spans.get(request_id)


def _published(request_id):
    """A published record with this id, newest first (late marks land here)."""
    if not request_id:
        return None
    with _lock:
        items = list(_records)
    for turn in reversed(items):
        if turn.request_id == request_id:
            return turn
    return None


def mark(request_id, name, meta=None, at_ns=None):
    """Record one ABSOLUTE boundary mark for *request_id*. Never raises."""
    try:
        turn = _span(request_id)
        if turn is None:
            return False
        return turn.add(name, at_ns=at_ns, meta=meta)
    except Exception:
        return False


def set_active_request(request_id):
    """Name the turn that untagged marks (provider/audio) belong to."""
    try:
        _active_request["id"] = str(request_id or "")
    except Exception:
        pass


def active_request():
    """The turn named by :func:`set_active_request` ("" when none)."""
    return _active_request.get("id", "")


def mark_active(name, meta=None):
    """Mark a boundary that belongs to whichever timeline is CURRENT.

    Used where the code observing the boundary has no transport identity: the
    provider stream clients (``provider_headers``) and the audio actor
    (``tts_first_byte`` / ``playback_started``). Resolution order:

    1. the voice turn this worker owns (the voice process — see
       :class:`LocalTurn`), then
    2. the newest in-flight request (the API process).

    Both processes run this same code, which is why the resolution lives here
    instead of at each call site.
    """
    turn = local_turn()
    if turn is not None:
        return turn.mark(name, meta)
    return mark(active_request(), name, meta=meta)


def mark_provider_headers(provider, model=None):
    """[PERF] P1-19 — the model provider's HTTP response arrived.

    The split that matters on a chat turn: everything before this mark is
    connect + queue + time-to-headers, everything after it is generation. One
    "time to first token" number cannot tell a slow provider apart from a slow
    prompt. First mark of the turn wins, so a provider fallback never rewrites
    the primary's boundary — its identity is in the meta either way.
    """
    return mark_active("provider_headers",
                       {"provider": str(provider or ""),
                        "model": str(model or "")})


def finish(request_id, at_ns=None):
    """Publish the turn's record (marks ``end``). Returns the snapshot or None.

    ``end`` is an ordinary mark, so the record is complete before any late
    client batch arrives; merging those afterwards (see
    :func:`merge_client_marks`) extends the same record instead of opening a
    second one for the same turn.
    """
    try:
        with _lock:
            turn = _spans.pop(request_id, None)
        if turn is None:
            return None
        if not turn.ordered():
            # Nothing was marked for this turn (no boundary, not even an
            # http_in): publish nothing rather than an `end`-only record that
            # would skew every percentile.
            return None
        end_ns = _now_ns() if at_ns is None else int(at_ns)
        turn.add("end", at_ns=end_ns)
        turn.finished_ns = end_ns
        with _lock:
            _records.append(turn)
        return turn.snapshot()
    except Exception:
        return None


def merge_client_marks(request_id, marks, client_now_ns=None):
    """Merge another process's marks into THIS turn (P1-19, requirement 5).

    The voice worker and the backend are separate OS processes, so their
    ``perf_counter_ns`` values are different clocks. Each batch carries one
    reference sample taken at send time (``client_now_ns``); the marks are
    translated with a single offset::

        offset = now_here - client_now_ns
        mark_here = mark_client + offset

    which is accurate to the one-way trip of a loopback request (well under a
    millisecond) and leaves the client's own mark spacing untouched. The offset
    is computed ONCE per turn and REUSED for every later batch, so the batches
    of one turn cannot drift apart, and a replayed batch stays idempotent
    because a repeated boundary name never rewrites the first mark.

    Returns the number of marks merged. Zero is not an error: an unknown turn
    (a backend restarted under a voice worker, say) is dropped, because
    telemetry is never worth failing a turn over.
    """
    if not request_id or not marks:
        return 0
    try:
        turn = _span(request_id) or _published(request_id)
        if turn is None:
            return 0
        if turn.clock_offset_ns is None:
            if client_now_ns:
                turn.clock_offset_ns = _now_ns() - int(client_now_ns)
            else:
                # No clock reference: assume both processes share a clock
                # (true for a boot-based perf_counter) and say so in the meta.
                turn.clock_offset_ns = 0
        tag = "client" if client_now_ns else "client-uncalibrated"
        return turn.add_client(marks, turn.clock_offset_ns, tag=tag)
    except Exception:
        return 0


# ── Local turns (voice worker process) ──────────────────────────────────────
#
# F50 makes the voice worker a pure I/O process that owns the microphone and
# the playback device; the backend owns everything else. A turn's first half
# (speech_end … stt_done) and its last half (tts_first_byte …) therefore exist
# only in THAT process, and its record cannot be written by the backend. The
# worker collects them here and ships them under the request_id it submits, so
# the backend's record and the worker's marks describe ONE turn.


class LocalTurn:
    """One voice turn's marks, before they are shipped to the backend.

    Same absolute-mark contract as :class:`_Turn` (first mark per name wins,
    bounded) so the two halves of a turn merge without translation.
    """

    __slots__ = ("label", "_turn", "_shipped")

    def __init__(self, label=""):
        self.label = label
        self._turn = _Turn("local", origin="voice", label=label,
                           limit=MAX_LOCAL_MARKS)
        #: How many marks have already been handed to the backend: the first
        #: batch rides the submission, later batches (playback) are shipped by
        #: the worker's publisher loop.
        self._shipped = 0

    def mark(self, name, meta=None):
        """Record one absolute boundary mark. Never raises."""
        try:
            return self._turn.add(name, meta=meta)
        except Exception:
            return False

    def marks(self):
        """Every mark of this turn, absolute and ordered (JSON-safe)."""
        return [[name, at_ns, meta]
                for (name, at_ns, meta) in self._turn.ordered()]

    def pending(self):
        """The marks NOT yet handed to the backend (the shipper's cursor)."""
        return self.marks()[self._shipped:]

    def ack(self, count):
        """Confirm *count* marks were handed over (cursor advances)."""
        try:
            self._shipped += max(0, int(count))
        except Exception:
            pass


def new_local_turn(label=""):
    """Start a fresh voice-process turn timeline (bounded by MAX_LOCAL_MARKS)."""
    try:
        return LocalTurn(label)
    except Exception:
        return LocalTurn()


def set_local_turn(turn):
    """Name the voice turn that untagged marks (the audio actor) belong to."""
    try:
        with _local["lock"]:
            _local["turn"] = turn
    except Exception:
        pass


def local_turn():
    """The voice turn currently being spoken, or None."""
    try:
        with _local["lock"]:
            return _local["turn"]
    except Exception:
        return None


def local_now_ns():
    """One sample of THIS process's clock, for the alignment offset."""
    try:
        return _now_ns()
    except Exception:
        return 0


# ── Reporting ───────────────────────────────────────────────────────────────


def recent(limit=50):
    """The most recent records, newest last — each one a full waterfall.

    A snapshot (not the live object) is returned: a published record keeps
    accepting late marks from the voice worker, and a reader must never see a
    half-merged list.
    """
    with _lock:
        items = list(_records)
    out = []
    for turn in items[-max(1, int(limit)):]:
        try:
            out.append(turn.snapshot())
        except Exception:
            continue
    return out


def _pct(sorted_values, pct):
    if not sorted_values:
        return None
    index = int(round((pct / 100.0) * (len(sorted_values) - 1)))
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def _metrics(values):
    """p50 / p90 / max for one series (median and p50 are the same number)."""
    return {
        "n": len(values),
        "median_ms": _pct(values, 50),
        "p50_ms": _pct(values, 50),
        "p90_ms": _pct(values, 90),
        "max_ms": values[-1] if values else None,
    }


def summary(limit=50):
    """Per-step p50 / p90 / max over the recent window, slowest stage first.

    For every step name the report carries BOTH numbers a step can have, and
    they can no longer contradict each other because both are derived from the
    same marks:

    * ``offset_*`` — how far into the turn the step happens (turn clock);
    * ``median_ms`` / ``p50_ms`` / ``p90_ms`` / ``max_ms`` — the step's OWN
      cost, i.e. the delta from the previous mark (the prefix of every turn
      sums to that turn's ``total_ms``).

    ``waterfall`` is the same data ordered by median offset, latest stage
    first, so the stage the turn is actually stuck in is the first row; the
    per-step cost is readable in the same row. The pre-P1-19 keys
    (``samples``, ``spans``, ``total`` and every ``*_ms`` inside them) are
    still present.
    """
    items = recent(limit)
    seen = set()
    for record in items:
        for step in record.get("steps", ()):
            name = step.get("name")
            if name:
                seen.add(name)
    ordered_names = ([name for name in WATERFALL_ORDER if name in seen]
                     + sorted(name for name in seen
                              if name not in WATERFALL_ORDER))

    spans = {}
    rows = []
    for name in ordered_names:
        offsets = sorted(record["ms"][name] for record in items
                         if name in record.get("ms", {}))
        deltas = sorted(step["delta_ms"] for record in items
                        for step in record.get("steps", ())
                        if step.get("name") == name)
        entry = _metrics(deltas)
        entry["offset_n"] = len(offsets)
        entry["offset_median_ms"] = _pct(offsets, 50)
        entry["offset_p50_ms"] = _pct(offsets, 50)
        entry["offset_p90_ms"] = _pct(offsets, 90)
        entry["offset_max_ms"] = offsets[-1] if offsets else None
        spans[name] = entry
        rows.append({
            "name": name,
            "n": entry["n"],
            "median_ms": entry["median_ms"],
            "p90_ms": entry["p90_ms"],
            "max_ms": entry["max_ms"],
            "median_offset_ms": entry["offset_median_ms"],
            "p90_offset_ms": entry["offset_p90_ms"],
            "max_offset_ms": entry["offset_max_ms"],
        })

    # Legacy span names keep reporting, as aliases of the mark they mean now.
    for alias, target in LEGACY_SPAN_ALIASES.items():
        if target in spans:
            row = dict(spans[target])
            row["alias_of"] = target
            spans[alias] = row

    rows.sort(key=lambda row: (row["median_offset_ms"] is None,
                               -(row["median_offset_ms"] or 0.0)))

    out = {
        "samples": len(items),
        "window": int(limit),
        "sorted_by": "median_offset_ms desc (slowest stage first)",
        "spans": spans,
        "waterfall": rows,
    }
    totals = sorted(record["total_ms"] for record in items
                    if record.get("total_ms") is not None)
    if totals:
        out["total"] = _metrics(totals)
    return out





