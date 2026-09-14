"""Local-agreement transcript stabilization for active conversation (F34).

The audit's local-Whisper change requires overlapping audio windows with
partial/final transcript IDs, local-agreement stabilization, and the rule
that *actions run only from committed transcripts, never from unstable
partials*. This module implements that contract so the conversation path
can consume overlapping transcription windows when they arrive and still
act deterministically.

A *window* is one transcription result with a stable id, plain text, its
audio range (``start_ms``/``end_ms``), its conversation *turn*, an optional
language tag and a final/partial flag. Windows overlap only when their
timestamped ranges demonstrably intersect by at least ``overlap_ratio``
(missing timestamps are NOT an overlap — see ``TranscriptWindow.overlaps``).
Two overlapping, *independent* (different audio) windows *agree* when their
normalized texts match. A transcript becomes *committed* when either a final
window arrives, or ``agreement_windows`` **consecutive** overlapping partial
windows agree AND the agreed text is long enough to risk an action.
Duplicates, late/out-of-order windows and windows from another turn are
ignored, and one final window commits exactly once (idempotent). Partial-only
text is held in ``unstable`` and is never returned by ``committed()``.
"""

import threading
import time

_AGREEMENT_WINDOWS_DEFAULT = 2


def _normalize(text):
    return " ".join((text or "").strip().lower().split())


class TranscriptWindow:
    """One transcription result over one identified audio window.

    *wid* is the window's stable identity, *start_ms*/*end_ms* its audio
    range, *turn* the conversation turn it belongs to (F34: windows from
    different turns must never be combined). Unknown timestamps are kept as
    ``None`` — they are NOT silently treated as "overlapping everything".
    """

    __slots__ = ("wid", "text", "final", "language", "start_ms", "end_ms",
                 "turn", "at")

    def __init__(self, wid, text, final=False, language=None,
                 start_ms=None, end_ms=None, turn=None):
        self.wid = str(wid or "")
        self.text = str(text or "")
        self.final = bool(final)
        self.language = language
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.turn = turn
        self.at = time.monotonic()

    def overlaps(self, other, ratio=0.5):
        """True only for a REAL, timestamped overlap of the two audio ranges.

        [F34] Missing timestamps return False: a window whose audio range is
        unknown can never be used as independent corroboration, so an
        unstable partial can never be promoted to a committed transcript by
        two unverifiable windows.
        """
        if self.start_ms is None or self.end_ms is None \
                or other.start_ms is None or other.end_ms is None:
            return False
        inter = min(self.end_ms, other.end_ms) - max(self.start_ms,
                                                     other.start_ms)
        if inter <= 0:
            return False
        shorter = min(self.end_ms - self.start_ms,
                      other.end_ms - other.start_ms)
        if shorter <= 0:
            return False
        return (inter / float(shorter)) >= ratio


def _independent(a, b):
    """True when two windows cover DIFFERENT audio (F34).

    Two different ids over the identical audio range are the same evidence
    counted twice, not two independent transcriptions of the same speech.
    """
    if a.wid and b.wid and a.wid == b.wid:
        return False
    if a.start_ms is None or b.start_ms is None:
        return False
    return (a.start_ms, a.end_ms) != (b.start_ms, b.end_ms)


class StableTranscript:
    __slots__ = ("text", "language", "windows", "committed_at")

    def __init__(self, text, language, windows):
        self.text = text
        self.language = language
        self.windows = tuple(windows)
        self.committed_at = time.monotonic()


class TranscriptStabilizer:
    """Turns partial/final overlapping windows into committed transcripts.

    ``push(window)`` returns a :class:`StableTranscript` the moment the
    input is committable (a final, or enough *consecutive, independent,
    overlapping* agreeing partials in the SAME turn), else None.

    [F34] The rules that keep unstable partials from authorizing actions:

    * **turn boundary** — a window from another turn is ignored outright; a
      ``begin_turn()`` (or ``reset()``) drops the accumulated windows and the
      previous turn's commitment.
    * **duplicates** — the same window id is counted once; re-pushing it is
      ignored (idempotent), so a retry can never manufacture agreement.
    * **reordering** — a window that ends before an already-accepted window
      is stale (late/out-of-order) and is ignored.
    * **consecutive agreement** — agreement only counts as a CONTIGUOUS run
      of windows ending at the newest one; a disagreement in between breaks
      the run, so "agree, contradict, agree" never commits.
    * **independence** — two agreeing windows must cover different,
      timestamped audio and must overlap by ``overlap_ratio``.
    * **idempotent commits** — one final window commits exactly once.

    ``committed()`` returns the last committed text (actions may only use
    this); ``unstable()`` exposes what is still being stabilized.
    """

    def __init__(self, agreement_windows=_AGREEMENT_WINDOWS_DEFAULT,
                 overlap_ratio=0.5, min_stable_chars=3, max_windows=16):
        self._agreement = max(1, int(agreement_windows))
        self._overlap_ratio = overlap_ratio
        self._min_stable = max(1, int(min_stable_chars))
        self._max_windows = max(4, int(max_windows))
        self._lock = threading.Lock()
        self._windows = []
        self._committed = None  # StableTranscript
        self._committed_wids = set()
        self._commit_count = 0
        self._unstable = None   # {wid, text, language, windows}
        self._seen_wids = set()
        self._turn = 0
        self._counters = {"duplicates": 0, "stale": 0, "cross_turn": 0,
                          "ignored": 0}

    # ── turn lifecycle ────────────────────────────────────────────────────
    def begin_turn(self, turn_id=None):
        """Start a new conversation turn; nothing from the old one carries in.

        Returns the active turn id. Any window already pushed for the old
        turn can no longer agree with anything, and the previous turn's
        committed text stops being the commitment.
        """
        with self._lock:
            self._turn = (int(turn_id) if isinstance(turn_id, int)
                          else self._turn + 1)
            self._reset_locked()
            return self._turn

    @property
    def turn(self):
        with self._lock:
            return self._turn

    def commit_count(self):
        """How many distinct final windows have committed (idempotency probe)."""
        with self._lock:
            return self._commit_count

    def counters(self):
        """Evidence counters for the windows rejected by each F34 rule."""
        with self._lock:
            return dict(self._counters)

    # ── window intake ─────────────────────────────────────────────────────
    def push(self, window):
        if window is None or not str(window.wid):
            return None
        with self._lock:
            if window.turn is None:
                window.turn = self._turn
            elif window.turn != self._turn:
                # A different turn's window can never corroborate this one.
                self._counters["cross_turn"] += 1
                return None
            if window.wid in self._seen_wids:
                # A duplicate is the same evidence, not a second opinion.
                self._counters["duplicates"] += 1
                return None
            if self._is_stale_locked(window):
                # Arrived late: an older audio range than what we already have.
                self._counters["stale"] += 1
                return None
            self._seen_wids.add(window.wid)
            self._windows.append(window)
            if len(self._windows) > self._max_windows:
                self._windows = self._windows[-self._max_windows:]
            if window.final:
                return self._commit_locked(window, [window])
            agreed = self._consecutive_agreement_locked(window)
            if agreed is None:
                self._unstable = {
                    "wid": window.wid,
                    "text": window.text,
                    "language": window.language,
                    "windows": list(self._windows[-8:]),
                }
                return None
            if len(agreed["windows"]) >= self._agreement \
                    and len(_normalize(agreed["text"])) >= self._min_stable:
                stable = StableTranscript(
                    agreed["text"], agreed["language"], agreed["windows"])
                self._committed = stable
                self._committed_wids.add(window.wid)
                self._commit_count += 1
                self._unstable = None
                return stable
            self._unstable = {
                "wid": window.wid,
                "text": window.text,
                "language": window.language,
                "windows": agreed["windows"],
            }
            return None

    def _is_stale_locked(self, window):
        """True for a late/out-of-order window (ends before the newest one)."""
        if window.end_ms is None:
            return False
        newest = max(
            (w.end_ms for w in self._windows if w.end_ms is not None),
            default=None,
        )
        return newest is not None and window.end_ms < newest

    def _consecutive_agreement_locked(self, window):
        """The contiguous, independent, overlapping run ending at *window*.

        Stops at the first window that does not overlap, does not agree, or
        is not independent — "agree, contradict, agree" therefore yields a
        run of one and never commits.
        """
        text = _normalize(window.text)
        if not text:
            return None
        chain = [window]
        priors = [w for w in self._windows[:-1] if w is not window]
        for prior in reversed(priors):
            current = chain[-1]
            if not current.overlaps(prior, self._overlap_ratio):
                break
            if _normalize(prior.text) != text:
                break
            if not _independent(current, prior):
                break
            chain.append(prior)
            if len(chain) >= self._agreement:
                break
        return {"text": window.text, "language": window.language,
                "windows": list(reversed(chain))}

    def _commit_locked(self, window, windows):
        if window.wid in self._committed_wids:
            # Idempotent: a retried final window commits exactly once.
            return None
        self._committed = StableTranscript(
            window.text, window.language, windows)
        self._committed_wids.add(window.wid)
        self._commit_count += 1
        self._unstable = None
        return self._committed

    def committed(self):
        with self._lock:
            return self._committed.text if self._committed is not None else ""

    def unstable(self):
        with self._lock:
            if self._unstable is None:
                return ""
            return self._unstable.get("text", "")

    def is_committed(self, text):
        """True only when *text* matches the COMMITTED transcript. Partial
        windows never satisfy this - the "actions only from committed
        transcripts" rule lives in this single testable place."""
        if not text:
            return False
        return _normalize(text) == _normalize(self.committed())

    def _reset_locked(self):
        self._windows = []
        self._committed = None
        self._committed_wids = set()
        self._commit_count = 0
        self._unstable = None
        self._seen_wids = set()

    def reset(self):
        with self._lock:
            self._reset_locked()


# Reusable turn-scoped stabilizer for the conversation path.
stabilizer = TranscriptStabilizer()