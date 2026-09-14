"""F24 — one absolute monotonic deadline / cancellation handle for a turn.

The audit found that each stage of a turn invented its own budget:

  * ``gemini_client`` and ``fireworks_client`` slept and re-POSTed without
    knowing how much of the caller's window was left, and the urllib3 retry
    strategy mounted underneath them ran *hidden* retries that could sleep
    and re-POST long past the deadline;
  * retry eligibility ("is this auth? a rate limit? a bad request?") was
    decided separately, in each client, with its own status lists;
  * ``executor.get_first_youtube_video`` had no timeout at all, so a stalled
    lookup could outlive any budget forever.

This module replaces all of that with ONE handle.  A :class:`Deadline`
carries an *absolute* monotonic expiry plus the cancellation event it was
derived from, and it is coerced from every shape a caller might have:

  * a ``Deadline`` — passed through unchanged;
  * a float — an absolute ``time.monotonic()`` timestamp;
  * a ``jobs.JobToken`` (duck-typed) — it already owns an absolute deadline
    and a cancellation event, so a job IS a budget and can be propagated
    as one;
  * ``None`` — no budget, which means "no ambient budget either" (see
    :func:`resolve`).

Because transports cannot always take an argument (the urllib3 retry
strategy is one), a handle can also be *bound* to the calling thread; the
adapter-level retry consults :func:`current` so even invisible retries obey
the same window.  Waits are always capped to the time that is actually
left, never granted fresh time, and a spent budget sends no request at all.

:class:`Failure` is the second half of the finding: one classifier that
both clients share, answering the only question a retry loop should ask —
"is this failure eligible to be replayed, and is there budget left for it?"
"""

import contextlib
import threading
import time

from urllib3.exceptions import MaxRetryError
from urllib3.util.retry import Retry


class DeadlineExpired(Exception):
    """Raised at a budget checkpoint once the absolute deadline has passed."""


class BudgetExhausted(DeadlineExpired):
    """A request or retry was attempted with no budget left to spend."""


# ── failure classification (one classifier for every transport) ────────────
AUTH = "auth"
RATE_LIMIT = "rate_limit"
SERVER = "server"
NOT_FOUND = "not_found"
VALIDATION = "validation"
TIMEOUT = "timeout"
CONNECTION = "connection"
CANCELLED = "cancelled"
BUDGET = "budget"
UNKNOWN = "unknown"

#: The only kinds that may ever be replayed — and only while budget remains.
RETRYABLE_KINDS = frozenset({RATE_LIMIT, SERVER, TIMEOUT, CONNECTION})

_AUTH_STATUSES = frozenset({401, 403})
_NOT_FOUND_STATUSES = frozenset({404, 410})
_RATE_LIMIT_STATUSES = frozenset({408, 425, 429})
_VALIDATION_STATUSES = frozenset({400, 405, 406, 409, 413, 415, 422})
_CANCELLED_STATUSES = frozenset({499})

_TIMEOUT_NAMES = frozenset({
    "Timeout", "TimeoutError", "ReadTimeout", "ConnectTimeout",
    "ReadTimeoutError", "ConnectTimeoutError", "timeout",
})
_CONNECTION_NAMES = frozenset({
    "ConnectionError", "NewConnectionError", "ProtocolError", "MaxRetryError",
    "RetryError", "ChunkedEncodingError", "SSLError", "OSError", "HTTPError",
})

#: A request slice shorter than this is not worth sending at all.
MIN_SLICE = 0.05


class Failure:
    """One classified failure — the single answer to "may this be replayed?"."""

    __slots__ = ("kind", "status", "detail")

    def __init__(self, kind, status=None, detail=""):
        self.kind = kind
        self.status = status
        self.detail = str(detail or "")

    @property
    def retryable(self):
        """True only for transient kinds (rate limit / server / connect)."""
        return self.kind in RETRYABLE_KINDS

    @property
    def terminal(self):
        """True when replaying this failure would be a pure waste of budget."""
        return not self.retryable

    def to_dict(self):
        return {"kind": self.kind, "status": self.status, "detail": self.detail[:300]}

    def __eq__(self, other):
        if not isinstance(other, Failure):
            return NotImplemented
        return (self.kind, self.status) == (other.kind, other.status)

    def __hash__(self):
        return hash((self.kind, self.status))

    def __repr__(self):
        return "Failure(kind=%r, status=%r)" % (self.kind, self.status)


def classify_status(status, detail=""):
    """Classify an HTTP status into a :class:`Failure`.

    Auth (401/403), not-found (404/410) and validation (400/405/…/422)
    failures are terminal: replaying them cannot succeed.  Only transient
    statuses (429/408/5xx) are marked retryable.
    """
    try:
        code = int(status)
    except (TypeError, ValueError):
        return Failure(UNKNOWN, None, detail or str(status or ""))
    if code in _AUTH_STATUSES:
        kind = AUTH
    elif code in _CANCELLED_STATUSES:
        kind = CANCELLED
    elif code in _NOT_FOUND_STATUSES:
        kind = NOT_FOUND
    elif code in _RATE_LIMIT_STATUSES:
        kind = RATE_LIMIT
    elif code in _VALIDATION_STATUSES:
        kind = VALIDATION
    elif 500 <= code < 600:
        kind = SERVER
    elif 400 <= code < 500:
        kind = VALIDATION
    else:
        kind = UNKNOWN
    return Failure(kind, code, detail)


def classify_exception(exc):
    """Classify a raised exception into a :class:`Failure`.

    Transport hiccups (connection resets, timeouts) are retryable; anything
    else — including a refusal caused by a spent budget — is terminal.
    """
    if exc is None:
        return Failure(UNKNOWN, None, "no error")
    if isinstance(exc, (BudgetExhausted, DeadlineExpired)):
        return Failure(BUDGET, None, str(exc))
    if isinstance(exc, TimeoutError):
        return Failure(TIMEOUT, None, str(exc))
    name = type(exc).__name__
    if name in _TIMEOUT_NAMES:
        return Failure(TIMEOUT, None, str(exc))
    if name in _CONNECTION_NAMES:
        return Failure(CONNECTION, None, str(exc))
    if isinstance(exc, OSError):
        return Failure(CONNECTION, None, str(exc))
    return Failure(UNKNOWN, None, str(exc))


def failure_of(result):
    """Classify a client result dict; ``None`` when it carries no failure.

    Client error results keep their historical ``{"error": <status>, ...}``
    shape and now also carry the classification under ``"failure"``, so a
    caller (brain's fallback chain, for one) can tell a terminal auth failure
    — which must never be replayed against another provider either — from a
    transient one.
    """
    if not isinstance(result, dict):
        return None
    known = result.get("failure")
    if isinstance(known, Failure):
        return known
    status = result.get("error")
    if isinstance(status, int):
        return classify_status(status, result.get("detail"))
    if status:
        return Failure(UNKNOWN, None, result.get("detail") or str(status))
    return None


def budget_allows(handle=None, attempt=1, max_attempts=2, minimum_remaining=0.0):
    """Attempts and time left?  Kind-independent replay gate.

    Used for a *deliberate* replay (a corrected payload, say), which is not
    driven by the failure classification but must still fit in the window.
    """
    if attempt >= max_attempts:
        return False
    handle = resolve(handle)
    if handle is None:
        return True
    remaining = handle.remaining()
    if remaining is None:
        return True
    return remaining > minimum_remaining


def retry_eligible(failure, handle=None, attempt=1, max_attempts=2,
                   minimum_remaining=0.0):
    """May *failure* be replayed as attempt *attempt* + 1?

    Three gates, in order: the failure must be transient, attempts must be
    left, and the budget must still have time in it.  The last gate is what
    used to be missing — every retry granted itself a fresh timeout.
    """
    if failure is None or not failure.retryable:
        return False
    return budget_allows(handle, attempt, max_attempts, minimum_remaining)


class Deadline:
    """One absolute monotonic deadline, plus the cancellation it derives from."""

    __slots__ = ("_expires_at", "_cancel", "_reason")

    def __init__(self, expires_at=None, cancel=None, reason="deadline exceeded"):
        self._expires_at = None if expires_at is None else float(expires_at)
        self._cancel = cancel
        self._reason = reason

    # ── construction ──
    @classmethod
    def after(cls, seconds):
        """A budget that expires *seconds* from now."""
        return cls(time.monotonic() + max(0.0, float(seconds)))

    @classmethod
    def at(cls, expires_at):
        """A budget with an already-known absolute expiry (monotonic clock)."""
        return cls(expires_at)

    @classmethod
    def unbounded(cls):
        """No deadline — carried so a cancellation can still be bound to it."""
        return cls(None)

    @classmethod
    def from_job(cls, job):
        """The handle a ``jobs.JobToken`` already implies (duck-typed)."""
        if job is None:
            return None
        cancel = getattr(job, "cancel_event", None)
        if cancel is None:
            cancel = getattr(job, "_cancel", None)
        reason = getattr(job, "cancel_reason", "") or "deadline exceeded"
        return cls(getattr(job, "deadline", None), cancel=cancel, reason=reason)

    @classmethod
    def coerce(cls, value):
        """Turn whatever a caller passes into a handle (or ``None``)."""
        if value is None or isinstance(value, Deadline) or isinstance(value, bool):
            return value if isinstance(value, Deadline) else None
        if isinstance(value, (int, float)):
            return cls(float(value))
        if hasattr(value, "deadline") or hasattr(value, "cancelled"):
            return cls.from_job(value)
        return None

    # ── state ──
    @property
    def expires_at(self):
        return self._expires_at

    @property
    def reason(self):
        return self._reason

    def remaining(self):
        """Seconds left, or ``None`` when the budget is unbounded."""
        if self._expires_at is None:
            return None
        return self._expires_at - time.monotonic()

    def expired(self):
        return self._expires_at is not None and time.monotonic() >= self._expires_at

    def cancelled(self):
        cancel = self._cancel
        if cancel is None:
            return False
        try:
            is_set = getattr(cancel, "is_set", None)
            if callable(is_set):
                return bool(is_set())
            if callable(cancel):
                return bool(cancel())
            return bool(cancel)
        except Exception:
            return False

    def stopped(self):
        """Cancelled or out of time — the only checkpoint callers need."""
        return self.cancelled() or self.expired()

    # ── checkpoints ──
    def check(self):
        """Raise :class:`DeadlineExpired` when the budget is spent."""
        if self.cancelled():
            raise DeadlineExpired("cancelled")
        if self.expired():
            raise DeadlineExpired(self._reason)
        return True

    def sleep(self, seconds):
        """Wait at most *seconds*, never past the deadline.

        Returns False when the budget was already gone (nothing was woken
        into) or expired during the wait, so a caller can stop instead of
        starting fresh work on expired time.
        """
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            seconds = 0.0
        if self.stopped():
            return False
        remaining = self.remaining()
        if remaining is None:
            if seconds > 0:
                time.sleep(seconds)
            return True
        if remaining <= 0:
            return False
        time.sleep(min(seconds, remaining) if seconds > 0 else 0)
        return not self.stopped()

    def timeout(self, default=(8, 45), minimum=MIN_SLICE):
        """``(connect, read)`` for ``requests``, sliced to the remaining budget.

        Returns ``None`` when the budget is spent — the caller must then send
        no request at all rather than a request it cannot afford.
        """
        if self.stopped():
            return None
        remaining = self.remaining()
        if remaining is None:
            return default
        if remaining < minimum:
            return None
        try:
            connect, read = default
        except (TypeError, ValueError):
            connect = read = float(default)
        return (max(minimum, min(float(connect), remaining)),
                max(minimum, min(float(read), remaining)))

    def seconds(self, default, minimum=MIN_SLICE):
        """A single scalar timeout sliced to the remaining budget, or ``None``."""
        if self.stopped():
            return None
        remaining = self.remaining()
        if remaining is None:
            return float(default)
        if remaining < minimum:
            return None
        return max(minimum, min(float(default), remaining))

    def child(self, seconds=None):
        """A sub-budget that can never outlive this one.

        The cancellation event is shared, so cancelling the parent stops the
        child too; a child that outlived its parent was exactly the "waits
        grant fresh time after expiry" defect.
        """
        expires = self._expires_at
        if seconds is not None:
            candidate = time.monotonic() + max(0.0, float(seconds))
            expires = candidate if expires is None else min(candidate, expires)
        return Deadline(expires, cancel=self._cancel, reason=self._reason)

    def bound(self):
        """Bind this handle to the calling thread (see :func:`bound`)."""
        return bound(self)

    def __repr__(self):
        return "Deadline(expires_at=%r, remaining=%r)" % (
            self._expires_at, self.remaining())


# ── thread-local propagation ───────────────────────────────────────────────
_STATE = threading.local()


def current():
    """The handle bound to this thread, or ``None``."""
    return getattr(_STATE, "deadline", None)


def bind(handle):
    """Park *handle* as this thread's budget; returns the previous value."""
    previous = getattr(_STATE, "deadline", None)
    _STATE.deadline = handle
    return previous


def unbind(previous):
    _STATE.deadline = previous


@contextlib.contextmanager
def bound(handle):
    """Run a block under *handle* so argument-less layers can honour it."""
    previous = bind(handle)
    try:
        yield handle
    finally:
        unbind(previous)


def resolve(handle):
    """The caller's handle, else the one bound to this thread.

    An explicit argument always wins; a client that was given none inherits
    the turn's budget instead of inventing an unbounded one.
    """
    coerced = Deadline.coerce(handle)
    if coerced is not None:
        return coerced
    return current()


def timeout_for(handle, default=(8, 45), minimum=MIN_SLICE):
    """``(connect, read)`` sliced to the resolved budget (``None`` when spent)."""
    resolved = resolve(handle)
    if resolved is None:
        return default
    return resolved.timeout(default, minimum=minimum)


def seconds_for(handle, default, minimum=MIN_SLICE):
    """A scalar timeout sliced to the resolved budget (``None`` when spent)."""
    resolved = resolve(handle)
    if resolved is None:
        return float(default)
    return resolved.seconds(default, minimum=minimum)


def wait(handle, seconds):
    """Sleep at most *seconds* of the resolved budget."""
    resolved = resolve(handle)
    if resolved is None:
        time.sleep(seconds)
        return True
    return resolved.sleep(seconds)


class BudgetedRetry(Retry):
    """urllib3 retry strategy that can never outlive the thread's budget.

    Adapter-level retries were the *hidden* extra attempts the audit found:
    they slept through their backoff and re-POSTed with no knowledge of the
    caller's window, so a 3-second budget could quietly become a minute.

    Every retry now consults the handle bound to the calling thread:

      * when the budget is spent the retry is refused outright — the next
        request is never sent;
      * the backoff (and any ``Retry-After``) is capped to the time that is
        actually left, so a wait can no longer hand out fresh time.
    """

    def increment(self, method=None, url=None, response=None, error=None,
                  _pool=None, _stacktrace=None):
        handle = current()
        if handle is not None and handle.stopped():
            raise MaxRetryError(_pool, url, BudgetExhausted(handle.reason))
        return super().increment(
            method=method, url=url, response=response, error=error,
            _pool=_pool, _stacktrace=_stacktrace,
        )

    def sleep(self, response=None):
        handle = current()
        if handle is None:
            return super().sleep(response)
        remaining = handle.remaining()
        if remaining is None:
            return super().sleep(response)
        if remaining <= 0:
            return
        wait_for = self.get_backoff_time()
        if self.respect_retry_after_header and response is not None:
            try:
                wait_for = max(wait_for, self.get_retry_after(response) or 0.0)
            except Exception:
                pass
        if wait_for <= 0:
            return
        time.sleep(min(wait_for, remaining))
