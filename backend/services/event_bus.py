"""S19 — one push channel for state changes, instead of polling.

The voice worker used to ASK the backend "is a task running?" and "is voice
enabled?" on a 1s-cached HTTP poll (``backend_task_running`` /
``voice_input_enabled``), and the UI polled ``/ui-state`` the same way. Every
question is a localhost round-trip, and a change is noticed up to a second
late — a reply could finish while the listener still believed Jarvis was
speaking.

Instead the backend PUSHES each state change the moment it happens over one
persistent SSE channel (``GET /events``). Subscribers (the voice worker's
listener, the UI) update on the event, in milliseconds. Polling stays as a
fallback so nothing breaks if the channel drops — see
``voice_mode._task_state_push``.

This module is deliberately dependency-free so ``brain`` and ``listener_state``
can emit into it without a cycle. It is thread-safe: emitters run on request
handler / task threads, while subscribers drain from the event loop.
"""

import queue
import threading

#: A subscriber that falls this far behind loses its OLDEST events rather than
#: growing without bound; the newest state always wins, so a burst can never
#: wedge a slow reader.
_QUEUE_MAX = 64

_lock = threading.Lock()
#: queue.Queue -> optional zero-arg wake callable (async subscribers park on an
#: asyncio event and pass a thread-safe setter here).
_subscribers = {}
#: Merged latest state, replayed to a new subscriber so a reconnecting reader
#: starts from the current truth instead of an empty stream.
_snapshot = {}


def subscribe(wakeup=None):
    """Register a subscriber. Returns its ``queue.Queue`` of event dicts.

    ``wakeup``, when given, is called from the emitting thread after an event
    is queued — use it to break an async reader out of a wait. An SSE
    connection passes a ``loop.call_soon_threadsafe`` wrapper.
    """
    q = queue.Queue(maxsize=_QUEUE_MAX)
    with _lock:
        _subscribers[q] = wakeup
    return q


def unsubscribe(q):
    with _lock:
        _subscribers.pop(q, None)


def publish(event_type, data):
    """Broadcast ``{"type": event_type, **data}`` and remember ``data``.

    ``data`` keys are merged into the snapshot, so ``{"task_running": True}``
    and later ``{"voice_input_enabled": False}`` accumulate into one state a
    late subscriber still receives in full.
    """
    event = {"type": event_type}
    event.update(data)
    with _lock:
        _snapshot.update(data)
        targets = list(_subscribers.items())
    for q, wakeup in targets:
        _offer(q, event)
        if wakeup is not None:
            try:
                wakeup()
            except Exception:  # never let a broken wake kill the emitter
                pass


def snapshot():
    with _lock:
        return dict(_snapshot)


def subscriber_count():
    with _lock:
        return len(_subscribers)


def _offer(q, event):
    """Enqueue, dropping the oldest event if the reader is wedged."""
    try:
        q.put_nowait(event)
        return
    except queue.Full:
        pass
    try:
        q.get_nowait()  # make room
    except queue.Empty:
        pass
    try:
        q.put_nowait(event)
    except queue.Full:
        pass


def reset():
    """Testing seam: drop all subscribers and state."""
    with _lock:
        _subscribers.clear()
        _snapshot.clear()
