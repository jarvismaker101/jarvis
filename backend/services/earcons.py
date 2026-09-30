import os
import threading
import time

try:
    import winsound
except ImportError:
    winsound = None

EARCONS_ENABLED = os.getenv("JARVIS_EARCONS_ENABLED", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

#: [P1-02] The reply-start cue is OFF by default, even when earcons are on.
#: Instant speech is a better cue than a beep, and the beep only delays the
#: first word. Set ``JARVIS_REPLY_START_EARCON=1`` to restore it.
REPLY_START_EARCON_ENABLED = os.getenv(
    "JARVIS_REPLY_START_EARCON", ""
).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

#: [P1-02] One cue at a time, claimed non-blockingly.
#:
#: A cue is decorative, so a cue arriving while one is already sounding is
#: DROPPED rather than queued: that coalesces a backlog of identical cues (each
#: call used to start an independent thread, so N calls meant N overlapping
#: beeps) and it makes it structurally impossible for the capture-complete cue
#: and the reply-start cue to overlap.
#:
#: Claiming is a non-blocking ``acquire``, so a caller is never held up.
_busy = threading.Lock()

#: Counters for diagnostics and tests. Never carries audio or text.
stats = {"played": 0, "coalesced": 0, "disabled": 0}


def _play_pattern(pattern):
    """Play *pattern* on its own daemon thread. Never blocks, never raises.

    [P1-02] This was ALREADY non-blocking: ``winsound.Beep`` (which blocks for
    the note's duration) runs on the spawned thread, and the caller returns
    after ~0.3ms of thread-spawn overhead. The only thing added here is the
    one-cue-at-a-time guard above.

    Returns True when the cue was started, False when it was skipped.
    """
    if not EARCONS_ENABLED or winsound is None:
        stats["disabled"] += 1
        return False

    # Capture the lock this cue was claimed on. `_busy` is module-global and the
    # release happens on ANOTHER thread later, so looking it up again inside
    # `_run` would release whatever `_busy` points at by then - a different lock
    # instance, which raises "release unlocked lock" and frees the real slot for
    # a concurrent cue.
    lock = _busy
    if not lock.acquire(blocking=False):
        # A cue is already sounding. Coalesce: the backlog is not worth the
        # overlap, and a decorative cue must never queue up behind another.
        stats["coalesced"] += 1
        return False

    def _run():
        try:
            for frequency, duration_ms, gap_seconds in pattern:
                try:
                    winsound.Beep(frequency, duration_ms)
                except RuntimeError:
                    return

                if gap_seconds:
                    time.sleep(gap_seconds)
        finally:
            lock.release()

    try:
        threading.Thread(target=_run, daemon=True).start()
    except Exception:
        # Thread creation failed: hand the slot back so one failure cannot
        # mute every later cue.
        lock.release()
        return False

    stats["played"] += 1
    return True


def play_ready_earcon():
    """The "I stopped, your turn" cue. NOT for barge-in (see [P1-02])."""
    return _play_pattern([(880, 90, 0.0)])


def play_capture_complete_earcon():
    """End-of-turn acknowledgement, played during the dead period before STT."""
    return _play_pattern([(640, 60, 0.03), (520, 70, 0.0)])


def play_reply_start_earcon():
    """Cue before the first word of a reply. OFF unless explicitly enabled."""
    if not REPLY_START_EARCON_ENABLED:
        stats["disabled"] += 1
        return False
    return _play_pattern([(520, 45, 0.02), (760, 75, 0.0)])

