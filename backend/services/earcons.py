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


def _play_pattern(pattern):
    if not EARCONS_ENABLED or winsound is None:
        return

    def _run():
        for frequency, duration_ms, gap_seconds in pattern:
            try:
                winsound.Beep(frequency, duration_ms)
            except RuntimeError:
                return

            if gap_seconds:
                time.sleep(gap_seconds)

    threading.Thread(target=_run, daemon=True).start()


def play_ready_earcon():
    _play_pattern([(880, 90, 0.0)])


def play_capture_complete_earcon():
    _play_pattern([(640, 60, 0.03), (520, 70, 0.0)])


def play_reply_start_earcon():
    _play_pattern([(520, 45, 0.02), (760, 75, 0.0)])
