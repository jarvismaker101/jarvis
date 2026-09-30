"""[P1-02] Earcons must stay off the capture loop and out of the stop path.

STEP 0 FINDING (the audit flagged this as an assumption to verify):
``earcons.py`` was ALREADY non-blocking. ``_play_pattern`` spawns a daemon
thread and returns; ``winsound.Beep`` — the only blocking call — runs on that
thread. Measured caller-block: p50 0.27ms, max 1.5ms across 20 samples, i.e.
pure thread-spawn overhead. So no offload worker was invented; the work here is
the two things that were genuinely wrong:

  * barge-in played the "ready" cue in the middle of the user's sentence;
  * nothing prevented two cues from overlapping (N calls meant N threads).

The cues are also DISABLED by default (``JARVIS_EARCONS_ENABLED`` unset), which
is worth knowing before reading the numbers as a live hot-path cost.
"""

import threading
import time
import unittest
from unittest.mock import patch

from backend.services import earcons
from backend.services import listener
from backend.services import voice


class EarconTestCase(unittest.TestCase):
    """Fresh earcon state per test: the guard is module-global by design."""

    def setUp(self):
        earcons._busy = threading.Lock()
        for key in earcons.stats:
            earcons.stats[key] = 0
        self._enable()

    def _enable(self, reply_start=False):
        patches = [
            patch.object(earcons, "EARCONS_ENABLED", True),
            patch.object(earcons, "REPLY_START_EARCON_ENABLED", reply_start),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)


class NonBlockingTests(EarconTestCase):
    """The play call never blocks the caller, whatever the backend does."""

    def test_a_slow_backend_does_not_block_the_caller(self):
        """A 2s Beep must cost the caller thread-spawn time, not 2s."""
        release = threading.Event()

        def _slow_beep(frequency, duration_ms):
            release.wait(30.0)      # cooperative: this test owns the release

        # No thread may outlive this test. A sleeping backend thread that
        # survives into a later test would call THAT test's mock (the module
        # attribute is resolved at call time) and pollute its counters.
        self.addCleanup(release.set)
        try:
            with patch.object(earcons, "winsound") as fake_winsound:
                fake_winsound.Beep = _slow_beep
                started = time.perf_counter()
                earcons.play_capture_complete_earcon()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
        finally:
            release.set()

        self.assertLess(elapsed_ms, 50.0,
                        "the capture thread waited %.0fms on the earcon backend"
                        % elapsed_ms)

    def test_the_beep_runs_on_another_thread_than_the_caller(self):
        """Proves the device write is off the capture thread, not just fast."""
        caller = threading.current_thread()
        seen = {}

        def _record_thread(frequency, duration_ms):
            seen["thread"] = threading.current_thread()

        with patch.object(earcons, "winsound") as fake_winsound:
            fake_winsound.Beep = _record_thread
            earcons.play_capture_complete_earcon()
            for _ in range(100):
                if "thread" in seen:
                    break
                time.sleep(0.01)

        self.assertIn("thread", seen, "the cue never reached the backend")
        self.assertIsNot(seen["thread"], caller,
                          "the Beep ran on the calling thread")

    def test_an_unavailable_backend_is_a_no_op(self):
        with patch.object(earcons, "winsound", None):
            for fn in (earcons.play_ready_earcon,
                       earcons.play_capture_complete_earcon,
                       earcons.play_reply_start_earcon):
                self.assertFalse(fn())

    def test_a_backend_error_never_propagates(self):
        with patch.object(earcons, "winsound") as fake_winsound:
            fake_winsound.Beep = lambda *a: (_ for _ in ()).throw(RuntimeError("no device"))
            for fn in (earcons.play_ready_earcon,
                       earcons.play_capture_complete_earcon):
                self.assertTrue(fn())          # started on its thread
        time.sleep(0.1)                        # the error lands here, harmlessly

    def test_a_thread_creation_failure_does_not_mute_later_cues(self):
        with patch.object(earcons, "threading") as fake_threading:
            fake_threading.Thread.side_effect = RuntimeError("cannot start thread")
            self.assertFalse(earcons.play_capture_complete_earcon())
        # The slot must have been handed back.
        with patch.object(earcons, "winsound") as fake_winsound:
            fake_winsound.Beep = lambda *a: None
            self.assertTrue(earcons.play_capture_complete_earcon())



class BargeInTests(EarconTestCase):
    """Requirement 2: no "ready" cue when the user barged in."""

    def _quiet_audio_stops(self):
        """Keep these tests hermetic: no real engine teardown runs."""
        for name in ("stop_elevenlabs", "stop_fish_audio", "stop_google_tts"):
            item = patch.object(voice, name)
            item.start()
            self.addCleanup(item.stop)

    def test_stop_speaking_skips_the_ready_cue_when_asked(self):
        self._quiet_audio_stops()
        with patch.object(voice, "play_ready_earcon") as ready, \
                patch.object(voice, "is_speaking", True):
            voice.stop_speaking(signal_ready=False)
        ready.assert_not_called()

    def test_other_stop_paths_keep_the_ready_cue(self):
        """The UI stop button still acknowledges "your turn"."""
        self._quiet_audio_stops()
        with patch.object(voice, "play_ready_earcon") as ready, \
                patch.object(voice, "is_speaking", True):
            voice.stop_speaking()
        ready.assert_called_once_with()

    def test_no_cue_at_all_when_nothing_was_playing(self):
        self._quiet_audio_stops()
        with patch.object(voice, "play_ready_earcon") as ready, \
                patch.object(voice, "is_speaking", False):
            voice.stop_speaking()
        ready.assert_not_called()

    def test_barge_in_passes_signal_ready_false(self):
        """The capture path must actually ask for the silent stop."""
        with patch.object(listener.listener_state, "is_speaking", return_value=True), \
                patch("backend.services.voice.stop_speaking") as stop, \
                patch.object(listener, "_post_backend_speak_stop", return_value=True):
            listener.barge_in_on_speech_onset()

        stop.assert_called_once_with(signal_ready=False)

    def test_barge_in_plays_no_ready_earcon_end_to_end(self):
        self._quiet_audio_stops()
        with patch.object(earcons, "winsound") as fake_winsound, \
                patch.object(voice, "is_speaking", True), \
                patch.object(listener.listener_state, "is_speaking", return_value=True), \
                patch.object(listener, "_post_backend_speak_stop", return_value=True):
            fake_winsound.Beep = lambda *a: None
            listener.barge_in_on_speech_onset()

        self.assertEqual(earcons.stats["played"], 0,
                         "barge-in must not start any earcon")

    def test_barge_in_still_stops_the_local_audio(self):
        """Removing the cue must not weaken the actual interruption."""
        self._quiet_audio_stops()
        with patch.object(voice, "is_speaking", True), \
                patch.object(listener.listener_state, "is_speaking", return_value=True), \
                patch.object(listener, "_post_backend_speak_stop", return_value=True), \
                patch.object(voice, "stop_fish_audio") as stop_fish:
            listener.barge_in_on_speech_onset()

        stop_fish.assert_called()
        self.assertEqual(voice.is_speaking, False,
                         "the stop must still have cleared the speaking flag")


class ReplyStartCueTests(EarconTestCase):
    """Requirement 3: the reply-start cue is off unless explicitly restored."""

    def test_reply_start_is_off_by_default(self):
        with patch.object(earcons, "winsound") as fake_winsound:
            fake_winsound.Beep = lambda *a: None
            self.assertFalse(earcons.play_reply_start_earcon())
        self.assertEqual(earcons.stats["played"], 0)

    def test_reply_start_can_be_restored_by_config(self):
        with patch.object(earcons, "REPLY_START_EARCON_ENABLED", True), \
                patch.object(earcons, "winsound") as fake_winsound:
            fake_winsound.Beep = lambda *a: None
            self.assertTrue(earcons.play_reply_start_earcon())
        self.assertEqual(earcons.stats["played"], 1)

    def test_capture_complete_still_plays(self):
        with patch.object(earcons, "winsound") as fake_winsound:
            fake_winsound.Beep = lambda *a: None
            self.assertTrue(earcons.play_capture_complete_earcon())


class CoalescingTests(EarconTestCase):
    """Requirement 1/4: no backlog, and two cues can never overlap."""

    #: The ready cue's note. Counting only this note keeps the device-level
    #: assertion immune to a stray thread from another test that woke up mid-way
    #: through a different cue's pattern (the module attribute it calls is
    #: resolved at call time, so it lands on THIS test's mock).
    READY_NOTE_MS = 90

    def test_a_burst_of_identical_cues_does_not_build_a_backlog(self):
        release = threading.Event()
        entered = threading.Event()
        beats = {"count": 0}
        seen = []
        trace = []

        def _blocking_beep(frequency, duration_ms):
            if duration_ms == self.READY_NOTE_MS:
                beats["count"] += 1
            seen.append((threading.current_thread().name, duration_ms,
                         round(time.perf_counter() - t_start, 3)))
            entered.set()
            release.wait(30.0)          # effectively infinite: no early timeout

        t_start = time.perf_counter()
        try:
            with patch.object(earcons, "winsound") as fake_winsound:
                fake_winsound.Beep = _blocking_beep
                first = earcons.play_ready_earcon()
                trace.append(("first", first, beats["count"]))
                self.assertTrue(first, "the first cue should start")
                self.assertTrue(entered.wait(5.0),
                                "the first cue never reached the backend")
                for index in range(24):
                    ok = earcons.play_ready_earcon()
                    trace.append((index, ok, beats["count"]))
                    if ok or beats["count"] != 1:
                        break
                started = [entry[1] for entry in trace[1:]]
        finally:
            release.set()
            time.sleep(0.2)

        self.assertEqual(beats["count"], 1,
                         "distinct ready cues reached the device: %r | %r"
                         % (seen, trace))
        self.assertEqual(started, [False] * len(started),
                         "a burst must coalesce to ONE cue, not 25 threads: %r"
                         % (trace,))
        # The module counters are the pollution-free signal: they are reset per
        # test and only the calls made HERE touch them.
        self.assertEqual(earcons.stats["played"], 1)
        self.assertEqual(earcons.stats["coalesced"], len(trace) - 1)

    def test_capture_complete_cannot_overlap_reply_start(self):
        """The two cues the audit calls out are mutually exclusive."""
        release = threading.Event()
        concurrent = {"max": 0, "live": 0}
        guard = threading.Lock()

        def _tracked_beep(frequency, duration_ms):
            with guard:
                concurrent["live"] += 1
                concurrent["max"] = max(concurrent["max"], concurrent["live"])
            release.wait(30.0)
            with guard:
                concurrent["live"] -= 1

        # Cooperative release so no backend thread outlives this test.
        self.addCleanup(release.set)
        try:
            with patch.object(earcons, "REPLY_START_EARCON_ENABLED", True), \
                    patch.object(earcons, "winsound") as fake_winsound:
                fake_winsound.Beep = _tracked_beep
                first = earcons.play_capture_complete_earcon()
                second = earcons.play_reply_start_earcon()
                release.set()
                time.sleep(0.2)
        finally:
            release.set()

        self.assertTrue(first, "the capture-complete cue should have played")
        self.assertFalse(second,
                         "the reply-start cue must not overlap a sounding cue")
        self.assertEqual(concurrent["max"], 1, "two cues overlapped")

    def test_the_slot_is_released_so_later_cues_still_play(self):
        with patch.object(earcons, "winsound") as fake_winsound:
            fake_winsound.Beep = lambda *a: None
            self.assertTrue(earcons.play_capture_complete_earcon())
            for _ in range(100):
                if earcons.stats["played"] == 1:
                    break
                time.sleep(0.01)
            time.sleep(0.25)          # the short pattern finishes
            self.assertTrue(earcons.play_capture_complete_earcon(),
                            "the guard must not latch shut")

    def test_the_cue_never_goes_through_the_audio_actor(self):
        """F50 / Fable-5: the actor is the ONE playback owner for a reply."""
        import backend.services.audio_actor as audio_actor

        with patch.object(audio_actor, "actor_feed") as feed, \
                patch.object(audio_actor, "actor_begin") as begin, \
                patch.object(earcons, "winsound") as fake_winsound:
            fake_winsound.Beep = lambda *a: None
            earcons.play_ready_earcon()
            earcons.play_capture_complete_earcon()

        feed.assert_not_called()
        begin.assert_not_called()
        time.sleep(0.15)

