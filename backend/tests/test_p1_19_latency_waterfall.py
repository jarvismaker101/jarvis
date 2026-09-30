"""P1-19 (responsiveness audit, phase 0) — a real per-turn waterfall.

The acceptance this pins down:

* a record is a list of ABSOLUTE ``(name, perf_counter_ns, meta)`` marks — no
  stored duration anywhere, so an offset and a duration can never disagree;
* one turn renders as an ordered waterfall whose step deltas sum to its total
  time;
* the backend process and the voice worker, which are separate OS processes,
  produce ONE record for one ``request_id`` (the worker's marks are merged onto
  this process's clock with a single per-turn offset);
* ``/latency`` reports p50 / p90 / max per step over the window, ordered by
  median offset, while keeping its pre-P1-19 keys;
* the telemetry never raises into the request path, and the ring stays bounded;
* the ``/latency`` endpoint path never pulls the ``listener`` module into the
  backend process.

No microphone, TTS engine, provider or external process is opened here.
"""

import inspect
import json
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import speech_recognition as sr

from backend.api import routes
from backend.services import fireworks_client
from backend.services import gemini_client
from backend.services import latency
from backend.services import listener
from backend.services import openai_compat_client
from backend import voice_mode


class _LatencyCase(unittest.TestCase):
    """Isolated module state so one test cannot read another's turns."""

    def setUp(self):
        latency._records.clear()
        latency._spans.clear()
        latency.set_active_request("")
        latency.set_local_turn(None)

    def tearDown(self):
        latency._records.clear()
        latency._spans.clear()
        latency.set_active_request("")
        latency.set_local_turn(None)


class RecordShapeTests(_LatencyCase):
    """A record stores absolute marks; durations are derived, never stored."""

    def test_marks_are_absolute_timestamps_with_meta(self):
        latency.begin("req-shape", origin="ui", label="hello")
        latency.mark("req-shape", "http_in", meta={"route": "/ask/stream"})
        latency.mark("req-shape", "first_token")
        record = latency.finish("req-shape")

        self.assertEqual(record["request_id"], "req-shape")
        self.assertEqual(record["origin"], "ui")
        for mark in record["marks"]:
            self.assertIsInstance(mark, list)
            self.assertEqual(len(mark), 3)
            self.assertIsInstance(mark[0], str)
            self.assertIsInstance(mark[1], int)
            self.assertIsInstance(mark[2], dict)
        # An absolute perf_counter value, not an elapsed time.
        now = time.perf_counter_ns()
        self.assertLess(now - record["marks"][0][1], 5_000_000_000)
        self.assertEqual(record["marks"][0][2], {"route": "/ask/stream"})

    def test_no_duration_is_stored_alongside_the_marks(self):
        latency.begin("req-dur")
        latency.mark("req-dur", "http_in")
        latency.mark("req-dur", "first_token")
        record = latency.finish("req-dur")
        # The only numbers in a step are its clock and its derived offsets.
        for step in record["steps"]:
            self.assertEqual(set(step),
                             {"name", "ns", "offset_ms", "delta_ms", "meta"})
        # The legacy duration-taking API is gone (it is what mixed the two).
        self.assertFalse(hasattr(latency, "mark_ms"))
        self.assertFalse(hasattr(latency, "mark_duration"))

    def test_first_mark_per_name_wins(self):
        latency.begin("req-first")
        latency.mark("req-first", "http_in", at_ns=1_000_000_000)
        latency.mark("req-first", "http_in", at_ns=9_000_000_000)
        record = latency.finish("req-first", at_ns=10_000_000_000)
        names = [step["name"] for step in record["steps"]]
        self.assertEqual(names.count("http_in"), 1)
        self.assertEqual(record["ms"]["http_in"], 0.0)

    def test_a_turn_carries_a_bounded_number_of_marks(self):
        latency.begin("req-bound")
        for index in range(latency.MAX_MARKS_PER_TURN + 25):
            latency.mark("req-bound", "mark-%d" % index)
        record = latency.finish("req-bound")
        self.assertLessEqual(len(record["marks"]),
                             latency.MAX_MARKS_PER_TURN)
        self.assertLessEqual(len(record["marks"]), latency.MAX_LOCAL_MARKS)

    def test_the_ring_stays_bounded(self):
        for index in range(latency.MAX_RECORDS + 15):
            request_id = "req-ring-%d" % index
            latency.begin(request_id)
            latency.mark(request_id, "http_in")
            latency.finish(request_id)
        records = latency.recent(limit=latency.MAX_RECORDS + 50)
        self.assertEqual(len(records), latency.MAX_RECORDS)
        # Oldest dropped, newest kept.
        self.assertEqual(records[-1]["request_id"],
                         "req-ring-%d" % (latency.MAX_RECORDS + 14))

    def test_a_turn_with_no_marks_publishes_nothing(self):
        latency.begin("req-empty")
        self.assertIsNone(latency.finish("req-empty"))
        self.assertEqual(latency.recent(), [])

    def test_telemetry_never_raises_on_bad_input(self):
        self.assertFalse(latency.mark("unknown-id", "http_in"))
        self.assertFalse(latency.mark(None, "http_in"))
        self.assertFalse(latency.mark("unknown-id", ""))
        self.assertIsNone(latency.finish("never-started"))
        self.assertEqual(latency.merge_client_marks(None, None), 0)
        self.assertEqual(latency.merge_client_marks("unknown", [["x", 1]]), 0)
        self.assertEqual(latency.merge_client_marks("unknown", [object()]), 0)
        self.assertFalse(latency.mark_active("tts_first_byte"))   # no turn

    def test_unserializable_meta_is_coerced_not_raised(self):
        latency.begin("req-meta")
        latency.mark("req-meta", "http_in", meta={"weird": {1, 2, 3}})
        record = latency.finish("req-meta")
        json.dumps(record)          # must be JSON-safe for the endpoint
        self.assertIn("weird", record["marks"][0][2])

    def test_begin_is_idempotent_so_a_reconnect_keeps_the_clock(self):
        first = latency.begin("req-reconnect")
        latency.mark("req-reconnect", "http_in", at_ns=1_000_000_000)
        second = latency.begin("req-reconnect")
        self.assertIs(first, second)
        record = latency.finish("req-reconnect", at_ns=2_000_000_000)
        self.assertEqual(record["ms"]["http_in"], 0.0)
        self.assertEqual(len(record["marks"]), 2)   # http_in + end


class _FakeVoiceClock:
    """A second process's clock: this one plus a constant offset."""

    def __init__(self, shift_ns=5_000_000_000):
        self.shift_ns = shift_ns

    def mark(self, name, at_ns, meta=None):
        return [name, int(at_ns) - self.shift_ns, dict(meta or {})]

    def now(self):
        """The reference sample a client takes at send time."""
        return latency.local_now_ns() - self.shift_ns


class WaterfallTests(_LatencyCase):
    """The waterfall: ordered, addable, and stitched across two processes."""

    def _acceptance_turn(self):
        """The audit's example turn, with the two processes' clocks simulated.

        speech_end 0 | capture_end +520 | stt_done +760 | http_in +770 |
        first_token +1230 | tts_first_byte +1480 | playback_started +1510
        """
        base = time.perf_counter_ns()
        clock = _FakeVoiceClock()
        latency.begin("req-voice-1", origin="voice",
                      label="what is on my screen")
        # The voice worker's capture/STT marks ride the submission itself.
        latency.merge_client_marks("req-voice-1", [
            clock.mark("speech_end", base),
            clock.mark("capture_end", base + 520_000_000),
            clock.mark("stt_done", base + 760_000_000,
                       {"engine": "local-whisper"}),
        ], client_now_ns=clock.now())
        latency.mark("req-voice-1", "http_in", at_ns=base + 770_000_000,
                     meta={"route": "/ask/stream"})
        latency.mark("req-voice-1", "first_token", at_ns=base + 1230_000_000)
        latency.mark("req-voice-1", "tts_first_byte",
                     at_ns=base + 1480_000_000)
        latency.mark("req-voice-1", "playback_started",
                     at_ns=base + 1510_000_000)
        return latency.finish("req-voice-1", at_ns=base + 1520_000_000), clock

    def test_step_deltas_sum_to_the_total_turn_time(self):
        record, _clock = self._acceptance_turn()
        steps = record["steps"]
        self.assertEqual(
            [step["name"] for step in steps],
            ["speech_end", "capture_end", "stt_done", "http_in", "first_token",
             "tts_first_byte", "playback_started", "end"])
        offsets = [step["offset_ms"] for step in steps]
        self.assertEqual(offsets, sorted(offsets))    # ordered waterfall
        self.assertEqual(offsets[0], 0.0)
        # Addable: the deltas are exactly the offsets, differenced.
        self.assertAlmostEqual(sum(step["delta_ms"] for step in steps),
                               record["total_ms"], places=1)
        self.assertAlmostEqual(record["total_ms"], offsets[-1], places=1)
        self.assertEqual(steps[0]["delta_ms"], 0.0)
        self.assertEqual(record["ms"]["capture_end"], 520.0)
        self.assertEqual(record["ms"]["stt_done"], 760.0)
        self.assertEqual(steps[2]["delta_ms"], 240.0)  # capture_end -> stt_done

    def test_the_engine_that_transcribed_is_recorded(self):
        record, _clock = self._acceptance_turn()
        meta = {step["name"]: step["meta"] for step in record["steps"]}
        self.assertEqual(meta["stt_done"]["engine"], "local-whisper")
        self.assertEqual(meta["stt_done"]["clock"], "client")
        # A mark taken here carries no clock tag: it is this process's own.
        self.assertNotIn("clock", meta["http_in"])

    def test_one_offset_is_computed_per_turn_and_reused_for_later_batches(self):
        record, clock = self._acceptance_turn()
        # A late batch (a playback boundary) arrives AFTER the turn was
        # published: it must land in the SAME record, on the SAME offset. The
        # client expresses it in ITS clock — two seconds past the turn's end —
        # which the stored offset translates back exactly.
        wanted_ns = record["finished_ns"] + 2_000_000_000
        client_ns = wanted_ns - record["clock_offset_ms"] * 1_000_000
        merged = latency.merge_client_marks(
            "req-voice-1", [["playback_finished", int(client_ns)]],
            client_now_ns=clock.now())
        self.assertEqual(merged, 1)
        again = latency.recent()[0]
        self.assertEqual(again["request_id"], record["request_id"])
        self.assertIn("playback_finished", again["ms"])
        self.assertEqual(again["clock_offset_ms"], record["clock_offset_ms"])
        self.assertAlmostEqual(again["ms"]["playback_finished"],
                               record["total_ms"] + 2_000.0, delta=1.0)

    def test_a_replayed_batch_cannot_rewrite_a_boundary(self):
        self._acceptance_turn()
        clock = _FakeVoiceClock()
        self.assertEqual(
            latency.merge_client_marks(
                "req-voice-1",
                [clock.mark("stt_done", time.perf_counter_ns())],
                client_now_ns=clock.now()), 0)
        self.assertEqual(len(latency.recent()[0]["ms"]), 8)

    def test_marks_without_a_clock_reference_are_flagged(self):
        latency.begin("req-uncalibrated")
        merged = latency.merge_client_marks(
            "req-uncalibrated", [["speech_end", latency.local_now_ns()]])
        record = latency.finish("req-uncalibrated")
        self.assertEqual(merged, 1)
        self.assertEqual(record["marks"][0][2]["clock"],
                         "client-uncalibrated")
        self.assertEqual(record["clock_offset_ms"], 0)

    def test_no_reported_duration_contradicts_its_offsets(self):
        for _ in range(6):
            self._acceptance_turn()
        for record in latency.recent():
            steps = record["steps"]
            for index, step in enumerate(steps):
                self.assertEqual(step["offset_ms"], round(
                    (step["ns"] - steps[0]["ns"]) / 1e6, 1))
                expected = 0.0 if index == 0 else round(
                    (step["ns"] - steps[index - 1]["ns"]) / 1e6, 1)
                self.assertEqual(step["delta_ms"], expected)
            # The backward-compatible name -> offset map is the same truth.
            self.assertEqual(
                record["ms"],
                {step["name"]: step["offset_ms"] for step in steps})


class ReportingTests(_LatencyCase):
    """``/latency``: p50 / p90 / max per step, slowest stage first."""

    def _publish(self, count, step_ms):
        for index in range(count):
            request_id = "req-report-%d" % index
            latency.begin(request_id, origin="voice")
            latency.mark(request_id, "http_in", at_ns=index * 1_000_000)
            latency.mark(request_id, "first_token",
                         at_ns=index * 1_000_000 + step_ms * 1_000_000)
            latency.finish(request_id,
                           at_ns=index * 1_000_000 + (step_ms + 10) * 1_000_000)

    def test_per_step_p50_p90_max_over_the_window(self):
        self._publish(50, step_ms=120)
        summary = latency.summary(50)
        self.assertEqual(summary["samples"], 50)
        self.assertEqual(summary["window"], 50)
        for name in ("http_in", "first_token", "end"):
            span = summary["spans"][name]
            self.assertEqual(span["n"], 50)
            for key in ("median_ms", "p50_ms", "p90_ms", "max_ms",
                        "offset_median_ms", "offset_p50_ms", "offset_p90_ms",
                        "offset_max_ms"):
                self.assertIn(key, span)
                self.assertIsNotNone(span[key])
            self.assertEqual(span["p50_ms"], span["median_ms"])
        # The step's own cost is the delta (offset difference), and the total
        # covers it: nothing here contradicts the offsets.
        self.assertAlmostEqual(
            summary["spans"]["first_token"]["median_ms"], 120.0, places=1)
        self.assertAlmostEqual(
            summary["spans"]["first_token"]["offset_median_ms"], 120.0,
            places=1)
        self.assertEqual(summary["total"]["n"], 50)
        self.assertAlmostEqual(summary["total"]["median_ms"], 130.0, places=1)

    def test_the_waterfall_is_sorted_by_median_offset_slowest_first(self):
        self._publish(10, step_ms=80)
        summary = latency.summary(50)
        offsets = [row["median_offset_ms"] for row in summary["waterfall"]]
        self.assertEqual(offsets, sorted(offsets, reverse=True))
        self.assertEqual([row["name"] for row in summary["waterfall"]][0], "end")
        self.assertIn("median_offset_ms", summary["sorted_by"])
        for row in summary["waterfall"]:
            for key in ("name", "median_ms", "p90_ms", "max_ms",
                        "median_offset_ms", "p90_offset_ms", "max_offset_ms"):
                self.assertIn(key, row)

    def test_a_slow_stage_is_visible_first(self):
        for index in range(20):
            request_id = "req-slow-%d" % index
            latency.begin(request_id)
            latency.mark(request_id, "http_in", at_ns=index * 1_000_000)
            latency.mark(request_id, "classify_done",
                         at_ns=index * 1_000_000 + 900_000_000)
            latency.mark(request_id, "first_token",
                         at_ns=index * 1_000_000 + 950_000_000)
            latency.finish(request_id,
                           at_ns=index * 1_000_000 + 960_000_000)
        summary = latency.summary(50)
        classify = summary["spans"]["classify_done"]
        self.assertAlmostEqual(classify["median_ms"], 900.0, places=1)
        self.assertAlmostEqual(classify["p90_ms"], 900.0, places=1)
        # The slow stage is a row of its own, ahead of every cheap stage.
        self.assertIn("classify_done",
                      [row["name"] for row in summary["waterfall"]])
        self.assertAlmostEqual(
            summary["spans"]["first_token"]["median_ms"], 50.0, places=1)

    def test_the_legacy_span_names_still_report(self):
        self._publish(5, step_ms=50)
        spans = latency.summary(50)["spans"]
        for alias, target in latency.LEGACY_SPAN_ALIASES.items():
            if target in spans:
                self.assertIn(alias, spans)
                self.assertEqual(spans[alias]["alias_of"], target)

    def test_an_empty_ring_summarises_without_raising(self):
        summary = latency.summary(50)
        self.assertEqual(summary["samples"], 0)
        self.assertEqual(summary["spans"], {})
        self.assertEqual(summary["waterfall"], [])
        self.assertNotIn("total", summary)


class EndpointTests(_LatencyCase):
    """The two endpoints, and the coupling requirement 7 removed."""

    def test_latency_keeps_its_response_shape_and_adds_the_waterfall(self):
        latency.begin("req-endpoint")
        latency.mark("req-endpoint", "http_in")
        latency.mark("req-endpoint", "first_token")
        latency.finish("req-endpoint")
        payload = routes.get_latency()
        self.assertIn("summary", payload)
        self.assertIn("recent", payload)
        self.assertIn("aec", payload)
        self.assertIn("listener_aec_errors", payload["aec"])
        summary = payload["summary"]
        self.assertIn("spans", summary)
        self.assertIn("waterfall", summary)
        self.assertIn("first_token", summary["spans"])
        for key in ("median_ms", "p90_ms", "max_ms"):
            self.assertIn(key, summary["spans"]["first_token"])
        self.assertIn("total", summary)
        self.assertIn("steps", payload["recent"][-1])

    def test_post_latency_client_merges_under_the_same_request_id(self):
        clock = _FakeVoiceClock()
        latency.begin("req-merge", origin="voice")
        response = routes.post_latency_client(routes.ClientMarks(
            request_id="req-merge",
            marks=[clock.mark("speech_end", time.perf_counter_ns()),
                   clock.mark("stt_done", time.perf_counter_ns() + 40_000_000,
                              {"engine": "inworld"})],
            client_now_ns=clock.now(),
        ))
        self.assertTrue(response["ok"])
        self.assertEqual(response["merged"], 2)
        record = latency.finish("req-merge")
        self.assertIn("speech_end", record["ms"])
        self.assertEqual(
            {step["name"]: step["meta"] for step in record["steps"]}
            ["stt_done"]["engine"], "inworld")

    def test_post_latency_client_never_raises_on_a_bad_batch(self):
        self.assertEqual(
            routes.post_latency_client(routes.ClientMarks(
                request_id="", marks=[["speech_end", "not-a-clock"]])),
            {"ok": True, "merged": 0})

    def test_the_latency_endpoint_does_not_import_the_listener(self):
        """Requirement 7: the backend process owns no microphone, so the
        endpoint must not pull the listener module in to read a counter."""
        source = inspect.getsource(routes)
        self.assertNotIn("from backend.services import listener", source)
        self.assertNotIn("_listener_aec_error_count", source)

    def test_the_aec_count_comes_from_the_voice_worker_snapshot(self):
        with routes._published_voice_lock:
            routes._published_voice["aec_errors"] = 3
        self.addCleanup(routes._published_voice.clear)
        self.assertEqual(routes.get_latency()["aec"]["listener_aec_errors"], 3)
        with routes._published_voice_lock:
            routes._published_voice.pop("aec_errors", None)
        self.assertEqual(routes.get_latency()["aec"]["listener_aec_errors"], 0)

    def test_a_fresh_process_serving_latency_never_loads_the_listener(self):
        """The real import graph, not just the source text."""
        repo_root = Path(__file__).resolve().parents[2]
        code = (
            "import sys; from backend.api import routes; routes.get_latency();"
            "print('backend.services.listener' in sys.modules)"
        )
        result = subprocess.run([sys.executable, "-c", code],
                                cwd=str(repo_root), capture_output=True,
                                text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("False", result.stdout)


class _FakeSpeaker:
    """A StreamSpeaker stand-in: the tests never touch a TTS engine."""

    def __init__(self):
        self.fed = []
        self.finished = False
        self.closed = False
        self.spoken_any = True

    def feed(self, delta):
        self.fed.append(delta)

    def finish(self):
        self.finished = True

    def close(self):
        self.closed = True


class VoiceWorkerTests(_LatencyCase):
    """Requirement 6: the worker's marks ride its own submission."""

    def _wait_for_turn_handover(self, turn, timeout=2.0):
        """Wait for the turn to be handed to the publisher.

        [P0-08] The handover now happens at the END of the turn, which runs on
        its own worker thread (the audit's requirement 3: `brain_thread`
        dispatches instead of waiting). These tests used to observe it
        synchronously; the property they pin — the turn is handed over and then
        released once its grace period expires — is unchanged, only the moment
        of the handover moved off the dispatcher.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if voice_mode._latency_ship["turn"] is turn:
                return True
            time.sleep(0.01)
        return voice_mode._latency_ship["turn"] is turn

    def tearDown(self):
        _LatencyCase.tearDown(self)
        voice_mode._latency_ship.update({"turn": None, "request_id": "",
                                        "until": 0.0})
        voice_mode.set_active_stream(None)

    def test_the_submission_carries_the_marks_under_the_same_request_id(self):
        captured = {}

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def __iter__(self):
                return iter([b'data: {"type": "completed", "reply": "ok"}\n\n'])

        def _urlopen(request, timeout=None):
            captured["path"] = request.full_url
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _Response()

        turn = latency.new_local_turn()
        turn.mark("speech_end")
        turn.mark("stt_done", {"engine": "local-whisper"})

        with patch.object(voice_mode, "urlopen", side_effect=_urlopen):
            reply = voice_mode._ask_backend(
                "hello", "req-voice-1", client_marks=turn.marks())

        self.assertEqual(reply, "ok")
        self.assertTrue(captured["path"].endswith("/ask/stream"))
        body = captured["body"]
        # One request id — the marks never mint a second turn.
        self.assertEqual(body["request_id"], "req-voice-1")
        self.assertEqual([mark[0] for mark in body["client_marks"]],
                         ["speech_end", "stt_done"])
        for mark in body["client_marks"]:
            self.assertIsInstance(mark[1], int)
        self.assertGreater(body["client_now_ns"], 0)

    def test_the_playback_marks_are_shipped_late_under_that_same_id(self):
        turn = latency.new_local_turn()
        turn.mark("speech_end")
        posted = []
        speaker = _FakeSpeaker()

        with patch.object(voice_mode, "backend_task_running",
                          return_value=False), \
             patch.object(voice_mode, "StreamSpeaker",
                          return_value=speaker), \
             patch.object(voice_mode, "speak"), \
             patch.object(voice_mode, "_post_backend",
                          side_effect=lambda path, payload, timeout=2.5:
                          (posted.append((path, payload)) or (True, {}))):

            def _fake_ask(text, request_id=None, **kwargs):
                posted.append(("/ask/stream", {
                    "request_id": request_id,
                    "client_marks": kwargs.get("client_marks", []),
                }))
                # The audio actor marks the turn while the reply is spoken.
                turn.mark("tts_first_byte")
                turn.mark("playback_started")
                return "the reply"

            with patch.object(voice_mode, "_ask_backend",
                              side_effect=_fake_ask):
                voice_mode._respond_to_utterance("hello", turn)
            # [P0-08] the turn runs on its own worker thread now
            self.assertTrue(self._wait_for_turn_handover(turn),
                            "the turn was never handed to the publisher")
            # The publisher's cadence ships whatever arrived after submission.
            voice_mode._ship_turn_marks()

        submitted = [payload for path, payload in posted
                     if path == "/ask/stream"][0]
        shipped = [payload for path, payload in posted
                   if path == "/latency/client"]
        self.assertEqual(len(shipped), 1)
        self.assertEqual(shipped[0]["request_id"], submitted["request_id"])
        self.assertEqual([mark[0] for mark in shipped[0]["marks"]],
                         ["tts_first_byte", "playback_started"])
        # The early batch was handed over with the submission, not re-sent.
        self.assertEqual([mark[0] for mark in submitted["client_marks"]],
                         ["speech_end"])
        self.assertGreater(shipped[0]["client_now_ns"], 0)

    def test_shipping_stops_at_the_turn_grace_period(self):
        turn = latency.new_local_turn()
        turn.mark("speech_end")
        with patch.object(voice_mode, "backend_task_running",
                          return_value=False), \
             patch.object(voice_mode, "StreamSpeaker",
                          return_value=_FakeSpeaker()), \
             patch.object(voice_mode, "speak"), \
             patch.object(voice_mode, "_post_backend",
                          return_value=(True, {})):
            with patch.object(voice_mode, "_ask_backend",
                              return_value="the reply"):
                voice_mode._respond_to_utterance("hello", turn)
            # [P0-08] the handover happens at the END of the per-turn worker
            # thread, so it is awaited INSIDE this patch window (outside it the
            # worker would reach the real StreamSpeaker/speak).
            self.assertTrue(self._wait_for_turn_handover(turn),
                            "the turn was never handed to the publisher")
        self.assertIs(voice_mode._latency_ship["turn"], turn)
        voice_mode._latency_ship["until"] = time.monotonic() - 1.0
        voice_mode._ship_turn_marks()
        self.assertIsNone(voice_mode._latency_ship["turn"])
        self.assertIsNone(latency.local_turn())


class ListenerMarkTests(_LatencyCase):
    """The voice process marks its own capture/STT boundaries."""

    def test_the_stt_boundaries_record_the_engine_that_transcribed(self):
        turn = latency.new_local_turn()
        audio = sr.AudioData(b"\x00" * 3200, 16000, 2)

        def _fake_stt(_audio):
            listener.LAST_STT_ENGINE = "inworld"
            return "hello", "hello", "en"

        with patch.object(listener, "_capture_audio", return_value=audio), \
             patch.object(listener, "recognize_multilingual",
                          side_effect=_fake_stt):
            self.assertEqual(listener.listen(marks=turn), "hello")
        marks = {mark[0]: mark[2] for mark in turn.marks()}
        self.assertIn("stt_start", marks)
        self.assertEqual(marks["stt_done"]["engine"], "inworld")
        self.assertEqual(marks["stt_done"]["language"], "en")

    def test_capture_marks_speech_end_and_capture_end(self):
        turn = latency.new_local_turn()

        class _Source:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        stream = [sr.AudioData(b"\x10\x00" * 1600, 16000, 2) for _ in range(3)]
        with patch.object(listener, "_get_microphone_source",
                          return_value=_Source()), \
             patch.object(listener.recognizer, "listen", return_value=stream), \
             patch.object(listener, "_aec_filter_chunk",
                          side_effect=lambda chunk, fid, t: (chunk, False,
                                                             False)), \
             patch.object(listener, "_should_confirm_speech_start",
                          return_value=True), \
             patch.object(listener, "barge_in_on_speech_onset"), \
             patch.object(listener, "play_capture_complete_earcon"), \
             patch.object(listener, "PARTIAL_TRANSCRIBE_MIN_SECONDS", 100.0), \
             patch.object(listener, "is_human_voice", return_value=True):
            audio = listener._capture_audio(turn)
        self.assertIsNotNone(audio)
        names = [mark[0] for mark in turn.marks()]
        self.assertEqual(names, ["speech_end", "capture_end"])
        # speech_end is the origin of the turn and precedes capture_end.
        self.assertEqual(turn.marks()[0][2]["frames"], 3)
        self.assertGreater(turn.marks()[1][1], turn.marks()[0][1])

    def test_capture_and_listen_still_work_without_a_telemetry_sink(self):
        with patch.object(listener, "_capture_audio", return_value=None):
            self.assertIsNone(listener.listen())


class ProviderMarkTests(_LatencyCase):
    """``provider_headers`` — the split that makes a slow turn diagnosable."""

    def test_the_response_boundary_lands_on_the_active_turn(self):
        latency.begin("req-provider", origin="ui")
        latency.set_active_request("req-provider")
        latency.mark_provider_headers("openrouter",
                                      "google/gemini-2.5-flash-lite")
        latency.mark("req-provider", "first_token")
        record = latency.finish("req-provider")
        meta = {step["name"]: step["meta"] for step in record["steps"]}
        self.assertEqual(meta["provider_headers"]["provider"], "openrouter")
        self.assertEqual(meta["provider_headers"]["model"],
                         "google/gemini-2.5-flash-lite")
        self.assertIn("provider_headers", record["ms"])

    def test_a_fallback_provider_cannot_rewrite_the_boundary(self):
        latency.begin("req-fallback")
        latency.set_active_request("req-fallback")
        self.assertTrue(latency.mark_provider_headers("gemini", "m1"))
        self.assertFalse(latency.mark_provider_headers("fireworks", "m2"))
        record = latency.finish("req-fallback")
        meta = {step["name"]: step["meta"] for step in record["steps"]}
        self.assertEqual(meta["provider_headers"]["provider"], "gemini")

    def test_with_no_turn_in_flight_nothing_is_recorded(self):
        self.assertFalse(latency.mark_provider_headers("gemini", "m"))
        self.assertEqual(latency.recent(), [])

    def test_the_openai_compatible_stream_marks_its_response(self):
        latency.begin("req-stream")
        latency.set_active_request("req-stream")
        # requests with decode_unicode=True yields str lines, not bytes.
        lines = ['data: {"choices": [{"delta": {"content": "hi"}}]}',
                 "data: [DONE]"]

        class _Response:
            status_code = 200

            def iter_lines(self, decode_unicode=True):
                return iter(lines)

        with patch.object(openai_compat_client._session, "post",
                          return_value=_Response()):
            deltas = list(openai_compat_client.ask_openai_compat_stream(
                [{"role": "user", "content": "hi"}], "test-model",
                "https://example.invalid/v1", "test-key"))
        self.assertEqual(deltas, ["hi"])
        record = latency.finish("req-stream")
        meta = {step["name"]: step["meta"] for step in record["steps"]}
        self.assertEqual(meta["provider_headers"]["provider"], "openai-compat")
        self.assertEqual(meta["provider_headers"]["model"], "test-model")

    def test_every_stream_adapter_marks_its_response(self):
        """Gemini and Fireworks reuse the shared marker; a dropped call would
        leave their streams invisible in the waterfall."""
        for module in (gemini_client, fireworks_client):
            with self.subTest(module=module.__name__):
                self.assertIn("_mark_headers(", inspect.getsource(module))


if __name__ == "__main__":
    unittest.main()


