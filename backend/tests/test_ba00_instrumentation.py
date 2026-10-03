"""BA-00: turn-level instrumentation for the browser agent.

Covers the span recorder, the wasted-turn census, the result classifier,
the cross-task percentile reservoir, the summary census lines, and the
end-to-end hookup (a stale-mark tool result lands in the census).
"""
import io
import json
import unittest
from unittest.mock import patch

from backend.services import browser_agent
from backend.services.browser_agent import _SwStats


class FakeLookClient:
    """Answers the three daemon calls a composite look makes."""

    def __init__(self):
        self.calls = []

    def call_tool(self, name, arguments):
        self.calls.append(name)
        if name == "evaluate":
            expr = (arguments or {}).get("expression", "")
            if "elements" in expr:
                return json.dumps({
                    "elements": [{
                        "x": 10, "y": 20, "w": 100, "h": 30,
                        "tag": "button", "label": "Go", "inView": True,
                        "cssPath": "body>button:nth-of-type(1)",
                        "epoch": {"doc": 111, "mut": 0}, "dpr": 1,
                    }],
                    "url": "https://example.com/",
                    "title": "Example",
                    "epoch": {"doc": 111, "mut": 0},
                    "dpr": 1,
                })
            return json.dumps({
                "url": "https://example.com/",
                "doc": 111,
                "dpr": 1,
            })
        if name == "list_tabs":
            return "[]"
        if name == "screenshot":
            from PIL import Image
            path = (arguments or {}).get("path", "")
            Image.new("RGB", (64, 64), (255, 0, 0)).save(path, format="PNG")
            return "saved"
        raise RuntimeError("unexpected tool %s" % name)


class SpanTests(unittest.TestCase):
    def test_done_records_and_returns_ms(self):
        stats = _SwStats()
        ms = browser_agent._Span("look.jpeg").done(stats)
        self.assertGreaterEqual(ms, 0)
        self.assertIn("look.jpeg", stats.spans)
        total, count = stats.spans["look.jpeg"]
        self.assertEqual(count, 1)
        self.assertGreaterEqual(total, 0)

    def test_done_without_stats_still_times(self):
        ms = browser_agent._Span("x").done(None)
        self.assertGreaterEqual(ms, 0)

    def test_span_map_is_bounded(self):
        stats = _SwStats()
        for i in range(browser_agent._SPAN_LIMIT + 20):
            stats.record_span("span-%d" % i, 1)
        self.assertLessEqual(len(stats.spans), browser_agent._SPAN_LIMIT)

    def test_span_table_sorted_descending(self):
        stats = _SwStats()
        stats.record_span("a", 5)
        stats.record_span("b", 50)
        table = stats.span_table()
        self.assertEqual(table[0][0], "b")
        self.assertEqual(table[0][1], 50)
        self.assertEqual(table[0][2], 1)


class WastedTurnTests(unittest.TestCase):
    def test_all_wasted_counts_one(self):
        stats = _SwStats()
        stats.begin_turn()
        stats.record_outcome(True)
        stats.record_outcome(True)
        stats.end_turn()
        self.assertEqual(stats.wasted_turns, 1)

    def test_mixed_turn_counts_zero(self):
        stats = _SwStats()
        stats.begin_turn()
        stats.record_outcome(True)
        stats.record_outcome(False)
        stats.end_turn()
        self.assertEqual(stats.wasted_turns, 0)

    def test_empty_turn_counts_zero(self):
        stats = _SwStats()
        stats.begin_turn()
        stats.end_turn()
        self.assertEqual(stats.wasted_turns, 0)

    def test_model_turns_count_steps_not_retries(self):
        stats = _SwStats()
        stats.record_model("s1", 10, ok=False, retry=True)
        stats.record_model("s1", 20, ok=True)
        self.assertEqual(stats.model_turns, 1)
        self.assertEqual(stats.model_calls, 1)


class ClassifierTests(unittest.TestCase):
    def test_mut_stale(self):
        wasted, kind = browser_agent._ba00_classify_result(
            "click_mark error: That mark is stale (the page content changed "
            "since the look). Call look again.")
        self.assertEqual((wasted, kind), (True, "mut"))

    def test_other_stale_variants(self):
        for text in (
            "That mark is from an older page (the document changed since the look).",
            "observed at a different display scale",
            "belongs to a different page",
            "belongs to a different frame",
            "element is gone from the page",
            "carries no page-identity stamp",
            "Target check failed (boom)",
            "mark 3 not found. Available marks: [1, 2].",
            "No marks available - call look first.",
        ):
            wasted, kind = browser_agent._ba00_classify_result("x error: " + text)
            self.assertEqual((wasted, kind), (True, "other"), text)

    def test_offscreen(self):
        wasted, kind = browser_agent._ba00_classify_result(
            "click_mark error: mark 2 is off-screen (outside the current "
            "viewport) - scroll it into view first")
        self.assertEqual((wasted, kind), (True, "offscreen"))

    def test_refusals(self):
        for text in (
            "tool call refused: the arguments for x were not valid JSON",
            "tool blocked by policy: no grant",
            "not replayed: the previous click attempt's outcome is unknown",
            "action blocked after 3 identical failures",
            "upload_file refused: origin unknown",
            "look failed: screenshot error: boom",
            "tool failed: browser died",
        ):
            wasted, kind = browser_agent._ba00_classify_result(text)
            self.assertTrue(wasted, text)
            self.assertEqual(kind, "")

    def test_clean_results_not_wasted(self):
        for text in (
            "Navigated to https://example.com",
            "clicked (url=https://example.com/, navigated=no)",
            "Page: https://example.com/\nTitle: Example",
            "",
            None,
        ):
            self.assertEqual(browser_agent._ba00_classify_result(text),
                             (False, ""))

    def test_page_text_does_not_false_positive(self):
        # Native daemon tools return raw page text — it must not trip the
        # census just for containing ordinary words.
        wasted, _kind = browser_agent._ba00_classify_result(
            "Error 404: the requested article was not found on this server.")
        self.assertFalse(wasted)


class NoteResultTests(unittest.TestCase):
    def test_counters_land_in_the_right_buckets(self):
        stats = _SwStats()
        stats.begin_turn()
        browser_agent._ba00_note_result(
            stats, "That mark is stale (the page content changed since the look)")
        browser_agent._ba00_note_result(stats, "belongs to a different frame")
        browser_agent._ba00_note_result(stats, "mark 1 is off-screen (outside the "
                                               "current viewport)")
        browser_agent._ba00_note_result(stats, "tool blocked by policy: x")
        stats.end_turn()
        self.assertEqual(stats.stale_refusals, 1)
        self.assertEqual(stats.stale_refusals_other, 1)
        self.assertEqual(stats.offscreen_refusals, 1)
        # Four refused outcomes, one turn: the turn is wasted.
        self.assertEqual(stats.wasted_turns, 1)

    def test_none_stats_never_raises(self):
        browser_agent._ba00_note_result(None, "anything")
        self.assertEqual(browser_agent._ba00_classify_result(None), (False, ""))


class PercentileTests(unittest.TestCase):
    def test_nearest_rank(self):
        self.assertEqual(browser_agent._percentile([], 50), 0)
        vals = [10, 20, 30, 40, 50]
        self.assertEqual(browser_agent._percentile(vals, 50), 30)
        self.assertEqual(browser_agent._percentile(vals, 90), 50)
        self.assertEqual(browser_agent._percentile(vals, 0), 10)

    def test_snapshot_shape_and_task_recording(self):
        saved_samples = list(browser_agent._perf_turn_ms)
        saved_tasks = browser_agent._perf_tasks
        try:
            stats = _SwStats()
            stats.record_model("s1", 100, ok=True)
            stats.record_model("s2", 300, ok=True)
            browser_agent._perf_record_task(stats)
            snap = browser_agent.browser_agent_perf_snapshot()
            self.assertEqual(snap["tasks_observed"], saved_tasks + 1)
            self.assertEqual(snap["turn_samples"], len(saved_samples) + 1)
            # Mean of (100, 300) = 200 must be inside [min, max].
            self.assertLessEqual(snap["mean_turn_ms_min"], 200)
            self.assertGreaterEqual(snap["mean_turn_ms_max"], 200)
            self.assertIn("mean_turn_ms_p50", snap)
            self.assertIn("mean_turn_ms_p90", snap)
        finally:
            browser_agent._perf_turn_ms[:] = saved_samples
            browser_agent._perf_tasks = saved_tasks

    def test_record_task_with_no_turns_counts_task_only(self):
        saved_samples = list(browser_agent._perf_turn_ms)
        saved_tasks = browser_agent._perf_tasks
        try:
            browser_agent._perf_record_task(_SwStats())
            snap = browser_agent.browser_agent_perf_snapshot()
            self.assertEqual(snap["tasks_observed"], saved_tasks + 1)
            self.assertEqual(snap["turn_samples"], len(saved_samples))
        finally:
            browser_agent._perf_turn_ms[:] = saved_samples
            browser_agent._perf_tasks = saved_tasks


class SummaryTests(unittest.TestCase):
    def test_census_lines_emitted(self):
        stats = _SwStats()
        stats.record_model("s1", 120, ok=True)
        stats.begin_turn()
        browser_agent._ba00_note_result(stats, "tool blocked by policy: x")
        stats.end_turn()
        stats.record_span("look.jpeg", 30)
        stats.note_image_upload(1000, 90)
        lines = []
        with patch.object(browser_agent, "append_activity_line",
                          side_effect=lambda s: lines.append(s)):
            with patch.object(browser_agent.config,
                              "BROWSER_AGENT_TIMEOUT", 480, create=True):
                browser_agent._emit_summary(stats, 0.0)
        blob = "".join(lines)
        self.assertIn("STOPWATCH census model_turns=1 wasted_turns=1", blob)
        self.assertIn("wasted_pct=100", blob)
        self.assertIn("STOPWATCH upload bytes_uploaded=1000", blob)
        self.assertIn("STOPWATCH span look.jpeg total_ms=30 count=1", blob)


class LookSpanTests(unittest.TestCase):
    def test_handle_look_records_stage_spans(self):
        stats = _SwStats()
        client = FakeLookClient()
        with patch.object(browser_agent, "append_activity_line",
                          lambda *a, **k: None):
            text, b64 = browser_agent._handle_look(client, {}, stats)
        self.assertTrue(b64)
        self.assertIn("Page: https://example.com/", text)
        for name in ("look.evaluate", "look.list_tabs", "look.screenshot",
                     "look.capture_state", "look.pil", "look.jpeg",
                     "look.b64"):
            self.assertIn(name, stats.spans, name)
            self.assertEqual(stats.spans[name][1], 1)
        self.assertEqual(stats.look_count, 1)
        self.assertGreater(stats.bytes_uploaded, 0)
        self.assertGreater(stats.image_tokens_est, 0)


class ModelSpanTests(unittest.TestCase):
    def test_retries_record_serialize_and_total_spans(self):
        stats = _SwStats()
        history = [{"role": "user", "content": "go"}]
        with patch.object(browser_agent, "_model_turn",
                          return_value=("done", [])), \
             patch.object(browser_agent, "append_activity_line",
                          lambda *a, **k: None):
            text, calls = browser_agent._model_turn_with_retries(
                history, [], step="s1", stats=stats)
        self.assertEqual(text, "done")
        self.assertIn("model.serialize", stats.spans)
        self.assertIn("model.total", stats.spans)
        self.assertGreater(stats.upload_bytes_total, 0)
        self.assertEqual(stats.model_turns, 1)


class RunOneToolCensusTests(unittest.TestCase):
    def _run(self, call, session=None):
        stats = _SwStats()
        history = []
        stats.begin_turn()
        with patch.object(browser_agent, "append_activity_line",
                          lambda *a, **k: None), \
             patch.object(browser_agent, "_log_tool_result",
                          lambda *a, **k: None), \
             patch.object(browser_agent, "narrate_activity",
                          lambda *a, **k: None):
            browser_agent._run_one_tool(FakeLookClient(), history, call,
                                        session if session is not None else {},
                                        stats)
        stats.end_turn()
        return stats, history

    def test_missing_mark_counts_stale_other_and_wastes_turn(self):
        stats, history = self._run(
            {"name": "click_mark", "arguments": {"index": 9}, "id": "c1"})
        self.assertEqual(stats.stale_refusals_other, 1)
        self.assertEqual(stats.stale_refusals, 0)
        self.assertEqual(stats.wasted_turns, 1)
        self.assertIn("not found", history[-1]["content"])

    def test_parse_error_counts_wasted(self):
        stats, history = self._run(
            {"name": "click_mark", "arguments": {}, "id": "c2",
             "parse_error": "boom"})
        self.assertEqual(stats.wasted_turns, 1)
        self.assertIn("tool call refused", history[-1]["content"])


if __name__ == "__main__":
    unittest.main()
