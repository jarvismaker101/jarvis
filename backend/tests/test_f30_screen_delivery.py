"""F30 — deliver the screen answer before its decorations.

Acceptance (audit report): "Older A finishing after B cannot replace B; images
do not delay text; stale races fail; hanging enrichment cannot create
unbounded work."

Baseline defects pinned here:
  * the capture identity was created AFTER the slow vision call, so a late
    answer from an OLDER question could replace a newer one;
  * enrichment threads were unbounded (one per answer, forever);
  * patch merge was not atomic and a patch could replace the authoritative
    tip/evidence.
"""

import threading
import time
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from backend.api import routes
from backend.core import brain


def _reset_screen_state():
    routes._screen_answer_id = 0
    routes._screen_answer_capture = ""
    routes._screen_answer_seq = 0
    routes._screen_answer_data = {
        "id": 0, "tip": "", "evidence": [], "links": [], "images": [],
        "region": {}, "request_id": "", "capture_id": "",
        "revision": 0, "enriched": False, "timestamp": 0,
    }


class CaptureGenerationTests(unittest.TestCase):
    def test_generation_is_registered_before_analysis(self):
        first = brain.begin_screen_capture("req-1")
        second = brain.begin_screen_capture("req-2")
        self.assertLess(first["seq"], second["seq"])
        self.assertNotEqual(first["capture_id"], second["capture_id"])

    def test_order_matters_not_wall_clock(self):
        early = brain.begin_screen_capture()
        late = brain.begin_screen_capture()
        self.assertLess(early["seq"], late["seq"])
        self.assertIn("capture_id", early)


class StaleInitialPostTests(unittest.TestCase):
    """Older A finishing after B cannot replace B."""

    def setUp(self):
        _reset_screen_state()

    def _post(self, **kwargs):
        payload = routes.ScreenAnswer(**kwargs)
        return routes.post_screen_answer(payload)

    def test_late_initial_post_from_an_older_capture_is_refused(self):
        # B (generation 2) publishes first...
        self._post(tip="Answer B", capture_id="capB", capture_seq=2)
        # ...then A (generation 1) finishes and tries to publish.
        with self.assertRaises(HTTPException) as caught:
            self._post(tip="Answer A", capture_id="capA", capture_seq=1)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(routes._screen_answer_data["tip"], "Answer B")

    def test_newer_initial_post_replaces_the_answer(self):
        self._post(tip="Answer A", capture_id="capA", capture_seq=1)
        self._post(tip="Answer B", capture_id="capB", capture_seq=2)
        self.assertEqual(routes._screen_answer_data["tip"], "Answer B")
        self.assertEqual(routes._screen_answer_data["capture_seq"], 2)

    def test_unsequenced_posts_still_work(self):
        # Back-compat: a caller that does not send a generation keeps the old
        # behaviour (last write wins).
        self._post(tip="A", capture_id="capA")
        self._post(tip="B", capture_id="capB")
        self.assertEqual(routes._screen_answer_data["tip"], "B")


class EnrichmentPatchTests(unittest.TestCase):
    def setUp(self):
        _reset_screen_state()
        self.posted = routes.post_screen_answer(
            routes.ScreenAnswer(tip="Authoritative", capture_id="capA",
                                capture_seq=1,
                                evidence=[{"source": "screen", "snippet": "seen on screen"}]))

    def _patch(self, **kwargs):
        return routes.post_screen_answer(routes.ScreenAnswer(**kwargs))

    def test_patch_adds_decorations_in_place(self):
        answer_id = self.posted["id"]
        result = self._patch(id=answer_id, capture_id="capA", capture_seq=1,
                             tip="", evidence=[],
                             links=[{"label": "Docs", "url": "https://d.test"}])
        self.assertEqual(result["id"], answer_id, "no new answer id")
        self.assertTrue(result["revision"] >= 1)
        data = routes._screen_answer_data
        self.assertEqual(data["tip"], "Authoritative")
        self.assertEqual(data["links"][0]["url"], "https://d.test")

    def test_patch_may_not_replace_the_answer_text(self):
        with self.assertRaises(HTTPException) as caught:
            self._patch(id=self.posted["id"], capture_id="capA", capture_seq=1,
                        tip="Decorations win")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(routes._screen_answer_data["tip"], "Authoritative")

    def test_patch_may_not_replace_the_evidence(self):
        with self.assertRaises(HTTPException) as caught:
            self._patch(id=self.posted["id"], capture_id="capA", capture_seq=1,
                        tip="", evidence=[{"source": "screen", "snippet": "invented"}])
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(routes._screen_answer_data["evidence"][0]["snippet"],
                         "seen on screen")

    def test_patch_echoing_the_same_text_is_allowed(self):
        result = self._patch(id=self.posted["id"], capture_id="capA",
                             capture_seq=1, tip="Authoritative",
                             evidence=[{"source": "screen", "snippet": "seen on screen"}])
        self.assertTrue(result["updated"])

    def test_patch_from_an_older_generation_is_refused(self):
        routes.post_screen_answer(routes.ScreenAnswer(
            tip="Answer B", capture_id="capB", capture_seq=2))
        with self.assertRaises(HTTPException) as caught:
            routes.post_screen_answer(routes.ScreenAnswer(
                tip="", id=1, capture_id="capA", capture_seq=1,
                links=[{"label": "x", "url": "https://x.test"}]))
        self.assertEqual(caught.exception.status_code, 409)

    def test_patch_to_a_non_newest_id_is_refused(self):
        routes.post_screen_answer(routes.ScreenAnswer(
            tip="Answer B", capture_id="capB", capture_seq=2))
        with self.assertRaises(HTTPException) as caught:
            self._patch(id=self.posted["id"], capture_id="capA", capture_seq=1,
                        tip="", links=[{"label": "x", "url": "https://x.test"}])
        self.assertEqual(caught.exception.status_code, 409)

    def test_concurrent_publishes_do_not_lose_the_newest(self):
        _reset_screen_state()
        errors = []

        def worker(seq):
            try:
                routes.post_screen_answer(routes.ScreenAnswer(
                    tip="tip-%d" % seq, capture_id="cap%d" % seq,
                    capture_seq=seq))
            except HTTPException:
                pass
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(1, 9)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertFalse(errors)
        self.assertEqual(routes._screen_answer_data["capture_seq"], 8,
                         "the newest generation must be the one on screen")


class BoundedEnrichmentTests(unittest.TestCase):
    def setUp(self):
        brain._screen_enrich_active = 0

    def tearDown(self):
        brain._screen_enrich_active = 0

    def test_enrichment_is_capped(self):
        started = []
        release = threading.Event()

        def fake_enrich(*args, **kwargs):
            started.append(args)
            release.wait(2)

        capture = brain.begin_screen_capture()
        with patch.object(brain, "_enrich_screen_answer",
                          side_effect=fake_enrich):
            results = [
                brain._start_screen_enrichment(
                    1, capture, "tip", [], [], "topic", False, {})
                for _ in range(brain.SCREEN_ENRICH_MAX_THREADS + 3)
            ]
        self.assertTrue(all(results[:brain.SCREEN_ENRICH_MAX_THREADS]))
        self.assertTrue(any(r is False for r in results[brain.SCREEN_ENRICH_MAX_THREADS:]),
                        "past the cap the decoration phase must be skipped")
        release.set()
        deadline = time.time() + 3
        while brain._screen_enrich_active and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(brain._screen_enrich_active, 0,
                         "the slot must be released when the phase ends")

    def test_answer_is_published_before_enrichment_starts(self):
        order = []
        capture = brain.begin_screen_capture()

        def fake_push(*args, **kwargs):
            order.append("publish")
            return 7

        def fake_start(*args, **kwargs):
            order.append("enrich")
            return True

        with patch.object(brain, "push_screen_answer", side_effect=fake_push), \
             patch.object(brain, "_start_screen_enrichment",
                          side_effect=fake_start), \
             patch.object(brain, "analyze_screen",
                          return_value={"tip": "the answer",
                                        "evidence": [], "topic": "cats"}), \
             patch.object(brain, "sync_voice_log"), \
             patch.object(brain, "_screen_qa_busy",
                          threading.Semaphore(1)):
            reply = brain.process_message("what is on my screen",
                                          sync_voice=False)
        self.assertIn("publish", order)
        if "enrich" in order:
            self.assertLess(order.index("publish"), order.index("enrich"),
                            "text must never wait for decorations")
        self.assertEqual(reply, "the answer")


class WiringTests(unittest.TestCase):
    def test_capture_is_registered_before_analysis(self):
        import inspect

        code = inspect.getsource(brain._process_message_inner)
        begin = code.index("begin_screen_capture")
        analyse = code.index("analyze_screen(msg)")
        self.assertLess(begin, analyse,
                        "the generation must be registered before analysis")

    def test_no_unbounded_thread_per_answer(self):
        import inspect

        code = inspect.getsource(brain._process_message_inner)
        segment = code[code.index("begin_screen_capture"):]
        self.assertIn("_start_screen_enrichment", segment)

    def test_push_carries_the_generation(self):
        import inspect

        code = inspect.getsource(brain.push_screen_answer)
        self.assertIn("capture_seq", code)


if __name__ == "__main__":
    unittest.main()
