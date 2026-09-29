"""F07 — remember WORK, not just chat.

Acceptance (audit report): "After restart, a report follow-up retrieves its
actual path and sources; every request links to its outcome; no event field
leaks seeded secrets."

Baseline defects pinned here:
  * there was no central IDENTIFIED request/terminal-event contract — a chat
    exchange row existed, but a request could not be linked to its outcome;
  * research artifact/source structure was lost (only a clipped description);
  * clipping the serialised detail could break the structure a follow-up needs;
  * no unified work-event retrieval for normal chat.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from backend.core import memory_store

SECRET = "sk-live-F07SECRET0123456789"


class WorkEventTestCase(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp(prefix="f07mem")
        self._path = os.path.join(self._dir, "memory.db")
        memory_store.configure(self._path)
        self.assertTrue(memory_store.MEMORY_ENABLED)

    def tearDown(self):
        try:
            memory_store.configure(":memory:")
        except Exception:
            pass
        try:
            os.remove(self._path)
        except Exception:
            pass


class RequestOutcomeLinkTests(WorkEventTestCase):
    def test_every_request_links_to_its_outcome(self):
        rid = memory_store.record_request("find the price of the new laptop",
                                          route="research")
        self.assertTrue(rid)
        self.assertEqual(memory_store.get_work_request(rid)["status"], "open")
        memory_store.record_result(rid, "completed", summary="799 dollars",
                                   artifacts=[{"kind": "report",
                                               "path": r"C:\reports\price.md"}])
        row = memory_store.get_work_request(rid)
        self.assertEqual(row["status"], "completed")
        self.assertIsNotNone(row["closed_at"])
        events = memory_store.work_events(request_id=rid)
        kinds = [e["kind"] for e in events]
        self.assertIn("work_request", kinds)
        self.assertIn("work_result", kinds)
        # The terminal event carries the same request id — that is the link.
        for event in events:
            self.assertEqual(event["request_id"], rid)

    def test_suspension_keeps_the_request_open(self):
        rid = memory_store.record_request("book me a table", route="browser")
        memory_store.record_suspension(rid, "Which restaurant?", "cp-9")
        row = memory_store.get_work_request(rid)
        self.assertEqual(row["status"], "suspended")
        self.assertIsNone(row["closed_at"])
        suspensions = memory_store.work_events(request_id=rid,
                                               kind="work_suspension")
        self.assertEqual(len(suspensions), 1)
        self.assertIn("cp-9", suspensions[0]["detail"])

    def test_current_request_identity_is_per_call(self):
        rid = memory_store.begin_request("do the thing", route="task")
        self.assertEqual(memory_store.current_request_id(), rid)

    def test_record_task_outcome_links_and_keeps_artifacts(self):
        rid = memory_store.record_request("download the invoice",
                                          route="browser_agent")
        memory_store.record_result(
            rid, "completed", summary="Downloaded.",
            artifacts=[{"kind": "file", "path": r"C:\Users\me\invoice.pdf",
                        "source": "https://billing.test/inv/9"}])
        work = memory_store.retrieve_work(request_id=rid)
        paths = [a.get("path") for a in work["artifacts"]]
        self.assertIn(r"C:\Users\me\invoice.pdf", paths)
        self.assertTrue(any(a.get("url") or a.get("source")
                            for a in work["artifacts"]))


class RestartRetrievalTests(WorkEventTestCase):
    """Acceptance: after a RESTART a report follow-up finds path + sources."""

    def _seed_report(self):
        rid = memory_store.record_request("research the best tanks",
                                          route="research")
        memory_store.record_result(
            rid, "completed", summary="Report ready.",
            artifacts=[
                {"kind": "report", "path": r"C:\jarvis\reports\tanks.md",
                 "title": "best tanks"},
                {"kind": "source", "url": "https://a.test/tanks",
                 "title": "Tank weekly"},
                {"kind": "source", "url": "https://b.test/tanks",
                 "title": "Defense digest"},
            ])
        return rid

    def test_report_followup_retrieves_path_and_sources_after_restart(self):
        self._seed_report()
        # "Restart": drop the connection so the next read reopens the DB.
        memory_store._local.conn = None
        work = memory_store.retrieve_work("where is that report?")
        self.assertTrue(work["request"])
        self.assertEqual(work["request"]["status"], "completed")
        paths = [a.get("path") for a in work["artifacts"]]
        self.assertIn(r"C:\jarvis\reports\tanks.md", paths)
        urls = {s.get("url") for s in work["sources"]}
        self.assertEqual(urls, {"https://a.test/tanks", "https://b.test/tanks"})

    def test_work_context_is_bounded_and_has_the_location(self):
        self._seed_report()
        block = memory_store.work_context("show me that report again")
        self.assertIn(r"C:\jarvis\reports\tanks.md", block)
        self.assertIn("https://a.test/tanks", block)
        self.assertLessEqual(len(block.splitlines()), 10)

    def test_work_context_is_empty_for_a_new_request(self):
        self._seed_report()
        self.assertEqual(memory_store.work_context("open youtube"), "")
        self.assertEqual(memory_store.work_context(""), "")

    def test_looks_like_work_followup(self):
        for text in ("where is that report?", "show me the last task",
                     "what happened to that research", "send me its sources"):
            self.assertTrue(memory_store.looks_like_work_followup(text), text)
        for text in ("play some music", "what is the price of tea",
                     "open gmail"):
            self.assertFalse(memory_store.looks_like_work_followup(text), text)


class SecretRedactionTests(WorkEventTestCase):
    """Acceptance: no event field leaks seeded secrets."""

    def _all_text(self):
        rows = []
        try:
            conn = memory_store._conn()
            for table in ("events", "work_requests"):
                for row in conn.execute("SELECT * FROM %s" % table).fetchall():
                    rows.append(" | ".join(str(v) for v in tuple(row)))
        except Exception:
            pass
        return "\n".join(rows)

    def test_request_and_result_never_store_a_secret(self):
        rid = memory_store.record_request("use the key %s please" % SECRET)
        memory_store.record_result(
            rid, "completed",
            summary="done with %s" % SECRET,
            artifacts=[{"kind": "url",
                        "url": "https://x.test/?token=%s" % SECRET,
                        "title": "key %s" % SECRET}],
            evidence=["used %s" % SECRET])
        memory_store.record_suspension(rid, "which key, %s?" % SECRET, "cp-1")
        blob = self._all_text()
        self.assertNotIn(SECRET, blob)
        self.assertNotIn("F07SECRET", blob)
        # The record still exists and is retrievable (masked, not dropped).
        self.assertTrue(memory_store.get_work_request(rid))

    def test_artifacts_are_masked_field_by_field(self):
        rid = memory_store.record_request("fetch a page")
        memory_store.record_result(
            rid, "completed", summary="ok",
            artifacts=[{"kind": "source", "url": "https://a.test/x",
                        "title": "auth %s" % SECRET}])
        rows = memory_store.recent_work_requests(limit=5)
        blob = json.dumps(rows, default=str)
        self.assertNotIn(SECRET, blob)
        self.assertIn("https://a.test/x", blob)

    def test_request_ids_never_collide_in_one_tick(self):
        ids = {memory_store.new_request_id() for _ in range(200)}
        self.assertEqual(len(ids), 200)


class StructuredArtifactTests(WorkEventTestCase):
    """Clipping must not break the structure a follow-up needs."""

    def test_long_artifacts_stay_valid_structured_json(self):
        artifacts = [{"kind": "source", "url": "https://a.test/%d" % i,
                      "title": "T" * 900, "line": i} for i in range(40)]
        rid = memory_store.record_request("research lots")
        memory_store.record_result(rid, "completed", summary="s" * 5000,
                                   artifacts=artifacts)
        row = memory_store.get_work_request(rid)
        self.assertIsInstance(row["artifacts"], list)
        self.assertTrue(row["artifacts"])
        for item in row["artifacts"]:
            self.assertIsInstance(item, dict)
            self.assertIn("url", item)
            self.assertLessEqual(len(item["title"]), 300)
        self.assertLessEqual(len(row["artifacts"]), 24)

    def test_artifacts_merge_across_reports(self):
        rid = memory_store.record_request("compile the report")
        memory_store.record_result(rid, "partial",
                                   artifacts=[{"kind": "report",
                                               "path": r"C:\r\a.md"}])
        memory_store.record_result(rid, "completed",
                                   artifacts=[{"kind": "source",
                                               "url": "https://s.test/1"}])
        paths = {a.get("path") for a in memory_store.get_work_request(rid)["artifacts"]}
        urls = {a.get("url") for a in memory_store.get_work_request(rid)["artifacts"]}
        self.assertIn(r"C:\r\a.md", paths)
        self.assertIn("https://s.test/1", urls)

    def test_unknown_artifact_fields_are_dropped_not_serialised(self):
        rid = memory_store.record_request("x")
        memory_store.record_result(
            rid, "completed",
            artifacts=[{"kind": "page", "url": "https://a.test",
                        "dom": "X" * 5000, "cookies": SECRET}])
        blob = json.dumps(memory_store.get_work_request(rid), default=str)
        self.assertNotIn("dom", blob)
        self.assertNotIn(SECRET, blob)


class ProjectionTests(WorkEventTestCase):
    def test_chat_projection_is_derived_from_work_events(self):
        memory_store.record_event("chat_exchange", "user: hello",
                                  request_id="r1")
        rid = memory_store.record_request("find the report")
        memory_store.record_result(rid, "completed", summary="Found it.",
                                   artifacts=[{"kind": "report",
                                               "path": r"C:\r\b.md"}])
        messages = memory_store.chat_projection(limit=10)
        self.assertTrue(messages)
        self.assertEqual({m["role"] for m in messages} & {"user", "assistant"},
                         {"user", "assistant"})
        self.assertTrue(any(m.get("request_id") for m in messages))

    def test_episodic_projection_only_returns_terminal_work(self):
        open_id = memory_store.record_request("still running")
        memory_store.record_suspension(open_id, "which one?")
        done_id = memory_store.record_request("finished work")
        memory_store.record_result(done_id, "completed", summary="done",
                                   artifacts=[{"kind": "report",
                                               "path": r"C:\r\c.md"}])
        episodes = memory_store.episodic_projection(limit=10)
        ids = {e["request_id"] for e in episodes}
        self.assertIn(done_id, ids)
        self.assertNotIn(open_id, ids)
        episode = [e for e in episodes if e["request_id"] == done_id][0]
        self.assertEqual(episode["artifacts"][0]["path"], r"C:\r\c.md")
        self.assertNotIn(SECRET, json.dumps(episodes, default=str))


class BrainWiringTests(unittest.TestCase):
    def test_handle_chat_identifies_the_request(self):
        import inspect
        from backend.core import brain

        code = inspect.getsource(brain.handle_chat)
        self.assertIn("begin_request", code)
        self.assertIn("record_result", code)

    def test_chat_prompt_includes_work_context(self):
        import inspect
        from backend.core import brain

        # The F07 work-context block is injected into the chat system prompt.
        # [PERF] The two memory-context reads are memoised by
        # _memory_context_cached (they run twice per turn: once in the
        # speculative racer, once in the selected route), so the read itself
        # lives there and the build consumes its result. Both halves are
        # asserted: the read, and the build actually appending the block.
        reader = inspect.getsource(brain._memory_context_cached)
        self.assertIn("work_context", reader)
        self.assertIn("memory_context", reader)

        code = inspect.getsource(brain._build_chat_messages)
        self.assertIn("work_block", code)
        self.assertIn("system_prompt = system_prompt", code)

    def test_research_records_the_report_path(self):
        import inspect
        from backend.core import brain

        code = inspect.getsource(brain.handle_research_intent)
        self.assertIn("report_path", code)
        self.assertIn("record_result", code)
        self.assertIn("kind", code)

    def test_browser_task_links_suspension_and_artifacts(self):
        import inspect
        from backend.core import brain

        code = inspect.getsource(brain._execute_deferred_opencode)
        self.assertIn("record_suspension", code)
        self.assertIn("_task_result_artifacts", code)
        self.assertIn("work_request_id", code)

    def test_task_result_artifacts_extracts_paths_and_urls(self):
        from backend.core import brain
        from backend.services.task_result import TaskResult

        result = TaskResult.completed("done")
        result.evidence = [r"Saved to C:\Users\me\report.md",
                           "see https://example.test/a?b=1 for details"]
        artifacts = brain._task_result_artifacts(result)
        paths = [a.get("path") for a in artifacts]
        urls = [a.get("url") for a in artifacts]
        self.assertIn(r"C:\Users\me\report.md", paths)
        self.assertIn("https://example.test/a?b=1", urls)


if __name__ == "__main__":
    unittest.main()
