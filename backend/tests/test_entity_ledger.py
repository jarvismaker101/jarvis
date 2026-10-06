import os
import tempfile
import unittest

from backend.core import brain
from backend.core import entity_ledger as ledger
from backend.services import code_tools


def _tmp_files(tmp, *names):
    """Real files on disk (liveness counts) + ledger entities for them."""
    paths = []
    for name in names:
        path = os.path.join(tmp, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("x")
        ledger.record_entity(name, path, kind="file", source="created")
        paths.append(path)
    return paths


def _boost(entity_id, times=4):
    for _ in range(times):
        ledger.mention_entity(entity_id)


class LedgerRecordTests(unittest.TestCase):
    """Rank 1: the ledger upserts, orders focus, and caps size."""

    def setUp(self):
        ledger.reset()

    def tearDown(self):
        ledger.reset()

    def test_upsert_dedupes_by_path_and_bumps_version(self):
        first = ledger.record_entity("a.txt", "C:\\d\\a.txt", kind="file")
        second = ledger.record_entity("a.txt", "C:\\d\\a.txt", kind="file")
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(len(ledger.snapshot()["entities"]), 1)
        self.assertEqual(ledger.snapshot()["entities"][0]["version"], 2)

    def test_focus_head_is_most_recent_of_kind(self):
        ledger.record_entity("old", "C:\\d\\old", kind="folder")
        ledger.record_entity("new", "C:\\d\\new", kind="folder")
        self.assertEqual(ledger.focus_head("folder")["display_name"], "new")

    def test_mention_makes_focus_head(self):
        one = ledger.record_entity("one.txt", "C:\\d\\one.txt", kind="file")
        ledger.record_entity("two.txt", "C:\\d\\two.txt", kind="file")
        ledger.mention_entity(one["id"])
        self.assertEqual(ledger.focus_head("file")["id"], one["id"])

    def test_notebook_mirror_lands_in_ledger(self):
        brain._notebook_entities[:] = []
        self.addCleanup(brain._notebook_entities.clear)
        brain.notebook_record_entity("Mayank Malik", "C:\\d\\Mayank Malik", kind="folder")
        head = ledger.focus_head("folder")
        self.assertIsNotNone(head)
        self.assertEqual(head["canon"], os.path.normpath("C:\\d\\Mayank Malik"))

    def test_write_file_records_created_entity(self):
        self.addCleanup(brain._notebook_entities.clear)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "hello.txt")
            result = code_tools.write_file(path, "hello jarvis")
            self.assertTrue(result["ok"], result)
            head = ledger.focus_head("file")
            self.assertIsNotNone(head)
            self.assertEqual(head["canon"], os.path.normpath(path))
            self.assertEqual(head["provenance"], "created")


class ResolverTests(unittest.TestCase):
    """Rank 1: mentions bind by scoring — never the nearest noun."""

    def setUp(self):
        ledger.reset()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def tearDown(self):
        ledger.reset()

    def test_explicit_quoted_name_beats_recency(self):
        _tmp_files(self._tmp.name, "old.txt", "new.txt")
        status, ent = ledger.resolve_mention("open 'old.txt' please")
        self.assertEqual(status, "bound")
        self.assertIn("old.txt", ent["display_name"])

    def test_the_file_you_just_made_binds_newest_created(self):
        _tmp_files(self._tmp.name, "first.txt", "second.txt")
        head = ledger.focus_head("file")
        _boost(head["id"])
        status, ent = ledger.resolve_mention("what was the file you just made")
        self.assertEqual(status, "bound")
        self.assertIn("second.txt", ent["display_name"])

    def test_bare_it_uses_expected_kind(self):
        ledger.record_entity("Mayank Malik", "C:\\d\\Mayank Malik", kind="folder")
        _tmp_files(self._tmp.name, "note.txt")
        status, ent = ledger.resolve_mention("delete it",
                                             expected_kind="file")
        self.assertEqual(status, "bound")
        self.assertIn("note.txt", ent["display_name"])

    def test_close_tie_asks_once_naming_both(self):
        _tmp_files(self._tmp.name, "a.txt", "b.txt")
        status, payload = ledger.resolve_mention("delete it",
                                                 expected_kind="file")
        self.assertEqual(status, "ask")
        self.assertIn("a.txt", payload["question"])
        self.assertIn("b.txt", payload["question"])
        self.assertEqual(len(payload["options"]), 2)

    def test_rejected_entity_is_never_offered_again(self):
        _tmp_files(self._tmp.name, "a.txt", "b.txt")
        head = ledger.focus_head("file")
        _boost(head["id"])
        status, ent = ledger.resolve_mention("delete it",
                                             expected_kind="file")
        self.assertEqual(status, "bound")
        binding = ledger.get_last_binding()
        ledger.reject_entity(binding["entity_id"])
        status2, ent2 = ledger.resolve_mention(
            binding["text"], expected_kind="file")
        self.assertEqual(status2, "bound")
        self.assertNotEqual(ent2["id"], ent["id"])

    def test_no_mention_returns_none(self):
        self.assertEqual(ledger.resolve_mention("what is the weather"),
                         ("none", ""))

    def test_mention_without_candidates_asks(self):
        status, payload = ledger.resolve_mention("delete it",
                                                 expected_kind="file")
        self.assertEqual(status, "ask")
        self.assertEqual(payload["options"], [])


class ReferenceCorrectionTests(unittest.TestCase):
    """Rank 1: "no, not that one" revises the binding, never guesses."""

    def setUp(self):
        ledger.reset()
        brain._notebook_entities[:] = []
        brain._notebook_requests[:] = []
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def tearDown(self):
        ledger.reset()
        brain._notebook_entities[:] = []
        brain._notebook_requests[:] = []

    def _bind_first_of_two(self):
        _tmp_files(self._tmp.name, "a.txt", "b.txt")
        head = ledger.focus_head("file")
        _boost(head["id"])
        status, ent = ledger.resolve_mention("delete it",
                                             expected_kind="file")
        self.assertEqual(status, "bound")
        return ent

    def test_not_that_one_switches_to_other(self):
        first = self._bind_first_of_two()
        reply = brain.handle_reference_correction("no, not that one")
        self.assertIsNotNone(reply)
        self.assertIn("instead", reply)
        self.assertNotIn(first["display_name"], reply)

    def test_nothing_left_to_offer_asks_for_name(self):
        _tmp_files(self._tmp.name, "only.txt")
        ledger.resolve_mention("delete it", expected_kind="file")
        reply = brain.handle_reference_correction("i meant the other one")
        self.assertIsNotNone(reply)
        self.assertIn("exact name", reply)

    def test_without_binding_falls_through(self):
        self.assertIsNone(brain.handle_reference_correction("no, not that one"))

    def test_unrelated_text_falls_through(self):
        self._bind_first_of_two()
        self.assertIsNone(brain.handle_reference_correction("what is the weather"))

    def test_r7_phrases_stay_on_r7_path(self):
        self._bind_first_of_two()
        self.assertIsNone(brain.handle_reference_correction("no, i meant check it"))


if __name__ == "__main__":
    unittest.main()
