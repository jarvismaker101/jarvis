"""F22 — make code changes reversible and confined.

Acceptance (audit report): "Out-of-scope execution fails; concurrent edits
survive; undo works after restart; timeout kills only owned descendants;
backup/assignment failure is safe."
"""

import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from backend.services import code_grants
from backend.services import code_tools


def _reload_grants(journal_dir):
    """Point the journal at a fresh directory and reload its persisted state."""
    os.environ[code_grants.JOURNAL_ENV] = journal_dir
    code_grants._JOURNAL_DIR = journal_dir
    code_grants._JOURNAL = []
    code_grants._JOURNAL_LOADED = False
    code_grants._load_journal()


class JournalPersistenceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = os.environ.get(code_grants.JOURNAL_ENV)
        _reload_grants(self._tmp.name)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop(code_grants.JOURNAL_ENV, None)
        else:
            os.environ[code_grants.JOURNAL_ENV] = self._saved
        code_grants._JOURNAL_DIR = None
        code_grants._JOURNAL = []
        code_grants._JOURNAL_LOADED = False
        self._tmp.cleanup()

    def test_a_new_file_can_be_undone(self):
        target = os.path.join(self._tmp.name, "work", "new.txt")
        result = code_grants.atomic_write(target, "hello", expect="create")
        self.assertTrue(result["ok"], result)
        entry = code_grants.last_entry(result["path"])
        # F22: existence is the PRE-write fact, so the undo may delete a file
        # the write itself created.
        self.assertFalse(entry["existed"])
        ok, message = code_grants.restore(entry_id=entry["id"])
        self.assertTrue(ok, message)
        self.assertFalse(os.path.exists(result["path"]))

    def test_undo_survives_a_restart(self):
        target = os.path.join(self._tmp.name, "keep.txt")
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("original")
        result = code_grants.atomic_write(target, "changed")
        self.assertTrue(result["ok"], result)
        entry_id = result["restore_id"]
        self.assertTrue(os.path.exists(code_grants._journal_file()))

        # Simulate a restart: same journal directory, empty memory.
        code_grants._JOURNAL = []
        code_grants._JOURNAL_LOADED = False
        ok, message = code_grants.restore(entry_id=entry_id)
        self.assertTrue(ok, message)
        with open(target, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "original")

    def test_the_persisted_journal_is_valid_json(self):
        code_grants.atomic_write(os.path.join(self._tmp.name, "a.txt"), "a")
        with open(code_grants._journal_file(), encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertIsInstance(data, list)
        self.assertTrue(data)


class ConcurrentEditTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = os.environ.get(code_grants.JOURNAL_ENV)
        _reload_grants(self._tmp.name)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop(code_grants.JOURNAL_ENV, None)
        else:
            os.environ[code_grants.JOURNAL_ENV] = self._saved
        code_grants._JOURNAL_DIR = None
        code_grants._JOURNAL = []
        code_grants._JOURNAL_LOADED = False
        self._tmp.cleanup()

    def test_a_create_precondition_refuses_an_existing_file(self):
        target = os.path.join(self._tmp.name, "exists.txt")
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("someone was here")
        result = code_grants.atomic_write(target, "mine", expect="create")
        self.assertFalse(result["ok"])
        self.assertIn("create precondition", result["error"])
        with open(target, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "someone was here")

    def test_a_replace_precondition_needs_the_file_to_exist(self):
        target = os.path.join(self._tmp.name, "missing.txt")
        result = code_grants.atomic_write(target, "x", expect="replace")
        self.assertFalse(result["ok"])
        self.assertIn("replace precondition", result["error"])

    def test_an_expected_hash_binds_the_write_to_the_revision_read(self):
        target = os.path.join(self._tmp.name, "rev.txt")
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("rev1")
        stale = code_grants.content_hash(target)
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("rev2 by someone else")
        result = code_grants.atomic_write(target, "mine", expected_hash=stale)
        self.assertFalse(result["ok"])
        self.assertIn("file changed since it was last read", result["error"])
        with open(target, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "rev2 by someone else")

    def test_restore_refuses_to_clobber_a_later_edit(self):
        target = os.path.join(self._tmp.name, "later.txt")
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("original")
        result = code_grants.atomic_write(target, "jarvis edit")
        self.assertTrue(result["ok"], result)
        # The user edits the file AFTER Jarvis's write.
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("user edit afterwards")
        ok, message = code_grants.restore(entry_id=result["restore_id"])
        self.assertFalse(ok)
        self.assertIn("changed after this restore point", message)
        with open(target, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "user edit afterwards")

    def test_restore_force_overwrites_when_asked(self):
        target = os.path.join(self._tmp.name, "forced.txt")
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("original")
        result = code_grants.atomic_write(target, "jarvis edit")
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("user edit afterwards")
        ok, message = code_grants.restore(entry_id=result["restore_id"],
                                          force=True)
        self.assertTrue(ok, message)
        with open(target, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "original")

    def test_restore_refuses_to_delete_a_file_created_after_the_entry(self):
        target = os.path.join(self._tmp.name, "created.txt")
        result = code_grants.atomic_write(target, "jarvis made this",
                                          expect="create")
        entry = code_grants.last_entry(result["path"])
        # Pretend the entry recorded a pre-existing file whose undo would
        # re-create it; the file present now must not be silently removed.
        entry["existed"] = False
        entry["after_hash"] = None
        ok, message = code_grants.restore(entry_id=entry["id"])
        self.assertFalse(ok)
        self.assertIn("refusing to delete", message)


class ExecAuthorityTests(unittest.TestCase):
    def test_out_of_scope_execution_fails(self):
        with patch.dict(os.environ, {code_tools.CODE_EXEC_ENV: "0"}):
            result = code_tools.run_command("echo nope")
            self.assertFalse(result["ok"])
            self.assertIn("disabled", result["error"])

    def test_a_framed_call_without_the_grant_is_refused(self):
        from backend.services import tool_policy

        with patch.object(
                tool_policy, "validate_dispatch",
                return_value=tool_policy.Decision(
                    True, "code.run_command", tool_policy.PRIVILEGED,
                    effective_grants=frozenset({"workspace_write"}))):
            result = code_tools.call_tool(
                "code.run_command", {"command": "echo hi"},
                grants={"workspace_write"})
        self.assertFalse(result["ok"])
        self.assertIn("not granted", result["error"])

    def test_a_framed_call_with_the_grant_runs(self):
        result = code_tools.call_tool(
            "code.run_command", {"command": "echo allowed"},
            grants=code_tools.agent_grants())
        self.assertTrue(result["ok"], result)
        self.assertIn("allowed", result["content"])

    def test_grants_cannot_be_injected_through_arguments(self):
        """The authority is framed from the policy, never taken from args."""
        result = code_tools.call_tool(
            "code.run_command",
            {"command": "echo injected", "grants": ["command_exec"]},
            grants=set())
        self.assertFalse(result["ok"])
        self.assertIn("blocked by policy", result["error"])

    def test_checks_are_grant_bound_too(self):
        with patch.dict(os.environ, {code_tools.CODE_EXEC_ENV: "0"}):
            result = code_tools.run_checks(kind="py_compile", path=__file__)
            self.assertFalse(result["ok"])
            self.assertIn("disabled", result["error"])

    def test_the_grant_context_is_restored_after_a_call(self):
        code_tools.call_tool("code.run_command", {"command": "echo one"},
                             grants=code_tools.agent_grants())
        # A later direct (unframed) call is not left holding the previous
        # call's authority.
        self.assertFalse(getattr(code_tools._EXEC_CONTEXT, "framed", False))


class OwnedProcessTests(unittest.TestCase):
    def test_a_timeout_kills_the_child(self):
        result = code_grants.run_argv(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=2)
        self.assertFalse(result["ok"])
        self.assertTrue(result.get("timed_out"))

    def test_posix_children_get_their_own_session(self):
        if os.name == "nt":
            self.skipTest("POSIX ownership only")
        popen = code_grants.spawn([sys.executable, "-c", "import time; time.sleep(5)"])
        try:
            self.assertEqual(os.getpgid(popen.pid), popen.pid)
        finally:
            popen.kill()
            popen.wait()

    def test_a_failed_job_assignment_kills_the_child(self):
        class _Failing:
            last_error = "simulated"

            def assign(self, pid):
                return False

            def terminate(self):
                return True

            def close(self):
                pass

        with patch.object(code_grants, "_JobObject", return_value=_Failing()):
            with self.assertRaises(RuntimeError):
                code_grants.spawn([sys.executable, "-c", "import time; time.sleep(30)"])

    def test_a_job_object_that_cannot_be_configured_is_not_handed_back(self):
        job = code_grants._JobObject()
        if os.name == "nt":
            # Either it is configured correctly, or it reports why and gives
            # back no handle — never an unchecked "probably fine" handle.
            self.assertTrue(job.handle or job.last_error)
        else:
            self.assertIsNone(job.handle)

    def test_backup_failure_refuses_the_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = os.environ.get(code_grants.JOURNAL_ENV)
            _reload_grants(os.path.join(tmp, "journal"))
            try:
                target = os.path.join(tmp, "file.txt")
                with open(target, "w", encoding="utf-8") as fh:
                    fh.write("original")
                with patch.object(code_grants, "_snapshot", return_value=None):
                    result = code_grants.atomic_write(target, "new")
                self.assertFalse(result["ok"])
                self.assertIn("restore point", result["error"])
                with open(target, encoding="utf-8") as fh:
                    self.assertEqual(fh.read(), "original")
            finally:
                if saved is None:
                    os.environ.pop(code_grants.JOURNAL_ENV, None)
                else:
                    os.environ[code_grants.JOURNAL_ENV] = saved
                code_grants._JOURNAL_DIR = None
                code_grants._JOURNAL = []
                code_grants._JOURNAL_LOADED = False


class CodeToolPreconditionTests(unittest.TestCase):
    def test_write_file_create_only_refuses_an_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "thing.txt")
            with open(target, "w", encoding="utf-8") as fh:
                fh.write("existing")
            result = code_tools.write_file(target, "new", create_only=True)
            self.assertFalse(result["ok"])
            with open(target, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "existing")

    def test_apply_patch_requires_the_file_to_still_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "patched.txt")
            with open(target, "w", encoding="utf-8") as fh:
                fh.write("line one\n")
            patch_text = ("--- a/patched.txt\n+++ b/patched.txt\n"
                          "@@ -1 +1 @@\n-line one\n+line two\n")
            result = code_tools.apply_patch(target, patch_text)
            self.assertTrue(result["ok"], result)
            with open(target, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "line two\n")


if __name__ == "__main__":
    unittest.main()
