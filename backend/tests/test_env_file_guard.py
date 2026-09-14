"""The ``.env`` safety net itself has to work.

``backend/tests/__init__.py`` snapshots the developer's real ``.env`` when the
test package is imported and restores it at interpreter exit if a test deleted
or rewrote it. That net only helps if it actually restores a removed file and
leaves a healthy one completely alone, so both halves are pinned here.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import tests as tests_package

_CONTENT = b"FISH_API_KEY=sk-fish-not-a-real-key\n"


class EnvGuardTests(unittest.TestCase):
    def _guard(self, directory, snapshot=_CONTENT):
        target = Path(directory) / ".env"
        patcher_path = patch.object(tests_package, "_ENV_PATH", target)
        patcher_snapshot = patch.object(tests_package, "_SNAPSHOT", snapshot)
        patcher_path.start()
        patcher_snapshot.start()
        self.addCleanup(patcher_snapshot.stop)
        self.addCleanup(patcher_path.stop)
        return target

    def test_a_deleted_env_is_restored_from_the_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = self._guard(tmp)
            target.write_bytes(_CONTENT)
            target.unlink()
            self.assertFalse(target.exists())

            tests_package._restore_env()

            self.assertEqual(target.read_bytes(), _CONTENT)

    def test_a_rewritten_env_is_restored_from_the_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = self._guard(tmp)
            target.write_bytes(b"")  # the ambiguous "truncated .env" case
            tests_package._restore_env()
            self.assertEqual(target.read_bytes(), _CONTENT)

    def test_a_healthy_env_is_never_written_to(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = self._guard(tmp)
            target.write_bytes(_CONTENT)
            with patch.object(Path, "write_bytes",
                              side_effect=AssertionError("guard wrote to .env")):
                tests_package._restore_env()  # must be a no-op
            self.assertEqual(target.read_bytes(), _CONTENT)

    def test_the_guard_is_idempotent(self):
        before = tests_package._SNAPSHOT
        tests_package._install_guard()
        self.assertIs(tests_package._SNAPSHOT, before)

    def test_a_missing_file_is_not_recreated_out_of_nothing(self):
        # A session that started with no .env must not conjure one at exit.
        with tempfile.TemporaryDirectory() as tmp:
            target = self._guard(tmp, snapshot=None)
            self.assertFalse(target.exists())
            tests_package._restore_env()
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
