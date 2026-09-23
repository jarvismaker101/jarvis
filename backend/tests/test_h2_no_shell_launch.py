"""H2 (2026-09-23 audit) — launch_app must never shell out arbitrary words.

The old tail ran ``subprocess.Popen(name, shell=True)`` for any single
alphanumeric token, so "powershell", "cmd", "regedit", "format" all became
silent shell commands from the backend process.
"""
import unittest
from unittest.mock import patch

from backend.core import executor


class NoShellLaunchTests(unittest.TestCase):
    def test_unknown_app_fails_closed_without_shell(self):
        with patch.object(executor.glob, "glob", return_value=[]), \
             patch.object(executor.os.path, "exists", return_value=False), \
             patch.object(executor.subprocess, "Popen") as popen, \
             patch.object(executor.os, "startfile", create=True) as startfile:
            self.assertFalse(executor.launch_app("definitelynotanapp"))
        popen.assert_not_called()
        startfile.assert_not_called()

    def test_non_allowlisted_alnum_words_never_reach_the_shell(self):
        # The explicit system_apps allowlist ("cmd" -> "cmd.exe", fixed
        # constants, no shell) is intentional and kept; the H2 hole was the
        # alnum TAIL that shelled out anything else. These tokens match no
        # allowlist yet pass the old isalnum gate.
        for word in ("asdfqwe", "notepadx", "winwordd", "maliciousname"):
            with patch.object(executor.glob, "glob", return_value=[]), \
                 patch.object(executor.os.path, "exists", return_value=False), \
                 patch.object(executor.subprocess, "Popen") as popen, \
                 patch.object(executor.os, "startfile", create=True):
                self.assertFalse(executor.launch_app(word))
            popen.assert_not_called()

    def test_vscode_path_fallback_uses_argv_not_shell(self):
        with patch.object(executor.os.path, "exists", return_value=False), \
             patch.object(executor.shutil, "which",
                          return_value="C:\\bin\\code.exe"), \
             patch.object(executor.subprocess, "Popen") as popen, \
             patch.object(executor.os, "startfile", create=True):
            self.assertTrue(executor.launch_app("vscode"))
        popen.assert_called_once()
        args = popen.call_args[0][0]
        self.assertIsInstance(args, list)
        self.assertEqual(args, ["C:\\bin\\code.exe"])


if __name__ == "__main__":
    unittest.main()
