"""F54 — boot must not wait on a sick task-engine daemon, and must not lie.

Two defects found while investigating "boot takes several seconds longer than
it used to, and browser tasks fail with a connection error on port 9570":

1. ``ensure_brave_mcp_daemon`` polled the port for its whole 10s timeout even
   after the child had already died (measured 10.42s with the vendored copy
   missing its ``node_modules``), and that wait sat directly in front of the
   visible launch — hence the gap between the activity console and the next
   window.
2. It ended with ``return _BRAVE_MCP_PID is not None``, so a crashed child was
   reported as a started daemon. The browser agent then skipped its own
   "daemon could not be started" branch and failed later with a raw
   ``HTTPConnectionPool ... Failed to establish a new connection`` instead.

These tests pin the corrected contract: the answer is the truth about the
port, a dead child is detected immediately, the child's output is captured so
a failure explains itself, and the watcher warms the daemon off the boot path.
"""

import os
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from backend.services import opencode_client


def _fake_proc(pid=31337, alive=True, returncode=None):
    proc = Mock()
    proc.pid = pid
    proc.poll.return_value = None if alive else (returncode or 1)
    proc.returncode = returncode
    return proc


def _never_listening(*_args, **_kwargs):
    """``_pids_on_port`` stand-in: the port never opens."""
    return set()


def _rendered(warn):
    """The logged warnings as the reader would see them (args interpolated).

    ``str(call)`` shows the repr of the argument tuple, which escapes the
    backslashes of a Windows path and makes a path assertion meaningless.
    """
    messages = []
    for call in warn.call_args_list:
        if call.args:
            try:
                messages.append(str(call.args[0]) % tuple(call.args[1:]))
            except (TypeError, ValueError):
                messages.append(str(call.args))
        else:
            messages.append(str(call))
    return " ".join(messages)


class _DaemonTestBase(unittest.TestCase):
    """Hermetic seams for the daemon starter: no real spawn, private log."""

    def setUp(self):
        self._saved_pid = opencode_client._BRAVE_MCP_PID
        opencode_client._BRAVE_MCP_PID = None
        self._log_dir = tempfile.mkdtemp(prefix="f54-brave-mcp-")
        self._log_path = os.path.join(self._log_dir, "brave_mcp.log")

    def tearDown(self):
        opencode_client._BRAVE_MCP_PID = self._saved_pid

    def _run_starter(self, proc, ports):
        """Run the starter with every seam patched; return (ok, popen_mock)."""
        with patch.object(opencode_client, "_pids_on_port",
                          side_effect=ports), \
             patch.object(opencode_client, "_brave_mcp_log_path",
                          return_value=self._log_path), \
             patch.object(opencode_client.shutil, "which",
                          return_value=r"C:\node\node.exe"), \
             patch.object(opencode_client.subprocess, "Popen",
                          return_value=proc) as popen:
            ok = opencode_client.ensure_brave_mcp_daemon()
        return ok, popen


class DeadDaemonFailsFastTests(_DaemonTestBase):
    def test_dead_child_is_a_failure_not_a_success(self):
        """A child that already exited can never serve the port (bug #2)."""
        proc = _fake_proc(alive=False, returncode=1)

        with patch.object(opencode_client.logging, "warning") as warn:
            started = time.monotonic()
            ok, _popen = self._run_starter(proc, _never_listening)
            elapsed = time.monotonic() - started

        self.assertFalse(ok)
        self.assertLess(
            elapsed, 2.0,
            "a dead child must not be waited out for the whole readiness timeout")
        self.assertIsNone(
            opencode_client._BRAVE_MCP_PID,
            "a dead pid must not stay tracked: Windows recycles pids, and a "
            "later teardown would then kill a foreign process")
        logged = _rendered(warn)
        self.assertIn("exited immediately", logged)
        self.assertIn(self._log_path, logged,
                      "the failure must point at the captured daemon output")

    def test_live_child_that_never_binds_is_still_a_failure(self):
        """The old code returned True here purely because Popen returned."""
        proc = _fake_proc(alive=True)

        with patch.object(opencode_client, "BRAVE_MCP_READY_TIMEOUT", 0.5), \
             patch.object(opencode_client.logging, "warning") as warn:
            ok, _popen = self._run_starter(proc, _never_listening)

        self.assertFalse(ok, "readiness is verified, never assumed")
        logged = _rendered(warn)
        self.assertIn("did not become ready", logged)

    def test_child_output_is_captured_to_the_daemon_log(self):
        """A hidden child's stderr must land somewhere diagnosable (bug #1)."""
        proc = _fake_proc(alive=True)

        ok, popen = self._run_starter(proc, [set(), {31337}])

        self.assertTrue(ok)
        kwargs = popen.call_args.kwargs
        self.assertIs(kwargs["stderr"], opencode_client.subprocess.STDOUT)
        self.assertNotEqual(kwargs["stdout"], opencode_client.subprocess.DEVNULL)
        with open(self._log_path, encoding="utf-8") as handle:
            self.assertIn("starting brave-MCP daemon", handle.read())

    def test_already_listening_daemon_is_reused_without_a_spawn(self):
        ok, popen = self._run_starter(_fake_proc(), [{999}])

        self.assertTrue(ok)
        popen.assert_not_called()


class BootPathTests(unittest.TestCase):
    """The watcher's boot path must not block on the daemon's readiness."""

    def test_warm_up_runs_off_the_calling_thread(self):
        from backend import watcher

        seen = {}
        released = threading.Event()

        def slow_starter():
            seen["thread"] = threading.current_thread().name
            seen["started"] = True
            released.wait(timeout=5)

        with patch("backend.services.opencode_client.ensure_brave_mcp_daemon",
                   slow_starter):
            started = time.monotonic()
            thread = watcher.warm_browser_task_engine()
            elapsed = time.monotonic() - started
            # The worker is parked inside the starter while the caller has
            # already returned, so boot can go on to spawn the visible windows.
            self.assertTrue(seen.get("started"), "the warm-up must actually run")
            self.assertTrue(thread.is_alive())
            released.set()
            thread.join(timeout=5)

        self.assertLess(elapsed, 0.5,
                        "warming the daemon must not block the boot path")
        self.assertEqual(seen["thread"], "brave-mcp-warmup")
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()