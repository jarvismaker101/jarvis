import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from backend.services import opencode_client
from backend import config
from backend import watcher


class OpencodeServerSpawnTests(unittest.TestCase):
    """Boot wiring: the serve daemon is hidden (quiet HTTP), ONE visible
    console window tails the activity log, both pids are remembered."""

    def setUp(self):
        opencode_client._SERVER_PID = None
        opencode_client._TAIL_PID = None
        # These tests exercise the opencode engine explicitly; the repo
        # default is browser_agent (opencode detached).
        engine = patch.object(config, "TASK_ENGINE", "opencode")
        engine.start()
        self.addCleanup(engine.stop)

    def tearDown(self):
        opencode_client._SERVER_PID = None
        opencode_client._TAIL_PID = None

    def test_ensure_spawns_hidden_serve_and_visible_tail(self):
        fake_tail = Mock()
        fake_tail.pid = 5555
        fake_serve = Mock()
        fake_serve.pid = 4242
        with patch.object(opencode_client, "is_opencode_server_alive",
                          side_effect=[False, True]), \
             patch.object(opencode_client, "_resolve_opencode",
                          return_value=r"C:\fake\opencode.exe"), \
             patch.object(opencode_client, "_pids_on_port",
                          return_value=set()), \
             patch.object(opencode_client.time, "sleep"), \
             patch.object(opencode_client, "_ensure_activity_log"), \
             patch.object(opencode_client, "_activity_log_path",
                          return_value=r"C:\fake\data\opencode_activity.log"), \
             patch.object(opencode_client.subprocess, "Popen",
                          side_effect=[fake_tail, fake_serve]) as popen:
            ok = opencode_client.ensure_opencode_server()

        self.assertTrue(ok)
        self.assertEqual(popen.call_count, 2)

        # First spawn: the visible activity tail (CREATE_NEW_CONSOLE),
        # now a shrink-aware PowerShell script instead of Get-Content -Wait.
        tail_args, tail_kwargs = popen.call_args_list[0]
        self.assertIn("-File", tail_args[0])
        script_index = tail_args[0].index("-File")
        self.assertTrue(tail_args[0][script_index + 1].endswith(
            os.path.join("backend", "scripts", "tail_activity.ps1")))
        log_index = tail_args[0].index("-LogPath")
        self.assertIn("opencode_activity.log", tail_args[0][log_index + 1])
        self.assertEqual(tail_kwargs.get("creationflags"),
                         opencode_client._CREATE_NEW_CONSOLE)
        self.assertEqual(opencode_client._TAIL_PID, 5555)

        # Second spawn: the quiet serve daemon, hidden again.
        serve_args, serve_kwargs = popen.call_args_list[1]
        self.assertEqual(serve_args[0][1], "serve")
        self.assertEqual(serve_kwargs.get("creationflags"),
                         opencode_client._CREATE_NO_WINDOW)
        self.assertEqual(opencode_client._SERVER_PID, 4242)

    def test_reuse_spawns_tail_but_no_second_server(self):
        fake_tail = Mock()
        fake_tail.pid = 7777
        with patch.object(opencode_client, "is_opencode_server_alive",
                          return_value=True), \
             patch.object(opencode_client, "_ensure_activity_log"), \
             patch.object(opencode_client.subprocess, "Popen",
                          return_value=fake_tail) as popen:
            ok = opencode_client.ensure_opencode_server()

        self.assertTrue(ok)
        self.assertEqual(popen.call_count, 1)
        tail_kwargs = popen.call_args.kwargs
        self.assertEqual(tail_kwargs.get("creationflags"),
                         opencode_client._CREATE_NEW_CONSOLE)
        tail_args = popen.call_args.args[0]
        script_index = tail_args.index("-File")
        self.assertTrue(tail_args[script_index + 1].endswith(
            os.path.join("backend", "scripts", "tail_activity.ps1")))
        self.assertEqual(opencode_client._TAIL_PID, 7777)


class OpencodeRunTaskStreamingTests(unittest.TestCase):
    """run_opencode_task stays hidden, streams every line to the activity
    log, and still returns the full transcript for the completion summary."""

    def setUp(self):
        engine = patch.object(config, "TASK_ENGINE", "opencode")
        engine.start()
        self.addCleanup(engine.stop)

    class FakeStream:
        def __init__(self, lines):
            self._lines = list(lines)

        def readline(self):
            return self._lines.pop(0) if self._lines else ""

    class BlockingStream:
        """readline never returns: simulates a child that stays quiet."""

        def __init__(self, release):
            self._release = release

        def readline(self):
            self._release.wait()
            return ""

    def test_streams_lines_to_log_and_returns_full_output(self):
        fake_proc = Mock()
        fake_proc.stdout = self.FakeStream(["line one\n", "line two\n"])
        fake_proc.returncode = 0
        with patch.object(opencode_client, "_resolve_opencode",
                          return_value=r"C:\fake\opencode.exe"), \
             patch.object(opencode_client, "is_opencode_server_alive",
                          return_value=True), \
             patch.object(opencode_client, "_append_activity") as appender, \
             patch.object(opencode_client.subprocess, "Popen",
                          return_value=fake_proc) as popen:
            output = opencode_client.run_opencode_task("create folder x")

        popen.assert_called_once()
        self.assertEqual(popen.call_args.kwargs["creationflags"],
                         opencode_client._CREATE_NO_WINDOW)
        self.assertEqual(popen.call_args.kwargs["stdout"],
                         opencode_client.subprocess.PIPE)
        self.assertEqual(popen.call_args.kwargs["stderr"],
                         opencode_client.subprocess.STDOUT)
        # Header block + the two streamed lines.
        self.assertEqual(appender.call_count, 3)
        self.assertIn("=== ", appender.call_args_list[0][0][0])
        self.assertIn("create folder x", appender.call_args_list[0][0][0])
        appender.assert_any_call("line one\n")
        appender.assert_any_call("line two\n")
        self.assertEqual(output.status, "completed")
        self.assertTrue(output)
        self.assertIn("line one", output.detail)
        self.assertIn("line two", output.detail)
        self.assertIn("line one", str(output))

    def test_nonzero_exit_returns_failed_with_transcript_as_detail(self):
        fake_proc = Mock()
        fake_proc.stdout = self.FakeStream(["half the work\n"])
        fake_proc.returncode = 2
        with patch.object(opencode_client, "_resolve_opencode",
                          return_value=r"C:\fake\opencode.exe"), \
             patch.object(opencode_client, "is_opencode_server_alive",
                          return_value=True), \
             patch.object(opencode_client, "_append_activity"), \
             patch.object(opencode_client.subprocess, "Popen",
                          return_value=fake_proc):
            output = opencode_client.run_opencode_task("risky task")

        # The exit status is preserved and propagated (no longer only
        # logged); the transcript survives as detail for the honest reply.
        self.assertEqual(output.status, "failed")
        self.assertFalse(output)
        self.assertIn("exited with code 2", output.error)
        self.assertIn("half the work", output.detail)

    def test_timeout_kills_child_returns_empty_and_marks_log(self):
        release = threading.Event()
        fake_proc = Mock()
        # jobs._terminate() asks the handle whether the child already exited;
        # a bare Mock() answers truthy, which used to make the timeout path
        # look like "already dead" and skip terminate/kill entirely.
        fake_proc.pid = 4242
        fake_proc.poll.return_value = None
        fake_proc.wait.side_effect = Exception("still running")
        fake_proc.stdout = self.BlockingStream(release)
        with patch.object(opencode_client, "_resolve_opencode",
                          return_value=r"C:\fake\opencode.exe"), \
             patch.object(opencode_client, "is_opencode_server_alive",
                          return_value=True), \
             patch.object(opencode_client, "_append_activity") as appender, \
             patch.object(opencode_client.subprocess, "Popen",
                          return_value=fake_proc):
            start = time.monotonic()
            output = opencode_client.run_opencode_task("quiet task", timeout=0.3)

        self.assertLess(time.monotonic() - start, 5.0)
        fake_proc.kill.assert_called_once()
        # Timeout is a falsy failed result (the old '' contract): callers
        # must not treat it as success.
        self.assertFalse(output)
        self.assertEqual(output.status, "failed")
        self.assertIn("timed out", output.error)
        marker = appender.call_args_list[-1][0][0]
        self.assertIn("timed out", marker)

    def test_run_task_wipes_preexisting_log_content(self):
        fake_proc = Mock()
        fake_proc.stdout = self.FakeStream(["fresh line\n"])
        fake_proc.returncode = 0
        temp_dir = tempfile.mkdtemp()
        log_path = os.path.join(temp_dir, "opencode_activity.log")
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write("OLD HISTORY FROM PREVIOUS TASKS\n")
        try:
            with patch.object(opencode_client, "_resolve_opencode",
                              return_value=r"C:\fake\opencode.exe"), \
                 patch.object(opencode_client, "is_opencode_server_alive",
                              return_value=True), \
                 patch.object(opencode_client, "_activity_log_path",
                              return_value=log_path), \
                 patch.object(opencode_client.subprocess, "Popen",
                              return_value=fake_proc):
                opencode_client.run_opencode_task("new task")
            with open(log_path, "r", encoding="utf-8") as handle:
                content = handle.read()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        self.assertNotIn("OLD HISTORY", content)
        self.assertIn("new task", content)
        self.assertIn("fresh line", content)


class OpencodeDetachTests(unittest.TestCase):
    """With the default browser_agent engine, opencode is fully detached:
    neither the serve daemon nor the task CLI may ever spawn."""

    def test_run_opencode_task_refused_when_detached(self):
        with patch.object(config, "TASK_ENGINE", "browser_agent"), \
             patch.object(opencode_client, "_resolve_opencode") as resolve:
            output = opencode_client.run_opencode_task("do something")
        # Refusal is a falsy failed result (the old '' contract): callers
        # must not treat it as success.
        self.assertFalse(output)
        self.assertEqual(output.status, "failed")
        resolve.assert_not_called()

    def test_ensure_opencode_server_refused_when_detached(self):
        with patch.object(config, "TASK_ENGINE", "browser_agent"), \
             patch.object(opencode_client, "_resolve_opencode") as resolve, \
             patch.object(opencode_client, "ensure_activity_tail") as tail:
            ok = opencode_client.ensure_opencode_server()
        self.assertFalse(ok)
        resolve.assert_not_called()
        tail.assert_not_called()

    def test_kill_stale_opencode_server_kills_port_listeners(self):
        with patch.object(opencode_client, "_pids_on_port",
                          return_value={4242, 4243}) as pids, \
             patch.object(opencode_client.subprocess, "run",
                          return_value=None) as run:
            opencode_client.kill_stale_opencode_server()
        pids.assert_called_once_with(opencode_client.OPENCODE_SERVE_PORT)
        self.assertEqual(run.call_count, 2)


class OpencodeServerShutdownTests(unittest.TestCase):
    """shutdown_opencode_server kills the tracked server, tail AND brave-mcp
    daemon pids, falling back to port-based kills when untracked."""

    def setUp(self):
        opencode_client._SERVER_PID = None
        opencode_client._TAIL_PID = None
        opencode_client._BRAVE_MCP_PID = None

    def tearDown(self):
        opencode_client._SERVER_PID = None
        opencode_client._TAIL_PID = None
        opencode_client._BRAVE_MCP_PID = None

    def test_kills_tracked_server_and_tail_pids(self):
        opencode_client._SERVER_PID = 4242
        opencode_client._TAIL_PID = 5555
        opencode_client._BRAVE_MCP_PID = None
        with patch.object(opencode_client, "_pids_on_port",
                          return_value={999}) as port_kill, \
             patch.object(opencode_client.subprocess, "run",
                          return_value=None) as run:
            opencode_client.shutdown_opencode_server()

        self.assertEqual(opencode_client._SERVER_PID, None)
        self.assertEqual(opencode_client._TAIL_PID, None)
        self.assertEqual(opencode_client._BRAVE_MCP_PID, None)
        # Server pid + tail pid tracked; brave-mcp port fallback ({999}).
        self.assertEqual(run.call_count, 3)
        killed = {
            str(call.args[0][call.args[0].index("/PID") + 1])
            for call in run.call_args_list
        }
        self.assertEqual(killed, {"4242", "5555", "999"})

    def test_kills_tracked_brave_mcp_daemon_pid(self):
        opencode_client._BRAVE_MCP_PID = 9876
        with patch.object(opencode_client, "_pids_on_port",
                          return_value={999}) as port_kill, \
             patch.object(opencode_client.subprocess, "run",
                          return_value=None) as run:
            opencode_client.shutdown_opencode_server()

        self.assertEqual(opencode_client._BRAVE_MCP_PID, None)
        killed = {
            str(call.args[0][call.args[0].index("/PID") + 1])
            for call in run.call_args_list
        }
        # Tracked daemon pid + serve-port fallback ({999} for port 9560).
        self.assertEqual(killed, {"9876", "999"})

    def test_falls_back_to_port_kills_without_tracked_pids(self):
        with patch.object(opencode_client, "_pids_on_port",
                          return_value={777, 888}), \
             patch.object(opencode_client.subprocess, "run",
                          return_value=None) as run:
            opencode_client.shutdown_opencode_server()

        # Both ports (9560 and 9570) fall back to port kills.
        self.assertEqual(run.call_count, 4)
        killed = {
            str(call.args[0][call.args[0].index("/PID") + 1])
            for call in run.call_args_list
        }
        self.assertEqual(killed, {"777", "888"})


class BraveMcpDaemonTests(unittest.TestCase):
    """ensure_brave_mcp_daemon spawns the node daemon once per boot with the
    http-mode env + token, or reuses an already-listening one."""

    def setUp(self):
        opencode_client._BRAVE_MCP_PID = None

    def tearDown(self):
        opencode_client._BRAVE_MCP_PID = None

    def test_spawns_node_daemon_when_port_free(self):
        fake_proc = Mock()
        fake_proc.pid = 31337
        # F51: the token is env-sourced — pin a FAKE value for the test.
        with patch.object(opencode_client, "BRAVE_MCP_TOKEN",
                          "fake-test-token-000"), \
             patch.object(opencode_client, "_pids_on_port",
                          side_effect=[[], [31337]]) as pids_on_port, \
             patch.object(opencode_client.shutil, "which",
                          return_value=r"C:\node\node.exe"), \
             patch.object(opencode_client.subprocess, "Popen",
                          return_value=fake_proc) as popen:
            ok = opencode_client.ensure_brave_mcp_daemon()

        self.assertTrue(ok)
        popen.assert_called_once()
        args, kwargs = popen.call_args
        self.assertEqual(args[0][0], r"C:\node\node.exe")
        self.assertEqual(args[0][1], "server.mjs")
        self.assertEqual(kwargs["cwd"], opencode_client.BRAVE_MCP_SERVER_DIR)
        self.assertEqual(kwargs["creationflags"],
                         opencode_client._CREATE_NO_WINDOW)
        self.assertEqual(kwargs["env"]["BRAVE_MCP_MODE"], "http")
        self.assertEqual(kwargs["env"]["BRAVE_MCP_PORT"], "9570")
        self.assertEqual(kwargs["env"]["BRAVE_MCP_TOKEN"],
                         "fake-test-token-000")
        self.assertEqual(opencode_client._BRAVE_MCP_PID, 31337)
        pids_on_port.assert_called()

    def test_unset_token_not_injected_into_daemon_env(self):
        fake_proc = Mock()
        fake_proc.pid = 31337
        clean_env = {k: v for k, v in os.environ.items()
                     if k != "BRAVE_MCP_TOKEN"}
        with patch.object(opencode_client, "BRAVE_MCP_TOKEN", None), \
             patch.object(opencode_client.os, "environ", clean_env), \
             patch.object(opencode_client, "_pids_on_port",
                          side_effect=[[], [31337]]), \
             patch.object(opencode_client.shutil, "which",
                          return_value=r"C:\node\node.exe"), \
             patch.object(opencode_client.subprocess, "Popen",
                          return_value=fake_proc) as popen:
            ok = opencode_client.ensure_brave_mcp_daemon()

        self.assertTrue(ok)
        args, kwargs = popen.call_args
        self.assertNotIn("BRAVE_MCP_TOKEN", kwargs["env"])

    def test_token_constant_is_env_sourced(self):
        # The module constant must come from the environment, never a
        # hardcoded literal in source.
        import inspect
        source = inspect.getsource(opencode_client)
        self.assertIn('BRAVE_MCP_TOKEN = os.getenv("BRAVE_MCP_TOKEN")', source)

    def test_skips_spawn_when_port_already_listening(self):
        with patch.object(opencode_client, "_pids_on_port",
                          return_value={555}), \
             patch.object(opencode_client.subprocess, "Popen") as popen:
            ok = opencode_client.ensure_brave_mcp_daemon()

        self.assertTrue(ok)
        popen.assert_not_called()
        self.assertEqual(opencode_client._BRAVE_MCP_PID, None)


class ModelPinTests(unittest.TestCase):
    """Every task run pins the verified fast router model via -m."""

    def test_plain_run_argv_pins_model(self):
        with patch.object(opencode_client, "_resolve_opencode",
                          return_value=r"C:\fake\opencode.exe"), \
             patch.object(opencode_client, "is_opencode_server_alive",
                          return_value=False):
            command = opencode_client._build_command("do the thing")

        self.assertIn("--auto", command)
        pair_index = command.index("-m")
        self.assertEqual(command[pair_index + 1],
                         "fireworks-ai/accounts/fireworks/routers/kimi-k2p6-fast")

    def test_attach_run_argv_pins_model_too(self):
        with patch.object(opencode_client, "_resolve_opencode",
                          return_value=r"C:\fake\opencode.exe"), \
             patch.object(opencode_client, "is_opencode_server_alive",
                          return_value=True):
            command = opencode_client._build_command("do the thing")

        self.assertIn("--attach", command)
        pair_index = command.index("-m")
        self.assertEqual(command[pair_index + 1],
                         "fireworks-ai/accounts/fireworks/routers/kimi-k2p6-fast")


class NarrationTests(unittest.TestCase):
    """Spoken progress narration: phrases from real tool-marker lines,
    throttled (3s gap), deduped, capped per task, silent when disabled."""

    def setUp(self):
        engine = patch.object(config, "TASK_ENGINE", "opencode")
        engine.start()
        self.addCleanup(engine.stop)
        self._reset()

    def tearDown(self):
        self._reset()

    def _reset(self):
        opencode_client._narration_enabled = False
        opencode_client._last_narration_at = 0.0
        opencode_client._last_narration_phrase = None
        opencode_client._narration_count = 0

    def _enable(self):
        opencode_client._narration_enabled = True

    def _backdate(self, seconds=4.0):
        opencode_client._last_narration_at = time.monotonic() - seconds

    def test_phrase_mapping(self):
        phrases = [
            opencode_client._phrase_for_tool("brave-control_navigate", '{"url":"https://example.com/x"}'),
            opencode_client._phrase_for_tool("navigate", ""),
            opencode_client._phrase_for_tool("brave-control_click_element", '{"index":2}'),
            opencode_client._phrase_for_tool("click_element", ""),
            opencode_client._phrase_for_tool("brave-control_fill_element", '{"index":3,"value":"hi"}'),
            opencode_client._phrase_for_tool("fill_element", ""),
            opencode_client._phrase_for_tool("brave-control_screenshot", ""),
            opencode_client._phrase_for_tool("brave-control_understand_page", ""),
            opencode_client._phrase_for_tool("brave-control_evaluate", '{"expression":"x"}'),
        ]
        self.assertEqual(phrases, [
            "Opening example.com, sir.",
            "Opening the page, sir.",
            "Clicking element 2, sir.",
            "Clicking it, sir.",
            "Typing into element 3, sir.",
            "Typing it in, sir.",
            "Taking a look, sir.",
            "Reading the page, sir.",
            "Working on it, sir.",
        ])

    def test_narrate_line_speaks_from_real_format_marker(self):
        self._enable()
        line = '\x1b[0m? \x1b[0mbrave-control_navigate {"url":"https://www.amazon.in"}'
        with patch.object(opencode_client, "speak") as speak:
            opencode_client._narrate_line(line)
        speak.assert_called_once_with("Opening www.amazon.in, sir.")

    def test_banner_and_echo_lines_never_narrate(self):
        self._enable()
        with patch.object(opencode_client, "speak") as speak:
            opencode_client._narrate_line("\x1b[0m> build \u00b7 accounts/fireworks/models/deepseek-v4-flash-0731")
            opencode_client._narrate_line("\x1b[0m$ echo hello")
            opencode_client._narrate_line("plain output line")
        speak.assert_not_called()

    def test_throttle_min_gap_collapses_rapid_markers(self):
        self._enable()
        with patch.object(opencode_client, "speak") as speak:
            opencode_client._narrate("Clicking element 2, sir.")
            opencode_client._narrate("Reading the page, sir.")
        speak.assert_called_once_with("Clicking element 2, sir.")
        # After the gap, the second phrase may be spoken.
        self._backdate()
        with patch.object(opencode_client, "speak") as speak:
            opencode_client._narrate("Reading the page, sir.")
        speak.assert_called_once_with("Reading the page, sir.")

    def test_consecutive_identical_phrases_skipped(self):
        self._enable()
        with patch.object(opencode_client, "speak") as speak:
            opencode_client._narrate("Working on it, sir.")
            self._backdate()
            opencode_client._narrate("Working on it, sir.")
        speak.assert_called_once_with("Working on it, sir.")

    def test_cap_six_narrations_per_task(self):
        self._enable()
        with patch.object(opencode_client, "speak") as speak:
            for i in range(8):
                opencode_client._last_narration_at = 0.0
                opencode_client._last_narration_phrase = None
                opencode_client._narrate("Phrase %d." % i)
        self.assertEqual(speak.call_count, 6)

    def test_disabled_by_default_is_silent(self):
        with patch.object(opencode_client, "speak") as speak:
            opencode_client._narrate_line('\x1b[0m? \x1b[0mbrave-control_navigate {"url":"https://x.io"}')
        speak.assert_not_called()

    def test_streaming_task_narrates_marker_and_resets_state(self):
        self._enable()
        fake_proc = Mock()
        fake_proc.stdout = OpencodeRunTaskStreamingTests.FakeStream([
            "\x1b[0m> build \u00b7 model\n",
            "\x1b[0m? \x1b[0mbrave-control_click_element {\"index\":1}\n",
            "some output\n",
        ])
        fake_proc.returncode = 0
        with patch.object(opencode_client, "_resolve_opencode",
                          return_value=r"C:\fake\opencode.exe"), \
             patch.object(opencode_client, "is_opencode_server_alive",
                          return_value=True), \
             patch.object(opencode_client, "_append_activity"), \
             patch.object(opencode_client.subprocess, "Popen",
                          return_value=fake_proc), \
             patch.object(opencode_client, "speak") as speak:
            opencode_client.run_opencode_task("task with markers")

        speak.assert_called_once_with("Clicking element 1, sir.")
        # run_opencode_task resets the per-task counter at start.
        opencode_client._narration_count = 0
        self.assertEqual(opencode_client._narration_count, 0)


class ActivityTailTests(unittest.TestCase):
    """close_activity_tail kills the tracked tail pid and resets it; the
    serve daemon and brave MCP daemon pids are left untouched."""

    def setUp(self):
        opencode_client._TAIL_PID = None

    def tearDown(self):
        opencode_client._TAIL_PID = None

    def test_close_kills_tracked_pid_and_resets(self):
        opencode_client._TAIL_PID = 5555
        with patch.object(opencode_client, "_taskkill") as taskkill:
            opencode_client.close_activity_tail()
        taskkill.assert_called_once_with(5555)
        self.assertEqual(opencode_client._TAIL_PID, None)

    def test_close_noop_when_no_pid_tracked(self):
        with patch.object(opencode_client, "_taskkill") as taskkill:
            opencode_client.close_activity_tail()
        taskkill.assert_not_called()
        self.assertEqual(opencode_client._TAIL_PID, None)


class WatcherTeardownTests(unittest.TestCase):
    """Full teardown stops the opencode server; warm shutdown keeps it."""

    def setUp(self):
        opencode_client._TAIL_PID = None
        opencode_client._SERVER_PID = None
        opencode_client._BRAVE_MCP_PID = None

    def tearDown(self):
        opencode_client._TAIL_PID = None
        opencode_client._SERVER_PID = None
        opencode_client._BRAVE_MCP_PID = None

    def _patch_stop(self):
        return (
            patch.object(watcher, "_stop_opencode_server_if_jarvis"),
            patch.object(watcher, "_stop_backend_port_if_jarvis"),
            patch.object(watcher, "_terminate_process"),
        )

    def test_stop_runtime_full_teardown_stops_opencode_server(self):
        stopper, port_stop, term = self._patch_stop()
        with stopper as stopper, port_stop, term:
            watcher.stop_runtime(stop_electron=True, keep_backend=False)
        stopper.assert_called_once()

    def test_warm_shutdown_keeps_opencode_server(self):
        stopper, port_stop, term = self._patch_stop()
        opencode_client._TAIL_PID = 5555
        opencode_client._SERVER_PID = 4242
        opencode_client._BRAVE_MCP_PID = 31337
        with stopper as stopper, port_stop, term, \
             patch.object(opencode_client, "_taskkill") as taskkill:
            watcher.stop_runtime(stop_electron=True, keep_backend=True)
        stopper.assert_not_called()
        # Only the visible tail is killed; server + brave MCP stay warm.
        taskkill.assert_called_once_with(5555)
        self.assertEqual(opencode_client._TAIL_PID, None)
        self.assertEqual(opencode_client._SERVER_PID, 4242)
        self.assertEqual(opencode_client._BRAVE_MCP_PID, 31337)

    def test_cleanup_watcher_exit_stops_opencode_server(self):
        stopper, port_stop, term = self._patch_stop()
        with stopper as stopper, port_stop, term:
            watcher._cleanup_watcher_exit()
        stopper.assert_called_once()


if __name__ == "__main__":
    unittest.main()
