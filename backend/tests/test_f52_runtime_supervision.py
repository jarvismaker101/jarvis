"""F52 — Supervise a Versioned, Observable Runtime.

Acceptance (audit report):

* wrong-token/old-version services are not reused;
* foreign listeners survive;
* worker crashes recover while UI lives;
* old exits cannot delete new stamps;
* full shutdown leaves no unintended daemon.

Everything here drives the REAL code paths with mocked process/HTTP APIs:
no supervisor, backend, daemon or Electron process is ever started or killed.
"""

import json
import os
import pathlib
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from backend import watcher
from backend.services import local_auth
from backend.services import opencode_client
from backend.services import runtime_identity


class FakeProc:
    """Minimal Popen stand-in: poll/wait/kill + a pid."""

    def __init__(self, pid, exit_code=None):
        self.pid = pid
        self._exit_code = exit_code
        self.killed = False

    def poll(self):
        return self._exit_code

    def wait(self, timeout=None):
        return self._exit_code

    def kill(self):
        self.killed = True
        self._exit_code = 0


class WatcherTestCase(unittest.TestCase):
    """Isolates watcher globals, the owned-handle registry and stamp dir."""

    GLOBALS = (
        "jarvis_running",
        "backend_proc",
        "voice_proc",
        "electron_proc",
        "whisper_daemon_proc",
        "whisper_daemon_ok",
    )

    def setUp(self):
        self._saved = {name: getattr(watcher, name) for name in self.GLOBALS}
        watcher.stop_worker_supervisor()
        watcher._clear_owned()
        self.runtime_dir = pathlib.Path(tempfile.mkdtemp(prefix="f52-runtime-"))
        self.addCleanup(shutil.rmtree, str(self.runtime_dir), ignore_errors=True)
        patcher = patch.object(runtime_identity, "_RUNTIME_DIR", self.runtime_dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in self.GLOBALS:
            setattr(watcher, name, None if name != "jarvis_running" else False)

    def tearDown(self):
        watcher.stop_worker_supervisor()
        watcher._clear_owned()
        for name, value in self._saved.items():
            setattr(watcher, name, value)

    # ── helpers ────────────────────────────────────────────────────────────
    def write_stamp(self, role, **fields):
        payload = {
            "role": role,
            "instance_id": "worker-instance",
            "pid": 4242,
            "started_at": 1234.5,
            "protocol": runtime_identity.protocol_version(),
            "build": "dev",
        }
        payload.update(fields)
        (self.runtime_dir / ("%s-instance.json" % role)).write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return payload

    def expected_fingerprint(self):
        return watcher._expected_token_fingerprint()

    def live_health(self, **fields):
        health = {
            "ok": True,
            "service": "jarvis-backend",
            "instance_id": "worker-instance",
            "pid": 4242,
            "protocol": runtime_identity.protocol_version(),
            "build": "dev",
            "auth": self.expected_fingerprint(),
        }
        health.update(fields)
        return health


# ── Acceptance: old exits cannot delete new stamps ──────────────────────────
class StampOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.runtime_dir = pathlib.Path(tempfile.mkdtemp(prefix="f52-stamp-"))
        self.addCleanup(shutil.rmtree, str(self.runtime_dir), ignore_errors=True)
        patcher = patch.object(runtime_identity, "_RUNTIME_DIR", self.runtime_dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_start_identity_is_stable(self):
        first = runtime_identity.started_at()
        time.sleep(0.02)
        self.assertEqual(first, runtime_identity.started_at())

    def test_stamp_carries_the_stable_start_identity(self):
        stamp = runtime_identity.write_instance_file("backend")
        self.assertIsNotNone(stamp)
        self.assertEqual(stamp["started_at"], runtime_identity.started_at())
        self.assertEqual(
            runtime_identity.read_instance_file("backend")["started_at"],
            runtime_identity.started_at(),
        )

    def test_old_exit_cannot_delete_new_stamp(self):
        foreign = {
            "role": "backend",
            "instance_id": "older-instance",
            "pid": 1111,
            "started_at": 1.0,
            "protocol": runtime_identity.protocol_version(),
        }
        (self.runtime_dir / "backend-instance.json").write_text(
            json.dumps(foreign), encoding="utf-8"
        )
        self.assertFalse(runtime_identity.clear_instance_file("backend"))
        self.assertEqual(
            runtime_identity.read_instance_file("backend")["instance_id"],
            "older-instance",
        )

    def test_owning_process_clears_its_own_stamp(self):
        runtime_identity.write_instance_file("voice")
        self.assertTrue(runtime_identity.clear_instance_file("voice"))
        self.assertIsNone(runtime_identity.read_instance_file("voice"))

    def test_supervisor_clears_only_the_worker_it_retired(self):
        payload = {
            "role": "backend",
            "instance_id": "child-instance",
            "pid": 4242,
            "started_at": 1.0,
            "protocol": runtime_identity.protocol_version(),
        }
        (self.runtime_dir / "backend-instance.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        # A supervisor that never owned that pid must not delete the stamp.
        self.assertFalse(
            runtime_identity.clear_instance_file("backend", owned_pid=9999)
        )
        self.assertIsNotNone(runtime_identity.read_instance_file("backend"))
        # The supervisor that retired pid 4242 may clear it.
        self.assertTrue(
            runtime_identity.clear_instance_file("backend", owned_pid=4242)
        )
        self.assertIsNone(runtime_identity.read_instance_file("backend"))

    def test_concurrent_writers_publish_a_valid_stamp_without_temp_litter(self):
        errors = []

        def writer():
            try:
                for _ in range(5):
                    stamp = runtime_identity.write_instance_file("backend")
                    if not stamp or stamp["role"] != "backend":
                        errors.append("bad write result")
            except Exception as exc:  # pragma: no cover - failure detail
                errors.append(repr(exc))

        threads = [threading.Thread(target=writer) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        final = runtime_identity.read_instance_file("backend")
        self.assertIsNotNone(final)
        self.assertEqual(final["instance_id"], runtime_identity.instance_id())
        self.assertEqual(sorted(p.name for p in self.runtime_dir.iterdir()),
                         ["backend-instance.json"])


# ── Acceptance: wrong-token / old-version services are not reused ───────────
class BackendReuseIdentityTests(WatcherTestCase):
    def test_verified_identity_is_reused(self):
        self.write_stamp("backend")
        self.assertTrue(watcher._backend_matches_stamp(self.live_health()))

    def test_old_version_is_not_reused(self):
        self.write_stamp("backend", protocol=1)
        health = self.live_health(protocol=1)
        self.assertFalse(watcher._backend_matches_stamp(health))

    def test_wrong_token_is_not_reused(self):
        """A restarted watcher mints a new token: the warm backend is stale."""
        self.write_stamp("backend", auth="deadbeef")
        health = self.live_health(auth="deadbeef")
        self.assertNotEqual("deadbeef", self.expected_fingerprint())
        self.assertFalse(watcher._backend_matches_stamp(health))

    def test_unarmed_worker_is_not_reused(self):
        self.write_stamp("backend", auth="off")
        health = self.live_health(auth="off")
        self.assertFalse(watcher._backend_matches_stamp(health))

    def test_instance_mismatch_is_not_reused(self):
        self.write_stamp("backend", instance_id="someone-else")
        self.assertFalse(watcher._backend_matches_stamp(self.live_health()))

    def test_stamp_of_another_pid_is_not_reused(self):
        self.write_stamp("backend", pid=5555)
        self.assertFalse(watcher._backend_matches_stamp(self.live_health()))

    def test_missing_stamp_is_not_reused(self):
        self.assertFalse(watcher._backend_matches_stamp(self.live_health()))

    def test_non_jarvis_listener_is_not_reused(self):
        self.write_stamp("backend")
        self.assertFalse(
            watcher._backend_matches_stamp(self.live_health(service="something-else"))
        )

    def test_expected_fingerprint_matches_the_armed_worker_report(self):
        token = local_auth.mint_token()
        local_auth.configure(token)
        try:
            self.assertEqual(
                local_auth.token_fingerprint(), local_auth.fingerprint_for(token)
            )
        finally:
            local_auth.configure("")
        self.assertEqual(local_auth.fingerprint_for(""), "off")
        self.assertEqual(local_auth.fingerprint_for("too-short"), "off")

    def test_health_reports_the_listening_pid(self):
        from fastapi.testclient import TestClient
        from backend.main import app

        body = TestClient(app).get("/health").json()
        self.assertEqual(body["service"], "jarvis-backend")
        self.assertEqual(body["pid"], os.getpid())
        self.assertEqual(body["protocol"], runtime_identity.protocol_version())


# ── Acceptance: foreign listeners survive ──────────────────────────────────
class ForeignListenerTests(WatcherTestCase):
    def test_only_the_attributed_pid_is_killed(self):
        health = self.live_health(pid=4242)
        with patch.object(watcher, "_pids_on_port", return_value={4242, 9999}), \
             patch.object(watcher, "_taskkill_pid") as kill:
            self.assertTrue(watcher._kill_backend_listener_if_owned(health))
        self.assertEqual([call.args[0] for call in kill.call_args_list], [4242])

    def test_port_fallback_is_attribution_scoped(self):
        with patch.object(watcher, "_backend_health",
                          return_value=self.live_health(pid=4242)), \
             patch.object(watcher, "_pids_on_port", return_value={4242, 9999}), \
             patch.object(watcher, "_taskkill_pid") as kill:
            self.assertTrue(watcher._stop_backend_port_if_jarvis())
        self.assertEqual([call.args[0] for call in kill.call_args_list], [4242])

    def test_foreign_listener_on_the_port_is_never_killed(self):
        # /health claims pid 4242 but the live listener is a foreign pid 7777.
        health = self.live_health(pid=4242)
        with patch.object(watcher, "_pids_on_port", return_value={7777}), \
             patch.object(watcher, "_taskkill_pid") as kill:
            self.assertFalse(watcher._kill_backend_listener_if_owned(health))
            self.assertFalse(watcher._stop_backend_port_if_jarvis())
        kill.assert_not_called()

    def test_non_jarvis_service_is_never_killed(self):
        with patch.object(watcher, "_pids_on_port", return_value={7777}), \
             patch.object(watcher, "_taskkill_pid") as kill:
            self.assertFalse(
                watcher._kill_backend_listener_if_owned(
                    {"ok": True, "service": "something-else", "pid": 7777}
                )
            )
        kill.assert_not_called()

    def test_unattributable_jarvis_listener_is_not_killed(self):
        health = {"ok": True, "service": "jarvis-backend", "instance_id": "x"}
        with patch.object(watcher, "_pids_on_port", return_value={7777}), \
             patch.object(watcher, "_taskkill_pid") as kill:
            self.assertIsNone(watcher._jarvis_backend_pid(health))
            self.assertFalse(watcher._kill_backend_listener_if_owned(health))
        kill.assert_not_called()

    def test_stamp_attributes_the_listening_pid(self):
        self.write_stamp("backend", instance_id="x", pid=4242)
        health = {"ok": True, "service": "jarvis-backend", "instance_id": "x"}
        with patch.object(watcher, "_pids_on_port", return_value={4242, 7777}), \
             patch.object(watcher, "_taskkill_pid") as kill:
            self.assertTrue(watcher._kill_backend_listener_if_owned(health))
        self.assertEqual([call.args[0] for call in kill.call_args_list], [4242])


# ── Acceptance: worker crashes recover while the UI lives ──────────────────
class WorkerCrashRecoveryTests(WatcherTestCase):
    def setUp(self):
        super().setUp()
        watcher.jarvis_running = True
        watcher._register_owned("electron", FakeProc(1001))

    def test_crashed_voice_worker_is_restarted_and_handle_replaced(self):
        spawned = []

        def respawn():
            proc = FakeProc(2000 + len(spawned))
            spawned.append(proc)
            return proc

        watcher._register_owned("voice", FakeProc(2001, exit_code=3), respawn=respawn)
        restarted = watcher.supervise_owned_workers()

        self.assertEqual(restarted, ["voice"])
        self.assertEqual(len(spawned), 1)
        entry = watcher._owned_entry("voice")
        self.assertIs(entry["proc"], spawned[0])
        self.assertIsNone(entry["proc"].poll())

    def test_restart_budget_bounds_recovery(self):
        spawned = []

        def respawn():
            # Comes back dead again → the next pass sees another crash.
            proc = FakeProc(3000 + len(spawned), exit_code=1)
            spawned.append(proc)
            return proc

        watcher._register_owned("backend", FakeProc(3001, exit_code=2), respawn=respawn)
        for _ in range(watcher.WORKER_RESTART_BUDGET + 3):
            watcher.supervise_owned_workers()

        self.assertEqual(len(spawned), watcher.WORKER_RESTART_BUDGET)
        self.assertIsNotNone(watcher._owned_entry("backend")["proc"])

    def test_healthy_workers_are_left_alone(self):
        respawn = Mock(return_value=FakeProc(4001))
        watcher._register_owned("voice", FakeProc(4000), respawn=respawn)
        self.assertEqual(watcher.supervise_owned_workers(), [])
        respawn.assert_not_called()

    def test_closed_ui_is_a_shutdown_not_a_crash(self):
        respawn = Mock(return_value=FakeProc(5001))
        watcher._register_owned("electron", FakeProc(1002, exit_code=0))
        watcher._register_owned("voice", FakeProc(5000, exit_code=9), respawn=respawn)
        self.assertEqual(watcher.supervise_owned_workers(), [])
        respawn.assert_not_called()

    def test_nothing_is_restarted_outside_a_running_session(self):
        watcher.jarvis_running = False
        respawn = Mock(return_value=FakeProc(6001))
        watcher._register_owned("voice", FakeProc(6000, exit_code=9), respawn=respawn)
        self.assertEqual(watcher.supervise_owned_workers(), [])
        respawn.assert_not_called()

    def test_worker_without_a_respawn_spec_is_left_down(self):
        watcher._register_owned("voice", FakeProc(7000, exit_code=9))
        self.assertEqual(watcher.supervise_owned_workers(), [])

    def test_supervisor_start_and_stop(self):
        thread = watcher.start_worker_supervisor()
        self.assertTrue(thread.is_alive())
        watcher.stop_worker_supervisor()
        self.assertIsNone(watcher._worker_supervisor_thread)
        # Idempotent: stopping twice is safe, and a fresh start is possible.
        watcher.stop_worker_supervisor()
        thread = watcher.start_worker_supervisor()
        self.assertTrue(thread.is_alive())
        watcher.stop_worker_supervisor()
        thread.join(timeout=5)


# ── Acceptance: launch adopts/keeps creation-identity-bound handles ─────────
class LaunchWiringTests(WatcherTestCase):
    def setUp(self):
        super().setUp()
        patches = [
            patch.object(watcher, "_warn_if_low_memory"),
            patch.object(watcher, "_invalidate_backend_approvals"),
            patch.object(watcher, "_taskkill_pid"),
            patch.object(watcher, "_wait_for_port_release", return_value=True),
            patch.object(watcher, "_backend_has_research_endpoint", return_value=True),
            patch.object(opencode_client, "ensure_activity_tail"),
            patch.object(opencode_client, "kill_stale_opencode_server"),
            patch.object(opencode_client, "ensure_opencode_server"),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.popen = patch.object(watcher.subprocess, "Popen")
        self.popen_mock = self.popen.start()
        self.addCleanup(self.popen.stop)

    def test_verified_warm_backend_is_reused_and_its_pid_retained(self):
        self.write_stamp("backend")
        voice = FakeProc(6001)
        electron = FakeProc(6002)
        with patch.object(watcher, "_backend_health", return_value=self.live_health()), \
             patch.object(watcher, "_pids_on_port", return_value={4242}), \
             patch.object(watcher, "_spawn_with_retry",
                          side_effect=[voice, electron]) as spawn, \
             patch.object(watcher, "_spawn_backend") as spawn_backend, \
             patch.object(watcher, "start_worker_supervisor") as supervise:
            watcher.launch_jarvis()

        spawn_backend.assert_not_called()
        watcher._taskkill_pid.assert_not_called()
        entry = watcher._owned_entry("backend")
        self.assertEqual(entry["pid"], 4242)
        self.assertTrue(entry["adopted"])
        self.assertEqual(entry["identity"]["instance_id"], "worker-instance")
        self.assertTrue(callable(entry["respawn"]))
        # The adopted pid is still handed to the voice worker's shutdown path.
        self.assertEqual(spawn.call_args_list[0].kwargs["env"]["JARVIS_BACKEND_PID"],
                         "4242")
        self.assertEqual(watcher._owned_entry("voice")["pid"], 6001)
        self.assertEqual(watcher._owned_entry("electron")["pid"], 6002)
        self.assertTrue(watcher.jarvis_running)
        supervise.assert_called_once()

    def test_wrong_token_backend_is_stopped_and_replaced(self):
        self.write_stamp("backend", auth="oldlaunch")
        stale_health = self.live_health(auth="oldlaunch")
        fresh = FakeProc(6001)
        with patch.object(watcher, "_backend_health", return_value=stale_health), \
             patch.object(watcher, "_pids_on_port", side_effect=[{4242}, set()]), \
             patch.object(watcher, "_spawn_backend", return_value=fresh) as spawn_backend, \
             patch.object(watcher, "_spawn_with_retry",
                          side_effect=[FakeProc(6002), FakeProc(6003)]), \
             patch.object(watcher, "start_worker_supervisor"):
            watcher.launch_jarvis()

        # The stale, attributable pid is stopped (and nothing else).
        self.assertEqual(
            [call.args[0] for call in watcher._taskkill_pid.call_args_list], [4242]
        )
        # And its authority was invalidated before the restart.
        watcher._invalidate_backend_approvals.assert_called_once()
        spawn_backend.assert_called_once()
        entry = watcher._owned_entry("backend")
        self.assertIs(entry["proc"], fresh)
        self.assertEqual(entry["pid"], 6001)
        self.assertFalse(entry["adopted"])

    def test_unattributable_listener_is_left_alone_and_launch_cancelled(self):
        foreign = {"ok": True, "service": "jarvis-backend",
                   "instance_id": "foreign", "pid": 4242, "protocol": 1}
        with patch.object(watcher, "_backend_health", return_value=foreign), \
             patch.object(watcher, "_pids_on_port", return_value={7777}), \
             patch.object(watcher, "_wait_for_port_release", return_value=False), \
             patch.object(watcher, "_spawn_backend") as spawn_backend, \
             patch.object(watcher, "_spawn_with_retry") as spawn:
            watcher.launch_jarvis()

        watcher._taskkill_pid.assert_not_called()
        spawn_backend.assert_not_called()
        spawn.assert_not_called()
        self.assertFalse(watcher.jarvis_running)
        self.assertEqual(watcher._owned_pids(), set())


# ── Backend spawn/recovery share one validated routine ─────────────────────
class BackendSpawnTests(WatcherTestCase):
    def test_a_ready_backend_is_returned(self):
        proc = FakeProc(8001)
        with patch.object(watcher.subprocess, "Popen", return_value=proc) as popen, \
             patch.object(watcher, "wait_for_backend_ready", return_value=True):
            self.assertIs(watcher._spawn_backend({}), proc)
        self.assertEqual(popen.call_count, 1)
        self.assertIn("uvicorn", popen.call_args.args[0])

    def test_an_unready_backend_consumes_the_budget_and_cleans_up(self):
        with patch.object(watcher.subprocess, "Popen",
                          side_effect=lambda *a, **k: FakeProc(8002)) as popen, \
             patch.object(watcher, "wait_for_backend_ready", return_value=False), \
             patch.object(watcher, "_kill_backend_listener_if_owned") as cleanup, \
             patch.object(watcher, "_taskkill_pid") as kill, \
             patch.object(watcher.time, "sleep"):
            self.assertIsNone(watcher._spawn_backend({}, attempts=2))
        self.assertEqual(popen.call_count, 2)
        self.assertEqual(cleanup.call_count, 2)
        # The failed child is OUR OWN pid, so it is always safe to tree-kill.
        self.assertEqual([call.args[0] for call in kill.call_args_list], [8002, 8002])


# ── Acceptance: full shutdown leaves no unintended daemon ──────────────────
class ProcessSetTests(WatcherTestCase):
    def test_full_shutdown_stops_every_owned_worker_and_daemon(self):
        for role, pid in (("backend", 4242), ("voice", 4243),
                          ("electron", 4244), ("whisper_daemon", 4245)):
            watcher._register_owned(role, FakeProc(pid))
        watcher.jarvis_running = True

        with patch.object(watcher, "_taskkill_pid") as kill, \
             patch.object(watcher, "_invalidate_backend_approvals") as invalidate, \
             patch.object(watcher, "_stop_opencode_server_if_jarvis") as opencode, \
             patch.object(watcher, "_stop_backend_port_if_jarvis") as port_stop, \
             patch.object(watcher, "_whisper_daemon_health", return_value=None):
            watcher.stop_runtime(stop_electron=True, keep_backend=False)

        killed = {call.args[0] for call in kill.call_args_list}
        self.assertEqual(killed, {4242, 4243, 4244, 4245})
        self.assertEqual(watcher._owned_pids(), set())
        self.assertIsNone(watcher.whisper_daemon_proc)
        self.assertFalse(watcher.whisper_daemon_ok)
        self.assertFalse(watcher.jarvis_running)
        invalidate.assert_called_once()
        opencode.assert_called_once()
        port_stop.assert_called_once()

    def test_full_shutdown_clears_only_the_retired_workers_stamp(self):
        self.write_stamp("backend", instance_id="retired", pid=4242)
        watcher._register_owned("backend", FakeProc(4242))
        with patch.object(watcher, "_taskkill_pid"), \
             patch.object(watcher, "_invalidate_backend_approvals"), \
             patch.object(watcher, "_stop_opencode_server_if_jarvis"), \
             patch.object(watcher, "_stop_backend_port_if_jarvis"), \
             patch.object(watcher, "_stop_whisper_daemon", return_value=False):
            watcher.stop_runtime(stop_electron=True, keep_backend=False)
        self.assertIsNone(runtime_identity.read_instance_file("backend"))

    def test_full_shutdown_leaves_a_newer_stamp_alone(self):
        self.write_stamp("backend", instance_id="newer", pid=7777)
        watcher._register_owned("backend", FakeProc(4242))
        with patch.object(watcher, "_taskkill_pid"), \
             patch.object(watcher, "_invalidate_backend_approvals"), \
             patch.object(watcher, "_stop_opencode_server_if_jarvis"), \
             patch.object(watcher, "_stop_backend_port_if_jarvis"), \
             patch.object(watcher, "_stop_whisper_daemon", return_value=False):
            watcher.stop_runtime(stop_electron=True, keep_backend=False)
        stamp = runtime_identity.read_instance_file("backend")
        self.assertIsNotNone(stamp)
        self.assertEqual(stamp["instance_id"], "newer")

    def test_warm_sleep_keeps_the_backend_and_daemons(self):
        for role, pid in (("backend", 4242), ("voice", 4243),
                          ("electron", 4244), ("whisper_daemon", 4245)):
            watcher._register_owned(role, FakeProc(pid))
        watcher.jarvis_running = True

        with patch.object(watcher, "_taskkill_pid") as kill, \
             patch.object(watcher, "_invalidate_backend_approvals") as invalidate, \
             patch.object(watcher, "_stop_opencode_server_if_jarvis") as opencode, \
             patch.object(opencode_client, "close_activity_tail") as tail:
            watcher.stop_runtime(stop_electron=True, keep_backend=True)

        killed = {call.args[0] for call in kill.call_args_list}
        self.assertEqual(killed, {4243, 4244})       # voice + UI only
        self.assertIsNotNone(watcher._owned_entry("backend"))
        self.assertIsNotNone(watcher._owned_entry("whisper_daemon"))
        invalidate.assert_not_called()
        opencode.assert_not_called()
        tail.assert_called_once()
        self.assertFalse(watcher.jarvis_running)

    def test_warm_sleep_does_not_stop_an_adopted_whisper_daemon(self):
        watcher._register_owned("whisper_daemon", None, pid=9001, adopted=True)
        with patch.object(watcher, "_taskkill_pid") as kill, \
             patch.object(opencode_client, "close_activity_tail"):
            watcher.stop_runtime(stop_electron=True, keep_backend=True)
        kill.assert_not_called()
        self.assertEqual(watcher._owned_entry("whisper_daemon")["pid"], 9001)

    def test_an_aborted_launch_keeps_the_resident_daemons(self):
        watcher._register_owned("whisper_daemon", FakeProc(9001))
        with patch.object(watcher, "_taskkill_pid") as kill, \
             patch.object(watcher, "_stop_opencode_server_if_jarvis") as opencode, \
             patch.object(watcher, "_stop_backend_port_if_jarvis") as port_stop:
            watcher.stop_runtime(stop_electron=False, stop_daemons=False)
        kill.assert_not_called()
        opencode.assert_not_called()
        port_stop.assert_not_called()
        self.assertIsNotNone(watcher._owned_entry("whisper_daemon"))

    def test_attributed_whisper_daemon_is_stopped_on_full_shutdown(self):
        health = {"ok": True, "service": "jarvis-whisper", "pid": 9001}
        with patch.object(watcher, "_whisper_daemon_health", return_value=health), \
             patch.object(watcher, "_pids_on_port", return_value={9001}), \
             patch.object(watcher, "_taskkill_pid") as kill:
            self.assertTrue(watcher._stop_whisper_daemon())
        self.assertEqual([call.args[0] for call in kill.call_args_list], [9001])

    def test_foreign_whisper_listener_survives(self):
        health = {"ok": True, "service": "jarvis-whisper", "pid": 9001}
        with patch.object(watcher, "_whisper_daemon_health", return_value=health), \
             patch.object(watcher, "_pids_on_port", return_value={7777}), \
             patch.object(watcher, "_taskkill_pid") as kill:
            self.assertIsNone(watcher._attributed_whisper_pid())
            self.assertFalse(watcher._stop_whisper_daemon())
        kill.assert_not_called()

    def test_whisper_stamp_attributes_an_older_daemon(self):
        self.write_stamp("whisper_daemon", instance_id="whisper", pid=9001)
        with patch.object(watcher, "_whisper_daemon_health", return_value=None), \
             patch.object(watcher, "_pids_on_port", return_value={9001}), \
             patch.object(watcher, "_taskkill_pid") as kill:
            self.assertTrue(watcher._stop_whisper_daemon())
        self.assertEqual([call.args[0] for call in kill.call_args_list], [9001])

    def test_stop_sets_are_explicit(self):
        self.assertEqual(set(watcher.SLEEP_STOP_ROLES), {"voice", "electron"})
        self.assertEqual(
            set(watcher.SHUTDOWN_STOP_ROLES),
            {"backend", "voice", "electron", "whisper_daemon"},
        )
        self.assertTrue(set(watcher.SLEEP_STOP_ROLES) <= set(watcher.SHUTDOWN_STOP_ROLES))


if __name__ == "__main__":
    unittest.main()
