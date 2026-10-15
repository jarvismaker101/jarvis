"""Live Fix 31 regression tests for bugs B78-B81."""

import ctypes
import pytest

from backend.core import brain
from backend.services import approvals, screen_control, screen_state, screen_ui_elements


@pytest.fixture(autouse=True)
def _clean_ledgers(monkeypatch):
    """Reset screen_state, approvals, window enum, and brain action ledgers."""
    monkeypatch.setenv("JARVIS_SCREEN_STATE_PATH", "")
    screen_state.set_enabled(False, reason="test_reset")
    screen_state.clear_interactions()
    screen_state.clear_pending_plan()
    screen_state.clear_refused_chain()
    approvals.clear()
    screen_control._clear_window_enumeration()
    with brain._last_action_lock:
        brain._last_action.update({
            "kind": "", "command": "", "summary": "",
            "done": False, "seq": 0, "at": 0.0,
        })
    yield
    screen_state.set_enabled(False, reason="test_teardown")
    screen_state.clear_interactions()
    screen_state.clear_pending_plan()
    screen_state.clear_refused_chain()
    approvals.clear()
    screen_control._clear_window_enumeration()


class TestB78DeicticRerunAfterToggleOn:
    """B78: off-refused single command survives toggle-ON for deictic re-run."""

    def test_off_refused_single_command_survives_toggle_on_and_reruns(self, monkeypatch):
        cmd = "minimize all the windows on my screen except for the brave window"
        r1 = screen_control.maybe_handle_screen_control_message(cmd)
        assert "Screen controls are off" in r1
        assert screen_state.get_last_request()["command"] == cmd

        r2 = screen_control.maybe_handle_screen_control_message("turn on screen controls")
        assert r2 == "Screen controls are on, sir."
        assert screen_state.is_enabled() is True
        assert screen_state.get_last_request()["command"] == cmd
        assert brain._deictic_rerun_source() == cmd

        live = {
            101: {"hwnd": 101, "title": "YouTube - Brave", "iconic": False,
                  "rect": {"left": 0, "top": 0, "width": 800, "height": 600},
                  "process": "brave", "window_class": "Chrome_WidgetWin_1"},
            202: {"hwnd": 202, "title": "Windows PowerShell", "iconic": False,
                  "rect": {"left": 50, "top": 50, "width": 600, "height": 400},
                  "process": "powershell", "window_class": "ConsoleWindowClass"},
            303: {"hwnd": 303, "title": "Command Prompt", "iconic": False,
                  "rect": {"left": 100, "top": 100, "width": 600, "height": 400},
                  "process": "cmd", "window_class": "ConsoleWindowClass"},
        }
        monkeypatch.setattr(
            screen_control,
            "_list_managable_windows",
            lambda: list(live.values()),
        )

        def _fake_exec(steps, **kw):
            for s in steps or []:
                try:
                    h = int((s or {}).get("hwnd") or 0)
                except (TypeError, ValueError):
                    continue
                if h in live:
                    live[h] = dict(live[h], iconic=True)

        monkeypatch.setattr(screen_control, "execute_steps", _fake_exec)
        monkeypatch.setattr(
            screen_control, "_collect_window_plan_leftovers", lambda plan, steps: []
        )

        r3 = brain._process_message_inner("now do that task")
        assert r3 == "Minimizing 2 windows, sir. Leaving Brave."

    def test_toggle_off_still_clears_last_request_and_refused_chain_keeps_priority(self):
        screen_state.set_enabled(True, reason="test")
        screen_state.note_request("minimize all windows")
        r_off = screen_control.maybe_handle_screen_control_message("turn off screen controls")
        assert r_off == "Screen controls are off, sir."
        assert screen_state.get_last_request()["command"] == ""
        assert brain._deictic_rerun_source() == ""

        # Refused chain takes priority over a single request when both exist.
        screen_state.note_request("minimize notepad")
        screen_state.note_refused_chain("click search bar then type hello")
        screen_control.maybe_handle_screen_control_message("turn on screen controls")
        assert brain._deictic_rerun_source() == "click search bar then type hello"


class TestB79WindowRecountEnumeration:
    """B79: per-HWND drop diagnostics and fail-open revalidation on transient filters."""

    def test_transient_cloak_or_off_monitor_revalidates_fail_open(self, monkeypatch):
        hwnd_list = [11, 22, 33]
        titles = {
            11: "YouTube - Brave",
            22: r"C:\Users\mayan\jarvis-assistant - cmd.exe",
            33: r"C:\Users\mayan\jarvis-assistant - powershell.exe",
        }
        classes = {
            11: "Chrome_WidgetWin_1",
            22: "ConsoleWindowClass",
            33: "CASCADIA_HOSTING_WINDOW_CLASS",
        }
        cloak_calls = {22: 0}
        mon_calls = {33: 0}

        class FakeUser32:
            def EnumWindows(self, cb, lparam):
                for h in hwnd_list:
                    cb(h, lparam)
                return 1

            def IsWindowVisible(self, hwnd):
                return 1

            def GetWindowTextW(self, hwnd, buf, length):
                val = titles.get(int(hwnd), "")
                buf.value = val
                return len(val)

            def GetClassNameW(self, hwnd, buf, length):
                val = classes.get(int(hwnd), "")
                buf.value = val
                return len(val)

            def GetWindowLongW(self, hwnd, idx):
                # HWND 22 reports WS_EX_TOOLWINDOW, but is ConsoleWindowClass.
                return 0x80 if int(hwnd) == 22 else 0

            def IsIconic(self, hwnd):
                return 0

            def MonitorFromWindow(self, hwnd, flags):
                h = int(hwnd)
                if h == 33:
                    mon_calls[33] += 1
                    return 0 if mon_calls[33] == 1 else 1
                return 1

            def GetWindowRect(self, hwnd, rect_ptr):
                r = ctypes.cast(rect_ptr, ctypes.POINTER(ctypes.wintypes.RECT)).contents
                r.left, r.top, r.right, r.bottom = 10, 10, 510, 410
                return 1

            def GetWindowThreadProcessId(self, hwnd, pid_ptr):
                return 1

        class FakeDwmapi:
            def DwmGetWindowAttribute(self, hwnd, attr, ptr, size):
                h = int(hwnd.value if hasattr(hwnd, "value") else hwnd)
                val_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.wintypes.DWORD))
                if h == 22:
                    cloak_calls[22] += 1
                    val_ptr.contents.value = 1 if cloak_calls[22] == 1 else 0
                else:
                    val_ptr.contents.value = 0
                return 0

        monkeypatch.setattr(ctypes.windll, "user32", FakeUser32())
        monkeypatch.setattr(ctypes.windll, "dwmapi", FakeDwmapi())
        monkeypatch.setattr(screen_control, "_process_image_name", lambda pid, cache: "")

        wins = screen_control._list_managable_windows()
        assert [w["hwnd"] for w in wins] == [11, 22, 33]
        assert screen_control._get_last_window_enum_drops() == []

    def test_persistent_ghost_is_dropped_and_recorded_in_diagnostics(self, monkeypatch):
        class FakeUser32:
            def EnumWindows(self, cb, lparam):
                cb(99, lparam)
                return 1

            def IsWindowVisible(self, hwnd):
                return 1

            def GetWindowTextW(self, hwnd, buf, length):
                buf.value = "Suspended Ghost App"
                return len(buf.value)

            def GetClassNameW(self, hwnd, buf, length):
                buf.value = "ApplicationFrameWindow"
                return len(buf.value)

            def GetWindowLongW(self, hwnd, idx):
                return 0

            def IsIconic(self, hwnd):
                return 0

            def MonitorFromWindow(self, hwnd, flags):
                return 0

        monkeypatch.setattr(ctypes.windll, "user32", FakeUser32())
        wins = screen_control._list_managable_windows()
        assert wins == []
        drops = screen_control._get_last_window_enum_drops()
        assert len(drops) == 1
        assert drops[0]["hwnd"] == 99
        assert drops[0]["reason"] == "off_monitor"


class TestB80WrongBarReadbackIdentity:
    """B80: read_control_values exposes control name and _verify_not_wrong_bar catches address bar."""

    def test_read_control_values_includes_name_and_wrong_bar_fires(self, monkeypatch):
        monkeypatch.setattr(screen_ui_elements, "is_available", lambda: True)
        monkeypatch.setattr(
            screen_ui_elements,
            "get_foreground_window_elements",
            lambda hwnd=None: [
                {
                    "uid": "obs1:1",
                    "runtime_id": "rid-omnibox",
                    "control_type": "Edit",
                    "name": "Address and search bar",
                    "focused": True,
                },
                {
                    "uid": "obs1:2",
                    "runtime_id": "rid-yt-search",
                    "control_type": "Edit",
                    "name": "Search",
                    "focused": False,
                },
            ],
        )
        screen_ui_elements._element_cache.clear()
        screen_ui_elements._element_cache["obs1:1"] = {
            "wrapper": object(),
            "name": "Address and search bar",
        }
        screen_ui_elements._element_cache["obs1:2"] = {
            "wrapper": object(),
            "name": "Search",
        }
        monkeypatch.setattr(
            screen_ui_elements,
            "_read_wrapper_value",
            lambda w: "mr.beast" if w is screen_ui_elements._element_cache["obs1:1"]["wrapper"] else "",
        )

        controls = screen_ui_elements.read_control_values(hwnd=555)
        assert controls[0]["name"] == "Address and search bar"
        assert controls[0]["value"] == "mr.beast"

        steps = [{"action": "type", "text": "mr.beast", "description": "Type mr.beast"}]
        verdict = screen_control._verify_not_wrong_bar(
            command_text="type mr.beast in it",
            steps_description="Typing mr.beast, sir.",
            steps=steps,
            hwnd=555,
            intent_context="click in the youtube search bar on my screen | type mr.beast in it",
        )
        assert verdict is False


class TestB81UnverifiedTypeStepResolve:
    """B81: verified is None on a type step triggers one clear+retype resolve and honest marker."""

    def test_unverified_type_step_retries_once_with_clear_and_marks_if_still_unverified(
        self, monkeypatch
    ):
        screen_state.set_enabled(True, reason="test_b81")
        monkeypatch.setattr(brain, "_chain_stop_requested", lambda: False)

        planned_calls = []

        def fake_plan_command(text):
            planned_calls.append(text)
            return {
                "ok": True,
                "confidence": 0.95,
                "summary": "Typing mr.beast into the YouTube search bar, sir.",
                "steps": [
                    {"action": "click", "x": 400, "y": 120, "description": "Click search bar"},
                    {"action": "type", "text": "mr.beast", "description": "Type mr.beast"},
                ],
                "screen_generation": screen_state.generation(),
            }

        executed_step_lists = []

        def fake_execute_steps(steps, **kwargs):
            executed_step_lists.append(list(steps))

        monkeypatch.setattr(brain, "plan_command", fake_plan_command)
        monkeypatch.setattr(screen_control, "execute_steps", fake_execute_steps)
        # Simulate UIA readback unavailable -> _verify_action_with_steps returns None.
        monkeypatch.setattr(
            screen_control, "_verify_action_with_steps", lambda *a, **kw: None
        )

        chain_plan = {
            "steps": [
                {"kind": "screen_control", "text": "click in the youtube search bar"},
                {"kind": "screen_control", "text": "type mr.beast in it"},
            ],
            "source": "click in the youtube search bar then type mr.beast in it",
        }
        res = brain._mi_screen_control_step(
            {"kind": "screen_control", "text": "type mr.beast in it"},
            chain_plan=chain_plan,
            chain_index=1,
        )

        assert len(planned_calls) == 2
        assert len(executed_step_lists) == 2
        # Retry injected Ctrl+A + Backspace between the leading click and the type step.
        retry_actions = [s["action"] for s in executed_step_lists[1]]
        assert retry_actions == ["click", "hotkey", "press", "type"]
        assert res["status"] == "ok"
        assert "not fully sure it worked" in res["fragment"].lower()

    def test_unverified_type_step_resolves_cleanly_when_second_attempt_verifies(
        self, monkeypatch
    ):
        screen_state.set_enabled(True, reason="test_b81_resolve")
        monkeypatch.setattr(brain, "_chain_stop_requested", lambda: False)
        monkeypatch.setattr(
            brain,
            "plan_command",
            lambda text: {
                "ok": True,
                "confidence": 0.95,
                "summary": "Typing mr.beast into the YouTube search bar, sir.",
                "steps": [{"action": "type", "text": "mr.beast", "description": "Type mr.beast"}],
                "screen_generation": screen_state.generation(),
            },
        )
        monkeypatch.setattr(screen_control, "execute_steps", lambda *a, **kw: None)
        verdicts = iter([None, True])
        monkeypatch.setattr(
            screen_control,
            "_verify_action_with_steps",
            lambda *a, **kw: next(verdicts),
        )

        res = brain._mi_screen_control_step(
            {"kind": "screen_control", "text": "type mr.beast in it"},
            chain_plan={"steps": [], "source": "type mr.beast in it"},
            chain_index=0,
        )
        assert res["status"] == "ok"
        assert res["fragment"] == "Typing mr.beast into the YouTube search bar, sir."
        assert "not fully sure" not in res["fragment"].lower()
