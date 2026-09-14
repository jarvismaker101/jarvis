"""Windows connector for active-window state and UI fallback actions."""

import ctypes
from ctypes import wintypes

from backend.services import screen_ui_elements


user32 = ctypes.windll.user32


def _get_window_title(hwnd):
    buffer = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, buffer, len(buffer))
    return buffer.value.strip()


def get_active_window():
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return {"hwnd": 0, "title": "", "bounds": None}

    rect = wintypes.RECT()
    bounds = None
    if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        bounds = {
            "left": int(rect.left),
            "top": int(rect.top),
            "right": int(rect.right),
            "bottom": int(rect.bottom),
            "width": int(rect.right - rect.left),
            "height": int(rect.bottom - rect.top),
        }

    return {"hwnd": int(hwnd), "title": _get_window_title(hwnd), "bounds": bounds}


def snapshot(max_elements=20):
    active = get_active_window()
    elements = []
    if screen_ui_elements.is_available():
        try:
            raw_elements = screen_ui_elements.get_foreground_window_elements()
            for element in raw_elements[:max_elements]:
                elements.append(
                    {
                        "name": element.get("name", ""),
                        "control_type": element.get("control_type", ""),
                        "enabled": element.get("enabled", True),
                        "automation_id": element.get("automation_id", ""),
                        "class_name": element.get("class_name", ""),
                    }
                )
        except Exception:
            elements = []

    return {
        "connector": "windows",
        "available": True,
        "active_window": active,
        "uia_available": screen_ui_elements.is_available(),
        "visible_controls": elements,
        "capabilities": [
            "windows.inspect_active_window",
            "windows.screen_action",
            "windows.hotkey",
        ],
    }


def perform_screen_action(command):
    """Run a screen action through the connector.

    F19-enforcement: the connector used to bypass the top-level enabled
    check entirely, so a plan could reach execute_steps while screen control
    was off. The authoritative gate lives in screen_state.guard(), and it is
    applied here as well as inside _execute_or_queue.
    """
    from backend.services import screen_control
    from backend.services import screen_state

    allowed, reason = screen_state.guard()
    if not allowed:
        return "Screen controls are off. Say turn on screen controls first."

    # F19: capture the control generation BEFORE planning begins. Stamping it
    # after _plan_with_tree() returned let a plan that was produced while the
    # user was disabling control adopt the new generation and execute.
    request_generation = screen_state.generation()

    direct_plan = screen_control._build_direct_plan(command)
    if isinstance(direct_plan, dict):
        direct_plan["screen_generation"] = request_generation
        return screen_control._execute_or_queue(direct_plan, command)

    plan = screen_control._plan_with_tree(command)
    if isinstance(plan, dict):
        plan["screen_generation"] = request_generation
    return screen_control._execute_or_queue(plan, command)
