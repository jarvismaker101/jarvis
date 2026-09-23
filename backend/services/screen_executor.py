import contextlib
import ctypes
import importlib
import logging
import re
import time
from ctypes import wintypes

user32 = ctypes.windll.user32

_dpi_initialized_executor = False


def _ensure_dpi_awareness():
    """Ensure process is DPI-aware so SetCursorPos/SendInput use physical pixels.

    Mirrors backend.services.screen_capture._ensure_dpi_awareness (prefer reuse
    to avoid drift). Works as side-effect fallback if that module is unavailable.
    """
    global _dpi_initialized_executor
    if _dpi_initialized_executor:
        return
    try:
        from backend.services.screen_capture import _ensure_dpi_awareness as _sc_dpi

        _sc_dpi()
        _dpi_initialized_executor = True
        return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                user32.SetProcessDPIAware()
            except Exception:
                pass
    _dpi_initialized_executor = True


SW_RESTORE = 9
SW_MAXIMIZE = 3
SW_MINIMIZE = 6

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_HWHEEL = 0x01000
MOUSEEVENTF_WHEEL = 0x0800

INPUT_MOUSE = 0


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_ushort),
        ("wParamH", ctypes.c_ushort),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("_input", _INPUT_UNION),
    ]



KEY_ALIASES = {
    "control": "ctrl",
    "return": "enter",
    "escape": "esc",
    "spacebar": "space",
    "windows": "windows",
    "win": "windows",
    "pageup": "page up",
    "page down": "page down",
    "pagedown": "page down",
    "page up": "page up",
    "arrow up": "up",
    "arrow down": "down",
    "arrow left": "left",
    "arrow right": "right",
}


def _get_keyboard():
    try:
        return importlib.import_module("keyboard")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Screen typing and hotkeys require the 'keyboard' package in backend\\venv. "
            "Install backend requirements before using screen controls."
        ) from exc


#: C3: the only shapes that may reach the keyboard library - alias-mapped
#: vocabulary words, function keys, single printable characters. NO '+' -
#: chords are constructed by press_keys itself from separate whitelisted
#: tokens, never parsed out of one raw string.
_SAFE_EXECUTOR_KEY_RE = re.compile(r"^(?:f\d{1,2}|[a-z0-9 _-]{1,16})$")


def _normalize_key_name(key):
    normalized = " ".join((key or "").strip().lower().split())
    if not normalized or not _SAFE_EXECUTOR_KEY_RE.match(normalized):
        # C3: raw model strings never reach press_and_release.
        return None
    return KEY_ALIASES.get(normalized, normalized)


def move_mouse(x, y):
    _ensure_dpi_awareness()
    user32.SetCursorPos(int(x), int(y))


def _get_foreground_window():
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        raise RuntimeError("I couldn't find an active window to control.")
    return hwnd


def minimize_foreground_window():
    hwnd = _get_foreground_window()
    user32.ShowWindow(hwnd, SW_MINIMIZE)


def maximize_foreground_window():
    hwnd = _get_foreground_window()
    user32.ShowWindow(hwnd, SW_MAXIMIZE)


def restore_foreground_window():
    hwnd = _get_foreground_window()
    user32.ShowWindow(hwnd, SW_RESTORE)


def click(x, y, button="left", clicks=1):
    _ensure_dpi_awareness()
    move_mouse(x, y)
    time.sleep(0.08)

    if button == "right":
        down_flag = MOUSEEVENTF_RIGHTDOWN
        up_flag = MOUSEEVENTF_RIGHTUP
    else:
        down_flag = MOUSEEVENTF_LEFTDOWN
        up_flag = MOUSEEVENTF_LEFTUP

    # C3: clicks are bounded here too (the plan gate clamps 1..10; this is the
    # defense-in-depth backstop). A bad type degrades to ONE click, never a
    # ValueError mid-plan.
    try:
        repeat = int(clicks)
    except (TypeError, ValueError):
        repeat = 1
    for _ in range(max(1, min(10, repeat))):
        inp_down = INPUT(type=INPUT_MOUSE)
        inp_down._input.mi.dwFlags = down_flag
        inp_up = INPUT(type=INPUT_MOUSE)
        inp_up._input.mi.dwFlags = up_flag

        user32.SendInput(1, ctypes.byref(inp_down), ctypes.sizeof(INPUT))
        user32.SendInput(1, ctypes.byref(inp_up), ctypes.sizeof(INPUT))
        time.sleep(0.08)


def scroll(direction="down", amount=600):
    _ensure_dpi_awareness()
    delta = abs(int(amount))
    if direction in {"left", "right"}:
        if direction == "left":
            delta *= -1
        user32.mouse_event(MOUSEEVENTF_HWHEEL, 0, 0, delta, 0)
        return

    if direction == "down":
        delta *= -1
    user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, delta, 0)


def type_text(text):
    keyboard = _get_keyboard()
    keyboard.write(text or "", delay=0.02)


def press_keys(keys):
    keyboard = _get_keyboard()
    normalized = [_normalize_key_name(key) for key in keys if key]
    # C3: an unwhitelisted token fails the step loudly instead of being typed.
    if any(k is None for k in normalized):
        raise ValueError("unsafe key token in press/hotkey step")
    normalized = [k for k in normalized if k]
    if not normalized:
        return

    if len(normalized) == 1:
        keyboard.press_and_release(normalized[0])
        return

    keyboard.press_and_release("+".join(normalized))


def _wait_for_foreground(hwnd, timeout=1.0, poll=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if user32.GetForegroundWindow() == hwnd:
            return True
        time.sleep(poll)
    return False


#: Actions whose effect lands wherever the keyboard focus currently is.
_KEYBOARD_ACTIONS = {"type", "press", "hotkey", "scroll"}


def _foreground_is(hwnd):
    try:
        return int(user32.GetForegroundWindow() or 0) == int(hwnd)
    except Exception:
        return False


def _ensure_target_focus(target_hwnd, description):
    """F42: keyboard effects must land in the window the plan targeted.

    Focus can be stolen between steps (a dialog pops up, the user clicks
    elsewhere). Typing then follows the *new* focus — the "later typing can
    follow stolen focus" defect. Re-acquire the planned window once; if it
    cannot be made foreground, stop rather than type into whatever has focus.
    """
    if not target_hwnd:
        return
    if _foreground_is(target_hwnd):
        return
    try:
        if user32.IsIconic(target_hwnd):
            user32.ShowWindow(target_hwnd, SW_RESTORE)
        user32.SetForegroundWindow(target_hwnd)
    except Exception:
        pass
    if not _wait_for_foreground(target_hwnd, timeout=0.6):
        raise RuntimeError(
            "Focus moved to another window before I could %s, sir - I stopped "
            "so the input did not land somewhere else." % description
        )


def _root_window_at_point(x, y):
    """Root (top-level) HWND under a screen point, or 0 when unknown."""
    try:
        point = wintypes.POINT(int(x), int(y))
        hwnd = user32.WindowFromPoint(point)
    except Exception:
        return 0
    if not hwnd:
        return 0
    try:
        GA_ROOT = 2
        root = user32.GetAncestor(hwnd, GA_ROOT)
        return int(root or hwnd)
    except Exception:
        return int(hwnd)


def _verify_hit_test(step, target_hwnd):
    """F42: refuse a click whose point no longer belongs to the target window.

    A stale coordinate is not made safe merely by being on screen: if another
    window now covers it (a dialog, a popup, the user switching apps) the click
    would land there. When the probe itself is unavailable we do not block the
    action — the coordinate is still absolute — but a *negative* answer stops
    the plan.
    """
    if not target_hwnd:
        return
    x, y = step.get("x"), step.get("y")
    if x is None or y is None:
        return
    root = _root_window_at_point(x, y)
    if not root:
        return
    if root != int(target_hwnd):
        raise RuntimeError(
            "Another window moved over the spot I planned to click, sir - "
            "I've stopped rather than click on it. Please ask me again."
        )


class ScreenRevoked(RuntimeError):
    """Raised when the screen-control gate denies an effect mid-plan.

    F19: revocation is authoritative *inside* execution, not only at the
    request boundary. Callers translate this into a spoken refusal and must
    not attempt any further effect.
    """


@contextlib.contextmanager
def _effect_boundary(gate):
    """Enter the authoritative effect boundary for ONE screen effect.

    *gate* is a zero-argument callable returning a context manager (in
    practice ``lambda: screen_state.effect_gate(stamp)``). It is entered
    immediately before the effect, so the revocation check and the effect
    itself are serialized — a disable cannot land between them.

    Any failure to establish authorization fails closed: an unevaluatable
    check must never authorise a keypress or a click.
    """
    if gate is None:
        yield
        return
    try:
        boundary = gate()
    except Exception as exc:  # pragma: no cover - defensive
        raise ScreenRevoked(
            "Screen control authorization could not be verified (%s). I "
            "stopped that action, sir." % (exc,)
        )
    try:
        with boundary:
            yield
    except ScreenRevoked:
        raise
    except PermissionError as exc:
        raise ScreenRevoked(
            "Screen control was revoked before this action ran (%s). I stopped "
            "the rest of it, sir." % (exc,)
        )


def execute_steps(steps, target_hwnd=None, gate=None):
    _ensure_dpi_awareness()
    # F19: the control generation is enforced immediately before the first
    # effect (focus acquisition itself is an effect) and again before every
    # individual step, so disabling screen control mid-plan stops the
    # remainder instead of letting an already-started plan run to completion.
    # If we know which window we planned against, bring it to front first.
    # F42: focus failure STOPS the plan — a click aimed at a window that never
    # reached the foreground would land on whoever did.
    if target_hwnd:
        with _effect_boundary(gate):
            try:
                if user32.IsIconic(target_hwnd):
                    user32.ShowWindow(target_hwnd, SW_RESTORE)
                user32.SetForegroundWindow(target_hwnd)
                if not _wait_for_foreground(target_hwnd):
                    raise RuntimeError(
                        "The target window did not come to the foreground, sir. "
                        "Please ask me again so I can re-plan."
                    )
            except RuntimeError:
                raise
            except ScreenRevoked:
                raise
            except Exception as exc:
                # F42: still treat it as a stop — never continue into a different
                # window's geometry with translated/stale coordinates.
                raise RuntimeError(
                    "I could not bring the target window forward (%s), sir. "
                    "Please bring it up and ask again." % exc
                )
    else:
        with _effect_boundary(gate):
            pass

    for step in steps:
        action = (step.get("action") or "").lower()

        ui_uid = step.get("ui_uid")
        if ui_uid:
            revoked = None
            try:
                from backend.services.screen_ui_elements import invoke_element
                # F19: the UIA revalidation + invoke is itself an effect and
                # runs inside the boundary, so revocation cannot land between
                # the check and the invoke.
                with _effect_boundary(gate):
                    success = invoke_element(
                        ui_uid,
                        action=action,
                        text=step.get("text", ""),
                        hwnd=target_hwnd,
                        runtime_id=step.get("ui_runtime_id", ""),
                    )
                if success:
                    time.sleep(0.12)
                    continue
            except ScreenRevoked as exc:
                revoked = exc
            except Exception as exc:
                logging.warning("UI Automation invoke failed for %s: %s", ui_uid, exc)
            if revoked is not None:
                raise revoked
            # F42: an unresolvable UIA target is a STOP, never a coordinate
            # click on whatever may now sit at the stale point.
            raise RuntimeError(
                "The control this step targets could not be revalidated in the "
                "live accessibility tree. Please ask me again so I can re-plan."
            )

        # F19: every low-level effect runs inside the boundary as well — a
        # revocation that lands between two steps prevents the second one.
        with _effect_boundary(gate):
            # F42: revalidate the live Windows target immediately before the
            # effect. Keyboard input is bound to the planned window's focus and
            # mouse input is hit-tested against it, so a stolen focus or a
            # covering window stops the plan instead of misdirecting input.
            if action in _KEYBOARD_ACTIONS:
                _ensure_target_focus(
                    target_hwnd,
                    "type into the window I planned against"
                    if action == "type" else "send that input",
                )
            elif action in {"click", "double_click", "right_click"}:
                _verify_hit_test(step, target_hwnd)

            if action == "click":
                click(
                    step.get("x"),
                    step.get("y"),
                    button=step.get("button", "left"),
                    clicks=step.get("clicks", 1),
                )
            elif action == "double_click":
                click(step.get("x"), step.get("y"), button="left", clicks=2)
            elif action == "right_click":
                click(step.get("x"), step.get("y"), button="right", clicks=1)
            elif action in {"move", "hover"}:
                move_mouse(step.get("x"), step.get("y"))
            elif action == "minimize_window":
                minimize_foreground_window()
            elif action == "maximize_window":
                maximize_foreground_window()
            elif action == "restore_window":
                restore_foreground_window()
            elif action == "scroll":
                scroll(step.get("direction", "down"), step.get("amount", 600))
            elif action == "type":
                type_text(step.get("text", ""))
            elif action in {"press", "hotkey"}:
                press_keys(step.get("keys", []))
            else:
                # F41: a step the executor cannot perform is NOT a success.
                # Silently skipping it let a plan that did nothing report
                # completion, and let dependent steps run as if it had.
                raise RuntimeError(
                    "I don't know how to perform the screen action '%s', sir - "
                    "I stopped instead of pretending it worked." % (action or "unknown")
                )

        time.sleep(0.12)
