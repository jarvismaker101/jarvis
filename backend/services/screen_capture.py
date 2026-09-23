import base64
import ctypes
import importlib
import io
import logging
import time
from ctypes import wintypes

from PIL import Image, ImageDraw, ImageFont

from backend.services import screen_geometry
from backend.services import tool_policy


user32 = ctypes.windll.user32
_dpi_initialized = False

#: F43: the last NON-Jarvis foreground window seen before the overlay/UI took
#: focus. A screen request that arrives while our own overlay owns the
#: foreground must still target what the user was actually looking at.
#: Only the *identity* (hwnd + process) is trusted from here; geometry is
#: always re-acquired live so a moved/resized/closed window cannot be
#: described by stale bounds.
_last_user_target = {"hwnd": None, "title": "", "process_id": None}

#: Window-title fragments that identify Jarvis's own UI (never a user target).
_OWN_WINDOW_MARKERS = ("jarvis", "jarvis overlay", "jarvis-assistant")

#: F43: monotonically increasing capture epoch. Every capture carries the epoch
#: it was taken in, so a consumer can tell whether a frame, a UIA tree and a
#: plan all describe the same observation instead of silently mixing frames.
_capture_epoch = 0


def current_capture_epoch():
    """F43: the epoch of the most recent capture."""
    return _capture_epoch


def _next_capture_epoch():
    global _capture_epoch
    _capture_epoch += 1
    return _capture_epoch


def _get_mss():
    try:
        return importlib.import_module("mss")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Screen capture requires the 'mss' package in backend\\venv. "
            "Install backend requirements before using screen controls."
        ) from exc


def _ensure_dpi_awareness():
    global _dpi_initialized
    if _dpi_initialized:
        return
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
    _dpi_initialized = True


def _get_window_title(hwnd):
    buffer = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, buffer, len(buffer))
    return buffer.value.strip()


def dpi_scale_for_window(hwnd):
    """Per-window DPI scale (F43). 1.0 when unknown; physical px == logical*scale.

    Jarvis runs DPI-aware, so captured pixels are already physical; the scale
    is recorded so a consumer can convert to CSS/DIP pixels when needed.
    """
    try:
        get_dpi = ctypes.windll.user32.GetDpiForWindow
    except Exception:
        return 1.0
    try:
        if not hwnd:
            return 1.0
        dpi = int(get_dpi(hwnd))
        if dpi <= 0:
            return 1.0
        return round(dpi / 96.0, 3)
    except Exception:
        return 1.0



def _window_bounds(hwnd):
    """Explicit bounds for *hwnd* (F42: reacquire a KNOWN window, not just the
    current foreground). Returns None when the window is gone or minimised to
    nothing."""
    if not hwnd:
        return None
    rect = wintypes.RECT()
    try:
        DWMWA_EXTENDED_FRAME_BOUNDS = 9
        ctypes.windll.dwmapi.DwmGetWindowAttribute(
            hwnd, DWMWA_EXTENDED_FRAME_BOUNDS,
            ctypes.byref(rect), ctypes.sizeof(rect)
        )
    except Exception:
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None

    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return None

    # F42: the target is identified by HWND *and* process identity, so a
    # window that was closed and REPLACED by a new one reusing the same HWND
    # is detected instead of being treated as the original target.
    process_id = 0
    try:
        _thread_id = wintypes.DWORD()
        _pid = wintypes.DWORD()
        ctypes.windll.user32.GetWindowThreadProcessId(
            hwnd, ctypes.byref(_pid)
        )
        process_id = int(_pid.value)
    except Exception:
        process_id = 0

    return {
        "left": int(rect.left),
        "top": int(rect.top),
        "width": int(width),
        "height": int(height),
        "window_title": _get_window_title(hwnd),
        "hwnd": hwnd,
        "process_id": process_id,
    }


def get_window_bounds(hwnd):
    """Public, explicit-HWND bounds lookup (F42)."""
    _ensure_dpi_awareness()
    return _window_bounds(hwnd)


def is_own_window(hwnd):
    """True when *hwnd* is one of Jarvis's own UI windows (F43)."""
    if not hwnd:
        return False
    title = (_get_window_title(hwnd) or "").strip().lower()
    return any(marker in title for marker in _OWN_WINDOW_MARKERS)


def note_foreground_target():
    """Record the last non-Jarvis foreground window (F43).

    Called before Jarvis's overlay takes focus so a subsequent screen request
    can still resolve "the window the user was looking at". F43: only the
    identity is retained — geometry is re-acquired live on every use.
    """
    _ensure_dpi_awareness()
    try:
        hwnd = user32.GetForegroundWindow()
    except Exception:
        return None
    if not hwnd or is_own_window(hwnd):
        return None
    _last_user_target["hwnd"] = int(hwnd)
    _last_user_target["title"] = _get_window_title(hwnd)
    _last_user_target["process_id"] = _window_process_id(hwnd)
    return last_foreground_target()


def _window_process_id(hwnd):
    try:
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(pid.value)
    except Exception:
        return 0


def last_foreground_target():
    """The last recorded non-Jarvis foreground target, with LIVE geometry.

    F43: the cached bounds used to be handed out verbatim, so a target that
    had since moved, resized or closed was still described by the coordinates
    it had when it was recorded. The identity (hwnd/process) is what we keep;
    the geometry is re-acquired now, and a target that is gone or has been
    replaced is reported as unavailable instead of being faked.
    """
    hwnd = _last_user_target.get("hwnd")
    if hwnd is None:
        return None
    bounds = _window_bounds(hwnd)
    if not bounds:
        return None
    expected_pid = _last_user_target.get("process_id")
    actual_pid = bounds.get("process_id")
    if expected_pid and actual_pid and int(expected_pid) != int(actual_pid):
        # The HWND was recycled by a different program — not our target.
        _last_user_target["hwnd"] = None
        return None
    _last_user_target["title"] = bounds.get("window_title", "")
    return {
        "hwnd": int(hwnd),
        "title": bounds.get("window_title", ""),
        "bounds": dict(bounds),
        "process_id": actual_pid,
    }


def _get_foreground_window_bounds():
    _ensure_dpi_awareness()
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None

    # F43: Jarvis's own overlay may own the foreground when a voice request
    # arrives. Never capture our own window as "the screen the user means" —
    # fall back to the last non-Jarvis target that was recorded, with its
    # geometry RE-ACQUIRED live (a stale cached rect used to be returned).
    if is_own_window(hwnd):
        recorded = last_foreground_target()
        if recorded and recorded.get("bounds"):
            return dict(recorded["bounds"])
        return None

    bounds = _window_bounds(hwnd)
    if bounds and not is_own_window(hwnd):
        _last_user_target["hwnd"] = int(hwnd)
        _last_user_target["title"] = bounds.get("window_title", "")
        _last_user_target["process_id"] = bounds.get("process_id")
    return bounds



def _capture_region(region):
    mss = _get_mss()
    with mss.mss() as sct:
        shot = sct.grab(region)
        image = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        return image


def _get_monitors():
    """The mss monitor list (index 0 = virtual desktop union) — F43."""
    mss = _get_mss()
    with mss.mss() as sct:
        return [dict(monitor) for monitor in sct.monitors]


def _monitor_region(monitor, index=None):
    return {
        "left": int(monitor["left"]),
        "top": int(monitor["top"]),
        "width": int(monitor["width"]),
        "height": int(monitor["height"]),
        "window_title": "",
        "hwnd": None,
        "monitor_index": index,
        "coordinate_space": screen_geometry.COORD_SCREEN_PIXELS,
    }


def _get_primary_monitor_region():
    monitors = _get_monitors()
    index = 1 if len(monitors) > 1 else 0
    return _monitor_region(monitors[index], index)


def monitor_region_for_point(x, y):
    """The monitor containing the point (x, y) (F43).

    Multi-monitor and negative origins are handled explicitly; a point outside
    every monitor falls back to the primary.
    """
    monitors = _get_monitors()
    index = screen_geometry.monitor_index_for_point(monitors, x, y)
    if index is None:
        return _get_primary_monitor_region()
    return _monitor_region(monitors[index], index)



def _resize_for_vision(image, max_side=2560):
    width, height = image.size
    largest_side = max(width, height)
    if largest_side <= max_side:
        return image

    scale = max_side / float(largest_side)
    resized = image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.LANCZOS,
    )
    return resized


def _draw_grid_overlay(image, grid_step=100):
    """Draw subtle ruler tick marks along the top and left edges.

    The marks give the vision model spatial reference points without
    obscuring actual UI content.
    """
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    width, height = overlay.size

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    tick_color = (255, 40, 40, 180)
    label_color = (255, 40, 40, 200)
    tick_len = 10

    # Small frame annotation at origin gives the model in-image ground truth for the coordinate frame.
    try:
        small_font = ImageFont.truetype("arial.ttf", 10)
    except Exception:
        small_font = font
    draw.text((2, 1), "(0,0)", fill=label_color, font=small_font)
    # tiny rightward x arrow, subtle and top-left only (does not span content)
    draw.line([(22, 6), (30, 6)], fill=tick_color, width=1)
    draw.line([(28, 4), (30, 6), (28, 8)], fill=tick_color, width=1)

    # Normalized 0..1000 ruler: ticks/labels every 10 percent (100..900)
    for i in range(1, 10):
        label = str(i * 100)
        x = int(round(width * i / 10.0))
        # clamp x to drawable area (avoid overflow at far edge)
        if x >= width:
            x = width - 1
        draw.line([(x, 0), (x, tick_len)], fill=tick_color, width=1)
        draw.text((min(x + 2, width - 18), 1), label, fill=label_color, font=font)

    for i in range(1, 10):
        label = str(i * 100)
        y = int(round(height * i / 10.0))
        if y >= height:
            y = height - 1
        draw.line([(0, y), (tick_len, y)], fill=tick_color, width=1)
        draw.text((1, min(y + 2, height - 16)), label, fill=label_color, font=font)

    return overlay


def draw_som_overlay(image, elements):
    """Draw Set-of-Mark (SoM) numbered bounding boxes over UI/OCR elements.
    
    This drastically improves Vision Language Model spatial accuracy.
    """
    if not elements:
        return image

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")

    try:
        font = ImageFont.truetype("arialbd.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for el in elements:
        el_id = el.get("id")
        if el_id is None:
            continue
            
        left, top = el.get("left", 0), el.get("top", 0)
        right, bottom = el.get("right", 0), el.get("bottom", 0)
        
        # Determine color based on source type
        is_ui = el.get("source") == "ui"
        box_fill = (40, 150, 255, 40) if is_ui else (40, 255, 100, 40)
        box_outline = (40, 150, 255, 200) if is_ui else (40, 255, 100, 200)
        
        # Draw bounding box
        draw.rectangle([left, top, right, bottom], fill=box_fill, outline=box_outline, width=2)
        
        # Draw ID badge
        tag = f"[{el_id}]"
        
        try:
            bbox = font.getbbox(tag)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = font.getsize(tag)
            
        pad = 2
        tag_bg = (0, 0, 0, 220)
        
        draw.rectangle(
            [left, top, left + text_w + pad*2, top + text_h + pad*2],
            fill=tag_bg
        )
        draw.text((left + pad, top + pad), tag, fill=(255, 255, 255, 255), font=font)

    return overlay


def resolve_ui_hwnd():
    """The window whose UI tree belongs with the current frame (F43).

    The Accessibility walk must describe the SAME observation as the
    screenshot. For a full-screen grab that means the real foreground window —
    but never Jarvis's own overlay, which would otherwise feed the planner a
    tree of our own capsule describing the user's screen.
    """
    _ensure_dpi_awareness()
    try:
        hwnd = user32.GetForegroundWindow()
    except Exception:
        return None
    if not hwnd:
        return None
    if is_own_window(hwnd):
        target = last_foreground_target() or {}
        return target.get("hwnd")
    return int(hwnd)


def _mint_capture_epoch(region):
    """H4: mint this observation's epoch and stamp it on the recorded user
    target, so an envelope can name the CAPTURE's own epoch instead of
    whatever happens to be current when the envelope is written."""
    ep = _next_capture_epoch()
    try:
        if region.get("hwnd") and not is_own_window(region.get("hwnd")):
            _last_user_target["capture_epoch"] = ep
    except Exception:
        pass
    return ep


def _serialize_capture(region, image, capture_mode):
    vision_image = _resize_for_vision(image)
    annotated = _draw_grid_overlay(vision_image)

    buffer = io.BytesIO()
    annotated.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    return {
        "capture_mode": capture_mode,
        "window_title": region["window_title"],
        "origin_left": region["left"],
        "origin_top": region["top"],
        "capture_width": image.width,
        "capture_height": image.height,
        "vision_width": vision_image.width,
        "vision_height": vision_image.height,
        "image_data_url": f"data:image/png;base64,{encoded}",
        "raw_vision_image": vision_image,
        "hwnd": region.get("hwnd"),
        # F43: the window whose Accessibility tree belongs with THIS frame.
        # The screenshot and the UIA walk must be one observation; when the
        # capture is a full-screen grab the tree still has to come from the
        # real foreground window (or the recorded user target when Jarvis's
        # overlay owns focus) rather than from "whatever is foreground now"
        # at some later moment during planning.
        "ui_hwnd": region.get("hwnd") or resolve_ui_hwnd(),
        # F42: the process that owned the target window at capture time. Kept
        # through planning so execution can detect an HWND that was closed and
        # reused by a different program (replacement) instead of clicking on
        # whatever now lives at the old coordinates.
        "process_id": region.get("process_id"),
        # F43: the observation this frame belongs to. A UIA tree, an OCR pass
        # and a plan must all be derived from ONE epoch; mixing frames is how a
        # click intended for the screenshot's window landed on another one.
        "capture_epoch": _mint_capture_epoch(region),
        # F43: an explicit coordinate-space contract. origin_*/capture_* are
        # absolute physical desktop pixels (screen_pixels, may be negative on a
        # multi-monitor desktop); vision_* are the pixels of the image the
        # model sees; image_data_url is in vision_pixels. Every conversion in
        # the stack names which space it is converting from.
        "region_space": screen_geometry.COORD_SCREEN_PIXELS,
        "coordinate_space": screen_geometry.COORD_VISION_PIXELS,
        "monitor_index": region.get("monitor_index"),
        "dpi_scale": dpi_scale_for_window(region.get("hwnd")),
    }


def capture_active_window():
    _ensure_dpi_awareness()

    region = _get_foreground_window_bounds()
    if region is None:
        logging.warning(
            "No foreground window found — falling back to primary screen capture. "
            "Screen control coordinates may be incorrect if the target window is not fullscreen."
        )
        region = _get_primary_monitor_region()

    image = _capture_region(region)
    return _serialize_capture(region, image, "active_window")


def capture_primary_screen():
    _ensure_dpi_awareness()
    region = _get_primary_monitor_region()
    image = _capture_region(region)
    return _serialize_capture(region, image, "primary_screen")


def capture_region_around_cursor(size=320, margin_scale=0.5):
    """Capture a square region centered on the current cursor position.

    Used for questions like "what does this highlighted area mean?" —
    we interpret "this / this area / this thing" as the region nearest
    the cursor. Returns a capture dict plus the region bounds.
    """
    _ensure_dpi_awareness()

    point = wintypes.POINT()
    if not user32.GetCursorPos(ctypes.byref(point)):
        return capture_primary_screen()

    # F43: take the region from the monitor that CONTAINS the cursor (negative
    # origins included), not from the primary monitor.
    monitor = monitor_region_for_point(point.x, point.y)
    m_left, m_top = monitor["left"], monitor["top"]
    m_right, m_bottom = m_left + monitor["width"], m_top + monitor["height"]

    radius = max(80, int(size / 2))
    left = max(m_left, point.x - radius)
    top = max(m_top, point.y - radius)
    right = min(m_right, point.x + radius)
    bottom = min(m_bottom, point.y + radius)
    if right <= left or bottom <= top:
        return capture_primary_screen()

    fg = user32.GetForegroundWindow()
    if is_own_window(fg):
        target = last_foreground_target() or {}
        target_hwnd = target.get("hwnd")
        target_title = target.get("title", "")
    else:
        target_hwnd = fg or None
        target_title = _get_window_title(fg) if fg else ""

    region = {
        "left": left,
        "top": top,
        "width": right - left,
        "height": bottom - top,
        "window_title": target_title,
        "hwnd": target_hwnd,
        "monitor_index": monitor.get("monitor_index"),
        "coordinate_space": screen_geometry.COORD_SCREEN_PIXELS,
        "cursor_x": point.x,
        "cursor_y": point.y,
    }

    image = _capture_region(region)
    capture = _serialize_capture(region, image, "cursor_region")
    capture["region"] = region
    return capture


def capture_for_screen_control(delay_ms=200):
    """Capture with a brief wait so the target window can regain focus after voice input."""
    time.sleep(delay_ms / 1000.0)
    return capture_active_window()


def annotate_capture(capture, elements):
    """Draw the Set-of-Mark numbered boxes and update the image payload."""
    if not elements or "raw_vision_image" not in capture:
        return capture

    annotated = draw_som_overlay(capture["raw_vision_image"], elements)
    # Also draw the grid on top of it so the grid is still visible
    annotated = _draw_grid_overlay(annotated)

    buffer = io.BytesIO()
    annotated.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    
    capture["image_data_url"] = f"data:image/png;base64,{encoded}"
    return capture


# ── F21: remove secrets from the pixels that leave this process ────────────
#: Solid fill used to blank a redacted region. Opaque and uniform, so nothing
#: of the original value survives in the PNG.
_REDACT_FILL = (0, 0, 0)


def redact_image_regions(image, regions, fill=_REDACT_FILL):
    """Blank the rectangles in *regions* on *image*; returns how many applied.

    Coordinates are in the image's own pixel space and are clamped to it.
    """
    applied = 0
    for region in regions or []:
        if not isinstance(region, dict):
            continue
        try:
            x = int(region.get("x"))
            y = int(region.get("y"))
            w = int(region.get("w") or 0)
            h = int(region.get("h") or 0)
        except (TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        box = (max(0, x), max(0, y),
               min(image.width, x + w), min(image.height, y + h))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        try:
            image.paste(fill, box)
            applied += 1
        except Exception:
            continue
    return applied


def refresh_capture_image_data(capture):
    """Re-encode ``image_data_url`` from the (possibly redacted) vision image."""
    image = capture.get("raw_vision_image")
    if image is None:
        return False
    try:
        annotated = _draw_grid_overlay(image)
        buffer = io.BytesIO()
        annotated.save(buffer, format="PNG")
        capture["image_data_url"] = "data:image/png;base64,%s" % (
            base64.b64encode(buffer.getvalue()).decode("ascii"))
        capture["vision_width"] = image.width
        capture["vision_height"] = image.height
        return True
    except Exception:
        return False


def _element_is_sensitive(element):
    """True when a UIA/OCR element's identity says it holds a secret."""
    if not isinstance(element, dict):
        return False
    if element.get("is_password") or element.get("isPassword"):
        return True
    return tool_policy.is_sensitive_field(
        element.get("name"), element.get("text"), element.get("value"),
        element.get("placeholder"), element.get("control_type"),
        element.get("automation_id"), element.get("className"),
    )


def redact_capture(capture, words=None, elements=None, fill=_REDACT_FILL):
    """F21: blank the parts of a screenshot that provably hold a secret.

    Detection is deliberately conservative and evidence-based: OCR text whose
    value matches a credential SHAPE (a key, a bearer token, a ``key=value``
    pair), plus fields whose own identity flags them as password/OTP/card
    inputs. Everything passes through the single ``tool_policy`` redaction
    boundary, so a detection rule only has to be written once.

    The redaction happens on ``raw_vision_image`` BEFORE annotation, so the
    grid and the SoM boxes are drawn on the redacted pixels and the encoded
    data URL is refreshed to match. Returns the number of regions blanked.
    """
    image = capture.get("raw_vision_image")
    if image is None:
        return 0

    regions = []
    for word in words or []:
        if not isinstance(word, dict):
            continue
        text = str(word.get("text") or "")
        if not text:
            continue
        if tool_policy.mask_secrets(text) != text:
            regions.append(word)
    for element in elements or []:
        if _element_is_sensitive(element):
            regions.append(element)

    applied = redact_image_regions(image, regions, fill)
    if applied:
        refresh_capture_image_data(capture)
        capture["redacted_regions"] = applied
    return applied
