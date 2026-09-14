"""Explicit coordinate spaces and rectangle math for the Windows screen stack.

Fable-5 audit G7:

  F43 — capture and coordinates are explicit. Every geometry value that
    crosses a boundary (a model's normalized point, a UIA bounding box, the
    screenshot pixels, the physical desktop) now carries an explicit space
    token instead of an implicit assumption, and every conversion happens
    once, in a named adapter. Multi-monitor and mixed-DPI desktops stop
    depending on accidental primary-screen geometry: a control on a monitor
    whose origin is negative (left of / above the primary) is a first-class
    citizen.

  F44 — equal candidates resolve by coordinates *in the same frame*. The
    helpers here let candidate ranking compare two rectangles only when they
    share a coordinate space.

Pure data module: stdlib only, no backend imports (no cycles).
"""

# ── The coordinate-space vocabulary (F43) ──────────────────────────────────
#: Absolute physical desktop pixels. Can legitimately be NEGATIVE on a
#: multi-monitor desktop (a monitor left of / above the primary).
COORD_SCREEN_PIXELS = "screen_pixels"
#: Relative to the target window's top-left (used by some planners).
COORD_WINDOW_PIXELS = "window_pixels"
#: Pixels of the (possibly resized) screenshot the vision model actually sees.
COORD_VISION_PIXELS = "vision_pixels"
#: The model's 0..1000 grid on each axis of the image.
COORD_NORMALIZED_1000 = "normalized_1000"

COORD_SPACES = (
    COORD_SCREEN_PIXELS,
    COORD_WINDOW_PIXELS,
    COORD_VISION_PIXELS,
    COORD_NORMALIZED_1000,
)

#: Accepted spellings a model / step may use, mapped to the canonical token.
_ALIASES = {
    "screen": COORD_SCREEN_PIXELS,
    COORD_SCREEN_PIXELS: COORD_SCREEN_PIXELS,
    "physical": COORD_SCREEN_PIXELS,
    "physical_pixels": COORD_SCREEN_PIXELS,
    "desktop": COORD_SCREEN_PIXELS,
    "window": COORD_WINDOW_PIXELS,
    COORD_WINDOW_PIXELS: COORD_WINDOW_PIXELS,
    "window_relative": COORD_WINDOW_PIXELS,
    "vision": COORD_VISION_PIXELS,
    COORD_VISION_PIXELS: COORD_VISION_PIXELS,
    "image": COORD_VISION_PIXELS,
    "image_pixels": COORD_VISION_PIXELS,
    "pixels": COORD_VISION_PIXELS,
    "normalized": COORD_NORMALIZED_1000,
    COORD_NORMALIZED_1000: COORD_NORMALIZED_1000,
    "0..1000": COORD_NORMALIZED_1000,
    "0-1000": COORD_NORMALIZED_1000,
}


def normalize_space(space, default=COORD_SCREEN_PIXELS):
    """Coerce a space label into the vocabulary (unknown -> *default*)."""
    text = str(space or "").strip().lower()
    return _ALIASES.get(text, default)


# ── Rectangle helpers (F44 same-frame comparisons) ─────────────────────────
def _coerce_rect(rect):
    """Accept {left,top,right,bottom} or {x,y,w,h} and return a normal rect."""
    if not isinstance(rect, dict):
        return None
    if all(k in rect for k in ("left", "top", "right", "bottom")):
        try:
            left = int(rect["left"]); top = int(rect["top"])
            right = int(rect["right"]); bottom = int(rect["bottom"])
        except Exception:
            return None
        return {"left": left, "top": top, "right": right, "bottom": bottom}
    if all(k in rect for k in ("x", "y", "w", "h")):
        try:
            left = int(rect["x"]); top = int(rect["y"])
            width = int(rect["w"]); height = int(rect["h"])
        except Exception:
            return None
        return {"left": left, "top": top, "right": left + width,
                "bottom": top + height}
    return None


def center_of(rect):
    """Center point of *rect*, or None when it is not a usable rectangle."""
    norm = _coerce_rect(rect)
    if not norm:
        return None
    return {
        "x": int(round((norm["left"] + norm["right"]) / 2.0)),
        "y": int(round((norm["top"] + norm["bottom"]) / 2.0)),
    }


def rects_overlap(a, b, pad=0):
    """True when two rectangles intersect (with optional *pad* slop).

    A zero-area rectangle (a single point) still counts as overlapping when it
    lies inside the other, so point-vs-box containment works.
    """
    ra = _coerce_rect(a)
    rb = _coerce_rect(b)
    if not ra or not rb:
        return False
    return (
        ra["left"] - pad < rb["right"] + pad
        and ra["right"] + pad > rb["left"] - pad
        and ra["top"] - pad < rb["bottom"] + pad
        and ra["bottom"] + pad > rb["top"] - pad
    )


def point_in_rect(x, y, rect, pad=0):
    """True when the point (x, y) lies inside *rect* (with optional *pad*)."""
    norm = _coerce_rect(rect)
    if not norm:
        return False
    return (
        norm["left"] - pad <= x <= norm["right"] + pad
        and norm["top"] - pad <= y <= norm["bottom"] + pad
    )


def rects_overlap_area(a, b):
    """Area of the intersection of two rects, or 0 when they do not overlap."""
    ra = _coerce_rect(a)
    rb = _coerce_rect(b)
    if not ra or not rb:
        return 0
    inter_w = min(ra["right"], rb["right"]) - max(ra["left"], rb["left"])
    inter_h = min(ra["bottom"], rb["bottom"]) - max(ra["top"], rb["top"])
    if inter_w <= 0 or inter_h <= 0:
        return 0
    return inter_w * inter_h


def rects_duplicate(a, b, coverage=0.5):
    """F44: two OCR/UIA boxes are the same physical entity when they overlap
    enough that either one is covered by the intersection.

    Same text found in two NON-overlapping (or barely overlapping) places is
    two separate spatial entities — repeated labels must survive, only
    duplicate detections of the *same* on-screen region collapse.
    """
    inter = rects_overlap_area(a, b)
    if inter <= 0:
        return False
    area_a = max(1, rects_overlap_area(a, a))
    area_b = max(1, rects_overlap_area(b, b))
    return inter / area_a >= coverage or inter / area_b >= coverage


def translate_rect(rect, dx, dy):
    """Shift *rect* by (dx, dy); returns a new dict (or None)."""
    norm = _coerce_rect(rect)
    if not norm:
        return None
    return {
        "left": norm["left"] + int(dx),
        "top": norm["top"] + int(dy),
        "right": norm["right"] + int(dx),
        "bottom": norm["bottom"] + int(dy),
    }


# ── Monitor bookkeeping (F43) ──────────────────────────────────────────────
def monitor_bounds(monitor):
    """Bounds dict for one mss-style monitor entry."""
    if not isinstance(monitor, dict):
        return None
    try:
        left = int(monitor["left"])
        top = int(monitor["top"])
        width = int(monitor["width"])
        height = int(monitor["height"])
    except Exception:
        return None
    return {"left": left, "top": top, "right": left + width,
            "bottom": top + height}


def monitor_index_for_point(monitors, x, y):
    """Index of the monitor containing (x, y).

    ``monitors`` is an mss ``sct.monitors`` list: index 0 is the *virtual*
    desktop union, real monitors start at index 1. A point outside every
    monitor falls back to the primary (index 1) or None when there are no
    real monitors. Negative origins are handled normally.
    """
    if not monitors or len(monitors) < 2:
        return None
    for index in range(1, len(monitors)):
        bounds = monitor_bounds(monitors[index])
        if bounds and point_in_rect(x, y, bounds):
            return index
    return 1


def virtual_desktop_bounds(monitors):
    """Union of every monitor — the true bounds of a multi-monitor desktop."""
    if not monitors:
        return None
    return monitor_bounds(monitors[0])

