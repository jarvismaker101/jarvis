"""Windows UI Automation element extraction for screen-control grounding.

Uses pywinauto (UIA backend) when available, gracefully degrades to no-op.
The extracted element names, types, and bounding boxes are fed into the
vision prompt so the LLM has a structured map of what is clickable.
"""

import ctypes
import logging
import time

_pywinauto = None
_IMPORT_ATTEMPTED = False
_IMPORT_ERROR = None

_element_cache = {}
_CACHE_TTL_SECONDS = 8.0
#: F42: how long invoke_element re-validates through a FRESH UIA resolution
#: before giving up (never a silent coordinate fallback).
_RESOLVE_RETRY_SECONDS = 2.0

#: F42/F29: the window the cache was most recently walked from. The UIA-first
#: fast path must know which HWND produced the elements it is matching.
_last_hwnd = None

#: F42: the HWND (and process) the *current* cache contents belong to. Every
#: record is scoped to it so a wrapper from one window can never satisfy a
#: lookup for another.
_cache_hwnd = None
_cache_process_id = None

#: F44: every walk gets an observation id and its own monotonic counter, so
#: element uids are observation-scoped identifiers rather than ``str(id(obj))``
#: — CPython recycles object ids, which could silently alias a stale element to
#: a live one.
_observation_seq = 0
_observation_id = 0
_walk_counter = 0


def current_observation_id():
    """The id of the most recent accessibility observation (F44)."""
    return _observation_id


def _window_process_id(hwnd):
    try:
        pid = ctypes.c_ulong(0)
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(pid.value)
    except Exception:
        return 0


user32 = ctypes.windll.user32

# Element types worth surfacing to the vision model.
_USEFUL_CONTROL_TYPES = {
    "Button",
    "CheckBox",
    "ComboBox",
    "DataItem",
    "Document",
    "Edit",
    "Group",
    "Hyperlink",
    "Image",
    "ListItem",
    "Menu",
    "MenuItem",
    "Pane",
    "RadioButton",
    "ScrollBar",
    "Slider",
    "Tab",
    "TabItem",
    "Text",
    "ToolBar",
    "TreeItem",
}


def is_available():
    """Return True if UI Automation extraction is usable."""
    return _load_pywinauto() is not None


def current_hwnd():
    """HWND whose accessibility subtree the cache was most recently built from.

    F29: lets the UIA-first fast path name the window it matched against
    without re-deriving the foreground target.
    """
    return _last_hwnd


def _load_pywinauto():
    """Import pywinauto lazily so COM-init issues don't break startup."""
    global _pywinauto, _IMPORT_ATTEMPTED, _IMPORT_ERROR

    if _pywinauto is not None:
        return _pywinauto
    if _IMPORT_ATTEMPTED:
        return None

    _IMPORT_ATTEMPTED = True
    try:
        import pywinauto  # type: ignore

        _pywinauto = pywinauto
        _IMPORT_ERROR = None
        return _pywinauto
    except Exception as exc:
        _IMPORT_ERROR = exc
        logging.warning("UI Automation disabled: %s", exc)
        return None


def get_foreground_window_elements(max_elements=140, max_depth=9, hwnd=None):
    """Return a list of UI element dicts from the active window.

    Each dict has keys:
        control_type, name, left, top, right, bottom, enabled, depth,
        automation_id, class_name, uid, parent_uid, runtime_id
    Coordinates are in screen pixel space and MAY BE NEGATIVE on a
    multi-monitor desktop (a monitor left of / above the primary) — F43.
    Returns an empty list if pywinauto is not installed or on error.
    """
    global _last_hwnd, _cache_hwnd, _cache_process_id, _observation_seq, _observation_id, _walk_counter
    pywinauto = _load_pywinauto()
    if pywinauto is None:
        return []

    if hwnd is None:
        hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return []
    _last_hwnd = int(hwnd)
    _cache_hwnd = int(hwnd)
    _cache_process_id = _window_process_id(hwnd)
    # F44: a fresh observation namespace per walk.
    _observation_seq += 1
    _observation_id = _observation_seq
    _walk_counter = 0

    try:
        app = pywinauto.Application(backend="uia").connect(
            handle=hwnd, timeout=2
        )
        window = app.window(handle=hwnd)
    except Exception as exc:
        logging.warning("UI Automation connect failed: %s", exc)
        return []

    elements = []
    _element_cache.clear()
    try:
        _walk(window.wrapper_object(), elements, max_elements, max_depth,
              depth=0, parent_uid="")
    except Exception as exc:
        logging.warning("UI Automation walk failed: %s", exc)

    return elements


def _walk(wrapper, out, max_elements, max_depth, depth, parent_uid=""):
    """Recursively collect useful child elements (F43/F44).

    Each emitted element carries:
      * ``parent_uid`` — the uid of its parent **in this same observation**,
        preserving the original hierarchy (F44);
      * ``runtime_id`` / legacy ``automation_id`` — the semantic runtime
        identity used to resolve the element again after the cache expires
        (F42), doing away with a bare object that rots after the TTL;
      * ``observation_id`` — the observation this element was seen in, so a
        parent/child relationship is scoped to one walk (F44).

    F44 fixes two hierarchy defects:

      1. **No dangling parents.** A container that is not itself interesting
         but *has* surviving descendants is emitted as a ``placeholder``
         element, so ``parent_uid`` can never point at a node the consumer
         never received. The placeholder's bounds are the union of its
         emitted subtree, so it is a real structural node, not a guess.
      2. **No object-identity uids.** Uids come from a per-observation
         monotonic counter rather than ``str(id(child))``.

    Controls whose bounds go NEGATIVE (a monitor left of/above the primary)
    are kept — only truly zero-area ones are skipped (F43); a degenerate box
    still recurses, forwarding the *original* parent so an omitted level
    cannot orphan its children.
    """
    global _walk_counter

    if depth > max_depth or len(out) >= max_elements:
        return

    try:
        children = wrapper.children()
    except Exception:
        return

    for child in children:
        if len(out) >= max_elements:
            return

        try:
            ctype = child.element_info.control_type or ""
        except Exception:
            ctype = ""

        _walk_counter += 1
        child_uid = "obs%d:%d" % (_observation_id, _walk_counter)

        try:
            name = (child.element_info.name or "").strip()
        except Exception:
            name = ""

        try:
            rect = child.rectangle()
            left, top, right, bottom = rect.left, rect.top, rect.right, rect.bottom
        except Exception:
            _walk(child, out, max_elements, max_depth, depth + 1, parent_uid)
            continue

        # F43: keep negative-origin controls (multi-monitor left/above). Only
        # skip a truly degenerate (non-positive-area) rectangle.
        w = right - left
        h = bottom - top
        if w < 5 or h < 5:
            # Not a usable target itself, but its descendants may be. Forward
            # the ORIGINAL parent so the omitted level never becomes a
            # dangling parent reference (F44).
            _walk(child, out, max_elements, max_depth, depth + 1, parent_uid)
            continue

        try:
            enabled = child.is_enabled()
        except Exception:
            enabled = True

        interesting = bool(name) or ctype in {
            "Button", "Edit", "CheckBox", "ComboBox", "Tab", "TabItem"
        }

        # Recurse FIRST: only then do we know whether this container has
        # surviving descendants and therefore must be kept as a structural
        # placeholder to keep the hierarchy connected (F44).
        start = len(out)
        _walk(child, out, max_elements, max_depth, depth + 1, child_uid)
        has_children = len(out) > start

        if not interesting and not has_children:
            continue

        automation_id = ""
        class_name = ""
        runtime_id = ""
        try:
            automation_id = (child.element_info.automation_id or "").strip()
        except Exception:
            automation_id = ""
        try:
            class_name = (child.element_info.class_name or "").strip()
        except Exception:
            class_name = ""
        try:
            rid = child.element_info.runtime_id
            if rid is not None:
                runtime_id = str(rid)
            elif automation_id:
                runtime_id = "%s:%s" % (ctype, automation_id)
        except Exception:
            runtime_id = ("%s:%s" % (ctype, automation_id)) if automation_id else ""

        _element_cache[child_uid] = {
            "created_at": time.monotonic(),
            "wrapper": child,
            # F42: the semantic runtime identity is stored with the wrapper
            # so a cache that expires mid-approval can be rewalked and the
            # SAME control re-resolved instead of clicking stale coords.
            "runtime_id": runtime_id,
            # F42: the cache entry is scoped to the window (and process)
            # it was walked from. A wrapper from window A must never
            # satisfy a lookup aimed at window B, and a replaced window
            # reusing an HWND is detected by the process mismatch.
            "hwnd": _cache_hwnd,
            "process_id": _cache_process_id,
            "control_type": ctype,
            "observation_id": _observation_id,
        }

        # F41: which control currently holds the keyboard focus. A type step
        # without an explicit target lands wherever focus is, so verification
        # needs to know which field to read back.
        try:
            focused = bool(child.has_keyboard_focus())
        except Exception:
            focused = False

        record = {
            "control_type": ctype,
            "name": name,
            "left": int(left),
            "top": int(top),
            "right": int(right),
            "bottom": int(bottom),
            "enabled": bool(enabled),
            "depth": int(depth),
            "automation_id": automation_id,
            "class_name": class_name,
            "uid": child_uid,
            "parent_uid": parent_uid,
            "runtime_id": runtime_id,
            "observation_id": _observation_id,
            "focused": focused,
            #: True for a structural container kept only so its children have
            #: a parent in the serialized tree (F44).
            "placeholder": not interesting,
        }
        if not interesting:
            # Union bounds of the emitted subtree: a real structural node.
            kids = out[start:]
            if kids:
                record["left"] = min(int(k.get("left", left)) for k in kids)
                record["top"] = min(int(k.get("top", top)) for k in kids)
                record["right"] = max(int(k.get("right", right)) for k in kids)
                record["bottom"] = max(int(k.get("bottom", bottom)) for k in kids)

        # Insert BEFORE the subtree we just appended (pre-order preserved).
        out.insert(start, record)


def to_vision_coords(elements, capture):
    """Convert screen coordinates to vision-image coordinates.

    *capture* is the dict returned by screen_capture functions.
    """
    if not elements or not capture:
        return elements

    origin_left = capture.get("origin_left", 0)
    origin_top = capture.get("origin_top", 0)
    cap_w = capture.get("capture_width", 1)
    cap_h = capture.get("capture_height", 1)
    vis_w = capture.get("vision_width", cap_w)
    vis_h = capture.get("vision_height", cap_h)

    x_scale = vis_w / float(cap_w) if cap_w else 1.0
    y_scale = vis_h / float(cap_h) if cap_h else 1.0

    converted = []
    for el in elements:
        left = int(round((el["left"] - origin_left) * x_scale))
        top = int(round((el["top"] - origin_top) * y_scale))
        right = int(round((el["right"] - origin_left) * x_scale))
        bottom = int(round((el["bottom"] - origin_top) * y_scale))

        if right <= 0 or bottom <= 0 or left >= vis_w or top >= vis_h:
            continue

        left = max(0, min(left, vis_w - 1))
        top = max(0, min(top, vis_h - 1))
        right = max(0, min(right, vis_w - 1))
        bottom = max(0, min(bottom, vis_h - 1))
        if right <= left or bottom <= top:
            continue

        converted.append(
            {
                **el,
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
            }
        )
    return converted


def _to_vision_coords(elements, capture):
    """Backward-compatible alias for callers that still use the old name."""
    return to_vision_coords(elements, capture)


def format_for_prompt(elements, capture=None, max_items=25):
    """Format UI elements as a compact string for the vision prompt.

    If *capture* is provided, coordinates are converted to vision-image
    pixel space so they match the screenshot the model sees.
    """
    if not elements:
        return ""

    # F44: structural placeholders exist only to keep the serialized hierarchy
    # connected; they are not candidate targets, so they do not consume the
    # prompt's item budget.
    candidates = [el for el in elements if not el.get("placeholder")]
    items = candidates[:max_items]
    if capture:
        items = to_vision_coords(items, capture)

    lines = ["Active-window UI elements (via Windows UI Automation):"]
    for el in items:
        cx = (el["left"] + el["right"]) // 2
        cy = (el["top"] + el["bottom"]) // 2
        label = f'{el["control_type"]}'
        if el["name"]:
            label += f' "{el["name"]}"'
        status = "enabled" if el["enabled"] else "disabled"
        lines.append(f"- {label} at ({cx}, {cy}), {status}")

    return "\n".join(lines) + "\n"


def _cache_record_is_live(record, hwnd=None, now=None):
    """F42: a cache record is only usable when it is unexpired AND belongs to
    the window we are about to act on.

    Before this check ``_find_cached_by_runtime_id`` returned any matching
    record regardless of age, so a "fresh resolution" could hand back the very
    expired wrapper it was supposed to replace — and a record built from a
    different window could satisfy a lookup for the current target.
    """
    if not isinstance(record, dict):
        return False
    wrapper = record.get("wrapper")
    if wrapper is None:
        return False
    created_at = record.get("created_at")
    if created_at is None:
        return False
    if (now if now is not None else time.monotonic()) - created_at > _CACHE_TTL_SECONDS:
        return False
    if hwnd is not None:
        record_hwnd = record.get("hwnd")
        if record_hwnd is None or int(record_hwnd) != int(hwnd):
            return False
    return True


def _read_wrapper_value(wrapper):
    """F41: the current value/text of one UIA control, or None if unreadable.

    ``None`` is a meaningful answer: it means the postcondition could not be
    observed, so the caller must report uncertainty instead of success.
    """
    if wrapper is None:
        return None
    # ValuePattern first — Edit/ComboBox expose the typed text here.
    try:
        value = wrapper.get_value()
    except Exception:
        value = None
    if value is None:
        try:
            legacy = wrapper.legacy_properties()
        except Exception:
            legacy = None
        if isinstance(legacy, dict):
            value = legacy.get("Value")
    if value is None:
        try:
            value = wrapper.window_text()
        except Exception:
            value = None
    if value is None:
        return None
    try:
        return str(value)
    except Exception:
        return None


def read_control_values(hwnd=None):
    """F41: current value and focus state for every control in ONE fresh walk.

    Returns a list of ``{uid, runtime_id, control_type, focused, value}``.
    ``value`` is None when the control exposes no readable value, which the
    caller must treat as "cannot verify", never as "verified".
    """
    if not is_available():
        return []
    elements = get_foreground_window_elements(hwnd=hwnd)
    out = []
    for elem in elements:
        uid = elem.get("uid") or ""
        record = _element_cache.get(uid) if isinstance(_element_cache, dict) else None
        out.append({
            "uid": uid,
            "runtime_id": elem.get("runtime_id") or "",
            "control_type": elem.get("control_type") or "",
            "focused": bool(elem.get("focused")),
            "value": _read_wrapper_value((record or {}).get("wrapper")),
        })
    return out


def _find_cached_by_runtime_id(runtime_id, hwnd=None, require_live=True):
    """Best cache record whose semantic identity matches.

    F42: *require_live* rejects expired records and records belonging to a
    different HWND. Callers wanting a genuinely fresh resolution must never be
    handed the stale wrapper they already rejected.
    """
    if not runtime_id or not isinstance(_element_cache, dict):
        return None
    now = time.monotonic()
    for record in _element_cache.values():
        if not isinstance(record, dict):
            continue
        if record.get("runtime_id") and record.get("runtime_id") == runtime_id:
            if require_live:
                if _cache_record_is_live(record, hwnd=hwnd, now=now):
                    return record
            elif record.get("wrapper") is not None:
                return record
    return None


def find_element_by_runtime_id(runtime_id, hwnd=None, timeout=_RESOLVE_RETRY_SECONDS):
    """F42: fresh UIA resolution of a target by its semantic runtime identity.

    Runs a NEW accessibility walk (of *hwnd* when known, else the foreground
    window) and matches the stored ``runtime_id``. Returns the cache record
    (with a live wrapper) or None. Never returns stale coordinates, and never
    returns an already-expired cache entry as if it were a fresh resolution.
    """
    if not runtime_id or not is_available():
        return None
    deadline = time.monotonic() + max(0.0, float(timeout or 0.0))
    tries = 0
    while True:
        # Only a LIVE record for the intended window counts as resolution.
        record = _find_cached_by_runtime_id(runtime_id, hwnd=hwnd,
                                            require_live=True)
        if record:
            return record
        # Re-walk the target window; clears/repopulates the cache.
        try:
            get_foreground_window_elements(hwnd=hwnd)
        except Exception as exc:
            logging.debug("Fresh UIA re-walk failed for %s: %s", runtime_id, exc)
            return None
        record = _find_cached_by_runtime_id(runtime_id, hwnd=hwnd,
                                            require_live=True)
        if record:
            return record
        tries += 1
        if tries >= 3 or time.monotonic() >= deadline:
            return None
        time.sleep(0.1)


def invoke_element(uid, action="click", text="", hwnd=None, runtime_id=""):
    """Programmatically trigger a UI Automation wrapper object.

    F42: the target is always re-resolved against the LIVE accessibility tree
    of the planned HWND before anything is attempted:

      * an expired cached wrapper is replaced by a fresh resolution;
      * a cache entry belonging to a different window is never used;
      * a cleared cache (no record at all) triggers a fresh walk when a
        semantic runtime identity is available, instead of silently refusing;
      * a target that cannot be resolved returns False — the caller must NOT
        fall back to stale coordinates.
    """
    entry = _element_cache.get(uid) if uid else None
    if isinstance(entry, dict):
        rewind_rid = runtime_id or entry.get("runtime_id", "")
    else:
        rewind_rid = runtime_id

    if isinstance(entry, dict):
        if not _cache_record_is_live(entry, hwnd=hwnd):
            entry = None

    if entry is None and rewind_rid:
        # Expired, window-mismatched, or cleared cache: resolve again.
        entry = find_element_by_runtime_id(rewind_rid, hwnd=hwnd)
        if entry is None:
            logging.info(
                "UI target uid=%s could not be re-resolved live "
                "(runtime_id=%s) — refusing coordinate fallback.",
                uid, rewind_rid,
            )
            return False

    if isinstance(entry, dict):
        wrapper = entry.get("wrapper")
    elif entry is not None:
        wrapper = entry  # legacy bare wrapper
    else:
        wrapper = None

    if not wrapper:
        return False

    _INVOKABLE_TYPES = {"Button", "Hyperlink", "MenuItem", "CheckBox"}

    try:
        if action in {"click", "double_click", "right_click"}:
            # Only use invoke() on controls that truly support it
            if action == "click" and hasattr(wrapper, "invoke"):
                ctrl_type = ""
                try:
                    ctrl_type = wrapper.element_info.control_type or ""
                except Exception:
                    pass
                if ctrl_type in _INVOKABLE_TYPES:
                    try:
                        wrapper.invoke()
                        return True
                    except Exception:
                        pass

            # Fallback to mouse movement
            if action == "click":
                wrapper.click_input()
            elif action == "double_click":
                wrapper.double_click_input()
            elif action == "right_click":
                wrapper.right_click_input()
            return True

        elif action == "type":
            try:
                wrapper.set_focus()
            except Exception:
                pass
            wrapper.type_keys(text, with_spaces=True)
            return True

    except Exception as exc:
        logging.warning("UI Automation invoke failed for %s: %s", uid, exc)
        return False

    return False
