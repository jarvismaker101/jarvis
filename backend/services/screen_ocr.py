"""OCR text extraction from screenshots for screen-control grounding.

Uses pytesseract when available, gracefully degrades to no-op otherwise.
The extracted text labels and their positions are fed into the vision
prompt so the LLM doesn't have to "read" on-screen text itself.
"""

import logging
import os
import re

from backend.services import screen_geometry

_HAS_TESSERACT = False
try:
    import pytesseract  # type: ignore

    # Probe common Windows install locations if not on PATH.
    _COMMON_PATHS = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
    ]
    for _path in _COMMON_PATHS:
        if os.path.isfile(_path):
            pytesseract.pytesseract.tesseract_cmd = _path
            break

    pytesseract.get_tesseract_version()
    _HAS_TESSERACT = True
except Exception:
    pass


def is_available():
    """Return True if OCR extraction is usable."""
    return _HAS_TESSERACT


def extract_text_regions(pil_image):
    """Return a list of text regions detected in *pil_image*.

    Each region is a dict with keys:
        text, left, top, right, bottom, confidence
    Coordinates are in the image's own pixel space.
    F44: ``block_num`` / ``par_num`` / ``line_num`` — the raw Tesseract
    structural identity of every word — are carried through so downstream
    merging (``_merge_nearby_words``) and deduplication never confuse two
    repeated labels that happen to share text.
    """
    if not _HAS_TESSERACT:
        return []

    try:
        data = pytesseract.image_to_data(
            pil_image, output_type=pytesseract.Output.DICT
        )
    except Exception as exc:
        logging.warning("OCR extraction failed: %s", exc)
        return []

    def _as_int(values, index, default=0):
        try:
            return int(values[index])
        except Exception:
            return default

    regions = []
    n = len(data.get("text", []))
    for i in range(n):
        raw = (data["text"][i] or "").strip()
        if not raw or len(raw) < 2:
            continue
        conf = int(data["conf"][i]) if str(data["conf"][i]) != "-1" else 0
        if conf < 60:
            continue

        left = _as_int(data["left"], i)
        top = _as_int(data["top"], i)
        width = _as_int(data["width"], i)
        height = _as_int(data["height"], i)
        regions.append(
            {
                "text": raw,
                "left": left,
                "top": top,
                "right": left + width,
                "bottom": top + height,
                "confidence": conf,
                "block_num": _as_int(data["block_num"], i),
                "par_num": _as_int(data["par_num"], i),
                "line_num": _as_int(data["line_num"], i),
            }
        )

    return regions


def _same_ocr_line(a, b):
    """F44: do two word boxes carry the same Tesseract structural identity?

    When both boxes carry block/paragraph/line numbers and they disagree, the
    words belong to different lines and must never be merged into one label
    even if their pixel rows happen to align. Missing identity is treated as
    "compatible" so non-Tesseract region sources keep working.
    """
    for key in ("block_num", "par_num", "line_num"):
        av = a.get(key)
        bv = b.get(key)
        if av is None or bv is None:
            continue
        try:
            if int(av) != int(bv):
                return False
        except (TypeError, ValueError):
            continue
    return True


def _merge_nearby_words(regions, x_gap=None):
    """Merge horizontally adjacent word boxes into short phrases.

    F44 fixes two merging defects:

      * **Structural compatibility.** The carried block/paragraph/line
        identity is now *checked*, not merely copied: words from different
        Tesseract lines are never fused just because their rows align.
      * **Bounded, non-negative gaps.** The horizontal gap must be a real
        forward gap inside ``[0, x_gap)``. A negative gap means the boxes
        overlap or arrive out of reading order; merging those would fuse two
        distinct controls into a single label.

    The merged phrase keeps the *first* word's structural identity, so an
    identical label on another line or block stays a distinct spatial entity.
    """
    if not regions:
        return regions

    if x_gap is None:
        avg_h = sum(r["bottom"] - r["top"] for r in regions) / len(regions)
        x_gap = max(12, int(avg_h * 0.6))

    sorted_r = sorted(regions, key=lambda r: (r["top"], r["left"]))
    merged = []
    current = dict(sorted_r[0])

    for r in sorted_r[1:]:
        same_line = abs(r["top"] - current["top"]) < max(
            8, (current["bottom"] - current["top"]) * 0.5
        )
        try:
            gap = int(r["left"]) - int(current["right"])
        except (TypeError, ValueError):
            gap = x_gap  # unmeasurable → do not merge
        close_x = 0 <= gap < x_gap
        compatible = _same_ocr_line(current, r)

        if same_line and close_x and compatible:
            current["text"] += " " + r["text"]
            current["right"] = max(current["right"], r["right"])
            current["bottom"] = max(current["bottom"], r["bottom"])
            current["confidence"] = min(current["confidence"], r["confidence"])
            # The merged phrase belongs to the region where reading started;
            # its block/paragraph/line identity is already carried by
            # ``current`` (copied from the first word).
        else:
            merged.append(current)
            current = dict(r)

    merged.append(current)
    return merged


def format_for_prompt(regions, max_items=30):
    """Format OCR regions into a compact string for the vision prompt.

    Merges adjacent words and limits output length. F44: deduplication is
    location-aware — two identical labels at distinct positions are both kept
    (repeated labels like two "Edit" buttons), only overlapping duplicates of
    the same physical region collapse.
    """
    if not regions:
        return ""

    merged = _merge_nearby_words(regions)
    merged.sort(key=lambda r: (r["top"], r["left"]))

    # Location-aware dedupe: skip only when the same text overlaps a region
    # we already emitted.
    kept = []
    for r in merged:
        key = re.sub(r"\s+", " ", r["text"].lower().strip())
        if not key:
            continue
        duplicate = False
        for other in kept:
            other_key = re.sub(
                r"\s+", " ", str(other.get("text", "")).lower().strip()
            )
            if other_key == key and screen_geometry.rects_duplicate(r, other):
                duplicate = True
                break
        if not duplicate:
            kept.append(r)

    items = kept[:max_items]

    lines = ["Visible text detected on screen (OCR):"]
    for r in items:
        cx = (r["left"] + r["right"]) // 2
        cy = (r["top"] + r["bottom"]) // 2
        lines.append(f'- "{r["text"]}" near ({cx}, {cy})')

    return "\n".join(lines) + "\n"
