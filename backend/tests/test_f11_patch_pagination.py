"""F11 (Fable-5 audit): the native coding inspection/edit loop.

Covers the four things the audit's correction/acceptance demanded:

1. **Pagination with no gaps** — a search/read_range page that is bounded for
   display still exposes a cursor at the FIRST UNRETURNED item, so walking the
   pages reproduces the full content exactly once (no gap, no duplicate).
2. **Bounded range reads** — ``code.read_range`` streams the file line by line;
   it never pulls the whole file into memory to return one window.
3. **Structured data feeds planning** — path/hash/lines/matches/cursor come
   back intact and are what later steps resolve ``{{step0.*}}`` against;
   clipping applies to the human rendering only.
4. **Validated patch application** — zero-length insertions land AFTER their
   anchor line, CRLF and no-final-newline files are rebuilt byte-exactly, a
   patch naming another file writes nothing, and any malformed/ambiguous input
   (missing headers, count mismatch, bad hunk header, non-applying hunk) is
   refused with an explicit reason and no write. ``dry_run`` never writes.

Run:  backend\\venv\\Scripts\\python.exe -m unittest backend.tests.test_f11_patch_pagination
"""

import builtins
import copy
import json
import os
import re
import tempfile
import unittest
from unittest.mock import patch

from backend.services import code_grants
from backend.services import code_tools
from backend.services.task_agent import agent


def _write_text(path, text, newline=""):
    with open(path, "w", encoding="utf-8", newline=newline) as handle:
        handle.write(text)


def _write_bytes(path, payload):
    with open(path, "wb") as handle:
        handle.write(payload)


def _read_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


_MATCH_LINE_RE = re.compile(r"^(?P<where>.+?):(?P<line>\d+): (?P<text>.*)$")


def _body_lines(content):
    """The rendered search page items, notes excluded."""
    return [line for line in (content or "").splitlines()
            if _MATCH_LINE_RE.match(line)]


def _parse_match(body_line):
    """``rel/path:line: text`` -> (path, line)."""
    match = _MATCH_LINE_RE.match(body_line)
    return match.group("where"), int(match.group("line"))


class _UnboundedReadSpy:
    """Wraps a text file object and records whole-file reads.

    ``read()`` with no size (or a negative size) and ``readlines()`` are the
    two ways a bounded range read could accidentally slurp the whole file.
    """

    def __init__(self, handle, path, seen):
        self._handle = handle
        self._path = path
        self._seen = seen

    def read(self, size=-1):
        if size is None or size < 0:
            self._seen.append(("read", self._path))
        return self._handle.read(size)

    def readlines(self, *args, **kwargs):
        self._seen.append(("readlines", self._path))
        return self._handle.readlines(*args, **kwargs)

    def __iter__(self):
        return iter(self._handle)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self._handle.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._handle, name)


class SearchPaginationTests(unittest.TestCase):
    """F11 — a clipped search page must not move the cursor past the clip."""

    def _needle_file(self, tmp, count=40, name="many.txt"):
        path = os.path.join(tmp, name)
        with open(path, "w", encoding="utf-8") as handle:
            for number in range(1, count + 1):
                if number % 2 == 1:
                    handle.write("needle line %03d\n" % number)
                else:
                    handle.write("filler line %03d\n" % number)
        return path

    def test_walking_pages_reproduces_every_match_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._needle_file(tmp, 40)
            expected = list(range(1, 41, 2))
            seen = []
            pages = 0
            cursor = 0
            with patch.object(code_tools, "PAGE_TEXT_BUDGET", 60):
                while True:
                    page = code_tools.search("needle", path=tmp,
                                             max_results=5, cursor=cursor)
                    self.assertTrue(page["ok"], page)
                    pages += 1
                    # the rendered page and the structured page are the same
                    # items: no match may be shown without being returned (or
                    # returned without being shown).
                    rendered = [_parse_match(line)
                                for line in _body_lines(page["content"])]
                    structured = [(os.path.relpath(m["path"], tmp), m["line"])
                                  for m in page["matches"]]
                    self.assertEqual(rendered, structured)
                    seen.extend(m["line"] for m in page["matches"])
                    if not page["has_more"]:
                        break
                    self.assertLess(pages, 100, "pagination did not terminate")
                    cursor = page["next_cursor"]
            self.assertGreater(pages, 1, "the tiny budget must paginate")
            self.assertEqual(seen, expected,
                             "pages must reproduce the matches in order")

    def test_cursor_points_at_the_first_unreturned_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._needle_file(tmp, 8)
            with patch.object(code_tools, "PAGE_TEXT_BUDGET", 60):
                page = code_tools.search("needle", path=tmp, max_results=8)
            self.assertTrue(page["ok"], page)
            self.assertTrue(page["page_clipped"])
            self.assertGreater(page["omitted_matches"], 0)
            self.assertTrue(page["has_more"])
            returned = [m["line"] for m in page["matches"]]
            # The regression: the cursor used to be placed after the last
            # SCANNED match, skipping every match the display clip dropped.
            self.assertEqual(page["next_cursor"]["after_line"], returned[-1])
            self.assertLess(len(returned), 8)
            with patch.object(code_tools, "PAGE_TEXT_BUDGET", 60):
                following = code_tools.search("needle", path=tmp,
                                              max_results=8,
                                              cursor=page["next_cursor"])
            first_next = following["matches"][0]["line"]
            self.assertEqual(first_next, returned[-1] + 2)  # needles are odd
            self.assertNotIn(first_next, returned)

    def test_pages_span_multiple_files_without_gaps_or_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = []
            for name in ("a.txt", "b.txt"):
                with open(os.path.join(tmp, name), "w",
                          encoding="utf-8") as handle:
                    for number in range(1, 6):
                        handle.write("needle %s %02d\n" % (name, number))
                        expected.append((name, number))
            seen = []
            cursor = 0
            with patch.object(code_tools, "PAGE_TEXT_BUDGET", 45):
                for _ in range(50):
                    page = code_tools.search("needle", path=tmp,
                                             max_results=2, cursor=cursor)
                    self.assertTrue(page["ok"], page)
                    for match in page["matches"]:
                        seen.append((os.path.basename(match["path"]),
                                     match["line"]))
                    if not page["has_more"]:
                        break
                    cursor = page["next_cursor"]
            self.assertEqual(seen, expected)
            self.assertEqual(len(seen), len(set(seen)))

    def test_a_structured_cursor_survives_the_dispatch_schema(self):
        """{{step0.next_cursor}} arrives as JSON text — it must still resume."""
        with tempfile.TemporaryDirectory() as tmp:
            self._needle_file(tmp, 8)
            first = code_tools.search("needle", path=tmp, max_results=2)
            self.assertTrue(first["has_more"])
            cursor_text = json.dumps(first["next_cursor"])
            resumed = code_tools.call_tool(
                "code.search",
                {"pattern": "needle", "path": tmp, "max_results": 2,
                 "cursor": cursor_text},
                grants=code_tools.agent_grants())
            self.assertTrue(resumed["ok"], resumed)
            self.assertEqual(resumed["matches"][0]["line"],
                             first["matches"][-1]["line"] + 2)

    def test_match_text_is_structured_back_to_the_caller(self):
        with tempfile.TemporaryDirectory() as tmp:
            long_line = "needle " + "x" * 1500
            _write_text(os.path.join(tmp, "long.txt"), long_line + "\n")
            result = code_tools.search("needle", path=tmp, max_results=5)
            self.assertTrue(result["ok"], result)
            self.assertEqual(len(result["matches"]), 1)
            # the structured artifact is complete; only the rendering is cut
            self.assertEqual(result["matches"][0]["text"], long_line)
            self.assertIn("chars in the structured payload",
                          result["content"])
            self.assertLess(len(result["content"]),
                            code_tools._LINE_DISPLAY_MAX + 400)


class BoundedRangeReadTests(unittest.TestCase):
    """F11 — read a window, never the whole file; page it honestly."""

    def test_range_read_never_reads_the_whole_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "big.txt")
            with open(path, "w", encoding="utf-8") as handle:
                for number in range(1, 4001):
                    handle.write("line %04d %s\n" % (number, "y" * 40))

            seen = []
            real_open = builtins.open
            target = os.path.normcase(os.path.realpath(path))

            def spy_open(file, mode="r", *args, **kwargs):
                handle = real_open(file, mode, *args, **kwargs)
                try:
                    same = (os.path.normcase(os.path.realpath(str(file)))
                            == target)
                except Exception:
                    same = False
                if same and "b" not in str(mode):
                    return _UnboundedReadSpy(handle, str(file), seen)
                return handle

            with patch("builtins.open", side_effect=spy_open):
                result = code_tools.read_range(path, start_line=2000,
                                               max_lines=5)
            self.assertTrue(result["ok"], result)
            self.assertEqual(seen, [],
                             "a bounded range must not read() the whole file")
            self.assertEqual(result["total_lines"], 4000)
            self.assertEqual([line["line"] for line in result["lines"]],
                             [2000, 2001, 2002, 2003, 2004])
            self.assertIn("line 2000", result["content"])
            self.assertEqual(result["line_count"], 5)

    def test_walking_range_pages_reproduces_every_line_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "paged.txt")
            expected = ["line %03d" % number for number in range(1, 301)]
            _write_text(path, "\n".join(expected) + "\n")
            collected = []
            start = 1
            pages = 0
            with patch.object(code_tools, "PAGE_TEXT_BUDGET", 90):
                for _ in range(1000):
                    page = code_tools.read_range(path, start_line=start,
                                                 max_lines=200)
                    self.assertTrue(page["ok"], page)
                    pages += 1
                    collected.extend(page["lines"])
                    if not page["has_more"]:
                        break
                    self.assertEqual(page["next_start"],
                                     page["lines"][-1]["line"] + 1)
                    start = page["next_start"]
            self.assertGreater(pages, 1)
            self.assertEqual([line["text"] for line in collected], expected)
            self.assertEqual([line["line"] for line in collected],
                             list(range(1, 301)))

    def test_long_line_is_complete_in_the_structured_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "huge_line.txt")
            long_line = "head " + "z" * 5000
            _write_text(path, long_line + "\ntail\n")
            page = code_tools.read_range(path, max_lines=1)
            self.assertTrue(page["ok"], page)
            self.assertEqual(page["lines"][0]["text"], long_line)
            self.assertLess(len(page["content"]), 500)
            self.assertTrue(page["has_more"])
            self.assertEqual(page["next_start"], 2)

    def test_content_hash_matches_the_whole_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "hashed.txt")
            _write_text(path, "alpha\nbeta\ngamma\n")
            page = code_tools.read_range(path, max_lines=1)
            self.assertEqual(page["content_hash"],
                             code_grants.content_hash(path))


class UnifiedPatchValidationTests(unittest.TestCase):
    """F11 — a patch either produces the intended bytes or writes nothing."""

    def _target(self, tmp, payload=b"one\ntwo\n", name="f.txt"):
        path = os.path.join(tmp, name)
        _write_bytes(path, payload)
        return path

    # ── zero-length insertions ──────────────────────────────────────────
    def test_insertion_after_line_one_lands_after_line_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._target(tmp)
            patch_text = ("--- a/f.txt\n+++ b/f.txt\n"
                          "@@ -1,0 +2,1 @@\n+inserted\n")
            result = code_tools.apply_patch(path, patch_text)
            self.assertTrue(result["ok"], result)
            self.assertTrue(result["applied"])
            self.assertEqual(_read_bytes(path), b"one\ninserted\ntwo\n")

    def test_zero_zero_insertion_lands_before_the_first_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._target(tmp)
            patch_text = ("--- a/f.txt\n+++ b/f.txt\n"
                          "@@ -0,0 +1,1 @@\n+top\n")
            result = code_tools.apply_patch(path, patch_text)
            self.assertTrue(result["ok"], result)
            self.assertEqual(_read_bytes(path), b"top\none\ntwo\n")

    def test_insertion_after_the_last_line_appends(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._target(tmp)
            patch_text = ("--- a/f.txt\n+++ b/f.txt\n"
                          "@@ -2,0 +3,1 @@\n+appended\n")
            result = code_tools.apply_patch(path, patch_text)
            self.assertTrue(result["ok"], result)
            self.assertEqual(_read_bytes(path), b"one\ntwo\nappended\n")

    # ── newline semantics ───────────────────────────────────────────────
    def test_crlf_file_stays_crlf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._target(tmp, b"alpha\r\nbeta\r\n")
            patch_text = ("--- a/f.txt\n+++ b/f.txt\n"
                          "@@ -1,2 +1,2 @@\n alpha\n-beta\n+BETA\n")
            result = code_tools.apply_patch(path, patch_text)
            self.assertTrue(result["ok"], result)
            self.assertEqual(_read_bytes(path), b"alpha\r\nBETA\r\n")

    def test_crlf_insertion_uses_crlf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._target(tmp, b"alpha\r\nbeta\r\n")
            patch_text = ("--- a/f.txt\n+++ b/f.txt\n"
                          "@@ -1,0 +2,1 @@\n+inserted\n")
            result = code_tools.apply_patch(path, patch_text)
            self.assertTrue(result["ok"], result)
            self.assertEqual(_read_bytes(path),
                             b"alpha\r\ninserted\r\nbeta\r\n")

    def test_no_final_newline_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._target(tmp, b"a\nb")
            patch_text = ("--- a/f.txt\n+++ b/f.txt\n"
                          "@@ -2 +2 @@\n-b\n+B\n")
            result = code_tools.apply_patch(path, patch_text)
            self.assertTrue(result["ok"], result)
            self.assertEqual(_read_bytes(path), b"a\nB")

    def test_no_final_newline_survives_an_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._target(tmp, b"a\nb")
            patch_text = ("--- a/f.txt\n+++ b/f.txt\n"
                          "@@ -1,2 +1,3 @@\n a\n b\n+c\n")
            result = code_tools.apply_patch(path, patch_text)
            self.assertTrue(result["ok"], result)
            self.assertEqual(_read_bytes(path), b"a\nb\nc")

    def test_lf_file_is_not_turned_into_crlf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._target(tmp, b"a\nb\n")
            patch_text = ("--- a/f.txt\n+++ b/f.txt\n"
                          "@@ -1,2 +1,3 @@\n a\n b\n+c\n")
            result = code_tools.apply_patch(path, patch_text)
            self.assertTrue(result["ok"], result)
            self.assertEqual(_read_bytes(path), b"a\nb\nc\n")

    # ── target validation ───────────────────────────────────────────────
    def test_wrong_target_patch_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._target(tmp, b"keep me\n")
            before = _read_bytes(path)
            patch_text = ("--- a/somewhere_else.txt\n+++ b/somewhere_else.txt\n"
                          "@@ -1 +1 @@\n-keep me\n+clobbered\n")
            result = code_tools.apply_patch(path, patch_text)
            self.assertFalse(result["ok"], result)
            self.assertIn("target mismatch", result["error"])
            self.assertFalse(result.get("applied"))
            self.assertEqual(_read_bytes(path), before)

    def test_patch_naming_two_files_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._target(tmp, b"keep me\n")
            before = _read_bytes(path)
            patch_text = ("--- a/f.txt\n+++ b/f.txt\n@@ -1 +1 @@\n-x\n+y\n"
                          "--- a/other.txt\n+++ b/other.txt\n"
                          "@@ -1 +1 @@\n-x\n+y\n")
            result = code_tools.apply_patch(path, patch_text)
            self.assertFalse(result["ok"], result)
            self.assertIn("more than one target", result["error"])
            self.assertEqual(_read_bytes(path), before)

    def test_deletion_patch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._target(tmp, b"keep me\n")
            before = _read_bytes(path)
            patch_text = ("--- a/f.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-keep me\n")
            result = code_tools.apply_patch(path, patch_text)
            self.assertFalse(result["ok"], result)
            self.assertEqual(_read_bytes(path), before)

    # ── malformed / ambiguous input: no write, explicit reason ──────────
    def test_missing_headers_write_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._target(tmp, b"one\ntwo\n")
            before = _read_bytes(path)
            result = code_tools.apply_patch(
                path, "@@ -1 +1 @@\n-one\n+ONE\n")
            self.assertFalse(result["ok"], result)
            self.assertIn("Malformed patch", result["error"])
            self.assertIn("missing", result["error"])
            self.assertEqual(_read_bytes(path), before)

    def test_malformed_hunk_header_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._target(tmp, b"one\ntwo\n")
            before = _read_bytes(path)
            result = code_tools.apply_patch(
                path, "--- a/f.txt\n+++ b/f.txt\n@@ -1 1 @@\n-one\n+ONE\n")
            self.assertFalse(result["ok"], result)
            self.assertIn("malformed hunk header", result["error"])
            self.assertEqual(_read_bytes(path), before)

    def test_count_mismatch_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._target(tmp, b"one\ntwo\n")
            before = _read_bytes(path)
            result = code_tools.apply_patch(
                path,
                "--- a/f.txt\n+++ b/f.txt\n@@ -1,5 +1,5 @@\n-one\n+ONE\n")
            self.assertFalse(result["ok"], result)
            self.assertIn("count mismatch", result["error"])
            self.assertEqual(_read_bytes(path), before)

    def test_non_applying_hunk_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._target(tmp, b"one\ntwo\n")
            before = _read_bytes(path)
            result = code_tools.apply_patch(
                path,
                "--- a/f.txt\n+++ b/f.txt\n@@ -1,2 +1,2 @@\n-nope\n-alsonope\n+x\n+y\n")
            self.assertFalse(result["ok"], result)
            self.assertIn("does not apply", result["error"])
            self.assertEqual(_read_bytes(path), before)

    def test_hunk_is_not_silently_relocated(self):
        """A hunk whose line number is wrong is refused, not moved."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._target(tmp, b"one\ntwo\nthree\nfour\n")
            before = _read_bytes(path)
            # the context exists at line 3, but the header says line 1
            result = code_tools.apply_patch(
                path,
                "--- a/f.txt\n+++ b/f.txt\n@@ -1,1 +1,1 @@\n-three\n+THREE\n")
            self.assertFalse(result["ok"], result)
            self.assertIn("does not apply", result["error"])
            self.assertEqual(_read_bytes(path), before)

    # ── dry run / preconditions / journal ───────────────────────────────
    def test_dry_run_never_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._target(tmp, b"one\ntwo\n")
            before = _read_bytes(path)
            journal_before = len(code_tools.change_history())
            result = code_tools.apply_patch(
                path, "--- a/f.txt\n+++ b/f.txt\n@@ -1 +1 @@\n-one\n+ONE\n",
                dry_run=True)
            self.assertTrue(result["ok"], result)
            self.assertFalse(result["applied"])
            self.assertTrue(result["dry_run"])
            self.assertIn("+ONE", result["diff"])
            self.assertEqual(_read_bytes(path), before)
            self.assertEqual(len(code_tools.change_history()), journal_before)

    def test_dry_run_validates_before_previewing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._target(tmp, b"one\ntwo\n")
            before = _read_bytes(path)
            result = code_tools.apply_patch(
                path, "--- a/f.txt\n+++ b/f.txt\n@@ -1,9 +1,9 @@\n-one\n+ONE\n",
                dry_run=True)
            self.assertFalse(result["ok"], result)
            self.assertEqual(_read_bytes(path), before)

    def test_expected_hash_precondition_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._target(tmp, b"one\ntwo\n")
            before = _read_bytes(path)
            stale = code_tools.apply_patch(
                path, "--- a/f.txt\n+++ b/f.txt\n@@ -1 +1 @@\n-one\n+ONE\n",
                expected_hash="0" * 64)
            self.assertFalse(stale["ok"], stale)
            self.assertEqual(_read_bytes(path), before)

            stale_dry = code_tools.apply_patch(
                path, "--- a/f.txt\n+++ b/f.txt\n@@ -1 +1 @@\n-one\n+ONE\n",
                dry_run=True, expected_hash="0" * 64)
            self.assertFalse(stale_dry["ok"], stale_dry)

            good = code_tools.apply_patch(
                path, "--- a/f.txt\n+++ b/f.txt\n@@ -1 +1 @@\n-one\n+ONE\n",
                expected_hash=code_grants.content_hash(path))
            self.assertTrue(good["ok"], good)
            self.assertEqual(_read_bytes(path), b"ONE\ntwo\n")

    def test_patch_records_a_restore_point_and_undo_restores_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._target(tmp, b"alpha\r\nbeta\r\n")
            result = code_tools.apply_patch(
                path,
                "--- a/f.txt\n+++ b/f.txt\n@@ -1,2 +1,2 @@\n alpha\n-beta\n+BETA\n")
            self.assertTrue(result["ok"], result)
            self.assertTrue(result.get("restore_id"))
            self.assertTrue(code_grants.last_entry(path))
            undone = code_tools.undo_last_change(path)
            self.assertTrue(undone["ok"], undone)
            self.assertEqual(_read_bytes(path), b"alpha\r\nbeta\r\n")

    def test_patch_without_a_trailing_newline_in_the_patch_body(self):
        """A hunk body may end without a final newline in the patch text."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._target(tmp, b"one\ntwo\n")
            result = code_tools.apply_patch(
                path, "--- a/f.txt\n+++ b/f.txt\n@@ -1 +1 @@\n-one\n+ONE")
            self.assertTrue(result["ok"], result)
            self.assertEqual(_read_bytes(path), b"ONE\ntwo\n")

    def test_inspect_diff_reports_the_hash_planning_needs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._target(tmp, b"one\ntwo\n")
            before_hash = code_grants.content_hash(path)
            result = code_tools.apply_patch(
                path, "--- a/f.txt\n+++ b/f.txt\n@@ -1 +1 @@\n-one\n+ONE\n")
            self.assertTrue(result["ok"], result)
            inspected = code_tools.inspect_diff(path)
            self.assertTrue(inspected["ok"], inspected)
            self.assertTrue(inspected["has_changes"])
            self.assertEqual(inspected["content_hash"],
                             code_grants.content_hash(path))
            self.assertEqual(inspected["restore_point_hash"], before_hash)


class StructuredPlanningTests(unittest.TestCase):
    """F11 — clipping is for speech; the structured payload feeds planning."""

    def _needle_file(self, tmp, count, name="paged.txt"):
        path = os.path.join(tmp, name)
        with open(path, "w", encoding="utf-8") as handle:
            for number in range(1, count + 1):
                handle.write("needle %03d %s\n" % (number, "x" * 40))
        return path

    def test_rendering_never_mutates_the_structured_payload(self):
        result = {
            "ok": True,
            "content": "y" * 5000,
            "error": "",
            "path": "C:\\proj\\app.py",
            "matches": [{"path": "C:\\proj\\app.py", "line": 3,
                         "text": "needle"}],
            "match_count": 1,
            "has_more": True,
            "next_cursor": {"file": 0, "after_line": 3},
            "content_hash": "abc",
        }
        snapshot = copy.deepcopy(result)
        rendered = agent._code_tool_text(result)
        self.assertEqual(result, snapshot,
                         "the spoken rendering must not touch the payload")
        self.assertIn("cursor", rendered)
        self.assertLess(len(rendered), 900)

    def test_search_page_and_cursor_agree_after_agent_rendering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._needle_file(tmp, 60)
            expected = list(range(1, 61))
            seen = []
            cursor = None
            for _ in range(30):
                args = {"pattern": "needle", "path": tmp, "max_results": 50}
                if cursor is not None:
                    args["cursor"] = cursor
                step = {"tool": "code.search", "args": args}
                text, structured = agent._execute_step_structured(step, {})
                self.assertTrue(structured["ok"], structured)
                rendered = [_parse_match(line) for line in _body_lines(text)]
                shown = [m["line"] for m in structured["matches"]]
                self.assertEqual([line for _where, line in rendered], shown,
                                 "the spoken page must show exactly the "
                                 "matches the cursor accounts for")
                seen.extend(shown)
                if not structured["has_more"]:
                    break
                cursor = json.dumps(structured["next_cursor"])
                self.assertEqual(structured["next_cursor"]["after_line"],
                                 shown[-1])
            self.assertEqual(seen, expected)

    def test_read_range_structured_fields_resolve_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lines.txt")
            _write_text(path, "alpha\nbeta\ngamma\n")
            step = {"tool": "code.read_range",
                    "args": {"path": path, "start_line": 1, "max_lines": 2}}
            _text, structured = agent._execute_step_structured(step, {})
            self.assertTrue(structured["ok"], structured)
            self.assertEqual(structured["content_hash"],
                             code_grants.content_hash(path))
            self.assertEqual([line["text"] for line in structured["lines"]],
                             ["alpha", "beta"])
            observations = {}
            agent._record_observation(observations, 0, step, structured,
                                      "read", "ok")
            resolved = agent._resolve_step_args(
                {"path": "{{step0.path}}",
                 "text": "{{step0.lines.1.text}}",
                 "cursor": "{{step0.next_start}}"}, observations)
            self.assertEqual(resolved["path"], structured["path"])
            self.assertEqual(resolved["text"], "beta")
            # a bounded page still had line 3 to hand out: the cursor is
            # resolvable and points at the first unreturned line
            self.assertEqual(structured["next_start"], 3)
            self.assertEqual(resolved["cursor"], "3")

            whole = code_tools.read_range(path, max_lines=10)
            self.assertFalse(whole["has_more"])
            self.assertNotIn("next_start", whole)
            observations_2 = {}
            agent._record_observation(observations_2, 0, step, whole,
                                      "read", "ok")
            # no more lines, so the placeholder stays literal (fail-honest,
            # never a guess)
            self.assertEqual(
                agent._resolve_step_args({"cursor": "{{step0.next_start}}"},
                                         observations_2)["cursor"],
                "{{step0.next_start}}")

    def test_search_structured_fields_resolve_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._needle_file(tmp, 3, name="obs.txt")
            step = {"tool": "code.search",
                    "args": {"pattern": "needle", "path": tmp,
                             "max_results": 2}}
            _text, structured = agent._execute_step_structured(step, {})
            self.assertTrue(structured["ok"], structured)
            self.assertTrue(structured["has_more"])
            observations = {}
            agent._record_observation(observations, 0, step, structured,
                                      "found", "ok")
            resolved = agent._resolve_step_args(
                {"path": "{{step0.matches.0.path}}",
                 "cursor": "{{step0.next_cursor}}"}, observations)
            self.assertEqual(resolved["path"],
                             structured["matches"][0]["path"])
            self.assertEqual(json.loads(resolved["cursor"]),
                             structured["next_cursor"])


if __name__ == "__main__":
    unittest.main()
