"""Native code-agent tools for Jarvis.

A stripped-down, built-in toolkit that gives Jarvis the *basic* file and
shell capabilities of a coding agent (opencode / codex / Claude Code)
WITHOUT shelling out to an external agent for every simple task:

    - read_file      -> read a text file
    - write_file     -> create or overwrite a text file
    - list_directory -> list a folder's contents (lightweight navigation)
    - create_folder  -> create a folder (and parents)
    - run_command    -> run a shell / cmd command and capture output
    - run_script     -> run a Python (.py) or Windows batch (.bat/.cmd) script
    - search         -> search file contents (structured matches + cursor)
    - read_range     -> read a bounded line range (cursor + content hash)
    - apply_patch    -> apply a unified diff (journalled, version-checked)
    - inspect_diff   -> diff a file against its last restore point
    - run_checks     -> run pytest / py_compile / node --check

Every tool returns a small, uniform dict::

    {"ok": bool, "content": str, "error": str, "exit_code": int, ...}

matching the connector convention used elsewhere in the task-agent brain.
Callers can render ``content`` directly as a spoken confirmation or show it
in the UI. Tools are deliberately narrow and guarded so Jarvis can safely
delegate simple work to its own hands.

Safety / limits:
    - Command execution is synchronous and time-limited (default 30s,
      override with ``JARVIS_TOOL_COMMAND_TIMEOUT``).
    - Output is truncated to ``JARVIS_TOOL_MAX_OUTPUT`` (default 8000 chars)
      so a runaway command can't flood memory.
    - File paths are resolved against the repo root by default; absolute
      paths are allowed. All file operations are plain text.

F22 (Fable-5 audit, G2): writes are confined to the granted workspace roots,
go through an atomic journalled writer (restore points under
``data/change_journal``), and scripts run as a direct argv inside a Windows
Job Object so a timeout or stop kills the whole process tree.

F11 (Fable-5 audit, G4): the inspection/edit loop — ``code.search``,
bounded ``code.read_range``, ``code.apply_patch``, ``code.inspect_diff`` and
``code.run_checks`` return structured artifacts and continuation cursors
instead of silently losing the rest of a file. The shell tool remains for
explicit shell work only; it is no longer the substitute for every missing
file operation.

F11 follow-up (pagination + validated patches): a page is bounded by the
DISPLAY budget, and the cursor points at the first item the page did not
return, so paging never skips context; ``read_range`` streams the file instead
of loading it; and ``apply_patch`` validates the target, the hunk counts and
the hunk positions, rebuilds the file's own line endings exactly, and writes
nothing at all when the input is unsupported or ambiguous.
"""

import ast
import fnmatch
import inspect
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading

from backend.services import code_grants
from backend.services import tool_policy

# CREATE_NO_WINDOW: run shell commands without flashing a console window.
_CREATE_NO_WINDOW = 0x08000000

_DEFAULT_TIMEOUT = int(os.getenv("JARVIS_TOOL_COMMAND_TIMEOUT", "30"))
_MAX_OUTPUT = int(os.getenv("JARVIS_TOOL_MAX_OUTPUT", "8000"))


def _repo_root():
    """Best-effort repo root (backend/services -> project root)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


def _resolve_path(path):
    """Resolve *path* to an absolute path, defaulting to the repo root."""
    return code_grants.resolve(path)


def _clip(text):
    """Trim output to a bounded size and strip blank edge noise."""
    if not text:
        return ""
    if len(text) > _MAX_OUTPUT:
        return text[:_MAX_OUTPUT].rstrip() + "\n...[truncated]"
    return text.strip()


def read_file(path):
    """Read a text file and return its content."""
    resolved = _resolve_path(path)
    if not resolved or not os.path.exists(resolved):
        return {"ok": False, "error": f"File not found: {path}", "content": "", "path": resolved}
    if os.path.isdir(resolved):
        return {"ok": False, "error": f"'{path}' is a directory, not a file.", "content": "", "path": resolved}
    try:
        with open(resolved, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()
        return {"ok": True, "content": _clip(content), "error": "", "path": resolved, "exit_code": 0}
    except OSError as exc:
        return {"ok": False, "error": str(exc), "content": "", "path": resolved}


def write_file(path, content="", expected_hash=None, create_only=False):
    """Create or overwrite a (text) file at *path*.

    F22: the write is scope-checked against the granted workspace roots, goes
    through an atomic temp-file + fsync + replace (so a crash can never leave
    a half-written file), refuses to clobber a file that changed since it was
    last read (``expected_hash``), and records a restore point so
    :func:`undo_last_change` can put the previous content back.

    The precondition is explicit and travels with the approved diff:
    ``create_only=True`` binds the write to a CREATE precondition (an existing
    file is refused), while ``expected_hash`` binds it to a REPLACE
    precondition (the file must exist and still hash to what was read). Both
    were optional before, so an approved "create this file" could silently
    overwrite an unrelated file that appeared in between.
    """
    resolved = _resolve_path(path)
    if not resolved:
        return {"ok": False, "error": "No file path given.", "content": "", "path": ""}
    result = code_grants.atomic_write(
        resolved, content or "", expected_hash=expected_hash,
        expect="create" if create_only else None)
    if result.get("ok") and result.get("restore_id"):
        result["content"] += " (restore id %s)" % result["restore_id"]
    return result


def undo_last_change(path=None):
    """F22: restore the most recent write (optionally for *path*)."""
    ok, message = code_grants.restore_last(path=path)
    return {"ok": ok, "content": message, "error": "" if ok else message,
            "path": path or ""}


def change_history(path=None):
    """F22: the recorded change journal (most recent last)."""
    return code_grants.entries()


def create_folder(path):
    """Create a folder (and any missing parents) at *path*."""
    resolved = _resolve_path(path)
    if not resolved:
        return {"ok": False, "error": "No folder path given.", "content": "", "path": ""}
    allowed, _resolved, reason = code_grants.grant_check(resolved)
    if not allowed:
        return {"ok": False, "error": reason, "content": "", "path": _resolved}
    try:
        os.makedirs(resolved, exist_ok=True)
        return {"ok": True, "content": f"Folder ready: {resolved}", "error": "", "path": resolved, "exit_code": 0}
    except OSError as exc:
        return {"ok": False, "error": str(exc), "content": "", "path": resolved}


def list_directory(path="."):
    """List the contents of a directory (names + kind)."""
    resolved = _resolve_path(path or ".")
    if not os.path.isdir(resolved):
        return {"ok": False, "error": f"Directory not found: {path}", "content": "", "path": resolved}
    try:
        entries = sorted(os.listdir(resolved))
        lines = []
        for name in entries[:500]:
            full = os.path.join(resolved, name)
            kind = "dir" if os.path.isdir(full) else "file"
            lines.append(f"[{kind}] {name}")
        return {"ok": True, "content": _clip("\n".join(lines)) or "(empty)", "error": "", "path": resolved, "exit_code": 0}
    except OSError as exc:
        return {"ok": False, "error": str(exc), "content": "", "path": resolved}

def run_command(command, cwd=None, timeout=None, job=None):
    """Run a shell / cmd command and capture stdout+stderr.

    Uses the system shell (cmd.exe on Windows) so pipelines, ``dir``,
    ``python -m ...`` etc. all behave like they do in a real terminal.

    F22: this is PRIVILEGED execution, so it is bound to the ``command_exec``
    grant (never to a file path) and runs as an OWNED process — a Job Object on
    Windows, its own session on POSIX — so a timeout or a cancellation kills
    the whole tree and nothing else. It used to run through a bare
    ``subprocess.run(shell=True)`` with no ownership: descendants survived the
    timeout, and no authority was consulted at all.
    """
    command = (command or "").strip()
    if not command:
        return {"ok": False, "error": "No command given.", "content": "", "exit_code": -1}
    denied = _exec_authority_error()
    if denied:
        return {"ok": False, "error": denied, "content": "", "exit_code": -1}
    timeout = timeout or _DEFAULT_TIMEOUT
    cwd = cwd or _repo_root()
    if os.name == "nt":
        argv = ["cmd", "/c", command]
    else:
        argv = ["/bin/sh", "-c", command]
    try:
        result = code_grants.run_argv(
            argv, cwd=cwd, timeout=timeout, job=job, shell=False)
    except Exception as exc:  # ownership could not be established
        return {"ok": False, "error": f"Could not run command: {exc}",
                "content": "", "exit_code": -1}
    output = _clip((result.get("content") or ""))
    if result.get("ok"):
        return {"ok": True,
                "content": output or "(no output)",
                "error": "", "exit_code": result.get("exit_code", 0)}
    return {
        "ok": False,
        "content": output or "(error, no output)",
        "error": result.get("error") or "command failed",
        "exit_code": result.get("exit_code", -1),
    }


def run_script(path=None, language=None, code=None, args=None, timeout=None,
               job=None):
    """Run a Python (.py) or Windows batch (.bat/.cmd) script.

    Either give an existing *path* to a script, or provide *code* to execute
    inline with a *language* (e.g. ``python``/``bat``). *args* is a list of
    command-line arguments passed to the script.

    F22: the script runs as a direct argv (never a re-quoted shell string)
    inside a Windows Job Object, so a timeout or a stop terminates the whole
    process tree instead of orphaning it. Inline code and a file on disk are
    BOTH privileged execution and are both gated by the ``command_exec``
    grant: a script's location never confined its effects, so checking the path
    only (which is all the old version did) said nothing about what it ran.
    """
    timeout = timeout or _DEFAULT_TIMEOUT
    import sys

    denied = _exec_authority_error()
    if denied:
        return {"ok": False, "error": denied, "content": "", "exit_code": -1}

    # Inline code mode: feed source to the interpreter as one argv element —
    # no shell string reconstruction, so no quoting can be mis-parsed.
    if code and not path:
        lang = (language or "python").lower().strip()
        if lang in ("sh", "bash", "bat", "batch", "cmd"):
            argv = ["cmd", "/c", str(code)]
        else:
            argv = [sys.executable, "-c", str(code)]
        result = code_grants.run_argv(argv, cwd=_repo_root(), timeout=timeout,
                                      job=job)
        return {**result, "content": _clip(result.get("content", "")) or
                ("(no output)" if result.get("ok") else result.get("error", ""))}

    resolved = _resolve_path(path or "")
    if not resolved or not os.path.exists(resolved):
        return {"ok": False, "error": f"Script not found: {path}", "content": "", "exit_code": -1}
    ext = os.path.splitext(resolved)[1].lower()

    if ext not in (".py", ".bat", ".cmd"):
        return {"ok": False, "error": f"Unsupported script type: {ext} (use .py, .bat or .cmd)", "content": "", "exit_code": -1}

    allowed, _resolved, reason = code_grants.grant_check(resolved)
    if not allowed:
        return {"ok": False, "error": reason, "content": "", "exit_code": -1, "path": _resolved}

    argv = code_grants.build_argv(resolved, args)
    result = code_grants.run_argv(argv, cwd=os.path.dirname(resolved) or _repo_root(),
                                  timeout=timeout, job=job)
    result["path"] = resolved
    result["content"] = _clip(result.get("content", "")) or (
        "(no output)" if result.get("ok") else result.get("error", ""))
    return result


# ── F22: privileged execution authority ───────────────────────────────────
#: thread-local, set by call_tool from the VALIDATED grants. It is deliberately
#: NOT a parameter of the tool functions: a model-supplied ``grants`` argument
#: would be privilege escalation by argument injection (the dispatch schema is
#: derived from the signature, so any parameter is model-reachable).
_EXEC_CONTEXT = threading.local()


def _exec_authority_error():
    """Return a refusal message when privileged execution is not authorized.

    Two independent gates, both fail-closed:
      * a framed dispatch must hold the ``command_exec`` grant;
      * regardless of framing, execution must not be switched off globally
        (``JARVIS_CODE_EXEC``).
    """
    flag = str(os.getenv(CODE_EXEC_ENV, "1")).strip().lower()
    if flag in ("0", "false", "no", "off"):
        return ("command execution is disabled (set %s=1 to allow it)"
                % CODE_EXEC_ENV)
    framed = getattr(_EXEC_CONTEXT, "framed", False)
    if not framed:
        return ""
    grants = getattr(_EXEC_CONTEXT, "grants", None) or set()
    needed = CODE_REQUIRED_GRANTS[tool_policy.PRIVILEGED]
    if needed not in grants:
        return ("command execution is not granted for this job (needs the %r "
                "grant)" % needed)
    return ""


def _shell_quote(arg):
    """Quote an argument for cmd.exe-style invocation."""
    arg = str(arg)
    if not arg:
        return '""'
    if any(ch in arg for ch in ' &|<>^()'):
        return f'"{arg}"'
    return arg


# ── F11 (Fable-5 audit, G4): inspection / edit loop ───────────────────────
#
# The original toolkit could only read a whole file (and clip it) or blindly
# overwrite it. The tools below give the coding loop real inspection and
# surgical edit capabilities, with structured artifacts and continuation
# cursors instead of silently losing the rest of a file.

#: Directories never searched by code.search.
_SEARCH_SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", "venv", ".venv",
    "dist", "build", "out", ".idea", ".vscode", "site-packages", ".workbuddy-ai",
}
#: Files larger than this are skipped by code.search (binary-ish / generated).
_SEARCH_MAX_FILE_BYTES = 1024 * 1024
#: Hard bound on how many files one search call will open.
_SEARCH_MAX_SCAN_FILES = 4000
#: F11: per-match stored text cap. The match's own ``text`` is the structured
#: artifact a planner can consume; beyond this the line is marked truncated
#: rather than silently cut.
_SEARCH_TEXT_MAX = 4000
#: F11: how much of one line the HUMAN rendering shows before it says so.
_LINE_DISPLAY_MAX = 200

#: F11: display budget (characters) for ONE structured page.
#:
#: This is the heart of the pagination fix: search/read_range return a page
#: whose human ``content`` fits this budget, and the continuation cursor points
#: at the first item that is NOT in that page. Previously the tools returned a
#: page sized by ``max_results``/``max_lines`` (up to thousands of characters)
#: and the task-agent clipped it for speech to 800 characters, while the cursor
#: still pointed PAST everything the clip had thrown away — so the next page
#: skipped that unseen context. The budget is deliberately below the agent's
#: ``_STEP_TEXT_LIMIT`` (800) so the clip never has to run for these tools.
#: Override with ``JARVIS_TOOL_PAGE_CHARS``.
PAGE_TEXT_BUDGET = int(os.getenv("JARVIS_TOOL_PAGE_CHARS", "700"))


def _page_budget():
    """The effective display budget for one page (never above _MAX_OUTPUT)."""
    try:
        budget = int(PAGE_TEXT_BUDGET)
    except (TypeError, ValueError):
        budget = 700
    return max(40, min(budget, max(40, _MAX_OUTPUT)))


def _page_slice(rendered_items):
    """The longest prefix of *rendered_items* that fits one display page.

    Returns ``(kept, truncated)``. The first item is always kept (even when it
    alone exceeds the budget) so a page can never be empty while items remain:
    an empty page with a cursor past it would skip content forever.
    """
    budget = _page_budget()
    kept = []
    used = 0
    for item in rendered_items:
        cost = len(item) + (1 if kept else 0)  # + newline
        if kept and used + cost > budget:
            break
        kept.append(item)
        used += cost
    return kept, len(kept) < len(rendered_items)


def _display_path(root, path):
    """``path`` relative to *root* (best effort), for human rendering."""
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return path


def _render_match(root, match):
    """One search match as ``relative/path:line: text`` (display only).

    The structured ``text`` is complete; only this rendering is capped, and it
    says so instead of silently cutting the line.
    """
    text = match["text"]
    if len(text) > _LINE_DISPLAY_MAX:
        text = "%s... [+%d chars in the structured payload]" % (
            text[:_LINE_DISPLAY_MAX], len(text) - _LINE_DISPLAY_MAX)
    return "%s:%d: %s" % (_display_path(root, match["path"]),
                          match["line"], text)


def _iter_text_files(root, glob_pattern="*"):
    """Deterministically sorted candidate files under *root*.

    Directories in _SEARCH_SKIP_DIRS are pruned; oversized files are dropped.
    Yields absolute paths.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SEARCH_SKIP_DIRS)
        for name in sorted(filenames):
            if glob_pattern and not fnmatch.fnmatch(name, glob_pattern):
                continue
            full = os.path.join(dirpath, name)
            try:
                if os.path.getsize(full) > _SEARCH_MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield full


def _parse_cursor(cursor):
    """Normalise a search cursor into ``(start_file, after_line)``.

    F11 — a cursor reaches this tool in three shapes and all three must mean
    the same thing, or "resume where the last page stopped" silently restarts
    from the beginning (duplicating matches) or skips ahead: a dict from a
    direct call, an int (file index) from the old API, or the JSON/repr string
    an observation placeholder (``{{step0.next_cursor}}``) produces.
    """
    if isinstance(cursor, str):
        text = cursor.strip()
        if text.startswith("{"):
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = loader(text)
                except (ValueError, SyntaxError):
                    continue
                if isinstance(parsed, dict):
                    cursor = parsed
                    break
    if isinstance(cursor, dict):
        try:
            return (max(0, int(cursor.get("file") or 0)),
                    max(0, int(cursor.get("after_line") or 0)))
        except (TypeError, ValueError):
            return 0, 0
    try:
        return max(0, int(cursor or 0)), 0
    except (TypeError, ValueError):
        return 0, 0


def search(pattern, path=".", glob_pattern="*", max_results=50, cursor=0,
           ignore_case=True, regex=False):
    """Search file contents for *pattern*; return structured matches.

    F11 — matches are returned as ``{path, line, text}`` artifacts and the
    page is bounded TWICE: by *max_results* and by the display budget
    (:data:`PAGE_TEXT_BUDGET`). The continuation cursor always points at the
    first match that is NOT part of the returned page — ``{"file": i,
    "after_line": n}`` for a partially scanned file — so walking the pages
    replays every match exactly once with no gap and no duplicate. Before the
    fix the cursor was derived from ``max_results`` alone, so a display clip
    that discarded matches moved the cursor past them and the tail of the page
    was skipped forever. A plain integer cursor is also accepted and means
    "start at file i".
    """
    resolved = _resolve_path(path or ".")
    if not resolved or not os.path.isdir(resolved):
        return {"ok": False, "error": f"Directory not found: {path}",
                "content": "", "path": resolved}
    if not pattern:
        return {"ok": False, "error": "No search pattern given.",
                "content": "", "path": resolved}
    max_results = max(1, min(int(max_results or 50), 500))
    start_file, after_line = _parse_cursor(cursor)

    if regex:
        try:
            flags = re.IGNORECASE if ignore_case else 0
            matcher = re.compile(pattern, flags).search
        except re.error as exc:
            return {"ok": False, "error": f"Invalid regex: {exc}",
                    "content": "", "path": resolved}
    else:
        needle = pattern.lower() if ignore_case else pattern

        def matcher(line):
            hay = line.lower() if ignore_case else line
            return needle in hay

    files = list(_iter_text_files(resolved, glob_pattern or "*"))
    total_files = len(files)
    matches = []
    positions = []  # (file index, line number) parallel to `matches`
    scanned = 0
    scan_cursor = None
    stop = False
    for index in range(start_file, total_files):
        if scanned >= _SEARCH_MAX_SCAN_FILES:
            scan_cursor = {"file": index, "after_line": 0}
            break
        full = files[index]
        scanned += 1
        resume_from = after_line if index == start_file else 0
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                for lineno, line in enumerate(fh, 1):
                    if lineno <= resume_from:
                        continue
                    if matcher(line):
                        matches.append({
                            "path": full,
                            "line": lineno,
                            "text": line.rstrip()[: _SEARCH_TEXT_MAX],
                        })
                        positions.append((index, lineno))
                        if len(matches) >= max_results:
                            # Stopped mid-file: resume after this exact line
                            # so nothing is repeated or lost.
                            scan_cursor = {"file": index,
                                           "after_line": lineno}
                            stop = True
                            break
        except (OSError, UnicodeError):
            continue
        if stop:
            break

    # ── F11 pagination: the page IS the display slice ────────────────────
    # `content` is the spoken/human rendering, so the page must be exactly
    # what fits it; `matches`/`positions` are trimmed to the same items (the
    # structured payload for those items stays intact) and the cursor is
    # placed after the LAST RETURNED match, never after the last scanned one.
    rendered = [_render_match(resolved, match) for match in matches]
    kept_rendered, truncated = _page_slice(rendered)
    page_matches = matches[:len(kept_rendered)]
    page_positions = positions[:len(kept_rendered)]
    if truncated and page_positions:
        last_file, last_line = page_positions[-1]
        next_cursor = {"file": last_file, "after_line": last_line}
    else:
        next_cursor = scan_cursor

    has_more = next_cursor is not None and next_cursor["file"] < total_files
    content = "\n".join(kept_rendered) or "(no matches)"
    result = {
        "ok": True,
        "content": content,
        "error": "",
        "path": resolved,
        "exit_code": 0,
        "matches": page_matches,
        "match_count": len(page_matches),
        "scanned_files": scanned,
        "total_files": total_files,
        "has_more": bool(has_more),
        # Explicit, never silent: how many matches this page found but did
        # not return, and whether the display budget (not max_results) was
        # what ended the page.
        "omitted_matches": len(matches) - len(page_matches),
        "page_clipped": bool(truncated),
    }
    if has_more:
        result["next_cursor"] = next_cursor
        result["content"] += (
            "\n...[more — resume with cursor=%s]" % next_cursor)
    return result


def _render_numbered(lineno, text):
    """One line as ``   N | text`` for the human rendering."""
    shown = text
    if len(shown) > _LINE_DISPLAY_MAX:
        shown = "%s... [+%d chars in the structured payload]" % (
            shown[:_LINE_DISPLAY_MAX], len(text) - _LINE_DISPLAY_MAX)
    return "%6d | %s" % (lineno, shown)


def read_range(path, start_line=1, end_line=None, max_lines=200):
    """Read a bounded line range of a file (1-based, inclusive).

    F11 — three properties the audit asked for:

      * **Bounded memory**: the file is streamed line by line (``for line in
        fh``); it is never ``read()`` into memory just to return a slice. Only
        the requested window is kept, plus a counter for ``total_lines``.
      * **An honest cursor**: the returned page is the longest prefix of the
        window whose numbering fits :data:`PAGE_TEXT_BUDGET`; ``next_start``
        is the first line NOT returned, so a display clip can never leave a
        gap between what was shown and where the next read resumes.
      * **Structured, unclipped lines**: ``lines`` carries the full text of
        every returned line (``{line, text}``) and ``content_hash`` the hash
        of the whole file, so planning gets artifacts rather than a
        speech-friendly string. Only ``content`` (the human rendering) is
        clipped per line.
    """
    resolved = _resolve_path(path)
    if not resolved or not os.path.isfile(resolved):
        return {"ok": False, "error": f"File not found: {path}",
                "content": "", "path": resolved}
    max_lines = max(1, min(int(max_lines or 200), 2000))
    try:
        start_line = max(1, int(start_line or 1))
    except (TypeError, ValueError):
        start_line = 1
    if end_line is not None:
        try:
            end_line = int(end_line)
        except (TypeError, ValueError):
            end_line = None
    if end_line is not None and end_line < start_line:
        end_line = start_line - 1  # empty range (start past EOF)

    window = []
    numbered = []
    used = 0
    budget = _page_budget()
    total = 0
    try:
        with open(resolved, "r", encoding="utf-8", errors="replace") as fh:
            for lineno, raw in enumerate(fh, 1):
                total = lineno
                if lineno < start_line:
                    continue
                if end_line is not None and lineno > end_line:
                    continue
                if len(window) >= max_lines:
                    continue
                text = raw.rstrip("\n")
                line_render = _render_numbered(lineno, text)
                cost = len(line_render) + (1 if numbered else 0)
                if numbered and used + cost > budget:
                    # The page is full: keep counting (cheap, one line at a
                    # time) but never return this line, so next_start points
                    # exactly at it.
                    continue
                window.append({"line": lineno, "text": text})
                numbered.append(line_render)
                used += cost
    except OSError as exc:
        return {"ok": False, "error": str(exc), "content": "", "path": resolved}

    if end_line is None:
        end_line = start_line + max_lines - 1
    end_line = min(end_line, total)
    if end_line < start_line:
        end_line = start_line - 1  # empty range (start past EOF)

    returned_end = window[-1]["line"] if window else start_line - 1
    has_more = returned_end < total
    result = {
        "ok": True,
        "content": "\n".join(numbered) or "(empty range)",
        "error": "",
        "path": resolved,
        "exit_code": 0,
        "start_line": start_line,
        # end_line is the last line actually RETURNED (never one that was
        # dropped for display), so end_line/has_more/next_start always agree.
        "end_line": returned_end,
        "total_lines": total,
        "has_more": bool(has_more),
        "content_hash": code_grants.content_hash(resolved),
        "lines": window,
        "line_count": len(window),
    }
    if has_more:
        result["next_start"] = returned_end + 1
        result["content"] += (
            "\n...[%d more lines — continue with start_line=%d]"
            % (total - returned_end, returned_end + 1))
    return result


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

#: Patch preamble lines that carry no hunk body.
_DIFF_META_PREFIXES = (
    "diff ", "index ", "new file", "deleted file", "old mode", "new mode",
    "similarity index", "dissimilarity index", "rename ", "copy ", "Binary ",
)


def _parse_unified_diff(patch):
    """Parse a unified diff into ``(target_path_hint, hunks)``.

    hunks: ``[{"old_start", "old_count", "new_start", "new_count", "ops"}]``
    where ``ops`` is ``[(kind, text)]`` with kind in ``" +-"`` and the text
    without its prefix.

    F11 — the parse is VALIDATED, not hopeful. It raises :class:`ValueError`
    with an explicit reason for anything it cannot apply byte-exactly:

      * missing ``---``/``+++`` file headers (the target would be a guess);
      * a malformed hunk header (a line starting with ``@@`` that is not one);
      * a hunk body whose context/removed and context/added line counts do not
        match the numbers in its ``@@`` header (never trusted silently);
      * a patch that names more than one target file;
      * a deletion (``+++ /dev/null``), which ``apply_patch`` cannot express.

    The counts are also what ends a hunk, so an added line that happens to look
    like a file header (``+++ x``) is kept as content while the hunk is still
    incomplete instead of being mistaken for a new section.
    """
    target = None
    saw_old_header = False
    hunks = []
    current = None
    for raw in (patch or "").splitlines():
        if current is not None and _hunk_complete(current):
            _check_hunk_counts(current)
            current = None

        if current is None and raw.startswith("--- "):
            saw_old_header = True
            continue
        if current is None and raw.startswith("+++ "):
            name = raw[4:].strip().split("\t")[0].strip().strip('"')
            if name == "/dev/null":
                raise ValueError(
                    "patch deletes the file (+++ /dev/null); apply_patch "
                    "cannot express a deletion")
            name = re.sub(r"^[ab]/", "", name)
            if target is not None and name != target:
                raise ValueError(
                    "patch names more than one target file (%r and %r)"
                    % (target, name))
            target = name
            continue
        if current is None and raw.startswith(_DIFF_META_PREFIXES):
            continue

        match = _HUNK_RE.match(raw)
        if match:
            current = {
                "old_start": int(match.group(1)),
                "old_count": int(match.group(2)) if match.group(2) is not None
                else 1,
                "new_start": int(match.group(3)),
                "new_count": int(match.group(4)) if match.group(4) is not None
                else 1,
                "ops": [],
            }
            hunks.append(current)
            continue
        if raw.startswith("@@"):
            raise ValueError("malformed hunk header: %r" % raw[:80])
        if current is None:
            continue  # preamble prose outside any hunk — tolerated
        if raw.startswith("\\"):
            continue  # "\ No newline at end of file"
        if raw == "":
            # A bare empty line is a context line with empty text.
            current["ops"].append((" ", ""))
            continue
        if raw[0] in " +-":
            current["ops"].append((raw[0], raw[1:]))
            continue
        raise ValueError(
            "hunk @@ -%d,%d @@ ends before it contains its stated line counts "
            "(unexpected line %r)" % (current["old_start"],
                                      current["old_count"], raw[:60]))

    if current is not None:
        _check_hunk_counts(current)
    if not hunks:
        raise ValueError("no hunks found in patch")
    if target is None:
        raise ValueError(
            "missing file headers: a validated patch needs '--- <old>' and "
            "'+++ <new>' lines naming the target")
    if not saw_old_header:
        raise ValueError("missing '---' header (only a '+++ <new>' line was "
                         "found, so the patch target is ambiguous)")
    return target, hunks


def _hunk_complete(hunk):
    """True once a hunk's body holds exactly the line counts it declared."""
    need = sum(1 for kind, _ in hunk["ops"] if kind in (" ", "-"))
    added = sum(1 for kind, _ in hunk["ops"] if kind in (" ", "+"))
    return need >= hunk["old_count"] and added >= hunk["new_count"]


def _check_hunk_counts(hunk):
    """Refuse a hunk whose body does not match its declared counts."""
    need = sum(1 for kind, _ in hunk["ops"] if kind in (" ", "-"))
    added = sum(1 for kind, _ in hunk["ops"] if kind in (" ", "+"))
    if need != hunk["old_count"] or added != hunk["new_count"]:
        raise ValueError(
            "hunk @@ -%d,%d +%d,%d @@ declares %d old/%d new line(s) but its "
            "body has %d old/%d new — refusing to apply a count mismatch"
            % (hunk["old_start"], hunk["old_count"], hunk["new_start"],
               hunk["new_count"], hunk["old_count"], hunk["new_count"],
               need, added))


def _split_lines_with_ends(text):
    """Split *text* into ``(line_contents, line_terminators)``.

    Unlike ``str.splitlines`` this only breaks on ``\\n``/``\\r\\n``, so a
    diff-vs-file comparison never invents a line break at a Unicode boundary.
    The terminator of the last line is ``""`` when the file has no final
    newline, which is exactly what preserves that property on write.
    """
    contents = []
    ends = []
    start = 0
    while True:
        index = text.find("\n", start)
        if index == -1:
            tail = text[start:]
            if tail:
                contents.append(tail)
                ends.append("")
            elif not contents:
                pass  # empty file
            break
        chunk = text[start:index]
        if chunk.endswith("\r"):
            contents.append(chunk[:-1])
            ends.append("\r\n")
        else:
            contents.append(chunk)
            ends.append("\n")
        start = index + 1
    return contents, ends


def _dominant_end(ends):
    """The file's prevailing line terminator (``\\r\\n`` wins ties)."""
    crlf = sum(1 for end in ends if end == "\r\n")
    lf = sum(1 for end in ends if end == "\n")
    if crlf and crlf >= lf:
        return "\r\n"
    return "\n"


def _apply_hunks(lines, hunks, ends=None):
    """Apply parsed hunks to *lines* (list of str).

    F11 — exact application only. Every hunk must match at its stated
    position; when it does not, the whole patch fails honestly (the old
    ±20-line "relocation" search could silently write a change somewhere the
    patch never named — a wrong write is worse than no write).

    Line-count semantics: a zero-length old range (``-N,0``) is an INSERTION
    AFTER line *N* (``-0,0`` inserts before line 1), not before line N —
    the classic off-by-one that put an inserted line above its anchor.

    When *ends* (the per-line terminators from :func:`_split_lines_with_ends`)
    is given, returns ``(new_lines, new_ends)`` so the caller can rebuild the
    file byte-for-byte; otherwise just the new list of lines.
    """
    out = list(lines)
    out_ends = list(ends) if ends is not None else None
    fallback_end = _dominant_end(ends) if ends else "\n"
    offset = 0
    for hunk in hunks:
        old_start = hunk["old_start"]
        old_count = hunk["old_count"]
        ops = hunk["ops"]
        need = [text for kind, text in ops if kind in (" ", "-")]
        pos = (old_start if old_count == 0 else old_start - 1) + offset

        if pos < 0 or pos + len(need) > len(out):
            raise ValueError(
                "hunk @@ -%d,%d @@ does not apply: it falls outside the file"
                % (old_start, old_count))
        if out[pos:pos + len(need)] != need:
            raise ValueError(
                "hunk @@ -%d,%d @@ does not apply: context mismatch"
                % (old_start, old_count))

        new_block = [text for kind, text in ops if kind != "-"]
        if out_ends is None:
            out[pos:pos + len(need)] = new_block
        else:
            block_ends = []
            context_index = 0
            for kind, _text in ops:
                if kind == "-":
                    continue
                if kind == " ":
                    end = out_ends[pos + context_index] or fallback_end
                    context_index += 1
                    block_ends.append(end)
                else:
                    block_ends.append(fallback_end)
            out[pos:pos + len(need)] = new_block
            out_ends[pos:pos + len(need)] = block_ends
        offset += len(new_block) - len(need)
    if out_ends is None:
        return out
    return out, out_ends


def _target_matches(header_target, resolved):
    """True when a patch's ``+++`` name can only mean *resolved*.

    Compared on the basename (case-insensitively on Windows) so a patch may
    name ``a/src/mod.py`` for ``C:\\proj\\src\\mod.py``; a patch that names a
    different file is refused outright rather than applied to the wrong file.
    """
    header = str(header_target or "").strip().strip('"')
    if not header:
        return False
    name = os.path.basename(header.replace("\\", "/"))
    actual = os.path.basename(resolved)
    if os.name == "nt":
        return name.lower() == actual.lower()
    return name == actual


def apply_patch(path, patch, dry_run=False, expected_hash=None):
    """Apply a unified-diff patch to *path*.

    F11 — surgical edits instead of blind overwrites, and a VALIDATED edit:
    the patch must name *path* (a patch for another file is refused), its hunk
    headers must be well formed with bodies that match their declared line
    counts, and every hunk must apply at its stated position. Any unsupported
    or ambiguous input returns ``ok=False`` with an explicit reason and writes
    NOTHING. Line endings and the final-newline state are rebuilt byte-exactly
    from the file's own terminators, so a CRLF file stays CRLF.

    The write itself goes through the F22 atomic journalled writer
    (scope-checked, restore point recorded, optional ``expected_hash`` version
    check). ``dry_run=True`` never writes; it validates and returns the diff.
    """
    resolved = _resolve_path(path)
    if not resolved or not os.path.isfile(resolved):
        return {"ok": False, "error": f"File not found: {path}",
                "content": "", "path": resolved, "applied": False}
    if not (patch or "").strip():
        return {"ok": False, "error": "Empty patch.", "content": "",
                "path": resolved, "applied": False}
    try:
        target, hunks = _parse_unified_diff(patch)
    except ValueError as exc:
        return {"ok": False, "error": f"Malformed patch: {exc}",
                "content": "", "path": resolved, "applied": False}
    if not _target_matches(target, resolved):
        return {"ok": False,
                "error": ("patch target mismatch: the patch names %r but this "
                          "call targets %r — refusing to write"
                          % (target, os.path.basename(resolved))),
                "content": "", "path": resolved, "applied": False,
                "target": target}

    try:
        with open(resolved, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        return {"ok": False, "error": str(exc), "content": "",
                "path": resolved, "applied": False}
    try:
        original = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return {"ok": False,
                "error": ("%s is not valid UTF-8 (%s); refusing to patch — a "
                          "byte-exact result cannot be guaranteed" % (resolved, exc)),
                "content": "", "path": resolved, "applied": False}

    lines, ends = _split_lines_with_ends(original)
    try:
        new_lines, new_ends = _apply_hunks(lines, hunks, ends)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "content": "",
                "path": resolved, "applied": False}
    if new_ends:
        # Preserve the file's own final-newline state exactly: a file that did
        # not end with a newline still does not, whatever the patch appended.
        new_ends[-1] = ends[-1] if ends else "\n"
    new_content = "".join(text + end
                          for text, end in zip(new_lines, new_ends))

    if new_content.encode("utf-8") == raw:
        return {"ok": True, "content": "Patch is a no-op; file unchanged.",
                "error": "", "path": resolved, "exit_code": 0,
                "applied": False, "diff": "", "hunks": len(hunks)}

    if expected_hash and code_grants.content_hash(resolved) != expected_hash:
        return {"ok": False,
                "error": ("file changed since it was last read (expected %s) — "
                          "re-read and retry" % str(expected_hash)[:12]),
                "content": "", "path": resolved, "applied": False}

    diff = code_grants.preview_diff(resolved, new_content)
    if dry_run:
        return {"ok": True, "content": _clip(diff) or "(no diff)",
                "error": "", "path": resolved, "exit_code": 0,
                "applied": False, "diff": diff, "hunks": len(hunks),
                "dry_run": True}

    result = code_grants.atomic_write(resolved, new_content,
                                      expected_hash=expected_hash,
                                      expect="replace")
    result["applied"] = bool(result.get("ok"))
    result["hunks"] = len(hunks)
    if result.get("ok"):
        result["diff"] = diff
        result["content"] = ("Patched %s (%d hunk%s)."
                             % (resolved, len(hunks),
                                "" if len(hunks) == 1 else "s"))
        if result.get("restore_id"):
            result["content"] += " (restore id %s)" % result["restore_id"]
    return result


def inspect_diff(path=None):
    """Inspect pending changes: current file vs. its last restore point.

    F11 — the journal from F22 already records every write Jarvis makes;
    this turns it into a reviewable artifact. With *path*, returns the
    unified diff between the most recent restore point and the current
    content. Without *path*, returns the recent change journal.
    """
    import difflib
    if not path:
        entries = code_grants.entries()[-10:]
        if not entries:
            return {"ok": True, "content": "(no recorded changes)",
                    "error": "", "entries": [], "exit_code": 0}
        lines = [
            "%s  %s  %s" % (entry["id"], entry.get("kind") or "write",
                            entry.get("path") or "?")
            for entry in entries
        ]
        return {"ok": True, "content": "\n".join(lines), "error": "",
                "entries": entries, "exit_code": 0}

    resolved = _resolve_path(path)
    entry = code_grants.last_entry(resolved)
    if entry is None:
        return {"ok": True,
                "content": "No recorded change for %s." % resolved,
                "error": "", "path": resolved, "exit_code": 0,
                "diff": ""}
    before = ""
    backup = entry.get("backup")
    if entry.get("existed") and backup and os.path.exists(backup):
        try:
            with open(backup, "r", encoding="utf-8", errors="replace") as fh:
                before = fh.read()
        except OSError:
            before = ""
    after = ""
    if os.path.exists(resolved):
        try:
            with open(resolved, "r", encoding="utf-8", errors="replace") as fh:
                after = fh.read()
        except OSError:
            after = ""
    diff = "\n".join(difflib.unified_diff(
        before.splitlines(), after.splitlines(),
        fromfile="restore-point/%s" % os.path.basename(resolved),
        tofile="current/%s" % os.path.basename(resolved), lineterm=""))
    return {
        "ok": True,
        "content": _clip(diff) or "(no difference from restore point)",
        "error": "",
        "path": resolved,
        "exit_code": 0,
        "diff": diff,
        "restore_id": entry["id"],
        "has_changes": bool(diff),
        # F11: the hash the planning loop needs to version-check a follow-up
        # patch is the hash of what is on disk RIGHT NOW (same function
        # read_range reports), so a stale-hash write is refused rather than
        # silently overwriting an edit made after the inspection.
        "content_hash": code_grants.content_hash(resolved),
        "restore_point_hash": entry.get("before_hash"),
    }


def run_checks(kind="auto", path=None, timeout=None, job=None):
    """Run the project's checks and return structured verdicts.

    F11 — the coding loop's feedback signal: syntax/compile checks and test
    suites as a first-class tool instead of improvising shell calls.
    ``kind`` is ``auto`` (infer from *path*: .py → py_compile, .js →
    node --check, otherwise the pytest suite), ``pytest``, ``py_compile`` or
    ``node_check``. Everything runs through the F22 Job-Object runner, so a
    hung suite is killed with its whole process tree.

    F22: checks are PRIVILEGED execution (they run arbitrary project code), so
    they are gated by the same ``command_exec`` grant as ``run_command`` — an
    inconsistency here was one of the audit's findings.
    """
    kind = (kind or "auto").strip().lower()
    timeout = int(timeout or (180 if kind in ("auto", "pytest") else 60))
    denied = _exec_authority_error()
    if denied:
        return {"ok": False, "error": denied, "content": "", "exit_code": -1}
    resolved = _resolve_path(path) if path else None

    if kind == "auto":
        if resolved and os.path.isfile(resolved):
            ext = os.path.splitext(resolved)[1].lower()
            kind = "py_compile" if ext == ".py" else (
                "node_check" if ext in (".js", ".mjs", ".cjs") else "pytest")
        else:
            kind = "pytest"

    checks = []
    if kind == "py_compile":
        targets = []
        if resolved and os.path.isfile(resolved):
            targets = [resolved]
        elif resolved and os.path.isdir(resolved):
            for dirpath, _dirnames, filenames in os.walk(resolved):
                for name in sorted(filenames):
                    if name.endswith(".py"):
                        targets.append(os.path.join(dirpath, name))
                        if len(targets) >= 200:
                            break
        if not targets:
            return {"ok": False, "error": "No Python files to compile.",
                    "content": "", "exit_code": -1}
        argv = [sys.executable, "-m", "py_compile"] + targets
        label = "py_compile %d file(s)" % len(targets)
    elif kind == "node_check":
        node = shutil.which("node")
        if not node:
            return {"ok": False, "error": "node is not on PATH.",
                    "content": "", "exit_code": -1}
        if not resolved or not os.path.isfile(resolved):
            return {"ok": False, "error": "node_check needs a file path.",
                    "content": "", "exit_code": -1}
        argv = [node, "--check", resolved]
        label = "node --check %s" % os.path.basename(resolved)
    elif kind == "pytest":
        target = resolved or os.path.join(_repo_root(), "backend", "tests")
        argv = [sys.executable, "-m", "pytest", target, "-q"]
        label = "pytest %s" % os.path.basename(str(target).rstrip(os.sep))
    else:
        return {"ok": False,
                "error": "Unknown check kind: %s "
                         "(use auto, pytest, py_compile, node_check)" % kind,
                "content": "", "exit_code": -1}

    result = code_grants.run_argv(argv, cwd=_repo_root(), timeout=timeout,
                                  job=job)
    ok = bool(result.get("ok"))
    checks.append({"name": label, "ok": ok,
                   "exit_code": result.get("exit_code")})
    content = _clip(result.get("content", "")) or (
        "checks passed" if ok else result.get("error", "checks failed"))
    return {
        "ok": ok,
        "content": ("%s: %s\n%s" % ("PASS" if ok else "FAIL", label, content)),
        "error": "" if ok else result.get("error", "checks failed"),
        "exit_code": result.get("exit_code", -1),
        "checks": checks,
        "command": argv,
    }


# --- Tool registry (name -> (callable, short description)) -----------------
TOOL_REGISTRY = {
    "code.read_file": (read_file, "Read a text file; returns its contents."),
    "code.write_file": (write_file, "Create or overwrite a text file."),
    "code.list_directory": (list_directory, "List a directory's contents."),
    "code.create_folder": (create_folder, "Create a folder (and parents)."),
    "code.run_command": (run_command, "Run a shell/cmd command and capture its output."),
    "code.run_script": (run_script, "Run a Python (.py) or batch (.bat/.cmd) script."),
    # F11 (G4): the inspection/edit loop. Every page is returned with the
    # structured artifact (path/hash/lines/matches/cursor) and a continuation
    # cursor that points at the first item the page did NOT return.
    "code.search": (search, "Search file contents; structured matches with a continuation cursor (args.pattern, args.path, args.glob_pattern, args.max_results, args.cursor, args.regex). The page may be display-bounded; next_cursor always resumes at the first unreturned match."),
    "code.read_range": (read_range, "Read a bounded line range of a file with a continuation cursor, structured lines and content hash (args.path, args.start_line, args.end_line, args.max_lines). Streams the file; next_start is the first unreturned line."),
    "code.apply_patch": (apply_patch, "Apply a validated unified-diff patch to a file, journalled and version-checked (args.path, args.patch, args.dry_run, args.expected_hash). The patch must name args.path, its hunks must match their declared counts and apply exactly, and line endings/final-newline state are preserved; anything unsupported writes nothing."),
    "code.inspect_diff": (inspect_diff, "Show the diff between a file and its last restore point, or the recent change journal (optional args.path)."),
    "code.run_checks": (run_checks, "Run project checks: pytest, py_compile or node --check (args.kind, args.path, args.timeout)."),
}


# ── F17: one typed effect boundary for the native code tools ───────────────
#: The authority vocabulary these tools use. Reading is free; writing the
#: workspace needs ``workspace_write`` (the write itself is confined to the
#: granted roots by code_grants, F22); RUNNING a program needs
#: ``command_exec`` because the model does not have to have written it.
CODE_REQUIRED_GRANTS = {
    tool_policy.MUTATE: "workspace_write",
    tool_policy.PRIVILEGED: "command_exec",
}

#: Environment switch for command execution (default on, as before F17).
CODE_EXEC_ENV = "JARVIS_CODE_EXEC"

#: Path types, so a schema can be derived from each tool's signature.
_SCHEMA_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

#: Declared types for the arguments that carry an effect. The tools are
#: unannotated, so the contract is stated once here instead of being inferred
#: (or, worse, not checked at all).
_PARAM_TYPES = {
    "path": "string",
    "content": "string",
    "command": "string",
    "cwd": "string",
    "timeout": "integer",
    "pattern": "string",
    "glob_pattern": "string",
    "max_results": "integer",
    # F11: a SEARCH cursor is a dict {"file", "after_line"} when it comes back
    # from a tool call, but an observation placeholder
    # ({{step0.next_cursor}}) hands it over as a JSON string — both are the
    # same cursor and both must pass the typed boundary, or the resumed step
    # is refused and the unreturned matches are never read.
    "cursor": ["integer", "object", "string"],
    "start_line": "integer",
    "end_line": "integer",
    "max_lines": "integer",
    "patch": "string",
    "dry_run": "boolean",
    "expected_hash": "string",
    "kind": "string",
    "code": "string",
    "language": "string",
    "script_path": "string",
    "args": "array",
    "query": "string",
    "limit": "integer",
}


def _schema_for(func):
    """A typed JSON schema derived from *func*'s signature.

    Deriving it (rather than hand-writing three copies) keeps the declared
    contract and the real one from drifting apart: a parameter with no default
    is REQUIRED, and a wrongly-typed argument is refused before the effect.
    """
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return {"type": "object"}
    properties = {}
    required = []
    for pname, param in parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        kind = _SCHEMA_TYPES.get(param.annotation) or _PARAM_TYPES.get(pname)
        properties[pname] = {"type": kind} if kind else {}
        if param.default is param.empty:
            required.append(pname)
    return {"type": "object", "properties": properties, "required": required}


def agent_grants():
    """The authority our own orchestrated task paths hold (F17).

    This is deliberately explicit rather than implicit: a bare
    ``code_tools.call_tool(...)`` without grants is refused, so every caller
    that may change the workspace or run a program has to say so.
    """
    grants = {"workspace_write"}
    flag = str(os.getenv(CODE_EXEC_ENV, "1")).strip().lower()
    if flag not in ("0", "false", "no", "off"):
        grants.add("command_exec")
    return grants


def call_tool(tool, args=None, grants=None, origin="model"):
    """Dispatch a tool name (see TOOL_REGISTRY) with an args dict.

    Returns the uniform result dict, or an ``ok=False`` dict for unknown tools
    so callers never have to handle a raw KeyError.

    F17: the dispatch is validated against the same typed policy boundary as
    the browser tools — name, argument schema and the authority the operation
    needs — BEFORE the tool runs. ``grants`` empty (the default) means no
    authority: a direct invocation that never declared what it may do is
    refused instead of quietly writing files or spawning processes. Our own
    task paths pass :func:`agent_grants`.
    """
    args = args if isinstance(args, dict) else {}
    if tool not in TOOL_REGISTRY:
        return {"ok": False, "error": f"Unknown tool: {tool}", "content": "", "exit_code": -1}
    decision = tool_policy.validate_dispatch(
        tool, args, frozenset(TOOL_REGISTRY), schema=TOOL_SCHEMAS.get(tool),
        grants=grants, origin=origin,
        required_grants=CODE_REQUIRED_GRANTS)
    if not decision.allowed:
        logging.warning("[CODE_TOOLS] %s blocked by policy: %s",
                        tool, decision.reason)
        return {"ok": False, "error": "tool blocked by policy: %s"
                % decision.reason, "content": "", "exit_code": -1}
    entry = TOOL_REGISTRY.get(tool)
    try:
        fn, _ = entry
        signature = inspect.signature(fn)
        kwargs = {k: v for k, v in args.items() if k in signature.parameters}
        # F22: frame the VALIDATED authority for the duration of the call, so a
        # privileged tool consults the grants the policy actually granted — not
        # anything the model could pass as an argument.
        previous_framed = getattr(_EXEC_CONTEXT, "framed", False)
        previous_grants = getattr(_EXEC_CONTEXT, "grants", None)
        _EXEC_CONTEXT.framed = True
        _EXEC_CONTEXT.grants = {
            str(g) for g in (getattr(decision, "effective_grants", None)
                             or grants or ())
        }
        try:
            return fn(**kwargs)
        finally:
            _EXEC_CONTEXT.framed = previous_framed
            _EXEC_CONTEXT.grants = previous_grants
    except Exception as exc:  # noqa: BLE001 - report any tool failure uniformly
        logging.warning("[CODE_TOOLS] %s failed: %s", tool, exc)
        return {"ok": False, "error": str(exc), "content": "", "exit_code": -1}


#: Typed argument schemas, derived from each tool's own signature (F17).
TOOL_SCHEMAS = {
    name: _schema_for(entry[0]) for name, entry in TOOL_REGISTRY.items()
}

