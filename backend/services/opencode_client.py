"""opencode CLI integration for Jarvis.

When Jarvis can't complete a task with its own local actions, it can hand
the request off to the opencode CLI (headless, non-interactive). opencode
runs with the user's already-configured model and tools, so "no matter
what it is" the task gets a real agentic attempt.

A persistent `opencode serve` daemon (started by the watcher at boot) keeps
the model/agent hot in memory. `run_opencode_task` detects the live server
and attaches to it (`opencode run --attach <url>`), skipping the cold-start
of spawning a fresh serve instance per task; it falls back to a one-shot
`opencode run` only when no server is reachable.
"""

import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from urllib.request import Request, urlopen

from backend.services import tool_policy
from backend.services.voice import speak
from backend.services.task_result import TaskResult, result_from_reported_text

OPENCODE_CMD = os.getenv("JARVIS_OPENCODE_CMD", "opencode")
#: F16 — the executor name a capability contract must name to authorize a
#: spawn of this CLI (mirrors capability_resolver.OPENCODE without importing it).
OPENCODE_ENGINE = "opencode"
_TIMEOUT = int(os.getenv("JARVIS_OPENCODE_TIMEOUT", "180"))
# Persistent opencode serve daemon (started by the watcher).
OPENCODE_SERVE_PORT = int(os.getenv("JARVIS_OPENCODE_PORT", "9560"))
OPENCODE_SERVE_URL = f"http://127.0.0.1:{OPENCODE_SERVE_PORT}"
# Disable attaching to the persistent server with JARVIS_OPENCODE_SERVE=0.
_OPENCODE_SERVE = os.getenv("JARVIS_OPENCODE_SERVE", "1") != "0"
# opencode's headless `run` mode needs --auto: a permission prompt cannot be
# answered by anyone when spawned from Jarvis, and without it opencode
# auto-rejects tool calls (e.g. creating a folder on the Desktop) and fails.
# Disable by setting JARVIS_OPENCODE_AUTO=0.
_OPENCODE_AUTO = os.getenv("JARVIS_OPENCODE_AUTO", "1") != "0"
# Pin the task-run model (verified id) so every hand-off uses the same
# fast router regardless of the config default.
_OPENCODE_MODEL = "fireworks-ai/accounts/fireworks/routers/kimi-k2p6-fast"

# CREATE_NO_WINDOW (not DETACHED_PROCESS = 0x00000008) for a hidden child:
# the serve daemon is a quiet HTTP daemon, so it stays headless; the per-task
# `opencode run` is hidden too (its output lands in the activity log). The
# activity tail window uses CREATE_NEW_CONSOLE: one visible terminal that
# live-streams what opencode is doing.
_CREATE_NO_WINDOW = 0x08000000
_CREATE_NEW_CONSOLE = 0x00000010

# Pid of the persistent serve daemon spawned by this process, remembered for
# a clean shutdown. The port-based kill remains the fallback when the pid is
# unknown (e.g. the daemon predates this watcher run).
_SERVER_PID = None

# Pid of the visible activity-tail console window (PowerShell Get-Content
# -Wait against the activity log), killed on shutdown alongside the server.
_TAIL_PID = None

# Persistent brave-control MCP daemon (http mode): the browser is launched
# once per jarvis boot and shared by every opencode session, instead of each
# session spawning its own stdio MCP child. The token comes from the env
# (BRAVE_MCP_TOKEN in .env) and MUST match the Authorization header in
# ~/.config/opencode/opencode.jsonc — the user syncs that file during
# rotation.
BRAVE_MCP_PORT = int(os.getenv("BRAVE_MCP_PORT", "9570"))
BRAVE_MCP_TOKEN = os.getenv("BRAVE_MCP_TOKEN")

# G6 (audit F13/F40): jarvis owns the daemon source of truth in the repo at
# integrations/brave-control/server.mjs (vendored from brave-control, which
# the audit could only inspect on disk). The default points at the vendored
# copy; an explicit BRAVE_MCP_SERVER_DIR env still wins when a debugging copy
# of the daemon is needed.
def _default_brave_mcp_server_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))
    vendored = os.path.join(repo_root, "integrations", "brave-control")
    if os.path.isdir(vendored):
        return vendored
    return r"C:\Users\mayan\mcp-servers\brave-control"


BRAVE_MCP_SERVER_DIR = os.getenv("BRAVE_MCP_SERVER_DIR") or _default_brave_mcp_server_dir()
_BRAVE_MCP_PID = None

# F54 — one starter, one spawn. The watcher warms the daemon at boot while a
# task may call the same starter at the same moment; two spawns racing for one
# port would leave the loser dying at bind time and the port owned by a process
# nobody tracked.
_BRAVE_MCP_START_LOCK = threading.Lock()

#: How long a freshly spawned daemon gets to start listening. Readiness is
#: verified, never assumed — and the wait is bounded so a sick daemon cannot
#: hold a caller (or the boot path) hostage.
try:
    BRAVE_MCP_READY_TIMEOUT = max(
        1.0, float(os.getenv("JARVIS_BRAVE_MCP_READY_TIMEOUT", "8")))
except (TypeError, ValueError):
    BRAVE_MCP_READY_TIMEOUT = 8.0

# ── Spoken narration of task progress ──
#
# The intended second voice: short status phrases spoken WHILE the opencode
# agent works. Deliberately bypasses the voice-mode mute gates (those mute
# JARVIS's own replies, not the agent's narration) and speaks directly.
# Conservative by default: off unless brain hands the task off.
_narration_enabled = False
_NARRATION_MIN_GAP_S = 3.0
_NARRATION_MAX_PER_TASK = 6
_last_narration_at = 0.0
_last_narration_phrase = None
_narration_count = 0

# Real tool-call marker format, observed in data/opencode_activity.log:
#     ? brave-control_navigate {"url":"https://www.amazon.in"}
# (ANSI codes stripped; the tool name may carry an MCP prefix; args optional.)
_TOOL_CALL_RE = re.compile(r"^\?\s+([a-zA-Z0-9_-]+)(?:\s+(.+))?$")


def set_narration_enabled(enabled):
    """Turn spoken progress narration on/off (brain flips it per task)."""
    global _narration_enabled
    _narration_enabled = bool(enabled)


def _reset_narration_state():
    global _last_narration_at, _last_narration_phrase, _narration_count
    _last_narration_at = 0.0
    _last_narration_phrase = None
    _narration_count = 0


def _narrate(phrase):
    """Speak one short phrase, throttled: 3s min gap, no consecutive
    duplicates, hard cap per task. Only speaks when narration is enabled."""
    global _last_narration_at, _last_narration_phrase, _narration_count
    if not _narration_enabled:
        return
    now = time.monotonic()
    if phrase == _last_narration_phrase:
        return
    if now - _last_narration_at < _NARRATION_MIN_GAP_S:
        return
    if _narration_count >= _NARRATION_MAX_PER_TASK:
        return
    _last_narration_at = now
    _last_narration_phrase = phrase
    _narration_count += 1
    try:
        speak(phrase)
    except Exception as exc:
        logging.warning("[OPENCODE] Narration failed: %s", exc)


def _phrase_for_tool(tool_name, args_text):
    """Map a tool-call line to a short spoken phrase (Jarvis tone)."""
    # MCP servers prefix tool names (brave-control_click_element). Drop the
    # prefix only when it is a server name (contains a hyphen); bare tool
    # names like click_element must NOT be split.
    name = tool_name
    head, sep, rest = tool_name.partition("_")
    if sep and ("-" in head or ":" in head):
        name = rest
    if name in ("navigate", "goto", "open"):
        url = ""
        if args_text:
            match = re.search(r'"url"\s*:\s*"([^"]+)"', args_text)
            if match:
                url = match.group(1)
        if url.startswith(("http://", "https://")):
            host = url.split("/")[2]
            return "Opening %s, sir." % host
        return "Opening the page, sir."
    if name == "click_element":
        index = ""
        if args_text:
            match = re.search(r'"index"\s*:\s*(\d+)', args_text)
            if match:
                index = match.group(1)
        return "Clicking element %s, sir." % index if index else "Clicking it, sir."
    if name == "fill_element":
        index = ""
        if args_text:
            match = re.search(r'"index"\s*:\s*(\d+)', args_text)
            if match:
                index = match.group(1)
        return "Typing into element %s, sir." % index if index else "Typing it in, sir."
    if name == "screenshot":
        return "Taking a look, sir."
    if name == "understand_page":
        return "Reading the page, sir."
    return "Working on it, sir."


def _narrate_line(line):
    """Inspect one streamed output line for a tool-call marker and narrate."""
    stripped = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line or "")
    match = _TOOL_CALL_RE.match(stripped)
    if not match:
        return
    _narrate(_phrase_for_tool(match.group(1), match.group(2) or ""))


def _activity_log_path():
    """Absolute path of the live opencode activity log (repo data dir)."""
    return os.path.join(_repo_root(), "data", "opencode_activity.log")


def _ensure_activity_log():
    """Create the activity log's parent dir and the (empty) file if missing."""
    log_path = _activity_log_path()
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if not os.path.exists(log_path):
            with open(log_path, "a", encoding="utf-8", errors="replace"):
                pass
    except Exception as exc:
        logging.warning("[OPENCODE] Could not create activity log: %s", exc)


def _append_activity(text):
    """Append one chunk of text to the activity log, best effort.

    F21: the opencode child prints whatever it read, wrote or diffed — which
    includes the contents of config files, env dumps and anything it typed into
    a form. The log is a persisted artifact, so it crosses the redacted egress
    boundary on the way in.
    """
    try:
        with open(_activity_log_path(), "a", encoding="utf-8", errors="replace") as handle:
            handle.write(tool_policy.redact_for_egress(text)
                         if isinstance(text, str) else str(text))
            handle.flush()
    except Exception:
        pass


def _truncate_activity_log():
    """Empty the activity log so it holds only the current task.

    Runs at the very start of every task; the visible tail script notices
    the shrink, clears its console and follows the fresh log from the top.
    """
    try:
        with open(_activity_log_path(), "w", encoding="utf-8", errors="replace"):
            pass
    except Exception as exc:
        logging.warning("[OPENCODE] Could not truncate activity log: %s", exc)


def _spawn_activity_tail():
    """Open the one visible console window that live-tails the activity log.

    A never-exiting PowerShell script (backend/scripts/tail_activity.ps1)
    polls the absolute log path instead of Get-Content -Wait: it starts at
    the current end of the file and, when the log is truncated (new task),
    clears the screen and follows from the top. Returns the pid for
    shutdown, or None when the spawn failed.
    """
    log_path = _activity_log_path()
    script = os.path.join(_repo_root(), "backend", "scripts", "tail_activity.ps1")
    try:
        proc = subprocess.Popen(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                script,
                "-LogPath",
                log_path,
            ],
            creationflags=_CREATE_NEW_CONSOLE,
        )
        return proc.pid
    except Exception as exc:
        logging.warning("[OPENCODE] Could not start activity tail: %s", exc)
        return None


def _taskkill(pid):
    """Force-kill *pid* and its child tree, best effort."""
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        logging.warning("[OPENCODE] Could not kill pid %s: %s", pid, exc)


def truncate_activity_log():
    """Public wrapper: empty the activity log so it holds only the current task."""
    _truncate_activity_log()


def append_activity_line(text):
    """Public wrapper: append one line of text to the activity log."""
    _append_activity(text)


def narrate_activity(text):
    """Public wrapper: speak one short narration phrase (throttled)."""
    _narrate(text)


def shutdown_opencode_server():
    """Terminate the persistent serve daemon, the activity-tail window AND
    the brave-control MCP daemon, best effort.

    Kills the tracked spawned pids when this process has them; the server
    and the MCP daemon fall back to killing whatever listens on their ports
    when the pid is unknown. Never raises: teardown runs from shutdown paths
    that must not be interrupted.
    """
    global _SERVER_PID, _TAIL_PID, _BRAVE_MCP_PID

    targets = []
    if _SERVER_PID is not None:
        targets.append(_SERVER_PID)
        _SERVER_PID = None
    else:
        targets.extend(_pids_on_port(OPENCODE_SERVE_PORT))
    if _TAIL_PID is not None:
        targets.append(_TAIL_PID)
        _TAIL_PID = None
    if _BRAVE_MCP_PID is not None:
        targets.append(_BRAVE_MCP_PID)
        _BRAVE_MCP_PID = None
    else:
        targets.extend(_pids_on_port(BRAVE_MCP_PORT))

    for pid in targets:
        logging.info("[OPENCODE] Terminating process pid %s", pid)
        _taskkill(pid)


def close_activity_tail():
    """Close the visible activity-tail console, keeping the warm stack.

    The serve daemon (port 9560) and the brave MCP daemon stay alive so the
    next wake is still instant; only the per-session visible tail goes away.
    Safe no-op when no tail pid is tracked (e.g. already closed).
    """
    global _TAIL_PID
    if _TAIL_PID is None:
        return
    pid = _TAIL_PID
    _TAIL_PID = None
    logging.info("[OPENCODE] Closing activity tail pid %s", pid)
    _taskkill(pid)


def _resolve_opencode() -> str:
    """Locate the opencode CLI, preferring the real .exe binary.

    npm global installs create a shim dir with:
      - opencode     (POSIX shell script — useless with shell=False on Win32)
      - opencode.cmd / opencode.ps1 (shell shims — require cmd/powershell)
      - node_modules/opencode-ai/bin/opencode.exe (the actual binary)

    Spawning the .cmd via subprocess.run(shell=True) together with
    CREATE_NO_WINDOW/DETACHED_PROCESS is fragile ("Access is denied",
    exit-3 crashes), so we resolve the native .exe and exec it directly.
    """
    # 1) Explicit override.
    override = os.getenv("JARVIS_OPENCODE_CMD")
    if override:
        resolved = shutil.which(override)
        if resolved:
            if override.lower().endswith(".exe") or _is_native_binary(resolved):
                return resolved

    # 2) Autodetect the real binary next to any .cmd shim on PATH.
    for name in (OPENCODE_CMD,):
        shim = shutil.which(name + ".cmd") or shutil.which(name)
        if not shim:
            continue
        shim_dir = os.path.dirname(os.path.abspath(shim))
        bin_candidates = (
            os.path.join(shim_dir, "node_modules", "opencode-ai", "bin", "opencode.exe"),
            os.path.join(shim_dir, "opencode.exe"),
        )
        for candidate in bin_candidates:
            if os.path.isfile(candidate):
                return candidate

        # Fall back to the .cmd shim only if no native binary exists.
        if shim.lower().endswith(".cmd") or not _is_native_binary(shim):
            return shim

    # 3) Last resort: bare which().
    resolved = shutil.which(OPENCODE_CMD)
    if resolved and _is_native_binary(resolved):
        return resolved
    return resolved or ""


def _is_native_binary(path):
    """True when *path* is a directly executable file (exe), not a script."""
    lower = path.lower()
    return lower.endswith(".exe")


def is_opencode_server_alive() -> bool:
    """True when the persistent opencode serve daemon answers on its port."""
    if not _OPENCODE_SERVE:
        return False
    try:
        request = Request(f"{OPENCODE_SERVE_URL}/session", method="GET")
        with urlopen(request, timeout=1.0) as response:
            return response.status == 200
    except Exception:
        return False


def _build_command(task: str):
    """Return the argv list to spawn for a headless opencode run.

    When the persistent opencode serve daemon is reachable, attach to it so
    the warm agent (model already resident) handles the task instead of a
    cold-spawned instance.
    """
    executable = _resolve_opencode()
    if is_opencode_server_alive():
        logging.info("[OPENCODE] Attaching to persistent server at %s", OPENCODE_SERVE_URL)
        command = [executable, "run", task, "--attach", OPENCODE_SERVE_URL]
    else:
        command = [executable, "run", task]
    if _OPENCODE_AUTO:
        command.append("--auto")
    # Same pinned model for both the attach and the plain run variant.
    command.extend(["-m", _OPENCODE_MODEL])
    return command


def is_opencode_available() -> bool:
    try:
        return bool(_resolve_opencode())
    except Exception:
        return False


def _contract_dispatch_refusal(contract):
    """Empty string when *contract* authorizes an opencode run, else why not.

    F16 — "carry required capability, selected executor, availability, and
    explicit grant through immutable approval/execution". A contract frozen
    at consent time is the ONLY authority that can override the live
    configuration; a tampered/foreign/grantless contract is refused before
    any spawn.
    """
    if contract is None:
        return ""
    try:
        contract.verify(executor=OPENCODE_ENGINE, require_grant=True)
    except Exception as exc:
        return str(exc) or "invalid capability contract"
    return ""


def _append_dispatch_refusal(reason):
    """Record a refused dispatch in the activity log (best effort)."""
    try:
        _append_activity("=== %s === [dispatch refused: %s]\n"
                         % (time.strftime("%Y-%m-%d %H:%M:%S"), reason))
    except Exception:
        pass


def run_opencode_task(task: str, timeout: int = None,
                      contract=None) -> TaskResult:
    """Run a task through opencode and return a TaskResult.

    Uses `opencode run <task>` in the project directory so opencode picks
    up the repo's configured model, skills, and permissions. The run stays
    hidden (CREATE_NO_WINDOW); every output line is appended to the activity
    log (live-streamed by the visible tail window) AND accumulated as the
    result detail.

    The result is str-compatible with the old plain-string contract, and
    every failure status is FALSY, so existing `if output:` guards keep
    meaning 'the task produced usable output':
      * engine refused / CLI missing / spawn failure / timeout ->
        TaskResult.failed (falsy; the old code returned '').
      * exit 0 (or None) -> TaskResult.completed with the transcript.
      * nonzero exit -> TaskResult.failed with error
        'opencode exited with code <rc>' and the transcript as detail
        (the exit status is now preserved, no longer only logged).

    F16 — dispatch authority travels WITH the call. When *contract* (an
    immutable :class:`~backend.services.capability_contract.ExecutionContract`
    frozen at consent time) is supplied, this function verifies that the
    contract names ``opencode`` as the executor and carries an explicit user
    grant; the LIVE config is then irrelevant, so a ``TASK_ENGINE`` change
    after consent cannot change the executor. Without a contract (legacy
    callers) the configured engine gate below still applies, and a
    configuration error now fails CLOSED instead of authorizing a spawn.
    """
    refusal = _contract_dispatch_refusal(contract)
    if refusal:
        logging.warning("[OPENCODE] Dispatch refused: %s", refusal)
        _append_dispatch_refusal(refusal)
        return TaskResult.failed("opencode dispatch refused: %s" % refusal)
    if contract is None and not _opencode_engine_enabled():
        logging.warning("[OPENCODE] Engine detached (TASK_ENGINE != opencode); task refused.")
        return TaskResult.failed("opencode engine detached")
    executable = _resolve_opencode()
    if not executable:
        logging.warning("[OPENCODE] CLI not found; cannot hand off task.")
        return TaskResult.failed("opencode CLI not found")

    _reset_narration_state()
    base = os.environ.get("JARVIS_PROJECT_DIR") or _repo_root()
    command = _build_command(task)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    _truncate_activity_log()
    _append_activity("\n=== %s === %s ===\n" % (stamp, task))

    # F20: this run owns a cancellation token. A stop request cancels the
    # job, which terminates the whole opencode process tree — not just the
    # wrapper the way a bare proc.kill() did.
    from backend.services import jobs as job_registry
    job = job_registry.new_job(kind="opencode", label=(task or "")[:80])

    try:
        proc = subprocess.Popen(
            command,
            cwd=base,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=_CREATE_NO_WINDOW,
        )
        job.attach(proc)
    except Exception as exc:
        job.finish()
        _append_activity("=== %s === [opencode spawn failed: %s]\n" % (stamp, exc))
        logging.warning("[OPENCODE] Spawn failed: %s", exc)
        return TaskResult.failed("opencode spawn failed: %s" % exc)

    line_q = queue.Queue()

    def _reader():
        for line in iter(proc.stdout.readline, ""):
            line_q.put(line)
        line_q.put(None)

    threading.Thread(target=_reader, daemon=True).start()

    # Stream lines to the activity log while accumulating the transcript.
    # Deadline is re-checked on every loop turn so a quiet child still gets
    # killed at the timeout (the reader thread drains via the queue).
    deadline = time.monotonic() + (timeout or _TIMEOUT)
    lines = []
    timed_out = False
    cancelled = False
    try:
        while True:
            if job.should_stop():
                cancelled = True
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                line = line_q.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                if job.should_stop():
                    cancelled = True
                    break
                continue
            if line is None:
                break
            lines.append(line)
            _append_activity(line)
            _narrate_line(line)
    finally:
        job.finish()

    if cancelled:
        # terminate_processes is idempotent: whoever cancelled the job has
        # already run it, and the Job Object / taskkill takes the child tree
        # down, never just the wrapper.
        job.terminate_processes()
        _append_activity("[opencode] task cancelled by user\n")
        return TaskResult.stopped()

    if timed_out:
        job.terminate_processes()
        _append_activity("[opencode] task timed out after %ss\n" % (timeout or _TIMEOUT))
        logging.warning("[OPENCODE] Hand-off timed out after %ss.", timeout or _TIMEOUT)
        return TaskResult.failed(
            "opencode timed out after %ss" % (timeout or _TIMEOUT))

    raw = "".join(lines)
    # F21: the captured output is returned to the caller, which puts it in the
    # model transcript, the UI and (for a failure) the result detail — so it
    # crosses the redacted egress boundary here too, not just into the log.
    output = tool_policy.redact_for_egress(_clean_attach_output(raw)).strip()
    try:
        proc.wait(timeout=5)
    except Exception:
        pass
    if proc.returncode not in (0, None):
        logging.warning("[OPENCODE] exit=%s", proc.returncode)
        return TaskResult.failed(
            "opencode exited with code %s" % proc.returncode, detail=output)
    if proc.returncode is None:
        # F03: the wait above failed, so the exit status is UNKNOWN. An
        # unobserved status is not success — the transcript is kept as partial
        # evidence instead of being published as verified completion.
        logging.warning("[OPENCODE] exit status unknown after wait")
        return TaskResult.partial(
            output or "opencode finished with an unknown exit status.",
            detail=output, evidence=["exit status was never observed"],
            unmet_goals=["confirm whether the task actually finished"])
    return result_from_reported_text(
        output, detail=output, assume_success=True)


def _repo_root() -> str:
    """Best-effort repo root: two levels up from this services file."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


def _clean_attach_output(text: str) -> str:
    """Make raw attached-run output speakable: drop ANSI codes, prompts and
    blank lines, keep the meaningful trailing lines."""
    try:
        text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text or "")
    except Exception:
        text = text or ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    # Drop the "> build · model" banner line.
    if lines and lines[0].startswith(">"):
        lines = lines[1:]
    return "\n".join(lines)


def _pids_on_port(port):
    pids = set()
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
        )
    except Exception:
        return pids

    marker = f":{port}"
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        if marker not in parts[1] or parts[-2].upper() != "LISTENING":
            continue
        try:
            pids.add(int(parts[-1]))
        except ValueError:
            continue
    return pids


def ensure_activity_tail():
    """Start (or reuse) the visible activity-tail console. Idempotent:
    spawns the tail script only when no tail pid is tracked, so both the
    opencode engine and the browser-agent engine share one console."""
    global _TAIL_PID
    _ensure_activity_log()
    if _TAIL_PID is None:
        _TAIL_PID = _spawn_activity_tail()


def _task_engine():
    """The configured hand-off engine, read through ONE seam (F16).

    Kept as a function so the fail-closed behaviour below is testable: a
    configuration read that raises must never be an authorization.
    """
    from backend import config
    return config.TASK_ENGINE


def _opencode_engine_enabled():
    """True only when the opencode engine is selected. With the default
    browser_agent engine opencode is fully detached: no serve daemon, no
    task subprocess — the code stays, but nothing may spawn it.

    F16 — this gate FAILS CLOSED. It used to ``return True`` when reading
    the configuration raised, so a broken/missing config silently authorized
    spawning the coding CLI ("error paths can fail open").
    """
    try:
        return _task_engine() == OPENCODE_ENGINE
    except Exception as exc:
        logging.warning(
            "[OPENCODE] Engine configuration unreadable (%s); failing closed.",
            exc)
        return False


def kill_stale_opencode_server():
    """Kill any opencode serve daemon left over from earlier sessions.

    Best effort: when the engine is browser_agent the watcher calls this
    at boot so a pre-switch daemon on the serve port cannot linger.
    """
    for pid in _pids_on_port(OPENCODE_SERVE_PORT):
        logging.info("[OPENCODE] Killing stale serve daemon (pid %s)", pid)
        _taskkill(pid)


def ensure_opencode_server(attempts=2):
    """Start (or reuse) the persistent opencode serve daemon + activity tail.

    The watcher calls this at boot so the opencode agent stays warm between
    tasks. Returns True when a server is reachable, False otherwise. The
    serve daemon itself is quiet (HTTP) so it spawns hidden; the visible
    part is one console window tailing the activity log, where the per-task
    `opencode run` output streams as it happens. Both pids are remembered
    for a clean teardown by shutdown_opencode_server().

    Detached unless TASK_ENGINE selects the opencode engine.
    """
    global _SERVER_PID

    if not _opencode_engine_enabled():
        logging.info("[OPENCODE] Engine detached (TASK_ENGINE != opencode); not started.")
        return False
    ensure_activity_tail()

    if is_opencode_server_alive():
        logging.info("[OPENCODE] Persistent server already running at %s", OPENCODE_SERVE_URL)
        return True

    executable = _resolve_opencode()
    if not executable:
        logging.warning("[OPENCODE] CLI not found; cannot start serve daemon.")
        return False

    for attempt in range(1, attempts + 1):
        if attempt > 1:
            for pid in _pids_on_port(OPENCODE_SERVE_PORT):
                logging.warning("[OPENCODE] Killing stale listener on port %s (pid %s)", OPENCODE_SERVE_PORT, pid)
                _taskkill(pid)
            time.sleep(1.0)

        try:
            proc = subprocess.Popen(
                [executable, "serve", "--port", str(OPENCODE_SERVE_PORT)],
                creationflags=_CREATE_NO_WINDOW,
            )
            _SERVER_PID = proc.pid
        except Exception as exc:
            logging.warning("[OPENCODE] Could not start serve daemon (attempt %s): %s", attempt, exc)
            continue

        deadline = time.time() + 30
        while time.time() < deadline:
            if is_opencode_server_alive():
                logging.info("[OPENCODE] Persistent server ready at %s", OPENCODE_SERVE_URL)
                return True
            time.sleep(0.5)

        # The spawned process never became ready; drop the tracked pid so a
        # later retry (or shutdown) does not chase a dead handle.
        if _SERVER_PID is not None:
            logging.warning("[OPENCODE] Serve daemon did not become ready (pid %s) - killing it.", _SERVER_PID)
            _taskkill(_SERVER_PID)
            _SERVER_PID = None

    logging.warning("[OPENCODE] Serve daemon did not become ready.")
    return False


def _brave_mcp_log_path():
    """Absolute path of the brave-MCP daemon's captured output (F54)."""
    return os.path.join(_repo_root(), "data", "logs", "brave_mcp.log")


def _spawn_brave_mcp_daemon(executable, env):
    """Spawn the node daemon hidden, capturing its output to a log file (F54).

    The daemon runs under CREATE_NO_WINDOW, so unredirected stdout/stderr went
    nowhere at all: when the vendored copy of the server had no
    ``node_modules``, the child died instantly with ERR_MODULE_NOT_FOUND and
    the only symptom anyone could see was a bare "connection refused" from the
    MCP client much later. The child's own words now land in the log that the
    failure message points at, so a dead daemon explains itself.
    """
    log_path = _brave_mcp_log_path()
    handle = None
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handle = open(log_path, "a", encoding="utf-8", errors="replace")
        handle.write("\n=== %s starting brave-MCP daemon ===\n"
                     % time.strftime("%Y-%m-%d %H:%M:%S"))
        handle.flush()
    except Exception as exc:
        logging.warning("[BRAVE-MCP] Could not open daemon log %s: %s",
                        log_path, exc)
        handle = None

    output = handle if handle is not None else subprocess.DEVNULL
    try:
        return subprocess.Popen(
            [executable, "server.mjs"],
            cwd=BRAVE_MCP_SERVER_DIR,
            env=env,
            creationflags=_CREATE_NO_WINDOW,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    finally:
        # The child holds its own descriptor; this process must not keep one.
        if handle is not None:
            handle.close()


def ensure_brave_mcp_daemon():
    """Start the persistent brave-control MCP HTTP daemon if needed.

    Called by the watcher at boot (next to ensure_opencode_server) so the
    browser is launched once per jarvis boot and shared by every opencode
    session via the remote MCP registration in ~/.config/opencode. The
    daemon runs hidden (CREATE_NO_WINDOW) with BRAVE_MCP_MODE=http, token
    auth, and its own idle browser recycle. Idempotent: when port 9570 is
    already listening the daemon is left untouched. The spawned pid is
    remembered for a clean teardown by shutdown_opencode_server().

    F54 — the return value is now the TRUTH about the port. It used to end
    with ``return _BRAVE_MCP_PID is not None``, which reported success for a
    child that had already crashed, so the browser agent skipped its own
    "daemon could not be started" branch and walked straight into a
    connection-refused from the MCP client (the confusing failure the user
    saw). A dead child is also detected immediately instead of being waited
    out: the readiness loop cost a full 10s on every boot with a broken
    install, which is exactly the delay users saw between the activity
    console and the next one.
    """
    global _BRAVE_MCP_PID

    with _BRAVE_MCP_START_LOCK:
        if _pids_on_port(BRAVE_MCP_PORT):
            logging.info("[BRAVE-MCP] Daemon already listening on port %s", BRAVE_MCP_PORT)
            return True

        executable = shutil.which("node") or "node"
        env = dict(os.environ)
        env["BRAVE_MCP_MODE"] = "http"
        env["BRAVE_MCP_PORT"] = str(BRAVE_MCP_PORT)
        if BRAVE_MCP_TOKEN:
            env["BRAVE_MCP_TOKEN"] = BRAVE_MCP_TOKEN
        try:
            proc = _spawn_brave_mcp_daemon(executable, env)
            _BRAVE_MCP_PID = proc.pid
        except Exception as exc:
            logging.warning("[BRAVE-MCP] Could not start daemon: %s", exc)
            return False

        deadline = time.time() + BRAVE_MCP_READY_TIMEOUT
        while time.time() < deadline:
            if _pids_on_port(BRAVE_MCP_PORT):
                logging.info("[BRAVE-MCP] Daemon ready on port %s (pid %s)",
                             BRAVE_MCP_PORT, _BRAVE_MCP_PID)
                return True
            if proc.poll() is not None:
                # A child that already exited will never bind the port. Fail in
                # about a second with its own error, not after the whole wait.
                # The tracked pid is dropped because the process is gone: a
                # dead pid can be recycled by Windows, and a teardown that
                # killed the recycled one would hit a foreign process (F52).
                logging.warning(
                    "[BRAVE-MCP] Daemon exited immediately (exit code %s) - port "
                    "%s cannot be served. Its output is in %s.",
                    proc.returncode, BRAVE_MCP_PORT, _brave_mcp_log_path())
                _BRAVE_MCP_PID = None
                return False
            time.sleep(0.3)
        logging.warning(
            "[BRAVE-MCP] Daemon did not become ready on port %s within %.0fs - "
            "its output is in %s.",
            BRAVE_MCP_PORT, BRAVE_MCP_READY_TIMEOUT, _brave_mcp_log_path())
        return False