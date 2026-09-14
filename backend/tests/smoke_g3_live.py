"""Live smoke test for G3 (F23/F26/F30) — boots the real FastAPI app and
drives an SSE stream through attach -> disconnect -> resume, checks the
/ask execute-once guarantee, and exercises the screen-answer
publish/patch/stale-409 flow.

This is intentionally NOT collected by pytest (no test_*.py name): it needs
a live server and real provider configuration. Run it manually:

    backend/venv/Scripts/python.exe backend/tests/smoke_g3_live.py

Exits 0 when every check passes, 1 otherwise.
"""

import json
import os
import subprocess
import sys
import time

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PORT = 9998
BASE = f"http://127.0.0.1:{PORT}"

FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    if not ok:
        FAILURES.append(name)


def wait_for_server(proc, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            r = requests.get(BASE + "/health", timeout=2)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    return False


def sse_frames(response):
    """Yield parsed JSON payloads from an SSE response line stream."""
    buf = ""
    for raw in response.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        if raw == "":
            # End of one SSE event; find its data line.
            for line in buf.split("\n"):
                if line.startswith("data:"):
                    try:
                        yield json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        pass
            buf = ""
        else:
            buf += raw + "\n"


def attach_stream(request_id, message, last_event_id=-1, max_seconds=95.0):
    """POST /ask/stream and consume frames; returns (frames, saw_terminal)."""
    frames = []
    saw_terminal = False
    with requests.post(
        BASE + "/ask/stream",
        json={
            "message": message,
            "request_id": request_id,
            "last_event_id": last_event_id,
        },
        stream=True,
        timeout=(5, max_seconds),
    ) as resp:
        resp.raise_for_status()
        for payload in sse_frames(resp):
            frames.append(payload)
            if payload.get("type") in ("completed", "interrupted", "error"):
                saw_terminal = True
                break
    return frames, saw_terminal


def read_frames_for(request_id, message, seconds):
    """Attach and read for at most *seconds*, then drop the connection."""
    frames = []

    def _consume():
        try:
            with requests.post(
                BASE + "/ask/stream",
                json={
                    "message": message,
                    "request_id": request_id,
                    "last_event_id": -1,
                },
                stream=True,
                timeout=(5, seconds + 5),
            ) as resp:
                resp.raise_for_status()
                for payload in sse_frames(resp):
                    frames.append(payload)
                    if payload.get("type") in ("completed", "interrupted",
                                               "error"):
                        return
        except requests.RequestException:
            pass

    import threading
    t = threading.Thread(target=_consume, daemon=True)
    t.start()
    t.join(timeout=seconds)
    # Leaving the thread blocked on read: the daemon thread dies with the
    # process; the requests response is closed by GC. The important part is
    # what we captured before disconnecting.
    return frames


def main():
    env = dict(os.environ)
    env["JARVIS_BACKEND_PORT"] = str(PORT)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app:app",
         "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=REPO_ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_for_server(proc):
            print("FATAL: server did not come up")
            return 1

        rid = f"req-smoke-{int(time.time())}"
        message = "hello there"

        # ── Phase A: attach, read briefly, disconnect ────────────────
        first = read_frames_for(rid, message, seconds=3.0)
        check("attach yields frames", len(first) > 0,
              f"{len(first)} frames before disconnect")
        numbered = [f["seq"] for f in first
                    if isinstance(f.get("seq"), int)]
        check("frames carry seq + request_id",
              all(f.get("request_id") == rid for f in first),
              f"seqs seen: {numbered[:6]}")
        check("numbered seqs are strictly increasing",
              numbered == sorted(numbered) and len(set(numbered)) == len(numbered),
              f"seqs: {numbered[:6]}")
        cursor = max(numbered) if numbered else -1
        terminal = next(
            (f for f in first
             if f.get("type") in ("completed", "interrupted", "error")), None)
        fast_terminal = terminal is not None
        # A fast worker may finish while we are still attached. To keep the
        # resume check deterministic, pretend the terminal frame was never
        # seen and replay it from the buffer — the assertion is the same
        # either way: reconnecting after event k must replay exactly the
        # frames with seq > k and then end, never re-executing the message.
        resume_from = cursor - 1 if fast_terminal else cursor

        # ── Phase B: reconnect and resume after the cursor ───────────
        resumed, terminal_seen = attach_stream(rid, message, resume_from)
        check("resume produces a terminal frame", terminal_seen)
        replayed = [f["seq"] for f in resumed
                    if isinstance(f.get("seq"), int)]
        check("resume never re-sends an old event",
              all(s > resume_from for s in replayed),
              f"cursor {resume_from}, replayed {replayed[:6]}")
        if terminal is None:
            terminal = next(
                (f for f in resumed
                 if f.get("type") in ("completed", "interrupted", "error")),
                {})
        if fast_terminal:
            replayed_terminal = next(
                (f for f in resumed
                 if f.get("type") in ("completed", "interrupted", "error")),
                {})
            check("replayed terminal matches the original",
                  replayed_terminal.get("reply") == terminal.get("reply"),
                  f"replayed {replayed_terminal.get('reply')!r:.60}")
        reply = terminal.get("reply", "")

        # ── /ask/status ──────────────────────────────────────────────
        r = requests.get(BASE + f"/ask/status/{rid}", timeout=5)
        check("status lookup 200", r.status_code == 200)
        snap = r.json()
        check("status says done", snap.get("done") is True, str(snap))
        if terminal.get("type") == "completed":
            check("status reply matches terminal reply",
                  snap.get("reply") == reply)

        r = requests.get(BASE + "/ask/status/req-does-not-exist", timeout=5)
        check("unknown request id 404", r.status_code == 404)

        # ── /ask execute-once (retried POST must not re-run) ─────────
        t0 = time.time()
        r = requests.post(BASE + "/ask",
                          json={"message": message, "request_id": rid},
                          timeout=30)
        elapsed = time.time() - t0
        check("retried /ask returns quickly", elapsed < 20,
              f"{elapsed:.1f}s")
        data = r.json()
        if terminal.get("type") == "completed":
            check("retried /ask returns the same reply",
                  data.get("reply") == reply,
                  f"got {data.get('reply')!r:.60}")

        # ── F30: publish / patch / stale rejection ───────────────────
        cap = "cap-smoke-1"
        r = requests.post(BASE + "/screen-answer", json={
            "tip": "tip", "evidence": [], "links": [], "images": [],
            "capture_id": cap, "request_id": rid,
        }, timeout=5)
        answer_id = r.json().get("id")
        check("screen answer publish allocates an id", bool(answer_id))

        r = requests.post(BASE + "/screen-answer", json={
            "tip": "tip", "id": answer_id, "capture_id": cap,
            "links": [{"label": "More", "url": "http://x"}],
        }, timeout=5)
        check("patch same id updates in place",
              r.status_code == 200 and r.json().get("updated") is True)

        r = requests.post(BASE + "/screen-answer", json={
            "tip": "tip", "id": answer_id, "capture_id": "cap-other",
            "links": [{"label": "St", "url": "http://y"}],
        }, timeout=5)
        check("stale capture rejected 409", r.status_code == 409)

        r = requests.post(BASE + "/screen-answer", json={
            "tip": "tip", "id": answer_id + 99, "capture_id": cap,
        }, timeout=5)
        check("stale answer id rejected 409", r.status_code == 409)

        r = requests.get(BASE + "/screen-answer", timeout=5)
        stored = r.json()
        check("stored answer kept its id and gained the link",
              stored.get("id") == answer_id and len(stored.get("links", [])) == 1
              and stored.get("enriched") is True)

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    print()
    if FAILURES:
        print(f"SMOKE FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("SMOKE OK — all G3 live checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
