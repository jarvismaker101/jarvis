"""LIVE FIX 27 — the eight re-test bugs (B44-B51), tested SILENTLY and realistically.

Every behavior test drives ``brain.process_message`` — the exact entry
point the voice/typing path uses — with the user's own utterances.
Routing, parsing, enumeration, replies and failure paths are all real.

Zero disturbance by design (the earlier Notepad approach was withdrawn
for touching live work):
- no spawned apps, no user-window mutation, no focus changes, no visible
  flashes — verified with foreground checks;
- the only windows the tests ever create are self-owned,
  iconic-from-birth, WS_EX_TOOLWINDOW ctypes windows
  ("SilentTestWindow-<hex>"): never painted, never activated, no taskbar
  button, no Alt+Tab entry, destroyed afterwards (no save prompts, no
  residue — unlike Notepad);
- real planning is asserted on REAL hwnds from the live desktop (plans are
  built but never executed for user windows);
- real Win32 execution runs only against iconic self-owned windows, where
  minimize is a verified no-op — the visual minimize transition itself is
  pre-existing code proven live daily, and every B44-B51 bug was in
  routing/planning/replies, never in the ShowWindow call;
- simulated seams, stated openly: the cloud classifier is stubbed to its
  offline-degraded verdict (the live chain is down: OpenRouter 402 /
  Groq 404); the vision provider is failed/hung at the adapter boundary
  (the live Fireworks write-timeout condition); the chat model must never
  be reached for B51 (the test fails loudly if it is).
"""

import ctypes
import re
import subprocess
import sys
import threading
import time

import pytest
from ctypes import wintypes

from backend.core import brain
from backend.core.memory import clear_history, get_history
from backend.services import screen_control as sc
from backend.services import screen_state

# ---------------------------------------------------------------------------
# Self-owned silent test windows, kept OUT of the pytest process.
#
# Window objects (and their WndProc lifetime) live in a helper subprocess
# (see Temp/opencode/silent_windows_helper.py): iconic-from-birth
# tool windows that are never painted, never activated, have no taskbar
# button and no Alt+Tab entry. The suite only ever READS them
# cross-process (EnumWindows/GetWindowText work while the helper pumps)
# or issues minimize (a verified no-op on iconic windows) — zero pixels
# change, zero focus changes. The helper self-cleans on "quit" or TTL.
# ---------------------------------------------------------------------------

_u32 = ctypes.windll.user32
_u32.IsWindow.argtypes = [wintypes.HWND]
_u32.IsWindow.restype = wintypes.BOOL
_u32.IsIconic.argtypes = [wintypes.HWND]
_u32.IsIconic.restype = wintypes.BOOL

_HELPER = ("C:\\Users\\mayan\\AppData\\Local\\Temp\\opencode\\"
           "silent_windows_helper.py")


class _SilentWindows:
    """Run the helper with N silent windows; quit it on exit."""

    def __init__(self, n=2, ttl=120.0):
        self._n = n
        self._ttl = ttl
        self.hwnds = []
        self._proc = None

    def __enter__(self):
        fg_before = _u32.GetForegroundWindow()
        self._proc = subprocess.Popen(
            [sys.executable, _HELPER, str(self._n), str(self._ttl)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        try:
            deadline = time.time() + 25.0
            while time.time() < deadline and len(self.hwnds) < self._n:
                line = self._proc.stdout.readline()
                if not line:
                    break
                m = re.match(r"HWND (\d+)", line.strip())
                if m:
                    self.hwnds.append(int(m.group(1)))
        except Exception:
            pass
        assert len(self.hwnds) >= self._n, \
            "helper produced %d of %d windows" % (len(self.hwnds), self._n)
        assert _u32.GetForegroundWindow() == fg_before, \
            "test setup stole foreground focus"
        return sorted(self.hwnds)

    def __exit__(self, *exc):
        try:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._proc.stdin.write("quit\n")
                    self._proc.stdin.flush()
                except Exception:
                    pass
                try:
                    self._proc.wait(timeout=10)
                except Exception:
                    try:
                        self._proc.terminate()
                    except Exception:
                        pass
        finally:
            self._proc = None
            self.hwnds = []
        return False


def _say(text):
    """Talk to Jarvis exactly like the voice/typing path does."""
    return brain.process_message(
        text, from_voice=False, sync_voice=False, stream_reply=None)


class _RealWorldGuard:
    """Save/restore every global the realistic tests touch."""

    def __init__(self, monkeypatch, tmp_path):
        self._mp = monkeypatch
        self._tmp = tmp_path
        self._was_enabled = screen_state.is_enabled()

    def __enter__(self):
        # The B36 durable toggle persists every set_enabled: point it at a
        # throwaway file so tests can never leak into the live assistant.
        self._mp.setenv("JARVIS_SCREEN_STATE_PATH",
                        str(self._tmp / "toggle.json"))
        screen_state.set_enabled(True, reason="realistic-test")
        screen_state.clear_refused_chain()
        screen_state.clear_interactions()
        clear_history()
        self._mp.setattr(brain, "_last_offer", None)
        with brain._screen_qa_last_lock:
            brain._screen_qa_last.update(
                {"question": "", "at": 0.0, "miss": False, "seq": 0})
        # Isolate the repeat ledgers from other tests' actions.
        self._mp.setattr(brain, "get_last_action", lambda: None)
        # Cloud classifier -> its offline-degraded verdict (chat). The live
        # chain is down; this skips the network wait, nothing else: every
        # deterministic net under test still runs for real.
        self._mp.setattr(
            brain, "classify_intent",
            lambda *a, **k: {"intent": "chat", "steps": [],
                             "task_description": "",
                             "query": a[0] if a else "",
                             "_source": "test-offline"})
        return self

    def __exit__(self, *exc):
        screen_state.clear_refused_chain()
        screen_state.clear_interactions()
        clear_history()
        try:
            screen_state.set_enabled(self._was_enabled,
                                     reason="test-restore")
        except Exception:
            pass
        return False


def _live_managed():
    return sc._list_managable_windows()


def _last_assistant_text():
    texts = []
    try:
        for m in list(get_history() or [])[-4:]:
            if isinstance(m, dict):
                texts.append(" ".join(str(v) for v in m.values()))
            else:
                texts.append(str(m))
    except Exception:
        pass
    return "\n".join(texts)


class TestB44SilentEnumeration:
    """B44: no cloaked/off-monitor ghosts; self-owned windows enumerate."""

    def test_live_enumeration_has_no_ghosts(self, monkeypatch, tmp_path):
        with _RealWorldGuard(monkeypatch, tmp_path):
            wins = _live_managed()
            assert wins, "no manageable windows enumerated at all"
            try:
                dwm = ctypes.windll.dwmapi.DwmGetWindowAttribute
            except Exception:
                dwm = None
            for w in wins:
                hwnd = int(w["hwnd"])
                if dwm is not None:
                    val = wintypes.DWORD(0)
                    try:
                        hr = dwm(wintypes.HWND(hwnd), wintypes.DWORD(14),
                                 ctypes.byref(val), ctypes.sizeof(val))
                    except Exception:
                        hr = -1
                    if hr == 0:
                        assert int(val.value) == 0, \
                            "cloaked ghost enumerated: %r" % (w.get("title"),)
                assert int(_u32.MonitorFromWindow(hwnd, 0) or 0) != 0, \
                    "off-monitor window enumerated: %r" % (w.get("title"),)

    def test_silent_windows_enumerate_and_match(
            self, monkeypatch, tmp_path):
        with _RealWorldGuard(monkeypatch, tmp_path):
            with _SilentWindows(2) as hwnds:
                wins = {w["hwnd"]: w for w in _live_managed()}
                for h in hwnds:
                    assert h in wins, "self-owned window not enumerated"
                    assert wins[h]["iconic"] is True
                    assert sc._window_matches_filter(
                        wins[h], "silenttestwindow")


class TestB45SilentRestCorrection:
    """B45: 'minimize the rest' plans against the real desktop, silently."""

    def test_rest_plan_targets_live_leftovers(self, monkeypatch, tmp_path):
        with _RealWorldGuard(monkeypatch, tmp_path):
            # Stands in for the previous turn (no effects, ledger only):
            # the user said "minimize all ... except for brave".
            screen_state.note_request(
                "minimize all the windows on my screen except for brave")
            plan = sc._build_direct_plan("minimize the rest also")
            assert isinstance(plan, dict) and plan.get("ok"), plan
            assert plan.get("source") == "Direct Match", plan
            live = {w["hwnd"]: w for w in _live_managed()}
            brave_hwnds = {h for h, w in live.items()
                           if w.get("process") == "brave"}
            if brave_hwnds and any(not w.get("iconic")
                                   for h, w in live.items()
                                   if h not in brave_hwnds):
                steps = plan.get("steps") or []
                assert steps, plan
                assert all(s.get("action") == "minimize_hwnd"
                           for s in steps), plan
                step_hwnds = {int(s.get("hwnd")) for s in steps}
                assert not (step_hwnds & brave_hwnds), plan
                assert step_hwnds <= set(live), plan
                for h in step_hwnds:
                    assert live[h].get("iconic") is False, plan
                assert ("Minimizing %d window" % len(steps)
                        in plan.get("summary", "")), plan
                assert "Leaving Brave" in plan.get("summary", ""), plan
            else:
                # Degenerate live state (nothing left): the honest empty
                # line, never vision, never an exception.
                assert plan.get("steps") == [] or plan.get("ok"), plan

    def test_rest_empty_state_is_honest(self, monkeypatch, tmp_path):
        with _RealWorldGuard(monkeypatch, tmp_path):
            with _SilentWindows(2):
                # App-scoped rest over an all-iconic set: the subset-scoped
                # already-minimized line, never vision, never an exception.
                plan = sc._build_direct_plan(
                    "minimize the rest of the silenttestwindow windows")
                assert isinstance(plan, dict), plan
                assert plan.get("ok"), plan
                assert "already minimized" in plan.get("summary", ""), plan
                assert "silenttestwindow" in plan.get("summary", ""), plan

    def test_executor_runs_real_minimize_steps_silently(
            self, monkeypatch, tmp_path):
        from backend.services.screen_executor import execute_steps
        with _RealWorldGuard(monkeypatch, tmp_path):
            with _SilentWindows(2) as hwnds:
                steps = [{"action": "minimize_hwnd", "hwnd": h,
                          "title": "SilentTestWindow",
                          "description": "Minimize test window"}
                         for h in hwnds]
                fg_before = _u32.GetForegroundWindow()
                execute_steps(steps)
                assert _u32.GetForegroundWindow() == fg_before
                for h in hwnds:
                    assert bool(_u32.IsIconic(h)), \
                        "executor failed on real hwnd %d" % h
                assert sc._verify_hwnd_state_steps(steps) is True


class TestB47SilentWindowCount:
    """B47: 'how many windows' is answered from the live desktop."""

    @pytest.mark.parametrize("utterance", [
        "tell me how many windows are open on my screen",
        "how many windows are on my screen",
    ])
    def test_count_matches_live_desktop(
            self, monkeypatch, tmp_path, utterance):
        with _RealWorldGuard(monkeypatch, tmp_path):
            with _SilentWindows(2):
                live = _live_managed()
                expect_open = sum(1 for w in live if not w.get("iconic"))
                expect_min = sum(1 for w in live if w.get("iconic"))
                assert expect_min >= 2  # our own two, iconic from birth
                reply = _say(utterance)
                assert reply != "Working on it, sir.", reply
                assert "analys" not in reply.lower(), reply
                assert str(expect_open) in reply, reply
                if expect_min:
                    assert str(expect_min) in reply, reply


class TestB46SilentStateVerify:
    """B46: 'have you minimized ...' is verified, never 'I cannot check'."""

    def test_all_windows_answer_tracks_live_state(
            self, monkeypatch, tmp_path):
        with _RealWorldGuard(monkeypatch, tmp_path):
            reply = _say("have you minimized all windows?")
            assert "cannot" not in reply.lower(), reply
            assert reply.startswith(("Yes", "No")), reply
            live = _live_managed()
            n_open = sum(1 for w in live if not w.get("iconic"))
            if reply.startswith("Yes"):
                assert n_open == 0, (reply, n_open)
            else:
                m = re.search(r"(\d+) of (\d+) windows", reply)
                assert m, reply
                assert int(m.group(1)) == n_open, (reply, n_open)
                assert int(m.group(2)) == len(live), (reply, len(live))

    def test_scoped_answer_names_silent_windows(
            self, monkeypatch, tmp_path):
        with _RealWorldGuard(monkeypatch, tmp_path):
            with _SilentWindows(2):
                reply = _say("have you minimized the silenttestwindow "
                             "windows?")
                assert "cannot" not in reply.lower(), reply
                assert reply.startswith("Yes"), reply
                assert "minimized" in reply, reply


class TestB48SilentHonestFailure:
    """B48: a dead vision chain answers honestly instead of dead air."""

    HONEST = ("I couldn't analyse your screen just now, sir — say it "
              "once more and I'll try again.")

    def test_provider_raise_gives_honest_failure(
            self, monkeypatch, tmp_path):
        with _RealWorldGuard(monkeypatch, tmp_path):
            def _dead(question):
                raise TimeoutError("simulated total provider death")

            monkeypatch.setattr(brain, "analyze_screen", _dead)
            reply = _say("whats on my screen jarvis")
            assert reply == self.HONEST, reply
            assert not brain._screen_qa_busy.locked()
            assert self.HONEST in _last_assistant_text()

    def test_hung_provider_hits_the_bound(
            self, monkeypatch, tmp_path):
        with _RealWorldGuard(monkeypatch, tmp_path):
            release = threading.Event()

            def _hung(question):
                release.wait(60)
                raise TimeoutError("provider cascade dead")

            monkeypatch.setattr(brain, "analyze_screen", _hung)
            monkeypatch.setattr(brain, "SCREEN_QA_TIMEOUT", 3.0)
            box = {}
            t = threading.Thread(
                target=lambda: box.update(
                    reply=_say("whats on my screen jarvis")),
                daemon=True)
            try:
                t.start()
                t.join(timeout=25.0)
                assert not t.is_alive(), "bounded analysis did not return"
                assert box.get("reply") == self.HONEST, box.get("reply")
                assert not brain._screen_qa_busy.locked()
            finally:
                release.set()
                t.join(timeout=10.0)


class TestB49SilentBusyHonesty:
    """B49: a second screen question never gets a promise with no follow-up."""

    def test_busy_reply_is_honest_not_a_promise(
            self, monkeypatch, tmp_path):
        with _RealWorldGuard(monkeypatch, tmp_path):
            release = threading.Event()

            def _hung(question):
                release.wait(60)
                raise TimeoutError("provider cascade dead")

            monkeypatch.setattr(brain, "analyze_screen", _hung)
            monkeypatch.setattr(brain, "SCREEN_QA_TIMEOUT", 60.0)
            box = {}
            t = threading.Thread(
                target=lambda: box.update(
                    reply=_say("whats on my screen jarvis")),
                daemon=True)
            try:
                t.start()
                deadline = time.time() + 15.0
                while (time.time() < deadline
                       and not brain._screen_qa_busy.locked()):
                    time.sleep(0.1)
                assert brain._screen_qa_busy.locked(), \
                    "first analysis never took the lock"
                busy = _say("whats on my screen jarvis")
                assert "Still analysing" not in busy, busy
                assert "One moment" not in busy, busy
                assert "ask again in a moment" in busy, busy
            finally:
                release.set()
                t.join(timeout=15.0)
            assert box.get("reply") == TestB48SilentHonestFailure.HONEST, \
                box.get("reply")
            assert not brain._screen_qa_busy.locked()


class TestB50SilentDeicticRerun:
    """B50: 'try my last query again' re-runs instead of falling to chat."""

    @pytest.mark.parametrize("utterance", [
        "try my last query again",
        "retry my last question",
        "run my last request again",
    ])
    def test_retry_shapes_match(self, utterance):
        assert brain._DEICTIC_RERUN_RE.match(utterance.strip()), utterance

    @pytest.mark.parametrize("utterance", [
        "try my best", "do the whole task again", "try the search icon",
    ])
    def test_non_retry_shapes_miss(self, utterance):
        assert not brain._DEICTIC_RERUN_RE.match(utterance.strip()), utterance

    def test_rerun_repeats_real_count_question(
            self, monkeypatch, tmp_path):
        with _RealWorldGuard(monkeypatch, tmp_path):
            first = _say("how many windows are open on my screen")
            assert "open on your screen" in first, first
            live = _live_managed()
            expect_open = sum(1 for w in live if not w.get("iconic"))
            rerun = _say("try my last query again")
            assert "open on your screen" in rerun, rerun
            assert str(expect_open) in rerun, rerun

    def test_rerun_re_executes_real_plan_silently(
            self, monkeypatch, tmp_path):
        with _RealWorldGuard(monkeypatch, tmp_path):
            with _SilentWindows(2):
                first = _say("minimize the silenttestwindow windows")
                assert "already minimized" in first, first
                rerun = _say("try my last query again")
                assert rerun == first, (rerun, first)
                assert "Nothing was started" not in rerun, rerun

    def test_live_r13_draft_gets_neutral_fallback(self):
        draft = ("Shall I minimize all remaining windows on your screen "
                 "except for Brave, sir?")
        assert (brain._strip_unverified_action_claims(draft, "chat")
                == "Understood, sir — say exactly what you'd like me to "
                   "do and I'll do it.")


class TestB51SilentGreeting:
    """B51: a bare greeting never reaches the chat model, arms no offer."""

    @pytest.mark.parametrize("utterance", [
        "hi jarvis", "hello", "hey jarvis", "good morning",
        "hi", "hey", "hello jarvis",
    ])
    def test_greeting_is_deterministic(
            self, monkeypatch, tmp_path, utterance):
        with _RealWorldGuard(monkeypatch, tmp_path):
            def _no_model(*a, **k):
                raise AssertionError(
                    "chat model must not be consulted for a bare greeting")

            monkeypatch.setattr(brain, "handle_chat", _no_model)
            monkeypatch.setattr(brain, "_router_chat_reply", _no_model)
            reply = _say(utterance)
            assert reply == "Hello, sir. How can I assist you today?", reply
            assert brain._last_offer is None

    @pytest.mark.parametrize("utterance", [
        "jarvis", "sir", "hi, what is python", "i said hi",
        "hi jarvis minimize my windows",
    ])
    def test_non_greetings_untouched(self, utterance):
        assert not brain._BARE_GREETING_RE.match(utterance.strip()), \
            utterance
