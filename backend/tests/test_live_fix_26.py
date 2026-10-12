"""LIVE FIX 26 — the eleven re-test bugs (B33-B43).

Second window/chain re-test: ordinal exclusions, partial terminal matches,
pronoun/bare minimize, a lost screen-controls toggle, deictic re-run amnesia,
a stray-glyph chain collapse, a bogus research sibling, an STT-typo sinkhole,
a literal "non-Brave" filter, and a noisy speak/stop warning. One class per
bug; fakes for enumeration; no live model calls.
"""

import json
import sys

import pytest

from backend.services import screen_control as sc
from backend.services import screen_state


def _win(hwnd, title, iconic=False, top=0, left=0, process="", wclass=""):
    return {"hwnd": hwnd, "title": title, "iconic": iconic,
            "rect": {"top": top, "left": left, "width": 100, "height": 100},
            "process": process, "window_class": wclass}


class _ScreenStateGuard:
    """Save/restore the global toggle + ledgers around a test."""

    def __init__(self, monkeypatch):
        self._mp = monkeypatch
        self._was_enabled = screen_state.is_enabled()

    def __enter__(self):
        screen_state.clear_refused_chain()
        screen_state.clear_interactions()
        return self

    def __exit__(self, *exc):
        screen_state.clear_refused_chain()
        screen_state.clear_interactions()
        try:
            if screen_state.is_enabled() != self._was_enabled:
                screen_state.set_enabled(self._was_enabled, reason="test-restore")
        except Exception:
            pass
        return False


class TestB33OrdinalExclusion:
    WINS = [
        _win(1, "Brave", top=0, left=0, process="brave"),
        _win(2, "Notepad", top=0, left=500, process="notepad"),
        _win(3, "Cmd", top=500, left=0, process="cmd"),
    ]

    def _setup(self, monkeypatch):
        monkeypatch.setattr(
            sc, "_list_managable_windows",
            lambda: [dict(w) for w in self.WINS])

    def test_except_first_window(self, monkeypatch):
        self._setup(monkeypatch)
        plan = sc._parse_qualified_window_command(
            "minimize all the windows on my screen except for the first window")
        assert plan["ok"]
        assert {s["hwnd"] for s in plan["steps"]} == {2, 3}
        assert "Leaving the first window." in plan["summary"]

    def test_except_last_one(self, monkeypatch):
        self._setup(monkeypatch)
        plan = sc._parse_qualified_window_command(
            "minimize all the windows except the last one")
        assert plan["ok"]
        assert {s["hwnd"] for s in plan["steps"]} == {1, 2}
        assert "Leaving the last window." in plan["summary"]

    def test_except_second(self, monkeypatch):
        self._setup(monkeypatch)
        plan = sc._parse_qualified_window_command(
            "minimize all the windows except the second window")
        assert plan["ok"]
        assert {s["hwnd"] for s in plan["steps"]} == {1, 3}

    def test_name_exclusion_still_wins(self, monkeypatch):
        self._setup(monkeypatch)
        plan = sc._parse_qualified_window_command(
            "minimize all the windows except brave")
        assert plan["ok"]
        assert {s["hwnd"] for s in plan["steps"]} == {2, 3}
        assert "Leaving Brave." in plan["summary"]

    def test_out_of_range_excludes_nothing(self, monkeypatch):
        monkeypatch.setattr(
            sc, "_list_managable_windows",
            lambda: [dict(self.WINS[0])])
        plan = sc._parse_qualified_window_command(
            "minimize all the windows except the second window")
        assert plan["ok"]
        assert {s["hwnd"] for s in plan["steps"]} == {1}


class TestB34TerminalMatching:
    def test_preview_and_alt_terminals_match(self):
        # Title fallback (empty process) for a real terminal title.
        assert sc._window_matches_filter(
            {"title": "Windows Terminal (Preview)", "process": "",
             "window_class": ""}, "terminal")
        assert sc._window_matches_filter(
            {"title": "x", "process": "windowsterminalpreview",
             "window_class": ""}, "terminal")
        assert sc._window_matches_filter(
            {"title": "x", "process": "warp", "window_class": ""}, "terminal")
        assert sc._window_matches_filter(
            {"title": "x", "process": "alacritty", "window_class": ""},
            "terminal")
        assert sc._window_matches_filter(
            {"title": "x", "process": "mintty", "window_class": ""}, "terminal")

    def test_title_fallback_when_process_unknown(self):
        assert sc._window_matches_filter(
            {"title": "Windows PowerShell"}, "terminal")
        assert sc._window_matches_filter(
            {"title": "Command Prompt", "process": ""}, "terminal")
        assert not sc._window_matches_filter(
            {"title": "Untitled - Notepad", "process": ""}, "terminal")
        # A known browser process whose title names no terminal never matches.
        assert not sc._window_matches_filter(
            {"title": "YouTube - Brave", "process": "brave"}, "terminal")

    def test_scoped_already_minimized(self, monkeypatch):
        wins = [_win(11, "Windows PowerShell", iconic=True, process=""),
                _win(12, "Cmd", iconic=True, process="cmd"),
                _win(13, "Brave", iconic=False, process="brave")]
        monkeypatch.setattr(sc, "_list_managable_windows", lambda: wins)
        plan = sc._parse_qualified_window_command(
            "minimize all the terminal windows")
        assert plan["ok"] and not plan["steps"]
        assert plan["summary"] == (
            "The 2 terminal windows I can see are already minimized, sir.")

    def test_non_terminal_never_matches(self):
        assert not sc._window_matches_filter(
            {"title": "Brave", "process": "brave", "window_class": "x"},
            "terminal")


class TestB35PronounAndBareMinimize:
    WINS = [
        _win(1, "Brave", top=0, left=0, process="brave"),
        _win(2, "Windows PowerShell", top=0, left=500, process=""),
        _win(3, "Cmd", top=500, left=0, process="cmd"),
    ]

    def _setup(self, monkeypatch):
        monkeypatch.setattr(
            sc, "_list_managable_windows",
            lambda: [dict(w) for w in self.WINS])

    def test_minimize_all_of_them_uses_recent_app(self, monkeypatch):
        self._setup(monkeypatch)
        with _ScreenStateGuard(monkeypatch):
            screen_state.note_request("minimize all the terminal windows")
            plan = sc._parse_qualified_window_command(
                "no no, look at my screen. there are still terminal "
                "windows open. minimize all of them.")
            assert plan is not None and plan["ok"]
            assert {s["hwnd"] for s in plan["steps"]} == {2, 3}

    def test_minimize_them_without_app_minimizes_all(self, monkeypatch):
        self._setup(monkeypatch)
        with _ScreenStateGuard(monkeypatch):
            plan = sc._parse_qualified_window_command("minimize them")
            assert plan is not None and plan["ok"]
            assert {s["hwnd"] for s in plan["steps"]} == {1, 2, 3}

    def test_bare_plural_is_minimize_all(self, monkeypatch):
        self._setup(monkeypatch)
        plan = sc._parse_qualified_window_command(
            "minimize the windows visible on my screen")
        assert plan is not None and plan["ok"]
        assert {s["hwnd"] for s in plan["steps"]} == {1, 2, 3}
        assert plan["summary"].startswith("Minimizing 3 windows, sir.")

    def test_singular_bare_stays_active_window(self, monkeypatch):
        self._setup(monkeypatch)
        assert sc._parse_qualified_window_command(
            "minimize the window on my screen") is None

    def test_compound_tail_declines(self, monkeypatch):
        self._setup(monkeypatch)
        assert sc._parse_qualified_window_command(
            "minimize the windows and then open chrome") is None


class TestB36ToggleDurability:
    def test_refused_chain_roundtrip(self, monkeypatch):
        with _ScreenStateGuard(monkeypatch):
            screen_state.note_refused_chain("click x then type y")
            assert (screen_state.get_refused_chain()["source"]
                    == "click x then type y")
            screen_state.clear_refused_chain()
            assert screen_state.get_refused_chain()["source"] == ""

    def test_disable_clears_enable_preserves(self, monkeypatch):
        with _ScreenStateGuard(monkeypatch):
            screen_state.set_enabled(True, reason="test")
            screen_state.note_refused_chain("do stuff")
            screen_state.set_enabled(True, reason="test")
            assert screen_state.get_refused_chain()["source"] == "do stuff"
            screen_state.set_enabled(False, reason="test")
            assert screen_state.get_refused_chain()["source"] == ""

    def test_set_enabled_persists(self, tmp_path, monkeypatch):
        p = tmp_path / "toggle.json"
        monkeypatch.setenv("JARVIS_SCREEN_STATE_PATH", str(p))
        with _ScreenStateGuard(monkeypatch):
            screen_state.set_enabled(True, reason="test")
            assert json.loads(p.read_text(encoding="utf-8"))["enabled"] is True
            screen_state.set_enabled(False, reason="test")
            assert json.loads(p.read_text(encoding="utf-8"))["enabled"] is False

    def test_rehydrate_from_file(self, tmp_path, monkeypatch):
        p = tmp_path / "toggle.json"
        p.write_text('{"enabled": true, "ts": 123}', encoding="utf-8")
        monkeypatch.setenv("JARVIS_SCREEN_STATE_PATH", str(p))
        monkeypatch.setattr(screen_state, "_hydrated", False)
        monkeypatch.setattr(screen_state, "_enabled", False)
        assert screen_state.is_enabled() is True


class TestB37DeicticRerun:
    HITS = ["now do that task", "do that task", "do the task",
            "run that again", "do that now", "go ahead and do that",
            "please do that task, sir"]
    MISSES = ["do the whole task again", "again", "continue",
              "do that thing tomorrow", "perform the action",
              "do the task I told you about yesterday"]

    def test_regex_hits(self):
        from backend.core import brain
        for s in self.HITS:
            assert brain._DEICTIC_RERUN_RE.match(s.strip()), s

    def test_regex_misses(self):
        from backend.core import brain
        for s in self.MISSES:
            assert not brain._DEICTIC_RERUN_RE.match(s.strip()), s

    def test_source_priority(self, monkeypatch):
        from backend.core import brain
        monkeypatch.setattr(brain, "get_last_action", lambda: None)
        with _ScreenStateGuard(monkeypatch):
            assert brain._deictic_rerun_source() == ""
            screen_state.note_request("minimize all terminals")
            assert (brain._deictic_rerun_source()
                    == "minimize all terminals")
            screen_state.note_refused_chain("click x then type y")
            assert (brain._deictic_rerun_source()
                    == "click x then type y")

    def test_deictic_source_never_loops(self, monkeypatch):
        from backend.core import brain
        monkeypatch.setattr(brain, "get_last_action", lambda: None)
        with _ScreenStateGuard(monkeypatch):
            screen_state.note_refused_chain("now do that task")
            screen_state.note_request("do that task")
            assert brain._deictic_rerun_source() == ""


class TestB38JuxtapositionStability:
    CMD = ("click in the youtube search bar type mr.beast and press enter "
           "then play the first video that appears for the search")

    def test_glyph_prefixed_identical_to_plain(self):
        from backend.services import multi_intent as mi
        plain = mi.build_chain(self.CMD)
        glyphed = mi.build_chain("✕ " + self.CMD)
        assert plain is not None and glyphed is not None
        assert ([(s["kind"], s["text"]) for s in plain["steps"]]
                == [(s["kind"], s["text"]) for s in glyphed["steps"]])
        texts = [s["text"] for s in glyphed["steps"]]
        assert any("type mr.beast" in t for t in texts)

    def test_bullet_and_arrow_prefixes(self):
        from backend.services import multi_intent as mi
        for prefix in ["• ", "· ", "» ", "> "]:
            plan = mi.build_chain(prefix + self.CMD)
            assert plan is not None
            assert any("type mr.beast" in s["text"]
                       for s in plan["steps"]), prefix


class TestB40StrayGlyph:
    def test_ack_never_contains_glyph(self):
        from backend.services import multi_intent as mi
        plan = mi.build_chain(
            "✕ click in the youtube search bar type mr.beast and press "
            "enter then play the first video that appears for the search")
        assert plan is not None
        ack = mi.render_ack(plan)
        assert "✕" not in ack
        assert "type mr.beast" in ack

    def test_label_drops_leading_glyph(self):
        from backend.services import multi_intent as mi
        assert (mi._short_control_label("✕ click in the youtube search bar")
                == "click in the youtube search bar")

    def test_interior_content_untouched(self):
        from backend.services import multi_intent as mi
        assert mi._strip_stray_glyphs("type mr.beast") == "type mr.beast"


class TestB39OrdinalPlayNoSibling:
    def test_ordinal_play_with_result_clause(self):
        from backend.services import multi_intent as mi
        plan = mi.build_chain(
            "click in the youtube search bar type mr.beast and press enter "
            "then play the first video that appears for the search")
        assert plan is not None
        assert [s["kind"] for s in plan["steps"]] == [
            "screen_control", "screen_control", "screen_control",
            "screen_control"]
        assert plan["steps"][-1]["text"].startswith("play the first video")

    def test_real_lookup_still_grows_sibling(self):
        from backend.services import multi_intent as mi
        plan = mi.build_chain(
            "click the search bar to search for flights and book a table")
        assert plan is not None
        assert "research" in [s["kind"] for s in plan["steps"]]

    def test_plain_ordinal_play_alone(self):
        from backend.services import multi_intent as mi
        assert mi._ORDINAL_PLAY_RE.search(
            "play the first video that appears for the search")


class TestB41TypoTolerance:
    def test_doubled_letter_rescued(self):
        assert (sc._repair_verb_typos("miinimize all the windows")
                == "minimize all the windows")
        assert (sc._repair_verb_typos("minimiize the window")
                == "minimize the window")

    def test_legitimate_doubles_untouched(self):
        assert (sc._repair_verb_typos("click the button")
                == "click the button")
        assert (sc._repair_verb_typos("scroll down") == "scroll down")
        assert (sc._repair_verb_typos("miniature models")
                == "miniature models")

    def test_typo_command_parses_with_exclusion(self, monkeypatch):
        monkeypatch.setattr(
            sc, "_list_managable_windows",
            lambda: [_win(1, "Brave", process="brave"),
                     _win(2, "Notepad", process="notepad")])
        plan = sc._parse_qualified_window_command(
            "miinimize all the windows on my screen except for the brave "
            "window")
        assert plan is not None and plan["ok"]
        assert {s["hwnd"] for s in plan["steps"]} == {2}
        assert "Leaving Brave." in plan["summary"]


class TestB42NonExclusion:
    def _setup(self, monkeypatch):
        monkeypatch.setattr(
            sc, "_list_managable_windows",
            lambda: [_win(1, "Brave", process="brave"),
                     _win(2, "Notepad", process="notepad")])

    def test_non_hyphenated(self, monkeypatch):
        self._setup(monkeypatch)
        plan = sc._parse_qualified_window_command(
            "click the minimize button on all non-Brave windows")
        assert plan is not None and plan["ok"]
        assert {s["hwnd"] for s in plan["steps"]} == {2}
        assert "Leaving Brave." in plan["summary"]

    def test_non_spaced(self, monkeypatch):
        self._setup(monkeypatch)
        plan = sc._parse_qualified_window_command(
            "minimize all non brave windows")
        assert plan is not None and plan["ok"]
        assert {s["hwnd"] for s in plan["steps"]} == {2}

    def test_positive_filter_unaffected(self, monkeypatch):
        monkeypatch.setattr(
            sc, "_list_managable_windows",
            lambda: [_win(1, "Brave", process="brave"),
                     _win(2, "Cmd", process="cmd")])
        plan = sc._parse_qualified_window_command(
            "minimize all the terminal windows")
        assert plan is not None and plan["ok"]
        assert {s["hwnd"] for s in plan["steps"]} == {2}


class TestB43SpeakStopRetryQuiet:
    def _worker(self):
        from backend.services import listener as _lis
        return _lis._SpeakStopWorker()

    class _Resp:
        status = 200

        def read(self):
            return b"{}"

    def test_recovered_transient_warns_nothing(self, capsys):
        w = self._worker()
        calls = {"n": 0}
        resp = self._Resp()

        class _FakeConn:
            def request(self, *a, **k):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError(
                        "[WinError 10053] An established connection was "
                        "aborted by the software in your host machine")

            def getresponse(self):
                return resp

            def close(self):
                pass

        w._open_connection = lambda: _FakeConn()
        w._deliver()
        out = capsys.readouterr().out
        assert "speak/stop failed" not in out
        assert w.stats["delivered"] == 1
        assert w.stats["failed"] == 0

    def test_double_failure_warns_once(self, capsys):
        w = self._worker()

        class _BadConn:
            def request(self, *a, **k):
                raise OSError("[WinError 10053] aborted")

            def close(self):
                pass

        w._open_connection = lambda: _BadConn()
        w._deliver()
        w._deliver()
        out = capsys.readouterr().out
        assert out.count("speak/stop failed") == 1
        assert w.stats["failed"] == 2

    @pytest.mark.skipif(sys.platform != "win32", reason="Win32 only")
    def test_win32_sanity(self):
        assert sys.platform == "win32"
