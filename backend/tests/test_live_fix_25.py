"""LIVE FIX 25 — B25-B32 (one class per bug; fakes only, no screen/network)."""
import sys
import time

import pytest

from backend.core import brain
from backend.services import multi_intent
from backend.services import screen_control as sc
from backend.services import screen_state


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(brain, "_last_offer", None)
    monkeypatch.setattr(brain, "_mi_chain_active", False)
    monkeypatch.setitem(brain._screen_qa_last, "seq", 0)
    monkeypatch.setitem(brain._mi_replicate_last, "seq", 0)
    screen_state.clear_interactions()
    screen_state.clear_pending_plan()
    sc._clear_window_enumeration()
    yield
    screen_state.clear_interactions()
    screen_state.clear_pending_plan()


class TestB25CompoundFeedbackFilter:
    """B25: compound feedback must not corrupt the window app filter."""

    WINS = [
        {"hwnd": 1, "title": "dev terminal", "iconic": False,
         "rect": {"left": 0, "top": 0, "width": 9, "height": 9}},
        {"hwnd": 2, "title": "ops terminal", "iconic": False,
         "rect": {"left": 10, "top": 0, "width": 9, "height": 9}},
        {"hwnd": 3, "title": "(22) YouTube - Brave", "iconic": False,
         "rect": {"left": 20, "top": 0, "width": 9, "height": 9}},
    ]

    def _fake(self, monkeypatch, wins=None):
        monkeypatch.setattr(
            sc, "_list_managable_windows",
            lambda: [dict(w) for w in (wins if wins is not None
                                       else self.WINS)])

    def test_t2_compound_feedback(self, monkeypatch):
        self._fake(monkeypatch)
        plan = sc._parse_qualified_window_command(
            "some terminal windows are still open you only minimized 1 , "
            "minimize all terminal windows on my screen")
        assert plan["ok"] and sorted(s["hwnd"] for s in plan["steps"]) == [1, 2]

    def test_clean_command_unchanged(self, monkeypatch):
        self._fake(monkeypatch)
        plan = sc._parse_qualified_window_command(
            "minimize all the terminal windows on my screen")
        assert [s["hwnd"] for s in plan["steps"]] == [1, 2]

    def test_of_the_prefix(self, monkeypatch):
        self._fake(monkeypatch)
        assert len(sc._parse_qualified_window_command(
            "minimize all of the windows")["steps"]) == 3

    def test_exclusion_still_works(self, monkeypatch):
        self._fake(monkeypatch)
        plan = sc._parse_qualified_window_command(
            "minimize all the windows on my screen except for brave window")
        assert {s["hwnd"] for s in plan["steps"]} == {1, 2}

    def test_appnamed_does_not_cross_second_verb(self, monkeypatch):
        self._fake(monkeypatch)
        plan = sc._parse_qualified_window_command(
            "i said minimize the window, minimize terminal windows")
        assert plan["ok"] and {s["hwnd"] for s in plan["steps"]} == {1, 2}

    def test_bare_object_still_defers(self, monkeypatch):
        self._fake(monkeypatch)
        assert sc._parse_qualified_window_command(
            "minimize the window on my screen") is None

    def test_honest_failure_names_clean_filter(self, monkeypatch):
        self._fake(monkeypatch, [self.WINS[2]])
        plan = sc._parse_qualified_window_command(
            "you only minimized 1 , minimize all terminal windows")
        assert plan["reason"] == "I couldn't find any terminal windows, sir."


class TestB26ProcessAliasMatching:
    """B26: "terminal" matches terminal windows by process/class."""

    TERMS = [
        {"hwnd": 11, "title": "Windows PowerShell", "iconic": False,
         "rect": None, "process": "windowsterminal",
         "window_class": "CASCADIA_HOSTING_WINDOW_CLASS"},
        {"hwnd": 12, "title": "Command Prompt", "iconic": False,
         "rect": None, "process": "cmd",
         "window_class": "ConsoleWindowClass"},
        {"hwnd": 13, "title": "(22) YouTube - Brave", "iconic": False,
         "rect": None, "process": "brave",
         "window_class": "Chrome_WidgetWin_1"},
    ]

    def test_t4_terminal_matches_by_process(self, monkeypatch):
        monkeypatch.setattr(sc, "_list_managable_windows",
                            lambda: [dict(w) for w in self.TERMS])
        plan = sc._parse_qualified_window_command(
            "minimize all the terminal windows on my screen")
        assert plan["ok"] and {s["hwnd"] for s in plan["steps"]} == {11, 12}

    def test_powershell_alias(self):
        assert sc._window_matches_filter(
            dict(self.TERMS[0], process="pwsh"), "powershell")

    def test_command_prompt_alias(self):
        assert sc._window_matches_filter(self.TERMS[1], "command prompt")

    def test_no_terminals_stays_honest(self, monkeypatch):
        monkeypatch.setattr(sc, "_list_managable_windows",
                            lambda: [dict(self.TERMS[2])])
        plan = sc._parse_qualified_window_command(
            "minimize all the terminal windows")
        assert plan["reason"] == "I couldn't find any terminal windows, sir."

    def test_except_terminal_keeps_them(self, monkeypatch):
        monkeypatch.setattr(sc, "_list_managable_windows",
                            lambda: [dict(w) for w in self.TERMS])
        plan = sc._parse_qualified_window_command(
            "minimize all the windows except the terminal")
        assert {s["hwnd"] for s in plan["steps"]} == {13}

    def test_all_iconic_terminals_honest(self, monkeypatch):
        wins = ([dict(w, iconic=True) for w in self.TERMS[:2]]
                + [dict(self.TERMS[2])])
        monkeypatch.setattr(sc, "_list_managable_windows", lambda: wins)
        plan = sc._parse_qualified_window_command(
            "minimize all the terminal windows")
        assert plan["ok"] and not plan["steps"]
        assert (plan["summary"]
                == "All 2 terminal windows are already minimized, sir.")

    def test_title_only_fakes_unchanged(self):
        assert not sc._window_matches_filter(
            {"title": "Windows PowerShell"}, "terminal")

    @pytest.mark.skipif(sys.platform != "win32", reason="Win32 only")
    def test_enumeration_carries_process_keys(self):
        for w in sc._list_managable_windows():
            assert "process" in w and "window_class" in w

    def test_process_name_never_raises(self):
        assert sc._process_image_name(0) == ""
        assert sc._process_image_name("x") == ""


class TestB27LookAgainAndFallback:
    """B27: bare "look again" re-plans the last window command."""

    def _last(self, monkeypatch, seq=5):
        monkeypatch.setattr(brain, "get_last_action", lambda: {
            "kind": "screen_control",
            "command": "minimize all the terminal windows",
            "at": time.time(), "seq": seq})
        monkeypatch.setitem(brain._screen_qa_last, "seq", 1)
        monkeypatch.setitem(brain._mi_replicate_last, "seq", 0)

    def test_look_again_retries_window_command(self, monkeypatch):
        self._last(monkeypatch)
        assert (brain._screen_command_relook("look again")
                == "minimize all the terminal windows")

    def test_newer_screen_question_keeps_its_retry(self, monkeypatch):
        self._last(monkeypatch, seq=1)
        monkeypatch.setitem(brain._screen_qa_last, "seq", 9)
        assert brain._screen_command_relook("look again") == ""

    def test_pointing_line_not_bare(self, monkeypatch):
        self._last(monkeypatch)
        assert (brain._screen_command_relook(
            "look again there is a video on top right") == "")

    def test_t3_claim_gets_screen_fallback(self):
        out = brain._strip_unverified_action_claims(
            "I will locate and minimize the remaining terminal windows "
            "right away, sir.", "chat")
        assert "folder" not in out and out == brain._CHAT_SCREEN_FALLBACK

    def test_file_claim_keeps_file_fallback(self):
        assert (brain._strip_unverified_action_claims(
            "I will get that created right away, sir.", "chat")
            == brain._CHAT_NO_ACTION_FALLBACK)

    def test_try_again_retries_failed_direct_request(self, monkeypatch):
        screen_state.set_enabled(True)
        try:
            screen_state.clear_interactions()
            screen_state.add_interaction(
                "minimize all except brave", "ok",
                plan={"steps": [{"action": "minimize_hwnd", "hwnd": 99}]})
            time.sleep(0.01)
            screen_state.note_request("minimize all the terminal windows")
            seen = []
            monkeypatch.setattr(
                sc, "_build_direct_plan",
                lambda c: seen.append(c) or {"ok": False, "reason": "r",
                                            "steps": []})
            monkeypatch.setattr(
                sc, "_plan_with_tree",
                lambda c: pytest.fail("vision used"))
            sc.maybe_handle_screen_control_message("try again")
            assert seen == ["minimize all the terminal windows"]
        finally:
            screen_state.set_enabled(False)

    def test_clear_interactions_clears_request(self):
        screen_state.note_request("x")
        screen_state.clear_interactions()
        assert screen_state.get_last_request()["command"] == ""


class TestB28JuxtaposedTyping:
    """B28: "click X type Y" with no connector is two steps."""

    T5 = ("click in the youtube search bar on my screen type mr.beast "
          "and click search and play the first video that comes up")

    def test_t5_four_steps(self):
        plan = multi_intent.build_chain(self.T5)
        assert [s["kind"] for s in plan["steps"]] == ["screen_control"] * 4
        assert plan["steps"][1]["text"] == "type mr.beast"

    def test_t5_with_then_same(self):
        plan = multi_intent.build_chain(
            self.T5.replace(" type mr.beast", " then type mr.beast"))
        assert [s["text"] for s in plan["steps"]][:2] == [
            "click in the youtube search bar on my screen", "type mr.beast"]

    def test_backspace_case_unchanged(self):
        plan = multi_intent.build_chain(
            "backspace whatever is written in it, type mr.beast and then "
            "search")
        assert [s["text"] for s in plan["steps"]] == [
            "backspace whatever is written in it", "type mr.beast",
            "press enter"]

    @pytest.mark.parametrize("clause", [
        "click the font type dropdown", "click the type button",
        "click the notepad window type the report in the file",
        "type mr.beast in the search bar"])
    def test_no_false_split(self, clause):
        assert (multi_intent._split_control_juxtaposition([clause])
                == [clause])

    def test_ack_shows_four_steps(self):
        assert ("4 steps"
                in multi_intent.render_ack(multi_intent.build_chain(self.T5)))


class TestB29ConfirmDuringPlanningAndLiveGate:
    """B29: bare confirm with no armed plan + R13 on live speech."""

    def test_live_gate_never_speaks_single_sentence_claim(self):
        out = "".join(brain._gate_live_deltas(
            ["Clicking 'Tab", " search', sir."]))
        assert "Clicking" not in out and "Nothing was started" in out

    def test_live_gate_passthrough_is_byte_identical(self):
        assert ("".join(brain._gate_live_deltas(
            ["Hello ", "sir. ", "All good."]))
            == "Hello sir. All good.")

    def test_live_gate_fallback_spoken_once(self):
        out = "".join(brain._gate_live_deltas(
            ["Clicking it now, sir. ", "Typing it now, sir."]))
        assert out.count("Nothing was started") == 1

    def test_live_gate_keeps_restart_dedup(self):
        assert ("".join(brain._gate_live_deltas(
            ["Loud", "Loud and clear, sir."]))
            == "Loud and clear, sir.")

    def test_answered_path_gated_before_stream(self, monkeypatch):
        monkeypatch.setattr(brain, "add_message", lambda a, *k: None)
        monkeypatch.setattr(brain, "_commit_chat", lambda a, *k: None)
        monkeypatch.setattr(brain, "memory_store", None)
        heard = []
        brain.handle_chat("confirm", stream=heard.append,
                          answered="Clicking 'Tab search', sir.")
        assert "Clicking" not in "".join(heard)

    def test_racer_adopt_gates_live_speech(self, monkeypatch):
        monkeypatch.setattr(brain, "get_history", lambda: [])
        monkeypatch.setattr(brain, "_build_chat_messages", lambda a, *k: {
            "path": "llm",
            "messages": [{"role": "user", "content": "confirm"}]})
        monkeypatch.setattr(brain, "_stream_chat_deltas",
                            lambda a, *k: iter(["Clicking 'Tab search', "
                                                "sir."]))
        racer = brain._ChatRacer("confirm", True, history=[])
        try:
            assert "Clicking" not in "".join(racer.adopt())
        finally:
            try:
                racer.cancel()
            except Exception:
                pass

    def test_confirm_while_planning_is_honest(self, monkeypatch):
        monkeypatch.setattr(screen_state, "has_pending_plan", lambda: False)
        monkeypatch.setattr(sc, "screen_planning_in_progress", lambda: True)
        assert ("still being prepared" in
                brain._bare_confirm_without_plan_reply("confirm"))

    def test_confirm_while_chain_running_is_honest(self, monkeypatch):
        monkeypatch.setattr(screen_state, "has_pending_plan", lambda: False)
        monkeypatch.setattr(sc, "screen_planning_in_progress",
                            lambda: False)
        monkeypatch.setattr(brain, "_mi_chain_active", True)
        assert ("still being prepared" in
                brain._bare_confirm_without_plan_reply("yes"))

    def test_plan_armed_late_goes_to_screen_gate(self, monkeypatch):
        monkeypatch.setattr(screen_state, "has_pending_plan", lambda: True)
        monkeypatch.setattr(sc, "maybe_handle_screen_control_message",
                            lambda t: "Clicking 'Search', sir.")
        assert (brain._bare_confirm_without_plan_reply("confirm")
                == "Clicking 'Search', sir.")

    def test_approval_word_with_nothing_pending(self, monkeypatch):
        monkeypatch.setattr(screen_state, "has_pending_plan", lambda: False)
        monkeypatch.setattr(sc, "screen_planning_in_progress",
                            lambda: False)
        monkeypatch.setattr(brain, "_mi_chain_active", False)
        assert ("no screen action waiting" in
                brain._bare_confirm_without_plan_reply(
                    "confirm screen action"))

    def test_plain_yes_nothing_pending_stays_chat(self, monkeypatch):
        monkeypatch.setattr(screen_state, "has_pending_plan", lambda: False)
        monkeypatch.setattr(sc, "screen_planning_in_progress",
                            lambda: False)
        monkeypatch.setattr(brain, "_mi_chain_active", False)
        assert brain._bare_confirm_without_plan_reply("yes") is None

    def test_negation_never_confirms(self, monkeypatch):
        monkeypatch.setattr(sc, "screen_planning_in_progress", lambda: True)
        assert (brain._bare_confirm_without_plan_reply("no, don't do it")
                is None)


class TestB30ClickTargetCorrection:
    """B30: second-person click-target corrections re-route."""

    def _recent(self, monkeypatch):
        monkeypatch.setattr(brain, "get_last_action", lambda: {
            "kind": "chain", "command": "x", "at": time.time(), "seq": 1})
        monkeypatch.setattr(brain, "_memory_recall_request", lambda m: False)

    def test_t7_rewrites(self, monkeypatch):
        self._recent(monkeypatch)
        assert (brain._screen_action_correction_command(
            "you are not clicking the search icon after typing , instead "
            "you are clicking something else")
            == "click the search icon after typing")

    def test_instead_of_shape(self, monkeypatch):
        self._recent(monkeypatch)
        assert (brain._screen_action_correction_command(
            "you clicked the tab search instead of the search icon")
            == "click the search icon")

    def test_no_recent_activity_stays_chat(self, monkeypatch):
        monkeypatch.setattr(brain, "get_last_action", lambda: {})
        monkeypatch.setattr(brain, "_take_chain_plan", lambda: None)
        assert (brain._screen_action_correction_command(
            "you are not clicking the search icon") is None)

    def test_vague_target_rejected(self, monkeypatch):
        self._recent(monkeypatch)
        assert (brain._screen_action_correction_command(
            "you are not clicking anything") is None)

    def test_zed_turn_untouched(self, monkeypatch):
        self._recent(monkeypatch)
        assert (brain._screen_action_correction_command(
            "no about zed editor, tell me what it is") is None)

    def test_recall_never_hijacked(self, monkeypatch):
        self._recent(monkeypatch)
        monkeypatch.setattr(brain, "_memory_recall_request", lambda m: True)
        assert (brain._screen_action_correction_command(
            "you are not clicking the search icon") is None)


class TestB31OfferConfirmLoop:
    """B31: chat cannot offer/approve screen actions; bare yes takes only
    real lookup offers."""

    @pytest.mark.parametrize("reply", [
        "Understood, sir. Shall I try clicking that search button again?",
        "Understood, sir. Shall I open YouTube and run the search for "
        "Mr. Beast now?",
        "Shall I go ahead and click the search icon attached to the bar, "
        "sir?",
    ])
    def test_screen_offers_not_registered(self, monkeypatch, reply):
        monkeypatch.setattr(brain, "_last_offer", None)
        monkeypatch.setattr(brain, "_resolve_reference_query",
                            lambda m: "q")
        brain._remember_chat_offer("x", reply)
        assert brain._last_offer is None

    def test_lookup_offer_still_registered(self, monkeypatch):
        monkeypatch.setattr(brain, "_last_offer", None)
        monkeypatch.setattr(brain, "_resolve_reference_query",
                            lambda m: "mr beast")
        brain._remember_chat_offer("who is mr beast",
                                   "Shall I look it up, sir?")
        assert brain._last_offer["kind"] == "search"

    def test_bare_yes_consumes_offer(self, monkeypatch):
        monkeypatch.setattr(brain, "_other_approval_pending", lambda: False)
        monkeypatch.setattr(brain, "_last_offer",
                            {"kind": "search", "text": "x",
                             "query": "mr beast", "at": time.time(),
                             "expires": time.time() + 60})
        seen = []
        monkeypatch.setattr(brain, "handle_research_intent",
                            lambda q, **k: seen.append(q) or "ok")
        assert (brain._consume_offered_action("yes") == "ok"
                and seen == ["mr beast"])

    def test_bare_yes_defers_to_pending_plan(self, monkeypatch):
        monkeypatch.setattr(brain, "_other_approval_pending", lambda: True)
        monkeypatch.setattr(brain, "_last_offer",
                            {"kind": "search", "query": "q",
                             "at": time.time(),
                             "expires": time.time() + 60})
        assert brain._consume_offered_action("confirm") is None
        assert brain._last_offer is not None

    def test_superseded_offer_ignores_bare_yes(self, monkeypatch):
        monkeypatch.setattr(brain, "_other_approval_pending", lambda: False)
        monkeypatch.setattr(brain, "_last_offer",
                            {"kind": "search", "query": "q",
                             "at": time.time(),
                             "expires": time.time() + 60,
                             "superseded": True})
        assert brain._consume_offered_action("yes") is None

    @pytest.mark.parametrize("reply", [
        "I need you to confirm the screen action to proceed, sir.",
        "I am ready, sir. Shall I click the YouTube search bar and type "
        "'mr.beast'?",
    ])
    def test_chat_approval_requests_stripped(self, reply):
        out = brain._strip_unverified_action_claims(reply, "chat")
        assert ("confirm the screen" not in out
                and "Shall I click" not in out)

    def test_capability_and_lookup_offers_survive(self):
        for s in ("I can create files for you, sir.",
                  "Shall I look it up, sir?"):
            assert brain._strip_unverified_action_claims(s, "chat") == s

    def test_all_fallbacks_pass_their_own_gate(self):
        for f in brain._CHAT_FALLBACKS:
            assert brain._strip_unverified_action_claims(f, "chat") == f
            assert not brain._is_lookup_offer(f)

    def test_persona_forbids_screen_offers(self, monkeypatch):
        monkeypatch.setattr(brain, "_memory_context_cached", lambda m: ("", ""))
        monkeypatch.setattr(brain, "force_search", lambda m: False)
        monkeypatch.setattr(brain, "should_search", lambda m: False)
        monkeypatch.setattr(brain, "should_search_live_event",
                            lambda m, h: False)
        built = brain._build_chat_messages("hi", history=[])
        assert "Never offer to do a screen action" in built["system_prompt"]


class TestB32WholeTaskRepeat:
    """B32: "do the whole task again" re-runs the stored chain."""

    @pytest.mark.parametrize("t", [
        "do the whole task again", "redo the whole task",
        "run the whole thing again", "please do the entire task once more",
        "start it all over again"])
    def test_shapes_match(self, t):
        assert brain._REPEAT_WHOLE_TASK_RE.match(t)

    @pytest.mark.parametrize("t", [
        "again", "do the rest", "do the whole task", "continue",
        "tell me about the whole task again"])
    def test_non_shapes(self, t):
        assert not brain._REPEAT_WHOLE_TASK_RE.match(t)

    def test_rebuilds_from_source_and_forces(self, monkeypatch):
        src = ("click in the youtube search bar on my screen type mr.beast "
               "and click search and play the first video that comes up")
        monkeypatch.setattr(brain, "_take_chain_plan", lambda: {
            "plan": {"source": src,
                     "steps": [{"kind": "screen_control", "text": "x"}] * 3},
            "results": [], "at": time.time()})
        seen = {}

        def fake(m, plan, **kw):
            seen.update(plan=plan, kw=kw)
            return "Sir, I'll do this in 4 steps."

        monkeypatch.setattr(brain, "handle_multi_intent", fake)
        monkeypatch.setattr(brain, "note_executed_action", lambda *a: None)
        assert (brain._rerun_whole_chain("do the whole task again")
                .startswith("Sir"))
        assert (len(seen["plan"]["steps"]) == 4
                and seen["kw"]["force_rerun"] is True)

    def test_no_chain_returns_none(self, monkeypatch):
        monkeypatch.setattr(brain, "_take_chain_plan", lambda: None)
        assert brain._rerun_whole_chain("do the whole task again") is None
