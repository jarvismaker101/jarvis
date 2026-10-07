"""Live-fix tests: "search this stream" must SEARCH, not ask.

Replays the exact live transcript requests that used to fall into a
five-turn chat ask-loop:

    "can you search about this stream on internet?"
    "find anything about that stream."
    "use any searching."
    "don't ask any questions, just search it."
    "execute it."

Covered:
  * the screen answer records its topic as an entity ("this stream" binds);
  * each request routes to research with the on-screen title as the query;
  * chains that say "search this youtube creator" consume the screen step;
  * "execute it" consumes the search Jarvis last offered;
  * a repeat chain answers from its last report instead of re-running.
"""
import unittest
from unittest.mock import patch

from backend.core import brain
from backend.core import entity_ledger
from backend.services import multi_intent
from backend.services.task_agent import agent as task_agent

TITLE = "JEFF DOMINIC - OI OI OI | GTA RP | Full Stream"
SCREEN_Q = "what is on my screen right now?"
ASK_1 = "can you search about this stream on internet?"
ASK_2 = "find anything about that stream."
ASK_3 = "use any searching."
ASK_4 = "don't ask any questions, just search it."
ASK_5 = "execute it."
OFFER_REPLY = "I can perform that web search immediately, sir."
CHAIN_1 = "look at my screen and search this youtube creator on internet."
CHAIN_2 = ("whatever youtube creator name do you see on my screen, i want "
           "you to search that youtube creator name on internet.")
CREATOR = "AI Explained"
CREATOR_ASK = ("look at my screen ther is a video about grok bot it also "
               "shows its content creator name and i want you to search "
               "about this creator on internet and tell me what you find")
CREATOR_CORRECTION = ("not the video i wanted you to search about the "
                      "creator of this video")


class LiveFixBase(unittest.TestCase):
    def setUp(self):
        brain._mi_chain_active = False
        brain._pending_confirmation = None
        brain._pending_opencode_task = None
        brain._pending_browser_clarification = None
        brain._held_redirect = None
        brain._proactive_research_fired = False
        brain._last_offer = None
        brain._last_chain_run = None
        brain._last_screen_topic = {"text": "", "at": 0.0}
        brain._last_screen_creator = {"text": "", "at": 0.0}
        brain._last_research_topic = None
        task_agent._pending_task_action = None
        entity_ledger.reset()
        self.addCleanup(entity_ledger.reset)
        self.addCleanup(setattr, brain, "_last_offer", None)
        self.addCleanup(setattr, brain, "_last_chain_run", None)
        self.addCleanup(setattr, brain, "_last_screen_topic",
                        {"text": "", "at": 0.0})
        self.addCleanup(setattr, brain, "_last_screen_creator",
                        {"text": "", "at": 0.0})
        self.addCleanup(setattr, brain, "_last_research_topic", None)

    def _seed_screen(self):
        entity_ledger.record_entity(TITLE, TITLE, kind="topic",
                                    source="screen")
        brain._set_last_screen_topic(TITLE)


class ScreenTopicRecordingTests(LiveFixBase):
    def test_screen_answer_records_topic_entity(self):
        with patch.object(brain, "classify_intent",
                          return_value={"intent": "screen",
                                        "_source": "test"}), \
             patch.object(brain, "orchestrator_select_route",
                          return_value="legacy"), \
             patch.object(brain, "analyze_screen", return_value={
                 "tip": "You are watching a GTA RP stream.",
                 "topic": TITLE, "evidence": [], "grounding_links": [],
                 "show_images": False}), \
             patch.object(brain, "begin_screen_capture",
                          return_value={"capture_id": "c1", "seq": 1}), \
             patch.object(brain, "push_screen_answer", return_value=None), \
             patch.object(brain, "_commit_chat"):
            brain.process_message(SCREEN_Q, sync_voice=False,
                                  commit_response=False)
        self.assertEqual(brain._get_last_screen_topic(), TITLE)
        verdict, ent = entity_ledger.resolve_mention(ASK_1)
        self.assertEqual(verdict, "bound")
        self.assertEqual(ent["display_name"], TITLE)


class SearchShapedNetTests(LiveFixBase):
    def test_exact_phrases_are_search_shaped(self):
        for text in (ASK_1, ASK_2, ASK_3, ASK_4):
            self.assertTrue(brain.is_search_shaped_message(text), text)
        self.assertFalse(brain.is_search_shaped_message(ASK_5))

    def test_guards_keep_non_search_phrases_routed(self):
        for text in (
            "open youtube and search coldplay",
            "how do i search in google",
            "did you search for the news",
            "search youtube in chrome",
            "i will search it later",
            "what is on my screen right now?",
        ):
            self.assertFalse(brain.is_search_shaped_message(text), text)

    def test_is_deictic_query(self):
        for q in ("that youtube creator name", "this stream",
                  "this stream on internet",
                  "search this youtube creator on internet"):
            self.assertTrue(brain._is_deictic_query(q), q)
        for q in ("JEFF DOMINIC GTA RP", "Interstellar movie"):
            self.assertFalse(brain._is_deictic_query(q), q)

    def test_deictic_search_resolves_to_screen_topic(self):
        self._seed_screen()
        for text in (ASK_1, ASK_2, ASK_3, ASK_4):
            self.assertEqual(brain._search_request_query(text), TITLE, text)

    def test_unresolvable_deictic_search_asks_once(self):
        with patch.object(brain, "classify_intent",
                          return_value={"intent": "chat", "reply": ""}):
            reply = brain.process_message(ASK_1, sync_voice=False)
        self.assertIn("name it once", reply)

    def test_exact_transcript_requests_all_search(self):
        self._seed_screen()
        calls = []

        def fake_research(query, **kwargs):
            calls.append(query)
            return "Quick lookup for that, sir."

        outputs = []
        with patch.object(brain, "handle_research_intent",
                          side_effect=fake_research), \
             patch.object(brain, "orchestrator_select_route",
                          return_value="legacy"), \
             patch.object(brain, "classify_intent") as classifier, \
             patch.object(brain, "handle_chat", return_value="chat"):
            for phrase in (ASK_1, ASK_2, ASK_3, ASK_4):
                outputs.append((phrase, brain.process_message(
                    phrase, sync_voice=False)))
        self.assertEqual(calls, [TITLE, TITLE, TITLE, TITLE])
        classifier.assert_not_called()
        print("\n--- live replay: ---")
        for phrase, reply in outputs:
            print("USER:", phrase)
            print("JARVIS:", reply)

    def test_execute_it_consumes_offered_search(self):
        self._seed_screen()
        calls = []

        def fake_research(query, **kwargs):
            calls.append(query)
            return "Quick lookup for that, sir."

        with patch.object(brain, "handle_research_intent",
                          side_effect=fake_research), \
             patch.object(brain, "classify_intent") as classifier:
            brain._remember_chat_offer(ASK_1, OFFER_REPLY)
            reply = brain.process_message(ASK_5, sync_voice=False)
        self.assertEqual(calls, [TITLE])
        self.assertIn("Quick lookup", reply)
        classifier.assert_not_called()

    def test_derive_research_query_never_returns_pointer(self):
        with patch.object(brain, "classify_intent",
                          return_value={"intent": "research",
                                        "query": "this stream"}):
            self.assertEqual(brain.derive_research_query(ASK_1), "")
        self._seed_screen()
        with patch.object(brain, "classify_intent",
                          return_value={"intent": "research",
                                        "query": "this stream"}):
            self.assertEqual(brain.derive_research_query(ASK_1), TITLE)


class ChainLiveFixTests(LiveFixBase):
    def test_chain_anaphora_consumes_screen_for_live_phrasings(self):
        for text in (CHAIN_1, CHAIN_2):
            plan = multi_intent.build_chain(text)
            self.assertIsNotNone(plan, text)
            kinds = [s["kind"] for s in plan["steps"]]
            self.assertEqual(kinds, ["screen", "research"], text)
            self.assertEqual(plan["steps"][1]["consumes"], [0], text)

    def test_chain_research_uses_screen_topic_not_deictic_text(self):
        self._seed_screen()
        captured = {}

        def fake_quick(query):
            captured["query"] = query
            return {"query": query, "spoken_summary": "About the streamer."}

        plan = multi_intent.build_chain(CHAIN_1)
        self.assertIsNotNone(plan)
        with patch.object(brain, "analyze_screen", return_value={
                "tip": "A GTA RP stream.", "topic": TITLE}), \
             patch.object(brain, "run_quick_search", side_effect=fake_quick):
            brain._run_multi_intent_chain(CHAIN_1, plan)
        self.assertEqual(captured.get("query"), TITLE)

    def test_chain_research_without_topic_uses_tip_fallback(self):
        captured = {}
        plan = multi_intent.build_chain(CHAIN_1)
        self.assertIsNotNone(plan)
        with patch.object(brain, "analyze_screen", return_value={
                "tip": "A GTA RP stream."}), \
             patch.object(brain, "run_quick_search",
                          side_effect=lambda q: captured.setdefault("q", q)):
            brain._run_multi_intent_chain(CHAIN_1, plan)
        self.assertEqual(captured.get("q"), "A GTA RP stream.")

    def test_finish_report_names_searched_query(self):
        messages = []
        with patch.object(
                brain, "_notify_async_reply",
                side_effect=lambda text, spoken=None:
                    messages.append(text)):
            brain._finish_multi_intent({"source": "x"}, [
                {"kind": "research", "status": "ok",
                 "fragment": "Interstellar is a 2014 film.",
                 "output_query": "Interstellar movie"}])
        self.assertTrue(messages)
        self.assertIn('I searched "Interstellar movie"', messages[0])

    def test_repeat_chain_answers_from_last_report(self):
        plan = multi_intent.build_chain(CHAIN_1)
        self.assertIsNotNone(plan)
        with patch.object(brain, "_run_multi_intent_chain") as runner:
            first = brain.handle_multi_intent(CHAIN_1, plan)
            self.assertIn("2 steps", first)
            brain._mi_chain_active = False
            brain._record_chain_run(plan, "Sir, done. Found the streamer.")
            second = brain.handle_multi_intent(CHAIN_1, plan)
        self.assertIn("did that a moment ago", second)
        self.assertIn("Found the streamer", second)
        runner.assert_called_once()

    def test_same_chain_while_active_says_already_on_it(self):
        plan = multi_intent.build_chain(CHAIN_1)
        self.assertIsNotNone(plan)
        brain._mi_chain_active = True
        try:
            brain._record_chain_run(plan, "done")
            reply = brain.handle_multi_intent(CHAIN_1, plan)
        finally:
            brain._mi_chain_active = False
        self.assertIn("already on that one", reply)


class CreatorResolutionTests(LiveFixBase):
    """Live fix: "search about this creator" searches the MAKER on screen."""

    def test_chain_research_searches_the_creator_name(self):
        captured = {}

        def fake_quick(query):
            captured["query"] = query
            return {"query": query, "spoken_summary": "About the creator."}

        plan = multi_intent.build_chain(CREATOR_ASK)
        self.assertIsNotNone(plan)
        self.assertEqual([s["kind"] for s in plan["steps"]],
                         ["screen", "research"])
        with patch.object(brain, "analyze_screen", return_value={
                "tip": "A video about Grok Bot.", "topic": "Grok Bot video",
                "creator": CREATOR}), \
             patch.object(brain, "run_quick_search", side_effect=fake_quick):
            brain._run_multi_intent_chain(CREATOR_ASK, plan)
        self.assertEqual(captured.get("query"), CREATOR)
        self.assertEqual(brain._get_last_screen_creator(), CREATOR)

    def test_corrected_followup_searches_the_creator(self):
        self._seed_screen()
        brain._set_last_screen_creator(CREATOR)
        for text in (CREATOR_CORRECTION,
                     "search about this creator on the internet",
                     "i meant the channel that made this video, search it"):
            self.assertEqual(brain._search_request_query(text), CREATOR, text)

    def test_without_a_known_creator_the_topic_stays_the_fallback(self):
        self._seed_screen()
        self.assertEqual(
            brain._resolve_reference_query("search about this creator"),
            TITLE)


if __name__ == "__main__":
    unittest.main()
