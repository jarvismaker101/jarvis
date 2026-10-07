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
JAMES = "James Dark"
REFINE_ASK = ("ok but you searched a different man by that same name , i "
              "want you to seach the youtube content creator by that name")
MONALISA_ASK = "search on the internet about the creator of monalisa"
RELEASE_ASK = ("find out how people are reacting to that release from "
               "openai")
RELEASE_MEANT = ("no i meant the reactions to that release you just "
                 "researched for me")
VERSE_ASK = ("research about this verse on my screen from bhagwat gita "
             "and tell me what does it actually say")
VERSE_CONFIRM = "yes that is the verse i wanted you to research about"
VERSE_TOPIC = "Bhagavad Gita Chapter 4 verse 5"


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


class ResearchRefinementTests(LiveFixBase):
    """Live fix: name corrections refine the last researched NAME; a
    self-referential classifier query is never searched or stored."""

    def test_correction_refines_to_the_last_name(self):
        brain._set_last_research_topic(JAMES)
        self.assertEqual(brain._refine_last_name_query(REFINE_ASK),
                         "James Dark youtube channel")
        with patch.object(brain, "classify_intent",
                          return_value={
                              "intent": "research",
                              "query": "YouTube content creator with the "
                                       "name previously searched"}):
            self.assertEqual(brain.derive_research_query(REFINE_ASK),
                             "James Dark youtube channel")

    def test_meta_queries_are_rejected_and_never_stored(self):
        with patch.object(brain, "classify_intent",
                          return_value={"intent": "research",
                                        "query": "the name previously searched"}):
            self.assertEqual(brain.derive_research_query("look into it"), "")
        brain._set_last_research_topic(JAMES)
        brain._set_last_research_topic(
            "YouTube content creator with the name previously searched")
        self.assertEqual(brain._last_research_topic, JAMES)

    def test_named_subject_clause_is_not_replaced_by_screen_topic(self):
        self._seed_screen()
        brain._set_last_screen_creator(JAMES)
        with patch.object(brain, "classify_intent",
                          return_value={"intent": "research",
                                        "query": "creator of Mona Lisa"}):
            query = brain._resolve_search_query(MONALISA_ASK)
        self.assertEqual(query, "creator of Mona Lisa")

    def test_live_correction_routes_to_refined_query(self):
        brain._set_last_research_topic(JAMES)
        calls = []

        def fake_research(query, **kwargs):
            calls.append(query)
            return "Quick lookup for that, sir."

        with patch.object(brain, "classify_intent",
                          return_value={
                              "intent": "research",
                              "query": "YouTube content creator with the "
                                       "name previously searched"}), \
             patch.object(brain, "handle_research_intent",
                          side_effect=fake_research), \
             patch.object(brain, "orchestrator_select_route",
                          return_value="legacy"), \
             patch.object(brain, "handle_chat", return_value="chat"):
            reply = brain.process_message(REFINE_ASK, sync_voice=False)
        self.assertEqual(calls, ["James Dark youtube channel"])
        self.assertIn("Quick lookup", reply)


class ReactionQueryTests(LiveFixBase):
    """Live fix: reactions/opinions about the last researched subject.

    Transcript: after a chain researched OpenAI, "find out how people are
    reacting to that release from openai" searched the sentence verbatim,
    and the correction "no i meant the reactions to that release you just
    researched for me" fell to chat, which answered from stale context.
    """

    def test_reaction_request_rewrites_to_the_researched_subject(self):
        brain._set_last_research_topic("OpenAI AI math results")
        self.assertEqual(brain._reaction_query(RELEASE_ASK),
                         "reactions to OpenAI AI math results")
        self.assertEqual(brain._resolve_search_query(RELEASE_ASK),
                         "reactions to OpenAI AI math results")

    def test_screen_topic_is_the_fallback_referent(self):
        self._seed_screen()
        self.assertEqual(brain._reaction_query(RELEASE_ASK),
                         "reactions to " + TITLE)

    def test_meant_correction_restates_the_same_request(self):
        brain._set_last_research_topic("OpenAI AI math results")
        self.assertEqual(brain._meant_refinement_query(RELEASE_MEANT),
                         "reactions to OpenAI AI math results")

    def test_no_stacking_when_the_stored_subject_is_already_reactions(self):
        brain._set_last_research_topic("reactions to OpenAI AI math results")
        self.assertEqual(brain._reaction_query(RELEASE_MEANT),
                         "reactions to OpenAI AI math results")

    def test_instruction_sentences_are_never_referents(self):
        brain._set_last_research_topic(
            "find out how people are reacting to that release from openai")
        self.assertEqual(brain._reaction_query(RELEASE_MEANT), "")

    def test_concrete_subject_keeps_its_own_words(self):
        brain._set_last_research_topic("OpenAI AI math results")
        self.assertEqual(
            brain._reaction_query(
                "search for what people are saying about the new OpenAI "
                "o3 model"),
            "reactions to the new OpenAI o3 model")

    def test_without_a_referent_the_pointer_is_not_searched(self):
        self.assertEqual(brain._reaction_query(RELEASE_ASK), "")
        self.assertEqual(brain._meant_refinement_query(RELEASE_MEANT), "")

    def test_plain_searches_are_untouched(self):
        self.assertEqual(brain._reaction_query(
            "search the best gaming laptop under 2000 dollars"), "")

    def test_live_meant_correction_routes_to_research_not_chat(self):
        brain._set_last_research_topic("OpenAI AI math results")
        calls = []

        def fake_research(query, **kwargs):
            calls.append(query)
            return "Quick lookup for that, sir."

        with patch.object(brain, "classify_intent",
                          return_value={"intent": "chat"}), \
             patch.object(brain, "handle_research_intent",
                          side_effect=fake_research), \
             patch.object(brain, "orchestrator_select_route",
                          return_value="legacy"), \
             patch.object(brain, "handle_chat", return_value="chat"):
            reply = brain.process_message(RELEASE_MEANT, sync_voice=False)
        self.assertEqual(calls, ["reactions to OpenAI AI math results"])
        self.assertIn("Quick lookup", reply)


class BackReferenceTests(LiveFixBase):
    """Live fix: "yes that is the verse i wanted you to research about".

    The screen showed Bhagavad Gita Chapter 4 verse 5; the classifier
    turned the consent into the query "verse research" (lab suppliers),
    and later a chat fallback improvised a wrong verse. A pure
    back-reference must resolve to what was last seen — never be
    classified, never reach chat.
    """

    def test_consent_backreference_resolves_to_the_screen_topic(self):
        brain._set_last_screen_topic(VERSE_TOPIC)
        self.assertEqual(brain._backreference_query(VERSE_CONFIRM),
                         VERSE_TOPIC)
        self.assertEqual(brain._resolve_search_query(VERSE_CONFIRM),
                         VERSE_TOPIC)

    def test_short_consent_also_resolves(self):
        brain._set_last_screen_topic(VERSE_TOPIC)
        self.assertEqual(brain._resolve_search_query("yes research that"),
                         VERSE_TOPIC)

    def test_route_never_consults_the_classifier(self):
        brain._set_last_screen_topic(VERSE_TOPIC)
        calls = []

        def fake_research(query, **kwargs):
            calls.append(query)
            return "Quick lookup for that, sir."

        with patch.object(brain, "classify_intent",
                          return_value={"intent": "research",
                                        "query": "verse research"}) as cls, \
             patch.object(brain, "handle_research_intent",
                          side_effect=fake_research), \
             patch.object(brain, "orchestrator_select_route",
                          return_value="legacy"), \
             patch.object(brain, "handle_chat", return_value="chat"):
            reply = brain.process_message(VERSE_CONFIRM, sync_voice=False)
        self.assertEqual(calls, [VERSE_TOPIC])
        self.assertIn("Quick lookup", reply)
        cls.assert_not_called()

    def test_without_a_topic_it_asks_once(self):
        self.assertEqual(brain._resolve_search_query(VERSE_CONFIRM), "")
        with patch.object(brain, "handle_research_intent") as research, \
             patch.object(brain, "orchestrator_select_route",
                          return_value="legacy"), \
             patch.object(brain, "handle_chat", return_value="chat"):
            reply = brain.process_message(VERSE_CONFIRM, sync_voice=False)
        research.assert_not_called()
        self.assertIn("name it", reply.lower())

    def test_own_subject_searches_are_not_backreferences(self):
        brain._set_last_screen_topic(VERSE_TOPIC)
        self.assertEqual(brain._backreference_query("research the news"),
                         "")
        self.assertEqual(
            brain._backreference_query(
                "search for that movie review on imdb"), "")


if __name__ == "__main__":
    unittest.main()
