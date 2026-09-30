"""P1-01 — chunk the streamed reply so the first chunk comes sooner and no
word is ever split in half.

The defects pinned here (old ``StreamSpeaker._process_text``):

  * the sentence rule required whitespace AFTER the punctuation, so a finished
    sentence was not recognised until the NEXT delta arrived — the first words
    waited on a delta the model may not send for a while;
  * once 40 characters were buffered the whole buffer was flushed wherever the
    cut happened to fall, so TTS could be handed a fragment like
    ``"supercali"``;
  * a complete sentence followed by a pause sat in the buffer until the 300ms
    stall flush, because nothing treated the trailing punctuation as a boundary.

Audio, TTS and the network are mocked; nothing here touches a device.
"""

import re
import time
import unittest
from unittest.mock import patch

from backend.services import voice as voice_mod


def _tokens(text):
    return text.split()


class _SpeakerHarness(unittest.TestCase):
    """Real StreamSpeaker with the playback boundary mocked."""

    def setUp(self):
        with voice_mod._state_lock:
            voice_mod._speech_generation += 1

    def _speaker(self, drive_worker=False):
        sp = voice_mod.StreamSpeaker()
        captured = []
        orig_put = sp._queue.put

        def cap_put(item):
            if item is not voice_mod._STREAM_STOP:
                captured.append(item)
            return orig_put(item)

        if not drive_worker:
            # Deterministic: nothing may pull the item behind our back, and no
            # stall flush may fire while a test is asserting "nothing yet".
            sp._start_worker = lambda: None

        self._patches = [
            patch.object(voice_mod, "prefetch_fish_audio"),
            patch.object(voice_mod, "_speak_chunk", return_value=True),
            patch.object(voice_mod, "play_ready_earcon"),
            patch.object(sp._queue, "put", side_effect=cap_put),
        ]
        for item in self._patches:
            item.start()
            self.addCleanup(item.stop)
        self.addCleanup(sp.close)
        return sp, captured

    def _wait_for(self, predicate, timeout=1.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()


class FirstChunkTests(_SpeakerHarness):
    """Requirement 1: break at the first clause, not at a full sentence."""

    def test_first_chunk_breaks_at_a_clause_without_a_following_delta(self):
        sp, captured = self._speaker()
        # ONE delta. No whitespace follows the comma, and no later delta ever
        # arrives — the old rule could not see this boundary at all.
        sp.feed("Hello there friend, how are you today")
        self.assertTrue(captured, "no chunk was emitted for a clear clause break")
        self.assertEqual(captured[0], "Hello there friend,")

    def test_the_first_chunk_is_not_the_whole_sentence(self):
        sp, captured = self._speaker()
        sp.feed("Hello there friend, how are you today")
        self.assertNotIn("today", captured[0],
                         "the first chunk waited for the whole sentence")

    def test_a_clause_break_needs_three_words_before_it(self):
        sp, captured = self._speaker()
        sp.feed("One two, three four five six seven eight nine")
        self.assertNotIn("One two,", captured,
                         "a two-word head must not start playback")

    def test_three_words_before_a_clause_is_enough(self):
        sp, captured = self._speaker()
        sp.feed("One two three, four five six")
        self.assertEqual(captured[0], "One two three,")

    def test_the_first_chunk_arrives_on_the_clause_delta_not_later(self):
        """The latency acceptance: word-by-word deltas flush at the clause."""
        sp, captured = self._speaker()
        deltas = ["Hello ", "there ", "friend,", " how are you today"]
        fed = 0
        for delta in deltas:
            sp.feed(delta)
            fed += 1
            if captured:
                break
        self.assertTrue(captured, "no first chunk after any delta")
        self.assertEqual(fed, 3,
                         "playback waited for a delta past the clause boundary")
        self.assertEqual(captured[0], "Hello there friend,")

    def test_a_long_delta_without_any_punctuation_cuts_early(self):
        """The threshold is on the BUFFER, so playback starts mid-sentence."""
        sp, captured = self._speaker()
        source = "a" * 12 + " " + "b" * 12 + " " + "c" * 12
        self.assertGreater(len(source), voice_mod.FIRST_CHUNK_MIN_CHARS)
        sp.feed(source)                          # ONE delta, no punctuation
        self.assertTrue(captured, "no first chunk for a punctuation-free delta")
        chunk = captured[0]
        self.assertLess(len(chunk), len(source), "nothing was cut but a chunk came")
        # ...and the cut is on a word boundary, not mid-word.
        self.assertEqual(source[len(chunk)], " ")
        self.assertEqual(_tokens(chunk), _tokens(source)[:len(_tokens(chunk))])

    def test_a_sentence_ending_the_buffer_is_not_held_for_a_delta(self):
        """Requirement 3: punctuation at the end of the buffer is a boundary."""
        sp, captured = self._speaker(drive_worker=True)
        sp.feed("All done.")
        self.assertEqual(captured, [], "flushed before the settle window")
        started = time.perf_counter()
        self.assertTrue(self._wait_for(lambda: captured, 1.0),
                        "a complete sentence was held indefinitely")
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, voice_mod.STALL_FLUSH_SECONDS,
                        "the 300ms stall path fired, not the settle rule")
        self.assertEqual(captured[0], "All done.")

    def test_a_clause_inside_a_number_is_not_a_boundary(self):
        """'3.5' must not read as 'three point' + 'five'."""
        sp, captured = self._speaker()
        sp.feed("the value is 3.5 million")
        self.assertEqual(captured, [],
                         "a decimal point was treated as a clause break")


class WordIntegrityTests(_SpeakerHarness):
    """Requirement 4: a chunk never ends inside a word."""

    def test_a_word_straddling_the_length_cap_is_never_cut(self):
        """The regression test the audit asks for.

        The cap lands in the middle of a word; the chunk must extend to the
        next word boundary instead of cutting.
        """
        sp, captured = self._speaker()
        sp.feed("Hello there friend, ")          # first chunk goes out here
        head = "ab " * 66                        # 198 chars
        sp.feed(head + "supercalifragilistic" + " tail")
        self.assertTrue(captured[1:], "the long run produced no later chunk")
        chunk = captured[1]
        self.assertIn("supercalifragilistic", chunk,
                      "the word straddling the cap was split: %r" % chunk[-30:])
        self.assertGreater(len(chunk), voice_mod.LATER_CHUNK_MAX_CHARS,
                           "the chunk was cut at the cap instead of the word")

    def test_a_word_split_across_deltas_is_never_emitted_in_fragments(self):
        """The historical bug: 40 chars buffered mid-word flushed immediately."""
        long_word = "pneumonoultramicroscopicsilicovolcanoconiosis"
        self.assertEqual(len(long_word), 45)
        sp, captured = self._speaker()
        sp.feed(long_word[:40])                  # ends exactly on the old cap
        sp.feed(long_word[40:])
        sp.finish()
        self.assertEqual(captured, [long_word],
                         "the word was emitted in fragments: %r" % (captured,))

    def test_no_chunk_ever_ends_mid_word(self):
        """Token integrity: no fragment, no loss, original order."""
        sp, captured = self._speaker()
        source = ("The quick brown fox jumps over the lazy dog, and then it "
                  "runs away into the deep dark forest where nobody follows, "
                  "because the night is long and the path is narrow indeed.")
        for index in range(0, len(source), 7):
            sp.feed(source[index:index + 7])
        sp.finish()
        self.assertTrue(captured)
        self.assertEqual(_tokens(" ".join(captured)), _tokens(source),
                         "a word was split, dropped or reordered")

    def test_a_single_unbroken_token_is_not_cut_mid_word(self):
        sp, captured = self._speaker()
        sp.feed("x" * 400)                       # no boundary anywhere
        self.assertEqual(captured, [], "cut a word with no boundary to use")

    def test_a_long_unbroken_run_is_cut_at_a_word_boundary(self):
        sp, captured = self._speaker()
        sp.feed("Hello there friend, ")
        sp.feed("word " * 60)                    # 300 chars, no punctuation
        chunk = captured[1]
        self.assertLessEqual(len(chunk), voice_mod.LATER_CHUNK_MAX_CHARS + 1)
        self.assertEqual(chunk.strip().split()[-1], "word",
                         "cut mid-word: %r" % chunk[-20:])

    def test_the_stream_ending_mid_word_still_emits_whole_words(self):
        sp, captured = self._speaker()
        sp.feed("the quick brown fox jump")
        sp.finish()
        self.assertEqual(_tokens(" ".join(captured)),
                         _tokens("the quick brown fox jump"))
        self.assertEqual(captured[-1], "the quick brown fox jump")

    def test_text_reaches_tts_unaltered(self):
        """F12: wording is never rewritten, only re-cut at whitespace."""
        sp, captured = self._speaker()
        source = ("Alpha beta gamma, delta epsilon zeta. Eta theta iota kappa "
                  "lambda mu, nu xi omicron pi rho sigma tau.")
        for index in range(0, len(source), 5):
            sp.feed(source[index:index + 5])
        sp.finish()
        normalise = lambda t: re.sub(r"\s+", " ", t).strip()
        self.assertEqual(normalise(" ".join(captured)), normalise(source))
        self.assertEqual(_tokens(" ".join(captured)), _tokens(source))



class EmptyChunkTests(_SpeakerHarness):
    """Every chunk handed to _enqueue holds a complete word."""

    def test_empty_and_blank_deltas_enqueue_nothing(self):
        sp, captured = self._speaker()
        for delta in ("", None, " ", "   ", "\n", "\t "):
            sp.feed(delta)
        self.assertEqual(captured, [], "an empty utterance reached TTS")

    def test_enqueue_directly_rejects_an_empty_chunk(self):
        sp, captured = self._speaker()
        for value in ("", "   ", None):
            sp._enqueue(value)
        self.assertEqual(captured, [])

    def test_no_chunk_is_whitespace_only(self):
        sp, captured = self._speaker()
        sp.feed("Hello there friend,   how are you today.  ")
        sp.finish()
        self.assertTrue(captured)
        for chunk in captured:
            self.assertTrue(chunk.strip(), "whitespace-only chunk: %r" % chunk)
            self.assertEqual(chunk, chunk.strip())


class LaterChunkTests(_SpeakerHarness):
    """Requirement 2: later chunks follow sentence boundaries."""

    def test_a_later_sentence_boundary_is_used(self):
        sp, captured = self._speaker(drive_worker=True)
        sp.feed("Hello there friend, how are you today. And also tomorrow.")
        self.assertEqual(captured[0], "Hello there friend,")
        self.assertTrue(self._wait_for(lambda: len(captured) >= 2, 1.0),
                        "the second sentence never reached the queue")
        self.assertEqual(captured[1], "how are you today.")

    def test_the_sentence_split_still_happens_on_the_next_delta(self):
        sp, captured = self._speaker()
        sp.feed("Alpha beta gamma delta. ")
        self.assertEqual(captured[0], "Alpha beta gamma delta.")

    def test_a_sentence_at_the_buffer_end_is_flushed(self):
        sp, captured = self._speaker(drive_worker=True)
        sp.feed("First we check the disk. ")
        sp.feed("Then we check the memory.")
        self.assertTrue(self._wait_for(lambda: len(captured) >= 2, 1.0))
        self.assertEqual(captured[1], "Then we check the memory.")


class DedupTests(_SpeakerHarness):
    """P0-05: chunk boundaries feed the cache key, so they must be stable."""

    def _chunks_for(self, text):
        sp, captured = self._speaker()
        sp.feed(text)
        sp.finish()
        return captured

    def test_identical_text_produces_identical_chunks(self):
        text = ("Hello there friend, this is a longer sentence that will be "
                "cut at several boundaries so the shape is visible end to end.")
        first = self._chunks_for(text)
        second = self._chunks_for(text)
        self.assertEqual(first, second,
                         "chunking is not deterministic; identical sentences "
                         "would no longer share one synthesis")
        self.assertGreater(len(first), 1)


class ConstantTests(unittest.TestCase):
    """The knobs the audit names, pinned so a later edit cannot drift them."""

    def test_first_chunk_knobs(self):
        self.assertEqual(voice_mod.FIRST_CHUNK_MIN_WORDS, 3)
        self.assertEqual(voice_mod.FIRST_CHUNK_MIN_CHARS, 30)

    def test_the_stall_flush_is_still_the_last_resort(self):
        self.assertEqual(voice_mod.STALL_FLUSH_SECONDS, 0.3)

    def test_the_settle_window_precedes_the_stall(self):
        self.assertLess(voice_mod.PUNCTUATION_SETTLE_SECONDS,
                        voice_mod.STALL_FLUSH_SECONDS)

    def test_the_poll_is_finer_than_the_settle_window(self):
        """Otherwise the 120ms rule could not fire on time."""
        self.assertLess(voice_mod.STREAM_POLL_SECONDS,
                        voice_mod.PUNCTUATION_SETTLE_SECONDS)

    def test_the_later_chunk_cap_is_generous(self):
        self.assertGreaterEqual(voice_mod.LATER_CHUNK_MAX_CHARS, 200)

    def test_clause_breaks_match_the_audit_list(self):
        for char in ",;:\u2014.!?":
            self.assertIn(char, voice_mod.FIRST_CHUNK_CLAUSE_BREAKS)

    def test_min_flush_is_still_accepted_for_compatibility(self):
        sp = voice_mod.StreamSpeaker(min_flush_chars=80)
        self.addCleanup(sp.close)
        self.assertEqual(sp._min_flush, 80)


if __name__ == "__main__":
    unittest.main()

