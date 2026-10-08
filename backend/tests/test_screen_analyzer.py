import unittest

from backend.services.screen_analyzer import is_screen_question


class ScreenAnalyzerTests(unittest.TestCase):
    def test_generic_what_is_this_is_not_treated_as_screen_question(self):
        self.assertFalse(is_screen_question("what is this song"))

    def test_generic_what_do_you_see_is_not_treated_as_screen_question(self):
        self.assertFalse(is_screen_question("what do you see"))

    def test_screen_reference_with_question_cue_still_matches(self):
        self.assertTrue(
            is_screen_question("on my screen there is something about Tesla, tell me more")
        )

    def test_answer_ask_with_a_screen_reference_is_a_screen_question(self):
        # Live transcript: the fast path swallowed this as plain chat because
        # no wh-word was present; the ask itself ("the answer to this KBC
        # question") is the cue.
        self.assertTrue(
            is_screen_question(
                "give me the answer to this KBC question on my screen")
        )

    def test_screen_control_verbs_still_win_over_the_answer_cue(self):
        self.assertFalse(
            is_screen_question("click the answer button on my screen"))


if __name__ == "__main__":
    unittest.main()
