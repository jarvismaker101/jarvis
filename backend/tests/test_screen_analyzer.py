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


if __name__ == "__main__":
    unittest.main()
