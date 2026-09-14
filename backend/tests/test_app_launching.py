import unittest
from unittest.mock import patch
from backend.core.brain import process_message


class AppLaunchingTests(unittest.TestCase):
    @patch("backend.core.brain.execute_multiple")
    def test_command_open_brave_routes_to_launch_app(self, mock_execute):
        # We process the command "command open brave"
        process_message("command open brave", from_voice=True, sync_voice=False)
        
        # Verify that execute_multiple was called with the launch_app action
        mock_execute.assert_called_once()
        actions = mock_execute.call_args[0][0]
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["action"], "launch_app")
        self.assertEqual(actions[0]["input"], "brave")
        self.assertEqual(actions[0]["browser"], "brave")

    @patch("backend.core.brain.execute_multiple")
    def test_command_open_notepad_routes_to_launch_app(self, mock_execute):
        process_message("command open notepad", from_voice=True, sync_voice=False)
        
        mock_execute.assert_called_once()
        actions = mock_execute.call_args[0][0]
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["action"], "launch_app")
        self.assertEqual(actions[0]["input"], "notepad")
        self.assertIsNone(actions[0]["browser"])


if __name__ == "__main__":
    unittest.main()
