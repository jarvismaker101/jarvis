"""S13 - background results join the conversation history.

Acceptance: a research summary, task completion or reminder that was
actually delivered is ALSO an assistant message in chat history, marked as
a background result - so "what was the second point?" or "open that" just
works. A result that was never delivered is never remembered.
"""

import unittest
from unittest.mock import patch

from backend.core import memory
from backend.core import brain


class _HistoryIsolation(unittest.TestCase):
    def setUp(self):
        memory.clear_history()
        self.addCleanup(memory.clear_history)


class DeliveredResultHistoryTests(_HistoryIsolation):
    """Only a delivered result is remembered, and it is marked in-band."""

    def _notify(self, text, spoken=None):
        recorded = []

        def callback(text, spoken=None):
            recorded.append((text, spoken))

        with patch.object(brain, "_async_reply_callback", callback):
            delivered = brain._notify_async_reply(text, spoken)
        return delivered, recorded

    def test_a_delivered_result_joins_the_history_marked(self):
        text = "Research complete. Two findings: A, and B."
        delivered, recorded = self._notify(text)
        self.assertTrue(delivered)
        self.assertEqual(recorded, [(text, None)])
        history = memory.get_history()
        self.assertEqual(history[-1], {
            "role": "assistant",
            "content": "[background result] %s" % text,
        })

    def test_the_text_is_carried_byte_exact(self):
        text = "Opening https://example.com/x?y=1 - done, Sir."
        self._notify(text)
        self.assertEqual(
            memory.get_history()[-1]["content"],
            "[background result] Opening https://example.com/x?y=1 - done, Sir.")

    def test_an_undelivered_result_never_joins_the_history(self):
        def broken(text, spoken=None):
            raise RuntimeError("surface refused")

        with patch.object(brain, "_async_reply_callback", broken):
            delivered = brain._notify_async_reply("nobody heard this")
        self.assertFalse(delivered)
        self.assertEqual(memory.get_history(), [])

    def test_no_callback_is_not_a_delivery(self):
        with patch.object(brain, "_async_reply_callback", None):
            delivered = brain._notify_async_reply("no surface at all")
        self.assertFalse(delivered)
        self.assertEqual(memory.get_history(), [])

    def test_an_empty_text_is_ignored_entirely(self):
        recorded = []
        with patch.object(brain, "_async_reply_callback",
                          lambda t, s=None: recorded.append(t)):
            self.assertFalse(brain._notify_async_reply(""))
        self.assertEqual(recorded, [])
        self.assertEqual(memory.get_history(), [])

    def test_a_history_write_failure_still_reports_delivery(self):
        # The delivery acknowledgement (F10) must not hinge on the new
        # history write: the user DID hear the result.
        recorded = []
        with patch.object(brain, "_async_reply_callback",
                          lambda t, s=None: recorded.append(t)), \
                patch.object(brain, "add_message",
                             side_effect=RuntimeError("disk full")):
            delivered = brain._notify_async_reply("delivered anyway")
        self.assertTrue(delivered)
        self.assertEqual(recorded, ["delivered anyway"])


if __name__ == "__main__":
    unittest.main()
