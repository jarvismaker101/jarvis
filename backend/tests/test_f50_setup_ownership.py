"""F50 — backend ownership of setup launches and voice-state generations.

Acceptance pinned here: the setup launch is a BACKEND job (one owner, stoppable,
journalled) rather than something the voice worker spawns itself, and a
voice-state snapshot published by a superseded/expired worker incarnation is
never served to the UI as current truth.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    TestClient = None

from backend.api import routes
from backend.core import memory_store
from backend.services import intelligence_state as istate


class MemoryOwnershipTests(unittest.TestCase):
    """F50 — the BACKEND is the store's designated writer, in production too."""

    def setUp(self):
        istate.reset()
        self._tmp = tempfile.TemporaryDirectory()
        memory_store.MEMORY_ENABLED = True

    def tearDown(self):
        memory_store.stop_scheduler()
        memory_store.close()
        memory_store.MEMORY_ENABLED = False
        istate.reset()
        self._tmp.cleanup()

    def test_opening_the_store_declares_the_backend_as_its_writer(self):
        memory_store.configure(os.path.join(self._tmp.name, "mem.db"))
        self.assertEqual(istate.durable.owner_of(istate.RESOURCE_MEMORY),
                         "backend")
        # A store reached through _conn() alone (production startup) declares
        # ownership too: the lease is gone until the connection is opened.
        memory_store.configure(os.path.join(self._tmp.name, "mem2.db"))
        istate.durable.release(istate.RESOURCE_MEMORY, "backend")
        self.assertEqual(istate.durable.owner_of(istate.RESOURCE_MEMORY), "")
        memory_store._ownership_declared = False
        memory_store._conn()
        self.assertEqual(istate.durable.owner_of(istate.RESOURCE_MEMORY),
                         "backend")

    def test_a_foreign_live_owner_blocks_writes(self):
        istate.durable.claim(istate.RESOURCE_MEMORY, "voice-io")
        memory_store.configure(os.path.join(self._tmp.name, "mem.db"))
        with self.assertRaises(istate.NotDurableOwner):
            memory_store.record_event("kind", "a summary")
        # …and the refusal is not a transient state the store writes around.
        with self.assertRaises(istate.NotDurableOwner):
            memory_store.begin_request("hello")


class SetupLaunchTests(unittest.TestCase):
    def test_setup_launch_is_a_backend_job(self):
        calls = []

        def fake_launch(app):
            calls.append(app)
            return True

        with patch("backend.core.executor.launch_app", fake_launch):
            body = routes.launch_voice_setup({"setup": "normal"})
        self.assertTrue(body["ok"])
        self.assertTrue(body["job_id"])
        # The handler runs on the effect thread; wait for it to finish.
        import time

        deadline = time.time() + 5
        while time.time() < deadline and len(calls) < len(
                routes.NORMAL_SETUP_APPS):
            time.sleep(0.02)
        self.assertEqual(calls, list(routes.NORMAL_SETUP_APPS))

    def test_unknown_setup_is_rejected(self):
        from fastapi import HTTPException

        with self.assertRaises(HTTPException):
            routes.launch_voice_setup({"setup": "rocket"})

    def test_setup_launch_is_journalled(self):
        istate.reset()
        with patch("backend.core.executor.launch_app", lambda app: True):
            routes.launch_voice_setup({"setup": "normal"})
        import time

        time.sleep(0.2)
        events = istate.journal.events(kind="setup")
        self.assertTrue(events, "the setup job must enter the shared history")
        first = events[0].to_dict() if hasattr(events[0], "to_dict") else events[0]
        self.assertEqual(first["effect"], "normal-setup")


class VoiceStateGenerationTests(unittest.TestCase):
    def setUp(self):
        istate.reset()
        routes._published_voice.clear()
        routes._published_voice_seq = 0

    def tearDown(self):
        istate.reset()
        routes._published_voice.clear()
        routes._published_voice_seq = 0

    def test_identified_publish_is_served_while_live(self):
        routes.publish_voice_state({
            "owner": "voice-io@123", "role": istate.ROLE_VOICE,
            "generation": 1, "listening": True,
        })
        state = routes.get_published_voice_state()
        self.assertTrue(state.get("listening"))

    def test_superseded_generation_publish_is_rejected(self):
        routes.publish_voice_state({
            "owner": "voice-io@123", "role": istate.ROLE_VOICE,
            "generation": 1, "listening": True,
        })
        istate.worker_states.supersede(istate.ROLE_VOICE, "voice-io@456", pid=456)
        body = routes.publish_voice_state({
            "owner": "voice-io@123", "role": istate.ROLE_VOICE,
            "generation": 1, "listening": False,
        })
        self.assertTrue(body.get("stale"), "an old incarnation cannot publish")
        # The superseded incarnation's snapshot is NOT served as current truth…
        self.assertEqual(routes.get_published_voice_state(), {})
        # …until the new incarnation publishes its own state.
        routes.publish_voice_state({
            "owner": "voice-io@456", "role": istate.ROLE_VOICE,
            "generation": 2, "listening": True,
        })
        self.assertTrue(routes.get_published_voice_state().get("listening"))

    def test_expired_worker_state_is_not_served(self):
        routes.publish_voice_state({
            "owner": "voice-io@123", "role": istate.ROLE_VOICE,
            "generation": 1, "listening": True,
        })
        istate.worker_states.clear(istate.ROLE_VOICE)
        self.assertEqual(routes.get_published_voice_state(), {},
                         "an expired worker's state is not current truth")

    def test_legacy_publisher_without_identity_still_works(self):
        routes.publish_voice_state({"listening": True})
        self.assertTrue(routes.get_published_voice_state().get("listening"))


if __name__ == "__main__":
    unittest.main()
