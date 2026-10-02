"""Shared test bootstrap.

The live Fish WebSocket TTS engine (S7) is ON in production and would open a
real socket to api.fish.audio from any test that speaks a reply. Unit tests
must never touch the network, so the engine is switched off for the whole
test run here; the S7 specs exercise it through injected sessions instead.
"""

import os

os.environ.setdefault("JARVIS_FISH_WS", "0")