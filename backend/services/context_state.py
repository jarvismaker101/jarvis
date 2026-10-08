"""Conversation context state - screen observations (RANK 1 of the redesign).

The redesign's first rank: a screen look is captured ONCE and kept, together
with an inventory of the nameable things the vision model saw, so a later
resolution ("research it", "no not that, the image to the left") can re-use
the SAME observation instead of re-capturing a screen that may have changed
(a Shorts feed advances between two captures). RANK 2 (the referent ledger)
extends this module with typed referents, salience and pendings.

Privacy: the image data is held in memory ONLY - newest observations, a few
minutes, never logged or written to disk; ``clear()`` wipes it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


#: Metadata of an observation is kept this long.
OBS_TTL_S = 900.0
#: The image data of an observation is kept this long (newest two only).
IMAGE_TTL_S = 300.0
#: Observation metadata records kept, newest first.
MAX_OBSERVATIONS = 3
#: Observations whose image data is kept (the newest two).
MAX_IMAGES = 2


@dataclass
class Item:
    """One nameable thing visible on screen (a video, image, verse...)."""

    id: str
    label: str
    label_is_text: bool = True
    truncated: bool = False
    kind: str = "other"
    creator: str = ""
    bbox: tuple = (0, 0, 0, 0)
    primary: bool = False


@dataclass
class Observation:
    """One captured look at the screen (RANK 1)."""

    id: str = ""
    at: float = 0.0
    mode: str = "full"                 # "full" | "region"
    utterance: str = ""
    answer: str = ""
    items: list = field(default_factory=list)      # list[Item]
    rejected: set = field(default_factory=set)     # item ids the user refused
    image_data_url: str = ""           # RAM only; expires; never logged
    img_hash: str = ""                 # same-screen detection
    region: dict | None = None
    # The vision attempt record that produced this observation (F37).
    vision_provider: str | None = None
    vision_model: str | None = None
    vision_attempts: list = field(default_factory=list)
    vision_degraded: bool = False


class ObservationStore:
    """Thread-safe ring of recent observations (metadata + RAM-only images)."""

    def __init__(self, clock=None, max_observations=MAX_OBSERVATIONS,
                 max_images=MAX_IMAGES, obs_ttl=OBS_TTL_S,
                 image_ttl=IMAGE_TTL_S):
        self._lock = threading.RLock()
        self._clock = clock or time.time
        self._max_observations = int(max_observations)
        self._max_images = int(max_images)
        self._obs_ttl = float(obs_ttl)
        self._image_ttl = float(image_ttl)
        self._items = []          # newest last
        self._next_id = 1

    def next_id(self):
        with self._lock:
            obs_id = "O%d" % self._next_id
            self._next_id += 1
            return obs_id

    def add(self, obs):
        with self._lock:
            self._items.append(obs)
            self._prune()
            return obs

    def get(self, obs_id):
        if not obs_id:
            return None
        with self._lock:
            self._prune()
            for obs in self._items:
                if obs.id == obs_id:
                    return obs
        return None

    def latest(self, max_age=None, with_image=False):
        """The newest observation (optionally fresh / carrying its image)."""
        with self._lock:
            self._prune()
            for obs in reversed(self._items):
                if with_image and not obs.image_data_url:
                    continue
                if max_age is not None and \
                        (self._clock() - obs.at) > float(max_age):
                    continue
                return obs
        return None

    def clear(self):
        with self._lock:
            self._items = []

    def _prune(self):
        now = self._clock()
        kept = []
        for obs in self._items:
            if now - obs.at > self._obs_ttl:
                continue
            if obs.image_data_url and now - obs.at > self._image_ttl:
                obs.image_data_url = ""
            kept.append(obs)
        kept = kept[-self._max_observations:]
        with_images = kept[-self._max_images:] if self._max_images > 0 else []
        keep_ids = {id(obs) for obs in with_images if obs.image_data_url}
        for obs in kept:
            if id(obs) not in keep_ids:
                obs.image_data_url = ""
        self._items = kept


#: The process-wide store.
OBSERVATIONS = ObservationStore()
