"""One real playback owner for Jarvis audio (F32).

Before this module there were three independent playback authorities:
``sd.stop()`` (kills *all* PortAudio streams) racing explicit
``OutputStream`` objects in ``fish_voice._do_pcm_stream``, plus a playback
caller that could block waiting on a prefetch owner before registering its
own stoppable handle. Caches were keyed by text only, no generation
token lived in the audio layer, and nothing tracked how far a spoken
utterance had actually progressed.

``AudioActor`` is that single owner. It owns:

* the **generation token** - ``begin()`` bumps it; every chunk carries the
  generation it was produced for, and stale-generation PCM is discarded at
  every await/play boundary instead of leaking into a newer answer;
* the **PCM ring buffer** - bounded, per-utterance, drained in order;
* the **synthesis subscriptions** - producers (prefetch threads) push
  chunks with ``feed_chunk()``; the play loop *subscribes* to arriving
  chunks (a waitable event) instead of blocking on full-synthesis waits;
* the **output stream** - created through an injectable ``stream_factory``;
  ``abort()`` stops/aborts that specific stream (never ``sd.stop()`` for
  everyone) and discards buffered PCM;
* the **spoken cursor** - ``spoke_bytes()`` tracks how far the current
  utterance has actually been written to the device, so interrupted speech
  can resume from exactly that point.

The module is hardware-free by construction: tests inject a recording
``stream_factory``; the default wraps ``sounddevice.OutputStream``.

[Fable-5 F32] "Give Playback One Real Owner".
"""

import threading
import time
from collections import deque, namedtuple

from backend.services import latency as _latency

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 2  # s16le
#: Bounded ring capacity (in PCM chunks). ~32 chunks x 4kB ~ 128 kB per
#: utterance of jitter allowance for slow synthesis between await points.
DEFAULT_RING_CHUNKS = 32
DEFAULT_CHUNK_BYTES = 4096
#: How long a producer waits for ring space before giving up (F32
#: backpressure: overflow must never silently drop PCM).
FEED_TIMEOUT = 30.0
#: [P0-06] Bounded floor for the play loop's wait for new audio.
#:
#: The loop used to sleep in 250ms slices on ``_stop``, a DIFFERENT event from
#: the one producers set, so the first chunk of an utterance could wait up to a
#: quarter second before it reached the device. It now waits on the ring
#: condition variable that producers notify, so the wake is immediate.
#:
#: The floor exists ONLY so a missed notify cannot park the loop forever - it
#: is not a polling interval, and it is two orders of magnitude below the old
#: sleep so an idle ring is never perceptible.
PLAY_WAIT_FLOOR_SECONDS = 0.05

#: One immutable, generation-tagged chunk (F32). The generation travels with
#: the bytes so a chunk produced for an old answer can never be written into a
#: newer one, no matter how the ring is drained.
Chunk = namedtuple("Chunk", "key generation pcm")

_stream_lock = threading.Lock()
_stream_factory_override = None


class PlaybackAborted(Exception):
    """Raised by the play loop when the current generation is aborted."""


class SoundDeviceStream:
    """Default output stream wrapper around ``sounddevice.OutputStream``.

    Exposes just the four primitives the actor needs, so tests can supply
    a recording fake with the same surface.
    """

    def __init__(self, samplerate=DEFAULT_SAMPLE_RATE, channels=DEFAULT_CHANNELS,
                 device=None, blocksize=4096):
        import sounddevice as sd

        self._stream = sd.OutputStream(
            samplerate=samplerate,
            channels=channels,
            dtype="int16",
            device=device,
            blocksize=blocksize,
            latency="high",
        )
        # A blocking PortAudio output stream is opened in the STOPPED state,
        # and ``Pa_WriteStream`` rejects writes until it is started
        # (``paStreamIsStopped`` / -9983). sounddevice only calls ``start()``
        # from its ``with`` statement, never from ``__init__``, so it has to
        # be done explicitly here. Without this every ``write()`` raises and
        # NO audio is ever heard, whatever engine produced the PCM.
        self._stream.start()

    def write(self, pcm_bytes):
        import numpy as np

        arr = np.frombuffer(pcm_bytes, dtype=np.int16)
        if arr.size:
            self._stream.write(arr)

    def stop(self):
        try:
            self._stream.stop()
        except Exception:
            pass

    def close(self):
        try:
            self._stream.close()
        except Exception:
            pass


class RawPcmStream:
    """Adapts an existing PortAudio stream to the actor's byte contract.

    The actor writes raw s16le bytes; ``sd.OutputStream.write`` wants an array
    shaped ``(frames, channels)``. This wrapper does that conversion (and
    nothing else), so a decoded-MP3 path can still hand its stream to the ONE
    owner instead of writing to it behind the owner's back.
    """

    def __init__(self, stream, channels=DEFAULT_CHANNELS):
        self._stream = stream
        self._channels = max(1, int(channels))

    def write(self, pcm_bytes):
        import numpy as np

        arr = np.frombuffer(pcm_bytes, dtype=np.int16)
        if self._channels > 1:
            usable = (arr.size // self._channels) * self._channels
            if usable != arr.size:
                arr = arr[:usable]
            if arr.size:
                arr = arr.reshape((-1, self._channels))
        if arr.size:
            self._stream.write(arr)

    def stop(self):
        try:
            self._stream.stop()
        except Exception:
            pass

    def close(self):
        try:
            self._stream.close()
        except Exception:
            pass


def make_sounddevice_factory(device=None, samplerate=DEFAULT_SAMPLE_RATE,
                             channels=DEFAULT_CHANNELS, blocksize=4096):
    """A stream factory the actor can call from its own play thread.

    The stream is created INSIDE the play loop's thread, so PortAudio objects
    stay on one thread.
    """
    def _factory():
        import sounddevice as sd

        stream = sd.OutputStream(samplerate=samplerate, channels=channels,
                                 dtype="int16", device=device, blocksize=blocksize,
                                 latency="high")
        # Same reason as SoundDeviceStream.__init__: a blocking PortAudio
        # stream must be started before Pa_WriteStream will accept data.
        stream.start()
        return RawPcmStream(stream, channels=channels)

    return _factory


def set_stream_factory(factory):
    """Test hook: inject a stream factory (callable -> stream object)."""
    global _stream_factory_override
    with _stream_lock:
        _stream_factory_override = factory


def _default_stream_factory():
    return SoundDeviceStream()


def get_stream_factory():
    with _stream_lock:
        return _stream_factory_override or _default_stream_factory


class AudioActor:
    """Single owner of output-stream playback, PCM buffering and the
    current-generation contract."""

    def __init__(self, stream_factory=None, ring_chunks=DEFAULT_RING_CHUNKS,
                 chunk_bytes=DEFAULT_CHUNK_BYTES):
        self._factory = stream_factory or get_stream_factory()
        self._max_chunks = max(2, int(ring_chunks))
        self._chunk_bytes = max(256, int(chunk_bytes))

        self._gen_lock = threading.Lock()
        self._generation = 0
        self._active_key = None
        self._play_state = "idle"  # idle | playing | aborted | ended

        self._ring_lock = threading.Lock()
        self._ring_cv = threading.Condition(self._ring_lock)
        self._ring = deque()
        self._utterance_pcm = bytearray()  # complete PCM for cursor resume
        self._spoken_bytes = 0
        self._eof = False
        self._dropped_chunks = 0
        self._last_error = None
        #: [PERF] P1-19 — one mark attempt per producer/consumer side. The turn
        #: keeps only the FIRST mark of a name, so a multi-sentence reply marks
        #: the true first byte of the turn even though each sentence starts its
        #: own utterance.
        self._fed_once = False
        self._played_once = False

        self._chunk_wait = threading.Event()
        self._stop = threading.Event()
        self._stream = None

    # ── lifecycle ──────────────────────────────────────────────────────
    def begin(self, key):
        """Start a new utterance: bump the generation, clear the ring and
        cursor. Old-generation chunks are discarded from here on. Returns
        the generation token every chunk must carry."""
        with self._gen_lock:
            self._generation += 1
            generation = self._generation
            self._active_key = key
            self._play_state = "playing"
        with self._ring_cv:
            self._ring.clear()
            self._utterance_pcm.clear()
            self._spoken_bytes = 0
            self._eof = False
            self._ring_cv.notify_all()
        self._chunk_wait.clear()
        self._stop.clear()
        # [PERF] P1-19 — a new utterance may mark its first byte again; the TURN
        # keeps only the first mark of a name, so a later sentence can never
        # rewrite the boundary the turn already recorded.
        self._fed_once = False
        self._played_once = False
        return generation

    def _is_current(self, key, generation):
        with self._gen_lock:
            return (
                generation == self._generation
                and (key is None or key == self._active_key)
                and self._play_state in ("playing", "ended")
            )

    def generation(self):
        with self._gen_lock:
            return self._generation

    def state(self):
        with self._gen_lock:
            return self._play_state

    def feed_chunk(self, key, generation, pcm_bytes, timeout=None):
        """Producers push immutable, generation-tagged PCM chunks here.

        Chunks from a stale generation are dropped at the boundary - they can
        never reach a newer answer's stream. When the ring is full the
        producer WAITS for space (backpressure) instead of dropping audio;
        ``FEED_TIMEOUT`` bounds that wait so a dead consumer cannot hang the
        producer forever. Returns True when the chunk was accepted.
        """
        if not pcm_bytes:
            return False
        if not self._is_current(key, generation):
            return False
        limit = FEED_TIMEOUT if timeout is None else float(timeout)
        deadline = time.monotonic() + limit
        with self._ring_cv:
            while len(self._ring) >= self._max_chunks:
                if not self._is_current(key, generation):
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # Never silently truncate: the caller is told the stream
                    # could not be kept intact.
                    self._dropped_chunks += 1
                    self._last_error = "ring full; chunk rejected after %ss" % limit
                    return False
                self._ring_cv.wait(min(0.25, remaining))
            self._ring.append(Chunk(key, generation, bytes(pcm_bytes)))
            self._utterance_pcm.extend(bytes(pcm_bytes))
            # [P0-06] Wake a play loop that is waiting for this chunk. Without
            # this notify the consumer could only find new audio on its next
            # timer tick, which is exactly the latency this item removes. Sent
            # under the lock, like every other notify on this condition.
            self._ring_cv.notify_all()
        self._chunk_wait.set()
        # [PERF] P1-19 — the first synthesized audio of this turn exists. The
        # gap between this mark and `playback_started` is synthesis/prefetch;
        # the gap before it is model + TTS time to the first byte.
        if not self._fed_once:
            self._fed_once = True
            try:
                _latency.mark_active("tts_first_byte", {"bytes": len(pcm_bytes)})
            except Exception:
                pass
        return True

    def end_utterance(self, key=None, generation=None):
        """Mark the current utterance's synthesis as complete (EOF).

        The play loop ends when the ring is drained and EOF is set, instead of
        polling forever. Ignored when the caller's generation is stale.
        """
        with self._gen_lock:
            if generation is not None and generation != self._generation:
                return False
            if key is not None and self._active_key is not None and key != self._active_key:
                return False
            if self._play_state == "playing":
                self._play_state = "ended"
        with self._ring_cv:
            self._eof = True
            self._ring_cv.notify_all()
        self._chunk_wait.set()
        return True

    def dropped_chunks(self):
        with self._ring_lock:
            return self._dropped_chunks

    def last_error(self):
        with self._ring_lock:
            return self._last_error

    def play(self, on_chunk=None, stream=None, stream_factory=None):
        """Blocking play loop: drain arriving prefetched chunks in order.

        Ends when the utterance is marked complete (:meth:`end_utterance`),
        when the generation changes, or when ``abort()`` is called. Returns
        the number of bytes actually written to the device.

        *stream_factory* is called HERE (on the playing thread) when no stream
        is supplied, so a caller with a specific format keeps its own device
        while the actor still owns the loop, the abort and the cursor.
        """
        if stream is not None:
            pass
        elif stream_factory is not None:
            stream = stream_factory()
        else:
            stream = self._factory()
        self._stream = stream
        written = 0
        # [PERF] P1-19 — allow one first-write mark for THIS play loop.
        self._played_once = False
        try:
            while True:
                if self._play_state not in ("playing", "ended"):
                    break
                with self._ring_cv:
                    chunk = self._ring.popleft() if self._ring else None
                    if chunk is not None:
                        self._ring_cv.notify_all()
                    eof = self._eof
                if chunk is None:
                    if eof:
                        break
                    # [P0-06] Subscribe to arriving synthesis through the SAME
                    # condition variable the producers notify, instead of
                    # sleeping on a different event in 250ms slices. The chunk
                    # that lands next now wakes this loop within a millisecond
                    # rather than on the next timer tick, which is what removes
                    # the per-utterance stall before the first device write.
                    #
                    # The wait is bounded by PLAY_WAIT_FLOOR_SECONDS only so a
                    # missed notify cannot park the loop. Nothing inside this
                    # block takes _gen_lock: _is_current() would, and mixing
                    # that with the ring lock is how a deadlock starts. The
                    # state reads below are plain attribute reads, exactly as
                    # the rest of this loop already does it.
                    with self._ring_cv:
                        while (not self._ring and not self._eof
                               and self._play_state in ("playing", "ended")):
                            self._ring_cv.wait(PLAY_WAIT_FLOOR_SECONDS)
                    # The abort fast path stays: _stop is checked after the
                    # wait, so a stop lands immediately instead of waiting for
                    # the floor to expire.
                    if self._stop.is_set():
                        break
                    if self._play_state not in ("playing", "ended"):
                        break
                    continue
                # Generation re-check immediately before the device write: a
                # chunk tagged with an older generation is discarded here even
                # if it somehow reached this ring.
                if not self._is_current(chunk.key, chunk.generation):
                    continue
                try:
                    stream.write(chunk.pcm)
                except Exception:
                    # A stop landing between the generation check above and
                    # this write leaves the stream stopped underneath us. That
                    # is an intentional abort, not a playback failure, so end
                    # the utterance quietly instead of reporting a PortAudio
                    # error for a stop the user asked for.
                    if self._play_state not in ("playing", "ended"):
                        break
                    raise
                written += len(chunk.pcm)
                with self._ring_lock:
                    # Consumed-frame accounting: the cursor is where the
                    # DEVICE was fed, never where synthesis was submitted.
                    self._spoken_bytes += len(chunk.pcm)
                # [PERF] P1-19 — the device really has audio now: this is the
                # moment the user can hear something, as opposed to bytes
                # merely existing in the ring.
                if not self._played_once:
                    self._played_once = True
                    try:
                        _latency.mark_active("playback_started")
                    except Exception:
                        pass
                if callable(on_chunk):
                    on_chunk(chunk.pcm)
        finally:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
            if self._stream is stream:
                self._stream = None
        return written

    def abort(self):
        """Stop/abort *this* stream and discard buffered PCM.

        Never calls ``sd.stop()`` - other audio (earcons, other owners) is
        untouched. Returns the byte position the utterance reached before
        the abort, i.e. the spoken cursor for an accurate resume.
        """
        with self._gen_lock:
            self._play_state = "aborted"
        self._stop.set()
        self._chunk_wait.set()
        stream = self._stream
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
        with self._ring_cv:
            self._ring.clear()
            cursor = self._spoken_bytes
            self._ring_cv.notify_all()
        return cursor

    # ── cursor / resume ────────────────────────────────────────────────
    def spoke_bytes(self):
        with self._ring_lock:
            return self._spoken_bytes

    def cursor(self):
        return self.spoke_bytes()

    def utterance_pcm(self):
        with self._ring_lock:
            return bytes(self._utterance_pcm)

    def resume_from(self, from_byte=None):
        """Re-enqueue the CURRENT utterance's unplayed tail.

        The resume point is the max of the caller's request and the CONSUMED
        cursor: bytes the device already played are never replayed, and bytes
        that were submitted but not yet played are not skipped. Returns the
        number of bytes re-enqueued (0 when nothing is left).
        """
        with self._ring_cv:
            pcm = bytes(self._utterance_pcm)
            consumed = self._spoken_bytes
            start = consumed if from_byte is None else max(int(from_byte), consumed)
            if start >= len(pcm):
                return 0
            self._ring.clear()
            self._eof = False
        with self._gen_lock:
            if self._play_state != "playing":
                self._play_state = "playing"
        self._stop.clear()
        generation = self.generation()
        active = self._active_key
        tail = pcm[start:]
        chunks = [tail[i:i + self._chunk_bytes]
                  for i in range(0, len(tail), self._chunk_bytes)]
        for chunk in chunks:
            self.feed_chunk(active, generation, chunk)
        return len(tail)


def audio_cache_key(model, reference_id, fmt, text):
    """Stable cache key that includes the TTS identity.

    [F32] "Key audio by model/reference/format/text" - the same sentence
    spoken by two references (or two models) must never share a cache slot.
    """
    return (str(model or ""), str(reference_id or ""), str(fmt or "pcm"),
            str(text or ""))


# Module-level singleton so fish_voice / voice / listener all talk to the
# SAME owner without threading a reference through call sites.
_actor = AudioActor()


def get_actor():
    return _actor


def actor_begin(key):
    return _actor.begin(key)


def actor_feed(key, generation, pcm_bytes, timeout=None):
    return _actor.feed_chunk(key, generation, pcm_bytes, timeout=timeout)


def actor_end(key=None, generation=None):
    return _actor.end_utterance(key=key, generation=generation)


def actor_abort():
    return _actor.abort()


def actor_spoke_bytes():
    return _actor.spoke_bytes()


def actor_play(on_chunk=None, stream=None, stream_factory=None):
    return _actor.play(on_chunk=on_chunk, stream=stream,
                       stream_factory=stream_factory)


def actor_is_current(key, generation):
    return _actor._is_current(key, generation)


