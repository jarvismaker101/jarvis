import io
import math
import os
import shutil
import tempfile
import threading
import time
import traceback

import requests
from pydub import AudioSegment
from pydub.playback import play

from backend.config import FISH_API_KEY, FISH_MODEL, FISH_REFERENCE_ID, FISH_VOLUME_BOOST_DB
# F32: one playback owner + identity-keyed audio; F33: every chunk that
# actually reaches the device feeds the AEC reference path.
from backend.services.audio_actor import (
    actor_abort,
    actor_begin,
    actor_end,
    actor_feed,
    actor_is_current,
    actor_is_paused,
    actor_play,
    audio_cache_key,
    make_sounddevice_factory,
)
from backend.services.echo_cancel import (
    feed_reference as _aec_feed_reference,
)


def _resolve_tts_model():
    """TTS model id per call: registry tts_model else env default FISH_MODEL.

    Registry read is per-call with safe degradation to the env constant.
    Provider field is ignored for the Fish HTTP call (always Fish Audio).
    """
    try:
        from backend.services import model_registry
        sel = model_registry.get_model_for_role("tts")
        m = str(sel.get("model") or "").strip()
        if m:
            return m
    except Exception:
        pass
    return FISH_MODEL

try:
    import av
except Exception:
    av = None

try:
    from pydub.playback import _play_with_simpleaudio
except Exception:
    _play_with_simpleaudio = None

for _bin, _pref in (("ffmpeg", "converter"), ("ffprobe", "ffprobe")):
    _found = shutil.which(_bin)
    if _found:
        setattr(AudioSegment, _pref, _found)


TTS_URL = "https://api.fish.audio/v1/tts"

# Reused across every request — skips the TCP + TLS handshake that a fresh
# `requests.post` pays on each TTS call (~100-300ms on Windows).
_session = requests.Session()

# Shared decoded-audio cache. Playback fetches through here so a prefetch
# thread can synthesise the next chunk while the current one plays.
_audio_cache = {}
_in_flight = {}
_cache_lock = threading.Lock()
_CACHE_MAX = 32

#: [P0-05] How long a JOINER waits for progress from a synthesis it joined.
#: The owner's own HTTP read is bounded separately; this only stops a joiner
#: from streaming forever off a synthesis that has genuinely hung.
_INFLIGHT_JOIN_TIMEOUT_SECONDS = 60.0

#: [P0-05] Bound on the in-flight map. Owners pop their own entry, so this only
#: bites when an owner dies; completed entries are evicted oldest-first, the
#: same rule _CACHE_MAX applies to the decoded-audio cache.
_INFLIGHT_MAX = 8

_playback_lock = threading.Lock()
_current_playback = None


class _InflightPCM:
    """One synthesis in flight, joinable by later callers as bytes arrive.

    [P0-05] An in-flight entry used to be all-or-nothing. If a prefetch thread
    claimed the entry first, the playback thread became a mere waiter: it could
    not start until synthesis had FINISHED, then replayed the finished bytes.
    Sentence 1 therefore lost streaming playback whenever the prefetch won the
    race — and the prefetch usually won, because the playback worker first
    reads the model registry and starts an earcon thread.

    The owner now publishes every chunk here as it reads it, and a joiner plays
    them AS THEY ARRIVE: the first bytes are worth as much to the joiner as
    they are to the owner, so nothing is gained by making it wait for the last
    one.

    ``wait()`` keeps the old Event-shaped contract for callers that only want
    to know when the synthesis finished, so an entry for this path and an entry
    for the MP3 path stay interchangeable.
    """

    #: A joiner re-checks at most this often, so a missed notify can never park
    #: playback. Every publish/finish also notifies, so this is a safety floor.
    WAIT_SLICE = 0.05

    def __init__(self):
        self._cv = threading.Condition()
        self._buffer = bytearray()
        self._done = False
        #: True while the owner is ITSELF feeding the device. A second
        #: ``play=True`` caller must not stream alongside it: two players on
        #: one sentence would double-speak it, so that case keeps the old
        #: wait-then-replay behaviour.
        self.playing = False

    # ── owner side ─────────────────────────────────────────────────────────
    def publish(self, chunk):
        """Add freshly synthesised bytes and wake every joiner."""
        if not chunk:
            return
        with self._cv:
            self._buffer.extend(chunk)
            self._cv.notify_all()

    def finish(self):
        """Mark the synthesis over. Idempotent, and safe on every exit path."""
        with self._cv:
            self._done = True
            self._cv.notify_all()

    # ── joiner side ────────────────────────────────────────────────────────
    def iter_from(self, offset=0, timeout=None):
        """Yield the buffer's bytes as they arrive, in synthesis order.

        Ends when the owner finishes, or when *timeout* elapses with no further
        bytes — a hung synthesis must not hang its joiner.
        """
        position = max(0, int(offset))
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            with self._cv:
                while position >= len(self._buffer) and not self._done:
                    if deadline is None:
                        self._cv.wait(self.WAIT_SLICE)
                        continue
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return
                    self._cv.wait(min(self.WAIT_SLICE, remaining))
                if position >= len(self._buffer):
                    return
                chunk = bytes(self._buffer[position:])
                position = len(self._buffer)
            if chunk:
                yield chunk

    def snapshot(self):
        with self._cv:
            return bytes(self._buffer)

    def is_done(self):
        with self._cv:
            return self._done

    def wait(self, timeout=None):
        """Event-compatible completion wait; True when the synthesis ended."""
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._cv:
            while not self._done:
                if deadline is None:
                    self._cv.wait(self.WAIT_SLICE)
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(min(self.WAIT_SLICE, remaining))
            return True


def _claim_inflight(ck):
    """Claim ownership of *ck*'s synthesis, or hand back the live entry.

    Returns ``(entry, is_owner)``. De-duplication is unchanged: two callers
    asking for the same text still share exactly ONE synthesis, and the loser
    now joins the winner's stream instead of replaying its finished output.
    """
    with _cache_lock:
        entry = _in_flight.get(ck)
        if entry is not None:
            return entry, False
        entry = _InflightPCM()
        _in_flight[ck] = entry
        if len(_in_flight) > _INFLIGHT_MAX:
            for old_key in list(_in_flight):
                if old_key == ck:
                    continue
                candidate = _in_flight.get(old_key)
                if isinstance(candidate, _InflightPCM) and candidate.is_done():
                    _in_flight.pop(old_key, None)
                    if len(_in_flight) <= _INFLIGHT_MAX:
                        break
    return entry, True


def _pcm_cache_key(text):
    """[F32] Cache identity = TTS model + reference + format + text.

    The same sentence spoken by two references (or two models) must never
    share a cache slot — pre-G10 the PCM cache was keyed by text only.
    """
    return audio_cache_key(
        _resolve_tts_model(), FISH_REFERENCE_ID, "pcm", str(text or ""))


def _looks_like_mp3(content):
    if content[:3] == b"ID3":
        return True
    return len(content) >= 2 and content[0] == 0xFF and (content[1] & 0xE0) == 0xE0


def _decode_audio(content):
    """Decode MP3 bytes via bundled PyAV first (robust), falling back to pydub+system ffmpeg."""
    if av is not None:
        try:
            container = av.open(io.BytesIO(content), mode="r", format="mp3")
            try:
                audio_stream = None
                for stream in container.streams:
                    if stream.type == "audio":
                        audio_stream = stream
                        break
                if audio_stream is None:
                    raise ValueError("no audio stream")

                resampler = av.AudioResampler(format="s16", layout="mono", rate=44100)
                chunks = []
                for frame in container.decode(audio_stream):
                    for resampled in resampler.resample(frame):
                        chunks.append(resampled.to_ndarray())

                if not chunks:
                    raise ValueError("no decoded audio")

                import numpy as np

                data = np.concatenate(chunks, axis=1)
                raw = data.tobytes()
                return AudioSegment(
                    data=raw,
                    sample_width=2,
                    frame_rate=44100,
                    channels=1,
                )
            finally:
                container.close()
        except Exception as exc:
            print("[FISH] PyAV decode failed, falling back to ffmpeg:", exc)
    return None


def _boost_volume(audio):
    if FISH_VOLUME_BOOST_DB <= 0:
        return audio

    peak = audio.max
    amp = audio.max_possible_amplitude
    if peak > 0 and amp > 0:
        gain_db = min(FISH_VOLUME_BOOST_DB, 20 * math.log10(0.9 * amp / peak))
    else:
        gain_db = FISH_VOLUME_BOOST_DB

    return audio.apply_gain(gain_db) if gain_db > 0 else audio


def _request_audio(text):
    tts_model = _resolve_tts_model()
    headers = {
        "Authorization": f"Bearer {FISH_API_KEY}",
        "Content-Type": "application/json",
        "model": tts_model,
    }
    payload = {
        "text": text,
        "model": tts_model,
        "format": "mp3",
    }
    if FISH_REFERENCE_ID:
        payload["reference_id"] = FISH_REFERENCE_ID

    response = _session.post(TTS_URL, json=payload, headers=headers, timeout=(5.05, 90))
    if response.status_code != 200:
        print("[FISH] Error:", response.status_code, response.text[:300])
        return None
    if not response.content or not _looks_like_mp3(response.content):
        print("[FISH] Response was not audio:", response.headers.get("Content-Type"), "len", len(response.content), repr(response.content[:60]))
        return None
    return response.content


def _request_audio_playable(text):
    """Fetch MP3 bytes, decode to a mono/44.1kHz AudioSegment and boost volume.

    Returns None on any failure (already has one retry like the old code).
    """
    content = _request_audio(text)
    if content is None:
        print("[FISH] Retrying once...")
        content = _request_audio(text)
    if content is None:
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as file:
        file.write(content)
        path = file.name
    try:
        audio = _decode_audio(content)
        if audio is None:
            audio = AudioSegment.from_file(path, format="mp3")
        return _boost_volume(audio)
    except Exception:
        print("[FISH] Decode failed:")
        traceback.print_exc()
        return None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _fetch_audio(text):
    """Fetch + decode + boost, shared between prefetch and playback threads.

    Cached by IDENTITY (model + reference + format + text) and deduplicated,
    so a prefetching thread and the playback thread never issue a duplicate
    TTS request for the same sentence. If the fetch fails, the failure is
    cached too — the playback thread then falls through to the local-TTS
    ladder immediately instead of re-waiting on the network timeout.
    """
    # [F32] Identity-consistent caching: the decoded form is keyed by
    # model + reference + 'mp3' + text, the same identity rule the PCM
    # path uses. Two references (or two models) never share decoded audio.
    ck = audio_cache_key(
        _resolve_tts_model(), FISH_REFERENCE_ID, "mp3", str(text or ""))
    while True:
        with _cache_lock:
            if ck in _audio_cache:
                return _audio_cache[ck]
            done = _in_flight.get(ck)
            if done is None:
                done = threading.Event()
                _in_flight[ck] = done
                break
        if not done.wait(timeout=60):
            # The other thread stalled long enough — take the fetch over.
            # (F32: this used to pop the undefined name `ck`... which was
            # the same key it should have used all along.)
            with _cache_lock:
                _in_flight.pop(ck, None)
                continue
        with _cache_lock:
            return _audio_cache.get(ck)

    try:
        audio = _request_audio_playable(text)
        with _cache_lock:
            _audio_cache[ck] = audio
            if len(_audio_cache) > _CACHE_MAX:
                _audio_cache.pop(next(iter(_audio_cache)))
        return audio
    finally:
        with _cache_lock:
            _in_flight.pop(ck, None)
        done.set()


# S23 — the PCM boost is a *smoothed limiter*, not a per-chunk normaliser.
# The old gain was recomputed from every 46 ms chunk's own peak, so quiet
# fragments (breaths, word endings) got the full boost while loud vowels got
# almost none: the voice's dynamics flattened and the gain stepped at every
# chunk boundary, which is audible as pumping. The applied gain now attacks
# quickly (so a transient can never clip) and releases over a few hundred ms
# (so it moves like a limiter instead of jittering).
_PCM_ATTACK_PER_CHUNK = 0.5
_PCM_RELEASE_PER_CHUNK = 0.15

# The smoothed gain is per-stream state. It lives in thread-local storage so
# `_boost_pcm_chunk` keeps its single-argument seam (callers/tests patch it as
# ``_boost_pcm_chunk(chunk)``) while still smoothing across the chunks of one
# stream. Each capture stream is consumed by one thread, and
# `_pcm_chunks_from_response` resets the state when it starts.
_PCM_GAIN_STATE = threading.local()


def _reset_pcm_gain_state():
    _PCM_GAIN_STATE.gain = None


def _boost_pcm_chunk(chunk_bytes):
    """Apply FISH_VOLUME_BOOST_DB to a PCM s16le chunk (mono, 44100).

    The gain is smoothed across the chunks of the current stream (thread-local
    state reset by `_pcm_chunks_from_response`); a per-chunk cap guarantees the
    chunk itself can never clip.
    """
    if FISH_VOLUME_BOOST_DB <= 0 or not chunk_bytes:
        return chunk_bytes
    try:
        import numpy as np

        arr = np.frombuffer(chunk_bytes, dtype=np.int16)
        if arr.size == 0:
            return chunk_bytes
        peak = int(np.max(np.abs(arr)))
        if peak == 0:
            return chunk_bytes
        amp = 32767
        # Gain that leaves 10% headroom for THIS chunk's peak; never attenuate
        # (the old behaviour only ever boosted, and attenuation would be a
        # separate loudness decision).
        headroom_db = 20 * math.log10(0.9 * amp / peak)
        target_db = min(FISH_VOLUME_BOOST_DB, headroom_db)
        target = 10 ** (max(0.0, target_db) / 20.0)
        prev = getattr(_PCM_GAIN_STATE, "gain", None)
        if prev is None:
            gain = target
        elif target < prev:
            gain = prev + _PCM_ATTACK_PER_CHUNK * (target - prev)
        else:
            gain = prev + _PCM_RELEASE_PER_CHUNK * (target - prev)
        _PCM_GAIN_STATE.gain = gain
        # Limiter safety: cap the applied gain to the current chunk's headroom
        # so a sudden transient cannot wrap/clip even mid-release.
        gain = min(gain, amp / peak * 0.98)
        if gain <= 1.0:
            return chunk_bytes
        boosted = np.clip(arr.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
        return boosted.tobytes()
    except Exception:
        return chunk_bytes


def _aec_chunk_for(rate, channels):
    """Build the AEC reference hook for a playback format.

    [F32/F33] Every chunk that reaches the DEVICE feeds the reference path,
    tagged with the format it was rendered in, so the listener-side alignment
    measures real time instead of assuming a rate.
    """
    def _hook(chunk):
        try:
            _aec_feed_reference(chunk, rate, 2, channels=channels)
        except Exception:
            pass

    return _hook


def _aec_chunk(chunk):
    """F32/F33: every chunk that reaches the DEVICE feeds the AEC reference."""
    try:
        _aec_feed_reference(chunk, 44100, 2, channels=1)
    except Exception:
        pass


def _stream_key(ck):
    return ck if ck is not None else "pcm"


def _player_thread(key, done, written, on_chunk=None, stream_factory=None):
    """Run the actor's play loop on its own thread (the single owner)."""
    chunk_hook = on_chunk or _aec_chunk

    def _loop():
        try:
            written["bytes"] = actor_play(on_chunk=chunk_hook,
                                          stream_factory=stream_factory)
        except Exception as exc:  # pragma: no cover - hardware path
            written["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_loop, name="audio-actor-play", daemon=True)
    thread.start()
    return thread


def _drain_seconds(byte_count, rate=44100, channels=1):
    """A bounded wait for draining *byte_count* bytes of s16 PCM.

    Audio duration plus a margin, with a small floor: this is only a worst-case
    guard, because the actor ends the wait as soon as the utterance is marked
    complete or aborted.
    """
    return max(5.0, (byte_count / float(rate * channels * 2)) + 5.0)


def _play_pcm_through_actor(pcm_bytes, handle, ck=None, stream_factory=None,
                            rate=44100, channels=1):
    """F32: play fully-synthesised PCM through the ONE owner.

    The producer only SUBMITS chunks; ``AudioActor.play`` writes them and
    accounts the consumed frames. Returns True when the audio was played (or
    the user stopped it), False when the ring could not keep up and audio
    would have been lost — or when the output device refused to open.
    """
    if not pcm_bytes:
        return False
    key = _stream_key(ck)
    generation = actor_begin(key)
    done = threading.Event()
    written = {}
    _player_thread(key, done, written,
                   on_chunk=_aec_chunk_for(rate, channels),
                   stream_factory=stream_factory)
    failed = False
    try:
        chunk_size = 4096
        for index in range(0, len(pcm_bytes), chunk_size):
            if handle is not None and handle.stopped:
                actor_abort()
                break
            if written.get("error") is not None:
                # The device refused to open (or died): stop feeding at once
                # instead of filling the ring behind a dead consumer.
                actor_abort()
                failed = True
                break
            chunk = pcm_bytes[index:index + chunk_size]
            if not chunk:
                continue
            if not actor_feed(key, generation, chunk):
                if not actor_is_current(key, generation):
                    break  # a newer utterance or a stop took over
                # Backpressure timed out: report honestly instead of
                # pretending the whole sentence was spoken.
                actor_abort()
                failed = True
                break
        else:
            actor_end(key, generation)
    finally:
        if not failed and actor_is_current(key, generation):
            actor_end(key, generation)
    # [P1-07] A pause parks the play loop WITHOUT ending the utterance, so the
    # drain timeout has to be suspended while the actor is paused. Otherwise a
    # pause longer than the drain budget would look exactly like a finished
    # sentence, the caller would start the next one, and two play loops would
    # write to the one device.
    deadline = time.monotonic() + _drain_seconds(len(pcm_bytes), rate, channels)
    while not done.wait(timeout=0.25):
        if time.monotonic() > deadline and not actor_is_paused():
            break
    if written.get("error") is not None:
        raise written["error"]
    return not failed


def _replay_cached_pcm(cached_bytes, ck=None):
    """Replay cached PCM (44100 mono s16) through the one playback owner."""
    if not cached_bytes:
        return False
    handle = _register_sounddevice_playback()
    try:
        if handle.stopped:
            return True
        return _play_pcm_through_actor(bytes(cached_bytes), handle, ck=ck)
    finally:
        _clear_sounddevice_playback(handle)


def _pcm_chunks_from_response(response):
    """Yield aligned, boosted PCM chunks from a streaming TTS response.

    FIX1: odd-byte splits across HTTP chunks are stitched back together so no
    sample is ever half-written to the device.
    """
    residual = b""
    _reset_pcm_gain_state()
    for chunk in response.iter_content(chunk_size=4096):
        if not chunk:
            continue
        chunk = residual + chunk
        aligned_len = (len(chunk) // 2) * 2
        if aligned_len < len(chunk):
            residual = chunk[aligned_len:]
            chunk = chunk[:aligned_len]
        else:
            residual = b""
        if not chunk:
            continue
        boosted = _boost_pcm_chunk(chunk)
        if boosted:
            yield boosted
    if residual:
        # Flush the final odd byte padded to a full sample: never drop audio.
        boosted = _boost_pcm_chunk(residual + b"\x00")
        if boosted:
            yield boosted


def _publishing_chunks(response, inflight):
    """Yield the response's PCM chunks, publishing each one to *inflight*.

    [P0-05] The publish happens BEFORE the chunk is handed to the actor feed,
    so a joiner's copy is never behind the owner's: whatever the owner is about
    to play, the joiner can already play too.
    """
    for chunk in _pcm_chunks_from_response(response):
        inflight.publish(chunk)
        yield chunk


def _stream_pcm_to_actor(key, chunks, handle, out=None):
    """Feed *chunks* to the ONE playback owner as they arrive.

    Returns ``(full, failed)``. Shared by the synthesising owner and by a
    P0-05 joiner streaming off an in-flight synthesis, so both get exactly the
    same stop/abort/drain contract — a joiner that behaved differently would be
    a second playback implementation waiting to drift.

    *out* receives ``{"generation": ...}`` as soon as this call opens its own
    utterance, so a caller can tell "audio already reached the device" from
    "nothing was ever played" even when this raises.
    """
    generation = actor_begin(key)
    if out is not None:
        out["generation"] = generation
    done = threading.Event()
    written = {}
    _player_thread(key, done, written)
    full = bytearray()
    failed = False
    read_error = None
    try:
        for chunk in chunks:
            if handle is not None and handle.stopped:
                actor_abort()
                break
            full.extend(chunk)
            if not actor_feed(key, generation, chunk):
                if not actor_is_current(key, generation):
                    break
                actor_abort()
                failed = True
                break
        else:
            actor_end(key, generation)
    except BaseException as exc:
        read_error = exc
    finally:
        if not failed and actor_is_current(key, generation):
            actor_end(key, generation)
        done.wait(timeout=_drain_seconds(len(full)))
    if written.get("error") is not None:
        raise written["error"]
    if read_error is not None:
        raise read_error
    return full, failed


def _play_inflight_stream(inflight, ck, handle):
    """[P0-05] Play a synthesis that is already in flight, as it arrives.

    The owner of this synthesis is a prefetch, so it is not feeding the device:
    this caller becomes the player and starts at the FIRST chunk instead of
    after the last one.
    """
    key = _stream_key(ck)
    opened = {}
    full, failed = _stream_pcm_to_actor(
        key,
        inflight.iter_from(0, timeout=_INFLIGHT_JOIN_TIMEOUT_SECONDS),
        handle,
        out=opened)
    if not full:
        return False
    return not failed


def _do_pcm_stream(text, play=True):
    """Core streaming: fetch PCM chunks, optionally play, always cache full.

    F32: there is exactly ONE playback owner — ``AudioActor``. This function
    no longer creates or writes an ``OutputStream`` of its own; it submits
    immutable generation-tagged chunks and the actor's play loop writes them
    (and accounts the frames it consumed).

    Cancellation is registered BEFORE any cache/in-flight wait, so a stop that
    arrives during a prefetch wait still aborts this playback instead of
    registering a handle only after it was already cancelled.
    """
    if not FISH_API_KEY or not text or not text.strip():
        return False
    # [F32] Identity-keyed cache/in-flight (model + reference + format + text).
    ck = _pcm_cache_key(text)
    key = _stream_key(ck)

    cached = None
    with _cache_lock:
        entry = _audio_cache.get(ck)
        if isinstance(entry, (bytes, bytearray)) and entry:
            cached = bytes(entry)

    # [F32] Early cancellation registration: the stoppable handle exists
    # before the first wait, so a stop during prefetch/read/write is honoured.
    handle = _register_sounddevice_playback() if play else None
    if handle is not None and handle.stopped:
        _clear_sounddevice_playback(handle)
        return True

    is_owner = False
    #: [P0-05] Filled in by _stream_pcm_to_actor the moment this call opens its
    #: own utterance, so the error path below can still tell "audio already
    #: reached the device" from "nothing was ever played".
    opened = {}
    inflight = None
    try:
        if not play:
            if cached is not None:
                return True
        elif cached is not None:
            return _play_pcm_through_actor(cached, handle, ck=ck)

        # ── in-flight deduplication, keyed by the identity key ──
        inflight, is_owner = _claim_inflight(ck)
        if is_owner and play:
            # [P0-05] Announce ownership of the DEVICE before the network call,
            # not after it: `playing` is what stops a second play caller from
            # streaming the same sentence alongside this one, so it must be
            # true for the whole life of a playing owner — including the window
            # while this thread is still waiting for the TTS response.
            inflight.playing = True
        if not is_owner:
            # [P0-05] A prefetch is already synthesising this sentence and is
            # NOT feeding the device, so join its stream and start at the first
            # chunk. This is what makes losing the race cost nothing: the
            # player no longer waits for the whole sentence to be synthesised.
            if (play and isinstance(inflight, _InflightPCM)
                    and not inflight.playing):
                return _play_inflight_stream(inflight, ck, handle)
            # Otherwise keep the documented behaviour: wait for the owner (a
            # hanging synthesis still has to time out), then play the shared
            # result. A second play=True caller must NOT stream alongside a
            # playing owner — that would speak the sentence twice.
            if not inflight.wait(timeout=60):
                return False
            with _cache_lock:
                entry = _audio_cache.get(ck)
                cached = bytes(entry) if isinstance(entry, (bytes, bytearray)) and entry else None
            if cached is None:
                return False
            if not play:
                return True
            return _play_pcm_through_actor(cached, handle, ck=ck)

        tts_model = _resolve_tts_model()
        headers = {
            "Authorization": f"Bearer {FISH_API_KEY}",
            "Content-Type": "application/json",
            "model": tts_model,
        }
        payload = {
            "text": text,
            "model": tts_model,
            "format": "pcm",
            "latency": "balanced",
            "sample_rate": 44100,
        }
        if FISH_REFERENCE_ID:
            payload["reference_id"] = FISH_REFERENCE_ID

        response = _session.post(TTS_URL, json=payload, headers=headers,
                                 timeout=(5.05, 90), stream=True)
        try:
            if response.status_code != 200:
                return False
            if not play:
                # Prefetch: publish each chunk as it arrives so a joiner can
                # play immediately, then cache the whole sentence. The publish
                # must happen even though nothing plays here: this thread is
                # the one holding the synthesis the joiner is waiting on.
                full = bytearray()
                try:
                    for chunk in _pcm_chunks_from_response(response):
                        full.extend(chunk)
                        inflight.publish(chunk)
                finally:
                    inflight.finish()
                if not full:
                    return False
                _cache_pcm(ck, bytes(full))
                return True

            # [F32] ONE owner: the actor plays while this thread only feeds.
            full, failed = _stream_pcm_to_actor(
                key, _publishing_chunks(response, inflight), handle, out=opened)
            if not full:
                return False
            if failed:
                return False
            _cache_pcm(ck, bytes(full))
            return True
        finally:
            try:
                response.close()
            except Exception:
                pass
    except Exception as exc:
        # FIX3/F32: audio already reached the device — never report failure
        # (and never fall back and replay the sentence from the beginning).
        # `generation` is only set once THIS call opened its own utterance, so
        # a stale cursor from an earlier sentence cannot be mistaken for it.
        generation = opened.get("generation")
        if play and generation is not None and _actor_heard_audio():
            print(f"[FISH] pcm stream error after partial playback: {exc}")
            return True
        print(f"[FISH] pcm stream error: {exc}")
        traceback.print_exc()
        return False
    finally:
        if handle is not None:
            _clear_sounddevice_playback(handle)
        popped = None
        with _cache_lock:
            if is_owner:
                # [F32] the in-flight entry is keyed by the IDENTITY key; the
                # old code popped by text and leaked the entry forever.
                popped = _in_flight.pop(ck, None)
        if popped is not None:
            # [P0-05] Every exit path must release a joiner: an entry left
            # unfinished would hang the next play=True caller for the full
            # timeout even though the synthesis is over (or never started).
            if isinstance(popped, _InflightPCM):
                popped.finish()
            else:
                popped.set()


def _actor_heard_audio():
    """True when the playback owner actually wrote audio to the device.

    [F32] consumed-frame accounting is the only trustworthy answer to "did the
    user already hear part of this?" — the old code guessed from a local
    ``wrote_any`` flag that counted PRODUCED chunks, not played ones.
    """
    try:
        from backend.services import audio_actor
        return audio_actor.get_actor().spoke_bytes() > 0
    except Exception:
        return False


def _cache_pcm(ck, pcm_bytes):
    """Store utterance PCM under its identity key with bounded eviction."""
    with _cache_lock:
        _audio_cache[ck] = bytes(pcm_bytes)
        while len(_audio_cache) > _CACHE_MAX:
            oldest = next(iter(_audio_cache))
            if oldest == ck:
                break
            _audio_cache.pop(oldest, None)


def prefetch_fish_audio(text):
    """Synthesise *text* in the background so playback never waits on TTS.

    The decoded audio lands in the shared cache; when the playback loop
    reaches this sentence, `speak_fish_audio` finds it ready and starts
    instantly. Surfaces as no-op if Fish is unavailable.
    Uses streaming PCM path (collect-to-cache, no playback) with fallback to legacy.
    """
    if not FISH_API_KEY or not text or not text.strip():
        return

    def _prefetch():
        # Try streaming collect-to-cache first
        if _do_pcm_stream(text, play=False):
            return
        # Fallback to legacy MP3 — FIX5: cache raw PCM bytes so _do_pcm_stream
        # can replay it. [F32] stored under the IDENTITY key, the same slot the
        # play path looks in (it used the bare text before, so the replay never
        # found the prefetched bytes and re-synthesised the sentence).
        audio = _fetch_audio(text)
        if audio is not None:
            try:
                pcm_bytes = audio.set_frame_rate(44100).set_channels(1).raw_data
                pcm_bytes = bytes(pcm_bytes)
                ck = _pcm_cache_key(text)
                with _cache_lock:
                    cached = _audio_cache.get(ck)
                    already = isinstance(cached, (bytes, bytearray))
                if not already:
                    # _cache_pcm takes the cache lock itself: never call it
                    # while already holding that (non-reentrant) lock.
                    _cache_pcm(ck, pcm_bytes)
            except Exception:
                pass

    threading.Thread(target=_prefetch, daemon=True).start()


def warm_up_fish_tts():
    """Prime the Fish model and HTTP session so the first real reply is fast.

    The first TTS request of a run pays noticeably slower synthesis. Calling
    this at boot (in a background thread) makes that warm-up happen early;
    the tiny warm-up audio is merely cached, never played.
    """
    try:
        _fetch_audio("Warm up, sir.")
    except Exception:
        pass


TTS_OUTPUT_DEVICE_ENV = "JARVIS_TTS_OUTPUT_DEVICE"

# Cached device resolution — queried once per process, not per chunk
_cached_device = None
_cached_device_valid = False
_cached_device_lock = threading.Lock()


def _get_cached_device():
    global _cached_device, _cached_device_valid
    with _cached_device_lock:
        if _cached_device_valid:
            return _cached_device
        dev = _resolve_output_device()
        _cached_device = dev
        _cached_device_valid = True
        return dev


def _clear_device_cache():
    global _cached_device, _cached_device_valid
    with _cached_device_lock:
        _cached_device_valid = False
        _cached_device = None


def _resolve_output_device():
    """Resolve the TTS render device from JARVIS_TTS_OUTPUT_DEVICE.

    * unset / empty / default|system|auto  -> None (Windows default output)
    * numeric                              -> that exact sounddevice index,
                                              validated as an output device
    * anything else                        -> case-insensitive substring match
                                              against output-device names;
                                              clean 48k stereo endpoints score
                                              highest among matches

    Never raises and never pins broken hardware: when nothing usable matches
    we log loudly and fall back to the system default. The previous version
    hard-pinned any endpoint whose name contained "oneplus", which routed
    every reply into a possibly-silent Bluetooth profile with no way out —
    explicit routing is now opt-in via the env var.
    """
    wanted = (os.getenv(TTS_OUTPUT_DEVICE_ENV) or "").strip()
    if not wanted or wanted.lower() in ("default", "system", "auto"):
        return None

    try:
        import sounddevice as sd

        devs = sd.query_devices()
    except Exception as e:
        print(f"[FISH] device resolution failed ({e}) - using system default")
        return None

    if wanted.isdigit():
        idx = int(wanted)
        if 0 <= idx < len(devs) and devs[idx].get("max_output_channels", 0) >= 1:
            d = devs[idx]
            print(
                f"[FISH] output device pinned by index: idx={idx} "
                f"'{d.get('name')}' ch={d.get('max_output_channels')} "
                f"sr={d.get('default_samplerate')}"
            )
            return idx
        print(
            f"[FISH] {TTS_OUTPUT_DEVICE_ENV}={wanted} is not an output-capable "
            "sounddevice index - using system default"
        )
        return None

    needle = wanted.lower()
    candidates = []
    for i, d in enumerate(devs):
        ch = d.get("max_output_channels", 0)
        if ch < 2:
            continue
        name = d.get("name", "")
        if needle not in name.lower():
            continue
        sr = int(d.get("default_samplerate") or 0)
        score = 0
        if sr == 48000 and ch == 2:
            score = 3
        elif sr == 48000:
            score = 2
        elif ch == 2:
            score = 1
        candidates.append((score, i, name))
    if not candidates:
        print(
            f"[FISH] no output device name containing '{wanted}' - "
            "using system default"
        )
        return None
    candidates.sort(key=lambda c: (-c[0], c[1]))
    _, idx, name = candidates[0]
    print(f"[FISH] output device matched by name '{wanted}': idx={idx} '{name}'")
    return idx


class _SoundDevicePlayback:
    """Stoppable handle so stop_speaking()/stop_fish_audio() can cut playback.

    [F32] "Stop" means aborting THIS utterance through the one playback owner:
    the actor stops and discards its own stream and ring. It deliberately does
    NOT call ``sd.stop()``, which kills every PortAudio stream in the process
    (earcons, other engines) and is therefore not a reliable single-utterance
    abort.
    """

    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True
        try:
            actor_abort()
        except Exception:
            pass


def _register_sounddevice_playback():
    global _current_playback

    handle = _SoundDevicePlayback()
    with _playback_lock:
        _current_playback = handle
    return handle


def _clear_sounddevice_playback(handle):
    global _current_playback

    with _playback_lock:
        if _current_playback is handle:
            _current_playback = None


def _play_via_sounddevice(audio):
    """Play AudioSegment via sounddevice, resampled to 48k stereo.

    Route selection (first attempt wins):
      1. JARVIS_TTS_OUTPUT_DEVICE when set (index or name-substring pin)
      2. Windows default render device

    WASAPI output devices on Windows (OnePlus/Realtek) reject 44.1kHz, so we
    upsample to 48kHz stereo first. If a pinned endpoint cannot be opened we
    retry once on the system default before giving up and letting the caller
    fall through to simpleaudio/ffplay.

    Near-silent decoded audio is rejected outright so a dead Fish response
    falls through to the next engine instead of wasting playback time.

    Bluetooth note: keeping a headset mic open can flip buds into the
    Hands-Free profile where the A2DP sink renders nothing audible even
    though the mixer consumes the stream. If logs say "playback done" but
    you hear silence, switch JARVIS_MIC_NAME off the headset or pin
    JARVIS_TTS_OUTPUT_DEVICE to speakers (see check_tts_devices script).
    """
    # Register BEFORE anything else (even the lazy imports): a stop arriving
    # during setup must short-circuit instead of racing past an unregistered
    # handle. Exactly one handle per call; `finally` always unregisters it.
    handle = _register_sounddevice_playback()
    try:
        if handle.stopped:
            print("[FISH] sounddevice playback interrupted")
            return True

        peak = audio.max
        if peak < 40:
            print(f"[FISH] decoded audio near-silent (peak={peak}) - skipping playback")
            return False

        dev_index = _get_cached_device()

        target = audio.set_frame_rate(48000)
        if target.channels == 1:
            target = target.set_channels(2)
        # F32: raw s16le bytes for the ONE playback owner — no local stream
        # and no sd.play() (which drives a global stream nobody owns).
        pcm = target.raw_data

        attempts = [dev_index, None] if dev_index is not None else [None]

        for attempt in attempts:
            if handle.stopped:
                print("[FISH] sounddevice playback interrupted")
                return True
            label = f"idx={attempt}" if attempt is not None else "system default"
            try:
                print(f"[FISH] sounddevice playing {len(audio)}ms to {label}")
                ok = _play_pcm_through_actor(
                    pcm, handle, ck=("decoded", 48000, 2),
                    stream_factory=make_sounddevice_factory(
                        device=attempt, samplerate=48000, channels=2),
                    rate=48000, channels=2)
            except Exception as exc:
                print(f"[FISH] could not open {label}: {exc}")
                if attempt is not None and attempt == dev_index:
                    _clear_device_cache()
                continue
            if handle.stopped:
                print("[FISH] sounddevice playback interrupted")
            else:
                print("[FISH] sounddevice 48k playback done")
            return ok

        return False
    except Exception as e:
        print(f"[FISH] sounddevice error: {e}")
        traceback.print_exc()
        return False
    finally:
        _clear_sounddevice_playback(handle)


def speak_fish_audio(text, before_playback=None):
    global _current_playback

    try:
        if not FISH_API_KEY:
            print("[FISH] Missing FISH_API_KEY")
            return False

        # Streaming PCM path: incremental playback starts at first chunk (~1s earlier),
        # no decode, no temp files. Falls back to whole-MP3 on failure.
        try:
            if _do_pcm_stream(text, play=True):
                return True
        except Exception as exc:
            print(f"[FISH] pcm stream failed, falling back: {exc}")

        audio = _fetch_audio(text)
        if audio is None:
            return False

        if callable(before_playback):
            before_playback()

        # 1) sounddevice (48k stereo to default device — confirmed audible on OnePlus earbuds)
        if _play_via_sounddevice(audio):
            return True

        # 2) simpleaudio (WASAPI) fallback — note: rejects 44.1kHz, so it is NOT preferred
        if _play_with_simpleaudio:
            try:
                print(f"[FISH] simpleaudio playing {len(audio)}ms to default device")
                playback = _play_with_simpleaudio(audio)
                with _playback_lock:
                    _current_playback = playback
                while playback.is_playing():
                    time.sleep(0.05)
                with _playback_lock:
                    if _current_playback is playback:
                        _current_playback = None
                print("[FISH] simpleaudio done")
                return True
            except Exception as e:
                print(f"[FISH] simpleaudio failed: {e}")

        # 3) pydub ffplay fallback
        print(f"[FISH] falling back to pydub ffplay {len(audio)}ms")
        play(audio)
        print("[FISH] ffplay done")

        return True
    except Exception:
        print("[FISH] Exception:")
        traceback.print_exc()
        return False


def stop_fish_audio(abort_actor=True):
    global _current_playback
    # [F32] abort the single playback owner: stops THIS stream, drops its
    # ring and discards stale-generation PCM (never sd.stop() for everyone).
    #
    # [P0-07] Both this and ``stop_google_tts`` drive the SAME actor, so
    # ``voice.stop_speaking`` passes abort_actor=False for one of them rather
    # than cutting the actor twice on every stop.
    if abort_actor:
        actor_abort()
    with _playback_lock:
        playback = _current_playback

    if playback:
        try:
            playback.stop()
        except Exception:
            pass

    # Drop the handle so already-buffered PCM is never resumed and a fresh
    # speak() starts clean. The exiting playback thread's finally only
    # clears its OWN handle (safe against a concurrent new registration:
    # compare-and-clear under the lock never wipes a newer handle).
    with _playback_lock:
        if playback is not None and _current_playback is playback:
            _current_playback = None
            # [F32] pop the identity key in the owner finally.
