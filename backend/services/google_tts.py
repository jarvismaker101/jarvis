"""Google Translate TTS — the zero-cost, key-less TTS fallback (tts role).

Why this engine exists
----------------------
Fish Audio is the primary voice, but it is metered: once the account's
credits are spent every reply degrades to SAPI5 (or ElevenLabs). Google's
``translate_tts`` endpoint needs no API key, no billing account and no
signup, so it is a genuine free fallback the user can pick per-role from
the VOICE MODEL (TTS) section of the sidebar.

Contract
--------
``speak_google_tts(text, before_playback=None) -> bool`` mirrors
``speak_fish_audio`` exactly, so ``voice._speak_chunk`` can treat both
engines interchangeably: True when the audio was played (or the user
stopped it), False when the engine could not produce audio at all so the
ladder can fall through to the next engine.

One playback owner (F32)
------------------------
Playback is delegated to the SAME ``audio_actor`` instance Fish uses, via
``fish_voice``'s playback primitives. Standing up a second playback
authority here would reintroduce exactly the bug F32 removed: two writers
racing one output stream, ``sd.stop()`` killing unrelated audio, and PCM
cache slots colliding across engines. Those helpers are private to the
``backend.services`` package and are imported deliberately — they are the
"play these PCM bytes through the one owner" path, and re-implementing
them would duplicate the backpressure/abort bookkeeping that is easy to
get subtly wrong. Fish remains the owner of the device-selection and
barge-in semantics; this module only supplies PCM.

No new dependency
-----------------
The endpoint is plain HTTP (``requests``, already a dependency) and the
MP3 is decoded by the PyAV path already used by ``fish_voice._decode_audio``.
``gTTS`` was deliberately NOT used: it pins ``click<8.2``, which downgrades
``click`` and breaks ``typer``/``uvicorn`` in this venv.

Endpoint limits (measured, not assumed)
---------------------------------------
``translate_tts`` answers HTTP 400 for ``q`` longer than ~200 characters,
so text is split on sentence then word boundaries before fetching. That
limit is also why no ``tk`` token is needed: the token only matters for the
over-200-character form this module never sends.

Voices
------
The endpoint serves ONE voice per language. Probing the regional hosts
(``translate.google.co.uk`` / ``.com.au`` / ``.co.in``) returned
byte-identical audio, so offering "en-US / en-GB / en-AU" would be a fake
distinction. The selectable models are therefore real languages, each
verified live against the endpoint (see ``LANGUAGES``).

Tempo
-----
The endpoint's audio is decoded correctly (24 kHz MP3 -> 44.1 kHz PCM keeps
its exact duration), but the ``tw-ob`` voice is drawly next to Fish and reads
as "slow motion". Decoded PCM is therefore time-stretched by
``DEFAULT_SPEED`` (1.5x) with ffmpeg's ``atempo``, which preserves pitch — a
plain resample would raise the pitch and turn the speaker into a chipmunk.
Set ``JARVIS_GOOGLE_TTS_SPEED`` to change it (1.0 disables the stretch).
"""

import math
import os
import threading
import time
import traceback

import requests

from backend.services import fish_voice as _fish
from backend.services.audio_actor import actor_abort, audio_cache_key

_TTS_URL = "https://translate.google.com/translate_tts"
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

#: The endpoint rejects ``q`` over ~200 chars with HTTP 400. Stay under it.
MAX_CHARS = 190

#: Selectable voices. Every entry was probed live and returned real MP3.
LANGUAGES = (
    "en", "hi", "es", "fr", "de", "it", "pt", "ja",
    "ko", "zh-CN", "ru", "ar", "nl", "pl", "tr", "sv",
)
DEFAULT_LANGUAGE = "en"

#: The endpoint's ``tw-ob`` voice is markedly drawlier than Fish's, which
#: reads as "slow motion" next to the rest of Jarvis. The PCM is therefore
#: time-stretched on the way out. 1.0 disables the stretch entirely.
#: Override per machine with ``JARVIS_GOOGLE_TTS_SPEED``.
DEFAULT_SPEED = 1.5
#: ffmpeg's ``atempo`` takes 0.5-2.0 per instance, so clamping keeps a typo
#: in the env var from producing either silence or a chipmunk.
_SPEED_FLOOR = 0.5
_SPEED_CEILING = 2.0
#: Everything this module hands the actor is 44100 Hz mono s16.
_RATE = 44100

#: Identity -> decoded PCM (44100 mono s16). Bounded, same eviction shape
#: as the Fish cache. Keyed through ``audio_cache_key`` so the same sentence
#: spoken by two engines (or two languages) never shares a slot.
_CACHE_MAX = 64
_cache = {}
_cache_lock = threading.Lock()

#: Reused across every request — skips the TCP + TLS handshake a fresh
#: ``requests.get`` would pay on each chunk.
_session = requests.Session()


def _resolve_language():
    """Language id for the selected tts model, else the default.

    Per-call registry read with safe degradation to the default, mirroring
    ``fish_voice._resolve_tts_model``: a registry hiccup must never take the
    voice path down.
    """
    try:
        from backend.services import model_registry
        sel = model_registry.get_model_for_role("tts")
        if str(sel.get("provider") or "").strip() == "gtts":
            model = str(sel.get("model") or "").strip()
            if model:
                return model
    except Exception:
        pass
    return DEFAULT_LANGUAGE


def _resolve_speed():
    """Playback speed multiplier for this engine (clamped, never raises).

    Read per call rather than cached at import so a test — or the user
    editing ``.env`` and restarting — takes effect without reimporting.
    """
    raw = str(os.getenv("JARVIS_GOOGLE_TTS_SPEED", "") or "").strip()
    if not raw:
        return DEFAULT_SPEED
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_SPEED
    # NaN/inf are typos, not an intent to max out: NaN in particular would
    # survive min()/max() comparisons and clamp to the ceiling.
    if not math.isfinite(value) or value <= 0:
        return DEFAULT_SPEED
    return max(_SPEED_FLOOR, min(_SPEED_CEILING, value))


def _speed_tag(speed):
    """Stable, cache-safe rendering of a speed for the identity key."""
    return "%.2f" % float(speed)


def _cache_key(text, lang, speed=None):
    """Identity for one utterance: engine + language + speed + text.

    The speed belongs in the key: without it, changing the setting would
    replay the previous tempo from cache instead of re-synthesising.
    """
    if speed is None:
        speed = _resolve_speed()
    return audio_cache_key("google-tts", lang,
                           "pcm@%sx" % _speed_tag(speed), str(text or ""))


def _store(ck, pcm):
    with _cache_lock:
        _cache[ck] = bytes(pcm)
        while len(_cache) > _CACHE_MAX:
            oldest = next(iter(_cache))
            if oldest == ck:
                break
            _cache.pop(oldest, None)


def _load(ck):
    with _cache_lock:
        return _cache.get(ck)


def split_for_google(text, limit=MAX_CHARS):
    """Split *text* into endpoint-sized pieces on sentence/word boundaries.

    The endpoint refuses anything over ~200 characters, so a long run is cut
    at the last sentence terminator that fits, then at the last space, and
    only for a single unbroken word is it hard-sliced. Returns [] for
    whitespace-only input.
    """
    clean = " ".join(str(text or "").split())
    if not clean:
        return []
    if len(clean) <= limit:
        return [clean]

    pieces = []
    remaining = clean
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = -1
        for sep in (". ", "! ", "? ", "; ", ", "):
            idx = window.rfind(sep)
            if idx > cut:
                # Keep the terminator, drop the space after it.
                cut = idx + len(sep) - 1
        if cut <= 0:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit
        piece = remaining[:cut].strip()
        if not piece:
            piece = remaining[:limit]
            cut = limit
        pieces.append(piece)
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _fetch_mp3(text, lang, timeout=(3.05, 15)):
    """MP3 bytes for one endpoint-sized piece, or None.

    One retry: the endpoint occasionally answers a transient 5xx or an empty
    body, and a single retry is far cheaper than degrading the whole reply to
    SAPI5.
    """
    params = {"ie": "UTF-8", "q": text, "tl": lang, "client": "tw-ob"}
    headers = {"User-Agent": _USER_AGENT,
               "Referer": "https://translate.google.com/"}
    last = "unknown"
    for attempt in range(2):
        try:
            resp = _session.get(_TTS_URL, params=params, headers=headers,
                                timeout=timeout)
            if resp.status_code == 200 and len(resp.content) > 512:
                return resp.content
            last = "HTTP %s" % resp.status_code
        except Exception as exc:
            last = "%s: %s" % (type(exc).__name__, exc)
        if attempt == 0:
            time.sleep(0.15)
    print("[GOOGLE-TTS] fetch failed (%s)" % last)
    return None


def _speed_up(pcm_bytes, factor):
    """Time-stretch *pcm_bytes* by *factor*, preserving pitch.

    Uses ffmpeg's ``atempo`` through PyAV, which is a WSOLA-style
    time-stretch: the voice speaks faster but keeps its pitch. A plain
    resample would have been two lines, but it shifts the pitch up with the
    tempo and turns the speaker into a chipmunk.

    Any failure returns the untouched PCM, so a broken filter degrades to
    normal-speed speech rather than silence.
    """
    if not pcm_bytes or factor == 1.0:
        return pcm_bytes
    try:
        import av
        import numpy as np
    except Exception:
        return pcm_bytes
    try:
        graph = av.filter.Graph()
        source = graph.add(
            "abuffer",
            "sample_rate=%d:sample_fmt=s16:channel_layout=mono"
            ":time_base=1/%d" % (_RATE, _RATE))
        tempo = graph.add("atempo", "%.4f" % factor)
        sink = graph.add("abuffersink")
        source.link_to(tempo)
        tempo.link_to(sink)
        graph.configure()

        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        out = []
        # Feed one second at a time: a single huge frame would work too, but
        # bounded blocks keep the filter's internal buffer small.
        for start in range(0, samples.size, _RATE):
            block = samples[start:start + _RATE]
            frame = av.AudioFrame.from_ndarray(
                block.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = _RATE
            frame.pts = start
            graph.push(frame)
            while True:
                try:
                    out.append(graph.pull().to_ndarray().tobytes())
                except Exception:
                    break
        graph.push(None)
        while True:
            try:
                out.append(graph.pull().to_ndarray().tobytes())
            except Exception:
                break
        stretched = b"".join(out)
        return stretched or pcm_bytes
    except Exception as exc:
        print("[GOOGLE-TTS] speed-up unavailable, using normal speed: %s" % exc)
        return pcm_bytes


def _decode_pcm(content, speed=None):
    """MP3 bytes -> 44100 mono s16 PCM at *speed*, or None.

    Reuses the PyAV decoder Fish already depends on so both engines hand the
    playback owner byte-identical audio formats.
    """
    audio = _fish._decode_audio(content)
    if audio is None:
        return None
    try:
        audio = audio.set_frame_rate(_RATE).set_channels(1)
        pcm = bytes(audio.raw_data)
    except Exception as exc:
        print("[GOOGLE-TTS] resample failed: %s" % exc)
        return None
    if not pcm:
        return None
    return _speed_up(pcm, _resolve_speed() if speed is None else speed)


def synthesise_pcm(text, lang=None, speed=None):
    """Decoded PCM for the whole *text* (cached), or None on failure.

    A piece that fails aborts the whole utterance: speaking half a sentence
    and then falling silent is worse than returning None so the ladder can
    hand the complete sentence to another engine.

    *speed* is resolved ONCE here and threaded through both the cache key and
    the stretch, so the two can never disagree if the env var changes
    mid-utterance.
    """
    lang = lang or _resolve_language()
    if speed is None:
        speed = _resolve_speed()
    text = str(text or "").strip()
    if not text:
        return None

    ck = _cache_key(text, lang, speed)
    cached = _load(ck)
    if cached is not None:
        return cached

    parts = split_for_google(text)
    if not parts:
        return None

    pcm = bytearray()
    for piece in parts:
        mp3 = _fetch_mp3(piece, lang)
        if mp3 is None:
            return None
        decoded = _decode_pcm(mp3, speed)
        if not decoded:
            return None
        pcm.extend(decoded)

    if not pcm:
        return None
    result = bytes(pcm)
    _store(ck, result)
    return result


def _play_pcm(pcm_bytes, ck):
    """Play decoded PCM through the ONE playback owner (F32)."""
    handle = _fish._register_sounddevice_playback()
    try:
        if handle.stopped:
            return True
        return _fish._play_pcm_through_actor(
            bytes(pcm_bytes), handle, ck=ck, rate=44100, channels=1)
    finally:
        _fish._clear_sounddevice_playback(handle)


def speak_google_tts(text, before_playback=None):
    """Speak *text* with Google Translate TTS.

    Returns True when the audio was played (or the user stopped it) and
    False when no audio could be produced, so the caller's ladder can fall
    through. Never raises.
    """
    try:
        clean = str(text or "").strip()
        if not clean:
            return False
        lang = _resolve_language()
        speed = _resolve_speed()
        pcm = synthesise_pcm(clean, lang, speed)
        if not pcm:
            return False
        if callable(before_playback):
            before_playback()
        return bool(_play_pcm(pcm, _cache_key(clean, lang, speed)))
    except Exception:
        print("[GOOGLE-TTS] Exception:")
        traceback.print_exc()
        return False


def prefetch_google_tts(text):
    """Synthesise *text* in the background so playback never waits on TTS.

    The decoded audio lands in this module's cache; when the playback loop
    reaches that sentence ``speak_google_tts`` finds it ready and starts
    instantly.
    """
    if not text or not str(text).strip():
        return
    threading.Thread(target=synthesise_pcm, args=(text,), daemon=True).start()


def stop_google_tts():
    """Abort playback.

    Same single owner as Fish, so this is literally the same abort — the
    call exists so ``voice.stop_speaking`` reads symmetrically across
    engines rather than reaching into the Fish module for a Google stream.
    """
    try:
        actor_abort()
    except Exception:
        pass


def warm_up_google_tts():
    """Prime DNS/TLS/session so the first real reply is not the slow one."""
    try:
        synthesise_pcm("Warm up, sir.")
    except Exception:
        pass
