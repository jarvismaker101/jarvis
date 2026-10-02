import os
import queue
import re
import threading
import time
import traceback

import pyttsx3

from backend import listener_state
from backend.services.earcons import play_ready_earcon, play_reply_start_earcon
from backend.services.elevenlabs_voice import speak_elevenlabs, stop_elevenlabs
from backend.services.fish_voice import (
    fish_ws_begin_reply,
    fish_ws_end_reply,
    fish_ws_session,
    play_ws_reply,
    prefetch_fish_audio,
    speak_fish_audio,
    stop_fish_audio,
    warm_up_fish_tts,
    ws_audio_reached_device,
)
from backend.services.google_tts import (
    prefetch_google_tts,
    speak_google_tts,
    stop_google_tts,
    warm_up_google_tts,
)
# F31: the shared typed channel contract. The speaker narrates ONLY
# final-answer text; a reasoning delta is preserved, never spoken.
from backend.services.openai_compat_client import StreamDelta


is_speaking = False
_current_text = ""
_speech_generation = 0
_state_lock = threading.Lock()
_engine = None
PREFER_LOCAL_TTS = os.getenv("JARVIS_PREFER_LOCAL_TTS", "1") != "0"
LOCAL_TTS_RATE = int(os.getenv("JARVIS_LOCAL_TTS_RATE", "180"))
REMOTE_TTS_CHAR_LIMIT = int(os.getenv("JARVIS_REMOTE_TTS_CHAR_LIMIT", "150"))
FISH_TTS_CHAR_LIMIT = int(os.getenv("JARVIS_FISH_TTS_CHAR_LIMIT", "1800"))
GOOGLE_TTS_CHAR_LIMIT = int(os.getenv("JARVIS_GOOGLE_TTS_CHAR_LIMIT", "1800"))

#: [P1-01] Streaming chunk boundaries.
#:
#: The old rule needed whitespace AFTER the sentence punctuation, so a finished
#: sentence was not recognised until the NEXT delta arrived — that was the
#: first-chunk delay. And once 40 characters had accumulated the buffer was
#: flushed wherever the cut happened to fall, which split words in half.
FIRST_CHUNK_CLAUSE_BREAKS = ",;:\u2014.!?"
#: Words that must precede a clause break before it may start playback.
FIRST_CHUNK_MIN_WORDS = 3
#: Fallback for the first chunk when no clause break has arrived yet: cut at
#: the last word boundary once this much text is buffered.
FIRST_CHUNK_MIN_CHARS = 30
#: Upper bound for a later chunk. Reached only by text with no sentence
#: boundary in it; the cut still lands on a word boundary.
LATER_CHUNK_MAX_CHARS = 200
#: A sentence that ENDS the buffer (the model finished the sentence and then
#: paused, so no following delta is coming) counts as a boundary after this
#: long. Without it a complete sentence would sit in the buffer waiting for a
#: delta that never arrives.
PUNCTUATION_SETTLE_SECONDS = 0.12
#: Last resort for a buffer with no boundary at all. Unchanged [P1-01].
STALL_FLUSH_SECONDS = 0.3
#: The playback loop must poll finer than PUNCTUATION_SETTLE_SECONDS, or the
#: 120ms rule could not fire on time. A Queue.get timeout is a cheap condition
#: wait, so this costs nothing meaningful while idle.
STREAM_POLL_SECONDS = 0.05
#: Punctuation that means "this sentence is complete" for the settle rule.
_SENTENCE_END_CHARS = ".!?\u2026"
_WORD_BOUNDARY_RE = re.compile(r"\s")


def _word_count(text):
    return len(text.split())


def _last_word_boundary(text):
    """Index of the LAST whitespace in *text*, or None.

    The last one (rather than the first) puts as much complete text as is
    available into the first chunk, which is what starts playback soonest.
    """
    found = None
    for match in _WORD_BOUNDARY_RE.finditer(text):
        found = match.start()
    return found


def _next_word_boundary(text, cap):
    """Index of the FIRST whitespace at or after *cap*, or None.

    "Extend to the next word boundary rather than cutting": when a length cap
    lands inside a word the chunk grows to finish it. None means the word runs
    past the cap with no boundary in sight — wait instead of cutting.
    """
    for match in _WORD_BOUNDARY_RE.finditer(text):
        if match.start() >= cap:
            return match.start()
    return None


def _clause_cut(text):
    """Index just after the first usable clause break, or None.

    Usable means at least FIRST_CHUNK_MIN_WORDS words precede it AND the
    punctuation is either the last character of the buffer (the model has not
    sent the next delta yet — cutting here is the latency win) or followed by
    whitespace. The whitespace requirement keeps "3.5" or a URL from reading as
    a clause break, which would split a real token in two.
    """
    for index, char in enumerate(text):
        if char not in FIRST_CHUNK_CLAUSE_BREAKS:
            continue
        if _word_count(text[:index]) < FIRST_CHUNK_MIN_WORDS:
            continue
        if index + 1 == len(text) or text[index + 1].isspace():
            return index + 1
    return None


def _stream_cut(text, first):
    """``(chunk, remainder)`` to hand to TTS now, or None.

    *first* selects the first-chunk rule (clause boundary, else an early word
    boundary) instead of the later-chunk rule. Both paths cut at a word
    boundary: a returned chunk never ends inside a word.
    """
    if first:
        cut = _clause_cut(text)
        if cut is not None:
            return text[:cut], text[cut:]
        # No clause break yet: once this much has been buffered, start anyway
        # at the last word boundary. The condition is on the BUFFER length —
        # waiting for a boundary that lands past the threshold would mean
        # waiting for another delta, which is the delay this item removes.
        if len(text) >= FIRST_CHUNK_MIN_CHARS:
            cut = _last_word_boundary(text)
            if cut is not None:
                return text[:cut], text[cut + 1:]
        return None

    if len(text) > LATER_CHUNK_MAX_CHARS:
        cut = _next_word_boundary(text, LATER_CHUNK_MAX_CHARS - 1)
        if cut is not None:
            return text[:cut], text[cut + 1:]
    return None



# [S12] Markdown is for screens, not speakers: if a model slips one of these
# past the voice prompt, strip it before the text reaches TTS.
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*")
_MD_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_MD_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")
_MD_BULLET_RE = re.compile(r"(?m)^\s{0,3}(?:[-*+]|\u2022)\s+")
_MD_EMPHASIS_RE = re.compile(r"\*{1,3}|_{2,}")
_MD_URL_RE = re.compile(r"https?://\S+|www\.\S+")


def strip_markdown_for_speech(text):
    """Remove markdown and bare URLs so TTS never reads formatting aloud (S12)."""
    if not text:
        return text
    text = _MD_IMAGE_RE.sub("", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_CODE_FENCE_RE.sub("", text)
    text = _MD_INLINE_CODE_RE.sub(r"\1", text)
    text = _MD_HEADING_RE.sub("", text)
    text = _MD_BULLET_RE.sub("", text)
    text = _MD_EMPHASIS_RE.sub("", text)
    text = _MD_URL_RE.sub("", text)
    return text


def clean_text(text):
    text = strip_markdown_for_speech(text)
    text = text.replace("\n", ". ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_speech_chunks(text, max_chars=None):
    """Split long text into natural sentence chunks for low-latency TTS.

    The first chunk is returned as soon as a sentence boundary is found,
    so playback can start while the rest of the reply is still being
    generated. *max_chars* (default FISH_TTS_CHAR_LIMIT) caps chunk size.
    """
    if max_chars is None:
        max_chars = FISH_TTS_CHAR_LIMIT
    max_chars = max(80, int(max_chars))

    if len(text) <= max_chars:
        chunks = [text]
    else:
        sentence_end = re.compile(r"(?<=[.!?])\s+")
        parts = sentence_end.split(text)

        chunks = []
        current = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if not current:
                current = part
            elif len(current) + 1 + len(part) <= max_chars:
                current += " " + part
            else:
                chunks.append(current)
                current = part
        if current:
            chunks.append(current)
        chunks = chunks or [text]

    # Short first chunk cap ~120 chars for low latency (comma > space > hard cut)
    if chunks and len(chunks[0]) > 120:
        first = chunks[0]
        comma_idx = first.rfind(",", 0, 120)
        if comma_idx >= 30:
            new_first = first[: comma_idx + 1].strip()
            remainder = first[comma_idx + 1 :].strip()
        else:
            space_idx = first.rfind(" ", 0, 120)
            if space_idx != -1:
                new_first = first[:space_idx].strip()
                remainder = first[space_idx + 1 :].strip()
            else:
                new_first = first[:120].strip()
                remainder = first[120:].strip()
        if remainder:
            chunks = [new_first, remainder] + chunks[1:]
        else:
            chunks = [new_first] + chunks[1:]
    return chunks or [text]


def _is_current_generation(generation):
    with _state_lock:
        return _speech_generation == generation


_tts_lock = threading.Lock()


def _speak_local(clean, generation, play_earcon=True):
    global _engine

    if not _is_current_generation(generation):
        return True

    # Only one pyttsx3 engine can run at a time.
    acquired = _tts_lock.acquire(timeout=8)
    if not acquired:
        return False

    engine = None
    try:
        if not _is_current_generation(generation):
            return True

        engine = pyttsx3.init()
        engine.setProperty("rate", LOCAL_TTS_RATE)
        engine.setProperty("volume", 1.0)
        _engine = engine

        if play_earcon:
            play_reply_start_earcon()
        engine.say(clean)
        engine.runAndWait()
        return True
    except Exception:
        print("[VOICE] Local TTS error:")
        traceback.print_exc()
        return False
    finally:
        if _engine is engine:
            _engine = None
        _tts_lock.release()


def stop_speaking(signal_ready=True):
    """Stop playback now.

    ``signal_ready`` controls the "I stopped, your turn" cue. [P1-02] The
    barge-in path passes False: the user is *already talking*, so a beep
    confirming "I can hear you" is noise, and it is the worst-placed cue in the
    system — it lands in the middle of their sentence. Every other caller (the
    UI stop button, a task mute) keeps the cue.
    """
    global is_speaking, _speech_generation, _engine

    try:
        stop_elevenlabs()
    except Exception:
        pass

    try:
        stop_fish_audio()
    except Exception:
        pass

    # [S7] A stop also closes the live Fish session, so its reader thread and
    # socket do not outlive the reply they were synthesising for.
    try:
        fish_ws_end_reply()
    except Exception:
        pass

    # [P0-07] stop_fish_audio() already aborted the ONE shared playback actor.
    # stop_google_tts drives that same actor, so asking it to abort again would
    # cut the same device twice on every stop.
    try:
        stop_google_tts(abort_actor=False)
    except Exception:
        pass

    # [P0-07] pyttsx3 owns its event loop on whichever thread called
    # runAndWait(). Calling stop() from this (different) thread while no loop
    # is running just queues a stale stop command that the NEXT utterance
    # would consume - silencing a reply nobody asked to cancel. So only
    # interrupt it when it is genuinely mid-loop. ``_inLoop`` is pyttsx3's own
    # documented "running an event loop" flag.
    try:
        engine = _engine
        if engine is not None and getattr(engine, "_inLoop", False):
            engine.stop()
    except Exception:
        pass

    should_signal_ready = False
    with _state_lock:
        should_signal_ready = is_speaking
        _speech_generation += 1
        is_speaking = False
        listener_state.set_speaking(False)

    if should_signal_ready and signal_ready:
        # [P1-02] Never reached from barge-in (signal_ready=False there). Kept
        # off the capture thread's critical section regardless: play_ready_earcon
        # only spawns its own daemon thread and returns.
        play_ready_earcon()

    print("[VOICE] Speech stopped")


def pause_speaking():
    """Pause the current playback while KEEPING the remainder resumable.

    [P1-07] A TRUE pause, not a stop — that distinction is the whole point:

      * the device stops immediately, because the actor's play loop is parked
        and its already-buffered audio is cut;
      * the byte POSITION is preserved (``pause()`` returns the cursor), so
        "continue" replays the remainder of the interrupted chunk from exactly
        where it stopped instead of re-synthesising it;
      * the sentences still queued or buffered survive, because the generation
        is deliberately NOT bumped (the old implementation called
        ``stop_speaking()``, which is what made pause behave like stop).

    Returns True when there was something to resume.
    """
    speaker = get_active_stream()
    paused = False
    if speaker is not None:
        try:
            paused = bool(speaker.pause())
        except Exception:
            paused = False
    else:
        # No stream speaker owns this audio (a backend ``speak()`` reply, a task
        # announcement): pause the ONE actor directly. This is also the path the
        # backend process uses for typed-UI replies.
        try:
            paused = _pause_actor_playback() is not None
        except Exception:
            paused = False
    remaining = pending_speaking_text()
    if remaining:
        listener_state.set_remaining(remaining)
    if not paused and not remaining:
        # Nothing was playing: a pause is a safe no-op and must NOT stop
        # anything. Claiming success here is what turned "pause" into "stop".
        print("[VOICE] Speech pause — nothing was playing")
        return False
    print(f"[VOICE] Speech paused — {len(remaining)} chars remain")
    return True


def resume_local_playback():
    """Un-park locally paused audio. True when audio actually resumed.

    Never starts a second playback: a pause leaves the actor's play loop PARKED,
    so the resume is a replay of the unplayed tail into that same loop. The
    speaker's gate is opened afterwards, which is what lets the reply carry on
    with the sentences that were still queued.
    """
    resumed = False
    try:
        resumed = bool(_resume_actor_playback())
    except Exception:
        resumed = False
    speaker = get_active_stream()
    if speaker is not None:
        try:
            speaker.resume()
        except Exception:
            pass
    return resumed


def resume_speaking():
    """Speak what a pause left unplayed (F35 / P1-07).

    Returns the resumed TEXT ("" when there was nothing), so a caller can report
    the truth. Two shapes, in this order:

    1. a byte-level pause: the parked play loop replays the remainder from its
       cursor and the stream speaker's queue carries on. The text snapshot is
       deliberately NOT re-spoken here — every queued sentence would then exist
       twice (once in the snapshot, once in the speaker's queue);
    2. nothing parked (nothing was mid-playback): the legacy path, which
       re-speaks the stored remainder.
    """
    audio_resumed = resume_local_playback()
    remaining = listener_state.pop_remaining()
    if audio_resumed:
        if remaining:
            print(f"[VOICE] Resumed unplayed audio — {len(remaining)} chars still queued")
        return remaining
    if not remaining:
        return ""
    print(f"[VOICE] Resuming {len(remaining)} chars of unplayed speech")
    speak(remaining)
    return remaining


def pending_speaking_text():
    """The narration that has not reached the device yet."""
    parts = []
    speaker = get_active_stream()
    if speaker is not None:
        try:
            queued = speaker.pending_text()
        except Exception:
            queued = ""
        if queued:
            parts.append(queued)
    stored = listener_state.get_remaining()
    if stored:
        parts.append(stored)
    if not parts and _current_text:
        parts.append(_current_text)
    return " ".join(part.strip() for part in parts if part and part.strip())


def _resolve_tts_provider():
    """The tts provider the user selected in the sidebar ("fish" | "gtts").

    Degrades to "fish" when the registry cannot be read, matching how every
    other registry read on this path behaves — a settings hiccup must never
    take the voice path down.
    """
    try:
        from backend.services import model_registry
        sel = model_registry.get_model_for_role("tts")
        prov = str(sel.get("provider") or "").strip()
        if prov:
            return prov
    except Exception:
        pass
    return "fish"


#: [P1-08] A reply session resolves its tts provider EXACTLY ONCE and then
#: carries it, instead of asking the registry three times per chunk. The
#: engine cannot meaningfully change mid-utterance, and the settings registry
#: still re-stats its file on every resolve, so a mid-reply change lands on the
#: NEXT reply (live switching is preserved). ``StreamSpeaker`` stores its own
#: copy; a bare ``None`` provider means "resolve now", which keeps direct
#: callers (and tests) honest.

def _resolve_session_tts_provider():
    """The provider for a NEW reply session."""
    return _resolve_tts_provider()


def _cloud_tts_ladder(provider=None):
    """Cloud engines in the order they should be tried for this reply.

    The user's selected engine is ALWAYS first, so an explicit choice is
    honoured. The other engine follows as a fallback rather than being
    dropped: a metered engine that has run out of credits, or a free engine
    that is unreachable, should degrade the voice — not silence the reply.
    Local SAPI5 and ElevenLabs stay behind both, exactly as before.
    """
    engines = {
        "fish": ("fish", speak_fish_audio, FISH_TTS_CHAR_LIMIT),
        "gtts": ("google", speak_google_tts, GOOGLE_TTS_CHAR_LIMIT),
    }
    chosen = _resolve_tts_provider() if provider is None else provider
    if chosen == "gtts":
        order = ("gtts", "fish")
    else:
        order = ("fish", "gtts")
    return [engines[name] for name in order]


def _tts_char_limit(provider=None):
    """Chunking limit for the engine that will actually speak."""
    chosen = _resolve_tts_provider() if provider is None else provider
    if chosen == "gtts":
        return GOOGLE_TTS_CHAR_LIMIT
    return FISH_TTS_CHAR_LIMIT


def warm_up_selected_tts():
    """Prime whichever engine will actually speak.

    Warming the unselected engine wastes a synthesis call and still leaves the
    first real reply paying the cold-start cost, so the choice is resolved
    here rather than hard-coded to one vendor at startup.
    """
    if _resolve_tts_provider() == "gtts":
        warm_up_google_tts()
    else:
        warm_up_fish_tts()


def _playback_is_active():
    """True while any engine in this process is playing audio."""
    with _state_lock:
        return bool(is_speaking)


# ── [P1-07] pause / resume of the ONE playback owner ───────────────────
#
# The actor is a module singleton, so these are resolved lazily by name rather
# than captured at import time (the same reason the TTS engines are resolved per
# call): a test that patches the actor's pause keeps working, and a hoisted
# reference cannot silently drive the real device.

def _pause_actor_playback():
    """Cut the device and park the play loop, keeping the cursor. None if idle."""
    from backend.services.audio_actor import get_actor
    return get_actor().pause()


def _resume_actor_playback():
    """Replay a paused utterance's unplayed tail. True when audio resumed."""
    from backend.services.audio_actor import get_actor
    return get_actor().resume_playback()


def _prefetch_implementations():
    """Engines that can warm themselves ahead of playback, by provider string.

    Built per call, like `_cloud_tts_ladder`: the engine functions are resolved
    from the module namespace when the prefetch happens, not captured at import.

    A provider that is NOT in this map has no prefetch implementation, and the
    correct behaviour then is to do nothing. Warming a different provider would
    spend a metered call (Fish) and still leave the selected engine cold, so
    "no-op" is the only safe fallback.
    """
    return {
        "fish": prefetch_fish_audio,
        "gtts": prefetch_google_tts,
    }


def _prefetch_tts_audio(text, plays_next=False, provider=None):
    """Warm the engine that will actually speak the NEXT sentence.

    Prefetching the wrong engine wastes a synthesis call and leaves the real
    one cold, so this follows the same selection as `_cloud_tts_ladder`. The
    selection is resolved EXACTLY ONCE per call — and, since P1-08, per reply
    session (``generation``) — so a settings change landing mid-call cannot
    warm a different engine than the one that was chosen.

    Every prefetch in this module goes through here (P1-09). A direct
    ``prefetch_fish_audio`` call spends metered Fish credits even when Google
    is the selected engine, and leaves the selected engine cold — so the only
    Fish call this helper can make is the one the user asked for.

    ``plays_next`` (P0-05) marks *text* as the sentence that will play as soon
    as the worker is free. While nothing is playing that sentence must NOT be
    prefetched: racing its own playback is what made the first sentence wait
    for a full synthesis instead of streaming. Both halves of the rule live
    here so that every caller inherits it, rather than each call site having
    to remember it.
    """
    if not text or not str(text).strip():
        return
    if plays_next and not _playback_is_active():
        return
    warm = _prefetch_implementations().get(
        _resolve_tts_provider() if provider is None else provider)
    if warm is None:
        return
    warm(text)


def _speak_chunk(chunk, generation, is_first_chunk=False, provider=None):
    """Speak one pre-chunked piece using the TTS ladder.

    Cloud engines first — the selected one, then the other. Fish uses
    simpleaudio/WASAPI and works reliably even from a hidden Electron child
    (pyttsx3/SAPI5 often returns instantly with no sound when run
    hidden/minimized, leaving the UI in 'speaking' with silence), and Google
    is the key-less zero-cost fallback. Local SAPI5 is the offline path and
    ElevenLabs the last resort. Shared by `speak()` and `StreamSpeaker`.
    """
    # Parallel earcon + first cloud fetch: overlap download with earcon
    earcon_started = False
    if is_first_chunk:
        try:
            threading.Thread(target=play_reply_start_earcon, daemon=True).start()
            earcon_started = True
        except Exception:
            pass

    before = None if earcon_started else (play_reply_start_earcon if is_first_chunk else None)

    # The FIRST eligible engine runs unconditionally (both callers already
    # gate on the generation); only the fallback engine is generation-checked,
    # so a stop landing mid-ladder never kicks off a fresh synthesis. The
    # earcon hook goes to whichever engine actually runs first, so a fallback
    # can never replay it (the hook fires only after synthesis succeeds).
    is_primary = True
    for _name, engine, limit in _cloud_tts_ladder(provider):
        if not is_primary and not _is_current_generation(generation):
            return True
        if len(chunk) > limit:
            continue
        try:
            if engine(chunk, before_playback=(before if is_primary else None)):
                return True
        except Exception:
            print("[VOICE] %s engine error:" % _name)
            traceback.print_exc()
        is_primary = False

    # Local SAPI5 fallback — but verify it actually played (>0.8s for >10 chars).
    # In hidden sessions pyttsx3 can return True instantly with no audio.
    if _speak_local(chunk, generation, play_earcon=not earcon_started):
        # quick sanity: very short elapsed for a long chunk means silent SAPI5
        # — treat as failure so we still try ElevenLabs.
        return True

    if len(chunk) <= REMOTE_TTS_CHAR_LIMIT and _is_current_generation(generation):
        success = speak_elevenlabs(chunk, before_playback=None)
        if success:
            return True

    # final safety net — try local again even if prefer was false
    _speak_local(chunk, generation, play_earcon=False)
    return True
    return True


def speak(text):
    global is_speaking, _current_text, _speech_generation

    def _run():
        global is_speaking, _current_text, _engine

        try:
            clean = clean_text(text)

            with _state_lock:
                is_speaking = True
                _current_text = clean
                listener_state.set_speaking(True)
                listener_state.set_remaining("")

            if not clean:
                return

            # Voice each queued sentence while the NEXT one is already being
            # synthesised in the background — hides the Fish round-trip behind
            # the current chunk's playback instead of stalling between chunks.
            # [P1-08] The engine is resolved ONCE for the whole reply, not three
            # times per chunk; the registry's own mtime check means a selection
            # changed mid-reply still applies to the NEXT one.
            provider = _resolve_session_tts_provider()
            chunks = split_speech_chunks(clean)
            is_first_chunk = True

            # [S7] ONE live Fish session per reply: every chunk is fed into a
            # single WebSocket and the reply plays as one continuous utterance
            # (earlier first audio, no per-sentence intonation reset). The
            # per-sentence ladder below stays the fallback and runs only when
            # the WS path produced NO audio at all.
            fish_ws_begin_reply()
            ws_session = fish_ws_session() if provider == "fish" else None
            ws_spoke = False
            try:
                if ws_session is not None and chunks:
                    for chunk in chunks:
                        ws_session.feed(chunk)
                    ws_session.finish_text()
                    outcome = play_ws_reply(ws_session,
                                            is_current=lambda: _is_current_generation(generation),
                                            earcon=play_reply_start_earcon)
                    ws_spoke = bool(outcome.get("audio_started"))
                    if ws_spoke:
                        return
            finally:
                fish_ws_end_reply()

            for index, chunk in enumerate(chunks):
                if not _is_current_generation(generation):
                    return
                if index + 1 < len(chunks):
                    # chunks[index] plays first, so the next chunk is never the
                    # sentence about to play and is always safe to warm.
                    _prefetch_tts_audio(chunks[index + 1], provider=provider)
                _speak_chunk(chunk, generation, is_first_chunk=is_first_chunk,
                             provider=provider)
                is_first_chunk = False

        except Exception:
            print("Voice error:")
            traceback.print_exc()

        finally:
            should_signal_ready = False
            with _state_lock:
                if generation == _speech_generation:
                    is_speaking = False
                    listener_state.set_speaking(False)
                    should_signal_ready = True

            if should_signal_ready:
                play_ready_earcon()

    with _state_lock:
        _speech_generation += 1
        generation = _speech_generation

    threading.Thread(target=_run, daemon=True).start()


_STREAM_STOP = object()
_current_stream = None


def set_active_stream(speaker):
    """Register the StreamSpeaker for the current voice turn.

    Background announcements (e.g. opencode task completions) are appended
    to this stream so they never cancel a reply that is mid-answer.
    """
    global _current_stream
    with _state_lock:
        _current_stream = speaker


def get_active_stream():
    return _current_stream


def announce_to_stream(text):
    """Feed a background announcement into the active stream (if any).

    Returns True if the text was queued for speech, False if the caller
    should fall back to a normal `speak()`.
    """
    speaker = get_active_stream()
    if speaker is not None and speaker.enqueue_external(text):
        return True
    return False


class StreamSpeaker:
    """Streaming TTS: voice each complete sentence as the reply streams in.

    `speak()` waits for the entire reply, so a long answer delays the first
    sound by the full generation time. This class is fed reply text piecemeal
    (LLM stream deltas) and voices complete sentences immediately — the first
    words start within a couple of seconds while the rest keeps streaming.
    Sentences play back-to-back on a single worker so later parts of the same
    reply never cancel earlier ones. A fresh `speak()` or `stop_speaking()`
    still interrupts cleanly, while background announcements fed via
    `enqueue_external()` ride the same generation and never cancel anything.
    """

    def __init__(self, min_flush_chars=40):
        global _speech_generation

        self._buffer = ""
        self._buffer_lock = threading.Lock()
        self._last_feed_ts = time.monotonic()
        # [P1-01] Retained for API compatibility only: this used to be the
        # character count that flushed the buffer wherever the cut fell, which
        # split words in half. Chunk boundaries are now decided by
        # `_stream_cut` (clause / sentence / word boundary), so nothing gates
        # on `_min_flush` any more. Callers that pass it keep working.
        self._min_flush = max(40, int(min_flush_chars))
        self._sentence_re = re.compile(r"(?<=[.!?])\s+")
        self._finished = False
        self._closed = False
        self._spoken_any = False
        self._active = False
        self._played_first = False
        self._first_chunk_enqueued = False
        self._worker = None
        self._queue = queue.Queue()
        # [P1-07] A TRUE pause, distinct from ``close()``. Setting ``_paused``
        # parks the worker before it starts the next chunk, so the queued and
        # buffered sentences keep their place and the reply carries on from the
        # same utterance when "continue" arrives. It is deliberately NOT a
        # generation bump: that is what made the old pause behave like a stop.
        self._paused = False
        self._pause_cv = threading.Condition()
        self._pause_cursor = None
        #: [P1-07] The chunk the worker is HOLDING at the pause gate. It has
        #: already left the queue, so a paused reply has to remember it here:
        #: ``pending_text`` must not pretend the queue is empty, and "continue"
        #: must play this exact chunk.
        self._held_chunk = None
        # F31: the reasoning channel of this turn, preserved verbatim and
        # never handed to TTS (only final-answer text is narrated).
        self._reasoning = ""

        with _state_lock:
            _speech_generation += 1
            self._generation = _speech_generation
        #: [P1-08] This reply session's tts engine, resolved on FIRST use and
        #: then carried: the audio path used to ask the settings registry three
        #: times per chunk. The registry still re-stats its file per resolve, so
        #: a selection changed mid-reply lands on the next reply.
        self._tts_provider = None
        # [S7] Live Fish WebSocket mode for THIS reply: sentences are fed into
        # one session instead of being spoken one HTTP request at a time.
        # `_ws_fed` remembers what the session consumed, so a session that
        # failed BEFORE any audio can hand every sentence back to the ladder
        # (a session that already spoke must never replay).
        self._ws_mode = False
        self._ws_fed = []
        self._ws_thread = None
        self._ws_outcome = None
        #: [S7] ONE WebSocket per reply session. Once this reply's session is
        #: finished (handed back or ended) the reply stays on the ladder: a
        #: new session for the same reply would only re-consume its sentences.
        self._ws_disabled = False
        self._ws_ended = False

    def _tts_engine(self):
        """[P1-08] The tts provider for THIS reply session (resolved once)."""
        if self._tts_provider is None:
            self._tts_provider = _resolve_session_tts_provider()
        return self._tts_provider

    @property
    def spoken_any(self):
        """True once any sentence has been handed to the playback queue."""
        return self._spoken_any

    @property
    def reasoning(self):
        """Reasoning text received on the thinking channel (never spoken)."""
        return self._reasoning

    def feed(self, delta):
        """Append a streamed text delta and voice any complete sentences.

        *delta* is answer text (``str``), or a typed
        :class:`~backend.services.openai_compat_client.StreamDelta`. A
        ``reasoning`` delta is preserved on this speaker's reasoning channel
        and is NEVER enqueued for speech — a mixed thought/final response
        narrates only its final answer.
        """
        if self._finished or self._closed or delta is None:
            return
        if isinstance(delta, StreamDelta):
            if not delta.is_final:
                if delta.text:
                    with self._buffer_lock:
                        self._reasoning += delta.text
                return
            delta = delta.text
        if not delta:
            return
        self._process_text(delta)

    def enqueue_external(self, text):
        """Append a background announcement to the same playback stream.

        Runs on this stream's own generation, so it queues BEHIND whatever
        is already speaking and never cancels an in-progress reply. Returns
        True if the text was queued.

        F31: an announcement is a SEPARATE utterance. Whatever answer text is
        still buffered is flushed as its own chunk FIRST (and the announcement
        is flushed immediately after it), so announcement words can never be
        spliced into an unfinished answer — 'Working on the task' followed by
        'Task finished.' stays two utterances, never 'taskTask finished.'.
        """
        if self._closed or not text or not text.strip():
            return False
        pending = None
        with self._buffer_lock:
            if self._buffer.strip():
                pending, self._buffer = self._buffer, ""
        if pending:
            self._enqueue(pending.strip())
        # The announcement never enters the answer buffer: it is split on its
        # own sentence boundaries and enqueued as its own utterances, so the
        # answer's next delta can never be appended to it either.
        for piece in self._sentence_re.split(text.strip()):
            piece = piece.strip()
            if piece:
                self._enqueue(piece)
        return True

    def _process_text(self, text):
        to_enqueue = []
        with self._buffer_lock:
            self._last_feed_ts = time.monotonic()
            # Append deltas EXACTLY as received — the LLM owns spacing, so
            # 'hel' + 'lo' must stay 'hello', never 'hel lo'.
            self._buffer += text
            # `first_pending` is tracked locally because `_enqueue` — which is
            # what flips `_first_chunk_enqueued` — is called below, off this
            # lock, so the attribute still reads "no chunk yet" for a delta that
            # yields several chunks.
            first_pending = not self._first_chunk_enqueued
            while True:
                if first_pending:
                    # [P1-01] The first chunk may break mid-sentence at a clause
                    # boundary. Checked BEFORE the sentence split so a delta
                    # holding a whole sentence still starts playback with the
                    # small clause-sized piece: the first TTS request is issued
                    # either way, but a shorter one returns audio sooner.
                    cut = _stream_cut(self._buffer, first=True)
                    if cut is not None:
                        chunk, self._buffer = cut
                        if chunk.strip():
                            to_enqueue.append(chunk.strip())
                        first_pending = False
                        continue
                # Complete sentences, as before.
                parts = self._sentence_re.split(self._buffer)
                if len(parts) > 1:
                    self._buffer = parts[-1]
                    for part in parts[:-1]:
                        if part.strip():
                            to_enqueue.append(part.strip())
                            first_pending = False
                    continue
                # Later chunks: only the length cap remains.
                cut = _stream_cut(self._buffer, first=False)
                if cut is None:
                    break
                chunk, self._buffer = cut
                if chunk.strip():
                    to_enqueue.append(chunk.strip())
        for sentence in to_enqueue:
            self._enqueue(sentence)
        # Ensure worker is running for stall flush even if nothing enqueued yet
        with self._buffer_lock:
            has_pending = bool(self._buffer.strip())
        if has_pending:
            self._start_worker()

    def flush(self):
        """Voice whatever is buffered even without a sentence boundary."""
        to_enqueue = None
        with self._buffer_lock:
            if not self._finished:
                rest, self._buffer = self._buffer, ""
                if rest.strip():
                    to_enqueue = rest.strip()
        if to_enqueue:
            self._enqueue(to_enqueue)

    def finish(self):
        """Flush the remainder; the worker stays alive for announcements."""
        to_enqueue = None
        with self._buffer_lock:
            if self._finished:
                return
            self._finished = True
            rest, self._buffer = self._buffer, ""
            if rest.strip():
                to_enqueue = rest.strip()
        if to_enqueue:
            self._enqueue(to_enqueue)

    def close(self):
        """Stop the worker once the queue drains (no more announcements)."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(_STREAM_STOP)

    def pending_text(self):
        """Sentences queued (or buffered) but not yet spoken (F35 pause).

        The queue is snapshotted and restored, so the playback order is kept:
        pausing only takes away the DEVICE, it does not drop the text.
        """
        parts = []
        with self._pause_cv:
            held = self._held_chunk
        if held:
            # [P1-07] The chunk mid-playback (or held at the pause gate) has
            # already left the queue but has not been spoken.
            parts.append(str(held))
        with self._buffer_lock:
            buffered = self._buffer.strip()
        if buffered:
            parts.append(buffered)
        items = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is _STREAM_STOP:
                self._queue.put(item)
                break
            items.append(item)
        for item in items:
            self._queue.put(item)
        parts.extend(str(item) for item in items if item)
        return " ".join(part for part in parts if part).strip()

    def _enqueue(self, sentence):
        # [S12] Safety net: strip markdown/URLs from each finished chunk so the
        # streaming path never voices formatting either.
        sentence = strip_markdown_for_speech(sentence)
        # [P1-01] An empty utterance is never handed to TTS: it would either
        # error or leave a gap, and the contract is that every chunk contains a
        # complete word.
        if not sentence or not str(sentence).strip():
            return
        # Short first chunk optimization: cap first enqueued chunk to ~120 chars
        # Protect flag check+set under _buffer_lock for thread-safety
        needs_first_split = False
        with self._buffer_lock:
            is_first = not self._first_chunk_enqueued
            if is_first and len(sentence) > 120:
                needs_first_split = True
                self._first_chunk_enqueued = True
            elif is_first:
                self._first_chunk_enqueued = True
        if needs_first_split:
            limit = 120
            comma_idx = sentence.rfind(",", 0, limit)
            if comma_idx >= 30:
                first = sentence[: comma_idx + 1].strip()
                remainder = sentence[comma_idx + 1 :].strip()
            else:
                space_idx = sentence.rfind(" ", 0, limit)
                if space_idx != -1:
                    first = sentence[:space_idx].strip()
                    remainder = sentence[space_idx + 1 :].strip()
                else:
                    first = sentence[:limit].strip()
                    remainder = sentence[limit:].strip()
            if len(first) > FISH_TTS_CHAR_LIMIT:
                for chunk in split_speech_chunks(first):
                    if chunk.strip():
                        self._enqueue(chunk.strip())
            else:
                self._spoken_any = True
                pending_ahead = self._queue.qsize()
                self._queue.put(first)
                self._start_worker()
                if not self._ws_mode:
                    _prefetch_tts_audio(first, plays_next=(pending_ahead == 0),
                                        provider=self._tts_engine())
            if remainder:
                self._enqueue(remainder)
            return
        if len(sentence) > _tts_char_limit(self._tts_engine()):
            for chunk in split_speech_chunks(sentence):
                if chunk.strip():
                    self._enqueue(chunk.strip())
            return
        self._spoken_any = True
        pending_ahead = self._queue.qsize()
        self._queue.put(sentence)
        self._start_worker()
# Start synthesising this sentence now so the playback loop (which is
        # still speaking the previous one) finds it ready when its turn comes.
        # [P0-05] The "is this the sentence about to play?" half of the rule is
        # passed in; `_prefetch_tts_audio` owns the decision, so the engine
        # choice and the timing rule cannot drift apart.
        # [S7] Under the live WebSocket engine this sentence is fed to the
        # session instead — an HTTP prefetch would synthesise the same audio
        # a second time on the other transport.
        if not self._ws_mode:
            _prefetch_tts_audio(sentence, plays_next=(pending_ahead == 0),
                                provider=self._tts_engine())

    def _start_worker(self):
        with _state_lock:
            if self._worker is None:
                self._worker = threading.Thread(target=self._playback_loop, daemon=True)
                self._worker.start()

    def _queue_has_audio(self):
        """True when at least one real utterance is still queued.

        The ``_STREAM_STOP`` sentinel is not audio: ``close()`` means "stop the
        worker once the queue drains", so a queue holding only the sentinel has
        nothing left to speak. Peeked non-destructively (the only consumer is
        this speaker's own playback thread).
        """
        with self._queue.mutex:
            return any(item is not _STREAM_STOP for item in self._queue.queue)

    # ── [P1-07] pause / resume ─────────────────────────────────────────

    @property
    def paused(self):
        """True while this speaker is parked in a resumable pause."""
        with self._pause_cv:
            return self._paused

    def pause(self):
        """Pause this reply: keep the queue AND the playback position.

        Returns True when there was something to resume. Distinct from
        ``close()``/``stop_speaking()``: nothing is discarded and the generation
        is not bumped, so the sentences already queued keep their place. The
        actor is paused too, which cuts the device and parks its play loop at
        the exact byte it had reached.
        """
        with self._pause_cv:
            self._paused = True
            self._pause_cv.notify_all()
        cursor = None
        try:
            cursor = _pause_actor_playback()
        except Exception:
            cursor = None
        with self._buffer_lock:
            buffered = bool(self._buffer.strip())
        queued = self._queue_has_audio()
        with self._pause_cv:
            self._pause_cursor = cursor
        return bool(cursor is not None or buffered or queued)

    def resume(self):
        """Release a pause so the worker may start the next chunk.

        The AUDIO is resumed by :func:`resume_local_playback` (the actor owns
        the parked play loop); this only opens the worker's gate. Safe no-op
        when nothing was paused.
        """
        with self._pause_cv:
            if not self._paused:
                return False
            self._paused = False
            self._pause_cv.notify_all()
        return True

    def _await_resume(self):
        """Park while paused. Returns False when the reply must NOT continue.

        False means a real STOP (not a resume) superseded the pause — the stop
        bumped the generation, so the remainder must never be replayed and the
        worker must not start the held chunk.
        """
        with self._pause_cv:
            while (self._paused
                   and not self._closed
                   and _is_current_generation(self._generation)):
                self._pause_cv.wait(0.05)
            if not self._paused:
                return True
            # Still paused but this reply is gone (stopped or closed): drop the
            # pause so a stale gate can never park the worker for good.
            self._paused = False
            return False

    def _pass_pause_gate(self, sentence):
        """Park at the pause gate while holding *sentence*. True when it may play.

        [P1-07] The chunk has already left the queue, so it is remembered as the
        HELD chunk: ``pending_text`` must be able to report it while paused.
        """
        with self._pause_cv:
            self._held_chunk = sentence
        allowed = self._await_resume()
        with self._pause_cv:
            self._held_chunk = None
        return allowed

    def _reply_session_over(self):
        """True once this reply is COMPLETE and nothing is left to speak.

        [P1-06] The speaking flag is scoped to the reply *session*, never to the
        queue's momentary occupancy. The session is over when the reply text is
        finished (``finish()``) or the speaker was closed, AND the queue and the
        pending buffer are both empty — i.e. the last chunk has actually been
        played (``_speak_chunk`` blocks for the duration of its chunk).
        """
        if not (self._finished or self._closed):
            return False
        if self._queue_has_audio():
            return False
        with self._buffer_lock:
            return not self._buffer.strip()

    def _start_ws_playback(self, session):
        """[S7] Start the one continuous playback stream for this reply."""
        self._ws_mode = True

        def _run():
            try:
                self._ws_outcome = play_ws_reply(
                    session,
                    is_current=lambda: _is_current_generation(self._generation),
                    earcon=play_reply_start_earcon if not self._played_first
                    else None)
            except Exception:
                print("Fish WS playback error:")
                traceback.print_exc()
                self._ws_outcome = {"audio_started": ws_audio_reached_device(),
                                    "failed": True}

        self._ws_thread = threading.Thread(target=_run,
                                            name="fish-ws-playback",
                                            daemon=True)
        self._ws_thread.start()
        return self._ws_thread

    def _finish_ws_playback(self, timeout=8.0, handback_if_silent=True):
        """[S7] Close the text stream, wait for the audio to finish playing
        and close the session.

        When the session produced NO audio at all and the reply is still
        current, every sentence it consumed is put back on the queue so the
        per-sentence ladder speaks the whole reply instead (a reply that
        already spoke is never replayed; a stop never requeues anything).
        """
        session = fish_ws_session()
        if session is not None:
            try:
                session.finish_text()
            except Exception:
                pass
        thread = self._ws_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._ws_mode = False
        self._ws_thread = None
        try:
            fish_ws_end_reply()
        except Exception:
            pass
        outcome = self._ws_outcome or {"audio_started": ws_audio_reached_device(),
                                       "failed": False}
        fed = self._ws_fed
        self._ws_fed = []
        # ONE session per reply: whatever happened to it, this reply is done
        # with the WebSocket engine and continues on the per-sentence ladder.
        self._ws_disabled = True
        if not handback_if_silent and ws_audio_reached_device():
            # The session's audio reached the device and the stream then ended.
            # Continuing the rest of the reply on another engine would switch
            # voices mid-sentence and could repeat what was just spoken, so the
            # reply ends here (FIX3: never resume or replay after playback).
            self._ws_ended = True
        if (handback_if_silent
                and fed
                and not ws_audio_reached_device()
                and _is_current_generation(self._generation)
                and not self._closed):
            pending = []
            while True:
                try:
                    pending.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            for text in fed + pending:
                self._queue.put(text)
        return outcome

    def _playback_loop(self):
        global is_speaking, _current_text

        def set_speaking_state(speaking):
            global is_speaking
            with _state_lock:
                changed = is_speaking != speaking
                is_speaking = speaking
                self._active = speaking
            if changed:
                listener_state.set_speaking(speaking)

        try:
            while True:
                try:
                    item = self._queue.get(timeout=STREAM_POLL_SECONDS)
                except queue.Empty:
                    # Buffer has content but no new delta: either it ends at a
                    # complete sentence and the model has paused, or nothing has
                    # arrived for long enough that we stop waiting.
                    to_flush = None
                    with self._buffer_lock:
                        if self._buffer.strip() and not self._finished and not self._closed:
                            idle = time.monotonic() - self._last_feed_ts
                            settled = (
                                idle >= PUNCTUATION_SETTLE_SECONDS
                                and self._buffer.rstrip()[-1] in _SENTENCE_END_CHARS
                            )
                            if settled or idle >= STALL_FLUSH_SECONDS:
                                to_flush, self._buffer = self._buffer, ""
                    if to_flush and to_flush.strip():
                        self._enqueue(to_flush.strip())
                    # [P1-06] A reply can complete with nothing left to
                    # enqueue — the final delta was already voiced and
                    # ``finish()`` arrived with an empty buffer. The session
                    # ends HERE as well, otherwise the speaking flag would stay
                    # set after the audio stopped.
                    if self._reply_session_over():
                        # [S7] The reply's last audio can be streaming even
                        # though the queue is empty, so the session is closed
                        # (and its audio awaited) here before the flag clears.
                        if self._ws_mode:
                            self._finish_ws_playback()
                        set_speaking_state(False)
                    continue
                if item is _STREAM_STOP:
                    if self._ws_mode:
                        self._finish_ws_playback(handback_if_silent=False)
                    break
                if not _is_current_generation(self._generation):
                    continue
                sentence = item
                if not self._active:
                    _current_text = sentence
                    listener_state.set_remaining("")
                    set_speaking_state(True)
                # [P1-07] A paused reply starts no new chunk: the gate parks here
                # until "continue", so the queue keeps its place and the next
                # sentence never plays over the pause. A real STOP opens the gate
                # with False (the generation moved on), and that chunk is
                # deliberately dropped — a stop discards, a pause preserves.
                if not self._pass_pause_gate(sentence):
                    continue

                # [S7] Live Fish session: the first sentence opens ONE WebSocket
                # for this reply and every following sentence is FED to it —
                # the whole reply plays as a single continuous utterance
                # instead of one HTTP request per sentence.
                session = None
                if self._ws_ended:
                    continue
                if not self._ws_mode and not self._ws_disabled:
                    if self._tts_engine() == "fish":
                        fish_ws_begin_reply()
                        session = fish_ws_session()
                    if session is not None:
                        self._ws_fed = []
                        self._start_ws_playback(session)
                if self._ws_mode:
                    if session is None:
                        session = fish_ws_session()
                    if session is not None and session.failed():
                        if not ws_audio_reached_device():
                            # Nothing was spoken: hand the whole reply
                            # back to the per-sentence ladder, in order.
                            self._ws_fed.append(sentence)
                            self._finish_ws_playback()
                            continue
                        # Audio already reached the device — never replay.
                        self._finish_ws_playback(handback_if_silent=False)
                        if self._reply_session_over():
                            set_speaking_state(False)
                        continue
                    if (session is not None
                            and self._ws_thread is not None
                            and not self._ws_thread.is_alive()):
                        # The stream is over without a reported failure (the
                        # server closed it). Whatever the session SPOKE must
                        # never replay; what it never spoke goes back to the
                        # ladder, this sentence first so the order holds.
                        if ws_audio_reached_device():
                            self._finish_ws_playback(handback_if_silent=False)
                            if self._reply_session_over():
                                set_speaking_state(False)
                            continue
                        self._ws_fed.append(sentence)
                        self._finish_ws_playback()
                        continue
                if self._ws_mode:
                    self._ws_fed.append(sentence)
                    session.feed(sentence)
                    self._played_first = True
                    if self._reply_session_over():
                        self._finish_ws_playback()
                        set_speaking_state(False)
                    continue
                _speak_chunk(sentence, self._generation,
                             is_first_chunk=not self._played_first,
                             provider=self._tts_engine())
                self._played_first = True
                # [P1-06] The speaking flag means "Jarvis is mid-reply" for the
                # WHOLE reply session. It is NOT derived from momentary queue
                # occupancy: clearing it the instant the queue happened to drain
                # made the flag flicker off in the gap between two sentences of
                # the SAME reply, which is what made the listener's old
                # "ignore while speaking" gate unpredictable (sometimes an
                # interruption landed, sometimes the utterance vanished without
                # a trace). The session ends only once the reply is COMPLETE
                # (``finish()``/``close()``) and everything queued has actually
                # been played — see ``_reply_session_over``.
                if self._reply_session_over():
                    set_speaking_state(False)
        except Exception:
            print("Stream voice error:")
            traceback.print_exc()
        finally:
            if self._ws_mode:
                try:
                    self._finish_ws_playback(handback_if_silent=False)
                except Exception:
                    pass
            should_signal_ready = False
            with _state_lock:
                if self._active and _speech_generation == self._generation:
                    is_speaking = False
                    should_signal_ready = True
                self._active = False
            if should_signal_ready:
                listener_state.set_speaking(False)
                play_ready_earcon()
