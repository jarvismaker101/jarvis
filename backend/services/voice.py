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
    prefetch_fish_audio,
    speak_fish_audio,
    stop_fish_audio,
    warm_up_fish_tts,
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


def clean_text(text):
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


def stop_speaking():
    global is_speaking, _speech_generation, _engine

    try:
        stop_elevenlabs()
    except Exception:
        pass

    try:
        stop_fish_audio()
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

    if should_signal_ready:
        play_ready_earcon()

    print("[VOICE] Speech stopped")


def pause_speaking():
    """Stop the current playback but KEEP the remainder for a resume (F35).

    Distinct from ``stop_speaking()``: a pause makes the remaining narration
    resumable (``listener_state.set_remaining``), so "continue" has a real
    target instead of depending on whatever the stream happened to leave
    behind. Returns True when there was something left to resume.
    """
    remaining = pending_speaking_text()
    stop_speaking()
    if remaining:
        listener_state.set_remaining(remaining)
        print(f"[VOICE] Speech paused — {len(remaining)} chars remain")
        return True
    print("[VOICE] Speech paused — nothing remained")
    return False


def resume_speaking():
    """Speak the text a pause (or an interruption) left unplayed (F35).

    Returns the resumed text ("" when there was nothing to resume), so a
    caller can report the truth instead of silently doing nothing.
    """
    remaining = listener_state.pop_remaining()
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


def _cloud_tts_ladder():
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
    order = ("gtts", "fish") if _resolve_tts_provider() == "gtts" else ("fish", "gtts")
    return [engines[name] for name in order]


def _tts_char_limit():
    """Chunking limit for the engine that will actually speak."""
    if _resolve_tts_provider() == "gtts":
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


def _prefetch_tts_audio(text, plays_next=False):
    """Warm the engine that will actually speak the NEXT sentence.

    Prefetching the wrong engine wastes a synthesis call and leaves the real
    one cold, so this follows the same selection as `_cloud_tts_ladder`. The
    selection is resolved EXACTLY ONCE per call, so a settings change landing
    mid-call cannot warm a different engine than the one that was chosen.

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
    warm = _prefetch_implementations().get(_resolve_tts_provider())
    if warm is None:
        return
    warm(text)


def _speak_chunk(chunk, generation, is_first_chunk=False):
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
    for _name, engine, limit in _cloud_tts_ladder():
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
            chunks = split_speech_chunks(clean)
            is_first_chunk = True
            for index, chunk in enumerate(chunks):
                if not _is_current_generation(generation):
                    return
                if index + 1 < len(chunks):
                    # chunks[index] plays first, so the next chunk is never the
                    # sentence about to play and is always safe to warm.
                    _prefetch_tts_audio(chunks[index + 1])
                _speak_chunk(chunk, generation, is_first_chunk=is_first_chunk)
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
        # F31: the reasoning channel of this turn, preserved verbatim and
        # never handed to TTS (only final-answer text is narrated).
        self._reasoning = ""

        with _state_lock:
            _speech_generation += 1
            self._generation = _speech_generation

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
            parts = self._sentence_re.split(self._buffer)
            if len(parts) > 1:
                self._buffer = parts[-1]
                for part in parts[:-1]:
                    if part.strip():
                        to_enqueue.append(part.strip())
            if len(self._buffer) >= self._min_flush:
                long, self._buffer = self._buffer, ""
                if long.strip():
                    to_enqueue.append(long.strip())
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
                _prefetch_tts_audio(first, plays_next=(pending_ahead == 0))
            if remainder:
                self._enqueue(remainder)
            return
        if len(sentence) > _tts_char_limit():
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
        _prefetch_tts_audio(sentence, plays_next=(pending_ahead == 0))

    def _start_worker(self):
        with _state_lock:
            if self._worker is None:
                self._worker = threading.Thread(target=self._playback_loop, daemon=True)
                self._worker.start()

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
                    item = self._queue.get(timeout=0.3)
                except queue.Empty:
                    # Stall flush: buffer has content but no new delta for ~300ms
                    to_flush = None
                    with self._buffer_lock:
                        if self._buffer.strip() and not self._finished and not self._closed:
                            if time.monotonic() - self._last_feed_ts >= 0.3:
                                to_flush, self._buffer = self._buffer, ""
                    if to_flush and to_flush.strip():
                        self._enqueue(to_flush.strip())
                    continue
                if item is _STREAM_STOP:
                    break
                if not _is_current_generation(self._generation):
                    continue
                sentence = item
                if not self._active:
                    _current_text = sentence
                    listener_state.set_remaining("")
                    set_speaking_state(True)
                _speak_chunk(sentence, self._generation, is_first_chunk=not self._played_first)
                self._played_first = True
                # Queue drained — clear the speaking flag so the listener
                # accepts the next command. Previously the worker stayed
                # idle-blocked on the queue with is_speaking stuck True, so
                # the voice listener dropped every command after the first
                # reply (voice worked once, then went deaf until restart).
                if self._queue.empty():
                    set_speaking_state(False)
        except Exception:
            print("Stream voice error:")
            traceback.print_exc()
        finally:
            should_signal_ready = False
            with _state_lock:
                if self._active and _speech_generation == self._generation:
                    is_speaking = False
                    should_signal_ready = True
                self._active = False
            if should_signal_ready:
                listener_state.set_speaking(False)
                play_ready_earcon()
