"""Wake/command separation for the passive watcher (F36).

Before this module the watcher treated the whole captured phrase as a wake
decision only: "wake up jarvis and search for cats" matched a wake phrase and
then **discarded** the "and search for cats" tail, forcing the user to wait
for the full stack to boot before re-speaking the command.

[Fable-5 F36] "Separate wake detection from command transcription":

* **Wake detection** runs on a lightweight keyword-spotting path (fuzzy
  phrase match, or ``openwakeword`` when installed and ``JARVIS_WAKE_ENGINE``
  selects it / ``auto`` resolves to it) with an optional **online Whisper
  verification** of the *pre-roll* audio when ``JARVIS_WAKE_ONLINE_VERIFY``
  is set — false positives on background TV chatter are confirmed before the
  expensive stack boots.
* **Command transcription** is the *rest* of the phrase: the tokens after the
  matched wake window are extracted and forwarded to the running backend
  (POST ``/ask``) so a wake-plus-command phrase acts on both parts.
* **Pre-roll** keeps the last ``JARVIS_WAKE_PRE_ROLL_SECONDS`` of captured
  audio in a bounded ring, so verification and future transcript windows can
  overlap the phrase start (the wake word is often clipped by listen-onset).

Hardware/model-dependent pieces ([ASSUMPTION]-flagged in the audit) degrade
gracefully: an uninstalled keyword-spot model falls back to the fuzzy path
that already ships, and verification is off by default.
"""

import json
import os
import threading
import time
import uuid
from collections import deque

from backend.config import (
    WAKE_ENGINE,
    WAKE_MODELS_DIR,
    WAKE_ONLINE_VERIFY,
    WAKE_PRE_ROLL_SECONDS,
)

KEYWORD_SPOT_SAMPLE_RATE = 16000

# ── F34/F36: the one explicit cloud-STT egress policy ───────────────────────
# Both the wake path (watcher) and the active-conversation path (listener)
# consult this single policy before any audio may be sent to a cloud STT:
#   * "on" (default)  — cloud engines may be used (existing behaviour);
#   * "off"/"local"   — LOCAL ONLY: no audio ever leaves the machine. The
#     local whisper daemon is the only engine that may be tried, and if it is
#     unavailable the utterance is simply not transcribed.
# Configured with JARVIS_STT_CLOUD_POLICY (on|off|local|local-only) or the
# JARVIS_STT_LOCAL_ONLY=1 shorthand. Read per call so a policy change takes
# effect on the next utterance without a restart.
CLOUD_STT_POLICY_ENV = "JARVIS_STT_CLOUD_POLICY"
LOCAL_ONLY_ENV = "JARVIS_STT_LOCAL_ONLY"
CLOUD_STT_OFF_VALUES = frozenset((
    "off", "0", "no", "false", "disabled", "local", "local-only",
    "local_only", "localonly", "never",
))


def cloud_stt_policy():
    """The active cloud-egress policy: ``"on"`` or ``"off"`` (local-only)."""
    value = (os.getenv(CLOUD_STT_POLICY_ENV)
             or os.getenv("JARVIS_STT_CLOUD")
             or "").strip().lower()
    if value in CLOUD_STT_OFF_VALUES:
        return "off"
    local_only = (os.getenv(LOCAL_ONLY_ENV, "") or "").strip().lower()
    if local_only in ("1", "true", "yes", "on"):
        return "off"
    return "on"


def cloud_stt_allowed():
    """True when audio may be sent to a cloud STT under the current policy."""
    return cloud_stt_policy() != "off"


def _norm(text):
    import re

    return re.sub(r"\s+", " ", str(text or "").strip().lower())


# ── command extraction ────────────────────────────────────────────────────
def _is_wake(text):
    try:
        from backend.watcher import is_wake_word

        return is_wake_word(text)
    except Exception:
        return _norm(text) in {"wake up jarvis", "jarvis wake up", "jarvis"}


def _raw_tokens(raw):
    """``[(text, start, end)]`` of the RAW phrase — offsets into the original.

    Normalization is only ever used to DECIDE (``_is_wake``); every payload
    handed onward is a byte-exact slice of the original transcription, so
    case, punctuation, paths, URLs, flags, quotes and whitespace survive the
    wake route untouched (F12).
    """
    import re

    return [(m.group(0), m.start(), m.end())
            for m in re.finditer(r"\S+", str(raw or ""))]


def _wake_vocabulary():
    """Every token that appears in a known wake phrase (EN + HI + patterns)."""
    try:
        from backend.watcher import (
            JARVIS_VARIANTS,
            WAKE_PATTERNS,
            WAKE_VARIANTS_EN,
            WAKE_VARIANTS_HI,
        )
    except Exception:
        return {"wake", "up", "jarvis"}
    words = set()
    for phrase in (list(WAKE_VARIANTS_EN) + list(WAKE_VARIANTS_HI)
                   + list(JARVIS_VARIANTS) + list(WAKE_PATTERNS)):
        words.update(_norm(phrase).split())
    return words


def _is_wake_only_tail(tail):
    """True when a candidate tail is just more wake phrase, not a command.

    "jarvis wake up" and "jarvis chalu ho" are wake-only utterances: the
    shortest-span extractor used to hand the trailing "wake up"/"chalu ho"
    onward as a command, which created a task from a wake phrase. A tail
    made only of wake vocabulary is not a command.
    """
    if not str(tail or "").strip():
        return True
    if _is_wake(tail):
        return True
    tokens = _norm(tail).split()
    if not tokens:
        return True
    return all(token in _wake_vocabulary() for token in tokens)


def _strip_wake_lead_in(tail, vocab):
    """Drop a multi-word wake lead-in from the start of a command tail.

    "jarvis wake up and search for cats" leaves the tail "wake up and search
    for cats": a leading run of TWO OR MORE wake words is a continuation of
    the wake phrase, so it is consumed and the literal command starts at
    "and search for cats". A SINGLE leading wake-vocabulary word is kept —
    "start", "listen" and "activate" are ordinary command verbs too, so
    "Jarvis, start the timer" must keep its "start".
    """
    tokens = _raw_tokens(tail)
    if len(tokens) < 2:
        return tail
    run = 0
    while run < len(tokens) and _norm(tokens[run][0]) in vocab:
        run += 1
    if run < 2 or run >= len(tokens):
        return tail
    return tail[tokens[run][1]:].strip()


def extract_command(wake_transcript, candidates=()):
    """The LITERAL command tail after the matched wake window (F12/F36).

    Returns the tail ("" when the phrase was wake-only or the matched window
    could not be located — never raises). The tail is a byte-exact slice of
    the raw transcription: case, punctuation, URLs, flags, quotes, newlines
    and indentation are preserved, because typing is never reconstructed
    from normalized tokens (F12).

    The wake span is the SHORTEST wake window (leftmost), so a command verb
    that merely looks like a wake word ("Jarvis, start the timer") stays in
    the command. A tail that is itself only more wake phrase ("jarvis wake
    up", "jarvis chalu ho") yields NO command, and a multi-word wake
    lead-in ("jarvis wake up and search for cats") is consumed rather than
    forwarded as part of the command.
    """
    pool = [str(c) for c in (candidates or ()) if str(c or "").strip()]
    wake_raw = str(wake_transcript or "")
    if wake_raw.strip() and wake_raw not in pool:
        pool.insert(0, wake_raw)
    # Prefer the LONGEST candidate: a fuzzy whisper often collapses "wake up
    # jarvis" and "wake up jarvis open chrome" into near-duplicates, and the
    # collapsed variant usually carries the command tail.
    pool.sort(key=len, reverse=True)
    vocab = _wake_vocabulary()
    for candidate in pool:
        tokens = _raw_tokens(candidate)
        if len(tokens) < 2:
            continue
        span_end = None
        for length in range(1, min(6, len(tokens)) + 1):
            for start in range(0, len(tokens) - length + 1):
                end = start + length - 1
                window = candidate[tokens[start][1]:tokens[end][2]]
                if _is_wake(window):
                    span_end = end
                    break
            if span_end is not None:
                break
        if span_end is None:
            continue
        tail = candidate[tokens[span_end][2]:].strip()
        if _is_wake_only_tail(tail):
            # A wake phrase is not a task.
            return ""
        tail = _strip_wake_lead_in(tail, vocab)
        if not tail:
            return ""
        return tail
    return ""


def _tail_or_empty(candidate, tail_start):
    """The literal tail after a matched wake span, or "" when wake-only.

    Only the separator whitespace around the tail is stripped; everything
    the user typed inside the payload is byte-exact.
    """
    tail = candidate[tail_start:].strip()
    if _is_wake_only_tail(tail):
        return ""
    return tail



# ── pre-roll ──────────────────────────────────────────────────────────────
class WakePreRollBuffer:
    """Bounded ring of recently captured mic audio.

    ``feed(audio)`` accepts an ``sr.AudioData`` (the watcher already holds
    one per listen); ``drain()`` empties the ring as a single AudioData whose
    duration is capped at ``JARVIS_WAKE_PRE_ROLL_SECONDS`` (0 disables).
    """

    def __init__(self, pre_roll_seconds=WAKE_PRE_ROLL_SECONDS):
        self._seconds = max(0.0, float(pre_roll_seconds))
        self._chunks = deque()
        self._lock = threading.Lock()

    def feed(self, audio):
        if audio is None or self._seconds <= 0:
            return
        frame = bytes(getattr(audio, "frame_data", b"") or b"")
        if not frame:
            return
        rate = int(getattr(audio, "sample_rate", 16000) or 16000)
        width = int(getattr(audio, "sample_width", 2) or 2)
        cap = int(rate * width * self._seconds)
        with self._lock:
            self._chunks.append((frame, rate, width))
            total = sum(len(c) for c, _, _ in self._chunks)
            while total > cap and self._chunks:
                old = self._chunks.popleft()
                total -= len(old[0])

    def drain(self):
        with self._lock:
            chunks = list(self._chunks)
            self._chunks.clear()
        if not chunks:
            return None
        rate, width = chunks[0][1], chunks[0][2]
        joined = b"".join(c for c, _, _ in chunks)
        if self._seconds > 0:
            cap = int(rate * width * self._seconds)
            joined = joined[-cap:] if len(joined) > cap else joined
        # sr.AudioData lazily: the buffer is usable from processes/tests
        # that never import speech_recognition.
        try:
            import speech_recognition as sr

            return sr.AudioData(joined, rate, width)
        except Exception:
            return None

    def __len__(self):
        with self._lock:
            return sum(len(c) for c, _, _ in self._chunks)


_pre_roll = WakePreRollBuffer()


def pre_roll():
    return _pre_roll


def reset_pre_roll():
    with _pre_roll._lock:
        _pre_roll._chunks.clear()


# ── keyword spotting ──────────────────────────────────────────────────────
_kws_model = None
_kws_model_lock = threading.Lock()
_kws_model_failed = False


def _load_kws_model():
    """Guarded openwakeword load ([ASSUMPTION]: spotting quality is a
    hardware question; an uninstalled model falls back to the shipped
    fuzzy path)."""
    global _kws_model, _kws_model_failed
    if _kws_model is not None or _kws_model_failed:
        return _kws_model
    if WAKE_ENGINE in ("fuzzy", "phrase", "off", "0", "none"):
        return None
    with _kws_model_lock:
        if _kws_model is not None or _kws_model_failed:
            return _kws_model
        try:
            from openwakeword.model import Model

            if WAKE_MODELS_DIR:
                _kws_model = Model(wakeword_models=[WAKE_MODELS_DIR])
            else:
                _kws_model = Model()
            print("[WAKE] openwakeword keyword-spot model loaded")
        except Exception as exc:
            _kws_model_failed = True
            print(f"[WAKE] keyword-spot unavailable ({exc}) - fuzzy path")
    return _kws_model


def resolve_engine():
    """Report which wake path resolves at runtime (tests/diagnostics)."""
    return "openwakeword" if _load_kws_model() is not None else "fuzzy"


def keyword_spot(audio):
    """True when the keyword-spot model fires on *audio*.

    Returns False whenever the engine is unavailable — callers keep their
    existing fuzzy wake semantics untouched.
    """
    model = _load_kws_model()
    if model is None or audio is None:
        return False
    try:
        import numpy as np

        frame = bytes(getattr(audio, "frame_data", b"") or b"")
        if len(frame) < 1280:
            return False
        rate = int(getattr(audio, "sample_rate", 16000) or 16000)
        if rate != KEYWORD_SPOT_SAMPLE_RATE:
            arr = np.frombuffer(frame, dtype=np.int16)
            step = max(1, round(rate / float(KEYWORD_SPOT_SAMPLE_RATE)))
            frame = arr[::step].astype(np.int16).tobytes()
        scores = model.predict(frame) or {}
        return any(float(v) >= 0.5 for v in scores.values())
    except Exception:
        return False


# ── online verification of the pre-roll ───────────────────────────────────
def _as_wav_bytes(audio):
    """The local whisper daemon's wire format for *audio*.

    [F36] The verification path used to hand the daemon an ``sr.AudioData``
    object where it expects WAV BYTES — an incompatible interface that made
    every verification request a no-op. This converts explicitly, and
    prefers ``get_wav_data()`` when the object provides it.
    """
    if audio is None:
        return None
    getter = getattr(audio, "get_wav_data", None)
    if callable(getter):
        try:
            wav = getter()
        except Exception:
            wav = None
        if wav:
            return bytes(wav)
    frame = bytes(getattr(audio, "frame_data", b"") or b"")
    return frame or None


def online_verify():
    """Confirm the wake hit by LOCALLY transcribing the drained pre-roll.

    True → proceed with launch. Skipped (True) when verification is disabled;
    the pre-roll is always drained so the next phrase starts clean.

    [F36] Verification is bounded local evidence: it runs on the local
    whisper daemon only (never a cloud STT), and a verification ERROR now
    FAILS CLOSED — an unverifiable wake hit does not boot the stack. The
    previous code passed an incompatible object to the daemon and then
    proceeded on the error, so verification could never actually brake a
    false positive.
    """
    if not WAKE_ONLINE_VERIFY:
        return True
    audio = _pre_roll.drain()
    if audio is None:
        return True
    wav = _as_wav_bytes(audio)
    if not wav:
        return True
    try:
        from backend.watcher import is_wake_word, _transcribe_with_daemon

        result = _transcribe_with_daemon(wav)
        raw = result[0] if result else None
        if not raw:
            # No local transcript at all: nothing confirms the wake hit.
            print("[WAKE] Verification could not transcribe the pre-roll - "
                  "not launching")
            return False
        if is_wake_word(str(raw)):
            return True
        print(f"[WAKE] Verification rejected launch: {raw!r}")
        return False
    except Exception as exc:
        print(f"[WAKE] Verification error ({exc}) - not launching")
        return False


# ── wake-command forwarding ───────────────────────────────────────────────
_forwarded_lock = threading.Lock()
_forwarded_ids = deque(maxlen=64)


def reset_forward_state():
    """Forget which wake requests were forwarded (tests/diagnostics)."""
    with _forwarded_lock:
        _forwarded_ids.clear()


def forward_wake_command(command, timeout=6.0, request_id=None,
                         wait_ready=False, ready_timeout=20.0):
    """POST a wake-phrase command tail to the running backend, ONCE.

    "wake up jarvis and search for cats" boots the stack AND acts on the
    tail. The literal *command* is forwarded byte-exact (case, punctuation,
    flags, paths and URLs intact — F12); never normalized.

    F36 forwarding guarantees:
      * an authenticated request — the per-launch ``X-Jarvis-Token`` the
        supervisor minted travels on the header, like every other local
        command;
      * an identified request — a stable ``request_id`` is minted (or taken
        from the caller) and carried in the payload, and the same id is
        forwarded AT MOST ONCE, so a launch retry or a dropped connection can
        never execute one spoken command tail twice;
      * readiness handling — with *wait_ready* the caller asks this function
        to wait (bounded by *ready_timeout*) for the backend to answer
        ``/health`` before the single POST, instead of firing into a port
        that is still booting.

    Best-effort: returns False (never raises) when nothing could be sent.
    """
    text = str(command or "").strip()
    if not text:
        return False
    request_id = str(request_id or "").strip() or "wake-%s" % uuid.uuid4().hex[:12]
    with _forwarded_lock:
        if request_id in _forwarded_ids:
            print(f"[WAKE] Wake command {request_id} already forwarded - ignoring")
            return False
        _forwarded_ids.append(request_id)

    if wait_ready:
        try:
            from backend.watcher import wait_for_backend_ready

            wait_for_backend_ready(timeout_seconds=max(0.0, float(ready_timeout)))
        except Exception:
            pass

    try:
        from backend.config import BACKEND_PORT
        from backend.services import local_auth
        from urllib.request import Request, urlopen

        payload = json.dumps({
            "message": text,
            "request_id": request_id,
            "origin": "wake",
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        token = os.getenv("JARVIS_LOCAL_TOKEN", "")
        if token:
            headers[local_auth.HEADER] = token
        request = Request(
            f"http://127.0.0.1:{BACKEND_PORT}/ask",
            data=payload,
            method="POST",
            headers=headers,
        )
        with urlopen(request, timeout=timeout) as resp:
            json.loads(resp.read().decode("utf-8"))
        print(f"[WAKE] Forwarded wake command ({request_id}): {text!r}")
        return True
    except Exception as exc:
        print(f"[WAKE] Could not forward wake command ({exc})")
        return False
