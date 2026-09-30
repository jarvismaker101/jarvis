import base64
import json
import os
import re
from urllib.request import Request, urlopen

import requests
import speech_recognition as sr

from backend.config import GROQ_API_KEY

GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
GROQ_STT_URL = os.getenv(
    "GROQ_STT_URL",
    "https://api.groq.com/openai/v1/audio/transcriptions",
)
INWORLD_STT_API_KEY = os.getenv("INWORLD_STT_API_KEY")
INWORLD_STT_URL = os.getenv(
    "INWORLD_STT_URL",
    "https://api.inworld.ai/stt/v1/transcribe",
)
INWORLD_STT_MODEL = os.getenv("INWORLD_STT_MODEL", "inworld/inworld-stt-1")
#: [P0-03] Inworld's budget, split into (connect, read) so a wedged TCP connect
#: cannot consume the whole read allowance. With one STT engine per turn these
#: two numbers ARE the worst case of a turn, so they are explicit and tunable
#: instead of an anonymous ``timeout=(5, 15)``.
INWORLD_STT_CONNECT_TIMEOUT_SECONDS = float(
    os.getenv("JARVIS_INWORLD_STT_CONNECT_TIMEOUT", "3"))
INWORLD_STT_READ_TIMEOUT_SECONDS = float(
    os.getenv("JARVIS_INWORLD_STT_READ_TIMEOUT", "10"))
INWORLD_STT_TIMEOUT = (
    INWORLD_STT_CONNECT_TIMEOUT_SECONDS,
    INWORLD_STT_READ_TIMEOUT_SECONDS,
)
LOCAL_WHISPER_PORT = int(os.getenv("JARVIS_WHISPER_PORT", "8767"))
LOCAL_WHISPER_URL = f"http://127.0.0.1:{LOCAL_WHISPER_PORT}"

# ── STT hallucination gate ──────────────────────────────────────────────────
# STT engines do not stay silent on non-speech: over noise, TTS playback echo
# and near-silent captures they emit memorised filler instead of "nothing".
# Three shapes were observed live and all reached the brain as user speech:
#   * the wake-word biasing prompt echoed back verbatim
#     ("jarvis, wake up, jervis, utho, jago, chalu"),
#   * one token looped ("chalu, chalu, chalu, ..."),
#   * a memorised silence phrase ("a ver si te acuerdas de esto" — Whisper's
#     most documented hallucination).
# Every path that COMMITS a transcript (conversation listener, its partial
# windows, and the watcher's wake matching) now rejects those shapes here.
# Real utterances — including short wake phrases and literal command payloads —
# must pass: each rule is deliberately narrow.

#: The exact wake-bias prompt vocabulary (whisper_daemon.INITIAL_PROMPT).
#: A real wake phrase is at most 4 tokens, so 5+ tokens drawn ONLY from this
#: set is a prompt echo, never a human phrase.
WAKE_ECHO_VOCAB = frozenset({
    "jarvis", "jervis", "wake", "up", "utho", "jago", "chalu",
})

#: Words whose repetition is still meaningful user intent (a panicked
#: "stop stop stop" must NEVER be discarded as a hallucination).
SAFETY_TOKENS = frozenset({
    "no", "stop", "cancel", "wait", "halt", "yes", "ok", "okay",
})

#: Tokens that alone (or in <=2-token utterances) are never a command.
FILLER_TOKENS = frozenset({
    "um", "uh", "uhm", "hmm", "mmm", "mm", "ah", "eh", "you", "the",
    "and", "but", "so", "it", "a",
})

#: Memorised non-speech phrases STT engines emit on silence/noise.
SILENCE_HALLUCINATION_PHRASES = (
    "a ver si te acuerdas de esto",
    "a ver si te acuerdas",
    "gracias por ver",
    "subtitulos por amara",
    "subtitulos realizados por",
    "suscribete a mi canal",
    "gracias por su atencion",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "subscribe to my channel",
)


def _hallucination_tokens(text):
    normalized = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return [token for token in re.sub(r"\s+", " ", normalized).strip().split(" ") if token]


def is_hallucinated_transcript(text):
    """True when *text* looks like STT degeneracy instead of user speech.

    Deterministic and side-effect free: the decision is made on the tokens
    alone. Each rule is narrow on purpose — this gate drops junk, it must
    never drop a real command, wake phrase, or repeated safety word.
    """
    tokens = _hallucination_tokens(text)
    if not tokens:
        return False

    normalized = " ".join(tokens)

    # 1. Filler-only utterances ("you", "um uh") are never commands.
    if len(tokens) <= 2 and all(token in FILLER_TOKENS for token in tokens):
        return True

    # 2. Memorised silence phrases, allowing only a tiny (<=2 token) frame
    #    around them ("jarvis, a ver si te acuerdas de esto"). A real
    #    sentence ABOUT the phrase has >=3 surrounding tokens and survives.
    for phrase in SILENCE_HALLUCINATION_PHRASES:
        if phrase in normalized and len(tokens) - len(phrase.split()) <= 2:
            return True

    # 3. Wake-bias prompt echo: 5+ tokens drawn only from the prompt vocab.
    if len(tokens) >= 5 and all(token in WAKE_ECHO_VOCAB for token in tokens):
        return True

    # Repetition rules never fire on safety vocabulary.
    if any(token in SAFETY_TOKENS for token in tokens):
        return False

    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    top_count = max(counts.values())

    # 4. A single token repeated for the whole utterance ("chalu chalu ...").
    if top_count >= 3 and top_count == len(tokens):
        return True

    # 5. One token dominating a longer babble ("chalu chalu wake chalu chalu").
    if top_count >= 5 and top_count / len(tokens) >= 0.7:
        return True

    # 6. A short phrase looped over a long utterance: the same adjacent word
    #    pair four or more times ("play music play music play music play
    #    music"). A real sentence rarely repeats any pair even twice, and a
    #    command said three times in one breath still passes.
    if len(tokens) >= 8:
        bigrams = {}
        for index in range(len(tokens) - 1):
            pair = (tokens[index], tokens[index + 1])
            bigrams[pair] = bigrams.get(pair, 0) + 1
        if max(bigrams.values()) >= 4:
            return True

    return False


def _api_language(language):
    if not language:
        return None
    return language.split("-", 1)[0].strip().lower() or None


def _recognize_groq(audio_data, language):
    if not GROQ_API_KEY:
        raise sr.RequestError("GROQ_API_KEY is missing for Groq STT fallback")

    wav_bytes = audio_data.get_wav_data()
    data = {
        "model": GROQ_STT_MODEL,
        "temperature": "0",
        "response_format": "json",
    }
    api_language = _api_language(language)
    if api_language:
        data["language"] = api_language

    try:
        response = requests.post(
            GROQ_STT_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            data=data,
            files={"file": ("speech.wav", wav_bytes, "audio/wav")},
            timeout=(5, 30),
        )
    except requests.RequestException as exc:
        raise sr.RequestError(f"Groq STT request failed: {exc}") from exc

    if response.status_code != 200:
        raise sr.RequestError(
            f"Groq STT failed: {response.status_code} {response.text[:200]}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise sr.RequestError("Groq STT returned invalid JSON") from exc

    transcript = (payload.get("text") or "").strip()
    if not transcript:
        raise sr.UnknownValueError()
    return transcript


def recognize_inworld(audio_data, language=None, prompts=None):
    if not INWORLD_STT_API_KEY:
        raise sr.RequestError("INWORLD_STT_API_KEY is missing for Inworld STT")

    transcribe_config = {
        "modelId": INWORLD_STT_MODEL,
        "audioEncoding": "AUTO_DETECT",
        "sampleRateHertz": audio_data.sample_rate,
        "numberOfChannels": 1,
    }
    api_language = _api_language(language)
    if api_language:
        transcribe_config["language"] = api_language
    if prompts:
        transcribe_config["prompts"] = list(prompts)

    payload = {
        "transcribeConfig": transcribe_config,
        "audioData": {
            "content": base64.b64encode(audio_data.get_wav_data()).decode("ascii"),
        },
    }

    try:
        response = requests.post(
            INWORLD_STT_URL,
            headers={
                "Authorization": f"Basic {INWORLD_STT_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=INWORLD_STT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise sr.RequestError(f"Inworld STT request failed: {exc}") from exc

    if response.status_code != 200:
        try:
            detail = response.json().get("message") or response.text
        except ValueError:
            detail = response.text
        raise sr.RequestError(
            f"Inworld STT failed: {response.status_code} {str(detail)[:200]}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise sr.RequestError("Inworld STT returned invalid JSON") from exc

    transcript = ((payload.get("transcription") or {}).get("transcript") or "").strip()
    if not transcript:
        raise sr.UnknownValueError()
    return transcript


def recognize_local_whisper(audio_data, timeout=None):
    """Transcribe via the local whisper daemon (mirrors watcher's
    _transcribe_with_daemon). Returns (transcript, language-or-None).

    *timeout* bounds this one request. The final commit in ``listen()`` uses the
    default (generous) budget because it IS the answer; an F34 partial window
    is only an early hint and must never be able to stall the real-time capture
    loop, so it passes a much smaller one.
    """
    request = Request(
        f"{LOCAL_WHISPER_URL}/transcribe",
        data=audio_data.get_wav_data(),
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=float(timeout or 15.0)) as response:
            payload = json.load(response)
    except Exception as exc:
        raise sr.RequestError(f"Local whisper request failed: {exc}") from exc

    if not payload.get("ok"):
        raise sr.RequestError(
            f"Local whisper failed: {payload.get('error') or 'unknown error'}"
        )
    text = (payload.get("text") or "").strip()
    if not text:
        raise sr.UnknownValueError()
    return text, payload.get("language") or None


def recognize_google_or_groq(
    recognizer,
    audio_data,
    language,
    *,
    show_all=False,
    log_prefix="STT",
):
    try:
        return recognizer.recognize_google(audio_data, language=language, show_all=show_all)
    except sr.RequestError as google_exc:
        print(
            f"[{log_prefix}] Google STT unavailable [{language}]: {google_exc} | "
            "trying Groq fallback"
        )
        try:
            transcript = _recognize_groq(audio_data, language)
        except sr.UnknownValueError:
            if show_all:
                return {}
            raise
        except sr.RequestError as groq_exc:
            raise sr.RequestError(
                f"google failed: {google_exc}; groq fallback failed: {groq_exc}"
            ) from groq_exc

        if show_all:
            return {"alternative": [{"transcript": transcript}]}
        return transcript
