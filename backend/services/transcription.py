import base64
import json
import os
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
LOCAL_WHISPER_PORT = int(os.getenv("JARVIS_WHISPER_PORT", "8767"))
LOCAL_WHISPER_URL = f"http://127.0.0.1:{LOCAL_WHISPER_PORT}"


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
            timeout=(5, 15),
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


def recognize_local_whisper(audio_data):
    """Transcribe via the local whisper daemon (mirrors watcher's
    _transcribe_with_daemon). Returns (transcript, language-or-None)."""
    request = Request(
        f"{LOCAL_WHISPER_URL}/transcribe",
        data=audio_data.get_wav_data(),
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=15.0) as response:
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
