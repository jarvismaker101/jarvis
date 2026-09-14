import os
import tempfile
import threading
import time
import traceback

import requests
from pydub import AudioSegment
from pydub.playback import play

from backend.config import ELEVENLABS_API_KEY

try:
    from pydub.playback import _play_with_simpleaudio
except Exception:
    _play_with_simpleaudio = None


API_KEY = ELEVENLABS_API_KEY
VOICE_ID = "VldzKzUj8YJIgCXVYvix"

_playback_lock = threading.Lock()
_current_playback = None


def speak_elevenlabs(text, before_playback=None):
    global _current_playback

    try:
        if not API_KEY:
            print("[ELEVENLABS] Missing API key")
            return False

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
        headers = {
            "xi-api-key": API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "optimize_streaming_latency": 3,
        }

        response = requests.post(url, json=data, headers=headers, timeout=(3.05, 12))
        if response.status_code != 200:
            print("[ELEVENLABS] Error:", response.text)
            return False

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as file:
            file.write(response.content)
            path = file.name

        try:
            audio = AudioSegment.from_file(path, format="mp3")
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

        if callable(before_playback):
            before_playback()

        if _play_with_simpleaudio:
            playback = _play_with_simpleaudio(audio)
            with _playback_lock:
                _current_playback = playback

            while playback.is_playing():
                time.sleep(0.05)

            with _playback_lock:
                if _current_playback is playback:
                    _current_playback = None
        else:
            play(audio)

        return True
    except Exception:
        print("[ELEVENLABS] Exception:")
        traceback.print_exc()
        return False


def stop_elevenlabs():
    with _playback_lock:
        playback = _current_playback

    if playback:
        try:
            playback.stop()
        except Exception:
            pass
