"""Find an audible output device for Jarvis TTS.

The Fish TTS pipeline renders through sounddevice. When a Bluetooth headset
mic stays open, the A2DP sink can go silent even though playback reports
success. Use this tool to find a device you can actually hear, then pin it:

    python -m backend.scripts.check_tts_devices                  # list outputs
    python -m backend.scripts.check_tts_devices --tone           # beep on default
    python -m backend.scripts.check_tts_devices --tone --device 16
    python -m backend.scripts.check_tts_devices --tone --like realtek
    python -m backend.scripts.check_tts_devices --tone --all     # beep each output

Then set JARVIS_TTS_OUTPUT_DEVICE=<index or name substring> in .env.
"""

import argparse

import numpy as np


def make_chime(sample_rate=48000):
    """Three short ascending tones as mono float->int16 numpy array."""
    durations = (0.18, 0.18, 0.24)
    freqs = (660, 880, 990)
    pieces = []
    for freq, dur in zip(freqs, durations):
        n = int(sample_rate * dur)
        t = np.arange(n) / sample_rate
        wave = 0.35 * np.sin(2 * np.pi * freq * t)
        fade = min(n // 8, 400)
        env = np.ones(n)
        if fade:
            env[:fade] = np.linspace(0.0, 1.0, fade)
            env[-fade:] = np.linspace(1.0, 0.0, fade)
        pieces.append(wave * env)
        pieces.append(np.zeros(int(sample_rate * 0.06)))
    data = np.concatenate(pieces)
    return (data * 32767 / max(float(np.max(np.abs(data))), 1e-9)).astype(np.int16)


def list_outputs():
    import sounddevice as sd

    hostapis = {i: h["name"] for i, h in enumerate(sd.query_hostapis())}
    default_out = sd.default.device[1]
    print("Default Windows output:", default_out)
    rows = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_output_channels", 0) < 1:
            continue
        marker = "*" if i == default_out else " "
        rows.append(
            (
                marker,
                i,
                hostapis.get(d.get("hostapi"), "?"),
                d.get("max_output_channels"),
                int(d.get("default_samplerate") or 0),
                d.get("name"),
            )
        )
    for marker, i, api, ch, sr, name in rows:
        print(f"{marker} idx={i:3d} {api:22s} ch={ch} sr={sr:6d}  {name}")
    print("\n(* = current default)  Pin via JARVIS_TTS_OUTPUT_DEVICE=<idx-or-name>")


def play_on(devices, chime, sample_rate=48000):
    import sounddevice as sd

    names = sd.query_devices()
    playable = []
    for idx in devices:
        try:
            info = names[idx]
        except (IndexError, TypeError):
            print(f"idx={idx}: no such device")
            continue
        print(f"playing test tone on idx={idx} '{info.get('name')}' ...")
        try:
            sd.play(chime, samplerate=sample_rate, device=idx)
            sd.wait()
            print("   -> opened + played OK (did you hear it?)")
            playable.append(idx)
        except Exception as exc:
            print(f"   -> FAILED to play: {exc}")
    print("\nPlayable devices:", playable)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tone", action="store_true", help="play a test tone")
    parser.add_argument("--device", type=int, help="output device index to test")
    parser.add_argument("--like", help="name-substring filter for outputs")
    parser.add_argument(
        "--all", action="store_true", help="test every stereo-capable output"
    )
    args = parser.parse_args()

    if not args.tone:
        list_outputs()
        return

    import sounddevice as sd

    chime = make_chime()
    chime_stereo = np.stack([chime, chime], axis=1)

    if args.device is not None:
        devices = [args.device]
    elif args.like:
        needle = args.like.lower()
        devices = [
            i
            for i, d in enumerate(sd.query_devices())
            if d.get("max_output_channels", 0) >= 2
            and needle in d.get("name", "").lower()
        ]
    elif args.all:
        devices = [
            i
            for i, d in enumerate(sd.query_devices())
            if d.get("max_output_channels", 0) >= 2
        ]
    else:
        devices = [int(sd.default.device[1])]

    play_on(devices, chime_stereo)


if __name__ == "__main__":
    main()