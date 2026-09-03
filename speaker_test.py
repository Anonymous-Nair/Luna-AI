"""
Speaker output test -- isolates whether sound output itself works,
separate from Qwen3-TTS entirely.

Same idea as the mic device issue: PortAudio can list your speakers
multiple times across host APIs (MME, DirectSound, WASAPI, WDM-KS),
and the "default" one isn't always the one actually wired to what
you're listening on.
"""

import numpy as np
import sounddevice as sd

# Set this after checking the printed list below if the default doesn't work.
OUTPUT_DEVICE_INDEX = None

print("=" * 70)
print("OUTPUT DEVICES (grouped by host API)")
print("=" * 70)

devices = sd.query_devices()
hostapis = sd.query_hostapis()

for api_idx, api in enumerate(hostapis):
    print(f"\n--- {api['name']} ---")
    found_any = False
    for dev_idx, dev in enumerate(devices):
        if dev["hostapi"] == api_idx and dev["max_output_channels"] > 0:
            found_any = True
            default_marker = ""
            if dev_idx == api.get("default_output_device", -1):
                default_marker = "  <-- default for this API"
            print(f"    [{dev_idx}] {dev['name']}  "
                  f"(out ch: {dev['max_output_channels']}, "
                  f"default rate: {dev['default_samplerate']:.0f}Hz){default_marker}")
    if not found_any:
        print("    (no output devices under this API)")

print()
print(f"Overall system default output device index: {sd.default.device[1]}")
print()

print(f"Playing a 1-second 440Hz tone through device: "
      f"{OUTPUT_DEVICE_INDEX if OUTPUT_DEVICE_INDEX is not None else '(system default)'}")
print("You should hear a plain beep NOW.")

sr = 44100
duration = 1.0
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
tone = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

sd.play(tone, sr, device=OUTPUT_DEVICE_INDEX)
sd.wait()

print("Done. If you heard nothing, try setting OUTPUT_DEVICE_INDEX to one of")
print("the indices printed above (prefer the WASAPI entry) and run again.")
