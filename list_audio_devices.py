"""
List every audio device PortAudio (sounddevice) can see, grouped by host API.

Windows often exposes the same physical microphone multiple times under
different backends (MME, DirectSound, WASAPI, WDM-KS). The "default" one
sounddevice picks isn't always the one that actually passes audio -- this
script shows all of them so you can find the right index.

Run it, look for your actual mic's name, and note its index under the
WASAPI section specifically (WASAPI is usually the most reliable backend
on modern Windows). Then set that index as MIC_DEVICE_INDEX in
luna_orchestrator.py and DEVICE_INDEX in mic_test.py.
"""

import sounddevice as sd

print("=" * 70)
print("HOST APIs")
print("=" * 70)
for i, api in enumerate(sd.query_hostapis()):
    print(f"[{i}] {api['name']}")

print()
print("=" * 70)
print("INPUT DEVICES (grouped by host API)")
print("=" * 70)

devices = sd.query_devices()
hostapis = sd.query_hostapis()

for api_idx, api in enumerate(hostapis):
    print(f"\n--- {api['name']} ---")
    found_any = False
    for dev_idx, dev in enumerate(devices):
        if dev["hostapi"] == api_idx and dev["max_input_channels"] > 0:
            found_any = True
            default_marker = ""
            if dev_idx == api.get("default_input_device", -1):
                default_marker = "  <-- default for this API"
            print(f"    [{dev_idx}] {dev['name']}  "
                  f"(in ch: {dev['max_input_channels']}, "
                  f"default rate: {dev['default_samplerate']:.0f}Hz){default_marker}")
    if not found_any:
        print("    (no input devices under this API)")

print()
print("=" * 70)
print(f"Overall system default input device index: {sd.default.device[0]}")
print("=" * 70)
print()
print("Pick the index next to your actual microphone's name -- prefer the")
print("WASAPI entry if your mic appears more than once. Set that number as")
print("MIC_DEVICE_INDEX (orchestrator) / DEVICE_INDEX (mic_test.py).")
