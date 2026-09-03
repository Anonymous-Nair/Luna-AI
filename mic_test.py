import sounddevice as sd
import numpy as np
import time

# Set this after running list_audio_devices.py -- try the WASAPI entry for
# your actual mic first. Leave as None to use whatever sounddevice picks
# as default (which may be the problem).
DEVICE_INDEX = None

print("MIC TEST")
print(f"Using device index: {DEVICE_INDEX if DEVICE_INDEX is not None else '(system default)'}")
if DEVICE_INDEX is not None:
    info = sd.query_devices(DEVICE_INDEX)
    print(f"Device: {info['name']}")
print("Speak normally for 10 seconds...")
print()

def callback(indata, frames, time_info, status):
    if status:
        print(f"STATUS: {status}")
    x = np.asarray(indata[:, 0], dtype=np.float32)
    rms = np.sqrt(np.mean(x * x))
    db = 20 * np.log10(max(rms, 1e-9))
    print(f"RMS={rms:.5f}  dB={db:6.1f}", flush=True)

with sd.InputStream(
    samplerate=16000,
    channels=1,
    dtype="float32",
    blocksize=480,
    device=DEVICE_INDEX,
    callback=callback
):
    time.sleep(10)

print("DONE")
