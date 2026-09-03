"""
Luna -- brain to voice bridge test (no mic yet)
=================================================
Purpose: verify Qwen3.5:4b (via Ollama) and Qwen3-TTS talk to each other
correctly before we wire in speech input.

Type a message, press Enter -> Luna (qwen3.5:4b) replies -> Qwen3-TTS speaks it.
Type "quit" to exit.
"""

import torch
import requests
import sounddevice as sd
from qwen_tts import Qwen3TTSModel

# ============================================================
# Config -- keep in sync with luna_orchestrator.py
# ============================================================
OLLAMA_URL = "http://localhost:11434/api/chat"
CHAT_MODEL = "qwen3.5:4b"
SYSTEM_PROMPT = (
    "You are Luna, a concise, capable local voice assistant. "
    "Keep replies short and natural -- a few sentences unless asked for more."
)

TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
TTS_SPEAKER = "Ryan"
TTS_LANGUAGE = "English"
TTS_DEVICE = "cuda:0"   # switch to "cpu" if you don't have a CUDA build of torch installed


def ask_ollama(history: list, user_text: str) -> str:
    history.append({"role": "user", "content": user_text})
    resp = requests.post(
        OLLAMA_URL,
        json={"model": CHAT_MODEL, "messages": history, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    reply = resp.json()["message"]["content"]
    history.append({"role": "assistant", "content": reply})
    return reply


def main():
    print(f"[startup] loading Qwen3-TTS ({TTS_MODEL_ID}) on {TTS_DEVICE} ...")
    tts_model = Qwen3TTSModel.from_pretrained(
        TTS_MODEL_ID,
        device_map=TTS_DEVICE,
        dtype=torch.bfloat16,
    )
    print("[startup] Qwen3-TTS ready.")

    print(f"[startup] checking Ollama at {OLLAMA_URL} ...")
    try:
        test = requests.post(
            OLLAMA_URL,
            json={"model": CHAT_MODEL, "messages": [{"role": "user", "content": "Say ready."}], "stream": False},
            timeout=30,
        )
        test.raise_for_status()
        print(f"[startup] Ollama OK, {CHAT_MODEL} responded: {test.json()['message']['content'][:60]!r}")
    except Exception as e:
        print(f"[startup][ERROR] Could not reach Ollama / {CHAT_MODEL}: {e}")
        print("Make sure 'ollama serve' is running and 'ollama pull qwen3.5:4b' completed.")
        return

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("\n=== Luna brain-to-voice test ready. Type a message (or 'quit'). ===\n")
    while True:
        user_text = input("[you] ").strip()
        if not user_text:
            continue
        if user_text.lower() in ("quit", "exit"):
            break

        reply = ask_ollama(history, user_text)
        print(f"[luna] {reply}")

        wavs, sr = tts_model.generate_custom_voice(
            text=reply,
            language=TTS_LANGUAGE,
            speaker=TTS_SPEAKER,
        )
        sd.play(wavs[0], sr)
        sd.wait()


if __name__ == "__main__":
    main()


# ============================================================
# Quick setup check before running
# ============================================================
"""
1. In one terminal:  ollama serve            (if not already running as a service)
2. Confirm the model is present:  ollama list   -> should show qwen3.5:4b
3. In your venv:
     pip install qwen-tts torch sounddevice soundfile requests
     (torch needs the CUDA build matching your driver -- see pytorch.org)
4. Run:  python luna_voice_test.py
5. Type something and press Enter. First run will download the TTS
   weights from Hugging Face (a few GB) -- that only happens once.
"""
