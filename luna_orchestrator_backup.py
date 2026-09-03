"""
LUNA ORCHESTRATOR — HIGH-SPEED / CUDA-ISOLATED EDITION

Pipeline
--------
Microphone -> WebRTC VAD -> dedicated Whisper/CTranslate2 GPU worker
          -> Ollama Qwen3.5 -> Qwen3-TTS STREAMING GPU -> speakers
          -> HUD WebSocket

Why the Whisper worker is a separate process
---------------------------------------------
Your machine has two different CUDA stacks in the same Python installation:

  PyTorch/Qwen3-TTS : CUDA 13.0 + cuDNN 9.2
  CTranslate2      : CUDA 12 + cuDNN 9

Windows DLL loading can make those stacks collide because both use DLLs such
as cudnn64_9.dll. Whisper therefore runs in a persistent child process. The
TTS/brain process never loads CTranslate2's CUDA DLLs.

This is intentional. It fixes the cuBLAS/cuDNN collision instead of trying
to keep two incompatible CUDA DLL families alive in one Windows process.

Performance goals
-----------------
- Whisper: GPU, int8_float16, beam_size=1, no timestamps, no previous-text
  conditioning.
- VAD: shorter end-of-speech timeout for lower perceived latency.
- Ollama: persistent HTTP session, keep_alive, thinking disabled, short
  voice-oriented output budget, optimized sampling.
- TTS: CUDA/BF16, TF32, persistent Qwen Base model, true chunked streaming playback.
- Audio: continuous capture; no browser SpeechRecognition dependency.
- Diagnostics: every turn prints STT / brain / TTS timings.

Run
---
    cd E:/Luna
    py -3.12 luna_orchestrator.py
"""

import asyncio
import collections
import json
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
import websockets

try:
    import webrtcvad
except ImportError:
    raise SystemExit(
        "webrtcvad is required for continuous listening.\n"
        "Windows: pip install webrtcvad-wheels\n"
        "Linux/Mac: pip install webrtcvad"
    )


# ============================================================
# GLOBAL CONFIG
# ============================================================

E_DRIVE_CACHE = Path("E:/luna/model_cache")
E_DRIVE_CACHE.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(E_DRIVE_CACHE / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(E_DRIVE_CACHE / "torch"))

# ---------------- Ollama ----------------
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

CHAT_MODEL = "qwen3.5:4b"
CODE_MODEL = "ornith:9b"

# Keep spoken responses short by default. The model can still answer longer
# when explicitly asked.
SYSTEM_PROMPT_CHAT = (
    "You are Luna, a fast local voice assistant. "
    "Answer directly and naturally. "
    "For normal voice questions, use 1-3 short sentences. "
    "Do not explain your reasoning unless asked. "
    "Do not use markdown unless requested. "
    "Never say 'mute' or 'stop' unless the user asks about those words."
)

SYSTEM_PROMPT_CODE = (
    "You are Luna in coding mode. "
    "Answer directly and efficiently. "
    "For normal questions, be concise. "
    "For code/debugging requests, give the necessary code and brief explanation."
)

OLLAMA_TIMEOUT = 120
OLLAMA_KEEP_ALIVE = "30m"

# qwen3.5 responds much faster when thinking is explicitly disabled for
# normal voice conversation.
OLLAMA_THINK = False

CHAT_NUM_CTX = 8192
CODE_NUM_CTX = 16384

# Maximum visible tokens. Voice replies normally need far fewer.
CHAT_NUM_PREDICT = 120
CODE_NUM_PREDICT = 512

OLLAMA_TEMPERATURE = 0.45
OLLAMA_TOP_P = 0.85
OLLAMA_TOP_K = 30
OLLAMA_MIN_P = 0.02

# ---------------- Whisper ----------------
WHISPER_MODEL_SIZE = "small.en"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "int8_float16"

# Faster decoding settings.
WHISPER_BEAM_SIZE = 1
WHISPER_BEST_OF = 1

# ---------------- Qwen3-TTS ----------------
TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
TTS_SPEAKER = "Ryan"
TTS_LANGUAGE = "English"
TTS_DEVICE = "cuda:0"

# ---------------- Audio / VAD ----------------
SAMPLE_RATE = 16000
FRAME_MS = 30
BLOCK_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)
FRAME_BYTES = BLOCK_SIZE * 2

VAD_AGGRESSIVENESS = 2
VAD_PADDING_MS = 210

# Lower than the old 600ms. This is one of the biggest perceived-latency wins.
VAD_SILENCE_MS = 390

MIN_UTTERANCE_MS = 180
MAX_UTTERANCE_MS = 15000

# Do not make this too small: speaker buffers can still feed the mic briefly.
SPEAK_COOLDOWN_MS = 500

# ---------------- Wake / control phrases ----------------
WAKE_PHRASES = [
    "wake up luna",
    "wake luna up",
    "hey luna wake up",
    "luna wake up",
]

MUTE_PHRASES = [
    "mute",
    "go to sleep luna",
    "goodnight luna",
    "luna go to sleep",
    "sleep luna",
    "shutdown luna",
    "shut down luna",
    "luna shutdown",
    "power down luna",
]

INTERRUPT_STOP_PHRASES = [
    "stop",
    "luna stop",
    "stop talking",
    "quiet",
    "shush",
]

TO_CODING = [
    "switch to coding mode",
    "coding mode",
    "switch to code mode",
]

TO_CHAT = [
    "switch to chat mode",
    "chat mode",
    "switch to conversation mode",
]

# ---------------- HUD ----------------
WS_HOST = "localhost"
WS_PORT = 8765

# ---------------- Hotkey ----------------
USE_MANUAL_HOTKEY = True
MANUAL_MUTE_HOTKEY = "f9"

# ---------------- Diagnostics ----------------
MIC_DEVICE_INDEX = None
DEBUG_AUDIO = True
PRINT_TIMINGS = True


# ============================================================
# STATE
# ============================================================

class State:
    mode = "chat"
    awake = False
    is_speaking = False
    history_chat = []
    history_code = []
    ws_clients = set()


state = State()

# Audio from the VAD collector.
utterance_queue: "queue.Queue[np.ndarray]" = queue.Queue()
raw_frame_queue: "queue.Queue[bytes]" = queue.Queue()


# ============================================================
# CUDA HELPERS
# ============================================================

def _add_dll_dir(path: Path, label: str, handles: list) -> bool:
    """Register a Windows DLL directory and keep the handle alive."""
    if not path.is_dir():
        print(f"[startup][WARN] {label} DLL directory not found: {path}")
        return False

    try:
        handles.append(os.add_dll_directory(str(path)))
        print(f"[startup]   [{label}] {path}")
        return True
    except Exception as exc:
        print(f"[startup][WARN] Could not register {label}: {exc}")
        return False


def _ensure_whisper_cuda12_packages() -> None:
    """Ensure the complete CUDA 12 runtime used by CTranslate2 is present."""
    required = [
        ("nvidia.cuda_runtime", "nvidia-cuda-runtime-cu12"),
        ("nvidia.cublas", "nvidia-cublas-cu12"),
        ("nvidia.cudnn", "nvidia-cudnn-cu12==9.*"),
        ("nvidia.nvjitlink", "nvidia-nvjitlink-cu12"),
    ]

    missing = []
    for module_name, pip_name in required:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name)

    if not missing:
        return

    print("[whisper] Missing CUDA 12 runtime package(s): " + ", ".join(missing), flush=True)
    print("[whisper] Installing the missing NVIDIA runtime package(s)...", flush=True)

    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", *missing]
    try:
        subprocess.check_call(cmd)
    except Exception as exc:
        raise RuntimeError(
            "Could not install the CUDA 12 packages required by "
            f"CTranslate2: {missing}. Error: {exc}"
        ) from exc


def _resolve_nvidia_package_root(package_name: str) -> Path | None:
    """Resolve an NVIDIA namespace package without relying on site.getsitepackages()."""
    import importlib

    try:
        module = importlib.import_module(f"nvidia.{package_name}")
        paths = getattr(module, "__path__", None)
        if paths:
            root = Path(next(iter(paths)))
            if root.is_dir():
                return root
    except Exception:
        pass

    try:
        import sysconfig
        root = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / package_name
        if root.is_dir():
            return root
    except Exception:
        pass

    return None


def _setup_whisper_cuda12():
    """Create a private CUDA 12 DLL namespace for the Whisper child process."""
    if os.name != "nt":
        return []

    _ensure_whisper_cuda12_packages()

    package_names = ("cuda_runtime", "cuda_nvrtc", "cublas", "cudnn", "nvjitlink")
    package_roots = {}
    for name in package_names:
        root = _resolve_nvidia_package_root(name)
        if root is not None:
            package_roots[name] = root

    if "cublas" not in package_roots:
        raise RuntimeError(
            "CUDA 12 cuBLAS package was not found in the active Python environment. "
            "Expected nvidia-cublas-cu12."
        )

    handles = []
    ordered = (
        ("CUDA-RUNTIME", "cuda_runtime"),
        ("NVJITLINK", "nvjitlink"),
        ("CUDA-NVRTC", "cuda_nvrtc"),
        ("cuBLAS", "cublas"),
        ("cuDNN", "cudnn"),
    )

    dll_dirs = []
    for label, package_name in ordered:
        root = package_roots.get(package_name)
        if root is None:
            print(f"[whisper][WARN] NVIDIA package missing: {package_name}", flush=True)
            continue

        bin_dir = root / "bin"
        if not bin_dir.is_dir():
            bin_dir = root / "lib"

        if bin_dir.is_dir():
            _add_dll_dir(bin_dir, label, handles)
            dll_dirs.append(bin_dir)
        else:
            print(f"[whisper][WARN] {label} DLL directory not found: {root}", flush=True)

    if not dll_dirs:
        raise RuntimeError("No CUDA 12 DLL directories were resolved for the Whisper worker.")

    os.environ["PATH"] = ";".join(str(p) for p in dll_dirs) + ";" + os.environ.get("PATH", "")

    # Preload the dependency chain. This catches WinError 126 at the exact DLL
    # instead of allowing CTranslate2 to report the misleading generic error.
    import ctypes

    preload = (
        ("cudart64_12.dll", True),
        ("nvJitLink_12.dll", False),
        ("nvrtc64_120_0.dll", False),
        ("cublas64_12.dll", True),
        ("cublasLt64_12.dll", True),
        ("cudnn64_9.dll", True),
    )

    for dll_name, required in preload:
        found = next((d / dll_name for d in dll_dirs if (d / dll_name).is_file()), None)
        if found is None:
            if required:
                raise RuntimeError(f"Required Whisper CUDA DLL missing: {dll_name}.")
            continue

        try:
            ctypes.WinDLL(str(found))
            print(f"[whisper] preloaded {dll_name} -> {found}", flush=True)
        except OSError as exc:
            winerror = getattr(exc, "winerror", None)
            raise RuntimeError(
                f"Windows could not load {dll_name} from {found}. "
                f"WinError={winerror}. {exc}. "
                "A dependency of the CUDA 12 DLL is missing or incompatible."
            ) from exc

    return handles

def _setup_tts_torch_cuda():
    """
    CUDA environment for the main LUNA process.

    Do NOT put NVIDIA cuDNN 9.24 ahead of PyTorch here. Your Torch build
    reports cuDNN 9.2 and Qwen3-TTS already passes a native cuDNN convolution
    test with that stack.
    """
    import torch

    handles = []

    if os.name == "nt":
        torch_lib = Path(torch.__file__).resolve().parent / "lib"

        # Torch first.
        _add_dll_dir(torch_lib, "PyTorch", handles)

        # CUDA 12 cuBLAS/NVRTC are safe to expose here because their DLL names
        # differ from the CUDA 13 Torch libraries. Do NOT expose nvidia/cudnn.
        try:
            import sysconfig

            nvidia = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
            if not nvidia.is_dir():
                try:
                    import nvidia
                    nvidia = Path(list(nvidia.__path__)[0])
                except Exception:
                    pass
            _add_dll_dir(nvidia / "cublas" / "bin", "cuBLAS", handles)
            _add_dll_dir(nvidia / "cuda_nvrtc" / "bin", "CUDA-NVRTC", handles)

            safe_dirs = [
                torch_lib,
                nvidia / "cublas" / "bin",
                nvidia / "cuda_nvrtc" / "bin",
            ]
            valid = [str(p) for p in safe_dirs if p.is_dir()]
            os.environ["PATH"] = ";".join(valid) + ";" + os.environ.get(
                "PATH", ""
            )

        except Exception as exc:
            print(f"[startup][WARN] Optional CUDA path setup: {exc}")

    return handles


# ============================================================
# WHISPER GPU WORKER
# ============================================================

def whisper_worker_main(request_q, result_q):
    """
    Persistent Whisper service.

    It has its own process and therefore its own Windows CUDA DLL namespace.
    """
    try:
        print("[whisper] starting isolated CUDA worker...", flush=True)

        _dll_handles = _setup_whisper_cuda12()

        from faster_whisper import WhisperModel

        print(
            f"[whisper] loading {WHISPER_MODEL_SIZE} "
            f"on {WHISPER_DEVICE}...",
            flush=True,
        )

        model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            download_root=str(E_DRIVE_CACHE / "whisper"),
            cpu_threads=max(1, (os.cpu_count() or 8) // 2),
            num_workers=1,
        )

        # Fail fast. This is intentionally a real GPU inference call.
        dummy = np.zeros(16000, dtype=np.float32)
        t0 = time.perf_counter()
        segments, _ = model.transcribe(
            dummy,
            language="en",
            beam_size=WHISPER_BEAM_SIZE,
            best_of=WHISPER_BEST_OF,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
        )
        list(segments)
        warm_ms = (time.perf_counter() - t0) * 1000.0

        print(
            f"[whisper] GPU READY — warmup {warm_ms:.0f} ms",
            flush=True,
        )

        result_q.put(("__READY__", ""))

        while True:
            item = request_q.get()

            if item is None:
                break

            request_id, audio = item

            try:
                started = time.perf_counter()

                segments, _ = model.transcribe(
                    audio,
                    language="en",
                    beam_size=WHISPER_BEAM_SIZE,
                    best_of=WHISPER_BEST_OF,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    without_timestamps=True,
                    vad_filter=False,
                )

                text = " ".join(
                    seg.text.strip()
                    for seg in segments
                    if seg.text and seg.text.strip()
                ).strip()

                elapsed_ms = (time.perf_counter() - started) * 1000.0

                result_q.put(
                    ("RESULT", request_id, text, elapsed_ms)
                )

            except Exception as exc:
                result_q.put(
                    (
                        "ERROR",
                        request_id,
                        type(exc).__name__,
                        str(exc),
                    )
                )

    except Exception as exc:
        print(
            f"[whisper][FATAL] {type(exc).__name__}: {exc}",
            flush=True,
        )
        result_q.put(
            (
                "__FATAL__",
                type(exc).__name__,
                str(exc),
            )
        )


# ============================================================
# STT CLIENT
# ============================================================

stt_process = None
stt_request_q = None
stt_result_q = None
stt_call_lock = threading.Lock()
stt_request_counter = 0


def start_whisper_worker():
    global stt_process, stt_request_q, stt_result_q

    ctx = mp.get_context("spawn")

    stt_request_q = ctx.Queue()
    stt_result_q = ctx.Queue()

    stt_process = ctx.Process(
        target=whisper_worker_main,
        args=(stt_request_q, stt_result_q),
        name="Luna-Whisper-GPU",
        daemon=True,
    )

    stt_process.start()

    deadline = time.time() + 180.0

    while time.time() < deadline:
        try:
            msg = stt_result_q.get(timeout=0.25)

            if msg[0] == "__READY__":
                return

            if msg[0] == "__FATAL__":
                raise RuntimeError(
                    f"Whisper worker failed: {msg[1]}: {msg[2]}"
                )

        except queue.Empty:
            if not stt_process.is_alive():
                raise RuntimeError(
                    "Whisper GPU worker exited during startup."
                )

    raise TimeoutError("Whisper GPU worker startup timed out.")


def transcribe(audio_np: np.ndarray) -> str:
    """
    Synchronous STT request against the persistent GPU worker.

    The lock prevents two callers from consuming each other's result.
    """
    global stt_request_counter

    with stt_call_lock:
        stt_request_counter += 1
        request_id = stt_request_counter

        stt_request_q.put((request_id, audio_np))

        while True:
            msg = stt_result_q.get()

            if msg[0] == "RESULT":
                _, returned_id, text, elapsed_ms = msg

                if returned_id != request_id:
                    # Should never happen because calls are serialized.
                    continue

                if PRINT_TIMINGS:
                    print(
                        f"[timing] STT {elapsed_ms:.0f} ms"
                    )

                return text

            if msg[0] == "ERROR":
                _, returned_id, name, error = msg

                if returned_id != request_id:
                    continue

                raise RuntimeError(
                    f"Whisper worker {name}: {error}"
                )

            if msg[0] == "__FATAL__":
                raise RuntimeError(
                    f"Whisper worker failed: {msg[1]}: {msg[2]}"
                )


def stop_whisper_worker():
    global stt_process

    if stt_request_q is not None:
        try:
            stt_request_q.put_nowait(None)
        except Exception:
            pass

    if stt_process is not None:
        try:
            stt_process.join(timeout=2.0)
        except Exception:
            pass

        if stt_process.is_alive():
            try:
                stt_process.terminate()
            except Exception:
                pass


# ============================================================
# HUD WEBSOCKET
# ============================================================

async def ws_handler(websocket):
    state.ws_clients.add(websocket)

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
            except (TypeError, ValueError):
                continue

            if not isinstance(data, dict):
                continue

            if data.get("type") == "speak_request":
                text = str(data.get("text", "")).strip()

                if text:
                    loop = asyncio.get_running_loop()

                    threading.Thread(
                        target=speak,
                        args=(loop, text),
                        daemon=True,
                    ).start()

    finally:
        state.ws_clients.discard(websocket)


async def broadcast(payload: dict):
    if not state.ws_clients:
        return

    msg = json.dumps(payload)
    dead = []

    for ws in list(state.ws_clients):
        try:
            await ws.send(msg)
        except Exception:
            dead.append(ws)

    for ws in dead:
        state.ws_clients.discard(ws)


def push_state(loop, hud_state, text=""):
    try:
        asyncio.run_coroutine_threadsafe(
            broadcast(
                {
                    "state": hud_state,
                    "text": text,
                    "mode": state.mode,
                }
            ),
            loop,
        )
    except Exception:
        pass



# ============================================================
# QWEN3-TTS STREAMING
# ============================================================

# IMPORTANT:
# The dffdeeq streaming fork exposes true streaming through
# stream_generate_voice_clone(). That API is for the Qwen3-TTS Base model,
# not the CustomVoice speaker API used by the old generate_custom_voice().
#
# Therefore Luna uses the local 1.7B Base model and a local reference clip.
# Audio chunks are sent directly to sounddevice as soon as Qwen yields them;
# the complete waveform is never generated before playback starts.

TTS_MODEL_ID = r"E:\TTS\models\Qwen3-TTS-12Hz-0.6B-Base"
TTS_REF_AUDIO = r"E:\TTS\luna_reference_test.wav"
TTS_REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. "
    "But you know what? You blew it! And thanks to you."
)
TTS_LANGUAGE = "English"
TTS_DEVICE = "cuda:0"
TTS_EMIT_EVERY_FRAMES = 8
TTS_DECODE_WINDOW_FRAMES = 80
TTS_OVERLAP_SAMPLES = 0

# Set True only if the installed streaming fork exposes the optimization API.
# We keep compilation disabled initially because the priority is a reliable
# Windows integration with the existing CUDA/Torch stack.
TTS_ENABLE_STREAMING_OPTIMIZATIONS = False

# Audio playback is streamed through one persistent OutputStream per utterance.
# A small queue lets the generator and sounddevice run independently so model
# decoding does not have to wait for the sound device on every chunk.
TTS_AUDIO_QUEUE_MAX = 8

torch = None
tts_model = None
tts_voice_prompt = None
_tts_dll_handles = []


def init_tts():
    """Load Torch + the Qwen streaming Base model only in the main process."""
    global torch, tts_model, tts_voice_prompt, _tts_dll_handles

    import torch as _torch

    torch = _torch
    _tts_dll_handles = _setup_tts_torch_cuda()

    print(f"[startup] PyTorch CUDA: {torch.version.cuda}")
    print(f"[startup] PyTorch cuDNN: {torch.backends.cudnn.version()}")
    print(
        f"[startup] GPU: "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}"
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen streaming TTS requires CUDA, but CUDA is unavailable.")

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if not Path(TTS_REF_AUDIO).is_file():
        raise FileNotFoundError(
            f"Qwen streaming TTS reference audio was not found: {TTS_REF_AUDIO}"
        )

    print(f"[startup] loading Qwen3-TTS streaming Base model ({TTS_MODEL_ID}) ...")

    from qwen_tts import Qwen3TTSModel

    # The streaming fork accepts the normal Qwen model-loading API.
    tts_model = Qwen3TTSModel.from_pretrained(
        TTS_MODEL_ID,
        device_map=TTS_DEVICE,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    if not hasattr(tts_model, "stream_generate_voice_clone"):
        raise RuntimeError(
            "The installed qwen_tts package does not expose "
            "stream_generate_voice_clone(). The streaming fork is not active."
        )

    # Build the speaker prompt once. This is deliberately outside speak(), so
    # every Luna response does not repeatedly process the reference clip.
    print("[startup] building Qwen voice-clone prompt ...")
    prompt_started = time.perf_counter()

    with torch.inference_mode():
        tts_voice_prompt = tts_model.create_voice_clone_prompt(
            ref_audio=TTS_REF_AUDIO,
            ref_text=TTS_REF_TEXT,
        )

    prompt_ms = (time.perf_counter() - prompt_started) * 1000.0

    if TTS_ENABLE_STREAMING_OPTIMIZATIONS and hasattr(
        tts_model, "enable_streaming_optimizations"
    ):
        print("[startup] enabling Qwen streaming optimizations ...")
        tts_model.enable_streaming_optimizations(
            decode_window_frames=TTS_DECODE_WINDOW_FRAMES,
            use_compile=False,
        )

    print(
        f"[startup] Qwen3-TTS streaming ready "
        f"(voice prompt {prompt_ms:.0f} ms)."
    )


tts_speak_lock = threading.Lock()


def _play_streaming_qwen_audio(
    loop,
    text: str,
    audio_queue: queue.Queue,
    producer_done: threading.Event,
    producer_error: list,
    stop_event: threading.Event,
):
    """Consume Qwen PCM chunks and play them continuously with sounddevice."""
    output_stream = None
    playback_started = time.perf_counter()
    first_chunk = True

    try:
        while not producer_done.is_set() or not audio_queue.empty():
            if stop_event.is_set() and audio_queue.empty():
                break

            try:
                item = audio_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if item is None:
                break

            chunk, sr = item
            chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)

            if chunk.size == 0:
                continue

            if output_stream is None:
                output_stream = sd.OutputStream(
                    samplerate=int(sr),
                    channels=1,
                    dtype="float32",
                    blocksize=0,
                )
                output_stream.start()

            if first_chunk:
                first_chunk = False
                first_ms = (time.perf_counter() - playback_started) * 1000.0
                if PRINT_TIMINGS:
                    print(f"[timing] TTS first audio chunk {first_ms:.0f} ms")

                state.is_speaking = True
                push_state(loop, "speaking", text)

            if stop_event.is_set():
                break

            output_stream.write(chunk)

    except Exception as exc:
        producer_error.append(exc)
    finally:
        if output_stream is not None:
            try:
                output_stream.stop()
            except Exception:
                pass
            try:
                output_stream.close()
            except Exception:
                pass


def speak(loop, text: str):
    """
    Generate and play Qwen3-TTS audio as a real stream.

    The generator and playback run concurrently. Qwen emits a chunk every
    TTS_EMIT_EVERY_FRAMES frames and sounddevice starts playback on the first
    available chunk instead of waiting for the whole utterance.
    """
    if not text.strip():
        return

    if tts_model is None or tts_voice_prompt is None:
        print("[ERROR] Qwen streaming TTS is not initialized.")
        return

    with tts_speak_lock:
        audio_queue: queue.Queue = queue.Queue(maxsize=TTS_AUDIO_QUEUE_MAX)
        producer_done = threading.Event()
        stop_event = threading.Event()
        producer_error = []

        state.is_speaking = True
        push_state(loop, "speaking", text)

        generation_started = time.perf_counter()
        first_chunk_time = [None]

        def producer():
            try:
                with torch.inference_mode():
                    for chunk, sr in tts_model.stream_generate_voice_clone(
                        text=text,
                        language=TTS_LANGUAGE,
                        voice_clone_prompt=tts_voice_prompt,
                        emit_every_frames=TTS_EMIT_EVERY_FRAMES,
                        decode_window_frames=TTS_DECODE_WINDOW_FRAMES,
                        overlap_samples=TTS_OVERLAP_SAMPLES,
                    ):
                        if stop_event.is_set():
                            break

                        if first_chunk_time[0] is None:
                            first_chunk_time[0] = (
                                time.perf_counter() - generation_started
                            ) * 1000.0
                            if PRINT_TIMINGS:
                                print(
                                    f"[timing] TTS first chunk generated "
                                    f"{first_chunk_time[0]:.0f} ms"
                                )

                        # Backpressure is intentional: never let an unlimited
                        # queue accumulate seconds of audio in RAM.
                        while not stop_event.is_set():
                            try:
                                audio_queue.put((chunk, sr), timeout=0.05)
                                break
                            except queue.Full:
                                continue
            except Exception as exc:
                producer_error.append(exc)
            finally:
                producer_done.set()
                try:
                    audio_queue.put_nowait(None)
                except queue.Full:
                    pass

        producer_thread = threading.Thread(
            target=producer,
            daemon=True,
            name="Luna-Qwen-TTS-Generator",
        )
        producer_thread.start()

        playback_thread = threading.Thread(
            target=_play_streaming_qwen_audio,
            args=(
                loop,
                text,
                audio_queue,
                producer_done,
                producer_error,
                stop_event,
            ),
            daemon=True,
            name="Luna-Qwen-TTS-Playback",
        )
        playback_thread.start()

        try:
            # While Qwen generates, the main speak() thread monitors the same
            # VAD utterance queue for interrupt / mute commands.
            while producer_thread.is_alive() or playback_thread.is_alive():
                try:
                    interrupt_audio = utterance_queue.get(timeout=0.025)
                except queue.Empty:
                    continue

                itext = transcribe(interrupt_audio).lower()

                if not itext:
                    continue

                if any(phrase in itext for phrase in MUTE_PHRASES):
                    stop_event.set()
                    state.awake = False
                    push_state(loop, "sleeping", "Muted.")
                    break

                if any(phrase in itext for phrase in INTERRUPT_STOP_PHRASES):
                    stop_event.set()
                    break

                # Everything else during playback is treated as echo/noise.

        finally:
            stop_event.set()
            producer_thread.join(timeout=2.0)
            playback_thread.join(timeout=2.0)

            try:
                sd.stop()
            except Exception:
                pass

        if producer_error:
            exc = producer_error[0]
            print(
                f"[ERROR] Qwen3-TTS streaming failed: "
                f"{type(exc).__name__}: {exc}"
            )

        generation_ms = (time.perf_counter() - generation_started) * 1000.0
        if PRINT_TIMINGS:
            print(f"[timing] TTS stream complete {generation_ms:.0f} ms")

        cooldown_deadline = time.time() + SPEAK_COOLDOWN_MS / 1000.0
        while True:
            remaining = cooldown_deadline - time.time()
            if remaining <= 0:
                break

            try:
                utterance_queue.get(timeout=remaining)
                if DEBUG_AUDIO:
                    print(
                        "[debug] discarded utterance during "
                        "post-speech echo cooldown"
                    )
            except queue.Empty:
                break

        state.is_speaking = False
        push_state(
            loop,
            "listening" if state.awake else "sleeping",
        )


# ============================================================
# OLLAMA BRAIN
# ============================================================

ollama_session = requests.Session()

ollama_session.headers.update(
    {
        "Content-Type": "application/json",
        "Connection": "keep-alive",
    }
)


def detect_mode_switch(text: str):
    low = text.lower()

    if any(p in low for p in TO_CODING):
        return "coding"

    if any(p in low for p in TO_CHAT):
        return "chat"

    return None


def _ollama_request(model, history, num_ctx, num_predict):
    """
    Single persistent HTTP call.

    Important:
      think=False is top-level, not inside options.
      This is the correct Ollama API behavior for Qwen thinking models.
    """
    payload = {
        "model": model,
        "messages": history,
        "stream": False,
        "think": OLLAMA_THINK,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "top_p": OLLAMA_TOP_P,
            "top_k": OLLAMA_TOP_K,
            "min_p": OLLAMA_MIN_P,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "repeat_penalty": 1.05,
        },
    }

    return ollama_session.post(
        OLLAMA_URL,
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )


def ask_ollama(user_text: str) -> str:
    if state.mode == "coding":
        model = CODE_MODEL
        history = state.history_code
        system = SYSTEM_PROMPT_CODE
        num_ctx = CODE_NUM_CTX
        num_predict = CODE_NUM_PREDICT
    else:
        model = CHAT_MODEL
        history = state.history_chat
        system = SYSTEM_PROMPT_CHAT
        num_ctx = CHAT_NUM_CTX
        num_predict = CHAT_NUM_PREDICT

    if not history:
        history.append(
            {
                "role": "system",
                "content": system,
            }
        )

    history.append(
        {
            "role": "user",
            "content": user_text,
        }
    )

    # Keep enough recent context to feel conversational while preventing
    # a huge prompt from slowing every subsequent turn.
    max_messages = 14

    if len(history) > max_messages:
        system_message = history[0]
        history[:] = [
            system_message,
            *history[-(max_messages - 1):],
        ]

    started = time.perf_counter()

    resp = _ollama_request(
        model,
        history,
        num_ctx,
        num_predict,
    )

    resp.raise_for_status()

    elapsed_ms = (
        time.perf_counter() - started
    ) * 1000.0

    data = resp.json()

    message = data.get("message", {})
    reply = str(message.get("content", "")).strip()

    if not reply:
        # Some Ollama/model combinations can return thinking text when a
        # client/model configuration changes. Never speak the hidden trace.
        reply = "I'm here."

    history.append(
        {
            "role": "assistant",
            "content": reply,
        }
    )

    if PRINT_TIMINGS:
        total_ns = data.get("total_duration")
        eval_count = data.get("eval_count")

        if total_ns:
            server_ms = total_ns / 1_000_000.0
            print(
                f"[timing] Ollama wall={elapsed_ms:.0f} ms "
                f"server={server_ms:.0f} ms "
                f"tokens={eval_count if eval_count is not None else '?'}"
            )
        else:
            print(
                f"[timing] Ollama {elapsed_ms:.0f} ms"
            )

    print(
        f"[brain] {model} -> {reply}"
    )

    return reply


# ============================================================
# AUDIO CAPTURE / VAD
# ============================================================

def audio_callback(indata, frames, time_info, status):
    if status and DEBUG_AUDIO:
        print(
            f"[debug] audio callback status: {status}"
        )

    raw_frame_queue.put(bytes(indata))


_last_level_print = [0.0]


def _debug_mic_level(
    frame_bytes: bytes,
    is_speech: bool,
):
    now = time.time()

    if now - _last_level_print[0] < 1.0:
        return

    _last_level_print[0] = now

    samples = (
        np.frombuffer(
            frame_bytes,
            dtype=np.int16,
        ).astype(np.float32)
        / 32768.0
    )

    rms = float(
        np.sqrt(np.mean(samples ** 2))
    )

    db = (
        20 * np.log10(rms)
        if rms > 1e-9
        else -180
    )

    print(
        f"[debug] mic level: {db:6.1f} dB   "
        f"vad_speech={is_speech}"
    )


def vad_worker():
    vad = webrtcvad.Vad(
        VAD_AGGRESSIVENESS
    )

    num_padding_frames = max(
        1,
        int(VAD_PADDING_MS / FRAME_MS),
    )

    num_silence_frames = max(
        1,
        int(VAD_SILENCE_MS / FRAME_MS),
    )

    max_utterance_frames = int(
        MAX_UTTERANCE_MS / FRAME_MS
    )

    ring_buffer = collections.deque(
        maxlen=num_padding_frames
    )

    triggered = False
    voiced_frames = []
    silence_run = 0

    while True:
        frame = raw_frame_queue.get()

        if len(frame) != FRAME_BYTES:
            if DEBUG_AUDIO:
                print(
                    f"[debug] dropped frame: "
                    f"got {len(frame)} bytes, "
                    f"expected {FRAME_BYTES}"
                )
            continue

        is_speech = vad.is_speech(
            frame,
            SAMPLE_RATE,
        )

        if DEBUG_AUDIO:
            _debug_mic_level(
                frame,
                is_speech,
            )

        if not triggered:
            ring_buffer.append(
                (frame, is_speech)
            )

            num_voiced = sum(
                1
                for _, speech in ring_buffer
                if speech
            )

            if (
                num_voiced
                >= 0.9 * ring_buffer.maxlen
            ):
                triggered = True

                if DEBUG_AUDIO:
                    print(
                        "[vad] speech START"
                    )

                voiced_frames.extend(
                    frame
                    for frame, _ in ring_buffer
                )

                ring_buffer.clear()
                silence_run = 0

        else:
            voiced_frames.append(frame)

            silence_run = (
                0
                if is_speech
                else silence_run + 1
            )

            hit_silence = (
                silence_run
                >= num_silence_frames
            )

            hit_max_length = (
                len(voiced_frames)
                >= max_utterance_frames
            )

            if hit_silence or hit_max_length:
                triggered = False

                duration_ms = (
                    len(voiced_frames)
                    * FRAME_MS
                )

                if (
                    duration_ms
                    >= MIN_UTTERANCE_MS
                ):
                    if DEBUG_AUDIO:
                        reason = (
                            "max length reached"
                            if hit_max_length
                            else "silence"
                        )

                        print(
                            f"[vad] speech END "
                            f"({reason}) -> "
                            f"utterance "
                            f"{duration_ms}ms "
                            f"(queued for Whisper)"
                        )

                    audio_bytes = b"".join(
                        voiced_frames
                    )

                    audio_np = (
                        np.frombuffer(
                            audio_bytes,
                            dtype=np.int16,
                        )
                        .astype(np.float32)
                        / 32768.0
                    )

                    utterance_queue.put(
                        audio_np
                    )

                elif DEBUG_AUDIO:
                    print(
                        f"[vad] speech END -> "
                        f"utterance "
                        f"{duration_ms}ms "
                        f"(too short, discarded)"
                    )

                voiced_frames = []
                ring_buffer.clear()
                silence_run = 0


# ============================================================
# BRAIN WORKER
# ============================================================

def brain_worker(loop):
    print(
        '[ready] Listening continuously. '
        'Say "wake up Luna" to begin.'
    )

    push_state(
        loop,
        "sleeping",
        'Say "wake up Luna" to begin.',
    )

    while True:
        audio = utterance_queue.get()

        try:
            text = transcribe(audio)

            if not text:
                if DEBUG_AUDIO:
                    print(
                        "[whisper] utterance captured "
                        "but no words recognized"
                    )
                continue

            low = text.lower()

            print(f"[you] {text}")

            # ---------------- sleeping ----------------
            if not state.awake:
                if any(
                    phrase in low
                    for phrase in WAKE_PHRASES
                ):
                    state.awake = True

                    push_state(
                        loop,
                        "listening",
                        "Yes?",
                    )

                    # Keep wake acknowledgement short without blocking the brain worker.
                    threading.Thread(
                        target=speak,
                        args=(loop, "Yes?"),
                        daemon=True,
                    ).start()

                continue

            # ---------------- mute ----------------
            if any(phrase in low for phrase in MUTE_PHRASES):
                state.awake = False
                try:
                    sd.stop()
                except Exception:
                    pass
                push_state(loop, "sleeping", "Going to sleep.")
                continue

            # ---------------- mode switch ----------------
            switch = detect_mode_switch(text)
            if switch:
                state.mode = switch
                push_state(loop, "listening", f"Switched to {switch} mode.")
                continue

            # ---------------- brain ----------------
            push_state(
                loop,
                "thinking",
                text,
            )

            reply = ask_ollama(text)

            speak(
                loop,
                reply,
            )

        except Exception as exc:
            import traceback

            print(
                f"[ERROR] brain_worker: "
                f"{type(exc).__name__}: {exc}"
            )

            traceback.print_exc()

            state.is_speaking = False

            push_state(
                loop,
                "listening"
                if state.awake
                else "sleeping",
            )


# ============================================================
# MANUAL F9 HOTKEY
# ============================================================

def start_manual_hotkey(loop):
    try:
        import keyboard
    except ImportError:
        print(
            "[startup] 'keyboard' not installed -- "
            "skipping manual hotkey."
        )
        return

    def toggle(_=None):
        state.awake = not state.awake

        if state.awake:
            push_state(
                loop,
                "listening",
                "Manually woken.",
            )

            print(
                "[hotkey] Luna manually woken."
            )

        else:
            try:
                sd.stop()
            except Exception:
                pass

            state.is_speaking = False

            push_state(
                loop,
                "sleeping",
                "Manually muted.",
            )

            print(
                "[hotkey] Luna manually muted."
            )

    keyboard.add_hotkey(
        MANUAL_MUTE_HOTKEY,
        toggle,
    )

    print(
        "[startup] Manual wake/mute hotkey: "
        f"{MANUAL_MUTE_HOTKEY.upper()}"
    )


# ============================================================
# MAIN
# ============================================================

async def async_main():
    loop = asyncio.get_running_loop()

    # Start the isolated Whisper GPU service BEFORE loading TTS.
    # This keeps CTranslate2 CUDA 12 DLLs in the child process only.
    start_whisper_worker()

    # Load Qwen3-TTS in the main CUDA 13 / PyTorch process.
    init_tts()

    # Warm Ollama model into memory before the first real command.
    # Empty prompt + keep_alive loads without generating an answer.
    try:
        print(
            f"[startup] Warming Ollama model: {CHAT_MODEL} ..."
        )

        warm_started = time.perf_counter()

        warm = ollama_session.post(
            OLLAMA_URL,
            json={
                "model": CHAT_MODEL,
                "messages": [{"role": "user", "content": ""}],
                "stream": False,
                "think": OLLAMA_THINK,
                "keep_alive": OLLAMA_KEEP_ALIVE,
            },
            timeout=OLLAMA_TIMEOUT,
        )

        warm.raise_for_status()

        warm_ms = (
            time.perf_counter() - warm_started
        ) * 1000.0

        print(
            f"[startup] Ollama {CHAT_MODEL} warm "
            f"({warm_ms:.0f} ms)"
        )

    except Exception as exc:
        print(
            f"[startup][WARN] Ollama warm-up failed: "
            f"{type(exc).__name__}: {exc}"
        )
        print(
            "[startup] Luna will still try Ollama when a command arrives."
        )

    server = await websockets.serve(
        ws_handler,
        WS_HOST,
        WS_PORT,
    )

    print(
        f"[startup] HUD WebSocket listening on "
        f"ws://{WS_HOST}:{WS_PORT}"
    )

    print(
        "[startup] Available audio input devices:"
    )

    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = None

    for idx, dev in enumerate(
        sd.query_devices()
    ):
        if dev.get(
            "max_input_channels",
            0,
        ) > 0:
            marker = (
                "  <-- default"
                if idx == default_in
                else ""
            )

            print(
                f"    [{idx}] "
                f"{dev['name']} "
                f"(in ch: "
                f"{dev['max_input_channels']})"
                f"{marker}"
            )

    chosen = (
        MIC_DEVICE_INDEX
        if MIC_DEVICE_INDEX is not None
        else default_in
    )

    print(
        f"[startup] Using input device index: "
        f"{chosen}"
    )

    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="int16",
        channels=1,
        callback=audio_callback,
        device=chosen,
    )

    stream.start()

    print(
        "[startup] Microphone stream started (continuous)."
    )

    threading.Thread(
        target=vad_worker,
        daemon=True,
        name="Luna-VAD",
    ).start()

    threading.Thread(
        target=brain_worker,
        args=(loop,),
        daemon=True,
        name="Luna-Brain",
    ).start()

    if USE_MANUAL_HOTKEY:
        start_manual_hotkey(loop)

    print(
        "[ready] LUNA HIGH-SPEED PIPELINE ONLINE."
    )

    try:
        await server.wait_closed()
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

        stop_whisper_worker()


def main():
    # Required for Windows multiprocessing with spawn.
    mp.freeze_support()

    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print(
            "\n[shutdown] Luna stopped."
        )
    finally:
        stop_whisper_worker()


if __name__ == "__main__":
    main()
