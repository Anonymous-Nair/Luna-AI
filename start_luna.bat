@echo off
REM ============================================================
REM LUNA Launcher
REM Opens two PowerShell windows:
REM   1) Ollama    -- serves qwen3.5:4b (and ornith:9b) on :11434
REM   2) Orchestrator -- STT (faster-whisper) + Qwen3-TTS + brain routing
REM ============================================================

REM --- EDIT THIS if your project folder is somewhere else ---
set LUNA_DIR=E:\luna

echo Starting LUNA...
echo.

REM --- Window 1: Ollama (brain server) ---
REM If Ollama is already running as a background service/tray app on your
REM machine, this window may show "address already in use" -- that's fine,
REM it just means Ollama was already up and this window can be closed.
start "LUNA - Ollama Brain (:11434)" powershell -NoExit -Command "ollama serve"

echo Waiting a few seconds for Ollama to come up...
timeout /t 4 /nobreak >nul

REM --- Window 2: Orchestrator (STT + Qwen3-TTS + brain routing + HUD websocket) ---
start "LUNA - Orchestrator (STT/TTS/:8765)" powershell -NoExit -Command "cd '%LUNA_DIR%'; .\venv\Scripts\Activate.ps1; python luna_orchestrator.py"

echo.
echo Two windows launching:
echo   1) Ollama Brain
echo   2) Luna Orchestrator (STT / Qwen3-TTS)
echo.
echo Once the Orchestrator window shows:
echo   [ready] Listening continuously. Say "wake up Luna" to begin.
echo open luna_hud.html in your browser and start talking.
echo.
pause
