@echo off
rem ============================================================
rem  Jarvis Launcher - NOISY ENVIRONMENT (immediate launch)
rem  Use this in loud rooms / offices: raises the mic energy
rem  thresholds so background chatter and crowd noise are NOT
rem  picked up. Only your near-field (louder, closer to the mic)
rem  speech triggers Jarvis.
rem
rem  Tuning:  JARVIS_IDLE_ENERGY_THRESHOLD      (listener; 300 quiet, max via JARVIS_MAX_ENERGY_THRESHOLD)
rem           JARVIS_WATCHER_ENERGY_THRESHOLD   (wake word; 80 quiet, max 5000)
rem           JARVIS_MAX_ENERGY_THRESHOLD       (clamps listener thresholds; default 700, quiet)
rem           JARVIS_SPEECH_START_VAD_RATIO     (0.18 quiet - higher = stricter)
rem           JARVIS_FINAL_SPEECH_VAD_RATIO     (0.08 quiet - higher = stricter)
rem ============================================================
cd /d "%~dp0"

set "PYTHON=%~dp0backend\venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

rem ----- Noise-mode tuning -----
set "JARVIS_IDLE_ENERGY_THRESHOLD=2500"
set "JARVIS_WATCHER_ENERGY_THRESHOLD=2000"
set "JARVIS_MAX_ENERGY_THRESHOLD=5000"
set "JARVIS_SPEECH_START_VAD_RATIO=0.5"
set "JARVIS_FINAL_SPEECH_VAD_RATIO=0.35"

start /min "Jarvis Assistant (Noisy)" cmd /k ""%PYTHON%" -m backend.watcher --launch"
exit