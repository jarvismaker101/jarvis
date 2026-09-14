@echo off
cd /d "%~dp0"

set "PYTHON=%~dp0backend\venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

start /min "Jarvis Watcher" cmd /k ""%PYTHON%" -m backend.watcher"
exit
