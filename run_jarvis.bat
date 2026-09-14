@echo off
cd /d "%~dp0"

set "PYTHON=%~dp0backend\venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

start /min "Jarvis Assistant" cmd /k ""%PYTHON%" -m backend.watcher --launch"
exit
