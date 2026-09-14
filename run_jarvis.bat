@echo off
cd /d "%~dp0"

set "PYTHON=%~dp0backend\venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

rem The watcher window is minimised, so its progress was invisible: a launch
rem that kept retrying or failed outright looked like "it just hangs on
rem terminals". Its console output is now also kept in data\logs\watcher.log,
rem which is the first place to look when a launch does not come up.
if not exist "%~dp0data\logs" mkdir "%~dp0data\logs"
start /min "Jarvis Assistant" cmd /k ""%PYTHON%" -m backend.watcher --launch >> "%~dp0data\logs\watcher.log" 2>&1"
exit