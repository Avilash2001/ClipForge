@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ClipForge is not set up. Double-click Setup.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" app.py
pause
