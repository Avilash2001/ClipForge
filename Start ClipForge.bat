@echo off
setlocal
set "PATH=%~dp0cuda-libs;%PATH%"
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo ClipForge is not set up. Double-click Setup.bat first.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "%~dp0app.py"
