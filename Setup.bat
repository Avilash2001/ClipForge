@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title ClipForge Setup

echo ================================================
echo   ClipForge v1.7 - Windows Setup
echo ================================================
echo.
echo IMPORTANT: Close all ClipForge windows and Python terminals using this folder.
echo This setup does NOT upgrade pip itself, avoiding Windows file-lock errors.
echo.

echo Detecting Python 3.12 or 3.11...
set "PYTHON_LAUNCHER="

where py >nul 2>nul
if not errorlevel 1 (
  py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,12) else 1)" >nul 2>nul
  if not errorlevel 1 set "PYTHON_LAUNCHER=py -3.12"
  if not defined PYTHON_LAUNCHER (
    py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,11) else 1)" >nul 2>nul
    if not errorlevel 1 set "PYTHON_LAUNCHER=py -3.11"
  )
)

if not defined PYTHON_LAUNCHER (
  where python >nul 2>nul
  if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if (3,11) <= sys.version_info[:2] < (3,13) else 1)" >nul 2>nul
    if not errorlevel 1 set "PYTHON_LAUNCHER=python"
  )
)

if not defined PYTHON_LAUNCHER (
  echo No supported Python 3.11 or 3.12 installation was detected.
  echo Install with: winget install --id Python.Python.3.12 -e
  echo Then reopen this window and run Setup.bat again.
  goto failed
)

echo Using %PYTHON_LAUNCHER%

if exist ".venv\Scripts\python.exe" (
  echo Checking existing virtual environment...
  ".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if (3,11) <= sys.version_info[:2] < (3,13) else 1)" >nul 2>nul
  if errorlevel 1 (
    echo Existing .venv uses an unsupported Python version.
    echo Rename or remove ONLY .venv, then rerun Setup.bat.
    goto failed
  )
) else (
  echo Creating virtual environment...
  %PYTHON_LAUNCHER% -m venv .venv
  if errorlevel 1 goto failed
)

set "VENV_PY=.venv\Scripts\python.exe"

echo Checking pip without upgrading it...
"%VENV_PY%" -m pip --version >nul 2>nul
if errorlevel 1 (
  echo pip is missing or damaged. Trying Python's bundled ensurepip repair...
  "%VENV_PY%" -m ensurepip --upgrade
  if errorlevel 1 (
    echo ensurepip could not repair pip. Backing up the broken virtual environment...
    set "BACKUP_NAME=.venv-broken-%RANDOM%"
    call :backup_and_recreate
    if errorlevel 1 goto failed
  )
)

"%VENV_PY%" -m pip --version
if errorlevel 1 (
  echo pip is still unavailable after repair.
  goto failed
)

echo.
echo Installing project dependencies (pip itself will NOT be upgraded)...
"%VENV_PY%" -m pip install --disable-pip-version-check --no-warn-script-location -r requirements.txt
if errorlevel 1 goto failed

where nvidia-smi >nul 2>nul
if not errorlevel 1 (
  echo.
  echo NVIDIA GPU detected. Checking the CUDA libraries for GPU transcription...
  "%VENV_PY%" -c "import asr, sys; ok, why = asr.cuda_status(); print(why); sys.exit(0 if ok else 1)"
  if errorlevel 1 (
    echo Installing CUDA 12 / cuDNN 9 runtime wheels for GPU Whisper - about 1.5 GB...
    "%VENV_PY%" -m pip install --disable-pip-version-check --no-warn-script-location -r requirements-gpu.txt
    if errorlevel 1 echo WARNING: GPU libraries could not be installed. Whisper will run on the CPU.
  )
)

if not exist "input" mkdir input
if not exist "output" mkdir output
if not exist "logs" mkdir logs

echo.
echo Checking external tools...
where ffmpeg >nul 2>nul
if errorlevel 1 echo WARNING: FFmpeg not found. Install with: winget install --id Gyan.FFmpeg.Shared -e
where deno >nul 2>nul
if errorlevel 1 echo WARNING: Deno not found. Install with: winget install --id DenoLand.Deno -e
where ollama >nul 2>nul
if errorlevel 1 echo WARNING: Ollama not found. Install from https://ollama.com/download/windows

echo.
echo Setup completed. Python environment is ready.
echo Then run: Start ClipForge.bat
pause
exit /b 0

:backup_and_recreate
ren ".venv" "%BACKUP_NAME%"
if errorlevel 1 (
  echo Could not rename .venv. Make sure all Python/ClipForge processes are closed.
  exit /b 1
)
%PYTHON_LAUNCHER% -m venv .venv
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip --version
if errorlevel 1 exit /b 1
exit /b 0

:failed
echo.
echo Setup failed. The input, output and logs folders have not been deleted.
echo Copy the error above for troubleshooting.
pause
exit /b 1
