#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="$(command -v python3.12 || command -v python3.11 || command -v python3 || true)"
if [[ -z "$PYTHON" ]]; then
  echo "Python 3.11 or 3.12 was not found. Install one and rerun ./setup.sh."
  exit 1
fi
"$PYTHON" - <<'PY'
import sys
if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit("ClipForge requires Python 3.11 or 3.12.")
print("Using Python", sys.version.split()[0])
PY

if [[ ! -x .venv/bin/python ]]; then
  echo "Creating virtual environment..."
  "$PYTHON" -m venv .venv
fi

VENV_PY=".venv/bin/python"
if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
  echo "pip is missing or damaged; repairing with Python's bundled ensurepip..."
  "$VENV_PY" -m ensurepip --upgrade
fi
"$VENV_PY" -m pip --version

echo "Installing project dependencies (pip itself will NOT be upgraded)..."
"$VENV_PY" -m pip install --disable-pip-version-check -r requirements.txt
mkdir -p input output logs

echo
echo "ClipForge Python setup complete."
for tool in ffmpeg ffprobe deno ollama; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "External dependency not found: $tool"
  fi
done
