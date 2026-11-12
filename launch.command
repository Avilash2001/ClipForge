#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ ! -x .venv/bin/python ]]; then
  osascript -e 'display dialog "Run setup.sh in the ClipForge folder first." buttons {"OK"}'
  exit 1
fi
mkdir -p logs input output
nohup .venv/bin/python app.py >> logs/gui.log 2>&1 </dev/null &
