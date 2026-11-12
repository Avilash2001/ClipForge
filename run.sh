#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ ! -x .venv/bin/python ]]; then
  echo "Run ./setup.sh first."
  exit 1
fi
mkdir -p logs input output
exec .venv/bin/python app.py
