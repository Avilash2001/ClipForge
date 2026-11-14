"""Runtime path policy: source runs use ignored repo folders; frozen builds use user data."""
from pathlib import Path
import sys

from platformdirs import user_data_dir

PROJECT_DIR = Path(__file__).resolve().parent
if getattr(sys, "frozen", False):
    DATA_DIR = Path(user_data_dir("ClipForge", appauthor=False))
else:
    DATA_DIR = PROJECT_DIR

INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
LOG_DIR = DATA_DIR / "logs"
CACHE_DIR = DATA_DIR / "cache"  # saved transcripts so re-runs skip Whisper

for directory in (INPUT_DIR, OUTPUT_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)
