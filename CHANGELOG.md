# Changelog

## 1.3.2

- Fix Windows setup detection: automatically selects Python 3.12 first, then Python 3.11.
- Accepts a supported `python` executable on PATH when the Python launcher is unavailable.
- Validates an existing virtual environment before installing dependencies and gives a safe recovery instruction without touching media folders.
- Aligns the Windows setup script with the already-declared Python 3.11/3.12 support range.

## 1.3.0

- Reworked project distribution for source installs on Windows, macOS and Linux.
- Added a GitHub Actions workflow to build per-OS desktop artifacts from version tags.
- Added `.gitignore` rules for local input/output/logs, model caches, generated media and cookie exports.
- Isolated FFmpeg/ffprobe processes from terminal Ctrl+C and console close events; added a GUI close guard during active jobs.
- Added multilingual ASR quality recovery: per-segment language detection, left/right stereo channel retry, and Whisper Medium retry.
- Expanded multimodal context review to nine chronological frames plus transcript context before/during/after the candidate.
- Switched Ollama review to structured JSON output, disabled thinking for response parsing, added smaller-batch retry, and scored setup/escalation/payoff/story coherence.
- Added menu/loading/starting-screen rejection.
- Added word-timed karaoke ASS captions, SRT/ASS sidecars, audio loudness normalisation, thumbnails and per-clip social metadata suggestions.
- Added portable user-data paths for packaged applications and OS-specific folder opening.

## 1.2.0

- Mixed-language Whisper transcription, transcript-quality checks, visual-motion analysis and local vision ranking were introduced.

## 1.1.1

- Added browser-cookie selection for YouTube downloads and URL paste protection.
