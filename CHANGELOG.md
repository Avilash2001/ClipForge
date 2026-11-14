# Changelog

## v1.5.0

Picking:

- Transcript-first moment picker. The local LLM reads the transcript in 12-minute chunks, with `[LOUD]` and `[CHAT SPIKE]` markers. It picks shareable moments (first reactions, discoveries, freakouts, fails, hot takes, clutch plays) and scores each one 0-100.
- Scoring is harsh. Moments that need earlier context are scored below 30, and a **Min score** filter (default 60) keeps only moments that clear the bar.
- Clips are variable length (20-60s by default). They start on a full sentence with leading filler ("uh, so yeah") trimmed, and end after the payoff plus a short reaction tail.
- New signals: loudness spikes measured against a rolling one-minute baseline, and YouTube live-chat replay velocity (live-stream replays only).
- Transcript clean-up drops hallucinated Whisper output before it reaches ranking or burned-in captions. This covers low-confidence segments, characters from scripts outside the selected language, character floods and repetition loops.

Look:

- New default layout, **Auto (facecam if found)**. A detected streamer webcam goes on top and centre-cropped gameplay goes below. Without a webcam the layout falls back to **Gameplay zoom**, which crops gameplay to fill most of the frame.
- New layout, **Co-op split-screen (stacked)**. Side-by-side co-op footage is detected per clip and re-stacked vertically.
- Letterbox bars are trimmed per clip.
- Manual **Facecam** corner presets for streams where auto-detection misses.
- An AI hook title is burned on the video as a white rounded "sticker" card, with emoji support.
- Captions are now big uppercase groups of 2-4 words. The spoken word lights up yellow and each group pops in.
- Thumbnails are taken from the finished vertical clip. A `.post.txt` file is written for each clip.

Other:

- Review & Publish shows vertical previews, the 0-100 score, category and layout.
- The URL field accepts Twitch VOD and Kick links as well as YouTube (downloaded through yt-dlp).
- `opencv-python-headless` is pinned below 5, because OpenCV 5 removed the bundled face detector. Re-run setup to update an existing environment.

## v1.4.0

- Added Story Auto for adaptive clip count, longer story windows and denser scanning.
- Tightened visual-AI criteria to favor complete standalone arcs and reject unfinished action fragments.
- De-duplicate nearby cuts from the same event and add soft audio/video transitions.
- Added the in-app Review & Publish gallery with thumbnail previews, play/copy-post actions and persistent Posted tracking.

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
