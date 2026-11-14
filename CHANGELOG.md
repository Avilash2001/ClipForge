# Changelog

## v1.6.0

Picking quality. A review of v1.5 output found clips that opened on loading/buy screens, titles that overstated what happened, and banter about skins scored as the best moment.

- **Grounded picks.** The picker must quote the payoff line word for word. Moments whose quote is not actually in the transcript are dropped. The prompt now lists what counts as filler: settings, skins, mic/ping talk, buying, menus, inside jokes, and planning with no outcome.
- **Edge check.** Each finalist's first and last seconds are labelled by the vision model, and clips are trimmed so they never open on a menu, loading, scoreboard or death screen, or end on a menu. Hopeless clips are dropped.
- **Critic pass.** The text model re-judges the finalists side by side, as a stranger scrolling Shorts would, and states what actually happens in each clip. It fact-checks every title against the speech and frames and rewrites overstated titles. Its score feeds the final ranking and the Min score filter.
- **Picking model.** A separate Ollama model is used for picking and the critic. **Auto** picks the best installed judge (`gemma4:12b` if present) and otherwise uses the vision model. In side-by-side tests, qwen3.5:9b and qwen2.5-coder:14b both accepted a false "one bullet kills the entire squad" title. gemma4:12b correctly saw that the streamer was the one being killed.
- A title the critic marks as unsupported is rewritten, or removed if no accurate rewrite is given. It is never burned in as-is.
- Overlapping duplicate words from Whisper no longer stack two captions on top of each other.
- **Only the minutes it needs.** With a usable transcript, moments are picked first and analysis frames are decoded only around them, instead of decoding a frame for every second of the whole stream. The full scan is still used when there is no speech.
- **Story games.** The picker, visual reviewer and critic are told that cutscene dialogue belongs to the game's characters. Titles describe the scene, e.g. "Prison fight breaks out over a MOM joke", instead of crediting the streamer with a character's lines.
- **Uneven and moving split-screen.** A divider anywhere in the middle half of the frame is detected per clip, not only one at the exact centre. A divider that stays centred means 50/50 co-op, which is stacked. A divider that is off-centre, or that slides around within the clip (as A Way Out's does), gets a crop inside the panel that is visible in every frame, so the divider never lands mid-clip.
- **Fix:** two jobs on the same video (e.g. the app and the CLI) could pick the same output folder and overwrite each other's `clips.json`. A job now claims its folder when it starts.
- **No padding.** If no moment clears the Min score, the job finishes without rendering a known-bad clip and the app says so. Only a near miss (within 15 points) is still shown.
- **Transcript cache.** Re-running a video with the same Whisper settings reuses its transcript instead of re-transcribing. The cache is in `cache/transcripts`; use **Re-transcribe** or `--fresh-transcript` to force a new transcription.
- Snapping to sentences respects hard start/end limits, so trimming can never snap back onto a loading screen.

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
