# ClipForge — Local Gaming Clipping Studio

**Turn a gaming VOD you own or are authorised to use into short-form vertical clips, using local AI.**

ClipForge is a cross-platform Python desktop app for Windows, macOS and Linux. It uses faster-whisper for speech transcription, Ollama with a vision-language model to review gameplay frames and speech context, and FFmpeg for rendering. The app does not require paid APIs or a subscription.

> **Project status:** experimental / community project. This release is a functional local-first workflow, not a guaranteed automatic viral-clip detector. Always review clips before publishing.

## Features

- Windows, macOS and Linux source setup scripts.
- Optional YouTube download through yt-dlp, including browser-session cookies.
- Local MP4/MKV/WebM/MOV/AVI input.
- Mixed-language transcription mode for English, Bengali and Hindi using faster-whisper's per-segment language detection.
- Transcript-quality checks that reject punctuation-only output, plus a Medium-model retry and stereo left/right channel fallback when speech is missing from a mono downmix.
- Visual activity analysis from sampled frames.
- Multimodal candidate review: Ollama receives chronological 3×3 frame contact sheets plus transcript context before/during/after each candidate.
- Story-aware scoring for setup, escalation, payoff and reaction; explicit rejection of stream starting screens, menus, lobbies, loading screens and scoreboards.
- Non-overlapping candidate selection.
- 9:16 output with either full-gameplay fit / blurred background or fill / center crop.
- Optional word-timed karaoke-style ASS captions (burned in when the local FFmpeg build has libass) and SRT/ASS sidecars.
- Audio loudness normalisation, thumbnails, and per-clip social title/caption/hashtag suggestions from the local vision model.
- Live progress, ETA, logs and cancellation in the GUI.
- FFmpeg child processes are isolated from terminal close/Ctrl+C events; the GUI warns before closing during an active job.
- Local `input/`, `output/` and `logs/` runtime directories are excluded from Git.

## What it does not do

- It does not guarantee detection of kills, clutches or other game-specific events in every game. It judges sampled video frames and available transcript/audio signals. Very brief events can fall between sampled frames.
- Speech recognition can still struggle with rapid English/Bengali/Hindi code-switching, overlapping Discord voices, noisy microphones, or game audio much louder than speech. If no usable speech is present in the recording's audio stream, captions and transcript-based context are skipped.
- Visual AI reviews contact sheets, not every frame as a human editor would.
- Social titles and hashtags are suggestions, not factual guarantees.
- Publishing to YouTube Shorts / Instagram Reels is manual. There is no automatic upload integration.
- No copyrighted music or stock sound effects are bundled. The original game/commentary audio is preserved and can be normalised.

## Requirements

### Required on all operating systems

1. **Python 3.11 or 3.12** (both are supported).
2. **FFmpeg and ffprobe** available on `PATH`.
3. **Ollama** installed and running for visual AI ranking.
4. A local vision-capable Ollama model. The default is `qwen3.5:9b`.
5. **Deno 2.3+** recommended for yt-dlp's YouTube JavaScript challenge solver. `yt-dlp[default]` installs the matching `yt-dlp-ejs` scripts; Deno is a separate runtime.

Internet access is required to download YouTube source videos and first-time Whisper/Ollama models. Transcription, visual analysis, ranking and rendering run locally after setup.

### Hardware guidance

- 16 GB RAM is a practical minimum; 32 GB is recommended for larger videos and local vision models.
- An NVIDIA GPU with NVENC accelerates rendering on supported systems. CPU rendering is used automatically when NVENC is unavailable.
- The default Whisper device is CPU to avoid CUDA/cuDNN setup complexity. CUDA transcription is optional and requires a CTranslate2-compatible CUDA/cuDNN environment.
- Model memory depends on the Ollama model/quantisation. `qwen3.5:9b` is the default; smaller Qwen3.5 vision tags can be selected in the app if memory is constrained.

## Install from GitHub

> **Installer note:** Setup installs dependencies using the pip bundled with the virtual environment. It deliberately does not self-upgrade pip, avoiding Windows file-lock failures while pip replaces its own files. If pip is incomplete, Windows setup tries `ensurepip`; if repair fails, it backs up the broken `.venv` before creating a clean one. Runtime media and logs are not deleted.


### Windows 10/11

1. Install **Python 3.11 or 3.12 x64** from <https://www.python.org/downloads/>. Enable **Add python.exe to PATH**. `Setup.bat` automatically tries Python 3.12 first, then 3.11. If you already have Python 3.12, you do not need to install 3.11.
2. Install FFmpeg (including ffprobe). With winget:

   ```powershell
   winget install --id Gyan.FFmpeg.Shared -e
   ```

3. Install Deno:

   ```powershell
   winget install --id DenoLand.Deno -e
   ```

4. Install Ollama from <https://ollama.com/download/windows>, then download the model:

   ```powershell
   ollama pull qwen3.5:9b
   ```

5. Clone/download this repository and open its folder:

   ```powershell
   git clone https://github.com/YOUR-USERNAME/ClipForge.git
   cd ClipForge
   ```

6. Double-click `Setup.bat`. It creates `.venv`, installs Python dependencies and creates ignored `input/`, `output/` and `logs/` directories.
7. Double-click `Start ClipForge.bat`.

For console diagnostics, use `Start ClipForge (Debug).bat`.

### macOS (Apple Silicon or Intel)

Install Homebrew if you do not have it: <https://brew.sh/>. Then install system tools:

```bash
brew install python@3.11 ffmpeg deno
```

Install Ollama from <https://ollama.com/download/mac> (or its current official macOS installer), launch it, and run:

```bash
ollama pull qwen3.5:9b
```

Clone the repository and install:

```bash
git clone https://github.com/YOUR-USERNAME/ClipForge.git
cd ClipForge
chmod +x setup.sh run.sh launch.command
./setup.sh
```

Launch with `./run.sh` from Terminal or double-click `launch.command` in Finder. The `.command` launcher runs the GUI detached and writes its console log to the ignored `logs/gui.log` file.

### Linux (Ubuntu/Debian example)

Install system packages:

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-tk ffmpeg curl
```

Install Deno using its official installer:

```bash
curl -fsSL https://deno.land/install.sh | sh
```

Add Deno to your shell `PATH` if the installer asks. Install Ollama using its official instructions at <https://ollama.com/download/linux>, then:

```bash
ollama pull qwen3.5:9b
```

Clone and install:

```bash
git clone https://github.com/YOUR-USERNAME/ClipForge.git
cd ClipForge
chmod +x setup.sh run.sh
./setup.sh
./run.sh
```

For Fedora, Arch or other distributions, install equivalent packages for Python 3.11 or 3.12, Tkinter, FFmpeg and ffprobe. The GUI requires a working desktop session and Tk support.

## First run

1. Ensure Ollama is running and the selected vision model is present (`ollama list`).
2. Select either a YouTube URL or a local video file. For your own VODs, local input is the most reliable route.
3. If downloading from YouTube, select the browser in which you are signed in. Fully exit the browser before starting if yt-dlp reports a cookie database lock. Browser cookies are sensitive credentials; never export/share/commit them.
4. Set the game profile, clip count, duration and scan interval.
5. Set speech language to **Mixed (English + Bengali + Hindi)** for code-switched voice chat.
6. Keep local Ollama visual AI enabled. Use `Fit + blurred background` to preserve the full HUD; use `Fill (center crop)` only when cropping game edges is acceptable.
7. Leave **Burn in animated word-highlight captions** enabled if you want captions. Only validated speech is used. If FFmpeg lacks libass, ClipForge keeps sidecar caption files instead.
8. Start the job and keep the GUI open. The window intentionally refuses to close while a job is active; use **Cancel job** and wait for the safe-stop message.

## Output layout

Each completed job is saved in `output/<video-name>/`:

```text
clip_01_000120s.mp4
clip_01_000120s.srt        # when speech was validated
clip_01_000120s.ass        # when word-timed captions were generated
clip_01_000120s.jpg        # thumbnail
clip_01_000120s.metadata.json
...
transcript.json
clips.json
```

`clips.json` includes timestamps, scores, screen classification, story explanation, generated social copy and caption/render status. Review the actual clips before publication.

## Command line (optional)

The GUI is the main workflow. A basic CLI is also available:

```bash
python clipper.py "D:/Videos/my-vod.mp4" --count 10 --clip-length 35 --stride 15
```

Inside the project environment, use `.venv/Scripts/python` on Windows or `.venv/bin/python` on macOS/Linux.

## Git hygiene

The `.gitignore` excludes the runtime `input/`, `output/` and `logs/` folders, virtual environments, caches, generated media, transcripts, model files and cookie exports. These folders are created locally by the setup script and should never be committed.

Before pushing, check:

```bash
git status --short
```

Do not force-add ignored media, logs, browser databases, cookies or model files.

## Build desktop release packages

PyInstaller cannot cross-compile a Windows executable from macOS/Linux or vice versa. The GitHub Actions workflow builds on a runner for each OS when you push a version tag. See PyInstaller's platform requirements: <https://pyinstaller.org/en/latest/requirements.html>.

To publish a release from your own fork:

```bash
git add .
git commit -m "Release ClipForge 1.3.2"
git push origin main
git tag v1.3.2
git push origin v1.3.2
```

The workflow builds Windows, macOS and Linux artifacts and attaches them to a GitHub Release. The app bundles do **not** include FFmpeg, Ollama, the Qwen model, or the Whisper model weights; users install/download those separately. For source installs, users can simply clone and run the OS setup script.

## Troubleshooting

- **Windows setup reports `WinError 32` involving `pip\_vendor` or a file being used by another process:** close every ClipForge window and Python terminal using that project folder, then run the updated `Setup.bat` again. v1.3.2 no longer upgrades pip as part of setup. If pip was damaged by an earlier failed self-upgrade, setup tries `ensurepip`; if that fails it preserves the old environment as `.venv-broken-*` and creates a clean `.venv`.

- **`0xC000013A` while rendering on Windows:** FFmpeg was interrupted by a console-control/terminal shutdown event. Use the normal `Start ClipForge.bat` launcher and keep the GUI open during a job. The FFmpeg child process is launched without a console window. The normal launcher uses `pythonw`; FFmpeg is started without a console window.
- **YouTube says sign in / bot check:** Update yt-dlp with `pip install -U "yt-dlp[default]"`, install Deno 2.3+, use a signed-in browser session, and only download content you are authorised to use.
- **Chrome DPAPI error on Windows:** Try Firefox or Edge. Some Chrome installations protect cookies in ways yt-dlp cannot decrypt from the current user context.
- **Transcript empty or punctuation-only:** ClipForge tests the mixed-language mode, checks left/right stereo channels when available, retries Whisper Medium, and then omits transcript-based ranking/captions if speech remains unusable. This can indicate the stream recording did not include Discord/microphone audio, voice is too quiet, or speakers overlap heavily.
- **Ollama returns invalid/empty JSON:** v1.3 requests structured JSON with thinking disabled and retries batches in smaller groups. Update Ollama and verify the model supports images. The app logs and falls back to visual/audio scoring if local vision ranking is unavailable.
- **Model not found:** run `ollama pull qwen3.5:9b` or choose a model name shown by `ollama list`.
- **FFmpeg has no ASS filter:** SRT/ASS caption files remain as sidecars but captions cannot be burned in. Install an FFmpeg build with libass enabled.
- **NVENC unavailable:** CPU H.264 encoding is selected automatically. On Linux, install an NVIDIA driver and an FFmpeg build with NVENC support.
- **GUI fails on Linux:** install the distribution's Tk package (`python3-tk` on Ubuntu/Debian) and run from a graphical desktop session.

## License

ClipForge project code is MIT-licensed. Third-party packages, FFmpeg, Whisper model weights, Ollama models and any source videos retain their own licences and terms. See `LICENSE` and the respective upstream projects.


### Python version detection on Windows

`Setup.bat` detects Python 3.12 first, then 3.11, through the Python launcher. It also accepts a `python` executable on `PATH` if it is 3.11 or 3.12. If neither is found, install Python 3.12 with `winget install --id Python.Python.3.12 -e`, reopen the terminal, and rerun setup.
