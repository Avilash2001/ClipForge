from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
import base64
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import requests
import yt_dlp

import asr
import clip_style
import sources
from clip_style import (
    FACECAM_AUTO, LAYOUT_AUTO, LAYOUT_SPLIT, Box,
)
from moments import (
    Moment, SignalWindow, chat_spikes, clean_transcript, critique, extract_json_object, loudness_spikes,
    motion_spikes, ollama_json, parse_live_chat, pick_moments, range_peak, signal_windows, snap_to_speech,
    tidy_hook_title, title_risks, verify_titles, LITERAL_TITLE_RULES,
)
from paths import CACHE_DIR, INPUT_DIR, OUTPUT_DIR

DEFAULT_INPUT_DIR = INPUT_DIR
DEFAULT_OUTPUT_DIR = OUTPUT_DIR
OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"
APP_VERSION = "1.7.0"
DEFAULT_MIN_SCORE = 60
YOUTUBE_ID_IN_NAME = re.compile(r"\[([A-Za-z0-9_-]{11})\]")
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}

HYPE_WORDS = {
    "kill", "killed", "clutch", "win", "won", "victory", "headshot",
    "double", "triple", "quad", "wipe", "destroyed", "insane", "crazy",
    "no way", "let's go", "lets go", "oh my god", "what", "bro",
    "legendary", "boss", "finally", "again", "help", "run", "dead",
}

ProgressCallback = Callable[[str, float, float, str], None]
LogCallback = Callable[[str], None]


class JobCancelled(Exception):
    """Raised when the user requests cancellation between processing stages."""


@dataclass
class Candidate:
    index: int
    start: float
    end: float
    audio_score: float
    transcript: str
    transcript_score: float
    visual_score: float = 0.0
    final_score: float = 0.0
    reason: str = ""
    screen_type: str = "unknown"
    story_score: float = 0.0
    event_type: str = "unknown"
    hook: str = ""
    setup: str = ""
    payoff: str = ""
    social_title: str = ""
    social_caption: str = ""
    hashtags: list[str] = None
    context_before: str = ""
    context_after: str = ""
    # v1.5: transcript-first moments, spike signals and publish-readiness.
    origin: str = "window"        # "moment" (transcript pick), "signal" (spike, maybe silent) or "window" (scan)
    category: str = ""
    hook_title: str = ""          # burned-in on-video title
    moment_score: float = 0.0     # 0..100 from the transcript pass
    loud_score: float = 0.0       # 0..1 peak loudness spike in the clip
    chat_score: float = 0.0       # 0..1 peak live-chat spike in the clip
    quality: float = -1.0         # 0..100 publish-readiness; -1 = not judged by AI
    split_screen: bool = False
    content_band: tuple = (0.0, 1.0)
    seam_x: Optional[float] = None   # position of an off-centre split-screen divider
    critic_score: float = -1.0     # 0..100 from the side-by-side critic pass; -1 = not judged
    critic_note: str = ""
    edge_labels: list = None        # screen types of the first/last frames
    rejected: str = ""              # why a final check threw this clip out; rejected clips are never rendered
    signal_strength: float = 0.0    # 0..1 combined loudness/chat/motion spike (signal-driven candidates)
    title_claims: list = None       # the critic's claim-by-claim fact-check of the title
    critic_title: str = ""          # the critic's own literal title (written without seeing ours)
    critic_frames: str = ""         # what the critic saw in the frames
    payoff_quote: str = ""          # the picker's verbatim payoff line from the transcript (evidence, not a claim)


def _emit(progress: Optional[ProgressCallback], stage: str, stage_pct: float,
          overall_pct: float, message: str) -> None:
    if progress:
        progress(stage, max(0.0, min(100.0, stage_pct)),
                 max(0.0, min(100.0, overall_pct)), message)


def _log(log: Optional[LogCallback], message: str) -> None:
    if log:
        log(message)


def _check_cancel(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event and cancel_event.is_set():
        raise JobCancelled("Job cancelled by user.")


def _run_process(command: list[str], **kwargs):
    """Isolate FFmpeg/ffprobe from terminal Ctrl+C and terminal-close events."""
    if os.name == "nt":
        flags = kwargs.pop("creationflags", 0)
        kwargs["creationflags"] = flags | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kwargs.setdefault("start_new_session", True)
    return subprocess.run(command, **kwargs)


def _subprocess_failure(exc: subprocess.CalledProcessError, stage: str) -> RuntimeError:
    if getattr(exc, "returncode", None) in (3221225786, -1073741510):
        return RuntimeError(
            f"{stage} was interrupted by a terminal/CTRL+C event. Keep ClipForge open during a job; "
            "this version launches FFmpeg without attaching it to the terminal."
        )
    detail = (getattr(exc, "stderr", None) or "").strip()
    return RuntimeError(f"{stage} failed (exit code {exc.returncode}): {detail[-1800:] or 'No FFmpeg details reported.'}")


def check_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            "Missing required command(s): " + ", ".join(missing) +
            ". Install FFmpeg, reopen the app, and try again."
        )


AUTO = "Auto"
# Text models that judged clips best in testing, best first. Used when the picking model is Auto.
PICKER_PREFERENCE = ["gemma4:12b", "qwen3.5:27b", "gemma4:27b"]
# Vision models for the contact-sheet review, best first. In v1.7 testing gemma4:12b got who-did-what right
# where qwen3.5:9b called the streamer's own death a "one bullet clutch".
VISION_PREFERENCE = ["gemma4:12b", "qwen3.5:9b", "gemma4:27b", "qwen3.5:27b", "qwen2.5vl:7b"]
# Edge check (label the first/last frames): qwen3.5:9b made every trim decision right on 72 hand-labelled
# frames (gemma4:12b missed a spectate screen under a "WON" banner).
EDGE_PREFERENCE = ["qwen3.5:9b", "gemma4:12b", "qwen3.5:27b", "gemma4:27b", "qwen2.5vl:7b"]


def installed_ollama_models() -> list[str]:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        response.raise_for_status()
        return [m.get("name", "") for m in response.json().get("models", [])]
    except Exception:
        return []


def ollama_capabilities(model: str) -> set[str]:
    """Capabilities Ollama reports for a model (e.g. {"completion", "vision"}); empty if unknown."""
    try:
        response = requests.post(f"{OLLAMA_URL}/api/show", json={"model": model}, timeout=5)
        response.raise_for_status()
        return {str(c) for c in response.json().get("capabilities", []) or []}
    except Exception:
        return set()


def preferred_picker_model(fallback: str) -> str:
    """Best installed judge for transcript picking and the critic; falls back to the vision model."""
    installed = set(installed_ollama_models())
    for name in PICKER_PREFERENCE:
        if name in installed:
            return name
    return fallback


def preferred_vision_model(fallback: str = DEFAULT_OLLAMA_MODEL, preference: Optional[list[str]] = None) -> str:
    """Best installed model that can see images (frame review by default; pass EDGE_PREFERENCE for edges)."""
    installed = installed_ollama_models()
    for name in preference or VISION_PREFERENCE:
        if name in installed:
            return name
    for name in installed:
        if "vision" in ollama_capabilities(name):
            return name
    return fallback


def resolve_models(vision_choice: str, picker_choice: Optional[str],
                   log: Optional[LogCallback] = None) -> tuple[str, str, str]:
    """Turn "Auto" (or empty) model choices into installed Ollama model names.

    Returns (frame-review model, picking/critic model, edge-check model). With a vision model chosen
    by hand, the edge check uses it too.
    """
    vision = (vision_choice or "").strip()
    if not vision or vision.lower() == AUTO.lower():
        vision = preferred_vision_model()
        edge = preferred_vision_model(vision, EDGE_PREFERENCE)
    else:
        edge = vision
        if ollama_capabilities(vision) and "vision" not in ollama_capabilities(vision):
            _log(log, f"WARNING: {vision} cannot read images; the frame review needs a vision model.")
    picker = (picker_choice or "").strip()
    if not picker or picker.lower() == AUTO.lower():
        picker = preferred_picker_model(vision)
    return vision, picker, edge


def unload_ollama_models(log: Optional[LogCallback] = None) -> list[str]:
    """Ask Ollama to release GPU memory now (before Whisper runs on the same GPU)."""
    try:
        loaded = [m.get("name", "") for m in requests.get(f"{OLLAMA_URL}/api/ps", timeout=2).json().get("models", [])]
    except Exception:
        return []
    for name in loaded:
        try:
            requests.post(f"{OLLAMA_URL}/api/generate", json={"model": name, "keep_alive": 0}, timeout=15)
        except Exception:
            pass
    if loaded:
        _log(log, f"Freed GPU memory held by Ollama ({', '.join(loaded)}) for Whisper.")
    return loaded


def get_ollama_status(model: str = DEFAULT_OLLAMA_MODEL) -> tuple[bool, str]:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        response.raise_for_status()
        names = [m.get("name", "") for m in response.json().get("models", [])]
        found = any(name == model or name.startswith(model + ":") for name in names)
        if found:
            return True, f"Ollama online · {model} available"
        return True, f"Ollama online · model {model} not found (run: ollama pull {model})"
    except Exception:
        return False, "Ollama offline · start Ollama to use AI ranking"


def download_youtube(url: str, destination: Path,
                     progress: Optional[ProgressCallback] = None,
                     cancel_event: Optional[threading.Event] = None,
                     log: Optional[LogCallback] = None,
                     cookies_browser: Optional[str] = None,
                     info_out: Optional[dict] = None) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    before = {p.resolve() for p in destination.iterdir() if p.is_file()}
    last_report = 0.0

    def hook(info: dict) -> None:
        nonlocal last_report
        _check_cancel(cancel_event)
        status = info.get("status")
        if status == "downloading":
            total = info.get("total_bytes") or info.get("total_bytes_estimate")
            downloaded = info.get("downloaded_bytes", 0)
            pct = (downloaded / total * 100) if total else 0
            now = time.monotonic()
            if now - last_report >= 0.2 or pct >= 99.9:
                speed = info.get("speed")
                eta = info.get("eta")
                extra = f" · {speed / 1024 / 1024:.1f} MB/s" if speed else ""
                if eta is not None:
                    extra += f" · about {int(eta)} sec left"
                _emit(progress, "Downloading", pct, 2 + pct * 0.08,
                      f"Downloading video: {pct:.1f}%{extra}")
                last_report = now
        elif status == "finished":
            _emit(progress, "Downloading", 100, 10, "Download complete; preparing media file")

    class QuietLogger:
        def debug(self, msg):
            pass
        def warning(self, msg):
            _log(log, f"yt-dlp warning: {msg}")
        def error(self, msg):
            _log(log, f"yt-dlp error: {msg}")

    opts = {
        "format": "bv*+ba/b",
        "outtmpl": str(destination / "%(title).120B [%(id)s].%(ext)s"),
        "noplaylist": True,
        "windowsfilenames": True,
        "merge_output_format": "mp4",
        "progress_hooks": [hook],
        "logger": QuietLogger(),
        "quiet": True,
        "no_warnings": False,
        "overwrites": False,
    }
    if cookies_browser:
        # yt-dlp's Python API expects (browser, profile, keyring, container).
        # Read the user's existing browser session locally; never export or log cookie values.
        opts["cookiesfrombrowser"] = (cookies_browser.lower(), None, None, None)
        _log(log, f"Using the signed-in {cookies_browser} browser session for YouTube authentication.")
        _log(log, "If cookie extraction reports a database lock, fully exit that browser and retry.")
    else:
        _log(log, "No browser cookies selected; YouTube may require sign-in for some videos.")
    _log(log, "Starting yt-dlp download. Only download videos you own or are authorised to use.")
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:
        error_text = str(exc)
        lowered = error_text.lower()
        if "sign in to confirm" in lowered or "use --cookies-from-browser" in lowered or "cookies for the authentication" in lowered:
            guidance = (
                "YouTube blocked the anonymous download. In ClipForge, set 'YouTube login cookies' "
                "to the browser where you are signed into YouTube, then retry. Before retrying, "
                "open youtube.com in that browser and confirm the video plays. On Windows, fully "
                "exit the selected browser before starting the job so yt-dlp can read its cookie "
                "database. If you use Chrome and it still fails, try Edge or Firefox. Update yt-dlp "
                "from Setup.bat if the challenge persists. Do not send anyone your browser cookies."
            )
            raise RuntimeError(guidance) from exc
        if "could not copy" in lowered and "cookie" in lowered:
            raise RuntimeError(
                f"Could not read {cookies_browser or 'browser'} cookies. Fully exit that browser "
                "including background processes, then retry. Original error: {error_text}"
            ) from exc
        raise
    _check_cancel(cancel_event)
    if info_out is not None and isinstance(info, dict):
        info_out.update({key: info.get(key) for key in ("id", "title", "channel", "uploader", "was_live")})
    video_id = str(info.get("id", "")) if isinstance(info, dict) else ""
    after = [p for p in destination.iterdir()
             if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS and p.resolve() not in before]
    # Prefer the file containing this video's ID. This also handles a repeat download
    # where yt-dlp reuses a file that was already in the Input directory.
    matching_id = [p for p in destination.iterdir()
                   if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
                   and video_id and video_id in p.name]
    if matching_id:
        return max(matching_id, key=lambda p: p.stat().st_mtime)
    if not after:
        after = [p for p in destination.iterdir()
                 if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    if not after:
        raise RuntimeError("yt-dlp finished but no video file was found in the Input folder.")
    return max(after, key=lambda p: p.stat().st_mtime)


def fetch_live_chat(url: str, work_dir: Path, cookies_browser: Optional[str] = None,
                    log: Optional[LogCallback] = None,
                    cancel_event: Optional[threading.Event] = None) -> list[float]:
    """Download a YouTube live-stream replay's chat and return message offsets in seconds.

    Only replays of YouTube live streams have this; ordinary uploads return an empty list.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    def hook(_info: dict) -> None:
        _check_cancel(cancel_event)

    class QuietLogger:
        def debug(self, msg):
            pass

        def warning(self, msg):
            pass

        def error(self, msg):
            _log(log, f"Live chat: {msg}")

    opts = {
        "skip_download": True,
        "writesubtitles": True,
        "subtitleslangs": ["live_chat"],
        "outtmpl": str(work_dir / "chat.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "logger": QuietLogger(),
        "progress_hooks": [hook],
    }
    if cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser.lower(), None, None, None)
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.extract_info(url, download=True)
    files = sorted(work_dir.glob("*.live_chat.json"))
    if not files:
        return []
    return parse_live_chat(files[0])


def video_dimensions(video: Path) -> tuple[int, int]:
    result = _run_process([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
        "-of", "csv=p=0", str(video)
    ], check=False, capture_output=True, text=True)
    try:
        width, height = (int(v) for v in result.stdout.strip().splitlines()[0].split(",")[:2])
        return width, height
    except (ValueError, IndexError):
        return 1920, 1080


def duration_seconds(video: Path) -> float:
    result = _run_process(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        check=True, text=True, capture_output=True
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Could not read video duration for {video.name}.") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("The selected video has an invalid or zero duration.")
    return duration


def extract_audio_wav(video: Path, wav_path: Path) -> None:
    _run_process([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)
    ], check=True, capture_output=True, text=True)


def audio_channel_count(video: Path) -> int:
    result = _run_process([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=channels", "-of", "default=noprint_wrappers=1:nokey=1",
        str(video)
    ], check=False, capture_output=True, text=True)
    try:
        return int(result.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return 1


def extract_channel_wav(video: Path, wav_path: Path, channel: str) -> None:
    """Extract FL or FR separately for recordings with isolated stereo voice/game tracks."""
    if channel not in {"FL", "FR"}:
        raise ValueError("Unsupported audio channel")
    _run_process([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-map", "0:a:0", "-af", f"pan=mono|c0={channel}", "-ar", "16000",
        "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)
    ], check=True, capture_output=True, text=True)


def audio_energy_by_second(wav_path: Path, raw: bool = False) -> list[float]:
    # NumPy is installed as a dependency of faster-whisper; vectorising this avoids
    # iterating through tens of millions of samples in Python.
    import numpy as np

    with wave.open(str(wav_path), "rb") as wf:
        rate = wf.getframerate()
        width = wf.getsampwidth()
        if width != 2:
            raise RuntimeError("Expected 16-bit PCM audio.")
        values: list[float] = []
        while True:
            raw = wf.readframes(rate)
            if not raw:
                break
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
            if samples.size == 0:
                values.append(0.0)
            else:
                values.append(float(np.sqrt(np.mean(samples * samples)) / 32768.0))
    if not values:
        return [0.0]
    return values if raw else normalise_energy(values)


def normalise_energy(values: list[float]) -> list[float]:
    """Scale raw per-second RMS so the 90th percentile is 1.0."""
    if not values:
        return [0.0]
    ordered = sorted(values)
    ceiling = max(ordered[min(len(ordered) - 1, int(len(ordered) * 0.90))], 0.015)
    return [min(1.0, value / ceiling) for value in values]


def transcribe(audio_path: Path, settings: "asr.WhisperSettings", duration: float,
               progress: Optional[ProgressCallback] = None,
               cancel_event: Optional[threading.Event] = None,
               log: Optional[LogCallback] = None,
               language_mode: str = asr.MIXED):
    """Transcribe the extracted 16 kHz WAV. Returns (raw rows, language, settings actually used).

    GPU failures fall back to the CPU automatically (see asr.transcribe).
    """
    _emit(progress, "Transcribing", 0, 15, f"Loading Whisper {settings.label()}...")

    def on_progress(processed: float, total: float, remaining: Optional[float]) -> None:
        pct = processed / total * 100 if total else 0
        message = f"Transcribed {processed/60:.1f} / {total/60:.1f} min"
        if remaining is not None and pct < 99.9:
            message += f" · ETA {format_eta(remaining)}"
        _emit(progress, "Transcribing", pct, 15 + pct * 0.60, message)

    rows, language, used = asr.transcribe(
        audio_path, settings, duration, language_mode=language_mode, on_progress=on_progress,
        check_cancel=lambda: _check_cancel(cancel_event), log=lambda m: _log(log, m), cancelled=(JobCancelled,))
    _emit(progress, "Transcribing", 100, 75, "Transcription complete")
    return rows, language, used


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def transcript_quality(rows: list[dict]) -> dict:
    """Detect empty/punctuation-only Whisper output before it contaminates ranking."""
    text = " ".join(str(row.get("text", "")) for row in rows)
    words = re.findall(r"[^\W_]+(?:['’\-][^\W_]+)?", text, flags=re.UNICODE)
    alnum_chars = sum(1 for ch in text if ch.isalnum())
    nonspace_chars = sum(1 for ch in text if not ch.isspace())
    ratio = alnum_chars / max(1, nonspace_chars)
    usable = len(words) >= 8 and alnum_chars >= 30 and ratio >= 0.45
    return {
        "usable": usable,
        "word_count": len(words),
        "alphanumeric_characters": alnum_chars,
        "alphanumeric_ratio": round(ratio, 3),
    }


def visual_motion_by_second(video: Path, duration: float, frame_dir: Path,
                            progress: Optional[ProgressCallback] = None,
                            cancel_event: Optional[threading.Event] = None,
                            log: Optional[LogCallback] = None) -> list[float]:
    """
    Decode one small frame per second with FFmpeg (sequential decode), then
    measure frame-to-frame change. Cached thumbnails are reused by visual AI.
    This is a generic activity heuristic, not semantic kill detection.
    """
    try:
        import cv2  # noqa: F401 - availability check for a friendly setup error
        import numpy as np  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("OpenCV is missing. Run Setup.bat to install the visual-analysis dependency.") from exc

    frame_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(frame_dir / "frame_%06d.jpg")
    _log(log, "Sampling one gameplay frame per second for visual analysis...")
    _emit(progress, "Visual analysis", 0, 75.5, "Sampling low-resolution gameplay frames...")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-an",
        "-vf", "fps=1,scale=320:180:flags=bilinear",
        "-q:v", "7", "-start_number", "0", pattern
    ]
    _run_process(cmd, check=True, capture_output=True, text=True)
    frame_files = sorted(frame_dir.glob("frame_*.jpg"))
    if not frame_files:
        raise RuntimeError("FFmpeg could not extract visual-analysis frames.")
    motion = motion_from_frames(frame_dir, duration, progress, cancel_event)
    _log(log, f"Visual motion analysis complete ({len(frame_files)} sampled frames).")
    return motion


def extract_window_frames(video: Path, windows: list[tuple[float, float]], frame_dir: Path,
                          cancel_event: Optional[threading.Event] = None) -> int:
    """Decode 1 fps analysis frames only around candidate moments (not the whole stream).

    Frames are named by their second (frame_000123.jpg), exactly like a full-video pass, so
    contact sheets, motion and layout checks work the same way.
    """
    frame_dir.mkdir(parents=True, exist_ok=True)
    merged: list[list[float]] = []
    for a, b in sorted((max(0.0, a), b) for a, b in windows):
        if merged and a <= merged[-1][1] + 5:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    for a, b in merged:
        _check_cancel(cancel_event)
        first = int(a)
        _run_process([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", str(first), "-i", str(video),
            "-t", f"{max(1.0, b - first):.0f}", "-an", "-vf", "fps=1,scale=320:180:flags=bilinear",
            "-q:v", "7", "-start_number", str(first), str(frame_dir / "frame_%06d.jpg"),
        ], check=True, capture_output=True, text=True)
    return int(sum(b - a for a, b in merged))


def motion_from_frames(frame_dir: Path, duration: float,
                       progress: Optional[ProgressCallback] = None,
                       cancel_event: Optional[threading.Event] = None) -> list[float]:
    """Frame-to-frame change per second (0..1, normalised to the 95th percentile)."""
    import cv2
    import numpy as np

    motion = [0.0] * max(1, int(math.ceil(duration)))
    files = sorted(frame_dir.glob("frame_*.jpg"))
    previous, previous_second = None, None
    for n, frame_path in enumerate(files):
        _check_cancel(cancel_event)
        try:
            second = int(frame_path.stem.split("_")[-1])
        except ValueError:
            continue
        frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            previous = None
            continue
        if previous is not None and previous_second == second - 1 and second < len(motion):
            motion[second] = float(np.mean(cv2.absdiff(frame, previous)) / 255.0)
        previous, previous_second = frame, second
        if progress and (n % 20 == 0 or n == len(files) - 1):
            pct = (n + 1) / max(1, len(files)) * 100
            _emit(progress, "Visual analysis", pct, 75.5 + pct * 0.045,
                  f"Analysing gameplay movement: {n + 1}/{len(files)} sampled frames")
    positive = [v for v in motion if v > 0]
    ceiling = max(float(np.percentile(positive, 95)) if positive else 0.01, 0.008)
    return [max(0.0, min(1.0, value / ceiling)) for value in motion]


def stream_motion(video: Path, duration: float, cancel_event: Optional[threading.Event] = None) -> list[float]:
    """Whole-stream 0..1 on-screen change per second, cheaply, for the signal-driven moment path.

    A small analysis proxy (<= 400 px tall) is decoded at 1 fps. A full-resolution file is read
    from its keyframes only (every ~2-6 s on YouTube/OBS recordings), which takes seconds per hour
    instead of minutes; the change between keyframes is spread over the seconds between them.
    """
    import numpy as np

    n = max(1, int(math.ceil(duration)))
    times: list[float] = []
    frames: list = []
    _, height = video_dimensions(video)
    if height <= 400:
        proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video), "-an",
             "-vf", "fps=1,scale=64:36:flags=area,format=gray", "-f", "rawvideo", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
        size = 64 * 36
        second = 0
        try:
            while True:
                chunk = proc.stdout.read(size)
                if len(chunk) < size:
                    break
                if second % 300 == 0:
                    _check_cancel(cancel_event)
                frames.append(np.frombuffer(chunk, dtype=np.uint8).reshape(36, 64).astype(np.float32))
                times.append(float(second))
                second += 1
        finally:
            proc.stdout.close()
            proc.wait()
    else:
        import av

        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "NONKEY"
            stream.thread_type = "AUTO"
            for i, frame in enumerate(container.decode(stream)):
                if i % 200 == 0:
                    _check_cancel(cancel_event)
                if frame.time is None:
                    continue
                times.append(float(frame.time))
                frames.append(frame.to_ndarray(format="gray", width=64, height=36).astype(np.float32))
    motion = np.zeros(n, dtype=np.float64)
    for k in range(1, len(frames)):
        t0, t1 = times[k - 1], times[k]
        if t1 <= t0:
            continue
        change = float(np.mean(np.abs(frames[k] - frames[k - 1]))) / 255.0
        a_sec, b_sec = int(t0), max(int(t0) + 1, int(math.ceil(t1)))
        motion[a_sec:min(n, b_sec)] = np.maximum(motion[a_sec:min(n, b_sec)], change)
    positive = motion[motion > 0]
    ceiling = max(float(np.percentile(positive, 95)) if positive.size else 0.01, 0.008)
    return np.clip(motion / ceiling, 0.0, 1.0).round(4).tolist()


def _motion_score_for_range(motion: list[float], start: float, end: float) -> float:
    a = max(0, int(start))
    b = min(len(motion), max(a + 1, int(math.ceil(end))))
    values = motion[a:b] or [0.0]
    ordered = sorted(values)
    p90 = ordered[min(len(ordered)-1, int(len(ordered) * 0.90))]
    return max(0.0, min(1.0, 0.65 * (sum(values) / len(values)) + 0.35 * p90))


def _make_contact_sheet(frame_dir: Path, candidate: Candidate, cv2_module) -> str | None:
    """Create a time-ordered 3x3 contact sheet showing setup, action and aftermath."""
    import numpy as np
    width, height, header = 320, 180, 42
    canvas = np.zeros((height * 3 + header, width * 3, 3), dtype=np.uint8)
    title = f"C{candidate.index} | {candidate.start:.0f}s-{candidate.end:.0f}s | chronological"
    cv2_module.putText(canvas, title[:90], (8, 27), cv2_module.FONT_HERSHEY_SIMPLEX,
                       0.58, (255, 255, 255), 1, cv2_module.LINE_AA)
    fractions = [0.04, 0.16, 0.28, 0.40, 0.52, 0.64, 0.76, 0.88, 0.98]
    for i, fraction in enumerate(fractions):
        row, col = divmod(i, 3)
        x, y = col * width, header + row * height
        timestamp = candidate.start + (candidate.end - candidate.start) * fraction
        frame_path = frame_dir / f"frame_{int(round(timestamp)):06d}.jpg"
        frame = cv2_module.imread(str(frame_path), cv2_module.IMREAD_COLOR)
        if frame is None:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            frame = cv2_module.resize(frame, (width, height), interpolation=cv2_module.INTER_AREA)
        cv2_module.putText(frame, f"{timestamp:.1f}s", (8, 22), cv2_module.FONT_HERSHEY_SIMPLEX,
                           0.55, (0, 255, 255), 2, cv2_module.LINE_AA)
        canvas[y:y+height, x:x+width] = frame
    ok, encoded = cv2_module.imencode(".jpg", canvas, [cv2_module.IMWRITE_JPEG_QUALITY, 76])
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


_extract_json_object = extract_json_object


def transcript_for_range(rows: list[dict], start: float, end: float) -> str:
    return " ".join(r["text"] for r in rows if r["end"] > start and r["start"] < end).strip()


def text_score(text: str) -> float:
    lower = text.lower()
    score = 0.0
    for word in HYPE_WORDS:
        if word in lower:
            score += 0.16 if " " in word else 0.08
    score += min(0.25, text.count("!") * 0.06)
    score += min(0.12, text.count("?") * 0.03)
    return min(1.0, score)


def build_candidates(duration: float, energies: list[float], transcript: list[dict],
                     motion: list[float], clip_len: int, stride: int,
                     transcript_usable: bool, loud: Optional[list[float]] = None,
                     chat: Optional[list[float]] = None) -> list[Candidate]:
    candidates: list[Candidate] = []
    if duration <= clip_len:
        starts = [0]
    else:
        starts = list(range(0, max(1, int(duration - clip_len) + 1), stride))
        last_start = max(0, int(duration - clip_len))
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
    for idx, start in enumerate(starts):
        end = min(duration, start + clip_len)
        a = max(0, int(start))
        b = min(len(energies), int(math.ceil(end)))
        local = energies[a:b] or [0.0]
        audio = 0.55 * (sum(local) / len(local)) + 0.45 * max(local)
        text = transcript_for_range(transcript, start, end) if transcript_usable else ""
        context_before = transcript_for_range(transcript, max(0, start - 12), start) if transcript_usable else ""
        context_after = transcript_for_range(transcript, end, min(duration, end + 12)) if transcript_usable else ""
        ts = text_score(text) if transcript_usable else 0.0
        visual = _motion_score_for_range(motion, start, end)
        loud_peak = range_peak(loud or [], start, end)
        chat_peak = range_peak(chat or [], start, end)
        signal = max(loud_peak, chat_peak)
        if transcript_usable:
            baseline = 0.20 * audio + 0.20 * visual + 0.40 * ts + 0.20 * signal
        else:
            baseline = 0.25 * audio + 0.55 * visual + 0.20 * signal
        # Strongly demote near-static sections such as countdowns and menus.
        if visual < 0.06:
            baseline *= 0.35
        candidates.append(Candidate(
            index=idx, start=float(start), end=float(end),
            audio_score=audio, transcript=text[:900],
            transcript_score=ts, visual_score=visual,
            final_score=baseline, context_before=context_before[:240], context_after=context_after[:240],
            loud_score=loud_peak, chat_score=chat_peak,
        ))
    return candidates


def candidates_from_moments(moments: list[Moment], duration: float, energies: list[float],
                            transcript: list[dict], motion: list[float], loud: list[float],
                            chat: list[float], min_len: int, max_len: int) -> list[Candidate]:
    """Turn LLM-picked moments into candidates with sentence-snapped, variable-length boundaries."""
    candidates: list[Candidate] = []
    for idx, moment in enumerate(sorted(moments, key=lambda m: m.start)):
        start, end = snap_to_speech(moment.start, moment.end, transcript, duration, min_len, max_len)
        a = max(0, int(start))
        b = min(len(energies), int(math.ceil(end)))
        local = energies[a:b] or [0.0]
        audio = 0.55 * (sum(local) / len(local)) + 0.45 * max(local)
        visual = _motion_score_for_range(motion, start, end)
        loud_peak = range_peak(loud, start, end)
        chat_peak = range_peak(chat, start, end)
        text = transcript_for_range(transcript, start, end)
        candidate = Candidate(
            index=idx, start=start, end=end, audio_score=audio, transcript=text[:900],
            transcript_score=text_score(text), visual_score=visual,
            context_before=transcript_for_range(transcript, max(0, start - 12), start)[:240],
            context_after=transcript_for_range(transcript, end, min(duration, end + 12))[:240],
            origin="moment", category=moment.category, hook_title=moment.hook_title,
            moment_score=moment.score, loud_score=loud_peak, chat_score=chat_peak,
            reason=moment.why, social_title=moment.hook_title, social_caption=moment.post_caption,
            payoff=moment.payoff_quote, payoff_quote=moment.payoff_quote,
            hashtags=list(moment.hashtags), event_type=moment.category,
        )
        _score_without_vision(candidate)
        candidates.append(candidate)
    return candidates


def candidates_from_signals(windows: list[SignalWindow], duration: float, energies: list[float],
                            transcript: list[dict], transcript_usable: bool, motion: list[float],
                            loud: list[float], chat: list[float], first_index: int) -> list[Candidate]:
    """Candidates found from loudness/chat/motion spikes alone - silent clutches and fails."""
    candidates: list[Candidate] = []
    for offset, window in enumerate(windows):
        start, end = window.start, window.end
        if transcript_usable:
            # If people are talking at the edges, still start/end on a sentence.
            start, end = snap_to_speech(start, end, transcript, duration, min_len=max(12.0, (end - start) * 0.7),
                                        max_len=(end - start) + 12)
        a, b = max(0, int(start)), min(len(energies), int(math.ceil(end)))
        local = energies[a:b] or [0.0]
        text = transcript_for_range(transcript, start, end) if transcript_usable else ""
        candidate = Candidate(
            index=first_index + offset, start=start, end=end,
            audio_score=0.55 * (sum(local) / len(local)) + 0.45 * max(local),
            transcript=text[:900], transcript_score=text_score(text) if text else 0.0,
            visual_score=_motion_score_for_range(motion, start, end),
            context_before=transcript_for_range(transcript, max(0, start - 12), start)[:240] if transcript_usable else "",
            context_after=transcript_for_range(transcript, end, min(duration, end + 12))[:240] if transcript_usable else "",
            origin="signal", category="signal_spike", reason=window.describe(),
            loud_score=range_peak(loud, start, end), chat_score=range_peak(chat, start, end),
            signal_strength=window.strength,
        )
        _score_without_vision(candidate)
        candidates.append(candidate)
    return candidates


def merge_candidates(moment_candidates: list[Candidate], signal_candidates: list[Candidate],
                     max_overlap: float = 0.35) -> list[Candidate]:
    """Transcript moments first; a signal window that mostly overlaps one is a duplicate and is dropped."""
    merged = list(moment_candidates)
    for candidate in sorted(signal_candidates, key=lambda c: c.signal_strength, reverse=True):
        duplicate = False
        for other in merged:
            inter = max(0.0, min(candidate.end, other.end) - max(candidate.start, other.start))
            if inter >= max_overlap * min(candidate.end - candidate.start, other.end - other.start):
                duplicate = True
                if other.origin == "moment":
                    # The spike confirms the transcript moment.
                    other.signal_strength = max(other.signal_strength, candidate.signal_strength)
                break
        if not duplicate:
            merged.append(candidate)
    return sorted(merged, key=lambda c: c.start)


def _score_without_vision(c: Candidate) -> None:
    """Scores used when the vision review is disabled or unavailable."""
    signal = max(c.loud_score, c.chat_score)
    if c.origin == "moment":
        c.final_score = 0.75 * (c.moment_score / 100) + 0.15 * signal + 0.10 * c.visual_score
        c.quality = c.moment_score
    elif c.origin == "signal":
        # Never judged by a model: rank it, but leave quality unknown so the min-score filter decides later.
        c.final_score = 0.55 * c.signal_strength + 0.30 * c.visual_score + 0.15 * c.audio_score
        c.quality = -1.0
    else:
        c.final_score = 0.30 * c.audio_score + 0.55 * c.visual_score + 0.15 * signal
        if c.visual_score < 0.06:
            c.final_score *= 0.25
        c.quality = -1.0


def _rank_batch(candidates: list[Candidate], model: str, game_profile: str,
                image_sheets: list[str], transcript_usable: bool, creator: str = "") -> list[Candidate]:
    rows = []
    for c in candidates:
        row = {
            "index": c.index,
            "time_range": f"{c.start:.1f}-{c.end:.1f}s",
            "audio_activity": round(c.audio_score, 3),
            "visual_motion": round(c.visual_score, 3),
            "voice_spike": round(c.loud_score, 2),
            "chat_spike": round(c.chat_score, 2),
            "spoken_context_before": c.context_before[:220] if transcript_usable else "",
            "spoken_context_during": c.transcript[:500] if transcript_usable else "",
            "spoken_context_after": c.context_after[:220] if transcript_usable else "",
        }
        if c.origin == "moment":
            row["proposed_hook_title"] = c.hook_title
            row["transcript_editor_note"] = c.reason[:160]
        elif c.origin == "signal":
            row["found_by"] = (f"{c.reason} (no transcript pick). There may be little or no speech: judge what "
                               "visibly happens in the frames - a clutch, a death, a fail, a surprise - or score it low.")
        rows.append(row)
    who = (f"The streamer is {creator}; name them in titles. " if creator
           else "The streamer's name is unknown; do not invent one. ")
    prompt = (
        f"You are a harsh short-form gaming editor reviewing {game_profile} footage for YouTube Shorts / TikTok. {who}"
        "Each attached image is a 3x3 contact sheet for one candidate; its nine frames are in chronological order. "
        "In story games much of the speech is the game's own characters (cutscene dialogue), not the streamer. Only credit words or reactions to the streamer when it is clearly them (commentary, reactions, talking to teammates); for a cutscene moment, title the scene itself, e.g. 'Bank heist goes WRONG in the first minute'. "
        "Use the visual sequence AND the speech context together. Do not infer a kill, win, clutch or funny event "
        "unless visible frames or transcript support it. Judge whether the clip is a COMPLETE, standalone mini-story: "
        "a viewer who has not watched the VOD must understand the hook/setup, action/escalation, payoff, and reaction. "
        "Score harshly: 0.9+ only for something that would go viral on its own, 0.7-0.89 clearly shareable, "
        "0.5-0.69 decent, below 0.5 not worth posting. Most candidates should score below 0.7. "
        "A clip that starts after the action, starts in a loading/buy/menu state, needs earlier context, "
        "or ends before the payoff/reaction must receive story_score <= 0.30. Do not promote an unfinished fragment "
        "merely because it is loud or high-motion. "
        "Reject stream starting soon screens, countdowns, menus, lobbies, character selection, loading screens, settings, "
        "long static scenes and ordinary scoreboards. Prefer first reactions, crazy discoveries, freakouts, funny fails, "
        "hot takes, close calls and impressive plays. Do not invent events. "
        "Return ONLY JSON matching the provided schema, with a result for every index. Scores are 0..1. "
        "screen_type must be active_gameplay, menu_lobby, starting_screen, loading_screen, scoreboard, cutscene, or other. "
        "Rejected screen types must have score and story_score <= 0.05. Use factual reasons. "
        "hook_title is the text burned on the video: max 60 characters, third person, curiosity-driven, CAPITALISE 1-3 key "
        "words, may end with one emoji; keep proposed_hook_title if it is accurate. " + LITERAL_TITLE_RULES + " "
        "social_caption must be a ready-to-post "
        "1-2 sentence description with a natural call to react; hashtags 3-5 items without #. "
        "Name a specific game in titles, captions or hashtags only if the game profile names it; otherwise use "
        "generic tags such as gaming, fps, clutch, funny. Candidate details: "
        + json.dumps(rows, ensure_ascii=False)
    )
    schema = {
        "type": "object",
        "properties": {
            "ranked": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"}, "score": {"type": "number"},
                        "story_score": {"type": "number"}, "screen_type": {"type": "string"},
                        "event_type": {"type": "string"}, "hook": {"type": "string"},
                        "setup": {"type": "string"}, "payoff": {"type": "string"},
                        "reason": {"type": "string"}, "hook_title": {"type": "string"},
                        "social_title": {"type": "string"}, "social_caption": {"type": "string"},
                        "hashtags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["index", "score", "story_score", "screen_type", "reason"],
                },
            }
        },
        "required": ["ranked"],
    }
    data = ollama_json(OLLAMA_URL, model, prompt, schema, images=image_sheets,
                       temperature=0.05, num_ctx=12288, num_predict=1800, timeout=1800)
    by_index = {c.index: c for c in candidates}
    result = []
    for item in data.get("ranked", []):
        try:
            candidate = by_index.get(int(item.get("index", -1)))
            if candidate is None:
                continue
            score = max(0.0, min(1.0, float(item.get("score", 0))))
            story_score = max(0.0, min(1.0, float(item.get("story_score", score))))
            screen_type = str(item.get("screen_type", "unknown")).lower().replace(" ", "_")
            rejected = any(term in screen_type for term in (
                "menu", "lobby", "starting", "countdown", "loading", "waiting", "scoreboard"
            ))
            if rejected:
                score = min(score, 0.05)
                story_score = min(story_score, 0.05)
            candidate.screen_type = screen_type
            if candidate.origin != "moment" or not candidate.event_type or candidate.event_type == "unknown":
                candidate.event_type = str(item.get("event_type", "unknown"))[:80]
            candidate.hook = str(item.get("hook", ""))[:160]
            candidate.setup = str(item.get("setup", ""))[:180]
            candidate.payoff = str(item.get("payoff", ""))[:180]
            vision_reason = str(item.get("reason", "Visual AI reviewed the clip"))[:180]
            candidate.reason = (f"{candidate.reason} | {vision_reason}"[:300]
                                if candidate.origin == "moment" and candidate.reason else vision_reason)
            vision_title = tidy_hook_title(item.get("hook_title") or item.get("social_title") or "")
            if vision_title and not candidate.hook_title:
                candidate.hook_title = vision_title
            candidate.social_title = candidate.hook_title or vision_title
            caption = str(item.get("social_caption", "")).strip()[:260]
            if caption:
                candidate.social_caption = caption
            raw_tags = item.get("hashtags", [])
            if isinstance(raw_tags, list) and raw_tags:
                candidate.hashtags = [str(t).lstrip("#")[:40] for t in raw_tags[:5]]
            candidate.story_score = story_score
            signal = max(candidate.loud_score, candidate.chat_score)
            vision_quality = 100 * (0.55 * score + 0.45 * story_score)
            if candidate.origin == "moment":
                candidate.final_score = (0.50 * candidate.moment_score / 100 + 0.20 * score + 0.15 * story_score
                                         + 0.10 * signal + 0.05 * candidate.visual_score)
                candidate.quality = 0.65 * candidate.moment_score + 0.35 * vision_quality
            elif candidate.origin == "signal":
                candidate.final_score = (0.40 * score + 0.35 * story_score + 0.15 * candidate.signal_strength
                                         + 0.10 * candidate.visual_score)
                candidate.quality = vision_quality
            elif transcript_usable:
                candidate.final_score = (0.32 * score + 0.28 * story_score + 0.14 * candidate.visual_score
                                         + 0.06 * candidate.audio_score + 0.10 * candidate.transcript_score
                                         + 0.10 * signal)
                candidate.quality = vision_quality
            else:
                candidate.final_score = (0.40 * score + 0.35 * story_score + 0.12 * candidate.visual_score
                                         + 0.05 * candidate.audio_score + 0.08 * signal)
                candidate.quality = vision_quality
            if rejected:
                candidate.final_score = min(candidate.final_score, 0.02)
                candidate.quality = 0.0
            result.append(candidate)
        except (TypeError, ValueError):
            continue
    if not result:
        raise ValueError("Ollama returned no usable candidate rankings.")
    return result


def ollama_rank(candidates: list[Candidate], model: str,
                frame_dir: Path,
                progress: Optional[ProgressCallback] = None,
                cancel_event: Optional[threading.Event] = None,
                log: Optional[LogCallback] = None,
                game_profile: str = "Auto / mixed games",
                transcript_usable: bool = False, creator: str = "") -> list[Candidate]:
    # Broad visual shortlist; small batches reduce local context pressure and malformed responses.
    shortlist = sorted(candidates, key=lambda c: c.final_score, reverse=True)[:64]
    batch_size = 4
    batches = [shortlist[i:i + batch_size] for i in range(0, len(shortlist), batch_size)]
    ranked_by_index: dict[int, Candidate] = {}
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for local visual AI ranking. Run setup again.") from exc
    for batch_no, batch in enumerate(batches, 1):
        _check_cancel(cancel_event)
        _log(log, f"Multimodal story review batch {batch_no}/{len(batches)} ({len(batch)} candidates)...")
        image_sheets = [_make_contact_sheet(frame_dir, c, cv2) for c in batch]
        if any(image is None for image in image_sheets):
            raise RuntimeError("Could not create one or more gameplay contact sheets.")
        try:
            ranked = _rank_batch(batch, model, game_profile, image_sheets, transcript_usable, creator)
        except Exception as first_error:
            _log(log, f"Vision batch retry with smaller candidate groups: {first_error}")
            ranked = []
            for sub_start in range(0, len(batch), 2):
                sub = batch[sub_start:sub_start + 2]
                sub_images = [_make_contact_sheet(frame_dir, c, cv2) for c in sub]
                ranked.extend(_rank_batch(sub, model, game_profile, sub_images, transcript_usable, creator))
        ranked_by_index.update({c.index: c for c in ranked})
        pct = batch_no / max(1, len(batches)) * 100
        _emit(progress, "Visual story review", pct, 84 + pct * 0.04,
              f"Reviewed {min(batch_no * batch_size, len(shortlist))}/{len(shortlist)} gameplay candidates")
    return [ranked_by_index.get(c.index, c) for c in candidates]


def choose_top(candidates: list[Candidate], count: int,
               min_score: Optional[float] = None) -> list[Candidate]:
    """Pick distinct, standalone stories rather than adjacent fragments of one event.

    With ``min_score`` set, candidates the AI judged below that publish-readiness
    (0-100) are dropped: fewer good clips beat more filler.
    """
    ranked = sorted(candidates, key=lambda c: c.final_score, reverse=True)
    chosen: list[Candidate] = []
    for candidate in ranked:
        if candidate.rejected:
            continue
        if candidate.screen_type in {"menu_lobby", "starting_screen", "loading_screen", "scoreboard"}:
            continue
        if min_score is not None and 0 <= candidate.quality < min_score:
            continue
        # Windows may not overlap yet still be two cuts from the same round/event.
        # Keep the best complete arc and make the next pick meaningfully independent.
        story_gap = max(10.0, (candidate.end - candidate.start) * 0.40)
        independent = all(
            candidate.end + story_gap <= old.start or candidate.start >= old.end + story_gap
            for old in chosen
        )
        if independent:
            chosen.append(candidate)
        if len(chosen) >= count:
            break
    return sorted(chosen, key=lambda c: c.start)


def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_clip_srt(rows: list[dict], start: float, end: float, path: Path) -> bool:
    entries = []
    for seg in rows:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        a = max(0.0, seg["start"] - start)
        b = min(end - start, seg["end"] - start)
        text = re.sub(r"\s+", " ", str(seg.get("text", ""))).strip()
        if any(ch.isalnum() for ch in text):
            entries.append((a, b, text))
    if not entries:
        return False
    with path.open("w", encoding="utf-8") as f:
        for i, (a, b, text) in enumerate(entries, 1):
            f.write(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n\n")
    return True


def _ffmpeg_filter_path(path: Path) -> str:
    value = path.resolve().as_posix()
    return value.replace(":", r"\:").replace("'", r"\'").replace(",", r"\,")


def ffmpeg_has_ass_filter() -> bool:
    try:
        result = _run_process(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True, check=False)
        return " ass " in f" {result.stdout} " or " subtitles " in f" {result.stdout} "
    except Exception:
        return False


def build_render_graph(plan: clip_style.LayoutPlan, duration: float, *, title: bool,
                       ass_path: Optional[Path]) -> str:
    """Compose layout -> hook title overlay -> captions -> soft fades into one filter graph."""
    parts = [plan.filter]
    current = "base"
    if title:
        parts.append("[1:v]format=rgba,fade=t=in:st=0.10:d=0.30:alpha=1[title]")
        parts.append(f"[{current}][title]overlay=x=(W-w)/2:y={plan.title_center_y}-h/2:eof_action=pass[titled]")
        current = "titled"
    if ass_path is not None:
        parts.append(f"[{current}]ass=filename='{_ffmpeg_filter_path(ass_path)}'[captioned]")
        current = "captioned"
    # A short visual ease prevents a jarring hard cut even when the source event
    # begins or ends at an energetic moment.
    parts.append(f"[{current}]fade=t=in:st=0:d=0.18,fade=t=out:st={max(0.0, duration - 0.22):.3f}:d=0.22,"
                 "format=yuv420p[v]")
    return ";".join(parts)


def render_vertical(video: Path, candidate: Candidate, output: Path, use_nvenc: bool,
                    log: Optional[LogCallback] = None,
                    layout: str = LAYOUT_AUTO,
                    ass_path: Optional[Path] = None,
                    normalize_audio: bool = True,
                    plan: Optional[clip_style.LayoutPlan] = None,
                    title_png: Optional[Path] = None, source_offset: float = 0.0) -> None:
    """Render one 1080x1920 clip. ``source_offset`` is the stream time of ``video``'s t=0 (sections)."""
    duration = max(0.1, candidate.end - candidate.start)
    if plan is None:
        src_w, src_h = video_dimensions(video)
        plan = clip_style.plan_layout(layout, src_w, src_h, None, split_screen=candidate.split_screen,
                                      band=tuple(candidate.content_band))
    has_title = bool(title_png and title_png.exists())
    graph = build_render_graph(plan, duration, title=has_title,
                               ass_path=ass_path if ass_path and ass_path.exists() else None)
    inputs = ["-ss", f"{max(0.0, candidate.start - source_offset):.3f}", "-i", str(video)]
    if has_title:
        inputs += ["-loop", "1", "-framerate", "30", "-t", f"{duration:.3f}", "-i", str(title_png)]
    common = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs,
        "-t", f"{duration:.3f}", "-filter_complex", graph,
        "-map", "[v]", "-map", "0:a?", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
    ]
    fades = f"afade=t=in:st=0:d=0.12,afade=t=out:st={max(0, duration - 0.18):.3f}:d=0.18"
    common.extend(["-af", f"loudnorm=I=-14:TP=-1.5:LRA=11,{fades}" if normalize_audio else fades])
    if use_nvenc:
        cmd = common + ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "20", "-b:v", "0", str(output)]
        try:
            _run_process(cmd, check=True, capture_output=True, text=True)
            return
        except subprocess.CalledProcessError as exc:
            if exc.returncode in (3221225786, -1073741510):
                raise _subprocess_failure(exc, "Video rendering") from exc
            _log(log, "NVENC failed for this clip; retrying with CPU encoding.")
            if exc.stderr:
                _log(log, exc.stderr[-1200:])
    cmd = common + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(output)]
    try:
        _run_process(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise _subprocess_failure(exc, "Video rendering") from exc


def create_thumbnail(video: Path, candidate: Candidate, output: Path,
                     rendered: Optional[Path] = None, source_offset: float = 0.0) -> bool:
    """Grab a preview frame; prefer the finished vertical clip so the preview matches what is posted."""
    if rendered is not None and rendered.is_file():
        at = max(0.0, (candidate.end - candidate.start) * 0.45)
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{at:.2f}", "-i", str(rendered),
               "-frames:v", "1", "-vf", "scale=540:-2", "-q:v", "3", str(output)]
    else:
        midpoint = candidate.start + (candidate.end - candidate.start) * 0.55 - source_offset
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, midpoint):.2f}",
               "-i", str(video), "-frames:v", "1", "-q:v", "2", str(output)]
    result = _run_process(cmd, check=False, capture_output=True, text=True)
    return result.returncode == 0 and output.exists()


def write_post_file(path: Path, candidate: Candidate) -> None:
    tags = " ".join("#" + str(t).lstrip("#") for t in (candidate.hashtags or []))
    lines = [candidate.hook_title or candidate.social_title or "Gaming highlight", ""]
    if candidate.social_caption:
        lines += [candidate.social_caption, ""]
    if tags:
        lines.append(tags)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def claim_job_dir(output_root: Path, job_name: str) -> Path:
    """Reserve an output folder for this job.

    A folder counts as taken as soon as it has any content - not only once a finished
    job wrote clips.json - so two jobs on the same video (e.g. the GUI and the CLI) can
    never write into, and overwrite, the same folder. Empty leftovers are reused.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    suffix = 1
    while True:
        job_dir = output_root / (job_name if suffix == 1 else f"{job_name}_{suffix}")
        try:
            job_dir.mkdir()
            return job_dir
        except FileExistsError:
            if job_dir.is_dir() and not any(job_dir.iterdir()):
                return job_dir
            suffix += 1


ASR_PIPELINE_VERSION = "asr-v3"   # bump when Whisper options change what a cached transcript would contain


def transcript_cache_path(video: Path, whisper_model: str, language_mode: str, device: str = "cpu",
                          compute_type: str = "", duration: float = 0.0) -> Path:
    """Cache key: the source plus every Whisper setting that changes the result.

    Model, device and compute type are all part of the key, so an old CPU ``small`` transcript is
    never reused for a GPU ``large-v3`` run. Files that carry a YouTube ID ("Title [ID].mp4", the
    audio-only download, the analysis proxy) share one key, so the audio-first URL flow and a
    later full download reuse the same transcript.
    """
    match = YOUTUBE_ID_IN_NAME.search(video.name)
    if match:
        source_key = f"youtube:{match.group(1)}|{int(round(duration))}"
    else:
        stat = video.stat()
        source_key = f"{video.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
    key = f"{source_key}|{whisper_model}|{device}|{compute_type}|{language_mode}|{ASR_PIPELINE_VERSION}"
    return CACHE_DIR / "transcripts" / (hashlib.sha1(key.encode("utf-8")).hexdigest()[:24] + ".json")


def _transcription_stage(media: Path, wav_path: Path, temp_dir: Path, settings: "asr.WhisperSettings",
                         duration: float, language_mode: str, progress, cancel_event,
                         log) -> tuple[list[dict], str, str, "asr.WhisperSettings"]:
    """Run Whisper with the stereo-channel and Medium fallbacks and return the best RAW rows.

    Raw rows are returned (not cleaned) so they can be cached and re-cleaned by future versions.
    ``media`` is the video or downloaded audio file (used to read individual stereo channels).
    """
    used_settings = [settings]

    def run(path: Path, run_settings: "asr.WhisperSettings"):
        raw, lang, used = transcribe(path, run_settings, duration, progress, cancel_event, log, language_mode)
        used_settings[0] = used
        cleaned, _ = clean_transcript(raw, language_mode)
        return raw, lang, transcript_quality(cleaned)

    raw, language, quality = run(wav_path, settings)
    settings = used_settings[0]   # a GPU failure may have switched to the CPU
    source = wav_path
    # Check individual stereo channels if a mixed-down track produced punctuation/no speech.
    if not quality["usable"] and audio_channel_count(media) >= 2:
        _log(log, "Transcript quality is poor. Checking left/right channels for isolated voice chat...")
        options = [(raw, language, quality, wav_path)]
        for channel in ("FL", "FR"):
            _check_cancel(cancel_event)
            channel_path = temp_dir / f"audio_{channel.lower()}.wav"
            try:
                extract_channel_wav(media, channel_path, channel)
                channel_raw, channel_language, channel_quality = run(channel_path, settings)
                options.append((channel_raw, channel_language, channel_quality, channel_path))
                _log(log, f"{channel} channel: {channel_quality['word_count']} word tokens.")
                if channel_quality["usable"]:
                    break
            except JobCancelled:
                raise
            except Exception as exc:
                _log(log, f"Could not transcribe {channel} channel: {exc}")
        raw, language, quality, source = max(
            options, key=lambda item: (item[2]["usable"], item[2]["word_count"], item[2]["alphanumeric_characters"]))
    # A punctuation-only result is never considered a transcript. Small models get one retry with
    # Whisper medium; large models are already the strongest option.
    if not quality["usable"] and settings.model in ("tiny", "base", "small"):
        _log(log, (
            f"Transcript validation failed ({quality['word_count']} words; "
            f"{quality['alphanumeric_characters']} alphanumeric characters). Retrying Whisper medium."
        ))
        retry = asr.WhisperSettings("medium", settings.device, settings.compute_type, settings.model_auto)
        retry_raw, retry_language, retry_quality = run(source, retry)
        if (retry_quality["usable"], retry_quality["word_count"]) > (quality["usable"], quality["word_count"]):
            raw, language, quality = retry_raw, retry_language, retry_quality
            settings = used_settings[0]
        if quality["usable"]:
            _log(log, f"Medium retry recovered {quality['word_count']} word tokens.")
    if not quality["usable"]:
        _log(log, "WARNING: Speech remains unusable. The voice-chat mix may be absent, very quiet or overlapped; "
                  "captions will be skipped.")
    return raw, language, source.name, settings


# Screen types for the first/last seconds of a clip. A clip must not open on any of
# BAD_START_SCREENS; it may end on a death or round banner (that is often the payoff),
# but not on a menu or loading screen.
EDGE_LABELS = ["gameplay", "menu_or_shop", "loading_or_lobby", "scoreboard_or_banner", "death_or_spectate",
               "cutscene", "stream_screen", "black_or_fade", "other"]
BAD_START_SCREENS = {"menu_or_shop", "loading_or_lobby", "scoreboard_or_banner", "death_or_spectate",
                     "stream_screen", "black_or_fade"}
BAD_END_SCREENS = {"menu_or_shop", "loading_or_lobby", "stream_screen", "black_or_fade"}


def edge_times(start: float, end: float) -> list[float]:
    length = max(1.0, end - start)
    lead = [0.4, min(2.0, length * 0.15), min(4.0, length * 0.3)]
    return [start + x for x in lead] + [end - lead[2], end - lead[1], end - lead[0]]


def _edge_sheet(video: Path, times: list[float], work_dir: Path, tag: str) -> Optional[str]:
    import cv2
    import numpy as np

    tiles = []
    for i, t in enumerate(times):
        out = work_dir / f"{tag}_{i}.jpg"
        _run_process(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, t):.2f}",
                      "-i", str(video), "-frames:v", "1", "-vf", "scale=480:270", "-q:v", "4", str(out)],
                     check=False, capture_output=True, text=True)
        tile = cv2.imread(str(out), cv2.IMREAD_COLOR)
        if tile is None or tile.shape[:2] != (270, 480):
            tile = np.zeros((270, 480, 3), dtype=np.uint8)
        label = f"{i + 1} {'START' if i < 3 else 'END'}"
        cv2.rectangle(tile, (0, 0), (150, 40), (0, 0, 0), -1)
        cv2.putText(tile, label, (8, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    sheet = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    ok, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(encoded.tobytes()).decode("ascii") if ok else None


def edge_review(candidates: list[Candidate], video: Path, model: str, work_dir: Path,
                log: Optional[LogCallback] = None,
                cancel_event: Optional[threading.Event] = None) -> dict[int, list[str]]:
    """Label the first and last seconds of each finalist (gameplay vs menu/loading/death screens)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    schema = {
        "type": "object",
        "properties": {"clips": {"type": "array", "items": {
            "type": "object",
            "properties": {"index": {"type": "integer"},
                           "frames": {"type": "array", "items": {"type": "string", "enum": EDGE_LABELS}}},
            "required": ["index", "frames"]}}},
        "required": ["clips"],
    }
    labels: dict[int, list[str]] = {}
    for offset in range(0, len(candidates), 3):
        _check_cancel(cancel_event)
        batch = candidates[offset:offset + 3]
        sheets = [_edge_sheet(video, edge_times(c.start, c.end), work_dir, f"edge{c.index}") for c in batch]
        if any(sheet is None for sheet in sheets):
            continue
        prompt = (
            "Each attached image belongs to one gaming clip and shows 6 numbered frames: frames 1-3 are the clip's "
            "first seconds, frames 4-6 its last seconds. Label every frame with exactly one screen type:\n"
            "- gameplay: the player is controlling a character in the game world (HUD and crosshair visible).\n"
            "- menu_or_shop: a menu, weapon/agent shop, inventory or settings screen covers the view.\n"
            "- loading_or_lobby: loading screen, match/map intro or 'versus' screen, lobby, queue, character select.\n"
            "- scoreboard_or_banner: scoreboard, round won/lost banner, match summary.\n"
            "- death_or_spectate: the player is dead: kill-cam, death recap, spectating a teammate, ragdoll or "
            "upside-down/death camera.\n"
            "- cutscene: a non-interactive story scene (often letterboxed, no HUD).\n"
            "- stream_screen: the stream's own screen, not the game: starting soon, be right back, just chatting, "
            "ending or offline card, a big streamer logo/avatar.\n"
            "- black_or_fade: an (almost) black or faded-out frame.\n- other: anything else.\n"
            "Images in order have these indexes: " + json.dumps([c.index for c in batch]) +
            ". Return JSON with 6 labels per clip, in frame order."
        )
        try:
            data = ollama_json(OLLAMA_URL, model, prompt, schema, images=sheets, temperature=0.0,
                               num_ctx=8192, num_predict=600, timeout=900)
        except Exception as exc:  # noqa: BLE001 - edge checks are an extra filter
            _log(log, f"Edge check failed for {len(batch)} clip(s): {exc}")
            continue
        valid = {c.index for c in batch}
        for item in data.get("clips", []) or []:
            try:
                index = int(item.get("index", -1))
            except (TypeError, ValueError):
                continue
            frames = [str(x) for x in (item.get("frames") or [])][:6]
            if index in valid and len(frames) == 6:
                labels[index] = frames
    return labels


def apply_edge_labels(c: Candidate, labels: list[str], transcript: list[dict], transcript_usable: bool,
                      duration: float, min_len: float, max_len: float) -> Optional[str]:
    """Trim menu/loading/death screens off the clip edges. Returns a reason string if the clip must be dropped."""
    times = edge_times(c.start, c.end)
    lead = 0
    while lead < 3 and labels[lead] in BAD_START_SCREENS:
        lead += 1
    trail = 0
    while trail < 3 and labels[5 - trail] in BAD_END_SCREENS:
        trail += 1
    if trail == 0 and all(label == "scoreboard_or_banner" for label in labels[3:]):
        # One banner can be the payoff; seconds of results table at the very end are dead air.
        trail = 3
    if lead == 0 and trail == 0:
        return None
    if lead == 3 and trail == 3:
        return f"starts on {labels[0]} and ends on {labels[5]}"
    not_before = None
    if lead:
        not_before = (times[lead - 1] + times[lead]) / 2 if lead < 3 else times[2] + 1.5
    not_after = None
    if trail:
        first_bad = 6 - trail
        not_after = (times[first_bad - 1] + times[first_bad]) / 2 if trail < 3 else times[3] - 1.0
    rough_start = max(c.start, not_before) if not_before is not None else c.start
    rough_end = min(c.end, not_after) if not_after is not None else c.end
    if transcript_usable:
        new_start, new_end = snap_to_speech(rough_start, rough_end, transcript, duration, min_len, max_len,
                                            not_before=not_before, not_after=not_after)
    else:
        new_start, new_end = rough_start, rough_end
    if new_end - new_start < max(8.0, 0.75 * min_len):
        return f"too short after trimming {labels[0] if lead else labels[5]} screens"
    c.start, c.end = round(new_start, 2), round(new_end, 2)
    return None


class StageTimer:
    """Wall-clock seconds per pipeline stage, reported in the log and clips.json."""

    def __init__(self):
        self.seconds: dict[str, float] = {}

    def add(self, name: str, seconds: float) -> None:
        self.seconds[name] = round(self.seconds.get(name, 0.0) + seconds, 1)

    def stage(self, name: str):
        timer = self

        class _Stage:
            def __enter__(self):
                self.t = time.monotonic()

            def __exit__(self, *exc):
                timer.add(name, time.monotonic() - self.t)
                return False

        return _Stage()


class _Background:
    """Run one function in a thread (e.g. the proxy download during transcription)."""

    def __init__(self, fn: Callable):
        self.result = None
        self.error: Optional[BaseException] = None
        self.seconds = 0.0
        self._thread = threading.Thread(target=self._run, args=(fn,), daemon=True)
        self._thread.start()

    def _run(self, fn: Callable) -> None:
        start = time.monotonic()
        try:
            self.result = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in wait()
            self.error = exc
        self.seconds = time.monotonic() - start

    def wait(self):
        self._thread.join()
        if self.error is not None:
            raise self.error
        return self.result


def stream_context(info: dict, limit: int = 320) -> str:
    """A short "Stream title / description" line for the LLM prompts (URL sources only)."""
    title = str(info.get("title") or "").strip()
    if not title:
        return ""
    lines = [ln.strip() for ln in str(info.get("description") or "").splitlines() if ln.strip()]
    # The first paragraph usually says what happens; links, socials and member plugs come later.
    about = " ".join(ln for ln in lines[:3] if "http" not in ln.lower() and "://" not in ln)
    text = f"Stream title: {title}." + (f" Description: {about}" if about else "")
    return text[:limit].rsplit(" ", 1)[0] if len(text) > limit else text


def _existing_download(input_dir: Path, video_id: Optional[str]) -> Optional[Path]:
    """A full video this job (or an earlier one) already downloaded into the Input folder."""
    if not video_id or not input_dir.is_dir():
        return None
    matches = [p for p in input_dir.iterdir()
               if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS and f"[{video_id}]" in p.name]
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def facecam_samples(video: Optional[Path], duration: float, sections: list, count: int = 24) -> list[tuple[Path, float]]:
    """(file, time) pairs to look for a webcam: across the whole video, or across the downloaded sections."""
    if sections:
        total = sum(max(1.0, s.length or (s.end - s.start)) for s in sections)
        samples: list[tuple[Path, float]] = []
        for section in sections:
            length = max(1.0, section.length or (section.end - section.start))
            n = max(2, round(count * length / total))
            samples.extend((section.path, length * (0.08 + 0.84 * i / max(1, n - 1))) for i in range(n))
        return samples
    if video is None:
        return []
    return [(video, duration * (0.04 + 0.92 * i / max(1, count - 1))) for i in range(count)]


def finalist_sheets(clips: list[Candidate], frame_dir: Path, model: str) -> dict[int, str]:
    """3x3 contact sheets (current clip edges) for a model that can see; empty if it cannot."""
    if "vision" not in ollama_capabilities(model):
        return {}
    try:
        import cv2
    except ImportError:
        return {}
    sheets = {}
    for c in clips:
        sheet = _make_contact_sheet(frame_dir, c, cv2)
        if sheet:
            sheets[c.index] = sheet
    return sheets


def fact_check_titles(clips: list[Candidate], transcript: list[dict], *, model: str, creator: str,
                      log: Optional[LogCallback] = None, images: Optional[dict[int, str]] = None,
                      check_all: bool = False) -> None:
    """Final literal-truth gate for burned-in titles.

    Titles that still contain a quantifier, superlative or gendered pronoun the evidence does not
    contain (or that the critic rewrote) get a narrow claim-by-claim fact-check. A title that still
    carries pure hype (whole/every/most/best/ever, "-est") afterwards is removed, never burned in.
    """
    items, evidence_by_index = [], {}
    for c in clips:
        if not c.hook_title and c.critic_title:
            c.hook_title = c.critic_title   # e.g. a silent signal clip the vision model left untitled
        if not c.hook_title:
            continue
        evidence = " ".join(x for x in (transcript_for_range(transcript, c.start, c.end)[:700], c.critic_note,
                                        c.critic_frames, c.setup, c.payoff) if x)
        evidence_by_index[c.index] = evidence
        risks = title_risks(c.hook_title, evidence)
        if check_all or risks or getattr(c, "_title_rewritten", False):
            items.append({"index": c.index, "title": c.hook_title, "evidence": evidence[:900], "check_words": risks})
    results = verify_titles(items, url=OLLAMA_URL, model=model, creator=creator, log=lambda m: _log(log, m),
                            images=images)
    for c in clips:
        if not c.hook_title or c.index not in evidence_by_index:
            continue
        result = results.get(c.index)
        title = (result or {}).get("title") or c.hook_title
        hype = title_risks(title, evidence_by_index[c.index], strict=True)
        if hype and c.critic_title and not title_risks(c.critic_title, evidence_by_index[c.index], strict=True):
            _log(log, f"  title replaced by the critic's literal title: {c.hook_title!r} -> {c.critic_title!r}")
            c.hook_title = c.critic_title
        elif hype:
            _log(log, f"  title removed (unsupported {', '.join(hype)}): {c.hook_title!r}")
            c.hook_title = ""
        elif title != c.hook_title:
            _log(log, f"  title made literal: {c.hook_title!r} -> {title!r}")
            c.hook_title = title
        if result and result.get("claims"):
            c.title_claims = result["claims"]
        c.social_title = c.hook_title or c.social_title


def process_job(source: str, output_root: Path, input_dir: Path, *, count: Optional[int] = None,
                clip_length: Optional[int] = None, stride: Optional[int] = None, whisper_model: str = asr.AUTO,
                whisper_device: str = asr.AUTO, ollama_model: str = AUTO,
                use_ollama: bool = True, use_nvenc: bool = True,
                game_profile: str = "Auto / mixed games",
                cookies_browser: Optional[str] = None,
                language_mode: str = asr.MIXED,
                render_layout: str = LAYOUT_AUTO,
                burn_captions: bool = True,
                normalize_audio: bool = True,
                facecam_mode: str = FACECAM_AUTO,
                min_score: Optional[float] = DEFAULT_MIN_SCORE,
                creator_name: str = "",
                burn_title: bool = True,
                use_chat: bool = True,
                picker_model: Optional[str] = None,
                use_transcript_cache: bool = True,
                use_signals: bool = True,
                section_download: bool = True,
                progress: Optional[ProgressCallback] = None,
                log: Optional[LogCallback] = None,
                cancel_event: Optional[threading.Event] = None) -> Path:
    check_tools()
    if ((count is not None and count < 1) or (clip_length is not None and clip_length < 10)
            or (stride is not None and stride < 5)):
        raise ValueError("Clip count must be >=1, duration >=10 sec and scan interval >=5 sec.")
    _check_cancel(cancel_event)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    timer = StageTimer()
    job_started = time.monotonic()
    job_temp = tempfile.TemporaryDirectory(prefix="clipforge_")
    try:
        return _process(source, output_root, input_dir, timer, job_started, Path(job_temp.name), **{
            "count": count, "clip_length": clip_length, "stride": stride, "whisper_model": whisper_model,
            "whisper_device": whisper_device, "ollama_model": ollama_model, "use_ollama": use_ollama,
            "use_nvenc": use_nvenc, "game_profile": game_profile, "cookies_browser": cookies_browser,
            "language_mode": language_mode, "render_layout": render_layout, "burn_captions": burn_captions,
            "normalize_audio": normalize_audio, "facecam_mode": facecam_mode, "min_score": min_score,
            "creator_name": creator_name, "burn_title": burn_title, "use_chat": use_chat,
            "picker_model": picker_model, "use_transcript_cache": use_transcript_cache,
            "use_signals": use_signals, "section_download": section_download, "progress": progress,
            "log": log, "cancel_event": cancel_event})
    finally:
        try:
            job_temp.cleanup()
        except OSError:
            pass


def _process(source: str, output_root: Path, input_dir: Path, timer: StageTimer, job_started: float,
             temp: Path, *, count, clip_length, stride, whisper_model, whisper_device, ollama_model, use_ollama,
             use_nvenc, game_profile, cookies_browser, language_mode, render_layout, burn_captions,
             normalize_audio, facecam_mode, min_score, creator_name, burn_title, use_chat, picker_model,
             use_transcript_cache, use_signals, section_download, progress, log, cancel_event) -> Path:
    is_url = source.strip().lower().startswith(("https://", "http://"))
    url = source.strip() if is_url else ""
    source_info: dict = {}
    video: Optional[Path] = None       # full-quality video (for URLs: only once sections/full file exist)
    media: Path                        # what Whisper and loudness read (video, or the audio-only download)
    remote_dir: Optional[Path] = None
    proxy_job: Optional[_Background] = None
    source_mode = "local file"

    def dl_progress(label: str, base: float, span: float):
        return lambda pct, extra: _emit(progress, "Downloading", pct, base + pct * span / 100,
                                        f"{label}: {pct:.1f}%{extra}")

    if is_url:
        with timer.stage("download (audio)"):
            _emit(progress, "Downloading", 0, 1, "Reading the stream's details...")
            info = sources.probe(url, cookies_browser, lambda m: _log(log, m))
            source_info.update({k: info.get(k) for k in ("id", "title", "channel", "uploader", "was_live", "duration",
                                                         "description")})
            _log(log, f"Source: {info.get('title')} · {info.get('channel') or info.get('uploader') or '?'} · "
                      f"{float(info.get('duration') or 0) / 60:.0f} min")
            existing = _existing_download(input_dir, info.get("id"))
            if existing is not None:
                video = media = existing
                source_mode = "earlier full download"
                _log(log, f"Using the full video already in the Input folder: {existing.name}")
            elif section_download:
                vid = re.sub(r"[^A-Za-z0-9_-]", "_", str(info.get("id") or "stream"))
                remote_dir = CACHE_DIR / "remote" / vid
                _log(log, "Downloading the audio track only first; full-quality video is fetched later, only for "
                          "the chosen clips. Only download videos you own or are authorised to use.")
                media = sources.download_audio(url, remote_dir, cookies_browser=cookies_browser,
                                               log=lambda m: _log(log, m), cancel_event=cancel_event,
                                               on_progress=dl_progress("Downloading audio", 1, 5))
                _log(log, f"Audio: {media.name} ({media.stat().st_size / 1e6:.0f} MB)")
                source_mode = "audio first + clip sections"

                def proxy_task():
                    path = sources.download_proxy(url, remote_dir, cookies_browser=cookies_browser,
                                                  log=lambda m: _log(log, m), cancel_event=cancel_event)
                    _log(log, f"Analysis proxy ready: {path.name} ({path.stat().st_size / 1e6:.0f} MB)")
                    started = time.monotonic()
                    motion_all = stream_motion(path, duration_seconds(path), cancel_event) if use_signals else []
                    return path, motion_all, time.monotonic() - started

                # The low-res proxy downloads while Whisper runs.
                proxy_job = _Background(proxy_task)
            else:
                video = media = download_youtube(url, input_dir, progress, cancel_event, log, cookies_browser,
                                                 info_out=source_info)
                source_mode = "full download"
    else:
        video = media = Path(source).expanduser().resolve()
        if not video.is_file() or video.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError("Select a valid local video file (MP4, MKV, WebM, MOV or AVI).")
        _emit(progress, "Input", 100, 6, f"Using local file: {video.name}")
    _check_cancel(cancel_event)
    _log(log, f"Media: {media}")

    _emit(progress, "Inspecting video", 0, 7, "Reading media details...")
    duration = duration_seconds(media)
    if video is not None:
        src_w, src_h = video_dimensions(video)
    else:
        src_w, src_h = sources.best_video_size(info)
    _log(log, f"Duration: {duration/60:.1f} minutes · {src_w}x{src_h}")
    auto_mode = count is None or clip_length is None or stride is None
    if clip_length:
        max_len, min_len = int(clip_length), max(10, int(round(clip_length * 0.4)))
    else:
        max_len, min_len = 60, 20
    max_len = max(10, min(max_len, int(duration)))
    min_len = min(min_len, max_len)
    window_len = clip_length or (30 if duration >= 60 else max(10, int(duration * 0.6)))
    stride = stride or max(5, min(10, window_len // 5))
    count = count or max(3, min(8, int(round(duration / 1200))))
    creator = (creator_name or "").strip() or str(source_info.get("channel") or source_info.get("uploader") or "").strip()
    # The picker, vision review and critic see the stream's own title and first description lines
    # (who is playing with whom), so titles can name the right person.
    context = stream_context(source_info)
    prompt_profile = f"{game_profile}. {context}" if context else game_profile
    _log(log, f"Up to {count} clips · {min_len}-{max_len}s moments · min score "
              f"{'off' if min_score is None else int(min_score)}/100 · layout: {render_layout}"
              + (f" · streamer: {creator}" if creator else ""))
    stem = str(source_info.get("title") or "") + (f" [{source_info.get('id')}]" if source_info.get("id") else "")
    job_name = re.sub(r"[^A-Za-z0-9_-]+", "_", stem if is_url else video.stem)[:70].strip("_") or "gaming_video"
    job_dir = claim_job_dir(output_root, job_name)

    # --- audio: loudness and speech -------------------------------------------------------------
    wav_path = temp / "audio_16k_mono.wav"
    with timer.stage("audio analysis"):
        _log(log, "Extracting mono 16 kHz audio with FFmpeg...")
        _emit(progress, "Audio analysis", 0, 11, "Extracting audio...")
        extract_audio_wav(media, wav_path)
        _check_cancel(cancel_event)
        _emit(progress, "Audio analysis", 65, 13, "Measuring audio activity and loudness spikes...")
        raw_rms = audio_energy_by_second(wav_path, raw=True)
        energies = normalise_energy(raw_rms)
        loud = loudness_spikes(raw_rms)
        _log(log, f"Loudness spikes: {sum(1 for v in loud if v >= 0.5)} second(s) clearly above the local baseline.")
        _emit(progress, "Audio analysis", 100, 15, "Audio analysis complete")

    settings = asr.resolve_settings(whisper_model, whisper_device, log=lambda m: _log(log, m))
    with timer.stage("transcription"):
        cache_path = transcript_cache_path(media, settings.model, language_mode, settings.device,
                                           settings.compute_type, duration)
        cached = None
        if use_transcript_cache and cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached = None
        if cached and isinstance(cached.get("raw_segments"), list):
            raw_rows, language = cached["raw_segments"], cached.get("language", "unknown")
            audio_source = cached.get("audio_source", "cache")
            _log(log, f"Re-using the saved transcript from an earlier run (Whisper {settings.label()}, "
                      f"{language_mode}); skipping Whisper.")
            _emit(progress, "Transcribing", 100, 75, "Re-used saved transcript")
        else:
            if settings.device == "cuda" and use_ollama:
                unload_ollama_models(log)
            raw_rows, language, audio_source, used = _transcription_stage(
                media, wav_path, temp, settings, duration, language_mode, progress, cancel_event, log)
            if used != settings:
                settings = used
                cache_path = transcript_cache_path(media, settings.model, language_mode, settings.device,
                                                   settings.compute_type, duration)
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({
                    "video": str(media), "whisper_model": settings.model, "device": settings.device,
                    "compute_type": settings.compute_type, "language_mode": language_mode,
                    "asr_version": ASR_PIPELINE_VERSION, "language": language, "audio_source": audio_source,
                    "raw_segments": raw_rows,
                }, ensure_ascii=False), encoding="utf-8")
            except OSError as exc:
                _log(log, f"Could not save the transcript cache: {exc}")
        transcript, clean_stats = clean_transcript(raw_rows, language_mode)
        quality = transcript_quality(transcript)
        quality["segments_dropped"] = clean_stats["segments_dropped"]
        quality["words_dropped"] = clean_stats["words_dropped"]
        if clean_stats["segments_dropped"] or clean_stats["words_dropped"]:
            _log(log, f"Transcript clean-up removed {clean_stats['segments_dropped']} unreliable segment(s) and "
                      f"{clean_stats['words_dropped']} hallucinated/low-confidence word(s).")

    transcript_usable = bool(quality["usable"])
    (job_dir / "transcript.json").write_text(
        json.dumps({
            "language": language, "language_mode": language_mode, "audio_source": audio_source,
            "whisper": {"model": settings.model, "device": settings.device, "compute_type": settings.compute_type},
            "quality": quality, "segments": transcript,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    if transcript_usable:
        _log(log, f"Transcript validated: {quality['word_count']} words; detected initial language={language}.")
    else:
        _log(log, "WARNING: No reliable transcript was produced. Empty/punctuation-only text will be ignored; "
                  "visual and audio ranking will be used instead.")
    _check_cancel(cancel_event)

    # --- live chat (YouTube live-stream replays only) -------------------------------------------
    chat: list[float] = []
    chat_url = url if is_url and re.search(r"(^|[/.])(youtube\.com|youtu\.be)/", url.lower()) else None
    if not chat_url and not is_url:
        match = YOUTUBE_ID_IN_NAME.search(video.name)
        if match:
            chat_url = f"https://www.youtube.com/watch?v={match.group(1)}"
    if use_chat and chat_url:
        with timer.stage("live chat"):
            _emit(progress, "Live chat", 0, 75.2, "Checking for a YouTube live-chat replay...")
            try:
                with tempfile.TemporaryDirectory(prefix="clipforge_chat_") as chat_dir:
                    offsets = fetch_live_chat(chat_url, Path(chat_dir), cookies_browser, log, cancel_event)
                if offsets:
                    chat = chat_spikes(offsets, duration)
                    _log(log, f"Live-chat replay: {len(offsets)} messages; "
                              f"{sum(1 for v in chat if v >= 0.5)} second(s) where chat sped up sharply.")
                else:
                    _log(log, "No live-chat replay for this video (only YouTube live-stream replays have one).")
            except JobCancelled:
                raise
            except Exception as exc:
                _check_cancel(cancel_event)
                _log(log, f"Live-chat replay unavailable; continuing without it. Details: {str(exc)[:300]}")

    # --- analysis video: the local file, or the low-res proxy for URL sources -------------------
    analysis_video: Optional[Path] = video
    whole_motion: list[float] = []
    if proxy_job is not None:
        _emit(progress, "Downloading", 100, 75.5, "Waiting for the analysis proxy download...")
        with timer.stage("proxy wait"):
            try:
                analysis_video, whole_motion, motion_seconds = proxy_job.wait()
                timer.add("download (proxy, in background)", proxy_job.seconds - motion_seconds)
                timer.add("stream motion", motion_seconds)
            except Exception as exc:  # noqa: BLE001
                _check_cancel(cancel_event)
                _log(log, f"Analysis proxy download failed ({str(exc)[:200]}); downloading the full video instead.")
                video = sources.download_full(url, input_dir, cookies_browser=cookies_browser,
                                              log=lambda m: _log(log, m), cancel_event=cancel_event,
                                              on_progress=dl_progress("Downloading video", 75.5, 1))
                analysis_video = video
                source_mode = "full download (proxy failed)"
                src_w, src_h = video_dimensions(video)
    if use_ollama and use_signals and not whole_motion and analysis_video is not None:
        with timer.stage("stream motion"):
            _emit(progress, "Visual analysis", 0, 75.6, "Measuring on-screen motion across the stream...")
            try:
                whole_motion = stream_motion(analysis_video, duration, cancel_event)
            except JobCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - motion is one of three signals
                _log(log, f"Stream motion unavailable: {exc}")

    frame_dir = temp / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    motion: list[float] = [0.0] * max(1, int(math.ceil(duration)))
    vision_model, text_model, edge_model = resolve_models(ollama_model, picker_model, log)
    if use_ollama:
        _log(log, f"Picking/critic model: {text_model} · frame review: {vision_model} · edge check: {edge_model}")

    # --- candidates: transcript moments + signal spikes ------------------------------------------
    moments: list[Moment] = []
    if transcript_usable and use_ollama:
        with timer.stage("moment picking (transcript)"):
            _log(log, f"Reading the transcript in 12-minute chunks for shareable moments ({text_model})...")
            _emit(progress, "Finding moments", 0, 76, "Reading the transcript for shareable moments...")
            try:
                moments = pick_moments(
                    transcript, duration, loud, chat, url=OLLAMA_URL, model=text_model,
                    min_len=min_len, max_len=max_len, game_profile=prompt_profile, creator=creator,
                    on_chunk=lambda i, n, found: _emit(
                        progress, "Finding moments", i / n * 100, 76 + 7 * i / n,
                        f"Read transcript chunk {i}/{n} · {found} possible moments so far"),
                    log=lambda m: _log(log, m), check_cancel=lambda: _check_cancel(cancel_event),
                )
                _log(log, f"Transcript moment picker found {len(moments)} possible moments.")
            except JobCancelled:
                raise
            except Exception as exc:
                _log(log, f"Transcript moment picker unavailable; continuing with signals. Details: {exc}")
    signal_list: list[SignalWindow] = []
    if use_ollama and use_signals:
        spikes = motion_spikes(whole_motion) if whole_motion else []
        signal_list = signal_windows(duration, loud, chat, spikes, exclude=[(m.start, m.end) for m in moments])
        _log(log, f"Signal path: {len(signal_list)} window(s) from loudness/chat/motion spikes that the transcript "
                  "picker did not cover (silent clutches, fails, surprises).")
        for w in signal_list:
            _log(log, f"  {w.start/60:5.1f}m  strength {w.strength:.2f}  ({w.describe()})")
    candidates: list[Candidate] = []
    if (moments or signal_list) and analysis_video is not None:
        with timer.stage("frame sampling"):
            _emit(progress, "Visual analysis", 0, 83, "Sampling frames around the candidate moments...")
            windows = [(m.start - 20, m.end + 20) for m in moments] + [(w.start - 20, w.end + 20) for w in signal_list]
            decoded = extract_window_frames(analysis_video, windows, frame_dir, cancel_event)
            motion = motion_from_frames(frame_dir, duration, None, cancel_event)
            _log(log, f"Sampled frames for {decoded / 60:.1f} of {duration / 60:.1f} minutes around the candidates.")
        moment_candidates = candidates_from_moments(moments, duration, energies, transcript, motion, loud, chat,
                                                    min_len, max_len)
        signal_candidates = candidates_from_signals(signal_list, duration, energies, transcript, transcript_usable,
                                                    motion, loud, chat, first_index=len(moment_candidates))
        candidates = merge_candidates(moment_candidates, signal_candidates)
        _log(log, f"{len(moment_candidates)} transcript moment(s) + {len(candidates) - len(moment_candidates)} "
                  "signal window(s) after de-duplication.")
    if not candidates:
        if analysis_video is None:
            raise RuntimeError("No video is available for analysis.")
        with timer.stage("frame sampling"):
            motion = visual_motion_by_second(analysis_video, duration, frame_dir, progress, cancel_event, log)
        candidates = build_candidates(duration, energies, transcript, motion, window_len, stride,
                                      transcript_usable, loud, chat)
        if transcript_usable:
            for c in candidates:
                c.start, c.end = snap_to_speech(c.start, c.end, transcript, duration,
                                                max(10, int(window_len * 0.6)), int(window_len * 1.3))
                c.transcript = transcript_for_range(transcript, c.start, c.end)[:900]
        _log(log, f"Generated {len(candidates)} scanned candidate windows.")
    _emit(progress, "Preparing candidates", 100, 84, f"{len(candidates)} candidates")
    if not candidates:
        raise RuntimeError("No candidate clips could be generated from this video.")

    # --- vision review -------------------------------------------------------------------------
    if use_ollama:
        with timer.stage("vision review"):
            try:
                _log(log, f"Checking candidates against the gameplay frames with local visual AI ({vision_model})...")
                candidates = ollama_rank(candidates, vision_model, frame_dir, progress, cancel_event, log,
                                         prompt_profile, transcript_usable, creator)
            except JobCancelled:
                raise
            except Exception as exc:
                _log(log, f"Local visual AI review unavailable; using transcript/signal scores. Details: {exc}")
                for c in candidates:
                    _score_without_vision(c)
                _emit(progress, "AI ranking", 100, 88, "Using fallback ranking")
    else:
        _log(log, "Local AI disabled; using local visual/audio/spike scores.")
        for c in candidates:
            _score_without_vision(c)
        _emit(progress, "AI ranking", 100, 88, "Local visual/audio ranking complete")

    # --- finalist checks: edges, critic, title fact-check -------------------------------------
    rejected_screens = {"menu_lobby", "starting_screen", "loading_screen", "scoreboard"}
    pool = [c for c in sorted(candidates, key=lambda c: c.final_score, reverse=True)
            if c.screen_type not in rejected_screens
            and not (min_score is not None and 0 <= c.quality < min_score * 0.7)]
    finalists = pool[:max(count * 2, count + 3)]
    checked = bool(use_ollama and finalists)
    if checked:
        with timer.stage("edge check"):
            _emit(progress, "Final checks", 0, 88, f"Checking the first/last seconds of {len(finalists)} finalists...")
            try:
                edge_dir = temp / "edges"
                edge_labels = edge_review(finalists, analysis_video, edge_model, edge_dir, log, cancel_event)
                for c in finalists:
                    labels = edge_labels.get(c.index)
                    if not labels:
                        continue
                    c.edge_labels = labels
                    before = (c.start, c.end)
                    problem = apply_edge_labels(c, labels, transcript, transcript_usable, duration, min_len, max_len)
                    if problem:
                        c.rejected = f"edge check: {problem}"
                        c.final_score *= 0.1
                        _log(log, f"  dropped {before[0]/60:.1f}m clip: {problem}")
                    elif (c.start, c.end) != before:
                        c.transcript = transcript_for_range(transcript, c.start, c.end)[:900]
                        _log(log, f"  trimmed {before[0]/60:.1f}m clip to {c.start:.1f}-{c.end:.1f}s "
                                  f"(edges: {', '.join(labels)})")
            except JobCancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                _log(log, f"Edge check unavailable: {exc}")
        live = [c for c in finalists if not c.rejected]
        if live:
            with timer.stage("critic + title fact-check"):
                _emit(progress, "Final checks", 50, 89, f"Critic re-judging {len(live)} finalists...")
                items = [{
                    "index": c.index,
                    "seconds": round(c.end - c.start),
                    "title": c.hook_title or c.social_title,
                    "category": c.category or c.event_type,
                    "speech_before": transcript_for_range(transcript, max(0, c.start - 20), c.start)[:300],
                    "speech_in_clip": transcript_for_range(transcript, c.start, c.end)[:900],
                    "speech_after": transcript_for_range(transcript, c.end, min(duration, c.end + 8))[:200],
                    "frames_summary": " ".join(x for x in (c.setup, c.payoff, c.reason) if x)[:400],
                    # Evidence the critic can check, unlike a hyped title: the line the picker built the clip on.
                    "key_line": c.payoff_quote[:200],
                } for c in live]
                sheets = finalist_sheets(live, frame_dir, text_model)
                if sheets:
                    # Titles first: a hype title ("squad PANICS as police arrive") talked the critic into
                    # scoring a calm drive 88; with the literal title it scored the same clip ~20.
                    _log(log, f"Making {len(live)} finalist title(s) literal against the frames and speech...")
                    fact_check_titles(live, transcript, model=text_model, creator=creator, log=log,
                                      images=sheets, check_all=True)
                    for item, c in zip(items, live):
                        item["title"] = c.hook_title or c.social_title
                    # A model that can see judges the frames itself; the vision model's summary
                    # anchored it on that model's mistakes ("spotted by security" in a nightclub).
                    for item in items:
                        item.pop("frames_summary", None)
                    _log(log, f"Critic ({text_model}) is checking every claim against the frames as well as the speech.")
                verdicts = critique(items, url=OLLAMA_URL, model=text_model, game_profile=prompt_profile,
                                    creator=creator, log=lambda m: _log(log, m), images=sheets)
                for c in live:
                    verdict = verdicts.get(c.index)
                    if not verdict:
                        continue
                    c.critic_score = verdict["stranger_score"]
                    c.critic_note = verdict["what_happens"] or verdict["reason"]
                    c.title_claims = verdict.get("claims") or None
                    c.critic_frames = verdict.get("frames_show", "")
                    if not c.hook_title and verdict["fixed_title"]:
                        c.hook_title = c.social_title = verdict["fixed_title"]   # e.g. an untitled silent clip
                        c._title_rewritten = True  # type: ignore[attr-defined]
                    elif not verdict["title_supported"]:
                        fixed = verdict["fixed_title"]
                        if fixed and fixed.lower() != (c.hook_title or "").lower():
                            _log(log, f"  title fixed: {c.hook_title!r} -> {fixed!r}")
                            c.hook_title = c.social_title = fixed
                            c._title_rewritten = True  # type: ignore[attr-defined]
                        else:
                            _log(log, f"  title removed (not supported by the clip): {c.hook_title!r}")
                            c.hook_title = ""
                    base_quality = c.quality if c.quality >= 0 else c.critic_score
                    c.quality = 0.5 * c.critic_score + 0.5 * base_quality
                    c.final_score = 0.45 * c.critic_score / 100 + 0.55 * c.final_score
                    if not verdict["keep"]:
                        # A nudge, not a cap: the model's single yes/no flips between runs, while its
                        # blended score (own rating + four checks) ranked clips consistently.
                        c.final_score *= 0.85
                if burn_title:
                    # Second look for anything the critic kept but that still reads as hype.
                    fact_check_titles(live, transcript, model=text_model, creator=creator, log=log, images=sheets)
        _emit(progress, "Final checks", 100, 90, "Final checks complete")

    leaderboard = sorted(finalists or candidates, key=lambda c: c.final_score, reverse=True)
    _log(log, f"{len(candidates)} possible moments; finalists best first:")
    for c in leaderboard[:12]:
        score = f"{c.quality:3.0f}" if c.quality >= 0 else "  -"
        flag = " REJECTED" if c.rejected else ""
        note = f"  [{c.critic_note[:60]}]" if c.critic_note else ""
        _log(log, f"  {score}  {c.start/60:5.1f}m  {(c.end - c.start):4.0f}s  {c.origin:6s} "
                  f"{(c.hook_title or c.social_title or c.reason or '')[:60]}{note}{flag}")
    # Once the final checks ran, only checked finalists may be rendered: a clip they rejected, or one
    # they never saw, can never slip through.
    selection_pool = finalists if checked else candidates
    selected = choose_top(selection_pool, count, min_score)
    if not selected and min_score is not None:
        near_misses = [c for c in selection_pool
                       if not c.rejected and (c.quality < 0 or c.quality >= max(1.0, min_score - 15))]
        selected = choose_top(near_misses, 1)
        if selected:
            _log(log, f"No moment reached the minimum score of {int(min_score)}; keeping the closest one so you can "
                      "judge it. Lower 'Min score' to get more clips.")
    if not selected:
        _log(log, "No moment in this video cleared the bar, so nothing was rendered. Most streams are filler; this "
                  "is the picker refusing to pad. Lower 'Min score' or use a stronger picking model to see more.")
    else:
        _log(log, f"Kept the best {len(selected)} of {len(candidates)} possible moments.")
    _log(log, "AI selection is based on the transcript and sampled frames; always review clips before publishing.")

    for c in selected:
        paths = [frame_dir / f"frame_{t:06d}.jpg" for t in range(int(c.start), int(math.ceil(c.end)))]
        try:
            traits = clip_style.clip_frame_traits([p for p in paths if p.exists()])
            c.split_screen = traits["split_screen"]
            c.content_band = tuple(traits["band"])
            c.seam_x = traits.get("seam_x")
        except Exception as exc:  # noqa: BLE001 - styling hints are optional
            _log(log, f"Could not inspect clip frames for layout hints: {exc}")

    # --- full-quality video for the chosen clips only ------------------------------------------
    sections: list = []
    if selected and video is None:
        with timer.stage("download (clip sections)"):
            ranges = sources.merge_ranges([(c.start, c.end) for c in selected], duration)
            _log(log, f"Downloading {len(ranges)} full-quality section(s) "
                      f"({sum(b - a for a, b in ranges) / 60:.1f} of {duration / 60:.0f} min) for the chosen clips...")
            try:
                from concurrent.futures import ThreadPoolExecutor

                done = [0]

                def fetch(item):
                    i, (a, b) = item
                    path = sources.download_section(url, a, b, remote_dir / "sections", cookies_browser=cookies_browser,
                                                    log=lambda m: _log(log, m), cancel_event=cancel_event)
                    section = sources.align_section(sources.Section(path, a, b), wav_path, _run_process)
                    done[0] += 1
                    _emit(progress, "Downloading clips", done[0] / len(ranges) * 100,
                          90 + 2 * done[0] / len(ranges), f"Downloaded section {done[0]}/{len(ranges)}")
                    _log(log, f"  section {i}: {a:.0f}-{b:.0f}s -> {path.name}, starts at {section.offset:.2f}s "
                              f"(audio match {section.confidence:.2f})")
                    if section.confidence < 0.45 or abs(section.offset - a) > 15:
                        raise RuntimeError(f"section {i} could not be aligned to the stream audio")
                    return section

                # Ranged downloads are latency-bound, so three at a time is much faster than one by one.
                with ThreadPoolExecutor(max_workers=3) as pool:
                    sections = list(pool.map(fetch, enumerate(ranges, 1)))
                _check_cancel(cancel_event)
                missing = [c for c in selected if sources.section_for(sections, c.start, c.end) is None]
                if missing:
                    raise RuntimeError(f"{len(missing)} clip(s) not covered by the downloaded sections")
            except JobCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - one full download always works
                _check_cancel(cancel_event)
                _log(log, f"Section download failed ({str(exc)[:240]}); downloading the full video instead.")
                sections = []
                video = sources.download_full(url, input_dir, cookies_browser=cookies_browser,
                                              log=lambda m: _log(log, m), cancel_event=cancel_event,
                                              on_progress=dl_progress("Downloading video", 90, 2))
                source_mode = "full download (sections failed)"
        if sections:
            src_w, src_h = video_dimensions(sections[0].path)
        elif video is not None:
            src_w, src_h = video_dimensions(video)

    def render_source(c: Candidate) -> tuple[Path, float]:
        if sections:
            section = sources.section_for(sections, c.start, c.end)
            if section is not None:
                return section.path, section.offset
        return video, 0.0

    facecam: Optional[Box] = None
    clip_facecams: dict[int, Box] = {}
    if selected and render_layout in (LAYOUT_AUTO, LAYOUT_SPLIT):
        with timer.stage("facecam"):
            if facecam_mode == FACECAM_AUTO:
                _emit(progress, "Facecam", 50, 92, "Looking for the streamer's webcam...")
                try:
                    # Is there a webcam at all, and where does it usually sit? For URL sources, ask the
                    # low-res proxy across the whole stream (the clip sections may be just one minute);
                    # each clip's own full-quality frames refine the box below.
                    stream_wide = video if video is not None else analysis_video
                    facecam = clip_style.detect_facecam(facecam_samples(stream_wide, duration, []), src_w, src_h,
                                                        temp / "facecam", _run_process, log=lambda m: _log(log, m))
                    if facecam is None and sections:
                        facecam = clip_style.detect_facecam(facecam_samples(None, duration, sections), src_w, src_h,
                                                            temp / "facecam_sections", _run_process,
                                                            log=lambda m: _log(log, m))
                except Exception as exc:
                    _log(log, f"Facecam detection failed; using the no-facecam layout. Details: {exc}")
            elif facecam_mode in clip_style.FACECAM_CORNERS:
                facecam = clip_style.corner_facecam(facecam_mode, src_w, src_h)
                _log(log, f"Facecam set manually: {facecam_mode} corner.")
            if render_layout == LAYOUT_SPLIT and facecam is None:
                _log(log, "No facecam available for the split layout; falling back to Gameplay zoom.")
            # Streamers move and resize their webcam during a stream (a bigger cam while talking, a
            # smaller one in a heist), so each clip is checked on its own frames. Only done when the
            # stream really has a webcam, so a game character's face in a cutscene is never taken
            # for one; a clip whose own check fails keeps the stream-wide box.
            if facecam is not None and facecam_mode == FACECAM_AUTO:
                for n, c in enumerate(selected, 1):
                    _check_cancel(cancel_event)
                    rsrc, off = render_source(c)
                    span = c.end - c.start
                    samples = [(rsrc, max(0.0, c.start - off) + span * (0.05 + 0.9 * i / 11)) for i in range(12)]
                    try:
                        own = clip_style.detect_facecam(samples, src_w, src_h, temp / f"facecam_{c.index}",
                                                        _run_process)
                    except Exception:  # noqa: BLE001 - the stream-wide box is a fine fallback
                        own = None
                    if own is not None and 0.5 <= own.w / max(1, facecam.w) <= 2.0:
                        clip_facecams[c.index] = own
                        moved = abs(own.x - facecam.x) > 0.04 * src_w or abs(own.y - facecam.y) > 0.04 * src_h
                        _log(log, f"  clip {n}: webcam at x={own.x}, y={own.y}, {own.w}x{own.h}"
                                  + (" (moved from the stream-wide position)" if moved else ""))
                    else:
                        _log(log, f"  clip {n}: using the stream-wide webcam box")

    try:
        encoders = _run_process(["ffmpeg", "-hide_banner", "-encoders"],
                                capture_output=True, text=True, check=False).stdout
        nvenc_available = "h264_nvenc" in encoders
    except Exception:
        nvenc_available = False
    nvenc_active = bool(use_nvenc and nvenc_available)
    _log(log, "NVIDIA NVENC encoding enabled." if nvenc_active else "Using CPU video encoding.")
    has_ass = ffmpeg_has_ass_filter()
    caption_family = clip_style.caption_font_family()

    manifest = []
    title_dir = temp / "titles"
    with timer.stage("render"):
        for n, candidate in enumerate(selected, 1):
            _check_cancel(cancel_event)
            name = f"clip_{n:02d}_{int(candidate.start):06d}s"
            mp4 = job_dir / f"{name}.mp4"
            srt = job_dir / f"{name}.srt"
            ass = job_dir / f"{name}.ass"
            thumbnail = job_dir / f"{name}.jpg"
            post_file = job_dir / f"{name}.post.txt"
            rsrc, offset = render_source(candidate)
            plan = clip_style.plan_layout(render_layout, src_w, src_h, clip_facecams.get(candidate.index, facecam),
                                          split_screen=candidate.split_screen, band=tuple(candidate.content_band),
                                          seam_x=candidate.seam_x)
            _log(log, f"Rendering clip {n}/{len(selected)}: {candidate.start:.1f}s – {candidate.end:.1f}s "
                      f"({candidate.end - candidate.start:.0f}s, {plan.name})")
            subtitle_written = write_clip_srt(transcript, candidate.start, candidate.end, srt) if transcript_usable else False
            ass_written = (clip_style.write_bold_captions(transcript, candidate.start, candidate.end, ass,
                                                          caption_y=plan.caption_y, font_family=caption_family)
                           if transcript_usable and burn_captions else False)
            if not subtitle_written:
                srt.unlink(missing_ok=True)
            if not ass_written:
                ass.unlink(missing_ok=True)
            burn_ass = ass if ass_written and has_ass else None
            if ass_written and burn_ass is None:
                _log(log, "FFmpeg lacks ASS/libass support; retaining caption sidecars without burning them in.")
            title_png = title_dir / f"{name}.png"
            title_size = None
            if burn_title and candidate.hook_title:
                try:
                    title_size = clip_style.render_title_card(candidate.hook_title, title_png)
                except Exception as exc:  # noqa: BLE001 - a title is never worth failing the clip
                    _log(log, f"Could not render the hook title card: {exc}")
            render_vertical(rsrc, candidate, mp4, nvenc_active, log, render_layout,
                            ass_path=burn_ass, normalize_audio=normalize_audio, plan=plan,
                            title_png=title_png if title_size else None, source_offset=offset)
            create_thumbnail(rsrc, candidate, thumbnail, rendered=mp4, source_offset=offset)
            write_post_file(post_file, candidate)
            row = asdict(candidate)
            manifest_row = {
                **row,
                "file": mp4.name,
                "subtitle": srt.name if subtitle_written else None,
                "ass_subtitle": ass.name if ass_written else None,
                "thumbnail": thumbnail.name if thumbnail.exists() else None,
                "post_file": post_file.name,
                "title_suggestion": candidate.hook_title or candidate.social_title or f"Gaming highlight #{n}",
                "social_caption": candidate.social_caption,
                "hashtags": candidate.hashtags or [],
                "captions_burned": bool(burn_ass),
                "title_burned": bool(title_size),
                "layout": plan.name,
                "facecam_box": plan.facecam.as_dict() if plan.facecam else None,
                "render_source": rsrc.name if rsrc else None,
                "posted": False,
            }
            manifest.append(manifest_row)
            (job_dir / f"{name}.metadata.json").write_text(
                json.dumps(manifest_row, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            pct = n / len(selected) * 100
            _emit(progress, "Rendering clips", pct, 92 + pct * 0.08, f"Rendered {n}/{len(selected)} clips")

    timer.add("total", time.monotonic() - job_started)
    _log(log, "Stage timings: " + " · ".join(f"{k} {v:.0f}s" for k, v in timer.seconds.items()))
    (job_dir / "clips.json").write_text(json.dumps({
        "app_version": APP_VERSION,
        "source": str(video or media),
        "source_input": source,
        "source_mode": source_mode,
        "sections": [{"file": s.path.name, "start": s.start, "end": s.end, "offset": s.offset,
                      "audio_match": s.confidence} for s in sections],
        "creator": creator,
        "language": language,
        "whisper": {"model": settings.model, "device": settings.device, "compute_type": settings.compute_type},
        "transcript_quality": quality,
        "transcript_usable": transcript_usable,
        "game_profile": game_profile,
        "story_auto": auto_mode,
        "clip_count": count,
        "clip_length": clip_length,
        "moment_length": [min_len, max_len],
        "scan_interval": stride,
        "min_score": min_score,
        "render_layout": render_layout,
        "facecam_mode": facecam_mode,
        "facecam_box": facecam.as_dict() if facecam else None,
        "burn_captions": burn_captions,
        "burn_title": burn_title,
        "normalize_audio": normalize_audio,
        "live_chat_used": bool(chat),
        "vision_model": vision_model,
        "edge_model": edge_model,
        "picker_model": text_model,
        "nvenc": nvenc_active,
        "transcript_moments": len(moments),
        "signal_windows": len(signal_list),
        "candidates_considered": len(candidates),
        # Every candidate and why it did or did not make it, best first (for "why wasn't X picked?").
        "leaderboard": [{
            "start": round(c.start, 1), "end": round(c.end, 1), "origin": c.origin,
            "title": c.hook_title or c.social_title, "quality": round(c.quality, 1),
            "final_score": round(c.final_score, 3), "critic_score": c.critic_score, "critic_note": c.critic_note,
            "screen_type": c.screen_type, "rejected": c.rejected,
            "chosen": any(c is s for s in selected),
        } for c in sorted(candidates, key=lambda c: c.final_score, reverse=True)],
        "timings_seconds": timer.seconds,
        "clips": manifest
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    _emit(progress, "Complete", 100, 100, f"Finished: {len(selected)} clips exported")
    _log(log, f"Finished. Output folder: {job_dir}")
    return job_dir
