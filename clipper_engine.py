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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import requests
import yt_dlp
from faster_whisper import WhisperModel

import clip_style
from clip_style import (
    FACECAM_AUTO, LAYOUT_AUTO, LAYOUT_SPLIT, Box,
)
from moments import (
    Moment, chat_spikes, clean_transcript, extract_json_object, loudness_spikes, ollama_json, tidy_hook_title,
    parse_live_chat, pick_moments, range_peak, snap_to_speech,
)
from paths import INPUT_DIR, OUTPUT_DIR

DEFAULT_INPUT_DIR = INPUT_DIR
DEFAULT_OUTPUT_DIR = OUTPUT_DIR
OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"
APP_VERSION = "1.5.0"
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
    origin: str = "window"        # "moment" (picked from the transcript) or "window" (sliding scan)
    category: str = ""
    hook_title: str = ""          # burned-in on-video title
    moment_score: float = 0.0     # 0..100 from the transcript pass
    loud_score: float = 0.0       # 0..1 peak loudness spike in the clip
    chat_score: float = 0.0       # 0..1 peak live-chat spike in the clip
    quality: float = -1.0         # 0..100 publish-readiness; -1 = not judged by AI
    split_screen: bool = False
    content_band: tuple = (0.0, 1.0)


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


def transcribe(audio_path: Path, model_name: str, device: str, duration: float,
               progress: Optional[ProgressCallback] = None,
               cancel_event: Optional[threading.Event] = None,
               log: Optional[LogCallback] = None,
               language_mode: str = "Mixed (English + Bengali + Hindi)"):
    """
    Transcribe the already-extracted 16 kHz WAV rather than decoding the video
    again through PyAV. Mixed mode asks faster-whisper to detect language per
    segment instead of locking the whole recording to the first detected language.
    """
    compute = "float16" if device == "cuda" else "int8"
    language_map = {
        "English": "en",
        "Bengali": "bn",
        "Hindi": "hi",
    }
    forced_language = language_map.get(language_mode)
    multilingual = forced_language is None
    mode_text = "per-segment mixed-language detection" if multilingual else f"forced {language_mode}"
    _log(log, f"Loading Whisper {model_name} on {device} ({compute}); {mode_text}...")
    _emit(progress, "Transcribing", 0, 15, "Loading speech recognition model...")
    model = WhisperModel(model_name, device=device, compute_type=compute)
    prompt = (
        "Casual gaming voice chat. Speakers may switch between English, Bengali "
        "(Bangla) and Hindi within the same conversation. Preserve the language "
        "actually spoken; do not translate. Keep gaming terms, player names and "
        "short reactions such as clutch, ace, headshot, revive, GG, bhai, yaar."
    )
    kwargs = {
        "beam_size": 5,
        "vad_filter": True,
        "word_timestamps": True,
        "condition_on_previous_text": False,
        "initial_prompt": prompt,
        "multilingual": multilingual,
        "language_detection_threshold": 0.35,
        "language_detection_segments": 3,
    }
    if forced_language:
        kwargs["language"] = forced_language
    segments, info = model.transcribe(str(audio_path), **kwargs)
    rows = []
    start_time = time.monotonic()
    last_report = 0.0
    for seg in segments:
        _check_cancel(cancel_event)
        rows.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
            # Confidence fields let clean_transcript() drop hallucinated segments.
            "avg_logprob": round(float(getattr(seg, "avg_logprob", 0.0)), 4),
            "no_speech_prob": round(float(getattr(seg, "no_speech_prob", 0.0)), 4),
            "compression_ratio": round(float(getattr(seg, "compression_ratio", 1.0)), 4),
            "words": [
                {"start": float(w.start), "end": float(w.end), "word": w.word,
                 "probability": round(float(getattr(w, "probability", 1.0)), 4)}
                for w in (seg.words or [])
            ]
        })
        now = time.monotonic()
        processed = min(duration, max(0.0, float(seg.end)))
        pct = processed / duration * 100 if duration else 0
        elapsed = max(0.1, now - start_time)
        media_per_second = processed / elapsed if processed > 0 else 0
        remaining = (duration - processed) / media_per_second if media_per_second > 0 else None
        if now - last_report >= 0.25 or pct >= 99.9:
            message = f"Transcribed {processed/60:.1f} / {duration/60:.1f} min"
            if remaining is not None:
                message += f" · ETA {format_eta(remaining)}"
            _emit(progress, "Transcribing", pct, 15 + pct * 0.60, message)
            last_report = now
    _emit(progress, "Transcribing", 100, 75, "Transcription complete")
    return rows, getattr(info, "language", "unknown")


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
        import cv2
        import numpy as np
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

    motion = [0.0] * max(1, int(math.ceil(duration)))
    previous = None
    for idx, frame_path in enumerate(frame_files):
        _check_cancel(cancel_event)
        frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue
        if previous is not None and idx < len(motion):
            diff = cv2.absdiff(frame, previous)
            motion[idx] = float(np.mean(diff) / 255.0)
        previous = frame
        if progress and (idx % 20 == 0 or idx == len(frame_files) - 1):
            pct = (idx + 1) / max(1, len(frame_files)) * 100
            _emit(progress, "Visual analysis", pct, 75.5 + pct * 0.045,
                  f"Analysing gameplay movement: {idx+1}/{len(frame_files)} sampled frames")

    positive = [v for v in motion if v > 0]
    ceiling = float(np.percentile(positive, 95)) if positive else 0.01
    ceiling = max(ceiling, 0.008)
    normalised = [max(0.0, min(1.0, value / ceiling)) for value in motion]
    _log(log, f"Visual motion analysis complete ({len(frame_files)} sampled frames).")
    return normalised


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
            hashtags=list(moment.hashtags), event_type=moment.category,
        )
        _score_without_vision(candidate)
        candidates.append(candidate)
    return candidates


def _score_without_vision(c: Candidate) -> None:
    """Scores used when the vision review is disabled or unavailable."""
    signal = max(c.loud_score, c.chat_score)
    if c.origin == "moment":
        c.final_score = 0.75 * (c.moment_score / 100) + 0.15 * signal + 0.10 * c.visual_score
        c.quality = c.moment_score
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
        rows.append(row)
    who = (f"The streamer is {creator}; name them in titles. " if creator
           else "The streamer's name is unknown; do not invent one. ")
    prompt = (
        f"You are a harsh short-form gaming editor reviewing {game_profile} footage for YouTube Shorts / TikTok. {who}"
        "Each attached image is a 3x3 contact sheet for one candidate; its nine frames are in chronological order. "
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
        "words, may end with one emoji; keep proposed_hook_title if it is accurate. social_caption must be a ready-to-post "
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
                    title_png: Optional[Path] = None) -> None:
    duration = max(0.1, candidate.end - candidate.start)
    if plan is None:
        src_w, src_h = video_dimensions(video)
        plan = clip_style.plan_layout(layout, src_w, src_h, None, split_screen=candidate.split_screen,
                                      band=tuple(candidate.content_band))
    has_title = bool(title_png and title_png.exists())
    graph = build_render_graph(plan, duration, title=has_title,
                               ass_path=ass_path if ass_path and ass_path.exists() else None)
    inputs = ["-ss", f"{candidate.start:.3f}", "-i", str(video)]
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
                     rendered: Optional[Path] = None) -> bool:
    """Grab a preview frame; prefer the finished vertical clip so the preview matches what is posted."""
    if rendered is not None and rendered.is_file():
        at = max(0.0, (candidate.end - candidate.start) * 0.45)
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{at:.2f}", "-i", str(rendered),
               "-frames:v", "1", "-vf", "scale=540:-2", "-q:v", "3", str(output)]
    else:
        midpoint = candidate.start + (candidate.end - candidate.start) * 0.55
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", str(midpoint),
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


def _transcribe_clean(audio_path: Path, model_name: str, device: str, duration: float, language_mode: str,
                      progress, cancel_event, log):
    rows, language = transcribe(audio_path, model_name, device, duration, progress, cancel_event, log, language_mode)
    cleaned, stats = clean_transcript(rows, language_mode)
    if stats["segments_dropped"] or stats["words_dropped"]:
        _log(log, f"Transcript clean-up removed {stats['segments_dropped']} unreliable segment(s) and "
                  f"{stats['words_dropped']} hallucinated/low-confidence word(s).")
    return cleaned, language, transcript_quality(cleaned)


def process_job(source: str, output_root: Path, input_dir: Path, *, count: Optional[int] = None,
                clip_length: Optional[int] = None, stride: Optional[int] = None, whisper_model: str = "small",
                whisper_device: str = "cpu", ollama_model: str = DEFAULT_OLLAMA_MODEL,
                use_ollama: bool = True, use_nvenc: bool = True,
                game_profile: str = "Auto / mixed games",
                cookies_browser: Optional[str] = None,
                language_mode: str = "Mixed (English + Bengali + Hindi)",
                render_layout: str = LAYOUT_AUTO,
                burn_captions: bool = True,
                normalize_audio: bool = True,
                facecam_mode: str = FACECAM_AUTO,
                min_score: Optional[float] = DEFAULT_MIN_SCORE,
                creator_name: str = "",
                burn_title: bool = True,
                use_chat: bool = True,
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

    is_url = source.strip().lower().startswith(("https://", "http://"))
    source_info: dict = {}
    if is_url:
        video = download_youtube(source.strip(), input_dir, progress, cancel_event, log, cookies_browser,
                                 info_out=source_info)
    else:
        video = Path(source).expanduser().resolve()
        if not video.is_file() or video.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError("Select a valid local video file (MP4, MKV, WebM, MOV or AVI).")
        _emit(progress, "Input", 100, 10, f"Using local file: {video.name}")
    _check_cancel(cancel_event)
    _log(log, f"Video: {video}")

    _emit(progress, "Inspecting video", 0, 10, "Reading video metadata...")
    duration = duration_seconds(video)
    src_w, src_h = video_dimensions(video)
    _log(log, f"Duration: {duration/60:.1f} minutes · {src_w}x{src_h}")
    auto_mode = count is None or clip_length is None or stride is None
    # Moments are variable length: a sentence-aligned story between min_len and max_len.
    if clip_length:
        max_len, min_len = int(clip_length), max(10, int(round(clip_length * 0.4)))
    else:
        max_len, min_len = 60, 20
    max_len = max(10, min(max_len, int(duration)))
    min_len = min(min_len, max_len)
    # Fixed windows are only the fallback when there is no usable speech or no local AI.
    window_len = clip_length or (30 if duration >= 60 else max(10, int(duration * 0.6)))
    stride = stride or max(5, min(10, window_len // 5))
    count = count or max(3, min(8, int(round(duration / 1200))))
    creator = (creator_name or "").strip() or str(source_info.get("channel") or source_info.get("uploader") or "").strip()
    _log(log, f"Up to {count} clips · {min_len}-{max_len}s moments · min score "
              f"{'off' if min_score is None else int(min_score)}/100 · layout: {render_layout}"
              + (f" · streamer: {creator}" if creator else ""))
    job_name = re.sub(r"[^A-Za-z0-9_-]+", "_", video.stem)[:70].strip("_") or "gaming_video"
    job_dir = output_root / job_name
    suffix = 2
    while job_dir.exists() and (job_dir / "clips.json").exists():
        job_dir = output_root / f"{job_name}_{suffix}"
        suffix += 1
    job_dir.mkdir(parents=True, exist_ok=True)

    # Extract once, then use the exact same WAV for loudness analysis and Whisper.
    # This avoids a second audio decode path through PyAV.
    with tempfile.TemporaryDirectory(prefix="clipforge_") as temp:
        wav_path = Path(temp) / "audio_16k_mono.wav"
        _log(log, "Extracting mono 16 kHz audio with FFmpeg...")
        _emit(progress, "Audio extraction", 0, 11, "Extracting audio...")
        extract_audio_wav(video, wav_path)
        _check_cancel(cancel_event)
        _emit(progress, "Audio analysis", 65, 13, "Measuring audio activity and loudness spikes...")
        raw_rms = audio_energy_by_second(wav_path, raw=True)
        energies = normalise_energy(raw_rms)
        loud = loudness_spikes(raw_rms)
        _log(log, f"Loudness spikes: {sum(1 for v in loud if v >= 0.5)} second(s) clearly above the local baseline.")
        _emit(progress, "Audio analysis", 100, 15, "Audio analysis complete")

        _log(log, f"Transcribing with faster-whisper ({whisper_model}, {whisper_device}); {language_mode}...")
        transcript, language, quality = _transcribe_clean(
            wav_path, whisper_model, whisper_device, duration, language_mode, progress, cancel_event, log)
        selected_audio_path = wav_path

        # Check individual stereo channels if a mixed-down track produced punctuation/no speech.
        if not quality["usable"] and audio_channel_count(video) >= 2:
            _log(log, "Transcript quality is poor. Checking left/right channels for isolated voice chat...")
            channel_candidates = [(transcript, language, quality, wav_path)]
            for channel in ("FL", "FR"):
                _check_cancel(cancel_event)
                channel_path = Path(temp) / f"audio_{channel.lower()}.wav"
                try:
                    extract_channel_wav(video, channel_path, channel)
                    channel_rows, channel_language, channel_quality = _transcribe_clean(
                        channel_path, whisper_model, whisper_device, duration, language_mode,
                        progress, cancel_event, log)
                    channel_candidates.append((channel_rows, channel_language, channel_quality, channel_path))
                    _log(log, f"{channel} channel: {channel_quality['word_count']} word tokens.")
                    if channel_quality["usable"]:
                        break
                except JobCancelled:
                    raise
                except Exception as exc:
                    _log(log, f"Could not transcribe {channel} channel: {exc}")
            transcript, language, quality, selected_audio_path = max(
                channel_candidates,
                key=lambda item: (item[2]["usable"], item[2]["word_count"], item[2]["alphanumeric_characters"])
            )

        # A punctuation-only result is never considered a transcript. Retry Medium if needed.
        if not quality["usable"] and whisper_model != "medium":
            _log(log, (
                f"Transcript validation failed ({quality['word_count']} words; "
                f"{quality['alphanumeric_characters']} alphanumeric characters). "
                "Retrying Whisper medium with per-segment mixed-language detection."
            ))
            retry_rows, retry_language, retry_quality = _transcribe_clean(
                selected_audio_path, "medium", whisper_device, duration, language_mode,
                progress, cancel_event, log)
            if (retry_quality["usable"], retry_quality["word_count"]) > (quality["usable"], quality["word_count"]):
                transcript, language, quality = retry_rows, retry_language, retry_quality
            if quality["usable"]:
                _log(log, f"Medium retry recovered {quality['word_count']} word tokens.")
            else:
                _log(log, "WARNING: Speech remains unusable. The voice-chat mix may be absent, very quiet or overlapped; captions will be skipped.")

    transcript_usable = bool(quality["usable"])
    (job_dir / "transcript.json").write_text(
        json.dumps({
            "language": language,
            "language_mode": language_mode,
            "audio_source": selected_audio_path.name,
            "quality": quality,
            "segments": transcript
        }, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    if transcript_usable:
        _log(log, f"Transcript validated: {quality['word_count']} words; detected initial language={language}.")
    else:
        _log(log, (
            "WARNING: No reliable transcript was produced. Empty/punctuation-only text will be ignored; "
            "visual and audio ranking will be used instead."
        ))
    _check_cancel(cancel_event)

    # Live-chat velocity (YouTube live-stream replays only).
    chat: list[float] = []
    is_youtube = is_url and re.search(r"(^|[/.])(youtube\.com|youtu\.be)/", source.strip().lower()) is not None
    chat_url = source.strip() if is_youtube else None
    if not chat_url and not is_url:
        # Files downloaded by ClipForge keep the YouTube ID in their name: "Title [VIDEOID].mp4".
        match = YOUTUBE_ID_IN_NAME.search(video.name)
        if match:
            chat_url = f"https://www.youtube.com/watch?v={match.group(1)}"
    if use_chat and chat_url:
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
            _log(log, f"Live-chat replay unavailable; continuing without it. Details: {str(exc)[:300]}")

    frame_temp_obj = tempfile.TemporaryDirectory(prefix="clipforge_frames_")
    frame_dir = Path(frame_temp_obj.name)
    motion = visual_motion_by_second(video, duration, frame_dir, progress, cancel_event, log)

    facecam: Optional[Box] = None
    if render_layout in (LAYOUT_AUTO, LAYOUT_SPLIT):
        if facecam_mode == FACECAM_AUTO:
            _emit(progress, "Facecam", 50, 80, "Looking for the streamer's webcam...")
            try:
                facecam = clip_style.detect_facecam(video, duration, src_w, src_h, frame_dir / "facecam",
                                                    _run_process, log=lambda m: _log(log, m))
            except Exception as exc:
                _log(log, f"Facecam detection failed; using the no-facecam layout. Details: {exc}")
        elif facecam_mode in clip_style.FACECAM_CORNERS:
            facecam = clip_style.corner_facecam(facecam_mode, src_w, src_h)
            _log(log, f"Facecam set manually: {facecam_mode} corner.")
        if render_layout == LAYOUT_SPLIT and facecam is None:
            _log(log, "No facecam available for the split layout; falling back to Gameplay zoom.")

    candidates: list[Candidate] = []
    if transcript_usable and use_ollama:
        _log(log, f"Reading the transcript in 12-minute chunks for shareable moments ({ollama_model})...")
        _emit(progress, "Finding moments", 0, 80.5, "Reading the transcript for shareable moments...")
        try:
            moments = pick_moments(
                transcript, duration, loud, chat, url=OLLAMA_URL, model=ollama_model,
                min_len=min_len, max_len=max_len, game_profile=game_profile, creator=creator,
                on_chunk=lambda i, n, found: _emit(
                    progress, "Finding moments", i / n * 100, 80.5 + 3.5 * i / n,
                    f"Read transcript chunk {i}/{n} · {found} possible moments so far"),
                log=lambda m: _log(log, m), check_cancel=lambda: _check_cancel(cancel_event),
            )
            _log(log, f"Transcript moment picker found {len(moments)} possible moments.")
            candidates = candidates_from_moments(moments, duration, energies, transcript, motion, loud, chat,
                                                 min_len, max_len)
        except JobCancelled:
            raise
        except Exception as exc:
            _log(log, f"Transcript moment picker unavailable; falling back to scanning windows. Details: {exc}")
    if not candidates:
        candidates = build_candidates(duration, energies, transcript, motion, window_len, stride,
                                      transcript_usable, loud, chat)
        if transcript_usable:
            # Even scanned windows should start on a sentence and finish the thought.
            for c in candidates:
                c.start, c.end = snap_to_speech(c.start, c.end, transcript, duration,
                                                max(10, int(window_len * 0.6)), int(window_len * 1.3))
                c.transcript = transcript_for_range(transcript, c.start, c.end)[:900]
        _log(log, f"Generated {len(candidates)} scanned candidate windows.")
    _emit(progress, "Preparing candidates", 100, 84, f"{len(candidates)} candidates")
    if not candidates:
        raise RuntimeError("No candidate clips could be generated from this video.")

    if use_ollama:
        try:
            _log(log, f"Checking candidates against the gameplay frames with local visual AI ({ollama_model})...")
            candidates = ollama_rank(
                candidates, ollama_model, frame_dir, progress, cancel_event, log,
                game_profile, transcript_usable, creator
            )
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

    leaderboard = sorted(candidates, key=lambda c: c.final_score, reverse=True)
    _log(log, f"{len(candidates)} possible moments, best first:")
    for c in leaderboard[:12]:
        score = f"{c.quality:3.0f}" if c.quality >= 0 else "  -"
        _log(log, f"  {score}  {c.start/60:5.1f}m  {(c.end - c.start):4.0f}s  "
                  f"{(c.hook_title or c.social_title or c.reason or '')[:70]}")
    selected = choose_top(candidates, count, min_score)
    if not selected:
        selected = choose_top(candidates, 1)
        if selected:
            _log(log, f"No moment reached the minimum score of {int(min_score or 0)}; keeping the single best one so "
                      "you can judge it. Lower 'Min score' to get more clips.")
    if not selected:
        raise RuntimeError("No clips were selected.")
    _log(log, f"Kept the best {len(selected)} of {len(candidates)} possible moments.")
    _log(log, "AI selection is based on the transcript and sampled frames; always review clips before publishing.")

    # Per-clip frame traits (co-op split-screen, letterbox bars) from the 1 fps analysis frames.
    for c in selected:
        paths = [frame_dir / f"frame_{t:06d}.jpg" for t in range(int(c.start), int(math.ceil(c.end)))]
        try:
            traits = clip_style.clip_frame_traits([p for p in paths if p.exists()])
            c.split_screen = traits["split_screen"]
            c.content_band = tuple(traits["band"])
        except Exception as exc:  # noqa: BLE001 - styling hints are optional
            _log(log, f"Could not inspect clip frames for layout hints: {exc}")
    frame_temp_obj.cleanup()

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
    with tempfile.TemporaryDirectory(prefix="clipforge_titles_") as title_dir:
        for n, candidate in enumerate(selected, 1):
            _check_cancel(cancel_event)
            name = f"clip_{n:02d}_{int(candidate.start):06d}s"
            mp4 = job_dir / f"{name}.mp4"
            srt = job_dir / f"{name}.srt"
            ass = job_dir / f"{name}.ass"
            thumbnail = job_dir / f"{name}.jpg"
            post_file = job_dir / f"{name}.post.txt"
            plan = clip_style.plan_layout(render_layout, src_w, src_h, facecam,
                                          split_screen=candidate.split_screen, band=tuple(candidate.content_band))
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
            title_png = Path(title_dir) / f"{name}.png"
            title_size = None
            if burn_title and candidate.hook_title:
                try:
                    title_size = clip_style.render_title_card(candidate.hook_title, title_png)
                except Exception as exc:  # noqa: BLE001 - a title is never worth failing the clip
                    _log(log, f"Could not render the hook title card: {exc}")
            render_vertical(video, candidate, mp4, nvenc_active, log, render_layout,
                            ass_path=burn_ass, normalize_audio=normalize_audio, plan=plan,
                            title_png=title_png if title_size else None)
            create_thumbnail(video, candidate, thumbnail, rendered=mp4)
            write_post_file(post_file, candidate)
            manifest_row = {
                **asdict(candidate),
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
                "posted": False,
            }
            manifest.append(manifest_row)
            (job_dir / f"{name}.metadata.json").write_text(
                json.dumps(manifest_row, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            pct = n / len(selected) * 100
            _emit(progress, "Rendering clips", pct, 88 + pct * 0.12,
                  f"Rendered {n}/{len(selected)} clips")

    (job_dir / "clips.json").write_text(json.dumps({
        "app_version": APP_VERSION,
        "source": str(video),
        "source_input": source,
        "creator": creator,
        "language": language,
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
        "candidates_considered": len(candidates),
        "clips": manifest
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    _emit(progress, "Complete", 100, 100, f"Finished: {len(selected)} clips exported")
    _log(log, f"Finished. Output folder: {job_dir}")
    return job_dir
