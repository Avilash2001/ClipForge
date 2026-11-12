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

from paths import INPUT_DIR, OUTPUT_DIR

DEFAULT_INPUT_DIR = INPUT_DIR
DEFAULT_OUTPUT_DIR = OUTPUT_DIR
OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"
APP_VERSION = "1.3.2"
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
                     cookies_browser: Optional[str] = None) -> Path:
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


def audio_energy_by_second(wav_path: Path) -> list[float]:
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
            "words": [
                {"start": float(w.start), "end": float(w.end), "word": w.word}
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


def _extract_json_object(text: str) -> dict:
    """Parse JSON directly, or recover the first JSON object embedded in model text."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Ollama returned an empty response.")
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for pos, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[pos:])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                continue
    raise ValueError("Ollama response did not contain a valid JSON object.")


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
                     transcript_usable: bool) -> list[Candidate]:
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
        if transcript_usable:
            baseline = 0.25 * audio + 0.25 * visual + 0.50 * ts
        else:
            baseline = 0.35 * audio + 0.65 * visual
        # Strongly demote near-static sections such as countdowns and menus.
        if visual < 0.06:
            baseline *= 0.35
        candidates.append(Candidate(
            index=idx, start=float(start), end=float(end),
            audio_score=audio, transcript=text[:900],
            transcript_score=ts, visual_score=visual,
            final_score=baseline, context_before=context_before[:240], context_after=context_after[:240]
        ))
    return candidates


def _rank_batch(candidates: list[Candidate], model: str, game_profile: str,
                image_sheets: list[str], transcript_usable: bool) -> list[Candidate]:
    rows = []
    for c in candidates:
        rows.append({
            "index": c.index,
            "time_range": f"{c.start:.1f}-{c.end:.1f}s",
            "audio_activity": round(c.audio_score, 3),
            "visual_motion": round(c.visual_score, 3),
            "spoken_context_before": c.context_before[:220] if transcript_usable else "",
            "spoken_context_during": c.transcript[:500] if transcript_usable else "",
            "spoken_context_after": c.context_after[:220] if transcript_usable else "",
        })
    prompt = (
        f"You are a careful gaming-video editor reviewing {game_profile} footage. "
        "Each attached image is a 3x3 contact sheet for one candidate; its nine frames are in chronological order. "
        "Use the visual sequence AND the speech context together. Do not infer a kill, win, clutch or funny event "
        "unless visible frames or transcript support it. Judge whether the clip has a coherent mini-story: "
        "hook/setup, action/escalation, payoff, reaction/aftermath. Pure gameplay without speech can still be good, "
        "but score narrative coherence honestly. A clip that starts after the action or ends before its payoff gets a lower story score. "
        "Reject stream starting soon screens, countdowns, menus, lobbies, character selection, loading screens, settings, "
        "long static scenes and ordinary scoreboards. Prefer active combat, close calls, objective changes, funny failures, "
        "impressive plays, victory/defeat, or a notable reaction. Do not invent events. "
        "Return ONLY JSON matching the provided schema, with a result for every index. Scores are 0..1. "
        "screen_type must be active_gameplay, menu_lobby, starting_screen, loading_screen, scoreboard, cutscene, or other. "
        "Rejected screen types must have score and story_score <= 0.05. Use factual reasons. social_title must be short and honest; "
        "social_caption one sentence; hashtags 3-5 items without #. Candidate details: " + json.dumps(rows, ensure_ascii=False)
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
                        "reason": {"type": "string"}, "social_title": {"type": "string"},
                        "social_caption": {"type": "string"},
                        "hashtags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["index", "score", "story_score", "screen_type", "reason"],
                },
            }
        },
        "required": ["ranked"],
    }
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt, "images": image_sheets}],
            "stream": False,
            "format": schema,
            "think": False,
            "options": {"temperature": 0.05, "num_ctx": 12288, "num_predict": 1800},
        },
        timeout=1800,
    )
    response.raise_for_status()
    message = response.json().get("message", {})
    raw_content = message.get("content", "") or ""
    raw_thinking = message.get("thinking", "") or ""
    data = None
    errors = []
    for raw in (raw_content, raw_thinking):
        try:
            data = _extract_json_object(raw)
            break
        except Exception as exc:
            errors.append(str(exc))
    if data is None:
        raise ValueError("Ollama returned no valid structured JSON: " + " / ".join(errors))
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
            candidate.event_type = str(item.get("event_type", "unknown"))[:80]
            candidate.hook = str(item.get("hook", ""))[:160]
            candidate.setup = str(item.get("setup", ""))[:180]
            candidate.payoff = str(item.get("payoff", ""))[:180]
            candidate.reason = str(item.get("reason", "Visual AI reviewed the clip"))[:180]
            candidate.social_title = str(item.get("social_title", ""))[:90]
            candidate.social_caption = str(item.get("social_caption", ""))[:220]
            raw_tags = item.get("hashtags", [])
            candidate.hashtags = [str(t).lstrip("#")[:40] for t in raw_tags[:5]] if isinstance(raw_tags, list) else []
            candidate.story_score = story_score
            if transcript_usable:
                candidate.final_score = (0.34 * score + 0.30 * story_score +
                                         0.16 * candidate.visual_score + 0.08 * candidate.audio_score +
                                         0.12 * candidate.transcript_score)
            else:
                candidate.final_score = (0.43 * score + 0.38 * story_score +
                                         0.13 * candidate.visual_score + 0.06 * candidate.audio_score)
            if rejected:
                candidate.final_score = min(candidate.final_score, 0.02)
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
                transcript_usable: bool = False) -> list[Candidate]:
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
            ranked = _rank_batch(batch, model, game_profile, image_sheets, transcript_usable)
        except Exception as first_error:
            _log(log, f"Vision batch retry with smaller candidate groups: {first_error}")
            ranked = []
            for sub_start in range(0, len(batch), 2):
                sub = batch[sub_start:sub_start + 2]
                sub_images = [_make_contact_sheet(frame_dir, c, cv2) for c in sub]
                ranked.extend(_rank_batch(sub, model, game_profile, sub_images, transcript_usable))
        ranked_by_index.update({c.index: c for c in ranked})
        pct = batch_no / max(1, len(batches)) * 100
        _emit(progress, "Visual story review", pct, 80 + pct * 0.08,
              f"Reviewed {min(batch_no * batch_size, len(shortlist))}/{len(shortlist)} gameplay candidates")
    return [ranked_by_index.get(c.index, c) for c in candidates]


def choose_top(candidates: list[Candidate], count: int) -> list[Candidate]:
    ranked = sorted(candidates, key=lambda c: c.final_score, reverse=True)
    chosen: list[Candidate] = []
    for candidate in ranked:
        if candidate.screen_type in {"menu_lobby", "starting_screen", "loading_screen", "scoreboard"}:
            continue
        if all(candidate.end <= old.start or candidate.start >= old.end for old in chosen):
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


def _ass_timestamp(seconds: float) -> str:
    centiseconds = max(0, int(round(seconds * 100)))
    hours, centiseconds = divmod(centiseconds, 360000)
    minutes, centiseconds = divmod(centiseconds, 6000)
    secs, centiseconds = divmod(centiseconds, 100)
    return f"{hours}:{minutes:02}:{secs:02}.{centiseconds:02}"


def _ass_escape(text: str) -> str:
    return (str(text).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
            .replace("\n", " ").replace("\r", " "))


def write_clip_ass(rows: list[dict], start: float, end: float, path: Path) -> bool:
    """Write word-timed karaoke captions; punctuation-only ASR output is excluded."""
    events = []
    for seg in rows:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        word_rows = [w for w in seg.get("words", []) if w.get("word") and any(ch.isalnum() for ch in w["word"])]
        if word_rows:
            for offset in range(0, len(word_rows), 7):
                group = word_rows[offset:offset + 7]
                begin = max(0.0, float(group[0]["start"]) - start)
                finish = min(end - start, float(group[-1]["end"]) - start)
                if finish <= begin:
                    continue
                karaoke = []
                for word in group:
                    duration_cs = max(1, int(round((float(word["end"]) - float(word["start"])) * 100)))
                    karaoke.append(f"{{\\k{duration_cs}}}{_ass_escape(word['word'].strip())}")
                events.append((begin, finish, " ".join(karaoke)))
        else:
            text = str(seg.get("text", "")).strip()
            if any(ch.isalnum() for ch in text):
                events.append((max(0.0, seg["start"] - start), min(end - start, seg["end"] - start), _ass_escape(text)))
    if not events:
        return False
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,62,&H0000D7FF,&H00FFFFFF,&H00000000,&H80000000,1,0,1,4,2,2,72,72,300,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with path.open("w", encoding="utf-8-sig") as f:
        f.write(header)
        for begin, finish, text in events:
            f.write(f"Dialogue: 0,{_ass_timestamp(begin)},{_ass_timestamp(finish)},Default,,0,0,0,,{text}\n")
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


def render_vertical(video: Path, candidate: Candidate, output: Path, use_nvenc: bool,
                    log: Optional[LogCallback] = None,
                    layout: str = "Fit + blurred background",
                    ass_path: Optional[Path] = None,
                    normalize_audio: bool = True) -> None:
    if layout == "Fill (center crop)":
        video_filter = "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1[v]"
    else:
        video_filter = (
            "[0:v]split=2[bg][fg];"
            "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=24:12[blur];"
            "[fg]scale=1080:1920:force_original_aspect_ratio=decrease[game];"
            "[blur][game]overlay=(W-w)/2:(H-h)/2,setsar=1[v]"
        )
    if ass_path and ass_path.exists():
        video_filter = video_filter.replace("[v]", f",ass=filename='{_ffmpeg_filter_path(ass_path)}'[v]")
    common = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(candidate.start), "-i", str(video),
        "-t", str(candidate.end - candidate.start), "-filter_complex", video_filter,
        "-map", "[v]", "-map", "0:a?", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", "-pix_fmt", "yuv420p",
    ]
    if normalize_audio:
        common.extend(["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"])
    if use_nvenc:
        cmd = common + ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "21", str(output)]
        try:
            _run_process(cmd, check=True, capture_output=True, text=True)
            return
        except subprocess.CalledProcessError as exc:
            if exc.returncode in (3221225786, -1073741510):
                raise _subprocess_failure(exc, "Video rendering") from exc
            _log(log, "NVENC failed for this clip; retrying with CPU encoding.")
            if exc.stderr:
                _log(log, exc.stderr[-1200:])
    cmd = common + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", str(output)]
    try:
        _run_process(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise _subprocess_failure(exc, "Video rendering") from exc


def create_thumbnail(video: Path, candidate: Candidate, output: Path) -> bool:
    midpoint = candidate.start + (candidate.end - candidate.start) * 0.55
    result = _run_process([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", str(midpoint),
        "-i", str(video), "-frames:v", "1", "-q:v", "2", str(output)
    ], check=False, capture_output=True, text=True)
    return result.returncode == 0 and output.exists()


def process_job(source: str, output_root: Path, input_dir: Path, *, count: int = 10,
                clip_length: int = 35, stride: int = 15, whisper_model: str = "small",
                whisper_device: str = "cpu", ollama_model: str = DEFAULT_OLLAMA_MODEL,
                use_ollama: bool = True, use_nvenc: bool = True,
                game_profile: str = "Auto / mixed games",
                cookies_browser: Optional[str] = None,
                language_mode: str = "Mixed (English + Bengali + Hindi)",
                render_layout: str = "Fit + blurred background",
                burn_captions: bool = True,
                normalize_audio: bool = True,
                progress: Optional[ProgressCallback] = None,
                log: Optional[LogCallback] = None,
                cancel_event: Optional[threading.Event] = None) -> Path:
    check_tools()
    if count < 1 or clip_length < 10 or stride < 5:
        raise ValueError("Clip count must be >=1, duration >=10 sec and scan interval >=5 sec.")
    _check_cancel(cancel_event)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    is_url = source.strip().lower().startswith(("https://", "http://"))
    if is_url:
        video = download_youtube(source.strip(), input_dir, progress, cancel_event, log, cookies_browser)
    else:
        video = Path(source).expanduser().resolve()
        if not video.is_file() or video.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError("Select a valid local video file (MP4, MKV, WebM, MOV or AVI).")
        _emit(progress, "Input", 100, 10, f"Using local file: {video.name}")
    _check_cancel(cancel_event)
    _log(log, f"Video: {video}")

    _emit(progress, "Inspecting video", 0, 10, "Reading video metadata...")
    duration = duration_seconds(video)
    _log(log, f"Duration: {duration/60:.1f} minutes")
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
        _emit(progress, "Audio analysis", 65, 13, "Measuring audio activity...")
        energies = audio_energy_by_second(wav_path)
        _emit(progress, "Audio analysis", 100, 15, "Audio analysis complete")

        _log(log, f"Transcribing with faster-whisper ({whisper_model}, {whisper_device}); {language_mode}...")
        transcript, language = transcribe(
            wav_path, whisper_model, whisper_device, duration,
            progress, cancel_event, log, language_mode
        )
        quality = transcript_quality(transcript)
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
                    channel_rows, channel_language = transcribe(
                        channel_path, whisper_model, whisper_device, duration,
                        progress, cancel_event, log, language_mode
                    )
                    channel_quality = transcript_quality(channel_rows)
                    channel_candidates.append((channel_rows, channel_language, channel_quality, channel_path))
                    _log(log, f"{channel} channel: {channel_quality['word_count']} word tokens.")
                    if channel_quality["usable"]:
                        break
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
            retry_rows, retry_language = transcribe(
                selected_audio_path, "medium", whisper_device, duration,
                progress, cancel_event, log, language_mode
            )
            retry_quality = transcript_quality(retry_rows)
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

    frame_temp_obj = tempfile.TemporaryDirectory(prefix="clipforge_frames_")
    frame_dir = Path(frame_temp_obj.name)
    motion = visual_motion_by_second(video, duration, frame_dir, progress, cancel_event, log)
    candidates = build_candidates(duration, energies, transcript, motion,
                                  clip_length, stride, transcript_usable)
    _log(log, f"Generated {len(candidates)} candidate windows.")
    _emit(progress, "Preparing candidates", 100, 80, f"Generated {len(candidates)} candidates")
    if not candidates:
        raise RuntimeError("No candidate clips could be generated from this video.")

    if use_ollama:
        try:
            _log(log, f"Ranking candidates with local visual AI ({ollama_model})...")
            candidates = ollama_rank(
                candidates, ollama_model, frame_dir, progress, cancel_event, log,
                game_profile, transcript_usable
            )
        except JobCancelled:
            raise
        except Exception as exc:
            _log(log, f"Local visual AI ranking unavailable; using visual/audio fallback. Details: {exc}")
            for c in candidates:
                c.final_score = 0.35 * c.audio_score + 0.65 * c.visual_score
                if c.visual_score < 0.06:
                    c.final_score *= 0.25
            _emit(progress, "AI ranking", 100, 88, "Using visual/audio fallback ranking")
    else:
        _log(log, "Visual AI ranking disabled; using local visual/audio scores.")
        for c in candidates:
            c.final_score = 0.35 * c.audio_score + 0.65 * c.visual_score
            if c.visual_score < 0.06:
                c.final_score *= 0.25
        _emit(progress, "AI ranking", 100, 88, "Local visual/audio ranking complete")

    selected = choose_top(candidates, count)
    if not selected:
        raise RuntimeError("No clips were selected.")
    _log(log, f"Selected {len(selected)} non-overlapping clips.")
    _log(log, "Visual AI selection is based on sampled frames; always review clips before publishing.")
    frame_temp_obj.cleanup()

    try:
        encoders = _run_process(["ffmpeg", "-hide_banner", "-encoders"],
                                  capture_output=True, text=True, check=False).stdout
        nvenc_available = "h264_nvenc" in encoders
    except Exception:
        nvenc_available = False
    nvenc_active = bool(use_nvenc and nvenc_available)
    _log(log, "NVIDIA NVENC encoding enabled." if nvenc_active else "Using CPU video encoding.")

    manifest = []
    for n, candidate in enumerate(selected, 1):
        _check_cancel(cancel_event)
        name = f"clip_{n:02d}_{int(candidate.start):06d}s"
        mp4 = job_dir / f"{name}.mp4"
        srt = job_dir / f"{name}.srt"
        ass = job_dir / f"{name}.ass"
        thumbnail = job_dir / f"{name}.jpg"
        _log(log, f"Rendering clip {n}/{len(selected)}: {candidate.start:.1f}s – {candidate.end:.1f}s")
        subtitle_written = write_clip_srt(transcript, candidate.start, candidate.end, srt) if transcript_usable else False
        ass_written = write_clip_ass(transcript, candidate.start, candidate.end, ass) if transcript_usable and burn_captions else False
        if not subtitle_written:
            srt.unlink(missing_ok=True)
        if not ass_written:
            ass.unlink(missing_ok=True)
        burn_ass = ass if ass_written and ffmpeg_has_ass_filter() else None
        if ass_written and burn_ass is None:
            _log(log, "FFmpeg lacks ASS/libass support; retaining caption sidecars without burning them in.")
        render_vertical(video, candidate, mp4, nvenc_active, log, render_layout,
                        ass_path=burn_ass, normalize_audio=normalize_audio)
        create_thumbnail(video, candidate, thumbnail)
        manifest_row = {
            **asdict(candidate),
            "file": mp4.name,
            "subtitle": srt.name if subtitle_written else None,
            "ass_subtitle": ass.name if ass_written else None,
            "thumbnail": thumbnail.name if thumbnail.exists() else None,
            "title_suggestion": candidate.social_title or f"Gaming highlight #{n}",
            "social_caption": candidate.social_caption,
            "hashtags": candidate.hashtags or [],
            "captions_burned": bool(burn_ass),
        }
        manifest.append(manifest_row)
        (job_dir / f"{name}.metadata.json").write_text(
            json.dumps(manifest_row, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pct = n / len(selected) * 100
        _emit(progress, "Rendering clips", pct, 88 + pct * 0.12,
              f"Rendered {n}/{len(selected)} clips")

    (job_dir / "clips.json").write_text(json.dumps({
        "source": str(video),
        "source_input": source,
        "language": language,
        "transcript_quality": quality,
        "transcript_usable": transcript_usable,
        "game_profile": game_profile,
        "render_layout": render_layout,
        "burn_captions": burn_captions,
        "normalize_audio": normalize_audio,
        "clips": manifest
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    _emit(progress, "Complete", 100, 100, f"Finished: {len(selected)} clips exported")
    _log(log, f"Finished. Output folder: {job_dir}")
    return job_dir

