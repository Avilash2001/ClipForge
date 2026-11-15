"""Remote sources (YouTube / Twitch / Kick through yt-dlp), downloaded the way a human editor would.

Long streams are not downloaded in full up front:

1. the **audio** track only (for Whisper, loudness and the moment picker), plus a small
   **240p analysis proxy** (for motion, contact sheets and the edge check) - a few hundred MB
   instead of several GB for a multi-hour stream;
2. once the clips are chosen, only those **sections** are downloaded in full quality
   (``download_ranges`` with a safety margin);
3. each section is aligned to the stream timeline by cross-correlating its audio with the
   full audio track, so keyframe-aligned section starts never shift a clip.

If a section download fails, the caller falls back to one full download.
"""
from __future__ import annotations

import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

AUDIO_FORMAT = "ba[acodec^=mp4a]/ba/b"
PROXY_FORMAT = "bv*[height<=240][vcodec^=avc1]/bv*[height<=240]/wv*[height>=144]/wv*"
FULL_FORMAT = "bv*+ba/b"
SECTION_MARGIN = 12.0   # seconds downloaded on each side of a clip (keyframes, trims, fades)

Log = Optional[Callable[[str], None]]


class SignInRequired(RuntimeError):
    """YouTube refused the anonymous request; browser cookies are needed."""


@dataclass
class Section:
    path: Path
    start: float         # requested stream start (seconds)
    end: float           # requested stream end
    offset: float = 0.0  # stream time of the file's t=0, measured by audio alignment
    confidence: float = 0.0
    length: float = 0.0  # media duration of the downloaded file

    def covers(self, start: float, end: float) -> bool:
        length = self.length or (self.end - self.start)
        return self.offset - 0.05 <= start and end <= self.offset + length + 0.05


class _Logger:
    def __init__(self, log: Log, prefix: str = "yt-dlp"):
        self.log, self.prefix = log, prefix

    def debug(self, msg):
        pass

    def warning(self, msg):
        if self.log and "ffmpeg" not in str(msg).lower():
            self.log(f"{self.prefix} warning: {msg}")

    def error(self, msg):
        if self.log:
            self.log(f"{self.prefix} error: {msg}")


def _base_opts(log: Log, cookies_browser: Optional[str]) -> dict:
    opts = {
        "noplaylist": True,
        "windowsfilenames": True,
        "quiet": True,
        "no_warnings": False,
        "logger": _Logger(log),
        "overwrites": False,
        "retries": 5,
        "fragment_retries": 10,
    }
    if cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser.lower(), None, None, None)
    return opts


def explain_download_error(exc: Exception, cookies_browser: Optional[str]) -> Exception:
    lowered = str(exc).lower()
    if "sign in to confirm" in lowered or "use --cookies-from-browser" in lowered or \
            "cookies for the authentication" in lowered or "login required" in lowered:
        return SignInRequired(
            "YouTube blocked the anonymous download. Set 'YouTube login browser' to the browser where you are "
            "signed into YouTube, fully exit that browser, and retry. Do not send anyone your browser cookies.")
    if "could not copy" in lowered and "cookie" in lowered:
        return RuntimeError(f"Could not read {cookies_browser or 'browser'} cookies. Fully exit that browser "
                            f"including background processes, then retry. Original error: {exc}")
    return exc if isinstance(exc, Exception) else RuntimeError(str(exc))


def probe(url: str, cookies_browser: Optional[str] = None, log: Log = None) -> dict:
    """Metadata only: id, title, channel, duration, was_live, available formats."""
    import yt_dlp

    try:
        with yt_dlp.YoutubeDL(_base_opts(log, cookies_browser)) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # noqa: BLE001
        raise explain_download_error(exc, cookies_browser) from exc
    if not isinstance(info, dict):
        raise RuntimeError("yt-dlp returned no metadata for this URL.")
    if info.get("is_live") or info.get("live_status") == "is_live":
        raise RuntimeError("This stream is still live. Wait until it has ended and the replay is available.")
    return info


def best_video_size(info: dict) -> tuple[int, int]:
    """Width/height of the best video format yt-dlp will pick for sections."""
    best = (0, 0)
    for fmt in info.get("formats") or []:
        if (fmt.get("vcodec") or "none") == "none":
            continue
        w, h = int(fmt.get("width") or 0), int(fmt.get("height") or 0)
        if h > best[1]:
            best = (w, h)
    return best if best[1] else (1920, 1080)


def _download(url: str, fmt: str, outtmpl: str, *, cookies_browser: Optional[str], log: Log,
              on_progress: Optional[Callable[[float, str], None]] = None,
              cancel_event: Optional[threading.Event] = None, extra: Optional[dict] = None) -> Path:
    import yt_dlp

    files: list[str] = []
    last = [0.0]

    def hook(d: dict) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Job cancelled by user.")
        if d.get("status") == "downloading" and on_progress:
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            done = d.get("downloaded_bytes", 0)
            now = time.monotonic()
            if now - last[0] >= 0.3:
                last[0] = now
                pct = done / total * 100 if total else 0.0
                speed = d.get("speed")
                extra_text = f" · {speed / 1048576:.1f} MB/s" if speed else ""
                on_progress(pct, extra_text)
        elif d.get("status") == "finished" and d.get("filename"):
            files.append(d["filename"])

    def post_hook(filename: str) -> None:
        files.append(filename)

    opts = _base_opts(log, cookies_browser)
    opts.update({"format": fmt, "outtmpl": outtmpl, "progress_hooks": [hook],
                 "post_hooks": [post_hook]})
    if extra:
        opts.update(extra)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            final = info.get("requested_downloads", [{}])[-1].get("filepath") if isinstance(info, dict) else None
    except Exception as exc:  # noqa: BLE001
        if cancel_event is not None and cancel_event.is_set():
            raise
        raise explain_download_error(exc, cookies_browser) from exc
    for candidate in [final, *reversed(files)]:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise RuntimeError(f"yt-dlp finished but the downloaded file was not found ({outtmpl}).")


def download_audio(url: str, dest: Path, **kwargs) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    return _download(url, AUDIO_FORMAT, str(dest / "%(title).100B [%(id)s].audio.%(ext)s"), **kwargs)


def download_proxy(url: str, dest: Path, **kwargs) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    return _download(url, PROXY_FORMAT, str(dest / "%(title).100B [%(id)s].proxy.%(ext)s"), **kwargs)


def download_full(url: str, dest: Path, **kwargs) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    return _download(url, FULL_FORMAT, str(dest / "%(title).120B [%(id)s].%(ext)s"),
                     extra={"merge_output_format": "mp4"}, **kwargs)


def merge_ranges(ranges: list[tuple[float, float]], duration: float, margin: float = SECTION_MARGIN,
                 join_gap: float = 30.0) -> list[tuple[float, float]]:
    """Pad each clip range by ``margin`` and merge ranges that are close, so nearby clips share a section."""
    padded = sorted((max(0.0, a - margin), min(duration, b + margin)) for a, b in ranges if b > a)
    merged: list[list[float]] = []
    for a, b in padded:
        if merged and a <= merged[-1][1] + join_gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(round(a, 2), round(b, 2)) for a, b in merged]


def download_section(url: str, start: float, end: float, dest: Path, **kwargs) -> Path:
    from yt_dlp.utils import download_range_func

    dest.mkdir(parents=True, exist_ok=True)
    tag = f"{int(start):06d}-{int(end):06d}"
    return _download(url, FULL_FORMAT, str(dest / f"%(id)s.section_{tag}.%(ext)s"),
                     extra={"download_ranges": download_range_func(None, [(start, end)]),
                            "force_keyframes_at_cuts": False, "merge_output_format": "mp4"},
                     **kwargs)


# ---------------------------------------------------------------------------
# Section alignment
# ---------------------------------------------------------------------------

def read_wav_slice(path: Path, start: float, seconds: float):
    """Samples [start, start+seconds) of a 16-bit mono WAV as float32, without loading the whole file."""
    import numpy as np

    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
            raise RuntimeError("Expected a 16-bit mono WAV.")
        rate = wf.getframerate()
        first = max(0, int(start * rate))
        if first >= wf.getnframes():
            return np.zeros(0, dtype=np.float32), rate, first / rate
        wf.setpos(first)
        data = wf.readframes(int(seconds * rate))
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0, rate, first / rate


def _decode_audio(path: Path, seconds: float, rate: int, runner: Callable) -> "object":
    import numpy as np

    result = runner(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-t", f"{seconds:.2f}",
                     "-vn", "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"],
                    check=True, capture_output=True)
    return np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0


def find_offset(reference, rate: int, probe_audio, approx_start: float, search: float = 25.0) -> tuple[float, float]:
    """Where in ``reference`` (the full stream audio) does ``probe_audio`` start?

    Returns (stream seconds, normalised correlation 0..1). Uses an FFT cross-correlation on
    a 4 kHz envelope-free signal: plain audio correlates sharply even after re-encoding.
    """
    import numpy as np

    step = 4  # 16 kHz -> 4 kHz is plenty for alignment and 4x cheaper
    ref = reference[::step]
    sig = probe_audio[::step]
    r = rate // step
    lo = max(0, int((approx_start - search) * r))
    hi = min(len(ref), int((approx_start + search) * r) + len(sig))
    window = ref[lo:hi]
    if len(sig) < r or len(window) <= len(sig):
        return approx_start, 0.0
    sig = sig - sig.mean()
    window = window - window.mean()
    n = 1 << int(np.ceil(np.log2(len(window) + len(sig))))
    corr = np.fft.irfft(np.fft.rfft(window, n) * np.conj(np.fft.rfft(sig, n)), n)[: len(window) - len(sig) + 1]
    # Normalise by the energy of each reference window so loud passages do not win by volume.
    csum = np.concatenate([[0.0], np.cumsum(window.astype(np.float64) ** 2)])
    energy = np.sqrt((csum[len(sig):] - csum[:-len(sig)]) * float(np.sum(sig.astype(np.float64) ** 2))) + 1e-9
    score = corr / energy[: len(corr)]
    best = int(np.argmax(score))
    return (lo + best) / r, float(max(0.0, min(1.0, score[best])))


def media_length(path: Path, runner: Callable) -> float:
    out = runner(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                  "default=noprint_wrappers=1:nokey=1", str(path)],
                 capture_output=True, text=True, check=False).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def align_section(section: Section, wav_path: Path, runner: Callable, probe_seconds: float = 20.0,
                  search: float = 25.0) -> Section:
    """Measure where the section file's t=0 sits on the stream timeline (audio cross-correlation)."""
    rate = 16000
    audio = _decode_audio(section.path, probe_seconds, rate, runner)
    lo = max(0.0, section.start - search)
    reference, rate, ref_start = read_wav_slice(wav_path, lo, 2 * search + probe_seconds + 1)
    offset, confidence = find_offset(reference, rate, audio, section.start - ref_start, search=search)
    section.offset, section.confidence = round(ref_start + offset, 3), round(confidence, 3)
    section.length = media_length(section.path, runner) or (section.end - section.start)
    return section


def section_for(sections: list[Section], start: float, end: float) -> Optional[Section]:
    """The downloaded section that contains the whole clip, if any."""
    for section in sections:
        if section.covers(start, end):
            return section
    return None
