"""Speech recognition: faster-whisper on an NVIDIA GPU when it works, the CPU otherwise.

* ``cuda_status()`` makes the CUDA 12 / cuDNN 9 DLLs that CTranslate2 needs findable
  (a ``cuda-libs`` folder, the pip ``nvidia-cublas-cu12`` / ``nvidia-cudnn-cu12`` wheels,
  or a CUDA toolkit install) and checks that they load.
* ``resolve_settings()`` turns the "Auto" choices into a concrete model/device/compute type:
  large-v3 in float16 on the GPU (batched decoding), ``small`` in int8 on the CPU.
* ``transcribe()`` runs Whisper and falls back automatically (GPU float16 -> GPU int8 on
  out-of-memory -> CPU) if anything on the GPU fails.
* Mixed English/Bengali/Hindi mode limits Whisper's language detection to those three
  languages and makes the decision per speech chunk. Unrestricted, Whisper labels Hindi as
  Urdu, Nepali or Marathi (or even Chinese/Japanese) and writes it in the wrong script.
"""
from __future__ import annotations

import ctypes
import gc
import glob
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

MIXED = "Mixed (English + Bengali + Hindi)"
LANGUAGE_MODES = [MIXED, "English", "Bengali", "Hindi"]
LANGUAGE_CODES = {"English": "en", "Bengali": "bn", "Hindi": "hi"}
MIXED_LANGUAGES = ("en", "hi", "bn")

AUTO = "Auto"
GPU_MODEL = "large-v3"          # turbo collapsed on Hindi speech in testing (CHANGELOG v1.7)
CPU_MODEL = "small"             # large models are far too slow on a CPU for multi-hour streams
WHISPER_MODELS = [AUTO, "large-v3", "large-v3-turbo", "medium", "small", "base", "tiny"]
WHISPER_DEVICES = [AUTO, "cuda", "cpu"]

# CTranslate2 4.x on Windows is built against CUDA 12 and cuDNN 9.
WINDOWS_CUDA_DLLS = ["cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll"]
LINUX_CUDA_LIBS = ["libcublas.so.12", "libcublasLt.so.12", "libcudnn.so.9"]

_added_dirs: set[str] = set()
_dll_handles: list = []   # keep preloaded libraries alive


@dataclass(frozen=True)
class WhisperSettings:
    model: str
    device: str
    compute_type: str
    model_auto: bool = False   # chosen by Auto (so a CPU fallback may pick a CPU-sized model)

    def label(self) -> str:
        return f"{self.model} on {self.device.upper() if self.device == 'cuda' else 'CPU'} ({self.compute_type})"


def _project_dirs() -> list[Path]:
    here = Path(__file__).resolve().parent
    dirs = [here]
    try:
        from paths import DATA_DIR
        dirs.append(Path(DATA_DIR))
    except Exception:  # noqa: BLE001 - paths is optional for this module
        pass
    if getattr(sys, "frozen", False):
        dirs.append(Path(sys.executable).resolve().parent)
    return dirs


def cuda_library_dirs() -> list[Path]:
    """Folders that may hold the CUDA 12 / cuDNN 9 runtime libraries, best first."""
    found: list[Path] = []
    for base in _project_dirs():
        found.append(base / "cuda-libs")
    # pip wheels: nvidia-cublas-cu12 / nvidia-cudnn-cu12 / nvidia-cuda-runtime-cu12 install into
    # site-packages/nvidia/<lib>/bin (Windows) or .../lib (Linux).
    for entry in sys.path:
        if not entry:
            continue
        nvidia = Path(entry) / "nvidia"
        if nvidia.is_dir():
            for sub in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
                found.extend([nvidia / sub / "bin", nvidia / sub / "lib"])
    for var in sorted(k for k in os.environ if k == "CUDA_PATH" or k.startswith("CUDA_PATH_V12")):
        found.append(Path(os.environ[var]) / "bin")
    if os.name == "nt":
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        for pattern in (r"NVIDIA\CUDNN\v9*\bin\12*", r"NVIDIA\CUDNN\v9*\bin"):
            found.extend(Path(p) for p in glob.glob(os.path.join(program_files, pattern)))
    unique: list[Path] = []
    for path in found:
        if path.is_dir() and path not in unique:
            unique.append(path)
    return unique


def prepare_cuda_libraries() -> list[str]:
    """Make CUDA/cuDNN libraries findable by CTranslate2. Safe to call repeatedly."""
    names = WINDOWS_CUDA_DLLS if os.name == "nt" else LINUX_CUDA_LIBS
    useful = []
    for folder in cuda_library_dirs():
        try:
            if not any((folder / name).exists() or list(folder.glob(name.split(".")[0] + "*"))
                       for name in names):
                continue
        except OSError:
            continue
        key = str(folder)
        useful.append(key)
        if key in _added_dirs:
            continue
        _added_dirs.add(key)
        if os.name == "nt":
            try:
                os.add_dll_directory(key)
            except (OSError, AttributeError):
                pass
            # CTranslate2 loads cuBLAS lazily with the classic search order, which uses PATH.
            os.environ["PATH"] = key + os.pathsep + os.environ.get("PATH", "")
    if os.name != "nt":
        # dlopen() ignores changes to LD_LIBRARY_PATH after start-up; preload by absolute path.
        for name in LINUX_CUDA_LIBS:
            for folder in useful:
                matches = sorted(Path(folder).glob(name + "*"))
                if matches:
                    try:
                        _dll_handles.append(ctypes.CDLL(str(matches[0]), mode=ctypes.RTLD_GLOBAL))
                    except OSError:
                        pass
                    break
    return useful


def missing_cuda_libraries() -> list[str]:
    """Names of required CUDA/cuDNN libraries that cannot be loaded right now."""
    missing = []
    for name in (WINDOWS_CUDA_DLLS if os.name == "nt" else LINUX_CUDA_LIBS):
        try:
            handle = ctypes.WinDLL(name) if os.name == "nt" else ctypes.CDLL(name)
            _dll_handles.append(handle)
        except OSError:
            missing.append(name)
    return missing


_status_cache: Optional[tuple[bool, str]] = None


def cuda_status(refresh: bool = False) -> tuple[bool, str]:
    """(usable, explanation) for running Whisper on an NVIDIA GPU."""
    global _status_cache
    if _status_cache is not None and not refresh:
        return _status_cache
    if sys.platform == "darwin":
        _status_cache = (False, "CUDA is not available on macOS")
        return _status_cache
    try:
        import ctranslate2
    except Exception as exc:  # noqa: BLE001
        _status_cache = (False, f"CTranslate2 could not be imported: {exc}")
        return _status_cache
    try:
        devices = ctranslate2.get_cuda_device_count()
    except Exception:  # noqa: BLE001
        devices = 0
    if devices < 1:
        _status_cache = (False, "no NVIDIA GPU visible to CTranslate2 (driver missing or no CUDA GPU)")
        return _status_cache
    folders = prepare_cuda_libraries()
    missing = missing_cuda_libraries()
    if missing:
        _status_cache = (False, "CUDA libraries not found: " + ", ".join(missing) +
                         ". Run Setup again (it installs nvidia-cublas-cu12 and nvidia-cudnn-cu12 on NVIDIA PCs) "
                         "or put the CUDA 12 / cuDNN 9 DLLs in the cuda-libs folder.")
        return _status_cache
    try:
        types = ctranslate2.get_supported_compute_types("cuda")
    except Exception:  # noqa: BLE001
        types = set()
    if "float16" not in types and "int8_float16" not in types:
        _status_cache = (False, "this GPU does not support float16 inference")
        return _status_cache
    where = ""
    if folders:
        first = Path(folders[0])
        where = (f" (CUDA libraries from the pip {first.parent.name} wheel)" if first.name in ("bin", "lib")
                 and first.parent.parent.name == "nvidia" else f" (CUDA libraries from {first})")
    _status_cache = (True, f"NVIDIA GPU ready for Whisper{where}")
    return _status_cache


def resolve_settings(model: str = AUTO, device: str = AUTO,
                     log: Optional[Callable[[str], None]] = None) -> WhisperSettings:
    """Turn the GUI/CLI choices into a concrete Whisper configuration."""
    model = (model or AUTO).strip()
    device = (device or AUTO).strip().lower()
    model_auto = model.lower() == AUTO.lower()
    if device in ("auto", "gpu"):
        ok, why = cuda_status()
        device = "cuda" if ok else "cpu"
        if log:
            log(f"Whisper device Auto: {'GPU' if ok else 'CPU'} - {why}.")
    elif device == "cuda":
        ok, why = cuda_status()
        if not ok:
            if log:
                log(f"CUDA requested but not usable ({why}); using the CPU.")
            device = "cpu"
    else:
        device = "cpu"
    if model_auto:
        model = GPU_MODEL if device == "cuda" else CPU_MODEL
    if device == "cuda":
        compute = "float16"
        try:
            import ctranslate2
            if "float16" not in ctranslate2.get_supported_compute_types("cuda"):
                compute = "int8_float16"
        except Exception:  # noqa: BLE001
            pass
    else:
        compute = "int8"
    return WhisperSettings(model, device, compute, model_auto)


def cpu_fallback(settings: WhisperSettings) -> WhisperSettings:
    model = CPU_MODEL if settings.model_auto else settings.model
    return WhisperSettings(model, "cpu", "int8", settings.model_auto)


class _LanguageLimiter:
    """Wraps CTranslate2's Whisper model so language detection only ever picks allowed languages.

    faster-whisper's per-segment ("multilingual") mode takes the top language token for every
    30 s window. Here the allowed languages are moved to the front (re-normalised), so a
    Hindi window that Whisper half-thinks is Urdu is decoded as Hindi instead.
    """

    def __init__(self, inner, allowed: tuple[str, ...]):
        self._inner = inner
        self._allowed = {f"<|{code}|>" for code in allowed}

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def detect_language(self, *args, **kwargs):
        results = self._inner.detect_language(*args, **kwargs)
        return [restrict_language_probs(item, self._allowed) for item in results]


def restrict_language_probs(probs: list[tuple[str, float]], allowed_tokens: set[str]) -> list[tuple[str, float]]:
    """Reorder [(language_token, p), ...] so the best allowed language comes first.

    Allowed probabilities are re-normalised among themselves; if Whisper gave every allowed
    language ~0, the list is returned unchanged.
    """
    allowed = [(tok, p) for tok, p in probs if tok in allowed_tokens]
    total = sum(p for _, p in allowed)
    if not allowed or total <= 1e-6:
        return list(probs)
    others = [(tok, p) for tok, p in probs if tok not in allowed_tokens]
    allowed = sorted(((tok, p / total) for tok, p in allowed), key=lambda item: item[1], reverse=True)
    return allowed + others


# No initial prompt. In testing on the user's streams a prompt (English description, or mixed-script
# examples) made large-v3 loop and drift on Hindi speech, and on unclear audio Whisper sometimes
# "heard" the prompt itself. Batch size for GPU decoding: 8 fits large-v3 float16 in ~6 GB.
GPU_BATCH_SIZE = 8


def whisper_options(language_mode: str, batched: bool = False) -> dict:
    """faster-whisper keyword arguments for a language mode.

    ``batched`` (GPU) uses faster-whisper's BatchedInferencePipeline: speech is cut into VAD
    chunks of up to 30 s, each chunk gets its own language decision (finer than the 30 s
    windows of sequential decoding, which suits code-switching), and decoding is ~3x faster.
    """
    forced = LANGUAGE_CODES.get(language_mode)
    options = {
        "beam_size": 5,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 700, "speech_pad_ms": 300},
        "word_timestamps": True,
        "condition_on_previous_text": False,
        "multilingual": forced is None,
        "language_detection_threshold": 0.35,
        "initial_prompt": None,
    }
    if batched:
        # Split each chunk at Whisper's own timestamps, so segments stay sentence-sized.
        options.update({"without_timestamps": False, "batch_size": GPU_BATCH_SIZE})
        del options["condition_on_previous_text"]
    else:
        options.update({"language_detection_segments": 3,
                        # Skip silent stretches where Whisper invents text (needs word timestamps).
                        "hallucination_silence_threshold": 2.0})
    if forced:
        options["language"] = forced
    return options


def _tokens(text: str) -> list[str]:
    import re

    return [t for t in re.split(r"[^\w']+", str(text or "").lower()) if t]


def is_prompt_echo(text: str, prompt: Optional[str]) -> bool:
    """True when a segment just repeats the initial prompt (Whisper does this on unclear audio)."""
    words = _tokens(text)
    if not prompt or len(words) < 4:
        return False
    vocab = set(_tokens(prompt))
    inside = sum(1 for w in words if w in vocab) / len(words)
    return inside >= 0.8 and " ".join(words) in " ".join(_tokens(prompt))


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda_error_out_of_memory" in text or "cublas_status_alloc_failed" in text


def transcribe(audio_path: Path, settings: WhisperSettings, duration: float, *,
               language_mode: str = MIXED,
               on_progress: Optional[Callable[[float, float, Optional[float]], None]] = None,
               check_cancel: Optional[Callable[[], None]] = None,
               log: Optional[Callable[[str], None]] = None,
               cancelled: tuple[type[BaseException], ...] = ()) -> tuple[list[dict], str, WhisperSettings]:
    """Transcribe a 16 kHz WAV. Returns (raw rows, first detected language, settings actually used).

    GPU failures (missing DLLs, driver errors, out of memory) fall back automatically: first to
    int8_float16 on out-of-memory, then to the CPU.
    """
    attempts = [settings]
    if settings.device == "cuda":
        if settings.compute_type == "float16":
            attempts.append(WhisperSettings(settings.model, "cuda", "int8_float16", settings.model_auto))
        attempts.append(cpu_fallback(settings))
    last_error: Optional[BaseException] = None
    for number, attempt in enumerate(attempts):
        if number and attempt.device == "cuda" and last_error is not None and not _is_oom(last_error):
            continue  # int8_float16 only helps when the GPU ran out of memory
        try:
            rows, language = _run_whisper(audio_path, attempt, duration, language_mode,
                                          on_progress, check_cancel, log)
            return rows, language, attempt
        except cancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - every GPU problem gets a fallback
            if attempt.device != "cuda":
                raise
            last_error = exc
            nxt = next((a for a in attempts[number + 1:]
                        if a.device == "cpu" or _is_oom(exc)), None)
            if log:
                log(f"GPU transcription failed ({str(exc)[:200]}); "
                    + (f"retrying with Whisper {nxt.label()}." if nxt else "no fallback left."))
    raise RuntimeError(f"Transcription failed: {last_error}")


def is_loop_row(row: dict) -> bool:
    """A segment that is a decoding loop: highly compressible or a handful of tokens repeated."""
    words = [w.get("word", "") for w in row.get("words") or []]
    tokens = [t for t in (_tokens(w) for w in words) for t in t]
    if row.get("compression_ratio", 0) > 2.4:
        return True
    return len(tokens) >= 8 and len(set(tokens)) / len(tokens) < 0.3


def _row(seg) -> dict:
    return {
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
        ],
    }


def _redecode_loops(model, audio_path: Path, rows: list[dict], language_mode: str, check_cancel, log) -> list[dict]:
    """Re-decode the stretches where batched decoding looped ("kar do kar do ..." x45).

    Batched decoding has no temperature fallback, so a chunk that starts looping stays looped and
    the real speech under it is lost. Those stretches (usually a few percent of the stream) are
    decoded again sequentially, where Whisper retries at higher temperature; a replacement is
    kept only if it is not a loop itself.
    """
    loops = [r for r in rows if is_loop_row(r)]
    if not loops:
        return rows
    ranges: list[list[float]] = []
    for r in sorted(loops, key=lambda r: r["start"]):
        a, b = max(0.0, r["start"] - 0.3), r["end"] + 0.3
        if ranges and a <= ranges[-1][1] + 1.0:
            ranges[-1][1] = max(ranges[-1][1], b)
        else:
            ranges.append([a, b])
    seconds = sum(b - a for a, b in ranges)
    if log:
        log(f"Re-decoding {len(loops)} looped segment(s) ({seconds / 60:.1f} min) with temperature fallback...")
    options = whisper_options(language_mode, batched=False)
    options.pop("vad_filter", None)
    options.pop("vad_parameters", None)
    # These stretches already looped at temperature 0, so go straight to sampling (2 candidates
    # per window): about 10x faster than the full 0.0-1.0 fallback ladder, which first re-creates
    # each loop at full length before giving up on it.
    options.update({"temperature": [0.4, 0.8], "best_of": 2})
    clip = [round(x, 2) for a, b in ranges for x in (a, b)]
    try:
        segments, _ = model.transcribe(str(audio_path), clip_timestamps=clip, **options)
        fresh = []
        for seg in segments:
            if check_cancel:
                check_cancel()
            row = _row(seg)
            if not is_loop_row(row) and row["text"]:
                fresh.append(row)
    except Exception as exc:  # noqa: BLE001 - keep the batched result (clean-up still trims loops)
        if log:
            log(f"Loop re-decode failed ({str(exc)[:160]}); keeping the batched transcript.")
        return rows

    def inside(row):
        return any(a - 0.05 <= row["start"] and row["end"] <= b + 0.05 for a, b in ranges)

    kept = [r for r in rows if not (is_loop_row(r) or (inside(r) and r["start"] >= 0))]
    # Rows inside a looped range that were fine are replaced too, so the stretch is not doubled.
    merged = sorted(kept + [r for r in fresh if inside(r)], key=lambda r: r["start"])
    if log:
        log(f"Loop re-decode recovered {sum(len(r['words']) for r in fresh)} words.")
    return merged


def _run_whisper(audio_path: Path, settings: WhisperSettings, duration: float, language_mode: str,
                 on_progress, check_cancel, log) -> tuple[list[dict], str]:
    from faster_whisper import WhisperModel

    batched = settings.device == "cuda"
    if batched:
        prepare_cuda_libraries()
    options = whisper_options(language_mode, batched=batched)
    mode_text = ("language picked per chunk from English/Hindi/Bengali" if options["multilingual"]
                 else f"forced {language_mode}")
    if log:
        log(f"Loading Whisper {settings.label()}{' (batched)' if batched else ''}; {mode_text}...")
    model = WhisperModel(settings.model, device=settings.device, compute_type=settings.compute_type)
    try:
        if options["multilingual"] and getattr(model.model, "is_multilingual", False):
            model.model = _LanguageLimiter(model.model, MIXED_LANGUAGES)
        if batched:
            from faster_whisper import BatchedInferencePipeline

            segments, info = BatchedInferencePipeline(model).transcribe(str(audio_path), **options)
        else:
            segments, info = model.transcribe(str(audio_path), **options)
        rows: list[dict] = []
        started = time.monotonic()
        last_report = 0.0
        prompt = options.get("initial_prompt")
        for seg in segments:
            if check_cancel:
                check_cancel()
            now = time.monotonic()
            processed = min(duration, max(0.0, float(seg.end)))
            if on_progress and (now - last_report >= 0.25):
                speed = processed / max(0.1, now - started)
                remaining = (duration - processed) / speed if speed > 0 else None
                on_progress(processed, duration, remaining)
                last_report = now
            if is_prompt_echo(seg.text, prompt):
                continue
            rows.append(_row(seg))
        if on_progress:
            on_progress(duration, duration, 0.0)
        rows.sort(key=lambda r: r["start"])
        if batched:
            rows = _redecode_loops(model, audio_path, rows, language_mode, check_cancel, log)
        return rows, getattr(info, "language", "unknown")
    finally:
        # Free GPU memory before the Ollama models load.
        del model
        gc.collect()
