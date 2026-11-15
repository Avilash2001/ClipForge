"""Moment finding for short-form clips.

This module holds the "what is worth clipping" logic that does not depend on
FFmpeg or the GUI:

* transcript clean-up (drops Whisper hallucinations before they reach ranking
  or burned-in captions),
* loudness and live-chat spike signals measured against a rolling baseline,
* a transcript-first local-LLM pass that reads the stream in ~12 minute chunks
  and returns variable-length moments with a harsh 0-100 score, and
* boundary snapping so clips start on a sentence and end after the reaction.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import requests

# Leading words that make a clip feel like it started mid-thought.
FILLER_WORDS = {
    "uh", "uhh", "um", "umm", "erm", "hmm", "mm", "mhm", "ah", "so", "yeah", "anyway",
    "anyways", "okay", "ok", "like", "and", "but", "well", "right", "alright",
}
MOMENT_CATEGORIES = [
    "first_reaction", "crazy_discovery", "freakout", "funny_fail", "hot_take",
    "clutch_play", "rage", "roast_banter", "wholesome", "awkward", "other",
]
_SENTENCE_END = re.compile(r"[.!?…।]['\")\]]*$")


# ---------------------------------------------------------------------------
# Transcript clean-up
# ---------------------------------------------------------------------------

def _allowed_scripts(language_mode: str) -> set[str]:
    mode = (language_mode or "").lower()
    if mode.startswith("english"):
        return {"LATIN"}
    if mode.startswith("hindi"):
        return {"LATIN", "DEVANAGARI"}
    if mode.startswith("bengali"):
        return {"LATIN", "BENGALI"}
    return {"LATIN", "DEVANAGARI", "BENGALI"}


def _char_script(ch: str) -> str:
    return unicodedata.name(ch, "UNKNOWN").split(" ")[0]


def _has_foreign_script(text: str, allowed: set[str]) -> bool:
    return any(ch.isalpha() and _char_script(ch) not in allowed for ch in text)


def _mixes_indic_scripts(word: str) -> bool:
    """A single word spelled in two Indic scripts (e.g. Bengali + Devanagari) only comes from decode loops."""
    scripts = {_char_script(ch) for ch in word if ch.isalpha()} - {"LATIN"}
    return len(scripts) > 1


def _normalise_token(word: str) -> str:
    """Lower-case word without punctuation. Combining marks are kept: Hindi/Bengali vowel signs are
    part of the word (without them "को" and "के" would both become "क")."""
    return "".join(ch for ch in str(word).lower()
                   if ch == "'" or unicodedata.category(ch)[0] in ("L", "M", "N"))


# Whisper was trained on YouTube audio and, on noise, invents the channel outro. One mention can be
# real speech; two or more in one segment are dropped.
OUTRO_HALLUCINATIONS = {"subscribe", "subscribed", "सब्सक्राइब", "सबस्क्राइब", "सब्स्क्राइब"}
MAX_WORD_SECONDS = 1.2


def _cap_word_durations(words: list[dict], limit: float = MAX_WORD_SECONDS) -> None:
    """Batched Whisper cuts silence out before decoding; a word next to a removed pause can come back
    stretched across it (one word "lasting" 15 s). Speech is at the word's start, so cap its end."""
    for w in words:
        try:
            if float(w["end"]) - float(w["start"]) > limit:
                w["end"] = round(float(w["start"]) + limit, 3)
        except (KeyError, TypeError, ValueError):
            continue


def _drop_phrase_loops(words: list[dict], keep: int = 2) -> tuple[list[dict], int]:
    """Collapse a 2-8 word phrase repeated back to back (e.g. "like karenge like karenge ...").

    Keeps the first ``keep`` copies; a phrase needs 3+ consecutive copies to count as a loop.
    Returns (kept words, number dropped).
    """
    tokens = [_normalise_token(w.get("word", "")) for w in words]
    drop = [False] * len(words)
    for n in (2, 3, 4, 5, 6, 7, 8):
        i = 0
        while i + 3 * n <= len(tokens):
            phrase = tokens[i:i + n]
            if not all(phrase) or len(set(phrase)) == 1:
                i += 1
                continue
            copies, j = 1, i + n
            while j + n <= len(tokens) and tokens[j:j + n] == phrase:
                copies, j = copies + 1, j + n
            if copies >= 3:
                for k in range(i + keep * n, j):
                    drop[k] = True
                i = j
            else:
                i += 1
    kept = [w for w, d in zip(words, drop) if not d]
    return kept, len(words) - len(kept)


def clean_transcript(rows: list[dict], language_mode: str = "Mixed") -> tuple[list[dict], dict]:
    """Remove low-confidence / hallucinated Whisper output.

    Uses faster-whisper's per-segment confidence (when present), drops words in
    scripts that cannot occur in the selected language mode (e.g. CJK or Tibetan
    glyphs inside a Hindi stream), replacement characters, character floods and
    runaway repetition loops. Older transcripts without confidence fields are
    still filtered by the text rules.
    """
    allowed = _allowed_scripts(language_mode)
    cleaned: list[dict] = []
    stats = {"segments_in": len(rows), "segments_dropped": 0, "words_dropped": 0}
    for seg in rows:
        text = str(seg.get("text", ""))
        avg = seg.get("avg_logprob")
        nsp = seg.get("no_speech_prob")
        ratio = seg.get("compression_ratio")
        compact = re.sub(r"\s+", "", text)
        # faster-whisper reports avg_logprob / no_speech_prob / compression_ratio per 30 s
        # decoding window, shared by every segment cut from it - so only extreme window
        # values are trusted here; per-segment judgement uses word probabilities below.
        probs = [float(w["probability"]) for w in (seg.get("words") or []) if w.get("probability") is not None]
        mean_prob = sum(probs) / len(probs) if probs else 1.0
        latin_only = all(_char_script(ch) == "LATIN" for ch in text if ch.isalpha())
        # Thresholds were re-tuned on large-v3 output (v1.7): real code-switched speech that matched
        # YouTube's own captions had mean word confidence 0.17-0.35, so only very low values are
        # treated as invented. Whisper's compression-ratio rule is calibrated on Latin text; Hindi
        # and Bengali in UTF-8 compress far better, so for those the loop check below decides.
        bad = (
            (ratio is not None and ratio > (2.4 if latin_only else 4.0))
            or (avg is not None and nsp is not None and nsp > 0.8 and avg < -1.0)
            or (len(probs) >= 3 and mean_prob < 0.15)
            or (1 <= len(probs) <= 2 and mean_prob < 0.08)
            or bool(re.search(r"(.)\1{7,}", compact))
        )
        if bad:
            stats["segments_dropped"] += 1
            continue
        words_in = seg.get("words") or []
        words_out: list[dict] = []
        repeat_token, repeat_count, repeats = "", 0, 0
        for word in words_in:
            raw = str(word.get("word", ""))
            prob = word.get("probability")
            token = _normalise_token(raw)
            drop = (
                "�" in raw
                or (prob is not None and prob < 0.08)
                or _has_foreign_script(raw, allowed)
                or _mixes_indic_scripts(raw)
            )
            if token and token == repeat_token:
                repeat_count += 1
                if repeat_count >= 4:  # a fourth+ identical word in a row is almost always a decode loop
                    drop = True
                    repeats += 1
            else:
                repeat_token, repeat_count = token, 1
            if drop:
                stats["words_dropped"] += 1
                continue
            words_out.append(dict(word))
        outro = [w for w in words_out if _normalise_token(w.get("word", "")) in OUTRO_HALLUCINATIONS]
        if len(outro) >= 2:   # "subscribe subscribe ...": Whisper's YouTube-outro hallucination
            words_out = [w for w in words_out if w not in outro]
            stats["words_dropped"] += len(outro)
        words_out, phrase_repeats = _drop_phrase_loops(words_out)
        _cap_word_durations(words_out)
        stats["words_dropped"] += phrase_repeats
        repeats += phrase_repeats
        if words_in:
            mostly_loop = len(words_in) >= 8 and repeats > 0.5 * len(words_in)
            if mostly_loop or not any(any(ch.isalnum() for ch in str(w.get("word", ""))) for w in words_out):
                stats["segments_dropped"] += 1
                continue
            new_text = "".join(str(w.get("word", "")) for w in words_out).strip()
        else:
            kept_tokens = [tok for tok in text.split()
                           if "�" not in tok and not _has_foreign_script(tok, allowed)]
            new_text = " ".join(kept_tokens).strip()
            if not any(ch.isalnum() for ch in new_text):
                stats["segments_dropped"] += 1
                continue
        row = dict(seg)
        row["text"] = re.sub(r"\s+", " ", new_text)
        row["words"] = words_out
        cleaned.append(row)
    cleaned = _drop_cross_segment_loops(cleaned, stats)
    return cleaned, stats


def _drop_cross_segment_loops(rows: list[dict], stats: dict, max_gap: float = 3.0) -> list[dict]:
    """Repetition loops that Whisper spreads over many tiny segments ("bye bye" x15 as 2-word lines).

    Words of neighbouring segments (less than ``max_gap`` apart) are treated as one stream: a word
    repeated more than 3 times, or a 2-8 word phrase more than twice, in a row is cut back, and
    segments left with nothing are dropped.
    """
    flat: list[tuple[int, dict]] = []
    for index, row in enumerate(rows):
        for word in row.get("words") or []:
            flat.append((index, word))
    if len(flat) < 4:
        return rows
    drop: set[int] = set()
    run_start = 0
    for k in range(1, len(flat) + 1):
        gap_break = k == len(flat) or (float(flat[k][1].get("start", 0)) - float(flat[k - 1][1].get("end", 0))) > max_gap
        if not gap_break:
            continue
        run = flat[run_start:k]
        words = [w for _, w in run]
        tokens = [_normalise_token(w.get("word", "")) for w in words]
        count = 0
        for i, token in enumerate(tokens):
            count = count + 1 if i and token and token == tokens[i - 1] else 1
            if count > 3:
                drop.add(id(words[i]))
        _, removed_ids = _phrase_loop_ids(words)
        drop.update(removed_ids)
        # "subscribe" spread over several tiny segments is the same outro hallucination.
        outro = [w for w, token in zip(words, tokens) if token in OUTRO_HALLUCINATIONS]
        if len(outro) >= 2:
            drop.update(id(w) for w in outro)
        run_start = k
    if not drop:
        return rows
    kept_rows = []
    for row in rows:
        words = row.get("words") or []
        remaining = [w for w in words if id(w) not in drop]
        stats["words_dropped"] += len(words) - len(remaining)
        if words and not any(any(ch.isalnum() for ch in str(w.get("word", ""))) for w in remaining):
            stats["segments_dropped"] += 1
            continue
        if len(remaining) != len(words):
            row = dict(row)
            row["words"] = remaining
            row["text"] = re.sub(r"\s+", " ", "".join(str(w.get("word", "")) for w in remaining)).strip()
        kept_rows.append(row)
    return kept_rows


def _phrase_loop_ids(words: list[dict], keep: int = 2) -> tuple[list[dict], set[int]]:
    kept, _ = _drop_phrase_loops(words, keep)
    kept_ids = {id(w) for w in kept}
    return kept, {id(w) for w in words if id(w) not in kept_ids}


# ---------------------------------------------------------------------------
# Spike signals
# ---------------------------------------------------------------------------

def _rolling_median(values, window: int):
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    window = int(window) | 1  # odd so the window is centred
    if window > arr.size:
        window = arr.size if arr.size % 2 else max(1, arr.size - 1)
    half = window // 2
    padded = np.pad(arr, (half, half), mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(view, axis=1)[: arr.size]


def loudness_spikes(rms: list[float], baseline_seconds: int = 61) -> list[float]:
    """Return 0..1 per second for "the streamer suddenly got loud".

    Compares a 3-second loudness average with the rolling one-minute median in
    decibels, so a stream that is loud all the time does not flag every second.
    +3 dB over the local baseline starts to count; +12 dB is a full spike.
    """
    import numpy as np

    arr = np.asarray(rms, dtype=np.float64)
    if arr.size < 5:
        return [0.0] * int(arr.size)
    short = np.convolve(arr, np.ones(3) / 3.0, mode="same")
    baseline = _rolling_median(arr, baseline_seconds)
    eps = 1e-4
    gain_db = 20.0 * np.log10((short + eps) / (baseline + eps))
    # Ignore near-silent sections where tiny absolute changes create large ratios.
    floor = max(float(np.percentile(arr, 20)), 0.003)
    gain_db[short < floor * 1.5] = 0.0
    return np.clip((gain_db - 3.0) / 9.0, 0.0, 1.0).round(4).tolist()


def spike_peaks(spikes: list[float], min_value: float = 0.45, min_gap: int = 20) -> list[int]:
    order = sorted(range(len(spikes)), key=lambda i: spikes[i], reverse=True)
    chosen: list[int] = []
    for i in order:
        if spikes[i] < min_value:
            break
        if all(abs(i - j) >= min_gap for j in chosen):
            chosen.append(i)
    return sorted(chosen)


def parse_live_chat(path: Path) -> list[float]:
    """Read a yt-dlp ``.live_chat.json`` file and return message offsets in seconds."""
    offsets: list[float] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            replay = obj.get("replayChatItemAction") or {}
            offset = replay.get("videoOffsetTimeMsec")
            if offset is None:
                continue
            try:
                seconds = int(offset) / 1000.0
            except (TypeError, ValueError):
                continue
            messages = sum(1 for action in replay.get("actions", []) if "addChatItemAction" in action)
            offsets.extend([seconds] * max(1, messages))
    return offsets


def chat_spikes(offsets: list[float], duration: float, window: int = 10,
                baseline_seconds: int = 301) -> list[float]:
    """Return 0..1 per second for "chat suddenly started moving faster than normal"."""
    import numpy as np

    n = max(1, int(math.ceil(duration)))
    if not offsets:
        return [0.0] * n
    counts = np.zeros(n, dtype=np.float64)
    for value in offsets:
        idx = int(value)
        if 0 <= idx < n:
            counts[idx] += 1
    rate = np.convolve(counts, np.ones(window), mode="same")
    baseline = _rolling_median(rate, baseline_seconds)
    ratio = (rate + 1.0) / (baseline + 1.0)
    # Need a real burst: at least 1.5x the local rate and several messages.
    spikes = np.clip((ratio - 1.5) / 2.0, 0.0, 1.0)
    spikes[rate < 4] = 0.0
    return spikes.round(4).tolist()


def motion_spikes(motion: list[float], baseline_seconds: int = 121) -> list[float]:
    """0..1 per second where on-screen change jumps well above its local baseline.

    ``motion`` is the 0..1 frame-difference curve (normalised to its 95th percentile).
    A game that is always busy does not flag every second: a spike needs a 3-second
    average about 1.5x the rolling two-minute median, and a real absolute change.
    """
    import numpy as np

    arr = np.asarray(motion, dtype=np.float64)
    if arr.size < 5:
        return [0.0] * int(arr.size)
    short = np.convolve(arr, np.ones(3) / 3.0, mode="same")
    baseline = _rolling_median(arr, baseline_seconds)
    ratio = (short + 0.03) / (baseline + 0.03)
    spikes = np.clip((ratio - 1.4) / 1.6, 0.0, 1.0)
    spikes[short < 0.2] = 0.0
    return spikes.round(4).tolist()


@dataclass
class SignalWindow:
    """A stretch of stream picked from signals alone (no words needed): a silent clutch or fail."""
    start: float
    end: float
    peak: float          # second where the combined signal peaked
    strength: float      # 0..1 combined signal at the peak
    loud: float = 0.0
    chat: float = 0.0
    motion: float = 0.0

    def describe(self) -> str:
        parts = [name for name, value in (("loudness", self.loud), ("chat", self.chat), ("motion", self.motion))
                 if value >= 0.35]
        return "signal spike: " + (", ".join(parts) or "combined")


# Live-chat replay lags the action: viewers type after they see it, behind the stream delay.
CHAT_LAG_SECONDS = 8


def combined_signal(duration: float, loud: list[float], chat: list[float],
                    motion_spk: Optional[list[float]] = None, chat_lag: int = CHAT_LAG_SECONDS):
    """Per-second 0..1 "something happened here" curve plus the aligned component curves."""
    import numpy as np

    n = max(1, int(math.ceil(duration)))

    def fit(values, shift: int = 0, smooth: int = 1):
        arr = np.zeros(n, dtype=np.float64)
        if values:
            src = np.asarray(values[:n], dtype=np.float64)
            arr[: src.size] = src
        if shift:
            arr = np.concatenate([arr[shift:], np.zeros(shift)])  # chat at t+lag belongs to t
        if smooth > 1:
            # A burst counts for the few seconds around it (max filter).
            arr = np.max(np.lib.stride_tricks.sliding_window_view(
                np.pad(arr, (smooth // 2, smooth // 2), mode="edge"), smooth), axis=1)[:n]
        return arr

    L = fit(loud, smooth=3)
    C = fit(chat, shift=max(0, int(chat_lag)), smooth=9)
    M = fit(motion_spk or [], smooth=3)
    combined = 0.45 * L + 0.40 * C + 0.15 * M
    # Two independent signals agreeing is much stronger evidence than either alone.
    agree = ((L >= 0.5).astype(int) + (C >= 0.5).astype(int) + (M >= 0.5).astype(int)) >= 2
    combined = np.where(agree, np.minimum(1.0, combined + 0.25), combined)
    return combined, L, C, M


def signal_windows(duration: float, loud: list[float], chat: list[float],
                   motion_spk: Optional[list[float]] = None, *,
                   exclude: Optional[list[tuple[float, float]]] = None,
                   max_windows: Optional[int] = None, min_strength: float = 0.35,
                   lead: float = 22.0, tail: float = 10.0, min_gap: int = 45) -> list[SignalWindow]:
    """Turn loudness / chat / motion spikes into candidate windows for the vision review.

    The window starts ``lead`` seconds before the peak (the build-up) and ends ``tail``
    seconds after (the reaction). Windows mostly covered by ``exclude`` ranges (moments
    the transcript picker already found) are skipped, so the two paths never duplicate.
    """
    combined, L, C, M = combined_signal(duration, loud, chat, motion_spk)
    limit = max_windows if max_windows is not None else max(4, min(15, int(duration / 600) + 2))
    excluded = list(exclude or [])
    windows: list[SignalWindow] = []
    for peak in spike_peaks(combined.tolist(), min_value=min_strength, min_gap=min_gap):
        start = max(0.0, peak - lead)
        end = min(float(duration), peak + tail)
        if end - start < 8:
            continue
        covered = max((max(0.0, min(end, b) - max(start, a)) for a, b in excluded), default=0.0)
        if covered >= 0.35 * (end - start):
            continue
        lo, hi = max(0, int(peak) - 2), min(len(combined), int(peak) + 3)
        windows.append(SignalWindow(
            start=round(start, 2), end=round(end, 2), peak=float(peak), strength=round(float(combined[peak]), 3),
            loud=round(float(L[lo:hi].max()), 3), chat=round(float(C[lo:hi].max()), 3),
            motion=round(float(M[lo:hi].max()), 3)))
    windows.sort(key=lambda w: w.strength, reverse=True)
    return sorted(windows[:limit], key=lambda w: w.start)


def range_peak(values: list[float], start: float, end: float) -> float:
    if not values:
        return 0.0
    a = max(0, int(start))
    b = min(len(values), max(a + 1, int(math.ceil(end))))
    window = values[a:b]
    return float(max(window)) if window else 0.0


# ---------------------------------------------------------------------------
# Boundary snapping
# ---------------------------------------------------------------------------

def _flat_words(rows: list[dict]) -> list[dict]:
    words: list[dict] = []
    for seg in rows:
        seg_words = [w for w in (seg.get("words") or []) if any(ch.isalnum() for ch in str(w.get("word", "")))]
        if seg_words:
            for i, w in enumerate(seg_words):
                words.append({
                    "start": float(w["start"]), "end": float(w["end"]), "word": str(w["word"]).strip(),
                    "seg_first": i == 0, "seg_last": i == len(seg_words) - 1,
                })
        elif any(ch.isalnum() for ch in str(seg.get("text", ""))):
            words.append({
                "start": float(seg["start"]), "end": float(seg["end"]), "word": str(seg["text"]).strip(),
                "seg_first": True, "seg_last": True,
            })
    words.sort(key=lambda w: w["start"])
    for i, w in enumerate(words):
        prev = words[i - 1] if i else None
        w["sentence_start"] = bool(
            w["seg_first"] or prev is None or _SENTENCE_END.search(prev["word"])
            or (w["start"] - prev["end"]) > 0.9
        )
        nxt = words[i + 1] if i + 1 < len(words) else None
        w["sentence_end"] = bool(
            w["seg_last"] or nxt is None or _SENTENCE_END.search(w["word"])
            or (nxt["start"] - w["end"]) > 0.9
        )
    return words


def snap_to_speech(start: float, end: float, rows: list[dict], duration: float,
                   min_len: float, max_len: float, *, not_before: Optional[float] = None,
                   not_after: Optional[float] = None) -> tuple[float, float]:
    """Move a rough (start, end) onto sentence boundaries.

    The clip starts at the beginning of a sentence (never on filler such as
    "uh, so yeah"), ends after the sentence that carries the payoff, and keeps a
    short tail for the reaction. Length is clamped to [min_len, max_len].
    ``not_before`` / ``not_after`` are hard limits (e.g. a loading screen that must
    stay outside the clip); they win over sentence alignment.
    """
    floor = max(0.0, float(not_before)) if not_before is not None else 0.0
    ceiling = min(float(duration), float(not_after)) if not_after is not None else float(duration)
    start = max(floor, float(start))
    end = min(ceiling, max(start + 1.0, float(end)))
    words = [w for w in _flat_words(rows) if w["start"] >= floor - 0.05 and w["end"] <= ceiling + 0.05]
    if not words:
        new_start, new_end = start, end
    else:
        # Start: closest sentence start at/before the rough start (up to 6s back),
        # else the first one shortly after.
        before = [w for w in words if w["sentence_start"] and start - 6.0 <= w["start"] <= start + 0.3]
        after = [w for w in words if w["sentence_start"] and start < w["start"] <= start + 4.0]
        if before:
            first = before[-1]
        elif after:
            first = after[0]
        else:
            inside = [w for w in words if w["start"] >= start - 0.3]
            first = inside[0] if inside and inside[0]["start"] < end else None
        if first is not None:
            idx = words.index(first)
            trimmed = 0
            while (trimmed < 3 and idx + 1 < len(words)
                   and _normalise_token(words[idx]["word"]) in FILLER_WORDS
                   and words[idx + 1]["start"] - words[idx]["end"] < 1.5
                   and words[idx + 1]["start"] < end - min_len * 0.5):
                idx += 1
                trimmed += 1
            new_start = max(floor, words[idx]["start"] - 0.25)
        else:
            new_start = start
        # End: finish the sentence that is running at the rough end (up to 6s later).
        tail = [w for w in words if w["sentence_end"] and end - 0.5 <= w["end"] <= end + 6.0]
        new_end = min(ceiling, (tail[0]["end"] + 0.7) if tail else end)
        if new_end - new_start < min_len:
            # Too short: first pull in ~40% of the missing time as setup (from an earlier sentence start)...
            deficit = min_len - (new_end - new_start)
            earlier = [w for w in words if w["sentence_start"]
                       and new_start - deficit * 0.6 <= w["start"] - 0.25 <= new_start - deficit * 0.25]
            if earlier:
                new_start = max(floor, earlier[0]["start"] - 0.25)
        if new_end - new_start < min_len:
            # ...then let the payoff/reaction run on to later sentence ends.
            later = [w for w in words if w["sentence_end"] and w["end"] + 0.7 >= new_start + min_len
                     and w["end"] + 0.7 <= new_start + max_len]
            new_end = min(ceiling, (later[0]["end"] + 0.7) if later else new_start + min_len)
        if new_end - new_start > max_len:
            fits = [w for w in words if w["sentence_end"]
                    and new_start + min_len <= w["end"] + 0.7 <= new_start + max_len]
            new_end = (fits[-1]["end"] + 0.7) if fits else new_start + max_len
    new_end = min(ceiling, new_end)
    if new_end - new_start < min(min_len, duration):
        new_start = max(floor, new_end - min_len)
        new_end = min(ceiling, max(new_end, new_start + min_len))
    return round(new_start, 2), round(new_end, 2)


# ---------------------------------------------------------------------------
# Local LLM moment picker
# ---------------------------------------------------------------------------

@dataclass
class Moment:
    start: float
    end: float
    score: float
    category: str = "other"
    hook_title: str = ""
    why: str = ""
    post_caption: str = ""
    hashtags: list[str] = field(default_factory=list)
    needs_context: bool = False
    payoff_quote: str = ""


def extract_json_object(text: str) -> dict:
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


def salvage_json_array(text: str) -> Optional[dict]:
    """Recover the complete items of a ``{"key": [ ... ]}`` reply that was cut off.

    With structured output, gemma4 sometimes writes one or two items and then pads with
    whitespace until it hits ``num_predict``; the JSON never closes. Every item that is
    complete is still good.
    """
    match = re.match(r'\s*\{\s*"(\w+)"\s*:\s*\[', text or "")
    if not match:
        return None
    decoder = json.JSONDecoder()
    items, pos = [], match.end()
    while True:
        while pos < len(text) and text[pos] in " \t\r\n,":
            pos += 1
        if pos >= len(text) or text[pos] != "{":
            break
        try:
            item, pos = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            break
        items.append(item)
    return {match.group(1): items} if items else None


def ollama_json(url: str, model: str, prompt: str, schema: dict, *, images: Optional[list[str]] = None,
                temperature: float = 0.1, num_ctx: int = 12288, num_predict: int = 1800,
                timeout: int = 1800) -> dict:
    message = {"role": "user", "content": prompt}
    if images:
        message["images"] = images
    response = requests.post(
        f"{url}/api/chat",
        json={
            "model": model,
            "messages": [message],
            "stream": False,
            "format": schema,
            "think": False,
            "options": {"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json().get("message", {})
    errors = []
    for raw in (payload.get("content", "") or "", payload.get("thinking", "") or ""):
        try:
            return extract_json_object(raw)
        except Exception as exc:  # noqa: BLE001 - try the next field
            errors.append(str(exc))
    salvaged = salvage_json_array(payload.get("content", "") or "")
    if salvaged is not None:
        return salvaged
    raise ValueError("Ollama returned no valid structured JSON: " + " / ".join(errors))


def transcript_lines(rows: list[dict], start: float, end: float,
                     loud: list[float], chat: list[float]) -> list[str]:
    lines = []
    for seg in rows:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        text = re.sub(r"\s+", " ", str(seg.get("text", ""))).strip()
        if not any(ch.isalnum() for ch in text):
            continue
        marks = []
        if range_peak(loud, seg["start"], seg["end"] + 1) >= 0.5:
            marks.append("[LOUD]")
        if range_peak(chat, seg["start"], seg["end"] + 4) >= 0.5:
            marks.append("[CHAT SPIKE]")
        suffix = (" " + " ".join(marks)) if marks else ""
        lines.append(f"[{int(seg['start'])}s] {text[:240]}{suffix}")
    return lines


MOMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "moments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "category": {"type": "string", "enum": MOMENT_CATEGORIES},
                    "score": {"type": "integer"},
                    "needs_context": {"type": "boolean"},
                    "payoff_quote": {"type": "string"},
                    "hook_title": {"type": "string"},
                    "why": {"type": "string"},
                    "post_caption": {"type": "string"},
                    "hashtags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["start", "end", "category", "score", "needs_context", "payoff_quote", "hook_title", "why"],
            },
        }
    },
    "required": ["moments"],
}


# Titles must be literally true. Small models drift towards hype ("the WHOLE school", "the most
# ABSURD..."); these rules are shared by the picker, the vision review and the critic.
LITERAL_TITLE_RULES = (
    "Titles must be LITERALLY true, not hype: every quantity and quantifier must match what is shown or said "
    "(a group of students is not 'the WHOLE school', one enemy is not 'the whole team', two kills is not an 'ACE'; "
    "avoid whole/entire/all/every/everyone/nobody/always/never unless literally so). Get WHO did WHAT to WHOM right: "
    "do not credit the streamer with a teammate's or a game character's action or line, and do not turn being killed "
    "into killing. No superlatives or rankings that cannot be checked (most, best, worst, craziest, ever, of all time, "
    "world record). Never use he/she/his/her for the streamer or players; use the name, 'they' or the role. "
    "If you are unsure of a detail, leave it out rather than guess."
)

# Words that make a title a factual claim worth a second look when the evidence does not contain them.
ABSOLUTE_WORDS = {
    "whole", "entire", "all", "every", "everyone", "everybody", "everything", "nobody", "nothing", "always",
    "never", "most", "best", "worst", "ever", "only", "first", "last", "biggest", "craziest", "fastest",
    "greatest", "hardest", "funniest", "wildest", "impossible", "record", "ace", "team", "squad", "lobby",
    # Claims about how someone felt: in the final test "CARRYMINATI panics as the police arrive" was burned
    # on a clip where he calmly asks for music. These need an audible outburst and a matching face.
    "panics", "panic", "panicking", "freaks", "freaking", "loses", "rage", "rages", "raging", "screams",
    "screaming", "terrified", "furious", "shocked", "meltdown", "desperate", "crazy", "insane",
}
GENDERED_WORDS = {"he", "him", "his", "she", "her", "hers", "himself", "herself"}
# Pure hype: if one of these survives the fact-check without being in the evidence, the title is dropped.
STRICT_TITLE_WORDS = {"whole", "entire", "every", "everyone", "everybody", "nobody", "always", "never", "most",
                      "best", "worst", "ever"}
_NOT_SUPERLATIVES = {"interest", "honest", "quest", "rest", "request", "contest", "protest", "harvest", "forest",
                     "chest", "guest", "test", "latest", "suggest", "manifest", "invest", "arrest"}


def title_risks(title: str, evidence: str = "", *, strict: bool = False) -> list[str]:
    """Words in a title that make a factual claim the evidence text does not itself contain.

    Default (trigger) mode flags quantifiers, superlatives (incl. "-est" words) and gendered
    pronouns, so the title gets a literal-truth fact-check. ``strict`` mode flags only pure
    hype words and superlatives: used as the final gate after the fact-check.
    """
    tokens = [_normalise_token(t) for t in re.split(r"[\s/]+", str(title or ""))]
    tokens = [t for t in tokens if t]
    seen = {_normalise_token(t) for t in re.split(r"[\s/]+", str(evidence or ""))}
    risky = []
    for token in tokens:
        superlative = len(token) > 5 and token.endswith("est") and token not in _NOT_SUPERLATIVES
        if strict:
            flagged = (token in STRICT_TITLE_WORDS or superlative) and token not in seen
        else:
            flagged = ((token in ABSOLUTE_WORDS or superlative) and token not in seen) or token in GENDERED_WORDS
        if flagged and token not in risky:
            risky.append(token)
    return risky


def moment_prompt(lines: list[str], chunk_start: float, chunk_end: float, *, min_len: int, max_len: int,
                  game_profile: str, creator: str) -> str:
    who = (f"The streamer is {creator}. Name them in hook titles (e.g. '{creator.split()[0].upper()} can't believe...')."
           if creator else "The streamer's name is unknown; write hook titles without inventing a name.")
    return (
        "You are a ruthless short-form video editor cutting a gaming livestream into YouTube Shorts / TikTok / Reels. "
        f"Game: {game_profile}. {who}\n"
        f"Below is the timestamped transcript for {chunk_start/60:.0f}-{chunk_end/60:.0f} min of the stream. "
        "[LOUD] marks where the streamer's voice/volume suddenly spiked. [CHAT SPIKE] marks where live chat suddenly "
        "moved much faster than normal. Both are strong hints that something happened, but the words decide.\n\n"
        "Pick only moments a STRANGER scrolling Shorts would stop for: something actually HAPPENS and people react to it. "
        "In story games much of the speech is the game's own characters (cutscene dialogue), not the streamer. Only credit words or reactions to the streamer when it is clearly them (commentary, reactions, talking to teammates); for a cutscene moment, title the scene itself, e.g. 'Bank heist goes WRONG in the first minute'. "
        "Good: a death, kill, clutch or fail with a big reaction; a surprise or crazy discovery; a full-on freakout or "
        "rage; a genuinely funny line that lands on its own; a first reaction to something new.\n"
        "Filler that must score below 40: talking about settings, graphics, skins, mic/audio problems, ping or lag; "
        "buying weapons or waiting in menus/lobbies; planning or callouts with no outcome; inside jokes and nicknames; "
        "people just repeating a word; anything where you cannot say what happened.\n"
        "Rules:\n"
        "- Be harsh. Return 0-3 moments for this section; returning none is normal and correct for most sections.\n"
        "- Each moment must make sense to someone who has never seen this stream. If it needs earlier context, set "
        "needs_context=true and score it below 30.\n"
        "- payoff_quote: copy, word for word from the transcript, the line where the payoff/reaction happens. If you "
        "cannot point to such a line, it is not a moment.\n"
        "- start = where the setup begins: the start of a full sentence, never mid-sentence, never on filler like "
        "'uh', 'so yeah', 'anyway'. end = just after the payoff and the reaction land. "
        f"Each moment must be {min_len}-{max_len} seconds long. Use the [Ns] timestamps (seconds).\n"
        "- score is an integer 0-100: 90+ would go viral on its own; 70-89 strong and clearly shareable; "
        "50-69 decent; below 50 is not worth posting. Most moments should NOT score above 70.\n"
        "- hook_title: the text shown on the video. Max 60 characters, third person, present tense, curiosity-driven. "
        "Write it in normal sentence case and put ONLY 1-3 key words in CAPITALS (never the whole title). May end with "
        "one emoji. Good: \"TIM can't believe who's STREAMING inside GTA 6 💀\", \"Squad gets SNIPED through a wall 😭\". "
        "Bad: \"SQUAD GETS SNIPED THROUGH A WALL\". Write it in English even if the speech is Hindi or Bengali. "
        + LITERAL_TITLE_RULES + "\n"
        "- post_caption: one or two ready-to-post sentences ending with a question or call to react. hashtags: 3-5, no #. "
        "Name a specific game only if the Game line above names it; otherwise use generic tags (gaming, fps, funny...).\n"
        "- why: one short factual sentence saying what happens.\n\n"
        "Transcript:\n" + "\n".join(lines)
    )


def quote_support(quote: str, rows: list[dict], start: float, end: float, pad: float = 8.0) -> float:
    """Fraction of the quote's words that really occur in the transcript around the moment (0..1)."""
    tokens = [_normalise_token(t) for t in str(quote or "").split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return 0.0
    nearby = set()
    for seg in rows:
        if seg["end"] >= start - pad and seg["start"] <= end + pad:
            nearby.update(_normalise_token(t) for t in str(seg.get("text", "")).split())
    return sum(1 for t in tokens if t in nearby) / len(tokens)


_TITLE_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to", "for", "with", "his", "her", "their",
    "is", "are", "was", "be", "this", "that", "it", "he", "she", "they", "you", "him", "them", "from", "by",
    "gets", "get", "can't", "cant", "who", "what", "when", "into", "out", "up", "just", "so", "not",
}
_TITLE_ACRONYMS = {"GTA", "FPS", "NPC", "MVP", "GG", "AFK", "OP", "RNG", "ACE", "AI", "PC", "VR", "XP", "HP"}


def tidy_hook_title(title: str, max_chars: int = 70) -> str:
    """Normalise an LLM hook title to the "sentence case with 1-3 CAPS words" style.

    Small local models often SHOUT THE WHOLE TITLE; that reads as spam on a phone.
    """
    text = re.sub(r"\[\s*(LOUD|CHAT[^\]]*|\d+s)\s*\]", " ", str(title or ""), flags=re.I)  # prompt markers
    text = re.sub(r"\s+", " ", text).strip().strip("\"'“”‘’").strip()
    if not text:
        return ""
    words = text.split(" ")
    letters = [ch for ch in text if ch.isalpha() and ch.isascii()]
    upper_ratio = sum(ch.isupper() for ch in letters) / max(1, len(letters))
    caps_words = [w for w in words if sum(c.isalpha() for c in w) >= 2 and w.upper() == w and any(c.isalpha() for c in w)]
    if upper_ratio > 0.7 or len(caps_words) > 3:
        def core(word: str) -> str:
            return re.sub(r"[^\w']", "", word.lower())
        # Keep gaming acronyms in capitals, plus the two longest content words for emphasis.
        keep = {i for i, w in enumerate(words) if core(w).upper() in _TITLE_ACRONYMS}
        ranked = sorted((i for i, w in enumerate(words) if len(core(w)) >= 4 and core(w) not in _TITLE_STOPWORDS
                         and i not in keep), key=lambda i: len(core(words[i])), reverse=True)
        keep.update(ranked[:2])
        rebuilt = []
        for i, w in enumerate(words):
            if i in keep:
                rebuilt.append(w.upper())
            elif i == 0:
                rebuilt.append(w[:1].upper() + w[1:].lower())
            else:
                rebuilt.append(w.lower())
        text = " ".join(rebuilt)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip(",;:-") + "…"
    return text


def _overlap_fraction(a: Moment, b: Moment) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    shortest = max(0.1, min(a.end - a.start, b.end - b.start))
    return inter / shortest


def dedupe_moments(moments: list[Moment], max_overlap: float = 0.35) -> list[Moment]:
    kept: list[Moment] = []
    for moment in sorted(moments, key=lambda m: m.score, reverse=True):
        if all(_overlap_fraction(moment, other) < max_overlap for other in kept):
            kept.append(moment)
    return sorted(kept, key=lambda m: m.start)


def pick_moments(rows: list[dict], duration: float, loud: list[float], chat: list[float], *,
                 url: str, model: str, min_len: int, max_len: int, game_profile: str = "Auto / mixed games",
                 creator: str = "", chunk_seconds: int = 720, overlap_seconds: int = 60,
                 on_chunk: Optional[Callable[[int, int, int], None]] = None,
                 log: Optional[Callable[[str], None]] = None,
                 check_cancel: Optional[Callable[[], None]] = None) -> list[Moment]:
    """Read the transcript in chunks and let the local LLM pick standalone moments."""
    step = max(60, chunk_seconds - overlap_seconds)
    starts = list(range(0, max(1, int(math.ceil(duration))), step))
    moments: list[Moment] = []
    failures = 0
    for number, chunk_start in enumerate(starts, 1):
        if check_cancel:
            check_cancel()
        chunk_end = min(duration, chunk_start + chunk_seconds)
        lines = transcript_lines(rows, chunk_start, chunk_end, loud, chat)
        if len(lines) < 3:
            if on_chunk:
                on_chunk(number, len(starts), len(moments))
            continue
        prompt = moment_prompt(lines, chunk_start, chunk_end, min_len=min_len, max_len=max_len,
                               game_profile=game_profile, creator=creator)
        try:
            data = ollama_json(url, model, prompt, MOMENT_SCHEMA, temperature=0.2,
                               num_ctx=16384, num_predict=2400, timeout=900)
        except Exception as exc:  # noqa: BLE001 - one bad chunk should not sink the job
            failures += 1
            if log:
                log(f"Moment picker chunk {number}/{len(starts)} failed: {exc}")
            if failures >= 3 and not moments:
                raise
            continue
        found = 0
        for item in data.get("moments", []) or []:
            try:
                start = float(item.get("start"))
                end = float(item.get("end"))
                score = max(0.0, min(100.0, float(item.get("score", 0))))
            except (TypeError, ValueError):
                continue
            if not (chunk_start - 10 <= start < end <= chunk_end + 30):
                continue
            needs_context = bool(item.get("needs_context", False))
            if needs_context:
                score = min(score, 25.0)
            # Grounding: the payoff line must really be in the transcript near the moment.
            support = quote_support(item.get("payoff_quote", ""), rows, start, end)
            if support < 0.5:
                if log:
                    log(f"  dropped {start:.0f}-{end:.0f}s: payoff quote not found in transcript "
                        f"({str(item.get('payoff_quote', ''))[:60]!r})")
                continue
            tags = item.get("hashtags") if isinstance(item.get("hashtags"), list) else []
            moments.append(Moment(
                start=start, end=end, score=score,
                category=str(item.get("category", "other"))[:40],
                hook_title=tidy_hook_title(item.get("hook_title", "")),
                why=str(item.get("why", "")).strip()[:200],
                post_caption=str(item.get("post_caption", "")).strip()[:260],
                hashtags=[str(t).lstrip("#").strip()[:40] for t in tags[:5] if str(t).strip()],
                needs_context=needs_context,
                payoff_quote=str(item.get("payoff_quote", "")).strip()[:200],
            ))
            found += 1
        if log:
            log(f"Moment picker chunk {number}/{len(starts)} ({chunk_start/60:.0f}-{chunk_end/60:.0f} min): "
                f"{found} possible moment(s).")
        if on_chunk:
            on_chunk(number, len(starts), len(moments))
    return dedupe_moments(moments)


# ---------------------------------------------------------------------------
# Critic pass: re-judge the finalists side by side
# ---------------------------------------------------------------------------

CLAIMS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "claim": {"type": "string"},
            "evidence": {"type": "string"},
            "literally_true": {"type": "boolean"},
        },
        "required": ["claim", "evidence", "literally_true"],
    },
}

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "frames_show": {"type": "string"},
                    "what_happens": {"type": "string"},
                    "event_visible": {"type": "boolean"},
                    "clear_reaction": {"type": "boolean"},
                    "standalone": {"type": "boolean"},
                    "payoff": {"type": "boolean"},
                    "stranger_score": {"type": "integer"},
                    "title_claims": CLAIMS_SCHEMA,
                    "title_supported": {"type": "boolean"},
                    "fixed_title": {"type": "string"},
                    "keep": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "frames_show", "what_happens", "event_visible", "clear_reaction", "standalone",
                             "payoff", "stranger_score", "title_claims", "title_supported", "fixed_title", "keep",
                             "reason"],
            },
        }
    },
    "required": ["verdicts"],
}

STORY_GAME_NOTE = (
    "In story games much of the speech is the game's own characters (cutscene dialogue), not the streamer. Only "
    "credit words or reactions to the streamer when it is clearly them (commentary, reactions, talking to "
    "teammates); for a cutscene moment, title the scene itself, e.g. 'Bank heist goes WRONG in the first minute'. "
    "A title that credits the streamer with a game character's line is NOT supported."
)


FRAMES_NOTE = (
    "Each clip also has an attached image: a 3x3 contact sheet of its frames in time order (images are in the "
    "same order as the clips). LOOK at it. frames_summary was written by another model and can be wrong; when "
    "it disagrees with what you see, trust the frames. A claim about an action (caught, killed, crashed, "
    "exploded) is true only if the frames or the speech show it. The streamer's webcam is in the frames: use "
    "their face to judge claims about their reaction (laughing is not panicking)."
)


def critic_prompt(items: list[dict], game_profile: str, creator: str, with_frames: bool = False) -> str:
    who = f"The streamer is {creator}. " if creator else ""
    return (
        "You are the final reviewer for a gaming Shorts channel. A first pass already picked these candidate clips "
        f"from a {game_profile} stream. {who}Re-judge them side by side, harshly, as a STRANGER scrolling Shorts who "
        "has never seen this streamer.\n"
        "key_line, when present, is the line in the speech where the first pass heard the payoff - verbatim "
        "speech, so translate it and use it as evidence, but judge for yourself whether it is a real payoff.\n"
        "For each clip:\n"
        "- frames_show: (only when images are attached, else write 'no frames') first describe ONLY what you "
        "see in the 9 frames, in order: the place, what the characters are doing, any visible event, and the "
        "streamer's face in the webcam (smiling, shouting, calm). No guessing from the title.\n"
        "- what_happens: in one plain, literal sentence, what actually happens in the clip (from the speech and the "
        "frames or frame summary): who does what to whom, and how many. If you cannot say, write 'nothing clear'.\n"
        "- stranger_score 0-100: would a stranger stop scrolling and watch to the end? Something must HAPPEN with a "
        "clear payoff (a death, kill, clutch, fail, surprise, rage, a line that is funny on its own). Talk about "
        "settings, graphics, skins, audio/mic, ping, buying weapons, menus, planning without an outcome, inside jokes, "
        "or people repeating a word scores below 35. 'nothing clear' scores below 20.\n"
        "- title_claims: split the title into every factual claim it makes: WHO (which person or character), DID "
        "WHAT, TO WHOM, HOW MANY / HOW MUCH (any quantifier or number: whole, entire, all, every, squad, team, two, "
        "ace...), and any superlative or judgement (most, best, worst, craziest, ever, insane, brutal, loses it). For "
        "each, quote the evidence from speech_in_clip or frames_summary and set literally_true. It is true only if "
        "the evidence states or plainly shows exactly that: 'a group of NPCs' does NOT support 'the WHOLE school'; "
        "one teammate dying does NOT support 'the whole squad'; the streamer being killed does NOT support the "
        "streamer killing. A superlative or ranking ('the most ABSURD', 'the craziest') can never be checked, so it "
        "is false. A big reaction ('freaks out', 'loses it') needs an audible outburst in the speech (shouting, "
        "swearing, [LOUD]). A he/she/his/her pronoun for the streamer or a player is false (their gender is not "
        "given).\n"
        "- title_supported: true only if EVERY claim is literally_true.\n"
        "- fixed_title: ALWAYS fill it. If the title is supported, repeat it exactly. If not, write the most "
        "literal version that keeps the hook: drop or correct each false claim (e.g. 'Avixion gets ambushed by "
        "the WHOLE school' -> 'Avixion gets JUMPED by a gang of students'). Same style: sentence case, 1-3 CAPS "
        "words, optional emoji, max 60 characters, English.\n"
        "- Four yes/no checks, answered strictly:\n"
        "  event_visible: something visibly HAPPENS in the frames - a fight, kill, death, crash, explosion, chase, "
        "fall, big reveal - not just walking, driving, standing, talking or menus.\n"
        "  clear_reaction: the streamer clearly reacts to it - big laugh, shout, shock, rage - in the speech or "
        "on their face in the webcam. Calm commentary is false.\n"
        "  standalone: a stranger understands the setup without earlier context or knowing the people.\n"
        "  payoff: the clip lands something by the end - the event resolves or a punchline hits.\n"
        "- keep: true only if you would actually post it.\n"
        + STORY_GAME_NOTE + "\n" + LITERAL_TITLE_RULES + "\n"
        + (FRAMES_NOTE + "\n" if with_frames else "") +
        "Compare the clips with each other; be consistent. Return a verdict for every index.\n\n"
        "Clips:\n" + json.dumps(items, ensure_ascii=False, indent=1)
    )


VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "titles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "claims": CLAIMS_SCHEMA,
                    "literal_title": {"type": "string"},
                },
                "required": ["index", "claims", "literal_title"],
            },
        }
    },
    "required": ["titles"],
}


def verify_prompt(items: list[dict], creator: str, with_frames: bool = False) -> str:
    who = f"The streamer is {creator}. " if creator else ""
    return (
        "You are a fact-checker for gaming Shorts titles. " + who +
        "For each title, list its factual claims (who, did what, to whom, how many, any superlative or judgement) "
        "with the evidence quoted from 'evidence', and mark each literally_true. Words flagged in 'check_words' "
        "need special care: keep them only if the evidence literally supports them. A word about how someone felt "
        "(panics, freaks out, loses it, rage, screams, shocked, desperate) is true only if the speech has a real "
        "outburst (shouting, swearing, repeated exclamations) AND, when frames are attached, the streamer's face "
        "matches - a smiling or calm streamer is not panicking; use a neutral verb such as 'reacts'. 'Crazy' or "
        "'insane' needs the speech to say it or the frames to show something extraordinary. Then write "
        "literal_title: the "
        "title with every unsupported claim removed or corrected, keeping the hook and style (sentence case, 1-3 "
        "CAPS words, optional emoji, max 60 characters). If the title is already literally true, repeat it "
        "exactly.\n" + LITERAL_TITLE_RULES + "\n" + (FRAMES_NOTE + "\n" if with_frames else "")
        + "\nTitles:\n" + json.dumps(items, ensure_ascii=False, indent=1)
    )


RUBRIC_WEIGHTS = {"event_visible": 30, "clear_reaction": 25, "standalone": 20, "payoff": 25}


def rubric_score(verdict: dict) -> float:
    """0-100 from the critic's four yes/no checks."""
    return float(sum(weight for key, weight in RUBRIC_WEIGHTS.items() if bool(verdict.get(key, False))))


def _claims(raw) -> list[dict]:
    claims = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict) and str(item.get("claim", "")).strip():
            claims.append({"claim": str(item.get("claim", "")).strip()[:120],
                           "evidence": str(item.get("evidence", "")).strip()[:160],
                           "literally_true": bool(item.get("literally_true", False))})
    return claims


def critique(items: list[dict], *, url: str, model: str, game_profile: str = "Auto / mixed games",
             creator: str = "", log: Optional[Callable[[str], None]] = None,
             batch_size: Optional[int] = None, images: Optional[dict[int, str]] = None) -> dict[int, dict]:
    """Ask the text model to re-judge finalists. Returns {index: verdict}; missing indexes were not judged.

    A title counts as supported only if every claim the critic listed is literally true; a claim
    list with a false entry overrides a careless ``title_supported: true``.

    With frames attached, each clip is judged in its own call (``batch_size`` 1): in testing, a batch of
    clips made gemma4:12b mark whole batches all-yes or all-no; one at a time its checks were
    consistent. Text-only judging keeps batches of 6 so the clips are compared side by side.
    """
    verdicts: dict[int, dict] = {}
    if batch_size is None:
        batch_size = 1 if images else 6
    for offset in range(0, len(items), batch_size):
        batch = items[offset:offset + batch_size]
        try:
            sheets = [images.get(item["index"]) for item in batch] if images else []
            with_frames = bool(sheets) and all(sheets)
            data = ollama_json(url, model, critic_prompt(batch, game_profile, creator, with_frames), CRITIC_SCHEMA,
                               images=sheets if with_frames else None,
                               temperature=0.0, num_ctx=16384, num_predict=2600, timeout=900)
        except Exception as exc:  # noqa: BLE001 - the critic is an extra filter, never a hard dependency
            if log:
                log(f"Critic pass failed for {len(batch)} clip(s): {exc}")
            continue
        valid = {item["index"] for item in batch}
        for verdict in data.get("verdicts", []) or []:
            try:
                index = int(verdict.get("index", -1))
                if index not in valid:
                    continue
                claims = _claims(verdict.get("title_claims"))
                supported = bool(verdict.get("title_supported", True)) and all(c["literally_true"] for c in claims)
                verdicts[index] = {
                    # Small local models are better at concrete yes/no checks than at free 0-100 scores,
                    # so the score blends both: a clip needs a visible event or a clear reaction to rank.
                    "stranger_score": round(0.5 * max(0.0, min(100.0, float(verdict.get("stranger_score", 0))))
                                            + 0.5 * rubric_score(verdict), 1),
                    "rubric": {k: bool(verdict.get(k, False)) for k in RUBRIC_WEIGHTS},
                    "keep": bool(verdict.get("keep", False)),
                    "title_supported": supported,
                    "claims": claims,
                    "fixed_title": tidy_hook_title(verdict.get("fixed_title", "")),
                    "frames_show": str(verdict.get("frames_show", "")).strip()[:400],
                    "what_happens": str(verdict.get("what_happens", "")).strip()[:200],
                    "reason": str(verdict.get("reason", "")).strip()[:200],
                }
            except (TypeError, ValueError):
                continue
    return verdicts


def verify_titles(items: list[dict], *, url: str, model: str, creator: str = "",
                  log: Optional[Callable[[str], None]] = None,
                  images: Optional[dict[int, str]] = None) -> dict[int, dict]:
    """Claim-by-claim fact-check of titles. ``items`` hold index, title, evidence and check_words.

    Returns {index: {"title", "claims", "ok"}} where ``ok`` means every claim was literally true and
    ``title`` is the literal version. With frames attached each title is checked in its own call.
    """
    if not items:
        return {}
    groups = [[item] for item in items] if images else [items]
    out: dict[int, dict] = {}
    for group in groups:
        sheets = [images.get(item["index"]) for item in group] if images else []
        with_frames = bool(sheets) and all(sheets)
        try:
            data = ollama_json(url, model, verify_prompt(group, creator, with_frames), VERIFY_SCHEMA,
                               images=sheets if with_frames else None,
                               temperature=0.0, num_ctx=12288, num_predict=1800, timeout=600)
        except Exception as exc:  # noqa: BLE001
            if log:
                log(f"Title fact-check failed: {exc}")
            continue
        valid = {item["index"] for item in group}
        for row in data.get("titles", []) or []:
            try:
                index = int(row.get("index", -1))
            except (TypeError, ValueError):
                continue
            if index not in valid:
                continue
            claims = _claims(row.get("claims"))
            out[index] = {"title": tidy_hook_title(row.get("literal_title", "")), "claims": claims,
                          "ok": bool(claims) and all(c["literally_true"] for c in claims)}
    return out