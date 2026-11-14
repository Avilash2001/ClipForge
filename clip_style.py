"""Short-form look: facecam detection, vertical layouts, hook title cards and bold captions.

Output is always 1080x1920. Layouts:

* Facecam top + gameplay: the streamer's webcam box fills the top band, centre-cropped
  gameplay fills the bottom, the hook title sits on the seam.
* Gameplay zoom: no facecam; gameplay is centre-cropped to fill most of the frame
  over a blurred copy of itself.
* Co-op split-screen (stacked): side-by-side co-op games (one half per player) are
  re-stacked vertically so neither player's view is cut through the middle.
* Fit + blurred background / Fill (center crop): the original v1 layouts.
"""
from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

OUT_W, OUT_H = 1080, 1920

LAYOUT_AUTO = "Auto (facecam if found)"
LAYOUT_SPLIT = "Facecam top + gameplay"
LAYOUT_ZOOM = "Gameplay zoom"
LAYOUT_FIT = "Fit + blurred background"
LAYOUT_FILL = "Fill (center crop)"
LAYOUT_COOP = "Co-op split-screen (stacked)"
LAYOUTS = [LAYOUT_AUTO, LAYOUT_SPLIT, LAYOUT_ZOOM, LAYOUT_COOP, LAYOUT_FIT, LAYOUT_FILL]

FACECAM_AUTO = "Auto-detect"
FACECAM_OFF = "None"
FACECAM_CORNERS = ["Top-left", "Top-right", "Bottom-left", "Bottom-right"]
FACECAM_OPTIONS = [FACECAM_AUTO, FACECAM_OFF, *FACECAM_CORNERS]

# Split layout geometry (facecam band on top, gameplay below).
SPLIT_TOP_H = 740
SPLIT_BOTTOM_H = OUT_H - SPLIT_TOP_H
# Gameplay-zoom geometry.
ZOOM_GAME_H = 1180
ZOOM_GAME_Y = 370


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    face_cx: Optional[float] = None
    face_cy: Optional[float] = None

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass
class LayoutPlan:
    name: str
    filter: str          # filter_complex fragment from [0:v] to [base]
    title_center_y: int  # where the hook title card is centred
    caption_y: int       # where captions are centred
    facecam: Optional[Box] = None


def _even(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

def _font_dirs() -> list[Path]:
    if os.name == "nt":
        windir = Path(os.environ.get("WINDIR", "C:/Windows"))
        local = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Windows/Fonts"
        return [windir / "Fonts", local]
    if sys.platform == "darwin":
        return [Path("/System/Library/Fonts/Supplemental"), Path("/System/Library/Fonts"),
                Path("/Library/Fonts"), Path.home() / "Library/Fonts"]
    return [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".local/share/fonts",
            Path.home() / ".fonts"]


def find_font(names: list[str]) -> Optional[Path]:
    dirs = [d for d in _font_dirs() if d.is_dir()]
    for name in names:
        for directory in dirs:
            direct = directory / name
            if direct.is_file():
                return direct
        for directory in dirs:
            try:
                match = next(directory.rglob(name), None)
            except OSError:
                match = None
            if match is not None:
                return match
    return None


TITLE_FONT_FILES = ["arialbd.ttf", "Arial Bold.ttf", "Arial-BoldMT.ttf", "LiberationSans-Bold.ttf",
                    "DejaVuSans-Bold.ttf", "NotoSans-Bold.ttf", "Roboto-Bold.ttf"]
EMOJI_FONT_FILES = ["seguiemj.ttf", "Apple Color Emoji.ttc", "NotoColorEmoji.ttf"]


def caption_font_family() -> str:
    """Heavy font family for libass captions (resolved by the OS font provider)."""
    if find_font(["ariblk.ttf", "Arial Black.ttf"]):
        return "Arial Black"
    if find_font(["Impact.ttf", "impact.ttf"]):
        return "Impact"
    return "DejaVu Sans"


# ---------------------------------------------------------------------------
# Hook title card (rendered with Pillow, overlaid with FFmpeg - no libass needed)
# ---------------------------------------------------------------------------

def _is_emoji(ch: str) -> bool:
    cp = ord(ch)
    return (0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF or 0x2B00 <= cp <= 0x2BFF
            or cp in (0x2122, 0x2139, 0x3030, 0x303D, 0x00A9, 0x00AE))


def _strip_invisible(text: str) -> str:
    return re.sub("[\u200d\ufe0e\ufe0f]", "", text)


def _runs(text: str) -> list[tuple[str, bool]]:
    runs: list[tuple[str, bool]] = []
    for ch in text:
        emoji = _is_emoji(ch)
        if runs and runs[-1][1] == emoji and not emoji:
            runs[-1] = (runs[-1][0] + ch, False)
        else:
            runs.append((ch, emoji))
    return runs


def _render_emoji(ch: str, size: int, emoji_font: Optional[Path]):
    from PIL import Image, ImageDraw, ImageFont

    if emoji_font is None:
        return None
    for font_size in (size, 109, 160, 96, 64, 48):
        try:
            font = ImageFont.truetype(str(emoji_font), font_size)
        except OSError:
            continue
        try:
            canvas = Image.new("RGBA", (font_size * 2, font_size * 2), (0, 0, 0, 0))
            ImageDraw.Draw(canvas).text((font_size // 4, font_size // 4), ch, font=font, embedded_color=True)
            bbox = canvas.getbbox()
            if not bbox:
                return None
            glyph = canvas.crop(bbox)
            scale = size / max(glyph.width, glyph.height)
            return glyph.resize((max(1, int(glyph.width * scale)), max(1, int(glyph.height * scale))),
                                Image.LANCZOS)
        except OSError:
            continue
    return None


def render_title_card(text: str, out_path: Path, *, max_width: int = 900, font_size: int = 58,
                      max_lines: int = 3) -> Optional[tuple[int, int]]:
    """Render a white rounded "sticker" title card like the popular clipping style.

    Returns the PNG size, or None when there is no usable text or no font.
    """
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    text = _strip_invisible(re.sub(r"\s+", " ", str(text or "")).strip())
    if not any(ch.isalnum() for ch in text):
        return None
    font_path = find_font(TITLE_FONT_FILES)
    emoji_font = find_font(EMOJI_FONT_FILES)
    if emoji_font is None:
        text = "".join(ch for ch in text if not _is_emoji(ch)).strip()

    pad_x, pad_y, radius, border, shadow = 34, 22, 26, 4, 18
    inner_max = max_width - 2 * pad_x

    def load(size: int):
        if font_path is None:
            return ImageFont.load_default(size=size)
        return ImageFont.truetype(str(font_path), size)

    def width_of(fragment: str, font, size: int) -> float:
        total = 0.0
        for run, emoji in _runs(fragment):
            total += size * 1.12 * len(run) if emoji else font.getlength(run)
        return total

    # Keep a trailing emoji glued to the word before it so it never sits alone on a line.
    tokens: list[str] = []
    for word in text.split(" "):
        if tokens and word and all(_is_emoji(ch) for ch in word):
            tokens[-1] = f"{tokens[-1]} {word}"
        elif word:
            tokens.append(word)

    def greedy(font, size: int, limit: float) -> list[str]:
        lines: list[str] = []
        current = ""
        for word in tokens:
            trial = f"{current} {word}".strip()
            if width_of(trial, font, size) <= limit or not current:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def wrap(size: int):
        font = load(size)
        lines = greedy(font, size, inner_max)
        # Balance the lines: shrink the wrap width while the line count stays the same.
        limit = inner_max
        while len(lines) > 1 and limit > inner_max * 0.45:
            trial = greedy(font, size, limit - 16)
            if len(trial) != len(lines):
                break
            lines, limit = trial, limit - 16
        return font, lines

    size = font_size
    font, lines = wrap(size)
    while (len(lines) > max_lines or any(width_of(line, font, size) > inner_max for line in lines)) and size > 38:
        size -= 3
        font, lines = wrap(size)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while lines[-1] and width_of(lines[-1] + "…", font, size) > inner_max:
            lines[-1] = lines[-1][:-1]
        lines[-1] = lines[-1].rstrip() + "…"

    ascent, descent = font.getmetrics()
    line_h = int((ascent + descent) * 1.06)
    text_w = int(max(width_of(line, font, size) for line in lines))
    box_w = text_w + 2 * pad_x
    box_h = line_h * len(lines) + 2 * pad_y
    img_w, img_h = box_w + 2 * shadow, box_h + 2 * shadow

    image = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    shade = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    ImageDraw.Draw(shade).rounded_rectangle(
        (shadow, shadow + 6, shadow + box_w, shadow + box_h + 6), radius=radius, fill=(0, 0, 0, 120))
    image = Image.alpha_composite(image, shade.filter(ImageFilter.GaussianBlur(9)))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((shadow, shadow, shadow + box_w, shadow + box_h), radius=radius,
                           fill=(255, 255, 255, 255), outline=(12, 12, 12, 255), width=border)
    y = shadow + pad_y
    for line in lines:
        x = shadow + (box_w - width_of(line, font, size)) / 2
        for run, emoji in _runs(line):
            if emoji:
                for ch in run:
                    glyph = _render_emoji(ch, int(size * 0.98), emoji_font)
                    if glyph is not None:
                        image.alpha_composite(glyph, (int(x + size * 0.07), int(y + (line_h - glyph.height) / 2)))
                    x += size * 1.12
            else:
                draw.text((x, y + (line_h - (ascent + descent)) / 2), run, font=font, fill=(10, 10, 10, 255))
                x += font.getlength(run)
        y += line_h
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return img_w, img_h


# ---------------------------------------------------------------------------
# Word-by-word captions (ASS, burned in with libass)
# ---------------------------------------------------------------------------

_CAPTION_WHITE = "&HFFFFFF&"
_CAPTION_YELLOW = "&H00E1FF&"  # ASS colours are BGR: RGB(255, 225, 0)


def _ass_time(seconds: float) -> str:
    cs = max(0, int(round(seconds * 100)))
    hours, cs = divmod(cs, 360000)
    minutes, cs = divmod(cs, 6000)
    secs, cs = divmod(cs, 100)
    return f"{hours}:{minutes:02}:{secs:02}.{cs:02}"


def _caption_word(raw: str) -> str:
    word = re.sub(r"^[^\w']+|[^\w']+$", "", str(raw).strip(), flags=re.UNICODE)
    word = word.replace("\\", "").replace("{", "").replace("}", "")
    return word.upper()


def caption_groups(rows: list[dict], start: float, end: float, *, max_words: int = 4,
                   max_chars: int = 16, max_gap: float = 0.6) -> list[list[dict]]:
    """Split clip words into short punchy caption groups (2-4 words)."""
    words: list[dict] = []
    for seg in rows:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        for w in seg.get("words") or []:
            text = _caption_word(w.get("word", ""))
            if not text or not any(ch.isalnum() for ch in text):
                continue
            ws, we = float(w["start"]), float(w["end"])
            if we <= start or ws >= end:
                continue
            words.append({"start": max(0.0, ws - start), "end": min(end - start, we - start),
                          "text": text, "raw": str(w.get("word", "")).strip()})
    words.sort(key=lambda w: w["start"])
    groups: list[list[dict]] = []
    current: list[dict] = []
    chars = 0
    for w in words:
        boundary = bool(current) and (
            len(current) >= max_words
            or chars + 1 + len(w["text"]) > max_chars
            or w["start"] - current[-1]["end"] > max_gap
            or re.search(r"[.!?,…।]$", current[-1]["raw"])
        )
        if boundary:
            groups.append(current)
            current, chars = [], 0
        current.append(w)
        chars += len(w["text"]) + (1 if chars else 0)
    if current:
        groups.append(current)
    return groups


def write_bold_captions(rows: list[dict], start: float, end: float, path: Path, *,
                        caption_y: int, font_family: Optional[str] = None,
                        font_size: int = 96) -> bool:
    """Write big uppercase captions where the spoken word lights up yellow."""
    groups = caption_groups(rows, start, end)
    if not groups:
        return False
    family = font_family or caption_font_family()
    clip_len = end - start
    events: list[tuple[float, float, str]] = []
    for g_index, group in enumerate(groups):
        next_start = groups[g_index + 1][0]["start"] if g_index + 1 < len(groups) else clip_len
        last_end = group[-1]["end"]
        # Hold the caption until the next group if the pause is short (avoids flicker).
        g_end = next_start if next_start - last_end < 0.45 else min(next_start, last_end + 0.25)
        g_end = min(g_end, clip_len)
        for i, word in enumerate(group):
            ev_start = group[0]["start"] if i == 0 else word["start"]
            ev_end = group[i + 1]["start"] if i + 1 < len(group) else g_end
            if ev_end - ev_start < 0.04:
                ev_end = ev_start + 0.04
            parts = []
            for j, other in enumerate(group):
                colour = _CAPTION_YELLOW if j == i else _CAPTION_WHITE
                parts.append(f"{{\\c{colour}}}{other['text']}")
            pop = "\\fscx82\\fscy82\\t(0,90,\\fscx100\\fscy100)" if i == 0 else ""
            events.append((ev_start, ev_end, f"{{\\an5\\pos({OUT_W // 2},{caption_y}){pop}}}" + " ".join(parts)))
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\nWrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding\n"
        f"Style: Pop,{family},{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,1,0,1,7,3,5,"
        "70,70,0,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    with path.open("w", encoding="utf-8-sig") as handle:
        handle.write(header)
        for ev_start, ev_end, text in events:
            handle.write(f"Dialogue: 0,{_ass_time(ev_start)},{_ass_time(ev_end)},Pop,,0,0,0,,{text}\n")
    return True


# ---------------------------------------------------------------------------
# Facecam detection
# ---------------------------------------------------------------------------

def corner_facecam(corner: str, src_w: int, src_h: int) -> Optional[Box]:
    if corner not in FACECAM_CORNERS:
        return None
    w, h = _even(src_w * 0.27), _even(src_h * 0.30)
    x = 0 if "left" in corner.lower() else src_w - w
    y = 0 if corner.lower().startswith("top") else src_h - h
    return Box(x, y, w, h)


def _snap_side(edges, fixed_a: int, fixed_b: int, search: range, vertical: bool, threshold: float = 0.33):
    """Find the strongest persistent straight edge (an overlay border) in a search range."""
    best_pos, best_val = None, 0.0
    for pos in search:
        line = edges[fixed_a:fixed_b, pos] if vertical else edges[pos, fixed_a:fixed_b]
        if line.size == 0:
            continue
        value = float(line.mean())
        if value > best_val:
            best_pos, best_val = pos, value
    return best_pos if best_val >= threshold else None


def detect_facecam(video: Path, duration: float, src_w: int, src_h: int, work_dir: Path,
                   run_process: Callable, log: Optional[Callable[[str], None]] = None,
                   samples: int = 24) -> Optional[Box]:
    """Find a streamer webcam overlay: a face that stays in the same place across the stream.

    Samples frames across the video, runs OpenCV's bundled frontal-face detector,
    keeps the face cluster that is present in many frames at a fixed position, then
    estimates the webcam box around it and snaps its sides to persistent overlay
    borders (or the frame edge).
    """
    import cv2
    import numpy as np

    work_dir.mkdir(parents=True, exist_ok=True)
    scale_w = 960
    frames = []
    for i in range(samples):
        t = duration * (0.04 + 0.92 * i / max(1, samples - 1))
        out = work_dir / f"cam_{i:03d}.jpg"
        run_process(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(video),
                     "-frames:v", "1", "-vf", f"scale={scale_w}:-2", "-q:v", "3", str(out)],
                    check=False, capture_output=True, text=True)
        image = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            frames.append(image)
    if len(frames) < 6:
        return None
    fh, fw = frames[0].shape[:2]
    if not hasattr(cv2, "CascadeClassifier"):
        if log:
            log("Facecam: this OpenCV build has no face detector (OpenCV 5+). Re-run setup to install "
                "opencv-python-headless<5, or pick the facecam corner manually.")
        return None
    cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    if cascade.empty():
        return None
    detections = []
    for index, gray in enumerate(frames):
        if gray.shape[:2] != (fh, fw):
            continue
        eq = cv2.equalizeHist(gray)
        found = cascade.detectMultiScale(eq, scaleFactor=1.08, minNeighbors=6, minSize=(22, 22),
                                         maxSize=(int(fh * 0.6), int(fh * 0.6)))
        for (x, y, w, h) in found:
            detections.append((index, float(x), float(y), float(w), float(h)))
    clusters: list[list[tuple]] = []
    for det in detections:
        _, x, y, w, h = det
        cx, cy = x + w / 2, y + h / 2
        for cluster in clusters:
            mcx = sum(d[1] + d[3] / 2 for d in cluster) / len(cluster)
            mcy = sum(d[2] + d[4] / 2 for d in cluster) / len(cluster)
            mw = sum(d[3] for d in cluster) / len(cluster)
            if math.hypot(cx - mcx, cy - mcy) < 0.7 * max(w, mw) and 0.6 < w / mw < 1.66:
                cluster.append(det)
                break
        else:
            clusters.append([det])
    need = max(4, math.ceil(0.3 * len(frames)))
    scored = []
    for cluster in clusters:
        support = len({d[0] for d in cluster})
        if support < need:
            continue
        xs = [d[1] + d[3] / 2 for d in cluster]
        spread = float(np.std(xs)) / max(1.0, float(np.median([d[3] for d in cluster])))
        scored.append((support, -spread, cluster))
    if not scored:
        if log:
            log("Facecam: no stable webcam face found; using the no-facecam layout.")
        return None
    support, _, cluster = max(scored, key=lambda item: (item[0], item[1]))
    fx = float(np.median([d[1] for d in cluster]))
    fy = float(np.median([d[2] for d in cluster]))
    fwid = float(np.median([d[3] for d in cluster]))
    fhei = float(np.median([d[4] for d in cluster]))
    fcx, fcy = fx + fwid / 2, fy + fhei / 2

    edges = np.mean([cv2.Canny(g, 60, 160).astype(np.float32) / 255.0
                     for g in frames if g.shape[:2] == (fh, fw)], axis=0)
    y_a, y_b = int(_clamp(fy - 0.6 * fhei, 0, fh)), int(_clamp(fy + 1.6 * fhei, 0, fh))
    x_a, x_b = int(_clamp(fx - 0.6 * fwid, 0, fw)), int(_clamp(fx + 1.6 * fwid, 0, fw))

    guess_left = fcx - 1.75 * fwid
    guess_right = fcx + 1.75 * fwid
    guess_top = fcy - 1.15 * fhei
    guess_bottom = fcy + 1.55 * fhei

    left = _snap_side(edges, y_a, y_b, range(int(_clamp(fx - 2.4 * fwid, 0, fw)), int(_clamp(fx - 0.3 * fwid, 0, fw))), True)
    right = _snap_side(edges, y_a, y_b, range(int(_clamp(fx + 1.3 * fwid, 0, fw)), int(_clamp(fx + 3.4 * fwid, 0, fw))), True)
    top = _snap_side(edges, x_a, x_b, range(int(_clamp(fy - 1.8 * fhei, 0, fh)), int(_clamp(fy - 0.2 * fhei, 0, fh))), False)
    bottom = _snap_side(edges, x_a, x_b, range(int(_clamp(fy + 1.4 * fhei, 0, fh)), int(_clamp(fy + 3.2 * fhei, 0, fh))), False)

    # No border found: webcams are usually flush with the frame edge when the guess is close to it.
    def settle_low(found, guess, size):
        if found is not None:
            return float(found)
        return 0.0 if guess < 0.06 * size else guess

    def settle_high(found, guess, size):
        if found is not None:
            return float(found)
        return float(size) if guess > 0.94 * size else guess

    bx0 = _clamp(settle_low(left, guess_left, fw), 0, fw)
    bx1 = _clamp(settle_high(right, guess_right, fw), 0, fw)
    by0 = _clamp(settle_low(top, guess_top, fh), 0, fh)
    by1 = _clamp(settle_high(bottom, guess_bottom, fh), 0, fh)
    bw, bh = bx1 - bx0, by1 - by0
    if not (1.6 * fwid <= bw <= 6.5 * fwid and 1.5 * fhei <= bh <= 5.5 * fhei):
        bx0, bx1 = _clamp(guess_left, 0, fw), _clamp(guess_right, 0, fw)
        by0, by1 = _clamp(guess_top, 0, fh), _clamp(guess_bottom, 0, fh)
        bw, bh = bx1 - bx0, by1 - by0
    k = src_w / float(fw)
    box = Box(_even(bx0 * k), _even(by0 * k), _even(bw * k), _even(bh * k), face_cx=fcx * k, face_cy=fcy * k)
    box.w = min(box.w, src_w - box.x)
    box.h = min(box.h, src_h - box.y)
    if box.w < src_w * 0.08 or box.w > src_w * 0.65:
        if log:
            log("Facecam: detected face box had an implausible size; using the no-facecam layout.")
        return None
    if log:
        log(f"Facecam detected in {support}/{len(frames)} sampled frames at x={box.x}, y={box.y}, "
            f"{box.w}x{box.h}.")
    return box


# ---------------------------------------------------------------------------
# Layout filter graphs
# ---------------------------------------------------------------------------

def _blur_background(src_label: str, out_label: str) -> str:
    # Blurring at quarter resolution is ~10x faster than boxblur at 1080x1920 and looks smoother.
    return (f"[{src_label}]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,"
            f"boxblur=10:2,scale={OUT_W}:{OUT_H}:flags=bilinear,eq=brightness=-0.10:saturation=1.15,"
            f"setsar=1[{out_label}]")


def _centre_crop(src_w: int, src_h: int, aspect: float, band: Optional[tuple[int, int]] = None,
                 x_range: Optional[tuple[int, int]] = None) -> tuple[int, int, int, int]:
    """Largest centred crop with the given aspect inside an optional column range / row band."""
    x0, x1 = x_range or (0, src_w)
    y0, y1 = band or (0, src_h)
    area_w, area_h = x1 - x0, y1 - y0
    if area_w / area_h > aspect:
        h = _even(area_h)
        w = _even(min(area_w, area_h * aspect))
    else:
        w = _even(area_w)
        h = _even(min(area_h, area_w / aspect))
    x = x0 + max(0.0, (area_w - w) / 2)
    y = y0 + max(0.0, (area_h - h) / 2)
    return w, h, _even(x) if x > 1 else 0, _even(y) if y > 1 else 0


def _cam_crop(box: Box, aspect: float) -> tuple[int, int, int, int]:
    face_cx = box.face_cx if box.face_cx is not None else box.x + box.w / 2
    face_cy = box.face_cy if box.face_cy is not None else box.y + box.h * 0.45
    if box.w / box.h > aspect:
        ch = box.h
        cw = ch * aspect
        cx = _clamp(face_cx - cw / 2, box.x, box.x + box.w - cw)
        cy = box.y
    else:
        cw = box.w
        ch = cw / aspect
        cx = box.x
        cy = _clamp(face_cy - 0.45 * ch, box.y, box.y + box.h - ch)
    return _even(cw), _even(ch), _even(cx) if cx > 1 else 0, _even(cy) if cy > 1 else 0


def clip_frame_traits(frame_paths: list[Path]) -> dict:
    """Inspect a clip's 1 fps analysis frames for split-screen co-op and letterbox bars.

    Returns {"split_screen": bool, "band": (top_fraction, bottom_fraction)} where band is
    the vertical extent of real picture content (black letterbox bars excluded).
    """
    import cv2
    import numpy as np

    seam_votes, considered = 0, 0
    letterboxed: list[tuple[float, float]] = []
    for path in frame_paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        gray = gray.astype(np.float32)
        h, w = gray.shape[:2]
        if float(gray.mean()) < 18:  # black/fade frames say nothing about the layout
            continue
        considered += 1
        # A letterbox row is dark across most of its width (small overlays/subtitles don't count).
        content = np.where(np.median(gray, axis=1) >= 22)[0]
        if content.size:
            top, bottom = content[0] / h, (content[-1] + 1) / h
            if 0.04 <= top <= 0.33 or 0.67 <= bottom <= 0.96:
                letterboxed.append((top, bottom))
        dx = np.abs(np.diff(gray, axis=1))
        c = w // 2 - 1
        centre = dx[:, c - 1:c + 2].max(axis=1)
        others = np.concatenate([dx[:, c - 6:c - 1], dx[:, c + 2:c + 7]], axis=1).max(axis=1)
        if float(np.mean(centre > np.maximum(others * 1.5, 6.0))) >= 0.25:
            seam_votes += 1
    split = considered >= 3 and seam_votes / considered >= 0.5
    band = (0.0, 1.0)
    # Trim bars only when most of the clip is letterboxed (e.g. a cutscene); a fixed crop
    # cannot follow a clip that switches between letterboxed and full-frame shots.
    if considered and len(letterboxed) / considered >= 0.6:
        top = float(np.median([t for t, _ in letterboxed]))
        bottom = float(np.median([b for _, b in letterboxed]))
        band = (top if 0.04 <= top <= 0.33 else 0.0, bottom if 0.67 <= bottom <= 0.96 else 1.0)
    return {"split_screen": bool(split), "band": band}


def plan_layout(layout: str, src_w: int, src_h: int, facecam: Optional[Box] = None, *,
                split_screen: bool = False, band: tuple[float, float] = (0.0, 1.0)) -> LayoutPlan:
    """Return the filter graph and overlay positions for a layout choice."""
    rows = (_even(src_h * band[0]) if band[0] > 0 else 0, _even(src_h * band[1]) if band[1] < 1 else src_h)
    if layout in (LAYOUT_AUTO, LAYOUT_SPLIT) and facecam is not None:
        cw, ch, cx, cy = _cam_crop(facecam, OUT_W / SPLIT_TOP_H)
        gw, gh, gx, gy = _centre_crop(src_w, src_h, OUT_W / SPLIT_BOTTOM_H, rows)
        graph = (
            "[0:v]split=2[cam_src][game_src];"
            f"[cam_src]crop={cw}:{ch}:{cx}:{cy},scale={OUT_W}:{SPLIT_TOP_H}:flags=lanczos,setsar=1[cam];"
            f"[game_src]crop={gw}:{gh}:{gx}:{gy},scale={OUT_W}:{SPLIT_BOTTOM_H}:flags=lanczos,setsar=1[game];"
            "[cam][game]vstack=inputs=2,"
            f"drawbox=x=0:y={SPLIT_TOP_H - 3}:w={OUT_W}:h=6:color=black@0.85:t=fill[base]"
        )
        return LayoutPlan(LAYOUT_SPLIT, graph, title_center_y=SPLIT_TOP_H,
                          caption_y=SPLIT_TOP_H + int(SPLIT_BOTTOM_H * 0.52), facecam=facecam)
    if layout == LAYOUT_COOP or (layout in (LAYOUT_AUTO, LAYOUT_ZOOM, LAYOUT_SPLIT) and split_screen):
        half_h = OUT_H // 2
        half = src_w // 2
        lw, lh, lx, ly = _centre_crop(src_w, src_h, OUT_W / half_h, rows, (0, half))
        rw, rh, rx, ry = _centre_crop(src_w, src_h, OUT_W / half_h, rows, (half, src_w))
        graph = (
            "[0:v]split=2[p1_src][p2_src];"
            f"[p1_src]crop={lw}:{lh}:{lx}:{ly},scale={OUT_W}:{half_h}:flags=lanczos,setsar=1[p1];"
            f"[p2_src]crop={rw}:{rh}:{rx}:{ry},scale={OUT_W}:{half_h}:flags=lanczos,setsar=1[p2];"
            "[p1][p2]vstack=inputs=2,"
            f"drawbox=x=0:y={half_h - 3}:w={OUT_W}:h=6:color=black@0.85:t=fill[base]"
        )
        return LayoutPlan(LAYOUT_COOP, graph, title_center_y=half_h, caption_y=half_h + 330)
    if layout in (LAYOUT_AUTO, LAYOUT_SPLIT, LAYOUT_ZOOM):
        gw, gh, gx, gy = _centre_crop(src_w, src_h, OUT_W / ZOOM_GAME_H, rows)
        graph = (
            "[0:v]split=2[bg_src][game_src];" + _blur_background("bg_src", "bg") + ";"
            f"[game_src]crop={gw}:{gh}:{gx}:{gy},scale={OUT_W}:{ZOOM_GAME_H}:flags=lanczos,setsar=1[game];"
            f"[bg][game]overlay=0:{ZOOM_GAME_Y}[base]"
        )
        return LayoutPlan(LAYOUT_ZOOM, graph, title_center_y=ZOOM_GAME_Y,
                          caption_y=ZOOM_GAME_Y + int(ZOOM_GAME_H * 0.66))
    if layout == LAYOUT_FILL:
        graph = (f"[0:v]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
                 f"crop={OUT_W}:{OUT_H},setsar=1[base]")
        return LayoutPlan(LAYOUT_FILL, graph, title_center_y=330, caption_y=1300)
    graph = (
        "[0:v]split=2[bg_src][fg_src];" + _blur_background("bg_src", "bg") + ";"
        f"[fg_src]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:flags=lanczos,setsar=1[game];"
        "[bg][game]overlay=(W-w)/2:(H-h)/2[base]"
    )
    fit_h = int(OUT_W * src_h / max(1, src_w))
    top = (OUT_H - fit_h) // 2
    return LayoutPlan(LAYOUT_FIT, graph, title_center_y=max(260, top - 120),
                      caption_y=min(OUT_H - 380, top + fit_h + 160))
