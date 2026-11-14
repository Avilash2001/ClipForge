from __future__ import annotations

import argparse
from pathlib import Path

from clip_style import FACECAM_AUTO, FACECAM_OPTIONS, LAYOUT_AUTO, LAYOUTS
from clipper_engine import APP_VERSION, DEFAULT_INPUT_DIR, DEFAULT_MIN_SCORE, DEFAULT_OUTPUT_DIR, process_job


def report(stage, stage_pct, overall_pct, message):
    print(f"[{overall_pct:5.1f}%] {stage}: {message}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=f"ClipForge {APP_VERSION} · Local gaming VOD clipper")
    parser.add_argument("--version", action="version", version=f"ClipForge {APP_VERSION}")
    parser.add_argument("source", help="YouTube URL or local video file")
    parser.add_argument("--count", type=int, default=None, help="Max clips (default: auto)")
    parser.add_argument("--clip-length", type=int, default=None,
                        help="Max clip length in seconds (default: auto, 20-60s moments)")
    parser.add_argument("--stride", type=int, default=None, help="Fallback scan interval (default: auto)")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                        help="Keep only moments scoring at least this (0-100); -1 disables the filter")
    parser.add_argument("--facecam", default=FACECAM_AUTO, choices=FACECAM_OPTIONS)
    parser.add_argument("--creator", default="", help="Streamer name for hook titles (default: YouTube channel)")
    parser.add_argument("--no-title", action="store_true", help="Do not burn the hook title on the video")
    parser.add_argument("--no-chat", action="store_true", help="Do not use YouTube live-chat replay spikes")
    parser.add_argument("--picker-model", default=None,
                        help="Ollama text model for moment picking and the critic (default: best installed, e.g. gemma4:12b)")
    parser.add_argument("--fresh-transcript", action="store_true",
                        help="Re-run Whisper even if this video was transcribed before")
    parser.add_argument("--whisper-model", default="small", choices=["tiny", "base", "small", "medium"])
    parser.add_argument("--whisper-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--ollama-model", default="qwen3.5:9b")
    parser.add_argument("--no-ollama", action="store_true")
    parser.add_argument("--cpu-encode", action="store_true")
    parser.add_argument("--no-captions", action="store_true", help="Do not burn word-highlight captions")
    parser.add_argument("--no-normalize-audio", action="store_true")
    parser.add_argument("--language-mode", default="Mixed (English + Bengali + Hindi)",
                        choices=["Mixed (English + Bengali + Hindi)", "English", "Bengali", "Hindi"])
    parser.add_argument("--game-profile", default="Auto / mixed games")
    parser.add_argument("--layout", default=LAYOUT_AUTO, choices=LAYOUTS)
    parser.add_argument("--cookies-browser", default=None,
                        choices=[None, "chrome", "edge", "brave", "firefox", "vivaldi", "opera"])
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    result = process_job(
        args.source, Path(args.output), DEFAULT_INPUT_DIR,
        count=args.count, clip_length=args.clip_length, stride=args.stride,
        whisper_model=args.whisper_model, whisper_device=args.whisper_device,
        ollama_model=args.ollama_model, use_ollama=not args.no_ollama,
        use_nvenc=not args.cpu_encode,
        language_mode=args.language_mode, game_profile=args.game_profile,
        render_layout=args.layout, burn_captions=not args.no_captions,
        normalize_audio=not args.no_normalize_audio, cookies_browser=args.cookies_browser,
        facecam_mode=args.facecam, min_score=None if args.min_score < 0 else args.min_score,
        creator_name=args.creator, burn_title=not args.no_title, use_chat=not args.no_chat,
        picker_model=args.picker_model, use_transcript_cache=not args.fresh_transcript,
        progress=report, log=lambda m: print(m, flush=True)
    )
    print(f"\nDone. Clips saved to: {result}")


if __name__ == "__main__":
    main()
