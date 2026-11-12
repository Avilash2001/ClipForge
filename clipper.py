from __future__ import annotations

import argparse
from pathlib import Path

from clipper_engine import APP_VERSION, DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, process_job


def report(stage, stage_pct, overall_pct, message):
    print(f"[{overall_pct:5.1f}%] {stage}: {message}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=f"ClipForge {APP_VERSION} · Local gaming VOD clipper")
    parser.add_argument("--version", action="version", version=f"ClipForge {APP_VERSION}")
    parser.add_argument("source", help="YouTube URL or local video file")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--clip-length", type=int, default=35)
    parser.add_argument("--stride", type=int, default=15)
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
    parser.add_argument("--layout", default="Fit + blurred background",
                        choices=["Fit + blurred background", "Fill (center crop)"])
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
        progress=report, log=lambda m: print(m, flush=True)
    )
    print(f"\nDone. Clips saved to: {result}")


if __name__ == "__main__":
    main()
