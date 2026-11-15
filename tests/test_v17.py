"""Tests for v1.7: GPU Whisper selection/fallback, signal-driven moments, literal titles, section downloads."""
import json
import math
import shutil
import subprocess
import sys
import types
import wave

import pytest

sys.modules.setdefault("yt_dlp", types.ModuleType("yt_dlp"))
fw = types.ModuleType("faster_whisper")
fw.WhisperModel = object
sys.modules.setdefault("faster_whisper", fw)

import asr  # noqa: E402
import clipper_engine as ce  # noqa: E402
import moments  # noqa: E402
import sources  # noqa: E402
from clipper_engine import Candidate  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


# --- Whisper: device/model choice, language limiting, fallback ------------------------------

def test_auto_uses_large_v3_on_gpu_and_small_on_cpu(monkeypatch):
    monkeypatch.setattr(asr, "cuda_status", lambda refresh=False: (True, "ok"))
    gpu = asr.resolve_settings("Auto", "Auto")
    assert (gpu.model, gpu.device, gpu.compute_type) == (asr.GPU_MODEL, "cuda", "float16") and gpu.model_auto
    monkeypatch.setattr(asr, "cuda_status", lambda refresh=False: (False, "no GPU"))
    cpu = asr.resolve_settings("Auto", "Auto")
    assert (cpu.model, cpu.device, cpu.compute_type) == (asr.CPU_MODEL, "cpu", "int8")
    forced = asr.resolve_settings("medium", "cuda")   # CUDA asked for but unusable -> CPU, model kept
    assert (forced.model, forced.device) == ("medium", "cpu")


def test_language_limiter_never_picks_a_language_outside_the_mix():
    probs = [("<|ur|>", 0.55), ("<|hi|>", 0.25), ("<|ja|>", 0.1), ("<|en|>", 0.08), ("<|bn|>", 0.02)]
    out = asr.restrict_language_probs(probs, {"<|en|>", "<|hi|>", "<|bn|>"})
    assert out[0][0] == "<|hi|>" and out[0][1] > 0.6          # Urdu guess becomes Hindi
    assert {tok for tok, _ in out} == {tok for tok, _ in probs}  # nothing is lost

    class Inner:
        is_multilingual = True

        def detect_language(self, _features):
            return [probs]

    limiter = asr._LanguageLimiter(Inner(), ("en", "hi", "bn"))
    assert limiter.is_multilingual and limiter.detect_language(None)[0][0][0] == "<|hi|>"


def test_gpu_failure_falls_back_to_cpu(monkeypatch):
    calls = []

    def fake_run(path, settings, duration, mode, on_progress, check_cancel, log):
        calls.append(settings)
        if settings.device == "cuda":
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
        return [{"start": 0, "end": 1, "text": "hi"}], "en"

    monkeypatch.setattr(asr, "_run_whisper", fake_run)
    settings = asr.WhisperSettings("large-v3", "cuda", "float16", model_auto=True)
    rows, _, used = asr.transcribe("x.wav", settings, 10)
    assert rows and used.device == "cpu" and used.model == asr.CPU_MODEL
    assert [c.device for c in calls] == ["cuda", "cpu"]   # a missing DLL skips the int8 GPU retry


def test_gpu_out_of_memory_retries_int8_on_the_gpu_first(monkeypatch):
    calls = []

    def fake_run(path, settings, *args):
        calls.append(settings.compute_type)
        if settings.compute_type == "float16":
            raise RuntimeError("CUDA failed with error out of memory")
        return [], "hi"

    monkeypatch.setattr(asr, "_run_whisper", fake_run)
    _, _, used = asr.transcribe("x.wav", asr.WhisperSettings("large-v3", "cuda", "float16"), 10)
    assert calls == ["float16", "int8_float16"] and used.device == "cuda"


def test_cuda_library_dirs_find_pip_wheels(tmp_path, monkeypatch):
    wheel_bin = tmp_path / "nvidia" / "cublas" / "bin"
    wheel_bin.mkdir(parents=True)
    monkeypatch.syspath_prepend(str(tmp_path))
    assert wheel_bin in asr.cuda_library_dirs()


def test_whisper_options_follow_the_language_mode():
    mixed = asr.whisper_options(asr.MIXED)
    assert mixed["multilingual"] and "language" not in mixed
    hindi = asr.whisper_options("Hindi")
    assert hindi["language"] == "hi" and not hindi["multilingual"]


def test_transcript_cache_key_includes_device_and_compute(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    cpu = ce.transcript_cache_path(video, "small", "Mixed", "cpu", "int8")
    gpu = ce.transcript_cache_path(video, "small", "Mixed", "cuda", "float16")
    assert cpu != gpu
    assert gpu != ce.transcript_cache_path(video, "large-v3", "Mixed", "cuda", "float16")
    # Audio-only download and full video of the same YouTube stream share one transcript.
    audio = tmp_path / "Stream [KzsStyJEDUw].audio.m4a"
    full = tmp_path / "Stream [KzsStyJEDUw].mp4"
    audio.write_bytes(b"a")
    full.write_bytes(b"bb")
    assert ce.transcript_cache_path(audio, "large-v3", "Mixed", "cuda", "float16", 13634.2) == \
        ce.transcript_cache_path(full, "large-v3", "Mixed", "cuda", "float16", 13634.4)


# --- signal-driven (non-verbal) moments ------------------------------------------------------

def test_motion_spikes_flag_a_burst_not_a_busy_baseline():
    busy = [0.6 + 0.05 * math.sin(i) for i in range(300)]
    assert max(moments.motion_spikes(busy)) < 0.2
    burst = [0.3] * 300
    for i in range(150, 154):
        burst[i] = 1.0
    spikes = moments.motion_spikes(burst)
    assert max(spikes[148:156]) > 0.5 and max(spikes[:120]) == 0


def test_signal_windows_find_a_silent_spike_and_skip_covered_ones():
    n = 1200
    loud = [0.0] * n
    chat = [0.0] * n
    for i in range(400, 404):      # a shout with no words
        loud[i] = 1.0
    for i in range(810, 820):      # chat burst ~8 s after a play at ~802 s
        chat[i] = 0.9
    for i in range(1000, 1003):    # a shout inside a transcript moment
        loud[i] = 1.0
    windows = moments.signal_windows(n, loud, chat, None, exclude=[(985, 1025)])
    peaks = [w.peak for w in windows]
    assert any(395 <= p <= 405 for p in peaks)
    assert any(795 <= p <= 812 for p in peaks)                  # chat lag shifts the window earlier
    assert not any(985 <= p <= 1025 for p in peaks)
    w = next(w for w in windows if 395 <= w.peak <= 405)
    assert w.end - w.start >= 25 and w.start < w.peak < w.end and w.loud >= 0.9


def test_signal_candidates_merge_without_duplicating_transcript_moments():
    moment = Candidate(0, 100, 140, .5, "", 0, origin="moment", moment_score=70)
    windows = [moments.SignalWindow(105, 137, 127, 0.8, loud=0.9),       # same event -> dropped
               moments.SignalWindow(500, 532, 522, 0.6, chat=0.7)]       # silent clutch -> kept
    energies = [0.2] * 700
    signal = ce.candidates_from_signals(windows, 700, energies, [], False, [0.3] * 700, [0.0] * 700,
                                        [0.0] * 700, first_index=1)
    merged = ce.merge_candidates([moment], signal)
    assert [c.origin for c in merged] == ["moment", "signal"]
    assert merged[1].start == 500 and merged[1].index == 2 and merged[1].quality == -1.0
    assert moment.signal_strength == 0.8   # the spike confirmed the transcript moment


def test_rejected_clips_are_never_chosen():
    good = Candidate(0, 0, 30, .5, "", 0, final_score=.5, quality=70)
    rejected = Candidate(1, 100, 130, .5, "", 0, final_score=.9, quality=80, rejected="edge check: menu")
    assert [c.index for c in ce.choose_top([good, rejected], 3, 60)] == [0]
    assert ce.choose_top([rejected], 1) == []


# --- literal titles ---------------------------------------------------------------------------

def test_title_risks_flag_hype_quantifiers_and_pronouns():
    assert moments.title_risks("AVIXION gets ambushed by the WHOLE school 💀", "a group of students attack") == ["whole"]
    assert moments.title_risks("The most ABSURD insults in the game", "insults") == ["most"]
    assert moments.title_risks("Craziest flick EVER", "") == ["craziest", "ever"]
    assert moments.title_risks("AVIXION loses his MIND", "") == ["loses", "his"]   # feelings are claims too
    assert moments.title_risks("Squad gets SNIPED through a wall", "the squad got sniped") == []
    # Final gate: quantifiers the evidence literally contains are fine; pronouns are not hype.
    assert moments.title_risks("Every door is LOCKED", "every door is locked", strict=True) == []
    assert moments.title_risks("Guard loses his TEMPER", "", strict=True) == []


def test_critic_claims_override_a_careless_supported_flag(monkeypatch):
    reply = {"verdicts": [{
        "index": 3, "what_happens": "A group of students attack Avixion.", "stranger_score": 80,
        "title_claims": [{"claim": "Avixion is ambushed", "evidence": "students attack", "literally_true": True},
                         {"claim": "by the whole school", "evidence": "a group", "literally_true": False}],
        "title_supported": True, "fixed_title": "Avixion gets JUMPED by a gang of students", "keep": True,
        "reason": "clear payoff"}]}
    monkeypatch.setattr(moments, "ollama_json", lambda *a, **k: reply)
    verdict = moments.critique([{"index": 3, "title": "Avixion gets ambushed by the WHOLE school"}],
                               url="u", model="m")[3]
    assert verdict["title_supported"] is False
    assert verdict["fixed_title"] == "Avixion gets JUMPED by a gang of students"


def test_fact_check_removes_titles_that_stay_hyped(monkeypatch):
    clips = [Candidate(0, 0, 20, .5, "", 0, hook_title="The most ABSURD insults in the game"),
             Candidate(1, 50, 70, .5, "", 0, hook_title="Avixion gets ambushed by the WHOLE school")]
    monkeypatch.setattr(ce, "verify_titles", lambda items, **k: {
        0: {"title": "The most ABSURD insults in the game", "claims": [], "ok": False},
        1: {"title": "Avixion gets JUMPED by students", "claims": [], "ok": False}})
    ce.fact_check_titles(clips, [], model="m", creator="Avixion")
    assert clips[0].hook_title == ""                                  # still a superlative -> removed
    assert clips[1].hook_title == "Avixion gets JUMPED by students"   # literal rewrite kept


# --- audio-first URL flow ---------------------------------------------------------------------

def test_section_ranges_are_padded_and_merged():
    ranges = sources.merge_ranges([(100, 130), (150, 170), (1000, 1040)], duration=1030, margin=12)
    assert ranges == [(88.0, 182.0), (988.0, 1030.0)]


def test_audio_alignment_finds_the_section_offset():
    import numpy as np

    rng = np.random.default_rng(3)
    rate = 16000
    full = rng.standard_normal(rate * 120).astype(np.float32) * 0.1
    true_start = 47.37
    probe = full[int(true_start * rate):int((true_start + 20) * rate)]
    offset, confidence = sources.find_offset(full, rate, probe, approx_start=45.0)
    assert abs(offset - true_start) < 0.01 and confidence > 0.9


def test_section_lookup_uses_the_measured_offset(tmp_path):
    s = sources.Section(tmp_path / "a.mp4", 88, 182, offset=86.5, length=96.0)
    assert sources.section_for([s], 100, 130) is s
    assert sources.section_for([s], 170, 185) is None


@needs_ffmpeg
def test_section_alignment_on_a_real_cut(tmp_path):
    """A keyframe-aligned section that starts earlier than asked is still placed exactly."""
    import numpy as np

    rate = 16000
    rng = np.random.default_rng(7)
    full = (rng.standard_normal(rate * 90) * 3000).astype("<i2")
    wav = tmp_path / "full.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(full.tobytes())
    section_file = tmp_path / "section.m4a"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "31.6", "-t", "30", "-i", str(wav),
                    "-c:a", "aac", "-b:a", "128k", str(section_file)], check=True)
    section = sources.align_section(sources.Section(section_file, 33.0, 63.0), wav, subprocess.run)
    assert abs(section.offset - 31.6) < 0.05 and section.confidence > 0.6


def test_facecam_samples_spread_over_sections(tmp_path):
    secs = [sources.Section(tmp_path / "a.mp4", 0, 80, length=80.0),
            sources.Section(tmp_path / "b.mp4", 500, 540, length=40.0)]
    samples = ce.facecam_samples(None, 1000, secs, count=24)
    files = [p.name for p, _ in samples]
    assert files.count("a.mp4") > files.count("b.mp4") >= 2
    assert all(0 < t < 80 for p, t in samples if p.name == "a.mp4")


@needs_ffmpeg
def test_stream_motion_reads_proxy_and_keyframes(tmp_path):
    clip = tmp_path / "proxy.mp4"
    # 20 s grey, then 20 s of moving test pattern.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=gray:size=426x240:rate=30:d=20",
                    "-f", "lavfi", "-i", "testsrc2=size=426x240:rate=30:d=20", "-filter_complex",
                    "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]", "-g", "60", "-pix_fmt", "yuv420p", str(clip)],
                   check=True)
    motion = ce.stream_motion(clip, 40)
    assert len(motion) == 40 and max(motion[2:18]) < 0.05 and max(motion[22:39]) > 0.3
    big = tmp_path / "full.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip), "-vf", "scale=1280:720",
                    "-g", "60", "-pix_fmt", "yuv420p", str(big)], check=True)
    key_motion = ce.stream_motion(big, 40)
    assert max(key_motion[2:16]) < 0.02 and min(key_motion[24:38]) > 0.1   # keyframes ~2 s apart


def test_signal_origin_is_reported_to_the_vision_model(monkeypatch):
    seen = {}

    def fake_json(url, model, prompt, schema, **kwargs):
        seen["prompt"] = prompt
        return {"ranked": [{"index": 5, "score": 0.8, "story_score": 0.7, "screen_type": "active_gameplay",
                            "reason": "1v3 clutch visible", "hook_title": "Clutch with 1 HP"}]}

    monkeypatch.setattr(ce, "ollama_json", fake_json)
    c = Candidate(5, 10, 40, .4, "", 0, origin="signal", reason="signal spike: loudness", signal_strength=0.7)
    ranked = ce._rank_batch([c], "m", "Valorant", ["img"], False)
    assert "no transcript pick" in seen["prompt"] and "LITERALLY true" in seen["prompt"]
    assert ranked[0].quality > 60 and ranked[0].hook_title == "Clutch with 1 HP"


def test_manifest_json_is_serialisable_with_new_fields():
    c = Candidate(0, 0, 10, .1, "", 0, title_claims=[{"claim": "x", "evidence": "y", "literally_true": True}])
    json.dumps(ce.asdict(c))


# --- captions in Hindi / Bengali ------------------------------------------------------------

def _indic_seg(start, text):
    words, t = [], start
    for token in text.split():
        words.append({"start": t, "end": t + 0.3, "word": " " + token, "probability": 0.9})
        t += 0.35
    return {"start": start, "end": t, "text": text, "words": words}


def test_caption_words_keep_indic_vowel_signs():
    import clip_style as cs

    assert cs._caption_word(" रहा,") == "रहा"
    assert cs._caption_word("है?") == "है"
    assert cs._caption_word("হলো!") == "হলো"
    assert cs._caption_word("'bro'") == "'BRO'"


def test_captions_never_mix_scripts_and_switch_font(tmp_path, monkeypatch):
    import clip_style as cs

    monkeypatch.setattr(cs, "script_font", lambda script: {"DEVANAGARI": "Nirmala UI"}.get(script))
    rows = [_indic_seg(0.0, "भाई क्या है"), _indic_seg(1.05, "ভাই এটা"), _indic_seg(1.8, "bro what")]
    groups = cs.caption_groups(rows, 0, 4)
    assert all(len({cs.text_script(w["text"]) for w in g}) == 1 for g in groups)
    path = tmp_path / "c.ass"
    cs.write_bold_captions(rows, 0, 4, path, caption_y=1300, font_family="Arial Black")
    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.startswith("Dialogue")]
    hindi = [line for line in lines if "भाई" in line]
    latin = [line for line in lines if "BRO" in line]
    assert hindi and all(r"\fnNirmala UI\fsp0" in line for line in hindi)
    assert latin and not any(r"\fn" in line for line in latin)


def test_gpu_decoding_is_batched_and_unprompted():
    gpu = asr.whisper_options(asr.MIXED, batched=True)
    assert gpu["batch_size"] >= 4 and gpu["without_timestamps"] is False and gpu["initial_prompt"] is None
    assert "condition_on_previous_text" not in gpu and "hallucination_silence_threshold" not in gpu
    cpu = asr.whisper_options(asr.MIXED)
    assert "batch_size" not in cpu and cpu["hallucination_silence_threshold"] == 2.0


def test_prompt_echo_is_detected_only_for_long_copies():
    prompt = "Gaming stream with friends. Let's go, clutch, headshot, GG, revive, spike plant."
    assert asr.is_prompt_echo("Let's go, clutch, headshot, GG, revive, spike plant.", prompt)
    assert not asr.is_prompt_echo("clutch headshot", prompt)
    assert not asr.is_prompt_echo("he is planting the spike right now", prompt)


def test_clean_transcript_drops_words_spelled_in_two_indic_scripts():
    words = [{"start": 0.0, "end": 0.3, "word": " आरे", "probability": 0.9},
             {"start": 0.4, "end": 0.7, "word": " আरे", "probability": 0.9},   # Bengali + Devanagari
             {"start": 0.8, "end": 1.1, "word": " ভাই", "probability": 0.9}]
    cleaned, stats = moments.clean_transcript([{"start": 0, "end": 1.1, "text": "आरे আरे ভাই", "words": words}])
    assert stats["words_dropped"] == 1 and cleaned[0]["text"] == "आरे ভাই"


def test_split_layout_gameplay_crop_avoids_the_webcam_corner():
    import clip_style as cs

    cam = cs.Box(1704, 838, 856, 602, face_cx=2074.0, face_cy=1082.0)   # borderless cut-out, bottom right
    plan = cs.plan_layout(cs.LAYOUT_AUTO, 2560, 1440, cam)
    crop = [p for p in plan.filter.split(";") if p.startswith("[game_src]crop=")][0]
    w, _, x, _ = (int(v) for v in crop.split("crop=")[1].split(",")[0].split(":"))
    assert x + w <= cam.x
    centred = cs.plan_layout(cs.LAYOUT_AUTO, 2560, 1440, cs.Box(1100, 0, 360, 300))  # top-centre cam: no shift
    crop = [p for p in centred.filter.split(";") if p.startswith("[game_src]crop=")][0]
    assert crop.split("crop=")[1].split(",")[0].split(":")[2] == "620"


def test_clean_up_keeps_real_low_confidence_speech_but_drops_loops():
    def seg(start, text, prob, ratio=1.2):
        words, t = [], start
        for token in text.split():
            words.append({"start": t, "end": t + 0.3, "word": " " + token, "probability": prob})
            t += 0.35
        return {"start": start, "end": t, "text": text, "words": words, "compression_ratio": ratio}

    rows = [seg(0, "No, you should have died earlier", 0.29),                       # real, matched YouTube
            seg(5, "आमि डिस्कार्ड थेके बेरोलाम आमार पिंग", 0.85, ratio=2.5),       # Indic text, high ratio
            seg(10, "कि वह एक " + "हुआ " * 20, 0.6, ratio=5.1),                     # mostly a decode loop
            seg(20, "What?", 0.01)]                                                  # near-zero single word
    cleaned, stats = moments.clean_transcript(rows)
    assert [r["start"] for r in cleaned] == [0, 5]
    assert stats["segments_dropped"] == 2


def test_stream_context_keeps_the_premise_not_the_links():
    info = {"title": "PREPPING FOR GTA 6", "description": "Carry teams up with Techno Gamerz for a heist.\n\n"
                                                          "Members: https://youtube.com/x\nFollow me on IG"}
    text = ce.stream_context(info)
    assert text.startswith("Stream title: PREPPING FOR GTA 6.") and "Techno Gamerz" in text and "http" not in text
    assert ce.stream_context({}) == ""


def test_truncated_structured_reply_keeps_complete_items():
    text = '{\n  "moments": [\n    {"start": 7419, "end": 7446, "score": 65},\n    {"start": 75' + " " * 40
    assert moments.salvage_json_array(text) == {"moments": [{"start": 7419, "end": 7446, "score": 65}]}
    assert moments.salvage_json_array("not json") is None


def test_clean_transcript_collapses_phrase_loops():
    text = "marker lagayega " + "like karenge " * 8 + "ok"
    words, t = [], 0.0
    for token in text.split():
        words.append({"start": t, "end": t + 0.3, "word": " " + token, "probability": 0.9})
        t += 0.35
    cleaned, stats = moments.clean_transcript([{"start": 0, "end": t, "text": text, "words": words}])
    assert stats["segments_dropped"] == 1            # 12 of 19 words were the loop -> mostly a loop
    short = "bhai " + "like karenge " * 3 + "chal"
    words = [{"start": i * 0.4, "end": i * 0.4 + 0.3, "word": " " + w, "probability": 0.9}
             for i, w in enumerate(short.split())]
    cleaned, _ = moments.clean_transcript([{"start": 0, "end": 4, "text": short, "words": words}])
    assert cleaned[0]["text"] == "bhai like karenge like karenge chal"


def test_edges_that_jump_around_are_not_a_split_screen_divider(tmp_path):
    """Door frames and building edges in a normal game land somewhere new every second."""
    import numpy as np
    from PIL import Image

    import clip_style as cs

    rng = np.random.default_rng(5)
    for i, seam in enumerate([0.55, 0.58, 0.70, 0.65, 0.57, 0.52, 0.30, 0.60, 0.65, 0.45]):
        cut = int(320 * seam)
        left = rng.integers(60, 120, (180, cut)).astype(np.uint8)
        right = rng.integers(150, 210, (180, 320 - cut)).astype(np.uint8)
        Image.fromarray(np.hstack([left, right])).save(tmp_path / f"frame_{i:06d}.jpg")
    traits = cs.clip_frame_traits(sorted(tmp_path.glob("*.jpg")))
    assert traits["seam_x"] is None and not traits["split_screen"]


def test_word_stretched_over_a_removed_pause_is_capped():
    words = [{"start": 10.0, "end": 10.3, "word": " main", "probability": 0.9},
             {"start": 10.3, "end": 25.5, "word": " nahi", "probability": 0.9},
             {"start": 25.5, "end": 25.8, "word": " paaya", "probability": 0.9}]
    cleaned, _ = moments.clean_transcript([{"start": 10, "end": 25.8, "text": "main nahi paaya", "words": words}])
    assert cleaned[0]["words"][1]["end"] <= 10.3 + moments.MAX_WORD_SECONDS + 1e-6


def test_critic_looks_at_the_frames_when_it_can(monkeypatch):
    calls = []

    def fake_json(url, model, prompt, schema, images=None, **kwargs):
        calls.append((images, "LOOK at it" in prompt))
        if images == ["sheet1"]:
            return {"verdicts": [{"index": 1, "frames_show": "people dancing in a club", "what_happens": "dancing",
                                  "event_visible": False, "clear_reaction": False, "standalone": True,
                                  "payoff": False, "stranger_score": 20, "keep": False, "reason": "",
                                  "title_claims": [{"claim": "caught by guards", "evidence": "dancing",
                                                    "literally_true": False}],
                                  "title_supported": True, "fixed_title": "Dancing in the club"}]}
        return {"verdicts": []}

    monkeypatch.setattr(moments, "ollama_json", fake_json)
    items = [{"index": 1, "title": "Team gets CAUGHT"}, {"index": 2, "title": "Clutch"}]
    verdicts = moments.critique(items, url="u", model="m", images={1: "sheet1", 2: "sheet2"})
    assert calls == [(["sheet1"], True), (["sheet2"], True)]          # one clip per call when it can see
    assert verdicts[1]["title_supported"] is False and verdicts[1]["fixed_title"] == "Dancing in the club"
    assert verdicts[1]["frames_show"] == "people dancing in a club"
    calls.clear()
    moments.critique(items, url="u", model="m")                        # no frames: side by side, one call
    assert calls == [(None, False)]


def test_loop_rows_are_recognised():
    loop = {"compression_ratio": 19.0, "words": [{"word": " कर"}, {"word": " दो"}] * 20}
    real = {"compression_ratio": 1.6, "words": [{"word": " " + w} for w in "भाई गाड़ी कहां है जल्दी चल".split()]}
    assert asr.is_loop_row(loop) and not asr.is_loop_row(real)


def test_subscribe_spread_over_segments_is_dropped():
    def seg(start, words):
        ws = [{"start": start + i * 0.4, "end": start + i * 0.4 + 0.3, "word": " " + w, "probability": 0.9}
              for i, w in enumerate(words.split())]
        return {"start": start, "end": ws[-1]["end"], "text": words, "words": ws}

    rows = [seg(0, "प्रोपेलर लगने वाला है"), seg(2, "सब्सक्राइब"), seg(3, "सब्सक्राइब"), seg(4, "सब्सक्राइब")]
    cleaned, _ = moments.clean_transcript(rows)
    assert [r["text"] for r in cleaned] == ["प्रोपेलर लगने वाला है"]


def test_rubric_blends_into_the_critic_score(monkeypatch):
    reply = {"verdicts": [{"index": 0, "frames_show": "x", "what_happens": "y", "event_visible": True,
                           "clear_reaction": True, "standalone": False, "payoff": False, "stranger_score": 80,
                           "title_claims": [], "title_supported": True, "fixed_title": "t", "keep": True, "reason": ""}]}
    monkeypatch.setattr(moments, "ollama_json", lambda *a, **k: reply)
    verdict = moments.critique([{"index": 0, "title": "t"}], url="u", model="m")[0]
    assert verdict["stranger_score"] == 0.5 * 80 + 0.5 * (30 + 25)


def test_clip_ending_on_seconds_of_scoreboard_is_trimmed():
    rows = [{"start": float(t), "end": float(t) + 2.5, "text": "go go go now.",
             "words": [{"start": float(t), "end": float(t) + 0.4, "word": " go", "probability": 0.9}]}
            for t in range(100, 160, 3)]
    c = Candidate(0, 100.0, 140.0, .5, "", 0)
    labels = ["gameplay", "gameplay", "death_or_spectate"] + ["scoreboard_or_banner"] * 3
    assert ce.apply_edge_labels(c, labels, rows, True, 500, 15, 45) is None
    assert c.end < 140.0 - 3.0
    one_banner = Candidate(1, 100.0, 140.0, .5, "", 0)
    labels = ["gameplay"] * 3 + ["gameplay", "gameplay", "scoreboard_or_banner"]
    assert ce.apply_edge_labels(one_banner, labels, rows, True, 500, 15, 45) is None and one_banner.end == 140.0
