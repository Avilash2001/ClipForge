import json
import sys
import types

sys.modules.setdefault("yt_dlp", types.ModuleType("yt_dlp"))
fw = types.ModuleType("faster_whisper")
fw.WhisperModel = object
sys.modules.setdefault("faster_whisper", fw)

import clip_style as cs  # noqa: E402
from clipper_engine import Candidate, choose_top, build_render_graph  # noqa: E402
from moments import (  # noqa: E402
    Moment, chat_spikes, clean_transcript, dedupe_moments, loudness_spikes, parse_live_chat, snap_to_speech,
)


def _words(start, text, step=0.4, prob=0.9):
    out, t = [], start
    for token in text.split():
        out.append({"start": t, "end": t + step * 0.8, "word": " " + token, "probability": prob})
        t += step
    return out


def _seg(start, text, **extra):
    words = _words(start, text)
    row = {"start": start, "end": words[-1]["end"], "text": text, "words": words}
    row.update(extra)
    return row


# --- transcript clean-up ---------------------------------------------------

def test_clean_transcript_drops_foreign_script_and_floods():
    rows = [
        _seg(0, "let's go bro"),
        _seg(5, "ༀༀༀༀༀༀༀༀༀༀ"),                      # Tibetan flood
        _seg(10, "अरे निकल 嗎 बेट से"),                  # CJK glyph inside Hindi
        _seg(15, "garbage words here"),                # very low word confidence
    ]
    for word in rows[3]["words"]:
        word["probability"] = 0.25
    cleaned, stats = clean_transcript(rows, "Mixed (English + Bengali + Hindi)")
    texts = [r["text"] for r in cleaned]
    assert texts == ["let's go bro", "अरे निकल बेट से"]
    assert stats["segments_dropped"] == 2
    assert stats["words_dropped"] >= 1


def test_clean_transcript_breaks_repetition_loops_and_low_probability_words():
    row = _seg(0, "watch watch watch watch watch out")
    row["words"][-1]["probability"] = 0.05
    cleaned, _ = clean_transcript([row], "English")
    assert cleaned[0]["text"] == "watch watch watch"


# --- spike signals -----------------------------------------------------------

def test_loudness_spike_detects_a_shout_relative_to_baseline():
    rms = [0.05] * 120
    for i in range(60, 63):
        rms[i] = 0.4                       # ~18 dB shout
    spikes = loudness_spikes(rms)
    assert max(spikes[58:66]) > 0.8
    assert max(spikes[:40]) == 0.0


def test_chat_spikes_from_live_chat_file(tmp_path):
    lines = []
    for second in range(0, 600, 5):        # steady chat: one message every 5s
        lines.append({"replayChatItemAction": {"videoOffsetTimeMsec": str(second * 1000),
                                               "actions": [{"addChatItemAction": {}}]}})
    for i in range(60):                    # burst of 60 messages around 300s
        lines.append({"replayChatItemAction": {"videoOffsetTimeMsec": str(300000 + i * 100),
                                               "actions": [{"addChatItemAction": {}}]}})
    path = tmp_path / "chat.live_chat.json"
    path.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    offsets = parse_live_chat(path)
    assert len(offsets) == 120 + 60
    spikes = chat_spikes(offsets, 600)
    assert max(spikes[295:315]) == 1.0
    assert max(spikes[:200]) == 0.0


# --- boundary snapping ----------------------------------------------------------

def test_snap_starts_on_sentence_and_skips_filler():
    rows = [
        _seg(0.0, "We need to go left now."),
        _seg(3.0, "uh so yeah he is right there behind the box."),
        _seg(8.0, "Oh my god he shot me through the wall."),
        _seg(12.0, "That is insane bro."),
    ]
    start, end = snap_to_speech(4.0, 11.0, rows, 60, min_len=5, max_len=30)
    # The rough start falls inside the filler sentence: snap to it and trim "uh so yeah",
    # starting 0.25s before "he" (word 4 of that sentence, at 4.2s).
    assert abs(start - 3.95) < 0.01
    # The rough end lands mid-sentence: finish "...through the wall." plus a reaction tail.
    assert abs(end - (rows[2]["words"][-1]["end"] + 0.7)) < 0.01


def test_snap_respects_max_length():
    rows = [_seg(float(i * 4), "this is one long rambling sentence.") for i in range(30)]
    start, end = snap_to_speech(0.0, 110.0, rows, 200, min_len=15, max_len=40)
    assert end - start <= 40.0
    assert end - start >= 15.0


def test_dedupe_keeps_highest_scoring_overlap():
    a = Moment(10, 40, 80, hook_title="a")
    b = Moment(15, 45, 60, hook_title="b")
    c = Moment(100, 130, 70, hook_title="c")
    assert [m.hook_title for m in dedupe_moments([a, b, c])] == ["a", "c"]


# --- selection ---------------------------------------------------------------------

def test_choose_top_applies_min_score():
    candidates = [
        Candidate(0, 0, 30, .5, "", 0, final_score=.9, quality=45),
        Candidate(1, 100, 130, .5, "", 0, final_score=.8, quality=75),
        Candidate(2, 200, 230, .5, "", 0, final_score=.7, quality=-1),  # not judged by AI
    ]
    assert [c.index for c in choose_top(candidates, 5, min_score=60)] == [1, 2]
    assert [c.index for c in choose_top(candidates, 5)] == [0, 1, 2]


# --- captions & layouts -------------------------------------------------------------

def test_caption_groups_are_short_and_uppercase():
    rows = [_seg(0.0, "oh my god he is actually right behind you bro, run")]
    groups = cs.caption_groups(rows, 0, 10)
    assert all(1 <= len(g) <= 4 for g in groups)
    assert all(sum(len(w["text"]) for w in g) <= 16 for g in groups)
    assert groups[0][0]["text"] == "OH"


def test_bold_captions_highlight_one_word_per_event(tmp_path):
    rows = [_seg(1.0, "we going up right")]
    path = tmp_path / "c.ass"
    assert cs.write_bold_captions(rows, 0, 5, path, caption_y=1300, font_family="Arial")
    events = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.startswith("Dialogue")]
    assert len(events) == 4
    assert all(line.count("&H00E1FF&") == 1 for line in events)
    assert "\\pos(540,1300)" in events[0]


def test_split_layout_with_facecam_uses_stack_and_seam_title():
    cam = cs.Box(0, 700, 480, 380, face_cx=240, face_cy=850)
    plan = cs.plan_layout(cs.LAYOUT_AUTO, 1920, 1080, cam)
    assert plan.name == cs.LAYOUT_SPLIT
    assert "vstack" in plan.filter and plan.title_center_y == cs.SPLIT_TOP_H


def test_auto_layout_without_facecam_zooms_and_coop_stacks_halves():
    assert cs.plan_layout(cs.LAYOUT_AUTO, 1920, 1080).name == cs.LAYOUT_ZOOM
    coop = cs.plan_layout(cs.LAYOUT_AUTO, 1920, 1080, split_screen=True)
    assert coop.name == cs.LAYOUT_COOP
    assert "crop=960:" in coop.filter  # each player half is cropped separately


def test_letterbox_band_limits_the_gameplay_crop():
    plan = cs.plan_layout(cs.LAYOUT_ZOOM, 1920, 1080, band=(0.15, 0.85))
    # 70% of 1080 rows remain -> crop height 756 starting at row 162
    assert "crop=692:756:614:162" in plan.filter


def test_render_graph_chains_title_captions_and_fades(tmp_path):
    plan = cs.plan_layout(cs.LAYOUT_ZOOM, 1920, 1080)
    graph = build_render_graph(plan, 20.0, title=True, ass_path=tmp_path / "x.ass")
    assert "[base][title]overlay" in graph and "[titled]ass=" in graph and graph.endswith("[v]")


def test_title_card_renders(tmp_path):
    size = cs.render_title_card("TIM can't believe who's STREAMING inside GTA 6 💀", tmp_path / "t.png")
    assert size is not None and size[0] <= 960 and size[1] > 80


def test_tidy_hook_title_tames_all_caps_but_keeps_good_titles():
    from moments import tidy_hook_title

    good = "TIM can't believe who's STREAMING inside GTA 6 💀"
    assert tidy_hook_title(good) == good
    shouted = tidy_hook_title("ONE BULLET KILLS THE WHOLE SQUAD?! 😱")
    caps = [w for w in shouted.split() if w.isupper() and len(w) > 1]
    assert 1 <= len(caps) <= 2 and shouted.endswith("😱")
    assert "FPS" in tidy_hook_title("GRAPHICS SETTINGS DESTROY FPS RATE")


def test_clean_transcript_keeps_confident_segments_in_a_low_confidence_window():
    # Window-level avg_logprob is shared by all segments of a 30 s decode window; a confident
    # English sentence must survive even when the window average is dragged down.
    rows = [_seg(0, "I am calling you", avg_logprob=-1.3, no_speech_prob=0.5, compression_ratio=1.4)]
    cleaned, _ = clean_transcript(rows, "Mixed")
    assert [r["text"] for r in cleaned] == ["I am calling you"]
