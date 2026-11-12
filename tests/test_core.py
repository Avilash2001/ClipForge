import sys
import types

# Keep the small unit tests runnable without downloading Whisper/yt-dlp model dependencies.
sys.modules.setdefault("yt_dlp", types.ModuleType("yt_dlp"))
fw = types.ModuleType("faster_whisper")
fw.WhisperModel = object
sys.modules.setdefault("faster_whisper", fw)

from clipper_engine import Candidate, _extract_json_object, choose_top, transcript_quality  # noqa: E402


def test_punctuation_only_transcript_is_rejected():
    rows = [{"start": 1.0, "end": 2.0, "text": ", , ,", "words": []}]
    assert transcript_quality(rows)["usable"] is False


def test_mixed_script_transcript_is_accepted_when_substantive():
    rows = [
        {"start": 1.0, "end": 2.0, "text": "Let's go, we won that round!", "words": []},
        {"start": 2.1, "end": 3.0, "text": "দারুণ খেলেছিস ভাই, একদম সুন্দর প্লে করেছে", "words": []},
        {"start": 3.1, "end": 4.0, "text": "बहुत अच्छा था, शानदार खेला भाई", "words": []},
    ]
    assert transcript_quality(rows)["usable"] is True


def test_json_parser_accepts_plain_or_embedded_json():
    assert _extract_json_object('{"ranked": []}') == {"ranked": []}
    assert _extract_json_object('Result: {"ranked": []} thanks') == {"ranked": []}


def test_choose_top_drops_menus_and_overlaps():
    candidates = [
        Candidate(0, 0, 35, .9, "", 0, final_score=.99, screen_type="starting_screen"),
        Candidate(1, 30, 65, .9, "", 0, final_score=.95, screen_type="active_gameplay"),
        Candidate(2, 60, 95, .8, "", 0, final_score=.90, screen_type="active_gameplay"),
        Candidate(3, 100, 135, .7, "", 0, final_score=.80, screen_type="active_gameplay"),
    ]
    picked = choose_top(candidates, 3)
    assert [c.index for c in picked] == [1, 3]
