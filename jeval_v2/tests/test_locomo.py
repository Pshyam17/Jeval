import pytest

from jeval_v2.locomo import (build_pairs, fold_splits, pair_id, report, score_answer, segment_sample, select)


def sample(sample_id, turns=7):
    return {"sample_id": sample_id, "qa": [], "conversation": {
        "speaker_a": "A", "speaker_b": "B",
        "session_1_date_time": "1 May 2023",
        "session_1": [{"speaker": "A", "dia_id": f"D1:{i}", "text": f"hello {i}"} for i in range(1, turns + 1)],
        "session_2_date_time": "2 May 2023",
        "session_2": [{"speaker": "B", "dia_id": "D2:1", "text": "hi", "blip_caption": "a dog"}]}}


def test_segments_keep_dialog_ids_and_dates():
    segments = segment_sample(sample("c1"), turns=6)
    assert [s["id"] for s in segments] == ["c1:s1:t0", "c1:s1:t6", "c1:s2:t0"]
    assert segments[1]["dia_ids"] == ["D1:7"]
    assert segments[0]["original"].startswith("[Session 1, 1 May 2023]")
    assert "[shared image: a dog]" in segments[2]["original"]


def test_folds_are_disjoint_and_cover_every_conversation_as_test_once():
    groups = [f"c{i}" for i in range(10)]
    tested = []
    for fold in range(5):
        split = fold_splits(groups, 5, fold)
        assert sum(v == "test" for v in split.values()) == 2
        assert sum(v == "validation" for v in split.values()) == 2
        assert sum(v == "train" for v in split.values()) == 6
        tested += [g for g, v in split.items() if v == "test"]
    assert sorted(tested) == sorted(groups)
    with pytest.raises(ValueError):
        fold_splits(groups, 11, 0)


def test_pairs_never_split_a_conversation():
    segments = segment_sample(sample("c1"), 6) + segment_sample(sample("c2"), 6) + segment_sample(sample("c3"), 6)
    candidates = [{"segment_id": s["id"], "cand": k, "text": f"c{k}"} for s in segments for k in range(2)]
    pairs = build_pairs(segments, candidates, folds=3, fold=0)
    splits = {}
    for pair in pairs:
        assert splits.setdefault(pair["group_id"], pair["split"]) == pair["split"]
    assert set(splits.values()) == {"train", "validation", "test"}


def toy_selection(tolerance=0.25):
    segments = [{"id": "g:0", "group_id": "g", "words": 10, "original": "o0"},
                {"id": "g:1", "group_id": "g", "words": 10, "original": "o1"}]
    words = {0: "one two three four five", 1: "one two three four five six", 2: "one two", 3: "one two three four five"}
    candidates = [{"segment_id": s["id"], "cand": k, "text": text} for s in segments for k, text in words.items()]
    scores = {pair_id(s["id"], k): {"cosine": cos, "epe": epe}
              for s in segments for k, (cos, epe) in enumerate([(0.3, 0.2), (0.4, 0.5), (0.0, 0.0), (0.1, 0.9)])}
    return select(segments, candidates, scores, ratio=0.5, tolerance=tolerance, seed=0)


def test_select_respects_length_band_and_each_rule():
    memories, stats = toy_selection()
    # candidate 2 (2 words) has the best scores but is outside the 5-word +/-25% band
    assert memories["cosine"]["g"] == "one two three four five\n\none two three four five"  # cand 3 (cos .1)
    assert memories["epe"]["g"] == "one two three four five\n\none two three four five"  # cand 0 (epe .2)
    assert memories["original"]["g"] == "o0\n\no1"
    assert stats["band_fallback_segments"] == 0
    assert stats["cosine_epe_same_choice_fraction"] == 0.0
    assert stats["word_ratio"]["cosine"] == 0.5


def test_select_rejects_missing_scores():
    segments = [{"id": "g:0", "group_id": "g", "words": 4, "original": "o"}]
    with pytest.raises(ValueError, match="missing score"):
        select(segments, [{"segment_id": "g:0", "cand": 0, "text": "a b"}], {}, 0.5, 0.25, 0)


def test_answer_scoring_conventions():
    assert score_answer("The 7 May 2023", "7 May 2023", 2) == 1.0
    assert score_answer("cats, dogs", "cats, dogs", 1) == 1.0
    assert score_answer("fish", "cats; dogs", 3) == 0.0
    assert score_answer("cats", "cats; dogs", 3) == 1.0


def test_report_pairs_arms_and_bootstraps_over_conversations():
    rows = []
    for group in ("c1", "c2", "c3"):
        for q in range(4):
            for arm, value in (("epe", 1.0), ("cosine", 0.5)):
                rows.append({"arm": arm, "id": f"{group}:{q}", "group_id": group, "category": 1, "f1": value})
    result = report(rows, bootstrap=50)
    assert result["conversations"] == 3
    diff = result["paired_difference"]["epe_minus_cosine"]
    assert diff["mean_f1_difference"] == pytest.approx(0.5)
    assert diff["group_bootstrap_95pct"] == pytest.approx([0.5, 0.5])
    with pytest.raises(ValueError):
        report(rows[:-1])
