import json

import pytest

from jeval_v2.aggregate import aggregate


def write(path, epe, seed, cosine=(0.2, 0.8, 0.1, 0.9), selection="mse"):
    path.write_text("\n".join(json.dumps({"id": str(i), "group_id": f"g{i}", "split": "test",
        "train_seed": seed, "train_selection": selection, "harm": i % 2,
        "cosine": cosine[i], "epe": epe[i]}) for i in range(4)))
    return path


def test_aggregate_checks_same_held_out_pairs_and_reports_seed_variation(tmp_path):
    paths = [write(tmp_path / "seed-0.jsonl", [0.1, 0.9, 0.2, 0.8], 0),
             write(tmp_path / "seed-1.jsonl", [0.2, 0.8, 0.3, 0.7], 1)]
    result = aggregate(paths)
    assert result["seeds"] == 2
    assert result["pairs"] == 4
    assert result["train_seeds"] == [0, 1]
    assert result["summary"]["epe_auprc"]["sample_std"] == 0
    with pytest.raises(ValueError, match="different held-out"):
        paths[1].write_text(paths[1].read_text().replace('"harm": 1', '"harm": 0', 1))
        aggregate(paths)


def test_repeated_seed_is_rejected(tmp_path):
    paths = [write(tmp_path / "a.jsonl", [0.1, 0.9, 0.2, 0.8], 0),
             write(tmp_path / "b.jsonl", [0.1, 0.9, 0.2, 0.8], 0)]
    with pytest.raises(ValueError, match="distinct"):
        aggregate(paths)


def test_missing_seed_provenance_is_rejected(tmp_path):
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text("\n".join(json.dumps({"id": str(i), "group_id": f"g{i}", "split": "test",
        "harm": i % 2, "cosine": 0.5, "epe": 0.5}) for i in range(4)))
    with pytest.raises(ValueError, match="missing train_seed"):
        aggregate([write(tmp_path / "a.jsonl", [0.1, 0.9, 0.2, 0.8], 0), legacy])


def test_mixed_selection_is_not_pooled(tmp_path):
    paths = [write(tmp_path / "a.jsonl", [0.1, 0.9, 0.2, 0.8], 0),
             write(tmp_path / "b.jsonl", [0.1, 0.9, 0.2, 0.8], 1, selection="auprc")]
    with pytest.raises(ValueError, match="different checkpoint selections"):
        aggregate(paths)


def test_cosine_tolerance_allows_device_jitter_but_not_real_differences(tmp_path):
    jittered = (0.2 + 1e-9, 0.8, 0.1, 0.9)
    paths = [write(tmp_path / "a.jsonl", [0.1, 0.9, 0.2, 0.8], 0),
             write(tmp_path / "b.jsonl", [0.1, 0.9, 0.2, 0.8], 1, cosine=jittered)]
    assert aggregate(paths)["seeds"] == 2
    write(tmp_path / "b.jsonl", [0.1, 0.9, 0.2, 0.8], 1, cosine=(0.3, 0.8, 0.1, 0.9))
    with pytest.raises(ValueError, match="not the same embeddings"):
        aggregate(paths)
