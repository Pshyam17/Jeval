import json

import pytest

from jeval_v2.aggregate import aggregate


def test_aggregate_checks_same_held_out_pairs_and_reports_seed_variation(tmp_path):
    paths = []
    for seed, scores in enumerate(([0.1, 0.9, 0.2, 0.8], [0.2, 0.8, 0.3, 0.7])):
        path = tmp_path / f"seed-{seed}.jsonl"
        path.write_text("\n".join(json.dumps({"id": str(i), "group_id": f"g{i}", "split": "test",
            "harm": i % 2, "cosine": [0.2, 0.8, 0.1, 0.9][i], "epe": scores[i]})
            for i in range(4)))
        paths.append(path)
    result = aggregate(paths)
    assert result["seeds"] == 2
    assert result["pairs"] == 4
    assert result["summary"]["epe_auprc"]["sample_std"] == 0
    with pytest.raises(ValueError, match="different held-out"):
        paths[1].write_text(paths[1].read_text().replace('"harm": 1', '"harm": 0', 1))
        aggregate(paths)
