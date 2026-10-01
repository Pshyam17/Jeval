import json

import pytest
import torch

from jeval_v2.data import read_pairs
from jeval_v2.model import Predictor, scores


def test_group_split_overlap_is_rejected(tmp_path):
    p = tmp_path / "pairs.jsonl"
    p.write_text("\n".join(json.dumps({"id": str(i), "group_id": "same", "split": split,
        "original": "original", "compressed": "compressed"}) for i, split in enumerate(("train", "test"))))
    with pytest.raises(ValueError, match="crosses"):
        read_pairs(str(p))


def test_identity_predictor_residual_equals_two_cosine_distance():
    original = torch.tensor([[1., 0.], [0., 1.]])
    compressed = torch.tensor([[0.6, 0.8], [0., 1.]])
    cosine, residual = scores(original, compressed, torch.nn.Identity())
    assert torch.allclose(residual, 2 * cosine, atol=1e-6)


def test_residual_predictor_can_represent_identity_with_bottleneck():
    predictor = Predictor(dim=4, hidden=2)
    for layer in predictor.net:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
    compressed = torch.randn(3, 4)
    assert torch.equal(predictor(compressed), compressed)
