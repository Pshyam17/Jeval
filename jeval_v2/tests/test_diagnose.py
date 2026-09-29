import numpy as np
import pytest

from jeval_v2.diagnose import diagnose_arrays


def test_systematic_shift_improves_held_out_reconstruction_and_identity_does_not():
    rng = np.random.default_rng(7)
    original = rng.normal(size=(240, 8))
    original /= np.linalg.norm(original, axis=1, keepdims=True)
    compressed = original + np.array([0.4, 0.2, -0.1, 0, 0, 0, 0, 0])
    compressed /= np.linalg.norm(compressed, axis=1, keepdims=True)
    splits = np.array(["train"] * 160 + ["validation"] * 80)
    groups = np.array([f"g{i // 4}" for i in range(240)])

    result = diagnose_arrays(original, compressed, splits, groups)
    assert result["mean_group_error_reduction"] > 0.1
    assert result["group_bootstrap_95pct_error_reduction"][0] > 0
    assert result["mean_intercept_squared_error"] < result["mean_identity_squared_error"]
    assert result["mean_learned_squared_error"] < result["mean_intercept_squared_error"]

    null = diagnose_arrays(original, original, splits, groups)
    assert abs(null["mean_group_error_reduction"]) < 1e-12


def test_diagnose_rejects_leaked_group():
    with pytest.raises(ValueError, match="overlap"):
        diagnose_arrays(np.eye(2), np.eye(2), np.array(["train", "validation"]),
                        np.array(["same", "same"]))
