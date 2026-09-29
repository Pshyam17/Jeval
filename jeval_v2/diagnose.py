"""Label-free, held-out linear test of a learnable compression transform."""
import numpy as np


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def fit_residual_map(compressed, original, groups, ridge=1e-3):
    """Fit x-c from c with equal total weight per training group."""
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    compressed, original = unit(compressed), unit(original)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1 / np.sqrt(counts[inverse])
    design = np.column_stack((compressed, np.ones(len(compressed))))
    weighted = design * weights[:, None]
    target = (original - compressed) * weights[:, None]
    penalty = np.eye(design.shape[1]) * ridge
    penalty[-1, -1] = 0  # do not penalize a constant compressor shift
    return np.linalg.solve(weighted.T @ weighted + penalty, weighted.T @ target)


def diagnose_arrays(original, compressed, splits, groups, ridge=1e-3,
                    split="validation", bootstrap=1000, seed=0):
    train = splits == "train"
    held = splits == split
    if not train.any() or not held.any():
        raise ValueError(f"train and {split} rows are required")
    if set(groups[train]) & set(groups[held]):
        raise ValueError("training and held-out groups overlap")
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    x, c = unit(original), unit(compressed)
    coef = fit_residual_map(c[train], x[train], groups[train], ridge)
    design = np.column_stack((c[held], np.ones(held.sum())))
    predicted = unit(c[held] + design @ coef)
    identity_error = np.sum((x[held] - c[held]) ** 2, axis=1)
    learned_error = np.sum((x[held] - predicted) ** 2, axis=1)
    delta = identity_error - learned_error
    names, inverse = np.unique(groups[held], return_inverse=True)
    group_delta = np.array([delta[inverse == i].mean() for i in range(len(names))])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(names), size=(bootstrap, len(names)))
    interval = np.quantile(group_delta[draws].mean(axis=1), [0.025, 0.975])
    return {
        "train_pairs": int(train.sum()), "held_out_pairs": int(held.sum()),
        "train_groups": int(len(set(groups[train]))), "held_out_groups": int(len(names)),
        "ridge": ridge, "mean_cos_prediction_to_compressed": float(np.mean(np.sum(predicted * c[held], axis=1))),
        "mean_identity_squared_error": float(identity_error.mean()),
        "mean_learned_squared_error": float(learned_error.mean()),
        "mean_group_error_reduction": float(group_delta.mean()),
        "group_bootstrap_95pct_error_reduction": interval.tolist(),
    }
