"""Per-fold predictor check: validation curve against the identity, then the label-free linear gate."""
import argparse
import json

import numpy as np

from jeval_v2.diagnose import diagnose_arrays, unit


def curve(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.startswith('{"epoch"')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="directory holding fold-K/pairs.npz")
    parser.add_argument("--fold-log", required=True, help="train log path with a {fold} placeholder")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2],
                        help="validation-only sweep")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    for fold in range(args.folds):
        print(f"== fold {fold}")
        epochs = curve(args.fold_log.format(fold=fold))
        if epochs:
            best = min(epochs, key=lambda e: e["validation_mse"])
            print(f"  epochs run {len(epochs)} | validation MSE first {epochs[0]['validation_mse']:.6f}, "
                  f"best {best['validation_mse']:.6f} at epoch {best['epoch']}, last {epochs[-1]['validation_mse']:.6f}")
        else:
            print("  no epoch lines found in the log")
        with np.load(f"{args.run_dir}/fold-{fold}/pairs.npz", allow_pickle=False) as data:
            original, compressed = data["original"], data["compressed"]
            splits, groups = data["splits"], data["groups"]
        held = splits == "validation"
        identity = np.mean(np.sum((unit(original[held]) - unit(compressed[held])) ** 2, axis=1)) / original.shape[1]
        print(f"  identity validation MSE (same per-element units) {identity:.6f}")
        for ridge in args.ridge:
            out = diagnose_arrays(original, compressed, splits, groups, ridge=ridge,
                                  split="validation", bootstrap=args.bootstrap)
            low, high = out["group_bootstrap_95pct_intercept_to_matrix_reduction"]
            print(f"  ridge {ridge:g} | squared error identity {out['mean_identity_squared_error']:.5f}, "
                  f"intercept {out['mean_intercept_squared_error']:.5f}, matrix {out['mean_learned_squared_error']:.5f} | "
                  f"intercept-to-matrix {out['mean_group_intercept_to_matrix_reduction']:.6f} [{low:.6f}, {high:.6f}]")


if __name__ == "__main__":
    main()
