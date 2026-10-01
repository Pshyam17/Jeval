"""Aggregate predeclared seeds over identical held-out pair predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def single(rows, field, path):
    values = {row.get(field) for row in rows}
    if len(values) != 1:
        raise ValueError(f"{path}: rows disagree on {field}")
    value = values.pop()
    if value is None:
        raise ValueError(f"{path}: missing {field}; re-run evaluate with this version")
    return value


def aggregate(paths, cosine_tolerance=1e-6):
    if len(paths) < 2:
        raise ValueError("at least two seed score files are required")
    if len({str(Path(path).resolve()) for path in paths}) != len(paths):
        raise ValueError("each seed needs a different score file")
    runs = []
    reference = reference_cosine = None
    for path in paths:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"empty score file: {path}")
        keyed = {row["id"]: row for row in rows}
        if len(keyed) != len(rows):
            raise ValueError(f"duplicate pair IDs: {path}")
        seed = single(rows, "train_seed", path)
        selection = single(rows, "train_selection", path)
        order = sorted(keyed)
        signature = {key: (keyed[key]["group_id"], keyed[key]["split"], keyed[key]["harm"]) for key in order}
        cos = np.array([keyed[key]["cosine"] for key in order])
        if reference is None:
            reference, reference_cosine = signature, cos
        else:
            if signature != reference:
                raise ValueError(f"different held-out examples: {path}")
            if not np.allclose(cos, reference_cosine, rtol=0, atol=cosine_tolerance):
                raise ValueError(f"cosine scores differ beyond {cosine_tolerance:g}; "
                                 f"these are not the same embeddings: {path}")
        labels = np.array([keyed[key]["harm"] for key in order])
        known = labels >= 0
        if len(set(labels[known])) != 2:
            raise ValueError("both labeled harm classes are required")
        epe = np.array([keyed[key]["epe"] for key in order])
        runs.append({"file": str(path), "train_seed": seed, "train_selection": selection,
                     "epe_auprc": float(average_precision_score(labels[known], epe[known])),
                     "cosine_auprc": float(average_precision_score(labels[known], cos[known])),
                     "epe_auroc": float(roc_auc_score(labels[known], epe[known])),
                     "cosine_auroc": float(roc_auc_score(labels[known], cos[known]))})
    seeds = [run["train_seed"] for run in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"training seeds must be distinct, got {seeds}; "
                         "these files do not measure between-seed variation")
    selections = {run["train_selection"] for run in runs}
    if len(selections) != 1:
        raise ValueError(f"cannot pool different checkpoint selections: {sorted(selections)}")
    for run in runs:
        for metric in ("auprc", "auroc"):
            run[f"epe_minus_cosine_{metric}"] = run[f"epe_{metric}"] - run[f"cosine_{metric}"]
    summary = {}
    for key in ("epe_auprc", "epe_auroc", "epe_minus_cosine_auprc", "epe_minus_cosine_auroc"):
        values = np.array([run[key] for run in runs])
        summary[key] = {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1)),
                        "min": float(values.min()), "max": float(values.max())}
    return {"seeds": len(runs), "train_seeds": sorted(seeds), "selection": selections.pop(),
            "pairs": len(reference), "runs": runs, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", nargs="+", help="one evaluation JSONL per training seed")
    parser.add_argument("--cosine-tolerance", type=float, default=1e-6,
                        help="absolute tolerance for identical cosine scores across seeds")
    args = parser.parse_args()
    print(json.dumps(aggregate(args.scores, args.cosine_tolerance), indent=2))


if __name__ == "__main__":
    main()
