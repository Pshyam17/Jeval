"""Aggregate predeclared seeds over identical held-out pair predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def aggregate(paths):
    if len(paths) < 2:
        raise ValueError("at least two seed score files are required")
    if len({str(Path(path).resolve()) for path in paths}) != len(paths):
        raise ValueError("each seed needs a different score file")
    runs = []
    reference = None
    for path in paths:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"empty score file: {path}")
        keyed = {row["id"]: row for row in rows}
        if len(keyed) != len(rows):
            raise ValueError(f"duplicate pair IDs: {path}")
        signature = {key: (row["group_id"], row["split"], row["harm"], row["cosine"])
                     for key, row in keyed.items()}
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(f"different held-out examples or cosine scores: {path}")
        labels = np.array([keyed[k]["harm"] for k in sorted(keyed)])
        known = labels >= 0
        if len(set(labels[known])) != 2:
            raise ValueError("both labeled harm classes are required")
        cos = np.array([keyed[k]["cosine"] for k in sorted(keyed)])
        epe = np.array([keyed[k]["epe"] for k in sorted(keyed)])
        runs.append({"file": str(path), "epe_auprc": float(average_precision_score(labels[known], epe[known])),
                     "cosine_auprc": float(average_precision_score(labels[known], cos[known])),
                     "epe_auroc": float(roc_auc_score(labels[known], epe[known])),
                     "cosine_auroc": float(roc_auc_score(labels[known], cos[known]))})
    for run in runs:
        for metric in ("auprc", "auroc"):
            run[f"epe_minus_cosine_{metric}"] = run[f"epe_{metric}"] - run[f"cosine_{metric}"]
    summary = {}
    for key in ("epe_auprc", "epe_auroc", "epe_minus_cosine_auprc", "epe_minus_cosine_auroc"):
        values = np.array([run[key] for run in runs])
        summary[key] = {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1)),
                        "min": float(values.min()), "max": float(values.max())}
    return {"seeds": len(runs), "pairs": len(reference), "runs": runs, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", nargs="+", help="one evaluation JSONL per training seed")
    args = parser.parse_args()
    print(json.dumps(aggregate(args.scores), indent=2))


if __name__ == "__main__":
    main()
