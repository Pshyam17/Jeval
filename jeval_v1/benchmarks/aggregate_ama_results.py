#!/usr/bin/env python3
"""
benchmarks/aggregate_ama_results.py

Merges per-episode JSONs produced by run_ama_episode.py, computes mean ± std,
prints a summary table, and writes the final aggregate JSON.

Supports domain filtering for stratified results (SOFTWARE, Game, etc.)

CLI:
    python benchmarks/aggregate_ama_results.py \
      --results-dir benchmarks/results/ama_bench_episodes \
      --out benchmarks/results/ama_bench_software_final.json \
      --domain SOFTWARE
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List


def _load_episodes(results_dir: Path, domain: str = None) -> tuple[List[dict], List[dict]]:
    ok, errors = [], []
    for f in sorted(results_dir.glob("episode_*.json")):
        data = json.loads(f.read_text())
        if "error" in data:
            errors.append(data)
        elif domain is not None and data.get("domain") != domain:
            continue  # Skip episodes from other domains
        else:
            ok.append(data)
    return ok, errors
        data = json.loads(f.read_text())
        if "error" in data:
            errors.append(data)
        else:
            ok.append(data)
    return ok, errors


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="benchmarks/results/ama_bench_episodes")
    parser.add_argument("--out",         default="benchmarks/results/ama_bench_software_final.json")
    parser.add_argument("--domain",      default="SOFTWARE", help="Domain for stratified results")
    parser.add_argument("--frozen-mode", action="store_true", default=True,
                        help="Use frozen eval mode (no within-episode updates)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_path    = Path(args.out)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    episodes, errors = _load_episodes(results_dir, domain=args.domain if args.domain != "all" else None)
    print(f"Loaded {len(episodes)} successful episodes ({args.domain}), {len(errors)} errors")

    if not episodes:
        print("No successful episodes to aggregate.")
        return

    scores            = [ep["score"] for ep in episodes]
    compression_ratios = [ep["compression_ratio"] for ep in episodes]
    orig_tokens       = [ep["original_tokens"] for ep in episodes]
    comp_tokens       = [ep["compressed_tokens"] for ep in episodes]

    mean_score = sum(scores) / len(scores)
    std_score  = _std(scores)
    mean_cr    = sum(compression_ratios) / len(compression_ratios)

    # Per question-type breakdown
    by_type: Dict[str, List[float]] = {}
    for ep in episodes:
        for qa in ep.get("per_question", []):
            t = qa.get("type", "unknown")
            by_type.setdefault(t, []).append(qa["score"])

    # Fair comparison reporting (architecture_v3.md §3.3)
    answer_model = os.environ.get("JEVAL_ANSWER_MODEL", "qwen/qwen3.5-122b-a10b")
    judge_model  = os.environ.get("JEVAL_JUDGE_MODEL", "qwen/qwen3.5-122b-a10b")
    embedding_model = "all-mpnet-base-v2"
    retrieval_k = 5

    print(f"\n{'='*60}")
    print(f"AMA-Bench {args.domain} — {len(episodes)} episodes ({'frozen' if args.frozen_mode else 'adaptive'} mode)")
    print(f"{'='*60}")
    print(f"  Overall accuracy:    {mean_score:.3f} ± {std_score:.3f}")
    print(f"  Compression ratio:   {mean_cr:.3f}  (compressed/original tokens)")
    print(f"  Total orig tokens:   {sum(orig_tokens):,}")
    print(f"  Total comp tokens:   {sum(comp_tokens):,}")
    print(f"  Errors:              {len(errors)}")
    print(f"\nFair Comparison Config (AMA-Agent parity):")
    print(f"  Answer model:        {answer_model}")
    print(f"  Judge model:         {judge_model}")
    print(f"  Embedding model:     {embedding_model}")
    print(f"  Retrieval k:         {retrieval_k}")
    print(f"\nBy QA type:")
    for qtype in sorted(by_type):
        vals = by_type[qtype]
        m = sum(vals) / len(vals)
        s = _std(vals)
        print(f"  Type {qtype}: {m:.3f} ± {s:.3f}  (n={len(vals)})")

    summary = {
        "episodes_ok":            len(episodes),
        "episodes_error":         len(errors),
        "mean_score":             mean_score,
        "std_score":              std_score,
        "mean_compression_ratio": mean_cr,
        "total_original_tokens":  sum(orig_tokens),
        "total_compressed_tokens": sum(comp_tokens),
        "by_type":                {t: {"mean": sum(v)/len(v), "std": _std(v), "n": len(v)}
                                    for t, v in by_type.items()},
        "errors":                 errors,
        # Fair comparison checklist (architecture_v3.md §3.3)
        "fair_comparison_config": {
            "answer_model":         answer_model,
            "judge_model":          judge_model,
            "embedding_model":      embedding_model,
            "retrieval_k":          retrieval_k,
            "split":                "test",
            "domain":               args.domain,
            "frozen_mode":          args.frozen_mode,
            "predictor_checkpoint": os.environ.get("JEVAL_PREDICTOR_CKPT", "none"),
        },
        "metrics_reported": ["accuracy", "compression_ratio"],
        "note": "F1 metric requires per-QA gold/pred alignment; add if comparing to AMA-Agent F1 scores",
    }

    # Append timestamp suffix if output file already exists
    if out_path.exists():
        ts = int(time.time())
        out_path = out_path.with_name(
            f"{out_path.stem}_{ts}{out_path.suffix}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote aggregate results to {out_path}")


if __name__ == "__main__":
    main()
