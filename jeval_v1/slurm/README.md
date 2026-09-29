# SLURM Scripts for AMA-Bench Evaluation

This directory contains SLURM array job scripts for running AMA-Bench evaluation at scale.

## Quick Start

```bash
# 1. Set up environment (add your NVIDIA API key)
source slurm/setup_env.sh

# 2. Submit AMA-Bench array job (all 208 episodes)
sbatch slurm/ama_bench_array.sh

# 3. Wait for jobs to complete, then aggregate results
sbatch slurm/aggregate_ama_results.sh
```

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `ama_bench_array.sh` | Array job: runs one episode per SLURM task (0-207) |
| `aggregate_ama_results.sh` | Aggregates per-episode JSONs into summary |
| `setup_env.sh` | Sets up conda environment and API keys |
| `smart_submit.sh` | Helper for conditional submission |
| `train_predictor.sh` | Training job for predictor head (optional) |

## Fair Comparison Configuration (AMA-Agent Parity)

To claim fair comparison against AMA-Agent, the following must match their setup:

### Required Configuration

| Parameter | Value | Where to Set |
|-----------|-------|--------------|
| **Dataset** | `AMA-bench/AMA-bench` | `--dataset` flag |
| **Split** | `test` | `--split` flag |
| **Answer Model** | `qwen/qwen3.5-122b-a10b` | `JEVAL_ANSWER_MODEL` env var |
| **Judge Model** | `qwen/qwen3.5-122b-a10b` | `JEVAL_JUDGE_MODEL` env var |
| **Embedding Model** | `all-mpnet-base-v2` | Hardcoded (v3.0 default) |
| **Retrieval K** | `5` | Hardcoded (parity setting) |
| **Eval Mode** | `frozen` | `--frozen-mode` flag (default) |

### Environment Variables

```bash
# Set these before submitting
export JEVAL_ANSWER_MODEL="qwen/qwen3.5-122b-a10b"
export JEVAL_JUDGE_MODEL="qwen/qwen3.5-122b-a10b"
export JEVAL_AMA_DOMAIN="SOFTWARE"  # For stratified results
export NVIDIA_API_KEY="nvapi-..."
```

### Frozen vs Adaptive Mode

- **Frozen mode** (default): No within-episode updates to schema, graph, or hot cache. Use for leaderboard comparison.
- **Adaptive mode**: Allows graph updates, cold-hit strengthening, miss-triggered recompression. Use for realistic deployment testing. Report separately.

```bash
# Frozen mode (fair comparison)
python benchmarks/run_ama_episode.py --episode-idx 0 --out result.json

# Adaptive mode (deployment simulation)
python benchmarks/run_ama_episode.py --episode-idx 0 --out result.json --no-frozen-mode
```

## Output Files

### Per-Episode (`benchmarks/results/ama_bench_episodes/episode_*.json`)

```json
{
  "episode_idx": 0,
  "episode_id": 12345,
  "domain": "SOFTWARE",
  "score": 0.75,
  "n_questions": 8,
  "per_question": [...],
  "compression_ratio": 0.42,
  "original_tokens": 5000,
  "compressed_tokens": 2100,
  "fair_comparison_config": {
    "answer_model": "qwen/qwen3.5-122b-a10b",
    "judge_model": "qwen/qwen3.5-122b-a10b",
    "embedding_model": "all-mpnet-base-v2",
    "retrieval_k": 5,
    "frozen_mode": true,
    "predictor_checkpoint": "none"
  }
}
```

### Aggregate (`benchmarks/results/ama_bench_SOFTWARE_final.json`)

```json
{
  "episodes_ok": 208,
  "episodes_error": 0,
  "mean_score": 0.425,
  "std_score": 0.031,
  "mean_compression_ratio": 0.45,
  "by_type": {...},
  "fair_comparison_config": {
    "answer_model": "qwen/qwen3.5-122b-a10b",
    "judge_model": "qwen/qwen3.5-122b-a10b",
    "embedding_model": "all-mpnet-base-v2",
    "retrieval_k": 5,
    "split": "test",
    "domain": "SOFTWARE",
    "frozen_mode": true,
    "predictor_checkpoint": "none"
  },
  "metrics_reported": ["accuracy", "compression_ratio"]
}
```

## Domain-Stratified Results

To run evaluation on specific domains:

```bash
# SOFTWARE domain only
export JEVAL_AMA_DOMAIN="SOFTWARE"
sbatch slurm/ama_bench_array.sh
sbatch slurm/aggregate_ama_results.sh

# All domains (full test set)
export JEVAL_AMA_DOMAIN="all"
sbatch slurm/ama_bench_array.sh
sbatch slurm/aggregate_ama_results.sh
```

Available domains: `SOFTWARE`, `Game`, `EMBODIED_AI`, `OPENWORLD_QA`, `TEXT2SQL`, `WEB`

## Troubleshooting

### Episode fails with "predictor not found"

The predictor is **optional** in v3.0. Scripts now auto-detect:
- If `checkpoints/predictor_v2_best.pt` exists: use it
- Otherwise: run without predictor (default deployment mode)

### Results show lower accuracy than expected

Check:
1. **Model parity**: Are you using Qwen3-32B family for answer generation?
2. **Judge parity**: Is the judge model matching AMA-Bench protocol?
3. **Frozen mode**: Are you running frozen eval (not adaptive)?
4. **Domain scope**: Are you comparing SOFTWARE-only to SOFTWARE-only (not all-domain average)?

### SLURM array job stuck

Check logs in `logs/ama_*.out` and `logs/ama_*.err`. Common issues:
- NVIDIA_API_KEY not set
- Conda environment not activated
- Working directory path incorrect

## Reporting Checklist

Before claiming results, verify:

- [ ] Answer model name/version documented
- [ ] Judge model name/version documented
- [ ] Embedding model documented
- [ ] Retrieval K documented
- [ ] Split (test/val) documented
- [ ] Domain scope (SOFTWARE/all) documented
- [ ] Frozen vs adaptive mode documented
- [ ] Predictor checkpoint ID (if used) documented
- [ ] Both Accuracy and F1 reported (if comparing to AMA-Agent)

See `docs/architecture_v3.md` §3.3 for full protocol.
