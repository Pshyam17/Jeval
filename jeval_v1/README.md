# Jeval: Pre-hoc Fidelity Gating for Agent Memory Compression

Jeval is a research library for **pre-hoc fidelity gating** of AI agent memory. Unlike systems that compress first and evaluate later, Jeval measures semantic fidelity *before* committing a compressed entry to memory — and falls back to higher-budget compression or extractive summarisation when the candidate fails the gate.

The core claim: **EPE-based pre-hoc fidelity gating outperforms post-hoc compression evaluation on agent memory tasks, with schema gap adding 117% additional signal over cosine EPE alone on causal elision failures.**

## Architecture

### Two-Tier Memory

```
Ingest
  ├── Cold Storage  (append-only, SQLite+FTS5, always written)
  └── Hot Cache     (compressed, bounded, cosine retrieval)
        ├── Novelty Gate   — skip near-duplicates
        └── Fidelity Gate  — CombinedEPE before commit
```

**Write path:** Every segment is written unconditionally to cold storage (original text, FTS5 indexed). Admission to the hot cache requires passing two gates:

1. **Novelty gate** — cosine EPE against a 50-embedding working set of recent originals. Near-duplicates are cold-only.
2. **Fidelity gate** — LLM generates a compressed candidate; `CombinedEPE = α·cosine_epe + (1−α)·schema_gap` is computed; candidate committed if below threshold, else retried (max 2) then extractive fallback.

**Retrieval path:** Confidence gate routes each query before any similarity search:

```
score = |E_q ∩ E_c| / max(|E_q ∩ E_o|, 1)

score ≥ high_confidence  →  hot cache (semantic cosine search)
score ≤ low_confidence   →  cold storage (FTS5 full-text search)
otherwise                →  both, merged
```

Entity extraction is regex-only: ALL_CAPS identifiers, file paths, step references, ≥2-digit numbers, error class names. Generic nouns excluded.

### EPE (Embedding Predictive Error)

```
EPE(orig, comp) = 1 − ⟨enc(orig), enc(comp)⟩
```

Encoder: frozen `all-mpnet-base-v2` (768-dim, L2-normalised). Cosine distance chosen over squared Euclidean — squared distance accumulates ~0.003 per dimension for orthogonal vectors, indistinguishable from verbatim match across 768 dims.

### Schema Gap

Regex-based fact-presence scoring across 6 domain schemas:

| Schema | Required facts |
|--------|---------------|
| `tool_call` | tool_name, result |
| `error` | error_type, location |
| `test_result` | failure_count, test_file |
| `migration_failure` | timing, table_name, error_type |
| `deployment` | environment, outcome |
| `file_modification` | file_path, action |

```
schema_gap(orig, comp, τ) = |F_o \ F_c| / max(|F_o|, 1)
```

**Motivating example:** *"migration failed on staging due to lock timeout after 30s on roles table"* vs *"migration failed on staging"* — `cosine_epe=0.2974`, `schema_gap=1.0`, `epe_final=0.6487`. Schema gap adds **117% additional signal** over cosine EPE alone.

### Combined EPE

```
epe_final = α · cosine_epe + (1 − α) · schema_gap      (default α = 0.5)
```

### Self-Healing: Miss-Triggered Recompression

Hot cache entries with high miss counters are rewritten from cold storage originals:

```
trigger: miss_counter > miss_threshold AND turns_since_last_rewrite > min_rewrite_gap
```

Eviction formula (normalised per cycle):

```
score = 0.3·time_n + 0.4·miss_n − 0.2·hit_n + 0.1·(1 − epe_novelty)
```

The 0.4 weight on miss counter is the key departure from LRU.

## Quick Start

```python
from jeval.memory.jeval_memory import JevalMemory

mem = JevalMemory(
    alpha=0.5,            # cosine vs schema gap blend
    high_confidence=0.7,  # hot cache routing threshold
    low_confidence=0.4,   # cold storage routing threshold
)

# build memory from agent session
mem.memory_construction(session_text, task="fix auth bug")

# retrieve with confidence-gated routing
context = mem.memory_retrieve("why did the migration fail", top_k=5)
```

## Ablation Parameters

| Parameter | Default | Controls |
|-----------|---------|---------|
| `alpha` | 0.5 | cosine EPE vs schema gap blend |
| `beta` | 1.0 | budget modulation strength |
| `high_confidence` | 0.7 | hot cache routing threshold |
| `low_confidence` | 0.4 | cold storage routing threshold |
| `miss_threshold` | 2 | recompression trigger |
| `min_rewrite_gap` | 5 | recompression cooldown (turns) |
| `eviction_weights` | (0.3, 0.4, 0.2, 0.1) | time / miss / hit / novelty |

## Stack

| Component | Implementation |
|-----------|---------------|
| Sentence encoder | all-mpnet-base-v2, sentence-transformers 3.0.1, frozen |
| LLM compressor | Mistral Small 3.1 24B via NVIDIA NIM |
| NLI classifier | cross-encoder/nli-MiniLM2-L6-H768 |
| Entity extraction | Regex-only; spaCy optional |
| Cold storage | SQLite 3.45 + FTS5 |
| Training | PyTorch 2.4.1, NVIDIA A100 |

## Benchmarks

```bash
# AMA-Bench (208 SOFTWARE episodes)
python benchmarks/run_ama_episode.py \
    --episode-idx 0 \
    --dataset AMA-bench/AMA-bench \
    --split test \
    --predictor checkpoints/predictor_v2_best.pt \
    --out benchmarks/results/ama_bench_episodes/episode_0.json

# DroidBench (NIM judge, 4 probe types)
export NVIDIA_API_KEY=...
python jeval/benchmarks/droid_bench.py
```

Published baselines (DroidBench): Factory AI 2.45/5 · Anthropic 2.33/5 · OpenAI 2.19/5

SimpleMem SOTA (AMA-Bench, LoCoMo F1): 43.24%

## Training

```bash
# generate pairs and train predictor
python train/generate_pairs.py \
    --n-faithful 5000 --n-hard-negative 5000 --swebench-pairs 500 \
    --out train/data/pairs_v2.jsonl

python train/train_predictor_v2.py \
    --pairs train/data/pairs_v2.jsonl \
    --out-path checkpoints/predictor_v2_best.pt \
    --lr 3e-4 1e-4 --epochs 100 --batch-size 32
```

Trained predictor achieves **83.46× EPE separation** (faithful vs lossy compressions). Target: 5×.

## Tests

```bash
python -m pytest tests/ --ignore=tests/demo -q
# 179 passed, 8 skipped in ~95s
```

## Citation

```bibtex
@article{jeval2026,
  title={Pre-hoc Fidelity Gating for Agent Memory Compression},
  author={Shyam, Preethi},
  year={2026}
}
```
