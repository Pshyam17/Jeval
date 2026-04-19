# Jeval Memory v3.0 Architecture Specification

## Executive Summary

v3.0 fixes fundamental mathematical flaws in v2.0's memory architecture by introducing **pre-hoc fidelity gating (cosine EPE + schema gap + predictor EPE)**, **schema novelty detection + schema updates**, **context reuse caching**, **deployable (judge-free) query/answer confidence routing**, **budget-aware retries**, and **proper cold-storage feedback loops**. The key insight: separate **construction fidelity** (did compression lose facts?) from **retrieval sufficiency** (did we fetch supporting evidence?) and make all decisions using signals available at deployment.

---

## Part 1: Mathematical Foundations

### 1.1 Core Problem Statement

**v2.0 Flaw:** Retrieval quality measured by cosine similarity, not evidence sufficiency for answering.

**v3.0 Solution:** Do **not** optimize on judge correctness (oracle). Instead:
- **Construction-time**: prevent lossy compression from entering hot cache (cosine EPE + schema gap + predictor EPE; detect unknown artifacts).
- **Retrieval-time**: route and retry based on deployable confidence (evidence support / coverage / hot trust).

```
Deployment objective (judge-free):
  Return an answer supported by retrieved evidence under strict latency and token budgets.

Subject to:
  |H| ≤ token_ceiling (hot cache size limit)
  latency < 5s per QA (retrieval deadline)
```

### 1.2 Query-Response Compatibility (QRC)

**v2.0 Flaw:** Confidence gate requires `original_entry`—impossible at decision time.

**v3.0 Math:** Use two deployable scalars:
- **Query confidence** \(C_q\): is hot cache likely sufficient?
- **Answer confidence** \(C_a\): is the produced answer supported by retrieved evidence?

```
Entity extractor:
  E(text) → set of entities (paths, ALL_CAPS ids, error classes, ≥2-digit numbers, step refs)

Hot peek:
  H_k(q) = top-k hot-cache hits for q (small k, e.g. 3)

Query confidence (hot sufficiency):
  C_q(q) = |E(q) ∩ E(H_k(q))| / max(|E(q)|, 1)

Given attempt n returns context c_n and answer a_n:
  Support:   C_sup = |E(a_n) ∩ E(c_n)| / max(|E(a_n)|, 1)
  Coverage:  C_cov = |E(q) ∩ E(a_n)| / max(|E(q)|, 1)

Hot trust (from build-time fidelity risk R_build attached to hot items used in c_n):
  C_trust = 1 − mean(R_build over used hot items)

Answer confidence:
  C_a = clip(w1*C_sup + w2*C_cov + w3*C_trust, [0,1])
```

Default thresholds/weights (initial, for ablation):
  θ_reuse = 0.85
  θ_confidence_stop = 0.72
  w1 = 0.5  # evidence support
  w2 = 0.3  # query coverage
  w3 = 0.2  # hot trust

Note: R_build is the build-time fidelity risk stored in hot-cache metadata:
  R_build = λ1 * cosine_epe + λ2 * schema_gap' + λ3 * predictor_epe
  where λ1 + λ2 + λ3 = 1, and all terms are clipped to [0,1]

Predictor EPE (optional but recommended):
  predictor_epe = ||P(enc(compressed)) - enc(original)||² / 4
  where P is a trained predictor head (e.g., AMA-trained checkpoint)

### 1.3 Context Reuse Cache

**Concept:** Pure caching across a session/episode. If a similar question is asked, reuse the retrieved context to save time and API costs. This is NOT overfitting, as it relies solely on query similarity and requires no ground truth.

```
Context Cache:
  cache: q_uuid → (question, context)

For new query q_new:
  If ∃q_cached with cosine_sim(q_new, q_cached) > θ_reuse:
    return cache[q_cached].context  # Reuse
  Else:
    context = Retrieve(q_new, H, C, G)
    cache[q_new] = (q_new, context)
```

### 1.4 Fixed Retrieval Policy (Non-oracle)

**v2.0 Flaw:** Fixed k=5 for all query types.
**Previous v3.0 Draft Flaw:** Learning from judge feedback *during* an episode (overfitting).

**v3.0 Math:** Use a fixed typed policy plus deployable escalation. If tuned, tune only on non-oracle metrics (latency/tokens/evidence support proxies), then freeze at deployment.

```
Policy: π(qa_type) → (k, routing)

Learned fixed values by type (example):
  k_base = {
    "factoid": 1, "verification": 2, "procedural": 3,
    "causal": 5, "entity": 3, "summary": 8
  }
  
  routing_base = {
    "factoid": "hot_cache", "verification": "hot_cache",
    "procedural": "enriched", "causal": "cold_storage",
    "entity": "hot_cache", "summary": "enriched"
  }
```

### 1.5 Confidence-Based Retry with Escalation

**v2.0 Flaw:** Single retrieval attempt, permanent failure on miss.
**Previous v3.0 Draft Flaw:** Retrying based on ground-truth judge feedback.

**v3.0 Math:** Use deployable confidence \(C_a\) (defined in §1.2) and enforce strict token budgets so escalation does not silently truncate away evidence.

```
For each QA:
  Attempt 1: k=k_base, routing=routing_base
  Attempt 2: route escalation first (hot→enriched/tool→cold), then k increase only if budget allows
  Attempt 3: cold/tool route with highest recall under remaining budget
  
  After generating answer for Attempt n:
    confidence = C_a(q, c_n, a_n)
    If confidence > θ_confidence_stop:
      return answer
      
  Stop when confident or max_attempts reached.
```

### 1.6 Truncation-Aware Schema Gap

**v2.0 Flaw:** 800-char truncation before ingestion destroys schema facts.

**v3.0 Math:** Apply truncation correction only when truncation ratio is known.

```
If r = len_trunc / max(1, len_orig) is known:
  schema_gap' = schema_gap × (1 − 0.5 × (1 − min(1, r)))

Else:
  schema_gap' = schema_gap
```

### 1.6.1 Predictor-Aware Build-Time Risk

**Goal:** Use the trained predictor as an explicit pre-hoc fidelity signal, not just offline analysis.

```
Given original x and candidate compression x_hat:
  cosine_epe   = 1 - <enc(x), enc(x_hat)>
  schema_gap'  = truncation-aware schema gap from §1.6
  predictor_epe = ||P(enc(x_hat)) - enc(x)||² / 4

Build-time risk:
  R_build = λ1 * cosine_epe + λ2 * schema_gap' + λ3 * predictor_epe
  subject to λ1 + λ2 + λ3 = 1

Gate decision:
  accept candidate iff R_build <= τ_commit
```

### 1.7 Cold Storage Feedback Loop

**v2.0 Flaw:** Cold storage hits don't update graph or hot cache.

**v3.0 Math:**
On cold_storage hit for query Q returning segment s:
1. Graph Edge Strengthening: `w(h, s) ← w(h, s) + η_hit`
2. Promotion Tracking: Promote `s` to Hot Cache if frequently hit and recent.

### 1.8 Schema Novelty Detection and Schema Update Loop

**Failure Mode:** Schema gap only catches loss for known artifact types; unknown artifacts silently regress to cosine-only.

**v3.0 Math:** Maintain a schema set \(\mathcal{S}\) with a computable fit score per schema.

```
Fit:
  Fit(x, s) ∈ [0,1]   # pattern/field extraction success for schema s on artifact x

Novelty:
  Novelty(x | S) = 1 − max_{s ∈ S} Fit(x, s)

If Novelty(x | S) > τ_novel:
  mark artifact as unknown, store cold, avoid aggressive hot compression,
  enqueue x for offline schema induction

Schema induction (offline):
  cluster unknown artifacts by structural signature φ(x)
  if cluster size ≥ m:
    create schema s_new where required fields are stable across the cluster
    add s_new to S
```

---

## Part 2: Architecture Components

### 2.1 Module Specifications

#### `jeval/memory/session_aware.py`

```python
class ContextReuseCache:
    """Caches retrieval context by question similarity. (Pure caching, no learning)"""
    def get_or_retrieve(q_uuid, question, retrieve_fn) -> (context, was_cached)

class StaticRetrievalPolicy:
    """Returns fixed k and routing per QA type (non-oracle defaults)."""
    def get_k(qa_type: str) -> int
    def get_routing(qa_type: str) -> str

class ConfidenceRetryEscalation:
    """Retries with budget-aware escalation, judged by deployable confidence."""
    def execute(question, mem, qa_type, policy, llm_fn, confidence_fn) -> (answer, attempts)
```

#### `jeval/memory/jeval_memory.py` (v3.0 changes)

```python
class JevalMemory:
    def retrieve_with_routing(self, query: str, k: int = 5, routing_preference: str = "hot_cache") -> str:
        """Explicit routing control: hot_cache, cold_storage, enriched."""
        
    def _cold_fallback(self, query: str, k: int) -> str:
        """Cold storage search with empty result fallback."""
```

#### `jeval/memory/schema_gap.py` (v3.0 changes)

```python
class TruncationAwareSchemaGap:
    """Schema gap computation accounting for pre-ingestion truncation."""
```

#### `jeval/memory/co_retrieval_graph.py` (v3.0 changes)

```python
class CoRetrievalGraph:
    def record_retrieval(self, seg_ids: list[str], was_cold_hit: bool = False) -> None:
        """Increment edge weights. Strengthen more if was_cold_hit."""
```

---

## Part 3: AMA-Bench Integration

### Fixed `run_ama_episode.py`

```python
def run_episode(episode_idx, dataset_name, split, out_path):
    # Load episode and build trajectory text WITH TRUNCATION TRACKING
    ...
    
    # Initialize memory (NO PREDICTOR - v3 uses schema-based budget)
    mem = JevalMemoryV2(db_path=".jeval/ama_bench_memory.db")
    for line in traj_lines: mem.ingest(line)
    
    # Initialize deployment components (No within-episode learning)
    context_cache = ContextReuseCache(mem._encoder)
    retrieval_policy = StaticRetrievalPolicy()  # Loaded fixed weights
    retry_logic = ConfidenceRetryEscalation(max_attempts=3)
    
    for qa in qas:
        # Get context (with reuse check)
        def retrieve_fn(q):
            k = retrieval_policy.get_k(qa_type)
            routing = retrieval_policy.get_routing(qa_type)
            return mem.retrieve_with_routing(q, k=k, routing_preference=routing)
        
        context, was_cached = context_cache.get_or_retrieve(q_uuid, question, retrieve_fn)
        
        # Retry with escalation using confidence heuristic
        answer, attempts = retry_logic.execute(
            question=question, mem=mem, qa_type=qa_type, policy=retrieval_policy,
            llm_fn=..., confidence_fn=...
        )
        
        # Evaluate correct purely for final reporting (not fed back into memory)
        correct, reasoning = _judge(client, question, reference, answer)
        scores.append(1.0 if correct else 0.0)
```

### AMA Evaluation Modes (Explicit)

- **Frozen AMA eval mode (default for fair benchmarking)**:
  - no schema induction during an episode
  - no hot-cache promotions during an episode
  - no graph weight updates during an episode
  - allow only `ContextReuseCache` (pure reuse of retrieved contexts)

- **Session-adaptive mode (for realistic deployments)**:
  - allow miss-triggered recompression and graph updates
  - perform schema induction offline after the episode
  - report results separately from frozen mode

### AMA-Agent Fair Comparison Protocol (Required)

To claim fair comparison against AMA-Agent, evaluation must mirror their reported setup:

1. **Dataset / split parity**
   - Use `AMA-bench/AMA-bench`, `split="test"`.
   - Report both:
     - full test set (all supported domains), and
     - domain-stratified results (at minimum SOFTWARE).
   - Do not compare a SOFTWARE-only score against AMA-Agent all-domain averages.

2. **Backbone parity**
   - Use the same answer-generation backbone family as the baseline under comparison.
   - Primary parity target: Qwen3-32B (and optionally Qwen3-8B for secondary table parity).
   - Keep decoding settings fixed across methods (temperature, max tokens, stop policy).

3. **Judge parity**
   - Use LLM-as-judge configured to match AMA-Bench protocol (Qwen3-32B judge).
   - Keep judge prompt, temperature, and parsing rules fixed for all compared systems.
   - If a different judge is used (e.g., Mistral/NIM), label results as non-comparable and
     report them separately as internal diagnostics only.

4. **Metric parity**
   - Report both **Accuracy** and **F1** (not only episode-average correctness).
   - Compute metrics at QA level, then aggregate by capability and by domain.

5. **Retrieval parity controls**
   - Fix top-k retrieval at the parity setting used for similarity retrieval (`K=5`).
   - Keep embedding/index configuration fixed within each method during comparisons.
   - Any method-specific tool routes (graph/keyword/cold fallback) must be documented.

6. **Frozen-eval policy for benchmarking**
   - Use **Frozen AMA eval mode** for leaderboard-style comparison.
   - No within-episode schema updates, graph updates, or hot promotions in frozen runs.
   - Session-adaptive runs must be reported in a separate table.

7. **Reporting checklist in result artifacts**
   - Include in every summary file:
     - answer model name/version
     - judge model name/version
     - embedding model
     - retrieval `k`
     - split/domain scope
     - frozen vs adaptive mode
     - predictor checkpoint ID (if used)

---

## Part 4: What Was Removed from v2.0 & Previous Drafts

### 4.1 Within-Episode Judge Learning (REMOVED)
**Reason:** Adapting policy `k` and routing based on ground-truth judge feedback *during* a test episode is overfitting. It relies on information (the ground truth answer and the judge) that is unavailable in real-world deployment. We now use a Fixed Policy learned during training and a Confidence heuristic for retries.

### 4.2 Predictor Integration (REMOVED)
**Reason:** Architectural mismatch. Predictor outputs embeddings for EPE proxy, but v3.0 uses schema-based budget allocation.

### 4.3 Confidence Gate Routing (REMOVED)
**Reason:** Requires `original_entry` at decision time—impossible without cold storage lookup first. Replaced by the Static Retrieval Policy.

---

## Part 5: Implementation Checklist

### Phase 1: Core Components
- [ ] `jeval/memory/session_aware.py` - Context cache, static policy, confidence retry
- [ ] `jeval/memory/jeval_memory.py` - Add `retrieve_with_routing` method
- [ ] `jeval/memory/schema_gap.py` - Add `TruncationAwareSchemaGap`
- [ ] `jeval/memory/co_retrieval_graph.py` - Add cold-hit strengthening, differential decay
- [ ] `jeval/epe/combined.py` - Add predictor_epe support and weighted `R_build`
- [ ] `jeval/memory/jeval_memory.py` - Gate on predictor-aware `R_build` (`τ_commit`)

### Phase 2: AMA-Bench Integration
- [ ] `benchmarks/run_ama_episode.py` - Rewrite with session-aware loop and confidence checks
- [ ] `jeval/benchmarks/ama_bench_eval.py` - Remove predictor, fix fallback

### Phase 3: Testing
- [ ] Unit tests for `ContextReuseCache` (similarity threshold behavior)
- [ ] Unit tests for `ConfidenceRetryEscalation` (escalation logic)
- [ ] Integration test: Single AMA-Bench episode end-to-end
- [ ] Ablation: Context reuse on/off, retry on/off

### Phase 4: Validation
- [ ] Run test episodes in SOFTWARE domain, compare to v2.0 baseline
- [ ] Measure context reuse hit rate (expected: 20-40% for related QAs)
- [ ] Measure retry recovery rate using confidence heuristic
- [ ] Ablation: `(λ1, λ2, λ3)` sweep for cosine/schema/predictor contributions

---

## Part 7: Math-to-Code Checklist (Validation Matrix)

| Math Component | Equation / Rule | Intended Module | Status |
|---|---|---|---|
| Build-time combined risk | `R_build = λ1*cosine_epe + λ2*schema_gap' + λ3*predictor_epe` | `jeval/epe/combined.py`, `jeval/memory/jeval_memory.py` | Partial |
| Pre-hoc gate | `accept iff R_build <= τ_commit` | `jeval/memory/jeval_memory.py` | Partial |
| Query confidence | `C_q(q) = |E(q)∩E(H_k(q))| / max(|E(q)|,1)` | `jeval/memory/session_aware.py` | Not implemented |
| Answer confidence | `C_a = clip(w1*C_sup + w2*C_cov + w3*C_trust)` | `jeval/memory/session_aware.py` | Not implemented |
| Budget-aware retry escalation | route escalation first, increase `k` only if budget allows | `jeval/memory/session_aware.py` | Not implemented |
| Truncation-aware schema gap | `schema_gap'` from §1.6 | `jeval/memory/schema_gap.py` | Partial |
| Schema novelty detection | `Novelty(x|S) = 1 - max_s Fit(x,s)` | `jeval/memory/schema_gap.py` + ingestion pipeline | Partial |
| Schema induction loop | cluster unknown artifacts, create `s_new` | offline tooling + schema registry | Not implemented |
| Frozen vs adaptive AMA mode | episode-time update constraints | `benchmarks/run_ama_episode.py` | Not implemented |
| Fair-comparison scoring parity | Qwen3-32B judge + Accuracy/F1 + matched backbone/split | `jeval/benchmarks/ama_bench_eval.py`, reporting scripts | Partial |

Legend:
- **Implemented**: in code and used in runtime path
- **Partial**: implemented pieces exist but not fully wired to runtime decisions
- **Not implemented**: documented only

---

## Part 6: Expected Improvements

| Metric | v2.0 | v3.0 (Projected) | Source of Improvement |
|--------|------|------------------|----------------------|
| General accuracy (Test Set) | ~35% | 40-45% | EPE-gated hot cache + deployable routing/retries |
| Context reuse rate | 0% | 20-40% | ContextReuseCache |
| Retry recovery rate | 0% | 10-20% | ConfidenceRetryEscalation |
| Cold hit utilization | Low | High | Feedback loop to graph |

---

## Appendix A: Configuration Defaults

```python
# Session-aware defaults
CONTEXT_REUSE_THRESHOLD = 0.85  # Cosine similarity for reuse
RETRY_MAX_ATTEMPTS = 3
RETRY_K_MULTIPLIER = 2.0

# Graph decay rates
DECAY_SYSTEM = 0.995    # half-life ~1400 queries
DECAY_CAUSAL = 0.98     # half-life ~35 queries
DECAY_SEMANTIC = 0.95   # half-life ~14 queries
DECAY_CO_OCCUR = 0.90   # half-life ~7 queries
```

---

*Document version: v3.0 (Corrected)*
*Last updated: 2026-04-19*
