# Jeval Memory v3.0 Architecture Specification

## Executive Summary

v3.0 fixes fundamental mathematical flaws in v2.0's memory architecture by introducing **training-time policy learning**, **context reuse caching**, **confidence-based retries**, and **proper cold-storage feedback loops**. The key insight: optimize for a **general memory system** that maximizes expected judge correctness over a test set, using fixed inference policies, rather than overfitting via per-episode judge feedback.

---

## Part 1: Mathematical Foundations

### 1.1 Core Problem Statement

**v2.0 Flaw:** Retrieval quality measured by cosine similarity, not answer correctness.

**v3.0 Solution:** Optimize for judge correctness directly, but do so *only during training*. At deployment, use a fixed policy and confidence heuristics to avoid overfitting and unrealistic access to ground truth.

```
Training Objective Function:
  max E[JudgeCorrect(Q, Answer(LLM, Retrieve(Q, H, C, G; π)))] over Q ∈ TrainSet
  
Test/Deployment Objective:
  Evaluate expected correctness using fixed policy π*, without judge access.

Subject to:
  |H| ≤ token_ceiling (hot cache size limit)
  latency < 5s per QA (retrieval deadline)
```

### 1.2 Query-Response Compatibility (QRC)

**v2.0 Flaw:** Confidence gate requires `original_entry`—impossible at decision time.

**v3.0 Math:**

```
Definition: Query Intent
  Q → (I_q, R_q, C_q)
  
  I_q ∈ {FACTOID, PROCEDURAL, CAUSAL, ENTITY, SUMMARY, VERIFICATION}
  R_q = {r₁, ..., rₙ}  (required information slots)
  C_q = {c₁, ..., cₖ}  (optional constraints)

Definition: Response Facts
  R → ℱ(R) = {f₁, ..., fₘ}
  
  f = (type, value, confidence)

Compatibility Score:
  C(Q, R) = α·C_slot + β·C_constraint + γ·C_coverage
```

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

### 1.4 Training-Time Retrieval Policy

**v2.0 Flaw:** Fixed k=5 for all query types.
**Previous v3.0 Draft Flaw:** Learning from judge feedback *during* an episode (overfitting).

**v3.0 Math:** Learn optimal parameters `(k, routing)` per query type on a training set. Freeze these parameters at deployment.

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

**v3.0 Math:** Use a confidence heuristic (e.g., self-reflection, entity matching, or QRC score) to determine if a retry with expanded context is needed.

```
For each QA:
  Attempt 1: k=k_base, routing=routing_base
  Attempt 2: k=k_base×2, routing=next_best
  Attempt 3: k=k_base×4, routing=cold_storage
  
  After generating answer for Attempt n:
    confidence = compute_confidence_heuristic(answer, context)
    If confidence > θ_confidence:
      return answer
      
  Stop when confident or max_attempts reached.
```

### 1.6 Truncation-Aware Schema Gap

**v2.0 Flaw:** 800-char truncation before ingestion destroys schema facts.

**v3.0 Math:**

```
Schema Gap (truncation-aware):
  gap = base_gap × discount
  discount = 1 - (1 - len_trunc/len_orig) × 0.5
```

### 1.7 Cold Storage Feedback Loop

**v2.0 Flaw:** Cold storage hits don't update graph or hot cache.

**v3.0 Math:**
On cold_storage hit for query Q returning segment s:
1. Graph Edge Strengthening: `w(h, s) ← w(h, s) + η_hit`
2. Promotion Tracking: Promote `s` to Hot Cache if frequently hit and recent.

---

## Part 2: Architecture Components

### 2.1 Module Specifications

#### `jeval/memory/session_aware.py`

```python
class ContextReuseCache:
    """Caches retrieval context by question similarity. (Pure caching, no learning)"""
    def get_or_retrieve(q_uuid, question, retrieve_fn) -> (context, was_cached)

class StaticRetrievalPolicy:
    """Returns fixed k and routing per QA type, learned from training data."""
    def get_k(qa_type: str) -> int
    def get_routing(qa_type: str) -> str

class ConfidenceRetryEscalation:
    """Retries with expanding context on failure, judged by a heuristic."""
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

---

## Part 6: Expected Improvements

| Metric | v2.0 | v3.0 (Projected) | Source of Improvement |
|--------|------|------------------|----------------------|
| General accuracy (Test Set) | ~35% | 40-45% | Training-optimized fixed policy |
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
