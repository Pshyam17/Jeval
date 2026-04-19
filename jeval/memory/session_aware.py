"""
Session-aware memory components for v3.0 architecture.

Implements:
- ContextReuseCache: Caches retrieval context by question similarity (§1.7)
- StaticRetrievalPolicy: Returns fixed k and routing per QA type (§1.8)
- ConfidenceRetryEscalation: Retries with budget-aware escalation (§1.9)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from jeval.memory.entity_extraction import extract_entity_texts

if TYPE_CHECKING:
    from jeval.memory.jeval_memory import JevalMemory


# =============================================================================
# CONFIGURATION DEFAULTS (from architecture_v3.md Appendix A)
# =============================================================================

DEFAULT_K_BASE = {
    "factoid": 1,       # Single fact lookup
    "verification": 2,  # Confirm true/false
    "procedural": 3,    # Step-by-step how-to
    "causal": 5,        # Why/reasoning questions
    "entity": 3,        # Named entity lookups
    "summary": 8,       # Broad overview
    "unknown": 5,       # Fallback
}

DEFAULT_ROUTING_BASE = {
    "factoid": "hot_cache",      # Recent facts in hot cache
    "verification": "hot_cache", # Recent state in hot cache
    "procedural": "enriched",    # Need steps from cold
    "causal": "enriched",        # Need reasoning chains
    "entity": "hot_cache",       # Recent entities in hot cache
    "summary": "enriched",       # Broad context needed
    "unknown": "hot_cache",      # Default to fast path
}

THRESHOLD_REUSE = 0.85                  # θ_reuse: cosine sim for cache reuse
THRESHOLD_QUERY_CONFIDENCE = 0.60       # θ_query: min C_q for hot-cache route
THRESHOLD_ANSWER_CONFIDENCE = 0.70      # θ_confidence_stop: stop retrying
THRESHOLD_ANSWER_CONFIDENCE_LOW = 0.40  # Below this: escalate to cold
WEIGHT_EVIDENCE_SUPPORT = 0.50          # w1: answer grounded in context
WEIGHT_QUERY_COVERAGE = 0.30            # w2: answer addresses query
WEIGHT_HOT_TRUST = 0.20                 # w3: trust in hot-cache entries
RETRY_MAX_ATTEMPTS = 3
RETRY_K_MULTIPLIER = 2.0


@dataclass
class CachedContext:
    """Stored context for reuse checking."""
    question: str
    context: str
    embedding: np.ndarray


class ContextReuseCache:
    """
    Caches retrieval context by question similarity.
    Pure caching, no learning from ground truth (§1.7).

    Cache: q_uuid → (question, context, question_embedding)

    For new query q_new:
    1. Encode: e_new = enc(q_new)
    2. Find best match: q_cached = argmax_{q in cache} cosine_sim(e_new, cache[q].embedding)
    3. If cosine_sim(q_new, q_cached) > θ_reuse:
         return cache[q_cached].context  # Cache hit
       Else:
         context = Retrieve(q_new, H, C, G)
         cache[q_new] = (q_new, context, e_new)  # Cache miss, store for future

    Threshold: θ_reuse = 0.85
    - High bar prevents stale or weakly-related context from being reused
    - Expected hit rate: 20-40% for related QAs in same episode
    """

    def __init__(
        self,
        encoder,
        threshold: float = THRESHOLD_REUSE,
        max_cache_size: int = 100,
    ):
        self._encoder = encoder
        self._threshold = threshold
        self._max_cache_size = max_cache_size
        self._cache: dict[str, CachedContext] = {}
        self._insertion_order: list[str] = []  # For LRU eviction

    def get_or_retrieve(
        self,
        q_uuid: str,
        question: str,
        retrieve_fn: Callable[[str], str],
    ) -> tuple[str, bool]:
        """
        Returns (context, was_cached).

        If question embedding is similar enough to a cached question,
        return the cached context. Otherwise, call retrieve_fn and cache.
        """
        # Check for exact UUID match first (same question asked again)
        if q_uuid in self._cache:
            return self._cache[q_uuid].context, True

        # Encode new question
        new_emb = self._encoder.encode([question])[0]

        # Find best match in cache
        best_sim = 0.0
        best_uuid: Optional[str] = None

        for cached_uuid, cached_ctx in self._cache.items():
            sim = float(np.dot(new_emb, cached_ctx.embedding))
            if sim > best_sim:
                best_sim = sim
                best_uuid = cached_uuid

        # Check threshold
        if best_sim > self._threshold and best_uuid is not None:
            # Cache hit - reuse context from similar question
            return self._cache[best_uuid].context, True

        # Cache miss - retrieve and store
        context = retrieve_fn(question)
        self._add_to_cache(q_uuid, question, context, new_emb)
        return context, False

    def _add_to_cache(
        self,
        q_uuid: str,
        question: str,
        context: str,
        embedding: np.ndarray,
    ) -> None:
        """Add entry to cache with LRU eviction."""
        # Evict if at capacity
        if len(self._cache) >= self._max_cache_size:
            oldest_uuid = self._insertion_order.pop(0)
            del self._cache[oldest_uuid]

        self._cache[q_uuid] = CachedContext(
            question=question,
            context=context,
            embedding=embedding,
        )
        self._insertion_order.append(q_uuid)

    def clear(self) -> None:
        """Clear all cached contexts."""
        self._cache.clear()
        self._insertion_order.clear()


class StaticRetrievalPolicy:
    """
    Returns fixed k and routing per QA type (§1.8).

    Policy: π(qa_type) → (k, routing)

    Default values (learned offline, frozen at deployment):
    - k_base: retrieval depth per type
    - routing_base: retrieval route per type

    Routing options:
    - "hot_cache": cosine retrieval from hot cache only
    - "enriched": hot cache + cold storage append
    - "cold_storage": cold storage search only
    """

    def __init__(
        self,
        k_base: dict = None,
        routing_base: dict = None,
    ):
        self._k_base = k_base or DEFAULT_K_BASE
        self._routing_base = routing_base or DEFAULT_ROUTING_BASE

    def get_k(self, qa_type: str) -> int:
        """Return retrieval depth for QA type."""
        return self._k_base.get(qa_type, self._k_base["unknown"])

    def get_routing(self, qa_type: str) -> str:
        """Return routing strategy for QA type."""
        return self._routing_base.get(qa_type, self._routing_base["unknown"])

    def get_qa_type(self, question: str) -> str:
        """
        Classify question into type based on keywords and structure.

        Classification rules (ordered by priority):
        - entity: mentions specific files, error types, step numbers (checked first)
        - verification: "is", "does", "can", "should", "true/false"
        - procedural: "how to", "steps", "procedure", "process"
        - causal: "why", "reason", "cause", "because"
        - factoid: "what is", "define", "name of", "when"
        - summary: "summarize", "overview", "describe"
        """
        q_lower = question.lower()

        # Entity questions FIRST (mentions technical identifiers)
        # This must come before factoid since "What is in src/config/env.ts?" is entity-focused
        entities = extract_entity_texts(question)
        if entities:
            # Check if entities are file paths, error types, or step refs
            for e in entities:
                if '.' in e or 'error' in e or 'step' in e or re.match(r'\d+', e):
                    return "entity"

        # Verification questions (yes/no, true/false)
        if re.search(r'\b(is|does|do|can|could|should|would|will|are|was|were)\b', q_lower):
            if re.search(r'\b(true|false|correct|right)\b', q_lower):
                return "verification"
            # Check if it's a simple yes/no question
            if q_lower.startswith(('is ', 'does ', 'do ', 'can ', 'could ', 'should ')):
                return "verification"

        # Procedural questions
        if re.search(r'\b(how to|how do|steps?|procedure|process|method|way to)\b', q_lower):
            return "procedural"

        # Causal questions
        if re.search(r'\b(why|reason|cause|because|due to|result of)\b', q_lower):
            return "causal"

        # Summary questions
        if re.search(r'\b(summarize|overview|describe|explain|tell me about)\b', q_lower):
            return "summary"

        # Factoid questions
        if re.search(r'\b(what is|what are|define|name of|when did|when was|who is|who was)\b', q_lower):
            return "factoid"

        return "unknown"


@dataclass
class ConfidenceResult:
    """Result of confidence computation."""
    evidence_support: float  # C_sup
    query_coverage: float    # C_cov
    hot_trust: float         # C_trust
    overall: float           # C_a


class ConfidenceRetryEscalation:
    """
    Retries with budget-aware escalation, judged by deployable confidence (§1.9).

    For each QA:
    Attempt 1:
      k = k_base[qa_type]
      routing = routing_base[qa_type]
      Generate answer a_1 from context c_1
      confidence = C_a(q, c_1, a_1)
      If confidence ≥ θ_confidence_stop: return a_1

    Attempt 2 (if C_a < 0.70):
      routing = escalate_route(routing_base[qa_type])
      k = min(k_base[qa_type] * 2, k_max_remaining)
      Generate answer a_2 from context c_2
      confidence = C_a(q, c_2, a_2)
      If confidence ≥ θ_confidence_stop: return a_2

    Attempt 3 (if C_a < 0.70):
      routing = "cold_storage"
      k = k_max_remaining
      Generate answer a_3 from context c_3
      return a_3  # Best effort, no more retries

    Escalation function:
    - escalate_route("hot_cache") → "enriched"
    - escalate_route("enriched") → "cold_storage"
    - escalate_route("cold_storage") → "cold_storage"
    """

    def __init__(
        self,
        max_attempts: int = RETRY_MAX_ATTEMPTS,
        confidence_threshold: float = THRESHOLD_ANSWER_CONFIDENCE,
    ):
        self._max_attempts = max_attempts
        self._threshold = confidence_threshold
        self._low_threshold = THRESHOLD_ANSWER_CONFIDENCE_LOW

    def execute(
        self,
        question: str,
        mem: JevalMemory,
        qa_type: str,
        policy: StaticRetrievalPolicy,
        llm_fn: Callable[[str, str], str],
        confidence_fn: Callable[[str, str, str], ConfidenceResult],
        task: str = "",
    ) -> tuple[str, int]:
        """
        Returns (answer, attempts_made).

        Executes retry loop with escalation until confidence threshold met
        or max attempts reached.
        """
        routing = policy.get_routing(qa_type)
        k = policy.get_k(qa_type)
        k_max = 8  # Maximum retrieval depth

        for attempt in range(1, self._max_attempts + 1):
            # Get context with current routing and k
            context = self._retrieve_with_routing(mem, question, k, routing)

            # Generate answer
            answer = llm_fn(question, context)

            # Compute confidence
            confidence = confidence_fn(question, context, answer)

            if confidence.overall >= self._threshold:
                # Confident enough - return answer
                return answer, attempt

            # Not confident - escalate for next attempt
            if attempt < self._max_attempts:
                routing = self._escalate_route(routing)
                k = min(k * RETRY_K_MULTIPLIER, k_max)

        # Max attempts reached - return last answer
        return answer, self._max_attempts

    def _retrieve_with_routing(
        self,
        mem: JevalMemory,
        query: str,
        k: int,
        routing: str,
    ) -> str:
        """Route retrieval based on strategy."""
        if hasattr(mem, 'retrieve_with_routing'):
            return mem.retrieve_with_routing(query, k=k, routing_preference=routing)
        else:
            # Fallback to standard retrieve
            return mem.retrieve(query, k=k)

    def _escalate_route(self, routing: str) -> str:
        """Escalate to more comprehensive route."""
        escalation_map = {
            "hot_cache": "enriched",
            "enriched": "cold_storage",
            "cold_storage": "cold_storage",
        }
        return escalation_map.get(routing, "cold_storage")


def compute_query_confidence(
    question: str,
    hot_results: list[dict],
) -> float:
    """
    Compute query confidence C_q (§1.3).

    C_q(q) = |E(q) ∩ E(H_k(q))| / max(|E(q)|, 1)

    Interpretation:
    - C_q = 1.0: All query entities appear in hot-cache results
    - C_q = 0.0: No query entities in hot cache
    """
    question_entities = extract_entity_texts(question)

    if not question_entities:
        # No entities extracted - default to high confidence
        # (generic questions can be answered from hot cache)
        return 1.0

    # Extract entities from hot results
    hot_entities: set[str] = set()
    for result in hot_results:
        text = result.get("text", "")
        hot_entities.update(extract_entity_texts(text))

    # Compute overlap
    overlap = question_entities & hot_entities
    confidence = len(overlap) / max(len(question_entities), 1)

    return confidence


def compute_answer_confidence(
    question: str,
    context: str,
    answer: str,
    hot_trust: float = 1.0,
) -> ConfidenceResult:
    """
    Compute answer confidence C_a (§1.4).

    Evidence support:
      C_sup = |E(a_n) ∩ E(c_n)| / max(|E(a_n)|, 1)
      → What fraction of answer entities are grounded in retrieved context?

    Query coverage:
      C_cov = |E(q) ∩ E(a_n)| / max(|E(q)|, 1)
      → Does the answer address all query entities?

    Hot trust:
      C_trust = 1 − mean(R_build over hot entries used in c_n)
      → How much trust do we have in the hot-cache entries we retrieved?

    Answer confidence:
      C_a = clip(w1*C_sup + w2*C_cov + w3*C_trust, [0, 1])
    """
    answer_entities = extract_entity_texts(answer)
    context_entities = extract_entity_texts(context)
    question_entities = extract_entity_texts(question)

    # Evidence support: answer entities grounded in context
    if answer_entities:
        evidence_support = len(answer_entities & context_entities) / len(answer_entities)
    else:
        evidence_support = 1.0  # No entities = nothing to ground

    # Query coverage: answer addresses question entities
    if question_entities:
        query_coverage = len(question_entities & answer_entities) / len(question_entities)
    else:
        query_coverage = 1.0  # No entities = full coverage

    # Hot trust: provided by caller (computed from R_build metadata)
    hot_trust = max(0.0, min(1.0, hot_trust))

    # Weighted combination
    overall = (
        WEIGHT_EVIDENCE_SUPPORT * evidence_support +
        WEIGHT_QUERY_COVERAGE * query_coverage +
        WEIGHT_HOT_TRUST * hot_trust
    )
    overall = max(0.0, min(1.0, overall))

    return ConfidenceResult(
        evidence_support=evidence_support,
        query_coverage=query_coverage,
        hot_trust=hot_trust,
        overall=overall,
    )
