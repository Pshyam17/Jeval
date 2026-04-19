from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from jeval.artifacts.detector import is_artifact
from jeval.compress.extractive import ExtractiveBackend
from jeval.encoders.sentence_encoder import FrozenEncoder
from jeval.epe.combined import CombinedEPE
from jeval.memory.anchor_extractor import AnchorExtractor
from jeval.memory.bm25_index import BM25Index
from jeval.memory.co_retrieval_graph import CoRetrievalGraph
from jeval.memory.cold_storage import ColdStorage
from jeval.memory.confidence_gate import ConfidenceGate
from jeval.memory.contradiction_detector import ContradictionDetector
from jeval.memory.fact_index import FactIndex, extract_entities
from jeval.memory.hot_cache import HotCache
from jeval.memory.novelty_gate import NoveltyGate
from jeval.memory.query_classifier import QueryClassifier
from jeval.memory.recompressor import MissTriggeredRecompressor
from jeval.memory.schema_gap import SchemaGapVerifier
from jeval.memory.segmenter import SessionSegmenter
from jeval.memory.timeout_compressor import TimeoutCompressor
from jeval.strata.classifier import ContentClassifier

logger = logging.getLogger(__name__)

# Base budgets per content type — starting point before EPE modulation.
_BASE_BUDGETS: dict[str, float] = {
    "FACTUAL": 0.50,
    "CAUSAL": 0.45,
    "ENTITY": 0.45,
    "TEMPORAL": 0.35,
    "CONTRASTIVE": 0.35,
    "BACKGROUND": 0.20,
}


class _ArtifactDetector:
    def has_artifact(self, text: str) -> bool:
        return is_artifact(text)


class _NIMCaller:
    """Thin wrapper that exposes .call(prompt) -> str for TimeoutCompressor."""

    def __init__(self) -> None:
        try:
            from openai import OpenAI
            api_key = os.environ.get("NVIDIA_API_KEY", "")
            self._client = OpenAI(
                api_key=api_key,
                base_url=os.environ.get("JEVAL_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            )
            self._model = os.environ.get("JEVAL_NIM_MODEL", "mistralai/mistral-small-3.1-24b-instruct-2503")
            self._available = bool(api_key)
        except ImportError:
            self._client = None
            self._available = False

    def call(self, prompt: str) -> str:
        if not self._available or self._client is None:
            raise RuntimeError("NVIDIA_API_KEY not set or openai not installed")
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()


class _ExtractiveCallerWrapper:
    """Wraps ExtractiveBackend as a .call(prompt) compatible object for tests."""

    def __init__(self, budget: float = 0.5):
        self._backend = ExtractiveBackend()
        self._budget = budget

    def call(self, prompt: str) -> str:
        if "Segment:\n" in prompt:
            text = prompt.split("Segment:\n", 1)[1]
        else:
            text = prompt
        return self._backend.compress(text, self._budget)


class JevalMemory:
    def __init__(
        self,
        db_path: Path | str = ".jeval/memory.db",
        hot_cache_token_ceiling: int = 32000,
        novelty_threshold: float = 0.05,
        fidelity_threshold: float = 0.25,
        alpha: float = 0.5,
        beta: float = 1.0,
        high_confidence: float = 0.7,
        low_confidence: float = 0.4,
        miss_threshold: int = 2,
        min_rewrite_gap: int = 5,
        eviction_weights: tuple = (0.3, 0.4, 0.2, 0.1),
        session_id: Optional[str] = None,
        encoder_model: str = "all-mpnet-base-v2",
        _caller=None,
        encoder=None,
        compressor=None,
        classifier=None,
    ):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._encoder = encoder or FrozenEncoder(encoder_model)
        self._session_id = session_id or str(uuid.uuid4())
        self._fidelity_threshold = fidelity_threshold
        self._alpha = alpha
        self._beta = beta
        self._base_budgets = _BASE_BUDGETS
        self._seq_counter = 0
        self._turn_counter = 0
        self._last_budget: float = 0.5

        _jeval_dir = Path(".jeval")
        _jeval_dir.mkdir(parents=True, exist_ok=True)
        self._query_log_path = _jeval_dir / "query_log.jsonl"

        self._cold = ColdStorage(db_path)
        self._fact_index = FactIndex(db_path)
        self._hot_cache = HotCache(
            self._encoder,
            token_ceiling=hot_cache_token_ceiling,
            eviction_weights=eviction_weights,
        )
        self._novelty_gate = NoveltyGate(self._encoder, threshold=novelty_threshold)
        self._anchor_extractor = AnchorExtractor()
        self._contradiction_detector = ContradictionDetector(self._encoder, self._hot_cache)
        self._artifact_detector = _ArtifactDetector()
        self._query_classifier = QueryClassifier()
        self._segmenter = SessionSegmenter()
        self._extractive = ExtractiveBackend()
        self._schema_verifier = SchemaGapVerifier()
        self._bm25 = BM25Index()
        self._graph = CoRetrievalGraph(db_path.parent / "co_retrieval.db")

        if compressor is not None:
            self._compressor = compressor
        else:
            caller = _caller or _NIMCaller()
            self._compressor = TimeoutCompressor(caller, timeout_seconds=30.0)

        self._combined_epe = CombinedEPE(self._encoder, self._schema_verifier, alpha=alpha)
        self._confidence_gate = ConfidenceGate(
            encoder=self._encoder,
            high_threshold=high_confidence,
            low_threshold=low_confidence,
        )
        self._recompressor = MissTriggeredRecompressor(
            compressor=self._compressor,
            encoder=self._encoder,
            cold_storage=self._cold,
            hot_cache=self._hot_cache,
            miss_threshold=miss_threshold,
            min_rewrite_gap=min_rewrite_gap,
        )

        if classifier is not None:
            self._classifier = classifier
            self._classifier_available = True
        else:
            try:
                self._classifier = ContentClassifier()
                self._classifier_available = True
            except Exception:
                self._classifier = None
                self._classifier_available = False

    def _next_seq_id(self) -> int:
        self._seq_counter += 1
        return self._seq_counter

    def ingest(self, text: str) -> dict:
        seq_id = self._next_seq_id()

        # 1. cold storage — unconditional
        cold_id = self._cold.append(text, seq_id, self._session_id)

        # 2. novelty gate
        is_novel, epe_novelty = self._novelty_gate.is_novel(text)
        if not is_novel:
            return {
                "action": "cold_only",
                "seq_id": seq_id,
                "epe_novelty": epe_novelty,
                "epe_fidelity": None,
                "cosine_epe": None,
                "schema_gap": None,
                "epe_final": None,
                "content_type": None,
                "budget": None,
                "anchors": [],
                "token_count_original": len(text.split()),
                "token_count_compressed": None,
            }

        # 3. entity extraction + fact index
        try:
            entities = extract_entities(text)
            self._fact_index.write_entities(entities, seq_id, cold_id, self._session_id)
        except ImportError:
            entities = []

        self._anchor_extractor.update_corpus(text)

        # 4. content-type classification
        if self._classifier_available:
            content_type = self._classifier.top_label(text)
        else:
            content_type = "BACKGROUND"

        # 5. anchor extraction
        anchors = self._anchor_extractor.extract(text, entities)

        # 6. reference count override check
        has_high_ref = any(
            self._fact_index.get_ref_count(e["text"], self._session_id) >= 3
            for e in entities
        )

        # 7. LLM compression with fidelity gate
        compressed, action = self._compress_with_fidelity_gate(
            text, content_type, anchors, has_high_ref
        )

        # 8. combined EPE on committed compression
        cosine_epe, schema_gap, epe_final = self._combined_epe.compute(
            text, compressed, content_type
        )

        # 9. hot cache write
        orig_emb = self._encoder.encode([text])[0]
        comp_emb = self._encoder.encode([compressed])[0]
        self._hot_cache.store(
            text=compressed,
            embedding=comp_emb,
            seq_id=seq_id,
            content_type=content_type,
            epe_score=epe_novelty,
            metadata={
                "cosine_epe": cosine_epe,
                "schema_gap": schema_gap,
                "epe_final": epe_final,
                "original_seq_id": seq_id,
                "budget": self._last_budget,
            },
        )
        self._novelty_gate.update_working_set(text, orig_emb)

        # 10. BM25 index update
        self._bm25.add(str(seq_id), text)

        # 11. async: contradiction detection + recompressor check
        threading.Thread(
            target=self._async_post_ingest,
            args=(text, seq_id, self._turn_counter),
            daemon=True,
        ).start()

        self._turn_counter += 1

        return {
            "action": action,
            "seq_id": seq_id,
            "epe_novelty": epe_novelty,
            "epe_fidelity": cosine_epe,
            "cosine_epe": cosine_epe,
            "schema_gap": schema_gap,
            "epe_final": epe_final,
            "content_type": content_type,
            "budget": self._last_budget,
            "anchors": anchors,
            "token_count_original": len(text.split()),
            "token_count_compressed": len(compressed.split()),
            "compressed_text": compressed,
        }

    def _compress_with_fidelity_gate(
        self,
        text: str,
        content_type: str,
        anchors: list[str],
        has_high_ref: bool,
    ) -> tuple[str, str]:
        schema_type = (
            content_type
            if content_type in self._schema_verifier._compiled
            else self._schema_verifier.detect_schema_type(text)
        )
        schema_gap_original = self._schema_verifier.compute_gap(text, schema_type)
        cosine_epe_proxy = 1.0

        epe_for_budget = self._alpha * cosine_epe_proxy + (1.0 - self._alpha) * schema_gap_original

        base = self._base_budgets.get(content_type.upper(), 0.5)
        budget = min(base * (1.0 + self._beta * epe_for_budget), 1.0)

        if self._artifact_detector.has_artifact(text) or has_high_ref:
            budget = 1.0

        self._last_budget = budget

        orig_emb = self._encoder.encode([text])[0]
        for attempt in range(3):
            current_budget = min(budget + 0.2 * attempt, 1.0)
            try:
                candidate = self._compressor.compress(text, current_budget, anchors)
            except (TimeoutError, RuntimeError):
                return self._extractive_fallback(text, budget), "extractive_fallback"

            cand_emb = self._encoder.encode([candidate])[0]
            fidelity_epe = float(1.0 - np.dot(orig_emb, cand_emb))

            if fidelity_epe <= self._fidelity_threshold:
                return candidate, "cached"

        return self._extractive_fallback(text, budget), "extractive_fallback"

    def _extractive_fallback(self, text: str, budget: float) -> str:
        return text  # keep original when LLM unavailable

    def _async_post_ingest(self, text: str, seq_id: int, turn: int) -> None:
        stale_ids = self._contradiction_detector.check(text, self._session_id)
        for sid in stale_ids:
            if sid != seq_id:
                self._hot_cache.mark_stale(sid)

    def retrieve(self, query: str, k: int = 5) -> str:
        """Three-pass retrieval: cosine + BM25 + co-retrieval graph."""
        # Pass 1: cosine from hot cache
        hot_results = self._hot_cache.retrieve(query, k=k * 2)
        hot_ids = [str(r["seq_id"]) for r in hot_results]

        # Pass 2: BM25 lexical search
        bm25_results = self._bm25.search(query, top_k=k * 2)
        bm25_ids = [seg_id for seg_id, _ in bm25_results]

        # Pass 3: graph walk from seed nodes
        seed_ids = list(dict.fromkeys(hot_ids + bm25_ids))
        graph_ids = []
        for sid in seed_ids[:5]:
            neighbors = self._graph.get_neighbors(sid, min_weight=0.2, top_k=3)
            graph_ids += [n[0] for n in neighbors]

        # Merge all candidate IDs
        all_ids = list(dict.fromkeys(seed_ids + graph_ids))

        # Record co-retrieval for graph learning
        if seed_ids:
            self._graph.record_retrieval(seed_ids[:k])

        # Fetch content — prefer hot cache, fall back to cold storage
        hot_map = {str(r["seq_id"]): r["text"] for r in hot_results}
        results = []
        for sid in all_ids[:k]:
            if sid in hot_map:
                results.append(hot_map[sid])
            else:
                try:
                    cold = self._cold.get_by_seq_id(int(sid), self._session_id)
                    if cold:
                        results.append(cold["content"])
                except Exception:
                    pass

        if results:
            return "\n".join(results)

        # Final fallback: cold storage search
        return self._cold_fallback(query, k)

    def memory_retrieve(self, query: str, top_k: int = 5) -> str:
        """Alias for retrieve() — AMA-Bench interface."""
        return self.retrieve(query, k=top_k)

    def _cold_fallback(self, query: str, k: int) -> str:
        cold = self._cold.search(query, limit=k)
        self._log_query(query, "context", "cold_storage", 0.0, -1)
        if not cold:
            cold = self._cold.search("", limit=k)
        return self._format_cold(cold)

    def _format_hot(self, results: list[dict]) -> str:
        return "\n".join(r["text"] for r in results)

    def _format_cold(self, cold_results: list[dict]) -> str:
        return "\n".join(r.get("content", "") for r in cold_results)

    def _format_merged(self, hot_results: list[dict], cold_results: list[dict]) -> str:
        parts = [r["text"] for r in hot_results]
        parts += [r.get("content", "") for r in cold_results]
        return "\n".join(parts)

    def _log_query(
        self,
        query: str,
        query_type: str,
        routing: str,
        confidence: float,
        seq_id: int,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self._session_id,
            "query": query,
            "query_type": query_type,
            "routing": routing,
            "confidence": confidence,
            "seq_id_hit": seq_id,
            "turn": self._turn_counter,
        }
        try:
            with open(self._query_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.warning("jeval_memory: failed to write query log: %s", exc)

    def _precision_retrieve(self, query: str) -> str:
        """Direct cold-storage lookup for precision queries."""
        seq_id = self._query_classifier.extract_seq_id(query)
        precision_results: list[dict] = []
        if seq_id is not None:
            cold = self._cold.get_by_seq_id(seq_id, self._session_id)
            if cold:
                precision_results.append(cold)
        entity_hint = self._query_classifier.extract_entity_hint(query)
        if entity_hint:
            facts = self._fact_index.get_by_entity(entity_hint, self._session_id)
            for f in facts:
                cold = self._cold.get_by_id(f["segment_id"])
                if cold:
                    precision_results.append(cold)
        if precision_results:
            return self._query_classifier.merge_results(precision_results, [])
        return self._format_cold(self._cold.search(query, limit=5))

    def compress(self, session_text: str) -> str:
        segs = self._segmenter.segment(session_text)
        for seg in segs:
            self.ingest(seg)
        entries = self._hot_cache.get_all_entries()
        return "\n".join(f"[{e['seq_id']}] {e['text']}" for e in entries)

    def stats(self) -> dict:
        return {
            "hot_cache_size": self._hot_cache.size(),
            "hot_cache_tokens": self._hot_cache.token_count(),
            "cold_storage_size": self._cold.count(self._session_id),
            "fact_index_size": self._fact_index.count(self._session_id),
            "session_id": self._session_id,
            "novelty_threshold": self._novelty_gate._threshold,
            "fidelity_threshold": self._fidelity_threshold,
            "alpha": self._alpha,
            "beta": self._beta,
        }

    def new_session(self, session_id: Optional[str] = None) -> None:
        self._session_id = session_id or str(uuid.uuid4())
        self._seq_counter = 0
        self._turn_counter = 0
        self._hot_cache = HotCache(
            self._encoder,
            token_ceiling=self._hot_cache._token_ceiling,
            top_k=self._hot_cache._top_k,
            eviction_weights=self._hot_cache._eviction_weights,
        )
        self._novelty_gate.clear()
        self._anchor_extractor.reset()
        self._bm25.clear()
        self._contradiction_detector = ContradictionDetector(
            self._encoder, self._hot_cache
        )
        self._recompressor._hot_cache = self._hot_cache
