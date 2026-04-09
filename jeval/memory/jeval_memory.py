from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from jeval.artifacts.detector import is_artifact
from jeval.compress.extractive import ExtractiveBackend
from jeval.encoders.sentence_encoder import FrozenEncoder
from jeval.memory.anchor_extractor import AnchorExtractor
from jeval.memory.cold_storage import ColdStorage
from jeval.memory.contradiction_detector import ContradictionDetector
from jeval.memory.fact_index import FactIndex, extract_entities
from jeval.memory.hot_cache import HotCache
from jeval.memory.novelty_gate import NoveltyGate
from jeval.memory.query_classifier import QueryClassifier
from jeval.memory.segmenter import SessionSegmenter
from jeval.memory.timeout_compressor import TimeoutCompressor
from jeval.strata.classifier import ContentClassifier

_CONTENT_FLOOR: dict[str, float] = {
    "FACTUAL": 0.75,
    "CAUSAL": 0.70,
    "ENTITY": 0.70,
    "TEMPORAL": 0.55,
    "CONTRASTIVE": 0.55,
    "BACKGROUND": 0.20,
}


class _MemoryBudgetAllocator:
    def allocate(
        self,
        content_type: str,
        epe_score: float,
        has_artifact: bool,
        has_high_ref_entity: bool,
    ) -> float:
        if has_artifact or has_high_ref_entity:
            return 1.0
        floor = _CONTENT_FLOOR.get(content_type.upper(), 0.20)
        raw = floor + 0.25 * epe_score
        return max(0.30, min(1.0, raw))


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
                base_url="https://integrate.api.nvidia.com/v1",
            )
            self._model = "mistralai/mistral-small-3.1-24b-instruct-2503"
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
        # extract segment text from prompt
        if "Segment:\n" in prompt:
            text = prompt.split("Segment:\n", 1)[1]
        else:
            text = prompt
        return self._backend.compress(text, self._budget)


class JevalMemory:
    def __init__(
        self,
        db_path: Path | str = ".jeval/memory.db",
        hot_cache_token_ceiling: int = 8000,
        novelty_threshold: float = 0.15,
        fidelity_threshold: float = 0.20,
        session_id: Optional[str] = None,
        encoder_model: str = "all-mpnet-base-v2",
        _caller=None,  # injectable for testing: any object with .call(prompt) -> str
        encoder=None,  # accept shared FrozenEncoder instance; avoids duplicate model load
        compressor=None,  # accept shared compressor with .compress(text, budget, anchors) -> str
    ):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # use shared encoder if provided, otherwise create one from encoder_model
        self._encoder = encoder or FrozenEncoder(encoder_model)
        self._session_id = session_id or str(uuid.uuid4())
        self._fidelity_threshold = fidelity_threshold
        self._seq_counter = 0

        self._cold = ColdStorage(db_path)
        self._fact_index = FactIndex(db_path)
        self._hot_cache = HotCache(self._encoder, token_ceiling=hot_cache_token_ceiling)
        self._novelty_gate = NoveltyGate(self._encoder, threshold=novelty_threshold)
        self._anchor_extractor = AnchorExtractor()
        self._contradiction_detector = ContradictionDetector(self._encoder, self._hot_cache)
        self._budget_allocator = _MemoryBudgetAllocator()
        self._artifact_detector = _ArtifactDetector()
        self._query_classifier = QueryClassifier()
        self._segmenter = SessionSegmenter()
        self._extractive = ExtractiveBackend()

        # use shared compressor if provided, otherwise build one via TimeoutCompressor
        if compressor is not None:
            self._compressor = compressor
        else:
            caller = _caller or _NIMCaller()
            self._compressor = TimeoutCompressor(caller, timeout_seconds=3.0)

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

        # 1. unconditional cold storage write
        cold_id = self._cold.append(text, seq_id, self._session_id)

        # 2. novelty gate
        is_novel, epe_novelty = self._novelty_gate.is_novel(text)
        if not is_novel:
            return {
                "action": "cold_only",
                "seq_id": seq_id,
                "epe_novelty": epe_novelty,
                "epe_fidelity": None,
                "content_type": None,
                "budget": None,
                "anchors": [],
                "token_count_original": len(text.split()),
                "token_count_compressed": None,
            }

        # 3. update anchor extractor corpus
        self._anchor_extractor.update_corpus(text)

        # 4. entity extraction and fact index write
        try:
            entities = extract_entities(text)
            self._fact_index.write_entities(entities, seq_id, cold_id, self._session_id)
        except ImportError:
            entities = []

        # 5. content-type classification
        if self._classifier_available:
            content_type = self._classifier.top_label(text)
        else:
            content_type = "BACKGROUND"

        # 6. anchor extraction
        anchors = self._anchor_extractor.extract(text, entities)

        # 7. high-reference entity check
        has_high_ref = any(
            self._fact_index.get_ref_count(e["text"], self._session_id) >= 3
            for e in entities
        )

        # 8. budget allocation
        budget = self._budget_allocator.allocate(
            content_type=content_type,
            epe_score=epe_novelty,
            has_artifact=self._artifact_detector.has_artifact(text),
            has_high_ref_entity=has_high_ref,
        )

        # 9. compression with fidelity gate
        compressed, epe_fidelity, action = self._compress_with_fidelity_gate(
            text, budget, anchors
        )

        # 10. store in hot cache; novelty gate tracks original-text embedding
        # so that re-ingesting the same original text is correctly detected
        # as redundant regardless of how the compressor paraphrased it.
        orig_emb = self._encoder.encode([text])[0]
        comp_emb = self._encoder.encode([compressed])[0]
        self._hot_cache.store(compressed, comp_emb, seq_id, content_type, epe_novelty)
        self._novelty_gate.update_working_set(text, orig_emb)

        # 11. async contradiction detection (fire and forget)
        threading.Thread(
            target=self._run_contradiction_check,
            args=(text, seq_id),
            daemon=True,
        ).start()

        return {
            "action": action,
            "seq_id": seq_id,
            "epe_novelty": epe_novelty,
            "epe_fidelity": epe_fidelity,
            "content_type": content_type,
            "budget": budget,
            "anchors": anchors,
            "token_count_original": len(text.split()),
            "token_count_compressed": len(compressed.split()),
            "compressed_text": compressed,
        }

    def _compress_with_fidelity_gate(
        self,
        text: str,
        budget: float,
        anchors: list[str],
        max_retries: int = 2,
    ) -> tuple[str, float, str]:
        orig_emb = self._encoder.encode([text])[0]

        for attempt in range(max_retries + 1):
            current_budget = min(budget + 0.2 * attempt, 1.0)
            try:
                candidate = self._compressor.compress(text, current_budget, anchors)
            except (TimeoutError, RuntimeError):
                candidate = self._extractive_fallback(text, budget)
                cand_emb = self._encoder.encode([candidate])[0]
                epe = float(1.0 - np.dot(orig_emb, cand_emb))
                return candidate, epe, "extractive_fallback"

            cand_emb = self._encoder.encode([candidate])[0]
            epe = float(1.0 - np.dot(orig_emb, cand_emb))

            if epe <= self._fidelity_threshold:
                return candidate, epe, "cached"

        candidate = self._extractive_fallback(text, budget)
        cand_emb = self._encoder.encode([candidate])[0]
        epe = float(1.0 - np.dot(orig_emb, cand_emb))
        return candidate, epe, "extractive_fallback"

    def _extractive_fallback(self, text: str, budget: float) -> str:
        return self._extractive.compress(text, budget)

    def _run_contradiction_check(self, text: str, seq_id: int) -> None:
        stale_ids = self._contradiction_detector.check(text, self._session_id)
        for sid in stale_ids:
            if sid != seq_id:
                self._hot_cache.mark_stale(sid)

    def retrieve(self, query: str, k: int = 5) -> str:
        query_type = self._query_classifier.classify(query)

        precision_results: list[dict] = []
        context_results: list[dict] = []

        if query_type in ("precision", "ambiguous"):
            seq_id = self._query_classifier.extract_seq_id(query)
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

        if query_type in ("context", "ambiguous", "entity"):
            context_results = self._hot_cache.retrieve(query, k=k)
            if not context_results:
                context_results = self._cold.search(query, limit=k)

        if not precision_results and not context_results:
            cold_results = self._cold.search(query, limit=k)
            return self._format_cold(cold_results)

        return self._query_classifier.merge_results(precision_results, context_results)

    def _format_cold(self, cold_results: list[dict]) -> str:
        return "\n".join(r.get("content", "") for r in cold_results)

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
        }

    def new_session(self, session_id: Optional[str] = None) -> None:
        self._session_id = session_id or str(uuid.uuid4())
        self._seq_counter = 0
        self._hot_cache = HotCache(
            self._encoder,
            token_ceiling=self._hot_cache._token_ceiling,
            top_k=self._hot_cache._top_k,
        )
        self._novelty_gate.clear()
        self._anchor_extractor.reset()
        # rebuild contradiction detector pointing at new hot_cache
        self._contradiction_detector = ContradictionDetector(
            self._encoder, self._hot_cache
        )
