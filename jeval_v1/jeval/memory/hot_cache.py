from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

from jeval.encoders.sentence_encoder import FrozenEncoder

logger = logging.getLogger(__name__)


class HotCache:
    """
    In-memory compressed narrative store with embedding index.
    Bounded by token ceiling. Eviction uses a multi-factor score that
    prioritises evicting high-miss, low-hit, redundant (low epe_novelty) entries.
    """

    def __init__(
        self,
        encoder: FrozenEncoder,
        token_ceiling: int = 8000,
        top_k: int = 5,
        eviction_weights: tuple[float, float, float, float] = (0.3, 0.4, 0.2, 0.1),
    ):
        self._encoder = encoder
        self._token_ceiling = token_ceiling
        self._top_k = top_k
        # weights: (time, miss, hit, novelty)
        self._eviction_weights = eviction_weights
        self._entries: list[dict] = []
        self._lock = threading.Lock()

    def store(
        self,
        text: str,
        embedding: np.ndarray,
        seq_id: int,
        content_type: str,
        epe_score: float,
        metadata: Optional[dict] = None,
        access_turn: int = 0,
    ) -> None:
        entry = {
            "text": text,
            "embedding": embedding,
            "seq_id": seq_id,
            "content_type": content_type,
            # epe_score = novelty EPE from novelty gate (higher = more novel)
            "epe_score": epe_score,
            "epe_novelty": epe_score,   # alias for eviction formula
            "metadata": metadata or {},
            "stale": False,
            "token_count": len(text.split()),
            # per-entry counters for eviction formula
            "miss_counter": 0,
            "hit_count": 0,
            "last_accessed_turn": access_turn,
        }
        with self._lock:
            self._entries.append(entry)

        if self.token_count() > 0.80 * self._token_ceiling:
            threading.Thread(target=self._enforce_token_budget, daemon=True).start()

    def retrieve(self, query: str, k: Optional[int] = None) -> list[dict]:
        k = k if k is not None else self._top_k
        with self._lock:
            active = [e for e in self._entries if not e["stale"]]
        if not active:
            return []
        query_emb = self._encoder.encode([query])[0]
        scored = [
            (float(np.dot(query_emb, e["embedding"])), e) for e in active
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "text": e["text"],
                "seq_id": e["seq_id"],
                "content_type": e["content_type"],
                "epe_score": e["epe_score"],
                "similarity": sim,
                "metadata": e["metadata"],
            }
            for sim, e in scored[:k]
        ]

    def mark_stale(self, seq_id: int) -> None:
        with self._lock:
            for e in self._entries:
                if e["seq_id"] == seq_id:
                    e["stale"] = True

    def evict_stale(self) -> int:
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if not e["stale"]]
            return before - len(self._entries)

    def token_count(self) -> int:
        with self._lock:
            return sum(e["token_count"] for e in self._entries if not e["stale"])

    def size(self) -> int:
        with self._lock:
            return sum(1 for e in self._entries if not e["stale"])

    def get_all_embeddings(self) -> tuple[np.ndarray, list[int]]:
        with self._lock:
            active = [e for e in self._entries if not e["stale"]]
        if not active:
            return np.empty((0, self._encoder.dim()), dtype=np.float32), []
        embs = np.stack([e["embedding"] for e in active])
        seq_ids = [e["seq_id"] for e in active]
        return embs, seq_ids

    def get_all_entries(self) -> list[dict]:
        with self._lock:
            return [e for e in self._entries if not e["stale"]]

    def _eviction_score(
        self,
        entry: dict,
        max_time: float,
        max_miss: float,
        max_hit: float,
    ) -> float:
        """
        eviction_score = w_t * time_normalised
                       + w_m * miss_normalised
                       - w_h * hit_normalised
                       + w_n * (1 - epe_novelty)

        Higher score → evict first. All terms normalised to [0, 1] across current
        cache entries before this call; max values are passed in to avoid
        recomputing them per entry.
        """
        w_t, w_m, w_h, w_n = self._eviction_weights

        time_n = (entry["last_accessed_turn"] / max_time) if max_time > 0 else 0.0
        miss_n = (entry["miss_counter"] / max_miss) if max_miss > 0 else 0.0
        hit_n  = (entry["hit_count"] / max_hit) if max_hit > 0 else 0.0
        novelty = float(entry.get("epe_novelty", entry.get("epe_score", 0.5)))

        return w_t * time_n + w_m * miss_n - w_h * hit_n + w_n * (1.0 - novelty)

    def _enforce_token_budget(self) -> None:
        target = 0.70 * self._token_ceiling
        evicted = 0
        with self._lock:
            active = [e for e in self._entries if not e["stale"]]
            if not active:
                return

            # compute normalisation denominators across current active set
            max_time = max(e["last_accessed_turn"] for e in active) or 1.0
            max_miss = max(e["miss_counter"] for e in active) or 1.0
            max_hit  = max(e["hit_count"] for e in active) or 1.0

            # sort active entries by eviction score descending (highest = evict first)
            scored = sorted(
                active,
                key=lambda e: self._eviction_score(e, max_time, max_miss, max_hit),
                reverse=True,
            )

            for entry in scored:
                current = sum(
                    e["token_count"] for e in self._entries if not e["stale"]
                )
                if current <= target:
                    break
                # find and mark stale in-place
                for e in self._entries:
                    if e["seq_id"] == entry["seq_id"] and not e["stale"]:
                        e["stale"] = True
                        evicted += 1
                        break

        if evicted:
            logger.debug(
                "hot_cache eviction: removed %d entries, token_count=%d",
                evicted,
                self.token_count(),
            )
