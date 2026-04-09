from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from jeval.encoders.sentence_encoder import FrozenEncoder


class HotCache:
    """
    In-memory compressed narrative store with embedding index.
    Bounded by token ceiling. Eviction targets lowest-EPE (most redundant) entries.
    """

    def __init__(
        self,
        encoder: FrozenEncoder,
        token_ceiling: int = 8000,
        top_k: int = 5,
    ):
        self._encoder = encoder
        self._token_ceiling = token_ceiling
        self._top_k = top_k
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
    ) -> None:
        entry = {
            "text": text,
            "embedding": embedding,
            "seq_id": seq_id,
            "content_type": content_type,
            "epe_score": epe_score,
            "metadata": metadata or {},
            "stale": False,
            "token_count": len(text.split()),
        }
        with self._lock:
            self._entries.append(entry)

        # trigger async budget enforcement above 80% ceiling
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

    def _enforce_token_budget(self) -> None:
        target = 0.70 * self._token_ceiling
        evicted = 0
        with self._lock:
            # sort non-stale by epe_score ascending (lowest = most redundant)
            active = sorted(
                [(i, e) for i, e in enumerate(self._entries) if not e["stale"]],
                key=lambda x: x[1]["epe_score"],
            )
            for idx, entry in active:
                current = sum(
                    e["token_count"] for e in self._entries if not e["stale"]
                )
                if current <= target:
                    break
                self._entries[idx]["stale"] = True
                evicted += 1
