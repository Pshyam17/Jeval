from __future__ import annotations

import numpy as np

from jeval.encoders.sentence_encoder import FrozenEncoder
from jeval.memory.hot_cache import HotCache

POLARITY_PAIRS = [
    ("succeeded", "failed"), ("passed", "failed"), ("passed", "broke"),
    ("deployed", "reverted"), ("resolved", "persisted"), ("fixed", "regressed"),
    ("completed", "rolled back"), ("merged", "reverted"), ("healthy", "degraded"),
    ("succeeded", "rolled back"), ("passing", "failing"), ("working", "broken"),
]


class ContradictionDetector:
    """
    Async post-write contradiction checker. Compares new text against all
    hot-cache entries. Returns seq_ids to mark stale when EPE > threshold
    AND polarity is inverted.

    Always invoked via threading.Thread(daemon=True) — never blocks write path.
    """

    def __init__(
        self,
        encoder: FrozenEncoder,
        hot_cache: HotCache,
        epe_threshold: float = 0.28,
    ):
        self._encoder = encoder
        self._hot_cache = hot_cache
        self._epe_threshold = epe_threshold

    def check(self, new_text: str, session_id: str) -> list[int]:
        entries = self._hot_cache.get_all_entries()
        if not entries:
            return []

        new_emb = self._encoder.encode([new_text])[0]
        stale_ids: list[int] = []

        for e in entries:
            # cosine distance = 1 - dot for unit vectors
            epe = float(1.0 - np.dot(new_emb, e["embedding"]))
            if epe > self._epe_threshold and self._polarity_inverted(new_text, e["text"]):
                stale_ids.append(e["seq_id"])

        return stale_ids

    def _polarity_inverted(self, text_a: str, text_b: str) -> bool:
        a_lower = text_a.lower()
        b_lower = text_b.lower()
        for w1, w2 in POLARITY_PAIRS:
            # check both directions
            if (w1 in a_lower and w2 in b_lower) or (w2 in a_lower and w1 in b_lower):
                return True
        return False
