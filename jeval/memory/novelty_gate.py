from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from jeval.encoders.sentence_encoder import FrozenEncoder


class NoveltyGate:
    """
    EPE-based write-time gate. Compares new input against the working set
    of the most recent hot-cache embeddings.

    EPE here is cosine distance: 1 - dot(a, b) for unit vectors.
    Novel if min cosine distance to any working-set embedding > threshold.
    """

    def __init__(
        self,
        encoder: FrozenEncoder,
        threshold: float = 0.15,
        working_set_size: int = 50,
    ):
        self._encoder = encoder
        self._threshold = threshold
        self._working_set: deque[np.ndarray] = deque(maxlen=working_set_size)

    def is_novel(self, text: str) -> tuple[bool, float]:
        if not self._working_set:
            return True, 1.0
        emb = self._encoder.encode([text])[0]
        # cosine similarity to each working-set entry (all are L2-normalized)
        sims = np.array([float(np.dot(emb, w)) for w in self._working_set])
        # min cosine distance = 1 - max cosine similarity
        epe = float(1.0 - sims.max())
        return epe > self._threshold, epe

    def update_working_set(
        self,
        text: str,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        emb = embedding if embedding is not None else self._encoder.encode([text])[0]
        self._working_set.append(emb)

    def clear(self) -> None:
        self._working_set.clear()
