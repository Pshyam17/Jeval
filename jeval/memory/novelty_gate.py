from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from jeval.encoders.sentence_encoder import FrozenEncoder


class NoveltyGate:
    """
    Novel if combined EPE (cosine distance + entity diversity) > threshold.
    Cosine distance alone fails for structurally similar but semantically
    distinct segments (e.g. consecutive agent steps with different actions).
    Entity diversity catches these by checking for new named entities/numbers.
    """

    def __init__(
        self,
        encoder: FrozenEncoder,
        threshold: float = 0.05,
        working_set_size: int = 50,
    ):
        self._encoder = encoder
        self._threshold = threshold
        self._working_set: deque[np.ndarray] = deque(maxlen=working_set_size)
        self._seen_entities: set[str] = set()

    def _extract_key_tokens(self, text: str) -> set[str]:
        """Extract numbers, capitalized words, and short tokens as key entities."""
        import re
        tokens = re.findall(r'\b\d+\b|\b[A-Z][a-z]+\b|\b[A-Z]{2,}\b', text)
        return set(tokens)

    def is_novel(self, text: str) -> tuple[bool, float]:
        # Check entity novelty — new numbers or named entities mean novel content
        key_tokens = self._extract_key_tokens(text)
        new_entities = key_tokens - self._seen_entities
        if new_entities:
            return True, 1.0

        if not self._working_set:
            return True, 1.0

        emb = self._encoder.encode([text])[0]
        sims = np.array([float(np.dot(emb, w)) for w in self._working_set])
        epe = float(1.0 - sims.max())
        return epe > self._threshold, epe

    def update_working_set(
        self,
        text: str,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        emb = embedding if embedding is not None else self._encoder.encode([text])[0]
        self._working_set.append(emb)
        self._seen_entities.update(self._extract_key_tokens(text))

    def clear(self) -> None:
        self._working_set.clear()
        self._seen_entities.clear()
