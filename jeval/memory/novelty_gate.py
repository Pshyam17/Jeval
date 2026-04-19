from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from jeval.encoders.sentence_encoder import FrozenEncoder
from jeval.memory.schema_gap import SchemaGapVerifier


class NoveltyGate:
    """
    Novel if combined EPE (cosine distance + entity diversity) > threshold.
    Cosine distance alone fails for structurally similar but semantically
    distinct segments (e.g. consecutive agent steps with different actions).
    Entity diversity catches these by checking for new named entities/numbers.

    v3.0: Also integrates schema novelty detection (§1.10) for artifact types.
    """

    def __init__(
        self,
        encoder: FrozenEncoder,
        threshold: float = 0.05,
        working_set_size: int = 50,
        schema_novelty_threshold: float = 0.50,  # τ_novel
    ):
        self._encoder = encoder
        self._threshold = threshold
        self._working_set: deque[np.ndarray] = deque(maxlen=working_set_size)
        self._seen_entities: set[str] = set()
        self._schema_verifier = SchemaGapVerifier()
        self._schema_novelty_threshold = schema_novelty_threshold
        self._unknown_artifacts: list[str] = []  # Queue for offline schema induction

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

    def is_schema_novel(self, text: str) -> tuple[bool, float]:
        """
        Check if text is novel according to schema novelty detection (§1.10).

        Novelty(x | S) = 1 − max_{s ∈ S} Fit(x, s)

        Returns:
            (is_novel, novelty_score):
            - is_novel=True if novelty_score > τ_novel (0.50)
            - novelty_score=1.0 means fully novel (no schema matches)
            - novelty_score=0.0 means perfect schema match
        """
        novelty_score = self._schema_verifier.compute_novelty(text)
        is_novel = novelty_score > self._schema_novelty_threshold
        return is_novel, novelty_score

    def enqueue_for_induction(self, artifact: str) -> None:
        """
        Enqueue unknown artifact for offline schema induction (§1.11).

        Called when Novelty(x | S) > τ_novel - artifact doesn't match
        any known schema and should be considered for new schema discovery.
        """
        self._unknown_artifacts.append(artifact)

    def get_unknown_artifacts(self) -> list[str]:
        """
        Get queue of unknown artifacts for offline schema induction.

        Returns list of artifacts that didn't match any known schema.
        These can be passed to SchemaInduction for batch processing.
        """
        artifacts = self._unknown_artifacts.copy()
        self._unknown_artifacts.clear()
        return artifacts

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
