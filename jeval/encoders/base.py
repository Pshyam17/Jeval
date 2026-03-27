from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Encoder(ABC):
    """Abstract encoder interface for immutable embedding support."""

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts as a 2D float32 array [batch, dim]."""

    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the encoder output."""

    def normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize embedding vectors in-place-safe and numerically stable."""
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D array")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Zero-length embedding encountered during normalize")
        return embeddings / norms
