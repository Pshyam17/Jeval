from __future__ import annotations

from typing import List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]

from jeval.encoders.base import Encoder


class FrozenEncoder(Encoder):
    """Frozen sentence-transformer encoder for EPE target embeddings."""

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers package is required for FrozenEncoder")
        self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if embeddings.ndim != 2 or embeddings.shape[1] != self._dim:
            raise ValueError("Unexpected embedding shape from SentenceTransformer")
        # preserve contract: normalized output
        return self.normalize(embeddings.astype(np.float32, copy=False))

    def dim(self) -> int:
        return self._dim
