from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from jeval.encoders.sentence_encoder import FrozenEncoder

if TYPE_CHECKING:
    # SchemaGapVerifier is in jeval.memory which imports JevalMemory which
    # imports CombinedEPE — guard the import to break the circular dependency.
    # At runtime the verifier is passed as a constructor argument so no import
    # is needed beyond type checking.
    from jeval.memory.schema_gap import SchemaGapVerifier


class CombinedEPE:
    """
    Combines cosine EPE and schema gap into a single fidelity signal.

    Cosine EPE captures distributional drift; schema gap catches causal-detail
    elision that the encoder treats as semantically similar. Equal weighting
    (alpha=0.5) is the maximum-entropy prior before empirical calibration.
    """

    def __init__(
        self,
        encoder: FrozenEncoder,
        schema_verifier: SchemaGapVerifier,
        alpha: float = 0.5,
    ) -> None:
        self._encoder = encoder
        self._verifier = schema_verifier
        # alpha weights cosine EPE; (1-alpha) weights schema gap
        self._alpha = alpha

    def compute(
        self,
        original: str,
        compressed: str,
        content_type: str,
    ) -> tuple[float, float, float]:
        """
        Returns (cosine_epe, schema_gap, epe_final).

        epe_final = alpha * cosine_epe + (1 - alpha) * schema_gap

        All three values are returned so callers can log individual components
        for ablation analysis (e.g., set alpha=0 or alpha=1 in benchmarks).
        """
        emb_orig = self._encoder.encode([original])[0]
        emb_comp = self._encoder.encode([compressed])[0]
        cosine_epe = float(1.0 - np.dot(emb_orig, emb_comp))
        # clamp to [0, 1] — floating-point can produce tiny negatives near 0
        cosine_epe = max(0.0, min(1.0, cosine_epe))

        # content_type from the NLI classifier (FACTUAL/CAUSAL/…) rarely matches
        # schema keys (migration_failure/deployment/…).  When it doesn't, auto-detect
        # the best-matching schema from the original text so that the schema gap
        # contribution is non-zero for structured segments.
        schema_type = (
            content_type
            if content_type in self._verifier._compiled
            else self._verifier.detect_schema_type(original)
        )
        schema_gap = self._verifier.compute_gap_pair(original, compressed, schema_type)

        epe_final = self._alpha * cosine_epe + (1.0 - self._alpha) * schema_gap
        return cosine_epe, schema_gap, epe_final

    def compute_novelty(
        self,
        new_text: str,
        cached_embeddings: np.ndarray,
    ) -> float:
        """
        Novelty EPE: min cosine distance from new_text to any cached embedding.

        Used by the novelty gate at write time — no schema gap here because
        novelty measures distributional coverage of the cache, not fact fidelity.
        Returns 1.0 (fully novel) when the cache is empty.
        """
        if cached_embeddings.ndim == 0 or cached_embeddings.shape[0] == 0:
            return 1.0
        emb = self._encoder.encode([new_text])[0]
        # cached_embeddings rows are L2-normalised (FrozenEncoder guarantee)
        sims = cached_embeddings @ emb
        return float(1.0 - sims.max())
