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

    Build-Time Fidelity Risk (§1.5):
    Combined risk:
      R_build = λ1 * cosine_epe + λ2 * schema_gap + λ3 * predictor_epe
      subject to λ1 + λ2 + λ3 = 1

    Weights (context-dependent):
    | Mode | λ1 (cosine) | λ2 (schema) | λ3 (predictor) | When |
    |------|-------------|-------------|----------------|------|
    | Full (with predictor) | 0.50 | 0.35 | 0.15 | Predictor checkpoint available |
    | Default (no predictor) | 0.60 | 0.40 | 0.00 | Production deployment |

    Gate decision:
      accept candidate iff R_build ≤ τ_commit
    Threshold: τ_commit = 0.35
    """

    def __init__(
        self,
        encoder: FrozenEncoder,
        schema_verifier: SchemaGapVerifier,
        alpha: float = 0.5,
        lambda1: float = 0.60,  # cosine
        lambda2: float = 0.40,  # schema
        lambda3: float = 0.00,  # predictor (0.15 if available)
        predictor=None,  # Optional PreLNTransformerPredictor
    ) -> None:
        self._encoder = encoder
        self._verifier = schema_verifier
        self._alpha = alpha
        self._lambda1 = lambda1
        self._lambda2 = lambda2
        self._lambda3 = lambda3
        self._predictor = predictor

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

    def compute_r_build(
        self,
        original: str,
        compressed: str,
        content_type: str,
    ) -> float:
        """
        Compute R_build fidelity risk score (§1.5).

        Combined risk:
          R_build = λ1 * cosine_epe + λ2 * schema_gap + λ3 * predictor_epe

        Returns value in [0, 1] range.
        Lower values indicate higher fidelity (less lossy compression).
        Gate decision: accept candidate iff R_build ≤ τ_commit (0.35)
        """
        # Cosine EPE
        emb_orig = self._encoder.encode([original])[0]
        emb_comp = self._encoder.encode([compressed])[0]
        cosine_epe = float(1.0 - np.dot(emb_orig, emb_comp))
        cosine_epe = np.clip(cosine_epe, 0.0, 1.0)

        # Schema gap
        schema_type = (
            content_type
            if content_type in self._verifier._compiled
            else self._verifier.detect_schema_type(original)
        )
        schema_gap = self._verifier.compute_gap_pair(original, compressed, schema_type)

        # Predictor EPE (if available)
        if self._predictor is not None:
            import torch
            with torch.no_grad():
                pred_emb = self._predictor(emb_comp)
            predictor_epe = float(np.linalg.norm(pred_emb - emb_orig) ** 2) / 4.0
            predictor_epe = np.clip(predictor_epe, 0.0, 1.0)
        else:
            predictor_epe = 0.0

        # Combined risk
        r_build = (
            self._lambda1 * cosine_epe +
            self._lambda2 * schema_gap +
            self._lambda3 * predictor_epe
        )
        return float(np.clip(r_build, 0.0, 1.0))

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
