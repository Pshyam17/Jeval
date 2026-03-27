from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class EPEResult:
    epe: float
    baseline: float
    risk: Optional[float] = None


class EPEComputer:
    """Compute Embedding Prediction Error (EPE) for one or many segments."""

    @staticmethod
    def epe_score(emb_orig: np.ndarray, emb_pred: np.ndarray) -> float:
        """Sum of squared distances / 4, given normalized embeddings."""
        if emb_orig.shape != emb_pred.shape:
            raise ValueError("Original and predicted embeddings must match shape")
        if emb_orig.ndim != 1:
            raise ValueError("epe_score expects 1D embedding vector per sample")

        diff = emb_pred - emb_orig
        squared = np.square(diff)
        # 4 is the maximal squared euclidean distance between two unit vectors.
        return float(np.sum(squared) / 4.0)

    @staticmethod
    def compute_batch(orig: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """Compute EPE for batch [batch, dim]."""
        if orig.shape != pred.shape:
            raise ValueError("orig and pred must have identical shape")
        if orig.ndim != 2:
            raise ValueError("compute_batch expects 2D arrays")

        diffs = pred - orig
        sq = np.square(diffs)
        # Use axis=1 to sum across dims for each sample.
        return np.sum(sq, axis=1) / 4.0

    @staticmethod
    def training_loss(orig: np.ndarray, pred: np.ndarray) -> float:
        """Return sum-reduction MSE loss for training predictor. No mean normalization."""
        if orig.shape != pred.shape:
            raise ValueError("orig and pred must have identical shape")

        diffs = pred - orig
        return float(np.sum(np.square(diffs)))

    @staticmethod
    def z_scores(values: List[float]) -> List[float]:
        """Convert raw EPE values to session-normalized z-scores."""
        if not values:
            return []

        arr = np.asarray(values, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=0))
        if std == 0:
            # Avoid division by zero in uniform EPE session.
            return [0.0 for _ in values]

        return [(float(v) - mean) / std for v in values]
