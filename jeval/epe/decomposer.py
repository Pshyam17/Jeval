from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from jeval.epe.core import EPEComputer
from jeval.epe.weights import RISK_WEIGHTS


@dataclass(frozen=True)
class DecomposedEPE:
    raw_epe: float
    risk_weight: float
    weighted_epe: float
    normalized_type: str


class EPEDecomposer:
    """Break down EPE by content type and risk weights."""

    def decompose(self, epe_values: Dict[str, float]) -> List[DecomposedEPE]:
        """Given raw EPE by content type, return weighted components."""
        if not epe_values:
            return []

        decomposed = []
        for content_type, raw_epe in epe_values.items():
            weight = RISK_WEIGHTS.get(content_type, 0.5)
            decomposed.append(
                DecomposedEPE(
                    raw_epe=raw_epe,
                    risk_weight=weight,
                    weighted_epe=float(raw_epe * weight),
                    normalized_type=content_type,
                )
            )
        return decomposed

    def total_weighted_epe(self, epe_values: Dict[str, float]) -> float:
        """Total weighted EPE (risk-adjusted scalar)."""
        return float(sum(item.weighted_epe for item in self.decompose(epe_values)))

    def anomaly_z_scores(self, scores: List[float]) -> List[float]:
        """Proxy for z-scores if EPE per content bucket is required."""
        return EPEComputer.z_scores(scores)
