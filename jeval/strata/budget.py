from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from jeval.epe.core import EPEComputer

# Content types that carry semantic risk — penalize compression more heavily
PROTECT_TYPES = {"ENTITY", "FACTUAL", "CAUSAL"}

# Per-type base budget floor — how much to keep even at low EPE
_TYPE_FLOOR = {
    "FACTUAL":     0.75,
    "CAUSAL":      0.70,
    "ENTITY":      0.70,
    "TEMPORAL":    0.55,
    "CONTRASTIVE": 0.55,
    "BACKGROUND":  0.20,
}

# Per-type EPE sensitivity — how much z-score shifts the budget
_TYPE_SENSITIVITY = {
    "FACTUAL":     0.15,   # small shift — facts are facts regardless of EPE
    "CAUSAL":      0.20,   # medium shift — reasoning chains matter
    "ENTITY":      0.15,
    "TEMPORAL":    0.20,
    "CONTRASTIVE": 0.20,
    "BACKGROUND":  0.30,   # large shift — background can be aggressively compressed
}


@dataclass(frozen=True)
class SegmentPlan:
    raw_score: float
    z_score: float
    content_type: str
    artifact_override: bool
    budget: float


class BudgetAllocator:
    """
    Continuous EPE x content-type budget allocation.

    Budget formula per segment:
        base   = TYPE_FLOOR[content_type]
        delta  = TYPE_SENSITIVITY[content_type] * z_score
        budget = clip(base + delta, type_min, 1.0)

    This means:
    - High EPE (z >> 0)  → budget pushed toward 1.0 (preserve, compression is risky)
    - Low EPE  (z << 0)  → budget pushed toward floor (compress, meaning survives)
    - Artifact override  → always 1.0, EPE irrelevant
    - Unknown type       → falls back to BACKGROUND floor with BACKGROUND sensitivity
    """

    def __init__(self, min_budget: float = 0.10, max_budget: float = 1.0):
        self.min_budget = min_budget
        self.max_budget = max_budget

    def allocate(
        self,
        segment_text: str,
        epe: float,
        z_score: float,
        content_type: str,
        artifact_override: bool,
        confidence: Optional[float] = None,
    ) -> SegmentPlan:
        if artifact_override:
            return SegmentPlan(epe, z_score, content_type, True, 1.0)

        # Normalize content_type — classifier returns short keys like FACTUAL
        ct = content_type.upper().strip()

        floor = _TYPE_FLOOR.get(ct, _TYPE_FLOOR["BACKGROUND"])
        sensitivity = _TYPE_SENSITIVITY.get(ct, _TYPE_SENSITIVITY["BACKGROUND"])

        # Confidence-weighted floor: low confidence → trust type less, use BACKGROUND floor
        if confidence is not None and confidence < 0.5:
            bg_floor = _TYPE_FLOOR["BACKGROUND"]
            floor = floor * confidence + bg_floor * (1 - confidence)

        raw_budget = floor + sensitivity * z_score
        budget = float(max(self.min_budget, min(self.max_budget, raw_budget)))

        return SegmentPlan(epe, z_score, ct, False, budget)

    def compute_session_plans(
        self,
        epe_values: Mapping[int, float],
        content_types: Mapping[int, str],
        artifact_overrides: Mapping[int, bool],
        confidences: Mapping[int, float],
    ) -> Mapping[int, SegmentPlan]:
        if not epe_values:
            return {}

        z_scores = EPEComputer.z_scores(list(epe_values.values()))
        plans = {}

        for i, (seg_id, epe) in enumerate(epe_values.items()):
            ct = content_types.get(seg_id, "BACKGROUND")
            artifact_override = artifact_overrides.get(seg_id, False)
            confidence = confidences.get(seg_id, 0.0)
            plans[seg_id] = self.allocate(
                segment_text="",
                epe=epe,
                z_score=z_scores[i],
                content_type=ct,
                artifact_override=artifact_override,
                confidence=confidence,
            )
        return plans