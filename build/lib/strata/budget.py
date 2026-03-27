from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from jeval.epe.core import EPEComputer

PROTECT_TYPES = {"ENTITY", "FACTUAL", "CAUSAL"}


@dataclass(frozen=True)
class SegmentPlan:
    raw_score: float
    z_score: float
    content_type: str
    artifact_override: bool
    budget: float


class BudgetAllocator:
    """Allocate compression budgets per segment based on EPE and taxonomy."""

    def __init__(self, low_threshold: float = -0.5, high_threshold: float = 0.5):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def allocate(
        self,
        segment_text: str,
        epe: float,
        z_score: float,
        content_type: str,
        artifact_override: bool,
        confidence: Optional[float] = None,
    ) -> SegmentPlan:
        if confidence is None:
            confidence = 0.0

        if artifact_override:
            return SegmentPlan(epe, z_score, content_type, True, 1.0)

        if z_score > self.high_threshold:
            return SegmentPlan(epe, z_score, content_type, False, 1.0)

        if z_score < self.low_threshold and content_type == "BACKGROUND":
            return SegmentPlan(epe, z_score, content_type, False, 0.3)

        if confidence >= 0.35 and content_type in PROTECT_TYPES:
            return SegmentPlan(epe, z_score, content_type, False, 1.0)

        # Default safety bias: moderate compression.
        return SegmentPlan(epe, z_score, content_type, False, 0.7)

    def compute_session_plans(
        self,
        epe_values: Mapping[int, float],
        content_types: Mapping[int, str],
        artifact_overrides: Mapping[int, bool],
        confidences: Mapping[int, float],
    ) -> Mapping[int, SegmentPlan]:
        """Compute budget plans for a session with per-segment metadata."""
        if not epe_values:
            return {}

        z_scores = EPEComputer.z_scores(list(epe_values.values()))
        plans = {}

        for i, (seg_id, epe) in enumerate(epe_values.items()):
            content_type = content_types.get(seg_id, "BACKGROUND")
            artifact_override = artifact_overrides.get(seg_id, False)
            confidence = confidences.get(seg_id, 0.0)
            plans[seg_id] = self.allocate(
                segment_text="",
                epe=epe,
                z_score=z_scores[i],
                content_type=content_type,
                artifact_override=artifact_override,
                confidence=confidence,
            )
        return plans
