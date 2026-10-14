from __future__ import annotations

from dataclasses import dataclass
from typing import List

from jeval.strata.budget import SegmentPlan


@dataclass(frozen=True)
class CompressionReport:
    segment_count: int
    average_budget: float
    high_risk_count: int
    low_risk_count: int
    plans: List[SegmentPlan]

    @classmethod
    def from_plans(cls, plans: List[SegmentPlan]) -> CompressionReport:
        if not plans:
            return cls(segment_count=0, average_budget=0.0, high_risk_count=0, low_risk_count=0, plans=[])

        budgets = [p.budget for p in plans]
        high_risk = len([p for p in plans if p.budget >= 0.95])
        low_risk = len([p for p in plans if p.budget <= 0.35])
        return cls(
            segment_count=len(plans),
            average_budget=sum(budgets) / len(budgets),
            high_risk_count=high_risk,
            low_risk_count=low_risk,
            plans=plans,
        )
