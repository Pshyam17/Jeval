from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

from jeval.artifacts.index import ArtifactIndex, ArtifactEntry


@dataclass(frozen=True)
class ArtifactEvalResult:
    recall: float
    precision: float
    f1: float


class ArtifactEval:
    """Deterministic artifact recall/F1 evaluation."""

    @staticmethod
    def score(original_artifacts: ArtifactIndex, compressed_text: str) -> ArtifactEvalResult:
        ground_truth: Set[str] = {entry.path.lower() for entry in original_artifacts.to_list()}
        seen = set()

        words = [token.strip(".,:;()[]\"'") for token in compressed_text.split()]
        for token in words:
            normalized = token.lower()
            if normalized in ground_truth:
                seen.add(normalized)

        true_positive = len(seen)
        recall = true_positive / max(1, len(ground_truth))
        precision = true_positive / max(1, len(words))
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return ArtifactEvalResult(recall=recall, precision=precision, f1=f1)
