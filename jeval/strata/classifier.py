from __future__ import annotations

from typing import List, Mapping

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover
    CrossEncoder = None  # type: ignore[assignment]

FAST_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
PROD_MODEL = "cross-encoder/nli-deberta-v3-large"


class ContentClassifier:
    """Zero-shot NLI content classifier for session segmentation taxonomies."""

    def __init__(self, model_name: str = FAST_MODEL):
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

        # Label heuristics are explicit strings to maximize semantic accuracy.
        self._labels = [
            "is a specific technical fact, file path, error code, or API endpoint",
            "describes causation or reasoning (because, leads to, therefore)",
            "mentions a named entity such as person name, service name, or function",
            "describes temporal order or scheduling",
            "expresses contrast, negation, rejection or comparison",
            "is background context or ambient status update",
        ]

    def classify(self, text: str) -> Mapping[str, float]:
        if not text:
            raise ValueError("ContentClassifier.classify requires non-empty text")

        scores = self._model.predict([text], return_softmax=True)[0]
        # Map labels to confidences; preserve sorted order with comprehension.
        return {label: float(score) for label, score in zip(self._labels, scores)}

    def top_label(self, text: str) -> str:
        score_map = self.classify(text)
        # stable tie-breaking via list order
        return max(score_map, key=score_map.get)
