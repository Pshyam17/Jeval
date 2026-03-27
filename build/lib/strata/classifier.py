from __future__ import annotations

from typing import List, Mapping

try:
    import numpy as np
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover
    CrossEncoder = None  # type: ignore[assignment]

FAST_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
PROD_MODEL = "cross-encoder/nli-deberta-v3-large"

# Short keys used throughout budget.py and adaptive.py.
# Hypotheses are full descriptive strings — this is what drives NLI confidence.
# Keys are what get stored in SegmentPlan.content_type.
_LABEL_MAP = {
    "FACTUAL":     "is a specific technical fact, file path, error code, or API endpoint",
    "CAUSAL":      "describes causation or reasoning such as because, leads to, or therefore",
    "ENTITY":      "mentions a named entity such as a person, service name, or function name",
    "TEMPORAL":    "describes temporal order, scheduling, or a sequence of events",
    "CONTRASTIVE": "expresses contrast, negation, rejection, or comparison",
    "BACKGROUND":  "is background context, ambient status, or a casual pleasantry with no technical content",
}

_KEYS   = list(_LABEL_MAP.keys())
_HYPOTHESES = list(_LABEL_MAP.values())


class ContentClassifier:
    """Zero-shot NLI content type classifier. Returns short type keys (FACTUAL, CAUSAL, ...)."""

    def __init__(self, model_name: str = FAST_MODEL):
        self.model_name = model_name
        if CrossEncoder is None:
            raise RuntimeError("sentence-transformers is required for ContentClassifier")
        self._model = CrossEncoder(model_name)

    def classify(self, text: str) -> Mapping[str, float]:
        """Return {short_key: confidence} for all content types."""
        if not text:
            raise ValueError("ContentClassifier.classify requires non-empty text")

        # predict expects list of (text, hypothesis) pairs
        pairs = [(text, hyp) for hyp in _HYPOTHESES]
        raw_scores = self._model.predict(pairs, apply_softmax=True)

        return {key: float(np.asscalar(score)) for key, score in zip(_KEYS, raw_scores)}

    def top_label(self, text: str) -> str:
        """Return the short key with highest confidence: FACTUAL | CAUSAL | ... | BACKGROUND"""
        scores = self.classify(text)
        return max(scores, key=scores.get)