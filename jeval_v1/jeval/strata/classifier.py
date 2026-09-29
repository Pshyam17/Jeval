from __future__ import annotations

import numpy as np
from typing import Mapping

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

FAST_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
PROD_MODEL = "cross-encoder/nli-deberta-v3-large"

_LABEL_MAP = {
    "FACTUAL":     "is a specific technical fact, file path, error code, or API endpoint",
    "CAUSAL":      "describes causation or reasoning such as because, leads to, or therefore",
    "ENTITY":      "mentions a named entity such as a person, service name, or function name",
    "TEMPORAL":    "describes temporal order, scheduling, or a sequence of events",
    "CONTRASTIVE": "expresses contrast, negation, rejection, or comparison",
    "BACKGROUND":  "is background context, ambient status, or a casual pleasantry with no technical content",
}

_KEYS       = list(_LABEL_MAP.keys())
_HYPOTHESES = list(_LABEL_MAP.values())


class ContentClassifier:
    def __init__(self, model_name: str = FAST_MODEL):
        self.model_name = model_name
        if CrossEncoder is None:
            raise RuntimeError("sentence-transformers is required for ContentClassifier")
        self._model = CrossEncoder(model_name)

    def classify(self, text: str) -> Mapping[str, float]:
        if not text:
            raise ValueError("ContentClassifier.classify requires non-empty text")

        pairs = [(text, hyp) for hyp in _HYPOTHESES]
        # raw_scores shape: (n_pairs, 3) — columns are [contradiction, neutral, entailment]
        raw_scores = self._model.predict(pairs, apply_softmax=True)
        raw_scores = np.asarray(raw_scores)

        if raw_scores.ndim == 2:
            # NLI model: take entailment column (index 2)
            entailment_scores = raw_scores[:, 2]
        else:
            # Single-label model: scores already 1D
            entailment_scores = raw_scores

        return {key: float(score) for key, score in zip(_KEYS, entailment_scores)}

    def top_label(self, text: str) -> str:
        scores = self.classify(text)
        return max(scores, key=scores.get)
