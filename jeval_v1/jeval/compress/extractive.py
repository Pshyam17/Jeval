from __future__ import annotations

import math
import re
from collections import Counter
from typing import List

from jeval.compress.base import CompressorBackend

# Tokens that carry high information density — boost sentence score if present
_HIGH_VALUE_TOKENS = re.compile(
    r"\b(?:src/|/api/|\.ts|\.py|\.js|error|exception|failed|decided|rejected|"
    r"must|never|always|critical|jwt|secret|token|key|timeout|retry|port|endpoint)\b",
    re.IGNORECASE,
)


def _tfidf_scores(sentences: List[str]) -> List[float]:
    """Approximate tf-idf scoring — score each sentence by sum of rare token weights."""
    if not sentences:
        return []

    tokenize = lambda s: re.findall(r"[a-z0-9_/\.]+", s.lower())
    doc_tokens = [tokenize(s) for s in sentences]
    n = len(sentences)

    # Document frequency per token across all sentences
    df: Counter = Counter()
    for tokens in doc_tokens:
        df.update(set(tokens))

    scores = []
    for tokens in doc_tokens:
        tf: Counter = Counter(tokens)
        total = max(len(tokens), 1)
        score = 0.0
        for tok, count in tf.items():
            tf_val = count / total
            idf_val = math.log((n + 1) / (df[tok] + 1)) + 1.0
            score += tf_val * idf_val
        scores.append(score)

    return scores


class ExtractiveBackend(CompressorBackend):
    """
    Scored extractive compressor.

    Strategy:
    1. Split text into sentences.
    2. Score each sentence by tf-idf + high-value token bonus.
    3. Select top-K sentences by score where K = ceil(n * budget).
    4. Return selected sentences in original order (preserves narrative flow).

    Falls back to word truncation for single-sentence inputs.
    """

    def compress(self, text: str, budget: float) -> str:
        if not text or budget <= 0:
            return ""
        if budget >= 1.0:
            return text

        # Split on sentence boundaries
        sentences = [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", text) if s.strip()]

        if len(sentences) <= 1:
            # Single sentence — word-level budget truncation
            words = text.split()
            keep = max(1, math.ceil(len(words) * budget))
            return " ".join(words[:keep])

        scores = _tfidf_scores(sentences)

        # Bonus for high-value tokens
        for i, s in enumerate(sentences):
            bonus = len(_HIGH_VALUE_TOKENS.findall(s)) * 0.3
            scores[i] += bonus

        keep_n = max(1, math.ceil(len(sentences) * budget))

        # Select top-K indices, then sort back to original order
        ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
        selected = sorted(ranked[:keep_n])

        return " ".join(sentences[i] for i in selected)

    def name(self) -> str:
        return "extractive-tfidf"