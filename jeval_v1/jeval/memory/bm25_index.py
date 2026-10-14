from __future__ import annotations

import math
import re
from collections import defaultdict

_TOKEN_RE = re.compile(r'[\s\.,;:!?()\[\]{}\'"<>/\\|@#$%^&*+=`~-]+')
_MIN_TOKEN_LEN = 2

# BM25 hyperparameters
K1 = 1.5
B = 0.75


def _tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.split(text.lower())
    return [t for t in tokens if len(t) >= _MIN_TOKEN_LEN]


class BM25Index:
    """
    Lightweight in-memory BM25 index over text segments.

    Uses only the Python standard library.  Thread-safety is NOT guaranteed —
    callers that share an index across threads must coordinate externally.
    """

    def __init__(self) -> None:
        self.clear()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, seg_id: str, text: str) -> None:
        """Index *text* under *seg_id*. Re-adding an existing seg_id is a no-op."""
        if seg_id in self._tf:
            return

        tokens = _tokenize(text)
        tf: dict[str, int] = defaultdict(int)
        for t in tokens:
            tf[t] += 1

        self._tf[seg_id] = dict(tf)
        self._doc_len[seg_id] = len(tokens)
        self._total_tokens += len(tokens)
        self._n_docs += 1

        for term in tf:
            self._df[term] = self._df.get(term, 0) + 1

    def remove(self, seg_id: str) -> None:
        """Remove *seg_id* from the index."""
        if seg_id not in self._tf:
            return

        tf = self._tf.pop(seg_id)
        self._total_tokens -= self._doc_len.pop(seg_id)
        self._n_docs -= 1

        for term in tf:
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]

    def clear(self) -> None:
        """Reset all index state."""
        self._tf: dict[str, dict[str, int]] = {}
        self._doc_len: dict[str, int] = {}
        self._df: dict[str, int] = {}
        self._n_docs: int = 0
        self._total_tokens: int = 0

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Return [(seg_id, bm25_score)] sorted by score descending, top_k only.
        Segments with score 0 are omitted.
        """
        if self._n_docs == 0:
            return []

        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        avgdl = self._total_tokens / self._n_docs

        scores: dict[str, float] = {}
        for term in q_tokens:
            if term not in self._df:
                continue

            df_t = self._df[term]
            idf = math.log((self._n_docs - df_t + 0.5) / (df_t + 0.5) + 1.0)

            for seg_id, tf_map in self._tf.items():
                tf_t = tf_map.get(term, 0)
                if tf_t == 0:
                    continue
                dl = self._doc_len[seg_id]
                numerator = tf_t * (K1 + 1.0)
                denominator = tf_t + K1 * (1.0 - B + B * dl / avgdl)
                scores[seg_id] = scores.get(seg_id, 0.0) + idf * (numerator / denominator)

        ranked = sorted(
            ((sid, s) for sid, s in scores.items() if s > 0.0),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]
