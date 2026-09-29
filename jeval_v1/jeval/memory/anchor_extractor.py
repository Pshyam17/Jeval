from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np

_STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "and", "but", "or", "not",
    "this", "that", "these", "those", "it", "its", "which", "who", "what",
    "when", "where", "how", "why", "all", "each", "every", "any", "some",
    "no", "nor", "so", "yet", "both", "either", "neither", "one", "two",
    "after", "before", "during", "then", "than", "if", "because", "since",
    "while", "though", "although", "also", "just", "up", "down", "out",
    "about", "into", "through", "over", "under", "their", "there", "they",
    "we", "you", "i", "he", "she", "his", "her", "our", "my", "your",
    "step", "action", "observation",
}

_ALWAYS_ANCHOR_TYPES = {
    "CARDINAL", "ORDINAL", "QUANTITY",
    "step_reference", "error_code", "file_path",
}

# Tier 1 rule patterns applied to raw (original-case) token text
_RE_NUMBER    = re.compile(r"^\d+$")
_RE_FILE_PATH = re.compile(r"^[\w./\-]+\.\w{2,4}$")
_RE_ALL_CAPS  = re.compile(r"^[A-Z][A-Z_]{2,}$")
_RE_ERROR_CLS = re.compile(r"^[A-Z][a-z]+Error$")
_RE_STEP_REF  = re.compile(r"^step\s*\d+$", re.IGNORECASE)

# Tier 2 requires at least this many segments before the IDF distribution
# is stable enough to use. Below this threshold, only Tier 1 fires.
MIN_CORPUS_SEGMENTS_FOR_TIER2 = 20


def _is_tier1(tok: str) -> bool:
    return bool(
        _RE_NUMBER.match(tok)
        or _RE_FILE_PATH.match(tok)
        or _RE_ALL_CAPS.match(tok)
        or _RE_ERROR_CLS.match(tok)
        or _RE_STEP_REF.match(tok)
    )


class AnchorExtractor:
    def __init__(self, max_anchors: int = 10):
        self._max_anchors = max_anchors
        self._total_segments = 0
        self._seg_freq: dict[str, int] = defaultdict(int)

    def update_corpus(self, text: str) -> None:
        self._total_segments += 1
        for tok in set(self._tokenize(text)):
            if not self._is_stopword_or_short(tok):
                self._seg_freq[tok] += 1

    def extract(self, text: str, entities: list[dict]) -> list[str]:
        seen: set[str] = set()
        tier1: list[str] = []

        # Tier 1a: entity list entries whose type is always-anchor
        for e in entities:
            if e.get("type") in _ALWAYS_ANCHOR_TYPES:
                t = e["text"]
                if t not in seen:
                    seen.add(t)
                    tier1.append(t)

        # Tier 1b: tokens in the segment matching structural patterns
        for tok in self._tokenize_raw(text):
            if tok not in seen and _is_tier1(tok):
                seen.add(tok)
                tier1.append(tok)

        # Tier 2: median-IDF selection — only when corpus is large enough
        # to produce a stable bimodal distribution
        if self._total_segments < MIN_CORPUS_SEGMENTS_FOR_TIER2:
            return tier1[: self._max_anchors]

        corpus_idfs: list[float] = []
        for tok, n in self._seg_freq.items():
            if not self._is_stopword_or_short(tok) and not _is_tier1(tok):
                corpus_idfs.append(math.log(1 + self._total_segments / (1 + n)))

        if not corpus_idfs:
            return tier1[: self._max_anchors]

        median_idf = float(np.median(corpus_idfs))
        tier2: list[tuple[str, float]] = []

        for tok in set(self._tokenize(text)):
            if self._is_stopword_or_short(tok) or _is_tier1(tok) or tok in seen:
                continue
            n = self._seg_freq.get(tok, 0)
            idf = math.log(1 + self._total_segments / (1 + n))
            if idf >= median_idf:
                tier2.append((tok, idf))

        tier2.sort(key=lambda x: x[1], reverse=True)
        anchors = tier1 + [tok for tok, _ in tier2 if tok not in seen]
        return anchors[: self._max_anchors]

    def reset(self) -> None:
        self._total_segments = 0
        self._seg_freq.clear()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase tokenisation for IDF corpus tracking."""
        return re.findall(r"[a-zA-Z0-9_/.\-]+", text.lower())

    @staticmethod
    def _tokenize_raw(text: str) -> list[str]:
        """Original-case tokenisation for Tier 1 pattern matching."""
        return re.findall(r"[a-zA-Z0-9_/.\-]+", text)

    @staticmethod
    def _is_stopword_or_short(tok: str) -> bool:
        return len(tok) < 4 or tok.lower() in _STOPWORDS
