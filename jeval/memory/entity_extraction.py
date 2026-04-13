from __future__ import annotations

import re

# Patterns that extract the technical identifiers most relevant to agent-session
# routing accuracy. These supplement (or replace, when spaCy is absent) NER:
#
# ALL_CAPS symbols: JWT_SECRET, HTTP, GET, API — protocol keywords and
#   environment variables that survive compression less reliably than prose.
# File paths: src/config/env.ts — anchors that fact_index already tracks.
# Step references: step 4, step 6 — structural identifiers across the session.
# Multi-digit standalone numbers: 423, 503, 512 — counts and codes that carry
#   causal detail but are easily elided by a compressor targeting brevity.
# Error class names: ValueError, TimeoutException — typed error surfaces.
_RE_ALLCAPS = re.compile(r"\b[A-Z][A-Z_]{2,}\b")
_RE_FILE = re.compile(r"[\w./\-]+\.\w{2,4}")
_RE_STEP = re.compile(r"\bstep\s+\d+\b", re.IGNORECASE)
_RE_NUMBER = re.compile(r"\b\d{2,}\b")
_RE_ERROR = re.compile(r"\b[A-Z][a-zA-Z]+Error\b|\b[A-Z][a-zA-Z]+Exception\b")


def extract_entity_texts(text: str) -> set[str]:
    """
    Return a lower-cased set of technical entity strings from *text*.

    Primary path: spaCy NER (PERSON, ORG, GPE, PRODUCT, EVENT, CARDINAL,
    ORDINAL, QUANTITY) plus the regex patterns from fact_index (file paths,
    error codes, step references).  Fallback when spaCy is absent: the regex
    patterns plus ALL_CAPS symbols, multi-digit numbers, and error class names.

    Design intent: generic nouns like "deployment" or "migration" are
    intentionally NOT extracted by the regex path.  A query containing only
    generic nouns yields an empty entity set, which routes to cold_storage
    — the safe default when no specific technical entities are present.
    """
    entities: set[str] = set()

    try:
        from jeval.memory.fact_index import extract_entities as _spacy_extract
        for e in _spacy_extract(text):
            entities.add(e["text"].lower())
        return entities
    except ImportError:
        pass

    # spaCy unavailable — regex-only fallback
    for pat in (_RE_ALLCAPS, _RE_FILE, _RE_STEP, _RE_NUMBER, _RE_ERROR):
        for m in pat.finditer(text):
            entities.add(m.group().lower())
    return entities
