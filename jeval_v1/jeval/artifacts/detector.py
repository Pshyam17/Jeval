from __future__ import annotations

import re
from typing import List

# Patterns that identify structural artifacts — things that must survive compression verbatim.
# Ordered by specificity: file paths → credentials → endpoints → error codes.
# Technology names (Redis, Postgres) are NOT artifacts — they are ENTITY/CAUSAL content.
ARTIFACT_PATTERNS = [
    # File paths with directory separator: src/auth.ts, tests/foo.py, lib/utils/bar.js
    re.compile(r"\b(?:src|tests|lib|dist|config|app|server|routes|middleware|hooks)/[^\s,;\"']+"),

    # Credential and secret token names — must be SCREAMING_SNAKE_CASE to avoid false positives
    re.compile(r"\b(?:[A-Z][A-Z0-9]+_(?:SECRET|KEY|TOKEN|PASSWORD|PASS|CREDENTIAL|AUTH|CERT|API))\b"),

    # API endpoint paths starting with /
    re.compile(r"(?<!\w)/api/[^\s,;\"']+"),

    # HTTP status codes — only 4xx and 5xx, only when standalone (not inside larger numbers)
    re.compile(r"(?<!\d)(?:4[0-9]{2}|5[0-9]{2})(?!\d)"),
]


def is_artifact(text: str) -> bool:
    """Return True if text contains a structural artifact requiring budget=1.0."""
    if not text:
        return False
    return any(p.search(text) for p in ARTIFACT_PATTERNS)


def extract_artifacts(text: str) -> List[str]:
    """Extract all artifact tokens from text for ArtifactIndex registration."""
    seen: set = set()
    result: List[str] = []
    for pattern in ARTIFACT_PATTERNS:
        for match in pattern.finditer(text):
            token = match.group(0)
            if token not in seen:
                seen.add(token)
                result.append(token)
    return result