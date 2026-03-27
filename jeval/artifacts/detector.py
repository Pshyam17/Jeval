from __future__ import annotations

import re
from typing import List

ARTIFACT_PATTERNS = [
    re.compile(r"src/[^\s]*", re.IGNORECASE),
    re.compile(r"\b(?:JWT|Redis|Postgres|MongoDB|MySQL|sqlite)\b"),
    re.compile(r"\b(?:/api/[^\s]+)\b"),
    re.compile(r"\b(?:4\d{2}|5\d{2}|401|403|404|500)\b"),
    re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b"),
]


def is_artifact(text: str) -> bool:
    """Detect whether the text contains an artifact-like token requiring max budget."""
    if not text:
        return False

    for pattern in ARTIFACT_PATTERNS[:4]:
        if pattern.search(text):
            return True

    return False


def extract_artifacts(text: str) -> List[str]:
    """Extract artifact candidates from text for ArtifactIndex entries."""
    matches = []
    for pattern in ARTIFACT_PATTERNS[:4]:
        matches.extend(pattern.findall(text))
    # Deduplicate while preserving order.
    seen = set()
    artifacts = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            artifacts.append(m)
    return artifacts
