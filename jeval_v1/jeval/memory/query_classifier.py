from __future__ import annotations

import re
from typing import Optional

PRECISION_PATTERNS = [
    re.compile(r, re.IGNORECASE)
    for r in [
        r"step\s+\d+",
        r"at step",
        r"which step",
        r"what step",
        r"error code",
        r"HTTP\s+\d{3}",
        r"\d{3}\s+[A-Z]+",
        r"line\s+\d+",
        r"port\s+\d+",
    ]
]

ENTITY_LOOKUP_PATTERNS = [
    re.compile(r"[\w./\-]+\.\w{2,4}"),
    re.compile(r"[A-Z][A-Z_]{2,}"),
    re.compile(r"[A-Z][a-z]+Error"),
]

_STEP_NUM_RE = re.compile(r"\bstep\s+(\d+)\b", re.IGNORECASE)
_ENTITY_HINT_PATTERNS = [
    re.compile(r"[\w./\-]+\.\w{2,4}"),
    re.compile(r"[A-Z][A-Z_]{2,}"),
    re.compile(r"[A-Z][a-z]+Error"),
]


class QueryClassifier:
    def classify(self, query: str) -> str:
        is_precision = any(p.search(query) for p in PRECISION_PATTERNS)
        is_entity = any(p.search(query) for p in ENTITY_LOOKUP_PATTERNS)

        if is_precision and is_entity:
            return "ambiguous"
        if is_precision:
            return "precision"
        if is_entity:
            return "entity"
        return "context"

    def extract_seq_id(self, query: str) -> Optional[int]:
        m = _STEP_NUM_RE.search(query)
        return int(m.group(1)) if m else None

    def extract_entity_hint(self, query: str) -> Optional[str]:
        # return the most specific match (longest)
        best: Optional[str] = None
        for p in _ENTITY_HINT_PATTERNS:
            m = p.search(query)
            if m:
                candidate = m.group()
                if best is None or len(candidate) > len(best):
                    best = candidate
        return best

    def merge_results(
        self,
        precision_results: list[dict],
        context_results: list[dict],
        max_tokens: int = 1500,
    ) -> str:
        seen_seq_ids: set = set()
        prec_parts: list[str] = []
        ctx_parts: list[str] = []

        for r in precision_results:
            sid = r.get("seq_id")
            if sid in seen_seq_ids:
                continue
            seen_seq_ids.add(sid)
            content = r.get("content", r.get("text", ""))
            label = f"[step {sid}]" if sid is not None else "[fact]"
            prec_parts.append(f"{label} {content}")

        for r in context_results:
            sid = r.get("seq_id")
            if sid in seen_seq_ids:
                continue
            seen_seq_ids.add(sid)
            content = r.get("content", r.get("text", ""))
            ctx_parts.append(content)

        sections: list[str] = []
        if prec_parts:
            sections.append("\n".join(prec_parts))
        if ctx_parts:
            sections.append("\n".join(ctx_parts))

        result = "\n\n".join(sections)
        words = result.split()
        if len(words) > max_tokens:
            result = " ".join(words[:max_tokens])
        return result
