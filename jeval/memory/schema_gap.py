from __future__ import annotations

import re

# Schema definitions map content types to required and optional fact patterns.
# Required facts drive schema_gap computation; optional facts are available for
# downstream inspection but do not affect the gap score.
SCHEMAS: dict[str, dict[str, dict[str, str]]] = {
    "tool_call": {
        "required": {
            "tool_name":   r"[a-zA-Z_]\w*\s*\(",
            "result":      r"returned|output|result|response",
        },
        "optional": {
            "duration":    r"\d+\s*(ms|s|seconds)",
            "status":      r"success|fail|error|ok",
        },
    },
    "error": {
        "required": {
            "error_type":  r"[A-Z][a-zA-Z]+Error|[A-Z][a-zA-Z]+Exception|HTTP\s+[45]\d\d",
            "location":    r"[\w./\-]+\.\w{2,4}|line\s+\d+|at\s+\w+\.\w+",
        },
        "optional": {
            "line_number":     r"line\s+\d+|:\d+",
            "variable_name":   r"'[a-zA-Z_]\w*'|\"[a-zA-Z_]\w*\"",
        },
    },
    "test_result": {
        "required": {
            "failure_count":   r"\d+\s*(tests?\s+)?(fail|pass|error)",
            "test_file":       r"[\w./\-]*(test|spec)[\w./\-]*\.\w+",
        },
        "optional": {
            "assertion_count": r"\d+\s*assertion",
            "duration":        r"\d+\s*(ms|s|seconds)",
        },
    },
    "migration_failure": {
        "required": {
            "timing":          r"\d+\s*(s|ms|seconds|minutes|min)",
            "table_name":      r"\w+_table|\w+s\s+table|table\s+\w+",
            "error_type":      r"timeout|deadlock|lock|constraint|foreign\s+key",
        },
        "optional": {
            "connection_count": r"\d+\s*(concurrent\s+)?connections",
            "rollback":         r"roll.?back|revert",
        },
    },
    "deployment": {
        "required": {
            "environment":  r"staging|production|prod\b|dev\b|k8s|kubernetes",
            # "deployed" removed — it is the action verb, not an outcome state;
            # "deployed to staging" describes the action and should not match outcome.
            "outcome":      r"succeeded|failed|timeout|rolled?\s*back|reverted|rejected",
        },
        "optional": {
            "health_check": r"health\s*check|GET\s+/health|HTTP\s+200",
            "error_code":   r"HTTP\s+[45]\d\d|\d{3}\s+[A-Z]+",
        },
    },
    "file_modification": {
        "required": {
            "file_path":    r"[\w./\-]+\.\w{2,4}",
            "action":       r"created|modified|deleted|updated|added|removed|renamed",
        },
        "optional": {
            "line_count":   r"\d+\s*line",
            "function":     r"def\s+\w+|function\s+\w+|\w+\(\)",
        },
    },
}

# Pre-compile all patterns once at import time — avoids repeated compilation
# overhead inside compute_gap_pair which may be called at high frequency.
_COMPILED: dict[str, dict[str, dict[str, re.Pattern]]] = {}
for _ctype, _schema in SCHEMAS.items():
    _COMPILED[_ctype] = {}
    for _tier in ("required", "optional"):
        if _tier in _schema:
            _COMPILED[_ctype][_tier] = {
                name: re.compile(pat, re.IGNORECASE)
                for name, pat in _schema[_tier].items()
            }


class SchemaGapVerifier:
    """
    Computes the fraction of schema-required facts that are absent from text.

    Motivation: cosine EPE treats "migration failed on staging" and
    "migration failed on staging due to lock timeout after 30s on roles table"
    as semantically similar (gap ~0.08). Schema gap catches the causal detail
    loss that cosine EPE misses.
    """

    def __init__(self, schemas: dict = SCHEMAS) -> None:
        self._schemas = schemas
        # recompile if caller provides custom schemas (rare, but supported for ablation)
        if schemas is SCHEMAS:
            self._compiled = _COMPILED
        else:
            self._compiled = {}
            for ctype, schema in schemas.items():
                self._compiled[ctype] = {}
                for tier in ("required", "optional"):
                    if tier in schema:
                        self._compiled[ctype][tier] = {
                            name: re.compile(pat, re.IGNORECASE)
                            for name, pat in schema[tier].items()
                        }

    def extract_facts(self, text: str, content_type: str) -> dict[str, bool]:
        """Return {fact_name: present} for every required fact in the schema."""
        compiled = self._compiled.get(content_type, {})
        required = compiled.get("required", {})
        return {name: bool(pat.search(text)) for name, pat in required.items()}

    def compute_gap(self, text: str, content_type: str) -> float:
        """
        Fraction of required schema facts absent from a single text.

        Used at write time to score the original segment before compression:
        a high gap indicates the original text itself is sparse on required
        facts, which informs budget allocation independently of compression.
        Returns 0.0 when the content_type has no schema.
        """
        facts = self.extract_facts(text, content_type)
        if not facts:
            return 0.0
        n_required = len(facts)
        n_absent = sum(1 for present in facts.values() if not present)
        return n_absent / n_required

    def compute_gap_pair(
        self, original: str, compressed: str, content_type: str
    ) -> float:
        """
        Schema gap between original and compressed:
            |F_o \\ F_c| / max(|F_o|, 1)

        F_o = required facts present in the original.
        F_c = required facts present in the compressed.
        Numerator counts facts that survived in original but were lost in compressed.
        Returns 0.0 when content_type has no schema.
        """
        compiled = self._compiled.get(content_type, {})
        required = compiled.get("required", {})
        if not required:
            return 0.0

        facts_original: set[str] = {
            name for name, pat in required.items() if pat.search(original)
        }
        facts_compressed: set[str] = {
            name for name, pat in required.items() if pat.search(compressed)
        }

        lost = facts_original - facts_compressed
        return len(lost) / max(len(facts_original), 1)

    def detect_schema_type(self, text: str) -> str:
        """
        Return the schema key with the most required facts matched in text.

        Used when the NLI content-type (FACTUAL/CAUSAL/…) must be bridged to
        a schema key (migration_failure/deployment/…) for fact-loss detection.
        The NLI classifier labels serve budget allocation; schema keys serve
        fact fidelity scoring — they are parallel, not identical, type systems.

        Returns "" when no schema has any required facts matched.
        """
        best_type = ""
        best_count = 0
        for ctype, compiled in self._compiled.items():
            required = compiled.get("required", {})
            count = sum(1 for pat in required.values() if pat.search(text))
            if count > best_count:
                best_count = count
                best_type = ctype
        return best_type
