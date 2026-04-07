#!/usr/bin/env python3
"""
train/generate_pairs.py

Generates pairs_v2.jsonl for predictor v2 training.

Output schema per line:
  {"compressed": "...", "original": "...", "label": "faithful|hard_negative", "transform": "..."}

Counts:
  --n-faithful        synthetic faithful compressions (60-80% length, semantic-preserving)
  --n-hard-negative   synthetic hard negatives (specificity-dropping transformations)
  --swebench-pairs    hard negatives derived from SWE-bench_Verified problem statements

Step-abstraction transform is guaranteed ≥ 20% of hard negatives because the predictor
must learn that stripping step indices is semantically lossy — this is what fixes AMA-Bench
without baking regex hacks into the artifact detector.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Callable, List, Tuple

# ── Base event pool ───────────────────────────────────────────────────────────
# Cross-product of parameters yields the full original pool.

_FILES = [
    "src/auth.ts", "src/middleware/auth.ts", "src/config/env.ts",
    "src/api/users.ts", "src/db/migrations/0042_users.sql",
    "src/services/payments.py", "tests/auth.test.ts", "tests/api/users.test.py",
    "src/models/user.py", "src/utils/crypto.ts", "config/database.yml",
    "src/routes/api.ts", "src/hooks/useAuth.ts", "lib/session.py",
    "src/controllers/auth_controller.rb", "app/models/session.rb",
]

_ENVVARS = [
    "JWT_SECRET", "DATABASE_URL", "STRIPE_API_KEY", "AWS_ACCESS_KEY_ID",
    "REDIS_URL", "POSTGRES_PASSWORD", "SESSION_SECRET", "API_KEY",
    "OAUTH_CLIENT_SECRET", "ENCRYPTION_KEY", "SMTP_PASSWORD", "PAYMENTS_STRIPE_SECRET_KEY",
]

_ENDPOINTS = [
    "POST /api/auth/login", "GET /api/users", "POST /api/payments/charge",
    "PUT /api/users/{id}", "DELETE /api/sessions/{id}", "GET /api/health",
    "POST /api/webhooks/stripe", "GET /api/reports/summary",
    "PATCH /api/users/{id}/roles", "POST /api/auth/refresh",
]

_HTTP_CODES = ["400", "401", "403", "404", "409", "422", "429", "500", "502", "503"]

_TABLES = [
    "users", "sessions", "payments", "orders", "audit_logs",
    "api_keys", "roles", "permissions", "webhooks", "refresh_tokens",
]

_SERVICES = ["Redis", "Postgres", "MySQL", "MongoDB", "Elasticsearch", "RabbitMQ"]

_COUNTS = ["847", "1204", "423", "99", "512", "2048", "73", "301", "1500"]

_DURATIONS = ["4m32s", "12s", "2h15m", "45s", "8m", "30s", "1h02m", "7m48s"]

_STEP_NUMS = list(range(1, 120))

_TEST_MODULES = [
    "auth.test.ts", "payments.test.py", "users.test.rb",
    "api_integration.test.ts", "session.test.py", "rate_limit.test.ts",
]

_ERRORS = [
    "TypeError: Cannot read property 'id' of undefined",
    "ValueError: invalid literal for int() with base 10",
    "KeyError: 'access_token'",
    "IntegrityError: duplicate key value violates unique constraint",
    "ConnectionRefusedError: [Errno 111] Connection refused",
    "TimeoutError: database connection timeout after 30s",
    "AssertionError: expected 200 but got 401",
    "AttributeError: 'NoneType' object has no attribute 'user_id'",
    "PermissionError: write access denied on /var/log/app",
    "ImportError: cannot import name 'JWTBearer' from 'fastapi_jwt_auth'",
]


def _originals() -> List[str]:
    """Build a pool of ~600 distinct original software engineering events."""
    pool: List[str] = []

    # File modification events
    actions = [
        ("modified", "to add missing Authorization header validation"),
        ("modified", "to fix JWT_SECRET env var mismatch between dev and production"),
        ("created", "implementing rate-limit middleware with sliding window counter"),
        ("deleted", "removing deprecated session token storage logic"),
        ("refactored", "to extract auth logic into reusable middleware chain"),
        ("modified", "to add input sanitization before database insert"),
        ("created", "with POST and GET handlers for user management"),
        ("modified", "to rotate expired API keys on each request automatically"),
    ]
    for file in _FILES:
        for verb, reason in random.sample(actions, k=min(4, len(actions))):
            pool.append(f"{verb} {file} {reason}")

    # Environment variable events
    for var in _ENVVARS:
        pool.append(f"{var} mismatch between staging and production environments corrected")
        pool.append(f"rotated {var} in Vault — old key invalidated across all services")
        pool.append(f"added {var} to .env.example with documentation comment")

    # Test result events
    for mod in _TEST_MODULES:
        for n in random.sample([4, 7, 14, 23, 47, 112], k=3):
            pool.append(
                f"created {mod} — {n} unit tests covering auth middleware "
                f"and cookie handling, all passing"
            )
            pool.append(
                f"test suite {mod} failed: {n} assertions failed, "
                f"rate limiting logic broken after last merge"
            )

    # Database migration events
    for table in _TABLES:
        for count in random.sample(_COUNTS, k=2):
            pool.append(
                f"migration failed on staging — lock timeout after 30s on {table} table, "
                f"{count} concurrent connections"
            )
        pool.append(
            f"pt-osc migration on {table} succeeded in {random.choice(_DURATIONS)} "
            f"with zero downtime"
        )
        pool.append(
            f"added composite index on {table}(user_id, created_at) — "
            f"query time dropped from 840ms to 12ms"
        )

    # API/HTTP events
    for ep in _ENDPOINTS:
        for code in random.sample(_HTTP_CODES, k=2):
            pool.append(
                f"error {code} on {ep} — service not registered in Consul service discovery"
            )
        pool.append(f"created {ep} endpoint with request validation and error handling")

    # Service infrastructure events
    for svc in _SERVICES:
        pool.append(
            f"decided to use {svc} over Postgres for session storage due to "
            f"connection pool exhaustion under load"
        )
        pool.append(
            f"rejected {svc} for inter-service comms — team unfamiliar, "
            f"would require schema versioning overhead"
        )
        pool.append(f"{svc} cluster health check failing — replication lag {random.choice(_COUNTS)}ms")

    # Step-indexed agent trajectory events
    for step in random.sample(_STEP_NUMS, k=40):
        file = random.choice(_FILES)
        pool.append(
            f"at step {step} agent modified {file} to fix authentication bug "
            f"identified in previous observation"
        )
        pool.append(
            f"step {step} observation: test suite returned {random.choice(_COUNTS)} "
            f"failures after applying patch to {file}"
        )
        pool.append(
            f"step {step} action: rolled back migration on {random.choice(_TABLES)} table "
            f"due to lock timeout affecting {random.choice(_COUNTS)} concurrent connections"
        )

    # Error events
    for err in _ERRORS:
        pool.append(
            f"error in production: {err} — traced to missing null check after "
            f"OAuth token expiry"
        )
        pool.append(f"caught {err} at line {random.randint(10, 400)} in {random.choice(_FILES)}")

    random.shuffle(pool)
    return pool


# ── Faithful compression ──────────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "was", "were", "are", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "must", "that", "which", "this",
    "these", "those", "with", "from", "into", "onto", "upon", "over",
    "under", "about", "after", "before", "between", "through", "during",
    "successfully", "properly", "correctly", "effectively", "currently",
    "specifically", "automatically", "respectively", "additionally",
})

_VERBOSE_PHRASES = [
    (r"\bdue to the fact that\b", "because"),
    (r"\bin order to\b", "to"),
    (r"\bat this point in time\b", "now"),
    (r"\bfor the purpose of\b", "for"),
    (r"\bwith regard to\b", "regarding"),
    (r"\bprior to\b", "before"),
    (r"\bsubsequent to\b", "after"),
    (r"\bin the event that\b", "if"),
    (r"\bidentified in previous observation\b", "identified earlier"),
    (r"\bafter applying patch to\b", "after patching"),
    (r"\btraced to missing null check after\b", "traced to missing null check after"),
]


def _faithful_compress(text: str, target_ratio: float = 0.70) -> str:
    """Produce a compressed version preserving semantics at ~60-80% word count."""
    for pattern, replacement in _VERBOSE_PHRASES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    words = text.split()
    target = max(3, int(len(words) * target_ratio))

    # Keep words that carry meaning: numbers, uppercase tokens, hyphenated, path-like
    def is_content(w: str) -> bool:
        clean = w.strip(".,;:!?\"'()")
        return (
            any(c.isdigit() for c in clean)
            or clean[0].isupper()
            or "/" in clean
            or "-" in clean
            or "_" in clean
            or clean.lower() not in _STOPWORDS
        )

    # Two-pass: remove stopwords, then trim remaining if still over target
    filtered = [w for w in words if is_content(w)]
    if len(filtered) <= target:
        return " ".join(filtered)

    # Over target even without stopwords — drop every other low-info adverb/preposition
    result = []
    skip_next = False
    low_info = {"in", "on", "at", "to", "of", "for", "and", "or", "by", "as"}
    for w in filtered:
        if skip_next:
            skip_next = False
            continue
        if w.lower() in low_info and len(result) > target * 0.5:
            skip_next = True
            continue
        result.append(w)
        if len(result) >= target:
            break
    return " ".join(result)


def _generate_faithful(originals: List[str], n: int) -> List[dict]:
    pairs: List[dict] = []
    ratios = [0.62, 0.68, 0.72, 0.75, 0.78]
    for orig in (originals * ((n // len(originals)) + 1))[:n]:
        ratio = random.choice(ratios)
        compressed = _faithful_compress(orig, target_ratio=ratio)
        if compressed != orig and len(compressed.split()) >= 3:
            pairs.append({"compressed": compressed, "original": orig,
                          "label": "faithful", "transform": "paraphrase"})
    return pairs[:n]


# ── Hard negative transformations ─────────────────────────────────────────────

def _transform_step_abstraction(text: str) -> str:
    """Replace step-specific references with vague descriptions.

    This is the core transform for fixing AMA-Bench: the predictor learns that
    stripping step indices is a semantic loss, not just surface noise.
    """
    result = re.sub(
        r"\b(at\s+)?step\s+\d+\s+(action|observation)?\s*:?\s*",
        "during the task ",
        text, flags=re.IGNORECASE,
    )
    result = re.sub(r"\bat step \d+\b", "at some point", result, flags=re.IGNORECASE)
    result = re.sub(r"\bstep \d+\b", "a step", result, flags=re.IGNORECASE)
    # Collapse any resulting double spaces
    result = re.sub(r"\s{2,}", " ", result).strip()
    if result == text:
        # No step refs — abstract the whole event to a vague description
        result = re.sub(r"(modified|created|deleted|refactored)\s+\S+", "made a code change", result)
    return result


def _transform_number_removal(text: str) -> str:
    """Replace specific counts, durations, port numbers with vague quantifiers."""
    result = re.sub(r"\b\d{3,}\b", "many", text)          # large numbers
    result = re.sub(r"\b\d+m\d*s\b", "some time", result) # durations like 4m32s
    result = re.sub(r"\b\d+s\b", "some time", result)     # bare seconds
    result = re.sub(r"\b\d+ms\b", "some time", result)
    result = re.sub(r"\b\d+h\d*m?\b", "some time", result)
    result = re.sub(r"line \d+", "a line", result)
    result = re.sub(r"port \d+", "a port", result)
    result = re.sub(r"\b\d+ (unit |integration )?tests?\b", "some tests", result)
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result


def _transform_proper_noun_hypernym(text: str) -> str:
    """Replace file paths, env vars, service names with generic hypernyms."""
    # file paths → "the file"
    result = re.sub(r"\b\S+\.(ts|py|js|rb|sql|yml|yaml|json|toml)\b", "the file", text)
    # env vars (ALL_CAPS with underscores) → "the variable"
    result = re.sub(r"\b[A-Z][A-Z0-9_]{3,}\b", "the variable", result)
    # HTTP methods + paths → "the endpoint"
    result = re.sub(r"\b(GET|POST|PUT|PATCH|DELETE)\s+/\S+", "the endpoint", result)
    # known service names → "the service"
    result = re.sub(
        r"\b(Redis|Postgres|MySQL|MongoDB|Elasticsearch|RabbitMQ|Kafka|Consul|Vault|Stripe)\b",
        "the service", result,
    )
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result


def _transform_causal_elision(text: str) -> str:
    """Strip causal clauses — what happened without why."""
    result = re.sub(r"\s*(—|-)?\s*due to\b.*$", "", text, flags=re.IGNORECASE)
    result = re.sub(r"\s*(—|-)?\s*because\b.*$", "", result, flags=re.IGNORECASE)
    result = re.sub(r"\s*(—|-)?\s*after\b.*$", "", result, flags=re.IGNORECASE)
    result = re.sub(r"\s*(—|-)?\s*traced to\b.*$", "", result, flags=re.IGNORECASE)
    result = re.sub(r"\s*(—|-)?\s*identified in\b.*$", "", result, flags=re.IGNORECASE)
    result = result.strip().rstrip(".,;:")
    return result if len(result.split()) >= 3 else text


def _transform_specificity_drop(text: str) -> str:
    """Replace specific technical details with generic alternatives."""
    result = re.sub(
        r"error \d{3} on \S+", "an API error occurred", text, flags=re.IGNORECASE
    )
    result = re.sub(
        r"(TypeError|ValueError|KeyError|IntegrityError|ConnectionRefusedError"
        r"|TimeoutError|AssertionError|AttributeError|PermissionError|ImportError)"
        r"[^.]*",
        "an error occurred",
        result,
    )
    result = re.sub(r"lock timeout after \d+s", "a timeout", result)
    result = re.sub(r"\d+ concurrent connections", "high load", result)
    result = re.sub(r"query time dropped from \S+ to \S+", "performance improved", result)
    result = re.sub(r"replication lag \d+ms", "replication issues", result)
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result


_TRANSFORMS: List[Tuple[str, Callable[[str], str]]] = [
    ("step_abstraction",     _transform_step_abstraction),
    ("number_removal",       _transform_number_removal),
    ("proper_noun_hypernym", _transform_proper_noun_hypernym),
    ("causal_elision",       _transform_causal_elision),
    ("specificity_drop",     _transform_specificity_drop),
]


def _generate_hard_negatives(originals: List[str], n: int) -> List[dict]:
    """
    Generate n hard negatives with step_abstraction guaranteed ≥ 20%.
    Distribution across transforms is balanced after meeting that floor.
    """
    step_floor = max(int(n * 0.20), 1)
    per_other  = (n - step_floor) // (len(_TRANSFORMS) - 1)
    remainder  = n - step_floor - per_other * (len(_TRANSFORMS) - 1)

    counts = {"step_abstraction": step_floor}
    for name, _ in _TRANSFORMS:
        if name != "step_abstraction":
            counts[name] = per_other
    # distribute remainder into first non-step bucket
    for name in counts:
        if name != "step_abstraction":
            counts[name] += remainder
            break

    pairs: List[dict] = []
    orig_cycle = (originals * ((n // len(originals)) + 2))
    random.shuffle(orig_cycle)
    i = 0
    for tname, tfunc in _TRANSFORMS:
        needed = counts[tname]
        added  = 0
        while added < needed and i < len(orig_cycle):
            orig = orig_cycle[i]
            i += 1
            transformed = tfunc(orig)
            if transformed != orig and len(transformed.split()) >= 3:
                pairs.append({"compressed": transformed, "original": orig,
                               "label": "hard_negative", "transform": tname})
                added += 1

    random.shuffle(pairs)
    return pairs[:n]


# ── SWE-bench pairs ───────────────────────────────────────────────────────────

def _generate_swebench(n: int) -> List[dict]:
    """Load SWE-bench_Verified problem statements and apply random hard-negative transforms."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    problems = [row["problem_statement"] for row in ds if row.get("problem_statement")]
    random.shuffle(problems)

    pairs: List[dict] = []
    for prob in problems:
        if len(pairs) >= n:
            break
        # Take first sentence as the "original" (long statements are noisy)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", prob) if len(s.split()) > 8]
        if not sentences:
            continue
        orig = sentences[0][:400]
        tname, tfunc = random.choice(_TRANSFORMS)
        transformed = tfunc(orig)
        if transformed != orig and len(transformed.split()) >= 3:
            pairs.append({"compressed": transformed, "original": orig,
                           "label": "hard_negative", "transform": f"swebench_{tname}"})
    return pairs[:n]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pairs_v2.jsonl for predictor training")
    parser.add_argument("--n-faithful",      type=int, default=5000)
    parser.add_argument("--n-hard-negative", type=int, default=5000)
    parser.add_argument("--swebench-pairs",  type=int, default=500)
    parser.add_argument("--out",             default="train/data/pairs_v2.jsonl")
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Append timestamp suffix if output file already exists
    if out_path.exists():
        import time
        stem = out_path.stem
        suffix = out_path.suffix
        ts = int(time.time())
        out_path = out_path.with_name(f"{stem}_{ts}{suffix}")
        print(f"Output exists — writing to {out_path}")

    print("Building original event pool...")
    originals = _originals()
    print(f"  {len(originals)} originals in pool")

    print(f"Generating {args.n_faithful} faithful pairs...")
    faithful = _generate_faithful(originals, args.n_faithful)
    print(f"  generated {len(faithful)}")

    print(f"Generating {args.n_hard_negative} hard negative pairs...")
    hard_neg = _generate_hard_negatives(originals, args.n_hard_negative)
    print(f"  generated {len(hard_neg)}")
    step_abs_count = sum(1 for p in hard_neg if p["transform"] == "step_abstraction")
    print(f"  step_abstraction: {step_abs_count} ({step_abs_count/len(hard_neg):.1%})")

    print(f"Generating {args.swebench_pairs} SWE-bench pairs...")
    swebench = _generate_swebench(args.swebench_pairs)
    print(f"  generated {len(swebench)}")

    all_pairs = faithful + hard_neg + swebench
    random.shuffle(all_pairs)

    with out_path.open("w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\nWrote {len(all_pairs)} pairs to {out_path}")
    print("\nLabel distribution:")
    by_label: dict = {}
    by_transform: dict = {}
    for p in all_pairs:
        by_label[p["label"]] = by_label.get(p["label"], 0) + 1
        by_transform[p["transform"]] = by_transform.get(p["transform"], 0) + 1
    for label, count in sorted(by_label.items()):
        print(f"  {label}: {count}")
    print("\nTransform distribution:")
    for tname, count in sorted(by_transform.items(), key=lambda x: -x[1]):
        print(f"  {tname}: {count}")


if __name__ == "__main__":
    main()
