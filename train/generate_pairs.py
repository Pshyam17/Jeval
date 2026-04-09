#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

# encoder loaded once here, shared by all generation functions
from sentence_transformers import SentenceTransformer
_ENC: Optional[SentenceTransformer] = None

def _enc() -> SentenceTransformer:
    global _ENC
    if _ENC is None:
        print("loading encoder...")
        _ENC = SentenceTransformer("all-mpnet-base-v2")
    return _ENC


# ── original pool ─────────────────────────────────────────────────────────────

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
    pool: List[str] = []

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

    for var in _ENVVARS:
        pool.append(f"{var} mismatch between staging and production environments corrected")
        pool.append(f"rotated {var} in Vault — old key invalidated across all services")
        pool.append(f"added {var} to .env.example with documentation comment")

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

    for ep in _ENDPOINTS:
        for code in random.sample(_HTTP_CODES, k=2):
            pool.append(
                f"error {code} on {ep} — service not registered in Consul service discovery"
            )
        pool.append(f"created {ep} endpoint with request validation and error handling")

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

    for err in _ERRORS:
        pool.append(
            f"error in production: {err} — traced to missing null check after "
            f"OAuth token expiry"
        )
        pool.append(f"caught {err} at line {random.randint(10, 400)} in {random.choice(_FILES)}")

    random.shuffle(pool)
    return pool


# ── faithful compression ──────────────────────────────────────────────────────

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
]


def _faithful_compress(text: str, target_ratio: float = 0.70) -> str:
    for pattern, replacement in _VERBOSE_PHRASES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    words = text.split()
    target = max(3, int(len(words) * target_ratio))

    def is_content(w: str) -> bool:
        clean = w.strip(".,;:!?\"'()")
        return (
            any(c.isdigit() for c in clean)
            or (clean and clean[0].isupper())
            or "/" in clean
            or "-" in clean
            or "_" in clean
            or clean.lower() not in _STOPWORDS
        )

    filtered = [w for w in words if is_content(w)]
    if len(filtered) <= target:
        return " ".join(filtered)

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


# ── hard negative transforms ──────────────────────────────────────────────────

_OUTCOME_MAP = {
    "failed": "succeeded",
    "passed": "failed",
    "rolled back": "committed",
    "timed out": "completed",
    "crashed": "recovered",
    "deployed": "reverted",
    "resolved": "persisted",
    "fixed": "regressed",
    "succeeded": "failed",
    "completed": "timed out",
    "recovered": "crashed",
    "reverted": "deployed",
    "persisted": "resolved",
    "regressed": "fixed",
    "committed": "rolled back",
    "broke": "fixed",
}

def _transform_outcome_inversion(text: str) -> Optional[str]:
    for outcome, opposite in _OUTCOME_MAP.items():
        pattern = re.compile(r'\b' + re.escape(outcome) + r'\b', re.IGNORECASE)
        if pattern.search(text):
            result = pattern.sub(opposite, text, count=1)
            if result != text:
                return result
    return None


_STEP_RE = re.compile(r'\b(step\s+)(\d+)\b|\[step\s+(\d+)\]', re.IGNORECASE)

def _transform_step_number_swap(text: str) -> Optional[str]:
    m = _STEP_RE.search(text)
    if not m:
        return None
    n = int(m.group(2) or m.group(3))
    candidates = [n * 2, n + 37, abs(n - 13) + 1, n * 3 + 7]
    candidates = [c for c in candidates if c > 0 and c != n]
    if not candidates:
        return None
    new_n = random.choice(candidates)
    return _STEP_RE.sub(lambda mo: (mo.group(1) or "[step ") + str(new_n) + ("]" if mo.group(3) else ""), text, count=1)


_CAUSAL_RE = re.compile(
    r'(.*?)\s+(because|due to|caused by|triggered by|resulting from|after)\s+(.*)',
    re.IGNORECASE | re.DOTALL,
)

_CAUSE_INVERSIONS = [
    ("lock timeout", "successful completion"),
    ("memory was exhausted", "memory was available"),
    ("connection refused", "connection established"),
    ("authentication failure", "successful authentication"),
    ("permission denied", "permission granted"),
    ("network unreachable", "network connectivity restored"),
    ("disk full", "disk space available"),
    ("rate limit exceeded", "rate limit not reached"),
    ("invalid token", "valid token"),
    ("missing dependency", "all dependencies satisfied"),
]

def _transform_causal_inversion(text: str) -> Optional[str]:
    m = _CAUSAL_RE.search(text)
    if not m:
        return None
    effect, connector, cause = m.group(1), m.group(2), m.group(3)
    # pick an inversion that's different from the existing cause
    for orig_cause, inv_cause in random.sample(_CAUSE_INVERSIONS, len(_CAUSE_INVERSIONS)):
        if orig_cause.lower() not in cause.lower():
            return f"{effect} {connector} {inv_cause}"
    # fallback: use first inversion unconditionally
    return f"{effect} {connector} {_CAUSE_INVERSIONS[0][1]}"


_SENTIMENT_MAP = {
    "working":     "broken",
    "healthy":     "degraded",
    "passing":     "failing",
    "clean":       "corrupted",
    "stable":      "unstable",
    "valid":       "invalid",
    "correct":     "incorrect",
    "successful":  "unsuccessful",
    "broken":      "working",
    "failing":     "passing",
    "corrupted":   "clean",
    "unstable":    "stable",
    "invalid":     "valid",
    "incorrect":   "correct",
    "erroring":    "healthy",
    "degraded":    "healthy",
    "unsuccessful": "successful",
}

def _transform_sentiment_flip(text: str) -> Optional[str]:
    for word, opposite in _SENTIMENT_MAP.items():
        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
        if pattern.search(text):
            result = pattern.sub(opposite, text, count=1)
            if result != text:
                return result
    return None


_CLAUSE_RE = re.compile(
    r'(.+?)\s+(after|due to|because|—|triggered by|resulting from)\s+(.+)',
    re.IGNORECASE,
)

def _transform_context_swap(text_a: str, text_b: str) -> Optional[str]:
    # try to extract a causal clause from B and attach to subject of A
    m_b = _CLAUSE_RE.search(text_b)
    m_a = _CLAUSE_RE.search(text_a)
    if m_b:
        subject_a = m_a.group(1) if m_a else text_a
        connector_b = m_b.group(2)
        clause_b = m_b.group(3)
        result = f"{subject_a} {connector_b} {clause_b}"
        if result != text_a and result != text_b:
            return result
    if m_a:
        subject_b = m_b.group(1) if m_b else text_b
        connector_a = m_a.group(2)
        clause_a = m_a.group(3)
        result = f"{subject_b} {connector_a} {clause_a}"
        if result != text_a and result != text_b:
            return result
    return None


# ── batch encode-and-filter ───────────────────────────────────────────────────

def _filter_by_similarity(
    candidates: List[dict],
    threshold: float = 0.90,
    chunk: int = 512,
) -> Tuple[List[dict], int]:
    """
    Encode compressed+original in batches, reject pairs with cos_sim > threshold.
    Returns (kept_pairs, n_rejected).
    """
    enc = _enc()
    kept: List[dict] = []
    rejected = 0

    for i in range(0, len(candidates), chunk):
        batch = candidates[i : i + chunk]
        comp_texts = [p["compressed"] for p in batch]
        orig_texts = [p["original"]   for p in batch]

        comp_embs = enc.encode(comp_texts, batch_size=64, show_progress_bar=False,
                               normalize_embeddings=True)
        orig_embs = enc.encode(orig_texts, batch_size=64, show_progress_bar=False,
                               normalize_embeddings=True)

        sims = np.einsum("ij,ij->i", comp_embs, orig_embs)  # dot of normalized = cos_sim
        for pair, sim in zip(batch, sims):
            pair["_cos_sim"] = float(sim)
            if float(sim) <= threshold:
                kept.append(pair)
            else:
                rejected += 1

    return kept, rejected


# ── faithful generation ───────────────────────────────────────────────────────

def _generate_faithful(originals: List[str], n: int) -> List[dict]:
    pairs: List[dict] = []
    ratios = [0.62, 0.68, 0.72, 0.75, 0.78]
    pool = (originals * ((n // len(originals)) + 1))[:n * 2]
    random.shuffle(pool)
    candidates: List[dict] = []
    for orig in pool:
        ratio = random.choice(ratios)
        compressed = _faithful_compress(orig, target_ratio=ratio)
        if compressed != orig and len(compressed.split()) >= 3:
            candidates.append({
                "compressed": compressed,
                "original":   orig,
                "label":      "faithful",
                "strategy":   "paraphrase",
            })
        if len(candidates) >= n * 2:
            break

    # faithful pairs: keep only those with cos_sim <= 0.98 (near-identical check)
    # faithful pairs are MEANT to be similar, but identical = useless
    kept, _ = _filter_by_similarity(candidates, threshold=0.98)
    return kept[:n]


# ── hard negative generation ──────────────────────────────────────────────────

def _make_pool(originals: List[str]) -> List[str]:
    pool = originals * 10
    random.shuffle(pool)
    return pool


def _generate_strategy(
    strategy_name: str,
    transform_fn: Callable,
    originals: List[str],
    target: int,
    cos_threshold: float = 0.90,
    chunk: int = 512,
    is_pairwise: bool = False,
) -> Tuple[List[dict], int]:
    """
    Generate up to target pairs for one strategy.
    Each transform gets its own independent cursor over a fresh shuffled pool.
    Returns (pairs, n_filtered).
    """
    pool = _make_pool(originals)
    candidates: List[dict] = []
    total_filtered = 0

    if is_pairwise:
        # context_swap: pairs of originals
        i = 0
        while len(candidates) + total_filtered < target * 4 and i + 1 < len(pool):
            a, b = pool[i], pool[i + 1]
            i += 2
            result = transform_fn(a, b)
            if result and result != a and len(result.split()) >= 3:
                candidates.append({
                    "compressed": result,
                    "original":   a,
                    "label":      "hard_negative",
                    "strategy":   strategy_name,
                })
    else:
        i = 0
        while len(candidates) + total_filtered < target * 4 and i < len(pool):
            orig = pool[i]
            i += 1
            result = transform_fn(orig)
            if result and result != orig and len(result.split()) >= 3:
                candidates.append({
                    "compressed": result,
                    "original":   orig,
                    "label":      "hard_negative",
                    "strategy":   strategy_name,
                })

    # batch filter
    kept, n_rej = _filter_by_similarity(candidates, threshold=cos_threshold)
    total_filtered += n_rej
    return kept[:target], total_filtered


# ── SWE-bench pairs ───────────────────────────────────────────────────────────

_STRATEGY_TRIGGERS = {
    "outcome_inversion": lambda t: bool(re.search(
        r'\b(failed|passed|succeeded|completed|resolved|fixed|broke|crashed|'
        r'timed out|rolled back|deployed|rejected)\b', t, re.IGNORECASE
    )),
    "step_number_swap": lambda t: bool(_STEP_RE.search(t)),
    "causal_inversion": lambda t: bool(re.search(
        r'\b(because|due to|caused by|triggered by|resulting from|after)\b', t, re.IGNORECASE
    )),
    "sentiment_flip": lambda t: bool(re.search(
        r'\b(working|healthy|passing|clean|stable|valid|correct|successful|'
        r'broken|failing|corrupted|unstable|invalid|incorrect|erroring|degraded)\b',
        t, re.IGNORECASE
    )),
}

_STRATEGY_FNS = {
    "outcome_inversion": _transform_outcome_inversion,
    "step_number_swap":  _transform_step_number_swap,
    "causal_inversion":  _transform_causal_inversion,
    "sentiment_flip":    _transform_sentiment_flip,
}

def _generate_swebench(n: int, per_strategy: int) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    problems = [row["problem_statement"] for row in ds if row.get("problem_statement")]
    random.shuffle(problems)

    # extract sentences from problem statements
    sentences: List[str] = []
    for prob in problems:
        parts = [s.strip() for s in re.split(r'(?<=[.!?])\s+', prob) if len(s.split()) > 6]
        sentences.extend(p[:400] for p in parts)
    random.shuffle(sentences)

    # route each sentence to strategies that will actually fire
    buckets: dict = {s: [] for s in _STRATEGY_TRIGGERS}
    for sent in sentences:
        for strat, trigger in _STRATEGY_TRIGGERS.items():
            if trigger(sent):
                buckets[strat].append(sent)

    all_pairs: List[dict] = []
    for strat, sents in buckets.items():
        fn = _STRATEGY_FNS[strat]
        candidates: List[dict] = []
        for sent in sents:
            result = fn(sent)
            if result and result != sent and len(result.split()) >= 3:
                candidates.append({
                    "compressed": result,
                    "original":   sent,
                    "label":      "hard_negative",
                    "strategy":   f"swebench_{strat}",
                })
            if len(candidates) >= per_strategy * 3:
                break
        kept, _ = _filter_by_similarity(candidates, threshold=0.90)
        all_pairs.extend(kept[:per_strategy])

    random.shuffle(all_pairs)
    return all_pairs[:n]


# ── summary stats ─────────────────────────────────────────────────────────────

def _print_summary(pairs: List[dict], total_filtered: int) -> None:
    enc = _enc()
    faithful = [p for p in pairs if p["label"] == "faithful"]
    hard_neg  = [p for p in pairs if p["label"] == "hard_negative"]

    by_strategy: dict = {}
    for p in hard_neg:
        s = p.get("strategy", "unknown")
        by_strategy.setdefault(s, []).append(p)

    def avg_sim(ps: List[dict]) -> float:
        if not ps:
            return float("nan")
        if "_cos_sim" in ps[0]:
            return float(np.mean([p["_cos_sim"] for p in ps]))
        comps = [p["compressed"] for p in ps[:200]]
        origs = [p["original"]   for p in ps[:200]]
        ce = enc.encode(comps, normalize_embeddings=True, show_progress_bar=False)
        oe = enc.encode(origs, normalize_embeddings=True, show_progress_bar=False)
        return float(np.einsum("ij,ij->i", ce, oe).mean())

    f_sim = avg_sim(faithful[:200])
    hn_sim = avg_sim(hard_neg[:200])

    print("\nstrategy breakdown:")
    for strat in ["outcome_inversion", "step_number_swap", "causal_inversion",
                  "sentiment_flip", "context_swap"] + \
                 [k for k in by_strategy if k.startswith("swebench_")]:
        ps = by_strategy.get(strat, [])
        s = avg_sim(ps) if ps else float("nan")
        gap = f_sim - s if ps else float("nan")
        print(f"  {strat:30s}  n={len(ps):4d}  avg_cos_sim={s:.4f}  gap={gap:+.4f}")

    print(f"  {'filtered (cos>0.90)':30s}  n={total_filtered:4d}  pairs rejected")
    print(f"  {'total hard negatives':30s}  n={len(hard_neg)}")
    print(f"  {'total faithful':30s}  n={len(faithful)}")
    print(f"  {'faithful avg_cos_sim':30s}  {f_sim:.4f}")
    print(f"  {'hard_neg avg_cos_sim':30s}  {hn_sim:.4f}")
    print(f"  {'gap':30s}  {f_sim - hn_sim:+.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-faithful",      type=int, default=5000)
    parser.add_argument("--n-hard-negative", type=int, default=5000)
    parser.add_argument("--swebench-pairs",  type=int, default=500)
    parser.add_argument("--out",             default="train/data/pairs_v3.jsonl")
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        ts = int(time.time())
        out_path = out_path.with_name(f"{out_path.stem}_{ts}{out_path.suffix}")
        print(f"output exists — writing to {out_path}")

    # load encoder up front
    _enc()

    print("building original event pool...")
    originals = _originals()
    print(f"  {len(originals)} originals")

    print(f"generating {args.n_faithful} faithful pairs...")
    faithful = _generate_faithful(originals, args.n_faithful)
    print(f"  generated {len(faithful)}")

    per_strategy = args.n_hard_negative // 5
    total_filtered = 0
    hard_neg: List[dict] = []

    strategies = [
        ("outcome_inversion", _transform_outcome_inversion, False),
        ("step_number_swap",  _transform_step_number_swap,  False),
        ("causal_inversion",  _transform_causal_inversion,  False),
        ("sentiment_flip",    _transform_sentiment_flip,    False),
        ("context_swap",      _transform_context_swap,      True),
    ]

    for name, fn, pairwise in strategies:
        print(f"generating {per_strategy} {name} pairs...")
        pairs, n_rej = _generate_strategy(
            name, fn, originals, per_strategy,
            cos_threshold=0.90, is_pairwise=pairwise,
        )
        total_filtered += n_rej
        hard_neg.extend(pairs)
        print(f"  generated {len(pairs)}  filtered {n_rej}")

    if args.swebench_pairs > 0:
        print(f"generating {args.swebench_pairs} swebench pairs...")
        sw = _generate_swebench(args.swebench_pairs, per_strategy=args.swebench_pairs // 4)
        hard_neg.extend(sw)
        print(f"  generated {len(sw)}")

    all_pairs = faithful + hard_neg
    # strip internal _cos_sim before writing
    for p in all_pairs:
        p.pop("_cos_sim", None)
    random.shuffle(all_pairs)

    with out_path.open("w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"\nwrote {len(all_pairs)} pairs to {out_path}")
    _print_summary(faithful + hard_neg, total_filtered)


if __name__ == "__main__":
    main()
