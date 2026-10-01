#!/usr/bin/env python3
"""
jeval/benchmarks/droid_bench.py

End-to-end benchmark: jeval vs truncation vs simple-mem baselines.
Uses Factory's 4-probe evaluation methodology scored by Mistral via NVIDIA NIM.

Probe types (Factory methodology):
  RECALL       — can the agent recall a specific fact from the session?
  ARTIFACT     — does the compressed context contain a critical artifact?
  CONTINUATION — can the agent continue the task correctly from the compressed context?
  DECISION     — does the compressed context preserve a key decision and its rationale?

Score: LLM judge rates 0-5 per probe. Final score = mean across all probes.

Usage:
  export NVIDIA_API_KEY=...
  python3.12 jeval/benchmarks/droid_bench.py
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from typing import List

from openai import OpenAI

from jeval.compress.adaptive import AdaptiveCompressor
from jeval.compress.llm import LLMBackend
from jeval.baselines.truncation import TruncationCompressor
from jeval.baselines.simple_mem import SimpleMemCompressor
from jeval.ingest.base import Segment, Session

# ── NIM client ────────────────────────────────────────────────────────────────

_MODEL = "qwen/qwen3.5-122b-a10b"


def _nim_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["NVIDIA_API_KEY"],
        base_url="https://integrate.api.nvidia.com/v1",
    )


def nim_complete(client: OpenAI, prompt: str, max_tokens: int = 512) -> str:
    resp = client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


# ── Probe definitions ─────────────────────────────────────────────────────────

@dataclass
class Probe:
    probe_type: str   # RECALL | ARTIFACT | CONTINUATION | DECISION
    question: str
    answer: str       # ground truth expected in response


# ── Sessions ──────────────────────────────────────────────────────────────────

SESSIONS: list[dict] = [
    {
        "id": "auth-bug-fix",
        "description": "JWT authentication bug across 3 compression rounds",
        "segments": [
            Segment(text="Task: fix POST /api/login returning 401 for valid credentials", role="user", turn=0, source="droid"),
            Segment(text="modified src/middleware/auth.ts — added missing Authorization header check", role="assistant", turn=1, source="droid"),
            Segment(text="modified src/config/redis.ts — set maxRetriesPerRequest to 10 to fix connection timeout", role="assistant", turn=2, source="droid"),
            Segment(text="decided to use Redis over Postgres for session storage due to connection pool exhaustion under load testing", role="assistant", turn=3, source="droid"),
            Segment(text="rejected JWT stored in localStorage — XSS risk confirmed, switching to httpOnly cookies", role="assistant", turn=4, source="droid"),
            Segment(text="error 401 persists after auth.ts change — investigating JWT_SECRET env var mismatch between local and prod", role="assistant", turn=5, source="droid"),
            Segment(text="root cause found: JWT_SECRET was JWT_KEY in production env — corrected in src/config/env.ts", role="assistant", turn=6, source="droid"),
            Segment(text="standup note: team aligned on Redis session strategy, sprint blocker resolved", role="user", turn=7, source="droid"),
            Segment(text="created tests/auth.test.ts — 14 unit tests covering JWT middleware and cookie handling, all passing", role="assistant", turn=8, source="droid"),
            Segment(text="good morning, excited for the sprint review today", role="user", turn=9, source="droid"),
            Segment(text="rate limiting active: 5 requests per minute on POST /api/login via src/middleware/rateLimit.ts", role="assistant", turn=10, source="droid"),
            Segment(text="next: deploy to staging and verify POST /api/login returns 200 with valid credentials", role="assistant", turn=11, source="droid"),
        ],
        "probes": [
            Probe("ARTIFACT",     "What file was modified to fix the JWT_SECRET mismatch?",                 "src/config/env.ts"),
            Probe("ARTIFACT",     "What file contains the rate limiting middleware?",                        "src/middleware/rateLimit.ts"),
            Probe("DECISION",     "Why was Redis chosen over Postgres for session storage?",                 "connection pool exhaustion"),
            Probe("DECISION",     "Why was localStorage rejected for JWT storage?",                          "XSS risk"),
            Probe("RECALL",       "How many unit tests were created in tests/auth.test.ts?",                 "14"),
            Probe("RECALL",       "What was the maxRetriesPerRequest value set to in redis.ts?",             "10"),
            Probe("CONTINUATION", "What is the next step after the rate limiting fix?",                      "deploy to staging"),
        ],
    },
    {
        "id": "database-migration",
        "description": "Schema migration with rollback decisions and error handling",
        "segments": [
            Segment(text="Task: migrate users table to add nullable profile_image_url column without downtime", role="user", turn=0, source="droid"),
            Segment(text="created migrations/0042_add_profile_image_url.sql — ALTER TABLE users ADD COLUMN profile_image_url VARCHAR(512)", role="assistant", turn=1, source="droid"),
            Segment(text="decided to use nullable column (not NOT NULL) to avoid backfill lock on 2.3M row table", role="assistant", turn=2, source="droid"),
            Segment(text="error: migration failed on staging — lock timeout after 30s on users table, 847 concurrent connections", role="assistant", turn=3, source="droid"),
            Segment(text="rolled back migrations/0042 — switching to online schema change via pt-online-schema-change", role="assistant", turn=4, source="droid"),
            Segment(text="modified src/db/migration_runner.ts — added pt-osc wrapper with --max-load=Threads_running:100", role="assistant", turn=5, source="droid"),
            Segment(text="good morning team, coffee is ready in the kitchen", role="user", turn=6, source="droid"),
            Segment(text="pt-osc run succeeded on staging — migration completed in 4m32s with zero downtime, all connections maintained", role="assistant", turn=7, source="droid"),
            Segment(text="updated src/models/user.ts — added profileImageUrl?: string field with nullable mapping", role="assistant", turn=8, source="droid"),
            Segment(text="created tests/migration.test.ts — 8 tests verifying column existence, nullability, and index creation", role="assistant", turn=9, source="droid"),
            Segment(text="MIGRATION_SECRET_KEY rotated after staging run as per security protocol, updated in src/config/secrets.ts", role="assistant", turn=10, source="droid"),
            Segment(text="ready to run on production — requires DBA approval for tables above 1M rows per policy DB-SEC-04", role="assistant", turn=11, source="droid"),
        ],
        "probes": [
            Probe("ARTIFACT",     "What migration file was created for the schema change?",                  "migrations/0042_add_profile_image_url.sql"),
            Probe("ARTIFACT",     "What file was updated to add the profileImageUrl field to the user model?", "src/models/user.ts"),
            Probe("DECISION",     "Why was a nullable column chosen instead of NOT NULL?",                   "backfill lock"),
            Probe("DECISION",     "Why was the initial migration rolled back?",                               "lock timeout"),
            Probe("RECALL",       "How long did the pt-osc migration take on staging?",                      "4m32s"),
            Probe("RECALL",       "How many concurrent connections were present during the failed migration?", "847"),
            Probe("CONTINUATION", "What is required before running the migration on production?",             "DBA approval"),
        ],
    },
    {
        "id": "api-gateway-refactor",
        "description": "Microservice extraction with cascading dependency changes",
        "segments": [
            Segment(text="Task: extract payment processing from monolith into standalone payments-service", role="user", turn=0, source="droid"),
            Segment(text="created services/payments-service/src/index.ts — Express app on port 3002 with /charge and /refund endpoints", role="assistant", turn=1, source="droid"),
            Segment(text="decided to use async event queue (Redis Streams) over direct HTTP calls — avoids tight coupling and supports retry", role="assistant", turn=2, source="droid"),
            Segment(text="modified src/api/gateway.ts — added proxy route /payments/* → payments-service:3002 via http-proxy-middleware", role="assistant", turn=3, source="droid"),
            Segment(text="error 502 on POST /payments/charge — payments-service not registered in service discovery (Consul)", role="assistant", turn=4, source="droid"),
            Segment(text="registered payments-service in config/consul/services.json — health check on GET /health every 10s", role="assistant", turn=5, source="droid"),
            Segment(text="weekly sync note: platform team reviewing service mesh proposal next Thursday", role="user", turn=6, source="droid"),
            Segment(text="rejected gRPC for inter-service comms — team unfamiliar, would require proto schema versioning overhead", role="assistant", turn=7, source="droid"),
            Segment(text="PAYMENTS_STRIPE_SECRET_KEY added to src/config/env.ts and rotated in Vault — old key invalidated", role="assistant", turn=8, source="droid"),
            Segment(text="modified src/monolith/billing.ts — removed direct Stripe calls, now publishes to payments.charge stream", role="assistant", turn=9, source="droid"),
            Segment(text="created tests/payments-integration.test.ts — 22 tests covering charge, refund, and retry logic", role="assistant", turn=10, source="droid"),
            Segment(text="rollout plan: dark launch payments-service behind feature flag PAYMENTS_V2_ENABLED for 48h before cutover", role="assistant", turn=11, source="droid"),
        ],
        "probes": [
            Probe("ARTIFACT",     "What file configures the Consul service registration for payments-service?", "config/consul/services.json"),
            Probe("ARTIFACT",     "What file was modified to remove direct Stripe calls from the monolith?",    "src/monolith/billing.ts"),
            Probe("DECISION",     "Why was Redis Streams chosen over direct HTTP for inter-service communication?", "tight coupling"),
            Probe("DECISION",     "Why was gRPC rejected for inter-service communication?",                    "proto schema versioning"),
            Probe("RECALL",       "What port does payments-service run on?",                                   "3002"),
            Probe("RECALL",       "How many integration tests were created for payments-service?",              "22"),
            Probe("CONTINUATION", "What is the rollout strategy for payments-service before full cutover?",    "dark launch"),
        ],
    },
]


# ── LLM judge ─────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """\
You are evaluating a context compression system for an AI coding agent.

The agent was given this compressed context:
<compressed_context>
{context}
</compressed_context>

Answer this question based ONLY on the compressed context above:
Question: {question}

Expected answer contains: "{answer}"

Rate on a scale of 0-5:
5 = answer present and correct, no ambiguity
4 = answer present but requires inference
3 = answer partially present, key detail missing
2 = answer vaguely implied but not recoverable
1 = answer absent but context is coherent
0 = context is incoherent or misleading

Respond with JSON only: {{"score": <int>, "reasoning": "<one sentence>"}}"""


def judge_probe(client: OpenAI, compressed_context: str, probe: Probe) -> tuple[int, str]:
    prompt = JUDGE_PROMPT.format(
        context=compressed_context,
        question=probe.question,
        answer=probe.answer,
    )
    raw = nim_complete(client, prompt, max_tokens=200)
    try:
        clean = raw.strip().strip("```json").strip("```").strip()
        result = json.loads(clean)
        return int(result["score"]), result.get("reasoning", "")
    except Exception:
        for ch in raw:
            if ch.isdigit() and int(ch) <= 5:
                return int(ch), raw[:100]
        return 0, f"parse_failed: {raw[:100]}"


# ── Compression helpers ───────────────────────────────────────────────────────

from jeval.memory.jeval_memory import JevalMemory as JevalMemoryV2

def compress_with_jeval(session: Session) -> str:
    db_path = f".jeval/droid_memory.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    mem = JevalMemoryV2(db_path=db_path, frozen_mode=True)
    
    # We use LLMBackend internally but since JevalMemoryV2 uses NIM_Caller we just call ingest
    for seg in session:
        mem.ingest(seg.text)
    
    entries = mem._hot_cache.get_all_entries()
    # Sort by seq_id to maintain timeline order
    entries.sort(key=lambda x: x["seq_id"])
    return "\n".join(f"[{e['seq_id']}] {e['text']}" for e in entries)



def compress_with_baseline(session: Session, name: str, budget: float = 0.5) -> str:
    full_text = "\n".join(seg.text for seg in session)
    if name == "truncation":
        return TruncationCompressor().compress(full_text, budget)
    if name == "simple-mem":
        return SimpleMemCompressor().compress(full_text, budget)
    raise ValueError(name)


# ── Loader (mirrors AMABenchLoader interface) ─────────────────────────────────

class DroidBenchLoader:
    """Load Droid-style sessions for the Factory 4-probe benchmark."""

    def load_sessions(self) -> List[Session]:
        return [
            Session(session_id=cfg["id"], segments=cfg["segments"])
            for cfg in SESSIONS
        ]


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class SystemResult:
    name: str
    session_scores: List[float] = field(default_factory=list)
    probe_scores: List[tuple] = field(default_factory=list)  # (session_id, probe_type, score, reasoning)

    @property
    def mean_score(self) -> float:
        return statistics.mean(self.session_scores) if self.session_scores else 0.0

    @property
    def score_by_probe_type(self) -> dict:
        by_type: dict = {}
        for _, ptype, score, _ in self.probe_scores:
            by_type.setdefault(ptype, []).append(score)
        return {k: round(statistics.mean(v), 2) for k, v in by_type.items()}


# ── Runner ────────────────────────────────────────────────────────────────────

def run_benchmark() -> None:
    client = _nim_client()

    systems = {
        "jeval":      lambda s: compress_with_jeval(s),
        "truncation": lambda s: compress_with_baseline(s, "truncation", 0.5),
        "simple-mem": lambda s: compress_with_baseline(s, "simple-mem", 0.5),
    }

    results = {name: SystemResult(name=name) for name in systems}
    
    out_dir = os.path.join(os.path.dirname(__file__), "../../benchmarks/results/droid_bench_episodes")
    os.makedirs(out_dir, exist_ok=True)

    for sess_cfg in SESSIONS:
        session = Session(session_id=sess_cfg["id"], segments=sess_cfg["segments"])
        probes: List[Probe] = sess_cfg["probes"]
        
        session_log = {
            "session_id": sess_cfg["id"],
            "description": sess_cfg["description"],
            "original_segments": [s.text for s in session],
            "systems": {}
        }

        print(f"\n{'='*60}")
        print(f"Session: {sess_cfg['id']} — {sess_cfg['description']}")
        print(f"  {len(session)} segments, {len(probes)} probes")

        for sys_name, compress_fn in systems.items():
            print(f"\n  [{sys_name}] compressing...", end=" ", flush=True)
            compressed = compress_fn(session)
            orig_words = sum(len(s.text.split()) for s in session)
            token_reduction = 1.0 - len(compressed.split()) / max(1, orig_words)
            print(f"reduction={token_reduction:.0%}")
            
            sys_log = {
                "compressed_context": compressed,
                "token_reduction": token_reduction,
                "probes": []
            }

            session_probe_scores = []
            for probe in probes:
                score, reasoning = judge_probe(client, compressed, probe)
                session_probe_scores.append(score)
                results[sys_name].probe_scores.append((sess_cfg["id"], probe.probe_type, score, reasoning))
                print(f"    [{probe.probe_type:12s}] {score}/5 — {reasoning[:80]}")
                
                sys_log["probes"].append({
                    "type": probe.probe_type,
                    "question": probe.question,
                    "expected": probe.answer,
                    "score": score,
                    "reasoning": reasoning
                })

            session_mean = statistics.mean(session_probe_scores)
            results[sys_name].session_scores.append(session_mean)
            sys_log["session_mean"] = session_mean
            session_log["systems"][sys_name] = sys_log
            print(f"  [{sys_name}] session mean: {session_mean:.2f}/5")
            
        with open(os.path.join(out_dir, f"episode_{sess_cfg['id']}.json"), "w") as f:
            json.dump(session_log, f, indent=2)

    # ── Results table ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"{'System':<14} {'Overall':>8} {'RECALL':>8} {'ARTIFACT':>9} {'CONTINUE':>9} {'DECISION':>9}")
    print("-" * 60)

    for name, res in results.items():
        by_type = res.score_by_probe_type
        print(
            f"{name:<14} "
            f"{res.mean_score:>7.2f}/5"
            f"{by_type.get('RECALL', 0):>8.2f}"
            f"{by_type.get('ARTIFACT', 0):>9.2f}"
            f"{by_type.get('CONTINUATION', 0):>9.2f}"
            f"{by_type.get('DECISION', 0):>9.2f}"
        )

    print(f"\nFactory AI baseline (published Dec 2025): 2.45/5")
    print(f"Anthropic baseline (published Dec 2025):  2.33/5")
    print(f"OpenAI baseline (published Dec 2025):     2.19/5")

    raw = {
        name: {
            "mean": res.mean_score,
            "by_probe_type": res.score_by_probe_type,
            "all_probes": [
                {"session": s, "type": t, "score": sc, "reasoning": r}
                for s, t, sc, r in res.probe_scores
            ],
        }
        for name, res in results.items()
    }
    out_path = os.path.join(os.path.dirname(__file__), "../../benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\nRaw results saved to benchmark_results.json")


if __name__ == "__main__":
    run_benchmark()
