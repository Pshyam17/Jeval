import pytest
from jeval.benchmarks.ama_bench_eval import JevalMemory


def test_embedding_retrieval_no_keyword_overlap():
    mem = JevalMemory(k=3)
    mem.store("JWT_SECRET mismatch corrected in src/config/env.ts")
    mem.store("migration failed 847 concurrent connections rolled back")
    mem.store("all 14 auth tests passing after rate limit fix")
    # query shares zero keywords with the first segment
    results = mem.retrieve("authentication credential environment variable wrong")
    assert any("JWT_SECRET" in r for r in results)


def test_retrieve_returns_at_most_k():
    mem = JevalMemory(k=2)
    for i in range(5):
        mem.store(f"segment number {i} about database migration lock timeout")
    results = mem.retrieve("database lock timeout")
    assert len(results) <= 2


def test_retrieve_empty_memory():
    mem = JevalMemory(k=3)
    assert mem.retrieve("anything") == []


def test_store_then_retrieve_single():
    mem = JevalMemory(k=1)
    mem.store("JWT_SECRET env var mismatch in production")
    results = mem.retrieve("jwt secret production")
    assert len(results) == 1
    assert "JWT_SECRET" in results[0]
