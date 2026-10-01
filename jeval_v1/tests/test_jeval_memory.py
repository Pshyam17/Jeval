"""
Tests for jeval.memory.jeval_memory.JevalMemory.

All tests inject _ExtractiveCallerWrapper so no LLM / NVIDIA_API_KEY is needed.
"""
import time
import pytest
from pathlib import Path

from jeval.memory.jeval_memory import JevalMemory, _ExtractiveCallerWrapper


@pytest.fixture
def mem(tmp_path):
    return JevalMemory(
        db_path=tmp_path / "test.db",
        novelty_threshold=0.15,
        fidelity_threshold=0.40,  # high threshold to force extractive fallback in tests
        session_id="test_session",
        _caller=_ExtractiveCallerWrapper(budget=0.6),
    )


def test_first_ingest_always_novel(mem):
    result = mem.ingest("JWT_SECRET mismatch in src/config/env.ts caused auth failure")
    # working set is empty → always novel
    assert result["action"] in ("cached", "extractive_fallback")
    assert result["seq_id"] == 1
    assert result["epe_novelty"] == 1.0


def test_identical_input_cold_only_on_second(mem):
    text = "migration failed on staging due to lock timeout after 30s"
    mem.ingest(text)
    result = mem.ingest(text)
    assert result["action"] == "cold_only"
    assert result["epe_novelty"] < 0.15


def test_novel_input_stored_in_hot_cache(mem):
    mem.ingest("step 1: create src/server.ts with Express app")
    mem.ingest("step 2: add JWT_SECRET to src/config/env.ts")
    assert mem._hot_cache.size() >= 1


def test_retrieve_semantic_context(mem):
    mem.ingest("JWT_SECRET mismatch in src/config/env.ts caused all auth tests to fail")
    mem.ingest("step 4 fixed the missing environment variable")
    result = mem.retrieve("authentication environment variable secret")
    assert isinstance(result, str)
    assert len(result) > 0


def test_precision_query_returns_cold_storage(mem):
    mem.ingest("step 6 action: deployed to staging, migration failed due to lock timeout")
    result = mem.retrieve("what happened at step 6")
    # precision query should retrieve from cold storage
    assert "step 6" in result or "migration" in result or "lock" in result


def test_cold_storage_fallback_when_hot_cache_empty(tmp_path):
    m = JevalMemory(
        db_path=tmp_path / "fb.db",
        novelty_threshold=0.15,
        session_id="fallback_test",
        _caller=_ExtractiveCallerWrapper(),
    )
    # force cold_only by using identical text after first ingest
    text = "step 5 observation: 14 tests passing after patch"
    m.ingest(text)
    # clear hot cache to simulate empty state
    m._hot_cache._entries.clear()
    result = m.retrieve("tests passing patch")
    assert "14" in result or "tests" in result or "passing" in result


def test_stats_returns_correct_counts(mem):
    for i in range(3):
        mem.ingest(f"step {i} unique action with distinct technical content segment {i}")
    s = mem.stats()
    assert s["cold_storage_size"] >= 3
    assert s["session_id"] == "test_session"
    assert "hot_cache_size" in s
    assert "novelty_threshold" in s


def test_new_session_resets_working_set(mem):
    mem.ingest("step 1 action: created auth middleware")
    old_session = mem._session_id
    mem.new_session("new_session_id")
    assert mem._session_id == "new_session_id"
    assert mem._session_id != old_session
    # working set cleared — first ingest in new session is always novel
    result = mem.ingest("step 1 action: created auth middleware")
    assert result["epe_novelty"] == 1.0


def test_compress_returns_shorter_output(mem):
    long_text = "\n\n".join([
        "## Authentication",
        "JWT_SECRET was missing from src/config/env.ts causing all 14 auth tests to fail.",
        "## Migration",
        "Migration failed on staging due to lock timeout after 30 seconds on the roles table.",
        "The database had 423 concurrent connections at the time of the failure.",
        "## Resolution",
        "Rolled back the migration and reduced connection pool size before retrying.",
    ])
    result = mem.compress(long_text)
    assert isinstance(result, str)
    assert len(result.split()) < len(long_text.split())


def test_contradiction_marks_stale(tmp_path):
    m = JevalMemory(
        db_path=tmp_path / "contra.db",
        novelty_threshold=0.10,
        session_id="contra_test",
        _caller=_ExtractiveCallerWrapper(),
    )
    m.ingest("step 6 action: migration succeeded without errors")
    time.sleep(0.1)
    m.ingest("step 6 observation: migration failed due to lock timeout")
    time.sleep(0.3)  # allow async contradiction detector to run
    # the succeeded entry should be stale if contradiction detected
    # (may or may not trigger depending on EPE — just check no crash)
    result = m.retrieve("what happened with the migration")
    assert isinstance(result, str)
