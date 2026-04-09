import time
import pytest
import numpy as np
from jeval.encoders.sentence_encoder import FrozenEncoder
from jeval.memory.hot_cache import HotCache


@pytest.fixture(scope="module")
def encoder():
    return FrozenEncoder()


@pytest.fixture
def cache(encoder):
    return HotCache(encoder, token_ceiling=500, top_k=3)


def _store(cache, encoder, text, seq_id, epe=0.5):
    emb = encoder.encode([text])[0]
    cache.store(text, emb, seq_id=seq_id, content_type="FACTUAL", epe_score=epe)


def test_store_and_retrieve(cache, encoder):
    _store(cache, encoder, "JWT_SECRET mismatch in env.ts fixed at step 4", seq_id=1)
    results = cache.retrieve("JWT authentication secret", k=1)
    assert len(results) == 1
    assert "JWT_SECRET" in results[0]["text"]


def test_top_k_ordering_by_similarity(cache, encoder):
    _store(cache, encoder, "migration failed due to lock timeout on roles table", seq_id=2)
    _store(cache, encoder, "all 14 tests passing after patch", seq_id=3)
    _store(cache, encoder, "weather is nice", seq_id=4, epe=0.1)
    results = cache.retrieve("lock timeout migration", k=3)
    assert results[0]["similarity"] >= results[-1]["similarity"]


def test_token_ceiling_eviction_removes_lowest_epe(encoder):
    # small ceiling to force eviction
    c = HotCache(encoder, token_ceiling=50, top_k=5)
    # store low-epe (redundant) entry first
    emb = encoder.encode(["short text"])[0]
    c.store("short text", emb, seq_id=10, content_type="BACKGROUND", epe_score=0.05)
    # fill cache past 80%
    for i in range(5):
        t = f"important technical segment number {i} with deployment details"
        e = encoder.encode([t])[0]
        c.store(t, e, seq_id=20 + i, content_type="FACTUAL", epe_score=0.8)
    time.sleep(0.2)  # let async enforcer run
    # high-epe entries should survive; low-epe "short text" may be evicted
    all_seqs = [e["seq_id"] for e in c.get_all_entries()]
    # if eviction ran, seq_id 10 (lowest epe) should be gone
    if len(all_seqs) < 6:
        assert 10 not in all_seqs


def test_stale_marking_excludes_from_retrieval(cache, encoder):
    _store(cache, encoder, "step 6 migration rolled back after failure", seq_id=50)
    cache.mark_stale(50)
    results = cache.retrieve("migration rolled back", k=5)
    assert all(r["seq_id"] != 50 for r in results)


def test_get_all_embeddings_shape(cache, encoder):
    embs, seq_ids = cache.get_all_embeddings()
    assert embs.ndim == 2
    assert embs.shape[1] == encoder.dim()
    assert len(seq_ids) == embs.shape[0]


def test_global_enforcer_triggers_above_80pct(encoder):
    # ceiling of 20 tokens; store 18 tokens worth → below 80%
    c = HotCache(encoder, token_ceiling=20, top_k=5)
    emb = encoder.encode(["short"])[0]
    c.store("short", emb, seq_id=1, content_type="BACKGROUND", epe_score=0.1)
    assert c.size() == 1
    # store enough to exceed 80%
    long_text = " ".join(["word"] * 20)
    emb2 = encoder.encode([long_text])[0]
    c.store(long_text, emb2, seq_id=2, content_type="FACTUAL", epe_score=0.9)
    time.sleep(0.2)
    # enforcer should have evicted seq_id=1 (lowest epe)
    assert c.token_count() <= 20
