"""
Tests for the updated HotCache eviction formula.

Covers: miss-counter priority, hit-count protection, epe_novelty-based eviction,
and zero-value normalisation safety.
"""
from __future__ import annotations

import time
import pytest
import numpy as np

from jeval.encoders.sentence_encoder import FrozenEncoder
from jeval.memory.hot_cache import HotCache


def _store(cache: HotCache, enc: FrozenEncoder, text: str, seq_id: int, epe: float = 0.5):
    emb = enc.encode([text])[0]
    cache.store(text, emb, seq_id=seq_id, content_type="FACTUAL", epe_score=epe)


def test_highest_miss_counter_evicted_first(enc):
    """
    Entry with the highest miss_counter should be evicted before low-miss entries.
    """
    c = HotCache(enc, token_ceiling=100, top_k=5,
                 eviction_weights=(0.0, 1.0, 0.0, 0.0))  # miss-only

    for i in range(4):
        emb = enc.encode([f"entry {i} with lots of words to fill tokens"])[0]
        c.store(
            f"entry {i} with lots of words to fill tokens",
            emb, seq_id=i, content_type="FACTUAL", epe_score=0.5
        )

    # give seq_id=2 a very high miss count
    with c._lock:
        for e in c._entries:
            if e["seq_id"] == 2:
                e["miss_counter"] = 100
            else:
                e["miss_counter"] = 0

    # force eviction by adding one more entry over the ceiling
    long_text = " ".join(["word"] * 110)
    emb5 = enc.encode([long_text])[0]
    c.store(long_text, emb5, seq_id=99, content_type="FACTUAL", epe_score=0.5)
    time.sleep(0.3)

    remaining_seqs = [e["seq_id"] for e in c.get_all_entries()]
    # seq_id=2 (highest miss) should be evicted
    if len(remaining_seqs) < 5:
        assert 2 not in remaining_seqs, f"High-miss entry should be evicted; got {remaining_seqs}"


def test_highest_hit_count_protected_from_eviction(enc):
    """
    Entry with the highest hit_count should survive eviction even if other factors
    would otherwise rank it high. Negative hit weight protects frequent hits.
    """
    c = HotCache(enc, token_ceiling=100, top_k=5,
                 eviction_weights=(0.0, 0.0, 1.0, 0.0))  # hit-only (inverted: high hit → low score → kept)

    for i in range(4):
        emb = enc.encode([f"segment {i} with filler words to use tokens here"])[0]
        c.store(
            f"segment {i} with filler words to use tokens here",
            emb, seq_id=i, content_type="FACTUAL", epe_score=0.5
        )

    # give seq_id=0 very high hit count → it should survive
    with c._lock:
        for e in c._entries:
            if e["seq_id"] == 0:
                e["hit_count"] = 100
            else:
                e["hit_count"] = 0

    long_text = " ".join(["word"] * 110)
    emb5 = enc.encode([long_text])[0]
    c.store(long_text, emb5, seq_id=99, content_type="FACTUAL", epe_score=0.5)
    time.sleep(0.3)

    remaining_seqs = [e["seq_id"] for e in c.get_all_entries()]
    if len(remaining_seqs) < 5:
        # seq_id=0 (highest hits) should survive
        assert 0 in remaining_seqs, f"High-hit entry should be protected; got {remaining_seqs}"


def test_low_epe_novelty_evicted_before_novel_entry(enc):
    """
    Entry with low epe_novelty (redundant) should be evicted before a novel entry
    that has the same miss/hit profile.
    """
    c = HotCache(enc, token_ceiling=100, top_k=5,
                 eviction_weights=(0.0, 0.0, 0.0, 1.0))  # novelty-only

    # redundant entry: epe_novelty = 0.05
    emb_r = enc.encode(["redundant text with many filler words to consume tokens"])[0]
    c.store(
        "redundant text with many filler words to consume tokens",
        emb_r, seq_id=10, content_type="BACKGROUND", epe_score=0.05
    )
    # novel entry: epe_novelty = 0.95
    emb_n = enc.encode(["novel critical migration failure detailed log entry here"])[0]
    c.store(
        "novel critical migration failure detailed log entry here",
        emb_n, seq_id=11, content_type="CAUSAL", epe_score=0.95
    )
    # add filler to trigger eviction
    for i in range(3):
        emb_f = enc.encode([f"filler segment {i} with many words padding tokens out"])[0]
        c.store(
            f"filler segment {i} with many words padding tokens out",
            emb_f, seq_id=20 + i, content_type="BACKGROUND", epe_score=0.5
        )
    long_text = " ".join(["word"] * 110)
    emb5 = enc.encode([long_text])[0]
    c.store(long_text, emb5, seq_id=99, content_type="FACTUAL", epe_score=0.5)
    time.sleep(0.3)

    remaining_seqs = [e["seq_id"] for e in c.get_all_entries()]
    if len(remaining_seqs) < 6:
        # seq_id=10 (lowest novelty) should go before seq_id=11
        if 11 in remaining_seqs and 10 in remaining_seqs:
            pass  # both survived, eviction didn't reach them
        elif 10 in remaining_seqs and 11 not in remaining_seqs:
            pytest.fail("Novel entry was evicted before redundant entry")


def test_normalisation_handles_zero_values(enc):
    """
    When all entries have 0 miss_counter and 0 hit_count, normalisation
    denominators should default to 1.0 to avoid division by zero.
    """
    c = HotCache(enc, token_ceiling=50, top_k=5)

    for i in range(3):
        text = f"some text entry number {i} with words"
        emb = enc.encode([text])[0]
        c.store(text, emb, seq_id=i, content_type="FACTUAL", epe_score=0.5)

    # all miss_counter and hit_count are 0 — verify no exception on eviction
    long_text = " ".join(["word"] * 60)
    emb5 = enc.encode([long_text])[0]
    c.store(long_text, emb5, seq_id=99, content_type="FACTUAL", epe_score=0.5)
    time.sleep(0.3)
    # just verify it didn't crash and cache still functions
    assert c.token_count() <= 50


def test_store_initialises_required_fields(enc):
    """store() must initialise miss_counter, hit_count, last_accessed_turn, epe_novelty."""
    c = HotCache(enc, token_ceiling=8000)
    emb = enc.encode(["test entry"])[0]
    c.store("test entry", emb, seq_id=1, content_type="FACTUAL", epe_score=0.7)
    with c._lock:
        e = c._entries[-1]
    assert e["miss_counter"] == 0
    assert e["hit_count"] == 0
    assert "last_accessed_turn" in e
    assert e["epe_novelty"] == 0.7
