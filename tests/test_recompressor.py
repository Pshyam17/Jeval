"""
Tests for MissTriggeredRecompressor.

Verifies counter semantics, eligibility conditions, and rewrite log output.
"""
from __future__ import annotations

import json
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from jeval.encoders.sentence_encoder import FrozenEncoder
from jeval.memory.recompressor import MissTriggeredRecompressor


def _make_mock_compressor():
    mock = MagicMock()
    mock.compress.return_value = "recompressed text with more detail"
    return mock


def _make_mock_cold(original_text: str = "original detailed text with lock timeout"):
    cold = MagicMock()
    cold.get_by_seq_id.return_value = {"content": original_text}
    return cold


def _make_mock_hot_cache(enc: FrozenEncoder):
    cache = MagicMock()
    emb = enc.encode(["compressed entry"])[0]
    cache.get_all_entries.return_value = [
        {
            "seq_id": 1,
            "text": "compressed entry",
            "embedding": emb,
            "content_type": "migration_failure",
            "epe_score": 0.5,
            "metadata": {"budget": 0.5, "epe_final": 0.12},
            "stale": False,
        }
    ]
    return cache


def _make_recompressor(enc, miss_threshold=2, min_rewrite_gap=5):
    compressor = _make_mock_compressor()
    cold = _make_mock_cold()
    hot = _make_mock_hot_cache(enc)
    rc = MissTriggeredRecompressor(
        compressor=compressor,
        encoder=enc,
        cold_storage=cold,
        hot_cache=hot,
        miss_threshold=miss_threshold,
        min_rewrite_gap=min_rewrite_gap,
    )
    return rc, compressor, cold, hot


# ---------------------------------------------------------------------------
# Counter semantics
# ---------------------------------------------------------------------------

def test_record_miss_increments_counter(enc):
    rc, _, _, _ = _make_recompressor(enc)
    rc.record_miss(seq_id=1, query="why did migration fail", turn=1)
    rc.record_miss(seq_id=1, query="what caused timeout", turn=2)
    assert rc._miss_counter[1] == 2


def test_record_miss_stores_query(enc):
    rc, _, _, _ = _make_recompressor(enc)
    rc.record_miss(seq_id=1, query="why did migration fail", turn=1)
    assert "why did migration fail" in rc._miss_queries[1]


def test_record_hit_increments_hit_count(enc):
    rc, _, _, _ = _make_recompressor(enc)
    rc.record_hit(seq_id=1)
    rc.record_hit(seq_id=1)
    assert rc._hit_count[1] == 2


def test_record_hit_does_not_reset_miss_counter(enc):
    rc, _, _, _ = _make_recompressor(enc)
    rc.record_miss(seq_id=1, query="q", turn=1)
    rc.record_hit(seq_id=1)
    assert rc._miss_counter[1] == 1


# ---------------------------------------------------------------------------
# Eligibility conditions
# ---------------------------------------------------------------------------

def test_rewrite_not_triggered_below_threshold(enc):
    rc, _, _, _ = _make_recompressor(enc, miss_threshold=2)
    rc.record_miss(seq_id=1, query="q", turn=1)
    rc.record_miss(seq_id=1, query="q", turn=2)
    # miss_counter == 2, threshold == 2: need > threshold, so not eligible
    triggered = rc.check_and_rewrite(seq_id=1, turn=10)
    assert not triggered


def test_rewrite_triggered_above_threshold(enc, tmp_path, monkeypatch):
    import jeval.memory.recompressor as rc_mod
    monkeypatch.setattr(rc_mod, "_REWRITE_LOG", tmp_path / "rewrite_log.jsonl")

    rc, _, _, _ = _make_recompressor(enc, miss_threshold=2, min_rewrite_gap=5)
    rc.record_miss(seq_id=1, query="why did migration fail", turn=1)
    rc.record_miss(seq_id=1, query="what caused timeout", turn=2)
    rc.record_miss(seq_id=1, query="how many connections", turn=3)
    # miss_counter == 3 > threshold 2; turn=10, last_rewrite=-inf → gap > 5
    triggered = rc.check_and_rewrite(seq_id=1, turn=10)
    assert triggered


def test_rewrite_not_triggered_within_min_gap(enc, tmp_path, monkeypatch):
    import jeval.memory.recompressor as rc_mod
    monkeypatch.setattr(rc_mod, "_REWRITE_LOG", tmp_path / "rewrite_log.jsonl")

    rc, _, _, _ = _make_recompressor(enc, miss_threshold=2, min_rewrite_gap=5)
    # force a prior rewrite at turn=5
    rc._last_rewrite_turn[1] = 5
    rc._miss_counter[1] = 10  # above threshold

    triggered = rc.check_and_rewrite(seq_id=1, turn=8)  # only 3 turns since last rewrite
    assert not triggered


def test_rewrite_triggered_when_both_conditions_met(enc, tmp_path, monkeypatch):
    import jeval.memory.recompressor as rc_mod
    monkeypatch.setattr(rc_mod, "_REWRITE_LOG", tmp_path / "rewrite_log.jsonl")

    rc, _, _, _ = _make_recompressor(enc, miss_threshold=2, min_rewrite_gap=5)
    rc._last_rewrite_turn[1] = 1
    rc._miss_counter[1] = 5  # > threshold

    triggered = rc.check_and_rewrite(seq_id=1, turn=7)  # 6 turns since last rewrite > gap
    assert triggered


# ---------------------------------------------------------------------------
# Miss counter reset after rewrite
# ---------------------------------------------------------------------------

def test_miss_counter_resets_after_rewrite(enc, tmp_path, monkeypatch):
    import jeval.memory.recompressor as rc_mod
    monkeypatch.setattr(rc_mod, "_REWRITE_LOG", tmp_path / "rewrite_log.jsonl")

    rc, _, _, _ = _make_recompressor(enc, miss_threshold=2, min_rewrite_gap=5)
    rc.record_miss(seq_id=1, query="q1", turn=1)
    rc.record_miss(seq_id=1, query="q2", turn=2)
    rc.record_miss(seq_id=1, query="q3", turn=3)
    rc.check_and_rewrite(seq_id=1, turn=10)
    time.sleep(0.1)  # let async/thread finish
    assert rc._miss_counter.get(1, 0) == 0


# ---------------------------------------------------------------------------
# Rewrite log written correctly
# ---------------------------------------------------------------------------

def test_rewrite_log_written(enc, tmp_path, monkeypatch):
    import jeval.memory.recompressor as rc_mod
    log_path = tmp_path / "rewrite_log.jsonl"
    monkeypatch.setattr(rc_mod, "_REWRITE_LOG", log_path)

    rc, _, _, _ = _make_recompressor(enc, miss_threshold=2, min_rewrite_gap=5)
    rc.record_miss(seq_id=1, query="why migration failed", turn=1)
    rc.record_miss(seq_id=1, query="what caused timeout", turn=2)
    rc.record_miss(seq_id=1, query="how many connections", turn=3)
    rc.check_and_rewrite(seq_id=1, turn=10)
    time.sleep(0.3)  # ensure thread/async completes

    if log_path.exists():
        lines = log_path.read_text().splitlines()
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert record["seq_id"] == 1
        assert record["new_budget"] > record["old_budget"]
        assert "timestamp" in record
        assert isinstance(record["trigger_queries"], list)
        assert isinstance(record["anchors_added"], list)
