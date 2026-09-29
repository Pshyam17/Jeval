"""Tests for jeval.memory.bm25_index.BM25Index."""
import pytest

from jeval.memory.bm25_index import BM25Index


@pytest.fixture
def idx():
    return BM25Index()


# ---------------------------------------------------------------------------
# Basic add + search
# ---------------------------------------------------------------------------

def test_add_and_search_basic(idx):
    idx.add("seg1", "step 68 migration failed on staging environment")
    results = idx.search("step 68")
    assert len(results) == 1
    assert results[0][0] == "seg1"
    assert results[0][1] > 0.0


def test_search_empty_index_returns_empty(idx):
    assert idx.search("anything") == []


def test_scores_are_positive_floats(idx):
    idx.add("s1", "migration lock timeout after thirty seconds")
    results = idx.search("migration timeout")
    assert all(isinstance(score, float) and score > 0.0 for _, score in results)


# ---------------------------------------------------------------------------
# Exact step reference ranking
# ---------------------------------------------------------------------------

def test_exact_step_reference_ranked_above_other(idx):
    idx.add("seg68", "step 68 migration failed due to lock timeout")
    idx.add("seg72", "step 72 rollback completed successfully without errors")
    results = idx.search("step 68")
    ids = [r[0] for r in results]
    assert ids[0] == "seg68", f"Expected seg68 first, got {ids}"


def test_unrelated_document_not_in_results(idx):
    # A document with zero token overlap should not appear in results.
    idx.add("seg72", "deployment succeeded production release")
    results = idx.search("step 68 migration failed")
    ids = [r[0] for r in results]
    assert "seg72" not in ids


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------

def test_remove_removes_segment(idx):
    idx.add("seg1", "migration failed on staging")
    idx.add("seg2", "deployment succeeded in production")
    idx.remove("seg1")
    results = idx.search("migration staging")
    ids = [r[0] for r in results]
    assert "seg1" not in ids


def test_remove_nonexistent_is_noop(idx):
    idx.add("seg1", "some text here")
    idx.remove("nonexistent")  # must not raise
    results = idx.search("some text")
    assert len(results) == 1


def test_remove_updates_stats(idx):
    idx.add("s1", "hello world")
    idx.add("s2", "hello world again")
    idx.remove("s1")
    assert idx._n_docs == 1
    assert "s1" not in idx._tf


# ---------------------------------------------------------------------------
# Multi-token query
# ---------------------------------------------------------------------------

def test_multi_token_scores_higher_for_more_matching(idx):
    idx.add("partial", "migration failed")
    idx.add("full",    "migration failed lock timeout staging environment")
    results_dict = dict(idx.search("migration failed lock timeout"))
    # "full" matches more query tokens → higher score
    assert "full" in results_dict and "partial" in results_dict
    assert results_dict["full"] > results_dict["partial"]


def test_top_k_limits_results(idx):
    for i in range(20):
        idx.add(f"seg{i}", f"migration step {i} failed with timeout error")
    results = idx.search("migration failed timeout", top_k=5)
    assert len(results) <= 5


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------

def test_clear_resets_state(idx):
    idx.add("s1", "some content here")
    idx.clear()
    assert idx._n_docs == 0
    assert idx.search("content") == []

