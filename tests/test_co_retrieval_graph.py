"""Tests for jeval.memory.co_retrieval_graph.CoRetrievalGraph."""
import threading
import time

import pytest

from jeval.memory.co_retrieval_graph import CoRetrievalGraph, SMOOTHING


@pytest.fixture
def graph(tmp_path):
    return CoRetrievalGraph(tmp_path / "co.db")


# ---------------------------------------------------------------------------
# count and weight
# ---------------------------------------------------------------------------

def test_record_retrieval_increments_count(graph):
    graph.record_retrieval(["a", "b"])
    w = graph.get_edge_weight("a", "b")
    expected = 1 / (1 + SMOOTHING)
    assert abs(w - expected) < 1e-6


def test_record_retrieval_weight_grows_with_count(graph):
    for _ in range(5):
        graph.record_retrieval(["a", "b"])
    w = graph.get_edge_weight("a", "b")
    expected = 5 / (5 + SMOOTHING)
    assert abs(w - expected) < 1e-6


def test_record_retrieval_weight_approaches_one(graph):
    # Use 49 calls — stays below the 50-call decay boundary so background
    # decay cannot interfere with the assertion.
    for _ in range(49):
        graph.record_retrieval(["x", "y"])
    w = graph.get_edge_weight("x", "y")
    expected = 49 / (49 + SMOOTHING)  # ≈ 0.907
    assert abs(w - expected) < 1e-6
    assert w > 0.85


# ---------------------------------------------------------------------------
# ordering: seg_id_a < seg_id_b always
# ---------------------------------------------------------------------------

def test_pairs_stored_with_a_less_than_b(graph):
    # record with b > a order
    graph.record_retrieval(["z", "a"])
    # weight must be accessible regardless of order passed to get_edge_weight
    w1 = graph.get_edge_weight("a", "z")
    w2 = graph.get_edge_weight("z", "a")
    assert w1 == w2
    assert w1 > 0.0

    # directly verify the DB stores canonical order
    import sqlite3
    conn = sqlite3.connect(str(graph._db_path))
    row = conn.execute(
        "SELECT seg_id_a, seg_id_b FROM co_retrieval_edges"
    ).fetchone()
    conn.close()
    assert row[0] < row[1]


def test_multi_segment_retrieval_generates_all_pairs(graph):
    graph.record_retrieval(["a", "b", "c"])
    assert graph.get_edge_weight("a", "b") > 0.0
    assert graph.get_edge_weight("a", "c") > 0.0
    assert graph.get_edge_weight("b", "c") > 0.0


# ---------------------------------------------------------------------------
# get_neighbors
# ---------------------------------------------------------------------------

def test_get_neighbors_top_k_and_min_weight(graph):
    # build edges: a-b strong, a-c medium, a-d weak
    for _ in range(20):
        graph.record_retrieval(["a", "b"])
    for _ in range(10):
        graph.record_retrieval(["a", "c"])
    for _ in range(2):
        graph.record_retrieval(["a", "d"])

    # min_weight=0.3 should filter out d (weight ≈ 2/7 ≈ 0.286)
    neighbors = graph.get_neighbors("a", min_weight=0.3, top_k=5)
    neighbor_ids = [n[0] for n in neighbors]
    assert "b" in neighbor_ids
    assert "c" in neighbor_ids
    assert "d" not in neighbor_ids

    # sorted descending by weight
    weights = [n[1] for n in neighbors]
    assert weights == sorted(weights, reverse=True)


def test_get_neighbors_respects_top_k(graph):
    for seg in ["b", "c", "d", "e", "f", "g"]:
        for _ in range(10):
            graph.record_retrieval(["a", seg])
    neighbors = graph.get_neighbors("a", min_weight=0.0, top_k=3)
    assert len(neighbors) <= 3


def test_get_neighbors_unknown_seg_returns_empty(graph):
    assert graph.get_neighbors("nonexistent") == []


# ---------------------------------------------------------------------------
# decay
# ---------------------------------------------------------------------------

def test_decay_reduces_weights(graph):
    for _ in range(10):
        graph.record_retrieval(["a", "b"])
    before = graph.get_edge_weight("a", "b")
    graph.decay()
    after = graph.get_edge_weight("a", "b")
    assert after < before
    assert abs(after - before * 0.95) < 1e-6


def test_decay_deletes_below_threshold(graph):
    # single co-retrieval → weight ≈ 0.167; decay enough times to drop below 0.05
    graph.record_retrieval(["a", "b"])
    for _ in range(25):   # 0.167 * 0.95^25 ≈ 0.047 < 0.05
        graph.decay()
    w = graph.get_edge_weight("a", "b")
    assert w == 0.0


def test_decay_preserves_strong_edges(graph):
    for _ in range(50):
        graph.record_retrieval(["a", "b"])
    graph.decay()
    assert graph.get_edge_weight("a", "b") > 0.05


# ---------------------------------------------------------------------------
# concurrency
# ---------------------------------------------------------------------------

def test_concurrent_writes_no_lost_rows(tmp_path):
    g = CoRetrievalGraph(tmp_path / "concurrent.db")
    errors: list[Exception] = []

    def worker(thread_id: int) -> None:
        try:
            for i in range(20):
                g.record_retrieval([f"seg_{thread_id}", f"seg_{(thread_id + 1) % 5}"])
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent write errors: {errors}"

    # All 5 unique pairs should be present
    import sqlite3
    conn = sqlite3.connect(str(g._db_path))
    count = conn.execute("SELECT COUNT(*) FROM co_retrieval_edges").fetchone()[0]
    conn.close()
    assert count >= 5
