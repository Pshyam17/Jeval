import threading
import pytest
from pathlib import Path
from jeval.memory.cold_storage import ColdStorage, ColdStorageWriteError


@pytest.fixture
def db(tmp_path):
    return ColdStorage(tmp_path / "cold.db")


def test_append_and_retrieve(db):
    row_id = db.append("migration failed on staging", seq_id=1, session_id="s1")
    assert isinstance(row_id, int) and row_id > 0
    row = db.get_by_id(row_id)
    assert row is not None
    assert row["content"] == "migration failed on staging"
    assert row["seq_id"] == 1
    assert row["token_count"] == 4


def test_get_by_seq_id(db):
    db.append("step 7 action: deployed to staging", seq_id=7, session_id="s1")
    row = db.get_by_seq_id(7, "s1")
    assert row is not None
    assert "deployed" in row["content"]


def test_get_by_seq_id_wrong_session(db):
    db.append("some text", seq_id=1, session_id="s1")
    assert db.get_by_seq_id(1, "s2") is None


def test_fts_search_returns_relevant(db):
    db.append("JWT_SECRET mismatch in src/config/env.ts caused auth failure", seq_id=1, session_id="s1")
    db.append("weather is nice today outside", seq_id=2, session_id="s1")
    results = db.search("JWT_SECRET auth", limit=5)
    assert len(results) >= 1
    assert any("JWT_SECRET" in r["content"] for r in results)


def test_fts_search_no_match(db):
    db.append("migration failed due to lock timeout", seq_id=1, session_id="s1")
    results = db.search("xylophone saxophone oboe", limit=5)
    assert results == []


def test_concurrent_appends_no_corruption(db):
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            db.append(f"concurrent entry {i}", seq_id=i, session_id="s1")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"concurrent write errors: {errors}"
    assert db.count("s1") == 10
