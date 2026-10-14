from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class ColdStorageWriteError(Exception):
    pass


_SCHEMA = """
CREATE TABLE IF NOT EXISTS cold_storage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seq_id INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    token_count INTEGER NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS cold_fts USING fts5(
    content,
    content='cold_storage',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS cold_fts_insert
    AFTER INSERT ON cold_storage BEGIN
    INSERT INTO cold_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


class ColdStorage:
    def __init__(self, db_path: Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def append(self, content: str, seq_id: int, session_id: str) -> int:
        token_count = len(content.split())
        ts = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                with self._connect() as conn:
                    cur = conn.execute(
                        "INSERT INTO cold_storage (seq_id, session_id, content, timestamp_utc, token_count) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (seq_id, session_id, content, ts, token_count),
                    )
                    return cur.lastrowid
            except sqlite3.Error as exc:
                raise ColdStorageWriteError(f"cold storage write failed: {exc}") from exc

    def search(self, query: str, limit: int = 5) -> list[dict]:
        # Build an OR query from individual terms; FTS5 handles tokenisation.
        # Underscores and special chars in identifiers need LIKE fallback.
        terms = [t for t in query.split() if t]
        fts_query = " OR ".join(terms) if terms else query
        sql = (
            'SELECT cs.id, cs.seq_id, cs.content, cs.timestamp_utc '
            'FROM cold_fts '
            'JOIN cold_storage cs ON cold_fts.rowid = cs.id '
            'WHERE cold_fts MATCH ? '
            'ORDER BY rank LIMIT ?'
        )
        try:
            with self._connect() as conn:
                rows = conn.execute(sql, (fts_query, limit)).fetchall()
                if rows:
                    return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            pass
        # LIKE fallback for identifiers that FTS tokeniser splits incorrectly
        conditions = " OR ".join("content LIKE ?" for _ in terms) if terms else "content LIKE ?"
        params = [f"%{t}%" for t in terms] if terms else [f"%{query}%"]
        with self._connect() as conn:
            rows = conn.execute(
                f'SELECT id, seq_id, content, timestamp_utc FROM cold_storage '
                f'WHERE {conditions} LIMIT ?',
                params + [limit],
            ).fetchall()
            return [dict(r) for r in rows]

    def get_by_seq_id(self, seq_id: int, session_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, seq_id, session_id, content, timestamp_utc, token_count "
                "FROM cold_storage WHERE seq_id = ? AND session_id = ?",
                (seq_id, session_id),
            ).fetchone()
            return dict(row) if row else None

    def get_by_id(self, row_id: int) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, seq_id, session_id, content, timestamp_utc, token_count "
                "FROM cold_storage WHERE id = ?",
                (row_id,),
            ).fetchone()
            return dict(row) if row else None

    def count(self, session_id: Optional[str] = None) -> int:
        with self._connect() as conn:
            if session_id:
                return conn.execute(
                    "SELECT COUNT(*) FROM cold_storage WHERE session_id = ?",
                    (session_id,),
                ).fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM cold_storage").fetchone()[0]
