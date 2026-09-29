from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import spacy as _spacy_mod
    _nlp = _spacy_mod.load("en_core_web_sm")
    _spacy_available = True
except (ImportError, OSError) as _spacy_err:
    _nlp = None
    _spacy_available = False
    _spacy_err_msg = str(_spacy_err)

_NER_TYPES = {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "CARDINAL", "ORDINAL", "QUANTITY"}

_RE_FILE_PATH = re.compile(r"[\w./\-]+\.\w{2,4}")
_RE_ERROR_CODE = re.compile(r"[A-Z][A-Z_]+Error|HTTP \d{3}|\d{3} [A-Z]+")
_RE_STEP_REF = re.compile(r"step\s+\d+|\[step\s+\d+\]", re.IGNORECASE)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS fact_index (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    segment_id INTEGER NOT NULL,
    seq_id INTEGER NOT NULL,
    entity_text TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_value TEXT,
    ref_count INTEGER DEFAULT 1,
    timestamp_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fact_session ON fact_index(session_id);
CREATE INDEX IF NOT EXISTS idx_fact_seq ON fact_index(seq_id, session_id);
CREATE INDEX IF NOT EXISTS idx_fact_entity ON fact_index(entity_text, session_id);
"""


def extract_entities(text: str) -> list[dict]:
    """Extract entities from text using spaCy NER + regex fallbacks."""
    if not _spacy_available:
        raise ImportError(
            "spaCy and en_core_web_sm are required for entity extraction.\n"
            "Install with: pip install spacy && python -m spacy download en_core_web_sm\n"
            f"Original error: {_spacy_err_msg}"
        )
    seen: set[tuple] = set()
    entities: list[dict] = []

    doc = _nlp(text)
    for ent in doc.ents:
        if ent.label_ in _NER_TYPES:
            key = (ent.text, ent.label_)
            if key not in seen:
                seen.add(key)
                entities.append({"text": ent.text, "type": ent.label_, "value": None})

    for m in _RE_FILE_PATH.finditer(text):
        key = (m.group(), "file_path")
        if key not in seen:
            seen.add(key)
            entities.append({"text": m.group(), "type": "file_path", "value": None})

    for m in _RE_ERROR_CODE.finditer(text):
        key = (m.group(), "error_code")
        if key not in seen:
            seen.add(key)
            entities.append({"text": m.group(), "type": "error_code", "value": None})

    for m in _RE_STEP_REF.finditer(text):
        key = (m.group().lower(), "step_reference")
        if key not in seen:
            seen.add(key)
            entities.append({"text": m.group(), "type": "step_reference", "value": None})

    return entities


class FactIndex:
    def __init__(self, db_path: Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def write_entities(
        self,
        entities: list[dict],
        seq_id: int,
        segment_id: int,
        session_id: str,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            for e in entities:
                existing = conn.execute(
                    "SELECT id, ref_count FROM fact_index "
                    "WHERE entity_text = ? AND session_id = ?",
                    (e["text"], session_id),
                ).fetchone()
                if existing:
                    conn.execute(
                        "UPDATE fact_index SET ref_count = ref_count + 1, timestamp_utc = ? "
                        "WHERE id = ?",
                        (ts, existing["id"]),
                    )
                else:
                    conn.execute(
                        "INSERT INTO fact_index "
                        "(session_id, segment_id, seq_id, entity_text, entity_type, entity_value, timestamp_utc) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (session_id, segment_id, seq_id,
                         e["text"], e["type"], e.get("value"), ts),
                    )

    def get_by_seq_id(self, seq_id: int, session_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM fact_index WHERE seq_id = ? AND session_id = ?",
                (seq_id, session_id),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_by_entity(self, entity_text: str, session_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM fact_index WHERE entity_text = ? AND session_id = ?",
                (entity_text, session_id),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_high_reference(self, session_id: str, min_ref_count: int = 3) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM fact_index WHERE session_id = ? AND ref_count >= ? "
                "ORDER BY ref_count DESC",
                (session_id, min_ref_count),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_ref_count(self, entity_text: str, session_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT ref_count FROM fact_index WHERE entity_text = ? AND session_id = ?",
                (entity_text, session_id),
            ).fetchone()
            return row["ref_count"] if row else 0

    def count(self, session_id: Optional[str] = None) -> int:
        with self._connect() as conn:
            if session_id:
                return conn.execute(
                    "SELECT COUNT(*) FROM fact_index WHERE session_id = ?",
                    (session_id,),
                ).fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM fact_index").fetchone()[0]
