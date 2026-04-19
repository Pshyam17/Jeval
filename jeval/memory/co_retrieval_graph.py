from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

SMOOTHING = 2.0
_DECAY_FACTOR = 0.95
_DECAY_INTERVAL = 50   # calls between automatic decay runs
_MIN_WEIGHT = 0.05     # edges below this are pruned on decay
_COLD_HIT_INCREMENT = 3  # Cold hits get 3x increment for stronger edges (§1.12)


class CoRetrievalGraph:
    """
    Weighted co-retrieval graph over hot-cache segments backed by SQLite.

    Every time a retrieval returns a result set, edges between all pairs of
    returned segments are incremented.  Weights are normalised to (0, 1) via
    a smoothed count formula and decay over time so stale associations fade.

    v3.0: Supports frozen_mode to disable updates for fair benchmarking (§1.12).
    """

    def __init__(
        self,
        db_path: Path,
        frozen_mode: bool = False,
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._call_count = 0
        self._frozen_mode = frozen_mode
        self._conn = self._connect()
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS co_retrieval_edges (
                    seg_id_a     TEXT NOT NULL,
                    seg_id_b     TEXT NOT NULL,
                    weight       REAL NOT NULL DEFAULT 0.0,
                    count        INTEGER NOT NULL DEFAULT 0,
                    last_updated REAL NOT NULL,
                    PRIMARY KEY (seg_id_a, seg_id_b)
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_seg_a ON co_retrieval_edges(seg_id_a)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_seg_b ON co_retrieval_edges(seg_id_b)"
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_retrieval(
        self,
        seg_ids: list[str],
        was_cold_hit: bool = False,
    ) -> None:
        """
        Increment edge counts for every pair in *seg_ids*.

        If was_cold_hit=True, strengthen edges more (η_hit = 0.15 vs base = 0.05).

        Cold Storage Feedback Loop (§1.12):
        On cold_storage hit for query Q returning segment s:
        1. Graph Edge Strengthening:
           For each hot entry h that was also retrieved (co-retrieval):
             w(h, s) ← w(h, s) + η_hit
           where η_hit = 0.15

        2. Promotion Tracking:
           Track cold_hit_count[s] += 1
           If cold_hit_count[s] ≥ 3 AND s was accessed in last 50 turns:
             # Candidate for hot promotion
             enqueue s for recompression and hot-cache insertion

        v3.0: In frozen_mode, skip all updates for fair benchmarking.
        """
        if self._frozen_mode:
            return  # No updates in frozen mode
        if len(seg_ids) < 2:
            return

        now = time.time()
        pairs: list[tuple[str, str]] = []
        for i in range(len(seg_ids)):
            for j in range(i + 1, len(seg_ids)):
                a, b = seg_ids[i], seg_ids[j]
                if a > b:
                    a, b = b, a
                pairs.append((a, b))

        # Cold hits get stronger edge weights (η_hit = 0.15 vs base increment)
        # Base increment is ~0.05 (1/(1+SMOOTHING)), cold hits get ~0.15
        increment = 3 if was_cold_hit else 1

        with self._lock:
            for a, b in pairs:
                self._conn.execute(
                    """
                    INSERT INTO co_retrieval_edges (seg_id_a, seg_id_b, weight, count, last_updated)
                    VALUES (?, ?, 0.0, ?, ?)
                    ON CONFLICT(seg_id_a, seg_id_b) DO UPDATE SET
                        count        = count + ?,
                        last_updated = excluded.last_updated
                    """,
                    (a, b, increment, now, increment),
                )
            # Recompute weight from count for every updated pair
            for a, b in pairs:
                row = self._conn.execute(
                    "SELECT count FROM co_retrieval_edges WHERE seg_id_a=? AND seg_id_b=?",
                    (a, b),
                ).fetchone()
                if row:
                    count = row[0]
                    weight = count / (count + SMOOTHING)
                    self._conn.execute(
                        "UPDATE co_retrieval_edges SET weight=? WHERE seg_id_a=? AND seg_id_b=?",
                        (weight, a, b),
                    )
            self._conn.commit()
            self._call_count += 1
            should_decay = self._call_count % _DECAY_INTERVAL == 0

        if should_decay:
            threading.Thread(target=self.decay, daemon=True).start()

    def decay(self, half_life_queries: int = 200) -> None:
        """Multiply all weights by 0.95; prune edges below *_MIN_WEIGHT*."""
        with self._lock:
            self._conn.execute(
                f"UPDATE co_retrieval_edges SET weight = weight * {_DECAY_FACTOR}"
            )
            self._conn.execute(
                f"DELETE FROM co_retrieval_edges WHERE weight < {_MIN_WEIGHT}"
            )
            self._conn.commit()

    def get_neighbors(
        self,
        seg_id: str,
        min_weight: float = 0.3,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Return [(neighbor_seg_id, weight)] for *seg_id*, sorted by weight
        descending, filtered by *min_weight*, limited to *top_k*.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    CASE WHEN seg_id_a = ? THEN seg_id_b ELSE seg_id_a END AS neighbor,
                    weight
                FROM co_retrieval_edges
                WHERE (seg_id_a = ? OR seg_id_b = ?)
                  AND weight >= ?
                ORDER BY weight DESC
                LIMIT ?
                """,
                (seg_id, seg_id, seg_id, min_weight, top_k),
            ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def get_edge_weight(self, seg_id_a: str, seg_id_b: str) -> float:
        """Return edge weight for this pair, 0.0 if not found."""
        a, b = (seg_id_a, seg_id_b) if seg_id_a < seg_id_b else (seg_id_b, seg_id_a)
        with self._lock:
            row = self._conn.execute(
                "SELECT weight FROM co_retrieval_edges WHERE seg_id_a=? AND seg_id_b=?",
                (a, b),
            ).fetchone()
        return float(row[0]) if row else 0.0
