from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from jeval.encoders.sentence_encoder import FrozenEncoder
from jeval.memory.cold_storage import ColdStorage
from jeval.memory.hot_cache import HotCache

logger = logging.getLogger(__name__)

_REWRITE_LOG = Path("benchmarks/results/rewrite_log.jsonl")


class MissTriggeredRecompressor:
    """
    Tracks retrieval misses per seq_id and schedules budget-bumped recompression
    when a compressed entry repeatedly fails to satisfy queries.

    The hot cache self-improvement loop: cold-storage misses are the signal that
    compression discarded information the user actually needed; rewrites address
    this by increasing budget and adding miss-query entities as anchors.
    """

    def __init__(
        self,
        compressor,             # TimeoutCompressor instance
        encoder: FrozenEncoder,
        cold_storage: ColdStorage,
        hot_cache: HotCache,
        miss_threshold: int = 2,
        min_rewrite_gap: int = 5,   # turns
    ) -> None:
        self._compressor = compressor
        self._encoder = encoder
        self._cold = cold_storage
        self._hot_cache = hot_cache
        self._miss_threshold = miss_threshold
        self._min_rewrite_gap = min_rewrite_gap

        # per seq_id counters
        self._miss_counter: dict[int, int] = {}
        self._hit_count: dict[int, int] = {}
        self._last_rewrite_turn: dict[int, int] = {}
        self._miss_queries: dict[int, list[str]] = {}

    def record_miss(self, seq_id: int, query: str, turn: int) -> None:
        self._miss_counter[seq_id] = self._miss_counter.get(seq_id, 0) + 1
        if seq_id not in self._miss_queries:
            self._miss_queries[seq_id] = []
        self._miss_queries[seq_id].append(query)

    def record_hit(self, seq_id: int) -> None:
        # hit_count persists; does not reset miss_counter — misses are sticky
        self._hit_count[seq_id] = self._hit_count.get(seq_id, 0) + 1

    def check_and_rewrite(self, seq_id: int, turn: int) -> bool:
        """
        Returns True if a rewrite was scheduled, False otherwise.

        Eligibility requires both:
        - miss_counter[seq_id] > miss_threshold
        - turns since last rewrite > min_rewrite_gap
        """
        miss_count = self._miss_counter.get(seq_id, 0)
        last_rewrite = self._last_rewrite_turn.get(seq_id, -self._min_rewrite_gap - 1)

        if miss_count <= self._miss_threshold:
            return False
        if (turn - last_rewrite) <= self._min_rewrite_gap:
            return False

        # reset miss counter and record rewrite turn before scheduling
        # to prevent duplicate rewrites from concurrent check calls
        self._miss_counter[seq_id] = 0
        self._last_rewrite_turn[seq_id] = turn

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._rewrite(seq_id))
            else:
                loop.run_until_complete(self._rewrite(seq_id))
        except RuntimeError:
            # no event loop — fall back to synchronous rewrite
            import threading
            threading.Thread(
                target=self._rewrite_sync, args=(seq_id,), daemon=True
            ).start()

        return True

    async def _rewrite(self, seq_id: int) -> None:
        self._rewrite_sync(seq_id)

    def _rewrite_sync(self, seq_id: int) -> None:
        """
        Synchronous rewrite body. Called from both async and threaded contexts.

        Fetches the original from cold storage, bumps budget by 0.2, recompresses
        with miss-query entities added as anchors, then swaps the hot cache entry.
        """
        # find the hot cache entry to get current budget and EPE
        entries = self._hot_cache.get_all_entries()
        old_entry = next((e for e in entries if e["seq_id"] == seq_id), None)

        if old_entry is None:
            logger.debug("recompressor: seq_id %d not in hot cache, skipping", seq_id)
            return

        old_budget = old_entry.get("metadata", {}).get("budget", 0.5)
        old_epe_final = old_entry.get("metadata", {}).get("epe_final", 0.0)
        new_budget = min(old_budget + 0.2, 1.0)

        # cold storage fetch — session_id not tracked here; search by seq_id broadly
        original = None
        for session_id_guess in [None, ""]:
            try:
                original = self._cold.get_by_seq_id(seq_id, session_id_guess or "")
                if original:
                    break
            except Exception:
                pass

        if original is None:
            # fall back to seq_id search without session filter
            try:
                with self._cold._connect() as conn:
                    row = conn.execute(
                        "SELECT * FROM cold_storage WHERE seq_id = ?", (seq_id,)
                    ).fetchone()
                    if row:
                        original = dict(row)
            except Exception:
                pass

        if original is None:
            logger.debug(
                "recompressor: seq_id %d not in cold storage, skipping", seq_id
            )
            return

        original_text = original.get("content", "")

        # build anchor list from miss-query entities
        trigger_queries = self._miss_queries.get(seq_id, [])
        anchors_added: list[str] = []
        for q in trigger_queries:
            words = [w.lower() for w in q.split() if len(w) >= 3]
            anchors_added.extend(words)
        anchors_added = list(dict.fromkeys(anchors_added))  # deduplicate, preserve order

        try:
            new_compressed = self._compressor.compress(original_text, new_budget, anchors_added)
        except Exception as exc:
            logger.warning("recompressor: compress failed for seq_id %d: %s", seq_id, exc)
            return

        new_emb = self._encoder.encode([new_compressed])[0]
        orig_emb = self._encoder.encode([original_text])[0]
        import numpy as np
        new_cosine_epe = float(1.0 - np.dot(orig_emb, new_emb))
        new_epe_final = new_cosine_epe  # schema gap not recomputed on rewrite path

        # evict old entry and insert updated one
        self._hot_cache.mark_stale(seq_id)

        new_metadata = dict(old_entry.get("metadata", {}))
        new_metadata.update({
            "budget": new_budget,
            "epe_final": new_epe_final,
            "rewritten": True,
        })

        self._hot_cache.store(
            text=new_compressed,
            embedding=new_emb,
            seq_id=seq_id,
            content_type=old_entry.get("content_type", "BACKGROUND"),
            epe_score=old_entry.get("epe_score", 0.5),
            metadata=new_metadata,
        )

        self._log_rewrite(
            seq_id=seq_id,
            old_budget=old_budget,
            new_budget=new_budget,
            old_epe_final=old_epe_final,
            new_epe_final=new_epe_final,
            trigger_queries=trigger_queries,
            anchors_added=anchors_added,
        )

    def _log_rewrite(
        self,
        seq_id: int,
        old_budget: float,
        new_budget: float,
        old_epe_final: float,
        new_epe_final: float,
        trigger_queries: list[str],
        anchors_added: list[str],
    ) -> None:
        _REWRITE_LOG.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seq_id": seq_id,
            "old_budget": old_budget,
            "new_budget": new_budget,
            "old_epe_final": old_epe_final,
            "new_epe_final": new_epe_final,
            "miss_count": self._miss_threshold + 1,  # threshold was exceeded
            "trigger_queries": trigger_queries,
            "anchors_added": anchors_added,
        }
        try:
            with open(_REWRITE_LOG, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.warning("recompressor: failed to write rewrite log: %s", exc)
