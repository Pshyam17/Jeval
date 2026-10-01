from __future__ import annotations

import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import AsyncGenerator

import numpy as np

from demo.events import (
    CompressionCompleteEvent,
    CompressionStartEvent,
    CompressionTokenEvent,
    ContradictionEvent,
    SegmentIngestEvent,
    SessionStatsEvent,
)
from demo.streaming_compressor import StreamingCompressor
from demo.umap_projector import UMAPProjector
from jeval.encoders.sentence_encoder import FrozenEncoder
from jeval.memory.jeval_memory import JevalMemory

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "and", "or", "but", "if", "as", "it", "its", "this", "that",
    "these", "those", "after", "before", "during", "via",
}

_PUNCT_STRIP = re.compile(r"[^\w/.\-]")

_executor = ThreadPoolExecutor(max_workers=4)


async def _compress_async(
    compressor: StreamingCompressor,
    text: str,
    budget: float,
    anchors: list[str],
    timeout: float = 8.0,
) -> list[str]:
    """Run compress_full in a thread with a hard timeout.

    Returns a list of tokens (split on spaces for streaming simulation).
    Raises TimeoutError if the timeout is exceeded.
    """
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                _executor,
                lambda: compressor.compress_full(text, budget, anchors),
            ),
            timeout=timeout,
        )
        return result.split()
    except asyncio.TimeoutError:
        raise TimeoutError(f"compression timed out after {timeout}s")


class AgentBridge:
    def __init__(
        self,
        memory: JevalMemory,
        encoder: FrozenEncoder,
        projector: UMAPProjector,
        compressor: StreamingCompressor,
        replay_path: Path,
        replay_speed: float = 1.0,
    ):
        self._memory = memory
        self._encoder = encoder
        self._projector = projector
        self._compressor = compressor
        self._replay_path = Path(replay_path)
        self._replay_speed = replay_speed
        # snapshot of stale seq_ids before each ingest for delta detection
        self._known_stale: set[int] = set()
        self._seq_counter = 0  # local counter for timeout fallback seq_ids

    async def run(self) -> AsyncGenerator[object, None]:
        await asyncio.sleep(3.0)  # allow browser to connect before events start
        entries = self._load_replay()

        for entry in entries:
            delay = entry["delay_seconds"] / self._replay_speed
            await asyncio.sleep(delay)

            text = entry["text"]

            # snapshot stale set before ingest so we can diff afterward
            pre_stale = self._snapshot_stale()

            # run ingest in executor — it calls NIM synchronously, which
            # blocks the event loop and prevents WebSocket connection handling
            loop = asyncio.get_event_loop()
            self._seq_counter += 1
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        _executor,
                        lambda t=text: self._memory.ingest(t),
                    ),
                    timeout=45.0,
                )
            except asyncio.TimeoutError:
                print(f"ingest timed out for seq={self._seq_counter} — skipping", flush=True)
                coords = [0.0, 0.0, 0.0]
                yield SegmentIngestEvent(
                    seq_id=self._seq_counter,
                    action="timeout",
                    content_type="unknown",
                    epe_novelty=0.5,
                    budget=0.0,
                    anchors=[],
                    token_count_original=len(text.split()),
                    token_count_compressed=0,
                    original_tokens=[],
                    coords_3d=coords,
                )
                continue
            seq_id: int = result["seq_id"]
            action: str = result["action"]
            epe_novelty: float = result.get("epe_novelty") or 0.0
            budget: float = result.get("budget") or 0.0
            anchors: list[str] = result.get("anchors") or []
            content_type: str = result.get("content_type") or "BACKGROUND"

            # reuse the compressed_text from ingest() — avoids a second NIM call
            compressed_text = result.get("compressed_text") or ""
            epe_fidelity = 0.0

            if action != "cold_only":
                yield CompressionStartEvent(
                    seq_id=seq_id,
                    budget=budget,
                    budget_pct=int(budget * 100),
                    content_type=content_type,
                    attempt=1,
                )

                anchor_set = set(anchors)
                tokens = compressed_text.split() if compressed_text else []
                if tokens:
                    for token in tokens:
                        is_anchor = token.strip(".,;:!?") in anchor_set
                        yield CompressionTokenEvent(
                            seq_id=seq_id,
                            token=token + " ",
                            is_anchor=is_anchor,
                        )
                        await asyncio.sleep(0.04)  # 40ms between tokens for visual effect
                else:
                    # fallback: use extractive if ingest didn't return compressed text
                    words = text.split()
                    compressed_text = " ".join(
                        words[:max(1, int(len(words) * budget))]
                    )
                    yield CompressionTokenEvent(
                        seq_id=seq_id,
                        token=compressed_text,
                        is_anchor=False,
                    )

                epe_fidelity = result.get("epe_fidelity") or 0.0

                yield CompressionCompleteEvent(
                    seq_id=seq_id,
                    epe_fidelity=epe_fidelity,
                    passed_gate=epe_fidelity <= self._memory._fidelity_threshold,
                    attempt=1,
                    compressed_text=compressed_text,
                )

            # project to 3D using the original text embedding
            # run encoder + UMAP in executor — numpy operations can take 100-500ms
            emb = await loop.run_in_executor(
                _executor,
                lambda t=text: self._encoder.encode([t])[0],
            )
            coords_3d = await loop.run_in_executor(
                _executor,
                lambda e=emb: self._projector.transform(e),
            )

            original_tokens = self._tokenise_with_roles(text, anchors)

            yield SegmentIngestEvent(
                seq_id=seq_id,
                action=action,
                content_type=content_type,
                epe_novelty=epe_novelty,
                budget=budget,
                anchors=anchors,
                token_count_original=result.get("token_count_original") or len(text.split()),
                token_count_compressed=result.get("token_count_compressed") or len(compressed_text.split()),
                original_tokens=original_tokens,
                coords_3d=coords_3d,
            )

            # wait briefly for async contradiction detector before checking
            await asyncio.sleep(0.15)

            post_stale = self._snapshot_stale()
            new_stale = post_stale - pre_stale
            for stale_seq_id in new_stale:
                epe_between = self._estimate_epe_between(seq_id, stale_seq_id)
                yield ContradictionEvent(
                    new_seq_id=seq_id,
                    stale_seq_id=stale_seq_id,
                    epe_between=epe_between,
                )

            stats = self._memory.stats()
            yield SessionStatsEvent(
                hot_cache_size=stats["hot_cache_size"],
                hot_cache_tokens=stats["hot_cache_tokens"],
                cold_storage_size=stats["cold_storage_size"],
                fact_index_size=stats["fact_index_size"],
                session_id=stats["session_id"],
            )

    def _load_replay(self) -> list[dict]:
        entries = []
        with self._replay_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def _snapshot_stale(self) -> set[int]:
        hot_cache = self._memory._hot_cache
        stale: set[int] = set()
        with hot_cache._lock:
            for entry in hot_cache._entries:
                if entry["stale"]:
                    stale.add(entry["seq_id"])
        return stale

    def _estimate_epe_between(self, seq_id_a: int, seq_id_b: int) -> float:
        hot_cache = self._memory._hot_cache
        emb_a = None
        emb_b = None
        with hot_cache._lock:
            for entry in hot_cache._entries:
                if entry["seq_id"] == seq_id_a:
                    emb_a = entry["embedding"]
                if entry["seq_id"] == seq_id_b:
                    emb_b = entry["embedding"]
        if emb_a is not None and emb_b is not None:
            return float(1.0 - np.dot(emb_a, emb_b))
        return 0.5  # unknown; return mid-range estimate

    def _tokenise_with_roles(self, text: str, anchors: list[str]) -> list[dict]:
        anchor_set = {a.lower() for a in anchors}
        tokens = []
        for raw in text.split():
            clean = _PUNCT_STRIP.sub("", raw).lower()
            if clean in anchor_set:
                role = "anchor"
            elif clean in STOPWORDS:
                role = "stopword"
            else:
                role = "compressible"
            tokens.append({"text": raw, "role": role})
        return tokens
