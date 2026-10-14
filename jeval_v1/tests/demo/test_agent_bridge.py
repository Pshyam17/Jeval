from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from demo.agent_bridge import AgentBridge
from demo.events import (
    CompressionStartEvent,
    SegmentIngestEvent,
    SessionStatsEvent,
)


@pytest.fixture(scope="module")
def encoder():
    from jeval.encoders.sentence_encoder import FrozenEncoder
    return FrozenEncoder()


@pytest.fixture(scope="module")
def projector(encoder):
    from demo.umap_projector import UMAPProjector
    proj = UMAPProjector()
    texts = [
        "created auth middleware",
        "test suite failed",
        "deployment failed health check",
        "JWT_SECRET mismatch fixed",
        "all 23 tests passing",
        "modified routes rate limiting",
        "rolled back migration",
        "error 503 staging",
        "npm install 847 packages",
        "TypeError at line 34",
    ]
    proj.seed_from_encoder(encoder, texts)
    return proj


@pytest.fixture(scope="module")
def memory(tmp_path_factory, encoder):
    from jeval.memory.jeval_memory import JevalMemory
    from demo.umap_projector import UMAPProjector
    db = tmp_path_factory.mktemp("bridge_mem") / "test.db"
    # Use extractive caller so no LLM needed
    from jeval.compress.extractive import ExtractiveBackend

    class _FakeCaller:
        def call(self, prompt: str) -> str:
            text = prompt.split("Segment:\n", 1)[-1] if "Segment:\n" in prompt else prompt
            words = text.split()
            return " ".join(words[: max(1, len(words) // 2)])

    return JevalMemory(
        db_path=db,
        session_id="test_bridge",
        _caller=_FakeCaller(),
    )


@pytest.fixture(scope="module")
def compressor():
    mock = MagicMock()
    # compress_streaming yields a few tokens
    mock.compress_streaming.return_value = iter(["compressed ", "text ", "here"])
    mock.compress_full.return_value = "compressed text here"
    return mock


@pytest.fixture(scope="module")
def replay_path(tmp_path_factory):
    entries = [
        {"text": f"step {i} action: doing task {i}", "delay_seconds": 0.0}
        for i in range(1, 7)
    ]
    # add a duplicate to trigger cold_only
    entries.append({"text": "step 1 action: doing task 1", "delay_seconds": 0.0})
    p = tmp_path_factory.mktemp("replay") / "session.jsonl"
    with p.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return p


def _make_bridge(memory, encoder, projector, compressor, replay_path):
    return AgentBridge(
        memory=memory,
        encoder=encoder,
        projector=projector,
        compressor=compressor,
        replay_path=replay_path,
        replay_speed=1.0,
    )


def _collect(bridge: AgentBridge) -> list:
    async def _run():
        events = []
        async for event in bridge.run():
            events.append(event)
        return events

    return asyncio.run(_run())


def test_replay_yields_correct_number_of_ingest_events(
    memory, encoder, projector, compressor, replay_path
):
    bridge = _make_bridge(memory, encoder, projector, compressor, replay_path)
    # reset mock to make streaming work each call
    compressor.compress_streaming.return_value = iter(["ok"])
    events = _collect(bridge)
    ingests = [e for e in events if isinstance(e, SegmentIngestEvent)]
    # 7 entries in fixture → 7 ingest events
    assert len(ingests) == 7


def test_cold_only_segments_produce_no_compression_start(
    memory, encoder, projector, compressor, replay_path
):
    bridge = _make_bridge(memory, encoder, projector, compressor, replay_path)
    compressor.compress_streaming.return_value = iter(["ok"])
    events = _collect(bridge)
    ingests = [e for e in events if isinstance(e, SegmentIngestEvent)]
    cold_only_seq_ids = {e.seq_id for e in ingests if e.action == "cold_only"}
    compression_starts = {e.seq_id for e in events if isinstance(e, CompressionStartEvent)}
    # no cold_only seq_id should have a corresponding compression_start
    assert cold_only_seq_ids.isdisjoint(compression_starts)


def test_all_events_have_valid_seq_ids(
    memory, encoder, projector, compressor, replay_path
):
    bridge = _make_bridge(memory, encoder, projector, compressor, replay_path)
    compressor.compress_streaming.return_value = iter(["ok"])
    events = _collect(bridge)
    for event in events:
        if hasattr(event, "seq_id"):
            assert event.seq_id > 0, f"invalid seq_id on {event}"


def test_delay_respected_within_tolerance(
    memory, encoder, projector, compressor, tmp_path_factory
):
    entries = [
        {"text": "step A action: fast test", "delay_seconds": 0.1},
        {"text": "step B action: also fast", "delay_seconds": 0.1},
    ]
    p = tmp_path_factory.mktemp("delay") / "s.jsonl"
    with p.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    bridge = AgentBridge(
        memory=memory,
        encoder=encoder,
        projector=projector,
        compressor=compressor,
        replay_path=p,
        replay_speed=1.0,
    )
    compressor.compress_streaming.return_value = iter(["ok"])
    t0 = time.monotonic()
    _collect(bridge)
    elapsed = time.monotonic() - t0
    # two entries × 0.1s delay = 0.2s minimum
    # upper bound is generous: encoding + UMAP transform + ingest adds ~1-2s per entry
    assert elapsed >= 0.15
    assert elapsed < 10.0
