from __future__ import annotations

import asyncio
import datetime
import json
import os
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# absolute paths relative to this file — correct regardless of working directory
_demo_dir = Path(__file__).parent
_repo_root = _demo_dir.parent

_encoder = None
_memory = None
_projector = None
_compressor = None
_bridge = None

_nim_ready = False
_first_client_connected = asyncio.Event()
_query_log_path = _repo_root / ".jeval" / "query_log.jsonl"


class ConnectionManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []
        # replay buffer — new clients receive all past events on connect
        self._history: list[str] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        # signal that a browser has connected — starts replay if waiting
        _first_client_connected.set()
        # catch up new client with all past events
        for msg in self._history:
            try:
                await ws.send_text(msg)
            except Exception:
                return
        self.active.append(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, event: object) -> None:
        msg = json.dumps(asdict(event))
        self._history.append(msg)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)


manager = ConnectionManager()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    if _nim_ready:
        return {"status": "ready"}
    from fastapi import HTTPException
    raise HTTPException(status_code=503, detail="warming up")


@app.get("/")
async def index():
    return FileResponse(str(_demo_dir / "static" / "index.html"))


@app.get("/latent")
async def latent():
    return FileResponse(str(_demo_dir / "static" / "latent.html"))


@app.get("/compression")
async def compression():
    return FileResponse(str(_demo_dir / "static" / "compression.html"))


@app.get("/queries")
async def query_log():
    if not _query_log_path.exists():
        return {"total": 0, "queries": []}
    lines = _query_log_path.read_text().splitlines()
    queries = [json.loads(l) for l in lines if l.strip()]
    from collections import Counter
    types = Counter(q["query_type"] for q in queries)
    return {
        "total": len(queries),
        "by_type": dict(types),
        "queries": queries,
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "retrieve":
                query = msg.get("query", "").strip()
                if query and _memory:
                    result = _memory.retrieve(query)
                    from demo.events import RetrievalEvent
                    from jeval.memory.query_classifier import QueryClassifier
                    qc = QueryClassifier()
                    query_type = qc.classify(query)
                    event = RetrievalEvent(
                        query=query,
                        query_type=query_type,
                        result_seq_ids=[],
                        result_text=result,
                    )
                    await manager.broadcast(event)

                    # persist query for post-showcase analysis
                    _query_log_path.parent.mkdir(exist_ok=True)
                    log_entry = {
                        "timestamp": datetime.datetime.utcnow().isoformat(),
                        "query": query,
                        "query_type": query_type,
                        "result_seq_ids": event.result_seq_ids,
                        "result_text": result[:200],
                        "session_id": _memory.stats().get("session_id", ""),
                    }
                    with _query_log_path.open("a") as f:
                        f.write(json.dumps(log_entry) + "\n")

    except WebSocketDisconnect:
        await manager.disconnect(ws)


@app.on_event("startup")
async def startup():
    global _encoder, _memory, _projector, _compressor, _bridge, _nim_ready

    from jeval.encoders.sentence_encoder import FrozenEncoder
    from jeval.memory.jeval_memory import JevalMemory
    from demo.umap_projector import UMAPProjector
    from demo.streaming_compressor import StreamingCompressor
    from demo.agent_bridge import AgentBridge
    from demo.data.seed_texts import SEED_TEXTS

    _encoder = FrozenEncoder()

    _projector = UMAPProjector()
    _projector.seed_from_encoder(_encoder, SEED_TEXTS)

    api_key = os.environ.get("NVIDIA_API_KEY", "")
    _compressor = StreamingCompressor(
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1",
        model="mistralai/mistral-small-3.1-24b-instruct-2503",
    )

    # warm up NIM — first call after inactivity triggers model loading
    # which takes 30-60 seconds. fire a minimal call now so the model
    # is ready before the first replay segment arrives.
    print("warming up NIM...", flush=True)
    try:
        from concurrent.futures import ThreadPoolExecutor as _TPE
        _warmup_executor = _TPE(max_workers=1)
        loop = asyncio.get_event_loop()
        await asyncio.wait_for(
            loop.run_in_executor(
                _warmup_executor,
                lambda: _compressor.compress_full("warmup", 0.5, []),
            ),
            timeout=90.0,  # allow up to 90s for cold start
        )
        print("NIM warm — waiting for browser", flush=True)
    except Exception as e:
        print(f"NIM warm-up failed: {e} — continuing anyway", flush=True)

    _nim_ready = True
    asyncio.create_task(_run_bridge())


async def _run_bridge():
    global _memory, _bridge

    from jeval.memory.jeval_memory import JevalMemory
    from demo.agent_bridge import AgentBridge

    print("waiting for browser to connect...", flush=True)
    await _first_client_connected.wait()
    await asyncio.sleep(2.0)
    print("browser connected — starting replay", flush=True)

    while True:
        # clear demo database so every replay starts fresh
        demo_db = _repo_root / ".jeval" / "demo_memory.db"
        for p in [demo_db,
                  Path(str(demo_db) + "-wal"),
                  Path(str(demo_db) + "-shm")]:
            if p.exists():
                p.unlink()
        # also clear replay buffer so new clients see only the current session
        manager._history.clear()

        api_key = os.environ.get("NVIDIA_API_KEY", "")
        _memory = JevalMemory(
            db_path=demo_db,
            hot_cache_token_ceiling=6000,
            novelty_threshold=0.12,
            fidelity_threshold=0.25,
            session_id="demo_session_001",
            encoder=_encoder,
            compressor=_compressor if api_key else None,
        )

        replay_speed = float(os.environ.get("REPLAY_SPEED", "1.0"))
        _bridge = AgentBridge(
            memory=_memory,
            encoder=_encoder,
            projector=_projector,
            compressor=_compressor,
            replay_path=_demo_dir / "data" / "sample_session.jsonl",
            replay_speed=replay_speed,
        )

        async for event in _bridge.run():
            await manager.broadcast(event)

        print("session complete — restarting in 10s", flush=True)
        await asyncio.sleep(10.0)
        _first_client_connected.clear()
        await _first_client_connected.wait()
        await asyncio.sleep(2.0)
        print("browser connected — starting replay", flush=True)
