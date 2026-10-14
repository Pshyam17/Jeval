from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TokenInfo:
    text: str
    role: Literal["anchor", "compressible", "stopword"] = "compressible"


@dataclass
class SegmentIngestEvent:
    type: Literal["segment_ingest"] = "segment_ingest"
    seq_id: int = 0
    action: str = ""
    content_type: str = ""
    epe_novelty: float = 0.0
    budget: float = 0.0
    anchors: list[str] = field(default_factory=list)
    token_count_original: int = 0
    token_count_compressed: int = 0
    original_tokens: list[dict] = field(default_factory=list)
    coords_3d: list[float] = field(default_factory=list)


@dataclass
class CompressionStartEvent:
    type: Literal["compression_start"] = "compression_start"
    seq_id: int = 0
    budget: float = 0.0
    budget_pct: int = 0
    content_type: str = ""
    attempt: int = 0


@dataclass
class CompressionTokenEvent:
    type: Literal["compression_token"] = "compression_token"
    seq_id: int = 0
    token: str = ""
    is_anchor: bool = False


@dataclass
class CompressionCompleteEvent:
    type: Literal["compression_complete"] = "compression_complete"
    seq_id: int = 0
    epe_fidelity: float = 0.0
    passed_gate: bool = True
    attempt: int = 0
    compressed_text: str = ""


@dataclass
class ContradictionEvent:
    type: Literal["contradiction"] = "contradiction"
    new_seq_id: int = 0
    stale_seq_id: int = 0
    epe_between: float = 0.0


@dataclass
class RetrievalEvent:
    type: Literal["retrieval"] = "retrieval"
    query: str = ""
    query_type: str = ""
    result_seq_ids: list[int] = field(default_factory=list)
    result_text: str = ""


@dataclass
class SessionStatsEvent:
    type: Literal["session_stats"] = "session_stats"
    hot_cache_size: int = 0
    hot_cache_tokens: int = 0
    cold_storage_size: int = 0
    fact_index_size: int = 0
    session_id: str = ""
