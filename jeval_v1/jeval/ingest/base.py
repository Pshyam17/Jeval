from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional


@dataclass(frozen=True)
class Segment:
    """One normalized logical unit extracted from session history."""

    text: str
    role: str  # user|assistant|tool|system
    turn: int
    source: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not self.text:
            raise ValueError("Segment.text cannot be empty")
        if self.role not in {"user", "assistant", "tool", "system"}:
            raise ValueError(f"Invalid role in Segment: {self.role}")


@dataclass
class Session:
    """Represents a sequence of semantically-ordered session segments."""

    segments: List[Segment] = field(default_factory=list)
    session_id: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def append(self, segment: Segment) -> None:
        """Append one normalized segment to the session."""
        if not isinstance(segment, Segment):
            raise TypeError("session.append requires a Segment instance")
        self.segments.append(segment)

    def extend(self, segments: List[Segment]) -> None:
        """Extend session with a list of segments in order."""
        if any(not isinstance(s, Segment) for s in segments):
            raise TypeError("session.extend requires Segment instances")
        self.segments.extend(segments)

    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self):
        return iter(self.segments)
