from __future__ import annotations

from typing import List

from jeval.ingest.base import Segment, Session


class AMABenchLoader:
    """Load AMA benchmark for agent memory compression."""

    def load_sessions(self) -> List[Session]:
        """Load AMA sessions."""
        # Placeholder: generate synthetic sessions
        sessions = []
        for i in range(10):
            segments = [
                Segment(text=f"User question {i}: How to implement feature X?", role="user", turn=0, source="ama"),
                Segment(text=f"Agent response {i}: Here's the implementation...", role="assistant", turn=1, source="ama"),
                Segment(text=f"Follow-up {i}: What about edge cases?", role="user", turn=2, source="ama"),
                Segment(text=f"Agent follow-up {i}: Handle them like this...", role="assistant", turn=3, source="ama"),
            ]
            session = Session(session_id=f"ama_{i}", segments=segments)
            sessions.append(session)
        return sessions