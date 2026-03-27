from __future__ import annotations

from typing import List

from jeval.ingest.base import Segment, Session


class SWEBenchLoader:
    """Load SWE-Bench dataset as agent memory sessions."""

    def load_sessions(self) -> List[Session]:
        """Load SWE-Bench problems as sessions."""
        try:
            from datasets import load_dataset
            dataset = load_dataset("princeton-nlp/SWE-bench", split="test")
        except ImportError:
            # Fallback: dummy data
            return self._dummy_sessions()

        sessions = []
        for item in dataset:
            problem = item["problem_statement"]
            solution = item["solution"]
            segments = [
                Segment(content=problem, turn=0),
                Segment(content=solution, turn=1),
            ]
            session = Session(session_id=item["instance_id"], segments=segments)
            sessions.append(session)
        return sessions

    def _dummy_sessions(self) -> List[Session]:
        """Dummy sessions for testing."""
        return [
            Session(
                session_id="dummy1",
                segments=[
                    Segment(text="Fix the bug in src/main.py", role="user", turn=0, source="swe-bench"),
                    Segment(text="Changed line 10 to use correct variable.", role="assistant", turn=1, source="swe-bench"),
                ],
            )
        ]