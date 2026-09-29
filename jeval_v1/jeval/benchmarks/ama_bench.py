from __future__ import annotations

import json
from typing import List, Optional

from datasets import load_dataset
from jeval.ingest.base import Segment, Session


class AMABenchLoader:
    """Load AMA benchmark for agent memory compression."""

    def load_sessions(
        self,
        split: str = "test",
        domain: Optional[str] = None,
        max_episodes: Optional[int] = None,
    ) -> List[Session]:
        """Load AMA benchmark sessions from the Hugging Face dataset."""
        ds = load_dataset("AMA-bench/AMA-bench", split=split)
        episodes = list(ds)
        if domain is not None and domain != "all":
            episodes = [ep for ep in episodes if ep["domain"] == domain]
        if max_episodes is not None:
            episodes = episodes[:max_episodes]

        sessions: List[Session] = []
        for ep in episodes:
            traj = ep["trajectory"]
            if isinstance(traj, str):
                traj = json.loads(traj)

            segments: List[Segment] = []
            for turn in traj:
                idx = turn.get("turn_idx", 0)
                act = turn.get("action", "")
                obs = turn.get("observation", "")
                if act:
                    segments.append(Segment(
                        text=f"Step {idx} action: {act[:800]}",
                        role="assistant",
                        turn=idx * 2,
                        source="ama",
                    ))
                if obs:
                    segments.append(Segment(
                        text=f"Step {idx} observation: {obs[:800]}",
                        role="tool",
                        turn=idx * 2 + 1,
                        source="ama",
                    ))

            sessions.append(Session(
                session_id=str(ep["episode_id"]),
                segments=segments,
                metadata={
                    "domain": ep.get("domain"),
                    "task": ep.get("task"),
                },
            ))
        return sessions