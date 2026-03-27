from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from jeval.artifacts.detector import extract_artifacts, is_artifact


@dataclass(frozen=True)
class ArtifactEntry:
    path: str
    artifact_type: str  # file|endpoint|variable|error|service
    status: str  # created|modified|examined|deleted|mentioned
    last_change: str
    turn: int
    epe_at_mention: float


@dataclass
class ArtifactIndex:
    """Structured side-channel index; never compressed."""

    entries: Dict[str, ArtifactEntry] = field(default_factory=dict)

    def update_from_text(self, text: str, turn: int, epe: float) -> None:
        if not text:
            return

        if not is_artifact(text):
            return

        for token in extract_artifacts(text):
            key = token.lower()
            entry = ArtifactEntry(
                path=token,
                artifact_type=self._infer_type(token),
                status="mentioned",
                last_change="mentioned",
                turn=turn,
                epe_at_mention=epe,
            )
            self.entries[key] = entry

    def _infer_type(self, token: str) -> str:
        lowered = token.lower()
        if lowered.startswith("src/") or any(lowered.endswith(s) for s in [".ts", ".py", ".js", ".go", ".java"]):
            return "file"
        if lowered.startswith("/api/"):
            return "endpoint"
        if lowered.isdigit() or lowered.startswith("4") or lowered.startswith("5"):
            return "error"
        return "variable"

    def to_list(self) -> List[ArtifactEntry]:
        return list(self.entries.values())

    def merge(self, other: ArtifactIndex) -> None:
        """Merge another index; newer turn overwrites existing entries."""
        for key, entry in other.entries.items():
            existing = self.entries.get(key)
            if existing is None or entry.turn >= existing.turn:
                self.entries[key] = entry
