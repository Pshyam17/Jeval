from __future__ import annotations

from jeval.compress.base import CompressorBackend


class TruncationCompressor(CompressorBackend):
    """Simple truncation to fixed length based on budget."""

    def compress(self, text: str, budget: float) -> str:
        """Truncate text to budget fraction of original length."""
        if budget >= 1.0:
            return text
        target_len = int(len(text) * budget)
        return text[:target_len]

    def name(self) -> str:
        return "truncation"