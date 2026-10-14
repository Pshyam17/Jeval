from __future__ import annotations

import re
from jeval.compress.base import CompressorBackend


class SimpleMemCompressor(CompressorBackend):
    """Compress by removing redundant segments (simple deduplication)."""

    def compress(self, text: str, budget: float) -> str:
        """Remove duplicate sentences and truncate to budget."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Remove duplicates while preserving order
        seen = set()
        unique_sentences = []
        for s in sentences:
            s_lower = s.lower().strip()
            if s_lower not in seen and s_lower:
                seen.add(s_lower)
                unique_sentences.append(s)

        deduped = ' '.join(unique_sentences)
        # Then truncate to budget
        if budget >= 1.0:
            return deduped
        target_len = int(len(deduped) * budget)
        return deduped[:target_len]

    def name(self) -> str:
        return "simple-mem"