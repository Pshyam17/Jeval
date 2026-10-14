from __future__ import annotations

from jeval.compress.base import CompressorBackend


class LLMLinguaCompressor(CompressorBackend):
    """LLM-based compression using Microsoft LLM-Lingua."""

    def __init__(self):
        try:
            from llmlingua import PromptCompressor
            self.compressor = PromptCompressor()
        except ImportError:
            self.compressor = None

    def compress(self, text: str, budget: float) -> str:
        """Compress using LLM-Lingua if available, else fallback to truncation."""
        if self.compressor is None:
            # Fallback: simple truncation
            target_len = int(len(text) * budget)
            return text[:target_len]

        # Assume budget is compression ratio, e.g., 0.5 means compress to 50%
        # LLM-Lingua typically takes instruction and text
        # For simplicity, assume text is the prompt to compress
        try:
            compressed = self.compressor.compress_prompt(
                text, instruction="", ratio=budget
            )
            return compressed["compressed_prompt"]
        except Exception:
            # Fallback
            target_len = int(len(text) * budget)
            return text[:target_len]

    def name(self) -> str:
        return "llm-lingua"