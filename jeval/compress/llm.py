from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI
import openai

from jeval.compress.base import CompressorBackend
from jeval.compress.extractive import ExtractiveBackend


class LLMBackend(CompressorBackend):
    """OpenAI-compatible compressor backend with graceful fallback to extractive."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self._fallback = ExtractiveBackend()

    def compress(self, text: str, budget: float) -> str:
        if not text:
            return ""
        if self.api_key is None:
            return self._fallback.compress(text, budget)

        client = OpenAI(api_key=self.api_key)
        prompt = self._compose_prompt(text, budget)

        try:
            choose = "compress" if budget < 0.75 else "lightly compress"
            response = client.responses.create(
                model=self.model,
                input=prompt,
                max_tokens=max(64, int(len(text.split()) * budget)),
            )
            compressed = response.output_text
            if not compressed:
                return self._fallback.compress(text, budget)
            return compressed
        except (openai.APIError, openai.RateLimitError, ValueError, TypeError):
            return self._fallback.compress(text, budget)

    def _compose_prompt(self, text: str, budget: float) -> str:
        return (
            "Compress this text while preserving semantic fidelity. "
            f"Target length ratio: {budget:.2f}.\n\nOriginal:\n{text}\n\nCompressed:"
        )

    def name(self) -> str:
        return "openai-llm"
