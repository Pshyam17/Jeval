from __future__ import annotations

import os
from typing import Optional

from jeval.compress.base import CompressorBackend
from jeval.compress.extractive import ExtractiveBackend

try:
    from openai import OpenAI
    import openai as _openai
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]
    _openai = None  # type: ignore[assignment]

# NVIDIA NIM endpoint — swap base_url to use OpenAI or any OAI-compatible provider
_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
_NIM_MODEL    = "mistralai/mistral-small-3.1-24b-instruct-2503"


class LLMBackend(CompressorBackend):
    """
    LLM-based semantic compressor wired to NVIDIA NIM (Mistral Small).

    Falls back to ExtractiveBackend if:
    - openai package not installed
    - NVIDIA_API_KEY not set
    - API call fails (rate limit, timeout, etc.)

    Constructor params allow swapping provider without touching call sites:
        LLMBackend()                          # NIM/Mistral default
        LLMBackend(base_url=None, model="gpt-4o-mini", api_key_env="OPENAI_API_KEY")
    """

    def __init__(
        self,
        base_url: Optional[str] = _NIM_BASE_URL,
        model: str = _NIM_MODEL,
        api_key_env: str = "NVIDIA_API_KEY",
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key or os.getenv(api_key_env)
        self._fallback = ExtractiveBackend()
        self._client: Optional[object] = None

    def _get_client(self):
        if OpenAI is None:
            return None
        if self._client is None and self.api_key:
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def compress(self, text: str, budget: float) -> str:
        if not text:
            return ""
        if budget >= 1.0:
            return text

        client = self._get_client()
        if client is None or self.api_key is None:
            return self._fallback.compress(text, budget)

        target_words = max(8, int(len(text.split()) * budget))
        prompt = self._compose_prompt(text, budget, target_words)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max(64, target_words * 2),
                temperature=0.1,  # low temp for deterministic compression
            )
            compressed = response.choices[0].message.content.strip()
            if not compressed:
                return self._fallback.compress(text, budget)
            return compressed

        except Exception:
            if _openai and isinstance(
                Exception, (_openai.APIError, _openai.RateLimitError)
            ):
                pass
            return self._fallback.compress(text, budget)

    def _compose_prompt(self, text: str, budget: float, target_words: int) -> str:
        aggressiveness = "aggressively" if budget < 0.4 else "lightly"
        return (
            f"Compress the following text {aggressiveness}. "
            f"Target: ~{target_words} words (ratio {budget:.0%}). "
            "Preserve all file paths, variable names, error codes, API endpoints, "
            "decisions, and causal reasoning exactly. "
            "Drop filler, pleasantries, and redundant phrasing. "
            "Return only the compressed text, no preamble.\n\n"
            f"Text:\n{text}"
        )

    def name(self) -> str:
        return f"llm-{self.model.split('/')[-1]}"