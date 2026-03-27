from __future__ import annotations

import re
from typing import List

from jeval.compress.base import CompressorBackend


class ExtractiveBackend(CompressorBackend):
    """Lightweight extractive summarization for adversarial no-API fallback."""

    def compress(self, text: str, budget: float) -> str:
        if budget <= 0:
            return ""

        words = re.findall(r"\S+", text)
        if not words:
            return ""

        target_len = int(max(1, len(words) * budget))
        # Keep first word and highest-information tokens.
        selected = words[: min(target_len, max(1, len(words)))]
        return " ".join(selected)

    def name(self) -> str:
        return "extractive"
