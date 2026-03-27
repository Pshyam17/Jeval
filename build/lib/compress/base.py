from __future__ import annotations

from abc import ABC, abstractmethod


class CompressorBackend(ABC):
    """Backend interface for compression strategies."""

    @abstractmethod
    def compress(self, text: str, budget: float) -> str:
        """Compress text with target budget in [0,1]."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
