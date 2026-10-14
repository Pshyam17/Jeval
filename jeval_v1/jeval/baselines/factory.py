from __future__ import annotations

from typing import Dict, Type

from jeval.baselines.llm_lingua import LLMLinguaCompressor
from jeval.baselines.simple_mem import SimpleMemCompressor
from jeval.baselines.truncation import TruncationCompressor
from jeval.compress.base import CompressorBackend


class BaselineFactory:
    """Factory for baseline compression backends."""

    _backends: Dict[str, Type[CompressorBackend]] = {
        "truncation": TruncationCompressor,
        "llm-lingua": LLMLinguaCompressor,
        "simple-mem": SimpleMemCompressor,
    }

    @classmethod
    def create(cls, name: str) -> CompressorBackend:
        """Create a baseline compressor by name."""
        backend_cls = cls._backends.get(name.lower())
        if backend_cls is None:
            raise ValueError(f"Unknown baseline: {name}")
        return backend_cls()

    @classmethod
    def list_backends(cls) -> list[str]:
        """List available baseline names."""
        return list(cls._backends.keys())