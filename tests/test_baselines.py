import pytest
from jeval.baselines.factory import BaselineFactory
from jeval.baselines.truncation import TruncationCompressor
from jeval.baselines.simple_mem import SimpleMemCompressor


def test_baseline_factory():
    """Test baseline factory creates correct compressors."""
    trunc = BaselineFactory.create("truncation")
    assert isinstance(trunc, TruncationCompressor)
    assert trunc.name() == "truncation"

    mem = BaselineFactory.create("simple-mem")
    assert isinstance(mem, SimpleMemCompressor)
    assert mem.name() == "simple-mem"

    # Test unknown baseline
    with pytest.raises(ValueError):
        BaselineFactory.create("unknown")

    # Test list backends
    backends = BaselineFactory.list_backends()
    assert "truncation" in backends
    assert "simple-mem" in backends
    assert "llm-lingua" in backends


def test_truncation_compressor():
    """Test truncation compressor."""
    compressor = TruncationCompressor()
    text = "This is a test sentence with multiple words."

    # Full budget
    result = compressor.compress(text, 1.0)
    assert result == text

    # Half budget
    result = compressor.compress(text, 0.5)
    assert len(result) == len(text) // 2
    assert result == text[:len(text) // 2]


def test_simple_mem_compressor():
    """Test simple memory compressor with deduplication."""
    compressor = SimpleMemCompressor()
    text = "Hello world. Hello world. Goodbye world."

    # Should deduplicate sentences
    result = compressor.compress(text, 1.0)
    assert "Hello world" in result
    assert "Goodbye world" in result
    # Should have only one "Hello world"
    assert result.count("Hello world") == 1

    # Test truncation after dedup
    result = compressor.compress(text, 0.5)
    assert len(result) <= len(text) * 0.5 + 10  # Allow some tolerance


def test_llm_lingua_compressor_fallback():
    """Test LLM-Lingua compressor fallback when not installed."""
    from jeval.baselines.llm_lingua import LLMLinguaCompressor

    compressor = LLMLinguaCompressor()
    text = "This is a test sentence."

    # Should fallback to truncation since llmlingua not installed
    result = compressor.compress(text, 0.5)
    assert len(result) == len(text) // 2
    assert compressor.name() == "llm-lingua"