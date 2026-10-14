import time
import pytest
from jeval.memory.timeout_compressor import TimeoutCompressor


class _FastCaller:
    def call(self, prompt: str) -> str:
        return "compressed: " + prompt[:30]


class _SlowCaller:
    def call(self, prompt: str) -> str:
        time.sleep(10)
        return "never"


def test_normal_compression_returns_result():
    tc = TimeoutCompressor(_FastCaller(), timeout_seconds=2.0)
    result = tc.compress("some text to compress", budget=0.5, anchors=[])
    assert "compressed:" in result
    tc.shutdown()


def test_timeout_raises_timeout_error():
    tc = TimeoutCompressor(_SlowCaller(), timeout_seconds=0.5)
    with pytest.raises(TimeoutError):
        tc.compress("some text", budget=0.5, anchors=[])
    tc.shutdown()


def test_anchor_list_included_in_prompt():
    seen_prompts: list[str] = []

    class _CaptureCaller:
        def call(self, prompt: str) -> str:
            seen_prompts.append(prompt)
            return "ok"

    tc = TimeoutCompressor(_CaptureCaller(), timeout_seconds=2.0)
    tc.compress("some segment text", budget=0.4, anchors=["JWT_SECRET", "src/auth.ts"])
    assert seen_prompts
    assert "JWT_SECRET" in seen_prompts[0]
    assert "src/auth.ts" in seen_prompts[0]
    tc.shutdown()


def test_empty_anchor_list_clean_prompt():
    seen_prompts: list[str] = []

    class _CaptureCaller:
        def call(self, prompt: str) -> str:
            seen_prompts.append(prompt)
            return "ok"

    tc = TimeoutCompressor(_CaptureCaller(), timeout_seconds=2.0)
    tc.compress("some segment text", budget=0.5, anchors=[])
    assert seen_prompts
    assert "MUST preserve" not in seen_prompts[0]
    tc.shutdown()
