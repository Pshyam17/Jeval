import os
import pytest

from demo.streaming_compressor import COMPRESSION_PROMPT, StreamingCompressor

_api_key = os.environ.get("NVIDIA_API_KEY", "")
_has_key = bool(_api_key)

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL = "mistralai/mistral-small-3.1-24b-instruct-2503"


@pytest.mark.skipif(not _has_key, reason="NVIDIA_API_KEY not set")
def test_compress_full_returns_nonempty():
    compressor = StreamingCompressor(
        api_key=_api_key, base_url=NIM_BASE_URL, model=NIM_MODEL
    )
    result = compressor.compress_full(
        text="step 7 action: corrected environment variable name — renamed JWT_KEY to JWT_SECRET in src/config/env.ts line 12",
        budget=0.5,
        anchors=["JWT_SECRET", "src/config/env.ts"],
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_anchor_line_present_when_anchors_provided():
    budget_pct = 60
    anchors = ["JWT_SECRET", "src/auth.ts"]
    anchor_line = f"You MUST preserve these tokens verbatim: {', '.join(anchors)}.\n"
    prompt = COMPRESSION_PROMPT.format(
        budget_pct=budget_pct,
        anchor_line=anchor_line,
        text="some segment text",
    )
    assert "You MUST preserve these tokens verbatim" in prompt
    assert "JWT_SECRET" in prompt
    assert "src/auth.ts" in prompt


def test_empty_anchors_produces_clean_prompt():
    budget_pct = 50
    anchor_line = ""
    prompt = COMPRESSION_PROMPT.format(
        budget_pct=budget_pct,
        anchor_line=anchor_line,
        text="some segment text",
    )
    assert "MUST preserve" not in prompt
    assert "verbatim" not in prompt
    assert "50%" in prompt
