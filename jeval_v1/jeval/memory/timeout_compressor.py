from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError


class TimeoutCompressor:
    """
    Wraps any compressor with a `.call(prompt) -> str` method and enforces
    a hard wall-clock timeout. Uses ThreadPoolExecutor because signal.alarm
    is not available inside threads on macOS.
    """

    def __init__(self, compressor, timeout_seconds: float = 3.0):
        self._compressor = compressor
        self._timeout = timeout_seconds
        self._executor = ThreadPoolExecutor(max_workers=2)

    def compress(self, text: str, budget: float, anchors: list[str]) -> str:
        prompt = self._build_prompt(text, budget, anchors)
        future = self._executor.submit(self._compressor.call, prompt)
        try:
            return future.result(timeout=self._timeout)
        except FuturesTimeoutError:
            future.cancel()
            raise TimeoutError(f"compression timed out after {self._timeout}s")

    def _build_prompt(self, text: str, budget: float, anchors: list[str]) -> str:
        budget_pct = int(budget * 100)
        anchor_line = ""
        if anchors:
            anchor_list = ", ".join(anchors)
            anchor_line = (
                f"You MUST preserve these tokens verbatim in your output: {anchor_list}.\n"
            )
        return (
            f"You are a precise text compressor for AI agent memory.\n"
            f"Compress the following segment to approximately {budget_pct}% of its original length.\n"
            f"{anchor_line}"
            f"Do not add information not present in the original.\n"
            f"Return only the compressed text, nothing else.\n\n"
            f"Segment:\n{text}"
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
