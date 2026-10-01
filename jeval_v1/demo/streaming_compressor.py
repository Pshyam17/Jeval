from typing import Generator

import httpx
import openai

COMPRESSION_PROMPT = """\
You are a precise text compressor for AI agent memory.
Compress the following segment to approximately {budget_pct}% of its original length.
{anchor_line}\
Do not add information not present in the original.
Return only the compressed text, nothing else.

Segment:
{text}"""


class StreamingCompressor:
    def __init__(self, api_key: str, base_url: str, model: str):
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(
                connect=5.0,   # connection establishment
                read=30.0,     # time between bytes — NIM can be slow
                write=5.0,     # request upload
                pool=5.0,      # connection pool wait
            ),
        )
        self._model = model

    def compress_streaming(
        self, text: str, budget: float, anchors: list[str]
    ) -> Generator[str, None, None]:
        budget_pct = int(budget * 100)
        anchor_line = (
            f"You MUST preserve these tokens verbatim: {', '.join(anchors)}.\n"
            if anchors
            else ""
        )
        prompt = COMPRESSION_PROMPT.format(
            budget_pct=budget_pct, anchor_line=anchor_line, text=text
        )
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=512,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def compress(self, text: str, budget: float, anchors: list[str]) -> str:
        """Satisfy JevalMemory's compressor interface (.compress → str)."""
        return self.compress_full(text, budget, anchors)

    def compress_full(self, text: str, budget: float, anchors: list[str]) -> str:
        try:
            budget_pct = int(budget * 100)
            anchor_line = (
                f"You MUST preserve these tokens verbatim: {', '.join(anchors)}.\n"
                if anchors
                else ""
            )
            prompt = COMPRESSION_PROMPT.format(
                budget_pct=budget_pct, anchor_line=anchor_line, text=text
            )
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # convert all network/timeout errors to TimeoutError so
            # _compress_with_fidelity_gate catch block handles them
            raise TimeoutError(f"NIM call failed: {type(e).__name__}: {e}") from e
