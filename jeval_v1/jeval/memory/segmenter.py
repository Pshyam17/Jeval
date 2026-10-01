from __future__ import annotations

import re

_HEADER_RE = re.compile(r"^(#{2,3}\s+.+)$", re.MULTILINE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_MIN_TOKENS = 5
_MAX_TOKENS = 300
_MIN_BULLET_TOKENS = 20


class SessionSegmenter:
    def segment(self, text: str) -> list[str]:
        if not text.strip():
            return []

        raw = self._split_on_structure(text)
        merged = self._merge_short(raw)
        final = self._split_long(merged)
        return [s for s in final if s.strip()]

    def _split_on_structure(self, text: str) -> list[str]:
        """Split on markdown headers, then double newlines, then bullets."""
        segments: list[str] = []
        header_spans = [(m.start(), m.end()) for m in _HEADER_RE.finditer(text)]

        if header_spans:
            prev_end = 0
            for i, (hstart, _hend) in enumerate(header_spans):
                # pre-header content
                if i == 0 and hstart > 0:
                    segments.extend(self._split_paragraph(text[prev_end:hstart]))
                next_start = header_spans[i + 1][0] if i + 1 < len(header_spans) else len(text)
                header_block = text[hstart:next_start].strip()
                # split the header block itself on double newlines
                segments.extend(self._split_paragraph(header_block))
                prev_end = next_start
        else:
            # no headers — split the entire text on double newlines first
            for chunk in re.split(r"\n{2,}", text):
                chunk = chunk.strip()
                if chunk:
                    segments.extend(self._split_paragraph(chunk))

        return segments

    def _split_paragraph(self, text: str) -> list[str]:
        """Split on double newlines, then handle bullet lines."""
        paragraphs = re.split(r"\n{2,}", text.strip())
        result: list[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            lines = [l.strip() for l in para.splitlines() if l.strip()]
            bullet_lines = [l for l in lines if re.match(r"^[-*]\s+", l)]
            non_bullet = [l for l in lines if not re.match(r"^[-*]\s+", l)]

            if bullet_lines and not non_bullet:
                result.extend(self._merge_bullets(bullet_lines))
            elif bullet_lines and non_bullet:
                result.append(" ".join(non_bullet))
                result.extend(self._merge_bullets(bullet_lines))
            else:
                result.append(para)
        return result

    def _merge_bullets(self, bullets: list[str]) -> list[str]:
        """Merge adjacent bullets until each group reaches MIN_BULLET_TOKENS."""
        merged: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for b in bullets:
            tok_count = len(b.split())
            current.append(b)
            current_tokens += tok_count
            if current_tokens >= _MIN_BULLET_TOKENS:
                merged.append(" ".join(current))
                current = []
                current_tokens = 0
        if current:
            if merged:
                merged[-1] = merged[-1] + " " + " ".join(current)
            else:
                merged.append(" ".join(current))
        return merged

    def _merge_short(self, segments: list[str]) -> list[str]:
        """Merge segments under MIN_TOKENS with the next segment."""
        merged: list[str] = []
        i = 0
        while i < len(segments):
            seg = segments[i]
            tok_count = len(seg.split())
            if tok_count < _MIN_TOKENS and i + 1 < len(segments):
                segments[i + 1] = seg + " " + segments[i + 1]
                i += 1
                continue
            merged.append(seg)
            i += 1
        return merged

    def _split_long(self, segments: list[str]) -> list[str]:
        """Split segments over MAX_TOKENS at the nearest sentence boundary."""
        result: list[str] = []
        for seg in segments:
            if len(seg.split()) <= _MAX_TOKENS:
                result.append(seg)
                continue
            # split at sentence boundaries
            parts = _SENTENCE_BOUNDARY.split(seg)
            current: list[str] = []
            current_tokens = 0
            for part in parts:
                pt = len(part.split())
                if current_tokens + pt > _MAX_TOKENS and current:
                    result.append(" ".join(current))
                    current = [part]
                    current_tokens = pt
                else:
                    current.append(part)
                    current_tokens += pt
            if current:
                result.append(" ".join(current))
        return result
