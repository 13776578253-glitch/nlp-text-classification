"""Lightweight vocabulary-driven Chinese word segmentation for Batch A live inference."""

from __future__ import annotations

import re
from collections.abc import Iterable

_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_CJK_BOUNDARY_RE = re.compile(r"[\u4e00-\u9fff]\s+[\u4e00-\u9fff]")
_CHUNK_PART_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9']+|[^\s\u4e00-\u9fffA-Za-z0-9']")


def is_pretokenized_zh(text: str) -> bool:
    """Return True when Chinese text already uses whitespace token boundaries."""
    cjk_chars = _CJK_CHAR_RE.findall(text)
    if not cjk_chars:
        return True
    boundaries = len(_CJK_BOUNDARY_RE.findall(text))
    if boundaries >= 2 and boundaries / len(cjk_chars) >= 0.12:
        return True
    return bool(len(cjk_chars) <= 4 and boundaries >= 1)


class VocabularySegmenter:
    """Greedy Forward Maximum Matching (FMM) segmenter backed by a frozen model vocabulary."""

    def __init__(self, words: Iterable[str], max_word_len: int = 8) -> None:
        self._vocab = {
            word
            for word in words
            if word and " " not in word and len(word) <= max_word_len
        }
        self._max_len = max((len(w) for w in self._vocab), default=1)
        self._max_len = min(self._max_len, max_word_len)

    def _segment_cjk_span(self, span: str) -> list[str]:
        tokens: list[str] = []
        idx = 0
        length = len(span)
        while idx < length:
            matched: str | None = None
            limit = min(self._max_len, length - idx)
            for size in range(limit, 1, -1):
                candidate = span[idx : idx + size]
                if candidate in self._vocab:
                    matched = candidate
                    break
            if matched is not None:
                tokens.append(matched)
                idx += len(matched)
            else:
                tokens.append(span[idx])
                idx += 1
        return tokens

    def segment(self, text: str) -> tuple[str, list[str], bool]:
        """Segment raw Chinese text if needed, returning (spaced_text, tokens, applied)."""
        stripped = text.strip()
        if not stripped:
            return "", [], False
        if is_pretokenized_zh(stripped):
            tokens = stripped.split()
            return " ".join(tokens), tokens, False

        tokens: list[str] = []
        for chunk in stripped.split():
            for part in _CHUNK_PART_RE.findall(chunk):
                if _CJK_CHAR_RE.match(part[0]):
                    tokens.extend(self._segment_cjk_span(part))
                else:
                    tokens.append(part)
        if not tokens:
            tokens = stripped.split()
        return " ".join(tokens), tokens, True
