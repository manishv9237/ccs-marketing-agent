"""Sentence-aware chunking with overlap."""
from __future__ import annotations

import re

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for para in re.split(r"\n+", text):
        para = para.strip()
        if not para:
            continue
        out.extend(s.strip() for s in _SENT.split(para) if s.strip())
    return out


def chunk_text(text: str, target_words: int = 170, overlap_words: int = 30, min_words: int = 12) -> list[str]:
    """Group sentences into ~target_words chunks; carry a short overlap forward so
    facts that straddle a boundary stay retrievable."""
    sentences: list[str] = []
    for s in _sentences(text):
        words = s.split()
        if len(words) > target_words:  # very long run-on / table row: hard split
            for i in range(0, len(words), target_words):
                sentences.append(" ".join(words[i : i + target_words]))
        else:
            sentences.append(s)

    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for s in sentences:
        n = len(s.split())
        if cur and cur_words + n > target_words:
            chunks.append(" ".join(cur))
            # overlap: keep trailing sentences up to overlap_words
            keep: list[str] = []
            kept = 0
            for prev in reversed(cur):
                pw = len(prev.split())
                if kept + pw > overlap_words:
                    break
                keep.insert(0, prev)
                kept += pw
            cur, cur_words = keep, kept
        cur.append(s)
        cur_words += n
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if len(c.split()) >= min_words] or ([" ".join(sentences)] if sentences and len(" ".join(sentences).split()) >= 3 else [])
