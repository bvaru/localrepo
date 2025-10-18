from __future__ import annotations

from typing import Dict, List
from dataclasses import dataclass
import re
import uuid
from .models import Chunk


@dataclass
class ChunkingConfig:
    max_chars: int = 1200
    overlap: int = 200


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_into_chunks(text: str, max_chars: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        # Try to end on a sentence boundary if possible
        window = text[start:end]
        period_pos = window.rfind(". ")
        if period_pos != -1 and (end - start) > 200:
            end = start + period_pos + 1
            window = text[start:end]
        chunks.append(window.strip())
        if end >= n:
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c]


def create_chunks(
    text_by_page: Dict[int, str],
    doc_id: str,
    doc_title: str,
    config: ChunkingConfig | None = None,
) -> List[Chunk]:
    cfg = config or ChunkingConfig()
    all_chunks: List[Chunk] = []
    for page_num in sorted(text_by_page.keys()):
        page_text = _normalize_whitespace(text_by_page[page_num])
        if not page_text:
            continue
        parts = _split_into_chunks(page_text, cfg.max_chars, cfg.overlap)
        cursor = 0
        for part in parts:
            start_idx = page_text.find(part, cursor)
            if start_idx == -1:
                start_idx = cursor
            end_idx = start_idx + len(part)
            ch = Chunk(
                id=str(uuid.uuid4()),
                doc_id=doc_id,
                doc_title=doc_title,
                text=part,
                page_start=page_num,
                page_end=page_num,
                char_start=start_idx,
                char_end=end_idx,
            )
            all_chunks.append(ch)
            cursor = end_idx
    return all_chunks
