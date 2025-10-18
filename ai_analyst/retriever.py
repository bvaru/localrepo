from __future__ import annotations

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import math
import re

from .models import Chunk, RetrievalResult
from .store import DocumentStore


def _tokenize(text: str) -> List[str]:
    # Simple, fast tokenization with lowercase and alphanum filtering
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class BM25Index:
    doc_freq: Dict[str, int]
    idf: Dict[str, float]
    tokens_by_chunk: Dict[str, List[str]]
    avg_len: float
    N: int


class BM25Retriever:
    """Lightweight BM25 retriever over chunks. No external deps."""

    def __init__(self, store: DocumentStore, k1: float = 1.5, b: float = 0.75) -> None:
        self.store = store
        self.k1 = k1
        self.b = b
        self._index: Optional[BM25Index] = None
        self._build_index()

    def _build_index(self) -> None:
        chunks: List[Chunk] = self.store.get_chunks()
        N = len(chunks)
        tokens_by_chunk: Dict[str, List[str]] = {}
        doc_freq: Dict[str, int] = defaultdict(int)
        lengths: List[int] = []

        for ch in chunks:
            tokens = _tokenize(ch.text)
            tokens_by_chunk[ch.id] = tokens
            lengths.append(len(tokens))
            seen: set[str] = set(tokens)
            for tok in seen:
                doc_freq[tok] += 1

        avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0
        idf: Dict[str, float] = {}
        for tok, df in doc_freq.items():
            # BM25 IDF with +0.5 smoothing
            idf[tok] = math.log((N - df + 0.5) / (df + 0.5) + 1e-9)

        self._index = BM25Index(
            doc_freq=dict(doc_freq), idf=idf, tokens_by_chunk=tokens_by_chunk, avg_len=avg_len, N=N
        )

    def refresh(self) -> None:
        self._build_index()

    def _score_chunk(self, query_tokens: List[str], chunk_id: str) -> float:
        assert self._index is not None
        tokens = self._index.tokens_by_chunk.get(chunk_id, [])
        if not tokens or not query_tokens:
            return 0.0
        tf: Dict[str, int] = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        dl = len(tokens)
        denom_norm = self.k1 * (1 - self.b + self.b * (dl / (self._index.avg_len or 1.0)))
        score = 0.0
        for qt in query_tokens:
            f = tf.get(qt, 0)
            if f == 0:
                continue
            idf = self._index.idf.get(qt, 0.0)
            numer = f * (self.k1 + 1)
            denom = f + denom_norm
            score += idf * (numer / (denom or 1e-9))
        return score

    def search(self, query: str, k: int = 5, restrict_doc_ids: Optional[List[str]] = None) -> List[RetrievalResult]:
        assert self._index is not None
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        # Determine target chunk ids
        if restrict_doc_ids is None:
            target_chunks: List[Chunk] = self.store.get_chunks()
        else:
            target_chunks = []
            for doc_id in restrict_doc_ids:
                target_chunks.extend(self.store.get_chunks(doc_id))
        scored: List[Tuple[str, float]] = []
        for ch in target_chunks:
            s = self._score_chunk(q_tokens, ch.id)
            if s > 0:
                scored.append((ch.id, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        results: List[RetrievalResult] = []
        for chunk_id, s in scored[:k]:
            ch = self.store.get_chunk_by_id(chunk_id)
            results.append(RetrievalResult(chunk=ch, score=s))
        return results
