from __future__ import annotations

from typing import List, Tuple
import re
from collections import Counter

STOPWORDS = set(
    """
    a an the and or but if while of at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most other some such no nor not only own same so than too very s t can will just don don should now is are was were be been being have has had having do does did doing would could should might may must i you he she it we they me him her us them my your his its our their mine yours hers ours theirs this that these those as per via etc et al vs versus figure table
    """.split()
)


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def frequency_summarize(text: str, max_sentences: int = 5) -> str:
    sentences = _split_sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    tokens = _tokenize(text)
    if not tokens:
        return " ".join(sentences[:max_sentences])
    freq = Counter(tokens)
    # score sentences by sum of token frequencies
    scored: List[Tuple[float, str, int]] = []
    for idx, s in enumerate(sentences):
        stoks = _tokenize(s)
        score = sum(freq.get(t, 0) for t in stoks) / (len(stoks) or 1)
        scored.append((score, s, idx))
    scored.sort(key=lambda x: (-x[0], x[2]))
    top = sorted(scored[:max_sentences], key=lambda x: x[2])
    return " ".join(s for _, s, _ in top)
