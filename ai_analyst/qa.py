from __future__ import annotations

from typing import List, Optional, Dict
from .retriever import BM25Retriever
from .models import RetrievalResult
from .summarizer import frequency_summarize
from . import llm


ANSWER_SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer using ONLY the provided context. "
    "Cite sources with bracketed numbers like [1], [2] that correspond to the given citations. "
    "If the answer is not in the context, say you don't know. Be concise but precise."
)


def _compose_context(results: List[RetrievalResult]) -> str:
    blocks = []
    for idx, r in enumerate(results, start=1):
        citation = f"[{idx}] {r.chunk.as_citation()}"
        blocks.append(f"{citation}\n{r.chunk.text}")
    return "\n\n".join(blocks)


def _compose_citations(results: List[RetrievalResult]) -> List[str]:
    return [r.chunk.as_citation() for r in results]


def answer_query(retriever: BM25Retriever, query: str, k: int = 6, restrict_doc_ids: Optional[List[str]] = None) -> Dict[str, object]:
    results = retriever.search(query, k=k, restrict_doc_ids=restrict_doc_ids)
    if not results:
        return {"answer": "I couldn't find relevant content in the documents.", "citations": []}

    context = _compose_context(results)
    citations = _compose_citations(results)

    messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Question: "
                + query
                + "\n\nContext passages with citations:\n"
                + context
                + "\n\nAnswer clearly with citations in brackets."
            ),
        },
    ]
    llm_answer = llm.chat(messages)

    if llm_answer:
        return {"answer": llm_answer.strip(), "citations": citations}

    # Extractive fallback: concatenate top passages and summarize
    joined = "\n\n".join(r.chunk.text for r in results)
    summary = frequency_summarize(joined, max_sentences=6)
    # Add inline numeric citations sequentially
    numbered = "".join(f"[{i+1}] " for i in range(min(len(results), 3)))
    answer = summary + (" " + numbered if numbered else "")
    return {"answer": answer.strip(), "citations": citations}
