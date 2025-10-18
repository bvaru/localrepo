from __future__ import annotations

from typing import Dict, List, Iterable, Optional
from dataclasses import dataclass
from .models import Document, Chunk
from .chunker import create_chunks, ChunkingConfig


@dataclass
class StoredDoc:
    document: Document
    chunks: List[Chunk]


class DocumentStore:
    """In-memory document and chunk store."""

    def __init__(self) -> None:
        self._docs: Dict[str, StoredDoc] = {}
        self._chunk_lookup: Dict[str, Chunk] = {}
        self._doc_title_by_id: Dict[str, str] = {}

    def add_document(self, document: Document, config: Optional[ChunkingConfig] = None) -> List[Chunk]:
        if document.id in self._docs:
            raise ValueError(f"Document with id {document.id} already exists")
        chunks: List[Chunk] = create_chunks(
            text_by_page=document.text_by_page,
            doc_id=document.id,
            doc_title=document.title,
            config=config or ChunkingConfig(),
        )
        self._docs[document.id] = StoredDoc(document=document, chunks=chunks)
        self._doc_title_by_id[document.id] = document.title
        for ch in chunks:
            self._chunk_lookup[ch.id] = ch
        return chunks

    def remove_document(self, doc_id: str) -> None:
        stored = self._docs.pop(doc_id, None)
        self._doc_title_by_id.pop(doc_id, None)
        if stored:
            for ch in stored.chunks:
                self._chunk_lookup.pop(ch.id, None)

    def list_documents(self) -> List[Document]:
        return [sd.document for sd in self._docs.values()]

    def get_document(self, doc_id: str) -> Document:
        return self._docs[doc_id].document

    def get_chunks(self, doc_id: Optional[str] = None) -> List[Chunk]:
        if doc_id is None:
            result: List[Chunk] = []
            for sd in self._docs.values():
                result.extend(sd.chunks)
            return result
        return list(self._docs[doc_id].chunks)

    def get_chunk_by_id(self, chunk_id: str) -> Chunk:
        return self._chunk_lookup[chunk_id]

    def doc_title(self, doc_id: str) -> str:
        return self._doc_title_by_id[doc_id]
