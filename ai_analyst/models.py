from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class Document(BaseModel):
    id: str
    title: str
    source_path: Optional[str] = None
    num_pages: int
    text_by_page: Dict[int, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def full_text(self) -> str:
        ordered_pages: List[int] = sorted(self.text_by_page.keys())
        return "\n\n".join(self.text_by_page[p] for p in ordered_pages)

    def pages(self) -> List[int]:
        return sorted(self.text_by_page.keys())


class Chunk(BaseModel):
    id: str
    doc_id: str
    doc_title: str
    text: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int

    def as_citation(self) -> str:
        if self.page_start == self.page_end:
            return f"{self.doc_title} p. {self.page_start}"
        return f"{self.doc_title} pp. {self.page_start}-{self.page_end}"


class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float


class Section(BaseModel):
    name: str
    text: str
    page_start: int
    page_end: int


class ReferenceItem(BaseModel):
    raw_text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
