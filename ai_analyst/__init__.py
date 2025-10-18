"""AI Research & Document Analyst core package."""

from .models import Document, Chunk, RetrievalResult, Section, ReferenceItem
from .pdf_extractor import extract_pdf_from_path, extract_pdf_from_bytes
from .chunker import create_chunks
from .store import DocumentStore

__all__ = [
    "Document",
    "Chunk",
    "RetrievalResult",
    "Section",
    "ReferenceItem",
    "extract_pdf_from_path",
    "extract_pdf_from_bytes",
    "create_chunks",
    "DocumentStore",
]

__version__ = "0.1.0"
