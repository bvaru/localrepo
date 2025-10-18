from __future__ import annotations

from typing import Dict, Tuple, Optional


def _try_pymupdf_from_path(path: str) -> Tuple[Optional[str], Dict[int, str]]:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return None, {}
    doc = fitz.open(path)
    title = (doc.metadata or {}).get("title") or None
    pages: Dict[int, str] = {}
    for i, page in enumerate(doc, start=1):
        try:
            text = page.get_text("text")
        except Exception:
            text = page.get_text()  # fallback default
        pages[i] = text or ""
    doc.close()
    return title, pages


def _try_pymupdf_from_bytes(data: bytes) -> Tuple[Optional[str], Dict[int, str]]:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return None, {}
    doc = fitz.open(stream=data, filetype="pdf")
    title = (doc.metadata or {}).get("title") or None
    pages: Dict[int, str] = {}
    for i, page in enumerate(doc, start=1):
        try:
            text = page.get_text("text")
        except Exception:
            text = page.get_text()
        pages[i] = text or ""
    doc.close()
    return title, pages


def _try_pypdf_from_path(path: str) -> Tuple[Optional[str], Dict[int, str]]:
    try:
        from pypdf import PdfReader
    except Exception:
        return None, {}
    reader = PdfReader(path)
    title = None
    try:
        md = reader.metadata
        if md and md.title:
            title = str(md.title)
    except Exception:
        title = None
    pages: Dict[int, str] = {}
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages[i] = text
    return title, pages


def _try_pypdf_from_bytes(data: bytes) -> Tuple[Optional[str], Dict[int, str]]:
    try:
        from pypdf import PdfReader
        import io
    except Exception:
        return None, {}
    reader = PdfReader(io.BytesIO(data))
    title = None
    try:
        md = reader.metadata
        if md and md.title:
            title = str(md.title)
    except Exception:
        title = None
    pages: Dict[int, str] = {}
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages[i] = text
    return title, pages


def extract_pdf_from_path(path: str) -> Tuple[Optional[str], Dict[int, str]]:
    """Returns (title, text_by_page). Title may be None."""
    title, pages = _try_pymupdf_from_path(path)
    if pages:
        return title, pages
    title, pages = _try_pypdf_from_path(path)
    return title, pages


def extract_pdf_from_bytes(data: bytes) -> Tuple[Optional[str], Dict[int, str]]:
    """Returns (title, text_by_page). Title may be None."""
    title, pages = _try_pymupdf_from_bytes(data)
    if pages:
        return title, pages
    title, pages = _try_pypdf_from_bytes(data)
    return title, pages
