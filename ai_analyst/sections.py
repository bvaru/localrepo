from __future__ import annotations

from typing import Dict, List, Tuple
import re
from .models import Document, Section, ReferenceItem


SECTION_PATTERNS = [
    ("Abstract", r"^\s*abstract\b"),
    ("Introduction", r"^\s*introduction\b"),
    ("Background", r"^\s*background\b"),
    ("Methods", r"^\s*(materials and methods|methods|methodology)\b"),
    ("Results", r"^\s*results\b"),
    ("Discussion", r"^\s*discussion\b"),
    ("Conclusion", r"^\s*conclusion\b"),
    ("References", r"^\s*references\b"),
    ("Acknowledgements", r"^\s*(acknowledgements|acknowledgments)\b"),
]

FIGURE_PATTERN = r"^\s*(figure|fig\.)\s*\d+[:\.]?\s*"
TABLE_PATTERN = r"^\s*(table|tbl\.)\s*\d+[:\.]?\s*"


def _find_headings(text: str) -> List[Tuple[int, str]]:
    lines = text.splitlines()
    hits: List[Tuple[int, str]] = []
    for idx, line in enumerate(lines):
        lower = line.lower().strip()
        for name, pat in SECTION_PATTERNS:
            if re.match(pat, lower):
                hits.append((idx, name))
                break
    return hits


def _collect_sections(lines: List[str], hits: List[Tuple[int, str]], page_num: int) -> List[Section]:
    sections: List[Section] = []
    for i, (start_idx, name) in enumerate(hits):
        end_idx = hits[i + 1][0] if i + 1 < len(hits) else len(lines)
        body = "\n".join(lines[start_idx:end_idx]).strip()
        if body:
            sections.append(Section(name=name, text=body, page_start=page_num, page_end=page_num))
    return sections


def extract_sections(document: Document) -> List[Section]:
    sections: List[Section] = []
    for page_num in document.pages():
        text = document.text_by_page.get(page_num, "")
        lines = text.splitlines()
        hits = _find_headings(text)
        if hits:
            sections.extend(_collect_sections(lines, hits, page_num))
    # Merge consecutive same-named sections across pages
    merged: List[Section] = []
    for sec in sections:
        if merged and merged[-1].name == sec.name and merged[-1].page_end + 1 == sec.page_start:
            merged[-1].text += "\n" + sec.text
            merged[-1].page_end = sec.page_end
        else:
            merged.append(sec)
    return merged


def extract_figures_and_tables(document: Document) -> Dict[str, List[str]]:
    figures: List[str] = []
    tables: List[str] = []
    for page_num in document.pages():
        for line in document.text_by_page.get(page_num, "").splitlines():
            lower = line.lower()
            if re.match(FIGURE_PATTERN, lower):
                figures.append(line.strip())
            if re.match(TABLE_PATTERN, lower):
                tables.append(line.strip())
    return {"figures": figures, "tables": tables}


def extract_references(document: Document) -> List[ReferenceItem]:
    refs: List[ReferenceItem] = []
    # Find references section(s) and split into entries by line breaks or numbering
    sections = extract_sections(document)
    for sec in sections:
        if sec.name == "References":
            # naive split: empty lines or lines starting with numbers/author-year patterns
            block = sec.text
            entries: List[str] = []
            buff: List[str] = []
            for line in block.splitlines():
                if not line.strip():
                    if buff:
                        entries.append(" ".join(buff).strip())
                        buff = []
                    continue
                if re.match(r"^\s*(\d+\.|\[\d+\]|[A-Z][A-Za-z-]+\s*,\s*\d{4})", line):
                    if buff:
                        entries.append(" ".join(buff).strip())
                        buff = []
                    buff.append(line.strip())
                else:
                    buff.append(line.strip())
            if buff:
                entries.append(" ".join(buff).strip())
            for e in entries:
                if e:
                    refs.append(ReferenceItem(raw_text=e, page_start=sec.page_start, page_end=sec.page_end))
    return refs
