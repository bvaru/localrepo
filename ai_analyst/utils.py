from __future__ import annotations

from typing import Iterable, List
import re


def normalize_title(candidate: str | None, fallback: str) -> str:
    if candidate:
        t = candidate.strip()
        if t:
            return t
    return fallback


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "doc"
