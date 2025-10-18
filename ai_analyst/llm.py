from __future__ import annotations

from typing import List, Dict, Optional
import os


def _has_openai() -> bool:
    try:
        import openai  # type: ignore
        return bool(os.getenv("OPENAI_API_KEY"))
    except Exception:
        return False


def chat(messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.2, max_tokens: int = 600) -> Optional[str]:
    if not _has_openai():
        return None
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        # Legacy package fallback
        import openai  # type: ignore
        openai.api_key = os.getenv("OPENAI_API_KEY")
        try:
            resp = openai.ChatCompletion.create(
                model=model or "gpt-4o-mini",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp["choices"][0]["message"]["content"]
        except Exception:
            return None

    client = OpenAI()
    try:
        resp = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception:
        return None
