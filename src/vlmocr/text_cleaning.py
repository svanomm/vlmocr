"""Text-cleaning helpers for OCR markdown."""

from __future__ import annotations

import re

_TWO_PLUS_SPACES_RE = re.compile(r" {2,}")
_TWO_PLUS_NEWLINES_RE = re.compile(r"\n{2,}")


def clean_text(text: str) -> str:
    """Normalize common OCR whitespace and quote spacing issues.

    Args:
        text: Raw markdown text from OCR.

    Returns:
        Cleaned markdown text.
    """
    text = _TWO_PLUS_NEWLINES_RE.sub("\n\n", text)
    text = _TWO_PLUS_SPACES_RE.sub(" ", text)
    text = text.replace(f"{chr(34)} {chr(39)}", f"{chr(34)}{chr(34)}")
    text = text.replace(f"{chr(39)} {chr(34)}", f"{chr(39)}{chr(34)}")
    text = text.replace(f"{chr(34)} {chr(34)}", f"{chr(34)}{chr(34)}")
    text = text.replace(f"{chr(39)} {chr(39)}", f"{chr(39)}{chr(39)}")
    text = text.strip()

    if not any(char.isalnum() for char in text):
        return ""

    return text
