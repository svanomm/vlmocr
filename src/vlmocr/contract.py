"""Shared OCR output contract and path helpers for vlmocr."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

DEFAULT_DOCS_DIR = Path("docs")
DEFAULT_OUT_DIR = Path(".search/converted")
RAW_OCR_SUBDIR = Path("json/raw")
CLEANED_OCR_JSON_SUBDIR = Path("json")
CLEANED_MARKDOWN_SUBDIR = Path("md")
MARKDOWN_TOC_SUBDIR = Path("md/table of contents")


class RawOcrPage(TypedDict):
    """One OCR page in the raw per-page JSON payload."""

    index: int
    markdown: str


class RawOcrDocument(TypedDict):
    """The raw OCR document payload consumed by downstream conversion."""

    pages: list[RawOcrPage]


def get_raw_ocr_dir(out_dir: Path) -> Path:
    """Return the raw OCR JSON output directory under *out_dir*."""
    return out_dir / RAW_OCR_SUBDIR


def get_cleaned_ocr_json_dir(out_dir: Path) -> Path:
    """Return the cleaned OCR JSON output directory under *out_dir*."""
    return out_dir / CLEANED_OCR_JSON_SUBDIR


def get_cleaned_markdown_dir(out_dir: Path) -> Path:
    """Return the cleaned markdown output directory under *out_dir*."""
    return out_dir / CLEANED_MARKDOWN_SUBDIR


def get_markdown_toc_dir(out_dir: Path) -> Path:
    """Return the table-of-contents directory under *out_dir*."""
    return out_dir / MARKDOWN_TOC_SUBDIR


def build_raw_ocr_document(page_markdowns: list[str]) -> RawOcrDocument:
    """Build the canonical raw OCR payload from page markdown strings.

    Args:
        page_markdowns: Page markdown in document order.

    Returns:
        The canonical raw OCR payload with sequential page indexes.
    """
    return {
        "pages": [
            {"index": page_index, "markdown": markdown}
            for page_index, markdown in enumerate(page_markdowns)
        ]
    }


def validate_raw_ocr_document(payload: object) -> RawOcrDocument:
    """Validate the raw OCR payload expected by conversion.

    Args:
        payload: Decoded JSON payload to validate.

    Returns:
        The validated payload narrowed to the canonical raw OCR type.

    Raises:
        ValueError: If the payload does not match the expected schema.
    """
    if not isinstance(payload, dict):
        raise ValueError("Raw OCR payload must be a JSON object.")

    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise ValueError("Raw OCR payload must contain a 'pages' list.")

    validated_pages: list[RawOcrPage] = []
    for expected_index, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ValueError(
                f"Raw OCR page at position {expected_index} must be an object."
            )

        index = page.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError(
                f"Raw OCR page at position {expected_index} must have an integer 'index'."
            )
        if index != expected_index:
            raise ValueError(
                "Raw OCR pages must be in order with sequential indexes starting at 0."
            )

        markdown = page.get("markdown")
        if not isinstance(markdown, str):
            raise ValueError(
                f"Raw OCR page at position {expected_index} must have a string 'markdown'."
            )

        validated_pages.append({"index": index, "markdown": markdown})

    return {"pages": validated_pages}
