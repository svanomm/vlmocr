"""Shared OCR output contract and path helpers for vlmocr."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

DEFAULT_DOCS_DIR = Path("docs")
DEFAULT_OUT_DIR = Path(".search/converted")
RAW_OCR_SUBDIR = Path("json/raw")
CLEANED_OCR_JSON_SUBDIR = Path("json")
CLEANED_MARKDOWN_SUBDIR = Path("md")
MARKDOWN_TOC_SUBDIR = Path("md/table of contents")


@dataclass(frozen=True)
class ProjectPathStatus:
    """Validation status for one expected project directory."""

    label: str
    path: Path
    exists: bool


class RawOcrPage(TypedDict):
    """One OCR page in the raw per-page JSON payload."""

    index: int
    markdown: str


class RawOcrDocument(TypedDict):
    """The raw OCR document payload consumed by downstream conversion."""

    settings_hash: str
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


def get_project_directories(
    docs_dir: Path = DEFAULT_DOCS_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Path]:
    """Return the directories that make up the standard vlmocr workspace."""
    return {
        "docs": docs_dir,
        "output root": out_dir,
        "raw OCR JSON": get_raw_ocr_dir(out_dir),
        "cleaned OCR JSON": get_cleaned_ocr_json_dir(out_dir),
        "cleaned markdown": get_cleaned_markdown_dir(out_dir),
        "markdown table of contents": get_markdown_toc_dir(out_dir),
    }


def initialize_project_structure(
    docs_dir: Path = DEFAULT_DOCS_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> list[Path]:
    """Create the standard directory structure for a vlmocr project."""
    created_paths: list[Path] = []
    for path in get_project_directories(docs_dir=docs_dir, out_dir=out_dir).values():
        if path.exists():
            continue
        path.mkdir(parents=True, exist_ok=True)
        created_paths.append(path)
    return created_paths


def validate_project_structure(
    docs_dir: Path = DEFAULT_DOCS_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> list[ProjectPathStatus]:
    """Report whether the standard vlmocr workspace directories exist."""
    return [
        ProjectPathStatus(label=label, path=path, exists=path.exists())
        for label, path in get_project_directories(docs_dir=docs_dir, out_dir=out_dir).items()
    ]


def build_raw_ocr_document(page_markdowns: list[str], *, settings_hash: str) -> RawOcrDocument:
    """Build the canonical raw OCR payload from page markdown strings.

    Args:
        page_markdowns: Page markdown in document order.
        settings_hash: Hash of the OCR settings used to produce the pages.

    Returns:
        The canonical raw OCR payload with sequential page indexes.
    """
    payload: RawOcrDocument = {
        "settings_hash": settings_hash,
        "pages": [
            {"index": page_index, "markdown": markdown}
            for page_index, markdown in enumerate(page_markdowns)
        ]
    }

    return payload


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

    settings_hash = payload.get("settings_hash")
    if not isinstance(settings_hash, str):
        raise ValueError(
            "Raw OCR payload must have a string 'settings_hash'."
        )

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

    return {"settings_hash": settings_hash, "pages": validated_pages}
