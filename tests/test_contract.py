"""Tests for the raw OCR contract shared across vlmocr commands."""

from pathlib import Path

import pytest

from vlmocr.contract import (
    build_raw_ocr_document,
    get_cleaned_markdown_dir,
    get_cleaned_ocr_json_dir,
    get_markdown_toc_dir,
    get_raw_ocr_dir,
    initialize_project_structure,
    validate_project_structure,
    validate_raw_ocr_document,
)


def test_build_raw_ocr_document_creates_sequential_pages() -> None:
    """Building raw OCR output should produce the canonical page schema."""
    payload = build_raw_ocr_document(["# Page 1", "# Page 2"])

    assert payload == {
        "pages": [
            {"index": 0, "markdown": "# Page 1"},
            {"index": 1, "markdown": "# Page 2"},
        ]
    }


def test_validate_raw_ocr_document_accepts_canonical_payload() -> None:
    """The validator should accept the expected OCR schema."""
    payload = {"pages": [{"index": 0, "markdown": "Page one."}]}

    assert validate_raw_ocr_document(payload) == payload


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "JSON object"),
        ({}, "'pages' list"),
        ({"pages": ["bad"]}, "must be an object"),
        ({"pages": [{"markdown": "missing index"}]}, "integer 'index'"),
        ({"pages": [{"index": 1, "markdown": "wrong order"}]}, "sequential indexes"),
        ({"pages": [{"index": 0, "markdown": None}]}, "string 'markdown'"),
    ],
)
def test_validate_raw_ocr_document_rejects_invalid_payloads(
    payload: object, message: str
) -> None:
    """The validator should reject payloads that would break conversion."""
    with pytest.raises(ValueError, match=message):
        validate_raw_ocr_document(payload)


def test_initialize_project_structure_creates_expected_directories(
    tmp_path: Path,
) -> None:
    """The init helpers should create and validate the standard project tree."""
    docs_dir = tmp_path / "docs"
    out_dir = tmp_path / "converted"

    initialize_project_structure(docs_dir=docs_dir, out_dir=out_dir)

    assert docs_dir.is_dir()
    assert out_dir.is_dir()
    assert get_cleaned_ocr_json_dir(out_dir).is_dir()
    assert get_raw_ocr_dir(out_dir).is_dir()
    assert get_cleaned_markdown_dir(out_dir).is_dir()
    assert get_markdown_toc_dir(out_dir).is_dir()
    assert all(
        status.exists
        for status in validate_project_structure(docs_dir=docs_dir, out_dir=out_dir)
    )
