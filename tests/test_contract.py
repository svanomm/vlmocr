"""Tests for the raw OCR contract shared across vlmocr commands."""

import pytest

from vlmocr.contract import build_raw_ocr_document, validate_raw_ocr_document


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
