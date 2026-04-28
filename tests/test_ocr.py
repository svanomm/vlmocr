"""Tests for vlmocr OCR helpers."""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

import fitz
import pytest

from vlmocr.ocr import convert_file, get_pdf_info, render_page_to_image


def _create_test_pdf(num_pages: int) -> str:
    """Create a temporary PDF with the given number of pages."""
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1}")
    doc.save(path)
    doc.close()
    return path


def test_get_pdf_info() -> None:
    path = _create_test_pdf(5)
    try:
        page_count, file_size = get_pdf_info(path)
        assert page_count == 5
        assert file_size > 0
    finally:
        os.unlink(path)


def test_render_page_to_image_png() -> None:
    """render_page_to_image should return a valid base64-encoded PNG."""
    path = _create_test_pdf(3)
    try:
        doc = fitz.open(path)
        b64 = render_page_to_image(doc, 0, dpi=72, fmt="png")
        doc.close()

        raw = base64.b64decode(b64)
        assert raw[:4] == b"\x89PNG"
    finally:
        os.unlink(path)


def test_render_page_to_image_jpeg() -> None:
    """render_page_to_image should return a valid base64-encoded JPEG."""
    path = _create_test_pdf(2)
    try:
        doc = fitz.open(path)
        b64 = render_page_to_image(doc, 1, dpi=150, fmt="jpeg")
        doc.close()

        raw = base64.b64decode(b64)
        assert raw[:2] == b"\xff\xd8"
    finally:
        os.unlink(path)


def test_convert_file_invalid_max_workers(tmp_path: Path) -> None:
    """convert_file should raise ValueError when max_workers < 1."""
    path = _create_test_pdf(1)
    try:
        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            convert_file(
                client=None,
                file_path=path,
                output_dir=tmp_path,
                out_name="test",
                max_workers=0,
            )
        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            convert_file(
                client=None,
                file_path=path,
                output_dir=tmp_path,
                out_name="test",
                max_workers=-5,
            )
    finally:
        os.unlink(path)


def test_convert_file_retries_then_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """convert_file should retry failing pages and raise RuntimeError after exhausting retries."""
    import vlmocr.ocr as ocr_module

    path = _create_test_pdf(2)

    def always_fail(client, base64_image, model=None, fmt=None, max_tokens=None):
        raise ConnectionError("simulated API failure")

    monkeypatch.setattr(ocr_module, "_ocr_page", always_fail)

    try:
        with pytest.raises(RuntimeError, match="failed after"):
            convert_file(
                client=None,
                file_path=path,
                output_dir=tmp_path,
                out_name="test",
                max_workers=1,
                max_retries=2,
            )
    finally:
        os.unlink(path)
