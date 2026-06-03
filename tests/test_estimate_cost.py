"""Tests for mixed-document OCR cost estimation."""

from __future__ import annotations

from pathlib import Path

import fitz
from PIL import Image
import pytest

from vlmocr import estimate_cost


def _create_test_pdf(path: Path, *, pages: int) -> None:
    doc = fitz.open()
    for page_index in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {page_index + 1}")
    doc.save(path)
    doc.close()


def _create_test_image(path: Path, *, fmt: str = "PNG") -> None:
    image = Image.new("RGB", (24, 24), color=(10, 120, 80))
    image.save(path, format=fmt)


def _expected_cost(page_equivalents: int) -> float:
    input_cost = (
        page_equivalents
        * estimate_cost.OCR_INPUT_TOKENS_PER_PAGE
        / 1_000_000
        * estimate_cost.OCR_INPUT_COST_PER_1M_TOKENS
    )
    output_cost = (
        page_equivalents
        * estimate_cost.OCR_OUTPUT_TOKENS_PER_PAGE
        / 1_000_000
        * estimate_cost.OCR_OUTPUT_COST_PER_1M_TOKENS
    )
    return input_cost + output_cost


def test_count_pages_for_files_mixed_inputs_and_skips(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    image_path = tmp_path / "sample.png"
    skipped_path = tmp_path / "sample.gif"

    _create_test_pdf(pdf_path, pages=2)
    _create_test_image(image_path, fmt="PNG")
    skipped_path.write_bytes(b"gif")

    output_lines: list[str] = []
    estimated_cost = estimate_cost.count_pages_for_files(
        [pdf_path, image_path, skipped_path],
        output_fn=output_lines.append,
        source_label="mixed pending set",
    )

    assert estimated_cost == pytest.approx(_expected_cost(3))
    assert any("PDF files:" in line for line in output_lines)
    assert any("Image files:" in line for line in output_lines)
    assert any("Skipped unsupported files:" in line for line in output_lines)
    assert any("sample.gif" in line for line in output_lines)


def test_count_pages_scans_recursively_by_default(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    nested_dir = docs_dir / "nested"
    nested_dir.mkdir(parents=True)

    _create_test_pdf(nested_dir / "deep.pdf", pages=1)
    _create_test_image(docs_dir / "top.jpg", fmt="JPEG")

    output_lines: list[str] = []
    estimated_cost = estimate_cost.count_pages(docs_dir, output_fn=output_lines.append)

    assert estimated_cost == pytest.approx(_expected_cost(2))
    assert any("recursive" in line for line in output_lines)


def test_count_pages_can_disable_recursive_scan(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    nested_dir = docs_dir / "nested"
    nested_dir.mkdir(parents=True)

    _create_test_pdf(nested_dir / "deep.pdf", pages=1)
    _create_test_image(docs_dir / "top.jpg", fmt="JPEG")

    estimated_cost = estimate_cost.count_pages(docs_dir, recursive=False, output_fn=lambda _: None)

    assert estimated_cost == pytest.approx(_expected_cost(1))


def test_count_pages_reports_none_when_no_supported_documents(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "skip.tiff").write_bytes(b"tiff")

    output_lines: list[str] = []
    estimated_cost = estimate_cost.count_pages(docs_dir, output_fn=output_lines.append)

    assert estimated_cost is None
    assert any("No supported documents found" in line for line in output_lines)
    assert any("skip.tiff" in line for line in output_lines)
