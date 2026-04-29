"""OCR-only PDF page counting and cost estimation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pymupdf

from vlmocr.ocr import DEFAULT_OCR_MODEL

OCR_INPUT_TOKENS_PER_PAGE = 1400
OCR_OUTPUT_TOKENS_PER_PAGE = 800
OCR_INPUT_COST_PER_1M_TOKENS = 0.25
OCR_OUTPUT_COST_PER_1M_TOKENS = 1.50

OutputFunc = Callable[[str], None]


def count_pages(folder: Path, *, output_fn: OutputFunc = print) -> float | None:
    """Count PDF files and pages in a folder, with OCR cost estimates.

    Args:
        folder: Path to the folder containing PDF files.

    Returns:
        Estimated total OCR cost, or ``None`` when no PDFs are found.
    """
    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        output_fn(f"No PDF files found in {folder}")
        return None

    total_pages = 0
    file_details: list[tuple[str, int]] = []
    for pdf_path in pdf_files:
        with pymupdf.open(pdf_path) as doc:
            pages = len(doc)
        file_details.append((pdf_path.name, pages))
        total_pages += pages

    ocr_input_tokens = total_pages * OCR_INPUT_TOKENS_PER_PAGE
    ocr_output_tokens = total_pages * OCR_OUTPUT_TOKENS_PER_PAGE
    ocr_input_cost = ocr_input_tokens / 1_000_000 * OCR_INPUT_COST_PER_1M_TOKENS
    ocr_output_cost = ocr_output_tokens / 1_000_000 * OCR_OUTPUT_COST_PER_1M_TOKENS
    total_cost = ocr_input_cost + ocr_output_cost

    max_name_len = max(len(name) for name, _ in file_details)
    header = f"{'File':<{max_name_len}}  {'Pages':>5}"
    output_fn("")
    output_fn(f"PDF Report for: {folder}")
    output_fn("")
    output_fn(header)
    output_fn("-" * len(header))
    for name, pages in file_details:
        output_fn(f"{name:<{max_name_len}}  {pages:>5}")
    output_fn("-" * len(header))
    output_fn(f"{'Total files:':<{max_name_len}}  {len(pdf_files):>5}")
    output_fn(f"{'Total pages:':<{max_name_len}}  {total_pages:>5}")

    output_fn("")
    output_fn("--- OCR Cost Estimates ---")
    output_fn(f"OCR model:     {DEFAULT_OCR_MODEL}")
    output_fn(
        f"{'OCR input:':<{max_name_len}}  ${ocr_input_cost:.4f}"
        f"  ({ocr_input_tokens:,} tokens @ ${OCR_INPUT_COST_PER_1M_TOKENS}/1M)"
    )
    output_fn(
        f"{'OCR output:':<{max_name_len}}  ${ocr_output_cost:.4f}"
        f"  ({ocr_output_tokens:,} tokens @ ${OCR_OUTPUT_COST_PER_1M_TOKENS}/1M)"
    )
    output_fn(f"{'Total estimated:':<{max_name_len}}  ${total_cost:.4f}")
    return total_cost
