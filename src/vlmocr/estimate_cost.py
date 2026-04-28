"""OCR-only PDF page counting and cost estimation."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from vlmocr.ocr import DEFAULT_OCR_MODEL

OCR_IMAGE_TOKENS_PER_PAGE = 258
OCR_PROMPT_OVERHEAD_TOKENS = 100
OCR_OUTPUT_TOKENS_PER_PAGE = 800
OCR_INPUT_COST_PER_1M_TOKENS = 0.25
OCR_OUTPUT_COST_PER_1M_TOKENS = 1.50


def count_pages(folder: Path) -> None:
    """Count PDF files and pages in a folder, with OCR cost estimates.

    Args:
        folder: Path to the folder containing PDF files.
    """
    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {folder}")
        return

    total_pages = 0
    file_details: list[tuple[str, int]] = []
    for pdf_path in pdf_files:
        with pymupdf.open(pdf_path) as doc:
            pages = len(doc)
        file_details.append((pdf_path.name, pages))
        total_pages += pages

    ocr_input_tokens = total_pages * (
        OCR_IMAGE_TOKENS_PER_PAGE + OCR_PROMPT_OVERHEAD_TOKENS
    )
    ocr_output_tokens = total_pages * OCR_OUTPUT_TOKENS_PER_PAGE
    ocr_input_cost = ocr_input_tokens / 1_000_000 * OCR_INPUT_COST_PER_1M_TOKENS
    ocr_output_cost = ocr_output_tokens / 1_000_000 * OCR_OUTPUT_COST_PER_1M_TOKENS
    total_cost = ocr_input_cost + ocr_output_cost

    max_name_len = max(len(name) for name, _ in file_details)
    header = f"{'File':<{max_name_len}}  {'Pages':>5}"
    print(f"\nPDF Report for: {folder}\n")
    print(header)
    print("-" * len(header))
    for name, pages in file_details:
        print(f"{name:<{max_name_len}}  {pages:>5}")
    print("-" * len(header))
    print(f"{'Total files:':<{max_name_len}}  {len(pdf_files):>5}")
    print(f"{'Total pages:':<{max_name_len}}  {total_pages:>5}")

    print("\n--- OCR Cost Estimates ---")
    print(f"OCR model:     {DEFAULT_OCR_MODEL}")
    print(
        f"{'OCR input:':<{max_name_len}}  ${ocr_input_cost:.4f}"
        f"  ({ocr_input_tokens:,} tokens @ ${OCR_INPUT_COST_PER_1M_TOKENS}/1M)"
    )
    print(
        f"{'OCR output:':<{max_name_len}}  ${ocr_output_cost:.4f}"
        f"  ({ocr_output_tokens:,} tokens @ ${OCR_OUTPUT_COST_PER_1M_TOKENS}/1M)"
    )
    print(f"{'Total estimated:':<{max_name_len}}  ${total_cost:.4f}")
