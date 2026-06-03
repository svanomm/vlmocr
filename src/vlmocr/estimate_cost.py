"""OCR-only mixed-document page counting and cost estimation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pymupdf

from vlmocr.ocr import (
    DEFAULT_OCR_MODEL,
    SUPPORTED_OCR_IMAGE_EXTENSIONS,
    discover_ocr_documents,
)

OCR_INPUT_TOKENS_PER_PAGE = 1400
OCR_OUTPUT_TOKENS_PER_PAGE = 800
OCR_INPUT_COST_PER_1M_TOKENS = 0.25
OCR_OUTPUT_COST_PER_1M_TOKENS = 1.50

OutputFunc = Callable[[str], None]


def _print_skipped_files(skipped_files: list[Path], *, output_fn: OutputFunc) -> None:
    if not skipped_files:
        return

    output_fn("")
    output_fn("Skipped unsupported files:")
    for skipped_path in skipped_files:
        output_fn(f"  - {skipped_path}")


def _count_pages_for_document_files(
    document_files: list[Path],
    *,
    output_fn: OutputFunc = print,
    source_label: str,
    skipped_files: list[Path] | None = None,
) -> float | None:
    """Count page equivalents for mixed document files and estimate OCR costs."""
    supported_details: list[tuple[str, str, int]] = []
    discovered_skips = list(skipped_files or [])

    total_page_equivalents = 0
    total_pdf_files = 0
    total_image_files = 0

    for document_path in sorted(document_files):
        extension = document_path.suffix.lower()
        if extension == ".pdf":
            with pymupdf.open(document_path) as doc:
                pages = len(doc)
            supported_details.append((document_path.name, "pdf", pages))
            total_page_equivalents += pages
            total_pdf_files += 1
            continue

        if extension in SUPPORTED_OCR_IMAGE_EXTENSIONS:
            supported_details.append((document_path.name, "image", 1))
            total_page_equivalents += 1
            total_image_files += 1
            continue

        discovered_skips.append(document_path)

    if not supported_details:
        output_fn(f"No supported documents found in {source_label}")
        _print_skipped_files(discovered_skips, output_fn=output_fn)
        return None

    ocr_input_tokens = total_page_equivalents * OCR_INPUT_TOKENS_PER_PAGE
    ocr_output_tokens = total_page_equivalents * OCR_OUTPUT_TOKENS_PER_PAGE
    ocr_input_cost = ocr_input_tokens / 1_000_000 * OCR_INPUT_COST_PER_1M_TOKENS
    ocr_output_cost = ocr_output_tokens / 1_000_000 * OCR_OUTPUT_COST_PER_1M_TOKENS
    total_cost = ocr_input_cost + ocr_output_cost

    max_name_len = max(len(name) for name, _, _ in supported_details)
    max_type_len = max(len(kind) for _, kind, _ in supported_details)
    header = f"{'Document':<{max_name_len}}  {'Type':<{max_type_len}}  {'Pages':>5}"
    output_fn("")
    output_fn(f"Document Report for: {source_label}")
    output_fn("")
    output_fn(header)
    output_fn("-" * len(header))
    for name, kind, pages in supported_details:
        output_fn(f"{name:<{max_name_len}}  {kind:<{max_type_len}}  {pages:>5}")
    output_fn("-" * len(header))
    output_fn(
        f"{'Total documents:':<{max_name_len}}  {'':<{max_type_len}}  {len(supported_details):>5}"
    )
    output_fn(
        f"{'PDF files:':<{max_name_len}}  {'':<{max_type_len}}  {total_pdf_files:>5}"
    )
    output_fn(
        f"{'Image files:':<{max_name_len}}  {'':<{max_type_len}}  {total_image_files:>5}"
    )
    output_fn(
        f"{'Page equivalents:':<{max_name_len}}  {'':<{max_type_len}}  {total_page_equivalents:>5}"
    )

    _print_skipped_files(discovered_skips, output_fn=output_fn)

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


def count_pages(
    folder: Path,
    *,
    recursive: bool = True,
    output_fn: OutputFunc = print,
) -> float | None:
    """Count mixed OCR input files and page equivalents in a folder.

    Args:
        folder: Path to the folder containing OCR input files.
        recursive: Whether to scan folder recursively.
        output_fn: Output callback.

    Returns:
        Estimated total OCR cost, or ``None`` when no supported inputs are found.
    """
    discovered_documents, skipped_inputs = discover_ocr_documents(
        folder,
        recursive=recursive,
    )
    scan_mode = "recursive" if recursive else "top-level only"
    return count_pages_for_files(
        [document.path for document in discovered_documents],
        output_fn=output_fn,
        source_label=f"{folder} ({scan_mode})",
        skipped_files=[skipped.path for skipped in skipped_inputs],
    )


def count_pages_for_files(
    document_files: list[Path],
    *,
    output_fn: OutputFunc = print,
    source_label: str,
    skipped_files: list[Path] | None = None,
) -> float | None:
    """Count page equivalents and estimate OCR cost for an explicit file list."""
    return _count_pages_for_document_files(
        sorted(document_files),
        output_fn=output_fn,
        source_label=source_label,
        skipped_files=skipped_files,
    )
