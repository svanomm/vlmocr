"""Command-line interface for vlmocr."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from vlmocr import conversion, estimate_cost, ocr
from vlmocr.contract import DEFAULT_DOCS_DIR, DEFAULT_OUT_DIR, get_raw_ocr_dir


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level vlmocr argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="vlmocr",
        description="PDF OCR, conversion, and OCR cost estimation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ocr_parser = subparsers.add_parser(
        "ocr",
        help="Render PDFs and write raw per-page OCR JSON.",
    )
    ocr_parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    ocr_parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ocr_parser.add_argument("--api-key", default=None)
    ocr_parser.add_argument("--model", default=ocr.DEFAULT_OCR_MODEL)
    ocr_parser.add_argument("--dpi", type=int, default=ocr.DEFAULT_OCR_DPI)
    ocr_parser.add_argument(
        "--format",
        choices=["png", "jpeg"],
        default=ocr.DEFAULT_OCR_IMAGE_FORMAT,
    )
    ocr_parser.add_argument(
        "--max-workers",
        type=int,
        default=ocr.DEFAULT_OCR_MAX_WORKERS,
    )
    ocr_parser.add_argument(
        "--max-retries",
        type=int,
        default=ocr.DEFAULT_OCR_MAX_RETRIES,
    )

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert raw OCR JSON into cleaned markdown and cleaned JSON.",
    )
    convert_parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing raw OCR JSON. Defaults to <out-dir>/json/raw.",
    )
    convert_parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    convert_parser.add_argument(
        "--remove-frequent-lines",
        action="store_true",
        help="Remove repeated header/footer-like lines that appear on many pages.",
    )
    convert_parser.add_argument(
        "--no-inject-footnotes",
        dest="inject_footnotes",
        action="store_false",
        help="Leave <ref> and <note> tags in place instead of expanding them inline.",
    )
    convert_parser.set_defaults(inject_footnotes=True)

    estimate_parser = subparsers.add_parser(
        "estimate-cost",
        help="Estimate OCR-only page and token costs.",
    )
    estimate_parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the vlmocr CLI.

    Args:
        argv: Optional argument list.
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "ocr":
        ocr.ocr_documents(
            docs_dir=args.docs_dir,
            out_dir=args.out_dir,
            api_key=args.api_key,
            model=args.model,
            dpi=args.dpi,
            fmt=args.format,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
        )
        return

    if args.command == "convert":
        conversion.convert_directory(
            input_dir=args.input_dir or get_raw_ocr_dir(args.out_dir),
            out_dir=args.out_dir,
            remove_frequent_lines=args.remove_frequent_lines,
            inject_footnotes=args.inject_footnotes,
        )
        return

    if args.command == "estimate-cost":
        estimate_cost.count_pages(args.docs_dir)
        return

    parser.error(f"Unsupported command: {args.command}")
