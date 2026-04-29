"""PDF rendering and OpenRouter-based page OCR."""

from __future__ import annotations

import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dotenv
import fitz
from openai import OpenAI
from tqdm import tqdm

from vlmocr.contract import (
    DEFAULT_DOCS_DIR,
    DEFAULT_OUT_DIR,
    build_raw_ocr_document,
    get_raw_ocr_dir,
)

DEFAULT_OCR_MODEL = os.environ.get(
    "VLMOCR_MODEL", "google/gemini-3.1-flash-lite-preview"
)
DEFAULT_OCR_DPI = int(os.environ.get("VLMOCR_DPI", "200"))
DEFAULT_OCR_IMAGE_FORMAT = os.environ.get("VLMOCR_IMAGE_FORMAT", "png")
DEFAULT_VLM_TEMPERATURE = 0.0
DEFAULT_OCR_MAX_TOKENS = int(os.environ.get("VLMOCR_MAX_TOKENS", "4096"))
DEFAULT_OCR_MAX_WORKERS = int(os.environ.get("VLMOCR_MAX_WORKERS", "4"))
DEFAULT_OCR_MAX_RETRIES = int(os.environ.get("VLMOCR_MAX_RETRIES", "3"))

OCR_PROMPT = """
    This image is one page of a document. Extract the content of the page verbatim
        and convert it to Markdown.
    Convert tables into standard Markdown table syntax. For complex layouts you may use HTML syntax if necessary.
    Convert section headings to Markdown headers, preserving hierarchy (e.g., #, ##, ###).
    Preserve bold/italic formatting with Markdown syntax.
    Merge line-wrapped text and undo hyphenation only when caused by line breaks.
    Preserve reading order for multi-column layouts.
    Convert math to LaTeX: $$ for display math, $ for inline math.
    Wrap code snippets in triple backticks with language hints when clear.
    For figures, charts, diagrams, or images, write a detailed description
        wrapped in <image> tags (e.g., <image>Description...</image>);
        preserve figure captions as text.
    Wrap inline footnote references in <ref> tags, e.g. <ref num="1"/>.
    Wrap footnote text in <note> tags with a `num` attribute, e.g. <note num="1">Footnote text here.</note>
    Remove only repeated running headers/footers and standalone page numbers; keep content-bearing metadata.
    Output only the Markdown, no commentary or summaries.
    """


def check_conversions(
    docs_dir: Path = DEFAULT_DOCS_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> list[Path]:
    """Check which PDF files still need OCR conversion.

    Args:
        docs_dir: Directory containing input PDF files.
        out_dir: Base output directory for OCR artifacts.

    Returns:
        PDF paths that do not yet have raw OCR JSON output.
    """
    raw_json_dir = get_raw_ocr_dir(out_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    raw_json_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(docs_dir.glob("*.pdf"))
    needs_conversion: list[Path] = []

    for pdf_path in pdf_files:
        json_file = raw_json_dir / f"{pdf_path.stem}.json"
        if not json_file.exists():
            needs_conversion.append(pdf_path)

    return needs_conversion


def create_client(api_key: str | None = None) -> OpenAI:
    """Create and return an OpenRouter API client.

    Args:
        api_key: Optional explicit API key override.

    Returns:
        Authenticated OpenAI client configured for OpenRouter.

    Raises:
        ValueError: If no API key is available.
    """
    dotenv.load_dotenv()
    resolved_api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not resolved_api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is required for OCR. Get one from https://openrouter.ai/keys and set it in your environment, pass --api-key, or add OPENROUTER_API_KEY=your_key_here to a .env file in the project root."
        )

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=resolved_api_key,
    )


def render_page_to_image(
    doc: fitz.Document,
    page_index: int,
    dpi: int = DEFAULT_OCR_DPI,
    fmt: str = DEFAULT_OCR_IMAGE_FORMAT,
) -> str:
    """Render a single PDF page to a base64-encoded image.

    Args:
        doc: An open PyMuPDF document.
        page_index: Zero-based page index to render.
        dpi: Resolution in dots per inch.
        fmt: Image format, either ``"png"`` or ``"jpeg"``.

    Returns:
        Base64-encoded image string.

    Raises:
        ValueError: If the image format is unsupported.
    """
    if fmt not in {"png", "jpeg"}:
        raise ValueError(f"Unsupported image format: {fmt}")

    page = doc[page_index]
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    image_bytes = pix.tobytes(output="jpeg" if fmt == "jpeg" else "png")
    return base64.b64encode(image_bytes).decode("utf-8")


def _ocr_page(
    client: OpenAI,
    base64_image: str,
    *,
    model: str = DEFAULT_OCR_MODEL,
    fmt: str = DEFAULT_OCR_IMAGE_FORMAT,
    max_tokens: int = DEFAULT_OCR_MAX_TOKENS,
) -> str:
    """Send a page image to a vision model and return markdown text.

    Args:
        client: OpenRouter API client.
        base64_image: Base64-encoded page image.
        model: Model identifier.
        fmt: Image format.
        max_tokens: Maximum output tokens per page.

    Returns:
        Markdown text extracted from the page image.
    """
    mime = "image/jpeg" if fmt == "jpeg" else "image/png"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{base64_image}"},
                    },
                ],
            }
        ],
        temperature=DEFAULT_VLM_TEMPERATURE,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def convert_file(
    client: OpenAI,
    file_path: str | Path,
    *,
    output_dir: Path = DEFAULT_OUT_DIR,
    out_name: str | None = None,
    model: str = DEFAULT_OCR_MODEL,
    dpi: int = DEFAULT_OCR_DPI,
    fmt: str = DEFAULT_OCR_IMAGE_FORMAT,
    max_workers: int = DEFAULT_OCR_MAX_WORKERS,
    max_retries: int = DEFAULT_OCR_MAX_RETRIES,
) -> Path:
    """Convert a PDF file to raw per-page OCR JSON.

    Args:
        client: OpenRouter API client.
        file_path: Input PDF path.
        output_dir: Base output directory.
        out_name: Optional output filename stem.
        model: Vision model identifier.
        dpi: Render DPI.
        fmt: Image format.
        max_workers: OCR worker thread count.
        max_retries: OCR retry attempts per page.

    Returns:
        Path to the written raw OCR JSON file.

    Raises:
        ValueError: If max_workers is invalid.
        RuntimeError: If OCR does not produce output for every page.
    """
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")

    file_path = Path(file_path)
    output_name = out_name or file_path.stem
    with fitz.open(file_path) as doc:
        page_count = len(doc)
        page_images = [
            render_page_to_image(doc, i, dpi=dpi, fmt=fmt) for i in range(page_count)
        ]

    page_markdowns: list[str | None] = [None] * len(page_images)

    def _ocr_indexed(page_index: int) -> tuple[int, str]:
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                return page_index, _ocr_page(
                    client,
                    page_images[page_index],
                    model=model,
                    fmt=fmt,
                )
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    continue
        raise RuntimeError(
            f"Page {page_index} of '{output_name}' failed after {max_retries} attempts"
        ) from last_exc

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_ocr_indexed, i): i for i in range(len(page_images))}
        with tqdm(
            total=len(page_images), desc=f"  OCR {output_name}", leave=False
        ) as pbar:
            for future in as_completed(futures):
                idx, markdown = future.result()
                page_markdowns[idx] = markdown
                pbar.update(1)

    if any(markdown is None for markdown in page_markdowns):
        raise RuntimeError(
            f"OCR did not produce markdown for every page of '{output_name}'"
        )

    result = build_raw_ocr_document(
        [markdown for markdown in page_markdowns if markdown is not None]
    )
    raw_ocr_dir = get_raw_ocr_dir(output_dir)
    raw_ocr_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_ocr_dir / f"{output_name}.json"
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(result, json_file, ensure_ascii=False)

    return output_path


def get_pdf_info(file_path: str | Path) -> tuple[int, int]:
    """Get page count and file size for a PDF.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Tuple of ``(page_count, file_size_bytes)``.
    """
    file_path = Path(file_path)
    with fitz.open(file_path) as doc:
        page_count = len(doc)
    return page_count, file_path.stat().st_size


def ocr_documents(
    *,
    docs_dir: Path = DEFAULT_DOCS_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    api_key: str | None = None,
    model: str = DEFAULT_OCR_MODEL,
    dpi: int = DEFAULT_OCR_DPI,
    fmt: str = DEFAULT_OCR_IMAGE_FORMAT,
    max_workers: int = DEFAULT_OCR_MAX_WORKERS,
    max_retries: int = DEFAULT_OCR_MAX_RETRIES,
) -> list[Path]:
    """OCR all pending PDFs in a directory.

    Args:
        docs_dir: Directory containing PDFs.
        out_dir: Base output directory.
        api_key: Optional OpenRouter API key override.
        model: Vision model identifier.
        dpi: Render DPI.
        fmt: Image format.
        max_workers: OCR worker thread count.
        max_retries: OCR retry attempts per page.

    Returns:
        Paths of written raw OCR JSON files.
    """
    to_convert = check_conversions(docs_dir=docs_dir, out_dir=out_dir)
    if not to_convert:
        print("No files need conversion. Exiting.")
        return []

    print(f"Beginning conversion of ({len(to_convert)}) files.")
    client = create_client(api_key=api_key)
    outputs: list[Path] = []

    for pdf_path in tqdm(to_convert, desc="Converting files"):
        outputs.append(
            convert_file(
                client,
                pdf_path,
                output_dir=out_dir,
                out_name=pdf_path.stem,
                model=model,
                dpi=dpi,
                fmt=fmt,
                max_workers=max_workers,
                max_retries=max_retries,
            )
        )

    print("All conversions complete.")
    return outputs
