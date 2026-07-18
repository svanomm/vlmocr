"""Document rendering and OpenRouter-based page OCR."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dotenv
import fitz
from PIL import Image, ImageOps
from openai import OpenAI
from tqdm import tqdm

from vlmocr.contract import (
    DEFAULT_DOCS_DIR,
    DEFAULT_OUT_DIR,
    build_raw_ocr_document,
    get_raw_ocr_dir,
    validate_raw_ocr_document,
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

SUPPORTED_OCR_INPUT_EXTENSIONS = frozenset(
    {".pdf", ".jpg", ".jpeg", ".png", ".webp", ".bmp"}
)
SUPPORTED_OCR_IMAGE_EXTENSIONS = frozenset(
    SUPPORTED_OCR_INPUT_EXTENSIONS - {".pdf"}
)
UNSUPPORTED_OCR_INPUT_EXTENSIONS = frozenset({".tif", ".tiff", ".gif"})

DEFAULT_OCR_PROMPT_TEMPLATE = "default"
PROMPT_TEMPLATE_EXTENSION = ".md"
OCR_PROMPTS_DIR = Path(__file__).with_name("prompts")
OCR_PROMPT_PATH = OCR_PROMPTS_DIR / f"{DEFAULT_OCR_PROMPT_TEMPLATE}{PROMPT_TEMPLATE_EXTENSION}"

_PROMPT_TEMPLATE_NAME_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class OCRPromptTemplate:
    """Metadata describing a stored OCR prompt template."""

    name: str
    path: Path
    description: str


@dataclass(frozen=True)
class OCRInputDocument:
    """One supported OCR input and its deterministic output stem."""

    path: Path
    output_name: str
    source_type: str


@dataclass(frozen=True)
class SkippedOCRInput:
    """One discovered input file skipped by OCR discovery."""

    path: Path
    reason: str


@dataclass(frozen=True)
class OCRPageUsage:
    """Usage metadata returned by OpenRouter for one OCR request."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float | None


OutputFunc = Callable[[str], None]


def get_ocr_prompt_path() -> Path:
    """Return the canonical markdown file path for OCR prompt instructions."""
    return OCR_PROMPT_PATH


def get_ocr_prompts_dir() -> Path:
    """Return the canonical directory containing OCR prompt templates."""
    return OCR_PROMPTS_DIR


def normalize_prompt_template_name(name: str) -> str:
    """Normalize a user-provided template name into a safe slug."""
    normalized = _PROMPT_TEMPLATE_NAME_PATTERN.sub("-", name.strip().lower()).strip("-")
    if not normalized:
        raise ValueError("Template name must include letters or numbers.")
    return normalized


def get_ocr_prompt_template_path(
    template_name: str,
    *,
    prompts_dir: Path | None = None,
) -> Path:
    """Return the markdown path for a named OCR prompt template."""
    normalized_name = normalize_prompt_template_name(template_name)
    base_dir = prompts_dir or OCR_PROMPTS_DIR
    return base_dir / f"{normalized_name}{PROMPT_TEMPLATE_EXTENSION}"


def _parse_front_matter(markdown: str) -> tuple[dict[str, str], str]:
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, markdown

    end_index = None
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = idx
            break

    if end_index is None:
        return {}, markdown

    metadata: dict[str, str] = {}
    for line in lines[1:end_index]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition(":")
        if not separator:
            continue
        metadata[key.strip().lower()] = value.strip().strip('"').strip("'")

    body = "\n".join(lines[end_index + 1 :])
    return metadata, body


def _extract_template_description(markdown: str) -> str:
    metadata, _ = _parse_front_matter(markdown)
    description = metadata.get("description", "").strip()
    return description or "No description provided."


def _render_template_markdown(*, description: str, prompt: str) -> str:
    normalized_description = description.strip() or "User-defined OCR prompt template."
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise ValueError("OCR prompt cannot be empty.")

    return "\n".join(
        [
            "---",
            f"description: {normalized_description}",
            "---",
            "",
            normalized_prompt,
            "",
        ]
    )


def list_ocr_prompt_templates(prompts_dir: Path | None = None) -> list[OCRPromptTemplate]:
    """List available OCR prompt templates and their descriptions."""
    base_dir = prompts_dir or OCR_PROMPTS_DIR
    if not base_dir.exists():
        raise ValueError(f"OCR prompt templates directory not found: {base_dir}")

    templates: list[OCRPromptTemplate] = []
    for path in sorted(base_dir.glob(f"*{PROMPT_TEMPLATE_EXTENSION}")):
        try:
            markdown = path.read_text(encoding="utf-8")
        except OSError:
            continue

        templates.append(
            OCRPromptTemplate(
                name=path.stem,
                path=path,
                description=_extract_template_description(markdown),
            )
        )

    if not templates:
        raise ValueError(f"No OCR prompt templates found in {base_dir}")
    return templates


def read_ocr_prompt_template(
    template_name: str = DEFAULT_OCR_PROMPT_TEMPLATE,
    *,
    prompts_dir: Path | None = None,
) -> str:
    """Read OCR prompt instructions from a named template."""
    return read_ocr_prompt(
        get_ocr_prompt_template_path(template_name, prompts_dir=prompts_dir)
    )


def create_ocr_prompt_template(
    *,
    template_name: str,
    description: str,
    prompt: str,
    prompts_dir: Path | None = None,
    overwrite: bool = False,
) -> OCRPromptTemplate:
    """Create or update an OCR prompt template markdown file."""
    base_dir = prompts_dir or OCR_PROMPTS_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    normalized_name = normalize_prompt_template_name(template_name)
    template_path = base_dir / f"{normalized_name}{PROMPT_TEMPLATE_EXTENSION}"
    if template_path.exists() and not overwrite:
        raise ValueError(
            f"OCR prompt template already exists: {template_path}. Choose another name or overwrite it."
        )

    template_markdown = _render_template_markdown(
        description=description,
        prompt=prompt,
    )
    template_path.write_text(template_markdown, encoding="utf-8")

    return OCRPromptTemplate(
        name=normalized_name,
        path=template_path,
        description=description.strip() or "User-defined OCR prompt template.",
    )


def read_ocr_prompt(prompt_path: Path | None = None) -> str:
    """Read OCR prompt instructions from the markdown prompt file."""
    target_path = prompt_path or OCR_PROMPT_PATH

    try:
        prompt_text = target_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"OCR prompt file not found: {target_path}") from exc

    _, prompt_body = _parse_front_matter(prompt_text)
    normalized_prompt = prompt_body.strip()
    if not normalized_prompt:
        raise ValueError(f"OCR prompt file is empty: {target_path}")

    return normalized_prompt


def write_ocr_prompt(prompt: str, prompt_path: Path | None = None) -> None:
    """Write OCR prompt instructions to the markdown prompt file."""
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise ValueError("OCR prompt cannot be empty.")

    target_path = prompt_path or OCR_PROMPT_PATH

    # Preserve template metadata when editing files in the managed templates folder.
    if target_path.parent == OCR_PROMPTS_DIR:
        description = ""
        if target_path.exists():
            try:
                existing_text = target_path.read_text(encoding="utf-8")
            except OSError:
                existing_text = ""

            parsed_description = _extract_template_description(existing_text)
            if parsed_description != "No description provided.":
                description = parsed_description

        target_path.write_text(
            _render_template_markdown(description=description, prompt=normalized_prompt),
            encoding="utf-8",
        )
        return

    target_path.write_text(f"{normalized_prompt}\n", encoding="utf-8")


def _resolve_prompt(prompt: str | None) -> str:
    """Resolve prompt override or fallback to the markdown prompt file."""
    if prompt is None:
        return read_ocr_prompt()

    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise ValueError("OCR prompt cannot be empty.")
    return normalized_prompt


OCR_PROMPT = read_ocr_prompt()


def build_ocr_settings(
    *,
    model: str = DEFAULT_OCR_MODEL,
    dpi: int = DEFAULT_OCR_DPI,
    fmt: str = DEFAULT_OCR_IMAGE_FORMAT,
    prompt: str | None = None,
    temperature: float = DEFAULT_VLM_TEMPERATURE,
    max_tokens: int = DEFAULT_OCR_MAX_TOKENS,
) -> dict[str, str | int | float]:
    """Build the canonical OCR settings payload used for hashing."""
    resolved_prompt = _resolve_prompt(prompt)
    return {
        "model": model,
        "dpi": dpi,
        "image_format": fmt,
        "prompt": resolved_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def hash_ocr_settings(
    *,
    model: str = DEFAULT_OCR_MODEL,
    dpi: int = DEFAULT_OCR_DPI,
    fmt: str = DEFAULT_OCR_IMAGE_FORMAT,
    prompt: str | None = None,
    temperature: float = DEFAULT_VLM_TEMPERATURE,
    max_tokens: int = DEFAULT_OCR_MAX_TOKENS,
) -> str:
    """Return a stable hash for the OCR settings that affect output."""
    serialized = json.dumps(
        build_ocr_settings(
            model=model,
            dpi=dpi,
            fmt=fmt,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _raw_ocr_matches_settings(raw_json_path: Path, *, settings_hash: str) -> bool:
    """Return whether an existing raw OCR payload matches the current settings."""
    if not raw_json_path.exists():
        return False

    try:
        with open(raw_json_path, encoding="utf-8") as handle:
            payload = validate_raw_ocr_document(json.load(handle))
    except (OSError, json.JSONDecodeError, ValueError):
        return False

    return payload["settings_hash"] == settings_hash


def _path_sort_key(path: Path, *, docs_dir: Path) -> str:
    """Return a deterministic lowercase relative path sort key."""
    return path.relative_to(docs_dir).as_posix().lower()


def _resolve_output_names(paths: list[Path], *, docs_dir: Path) -> dict[Path, str]:
    """Resolve deterministic output names with extension suffixes for collisions."""
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        by_stem[path.stem].append(path)

    resolved: dict[Path, str] = {}
    for stem, grouped_paths in by_stem.items():
        if len(grouped_paths) == 1:
            resolved[grouped_paths[0]] = stem
            continue

        for path in grouped_paths:
            extension_suffix = path.suffix.lower().lstrip(".") or "file"
            resolved[path] = f"{stem}__{extension_suffix}"

    by_output_name: dict[str, list[Path]] = defaultdict(list)
    for path, output_name in resolved.items():
        by_output_name[output_name].append(path)

    for output_name, grouped_paths in by_output_name.items():
        if len(grouped_paths) == 1:
            continue

        for index, path in enumerate(
            sorted(grouped_paths, key=lambda item: _path_sort_key(item, docs_dir=docs_dir)),
            start=1,
        ):
            resolved[path] = f"{output_name}__{index}"

    return resolved


def discover_ocr_documents(
    docs_dir: Path = DEFAULT_DOCS_DIR,
    *,
    recursive: bool = True,
) -> tuple[list[OCRInputDocument], list[SkippedOCRInput]]:
    """Discover supported OCR inputs and explicitly skipped unsupported files."""
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    scan_iter = docs_dir.rglob("*") if recursive else docs_dir.glob("*")
    file_paths = sorted(
        (path for path in scan_iter if path.is_file()),
        key=lambda item: _path_sort_key(item, docs_dir=docs_dir),
    )

    supported_paths: list[Path] = []
    skipped_inputs: list[SkippedOCRInput] = []
    for path in file_paths:
        extension = path.suffix.lower()
        if extension in SUPPORTED_OCR_INPUT_EXTENSIONS:
            supported_paths.append(path)
            continue
        if extension in UNSUPPORTED_OCR_INPUT_EXTENSIONS:
            skipped_inputs.append(
                SkippedOCRInput(
                    path=path,
                    reason="format not supported yet",
                )
            )

    output_names = _resolve_output_names(supported_paths, docs_dir=docs_dir)
    discovered_documents = [
        OCRInputDocument(
            path=path,
            output_name=output_names[path],
            source_type="pdf" if path.suffix.lower() == ".pdf" else "image",
        )
        for path in supported_paths
    ]
    return discovered_documents, skipped_inputs


def _emit_skipped_input_warnings(
    skipped_inputs: list[SkippedOCRInput], *, output_fn: OutputFunc
) -> None:
    for skipped in skipped_inputs:
        output_fn(
            f"Skipping unsupported document format: {skipped.path} ({skipped.reason})."
        )


def check_conversions(
    docs_dir: Path = DEFAULT_DOCS_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    model: str = DEFAULT_OCR_MODEL,
    dpi: int = DEFAULT_OCR_DPI,
    fmt: str = DEFAULT_OCR_IMAGE_FORMAT,
    prompt: str | None = None,
    temperature: float = DEFAULT_VLM_TEMPERATURE,
    max_tokens: int = DEFAULT_OCR_MAX_TOKENS,
    recursive: bool = True,
    output_fn: OutputFunc = print,
) -> list[OCRInputDocument]:
    """Check which discovered documents still need OCR conversion.

    Args:
        docs_dir: Directory containing OCR input files.
        out_dir: Base output directory for OCR artifacts.
        model: Vision model identifier.
        dpi: Render DPI.
        fmt: Image format.
        prompt: Optional OCR prompt override. Defaults to the markdown prompt file.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens per page.
        recursive: Whether to scan docs_dir recursively.
        output_fn: Function used for skip warnings.

    Returns:
        Input documents that do not yet have raw OCR JSON output for the current settings.
    """
    raw_json_dir = get_raw_ocr_dir(out_dir)
    raw_json_dir.mkdir(parents=True, exist_ok=True)
    discovered_documents, skipped_inputs = discover_ocr_documents(
        docs_dir=docs_dir,
        recursive=recursive,
    )
    _emit_skipped_input_warnings(skipped_inputs, output_fn=output_fn)

    needs_conversion: list[OCRInputDocument] = []
    current_settings_hash = hash_ocr_settings(
        model=model,
        dpi=dpi,
        fmt=fmt,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    for document in discovered_documents:
        json_file = raw_json_dir / f"{document.output_name}.json"
        if not _raw_ocr_matches_settings(
            json_file, settings_hash=current_settings_hash
        ):
            needs_conversion.append(document)

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


def _validate_request_image_format(fmt: str) -> str:
    normalized_format = fmt.lower()
    if normalized_format not in {"png", "jpeg"}:
        raise ValueError(f"Unsupported image format: {fmt}")
    return normalized_format


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
    normalized_format = _validate_request_image_format(fmt)

    page = doc[page_index]
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    image_bytes = pix.tobytes(output="jpeg" if normalized_format == "jpeg" else "png")
    return base64.b64encode(image_bytes).decode("utf-8")


def render_image_to_image(
    file_path: str | Path,
    *,
    fmt: str = DEFAULT_OCR_IMAGE_FORMAT,
) -> str:
    """Render an image file to normalized base64-encoded PNG or JPEG."""
    normalized_format = _validate_request_image_format(fmt)
    file_path = Path(file_path)

    with Image.open(file_path) as image:
        oriented = ImageOps.exif_transpose(image)
        rgb_image = oriented.convert("RGB")
        image_buffer = io.BytesIO()
        rgb_image.save(
            image_buffer,
            format="JPEG" if normalized_format == "jpeg" else "PNG",
        )
        image_bytes = image_buffer.getvalue()

    return base64.b64encode(image_bytes).decode("utf-8")


def render_document_to_images(
    file_path: str | Path,
    *,
    dpi: int = DEFAULT_OCR_DPI,
    fmt: str = DEFAULT_OCR_IMAGE_FORMAT,
) -> list[str]:
    """Render a supported document into one or more base64-encoded page images."""
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    if extension == ".pdf":
        with fitz.open(file_path) as doc:
            return [
                render_page_to_image(doc, page_index, dpi=dpi, fmt=fmt)
                for page_index in range(len(doc))
            ]

    if extension in SUPPORTED_OCR_IMAGE_EXTENSIONS:
        return [render_image_to_image(file_path, fmt=fmt)]

    raise ValueError(
        f"Unsupported OCR input format for '{file_path.name}': {file_path.suffix or '<none>'}"
    )


def _coerce_usage_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    if isinstance(value, str):
        try:
            return max(0, int(float(value)))
        except ValueError:
            return 0
    return 0


def _coerce_usage_cost(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    if isinstance(value, str):
        try:
            return max(0.0, float(value))
        except ValueError:
            return None
    return None


def _extract_openrouter_usage(response: Any) -> OCRPageUsage:
    """Extract usage and cost metadata from an OpenRouter chat completion response."""
    usage_payload: dict[str, Any] = {}

    if hasattr(response, "model_dump"):
        dumped = response.model_dump(mode="json")
        if isinstance(dumped, dict):
            maybe_usage = dumped.get("usage")
            if isinstance(maybe_usage, dict):
                usage_payload = maybe_usage

    if not usage_payload:
        usage_obj = getattr(response, "usage", None)
        if hasattr(usage_obj, "model_dump"):
            dumped_usage = usage_obj.model_dump(mode="json")
            if isinstance(dumped_usage, dict):
                usage_payload = dumped_usage
        elif isinstance(usage_obj, dict):
            usage_payload = usage_obj

    prompt_tokens = _coerce_usage_int(usage_payload.get("prompt_tokens"))
    completion_tokens = _coerce_usage_int(usage_payload.get("completion_tokens"))
    total_tokens = _coerce_usage_int(usage_payload.get("total_tokens"))
    cost = _coerce_usage_cost(usage_payload.get("cost"))

    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    return OCRPageUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost=cost,
    )


def _ocr_page_with_usage(
    client: OpenAI,
    base64_image: str,
    *,
    model: str = DEFAULT_OCR_MODEL,
    fmt: str = DEFAULT_OCR_IMAGE_FORMAT,
    prompt: str | None = None,
    temperature: float = DEFAULT_VLM_TEMPERATURE,
    max_tokens: int = DEFAULT_OCR_MAX_TOKENS,
) -> tuple[str, OCRPageUsage]:
    """Send a page image to a vision model and return markdown plus OpenRouter usage."""
    resolved_prompt = _resolve_prompt(prompt)
    mime = "image/jpeg" if fmt == "jpeg" else "image/png"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": resolved_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{base64_image}"},
                    },
                ],
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    markdown = response.choices[0].message.content or ""
    usage = _extract_openrouter_usage(response)
    return markdown, usage


def _ocr_page(
    client: OpenAI,
    base64_image: str,
    *,
    model: str = DEFAULT_OCR_MODEL,
    fmt: str = DEFAULT_OCR_IMAGE_FORMAT,
    prompt: str | None = None,
    temperature: float = DEFAULT_VLM_TEMPERATURE,
    max_tokens: int = DEFAULT_OCR_MAX_TOKENS,
) -> str:
    """Send a page image to a vision model and return markdown text.

    Args:
        client: OpenRouter API client.
        base64_image: Base64-encoded page image.
        model: Model identifier.
        fmt: Image format.
        prompt: Optional OCR prompt override. Defaults to the markdown prompt file.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens per page.

    Returns:
        Markdown text extracted from the page image.
    """
    markdown, _ = _ocr_page_with_usage(
        client,
        base64_image,
        model=model,
        fmt=fmt,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return markdown


def convert_file(
    client: OpenAI,
    file_path: str | Path,
    *,
    output_dir: Path = DEFAULT_OUT_DIR,
    out_name: str | None = None,
    model: str = DEFAULT_OCR_MODEL,
    dpi: int = DEFAULT_OCR_DPI,
    fmt: str = DEFAULT_OCR_IMAGE_FORMAT,
    prompt: str | None = None,
    temperature: float = DEFAULT_VLM_TEMPERATURE,
    max_tokens: int = DEFAULT_OCR_MAX_TOKENS,
    max_workers: int = DEFAULT_OCR_MAX_WORKERS,
    max_retries: int = DEFAULT_OCR_MAX_RETRIES,
) -> Path:
    """Convert a supported document file to raw per-page OCR JSON.

    Args:
        client: OpenRouter API client.
        file_path: Input document path (PDF or supported image).
        output_dir: Base output directory.
        out_name: Optional output filename stem.
        model: Vision model identifier.
        dpi: Render DPI.
        fmt: Image format.
        prompt: Optional OCR prompt override. Defaults to the markdown prompt file.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens per page.
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
    resolved_prompt = _resolve_prompt(prompt)
    settings_hash = hash_ocr_settings(
        model=model,
        dpi=dpi,
        fmt=fmt,
        prompt=resolved_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    page_images = render_document_to_images(file_path, dpi=dpi, fmt=fmt)

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
                    prompt=resolved_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
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
        [markdown for markdown in page_markdowns if markdown is not None],
        settings_hash=settings_hash,
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
    prompt: str | None = None,
    temperature: float = DEFAULT_VLM_TEMPERATURE,
    max_tokens: int = DEFAULT_OCR_MAX_TOKENS,
    max_workers: int = DEFAULT_OCR_MAX_WORKERS,
    max_retries: int = DEFAULT_OCR_MAX_RETRIES,
    recursive: bool = True,
    documents: list[OCRInputDocument] | None = None,
) -> list[Path]:
    """OCR all pending documents in a directory.

    Args:
        docs_dir: Directory containing supported OCR documents.
        out_dir: Base output directory.
        api_key: Optional OpenRouter API key override.
        model: Vision model identifier.
        dpi: Render DPI.
        fmt: Image format.
        prompt: Optional OCR prompt override. Defaults to the markdown prompt file.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens per page.
        max_workers: OCR worker thread count.
        max_retries: OCR retry attempts per page.
        recursive: Whether to scan docs_dir recursively.
        documents: Optional precomputed pending documents from check_conversions.

    Returns:
        Paths of written raw OCR JSON files.
    """
    to_convert = documents
    if to_convert is None:
        to_convert = check_conversions(
            docs_dir=docs_dir,
            out_dir=out_dir,
            model=model,
            dpi=dpi,
            fmt=fmt,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            recursive=recursive,
        )

    if not to_convert:
        print("No files need conversion. Exiting.")
        return []

    print(f"Beginning conversion of ({len(to_convert)}) documents.")
    client = create_client(api_key=api_key)
    outputs: list[Path] = []

    for document in tqdm(to_convert, desc="Converting documents"):
        outputs.append(
            convert_file(
                client,
                document.path,
                output_dir=out_dir,
                out_name=document.output_name,
                model=model,
                dpi=dpi,
                fmt=fmt,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                max_workers=max_workers,
                max_retries=max_retries,
            )
        )

    print("All conversions complete.")
    return outputs
