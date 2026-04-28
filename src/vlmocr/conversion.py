"""Raw OCR JSON to cleaned markdown and cleaned JSON conversion."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from vlmocr.contract import (
    DEFAULT_OUT_DIR,
    get_cleaned_markdown_dir,
    get_cleaned_ocr_json_dir,
    get_markdown_toc_dir,
    get_raw_ocr_dir,
    validate_raw_ocr_document,
)
from vlmocr.text_cleaning import clean_text


def _inject_footnotes(markdown: str) -> str:
    """Inject footnote text inline after each reference tag.

    Args:
        markdown: Combined markdown text containing ref and note tags.

    Returns:
        Markdown with footnote text injected inline at each reference site.
    """
    note_pattern = re.compile(r'<note num="(\d+)">(.*?)</note>', re.DOTALL)
    notes: dict[str, str] = {
        match.group(1): match.group(2).strip()
        for match in note_pattern.finditer(markdown)
    }

    def _replace_ref(match: re.Match[str]) -> str:
        num = match.group(1)
        note_text = notes.get(num, "")
        if note_text:
            return f" [Footnote {num}: {note_text}]"
        return match.group(0)

    result = re.sub(r'<ref num="(\d+)"/>', _replace_ref, markdown)
    result = note_pattern.sub("", result)
    return result


def _remove_frequent_page_lines(
    page_markdowns: list[str],
    *,
    min_page_occurrences: int = 3,
    min_page_ratio: float = 0.5,
) -> list[str]:
    """Remove repeated header/footer-like lines that occur on many pages.

    Args:
        page_markdowns: Cleaned markdown for each page in document order.
        min_page_occurrences: Minimum number of pages a line must appear on.
        min_page_ratio: Minimum fraction of pages a line must appear on.

    Returns:
        Updated page markdown with frequent repeated lines removed.
    """
    if not page_markdowns:
        return page_markdowns

    page_line_counts: Counter[str] = Counter()
    for markdown in page_markdowns:
        unique_lines = {line.strip() for line in markdown.splitlines() if line.strip()}
        page_line_counts.update(unique_lines)

    required_pages = max(
        min_page_occurrences,
        int(len(page_markdowns) * min_page_ratio),
    )
    frequent_lines = {
        line for line, count in page_line_counts.items() if count >= required_pages
    }
    if not frequent_lines:
        return page_markdowns

    cleaned_pages: list[str] = []
    for markdown in page_markdowns:
        kept_lines = [
            line for line in markdown.splitlines() if line.strip() not in frequent_lines
        ]
        cleaned_pages.append("\n".join(kept_lines).strip())

    return cleaned_pages


def check_cleanable(input_dir: Path) -> list[Path]:
    """Return raw OCR JSON files that can be converted.

    Args:
        input_dir: Directory containing raw OCR JSON files.

    Returns:
        Raw OCR JSON paths.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    return sorted(input_dir.glob("*.json"))


def clean_file(
    raw_json_path: str | Path,
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    out_name: str | None = None,
    remove_frequent_lines: bool = False,
    inject_footnotes: bool = True,
) -> tuple[Path, Path]:
    """Clean a raw OCR JSON file and produce cleaned JSON and markdown.

    Args:
        raw_json_path: Path to the raw OCR JSON file.
        out_dir: Base output directory.
        out_name: Optional output filename stem.
        remove_frequent_lines: Whether to remove repeated header/footer lines.
        inject_footnotes: Whether to expand footnote references inline.

    Returns:
        Tuple of ``(markdown_path, cleaned_json_path)``.
    """
    raw_json_path = Path(raw_json_path)
    output_name = out_name or raw_json_path.stem
    json_dir = get_cleaned_ocr_json_dir(out_dir)
    md_dir = get_cleaned_markdown_dir(out_dir)
    toc_dir = get_markdown_toc_dir(out_dir)
    json_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)
    toc_dir.mkdir(parents=True, exist_ok=True)

    with open(raw_json_path, encoding="utf-8") as handle:
        data = validate_raw_ocr_document(json.load(handle))

    page_markdowns: list[str] = []
    for page in data.get("pages", []):
        page_markdowns.append(clean_text(page.get("markdown", "")))

    if remove_frequent_lines:
        page_markdowns = _remove_frequent_page_lines(page_markdowns)

    full_markdown: list[str] = []
    for page, cleaned in zip(data.get("pages", []), page_markdowns, strict=True):
        page["markdown"] = cleaned
        full_markdown.append(cleaned)

    combined_markdown = "\n".join(full_markdown)
    if inject_footnotes:
        combined_markdown = _inject_footnotes(combined_markdown)

    combined_markdown = re.sub(r"\n{3,}", "\n\n", combined_markdown)
    markdown_toc = "\n".join(
        line for line in combined_markdown.splitlines() if line.startswith("#")
    )

    md_path = md_dir / f"{output_name}.md"
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(combined_markdown)

    toc_path = toc_dir / f"{output_name}_toc.md"
    with open(toc_path, "w", encoding="utf-8") as handle:
        handle.write(markdown_toc)

    json_path = json_dir / f"{output_name}.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False)

    return md_path, json_path


def convert_directory(
    *,
    input_dir: Path | None = None,
    out_dir: Path = DEFAULT_OUT_DIR,
    remove_frequent_lines: bool = False,
    inject_footnotes: bool = True,
) -> list[tuple[Path, Path]]:
    """Convert every raw OCR JSON file in a directory.

    Args:
        input_dir: Raw OCR JSON directory. Defaults to ``out_dir/json/raw``.
        out_dir: Base output directory.
        remove_frequent_lines: Whether to remove repeated page lines.
        inject_footnotes: Whether to expand footnote references inline.

    Returns:
        Written markdown and cleaned JSON paths.
    """
    resolved_input_dir = input_dir or get_raw_ocr_dir(out_dir)
    raw_files = check_cleanable(resolved_input_dir)
    if not raw_files:
        print("No files need cleaning. Exiting.")
        return []

    print(f"Beginning cleaning of ({len(raw_files)}) files.")
    outputs: list[tuple[Path, Path]] = []
    for raw_file in tqdm(raw_files, desc="Cleaning files"):
        outputs.append(
            clean_file(
                raw_file,
                out_dir=out_dir,
                out_name=raw_file.stem,
                remove_frequent_lines=remove_frequent_lines,
                inject_footnotes=inject_footnotes,
            )
        )

    print("All cleaning complete.")
    return outputs
