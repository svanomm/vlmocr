"""Tests for vlmocr OCR conversion cleaning behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlmocr.conversion import clean_file


def _build_raw_json(path: Path, repeated_line: str, page_count: int = 12) -> None:
    """Create a synthetic raw OCR JSON file with repeated lines."""
    pages = [
        {"index": i, "markdown": f"{repeated_line}\nPage {i} content."}
        for i in range(page_count)
    ]
    path.write_text(json.dumps({"pages": pages}), encoding="utf-8")


def test_clean_file_keeps_frequent_lines_by_default(tmp_path: Path) -> None:
    """Frequent line removal is disabled by default."""
    repeated_line = "Company Confidential"
    raw_json_path = tmp_path / "raw.json"
    out_dir = tmp_path / "converted"
    _build_raw_json(raw_json_path, repeated_line)

    clean_file(raw_json_path, out_dir=out_dir, out_name="doc")

    md_path = out_dir / "md" / "doc.md"
    json_path = out_dir / "json" / "doc.json"

    assert md_path.exists()
    assert json_path.exists()
    assert repeated_line in md_path.read_text(encoding="utf-8")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert all(repeated_line in page["markdown"] for page in data["pages"])


def test_clean_file_removes_frequent_lines_when_enabled(tmp_path: Path) -> None:
    """Frequent line removal occurs only when explicitly enabled."""
    repeated_line = "Company Confidential"
    raw_json_path = tmp_path / "raw.json"
    out_dir = tmp_path / "converted"
    _build_raw_json(raw_json_path, repeated_line)

    clean_file(
        raw_json_path,
        out_dir=out_dir,
        out_name="doc",
        remove_frequent_lines=True,
    )

    md_path = out_dir / "md" / "doc.md"
    json_path = out_dir / "json" / "doc.json"

    assert repeated_line not in md_path.read_text(encoding="utf-8")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert all(repeated_line not in page["markdown"] for page in data["pages"])


def test_clean_file_does_not_over_remove_small_documents(tmp_path: Path) -> None:
    """Small documents should keep repeated lines even when cleanup is enabled."""
    repeated_line = "Company Confidential"
    raw_json_path = tmp_path / "raw.json"
    out_dir = tmp_path / "converted"
    raw_json_path.write_text(
        json.dumps(
            {
                "pages": [
                    {"index": 0, "markdown": f"{repeated_line}\nPage 0 content."},
                    {"index": 1, "markdown": f"{repeated_line}\nPage 1 content."},
                ]
            }
        ),
        encoding="utf-8",
    )

    clean_file(
        raw_json_path,
        out_dir=out_dir,
        out_name="doc",
        remove_frequent_lines=True,
    )

    md_path = out_dir / "md" / "doc.md"
    json_path = out_dir / "json" / "doc.json"

    assert repeated_line in md_path.read_text(encoding="utf-8")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert all(repeated_line in page["markdown"] for page in data["pages"])


def test_clean_file_rejects_invalid_raw_ocr_payload(tmp_path: Path) -> None:
    """Conversion should fail fast when OCR output breaks the frozen schema."""
    raw_json_path = tmp_path / "raw.json"
    out_dir = tmp_path / "converted"
    raw_json_path.write_text(
        json.dumps({"pages": [{"index": 1, "markdown": "Wrong page order."}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sequential indexes"):
        clean_file(raw_json_path, out_dir=out_dir, out_name="doc")


def test_clean_file_rejects_non_json_input(tmp_path: Path) -> None:
    """Conversion should raise a clear error for syntactically invalid JSON."""
    raw_json_path = tmp_path / "raw.json"
    out_dir = tmp_path / "converted"
    raw_json_path.write_text("not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        clean_file(raw_json_path, out_dir=out_dir, out_name="doc")
