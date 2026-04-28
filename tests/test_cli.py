"""Tests for the vlmocr command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest

import vlmocr.cli as cli


def test_main_dispatches_ocr_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI should dispatch the OCR command to the OCR module."""
    captured: dict[str, object] = {}

    def fake_ocr_documents(**kwargs: object) -> list[Path]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli.ocr, "ocr_documents", fake_ocr_documents)

    cli.main(["ocr", "--docs-dir", "docs", "--out-dir", "out"])

    assert captured["docs_dir"] == Path("docs")
    assert captured["out_dir"] == Path("out")


def test_main_dispatches_convert_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI should dispatch conversion arguments to the conversion module."""
    captured: dict[str, object] = {}

    def fake_convert_directory(**kwargs: object) -> list[tuple[Path, Path]]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli.conversion, "convert_directory", fake_convert_directory)

    cli.main(
        ["convert", "--input-dir", "raw", "--out-dir", "out", "--remove-frequent-lines"]
    )

    assert captured["input_dir"] == Path("raw")
    assert captured["out_dir"] == Path("out")
    assert captured["remove_frequent_lines"] is True
    assert captured["inject_footnotes"] is True


def test_main_dispatches_estimate_cost_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI should dispatch estimate-cost to the estimator module."""
    captured: dict[str, object] = {}

    def fake_count_pages(folder: Path) -> None:
        captured["folder"] = folder

    monkeypatch.setattr(cli.estimate_cost, "count_pages", fake_count_pages)

    cli.main(["estimate-cost", "--docs-dir", "docs"])

    assert captured["folder"] == Path("docs")
