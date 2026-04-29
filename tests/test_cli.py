"""Tests for the vlmocr command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

import vlmocr.cli as cli


def _create_test_pdf(path: Path, *, text: str = "Page 1") -> None:
    """Create a minimal PDF for CLI smoke tests."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


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


def test_main_ocr_writes_raw_json_to_selected_out_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The OCR CLI should write raw JSON under the selected output root."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    pdf_path = docs_dir / "sample.pdf"
    _create_test_pdf(pdf_path)
    out_dir = tmp_path / "converted"

    monkeypatch.setattr(cli.ocr, "create_client", lambda api_key=None: object())
    monkeypatch.setattr(
        cli.ocr,
        "_ocr_page",
        lambda client, base64_image, model=None, fmt=None, max_tokens=None: "# Page 1",
    )

    cli.main(
        [
            "ocr",
            "--docs-dir",
            str(docs_dir),
            "--out-dir",
            str(out_dir),
            "--max-workers",
            "1",
        ]
    )

    raw_json_path = out_dir / "json" / "raw" / "sample.json"
    assert raw_json_path.exists()
    assert json.loads(raw_json_path.read_text(encoding="utf-8")) == {
        "pages": [{"index": 0, "markdown": "# Page 1"}]
    }


def test_main_convert_uses_default_input_dir_and_writes_artifact_tree(
    tmp_path: Path,
) -> None:
    """The convert CLI should default to out_dir/json/raw and write all expected artifacts."""
    out_dir = tmp_path / "converted"
    raw_dir = out_dir / "json" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "sample.json").write_text(
        json.dumps(
            {
                "pages": [
                    {"index": 0, "markdown": "# Title\nBody line."},
                    {"index": 1, "markdown": "## Section\nMore body."},
                ]
            }
        ),
        encoding="utf-8",
    )

    cli.main(["convert", "--out-dir", str(out_dir)])

    assert (out_dir / "json" / "sample.json").exists()
    assert (out_dir / "md" / "sample.md").exists()
    assert (out_dir / "md" / "table of contents" / "sample_toc.md").exists()

    markdown = (out_dir / "md" / "sample.md").read_text(encoding="utf-8")
    toc = (out_dir / "md" / "table of contents" / "sample_toc.md").read_text(
        encoding="utf-8"
    )

    assert "# Title" in markdown
    assert "## Section" in toc


def test_main_without_args_launches_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invoking vlmocr with no args should open the interactive launcher."""
    launched = {"value": False}

    def fake_launch_tui() -> None:
        launched["value"] = True

    monkeypatch.setattr(cli, "launch_tui", fake_launch_tui)

    cli.main([])

    assert launched["value"] is True


def test_main_init_creates_project_structure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The init command should create the default working directories and print guidance."""
    docs_dir = tmp_path / "docs"
    out_dir = tmp_path / "converted"

    monkeypatch.setattr(cli, "DEFAULT_DOCS_DIR", docs_dir)
    monkeypatch.setattr(cli, "DEFAULT_OUT_DIR", out_dir)

    cli.main(["init"])

    captured = capsys.readouterr()

    assert docs_dir.is_dir()
    assert (out_dir / "json" / "raw").is_dir()
    assert (out_dir / "md").is_dir()
    assert "Next steps:" in captured.out
    assert "https://openrouter.ai/keys" in captured.out
    assert "OPENROUTER_API_KEY=your_key_here" in captured.out
    assert "vlmocr ocr" in captured.out


def test_main_init_rejects_custom_directory_arguments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The init command should no longer accept custom directory overrides."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["init", "--docs-dir", "docs"])

    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert "unrecognized arguments: --docs-dir docs" in captured.err


def test_main_convert_missing_input_dir_shows_first_run_guidance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Convert should explain how to initialize the workspace instead of showing a traceback."""
    out_dir = tmp_path / "converted"

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["convert", "--out-dir", str(out_dir)])

    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert "vlmocr init" in captured.err
    assert "vlmocr ocr" in captured.err
    assert str(out_dir / "json" / "raw") in captured.err


def test_launch_tui_can_run_init_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The interactive launcher should let users initialize a workspace."""
    docs_dir = tmp_path / "docs"
    out_dir = tmp_path / "converted"
    responses = iter(["1", "", "6"])
    output_lines: list[str] = []

    monkeypatch.setattr(cli, "DEFAULT_DOCS_DIR", docs_dir)
    monkeypatch.setattr(cli, "DEFAULT_OUT_DIR", out_dir)

    cli.launch_tui(
        input_fn=lambda prompt: next(responses),
        output_fn=output_lines.append,
    )

    assert docs_dir.is_dir()
    assert (out_dir / "json" / "raw").is_dir()
    assert any("interactive launcher" in line for line in output_lines)
    assert any("Next steps:" in line for line in output_lines)
    assert any("https://openrouter.ai/keys" in line for line in output_lines)


def test_launch_tui_ocr_default_options_skip_extra_questions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accepting default OCR options should skip the detailed prompt sequence."""
    docs_dir = tmp_path / "docs"
    out_dir = tmp_path / "converted"
    docs_dir.mkdir()
    prompts: list[str] = []
    output_lines: list[str] = []
    responses = iter(["2", "y", "y", "", "6"])
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "DEFAULT_DOCS_DIR", docs_dir)
    monkeypatch.setattr(cli, "DEFAULT_OUT_DIR", out_dir)

    def fake_count_pages(folder: Path, *, output_fn=print) -> float | None:
        assert folder == docs_dir
        output_fn("Total estimated: $12.3400")
        return 12.34

    def fake_ocr_documents(**kwargs: object) -> list[Path]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli.estimate_cost, "count_pages", fake_count_pages)
    monkeypatch.setattr(cli.ocr, "ocr_documents", fake_ocr_documents)

    cli.launch_tui(
        input_fn=lambda prompt: prompts.append(prompt) or next(responses),
        output_fn=output_lines.append,
    )

    assert prompts == [
        "Select an option [1-6]: ",
        "Use default OCR options? [Y/n]: ",
        "Proceed with OCR using the estimated cost above? [y/N]: ",
        "Press Enter to return to the menu...",
        "Select an option [1-6]: ",
    ]
    assert any("Total estimated: $12.3400" in line for line in output_lines)
    assert captured["docs_dir"] == docs_dir
    assert captured["out_dir"] == out_dir
    assert captured["api_key"] is None
    assert captured["model"] == cli.ocr.DEFAULT_OCR_MODEL
    assert captured["dpi"] == cli.ocr.DEFAULT_OCR_DPI
    assert captured["fmt"] == cli.ocr.DEFAULT_OCR_IMAGE_FORMAT
    assert captured["max_workers"] == cli.ocr.DEFAULT_OCR_MAX_WORKERS
    assert captured["max_retries"] == cli.ocr.DEFAULT_OCR_MAX_RETRIES


def test_launch_tui_ocr_requires_final_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The launcher should estimate cost and allow the user to cancel before OCR starts."""
    docs_dir = tmp_path / "docs"
    out_dir = tmp_path / "converted"
    docs_dir.mkdir()
    output_lines: list[str] = []
    responses = iter(["2", "y", "n", "", "6"])
    ocr_called = {"value": False}

    monkeypatch.setattr(cli, "DEFAULT_DOCS_DIR", docs_dir)
    monkeypatch.setattr(cli, "DEFAULT_OUT_DIR", out_dir)

    def fake_count_pages(folder: Path, *, output_fn=print) -> float | None:
        output_fn("Total estimated: $99.9900")
        return 99.99

    def fake_ocr_documents(**kwargs: object) -> list[Path]:
        ocr_called["value"] = True
        return []

    monkeypatch.setattr(cli.estimate_cost, "count_pages", fake_count_pages)
    monkeypatch.setattr(cli.ocr, "ocr_documents", fake_ocr_documents)

    cli.launch_tui(
        input_fn=lambda prompt: next(responses),
        output_fn=output_lines.append,
    )

    assert ocr_called["value"] is False
    assert any("Total estimated: $99.9900" in line for line in output_lines)
    assert any("OCR cancelled." in line for line in output_lines)


def test_launch_tui_ocr_custom_options_still_use_default_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Custom OCR settings should still run against the standard docs and output folders."""
    docs_dir = tmp_path / "docs"
    out_dir = tmp_path / "converted"
    docs_dir.mkdir()
    prompts: list[str] = []
    responses = iter(["2", "n", "test-key", "openai/gpt-4.1-mini", "200", "jpeg", "4", "2", "y", "", "6"])
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "DEFAULT_DOCS_DIR", docs_dir)
    monkeypatch.setattr(cli, "DEFAULT_OUT_DIR", out_dir)

    def fake_count_pages(folder: Path, *, output_fn=print) -> float | None:
        assert folder == docs_dir
        output_fn("Total estimated: $1.2300")
        return 1.23

    def fake_ocr_documents(**kwargs: object) -> list[Path]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli.estimate_cost, "count_pages", fake_count_pages)
    monkeypatch.setattr(cli.ocr, "ocr_documents", fake_ocr_documents)

    cli.launch_tui(
        input_fn=lambda prompt: prompts.append(prompt) or next(responses),
        output_fn=lambda message: None,
    )

    assert prompts == [
        "Select an option [1-6]: ",
        "Use default OCR options? [Y/n]: ",
        "OpenRouter API key (leave blank to use OPENROUTER_API_KEY or a .env file in the project root): ",
        f"Model [{cli.ocr.DEFAULT_OCR_MODEL}]: ",
        f"DPI [{cli.ocr.DEFAULT_OCR_DPI}]: ",
        f"Image format [{cli.ocr.DEFAULT_OCR_IMAGE_FORMAT}]: ",
        f"Max workers [{cli.ocr.DEFAULT_OCR_MAX_WORKERS}]: ",
        f"Max retries [{cli.ocr.DEFAULT_OCR_MAX_RETRIES}]: ",
        "Proceed with OCR using the estimated cost above? [y/N]: ",
        "Press Enter to return to the menu...",
        "Select an option [1-6]: ",
    ]
    assert captured["docs_dir"] == docs_dir
    assert captured["out_dir"] == out_dir
    assert captured["api_key"] == "test-key"
    assert captured["model"] == "openai/gpt-4.1-mini"
    assert captured["dpi"] == 200
    assert captured["fmt"] == "jpeg"
    assert captured["max_workers"] == 4
    assert captured["max_retries"] == 2


def test_launch_tui_convert_uses_default_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The convert launcher flow should stay on the standard input and output folders."""
    docs_dir = tmp_path / "docs"
    out_dir = tmp_path / "converted"
    prompts: list[str] = []
    captured: dict[str, object] = {}
    responses = iter(["3", "y", "n", "", "6"])

    monkeypatch.setattr(cli, "DEFAULT_DOCS_DIR", docs_dir)
    monkeypatch.setattr(cli, "DEFAULT_OUT_DIR", out_dir)

    def fake_convert_directory(**kwargs: object) -> list[tuple[Path, Path]]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli.conversion, "convert_directory", fake_convert_directory)

    cli.launch_tui(
        input_fn=lambda prompt: prompts.append(prompt) or next(responses),
        output_fn=lambda message: None,
    )

    assert prompts == [
        "Select an option [1-6]: ",
        "Remove repeated header/footer lines [y/N]: ",
        "Inject footnotes inline [Y/n]: ",
        "Press Enter to return to the menu...",
        "Select an option [1-6]: ",
    ]
    assert captured["input_dir"] == out_dir / "json" / "raw"
    assert captured["out_dir"] == out_dir
    assert captured["remove_frequent_lines"] is True
    assert captured["inject_footnotes"] is False


def test_launch_tui_validate_uses_default_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The validate launcher flow should inspect the standard project folders without asking."""
    docs_dir = tmp_path / "docs"
    out_dir = tmp_path / "converted"
    prompts: list[str] = []
    captured: dict[str, Path] = {}
    responses = iter(["4", "", "6"])

    monkeypatch.setattr(cli, "DEFAULT_DOCS_DIR", docs_dir)
    monkeypatch.setattr(cli, "DEFAULT_OUT_DIR", out_dir)

    def fake_print_project_status(output_fn, *, docs_dir: Path, out_dir: Path) -> None:
        captured["docs_dir"] = docs_dir
        captured["out_dir"] = out_dir

    monkeypatch.setattr(cli, "_print_project_status", fake_print_project_status)

    cli.launch_tui(
        input_fn=lambda prompt: prompts.append(prompt) or next(responses),
        output_fn=lambda message: None,
    )

    assert prompts == [
        "Select an option [1-6]: ",
        "Press Enter to return to the menu...",
        "Select an option [1-6]: ",
    ]
    assert captured["docs_dir"] == docs_dir
    assert captured["out_dir"] == out_dir


def test_main_ocr_missing_api_key_shows_setup_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """OCR should explain where to get an API key and how to configure it."""
    docs_dir = tmp_path / "docs"
    out_dir = tmp_path / "converted"
    docs_dir.mkdir()
    (docs_dir / "sample.pdf").write_bytes(b"fake pdf")

    monkeypatch.setattr(
        cli.ocr,
        "ocr_documents",
        lambda **kwargs: (_ for _ in ()).throw(
            ValueError("OPENROUTER_API_KEY is required for OCR")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["ocr", "--docs-dir", str(docs_dir), "--out-dir", str(out_dir)])

    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert "https://openrouter.ai/keys" in captured.err
    assert "OPENROUTER_API_KEY" in captured.err
    assert "--api-key" in captured.err
