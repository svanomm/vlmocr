"""Tests for vlmocr OCR helpers."""

from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path

import fitz
import pytest
from PIL import Image

import vlmocr.ocr as ocr_module
from vlmocr.ocr import create_client, get_pdf_info, render_page_to_image


def _create_test_pdf(num_pages: int) -> str:
    """Create a temporary PDF with the given number of pages."""
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1}")
    doc.save(path)
    doc.close()
    return path


def _create_test_image(path: Path, *, fmt: str = "PNG") -> None:
    image = Image.new("RGB", (16, 12), color=(80, 120, 200))
    image.save(path, format=fmt)


def test_get_pdf_info() -> None:
    path = _create_test_pdf(5)
    try:
        page_count, file_size = get_pdf_info(path)
        assert page_count == 5
        assert file_size > 0
    finally:
        os.unlink(path)


def test_render_page_to_image_png() -> None:
    """render_page_to_image should return a valid base64-encoded PNG."""
    path = _create_test_pdf(3)
    try:
        doc = fitz.open(path)
        b64 = render_page_to_image(doc, 0, dpi=72, fmt="png")
        doc.close()

        raw = base64.b64decode(b64)
        assert raw[:4] == b"\x89PNG"
    finally:
        os.unlink(path)


def test_render_page_to_image_jpeg() -> None:
    """render_page_to_image should return a valid base64-encoded JPEG."""
    path = _create_test_pdf(2)
    try:
        doc = fitz.open(path)
        b64 = render_page_to_image(doc, 1, dpi=150, fmt="jpeg")
        doc.close()

        raw = base64.b64decode(b64)
        assert raw[:2] == b"\xff\xd8"
    finally:
        os.unlink(path)


def test_render_image_to_image_accepts_jpg_alias(tmp_path: Path) -> None:
    """render_image_to_image should treat jpg as a JPEG output alias."""
    image_path = tmp_path / "sample.jpg"
    _create_test_image(image_path, fmt="JPEG")

    b64 = ocr_module.render_image_to_image(image_path, fmt="jpg")

    raw = base64.b64decode(b64)
    assert raw[:2] == b"\xff\xd8"


def test_convert_file_invalid_max_workers(tmp_path: Path) -> None:
    """convert_file should raise ValueError when max_workers < 1."""
    path = _create_test_pdf(1)
    try:
        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            ocr_module.convert_file(
                client=None,
                file_path=path,
                output_dir=tmp_path,
                out_name="test",
                max_workers=0,
            )
        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            ocr_module.convert_file(
                client=None,
                file_path=path,
                output_dir=tmp_path,
                out_name="test",
                max_workers=-5,
            )
    finally:
        os.unlink(path)


def test_convert_file_retries_then_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """convert_file should retry failing pages and raise RuntimeError after exhausting retries."""
    import vlmocr.ocr as ocr_module

    path = _create_test_pdf(2)

    def always_fail(
        client,
        base64_image,
        model=None,
        fmt=None,
        prompt=None,
        temperature=None,
        max_tokens=None,
    ):
        raise ConnectionError("simulated API failure")

    monkeypatch.setattr(ocr_module, "_ocr_page", always_fail)

    try:
        with pytest.raises(RuntimeError, match="failed after"):
            ocr_module.convert_file(
                client=None,
                file_path=path,
                output_dir=tmp_path,
                out_name="test",
                max_workers=1,
                max_retries=2,
            )
    finally:
        os.unlink(path)


def test_create_client_requires_openrouter_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """create_client should fail clearly when no API key is configured."""
    monkeypatch.setattr(ocr_module.dotenv, "load_dotenv", lambda: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="https://openrouter.ai/keys"):
        create_client()


def test_extract_openrouter_usage_parses_cost_and_tokens() -> None:
    """Usage extraction should read usage.cost and token counts from model_dump output."""

    class FakeResponse:
        def model_dump(self, mode: str = "json") -> dict[str, object]:
            return {
                "usage": {
                    "prompt_tokens": 42,
                    "completion_tokens": 8,
                    "total_tokens": 50,
                    "cost": "0.00123",
                }
            }

    usage = ocr_module._extract_openrouter_usage(FakeResponse())

    assert usage == ocr_module.OCRPageUsage(
        prompt_tokens=42,
        completion_tokens=8,
        total_tokens=50,
        cost=0.00123,
    )


def test_extract_openrouter_usage_falls_back_to_prompt_plus_completion() -> None:
    """When total_tokens is absent, usage extraction should sum prompt and completion tokens."""

    class FakeUsage:
        def model_dump(self, mode: str = "json") -> dict[str, object]:
            return {
                "prompt_tokens": "30",
                "completion_tokens": "12",
            }

    class FakeResponse:
        usage = FakeUsage()

    usage = ocr_module._extract_openrouter_usage(FakeResponse())

    assert usage == ocr_module.OCRPageUsage(
        prompt_tokens=30,
        completion_tokens=12,
        total_tokens=42,
        cost=None,
    )


def test_convert_file_writes_raw_json_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """convert_file should emit canonical raw OCR JSON under json/raw."""
    path = _create_test_pdf(2)

    monkeypatch.setattr(
        ocr_module,
        "_ocr_page",
        lambda client,
        base64_image,
        model=None,
        fmt=None,
        prompt=None,
        temperature=None,
        max_tokens=None: "# Extracted page",
    )

    try:
        output_path = ocr_module.convert_file(
            client=object(),
            file_path=path,
            output_dir=tmp_path,
            out_name="test",
            max_workers=1,
        )
    finally:
        os.unlink(path)

    assert output_path == tmp_path / "json" / "raw" / "test.json"
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "settings_hash": ocr_module.hash_ocr_settings(),
        "pages": [
            {"index": 0, "markdown": "# Extracted page"},
            {"index": 1, "markdown": "# Extracted page"},
        ]
    }


def test_convert_file_supports_image_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """convert_file should treat a supported image as a one-page document."""
    image_path = tmp_path / "sample.png"
    _create_test_image(image_path)

    monkeypatch.setattr(
        ocr_module,
        "_ocr_page",
        lambda client,
        base64_image,
        model=None,
        fmt=None,
        prompt=None,
        temperature=None,
        max_tokens=None: "# Image page",
    )

    output_path = ocr_module.convert_file(
        client=object(),
        file_path=image_path,
        output_dir=tmp_path,
        out_name="sample",
        max_workers=1,
    )

    assert output_path == tmp_path / "json" / "raw" / "sample.json"
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "settings_hash": ocr_module.hash_ocr_settings(),
        "pages": [{"index": 0, "markdown": "# Image page"}],
    }


def test_check_conversions_skips_only_matching_settings_hash(tmp_path: Path) -> None:
    """check_conversions should only skip files whose raw JSON matches current OCR settings."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    out_dir = tmp_path / "converted"
    raw_dir = out_dir / "json" / "raw"
    raw_dir.mkdir(parents=True)

    for name in ("matching", "changed", "legacy", "missing"):
        (docs_dir / f"{name}.pdf").write_bytes(b"")

    current_hash = ocr_module.hash_ocr_settings(model="test-model", dpi=300, fmt="jpeg")
    (raw_dir / "matching.json").write_text(
        json.dumps({"settings_hash": current_hash, "pages": []}),
        encoding="utf-8",
    )
    (raw_dir / "changed.json").write_text(
        json.dumps({"settings_hash": "different", "pages": []}),
        encoding="utf-8",
    )
    (raw_dir / "legacy.json").write_text(
        json.dumps({"pages": []}),
        encoding="utf-8",
    )

    pending = ocr_module.check_conversions(
        docs_dir=docs_dir,
        out_dir=out_dir,
        model="test-model",
        dpi=300,
        fmt="jpeg",
        output_fn=lambda message: None,
    )

    assert [document.path for document in pending] == [
        docs_dir / "changed.pdf",
        docs_dir / "legacy.pdf",
        docs_dir / "missing.pdf",
    ]
    assert [document.output_name for document in pending] == [
        "changed",
        "legacy",
        "missing",
    ]


def test_check_conversions_uses_extension_suffixes_for_same_stem(tmp_path: Path) -> None:
    """Same-stem mixed inputs should receive deterministic extension-suffixed output names."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    out_dir = tmp_path / "converted"

    (docs_dir / "report.pdf").write_bytes(b"")
    (docs_dir / "report.png").write_bytes(b"")

    pending = ocr_module.check_conversions(
        docs_dir=docs_dir,
        out_dir=out_dir,
        output_fn=lambda message: None,
    )

    assert [(document.path.name, document.output_name) for document in pending] == [
        ("report.pdf", "report__pdf"),
        ("report.png", "report__png"),
    ]


def test_check_conversions_recursive_and_warns_for_skipped_formats(tmp_path: Path) -> None:
    """Discovery should recurse and warn for explicitly skipped TIFF/GIF inputs."""
    docs_dir = tmp_path / "docs"
    nested_dir = docs_dir / "nested"
    nested_dir.mkdir(parents=True)
    out_dir = tmp_path / "converted"

    (nested_dir / "deep.pdf").write_bytes(b"")
    (docs_dir / "top.jpg").write_bytes(b"")
    (docs_dir / "skip.gif").write_bytes(b"")

    warnings: list[str] = []
    pending = ocr_module.check_conversions(
        docs_dir=docs_dir,
        out_dir=out_dir,
        output_fn=warnings.append,
    )

    assert [document.path for document in pending] == [
        nested_dir / "deep.pdf",
        docs_dir / "top.jpg",
    ]
    assert any("Skipping unsupported document format" in warning for warning in warnings)


def test_discover_ocr_documents_can_disable_recursion(tmp_path: Path) -> None:
    """Non-recursive discovery should ignore nested files."""
    docs_dir = tmp_path / "docs"
    nested_dir = docs_dir / "nested"
    nested_dir.mkdir(parents=True)

    (docs_dir / "top.pdf").write_bytes(b"")
    (nested_dir / "deep.pdf").write_bytes(b"")

    recursive_docs, _ = ocr_module.discover_ocr_documents(docs_dir, recursive=True)
    top_level_docs, _ = ocr_module.discover_ocr_documents(docs_dir, recursive=False)

    assert [document.path for document in recursive_docs] == [
        nested_dir / "deep.pdf",
        docs_dir / "top.pdf",
    ]
    assert [document.path for document in top_level_docs] == [docs_dir / "top.pdf"]


def test_read_and_write_ocr_prompt_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The OCR prompt helpers should read and write markdown prompt text."""
    prompt_path = tmp_path / "ocr_prompt.md"
    prompt_path.write_text("Prompt one.\n", encoding="utf-8")
    monkeypatch.setattr(ocr_module, "OCR_PROMPT_PATH", prompt_path)

    assert ocr_module.read_ocr_prompt() == "Prompt one."

    ocr_module.write_ocr_prompt("Prompt two")

    assert prompt_path.read_text(encoding="utf-8") == "Prompt two\n"
    assert ocr_module.read_ocr_prompt() == "Prompt two"


def test_hash_ocr_settings_changes_when_prompt_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing the prompt markdown should change the OCR settings hash."""
    prompt_path = tmp_path / "ocr_prompt.md"
    prompt_path.write_text("Prompt one.\n", encoding="utf-8")
    monkeypatch.setattr(ocr_module, "OCR_PROMPT_PATH", prompt_path)

    first_hash = ocr_module.hash_ocr_settings(model="test-model", dpi=300, fmt="jpeg")
    ocr_module.write_ocr_prompt("Prompt two")
    second_hash = ocr_module.hash_ocr_settings(model="test-model", dpi=300, fmt="jpeg")

    assert first_hash != second_hash


def test_hash_ocr_settings_normalizes_jpg_alias() -> None:
    """Settings hashes should treat jpg and jpeg as the same image format."""
    assert ocr_module.hash_ocr_settings(fmt="jpg") == ocr_module.hash_ocr_settings(
        fmt="jpeg"
    )


def test_list_ocr_prompt_templates_reads_front_matter_descriptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Template listing should include filename stems and parsed descriptions."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "default.md").write_text(
        "---\ndescription: Default profile.\n---\n\nDefault prompt body.\n",
        encoding="utf-8",
    )
    (prompts_dir / "literal.md").write_text(
        "---\ndescription: Literal profile.\n---\n\nLiteral prompt body.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ocr_module, "OCR_PROMPTS_DIR", prompts_dir)

    templates = ocr_module.list_ocr_prompt_templates()

    assert [(template.name, template.description) for template in templates] == [
        ("default", "Default profile."),
        ("literal", "Literal profile."),
    ]


def test_create_ocr_prompt_template_normalizes_name_and_writes_front_matter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Template creation should slugify names and store description metadata."""
    prompts_dir = tmp_path / "prompts"
    monkeypatch.setattr(ocr_module, "OCR_PROMPTS_DIR", prompts_dir)

    created = ocr_module.create_ocr_prompt_template(
        template_name="Econometrics Tables!",
        description="Extract tables with high fidelity.",
        prompt="Use markdown tables for every table.",
    )

    assert created.name == "econometrics-tables"
    assert created.path == prompts_dir / "econometrics-tables.md"
    assert created.path.exists()
    assert (
        ocr_module.read_ocr_prompt_template("econometrics-tables")
        == "Use markdown tables for every table."
    )
