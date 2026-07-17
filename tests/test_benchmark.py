"""Tests for deterministic benchmark setup, scoring, and persistence."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import fitz
import pytest

from vlmocr import benchmark
from vlmocr.contract import build_raw_ocr_document, validate_raw_ocr_document


def _create_test_pdf(path: Path, *, text: str = "Page one") -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_academic_benchmark_preset_has_ten_unique_cases() -> None:
    case_ids = [case["id"] for case in benchmark.ACADEMIC_BENCHMARK_PRESET]
    assert len(case_ids) == 10
    assert len(case_ids) == len(set(case_ids))


def test_score_markdown_pair_perfect_match() -> None:
    markdown = """# Title\n\nEquation: $a+b=c$\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n"""

    score = benchmark.score_markdown_pair(markdown, markdown)

    assert score.text_score == pytest.approx(1.0)
    assert score.math_score == pytest.approx(1.0)
    assert score.structure_score == pytest.approx(1.0)
    assert score.overall_score == pytest.approx(1.0)


def test_score_markdown_pair_penalizes_math_errors() -> None:
    gold = """# Result\n\n$$x^2 + y^2 = z^2$$\n"""
    candidate = """# Result\n\n$$x^2 + y^2 = z^3$$\n"""

    score = benchmark.score_markdown_pair(gold, candidate)

    assert score.text_score > 0.9
    assert score.math_score < 1.0
    assert score.overall_score < 1.0


def test_initialize_academic_benchmark_creates_one_page_gold(
    tmp_path: Path,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    raw_dir = tmp_path / "converted" / "json" / "raw"
    raw_dir.mkdir(parents=True)

    _create_test_pdf(docs_dir / "sample.pdf")

    raw_payload = {
        "settings_hash": "seed-hash",
        "pages": [
            {"index": 0, "markdown": "Page 1 body"},
            {"index": 1, "markdown": "Page 2 body with $x^2$"},
        ],
    }
    (raw_dir / "sample.json").write_text(
        json.dumps(raw_payload),
        encoding="utf-8",
    )

    manifest_path = tmp_path / "bench" / "manifest.json"

    benchmark.initialize_academic_benchmark(
        docs_dir=docs_dir,
        out_dir=tmp_path / "converted",
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        preset_cases=[
            {
                "id": "sample_case",
                "stem": "sample",
                "page": 2,
                "tags": ["math"],
                "note": "unit test case",
            }
        ],
        output_fn=lambda message: None,
    )

    loaded_manifest = benchmark.load_manifest(manifest_path)
    assert loaded_manifest.name == benchmark.DEFAULT_BENCHMARK_NAME
    assert len(loaded_manifest.cases) == 1

    case = loaded_manifest.cases[0]
    assert case.document == "sample.pdf"
    assert case.page == 2

    gold_path = manifest_path.parent / case.gold_json
    gold_payload = validate_raw_ocr_document(json.loads(gold_path.read_text(encoding="utf-8")))

    assert gold_payload["pages"] == [{"index": 0, "markdown": "Page 2 body with $x^2$"}]


def test_benchmark_history_database_records_incremental_results(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "history.db"
    history = benchmark.BenchmarkHistoryDatabase(db_path)

    run_id = history.start_run(
        manifest_path=tmp_path / "manifest.json",
        manifest_name="test-manifest",
        manifest_version="1.0",
        models=["model-a"],
        args_payload={"case_limit": 1},
    )

    history.record_case_result(
        run_id=run_id,
        model="model-a",
        result={
            "case_id": "case-1",
            "document": "sample.pdf",
            "page": 1,
            "text_score": 1.0,
            "math_score": 1.0,
            "structure_score": 1.0,
            "overall_score": 1.0,
            "candidate_json_path": "candidate.json",
            "gold_json_path": "gold.json",
            "error": None,
        },
    )

    history.record_model_summary(
        run_id=run_id,
        model="model-a",
        summary={
            "cases_total": 1,
            "cases_scored": 1,
            "cases_failed": 0,
            "text_score": 1.0,
            "math_score": 1.0,
            "structure_score": 1.0,
            "overall_score": 1.0,
        },
    )

    history.finish_run(
        run_id=run_id,
        status="completed",
        report_path=tmp_path / "report.json",
        error=None,
    )
    history.close()

    with sqlite3.connect(db_path) as connection:
        run_count = connection.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()[0]
        case_count = connection.execute(
            "SELECT COUNT(*) FROM benchmark_case_results"
        ).fetchone()[0]
        model_count = connection.execute(
            "SELECT COUNT(*) FROM benchmark_model_summaries"
        ).fetchone()[0]
        status = connection.execute(
            "SELECT status FROM benchmark_runs WHERE id = ?",
            (run_id,),
        ).fetchone()[0]

    assert run_count == 1
    assert case_count == 1
    assert model_count == 1
    assert status == "completed"


def test_run_benchmark_writes_report_and_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    _create_test_pdf(docs_dir / "sample.pdf", text="Benchmark page")

    bench_root = tmp_path / "bench"
    gold_dir = bench_root / "gold"
    gold_dir.mkdir(parents=True)

    gold_markdown = "# Sample\n\n$E=mc^2$"
    gold_payload = build_raw_ocr_document([gold_markdown], settings_hash="gold-seed")
    (gold_dir / "case1.json").write_text(
        json.dumps(gold_payload),
        encoding="utf-8",
    )

    manifest_payload = {
        "name": "test-benchmark",
        "version": "1.0",
        "created_at": "2026-01-01T00:00:00+00:00",
        "cases": [
            {
                "id": "case1",
                "document": "sample.pdf",
                "page": 1,
                "gold_json": "gold/case1.json",
                "tags": ["math"],
                "note": "single case",
            }
        ],
    }
    manifest_path = bench_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    monkeypatch.setattr(benchmark.ocr, "create_client", lambda api_key=None: object())
    monkeypatch.setattr(
        benchmark.ocr,
        "read_ocr_prompt_template",
        lambda template_name=benchmark.ocr.DEFAULT_OCR_PROMPT_TEMPLATE: "prompt",
    )

    def fake_ocr_page(
        client,
        base64_image,
        *,
        model=benchmark.ocr.DEFAULT_OCR_MODEL,
        fmt=benchmark.ocr.DEFAULT_OCR_IMAGE_FORMAT,
        prompt=None,
        temperature=benchmark.ocr.DEFAULT_VLM_TEMPERATURE,
        max_tokens=benchmark.ocr.DEFAULT_OCR_MAX_TOKENS,
    ) -> str:
        if model == "model-a":
            return gold_markdown
        return "# Sample\n\n$E=mc^3$"

    monkeypatch.setattr(benchmark.ocr, "_ocr_page", fake_ocr_page)

    out_dir = tmp_path / "converted"
    report = benchmark.run_benchmark(
        manifest_path=manifest_path,
        docs_dir=docs_dir,
        out_dir=out_dir,
        models=["model-a", "model-b"],
        case_limit=1,
        output_fn=lambda message: None,
    )

    assert report["manifest"]["case_count"] == 1
    assert len(report["models"]) == 2
    assert report["ranking"][0]["model"] == "model-a"

    history_path = benchmark.get_default_database_path(out_dir=out_dir)
    assert history_path.exists()

    with sqlite3.connect(history_path) as connection:
        run_count = connection.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()[0]
        case_count = connection.execute(
            "SELECT COUNT(*) FROM benchmark_case_results"
        ).fetchone()[0]

    assert run_count == 1
    assert case_count == 2
