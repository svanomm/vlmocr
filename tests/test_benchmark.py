"""Tests for deterministic benchmark setup, scoring, and persistence."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import fitz
import pytest

from vlmocr import benchmark
from vlmocr.contract import build_raw_ocr_document, validate_raw_ocr_document


def _create_test_pdf(path: Path, *, text: str = "Page one", pages: int = 1) -> None:
    doc = fitz.open()
    for page_index in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"{text} {page_index + 1}")
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


def test_score_markdown_pair_normalizes_formatting_and_math_aliases() -> None:
    gold = """## Syntax\n\n`predict` $[type]$ *newvar* where $a \\geq .$\n"""
    candidate = """Syntax\n\npredict [type] newvar where $a \\ge .$\n"""

    score = benchmark.score_markdown_pair(gold, candidate)

    assert score.text_score > 0.95
    assert score.math_score == pytest.approx(1.0)
    assert score.contract_score < score.content_score
    assert score.overall_score == pytest.approx(score.content_score)
    assert score.overall_score > score.legacy_overall_score
    assert score.audit.suspected_formatting_bias is True
    assert "inline_syntax_wrapped_as_math" in score.audit.flags


def test_score_markdown_pair_separates_contract_markup_from_content() -> None:
    gold = """# Result\n\nBody<ref num=\"1\"/>\n\n<note num=\"1\">Source note</note>\n"""
    candidate = """Result\n\nBody\n\nSource note\n"""

    score = benchmark.score_markdown_pair(gold, candidate)

    assert score.text_score == pytest.approx(1.0)
    assert score.overall_score == pytest.approx(1.0)
    assert score.contract_score < 1.0
    assert score.overall_score > score.legacy_overall_score
    assert score.audit.suspected_formatting_bias is True
    assert "contract_markup_difference" in score.audit.flags


def test_initialize_academic_benchmark_creates_one_page_gold(
    tmp_path: Path,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    raw_dir = tmp_path / "converted" / "json" / "raw"
    raw_dir.mkdir(parents=True)

    _create_test_pdf(docs_dir / "sample.pdf", pages=2)

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
    assert case.document == "benchmark/sample_case.pdf"
    assert case.page == 1

    benchmark_pdf_path = docs_dir / case.document
    assert benchmark_pdf_path.exists()
    with fitz.open(benchmark_pdf_path) as doc:
        assert len(doc) == 1

    gold_path = manifest_path.parent / case.gold_json
    gold_payload = validate_raw_ocr_document(json.loads(gold_path.read_text(encoding="utf-8")))

    assert gold_payload["pages"] == [{"index": 0, "markdown": "Page 2 body with $x^2$"}]


def test_verify_benchmark_gold_for_pdf_folder_detects_missing_gold(
    tmp_path: Path,
) -> None:
    docs_dir = tmp_path / "docs"
    benchmark_dir = docs_dir / "benchmark"
    benchmark_dir.mkdir(parents=True)
    _create_test_pdf(benchmark_dir / "case1.pdf")

    manifest_path = tmp_path / "bench" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "name": "test-benchmark",
        "version": "1.0",
        "created_at": "2026-01-01T00:00:00+00:00",
        "cases": [
            {
                "id": "case1",
                "document": "benchmark/case1.pdf",
                "page": 1,
                "gold_json": "gold/missing.json",
                "tags": [],
                "note": "missing gold",
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing converted gold JSON"):
        benchmark.verify_benchmark_gold_for_pdf_folder(
            docs_dir=docs_dir,
            manifest_path=manifest_path,
            output_fn=lambda message: None,
        )


def test_repair_gold_markdown_backslashes_escapes_non_n_sequences(
    tmp_path: Path,
) -> None:
    manifest_root = tmp_path / "bench"
    gold_dir = manifest_root / "gold"
    gold_dir.mkdir(parents=True)

    manifest_payload = {
        "name": "test-benchmark",
        "version": "1.0",
        "created_at": "2026-01-01T00:00:00+00:00",
        "cases": [
            {
                "id": "case1",
                "document": "benchmark/case1.pdf",
                "page": 1,
                "gold_json": "gold/case1.json",
                "tags": [],
                "note": "escape repair",
            }
        ],
    }
    manifest_path = manifest_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    broken_gold = r'''{
  "settings_hash": "seed-hash",
  "pages": [
    {
      "index": 0,
      "markdown": "Line one\nLine two with \hat{x} and \text{abc} and \neq and \\alpha."
    }
  ]
}
'''
    gold_path = gold_dir / "case1.json"
    gold_path.write_text(broken_gold, encoding="utf-8")

    summary = benchmark.repair_gold_markdown_backslashes(
        manifest_path=manifest_path,
        output_fn=lambda message: None,
    )

    fixed_text = gold_path.read_text(encoding="utf-8")

    assert summary["files_scanned"] == 1
    assert summary["files_updated"] == 1
    assert summary["files_remaining_invalid"] == 0
    assert summary["markdown_sections"] == 1
    assert summary["replacements"] == 2
    assert r"\\hat{x}" in fixed_text
    assert r"\\text{abc}" in fixed_text
    assert r"\neq" in fixed_text
    assert r"\n" in fixed_text
    assert r"\\alpha" in fixed_text

    parsed_payload = validate_raw_ocr_document(json.loads(fixed_text))
    assert parsed_payload["pages"][0]["index"] == 0


def test_repair_gold_markdown_backslashes_skips_missing_markdown_field(
    tmp_path: Path,
) -> None:
    manifest_root = tmp_path / "bench"
    gold_dir = manifest_root / "gold"
    gold_dir.mkdir(parents=True)

    manifest_payload = {
        "name": "test-benchmark",
        "version": "1.0",
        "created_at": "2026-01-01T00:00:00+00:00",
        "cases": [
            {
                "id": "case1",
                "document": "benchmark/case1.pdf",
                "page": 1,
                "gold_json": "gold/case1.json",
                "tags": [],
                "note": "missing markdown key",
            }
        ],
    }
    manifest_path = manifest_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    malformed_payload = r'''{
  "settings_hash": "seed-hash",
  "pages": [
    {
      "index": 0,
      "content": "\\hat{x}"
    }
  ]
}
'''
    gold_path = gold_dir / "case1.json"
    gold_path.write_text(malformed_payload, encoding="utf-8")

    summary = benchmark.repair_gold_markdown_backslashes(
        manifest_path=manifest_path,
        output_fn=lambda message: None,
    )

    assert summary["files_scanned"] == 1
    assert summary["files_updated"] == 0
    assert summary["files_skipped_no_markdown"] == 1
    assert summary["files_remaining_invalid"] == 0
    assert summary["markdown_sections"] == 0
    assert summary["replacements"] == 0
    assert gold_path.read_text(encoding="utf-8") == malformed_payload


def test_repair_gold_markdown_backslashes_reports_remaining_invalid_files(
    tmp_path: Path,
) -> None:
    manifest_root = tmp_path / "bench"
    gold_dir = manifest_root / "gold"
    gold_dir.mkdir(parents=True)

    manifest_payload = {
        "name": "test-benchmark",
        "version": "1.0",
        "created_at": "2026-01-01T00:00:00+00:00",
        "cases": [
            {
                "id": "case1",
                "document": "benchmark/case1.pdf",
                "page": 1,
                "gold_json": "gold/case1.json",
                "tags": [],
                "note": "contains non-letter invalid escape",
            }
        ],
    }
    manifest_path = manifest_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    invalid_gold = r'''{
  "settings_hash": "seed-hash",
  "pages": [
    {
      "index": 0,
      "markdown": "A markdown table escape: \| should remain for manual fix."
    }
  ]
}
'''
    gold_path = gold_dir / "case1.json"
    gold_path.write_text(invalid_gold, encoding="utf-8")

    summary = benchmark.repair_gold_markdown_backslashes(
        manifest_path=manifest_path,
        output_fn=lambda message: None,
    )

    assert summary["files_scanned"] == 1
    assert summary["files_updated"] == 0
    assert summary["files_skipped_no_markdown"] == 0
    assert summary["files_remaining_invalid"] == 1
    assert summary["markdown_sections"] == 1
    assert summary["replacements"] == 0
    assert gold_path.read_text(encoding="utf-8") == invalid_gold


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
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost_usd": 0.001,
            "dollars_per_1000_pages": 1.0,
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
            "pages_billed": 1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "total_cost_usd": 0.001,
            "dollars_per_1000_pages": 1.0,
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


def test_get_recent_benchmark_results_returns_latest_rows(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "history.db"
    history = benchmark.BenchmarkHistoryDatabase(db_path)

    run_id_1 = history.start_run(
        manifest_path=tmp_path / "manifest.json",
        manifest_name="test-manifest",
        manifest_version="1.0",
        models=["model-a"],
        args_payload={},
    )
    history.record_model_summary(
        run_id=run_id_1,
        model="model-a",
        summary={
            "cases_total": 1,
            "cases_scored": 1,
            "cases_failed": 0,
            "pages_billed": 1,
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20,
            "total_cost_usd": 0.001,
            "dollars_per_1000_pages": 1.0,
            "text_score": 1.0,
            "math_score": 1.0,
            "structure_score": 1.0,
            "overall_score": 1.0,
        },
    )
    history.finish_run(
        run_id=run_id_1,
        status="completed",
        report_path=tmp_path / "report-1.json",
        error=None,
    )

    run_id_2 = history.start_run(
        manifest_path=tmp_path / "manifest.json",
        manifest_name="test-manifest",
        manifest_version="1.0",
        models=["model-b"],
        args_payload={},
    )
    history.record_model_summary(
        run_id=run_id_2,
        model="model-b",
        summary={
            "cases_total": 2,
            "cases_scored": 2,
            "cases_failed": 0,
            "pages_billed": 2,
            "prompt_tokens": 20,
            "completion_tokens": 20,
            "total_tokens": 40,
            "total_cost_usd": 0.002,
            "dollars_per_1000_pages": 1.0,
            "text_score": 0.9,
            "math_score": 0.9,
            "structure_score": 0.9,
            "overall_score": 0.9,
        },
    )
    history.finish_run(
        run_id=run_id_2,
        status="completed",
        report_path=tmp_path / "report-2.json",
        error=None,
    )
    history.close()

    rows = benchmark.get_recent_benchmark_results(database_path=db_path, limit=10)

    assert len(rows) == 2
    assert rows[0]["run_id"] == run_id_2
    assert rows[0]["model"] == "model-b"
    assert rows[1]["run_id"] == run_id_1
    assert rows[1]["model"] == "model-a"


def test_rescore_benchmark_reports_updates_report_and_history(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "converted" / "benchmark" / "reports"
    report_dir.mkdir(parents=True)
    candidate_dir = tmp_path / "converted" / "benchmark" / "candidates" / "run-000001" / "model-a"
    candidate_dir.mkdir(parents=True)
    bench_root = tmp_path / "bench"
    gold_dir = bench_root / "gold"
    gold_dir.mkdir(parents=True)

    gold_markdown = "# Sample\n\n$E=mc^2$"
    gold_path = gold_dir / "case1.json"
    gold_path.write_text(
        json.dumps(build_raw_ocr_document([gold_markdown], settings_hash="gold")),
        encoding="utf-8",
    )

    candidate_path = candidate_dir / "case1.json"
    candidate_path.write_text(
        json.dumps(build_raw_ocr_document([gold_markdown], settings_hash="candidate")),
        encoding="utf-8",
    )

    report_path = report_dir / "run-000001.json"
    relative_candidate_path = Path("converted") / "benchmark" / "candidates" / "run-000001" / "model-a" / "case1.json"
    stale_case_result = {
        "case_id": "case1",
        "document": "benchmark/case1.pdf",
        "page": 1,
        "text_score": 0.0,
        "math_score": 0.0,
        "structure_score": 0.0,
        "overall_score": 0.0,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cost_usd": 0.001,
        "dollars_per_1000_pages": 1.0,
        "candidate_json_path": str(relative_candidate_path),
        "gold_json_path": str(gold_path),
        "error": None,
    }
    stale_summary = {
        "cases_total": 1,
        "cases_scored": 1,
        "cases_failed": 0,
        "pages_billed": 1,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "total_cost_usd": 0.001,
        "dollars_per_1000_pages": 1.0,
        "text_score": 0.0,
        "math_score": 0.0,
        "structure_score": 0.0,
        "overall_score": 0.0,
    }
    report_payload = {
        "run_id": 1,
        "created_at": "2026-07-18T00:00:00+00:00",
        "manifest": {
            "path": str(bench_root / "manifest.json"),
            "name": "test-benchmark",
            "version": "1.0",
            "case_count": 1,
        },
        "models": [
            {
                "model": "model-a",
                "summary": stale_summary,
                "cases": [stale_case_result],
            }
        ],
        "ranking": [
            {
                "model": "model-a",
                "overall_score": 0.0,
                "cases_failed": 0,
                "total_cost_usd": 0.001,
                "dollars_per_1000_pages": 1.0,
            }
        ],
    }
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")

    db_path = tmp_path / "converted" / "benchmark" / "history.db"
    history = benchmark.BenchmarkHistoryDatabase(db_path)
    run_id = history.start_run(
        manifest_path=bench_root / "manifest.json",
        manifest_name="test-benchmark",
        manifest_version="1.0",
        models=["model-a"],
        args_payload={},
    )
    assert run_id == 1
    history.record_case_result(run_id=run_id, model="model-a", result=stale_case_result)
    history.record_model_summary(run_id=run_id, model="model-a", summary=stale_summary)
    history.finish_run(run_id=run_id, status="completed", report_path=report_path, error=None)
    history.close()

    summary = benchmark.rescore_benchmark_reports(
        reports_dir=report_dir,
        database_path=db_path,
        output_fn=lambda message: None,
    )

    assert summary["reports_rescored"] == 1
    assert summary["models_rescored"] == 1
    assert summary["cases_rescored"] == 1
    assert summary["database_updated"] is True

    rescored_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert rescored_report["scoring"]["overall_score"] == "content_score"
    rescored_case = rescored_report["models"][0]["cases"][0]
    assert rescored_case["overall_score"] == pytest.approx(1.0)
    assert rescored_case["content_score"] == pytest.approx(1.0)
    assert rescored_case["score_audit"]["suspected_formatting_bias"] is False

    with sqlite3.connect(db_path) as connection:
        case_row = connection.execute(
            "SELECT overall_score, content_score, contract_score FROM benchmark_case_results WHERE run_id = ? AND model = ?",
            (run_id, "model-a"),
        ).fetchone()
        summary_row = connection.execute(
            "SELECT overall_score, content_score, contract_score FROM benchmark_model_summaries WHERE run_id = ? AND model = ?",
            (run_id, "model-a"),
        ).fetchone()

    assert case_row is not None
    assert summary_row is not None
    assert float(case_row[0]) == pytest.approx(1.0)
    assert float(case_row[1]) == pytest.approx(1.0)
    assert float(case_row[2]) == pytest.approx(1.0)
    assert float(summary_row[0]) == pytest.approx(1.0)
    assert float(summary_row[1]) == pytest.approx(1.0)
    assert float(summary_row[2]) == pytest.approx(1.0)


def test_run_benchmark_writes_report_and_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_dir = tmp_path / "docs"
    benchmark_docs_dir = docs_dir / "benchmark"
    benchmark_docs_dir.mkdir(parents=True)
    _create_test_pdf(benchmark_docs_dir / "case1.pdf", text="Benchmark page")

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
                "document": "benchmark/case1.pdf",
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

    def fake_ocr_page_with_usage(
        client,
        base64_image,
        *,
        model=benchmark.ocr.DEFAULT_OCR_MODEL,
        fmt=benchmark.ocr.DEFAULT_OCR_IMAGE_FORMAT,
        prompt=None,
        temperature=benchmark.ocr.DEFAULT_VLM_TEMPERATURE,
        max_tokens=benchmark.ocr.DEFAULT_OCR_MAX_TOKENS,
    ) -> tuple[str, benchmark.ocr.OCRPageUsage]:
        if model == "model-a":
            return (
                gold_markdown,
                benchmark.ocr.OCRPageUsage(
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                    cost=0.002,
                ),
            )
        return (
            "# Sample\n\n$E=mc^3$",
            benchmark.ocr.OCRPageUsage(
                prompt_tokens=120,
                completion_tokens=60,
                total_tokens=180,
                cost=0.003,
            ),
        )

    monkeypatch.setattr(benchmark.ocr, "_ocr_page_with_usage", fake_ocr_page_with_usage)

    out_dir = tmp_path / "converted"
    output_lines: list[str] = []
    report = benchmark.run_benchmark(
        manifest_path=manifest_path,
        docs_dir=docs_dir,
        out_dir=out_dir,
        models=["model-a", "model-b"],
        case_limit=1,
        output_fn=output_lines.append,
    )

    assert report["manifest"]["case_count"] == 1
    assert report["scoring"]["overall_score"] == "content_score"
    assert len(report["models"]) == 2
    assert report["ranking"][0]["model"] == "model-a"
    model_summaries = {
        model_report["model"]: model_report["summary"] for model_report in report["models"]
    }
    assert model_summaries["model-a"]["contract_score"] == pytest.approx(1.0)
    assert model_summaries["model-a"]["legacy_overall_score"] == pytest.approx(1.0)
    assert model_summaries["model-a"]["formatting_bias_cases"] == 0
    assert model_summaries["model-a"]["total_cost_usd"] == pytest.approx(0.002)
    assert model_summaries["model-a"]["dollars_per_1000_pages"] == pytest.approx(2.0)
    assert model_summaries["model-b"]["total_cost_usd"] == pytest.approx(0.003)
    assert model_summaries["model-b"]["dollars_per_1000_pages"] == pytest.approx(3.0)
    assert report["ranking"][0]["dollars_per_1000_pages"] == pytest.approx(2.0)
    assert report["models"][0]["cases"][0]["score_audit"]["suspected_formatting_bias"] is False

    history_path = benchmark.get_default_database_path(out_dir=out_dir)
    assert history_path.exists()
    assert "Benchmark progress: [  ] 0/2 pages completed" in output_lines
    assert any(
        line == "Progress [. ] 1/2 pages completed (model-a: case1)"
        for line in output_lines
    )
    assert any(
        line == "Progress [..] 2/2 pages completed (model-b: case1)"
        for line in output_lines
    )

    with sqlite3.connect(history_path) as connection:
        run_count = connection.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()[0]
        case_count = connection.execute(
            "SELECT COUNT(*) FROM benchmark_case_results"
        ).fetchone()[0]
        total_cost = connection.execute(
            "SELECT SUM(cost_usd) FROM benchmark_case_results"
        ).fetchone()[0]

    assert run_count == 1
    assert case_count == 2
    assert float(total_cost) == pytest.approx(0.005)
