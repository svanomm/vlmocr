"""Deterministic benchmark tooling for OCR model comparisons."""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import fitz
from openai import OpenAI

from vlmocr import ocr
from vlmocr.contract import (
    DEFAULT_DOCS_DIR,
    DEFAULT_OUT_DIR,
    build_raw_ocr_document,
    validate_raw_ocr_document,
)
from vlmocr.text_cleaning import clean_text

OutputFunc = Callable[[str], None]

DEFAULT_BENCHMARK_NAME = "academic-textbook-v1"
DEFAULT_BENCHMARK_VERSION = "1.0"

_BENCHMARK_SUBDIR = Path("benchmark")
_BENCHMARK_GOLD_SUBDIR = Path("gold")
_BENCHMARK_MANIFEST_FILENAME = "manifest.json"
_BENCHMARK_DATABASE_SUBPATH = Path("benchmark/history.db")
_BENCHMARK_REPORTS_SUBPATH = Path("benchmark/reports")
_BENCHMARK_CANDIDATES_SUBPATH = Path("benchmark/candidates")
_BENCHMARK_DOCS_SUBPATH = Path("benchmark")

OVERALL_TEXT_WEIGHT = 0.45
OVERALL_MATH_WEIGHT = 0.40
OVERALL_STRUCTURE_WEIGHT = 0.15

_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)
_MATH_SEGMENT_RE = re.compile(
    r"\$\$(.+?)\$\$|(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)",
    re.DOTALL,
)
_MULTI_WHITESPACE_RE = re.compile(r"\s+")
_MODEL_SLUG_RE = re.compile(r"[^a-z0-9]+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MARKDOWN_VALUE_KEY_RE = re.compile(r'"markdown"\s*:\s*"')
_SINGLE_BACKSLASH_LETTER_RE = re.compile(r"(?<!\\)\\([A-Za-z])")

_MATH_REPLACEMENTS = {
    "−": "-",
    "–": "-",
    "—": "-",
    "∗": "*",
    "⋅": "*",
    "×": "*",
    "÷": "/",
    "∕": "/",
    "≤": r"\\leq",
    "≥": r"\\geq",
    "≠": r"\\neq",
}

ACADEMIC_BENCHMARK_PRESET: tuple[dict[str, Any], ...] = (
    {
        "id": "abadie_p11_equation_dense",
        "stem": "Abadie, Athey, Imbens, Wooldridge (2017) -  When Should you Adjust Standard Errors for Clustering",
        "page": 11,
        "tags": ["math", "table"],
        "note": "High density of equations and table-style blocks.",
    },
    {
        "id": "cameron_miller_p31_variance_proofs",
        "stem": "Cameron and Miller (2015) - Practitioner's Guide to Cluster-Robust Inference",
        "page": 31,
        "tags": ["math"],
        "note": "Long mathematical derivations with nested notation.",
    },
    {
        "id": "balli_sorensen_p17_interaction_tables",
        "stem": "Balli and Sorensen (2012) - Interaction Effects in Econometrics",
        "page": 17,
        "tags": ["math", "table"],
        "note": "Mixed formulas and dense table layout.",
    },
    {
        "id": "balli_sorensen_p10_mixed_footnote",
        "stem": "Balli and Sorensen (2012) - Interaction Effects in Econometrics",
        "page": 10,
        "tags": ["math", "table", "footnote"],
        "note": "Mixed equations, footnotes, and tabular structure.",
    },
    {
        "id": "benoit_p4_log_transform_math",
        "stem": "Benoit (2011) - Linear Regression Models with Logarithmic Transformations",
        "page": 4,
        "tags": ["math"],
        "note": "Log-transform derivations with notation sensitivity.",
    },
    {
        "id": "norton_dowd_p12_odds_table",
        "stem": "Norton and Dowd (2018) - Log Odds and the Interpretation of Logit Models",
        "page": 12,
        "tags": ["math", "table"],
        "note": "Math-heavy interpretation table.",
    },
    {
        "id": "youssef_p6_diagnostic_matrix",
        "stem": "Youssef (2022) - Detecting of Multicollinearity, Autocorrelation, and Heteroscedasticity in Regression Analysis",
        "page": 6,
        "tags": ["table"],
        "note": "Complex table and matrix-like formatting.",
    },
    {
        "id": "wainer_p10_figure_table_mix",
        "stem": "Wainer (1984) - How to Display Data Badly",
        "page": 10,
        "tags": ["table", "image"],
        "note": "Figure caption plus tabular region.",
    },
    {
        "id": "anscombe_p2_small_mixed_layout",
        "stem": "Anscombe's quartet - Wikipedia",
        "page": 2,
        "tags": ["table", "footnote"],
        "note": "Short but structurally mixed content.",
    },
    {
        "id": "predict_p1_command_math_table",
        "stem": "predict",
        "page": 1,
        "tags": ["math", "table", "image"],
        "note": "Command syntax with formulas, table blocks, and image tags.",
    },
)


@dataclass(frozen=True)
class BenchmarkCase:
    """One deterministic benchmark case."""

    case_id: str
    document: str
    page: int
    gold_json: str
    tags: list[str]
    note: str


@dataclass(frozen=True)
class BenchmarkManifest:
    """Benchmark manifest metadata and selected cases."""

    name: str
    version: str
    created_at: str
    cases: list[BenchmarkCase]


@dataclass(frozen=True)
class CaseScore:
    """Deterministic component scores for one case."""

    text_score: float
    math_score: float
    structure_score: float
    overall_score: float


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_non_negative_float(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, float(value))


def _dollars_per_1000_pages(*, total_cost_usd: float, pages: int) -> float:
    if pages <= 0:
        return 0.0
    return _safe_non_negative_float((total_cost_usd / pages) * 1000.0)


def _slugify_model_name(model: str) -> str:
    lowered = model.strip().lower()
    slug = _MODEL_SLUG_RE.sub("-", lowered).strip("-")
    return slug or "model"


def _normalize_document_reference(document: str) -> str:
    normalized = document.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def get_default_benchmark_root(*, out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    """Return the default benchmark root under the output directory."""
    return out_dir / _BENCHMARK_SUBDIR / DEFAULT_BENCHMARK_NAME


def get_default_manifest_path(*, out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    """Return the default manifest path for the academic benchmark."""
    return get_default_benchmark_root(out_dir=out_dir) / _BENCHMARK_MANIFEST_FILENAME


def get_default_database_path(*, out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    """Return the default SQLite path for benchmark history."""
    return out_dir / _BENCHMARK_DATABASE_SUBPATH


def get_default_reports_dir(*, out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    """Return the default report output directory for benchmark runs."""
    return out_dir / _BENCHMARK_REPORTS_SUBPATH


def get_default_candidates_dir(*, out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    """Return the default candidate output directory for benchmark runs."""
    return out_dir / _BENCHMARK_CANDIDATES_SUBPATH


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON file is invalid: {path}") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _find_json_string_end(text: str, opening_quote_index: int) -> int:
    if opening_quote_index < 0 or opening_quote_index >= len(text):
        raise ValueError("Opening quote index is out of range.")
    if text[opening_quote_index] != '"':
        raise ValueError("Opening quote index must point to a double quote character.")

    escaped = False
    for index in range(opening_quote_index + 1, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            return index

    raise ValueError("Unterminated JSON string while repairing markdown escapes.")


def _escape_single_backslash_letter_not_n(markdown_value: str) -> tuple[str, int]:
    replacements = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal replacements
        letter = match.group(1)
        if letter == "n":
            return match.group(0)
        replacements += 1
        return "\\\\" + letter

    repaired = _SINGLE_BACKSLASH_LETTER_RE.sub(_replace, markdown_value)
    return repaired, replacements


def _repair_markdown_sections(json_text: str) -> tuple[str, int, int]:
    rebuilt: list[str] = []
    cursor = 0
    section_count = 0
    replacement_count = 0

    while True:
        match = _MARKDOWN_VALUE_KEY_RE.search(json_text, cursor)
        if match is None:
            rebuilt.append(json_text[cursor:])
            break

        opening_quote_index = match.end() - 1
        closing_quote_index = _find_json_string_end(json_text, opening_quote_index)

        raw_markdown = json_text[opening_quote_index + 1 : closing_quote_index]
        repaired_markdown, replacements = _escape_single_backslash_letter_not_n(
            raw_markdown
        )

        section_count += 1
        replacement_count += replacements

        rebuilt.append(json_text[cursor : opening_quote_index + 1])
        rebuilt.append(repaired_markdown)
        cursor = closing_quote_index

    return "".join(rebuilt), section_count, replacement_count


def repair_gold_markdown_backslashes(
    *,
    manifest_path: Path,
    output_fn: OutputFunc = print,
) -> dict[str, int]:
    """Repair single-backslash letter escapes in gold markdown JSON values.

    This only modifies `markdown` string values and skips sequences that start with
    `\n`, allowing manual follow-up for LaTeX commands that begin with `n`.
    """
    resolved_manifest_path = Path(manifest_path)
    manifest = load_manifest(resolved_manifest_path)

    files_scanned = 0
    files_updated = 0
    files_skipped_no_markdown = 0
    files_remaining_invalid = 0
    markdown_sections = 0
    replacements = 0

    for case in manifest.cases:
        gold_path = (resolved_manifest_path.parent / case.gold_json).resolve()
        if not gold_path.exists():
            raise FileNotFoundError(f"Benchmark gold JSON not found: {gold_path}")

        original_text = gold_path.read_text(encoding="utf-8")
        repaired_text, section_count, replacement_count = _repair_markdown_sections(
            original_text
        )

        files_scanned += 1

        if section_count == 0:
            files_skipped_no_markdown += 1
            output_fn(
                "Skipped benchmark gold escape repair (no markdown field): "
                f"{gold_path}"
            )
            continue

        markdown_sections += section_count
        replacements += replacement_count

        if repaired_text != original_text:
            gold_path.write_text(repaired_text, encoding="utf-8")
            files_updated += 1

        try:
            validate_raw_ocr_document(json.loads(gold_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            files_remaining_invalid += 1
            output_fn(
                "Benchmark gold JSON remains invalid after escape repair: "
                f"{gold_path} (line {exc.lineno}, column {exc.colno})"
            )
        except ValueError as exc:
            files_remaining_invalid += 1
            output_fn(
                "Benchmark gold JSON failed contract validation after escape repair: "
                f"{gold_path} ({exc})"
            )

    summary = {
        "files_scanned": files_scanned,
        "files_updated": files_updated,
        "files_skipped_no_markdown": files_skipped_no_markdown,
        "files_remaining_invalid": files_remaining_invalid,
        "markdown_sections": markdown_sections,
        "replacements": replacements,
    }
    output_fn(
        "Repaired benchmark gold markdown escapes: "
        f"{files_updated} updated files, {replacements} replacements across "
        f"{markdown_sections} markdown sections; "
        f"{files_skipped_no_markdown} files skipped (no markdown key); "
        f"{files_remaining_invalid} files still invalid."
    )
    return summary


def load_manifest(manifest_path: Path) -> BenchmarkManifest:
    """Load and validate a benchmark manifest from disk."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Benchmark manifest not found: {manifest_path}")

    payload = _read_json(manifest_path)
    if not isinstance(payload, dict):
        raise ValueError("Benchmark manifest must be a JSON object.")

    name = payload.get("name")
    version = payload.get("version")
    created_at = payload.get("created_at")
    cases_payload = payload.get("cases")

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Benchmark manifest must define a non-empty string 'name'.")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("Benchmark manifest must define a non-empty string 'version'.")
    if not isinstance(created_at, str) or not created_at.strip():
        raise ValueError(
            "Benchmark manifest must define a non-empty string 'created_at'."
        )
    if not isinstance(cases_payload, list) or not cases_payload:
        raise ValueError("Benchmark manifest must define a non-empty 'cases' list.")

    seen_case_ids: set[str] = set()
    cases: list[BenchmarkCase] = []
    for index, item in enumerate(cases_payload):
        if not isinstance(item, dict):
            raise ValueError(f"Benchmark case at index {index} must be a JSON object.")

        case_id = item.get("id")
        document = item.get("document")
        page = item.get("page")
        gold_json = item.get("gold_json")
        tags = item.get("tags", [])
        note = item.get("note", "")

        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Benchmark case at index {index} must define a string 'id'.")
        if case_id in seen_case_ids:
            raise ValueError(f"Benchmark case id is duplicated: {case_id}")
        seen_case_ids.add(case_id)

        if not isinstance(document, str) or not document.strip():
            raise ValueError(
                f"Benchmark case '{case_id}' must define a non-empty string 'document'."
            )
        if not isinstance(page, int) or page < 1:
            raise ValueError(f"Benchmark case '{case_id}' must define page >= 1.")
        if not isinstance(gold_json, str) or not gold_json.strip():
            raise ValueError(
                f"Benchmark case '{case_id}' must define a non-empty string 'gold_json'."
            )

        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError(f"Benchmark case '{case_id}' must define string-only 'tags'.")
        if not isinstance(note, str):
            raise ValueError(f"Benchmark case '{case_id}' must define a string 'note'.")

        cases.append(
            BenchmarkCase(
                case_id=case_id,
                document=document,
                page=page,
                gold_json=gold_json,
                tags=tags,
                note=note,
            )
        )

    return BenchmarkManifest(
        name=name,
        version=version,
        created_at=created_at,
        cases=cases,
    )


def _extract_single_page_pdf(
    *,
    source_pdf: Path,
    page: int,
    output_pdf: Path,
) -> None:
    if page < 1:
        raise ValueError(f"PDF page must be >= 1, got {page}.")

    with fitz.open(source_pdf) as source_doc:
        if page > len(source_doc):
            raise ValueError(
                f"Requested page {page} for '{source_pdf.name}', but document has only {len(source_doc)} pages."
            )

        single_page_doc = fitz.open()
        single_page_doc.insert_pdf(source_doc, from_page=page - 1, to_page=page - 1)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        single_page_doc.save(output_pdf)
        single_page_doc.close()


def verify_benchmark_gold_for_pdf_folder(
    *,
    docs_dir: Path = DEFAULT_DOCS_DIR,
    manifest_path: Path,
    benchmark_subpath: Path = _BENCHMARK_DOCS_SUBPATH,
    require_folder: bool = True,
    output_fn: OutputFunc = print,
) -> dict[str, int]:
    """Verify that each PDF in docs/benchmark has a matching manifest case and gold JSON."""
    docs_dir = Path(docs_dir)
    manifest_path = Path(manifest_path)
    benchmark_dir = docs_dir / benchmark_subpath

    if not benchmark_dir.exists():
        if require_folder:
            raise FileNotFoundError(
                f"Benchmark documents folder not found: {benchmark_dir}"
            )
        return {"pdf_count": 0, "verified_count": 0}

    benchmark_pdfs = sorted(benchmark_dir.glob("*.pdf"))
    if not benchmark_pdfs:
        if require_folder:
            raise ValueError(f"No benchmark PDFs found in {benchmark_dir}")
        return {"pdf_count": 0, "verified_count": 0}

    manifest = load_manifest(manifest_path)
    case_by_document = {
        _normalize_document_reference(case.document): case for case in manifest.cases
    }

    benchmark_prefix = benchmark_subpath.as_posix().rstrip("/") + "/"
    missing_pdf_for_case: list[str] = []
    for case in manifest.cases:
        normalized_document = _normalize_document_reference(case.document)
        if not normalized_document.startswith(benchmark_prefix):
            continue
        if not (docs_dir / Path(normalized_document)).exists():
            missing_pdf_for_case.append(normalized_document)

    if missing_pdf_for_case:
        missing_lines = "\n".join(f"  - {item}" for item in missing_pdf_for_case)
        raise FileNotFoundError(
            "Manifest cases reference missing benchmark PDFs:\n"
            f"{missing_lines}"
        )

    missing_case_entries: list[str] = []
    missing_gold: list[str] = []
    verified_count = 0

    for pdf_path in benchmark_pdfs:
        relative_document = pdf_path.relative_to(docs_dir).as_posix()
        case = case_by_document.get(relative_document)
        if case is None:
            missing_case_entries.append(relative_document)
            continue

        gold_path = (manifest_path.parent / case.gold_json).resolve()
        if not gold_path.exists():
            missing_gold.append(relative_document)
            continue

        _load_raw_ocr_payload(gold_path)
        verified_count += 1

    if missing_case_entries:
        missing_lines = "\n".join(f"  - {item}" for item in missing_case_entries)
        raise ValueError(
            "Benchmark PDFs missing manifest case entries:\n"
            f"{missing_lines}"
        )

    if missing_gold:
        missing_lines = "\n".join(f"  - {item}" for item in missing_gold)
        raise FileNotFoundError(
            "Benchmark PDFs missing converted gold JSON references:\n"
            f"{missing_lines}"
        )

    output_fn(
        f"Verified gold JSON for {verified_count} benchmark PDFs in {benchmark_dir}."
    )
    return {"pdf_count": len(benchmark_pdfs), "verified_count": verified_count}


def initialize_academic_benchmark(
    *,
    docs_dir: Path = DEFAULT_DOCS_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    raw_dir: Path | None = None,
    manifest_path: Path | None = None,
    overwrite: bool = False,
    preset_cases: list[dict[str, Any]] | None = None,
    output_fn: OutputFunc = print,
) -> Path:
    """Create the local academic benchmark manifest, docs/benchmark PDFs, and one-page gold files."""
    docs_dir = Path(docs_dir)
    out_dir = Path(out_dir)
    resolved_manifest_path = Path(manifest_path or get_default_manifest_path(out_dir=out_dir))
    resolved_raw_dir = Path(raw_dir or (out_dir / "json" / "raw"))
    benchmark_docs_dir = docs_dir / _BENCHMARK_DOCS_SUBPATH

    if resolved_manifest_path.exists() and not overwrite:
        raise ValueError(
            "Benchmark manifest already exists. Use --overwrite to recreate it."
        )

    selected_cases = list(preset_cases or ACADEMIC_BENCHMARK_PRESET)
    if not selected_cases:
        raise ValueError("Benchmark preset must include at least one case.")

    benchmark_docs_dir.mkdir(parents=True, exist_ok=True)
    gold_dir = resolved_manifest_path.parent / _BENCHMARK_GOLD_SUBDIR
    gold_dir.mkdir(parents=True, exist_ok=True)

    manifest_cases: list[dict[str, Any]] = []

    for item in selected_cases:
        case_id = item["id"]
        stem = item["stem"]
        page = int(item["page"])
        tags = list(item.get("tags", []))
        note = str(item.get("note", ""))

        source_document = docs_dir / f"{stem}.pdf"
        if not source_document.exists():
            raise FileNotFoundError(
                f"Benchmark source document not found: {source_document}"
            )

        case_pdf_path = benchmark_docs_dir / f"{case_id}.pdf"
        _extract_single_page_pdf(
            source_pdf=source_document,
            page=page,
            output_pdf=case_pdf_path,
        )

        source_raw = resolved_raw_dir / f"{stem}.json"
        if not source_raw.exists():
            raise FileNotFoundError(
                f"Raw OCR JSON for benchmark seed not found: {source_raw}"
            )

        raw_payload = validate_raw_ocr_document(_read_json(source_raw))
        page_index = page - 1
        if page_index >= len(raw_payload["pages"]):
            raise ValueError(
                f"Case '{case_id}' requested page {page}, but '{source_raw.name}' only has {len(raw_payload['pages'])} pages."
            )

        gold_payload = build_raw_ocr_document(
            [raw_payload["pages"][page_index]["markdown"]],
            settings_hash=raw_payload["settings_hash"],
        )
        gold_path = gold_dir / f"{case_id}.json"
        _write_json(gold_path, gold_payload)

        manifest_cases.append(
            {
                "id": case_id,
                "document": case_pdf_path.relative_to(docs_dir).as_posix(),
                "page": 1,
                "gold_json": gold_path.relative_to(resolved_manifest_path.parent).as_posix(),
                "tags": tags,
                "note": note,
                "source_document": source_document.name,
                "source_page": page,
            }
        )

    manifest_payload = {
        "name": DEFAULT_BENCHMARK_NAME,
        "version": DEFAULT_BENCHMARK_VERSION,
        "created_at": _utc_now_iso(),
        "cases": manifest_cases,
    }

    _write_json(resolved_manifest_path, manifest_payload)

    verification = verify_benchmark_gold_for_pdf_folder(
        docs_dir=docs_dir,
        manifest_path=resolved_manifest_path,
        benchmark_subpath=_BENCHMARK_DOCS_SUBPATH,
        require_folder=True,
        output_fn=output_fn,
    )

    output_fn(f"Wrote benchmark manifest: {resolved_manifest_path}")
    output_fn(f"Wrote {verification['pdf_count']} one-page benchmark PDFs to: {benchmark_docs_dir}")
    output_fn(f"Wrote {len(manifest_cases)} one-page gold files to: {gold_dir}")
    output_fn(
        "Review the generated gold files manually before treating scores as authoritative."
    )

    return resolved_manifest_path


def _normalize_text(text: str) -> str:
    normalized = clean_text(text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _MULTI_WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def _strip_math_segments(text: str) -> str:
    return _MATH_SEGMENT_RE.sub(" ", text)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    if len(left) > len(right):
        left, right = right, left

    previous = list(range(len(left) + 1))
    for row_index, right_char in enumerate(right, start=1):
        current = [row_index]
        for column_index, left_char in enumerate(left, start=1):
            insert_cost = current[column_index - 1] + 1
            delete_cost = previous[column_index] + 1
            replace_cost = previous[column_index - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current

    return previous[-1]


def _char_similarity(left: str, right: str) -> float:
    denominator = max(len(left), len(right), 1)
    distance = _levenshtein_distance(left, right)
    return _safe_float(1.0 - (distance / denominator))


def _token_f1(gold: list[str], candidate: list[str]) -> float:
    if not gold and not candidate:
        return 1.0

    gold_counts = Counter(gold)
    candidate_counts = Counter(candidate)
    true_positive = sum((gold_counts & candidate_counts).values())

    if true_positive == 0:
        return 0.0

    precision = true_positive / max(len(candidate), 1)
    recall = true_positive / max(len(gold), 1)
    return _safe_float((2 * precision * recall) / (precision + recall))


def _canonicalize_math(expr: str) -> str:
    canonical = expr.strip()
    for source, target in _MATH_REPLACEMENTS.items():
        canonical = canonical.replace(source, target)

    canonical = canonical.replace(r"\left", "")
    canonical = canonical.replace(r"\right", "")
    canonical = canonical.replace("\n", " ")
    canonical = _MULTI_WHITESPACE_RE.sub("", canonical)
    return canonical


def _extract_math_segments(text: str) -> list[str]:
    segments: list[str] = []
    for match in _MATH_SEGMENT_RE.finditer(text):
        body = match.group(1) if match.group(1) is not None else match.group(2)
        if body is None:
            continue
        segments.append(_canonicalize_math(body))
    return segments


def _math_score(gold_text: str, candidate_text: str) -> float:
    gold_math = _extract_math_segments(gold_text)
    candidate_math = _extract_math_segments(candidate_text)

    if not gold_math and not candidate_math:
        return 1.0

    compare_count = max(len(gold_math), len(candidate_math), 1)
    exact = 0
    similarity_total = 0.0

    for index in range(compare_count):
        gold_expr = gold_math[index] if index < len(gold_math) else ""
        candidate_expr = candidate_math[index] if index < len(candidate_math) else ""

        if gold_expr and candidate_expr and gold_expr == candidate_expr:
            exact += 1

        if not gold_expr or not candidate_expr:
            continue

        similarity_total += _char_similarity(gold_expr, candidate_expr)

    exact_rate = exact / compare_count
    similarity_rate = similarity_total / compare_count
    return _safe_float((0.35 * exact_rate) + (0.65 * similarity_rate))


def _count_f1(gold_count: int, candidate_count: int) -> float:
    if gold_count == 0 and candidate_count == 0:
        return 1.0

    true_positive = min(gold_count, candidate_count)
    if true_positive == 0:
        return 0.0

    precision = true_positive / max(candidate_count, 1)
    recall = true_positive / max(gold_count, 1)
    return _safe_float((2 * precision * recall) / (precision + recall))


def _extract_structure_counts(markdown: str) -> dict[str, int]:
    lines = markdown.splitlines()
    heading_count = sum(1 for line in lines if line.strip().startswith("#"))
    table_line_count = sum(1 for line in lines if line.count("|") >= 2)

    return {
        "headings": heading_count,
        "table_lines": table_line_count,
        "ref_tags": markdown.count("<ref num="),
        "note_tags": markdown.count("<note num="),
        "image_tags": markdown.count("<image>"),
        "display_math": len(_DISPLAY_MATH_RE.findall(markdown)),
        "inline_math": len(_INLINE_MATH_RE.findall(markdown)),
    }


def _structure_score(gold_text: str, candidate_text: str) -> float:
    gold_counts = _extract_structure_counts(gold_text)
    candidate_counts = _extract_structure_counts(candidate_text)

    feature_scores = [
        _count_f1(gold_counts[feature], candidate_counts[feature])
        for feature in sorted(gold_counts)
    ]
    return _safe_float(mean(feature_scores))


def score_markdown_pair(gold_markdown: str, candidate_markdown: str) -> CaseScore:
    """Compute deterministic text, math, and structure scores for one page."""
    normalized_gold = _normalize_text(gold_markdown)
    normalized_candidate = _normalize_text(candidate_markdown)

    text_gold = _strip_math_segments(normalized_gold)
    text_candidate = _strip_math_segments(normalized_candidate)

    text_similarity = _char_similarity(text_gold, text_candidate)
    token_similarity = _token_f1(_tokenize(text_gold), _tokenize(text_candidate))
    text_score = _safe_float((text_similarity + token_similarity) / 2)

    math_score = _math_score(normalized_gold, normalized_candidate)
    structure_score = _structure_score(normalized_gold, normalized_candidate)

    overall_score = _safe_float(
        (OVERALL_TEXT_WEIGHT * text_score)
        + (OVERALL_MATH_WEIGHT * math_score)
        + (OVERALL_STRUCTURE_WEIGHT * structure_score)
    )

    return CaseScore(
        text_score=text_score,
        math_score=math_score,
        structure_score=structure_score,
        overall_score=overall_score,
    )


def _load_raw_ocr_payload(path: Path) -> dict[str, Any]:
    return validate_raw_ocr_document(_read_json(path))


def _ocr_single_page_with_retries(
    *,
    client: OpenAI,
    base64_image: str,
    model: str,
    fmt: str,
    prompt_text: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> tuple[str, ocr.OCRPageUsage]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return ocr._ocr_page_with_usage(
                client,
                base64_image,
                model=model,
                fmt=fmt,
                prompt=prompt_text,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                continue

    raise RuntimeError(f"OCR failed after {max_retries} attempts") from last_error


def _render_case_page_image(
    *,
    source_path: Path,
    page: int,
    dpi: int,
    fmt: str,
) -> str:
    if source_path.suffix.lower() == ".pdf":
        with fitz.open(source_path) as doc:
            if page > len(doc):
                raise ValueError(
                    f"Requested page {page} for '{source_path.name}', but document has only {len(doc)} pages."
                )
            return ocr.render_page_to_image(doc, page - 1, dpi=dpi, fmt=fmt)

    if source_path.suffix.lower() in ocr.SUPPORTED_OCR_IMAGE_EXTENSIONS:
        if page != 1:
            raise ValueError(
                f"Image benchmark case '{source_path.name}' must use page 1, got {page}."
            )
        return ocr.render_image_to_image(source_path, fmt=fmt)

    raise ValueError(
        f"Unsupported benchmark source format for '{source_path.name}': {source_path.suffix or '<none>'}"
    )


def _summarize_model_cases(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [result for result in case_results if result["error"] is None]

    billed_cases = [result for result in successful if result.get("cost_usd") is not None]
    pages_billed = len(billed_cases)
    prompt_tokens = sum(int(result.get("prompt_tokens", 0)) for result in billed_cases)
    completion_tokens = sum(
        int(result.get("completion_tokens", 0)) for result in billed_cases
    )
    total_tokens = sum(int(result.get("total_tokens", 0)) for result in billed_cases)
    total_cost_usd = _safe_non_negative_float(
        sum(float(result.get("cost_usd", 0.0)) for result in billed_cases)
    )
    dollars_per_1000_pages = _dollars_per_1000_pages(
        total_cost_usd=total_cost_usd,
        pages=pages_billed,
    )

    if not successful:
        return {
            "cases_total": len(case_results),
            "cases_scored": 0,
            "cases_failed": len(case_results),
            "pages_billed": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "dollars_per_1000_pages": 0.0,
            "text_score": 0.0,
            "math_score": 0.0,
            "structure_score": 0.0,
            "overall_score": 0.0,
        }

    return {
        "cases_total": len(case_results),
        "cases_scored": len(successful),
        "cases_failed": len(case_results) - len(successful),
        "pages_billed": pages_billed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost_usd,
        "dollars_per_1000_pages": dollars_per_1000_pages,
        "text_score": _safe_float(mean(result["text_score"] for result in successful)),
        "math_score": _safe_float(mean(result["math_score"] for result in successful)),
        "structure_score": _safe_float(
            mean(result["structure_score"] for result in successful)
        ),
        "overall_score": _safe_float(
            mean(result["overall_score"] for result in successful)
        ),
    }


class BenchmarkHistoryDatabase:
    """SQLite-backed history store for benchmark runs and scores."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path)
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL,
                manifest_path TEXT NOT NULL,
                manifest_name TEXT NOT NULL,
                manifest_version TEXT NOT NULL,
                models_json TEXT NOT NULL,
                args_json TEXT NOT NULL,
                report_path TEXT,
                error TEXT
            );

            CREATE TABLE IF NOT EXISTS benchmark_case_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                model TEXT NOT NULL,
                case_id TEXT NOT NULL,
                document TEXT NOT NULL,
                page INTEGER NOT NULL,
                text_score REAL,
                math_score REAL,
                structure_score REAL,
                overall_score REAL,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                cost_usd REAL,
                dollars_per_1000_pages REAL,
                candidate_json_path TEXT,
                gold_json_path TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_benchmark_case_results_run_model
                ON benchmark_case_results(run_id, model);

            CREATE TABLE IF NOT EXISTS benchmark_model_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                model TEXT NOT NULL,
                cases_total INTEGER NOT NULL,
                cases_scored INTEGER NOT NULL,
                cases_failed INTEGER NOT NULL,
                text_score REAL NOT NULL,
                math_score REAL NOT NULL,
                structure_score REAL NOT NULL,
                overall_score REAL NOT NULL,
                pages_billed INTEGER NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                total_cost_usd REAL NOT NULL,
                dollars_per_1000_pages REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_benchmark_model_summaries_run
                ON benchmark_model_summaries(run_id);
            """
        )
        self._ensure_column("benchmark_case_results", "prompt_tokens INTEGER")
        self._ensure_column("benchmark_case_results", "completion_tokens INTEGER")
        self._ensure_column("benchmark_case_results", "total_tokens INTEGER")
        self._ensure_column("benchmark_case_results", "cost_usd REAL")
        self._ensure_column(
            "benchmark_case_results", "dollars_per_1000_pages REAL"
        )

        self._ensure_column("benchmark_model_summaries", "pages_billed INTEGER")
        self._ensure_column("benchmark_model_summaries", "prompt_tokens INTEGER")
        self._ensure_column(
            "benchmark_model_summaries", "completion_tokens INTEGER"
        )
        self._ensure_column("benchmark_model_summaries", "total_tokens INTEGER")
        self._ensure_column("benchmark_model_summaries", "total_cost_usd REAL")
        self._ensure_column(
            "benchmark_model_summaries", "dollars_per_1000_pages REAL"
        )
        self.connection.commit()

    def _ensure_column(self, table: str, column_definition: str) -> None:
        try:
            self.connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column_definition}"
            )
        except sqlite3.OperationalError as exc:
            if "duplicate column name" in str(exc).lower():
                return
            raise

    def start_run(
        self,
        *,
        manifest_path: Path,
        manifest_name: str,
        manifest_version: str,
        models: list[str],
        args_payload: dict[str, Any],
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO benchmark_runs (
                started_at,
                status,
                manifest_path,
                manifest_name,
                manifest_version,
                models_json,
                args_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _utc_now_iso(),
                "running",
                str(manifest_path),
                manifest_name,
                manifest_version,
                json.dumps(models, ensure_ascii=False),
                json.dumps(args_payload, ensure_ascii=False),
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def record_case_result(self, *, run_id: int, model: str, result: dict[str, Any]) -> None:
        self.connection.execute(
            """
            INSERT INTO benchmark_case_results (
                run_id,
                model,
                case_id,
                document,
                page,
                text_score,
                math_score,
                structure_score,
                overall_score,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost_usd,
                dollars_per_1000_pages,
                candidate_json_path,
                gold_json_path,
                error,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                model,
                result["case_id"],
                result["document"],
                result["page"],
                result["text_score"],
                result["math_score"],
                result["structure_score"],
                result["overall_score"],
                result["prompt_tokens"],
                result["completion_tokens"],
                result["total_tokens"],
                result["cost_usd"],
                result["dollars_per_1000_pages"],
                result["candidate_json_path"],
                result["gold_json_path"],
                result["error"],
                _utc_now_iso(),
            ),
        )
        self.connection.commit()

    def record_model_summary(
        self,
        *,
        run_id: int,
        model: str,
        summary: dict[str, Any],
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO benchmark_model_summaries (
                run_id,
                model,
                cases_total,
                cases_scored,
                cases_failed,
                text_score,
                math_score,
                structure_score,
                overall_score,
                pages_billed,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                total_cost_usd,
                dollars_per_1000_pages,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                model,
                summary["cases_total"],
                summary["cases_scored"],
                summary["cases_failed"],
                summary["text_score"],
                summary["math_score"],
                summary["structure_score"],
                summary["overall_score"],
                summary["pages_billed"],
                summary["prompt_tokens"],
                summary["completion_tokens"],
                summary["total_tokens"],
                summary["total_cost_usd"],
                summary["dollars_per_1000_pages"],
                _utc_now_iso(),
            ),
        )
        self.connection.commit()

    def finish_run(
        self,
        *,
        run_id: int,
        status: str,
        report_path: Path | None,
        error: str | None,
    ) -> None:
        self.connection.execute(
            """
            UPDATE benchmark_runs
               SET completed_at = ?,
                   status = ?,
                   report_path = ?,
                   error = ?
             WHERE id = ?
            """,
            (
                _utc_now_iso(),
                status,
                str(report_path) if report_path is not None else None,
                error,
                run_id,
            ),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def get_recent_benchmark_results(
    *,
    database_path: Path,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return recent benchmark model summary rows from history DB."""
    if limit < 1:
        raise ValueError("limit must be >= 1.")

    resolved_database = Path(database_path)
    if not resolved_database.exists():
        return []

    try:
        with sqlite3.connect(resolved_database) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    br.id AS run_id,
                    br.completed_at,
                    br.status,
                    ms.model,
                    ms.cases_total,
                    ms.cases_scored,
                    ms.cases_failed,
                    ms.overall_score,
                    ms.total_cost_usd,
                    ms.dollars_per_1000_pages
                  FROM benchmark_model_summaries AS ms
                  JOIN benchmark_runs AS br
                    ON br.id = ms.run_id
                 ORDER BY br.id DESC, ms.id DESC
                 LIMIT ?
                """,
                (limit,),
            ).fetchall()
    except sqlite3.OperationalError:
        return []

    return [dict(row) for row in rows]


def run_benchmark(
    *,
    manifest_path: Path,
    docs_dir: Path = DEFAULT_DOCS_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    database_path: Path | None = None,
    reports_dir: Path | None = None,
    api_key: str | None = None,
    models: list[str] | None = None,
    prompt_template: str = ocr.DEFAULT_OCR_PROMPT_TEMPLATE,
    dpi: int = ocr.DEFAULT_OCR_DPI,
    fmt: str = ocr.DEFAULT_OCR_IMAGE_FORMAT,
    temperature: float = ocr.DEFAULT_VLM_TEMPERATURE,
    max_tokens: int = ocr.DEFAULT_OCR_MAX_TOKENS,
    max_retries: int = ocr.DEFAULT_OCR_MAX_RETRIES,
    case_limit: int | None = None,
    case_ids: list[str] | None = None,
    output_fn: OutputFunc = print,
) -> dict[str, Any]:
    """Run live benchmark OCR against a deterministic one-page-per-case manifest."""
    docs_dir = Path(docs_dir)
    out_dir = Path(out_dir)
    manifest_path = Path(manifest_path)

    manifest = load_manifest(manifest_path)

    benchmark_docs_dir = docs_dir / _BENCHMARK_DOCS_SUBPATH
    if benchmark_docs_dir.exists():
        verify_benchmark_gold_for_pdf_folder(
            docs_dir=docs_dir,
            manifest_path=manifest_path,
            benchmark_subpath=_BENCHMARK_DOCS_SUBPATH,
            require_folder=False,
            output_fn=output_fn,
        )

    selected_cases = list(manifest.cases)
    if case_ids:
        wanted = set(case_ids)
        selected_cases = [case for case in selected_cases if case.case_id in wanted]

    if case_limit is not None:
        if case_limit < 1:
            raise ValueError("case_limit must be >= 1 when provided.")
        selected_cases = selected_cases[:case_limit]

    if not selected_cases:
        raise ValueError("No benchmark cases selected. Check --case-id and --case-limit.")

    selected_models = [model.strip() for model in (models or [])]
    selected_models = [model for model in selected_models if model]
    if not selected_models:
        raise ValueError("At least one non-empty model id is required.")

    prompt_text = ocr.read_ocr_prompt_template(prompt_template)
    client = ocr.create_client(api_key=api_key)

    resolved_database = Path(database_path or get_default_database_path(out_dir=out_dir))
    resolved_reports_dir = Path(reports_dir or get_default_reports_dir(out_dir=out_dir))
    resolved_reports_dir.mkdir(parents=True, exist_ok=True)

    candidate_root = get_default_candidates_dir(out_dir=out_dir)

    db = BenchmarkHistoryDatabase(resolved_database)
    run_id: int | None = None
    report_path: Path | None = None

    try:
        run_id = db.start_run(
            manifest_path=manifest_path,
            manifest_name=manifest.name,
            manifest_version=manifest.version,
            models=selected_models,
            args_payload={
                "docs_dir": str(docs_dir),
                "manifest_path": str(manifest_path),
                "prompt_template": prompt_template,
                "dpi": dpi,
                "fmt": fmt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "max_retries": max_retries,
                "case_limit": case_limit,
                "case_ids": case_ids or [],
            },
        )

        run_token = (
            f"run-{run_id:06d}-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )

        model_reports: list[dict[str, Any]] = []

        for model in selected_models:
            output_fn(f"Benchmarking model: {model}")
            model_slug = _slugify_model_name(model)
            model_dir = candidate_root / run_token / model_slug
            model_dir.mkdir(parents=True, exist_ok=True)

            case_results: list[dict[str, Any]] = []
            for case in selected_cases:
                gold_path = (manifest_path.parent / case.gold_json).resolve()
                source_path = docs_dir / case.document

                result: dict[str, Any]
                try:
                    if not source_path.exists():
                        raise FileNotFoundError(
                            f"Benchmark source document not found: {source_path}"
                        )
                    if not gold_path.exists():
                        raise FileNotFoundError(
                            f"Benchmark gold JSON not found: {gold_path}"
                        )

                    base64_image = _render_case_page_image(
                        source_path=source_path,
                        page=case.page,
                        dpi=dpi,
                        fmt=fmt,
                    )
                    markdown, usage = _ocr_single_page_with_retries(
                        client=client,
                        base64_image=base64_image,
                        model=model,
                        fmt=fmt,
                        prompt_text=prompt_text,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        max_retries=max_retries,
                    )

                    candidate_payload = build_raw_ocr_document(
                        [markdown],
                        settings_hash=ocr.hash_ocr_settings(
                            model=model,
                            dpi=dpi,
                            fmt=fmt,
                            prompt=prompt_text,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        ),
                    )
                    candidate_json_path = model_dir / f"{case.case_id}.json"
                    _write_json(candidate_json_path, candidate_payload)

                    gold_payload = _load_raw_ocr_payload(gold_path)
                    score = score_markdown_pair(
                        gold_payload["pages"][0]["markdown"],
                        candidate_payload["pages"][0]["markdown"],
                    )

                    result = {
                        "case_id": case.case_id,
                        "document": case.document,
                        "page": case.page,
                        "text_score": score.text_score,
                        "math_score": score.math_score,
                        "structure_score": score.structure_score,
                        "overall_score": score.overall_score,
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                        "cost_usd": usage.cost,
                        "dollars_per_1000_pages": (
                            _dollars_per_1000_pages(
                                total_cost_usd=_safe_non_negative_float(usage.cost),
                                pages=1,
                            )
                            if usage.cost is not None
                            else None
                        ),
                        "candidate_json_path": str(candidate_json_path),
                        "gold_json_path": str(gold_path),
                        "error": None,
                    }
                except Exception as exc:
                    result = {
                        "case_id": case.case_id,
                        "document": case.document,
                        "page": case.page,
                        "text_score": 0.0,
                        "math_score": 0.0,
                        "structure_score": 0.0,
                        "overall_score": 0.0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "cost_usd": None,
                        "dollars_per_1000_pages": None,
                        "candidate_json_path": None,
                        "gold_json_path": str(gold_path),
                        "error": str(exc),
                    }

                db.record_case_result(run_id=run_id, model=model, result=result)
                case_results.append(result)

            summary = _summarize_model_cases(case_results)
            db.record_model_summary(run_id=run_id, model=model, summary=summary)

            output_fn(
                "  "
                f"overall={summary['overall_score']:.3f}, "
                f"text={summary['text_score']:.3f}, "
                f"math={summary['math_score']:.3f}, "
                f"structure={summary['structure_score']:.3f}, "
                f"cost=${summary['total_cost_usd']:.6f}, "
                f"$/1k_pages=${summary['dollars_per_1000_pages']:.2f}, "
                f"failed={summary['cases_failed']}"
            )

            model_reports.append(
                {
                    "model": model,
                    "summary": summary,
                    "cases": case_results,
                }
            )

        ranking = sorted(
            (
                {
                    "model": report["model"],
                    "overall_score": report["summary"]["overall_score"],
                    "cases_failed": report["summary"]["cases_failed"],
                    "total_cost_usd": report["summary"]["total_cost_usd"],
                    "dollars_per_1000_pages": report["summary"][
                        "dollars_per_1000_pages"
                    ],
                }
                for report in model_reports
            ),
            key=lambda row: row["overall_score"],
            reverse=True,
        )

        report_payload = {
            "run_id": run_id,
            "created_at": _utc_now_iso(),
            "manifest": {
                "path": str(manifest_path),
                "name": manifest.name,
                "version": manifest.version,
                "case_count": len(selected_cases),
            },
            "models": model_reports,
            "ranking": ranking,
        }

        report_path = resolved_reports_dir / f"{run_token}.json"
        _write_json(report_path, report_payload)

        db.finish_run(
            run_id=run_id,
            status="completed",
            report_path=report_path,
            error=None,
        )

        output_fn(f"Benchmark report written: {report_path}")
        output_fn(f"Benchmark history database: {resolved_database}")

        return report_payload
    except Exception as exc:
        if run_id is not None:
            db.finish_run(
                run_id=run_id,
                status="failed",
                report_path=report_path,
                error=str(exc),
            )
        raise
    finally:
        db.close()
