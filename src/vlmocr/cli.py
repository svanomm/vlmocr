"""Command-line interface for vlmocr."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Sequence

from vlmocr import benchmark, conversion, estimate_cost, ocr
from vlmocr.contract import (
    DEFAULT_DOCS_DIR,
    DEFAULT_OUT_DIR,
    get_raw_ocr_dir,
    initialize_project_structure,
    validate_project_structure,
)

InputFunc = Callable[[str], str]
OutputFunc = Callable[[str], None]

MENU_MIN_WIDTH = 68
MENU_MAX_WIDTH = 94
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_ANSI_RESET = "\033[0m"

_LOGO_LINES = (
    r"  _   ____   __  ___  ____  ________ ",
    r" | | / / /  /  |/  / / __ \/ ___/ _ \\",
    r" | |/ / /__/ /|_/ / / /_/ / /__/ , _/",
    r" |___/____/_/  /_/  \____/\___/_/|_| ",
    r"                                     ",
)


def _supports_ansi(output_fn: OutputFunc) -> bool:
    if output_fn is not print:
        return False

    if os.environ.get("NO_COLOR"):
        return False

    stream = getattr(sys, "stdout", None)
    return bool(stream and stream.isatty())


def _style(text: str, *codes: str, enabled: bool) -> str:
    if not enabled or not codes:
        return text

    return f"\033[{';'.join(codes)}m{text}{_ANSI_RESET}"


def _visible_len(text: str) -> int:
    return len(_ANSI_RE.sub("", text))


def _menu_width() -> int:
    columns = shutil.get_terminal_size((MENU_MAX_WIDTH, 24)).columns
    return max(MENU_MIN_WIDTH, min(MENU_MAX_WIDTH, columns - 2))


def _center_text(text: str, width: int) -> str:
    visible_width = _visible_len(text)
    if visible_width >= width:
        return text

    padding = width - visible_width
    left = padding // 2
    right = padding - left
    return f"{' ' * left}{text}{' ' * right}"


def _truncate_text(text: str, width: int) -> str:
    if _visible_len(text) <= width:
        return text

    if width <= 3:
        return text[:width]

    return f"{text[: width - 3]}..."


def _pad_visible_right(text: str, width: int) -> str:
    visible_width = _visible_len(text)
    if visible_width >= width:
        return text

    return f"{text}{' ' * (width - visible_width)}"


def _render_panel(
    title: str,
    lines: Sequence[str],
    *,
    ansi_enabled: bool,
    border_color: str = "38;5;67",
    title_color: str = "1;38;5;123",
) -> str:
    width = _menu_width()
    inner_width = width - 4
    title_text = f" {title} " if title else ""
    dash_count = max(0, width - 2 - len(title_text))
    if ansi_enabled and title:
        top = "".join(
            [
                _style("+", border_color, enabled=True),
                _style(title_text, title_color, enabled=True),
                _style(f"{'-' * dash_count}+", border_color, enabled=True),
            ]
        )
    else:
        top = _style(
            f"+{title_text}{'-' * dash_count}+",
            border_color,
            enabled=ansi_enabled,
        )
    bottom = f"+{'-' * (width - 2)}+"
    rendered_lines = [top]

    for line in lines:
        clipped = _truncate_text(line, inner_width)
        rendered_lines.append(
            _style(
                f"| {_pad_visible_right(clipped, inner_width)} |",
                border_color,
                enabled=ansi_enabled,
            )
        )

    rendered_lines.append(_style(bottom, border_color, enabled=ansi_enabled))
    return "\n".join(rendered_lines)


def _render_logo(*, ansi_enabled: bool) -> str:
    width = _menu_width() - 4
    colors = ("1;38;5;45", "1;38;5;81", "1;38;5;117", "1;38;5;153", "1;38;5;189")
    lines = [
        _style(_center_text(line, width), color, enabled=ansi_enabled)
        for line, color in zip(_LOGO_LINES, colors, strict=True)
    ]
    lines.append("")
    lines.append(
        _style(
            _center_text("interactive launcher for docs -> OCR -> markdown", width),
            "38;5;145",
            enabled=ansi_enabled,
        )
    )
    return _render_panel("vlmocr", lines, ansi_enabled=ansi_enabled, border_color="38;5;75")


def _render_menu(*, ansi_enabled: bool) -> str:
    docs_line = f"docs    {DEFAULT_DOCS_DIR}"
    out_line = f"output  {DEFAULT_OUT_DIR}"
    menu_lines = [
        "mission control",
        "",
        docs_line,
        out_line,
        "",
        "[1] Init project structure",
        "[2] Run OCR on documents (PDF + images)",
        "[3] Convert raw OCR JSON",
        "[4] Validate current structure",
        "[5] Show quickstart",
        "[6] Show/edit default OCR prompt",
        "[7] Quit",
        "",
        "tip: press Ctrl+C at any prompt to leave the launcher.",
    ]
    return _render_panel(
        "interactive launcher",
        menu_lines,
        ansi_enabled=ansi_enabled,
        border_color="38;5;66",
        title_color="1;38;5;153",
    )


def _render_status_message(message: str, *, ansi_enabled: bool) -> str:
    return _style(message, "38;5;180", enabled=ansi_enabled)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level vlmocr argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="vlmocr",
        description="Document OCR (PDF + images), conversion, and OCR cost estimation.",
    )
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser(
        "init",
        help="Create and validate the standard vlmocr project structure.",
    )
    init_parser.set_defaults(docs_dir=DEFAULT_DOCS_DIR, out_dir=DEFAULT_OUT_DIR)

    ocr_parser = subparsers.add_parser(
        "ocr",
        help="Render documents (PDF + images) and write raw per-page OCR JSON.",
    )
    ocr_parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    ocr_parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ocr_parser.add_argument("--api-key", default=None)
    ocr_parser.add_argument("--model", default=ocr.DEFAULT_OCR_MODEL)
    ocr_parser.add_argument(
        "--prompt-template",
        default=None,
        help=(
            "Prompt template name in the OCR prompts folder. "
            "When omitted in an interactive terminal, the CLI asks which template to use."
        ),
    )
    ocr_parser.add_argument("--dpi", type=int, default=ocr.DEFAULT_OCR_DPI)
    ocr_parser.add_argument(
        "--format",
        choices=["png", "jpeg"],
        default=ocr.DEFAULT_OCR_IMAGE_FORMAT,
    )
    ocr_parser.add_argument(
        "--max-workers",
        type=int,
        default=ocr.DEFAULT_OCR_MAX_WORKERS,
    )
    ocr_parser.add_argument(
        "--max-retries",
        type=int,
        default=ocr.DEFAULT_OCR_MAX_RETRIES,
    )
    ocr_parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only scan the top level of --docs-dir for OCR inputs.",
    )
    ocr_parser.set_defaults(recursive=True)

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert raw OCR JSON into cleaned markdown and cleaned JSON.",
    )
    convert_parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing raw OCR JSON. Defaults to <out-dir>/json/raw.",
    )
    convert_parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    convert_parser.add_argument(
        "--remove-frequent-lines",
        action="store_true",
        help="Remove repeated header/footer-like lines that appear on many pages.",
    )
    convert_parser.add_argument(
        "--no-inject-footnotes",
        dest="inject_footnotes",
        action="store_false",
        help="Leave <ref> and <note> tags in place instead of expanding them inline.",
    )
    convert_parser.set_defaults(inject_footnotes=True)

    estimate_parser = subparsers.add_parser(
        "estimate-cost",
        help="Estimate OCR-only page and token costs.",
    )
    estimate_parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    estimate_parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only scan the top level of --docs-dir when estimating cost.",
    )
    estimate_parser.set_defaults(recursive=True)

    benchmark_init_parser = subparsers.add_parser(
        "benchmark-init-academic",
        help="Create a local 10-page academic benchmark manifest and one-page gold files.",
    )
    benchmark_init_parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    benchmark_init_parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    benchmark_init_parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory containing seeded raw OCR JSON. Defaults to <out-dir>/json/raw.",
    )
    benchmark_init_parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest output path. Defaults to <out-dir>/benchmark/academic-textbook-v1/manifest.json.",
    )
    benchmark_init_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing benchmark manifest and gold pages.",
    )

    benchmark_repair_parser = subparsers.add_parser(
        "benchmark-repair-gold-escapes",
        help=(
            "Repair single-backslash LaTeX-style escapes in benchmark gold markdown "
            "(leaves \\n sequences unchanged)."
        ),
    )
    benchmark_repair_parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Benchmark manifest path. Defaults to <out-dir>/benchmark/academic-textbook-v1/manifest.json.",
    )
    benchmark_repair_parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Used to resolve default manifest path when --manifest is omitted.",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run deterministic one-page benchmark cases for one or more models.",
    )
    benchmark_parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Benchmark manifest path. Defaults to <out-dir>/benchmark/academic-textbook-v1/manifest.json.",
    )
    benchmark_parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    benchmark_parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    benchmark_parser.add_argument("--api-key", default=None)
    benchmark_parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model identifier. Repeat --model to compare multiple models.",
    )
    benchmark_parser.add_argument(
        "--prompt-template",
        default=ocr.DEFAULT_OCR_PROMPT_TEMPLATE,
        help="OCR prompt template name for benchmark calls.",
    )
    benchmark_parser.add_argument("--dpi", type=int, default=ocr.DEFAULT_OCR_DPI)
    benchmark_parser.add_argument(
        "--format",
        choices=["png", "jpeg"],
        default=ocr.DEFAULT_OCR_IMAGE_FORMAT,
    )
    benchmark_parser.add_argument(
        "--max-retries",
        type=int,
        default=ocr.DEFAULT_OCR_MAX_RETRIES,
    )
    benchmark_parser.add_argument(
        "--temperature",
        type=float,
        default=ocr.DEFAULT_VLM_TEMPERATURE,
    )
    benchmark_parser.add_argument(
        "--max-tokens",
        type=int,
        default=ocr.DEFAULT_OCR_MAX_TOKENS,
    )
    benchmark_parser.add_argument(
        "--case-limit",
        type=int,
        default=None,
        help="Run only the first N cases from the manifest (useful for low-cost checks).",
    )
    benchmark_parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="Case id to run. Repeat --case-id to run an explicit subset.",
    )
    benchmark_parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="SQLite path for benchmark history. Defaults to <out-dir>/benchmark/history.db.",
    )
    benchmark_parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Directory for benchmark report JSON output.",
    )

    return parser


def _format_quickstart(docs_dir: Path, out_dir: Path) -> str:
    raw_dir = get_raw_ocr_dir(out_dir)
    return "\n".join(
        [
            "Next steps:",
            "  1. Create an OpenRouter API key at https://openrouter.ai/keys",
            (
                "  2. Save it in a `.env` file in your project root as "
                "`OPENROUTER_API_KEY=your_key_here`, or pass it with `--api-key`"
            ),
            (
                f"  3. Put supported documents in {docs_dir} "
                "(.pdf, .jpg, .jpeg, .png, .webp, .bmp; recursive by default)"
            ),
            "     TIFF/GIF files are currently skipped with warnings.",
            (
                "  4. Run "
                f"`vlmocr ocr --docs-dir {docs_dir} --out-dir {out_dir}` to write raw OCR JSON"
            ),
            (
                "  5. Run "
                f"`vlmocr convert --out-dir {out_dir}` to create cleaned markdown and cleaned JSON"
            ),
            f"  6. If you already have raw OCR JSON, place it in {raw_dir} before running convert",
        ]
    )


def _print_project_status(
    output_fn: OutputFunc,
    *,
    docs_dir: Path,
    out_dir: Path,
) -> None:
    output_fn("Project structure:")
    for status in validate_project_structure(docs_dir=docs_dir, out_dir=out_dir):
        marker = "ok" if status.exists else "missing"
        output_fn(f"  [{marker}] {status.label}: {status.path}")


def run_init_command(
    *,
    docs_dir: Path = DEFAULT_DOCS_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    output_fn: OutputFunc = print,
) -> list[Path]:
    """Create the standard vlmocr directory layout and print next steps."""
    created_paths = initialize_project_structure(docs_dir=docs_dir, out_dir=out_dir)

    if created_paths:
        output_fn("Created project directories:")
        for path in created_paths:
            output_fn(f"  - {path}")
    else:
        output_fn("Project directories already exist.")

    _print_project_status(output_fn, docs_dir=docs_dir, out_dir=out_dir)
    output_fn("")
    output_fn(_format_quickstart(docs_dir=docs_dir, out_dir=out_dir))
    return created_paths


def _prompt_text(input_fn: InputFunc, label: str, *, default: str = "") -> str:
    prompt = f"{label} [{default}]: " if default else f"{label}: "
    response = input_fn(prompt).strip()
    return response or default


def _prompt_path(input_fn: InputFunc, label: str, *, default: Path) -> Path:
    return Path(_prompt_text(input_fn, label, default=str(default)))


def _prompt_int(input_fn: InputFunc, label: str, *, default: int) -> int:
    while True:
        value = _prompt_text(input_fn, label, default=str(default))
        try:
            return int(value)
        except ValueError:
            print(f"Please enter an integer for {label.lower()}.")


def _prompt_bool(input_fn: InputFunc, label: str, *, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        value = input_fn(f"{label} [{suffix}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please answer y or n.")


def _pause(input_fn: InputFunc) -> None:
    input_fn("Press Enter to return to the menu...")


def _friendly_error_message(args: argparse.Namespace, exc: Exception) -> str:
    if args.command == "benchmark-init-academic":
        return "\n".join(
            [
                str(exc),
                "",
                (
                    "Run `vlmocr ocr --docs-dir "
                    f"{args.docs_dir} --out-dir {args.out_dir}` first so raw OCR JSON exists "
                    "for benchmark seeding."
                ),
                (
                    "Then rerun `vlmocr benchmark-init-academic` to generate the local "
                    "10-page benchmark manifest and gold files."
                ),
                "",
            ]
        )

    if args.command == "benchmark":
        details = [str(exc), ""]
        if "OPENROUTER_API_KEY" in str(exc):
            details.extend(
                [
                    "Get an API key from https://openrouter.ai/keys.",
                    "Then either:",
                    "  - create a `.env` file in your project root with `OPENROUTER_API_KEY=your_key_here`",
                    "  - set the OPENROUTER_API_KEY environment variable",
                    "  - or rerun the command with `--api-key <your_key>`",
                    "",
                ]
            )

        details.extend(
            [
                (
                    "If this is your first benchmark run, initialize local benchmark data with "
                    f"`vlmocr benchmark-init-academic --docs-dir {args.docs_dir} --out-dir {args.out_dir}`."
                ),
                (
                    "For a low-cost API smoke test, run one case only: "
                    "`vlmocr benchmark --case-limit 1`"
                ),
                "",
            ]
        )
        return "\n".join(details)

    if args.command == "convert":
        input_dir = args.input_dir or get_raw_ocr_dir(args.out_dir)
        return "\n".join(
            [
                str(exc),
                "",
                f"Run `vlmocr init --out-dir {args.out_dir}` to create the standard folders.",
                (
                    "If you are starting from source documents, add them to "
                    f"{DEFAULT_DOCS_DIR} and run `vlmocr ocr --out-dir {args.out_dir}` first."
                ),
                f"If you already have raw OCR JSON, place it in {input_dir} and rerun convert.",
                "",
            ]
        )

    if args.command == "ocr":
        details = [str(exc), ""]
        if "OPENROUTER_API_KEY" in str(exc):
            details.extend(
                [
                    "Get an API key from https://openrouter.ai/keys.",
                    "Then either:",
                    "  - create a `.env` file in your project root with `OPENROUTER_API_KEY=your_key_here`",
                    "  - set the OPENROUTER_API_KEY environment variable",
                    "  - or rerun the command with `--api-key <your_key>`",
                    "",
                ]
            )

        details.extend(
            [
                (
                    f"Run `vlmocr init --docs-dir {args.docs_dir} --out-dir {args.out_dir}` "
                    "to create the standard folders."
                ),
                (
                    f"Then add supported documents to {args.docs_dir} "
                    "(.pdf, .jpg, .jpeg, .png, .webp, .bmp) and rerun OCR."
                ),
                "TIFF/GIF files are currently skipped with warnings.",
                "",
            ]
        )
        return "\n".join(details)

    return f"{exc}\n"


def _run_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    try:
        if args.command == "init":
            run_init_command(docs_dir=args.docs_dir, out_dir=args.out_dir)
            return

        if args.command == "benchmark-init-academic":
            benchmark.initialize_academic_benchmark(
                docs_dir=args.docs_dir,
                out_dir=args.out_dir,
                raw_dir=args.raw_dir,
                manifest_path=args.manifest,
                overwrite=args.overwrite,
            )
            return

        if args.command == "benchmark-repair-gold-escapes":
            benchmark.repair_gold_markdown_backslashes(
                manifest_path=args.manifest or benchmark.get_default_manifest_path(
                    out_dir=args.out_dir
                ),
            )
            return

        if args.command == "ocr":
            selected_template, prompt_text = _resolve_ocr_prompt_for_command(
                args.prompt_template
            )
            print(f"Using OCR prompt template: {selected_template}")
            ocr.ocr_documents(
                docs_dir=args.docs_dir,
                out_dir=args.out_dir,
                api_key=args.api_key,
                model=args.model,
                dpi=args.dpi,
                fmt=args.format,
                prompt=prompt_text,
                max_workers=args.max_workers,
                max_retries=args.max_retries,
                recursive=args.recursive,
            )
            return

        if args.command == "convert":
            conversion.convert_directory(
                input_dir=args.input_dir or get_raw_ocr_dir(args.out_dir),
                out_dir=args.out_dir,
                remove_frequent_lines=args.remove_frequent_lines,
                inject_footnotes=args.inject_footnotes,
            )
            return

        if args.command == "estimate-cost":
            estimate_cost.count_pages(args.docs_dir, recursive=args.recursive)
            return

        if args.command == "benchmark":
            selected_models = args.model or [ocr.DEFAULT_OCR_MODEL]
            benchmark.run_benchmark(
                manifest_path=args.manifest or benchmark.get_default_manifest_path(
                    out_dir=args.out_dir
                ),
                docs_dir=args.docs_dir,
                out_dir=args.out_dir,
                database_path=args.database,
                reports_dir=args.reports_dir,
                api_key=args.api_key,
                models=selected_models,
                prompt_template=args.prompt_template,
                dpi=args.dpi,
                fmt=args.format,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                case_limit=args.case_limit,
                case_ids=args.case_id,
            )
            return
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(status=2, message=_friendly_error_message(args, exc))

    parser.error(f"Unsupported command: {args.command}")


def _prompt_api_key(input_fn: InputFunc) -> str | None:
    value = _prompt_text(
        input_fn,
        "OpenRouter API key (leave blank to use OPENROUTER_API_KEY or a .env file in the project root)",
        default="",
    )
    return value or None


def _stdin_is_interactive() -> bool:
    stream = getattr(sys, "stdin", None)
    return bool(stream and stream.isatty())


def _create_prompt_template_interactive(
    *,
    input_fn: InputFunc,
    output_fn: OutputFunc,
) -> tuple[str, str] | None:
    template_name = _prompt_text(
        input_fn,
        "Template name (leave blank to cancel)",
        default="",
    )
    if not template_name:
        output_fn("Template creation cancelled.")
        return None

    description = _prompt_text(
        input_fn,
        "Template description",
        default="User-defined OCR prompt template.",
    )

    output_fn("Enter OCR prompt markdown one line at a time.")
    output_fn("When finished, enter a single line containing END.")

    template_lines: list[str] = []
    while True:
        line = input_fn("template> ")
        if line.strip() == "END":
            break
        template_lines.append(line)

    template_prompt = "\n".join(template_lines).strip()
    if not template_prompt:
        output_fn("Template creation cancelled because the template prompt was empty.")
        return None

    try:
        created_template = ocr.create_ocr_prompt_template(
            template_name=template_name,
            description=description,
            prompt=template_prompt,
        )
    except ValueError as exc:
        if "already exists" not in str(exc):
            output_fn(str(exc))
            return None

        output_fn(str(exc))
        if not _prompt_bool(input_fn, "Overwrite existing template", default=False):
            output_fn("Template creation cancelled.")
            return None

        created_template = ocr.create_ocr_prompt_template(
            template_name=template_name,
            description=description,
            prompt=template_prompt,
            overwrite=True,
        )

    output_fn(
        f"Saved OCR prompt template '{created_template.name}' to {created_template.path}."
    )
    return created_template.name, ocr.read_ocr_prompt_template(created_template.name)


def _select_ocr_prompt_template(
    *,
    input_fn: InputFunc,
    output_fn: OutputFunc,
    allow_create: bool,
) -> tuple[str, str]:
    while True:
        templates = ocr.list_ocr_prompt_templates()
        default_index = next(
            (
                index
                for index, template in enumerate(templates, start=1)
                if template.name == ocr.DEFAULT_OCR_PROMPT_TEMPLATE
            ),
            1,
        )

        output_fn("Available OCR prompt templates:")
        for index, template in enumerate(templates, start=1):
            default_suffix = " (default)" if index == default_index else ""
            output_fn(
                f"  [{index}] {template.name}{default_suffix} - {template.description}"
            )

        create_index = len(templates) + 1
        if allow_create:
            output_fn(f"  [{create_index}] Create new template")

        selected = input_fn(f"Select OCR prompt template [{default_index}]: ").strip()

        if not selected:
            template = templates[default_index - 1]
            return template.name, ocr.read_ocr_prompt_template(template.name)

        if allow_create and selected == str(create_index):
            created = _create_prompt_template_interactive(
                input_fn=input_fn,
                output_fn=output_fn,
            )
            if created is not None:
                return created
            continue

        try:
            selected_index = int(selected)
        except ValueError:
            output_fn("Please enter one of the listed option numbers.")
            continue

        if not 1 <= selected_index <= len(templates):
            output_fn("Please enter one of the listed option numbers.")
            continue

        template = templates[selected_index - 1]
        return template.name, ocr.read_ocr_prompt_template(template.name)


def _resolve_ocr_prompt_for_command(prompt_template: str | None) -> tuple[str, str]:
    if prompt_template:
        normalized_template = ocr.normalize_prompt_template_name(prompt_template)
        return normalized_template, ocr.read_ocr_prompt_template(normalized_template)

    if _stdin_is_interactive():
        print("Select an OCR prompt template before OCR starts.")
        return _select_ocr_prompt_template(
            input_fn=input,
            output_fn=print,
            allow_create=True,
        )

    default_template = ocr.DEFAULT_OCR_PROMPT_TEMPLATE
    return default_template, ocr.read_ocr_prompt_template(default_template)


def _run_interactive_ocr(
    *,
    input_fn: InputFunc,
    output_fn: OutputFunc,
) -> None:
    use_defaults = _prompt_bool(input_fn, "Use default OCR options?", default=True)

    docs_dir = DEFAULT_DOCS_DIR
    out_dir = DEFAULT_OUT_DIR

    if use_defaults:
        api_key = None
        model = ocr.DEFAULT_OCR_MODEL
        dpi = ocr.DEFAULT_OCR_DPI
        fmt = ocr.DEFAULT_OCR_IMAGE_FORMAT
        max_workers = ocr.DEFAULT_OCR_MAX_WORKERS
        max_retries = ocr.DEFAULT_OCR_MAX_RETRIES
    else:
        output_fn(
            "You need an OpenRouter API key for OCR. Create one at https://openrouter.ai/keys."
        )
        output_fn(
            "You can paste it now, or leave it blank and set OPENROUTER_API_KEY in a .env file later."
        )
        api_key = _prompt_api_key(input_fn)
        model = _prompt_text(input_fn, "Model", default=ocr.DEFAULT_OCR_MODEL)
        dpi = _prompt_int(input_fn, "DPI", default=ocr.DEFAULT_OCR_DPI)
        fmt = _prompt_text(input_fn, "Image format", default=ocr.DEFAULT_OCR_IMAGE_FORMAT)
        max_workers = _prompt_int(
            input_fn,
            "Max workers",
            default=ocr.DEFAULT_OCR_MAX_WORKERS,
        )
        max_retries = _prompt_int(
            input_fn,
            "Max retries",
            default=ocr.DEFAULT_OCR_MAX_RETRIES,
        )

    try:
        selected_template, prompt_text = _select_ocr_prompt_template(
            input_fn=input_fn,
            output_fn=output_fn,
            allow_create=True,
        )
        output_fn(f"Using OCR prompt template: {selected_template}")
    except ValueError as exc:
        output_fn(str(exc))
        return

    try:
        to_convert = ocr.check_conversions(
            docs_dir=docs_dir,
            out_dir=out_dir,
            model=model,
            dpi=dpi,
            fmt=fmt,
            prompt=prompt_text,
            output_fn=output_fn,
        )
        if not to_convert:
            output_fn("No files need conversion.")
            return

        estimated_cost = estimate_cost.count_pages_for_files(
            [document.path for document in to_convert],
            output_fn=output_fn,
            source_label=f"{docs_dir} (pending OCR)",
        )
        if estimated_cost is None:
            return

        if not _prompt_bool(
            input_fn,
            "Proceed with OCR using the estimated cost above?",
            default=False,
        ):
            output_fn("OCR cancelled.")
            return

        ocr.ocr_documents(
            docs_dir=docs_dir,
            out_dir=out_dir,
            api_key=api_key,
            model=model,
            dpi=dpi,
            fmt=fmt,
            prompt=prompt_text,
            max_workers=max_workers,
            max_retries=max_retries,
            documents=to_convert,
        )
    except (FileNotFoundError, ValueError) as exc:
        output_fn(
            _friendly_error_message(
                argparse.Namespace(command="ocr", docs_dir=docs_dir, out_dir=out_dir),
                exc,
            )
        )


def _show_or_edit_ocr_prompt(
    *,
    input_fn: InputFunc,
    output_fn: OutputFunc,
) -> None:
    prompt_path = ocr.get_ocr_prompt_path()

    try:
        prompt_text = ocr.read_ocr_prompt(prompt_path)
    except ValueError as exc:
        output_fn(str(exc))
        return

    output_fn(f"OCR prompt file: {prompt_path}")
    output_fn(f"----- begin {prompt_path.name} -----")
    output_fn(prompt_text)
    output_fn(f"----- end {prompt_path.name} -----")

    if not _prompt_bool(input_fn, "Edit OCR prompt now", default=False):
        return

    output_fn("Enter updated OCR prompt markdown one line at a time.")
    output_fn("When finished, enter a single line containing END.")
    edited_lines: list[str] = []
    while True:
        line = input_fn("prompt> ")
        if line.strip() == "END":
            break
        edited_lines.append(line)

    updated_prompt = "\n".join(edited_lines).strip()
    if not updated_prompt:
        output_fn("OCR prompt update cancelled because the new prompt was empty.")
        return

    try:
        ocr.write_ocr_prompt(updated_prompt, prompt_path=prompt_path)
    except ValueError as exc:
        output_fn(str(exc))
        return

    output_fn(f"Saved OCR prompt to {prompt_path}.")


def launch_tui(
    *,
    input_fn: InputFunc = input,
    output_fn: OutputFunc = print,
) -> None:
    """Launch an interactive terminal menu for new and occasional users."""
    ansi_enabled = _supports_ansi(output_fn)

    output_fn("")
    output_fn(_render_logo(ansi_enabled=ansi_enabled))

    while True:
        output_fn("")
        output_fn(_render_menu(ansi_enabled=ansi_enabled))

        try:
            choice = input_fn("Select an option [1-7]: ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            output_fn(_render_status_message("Exiting vlmocr.", ansi_enabled=ansi_enabled))
            return

        if choice == "1":
            run_init_command(
                docs_dir=DEFAULT_DOCS_DIR,
                out_dir=DEFAULT_OUT_DIR,
                output_fn=output_fn,
            )
            _pause(input_fn)
            continue

        if choice == "2":
            _run_interactive_ocr(input_fn=input_fn, output_fn=output_fn)
            _pause(input_fn)
            continue

        if choice == "3":
            out_dir = DEFAULT_OUT_DIR
            input_dir = get_raw_ocr_dir(out_dir)
            remove_frequent_lines = _prompt_bool(
                input_fn,
                "Remove repeated header/footer lines",
                default=False,
            )
            inject_footnotes = _prompt_bool(
                input_fn,
                "Inject footnotes inline",
                default=True,
            )
            try:
                conversion.convert_directory(
                    input_dir=input_dir,
                    out_dir=out_dir,
                    remove_frequent_lines=remove_frequent_lines,
                    inject_footnotes=inject_footnotes,
                )
            except (FileNotFoundError, ValueError) as exc:
                output_fn(
                    _friendly_error_message(
                        argparse.Namespace(
                            command="convert",
                            input_dir=input_dir,
                            out_dir=out_dir,
                        ),
                        exc,
                    )
                )
            _pause(input_fn)
            continue

        if choice == "4":
            _print_project_status(
                output_fn,
                docs_dir=DEFAULT_DOCS_DIR,
                out_dir=DEFAULT_OUT_DIR,
            )
            _pause(input_fn)
            continue

        if choice == "5":
            output_fn(_format_quickstart(DEFAULT_DOCS_DIR, DEFAULT_OUT_DIR))
            _pause(input_fn)
            continue

        if choice == "6":
            _show_or_edit_ocr_prompt(input_fn=input_fn, output_fn=output_fn)
            _pause(input_fn)
            continue

        if choice == "7":
            output_fn(_render_status_message("Exiting vlmocr.", ansi_enabled=ansi_enabled))
            return

        if choice in {"q", "quit", "exit"}:
            output_fn(_render_status_message("Exiting vlmocr.", ansi_enabled=ansi_enabled))
            return

        output_fn(
            _render_status_message(
                "Please choose a menu option from 1 to 7.",
                ansi_enabled=ansi_enabled,
            )
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the vlmocr CLI.

    Args:
        argv: Optional argument list.
    """
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    if not raw_argv:
        launch_tui()
        return

    parser = build_parser()
    args = parser.parse_args(raw_argv)
    _run_command(args, parser)
