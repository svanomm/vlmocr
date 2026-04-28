# vlmocr

`vlmocr` is a standalone OCR package for rendering PDFs, sending page images to OpenRouter-served VLMs, validating the raw OCR contract, and converting the results into cleaned Markdown and cleaned JSON.

It preserves the external interface expected by downstream consumers:

- `vlmocr`
- `vlmocr init`
- `vlmocr ocr`
- `vlmocr convert`
- `vlmocr estimate-cost`

The frozen compatibility details live in [OCR_SPINOUT_CONTRACT.md](OCR_SPINOUT_CONTRACT.md).

## Requirements

- Python 3.12+
- `uv`
- `OPENROUTER_API_KEY` for OCR commands

Install the repo environment:

```bash
uv sync
```

OpenRouter setup for first-time users:

1. Create an account at `https://openrouter.ai/` if you do not already have one.
2. Create an API key at `https://openrouter.ai/keys`.
3. Put the key in a `.env` file in the project root:

```bash
OPENROUTER_API_KEY=your_key_here
```

You can also pass the key directly with `--api-key`, but the `.env` file is the simplest repeatable setup.

## Commands

First-run interactive launcher:

```bash
uv run vlmocr
```

This opens a terminal menu that lets you initialize the workspace, inspect the expected directory layout, and run the available commands with prompts.
It also explains where to create an OpenRouter API key and how to store it before you run OCR.

Recommended first-run setup:

```bash
uv run vlmocr init
```

`init` creates the default project structure and prints the next commands to run:

- `docs`
- `.search/converted/json/raw`
- `.search/converted/json`
- `.search/converted/md`
- `.search/converted/md/table of contents`

The `init` output also tells you to create an OpenRouter API key at `https://openrouter.ai/keys` and store it as `OPENROUTER_API_KEY` in a `.env` file in the project root before running OCR.

Local script entry point:

```bash
uv run vlmocr init
uv run vlmocr ocr --docs-dir docs --out-dir .search/converted
uv run vlmocr convert --input-dir .search/converted/json/raw --out-dir .search/converted
uv run vlmocr estimate-cost --docs-dir docs
```

Module execution:

```bash
uv run -m vlmocr init
uv run -m vlmocr ocr --docs-dir docs --out-dir .search/converted
uv run -m vlmocr convert --input-dir .search/converted/json/raw --out-dir .search/converted
uv run -m vlmocr estimate-cost --docs-dir docs
```

Release-mode `uvx` examples:

```bash
uvx --from git+https://github.com/<owner>/<repo>@v0.1.0 vlmocr estimate-cost --docs-dir docs
uvx --from git+https://github.com/<owner>/<repo>@v0.1.0 vlmocr ocr --docs-dir docs --out-dir .search/converted
uvx --from git+https://github.com/<owner>/<repo>@v0.1.0 vlmocr convert --input-dir .search/converted/json/raw --out-dir .search/converted
```

## Defaults and options

Defaults:

- `--docs-dir docs`
- `--out-dir .search/converted`
- `vlmocr` with no subcommand opens the interactive launcher
- `convert` without `--input-dir` reads from `<out-dir>/json/raw`

OCR options:

- `--api-key`
- `--model`
- `--dpi`
- `--format {png,jpeg}`
- `--max-workers`
- `--max-retries`

Environment overrides:

- `OPENROUTER_API_KEY`
- `VLMOCR_MODEL`
- `VLMOCR_DPI`
- `VLMOCR_IMAGE_FORMAT`
- `VLMOCR_MAX_TOKENS`
- `VLMOCR_MAX_WORKERS`
- `VLMOCR_MAX_RETRIES`

If `OPENROUTER_API_KEY` is missing, OCR fails with a clear error that tells you where to create the key and how to provide it through `.env`, environment variables, or `--api-key`.

## Output tree

Given `--out-dir <root>`, the package writes:

```text
<root>/json/raw/<name>.json
<root>/json/<name>.json
<root>/md/<name>.md
<root>/md/table of contents/<name>_toc.md
```

`estimate-cost` does not write output artifacts. It only inspects PDFs under `--docs-dir` and prints OCR-only cost estimates.

## Raw OCR schema

Raw OCR JSON must match this schema exactly:

```json
{
  "pages": [
    {
      "index": 0,
      "markdown": "# Page 1 markdown"
    }
  ]
}
```

Rules:

- `pages` is an ordered array
- each `index` is an integer
- indexes are sequential starting at `0`
- each `markdown` value is a string

`convert` validates this contract and rejects malformed or invalid raw input.
If the expected raw OCR input directory does not exist yet, the CLI now explains how to run `vlmocr init` and `vlmocr ocr` instead of showing a traceback.

## Development checks

```bash
uv run ruff check --fix
uv run ruff format
uv run pytest
```

## Notes

- OCR cost estimation is intentionally OCR-only.
- Repeated-line cleanup is optional and is designed to avoid over-removing lines in smaller documents.
