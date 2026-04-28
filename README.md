# vlmocr

`vlmocr` turns PDFs into clean, reusable text files.

In plain terms: it takes a PDF, renders each page as an image, asks a vision-language model (VLM) to read the page, and then saves the result as:

- Markdown for humans to read and edit
- JSON for scripts, pipelines, or downstream tools

This repo is useful when you have scanned papers, reports, manuals, or image-heavy PDFs that are hard to search, copy from, or repurpose.

Under the hood, `vlmocr` sends page images to OpenRouter-served VLMs, validates the raw OCR contract, and converts the results into cleaned Markdown and cleaned JSON. We currently use Gemini 3.1 Flash Lite for OCR based on its very high performance on [socOCRBench](https://noahdasanaike.github.io/posts/sococrbench.html) and cost effectiveness. With current API pricing, I am seeing an average of around **$1.50 per 1000 pages** of OCR.

## What OCR means

OCR stands for "optical character recognition." It is the process of turning text that appears inside an image or scanned document into actual machine-readable text.

Without OCR, a PDF scan is often just a stack of pictures. You can look at it, but searching it, copying from it, or feeding it into another tool is unreliable or impossible.

With OCR, the same document becomes much more useful:

- text can be searched
- sections can be copied and edited
- tables and headings can be preserved in a structured format
- downstream tools can process the content automatically

## Why this project uses VLMs for OCR

Traditional OCR systems are usually strongest at reading plain text characters on clean, simple pages. They can work well for straightforward scans, but they often struggle when a page includes things like:

- multi-column layouts
- tables
- footnotes
- charts, diagrams, or figures
- mixed formatting such as headings, bold text, code, or math

VLMs are often better for this kind of document OCR because they do not only identify characters one by one. They also look at the whole page and reason about layout and meaning.

That tends to make them better at:

- preserving reading order
- recognizing headings and document structure
- reconstructing tables in Markdown
- describing non-text visuals such as figures and charts
- keeping footnotes connected to the places where they are referenced

That does not mean VLMs are always better in every situation. They are usually slower than traditional OCR and they cost API money to run. But for messy, complex, or highly structured documents, they often produce output that needs much less manual cleanup afterward. That tradeoff is the main reason this repo is built around VLM-based OCR.

## Markdown conventions used by this repo

The OCR prompt in this repo asks the model to produce Markdown, but it also asks for a few extra tags so the output keeps document structure that plain Markdown would otherwise lose.

### Footnote tagging

Inline footnote references are wrapped like this:

```md
The sample was preserved at low temperature <ref num="1"/>.
```

The matching footnote text is wrapped like this:

```md
<note num="1">Stored at 4 C until analysis.</note>
```

This is useful because it keeps a clear machine-readable connection between the footnote marker in the main text and the footnote content itself.

That helps with:

- preserving citations and notes during cleanup
- making downstream parsing simpler and more reliable
- avoiding guesswork when a document has many repeated footnote numbers or dense page layouts

By default, `vlmocr convert` expands those references inline so the cleaned Markdown becomes easier for non-technical readers and text-processing tools to follow, for example:

```md
The sample was preserved at low temperature [Footnote 1: Stored at 4 C until analysis].
```

If you want to keep the original `<ref>` and `<note>` tags instead, use `--no-inject-footnotes`.

### Figure descriptions

When a page contains a figure, chart, diagram, or other non-text visual, the model is asked to describe it inside `<image>` tags:

```md
<image>Bar chart comparing quarterly revenue across four regions; the west region is highest in Q3 and Q4.</image>
```

This is useful because normal OCR only captures visible text. It usually does not preserve what a chart, figure, or diagram actually shows.

The figure-description convention helps by:

- keeping important visual meaning in a text-only output format
- making the converted document more searchable
- improving accessibility for readers who cannot inspect the original image easily
- giving downstream LLM or indexing workflows more context about the page

Figure captions are still preserved as ordinary text, so you keep both the original caption and a structured description of the visual content.

## What the cleaned output is for

After OCR runs, `vlmocr convert` cleans the raw output and writes:

- Markdown files for reading, editing, and search
- page-level JSON for accurate retrieval citations (e.g. you can provide specific page citations) 
- a headings-only Markdown table of contents file for quick navigation

The goal is not only to extract text, but to preserve as much of the document's structure and meaning as possible in formats that are easy to reuse.

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

## License
MIT License.
