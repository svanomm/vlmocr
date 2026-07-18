# vlmocr

`vlmocr` turns PDFs and images into Markdown using vision-language models (VLMs). We provide a flexible framework that allows you to pick from built-in OCR templates and easily create your own.

## Why vlmocr uses VLMs for OCR

Traditional OCR struggles with mixed layout pages (tables, footnotes, figures, multicolumn text, math). VLMs are much better at preserving structure and meaning across the whole page, or can transform page content into structured formats.

The default OCR model is Gemini 3.1 Flash Lite via OpenRouter for strong quality-cost tradeoffs, but you can use any VLM served by OpenRouter.

## Pipeline overview

The standard flow is:

1. Put supported documents in the docs folder (.pdf, .jpg, .jpeg, .png, .webp, .bmp).
2. Use the CLI to write raw per-page JSON under converted/json/raw.
3. Run conversion to clean output and write:
	 - merged Markdown under converted/md
	 - cleaned page-level JSON under converted/json
	 - a headings-only TOC under converted/md/table of contents

Discovery is recursive by default under docs. TIFF/GIF inputs are currently skipped with warnings.

## Prompt template framework

### Built-in templates

Built-in templates live in src/vlmocr/prompts. Each template is a Markdown file with optional front matter metadata:

```md
---
description: OCR profile focused on table fidelity.
---

Your OCR instructions go here.
```

### How template selection works

- Interactive OCR: if you run OCR in an interactive terminal without --prompt-template, vlmocr prompts you to choose a template.
- In that prompt picker, you can also create a new template immediately and use it for the current run.
- Non-interactive OCR without --prompt-template falls back to the default template.

Direct template selection from command line:

```bash
uv run vlmocr ocr --prompt-template academic-with-footnotes
```

### Creating your own templates

1. Create a template from the interactive OCR template picker (includes description and prompt body).
2. Add a new .md file to src/vlmocr/prompts manually.
3. We also provide an LLM prompt to generate new templates through an interview process. You can run it as a slash command in VS Code "/create-ocr-template".

Template names are normalized to safe slugs (for example, Economic Tables -> economic-tables).

## Markdown conventions in OCR output

The academic template uses explicit tags to preserve information plain Markdown can lose.

Footnote references:

```md
Text in the paragraph.<ref num="1"/>
```

Footnote body:

```md
<note num="1">Footnote text here.</note>
```

By default, convert injects footnotes inline and removes the note blocks:

```md
Text in the paragraph. [Footnote 1: Footnote text here.]
```

Use --no-inject-footnotes to keep original <ref> and <note> tags.

For non-text visuals, templates can request descriptions inside image tags:

```md
<image>Concise description of chart or figure content.</image>
```

## Requirements

- Python 3.12+
- `uv`
- OpenRouter API key for OCR commands

Install dependencies:

```bash
uv sync
```

Set your API key in a project-root .env file:

```bash
OPENROUTER_API_KEY=your_key_here
```

## Commands

Interactive launcher:

```bash
uv run vlmocr
```

This menu includes:

- init workflow
- OCR workflow with template selection and template creation
- conversion workflow
- project structure validation
- quickstart help
- show/edit default OCR prompt
- benchmark workflow (requires explicit model slug)

Command mode:

```bash
# OCR
uv run vlmocr ocr --docs-dir docs --out-dir converted --prompt-template default
uv run vlmocr ocr --docs-dir docs --out-dir converted --no-recursive

# Convert raw OCR JSON (defaults to converted/json/raw)
uv run vlmocr convert --out-dir converted

# Optional conversion controls
uv run vlmocr convert --remove-frequent-lines
uv run vlmocr convert --no-inject-footnotes

# Estimate OCR cost from mixed documents in docs
uv run vlmocr estimate-cost --docs-dir docs
uv run vlmocr estimate-cost --docs-dir docs --no-recursive

# Build local academic benchmark data (10 difficult one-page cases)
uv run vlmocr benchmark-init-academic --docs-dir docs --out-dir converted

# Run benchmark for one model
uv run vlmocr benchmark --out-dir converted --model google/gemini-3.1-flash-lite-preview

# Compare multiple models in one run
uv run vlmocr benchmark --out-dir converted --model google/gemini-3.1-flash-lite-preview --model openai/gpt-4.1-mini
```

## Deterministic benchmark workflow

`benchmark-init-academic` creates a local benchmark bundle under `converted/benchmark/academic-textbook-v1`:

- `manifest.json` with 10 fixed difficult cases (math/table/footnote/image-heavy)
- `gold/*.json` one-page expected outputs for deterministic scoring
- `docs/benchmark/*.pdf` one-page benchmark review PDFs (one file per selected case)
- automatic verification that each PDF in `docs/benchmark` has a matching gold JSON

`benchmark` then:

- renders exactly one page per benchmark case
- calls the selected model(s) on those pages
- scores candidate raw markdown against gold using deterministic text/math/structure metrics
- writes run reports to `converted/benchmark/reports`
- stores every run, model summary, and case result in `converted/benchmark/history.db`
- prints the 10 most recent benchmark model summaries after each benchmark run
- verifies `docs/benchmark` integrity before running if that folder exists
- records OpenRouter-reported usage and cost per case (`usage.cost`) when available
- reports per-model total benchmark cost in USD and normalized dollars per 1000 pages

Selected academic benchmark pages (v1):

- Abadie, Athey, Imbens, Wooldridge (2017) -  When Should you Adjust Standard Errors for Clustering, page 11
- Cameron and Miller (2015) - Practitioner's Guide to Cluster-Robust Inference, page 31
- Balli and Sorensen (2012) - Interaction Effects in Econometrics, page 17
- Balli and Sorensen (2012) - Interaction Effects in Econometrics, page 10
- Benoit (2011) - Linear Regression Models with Logarithmic Transformations, page 4
- Norton and Dowd (2018) - Log Odds and the Interpretation of Logit Models, page 12
- Youssef (2022) - Detecting of Multicollinearity, Autocorrelation, and Heteroscedasticity in Regression Analysis, page 6
- Wainer (1984) - How to Display Data Badly, page 10
- Anscombe's quartet - Wikipedia, page 2
- predict, page 1

## Supported OCR inputs

- Supported now: PDF, JPG/JPEG, PNG, WEBP, BMP
- Skipped in this phase: TIFF/TIF, GIF
- For same-stem mixed files, raw JSON uses extension suffixes to avoid collisions
	- example: report.pdf -> report__pdf.json and report.png -> report__png.json

## Workspace layout

vlmocr uses this default structure:

```text
docs/
converted/
	json/
		raw/                 # raw per-page OCR output from vlmocr ocr
	md/
	md/table of contents/
```

## OCR settings and rerun behavior

Raw OCR JSON stores a settings_hash that includes model, DPI, image format, prompt text, temperature, and max tokens.

This means:

- unchanged files with matching settings are skipped
- changing template or OCR settings causes affected documents to be reprocessed

Environment variable overrides:

- OPENROUTER_API_KEY
- VLMOCR_MODEL
- VLMOCR_DPI
- VLMOCR_IMAGE_FORMAT
- VLMOCR_MAX_TOKENS
- VLMOCR_MAX_WORKERS
- VLMOCR_MAX_RETRIES

## License

MIT License.
