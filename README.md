# vlmocr

`vlmocr` turns PDFs into clean, reusable text files.

The pipeline is simple: it takes a PDF, renders each page as an image, uses a vision-language model (VLM) to extract structured data, and then saves the result as:

- Markdown for humans to read and edit
- JSON for scripts, pipelines, or downstream tools

This repo is useful when you have scanned papers, reports, manuals, or image-heavy PDFs that are hard to search, copy from, or repurpose. Any math in the paper gets converted to LaTeX, and figures/charts are given plain-text descriptions.

Under the hood, `vlmocr` splits your PDFs into page images and sends them to OpenRouter-served VLMs, then converts the results into cleaned Markdown. Gemini 3.1 Flash Lite is the default model based on its very high performance on [socOCRBench](https://noahdasanaike.github.io/posts/sococrbench.html) and cost effectiveness. With current API pricing, I am seeing an average of around **$1.50 per 1000 pages** of OCR.

## Why this project uses VLMs for OCR

Traditional OCR systems struggle when a page includes things like:

- multi-column layouts
- tables
- footnotes
- charts, diagrams, or figures
- mixed formatting such as headings, bold text, code, or math

VLMs are better for this kind of document OCR because they do not only identify characters one by one, but instead look at the whole page and reason about layout and meaning. This makes them better at:

- preserving reading order
- recognizing headings and document structure
- reconstructing tables in Markdown
- describing non-text visuals such as figures and charts
- keeping footnotes connected to the places where they are referenced

An additional benefit of using a general-purpose multimodal VLM is that you can prompt it with custom instructions: write text descriptions of charts, convert math to LaTeX, and much more. You can easily change the prompt underlying `vlmocr` to suit your preferences, and can also experiment with model parameters such as temperature (which defaults to 0.0).

## Markdown conventions used by this repo

The OCR prompt in this repo asks the model to produce Markdown, but it also asks for a few extra tags so the output keeps document structure that plain Markdown would otherwise lose.

### Footnote tagging

Inline footnote references are wrapped like this:

```md
The sample was preserved at low temperature.<ref num="1"/>
```

The matching footnote text is wrapped like this:

```md
<note num="1">Stored at 4 C until analysis.</note>
```

This is useful because it keeps a clear machine-readable connection between the footnote marker in the main text and the footnote content itself. By default, `vlmocr convert` expands those references inline so the cleaned Markdown becomes easier for non-technical readers and text-processing tools to follow, for example:

```md
The sample was preserved at low temperature. [Footnote 1: Stored at 4 C until analysis.]
```

If you want to keep the original `<ref>` and `<note>` tags instead, use `--no-inject-footnotes`. You could also customize `vlmocr` to move all footnote text to the bottom of the document, for example.

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
- [`uv`](https://docs.astral.sh/uv/)
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

## Commands

```bash
uv run vlmocr
```

This opens a terminal interface that lets you initialize the workspace, inspect the expected directory layout, and run the available commands with prompts.
It also explains where to create an OpenRouter API key and how to store it before you run OCR.

## Defaults and options

Environment overrides:

- `OPENROUTER_API_KEY`
- `VLMOCR_MODEL`
- `VLMOCR_DPI`
- `VLMOCR_IMAGE_FORMAT`
- `VLMOCR_MAX_TOKENS`
- `VLMOCR_MAX_WORKERS`
- `VLMOCR_MAX_RETRIES`

## License
MIT License.
