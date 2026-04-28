# vlmocr

`vlmocr` is the OCR and conversion slice extracted from `skill-search`.

It owns:

- PDF rendering with PyMuPDF
- OpenRouter vision-model OCR
- Raw OCR contract validation
- OCR-to-markdown conversion and cleaning
- OCR-only cost estimation

## Commands

```bash
uv run vlmocr ocr --docs-dir docs --out-dir .search/converted
uv run vlmocr convert --input-dir .search/converted/json/raw --out-dir .search/converted
uv run vlmocr estimate-cost --docs-dir docs
```

## Output contract

The raw OCR output is written to `json/raw/<name>.json` under the selected
`--out-dir` and must preserve this schema:

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

The conversion command writes:

- `json/<name>.json`
- `md/<name>.md`
- `md/table of contents/<name>_toc.md`

## Environment

OCR requires `OPENROUTER_API_KEY`.

Optional overrides:

- `VLMOCR_MODEL`
- `VLMOCR_DPI`
- `VLMOCR_IMAGE_FORMAT`
- `VLMOCR_MAX_TOKENS`
- `VLMOCR_MAX_WORKERS`
- `VLMOCR_MAX_RETRIES`

## Move-out steps

When you are ready to spin this into its own repository:

1. Move this `vlmocr/` directory to its final location.
2. Initialize a git repository in that new location.
3. Run `uv sync`.
4. Run `uv run pytest`.
5. Tag a release and consume it from `skill-search` through pinned `uvx --from git+...` commands.
