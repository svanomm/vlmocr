---
description: High-fidelity transcription that preserves headers, footers, and page markers.
---

This image is one page of a document. Transcribe everything visible into Markdown with minimal normalization.
Keep line breaks as they appear unless a split clearly breaks a single word.
Preserve running headers, footers, and page numbers exactly as shown.
Convert section headings to Markdown headers only when visually explicit.
Convert tables into Markdown tables; use HTML tables only if Markdown would lose important structure.
Convert math to LaTeX: $$ for display math, $ for inline math.
For figures, charts, diagrams, or images, include a concise description in <image> tags while preserving nearby captions.
Keep footnote markers and footnote text exactly where they appear.
Output only Markdown text from the page with no commentary.
