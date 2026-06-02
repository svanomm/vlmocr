This image is one page of a document. Extract the content of the page verbatim
and convert it to Markdown.
Convert tables into standard Markdown table syntax. For complex layouts you may use HTML syntax if necessary.
Convert section headings to Markdown headers, preserving hierarchy (e.g., #, ##, ###).
Preserve bold/italic formatting with Markdown syntax.
Merge line-wrapped text and undo hyphenation only when caused by line breaks.
Preserve reading order for multi-column layouts.
Convert math to LaTeX: $$ for display math, $ for inline math.
Wrap code snippets in triple backticks with language hints when clear.
For figures, charts, diagrams, or images, write a detailed description
wrapped in <image> tags (e.g., <image>Description...</image>);
preserve figure captions as text.
Wrap inline footnote references in <ref> tags, e.g. <ref num="1"/>.
Wrap footnote text in <note> tags with a `num` attribute, e.g. <note num="1">Footnote text here.</note>
Remove only repeated running headers/footers and standalone page numbers; keep content-bearing metadata.
Output only the Markdown, no commentary or summaries.
