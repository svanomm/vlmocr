---
description: Extract a single recipe from a cookbook page image, including only the ingredients and directions in clean Markdown format.
---
You are extracting a single recipe from one cookbook page image.

Your job:
1. Read the recipe title from the image.
2. Extract only the recipe ingredients and directions.
3. Convert them into clean Markdown.

Output requirements:
- Return only the final Markdown body for the recipe.
- Do not include any intro, explanation, notes, page numbers, captions, photo text, branding, serving/time icons, or decorative text.
- Do not include the recipe title inside the Markdown body.
- The recipe title should be inferred separately and used as the filename. If you are able to return metadata, use the exact recipe title as filename with a .md extension.
- The Markdown body must contain exactly these sections, in this order:

## Ingredients
- ingredient 1
- ingredient 2

## Directions
1. step one
2. step two

Formatting rules:
- Use exactly a level 2 header named Ingredients.
- Use exactly a level 2 header named Directions.
- Ingredients must be a bullet list.
- Directions must be a numbered list.
- Preserve ingredient quantities, units, and parenthetical notes.
- Preserve the original step order.
- Fix obvious OCR mistakes, but do not invent missing text.
- If a line wraps in the image, merge it into the correct ingredient or direction.
- Remove marketing text, page furniture, and unrelated text.
- Do not add summaries, substitutions, tips, prep time, servings, or nutrition unless they are part of the ingredient list or directions.
- If some text is unreadable, make the best faithful extraction you can without hallucinating.

Return format:
- If your interface supports separate metadata, provide:
  - title: exact recipe title
  - markdown: the Markdown body only
- Otherwise return:
  - First line: TITLE: exact recipe title
  - Then the Markdown body only, using the exact format above.