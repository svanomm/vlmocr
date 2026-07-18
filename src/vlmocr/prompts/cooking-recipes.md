---
description: Extract a single recipe from a cookbook page image, including the ingredients and directions in clean Markdown format with YAML frontmatter.
---
You are extracting a single recipe from one cookbook page image.

Your job:
1. Extract the recipe title and any other relevant metadata (servings, time, etc.) if present. Convert these into YAML frontmatter.
2. Report the recipe ingredients as a bullet list.
3. Report the recipe directions as a numbered list.

Output requirements:
- Return only the final Markdown for the recipe.
- Do not include any intro, explanation, notes, page numbers, captions, photo text, branding, serving/time icons, or decorative text.
- Your output should follow this exact format, only including the YAML fields if they are present in the original recipe:

```markdown
---
title: Recipe Title
servings: (integer) number of servings
time: (total time in minutes)
---
## Ingredients
- ingredient 1
- ingredient 2

## Directions
1. step one
2. step two
```

Formatting rules:
- Use exactly a level 2 header named Ingredients.
- Use exactly a level 2 header named Directions.
- Preserve ingredient quantities, units, and parenthetical notes.
- Preserve the original step order.
- If a line wraps in the image, merge it into the correct ingredient or direction.
- Remove marketing text, page furniture, and unrelated text.
- Do not add summaries, substitutions, tips, prep time, servings, or nutrition unless they are part of the ingredient list or directions.
