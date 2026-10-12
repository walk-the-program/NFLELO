<!-- FINAL brand: Bento Grid, Walker 2026-10-10 ("final build aesthetic design choice"). Source of truth for site design.
     Archived briefs: context/archive/ (git tags design-ascii, design-brutalist-navy, design-brutalist, design-editorial, design-bento = v1 with a pink secondary).
     Condensed from Walker's pasted brief; the inline gradpanes SVG is omitted because the engine call regenerates it. -->

# NFLELO brand brief: Bento Grid (final)

**Use only these colors, fonts and rules. No new colors, tints, gradients or fonts.**

- Name: NFELO (renamed from NFLELO on 2026-10-11). Tagline: Machine Learning applied to Football.
- Aesthetic: Bento Grid (Modular · Clear · Product-proud). A grid of rounded tiles, each making one point.
- Motto: One tile, one idea. Mode: light.

## Walker's overrides (2026-10-10), which take precedence over the brief below
1. **Highlight word color:** the "ELO" headline word (and each page's one highlight word) uses **`#fbffa0`** (the accent), NOT the brief's darker `--accent-ink #a38400`. The darker shade looked like a bug to him. It must stay legible: the accent alone is 1.12:1 on the canvas, so it carries a **solid ink (#1c2126) text outline**. It sits on ONE CSS variable so it can switch to the blue (`#7db7eb`, also with the outline) in one line if the yellow doesn't read well.
2. **Secondary is now pure white** (`#ffffff`). The old pinkish secondary (`#f0c5c6`) is removed everywhere.
3. **Patterns on the site:** Walker's two favorites, **Pebble field** and **Posterised terrain**, used tastefully. The other three (aura, gradient panes, halftone) aren't used.

## Tokens
| Token | Hex | Job |
| --- | --- | --- |
| --primary | #7db7eb | Main action |
| --secondary | #ffffff | Support |
| --accent | #fbffa0 | One spark (the highlight word, per override 1) |
| --neutral-dark | #1c2126 | |
| --neutral-light | #f5f7f9 | |
| --canvas | #eaecee | Board |
| --ink | #1c2126 | Text |
| --muted | #64686c | Secondary text |
| --line | #cbced0 | Hairlines |
| --on-primary / --on-secondary / --on-accent | #141414 | |
| --accent-ink | #a38400 | Listed in the brief; NOT used for the highlight (override 1) |
| --pri-tint | #dde6ee | Selected and hover wash |
| --sec-tint | #edeff1 | Selected row, tonal fill |
| --ab-tile | #fbfbfc | Tile |
| --ab-tile-sec | #fcfcfd | Secondary-tinted tile |
| --ab-tile-pri | #e9f1fa | Primary-tinted tile |
| --ab-sub | #f1f1f2 | Sub-tile |
| --ab-pri-hover | #8dc0ed | |
| --ab-pri-press | #6796c1 | |
| --ab-on-pri-press | #141414 | |

Material variables:
- `--ab-shadow: 0 1px 2px rgba(28,33,38,0.04), 0 14px 30px -18px rgba(28,33,38,0.22)`
- `--ab-shadow-hi: 0 2px 4px rgba(28,33,38,0.05), 0 22px 40px -16px rgba(28,33,38,0.3)`
- `--ab-tile-line: rgba(28,33,38,0.08)`
- `--ab-pri-glow: rgba(125,183,235,0.4)`

## Rules
- **Canvas:** the board and gaps (25–35%). **Ink:** text, the huge numerals, the selected segment (never a filled tile). **Primary:** the ONE solid "01" action tile per view, switches on, the product mock. **Secondary (white):** the product stage, the lead stat tile, tags, progress bars. **Accent:** the highlight word, a "new" dot, the footer dot; 3% or less, never fills, buttons or borders.
- Exactly one primary action per view. Text only in ink, muted or `--on-*` (except the highlight word).
- **Focus:** 3px primary with an outer 1px ink ring.
- **Contrast:**
  - ink on canvas: 13.69:1
  - muted on canvas: 4.74:1
  - on-primary on primary: 8.64:1
  - on-secondary on secondary: 18.42:1
  - on-accent on accent: 17.47:1
  - **primary on canvas: 1.80:1 (graphics only)**
  - **accent on canvas: 1.12:1 (needs the ink outline)**
  - **white secondary on tile white: invisible without an outline.** Any white bar or fill needs an ink or `--line` edge.

## Component map
- **Card:** `--ab-tile` with a `--line` edge.
- **Primary button:** primary with `--on-primary` text.
- **Secondary button:** transparent (hover `--sec-tint`) with an edge.
- **Input:** canvas background with a `--line` edge; on focus, primary plus a 40% ring.
- **Segmented control:** selected is primary with `--on-primary` text.
- **Toggle:** on is primary; off is `--line` with a muted knob.
- **Tag:** secondary (white) with `--on-secondary` text and a `--line` edge.
- **Badge:** accent with `--on-accent` text.
- **List row:** selected is `--sec-tint` with a 2px primary left edge.
- **Data bar:** `--line` track, primary fill (a second series in white with an ink edge).

## Typography
- Figtree headings (700), Nunito Sans body and UI.
- Tiles radius 28, sub-tiles 18, controls pills.

## Materials
- **Board** `#eaecee`.
- **Tile:** `#fbfbfc`, radius 28, `--ab-shadow`, no border; hover lifts 2px with `--ab-shadow-hi`.
- **Primary tile:** `#7db7eb`, text `#141414`, radius 28, padding 22. Label bottom-left (22px, 600) and a 40px arrow circle top-right. Hover `#8dc0ed`, pressed `#6796c1`.
- **Secondary-tinted tile** `#fcfcfd`. **Stat tile** `#edeff1`, with a huge tabular ink numeral and a white progress bar (ink edge). **Sub-tile** `#f1f1f2`, radius 18.

## Patterns (exact, from Walker's engine)
- **Engine:** `~/Desktop/Branding Designer/js/patterns/`. Load order: `engine.js`, `families/field.js`, `stripe.js`, `grid.js`, `texture.js`, `print2.js`, `op2.js`, `pixel2.js`, `organic2.js`, `glitch2.js`, `collage2.js`, `fit.js`.
- **INKS:** `Patterns.inksFor({"canvas":"#eaecee","ink":"#1c2126","primary":"#7db7eb","secondary":"#ffffff","accent":"#fbffa0"}, "brand", {"tints":false})`, with seed `brand`. Both are raster, rendered via `Patterns.render` in a browser canvas.
- **Pebble field, card texture:** `render('pebbles', {"size":1,"gap":0.35,"erode":0.45,"mode":"mixed","lit":0.6,"scale":1,"grain":0.25,"tints":false,"tone":1}, INKS, "brand", …)`, PNG 1x (1280x720), cover. At most one textured card per view, with text on a solid plate.
- **Posterised terrain, hero background:** `render('terrain', {"levels":6,"pix":0.5,"dither":0.6,"style":"hills","scale":1,"grain":0.08,"tints":false,"tone":1}, INKS, "brand", …)`, PNG 2x (2560x1440), cover. One per page, with text on a solid plate.

## Do
- Rounded tiles (28) on the darker board with 14px gaps; one idea per tile; huge stat numerals.
- One solid primary tile beside the title.
- Vary the spans while keeping gaps and radii identical.

## Don't
- No tile borders, heavy shadows or gradients.
- No ink or accent tile fills, and never two primary tiles.
- No mixed radii, and no crowded copy.
