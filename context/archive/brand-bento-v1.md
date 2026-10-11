<!-- Walker's NFLELO brand brief: Bento Grid, pasted 2026-10-10 (condensed; the inline gradpanes SVG is omitted because the engine call regenerates it). -->

# NFLELO brand brief: Bento Grid

**Use only these colors, fonts and rules. No new colors, tints, gradients or fonts.** Every hex is final; if a design needs an unlisted color, use the nearest listed token.

- Name: NFLELO. Tagline: Machine Learning applied to Football.
- Aesthetic: Bento Grid (Modular · Clear · Product-proud; Apple feature grids). A grid of rounded tiles, each making a single point.
- Motto: One tile, one idea. Mode: light.

## Tokens
| Token | Hex | Job |
| --- | --- | --- |
| --primary | #7db7eb | Main action |
| --secondary | #f0c5c6 | Support |
| --accent | #f0ec5e | One spark |
| --neutral-dark | #1c2126 | |
| --neutral-light | #f5f7f9 | |
| --canvas | #eaecee | The board behind the tiles |
| --ink | #1c2126 | Text and icons |
| --muted | #64686c | Secondary text and captions |
| --line | #cbced0 | Hairlines, input borders |
| --on-primary / --on-secondary / --on-accent | #141414 | Text on those fills |
| --accent-ink | #aa810e | The highlight word as text (3.03:1, large text only) |
| --pri-tint | #dde6ee | Selected and hover wash |
| --sec-tint | #ebe6e8 | Selected row, tonal fill |
| --ab-tile | #fbfbfc | Tile |
| --ab-tile-sec | #f9f0f1 | Secondary-tinted tile (the product stage) |
| --ab-tile-pri | #e9f1fa | Primary-tinted tile |
| --ab-sub | #f1f1f2 | Sub-tile inside a tile |
| --ab-pri-hover | #8dc0ed | Primary hover |
| --ab-pri-press | #6796c1 | Primary pressed |
| --ab-on-pri-press | #141414 | |

Material variables:
- `--ab-shadow: 0 1px 2px rgba(28,33,38,0.04), 0 14px 30px -18px rgba(28,33,38,0.22)`
- `--ab-shadow-hi: 0 2px 4px rgba(28,33,38,0.05), 0 22px 40px -16px rgba(28,33,38,0.3)`
- `--ab-tile-line: rgba(28,33,38,0.08)`
- `--ab-pri-glow: rgba(125,183,235,0.4)`

## Where colors go
- **Canvas:** the board and the gaps between tiles. Never text, and never a tile with a border. Coverage 25–35%.
- **Ink:** headlines, the huge stat numerals, body text, and the selected segment. Never a filled tile. Coverage 15–20%.
- **Primary:** the ONE solid tile (the "01" action tile: a square with the label bottom-left and an arrow circle top-right), switches "on", and the product mock. Exactly one primary tile per view. Never body text or tile borders. Coverage about 10–12%.
- **Secondary:** the product-stage tint, the lead stat tile, status tags, and every progress bar. Never the main action or body text. Coverage about 10–15%.
- **Accent:** one spark: the headline word (as `--accent-ink` text), a "new" dot, and the footer dot. Never tile fills, buttons or borders. Coverage 3% or less.

## Hard rules
1. Exactly one primary-filled action per view.
2. Secondary never fills the main CTA and never sets body text.
3. Accent covers 3% or less; never on buttons, body text or borders.
4. Text only in `--ink`, `--muted`, or the matching `--on-*` token; the one exception is the highlight word.
5. Color budget: canvas 25–35%, ink 15–20%, primary 10–12%, secondary 10–15%, accent 3% or less.
6. No other colors, tints or gradients (hover and pressed use the tokens and Materials).
7. **Focus ring:** 3px `--primary` with an outer 1px `--ink` ring.
8. **Highlight word:** set as text in `#aa810e`, once per view. Never plain accent as text (1.05:1).

## Component map
- **Card / panel:** `--ab-tile`, ink text, edge `--line`.
- **Primary button:** primary background with `--on-primary` text, no border.
- **Secondary button:** transparent (hover `--sec-tint`), ink text, secondary edge.
- **Ghost link:** ink, underlined.
- **Input:** canvas background, ink text, `--line` edge. On focus, a primary edge plus a 3px primary ring at 40%.
- **Segmented control:** selected is a primary fill with `--on-primary` text; unselected is transparent with muted text.
- **Toggle:** on is a primary track with an `--on-primary` knob; off is a `--line` track with a muted knob.
- **Tag:** secondary with `--on-secondary` text.
- **Badge:** accent with `--on-accent` text.
- **Highlight word:** `--accent-ink` text.
- **Divider:** `--line`.
- **List row:** selected is `--sec-tint` with a 2px primary left edge; at rest it's transparent with a 1px `--line` bottom.
- **Data bar:** `--line` track, primary fill (second series secondary).

## Contrast
- ink on canvas: 13.69:1
- muted on canvas: 4.74:1
- on-primary on primary: 8.64:1
- on-secondary on secondary: 11.85:1
- on-accent on accent: 14.79:1
- accent-ink on canvas: 3.03:1
- accent on canvas: 1.05:1 (FAIL)
- **primary on canvas: 1.80:1 (FAIL; graphics only)**

## Typography
- Headings Figtree 700 (52px, 2.5px tracking); body Nunito Sans 17px; interface text and labels Nunito Sans.
- Shape: radius 12 for controls and 16 large. The materials use **28 for tiles and 18 for sub-tiles; controls are pills**.
- Recommended Inter Tight / Inter (Figtree and Nunito Sans are acceptable).

## Materials (tiles)
- **Board:** `#eaecee`. The gaps between tiles are this color.
- **Tile:** `background: #fbfbfc; border: 0; border-radius: 28px; box-shadow: var(--ab-shadow);` Hover lifts 2px with `--ab-shadow-hi`.
- **Primary tile (main action):** `#7db7eb`, text `#141414`, radius 28, padding 22. Label bottom-left (22px, 600) and a 40px arrow circle top-right at on-primary 18%. One per view. Hover `#8dc0ed`, pressed `#6796c1`.
- **Secondary-tinted tile:** `#f9f0f1`, radius 28 (the product stage).
- **Secondary-tinted stat tile:** `#ebe6e8`, radius 28. A huge tabular ink numeral (tracking -0.05em), with its progress bar in `#f0c5c6`.
- **Sub-tile:** `#f1f1f2`, radius 18.

## Patterns (exact, from Walker's engine)
- **Engine:** `~/Desktop/Branding Designer/js/patterns/`. Load order: `engine.js`, `families/field.js`, `stripe.js`, `grid.js`, `texture.js`, `print2.js`, `op2.js`, `pixel2.js`, `organic2.js`, `glitch2.js`, `collage2.js`, then `fit.js`.
- **INKS:** `Patterns.inksFor({"canvas":"#eaecee","ink":"#1c2126","primary":"#7db7eb","secondary":"#f0c5c6","accent":"#f0ec5e"}, "brand", {"tints":false})`, with seed `brand`.
- **Raster families** (pebbles, aura, terrain) are exported as PNG via `Patterns.render` in a browser canvas (2x = 2560x1440).
- Text never sits on a pattern; it sits on a canvas or tile plate (4.5:1 or better).

1. **Pebbles, card texture** (at most 1 textured card per view):
   `render('pebbles', {"size":1,"gap":0.35,"erode":0.45,"mode":"mixed","lit":0.6,"scale":1,"grain":0.25,"tints":false,"tone":1}, INKS, "brand", {width:1280,height:720})`, PNG 1x, cover.
2. **Gradient panes, section band** (160–320px tall, at most 2 per page, never two of the same next to each other):
   `renderSVG('gradpanes', {"rows":3,"cols":3,"gap":1.6,"steps":0,"radius":3,"scale":1,"grain":0.1,"tints":false,"tone":1}, INKS, "brand", {width:1200,height:675})`. Set the root to `preserveAspectRatio="xMidYMid slice"`.
3. **Aura wash, hero background** (one pattern hero per page):
   `render('aura', {"blobs":5,"layout":"scatter","soft":0.8,"mode":"auto","strength":0.85,"scale":1,"grain":0.32,"tints":false,"tone":1}, INKS, "brand", ...)`, PNG 2x, cover.
4. **Posterised terrain, hero background:**
   `render('terrain', {"levels":6,"pix":0.5,"dither":0.6,"style":"hills","scale":1,"grain":0.08,"tints":false,"tone":1}, INKS, "brand", ...)`, PNG 2x, cover.
5. **Halftone, card texture:**
   `tileSVG('halftone', {"cell":1.7,"levels":5,"shape":"round","angle":15,"field":"clouds","duo":true,"scale":1,"grain":0.18,"tints":false,"tone":1}, INKS, "brand", 160)`

## Do
- Build every block as a rounded tile (radius 28) on the slightly darker canvas, with 14px gaps and never a border or rule.
- Each tile says one thing. Stats get a huge numeral in the heading face with a small label.
- Exactly one solid primary tile (the action tile beside the title). Tint the product stage and the lead stat with secondary; the rest are plain tile white.
- Vary spans (2x2 hero, 1x1 stats, wide bars) while keeping gaps and radii identical.

## Don't
- No tile borders, no heavy shadows, no gradients; one very soft shadow is the whole depth.
- Never fill a tile with ink or accent, and never put two solid primary tiles in one view.
- Don't mix radii: sub-tiles 18, tiles 28, controls pills.
- Don't crowd copy: one idea per tile, with generous padding.
