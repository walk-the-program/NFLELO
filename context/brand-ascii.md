<!-- Walker's NFLELO brand brief: ASCII Art, pasted 2026-10-10 (condensed; the inline gradpanes SVG is omitted because the engine call regenerates it). -->

# NFLELO brand brief: ASCII Art

**Use only these colors, fonts and rules. No new colors, tints, gradients or fonts.** Every hex is final; if a design needs an unlisted color, use the nearest listed token.

- Name: NFLELO. Tagline: Machine Learning applied to Football.
- Aesthetic: ASCII Art (Textual · Generative · Nerdy; text-mode art, terminal UI, generative ASCII). Everything is built from monospace characters on a grid, with box-drawing panels and density-map art.
- Motto: Everything is a character. Mode: DARK.

## Tokens
| Token | Hex | Job |
| --- | --- | --- |
| --primary | #7db7eb | Main action |
| --secondary | #f0c5c6 | Support |
| --accent | #f0ec5e | One spark |
| --neutral-dark | #1c2126 | |
| --neutral-light | #f5f7f9 | |
| --canvas | #1c2126 | The screen |
| --ink | #f5f7f9 | Text and icons |
| --muted | #9ea1a5 | Secondary text and captions |
| --line | #3d4146 | Hairlines and frames |
| --on-primary / --on-secondary / --on-accent | #141414 | Text on those fills |
| --accent-ink | #f0ec5e | The highlight word as text |
| --pri-tint | #28333e | Selected and hover wash |
| --sec-tint | #3e3b40 | Selected row, tonal fill |
| --ab-panel | #292e33 | Panel |
| --ab-frame | #595d61 | Frame |
| --aa-sec | #f0c5c6 | |

CSS variables: `--aa-font: "JetBrains Mono", monospace; --aa-adv: 0.6; --aa-line: 1.14; --ab-font-ui / --ab-font-label: "JetBrains Mono", monospace`.

## Where colors go
- **Canvas:** the page and every panel (panels are `--ab-panel`, framed by 1px lines). Never text. Coverage 60–70%.
- **Ink:** the headline, body text, frames, and the brightest characters of the density map. Never large filled fields. Coverage 15–20%.
- **Primary:** the ONE CTA bar, the selected `[ segment ]`, the `[■ ON ]` switch, and the start of the density-map gradient. Never body text, borders, or a second CTA. Coverage about 10%.
- **Secondary:** prompts (`$`, `>`), progress blocks, status brackets, and the middle of the density-map gradient. Never the main CTA, body text, or the page background. Coverage about 10–15%.
- **Accent:** one spark: the last headline word, the footer block, and a few inverse cells in the art. Never buttons, body text, borders, or large fills. Coverage 5% or less.

## Hard rules
1. Exactly one primary-filled action per view.
2. Secondary never fills the main CTA and never sets body text.
3. Accent covers 5% or less; never on buttons, body text or borders.
4. Text only in `--ink`, `--muted`, or the matching `--on-*` token; the one exception is the highlight word.
5. Color budget: canvas 60–70%, ink 15–20%, primary about 10%, secondary 10–15%, accent 5% or less.
6. No other colors, tints or gradients.
7. The focus ring is `--primary`.
8. **Highlight word, once per view:** the LAST headline word is set in `--accent-ink`, followed by a blinking block caret █ (static under reduced motion). No fill and no band.

## Component map
- **Card / panel:** `--ab-panel` with ink text and a `--line` edge.
- **Primary button:** primary with `--on-primary` text.
- **Secondary button:** transparent (hover `--sec-tint`) with a secondary edge.
- **Ghost link:** ink, underlined.
- **Input:** canvas background, ink text, `--line` edge. On focus, a primary edge plus a 3px primary ring at 40%.
- **Segmented control:** selected is primary with `--on-primary` text; unselected is transparent with muted text.
- **Toggle:** on is primary with an `--on-primary` knob; off is `--line` with a muted knob. As text, the Materials version applies.
- **Tag:** secondary with `--on-secondary` text.
- **Badge:** accent with `--on-accent` text.
- **List row:** selected is `--sec-tint` with a 2px primary left edge; at rest it has a 1px `--line` bottom.
- **Data bar:** `--line` track with a primary fill (second series secondary).
- **Focus ring:** primary.

## Contrast
- ink on canvas: 15.10:1
- muted on canvas: 6.25:1
- on-primary on primary: 8.64:1
- on-secondary on secondary: 11.85:1
- on-accent on accent: 14.79:1
- accent-ink on canvas: 13.02:1
- primary on canvas: 7.60:1

## Typography
- **Brand fonts:** Figtree and Nunito Sans. **The aesthetic requires monospace for the headline, every label, the interface, and the art** ("never mix a proportional face into the interface or the art").
- **Recommended:** JetBrains Mono (headings) and IBM Plex Mono (body), from `https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap`.
- **Shape:** minimal and sharp, radius 0 in the materials.

## Materials (characters)
- **Density ramp:** a monospace face with the characters `" .:-=+*#%@"` chosen by brightness (0–9). Each cell is colored along #7db7eb → #f0c5c6 → #f0ec5e, fading toward #1c2126 as it dims.
- **Frame:** `border: 1px solid #3d4146; border-radius: 0; background: #1c2126;` Panels carry their title in the top edge as `┤ TITLE ├`. In the art, the frame is real `┌─┐│└┘` characters.
- **Segmented control:** text `"[ Design ]  Build   Launch"` in #7db7eb. Brackets mark the selected item; the others keep the same width without brackets.
- **Primary action:** `background: #7db7eb; color: #141414; border-radius: 0;` Label `> Save direction` followed by a blinking █ caret (static under reduced motion).
- **Switch:** `[■ ON ]` in #7db7eb, `[□ OFF]` in #9ea1a5. No track, no knob.
- **Progress:** repeating 1ch cells, filled #f0c5c6 and empty #3d4146 (`█████░░░░░`, 6px cells on a 1ch pitch).
- **Selection:** `background: #f0ec5e; color: #1c2126;` Inverse video, on a few cells of the art only.

## Patterns (exact, from Walker's engine)
- **Engine and load order:** as in `context/brand-bento.md`.
- **INKS:** `Patterns.inksFor({"canvas":"#1c2126","ink":"#f5f7f9","primary":"#7db7eb","secondary":"#f0c5c6","accent":"#f0ec5e"}, "brand", {"tints":false})`, with seed `brand`. Raster families are exported as PNG via `Patterns.render` in a browser canvas.
- Text never sits on a pattern.

1. **Pebbles, card texture** (at most 1 per view): `render('pebbles', {"size":1,"gap":0.35,"erode":0.45,"mode":"mixed","lit":0.6,"scale":1,"grain":0.25,"tints":false,"tone":1}, INKS, "brand", ...)`, PNG.
2. **Gradient panes, section band** (at most 2 per page): `renderSVG('gradpanes', {"rows":3,"cols":3,"gap":1.6,"steps":0,"radius":3,"scale":1,"grain":0.1,"tints":false,"tone":1}, INKS, "brand", {width:1200,height:675})`, slice.
3. **Aura wash, hero background:** `render('aura', {"blobs":5,"layout":"scatter","soft":0.8,"mode":"auto","strength":0.85,"scale":1,"grain":0.32,"tints":false,"tone":1}, INKS, "brand", ...)`, PNG 2x.
4. **Posterised terrain, hero background:** `render('terrain', {"levels":6,"pix":0.5,"dither":0.6,"style":"hills","scale":1,"grain":0.08,"tints":false,"tone":1}, INKS, "brand", ...)`, PNG 2x.
5. **Halftone, card texture:** `tileSVG('halftone', {"cell":1.7,"levels":5,"shape":"round","angle":15,"field":"clouds","duo":true,"scale":1,"grain":0.18,"tints":false,"tone":1}, INKS, "brand", 160)`

## Do
- Build every image from the character ramp `" .:-=+*#%@"` on a monospace grid, choosing each character by brightness and coloring the cells along the brand gradient.
- Frame panels with 1px lines, titled in the top edge (`┤ TITLE ├`). Write controls as text: `[ Design ] Build`, `[■ ON ]`, `> Save direction █`.
- Set the headline and every label in monospace, with the highlight word in the accent and prompt characters (`$`, `>`, `//`) in secondary.
- Keep the field mostly empty: sparse dots and a few annotations make the dense glyphs read.

## Don't
- No sprites, stepped pixel borders, rounded corners, shadows or glows on controls.
- Don't mix a proportional face into the interface or the art.
- Never fill panels with primary or accent; only the CTA bar, the selected item and the switch carry primary.
- Don't let the density map run edge to edge without empty cells, or color it with more than the three brand hues.
