<!-- Walker's NFLELO brand brief: Editorial Serif, pasted 2026-10-10. Source of truth for site design.
     The previous Raw Brutalist brief is archived in context/brand-brutalist.md (site saved at git tag design-brutalist).
     The two very long inline SVG/data-URI blocks from the pasted brief are omitted here; the exact engine calls below regenerate them. -->

# NFLELO brand brief for AI builders: Editorial Serif

**Use only the colors, fonts and rules below. Do not introduce new colors, tints, gradients or fonts unless they are listed here.** Every color value is final. If a design needs a color that isn't listed, use the nearest listed token.

## Brand
- Name: NFLELO
- Tagline: Machine Learning applied to Football

## Aesthetic: Editorial Serif
- Descriptor: Literate · Confident · Spacious
- Influences: Magazine editorial / Swiss columns / Book typography
- Summary: magazine pages set in a high-contrast serif, with hairline rules and room to breathe.
- Motto: Set it like a page.
- Mode: light (canvas is the light neutral, ink is the dark neutral)

## Color tokens
| Token | Hex | Role / job |
| --- | --- | --- |
| `--primary` | #a11413 | Main action |
| `--secondary` | #d14f34 | Support |
| `--accent` | #4a3122 | One spark |
| `--neutral-dark` | #261c1c | Brand neutral (dark) |
| `--neutral-light` | #f9f5f5 | Brand neutral (light) |
| `--canvas` | #f7f2ee | Background (the paper) |
| `--ink` | #261c1c | Text and icons |
| `--muted` | #6f6766 | Secondary text, captions, placeholders, disabled labels |
| `--line` | #d8d2cf | Hairlines, dividers, resting input borders |
| `--on-primary` | #ffffff | Text on a primary fill only |
| `--on-secondary` | #ffffff | Text on a secondary fill only |
| `--on-accent` | #ffffff | Text on an accent fill only |
| `--accent-ink` | #4a3122 | The accent as text: the italic highlight word |
| `--pri-tint` | #edd7d4 | A quiet primary wash: selected and hover states |
| `--sec-tint` | #f1d8d0 | A quiet secondary wash: selected rows, tonal fills |
| `--ab-hair-strong` | #261c1c | Materials |
| `--ab-eds-plate-dark` | #510a0a | Materials |
| `--ab-eds-plate` | #511919 | Materials (the duotone plate) |
| `--ab-eds-paper` | #f7f2ee | Materials |
| `--ab-eds-ink-hover` | #351b1b | Materials |
| `--ab-eds-primary-press` | #7e100f | Materials (primary pressed) |
| `--ab-eds-on-press` | #ffffff | Materials |

## Where each color goes
- **Canvas (#f7f2ee):** the paper. Page and card background; panels are the same paper separated by hairlines. Never for text or a second tinted background. Coverage 70–75%.
- **Ink (#261c1c):** headlines, body copy, rules (full ink for the masthead, a tint for hairlines) and the hover fill of the main action. Never for large background fields. Coverage 18–22%.
- **Primary (#a11413):** the one main action per page, the duotone cover plate and the selected tab underline. Never for body text, rules, or decorative fills outside the plate. Coverage about 8–10%.
- **Secondary (#d14f34):** the pull-quote tint, list "status" tags and the second bar in a chart. Always small and quiet. Never for the main action, body text or the page background. Coverage about 5–8%.
- **Accent (#4a3122):** the one italic headline word and a single "New" flag, set as text. Never for buttons, borders, rules or fills. Coverage 3% or less.

### Hard rules
1. Exactly one primary-filled action per view.
2. Secondary never fills the main call to action and never sets body text.
3. Accent covers 3% or less of any view and is never used on buttons, body text or borders.
4. Text is set only in `--ink`, `--muted`, or the matching `--on-*` token on its fill. The one exception is the single accent highlight word.
5. Color budget: canvas 70–75%, ink 18–22%, primary about 8–10%, secondary about 5–8%, accent 3% or less.
6. No other colors, tints or gradients. Hover and pressed states use only the tokens above or the Materials.
7. The focus ring is always `--primary`.
8. **Headline highlight word, once per view:** the last word is set in italic (same face, weight 400) in `--accent-ink`, with no fill and no underline. The rest of the headline stays upright ink.

## Component color map
| Element | Background | Text | Border / edge |
| --- | --- | --- | --- |
| Page | --canvas | --ink | none |
| Card / panel | --canvas | --ink | --line |
| Primary button (one per view) | --primary | --on-primary | none |
| Secondary button | transparent (hover: --sec-tint) | --ink | --secondary |
| Ghost link | none | --ink (underlined) | none |
| Input, at rest | --canvas | --ink (placeholder --muted) | --line |
| Input, focused | --canvas | --ink | --primary + a 3px ring of --primary at 40% opacity |
| Segmented, selected | --primary | --on-primary | none |
| Segmented, unselected | transparent | --muted | none |
| Toggle on / off | track --primary, knob --on-primary / track --line, knob --muted | --ink | none |
| Tag / chip | --secondary | --on-secondary | none |
| Badge ("New") | --accent | --on-accent | none |
| Divider | --line | | |
| List row, selected | --sec-tint | --ink | left 2px --primary |
| List row, at rest | transparent | --ink (meta --muted) | bottom 1px --line |
| Data bar | track --line | | fill --primary (second series --secondary) |
| Focus ring | | | --primary |

## Contrast
- ink on canvas: 14.93:1
- muted on canvas: 4.96:1
- on-primary on primary: 8.00:1
- on-secondary on secondary: 4.30:1 (large text only, 24px or 19px bold and up)
- on-accent on accent: 11.98:1
- accent-ink on canvas: 10.77:1
- primary on canvas: 7.20:1

## Typography
- Brand fonts listed: Figtree (heading) and Nunito Sans (body); Google Fonts link `https://fonts.googleapis.com/css2?family=Figtree:wght@400;600&family=Nunito+Sans:wght@400;700&display=swap`.
- **Interface text, captions and labels:** Nunito Sans (small caps, .12em tracking).
- **Shape:** minimal and sharp. Radius 4px, large radius 6px (the Do list says keep corners square, 0–2px).
- **Recommended for Editorial Serif:**
  - Headings: didone or soft serif (Instrument Serif, Bodoni Moda, Playfair Display, Gloock).
  - Body: text serif (Newsreader, Source Serif 4, Libre Caslon Text, EB Garamond).
  - Default pairing: **Instrument Serif / Newsreader**, from `https://fonts.googleapis.com/css2?family=Instrument+Serif:ital,wght@0,400;1,400&family=Newsreader:ital,wght@0,400;0,600;1,400&display=swap`.
  - Avoid rounded, techno, pixel and groovy faces.
- **Decision (Claude, 2026-10-10):** the aesthetic's Do and Don't require a serif headline (weight 400) and serif reading copy ("Run body copy in sans" is a Don't). So the site uses the listed recommended pairing, Instrument Serif for headlines and Newsreader for reading copy, with Nunito Sans for interface text and labels. Figtree is unused in this design.

## Materials (page)
- **Paper:** `background: #f7f2ee; color: #261c1c; border-radius: 0; box-shadow: none;` No cards; panels are the same paper.
- **Masthead rule:** `border-top: 3px solid #261c1c; border-bottom: 1px solid #261c1c; height: 7px;` Thick over thin, under the masthead only.
- **Hairline:** `border-top: 1px solid rgba(38,28,28,0.3);` For every divider and every panel edge, never a shadow.
- **Duotone plate:** `background: #511919; border: 1px solid #261c1c; border-radius: 0;` It carries a cropped brand initial, a coverline, and a caption below ("Fig. 1").
- **Primary action:** `background: #a11413; color: #ffffff; border-radius: 0; text-transform: uppercase; letter-spacing: .14em; font-size: 12px;` Hover: background #261c1c, color #f7f2ee. Pressed: #7e100f with text #ffffff.
- **Contents row:** `border-top: 1px solid rgba(38,28,28,0.16);` Name, a dotted leader (1px dotted rgba(38,28,28,0.3)), the status in small caps, and the page number right-aligned.

## Patterns (exact, from Walker's own engine)
- **Engine:** `~/Desktop/Branding Designer/js/patterns/`. Load `engine.js`, then `families/field.js`, `stripe.js`, `grid.js`, `texture.js`, then `fit.js`. These are plain scripts that define `window.Patterns`. The same seed, params and inks give identical output.
- **Inks** for every pattern (Brand colorway): `Patterns.inksFor({"canvas":"#f7f2ee","ink":"#261c1c","primary":"#a11413","secondary":"#d14f34","accent":"#4a3122"}, "brand", {"tints":false})`, seed `brand`, tone 1.
- Body text never sits on a pattern.

1. **Halftone field: card texture.** At most one textured card per view; text sits on a solid canvas plate inside it.
   `Patterns.tileSVG('halftone', {"cell":1.7,"levels":5,"shape":"round","angle":15,"field":"clouds","duo":true,"scale":1,"grain":0.18,"tints":false,"tone":1}, INKS, "brand", 160)` gives a seamless 160px tile.
2. **Glyph grid: card texture.** Same rules.
   `Patterns.tileSVG('glyphgrid', {"cell":3.3,"glyphs":"geometric","bands":6,"field":"warp","duo":true,"scale":1,"grain":0.1,"tints":false,"tone":1}, INKS, "brand", 160)` gives a seamless 160px tile.
3. **Flow filaments: hero background.** Full-bleed behind the hero, `background-size: cover`, one per page. The headline and copy sit on a solid canvas area or a canvas scrim (4.5:1 or better).
   `Patterns.renderSVG('flow', {"field":"flow","strands":4,"gap":1.8,"bend":0.9,"weight":1.25,"scale":1,"grain":0.1,"tints":false,"tone":1}, INKS, "brand", { width: 1600, height: 900 })`. It doesn't tile (about 346 KB as SVG); a PNG export (1280x720 or 2560x1440) is lighter.

## Do
- Set the headline in the serif at weight 400, with the highlight word in italic accent ink; contrast carries the weight.
- Separate everything with hairline rules. Use the thick-over-thin double rule once, under the masthead.
- Keep corners square (0–2px) and flat: no shadows, and no gradients outside the duotone plate.
- Set captions, tabs and labels in small-caps sans with .12em tracking. Keep reading copy in the text serif, with a drop cap on the first paragraph.

## Don't
- Bold the headline, round the controls or add a shadow to a card. This is print, not an app.
- Use accent as a fill, button or rule. It is one italic word.
- Fill the page with tinted cards. The page is one sheet of paper with hairlines.
- Run body copy in sans or letter-space the serif.
