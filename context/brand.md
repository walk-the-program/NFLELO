<!-- Walker's NFLELO brand brief: Raw Brutalist, NAVY palette, pasted 2026-10-10. Source of truth for site design.
     Archived: red Raw Brutalist (context/brand-brutalist-red.md, site at tag design-brutalist) and Editorial Serif (context/brand-editorial.md, tag design-editorial).
     The long inline SVG/data-URI blocks from the pasted brief are omitted; the exact engine calls below regenerate them. -->

# NFLELO brand brief: Raw Brutalist (navy)

**Use only the colors, fonts and rules below. No new colors, tints, gradients or fonts.** Every hex is final; if a design needs an unlisted color, use the nearest listed token.

## Brand
- Name: NFLELO. Tagline: Machine Learning applied to Football.
- Aesthetic: Raw Brutalist (Exposed · Severe · Honest; Swiss print / Brutalist web / Terminal). Hard edges, thick borders, plain mechanics.
- Motto: Nothing hidden. Nothing soft. Mode: light.

## Color tokens
| Token | Hex | Job |
| --- | --- | --- |
| `--primary` | #20436b | Main action |
| `--secondary` | #eff8a5 | Support |
| `--accent` | #ff4d37 | One spark |
| `--neutral-dark` | #26221c | Brand neutral (dark) |
| `--neutral-light` | #f9f8f5 | Brand neutral (light) |
| `--canvas` | #f9f8f5 | Background |
| `--ink` | #26221c | Text and icons |
| `--muted` | #706d68 | Secondary text, captions, placeholders, disabled labels |
| `--line` | #d9d8d4 | Hairlines, dividers, resting input borders |
| `--on-primary` | #ffffff | Text on a primary fill only |
| `--on-secondary` | #141414 | Text on a secondary fill only |
| `--on-accent` | #141414 | Text on an accent fill only |
| `--accent-ink` | #ff4d37 | The accent as text (3.10:1, large text only) |
| `--pri-tint` | #dfe2e4 | A quiet primary wash: selected and hover states |
| `--sec-tint` | #f7f8e8 | A quiet secondary wash: selected rows, tonal fills |

## Where each color goes
- **Canvas:** page and panel backgrounds and same-material controls. Never text. Coverage 60–70%.
- **Ink:** headlines, body text, icons, and hairlines (as a tint). Never a large background in light mode. Coverage 15–20%.
- **Primary:** the one flat CTA block per view and the "on" state of toggles. Never body text, decoration, or a second CTA. Coverage about 10%.
- **Secondary:** tag strips and table status cells. Never the main CTA, body text, or the page background. Coverage about 10–15%.
- **Accent:** a highlighter band behind one headline word (background only, with `#141414` text on it). Never colored text, buttons, borders, or large fills. Coverage 5% or less.

### Hard rules
1. Exactly one primary-filled action per view.
2. Secondary never fills the main CTA and never sets body text.
3. Accent covers 5% or less of a view; never on buttons, body text or borders.
4. Text only in `--ink`, `--muted`, or the matching `--on-*` token on its fill. The one exception is the accent highlight word.
5. Color budget: canvas 60–70%, ink 15–20%, primary about 10%, secondary about 10–15%, accent 5% or less.
6. No other colors, tints or gradients. Hover and pressed states use only the tokens or the Materials.
7. The focus ring is always `--primary`.
8. **The highlight word is a highlighter band:** `#ff4d37` behind the word, with text in `#141414`. Never accent-colored text.

## Component color map
| Element | Background | Text | Border / edge |
| --- | --- | --- | --- |
| Page | --canvas | --ink | none |
| Card / panel | --canvas | --ink | --ink |
| Primary button (one per view) | --primary | --on-primary | --ink |
| Secondary button | transparent (hover --sec-tint) | --ink | --secondary |
| Ghost link | none | --ink underlined | none |
| Input at rest / focused | --canvas | --ink (placeholder --muted) | --ink / --primary + a 3px ring of --primary at 40% |
| Segmented, selected / unselected | --ink / transparent | --canvas / --muted | none |
| Toggle on / off | track --primary, knob --on-primary / track --line, knob --muted | --ink | none |
| Tag / chip | --secondary | --on-secondary | none |
| Badge | --accent | --on-accent | none |
| Headline highlight word (one per view) | --accent band | --on-accent | none |
| Divider | --line | | |
| List row, selected / at rest | --sec-tint / transparent | --ink (meta --muted) | left 2px --primary / bottom 1px --line |
| Data bar | track --line | | fill --primary (second series --secondary) |
| Focus ring | | | --primary |

## Contrast
- ink on canvas: 14.89:1
- muted on canvas: 4.85:1
- on-primary on primary: 10.11:1
- on-secondary on secondary: 16.35:1
- on-accent on accent: 5.59:1
- accent-ink on canvas: 3.10:1 (large text only)
- accent on canvas: 3.10:1
- primary on canvas: 9.52:1

## Typography
- **Headings:** Figtree, 700, letter-spacing 2.5px.
- **Body:** Nunito Sans, 17px, 400, letter-spacing 0.5px.
- **Google Fonts:** https://fonts.googleapis.com/css2?family=Figtree:wght@400;600&family=Nunito+Sans:wght@400;700&display=swap (the site also loads Figtree 700).
- **Interface text, labels and captions:** system monospace. Nunito Sans is for reading copy only.
- **Shape:** bold and geometric, radius 0.
- **Recommended for Raw Brutalist:** Archivo Black / Archivo (default), Inter Tight / Inter, or Anton / IBM Plex Sans; mono JetBrains Mono, IBM Plex Mono or Space Mono. Avoid soft serif, rounded, deco and groovy faces. Figtree and Nunito Sans are acceptable.

## Materials (hard surfaces)
- **Hard surface:** `background: #f9f8f5; border: 2px solid #26221c; border-radius: 0; box-shadow: 6px 6px 0 #26221c;`
- **Pressed:** `transform: translate(6px, 6px); box-shadow: none;`
- **Primary action block:** `background: #20436b; color: #ffffff; border: 2px solid #26221c; border-radius: 0; box-shadow: 6px 6px 0 #26221c; font: 600 13px ui-monospace, "SF Mono", Menlo, monospace; text-transform: uppercase;` Hover: translate(-2px,-2px) with `8px 8px 0 #26221c`. Pressed: translate(6px,6px), no shadow.
- **Palette strips:** a full-width row with a 2px #26221c border. Each color is a flat band sized by its coverage, labeled inside, separated by 2px #26221c rules.
- **Selected segment:** `background: #26221c; color: #f9f8f5;` Joined boxes share 2px borders.
- **Field:** `background: #f9f8f5; border: 2px solid #26221c; border-radius: 0; font: 15px ui-monospace, "SF Mono", Menlo, monospace; caret-color: #26221c;`
- **Highlighter band:** `background: #ff4d37; color: #141414; padding: 0 .06em;` It sits behind one headline word and may run off the board edge.

## Patterns (exact, from Walker's engine)
- **Engine:** `~/Desktop/Branding Designer/js/patterns/`. Load `engine.js`, then `families/field.js`, `stripe.js`, `grid.js`, `texture.js`, then `fit.js` (they define `window.Patterns`). The same seed, params and inks give identical output.
- **INKS:** `Patterns.inksFor({"canvas":"#f9f8f5","ink":"#26221c","primary":"#20436b","secondary":"#eff8a5","accent":"#ff4d37"}, "brand", {"tints":false})`, with seed `brand` and tone 1. Body text never sits on a pattern.
1. **Halftone field: card texture.** At most one textured card per view, with text on a solid canvas plate (4.5:1 or better).
   `Patterns.tileSVG('halftone', {"cell":1.7,"levels":5,"shape":"round","angle":15,"field":"clouds","duo":true,"scale":1,"grain":0.18,"tints":false,"tone":1}, INKS, "brand", 160)`
2. **Glyph grid: card texture.** Same rules.
   `Patterns.tileSVG('glyphgrid', {"cell":3.3,"glyphs":"geometric","bands":6,"field":"warp","duo":true,"scale":1,"grain":0.1,"tints":false,"tone":1}, INKS, "brand", 160)`
3. **Flow filaments: hero background.** Full-bleed, `background-size: cover`, one per page. The headline and copy sit on a solid canvas area or scrim.
   `Patterns.renderSVG('flow', {"field":"flow","strands":4,"gap":1.8,"bend":0.9,"weight":1.25,"scale":1,"grain":0.1,"tints":false,"tone":1}, INKS, "brand", { width: 1600, height: 900 })`

## Do
- Zero radius and 2px ink borders on every control and panel.
- A hard 6px 6px 0 ink shadow on raised things, removed when pressed.
- Every label, tab, button and table cell in monospace, uppercase, numbered like [01], with plain arrows.
- A monochrome page; color appears only where a role says so.

## Don't
- Round any corner, blur any shadow, or add a gradient.
- Set accent as a text color.
- Soften the headline: it stays enormous, uppercase and tight, and may crop at the edge.
- Spend primary on anything but the single main action and active states.
