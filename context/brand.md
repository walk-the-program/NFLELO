<!-- Walker's NFLELO brand brief, pasted 2026-10-10. Source of truth for all site design. It supersedes Soft Form. -->

# NFLELO brand brief for AI builders

**Use only the colors, fonts and rules below. Do not introduce new colors, tints, gradients or fonts unless they are listed here.**

Every color value in this brief is final. If a design needs a color that isn't listed, use the nearest listed token instead of inventing one.

## Brand

- Name: NFLELO
- Tagline: Machine Learning applied to Football

## Aesthetic: Raw Brutalist

- Descriptor: Exposed · Severe · Honest
- Influences: Swiss print / Brutalist web / Terminal
- Summary: hard edges, thick borders and plain mechanics, with nothing hidden or softened.
- Motto: Nothing hidden. Nothing soft.
- Mode: light (canvas is the light neutral, ink is the dark neutral)

## Color tokens

Define these as CSS custom properties and reference them by name. The hexes are exact; do not adjust them.

| Token | Hex | Role | Job |
| --- | --- | --- | --- |
| `--primary` | #a11413 | Primary | Main action |
| `--secondary` | #d14f34 | Secondary | Support |
| `--accent` | #4a3122 | Accent | One spark |
| `--neutral-dark` | #261c1c | Derived | Brand neutral (dark). It becomes the canvas in dark mode and the ink in light mode |
| `--neutral-light` | #f9f5f5 | Derived | Brand neutral (light). It becomes the canvas in light mode and the ink in dark mode |
| `--canvas` | #f9f5f5 | Canvas | Background |
| `--ink` | #261c1c | Ink | Text and icons |
| `--muted` | #706868 | Derived | Secondary text, captions, placeholders, disabled labels |
| `--line` | #d9d4d4 | Derived | Hairlines, dividers, resting input borders |
| `--on-primary` | #ffffff | Derived | Text and icons on a primary fill only |
| `--on-secondary` | #ffffff | Derived | Text and icons on a secondary fill only |
| `--on-accent` | #ffffff | Derived | Text and icons on an accent fill only |
| `--accent-ink` | #4a3122 | Derived | The accent as text: the same hue family (yellows shift toward amber on light canvases), shaded (light canvas) or tinted (dark canvas) just enough to reach 3:1 on the canvas. Set the highlight word in it, never in plain accent |
| `--pri-tint` | #eedada | Derived | A quiet primary wash: selected-state backgrounds and hover on primary-linked items |
| `--sec-tint` | #f3dad6 | Derived | A quiet secondary wash: selected rows, tonal fills, hover on secondary items |

Canvas is derived from Neutral light. Use `--canvas`, not the raw neutral, for page and panel backgrounds.

## Where each color goes

- **Canvas (#f9f5f5), background.** Use for page and panel backgrounds and same-material controls. Never for text. Coverage 60–70%.
- **Ink (#261c1c), text and icons.** Use for headlines, body text, icons and hairlines (as a tint). Never for large background fields in light mode. Coverage 15–20%.
- **Primary (#a11413), main action.** Use for the one flat CTA block per view and the "on" state of toggles. Never for body text, decoration or a second CTA. Coverage about 10%.
- **Secondary (#d14f34), support.** Use for tag strips and the status cells of tables. Never for the main CTA, body text or the page background. Coverage about 10–15%.
- **Accent (#4a3122), one spark.** Use for a highlighter band behind one headline word (background only, with text on it). Never for colored text, buttons, borders or large fills. Coverage 5% or less.

### Hard rules

1. Exactly one primary-filled action (the main call to action) per view.
2. Secondary never fills the main call to action and never sets body text.
3. Accent covers 5% or less of any view and is never used on buttons, body text or borders.
4. Text is set only in `--ink`, `--muted`, or the matching `--on-*` token on its fill. The one exception is the single accent highlight word, and only where the highlight treatment below says so.
5. Color budget per view: canvas 60–70%, ink 15–20%, primary about 10%, secondary about 10–15%, accent 5% or less. Stay inside it.
6. No other colors, tints or gradients. Hover and pressed states use only the tokens above (for example `--pri-tint`, `--sec-tint`) or the shadows and overlays listed under Materials.
7. The focus ring is always `--primary`.
8. In this aesthetic the highlight word is a highlighter band: background `#4a3122` (`--accent`) behind the word, text in `#ffffff` (`--on-accent`). Never set accent-colored text.

## Component color map

Which token goes where. Shadows, blur and borders beyond these tokens come from Materials.

| Element | Background | Text | Border / edge |
| --- | --- | --- | --- |
| Page | --canvas | --ink | none |
| Card / panel | --canvas | --ink | --ink |
| Primary button (one per view) | --primary | --on-primary | --ink |
| Secondary button | transparent (hover: --sec-tint) | --ink | --secondary |
| Ghost / tertiary link | none | --ink (underlined) | none |
| Input field, at rest | --canvas | --ink (placeholder: --muted) | --ink |
| Input field, focused | --canvas | --ink | --primary (plus a 3px ring of --primary at 40% opacity) |
| Segmented control, selected | --ink | --canvas | none |
| Segmented control, unselected | transparent | --muted | none |
| Toggle, on | track --primary, knob --on-primary | --ink (label) | none |
| Toggle, off | track --line, knob --muted | --ink (label) | none |
| Tag / chip | --secondary | --on-secondary | none |
| Badge ("New", notification) | --accent | --on-accent | none |
| Headline highlight word (one per view) | --accent (band behind the word) | --on-accent | none |
| Divider / hairline | --line | n/a | n/a |
| List row, selected | --sec-tint | --ink | left edge 2px --primary |
| List row, at rest | transparent | --ink (meta: --muted) | bottom 1px --line |
| Progress / data bar | track --line | n/a | fill --primary (second series: --secondary) |
| Focus ring (any control) | n/a | n/a | --primary |

## Contrast pairs

| Pair | Foreground | Background | Ratio | Grade | Use |
| --- | --- | --- | --- | --- | --- |
| --ink on --canvas | #261c1c | #f9f5f5 | 15.34:1 | AAA | Body and headlines |
| --muted on --canvas | #706868 | #f9f5f5 | 5.02:1 | AA | Captions and secondary text |
| --on-primary on --primary | #ffffff | #a11413 | 8.00:1 | AAA | Primary button label |
| --on-secondary on --secondary | #ffffff | #d14f34 | 4.30:1 | AA large text only | Chip and tag label |
| --on-accent on --accent | #ffffff | #4a3122 | 11.98:1 | AAA | Badge label, highlight band |
| --accent-ink on --canvas | #4a3122 | #f9f5f5 | 11.07:1 | AAA | Highlight word and accent text |
| --accent on --canvas | #4a3122 | #f9f5f5 | 11.07:1 | AAA | Graphics and fills |
| --primary on --canvas | #a11413 | #f9f5f5 | 7.39:1 | AAA | Focus ring, graphics, large elements |

AAA is 7:1. AA is 4.5:1 for body text. "AA large text only" is 3:1, for text 24px and up or 19px bold and up.

## Typography

- **Heading font:** Figtree (sans). CSS: `"Figtree", -apple-system, sans-serif`
- **Body font:** Nunito Sans (sans). CSS: `"Nunito Sans", -apple-system, sans-serif`
- **Google Fonts:** https://fonts.googleapis.com/css2?family=Figtree:wght@400;600&family=Nunito+Sans:wght@400;700&display=swap
- **Headings:** 52px, weight 700, letter-spacing 2.5px, text-transform none.
- **Body:** 17px, weight 400, letter-spacing 0.5px, text-transform none.
- **Interface text** (tabs, buttons, field values, list cells): system monospace. The Nunito Sans body face is for reading copy only.
- **Small labels and captions:** system monospace.
- **Shape:** bold and geometric, radius 0px, large radius 0px.

### Recommended type for Raw Brutalist

- **Headings:** heavy display, neo-grotesk or condensed. Picks: Archivo Black, Inter Tight, Anton.
- **Body:** neo-grotesk. Picks: Inter, Archivo, IBM Plex Sans.
- **UI and mono:** JetBrains Mono, IBM Plex Mono, Space Mono.
- **Avoid:** soft serif, rounded, deco, groovy.
- **Why:** severe, honest layouts want a black-weight headline or a plain grotesk, because anything decorative softens the point.
- **Pairings** (heading / body): Archivo Black / Archivo; Inter Tight / Inter; Anton / IBM Plex Sans. The first is the recommended default.
- **Google Fonts for the recommended pairing:** https://fonts.googleapis.com/css2?family=Archivo:wght@400;600&family=Archivo+Black:wght@400&display=swap
- **This brand's heading font, Figtree,** is outside the recommended classes (humanist sans). It's acceptable, but check it against the picks.
- **This brand's body font, Nunito Sans,** is outside the recommended classes (humanist sans). It's acceptable, but check it against the picks.

## Materials (hard surfaces)

Surface recipes for this aesthetic, with the real values filled in. Apply them as written.

- **Hard surface:** `background: #f9f5f5; border: 2px solid #261c1c; border-radius: 0; box-shadow: 6px 6px 0 #261c1c;`
- **Pressed state:** `transform: translate(6px, 6px); box-shadow: none;`
- **Primary action block:** `background: #a11413; color: #ffffff; border: 2px solid #261c1c; border-radius: 0; box-shadow: 6px 6px 0 #261c1c; font: 600 13px ui-monospace, "SF Mono", Menlo, monospace; text-transform: uppercase;` Hover: translate(-2px, -2px) with `8px 8px 0 #261c1c`. Pressed: translate(6px, 6px) with no shadow.
- **Palette strips:** one full-width row with `border: 2px solid #261c1c`. Each color is a flat band sized by its coverage, with its label inside. Strips are separated by 2px #261c1c rules.
- **Selected segment:** `background: #261c1c; color: #f9f5f5;` Joined boxes share 2px #261c1c borders.
- **Field:** `background: #f9f5f5; border: 2px solid #261c1c; border-radius: 0; font: 15px ui-monospace, "SF Mono", Menlo, monospace; caret-color: #261c1c;`
- **Highlighter band:** `background: #4a3122; color: #ffffff; padding: 0 .06em;` It sits behind one headline word, never as colored text, and may run off the board edge.

## Patterns

Generative textures made only from the brand colors (the Patterns view in Branding Designer). Use them as hero backgrounds, cover panels, section dividers and swatch walls. Keep body text on a plain `--canvas` area, never on a pattern.

### Ink mapping

- ground = `--canvas`.
- ink1 = `--primary` (or `--ink` in the Mono colorway).
- ink2 = `--secondary`.
- spark = `--accent`, at most 5% of the pattern area.
- **Colorways:** Brand (the roles above), Mono (ink on canvas), Duotone (primary on canvas), Night (canvas and ink swapped). "+ tints" adds lighter mixes of primary and secondary toward the canvas.
- Never introduce a color a pattern doesn't get from these roles.

### Families that fit Raw Brutalist

- **Slab grid** (Modular grid, Mono): a seeded grid of modules: solid blocks, dot matrices, line fields, bar ramps and quarter circles.
- **Hard counter** (Counterchange, Mono): an even bar field with a centred figure where the bars swap places with the gaps.
- **Stair teeth** (Zigzag, Mono): interlocking teeth in flat colour bands, rounded, chevron or staircase.
- **Static** (Signal mosaic, Mono): glitched horizontal bands of pixel confetti, block mosaics, vertical smears and scanlines.
- **Tape rows** (Weave, Mono): stacked ribbons, each ruled with its own vertical stripe rhythm, like cloth tape laid edge to edge.
- **Block field** (Glyph grid, Mono): a lattice of tiny marks (squares, diamonds, rings and crosses) chosen by the height of a noise field.

No patterns are pinned for this brand yet. Pick from the families above.

## Do

- Use zero radius and 2px ink borders on every control and panel.
- Cast a hard offset shadow (6px 6px 0 ink) for raised things, and remove it entirely when pressed.
- Set every label, tab, button and table cell in monospace, uppercase, numbered like [01], with plain arrows.
- Keep the page monochrome: color appears only where a color role says so.

## Don't

- Round any corner, blur any shadow or add a gradient.
- Set the accent as text color. It is only ever the highlighter band behind a word.
- Soften the headline. It stays enormous, uppercase and tight, and may be cropped by the board edge.
- Spend primary on anything but the single main action and active states.

## CSS variables

```css
:root {
  --primary: #a11413;
  --secondary: #d14f34;
  --accent: #4a3122;
  --neutral-dark: #261c1c;
  --neutral-light: #f9f5f5;
  --on-primary: #ffffff;
  --on-accent: #ffffff;
  --font-heading: "Figtree", -apple-system, sans-serif;
  --font-body: "Nunito Sans", -apple-system, sans-serif;
  --heading-size: 52px;
  --body-size: 17px;
  --heading-weight: 700;
  --body-weight: 400;
  --heading-tracking: 2.5px;
  --body-tracking: 0.5px;
  --heading-transform: none;
  --body-transform: none;
  --radius: 0px;
  --radius-lg: 0px;
}

/* Derived tokens for the light-mode Raw Brutalist aesthetic. Computed; do not edit. */
:root {
  --canvas: #f9f5f5;
  --ink: #261c1c;
  --muted: #706868;
  --line: #d9d4d4;
  --on-secondary: #ffffff;
  --accent-ink: #4a3122;
  --pri-tint: #eedada;
  --sec-tint: #f3dad6;
}

/* Raw Brutalist material variables, referenced by the Materials above. */
:root {
  --ab-type-pick-w: 400;
  --ab-tr: 1;
  --ab-grid: rgba(38,28,28,0.06);
}
```
