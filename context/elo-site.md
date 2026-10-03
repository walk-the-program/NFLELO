# Elo site

Sub-project context for the static NFLELO website in `site/`. General project context is in `CONTEXT.md`; the model is in `context/data-and-elo.md`.

## Shape

- Static, no build step, no server. `site/index.html`, `site/styles.css`, `site/app.js` (plain JS, hand-built SVG charts, no CDN scripts) and `site/data/*.json`. Only external request: Google Fonts (Figtree for headings, Nunito Sans for body). Must be served over http (it fetches `./data/*.json`); `python3 -m http.server 8765 --directory site`, or `nflelo-site` in `.claude/launch.json`.
- Hostable on GitHub Pages by publishing `site/`. The weekly GitHub Action (not built yet) only needs to run `scripts/build.py` and commit `site/data/`.
- Sections, in order: header with four tiles, power ladder, this week, team explorer, luck board, league history (tapestry heatmap and parity), records, model scorecard with methodology.

## Exporter (`scripts/export_site.py`)

Reads `outputs/elo_games.csv`, `outputs/ratings_current.json`, `outputs/model_report.md` (legacy Brier only) and the cached `data/raw/schedules.csv`. Writes eight deterministic JSON files (about 0.42 MB total; `history.json` is 0.38 MB of it): `meta`, `ladder`, `upcoming`, `history`, `luck`, `tapestry`, `records`, `scorecard`. `build.py` calls `export()` at the end. Display names, conference and division live in `nflelo/meta.py`.

- **History is not downsampled.** One point per team per regular-season game week since 1970 (about 30k points). Playoff games do not update ratings, so they add nothing. If the file ever matters, drop to season-end points for pre-2000 seasons.
- **Spread sign.** nflverse `spread_line` is positive when the home team is favored (checked: games with `spread_line` above 7 average a home margin near +11). Elo spread is `(home + hfa - away) / 25` with the same sign, so the gap is `elo_spread - vegas_spread`. The site shows both as "FAVORITE -points".
- **This week** is the earliest week of the current season that still has unplayed games, so a Thursday `as_of` shows the rest of that week (the Thursday game is already in the ratings). The three largest absolute Elo-vs-Vegas gaps are flagged. If no lines exist, picks still show and the Vegas fields are null.
- **Partial season.** The season in progress is left out of season-end records, best/worst seasons and the scorecard; the tapestry keeps its column and marks it (`partial_season`).
- **Luck** is actual wins minus the sum of pre-game win probabilities, for last season and the current season to date.
- **Scorecard** headline numbers are recomputed from `elo_games.csv` on the held-out 2010-2025 window (Elo 0.2201 Brier, 64.0% accuracy; market 0.2104 and 66.6% on the 4,174 games with moneylines; legacy 0.2244 parsed from the report). Per-season Brier uses games with moneylines (2006 on).

## Design decisions

- **Token system (styles.css, three layers).** Layer 1 is Walker's tokens, verbatim at the top of `:root` (`--primary #3a5979`, `--secondary #667f99`, `--accent #2a4c6f`, `--neutral-dark #1c2126`, `--neutral-light #f5f7f9`, `--on-primary`, `--on-accent`, `--font-heading` Figtree, `--font-body` Nunito Sans, heading 52px / 700 / 2.5px tracking, body 17px / 400 / 0.5px tracking, radius 0). Layer 2 is semantic (`--bg`, `--bg-raised`, `--ink`, `--ink-2`, `--ink-3`, `--rule`, `--rule-strong`, `--accent-ink`, `--mark`, `--fill`, `--on-fill`, `--focus`), built from layer 1 with `color-mix`, so there is no raw hex in UI CSS. Layer 3 is data colors (Vegas orange, heatmap poles, up/down status), kept apart because they are not UI colors. The primitive `--accent` is the dark brand blue, so the semantic names `--mark` (lines, underline, focus) and `--accent-ink` (text) are used for the on-screen accent instead.
- **Themes.** Light is `--neutral-light` background with `--neutral-dark` text; dark swaps them, both via `prefers-color-scheme`. Light uses `--accent` for links and `--primary` for lines and filled controls (6.8:1 and 8.3:1 on the background). In dark, `#3a5979` is 2.2:1 and `#2a4c6f` is 1.8:1 on `#1c2126`, so dark uses tints of `--primary` mixed with `--neutral-light`: 55% primary for text and links (6.05:1), 70% primary for chart lines, nav underline and focus ring (4.4:1). Filled chips keep `--primary` with `--on-primary` text (7.3:1) and get an `--accent-ink` edge in dark so the shape holds against the background.
- **Type.** Headings, wordmark, big numerals and team abbreviations use Figtree at weight 700 with 2.5px tracking; section H2s are 52px at desktop (`clamp(2rem, 6.4vw, 52px)`), the wordmark goes to 112px. Body, tables, axes and small labels use Nunito Sans at 17px base with 0.5px tracking. Small label headings (tile and score labels, callout heading) use the body face. Both fonts have tabular figures (checked: `1111`, `0000` and `1630` render at the same width), so aligned columns keep working without a mono face. No tracking had to be tightened.
- **Radius** is 0 for every control, bar, tooltip and select (the select arrow is a CSS chevron). Bars no longer have rounded ends. Circular dots in charts and the legend dot key are marks, not shapes of the UI.
- **Data colors.** Elo series is `--primary` (dark: the 70% tint). Market/Vegas stays orange `#d95926` in both themes: validated with the dataviz script, contrast 4.2:1 dark and 3.6:1 light, CVD separation from Elo 16 to 17 and normal-vision 21 to 28. The only check that fails is the chroma floor, because the brand blue is desaturated (OKLCH chroma 0.04 to 0.06); it is separated by hue and lightness, and both series carry legend and direct labels. The home-field chart plots the actual home win rate as plain dots (not a line) so it cannot be mixed up with the Elo line. Team colors are lifted to 3:1 against the new backgrounds (constants in `app.js`), heatmap poles unchanged.
- Chart colors follow the `dataviz` skill; every chart has a legend or direct label where it has two series, a tooltip on hover and keyboard focus, and a table view.
- Deviations from `design-taste-frontend` (written for landing pages, not data pages): it prefers self-hosted fonts over a Google Fonts link, and asks for real imagery; this site uses the Google Fonts link and no imagery or logos. Records tables keep a hairline under each row because they are data tables. Rounded bar ends from the dataviz mark spec were dropped to honor radius 0.
- No em or en dashes anywhere in the page copy.

## Open items

- The weekly GitHub Action and Pages deployment.
- Playoff odds and starting-QB adjustment are not on the site yet.
