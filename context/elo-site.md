# Elo site

Sub-project context for the static NFLELO website in `site/`. General project context is in `CONTEXT.md`; the model is in `context/data-and-elo.md`.

## Shape

- Static, no build step, no server. `site/index.html`, `site/styles.css`, `site/app.js` (plain JS, hand-built SVG charts, no CDN scripts) and `site/data/*.json`. Only external request: Google Fonts (Archivo for text, JetBrains Mono for numbers). Must be served over http (it fetches `./data/*.json`); `python3 -m http.server 8765 --directory site`, or `nflelo-site` in `.claude/launch.json`.
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

- Dark first, light via `prefers-color-scheme`. One UI accent (blue, also the Elo series color). Team colors appear only inside data marks; dark team colors are lifted (or light ones darkened) in JS to hold 3:1 against the page surface, set as `--c-d` / `--c-l` per mark so the theme switches without a redraw.
- Chart colors follow the `dataviz` skill: Elo blue and market orange validated with its script (light and dark, all checks pass); heatmap is a diverging blue/red scale around 1500 with a gray midpoint; every chart has a legend or direct label where it has two series, a tooltip on hover and keyboard focus, and a table view.
- Deviations from `design-taste-frontend` (written for landing pages, not data pages): it prefers self-hosted fonts over a Google Fonts link, and asks for real imagery; this site uses the Google Fonts link requested in the brief and no imagery or logos. Records tables keep a hairline under each row because they are data tables.
- No em or en dashes anywhere in the page copy.

## Open items

- The weekly GitHub Action and Pages deployment.
- Playoff odds and starting-QB adjustment are not on the site yet.
