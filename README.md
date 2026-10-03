# NFL Elo

Elo ratings for every NFL franchise from 1970 to the present, built from open game data and tuned on held-out seasons. The current version is Elo v2. It predicts each game's home win probability and is scored against the betting market (Brier score).

The original 2025 paper, charts, and single-file script are preserved in [`legacy_2025/`](legacy_2025/). That folder is an archive, not a pipeline input; its spreadsheet is used only to cross-check the 1970 to 1998 data.

Data: nflverse (CC BY 4.0); 1970–1998 results: FiveThirtyEight (CC BY 4.0). The 1970 to 1998 games are vendored in [`data/sources/`](data/sources/README.md) with source, license, and checksum. Everything the pipeline reads is free and licensed for commercial use with attribution.

## Run it

```
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/build.py        # downloads nflverse, runs Elo, writes outputs/
.venv/bin/python scripts/tune.py         # optional: re-run the grid search (about 35 seconds)
.venv/bin/python -m pytest
```

`build.py --offline` reuses the cached download in `data/raw/`.

## Outputs

- `data/games.csv`: one row per game (regular season and playoffs), 1970 to now.
- `data/sources/`: the vendored FiveThirtyEight game file for 1970 to 1998 and its license notes.
- `outputs/elo_games.csv`: pre-game ratings, win probability, post-game ratings, and the home-field advantage used, per game.
- `outputs/ratings_current.json`: current rating, rank, and 7-day change per franchise.
- `outputs/model_report.md`: data validation (FiveThirtyEight vs the legacy spreadsheet and vs nflverse), tuning, and the held-out test against the legacy model and the betting market.
- `outputs/dashboards/team_summary.png`: the Team Summary dashboard (see below).

## Dashboard

`scripts/dashboard.py` draws the Team Summary PNG (power ladder, Elo vs. winning, biggest rises and falls, highest average Elo) from `outputs/elo_games.csv` and `outputs/ratings_current.json`. `build.py` regenerates it every run; to redraw it alone, or to print the four tables behind the panels:

```
.venv/bin/python scripts/dashboard.py --show-tables
```

## Website

`site/` is a static one-page site (no build step, no server code) that shows the power ladder, this week's picks against the Vegas line, a team explorer, the luck board, league history, records, and the model scorecard. It reads `site/data/*.json`, which `scripts/export_site.py` writes from `outputs/` and the cached schedule; `build.py` runs the exporter at the end of every run.

```
.venv/bin/python scripts/export_site.py                 # refresh site/data/ only
python3 -m http.server 8765 --directory site            # then open http://localhost:8765
```

The same server is configured as `nflelo-site` in `.claude/launch.json`. Any static host works, including GitHub Pages (publish the `site/` folder). Fonts load from Google Fonts; the charts are hand-built SVG with no other external requests. Details are in [`context/elo-site.md`](context/elo-site.md).

Project notes are in [`CONTEXT.md`](CONTEXT.md) and [`context/data-and-elo.md`](context/data-and-elo.md).
