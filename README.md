# NFL Elo

Elo ratings for every NFL franchise from 1970 to the present, built from open game data and tuned on held-out seasons. The current version is Elo v2. It predicts each game's home win probability and is scored against the betting market (Brier score).

The original 2025 paper, charts, and single-file script are preserved in [`legacy_2025/`](legacy_2025/).

Data: nflverse (CC-BY 4.0). Seasons before 1999 come from a hand-built game log in `legacy_2025/` whose original source is still an open question.

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

- `data/games.csv`: one row per game, 1970 to now.
- `outputs/elo_games.csv`: pre-game ratings, win probability, post-game ratings, and the home-field advantage used, per game.
- `outputs/ratings_current.json`: current rating, rank, and 7-day change per franchise.
- `outputs/model_report.md`: data validation, tuning, and the held-out test against the legacy model and the betting market.

Project notes are in [`CONTEXT.md`](CONTEXT.md) and [`context/data-and-elo.md`](context/data-and-elo.md).
