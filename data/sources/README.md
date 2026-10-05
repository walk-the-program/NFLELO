# Vendored data sources

## fivethirtyeight_nfl_elo.csv

NFL game results used for seasons 1970 to 1998 (and as an independent check against nflverse for 1999 to 2022).

- **Source:** FiveThirtyEight NFL Elo game file, `https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv`. That URL now redirects to ABC News, so the copy was taken from the Internet Archive: `https://web.archive.org/web/2024id_/https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv`. The dataset is also published in the FiveThirtyEight data repository, `https://github.com/fivethirtyeight/data`.
- **Retrieved:** 2026-10-02.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/. The license in the repository above covers the data; Sections 2(a)(1) and 4 allow copying, extracting, and reusing substantial portions of the database, including commercially, as long as attribution is kept.
- **Required attribution:** "NFL game results 1970–1998: FiveThirtyEight, CC BY 4.0."
- **Changes made:** the archived file (17,379 rows, 33 columns, seasons 1920 to 2022) was cut to seasons 1970 to 2022 (12,830 rows) and to eight columns: `date, season, neutral, playoff, team1, team2, score1, score2`. Values were not edited. The Elo, QB, and rating columns were dropped; this project computes its own ratings.
- **SHA-256 of the vendored file:** `786d0c4641b2a2961fb9a16197093cb1429090c19941864e1405c623a1e53287`
- **SHA-256 of the full archived download (not vendored):** `8f6831ac223de245b612044601bc369a6b222a808cb55e737939504118525733`

Column notes (verified against nflverse and the legacy spreadsheet, see `outputs/model_report.md`):

- `team1` is the home team unless `neutral` is 1; `team2` is the away team.
- `playoff` is blank for regular-season games and `w`, `d`, `c`, `s` for wild card, divisional, conference championship, and Super Bowl games. The pipeline maps these to `WC`, `DIV`, `CON`, `SB`.
- Team codes are franchise-consistent: Oilers and Titans are `TEN`, Colts are `IND`, Ravens `BAL` (from 1996), Texans `HOU` (from 2002), Cardinals `ARI`, Rams `LAR`, Raiders `OAK`, Chargers `LAC`, Washington `WSH`.
- The file has no week number; the pipeline derives it (see `nflelo/data.py`).

## playoff_seeds_2002_2025.csv

The actual playoff seeds of every conference, 2002 to 2025 (300 rows: `season, conf, seed, team`). Used only to validate the tiebreaker code (`nflelo/ml/sim/tiebreak.py`); it is never a model input.

- **Source:** the `seed` column of `data/standings.csv` in nflverse's `nfldata` repository, `https://raw.githubusercontent.com/nflverse/nfldata/master/data/standings.csv`, retrieved 2026-10-04. Seeds are facts of public record (the NFL's playoff brackets).
- **How it is used:** `nflelo/ml/sim/history.py` first derives every seeding consistent with that season's nflverse playoff games (bye teams, wild-card pairings, the #1 seed hosting the lowest remaining seed, higher seeds hosting). That fixes the seeds completely in 15 of the 48 conference-seasons. In the other 33 the bracket leaves some order open (usually #1 vs #2, or #3 vs #4), and this file decides it. The loader checks that every seeding in this file is one the bracket allows; all 48 are.
- **Changes made:** rows cut to 2002-2025 (the current eight-division league) and to the four columns; team codes mapped to this project's franchise IDs (`OAK` to `LV`, `SD` to `LAC`, `STL` to `LA`).
- **License:** the `nfldata` repository has no license file. What is vendored here is a list of facts (which team held which seed), not a creative work; it is used for testing only.
- **SHA-256 of the vendored file:** `be14411861d8f9f068d167e52cba9c0b5714bfed86d975734d40e5c750377a9c`
