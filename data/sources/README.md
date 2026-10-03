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
