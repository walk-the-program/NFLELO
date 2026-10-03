# Data and Elo v2

Sub-project context for the nflverse game dataset and the Elo v2 engine. General project context is in `CONTEXT.md`.

## Data sources and licenses

- **nflverse schedules (1999 to present).** Loaded with `nflreadpy.load_schedules()`, with the plain CSV `https://github.com/nflverse/nfldata/raw/master/data/games.csv` as a fallback. License CC-BY 4.0, so commercial use is fine with credit ("Data: nflverse (CC-BY 4.0)"). The raw download is cached in `data/raw/schedules.csv` (gitignored).
- **Legacy spreadsheet (1970 to 1998 used).** `legacy_2025/NFLELO_data.xlsx`. The original source is unknown. If it came from Pro-Football-Reference, their terms need a check before any commercial use. This is an open question for Walker.
- Legacy rows for 1999 and later are used only to validate nflverse, never in the dataset.

## Schema (`data/games.csv`)

One row per game: `game_id, season, date, week, game_type, home, away, home_score, away_score, neutral, source`, plus `home_moneyline, away_moneyline, spread_line, home_qb_id, away_qb_id, home_rest, away_rest` (nflverse rows only, blank for legacy rows). `home` and `away` are canonical franchise IDs. `game_type` is REG, WC, DIV, CON, or SB. `source` is `legacy` or `nflverse`. Seasons 1970 to 1998 are regular season only (6,140 games). Seasons 1999 on have regular season and playoffs; incomplete games are dropped, so the 2026 season holds only played games. The legacy file has no neutral-site flag, so `neutral` is only set for nflverse games (96 of them).

Season for legacy rows is derived from the date (January and February belong to the prior season). The legacy two-row-per-game format is collapsed by keeping the home row.

## Franchise mapping (`nflelo/teams.py`)

IDs are the current nflverse abbreviations. Mapping is by raw code and season:

- STL: before 1995 is the Cardinals (ARI); 1995 to 2015 is the Rams (LA).
- OAK, RAI, LVR: LV. SD, SDG: LAC. RAM, LAR, LA: LA. PHO: ARI. BOS: NE. OTI: TEN.
- HOU: before 1999 is the Oilers (TEN); 1999 on is the Texans (HOU, first games in 2002).
- BAL: before 1996 is the Colts (IND); 1996 on is the Ravens (BAL), a new franchise.
- Browns: CLE is one continuous franchise, matching the legacy script. The 1996 to 1998 gap is handled by the engine: absent teams still regress toward the mean each season. The Browns come back in 1999 with that regressed rating and do not get the expansion start. Treating the 1999 Browns as an expansion team is a reasonable alternative that was not tested.
- Expansion starts: any franchise first seen after 1970 starts at `expansion_start` (SEA and TB 1976, CAR and JAX 1995, BAL 1996, HOU 2002).

## Engine (`nflelo/elo.py`)

Standard Elo on a 1500 scale, one update per game. Expected home probability uses `rating_home + HFA - rating_away`; HFA is 0 for neutral games. The MOV multiplier is `ln(|margin|+1) * 2.2 / (diff * 0.001 + 2.2)`, where `diff` is the winner's rating minus the loser's, including HFA (538 form). Ties count 0.5 with multiplier 1. Between seasons every rating moves lambda of the way to 1500. HFA is either fixed or online: after each non-neutral game, `HFA += k_hfa * (actual_home - expected_home)`, starting at 65 in 1970. Playoff games are always predicted; they update ratings only when `include_playoffs` is on. Output per game: pre-ratings, HFA used, home win probability, MOV multiplier, post-ratings.

The legacy config (`LEGACY_CONFIG`) uses the legacy script's actual MOV rule: absolute rating difference and a cap of 2.0 on the multiplier.

## Evaluation protocol (`nflelo/evaluate.py`, `scripts/tune.py`)

Ratings warm up from 1970. Tuning scores REG games in 1980 to 2009 and sees no later games. The test window is 2010 to 2025 REG games; the partial 2026 season is excluded. Metrics: Brier (primary), log loss, accuracy. The market is the vig-removed moneyline from the nflverse schedule file (Lee Sharpe's nfldata; it does not document which sportsbook or whether lines are opening or closing), scored on the test games that have moneylines (4,174 of 4,175), the same games for every model.

Grid (3,072 configs, 33 seconds): K {15, 20, 25, 30}; lambda {0.10 to 0.50, 8 values}; HFA fixed {35, 45, 55, 65} or online with k_hfa {0.5, 1, 2, 4} (init 65), with and without MOV scaling of the HFA update; include_playoffs on or off; expansion start {1300, 1500}; MOV cap none or 2.0.

## Results

- **Validation.** Legacy xlsx vs nflverse, regular seasons 1999 to 2025: 6,967 of 6,967 games match on season, week, franchise pair, and both scores (100%), including home/away. This validates the xlsx format and the 1999+ code mapping. It cannot test the pre-1999 mapping rules, so `outputs/model_report.md` also lists franchises and games per franchise by season (every franchise plays the same number of games each year, which a mis-mapped code would break).
- **Legacy reproduction.** New engine with legacy parameters on legacy data, 1970 to 2025 REG, 13,107 games: Brier 0.2208 at lambda 0.15 (the legacy default). At lambda 0.20, the legacy sensitivity-grid best, Brier is 0.2201 and log loss 0.6326, matching the legacy numbers exactly.
- **Tuned winner (tune window only).** K 20, lambda 0.40, fixed HFA 65, no playoff updates, expansion start 1300, no MOV cap. Tune Brier 0.2199. The top ten configs span 0.0001 Brier, so this is a weak winner.
- **Default config (2026-10-02 decision).** The best online-HFA config from the tune window: K 20, lambda 0.40, online HFA (init 65, k_hfa 0.5), no playoff updates, expansion start 1300, no MOV cap. Tune Brier 0.2200, a tie with the overall winner. Online HFA was chosen because a fixed 65 implies a ~59% home win rate against ~53.6% in the 2020s.
- **Test 2010 to 2025, all 4,175 REG games.** Elo v2 (default) Brier 0.2201, log loss 0.6317, accuracy 64.0%. Legacy scaled: 0.2244, 0.6419, 63.4%. The fixed-HFA tune winner scored 0.2211.
- **Test against the market, 4,174 games with moneylines.** Elo v2 0.2201, legacy 0.2245, market 0.2104. Elo v2 trails the market by 0.0097 Brier (standard error 0.0014).
- **Home-field advantage.** The actual home win rate fell from 0.596 in the 1990s to 0.536 in the 2020s. The default's learned HFA averages 58.6 in the 1970s, 65.7 in the 2000s, 40.6 in the 2020s, and 37.8 now.

## Run

```
.venv/bin/python scripts/build.py          # data, Elo, outputs, report
.venv/bin/python scripts/tune.py           # re-run the grid; copy the winner into DEFAULT_CONFIG
.venv/bin/python -m pytest
```

`tune.py` writes `outputs/tuning_results.csv` and `outputs/tuning_best.json`; `build.py` reads them for the report.

## Dashboard (`scripts/dashboard.py`)

Renders `outputs/dashboards/team_summary.png` (16x11 in, 200 dpi, about 0.5 MB) from `outputs/elo_games.csv` and `outputs/ratings_current.json`; `build.py` calls it at the end. Four panels, each backed by a DataFrame from `build_tables()` (`--show-tables` prints them): power ladder (all 32 teams, bars from 1500, 7-day rating and rank change), all-time REG win % vs. all-time average Elo (both axes cover every REG game since 1970), top 10 peak-to-low swings in post-game rating with the season of each extreme, and top 10 all-time average Elo. Records and averages use REG games only (ties count half). Expansion-era franchises (BAL, HOU, CAR, JAX, SEA, TB) average over fewer games, and their swings start from the 1300 expansion rating. Team colors are in `nflelo/colors.py` (from the legacy palette, remapped to current codes; no logos).

## Known gaps

- No pre-1999 playoff games.
- No QB adjustment yet. Starting QB IDs are in the dataset for 1999 on.
- No pre-1999 market data, and no independent check of pre-1999 scores beyond franchise game counts.
- The legacy xlsx has no neutral-site flag.
- The default config does not learn from playoffs (playoff updates tied or lost on the tune window).
