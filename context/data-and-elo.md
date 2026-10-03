# Data and Elo v2

Sub-project context for the nflverse game dataset and the Elo v2 engine. General project context is in `CONTEXT.md`.

## Data sources and licenses

- **nflverse schedules (1999 to present).** Loaded with `nflreadpy.load_schedules()`, with the plain CSV `https://github.com/nflverse/nfldata/raw/master/data/games.csv` as a fallback. License CC-BY 4.0, so commercial use is fine with credit ("Data: nflverse (CC-BY 4.0)"). The raw download is cached in `data/raw/schedules.csv` (gitignored).
- **FiveThirtyEight NFL Elo game file (1970 to 1998 used).** Vendored at `data/sources/fivethirtyeight_nfl_elo.csv` (trimmed to 1970-2022 and eight columns; source URL, Internet Archive URL, retrieval date, sha256, and license in `data/sources/README.md`). License CC BY 4.0, so commercial use is fine with attribution: "NFL game results 1970–1998: FiveThirtyEight, CC BY 4.0." Site and dashboard credit line: "Data: nflverse (CC BY 4.0); 1970–1998 results: FiveThirtyEight (CC BY 4.0)." The live FiveThirtyEight URL now redirects to ABC News, hence the archive copy.
- **Legacy spreadsheet (validation only).** `legacy_2025/NFLELO_data.xlsx`. It came from Pro-Football-Reference and is **not a pipeline input** (see Resolved questions). It is loaded by `build.py` only to cross-check the 538 file for 1970-1998 REG and for the legacy-reproduction check. The build runs fine without it: if the file is missing, `build.py` skips both checks and writes "Legacy spreadsheet not present; skipped legacy validation." into the report, so the weekly job never depends on it.
- 538 rows for 1999 to 2022 are used only to validate nflverse (and vice versa), never in the dataset.

## Schema (`data/games.csv`)

One row per game: `game_id, season, date, week, game_type, home, away, home_score, away_score, neutral, source`, plus `home_moneyline, away_moneyline, spread_line, home_qb_id, away_qb_id, home_rest, away_rest` (nflverse rows only, blank for legacy rows). `home` and `away` are canonical franchise IDs. `game_type` is REG, WC, DIV, CON, or SB. `source` is `fivethirtyeight` or `nflverse`. Both sources include regular season and playoffs for every season: 1970 to 1998 has 6,409 games (6,140 REG, 269 playoff); incomplete games are dropped, so the 2026 season holds only played games. The moneyline, spread, QB, and rest columns are blank for 1970-1998. `neutral` comes from 538's `neutral` flag for 1970 to 1998 (29 games, all Super Bowls) and from nflverse's location for 1999 on (125 games).

## FiveThirtyEight loader (`data.load_538_games`)

- `team1` is home unless `neutral` is 1; `team2` is away. Verified: 6,140 of 6,140 REG games 1970-1998 agree with the legacy spreadsheet on home/away, and 99.98% of 1999-2022 non-neutral games agree with nflverse.
- `playoff` codes map to `game_type`: `w` to WC, `d` to DIV, `c` to CON, `s` to SB. 538 has no separate code for the 1982 strike-year tournament, whose eight first-round games are all WC, and 1990 on has four WC games per year. 1970 to 1977 has no WC round.
- There is no week column. REG week is the calendar week, Tuesday to Monday, counted from the season's first game (`calendar_weeks`); this reproduces the legacy spreadsheet's week for all 6,140 REG games. Playoff week is the last REG week plus 1 (WC), 2 (DIV), 3 (CON), or 4 (SB).
- The season is 538's `season` column, so January playoff games stay with the prior season, matching nflverse.

## Franchise mapping (`nflelo/teams.py`)

IDs are the current nflverse abbreviations. 538's codes are already franchise-consistent across eras (Oilers and Titans are TEN, Colts IND, Ravens BAL from 1996, Texans HOU from 2002, Cardinals ARI, Rams LAR, Raiders OAK, Chargers LAC, Washington WSH), so `franchise_538(code, season)` is a strict lookup. It raises on an unknown code or on a code before its franchise existed (BAL before 1996, HOU before 2002), which would mean the file's convention changed. The era-aware mapping below (`franchise`) is for nflverse and legacy codes. Mapping is by raw code and season:

- STL: before 1995 is the Cardinals (ARI); 1995 to 2015 is the Rams (LA).
- OAK, RAI, LVR: LV. SD, SDG: LAC. RAM, LAR, LA: LA. PHO: ARI. BOS: NE. OTI: TEN.
- HOU: before 1999 is the Oilers (TEN); 1999 on is the Texans (HOU, first games in 2002).
- BAL: before 1996 is the Colts (IND); 1996 on is the Ravens (BAL), a new franchise.
- Browns: CLE is one continuous franchise, matching the legacy script. The 1996 to 1998 gap is handled by the engine: absent teams still regress toward the mean each season. The Browns come back in 1999 with that regressed rating and do not get the expansion start. Treating the 1999 Browns as an expansion team is a reasonable alternative that was not tested.
- Expansion starts: any franchise first seen after 1970 starts at `expansion_start` (SEA and TB 1976, CAR and JAX 1995, BAL 1996, HOU 2002).

## Engine (`nflelo/elo.py`)

Standard Elo on a 1500 scale, one update per game. Expected home probability uses `rating_home + HFA - rating_away`; HFA is 0 for neutral games. The MOV multiplier is `ln(|margin|+1) * 2.2 / (diff * 0.001 + 2.2)`, where `diff` is the winner's rating minus the loser's, including HFA (538 form). Ties count 0.5 with multiplier 1. Between seasons every rating moves lambda of the way to 1500. HFA is either fixed or online: after each non-neutral game, `HFA += k_hfa * (actual_home - expected_home)`, starting at 65 in 1970. Playoff games are always predicted (1970 onward now); they update ratings only when `include_playoffs` is on. Output per game: pre-ratings, HFA used, home win probability, MOV multiplier, post-ratings.

The legacy config (`LEGACY_CONFIG`) uses the legacy script's actual MOV rule: absolute rating difference and a cap of 2.0 on the multiplier.

## Evaluation protocol (`nflelo/evaluate.py`, `scripts/tune.py`)

Ratings warm up from 1970. Tuning scores REG games in 1980 to 2009 and sees no later games. The test window is 2010 to 2025 REG games; the partial 2026 season is excluded. Metrics: Brier (primary), log loss, accuracy. The market is the vig-removed moneyline from the nflverse schedule file (Lee Sharpe's nfldata; it does not document which sportsbook or whether lines are opening or closing), scored on the test games that have moneylines (4,174 of 4,175), the same games for every model.

Grid (3,072 configs, 33 seconds): K {15, 20, 25, 30}; lambda {0.10 to 0.50, 8 values}; HFA fixed {35, 45, 55, 65} or online with k_hfa {0.5, 1, 2, 4} (init 65), with and without MOV scaling of the HFA update; include_playoffs on or off; expansion start {1300, 1500}; MOV cap none or 2.0.

## Results

- **Validation** (`data.validate_games`, printed by `build.py`, full tables in `outputs/model_report.md`). Games are keyed on season, game type, franchise pair, and n-th meeting of the pair that season. 538 vs the legacy spreadsheet, 1970-1998 REG: 6,140 of 6,140 match on both scores, home/away, neutral flag, and date (100%). 538 vs nflverse, 1999-2022 REG: 6,151 of 6,151 match on both scores (100%); home/away agrees on 99.98% of non-neutral games and 11 games differ on the neutral/host label (2005 Saints, Bills in Toronto and Detroit, 49ers in Arizona, and similar relocated games), which does not affect 1999+ because those rows come from nflverse. Playoffs 1999-2022: 270 of 270. The pre-1999 mapping rules are checked through the independent legacy codes and the franchises-per-season table in the report (every franchise plays the same number of games each year through 1998).
- **Legacy reproduction.** New engine with legacy parameters on the legacy spreadsheet alone (still loaded for this check), 1970 to 2025 REG, 13,107 games: Brier 0.2208 at lambda 0.15 (the legacy default). At lambda 0.20, the legacy sensitivity-grid best, Brier is 0.2201 and log loss 0.6326, matching the legacy numbers exactly.
- **Tuned winner (tune window only).** Re-run 2026-10-02 after the 538 switch. K 20, lambda 0.40, fixed HFA 65, **playoff updates on** (pre-1999 playoffs now exist), expansion start 1300, no MOV cap. Tune Brier 0.21982. The top ten configs span 0.0002 Brier, so this is a weak winner. The old winner without playoff updates scored 0.21990.
- **Default config (2026-10-02 decision, unchanged after the 538 switch).** K 20, lambda 0.40, online HFA (init 65, k_hfa 0.5), no playoff updates, expansion start 1300, no MOV cap. Tune Brier 0.22000. The overall winner beats it by 0.00018, below the 0.0005 bar for changing it. The best online-HFA config with playoff updates scores 0.21992 (0.00008 better). Online HFA was chosen because a fixed 65 implies a ~59% home win rate against ~53.6% in the 2020s.
- **Test 2010 to 2025, all 4,175 REG games.** Elo v2 (default) Brier 0.2201, log loss 0.6317, accuracy 64.0%, identical to the pre-538 build (the 1970-1998 REG rows are the same games, and playoffs do not update ratings). Legacy scaled: 0.2244, 0.6419, 63.4%. The new fixed-HFA tune winner (playoffs on) scored 0.2209; the best online-HFA config with playoffs on scored 0.2199.
- **Test against the market, 4,174 games with moneylines.** Elo v2 0.2201, legacy 0.2245, market 0.2104. Elo v2 trails the market by 0.0097 Brier (standard error 0.0014).
- **Home-field advantage.** The actual home win rate fell from 0.596 in the 1990s to 0.536 in the 2020s. The default's learned HFA averages 58.6 in the 1970s, 65.7 in the 2000s, 40.6 in the 2020s, and 37.8 now.

## Resolved questions

- **Where did the legacy 1970-1998 data come from?** Pro-Football-Reference. Sports Reference's terms (sports-reference.com/termsofuse.html section 5, and data_use.html) restrict building substitute databases or sites and machine-learning use, so the pipeline does not use that data. 1970-1998 now comes from FiveThirtyEight's CC BY 4.0 game file, which matches the legacy spreadsheet on all 6,140 REG games. `legacy_2025/` remains as the archived 2025 paper and is not a pipeline input; the spreadsheet is read only for the validation report and the legacy-reproduction check.

## Run

```
.venv/bin/python scripts/build.py          # data, Elo, outputs, report
.venv/bin/python scripts/tune.py           # re-run the grid; copy the winner into DEFAULT_CONFIG
.venv/bin/python -m pytest
```

`build.py` runs offline against `data/sources/` and `data/raw/schedules.csv` (`--offline`); only the nflverse download needs the network.

`tune.py` writes `outputs/tuning_results.csv` and `outputs/tuning_best.json`; `build.py` reads them for the report.

## Dashboard (`scripts/dashboard.py`)

Renders `outputs/dashboards/team_summary.png` (16x11 in, 200 dpi, about 0.5 MB) from `outputs/elo_games.csv` and `outputs/ratings_current.json`; `build.py` calls it at the end. Four panels, each backed by a DataFrame from `build_tables()` (`--show-tables` prints them): power ladder (all 32 teams, bars from 1500, 7-day rating and rank change), all-time REG win % vs. all-time average Elo (both axes cover every REG game since 1970), top 10 peak-to-low swings in post-game rating with the season of each extreme, and top 10 all-time average Elo. Records and averages use REG games only (ties count half). Expansion-era franchises (BAL, HOU, CAR, JAX, SEA, TB) average over fewer games, and their swings start from the 1300 expansion rating. Team colors are in `nflelo/colors.py` (from the legacy palette, remapped to current codes; no logos).

## Known gaps

- No QB adjustment yet. Starting QB IDs are in the dataset for 1999 on.
- No pre-1999 market data, and no independent check of pre-1999 scores beyond franchise game counts.
- Pre-1999 neutral flags come only from 538 (29 Super Bowls); other neutral-site regular-season games, if any, are not flagged.
- The default config does not learn from playoffs (playoff updates tied or lost on the tune window).
