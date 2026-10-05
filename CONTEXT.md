# NFL Elo Project: General Context

This is the main context document for the project. Read it first. Every decision, data source, and sub-project gets recorded here, or gets its own file under `context/` with a link from the index at the bottom.

Last updated: 2026-10-03 (ML M3 holdout run)

## What this project is

Walker Tracy wrote an NFL Elo analysis in fall 2025, covering regular seasons from 1970 to 2024 and later extended through 2025. It includes a paper, a Python pipeline, and about 20 charts. The goal now has two parts:

1. **Elo website.** Publish the Elo ratings and the best stats from the paper on a public site that updates automatically every Wednesday.
2. **Machine learning.** Use the much richer nflverse data (play-by-play, players, rosters, betting lines) to build models that can also go on the site.

Hard constraints: the data must be **free**, and it must be **safe for commercial use**. No paid or metered APIs.

**Purpose (Walker, 2026-10-05):** NFLELO is a portfolio piece for getting hired by NFL teams, not a betting product. The priority is the most rigorous, premium model possible, with the full set of site pages built around it. No betting features: no picks against the spread, no units, no EV bets, no sportsbook offers. Vegas appears only as a benchmark to measure against. Every page should read like work a front office would respect: clear methods, honest uncertainty, and verifiable results.

## What exists today (as of 2026-10-02)

**Elo v2, the current system.** `scripts/build.py` pulls nflverse, rebuilds `data/games.csv` (1970 onward), runs Elo, and writes `outputs/` (per-game ratings, current ratings JSON, model report). Details are in `context/data-and-elo.md`.

**Legacy 2025 analysis.** Everything below lives in `legacy_2025/`, an archive and not a pipeline input. The pipeline takes 1970-1998 games from FiveThirtyEight's CC BY 4.0 file in `data/sources/`; the spreadsheet is read only to cross-check it.

| Path (inside `legacy_2025/`) | What it is |
|---|---|
| `NFL_ELO_organized.py` | The entire pipeline in one file (~4,500 lines): data load, Elo, metrics, and plots. Configured through the `NFLConfig` class. |
| `NFLELO_data.xlsx` | Validation only, not a source (it came from Pro-Football-Reference; see `context/data-and-elo.md`). Game log, 26,214 team-game rows from 1970 to 2025. Columns: GameID, Team, Date, Day, Game Number, Week, Hosting, Opp, Result. |
| `NFL Elo Analysis 1970-2024.pdf`, `LaTeX Files/main.tex` | The paper. |
| `Outputs/`, `Outputs_Addendum/` | PNG charts and CSVs for 2024 and for the 2025 addendum. |
| `NFL_Dash.html` | An earlier static dashboard. |
| `Markdowns/` | Notes on documentation, the 2002 realignment, and team mapping fixes. |
| `README.md` | Usage notes for the legacy script. |
| `Other/` | Word drafts. Kept out of git because the repo is public. |

**ML foundations (M1 and M2, built 2026-10-03).** `nflelo/ml/` caches nflverse play-by-play for 1999-2025 into the gitignored `data/raw/ml/`, builds leak-proof team-efficiency features, and scores models in a walk-forward harness with a locked 2020-2025 holdout. Runs are logged to `experiments/runs/`. **M3 game model (passed 2026-10-04).** Logistic regression on Elo log-odds, opponent-adjusted EPA margin, and a starting-QB delta. DEV 2006-2019 Brier 0.2151 vs Elo 0.2178 (CI excludes zero). On the one holdout run (2020-2025, 1,615 games) it scored 0.2195 vs Elo 0.2231 and the market 0.2096, beating Elo in all six seasons; its ECE of 0.028 missed the 0.02 bar, which turned out to be too tight for the sample size. **M3 is marked passed** (Walker, 2026-10-04). Details are in `context/ml.md`.

### Legacy Elo model (2025)

- Start rating 1000. K = 20. Home-field advantage (HFA) = 55 Elo points, set to 0 for international games.
- Margin-of-victory multiplier in the 538 style.
- Between seasons, ratings regress 15% toward the mean (λ = 0.15).
- Regular season only, with no playoffs.
- Extra metrics: luck index (actual wins minus Elo-expected wins), close-game luck, parity, calibration (Brier score, log loss, Murphy diagram), strength of schedule, and upsets.
- Sensitivity grid: the best Brier score is about **0.2201** (log loss 0.6326) across 13,107 games. The top results are nearly tied, and the best one uses λ = 0.2, which is the edge of the grid that was searched.

## Findings from the 2026-10-02 review

1. **The 2025 rows in the spreadsheet have two columns swapped.** In all 544 rows from 2025-09-04 to 2026-01-04, `Day` holds the season year and `Game Number` holds the weekday. Moving to nflverse removes the hand-maintained file and fixes this.
2. **The fixed home-field advantage is too high for the modern NFL.** With 55 points, two equal teams give the home side about a 57.8% win chance. Actual home win rates in the nflverse data run 59.7% for 1999 and earlier, about 57% for 2000 to 2014, 56% for 2015 to 2019, and **53.1% for 2020 to 2024**. HFA should change by era, or be learned on a rolling basis.
3. **Season regression is probably too small.** The best grid result sits at the grid's edge (λ = 0.2). 538 used about a third. The grid should be wider, and tuning should use held-out seasons rather than the same games the model is scored on.
4. **The market benchmark exists now.** Betting-market moneylines from the nflverse schedule file (sportsbook and timing undocumented), with the vig removed, score a Brier of **0.211** on 5,100 regular-season games from 2006 to 2026. Our Elo scores 0.220, but over 1970 to 2025, so the two aren't directly comparable yet. The paper's "Further Research" asks how Elo compares to betting markets, and this data answers that.
5. **The paper's own future-work list fits nflverse well.** Playoffs are in the schedule data. Player-level ratings can be built from rosters, snap counts, and play-by-play. The starting QB for every game since 1999 is in the schedule file.

## Data source: nflverse

- Site: https://nflverse.nflverse.com/. Python access through `nflreadpy` (MIT license), e.g. `import nflreadpy as nfl; nfl.load_schedules()`. Files are also plain GitHub releases at `github.com/nflverse/nflverse-data/releases`, with no key and no account required.
- **Cost:** free.
- **License:** most nflverse data is **CC-BY 4.0**, which allows commercial use as long as we credit the source. **FTN charting data is CC-BY-SA 4.0** (share-alike), which means anything we build from it must be released under the same license, so we avoid it for any paid product. The code is MIT.
- **Caveat:** nflverse states that the underlying NFL data "belong to their respective owners." In practice, credit nflverse on the site. Don't use NFL or team logos, and don't put "NFL" in the site's name or branding. Plain team names and scores are fine as facts. Get real legal advice before charging money.
- **Coverage:** the schedule and results file `games.csv` runs from **1999 to the present** (46 columns). It includes scores, game type (REG and playoffs), location, rest days, roof, surface, temperature, wind, spread, total, moneylines, starting QB IDs and names, coaches, referee, and stadium. Play-by-play also starts in 1999.
- **Gap, now filled:** nflverse has no data before 1999. 1970 to 1998 comes from FiveThirtyEight's `nfl_elo` game file (CC BY 4.0), vendored in `data/sources/` with REG and playoff games. Credit line: "Data: nflverse (CC BY 4.0); 1970–1998 results: FiveThirtyEight (CC BY 4.0)."
- Data dictionaries are at https://nflreadr.nflverse.com/articles/. Datasets include pbp, player stats, team stats, rosters, depth charts, injuries, snap counts, participation, Next Gen Stats, ESPN QBR, draft picks, combine, contracts, trades, and FTN charting.

## Direction

### Elo site

- **Pipeline.** A scheduled GitHub Action runs every Wednesday. It pulls nflverse with `nflreadpy`, reruns Elo, and writes JSON. A static site reads the JSON. Hosting is free on GitHub Pages or Cloudflare Pages, with no server and no database.
- **Model fixes.** Done 2026-10-02: data switch to nflverse, learned HFA, a wider grid tuned on held-out seasons, and a playoffs option (tested; it doesn't help, so it's off). Held-out Brier for 2010-2025 went from 0.2244 to 0.2201. The market scores 0.2104 on the same games.
  - Next: a starting-QB adjustment.
  - Later: show Elo against the betting market on the site.
- **Site features:**
  - Weekly power ladder with movement since last week.
  - Playoff odds from a Monte Carlo season simulation.
  - Luck leaderboard.
  - Team history explorer covering 1970 to now.
  - Picks of the week vs. the betting line.
  - A running scoreboard of Elo vs. the market.
  - Upset tracker and the "any given Sunday" meter.

### Machine learning candidates, ranked by value per effort

1. **Game prediction model.** Combine Elo, efficiency from play-by-play (EPA), QB, rest, and weather, then score it against the market Brier of 0.211.
2. **QB-adjusted and player-aware ratings.** Team strength comes from the players who actually suit up, and ratings follow players when they change teams. This is the paper's own future-work idea.
3. **Explained luck.** Break the luck index into measurable pieces: fumble recovery rate, opponent kicking, one-score games, and turnovers.
4. **Excitement index.** Rank games by how much the win probability swings. nflverse play-by-play already includes win probability.
5. **Draft and contract value.** Predict career outcomes from the combine and draft slot, and estimate value relative to contract.

## Open questions for Walker

- Monetization is not a goal for now (the purpose is a portfolio for NFL teams). If that changes, revisit FTN and the other share-alike data, and get legal advice.

## Parked until Walker picks them back up (2026-10-04)

- **Hosting.** Recommended: Cloudflare Pages, connected to the repo, publishing `site/` with no build step. Walker creates the Cloudflare account and picks the domain; Claude does the rest.
- **GitHub cache purge.** The support request is drafted in `context/pending/github-purge-request.txt` and needs Walker's signed-in Chrome to submit.
- **One master notebook** (Walker, 2026-10-05): a single `.ipynb` that walks through the WHOLE project end to end: data, Elo, features, models, evaluation, the live pipeline, and the simulation. It calls the real package code instead of copying it. This is a wish for later, not the working method.

## Decision log

| Date | Decision |
|---|---|
| 2026-10-02 | This file is the single general context doc. Sub-project context goes in `context/*.md`, each linked below. |
| 2026-10-02 | Old work was moved to `legacy_2025/`. GitHub repo `walk-the-program/NFLELO` (public) is restructured to match. Claude handles all git and GitHub work, committing to main. |
| 2026-10-02 | Data: 1970-1998 from the legacy spreadsheet, 1999+ from nflverse. The overlap check matched 6,967 of 6,967 games. Superseded the same day by the next row. |
| 2026-10-02 | Data licensing: the legacy spreadsheet came from Pro-Football-Reference, whose terms restrict substitute databases and ML use. 1970-1998 now comes from FiveThirtyEight's NFL Elo game file (CC BY 4.0), vendored in `data/sources/`, REG and playoffs. It matches the legacy spreadsheet on 6,140 of 6,140 REG games and nflverse on 6,151 of 6,151 (1999-2022). Ratings and test Brier (0.2201) are unchanged; `DEFAULT_CONFIG` kept after a re-tune. |
| 2026-10-02 | Added the NFLELO demo site: a static page in `site/` fed by JSON from `scripts/export_site.py`, which `build.py` runs. It is not hosted yet; publishing it publicly on GitHub Pages needs Walker's OK. |
| 2026-10-02 | Elo v2 default: K 20, λ 0.40, online HFA (init 65, k_hfa 0.5), expansion start 1300, no playoff updates. It ties the fixed-HFA-65 winner on the tune window and was chosen because fixed HFA can't track the decline in home-field advantage. |
| 2026-10-03 | Removed Pro-Football-Reference-derived game data (`legacy_2025/NFLELO_data.xlsx`, both `comprehensive_nfl_data.csv` exports, and the original root `NFLELO_data.xlsx`) from the public repo and its entire git history, with Walker's approval. The files stay local and are gitignored. Old commit SHAs may stay reachable by direct link on GitHub until GitHub support purges its cache. |
| 2026-10-03 | ML plan written in `context/ml.md`. The ML work happens in its own chat, built together with Walker: game model first, and personnel models later under CC BY-SA. |
| 2026-10-04 | M4 Phase 1 built: `scripts/ml_predict.py` appends the ML model's pre-kickoff predictions to the public ledger `experiments/live/2026.csv`, the site shows Elo, Model and Vegas side by side plus projected wins (`site/data/ml.json`, optional), and `.github/workflows/weekly.yml` runs build, predict and export Wednesdays and Sundays and commits the results. The live record starts with week 5 (Thursday 2026-10-08). Details in `context/ml.md`. |
| 2026-10-03 | ML decisions D2-D6 (see `context/ml.md`): 2020-2025 holdout locked; pure model only, with betting lines as a benchmark and never a feature; walkthrough notebooks; ML packages in `requirements-ml.txt`. M1 and M2 start together. |
| 2026-10-03 | ML M1 and M2 done. The harness reproduces Elo v2 (0.220075) and the market (0.210426) exactly on 2010-2025. On DEV 2006-2019 (3,450 games), Elo minus the market is +0.0072 Brier (95% CI +0.0044 to +0.0101). Every ML agent runs on Opus, at Walker's request. |
| 2026-10-03 | ML M3 passes on DEV 2006-2019 (3,450 games). Chosen model A4s, by the one-SE rule: logistic regression on elo_logit, adj_epa_margin, and qb_delta_diff. Brier 0.2151 vs Elo 0.2178, a difference of -0.0027 (95% CI -0.0043 to -0.0011), ECE 0.011. Market 0.2106. Nearly all of the gain is the QB term, and it needs the confirmed starter: the Wednesday-only version is not significant. Tuned on 2000-2005: lambda 100, half-life 48 weeks, rho 0.25, QB k 100. The home/away QB asymmetry was checked and is noise in the data, not a bug. The holdout run awaits Walker's OK. |
| 2026-10-03 | M3 holdout sign-off, run once with Walker's OK (2020-2025 REG, n 1,615). A4s Brier 0.2195 vs Elo 0.2231, a difference of -0.0035 (95% CI -0.0063 to -0.0007); market 0.2096. A4s beat Elo in every season. ECE 0.028 misses the pre-registered 0.02 bar. A simulation found that a perfectly calibrated model at n 1,615 has a median ECE of 0.024 (5th-95th percentile 0.014 to 0.038), so the bar was too tight for this sample size; Elo (0.040) and the market (0.025) also exceed it. The verdict on the calibration criterion is Walker's. Nothing was retuned. |
| 2026-10-04 | Walker marked M3 passed. The 0.02 ECE bar is recorded as a flawed criterion (too tight for n 1,615), not quietly moved. The model was not recalibrated after seeing the holdout. From M4 on, the calibration check is sample-size-aware: ECE is compared with the distribution a perfectly calibrated model would produce at the same n. M4 started. |
| 2026-10-04 | The name stays NFLELO, with "NFL" kept on purpose (Walker). The site will be public, hosted by Walker alongside his other websites, not on GitHub Pages. Claude owns everything on GitHub, including the support request to purge the cached pre-rewrite commits (17 orphaned commits ending at a4233624; three of them added the removed files). |
| 2026-10-05 | M4 holdout sign-off, run once on pre-registered rules (commit e51e890). The margin model FAILS its primary: MAE beat Elo's spread by 0.059 points (10.126 vs 10.185), but the CI [-0.150, +0.035] includes zero; it passes all three secondaries. The market spread's MAE is 9.764. Playoff odds (tau_rest 4.5) PASS every rule: made-playoffs ECE 0.028 within the null, Brier 0.1234 vs 0.1258 for tau 0 (CI excludes zero). The odds stay hidden on the site (`publish_sim` false) until Walker decides. |
| 2026-10-05 | Walker: publish the playoff odds (`publish_sim` true) and keep the model spread on the site, with methodology copy stating that its edge over Elo's spread is not proven. M4 is complete. |
| 2026-10-05 | No-Elo study (DEV only, descriptive). The best model without Elo (adjusted EPA + QB + rest/neutral) scores 0.2182, a tie with Elo's 0.2178; A4s with Elo scores 0.2151, and adding Elo back is worth 0.0033 (CI excludes zero). Elo's value is mostly in weeks 1-9 (its long memory across seasons); by weeks 10-18 the no-Elo model beats Elo. Boosted trees did not help. This points to a better early-season prior, the M5 lineup work, as the next lever. Two live tests were rewritten to stop depending on the committed site data. |
| 2026-10-05 | Direction set by Walker: NFLELO is a portfolio for NFL teams. Model quality comes first ("the most premium model ever"), and the site pages from the competitor review follow (game pages, team pages, model performance, QB and player pages). There are no betting features; Vegas stays a benchmark only. |
| 2026-10-05 | Standing delegation: Walker stepped away and told Claude to progress to the most advanced version of the model, relying on Claude's own recommendations. Under it, Claude decides open method questions by its stated recommendation and records each one here as "Claude, under delegation". Every pre-registered holdout still runs exactly once, after its rules are committed. Planned order: finish M5 (sign-off, then the shadow model), then M5b (on-field plus-minus from participation, CC BY-SA as approved in D1), then M6 (play outcome model), then M7 (decision support). One ML agent at a time, all on Opus. |
| 2026-10-05 | M5 (box-score player values + pregame lineups) recorded as a negative result on 2012-2019 (n 2,047). No ladder step beats A4s (B1 +0.0002, B2 +0.0004, B3 +0.0008 Brier; every CI includes zero), so the one-SE rule keeps A4s. The lineup features carry real but small information (positive coefficients in every refit, +0.13 correlation with the market's departure from A4s). Found and fixed two leaks: weekly roster statuses are season-end values before 2016, and a position-group lookup used later games. Claude, under delegation: no M5 holdout (nothing to test), M5 does not enter the game model, and the lineup code is kept for a future site lineup view. Next: M5b. |
| 2026-10-05 | Claude, under delegation: M5b method approved (`context/ml-m5b-method.md`). Bayesian RAPM on participation data 2016+, with M5 box-score values as the priors. The primary test is season-ahead play-level MSE against team ratings and box-score values: develop on 2018-2019, then one holdout run on 2020-2025. The ratings are CC BY-SA. |
| 2026-10-05 | Claude, under delegation: M5b dev results accepted. RAPM beats team ratings (-0.0027 MSE, CI excludes zero) and box values (-0.0019) on 2018-2019 season-ahead play EPA (n 55,543 plays). Approved before the holdout: the participation side-repair (0.4% of plays dropped instead of 12.7%; exact on lists that were already complete), tuning box_scale, and scoring REG+POST as on dev. Known exception: M5's lookup_groups later-season fallback; M5b rebuilds priors cut at S-1, and M5 is not in production. |
| 2026-10-05 | M5b holdout run once (commit 21ccd41): primary FAIL. Season-ahead 2020-2025 play EPA MSE: RAPM beats team ratings (-0.0018, CI excludes zero) but loses to box-score values summed over the on-field players (+0.0010, CI excludes zero). Player-level beats team-level; on-field plus-minus adds nothing beyond box credit with public data. The ratings are kept as a descriptive view only. |
| 2026-10-05 | Claude, under delegation: next is M3b, game-model refinement (`context/ml-m3b-method.md`), aimed at the measured weak spots (early season, changed QB) before M6 and M7. Winners run as shadow models in 2026; A4s stays the live headline. |

## Context file index

- `CONTEXT.md`: this file, for the general project.
- `HANDOFF.md`: append-only session handoff log, newest first.
- `context/data-and-elo.md`: the data pipeline, franchise mapping, the Elo v2 config, and evaluation results.
- `context/elo-site.md`: the NFLELO one-page site (`site/`, data exported by `scripts/export_site.py`).
- `context/competitor-nfelo.md`: review of nfeloapp.com (features, model behavior, design), where we're ahead, and the gaps to close.
- `context/ml.md`: the machine learning plan and living record (game model, then player value, play model, and personnel decision support), with milestones M1-M7 and open decisions.
- `context/ml-m3-method.md`: the M3 game-model method (logistic regression on Elo, opponent-adjusted EPA, and a QB adjustment) and decisions M3-D1 to M3-D3 (approved 2026-10-03).
- `context/ml-m4-method.md`: the M4 method (live predictions, prediction ledger, margin model, playoff odds) and decisions M4-D1 to M4-D4 (all decided 2026-10-04).
- `context/m5-data-scope.md`: what free data can support pregame-lineup player values (sources, depth-chart accuracy, injury-report reliability, OL sample sizes, risks).
- `context/ml-m5-method.md`: the M5 method (per-player values from game-level stats, probabilistic pregame lineups, roster features, shadow deployment) and decisions M5-D1 to M5-D4 (approved 2026-10-05).
- `context/ml-m5b-method.md`: M5b on-field player ratings (Bayesian RAPM from participation data, CC BY-SA), tested season-ahead.
- `context/ml-m3b-method.md`: M3b game-model refinement (week-varying weights, Kalman-filter team strength, a better QB term).
