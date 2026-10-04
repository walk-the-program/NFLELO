# NFLELO Machine Learning: Plan and Context

This is the ML sub-project's context file. Read `CONTEXT.md` first, then this file. It is both the plan and the living record: when a decision gets made or a milestone is finished, update the status table and the decision log at the bottom.

Written 2026-10-03. Status: plan drafted, D1-D6 decided, M1 and M2 built, M3 built; holdout run 2026-10-03 (beats Elo, misses the ECE bar). Current scope: Stage A (M1-M4) and M5a, all on CC BY data.

---

## 1. What we're building, and in what order

The end goal Walker described: know who is on the field, predict how many yards a play will gain, and suggest the best personnel or call for the offense or defense. The first step is predicting games accurately.

Those goals are different problems, and each one builds on the one before it:

| Stage | Question it answers | Unit | Data from |
|---|---|---|---|
| A. Game model | Who wins, by how much, and how many total points? | Game | 1999+ (play-by-play) |
| B. Player value | How much is each player, especially the QB, worth to his team? | Player-season and player-game | 1999+ box scores, 2016+ on-field participation |
| C. Play model | Given the situation and the players on the field, what range of yards is likely on this play? | Play | 2016+ (participation) |
| D. Decision support | Which personnel or play type gives the best expected result here? | Decision | Built on C, with causal methods |

Stage A is the first milestone. It is also the foundation, because the data layer, the leak-proof feature builder, and the evaluation harness all get reused in B through D. Stage D is a research problem rather than a modeling exercise. Section 7 explains why.

### What "accurate" means here

We already have two reference points on the same 4,174 regular-season games from 2010 to 2025:

| Model | Brier (lower is better) | Accuracy |
|---|---|---|
| Elo v2 (ours) | 0.2201 | 64.0% |
| Betting market (vig removed) | 0.2104 | 66.6% |
| Always 50/50 | 0.2500 | n/a |

- **Goal 1:** beat Elo v2 by a margin that isn't noise.
- **Goal 2:** close as much of the gap to the market as we can.

Expect honest limits. Public models almost never beat closing betting lines over the long run, and a single season has only about 272 games. The standard error on a Brier difference over 4,000 games was about 0.0014. Improvements smaller than about 0.003 will be hard to prove, so every comparison is paired (same games) and comes with a bootstrap confidence interval.

---

## 2. Working agreement (the team part)

Walker wants to understand this at a deep technical level, not just receive a model. So the work runs like this:

1. **Explain before building.** At each milestone I write up the method: the math, why this approach and not the alternatives, and what could go wrong. Walker reads it and decides any open questions. Then I build.
2. **Two kinds of code.**
   - `notebooks/` holds numbered Jupyter notebooks for exploring and teaching: plots, intuition, and small experiments Walker can run cell by cell.
   - `nflelo/ml/` holds the real, tested package code that the website and weekly build use.
   Ideas start in a notebook and graduate into the package.
3. **Every experiment is logged.** Each training run writes a small JSON record: data version, features, parameters, scores, and the git commit. Results can be compared and reproduced later, and nothing is decided on vibes.
4. **Walker owns the decisions; I own the implementation.** Decisions are listed per milestone. The decision log at the bottom records what was chosen and why.
5. **Concept checkpoints.** Each milestone lists the ideas worth understanding before moving on, so the next stage makes sense.

---

## 3. Data: what exists, what we can use, and the rules

All of this is free. Access is through `nflreadpy`. Licensing matters because the site is public and might make money someday.

| Dataset | Years | What it gives us | License and use |
|---|---|---|---|
| Schedules and results | 1999+ (1970+ with FiveThirtyEight) | Scores, rest, roof, surface, weather, starting QBs, coaches, betting lines | nflverse CC BY 4.0. Use freely, with credit. |
| Play-by-play (`load_pbp`) | 1999+ | Every play: down, distance, field position, clock, score, play type, yards, and the nflfastR model outputs (expected points, EPA, win probability, completion probability, CPOE) | CC BY 4.0. Our main source. |
| Player stats, rosters, players | 1999+ | Weekly box scores, positions, IDs, ages, draft info | CC BY 4.0 |
| Injuries | Roughly 2009+ (to confirm in M1) | Weekly injury reports and game status | CC BY 4.0 |
| Depth charts | 2001+ (to confirm) | Who the team lists as the starter | CC BY 4.0 |
| Participation | **2016+** | **Every player on the field for every play** (by ID), formation, personnel grouping (e.g. "1 RB, 1 TE, 3 WR"), defenders in the box, pass rushers, man/zone, coverage type | **CC BY-SA 4.0** (share-alike). Credit "NFL Next Gen Stats via nflverse" (2022 and earlier) or "FTN Data via nflverse" (2023 on). **Since 2023 it is published only after the postseason.** |
| FTN charting | **2022+**, within 48 hours of each game | Play action, motion, RPO, screen, QB alignment, men in the box, backfield count, catchable ball, and more | **CC BY-SA 4.0**. Credit "FTN Data via nflverse". |
| Next Gen Stats (weekly) | 2016+ | Time to throw, separation, rushing yards over expected | Credit nflverse and NGS. Use as features only. |
| Snap counts, PFR advanced stats | 2012+ / 2018+ | Snap shares, pressures, and so on | **Avoid.** They come from Pro-Football-Reference, whose terms forbid using their data for machine-learning prediction. |
| Player tracking (x/y coordinates) | Only through Big Data Bowl releases | Positions 10 times a second | **Off-limits.** Kaggle competition terms, not open data. |

### Two consequences that shape the plan

- **Share-alike.** Anything we build from participation or FTN data (models, derived datasets, predictions) must be shared under the same CC BY-SA license. Our repo is already public, so this fits as long as the personnel-based models (Stages C and D) stay open. It would block keeping those models closed and selling them. Game-level models (A and B) built only on CC BY data have no such restriction. **Decision for Walker (D1).**
- **In-season personnel data doesn't exist for free.** Participation for the current season arrives after the Super Bowl. Personnel models are trained on past seasons. Using them live means someone enters the personnel by hand, which is what Walker described ("if I were able to get you what players are on the field"). During the season, FTN charting gives partial pre-snap information within 48 hours.

### Scale

- Play-by-play is roughly 45,000 to 50,000 plays per season, so about 1.3 million plays for 1999 to 2025, with about 370 columns.
- Stored as compressed parquet, that's a few hundred MB.
- It fits comfortably on a laptop, and no cloud compute is needed for anything in this plan.

---

## 4. Foundations (Milestones M1 and M2)

These are boring but decisive. Most sports-ML projects fail here through leakage, not because the model was weak.

### M1. Data layer and leak-proof features

- **Cache layer.** `nflelo/ml/data.py` downloads each dataset once per season into `data/raw/ml/` as parquet (gitignored). It records a content hash per file and has a `refresh` command that only re-pulls the current season. nflverse sometimes corrects past play-by-play, so the hashes let us notice when history changes.
- **Schema checks.** Every load is validated: required columns present, types right, play counts per game plausible, team codes mapped through `nflelo/teams.py`.
- **Point-in-time rule.** A feature for a game played at time T may only use information that existed before T. The feature builder takes an "as of" timestamp and filters every input by it.
- **Tests that prove it.** A test builds features for a game, then deliberately corrupts all data on or after kickoff, rebuilds, and asserts the features didn't change. That one test catches most leakage bugs.
- **Concept checkpoint:**
  - What leakage is.
  - Why season-level averages are poison (a week 3 feature built from a season average quietly uses weeks 4 to 18).
  - Why the market line can't be a feature in the "pure" model (see M3).

### M2. Evaluation harness

- **Walk-forward validation.** To predict season S, train only on seasons before S. Within a season, roll forward week by week so week 10 predictions use everything through week 9. This copies how the model would actually be used.
- **Three time windows:**
  - **Development:** 2006 to 2019. Tune and compare here as often as we like.
  - **Holdout:** 2020 to 2025. Touch it only at milestone sign-off, and decide in advance what we'll measure. If we keep peeking, it stops being a test.
  - **Live:** the 2026 season, predicted every Wednesday before games and scored afterward. This is the real exam, and the site can show it publicly.
- **Metrics:**
  - Win probability: Brier (primary), log loss, accuracy, and calibration (a reliability curve and expected calibration error).
  - Margin and total: mean absolute error, and CRPS for the full distribution.
  - All on the same games for every model.
- **Comparisons:** paired bootstrap confidence intervals on metric differences (resample games, recompute, take the 2.5th to 97.5th percentiles). An improvement counts only if its interval excludes zero.
- **Baselines inside the harness:**
  - 50/50
  - Home team always wins
  - Elo v2, which must reproduce 0.2201 on 2010 to 2025
  - The market, which must reproduce 0.2104
  If the harness can't reproduce known numbers, nothing else it says is trustworthy.
- **Market lines are never features** (D3). The harness keeps them in a separate benchmark column so they can't slip into a feature matrix.
- **Experiment registry:** `experiments/runs/*.json` plus a small script that prints a leaderboard.
- **Concept checkpoint:**
  - Proper scoring rules (why Brier and log loss reward honest probabilities and accuracy doesn't).
  - Calibration versus discrimination.
  - Why random cross-validation is wrong for time series.

---

## 5. Stage A: the game model (M3 and M4)

### Targets

1. Home win probability, the headline number.
2. Point margin, home minus away. Win probability and a spread both come out of a margin distribution, so this is the more useful target.
3. Total points, which lets us compare against the over/under.

### Features (all computed as of kickoff)

- **Ratings we already have:** Elo v2 pre-game ratings and the learned home-field edge.
- **Efficiency from play-by-play.** These drive most of the gain in the public literature:
  - Offensive and defensive EPA per play, split into pass and rush.
  - Success rate.
  - Explosive play rate.
  - Turnover-worthy plays.
  Garbage time is excluded (for example, plays where win probability is below 5% or above 95%).
- **Opponent adjustment.** Raw EPA rewards easy schedules. We fit a ridge regression where each play's EPA = offense strength + defense strength + home edge, weighted toward recent games. The fitted strengths are the adjusted ratings. This is the first "real" ML component and a good one to understand deeply.
- **Recency and early-season priors.** Weeks 1 to 3 have almost no current-season data. Each team's rating starts from last season's value pulled toward the league average and updates as games arrive (Bayesian shrinkage). We tune how fast it moves.
- **Quarterback:**
  - The starting QB comes from the schedule file.
  - Each QB gets a rolling EPA-plus-CPOE composite, shrunk toward a replacement-level prior for QBs with few dropbacks.
  - The feature is the starter's value minus the team's typical QB value, which captures injuries and benchings. This is the QB adjustment from the Elo roadmap, done properly.
- **Context:**
  - Rest days and the difference in rest.
  - Bye week.
  - Travel distance and time zones crossed (needs stadium coordinates, compiled once).
  - Divisional game, roof and weather, surface, neutral site, playoff game.
- **Injuries (later in M4):** players ruled out, weighted by position importance.

### Models, in order of complexity

1. **Logistic regression on a handful of features.** Interpretable, so every coefficient can be read. If it already beats Elo, we learn which features matter.
2. **Gradient-boosted trees (LightGBM),** which catch interactions and nonlinear effects. Monotonic constraints keep them sensible (better offense should never lower win probability).
3. **Margin model.** Predict the expected margin, then model its spread (NFL margins have a standard deviation of roughly 13 to 14 points). Win probability is P(margin > 0). The margin distribution has spikes at 3 and 7, which a plain normal curve misses, so we compare a normal against a key-number-aware discrete distribution.
4. **Bayesian state-space model.** This is the sophisticated comparison: team strengths drift week to week as a random walk (the Glickman-Stern style). It is principled about uncertainty and naturally handles early-season priors. It's slower, but valuable for understanding.
5. **Stacking and calibration.** Combine the best models (including Elo) with a simple meta-model fit on out-of-fold predictions, then apply isotonic or Platt calibration if the reliability curve shows bias.

### Pure model only (D3)

No model uses betting-market inputs (spread, total, or moneylines) as features. The market is a benchmark only: we score against it and can show it on the site, but our predictions must come from football data. This is the honest test of whether our features know something. A market-aware model was considered and declined on 2026-10-03.

The full M3 method, with its math, experiment ladder, pre-registered holdout run, and decisions M3-D1 to M3-D3, is in `context/ml-m3-method.md`.

### Milestone acceptance

- **M3 (first ML game model):** on the development window, the pure model beats Elo v2 on Brier with a paired 95% interval that excludes zero, and its expected calibration error is below 0.02. Then it gets one sign-off run on the holdout.
- **M4 (production game model):**
  - Margin and total models.
  - Weekly predictions wired into `scripts/build.py` and shown on the site next to Elo and Vegas.
  - Monte Carlo season simulation for playoff odds.
  - A live 2026 scorecard on the site.

### Concept checkpoints

- Regularization (ridge) and why opponent adjustment is a regression.
- Bias versus variance.
- Gradient boosting at a whiteboard level.
- Calibration methods.
- Why the margin distribution has key numbers.
- What a posterior distribution is (for the Bayesian model).

---

## 6. Stages B and C: players and plays (M5 and M6)

### M5. Player value

Split by license (decision 2026-10-03). **M5a** uses only CC BY data (play-by-play, player stats): the QB composite and box-score player value. That's everything the game model needs. **M5b** is the on-field adjusted plus-minus, which needs CC BY-SA participation data, so it comes after M5a.


- **QB first:** the composite from Stage A, extended with era adjustment and aging curves.
- **Adjusted plus-minus on plays (2016+).** This is the bridge to personnel.
  - Each play becomes a row, with +1 for each offensive player on the field and -1 for each defender (from participation), plus situation controls. The target is the play's EPA.
  - A heavily regularized ridge regression (thousands of players, collinear lineups) estimates each player's contribution.
  - This is the same idea as basketball's RAPM, and it produces the player ratings Walker's paper imagined: ratings that follow players to new teams.
- **Team strength from personnel:** the game model gets a feature built from the sum of active players' values. If it improves Stage A, that's the proof player ratings mean something.
- **Concept checkpoint:**
  - Collinearity, and why ridge (rather than plain least squares) is required.
  - Shrinkage toward position priors.
  - Sample-size problems for low-snap players.

### M6. Play outcome model: "how many yards will this play gain?"

- **Target:** yards gained, as a full distribution rather than a single number. Yards are heavy-tailed (most runs gain 0 to 6, a few break for 40+), and the mean alone is misleading. We predict probabilities over yard bins, from -10 or worse up to 99. This is the approach that won the NFL's own Big Data Bowl rushing contest, scored with CRPS. From the distribution we get expected yards, the chance of a first down, the chance of an explosive play, and the chance of a loss.
- **Inputs (pre-snap only):**
  - **Situation:** down, distance, yard line, score, time, timeouts, home/away.
  - **Formation and personnel:** grouping, men in the box, number of rushers on pass plays (careful: that is only known after the snap), shotgun versus under center.
  - **The specific players on the field:** player embeddings, vectors learned for each player, so the model can tell a great left tackle from a backup.
  - **Opponent defensive strength.**
  - **Play type:** run or pass, as an input when we evaluate a chosen call, or predicted when we don't know it.
- **Models:**
  1. Baseline: the historical distribution for the same down, distance, and field zone.
  2. Gradient boosting over bins.
  3. A neural network (PyTorch, running fine on a Mac) with player embeddings and a set-based layer for the 11 offensive and 11 defensive players, so player order doesn't matter.
- **Acceptance:** beat the down-distance baseline and a boosted model without player identities on CRPS, on held-out seasons. If player identities don't help, that's a real finding worth knowing.
- **Concept checkpoints:**
  - Predicting distributions versus point estimates.
  - CRPS.
  - Embeddings.
  - Permutation-invariant (set) networks.
  - Why "rushers" and "coverage" can't be inputs for a pre-snap prediction even though they're in the data.

---

## 7. Stage D: decision support (M7). Read this before getting excited.

"Which personnel should the offense use here?" sounds like "run the play model for every option and pick the best." That approach is **wrong in a specific, important way.**

- **Selection bias.** Teams choose personnel because of the situation and the opponent. Heavy personnel shows up on 3rd-and-1 at the goal line, where yards per play are naturally low. A naive model learns "heavy personnel = fewer yards" and recommends against it, confusing the situation with the choice.
- **The defense reacts.** If an offense always used the "best" grouping, the defense would adjust. The best real-world answer is a mix (game theory), not a single fixed choice.

So M7 is a causal-inference problem, and the plan treats it that way:

1. **Define a small action set:** personnel grouping (11, 12, 21, 13, 10, and so on) × run/pass. Recommendations stay within choices teams actually use in similar situations, because we can't learn about choices nobody makes.
2. **Estimate the effect of each action using methods built for observational data:**
   - A propensity model of how likely each choice is, given the situation.
   - Doubly robust estimators that combine that propensity model with the play model.
   - Matched comparisons of similar situations.
3. **Report expected outcome with uncertainty** for each action, and flag when the data is too thin to say.
4. **Validate with off-policy evaluation:** estimate how a recommended policy would have done on held-out seasons, using the same doubly robust tools.
5. **Look at defensive response:** do defenses that see a grouping often respond differently? This is a stretch goal.

**Acceptance:** recommendations that hold up in held-out off-policy evaluation, and a written note on where they don't. This milestone is genuinely research, and honest "we can't tell" answers are part of the deliverable.

**Concept checkpoints:**
- Correlation versus causation.
- Confounding.
- Propensity scores.
- Doubly robust estimation.
- Off-policy evaluation.
- Mixed strategies.

---

## 8. Code layout (planned)

```
nflelo/ml/
  data.py          # cached nflverse loaders, schema checks, data hashes
  asof.py          # point-in-time filtering helpers
  features/        # team_efficiency.py, opponent_adjust.py, qb.py, context.py, ...
  models/          # logistic.py, gbm.py, margin.py, bayes_state_space.py, stack.py, play_model.py
  eval/            # walk_forward.py, metrics.py, bootstrap.py, calibration.py
  registry.py      # experiment logging and leaderboard
notebooks/         # 01_data_tour.ipynb, 02_leakage_demo.ipynb, 03_opponent_adjusted_epa.ipynb, ...
experiments/runs/  # one JSON per run (committed)
tests/ml/          # leakage test, metric tests, feature unit tests
```

**Libraries:** all free and local.
- polars or pandas, pyarrow, scikit-learn, LightGBM, and statsmodels.
- PyTorch for M6.
- PyMC or NumPyro for the Bayesian model.
- Jupyter.

No paid APIs, no cloud.

---

## 9. Milestone status

| ID | Milestone | Depends on | Status |
|---|---|---|---|
| M1 | Data layer and leak-proof features | none | Built 2026-10-03, awaiting Walker's review |
| M2 | Evaluation harness, reproducing Elo 0.2201 and market 0.2104 | M1 | Built 2026-10-03 (reproduces both exactly), awaiting Walker's review |
| M3 | First ML game model beats Elo (paired CI excludes 0) | M2 | **Passed 2026-10-04.** Holdout: A4s beats Elo by 0.0035 Brier (CI [-0.0063, -0.0007]). The ECE of 0.028 is within the noise range for n = 1,615; the 0.02 bar is recorded as flawed. |
| M4 | Production game model on the site, with live 2026 scorecard and playoff odds | M3 | Phase 1 built 2026-10-04 (weekly predictions, ledger, site, Action), awaiting Walker's review. Phase 2 not started. |
| M5a | Player value from CC BY data: QB composite, box-score player value (no participation data) | M2 | Not started |
| M5b | On-field adjusted plus-minus (uses CC BY-SA participation data, 2016+) | M5a | Not started |
| M6 | Play outcome distribution model (CC BY-SA) | M1, M5b | Not started |
| M7 | Personnel decision support (causal) | M6 | Not started |

### M1/M2 build notes (2026-10-03)

**Where things are.** `nflelo/ml/data.py` (cache, manifest, schema checks; `python -m nflelo.ml.data status|fetch|refresh`), `asof.py` (as-of rule and `leakage_check`), `features/team_efficiency.py` (plus `features/leaky_demo.py`, the deliberate leak used as a test control), `eval/` (`windows`, `walk_forward`, `metrics`, `bootstrap`, `calibration`, `baselines`), and `registry.py`. Scripts: `scripts/ml_baselines.py` and `scripts/ml_leaderboard.py --window dev|holdout|reproduction`. Walkthrough: `notebooks/01_data_tour_and_leakage.ipynb`. Tests in `tests/ml/` run offline on synthetic frames.

**Data.** Play-by-play for 1999-2025 is cached: 1,279,628 rows (about 45,000 to 50,000 per season; 2021 onward is higher with 17 games), 306 MB on disk in `data/raw/ml/` (gitignored). Schedules are cached for 1999-2026. Every season passes the schema checks: required columns and types, 99-171 scrimmage plays per game, and every team code maps through `nflelo/teams.py`. Unmapped codes are reported as errors and never dropped. Gaps: one played 1999 game and two 2000 games have no play-by-play. Every 1999 game is missing its kickoff time.

**As-of rule.** A game's `as_of` is the earliest REG or POST kickoff in its (season, week), in Eastern time. A missing kickoff time counts as 00:00 ET on game day. Features use only games that kicked off strictly before `as_of`, so Thursday's result never reaches that week's Sunday games. The leakage test scrambles everything at or after `as_of` (plays, outcomes, and scores, including later games in the same week) and requires identical features. A full-season-average builder must fail the same test, and it does.

**Features.** Offensive and defensive EPA per play (all, pass, and rush), success rate, and play counts per team, pooled as (this season to date + w × last season), with w = 0.5 by default. The features count real snaps only: `play_type` pass or run, an EPA value, and no two-point tries. Pass versus rush follows nflfastR's `pass` and `rush` flags, so scrambles count as dropbacks. Plays with win probability below 0.05 or above 0.95 are dropped. Market columns are removed when the data loads, and a test fails if any feature name matches spread, total_line, moneyline, or vegas.

**Harness results.**
- Reproduction (REG 2010-2025, `allow_holdout=True`, labelled "reproduction"): Elo v2 0.220075 on n = 4,175, and on the 4,174 games with moneylines, Elo 0.220116 and market 0.210426. All three match `compare_on_test` exactly (difference 0).
- DEV baselines (REG 2006-2019, the 3,450 of 3,584 games with moneylines, identical for all four):

  | Model | Brier | Log loss | Accuracy | ECE |
  |---|---|---|---|---|
  | Market (benchmark) | 0.2106 | 0.6096 | 0.663 | 0.018 |
  | Elo v2 | 0.2178 | 0.6255 | 0.641 | 0.015 |
  | Home-win rate (trailing 10 seasons) | 0.2451 | 0.6847 | 0.566 | 0.007 |
  | 50/50 | 0.2493 | 0.6931 | 0.434 | 0.066 |

  Elo minus market Brier is +0.0072, with a 95% paired bootstrap CI of [+0.0044, +0.0101] (2,000 reps, seed 20261003).

**Caveats.**
- Elo v2 was tuned on 1980-2009, so 2006-2009 are in-sample for Elo, which makes its DEV score slightly optimistic.
- Elo updates game by game, so its Sunday predictions do see that week's Thursday result. The ML features use the stricter weekly as-of rule.
- nflfastR's EP and WP models were trained on data that includes seasons after the ones we predict. This is an accepted limitation shared by all public EPA work.
- The 50/50 baseline's accuracy (0.434) is just the away-win rate, because a 0.5 prediction counts as picking the away team.
- Registry runs record the git commit and a dirty flag. The first runs were logged before this code was committed.

---

### M3 build notes (2026-10-03)

**Where things are.** `nflelo/ml/features/opponent_adjust.py` (weighted ridge ratings per (season, week), cached in `data/raw/ml/ratings/`), `features/qb.py` (QB value, replacement prior, `qb_delta`), `features/context.py` (`rest_diff`, `neutral`, and the `elo_logit` feature), `models/logistic.py` (walk-forward fit, ties as two half rows, season-decay weights). Scripts: `scripts/ml_tune_ratings.py` and `scripts/ml_m3.py`. Walkthroughs: `notebooks/02_opponent_adjusted_epa.ipynb` and `notebooks/03_m3_results.ipynb`. Tests: `tests/ml/test_m3_features.py` and `tests/ml/test_m3_models.py`.

**Small edits to existing modules.** `asof.corrupt_from` now also shuffles passer IDs on late plays, and takes `scramble_starters` (default off, because M3-D1 treats the starter's identity as known); `leakage_check` passes it through. `eval/windows.py` gained a `tune` label (2000-2005) for the tuning run. The synthetic test frames gained `location`, rest days, starting-QB IDs, `qb_dropback`, `passer_id`, and `play_id`, drawn from separate random streams so the M1 values are unchanged.

**Implementation choices worth knowing.**
- The ridge fit runs on one row per (game, offense) with weight w x plays. That gives exactly the same answer as the play-level fit (a test checks it) and takes about 1 ms per week.
- `weeks_ago` counts game weeks on a timeline that skips the off-season; the most recent week has weight 1. Only this season and last season enter a team fit. QB values use the same weights over the QB's whole recorded career.
- The QB is `passer_id` on `qb_dropback` plays (passes, sacks, scrambles). Replacement level for season S pools the first 100 career dropbacks of QBs who debuted after 1999, from seasons before S.
- `elo_logit` uses Elo v2's pre-game ratings and the online HFA frozen before the week's first game.
- The A1 raw margin is (home off + away def allowed) minus (away off + home def allowed), from the M1 features.

**Tuning (M3-D2), 2000-2005 only.** 400 settings (lambda 25-3200, half-life 2-1000 weeks, rho 0.1-1). Chosen: **lambda 100, half-life 48 weeks, rho 0.25**, all interior to the grid (half-life is flat from about 12 weeks up). Next-week EPA-margin MSE 0.1304 vs 0.1482 for predicting zero (correlation 0.35, 1,518 games). QB **k = 100** pseudo-dropbacks (grid 25-3200; 3,036 starter-games). Lambda 100 implies a prior spread of team ratings of about 0.14 EPA per play. Run time 28 s.

**Ladder, DEV 2006-2019, 3,450 REG games with moneylines** (paired bootstrap vs Elo, 2,000 reps, seed 20261003):

| Step | Model | Brier | Log loss | Acc | ECE | vs Elo [95% CI] |
|---|---|---|---|---|---|---|
| | Elo v2 | 0.2178 | 0.6255 | 0.641 | 0.015 | |
| A0 | elo_logit | 0.2178 | 0.6257 | 0.645 | 0.018 | +0.0000 [-0.0003, +0.0003] |
| A1 | + raw EPA margin | 0.2180 | 0.6260 | 0.645 | 0.017 | +0.0001 [-0.0002, +0.0005] |
| A2 | + adjusted EPA margin | 0.2177 | 0.6254 | 0.644 | 0.012 | -0.0002 [-0.0008, +0.0005] |
| A3 | + adjusted pass and rush margins | 0.2180 | 0.6262 | 0.639 | 0.016 | +0.0002 [-0.0006, +0.0010] |
| A4 | A2 + QB deltas, home and away (actual starter) | 0.2150 | 0.6195 | 0.647 | 0.012 | -0.0028 [-0.0045, -0.0011] |
| **A4s** | **A2 + qb_delta_diff (home minus away; actual starter)** | **0.2151** | **0.6197** | **0.647** | **0.011** | **-0.0027 [-0.0043, -0.0011]** |
| A4b | A2 + QB deltas, home and away (last game's starter) | 0.2171 | 0.6244 | 0.647 | 0.011 | -0.0007 [-0.0018, +0.0005] |
| A4bs | A2 + qb_delta_diff (last game's starter) | 0.2172 | 0.6244 | 0.643 | 0.012 | -0.0006 [-0.0016, +0.0004] |
| A5 | A4 + rest_diff + neutral | 0.2150 | 0.6194 | 0.650 | 0.009 | -0.0028 [-0.0046, -0.0011] |
| A6 | boosted trees on A5 features | 0.2172 | 0.6253 | 0.652 | 0.020 | -0.0006 [-0.0032, +0.0020] |
| | Market (benchmark) | 0.2106 | 0.6096 | 0.663 | 0.018 | |

- A0 reproduces Elo (0.21783 vs 0.21782). No step scored below 0.213, the leak alarm.
- One-SE rule (M3-D3): the best is A5 (0.21498). Within one SE: A4 (+0.00001, SE 0.0003), A4s (+0.00010, SE 0.0003), and A5. **A4s (3 features) is chosen**, and it passes the M3 bar on DEV. (Before A4s was added, the pick was A4.)
- A4s coefficients, 2019 refit (trained 2001-2018; range over the 14 refits): intercept +0.048 [+0.021, +0.092], elo_logit +0.84 [+0.62, +0.87], adj_epa_margin +1.09 [+0.87, +2.21], qb_delta_diff +3.61 [+2.74, +3.76]. Every refit has the expected sign. In words: a backup 0.15 EPA per dropback worse than the QB play the team's rating remembers moves the game 0.54 log-odds against his team (60% to about 46% at home).
- Home/away QB asymmetry in A4 (2019 fit: home +4.74, away -2.31) is **data, not a bug**. On DEV the two deltas have the same distribution (mean 0.0015 vs 0.0019, SD 0.046 vs 0.043, nonzero 91% vs 90%, starter changed 10.5% vs 10.2%). The code computes both sides identically: each side's own franchise's dropbacks feed its "remembered" value, there are no home/away swaps, and neutral sites don't enter the QB term. The schedule's `home_qb_id` is the home team's main passer in 97.0% of DEV games and the away team's in 0.0% (away: 97.7% and 0.0%). The asymmetry is in the outcomes: in the 128 games with a home delta below -0.1, the home team won 34.8% against 58.9% predicted without the QB term (-24 points); in the 127 with an away delta below -0.1, it won 66.9% against 57.8% (+9 points). The market moves about the same both ways (-12 and +10 points), so this is most likely sampling noise. A4s minus A4 is +0.00009 [-0.00016, +0.00034].
- What the QB news is worth: A4s minus A4bs is -0.0021 [-0.0033, -0.0010] (A4 minus A4b: -0.0021 [-0.0035, -0.0008]). The gain sits in the 664 games where the starter differs from the team's previous game (Brier 0.226 to 0.215); when the starter is unchanged, the two score about the same. With Wednesday information only (A4b, A4bs), the gain over Elo is not significant.
- Team efficiency adds almost nothing once Elo is in the model (A2 vs A0: -0.0002). Elo and the adjusted margin correlate 0.90.

**Timing.** Tuning 28 s. Full ladder (load, all features, 10 walk-forward steps with bootstraps) about 19 s; ratings are cached, but a cold build adds under 1 s. One weekly update (2019 week 10: Elo, ratings, QB values, predict 13 games) 0.6 s; the once-a-season refit is a few milliseconds.

**Caveats and open items.**
- Holdout (2020-2025): run once on 2026-10-03; see "M3 holdout sign-off" below.
- A6 used scikit-learn's `HistGradientBoostingClassifier` (same monotonic constraints, fixed untuned settings) as a stand-in for LightGBM, which can't load here because its `libomp.dylib` is missing. The stand-in is accepted for M3; LightGBM is not being pursued.
- Pass and rush ratings (A3) reuse the combined fit's knobs; they were not tuned separately.
- The registry runs were logged from an uncommitted working tree (`git.dirty` true). Re-log after committing, as was done for M2.
- The `starter="last"` rule uses the previous game's starter even across the off-season, so week 1 counts as a "changed starter" whenever last season's finale had a different QB.

### M3 holdout sign-off (2026-10-03)

Walker approved the run. `python scripts/ml_m3.py --signoff` ran once, following `context/ml-m3-method.md` section 6. The model was frozen: A4s (`elo_logit`, `adj_epa_margin`, `qb_delta_diff`), with the knobs tuned on 2000-2005 (lambda 100, half-life 48, rho 0.25, QB k 100), season-decay training weights, and a walk-forward fit for each season S in 2020-2025 on REG 2001..S-1. Starter identity was treated as a pre-game fact (M3-D1). Before the real run, `--signoff-dry-run` put the same code path through DEV and reproduced the ladder's A4s and A4bs numbers exactly. Runs are `experiments/runs/20261003T215950Z_m3_signoff_{A4s,elo_v2,market,A4bs}.json`, labelled holdout and flagged `holdout: true`.

Games: REG 2020-2025 with moneylines, n = 1,615 (every REG game in the window has one), games hash `1b27abff81bddb2c`, identical for all rows.

| Model | Brier | Log loss | Acc | ECE |
|---|---|---|---|---|
| **A4s (primary)** | **0.2195** | **0.6310** | **0.643** | **0.028** |
| Elo v2 | 0.2231 | 0.6392 | 0.637 | 0.040 |
| Market (benchmark) | 0.2096 | 0.6081 | 0.669 | 0.025 |
| A4bs (Wednesday-only, secondary) | 0.2216 | 0.6360 | 0.637 | 0.031 |

Paired bootstrap 95% CIs (2,000 reps, seed 20261003):
- A4s minus Elo: Brier **-0.0035 [-0.0063, -0.0007]**, log loss -0.0082 [-0.0147, -0.0016]. Both exclude zero.
- A4s minus market: Brier +0.0099 [+0.0061, +0.0137], log loss +0.0229 [+0.0144, +0.0314].
- A4s minus A4bs: Brier -0.0020 [-0.0042, +0.0002]. Includes zero (on DEV it excluded zero).

Per season, Brier (descriptive):

| Season | n | A4s | Elo | A4s - Elo | Market |
|---|---|---|---|---|---|
| 2020 | 256 | 0.2125 | 0.2155 | -0.0030 | 0.2011 |
| 2021 | 272 | 0.2244 | 0.2308 | -0.0064 | 0.2163 |
| 2022 | 271 | 0.2200 | 0.2232 | -0.0032 | 0.2095 |
| 2023 | 272 | 0.2334 | 0.2337 | -0.0003 | 0.2187 |
| 2024 | 272 | 0.2072 | 0.2124 | -0.0052 | 0.2002 |
| 2025 | 272 | 0.2192 | 0.2223 | -0.0030 | 0.2116 |

Starter split, A4s minus Elo Brier (descriptive):
- Starter changed from the team's previous game (374 games): 0.2190 vs 0.2266, a diff of -0.0076 [-0.0156, +0.0011]. A4bs 0.2269, market 0.1994.
- Unchanged (1,241 games): 0.2197 vs 0.2220, a diff of -0.0023 [-0.0048, +0.0002]. A4bs 0.2200, market 0.2127.

The A4s coefficients for the 2025 fit were intercept +0.011, elo_logit +0.79, adj_epa_margin +1.28, and qb_delta_diff +3.72. All signs are as expected. Every coefficient is inside its DEV range except the intercept, which is below the DEV minimum of +0.021. That fits a home-field edge that kept shrinking.

**Verdict as pre-registered.** A4s beats Elo out of sample, in all six seasons and over the window, with an interval that excludes zero. Its ECE (0.028) is above the 0.02 bar, so the strict M3 bar is **not met**. Context, not a re-scoring:
- Elo (0.040) and the market (0.025) are also above 0.02 on this window.
- A simulation in notebook 03, section 6, sets up a perfectly calibrated model at n = 1,615, with its predictions resampled from A4s on DEV. Its ECE has a median of 0.024 and a 5th-95th percentile range of 0.014 to 0.038. At n = 3,450 the median is 0.017. So 0.028 is no clear evidence of miscalibration, and the 0.02 bar is too tight for a window this size.
- A fixed-n calibration test, such as a reliability-curve slope and intercept or a calibration bootstrap, belongs in the M4 plan.

Nothing was retuned.

Other observations:
- The market is about as good as on DEV (0.2096 vs 0.2106), while A4s and Elo both scored worse (0.2195 and 0.2231), so the gap to the market widened.
- The QB news gain (A4s vs A4bs) is about the same size as on DEV (-0.0020), but no longer significant. Starter changes were 23% of games, against 19% on DEV.

---

### M4 Phase 1 build notes (2026-10-04)

**Where things are.** `scripts/ml_predict.py` (weekly run), `nflelo/ml/live.py` (ledger rules, selection, scoring, projected wins; pandas only, so the plain site build can import it), `nflelo/ml/live_features.py` (features as of now), `scripts/export_site.py` (`build_ml`, `export_ml`: writes `site/data/ml.json` only when the ledger exists), `site/app.js` and `site/styles.css`, `.github/workflows/weekly.yml`, `tests/ml/test_live.py`. Committed outputs live in `experiments/live/`: `2026.csv` (the ledger), `model_2026.json` (the season fit) and `latest_2026.json` (the latest run's features and starters for every unplayed game; no market data).

**Week numbering.** nflverse's 2026 week 4 was played Thursday 2026-10-01 to Monday 2026-10-05, and week 5 starts Thursday 2026-10-08. The M4 write-up called that Thursday "week 6". The code follows nflverse: `live.LIVE_FROM_WEEK = {2026: 5}`, so the live record starts with the games of Thursday 2026-10-08, as intended, and the site says "Week 5".

**Season model.** A4s refit on REG 2001-2025 (6,471 games) through `ml_m3.build_frame`, the sign-off code path. Before the fit, the script refits 2025 the same way and compares it to the sign-off run's 2025 coefficients: identical (max |diff| 0). 2026 fit (`A4s-2026-c6b394e8`): intercept +0.009, elo_logit +0.80, adj_epa_margin +1.15, qb_delta_diff +3.70. The model is fit once per season and reused while its spec matches; `--refit` forces a new fit.

**Features as of now.** For a game in the week already under way, features are built exactly as in training (that week's as_of). Every later week uses the state at the next week's as_of, which is everything played so far; future weeks don't age the data. A check on 2025 weeks 1, 10 and 15 (pretending "now" is an hour before each week) reproduced the training frame's three features to 1e-15. Elo uses the current ratings, the HFA frozen before the feature week's first game when it has been played (else the current HFA), and the off-season regression when the season hasn't started. QB: nflverse's listed starter, else the team's starter in its most recent game before now (`qb_source` nflverse, last_starter, or mixed).

**Ledger.** Columns as in the spec plus `data_hash` (manifest hashes of 1999-2026 pbp and schedules, `data/games.csv` and `outputs/elo_games.csv`). `append_rows` refuses the whole run if any row's `run_at_utc` is at or after kickoff, if a run isn't later than the last one, or if the header differs; it only appends. `spread_model` is empty in Phase 1. Elo's numbers are the exporter's own (`elo_pick_table` with `current_elo`), and the market's are added after the model predicts. The script stops if `outputs/` is missing any game nflverse shows as final, so Elo is always current; `--now` refuses to write to the real ledger.

**First run** (2026-10-04 18:55 UTC, Sunday of week 4): 222 unplayed REG games, 214 logged, 8 week-4 games already under way excluded. QB sides over all unplayed games: 58 nflverse, 386 last_starter (nflverse lists starters only for the next week). nflverse had already filled week 5's starters by Sunday afternoon of week 4 (15 of 15 games), and they are not just last week's starter: 1 of 30 sides differs (Washington: Jayden Daniels listed, Marcus Mariota started week 4). So the Sunday refresh is worth keeping; whether the fields change again between Wednesday and Sunday is still to be measured from the ledger's `home_qb_id`/`away_qb_id` columns.

**Scoring.** `live.live_record` scores the last pre-kickoff row per game and, separately, the last Wednesday (Eastern) row, from week 5 on: Brier and accuracy for model and Elo on every scored game, plus model, Elo and market on the games with a moneyline, and Elo against the spread (side = sign of Elo spread minus market spread; push when the margin equals the line). Week 4 rows are in the ledger but outside the live record.

**Site.** This week shows Elo, Model and Vegas side by side (home win chance and line; the model's line says "None"). The three biggest model-vs-Vegas gaps (percentage points) are tagged and the headline follows them. A "QB change" tag shows when |qb_delta_diff| >= 0.08 EPA per dropback (`live.QB_CHANGE_THRESHOLD`; about 7 points of win chance in an even game, about 1.8 SD on DEV), naming the side with the larger delta. A new "03 / Rest of season" section (inserted by JS only when `ml.json` exists, kickers renumbered) shows projected wins by division, model bar plus Elo tick, each row expanding to its remaining games. The scorecard has a live 2026 block (empty-state copy until week 5 is scored) and the methodology a Model paragraph with the ledger link and last-run time. Model color: `--series-3`, `#7b4fc9` light and `#a06ae8` dark, checked with the dataviz validator against Elo and Vegas (all pairs pass CVD and normal-vision; Elo's known chroma-floor failure is unchanged).

**Weekly Action.** Wednesday 14:00 UTC and Sunday 12:00 UTC plus manual runs: build.py, ml_predict.py, export_site.py, then commit `experiments/live`, `site/data` and `outputs` and push. Cold run: about 28 seasons of play-by-play (about 13 MB each, roughly 360 MB) plus the ML requirements; a few minutes. `data/games.csv` is rebuilt each run but not committed.

## 10. Open decisions for Walker (start the ML chat here)

- **D1. Share-alike data: DECIDED 2026-10-03.** Accepted. Personnel-based work (M5b, M6, M7) will be released under CC BY-SA. Stages A and M5a use only CC BY data, so they stay unrestricted and are done first. This can be revisited later (for example, by licensing play-level data directly) without touching A or M5a.
- **D2. Holdout: DECIDED 2026-10-03.** 2020 to 2025 is locked. Tuning uses 2006 to 2019 only. The holdout is scored once per milestone sign-off, with the metrics fixed in advance.
- **D3. Pure versus market-aware: DECIDED 2026-10-03.** Pure only. Betting lines are a benchmark, never a feature.
- **D4. Notebook depth: DECIDED 2026-10-03.** Walkthroughs: finished notebooks Walker reads and runs cell by cell.
- **D5. Python environment: DECIDED 2026-10-03.** Same `.venv`, with ML packages in a separate `requirements-ml.txt`. The weekly site build installs only `requirements.txt`.
- **D6. First-week scope: DECIDED 2026-10-03.** Start M1 and M2 together, and finish with a notebook that tours the play-by-play data and demonstrates a leakage bug on purpose.

---

## 11. Decision log

| Date | Decision |
|---|---|
| 2026-10-03 | Plan written. Order: game model, then player value, then play model, then decision support. Avoid Pro-Football-Reference-derived nflverse datasets (snap counts, PFR advanced stats) for all ML because of their terms. Big Data Bowl tracking data is out of scope. |
| 2026-10-03 | D1 accepted: personnel-based models (M5b, M6, M7) will be CC BY-SA. The first scope is the game model (M1-M4) plus M5a (QB and box-score player value), which use only CC BY data. Player value is split into M5a (CC BY) and M5b (participation, CC BY-SA). |
| 2026-10-03 | D2-D6 decided. D2: 2020-2025 locked as holdout, tuning on 2006-2019. D3: pure model only; betting lines are a benchmark, never a feature (the market-aware model was dropped). D4: notebooks are walkthroughs. D5: separate `requirements-ml.txt`. D6: M1 and M2 built together, ending with a data-tour and leakage-demo notebook. |
| 2026-10-03 | M3 method approved (`context/ml-m3-method.md`). M3-D1: the starting QB's identity counts as a pre-game fact, and the Wednesday-only version (A4b) is always reported next to it. M3-D2: rating knobs (lambda, half-life, rho, QB k) are tuned on 2000-2005 with a next-week-EPA target. M3-D3: the one-standard-error rule picks the simplest model within noise. |
| 2026-10-04 | M3 marked passed by Walker. The holdout ECE of 0.028 missed the 0.02 bar, but a simulation shows that is within the normal range for a perfectly calibrated model at n 1,615 (median 0.024, 5th-95th percentile 0.014 to 0.038). The bar is recorded as flawed and the model was not recalibrated. From M4 on, calibration acceptance is sample-size-aware. |
| 2026-10-04 | M4 Phase 1 approved (`context/ml-m4-method.md`). M4-D1: a scheduled GitHub Action (Wed 10:00 ET plus a Sun 08:00 ET refresh) commits the ledger and site data; it does not host the site. M4-D2: the last prediction before kickoff is the scored one, and the Wednesday prediction is scored separately. M4-D4: totals deferred to M4b. M4-D3 (simulation strength shocks) is pending. Walker asked for a rest-of-season prediction, which is now part of Phase 1 (per-game probabilities, projected wins) and Phase 2 (full simulation). |
| 2026-10-04 | M4-D3 approved: the playoff simulation includes per-team strength shocks (tau_rest tuned on 2001-2005, backtested on DEV). Walker wants the game model eventually to account for the whole roster that suits up, not just the QB; see M5. |
| 2026-10-04 | M4 Phase 1 built. The live record starts with nflverse week 5 (first kickoff Thursday 2026-10-08; the write-up's "week 6" was off by one). QB change tag threshold 0.08 EPA per dropback. Model series color `--series-3`. |
