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
| M3b | Game-model refinement: Kalman team strength, week-varying weights, better QB term (C1-C4) | M3 | Built 2026-10-05 (dev only). One-SE pick **C2d** (Elo + Kalman `kf_margin` + QB delta): DEV Brier 0.2141 vs A4s 0.2151, -0.0010 [-0.0019, +0.0000]: **bar narrowly not met** (CI includes zero). C1 and every QB extra: no gain. Holdout pre-registration proposed, not run; awaiting Walker. Running as a **shadow model** in the 2026 ledger since 2026-10-05 (A4s stays the model). See "M3b build notes". |
| M4 | Production game model on the site, with live 2026 scorecard and playoff odds | M3 | Phase 1 built 2026-10-04 (weekly predictions, ledger, site, Action). Phase 2 built 2026-10-04; holdout sign-off run once on 2026-10-05: **margin model FAIL** on its primary (MAE vs Elo -0.059, CI [-0.150, +0.035] includes zero), **playoff odds PASS** (made-playoffs ECE 0.028 <= 0.044; Brier vs tau 0 -0.0024, CI upper bound -0.00015 < +0.001). Odds stay gated off the site until Walker decides. |
| M5a | Player value from CC BY data: QB composite, box-score player value (no participation data) | M2 | Built 2026-10-05 (player values, lineups, ladder on 2012-2019). **No gain over A4s**: B1-B3 within noise, the one-SE rule keeps B0 (A4s). Stopped before the notebook, holdout, shadow and site work; awaiting Walker. See "M5 build notes". |
| M5b | On-field adjusted plus-minus (uses CC BY-SA participation data, 2016+) | M5a | **Holdout run 2026-10-05: primary FAIL.** Beats team ratings, loses to box-score values. Kept as a descriptive player view. |
| M6 | Play outcome distribution model (CC BY-SA) | M1, M5b | Built 2026-10-05 (dev only). Every model beats the down-distance-zone baseline by 0.015-0.019 yards of CRPS (about 0.4%) with CIs well below 0. One-SE pick **GBM in both views**. Player identities add at most 0.002 yards (call view N2 minus N0 -0.0023 [-0.0040, -0.0004]; situation view nothing). Dev acceptance preview: PASS in both views under the calibration rule with the 0.010 ECE floor (added 2026-10-05, before any holdout number; null-only, the call view would miss by 0.0002). Holdout pre-registration proposed, not run. See "M6 build notes". |
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

### No-Elo study (2026-10-05, descriptive, DEV only)

Walker asked whether our own features can stand without Elo's log-odds, as a typical ML model would be built. `scripts/ml_m3_noelo.py` reuses the M3 ladder machinery unchanged (walk-forward REG 2001..S-1, season-decay weights, frozen knobs, ties as half rows) on the same 3,450 DEV games (hash `52b8194df3a2ebf4`). Runs are `experiments/runs/20261005T033941Z_m3_noelo_N{1..6}.json` (label dev, `"study": "no_elo"`), and notebook 03, section 7, reads them. It changes nothing live, and no holdout season was loaded. ECE p95 is the one-sided null used in M4; every variant is under it. CIs are paired bootstrap (2,000 reps, seed 20261003); negative means the variant is better.

| Id | Features | Brier | Log loss | Acc | ECE (null p95) | vs Elo 0.2178 | vs A4s 0.2151 |
|---|---|---|---|---|---|---|---|
| N1 | adj_epa_margin | 0.2209 | 0.6329 | 0.635 | 0.018 (0.024) | +0.0030 [+0.0007, +0.0053] | +0.0058 [+0.0036, +0.0080] |
| N2 | adj_epa_margin + qb_delta_diff | 0.2183 | 0.6275 | 0.643 | 0.016 (0.025) | +0.0005 [-0.0021, +0.0033] | +0.0033 [+0.0016, +0.0049] |
| **N3** | N2 + rest_diff + neutral | **0.2182** | 0.6271 | 0.647 | 0.012 (0.025) | +0.0004 [-0.0024, +0.0033] | +0.0031 [+0.0013, +0.0049] |
| N4 | adj pass + rush margins + qb_delta_diff + rest + neutral | 0.2188 | 0.6284 | 0.643 | 0.010 (0.024) | +0.0010 [-0.0018, +0.0039] | +0.0037 [+0.0019, +0.0056] |
| N5 | boosted trees (A6 settings) on N4 features | 0.2209 | 0.6339 | 0.640 | 0.021 (0.026) | +0.0030 [-0.0003, +0.0066] | +0.0058 [+0.0032, +0.0085] |
| N6 | N2 + raw_epa_margin (M1) | 0.2185 | 0.6277 | 0.645 | 0.013 (0.025) | +0.0007 [-0.0021, +0.0034] | +0.0034 [+0.0015, +0.0052] |

The best no-Elo model is N3; N2 is within 0.0001 of it.

| Weeks | n | N3 | N2 | A4s | Elo | N3 minus Elo | A4s minus N2 |
|---|---|---|---|---|---|---|---|
| 1-4 | 751 | 0.2277 | 0.2281 | 0.2243 | 0.2229 | +0.0049 [-0.0005, +0.0100] | -0.0038 [-0.0073, -0.0004] |
| 5-9 | 969 | 0.2167 | 0.2173 | 0.2126 | 0.2140 | +0.0027 [-0.0022, +0.0080] | -0.0047 [-0.0081, -0.0016] |
| 10-18 | 1,730 | 0.2149 | 0.2147 | 0.2125 | 0.2178 | -0.0028 [-0.0066, +0.0011] | -0.0022 [-0.0045, +0.0004] |
| All | 3,450 | 0.2182 | 0.2183 | 0.2151 | 0.2178 | +0.0004 | -0.0033 [-0.0049, -0.0016] |

- N3's log-odds correlate **0.86** with `elo_logit` (0.85, 0.86, 0.87 by bucket) and 0.94 with A4s's.
- A4s refit on one week bucket's training rows (2019 model, trained 2001-2018, ± bootstrap SE over 200 reps): `elo_logit` +1.04 ± 0.25 (weeks 1-4), +0.98 ± 0.21 (5-9), +0.68 ± 0.14 (10-18); `adj_epa_margin` -0.36 ± 1.18, +0.80 ± 1.04, +2.03 ± 0.90. The early-season fits are noisy; the medians over the 14 refits do not fall smoothly (elo_logit 0.47, 0.96, 0.70).

**Read.** Our features can stand on their own, but only to about Elo's level: the best no-Elo model ties Elo (0.2182 vs 0.2178), and it gets there only with the QB term. Elo and our features are mostly the same information (correlation 0.86), but not entirely. Adding Elo back improves Brier by 0.0033 with an interval that excludes zero, and most of that comes in weeks 1-9, when our EPA ratings still lean on a shrunk copy of last season. By weeks 10-18, the no-Elo model is ahead of Elo, Elo's added value falls to 0.002 (not significant), and Elo's weight in the model drops from about 1.0 to 0.7 as EPA's rises. Elo's edge is its long memory of results across seasons. Flexible trees (N5) and adding raw EPA (N6) don't recover it.

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

### M4 Phase 2 build notes (2026-10-04)

**Where things are.** `nflelo/ml/models/margin.py` (mean line, sigma, the two shapes, key-number fit, CRPS, sampling), `nflelo/ml/sim/tiebreak.py` (NFL tiebreakers and seeding), `nflelo/ml/sim/history.py` (actual fields and seeds from the bracket plus the public record), `nflelo/ml/sim/season.py` (the simulation), `nflelo/ml/sim/state.py` (simulation inputs from the live feature code; `as_of_view` rolls a past season back for backtests). Script: `scripts/ml_m4.py margin|tiebreak|tune|backtest|devcurve`. `eval/calibration.py` gained `ece_null_range` (the sample-size-aware check). Live: `scripts/ml_predict.py` (margin fit, `spread_model`, the simulation), `nflelo/ml/live.py` (`ats`, simulation history), `scripts/export_site.py` (`build_playoff_odds`). Reference data: `data/sources/playoff_seeds_2002_2025.csv` (see its README entry). Tests: `tests/ml/test_margin.py`, `test_tiebreak.py`, `test_sim.py`, additions to `test_live.py`. Saved outputs for the notebook: `data/raw/ml/m4/` (gitignored).

**Walker's override of spec section 4 ("single source").** The 2026 live win probability stays A4s all season, since the live record has started. The margin model supplies `spread_model` and drives the simulation. `ml_predict.model_columns` computes `p_home_model` exactly as Phase 1; a test checks it bit for bit with and without the margin model, and another pins `model_2026.json` (`A4s-2026-c6b394e8`). New ledger rows carry `model_version` "A4s-2026-c6b394e8+M-2026-4aab0795"; old rows keep an empty `spread_model` (no backfill).

**Margin model, DEV 2006-2019** (the 3,450 M3 games with moneylines, games hash 52b8194df3a2ebf4; walk-forward REG 2001..S-1, season half-life 8; registry runs `m4_margin_normal`, `m4_margin_keynum`):

| | Value |
|---|---|
| MAE, model | 10.667 |
| MAE, Elo spread (elo_diff/25) | 10.731 |
| MAE, market spread (benchmark) | 10.487 |
| Model minus Elo MAE | -0.064 [-0.118, -0.009], excludes zero |
| Model minus market MAE | +0.180 [+0.096, +0.268] |
| CRPS: continuous normal / rounded normal / key-number | 7.6638 / 7.6620 / 7.6480 |
| Shape rule | normal minus key-number +0.0141, SE 0.0042: outside one SE, **key-number chosen** |

| Implied win probability | Brier | Log loss | Acc | ECE | ECE null 5-95% | minus A4s Brier [95% CI] | within 1 SE |
|---|---|---|---|---|---|---|---|
| normal | 0.2151 | 0.6199 | 0.648 | 0.0129 | [0.010, 0.026] | +0.00004 [-0.00039, +0.00051] | yes |
| key-number | 0.2153 | 0.6203 | 0.648 | 0.0141 | [0.009, 0.025] | +0.00018 [-0.00036, +0.00074] | yes |
| A4s | 0.2151 | 0.6197 | 0.647 | 0.0114 | [0.010, 0.025] | | |

- Home team beats our spread, by predicted spread (5-95% range of a fair coin at that n): (-inf, -7] 0.545 of 213 [0.441, 0.559]; (-7, -3] 0.524 of 416; (-3, 0] 0.475 of 573; (0, 3] 0.501 of 689; (3, 7] 0.487 of 825, all inside; **(7, inf] 0.466 of 734 [0.470, 0.530], just outside**: big home favorites cover a little less than half, so the spread runs about a point too high on them. One bin of six outside a 90% band is roughly what chance gives, so this is noted, not acted on.
- 2019 fit: intercept +0.42, elo_logit +5.96, adj_epa_margin +9.09, qb_delta_diff +27.0, sigma 13.52. Key-number factors (2001-2005, |k| = 0..10): 0.03, 0.69, 0.62, 2.90, 0.87, 0.56, 1.02, 1.90, 0.63, 0.25, 1.42 (ties almost vanish: one tie in 1,270 games).
- 2026 fit (`experiments/live/margin_2026.json`, M-2026-4aab0795, REG 2001-2025, n 6,471): intercept +0.36, elo_logit +5.72, adj_epa_margin +9.98, qb_delta_diff +25.87, sigma 13.26. Before saving, the script refits 2019 on the same frame and matches the logged DEV fit exactly (max |diff| 0).
- Implementation: weighted least squares; sigma is the weighted residual SD with a degrees-of-freedom correction. The key-number shape multiplies the rounded normal by r(|k|) (|k| <= 20, else 1) and renormalizes; r is fit by iterative proportional fitting on 2001-2005 against an in-sample line on those seasons. Sampling is exact rejection sampling from the rounded normal.

**Tiebreakers.** All 40 conference-seasons 2006-2025 reproduce the actual field and seeds exactly (and all 8 of 2002-2005). No seed in those 48 was decided by a points-based step or a coin toss, so nothing is excused. Truth: the bracket alone fixes 15 conference-seasons; in 33 it leaves some order open and the public seeds (nfldata `standings.csv`) decide, after a check that they are among the seedings the bracket allows. During the build the check caught one bug (common-games records were matched to the wrong clubs when a group wasn't in sorted order), which had produced three mismatches (2015 AFC, 2017 NFC, 2025 AFC); it is fixed. Rules follow nfl.com's page (read 2026-10-04): two-club and three-club lists for divisions and wild cards, step 0 for wild cards, restarts (two left: two-club step 1; three left: division step 1 or wild-card sweep), one club advances per application, division winners seeded with wild-card tiebreakers. In simulations both combined-ranking steps use net points (only margins exist); net touchdowns are never available, so the coin toss follows net points.

**Simulation.** Vectorized draws and bracket; the tiebreakers run per simulated season in Python and are fast enough (about 0.25 ms per season). Team strength s_t = b1 c Elo_t + b2 adj_t + b3 qb_t with the listed starter carried forward; remaining REG means use each game's own features (the ledger's spread) with that QB term; playoff home edge = intercept + b1 c HFA; the Super Bowl drops it; playoff ties are a coin flip. Each simulated season draws one Normal(0, tau_rest) shock per team.

**tau_rest tuning, 2002-2005** (2001 is left out: 31 teams and six divisions, which the 2002+ tiebreakers don't cover; 16 starts, 512 team-starts, 5,000 simulations per start, common random numbers). Made-playoffs Brier: 0: 0.1106, 0.5: 0.1103, 1: 0.1106, 2: 0.1104, 3: 0.1107, 4: 0.1109, 6: 0.1119, 8: 0.1144. **Chosen 0.5**, but the curve is flat from 0 to 3. By season it disagrees: 2002 (model fit on 2001 alone) keeps improving up to 8, 2004-2005 prefer 0 to 1.

**Backtest, DEV 2006-2019** (56 starts, 1,792 team-starts, 10,000 simulations per start; tau 0.5 against 0 on the same random numbers):

| Outcome | Brier tau 0.5 | Brier tau 0 | diff [95% CI, team-starts] | [95% CI, by season] | ECE | ECE null 5-95% |
|---|---|---|---|---|---|---|
| Made playoffs | 0.1214 | 0.1213 | +0.0001 [-0.0000, +0.0003] | [-0.0000, +0.0003] | **0.042** | [0.012, 0.026] |
| Won division | 0.0881 | 0.0880 | +0.0001 [-0.0000, +0.0003] | [-0.0000, +0.0003] | 0.016 | [0.010, 0.024] |
| #1 seed | 0.0317 | 0.0317 | -0.0000 [-0.0001, +0.0001] | | 0.009 | [0.006, 0.014] |
| Reached SB | 0.0458 | 0.0458 | +0.0000 [-0.0001, +0.0002] | | 0.010 | [0.005, 0.015] |
| Won SB | 0.0283 | 0.0283 | +0.0000 [-0.0000, +0.0001] | | 0.006 | [0.002, 0.010] |

- **STOP condition met:** the tuned shock does not beat tau_rest = 0, and the made-playoffs odds fail the sample-size-aware calibration check. The reliability table shows overconfidence, as the spec predicted: predictions of 0.7-0.9 came true 64-75% of the time, 0.1-0.4 came true 21-45%. ECE by start week: 0.071 (week 4), 0.048, 0.035, 0.027 (week 16).
- Descriptive only, not a selection (`ml_m4.py devcurve`, 5,000 simulations per start): DEV made-playoffs Brier and ECE by tau: 0: 0.1214 (0.039), 1: 0.1212 (0.039), 2: 0.1206 (0.034), 3: 0.1199 (0.025), 4: 0.1196 (0.020), 6: 0.1195 (0.015). So shocks of 4-6 points would help on DEV, but the pre-registered tuning window (four seasons) is too small and noisy to find that.
- These 2002-2005 tuning and tau 0.5 backtest numbers were not logged; they are superseded by the DEV re-tune below.

**Re-tune on DEV (Walker's decision, 2026-10-04).** Grid 0 to 8 by 0.5, 5,000 simulations per start with common random numbers, 56 starts (1,792 team-starts), made-playoffs Brier, smallest tau within one team-start bootstrap SE of the best (registry `m4_tau_rest_tune_dev`):

| tau | 0 | 1 | 2 | 3 | 3.5 | 4 | **4.5** | 5 | 6 | 7 | 7.5 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Brier | 0.1214 | 0.1212 | 0.1206 | 0.1199 | 0.1199 | 0.1196 | **0.1194** | 0.1195 | 0.1195 | 0.1197 | 0.1200 | 0.1204 |
| minus best (SE) | +0.0020 (0.0007) | +0.0018 (0.0006) | +0.0012 (0.0005) | +0.0005 (0.0003) | +0.0005 (0.0003) | +0.0002 (0.0002) | best | +0.0001 (0.0002) | +0.0001 (0.0003) | +0.0003 (0.0005) | +0.0006 (0.0006) | +0.0010 (0.0007) |

Within one SE: 4.5 to 7.5. **Chosen: 4.5** (the best is also the smallest within one SE; 4.0 misses by a hair). ECE at 4.5 in the tuning sims (null 5-95%): made playoffs 0.016 [0.012, 0.028], division 0.023 [0.011, 0.025], #1 seed 0.012 [0.006, 0.015], reach SB 0.014 [0.004, 0.015] all inside; win SB 0.002 [0.002, 0.010] just under the band (a too-low ECE, not miscalibration: nearly every prediction is near 0).

DEV backtest at 4.5 vs 0 (10,000 simulations per start; **in-sample for tau, so descriptive**; registry `m4_playoff_odds`, `m4_playoff_odds_tau0`):

| Outcome | Brier 4.5 | Brier 0 | diff [95% CI, team-starts] | [95% CI, by season] | ECE | null 5-95% |
|---|---|---|---|---|---|---|
| Made playoffs | 0.1195 | 0.1213 | -0.0017 [-0.0031, -0.0004] | [-0.0044, +0.0008] | 0.018 | [0.012, 0.028] |
| Won division | 0.0882 | 0.0880 | +0.0001 [-0.0011, +0.0013] | [-0.0017, +0.0022] | 0.021 | [0.011, 0.025] |
| #1 seed | 0.0320 | 0.0317 | +0.0003 [-0.0005, +0.0010] | [-0.0008, +0.0014] | 0.012 | [0.006, 0.015] |
| Reached SB | 0.0465 | 0.0458 | +0.0007 [-0.0001, +0.0016] | [-0.0004, +0.0019] | 0.014 | [0.004, 0.015] |
| Won SB | 0.0281 | 0.0283 | -0.0002 [-0.0006, +0.0002] | [-0.0008, +0.0003] | 0.002 | [0.002, 0.010] (just under) |

- Reliability at 4.5: predictions of 0.8-0.9 came true 86%, 0.9-1.0 98.6%; the largest gaps are +6 points at 0.3-0.4 and -7 at 0.7-0.8. At tau 0 the 0.8-0.9 bin came true only 74%.
- Made playoffs by start week (Brier 4.5 / 0, ECE): week 4 0.1732 / 0.1764, 0.043; week 8 0.1383 / 0.1398, 0.032; week 12 0.1041 / 0.1056, 0.031; week 16 0.0625 / 0.0632, 0.026. Eight of 14 seasons improve, five get worse, one is level.
- The shock helps "made playoffs" and leaves the other outcomes about level; reach-SB Brier is slightly worse (+0.0007, CI includes 0).

**Timing.** Weekly run (`ml_predict.py --offline`, cache warm): 11.2 s end to end, of which the simulation is 1.0 s to build the state and 5.8 s for 20,000 seasons (tiebreakers 5.4 s). Tiebreaker reproduction 8.4 s for 24 seasons (mostly enumerating bracket-consistent seedings). 2002-2005 tuning 327 s; DEV tuning about 19 minutes (17 taus x 56 starts x 5,000 seasons); DEV backtest about 4 minutes; `ml_m4.py all` 1,373 s.

**Site gate.** `experiments/live/publish.json` has `"publish_sim": false`: the weekly job keeps simulating (tau 4.5) and appending to `sim_2026.csv`, but the exporter leaves the odds and the win ranges out of `ml.json` until the sign-off. Margin spreads are published.

**Sign-off code.** `ml_m4.py signoff-dry-run` runs the sign-off path on DEV without logging and compares every number with `margin_dev.json` and `backtest.json`; `ml_m4.py signoff` is the one holdout run (refuses if a holdout sign-off run already exists). Notebook: `notebooks/04_margin_and_playoff_odds.ipynb` (executed with tau 4.5).

### M4 holdout pre-registration (written 2026-10-04, before any 2020-2025 margin or playoff-odds number was computed)

Walker approved ONE holdout run covering the margin model and the playoff odds. This subsection is the protocol; it is committed before the run, and the result is recorded in this file whatever it shows. Nothing is retuned afterwards.

**Frozen before the run** (all in committed code; the run refuses to start if a holdout sign-off run already exists in `experiments/runs/`):
- Margin model: features `elo_logit`, `adj_epa_margin`, `qb_delta_diff`; weighted least squares fit walk-forward, season S fit on REG 2001..S-1 with weights 0.5 ** ((S - 1 - season) / 8); sigma = weighted residual SD of that fit (degrees-of-freedom corrected). Shape: **key-number** (`mm.CHOSEN_SHAPE`), factors fit on 2001-2005 only, |k| <= 20, symmetric. M3 rating knobs unchanged (lambda 100, half-life 48, rho 0.25, QB k 100).
- Playoff simulation: **tau_rest = 4.5** points (`sim.TAU_REST`, tuned on DEV 2006-2019 by the one-SE rule; see the Phase 2 notes), the key-number shape, 10,000 simulated seasons per start, seed `20261004 + 1000 * season + start week` for both the frozen tau and tau = 0 (common random numbers), NFL tiebreakers as validated (`tiebreak.py`).
- Bootstraps: 2,000 replicates, seed 20261003. ECE: 10 uniform bins; null range = 5th to 95th percentile of the ECE of a perfectly calibrated forecaster at the same n (500 draws, outcomes drawn from the predictions, seed 20261003, `calibration.ece_null_range`). Fair-coin bands for the spread bins: 20,000 binomial draws, seed 20261003.

**Data and windows.**
- Margin model: REG 2020-2025 games with a moneyline (the M3 sign-off game set: n = 1,615, games hash `1b27abff81bddb2c` expected); each must also have a market spread or the run stops. Walk-forward fits for each season 2020-2025.
- Playoff odds: seasons 2020-2025 (the 7-seed era), simulated from weeks 4, 8, 12 and 16 (24 starts, 768 team-starts). Each start uses only what was available an hour before that week's first kickoff: the margin fit for that season (REG 2001..S-1), features and Elo as of that moment, starters listed only for the start week (`state.as_of_view`). Truth: actual playoff fields and seeds from the nflverse bracket plus `data/sources/playoff_seeds_2002_2025.csv`.

**Pass rules (exact).**
- Margin model, primary: with d = |margin - model spread| - |margin - Elo spread (elo_diff / 25)| per game, the upper end of the 95% paired bootstrap CI of mean(d) is below 0. **PASS iff ci_high < 0.**
- Margin model, secondary (each reported pass or fail; they do not change the primary verdict):
  - S1: implied win-probability Brier (key-number shape) minus A4s Brier on the same games is at most one bootstrap SE (diff <= boot_se). This informs a possible 2027 switch to one source.
  - S2: ECE of the implied win probability is at or below the 95th percentile of its null.
  - S3: the home team beats the model's spread inside the fair-coin 5-95% band in at least 5 of the 6 spread bins ((-inf,-7], (-7,-3], (-3,0], (0,3], (3,7], (7,inf)).
  - Descriptive: market spread MAE, CRPS of key-number vs rounded normal, MAE by season.
- Playoff odds, primary (both must hold): P1, the made-playoffs ECE at the frozen tau is at or below the 95th percentile of its null at n = 768; P2, with d = (p_tau - y)^2 - (p_0 - y)^2 per team-start, the upper end of the 95% team-start paired bootstrap CI of mean(d) is below +0.001. **PASS iff P1 and P2.**
- Playoff odds, secondary: ECE at or below the 95th percentile of its null for won division, #1 seed, reached Super Bowl, won Super Bowl (each pass or fail).
- Every ECE rule (margin S2, playoff P1 and the playoff secondaries) is **one-sided**: PASS iff ECE <= p95, the 95th percentile of the ECE of a perfectly calibrated forecaster at the same n. An ECE below the 5th percentile is not miscalibration. Whether each ECE also falls inside the two-sided [p05, p95] range is reported as a descriptive field only (`ece_in_null_range`, `info_two_sided_in_null_range`).
- **Rule change, 2026-10-05:** the ECE rules were changed from two-sided (p05 <= ECE <= p95) to one-sided (ECE <= p95) before any holdout number was computed, because the two-sided version penalized better-than-null calibration (on DEV, won-Super-Bowl odds at tau 4.5 have ECE 0.002 against [0.002, 0.010], just under the band, because almost every prediction is near 0).
- Descriptive: made-playoffs Brier (frozen tau and tau = 0) and ECE by start week and by season; the season-cluster bootstrap CI; reliability tables.

**Procedure.** First `python scripts/ml_m4.py signoff-dry-run`: the same code path on DEV 2006-2019 without logging, which must reproduce the logged DEV margin numbers and the DEV backtest exactly (max |diff| 0). Then, once, `python scripts/ml_m4.py signoff`: runs on 2020-2025, prints every number and each rule's verdict, writes `data/raw/ml/m4/signoff_holdout.json`, and logs `m4_signoff_margin` and `m4_signoff_playoff_odds` to `experiments/runs/` with `holdout: true`. The site gate (`experiments/live/publish.json`) is opened only if the playoff-odds primary passes and Walker approves. The 2026 live win probability stays A4s whatever the margin result.

### M4 holdout sign-off (2026-10-05)

`python scripts/ml_m4.py signoff` ran once, on the committed pre-registration (commit e51e890, clean tree), with nothing changed. Frozen: key-number margin shape, tau_rest 4.5, 10,000 simulations per start, seeds 20261004 + 1000 * season + week. Runs: `experiments/runs/20261005T024754Z_m4_signoff_margin.json` and `..._m4_signoff_playoff_odds.json` (`holdout: true`); full output `data/raw/ml/m4/signoff_holdout.json`. Nothing was retuned.

Margin model: REG 2020-2025 with moneylines, n = 1,615, games hash `1b27abff81bddb2c` (as expected). Playoff odds: 24 starts, 768 team-starts.

| Rule | Value | Threshold | Result |
|---|---|---|---|
| Margin primary: MAE model minus Elo spread | -0.059 [-0.150, +0.035] (model 10.126, Elo 10.185) | CI upper bound < 0 | **FAIL** |
| Margin S1: implied Brier minus A4s | +0.00001 (SE 0.00047; 0.2195 vs 0.2195) | <= 1 SE | PASS |
| Margin S2: ECE of implied win probability | 0.0348 | <= p95 0.0363 | PASS |
| Margin S3: spread bins inside the fair-coin band | 5 of 6 | >= 5 | PASS |
| Odds P1: made-playoffs ECE | 0.0282 | <= p95 0.0441 | PASS |
| Odds P2: made-playoffs Brier, tau 4.5 minus tau 0 | -0.0024 [-0.0049, -0.00015] (0.1234 vs 0.1258) | CI upper bound < +0.001 | PASS |
| Odds secondary: division ECE | 0.0353 | <= 0.0378 | PASS |
| Odds secondary: #1 seed ECE | 0.0114 | <= 0.0234 | PASS |
| Odds secondary: reach SB ECE | 0.02187 | <= 0.02188 | PASS (by 0.00001) |
| Odds secondary: win SB ECE | 0.0039 | <= 0.0143 | PASS |

**Verdicts as pre-registered: margin model FAIL (primary); playoff odds PASS (primary and all secondaries).** The two-sided descriptive field is inside [p05, p95] for every ECE.

Descriptive, margin model:
- Market spread MAE 9.764; model minus market +0.362 [+0.230, +0.492] (wider than DEV's +0.180).
- CRPS: continuous normal 7.2801, rounded normal 7.2783, key-number 7.2541 (key-number still best, by 0.024, SE 0.006).
- Implied win-probability ECE: rounded normal 0.0329, key-number 0.0348, A4s 0.0283 (null 5-95% about [0.014, 0.036]).
- Home beats our spread by bin: (-inf, -7] 0.586 of 128 [0.430, 0.570] OUTSIDE (big away favorites cover too rarely, so our spread overrates them); (-7, -3] 0.554; (-3, 0] 0.453; (0, 3] 0.487; (3, 7] 0.480; (7, inf] 0.497.
- MAE by season (model / Elo / market): 2020 9.99 / 10.04 / 9.83; 2021 11.22 / 11.32 / 10.78; 2022 9.05 / 9.00 / 8.74; 2023 10.44 / 10.48 / 9.90; 2024 10.02 / 10.05 / 9.61; 2025 10.03 / 10.20 / 9.72. The model beats Elo in 5 of 6 seasons, but by about as much as on DEV (-0.06) on a window half the size, so the interval is twice as wide.
- 2025 fit: intercept +0.30, elo_logit +5.74, adj_epa_margin +10.20, qb_delta_diff +24.61, sigma 13.32.

Descriptive, playoff odds (made playoffs, Brier tau 4.5 / tau 0, ECE):
- By start week: week 4 0.1721 / 0.1758, 0.092; week 8 0.1327 / 0.1337, 0.055; week 12 0.1207 / 0.1250, 0.047; week 16 0.0682 / 0.0689, 0.033 (192 team-starts each). The shock helps at every start week.
- By season: 2020 0.0785 / 0.0743; 2021 0.1540 / 0.1618; 2022 0.1185 / 0.1187; 2023 0.1453 / 0.1463; 2024 0.0908 / 0.0886; 2025 0.1533 / 0.1653. Four of six improve. The season-cluster CI is [-0.0071, +0.0019] (6 clusters), descriptive only.
- Other outcomes, Brier tau 4.5 vs 0: division 0.1122 vs 0.1158 (-0.0037 [-0.0060, -0.0014]); #1 seed 0.0374 vs 0.0372; reach SB 0.0477 vs 0.0466 (+0.0011 [-0.0001, +0.0023]); win SB 0.0277 vs 0.0278.
- Reliability: the biggest gaps are -10.5 points at 0.6-0.7 and +6.2 at 0.5-0.6 (about 50 team-starts each); 0.8-0.9 came true 86%, 0.9-1.0 97.5%.

Notes:
- Per-week and per-season ECEs (0.03 to 0.12) are on 128-192 team-starts each and are much noisier than the pooled 0.028; they are not rules.
- Reach-SB passed its secondary by 0.00001, a coin flip's margin. Treat it as borderline.
- The margin primary failed for lack of power more than lack of effect: the point estimate matches DEV, and S1-S3 pass. Under the pre-registration that is still a FAIL. The live 2026 win probability stays A4s either way. The ledger's `spread_model` and the site spreads were not gated, so they stay up unless Walker decides otherwise.
- `experiments/live/publish.json` is unchanged (`publish_sim: false`); opening the playoff odds is Walker's call.

### M5 build notes (2026-10-05, stopped early: no gain)

Spec: `context/ml-m5-method.md` (approved; D1 (a), D2, D3, D4). Built through the ladder on the M5 window, then stopped under the "B1-B3 show no gain" rule. Notebook 05, the holdout, shadow logging and site work were not started.

**Where things are.** `nflelo/ml/data.py` (cached loaders for `player_stats`, `depth_charts`, `injuries`, `rosters_weekly`, `players`, and `load_participation_for_validation`; `load_depth_charts` normalizes both formats: week-shift rule, 2004 dropped, 2005+ by default, 2025+ ESPN snapshots resolved to the team's last snapshot strictly before kickoff), `nflelo/ml/players/` (`positions.py`, `credit.py`, `value.py`, `lineup.py`, `inputs.py`, and `validate.py`, the only reader of participation), `nflelo/ml/features/roster.py` (features plus `roster_leakage_check`), `scripts/ml_tune_players.py`, `scripts/ml_m5.py` (`--leakcheck`, `--validate`), `tests/ml/test_m5.py`. `eval/windows.py` gained `M5_DEV = (2012, 2019)` (label `m5dev`, guarded like DEV); `asof.corrupt_from` now also shuffles every credited-player ID and play-event flag on late plays. Saved outputs: `data/raw/ml/m5/` (gitignored).

**Data finding (corrects `m5-data-scope.md`).** Weekly-roster statuses are week-accurate only from 2016. In 2005-2015, 35-45% of player-weeks marked RES (and 13-36% marked CUT) had the player credited on a play that same week: the status is effectively the season-end status, which is future information. The first ladder run used it and a post-hoc count of "rank-1 starters on reserve" beat A4s by 0.0027 Brier while having zero correlation with the market's view (0.007), the signature of a leak. `lineup.roster_status_seasons` now detects week-accurate seasons from the data (2016-2019 pass), and the roster and inactive layers are switched off elsewhere. After the fix the same count has no gain (-0.0001 [-0.0009, +0.0007]) and correlates with the market (-0.15) as real injury news should.

**Credit and values.** Credit per the spec table, era-aware from the data (receiving 2003-2008, QB hits 1999-2005 and TFL 2003-2007 unavailable). The 2019 top-10 lists are sane (QB: Mahomes, Jackson, Prescott; DL: Hunter, Crosby; LB: Barrett, Judon, Watt; CB: Peters, White, Gilmore; S: Harris, Byard; WR: Godwin, Thomas; TE: Waller, Andrews; K: Lambo, Tucker). Tuned on 2001-2008 without outcomes (`m5_tune_players`): half-life 32 weeks; k = QB 4, DL 8, LB 8, RB 16, WR 16, TE 16, CB 16, K 16, S 32 (all interior); skill vs replacement QB 0.30, K 0.13, DL 0.09, WR 0.09, LB 0.08, TE 0.05, CB 0.03, S 0.03, RB 0.02. With/without lambda 4,000 plays (interior) on 2006-2008 team residuals: MSE 0.04848 vs 0.04897 box-only and 0.05368 intercept-only.

**Validation against participation (2016-2019, validation only).** Usage-based share MAE by group: QB 0.009, RB 0.110, OL 0.143 (pregame p), S 0.162, WR 0.190, LB 0.198, TE 0.228, CB 0.235 (bias -0.15: two chart slots vs nickel), DL 0.261. Lineup flags (rank-1 starters, p < 0.5): 63% of actual absences caught, 1.0% false alarms; 2019 with partial inactives 81% caught; Friday-only 57% (2019: 59%). Status probabilities (2009-2015): Questionable 0.60 (QB) to 0.87 (RB), Doubtful 0.01-0.05, Out about 0.

**Ladder** (REG 2012-2019 with moneylines, n = 2,047, games hash `890722aef13c4355`; B0 = A4s trained 2001..S-1, reproduces the M3 ladder exactly; other steps trained 2009..S-1; paired bootstrap vs B0, 2,000 reps, seed 20261003; all ECEs pass the one-sided null):

| Step | Brier | vs B0 [95% CI] |
|---|---|---|
| B0 A4s (chosen) | 0.2147 | |
| B0r A4s trained from 2009 | 0.2150 | +0.0004 [-0.0000, +0.0007] |
| B1 + lineup delta, box values | 0.2149 | +0.0002 [-0.0006, +0.0011] |
| B2 + lineup delta, with/without | 0.2150 | +0.0004 [-0.0004, +0.0012] |
| B3 B2 + preseason change | 0.2154 | +0.0008 [-0.0004, +0.0019] |
| B4 B3 split off/def | 0.2154 | +0.0008 [-0.0004, +0.0020] |
| B5 B3 with 0/1 probabilities | 0.2155 | +0.0008 [-0.0003, +0.0020] |
| B3f B3 Friday-only | 0.2155 | +0.0008 [-0.0004, +0.0019] |

Elo 0.2175, market 0.2110. Weeks 1-4: B3 +0.0009 (the preseason term does not fix the early season). Games with 2+ non-QB starters out (n = 755): B1 -0.0005 [-0.0019, +0.0009]. The lineup-delta coefficients are positive in every refit (B1 +3.3 to +5.1, B2 +2.4 to +4.2) and the features correlate +0.13 with the market's departure from A4s, so they carry real information, but too little to move Brier on 2,047 games. The preseason coefficients flip sign across refits.

**Leakage.** Every builder passes `roster_leakage_check` (play-by-play scrambled from as_of, later weeks' depth charts, injuries and rosters scrambled) on synthetic data and on real 2017 games; the positive control (the game's own players) fails. The first real-data run caught a with/without leak (a player's position group, which sets his prior strength, was read from his latest row, including later games); fixed and covered by a test. Known exceptions, all fixed tables: status probabilities estimated on 2009-2015 (overlaps 2012-2015, as the spec says), the era-gap and roster-accuracy tables, and position groups taken from each season's charts and rosters.

**Timing.** Tuning 122 s; ladder 72 s end to end; real-data leakage check 75 s.

### M5b build notes (2026-10-05, dev only; holdout not run)

Spec: `context/ml-m5b-method.md` (approved under delegation). Everything participation-derived is **CC BY-SA 4.0** and labelled so in its docstrings, file metadata (`data/raw/ml/m5b/LICENSE.txt`, a `license` key in every JSON) and run records. A test (`tests/ml/test_m5.py`) keeps participation and RAPM out of every module except `players/participation.py`, `players/rapm.py` and `scripts/ml_m5b.py`, so nothing reaches A4s.

**Where things are.** `nflelo/ml/data.py` (`load_participation`, M5b only; cached with manifest entries for 2016-2025), `nflelo/ml/players/participation.py` (cleaning, side repair, sparse design, snaps and shares), `nflelo/ml/players/rapm.py` (Bayesian RAPM, season chaining, posterior SDs, season-ahead API, leakage check, `TUNED`), `scripts/ml_m5b.py` (`clean`, `tune`, `dev`, `leakcheck`, `signoff-dry-run`, `signoff`), `tests/ml/test_m5b.py`, `notebooks/05_player_values.ipynb`, `notebooks/06_player_ratings.ipynb`. Also: `eval/windows.py` gained `M5B_DEV = (2018, 2019)` (label `m5bdev`, guarded like DEV); `eval/bootstrap.py` gained `cluster_bootstrap_diff` (resamples whole games); `players/value.py` gained `BoxValues.sums_at` (box values at an explicit as-of). Saved outputs: `data/raw/ml/m5b/` (gitignored).

**Data cleaning (finding).** The raw participation lists would have failed the 5% drop rule: plays with exactly 11 + 11 listed GSIS IDs were 87.3% of clean scrimmage plays in 2016 and 92.1% in 2018 (2016: one offensive or defensive player with no GSIS mapping; 2018-2019: a 12th offensive ID that is not one of the 22 players on the play). `players_on_play` lists all 22 (as NFL `nfl_id`s through 2022, GSIS IDs from 2023), and every one maps to GSIS through the players table (CC BY). The repair assigns each of the 22 to his team (the side he is listed on most often in that game; a player never listed in the game goes to the side that is one short). On every play whose lists were already 11 + 11 (2016-2019), the repair reproduces them exactly. Drop rate after the repair, kept plays (REG and POST, garbage time out):

| Season | Era | Clean plays | Raw drop | Repaired | Drop | Kept |
|---|---|---|---|---|---|---|
| 2016 | NGS | 28,488 | 12.7% | 3,511 | 0.42% | 28,367 |
| 2017 | NGS | 28,086 | 0.1% | 0 | 0.12% | 28,051 |
| 2018 | NGS | 27,629 | 7.9% | 2,169 | 0.07% | 27,611 |
| 2019 | NGS | 27,957 | 4.9% | 1,335 | 0.09% | 27,932 |
| 2020-2022 | NGS | 88,740 | 0.1% | 0 | 0.09% | 88,662 |
| 2023-2025 | FTN | 88,969 | 0.1% | 0 | 0.10% | 88,883 |

Players on the field per season: 1,833-1,858 (2016-2019), 1,964-2,034 (2020-2022), 1,905-1,969 (2023-2025). Per position group the counts are steady across the 2023 NGS-to-FTN switch (for example OL 312-335 and CB 224-237 every season 2020-2025). Every kept play has exactly 22 player-snaps. These are counts only; no 2020-2025 outcome was computed.

**Model.** As in the spec: one coefficient per (player, side), D is EPA allowed (value = -D). Situation terms: down 2/3/4, distance 3-6/7-10/11+, field zone 11-20/21-50/51-80/81+ yards to go, pass, plus home; unpenalized. Prior mean = `box_scale` x the player's M5 box value at the start of S (EPA per on-field play above replacement; 0 with no box record), sign flipped for defenders; a player on the other side of the ball gets prior 0 and `lam_other`. Season chaining in precision form: the fit for S uses H(S) = sum over T < S of decay^(S-1-T) H_T (the S-1 posterior with its evidence discounted toward the box prior, full covariance kept). Solved by conjugate gradients with a Jacobi preconditioner on the offset system (beta - prior); a dense Cholesky checks it (max difference about 1e-11) and gives exact posterior SDs, sigma^2 (H + Lambda)^-1. About 2,600-3,100 active players per fit; a fit takes under 0.1 s. Unit tests recover known effects from synthetic lineups (correlation > 0.9), and two always-together players are split evenly under equal priors and by the prior otherwise, with the pair's sum set by the data and a wider SD than a rotating teammate.

**Leakage (finding and fix).** The real-data check (`ml_m5b.py leakcheck`) corrupts all of 2019 (play-by-play outcomes, situations and credited IDs; participation lists; depth charts, injuries, rosters), rebuilds every input from raw, refits, and predicts the clean 2019 plays. The first run failed: the start-of-2019 box priors moved (max 0.0038). Cause: `credit.lookup_groups` falls back to a player's latest position-table entry when his season has none (its docstring says "nearest earlier"), which can be a later season. M5b now rebuilds the box credits for each S from play-by-play, depth charts and position tables cut at S-1. After the fix, all four models' 2019 predictions are identical (max diff 0.0 on 27,932 plays), the box priors and groups are unchanged, and the positive control (training on 2019 too) changes every play (max 0.77). The same `lookup_groups` behaviour still affects M5 itself (listed there as a known exception); it is not changed here.

**Tuning** (`m5b_tune`, 362 settings, coordinate descent; objective = pooled play-level EPA MSE of the season-ahead predictions for 2018 (from 2016-17) and 2019 (from 2016-18)). Frozen as `rapm.TUNED`:

| Knob | Chosen | Note |
|---|---|---|
| Ridge strength (plays) | QB 500, RB 1,000, OL 1,000, LB 2,000, TE 2,000, WR 4,000, S 4,000, CB 32,000, DL 1,024,000 | DL at the top of the grid = prior only: on-field data did not help DL |
| `lam_other` | 20,000 | not tuned |
| Decay | 1.0 (old seasons count fully) | edge of the grid |
| Situation terms | on | off costs +0.0066 MSE |
| `box_scale` | 0.25 | **added to the tuning** (the spec fixes the prior as the box value "in the same units"); 1.0 costs +0.0009, 0 costs +0.0002 |

Dev MSE 1.93863. One ridge strength for every group costs +0.0010.

**Dev table** (season-ahead 2018 and 2019, all kept REG and POST plays: 55,543 plays, 534 games, games hash `2e7aa347318c6509`; paired bootstrap resampling games, 2,000 reps, seed 20261003). Optimistic for RAPM: its knobs were tuned on these predictions; the baselines were not.

| Model | MSE | RAPM minus model [95% CI] |
|---|---|---|
| RAPM | 1.93863 | |
| Team ratings (M3 opponent-adjust as of week 1, + situation terms) | 1.94136 | -0.00273 [-0.00417, -0.00128] |
| Box values (M5, summed over the 22, refit) | 1.94050 | -0.00187 [-0.00319, -0.00048] |
| Intercept | 1.94914 | -0.01051 [-0.01352, -0.00734] |

By season: 2018 RAPM - team -0.0022 [-0.0040, -0.0004], RAPM - box -0.0026 [-0.0044, -0.0008]; 2019 RAPM - team -0.0033 [-0.0055, -0.0011], RAPM - box -0.0012 [-0.0032, +0.0008]. Box minus team -0.0009 [-0.0024, +0.0008]. Box baseline coefficients (offense sum, defense sum): 2018 +0.54, -0.99; 2019 +0.63, -0.74.

**Secondaries (dev).**
- Team-game EPA-per-play margin from the actual lineups (534 games): MSE RAPM 0.1793, team 0.1863, box 0.1890, intercept 0.1930; RAPM minus team -0.0069 [-0.0120, -0.0019], minus box -0.0097 [-0.0152, -0.0046].
- Year-to-year correlation of values (200+ snaps in both seasons; mean of 2016-17, 2017-18, 2018-19): chained fits 0.63-0.85 (they share data, so they must correlate); single-season fits (decay 0) OL 0.10, TE 0.09, S 0.19, LB 0.24, WR 0.32, RB 0.41, QB 0.43, CB 0.45 and DL 0.65 (both mostly prior). One season of on-field signal mostly does not repeat for linemen, tight ends and the back seven.
- Game model, descriptive: A4s logit + b x `rapm_lineup_delta_diff` (b fit on REG 2016..S-1; RAPM values with participation-based snap shares; expected lineup rescaled to 10 offensive and 11 defensive non-QB slots), REG 2018-2019 with moneylines, n = 512: A4s 0.2170, with the term 0.2177, +0.0007 [-0.0012, +0.0025]. b = -4.47 (2018), -1.15 (2019), although the term correlates +0.20 with the market's departure from A4s. 512 training games; reported only. The remembered lineups and expected shares use earlier weeks' participation of the same season, which a live run would not have.

**Sanity** (ratings fit on 2016-2019, players with 300+ snaps in 2019; value, SD; full lists in `ml_m5b.py dev` output and `data/raw/ml/m5b/ratings_through_2019.parquet`). QB top: Mahomes +0.131 ± 0.045, Garoppolo, Watson, Rodgers, Carr; bottom: Rudolph -0.043, Goff, Haskins, D. Jones, Flacco. RB: Henry, A. Jones, Fournette, Kamara, Ekeler; bottom C. Hyde, Freeman. WR: Hill, Woods, Jeffery, Kupp, J. Jones (spread only about ±0.03). TE: Andrews, Heuerman, Kelce. OL: Whitworth +0.051 ± 0.035, Easton, Pugh, Solder, Bozeman, T. Smith; bottom Ereck Flowers -0.067. DL (prior only): Hunter, Donald, D. Lawrence. LB: Judon, Kendricks, Mack, Warner, Za'Darius Smith. CB (almost prior): Peters, Fuller, Gilmore, White. S: Harrison Smith, K. Jackson, Mathieu, E. Thomas.

How far ratings moved from their priors (mean |value - prior|; posterior SD / prior SD): OL 0.0149 (0.82; prior is flat 0, so all OL spread is from the data), LB 0.0145 (0.81), QB 0.0290 (0.74), RB 0.0198 (0.79), TE 0.0100 (0.86), S 0.0075 (0.89), WR 0.0055 (0.93), CB 0.0012 (0.99), DL 0.0000 (1.00). Most individual intervals include zero.

**Registry.** `m5b_tune`, `m5b_dev_rapm`, `m5b_dev_team`, `m5b_dev_box`, `m5b_dev_intercept`, `m5b_dev_game_model` (label `m5bdev`, window REG+POST 2018-2019), logged from an uncommitted tree; re-log after the commit if a clean-commit record is wanted. `signoff-dry-run` reproduces `data/raw/ml/m5b/dev.json` exactly (max |diff| 0).

**Timing.** `clean` 7 s; `tune` 52 s (load 9 s, 362 settings 42 s); `dev` 53 s (evaluation 31 s, game model 6 s, stability 6 s); `leakcheck` 83 s; `signoff-dry-run` 52 s.

### M5b holdout pre-registration (proposed 2026-10-05; not yet run)

One holdout run, scored once, with the settings below frozen in committed code. The result is recorded here whatever it shows, and nothing is retuned afterwards.

**Frozen settings.**
- Ratings: `rapm.TUNED` = ridge strengths QB 500, RB 1,000, WR 4,000, TE 2,000, OL 1,000, DL 1,024,000, LB 2,000, CB 32,000, S 4,000; `lam_other` 20,000; decay 1.0; situation terms on; `box_scale` 0.25. Priors: M5 box values (`value.TUNED`: half-life 32 weeks, k as tuned in M5) at the start of each season S, rebuilt from data before S with position tables cut at S-1; position groups as of S-1. CG tolerance 1e-10.
- Plays: `participation.build_plays` as committed (M1 scrimmage plays, garbage time outside 0.05-0.95 win probability, REG and POST, side repair, exactly 11 + 11 GSIS IDs). Expected from the cleaning counts: 177,545 kept plays in 1,693 games for 2020-2025 (if nflverse has not revised the files).
- Baselines: (a) M3 opponent-adjust ratings (`opponent_adjust.TUNED`: lambda 100, half-life 48, rho 0.25) as of week 1 of S, plus the situation terms fit by weighted least squares on the training plays (each training season against the ratings as of the next season's week 1); (b) M5 box values summed over the 11 offensive and 11 defensive players (two coefficients) plus the situation terms, weighted least squares on the training plays with each training season's start-of-season values; (c) intercept-only reference. Training weights for all models: decay^(S-1-T) = 1.
- Bootstraps: 2,000 replicates, seed 20261003; play-level CIs resample games.

**Window.** Season-ahead for each S in 2020-2025: ratings and baselines fit on 2016..S-1, predicting every kept play of S. 2016-2019 data are training only.

**Primary (exact).** Pooled over all 2020-2025 kept plays, with d = (EPA - RAPM prediction)^2 - (EPA - baseline prediction)^2, **PASS iff the upper end of the 95% game-cluster bootstrap CI of mean(d) is below 0 for BOTH the team baseline and the box baseline.**

**Secondaries** (each reported; none changes the primary verdict):
- S1: team-game EPA-per-play margin MSE from the actual lineups; RAPM minus team and RAPM minus box, paired bootstrap over games; pass iff the CI upper bound is below 0 (reported separately for each).
- S2: the per-season table (MSE of each model and RAPM-minus-baseline CIs for each of 2020-2025). Descriptive.
- S3: year-to-year correlation of player values by position group, chained and single-season fits, 2016-2025. Descriptive.
- S4: game model, descriptive: A4s logit + b x `rapm_lineup_delta_diff` (b refit for each S on REG 2016..S-1) against A4s on REG 2020-2025 games with moneylines (the M3 sign-off game set: n = 1,615, games hash `1b27abff81bddb2c` expected), paired Brier CI.

**Procedure.** First `python scripts/ml_m5b.py signoff-dry-run`: the same code path on dev 2018-2019 without logging; it must print REPRODUCED against `data/raw/ml/m5b/dev.json` (run `ml_m5b.py dev` on the committed code first). Then, once, `python scripts/ml_m5b.py signoff`: it refuses to start if any `m5b_signoff_*` run with `holdout: true` exists in `experiments/runs/`, prints every number and each rule's verdict, writes `data/raw/ml/m5b/signoff_holdout.json`, and logs `m5b_signoff_rapm`, `_team`, `_box`, `_intercept` and `_game_model` with `holdout: true`. Whatever happens is recorded in this file.

### M3b build notes (2026-10-05, dev only; holdout not run)

Spec: `context/ml-m3b-method.md` (approved under delegation). Only CC BY data: play-by-play, schedules, and the nflverse players table's birth dates. No participation, no snap counts, nothing PFR-derived.

**Where things are.** `nflelo/ml/features/kalman.py` (state-space team strength: `game_table`, `run_filter`, the two likelihoods, `tune`, `build_features`, `TUNED`), `nflelo/ml/features/qb.py` (extended behind flags: `QBExtras`, `build_features_ext`, `ext_states`, `score_values`, `TUNED_EXTRAS`, `TUNED_EXPERIENCE_CHANGED`; with no active extras the M3 code path runs untouched), `scripts/ml_m3b.py` (`tune`, `dev`, `leakcheck`, `signoff-dry-run`, `signoff`), `tests/ml/test_m3b.py`, `notebooks/07_game_model_refinement.ipynb`. Small edits: `asof.corrupt_from` now also scrambles `cpoe` on late plays; `qb.dropback_plays` carries a `cpoe` column when the play-by-play has one (M3's aggregation ignores it). Saved outputs: `data/raw/ml/m3b/` (gitignored: `tune.json`, `dev.json`, `leakcheck.json`, `ladder_frame.parquet`, `coefs.parquet`).

**C2, the Kalman filter.** One latent strength per franchise (points vs average) plus home field, full 33 x 33 covariance, run from 1999 week 1 (every team at 0, SD 6). Each week: time update (weekly random-walk drift; at a new season, the mean is multiplied by gamma and off-season variance is added), then a prediction for every game of the week from the pre-week state (the as-of rule), then sequential updates with two measurements per game: the point margin (d + h + noise) and the raw home-minus-away EPA per play, scaled to points and with the league's home EPA edge removed (d + noise), with correlated noise. The EPA margin is raw on purpose: the measurement is about the difference between the two teams, so the filter adjusts for the opponent itself. One strength per team, not offense and defense: margins only identify the difference. Home field is a drifting state rather than a constant, because the home edge has fallen since 2000 (Elo v2 learns it online for the same reason); the tuned drift is tiny, so in practice it is a slowly updated estimate, and the logistic intercept, refit each season, absorbs the rest.
- Tuning (2000-2005 REG and POST, 1,586 games; 1999 burn-in; DEV never read). The margin-only likelihood the spec names turned out to be flat along a ridge in (scale, sigma_e, rho): four starts gave log likelihoods from -6,372.6 to -6,365.9 with scale anywhere from 29 to 91 and rho from -0.88 to 0.98. The objective is therefore the **joint** predictive density of both measurements (with the Jacobian of the EPA scale), which gives the same answer from all four starts (-6,071.9 to -6,072.2); its margin-only log likelihood is -6,368.6, within 2.7 of the best margin-only fit.
- **Frozen (`kalman.TUNED`): q_week 0.722 (0.85 points of drift per week, 3.5 per season), gamma 0.483, q_season 14.4 (3.8 points), sigma_m 12.36, sigma_e 13.52, rho 0.807, scale 39.9 points per EPA/play, q_hfa 0.00034 per week (about 0.08 points per season), epa_home 0.0086** (fixed from 1999-2005). None at a bound. Margin log likelihood per game -4.016 vs -4.092 for a constant forecast; RMSE 13.42 vs 14.49 points.
- **The EPA measurement adds almost nothing.** At the tuned noise levels one game's EPA margin adds about 3% to the information in its point margin (the two share most of the game's luck). Descriptive check, not a ladder step: switching it off gives a standalone DEV Brier of 0.2165 against 0.2167 with it. The filter's value is its structure (a step size that follows the uncertainty, and stronger off-season reversion than Elo's: it keeps 48% of a team's distance from average, Elo v2 60%).
- **Standalone** (P = Phi(kf_margin / sqrt(kf_sd^2 + sigma_m^2)), no QB term): DEV Brier **0.2167**, log loss 0.6236, accuracy 0.645, ECE 0.017; vs Elo -0.0011 [-0.0029, +0.0007]; vs A4s +0.0016 [-0.0004, +0.0036]. Weeks 1-4: 0.2232 (A4s 0.2243, Elo 0.2229). Changed-starter games: 0.2248 (it has no QB information). `kf_margin` correlates 0.94 with `elo_logit` and 0.94 with `adj_epa_margin`.

**C3, the QB term** (QB targets, not game outcomes: next game's EPA per dropback for the starter, dropback-weighted MSE).
- **Draft-field license finding.** nflverse-players' `CONTRIBUTING.md` lists "Draft information: draft year, draft round, draft pick, draft team (from PFR)". The players table's draft fields are Pro-Football-Reference-derived, so **draft round is not used**; the experience prior uses career dropbacks only (counted from play-by-play before as_of). Birth dates are "basic player information ... (mostly from GSIS)", CC BY via nflverse, and are used for aging.
- (a) CPOE composite, tuned on 2007-2008 (1,024 starter-games; history from 1999, CPOE from 2006): value = EPA value + 0.01 x (shrunk CPOE - replacement CPOE), CPOE shrunk with k = 400 attempts. wMSE 0.11457 vs 0.11490. Before 2006 everyone's CPOE term is 0, so A4s's training rows for 2001-2005 are unchanged.
- (b) Experience prior, 2000-2005 (3,036 starter-games): prior = replacement level + an offset by career dropbacks [0, 100), [100, 500), [500, 2000), 2000+ (QBs already playing in 1999 go in the top bucket): -0.054, +0.016, +0.141, +0.065 EPA per dropback. wMSE 0.10839 vs 0.10918.
- (c) Aging, 2000-2005 grid (peak 26-34, slope before 0-0.20, after 0 to -0.05 per year): peak 31, +0.14 per year before, 0 after. wMSE 0.10810 vs 0.10918. The slope is far too steep to be aging; it absorbs young QBs improving faster than the 48-week memory allows. (The first grid stopped at 0.03, then 0.08; both optima were at the edge, so the grid was widened.)
- (d) **Added after (b) scored worse on DEV**, labelled as such: (b)'s offsets re-tuned only on the 351 starter-games of 2000-2005 where the team's starter changed from its previous game (the games the prior is for): -0.065, +0.080, +0.033, +0.032. wMSE 0.11445 vs 0.11588 (the all-starter offsets give 0.11505 there).

**C1.** `early = max(0, 1 - (week - 1) / w0)`; interactions with `elo_logit` (or `kf_margin`) and `adj_epa_margin`. **w0 is tuned on DEV** (grid 1, 2, 3, 4, 5, 6, 8, 10, 12, 18; Brier minus A4s from +0.0001 to +0.0004): best w0 = 1, at the grid's edge, and no w0 beats A4s.

**C2d was added** to the spec's C2a-c: in C2b, `adj_epa_margin`'s coefficient turned negative in later refits (2019: -0.90; range -0.90 to +0.52) once `kf_margin` was in, so the fourth keep/replace cell (keep Elo, replace the ridge EPA margin with `kf_margin`) was run too.

**Ladder** (DEV REG 2006-2019 with moneylines, n = 3,450, games hash `52b8194df3a2ebf4`; A4s's protocol; paired bootstrap vs A4s, 2,000 reps, seed 20261003; ECE ok = at or below the 95th percentile of a calibrated forecaster at this n). A4s reproduces 0.21509.

| Step | Model | Brier | Log loss | Acc | ECE (p95) | minus A4s [95% CI] |
|---|---|---|---|---|---|---|
| A4s | elo_logit + adj_epa_margin + qb_delta_diff | 0.2151 | 0.6197 | 0.647 | 0.011 (0.025) | |
| A4bs | A4s, last game's starter | 0.2172 | 0.6244 | 0.643 | 0.012 (0.025) | +0.0021 [+0.0010, +0.0033] |
| C1 | A4s + early interactions, w0 = 1 | 0.2152 | 0.6199 | 0.647 | 0.013 (0.025) | +0.0001 [-0.0001, +0.0002] |
| C2a | kf_margin + adj_epa_margin + qb | 0.2146 | 0.6186 | 0.654 | 0.019 (0.027) | -0.0005 [-0.0020, +0.0010] |
| C2b | elo + kf_margin + adj_epa_margin + qb | 0.2141 | 0.6175 | 0.649 | 0.013 (0.026) | -0.0009 [-0.0019, +0.0001] |
| C2c | kf_margin + qb | 0.2145 | 0.6185 | 0.651 | 0.023 (0.026) | -0.0006 [-0.0021, +0.0010] |
| **C2d** | **elo_logit + kf_margin + qb_delta_diff** | **0.2141** | **0.6175** | **0.650** | **0.014 (0.026)** | **-0.0010 [-0.0019, +0.0000]** |
| C2k | kf_margin alone (logistic) | 0.2167 | 0.6235 | 0.647 | 0.015 (0.025) | +0.0016 [-0.0005, +0.0035] |
| KF | filter alone, no logistic | 0.2167 | 0.6236 | 0.645 | 0.017 (0.025) | +0.0016 [-0.0004, +0.0036] |
| C3a | A4s, CPOE composite | 0.2150 | 0.6196 | 0.644 | 0.016 (0.025) | -0.0000 [-0.0002, +0.0001] |
| C3b | A4s, experience prior | 0.2157 | 0.6210 | 0.643 | 0.014 (0.026) | +0.0006 [+0.0000, +0.0013] |
| C3c | A4s, aging | 0.2155 | 0.6206 | 0.643 | 0.015 (0.026) | +0.0004 [+0.0000, +0.0008] |
| C3d | A4s, experience prior tuned on changed starters | 0.2152 | 0.6199 | 0.647 | 0.013 (0.025) | +0.0001 [-0.0004, +0.0006] |
| C3abc | A4s, CPOE + experience + aging | 0.2161 | 0.6217 | 0.646 | 0.015 (0.025) | +0.0010 [+0.0003, +0.0016] |
| C4 | C2d + C3a | 0.2141 | 0.6175 | 0.650 | 0.015 (0.026) | -0.0010 [-0.0019, -0.0000] |
| | Elo v2 | 0.2178 | 0.6255 | 0.641 | 0.015 | |
| | Market | 0.2106 | 0.6096 | 0.663 | 0.018 | |

Every ECE passes the one-sided check. Leak alarm (better than A4s by more than 0.006): none. C4 combines the changes that beat A4s on their own: the best C2 (C2d) and the best C3 (C3a; C1 did not beat A4s).

**Splits** (Brier; minus A4s [95% CI]). A4s: weeks 1-4 0.2243 (market 0.2198, n 751), 5-9 0.2126 (0.2095, n 969), 10-18 0.2125 (0.2072, n 1,730); changed starter 0.2150 (market 0.2072, n 664; the starter differs from the team's previous game for either side), unchanged 0.2151 (0.2114, n 2,786). (The M3b spec quotes 0.2142 for A4s on changed-starter games; this definition gives 0.2150.)

| Step | Weeks 1-4 | Weeks 5-9 | Weeks 10-18 | Changed starter | Unchanged |
|---|---|---|---|---|---|
| C1 | +0.0003 [-0.0004, +0.0010] | +0.0000 | +0.0000 | +0.0003 [-0.0003, +0.0009] | +0.0000 |
| C2a | +0.0002 [-0.0026, +0.0028] | +0.0012 | -0.0017 [-0.0038, +0.0005] | +0.0008 [-0.0026, +0.0043] | -0.0008 |
| C2b | -0.0005 [-0.0022, +0.0012] | -0.0001 | -0.0016 [-0.0030, -0.0001] | -0.0001 [-0.0023, +0.0020] | -0.0011 |
| C2c | -0.0000 [-0.0028, +0.0026] | +0.0011 | -0.0017 [-0.0038, +0.0005] | +0.0010 [-0.0023, +0.0045] | -0.0009 |
| **C2d** | **-0.0006 [-0.0023, +0.0011]** | **-0.0002** | **-0.0016 [-0.0029, -0.0002]** | **-0.0000 [-0.0021, +0.0021]** | **-0.0012 [-0.0022, -0.0001]** |
| KF | -0.0011 [-0.0049, +0.0026] | +0.0026 | +0.0023 | +0.0098 [+0.0037, +0.0162] | -0.0003 |
| C3a | +0.0001 | -0.0000 | -0.0001 | +0.0001 [-0.0003, +0.0006] | -0.0001 |
| C3b | +0.0013 [-0.0003, +0.0028] | +0.0003 | +0.0006 | +0.0027 [+0.0001, +0.0052] | +0.0002 |
| C3c | -0.0001 | +0.0006 | +0.0006 [+0.0001, +0.0011] | +0.0007 [-0.0007, +0.0021] | +0.0004 |
| C3d | +0.0002 | -0.0001 | +0.0002 | +0.0002 [-0.0019, +0.0023] | +0.0001 |
| C3abc | +0.0011 | +0.0008 | +0.0010 [+0.0001, +0.0019] | +0.0031 [+0.0004, +0.0056] | +0.0005 |
| C4 | -0.0004 | -0.0002 | -0.0017 [-0.0030, -0.0003] | +0.0001 [-0.0020, +0.0022] | -0.0012 |

On changed-starter games every step stays about 0.008 behind the market (C2d 0.2150 vs 0.2072).

**Choice (one-SE rule).** Best DEV Brier: C4 (0.21410). Within one SE: C2b, C2d, C4. **Chosen: C2d** (3 features, no QB extras). C2d minus A4s **-0.00095 [-0.00191, +0.000001]**: the interval just includes zero, so **the M3b acceptance bar (CI excludes zero) is not met on DEV**, narrowly. Log loss: -0.0022 [-0.0043, -0.0001]. vs Elo -0.0037 [-0.0055, -0.0019]; vs market +0.0035 [+0.0013, +0.0059]. Wednesday twin (C2d with last game's starter): 0.2161, minus A4bs -0.0011 [-0.0021, -0.0001].
- C2d coefficients, 2019 refit (trained 2001-2018; range over the 14 refits): intercept -0.014 [-0.051, -0.014], elo_logit +0.433 [+0.176, +0.496], kf_margin +0.083 [+0.068, +0.114] per point, qb_delta_diff +3.495 [+2.592, +3.665]. Every refit has the expected signs. Per 1 SD (2019 fit, log-odds): kf_margin +0.49, elo_logit +0.33, qb_delta_diff +0.22.
- Per season, C2d minus A4s ranges from -0.0057 (2013) to +0.0021 (2015); 8 of 14 seasons favor C2d (2019 by -0.00001).

**Read.** The Kalman filter is the one real gain, a small one (0.001 Brier, borderline), and it comes from how the filter weighs results over time, not from EPA. It replaces the ridge EPA margin rather than Elo: Elo's long memory still helps next to it. Neither weakness that motivated M3b moved: the early season (weeks 1-4) is unchanged within noise under every candidate, and the changed-starter gap to the market (about 0.008) is not closed by any QB extra. Each QB extra improves the QB-EPA target it was tuned on and none improves game predictions; the market's edge on QB changes comes from information the play-by-play doesn't have (practice reports, the coach's plan, how hurt the starter is).

**Leakage.** The filter and every QB-extras variant pass `asof.leakage_check` on synthetic data (tests) and on real 2014-2017 data (`ml_m3b.py leakcheck`: 8 games from 2017 weeks 1, 2, 9, 17 and a wild-card game; Kalman 8/8 leak-free; QB variants a, b, c, d, abc with the actual starter (identity kept, M3-D1) and the last starter (identities scrambled) 8/8 each). Positive control, the filter's post-week margin: 0/8 leak-free, as required. A test checks that with no extras the QB builder's output equals M3's exactly (A4s unchanged).

**Registry.** `m3b_tune_kalman`, `m3b_tune_qb` (label tune), and `m3b_A4s`, `m3b_A4bs`, `m3b_C1`, `m3b_C2a`-`d`, `m3b_C2k`, `m3b_KF`, `m3b_C3a`-`d`, `m3b_C3abc`, `m3b_C4_C2d+C3a`, `m3b_C2d_last` (label dev, 2006-2019), logged from an uncommitted tree; re-log after the commit if a clean-commit record is wanted. `signoff-dry-run` reproduces `dev.json` exactly (max |Brier diff| 0).

**Timing.** `tune` 246 s (Kalman 227 s: four joint-likelihood starts plus four margin-only reference fits, about 0.03 s per filter pass over 1999-2005; QB extras 18 s). `dev` 18.5 s (features 10.7 s: Kalman 0.2 s, QB extras 6 s). `leakcheck` 18 s. `signoff-dry-run` 14 s. A full filter pass over 1999-2019 takes about 0.1 s, so a weekly update is trivial.

**Open items.**
- The spec's acceptance bar is missed by a hair (CI upper bound +0.000001). Whether to spend the holdout run on C2d is Walker's call; the pre-registration below is written so the run is ready if he says yes.
- The margin-only tuning objective in the spec was replaced by the joint likelihood (reason above).
- C2d and C3d were added beyond the spec; C3d after seeing C3b's DEV result.
- C1's w0 is DEV-tuned and sits at the grid's edge (1).

**Shadow deployment (2026-10-05).** C2d runs as a shadow model in the live 2026 pipeline, logged before kickoff and scored next to A4s. It never replaces A4s: the site's picks, the main ledger and the scorecard numbers stay A4s, and it was deployed before (and independently of) the M3b holdout run, which is still not run. The shadow's live record is new out-of-sample data, not a holdout score.
- *Fit.* `scripts/ml_predict.py` fits C2d once per season on REG 2001..season-1 (the A4s protocol: unpenalized logistic, season-decay weights with half-life 8, ties as two half rows) on the M3 frame plus `kf_margin` from `kalman.build_features(..., kalman.TUNED)`, the same path as `ml_m3b.py build_frame`. It is frozen in `experiments/live/shadow_2026.json` with `kalman.TUNED`, the spec, the training hash and the data fingerprint. Before fitting, it refits 2019 through the same code and stops unless the coefficients match the logged DEV fit (`experiments/runs/20261005T062809Z_m3b_C2d.json`); max |diff| was 0. 2026 fit: **`C2d-2026-6b53b499`**, n 6,471, intercept -0.0538, elo_logit 0.534, kf_margin 0.0643, qb_delta_diff 3.625. `--fit-shadow-only` fits it alone; `--refit-shadow` refits; `--no-shadow` skips it.
- *Kalman state as of now* (`live_features.kalman_asof`, `kalman.predict_at`). The same as-of rule as A4s: a game in the week in progress uses the state before that week's as_of; every later game uses the first week that hasn't started, i.e. every completed REG/POST game before now. The filter then takes the time update it would make before that week in training, so a played game's live prediction equals the filter's own pre-week prediction bit for bit (tested). Betting columns are dropped on entry (D3).
- *Ledger.* `experiments/live/<season>_shadow.csv`, append-only, columns `run_at_utc, game_id, kickoff_utc, model_version, p_home_shadow, kf_margin, kf_sd, data_hash`, written in the same run as the main ledger rows (same `run_at_utc` and data hash), with the main ledger's rules (rows at or after kickoff refused, all or nothing, runs move forward). The main ledger's schema and rows do not depend on the shadow: a test checks they are byte-identical with and without it (the real 2026-10-05 data gave the same result). A shadow failure is reported and skipped and never blocks the A4s record; a failed reproduction check or a spec mismatch still stops the run, like the A4s and margin models.
- *Scoring* (`live.shadow_record`). Each model's last row before kickoff, on exactly the games A4s is scored on (completed, week 5 on, with a main-ledger row); `missing` counts A4s-scored games with no shadow row, and A4s is re-scored on the paired games. Reports Brier and accuracy for both, the paired Brier difference (shadow minus A4s) with its standard error, and the counts. The exporter puts it in `ml.json` as `shadow` (null without the shadow ledger).
- *First real run.* 2026-10-05 06:37 UTC, a normal full run after `build.py` brought in the Sunday week 4 results (the 03:29 run had predicted before nflverse posted them, so 196 of the 209 predictions moved, by up to 3.6 points): 209 rows in each ledger, 32 simulation rows. The shadow differs from A4s by 1.5 percentage points on average (max 5.5).

### M3b holdout pre-registration (proposed 2026-10-05; not yet run; awaiting Walker)

One holdout run, scored once, with everything frozen in committed code. The result is recorded here whatever it shows; nothing is retuned afterwards. Note before deciding: **C2d did not clear the M3b bar on DEV** (CI [-0.0019, +0.0000] includes zero, narrowly).

**Frozen.** Model C2d = logistic on `elo_logit`, `kf_margin`, `qb_delta_diff` (actual starter, M3-D1), A4s's protocol: walk-forward, season S fit on REG 2001..S-1 with weights 0.5 ** ((S - 1 - season) / 8), ties as two half rows, unpenalized. `kf_margin` from `kalman.TUNED` (q_week 0.72158, gamma 0.48317, q_season 14.431, sigma_m 12.361, sigma_e 13.52, rho 0.80709, scale 39.921, q_hfa 0.00034383, epa_home 0.0085843; p0 36, h0 0, ph0 9), filter run from 1999 on REG and POST games. M3 knobs unchanged (lambda 100, half-life 48, rho 0.25, QB k 100). Elo v2 `DEFAULT_CONFIG`. Spec frozen in `scripts/ml_m3b.py` (`CHOSEN`, `SIGNOFF`).

**Games.** REG 2020-2025 with moneylines: n = 1,615, games hash `1b27abff81bddb2c`. The run stops if either differs.

**Primary (exact).** d = Brier(C2d) - Brier(A4s) on those games. **PASS iff the upper end of the 95% paired bootstrap CI (2,000 reps, seed 20261003) is below 0 AND C2d's ECE (10 uniform bins) is at or below the 95th percentile of a perfectly calibrated forecaster at the same n** (`calibration.ece_null_range`, 500 draws, seed 20261003; one-sided).

**Secondaries** (each reported; none changes the primary verdict):
- Weeks 1-4: C2d minus A4s and C2d minus market, paired CIs.
- Changed-starter games (the starter differs from the team's previous game, either side) and unchanged games: same comparisons.
- The Wednesday-only twin (C2d with last game's starter) minus A4bs, and minus A4s.
- Descriptive: C2d minus Elo and minus market, log loss, weeks 5-9 and 10-18, per season, the 2025-fit coefficients.

**Procedure.** First `python scripts/ml_m3b.py signoff-dry-run` on the committed code: the same code path on DEV without logging; it must print REPRODUCED against `data/raw/ml/m3b/dev.json` (run `ml_m3b.py dev` on the committed code first). Then, once and only with Walker's OK, `python scripts/ml_m3b.py signoff`: it refuses to start if any `m3b_signoff_*` run with `holdout: true` exists in `experiments/runs/`, prints every number and the verdict, writes `data/raw/ml/m3b/signoff_holdout.json`, and logs `m3b_signoff_{chosen, A4s, twin, A4bs, elo_v2, market}` with `holdout: true`. If it passes, C2d runs as a shadow model in the weekly ledger; A4s stays the 2026 headline.

### M6 build notes (2026-10-05, dev only; holdout not run)

Spec: `context/ml-m6-method.md` (approved under delegation). Everything here is **CC BY-SA 4.0** (participation-derived): the module docstrings say so, `data/raw/ml/m6/LICENSE.txt` covers the saved tables, networks and predictions, and every run record carries a `license` key. `tests/ml/test_m6.py` keeps `nflelo.ml.plays` from being imported anywhere outside M6, and `tests/ml/test_m5.py` lists the M6 modules among the only readers of participation, so nothing reaches A4s or the live CC BY artifacts.

**Where things are.** `nflelo/ml/plays/` (`data.py` play table, allowlist and bins; `metrics.py` CRPS and calibration; `baseline.py`; `gbm.py`; `net.py`), `scripts/ml_m6.py` (`fields`, `dev`, `leakcheck`, `signoff-dry-run`, `signoff`), `tests/ml/test_m6.py`, `notebooks/08_play_model.ipynb`. Also: `eval/windows.py` gained `M6_DEV = (2018, 2019)` (label `m6dev`, guarded like DEV); `data.load_participation` gained a `columns` argument (M6 reads only the pre-snap participation columns). PyTorch is in a new `requirements-m6.txt` (`-r requirements-ml.txt` plus `torch>=2.4`), **not** in `requirements-ml.txt`: the weekly GitHub Action installs that file, and on Linux the default torch wheel is the multi-GB CUDA build. Installed locally: torch 2.14.1 (CPU). Saved outputs: `data/raw/ml/m6/` (gitignored): `dev.json`, `dev_predictions.parquet`, `fields.json`, `leakcheck.json`, `signoff_m6dev.json`, and the 12 dev networks `net_{view}_{N0,N1,N2}_{2018,2019}.pt`.

**Play table.** The M5b kept plays (M1 scrimmage snaps, REG and POST, garbage time out, 11 + 11 GSIS IDs after the side repair), joined to play-by-play and to the participation pre-snap fields. Features are an explicit allowlist, the same for every model:
- situation: down, ydstogo, yardline_100, score differential, seconds left in the half and game, both teams' timeouts, home, roof (indoor / open), temperature, wind;
- structure: offensive RB / TE / WR / OL counts, defensive DL / LB / DB counts, formation (shotgun / pistol / under center), box count;
- ratings: the offense's M3 adjusted pass and rush offense and the defense's adjusted pass and rush defense, as of the play's week start (`oa.TUNED`, `compute_ratings` at the week's as_of);
- call view only: play type (nflfastR `pass`, so sacks and scrambles are passes).

`check_features` rejects anything off the allowlist or matching a banned post-snap or outcome pattern (pass rushers, coverage, man/zone, pressure, routes, time to throw, air yards, YAC, EPA, WPA, `*_result`, yards, touchdown, interception, fumble, sack, first down, ...); every model calls it. The builder reads only `PART_COLUMNS` from participation, so the post-snap columns are never loaded. Target: 53 bins (-10 or worse, -9..+40, 41+, TD) with the field fold (from yard line y, bins at y yards or more go to TD; losses past the own goal line go to the lowest possible bin); turnover = interception or lost fumble.

Dev data (2016-2019, kept plays): 28,367 / 28,051 / 27,611 / 27,932 plays, 267 games a season. Pass share 0.61-0.62; mean yards 5.4-5.7; exactly 0 yards 24-25%; TD 3.5-4.0%; turnover 1.96-2.04%; first down or TD 28-30%; 20+ yards 6.0-6.6%; loss 8.0-8.9%. The only missing features are temperature and wind (indoors, 24-27% of plays).

**Data finding: outcome-dependent missingness (fixed).** In the NGS seasons the participation formation is missing on about 1% of plays (2016: 259 of 28,367), and **83-89% of those plays are fumbles** (strip sacks; 2016-2018; 29-35% are sacks). The first model run learned it: a GBM on the features predicted turnovers with AUC 0.81 (Brier 0.0142 against 0.0199 for a constant), with formation the top feature by permutation importance. Fix: missing formation is filled from play-by-play's own `shotgun` flag (CC BY; it agrees with the participation formation on 98.6-99.7% of plays), a missing box count with the typical box for the formation (7 under center, 6 otherwise), missing personnel with 11 against nickel; no missing-value flag is a feature except indoor weather. After the fix the same check gives AUC 0.66 (play type is nearly all of it) and Brier 0.0198 (the constant's). A test enforces "no missing structure features".

**Source shift (2023, NGS to FTN), from `ml_m6.py fields`** (pre-snap fields on kept plays, 2016-2025; no outcome read). The raw formation vocabulary drops from 7 names (SHOTGUN, EMPTY, SINGLEBACK, I_FORM, PISTOL, JUMBO, WILDCAT) to 3 (SHOTGUN, UNDER CENTER, PISTOL), and personnel switches from group counts ("1 RB, 1 TE, 3 WR"; "6 OL" only with an extra lineman) to roster positions (C, G, T, FB; CB, DE, DT, NT, ILB, MLB, OLB, FS, SS). After the harmonized encoding the personnel means are steady across the switch (RB 1.08-1.12, TE 1.24-1.36, WR 2.48-2.61, OL 5.02-5.06, DL 3.07-3.41, LB 2.71-3.15, DB 4.77-4.89 every season). Formation shares move some (shotgun 0.57-0.63 in 2016-2022, 0.68 in 2023; pistol 0.016-0.045, then 0.096 in 2024). **The box count shifts:** mean 6.34-6.49 and SD 1.01-1.08 in 2016-2022; 6.27 / 1.09 in 2023, 6.10 / 0.93 in 2024, 6.23 / 0.84 in 2025; the 8+ box share halves from 0.11-0.16 to 0.05 in 2024-2025. No era indicator is used (a model predicting 2023 has never seen an FTN season, so it could not learn one); the holdout reports the NGS and FTN seasons separately instead.

**Models** (season-ahead for S: fit on 2016..S-1; settings tuned on 2016..S-2 against S-1, then refit on 2016..S-1).
- Baseline: raw-yard histograms per down x distance (1, 2, 3, 4-6, 7-9, 10, 11-15, 16+) x field zone (1-5, 6-10, 11-20, 21-40, 41-60, 61-80, 81-90, 91-99) (x play type in the call view), smoothed toward down x distance, down, and all plays with strength k, folded per play. k = 320 every time (grid 20-5,120, interior).
- GBM: `HistGradientBoostingClassifier` over the 53 bins plus a binary turnover model; grid learning rate {0.1, 0.05} x leaves {7, 15}, min leaf 200, L2 1, early stopping on S-1; chosen by S-1 CRPS. Picks: 0.1 / 7 leaves (32-46 iterations), and 0.05 / 7 (79) for situation 2019.
- Networks (PyTorch, CPU, 4 threads, deterministic, seed 20261003): trunk 2 x 128 ReLU with dropout 0.1; output biases started at the training marginals; folded-softmax NLL + turnover BCE; AdamW (weight decay 1e-4), batch 512, early stopping on S-1 (patience 4). N0: no players. N1: 8-dim embeddings per (side, player), zero-initialized, DeepSets (shared 32-unit phi, mean over the 11), separate offense and defense. N2: N1 plus each player's M5 box value and M5b RAPM value as of the start of the play's season (data before that season only) and a has-box-record flag, fed to phi. Trunk grid {lr 1e-3, 128}, {3e-4, 128}, {1e-3, 256} by S-1 loss (situation picked 1e-3 / 128, call 3e-4 / 128); N1 and N2 reuse N0's trunk and tune the embedding penalty over {1e-3, 1e-2, 1e-1} (picks 1e-2 or 1e-1). Epochs 3-15.

**Dev tables** (season-ahead 2018 and 2019, all kept REG and POST plays: 55,543 plays, 534 games, games hash `2e7aa347318c6509`, the same plays as the M5b dev; CIs resample games, 2,000 reps, seed 20261003; ECE one-sided null p95 about 0.0056 at this n; CRPS in yards).

Situation view (no play type):

| Model | CRPS | vs baseline [95% CI] | Log loss | Turnover Brier | ECE P(first) | one-SE |
|---|---|---|---|---|---|---|
| Baseline | 4.0245 | | 2.9731 | 0.01991 | 0.0061 (fails) | |
| **GBM** | **4.0059** | **-0.0185 [-0.0217, -0.0155]** | 2.9442 | 0.01984 | 0.0052 (passes) | **best, chosen** |
| N0 | 4.0092 | -0.0152 [-0.0188, -0.0117] | 2.9454 | 0.01987 | 0.0053 | |
| N1 | 4.0096 | -0.0148 [-0.0187, -0.0110] | 2.9445 | 0.01986 | 0.0052 | |
| N2 | 4.0098 | -0.0146 [-0.0185, -0.0106] | 2.9445 | 0.01986 | 0.0051 | |

Call-conditioned view (play type given):

| Model | CRPS | vs baseline [95% CI] | Log loss | Turnover Brier | ECE P(first) | one-SE |
|---|---|---|---|---|---|---|
| Baseline | 3.9484 | | 2.8710 | 0.01983 | 0.0046 (passes) | |
| **GBM** | **3.9304** | **-0.0180 [-0.0208, -0.0152]** | 2.8528 | 0.01976 | **0.0058 (fails; p95 0.0056)** | within, **chosen** |
| N0 | 3.9324 | -0.0160 [-0.0195, -0.0125] | 2.8513 | 0.01981 | 0.0060 (fails) | |
| N1 | 3.9310 | -0.0174 [-0.0213, -0.0139] | 2.8492 | 0.01980 | 0.0053 | |
| N2 | 3.9301 | -0.0183 [-0.0224, -0.0145] | 2.8493 | 0.01980 | 0.0042 | best |

- Every model beats the baseline by 0.015-0.019 yards of CRPS (about 0.4-0.5%), in both seasons and in every split with enough plays (run, pass, downs 1-3, every field zone; 4th down, 743 plays, is within noise). Most of a play's outcome is not knowable before the snap.
- **Between models:** situation, GBM minus N0 -0.0033 [-0.0055, -0.0011] (GBM better); call, GBM minus N0 -0.0020 [-0.0046, +0.0007], GBM minus N2 +0.0003 [-0.0027, +0.0033] (tied). By split the GBM is stronger on passes (situation: -0.0248 vs N0 -0.0137) and the networks on runs (situation: N1 -0.0198 vs GBM -0.0083).
- **Player-identity ablation:** situation view N1 minus N0 +0.0004 [-0.0009, +0.0018], N2 minus N0 +0.0006 [-0.0008, +0.0021] (no gain); call view N1 minus N0 -0.0014 [-0.0030, +0.0002], N2 minus N0 -0.0023 [-0.0040, -0.0004]. Knowing the 22 players adds at most about 0.002 yards of CRPS once personnel, formation, box and team ratings are known, and only when the play type is given. That agrees with M5b: individual on-field effects are small next to the play-level noise.
- **Embeddings vs snaps** (N1): norms fall with training snaps (Spearman -0.52 to -0.77; for example call 2019: under 100 snaps 0.036, 1,000-2,999 snaps 0.004). The penalty is per appearance, so rarely seen players are shrunk least and carry the largest (noisiest) vectors; a per-player penalty is the obvious next try if identities are revisited.
- **Calibration:** P(first down or TD) passes for GBM, N0, N1 and N2 in the situation view and for N1 and N2 in the call view; **the call-view GBM misses by 0.0002** (ECE 0.0058 vs p95 0.0056), as does N0 (0.0060), while the call-view baseline passes (0.0046). P(20+) fails the one-sided null for every model, the baseline included (ECE 0.0026-0.0061 against p95 0.0018-0.0024 at n = 55,543); P(loss) fails for every model except the situation-view baseline and GBM (ECE 0.0004-0.0072 against p95 0.0023-0.0027). At this n the null band is narrow, so gaps of a few tenths of a percentage point fail it. Turnover calibration passes except for the situation-view N0 and N2. Turnover Brier improves on the baseline by only 0.00002-0.00007. These null-only verdicts are descriptive: under the M6 calibration rule (pass if ECE <= null p95 **or** ECE <= 0.010, adopted 2026-10-05; see the pre-registration) every calibration check of every model passes on DEV (the largest ECE is 0.0072).
- **One-SE pick (complexity order baseline, GBM, N0, N1, N2): GBM in both views.** Dev acceptance preview (the pre-registered rule applied to dev): **PASS in both views** with the 0.010 ECE floor; on the null-only test the call view would fail on calibration alone (CRPS CI well below 0).

**Leakage.** `ml_m6.py leakcheck` (real 2019): every outcome and post-snap field of 2019 scrambled in the raw play-by-play and participation (all participation columns loaded), table rebuilt, every model refit on 2016-2018 and 2019 predicted in the call view: the 2019 feature matrix is identical (98% of the 2019 bins changed), all five models' predictions identical (max diff 0.0), and the positive control (a GBM also given the play's own yards) changes (max 1.0; `check_features` rejects that column). Rating features: corrupting everything from a week's as_of (2019 weeks 2, 11, 19) leaves the week's ratings unchanged; the next week's ratings (the control) move by 0.12-0.27. The same checks run synthetic in `tests/ml/test_m6.py`, plus the banned-columns test, DeepSets permutation invariance and determinism. The N2 priors come from data before each season (`ml_m5b.Ctx`, covered by `ml_m5b.py leakcheck`). `signoff-dry-run` reproduces `dev.json` exactly (max |diff| 0).

**Registry.** `m6_{situation,call}_{baseline,gbm,N0,N1,N2}` (label `m6dev`, window REG+POST 2018-2019), logged from an uncommitted tree, before the calibration floor was added (their `acceptance` fields use the null-only rule; every metric is unchanged). `dev.json` was rewritten under the new rule with `--no-log`. Re-log after the commit for a clean-commit record.

**Timing** (Apple silicon, CPU): inputs 29 s; per view and season 44-124 s (baseline 0.3 s; GBM grid and refit 24-68 s; N0 3-5 s; N1 8-22 s; N2 7-28 s; a final network fit 0.5-8 s); scoring 27 s; `dev` about 6 minutes end to end; `leakcheck` 4.5 minutes; `signoff-dry-run` 6 minutes; `fields` 2 s.

### M6 holdout pre-registration (proposed 2026-10-05; not yet run)

One holdout run, scored once, with everything frozen in committed code. The result is recorded here whatever it shows; nothing is retuned afterwards.

**Frozen.** The code as committed: `plays/data.py` (allowlist `FEATURES`, the fill rules for missing formation, box and personnel, bins and fold), `plays/baseline.py` (`K_GRID`), `plays/gbm.py` (`GRID`, `FIXED`, `TOV_PARAMS`, `MAX_ITER` 400, seed 20261003), `plays/net.py` (`NetConfig` defaults, `TRUNK_GRID`, `EMB_GRID`, 4 threads, deterministic), and the tuning protocol (settings fit on 2016..S-2 and chosen on S-1, then refit on 2016..S-1). Plays: the M5b cleaning as committed (expected 177,545 kept plays in 1,693 games for 2020-2025, from the M5b counts, if nflverse has not revised the files). Team ratings `oa.TUNED` as of each week's start. N2 priors: `ml_m5b.Ctx` box values and `rapm.TUNED` RAPM fits, as of each season's start. Bootstraps: 2,000 replicates, seed 20261003, resampling games; ECE null: 500 draws, seed 20261003.

**Model per view (fixed from DEV): GBM in the situation view and GBM in the call view** (`ml_m6.DEV_CHOSEN`, the DEV one-SE picks). The holdout's own one-SE pick is reported but does not replace them.

**Window.** Season-ahead for each S in 2020-2025, trained on 2016..S-1, predicting every kept play of S. 2016-2019 are training only.

**Primary (exact).** For each view, pooled over all 2020-2025 kept plays, with d = CRPS(GBM) - CRPS(baseline) per play: the view **passes iff the upper end of the 95% game-cluster bootstrap CI of mean(d) is below 0 AND the GBM's P(first down or TD) calibration check passes: ECE (10 uniform bins) at or below the 95th percentile of a perfectly calibrated forecaster at the same n** (`calibration.ece_null_range`; one-sided) **OR at or below 0.010** (`metrics.ECE_FLOOR`, one percentage point). **M6 PASSES iff both views pass.** The same floor applies to every pass/fail calibration check reported in the secondaries; the null-only verdict is reported alongside as a descriptive field (`ok_null`).

**Secondaries** (each reported; none changes the primary verdict):
- S1, player-identity ablation: N1 minus N0 and N2 minus N0 CRPS, both views, game-cluster CIs (a finding either way).
- S2, GBM against the networks: GBM minus N0, N1, N2.
- S3, splits: run / pass, downs 1-4, field zones (1-20, 21-50, 51-80, 81-99), and **era: 2020-2022 (NGS) vs 2023-2025 (FTN)**, every model minus the baseline with CIs.
- S4, per season 2020-2025, and the holdout's own one-SE table.
- S5, log loss; calibration of P(20+), P(loss) and P(turnover); turnover Brier vs the baseline.

**Rule change, 2026-10-05 (Claude, under Walker's standing delegation), made before any holdout number was computed:** every M6 pass/fail calibration rule passes if ECE <= null p95 **or ECE <= 0.010**. Reason: at n of about 55,000-180,000 plays the calibrated-null band is only about 0.005 wide, so the null test alone flags miscalibration far too small to matter; on DEV even the baseline fails it (situation view P(first down) ECE 0.0061 against p95 0.0056). The null-only result stays in every record as a descriptive field.

**Procedure.** First `python scripts/ml_m6.py dev` and then `python scripts/ml_m6.py signoff-dry-run` on the committed code: the dry run is the sign-off code path on DEV without logging and must print REPRODUCED against `data/raw/ml/m6/dev.json`. Then, once, under Walker's standing delegation (CONTEXT.md, 2026-10-05), `python scripts/ml_m6.py signoff`: it refuses to start if any `m6_signoff_*` run with `holdout: true` exists in `experiments/runs/`, prints every number and each rule's verdict, writes `data/raw/ml/m6/signoff_holdout.json`, and logs `m6_signoff_{situation,call}_{baseline,gbm,N0,N1,N2}` with `holdout: true`. Whatever happens is recorded in this file.

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
| 2026-10-05 | M5 method approved (`context/ml-m5-method.md`). D1: CC BY only, so playing time comes from usage stats and participation is used only for validation. D2: the final injury report and game-day inactives count as pregame facts, and the Friday-only version (B3f) is always reported. D3: the evaluation window is 2012-2019, compared with A4s on the same games. D4: M5 runs as a shadow model in 2026 and can be promoted in 2027. |
| 2026-10-05 | M5b built under the standing delegation (dev only). Participation sides are rebuilt from `players_on_play` (raw drop rate 12.7% in 2016 and 7.9% in 2018, under 0.5% after). `box_scale` (the prior is 0.25 x the box value) was added to the tuning. Settings frozen in `rapm.TUNED`; holdout pre-registration proposed in "M5b holdout pre-registration", awaiting review before the one run. |
| 2026-10-05 | M3b built under the standing delegation (dev only). The nflverse players table's draft fields come from Pro-Football-Reference (nflverse-players CONTRIBUTING.md), so draft round is not used anywhere; the QB experience prior uses career dropbacks only. The Kalman filter is tuned by the joint likelihood of margin and EPA margin (the margin-only likelihood is flat in the EPA knobs). C2d (Elo + `kf_margin` + QB delta) is the one-SE pick but misses the CI bar narrowly; its holdout pre-registration is proposed in "M3b holdout pre-registration" and not run. |
| 2026-10-05 | C2d deployed as a shadow model in the live 2026 pipeline (before and independent of the M3b holdout, which is still not run): frozen fit `C2d-2026-6b53b499` in `experiments/live/shadow_2026.json` (2019 refit reproduces the logged DEV fit exactly), its own append-only ledger `experiments/live/2026_shadow.csv`, scored on the same games as A4s, one line on the site's scorecard. A4s stays the 2026 model; the main ledger is byte-identical with or without the shadow. See "Shadow deployment" in the M3b notes. |
| 2026-10-05 | M6 built under the standing delegation (dev only). NGS participation leaves the formation missing on fumble plays (83-89% of formation-missing plays, 2016-2018), an outcome leak: missing formations are filled from play-by-play's `shotgun` flag and no missingness flag is a feature. The 2023 NGS-to-FTN switch is handled by one era-free encoding of formation and personnel; the box count still shifts (8+ boxes halve in 2024-2025), so the holdout reports the eras separately. PyTorch goes in a new `requirements-m6.txt`, not `requirements-ml.txt`, so the weekly Action does not install it. The holdout model is fixed from DEV (GBM, both views); pre-registration proposed in "M6 holdout pre-registration", not run. Calibration rules pass if ECE <= null p95 or ECE <= 0.010 (decided before any holdout number: the null band is about 0.005 wide at this n and even the baseline fails it). |

### M5b holdout sign-off (run once, 2026-10-05, commit 21ccd41, clean tree)

Pre-registered rules, season-ahead 2020-2025, 177,545 plays in 1,693 games, game-cluster CIs (2,000 reps, seed 20261003). The dry run reproduced dev exactly before the real run.

| Model | Play EPA MSE | RAPM minus model [95% CI] |
|---|---|---|
| RAPM | 1.88705 | |
| Team ratings (M3) | 1.88883 | -0.00178 [-0.00275, -0.00076] |
| Box values (M5, summed over on-field players) | **1.88608** | **+0.00097 [+0.00015, +0.00179]** |
| Intercept | 1.89579 | -0.00874 [-0.01047, -0.00699] |

- **Primary: FAIL.** RAPM beats team ratings but loses to box-score values summed over the same on-field players.
- **Secondary, team-game EPA margin:** RAPM beats team ratings (-0.0058 [-0.0089, -0.0027], pass) but not box values (-0.0008 [-0.0036, +0.0023], fail).
- **Per season:** RAPM minus box is worst in 2022 (+0.0022) and 2023 (+0.0032), around the NGS-to-FTN source change. Every other season is within noise.
- **Game model (descriptive):** A4s 0.2195 vs A4s plus the RAPM lineup term 0.2193, -0.0002 [-0.0007, +0.0003]. The coefficient's sign flips across seasons.
- **Stability:** single-season year-to-year correlations stay low for OL (0.06-0.24), TE (-0.21 to 0.39) and S (-0.10 to 0.30). Chained fits run 0.6-0.95, mostly through their priors.

**Reading.** Player-level information beats team-level information season-ahead: both player methods beat team ratings, so who is on the field matters. But on-field plus-minus adds nothing beyond box-score credit once you know who played, and dev tuning (box_scale, the ridge strengths) didn't carry to 2020-2025. For a public-data model this is a real finding: with about 28,000 plays a season and five linemen always on the field together, individual on-field effects are too collinear and noisy to beat simple credit. The RAPM ratings stay available as a descriptive player view with intervals; they don't feed the game model.
