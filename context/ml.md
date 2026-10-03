# NFLELO Machine Learning: Plan and Context

This is the ML sub-project's context file. Read `CONTEXT.md` first, then this file. It is both the plan and the living record: when a decision gets made or a milestone is finished, update the status table and the decision log at the bottom.

Written 2026-10-03. Status: plan drafted, D1-D6 decided, M1 and M2 in progress. Current scope: Stage A (M1-M4) and M5a, all on CC BY data.

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
| M3 | First ML game model beats Elo (paired CI excludes 0) | M2 | Not started |
| M4 | Production game model on the site, with live 2026 scorecard and playoff odds | M3 | Not started |
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
