# M5b Method: On-Field Player Ratings (Adjusted Plus-Minus)

Written 2026-10-05 by Claude under Walker's standing delegation (see the `CONTEXT.md` decision log). Read `context/ml-m5-method.md` and the M5 build notes in `context/ml.md` first.

Status: **approved under delegation 2026-10-05; build in progress.**

---

## 1. Why M5b, after M5's negative result

M5 valued players from their box-score credit, and those values didn't improve the game model. The likely reason is visible in M5's numbers: box scores can't see most of what most players do. Credit "skill" was only 0.02 to 0.09 for running backs and defenders, linemen have no stats at all, and usage-based playing time missed the nickel corners and the defensive line rotation.

The participation data (2016 to 2025) records **all 22 players on the field for every play**. That allows the method teams and serious public analysts use for player value: **regularized adjusted plus-minus (RAPM)**, which credits each player with the average effect of his presence on play outcomes, controlling for everyone else on the field. Walker approved share-alike data for exactly this work (D1, 2026-10-03).

M5b is first a **player evaluation product**: ratings for every player, at every position, with uncertainty. Its value to the game model is tested, but it doesn't depend on it.

## 2. The model

Every offensive and defensive scrimmage play from 2016 onward (garbage time excluded, the same filter as M1) becomes one row:

```
EPA(play) = mu + sum over the 11 offensive players of O[p]
               + sum over the 11 defensive players of D[p]
               + h * (offense is home) + situation terms + noise
```

- **Each player has two coefficients**, `O[p]` and `D[p]`, but in practice he plays on only one side. `D` is EPA *allowed*, so a good defender has a negative `D`, and his reported value is `-D`.
- **Situation terms** (down, distance bucket, field-zone bucket, pass/run) absorb what expected points already miss. They're small, because EPA is measured relative to the situation already.
- **Ridge with informative priors (Bayesian RAPM).** Each player's coefficient is shrunk toward a **prior mean**, not toward zero. The prior is his M5 box-score value converted to the same units, or the position's replacement level if he has no box record. Each position group gets its own ridge strength, because collinearity is extreme on the offensive line (five linemen are usually on the field together) and much lower for receivers.
- **Time.** Ratings are fit season by season. Each season's prior is last season's posterior, decayed toward the box prior. This is the same "carry forward and shrink" logic as the team ratings, so a player's rating follows him to a new team.
- **Uncertainty.** An approximate posterior SD for each player comes from the ridge system. Ratings are published with intervals, and a player with 50 snaps is shown as uncertain, not as an extreme.

**Scale:** about 45,000 plays a season, 22 players each, roughly 2,500 active players. That's a sparse system solved with a conjugate-gradient ridge solver in seconds.

## 3. Playing time

The same participation data gives exact snap shares, replacing M5's usage proxy. Expected snaps for a coming game come from each player's share over his previous games. Live, the current season's participation isn't released until after the Super Bowl (see the risk table), so in season the shares come from prior seasons plus the depth chart.

## 4. How it is tested

The test must match how the ratings would actually be used. Participation for the current season isn't available in-season, so the realistic use is **season-ahead**: ratings fit through season S-1, used for season S.

**Primary test: predicting play outcomes a season ahead.**
- Using ratings through S-1, predict each play in season S from the 22 players actually on the field: the sum of their ratings plus the situation terms.
- Compare with two baselines:
  - (a) **team ratings** from the M3 opponent-adjust ridge through S-1, which assign each play its offense's and defense's team rating;
  - (b) **M5 box-score player values**, summed over the same players.
- Metric: mean squared error of play EPA.
- Uncertainty: a paired bootstrap that **resamples games** (plays within a game aren't independent), 2,000 reps.

This is the real test of a player rating: rosters turn over every offseason, and only a player-level rating can follow players to their new teams.

**Secondary tests:**
- Team-game EPA margin in season S predicted from the actual lineups (MSE vs the same baselines).
- Year-to-year stability of the ratings, by position, reported.
- **Game model:** A4s plus a lineup delta from RAPM values and participation-based snap shares, scored against A4s. This is reported but is not the acceptance test. There's little power here, and M5 already showed how hard lineup effects are to prove at the game level.

**Windows.**
- Development: season-ahead predictions for **2018 and 2019** (trained on 2016 to 2017, then 2016 to 2018). Ridge strengths, decay, and situation terms are tuned here.
- Holdout: **2020 to 2025**, season-ahead, scored once after the rules are committed. This is the same locked window as the rest of the project.

**Acceptance:** RAPM beats both baselines on the holdout's play-level MSE, with paired CIs that exclude zero.

## 5. Licensing and credit

The ratings, the fitted model, and any data files derived from participation are **CC BY-SA 4.0**. Credit "NFL Next Gen Stats via nflverse" for 2016 to 2022 and "FTN Data via nflverse" for 2023 onward. The derived files carry a license note. Nothing from participation flows into A4s, which stays CC BY.

## 6. Risks

| Risk | Guard |
|---|---|
| Offensive linemen are inseparable (always on the field together) | Strong position-level ridge plus the box/with-without prior; report each lineman's interval, not just his point value |
| The data source changes in 2023 (NGS to FTN) | Check per-season player counts and IDs per play; report any season-level shift |
| Snaps on special teams or with missing IDs | Scrimmage plays only; drop plays with fewer than 11 identified players per side and report the drop rate |
| No in-season participation for live use | Season-ahead is the primary test, which matches live use exactly |
| Overfitting through tuning | Tune on the 2018 and 2019 predictions only; the holdout runs once |

## 7. What gets built

```
nflelo/ml/players/participation.py  # load + clean participation, build sparse play-by-player design
nflelo/ml/players/rapm.py           # Bayesian RAPM (ridge to prior), season chaining, posterior SDs
scripts/ml_m5b.py                   # tuning on 2018-19, season-ahead evaluation, sign-off, dry run
notebooks/05_player_values.ipynb    # M5: box-score values and lineups, and why they didn't move the game model
notebooks/06_player_ratings.ipynb   # M5b: RAPM from first principles, ratings with intervals, validation
tests/ml/                           # leakage (season-ahead only uses < S), solver correctness on synthetic lineups
```
