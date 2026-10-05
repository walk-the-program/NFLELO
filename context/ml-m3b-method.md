# M3b Method: Game-Model Refinement

Written 2026-10-05 by Claude under Walker's standing delegation. Read `context/ml-m3-method.md`, the M3 and No-Elo notes in `context/ml.md`, and the M5 and M5b results first.

Status: **approved under delegation 2026-10-05; built 2026-10-05 (dev only).** Result: the one-SE pick is C2d (Elo + Kalman `kf_margin` + QB delta), -0.0010 Brier vs A4s with a CI that just includes zero, so the acceptance bar is narrowly not met; C1 and the QB extras gave nothing. See "M3b build notes" and the proposed holdout pre-registration in `context/ml.md`.

## Why

After M5 and M5b, the evidence on where A4s loses ground is specific:

1. **Early season.** In weeks 1 to 4, A4s scores 0.2243 against Elo's 0.2229. Elo's weight in the model falls from about 1.0 early in the season to about 0.7 late, while EPA's rises, but A4s uses **fixed** weights for the whole season.
2. **Changed starting QB.** On DEV games where a team's starter changed, A4s scores 0.2142 against the market's 0.2072. The QB term is the model's best feature and its biggest remaining gap.
3. **Ratings built by hand.** Elo and the ridge EPA ratings are separate, hand-tuned systems. A single principled dynamic model of team strength has never been tried.

The 2026 live record stays frozen on A4s. Anything that wins here runs as a **shadow model** in the ledger, like M5 was meant to, and can become the headline in 2027.

## Candidates (DEV 2006-2019; same 3,450 games; paired CIs vs A4s; one-SE rule)

**C1. Week-varying weights.** Add `elo_logit × early` and `adj_epa_margin × early`, where `early = max(0, 1 - weeks_played / w0)` and `w0` is tuned on DEV. This lets the model lean on Elo early and on EPA late. It's cheap, and it targets weakness 1 directly.

**C2. A state-space team-strength model (Kalman filter).** Each team has a latent strength (offense and defense, in points) that drifts as a random walk from week to week and is pulled toward the mean between seasons. Two kinds of measurement update it after each game:
- the game's point margin (very noisy, about 13.5 points of noise);
- the game's opponent-adjusted EPA margin, converted to points (less noisy per game).

A Kalman filter gives each team's strength **and its uncertainty** in closed form, with no sampling. It's fast enough to run weekly.
- **Settings:** the drift variance, the between-season shrinkage, and the two noise levels. They're tuned on **2000-2005** by maximum likelihood of the next game's margin, keeping DEV out of the tuning, as in M3-D2.
- **Feature:** `kf_margin` (predicted home margin) and its SD.
- **Tests:** C2a uses `kf_margin` instead of `elo_logit`; C2b uses it alongside Elo; C2c replaces both Elo and EPA with it.

This is the Glickman-Stern idea from the original plan (section 5, model 4), done as a filter rather than full MCMC.

**C3. A better QB term.**
- (a) **CPOE in the QB value** (EPA plus CPOE composite), trained from 2007, since CPOE exists from 2006.
- (b) **A better prior for backups and rookies:** replacement level adjusted by experience (career dropbacks) and draft round, from the CC BY players table. Verify that table's draft fields are nflverse CC BY, not Pro-Football-Reference; if they aren't, drop draft round.
- (c) **Aging:** a small age-curve adjustment to the QB value.

Each is scored as A4s with only the QB value replaced, and reported overall and on changed-starter games.

**C4. The best of C1, C2, and C3 combined.** Choose by the one-SE rule.

## Acceptance

The chosen model beats A4s on DEV, with a paired CI that excludes zero and the one-sided sample-size-aware ECE check passing. Then:
- one pre-registered holdout run on 2020-2025, against A4s on the same 1,615 games;
- splits reported for weeks 1 to 4 and changed-starter games.

If it passes, it's logged as a shadow model in the weekly ledger from then on.

## Risks

- **Overfitting DEV through many candidates.** Use the one-SE rule, tune C2 and C3 on 2000-2005 where possible, and run the holdout once.
- **Leakage in the Kalman filter.** Strength for a game uses only games before its `as_of`. It must pass `leakage_check`.
- **The draft-data license.** Verify it before use.

## Build

```
nflelo/ml/features/kalman.py   # state-space team strength, tuning by likelihood on 2000-2005
nflelo/ml/features/qb.py       # (extend) CPOE composite, experience/draft prior, aging, behind flags
scripts/ml_m3b.py              # ladder C1-C4, sign-off, dry run
notebooks/07_game_model_refinement.ipynb
```
