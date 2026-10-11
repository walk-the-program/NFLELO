# M8 Method: Game Model v2

Written 2026-10-11. Walker approved items 1, 2, 3, 5 and 6 and asked for them to be included in the master model. Read `context/ml-m3-method.md`, `context/ml-m3b-method.md` and the M3/M3b/No-Elo notes in `context/ml.md` first.

Status: **approved by Walker 2026-10-11; built 2026-10-11 (DEV only).** Result: nothing beats C2d beyond noise; the one-SE pick is C2d (-0.0010 vs A4s, CI [-0.0019, +0.0000]), so DEV acceptance is not met and nothing is frozen. See "M8 build notes" and "M8 forward test" in `context/ml.md`.

## Goal

Beat A4s (DEV Brier 0.2151) and the Kalman candidate C2d (0.2141) on DEV 2006–2019 (the same 3,450 games, paired CIs). Then promote the winner to the site's headline model, with a pre-registered live forward test.

## Components

1. **Less noisy targets.** Fit models on point margin and on EPA margin (and a blend of the two) instead of win/loss. Convert to a win probability with a fitted noise scale: a normal CDF of predicted margin over sigma, then calibrated with a logistic on the margin. Compare with the win/loss logistic on the same features.
2. **Blending.** Stack A4s, C2d (Kalman), Elo, and the best margin-target model on the log-odds scale. Weights are non-negative, fit out-of-fold walk-forward (season by season, using only earlier seasons), with a small ridge toward equal weights.
3. **QB v2.**
   - (a) **Opponent-adjusted QB value:** a ridge regression on dropback EPA with passer and defense effects, so a QB isn't credited for weak defenses.
   - (b) **Injury-exit cleaning:** games where the QB had under 50% of the team's dropbacks, or left early, are down-weighted in his rating and don't count as "his" games in the team baseline.
   - (c) Keep the experience prior only if it helps. CPOE stays out (no gain in M3b). No PFR-derived draft data.
4. **Early season** (the weakest weeks; the roster-change prior failed in M5):
   - an offseason QB-change flag (the new starter differs from last season's main starter), interacted with weeks played;
   - a head-coach change (schedule file, CC BY);
   - ratings carry-over strength tuned on 2000–2005 (no DEV peeking).
5. **Context:**
   - travel distance (stadium coordinates compiled from public facts);
   - time-zone shift and body-clock effects (e.g. West Coast teams in 1pm ET games);
   - short week (Thursday), bye, and rest difference (re-test);
   - altitude (Denver), roof/surface, wind and temperature from the schedule;
   - international and neutral sites.

   Each is tested as an addition to the best base model.

## Protocol

- **Walk-forward on DEV 2006–2019** exactly as in M3. Knobs for the new rating pieces (QB v2 ridge, carry-over) are tuned on 2000–2005 with non-outcome targets where possible.
- **Ladder:**
  - E0 = A4s.
  - E1 = margin targets.
  - E2 = QB v2.
  - E3 = early-season features.
  - E4 = context features.
  - E5 = the best combination.
  - E6 = the stack of E5 + C2d + Elo (+A4s).
- **Selection:** the one-SE rule, simplest within one SE.
- **Reporting:** every step against A4s and C2d, with paired CIs (2,000 reps, seed 20261003), ECE (one-sided null or 0.01 floor), and splits for weeks 1–4, changed-starter games, and short weeks.
- **Leak alarm:** anything that beats A4s by more than 0.006 gets investigated before it's reported.

## Acceptance and promotion

- **DEV acceptance:** the chosen model beats A4s with a paired CI that excludes zero, and passes calibration.
- **If accepted:** it becomes the **site's headline model** from the next weekly run (Walker's instruction). A4s keeps being logged in the ledger as the frozen 2026 baseline, so its record continues unbroken. The scorecard shows both.
- **If not accepted:** it runs as a shadow model, like C2d.
- **The 2020–2025 holdout is spent.** No numbers are computed on it.
- **Forward test, pre-registered before the switch:**
  - M8 minus A4s Brier on all 2026 REG games from the switch week on, with a paired game-level CI, run once after the regular season ends;
  - a pooled 2026+2027 secondary;
  - the frozen model artifact committed before the first live M8 row.

## Build

```
nflelo/ml/features/qb.py        # (extend) opponent-adjusted QB value, injury-exit cleaning (behind flags; A4s unchanged)
nflelo/ml/features/context.py   # (extend) travel, time zones, short week, altitude, weather, coach change, offseason QB change
data/sources/stadiums.csv        # stadium coordinates and time zones (public facts), with sources noted
nflelo/ml/models/margin_target.py, nflelo/ml/models/stack.py
scripts/ml_m8.py                 # tune, dev ladder, freeze, forward-test stages
live wiring                      # ml_predict.py logs M8 (headline if accepted) plus A4s; site and scorecard show both
notebooks/00_nflelo_walkthrough.ipynb  # new M8 section
```
