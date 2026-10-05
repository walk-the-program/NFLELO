# M6 Method: The Play Outcome Model

Written 2026-10-05 by Claude under Walker's standing delegation. Read `context/ml.md` section 6 (the original M6 sketch) and the M5/M5b results first.

Status: **approved under delegation 2026-10-05.**

## 1. The question

Walker's original goal: *given the situation and the players on the field, how many yards will this play gain?* The answer is a **full distribution**, not one number. Most runs gain 0 to 6 yards, a few break for 40 or more, and incomplete passes sit at exactly zero. From the distribution we get:
- expected yards;
- P(first down or touchdown);
- P(explosive play, 20+ yards);
- P(loss);
- P(turnover), from a separate head.

M6 is also the engine M7 (decision support) needs.

## 2. Data and licensing

- **Play-by-play (CC BY)** gives the situation: down, distance, yard line, score margin, time, timeouts, home, roof and weather.
- **Participation (CC BY-SA, 2016+)** gives the pre-snap structure: offensive personnel grouping (e.g. "1 RB, 1 TE, 3 WR"), defensive personnel, offensive formation (shotgun, pistol, under center, and so on), defenders in the box, and the 22 player IDs.
- **FTN charting (2022+, CC BY-SA)** isn't in the core model, because the development seasons don't have it. It's a later ablation.
- **The model and its outputs are CC BY-SA**, as approved in D1.

**Pre-snap only.** These fields are known only after the snap and must never be inputs, enforced by a banned-columns test:
- the number of pass rushers;
- coverage type, man/zone, and pressure;
- routes and time to throw;
- air yards and anything else from the outcome.

## 3. Two versions

1. **Situation view** (what both teams can see at the snap): situation, personnel groupings, formation, box count, and the team ratings as of the game (the M3 opponent-adjusted offense and defense, pass and run).
2. **Call-conditioned view:** the above plus the play type (run or pass). This answers "what happens if we run here," which M7 needs. Run and pass are each one model with a play-type input, not separate models.

## 4. The target

Yards gained, binned: one bin each for **-10 or worse**, **-9 through +40**, **41 or more but short of the end zone**, and **touchdown**. The distribution is truncated to what the field allows: a play from the 8-yard line can't gain 30. The model predicts the bins, and the mass beyond the goal line is folded into the touchdown bin.

Incomplete passes are an outcome at 0 yards. Sacks count as negative yards. Turnovers are predicted by a separate binary head, because their cost isn't measured in yards.

## 5. Models, simplest first

1. **Baseline:** the historical distribution for the same down × distance bucket × field zone (× play type in the call-conditioned view), from the training seasons.
2. **Gradient-boosted multiclass** (scikit-learn HistGradientBoosting) on the situation and structure features. It's the strong tabular benchmark.
3. **Neural network** (PyTorch, CPU):
   - a shared trunk over the situation and structure features;
   - **player embeddings** with a permutation-invariant (DeepSets) pooling layer for the 11 offensive and 11 defensive players, so the order players are listed in doesn't matter;
   - a softmax head over the bins and a turnover head.
   - The embeddings are initialized from M5 box values and the M5b ratings as features.

   The point is to test whether player identities add anything once personnel and team ratings are known. M5b suggests they add little; M6 measures it directly at the play level.

## 6. Metrics and windows

- **Primary metric: CRPS over yards**, the discrete ranked probability score, lower is better. It scores the whole distribution, and it's the metric the NFL's own Big Data Bowl used for rushing yards.
- **Secondary metrics:** bin log loss; calibration of P(first down), P(20+ yards), and P(loss), using the one-sided sample-size-aware ECE check; and Brier for the turnover head.
- **Season-ahead protocol**, as in M5b: train on 2016..S-1 and predict S.
  - **Development:** predict 2018 (trained on 2016-17) and 2019 (trained on 2016-18).
  - **Holdout:** 2020-2025, once, after the rules are committed.
- **Uncertainty:** paired bootstrap resampling **games** (2,000 reps).

## 7. Acceptance

- **Primary:** the chosen model (one-SE rule, simplest within one SE) beats the baseline on holdout CRPS, with a game-cluster CI upper bound below 0, AND its P(first down) calibration passes.
- **Reported:**
  - the player-identity ablation (the network with embeddings vs without), as a finding either way;
  - the gradient-boosted model vs the network;
  - splits by run and pass, by down, and by field zone.

## 8. Risks

| Risk | Guard |
|---|---|
| Post-snap fields leak in | A banned-columns test and an explicit allowlist of pre-snap features |
| Team ratings for a play use later games | Ratings come from the M3 as-of machinery (week start); `leakage_check` applies |
| The NGS-to-FTN source change in 2023 shifts the formation and box fields | Check the field distributions by season before training; report any shift, and handle it with a season-era indicator if needed |
| The network overfits | Early stopping on a within-training validation season, weight decay, and a small model; report each embedding's norm vs its snap count |
| The PyTorch install | Add `torch` (CPU) to `requirements-ml.txt`; the free download is fine |

## 9. Build

```
nflelo/ml/plays/data.py       # play table: pre-snap features, bins, banned-columns allowlist
nflelo/ml/plays/baseline.py   # empirical situation baseline
nflelo/ml/plays/gbm.py        # boosted multiclass
nflelo/ml/plays/net.py        # PyTorch: trunk + DeepSets player pooling + heads
nflelo/ml/plays/metrics.py    # discrete CRPS, bin log loss, derived-probability calibration
scripts/ml_m6.py              # dev, sign-off, dry run
notebooks/08_play_model.ipynb # walkthrough: distributions vs point estimates, CRPS, embeddings, results
```
