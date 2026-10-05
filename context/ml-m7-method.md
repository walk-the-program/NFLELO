# M7 Method: Decision Support

Written 2026-10-05 by Claude under Walker's standing delegation. Read `context/ml.md` section 7 (why decision support is a causal problem) and the M6 build notes first.

Status: **approved under delegation 2026-10-05.**

## 1. Scope

M7 answers **"what should the team do here?"** for two kinds of decision. Both are things front offices and coaching staffs actually use:

- **Part A, fourth down (go, field goal, or punt).** This is the best-posed decision in football. The options are few and clear, each one's consequences can be modeled separately, and the outcome currency is **win probability**.
- **Part B, play type and personnel on early downs (run or pass, by personnel grouping).** This is harder. Teams choose these because of the situation, so the raw averages are biased (section 7 of `ml.md`). It needs observational causal methods, and the honest output includes "the data can't tell."

## 2. Our own win-probability model (the foundation)

Every decision is valued in win probability, so we need a WP model we control. nflfastR's WP is trained on all seasons, including our holdout, and it uses the betting spread.

- **Inputs:** score differential, seconds remaining, half, possession, down, distance, yard line, timeouts for each side, receiving the second-half kickoff, and the **pregame strength from A4s**. A4s supplies the team quality where nflfastR uses the market spread.
- **Model:** gradient boosting with monotone constraints (more points, more time to use, and better field position can never lower WP), plus isotonic calibration.
- **Windows:** train 2006..S-1, develop on 2016-2019, hold out 2020-2025, all season-ahead.
- **Tests:** play-level Brier and calibration of the final result, overall and by quarter. **Benchmark:** nflfastR's `wp` and `vegas_wp` on the same plays (reported; not used as inputs).

## 3. Part A: fourth-down decisions

For every fourth down, compute the WP after each option:

- **Go for it:** the conversion probability and resulting field position come from the **M6 call-conditioned yards distribution** (run and pass, mixed by the team's own historical mix in similar spots). Then WP of the resulting state if converted, and of the opponent's ball at that spot if not.
- **Field goal:** a make-probability model by distance, roof, wind, temperature, and the kicker's shrunk value (from M5). Then WP after a make (opponent's ball after the kickoff) or a miss (opponent's ball at the spot).
- **Punt:** a net-punt-distance distribution by field position (touchbacks, fair catches, returns), with a punter value. Then the WP of the resulting state.

**The recommendation** is the option with the highest WP, reported with its margin **and an uncertainty band**, from bootstrapping the component models. If the band crosses zero, the call is a toss-up.

**Selection bias.** Teams go for it on fourth down only when it looks good, so learning the conversion rate from fourth-down attempts alone would overstate it. Using M6 avoids this, because M6 learned yards from **all** downs in the same down-distance-field situations, with down as an input. We check this directly: compare predicted conversion rates with actual rates on 4th-and-1 and 4th-and-2 attempts, and on 3rd downs with the same distance.

**Validation:**
1. **Component calibration on the holdout:** conversion probability (on actual attempts), field-goal make probability, the punt distance distribution, and WP itself.
2. **Decision audit:** for every fourth down from 2020 to 2025, the recommendation, the team's actual choice, and the **WP the team gave up** by its choice, according to the model. We report this as **model-estimated**, not proven: the counterfactual outcome is never observed.
3. **Agreement with outcomes:** among plays where teams went for it although the model said kick (and the reverse), do the realized results line up with the model's estimated cost? Since teams choose with information we don't have, this is reported as a consistency check, not a proof.

## 4. Part B: early-down play type and personnel

**Actions:** personnel grouping (11, 12, 21, 13, 10, others pooled) × run or pass, on 1st and 2nd down. **Outcome:** EPA (and success).

1. **Propensity model:** P(action | situation, team, opponent, score, time) by multiclass GBM. **Overlap check:** recommend only among actions with a propensity of at least 5% in that situation. Teams never use some actions in some spots, and we can't learn about those.
2. **Effect estimates:** for each situation cell and action, a **doubly robust (AIPW)** estimate of expected EPA, combining the M6-based outcome model with propensity weighting, using cross-fitting by season.
3. **Report:** each action's estimated EPA, with a confidence interval, relative to the team's actual mix. Situations where the intervals overlap are labeled "no clear best."
4. **Off-policy evaluation on the holdout:** a DR estimate of how the recommended policy would have performed in 2020-2025 compared with what teams actually did, with a cluster bootstrap by game. A policy that can't beat the observed play in this estimate is reported as such.
5. **Game-theory caveat**, stated plainly: if a team always took the "best" action, the defense would adjust. The estimates describe marginal shifts from current tendencies, not a fixed strategy.

## 5. Acceptance (pre-registered before the holdout, then one run)

- **WP model:** holdout calibration passes (one-sided null ECE, or ECE ≤ 0.01) and its Brier is within 0.002 of nflfastR's `wp` on the same plays. A small gap is acceptable, since nflfastR trains on more data, including our holdout.
- **Part A:** conversion, field-goal, and punt components calibrated on the holdout (same rule). The decision audit is published whatever it shows.
- **Part B:** the OPE estimate of the recommended policy minus observed play, with a CI. **Pass if the lower bound is above 0.** Otherwise Part B ships as a descriptive tool with "no clear best" labels and no claimed improvement.

## 6. Licensing

Part A uses play-by-play (CC BY), M6 (CC BY-SA), and M5 kicker values (CC BY), so **Part A outputs are CC BY-SA**. Part B uses participation personnel, so it's **CC BY-SA** too. The WP model alone (play-by-play plus A4s) is CC BY.

## 7. Build

```
nflelo/ml/wp/model.py            # our WP model (monotone GBM + isotonic)
nflelo/ml/decisions/fourth.py    # go / FG / punt valuation with uncertainty
nflelo/ml/decisions/kicking.py   # FG make prob, punt distance distribution
nflelo/ml/decisions/policy.py    # propensities, AIPW cross-fit, OPE
scripts/ml_m7.py                 # dev, audit, sign-off, dry run
notebooks/09_decision_support.ipynb
```
