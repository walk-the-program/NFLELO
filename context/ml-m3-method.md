# M3 Method: The First ML Game Model

Written 2026-10-03, for Walker to read and approve before anything is built. It covers what the model is, the math behind each piece, how we'll test it without fooling ourselves, and the three decisions that are yours (section 9). Read `context/ml.md` first for the overall plan.

Status: **draft, awaiting Walker's decisions.**

---

## 1. The goal in one paragraph

Predict the home team's chance of winning each regular-season game, using only information available at the start of that game's week, and do it better than Elo v2. On the development seasons (2006 to 2019, the 3,450 games that have betting lines), Elo scores a Brier of **0.2178** and the market scores **0.2106**. M3 passes if it beats 0.2178 by a margin whose 95% paired confidence interval excludes zero, with calibration error (ECE) under 0.02. It then gets exactly one run on the locked 2020 to 2025 holdout.

**My honest expectation is about 0.214 to 0.216.** That would close roughly a third to half of the gap to the market. If anything scores below about 0.213, I'll treat it as a probable leak and hunt for the bug before celebrating. The notebook 01 demo showed how easily a leak beats the market.

---

## 2. Why a simple model first

The model is **logistic regression on four or five features.** That's deliberately plain, for three reasons.

1. **Small data.** There are about 250 regular-season games per season. Even training on every season from 2001, the model for 2019 sees about 4,500 games. Flexible models (boosted trees, neural nets) need far more rows than that to beat a good linear model on a noisy target like a single football game.
2. **Readability.** Every coefficient is a sentence you can check. For example: "one extra point of EPA margin per play is worth X in log-odds." If a coefficient has the wrong sign, we know something is broken.
3. **The heavy lifting happens in the features.** Most of the gain in the public literature comes from *good team ratings*, not from clever classifiers. M3 spends its effort on the ratings (section 4), then feeds them to a simple model.

We'll still run one boosted-tree comparison (section 6, step A6) so we *know* whether it helps, not just assume it.

### The logistic model

```
log( p / (1 - p) ) = b0 + b1*x1 + b2*x2 + ... + bk*xk
```

`p` is the home win probability. The left side is the **log-odds**. Logistic regression says the log-odds are a weighted sum of the features. That makes effects add up on the log-odds scale: a QB downgrade and a rest disadvantage each subtract their own amount. The intercept `b0` is the home-field edge when every feature is zero.

Elo is secretly the same thing. Elo's win probability is `1 / (1 + 10^(-d/400))`, where `d` is the rating gap plus home field. That is a logistic curve with log-odds `d * ln(10) / 400`. So **Elo's own log-odds is our first feature**, and the model learns how much to trust Elo and what to add on top. If the other features carried no information, the model would just reproduce Elo.

**Ties** (about 1 in 300 games) enter training as two half-weight rows, one win and one loss. This matches how the harness scores them (outcome 0.5).

---

## 3. The features

Every feature is computed **as of the start of the game's week** (the M1 rule). The one proposed exception is the starting QB's identity (decision M3-D1).

| # | Feature | What it measures | Source |
|---|---|---|---|
| 1 | `elo_logit` | Elo v2's log-odds for the game, including its learned home-field edge | Elo v2 pre-game ratings |
| 2 | `adj_epa_margin` | Expected EPA-per-play margin when these two teams meet, from opponent-adjusted ratings | Play-by-play, section 4 |
| 3 | `qb_delta_home`, `qb_delta_away` | How much better or worse this week's starting QB is than the QB play the team's rating already reflects | Play-by-play plus the schedule's starter, section 5 |
| 4 | `rest_diff` | Home minus away days of rest, capped at ±7 so a bye doesn't count as huge | Schedule |
| 5 | `neutral` | 1 for neutral-site games (London, Mexico City, and so on) | Schedule |

Not in M3: travel distance, weather, injuries beyond the QB, divisional games, and coaching. Those are M4 candidates once the base model is honest.

**Betting lines are never features** (decision D3). The M1 test that blocks market columns runs on every new feature builder.

---

## 4. Opponent-adjusted EPA: the main new idea

### The problem with raw EPA

M1's features average each team's EPA per play. A team that played four bad defenses has inflated offensive numbers. To fix this, we need to know how good its opponents were, and their ratings depend on *their* opponents, and so on. Everything is tangled together.

### The fix: one regression that rates everyone at once

Treat every play as an equation:

```
EPA(play) = mu + O[offense] + D[defense] + h * (offense is home) + noise
```

- `mu`: the league-average EPA per play.
- `O[team]`: how much a team's offense adds above average. Higher is better.
- `D[team]`: how much EPA a team's defense *allows* above average. Lower (more negative) is better.
- `h`: a small home-offense bump.

With about 32 offenses and 32 defenses, that's 65 unknowns fit to tens of thousands of plays. Solving them together untangles the schedule automatically: if a team's opponents were bad, their `D` values are high, and the team's `O` is credited less for the same raw EPA.

Then the matchup feature is:

```
adj_epa_margin = (O[home] + D[away]) - (O[away] + D[home])
```

In words: what the home offense should do against this defense, minus what the away offense should do against the home defense. Multiply by about 60 plays and you get a rough expected point margin, which makes the coefficient easy to sanity-check.

### Why ridge, not plain least squares

Plain least squares fails here for two reasons:
- **Early season:** after week 1, each team has played one opponent. Least squares will happily give a team a wild rating from one game.
- **Collinearity:** teams in the same division play each other twice, so some ratings are only identifiable as differences.

**Ridge regression** adds a penalty `lambda * (sum of O^2 + sum of D^2)` to the squared error. That pulls every rating toward zero, the league average, unless the data strongly says otherwise.

This is exactly a **Bayesian prior**. Ridge is the most likely answer when you believe, before seeing any plays, that team ratings are normally distributed around average with some spread `tau`. The penalty works out to `lambda = sigma^2 / tau^2`, the ratio of play-level noise to true team spread. NFL play EPA is very noisy (sigma is about 1.3 EPA per play), and true team differences are small (tau is about 0.1). So the data needs many plays to overcome the prior. That is the right behavior.

### Recency and the early-season problem

Teams change during a season and between seasons. Each play gets a weight:

```
weight = 0.5 ^ (weeks_ago / half_life)
```

So a play from `half_life` weeks ago counts half as much as this week's. **Last season's plays stay in the fit** with the same decay plus an extra off-season discount `rho`. In week 1, the ratings are last season's, shrunk toward average by the decay, the discount, and the ridge prior. As this season's plays arrive, they take over. This is the "Bayesian shrinkage for early weeks" from the plan, and it falls out of one weighted ridge fit rather than needing a separate system.

### Fitting

The fit runs once per (season, week): about 20 weeks × 20 seasons = 400 fits. Each fit uses a sparse design matrix of roughly 100,000 weighted plays by 65 columns, which takes well under a second. Garbage time and non-plays are excluded, as in M1.

We fit the combined rating first. Splitting into separate pass and rush ratings (doubling the columns) is ablation step A3.

### Tuning the rating system without touching the test seasons

The rating system has three knobs: `lambda`, `half_life`, and `rho`. If we tuned them by picking whatever gives the best game Brier on 2006 to 2019, the development score would be optimistic, because we'd have chosen the knobs on the same games we report.

So we tune them on a **different target in different years:** how well each week's ratings predict **next week's** per-game EPA margins, over **2000 to 2005 only**. Those seasons are before the development window, so the development score stays out-of-sample for these knobs. And the target ("does the rating forecast future team performance?") is what a rating is *for*. This is decision M3-D2.

---

## 5. The quarterback adjustment

A team's EPA rating already contains its usual QB. What it *misses* is a change: the starter is hurt, benched, or back from injury. So the feature is the **difference** between this week's starter and the QB play baked into the team's rating, not the starter's raw value.

### Rating each QB

For each QB, take EPA per dropback (passes, sacks, and scrambles) over all his previous games, with the same recency decay as section 4. Then shrink it:

```
qb_value = (n * raw_epa + k * prior) / (n + k)
```

- `n` is his weighted dropbacks.
- `prior` is **replacement level**: the average EPA per dropback of QBs in their first 100 career dropbacks, recomputed each season from past data only.
- `k` is the number of pseudo-dropbacks of prior belief, tuned on 2000 to 2005 like the ratings.

A rookie starts at replacement level. A 10-year starter's value is almost entirely his own record.

This is the same shrinkage idea as ridge, in its simplest form. With little data, trust the prior. With lots of data, trust the data.

**CPOE is out of the first version.** It only exists from 2006, so a training set that starts in 2001 can't use it. Adding CPOE to the QB rating is a later ablation, trained from 2007.

### The delta

```
qb_delta = qb_value(this week's starter) - weighted average qb_value of the team's
           dropbacks inside the rating window
```

The second term is the QB play the team's EPA rating "remembers." When the usual starter plays, `qb_delta` is near zero. When a backup starts, it's negative, often strongly.

### The catch: when do we know who's starting?

The schedule file records who *actually* started. That fact is only certain at kickoff, not on the Wednesday before. See decision M3-D1.

---

## 6. How we test it: the experiment ladder

**Training protocol (walk-forward).** To predict season S, fit the logistic coefficients on regular-season games from 2001 through S-1, with features computed exactly as they would have been at the time. Refit once per season. The features themselves update every week. Training sample weights decay by season (half-life about 8 seasons), so the intercept can follow the falling home-field edge.

Each step below is one logged run in the experiment registry, scored on the **same 3,450 development games**, against Elo, with a paired bootstrap CI:

| Step | Model | What it tells us |
|---|---|---|
| A0 | Logistic on `elo_logit` only | Should land at about Elo's 0.2178. If not, the training pipeline is broken. |
| A1 | A0 + raw EPA margin (M1 features) | How much plain efficiency adds |
| A2 | A0 + opponent-adjusted EPA margin | Whether adjustment beats raw. I expect it to. |
| A3 | A2 with pass and rush split | Whether pass efficiency carries more signal (the literature says yes) |
| A4 | Best of A2/A3 + QB delta, using the actual starter | The QB effect, given what is known at kickoff |
| A4b | Same, but assuming last week's starter | The QB effect, given only what is known on Wednesday |
| A5 | A4 + rest and neutral site | Context |
| A6 | LightGBM on the A5 features, monotonic constraints | Whether a flexible model finds anything logistic misses |

**Choosing the final model: the one-standard-error rule** (decision M3-D3). Among the variants whose development Brier is within one standard error of the best, pick the **simplest**. Small development-window edges often come from tuning noise, and the holdout punishes complexity bought with noise.

**Pre-registered sign-off on the holdout (written now, before anyone looks):**
- The chosen model, frozen, scored once on 2020 to 2025 regular-season games that have moneylines.
- Reported: Brier, log loss, accuracy, ECE, and paired 95% CIs against Elo v2 and against the market.
- The result is recorded whatever it is. If the model loses to Elo on the holdout, that is the finding. We don't go back and retune.

---

## 7. What could go wrong

| Risk | Guard |
|---|---|
| A feature quietly uses the future | Every new builder (`opponent_adjust`, `qb`, `context`) must pass M1's `leakage_check`, with the same positive control |
| The starting QB is known later than Wednesday | Decision M3-D1. A4b measures exactly what that knowledge is worth. |
| Elo's online home-field edge updates after Thursday games | Use the edge as of the week's first kickoff (a tiny effect, but it costs nothing to be strict) |
| Ablation shopping on the development seasons | The one-SE rule, plus a single frozen holdout run |
| Elo and EPA overlap (they're correlated) | Fine for prediction. Coefficients are read with care, and we don't claim "EPA matters more than Elo" from them alone. |
| nflfastR's EP/WP models were trained on later seasons | Accepted limitation, already noted in M1 |
| A coefficient has the wrong sign | Stop and investigate. It usually means a sign or join bug. |

---

## 8. What gets built

```
nflelo/ml/features/opponent_adjust.py   # weighted ridge ratings per (season, week), cached
nflelo/ml/features/qb.py                # QB value, replacement prior, qb_delta
nflelo/ml/features/context.py           # rest_diff, neutral
nflelo/ml/models/logistic.py            # walk-forward fit/predict, tie handling, season-decay weights
scripts/ml_tune_ratings.py              # 2000-2005 tuning of lambda, half_life, rho, k (next-week EPA target)
scripts/ml_m3.py                        # runs ladder A0-A6, logs each run, prints the table
notebooks/02_opponent_adjusted_epa.ipynb  # walkthrough: why raw EPA misleads, ridge as a prior, the 2000-05 tuning
notebooks/03_m3_results.ipynb             # the ladder, coefficients read in plain English, calibration plot
tests/ml/                                  # leakage checks for each builder, ridge sanity tests, A0 ≈ Elo
```

All of it is built by an Opus agent, then reviewed by me before it's committed.

---

## 9. Decisions for Walker

- **M3-D1. When is the starting QB "known"?**
  - **(a)** Treat the starter's identity as a pre-game fact, the one exception to the week-start rule. The market prices it too, and for the live site we'd refresh predictions once starters are announced. A4b still measures how much of the gain needs that late information.
  - **(b)** Strict Wednesday rule: assume last week's starter unless the injury report already rules him out. This is cleaner, but slower to catch injuries.
  - **Recommended: (a)**, with A4b always reported next to it, so the site can say how much the QB news is worth.
- **M3-D2. Tune the rating system on 2000 to 2005, using next-week EPA as the target.** This keeps the development score honest for those knobs. The alternative is nested walk-forward tuning on game Brier inside the development window, which is more expensive and more fragile. **Recommended: yes.**
- **M3-D3. The one-standard-error rule for picking the final model.** When variants are within noise of each other, pick the simplest. **Recommended: yes.**

## 10. Concept checkpoints

These are the ideas worth being comfortable with by the end of M3. Notebooks 02 and 03 walk through each one.

- Log-odds, and why Elo is a logistic model in disguise.
- Ridge regression as a Bayesian prior (`lambda = sigma^2 / tau^2`).
- Why opponent adjustment has to be one joint fit, not team-by-team averages.
- Recency weighting, and how last season's data becomes the early-season prior.
- Shrinkage toward replacement level for QBs.
- Tuning on a different target and in different years, to keep the test honest.
- The one-standard-error rule, and why a single pre-registered holdout run matters.
