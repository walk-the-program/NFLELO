# M4 Method: The Game Model Goes Live

Written 2026-10-04, for Walker to read and approve before anything is built. Read `context/ml-m3-method.md` first. M4 takes the M3 model (A4s), puts it on the site next to Elo and Vegas, adds a margin model and playoff odds, and starts a live 2026 scorecard that can't be faked.

Status: **Phase 1 approved 2026-10-04.** D1 yes, D2 yes, D4 deferred to M4b. D3 (strength shocks) approved 2026-10-04. Walker also asked for a rest-of-season prediction: Phase 1 publishes every remaining game's probability and each team's projected wins, and Phase 2 adds the full simulation.

---

## 1. What M4 delivers

1. **Weekly predictions** for every game: our win probability and spread, shown beside Elo and Vegas on the site's "This week" section.
2. **A prediction ledger**: every prediction is written down and committed *before* kickoff, so the live record is verifiable from git history.
3. **A live 2026 scorecard**: model vs Elo vs market, scored only on predictions the ledger proves were made in advance.
4. **A margin model**: the full distribution of the final score margin. It gives the spread, and it drives the simulation.
5. **Playoff odds**: a Monte Carlo simulation of the rest of the season, with real NFL tiebreakers. It reports the chances of making the playoffs, winning the division, getting the #1 seed, and winning the Super Bowl.

Totals (over/under) are proposed as a later follow-up (decision M4-D4).

### The clock

The 2026 season is in week 4 (nflverse numbering; it runs through Monday 2026-10-05). Weeks 1 to 4 can only ever be a **backtest**, because nobody wrote predictions down before those games. They'll appear on the site shaded and labelled that way. **The live record starts with week 5** (Thursday 2026-10-08). So the work is split in two phases:

- **Phase 1, live by Thursday 2026-10-08:** the prediction script, the ledger, the site columns, and the automated weekly run. These use the M3 model as is.
- **Phase 2, after that:** the margin model, tiebreakers, the playoff simulation, the scorecard section, and the margin model's single holdout run.

---

## 2. The weekly pipeline

```
scripts/build.py            (unchanged: Elo, site export)
scripts/ml_predict.py       (new: refresh nflverse, rebuild features as of now, predict upcoming games, append to the ledger)
scripts/export_site.py      (reads the ledger if present; the site still builds without it)
```

- **Kept separate on purpose.** The ML step runs as its own script, and the exporter treats its output as optional. So the Elo site never breaks because of the ML code (decision D5 still holds: the plain site build installs only `requirements.txt`).
- **Model refit.** A4s refits once per season on all completed seasons, as in walk-forward. For 2026, that means training on 2001 to 2025. The knobs stay frozen at the values tuned on 2000 to 2005.
- **Play-by-play refresh.** The current season is re-pulled each run (`data.refresh(2026)`). The manifest hash tells us when nflverse corrected earlier weeks.
- **Starting QBs.** nflverse fills `home_qb_id` and `away_qb_id` for upcoming games (30 of the 30 week-5 games have them). Phase 1 must confirm *when* nflverse updates those fields during the week, because the Sunday refresh (section 3) is only worth running if they change by Sunday morning. If a game has no starter listed, the model falls back to the team's previous starter and flags the game.

## 3. The ledger: making the live record honest

`experiments/live/2026.csv`, append-only and committed to git. One row per (game, prediction run):

```
run_at_utc, game_id, kickoff_utc, model_version, p_home_model, spread_model,
p_home_elo, spread_elo, p_home_market, spread_market, home_qb_id, away_qb_id, qb_source
```

Rules:
- **A row whose `run_at_utc` is at or after kickoff is rejected** by the script, and a test checks this.
- **The scored prediction for a game is the last row before kickoff** (decision M4-D2). The Wednesday row is also kept and scored separately. That gives a live measurement of what the QB news is worth, the A4s vs A4bs question, answered in real time.
- **Proof of timing is the git commit timestamp** on GitHub, which we can't backdate after the fact on a pushed public repo. That's why the automated run commits and pushes (decision M4-D1).
- **The market columns are recorded for scoring only.** They are written to the ledger *after* the model has predicted, and never read back by the feature code. The D3 ban test covers the ledger reader.

## 4. The margin model

The question: what is the probability distribution of (home score minus away score)?

**Mean.** A linear regression on the same three features as A4s (`elo_logit`, `adj_epa_margin`, `qb_delta_diff`), fit walk-forward like M3. Its prediction is our **spread**.

**Spread around the mean.** NFL margins vary by about 13 to 14 points around any forecast. We fit the standard deviation `sigma` walk-forward and test two shapes:

1. **Normal:** `margin ~ Normal(mean, sigma)`. Simple, but it ignores football's key numbers.
2. **Key-number-aware:** a discrete distribution over integer margins. Start from the normal, then reweight each margin by its historical frequency relative to what the normal predicts (3, 7, 6, 10, 14, and 4 are overrepresented; ties are rare). Fit the reweighting factors on 2001 to 2005 only, so they never see the DEV window.

**Why both?** The win probability is `P(margin > 0)`, plus half of `P(margin = 0)` for ties. The spread's cover probability depends heavily on how much mass sits on 3 and 7. We pick the shape by CRPS on DEV, using the one-SE rule.

**One number, one story.** The site shouldn't show a win probability that contradicts its own spread. If the margin model's implied win probability scores within one SE of A4s's Brier on DEV, **the margin model becomes the single source** for both numbers (and for the simulation). If it is clearly worse, we keep A4s for the probability and show the spread as a secondary number.

**Acceptance (DEV first, then one pre-registered holdout run):**
- Margin MAE beats Elo's spread MAE (`elo_diff / 25`), with a paired CI that excludes zero.
- The implied win probability's Brier is within one SE of A4s.
- Calibration uses the **new sample-size-aware check**: the model's ECE must fall inside the 5th to 95th percentile range of ECE values that a perfectly calibrated model would produce at the same n (simulated, as in M3's sign-off). The same check applies to the spread: the share of games where the home team beats our spread should be about 50% in every predicted-spread bin.
- Market spread MAE is reported as the benchmark, as always.

## 5. Playoff odds

### Tiebreakers first, and tested against history

Playoff odds are only as good as the seeding logic. We implement the NFL's actual procedure: division ties first, then wild cards, with the two-club and three-or-more-club steps (head-to-head, division record, common games with a 4-game minimum, conference record, strength of victory, strength of schedule, and on down to a coin flip). Points-based steps are approximated with simulated margins. Seven seeds per conference since 2020, six before.

**The test that matters:** feed in each real season's actual results and check that we reproduce the actual seeds, for every season from 2006 to 2025. Every mismatch is investigated. A season can only be excused if it was decided at a step we approximate (points-based or coin flip), and it gets listed by name.

### The simulation

For each of 20,000 simulated seasons:
1. **Draw a strength shock per team.** Ratings are estimates, and teams change over a season. A fixed-probability simulation ignores both and comes out overconfident: favorites "lock up" playoff spots too early. Each team gets one random shock to its mean margin, `Normal(0, tau_rest)`, held for the rest of that simulated season. This is the same idea as FiveThirtyEight's "hot" simulations, without needing to re-run the whole ratings fit inside every simulation.
2. **Play every remaining game** using the margin model (mean plus shocks, then the chosen margin shape). Ties happen naturally from the discrete distribution.
3. **Seed with the tiebreakers, then play the bracket.** Home field goes to the higher seed; the Super Bowl is neutral.
4. **QBs for future games:** each team's current listed starter. A known long-term injury therefore carries forward, but a return from injury won't be anticipated. This is a stated limitation.

### Calibrating the simulation

`tau_rest` is the one knob. We tune it on **2001 to 2005**: run the simulation from weeks 4, 8, 12, and 16 of each season, and choose the value that minimizes the Brier of "made playoffs." Then we **backtest on DEV 2006 to 2019** from the same weeks. The playoff odds must be calibrated (sample-size-aware ECE check), and they must beat the simple baseline of simulating with fixed probabilities (`tau_rest = 0`).

Cost: 20,000 seasons × about 150 remaining games is a few seconds in vectorized numpy. The DEV backtest (14 seasons × 4 start weeks) takes minutes.

## 6. On the site

- **This week:** each card gets a three-way row of **Elo · Model · Vegas** (win probability and spread). A small "QB change" tag appears when `qb_delta_diff` is large, with the starter's name. The "biggest disagreement" feature card switches to model vs Vegas.
- **Playoff odds:** a new section with a table per conference showing the chances of making the playoffs, winning the division, the #1 seed, and winning the Super Bowl, plus each team's change since last week. It follows Walker's tokens and the `dataviz` rules, like the rest of the site.
- **Scorecard:** a live 2026 block (model vs Elo vs market: Brier, accuracy, and record against the spread so far), with the backtest weeks 1 to 4 shaded and labelled. It includes the Wednesday-vs-final comparison.
- **Methodology:** one plain-English paragraph on the model, linking to the ledger file on GitHub so anyone can check the timestamps.
- **Unchanged:** the power ladder stays Elo. The ML model has no single "team rating" to rank by yet; that can come later.

## 7. What could go wrong

| Risk | Guard |
|---|---|
| A prediction written after kickoff | The script refuses it, a test covers it, and git timestamps let anyone audit it |
| nflverse's QB fields update too late to help | Phase 1 measures this before we rely on the Sunday run. If they lag, we keep only the Wednesday run and say so. |
| nflverse corrects earlier play-by-play | Manifest hashes are logged per run, and the ledger records `model_version` plus the data hash |
| Tiebreaker bugs | Reproducing every actual season's seeds from 2006 to 2025 |
| Overconfident playoff odds | `tau_rest` tuned on 2001 to 2005, plus a calibrated backtest on DEV |
| The live record looks bad after 4 weeks | About 60 games is noise. The scorecard shows confidence intervals and says so in plain words. |
| The automated job fails silently | The job fails loudly (a red check on GitHub), and the site shows the timestamp of the last prediction run |

## 8. Decisions for Walker

- **M4-D1. Automated runs.** Should I build a scheduled GitHub Action that runs Wednesday 10:00 ET (full refresh) and Sunday 08:00 ET (starter refresh, if Phase 1 shows it helps), and commits the ledger and `site/data/` to the public repo? This *does not host the site*; that decision stays separate. Without it, the live record depends on someone running the script before every kickoff. **Recommended: yes.**
- **M4-D2. Which prediction counts.** Score the last prediction before kickoff as the official one, and also score the Wednesday prediction separately. **Recommended: yes.**
- **M4-D3. The playoff-odds simulation includes the strength shocks** (section 5). It adds a knob, but without it the odds are overconfident. **Recommended: yes.**
- **M4-D4. Totals (over/under) model.** It needs different inputs (pace, weather, roof) and isn't needed for anything else in M4. Build it later as M4b, so M4 ships while the season is still young. **Recommended: defer to M4b.**

## 9. Concept checkpoints

- Predicting a distribution, not a point: why `P(margin > 0)` and the spread should come from one model.
- Key numbers, and why a normal curve gets cover probabilities wrong.
- Monte Carlo simulation, and why fixed probabilities make overconfident playoff odds.
- Tuning a simulation by backtesting its probabilities.
- What makes a live forecast record trustworthy (pre-registration, timestamps, no edits after the fact).
