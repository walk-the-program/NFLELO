# M5 Method: Player Values and the Pregame Lineup

Written 2026-10-05, for Walker to read and approve before anything is built. Read `context/m5-data-scope.md` first (what the free data can and can't support), then `context/ml-m3-method.md` (the game model this extends).

Status: **draft, awaiting Walker's decisions (section 9).**

---

## 1. The goal, and where the current model is weak

Walker's direction: value **every player** from his own per-game statistics, then account for **who is actually starting** before each game, not just the QB.

The current model (A4s) already handles the QB. The question M5 answers is whether the rest of the lineup adds anything the model doesn't already know. Two places in our own results say it should:

- **Early season.** In weeks 1 to 4 on DEV, A4s scores **0.2243**, which is *worse* than Elo alone at **0.2229**. Our efficiency ratings start each season from a shrunk copy of last season, so they don't know that a team lost three starters in free agency or added a star. The No-Elo study found the same thing from the other side: Elo's whole advantage is its long memory in weeks 1 to 9.
- **Lineup changes during the season.** A4s sees a team's recent efficiency, which reflects the players who were playing. When a starting left tackle and a top corner sit out, the model has no idea until the results come in, a week or two late.

**Honest expectation:** an improvement of about **0.001 to 0.003 Brier** over A4s, concentrated in weeks 1 to 4 and in games with several starters out. That is smaller than the QB term was worth (about 0.0027). Non-QB players matter one at a time far less than a QB, and the market prices injuries too.

---

## 2. The plan in three parts

1. **Player values (section 3):** turn each player's per-game stat lines into a value in a common unit, EPA per play contributed to his team.
2. **Who plays (section 4):** build each team's expected lineup before each game from the depth chart, the injury report, injured reserve, and the game-day inactive list.
3. **Game-model features (section 5):** add up the values of the expected lineup, compare that with the lineup the team's ratings already reflect, and feed the difference to the game model. A preseason version of the same comparison fixes the early-season problem.

Every number is computed **as of the game**, using only games already played, the same leakage rule as M1 to M4, and every new builder must pass `leakage_check`.

---

## 3. Player values from per-game statistics

### The common unit

Every player's value is expressed as **EPA per team play above a replacement-level player at his position**. "Replacement level" is what a typical backup produces (the same idea as the QB prior in M3). This unit matters because it lets the values add up: a lineup's value is the sum of its players' values, and the difference between two lineups is a difference in expected EPA per play, the same thing the team ratings measure.

### How credit is assigned, by position

| Position | Per-game credit | The problem it solves |
|---|---|---|
| QB | EPA per dropback (existing M3 method); CPOE added from 2006 as a test | Already built |
| RB | Rushing EPA per carry, plus receiving EPA per target, each minus what the team's *other* backs produce behind the same line in the same games | Separates the back from his blockers |
| WR, TE | Receiving EPA per target minus his QB's average EPA per target to other receivers, times his target share | Separates the receiver from his QB: a great QB makes every receiver look good |
| DL, LB | EPA of plays where he is credited with a sack, QB hit, tackle for loss, or forced fumble, per team defensive play | Rewards plays that change outcomes; plain tackles mostly measure volume, so they are left out |
| CB, S | EPA of plays where he is credited with an interception or pass defensed, per team defensive play | Sparse (credit is recorded on only about 30% of incompletions), so these values lean hard on the position prior |
| K | Field goals made above the expected rate for each distance band, converted to points | Kickers swing close games |
| OL | **No individual stats exist.** Value comes only from the with/without step below | |

**Era gaps.** QB hits and tackles for loss are missing from 2003 to 2008, and incomplete-pass targets from 2003 to 2008. Each stat is used only in the seasons where it is recorded, and the position prior absorbs the rest.

### Shrinkage

Each player's raw per-game credit is averaged with recency weighting and shrunk toward replacement level:

```
value = (n * raw + k_pos * replacement_pos) / (n + k_pos)
```

`n` is his weighted games (or plays). `k_pos` is a separate amount of prior belief for each position group, because a receiver's value stabilizes much faster than a safety's. This is the same formula as the QB value.

### The with/without layer (plus-minus at the game level)

Box scores can't see linemen, and they barely see corners. So a second layer looks at **team results with and without each starter**:

```
team offensive EPA/play residual (after opponent adjustment) =
    sum over active starters of  effect[player]  +  noise
```

This is a ridge regression across all team-games. The twist: each player's effect is shrunk toward **his box-score value** from above, not toward zero. Where the box score is informative (receivers), the with/without data barely moves it. Where the box score is blind (linemen), the with/without data is all there is, and heavy shrinkage toward the position average keeps one fluky game from defining a player. The data-scope check found 231 regular linemen with 6 or more missed games from 2016 to 2025. That's enough for a position-level effect and rough individual values, and no more.

### Tuning without touching the test seasons

The knobs (`k_pos`, recency half-life, ridge strength) are tuned on **2001 to 2008**, using a target that doesn't involve game outcomes: how well a player's value predicts **his own next 4 games of credited EPA**. That keeps DEV honest, the same approach as M3-D2.

---

## 4. Who plays: the pregame lineup

The data-scope research showed that no single source is reliable, but layered together they are:

| Layer | Available | What it does | Measured reliability |
|---|---|---|---|
| Depth chart | 2005 onward (2004 dropped; week-label shift corrected) | Lists the expected starters | Right about 92% of the time overall, but only 57% when a QB change happened |
| Injured reserve and other roster statuses | 2002 onward | Removes players who can't play | Near certain |
| Injury report, final status | 2009 onward | Out removes a player (absent 100% of the time); Doubtful removes him (99%); Questionable gives him a **67% chance** of playing | Measured on 2016 to 2025 |
| Game-day inactives | 2020 onward | Removes inactive players | Never wrong when it flags someone |

Each player gets a **probability of playing**, not a yes or no, and the lineup value uses those probabilities. A Questionable star counts as two-thirds of himself.

**Rotational positions.** A "starting" running back plays about 57% of snaps, a defensive lineman about 62%. Each position slot gets an expected share of team plays. Under decision M5-D1, that comes from public usage stats (carries, targets, credited plays), not from snap counts.

**Timing.** The injury report is final on Friday, and inactives come out 90 minutes before kickoff. The Sunday 8 a.m. run will have the injury report but **not** the inactives for early games. See decision M5-D2.

---

## 5. New features for the game model

1. **`lineup_delta_diff`:** (expected lineup value − the lineup value the team's ratings already reflect), home minus away, for all non-QB starters. "Already reflect" means the value of the lineups that played in the games the rating is built from, weighted the same way. This is the same idea as `qb_delta`. The QB keeps its own term, because it is so much larger.
2. **`preseason_change_diff`:** the value of this season's projected lineup minus last season's, home minus away. Player values follow players to their new teams, so free agency, trades, retirements, and rookies (at a replacement-level prior, adjusted by draft slot if that tests well) all show up. Its weight fades as current-season data arrives, so it matters in weeks 1 to 4 and barely at all by week 10. The model learns how fast.
3. **Later, as a test:** separate offense and defense deltas.

### The experiment ladder

Scored on the same games with paired CIs, as before (window in M5-D3):

| Step | Model | What it tells us |
|---|---|---|
| B0 | A4s | The reference |
| B1 | + `lineup_delta_diff`, box-score values only | Do the per-player stats carry signal about who's missing? |
| B2 | B1 with with/without values (linemen included) | Does the plus-minus layer add the linemen? |
| B3 | B2 + `preseason_change_diff` | Does the roster-change prior fix weeks 1 to 4? |
| B4 | B3 with offense and defense split | |
| B5 | B3 with Questionable as 0/1 instead of 67% | Do probabilities beat yes or no? |
| B3f | B3 without inactives (the "Friday-only" version) | What the live Sunday-morning run can actually know |

The one-SE rule picks the final model. Results are reported by week bucket (1 to 4, 5 to 9, 10 to 18) and for games where two or more starters are out, because that's where the gain should show up.

**Acceptance:** the chosen model beats A4s on the evaluation window, with a paired 95% CI that excludes zero and the sample-size-aware calibration check passing. Then one pre-registered holdout run on 2020 to 2025, reporting both the full version and the Friday-only version.

---

## 6. How it goes live

The 2026 live record is frozen on A4s, a decision from M4. M5 runs as a **shadow model** (decision M5-D4): each weekly run also logs M5's prediction in the ledger, under its own model version, before kickoff. It's scored live next to A4s but isn't the headline number. If it beats A4s on the holdout and holds up in the 2026 shadow record, it becomes the headline in 2027.

The site can show a "lineup watch" on each game card: the starters out or questionable and what they're worth in win chance. This is the most visible payoff of the whole milestone, and it's useful even when the probability gain is small.

---

## 7. What could go wrong

| Risk | Guard |
|---|---|
| Receivers get credit their QB earned | Receiver value is measured relative to his QB's average with other receivers |
| Linemen values are mostly noise | Heavy shrinkage toward the position average; report how far individual values actually move |
| The lineup quietly uses after-the-fact information | Every layer is filtered as of the game; inactives are a stated decision (M5-D2), and the Friday-only version is always reported |
| Old depth charts are stale (2001 to 2004) or mislabeled by a week | Use 2005 onward only, with the week-shift rule, both verified in the data scope |
| DEV has no inactives (they start in 2020), but the holdout does | Report the Friday-only version (B3f), which uses the same layers in both eras, so the comparison is fair |
| The gain is too small to prove | A smaller evaluation window means wider intervals. We report what we find, and the shadow record in 2026 adds evidence. |
| Defensive credit is thin and changes across eras | Era-aware stats, strong position priors, and a with/without layer that doesn't depend on stat credit |

---

## 8. What gets built

```
nflelo/ml/players/credit.py       # per-player, per-game credited EPA by position (section 3 table)
nflelo/ml/players/value.py        # recency-weighted, shrunk values; replacement levels; with/without ridge
nflelo/ml/players/lineup.py       # pregame lineup with play probabilities (depth chart + injuries + IR + INA)
nflelo/ml/features/roster.py      # lineup_delta_diff, preseason_change_diff
scripts/ml_tune_players.py        # 2001-2008 tuning on next-4-games player credit
scripts/ml_m5.py                  # ladder B0-B5 + B3f, sign-off, dry run
notebooks/05_player_values.ipynb  # walkthrough: crediting, shrinkage, linemen, lineups, results
tests/ml/                         # leakage checks for every builder, credit-sum checks, lineup rules
```

Built by Opus agents, reviewed by me, and committed only after the checks pass.

---

## 9. Decisions for Walker

- **M5-D1. License for snap shares.** How much a rotational starter plays is only measured in participation data (CC BY-SA). Using it would make the game model itself share-alike. Using public usage stats instead keeps it CC BY.
  - **(a)** CC BY only, estimating playing time from carries, targets, and credited plays. Participation is used only to check how good that estimate is, never as an input.
  - **(b)** Use participation snap shares directly. They're more accurate, but the game model and its predictions become CC BY-SA.
  - **Recommended: (a)**, with a later test of (b) to see whether the accuracy is worth the license.
- **M5-D2. Are game-day inactives and the final injury report "pregame facts"?** This extends M3-D1 (the starting QB). **Recommended: yes**, with the Friday-only version (B3f) always reported, so we know what the live Sunday-morning run can actually deliver.
- **M5-D3. Evaluation window: DEV 2012 to 2019, compared with A4s on the same games.** Injury reports start in 2009, so the lineup features need a few seasons of training data before they can be scored. Fewer games means less power to detect a small gain. **Recommended: yes.**
- **M5-D4. Shadow model in 2026.** Log M5's predictions in the ledger every week beside A4s, scored live but not the headline, and promote it in 2027 only if it earns it. **Recommended: yes.**

## 10. Concept checkpoints

- The credit-assignment problem: why a receiver's stats are partly his QB's.
- Replacement level, and why values must be measured above it for them to add up.
- Shrinkage with a different prior strength for each position.
- Plus-minus regression with an informative prior: ridge that shrinks toward a box-score estimate instead of zero.
- Probabilistic lineups, and why a Questionable player counts as two-thirds.
- Shadow deployment: earning promotion with a live record.
