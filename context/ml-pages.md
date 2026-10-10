# Model pages: the data contract

Written 2026-10-10. This is the contract between `scripts/export_ml_pages.py` (which writes the data) and the site pages that read it. It covers four files in `site/data/`. Field names, units and meanings here are binding: if the exporter changes a field, this file changes in the same commit.

Contents:
1. Common rules for all four files
2. `plays.json`: play outcome explorer (M6)
3. `winprob.json`: win probability charts (M7a WP model)
4. `fourth.json`: fourth-down decisions (M7a-v2 engine)
5. `playcalling.json`: early-down play calling (M7b)
6. Refresh, runtime and caches
7. Simplifications and choices made by the exporter

## 1. Common rules

- **Audience:** NFL front offices. **No betting features:** no lines, spreads, odds, picks, ATS, units or EV appear in these files, and pages must not add them. Tests reject any market field (`tests/test_export_ml_pages.py`).
- **Every file has** `schema` (1), `page`, `license` (`license`, `url`, `credit`), `headline`, `caveats` (a list of plain sentences). **Show the credit line and every caveat on the page.**
- **Headline numbers are loaded, never recomputed.** They come from the run registry (`experiments/runs/*.json`); each carries a `receipt` (`run` file, `commit`, `window`, `logged_at`) that the page can link: `https://github.com/walk-the-program/NFLELO/blob/main/<run>`.
- **Confidence intervals** use one shape everywhere: `{"diff", "lo", "hi", "level": 0.95, "excludes_zero", "n", "clusters"}` (game-clustered bootstrap).
- **Probabilities** are 0-1 floats (rounded to 3-4 decimals). Yards are yards. Clock strings look like `"Q4 1:45"`.
- **Files are deterministic:** no timestamps. A weekly run with no new games leaves them byte-identical.
- `yardline_100` is yards from the opponent's goal line (75 = own 25; 20 = opponent's 20).

## 2. `plays.json` (M6 play outcome model)

- **Path:** `site/data/plays.json`, about 0.87 MB.
- **Refresh:** yearly, after each season's participation data release (`python scripts/export_ml_pages.py plays`). It is static during the season.
- **License:** CC BY-SA 4.0. Credit: `license.credit` (nflverse; NFL Next Gen Stats via nflverse 2016-2022; FTN Data via nflverse 2023+).
- **Model:** the M6 call-view gradient-boosted model for 2026, fit on 2016-2025 by the M6 protocol (`settings`). It is the same fitted model the 2026 fourth-down engine uses (`settings.matches_frozen_record: true` confirms its settings equal the frozen M7a-v2 record: learning rate 0.1, 15 leaves, 52 trees).

Top-level fields:

| Field | Meaning |
|---|---|
| `season_model`, `trained_on`, `n_train_plays`, `settings` | Which model this is |
| `headline` | Locked 2020-2025 result for both views (`call`, `situation`): `crps_model`, `crps_baseline`, `crps_model_minus_baseline` (CI), `p_first_ece`, `p_first_calibration_ok`, `plays`, `receipt`; `verdict` "Pass"; `plain` one-sentence summary |
| `bins` | `labels` (53 labels: `"<=-10"`, `"-9"` ... `"40"`, `"41+"`, `"TD"`), `yards` (the 52 non-TD bins' yards; bin 0 = 10+ lost, bin 51 = 41+), `td_index` 52 |
| `grid` | The axes: `down` 1-4; `ydstogo` 1, 2, 3, 5, 7, 10, 15; `yardline_100` 5, 10, 20, 35, 50, 65, 80, 95; `call` run/pass; `personnel` 11/12/21/13 with labels. Cells with `ydstogo > yardline_100` are dropped (1,664 cells remain) |
| `held_neutral` | How the other inputs were held (show this on the page) |
| `fields` | One-line meaning of each cell field |
| `cells` | One object per grid cell (below) |

Each cell:

| Field | Unit | Meaning |
|---|---|---|
| `down`, `ydstogo`, `yardline_100`, `call`, `personnel` | | The cell |
| `exp_yards` | yards | Expected yards (descriptive: 10+ lost counted as -12, 41+ as 55, a TD as the yard line) |
| `p_first_or_td` | prob | P(reaching the line to gain, or a TD) |
| `p_20plus`, `p_loss`, `p_td` | prob | P(20+ yards), P(lost yards), P(touchdown) |
| `p_turnover` | prob | P(interception or lost fumble), a separate model |
| `bins_from`, `bins` | prob | Sparse yard distribution: `bins[i]` is the probability of bin `bins_from + i` (non-TD bins only). Bins under 0.0005 at either end are trimmed. **`sum(bins) + p_td` is 1 within 0.01** (tested) |
| `n_similar` | plays | 2016-2025 plays with the same down, distance, call and personnel within 5 yards of the yard line. **Show it, and flag cells under about 30 as thin support** (545 cells have 0: e.g. 1st and 3 outside goal to go) |
| `structure_pool` | plays | Size of the pool the defensive looks were drawn from |

Held neutral: tied score, 7:30 left in the 2nd quarter, 3 timeouts each, outdoors at the median outdoor temperature and wind (58F, 7 mph), league-average team ratings (0), home averaged (half the draws home, half away). Defensive personnel, formation and box count are not fixed: each cell averages the model over 32 draws of those fields from 2016-2025 plays with the same offensive personnel and call, in the most specific situation with at least 200 plays.

**Headline to show:** "On the locked 2020-2025 seasons, the play model beat the historical down-distance-field baseline: CRPS 3.771 vs 3.790 yards (difference -0.019, 95% CI -0.021 to -0.017), in every season." Use `headline.call`.

**Caveats that must appear:** the `caveats` list (a prediction for the snap, not a promise; held-neutral inputs; thin-support cells; run/pass here is the called type, not a recommendation).

## 3. `winprob.json` (M7a win probability)

- **Path:** `site/data/winprob.json`, about 0.14 MB now (64 games), about 0.6 MB by the end of the regular season.
- **Refresh:** weekly, in the Wednesday and Sunday Action runs (`live` step).
- **License:** CC BY 4.0 (play-by-play and A4s only; no participation data). Credit: `license.credit`.
- **Model:** our WP model for 2026, fit on 2006-2025 with A4s pregame strength as an input, exactly as in M7a (`model_fit`).

| Field | Meaning |
|---|---|
| `season`, `model`, `model_fit` | Season and model facts (trained_on, n_train, n_iter, calibrated false: the frozen model is the raw GBM) |
| `updated_through` | `week`, `games`, `last_game_date`, `held_back` (games final in nflverse but not yet in the Elo pipeline; they appear next run) |
| `headline` | Locked 2020-2025: `brier_ours` 0.1559, `brier_nflfastr` 0.1659, `ours_minus_nflfastr` (CI -0.0146 to -0.0055), `ece_ours`, `plays`, `verdict` "Pass", `receipt`, `plain` |
| `definitions` | Meanings of the per-game fields (show on hover or in a methods note) |
| `games_by_excitement` | Season list sorted by excitement (highest first): `rank`, `game_id`, `week`, `home`, `away`, scores, `excitement`, `biggest_swing`, `winner_min_wp`, `overtime` |
| `games` | One object per completed REG game, by week (below) |

Each game:

| Field | Unit | Meaning |
|---|---|---|
| `game_id`, `week`, `date`, `home`, `away`, `home_score`, `away_score`, `overtime`, `tie` | | The game |
| `pregame_home` | prob | A4s pregame home win probability (matches the live ledger) |
| `t` | seconds | Regulation seconds elapsed at each snap (0 = kickoff, 3600 = end). Never decreases |
| `wp` | prob | Home win probability at each snap (the state the previous play left). The last point is the result: 1, 0, or 0.5 for a tie. About one point per snap (~145 per game) |
| `excitement` | WP | Sum of absolute snap-to-snap changes in home WP (including the last step) |
| `winner_min_wp` | prob | Lowest WP the eventual winner had (a comeback measure; null for a tie) |
| `swing` | | The largest single change: `delta_home`, `abs`, `qtr`, `clock`, `offense`, `desc` (short play text), `play_id`, `wp_before`, `wp_after`. The play is the snap that began the change |

**Headline to show:** "Our win probability beats nflfastR's public model on 2020-2025 (Brier 0.1559 vs 0.1659; difference -0.0099, 95% CI -0.0146 to -0.0055) and uses no betting-market input."

**Caveats that must appear:** overtime is not modelled (the line jumps to the result at the end of regulation); one point per snap; WP is a model estimate.

## 4. `fourth.json` (M7a-v2 fourth-down engine)

- **Path:** `site/data/fourth.json`, about 0.35 MB now (880 fourth downs), about 1.0 MB by the end of the regular season.
- **Refresh:** the live part weekly (Wednesday and Sunday Action runs); the chart once per season (kept unchanged while the engine is unchanged).
- **License:** CC BY-SA 4.0 (the go option uses M6). Credit: `license.credit`.
- **Engine:** M7a-v2 candidate v2ad, frozen in `experiments/m7a_v2/frozen_2026.json`, with the 2026 WP model, the M6 model above, and field-goal and punt models fit on 2006-2025 (their kicker and punter values use earlier games only). **No bootstrap band** anywhere in this file.

| Field | Meaning |
|---|---|
| `label` | Always `"model-estimated"`. **Every number from this file must carry that label on the page** |
| `status` | The forward test. `forward_test` "pending", `plain` (show this sentence verbatim near the top), `earliest_run` 2027-02-15, `pooled_2026_2027_secondary_earliest` 2028-02-15, `frozen_commit`, `why`, `components_status` |
| `engine` | Model facts: `wp`, `m6`, `engine` (`name`, `candidate`, `frozen_at`, `frozen_file`, `frozen_sha256`) |
| `headline.holdout_audit_v1` | The locked 2020-2025 audit with the v1 engine (model-estimated): 22,576 fourth downs, `matched_share` 0.667, `matched_share_clear_calls` 0.798, `tossup_share` 0.518, `wp_lost_total` 97.9 wins, `wp_lost_per_team_game` 0.030, `wp_lost_by_choice` |
| `headline.wp_model` | Same block as `winprob.json` `headline` |
| `tossup_rule` | `margin_below` 0.02 and its plain explanation |
| `chart` | The static decision chart (below) |
| `live` | The 2026 fourth downs so far (below) |
| `definitions`, `caveats` | Show both |

**`chart`:** `label` ("Point estimates without the uncertainty band (no bootstrap)"; show it), `as_of`, `close_call_margin`, and `presets`, three game states:

- `tied_q2`: tied, 7:30 left in the 2nd quarter (averaged over home/away and who receives the second-half kickoff)
- `down4_early_q4`: down 4, 13:00 left in the 4th quarter (averaged over home/away)
- `up3_late_q4`: up 3, 4:00 left in the 4th quarter (averaged over home/away)

Each preset is columnar: `columns` names the arrays; `yardline_100` (1-99) and `ydstogo` (1-10, only where `ydstogo <= yardline_100`; 945 cells), `best` ("go", "fg", "punt"), `margin` (WP of best minus second best), `wp_go`, `wp_fg`, `wp_punt`, `p_conv` (P(convert) if going), `p_fg` (P(make)), `close_call` (margin under 0.02). Teams are league average, the kicker and punter league average, outdoors at 60F with a 5 mph wind, even teams. (`fingerprint` and `season` are internal.)

**`live`:** `label` "model-estimated"; `updated_through` (`week`, `games`, `last_game_date`, `held_back`); `summary` (`fourth_downs`, `games`, `matched_share`, `tossup_share`, `wp_lost_total`, `wp_lost_per_team_game`, `choices`, `recommended`); `teams`; `plays`.

`teams` (one per team): `games`, `fourth_downs`, `model_said_go`, `went_when_model_said_go`, `go_rate_when_model_said_go`, `model_said_go_clear` and `go_rate_when_model_said_go_clear` (only calls with margin of at least 0.02), `went_when_model_said_kick`, `matched_share`, `wp_lost_total` (wins, model-estimated), `wp_lost_per_game`.

`plays` is columnar: `columns` and `rows` (zip them). Columns: `game_id`, `play_id`, `week`, `team`, `opponent`, `home` (bool), `qtr`, `clock`, `score_diff` (team minus opponent), `ydstogo`, `yardline_100`, `choice` (what the team did: go/fg/punt), `recommended`, `second`, `margin`, `wp_go`, `wp_fg`, `wp_punt`, `p_convert`, `p_fg_make`, `wp_given_up` (best minus the choice; 0 when matched), `matched`, `tossup`, `desc` (short play text). Only completed plays of completed games are included (tested).

**Headline to show:** the status sentence first, then the 2026 summary labeled model-estimated, then the v1 holdout audit as context: "2020-2025: teams matched the model on 67% of fourth downs and gave up an estimated 0.03 wins per team-game, mostly by punting (model-estimated)."

**Caveats that must appear:** everything is model-estimated (the other choice is never observed); teams know things the model does not; the conversion component is under a forward test until at least 2027-02-15; the chart has no uncertainty band and close calls are toss-ups; ties, overtime, penalties and kneel-downs are excluded.

## 5. `playcalling.json` (M7b early-down play calling)

- **Path:** `site/data/playcalling.json`, about 0.13 MB.
- **Refresh:** yearly (static: built from the logged 2020-2025 sign-off run and its saved outputs in `data/raw/ml/m7b/`; the exporter checks they agree with the logged run).
- **License:** CC BY-SA 4.0 (personnel comes from participation data). Credit: `license.credit`.

| Field | Meaning |
|---|---|
| `headline` | `ope_epa_per_play` (+0.052, CI +0.038 to +0.067), `ope_success_rate` (+0.041), `placebo_epa_per_play` (CI includes 0), `sensitivity` (overlap threshold 0.10; weights trimmed at 10), `share_of_plays_policy_applies` 0.667, `epa_per_affected_play` 0.079, `rule`, `per_season` (OPE and placebo by season 2020-2025, `clear_share`), `verdict` "Pass", `receipt`, `plain` |
| `definitions` | The sample, the 60 cells (5 down-distance x 4 zones x 3 score states, each with a label), the 12 actions (personnel group x run/pass), and the meaning of gain, best, clear, candidate and the action rows |
| `seasons` | 2020-2025 |
| `cells` | 60 objects: `cell`, `down_distance`, `zone`, `score`, `by_season` |
| `team_view_2025` | Descriptive 2025 team view (below) |

`cells[].by_season["2020"...]`: `best` (an action like "12_pass", or null), `clear` and `label` ("clear best" or "no clear best"), `gain`, `gain_lo`, `gain_hi` (EPA per play vs the current mix, 95% CI), `n_train_plays`, `candidates`, `actions` (rows of `[action, gain, gain_lo, gain_hi, share_called]` for every candidate, best first; `share_called` is the share of the cell's training plays that called it). Season S's recommendations come from seasons 2016..S-1 only.

`team_view_2025`: `pass_cells` (the 17 cells where 2025's clear recommendation is a pass, with the recommended action), `league` (`plays`, `pass_rate`, `exact_match_rate`, `early_down_pass_rate_all_cells`), `teams` (sorted by `pass_rate`: `team`, `plays`, `pass_rate`, `vs_league`, `exact_match_rate`, `early_down_pass_rate_all_cells`), `fields`. Label it **descriptive**: these are observed 2025 REG play calls, not a model evaluation, and `vs_league` does not adjust for each team's mix of cells.

**Headline to show:** "Following the recommendations would have gained about 0.05 EPA per early-down play on the unseen 2020-2025 seasons (+0.052, 95% CI +0.038 to +0.067); a placebo with shuffled actions shows no gain. The advice is consistent every season: pass more on early downs, mostly from 12 and 11 personnel, especially on 1st and 10."

**Caveats that must appear:** unmeasured confounding; defenses adapt (marginal shifts, not a fixed strategy); the placebo's upper end hints at a small leftover bias (about 0.005 EPA per play); "no clear best" means the data can't rank the actions, not that they are equal.

## 6. Refresh, runtime and caches

| File | Command | When | Runtime (local, warm cache) |
|---|---|---|---|
| `plays.json` | `export_ml_pages.py plays` | yearly | 40 s (cold: 330 s, building the M6 table and fitting the model) |
| `playcalling.json` | `export_ml_pages.py playcalling` | yearly | under 1 s |
| `winprob.json` | `export_ml_pages.py live` (or `winprob`) | weekly | 6 s (cold: +10 s WP fit) |
| `fourth.json` | `export_ml_pages.py live` (or `fourth`) | weekly | 30 s (new games add about 1 s per 60 games; first chart 4 s) |

The weekly Action runs `python scripts/export_ml_pages.py live --no-refresh --budget 900` after `ml_predict.py` and before `export_site.py`, with `continue-on-error: true` and a 20-minute step timeout: a failure never blocks the site update or the ledger commit, and the existing `git add site/data` commits the two live files. A warm weekly run adds about 40 seconds.

Caches (gitignored, `data/raw/ml/pages/`, kept between Action runs by the existing `data/raw` cache): `wp_2026.joblib`, `m6call_2026.joblib`, `fourth_live_2026.parquet` (valued fourth downs by game; only new games are valued). The model caches rebuild when the season, the scikit-learn version or `CACHE_VERSION` changes; the fourth-down rows and the chart are also recomputed whenever the engine fingerprint (frozen file, WP and M6 model facts, toss-up cut) changes. The first Action run on an empty cache downloads the participation data and builds the M6 table (several minutes); the budget and timeout bound it, and the files stay as committed if it does not finish.

## 7. Simplifications and choices

- **Toss-up without a bootstrap:** margin under 0.02 WP. That cut best matches the 50-refit bootstrap toss-up flags on the 2018-2019 dev seasons (77% agreement). It marks about 59% of 2026 fourth downs so far as toss-ups, against 52% for the bootstrap on 2020-2025.
- **A4s for 2026 games** is the walk-forward A4s (fit on 2001-2025), computed as in M7a; it matches the live ledger's logged probabilities to within 0.00005.
- **Held-back games:** a game final in nflverse but not yet in `data/games.csv` waits one run (A4s needs its Elo input). In the Action, `build.py` runs first, so this only happens locally.
- **Tie games** are excluded from the fourth-down rows (as in M7a); their WP series is shown, ending at 0.5.
- **Unknown kicker or punter** (chart only): league-average value.
- **No new numbers on 2020-2025:** the only computation on those seasons is the descriptive 2025 team view of observed play calls.
