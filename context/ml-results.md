# NFLELO Model Results: The Scoreboard

Every model below was developed on earlier seasons, then judged **once** on the locked 2020-2025 seasons, using pass/fail rules committed to the public repo before that run. The commit hashes are the receipts: anyone can check that the rules came before the results. Failures are listed with the passes. Details and full tables are in `context/ml.md`; the method write-ups are in `context/ml-*-method.md`.

Last updated 2026-10-05.

## Results

| # | Model | What it does | Locked 2020-2025 result | Verdict |
|---|---|---|---|---|
| M3 | **Game model (A4s)** | Win probability from Elo, opponent-adjusted EPA, and a starting-QB adjustment | Brier **0.2195** vs Elo 0.2231 (-0.0035 [-0.0063, -0.0007]); won all 6 seasons. Market 0.2096 | **Pass.** Live in 2026; calibration bar recorded as flawed (too tight for n) |
| M4 | **Margin model** | Full distribution of the final margin, aware of key numbers like 3 and 7 | MAE 10.126 vs Elo spread 10.185 (-0.059 [-0.150, +0.035]) | **Fail** on the primary (CI includes 0); used for spreads and simulation |
| M4 | **Playoff odds** | 20,000-season simulation with real NFL tiebreakers and strength uncertainty | Made-playoffs Brier 0.1234 vs 0.1258 without uncertainty (CI excludes 0); calibrated | **Pass.** Live on the site |
| M4 | Tiebreakers | The NFL's seeding procedure | Reproduces all 48 actual conference seedings, 2002-2025 | Exact |
| M5 | Box-score player values + pregame lineups | Who's playing, and their per-game stat value | No gain over A4s on 2012-2019 (every CI includes 0) | **Negative result.** Not used in the game model |
| M5b | **On-field player ratings (RAPM)** | Plus-minus from all 22 players on every play, 2016+ | Beats team ratings season-ahead (-0.0018 MSE), but loses to box-score values (+0.0010) | **Fail.** Kept as a player view |
| M3b | Kalman-filter team strength | A state-space rating with uncertainty | Beat A4s on dev by -0.00095 (CI upper +0.000001), short of the bar | **Not tested on the holdout** (narrow dev miss). Running as a live 2026 shadow model |
| M6 | **Play outcome model** | Yards distribution for a play from pre-snap situation and personnel | CRPS -0.018 vs the situational baseline in both views, in every season and both data eras | **Pass** |
| M7a | **Win-probability model** | Live WP with no betting-market inputs | Brier **0.1559 vs nflfastR 0.1659** (-0.0099 [-0.0146, -0.0055]); Vegas WP 0.1497 | **Pass** (WP part) |
| M7a | Fourth-down tool | Go / FG / punt in win probability, with an uncertainty band | Conversion calibration ECE 0.027 missed the bar; FG and punt pass | **Fail** (conversion component). Audit published |
| M7b | **Early-down play calling** | Run or pass by personnel, using causal methods (propensity, AIPW, off-policy evaluation) | Recommended policy +0.052 EPA/play [+0.038, +0.067] over observed; placebo includes 0 | **Pass** |

## What the evidence says

1. **The quarterback is most of the story.** In M3, nearly all of the gain over Elo comes from knowing the starting QB, and it depends on knowing who actually starts.
2. **Player-level beats team-level, but public data limits how far individual credit goes.** Box-score credit summed over the players on the field beats team ratings when predicting the next season. On-field plus-minus doesn't beat box credit, and lineup effects don't move game predictions.
3. **Elo's edge is memory.** Our features alone tie Elo overall; Elo wins in weeks 1-4 and our features win in weeks 10-18.
4. **Teams are still too conservative.** The fourth-down audit estimates about 0.03 wins per team-game given up, mostly by punting (model-estimated). The early-down policy says pass more, especially from 12 personnel on 1st and 10.
5. **The market is still better at game prediction.** Brier 0.2096 vs our 0.2195 on 2020-2025. Our models are independent of it by design, so our measured edge over Elo and nflfastR is real.

## Discipline behind the numbers

- **Leak-proof features.** Every feature is built as of the start of the game's week, and a test that corrupts future data proves it. Four real leaks were caught and fixed along the way: pre-2016 roster statuses that are season-end values, two lookups that read a player's position from later games, and missing formations that flagged fumbles.
- **Locked holdout.** Tuning happened on 2006-2019, or on earlier windows. The locked 2020-2025 seasons were scored once per milestone, under rules written down in advance.
- **Rule changes are dated.** Every rule change was made and dated before its holdout number existed, with the reason recorded: the one-sided ECE check, the 0.01 calibration floor, and the punt CRPS rule.
- **A live record that can't be faked.** 2026 predictions are logged before kickoff in an append-only ledger and pushed to GitHub, so the timestamps can be audited.
- **Licensing:** CC BY data for the game model, and CC BY-SA for the personnel-based models (M5b, M6, M7a fourth down, M7b), with credit to nflverse, NFL Next Gen Stats, and FTN.
