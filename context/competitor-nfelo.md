# Competitor Review: nfelo (nfeloapp.com)

Reviewed 2026-10-05 in the browser: the games page, model performance, power ratings, the odds calculator, and the full navigation. These are observations, not their source code.

## What they have

**Breadth is their main strength.** About 20 sections, grouped in a left sidebar:
- **Model:** game projections, model performance, a confidence pool, pick'em, and a "+EV" betting card.
- **Teams:**
  - power ratings, a wide table with Elo, QB adjustment, spread value, week-over-week change, offensive and defensive EPA split by pass and rush, points, and Pythagorean wins;
  - EPA tiers, tendencies, strength of schedule, and win totals.
- **QBs:** rankings (QB Elo plus EPA), era-adjusted QB Elo, and a QB compare tool.
- **Other:** receiving leaders, head coaches with ATS records, a home-field tracker, and live QB EPA.
- **Tools:** odds, cover-probability, hold, hedge, parlay, and passer-rating calculators, each with a long explainer page that's built for search traffic.
- **Per-game box-score pages:** one for every game (e.g. `/games/steelers-vs-browns-box-score-week-04-2026/`), which also builds search traffic.

**Audience:** bettors. Everything is framed in spreads, ATS records, units, closing line value, and EV%.

## What their numbers show

- **Their model mostly repeats the market.** For 2026 week 4, the model's spread equaled the closing line in 13 of 16 games. Their own performance page shows winner accuracy 0.11 points *below* the closing line, and spread error (MAE 10.1) 0.01 *worse* than the closing line. The model is anchored to Vegas and adds only small nudges.
- **The performance claims can't be checked.** The season table runs from 2009, with ATS 56.8% against the opening line and +169 units. The page doesn't mark which seasons were predicted live and which were reconstructed afterward, and there is no timestamped record of when predictions were made.
- **They don't measure probability quality.** There's no Brier score, log loss, or calibration, and no confidence intervals on anything. The ATS and units figures have no uncertainty attached.
- **No playoff odds.** The navigation has win totals, but no season simulation with playoff, division, or Super Bowl odds.
- **QB is the only player adjustment.** There's no lineup or injury view beyond the QB.

## Design

- **A generic dark dashboard template:** sidebar, cards, dense tables, small type. It looks like a stock component kit rather than a designed product.
- **Ad-heavy:** display ads between game groups, and an "Offer Center" with sportsbook promotions.
- **NFL team logos everywhere.** That's a trademark risk; we decided not to use logos.
- **Very wide tables** (20+ columns of power ratings), with no headline or explanation of what matters.

## Where we're already ahead

1. **An independent model.** Our win probabilities and spreads don't use the betting line as an input (D3). The comparison with Vegas is real, not a market echo.
2. **Receipts.** A public, timestamped prediction ledger (no row can be written after kickoff), pre-registered holdout tests, confidence intervals, and calibration, including the results that failed (the M4 margin model).
3. **Playoff odds** from 20,000 simulated seasons, using real NFL tiebreakers that reproduce every actual seeding from 2002 to 2025.
4. **History back to 1970** (theirs starts in 1999 or 2009), with luck, parity, and records.
5. **An editorial design:** a computed plain-English headline for every section, wide layouts, Walker's own type and color system, no ads, no logos.
6. **Coming in M5:** a lineup watch showing which starters are out and what that's worth, beyond the QB.

## Gaps to close, by value per effort

1. **A game page for every matchup:** model vs Elo vs Vegas, QB and lineup notes, win probability, margin distribution, and the head-to-head history. This adds depth and search traffic.
2. **Team pages:** rating over time, adjusted EPA splits, remaining schedule with odds, and the playoff-odds trend.
3. **A model performance page:** the live 2026 ledger record, clearly separated from backtests; a calibration plot; Brier and accuracy vs Elo and Vegas with confidence intervals; and the full history of pre-registered tests. This is where we clearly beat them.
4. **QB rankings** from the M3 QB values (shrunk EPA per dropback), with a career view.
5. **An EPA tiers chart:** adjusted offense vs defense scatter from our ridge ratings. It's cheap to build.
6. **Strength of schedule:** remaining and played, from the simulation.
7. **Betting-oriented views** (edge vs the line, ATS record). This is a product decision for Walker: it changes the audience, and if ads or affiliate offers are ever involved, the legal exposure too.
8. **Calculators:** low value. Only build them if search traffic becomes a goal.
