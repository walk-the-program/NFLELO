# M5 Data Scope: Pregame Lineups and Per-Player Game Stats

Research done 2026-10-04 (read-only; scratch scripts were not kept in the repo). It answers whether free, CC BY data can support Walker's approach: value every player from per-player, game-level statistics, then account for each team's pregame starting lineup in the game model. This is the input to the M5 method write-up.

## Bottom line

- **Feasible.** Every source we need is CC BY and keyed by the same NFL GSIS player ID, so no crosswalk is needed.
- **The lineup has to be built in layers.** The depth chart alone is weak exactly when it matters: it named the right QB in only 57% of the games where the starter changed. The depth chart plus the injury report plus game-day inactives is reliable: those flags are never wrong when they mark a starter out, and they catch about 78% of actual absences.
- **Offensive linemen have no individual stats.** They need the with/without method, which will be noisy and heavily shrunk toward a position average.
- **Snap shares (how much a starter actually plays) exist only in participation data (CC BY-SA).** Walker approved share-alike for personnel work (D1), so M5 can use it, but that makes M5 share-alike.

## Sources and verdicts

| Dataset | Years | What it gives | Verdict |
|---|---|---|---|
| Weekly player stats (`load_player_stats`, week level) | 1999-2026 | One row per player per game *if he recorded a stat*. Offense, defense, kicking and punting in one file, with EPA for passers, rushers and receivers | Usable |
| Play-by-play player IDs | 1999+ | Passer, rusher, receiver, sack, QB hit, TFL, tackles, pass defensed, INT, fumbles, kicker, punter, penalty | Usable. Gaps: 2003-2008 lack incomplete-pass targets and TFL credit; 2003-2005 lack QB hits |
| Depth charts (`load_depth_charts`) | 2001-2024 weekly; 2025+ daily ESPN snapshots | Listed starters by position | Usable for 2005+. Drop 2004 (frozen); 2001-2003 are weak. Two formats need two code paths |
| Injury reports (`load_injuries`) | 2009+ | Final Friday status per player | Usable |
| Weekly rosters (`load_rosters_weekly`) | 2002+ | ACT, RES (IR), INA, and so on | Usable **from 2016 only** for statuses (correction 2026-10-05: before 2016 the status is effectively the season-end status; 35-45% of RES player-weeks played that week). INA (game-day inactive) only from 2020 (partial in 2019) |
| Players table (`load_players`) | all | IDs, positions | Usable |
| Next Gen Stats | 2016+ | QB and receiver tracking stats, qualifying players only | Thin; use with care |
| Participation | 2016-2025 | Who was on the field for every play | CC BY-SA: validation, or M5 if share-alike is accepted for it |
| Snap counts, PFR advanced, draft picks, combine | | | **Banned** (Pro-Football-Reference) |
| ESPN QBR | 2006+ | | **Not usable**: no license on the source repo |
| ff* datasets | | | **Avoid**: GPL-3.0 |
| Contracts (OverTheCap) | | | Not usable until the license is checked |

## Key measurements

**Depth chart QB1 vs the actual starter** (regular season, 2001-2025): right 92.0% of the time, vs 90.2% for "last game's starter starts again". On the 1,186 games where the starter changed, it was right only 56.7%.

**Week labels.** In 2007-2013 and 2015-2024, depth-chart week L matches schedule week L-1. Detect this when the chart has one more week than the schedule, and shift.

**Injury report status vs absence** (2016-2025):

| Status | Absent from the game |
|---|---|
| Out | 100% |
| Doubtful | 99% |
| Questionable | 33% |
| Practice report only | 9% |

**How much listed starters play** (participation, 2016-2025): QB, OL and S have a median of 100% of snaps. CB is 95% and WR 79%. RB (57%), TE (67%), DL (62%) and LB (78%) are rotational, and FB plays 15%. About 7% of listed starters at every position play zero snaps. In 2020-2025, 97% of those were flagged beforehand by the injury report (Out or Doubtful) or the inactive list.

**Offensive line with/without sample** (2016-2025, 4,561 team-games):

| Regulars missing | Share of team-games |
|---|---|
| 0 | 45% |
| 1 | 39% |
| 2 | 13% |
| 3 or more | 3% |

Of 603 regular linemen, 231 missed at least 6 games. Pregame flags catch 78% of actual misses, with no false alarms.

## Risks

1. Depth charts lag on lineup changes. Layer the injury report, IR, and inactives on top.
2. Timing can't be fully proven before 2025. The 2001-2024 depth charts and the weekly rosters have no timestamps. Injury timestamps are missing for 2009, 2025 and 2026. Only the 2025+ depth charts are truly timestamped.
3. "Starter" isn't an every-down player at RB, TE, DL, LB, FB or nickel back. Weighting by expected snaps needs participation data.
4. Individual credit is sparse for defenders, missing for linemen, and inconsistent across eras (2003-2008 gaps).
5. Offensive line effects will be noisy. Expect heavy shrinkage toward the position average.

## Not verified

- When during the week the 2001-2024 depth charts were published.
- Whether INA and RES were captured before kickoff.
- The legal standing of NFL-owned NGS data and the ESPN-sourced 2025+ depth charts, beyond nflverse's CC BY license.
- How the schedule's starting-QB fields are derived (treated as ground truth).
