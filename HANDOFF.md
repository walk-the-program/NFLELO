# Handoff log

Newest entry first. Append only; never edit or delete past entries.

## 2026-10-05 · ML build M1-M7 (live predictions, playoff odds, player and play models, decision support)

**Did:**
- Built ML milestones M1-M7 under `nflelo/ml/`: data, as-of, features, models, eval, live, sim, players, plays, wp, decisions.
- Scripts are `scripts/ml_*.py`. The notebooks are `notebooks/01-10`.
- Every method write-up is in `context/ml-*-method.md`. The results scoreboard is `context/ml-results.md`.
- Live 2026 pipeline:
  - `scripts/ml_predict.py` and `.github/workflows/weekly.yml` (Wed 14:00 and Sun 12:00 UTC);
  - the append-only ledger `experiments/live/2026.csv`, the shadow ledger `2026_shadow.csv`, and `sim_2026.csv`;
  - the site has the model, spreads, rest of season, and playoff odds.
- Competitor review: `context/competitor-nfelo.md`.

**Verified:**
- `pytest -q`: 296 passed (run at the end of the session).
- Every holdout (2020-2025) ran once on a clean commit, after its rules were committed and a dry run reproduced dev.
  - **Passes:** M3, playoff odds, M6, the M7a WP model, M7b.
  - **Fails:** the M4 margin model, M5b, and M7a conversion.
  - M5 is a negative result on dev, and M3b missed narrowly on dev (no holdout).
- The weekly GitHub Action succeeded on two manual runs.

**Decided:**
- Walker set the direction: a portfolio for NFL teams, model first, no betting features. Most later choices were made by Claude under Walker's standing delegation and are logged in the `CONTEXT.md` decision log.
- M3b was not given a holdout run after its narrow dev miss (no moving the goalposts). It runs as a 2026 shadow model instead.
- The 2026 live headline stays frozen on A4s.
- The M7a sign-off's first attempt was killed by the 2-hour tool limit before any outcome printed. It was restarted as a detached process; long runs need `nohup`.

**Next:** fix M7a's fourth-down conversion mix (use teams' actual 4th-down run/pass mix, not the league share) and test it on the 2026 live season, since the holdout is spent. Then the site track: a model performance page, game pages, and team pages.

**Blocked on Walker:** hosting (Cloudflare Pages account and domain), and the GitHub cache purge request (drafted in `context/pending/github-purge-request.txt`; needs his signed-in browser).

## 2026-10-03 · Elo v2 rebuild, NFLELO demo site, data licensing cleanup, ML plan

**Did:**
- Moved the 2025 paper, code, and outputs into `legacy_2025/`. Added `CONTEXT.md` (the general context doc) and `CLAUDE.md`, which imports it.
- Built Elo v2 (`nflelo/`, `scripts/build.py`, `scripts/tune.py`). 1999+ comes from nflverse. 1970–1998 comes from FiveThirtyEight's CC BY 4.0 file, stored in `data/sources/` with a provenance README and sha256.
- Added `scripts/dashboard.py`, which writes `outputs/dashboards/team_summary.png`.
- Built the NFLELO one-page site (`site/`, fed by `scripts/export_site.py` → `site/data/*.json`) in Walker's design tokens: Figtree/Nunito Sans, the navy palette, 0 radius, up to 2400px wide.
- Removed the Pro-Football-Reference-derived files (`NFLELO_data.xlsx`, both `comprehensive_nfl_data.csv`) from the public GitHub repo and its full history (filter-branch plus force-push, with Walker's approval). They remain local and are gitignored.
- Wrote the ML plan: `context/ml.md`.

**Verified:**
- `pytest`: 62 passed (last run after the history purge).
- `scripts/build.py --offline` exits 0 both with and without `legacy_2025/NFLELO_data.xlsx` present.
- Held-out 2010–2025 REG Brier: Elo v2 0.2201, legacy 0.2244, market 0.2104 (4,174 paired games, SE of the difference 0.0014).
- FiveThirtyEight vs legacy spreadsheet, 1970–98 REG: 6,140 of 6,140 games match. FiveThirtyEight vs nflverse, 1999–2022 REG: 6,151 of 6,151 match.
- GitHub: none of the 17 commits on `main` contains the removed files.
- Old commit `596daea` is still reachable by direct link, because GitHub caches commits.
- Site checked in headless Chromium at 390/1440/2000px, light and dark, and in the browser pane. No console errors.

**Decided:**
- **Elo default uses online (learned) home-field advantage**, not the tune-window winner's fixed 65. They tie within 0.0001 Brier, and a fixed 65 implies a ~59% home win rate against ~53.6% in the 2020s.
- **1970–98 comes from FiveThirtyEight (CC BY 4.0), not the legacy spreadsheet.** The spreadsheet came from Pro-Football-Reference, whose terms bar substitute databases and ML use.
- **ML avoids PFR-derived nflverse datasets** (snap counts, PFR advanced stats) for the same reason.
- **Share-alike data is approved.** Walker accepted CC BY-SA for the personnel-based work (M5b, M6, M7). The first ML scope (M1–M4 plus M5a) uses only CC BY data, so it stays unrestricted.
- **Player value is split by license:** M5a (QB and box-score value, CC BY) and M5b (on-field plus-minus from participation data, CC BY-SA).
- **Not done without Walker's OK:** the site is not hosted publicly yet, and the Wednesday auto-update is not built. GitHub Pages would make it public.

**Next:** Start the ML chat. Read `CONTEXT.md`, then `context/ml.md`. Settle open decisions D2–D6 in section 10 with Walker (D1 is decided), then build M1 (data layer and leakage test) and M2 (evaluation harness that reproduces Elo 0.2201 and market 0.2104). Nothing blocks it.

Open items Walker owns:
- File a GitHub support request to purge the cached old commits (e.g. `596daea`).
- Approve or decline public hosting.
- The "NFLELO" name has "NFL" in it; consider renaming before any commercial launch.
