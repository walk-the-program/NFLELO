"""Tune the M3 rating knobs on 2000-2005 only (decision M3-D2).

    python scripts/ml_tune_ratings.py            # grid search, print, log a "tune" run
    python scripts/ml_tune_ratings.py --no-log   # print only

Two stages, both scored on REG games of 2000-2005 with targets that are not
game outcomes, and with no data from 2006 or later loaded at all:

1. lambda, half_life, rho (team ratings, `opponent_adjust`): each week's
   ratings, fit on plays before that week's as_of, predict that week's per-game
   EPA-per-play margins (home offense minus away offense, garbage time out).
   Loss: mean squared error over games.
2. k (QB shrinkage, `qb`), with the chosen half_life and rho: each starter's
   value before the week predicts his own EPA per dropback in that game.
   Loss: dropback-weighted mean squared error.

The chosen values are the grid minimums. They are copied by hand into
`opponent_adjust.TUNED` and `qb.TUNED_K`; the script reports whether the code
constants match.
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from nflelo.ml import asof, registry  # noqa: E402
from nflelo.ml import data as mldata  # noqa: E402
from nflelo.ml.eval import windows  # noqa: E402
from nflelo.ml.features import opponent_adjust as oa  # noqa: E402
from nflelo.ml.features import qb  # noqa: E402

SEASONS = windows.TUNE                    # 2000-2005: targets
DATA_SEASONS = (1999, SEASONS[1])         # 1999 only warms up the 2000 ratings and QB values
LAMBDAS = (25, 50, 100, 200, 400, 800, 1600, 3200)
HALF_LIVES = (2, 4, 6, 8, 12, 16, 24, 48, 96, 1000)  # 1000 = practically no decay within a season
RHOS = (0.1, 0.25, 0.5, 0.75, 1.0)
KS = (25, 50, 100, 200, 300, 400, 600, 800, 1200, 1600, 3200)
PBP_COLUMNS = ["game_id", "posteam", "defteam", "play_type", "pass", "rush", "epa", "success", "wp",
               "two_point_attempt", "qb_dropback", "passer_id", "play_id"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-log", action="store_true", help="print results without writing a run file")
    a = ap.parse_args()
    t0 = time.perf_counter()
    years = range(DATA_SEASONS[0], DATA_SEASONS[1] + 1)
    pbp = mldata.load_pbp(years, columns=PBP_COLUMNS)
    sched = asof.add_asof(mldata.load_schedules(years))
    assert sched["season"].max() <= SEASONS[1] and pbp["game_id"].isin(sched["game_id"]).all()
    t_load = time.perf_counter() - t0

    # ---- stage 1: team ratings
    t1 = time.perf_counter()
    tab = oa.rating_table(pbp, sched)
    targets = oa.epa_margin_targets(tab, sched, SEASONS)
    keys = sorted(set(zip(targets["season"], targets["week"])))
    rows = []
    for lam, hl, rho in itertools.product(LAMBDAS, HALF_LIVES, RHOS):
        cfg = oa.RatingConfig(lam=lam, half_life=hl, rho=rho)
        r = oa.compute_ratings(tab, sched, keys, cfg)
        rows.append({"lam": lam, "half_life": hl, "rho": rho, **oa.score_ratings(r, targets)})
    grid = pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)
    best = grid.iloc[0]
    rcfg = oa.RatingConfig(lam=float(best["lam"]), half_life=float(best["half_life"]), rho=float(best["rho"]))
    t_stage1 = time.perf_counter() - t1

    print(f"== Stage 1: team ratings, next-week EPA margin, REG {SEASONS[0]}-{SEASONS[1]} ==")
    print(f"{len(targets)} games, {len(keys)} weekly fits per setting, grid "
          f"{len(LAMBDAS)} lambda x {len(HALF_LIVES)} half-life x {len(RHOS)} rho = {len(grid)} settings")
    print(f"Predict-zero MSE: {grid['mse_zero'].iloc[0]:.5f}")
    with pd.option_context("display.width", 160):
        print(grid.head(10).to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    edges = [k for k, vals in (("lam", LAMBDAS), ("half_life", HALF_LIVES), ("rho", RHOS))
             if best[k] in (min(vals), max(vals)) and not (k == "rho" and best[k] == 1.0)]
    print(f"Chosen: lambda {rcfg.lam:g}, half_life {rcfg.half_life:g} weeks, rho {rcfg.rho:g}"
          + (f"  (on the grid edge: {edges})" if edges else ""))
    for k, vals in (("lam", LAMBDAS), ("half_life", HALF_LIVES), ("rho", RHOS)):
        prof = grid.groupby(k)["mse"].min()
        print(f"  best MSE by {k}: " + ", ".join(f"{v:g}: {prof[v]:.5f}" for v in vals))

    # ---- stage 2: QB k
    t2 = time.perf_counter()
    qcfg = qb.QBConfig.from_ratings(rcfg, k=1.0)
    db = qb.dropback_plays(pbp, sched, qcfg)
    qg = qb.qb_game_table(db, sched)
    qt = qb.starter_targets(qg, sched, SEASONS)
    priors = qb.replacement_priors(db, range(SEASONS[0], SEASONS[1] + 1), qcfg.rookie_dropbacks)
    states, qbs = qb.week_states(qg, sched, sorted(set(zip(qt["season"], qt["week"]))), qcfg, priors)
    krows = [{"k": k, **qb.score_k(states, qbs, qt, k)} for k in KS]
    kgrid = pd.DataFrame(krows)
    best_k = float(kgrid.loc[kgrid["wmse"].idxmin(), "k"])
    t_stage2 = time.perf_counter() - t2
    print(f"\n== Stage 2: QB shrinkage k (half_life {rcfg.half_life:g}, rho {rcfg.rho:g}) ==")
    print(f"{len(qt)} starter-games, {int(qt['n'].sum())} dropbacks. Replacement priors (EPA/dropback): "
          + ", ".join(f"{s}: {v:+.3f}" for s, v in priors.items()))
    print(kgrid.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"Chosen: k = {best_k:g} pseudo-dropbacks" + ("  (on the grid edge)" if best_k in (min(KS), max(KS)) else ""))

    total = time.perf_counter() - t0
    print(f"\nTiming: load {t_load:.1f}s, stage 1 {t_stage1:.1f}s ({len(grid)} settings), "
          f"stage 2 {t_stage2:.1f}s, total {total:.1f}s")
    code_ok = (oa.TUNED.lam, oa.TUNED.half_life, oa.TUNED.rho, qb.TUNED_K) == (rcfg.lam, rcfg.half_life, rcfg.rho, best_k)
    print("Code constants (opponent_adjust.TUNED, qb.TUNED_K):",
          "match" if code_ok else f"DIFFER: code has {oa.TUNED.to_dict()} k={qb.TUNED_K}; update them")

    if not a.no_log:
        path = registry.log_run(
            "m3_tune_ratings", label="tune", seasons=SEASONS, game_ids=targets["game_id"],
            metrics={"n": int(len(targets)), "mse": float(best["mse"]), "mse_zero": float(best["mse_zero"]),
                     "r2": float(best["r2"]), "corr": float(best["corr"]),
                     "qb_wmse": float(kgrid["wmse"].min()), "qb_n_starter_games": int(len(qt))},
            features=["adj_epa_margin", "qb_value"],
            params={"chosen": {**rcfg.to_dict(), "k": best_k},
                    "grid": {"lam": LAMBDAS, "half_life": HALF_LIVES, "rho": RHOS, "k": KS},
                    "data_seasons": list(DATA_SEASONS)},
            data=registry.data_fingerprint(games_csv=False, ml_datasets=("pbp", "schedules"), seasons=years),
            notes="M3-D2 tuning: team-rating knobs on next-week per-game EPA margins, QB k on the starter's "
                  "next-game EPA per dropback. 2000-2005 REG targets; 1999 used only as history.",
            extra={"top10": grid.head(10).to_dict("records"), "k_grid": kgrid.to_dict("records"),
                   "profiles": {k: {str(v): float(m) for v, m in grid.groupby(k)["mse"].min().items()}
                                for k in ("lam", "half_life", "rho")},
                   "timing_s": {"load": t_load, "stage1": t_stage1, "stage2": t_stage2, "total": total}})
        print(f"Run written to {path.relative_to(registry.config.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
