"""Tune the M5 player-value knobs on 2001-2008, without game outcomes (context/ml-m5-method.md, section 3).

    python scripts/ml_tune_players.py            # run both tunings, print, log to experiments/runs/
    python scripts/ml_tune_players.py --no-log

1. Box-score values: recency half-life (one, shared) and prior strength k per
   position group. Target: a player's own credited EPA per on-field play over
   his NEXT 4 games in the same season (the anchor game included), predicted
   by his shrunk value as of before the anchor's week. Score: share-weighted
   MSE per group, relative to predicting replacement level (skill = 1 - MSE / MSE_replacement).
   Anchors: every REG game a player recorded usage in, 2001-2008; history from 1999.
2. With/without ridge strength lambda (in plays). Target: each team's NEXT 4
   team-game opponent-adjusted EPA residuals (offense and defense, same
   season), predicted by intercept + sum of on-field shares x player effects
   fit before the anchor week. Seasons 2006-2008 (linemen need depth charts,
   which start in 2005; 2004 is dropped). lambda = 1e7 is the box-only limit.

Nothing here reads 2009 or later. The chosen knobs are copied into
`nflelo.ml.players.value.TUNED` by hand after review.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nflelo import config  # noqa: E402
from nflelo.ml import registry  # noqa: E402
from nflelo.ml.features import opponent_adjust as oa, roster as rf  # noqa: E402
from nflelo.ml.players import credit as cr, inputs as m5in, lineup as lu, value as pv  # noqa: E402

SEASONS = (2001, 2008)
RIDGE_SEASONS = (2006, 2008)
NEXT = 4
HALF_LIVES = [4, 8, 16, 32, 64, 128]
KS = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
LAMS = [125, 250, 500, 1000, 2000, 4000, 8000, 16000, 64000, 1e7]
GROUPS = [g for g, c in cr.COMPONENTS.items() if c]


def next_window(pg: pd.DataFrame, n: int = NEXT) -> pd.DataFrame:
    """Anchor rows (REG, played) with the next-n-games sums (same player, same season): num_c, den."""
    p = pg[(pg["game_type"] == "REG") & (pg["inv"] > 0) & (pg["share"] > 0)].sort_values(
        ["gsis_id", "season", "kick_ns"], kind="stable").reset_index(drop=True)
    g = p.groupby(["gsis_id", "season"], sort=False)
    out = p[["gsis_id", "season", "week", "group", "kick_ns"]].copy()
    den = p["share"].copy()
    nums = {c: p[f"c_{c}"] * p[f"a_{c}"] for c in cr.ALL_COMPONENTS}
    for s in range(1, n):
        den = den + g["share"].shift(-s).fillna(0.0)
        for c in cr.ALL_COMPONENTS:
            nums[c] = nums[c] + (g[f"c_{c}"].shift(-s) * g[f"a_{c}"].shift(-s)).fillna(0.0)
    out["den"] = den
    for c in cr.ALL_COMPONENTS:
        out[f"t_{c}"] = nums[c] / den
        out[f"a_{c}"] = p[f"a_{c}"].to_numpy()
    return out


def tune_box(state, avail) -> tuple[pd.DataFrame, dict, float]:
    bv = pv.BoxValues(state.pg, state.sched)
    anchors = next_window(state.pg)
    anchors = anchors[anchors["season"].between(*SEASONS) & anchors["group"].isin(GROUPS)].reset_index(drop=True)
    C = len(cr.ALL_COMPONENTS)
    rows = []
    for hl in HALF_LIVES:
        num = np.zeros((len(anchors), C))
        den = np.zeros((len(anchors), C))
        for (S, W), idx in anchors.groupby(["season", "week"]).indices.items():
            bs = bv.sums(int(S), int(W), hl, 1e-9)
            c = bs.players.get_indexer(anchors.loc[idx, "gsis_id"])
            ok = c >= 0
            num[idx[ok]] = bs.num[c[ok]]
            den[idx[ok]] = bs.den[c[ok]]
        for g in GROUPS:
            m = (anchors["group"] == g).to_numpy()
            a = anchors[m]
            comps = cr.COMPONENTS[g]
            js = [cr.ALL_COMPONENTS.index(c) for c in comps]
            A = a[[f"a_{c}" for c in comps]].to_numpy(float)
            T = (a[[f"t_{c}" for c in comps]].to_numpy(float) * A).sum(axis=1)
            Rm = np.vstack([state.R[int(s)].loc[g, list(comps)].to_numpy(float) for s in a["season"]])
            wt = a["den"].to_numpy(float)
            base = (Rm * A).sum(axis=1)
            mse0 = float(np.sum(wt * (T - base) ** 2) / wt.sum())
            for k in KS:
                rate = (num[m][:, js] + k * Rm) / (den[m][:, js] + k)
                pred = (rate * A).sum(axis=1)
                mse = float(np.sum(wt * (T - pred) ** 2) / wt.sum())
                rows.append({"half_life": hl, "group": g, "k": k, "n": int(m.sum()), "wmse": mse, "wmse_repl": mse0,
                             "skill": 1 - mse / mse0})
    grid = pd.DataFrame(rows)
    best_k = grid.loc[grid.groupby(["half_life", "group"])["wmse"].idxmin()]
    by_hl = best_k.groupby("half_life").apply(lambda d: float((d["wmse"] / d["wmse_repl"]).sum()), include_groups=False)
    hl = float(by_hl.idxmin())
    chosen = best_k[best_k["half_life"] == hl].set_index("group")["k"].astype(float).to_dict()
    return grid, chosen, hl


def tune_ridge(state, inp, cfg: pv.ValueConfig) -> pd.DataFrame:
    sched = state.sched
    tab = oa.rating_table(inp.pbp, sched, oa.TUNED)
    reg = sched[(sched["game_type"] == "REG") & sched["season"].between(RIDGE_SEASONS[0] - cfg.ridge_seasons + 1,
                                                                         RIDGE_SEASONS[1])]
    keys = sorted(set(zip(reg["season"].astype(int), reg["week"].astype(int))))
    data_key = registry.games_hash([f"{k}:{v}" for k, v in m5in.fingerprint(SEASONS[1]).items()])
    ratings = oa.cached_ratings(tab, sched, keys, oa.TUNED, ("all",), data_key)
    rows = pv.residual_rows(tab[tab["game_id"].isin(reg["game_id"])], ratings)
    rd = pv.RidgeData(rows, state.onfield, sched)
    akeys = [k for k in keys if RIDGE_SEASONS[0] <= k[0] <= RIDGE_SEASONS[1]]
    box = pv.box_values(pv.BoxValues(state.pg, sched), akeys, cfg, state.R)
    # targets: for each anchor (team, S, W), that team's next 4 rows (same season, kickoff at or after the week's as_of)
    asofs = oa.week_asof(sched)
    of = state.onfield[(state.onfield["group"] != "K") & (state.onfield["share"] > 0)]
    out = []
    for lam in LAMS:
        c2 = pv.ValueConfig(cfg.half_life, cfg.k, cfg.early_games, lam, cfg.ridge_seasons, cfg.min_weight)
        sse = {"off": 0.0, "def": 0.0}
        sse0 = {"off": 0.0, "def": 0.0}
        wsum = {"off": 0.0, "def": 0.0}
        for S, W in akeys:
            pr = box[(box["season"] == S) & (box["week"] == W)]
            prv = pr.set_index("gsis_id")["v_box"]
            for side, sign in (("off", 1.0), ("def", -1.0)):
                f = rd.fit(S, W, pr, c2, side)
                e = pd.Series(f["e"].to_numpy(float), index=f["gsis_id"].to_numpy()) if len(f) else pd.Series(dtype=float)
                icpt = f.attrs.get("intercept", 0.0)
                rr = rd.sides[side]["rows"]
                fut = rr[(rr["season"] == S) & (rr["kick_ns"] >= asofs[(S, W)])].sort_values("kick_ns")
                fut = fut.groupby("team", sort=False).head(NEXT)
                lin = of[of["game_id"].isin(fut["game_id"]) & (of["side"] == side)]
                eff = lin["gsis_id"].map(e)
                eff = eff.fillna(lin["gsis_id"].map(prv).fillna(0.0) * sign)
                pred = (lin["share"] * eff).groupby([lin["game_id"], lin["team"]]).sum()
                p = pred.reindex(pd.MultiIndex.from_frame(fut[["game_id", "team"]])).fillna(0.0).to_numpy() + icpt
                y, wpl = fut["y"].to_numpy(float), fut["plays"].to_numpy(float)
                sse[side] += float(np.sum(wpl * (y - p) ** 2))
                sse0[side] += float(np.sum(wpl * (y - icpt) ** 2))
                wsum[side] += float(wpl.sum())
        out.append({"lam": lam, **{f"mse_{s}": sse[s] / wsum[s] for s in sse},
                    **{f"mse_icpt_{s}": sse0[s] / wsum[s] for s in sse}})
        out[-1]["mse"] = (sse["off"] + sse["def"]) / (wsum["off"] + wsum["def"])
        out[-1]["mse_icpt"] = (sse0["off"] + sse0["def"]) / (wsum["off"] + wsum["def"])
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    inp = m5in.load(SEASONS[1])
    state = rf.prepare(inp.pbp, inp.sched, inp.depth, inp.injuries, inp.rosters, inp.positions,
                       lu.default_probs(), inp.availability, pv.ValueConfig(), roster_seasons=inp.roster_seasons)
    t_prep = time.perf_counter() - t0
    print(f"Inputs 1999-{SEASONS[1]} loaded and credited in {t_prep:.1f}s: {len(state.pg):,} player-games.")
    print("Era gaps (component recorded = 1):")
    print(inp.availability.loc[1999:SEASONS[1]].T.to_string())

    t1 = time.perf_counter()
    grid, chosen_k, hl = tune_box(state, inp.availability)
    t_box = time.perf_counter() - t1
    best = grid[grid["half_life"] == hl].loc[lambda d: d.groupby("group")["wmse"].idxmin()]
    print(f"\nBox values, {SEASONS[0]}-{SEASONS[1]} anchors (next {NEXT} games, same season). Chosen half-life "
          f"{hl:g} weeks (sum of MSE / MSE_replacement over groups, each at its best k):")
    hl_tab = grid.loc[grid.groupby(["half_life", "group"])["wmse"].idxmin()].groupby("half_life").apply(
        lambda d: float((d["wmse"] / d["wmse_repl"]).sum()), include_groups=False)
    print("  " + ", ".join(f"{int(h)}: {v:.4f}" for h, v in hl_tab.items()))
    print(f"{'group':<6}{'n':>8}{'k':>7}{'skill':>8}{'wmse':>11}{'wmse_repl':>11}  k grid edge?")
    for _, r in best.iterrows():
        edge = "yes" if r["k"] in (KS[0], KS[-1]) else ""
        print(f"{r['group']:<6}{r['n']:>8}{r['k']:>7g}{r['skill']:>8.3f}{r['wmse']:>11.6f}{r['wmse_repl']:>11.6f}  {edge}")

    cfg = pv.ValueConfig(half_life=hl, k=tuple(sorted((chosen_k | {"OL": pv.DEFAULT_K["OL"]}).items())))
    t2 = time.perf_counter()
    rg = tune_ridge(state, inp, cfg)
    t_ridge = time.perf_counter() - t2
    lam = float(rg.loc[rg["mse"].idxmin(), "lam"])
    print(f"\nWith/without ridge, {RIDGE_SEASONS[0]}-{RIDGE_SEASONS[1]} anchors (next {NEXT} team-games, residual EPA/play, "
          "plays-weighted MSE; lambda 1e7 = box values only):")
    print(rg.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Chosen lambda: {lam:g}" + (" (box-only limit: the with/without layer does not help on 2006-2008)"
                                      if lam >= 1e7 else ""))
    timing = {"prepare_s": t_prep, "box_s": t_box, "ridge_s": t_ridge, "total_s": time.perf_counter() - t0}
    print(f"\nTiming: prepare {t_prep:.1f}s, box grid {t_box:.1f}s, ridge grid {t_ridge:.1f}s, "
          f"total {timing['total_s']:.1f}s")

    if not a.no_log:
        anchors_ids = [f"{r.gsis_id}:{r.season}:{r.week}" for r in next_window(state.pg).itertuples()
                       if SEASONS[0] <= r.season <= SEASONS[1]]
        path = registry.log_run(
            "m5_tune_players", label="tune", seasons=SEASONS, game_ids=anchors_ids,
            metrics={"n": len(anchors_ids), "half_life": hl, "lam": lam,
                     **{f"skill_{g}": float(best.set_index("group").at[g, "skill"]) for g in GROUPS},
                     "ridge_mse": float(rg["mse"].min()), "ridge_mse_box_only": float(rg.loc[rg["lam"] >= 1e7, "mse"].iloc[0]),
                     "ridge_mse_intercept_only": float(rg["mse_icpt"].iloc[0])},
            features=["player_value_box", "player_value_ww"],
            params={"chosen": {"half_life": hl, "k": chosen_k, "lam": lam},
                    "grid": {"half_life": HALF_LIVES, "k": KS, "lam": LAMS}, "next_games": NEXT,
                    "box_seasons": list(SEASONS), "ridge_seasons": list(RIDGE_SEASONS)},
            data=m5in.fingerprint(SEASONS[1]),
            notes="M5 tuning (spec section 3): box-value half-life and k per group on next-4-games credited EPA "
                  "(2001-2008); with/without lambda on next-4 team-game EPA residuals (2006-2008). No game outcomes.",
            extra={"box_grid": grid.to_dict("records"), "ridge_grid": rg.to_dict("records"), "timing": timing})
        print(f"Logged {path.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
