"""M7b early-down play type x personnel: dev evaluation, leakage check, sign-off.

LICENSE: CC BY-SA 4.0. The actions are offensive personnel groupings from
nflverse participation data (NFL Next Gen Stats via nflverse 2016-2022, FTN
Data via nflverse 2023+), read through the M6 play table, so everything this
script writes (frames, cell tables, recommendations, run records' numbers)
is CC BY-SA 4.0. Nothing here feeds A4s or any CC BY artifact. Spec:
context/ml-m7-method.md, section 4 (Part B) and section 5.

    python scripts/ml_m7b.py dev              # season-ahead 2018 and 2019; logs m7b_* runs (label m7bdev)
    python scripts/ml_m7b.py leakcheck        # real 2019: corrupt the season from weeks 1, 9, 17; positive controls
    python scripts/ml_m7b.py signoff-dry-run  # the sign-off code path on 2018-2019, no logging; must reproduce `dev`
    python scripts/ml_m7b.py signoff          # the ONE pre-registered holdout run, 2020-2025
    add --no-log to skip the registry (dev)

`signoff` refuses twice over: if any m7b_signoff_* run flagged holdout exists
(it is run once only), and unless the dry run reproduced dev.json on this exact
commit with a clean tree (`data/raw/ml/m7b/signoff_m7bdev.json`: reproduced =
true, its git sha = HEAD, not dirty, and the tree is still clean).

Season-ahead protocol. For a target season S: the cells and recommendations
come from 2016..S-1 with nuisances cross-fit by season (each training season's
propensity and outcome models are fit on the other training seasons); the
OPE of S uses nuisances fit on all of 2016..S-1 and S's outcomes. Team
tendencies and M3 ratings are as of each play's week start. CIs: game-cluster
bootstrap, 2,000 reps, seed 20261003. Primary (dev preview of the
pre-registered rule): recommended-minus-observed OPE (EPA per early-down
play) with CI lower bound > 0 AND the placebo (actions shuffled within
season x cell, the whole pipeline rerun) gain CI includes 0.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ml_m6  # noqa: E402
from nflelo import config  # noqa: E402
from nflelo.ml import data as mldata  # noqa: E402
from nflelo.ml import registry  # noqa: E402
from nflelo.ml.decisions import policy as pol  # noqa: E402
from nflelo.ml.eval import bootstrap, windows  # noqa: E402
from nflelo.ml.plays import metrics as pm  # noqa: E402

DEV = windows.M7B_DEV             # (2018, 2019)
HOLDOUT = windows.HOLDOUT         # (2020, 2025)
OUT = mldata.ML_DIR / "m7b"
SEED = pol.SEED
REPS = pol.REPS
CAL_REPS = 500
PLACEBO_SEED = SEED + 7
TEAM_VIEW = ("MIN", 2019)         # the notebook's team example (a run-first 2019 offense)
STOP_SHARE_2PLUS = 0.20
LIC = {"license": "CC BY-SA 4.0", "note": pol.LICENSE_NOTE}


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- inputs

def load_frame(last: int) -> tuple[pd.DataFrame, float]:
    """The M7b frame for 2016..last: the M6 play table (+ the play's EPA from the M5b kept plays), prepared."""
    t0 = time.perf_counter()
    w = ml_m6.load_world(last, priors={})
    t = w.table.merge(w.ctx.plays[["game_id", "play_id", "epa"]], on=["game_id", "play_id"], how="left",
                      validate="one_to_one")
    if t["epa"].isna().any():
        raise ValueError("kept plays without an EPA")
    f = pol.prepare(t)
    sec = time.perf_counter() - t0
    log(f"frame through {last}: {len(f):,} early-down plays in {f['game_id'].nunique():,} games ({sec:.0f}s)")
    return f, sec


# --------------------------------------------------------------------------- scoring

def ci(d, games) -> dict:
    return bootstrap.cluster_bootstrap_diff(np.asarray(d, float), games, reps=REPS, seed=SEED)


def pooled(work: dict, seasons, variant: str = "main") -> dict:
    """Concatenate the per-season test frames and arrays."""
    S = [s for s in seasons if s in work]
    test = pd.concat([work[s]["test"] for s in S], ignore_index=True)
    cat = lambda k: np.concatenate([work[s]["pred"][k] for s in S])  # noqa: E731
    v = lambda k: np.concatenate([work[s]["variants"][variant][k] for s in S])  # noqa: E731
    return {"test": test, "e": cat("e"), "mu_epa": cat("mu_epa"), "mu_succ": cat("mu_succ"),
            "rec": v("rec"), "d_epa": v("d_epa"), "d_succ": v("d_succ"), "obs_epa": v("obs_epa"),
            "obs_succ": v("obs_succ"), "g_epa": v("g_epa"), "g_succ": v("g_succ")}


def propensity_scores(work: dict, seasons) -> dict:
    P = pooled(work, seasons)
    a = P["test"]["action"].to_numpy(int)
    e = np.clip(P["e"], 1e-9, 1)
    # baseline: each training window's action shares by cell
    base = []
    for s in seasons:
        tk = work[s]["train_keys"]
        sh = pd.crosstab(tk["cell"], tk["action"]).reindex(columns=range(pol.N_ACTIONS), fill_value=0)
        sh = (sh + 0.5).div((sh + 0.5).sum(axis=1), axis=0)
        base.append(sh.reindex(work[s]["test"]["cell"]).to_numpy(float))
    base = np.concatenate(base)
    rows = np.arange(len(a))
    ll, ll_b = -np.log(e[rows, a]), -np.log(base[rows, a])
    games = P["test"]["game_id"]
    cal = {}
    for i, name in enumerate(pol.ACTIONS):
        c = pm.calibration_check((a == i).astype(float), P["e"][:, i], reps=CAL_REPS, seed=SEED)
        cal[name] = {**c, "share": float((a == i).mean())}
    return {"n": int(len(a)), "logloss": float(ll.mean()), "logloss_cell_baseline": float(ll_b.mean()),
            "logloss_vs_baseline": ci(ll - ll_b, games), "calibration": cal,
            "all_calibrated": bool(all(c["ok"] for c in cal.values())),
            "pass_share_pred": float(P["e"][:, 1::2].sum(axis=1).mean()),
            "pass_share_obs": float((a % 2 == 1).mean())}


def outcome_scores(work: dict, seasons) -> dict:
    P = pooled(work, seasons)
    t = P["test"]
    a = t["action"].to_numpy(int)
    rows = np.arange(len(a))
    y, ys = t["y_epa"].to_numpy(float), t["y_succ"].to_numpy(float)
    # baseline: cell x action mean EPA in the training window
    base_e, base_s = [], []
    for s in seasons:
        k = work[s]["train_keys"]
        tr = work[s]["test"]
        oof_y = work[s]["train_y"]
        m = pd.DataFrame({"cell": k["cell"], "action": k["action"], "y": oof_y[0], "ys": oof_y[1]})
        g = m.groupby(["cell", "action"])[["y", "ys"]].mean()
        idx = pd.MultiIndex.from_arrays([tr["cell"], tr["action"]])
        base_e.append(g["y"].reindex(idx).fillna(m["y"].mean()).to_numpy(float))
        base_s.append(g["ys"].reindex(idx).fillna(m["ys"].mean()).to_numpy(float))
    be, bs = np.concatenate(base_e), np.concatenate(base_s)
    mu, ms = P["mu_epa"][rows, a], P["mu_succ"][rows, a]
    games = t["game_id"]
    return {"n": int(len(a)), "mse": float(np.mean((y - mu) ** 2)), "mse_cell_action": float(np.mean((y - be) ** 2)),
            "mse_vs_cell_action": ci((y - mu) ** 2 - (y - be) ** 2, games),
            "mse_constant": float(np.mean((y - y.mean()) ** 2)),
            "brier_succ": float(np.mean((ys - ms) ** 2)), "brier_succ_cell_action": float(np.mean((ys - bs) ** 2)),
            "succ_cal": pm.calibration_check(ys, ms, reps=CAL_REPS, seed=SEED),
            "mean_mu_by_action": {n: float(P["mu_epa"][:, i].mean()) for i, n in enumerate(pol.ACTIONS)}}


def ope_block(P: dict) -> dict:
    t = P["test"]
    g = t["game_id"]
    on = P["rec"] >= 0
    y, ys = t["y_epa"].to_numpy(float), t["y_succ"].to_numpy(float)
    return {"n": int(len(t)), "games": int(g.nunique()), "share_policy_applies": float(on.mean()),
            "share_policy_changes_call": float((on & (P["rec"] != t["action"].to_numpy(int))).mean()),
            "epa": ci(P["d_epa"], g), "succ": ci(P["d_succ"], g),
            "epa_per_affected_play": float(P["d_epa"][on].mean()) if on.any() else 0.0,
            "observed_mean_epa": float(y.mean()), "observed_mean_succ": float(ys.mean()),
            "observed_dr_epa": float(P["obs_epa"].mean()), "observed_check_epa": ci(P["obs_epa"] - y, g),
            "observed_check_succ": ci(P["obs_succ"] - ys, g)}


def recs_summary(work: dict, seasons, variant: str = "main", top: int = 12) -> dict:
    out = {}
    for s in seasons:
        r = work[s]["variants"][variant]["recs"]
        c = work[s]["variants"][variant]["cells"]
        clear = r[r["clear"]].sort_values("gain_lo", ascending=False)
        out[int(s)] = {"cells": int(len(r)), "clear": int(r["clear"].sum()), "clear_share": float(r["clear"].mean()),
                       "plays_in_clear_cells": float(r.loc[r["clear"], "n_cell"].sum() / r["n_cell"].sum()),
                       "with_candidates": int((r["candidates"] > 0).sum()),
                       "best_counts": clear["best"].value_counts().to_dict(),
                       "candidates_per_cell": float(r["candidates"].mean()),
                       "clear_cells": clear.head(top).replace({np.nan: None}).to_dict("records"),
                       "share_cell_actions_candidate": float(c["candidate"].mean())}
    return out


def team_view(work: dict, team: str, S: int) -> dict | None:
    if S not in work:
        return None
    w = work[S]
    t = w["test"]
    v = w["variants"]["main"]
    m = (t["off_team"] == team).to_numpy()
    if not m.any():
        return None
    tt = t[m]
    a = tt["action"].to_numpy(int)
    y = tt["y_epa"].to_numpy(float)
    e, g = w["pred"]["e"][m], v["g_epa"][m]
    games = tt["game_id"]
    by_action = []
    for i, name in enumerate(pol.ACTIONS):
        el = e[:, i] >= pol.OVERLAP
        if el.sum() < 30:
            continue
        c = ci((g[el, i] - y[el]), games[el])
        by_action.append({"action": name, "team_share": float((a == i).mean()),
                          "league_share": float((t["action"].to_numpy(int) == i).mean()),
                          "n_elig": int(el.sum()), "gain": c["diff"], "lo": c["ci_low"], "hi": c["ci_high"]})
    recs = v["recs"].set_index("cell")
    cells = []
    for c_, sub in tt.groupby("cell"):
        ii = np.flatnonzero((tt["cell"] == c_).to_numpy())
        aa = a[ii]
        top = pd.Series(aa).value_counts(normalize=True)
        r = recs.loc[c_] if c_ in recs.index else None
        best = r["best"] if r is not None and isinstance(r["best"], str) else None
        cells.append({"cell": c_, "n": int(len(ii)), "pass_share": float((aa % 2 == 1).mean()),
                      "top_action": pol.ACTIONS[int(top.index[0])], "top_share": float(top.iloc[0]),
                      "obs_epa": float(y[ii].mean()), "league_best": best,
                      "league_clear": bool(r["clear"]) if r is not None else False,
                      "league_gain": None if r is None or pd.isna(r["gain"]) else float(r["gain"]),
                      "league_gain_lo": None if r is None or pd.isna(r["gain_lo"]) else float(r["gain_lo"]),
                      "team_share_best": float((aa == pol.ACTIONS.index(best)).mean()) if best else None})
    cells = sorted(cells, key=lambda d: -d["n"])
    return {"team": team, "season": int(S), "plays": int(m.sum()), "games": int(games.nunique()),
            "pass_share": float((a % 2 == 1).mean()), "league_pass_share": float((t["action"] % 2 == 1).mean()),
            "obs_epa": float(y.mean()), "ope": ci(v["d_epa"][m], games),
            "share_policy_applies": float((v["rec"][m] >= 0).mean()),
            "by_action": by_action, "cells": cells}


def score_window(work: dict, work_p: dict, seasons) -> dict:
    seasons = [s for s in seasons if s in work]
    P = pooled(work, seasons)
    a = P["test"]["action"].to_numpy(int)
    res = {"seasons": [min(seasons), max(seasons)], "n": int(len(a)), "games": int(P["test"]["game_id"].nunique()),
           "plays_hash": registry.games_hash(P["test"]["game_id"] + "_" + P["test"]["play_id"].astype(str)),
           "action_shares": {n: float((a == i).mean()) for i, n in enumerate(pol.ACTIONS)},
           "overlap": {"main": pol.overlap_stats(P["e"], a, pol.OVERLAP),
                       "thr10": pol.overlap_stats(P["e"], a, pol.OVERLAP_ALT)},
           "propensity": propensity_scores(work, seasons), "outcome": outcome_scores(work, seasons),
           "ope": {v: ope_block(pooled(work, seasons, v)) for v in pol.VARIANTS},
           "placebo": ope_block(pooled(work_p, seasons, "main")),
           "recs": {v: recs_summary(work, seasons, v) for v in pol.VARIANTS},
           "placebo_recs": recs_summary(work_p, seasons, "main", top=5)}
    ngs, ftn = [s for s in seasons if s < 2023], [s for s in seasons if s >= 2023]
    if ngs and ftn:            # participation source: NGS through 2022, FTN from 2023 (M6 harmonizes personnel)
        res["eras"] = {"ngs": ope_block(pooled(work, ngs)), "ftn": ope_block(pooled(work, ftn))}
    m, p = res["ope"]["main"]["epa"], res["placebo"]["epa"]
    res["rules"] = {"ope_ci_low": m["ci_low"], "ope_pass": bool(m["ci_low"] > 0),
                    "placebo_ci": [p["ci_low"], p["ci_high"]],
                    "placebo_pass": bool(p["ci_low"] <= 0 <= p["ci_high"]),
                    "pass": bool(m["ci_low"] > 0 and p["ci_low"] <= 0 <= p["ci_high"])}
    res["stops"] = {"overlap_too_poor": bool(res["overlap"]["main"]["share_2plus"] < STOP_SHARE_2PLUS),
                    "placebo_fake_gain": bool(not res["rules"]["placebo_pass"])}
    return res


def per_season(work: dict, work_p: dict) -> dict:
    out = {}
    for s in sorted(work):
        P, Q = pooled(work, [s]), pooled(work_p, [s])
        out[int(s)] = {"ope": ope_block(P)["epa"], "ope_succ": ope_block(P)["succ"], "placebo": ope_block(Q)["epa"],
                       "observed_check": ope_block(P)["observed_check_epa"],
                       "clear_share": float(work[s]["variants"]["main"]["recs"]["clear"].mean()),
                       "overlap_2plus": pol.overlap_stats(P["e"], P["test"]["action"].to_numpy(int),
                                                          pol.OVERLAP)["share_2plus"],
                       "nuisance": work[s]["nuisance"], "folds": work[s]["folds"], "seconds": work[s]["seconds"],
                       "placebo_seconds": work_p[s]["seconds"]}
    return out


# --------------------------------------------------------------------------- evaluation

def run_seasons(frame: pd.DataFrame, seasons, variants=tuple(pol.VARIANTS)) -> dict:
    work = {}
    for S in seasons:
        t0 = time.perf_counter()
        w = pol.season_work(frame, S, variants)
        s = frame["season"].to_numpy(int)
        tr = frame[(s >= pol.FIRST) & (s < S)]
        w["train_y"] = (tr["y_epa"].to_numpy(float), tr["y_succ"].to_numpy(float))
        work[S] = w
        log(f"  season {S}: {len(w['test']):,} plays, train {len(tr):,} ({time.perf_counter() - t0:.0f}s: "
            + ", ".join(f"{k} {v:.0f}s" for k, v in w["seconds"].items()) + ")")
    return work


def evaluate(window: tuple[int, int], allow_holdout: bool) -> tuple[dict, dict, dict]:
    windows.check(window, allow_holdout)
    t0 = time.perf_counter()
    frame, sec_in = load_frame(window[1])
    seasons = list(range(window[0], window[1] + 1))
    log("main pipeline:")
    work = run_seasons(frame, seasons)
    log("placebo (actions shuffled within season x cell, whole pipeline rerun):")
    work_p = run_seasons(pol.shuffle_within_cells(frame, PLACEBO_SEED), seasons, ("main",))
    res = {"window": list(window), "primary": score_window(work, work_p, seasons),
           "per_season": per_season(work, work_p), "team_view": team_view(work, TEAM_VIEW[0], TEAM_VIEW[1]),
           "config": {"actions": list(pol.ACTIONS), "cells": list(pol.CELLS), "features": pol.FEATURES,
                      "overlap": pol.OVERLAP, "overlap_alt": pol.OVERLAP_ALT, "trim_floor": pol.TRIM_FLOOR,
                      "cand_share": pol.CAND_SHARE, "min_taken": pol.MIN_TAKEN, "tend_m": pol.TEND_M,
                      "prior_league": pol.PRIOR_LEAGUE,
                      "prop_params": pol.PROP_PARAMS, "out_params": pol.OUT_PARAMS, "max_iter": pol.MAX_ITER,
                      "reps": REPS, "seed": SEED, "placebo_seed": PLACEBO_SEED},
           "inputs_seconds": sec_in, "total_seconds": time.perf_counter() - t0, **LIC}
    return res, work, work_p


def fmt_ci(c: dict | None, nd: int = 4) -> str:
    if not c:
        return "n/a"
    return f"{c['diff']:+.{nd}f} [{c['ci_low']:+.{nd}f}, {c['ci_high']:+.{nd}f}]"


def print_window(r: dict) -> None:
    log(f"\n=== M7b {r['seasons'][0]}-{r['seasons'][1]}: {r['n']:,} early-down plays, {r['games']} games ===")
    log("action shares: " + ", ".join(f"{k} {v:.3f}" for k, v in r["action_shares"].items()))
    for k, o in r["overlap"].items():
        log(f"overlap ({k}, thr {o['thr']}): >=2 eligible {o['share_2plus']:.3f}; mean eligible {o['mean_eligible']:.2f}"
            f" of 12; pairs eligible {o['share_pairs_eligible']:.3f}; observed action eligible "
            f"{o['share_observed_eligible']:.3f}")
        log("   by action: " + ", ".join(f"{a} {s:.2f}" for a, s in o["share_eligible_by_action"].items()))
    p = r["propensity"]
    log(f"propensity: log loss {p['logloss']:.4f} vs cell shares {p['logloss_cell_baseline']:.4f} "
        f"({fmt_ci(p['logloss_vs_baseline'])}); pass share pred {p['pass_share_pred']:.3f} obs {p['pass_share_obs']:.3f}")
    for a, c in p["calibration"].items():
        log(f"   {a:8s} share {c['share']:.3f} pred {c['mean_pred']:.3f} ECE {c['ece']:.4f} (p95 {c['null_p95']:.4f}) "
            f"{'ok' if c['ok'] else 'FAIL'}")
    o = r["outcome"]
    log(f"outcome: EPA MSE {o['mse']:.4f} vs cell x action mean {o['mse_cell_action']:.4f} "
        f"({fmt_ci(o['mse_vs_cell_action'])}), constant {o['mse_constant']:.4f}; success Brier {o['brier_succ']:.4f} "
        f"vs {o['brier_succ_cell_action']:.4f}; success ECE {o['succ_cal']['ece']:.4f} "
        f"{'ok' if o['succ_cal']['ok'] else 'FAIL'}")
    for v, b in r["ope"].items():
        log(f"OPE [{v}]: EPA/play {fmt_ci(b['epa'])}; success {fmt_ci(b['succ'])}; policy applies to "
            f"{b['share_policy_applies']:.3f} of plays (changes the call on {b['share_policy_changes_call']:.3f}); "
            f"{b['epa_per_affected_play']:+.4f} per affected play")
    b = r["ope"]["main"]
    log(f"observed-policy check: DR {b['observed_dr_epa']:+.4f} vs mean EPA {b['observed_mean_epa']:+.4f}: "
        f"{fmt_ci(b['observed_check_epa'])}; success {fmt_ci(b['observed_check_succ'])}")
    q = r["placebo"]
    log(f"PLACEBO: EPA/play {fmt_ci(q['epa'])}; applies to {q['share_policy_applies']:.3f} of plays")
    for s, x in r["recs"]["main"].items():
        log(f"recs for {s}: {x['clear']} of {x['cells']} cells clear ({x['clear_share']:.3f}; "
            f"{x['plays_in_clear_cells']:.3f} of plays); best {x['best_counts']}")
        for c in x["clear_cells"][:6]:
            log(f"   {c['cell']:35s} n {c['n_cell']:5d} best {c['best']:8s} gain {c['gain']:+.3f} "
                f"[{c['gain_lo']:+.3f}, {c['gain_hi']:+.3f}] (DR {c['dr_mean']:+.3f} vs obs {c['obs_mean']:+.3f})")
    for v in ("thr10", "trim10"):
        log(f"recs [{v}]: " + "; ".join(f"{s}: {x['clear']}/{x['cells']}" for s, x in r["recs"][v].items()))
    log("placebo recs: " + "; ".join(f"{s}: {x['clear']}/{x['cells']} clear" for s, x in r["placebo_recs"].items()))
    ru = r["rules"]
    log(f"RULES: OPE CI low {ru['ope_ci_low']:+.4f} ({'pass' if ru['ope_pass'] else 'fail'}); placebo CI "
        f"[{ru['placebo_ci'][0]:+.4f}, {ru['placebo_ci'][1]:+.4f}] ({'pass' if ru['placebo_pass'] else 'fail'}) "
        f"-> {'PASS' if ru['pass'] else 'FAIL'}")
    if any(r["stops"].values()):
        log(f"STOP CONDITION: {r['stops']}")


def print_team(tv: dict | None) -> None:
    if not tv:
        return
    log(f"\nTeam view {tv['team']} {tv['season']}: {tv['plays']} plays, pass share {tv['pass_share']:.3f} "
        f"(league {tv['league_pass_share']:.3f}), EPA {tv['obs_epa']:+.3f}; league policy on its plays "
        f"{fmt_ci(tv['ope'])} (applies {tv['share_policy_applies']:.2f})")
    for b in sorted(tv["by_action"], key=lambda d: -d["gain"]):
        log(f"   {b['action']:8s} team {b['team_share']:.3f} league {b['league_share']:.3f} gain {b['gain']:+.3f} "
            f"[{b['lo']:+.3f}, {b['hi']:+.3f}] (n elig {b['n_elig']})")


def write_license() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "LICENSE.txt").write_text("Every file in this folder is CC BY-SA 4.0: " + pol.LICENSE_NOTE + "\n")


def save_frames(work: dict, work_p: dict, label: str) -> None:
    write_license()
    parts, cells, recs = [], [], []
    for S, w in work.items():
        t = w["test"][["game_id", "play_id", "season", "week", "season_type", "off_team", "def_team", "down",
                       "ydstogo", "yardline_100", "score_diff", "cell", "action", "y_epa", "y_succ", "yards"]].copy()
        v = w["variants"]["main"]
        for i, a in enumerate(pol.ACTIONS):
            t[f"e_{a}"] = w["pred"]["e"][:, i]
            t[f"mu_{a}"] = w["pred"]["mu_epa"][:, i]
            t[f"g_{a}"] = v["g_epa"][:, i]
        t["rec"] = v["rec"]
        t["d_epa"] = v["d_epa"]
        t["d_succ"] = v["d_succ"]
        t["obs_dr"] = v["obs_epa"]
        t["d_placebo"] = work_p[S]["variants"]["main"]["d_epa"]
        parts.append(t)
        for vn, vv in w["variants"].items():
            cells.append(vv["cells"].assign(season=S, variant=vn))
            recs.append(vv["recs"].assign(season=S, variant=vn))
        cells.append(work_p[S]["variants"]["main"]["cells"].assign(season=S, variant="placebo"))
        recs.append(work_p[S]["variants"]["main"]["recs"].assign(season=S, variant="placebo"))
    pd.concat(parts, ignore_index=True).to_parquet(OUT / f"plays_{label}.parquet", index=False)
    pd.concat(cells, ignore_index=True).to_parquet(OUT / f"cells_{label}.parquet", index=False)
    pd.concat(recs, ignore_index=True).to_parquet(OUT / f"recs_{label}.parquet", index=False)


def log_runs(res: dict, work: dict, label: str, prefix: str, signoff: str | None = None) -> None:
    r = res["primary"]
    seasons = tuple(r["seasons"])
    fdat = registry.data_fingerprint(games_csv=False, ml_datasets=("pbp", "schedules", "participation"),
                                     seasons=range(pol.FIRST, seasons[1] + 1))
    gid = pd.concat([work[s]["test"]["game_id"] for s in work]).unique().tolist()
    ex = {**LIC, **({"signoff": signoff} if signoff else {}), "plays_hash": r["plays_hash"]}
    params = res["config"]
    o = r["ope"]["main"]
    registry.log_run(f"{prefix}_ope", label=label, seasons=seasons, game_ids=gid,
                     metrics={"n": o["n"], "ope_epa": o["epa"], "ope_succ": o["succ"],
                              "share_policy_applies": o["share_policy_applies"],
                              "epa_per_affected_play": o["epa_per_affected_play"],
                              "observed_check_epa": o["observed_check_epa"],
                              "observed_check_succ": o["observed_check_succ"],
                              "placebo_epa": r["placebo"]["epa"], "rules": r["rules"]},
                     features=pol.FEATURES, params=params, data=fdat,
                     notes="recommended minus observed policy, DR, EPA per early-down play", extra=ex)
    for v in ("thr10", "trim10"):
        b = r["ope"][v]
        registry.log_run(f"{prefix}_ope_{v}", label=label, seasons=seasons, game_ids=gid,
                         metrics={"n": b["n"], "ope_epa": b["epa"], "ope_succ": b["succ"],
                                  "share_policy_applies": b["share_policy_applies"]},
                         features=pol.FEATURES, params={**params, "variant": pol.VARIANTS[v]}, data=fdat,
                         notes=f"sensitivity: {v}", extra=ex)
    registry.log_run(f"{prefix}_placebo", label=label, seasons=seasons, game_ids=gid,
                     metrics={"n": r["placebo"]["n"], "ope_epa": r["placebo"]["epa"], "ope_succ": r["placebo"]["succ"],
                              "share_policy_applies": r["placebo"]["share_policy_applies"]},
                     features=pol.FEATURES, params={**params, "placebo": "actions shuffled within season x cell"},
                     data=fdat, notes="placebo: the OPE gain should be about 0", extra=ex)
    registry.log_run(f"{prefix}_propensity", label=label, seasons=seasons, game_ids=gid,
                     metrics={"n": r["propensity"]["n"], **{k: v for k, v in r["propensity"].items() if k != "n"},
                              "overlap": r["overlap"]},
                     features=pol.FEATURES, params={"gbm": pol.PROP_PARAMS}, data=fdat, extra=ex)
    registry.log_run(f"{prefix}_outcome", label=label, seasons=seasons, game_ids=gid,
                     metrics={"n": r["outcome"]["n"], **{k: v for k, v in r["outcome"].items() if k != "n"}},
                     features=pol.FEATURES + ["action"], params={"gbm": pol.OUT_PARAMS}, data=fdat, extra=ex)
    registry.log_run(f"{prefix}_policy", label=label, seasons=seasons, game_ids=gid,
                     metrics={"n": r["n"], "recs": r["recs"], "placebo_recs": r["placebo_recs"],
                              "action_shares": r["action_shares"]},
                     features=pol.FEATURES, params=params, data=fdat, notes="cells and recommendations per season",
                     extra={**ex, "per_season": res["per_season"]})
    if res.get("team_view"):
        tv = res["team_view"]
        registry.log_run(f"{prefix}_team_view", label=label, seasons=(tv["season"], tv["season"]),
                         game_ids=work[tv["season"]]["test"].loc[lambda d: d["off_team"] == tv["team"],
                                                                 "game_id"].unique().tolist(),
                         metrics={"n": tv["plays"], **tv}, features=pol.FEATURES, data=fdat,
                         notes=f"team view: {tv['team']} {tv['season']} vs the league recommendations", extra=ex)
    log(f"\nRuns written to {registry.RUNS_DIR.relative_to(config.ROOT)}/ (label {label})")


def stage_dev(log_: bool) -> int:
    res, work, work_p = evaluate(DEV, allow_holdout=False)
    print_window(res["primary"])
    log("\nPer season:")
    for s, v in res["per_season"].items():
        log(f"  {s}: OPE {fmt_ci(v['ope'])}; success {fmt_ci(v['ope_succ'])}; placebo {fmt_ci(v['placebo'])}; "
            f"observed check {fmt_ci(v['observed_check'])}; clear share {v['clear_share']:.3f}")
    print_team(res["team_view"])
    save_frames(work, work_p, "m7bdev")
    (OUT / "dev.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    if log_:
        log_runs(res, work, "m7bdev", "m7b")
    log(f"\ndev total {res['total_seconds']:.0f}s")
    return 0


# --------------------------------------------------------------------------- sign-off

def holdout_signoff_runs() -> list[str]:
    hits = []
    for path in glob.glob(str(registry.RUNS_DIR / "*_m7b_signoff_*.json")):
        if json.loads(Path(path).read_text()).get("holdout"):
            hits.append(Path(path).name)
    return hits


def key_numbers(r: dict) -> dict:
    """The numbers the dry run must reproduce exactly."""
    out = {"ope": r["ope"]["main"]["epa"]["diff"], "ope_lo": r["ope"]["main"]["epa"]["ci_low"],
           "ope_succ": r["ope"]["main"]["succ"]["diff"], "placebo": r["placebo"]["epa"]["diff"],
           "placebo_lo": r["placebo"]["epa"]["ci_low"], "obs_check": r["ope"]["main"]["observed_check_epa"]["diff"],
           "thr10": r["ope"]["thr10"]["epa"]["diff"], "trim10": r["ope"]["trim10"]["epa"]["diff"],
           "overlap": r["overlap"]["main"]["share_2plus"], "prop_ll": r["propensity"]["logloss"],
           "mse": r["outcome"]["mse"]}
    for s, x in r["recs"]["main"].items():
        out[f"clear_{s}"] = x["clear_share"]
    return out


def stage_signoff(window: tuple[int, int], allow_holdout: bool, log_: bool) -> int:
    """The pre-registered sign-off (context/ml.md, "M7b holdout pre-registration"). On the dev window with no
    logging it is the dry run: it must reproduce dev.json's primary block exactly."""
    label = "holdout" if windows.touches_holdout(window) else "m7bdev"
    dev_path = OUT / "dev.json"
    if not dev_path.exists():
        raise SystemExit("run `python scripts/ml_m7b.py dev` first")
    dev = json.loads(dev_path.read_text())
    if label == "holdout":
        prior = holdout_signoff_runs()
        if prior:
            raise SystemExit(f"an M7b holdout sign-off is already recorded ({prior}); it is run once only")
        dry = OUT / "signoff_m7bdev.json"
        rec = json.loads(dry.read_text()) if dry.exists() else {}
        if not rec.get("reproduced"):
            raise SystemExit("the dry run has not reproduced dev: run `python scripts/ml_m7b.py signoff-dry-run` first")
        now = registry.git_commit()
        if rec.get("git", {}).get("dirty") is not False or now.get("dirty") or rec["git"].get("sha") != now.get("sha"):
            raise SystemExit("the dry run must have reproduced dev on this exact commit with a clean tree "
                             f"(dry run {rec.get('git')}, now {now}); commit, then rerun signoff-dry-run")
    res, work, work_p = evaluate(window, allow_holdout)
    print_window(res["primary"])
    out = {"label": label, "window": list(window), "result": res, "rules": res["primary"]["rules"], **LIC}
    if label != "holdout":
        a, b = key_numbers(dev["primary"]), key_numbers(res["primary"])
        diffs = {k: abs(a[k] - b[k]) for k in a}
        out["reproduced"] = bool(max(diffs.values()) <= 1e-12)
        out["max_abs_diff"] = max(diffs.values())
        out["git"] = registry.git_commit()
        log(f"\nDry run vs dev.json: max |diff| {out['max_abs_diff']:.3g} -> "
            f"{'REPRODUCED' if out['reproduced'] else 'NOT REPRODUCED'}")
    save_frames(work, work_p, label)
    (OUT / f"signoff_{label}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    if log_:
        log_runs(res, work, label, "m7b_signoff", signoff="m7b_signoff")
    log(f"\nM7b PRIMARY: {'PASS' if res['primary']['rules']['pass'] else 'FAIL'} (OPE CI low "
        f"{res['primary']['rules']['ope_ci_low']:+.4f}; placebo CI {res['primary']['rules']['placebo_ci']})")
    return 0


# --------------------------------------------------------------------------- leakage

OUTCOME_COLS = ("epa", "yards", "bin", "td", "turnover", "ev_first", "ev_20", "ev_loss")
ACTION_COLS = ("off_rb", "off_te", "off_wr", "off_ol", "is_pass")
POST_CHOICE_COLS = ("def_dl", "def_lb", "def_db", "form_shotgun", "form_pistol", "form_under_center", "box")


def corrupt_table(t: pd.DataFrame, S: int, w: int, seed: int = 0) -> pd.DataFrame:
    """Season S: every outcome scrambled (all weeks); from week w on, the calls (personnel x play type) and
    the downstream structure fields permuted across plays. Situations are untouched."""
    c = t.copy()
    rng = np.random.default_rng(seed)
    ms = (c["season"] == S).to_numpy()
    for col in OUTCOME_COLS:
        if col in c.columns:
            c.loc[ms, col] = rng.permutation(c.loc[ms, col].to_numpy())
    c.loc[ms, "epa"] = c.loc[ms, "epa"].to_numpy(float) + rng.normal(0, 1, int(ms.sum()))
    mw = ms & (c["week"] >= w).to_numpy()
    perm = rng.permutation(int(mw.sum()))
    for cols in (ACTION_COLS, POST_CHOICE_COLS):
        block = c.loc[mw, list(cols)].to_numpy()
        c.loc[mw, list(cols)] = block[perm]
    return c


def stage_leakcheck(S: int = 2019, weeks=(1, 9, 17), seed: int = 0) -> int:
    """Real data: corrupt season S (outcomes, and calls from week w) and confirm everything the week-w
    recommendations and nuisance predictions depend on is unchanged; positive controls must move."""
    t_all = time.perf_counter()
    w6 = ml_m6.load_world(S, priors={})
    table = w6.table.merge(w6.ctx.plays[["game_id", "play_id", "epa"]], on=["game_id", "play_id"], how="left")
    frame = pol.prepare(table)
    clean = pol.season_work(frame, S, ("main",), reps=200)
    out = {"season": S, "weeks": {}, **LIC}
    ok_all = True
    for w in weeks:
        t0 = time.perf_counter()
        fc = pol.prepare(corrupt_table(table, S, w, seed))
        same_rows = fc[["game_id", "play_id"]].equals(frame[["game_id", "play_id"]])
        cw = pol.season_work(fc, S, ("main",), reps=200)
        tm = (clean["test"]["week"] == w).to_numpy()
        fa = clean["test"].loc[tm, pol.FEATURES].to_numpy(float)
        fb = cw["test"].loc[tm, pol.FEATURES].to_numpy(float)
        both_nan = np.isnan(fa) & np.isnan(fb)          # temperature and wind indoors
        feat_diff = float(np.where(both_nan, 0.0, np.nan_to_num(np.abs(fa - fb), nan=np.inf)).max())
        pred_diff = max(float(np.abs(clean["pred"][k][tm] - cw["pred"][k][tm]).max()) for k in ("e", "mu_epa", "mu_succ"))
        rc, rk = clean["variants"]["main"]["recs"], cw["variants"]["main"]["recs"]
        recs_same = bool(rc.fillna(-9).equals(rk.fillna(-9)))
        cells_same = bool(clean["variants"]["main"]["cells"].fillna(-9).equals(cw["variants"]["main"]["cells"].fillna(-9)))
        # positive controls: (1) tendencies of the first later week move; (2) nuisances fit through S move
        later = sorted(clean["test"].loc[clean["test"]["week"] > w, "week"].unique())
        if later:
            lm = (clean["test"]["week"] == later[0]).to_numpy()
            tend_ctrl = float(np.abs(clean["test"].loc[lm, pol.TENDENCY].to_numpy(float)
                                     - cw["test"].loc[lm, pol.TENDENCY].to_numpy(float)).max())
        else:
            tend_ctrl = float("nan")
        through = pol.fit_nuisance(fc[fc["season"].between(pol.FIRST, S)].reset_index(drop=True))
        pc = pol.predict_nuisance(through, cw["test"][tm])
        nu_ctrl = max(float(np.abs(pc[k] - clean["pred"][k][tm]).max()) for k in ("e", "mu_epa"))
        y_chg = float((fc.loc[fc["season"] == S, "y_epa"].to_numpy() != frame.loc[frame["season"] == S,
                                                                               "y_epa"].to_numpy()).mean())
        a_chg = float((fc.loc[(fc["season"] == S) & (fc["week"] >= w), "action"].to_numpy()
                       != frame.loc[(frame["season"] == S) & (frame["week"] >= w), "action"].to_numpy()).mean())
        ok = bool(same_rows and feat_diff == 0 and pred_diff == 0 and recs_same and cells_same and nu_ctrl > 0
                  and (np.isnan(tend_ctrl) or tend_ctrl > 0))
        ok_all &= ok
        out["weeks"][int(w)] = {"plays": int(tm.sum()), "same_rows": bool(same_rows), "feature_max_diff": feat_diff,
                                "prediction_max_diff": pred_diff, "recs_identical": recs_same,
                                "cells_identical": cells_same, "tendency_control_max_diff": tend_ctrl,
                                "control_week": int(later[0]) if later else None,
                                "nuisance_through_S_control_max_diff": nu_ctrl, "outcomes_changed": y_chg,
                                "actions_changed": a_chg, "leak_free": ok, "seconds": time.perf_counter() - t0}
        log(f"  week {w}: {int(tm.sum())} plays; features diff {feat_diff:.3g}; e / mu diff {pred_diff:.3g}; "
            f"recs identical {recs_same}; cells identical {cells_same}; controls: tendencies wk "
            f"{later[0] if later else '-'} {tend_ctrl:.3g}, fit-through-S {nu_ctrl:.3g}; outcomes changed "
            f"{y_chg:.2f}, calls changed {a_chg:.2f} -> {'LEAK-FREE' if ok else 'LEAK'} ({time.perf_counter() - t0:.0f}s)")
    out["leak_free"] = ok_all
    out["seconds"] = time.perf_counter() - t_all
    write_license()
    (OUT / "leakcheck.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    log(f"leakcheck: {'PASS' if ok_all else 'FAIL'} ({out['seconds']:.0f}s)")
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["dev", "leakcheck", "signoff-dry-run", "signoff"])
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    if a.stage == "dev":
        return stage_dev(not a.no_log)
    if a.stage == "leakcheck":
        return stage_leakcheck()
    if a.stage == "signoff-dry-run":
        return stage_signoff(DEV, allow_holdout=False, log_=False)
    if a.stage == "signoff":
        return stage_signoff(HOLDOUT, allow_holdout=True, log_=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
