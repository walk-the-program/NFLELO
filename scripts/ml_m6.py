"""M6 play outcome model: field report, dev evaluation, leakage check, sign-off.

LICENSE: CC BY-SA 4.0. This script reads nflverse participation data (NFL Next
Gen Stats via nflverse 2016-2022, FTN Data via nflverse 2023+), so everything it
writes (tables, saved networks, predictions, run records' derived numbers) is
CC BY-SA 4.0. Nothing here feeds A4s or any CC BY artifact. Spec:
context/ml-m6-method.md.

    python scripts/ml_m6.py fields           # pre-snap field distributions by season, 2016-2025 (no outcome is read)
    python scripts/ml_m6.py dev              # season-ahead dev (2018, 2019), both views; logs m6_* runs
    python scripts/ml_m6.py leakcheck        # real-data leakage checks on 2019 (season corruption + rating as-of)
    python scripts/ml_m6.py signoff-dry-run  # the sign-off code path on dev 2018-2019, no logging; must reproduce `dev`
    python scripts/ml_m6.py signoff          # the ONE pre-registered holdout run, 2020-2025 (refuses if one exists)
    add --no-log to skip the registry (dev)

Season-ahead protocol (spec section 6). For a target season S every model is
fit on the plays of 2016..S-1 and predicts every kept play of S (REG and POST,
M5b cleaning). Tuning uses only the training seasons: settings are fit on
2016..S-2 and compared on S-1 (baseline smoothing and GBM grid by validation
CRPS; network trunk grid and embedding penalty by validation loss, with early
stopping on S-1), then refit on 2016..S-1. Two views: "situation" (no play
type) and "call" (play type as an input). Models, simplest first: baseline,
gbm, N0 (network, no players), N1 (+ learned player embeddings), N2 (N1 + the
players' M5 box and M5b RAPM values as of the start of the season). N1 and N2
reuse the trunk setting N0 chose and tune only the embedding penalty.
Metrics: CRPS (primary), bin log loss, P(first down or TD) / P(20+) / P(loss)
calibration (one-sided null ECE), turnover Brier. CIs: paired bootstrap over
games (2,000 reps, seed 20261003). Selection: the one-SE rule in the order
above (simplest model whose CRPS is within one bootstrap SE of the best).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ml_m5b  # noqa: E402
from nflelo import config  # noqa: E402
from nflelo.ml import asof as asof_mod  # noqa: E402
from nflelo.ml import data as mldata  # noqa: E402
from nflelo.ml import registry  # noqa: E402
from nflelo.ml.eval import bootstrap, windows  # noqa: E402
from nflelo.ml.features import opponent_adjust as oa  # noqa: E402
from nflelo.ml.players import participation as pt  # noqa: E402
from nflelo.ml.players import positions as pos  # noqa: E402
from nflelo.ml.players import rapm  # noqa: E402
from nflelo.ml.plays import baseline as bl  # noqa: E402
from nflelo.ml.plays import data as pdata  # noqa: E402
from nflelo.ml.plays import gbm  # noqa: E402
from nflelo.ml.plays import metrics as pm  # noqa: E402
from nflelo.ml.plays import net  # noqa: E402

FIRST = 2016
DEV = windows.M6_DEV              # (2018, 2019)
HOLDOUT = windows.HOLDOUT         # (2020, 2025)
OUT = mldata.ML_DIR / "m6"
SEED = 20261003
REPS = 2000
CAL_REPS = 500
MODELS = ("baseline", "gbm", "N0", "N1", "N2")   # also the one-SE complexity order, simplest first
NETS = ("N0", "N1", "N2")
LICENSE = {"license": "CC BY-SA 4.0", "note": pdata.LICENSE_NOTE}
EVENTS = ("first", "20plus", "loss")
SPLITS = {
    "run": lambda t: t["is_pass"] == 0, "pass": lambda t: t["is_pass"] == 1,
    "down1": lambda t: t["down"] == 1, "down2": lambda t: t["down"] == 2,
    "down3": lambda t: t["down"] == 3, "down4": lambda t: t["down"] == 4,
    "zone_1_20": lambda t: t["yardline_100"] <= 20, "zone_21_50": lambda t: t["yardline_100"].between(21, 50),
    "zone_51_80": lambda t: t["yardline_100"].between(51, 80), "zone_81_99": lambda t: t["yardline_100"] >= 81,
    "era_ngs": lambda t: t["season"] < 2023, "era_ftn": lambda t: t["season"] >= 2023,
}
PAIRS = (("gbm", "N0"), ("N1", "N0"), ("N2", "N0"), ("N2", "N1"), ("gbm", "N1"), ("gbm", "N2"))


# --------------------------------------------------------------------------- inputs

@dataclass
class World:
    """Every input for seasons FIRST..last, built from data through `last` only."""
    last: int
    ctx: ml_m5b.Ctx
    table: pd.DataFrame
    priors: dict
    ratings: pd.DataFrame


def build_priors(ctx: ml_m5b.Ctx, seasons) -> dict:
    """Per season T: "off|id" / "def|id" -> box (M5 value at the start of T, own-side groups only), rapm
    (M5b season-ahead value for T, fit on plays before T), has_box. Data before T only (ml_m5b.Ctx)."""
    out = {}
    for T in seasons:
        pr = ctx.priors(T)
        r = ctx.data.fit(T, pr, rapm.TUNED).ratings()
        side_of = pr["group"].map(lambda g: pos.SIDE.get(g) if isinstance(g, str) else None)
        parts = []
        for side in ("off", "def"):
            vb = pr["v_box"].where(side_of == side, 0.0)
            hb = ((pr["n_box"] > 0) & (side_of == side)).astype(float)
            rr = r[r["side"] == side].set_index("gsis_id")["value"]
            ids = sorted(set(vb.index) | set(rr.index))
            parts.append(pd.DataFrame({"box": vb.reindex(ids).fillna(0.0).to_numpy(),
                                       "rapm": rr.reindex(ids).fillna(0.0).to_numpy(),
                                       "has_box": hb.reindex(ids).fillna(0.0).to_numpy()},
                                      index=[f"{side}|{i}" for i in ids]))
        out[int(T)] = pd.concat(parts)
    return out


def load_sources(last: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    yrs = range(FIRST, last + 1)
    pbp = mldata.load_pbp(yrs, columns=pdata.PBP_COLUMNS + ["season"])
    part = mldata.load_participation(yrs, columns=pdata.PART_COLUMNS)
    return pbp, part[pdata.PART_COLUMNS]


def load_world(last: int, pbp=None, part=None, ctx=None, ratings=None, priors=None) -> World:
    ctx = ctx if ctx is not None else ml_m5b.build_ctx(last)
    kept = ctx.plays[ctx.plays["season"].between(FIRST, last)]
    if ratings is None:
        ratings = pdata.week_ratings(ctx.tab, ctx.inp.sched, pdata.rating_keys(kept))
    if pbp is None or part is None:
        p0, q0 = load_sources(last)
        pbp = p0 if pbp is None else pbp
        part = q0 if part is None else part
    table = pdata.build_table(kept, pbp, part, ratings)
    priors = priors if priors is not None else build_priors(ctx, range(FIRST, last + 1))
    return World(last, ctx, table, priors, ratings)


# --------------------------------------------------------------------------- models

def predict_season(w: World, S: int, view: str, models=MODELS, save: bool = False) -> tuple[dict, dict]:
    """Season-ahead predictions for every kept play of S: {model: (p (n, 53), p_tov)} and fit info."""
    t = w.table
    train, test = t[t["season"] < S], t[t["season"] == S]
    inner, val = train[train["season"] < S - 1], train[train["season"] == S - 1]
    call = view == "call"
    preds, info = {}, {}
    if "baseline" in models:
        t0 = time.perf_counter()
        k, res = bl.tune_k(inner, val, call)
        preds["baseline"] = bl.Baseline.fit(train, call, k).predict(test)
        info["baseline"] = {"k": k, "val_crps": res, "seconds": time.perf_counter() - t0}
    if "gbm" in models:
        m = gbm.fit_season_ahead(train, view)
        preds["gbm"] = m.predict(test)
        info["gbm"] = m.info
    trunk = None
    for v in [x for x in NETS if x in models]:
        if v == "N0":
            cfgs = [net.NetConfig(variant="N0", **g) for g in net.TRUNK_GRID]
        else:
            if trunk is None:
                trunk = info["N0"]["chosen"] if "N0" in info else dict(net.TRUNK_GRID[0])
            cfgs = [net.NetConfig(variant=v, lr=trunk["lr"], hidden=trunk["hidden"], emb_l2=e) for e in net.EMB_GRID]
        model, inf = net.fit_season_ahead(train, view, cfgs, w.priors if v == "N2" else None)
        preds[v] = model.predict(test, w.priors if v == "N2" else None)
        info[v] = inf
        if v == "N0":
            trunk = {"lr": inf["chosen"]["lr"], "hidden": inf["chosen"]["hidden"]}
        if save:
            model.save(OUT / f"net_{view}_{v}_{S}.pt", {"season_ahead": S, "trained_on": [FIRST, S - 1]})
        if v == "N1" and save:
            norms = model.embedding_norms()
            snaps = snap_counts(train)
            info[v]["embedding_norms"] = norm_table(norms, snaps)
    return preds, info


def snap_counts(train: pd.DataFrame) -> pd.Series:
    o = np.char.add("off|", train[pt.OFF_COLS].to_numpy(object).astype(str).ravel())
    d = np.char.add("def|", train[pt.DEF_COLS].to_numpy(object).astype(str).ravel())
    return pd.Series(np.concatenate([o, d])).value_counts()


def norm_table(norms: pd.DataFrame, snaps: pd.Series) -> dict:
    x = norms.assign(snaps=snaps.reindex(norms["key"]).fillna(0).to_numpy())
    edges = [0, 100, 300, 1000, 3000, np.inf]
    x["bin"] = pd.cut(x["snaps"], edges, right=False, labels=["<100", "100-299", "300-999", "1000-2999", "3000+"])
    g = x.groupby("bin", observed=False)["norm"].agg(["count", "mean", "median"]).reset_index()
    rho = float(x[["norm", "snaps"]].corr(method="spearman").iloc[0, 1])
    return {"by_snaps": g.to_dict("records"), "spearman_norm_snaps": rho, "players": int(len(x))}


def frame_for(test: pd.DataFrame, preds: dict) -> pd.DataFrame:
    """Per-play losses and derived probabilities for every model (the scoring frame)."""
    f = test[["game_id", "play_id", "season", "week", "down", "ydstogo", "yardline_100", "is_pass", "bin", "yards",
              "turnover", "ev_first", "ev_20", "ev_loss"]].reset_index(drop=True)
    for m, (p, tov) in preds.items():
        pp = pm.per_play(p, tov, test)
        f[f"crps_{m}"] = pp["crps"].to_numpy()
        f[f"ll_{m}"] = pp["logloss"].to_numpy()
        f[f"tov_{m}"] = pp["tov"].to_numpy()
        f[f"ptov_{m}"] = tov
        ev = pm.event_probs(p, test["ydstogo"].to_numpy(float), test["yardline_100"].to_numpy(float))
        for k in EVENTS:
            f[f"p{k}_{m}"] = ev[k]
        f[f"ey_{m}"] = pm.expected_yards(p, test["yardline_100"].to_numpy(float))
    return f


def run_window(w: World, window: tuple[int, int], allow_holdout: bool, save: bool = False
               ) -> tuple[dict, dict]:
    """Frames and fit info per view for every season of the window."""
    windows.check(window, allow_holdout)
    frames, infos, timing = {}, {}, {}
    for view in pdata.VIEWS:
        parts = []
        for S in range(window[0], window[1] + 1):
            t0 = time.perf_counter()
            preds, info = predict_season(w, S, view, save=save)
            timing[f"{view}_{S}"] = time.perf_counter() - t0
            parts.append(frame_for(w.table[w.table["season"] == S], preds))
            infos[f"{view}_{S}"] = info
            print(f"  {view} {S}: {len(parts[-1]):,} plays, {timing[f'{view}_{S}']:.0f}s "
                  + " ".join(f"{m} {parts[-1][f'crps_{m}'].mean():.4f}" for m in MODELS), flush=True)
        frames[view] = pd.concat(parts, ignore_index=True)
    infos["timing"] = timing
    return frames, infos


# --------------------------------------------------------------------------- scoring

def cl(d, g) -> dict:
    return bootstrap.cluster_bootstrap_diff(np.asarray(d, float), g, reps=REPS, seed=SEED)


def score_view(f: pd.DataFrame) -> dict:
    g = f["game_id"].to_numpy(object)
    res = {"n": int(len(f)), "games": int(f["game_id"].nunique()), "games_hash": registry.games_hash(f["game_id"].unique()),
           "plays_hash": registry.games_hash((f["game_id"] + ":" + f["play_id"].astype(str)).tolist()), "models": {}}
    for m in MODELS:
        r = {"crps": float(f[f"crps_{m}"].mean()), "logloss": float(f[f"ll_{m}"].mean()),
             "tov_brier": float(f[f"tov_{m}"].mean())}
        for k, col in zip(EVENTS, ("ev_first", "ev_20", "ev_loss")):
            r[f"cal_{k}"] = pm.calibration_check(f[col].to_numpy(float), f[f"p{k}_{m}"].to_numpy(float), reps=CAL_REPS)
            r[f"brier_{k}"] = float(np.mean((f[f"p{k}_{m}"] - f[col]) ** 2))
        r["cal_tov"] = pm.calibration_check(f["turnover"].to_numpy(float), f[f"ptov_{m}"].to_numpy(float), reps=CAL_REPS)
        r["mean_expected_yards"] = float(f[f"ey_{m}"].mean())
        if m != "baseline":
            r["vs_baseline"] = {k: cl(f[f"{c}_{m}"] - f[f"{c}_baseline"], g)
                                for k, c in (("crps", "crps"), ("logloss", "ll"), ("tov_brier", "tov"))}
        res["models"][m] = r
    res["observed_mean_yards"] = float(f["yards"].mean())
    best = min(MODELS, key=lambda m: res["models"][m]["crps"])
    for m in MODELS:
        c = cl(f[f"crps_{m}"] - f[f"crps_{best}"], g) if m != best else None
        res["models"][m]["vs_best"] = c
        res["models"][m]["within_1se"] = bool(m == best or c["diff"] <= c["boot_se"])
    chosen = next(m for m in MODELS if res["models"][m]["within_1se"])
    res["one_se"] = {"best": best, "within": [m for m in MODELS if res["models"][m]["within_1se"]], "chosen": chosen}
    res["pairs"] = {f"{a}-{b}": cl(f[f"crps_{a}"] - f[f"crps_{b}"], g) for a, b in PAIRS}
    res["per_season"] = {}
    for S, h in f.groupby("season"):
        gs = h["game_id"].to_numpy(object)
        res["per_season"][int(S)] = {m: {"crps": float(h[f"crps_{m}"].mean()),
                                         **({"vs_baseline": cl(h[f"crps_{m}"] - h["crps_baseline"], gs)}
                                            if m != "baseline" else {})} for m in MODELS}
    res["splits"] = {}
    for name, fn in SPLITS.items():
        h = f[fn(f).to_numpy()]
        if not len(h):
            continue
        gs = h["game_id"].to_numpy(object)
        res["splits"][name] = {"n": int(len(h)), **{m: {"crps": float(h[f"crps_{m}"].mean()),
                                                        **({"vs_baseline": cl(h[f"crps_{m}"] - h["crps_baseline"], gs)}
                                                           if m != "baseline" else {})} for m in MODELS}}
    c = res["models"][chosen]
    vb = c.get("vs_baseline", {}).get("crps")
    res["acceptance_preview"] = {"chosen": chosen, "crps_vs_baseline": vb,
                                 "first_down_calibration_ok": c["cal_first"]["ok"],
                                 "pass": bool(chosen != "baseline" and vb is not None and vb["ci_high"] < 0
                                              and c["cal_first"]["ok"])}
    return res


def fmt_ci(c: dict | None, nd: int = 4) -> str:
    if not c:
        return ""
    return f"{c['diff']:+.{nd}f} [{c['ci_low']:+.{nd}f}, {c['ci_high']:+.{nd}f}]"


def print_view(view: str, r: dict) -> None:
    print(f"\n== {view} view: {r['n']:,} plays, {r['games']} games (games {r['games_hash']}) ==")
    print(f"{'model':<9}{'CRPS':>8}{'logloss':>9}{'tovBrier':>10}  {'CRPS vs baseline [95% CI]':<30}"
          f"{'1SE':>4}  {'ECE first (p95)':<17}{'ECE 20+':<9}{'ECE loss':<9}{'ECE tov':<8}")
    for m in MODELS:
        x = r["models"][m]
        cal = lambda k: (f"{x[f'cal_{k}']['ece']:.4f}{'' if x[f'cal_{k}']['ok_null'] else '!'}"  # noqa: E731
                         f"{'' if x[f'cal_{k}']['ok'] else 'X'}")
        print(f"{m:<9}{x['crps']:>8.4f}{x['logloss']:>9.4f}{x['tov_brier']:>10.5f}  "
              f"{fmt_ci(x.get('vs_baseline', {}).get('crps')):<30}{'yes' if x['within_1se'] else '':>4}"
              f"{' *' if m == r['one_se']['chosen'] else '  '}{cal('first')} ({x['cal_first']['null_p95']:.4f})  "
              f"{cal('20plus'):<9}{cal('loss'):<9}{cal('tov'):<8}")
    print(f"one-SE: best {r['one_se']['best']}, within {r['one_se']['within']}, chosen {r['one_se']['chosen']}  "
          f"(! = ECE above the one-sided null p95, descriptive; X = fails the rule: above both p95 and "
          f"{pm.ECE_FLOOR})")
    print("log loss vs baseline: " + "; ".join(f"{m} {fmt_ci(r['models'][m]['vs_baseline']['logloss'])}"
                                               for m in MODELS if m != "baseline"))
    print("turnover Brier vs baseline: " + "; ".join(f"{m} {fmt_ci(r['models'][m]['vs_baseline']['tov_brier'], 5)}"
                                                     for m in MODELS if m != "baseline"))
    print("between models (CRPS a - b): " + "; ".join(f"{k} {fmt_ci(v)}" for k, v in r["pairs"].items()))
    for S, v in r["per_season"].items():
        print(f"  {S}: " + "; ".join(f"{m} {v[m]['crps']:.4f}" + (f" ({fmt_ci(v[m]['vs_baseline'])})"
                                                                  if m != "baseline" else "") for m in MODELS))
    print("splits (CRPS; model minus baseline [95% CI]):")
    for name, v in r["splits"].items():
        print(f"  {name:<11}{v['n']:>7,}  base {v['baseline']['crps']:.4f}  "
              + "  ".join(f"{m} {fmt_ci(v[m]['vs_baseline'])}" for m in MODELS if m != "baseline"))
    a = r["acceptance_preview"]
    print(f"Acceptance rule on this window (chosen beats baseline, CI upper < 0, P(first) ECE <= null p95 or "
          f"{pm.ECE_FLOOR}): "
          f"{'PASS' if a['pass'] else 'FAIL'}")


def summarize_info(infos: dict) -> dict:
    """Chosen settings and timings per (view, season) for the record (no per-epoch noise)."""
    out = {}
    for k, v in infos.items():
        if k == "timing":
            continue
        out[k] = {}
        for m, i in v.items():
            if m == "baseline":
                out[k][m] = {"k": i["k"], "val_crps": i["val_crps"]}
            elif m == "gbm":
                out[k][m] = {"chosen": i["chosen"], "n_iter": i["n_iter"], "tov_iter": i["tov_iter"],
                             "val_crps": i["val_crps"]}
            else:
                out[k][m] = {"chosen": {kk: i["chosen"][kk] for kk in ("lr", "hidden", "emb_l2")},
                             "epochs": i["epochs"], "val_loss": i["val_loss"]}
    return out


def timing_info(infos: dict) -> dict:
    out = {"per_view_season_s": infos.get("timing", {})}
    for k, v in infos.items():
        if k == "timing":
            continue
        out[k] = {m: round(float(i.get("total_seconds", i.get("seconds", 0.0))), 2) for m, i in v.items()}
        for m in NETS:
            if m in v:
                out[k][f"{m}_final_fit_s"] = round(float(v[m]["final_seconds"]), 2)
    return out


def data_stats(t: pd.DataFrame) -> dict:
    """Plays, games and target shape per season (training/dev seasons only)."""
    out = {}
    for S, g in t.groupby("season"):
        out[int(S)] = {"plays": int(len(g)), "games": int(g["game_id"].nunique()), "pass_share": float(g["is_pass"].mean()),
                       "mean_yards": float(g["yards"].mean()), "zero_yards": float((g["bin"] == 10).mean()),
                       "td": float(g["td"].mean()), "turnover": float(g["turnover"].mean()),
                       "first_down": float(g["ev_first"].mean()), "gain20": float(g["ev_20"].mean()),
                       "loss": float(g["ev_loss"].mean()), "le_minus10": float((g["bin"] == 0).mean()),
                       "big41": float((g["bin"] == pdata.BIG_BIN).mean()),
                       "feature_missing": {c: float(g[c].isna().mean()) for c in pdata.FEATURES["call"]
                                           if g[c].isna().any()}}
    return out


# --------------------------------------------------------------------------- stages

def fingerprint(last: int) -> dict:
    fp = ml_m5b.fingerprint(last)
    yrs = list(range(FIRST, last + 1))
    for ds in ("pbp", "participation"):
        fp.update({f"data/raw/ml/{k}": v for k, v in mldata.manifest_hashes((ds,), yrs).items()})
    return fp


def write_license() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "LICENSE.txt").write_text(pdata.LICENSE_NOTE + "\nEvery file in this folder (tables, saved networks, "
                                     "predictions) is CC BY-SA 4.0.\n")


def evaluate(window: tuple[int, int], allow_holdout: bool, save: bool) -> tuple[dict, dict, World]:
    t0 = time.perf_counter()
    w = load_world(window[1])
    t_load = time.perf_counter() - t0
    print(f"Inputs loaded in {t_load:.0f}s: {len(w.table):,} plays {FIRST}-{window[1]}")
    frames, infos = run_window(w, window, allow_holdout, save=save)
    t1 = time.perf_counter()
    res = {"window": list(window), "views": {v: score_view(frames[v]) for v in pdata.VIEWS},
           "fits": summarize_info(infos)}
    res["timing"] = {"load_s": t_load, "scoring_s": time.perf_counter() - t1, **timing_info(infos)}
    for v in ("N1",):
        for k, i in infos.items():
            if k != "timing" and v in i and "embedding_norms" in i[v]:
                res.setdefault("embedding_norms", {})[k] = i[v]["embedding_norms"]
    return res, frames, w


def stage_dev(log: bool) -> int:
    print(f"M6 dev evaluation, season-ahead {DEV[0]}-{DEV[1]} (training from {FIRST})")
    res, frames, w = evaluate(DEV, False, save=True)
    res["data_stats"] = data_stats(w.table[w.table["season"] <= DEV[1]])
    for v in pdata.VIEWS:
        print_view(v, res["views"][v])
    if "embedding_norms" in res:
        for k, e in res["embedding_norms"].items():
            print(f"N1 embedding norm vs training snaps ({k}): spearman {e['spearman_norm_snaps']:+.2f}; "
                  + ", ".join(f"{r['bin']}: {r['mean']:.3f} (n {r['count']})" for r in e["by_snaps"]))
    print("\nTiming: " + json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in res["timing"].items()
                                     if k in ("load_s", "scoring_s")}) + " | per view-season: "
          + ", ".join(f"{k} {v:.0f}s" for k, v in res["timing"]["per_view_season_s"].items()))
    write_license()
    (OUT / "dev.json").write_text(json.dumps({**LICENSE, **res}, indent=1, default=float) + "\n")
    pd.concat([f.assign(view=v) for v, f in frames.items()]).to_parquet(OUT / "dev_predictions.parquet", index=False)
    if log:
        log_runs(res, frames, "m6dev", DEV, prefix="m6")
    return 0


def log_runs(res: dict, frames: dict, label: str, window, prefix: str, signoff=None) -> None:
    fp = fingerprint(window[1])
    notes = {"baseline": "Empirical down x distance x zone (x play type) distribution, smoothed toward coarser cells.",
             "gbm": "HistGradientBoosting multiclass over 53 yard bins + turnover classifier.",
             "N0": "PyTorch trunk MLP over the allowlisted features (no players).",
             "N1": "N0 + learned player embeddings, DeepSets pooling (offense and defense).",
             "N2": "N1 + M5 box and M5b RAPM values as of season start as player features."}
    for view in pdata.VIEWS:
        r = res["views"][view]
        gids = frames[view]["game_id"].unique()
        for m in MODELS:
            x = r["models"][m]
            extra = {"window": {"seasons": list(window), "game_type": "REG+POST"}, "license": LICENSE,
                     "view": view, "model": m, "plays_hash": r["plays_hash"],
                     "one_se": {**r["one_se"], "chosen_this": m == r["one_se"]["chosen"]},
                     "result": {k: v for k, v in x.items()},
                     "splits": {k: {"n": v["n"], "crps": v[m]["crps"],
                                    "vs_baseline": v[m].get("vs_baseline")} for k, v in r["splits"].items()},
                     "fits": {k: v.get(m) for k, v in res["fits"].items() if k.startswith(view)}}
            if signoff:
                extra["signoff"] = signoff
            if m == r["one_se"]["chosen"]:
                extra["acceptance"] = r["acceptance_preview"]
            registry.log_run(f"{prefix}_{view}_{m}", label=label, seasons=window, game_ids=gids,
                             metrics={"n": r["n"], "crps": x["crps"], "logloss": x["logloss"],
                                      "tov_brier": x["tov_brier"], "ece_first": x["cal_first"]["ece"]},
                             features=pdata.FEATURES[view], params={"ratings": oa.TUNED.to_dict(),
                                                                    "rapm": rapm.TUNED.to_dict()},
                             data=fp, notes=notes[m] + " CC BY-SA 4.0 (participation-derived)."
                             + (" CHOSEN by the one-SE rule." if m == r["one_se"]["chosen"] else ""), extra=extra)
    print(f"\nRuns written to {registry.RUNS_DIR.relative_to(config.ROOT)}/ (label {label})")


def holdout_signoff_runs() -> list[str]:
    hits = []
    for path in glob.glob(str(registry.RUNS_DIR / "*_m6_signoff_*.json")):
        if json.loads(Path(path).read_text()).get("holdout"):
            hits.append(Path(path).name)
    return sorted(hits)


# The model each view carries into the holdout: the one-SE pick on DEV 2018-2019 (dev.json, 2026-10-05).
# Pre-registered in context/ml.md ("M6 holdout pre-registration"); the holdout's own one-SE pick is secondary.
DEV_CHOSEN = {"situation": "gbm", "call": "gbm"}


def pass_rules(res: dict) -> dict:
    out = {}
    for v in pdata.VIEWS:
        r = res["views"][v]
        m = DEV_CHOSEN[v]
        c = r["models"][m]["vs_baseline"]["crps"]
        cal = r["models"][m]["cal_first"]
        out[v] = {"model": m, "crps_vs_baseline": c, "crps_vs_baseline_ci_high": c["ci_high"],
                  "first_down_ece": cal["ece"], "first_down_null_p95": cal["null_p95"],
                  "first_down_calibration_ok": cal["ok"], "first_down_calibration_ok_null_only": cal["ok_null"],
                  "first_down_ece_floor": cal["floor"], "pass": bool(c["ci_high"] < 0 and cal["ok"]),
                  "window_one_se": r["one_se"]}
    out["primary_pass"] = bool(all(out[v]["pass"] for v in pdata.VIEWS))
    for v in pdata.VIEWS:
        r = res["views"][v]
        out[v]["ablation_N1_minus_N0"] = r["pairs"]["N1-N0"]
        out[v]["ablation_N2_minus_N0"] = r["pairs"]["N2-N0"]
    return out


def stage_signoff(window: tuple[int, int], allow_holdout: bool, log: bool) -> int:
    """The pre-registered sign-off (context/ml.md, "M6 holdout pre-registration"). With the dev window and no
    logging it is the dry run, which must reproduce `dev.json` exactly."""
    label = "holdout" if windows.touches_holdout(window) else "m6dev"
    if label == "holdout":
        prior = holdout_signoff_runs()
        if prior:
            raise SystemExit(f"an M6 holdout sign-off is already recorded ({prior}); it is run once only")
    print(f"== M6 sign-off ({label}), seasons {window[0]}-{window[1]} season-ahead ==")
    res, frames, _ = evaluate(window, allow_holdout, save=False)
    for v in pdata.VIEWS:
        print_view(v, res["views"][v])
    rules = pass_rules(res)
    print("\nPre-registered rules:")
    for v in pdata.VIEWS:
        r = rules[v]
        print(f"  {v:<10} model {r['model']} (dev one-SE pick); CRPS vs baseline {fmt_ci(r['crps_vs_baseline'])}; "
              f"P(first) ECE {r['first_down_ece']:.4f} vs null p95 {r['first_down_null_p95']:.4f} or floor "
              f"{r['first_down_ece_floor']} -> "
              f"{'PASS' if r['pass'] else 'FAIL'}")
    print(f"M6 PRIMARY: {'PASS' if rules['primary_pass'] else 'FAIL'} (passes only if both views pass).")
    result = {**LICENSE, **res, "rules": rules, "label": label}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"signoff_{label}.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    if label != "holdout":
        ref = OUT / "dev.json"
        if not ref.exists():
            print("\nDry run: run `ml_m6.py dev` first to compare against it.")
            return 1
        old = json.loads(ref.read_text())
        d = max(ml_m5b._max_diff(json.loads(json.dumps(res[k], default=float)), old[k]) for k in ("views", "fits"))
        print(f"\nDry-run reproduction of dev.json: max |diff| {d:.1e} -> {'REPRODUCED' if d == 0 else 'NOT REPRODUCED'}")
        return 0 if d == 0 else 1
    if log:
        log_runs(res, frames, label, window, prefix="m6_signoff", signoff="m6_signoff")
    return 0


def stage_fields() -> int:
    """Pre-snap field distributions by season on the kept plays, 2016-2025. Reads participation pre-snap
    columns and the M5b kept-play keys only: no play-by-play outcome is loaded, nothing is scored."""
    yrs = range(FIRST, HOLDOUT[1] + 1)
    kept = pt.load_plays(yrs)
    part = mldata.load_participation(yrs, columns=pdata.PART_COLUMNS)
    rep = pdata.field_report(kept, part)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(rep.round(3).to_string(index=False))
    write_license()
    (OUT / "fields.json").write_text(json.dumps({**LICENSE, "fields": rep.to_dict("records")}, indent=1,
                                                default=float) + "\n")
    return 0


# --------------------------------------------------------------------------- leakage

def corrupt_season(pbp: pd.DataFrame, part: pd.DataFrame, S: int, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copies with every outcome / post-snap field of season S scrambled; the pre-snap inputs are kept.

    Play-by-play: yards gained, touchdown, td team, interception, lost fumble (noise or shuffles).
    Participation: every column outside PART_COLUMNS (rushers, coverage, pressure, routes, ...) if present.
    """
    rng = np.random.default_rng(seed)
    p = pbp.copy()
    m = (p["season"] == S).to_numpy() if "season" in p.columns else p["game_id"].str.startswith(f"{S}_").to_numpy()
    k = int(m.sum())
    p["yards_gained"] = p["yards_gained"].astype(float)
    p.loc[m, "yards_gained"] = rng.integers(-15, 60, k).astype(float)
    for c in ("touchdown", "interception", "fumble_lost"):
        p[c] = p[c].astype(float)
        p.loc[m, c] = rng.integers(0, 2, k).astype(float)
    p["td_team"] = p["td_team"].astype(object)
    p.loc[m, "td_team"] = rng.permutation(p.loc[m, "posteam"].to_numpy(object))
    q = part.copy()
    qm = q["nflverse_game_id"].astype(str).str.startswith(f"{S}_").to_numpy()
    for c in [c for c in q.columns if c not in pdata.PART_COLUMNS]:
        q[c] = q[c].astype(object)
        q.loc[qm, c] = rng.permutation(q.loc[qm, c].to_numpy(object))
    return p, q


def stage_leakcheck(S: int = DEV[1], seed: int = 0) -> int:
    """Real-data checks for season S.

    (1) Season corruption: every outcome and post-snap field of season S is
        scrambled in the raw play-by-play and participation (all participation
        columns are loaded here, post-snap ones included, so the builder's
        column allowlist is exercised); the play table is rebuilt; every model
        is refit on 2016..S-1 and predicts season S, in the call view. The
        season-S feature matrix and every model's predictions must be
        identical to the clean run. Positive control: a GBM with one extra
        feature built from the play's own yards must change.
    (2) Ratings as of week start: for a few weeks of S, everything from the
        week's as_of on is corrupted (asof.corrupt_from) and the week's rating
        features are rebuilt; they must not change. Control: the next week's
        ratings (which include this week's games) must change.
    The start-of-season player priors (N2) come from data before S; M5b's
    leakcheck covers their builder (`ml_m5b.py leakcheck`).
    """
    windows.check((S, S))
    t0 = time.perf_counter()
    ctx = ml_m5b.build_ctx(S)
    pbp, _ = load_sources(S)
    part_full = mldata.load_participation(range(FIRST, S + 1))
    clean = load_world(S, pbp=pbp, part=part_full, ctx=ctx)
    dp, dq = corrupt_season(pbp, part_full, S, seed)
    dirty = load_world(S, pbp=dp, part=dq, ctx=ctx, ratings=clean.ratings, priors=clean.priors)
    tc, td = clean.table[clean.table["season"] == S], dirty.table[dirty.table["season"] == S]
    feats = pdata.FEATURES["call"] + pt.OFF_COLS + pt.DEF_COLS
    same_features = bool(tc[feats].reset_index(drop=True).equals(td[feats].reset_index(drop=True)))
    outcome_changed = float((tc["bin"].to_numpy() != td["bin"].to_numpy()).mean())
    out = {"season": S, "plays": int(len(tc)), "features_identical": same_features,
           "outcomes_changed_share": outcome_changed, "models": {}}
    pc, _ = predict_season(clean, S, "call")
    pd_, _ = predict_season(dirty, S, "call")
    for m in MODELS:
        d = max(float(np.abs(pc[m][0] - pd_[m][0]).max()), float(np.abs(pc[m][1] - pd_[m][1]).max()))
        out["models"][m] = {"max_abs_diff": d, "leak_free": bool(d == 0.0)}
    # positive control: a GBM that also sees the play's own yards (bypasses the allowlist on purpose)
    from sklearn.ensemble import HistGradientBoostingClassifier
    ctrl = {}
    for name, w in (("clean", clean), ("dirty", dirty)):
        t = w.table.assign(own_yards=w.table["yards"])
        tr, te = t[t["season"] < S], t[t["season"] == S]
        cols = pdata.FEATURES["call"] + ["own_yards"]
        c = HistGradientBoostingClassifier(max_iter=30, max_leaf_nodes=7, random_state=SEED).fit(tr[cols], tr["bin"])
        ctrl[name] = c.predict_proba(te[cols])
    dc = float(np.abs(ctrl["clean"] - ctrl["dirty"]).max())
    try:
        pdata.check_features(pdata.FEATURES["call"] + ["own_yards"])
        allow_rejects = False
    except ValueError:
        allow_rejects = True
    out["positive_control"] = {"max_abs_diff": dc, "changed": bool(dc > 0), "allowlist_rejects_it": allow_rejects}
    # (2) ratings as of week start
    sched = ctx.inp.sched
    weeks = sorted(set(clean.table.loc[clean.table["season"] == S, "week"]))
    probe = [weeks[1], weeks[len(weeks) // 2], weeks[-3]]
    rows = []
    for W in probe:
        as_of = sched.loc[(sched["season"] == S) & (sched["week"] == W), "as_of"].min()
        p2, s2 = asof_mod.corrupt_from(ctx.inp.pbp, sched, as_of, seed)
        tab2 = oa.rating_table(p2, s2, oa.TUNED)
        plays = clean.table[(clean.table["season"] == S) & (clean.table["week"] == W)]
        a = pdata.rating_features(plays, pdata.week_ratings(ctx.tab, sched, [(S, W)]))
        b = pdata.rating_features(plays, pdata.week_ratings(tab2, s2, [(S, W)]))
        nxt = weeks[weeks.index(W) + 1]
        a2 = pdata.rating_features(plays.assign(week=nxt), pdata.week_ratings(ctx.tab, sched, [(S, nxt)]))
        b2 = pdata.rating_features(plays.assign(week=nxt), pdata.week_ratings(tab2, s2, [(S, nxt)]))
        rows.append({"week": int(W), "plays": int(len(plays)), "max_abs_diff": float((a - b).abs().to_numpy().max()),
                     "control_next_week_max_abs_diff": float((a2 - b2).abs().to_numpy().max())})
    out["ratings_asof"] = rows
    ok = (same_features and all(v["leak_free"] for v in out["models"].values()) and out["positive_control"]["changed"]
          and allow_rejects and all(r["max_abs_diff"] == 0 and r["control_next_week_max_abs_diff"] > 0 for r in rows))
    out["leak_free"] = bool(ok)
    print(json.dumps(out, indent=1, default=float))
    print(f"Result: {'LEAK-FREE (controls fail as they must)' if ok else 'PROBLEM'}; {time.perf_counter() - t0:.0f}s")
    write_license()
    (OUT / "leakcheck.json").write_text(json.dumps({**LICENSE, **out}, indent=1, default=float) + "\n")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["fields", "dev", "leakcheck", "signoff-dry-run", "signoff"])
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    if a.stage == "fields":
        return stage_fields()
    if a.stage == "dev":
        return stage_dev(not a.no_log)
    if a.stage == "leakcheck":
        return stage_leakcheck()
    if a.stage == "signoff-dry-run":
        return stage_signoff(DEV, allow_holdout=False, log=False)
    if a.stage == "signoff":
        return stage_signoff(HOLDOUT, allow_holdout=True, log=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
