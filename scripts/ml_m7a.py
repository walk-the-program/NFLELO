"""M7a fourth-down decision support: WP model, kicking models, fourth-down valuation, audit, leakage, sign-off.

LICENSE, by output. The WP model and the kicking models use play-by-play and
A4s only, so their predictions and metrics are CC BY 4.0. The fourth-down
valuations, recommendations, the audit and everything derived from them use
the M6 play model (nflverse participation data: NFL Next Gen Stats via
nflverse 2016-2022, FTN Data via nflverse 2023+), so they are CC BY-SA 4.0.
Every saved file and run record carries its license. Spec:
context/ml-m7-method.md (sections 2, 3 Part A, 5).

    python scripts/ml_m7a.py dev              # WP and kicking 2016-2019, fourth downs 2018-2019; logs m7a_* runs
    python scripts/ml_m7a.py audit            # the decision audit of the saved dev valuations (dev logs m7a_audit)
    python scripts/ml_m7a.py leakcheck        # real-data leakage checks on 2019 (weeks 1, 9, 17), positive controls
    python scripts/ml_m7a.py signoff-dry-run  # the sign-off code path on 2018-2019, no logging; must reproduce `dev`
    python scripts/ml_m7a.py signoff          # the ONE pre-registered holdout run, 2020-2025
    options: --boot B (bootstrap refits per season, default 50), --no-log

`signoff` refuses twice over: if any m7a_signoff_* run flagged holdout exists
(it is run once only), and unless the dry run reproduced dev.json on this exact
commit with a clean tree (`data/raw/ml/m7a/signoff_m7adev.json`: reproduced =
true, its git sha = HEAD, not dirty, and the tree is still clean).

Season-ahead protocol. For a target season S: the WP model is fit on REG
2006..S-1 (trees by early stopping on S-1), the FG and punt models on
2006..S-1 (K and trees chosen on S-1), the M6 call-view GBM on 2016..S-1 with
the M6 protocol; pregame strength is A4s walk-forward (fit on seasons < S,
features as of the week). Team inputs (kicker / punter values, M3 ratings,
A4s) use only games before the play's week. Scoring is on REG games of S.
CIs: game-cluster bootstrap, 2,000 reps, seed 20261003. Calibration rule
(`plays.metrics.calibration_check`): ECE <= the one-sided null p95 OR <= 0.010; it is
the pass/fail test for WP, conversion and FG. Punts pass on CRPS against the by-spot baseline (CI upper
bound < 0); their event calibrations and the game-clustered WP null are descriptive.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ml_m3  # noqa: E402
import ml_m6  # noqa: E402
from nflelo import config  # noqa: E402
from nflelo.ml import asof as asof_mod  # noqa: E402
from nflelo.ml import data as mldata  # noqa: E402
from nflelo.ml import registry  # noqa: E402
from nflelo.ml.decisions import fourth as fd  # noqa: E402
from nflelo.ml.decisions import kicking as kk  # noqa: E402
from nflelo.ml.eval import bootstrap, calibration, windows  # noqa: E402
from nflelo.ml.features import opponent_adjust as oa  # noqa: E402
from nflelo.ml.models import logistic  # noqa: E402
from nflelo.ml.plays import data as pdata  # noqa: E402
from nflelo.ml.plays import metrics as pm  # noqa: E402
from nflelo.ml.wp import model as wpm  # noqa: E402

warnings.filterwarnings("ignore", message="X does not have valid feature names")

FIRST_PBP = 1999
FIRST_WP = wpm.FIRST_TRAIN          # 2006
FIRST_FOURTH = 2018                 # first season with two M6 training seasons (2016..S-1, tuned on S-1)
DEV = windows.M7A_DEV               # (2016, 2019)
PRIMARY_DEV = (2018, 2019)          # the acceptance preview window = the fourth-down dev window
HOLDOUT = windows.HOLDOUT
OUT = mldata.ML_DIR / "m7a"
SEED = 20261003
REPS = 2000
CAL_REPS = 500
CLUSTER_NULL_REPS = 200
B_DEFAULT = 50
BRIER_GAP_MAX = 0.002
LIC_BY = {"license": "CC BY 4.0", "note": "Data: nflverse (CC BY 4.0). M7a WP and kicking models (play-by-play, A4s)."}
LIC_SA = {"license": "CC BY-SA 4.0", "note": fd.LICENSE_NOTE}
SD_BANDS = ((-99, -17), (-16, -9), (-8, -4), (-3, -1), (0, 0), (1, 3), (4, 8), (9, 16), (17, 99))
PIN_CLASSES = [i + 1 for i, (a, b) in enumerate(kk.R_BINS) if a >= 81]          # receiver starts inside his 20
LONG_CLASSES = [kk.TD_CLASS] + [i + 1 for i, (a, b) in enumerate(kk.R_BINS) if b <= 60]  # at/beyond his 40, or a TD


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- inputs

@dataclass
class Inputs:
    last: int
    pbp: pd.DataFrame
    a4s: pd.Series
    wp_table: pd.DataFrame
    m6_table: pd.DataFrame
    ratings: pd.DataFrame
    seconds: dict


@contextlib.contextmanager
def no_rating_cache():
    """Compute M3 ratings directly (the on-disk cache is keyed by the manifest, not by corrupted inputs)."""
    orig = ml_m3.oa.cached_ratings
    ml_m3.oa.cached_ratings = lambda tab, sched, keys, cfg, kinds, data_key: oa.compute_ratings(tab, sched, keys, cfg,
                                                                                               kinds)
    try:
        yield
    finally:
        ml_m3.oa.cached_ratings = orig


def a4s_pregame(last: int, games=None, pbp=None, sched=None, through: int | None = None) -> pd.Series:
    """Pregame A4s home-win probability for REG games FIRST_WP..last, walk-forward (season S fit on < S).

    `through` (positive control only): one fit on seasons TRAIN_START..through applied to every season.
    """
    if games is None or pbp is None or sched is None:
        g0, p0, s0 = ml_m3.load_inputs(last_season=last)
        games = g0 if games is None else games
        pbp = p0 if pbp is None else pbp
        sched = s0 if sched is None else sched
    frame, _ = ml_m3.build_frame(games, pbp, sched, last_season=last)
    feats = ml_m3.SIGNOFF_STEPS["A4s"]
    if through is None:
        p, _ = logistic.walk_forward(frame, feats, (FIRST_WP, last), train_start=ml_m3.TRAIN_START)
    else:
        s = frame["season"].to_numpy(int)
        tr = (s >= ml_m3.TRAIN_START) & (s <= through)
        m = logistic.fit_logistic(frame.loc[tr, feats].to_numpy(float), frame.loc[tr, "y"].to_numpy(float),
                                  logistic.season_weights(s[tr], through + 1))
        p = pd.Series(np.where(s >= FIRST_WP, m.predict_proba(frame[feats].to_numpy(float))[:, 1], np.nan),
                      index=frame.index)
    return pd.Series(p.to_numpy(float), index=frame["game_id"].to_numpy()).dropna()


def load_inputs(last: int) -> Inputs:
    t = {}
    t0 = time.perf_counter()
    pbp = mldata.load_pbp(range(FIRST_PBP, last + 1), columns=fd.PBP_COLUMNS)
    t["pbp"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    a4s = a4s_pregame(last)
    t["a4s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    wp_table = wpm.state_table(pbp[pbp["season"] >= FIRST_WP], a4s)
    t["wp_table"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    w = ml_m6.load_world(last, priors={})
    t["m6_world"] = time.perf_counter() - t0
    log(f"inputs through {last}: pbp {len(pbp):,} rows, A4s {len(a4s):,} games, WP states {len(wp_table):,}, "
        f"M6 plays {len(w.table):,} ({sum(t.values()):.0f}s)")
    return Inputs(last, pbp, a4s, wp_table, w.table, w.ratings, t)


def benchmarks(seasons) -> pd.DataFrame:
    """nflfastR wp and vegas_wp (benchmarks only; loaded with keep_market=True and never passed to a model)."""
    return mldata.load_pbp(list(seasons), columns=["game_id", "play_id", "wp", "vegas_wp"], keep_market=True)


# --------------------------------------------------------------------------- per-season work

def realized_wp(dec: pd.DataFrame, wpf: pd.DataFrame) -> np.ndarray:
    """The deciding team's model WP at the next snap after each decision (its final result if none)."""
    nx = wpf[["game_id", "play_id", "posteam", "p_wp"]].rename(columns={"play_id": "n_play", "posteam": "n_pos"})
    q = dec[["game_id", "play_id", "posteam"]].assign(_row=np.arange(len(dec)))
    m = pd.merge_asof(q.sort_values("play_id"), nx.sort_values("n_play"), left_on="play_id", right_on="n_play",
                      by="game_id", direction="forward", allow_exact_matches=False)
    v = np.where(m["n_pos"] == m["posteam"], m["p_wp"], 1 - m["p_wp"])
    v = np.where(m["p_wp"].isna(), dec["out_won"].to_numpy(float)[m["_row"].to_numpy(int)], v)
    out = np.zeros(len(dec))
    out[m["_row"].to_numpy(int)] = v
    return out


def season_work(inp: Inputs, S: int, B: int, bench: pd.DataFrame) -> dict:
    """Every model and prediction for season S (fit on seasons before S)."""
    r, t = {}, {}
    t0 = time.perf_counter()
    wp_iso = wpm.fit_season_ahead(inp.wp_table, S, calibrate=True)
    wp = wp_iso if wpm.CALIBRATE else wpm.without_calibration(wp_iso)
    t["wp"] = time.perf_counter() - t0
    te = inp.wp_table[inp.wp_table["season"] == S].reset_index(drop=True)
    wpf = te[["game_id", "play_id", "season", "week", "qtr", "posteam", "score_diff", "game_secs", "down",
              "ydstogo", "yardline_100", "a4s_prob", "y"]].copy()
    wpf["p_wp"] = wp.predict(te)
    wpf["p_wp_raw"] = wp_iso.predict_raw(te)
    wpf["p_wp_iso"] = wp_iso.predict(te)
    b = wpm.benchmark_columns(bench[bench["game_id"].str.startswith(f"{S}_")], te)
    wpf["bench_wp"], wpf["bench_vegas_wp"] = b["bench_wp"].to_numpy(), b["bench_vegas_wp"].to_numpy()
    r["wp"], r["wp_info"] = wpf, {k: v for k, v in wp.info.items()}
    # kicking
    t0 = time.perf_counter()
    pbpS = inp.pbp[inp.pbp["season"] <= S]
    fgm, fgx = kk.fit_fg_season_ahead(kk.fg_table(pbpS), S, FIRST_WP)
    f = fgx[(fgx["season"] == S) & (fgx["season_type"] == "REG")].reset_index(drop=True)
    r["fg"] = f[["game_id", "play_id", "season", "dist", "indoor", "kval", "made"]].assign(p=fgm.predict(f))
    pmod, px = kk.fit_punt_season_ahead(kk.punt_table(pbpS), S, FIRST_WP)
    pt = px[(px["season"] == S) & (px["season_type"] == "REG")].reset_index(drop=True)
    P = pmod.predict(pt)
    Pb = kk.PuntBaseline.fit(px[px["season"].between(FIRST_WP, S - 1)]).predict(pt)
    cls = pt["cls"].to_numpy(int)
    r["punt"] = pt[["game_id", "play_id", "season", "yardline_100", "cls", "r", "pval"]].assign(
        crps=kk.crps_ordered(P, cls), crps_base=kk.crps_ordered(Pb, cls),
        p_pin=P[:, PIN_CLASSES].sum(axis=1), ev_pin=np.isin(cls, PIN_CLASSES).astype(float),
        p_long=P[:, LONG_CLASSES].sum(axis=1), ev_long=np.isin(cls, LONG_CLASSES).astype(float),
        pb_pin=Pb[:, PIN_CLASSES].sum(axis=1), pb_long=Pb[:, LONG_CLASSES].sum(axis=1))
    r["kick_info"] = {"fg": fgm.info, "punt": pmod.info}
    t["kicking"] = time.perf_counter() - t0
    if S >= FIRST_FOURTH:
        t0 = time.perf_counter()
        comp, tables = fd.fit_components(S, inp.pbp, inp.wp_table, inp.m6_table, wp=wp, fg=(fgm, fgx),
                                         punt=(pmod, px))
        t["components"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        dec = fd.decision_table(inp.pbp, inp.a4s, seasons=[S])
        vals, boot = fd.value_season(comp, dec, inp.ratings, tables, B=B, seed=SEED, log=log)
        vals["realized_wp"] = realized_wp(vals, wpf)
        r["fourth"], r["boot"] = vals, boot
        t["fourth"] = time.perf_counter() - t0
        # M6 call view with the plays' own (actual) pre-snap features: 3rd and 4th downs of S
        m6S = inp.m6_table[(inp.m6_table["season"] == S) & (inp.m6_table["down"] >= 3)
                           & (inp.m6_table["season_type"] == "REG")].reset_index(drop=True)
        Pm = comp.yards.predict_X(m6S[pdata.FEATURES["call"]], m6S["yardline_100"].to_numpy(float))
        r["m6"] = m6S[["game_id", "play_id", "season", "down", "ydstogo", "is_pass", "ev_first"]].assign(
            pf=pm.event_probs(Pm, m6S["ydstogo"].to_numpy(float), m6S["yardline_100"].to_numpy(float))["first"])
        r["comp_info"] = comp.info
    r["seconds"] = t
    return r


# --------------------------------------------------------------------------- scoring

def cl(d, g) -> dict:
    return bootstrap.cluster_bootstrap_diff(np.asarray(d, float), g, reps=REPS, seed=SEED)


def ece_cluster_null(p, games, reps: int = CLUSTER_NULL_REPS, seed: int = SEED) -> dict:
    """ECE of a calibrated forecaster whose play outcomes share one uniform draw per game (descriptive).

    Each play's outcome is 1{U_game < p}: marginally calibrated, but plays of the same game move
    together, as real play outcomes (one game result) do. The independent-Bernoulli null of
    `calibration.ece_null_range` ignores that and is far too narrow for play-level WP.
    """
    p = np.asarray(p, float)
    codes = pd.factorize(np.asarray(games, dtype=object))[0]
    rng = np.random.default_rng(seed)
    e = [calibration.ece((rng.uniform(size=codes.max() + 1)[codes] < p).astype(float), p) for _ in range(reps)]
    lo, mid, hi = np.percentile(e, (5, 50, 95))
    return {"p05": float(lo), "median": float(mid), "p95": float(hi), "reps": reps}


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def logloss(y, p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def score_wp(f: pd.DataFrame, detail: bool = True) -> dict:
    """Our WP vs nflfastR wp and vegas_wp on the same plays (both benchmarks present)."""
    f = f[f["bench_wp"].notna() & f["bench_vegas_wp"].notna()]
    y, g = f["y"].to_numpy(float), f["game_id"].to_numpy(object)
    cols = {"m7a_wp": "p_wp", "m7a_wp_isotonic": "p_wp_iso", "nflfastr_wp": "bench_wp", "vegas_wp": "bench_vegas_wp"}
    res = {"n": int(len(f)), "games": int(f["game_id"].nunique()), "games_hash": registry.games_hash(f["game_id"].unique()),
           "dropped_no_benchmark": None, "models": {}}
    for k, c in cols.items():
        p = f[c].to_numpy(float)
        res["models"][k] = {"brier": brier(y, p), "logloss": float(logloss(y, p).mean()),
                            "cal": pm.calibration_check(y, p, reps=CAL_REPS)}
        if detail:
            res["models"][k]["cluster_null"] = ece_cluster_null(p, g)
    sq = lambda c: (f[c].to_numpy(float) - y) ** 2  # noqa: E731
    res["diffs"] = {f"m7a_wp-{b}": {"brier": cl(sq("p_wp") - sq(c), g),
                                    "logloss": cl(logloss(y, f["p_wp"]) - logloss(y, f[c]), g)}
                    for b, c in (("nflfastr_wp", "bench_wp"), ("vegas_wp", "bench_vegas_wp"),
                                 ("isotonic", "p_wp_iso"))}
    if detail:
        res["by_quarter"] = {}
        for q, h in f.groupby("qtr"):
            yy = h["y"].to_numpy(float)
            res["by_quarter"][int(q)] = {"n": int(len(h)), **{k: {"brier": brier(yy, h[c]),
                                                                  "ece": calibration.ece(yy, h[c].to_numpy(float))}
                                                              for k, c in cols.items()},
                                         "cal_m7a": pm.calibration_check(yy, h["p_wp"].to_numpy(float), reps=CAL_REPS)}
        res["by_score_band"] = {}
        for lo, hi in SD_BANDS:
            h = f[f["score_diff"].between(lo, hi)]
            if not len(h):
                continue
            yy = h["y"].to_numpy(float)
            res["by_score_band"][f"{lo}..{hi}"] = {
                "n": int(len(h)), **{k: {"brier": brier(yy, h[c]), "ece": calibration.ece(yy, h[c].to_numpy(float))}
                                     for k, c in cols.items()},
                "cal_m7a": pm.calibration_check(yy, h["p_wp"].to_numpy(float), reps=CAL_REPS)}
    return res


def score_fg(f: pd.DataFrame) -> dict:
    y, p = f["made"].to_numpy(float), f["p"].to_numpy(float)
    return {"n": int(len(f)), "brier": brier(y, p), "logloss": float(logloss(y, p).mean()),
            "cal": pm.calibration_check(y, p, reps=CAL_REPS),
            "by_distance": {f"{a}-{b}": {"n": int(m.sum()), "pred": float(p[m].mean()), "made": float(y[m].mean())}
                            for a, b in ((18, 29), (30, 39), (40, 49), (50, 70))
                            for m in [(f["dist"] >= a).to_numpy() & (f["dist"] <= b).to_numpy()] if m.any()}}


def score_punt(f: pd.DataFrame) -> dict:
    g = f["game_id"].to_numpy(object)
    return {"n": int(len(f)), "crps": float(f["crps"].mean()), "crps_baseline": float(f["crps_base"].mean()),
            "crps_vs_baseline": cl(f["crps"] - f["crps_base"], g),
            "cal_pin": pm.calibration_check(f["ev_pin"].to_numpy(float), f["p_pin"].to_numpy(float), reps=CAL_REPS),
            "cal_long": pm.calibration_check(f["ev_long"].to_numpy(float), f["p_long"].to_numpy(float), reps=CAL_REPS),
            "baseline_cal_pin": pm.calibration_check(f["ev_pin"].to_numpy(float), f["pb_pin"].to_numpy(float),
                                                     reps=CAL_REPS),
            "baseline_cal_long": pm.calibration_check(f["ev_long"].to_numpy(float), f["pb_long"].to_numpy(float),
                                                      reps=CAL_REPS)}


def pred_vs_actual(pred, actual, games) -> dict:
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    if not len(pred):
        return {"n": 0}
    c = cl(actual - pred, games)
    return {"n": int(len(pred)), "pred": float(pred.mean()), "actual": float(actual.mean()), "actual_minus_pred": c}


def selection_check(fr: pd.DataFrame, m6: pd.DataFrame) -> dict:
    """Predicted vs actual conversion on 4th-and-1/2 attempts (and 3rd downs at the same distance)."""
    out = {}
    go = fr[fr["choice"] == "go"]
    for d in (1, 2):
        h = go[go["ydstogo"] == d]
        out[f"4th_and_{d}_engine"] = pred_vs_actual(h["p_conv"], h["out_converted"], h["game_id"].to_numpy(object))
        a = m6[(m6["down"] == 4) & (m6["ydstogo"] == d)]
        out[f"4th_and_{d}_actual_features"] = pred_vs_actual(a["pf"], a["ev_first"], a["game_id"].to_numpy(object))
        t = m6[(m6["down"] == 3) & (m6["ydstogo"] == d)]
        out[f"3rd_and_{d}_actual_features"] = pred_vs_actual(t["pf"], t["ev_first"], t["game_id"].to_numpy(object))
    h = go[go["ydstogo"] <= 2]
    out["4th_and_1_2_engine"] = pred_vs_actual(h["p_conv"], h["out_converted"], h["game_id"].to_numpy(object))
    return out


def clock(qtr, game_secs) -> str:
    qs = int(game_secs - (4 - int(qtr)) * 900)
    return f"Q{int(qtr)} {qs // 60}:{qs % 60:02d}"


def audit(fr: pd.DataFrame, top: int = 10) -> dict:
    """The decision audit (every number is MODEL-ESTIMATED: counterfactual outcomes are never observed)."""
    n_games = int(fr["game_id"].nunique())
    has_band = "tossup" in fr.columns
    res = {"label": "model-estimated", "n": int(len(fr)), "games": n_games,
           "team_games": int(fr[["game_id", "posteam"]].drop_duplicates().shape[0]),
           "choice_counts": fr["choice"].value_counts().to_dict(),
           "recommended_counts": fr["best"].value_counts().to_dict(),
           "crosstab_choice_by_recommended": pd.crosstab(fr["choice"], fr["best"]).to_dict(),
           "matched_share": float(fr["matched"].mean()),
           "wp_lost_total": float(fr["lost"].sum()),
           "wp_lost_per_game": float(fr["lost"].sum() / n_games),
           "wp_lost_per_team_game": float(fr["lost"].sum() / fr[["game_id", "posteam"]].drop_duplicates().shape[0]),
           "wp_lost_by_choice": fr.groupby("choice")["lost"].sum().to_dict(),
           "wp_lost_by_recommended_go_when_kicked": float(fr.loc[(fr["best"] == "go") & (fr["choice"] != "go"),
                                                                 "lost"].sum()),
           "mean_margin": float(fr["margin"].mean())}
    if has_band:
        clear = ~fr["tossup"]
        res.update({"tossup_share": float(fr["tossup"].mean()),
                    "matched_share_clear_calls": float(fr.loc[clear, "matched"].mean()),
                    "wp_lost_clear_calls": float(fr.loc[clear, "lost"].sum()),
                    "clear_calls": int(clear.sum())})
    cols = ["season", "week", "posteam", "defteam", "qtr", "game_secs", "score_diff", "ydstogo", "yardline_100",
            "choice", "best", "wp_go", "wp_fg", "wp_punt", "lost", "p_conv", "p_fg"] + (
        ["band_lo", "band_hi", "tossup"] if has_band else []) + ["desc"]
    t = fr.sort_values("lost", ascending=False).head(top)[cols].copy()
    t["clock"] = [clock(q, s) for q, s in zip(t["qtr"], t["game_secs"])]
    t["desc"] = t["desc"].str.slice(0, 120)
    res["top_misses"] = t.to_dict("records")
    # consistency: realized outcomes vs the model's estimate, by agreement group (descriptive)
    groups = {"went_for_it_model_said_kick": (fr["choice"] == "go") & (fr["best"] != "go"),
              "kicked_model_said_go": (fr["choice"] != "go") & (fr["best"] == "go"),
              "punted_model_said_fg": (fr["choice"] == "punt") & (fr["best"] == "fg"),
              "kicked_fg_model_said_punt": (fr["choice"] == "fg") & (fr["best"] == "punt"),
              "matched": fr["matched"]}
    res["consistency"] = {}
    for name, m in groups.items():
        h = fr[m.to_numpy()]
        if not len(h):
            continue
        g = h["game_id"].to_numpy(object)
        c = {"n": int(len(h)), "model_wp_choice": float(h["wp_choice"].mean()),
             "model_wp_recommended": float(h["wp_best"].mean()),
             "realized_next_snap_wp": float(h["realized_wp"].mean()), "won": float(h["out_won"].mean()),
             "realized_minus_model_choice": cl(h["realized_wp"] - h["wp_choice"], g)}
        gg = h[h["choice"] == "go"]
        if len(gg):
            c["go_pred_conversion"] = float(gg["p_conv"].mean())
            c["go_actual_conversion"] = float(gg["out_converted"].mean())
        ff = h[h["choice"] == "fg"]
        if len(ff):
            c["fg_pred_make"] = float(ff["p_fg"].mean())
            c["fg_actual_make"] = float(ff["out_made"].mean())
        res["consistency"][name] = c
    # realized vs model for each chosen option (does each option's valuation hold up where it was chosen?)
    res["realized_by_choice"] = {
        o: {"n": int(len(h)), "model_wp_choice": float(h["wp_choice"].mean()),
            "realized_next_snap_wp": float(h["realized_wp"].mean()),
            "realized_minus_model": cl(h["realized_wp"] - h["wp_choice"], h["game_id"].to_numpy(object))}
        for o, h in fr.groupby("choice")}
    return res


def score_window(work: dict, seasons) -> dict:
    """The pre-registered block (acceptance rules) over `seasons`, plus the selection check and the audit."""
    S = [s for s in seasons if s in work]
    wpf = pd.concat([work[s]["wp"] for s in S], ignore_index=True)
    fg = pd.concat([work[s]["fg"] for s in S], ignore_index=True)
    pu = pd.concat([work[s]["punt"] for s in S], ignore_index=True)
    res = {"seasons": [int(min(S)), int(max(S))], "wp": score_wp(wpf), "fg": score_fg(fg), "punt": score_punt(pu)}
    w = res["wp"]
    gap = w["diffs"]["m7a_wp-nflfastr_wp"]["brier"]["diff"]
    rules = {"wp_calibration_ok": w["models"]["m7a_wp"]["cal"]["ok"], "wp_brier_gap_to_nflfastr": gap,
             "wp_brier_gap_ok": bool(gap <= BRIER_GAP_MAX), "fg_calibration_ok": res["fg"]["cal"]["ok"],
             # punt rule (2026-10-05, before any holdout number): the valuation consumes the whole distribution,
             # so it passes iff its CRPS beats the by-spot baseline with the game-cluster CI upper bound below 0;
             # the two event calibrations are descriptive
             "punt_crps_ok": bool(res["punt"]["crps_vs_baseline"]["ci_high"] < 0),
             "punt_events_calibration_descriptive": bool(res["punt"]["cal_pin"]["ok"] and res["punt"]["cal_long"]["ok"])}
    fourth = [s for s in S if "fourth" in work[s]]
    if fourth:
        fr = pd.concat([work[s]["fourth"] for s in fourth], ignore_index=True)
        m6 = pd.concat([work[s]["m6"] for s in fourth], ignore_index=True)
        go = fr[fr["choice"] == "go"]
        res["conversion"] = {
            "engine_on_attempts": {"n": int(len(go)), "brier": brier(go["out_converted"], go["p_conv"]),
                                   "cal": pm.calibration_check(go["out_converted"].to_numpy(float),
                                                               go["p_conv"].to_numpy(float), reps=CAL_REPS)},
            "m6_actual_features_4th": pm.calibration_check(m6.loc[m6["down"] == 4, "ev_first"].to_numpy(float),
                                                           m6.loc[m6["down"] == 4, "pf"].to_numpy(float), reps=CAL_REPS),
            "m6_actual_features_3rd": pm.calibration_check(m6.loc[m6["down"] == 3, "ev_first"].to_numpy(float),
                                                           m6.loc[m6["down"] == 3, "pf"].to_numpy(float), reps=CAL_REPS),
            "m6_actual_features_3rd_short": pm.calibration_check(
                m6.loc[(m6["down"] == 3) & (m6["ydstogo"] <= 2), "ev_first"].to_numpy(float),
                m6.loc[(m6["down"] == 3) & (m6["ydstogo"] <= 2), "pf"].to_numpy(float), reps=CAL_REPS)}
        rules["conversion_calibration_ok"] = res["conversion"]["engine_on_attempts"]["cal"]["ok"]
        res["selection"] = selection_check(fr, m6)
        res["audit"] = audit(fr)
        res["plays_hash"] = registry.games_hash((fr["game_id"] + ":" + fr["play_id"].astype(str)).tolist())
    rules["wp_pass"] = bool(rules["wp_calibration_ok"] and rules["wp_brier_gap_ok"])
    rules["components_pass"] = bool(rules["fg_calibration_ok"] and rules["punt_crps_ok"]
                                    and rules.get("conversion_calibration_ok", False))
    rules["pass"] = bool(rules["wp_pass"] and rules["components_pass"])
    res["rules"] = rules
    return res


def per_season(work: dict) -> dict:
    out = {}
    for S, w in sorted(work.items()):
        out[int(S)] = {"wp": score_wp(w["wp"], detail=False), "fg": score_fg(w["fg"]), "punt": score_punt(w["punt"]),
                       "wp_fit": {k: w["wp_info"].get(k) for k in ("n_iter", "n_train", "trained_on")},
                       "fg_fit": {k: w["kick_info"]["fg"].get(k) for k in ("K", "coef", "intercept", "n_train")},
                       "punt_fit": {k: w["kick_info"]["punt"].get(k) for k in ("K", "n_iter", "n_train")},
                       "components": {**{k: w["comp_info"].get(k) for k in ("ko_spot", "durations", "run_share", "m6")},
                                      "ko_start_mean_used": float(w["fourth"]["ko_start"].mean())}
                       if "comp_info" in w else None,
                       "seconds": w["seconds"]}
    return out


# --------------------------------------------------------------------------- printing

def fmt_ci(c: dict | None, nd: int = 4) -> str:
    if not c:
        return ""
    return f"{c['diff']:+.{nd}f} [{c['ci_low']:+.{nd}f}, {c['ci_high']:+.{nd}f}]"


def fmt_cal(c: dict) -> str:
    return (f"ECE {c['ece']:.4f} (null p95 {c['null_p95']:.4f}) "
            f"{'ok' if c['ok'] else 'FAIL'}{'' if c['ok_null'] else ' [floor]' if c['ok'] else ''}")


def print_window(r: dict) -> None:
    w = r["wp"]
    log(f"\n== WP model, REG {r['seasons'][0]}-{r['seasons'][1]}: {w['n']:,} plays, {w['games']} games "
        f"(games {w['games_hash']}) ==")
    log(f"{'model':<17}{'Brier':>8}{'logloss':>9}  calibration                        cluster-null ECE p95")
    for k, m in w["models"].items():
        log(f"{k:<17}{m['brier']:>8.4f}{m['logloss']:>9.4f}  {fmt_cal(m['cal']):<35}{m['cluster_null']['p95']:.4f}")
    for k, d in w["diffs"].items():
        log(f"  {k}: Brier {fmt_ci(d['brier'])}, log loss {fmt_ci(d['logloss'])}")
    log("  by quarter (Brier m7a / nflfastr / vegas; m7a ECE, rule):")
    for q, v in w["by_quarter"].items():
        log(f"    Q{q} n {v['n']:>6,}  {v['m7a_wp']['brier']:.4f} / {v['nflfastr_wp']['brier']:.4f} / "
            f"{v['vegas_wp']['brier']:.4f}   {fmt_cal(v['cal_m7a'])}")
    log("  by score difference (posteam):")
    for b, v in w["by_score_band"].items():
        log(f"    {b:>8} n {v['n']:>6,}  {v['m7a_wp']['brier']:.4f} / {v['nflfastr_wp']['brier']:.4f} / "
            f"{v['vegas_wp']['brier']:.4f}   {fmt_cal(v['cal_m7a'])}")
    f = r["fg"]
    log(f"\n== FG make: {f['n']:,} REG attempts: Brier {f['brier']:.4f}, {fmt_cal(f['cal'])}")
    for k, v in f["by_distance"].items():
        log(f"    {k:>6} n {v['n']:>5}  pred {v['pred']:.3f}  made {v['made']:.3f}")
    p = r["punt"]
    log(f"== Punts: {p['n']:,} REG punts: CRPS {p['crps']:.3f} yards vs by-spot baseline {p['crps_baseline']:.3f}, "
        f"diff {fmt_ci(p['crps_vs_baseline'], 3)}")
    log(f"    P(inside 20) {fmt_cal(p['cal_pin'])}; P(at/beyond 40 or TD) {fmt_cal(p['cal_long'])}")
    if "conversion" in r:
        c = r["conversion"]
        e = c["engine_on_attempts"]
        log(f"== Conversion (engine, actual go attempts n {e['n']}): Brier {e['brier']:.4f}, {fmt_cal(e['cal'])}")
        log(f"    M6 actual features: 4th downs {fmt_cal(c['m6_actual_features_4th'])}; "
            f"3rd downs {fmt_cal(c['m6_actual_features_3rd'])}; 3rd-and-1/2 {fmt_cal(c['m6_actual_features_3rd_short'])}")
        log("== Selection check (actual minus predicted conversion [95% CI]):")
        for k, v in r["selection"].items():
            if v.get("n"):
                log(f"    {k:<30} n {v['n']:>5}  pred {v['pred']:.3f}  actual {v['actual']:.3f}  "
                    f"{fmt_ci(v['actual_minus_pred'], 3)}")
        print_audit(r["audit"])
    ru = r["rules"]
    log(f"\nAcceptance rules on this window: WP calibration {'ok' if ru['wp_calibration_ok'] else 'FAIL'}, "
        f"Brier gap to nflfastR wp {ru['wp_brier_gap_to_nflfastr']:+.4f} (<= {BRIER_GAP_MAX}: "
        f"{'ok' if ru['wp_brier_gap_ok'] else 'FAIL'}); conversion {'ok' if ru.get('conversion_calibration_ok') else 'FAIL'}, "
        f"FG {'ok' if ru['fg_calibration_ok'] else 'FAIL'}, punt CRPS vs baseline {'ok' if ru['punt_crps_ok'] else 'FAIL'} "
        f"(punt event calibration, descriptive: {'ok' if ru['punt_events_calibration_descriptive'] else 'fail'})  "
        f"=> {'PASS' if ru['pass'] else 'FAIL'}")


def print_audit(a: dict) -> None:
    log(f"\n== Decision audit (MODEL-ESTIMATED), {a['n']:,} fourth downs, {a['games']} games ==")
    log(f"  choices {a['choice_counts']}; model recommends {a['recommended_counts']}")
    log(f"  matched {a['matched_share']:.3f}" + (f" (clear calls {a['matched_share_clear_calls']:.3f}); toss-up "
                                                  f"share {a['tossup_share']:.3f}" if "tossup_share" in a else ""))
    log(f"  model-estimated WP given up: total {a['wp_lost_total']:.2f} wins, {a['wp_lost_per_game']:.4f} per game, "
        f"{a['wp_lost_per_team_game']:.4f} per team-game; by choice {({k: round(v, 2) for k, v in a['wp_lost_by_choice'].items()})}")
    log("  top misses (model-estimated WP given up):")
    for t in a["top_misses"]:
        log(f"    {t['season']} wk{t['week']:>2} {t['posteam']:>3} v {t['defteam']:<3} {t['clock']:>8} "
            f"score {t['score_diff']:+.0f} 4th&{t['ydstogo']:.0f} at opp {t['yardline_100']:.0f}: {t['choice']} "
            f"(model {t['best']}) go {t['wp_go']:.3f} fg {t['wp_fg']:.3f} punt {t['wp_punt']:.3f} lost {t['lost']:.3f}")
    log("  consistency (descriptive):")
    for k, v in a["consistency"].items():
        log(f"    {k:<28} n {v['n']:>5}  model WP(choice) {v['model_wp_choice']:.3f}  model WP(rec) "
            f"{v['model_wp_recommended']:.3f}  realized next-snap WP {v['realized_next_snap_wp']:.3f}  won {v['won']:.3f}"
            + (f"  conv pred {v['go_pred_conversion']:.3f} actual {v['go_actual_conversion']:.3f}"
               if "go_pred_conversion" in v else ""))
    for k, v in a["realized_by_choice"].items():
        log(f"    chose {k:<5} n {v['n']:>5}  model {v['model_wp_choice']:.4f}  realized {v['realized_next_snap_wp']:.4f}  "
            f"diff {fmt_ci(v['realized_minus_model'])}")


# --------------------------------------------------------------------------- stages

def evaluate(window: tuple[int, int], allow_holdout: bool, B: int) -> tuple[dict, dict]:
    windows.check(window, allow_holdout)
    t0 = time.perf_counter()
    inp = load_inputs(window[1])
    bench = benchmarks(range(window[0], window[1] + 1))
    work = {}
    for S in range(window[0], window[1] + 1):
        ts = time.perf_counter()
        work[S] = season_work(inp, S, B, bench)
        log(f"  season {S} done ({time.perf_counter() - ts:.0f}s): " + ", ".join(
            f"{k} {v:.0f}s" for k, v in work[S]["seconds"].items()))
    primary = [s for s in range(window[0], window[1] + 1) if s >= FIRST_FOURTH]
    res = {"window": list(window), "primary_seasons": [min(primary), max(primary)], "boot": B,
           "primary": score_window(work, primary), "per_season": per_season(work),
           "inputs_seconds": inp.seconds, "total_seconds": time.perf_counter() - t0,
           "license": {"wp_and_kicking": LIC_BY, "fourth_down": LIC_SA}}
    return res, work


def write_license() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "LICENSE.txt").write_text(
        "Files in this folder, by name:\n"
        "  wp_*.parquet, fg_*.parquet, punt_*.parquet: CC BY 4.0 (" + LIC_BY["note"] + ")\n"
        "  fourth_*.parquet, boot_*.npy, m6_*.parquet, dev.json, signoff_*.json, audit_*.json, leakcheck.json: "
        "CC BY-SA 4.0 (" + LIC_SA["note"] + ")\n")


def save_frames(work: dict, label: str) -> None:
    write_license()
    for key in ("wp", "fg", "punt", "fourth", "m6"):
        parts = [w[key] for w in work.values() if key in w]
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(OUT / f"{key}_{label}.parquet", index=False)
    for S, w in work.items():
        if w.get("boot") is not None:
            np.save(OUT / f"boot_{label}_{S}.npy", w["boot"])


def log_runs(res: dict, work: dict, label: str, prefix: str, signoff: str | None = None) -> None:
    r = res["primary"]
    seasons = tuple(r["seasons"])
    fdat = registry.data_fingerprint(games_csv=True, ml_datasets=("pbp", "schedules", "participation"),
                                     seasons=range(FIRST_PBP, seasons[1] + 1))
    S = [s for s in range(seasons[0], seasons[1] + 1)]
    gid = lambda key: pd.concat([work[s][key]["game_id"] for s in S]).unique().tolist()  # noqa: E731
    extra = {"signoff": signoff} if signoff else {}
    w = r["wp"]
    for name, k in (("wp", "m7a_wp"), ("wp_isotonic", "m7a_wp_isotonic"), ("bench_nflfastr_wp", "nflfastr_wp"),
                    ("bench_vegas_wp", "vegas_wp")):
        m = w["models"][k]
        registry.log_run(f"{prefix}_{name}", label=label, seasons=seasons, game_ids=gid("wp"),
                         metrics={"n": w["n"], "brier": m["brier"], "logloss": m["logloss"], "ece": m["cal"]["ece"],
                                  "cal": m["cal"]},
                         features=wpm.FEATURES if k.startswith("m7a") else [], data=fdat,
                         params={"monotone": wpm.MONOTONE, "gbm": wpm.PARAMS, "calibrated": k == "m7a_wp_isotonic"},
                         notes="M7a WP model (CC BY)" if k.startswith("m7a") else "benchmark only, never a feature",
                         extra={**extra, **LIC_BY, "diffs": w["diffs"] if k == "m7a_wp" else None,
                                "rules": r["rules"] if k == "m7a_wp" else None})
    registry.log_run(f"{prefix}_fg", label=label, seasons=seasons, game_ids=gid("fg"),
                     metrics={"n": r["fg"]["n"], "brier": r["fg"]["brier"], "logloss": r["fg"]["logloss"],
                              "ece": r["fg"]["cal"]["ece"], "cal": r["fg"]["cal"]}, features=kk.FG_FEATURES, data=fdat,
                     params={"K_grid": kk.K_FG_GRID, "half_life": kk.FG_HALF_LIFE}, extra={**extra, **LIC_BY})
    registry.log_run(f"{prefix}_punt", label=label, seasons=seasons, game_ids=gid("punt"),
                     metrics={"n": r["punt"]["n"], "crps": r["punt"]["crps"], "crps_baseline": r["punt"]["crps_baseline"],
                              "crps_vs_baseline": r["punt"]["crps_vs_baseline"], "cal_pin": r["punt"]["cal_pin"],
                              "cal_long": r["punt"]["cal_long"]}, features=kk.PUNT_FEATURES, data=fdat,
                     params={"K_grid": kk.K_PUNT_GRID, "params": kk.PUNT_PARAMS}, extra={**extra, **LIC_BY})
    if "conversion" in r:
        registry.log_run(f"{prefix}_conversion", label=label, seasons=seasons, game_ids=gid("fourth"),
                         metrics={"n": r["conversion"]["engine_on_attempts"]["n"], **r["conversion"]},
                         features=pdata.FEATURES["call"], data=fdat,
                         params={"n_configs": fd.N_CONFIGS, "buckets": fd.BUCKETS, "mix": "empirical go-for-it run share"},
                         extra={**extra, **LIC_SA, "selection": r["selection"]})
        registry.log_run(f"{prefix}_audit", label=label, seasons=seasons, game_ids=gid("fourth"),
                         metrics={"n": r["audit"]["n"], **{k: v for k, v in r["audit"].items()
                                                           if k not in ("top_misses", "consistency",
                                                                        "crosstab_choice_by_recommended")}},
                         data=fdat, params={"boot": res["boot"], "level": 0.90},
                         notes="every audit number is model-estimated",
                         extra={**extra, **LIC_SA, "audit": r["audit"], "plays_hash": r.get("plays_hash")})
    log(f"\nRuns written to {registry.RUNS_DIR.relative_to(config.ROOT)}/ (label {label})")


def stage_dev(log_runs_: bool, B: int) -> int:
    res, work = evaluate(DEV, allow_holdout=False, B=B)
    print_window(res["primary"])
    log("\nPer season (WP Brier m7a / nflfastr / vegas; FG Brier; punt CRPS vs baseline):")
    for S, v in res["per_season"].items():
        m = v["wp"]["models"]
        log(f"  {S}: WP {m['m7a_wp']['brier']:.4f} / {m['nflfastr_wp']['brier']:.4f} / {m['vegas_wp']['brier']:.4f} "
            f"(isotonic {m['m7a_wp_isotonic']['brier']:.4f}; gap {fmt_ci(v['wp']['diffs']['m7a_wp-nflfastr_wp']['brier'])}); "
            f"FG {v['fg']['brier']:.4f} {fmt_cal(v['fg']['cal'])}; punt {v['punt']['crps']:.3f} vs "
            f"{v['punt']['crps_baseline']:.3f}")
    save_frames(work, "m7adev")
    (OUT / "dev.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    if log_runs_:
        log_runs(res, work, "m7adev", "m7a")
    log(f"\ndev total {res['total_seconds']:.0f}s")
    return 0


def stage_audit(label: str = "m7adev", log_runs_: bool = True) -> int:
    fr = pd.read_parquet(OUT / f"fourth_{label}.parquet")
    a = audit(fr)
    print_audit(a)
    log("\n  top misses, full context:")
    for t in a["top_misses"]:
        log(f"    {t['season']} wk{t['week']} {t['posteam']} v {t['defteam']} {t['clock']}: {t['desc']}")
    (OUT / f"audit_{label}.json").write_text(json.dumps({**a, **LIC_SA}, indent=1, default=float) + "\n")
    if log_runs_ and label == "m7adev":
        seasons = (int(fr["season"].min()), int(fr["season"].max()))
        registry.log_run("m7a_audit", label=label, seasons=seasons, game_ids=fr["game_id"].unique().tolist(),
                         metrics={"n": a["n"], "matched_share": a["matched_share"], "wp_lost_total": a["wp_lost_total"],
                                  "wp_lost_per_game": a["wp_lost_per_game"], "tossup_share": a.get("tossup_share")},
                         data={}, notes="every audit number is model-estimated", extra={**LIC_SA, "audit": a})
    return 0


def holdout_signoff_runs() -> list[str]:
    hits = []
    for path in glob.glob(str(registry.RUNS_DIR / "*_m7a_signoff_*.json")):
        if json.loads(Path(path).read_text()).get("holdout"):
            hits.append(Path(path).name)
    return hits


def key_numbers(r: dict) -> dict:
    """The numbers the dry run must reproduce exactly."""
    out = {"wp_brier": r["wp"]["models"]["m7a_wp"]["brier"], "wp_ece": r["wp"]["models"]["m7a_wp"]["cal"]["ece"],
           "wp_gap": r["wp"]["diffs"]["m7a_wp-nflfastr_wp"]["brier"]["diff"],
           "wp_gap_ci_high": r["wp"]["diffs"]["m7a_wp-nflfastr_wp"]["brier"]["ci_high"],
           "fg_brier": r["fg"]["brier"], "fg_ece": r["fg"]["cal"]["ece"], "punt_crps": r["punt"]["crps"],
           "punt_ece_pin": r["punt"]["cal_pin"]["ece"], "punt_ece_long": r["punt"]["cal_long"]["ece"]}
    if "conversion" in r:
        out.update({"conv_ece": r["conversion"]["engine_on_attempts"]["cal"]["ece"],
                    "audit_lost": r["audit"]["wp_lost_total"], "audit_matched": r["audit"]["matched_share"],
                    "audit_tossup": r["audit"].get("tossup_share", 0.0)})
    return out


def stage_signoff(window: tuple[int, int], allow_holdout: bool, log_: bool) -> int:
    """The pre-registered sign-off (context/ml.md, "M7a holdout pre-registration"). On the dev window with no
    logging it is the dry run: it must reproduce dev.json's primary block exactly."""
    label = "holdout" if windows.touches_holdout(window) else "m7adev"
    dev_path = OUT / "dev.json"
    if not dev_path.exists():
        raise SystemExit("run `python scripts/ml_m7a.py dev` first")
    dev = json.loads(dev_path.read_text())
    B = int(dev["boot"])
    if label == "holdout":
        prior = holdout_signoff_runs()
        if prior:
            raise SystemExit(f"an M7a holdout sign-off is already recorded ({prior}); it is run once only")
        dry = OUT / "signoff_m7adev.json"
        rec = json.loads(dry.read_text()) if dry.exists() else {}
        if not rec.get("reproduced"):
            raise SystemExit("the dry run has not reproduced dev: run `python scripts/ml_m7a.py signoff-dry-run` first")
        now = registry.git_commit()
        if rec.get("git", {}).get("dirty") is not False or now.get("dirty") or rec["git"].get("sha") != now.get("sha"):
            raise SystemExit("the dry run must have reproduced dev on this exact commit with a clean tree "
                             f"(dry run {rec.get('git')}, now {now}); commit, then rerun signoff-dry-run")
    res, work = evaluate(window, allow_holdout, B)
    print_window(res["primary"])
    out = {"label": label, "window": list(window), "result": res, "rules": res["primary"]["rules"]}
    if label != "holdout":
        a, b = key_numbers(dev["primary"]), key_numbers(res["primary"])
        diffs = {k: abs(a[k] - b[k]) for k in a}
        out["reproduced"] = bool(max(diffs.values()) <= 1e-12)
        out["max_abs_diff"] = max(diffs.values())
        out["git"] = registry.git_commit()
        log(f"\nDry run vs dev.json: max |diff| {out['max_abs_diff']:.3g} -> "
            f"{'REPRODUCED' if out['reproduced'] else 'NOT REPRODUCED'}")
    save_frames(work, label)
    (OUT / f"signoff_{label}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    if log_:
        log_runs(res, work, label, "m7a_signoff", signoff="m7a_signoff")
    log(f"\nM7a PRIMARY: {'PASS' if res['primary']['rules']['pass'] else 'FAIL'} "
        f"(WP {'pass' if res['primary']['rules']['wp_pass'] else 'fail'}, components "
        f"{'pass' if res['primary']['rules']['components_pass'] else 'fail'}; the audit is published either way)")
    return 0


# --------------------------------------------------------------------------- leakage

OUTCOME_COLS = ("field_goal_result", "touchdown", "td_team", "return_touchdown", "fourth_down_converted",
                "own_kickoff_recovery", "desc")
STATE_COLS = ("down", "ydstogo", "yardline_100", "score_differential", "posteam", "defteam", "play_type",
              "game_seconds_remaining", "half_seconds_remaining", "posteam_timeouts_remaining",
              "defteam_timeouts_remaining", "kicker_player_id", "punter_player_id", "pass", "roof", "temp", "wind")


def corrupt_m7a(pbp: pd.DataFrame, S: int, w: int, seed: int = 0) -> pd.DataFrame:
    """Season S from week w on: every outcome of week >= w scrambled; weeks after w also get their pre-snap
    states (yard line, score, down, teams, play type, kickers, ...) shuffled. Week-w states stay (they are
    the decisions being valued). `result` keeps its sign-zero pattern (ties stay ties) so the row set is stable."""
    rng = np.random.default_rng(seed)
    p = pbp.copy()
    late = ((p["season"] == S) & (p["week"] > w)).to_numpy()
    cur = ((p["season"] == S) & (p["week"] >= w)).to_numpy()
    for c in OUTCOME_COLS:
        if c in p.columns:
            p[c] = p[c].astype(object)
            p.loc[cur, c] = rng.permutation(p.loc[cur, c].to_numpy(object))
    if "result" in p.columns:
        r = p.loc[cur, "result"].to_numpy(float)
        p["result"] = p["result"].astype(float)
        p.loc[cur, "result"] = np.where(r == 0, 0.0, rng.choice([-1, 1], len(r)) * rng.integers(1, 30, len(r)))
    for c in STATE_COLS:
        if c in p.columns:
            p[c] = p[c].astype(object)
            p.loc[late, c] = rng.permutation(p.loc[late, c].to_numpy(object))
    for c in ("down", "ydstogo", "yardline_100", "score_differential", "game_seconds_remaining",
              "half_seconds_remaining", "posteam_timeouts_remaining", "defteam_timeouts_remaining", "pass", "temp",
              "wind", "touchdown", "return_touchdown", "fourth_down_converted", "own_kickoff_recovery"):
        if c in p.columns:
            p[c] = pd.to_numeric(p[c], errors="coerce")
    return p


VAL_COLS = ["wp_go", "wp_fg", "wp_punt", "p_conv", "p_fg", "kval", "pval", "ko_start"]


def stage_leakcheck(S: int = 2019, weeks=(1, 9, 17), seed: int = 0) -> int:
    """Real data, season S: corrupt S from week w on and confirm the week-w valuations are unchanged."""
    t_all = time.perf_counter()
    inp = load_inputs(S)
    games, pbp3, sched = ml_m3.load_inputs(last_season=S)
    sched = asof_mod.add_asof(sched) if "as_of" not in sched.columns else sched
    tab = oa.rating_table(pbp3, sched, oa.TUNED)
    comp, tables = fd.fit_components(S, inp.pbp, inp.wp_table, inp.m6_table)
    dec = fd.decision_table(inp.pbp, inp.a4s, seasons=[S])
    out = {"season": S, "weeks": {}, "license": LIC_SA}
    ok_all = True
    for w in weeks:
        t0 = time.perf_counter()
        wk = sched[(sched["season"] == S) & (sched["week"] == w)]
        as_of = wk["as_of"].min()
        wk_games = wk["game_id"].tolist()
        # (1) A4s as of the week: pbp and schedule from as_of on, games.csv results after week w
        p3c, s3c = asof_mod.corrupt_from(pbp3, sched, as_of, seed)
        gc = games.copy()
        lm = ((gc["season"] == S) & (gc["week"] > w)).to_numpy()
        rng = np.random.default_rng(seed)
        gc.loc[lm, "home_score"] = rng.integers(0, 50, int(lm.sum()))
        gc.loc[lm, "away_score"] = rng.integers(0, 50, int(lm.sum()))
        with no_rating_cache():
            a4_c = a4s_pregame(S, gc, p3c, s3c)
            a4_pc = a4s_pregame(S, gc, p3c, s3c, through=S)        # positive control: trained through S
        a4_diff = float(np.abs(a4_c.reindex(wk_games) - inp.a4s.reindex(wk_games)).max())
        a4_ctrl = float(np.abs(a4_pc.reindex(wk_games) - inp.a4s.reindex(wk_games)).max())
        # (2) M3 pass/rush ratings as of the week (the M6 rating inputs)
        r_clean = oa.compute_ratings(tab, sched, [(S, w)], oa.TUNED, ("pass", "rush"))
        r_c = oa.compute_ratings(oa.rating_table(p3c, s3c, oa.TUNED), s3c, [(S, w)], oa.TUNED, ("pass", "rush"))
        rat_diff = float(np.abs(r_clean[["off", "def"]].to_numpy() - r_c[["off", "def"]].to_numpy()).max())
        # (3) every M7a component refit from corrupted play-by-play; week-w valuations
        pbp_c = corrupt_m7a(inp.pbp, S, w, seed)
        wp_c = wpm.state_table(pbp_c[pbp_c["season"] >= FIRST_WP], a4_c)
        m6_c = inp.m6_table.copy()
        mm = ((m6_c["season"] == S) & (m6_c["week"] >= w)).to_numpy()
        rng = np.random.default_rng(seed + 1)
        for c in ("bin", "yards", "ev_first", "ev_20", "ev_loss", "turnover", "td"):
            m6_c.loc[mm, c] = rng.permutation(m6_c.loc[mm, c].to_numpy())
        comp_c, _ = fd.fit_components(S, pbp_c, wp_c, m6_c)
        rats = pd.concat([inp.ratings[~((inp.ratings["season"] == S) & (inp.ratings["week"] == w))], r_c])
        rats_clean = pd.concat([inp.ratings[~((inp.ratings["season"] == S) & (inp.ratings["week"] == w))], r_clean])
        d0 = dec[dec["week"] == w].reset_index(drop=True)
        v0, _ = fd.value_season(comp, d0, rats_clean)
        d1 = fd.decision_table(pbp_c, a4_c, seasons=[S])
        d1 = d1[d1["week"] == w].reset_index(drop=True)
        same_rows = d0[["game_id", "play_id"]].equals(d1[["game_id", "play_id"]])
        v1, _ = fd.value_season(comp_c, d1, rats)
        diff = float(np.abs(v0[VAL_COLS].to_numpy(float) - v1[VAL_COLS].to_numpy(float)).max()) if same_rows else np.inf
        # positive controls: a WP model trained through S on the corrupted labels; FG and punt models
        # whose player values also read the same game (allow_exact_matches) are covered synthetically in tests
        wp_pc = wpm.fit_final(wp_c[wp_c["season"].between(FIRST_WP, S)], comp_c.wp.info["n_iter"], None)
        v_pc = fd.option_values(comp_c, d1, fd.go_inputs(d1, comp_c, rats), wp=wp_pc)
        ctrl = float(np.abs(v1[["wp_go", "wp_fg", "wp_punt"]].to_numpy(float)
                            - v_pc[["wp_go", "wp_fg", "wp_punt"]].to_numpy(float)).max())
        chg = float((pbp_c.loc[(pbp_c["season"] == S) & (pbp_c["week"] >= w), "result"].to_numpy(float)
                     != inp.pbp.loc[(inp.pbp["season"] == S) & (inp.pbp["week"] >= w), "result"].to_numpy(float)).mean())
        ok = bool(a4_diff == 0 and rat_diff == 0 and diff == 0 and a4_ctrl > 0 and ctrl > 0)
        ok_all &= ok
        out["weeks"][int(w)] = {"games": len(wk_games), "decisions": int(len(d0)), "same_rows": bool(same_rows),
                                "a4s_max_diff": a4_diff, "a4s_control_max_diff": a4_ctrl,
                                "ratings_max_diff": rat_diff, "valuation_max_diff": diff,
                                "wp_control_max_diff": ctrl, "result_rows_changed": chg, "leak_free": ok,
                                "seconds": time.perf_counter() - t0}
        log(f"  week {w}: {len(d0)} decisions; A4s diff {a4_diff:.3g} (control {a4_ctrl:.3g}); ratings diff "
            f"{rat_diff:.3g}; valuations diff {diff:.3g} (WP-through-S control {ctrl:.3g}); "
            f"{'LEAK-FREE' if ok else 'LEAK'} ({time.perf_counter() - t0:.0f}s)")
    out["leak_free"] = ok_all
    out["seconds"] = time.perf_counter() - t_all
    write_license()
    (OUT / "leakcheck.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    log(f"leakcheck: {'PASS' if ok_all else 'FAIL'} ({out['seconds']:.0f}s)")
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["dev", "audit", "leakcheck", "signoff-dry-run", "signoff"])
    ap.add_argument("--boot", type=int, default=B_DEFAULT)
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    if a.stage == "dev":
        return stage_dev(not a.no_log, a.boot)
    if a.stage == "audit":
        return stage_audit(log_runs_=False)        # `dev` already logs m7a_audit; this stage prints and saves
    if a.stage == "leakcheck":
        return stage_leakcheck()
    if a.stage == "signoff-dry-run":
        return stage_signoff(PRIMARY_DEV, allow_holdout=False, log_=False)
    if a.stage == "signoff":
        return stage_signoff(HOLDOUT, allow_holdout=True, log_=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
