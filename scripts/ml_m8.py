"""M8: game model v2 on DEV (context/ml-m8-method.md). Ladder E0-E6 against A4s and C2d, freeze, forward test.

    python scripts/ml_m8.py tune               # QB v2 (2000-2005), Kalman carry-over (2000-2005 weeks 1-4),
                                               #   margin/EPA blend weight (2001-2005, leave-one-season-out)
    python scripts/ml_m8.py dev [--no-log]     # the ladder on the 3,450 DEV games; logs m8_* runs (label dev)
    python scripts/ml_m8.py leakcheck          # real-data leakage check of every new builder (2014-2017 data)
    python scripts/ml_m8.py freeze             # fit the chosen model on REG 2001-2025, write experiments/live/m8_2026.json
    python scripts/ml_m8.py forward            # the ONE pre-registered forward test (after the 2026 REG season)

Every logistic step uses A4s's protocol (walk-forward, season S fit on REG 2001..S-1, weights
0.5 ** ((S - 1 - season) / 8), ties as two half rows, M3 rating knobs). The ladder:

    E0    A4s (elo_logit + adj_epa_margin + qb_delta_diff) and C2d (elo_logit + kf_margin + qb_delta_diff)
    E1    margin targets on the A4s and the C2d features: point margin, EPA margin, blend; P(win) by a normal
          CDF with fitted sigma ("normal") or a logistic on the predicted margin ("logit")
    E2    QB v2 in C2d: (a) opponent-adjusted QB value, (b) injury-exit cleaning, (ab) both
    E3    early season added to C2d: (a) offseason QB change x weeks played, (b) head-coach change x weeks
          played, (c) Kalman carry-over re-tuned for weeks 1-4 (replaces kf_margin), (abc) all three
    E4    context added to C2d: travel + time zones, body clock, rest (rest_diff, short week, bye), altitude,
          venue (indoor, turf, surface mismatch, dome team outdoors late); E4_all = every deployable one;
          E4w = weather (NOT deployable: nflverse fills it after the game; descriptive only)
    E5    the components that beat C2d on their own, combined (logistic, and the best E1 target if it helped)
    E6    stack of E5 (or C2d) + C2d + Elo + A4s; E6b = stack of A4s + C2d + Elo + the best margin model

Selection: the one-SE rule over every deployable step (A4s and C2d included): among steps within one
paired-bootstrap SE of the best DEV Brier, the simplest. Every CI is paired (2,000 reps, seed 20261003).
ECE passes if it is at or below the 95th percentile of a calibrated forecaster at this n, or at most 0.01.
Leak alarm: anything better than A4s by more than 0.006 Brier is investigated before it is reported.

Nothing here reads 2020-2025 except `freeze` (training rows only; no metric is computed on them).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nflelo import config  # noqa: E402
from nflelo import data as elo_data  # noqa: E402
from nflelo.ml import asof, live, registry  # noqa: E402
from nflelo.ml import data as mldata  # noqa: E402
from nflelo.ml.eval import baselines, bootstrap, calibration, metrics, windows  # noqa: E402
from nflelo.ml.features import context  # noqa: E402
from nflelo.ml.features import kalman as kf  # noqa: E402
from nflelo.ml.features import opponent_adjust as oa  # noqa: E402
from nflelo.ml.features import qb  # noqa: E402
from nflelo.ml.models import logistic, margin_target as mt, stack  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ml_m3  # noqa: E402

DEV = windows.DEV
TRAIN_START = logistic.TRAIN_START
WF_FIRST = stack.STACK_FIRST_SEASON          # every base model is predicted from 2003 so the stack can train
OUT_DIR = mldata.ML_DIR / "m8"
PBP_COLUMNS = ml_m3.PBP_COLUMNS
A4S_DEV_BRIER = 0.2151
C2D_DEV_BRIER = 0.2141
DEV_N, DEV_HASH = 3450, "52b8194df3a2ebf4"
LEAK_MARGIN = 0.006
ECE_FLOOR = 0.01
A4S = ["elo_logit", "adj_epa_margin", "qb_delta_diff"]
C2D = ["elo_logit", "kf_margin", "qb_delta_diff"]
TARGETS = ("margin", "epa", "blend")
FREEZE_SEASON = 2026
FREEZE_PATH = live.LIVE_DIR / f"m8_{FREEZE_SEASON}.json"
FORWARD_LEDGER = live.LIVE_DIR / f"{FREEZE_SEASON}_m8.csv"

# Frozen by `tune` (2000-2005 / 2001-2005 only; see context/ml.md, "M8 build notes").
BLEND_W = 0.7    # weight on the point margin in the blend target (the rest on the EPA margin in points)

# --------------------------------------------------------------------------- feature groups

E3_GROUPS = {"E3a": ["qb_new_diff", "qb_new_x_wk"], "E3b": ["coach_new_diff", "coach_new_x_wk"]}
E4_GROUPS = {"E4_travel": ["travel_diff", "tz_diff"], "E4_body": ["body_early_diff", "body_late_diff"],
             "E4_rest": ["rest_diff", "bye_diff"], "E4_alt": ["alt_diff"],
             "E4_venue": ["indoor", "turf", "surface_mm_diff", "dome_cold_diff"]}
# short_diff (home short week minus away short week) is left out: a Thursday game puts both teams on a short
# week, so it is nonzero in 6 of 4,600 training games (2010, 2011, 2016) and the first run's fits nearly
# separated on it (a coefficient of -21). Short weeks are reported as a split instead.
E4_WEATHER = ["wind15", "cold32", "elo_x_wind15"]
QB_COLS = {"a": "qb_delta_diff_v2a", "b": "qb_delta_diff_v2b", "ab": "qb_delta_diff_v2ab"}


def v2_config(v: str) -> qb.QBv2:
    return {"a": qb.TUNED_V2_A, "b": qb.TUNED_V2_B, "ab": qb.TUNED_V2_AB}[v]


# --------------------------------------------------------------------------- data and features

def load_inputs(last_season: int = DEV[1]):
    games = elo_data.load_games()
    games = games[games["season"] <= last_season].reset_index(drop=True)
    years = range(ml_m3.DATA_SEASONS[0], last_season + 1)
    pbp = mldata.load_pbp(years, columns=PBP_COLUMNS)
    sched = asof.add_asof(mldata.load_schedules(years))
    return games, pbp, sched


def build_frame(games, pbp, sched, last_season: int = DEV[1], full: bool = True) -> tuple[pd.DataFrame, dict]:
    """M3's frame plus kf_margin, the targets, and (with `full`) every M8 feature. REG 2001..last_season."""
    frame, t = ml_m3.build_frame(games, pbp, sched, last_season=last_season)
    t.pop("ratings_obj", None)
    reg = sched.set_index("game_id").loc[frame["game_id"]].reset_index()

    t0 = time.perf_counter()
    k = kf.build_features(pbp, sched, reg, kf.TUNED)
    frame["kf_margin"], frame["kf_sd"] = k["kf_margin"].to_numpy(), k["kf_sd"].to_numpy()
    t["kalman"] = time.perf_counter() - t0
    # targets (outcomes: read only for training seasons)
    frame["margin"] = (reg["home_score"].astype(float) - reg["away_score"].astype(float)).to_numpy()
    gt = kf.game_table(pbp, sched, kf.TUNED).set_index("game_id")
    frame["epa"] = kf.TUNED.scale * gt["epa_margin"].reindex(frame["game_id"]).to_numpy(float)
    if BLEND_W is not None:
        frame["blend"] = np.where(np.isfinite(frame["epa"]), BLEND_W * frame["margin"] + (1 - BLEND_W) * frame["epa"],
                                  np.nan)
    # splits
    last = qb.last_starters(sched, reg)
    actual = reg.set_index("game_id")[["home_qb_id", "away_qb_id"]]
    frame["changed"] = ((actual["home_qb_id"].astype(str).to_numpy() != last["home_qb_id"].astype(str).to_numpy())
                        | (actual["away_qb_id"].astype(str).to_numpy() != last["away_qb_id"].astype(str).to_numpy()))
    rest = reg[["home_rest", "away_rest"]].apply(pd.to_numeric, errors="coerce")
    frame["short_week"] = (rest.min(axis=1) <= 5).to_numpy()
    if not full:
        return frame, t

    t0 = time.perf_counter()
    cfg = qb.tuned_config()
    for v, col in QB_COLS.items():
        d = qb.build_features_v2(pbp, sched, reg, cfg, "actual", v2_config(v))
        frame[col] = (d["qb_delta_home"] - d["qb_delta_away"]).to_numpy()
    t["qb_v2"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    e = context.m8_early_features(None, sched, reg)
    for c in context.M8_EARLY:
        frame[c] = e[c].to_numpy()
    frame["qb_new_x_wk"] = frame["qb_new_diff"] * frame["weeks_played"] / 17.0
    frame["coach_new_x_wk"] = frame["coach_new_diff"] * frame["weeks_played"] / 17.0
    kc = kf.build_features(pbp, sched, reg, kf.TUNED_CARRYOVER)
    frame["kf_margin_co"] = kc["kf_margin"].to_numpy()
    t["early"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cx = context.m8_context_features(None, sched, reg, weather=True)
    for c in context.M8_CONTEXT + context.M8_WEATHER:
        frame[c] = cx[c].to_numpy()
    frame["elo_x_wind15"] = frame["elo_logit"] * frame["wind15"]
    t["context"] = time.perf_counter() - t0
    feats = [c for c in frame.columns if c not in ("game_id", "season", "week", "home_team", "away_team", "y",
                                                   "margin", "epa", "blend", "changed", "short_week")]
    if frame[feats].isna().any().any():
        raise SystemExit(f"NaN in features: {frame[feats].isna().sum()[lambda s: s > 0].to_dict()}")
    return frame, t


def prepare(window=DEV):
    """Inputs, the full feature frame, and the scored-game mask (DEV REG games with moneylines)."""
    windows.check(window, allow_holdout=False)
    games, pbp, sched = load_inputs(last_season=window[1])
    base = baselines.baseline_frame(games, window, require_market=True)
    t0 = time.perf_counter()
    frame, ft = build_frame(games, pbp, sched, last_season=window[1])
    ft["total_features"] = time.perf_counter() - t0
    frame = frame.merge(base[["game_id", "p_elo_v2", baselines.MARKET_COL]], on="game_id", how="left")
    mask = frame["game_id"].isin(base["game_id"]).to_numpy()
    if mask.sum() != len(base):
        raise SystemExit(f"only {mask.sum()} of {len(base)} scored games found in the feature frame")
    sub = frame[mask]
    if not np.allclose(sub["y"].to_numpy(), base.set_index("game_id").loc[sub["game_id"], "y"].to_numpy()):
        raise SystemExit("outcomes disagree between the schedule and data/games.csv")
    return frame, mask, ft


# --------------------------------------------------------------------------- specs and prediction

def spec(step, kind="logistic", features=(), target=None, method=None, bases=(), group="", deployable=True,
         note=""):
    return {"step": step, "kind": kind, "features": list(features), "target": target, "method": method,
            "bases": list(bases), "group": group, "deployable": deployable, "note": note}


def complexity(sp: dict, specs: dict) -> int:
    if sp["kind"] == "logistic":
        return len(sp["features"])
    if sp["kind"] == "margin":
        return len(sp["features"]) + 1
    if sp["kind"] == "elo":
        return 1
    return sum(complexity(specs[b], specs) for b in sp["bases"]) + len(sp["bases"])


def predict(frame: pd.DataFrame, sp: dict, window, preds: dict) -> tuple[pd.Series, pd.DataFrame]:
    """Walk-forward predictions of one step over `window` (NaN outside it) and its per-season fits."""
    if sp["kind"] == "logistic":
        return logistic.walk_forward(frame, sp["features"], window, train_start=TRAIN_START)
    if sp["kind"] == "margin":
        return mt.walk_forward(frame, sp["features"], window, sp["target"], sp["method"], train_start=TRAIN_START)
    if sp["kind"] == "elo":
        p = pd.Series(1.0 / (1.0 + np.exp(-frame["elo_logit"].to_numpy(float))), index=frame.index)
        p[~frame["season"].between(*window)] = np.nan
        return p, pd.DataFrame()
    if sp["kind"] == "stack":
        base = pd.DataFrame({b: preds[b] for b in sp["bases"]})
        return stack.walk_forward(base, frame["y"].to_numpy(float), frame["season"].to_numpy(int),
                                  (max(window[0], WF_FIRST + 1), window[1]), first_season=WF_FIRST)
    raise ValueError(sp["kind"])


# --------------------------------------------------------------------------- scoring helpers

def score(y, p) -> dict:
    m = {**metrics.win_metrics(y, p), "ece": calibration.ece(y, p)}
    m["ece_null_p95"] = calibration.ece_null_range(p)["p95"]
    m["ece_pass"] = bool(m["ece"] <= m["ece_null_p95"] or m["ece"] <= ECE_FLOOR)
    return m


def ci(y, a, b, metric="brier") -> dict:
    return bootstrap.paired_bootstrap(y, a, b, metric)


def fmt_ci(c) -> str:
    return f"{c['diff']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]"


def split_masks(sub: pd.DataFrame) -> dict:
    wk = sub["week"].to_numpy()
    ch = sub["changed"].to_numpy(bool)
    return {"weeks_1_4": wk <= 4, "weeks_5_9": (wk >= 5) & (wk <= 9), "weeks_10_18": wk >= 10,
            "changed": ch, "unchanged": ~ch, "short_week": sub["short_week"].to_numpy(bool)}


def splits(y, p, a4s, c2d, mkt, sub) -> dict:
    out = {}
    for name, m in split_masks(sub).items():
        if not m.any():
            continue
        out[name] = {"n": int(m.sum()), "brier": float(np.mean((y[m] - p[m]) ** 2)),
                     "brier_a4s": float(np.mean((y[m] - a4s[m]) ** 2)),
                     "brier_c2d": float(np.mean((y[m] - c2d[m]) ** 2)),
                     "brier_market": float(np.mean((y[m] - mkt[m]) ** 2)),
                     "vs_a4s": ci(y[m], p[m], a4s[m]), "vs_c2d": ci(y[m], p[m], c2d[m])}
    return out


# --------------------------------------------------------------------------- stage: tune

def _targets_with_opp(qg: pd.DataFrame, sched: pd.DataFrame, seasons) -> pd.DataFrame:
    from nflelo.teams import franchise
    tg = qb.starter_targets(qg, sched, seasons)
    info = sched.drop_duplicates("game_id").set_index("game_id")
    other = np.where(tg["side"] == "home", tg["game_id"].map(info["away_team"]), tg["game_id"].map(info["home_team"]))
    tg["opp"] = [franchise(c, int(s)) for c, s in zip(other, tg["season"])]
    neutral = tg["game_id"].map(info["location"]).eq("Neutral").to_numpy()
    tg["home"] = ((tg["side"] == "home").to_numpy() & ~neutral).astype(float)
    return tg


def tune_qb_v2(log: bool) -> dict:
    t0 = time.perf_counter()
    years = range(ml_m3.DATA_SEASONS[0], windows.TUNE[1] + 1)
    pbp = mldata.load_pbp(years, columns=PBP_COLUMNS)
    sched = asof.add_asof(mldata.load_schedules(years))
    cfg = qb.tuned_config()
    db = qb.dropback_plays(pbp, sched, cfg)
    qg = qb.qb_game_table(db, sched)
    tg = _targets_with_opp(qg, sched, windows.TUNE)
    keys = sorted(set(zip(tg["season"].astype(int), tg["week"].astype(int))))
    qg2 = qb.qb_game_table_v2(db, sched)
    priors = qb.replacement_priors(db, {s for s, _ in keys}, cfg.rookie_dropbacks)

    def run(v2: qb.QBv2, opp: bool) -> float:
        st, qbs = qb.week_states_v2(qg2, sched, keys, cfg, priors, v2)
        return qb.score_v2(st, qbs, tg, opp)["wmse"]
    base_m3 = qb.score_k(*_plain_states(qg, sched, cfg, keys, priors), tg, cfg.k)["wmse"]
    base = run(qb.QBv2(), False)
    print(f"== QB v2 tuning, {windows.TUNE[0]}-{windows.TUNE[1]}: {len(tg)} starter-games, k {cfg.k:g} ==")
    print(f"baseline (M3 value) wMSE {base_m3:.5f}; v2 code path with nothing on {base:.5f} (must match)")
    if abs(base - base_m3) > 1e-12:
        raise SystemExit("QB v2 with nothing switched on does not reproduce the M3 value")
    grid = []
    for lam in (25, 50, 100, 200, 400, 800, 1600, 3200):
        v2 = qb.QBv2(lam_d=float(lam))
        st, qbs = qb.week_states_v2(qg2, sched, keys, cfg, priors, v2)
        grid.append({"lam_d": float(lam), "wmse_opp": qb.score_v2(st, qbs, tg, True)["wmse"],
                     "wmse_q_only": qb.score_v2(st, qbs, tg, False)["wmse"]})
    g = pd.DataFrame(grid)
    best_lam = float(g.loc[g["wmse_opp"].idxmin(), "lam_d"])
    print("(a) opponent adjustment (prediction Q + D[opponent] + h*home; and Q alone):\n" + g.to_string(index=False))
    ex = []
    for ew in (0.0, 0.25, 0.5, 0.75, 1.0):
        ex.append({"exit_weight": ew, "wmse_b": run(qb.QBv2(exit_weight=ew), False),
                   "wmse_ab": run(qb.QBv2(lam_d=best_lam, exit_weight=ew), True)})
    e = pd.DataFrame(ex)
    ew_b = float(e.loc[e["wmse_b"].idxmin(), "exit_weight"])
    ew_ab = float(e.loc[e["wmse_ab"].idxmin(), "exit_weight"])
    part = qb.partial_flags(qg2[qg2["season"].between(*windows.TUNE)], qb.QBv2(exit_weight=0.5))
    print(f"(b) injury-exit cleaning (partial QB-games: {part.mean():.1%} of QB-game rows, 2000-2005):\n"
          + e.to_string(index=False))
    out = {"n_targets": int(len(tg)), "base_wmse": base_m3, "lam_grid": grid, "lam_d": best_lam,
           "exit_grid": ex, "exit_weight_b": ew_b, "exit_weight_ab": ew_ab, "partial_share": float(part.mean()),
           "seconds": time.perf_counter() - t0}
    print(f"chosen: lam_d {best_lam:g}; exit_weight {ew_b:g} (b alone), {ew_ab:g} (with a). {out['seconds']:.1f}s")
    if log:
        registry.log_run("m8_tune_qb_v2", label="tune", seasons=windows.TUNE, game_ids=tg["game_id"].unique(),
                         metrics={"n": int(len(tg)), "wmse_base": base_m3,
                                  "wmse_a": float(g["wmse_opp"].min()), "wmse_b": float(e["wmse_b"].min()),
                                  "wmse_ab": float(e["wmse_ab"].min())},
                         params={"qb": cfg.to_dict(), "exit_share": 0.5, "exit_after": 5, "window_seasons": 8},
                         data=registry.data_fingerprint(games_csv=False, ml_datasets=("pbp", "schedules"),
                                                        seasons=years),
                         notes="M8 QB v2: defense-effect ridge strength and partial-game weight, tuned on 2000-2005 "
                               "next-game starter EPA per dropback (dropback-weighted MSE).", extra={"results": out})
    return out


def _plain_states(qg, sched, cfg, keys, priors):
    return qb.week_states(qg, sched, keys, cfg, priors)


def tune_carry(log: bool) -> dict:
    years = range(kf.BURN_IN_FIRST, kf.TUNE_SEASONS[1] + 1)
    pbp = mldata.load_pbp(years, columns=PBP_COLUMNS)
    sched = asof.add_asof(mldata.load_schedules(years))
    gt = kf.game_table(pbp, sched, kf.TUNED)
    r = kf.tune_carryover(gt, kf.TUNED)
    c = r["config"]
    print(f"== Kalman carry-over, REG weeks 1-4 of {kf.TUNE_SEASONS[0]}-{kf.TUNE_SEASONS[1]} ({r['n_games']} games) ==")
    print(f"TUNED gamma {kf.TUNED.gamma:.4f}, q_season {kf.TUNED.q_season:.3f}: -loglik {r['nll_base']:.2f}")
    print(f"early-season optimum gamma {c.gamma:.4f}, q_season {c.q_season:.3f}: -loglik {r['nll']:.2f} "
          f"({r['evaluations']} evaluations, {r['seconds']:.1f}s)")
    print("gamma profile (q_season fixed): " + ", ".join(f"{g['gamma']:.1f} {g['nll']:.2f}" for g in r["gamma_profile"]))
    if log:
        registry.log_run("m8_tune_carryover", label="tune", seasons=kf.TUNE_SEASONS,
                         game_ids=gt.loc[gt["season"].between(*kf.TUNE_SEASONS) & gt["week"].between(1, 4), "game_id"],
                         metrics={"n": r["n_games"], "neg_loglik": r["nll"], "neg_loglik_tuned": r["nll_base"]},
                         params={"base": kf.TUNED.to_dict(), "chosen": c.to_dict()},
                         data=registry.data_fingerprint(games_csv=False, ml_datasets=("pbp", "schedules"),
                                                        seasons=years),
                         notes="M8 E3c: Kalman between-season carry-over (gamma, q_season) by joint likelihood of "
                               "REG weeks 1-4, 2000-2005 (1999 burn-in); other knobs fixed at kalman.TUNED.",
                         extra={"gamma_profile": r["gamma_profile"]})
    return {k: v for k, v in r.items() if k != "config"} | {"config": c.to_dict()}


def tune_blend(log: bool) -> dict:
    """Blend weight w on the point margin, by leave-one-season-out log loss on REG 2001-2005 (A4s features)."""
    t0 = time.perf_counter()
    games, pbp, sched = load_inputs(last_season=windows.TUNE[1])
    frame, _ = build_frame(games, pbp, sched, last_season=windows.TUNE[1], full=False)
    frame = frame[frame["season"].between(TRAIN_START, windows.TUNE[1])].reset_index(drop=True)
    X = frame[A4S].to_numpy(float)
    y, marg, epa = (frame[c].to_numpy(float) for c in ("y", "margin", "epa"))
    season = frame["season"].to_numpy(int)
    grid = []
    for w in np.round(np.arange(0, 1.0001, 0.1), 2):
        t = np.where(np.isfinite(epa), w * marg + (1 - w) * epa, np.nan)
        ll = np.empty(len(y))
        for S in np.unique(season):
            tr, te = season != S, season == S
            f = mt.fit(X[tr], t[tr], marg[tr], y[tr], np.ones(tr.sum()))
            ll[te] = metrics.logloss_per_game(y[te], f.predict_proba(X[te], "normal"))
        grid.append({"w": float(w), "logloss": float(ll.mean()),
                     "corr_target_margin": float(np.corrcoef(t[np.isfinite(t)], marg[np.isfinite(t)])[0, 1])})
    g = pd.DataFrame(grid)
    w_best = float(g.loc[g["logloss"].idxmin(), "w"])
    print(f"== Blend weight, leave-one-season-out over REG {TRAIN_START}-{windows.TUNE[1]} (n {len(y)}), A4s features ==")
    print(g.to_string(index=False))
    print(f"chosen w {w_best:g} ({time.perf_counter() - t0:.1f}s)")
    if log:
        registry.log_run("m8_tune_blend", label="tune", seasons=(TRAIN_START, windows.TUNE[1]),
                         game_ids=frame["game_id"], metrics={"n": int(len(y)), "logloss": float(g["logloss"].min())},
                         params={"features": A4S, "method": "normal", "epa_scale": kf.TUNED.scale},
                         data=registry.data_fingerprint(games_csv=True, ml_datasets=("pbp", "schedules"),
                                                        seasons=range(ml_m3.DATA_SEASONS[0], windows.TUNE[1] + 1)),
                         notes="M8 E1: blend target w * margin + (1 - w) * EPA margin in points; w by "
                               "leave-one-season-out log loss on 2001-2005.", extra={"grid": grid})
    return {"grid": grid, "w": w_best}


def stage_tune(log: bool) -> int:
    q = tune_qb_v2(log)
    c = tune_carry(log)
    b = tune_blend(log)
    print("\nFreeze in code:")
    print(f"  qb.TUNED_V2_A = QBv2(lam_d={q['lam_d']:g})")
    print(f"  qb.TUNED_V2_B = QBv2(exit_weight={q['exit_weight_b']:g})")
    print(f"  qb.TUNED_V2_AB = QBv2(lam_d={q['lam_d']:g}, exit_weight={q['exit_weight_ab']:g})")
    print(f"  kalman.TUNED_CARRYOVER = replace(TUNED, gamma={c['config']['gamma']:.5g}, "
          f"q_season={c['config']['q_season']:.5g})")
    print(f"  ml_m8.BLEND_W = {b['w']:g}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tune.json").write_text(json.dumps({"qb_v2": q, "carryover": c, "blend": b}, indent=2, default=float))
    return 0


# --------------------------------------------------------------------------- stage: dev ladder

def ladder_specs() -> list[dict]:
    out = [spec("A4s", features=A4S, group="E0", note="M3 chosen model (reference)"),
           spec("C2d", features=C2D, group="E0", note="M3b one-SE pick (reference)"),
           spec("Elo", kind="elo", group="E0", note="Elo log-odds (elo_logit), a stack base only", deployable=False)]
    for base_name, feats in (("A4s", A4S), ("C2d", C2D)):
        for t in TARGETS:
            for m in mt.METHODS:
                out.append(spec(f"E1_{base_name}_{t}_{m}", "margin", feats, target=t, method=m, group="E1",
                                note=f"{t} target on the {base_name} features, P(win) by {m}"))
    for v, col in QB_COLS.items():
        out.append(spec(f"E2{v}", features=["elo_logit", "kf_margin", col], group="E2",
                        note={"a": "opponent-adjusted QB value", "b": "injury-exit cleaning", "ab": "both"}[v]))
    for name, cols in E3_GROUPS.items():
        out.append(spec(name, features=C2D + cols, group="E3",
                        note={"E3a": "offseason QB change x weeks played",
                              "E3b": "head-coach change x weeks played"}[name]))
    out.append(spec("E3c", features=["elo_logit", "kf_margin_co", "qb_delta_diff"], group="E3",
                    note="Kalman carry-over re-tuned on weeks 1-4 of 2000-2005 (replaces kf_margin)"))
    out.append(spec("E3abc", features=["elo_logit", "kf_margin_co", "qb_delta_diff"] + sum(E3_GROUPS.values(), []),
                    group="E3", note="E3a + E3b + E3c"))
    for name, cols in E4_GROUPS.items():
        out.append(spec(name, features=C2D + cols, group="E4", note="context: " + ", ".join(cols)))
    out.append(spec("E4_all", features=C2D + sum(E4_GROUPS.values(), []), group="E4",
                    note="every deployable context feature"))
    out.append(spec("E4w", features=C2D + E4_WEATHER, group="E4", deployable=False,
                    note="weather: NOT deployable (nflverse fills temp and wind after the game); descriptive only"))
    return out


def stage_dev(log: bool) -> int:
    t_start = time.perf_counter()
    frame, mask, ft = prepare(DEV)
    sub = frame[mask]
    y = sub["y"].to_numpy()
    mkt = sub[baselines.MARKET_COL].to_numpy()
    gh = registry.games_hash(sub["game_id"])
    print(f"DEV {DEV[0]}-{DEV[1]}: n = {len(sub)}, games hash {gh}. Features {ft['total_features']:.1f}s ("
          + ", ".join(f"{k} {v:.1f}s" for k, v in ft.items() if k != "total_features") + ")")
    if len(sub) != DEV_N or gh != DEV_HASH:
        print(f"DEV game set differs (expected n {DEV_N}, hash {DEV_HASH}). Stopping.", file=sys.stderr)
        return 1

    t_ladder = time.perf_counter()
    window = (WF_FIRST, DEV[1])
    preds, coefs, specs = {}, {}, {}

    def run(sp):
        p, c = predict(frame, sp, window, preds)
        preds[sp["step"]], coefs[sp["step"]], specs[sp["step"]] = p, c, sp
        return p[mask].to_numpy()

    def brier(s):
        return float(np.mean((y - preds[s][mask].to_numpy()) ** 2))

    for sp in ladder_specs():
        run(sp)
    b_a4s, b_c2d = brier("A4s"), brier("C2d")
    print(f"A4s reproduction: {b_a4s:.5f} (M3 {A4S_DEV_BRIER}); C2d {b_c2d:.5f} (M3b {C2D_DEV_BRIER})")
    if abs(b_a4s - A4S_DEV_BRIER) > 0.00005 or abs(b_c2d - C2D_DEV_BRIER) > 0.00005:
        print("A4s or C2d does not reproduce. Stopping.", file=sys.stderr)
        return 1

    # ---- E5: the components that beat C2d on their own
    comps = {}
    feats = list(C2D)
    qbest = min(("E2a", "E2b", "E2ab"), key=brier)
    if brier(qbest) < b_c2d:
        comps["qb"] = qbest
        feats = [specs[qbest]["features"][2] if f == "qb_delta_diff" else f for f in feats]
    if brier("E3c") < b_c2d:
        comps["carryover"] = "E3c"
        feats = ["kf_margin_co" if f == "kf_margin" else f for f in feats]
    for name in list(E3_GROUPS) + list(E4_GROUPS):
        if brier(name) < b_c2d:
            comps[name] = name
            feats += (E3_GROUPS | E4_GROUPS)[name]
    e1c = min((s for s in specs if s.startswith("E1_C2d_")), key=brier)
    e1_helps = brier(e1c) < b_c2d
    e5 = None
    if len(feats) > 3 or feats != C2D:
        e5 = "E5"
        run(spec("E5", features=feats, group="E5", note="combination: " + " + ".join(comps)))
        if e1_helps:
            run(spec("E5m", "margin", feats, target=specs[e1c]["target"], method=specs[e1c]["method"], group="E5",
                     note=f"E5 features with the {e1c} target and method"))
    elif e1_helps:
        e5 = e1c
    print(f"Components that beat C2d on their own: {comps or 'none'}; best E1 on C2d features {e1c} "
          f"({'beats' if e1_helps else 'does not beat'} C2d)")

    # ---- E6: stacks
    e1all = min((s for s in specs if s.startswith("E1_")), key=brier)
    bases6 = list(dict.fromkeys(([e5] if e5 else []) + ["C2d", "Elo", "A4s"]))
    run(spec("E6", "stack", bases=bases6, group="E6", note="stack of " + " + ".join(bases6)))
    bases6b = ["A4s", "C2d", "Elo", e1all]
    run(spec("E6b", "stack", bases=bases6b, group="E6", note="stack of " + " + ".join(bases6b)))
    t_ladder = time.perf_counter() - t_ladder

    # ---- scores, CIs, one-SE rule
    a4s, c2d = preds["A4s"][mask].to_numpy(), preds["C2d"][mask].to_numpy()
    steps = [s for s in specs if specs[s]["kind"] != "elo"]
    res = {}
    for s in steps:
        p = preds[s][mask].to_numpy()
        m = score(y, p)
        m["vs_a4s"], m["vs_c2d"] = ci(y, p, a4s), ci(y, p, c2d)
        m["vs_a4s_logloss"], m["vs_c2d_logloss"] = ci(y, p, a4s, "logloss"), ci(y, p, c2d, "logloss")
        m["vs_market"] = ci(y, p, mkt)
        res[s] = m
    cand = [s for s in steps if specs[s]["deployable"]]
    best = min(cand, key=lambda s: res[s]["brier"])
    pb = preds[best][mask].to_numpy()
    for s in cand:
        c = ci(y, preds[s][mask].to_numpy(), pb)
        res[s]["vs_best"] = c
        res[s]["within_1se"] = True if s == best else bool(c["diff"] <= c["boot_se"])
    eligible = [s for s in cand if res[s]["within_1se"]]
    chosen = min(eligible, key=lambda s: (complexity(specs[s], specs), res[s]["brier"]))
    alarm = [s for s in steps if res[s]["vs_a4s"]["diff"] < -LEAK_MARGIN]

    print(f"\nLadder, DEV {DEV[0]}-{DEV[1]}, n = {len(y)}. Paired bootstrap (2,000 reps, seed 20261003); ECE ok = "
          f"<= p95 of a calibrated forecaster at this n, or <= {ECE_FLOOR}.")
    print(f"{'step':<22}{'brier':>8}{'logloss':>9}{'acc':>7}{'ece':>7}{'p95':>7}{'ok':>3}  "
          f"{'vs A4s [95% CI]':<27}{'vs C2d [95% CI]':<27}{'1SE':>4}")
    for s in steps:
        m = res[s]
        flag = "" if not specs[s]["deployable"] else ("yes" if res[s]["within_1se"] else "")
        print(f"{s:<22}{m['brier']:>8.4f}{m['logloss']:>9.4f}{m['accuracy']:>7.3f}{m['ece']:>7.3f}"
              f"{m['ece_null_p95']:>7.3f}{'y' if m['ece_pass'] else 'N':>3}  {fmt_ci(m['vs_a4s']):<27}"
              f"{fmt_ci(m['vs_c2d']):<27}{flag:>4}" + (" *" if s == chosen else ""))
    me, mm = score(y, sub["p_elo_v2"].to_numpy()), score(y, mkt)
    print(f"{'Elo v2':<22}{me['brier']:>8.4f}{me['logloss']:>9.4f}{me['accuracy']:>7.3f}{me['ece']:>7.3f}")
    print(f"{'Market (benchmark)':<22}{mm['brier']:>8.4f}{mm['logloss']:>9.4f}{mm['accuracy']:>7.3f}{mm['ece']:>7.3f}")
    print(f"Leak alarm (better than A4s by more than {LEAK_MARGIN}): {alarm or 'none'}")
    print(f"Best: {best} ({res[best]['brier']:.5f}). Within one SE: {eligible}. CHOSEN (one-SE rule): {chosen}.")
    cm = res[chosen]
    accepted = bool(chosen != "A4s" and cm["vs_a4s"]["ci_high"] < 0 and cm["ece_pass"])
    print(f"DEV acceptance (beats A4s with CI excluding 0, ECE check): {'PASS' if accepted else 'FAIL'} "
          f"({chosen} minus A4s {fmt_ci(cm['vs_a4s'])}; minus C2d {fmt_ci(cm['vs_c2d'])}; ECE {cm['ece']:.4f})")

    # ---- splits
    main = list(dict.fromkeys(["C2d", e1c, e1all, "E2a", "E2b", "E2ab", "E3a", "E3b", "E3c", "E3abc", "E4_all", "E4w"]
                              + [s for s in ("E5", "E5m", "E6", "E6b") if s in specs] + [chosen]))
    split = {}
    print("\nSplits (Brier; minus A4s and minus C2d [95% CI]; A4s / C2d / market Brier on the same games):")
    for s in main:
        split[s] = splits(y, preds[s][mask].to_numpy(), a4s, c2d, mkt, sub)
        print(f"  {s}:")
        for k_, v in split[s].items():
            print(f"    {k_:<12} n{v['n']:>5} {v['brier']:.4f}  vsA4s {fmt_ci(v['vs_a4s'])}  vsC2d {fmt_ci(v['vs_c2d'])}"
                  f"  (A4s {v['brier_a4s']:.4f}, C2d {v['brier_c2d']:.4f}, mkt {v['brier_market']:.4f})")

    # ---- coefficients
    print("\nCoefficients (last DEV refit; range over the DEV refits):")
    for s in dict.fromkeys(["C2d", e1c, "E2a", "E3a", "E3b", "E4_all"] + [x for x in ("E5", "E5m", "E6", "E6b")
                                                                         if x in specs] + [chosen]):
        c = coefs[s]
        if c is None or c.empty:
            continue
        c = c[c["season"].between(*DEV)]
        last = c.iloc[-1]
        cols = [x for x in c.columns if x not in ("season", "n_train")]
        print(f"  {s} ({int(last['season'])} fit, n {int(last['n_train'])}): " + ", ".join(
            f"{x} {last[x]:+.4g} [{c[x].min():+.3g}, {c[x].max():+.3g}]" for x in cols))
    timing = {"features_s": ft, "ladder_s": t_ladder, "total_s": time.perf_counter() - t_start}
    print(f"\nTiming: features {ft['total_features']:.1f}s, ladder {t_ladder:.1f}s, total {timing['total_s']:.1f}s")

    # ---- save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    for s, p in preds.items():
        out[f"p_{s}"] = p.to_numpy()
    out["dev"] = mask
    out.to_parquet(OUT_DIR / "ladder_frame.parquet", index=False)
    pd.concat([c.assign(step=s) for s, c in coefs.items() if c is not None and not c.empty],
              ignore_index=True).to_parquet(OUT_DIR / "coefs.parquet")
    summary = {"games_hash": gh, "n": int(len(y)), "chosen": chosen, "best": best, "eligible": eligible,
               "accepted": accepted, "components": comps, "e1_best_c2d": e1c, "e1_best": e1all, "leak_alarm": alarm,
               "specs": specs, "chosen_spec": specs[chosen], "brier": {s: res[s]["brier"] for s in res},
               "timing": timing, "blend_w": BLEND_W}
    (OUT_DIR / "dev.json").write_text(json.dumps(summary, indent=2, default=float))

    if log:
        fp = registry.data_fingerprint(games_csv=True, ml_datasets=("pbp", "schedules"),
                                       seasons=range(ml_m3.DATA_SEASONS[0], DEV[1] + 1))
        fp["data/sources/stadiums.csv"] = registry.sha256_path(context.STADIUMS_CSV)
        common = {"ratings": oa.TUNED.to_dict(), "qb": qb.tuned_config().to_dict(), "kalman": kf.TUNED.to_dict(),
                  "train_start": TRAIN_START, "season_half_life": logistic.SEASON_HALF_LIFE,
                  "ties": "two half-weight rows", "elo_config": config.DEFAULT_CONFIG.to_dict()}
        for s in res:
            sp, m = specs[s], res[s]
            params = dict(common, model=sp["kind"], target=sp["target"], method=sp["method"], bases=sp["bases"])
            if any(c in sp["features"] for c in QB_COLS.values()):
                params["qb_v2"] = {v: v2_config(v).to_dict() for v in QB_COLS}
            if "kf_margin_co" in sp["features"]:
                params["kalman_carryover"] = kf.TUNED_CARRYOVER.to_dict()
            if sp["target"] == "blend":
                params["blend_w"] = BLEND_W
            if sp["kind"] == "stack":
                params["stack"] = {"lam": stack.STACK_LAM, "first_season": WF_FIRST}
            registry.log_run(
                f"m8_{s}", label="dev", seasons=DEV, game_ids=sub["game_id"],
                metrics={k: m[k] for k in ("n", "brier", "logloss", "accuracy", "ece", "ece_null_p95", "ece_pass")},
                features=sp["features"] or sp["bases"], params=params, data=fp,
                notes=(sp["note"] + (" CHOSEN by the one-SE rule." if s == chosen else "")).strip(),
                extra={"group": sp["group"], "deployable": sp["deployable"],
                       "comparisons": {k: m[k] for k in ("vs_a4s", "vs_c2d", "vs_a4s_logloss", "vs_c2d_logloss",
                                                         "vs_market", "vs_best") if k in m},
                       "one_se": {"best": best, "within": m.get("within_1se"), "chosen": s == chosen,
                                  "eligible": eligible, "complexity": complexity(sp, specs)},
                       "accepted": accepted if s == chosen else None, "splits": split.get(s),
                       "coefficients": coefs[s][coefs[s]["season"].between(*DEV)].to_dict("records")
                       if coefs[s] is not None and not coefs[s].empty else None,
                       "timing": timing if s == chosen else None})
        print(f"\n{len(res)} runs written to {registry.RUNS_DIR.relative_to(config.ROOT)}/")
    return 0


# --------------------------------------------------------------------------- stage: leakcheck (real data)

LEAK_SEASONS = (2014, 2017)


def stage_leakcheck() -> int:
    """Rebuild each new feature after scrambling everything from the game's as_of on (`asof.leakage_check`).

    Real data 2014-2017. Builders: QB v2 (a, b, ab) with the actual starter (identity kept, M3-D1) and with
    the last starter (identities scrambled too); the Kalman carry-over filter; the early-season features
    (the game's own starter and head coach are pre-game facts); the deployable context features. Controls
    that must fail: the weather features (nflverse records them after the game) and the filter's post-week
    margin.
    """
    t0 = time.perf_counter()
    years = range(LEAK_SEASONS[0], LEAK_SEASONS[1] + 1)
    pbp = mldata.load_pbp(years, columns=PBP_COLUMNS)
    sched = asof.add_asof(mldata.load_schedules(years))
    s17 = sched[sched["season"] == LEAK_SEASONS[1]].sort_values("kickoff")
    pick = []
    for wk, slots in ((1, (0, 1, -1)), (2, (1,)), (9, (0, -1)), (13, (2,)), (17, (3,))):
        rows = s17[(s17["week"] == wk) & (s17["game_type"] == "REG")]
        pick += [rows["game_id"].iloc[j] for j in slots]
    pick.append(s17[s17["game_type"] == "WC"]["game_id"].iloc[0])
    cfg = qb.tuned_config()
    builders = {}
    for v in QB_COLS:
        builders[f"qb_v2{v}_actual"] = (lambda p, s, g, v=v: qb.build_features_v2(p, s, g, cfg, "actual",
                                                                                    v2_config(v)), False, True)
        builders[f"qb_v2{v}_last"] = (lambda p, s, g, v=v: qb.build_features_v2(p, s, g, cfg, "last",
                                                                                  v2_config(v)), True, True)
    builders["kalman_carryover"] = (lambda p, s, g: kf.build_features(p, s, g, kf.TUNED_CARRYOVER), False, True)
    builders["early_season"] = (lambda p, s, g: context.m8_early_features(p, s, g), False, True)
    builders["context"] = (lambda p, s, g: context.m8_context_features(p, s, g)[context.M8_CONTEXT], False, True)
    builders["CONTROL_weather"] = (lambda p, s, g: context.m8_context_features(p, s, g, weather=True)[
        context.M8_WEATHER].assign(wind_raw=pd.to_numeric(g.set_index("game_id")["wind"], errors="coerce").fillna(-1),
                                   temp_raw=pd.to_numeric(g.set_index("game_id")["temp"], errors="coerce").fillna(-1)),
                                   False, False)
    builders["CONTROL_kalman_post_week"] = (lambda p, s, g: kf.build_features(p, s, g, kf.TUNED,
                                                                              positive_control=True), False, False)
    ok = True
    out = {}
    print(f"== M8 leakage check, real data {LEAK_SEASONS[0]}-{LEAK_SEASONS[1]}, {len(pick)} games ==")
    for name, (b, scramble, expect_free) in builders.items():
        r = asof.leakage_check(b, pbp, sched, pick, scramble_starters=scramble)
        free = bool(r["leak_free"].all())
        good = free == expect_free
        ok &= good
        out[name] = {"leak_free_games": int(r["leak_free"].sum()), "games": int(len(r)),
                     "expected_leak_free": expect_free, "as_expected": good}
        print(f"  {name:<28} leak-free {int(r['leak_free'].sum())}/{len(r)}  "
              f"{'as expected' if good else 'UNEXPECTED'}{'' if expect_free else ' (control: must fail)'}")
    print(f"{'PASS' if ok else 'FAIL'} ({time.perf_counter() - t0:.0f}s)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "leakcheck.json").write_text(json.dumps({"games": pick, "results": out, "pass": ok}, indent=2))
    return 0 if ok else 1


# --------------------------------------------------------------------------- stage: freeze

def _fit_params(frame: pd.DataFrame, sp: dict, specs: dict, season: int) -> dict:
    """The model `season` would use: fit on REG 2001..season-1 (stacks: bases walk-forward from 2003)."""
    if sp["kind"] == "logistic":
        tr = frame[frame["season"].between(TRAIN_START, season - 1)]
        m = logistic.fit_logistic(tr[sp["features"]].to_numpy(float), tr["y"].to_numpy(float),
                                  logistic.season_weights(tr["season"], season))
        return {"intercept": float(m.intercept_[0]), **dict(zip(sp["features"], map(float, m.coef_[0]))),
                "n_train": int(len(tr))}
    if sp["kind"] == "margin":
        tr = frame[frame["season"].between(TRAIN_START, season - 1)]
        f = mt.fit(tr[sp["features"]].to_numpy(float), tr[sp["target"]].to_numpy(float), tr["margin"].to_numpy(float),
                   tr["y"].to_numpy(float), logistic.season_weights(tr["season"], season))
        return {"intercept": float(f.beta[0]), **dict(zip(sp["features"], map(float, f.beta[1:]))),
                "conv_a": f.conv_a, "conv_b": f.conv_b, "sigma": f.sigma, "cal_a": f.cal_a, "cal_b": f.cal_b,
                "n_train": int(len(tr))}
    if sp["kind"] == "elo":
        return {}
    preds = {}
    for b in sp["bases"]:
        preds[b], _ = predict(frame, specs[b], (WF_FIRST, season - 1), preds)
    base = pd.DataFrame(preds)
    tr = frame["season"].between(WF_FIRST, season - 1).to_numpy()
    c, w = stack.fit_stack(stack.logit(base[tr].to_numpy(float)), frame["y"].to_numpy(float)[tr],
                           logistic.season_weights(frame["season"].to_numpy(int)[tr], season))
    return {"intercept": c, **dict(zip(sp["bases"], map(float, w))), "n_train": int(tr.sum()),
            "bases": {b: _fit_params(frame, specs[b], specs, season) for b in sp["bases"]}}


def _flat(d: dict, prefix="") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        elif isinstance(v, (int, float)):
            out[f"{prefix}{k}"] = float(v)
    return out


def stage_freeze() -> int:
    """Fit the DEV-chosen model on REG 2001-2025 for 2026 and write experiments/live/m8_2026.json.

    Reproducibility check first: the same code refits the 2019 model on 2001-2018 and must match the DEV
    run's logged 2019 fit. 2020-2025 rows are used for fitting only; nothing is scored on them.
    """
    if FREEZE_PATH.exists():
        raise SystemExit(f"{FREEZE_PATH.relative_to(config.ROOT)} already exists; the freeze is written once")
    dev_json = OUT_DIR / "dev.json"
    if not dev_json.exists():
        raise SystemExit("no dev.json; run `dev` first")
    dev = json.loads(dev_json.read_text())
    specs = dev["specs"]
    sp = specs[dev["chosen"]]
    print(f"== M8 freeze for {FREEZE_SEASON}: {sp['step']} ({sp['kind']}) ==")
    games, pbp, sched = load_inputs(last_season=FREEZE_SEASON - 1)
    frame, _ = build_frame(games, pbp, sched, last_season=FREEZE_SEASON - 1)
    runs = sorted(glob.glob(str(registry.RUNS_DIR / f"*_m8_{sp['step']}.json")))
    if not runs:
        raise SystemExit(f"no logged DEV run m8_{sp['step']} to check against")
    ref = json.loads(Path(runs[-1]).read_text())
    ref19 = next(r for r in ref["coefficients"] if int(r["season"]) == DEV[1])
    dev_frame = frame[frame["season"] <= DEV[1]].reset_index(drop=True)
    got19 = _fit_params(dev_frame, sp, specs, DEV[1])
    a, b = _flat(got19), {k: float(v) for k, v in ref19.items() if k != "season" and isinstance(v, (int, float))}
    common = [k for k in b if k in a]
    worst = max(abs(a[k] - b[k]) for k in common)
    print(f"reproduction: {DEV[1]} fit vs {Path(runs[-1]).name}: max |diff| {worst:.2e} over {len(common)} values")
    if worst > 1e-8:
        raise SystemExit("the 2019 refit does not reproduce the DEV run. Stopping.")
    fit = _fit_params(frame, sp, specs, FREEZE_SEASON)
    fp = registry.data_fingerprint(games_csv=True, ml_datasets=("pbp", "schedules"),
                                   seasons=range(ml_m3.DATA_SEASONS[0], FREEZE_SEASON))
    fp["data/sources/stadiums.csv"] = registry.sha256_path(context.STADIUMS_CSV)
    tr = frame[frame["season"].between(TRAIN_START, FREEZE_SEASON - 1)]
    version = f"M8-{FREEZE_SEASON}-{registry.games_hash(tr['game_id'])[:8]}"
    rec = {"model_version": version, "season": FREEZE_SEASON, "spec": sp, "fit": fit,
           "knobs": {"ratings": oa.TUNED.to_dict(), "qb": qb.tuned_config().to_dict(), "kalman": kf.TUNED.to_dict(),
                     "kalman_carryover": kf.TUNED_CARRYOVER.to_dict(),
                     "qb_v2": {v: v2_config(v).to_dict() for v in QB_COLS}, "blend_w": BLEND_W,
                     "stack": {"lam": stack.STACK_LAM, "first_season": WF_FIRST},
                     "elo_config": config.DEFAULT_CONFIG.to_dict(), "train_start": TRAIN_START,
                     "season_half_life": logistic.SEASON_HALF_LIFE},
           "training": {"seasons": [TRAIN_START, FREEZE_SEASON - 1], "n": int(len(tr)),
                        "games_hash": registry.games_hash(tr["game_id"])},
           "reproduction_check": {"season": DEV[1], "reference": Path(runs[-1]).name, "max_abs_diff": worst,
                                  "status": "reproduced"},
           "dev": {"chosen": dev["chosen"], "accepted": dev["accepted"], "brier": dev["brier"][dev["chosen"]],
                   "games_hash": dev["games_hash"]},
           "data": fp, "git": registry.git_commit(), "frozen_at_utc": pd.Timestamp.now("UTC").isoformat()}
    FREEZE_PATH.write_text(json.dumps(rec, indent=2, default=float) + "\n")
    print(f"wrote {FREEZE_PATH.relative_to(config.ROOT)} ({version}); commit it before the first live M8 row")
    return 0


# --------------------------------------------------------------------------- stage: forward test

def forward_runs() -> list[str]:
    return sorted(Path(p).name for p in glob.glob(str(registry.RUNS_DIR / "*_m8_forward_*.json")))


def stage_forward(log: bool) -> int:
    """The pre-registered forward test (context/ml.md, "M8 forward test"), run once after the 2026 REG season."""
    prior = forward_runs()
    if prior:
        raise SystemExit(f"the M8 forward test is already recorded ({prior}); it is run once only")
    if not FREEZE_PATH.exists():
        raise SystemExit(f"no frozen model ({FREEZE_PATH.name}); nothing to test")
    sched = asof.add_asof(mldata.load_schedules([FREEZE_SEASON]))
    reg = sched[sched["game_type"] == "REG"]
    if len(reg) == 0 or reg["home_score"].isna().any():
        raise SystemExit(f"the {FREEZE_SEASON} regular season is not complete "
                         f"({int(reg['home_score'].isna().sum())} REG games without a result); refusing")
    if not FORWARD_LEDGER.exists():
        raise SystemExit(f"no M8 ledger at {FORWARD_LEDGER.relative_to(config.ROOT)}")
    m8 = live.latest_before_kickoff(live.read_ledger(FORWARD_LEDGER))
    a4 = live.latest_before_kickoff(live.read_ledger(live.ledger_path(FREEZE_SEASON)))
    first = m8["kickoff_utc"].min()
    switch_week = int(reg.loc[reg["game_id"].isin(m8["game_id"]), "week"].min())
    games = reg[reg["week"] >= switch_week].copy()
    games["y"] = np.where(games["home_score"] > games["away_score"], 1.0,
                          np.where(games["home_score"] == games["away_score"], 0.5, 0.0))
    df = games[["game_id", "week", "y"]].merge(m8[["game_id", "p_home_model"]].rename(columns={"p_home_model": "m8"}),
                                               on="game_id", how="left")
    df = df.merge(a4[["game_id", "p_home_model"]].rename(columns={"p_home_model": "a4s"}), on="game_id", how="left")
    missing = df[df[["m8", "a4s"]].isna().any(axis=1)]
    df = df.dropna(subset=["m8", "a4s"])
    y = df["y"].to_numpy()
    c = ci(y, df["m8"].to_numpy(), df["a4s"].to_numpy())
    res = {"switch_week": switch_week, "first_m8_kickoff": str(first), "n": int(len(df)),
           "missing": missing["game_id"].tolist(), "brier_m8": float(np.mean((y - df["m8"]) ** 2)),
           "brier_a4s": float(np.mean((y - df["a4s"]) ** 2)), "m8_minus_a4s": c,
           "pass": bool(c["ci_high"] < 0)}
    print(json.dumps(res, indent=2, default=float))
    print(f"PRIMARY (M8 minus A4s Brier, CI upper bound < 0): {'PASS' if res['pass'] else 'FAIL'}")
    if log:
        registry.log_run(f"m8_forward_{FREEZE_SEASON}", label="forward", seasons=(FREEZE_SEASON, FREEZE_SEASON),
                         game_ids=df["game_id"], metrics={"n": res["n"], "brier": res["brier_m8"],
                                                          "brier_a4s": res["brier_a4s"]},
                         features=[], params={"frozen": FREEZE_PATH.name}, data={},
                         notes="M8 forward test, primary: M8 minus A4s Brier on 2026 REG games from the switch week.",
                         extra={"result": res})
    return 0


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["tune", "dev", "leakcheck", "freeze", "forward"])
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    if a.stage == "tune":
        return stage_tune(not a.no_log)
    if a.stage == "dev":
        return stage_dev(not a.no_log)
    if a.stage == "leakcheck":
        return stage_leakcheck()
    if a.stage == "freeze":
        return stage_freeze()
    return stage_forward(not a.no_log)


if __name__ == "__main__":
    raise SystemExit(main())
