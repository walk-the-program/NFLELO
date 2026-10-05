"""M3b: game-model refinement on DEV (context/ml-m3b-method.md). Ladder C1-C4 against A4s, sign-off machinery.

    python scripts/ml_m3b.py tune               # Kalman knobs (2000-2005) and QB extras (b, c: 2000-2005; a: 2007-2008)
    python scripts/ml_m3b.py dev [--no-log]     # the ladder on the 3,450 DEV games; logs m3b_* runs (label dev)
    python scripts/ml_m3b.py leakcheck          # real-data leakage check of the new builders (2014-2017 data)
    python scripts/ml_m3b.py signoff-dry-run    # the sign-off code path on DEV, no logging; must reproduce `dev`
    python scripts/ml_m3b.py signoff            # the ONE pre-registered holdout run (refuses if one exists)

Every step is A4s's protocol (logistic, walk-forward REG 2001..S-1, season-decay
weights, ties as half rows, M3 rating knobs) with one change:

    A4s   elo_logit + adj_epa_margin + qb_delta_diff                    (reference, M3)
    C1    A4s + elo_logit x early + adj_epa_margin x early,  early = max(0, 1 - (week - 1) / w0)
          w0 picked on DEV from W0_GRID (the one DEV-tuned knob in M3b; reported as such)
    C2a   kf_margin replaces elo_logit
    C2b   kf_margin added alongside elo_logit
    C2c   kf_margin + qb_delta_diff (replaces elo_logit and adj_epa_margin)
    C2d   elo_logit + kf_margin + qb_delta_diff (replaces adj_epa_margin; added to the spec's C2a-c)
    C2k   logistic on kf_margin alone (the A0 analogue)
    KF    the filter alone, no logistic: Phi(kf_margin / sqrt(kf_sd^2 + sigma_m^2))
    C3a/b/c  A4s with qb_delta_diff from the QB value with the CPOE composite / experience prior / aging
    C3d      the experience prior re-tuned on changed-starter games only (added after C3b's DEV result)
    C3abc    all three QB extras together
    C4*   combinations of the C1, best C2 and best C3 changes that beat A4s on their own

The final model is picked by the one-SE rule across every logistic step
(A4s included): among steps within one paired-bootstrap SE of the best DEV
Brier, take the simplest (fewest features, then fewest QB extras). Every CI is
paired against A4s (2,000 reps, seed 20261003). Leak alarm: any step better
than A4s by more than 0.006 Brier is investigated before it is reported.

Nothing here reads 2020-2025 except `signoff`, which is not run without Walker's OK.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import optimize  # noqa: E402

from nflelo import config  # noqa: E402
from nflelo import data as elo_data  # noqa: E402
from nflelo.ml import asof, registry  # noqa: E402
from nflelo.ml import data as mldata  # noqa: E402
from nflelo.ml.eval import baselines, bootstrap, calibration, metrics, windows  # noqa: E402
from nflelo.ml.features import kalman as kf  # noqa: E402
from nflelo.ml.features import opponent_adjust as oa  # noqa: E402
from nflelo.ml.features import qb  # noqa: E402
from nflelo.ml.models import logistic  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ml_m3  # noqa: E402

DEV = windows.DEV
TRAIN_START = logistic.TRAIN_START
OUT_DIR = mldata.ML_DIR / "m3b"
PBP_COLUMNS = ml_m3.PBP_COLUMNS + ["cpoe"]
A4S_DEV_BRIER = 0.2151
A4S_DEV_N = 3450
LEAK_MARGIN = 0.006
W0_GRID = (1, 2, 3, 4, 5, 6, 8, 10, 12, 18)
WEEK_BUCKETS = ((1, 4), (5, 9), (10, 18))
CPOE_TUNE = (2007, 2008)
HOLDOUT_N = 1615
HOLDOUT_HASH = "1b27abff81bddb2c"
A4S = ["elo_logit", "adj_epa_margin", "qb_delta_diff"]
QB_VARIANTS = ("a", "b", "c", "d", "abc")


# --------------------------------------------------------------------------- QB extras (frozen by `tune`)

def qb_variant(v: str) -> qb.QBExtras:
    """The tuned extras for one C3 variant: 'a' CPOE, 'b' experience prior (tuned on all starters), 'c' aging,
    'd' experience prior tuned on changed-starter games only, 'abc' the first three together."""
    parts = {"a": dict(cpoe_weight=qb.TUNED_EXTRAS.cpoe_weight, cpoe_k=qb.TUNED_EXTRAS.cpoe_k),
             "b": dict(experience=qb.TUNED_EXTRAS.experience),
             "c": dict(aging=qb.TUNED_EXTRAS.aging),
             "d": dict(experience=qb.TUNED_EXPERIENCE_CHANGED)}
    kw = {}
    for ch in v:
        kw.update(parts[ch])
    return qb.QBExtras(**kw)


# --------------------------------------------------------------------------- data and features

def load_inputs(last_season: int = DEV[1]):
    games = elo_data.load_games()
    games = games[games["season"] <= last_season].reset_index(drop=True)
    years = range(ml_m3.DATA_SEASONS[0], last_season + 1)
    pbp = mldata.load_pbp(years, columns=PBP_COLUMNS)
    sched = asof.add_asof(mldata.load_schedules(years))
    return games, pbp, sched


def build_frame(games, pbp, sched, last_season: int = DEV[1]) -> tuple[pd.DataFrame, dict]:
    """M3's frame (A4s features and more) plus kf_margin/kf_sd, the QB variants, and the changed-starter flag."""
    frame, t = ml_m3.build_frame(games, pbp, sched, last_season=last_season)
    t.pop("ratings_obj", None)
    reg = sched.set_index("game_id").loc[frame["game_id"]].reset_index()

    t0 = time.perf_counter()
    k = kf.build_features(pbp, sched, reg, kf.TUNED)
    frame["kf_margin"], frame["kf_sd"] = k["kf_margin"].to_numpy(), k["kf_sd"].to_numpy()
    t["kalman"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    births = qb.birth_dates(mldata.load_players())
    cfg = qb.tuned_config()
    st = qb.ext_states(pbp, sched, reg, cfg, births)
    last = qb.last_starters(sched, reg)
    actual = reg.set_index("game_id")[["home_qb_id", "away_qb_id"]]
    for v in QB_VARIANTS:
        ex = qb_variant(v)
        for name, starters in (("", actual), ("last_", last)):
            d = qb.qb_deltas_ext(st["states"], st["qbs"], reg, starters, cfg.k, ex)
            frame[f"qb_{name}delta_diff_{v}"] = (d["qb_delta_home"] - d["qb_delta_away"]).to_numpy()
    t["qb_extras"] = time.perf_counter() - t0
    a_h, a_a = actual["home_qb_id"].astype(str).to_numpy(), actual["away_qb_id"].astype(str).to_numpy()
    l_h, l_a = last["home_qb_id"].astype(str).to_numpy(), last["away_qb_id"].astype(str).to_numpy()
    frame["changed"] = (a_h != l_h) | (a_a != l_a)
    frame["weeks_played"] = (frame["week"] - 1).clip(lower=0)
    if frame[["kf_margin", "kf_sd"]].isna().any().any():
        raise SystemExit("NaN in Kalman features")
    return frame, t


def add_early(frame: pd.DataFrame, w0: float) -> list[str]:
    """Add early = max(0, 1 - weeks_played / w0) interactions; return their names."""
    early = np.maximum(0.0, 1.0 - frame["weeks_played"].to_numpy(float) / float(w0))
    names = []
    for f in ("elo_logit", "kf_margin", "adj_epa_margin"):
        col = f"{f}_x_early{w0:g}"
        frame[col] = frame[f].to_numpy(float) * early
        names.append(col)
    return names


def early_cols(features: list[str], w0: float) -> list[str]:
    """C1's interactions for a feature list: the rating feature(s) present and adj_epa_margin."""
    return [f"{f}_x_early{w0:g}" for f in ("elo_logit", "kf_margin", "adj_epa_margin") if f in features]


def swap_qb(features: list[str], v: str | None, last: bool = False) -> list[str]:
    """Replace qb_delta_diff with the variant's column (and/or the last-starter version)."""
    src = "qb_delta_diff"
    dst = ("qb_last_delta_diff" if last else "qb_delta_diff") + (f"_{v}" if v else "")
    return [dst if f == src else f for f in features]


# --------------------------------------------------------------------------- scoring helpers

def score(y, p) -> dict:
    m = {**metrics.win_metrics(y, p), "ece": calibration.ece(y, p)}
    null = calibration.ece_null_range(p)
    m["ece_null_p95"] = null["p95"]
    m["ece_pass"] = bool(m["ece"] <= null["p95"])  # one-sided: only too-large ECE fails
    return m


def ci(y, a, b, metric="brier") -> dict:
    return bootstrap.paired_bootstrap(y, a, b, metric)


def fmt_ci(c) -> str:
    return f"{c['diff']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]"


def splits(y, p, ref, mkt, frame_sub) -> dict:
    """Week buckets and changed-starter splits: Brier of p, ref (A4s) and market, and p minus ref / market CIs."""
    out = {}
    wk = frame_sub["week"].to_numpy()
    ch = frame_sub["changed"].to_numpy(bool)
    groups = {f"weeks_{a}_{b}": (wk >= a) & (wk <= b) for a, b in WEEK_BUCKETS}
    groups["changed"] = ch
    groups["unchanged"] = ~ch
    for name, m in groups.items():
        if not m.any():
            continue
        out[name] = {"n": int(m.sum()), "brier": float(np.mean((y[m] - p[m]) ** 2)),
                     "brier_a4s": float(np.mean((y[m] - ref[m]) ** 2)),
                     "brier_market": float(np.mean((y[m] - mkt[m]) ** 2)),
                     "vs_a4s": ci(y[m], p[m], ref[m]), "vs_market": ci(y[m], p[m], mkt[m])}
    return out


# --------------------------------------------------------------------------- stage: tune

def tune_kalman(log: bool) -> dict:
    years = range(kf.BURN_IN_FIRST, kf.TUNE_SEASONS[1] + 1)
    pbp = mldata.load_pbp(years, columns=PBP_COLUMNS)
    sched = asof.add_asof(mldata.load_schedules(years))
    gt = kf.game_table(pbp, sched, kf.TUNED)
    res = kf.tune(gt)
    c = res["config"]
    print(f"== Kalman tuning, {kf.TUNE_SEASONS[0]}-{kf.TUNE_SEASONS[1]} (filter from {kf.BURN_IN_FIRST}), "
          f"{res['n_games']} games, {res['seconds']:.1f}s, {res['evaluations']} evaluations ==")
    print("starts (joint loglik, margin loglik, knobs):")
    for s in res["starts"]:
        print(f"  {s['joint']:10.2f} {s['margin']:10.2f}  " + ", ".join(f"{k} {s[k]:.4g}" for k in kf.TUNED_KEYS))
    print("margin-only optima from the same starts (not used; flat ridge):")
    for s in res["margin_only_optima"]:
        print(f"  {s['loglik']:10.2f}  " + ", ".join(f"{k} {s[k]:.4g}" for k in kf.TUNED_KEYS))
    print("CHOSEN: " + ", ".join(f"{k} {getattr(c, k):.5g}" for k in (*kf.TUNED_KEYS, "epa_home")))
    print(f"margin loglik per game {res['loglik_per_game']:.4f} (constant-mean model "
          f"{res['loglik_constant_per_game']:.4f}); RMSE {res['rmse']:.3f} vs SD {res['rmse_constant']:.3f}; "
          f"at bounds: {res['at_bounds'] or 'none'}")
    drift = {"weekly_sd": c.q_week ** 0.5, "season_sd_17wk": (17 * c.q_week) ** 0.5,
             "offseason_sd": c.q_season ** 0.5, "hfa_sd_per_season": (20 * c.q_hfa) ** 0.5}
    print("in words: " + ", ".join(f"{k} {v:.2f} pts" for k, v in drift.items()))
    if log:
        registry.log_run("m3b_tune_kalman", label="tune", seasons=kf.TUNE_SEASONS, game_ids=gt.loc[
            gt["season"].between(*kf.TUNE_SEASONS), "game_id"], metrics={"n": res["n_games"],
            "loglik_per_game": res["loglik_per_game"], "joint_loglik": res["joint_loglik"], "rmse": res["rmse"]},
            params={"chosen": c.to_dict(), "objective": "joint predictive density of margin and EPA margin"},
            data=registry.data_fingerprint(games_csv=False, ml_datasets=("pbp", "schedules"), seasons=years),
            notes="M3b C2: Kalman filter knobs by maximum likelihood on 2000-2005 (1999 burn-in).",
            extra={"starts": res["starts"], "margin_only_optima": res["margin_only_optima"], "drift": drift})
    return res


def tune_qb(log: bool) -> dict:
    """C3 tuning: experience offsets and aging on 2000-2005, CPOE weight and k on 2007-2008 (QB EPA target)."""
    t0 = time.perf_counter()
    years = range(ml_m3.DATA_SEASONS[0], CPOE_TUNE[1] + 1)
    pbp = mldata.load_pbp(years, columns=PBP_COLUMNS)
    sched = asof.add_asof(mldata.load_schedules(years))
    births = qb.birth_dates(mldata.load_players())
    cfg = qb.tuned_config()
    out = {}
    # ---- (b) and (c) on 2000-2005
    sub = sched[sched["season"] <= windows.TUNE[1]]
    psub = pbp[pbp["game_id"].isin(set(sub["game_id"]))]
    db = qb.dropback_plays(psub, sub, cfg)
    qg = qb.qb_game_table_ext(db, sub, births)
    tg = qb.starter_targets(qg, sub, windows.TUNE)
    keys = sorted(set(zip(tg["season"].astype(int), tg["week"].astype(int))))
    st = qb.ext_states(psub, sub, sub, cfg, births, keys=keys)
    base = qb.score_values(st["states"], st["qbs"], tg, cfg.k, qb.QBExtras(cpoe_weight=0.0))
    base_m3 = qb.score_k(*_plain_states(psub, sub, cfg, keys), tg, cfg.k)
    print(f"== QB extras tuning ==\nbaseline (M3 QB value, k {cfg.k:g}) on {windows.TUNE[0]}-{windows.TUNE[1]}: "
          f"wMSE {base['wmse']:.5f} ({base['n_games']} starter-games); M3 code path {base_m3['wmse']:.5f}")

    def wmse_exp(d):
        return qb.score_values(st["states"], st["qbs"], tg, cfg.k, qb.QBExtras(experience=tuple(map(float, d))))["wmse"]
    r = optimize.minimize(wmse_exp, np.zeros(len(qb.EXP_EDGES)), method="Nelder-Mead",
                          options={"xatol": 1e-5, "fatol": 1e-9, "maxiter": 4000})
    exp = tuple(round(float(v), 4) for v in r.x)
    # bucket populations at prediction time (starter-games)
    car = []
    for (S, W), idx in tg.groupby(["season", "week"]).groups.items():
        s_ = st["states"][(int(S), int(W))]
        c = st["qbs"].get_indexer(tg.loc[idx, "qb"].astype(str))
        b = np.where(c >= 0, np.searchsorted(np.asarray(qb.EXP_EDGES[1:], float), s_.career[np.maximum(c, 0)],
                                             side="right"), 0)
        b = np.where((c >= 0) & s_.veteran[np.maximum(c, 0)], len(qb.EXP_EDGES) - 1, b)
        car.extend(b.tolist())
    counts = np.bincount(np.asarray(car), minlength=len(qb.EXP_EDGES)).tolist()
    out["experience"] = {"offsets": exp, "wmse": float(r.fun), "base_wmse": base["wmse"], "bucket_games": counts,
                         "edges": list(qb.EXP_EDGES)}
    print(f"(b) experience offsets {exp} (buckets {qb.EXP_EDGES}, starter-games {counts}): "
          f"wMSE {r.fun:.5f} vs {base['wmse']:.5f}")

    # (b2) the same offsets tuned only on starter-games where the team's starter changed from its previous
    # game: backups, rookies, and returns, the games the prior is for. Added after C3b's DEV result.
    games_ = sub[sub["game_type"] == "REG"]
    last = qb.last_starters(sub, games_)
    prev = np.array([last.at[g, f"{sd}_qb_id"] for g, sd in zip(tg["game_id"], tg["side"])], dtype=object)
    chg = tg["qb"].astype(str).to_numpy() != pd.Series(prev).astype(str).to_numpy()
    tgc = tg[chg].reset_index(drop=True)
    base_b2 = qb.score_values(st["states"], st["qbs"], tgc, cfg.k, qb.QBExtras(cpoe_weight=0.0))["wmse"]

    def wmse_exp2(d):
        return qb.score_values(st["states"], st["qbs"], tgc, cfg.k,
                               qb.QBExtras(experience=tuple(map(float, d))))["wmse"]
    r2 = optimize.minimize(wmse_exp2, np.zeros(len(qb.EXP_EDGES)), method="Nelder-Mead",
                           options={"xatol": 1e-5, "fatol": 1e-9, "maxiter": 4000})
    exp2 = tuple(round(float(v), 4) for v in r2.x)
    b_on_chg = wmse_exp2(exp)
    out["experience_changed"] = {"offsets": exp2, "wmse": float(r2.fun), "base_wmse": float(base_b2),
                                 "n_games": int(len(tgc)), "all_starter_offsets_wmse_here": float(b_on_chg)}
    print(f"(b2) experience offsets tuned on the {len(tgc)} changed-starter games: {exp2}: wMSE {r2.fun:.5f} vs "
          f"{base_b2:.5f} without (the all-starter offsets give {b_on_chg:.5f} here)")

    grid = []
    for peak in (26, 27, 28, 29, 30, 31, 32, 33, 34):
        for before in (0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20):
            for after in (0.0, -0.005, -0.01, -0.02, -0.03, -0.05):
                w = qb.score_values(st["states"], st["qbs"], tg, cfg.k,
                                    qb.QBExtras(aging=(peak, before, after)))["wmse"]
                grid.append({"peak": peak, "before": before, "after": after, "wmse": w})
    g = pd.DataFrame(grid).sort_values("wmse")
    best = g.iloc[0]
    aging = (float(best["peak"]), float(best["before"]), float(best["after"]))
    n_age = int(qg["age"].notna().sum())
    out["aging"] = {"curve": aging, "wmse": float(best["wmse"]), "base_wmse": base["wmse"],
                    "grid_top5": g.head(5).to_dict("records"), "qb_games_with_age": n_age,
                    "qb_games": int(len(qg))}
    print(f"(c) aging (peak, slope before, slope after) = {aging}: wMSE {best['wmse']:.5f} vs {base['wmse']:.5f}; "
          f"{n_age} of {len(qg)} QB-games have a birth date; top 5:\n{g.head(5).to_string(index=False)}")

    # ---- (a) CPOE on 2007-2008, history from 1999 (CPOE exists from 2006)
    tgc_all = qb.starter_targets(qb.qb_game_table_ext(qb.dropback_plays(pbp, sched, cfg), sched, births),
                                 sched, CPOE_TUNE)
    keys = sorted(set(zip(tgc_all["season"].astype(int), tgc_all["week"].astype(int))))
    stc = qb.ext_states(pbp, sched, sched, cfg, births, keys=keys)
    base_c = qb.score_values(stc["states"], stc["qbs"], tgc_all, cfg.k, qb.QBExtras())["wmse"]
    cg = []
    for w in (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03):
        for kc in (25, 50, 100, 200, 400, 800):
            cg.append({"cpoe_weight": w, "cpoe_k": kc, "wmse": qb.score_values(
                stc["states"], stc["qbs"], tgc_all, cfg.k, qb.QBExtras(cpoe_weight=w, cpoe_k=float(kc)))["wmse"]})
    cg = pd.DataFrame(cg).sort_values("wmse")
    bc = cg.iloc[0]
    out["cpoe"] = {"cpoe_weight": float(bc["cpoe_weight"]), "cpoe_k": float(bc["cpoe_k"]), "wmse": float(bc["wmse"]),
                   "base_wmse": float(base_c), "n_games": int(len(tgc_all)), "grid_top5": cg.head(5).to_dict("records")}
    print(f"(a) CPOE composite on {CPOE_TUNE[0]}-{CPOE_TUNE[1]} ({len(tgc_all)} starter-games): weight "
          f"{bc['cpoe_weight']:g} EPA/dropback per CPOE point, k {bc['cpoe_k']:g} attempts: wMSE {bc['wmse']:.5f} "
          f"vs {base_c:.5f} without; top 5:\n{cg.head(5).to_string(index=False)}")
    out["seconds"] = time.perf_counter() - t0
    print(f"QB tuning {out['seconds']:.1f}s")
    if log:
        registry.log_run("m3b_tune_qb", label="tune", seasons=(windows.TUNE[0], CPOE_TUNE[1]),
                         game_ids=sorted(set(tg["game_id"]) | set(tgc_all["game_id"])),
                         metrics={"n": int(len(tg) + len(tgc_all))}, params={"qb": cfg.to_dict()},
                         data=registry.data_fingerprint(games_csv=False, ml_datasets=("pbp", "schedules"),
                                                        seasons=years),
                         notes="M3b C3 QB extras: experience and aging on 2000-2005, CPOE on 2007-2008 "
                               "(next-game starter EPA per dropback, dropback-weighted MSE). Draft round not used "
                               "(nflverse players draft fields come from Pro-Football-Reference).",
                         extra={"results": out})
    return out


def _plain_states(pbp, sched, cfg, keys):
    db = qb.dropback_plays(pbp, sched, cfg)
    qg = qb.qb_game_table(db, sched)
    priors = qb.replacement_priors(db, {s for s, _ in keys}, cfg.rookie_dropbacks)
    return qb.week_states(qg, sched, keys, cfg, priors)


def stage_tune(log: bool) -> int:
    k = tune_kalman(log)
    q = tune_qb(log)
    c = k["config"]
    print("\nFreeze in code:")
    print("  kalman.TUNED = KalmanConfig(" + ", ".join(f"{x}={getattr(c, x):.5g}" for x in (*kf.TUNED_KEYS, "epa_home"))
          + ")")
    print(f"  qb.TUNED_EXTRAS = QBExtras(cpoe_weight={q['cpoe']['cpoe_weight']:g}, cpoe_k={q['cpoe']['cpoe_k']:g}, "
          f"experience={q['experience']['offsets']}, aging={q['aging']['curve']})")
    print(f"  qb.TUNED_EXPERIENCE_CHANGED = {q['experience_changed']['offsets']}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tune.json").write_text(json.dumps({"kalman": {**{k_: v for k_, v in k.items() if k_ != "config"},
                                                              "config": c.to_dict()}, "qb": q},
                                                  indent=2, default=float))
    return 0


# --------------------------------------------------------------------------- stage: dev ladder

def spec(step, features, qbv=None, w0=None, kind="logistic", note=""):
    return {"step": step, "features": list(features), "qb": qbv, "w0": w0, "kind": kind, "note": note}


def complexity(sp) -> tuple:
    return (len(sp["features"]), len(sp["qb"] or ""))


def resolve(frame: pd.DataFrame, sp: dict, last: bool = False) -> list[str]:
    """Frame columns for a spec: QB variant swapped in, early interactions added (and created) when w0 is set."""
    feats = swap_qb(sp["features"], sp["qb"], last)
    if sp["w0"] is not None:
        add_early(frame, sp["w0"])
        feats = feats + early_cols(sp["features"], sp["w0"])
    return feats


def predict(frame: pd.DataFrame, sp: dict, window=DEV, last: bool = False) -> tuple[pd.Series, pd.DataFrame]:
    if sp["kind"] == "kf":
        p = pd.Series(kf.win_prob(frame["kf_margin"], frame["kf_sd"], kf.TUNED.sigma_m), index=frame.index)
        p[~frame["season"].between(*window)] = np.nan
        return p, pd.DataFrame()
    feats = resolve(frame, sp, last)
    return logistic.walk_forward(frame, feats, window, train_start=TRAIN_START)


def prepare(window, allow_holdout):
    """Inputs, the feature frame, and the scored-game mask for `window` (REG games with moneylines)."""
    lo, hi = window
    windows.check(window, allow_holdout)
    games, pbp, sched = load_inputs(last_season=hi)
    base = baselines.baseline_frame(games, window, allow_holdout=allow_holdout, require_market=True)
    t0 = time.perf_counter()
    frame, ft = build_frame(games, pbp, sched, last_season=hi)
    ft["total_features"] = time.perf_counter() - t0
    frame = frame.merge(base[["game_id", "p_elo_v2", baselines.MARKET_COL]], on="game_id", how="left")
    mask = frame["game_id"].isin(base["game_id"]).to_numpy()
    if mask.sum() != len(base):
        raise SystemExit(f"only {mask.sum()} of {len(base)} scored games found in the feature frame")
    sub = frame[mask]
    if not np.allclose(sub["y"].to_numpy(), base.set_index("game_id").loc[sub["game_id"], "y"].to_numpy()):
        raise SystemExit("outcomes disagree between the schedule and data/games.csv")
    return frame, mask, ft


def stage_dev(log: bool) -> int:
    t_start = time.perf_counter()
    frame, mask, ft = prepare(DEV, allow_holdout=False)
    sub = frame[mask]
    y = sub["y"].to_numpy()
    elo_p, mkt_p = sub["p_elo_v2"].to_numpy(), sub[baselines.MARKET_COL].to_numpy()
    gh = registry.games_hash(sub["game_id"])
    print(f"DEV {DEV[0]}-{DEV[1]}: n = {len(sub)}, games hash {gh}. Features {ft['total_features']:.1f}s ("
          + ", ".join(f"{k} {v:.1f}s" for k, v in ft.items() if k != "total_features") + ")")

    t_ladder = time.perf_counter()
    preds, coefs, specs = {}, {}, {}

    def run(sp, last=False):
        p, c = predict(frame, sp, DEV, last)
        key = sp["step"]
        preds[key], coefs[key], specs[key] = p, c, sp
        return p[mask].to_numpy()

    a4s = run(spec("A4s", A4S, note="M3 chosen model (reference)"))
    b_a4s = float(np.mean((y - a4s) ** 2))
    print(f"A4s reproduction: Brier {b_a4s:.5f} on n = {len(y)} (M3 ladder {A4S_DEV_BRIER}, n {A4S_DEV_N})")
    if len(y) != A4S_DEV_N or abs(b_a4s - A4S_DEV_BRIER) > 0.00005:
        print("A4s does not reproduce. Stopping.", file=sys.stderr)
        return 1
    run(spec("A4bs", swap_qb(A4S, None, last=True), note="A4s with last game's starter (Wednesday rule)"))

    # ---- C1: week-varying weights, w0 grid (DEV-tuned)
    grid = []
    for w0 in W0_GRID:
        sp = spec(f"C1_w0_{w0:g}", A4S, w0=float(w0))
        p = run(sp)
        grid.append({"w0": float(w0), "brier": float(np.mean((y - p) ** 2)), "vs_a4s": ci(y, p, a4s)})
    best_w0 = min(grid, key=lambda g: g["brier"])["w0"]
    c1 = spec("C1", A4S, w0=best_w0, note=f"early interactions, w0 = {best_w0:g} (DEV-tuned over {list(W0_GRID)})")
    run(c1)

    # ---- C2: Kalman filter
    run(spec("C2a", ["kf_margin", "adj_epa_margin", "qb_delta_diff"], note="kf_margin replaces elo_logit"))
    run(spec("C2b", ["elo_logit", "kf_margin", "adj_epa_margin", "qb_delta_diff"], note="kf_margin added"))
    run(spec("C2c", ["kf_margin", "qb_delta_diff"], note="kf_margin replaces elo_logit and adj_epa_margin"))
    run(spec("C2d", ["elo_logit", "kf_margin", "qb_delta_diff"],
             note="kf_margin replaces adj_epa_margin, Elo kept (added: the fourth keep/replace cell)"))
    run(spec("C2k", ["kf_margin"], note="logistic on kf_margin alone (A0 analogue)"))
    run(spec("KF", [], kind="kf", note="filter alone: Phi(kf_margin / sqrt(kf_sd^2 + sigma_m^2))"))

    # ---- C3: QB value variants
    for v in QB_VARIANTS:
        run(spec(f"C3{v}", A4S, qbv=v, note={"a": "CPOE composite", "b": "experience prior", "c": "aging",
                                             "d": "experience prior tuned on changed-starter games (added after C3b)",
                                             "abc": "CPOE + experience + aging"}[v]))

    def brier(step):
        return float(np.mean((y - preds[step][mask].to_numpy()) ** 2))

    # ---- C4: combinations of the changes that beat A4s on their own
    comps = {}
    if brier("C1") < b_a4s:
        comps["C1"] = {"w0": best_w0}
    c2best = min(("C2a", "C2b", "C2c", "C2d"), key=brier)
    if brier(c2best) < b_a4s:
        comps["C2"] = {"features": specs[c2best]["features"], "from": c2best}
    c3best = min((f"C3{v}" for v in QB_VARIANTS), key=brier)
    if brier(c3best) < b_a4s:
        comps["C3"] = {"qb": specs[c3best]["qb"], "from": c3best}
    combos = []
    names = list(comps)
    for r_ in range(2, len(names) + 1):
        for combo in combinations(names, r_):
            feats = comps["C2"]["features"] if "C2" in combo else A4S
            sp = spec("C4_" + "+".join(comps[c].get("from", c) for c in combo), feats,
                      qbv=comps["C3"]["qb"] if "C3" in combo else None,
                      w0=best_w0 if "C1" in combo else None, note="combination: " + " + ".join(combo))
            run(sp)
            combos.append(sp["step"])
    t_ladder = time.perf_counter() - t_ladder

    # ---- scores, CIs vs A4s, one-SE rule
    steps = [s for s in preds if not s.startswith("C1_w0_")]
    res = {}
    for s in steps:
        p = preds[s][mask].to_numpy()
        m = score(y, p)
        m["vs_a4s"] = ci(y, p, a4s)
        m["vs_a4s_logloss"] = ci(y, p, a4s, "logloss")
        m["vs_elo"] = ci(y, p, elo_p)
        m["vs_market"] = ci(y, p, mkt_p)
        res[s] = m
    cand = [s for s in steps if specs[s]["kind"] == "logistic" and s != "A4bs"]
    best = min(cand, key=lambda s: res[s]["brier"])
    pb = preds[best][mask].to_numpy()
    for s in cand:
        c = ci(y, preds[s][mask].to_numpy(), pb)
        res[s]["vs_best"] = c
        res[s]["within_1se"] = True if s == best else bool(c["diff"] <= c["boot_se"])
    eligible = [s for s in cand if res[s]["within_1se"]]
    chosen = min(eligible, key=lambda s: (complexity(specs[s]), res[s]["brier"]))

    print(f"\nC1 grid (DEV-tuned w0; early = max(0, 1 - (week - 1) / w0)), Brier and minus A4s:")
    for g in grid:
        print(f"  w0 {g['w0']:>4g}: {g['brier']:.5f}  {fmt_ci(g['vs_a4s'])}" + ("  <- best" if g["w0"] == best_w0 else ""))

    print(f"\nLadder, DEV {DEV[0]}-{DEV[1]}, n = {len(y)}. diff = step minus A4s (paired bootstrap, 2,000 reps, "
          "seed 20261003). ECE ok = at or below the 95th percentile of a calibrated forecaster at this n.")
    print(f"{'step':<22}{'brier':>8}{'logloss':>9}{'acc':>7}{'ece':>8}{'p95':>7}{'ok':>4}  {'vs A4s [95% CI]':<28}"
          f"{'1SE':>4}")
    for s in steps:
        m = res[s]
        flag = "" if specs[s]["kind"] != "logistic" or s == "A4bs" else ("yes" if res[s]["within_1se"] else "")
        print(f"{s:<22}{m['brier']:>8.4f}{m['logloss']:>9.4f}{m['accuracy']:>7.3f}{m['ece']:>8.4f}"
              f"{m['ece_null_p95']:>7.3f}{'y' if m['ece_pass'] else 'N':>4}  {fmt_ci(m['vs_a4s']):<28}{flag:>4}"
              + (" *" if s == chosen else ""))
    me, mm = score(y, elo_p), score(y, mkt_p)
    print(f"{'Elo v2':<22}{me['brier']:>8.4f}{me['logloss']:>9.4f}{me['accuracy']:>7.3f}{me['ece']:>8.4f}")
    print(f"{'Market':<22}{mm['brier']:>8.4f}{mm['logloss']:>9.4f}{mm['accuracy']:>7.3f}{mm['ece']:>8.4f}")
    alarm = [s for s in steps if s != "A4s" and res[s]["vs_a4s"]["diff"] < -LEAK_MARGIN]
    print(f"Leak alarm (better than A4s by more than {LEAK_MARGIN}): {alarm or 'none'}")
    print(f"Best: {best} ({res[best]['brier']:.5f}). Within one SE: {eligible}. CHOSEN (one-SE rule): {chosen}.")
    cm = res[chosen]["vs_a4s"]
    print(f"Acceptance on DEV (beats A4s, CI excludes 0, one-sided ECE check): "
          f"{'PASS' if chosen != 'A4s' and cm['ci_high'] < 0 and res[chosen]['ece_pass'] else 'FAIL'}")

    # ---- splits for the main steps
    main = ["A4s", "C1", "C2a", "C2b", "C2c", "C2d", "KF", "C3a", "C3b", "C3c", "C3d", "C3abc"] + combos
    main = list(dict.fromkeys(main + [chosen]))
    split = {}
    print("\nSplits (Brier; step minus A4s [95% CI]; market Brier on the same games):")
    for s in main:
        split[s] = splits(y, preds[s][mask].to_numpy(), a4s, mkt_p, sub)
        cells = []
        for k_, v in split[s].items():
            cells.append(f"{k_} n{v['n']} {v['brier']:.4f} {fmt_ci(v['vs_a4s'])} mkt {v['brier_market']:.4f}")
        print(f"  {s}:\n    " + "\n    ".join(cells))

    # the Wednesday twin of the chosen model
    twin = spec(chosen + "_last", specs[chosen]["features"], qbv=specs[chosen]["qb"], w0=specs[chosen]["w0"],
                kind=specs[chosen]["kind"], note="the chosen model with last game's starter")
    if specs[chosen]["kind"] == "logistic":
        p_tw = run(twin, last=True)
        res[twin["step"]] = {**score(y, p_tw), "vs_a4s": ci(y, p_tw, a4s), "vs_a4bs": ci(y, p_tw, preds["A4bs"][mask])}
        print(f"\nWednesday twin {twin['step']}: Brier {res[twin['step']]['brier']:.4f}; minus A4bs "
              f"{fmt_ci(res[twin['step']]['vs_a4bs'])}; minus A4s {fmt_ci(res[twin['step']]['vs_a4s'])}")

    # ---- coefficients
    def show_coefs(s):
        c = coefs[s]
        if c is None or c.empty:
            return
        last = c.iloc[-1]
        cols = [x for x in c.columns if x not in ("season", "n_train")]
        print(f"  {s} ({int(last['season'])} fit, n {int(last['n_train'])}): " + ", ".join(
            f"{x} {last[x]:+.3f} [{c[x].min():+.3f}, {c[x].max():+.3f}]" for x in cols))
    print("\nCoefficients (last DEV refit; range over the 14 refits):")
    for s in dict.fromkeys(["A4s", "C1", "C2a", "C2b", "C2c", "C2d", "C2k", "C3abc", chosen]):
        show_coefs(s)
    corr = np.corrcoef(sub["kf_margin"], sub["elo_logit"])[0, 1]
    print(f"corr(kf_margin, elo_logit) on DEV = {corr:.3f}; corr(kf_margin, adj_epa_margin) = "
          f"{np.corrcoef(sub['kf_margin'], sub['adj_epa_margin'])[0, 1]:.3f}")
    qcorr = {v: float(np.corrcoef(sub["qb_delta_diff"], sub[f"qb_delta_diff_{v}"])[0, 1]) for v in QB_VARIANTS}
    print("corr(qb_delta_diff, variant): " + ", ".join(f"{k} {v:.3f}" for k, v in qcorr.items()))
    timing = {"features_s": ft, "ladder_s": t_ladder, "total_s": time.perf_counter() - t_start}
    print(f"\nTiming: features {ft['total_features']:.1f}s, ladder {t_ladder:.1f}s, total {timing['total_s']:.1f}s")

    # ---- save for the notebook and the dry run
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    for s, p in preds.items():
        out[f"p_{s}"] = p.to_numpy()
    out["dev"] = mask
    out.to_parquet(OUT_DIR / "ladder_frame.parquet", index=False)
    pd.concat([c.assign(step=s) for s, c in coefs.items() if c is not None and not c.empty],
              ignore_index=True).to_parquet(OUT_DIR / "coefs.parquet")
    summary = {"games_hash": gh, "n": int(len(y)), "chosen": chosen, "best": best, "eligible": eligible,
               "best_w0": best_w0, "c1_grid": grid, "leak_alarm": alarm, "kf_elo_corr": float(corr),
               "chosen_spec": specs[chosen], "twin": twin["step"],
               "brier": {s: res[s]["brier"] for s in res}, "timing": timing}
    (OUT_DIR / "dev.json").write_text(json.dumps(summary, indent=2, default=float))

    if log:
        fp = registry.data_fingerprint(games_csv=True, ml_datasets=("pbp", "schedules", "players"),
                                       seasons=list(range(ml_m3.DATA_SEASONS[0], DEV[1] + 1)) + [0])
        common = {"ratings": oa.TUNED.to_dict(), "qb": qb.tuned_config().to_dict(), "train_start": TRAIN_START,
                  "season_half_life": logistic.SEASON_HALF_LIFE, "ties": "two half-weight rows",
                  "elo_config": config.DEFAULT_CONFIG.to_dict(), "kalman": kf.TUNED.to_dict()}
        for s in list(res):
            sp = specs.get(s, twin)
            params = dict(common, model=sp["kind"], w0=sp["w0"],
                          qb_extras=qb_variant(sp["qb"]).to_dict() if sp["qb"] else None)
            if sp["w0"] is not None:
                params["w0_note"] = f"w0 tuned on DEV over {list(W0_GRID)}"
            m = res[s]
            registry.log_run(
                f"m3b_{s}", label="dev", seasons=DEV, game_ids=sub["game_id"],
                metrics={k: m[k] for k in ("n", "brier", "logloss", "accuracy", "ece", "ece_null_p95", "ece_pass")},
                features=resolve(frame, sp, s == twin["step"]) if sp["kind"] == "logistic" else ["kf_margin", "kf_sd"],
                params=params, data=fp, notes=(sp["note"] + (" CHOSEN by the one-SE rule." if s == chosen else "")).strip(),
                extra={"comparisons": {k: m[k] for k in ("vs_a4s", "vs_a4s_logloss", "vs_elo", "vs_market", "vs_best",
                                                         "vs_a4bs") if k in m},
                       "one_se": {"best": best, "within": m.get("within_1se"), "chosen": s == chosen,
                                  "eligible": eligible},
                       "splits": split.get(s), "c1_grid": grid if s == "C1" else None,
                       "coefficients": coefs[s].to_dict("records") if s in coefs and coefs[s] is not None
                       and not coefs[s].empty else None,
                       "timing": timing if s == chosen else None})
        print(f"\n{len(res)} runs written to {registry.RUNS_DIR.relative_to(config.ROOT)}/")
    return 0


# --------------------------------------------------------------------------- stage: leakcheck (real data)

LEAK_SEASONS = (2014, 2017)


def stage_leakcheck() -> int:
    """Rebuild each new feature after scrambling everything from the game's as_of on (`asof.leakage_check`).

    Real data 2014-2017 (the filter starts in 2014 here). Builders: the Kalman
    filter (TUNED) and every QB-extras variant with the actual starter
    (identity kept, M3-D1) and the last starter (identities scrambled too).
    Positive control: the filter's post-week margin, which must fail.
    """
    t0 = time.perf_counter()
    years = range(LEAK_SEASONS[0], LEAK_SEASONS[1] + 1)
    pbp = mldata.load_pbp(years, columns=PBP_COLUMNS)
    sched = asof.add_asof(mldata.load_schedules(years))
    births = qb.birth_dates(mldata.load_players())
    s17 = sched[sched["season"] == LEAK_SEASONS[1]].sort_values("kickoff")
    pick = []
    for wk, slots in ((1, (0, 1, -1)), (2, (1,)), (9, (0, -1)), (17, (3,))):
        rows = s17[(s17["week"] == wk) & (s17["game_type"] == "REG")]
        pick += [rows["game_id"].iloc[j] for j in slots]
    pick.append(s17[s17["game_type"] == "WC"]["game_id"].iloc[0])
    builders = {"kalman": (lambda p, s, g: kf.build_features(p, s, g, kf.TUNED), False, True)}
    for v in QB_VARIANTS:
        ex = qb_variant(v)
        builders[f"qb_{v}_actual"] = (lambda p, s, g, ex=ex: qb.build_features_ext(
            p, s, g, qb.tuned_config(), "actual", ex, births), False, True)
        builders[f"qb_{v}_last"] = (lambda p, s, g, ex=ex: qb.build_features_ext(
            p, s, g, qb.tuned_config(), "last", ex, births), True, True)
    builders["CONTROL_kalman_post_week"] = (lambda p, s, g: kf.build_features(p, s, g, kf.TUNED,
                                                                              positive_control=True), False, False)
    ok = True
    out = {}
    print(f"== M3b leakage check, real data {LEAK_SEASONS[0]}-{LEAK_SEASONS[1]}, {len(pick)} games ==")
    for name, (b, scramble, expect_free) in builders.items():
        r = asof.leakage_check(b, pbp, sched, pick, scramble_starters=scramble)
        free = bool(r["leak_free"].all())
        good = free == expect_free
        ok &= good
        out[name] = {"leak_free_games": int(r["leak_free"].sum()), "games": int(len(r)), "expected_leak_free":
                     expect_free, "as_expected": good}
        print(f"  {name:<28} leak-free {int(r['leak_free'].sum())}/{len(r)}  "
              f"{'as expected' if good else 'UNEXPECTED'}{'' if expect_free else ' (positive control)'}")
    print(f"{'PASS' if ok else 'FAIL'} ({time.perf_counter() - t0:.0f}s)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "leakcheck.json").write_text(json.dumps({"games": pick, "results": out, "pass": ok}, indent=2))
    return 0 if ok else 1


# --------------------------------------------------------------------------- stage: sign-off

# Frozen for the holdout pre-registration (context/ml.md, "M3b holdout pre-registration"): the one-SE pick on DEV.
CHOSEN = spec("C2d", ["elo_logit", "kf_margin", "qb_delta_diff"],
              note="A4s with adj_epa_margin replaced by the Kalman filter's kf_margin")
SIGNOFF = {"chosen": CHOSEN,
           "A4s": spec("A4s", A4S),
           "twin": spec(CHOSEN["step"] + "_last", CHOSEN["features"], qbv=CHOSEN["qb"], w0=CHOSEN["w0"]),
           "A4bs": spec("A4bs", A4S)}
LAST = {"twin", "A4bs"}  # scored with last game's starter (Wednesday rule)


def holdout_signoff_runs() -> list[str]:
    hits = []
    for path in glob.glob(str(registry.RUNS_DIR / "*_m3b_signoff_*.json")):
        if json.loads(Path(path).read_text()).get("holdout"):
            hits.append(Path(path).name)
    return sorted(hits)


def stage_signoff(window: tuple[int, int], allow_holdout: bool, log: bool) -> int:
    """The pre-registered sign-off. On DEV without logging it is the dry run, which must reproduce `dev`."""
    label = "holdout" if windows.touches_holdout(window) else "dev"
    if label == "holdout":
        prior = holdout_signoff_runs()
        if prior:
            raise SystemExit(f"an M3b holdout sign-off is already recorded ({prior}); it is run once only")
    print(f"== M3b sign-off ({label}), REG {window[0]}-{window[1]} with moneylines; frozen model "
          f"{CHOSEN['step']} = {' + '.join(CHOSEN['features'])}; Kalman {kf.TUNED.to_dict()} ==")
    frame, mask, ft = prepare(window, allow_holdout)
    sub = frame[mask]
    y = sub["y"].to_numpy()
    gh = registry.games_hash(sub["game_id"])
    print(f"n = {len(y)}, games hash {gh}")
    if label == "holdout" and (len(y) != HOLDOUT_N or gh != HOLDOUT_HASH):
        raise SystemExit(f"holdout game set differs from the pre-registration (n {HOLDOUT_N}, hash {HOLDOUT_HASH})")
    preds, coefs = {}, {}
    for k, sp in SIGNOFF.items():
        p, c = predict(frame, sp, window, last=k in LAST)
        preds[k], coefs[k] = p[mask].to_numpy(), c
    preds["elo_v2"] = sub["p_elo_v2"].to_numpy()
    preds["market"] = sub[baselines.MARKET_COL].to_numpy()
    res = {k: score(y, v) for k, v in preds.items()}
    print(f"{'model':<10}{'n':>6}{'brier':>9}{'logloss':>9}{'acc':>8}{'ece':>8}{'ece p95':>9}")
    for k, m in res.items():
        print(f"{k:<10}{m['n']:>6}{m['brier']:>9.4f}{m['logloss']:>9.4f}{m['accuracy']:>8.3f}{m['ece']:>8.4f}"
              f"{m['ece_null_p95']:>9.4f}")
    comp = {"chosen_minus_A4s_brier": ci(y, preds["chosen"], preds["A4s"]),
            "chosen_minus_A4s_logloss": ci(y, preds["chosen"], preds["A4s"], "logloss"),
            "chosen_minus_elo_brier": ci(y, preds["chosen"], preds["elo_v2"]),
            "chosen_minus_market_brier": ci(y, preds["chosen"], preds["market"]),
            "twin_minus_A4bs_brier": ci(y, preds["twin"], preds["A4bs"]),
            "twin_minus_A4s_brier": ci(y, preds["twin"], preds["A4s"])}
    print("\nPaired bootstrap 95% CIs (2,000 reps, seed 20261003), negative = first model better:")
    for k, c in comp.items():
        print(f"  {k:<28}{fmt_ci(c)}")
    prim = comp["chosen_minus_A4s_brier"]
    primary = bool(prim["ci_high"] < 0 and res["chosen"]["ece_pass"])
    print(f"PRIMARY ({CHOSEN['step']} minus A4s Brier, CI upper bound < 0, and ECE {res['chosen']['ece']:.4f} <= "
          f"null p95 {res['chosen']['ece_null_p95']:.4f}): {'PASS' if primary else 'FAIL'}")
    sp = splits(y, preds["chosen"], preds["A4s"], preds["market"], sub)
    print("\nSecondaries (descriptive):")
    for k in ("weeks_1_4", "changed", "unchanged", "weeks_5_9", "weeks_10_18"):
        v = sp[k]
        print(f"  {k:<12} n {v['n']:>5}  chosen {v['brier']:.4f}  A4s {v['brier_a4s']:.4f}  market "
              f"{v['brier_market']:.4f}  chosen-A4s {fmt_ci(v['vs_a4s'])}  chosen-market {fmt_ci(v['vs_market'])}")
    per = pd.DataFrame({"season": sub["season"].to_numpy(), "y": y, **{k: preds[k] for k in ("chosen", "A4s", "market")}})
    seasons = []
    for S, g in per.groupby("season"):
        b = {k: float(((g["y"] - g[k]) ** 2).mean()) for k in ("chosen", "A4s", "market")}
        seasons.append({"season": int(S), "n": int(len(g)), **b})
    print("  per season: " + "; ".join(f"{r['season']} {r['chosen'] - r['A4s']:+.4f}" for r in seasons))
    last = coefs["chosen"].iloc[-1]
    print(f"  {CHOSEN['step']} coefficients, {int(last['season'])} fit: " + ", ".join(
        f"{f} {last[f]:+.3f}" for f in ["intercept", *CHOSEN["features"]]))

    if label != "holdout":
        dev_json = OUT_DIR / "dev.json"
        if not dev_json.exists():
            print("no dev.json to compare with; run `dev` first", file=sys.stderr)
            return 1
        ref = json.loads(dev_json.read_text())
        want = {"chosen": ref["brier"].get(CHOSEN["step"]), "A4s": ref["brier"].get("A4s"),
                "twin": ref["brier"].get(CHOSEN["step"] + "_last"), "A4bs": ref["brier"].get("A4bs")}
        diffs = {k: abs(res[k]["brier"] - v) for k, v in want.items() if v is not None}
        same = ref.get("chosen") == CHOSEN["step"] and len(diffs) == 4 and max(diffs.values()) < 1e-12
        print(f"\nDry run vs `dev` ({dev_json.name}): max |Brier diff| {max(diffs.values()) if diffs else float('nan'):.2e}"
              f" -> {'REPRODUCED' if same else 'MISMATCH'}")
        return 0 if same else 1

    out = {"primary_pass": primary, "results": res, "comparisons": comp, "splits": sp, "per_season": seasons,
           "games_hash": gh, "n": int(len(y))}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "signoff_holdout.json").write_text(json.dumps(out, indent=2, default=float))
    if log:
        fp = registry.data_fingerprint(games_csv=True, ml_datasets=("pbp", "schedules", "players"),
                                       seasons=list(range(ml_m3.DATA_SEASONS[0], window[1] + 1)) + [0])
        params = {"ratings": oa.TUNED.to_dict(), "qb": qb.tuned_config().to_dict(), "kalman": kf.TUNED.to_dict(),
                  "train_start": TRAIN_START, "season_half_life": logistic.SEASON_HALF_LIFE,
                  "protocol": "context/ml.md, M3b holdout pre-registration; one run"}
        for k, m in res.items():
            sp_ = SIGNOFF.get(k)
            registry.log_run(
                f"m3b_signoff_{k}", label=label, seasons=window, game_ids=sub["game_id"],
                metrics={x: m[x] for x in ("n", "brier", "logloss", "accuracy", "ece", "ece_null_p95", "ece_pass")},
                features=swap_qb(sp_["features"], sp_["qb"], k in LAST) if sp_ else [],
                params=params if sp_ else {}, data=fp, notes=f"M3b sign-off: {k}",
                extra={"signoff": "m3b_signoff", "primary_pass": primary if k == "chosen" else None,
                       "comparisons": comp if k == "chosen" else None, "splits": sp if k == "chosen" else None,
                       "per_season": seasons if k == "chosen" else None,
                       "coefficients": coefs[k].to_dict("records") if k in coefs else None})
        print(f"\n{len(res)} runs written to {registry.RUNS_DIR.relative_to(config.ROOT)}/")
    return 0


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["tune", "dev", "leakcheck", "signoff-dry-run", "signoff"])
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    if a.stage == "tune":
        return stage_tune(not a.no_log)
    if a.stage == "dev":
        return stage_dev(not a.no_log)
    if a.stage == "leakcheck":
        return stage_leakcheck()
    if a.stage == "signoff-dry-run":
        return stage_signoff(DEV, allow_holdout=False, log=False)
    return stage_signoff(windows.HOLDOUT, allow_holdout=True, log=not a.no_log)


if __name__ == "__main__":
    raise SystemExit(main())
