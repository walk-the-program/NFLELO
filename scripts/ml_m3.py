"""Run the M3 experiment ladder A0-A6 on the development window and pick the final model.

    python scripts/ml_m3.py            # build features, run the ladder, log every step, print the table
    python scripts/ml_m3.py --no-log   # same, without writing run files

Spec: context/ml-m3-method.md (sections 6 and 8). Every step is scored on the
same games as scripts/ml_baselines.py: the 3,450 DEV (2006-2019) REG games that
have moneylines. Training is walk-forward: season S is predicted by a model fit
on REG games 2001..S-1 (all of them, lines or not), with season-decay weights.

    A0   logistic on elo_logit                       (must land within 0.001 of Elo's 0.2178)
    A1   A0 + raw EPA margin (M1 features)
    A2   A0 + opponent-adjusted EPA margin
    A3   A0 + adjusted pass and rush margins
    A4   best of A2/A3 + QB deltas, actual starter    (M3-D1)
    A4s  best of A2/A3 + qb_delta_diff (home minus away; one symmetric QB coefficient)
    A4b  same as A4, last week's starter              (strict Wednesday rule)
    A4bs same as A4s, last week's starter
    A5   A4 + rest_diff + neutral
    A6   boosted trees on the A5 features, monotonic constraints

The final model is picked by the one-standard-error rule (M3-D3): among steps
whose DEV Brier is within one paired-bootstrap standard error of the best,
take the simplest (fewest features; trees count as most complex).

The holdout (2020-2025) is never loaded: data stops at 2019.

Outputs: one registry run per step (experiments/runs/), and the feature frame
with every step's predictions in data/raw/ml/m3/ (gitignored) for notebook 03.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nflelo import config  # noqa: E402
from nflelo import data as elo_data  # noqa: E402
from nflelo.evaluate import outcome  # noqa: E402
from nflelo.ml import asof, registry  # noqa: E402
from nflelo.ml import data as mldata  # noqa: E402
from nflelo.ml.eval import baselines, bootstrap, calibration, metrics, windows  # noqa: E402
from nflelo.ml.features import context, opponent_adjust as oa, qb  # noqa: E402
from nflelo.ml.features import team_efficiency as te  # noqa: E402
from nflelo.ml.models import logistic  # noqa: E402

DEV = windows.DEV
DATA_SEASONS = (1999, DEV[1])
TRAIN_START = logistic.TRAIN_START
ELO_DEV_BRIER = 0.2178
A0_TOLERANCE = 0.001
LEAK_ALARM = 0.213
OUT_DIR = mldata.ML_DIR / "m3"
PBP_COLUMNS = ["game_id", "posteam", "defteam", "play_type", "pass", "rush", "epa", "success", "wp",
               "two_point_attempt", "qb_dropback", "passer_id", "play_id"]
GBM_PARAMS = dict(max_iter=300, learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=80,
                  l2_regularization=1.0, early_stopping=False, random_state=20261003)
# Signs a feature's effect on the home win probability must have in A6 (1 up, -1 down, 0 free).
MONOTONE = {"elo_logit": 1, "adj_epa_margin": 1, "adj_pass_margin": 1, "adj_rush_margin": 1,
            "qb_delta_home": 1, "qb_delta_away": -1, "qb_delta_diff": 1, "rest_diff": 1, "neutral": 0}


# --------------------------------------------------------------------------- data and features

def load_inputs():
    games = elo_data.load_games()
    games = games[games["season"] <= DEV[1]].reset_index(drop=True)  # nothing from the holdout
    years = range(DATA_SEASONS[0], DATA_SEASONS[1] + 1)
    pbp = mldata.load_pbp(years, columns=PBP_COLUMNS)
    sched = asof.add_asof(mldata.load_schedules(years))
    return games, pbp, sched


def data_key(years) -> str:
    h = mldata.manifest_hashes(("pbp", "schedules"), years)
    return registry.games_hash([f"{k}:{v}" for k, v in h.items()])


def build_frame(games: pd.DataFrame, pbp: pd.DataFrame, sched: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One row per REG game 2001-2019: outcome plus every ladder feature. Returns (frame, timings)."""
    t = {}
    reg = sched[(sched["game_type"] == "REG") & sched["season"].between(TRAIN_START, DEV[1])]
    reg = reg[reg["home_score"].notna()].reset_index(drop=True)
    frame = reg[["game_id", "season", "week", "home_team", "away_team"]].copy()
    frame["y"] = outcome(reg)

    t0 = time.perf_counter()
    elo = context.elo_features_from_games(games)
    missing = set(reg["game_id"]) - set(elo.index)
    if missing:
        raise SystemExit(f"{len(missing)} schedule games are not in data/games.csv, e.g. {sorted(missing)[:3]}")
    frame["elo_logit"] = elo.loc[reg["game_id"], "elo_logit"].to_numpy()
    t["elo"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    raw = te.build_features(pbp, sched, reg)
    margin = ((raw["home_off_epa_all"] + raw["away_def_epa_all"]) - (raw["away_off_epa_all"] + raw["home_def_epa_all"]))
    frame["raw_epa_margin"] = margin.fillna(0.0).to_numpy()
    t["raw_epa"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tab = oa.rating_table(pbp, sched, oa.TUNED)
    keys = sorted(set(zip(reg["season"].astype(int), reg["week"].astype(int))))
    ratings = oa.cached_ratings(tab, sched, keys, oa.TUNED, ("all", "pass", "rush"),
                                data_key(range(DATA_SEASONS[0], DATA_SEASONS[1] + 1)))
    for kind in ("all", "pass", "rush"):
        frame[oa.FEATURES[kind]] = oa.matchup_margin(ratings, reg, kind)
    t["ratings"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    qa = qb.build_features(pbp, sched, reg, qb.tuned_config(), starter="actual")
    ql = qb.build_features(pbp, sched, reg, qb.tuned_config(), starter="last")
    frame["qb_delta_home"], frame["qb_delta_away"] = qa["qb_delta_home"].to_numpy(), qa["qb_delta_away"].to_numpy()
    frame["qb_last_delta_home"] = ql["qb_delta_home"].to_numpy()
    frame["qb_last_delta_away"] = ql["qb_delta_away"].to_numpy()
    # Symmetric versions (A4s, A4bs): one coefficient for "home QB news minus away QB news".
    frame["qb_delta_diff"] = frame["qb_delta_home"] - frame["qb_delta_away"]
    frame["qb_last_delta_diff"] = frame["qb_last_delta_home"] - frame["qb_last_delta_away"]
    t["qb"] = time.perf_counter() - t0

    ctx = context.build_features(pbp, sched, reg)
    frame["rest_diff"], frame["neutral"] = ctx["rest_diff"].to_numpy(), ctx["neutral"].to_numpy()
    feats = [c for c in frame.columns if c not in ("game_id", "season", "week", "home_team", "away_team", "y")]
    te.assert_no_market_columns(frame[feats])  # decision D3
    if frame[feats].isna().any().any():
        raise SystemExit(f"NaN in features: {frame[feats].isna().sum()[lambda s: s > 0].to_dict()}")
    return frame, t | {"ratings_obj": ratings}


# --------------------------------------------------------------------------- models

def fit_gbm(features):
    from sklearn.ensemble import HistGradientBoostingClassifier

    def fitter(X, y, w):
        X2, y2, w2 = logistic.expand_ties(X, y, w)
        m = HistGradientBoostingClassifier(monotonic_cst=[MONOTONE[f] for f in features], **GBM_PARAMS)
        m.fit(X2, y2.astype(int), sample_weight=w2)
        return m
    return fitter


def ladder_specs(a2_or_a3: list[str]) -> list[dict]:
    qa, ql = ["qb_delta_home", "qb_delta_away"], ["qb_last_delta_home", "qb_last_delta_away"]
    base = ["elo_logit"]
    return [
        {"step": "A4", "features": base + a2_or_a3 + qa, "kind": "logistic"},
        {"step": "A4s", "features": base + a2_or_a3 + ["qb_delta_diff"], "kind": "logistic"},
        {"step": "A4b", "features": base + a2_or_a3 + ql, "kind": "logistic"},
        {"step": "A4bs", "features": base + a2_or_a3 + ["qb_last_delta_diff"], "kind": "logistic"},
        {"step": "A5", "features": base + a2_or_a3 + qa + ["rest_diff", "neutral"], "kind": "logistic"},
        {"step": "A6", "features": base + a2_or_a3 + qa + ["rest_diff", "neutral"], "kind": "gbm"},
    ]


FIRST_STEPS = [
    {"step": "A0", "features": ["elo_logit"], "kind": "logistic"},
    {"step": "A1", "features": ["elo_logit", "raw_epa_margin"], "kind": "logistic"},
    {"step": "A2", "features": ["elo_logit", "adj_epa_margin"], "kind": "logistic"},
    {"step": "A3", "features": ["elo_logit", "adj_pass_margin", "adj_rush_margin"], "kind": "logistic"},
]


def run_step(frame: pd.DataFrame, spec: dict) -> tuple[pd.Series, pd.DataFrame]:
    feats = spec["features"]
    fitter = fit_gbm(feats) if spec["kind"] == "gbm" else logistic.fit_logistic
    return logistic.walk_forward(frame, feats, DEV, train_start=TRAIN_START, fitter=fitter)


def complexity(spec: dict) -> int:
    return len(spec["features"]) + (100 if spec["kind"] == "gbm" else 0)


def score(y, p) -> dict:
    return {**metrics.win_metrics(y, p), "ece": calibration.ece(y, p)}


# --------------------------------------------------------------------------- weekly update timing

def time_weekly_update(games, pbp, sched, frame, spec, season=2019, week=10) -> dict:
    """Wall time of one Wednesday update for (season, week): ratings, QB values, Elo, then predict."""
    wk = sched[(sched["season"] == season) & (sched["week"] == week) & (sched["game_type"] == "REG")]
    train = frame[(frame["season"] >= TRAIN_START) & (frame["season"] < season)]
    t0 = time.perf_counter()
    model = logistic.fit_logistic(train[spec["features"]].to_numpy(float), train["y"].to_numpy(float),
                                  logistic.season_weights(train["season"], season))
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    elo = context.elo_features_from_games(games)
    tab = oa.rating_table(pbp, sched, oa.TUNED)
    kinds = ("pass", "rush") if "adj_pass_margin" in spec["features"] else ("all",)
    r = oa.compute_ratings(tab, sched, [(season, week)], oa.TUNED, kinds)
    row = pd.DataFrame({"elo_logit": elo.loc[wk["game_id"], "elo_logit"].to_numpy()})
    for k in kinds:
        row[oa.FEATURES[k]] = oa.matchup_margin(r, wk, k)
    qd = qb.build_features(pbp, sched, wk, qb.tuned_config(), starter="actual")
    row["qb_delta_home"], row["qb_delta_away"] = qd["qb_delta_home"].to_numpy(), qd["qb_delta_away"].to_numpy()
    row["qb_delta_diff"] = row["qb_delta_home"] - row["qb_delta_away"]
    ctx = context.build_features(pbp, sched, wk)
    row["rest_diff"], row["neutral"] = ctx["rest_diff"].to_numpy(), ctx["neutral"].to_numpy()
    p = model.predict_proba(row[spec["features"]].to_numpy(float))[:, 1]
    t_week = time.perf_counter() - t0
    check = frame.set_index("game_id").loc[wk["game_id"], spec["features"]].to_numpy(float)
    return {"season": season, "week": week, "games": int(len(wk)), "update_s": t_week, "season_refit_s": t_fit,
            "features_match_ladder": bool(np.allclose(check, row[spec["features"]].to_numpy(float))),
            "p_mean": float(p.mean())}


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-log", action="store_true", help="print results without writing run files")
    a = ap.parse_args()
    t_start = time.perf_counter()

    games, pbp, sched = load_inputs()
    t_load = time.perf_counter() - t_start
    base = baselines.baseline_frame(games, DEV, require_market=True)
    t0 = time.perf_counter()
    frame, ft = build_frame(games, pbp, sched)
    ratings = ft.pop("ratings_obj")
    t_features = time.perf_counter() - t0
    frame = frame.merge(base[["game_id", "p_elo_v2", baselines.MARKET_COL]], on="game_id", how="left")
    dev_mask = frame["game_id"].isin(base["game_id"]).to_numpy()
    if dev_mask.sum() != len(base):
        raise SystemExit(f"only {dev_mask.sum()} of {len(base)} DEV games found in the feature frame")
    dev = frame[dev_mask]
    y = dev["y"].to_numpy()
    if not np.allclose(y, base.set_index("game_id").loc[dev["game_id"], "y"].to_numpy()):
        raise SystemExit("outcomes disagree between the schedule and data/games.csv")
    elo_p = dev["p_elo_v2"].to_numpy()
    mkt_p = dev[baselines.MARKET_COL].to_numpy()
    elo_m = score(y, elo_p)
    print(f"DEV {DEV[0]}-{DEV[1]}: {len(dev)} REG games with moneylines (training: REG {TRAIN_START}..S-1, "
          f"{int((frame['season'] < DEV[0]).sum())} games before {DEV[0]}).")
    print(f"Elo v2 on these games: Brier {elo_m['brier']:.4f}. Feature build {t_features:.1f}s "
          f"(" + ", ".join(f"{k} {v:.1f}s" for k, v in ft.items()) + ")")

    t_ladder0 = time.perf_counter()
    results, preds, coefs = {}, {}, {}

    def do(spec):
        p, c = run_step(frame, spec)
        preds[spec["step"]] = p
        coefs[spec["step"]] = c
        pd_ = p[dev_mask].to_numpy()
        m = score(y, pd_)
        m["vs_elo_brier"] = bootstrap.paired_bootstrap(y, pd_, elo_p, "brier")
        m["vs_elo_logloss"] = bootstrap.paired_bootstrap(y, pd_, elo_p, "logloss")
        m["vs_market_brier"] = bootstrap.paired_bootstrap(y, pd_, mkt_p, "brier")
        results[spec["step"]] = {"spec": spec, "m": m}
        return m

    m0 = do(FIRST_STEPS[0])
    gap = m0["brier"] - elo_m["brier"]
    print(f"\nA0 check: logistic on elo_logit {m0['brier']:.5f} vs Elo {elo_m['brier']:.5f} "
          f"(diff {gap:+.5f}; tolerance {A0_TOLERANCE})")
    if abs(gap) > A0_TOLERANCE:
        print("A0 does not reproduce Elo. Stopping before the rest of the ladder.", file=sys.stderr)
        return 1
    for spec in FIRST_STEPS[1:]:
        do(spec)
    best23 = min(("A2", "A3"), key=lambda s: results[s]["m"]["brier"])
    extra_feats = [f for f in results[best23]["spec"]["features"] if f != "elo_logit"]
    for spec in ladder_specs(extra_feats):
        do(spec)
    t_ladder = time.perf_counter() - t_ladder0

    # ---- one-SE rule (M3-D3)
    steps = list(results)
    best = min(steps, key=lambda s: results[s]["m"]["brier"])
    pb = preds[best][dev_mask].to_numpy()
    for s in steps:
        ci = bootstrap.paired_bootstrap(y, preds[s][dev_mask].to_numpy(), pb, "brier")
        results[s]["vs_best"] = ci
        results[s]["within_1se"] = bool(ci["diff"] <= ci["boot_se"]) if s != best else True
    eligible = [s for s in steps if results[s]["within_1se"]]
    chosen = min(eligible, key=lambda s: (complexity(results[s]["spec"]), results[s]["m"]["brier"]))

    # ---- table
    print(f"\nLadder, DEV {DEV[0]}-{DEV[1]} (n = {len(dev)}). diff = model minus Elo v2 Brier, "
          "95% paired bootstrap CI (2,000 reps, seed 20261003); 1SE = within one SE of the best.")
    hdr = f"{'step':<5}{'model':<46}{'brier':>8}{'logloss':>9}{'acc':>7}{'ece':>8}  {'diff vs Elo [95% CI]':<28}{'1SE':>5}"
    print(hdr)
    print(f"{'':<5}{'Elo v2 (reference)':<46}{elo_m['brier']:>8.4f}{elo_m['logloss']:>9.4f}{elo_m['accuracy']:>7.3f}"
          f"{elo_m['ece']:>8.4f}")
    for s in steps:
        m, spec = results[s]["m"], results[s]["spec"]
        ci = m["vs_elo_brier"]
        name = ("GBM: " if spec["kind"] == "gbm" else "") + " + ".join(f.replace("_delta", "") for f in spec["features"])
        flag = "yes" if results[s]["within_1se"] else ""
        mark = " *" if s == chosen else ""
        print(f"{s:<5}{name[:45]:<46}{m['brier']:>8.4f}{m['logloss']:>9.4f}{m['accuracy']:>7.3f}{m['ece']:>8.4f}  "
              f"{ci['diff']:+.4f} [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]  {flag:>4}{mark}")
    print(f"Market (benchmark): Brier {score(y, mkt_p)['brier']:.4f}")
    low = [s for s in steps if results[s]["m"]["brier"] < LEAK_ALARM]
    if low:
        print(f"LEAK ALARM: {low} score below {LEAK_ALARM}; investigate before accepting.")

    cm = results[chosen]["m"]
    print(f"\nBest DEV Brier: {best} ({results[best]['m']['brier']:.5f}). Within one SE of it: {eligible}.")
    print(f"Chosen by the one-SE rule: {chosen} ({len(results[chosen]['spec']['features'])} features, "
          f"{results[chosen]['spec']['kind']}).")
    ci = cm["vs_elo_brier"]
    passed = ci["excludes_zero"] and ci["diff"] < 0 and cm["ece"] < 0.02
    print(f"M3 acceptance (beats Elo with CI excluding 0, ECE < 0.02): {'PASS' if passed else 'FAIL'} "
          f"(diff {ci['diff']:+.4f} [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}], ECE {cm['ece']:.4f})")
    for late, early in (("A4", "A4b"), ("A4s", "A4bs")):
        qn = bootstrap.paired_bootstrap(y, preds[late][dev_mask].to_numpy(), preds[early][dev_mask].to_numpy(), "brier")
        print(f"QB news ({late} minus {early} Brier): {qn['diff']:+.4f} [{qn['ci_low']:+.4f}, {qn['ci_high']:+.4f}]")
    sym = bootstrap.paired_bootstrap(y, preds["A4s"][dev_mask].to_numpy(), preds["A4"][dev_mask].to_numpy(), "brier")
    print(f"Symmetric vs separate QB terms (A4s minus A4 Brier): {sym['diff']:+.5f} "
          f"[{sym['ci_low']:+.5f}, {sym['ci_high']:+.5f}]")

    # ---- coefficients of the chosen model
    c = coefs[chosen]
    if "intercept" in c.columns:
        feats = results[chosen]["spec"]["features"]
        last = c.iloc[-1]
        print(f"\nCoefficients of {chosen}, model for {int(last['season'])} (trained on {TRAIN_START}-"
              f"{int(last['season']) - 1}, n = {int(last['n_train'])}); range over the 14 DEV refits in brackets:")
        print(f"  intercept       {last['intercept']:+.4f}  [{c['intercept'].min():+.4f}, {c['intercept'].max():+.4f}]")
        sd = dev[feats].std()
        for f in feats:
            print(f"  {f:<16}{last[f]:+.4f}  [{c[f].min():+.4f}, {c[f].max():+.4f}]   per 1 SD ({sd[f]:.3f}): "
                  f"{last[f] * sd[f]:+.3f} log-odds")
        expected = {f: MONOTONE.get(f.replace("qb_last_delta", "qb_delta"), 0) for f in feats}
        bad = [f for f in feats if expected[f] and np.sign(c[f]).ne(expected[f]).any()]
        print("  sign check:", "all coefficients have the expected sign in every refit" if not bad
              else f"UNEXPECTED SIGN in {bad}")

    timing = {"load_s": t_load, "features_s": t_features, "feature_parts_s": ft, "ladder_fits_s": t_ladder,
              "total_s": time.perf_counter() - t_start}
    wk = time_weekly_update(games, pbp, sched, frame, results[chosen]["spec"] if results[chosen]["spec"]["kind"] ==
                            "logistic" else results["A5"]["spec"])
    timing["weekly_update"] = wk
    print(f"\nTiming: load {t_load:.1f}s, features {t_features:.1f}s, ladder fits {t_ladder:.1f}s, "
          f"total {timing['total_s']:.1f}s. One weekly update ({wk['season']} week {wk['week']}, {wk['games']} games: "
          f"Elo, ratings, QB values, predict): {wk['update_s']:.2f}s, plus {wk['season_refit_s']:.2f}s for the "
          f"once-a-season refit. Features match the ladder: {wk['features_match_ladder']}.")

    # ---- save for notebook 03
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    for s in steps:
        out[f"p_{s}"] = preds[s].to_numpy()
    out["dev"] = dev_mask
    out.to_parquet(OUT_DIR / "ladder_frame.parquet", index=False)
    pd.concat([c.assign(step=s) for s, c in coefs.items()], ignore_index=True).to_parquet(OUT_DIR / "coefs.parquet")
    ratings.to_parquet(OUT_DIR / "ratings.parquet", index=False)

    if not a.no_log:
        params_common = {"ratings": oa.TUNED.to_dict(), "qb": qb.tuned_config().to_dict(),
                         "train_start": TRAIN_START, "season_half_life": logistic.SEASON_HALF_LIFE,
                         "ties": "two half-weight rows", "elo_config": config.DEFAULT_CONFIG.to_dict(),
                         "hfa": "Elo online HFA frozen before the week's first game"}
        fp = registry.data_fingerprint(games_csv=True, ml_datasets=("pbp", "schedules"),
                                       seasons=range(DATA_SEASONS[0], DATA_SEASONS[1] + 1))
        for s in steps:
            spec, m = results[s]["spec"], results[s]["m"]
            params = dict(params_common, model=spec["kind"])
            if spec["kind"] == "gbm":
                params["gbm"] = dict(GBM_PARAMS, library="sklearn HistGradientBoostingClassifier",
                                     monotone={f: MONOTONE[f] for f in spec["features"]})
            mets = {k: m[k] for k in ("n", "brier", "logloss", "accuracy", "ece")}
            notes = {"A0": "Elo log-odds only; must reproduce Elo.",
                     "A4": "QB delta with the actual starter (M3-D1: identity treated as a pre-game fact).",
                     "A4s": "Symmetric QB term: qb_delta_home - qb_delta_away, actual starter.",
                     "A4b": "QB delta with last game's starter (strict Wednesday rule).",
                     "A4bs": "Symmetric QB term with last game's starter (strict Wednesday rule).",
                     "A6": "Boosted trees; sklearn HistGradientBoosting (LightGBM's libomp is missing here)."}.get(s, "")
            registry.log_run(
                f"m3_{s}", label="dev", seasons=DEV, game_ids=dev["game_id"], metrics=mets,
                features=spec["features"], params=params, data=fp,
                notes=(notes + (" CHOSEN by the one-SE rule." if s == chosen else "")).strip(),
                extra={"comparisons": {"vs_elo_brier": m["vs_elo_brier"], "vs_elo_logloss": m["vs_elo_logloss"],
                                       "vs_market_brier": m["vs_market_brier"], "vs_best_brier": results[s]["vs_best"]},
                       "one_se": {"best": best, "within": results[s]["within_1se"], "chosen": s == chosen,
                                  "eligible": eligible},
                       "coefficients": coefs[s].to_dict("records"), "timing": timing if s == chosen else None})
        print(f"\n{len(steps)} runs written to {registry.RUNS_DIR.relative_to(config.ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
