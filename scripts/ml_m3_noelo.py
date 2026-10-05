"""No-Elo study: can the M3 features stand on their own, without Elo's log-odds? (descriptive, DEV only)

    python scripts/ml_m3_noelo.py            # run N1-N6, diagnostics, log one registry run per variant
    python scripts/ml_m3_noelo.py --no-log   # same, without writing run files

Same machinery as scripts/ml_m3.py: the same feature frame, walk-forward fit
(REG 2001..S-1), season-decay weights, frozen rating knobs, ties as two half
rows, and the same 3,450 DEV games (2006-2019 REG with moneylines). Nothing
here touches 2020-2025 or changes the live model.

    N1  adj_epa_margin                                        (the intercept carries home field)
    N2  adj_epa_margin + qb_delta_diff                        (A4s without Elo)
    N3  N2 + rest_diff + neutral
    N4  adj_pass_margin + adj_rush_margin + qb_delta_diff + rest_diff + neutral
    N5  boosted trees (A6 settings, monotone) on the N4 features
    N6  N2 + raw_epa_margin (M1, unadjusted)

Runs are logged as `m3_noelo_<id>` (label dev, extra "study": "no_elo"). The
best variant's run also carries the week-bucket and Elo-overlap diagnostics.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ml_m3  # noqa: E402
from nflelo import config  # noqa: E402
from nflelo.ml import registry  # noqa: E402
from nflelo.ml.eval import baselines, bootstrap, calibration, windows  # noqa: E402
from nflelo.ml.features import opponent_adjust as oa, qb  # noqa: E402
from nflelo.ml.models import logistic  # noqa: E402

DEV = windows.DEV
EXPECTED_GAMES_HASH = "52b8194df3a2ebf4"
A4S_FEATURES = ml_m3.SIGNOFF_STEPS["A4s"]
N4_FEATURES = ["adj_pass_margin", "adj_rush_margin", "qb_delta_diff", "rest_diff", "neutral"]
VARIANTS = [
    {"step": "N1", "features": ["adj_epa_margin"], "kind": "logistic"},
    {"step": "N2", "features": ["adj_epa_margin", "qb_delta_diff"], "kind": "logistic"},
    {"step": "N3", "features": ["adj_epa_margin", "qb_delta_diff", "rest_diff", "neutral"], "kind": "logistic"},
    {"step": "N4", "features": N4_FEATURES, "kind": "logistic"},
    {"step": "N5", "features": N4_FEATURES, "kind": "gbm"},
    {"step": "N6", "features": ["adj_epa_margin", "qb_delta_diff", "raw_epa_margin"], "kind": "logistic"},
]
BUCKETS = [("1-4", 1, 4), ("5-9", 5, 9), ("10-18", 10, 18)]
BOOT_COEF_REPS = 200


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2))


def bucket_of(week) -> np.ndarray:
    w = np.asarray(week, int)
    return np.select([(w >= lo) & (w <= hi) for _, lo, hi in BUCKETS], [b for b, _, _ in BUCKETS], "other")


def bucket_coefs(frame: pd.DataFrame) -> dict:
    """A4s refit on training rows from one week bucket only (walk-forward, same weights). Descriptive."""
    out = {}
    for name, lo, hi in BUCKETS:
        sub = frame[frame["week"].between(lo, hi)].reset_index(drop=True)
        _, c = logistic.walk_forward(sub, A4S_FEATURES, DEV, train_start=ml_m3.TRAIN_START)
        last = c.iloc[-1]
        # Bootstrap SE of the 2019 fit (training games resampled with replacement).
        tr = sub[(sub["season"] >= ml_m3.TRAIN_START) & (sub["season"] < DEV[1])]
        X, yy = tr[A4S_FEATURES].to_numpy(float), tr["y"].to_numpy(float)
        w = logistic.season_weights(tr["season"], DEV[1])
        rng = np.random.default_rng(20261003)
        draws = []
        for _ in range(BOOT_COEF_REPS):
            i = rng.integers(0, len(tr), len(tr))
            m = logistic.fit_logistic(X[i], yy[i], w[i])
            draws.append([m.intercept_[0], *m.coef_[0]])
        se = np.std(draws, axis=0, ddof=1)
        out[name] = {"fit_2019": {k: float(last[k]) for k in ["intercept", *A4S_FEATURES]},
                     "boot_se_2019": {k: float(v) for k, v in zip(["intercept", *A4S_FEATURES], se)},
                     "boot_reps": BOOT_COEF_REPS,
                     "n_train_2019": int(last["n_train"]),
                     "median_over_refits": {k: float(c[k].median()) for k in ["intercept", *A4S_FEATURES]},
                     "range_over_refits": {k: [float(c[k].min()), float(c[k].max())] for k in A4S_FEATURES}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-log", action="store_true", help="print results without writing run files")
    a = ap.parse_args()
    windows.check(DEV, allow_holdout=False)

    games, pbp, sched = ml_m3.load_inputs()                      # data stops at 2019
    base = baselines.baseline_frame(games, DEV, require_market=True)
    frame, ft = ml_m3.build_frame(games, pbp, sched)
    ft.pop("ratings_obj")
    frame = frame.merge(base[["game_id", "p_elo_v2"]], on="game_id", how="left")
    mask = frame["game_id"].isin(base["game_id"]).to_numpy()
    dev = frame[mask]
    gh = registry.games_hash(dev["game_id"])
    if mask.sum() != 3450 or gh != EXPECTED_GAMES_HASH:
        raise SystemExit(f"DEV games changed: n = {mask.sum()}, hash {gh} (expected 3450, {EXPECTED_GAMES_HASH})")
    y = dev["y"].to_numpy()
    week = dev["week"].to_numpy()
    elo_p = dev["p_elo_v2"].to_numpy()

    # Reference: A4s, recomputed with the ladder code (must reproduce 0.2151).
    p_a4s, _ = ml_m3.run_step(frame, {"step": "A4s", "features": A4S_FEATURES, "kind": "logistic"})
    a4s = p_a4s[mask].to_numpy()
    print(f"DEV {DEV[0]}-{DEV[1]}: n = {len(dev)}, games hash {gh}. "
          f"Elo v2 Brier {brier(y, elo_p):.4f}; A4s recomputed {brier(y, a4s):.4f}.")

    results, preds, coefs = {}, {}, {}
    for spec in VARIANTS:
        p, c = ml_m3.run_step(frame, spec)
        pd_ = p[mask].to_numpy()
        preds[spec["step"]], coefs[spec["step"]] = pd_, c
        m = ml_m3.score(y, pd_)
        null = calibration.ece_null_range(pd_)
        m["ece_null"] = null
        m["ece_ok"] = bool(m["ece"] <= null["p95"])               # one-sided, as in M4
        m["vs_elo_brier"] = bootstrap.paired_bootstrap(y, pd_, elo_p, "brier")
        m["vs_A4s_brier"] = bootstrap.paired_bootstrap(y, pd_, a4s, "brier")
        results[spec["step"]] = m

    hdr = (f"{'id':<4}{'model':<58}{'brier':>8}{'logloss':>9}{'acc':>7}{'ece':>7}{'p95':>7}  "
           f"{'vs Elo [95% CI]':<28}{'vs A4s [95% CI]':<28}")
    print("\n" + hdr)
    for spec in VARIANTS:
        s, m = spec["step"], results[spec["step"]]
        e, b = m["vs_elo_brier"], m["vs_A4s_brier"]
        name = ("GBM: " if spec["kind"] == "gbm" else "") + " + ".join(spec["features"])
        print(f"{s:<4}{name[:57]:<58}{m['brier']:>8.4f}{m['logloss']:>9.4f}{m['accuracy']:>7.3f}{m['ece']:>7.4f}"
              f"{m['ece_null']['p95']:>7.4f}  {e['diff']:+.4f} [{e['ci_low']:+.4f}, {e['ci_high']:+.4f}]  "
              f"{b['diff']:+.4f} [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]")

    # ---- diagnostics for the best no-Elo model
    best = min(results, key=lambda s: results[s]["brier"])
    pb = preds[best]
    bk = bucket_of(week)
    by_bucket = []
    for name, _, _ in BUCKETS:
        m_ = bk == name
        row = {"bucket": name, "n": int(m_.sum()), f"brier_{best}": brier(y[m_], pb[m_]),
               "brier_A4s": brier(y[m_], a4s[m_]), "brier_elo_v2": brier(y[m_], elo_p[m_]),
               "brier_N2": brier(y[m_], preds["N2"][m_])}
        row["best_minus_elo"] = bootstrap.paired_bootstrap(y[m_], pb[m_], elo_p[m_], "brier")
        row["A4s_minus_N2"] = bootstrap.paired_bootstrap(y[m_], a4s[m_], preds["N2"][m_], "brier")
        by_bucket.append(row)
    elo_logit = dev["elo_logit"].to_numpy()
    corr = {"best_vs_elo_logit": float(np.corrcoef(logit(pb), elo_logit)[0, 1]),
            "best_vs_A4s_logit": float(np.corrcoef(logit(pb), logit(a4s))[0, 1]),
            "adj_epa_margin_vs_elo_logit": float(np.corrcoef(dev["adj_epa_margin"], elo_logit)[0, 1])}
    corr_by_bucket = {name: float(np.corrcoef(logit(pb[bk == name]), elo_logit[bk == name])[0, 1])
                      for name, _, _ in BUCKETS}
    a4s_n2 = bootstrap.paired_bootstrap(y, a4s, preds["N2"], "brier")
    bc = bucket_coefs(frame)

    print(f"\nBest no-Elo model: {best} (Brier {results[best]['brier']:.4f}).")
    print(f"(a) Brier by week bucket:")
    print(f"  {'weeks':<7}{'n':>6}{best:>9}{'N2':>9}{'A4s':>9}{'Elo':>9}  {best + ' - Elo [95% CI]':<30}"
          f"{'A4s - N2 [95% CI]':<28}")
    for r in by_bucket:
        e, d = r["best_minus_elo"], r["A4s_minus_N2"]
        print(f"  {r['bucket']:<7}{r['n']:>6}{r[f'brier_{best}']:>9.4f}{r['brier_N2']:>9.4f}{r['brier_A4s']:>9.4f}"
              f"{r['brier_elo_v2']:>9.4f}  {e['diff']:+.4f} [{e['ci_low']:+.4f}, {e['ci_high']:+.4f}]    "
              f"{d['diff']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]")
    print(f"  all    A4s - N2: {a4s_n2['diff']:+.4f} [{a4s_n2['ci_low']:+.4f}, {a4s_n2['ci_high']:+.4f}]")
    print(f"(b) corr(logit p_{best}, elo_logit) = {corr['best_vs_elo_logit']:.3f}; by bucket "
          + ", ".join(f"{k} {v:.3f}" for k, v in corr_by_bucket.items())
          + f"; corr with A4s logit {corr['best_vs_A4s_logit']:.3f}; adj_epa_margin vs elo_logit "
          f"{corr['adj_epa_margin_vs_elo_logit']:.3f}")
    print("(c) A4s refit on one week bucket's training rows (2019 fit ± bootstrap SE; median over the 14 refits):")
    for name, v in bc.items():
        f, se, md = v["fit_2019"], v["boot_se_2019"], v["median_over_refits"]
        print(f"  weeks {name:<6} n_train {v['n_train_2019']:>5}  " + "  ".join(
            f"{k} {f[k]:+.3f} ±{se[k]:.3f} ({md[k]:+.3f})" for k in ["intercept", *A4S_FEATURES]))

    if not a.no_log:
        fp = registry.data_fingerprint(games_csv=True, ml_datasets=("pbp", "schedules"),
                                       seasons=range(ml_m3.DATA_SEASONS[0], DEV[1] + 1))
        params_common = {"ratings": oa.TUNED.to_dict(), "qb": qb.tuned_config().to_dict(),
                         "train_start": ml_m3.TRAIN_START, "season_half_life": logistic.SEASON_HALF_LIFE,
                         "ties": "two half-weight rows", "elo_config": config.DEFAULT_CONFIG.to_dict()}
        diagnostics = {"best": best, "by_week_bucket": by_bucket, "correlations": corr,
                       "corr_best_vs_elo_logit_by_bucket": corr_by_bucket, "A4s_minus_N2_brier": a4s_n2,
                       "a4s_coefs_by_week_bucket": bc,
                       "references": {"elo_v2_brier": brier(y, elo_p), "A4s_brier": brier(y, a4s)}}
        for spec in VARIANTS:
            s, m = spec["step"], results[spec["step"]]
            params = dict(params_common, model=spec["kind"])
            if spec["kind"] == "gbm":
                params["gbm"] = dict(ml_m3.GBM_PARAMS, library="sklearn HistGradientBoostingClassifier",
                                     monotone={f: ml_m3.MONOTONE[f] for f in spec["features"]})
            registry.log_run(
                f"m3_noelo_{s}", label="dev", seasons=DEV, game_ids=dev["game_id"],
                metrics={k: m[k] for k in ("n", "brier", "logloss", "accuracy", "ece")},
                features=spec["features"], params=params, data=fp,
                notes="No-Elo study (descriptive): M3 features without elo_logit. Changes nothing live.",
                extra={"study": "no_elo", "ece_null": m["ece_null"], "ece_ok": m["ece_ok"],
                       "comparisons": {"vs_elo_brier": m["vs_elo_brier"], "vs_A4s_brier": m["vs_A4s_brier"]},
                       "coefficients": coefs[s].to_dict("records"),
                       "diagnostics": diagnostics if s == best else None})
        print(f"\n{len(VARIANTS)} runs written to {registry.RUNS_DIR.relative_to(config.ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
