"""M4 Phase 2 experiments: margin model, tiebreaker check, tau_rest tuning, playoff-odds backtest, and the sign-off.

    python scripts/ml_m4.py margin            # margin model, DEV 2006-2019: MAE, CRPS, shape choice, implied Brier, calibration
    python scripts/ml_m4.py tiebreak          # reproduce every actual playoff field and seeding, 2002-2025 (results only)
    python scripts/ml_m4.py tune              # tau_rest on DEV 2006-2019 (one-SE rule, smallest tau), weeks 4, 8, 12, 16
    python scripts/ml_m4.py backtest          # DEV from the same weeks: the frozen tau_rest vs tau_rest = 0 (descriptive)
    python scripts/ml_m4.py all               # margin, tiebreak, tune, backtest in order
    python scripts/ml_m4.py signoff-dry-run   # the sign-off code path on DEV, no logging; must reproduce margin + backtest
    python scripts/ml_m4.py signoff           # the ONE pre-registered holdout run, 2020-2025 (context/ml.md); logs it
    add --no-log to skip the registry (all stages but signoff)

Spec: context/ml-m4-method.md sections 4 and 5; the pre-registration is the
"M4 holdout pre-registration" subsection of context/ml.md. Only `signoff`
loads a 2020-2025 outcome for scoring. The tiebreaker check uses standings and
playoff brackets only, never a model.

Choices fixed before any DEV number was seen:
- Margin mean: weighted least squares on the A4s features, walk-forward, REG
  2001..S-1, season half-life 8 (the M3 protocol). Sigma: the weighted residual
  SD of the same fit.
- Key-number factors: 2001-2005 only, |margin| <= 20, symmetric.
- Shape choice: CRPS on DEV, one-SE rule, the rounded normal is the simpler.

tau_rest history. The first pre-registered tuning (2002-2005) chose 0.5, which
failed on DEV (no gain over 0, made-playoffs ECE outside the null range).
Walker then decided to tune on DEV (decision D2 allows it): grid 0 to 8 by 0.5,
5,000 simulations per start with common random numbers across the grid,
made-playoffs Brier, and the smallest tau within one team-start bootstrap SE of
the best. DEV is then in-sample for tau, so the DEV backtest is descriptive and
the holdout is the test. Backtest and sign-off: 10,000 simulations per start,
the frozen tau against tau = 0 on the same random numbers. Seeds:
SIM_SEED + 1000 * season + start week.

Outputs: registry runs (experiments/runs/), and data/raw/ml/m4/ (gitignored) for notebook 04.
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nflelo import config  # noqa: E402
from nflelo.evaluate import market_prob  # noqa: E402
from nflelo.ml import registry  # noqa: E402
from nflelo.ml import data as mldata  # noqa: E402
from nflelo.ml.eval import baselines, bootstrap, calibration, metrics, windows  # noqa: E402
from nflelo.ml.features import context, opponent_adjust as oa  # noqa: E402
from nflelo.ml.models import logistic, margin as mm  # noqa: E402
from nflelo.ml.sim import history, season as sim, state  # noqa: E402

import ml_m3  # noqa: E402

DEV = windows.DEV
START_WEEKS = (4, 8, 12, 16)
TAU_GRID = tuple(float(x) for x in np.round(np.arange(0.0, 8.01, 0.5), 2))
TUNE_SIMS, BACKTEST_SIMS = 5_000, 10_000
SIM_SEED = 20261004
OUT = mldata.ML_DIR / "m4"
SPREAD_BINS = [-np.inf, -7, -3, 0, 3, 7, np.inf]
OUTCOMES = ("playoffs", "division", "seed1", "reach_sb", "win_sb")
# Pre-registered sign-off rules (context/ml.md, "M4 holdout pre-registration").
BRIER_MARGIN = 0.001          # playoff odds: upper CI bound of Brier(tau) - Brier(0) must be below this
COVER_BINS_REQUIRED = 5       # margin: at least 5 of the 6 spread bins inside the fair-coin band


# --------------------------------------------------------------------------- shared data

_CACHE: dict = {}


def inputs(last: int = DEV[1]):
    """games, pbp, schedules through `last`, and the M3 feature frame 2001..last with margins and market columns."""
    if last not in _CACHE:
        games, pbp, sched = ml_m3.load_inputs(last_season=last)
        frame, _ = ml_m3.build_frame(games, pbp, sched, last_season=last)
        s = sched.set_index("game_id")
        frame["margin"] = (s.loc[frame["game_id"], "home_score"] - s.loc[frame["game_id"], "away_score"]).to_numpy(float)
        # scoring-only columns, joined after the features are built (decision D3)
        frame["spread_market"] = pd.to_numeric(s.loc[frame["game_id"], "spread_line"], errors="coerce").to_numpy(float)
        frame["p_market"] = market_prob(s.loc[frame["game_id"]].reset_index()).to_numpy(float)
        _CACHE[last] = (games, pbp, sched, frame)
    return _CACHE[last]


def margin_fits(frame: pd.DataFrame, seasons: tuple[int, int]) -> dict:
    """Walk-forward margin fits {season: MarginFit} (train REG 2001..S-1)."""
    _, fits = mm.walk_forward(frame, mm.FEATURES, seasons)
    return {f.season: f for f in fits}


def keynums(frame: pd.DataFrame, shape: str):
    return mm.keynumbers_from_frame(frame) if shape == "keynum" else None


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


# --------------------------------------------------------------------------- 1. margin model

def margin_eval(window: tuple[int, int], allow_holdout: bool = False) -> tuple[dict, pd.DataFrame]:
    """Every margin-model number on `window`: REG games with moneylines (the M3 game set), walk-forward fits."""
    t0 = time.perf_counter()
    windows.check(window, allow_holdout)
    games, _, _, frame = inputs(window[1])
    base = baselines.baseline_frame(games, window, allow_holdout=allow_holdout, require_market=True)
    mask = frame["game_id"].isin(base["game_id"]).to_numpy()
    if mask.sum() != len(base):
        raise SystemExit(f"only {mask.sum()} of {len(base)} scored games in the frame")
    wf, fits = mm.walk_forward(frame, mm.FEATURES, window)
    p_a4s, _ = logistic.walk_forward(frame, ml_m3.SIGNOFF_STEPS["A4s"], window)
    kn = mm.keynumbers_from_frame(frame)
    dev = frame[mask].copy()
    dev["mu"], dev["sigma"], dev["p_A4s"] = wf.loc[mask, "mu"], wf.loc[mask, "sigma"], p_a4s[mask]
    dev["spread_elo"] = dev["elo_logit"] / context.LN10_400 / 25.0
    y, mu, sg = dev["margin"].to_numpy(), dev["mu"].to_numpy(), dev["sigma"].to_numpy()
    yw = dev["y"].to_numpy()
    if dev[["spread_market"]].isna().any().any():
        raise SystemExit("a scored game has no market spread")

    e_model, e_elo, e_mkt = np.abs(y - mu), np.abs(y - dev["spread_elo"].to_numpy()), np.abs(y - dev["spread_market"].to_numpy())
    mae_ci = bootstrap.paired_bootstrap_diff(e_model - e_elo)
    mae_mkt_ci = bootstrap.paired_bootstrap_diff(e_model - e_mkt)
    pm = {"normal": mm.normal_pmf(mu, sg), "keynum": mm.keynum_pmf(mu, sg, kn)}
    crps = {"normal_continuous": metrics.crps_normal_per_game(y, mu, sg)}
    crps.update({k: mm.crps_discrete_per_game(y, v) for k, v in pm.items()})
    best = min(("normal", "keynum"), key=lambda k: crps[k].mean())
    other = "keynum" if best == "normal" else "normal"
    se_ci = bootstrap.paired_bootstrap_diff(crps[other] - crps[best])
    within = se_ci["diff"] <= se_ci["boot_se"]
    chosen = "normal" if (best == "normal" or within) else "keynum"

    res = {"window": list(window), "n": int(len(dev)), "games_hash": registry.games_hash(dev["game_id"]),
           "mae": {"model": float(e_model.mean()), "elo": float(e_elo.mean()), "market": float(e_mkt.mean()),
                   "model_minus_elo": mae_ci, "model_minus_market": mae_mkt_ci},
           "crps": {k: float(v.mean()) for k, v in crps.items()},
           "shape_rule": {"best": best, "other_minus_best": se_ci, "within_1se": bool(within), "chosen": chosen},
           "keynum": kn.to_dict(), "shapes": {}}
    for shape, p in pm.items():
        pw = mm.win_prob(p)
        m = ml_m3.score(yw, pw)
        ci = bootstrap.paired_bootstrap(yw, pw, dev["p_A4s"].to_numpy(), "brier")
        null = calibration.ece_null_range(pw)
        res["shapes"][shape] = {**{k: m[k] for k in ("n", "brier", "logloss", "accuracy", "ece")},
                                "crps": float(crps[shape].mean()), "minus_A4s_brier": ci,
                                "within_1se_of_A4s": bool(ci["diff"] <= ci["boot_se"]),
                                "ece_null": null, "ece_ok": bool(m["ece"] <= null["p95"]),
                                "ece_in_null_range": bool(null["p05"] <= m["ece"] <= null["p95"])}
        dev[f"p_{shape}"] = pw
    a4 = ml_m3.score(yw, dev["p_A4s"].to_numpy())
    res["A4s"] = {k: a4[k] for k in ("n", "brier", "logloss", "accuracy", "ece")}
    res["A4s"]["ece_null"] = calibration.ece_null_range(dev["p_A4s"].to_numpy())

    # spread calibration: does the home team beat our spread about half the time in every bin?
    cover = []
    rng = np.random.default_rng(20261003)
    for lo, hi in zip(SPREAD_BINS[:-1], SPREAD_BINS[1:]):
        m = (mu > lo) & (mu <= hi) & (y != mu)
        n = int(m.sum())
        rate = float((y[m] > mu[m]).mean()) if n else float("nan")
        band = np.percentile(rng.binomial(n, 0.5, 20000) / n, [5, 95]) if n else [np.nan, np.nan]
        cover.append({"bin": f"({lo:g}, {hi:g}]", "n": n, "home_covers": rate, "p05": float(band[0]),
                      "p95": float(band[1]), "inside": bool(band[0] <= rate <= band[1])})
    res["cover_by_bin"] = cover
    per = pd.DataFrame({"season": dev["season"].to_numpy(), "em": e_model, "ee": e_elo, "ek": e_mkt})
    res["per_season"] = [{"season": int(s), "n": int(len(g)), "mae_model": float(g["em"].mean()),
                          "mae_elo": float(g["ee"].mean()), "mae_market": float(g["ek"].mean())}
                         for s, g in per.groupby("season")]
    res["coefficients"] = [f.to_dict() for f in fits]
    res["seconds"] = time.perf_counter() - t0
    return res, dev


def print_margin(res: dict, label: str) -> None:
    w = res["window"]
    print(f"== Margin model, {label} {w[0]}-{w[1]}, REG games with moneylines: n = {res['n']} "
          f"(games hash {res['games_hash']}) ==")
    c, ci, cm = res["mae"], res["mae"]["model_minus_elo"], res["mae"]["model_minus_market"]
    print(f"MAE  model {c['model']:.3f}   Elo spread (elo_diff/25) {c['elo']:.3f}   market spread {c['market']:.3f}")
    print(f"     model minus Elo  {ci['diff']:+.3f} [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]  "
          f"{'excludes' if ci['excludes_zero'] else 'includes'} zero")
    print(f"     model minus market {cm['diff']:+.3f} [{cm['ci_low']:+.3f}, {cm['ci_high']:+.3f}]")
    r = res["shape_rule"]
    print("CRPS " + "   ".join(f"{k} {v:.4f}" for k, v in res["crps"].items()))
    print(f"     shape rule: best {r['best']}; other minus best {r['other_minus_best']['diff']:+.4f} "
          f"(SE {r['other_minus_best']['boot_se']:.4f}) -> chosen: {r['chosen']}")
    print(f"{'shape':<8}{'brier':>8}{'logloss':>9}{'acc':>7}{'ece':>7}  {'ECE null 5-95%':<16}{'minus A4s Brier [95% CI]':<30}{'1SE':>4}")
    for k in ("normal", "keynum"):
        s = res["shapes"][k]
        d = s["minus_A4s_brier"]
        print(f"{k:<8}{s['brier']:>8.4f}{s['logloss']:>9.4f}{s['accuracy']:>7.3f}{s['ece']:>7.4f}  "
              f"[{s['ece_null']['p05']:.3f}, {s['ece_null']['p95']:.3f}]  "
              f"{d['diff']:+.5f} [{d['ci_low']:+.5f}, {d['ci_high']:+.5f}]   {'yes' if s['within_1se_of_A4s'] else 'no'}")
    a = res["A4s"]
    print(f"{'A4s':<8}{a['brier']:>8.4f}{a['logloss']:>9.4f}{a['accuracy']:>7.3f}{a['ece']:>7.4f}  "
          f"[{a['ece_null']['p05']:.3f}, {a['ece_null']['p95']:.3f}]")
    print("Home team beats our spread, by predicted spread (5-95% range of a fair coin at that n):")
    for b in res["cover_by_bin"]:
        print(f"  {b['bin']:<12} n {b['n']:>5}  {b['home_covers']:.3f}  [{b['p05']:.3f}, {b['p95']:.3f}]  "
              f"{'inside' if b['inside'] else 'OUTSIDE'}")
    print("MAE by season (model / Elo / market): " + "; ".join(
        f"{p['season']}: {p['mae_model']:.2f} / {p['mae_elo']:.2f} / {p['mae_market']:.2f}" for p in res["per_season"]))
    last = res["coefficients"][-1]
    print(f"{last['season']} fit: intercept {last['intercept']:+.3f}, "
          + ", ".join(f"{k} {v:+.3f}" for k, v in last["coef"].items()) + f", sigma {last['sigma']:.2f}")


def margin_params(kn) -> dict:
    return {"mean": "WLS on A4s features", "features": mm.FEATURES, "train_start": logistic.TRAIN_START,
            "season_half_life": logistic.SEASON_HALF_LIFE, "sigma": "weighted residual SD of the training fit",
            "ratings": oa.TUNED.to_dict(), "keynum": kn.to_dict()}


def run_margin(log: bool) -> dict:
    res, dev = margin_eval(DEV)
    print_margin(res, "DEV")
    OUT.mkdir(parents=True, exist_ok=True)
    dev.to_parquet(OUT / "margin_dev.parquet", index=False)
    (OUT / "margin_dev.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    if log:
        fp = registry.data_fingerprint(games_csv=True, ml_datasets=("pbp", "schedules"), seasons=range(1999, DEV[1] + 1))
        chosen = res["shape_rule"]["chosen"]
        for shape in ("normal", "keynum"):
            s = res["shapes"][shape]
            registry.log_run(
                f"m4_margin_{shape}", label="dev", seasons=DEV, game_ids=dev["game_id"],
                metrics={**{k: s[k] for k in ("n", "brier", "logloss", "accuracy", "ece")}, "crps": s["crps"],
                         "mae": res["mae"]["model"]},
                features=mm.FEATURES, params=dict(margin_params(mm.KeyNumbers.from_dict(res["keynum"])), shape=shape),
                data=fp,
                notes=f"M4 margin model, {shape} shape." + (" CHOSEN by the one-SE rule on CRPS." if shape == chosen else ""),
                extra={"comparisons": {"mae": res["mae"], "minus_A4s_brier": s["minus_A4s_brier"],
                                       "shape_rule": res["shape_rule"]},
                       "calibration": {"ece_null": s["ece_null"], "cover_by_bin": res["cover_by_bin"]},
                       "coefficients": res["coefficients"]})
        print(f"2 runs written to {registry.RUNS_DIR.relative_to(config.ROOT)}/")
    return res


# --------------------------------------------------------------------------- 2. tiebreakers

def run_tiebreak() -> pd.DataFrame:
    sched = mldata.load_schedules(range(2002, 2026))
    t0 = time.perf_counter()
    rep = history.reproduce(sched, range(2002, 2026))
    dt = time.perf_counter() - t0
    req = rep[rep["season"] >= 2006]
    print(f"== Tiebreakers: actual results -> playoff field and seeds ({dt:.2f}s for 24 seasons) ==")
    print(f"2006-2025 (required): {int(req['match'].sum())} of {len(req)} conference-seasons reproduced exactly; "
          f"2002-2005: {int(rep.loc[rep['season'] < 2006, 'match'].sum())} of {int((rep['season'] < 2006).sum())}.")
    print(f"Truth from the bracket alone: {int((rep['truth_from'] == 'bracket').sum())}; "
          f"bracket plus public record: {int((rep['truth_from'] != 'bracket').sum())}.")
    for r in rep[~rep["match"]].itertuples():
        print(f"  MISMATCH {r.season} {r.conf}: ours {r.ours} | truth {r.truth} ({r.truth_from}) {r.points_or_coin_steps}")
    print(f"Seasons where a points-based step or a coin toss decided a seed: {int((rep['points_or_coin_steps'] != '').sum())}")
    OUT.mkdir(parents=True, exist_ok=True)
    rep.to_csv(OUT / "tiebreak_reproduction.csv", index=False)
    return rep


# --------------------------------------------------------------------------- 3 and 4. simulations from past weeks

def past_state(season: int, week: int, fit: mm.MarginFit, shape: str, kn, tab, last: int):
    games, pbp, sched, _ = inputs(last)
    sched = sched[sched["season"] <= season]
    now = sched[(sched["season"] == season) & (sched["week"] == week)]["as_of"].min() - pd.Timedelta(hours=1)
    view = state.as_of_view(sched, now)
    kick = sched.drop_duplicates("game_id").set_index("game_id")["kickoff"]
    k = games["game_id"].map(kick)
    played = games[(games["season"] < 1999) | (k.notna() & (k < now))]
    played = played[played["season"] <= season]
    inp, _, _ = state.state_at(pbp[pbp["game_id"].str[:4].astype(int) <= season], view, played, season, now,
                               fit, shape, kn, tab=tab)
    return inp, now


def simulate_window(seasons: tuple[int, int], taus, n_sims: int, shape: str, label: str) -> pd.DataFrame:
    """Long frame: season, week, team, tau, the five probabilities and wins, plus the actual outcomes.

    Each start uses only what was known then: margin fits on REG 2001..S-1, the
    features and Elo as of an hour before the start week's first kickoff, and
    the starters nflverse would have listed. Seeds: SIM_SEED + 1000 * S + W."""
    games, pbp, sched, frame = inputs(seasons[1])
    kn = keynums(frame, shape)
    fits = margin_fits(frame, seasons)
    tab = oa.rating_table(pbp, sched, oa.TUNED)
    rows = []
    t_state = t_sim = 0.0
    for S in range(seasons[0], seasons[1] + 1):
        truth = history.outcomes(sched, S)
        for W in START_WEEKS:
            t0 = time.perf_counter()
            inp, _ = past_state(S, W, fits[S], shape, kn, tab, seasons[1])
            t1 = time.perf_counter()
            for tau in taus:
                out = sim.simulate(inp, n_sims, tau=float(tau), seed=SIM_SEED + 1000 * S + W)
                out = out.join(truth.add_prefix("true_"))
                out["season"], out["week"], out["tau"] = S, W, float(tau)
                rows.append(out.reset_index())
            t2 = time.perf_counter()
            t_state, t_sim = t_state + t1 - t0, t_sim + t2 - t1
            print(f"  {label} {S} week {W}: {int(inp.remaining.sum())} games left, state {t1 - t0:.1f}s, "
                  f"{len(taus)} x {n_sims:,} sims {t2 - t1:.1f}s", flush=True)
    df = pd.concat(rows, ignore_index=True)
    df.attrs["timing"] = {"state_s": t_state, "sim_s": t_sim}
    return df


def one_se_tau(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """Made-playoffs Brier per tau; the best tau; and the smallest tau within one team-start bootstrap SE of it."""
    keys = ["season", "week", "team"]
    se = {t: g.sort_values(keys)[["playoffs", "true_playoffs"]] for t, g in df.groupby("tau")}
    sq = {t: (g["playoffs"].to_numpy() - g["true_playoffs"].astype(float).to_numpy()) ** 2 for t, g in se.items()}
    best = min(sq, key=lambda t: (sq[t].mean(), t))
    rows = []
    for t in sorted(sq):
        ci = bootstrap.paired_bootstrap_diff(sq[t] - sq[best])
        rows.append({"tau": t, "brier": float(sq[t].mean()), "minus_best": ci["diff"], "se": ci["boot_se"],
                     "within_1se": bool(t == best or ci["diff"] <= ci["boot_se"])})
    table = pd.DataFrame(rows).set_index("tau")
    chosen = float(table.index[table["within_1se"]].min())
    return table, float(best), chosen


def ece_block(sub: pd.DataFrame) -> dict:
    out = {}
    for k in OUTCOMES:
        y = sub[f"true_{k}"].astype(float).to_numpy()
        e = calibration.ece(y, sub[k].to_numpy())
        null = calibration.ece_null_range(sub[k].to_numpy())
        out[k] = {"brier": brier(sub[k], y), "ece": e, "null": null, "inside": bool(null["p05"] <= e <= null["p95"])}
    return out


def run_tune(shape: str, log: bool) -> float:
    print(f"== tau_rest tuning on DEV {DEV[0]}-{DEV[1]}: weeks {START_WEEKS}, shape {shape}, grid {TAU_GRID[0]:g} to "
          f"{TAU_GRID[-1]:g} by 0.5, {TUNE_SIMS:,} sims per start, one-SE rule (smallest tau) ==")
    df = simulate_window(DEV, TAU_GRID, TUNE_SIMS, shape, "tune")
    table, best, chosen = one_se_tau(df)
    print(f"made-playoffs Brier by tau_rest (n = {len(df) // len(TAU_GRID)} team-starts; SE from a team-start bootstrap):")
    for t, r in table.iterrows():
        print(f"  {t:>4g}  {r['brier']:.4f}  minus best {r['minus_best']:+.5f}  SE {r['se']:.5f}  "
              f"{'within' if r['within_1se'] else ''}{'  <- best' if t == best else ''}{'  <- chosen' if t == chosen else ''}")
    ece = ece_block(df[df["tau"] == chosen])
    print(f"chosen tau_rest = {chosen:g} (best {best:g}). ECE at the chosen tau (null 5-95%):")
    for k, v in ece.items():
        print(f"  {k:<9} Brier {v['brier']:.4f}  ECE {v['ece']:.3f}  [{v['null']['p05']:.3f}, {v['null']['p95']:.3f}]  "
              f"{'inside' if v['inside'] else 'OUTSIDE'}")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "tune_sims.parquet", index=False)
    summary = {"table": table.reset_index().to_dict("records"), "best": best, "chosen": chosen, "ece": ece,
               "timing": df.attrs["timing"], "sims": TUNE_SIMS, "grid": list(TAU_GRID)}
    (OUT / "tune.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    if log:
        sub = df[df["tau"] == chosen]
        registry.log_run(
            "m4_tau_rest_tune_dev", label="dev", seasons=DEV,
            game_ids=[f"{s}_{w:02d}_{t}" for s, w, t in zip(sub["season"], sub["week"], sub["team"])],
            metrics={"n": int(len(sub)), "brier_playoffs": float(table.at[chosen, "brier"]),
                     "brier_playoffs_tau0": float(table.at[0.0, "brier"]), "tau_rest": chosen, "best_tau": best,
                     "ece_playoffs": ece["playoffs"]["ece"]},
            features=mm.FEATURES, params={"shape": shape, "grid": list(TAU_GRID), "sims": TUNE_SIMS,
                                          "start_weeks": list(START_WEEKS), "seed": SIM_SEED,
                                          "rule": "smallest tau within one team-start bootstrap SE of the best"},
            data=registry.data_fingerprint(games_csv=True, ml_datasets=("pbp", "schedules"), seasons=range(1999, DEV[1] + 1)),
            notes="tau_rest tuned on DEV made-playoffs Brier (Walker's decision after the 2002-2005 tuning failed on DEV); "
                  "game_ids are season_week_team.",
            extra={"table": summary["table"], "ece": ece, "timing": df.attrs["timing"]})
    return chosen


def cluster_bootstrap(d: np.ndarray, groups: np.ndarray, reps: int = 2000, seed: int = 20261003) -> dict:
    """CI for the mean of d, resampling whole groups (seasons) with replacement."""
    rng = np.random.default_rng(seed)
    keys = np.unique(groups)
    sums = np.array([d[groups == k].sum() for k in keys])
    cnts = np.array([(groups == k).sum() for k in keys])
    idx = rng.integers(0, len(keys), size=(reps, len(keys)))
    means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "excludes_zero": bool(lo > 0 or hi < 0),
            "boot_se": float(means.std(ddof=1)), "groups": int(len(keys)), "reps": reps}


def playoff_eval(df: pd.DataFrame, tau: float) -> dict:
    """Frozen tau against tau = 0 on the same starts and random numbers: Brier, paired CIs, ECE with null ranges,
    and the made-playoffs numbers by start week and by season."""
    a = df[df["tau"] == tau].sort_values(["season", "week", "team"]).reset_index(drop=True)
    b = df[df["tau"] == 0.0].sort_values(["season", "week", "team"]).reset_index(drop=True)
    assert (a[["season", "week", "team"]].to_numpy() == b[["season", "week", "team"]].to_numpy()).all()
    out = {"n": int(len(a)), "tau_rest": tau, "seasons": [int(a["season"].min()), int(a["season"].max())],
           "starts": int(a.groupby(["season", "week"]).ngroups), "metrics": {}}
    for k in OUTCOMES:
        y = a[f"true_{k}"].astype(float).to_numpy()
        pa, pb = a[k].to_numpy(), b[k].to_numpy()
        d = (pa - y) ** 2 - (pb - y) ** 2
        e, null = calibration.ece(y, pa), calibration.ece_null_range(pa)
        out["metrics"][k] = {"brier": brier(pa, y), "brier_tau0": brier(pb, y),
                             "diff": bootstrap.paired_bootstrap_diff(d),
                             "diff_by_season": cluster_bootstrap(d, a["season"].to_numpy()),
                             "ece": e, "ece_tau0": calibration.ece(y, pb), "ece_null": null,
                             "ece_ok": bool(e <= null["p95"]),
                             "ece_in_null_range": bool(null["p05"] <= e <= null["p95"])}
    for key, col in (("per_week", "week"), ("per_season", "season")):
        rows = []
        for v, g in a.groupby(col):
            y = g["true_playoffs"].astype(float)
            rows.append({col: int(v), "n": int(len(g)), "brier": brier(g["playoffs"], y),
                         "brier_tau0": brier(b.loc[g.index, "playoffs"], y),
                         "ece": calibration.ece(y.to_numpy(), g["playoffs"].to_numpy())})
        out[key] = rows
    out["reliability"] = calibration.reliability_table(a["true_playoffs"].astype(float), a["playoffs"], bins=10).to_dict("records")
    out["reliability_tau0"] = calibration.reliability_table(b["true_playoffs"].astype(float), b["playoffs"], bins=10).to_dict("records")
    return out


def print_playoff(out: dict, label: str) -> None:
    s = out["seasons"]
    print(f"== Playoff odds, {label} {s[0]}-{s[1]}: {out['starts']} starts, {out['n']} team-starts, "
          f"tau_rest {out['tau_rest']:g} vs 0 ==")
    print(f"{'outcome':<10}{'Brier tau':>10}{'Brier 0':>9}  {'diff [95% CI, team-starts]':<30}{'[95% CI, by season]':<24}"
          f"{'ECE':>6}  {'null 5-95%':<14}")
    for k, m in out["metrics"].items():
        ci, cc, nl = m["diff"], m["diff_by_season"], m["ece_null"]
        print(f"{k:<10}{m['brier']:>10.4f}{m['brier_tau0']:>9.4f}  {ci['diff']:+.4f} [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]"
              f"     [{cc['ci_low']:+.4f}, {cc['ci_high']:+.4f}]   {m['ece']:>6.3f}  [{nl['p05']:.3f}, {nl['p95']:.3f}]"
              f"{'' if m['ece_in_null_range'] else ('  (below p05)' if m['ece_ok'] else '  ABOVE p95')}")
    print("made playoffs by start week (Brier tau / tau 0, ECE): " + "; ".join(
        f"week {r['week']}: {r['brier']:.4f} / {r['brier_tau0']:.4f}, {r['ece']:.3f}" for r in out["per_week"]))
    print("made playoffs by season (Brier tau / tau 0): " + "; ".join(
        f"{r['season']}: {r['brier']:.4f} / {r['brier_tau0']:.4f}" for r in out["per_season"]))


def sim_params(shape: str, tau: float) -> dict:
    return {"shape": shape, "sims": BACKTEST_SIMS, "start_weeks": list(START_WEEKS), "seed": SIM_SEED,
            "seed_rule": "SIM_SEED + 1000 * season + start week", "tau_rest": tau}


def run_backtest(shape: str, tau: float, log: bool) -> dict:
    df = simulate_window(DEV, (0.0, tau), BACKTEST_SIMS, shape, "backtest")
    out = playoff_eval(df, tau)
    out["shape"], out["timing"] = shape, df.attrs["timing"]
    print_playoff(out, "DEV (in-sample for tau, descriptive)")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "backtest_sims.parquet", index=False)
    (OUT / "backtest.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    if log:
        for name, tau_v in (("m4_playoff_odds", tau), ("m4_playoff_odds_tau0", 0.0)):
            sub = df[df["tau"] == tau_v]
            y = sub["true_playoffs"].astype(float)
            registry.log_run(
                name, label="dev", seasons=DEV,
                game_ids=[f"{s}_{w:02d}_{t}" for s, w, t in zip(sub["season"], sub["week"], sub["team"])],
                metrics={"n": int(len(sub)), "brier": brier(sub["playoffs"], y), "ece": calibration.ece(y, sub["playoffs"]),
                         "tau_rest": tau_v},
                features=mm.FEATURES, params=sim_params(shape, tau_v),
                data=registry.data_fingerprint(games_csv=True, ml_datasets=("pbp", "schedules"),
                                               seasons=range(1999, DEV[1] + 1)),
                notes="Made-playoffs Brier of the season simulation from weeks 4/8/12/16 (DEV is in-sample for tau, "
                      "so descriptive); game_ids are season_week_team.",
                extra={"backtest": out if name == "m4_playoff_odds" else None})
    return out


# --------------------------------------------------------------------------- 5. sign-off (pre-registered)

def pass_rules(mres: dict, pout: dict, shape: str) -> dict:
    """The pre-registered pass rules (context/ml.md, 'M4 holdout pre-registration')."""
    s = mres["shapes"][shape]
    mp = pout["metrics"]["playoffs"]
    inside = sum(b["inside"] for b in mres["cover_by_bin"])
    r = {
        "margin_primary_mae_beats_elo": bool(mres["mae"]["model_minus_elo"]["ci_high"] < 0),
        "margin_s1_brier_within_1se_of_A4s": bool(s["minus_A4s_brier"]["diff"] <= s["minus_A4s_brier"]["boot_se"]),
        "margin_s2_ece_at_or_below_p95": bool(s["ece_ok"]),
        "margin_s3_cover_bins_inside": f"{inside} of {len(mres['cover_by_bin'])}",
        "margin_s3_pass": bool(inside >= COVER_BINS_REQUIRED),
        "odds_p1_playoffs_ece_at_or_below_p95": bool(mp["ece_ok"]),
        "odds_p2_brier_not_worse_than_tau0": bool(mp["diff"]["ci_high"] < BRIER_MARGIN),
        "odds_secondary_ece_at_or_below_p95": {k: bool(pout["metrics"][k]["ece_ok"]) for k in OUTCOMES[1:]},
        # Descriptive only, not a rule: whether each ECE also sits inside the two-sided 5th-95th range.
        "info_two_sided_in_null_range": {"margin": bool(s["ece_in_null_range"]),
                                         **{k: bool(pout["metrics"][k]["ece_in_null_range"]) for k in OUTCOMES}},
    }
    r["margin_primary_pass"] = r["margin_primary_mae_beats_elo"]
    r["odds_primary_pass"] = r["odds_p1_playoffs_ece_at_or_below_p95"] and r["odds_p2_brier_not_worse_than_tau0"]
    return r


def holdout_signoff_runs() -> list[str]:
    hits = []
    for path in glob.glob(str(registry.RUNS_DIR / "*_m4_signoff_*.json")):
        if json.loads(Path(path).read_text()).get("holdout"):
            hits.append(Path(path).name)
    return sorted(hits)


def _max_diff(a, b, path="") -> float:
    """Largest absolute difference between two nested JSON-like structures of numbers (ignores timings)."""
    if isinstance(a, dict) and isinstance(b, dict):
        keys = (set(a) & set(b)) - {"seconds", "timing", "shape"}
        return max([_max_diff(a[k], b[k]) for k in keys] or [0.0])
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return max([_max_diff(x, y) for x, y in zip(a, b)] or [0.0])
    if isinstance(a, bool) or isinstance(b, bool):
        return 0.0 if a == b else float("inf")
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if np.isnan(a) and np.isnan(b):
            return 0.0
        return abs(float(a) - float(b))
    return 0.0 if a == b else float("inf")


def signoff(window: tuple[int, int], allow_holdout: bool, log: bool) -> int:
    """The pre-registered sign-off: margin model and playoff odds, frozen, scored once on `window`.

    With window = DEV and no logging it is the dry run: the same code path must
    reproduce the logged DEV margin numbers and the DEV backtest exactly."""
    label = "holdout" if windows.touches_holdout(window) else "dev"
    if label == "holdout":
        prior = holdout_signoff_runs()
        if prior:
            raise SystemExit(f"a holdout sign-off is already recorded ({prior}); it is run once only")
    shape, tau = mm.CHOSEN_SHAPE, sim.TAU_REST
    print(f"== M4 sign-off ({label}), REG {window[0]}-{window[1]}: margin shape {shape}, tau_rest {tau:g}, "
          f"{BACKTEST_SIMS:,} sims per start, seeds SIM_SEED + 1000 * season + week ==")
    mres, mframe = margin_eval(window, allow_holdout)
    print_margin(mres, label)
    df = simulate_window(window, (0.0, tau), BACKTEST_SIMS, shape, "signoff")
    pout = playoff_eval(df, tau)
    pout["shape"] = shape
    print_playoff(pout, label)
    rules = pass_rules(mres, pout, shape)
    print("\nPre-registered rules:")
    for k, v in rules.items():
        print(f"  {k:<40} {v}")
    print(f"MARGIN MODEL: {'PASS' if rules['margin_primary_pass'] else 'FAIL'} (primary). "
          f"PLAYOFF ODDS: {'PASS' if rules['odds_primary_pass'] else 'FAIL'} (primary).")
    result = {"window": list(window), "label": label, "shape": shape, "tau_rest": tau, "margin": mres,
              "playoff_odds": pout, "rules": rules}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"signoff_{label}.json").write_text(json.dumps(result, indent=1, default=float) + "\n")

    if label == "dev":
        ref_m, ref_b = OUT / "margin_dev.json", OUT / "backtest.json"
        if ref_m.exists() and ref_b.exists():
            dm = _max_diff(mres, json.loads(ref_m.read_text()))
            db = _max_diff(json.loads(json.dumps(pout, default=float)), json.loads(ref_b.read_text()))
            print(f"\nDry-run reproduction: margin numbers max |diff| {dm:.1e} vs {ref_m.name}; "
                  f"playoff odds max |diff| {db:.1e} vs {ref_b.name} -> "
                  f"{'REPRODUCED' if dm == 0 and db == 0 else 'NOT REPRODUCED'}")
            result["dry_run"] = {"margin_max_abs_diff": dm, "odds_max_abs_diff": db}
            (OUT / f"signoff_{label}.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
            if not (dm == 0 and db == 0):
                return 1
        else:
            print("\nDry run: run `ml_m4.py margin` and `ml_m4.py backtest` first to compare against them.")

    if log:
        fp = registry.data_fingerprint(games_csv=True, ml_datasets=("pbp", "schedules"), seasons=range(1999, window[1] + 1))
        s = mres["shapes"][shape]
        registry.log_run(
            "m4_signoff_margin", label=label, seasons=window, game_ids=mframe["game_id"],
            metrics={**{k: s[k] for k in ("n", "brier", "logloss", "accuracy", "ece")}, "crps": s["crps"],
                     "mae": mres["mae"]["model"], "mae_elo": mres["mae"]["elo"], "mae_market": mres["mae"]["market"]},
            features=mm.FEATURES, params=dict(margin_params(mm.KeyNumbers.from_dict(mres["keynum"])), shape=shape,
                                              protocol="context/ml.md, M4 holdout pre-registration; one run"),
            data=fp, notes="M4 sign-off, margin model (frozen).",
            extra={"signoff": "m4_signoff", "result": mres, "rules": rules})
        a = df[df["tau"] == tau]
        registry.log_run(
            "m4_signoff_playoff_odds", label=label, seasons=window,
            game_ids=[f"{x}_{w:02d}_{t}" for x, w, t in zip(a["season"], a["week"], a["team"])],
            metrics={"n": int(len(a)), "brier": pout["metrics"]["playoffs"]["brier"],
                     "brier_tau0": pout["metrics"]["playoffs"]["brier_tau0"], "ece": pout["metrics"]["playoffs"]["ece"],
                     "tau_rest": tau},
            features=mm.FEATURES, params=dict(sim_params(shape, tau), protocol="context/ml.md, M4 holdout pre-registration; one run"),
            data=fp, notes="M4 sign-off, playoff odds (frozen tau_rest and shape) against tau_rest = 0.",
            extra={"signoff": "m4_signoff", "result": pout, "rules": rules})
        print(f"2 runs written to {registry.RUNS_DIR.relative_to(config.ROOT)}/ (label {label})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["margin", "tiebreak", "tune", "backtest", "all", "signoff-dry-run", "signoff"])
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--shape", choices=mm.SHAPES, help="override the shape (default: the DEV choice)")
    ap.add_argument("--tau", type=float, help="override tau_rest for the backtest (default: the frozen value)")
    a = ap.parse_args()
    log = not a.no_log
    t0 = time.perf_counter()
    if a.stage == "signoff-dry-run":
        return signoff(DEV, allow_holdout=False, log=False)
    if a.stage == "signoff":
        return signoff(windows.HOLDOUT, allow_holdout=True, log=True)
    shape, tau = a.shape or mm.CHOSEN_SHAPE, a.tau if a.tau is not None else sim.TAU_REST
    if a.stage in ("margin", "all"):
        r = run_margin(log)
        shape = a.shape or r["shape_rule"]["chosen"]
    if a.stage in ("tiebreak", "all"):
        rep = run_tiebreak()
        if not rep.loc[rep["season"] >= 2006, "match"].all():
            print("STOP: unexplained tiebreaker mismatches.", file=sys.stderr)
            return 1
    if a.stage in ("tune", "all"):
        chosen = run_tune(shape, log)
        if a.tau is None and chosen != sim.TAU_REST:
            print(f"NOTE: the tuned tau_rest ({chosen:g}) differs from the frozen sim.TAU_REST ({sim.TAU_REST:g}); "
                  "update nflelo/ml/sim/season.py before the backtest and sign-off.", file=sys.stderr)
            tau = chosen
    if a.stage in ("backtest", "all"):
        run_backtest(shape, tau, log)
    print(f"\ntotal {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
