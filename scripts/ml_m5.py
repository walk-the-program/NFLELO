"""Run the M5 ladder B0-B5 + B3f on the M5 evaluation window (context/ml-m5-method.md, section 5).

    python scripts/ml_m5.py                # features, ladder, log every step (label "m5dev"), print tables
    python scripts/ml_m5.py --no-log
    python scripts/ml_m5.py --leakcheck    # real-data leakage checks of the roster builders (2015-2017 data)
    python scripts/ml_m5.py --validate     # playing-time and lineup validation against participation (2016-2019)

Window (decision M5-D3): REG 2012-2019 games with moneylines, every step scored
on the SAME games, with A4s on that subset as the reference (B0). Nothing
loads a season after 2019; there is no holdout path in this script.

Training (walk-forward like M3): season S is predicted by a logistic model fit
on REG games of seasons 2009..S-1 (injury reports start in 2009, so the lineup
features exist from 2009), season-decay weights, ties as two half rows. B0 is
A4s exactly as in M3 (trained 2001..S-1); B0r is the same three features
trained 2009..S-1, so the training-window effect can be separated from the
feature effect.

    B0   A4s (elo_logit, adj_epa_margin, qb_delta_diff), trained 2001..S-1   reference
    B0r  A4s features, trained 2009..S-1                                     diagnostic
    B1   B0r + lineup_delta_diff, box-score values only
    B2   B0r + lineup_delta_diff, with/without values (linemen included)
    B3   B2 + preseason_change_diff + preseason_change_diff_wk
    B4   B3 with the lineup delta split into offense and defense
    B5   B3 with Questionable as 0/1 (every probability rounded)
    B3f  B3 without the inactive list (the Friday-only, Sunday-morning version)

Selection: one-SE rule (M3-D3) among B0-B5 (B0r and B3f are diagnostics).
Paired bootstrap CIs (2,000 reps, seed 20261003); the one-sided sample-size-
aware ECE check (ECE <= 95th percentile of a calibrated forecaster's ECE at
the same n). A step that beats B0 by more than LEAK_ALARM is treated as a
suspected leak.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ml_m3  # noqa: E402
from nflelo import config  # noqa: E402
from nflelo import data as elo_data  # noqa: E402
from nflelo.ml import registry  # noqa: E402
from nflelo.ml import data as mldata  # noqa: E402
from nflelo.ml.eval import baselines, bootstrap, calibration, windows  # noqa: E402
from nflelo.ml.features import opponent_adjust as oa, qb, roster as rf  # noqa: E402
from nflelo.ml.models import logistic  # noqa: E402
from nflelo.ml.players import credit as cr, inputs as m5in, lineup as lu, value as pv  # noqa: E402

WINDOW = windows.M5_DEV
LAST = WINDOW[1]
TRAIN_START_M5 = 2009
LEAK_ALARM = 0.006
OUT_DIR = mldata.ML_DIR / "m5"
A4S = ["elo_logit", "adj_epa_margin", "qb_delta_diff"]
BUCKETS = [("1-4", 1, 4), ("5-9", 5, 9), ("10-18", 10, 18)]
VARIANTS = {"full": lu.LineupConfig(), "friday": lu.LineupConfig(friday_only=True),
            "binary": lu.LineupConfig(binary=True)}
STEPS = [
    {"step": "B0", "features": A4S, "train_start": logistic.TRAIN_START, "candidate": True},
    {"step": "B0r", "features": A4S, "train_start": TRAIN_START_M5, "candidate": False},
    {"step": "B1", "features": A4S + ["lineup_delta_diff__box_full"], "train_start": TRAIN_START_M5, "candidate": True},
    {"step": "B2", "features": A4S + ["lineup_delta_diff__ww_full"], "train_start": TRAIN_START_M5, "candidate": True},
    {"step": "B3", "features": A4S + ["lineup_delta_diff__ww_full", "preseason_change_diff__ww_full",
                                      "preseason_change_diff_wk__ww_full"], "train_start": TRAIN_START_M5,
     "candidate": True},
    {"step": "B4", "features": A4S + ["lineup_delta_off_diff__ww_full", "lineup_delta_def_diff__ww_full",
                                      "preseason_change_diff__ww_full", "preseason_change_diff_wk__ww_full"],
     "train_start": TRAIN_START_M5, "candidate": True},
    {"step": "B5", "features": A4S + ["lineup_delta_diff__ww_binary", "preseason_change_diff__ww_binary",
                                      "preseason_change_diff_wk__ww_binary"], "train_start": TRAIN_START_M5,
     "candidate": True},
    {"step": "B3f", "features": A4S + ["lineup_delta_diff__ww_friday", "preseason_change_diff__ww_friday",
                                       "preseason_change_diff_wk__ww_friday"], "train_start": TRAIN_START_M5,
     "candidate": False},
]
EXPECTED_SIGN = {"elo_logit": 1, "adj_epa_margin": 1, "qb_delta_diff": 1, "lineup_delta_diff": 1,
                 "lineup_delta_off_diff": 1, "lineup_delta_def_diff": 1, "preseason_change_diff": 1,
                 "preseason_change_diff_wk": -1}


def base_name(f: str) -> str:
    return f.split("__")[0]


def brier(y, p) -> float:
    return float(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2))


# --------------------------------------------------------------------------- features

def build(log_timing: dict) -> dict:
    """Everything the ladder needs, data through 2019 only."""
    t0 = time.perf_counter()
    inp = m5in.load(LAST)
    games = elo_data.load_games()
    games = games[games["season"] <= LAST].reset_index(drop=True)
    log_timing["load_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    frame, ft = ml_m3.build_frame(games, inp.pbp, inp.sched, last_season=LAST)
    ratings = ft.pop("ratings_obj")
    log_timing["m3_frame_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cfg = pv.TUNED
    # status probabilities: a fixed table estimated on 2009-2015 only (spec section 4)
    pg0 = cr.player_games(inp.pbp, inp.sched, inp.positions, inp.depth, inp.availability)
    base = lu.attach_statuses(lu.chart_players(inp.depth), inp.injuries, inp.rosters, inp.roster_seasons)
    probs = lu.estimate_status_probs(base, pg0[pg0["inv"] > 0])
    state = rf.prepare(inp.pbp, inp.sched, inp.depth, inp.injuries, inp.rosters, inp.positions, probs,
                       inp.availability, cfg, tuple(VARIANTS.values()), inp.roster_seasons)
    log_timing["credit_lineups_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    reg = inp.sched[(inp.sched["game_type"] == "REG") & inp.sched["season"].between(TRAIN_START_M5, LAST)
                    & inp.sched["home_score"].notna()]
    keys = sorted(set(zip(reg["season"].astype(int), reg["week"].astype(int))))
    tab = oa.rating_table(inp.pbp, inp.sched, oa.TUNED)
    vals = rf.compute_values(state, keys, cfg, ratings, tab, with_ridge=True)
    log_timing["values_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    feats = {}
    terms_all = {}
    for vkind, vcol in (("box", "v_box"), ("ww", "v_ww")):
        for lname, lcfg in VARIANTS.items():
            if vkind == "box" and lname != "full":
                continue
            terms = rf.team_lineup_terms(state, vals, vcol, reg, lcfg, oa.TUNED)
            f = rf.features_from_terms(terms, reg)
            terms_all[f"{vkind}_{lname}"] = terms
            for c in rf.FEATURES:
                feats[f"{c}__{vkind}_{lname}"] = f[c]
            if vkind == "ww" and lname in ("full", "friday"):
                feats[f"starters_out_home__{lname}"] = f["starters_out_home"]
                feats[f"starters_out_away__{lname}"] = f["starters_out_away"]
    lf = pd.DataFrame(feats)
    log_timing["features_s"] = time.perf_counter() - t0
    frame = frame.merge(lf, left_on="game_id", right_index=True, how="left")
    return {"inp": inp, "games": games, "frame": frame, "state": state, "vals": vals, "probs": probs,
            "terms": terms_all, "ratings": ratings}


# --------------------------------------------------------------------------- ladder

def run(spec: dict, frame: pd.DataFrame):
    sub = frame[frame["season"] >= spec["train_start"]]
    sub = sub.dropna(subset=spec["features"])
    p, c = logistic.walk_forward(sub, spec["features"], WINDOW, train_start=spec["train_start"])
    return p.reindex(frame.index), c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--leakcheck", action="store_true")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    windows.check(WINDOW, allow_holdout=False)
    if a.leakcheck:
        return leakcheck()
    timing = {}
    t_start = time.perf_counter()
    b = build(timing)
    if a.validate:
        return validate(b)
    frame, games = b["frame"], b["games"]
    base = baselines.baseline_frame(games, WINDOW, require_market=True)
    frame = frame.merge(base[["game_id", "p_elo_v2", baselines.MARKET_COL]], on="game_id", how="left")
    mask = frame["game_id"].isin(base["game_id"]).to_numpy()
    if mask.sum() != len(base):
        raise SystemExit(f"only {mask.sum()} of {len(base)} scored games found in the feature frame")
    ev = frame[mask]
    y = ev["y"].to_numpy()
    gh = registry.games_hash(ev["game_id"])
    train_years = {s["step"]: f"{s['train_start']}..S-1" for s in STEPS}
    no_lineup = int((~frame.loc[frame["season"] >= TRAIN_START_M5, "lineup_delta_diff__ww_full"].abs().gt(0)).sum())
    print(f"M5 window REG {WINDOW[0]}-{WINDOW[1]} with moneylines: n = {len(ev)}, games hash {gh}.")
    print(f"Training: B0 on REG {logistic.TRAIN_START}..S-1 (as M3); every other step on REG {TRAIN_START_M5}..S-1 "
          f"(seasons {TRAIN_START_M5}-{WINDOW[0] - 1} for the {WINDOW[0]} model up to {TRAIN_START_M5}-{LAST - 1} "
          f"for {LAST}). Team-games with a zero lineup delta in {TRAIN_START_M5}-{LAST}: {no_lineup}.")
    print("Status probabilities (estimated on 2009-2015 rank-1 starters):")
    print(b["probs"].pivot(index="group", columns="status", values="p").round(3).to_string())

    t0 = time.perf_counter()
    preds, coefs, res = {}, {}, {}
    for spec in STEPS:
        p, c = run(spec, frame)
        preds[spec["step"]], coefs[spec["step"]] = p, c
        pe = p[mask].to_numpy()
        if np.isnan(pe).any():
            raise SystemExit(f"{spec['step']}: {np.isnan(pe).sum()} games without a prediction")
        m = ml_m3.score(y, pe)
        m["ece_null"] = calibration.ece_null_range(pe)
        m["ece_ok"] = bool(m["ece"] <= m["ece_null"]["p95"])
        res[spec["step"]] = m
    p0 = preds["B0"][mask].to_numpy()
    p_elo = ev["p_elo_v2"].to_numpy()
    p_mkt = ev[baselines.MARKET_COL].to_numpy()
    for s, m in res.items():
        pe = preds[s][mask].to_numpy()
        m["vs_B0_brier"] = bootstrap.paired_bootstrap(y, pe, p0, "brier")
        m["vs_B0_logloss"] = bootstrap.paired_bootstrap(y, pe, p0, "logloss")
        m["vs_elo_brier"] = bootstrap.paired_bootstrap(y, pe, p_elo, "brier")
    t_ladder = time.perf_counter() - t0
    # B0 must equal the M3 ladder's A4s predictions on these games (same code, same data)
    m3f = mldata.ML_DIR / "m3" / "ladder_frame.parquet"
    if m3f.exists():
        old = pd.read_parquet(m3f, columns=["game_id", "p_A4s"]).set_index("game_id")["p_A4s"]
        d = np.abs(old.reindex(ev["game_id"]).to_numpy() - p0)
        print(f"B0 vs the saved M3 ladder A4s predictions on these games: max |diff| {np.nanmax(d):.2e}")

    # one-SE rule
    cands = [s["step"] for s in STEPS if s["candidate"]]
    best = min(cands, key=lambda s: res[s]["brier"])
    pb = preds[best][mask].to_numpy()
    for s in res:
        ci = bootstrap.paired_bootstrap(y, preds[s][mask].to_numpy(), pb, "brier")
        res[s]["vs_best"] = ci
        res[s]["within_1se"] = bool(s == best or ci["diff"] <= ci["boot_se"])
    spec_of = {s["step"]: s for s in STEPS}
    eligible = [s for s in cands if res[s]["within_1se"]]
    chosen = min(eligible, key=lambda s: (len(spec_of[s]["features"]), res[s]["brier"]))

    print(f"\nLadder (n = {len(ev)}; diff = step minus B0 Brier, 95% paired bootstrap CI, 2,000 reps, seed 20261003; "
          "ECE p95 = one-sided null at this n):")
    print(f"{'step':<5}{'train':<10}{'brier':>8}{'logloss':>9}{'acc':>7}{'ece':>7}{'p95':>7}{'ok':>4}  "
          f"{'vs B0 [95% CI]':<29}{'1SE':>4}")
    for spec in STEPS:
        s, m = spec["step"], res[spec["step"]]
        c = m["vs_B0_brier"]
        print(f"{s:<5}{train_years[s]:<10}{m['brier']:>8.4f}{m['logloss']:>9.4f}{m['accuracy']:>7.3f}{m['ece']:>7.4f}"
              f"{m['ece_null']['p95']:>7.4f}{'y' if m['ece_ok'] else 'N':>4}  "
              f"{c['diff']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]  "
              f"{'yes' if m['within_1se'] and spec['candidate'] else '':>4}{' *' if s == chosen else ''}")
    print(f"Elo v2: {brier(y, p_elo):.4f}   Market (benchmark): {brier(y, p_mkt):.4f}")
    alarms = [s for s in res if res[s]["vs_B0_brier"]["diff"] < -LEAK_ALARM]
    print(f"Leak alarm (any step beating B0 by more than {LEAK_ALARM}): {alarms or 'none'}")
    print(f"Best candidate: {best}; within one SE: {eligible}; chosen by the one-SE rule: {chosen}.")
    c = res[chosen]["vs_B0_brier"]
    accept = chosen != "B0" and c["excludes_zero"] and c["diff"] < 0 and res[chosen]["ece_ok"]
    print(f"M5 acceptance on the window (chosen beats B0 with CI excluding 0 and ECE check passing): "
          f"{'PASS' if accept else 'FAIL'}")

    # buckets and the 2+-starters-out split
    week = ev["week"].to_numpy()
    out_h = ev["starters_out_home__full"].to_numpy()
    out_a = ev["starters_out_away__full"].to_numpy()
    splits = [(f"weeks {n}", (week >= lo) & (week <= hi)) for n, lo, hi in BUCKETS]
    splits += [("2+ starters out (either team)", (out_h >= 2) | (out_a >= 2)),
               ("fewer than 2 out", (out_h < 2) & (out_a < 2))]
    split_rows = []
    show = ["B0", "B0r", "B1", "B2", "B3", "B3f", chosen] if chosen not in ("B0", "B1", "B2", "B3") else \
        ["B0", "B0r", "B1", "B2", "B3", "B3f"]
    show = list(dict.fromkeys(show))
    print("\nBy split (Brier; diff = step minus B0 with 95% CI):")
    for name, m_ in splits:
        row = {"split": name, "n": int(m_.sum()), "elo": brier(y[m_], p_elo[m_]), "market": brier(y[m_], p_mkt[m_])}
        parts = []
        for s in show:
            ps = preds[s][mask].to_numpy()
            row[s] = brier(y[m_], ps[m_])
            if s != "B0":
                row[f"{s}_vs_B0"] = bootstrap.paired_bootstrap(y[m_], ps[m_], p0[m_], "brier")
                ci = row[f"{s}_vs_B0"]
                parts.append(f"{s} {row[s]:.4f} ({ci['diff']:+.4f} [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}])")
            else:
                parts.append(f"B0 {row[s]:.4f}")
        split_rows.append(row)
        print(f"  {name:<30} n {row['n']:>5}  Elo {row['elo']:.4f}  " + "  ".join(parts))

    # coefficients
    print(f"\nCoefficients of {chosen} (last refit; range over the {len(coefs[chosen])} refits):")
    cc = coefs[chosen]
    last = cc.iloc[-1]
    bad = []
    for f in ["intercept", *spec_of[chosen]["features"]]:
        sd = f"  per 1 SD ({ev[f].std():.4f}): {last[f] * ev[f].std():+.3f}" if f != "intercept" else ""
        print(f"  {f:<40}{last[f]:+.4f}  [{cc[f].min():+.4f}, {cc[f].max():+.4f}]{sd}")
        e = EXPECTED_SIGN.get(base_name(f), 0)
        if e and (np.sign(cc[f]) != e).any():
            bad.append(f)
    print("  sign check: " + ("all coefficients have the expected sign in every refit" if not bad
                              else f"UNEXPECTED SIGN in some refit: {bad}"))
    for s in ("B3", "B2", "B1"):
        if s != chosen:
            l3 = coefs[s].iloc[-1]
            print(f"  ({s} last refit: " + ", ".join(f"{base_name(f)} {l3[f]:+.3f} [{coefs[s][f].min():+.3f}, "
                                                    f"{coefs[s][f].max():+.3f}]" for f in spec_of[s]["features"][3:]) + ")")

    # feature summary
    fcols = ["lineup_delta_diff__box_full", "lineup_delta_diff__ww_full", "preseason_change_diff__ww_full",
             "qb_delta_diff", "adj_epa_margin"]
    print("\nFeature SDs on the window: " + ", ".join(f"{c} {ev[c].std():.4f}" for c in fcols))
    print("Correlations: " + ", ".join(
        f"{a_}~{b_} {np.corrcoef(ev[a_], ev[b_])[0, 1]:+.2f}" for a_, b_ in
        [("lineup_delta_diff__ww_full", "qb_delta_diff"), ("lineup_delta_diff__ww_full", "elo_logit"),
         ("preseason_change_diff__ww_full", "elo_logit"), ("lineup_delta_diff__box_full", "lineup_delta_diff__ww_full")]))

    timing["ladder_s"] = t_ladder
    timing["total_s"] = time.perf_counter() - t_start
    print("\nTiming: " + ", ".join(f"{k} {v:.1f}s" for k, v in timing.items()))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    for s in preds:
        out[f"p_{s}"] = preds[s].to_numpy()
    out["eval"] = mask
    out.to_parquet(OUT_DIR / "ladder_frame.parquet", index=False)
    pd.concat([c.assign(step=s) for s, c in coefs.items()], ignore_index=True).to_parquet(OUT_DIR / "coefs.parquet")
    b["vals"].to_parquet(OUT_DIR / "values.parquet", index=False)
    b["state"].pg.to_parquet(OUT_DIR / "player_games.parquet", index=False)
    b["state"].lineups[VARIANTS["full"]].to_parquet(OUT_DIR / "lineups_full.parquet", index=False)
    b["probs"].to_parquet(OUT_DIR / "status_probs.parquet", index=False)

    if not a.no_log:
        fp = m5in.fingerprint(LAST) | {"data/games.csv": registry.sha256_path(config.GAMES_CSV)}
        params = {"value": pv.TUNED.to_dict(), "ratings": oa.TUNED.to_dict(), "qb": qb.tuned_config().to_dict(),
                  "season_half_life": logistic.SEASON_HALF_LIFE, "ties": "two half-weight rows",
                  "status_probs": b["probs"].to_dict("records"), "status_prob_seasons": list(lu.ESTIMATION_SEASONS),
                  "elo_config": config.DEFAULT_CONFIG.to_dict()}
        notes = {"B0": "Reference: A4s as in M3 (trained 2001..S-1).",
                 "B0r": "Diagnostic: A4s features trained 2009..S-1.",
                 "B1": "lineup_delta_diff from box-score values.", "B2": "lineup_delta_diff from with/without values.",
                 "B3": "B2 + preseason roster change (fading).", "B4": "B3 with offense/defense split.",
                 "B5": "B3 with 0/1 play probabilities.", "B3f": "B3 without inactives (Friday-only; M5-D2)."}
        for spec in STEPS:
            s, m = spec["step"], res[spec["step"]]
            registry.log_run(
                f"m5_{s}", label="m5dev", seasons=WINDOW, game_ids=ev["game_id"],
                metrics={k: m[k] for k in ("n", "brier", "logloss", "accuracy", "ece")},
                features=spec["features"], params=dict(params, train_start=spec["train_start"]), data=fp,
                notes=notes[s] + (" CHOSEN by the one-SE rule." if s == chosen else ""),
                extra={"ece_null": m["ece_null"], "ece_ok": m["ece_ok"],
                       "comparisons": {"vs_B0_brier": m["vs_B0_brier"], "vs_B0_logloss": m["vs_B0_logloss"],
                                       "vs_elo_brier": m["vs_elo_brier"], "vs_best_brier": m["vs_best"]},
                       "one_se": {"best": best, "within": m["within_1se"], "chosen": s == chosen,
                                  "eligible": eligible, "candidate": spec["candidate"]},
                       "coefficients": coefs[s].to_dict("records"),
                       "splits": [{k: v for k, v in r.items()} for r in split_rows] if s == chosen else None,
                       "acceptance_pass": accept if s == chosen else None,
                       "timing": timing if s == chosen else None})
        print(f"\n{len(STEPS)} runs written to {registry.RUNS_DIR.relative_to(config.ROOT)}/")
    return 0


# --------------------------------------------------------------------------- validation (participation, CC BY-SA)

def validate(b: dict) -> int:
    from nflelo.ml.players import validate as va
    st, inp = b["state"], b["inp"]
    yrs = range(2016, LAST + 1)
    part = mldata.load_participation_for_validation(yrs)
    act = va.onfield_shares(part, inp.pbp[inp.pbp["season"].between(2016, LAST)], st.sched)
    groups = st.onfield.drop_duplicates("gsis_id", keep="last").set_index("gsis_id")["group"]
    est = st.onfield[st.onfield["season"].between(2016, LAST)]
    print(f"Participation games covered: {len(act.attrs['games_covered'])} (REG 2016-{LAST}).")
    se = va.share_errors(est, act, groups)
    print("\nPlaying-time estimate (usage-based share, linemen = pregame p_play) vs participation share, by group:")
    print(se.round(3).to_string(index=False))
    # pregame expected share of rank-1 starters vs actual
    lu_full = st.lineups[VARIANTS["full"]]
    x = lu_full[(lu_full["season"].between(2016, LAST)) & (lu_full["rank"] == 1)]
    reg = st.sched[st.sched["game_type"] == "REG"]
    tg = pd.concat([pd.DataFrame({"game_id": reg["game_id"], "season": reg["season"], "week": reg["week"],
                                  "team": rf.pos.map_teams(reg[f"{s}_team"], reg["season"])}) for s in ("home", "away")])
    x = x.merge(tg, on=["season", "week", "team"])
    on = act.groupby(["game_id", "gsis_id"])["actual"].sum()
    x["actual"] = on.reindex(pd.MultiIndex.from_frame(x[["game_id", "gsis_id"]])).fillna(0.0).to_numpy()
    x["pred"] = x["p_play"] * x["share_exp"]
    x = x[x["group"] != "K"]
    pre = x.groupby("group").apply(lambda d: pd.Series({
        "n": len(d), "mae": (d["pred"] - d["actual"]).abs().mean(), "bias": (d["pred"] - d["actual"]).mean(),
        "corr": np.corrcoef(d["pred"], d["actual"])[0, 1]}), include_groups=False)
    print("\nPregame expected share (p_play x expected share) of rank-1 starters vs participation share:")
    print(pre.round(3).to_string())
    for name, key in (("with inactives", "full"), ("Friday-only (no inactives)", "friday")):
        lu_ = st.lineups[VARIANTS[key]]
        lu_ = lu_[lu_["season"].between(2016, LAST) & (lu_["group"] != "K")]
        print(f"\nLineup flags vs actual absences (rank-1 starters, flag = p_play < 0.5), 2016-{LAST}, {name}:")
        print(va.absence_table(lu_, act, st.sched).round(3).to_string(index=False))
        print(f"  2019 only (INA partly recorded): ")
        print(va.absence_table(lu_[lu_["season"] == 2019], act, st.sched).round(3).to_string(index=False))
    return 0


# --------------------------------------------------------------------------- real-data leakage check

def leakcheck() -> int:
    """roster_leakage_check on real 2015-2017 data, box and with/without builders, plus the positive control."""
    t0 = time.perf_counter()
    inp = m5in.load(2017, first_season=2014)
    sched = inp.sched
    probs = lu.default_probs()
    tables = inp.tables()

    def builder(cfg):
        def b(pbp, s, g, t):
            return rf.build_features(pbp, s, g, t["depth"], t["injuries"], t["rosters"], inp.positions, probs,
                                     inp.availability, cfg, inp.roster_seasons)
        return b
    reg = sched[(sched["season"] == 2017) & (sched["game_type"] == "REG")].sort_values("kickoff")
    targets = [reg[reg["week"] == w]["game_id"].iloc[i] for w, i in ((1, 0), (1, 5), (9, 3), (15, 0), (15, 7))]
    print(f"Targets: {targets}")
    out = {}
    for name, cfg in (("box", rf.RosterConfig(values="box", value_cfg=pv.TUNED)),
                      ("ww", rf.RosterConfig(values="ww", value_cfg=pv.TUNED)),
                      ("ww_friday", rf.RosterConfig(values="ww", value_cfg=pv.TUNED,
                                                    lineup=lu.LineupConfig(friday_only=True))),
                      ("POSITIVE CONTROL (leaky)", rf.RosterConfig(values="box", value_cfg=pv.TUNED, leaky=True))):
        r = rf.roster_leakage_check(builder(cfg), inp.pbp, sched, tables, targets)
        out[name] = r
        print(f"{name:<26} leak-free {int(r['leak_free'].sum())}/{len(r)}  changed values {r['changed'].tolist()}")
    ok = all(out[k]["leak_free"].all() for k in ("box", "ww", "ww_friday")) and not out[
        "POSITIVE CONTROL (leaky)"]["leak_free"].any()
    print(f"Result: {'PASS' if ok else 'FAIL'} (honest builders leak-free, positive control fails); "
          f"{time.perf_counter() - t0:.0f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
