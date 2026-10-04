"""Weekly ML predictions (M4 Phase 1): refresh nflverse, predict every unplayed game, append to the ledger.

    python scripts/ml_predict.py              # refresh this season's schedule and play-by-play, predict, append
    python scripts/ml_predict.py --offline    # use the cached nflverse files only
    python scripts/ml_predict.py --now 2026-10-07T14:00:00Z --ledger /tmp/test.csv   # tests only

Run scripts/build.py first: the Elo numbers come from outputs/ and must include
every game nflverse already shows as final (the script stops otherwise).

What it does (context/ml-m4-method.md, sections 2 and 3):
1. Refreshes this season's nflverse schedule and play-by-play (`nflelo.ml.data.refresh`).
2. Loads the season model, A4s fit on REG 2001..season-1 with the frozen M3 knobs.
   The fit runs once per season and is saved to experiments/live/model_<season>.json
   (coefficients plus training metadata); later runs reuse it. The first fit also
   refits season-1 through the same code path and checks it reproduces the M3
   sign-off coefficients (or last season's live model), and stops if it doesn't.
3. Builds the three features as of now for every unplayed REG game
   (`nflelo.ml.live_features`) and predicts the home win probability.
4. Adds Elo's probability and spread (the exporter's own computation) and, only
   after the model has predicted, the vig-removed moneyline and spread line.
5. Appends one row per game that has not kicked off to experiments/live/<season>.csv,
   and writes experiments/live/latest_<season>.json (every unplayed game, with
   features and starters) for the site exporter.

Afterwards run scripts/export_site.py so site/data/ml.json picks up the new rows.
"""
import argparse
import glob
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nflelo import config  # noqa: E402
from nflelo import data as elo_data  # noqa: E402
from nflelo.evaluate import market_prob  # noqa: E402
from nflelo.ml import asof, live, live_features, registry  # noqa: E402
from nflelo.ml import data as mldata  # noqa: E402
from nflelo.ml.features import opponent_adjust as oa, qb  # noqa: E402
from nflelo.ml.models import logistic  # noqa: E402

import export_site  # noqa: E402  (scripts/ is on sys.path when run as a script)
import ml_m3  # noqa: E402

SEASON = 2026
FIRST_DATA_SEASON = mldata.FIRST_PBP_SEASON
COEF_TOLERANCE = 1e-6


def model_spec() -> dict:
    """Everything that defines the model; a saved model is reused only when this matches."""
    return {"model": "A4s", "features": live_features.FEATURES, "kind": "logistic, unpenalized",
            "ratings": oa.TUNED.to_dict(), "qb": qb.tuned_config().to_dict(), "train_start": logistic.TRAIN_START,
            "season_half_life": logistic.SEASON_HALF_LIFE, "ties": "two half-weight rows",
            "elo_config": config.DEFAULT_CONFIG.to_dict(), "qb_starter": "actual (M3-D1)",
            "hfa": "Elo online HFA frozen before the week's first game"}


# --------------------------------------------------------------------------- data

def refresh(season: int, offline: bool) -> list[dict]:
    if offline:
        need = [("schedules", s) for s in range(FIRST_DATA_SEASON, season + 1)] + \
               [("pbp", s) for s in range(FIRST_DATA_SEASON, season + 1)]
        missing = [f"{d}/{s}" for d, s in need if not mldata.cache_path(d, s).exists()]
        if missing:
            raise SystemExit(f"--offline but the cache lacks {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        return []
    res = mldata.refresh(season)
    for r in res:
        print(f"nflverse {r['dataset']} {r['season']}: rows {r['old_rows']} -> {r['rows']}, "
              f"{'changed' if r['changed'] else 'unchanged'}")
    return res


def load(season: int):
    years = range(FIRST_DATA_SEASON, season + 1)
    pbp = mldata.load_pbp(years, columns=ml_m3.PBP_COLUMNS)
    if not (pbp["game_id"].str[:4].astype(int) == season).any():
        raise SystemExit(f"nflverse has no {season} play-by-play yet")
    sched = asof.add_asof(mldata.load_schedules(years))
    games = elo_data.load_games()
    return pbp, sched, games


def data_hash(season: int) -> str:
    h = mldata.manifest_hashes(("pbp", "schedules"), range(FIRST_DATA_SEASON, season + 1))
    parts = [f"{k}:{v}" for k, v in h.items()]
    parts += [f"games.csv:{registry.sha256_path(config.GAMES_CSV)}",
              f"elo_games.csv:{registry.sha256_path(config.OUT_DIR / 'elo_games.csv')}"]
    return registry.games_hash(parts)


def check_elo_fresh(sched: pd.DataFrame, games: pd.DataFrame, elo_out: pd.DataFrame, season: int) -> None:
    """Every game nflverse shows as final must already be in data/games.csv and outputs/elo_games.csv."""
    done = set(sched.loc[(sched["season"] == season) & sched["home_score"].notna(), "game_id"])
    for name, ids in (("data/games.csv", set(games["game_id"])), ("outputs/elo_games.csv", set(elo_out["game_id"]))):
        gap = sorted(done - ids)
        if gap:
            raise SystemExit(f"{name} is missing {len(gap)} finished games (e.g. {gap[:3]}). "
                             "Run scripts/build.py first so Elo is current.")


# --------------------------------------------------------------------------- the season model

def reference_coefs(season: int) -> tuple[dict, str] | tuple[None, None]:
    """Coefficients the refit of `season` must reproduce: last season's live model, else the M3 sign-off run."""
    prev = live.model_path(season)
    if prev.exists():
        m = json.loads(prev.read_text())
        return {"intercept": m["intercept"], **m["coef"]}, str(prev.relative_to(config.ROOT))
    for path in sorted(glob.glob(str(registry.RUNS_DIR / "*_m3_signoff_A4s.json")), reverse=True):
        rec = json.loads(Path(path).read_text())
        for c in rec.get("coefficients") or []:
            if int(c["season"]) == season:
                return {k: c[k] for k in ["intercept", *live_features.FEATURES]}, str(Path(path).relative_to(config.ROOT))
    return None, None


def fit_model(season: int, pbp: pd.DataFrame, sched: pd.DataFrame, games: pd.DataFrame) -> dict:
    """Fit A4s for `season` on REG 2001..season-1 through the M3 sign-off code path (ml_m3.build_frame)."""
    last = season - 1
    g, p, s = (games[games["season"] <= last].reset_index(drop=True),
               pbp[pbp["game_id"].str[:4].astype(int) <= last].reset_index(drop=True),
               sched[sched["season"] <= last].reset_index(drop=True))
    frame, _ = ml_m3.build_frame(g, p, s, last_season=last)
    feats = live_features.FEATURES

    # Reproduction check: refit season-1 exactly as the walk-forward did.
    _, coefs = logistic.walk_forward(frame, feats, (last, last), train_start=logistic.TRAIN_START)
    got = {k: float(coefs.iloc[-1][k]) for k in ["intercept", *feats]}
    ref, ref_src = reference_coefs(last)
    repro = {"season": last, "refit": got, "reference": ref, "reference_file": ref_src}
    if ref is None:
        print(f"note: no reference coefficients for {last}; reproduction check skipped")
        repro["status"] = "skipped (no reference)"
    else:
        worst = max(abs(got[k] - ref[k]) for k in got)
        repro["max_abs_diff"] = worst
        if worst > COEF_TOLERANCE:
            raise SystemExit(f"refit of {last} does not reproduce {ref_src}: max |diff| {worst:.2e} "
                             f"(got {got}, expected {ref}). Stopping.")
        repro["status"] = "reproduced"
        print(f"reproduction check: {last} coefficients match {ref_src} (max |diff| {worst:.1e})")

    train = frame[(frame["season"] >= logistic.TRAIN_START) & (frame["season"] < season)]
    m = logistic.fit_logistic(train[feats].to_numpy(float), train["y"].to_numpy(float),
                              logistic.season_weights(train["season"], season))
    coef = {f: float(c) for f, c in zip(feats, m.coef_[0])}
    blob = json.dumps({"intercept": float(m.intercept_[0]), "coef": coef}, sort_keys=True)
    version = f"A4s-{season}-{hashlib.sha256(blob.encode()).hexdigest()[:8]}"
    fp = registry.data_fingerprint(games_csv=False, ml_datasets=("pbp", "schedules"),
                                   seasons=range(FIRST_DATA_SEASON, last + 1))
    return {
        "model_version": version, "season": season, "spec": model_spec(), "features": feats,
        "intercept": float(m.intercept_[0]), "coef": coef,
        "train": {"seasons": [logistic.TRAIN_START, last], "game_type": "REG", "n_games": int(len(train)),
                  "games_hash": registry.games_hash(train["game_id"]),
                  "weights": f"0.5 ** (({season} - 1 - season) / {logistic.SEASON_HALF_LIFE})"},
        "reproduction_check": repro,
        "data": fp,
        "git": registry.git_commit(),
        "fitted_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def season_model(season: int, pbp, sched, games, refit: bool) -> dict:
    path = live.model_path(season)
    if path.exists() and not refit:
        m = json.loads(path.read_text())
        if m.get("spec") != model_spec():
            raise SystemExit(f"{path} was fit with a different spec; rerun with --refit after review")
        print(f"model: reusing {path.relative_to(config.ROOT)} ({m['model_version']}, fit {m['fitted_at_utc']})")
        return m
    m = fit_model(season, pbp, sched, games)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m, indent=2, sort_keys=False) + "\n")
    print(f"model: fit {m['model_version']} on REG {m['train']['seasons'][0]}-{m['train']['seasons'][1]} "
          f"(n = {m['train']['n_games']}), saved to {path.relative_to(config.ROOT)}")
    return m


# --------------------------------------------------------------------------- predict

def predict(season: int, now: pd.Timestamp, pbp, sched, games, model: dict) -> pd.DataFrame:
    """One row per unplayed REG game of the season: model, Elo and (added last) market numbers."""
    todo = sched[(sched["season"] == season) & (sched["game_type"] == "REG") & sched["home_score"].isna()]
    if todo.empty:
        return pd.DataFrame()
    played = games[pd.to_datetime(games["date"]) < now.tz_convert("America/New_York").tz_localize(None).normalize()
                   + pd.Timedelta(days=1)]
    played = played[played["home_score"].notna()]
    elo = live_features.elo_state(played, season)

    # 1. the model, from market-free inputs
    feats = live_features.build(pbp, sched, todo, now, elo)
    out = feats.copy()
    out["p_home_model"] = live_features.predict(model, feats)
    out["kickoff_utc"] = live.to_utc(todo.set_index("game_id").loc[out.index, "kickoff"]).array

    # 2. Elo, exactly as the site exporter computes upcoming games
    elo_games = export_site.load_elo()
    ratings = export_site.load_ratings()
    picks = export_site.elo_pick_table(todo, export_site.current_elo(elo_games), ratings["hfa_current"])
    picks = picks.set_index("game_id").loc[out.index]
    out["p_home_elo"] = picks["p_home"].to_numpy(float)
    out["spread_elo"] = picks["elo_spread"].to_numpy(float)

    # 3. the market, joined only after the model has predicted (recorded for scoring, never a feature)
    t = todo.set_index("game_id").loc[out.index]
    out["p_home_market"] = market_prob(t).to_numpy(float)
    out["spread_market"] = pd.to_numeric(t["spread_line"], errors="coerce").to_numpy(float)
    out["gameday"], out["gametime"] = t["gameday"].to_numpy(), t["gametime"].to_numpy()
    return out


def ledger_rows(pred: pd.DataFrame, now: pd.Timestamp, model: dict, dhash: str) -> pd.DataFrame:
    ahead = pred[pred["kickoff_utc"] > now]
    r4 = lambda s: s.astype(float).round(4)  # noqa: E731
    rows = pd.DataFrame({
        "run_at_utc": live.utc_iso(now), "game_id": ahead.index.to_numpy(),
        "kickoff_utc": [live.utc_iso(k) for k in ahead["kickoff_utc"]],
        "model_version": model["model_version"], "p_home_model": r4(ahead["p_home_model"]).to_numpy(),
        "spread_model": np.nan,  # Phase 1: no margin model yet
        "p_home_elo": r4(ahead["p_home_elo"]).to_numpy(), "spread_elo": ahead["spread_elo"].round(1).to_numpy(),
        "p_home_market": r4(ahead["p_home_market"]).to_numpy(), "spread_market": ahead["spread_market"].to_numpy(),
        "home_qb_id": ahead["home_qb_id"].to_numpy(object), "away_qb_id": ahead["away_qb_id"].to_numpy(object),
        "qb_source": ahead["qb_source"].to_numpy(object), "data_hash": dhash,
    })
    return rows.sort_values(["kickoff_utc", "game_id"], kind="stable").reset_index(drop=True)[live.LEDGER_COLUMNS]


def latest_payload(pred: pd.DataFrame, now, model: dict, dhash: str, logged: set, next_week) -> dict:
    def f(x, d=4):
        return None if x is None or pd.isna(x) else round(float(x), d)
    games = []
    for gid, r in pred.sort_values(["kickoff_utc"], kind="stable").iterrows():
        games.append({
            "game_id": gid, "week": int(r["game_week"]), "feature_week": int(r["feature_week"]),
            "kickoff_utc": live.utc_iso(r["kickoff_utc"]), "home_team": r["home_team"], "away_team": r["away_team"],
            "p_home_model": f(r["p_home_model"]), "elo_logit": f(r["elo_logit"], 5),
            "adj_epa_margin": f(r["adj_epa_margin"], 5), "qb_delta_home": f(r["qb_delta_home"], 5),
            "qb_delta_away": f(r["qb_delta_away"], 5), "qb_delta_diff": f(r["qb_delta_diff"], 5),
            **{f"{s}_qb_{k}": (None if r[f"{s}_qb_{k}"] is None or pd.isna(r[f"{s}_qb_{k}"]) else r[f"{s}_qb_{k}"])
               for s in ("home", "away") for k in ("id", "name", "source")},
            "logged": gid in logged,
        })
    return {"season": SEASON, "run_at_utc": live.utc_iso(now), "model_version": model["model_version"],
            "data_hash": dhash, "next_feature_week": next_week,
            "note": "Every unplayed game at run time. Only games with logged = true were before kickoff and "
                    "went into the ledger; the others had already kicked off. No market data here.",
            "games": games}


# --------------------------------------------------------------------------- main

def summarize(pred: pd.DataFrame, rows: pd.DataFrame, season: int, sched: pd.DataFrame) -> None:
    excluded = pred.index.difference(rows["game_id"])
    print(f"\nUnplayed REG games: {len(pred)}. Ledger rows written: {len(rows)}. "
          f"Kicked off already (not logged): {len(pred) - len(rows)}"
          + (f" ({', '.join(sorted(excluded))})" if 0 < len(excluded) <= 16 else ""))
    side = pd.concat([pred["home_qb_source"], pred["away_qb_source"]]).value_counts().to_dict()
    print(f"QB sources over all unplayed games, per side: {side}; per logged row: "
          f"{rows['qb_source'].value_counts().to_dict() if len(rows) else {}}")
    gap = (pred["p_home_model"] - pred["p_home_market"]).dropna()
    if len(gap):
        print("Biggest model-vs-Vegas gaps (home win probability, model minus market):")
        for gid in gap.abs().sort_values(ascending=False).index[:5]:
            r = pred.loc[gid]
            print(f"  {gid:<18} model {r.p_home_model:.3f}  Vegas {r.p_home_market:.3f}  Elo {r.p_home_elo:.3f}  "
                  f"gap {100 * (r.p_home_model - r.p_home_market):+.1f} pts  qb_delta_diff {r.qb_delta_diff:+.3f}")
    reg = sched[(sched["season"] == season) & (sched["game_type"] == "REG")]
    proj = live.projected_wins(reg, pred["p_home_model"], pred["p_home_elo"]).sort_values("proj_model")
    fmt = lambda r: f"{r.team} {r.w}-{r.l}{'-' + str(r.t) if r.t else ''} model {r.proj_model:.1f} / Elo {r.proj_elo:.1f}"  # noqa: E731
    print("Projected wins, top 3:    " + "; ".join(fmt(r) for r in proj.tail(3)[::-1].itertuples()))
    print("Projected wins, bottom 3: " + "; ".join(fmt(r) for r in proj.head(3).itertuples()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--offline", action="store_true", help="use the cached nflverse files only")
    ap.add_argument("--now", help="pretend the run happens at this UTC time (tests; needs --ledger)")
    ap.add_argument("--ledger", type=Path, default=None, help="ledger file (default experiments/live/<season>.csv)")
    ap.add_argument("--refit", action="store_true", help="refit the season model even if one is saved")
    a = ap.parse_args()
    season = SEASON
    default_ledger = live.ledger_path(season)
    ledger = a.ledger or default_ledger
    if a.now and ledger.resolve() == default_ledger.resolve():
        raise SystemExit("--now writes made-up run times; point --ledger somewhere other than the real ledger")
    now = (live.to_utc(a.now) if a.now else pd.Timestamp.now(tz="UTC")).floor("s")

    refresh(season, a.offline)
    pbp, sched, games = load(season)
    check_elo_fresh(sched, games, export_site.load_elo(), season)
    model = season_model(season, pbp, sched, games, a.refit)
    dhash = data_hash(season)

    pred = predict(season, now, pbp, sched, games, model)
    if pred.empty:
        print(f"No unplayed {season} REG games; nothing to predict.")
        return 0
    rows = ledger_rows(pred, now, model, dhash)
    n = live.append_rows(rows, ledger)
    nxt = live_features.next_week(sched[sched["season"] == season], now)
    latest = latest_payload(pred, now, model, dhash, set(rows["game_id"]), nxt)
    lp = ledger.parent / live.latest_path(season).name
    lp.write_text(json.dumps(latest, indent=1) + "\n")
    print(f"run {live.utc_iso(now)}: appended {n} rows to {ledger}; wrote {lp}")
    summarize(pred, rows, season, sched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
