"""Weekly ML predictions (M4): refresh nflverse, predict every unplayed game, append to the ledger, simulate the season.

    python scripts/ml_predict.py              # refresh this season's schedule and play-by-play, predict, append
    python scripts/ml_predict.py --offline    # use the cached nflverse files only
    python scripts/ml_predict.py --now 2026-10-07T14:00:00Z --ledger /tmp/test.csv   # tests only
    python scripts/ml_predict.py --fit-margin-only   # fit and save the season's margin model, nothing else
    python scripts/ml_predict.py --fit-shadow-only   # fit and save the season's shadow model (C2d), nothing else
    python scripts/ml_predict.py --no-shadow         # skip the shadow model (the main ledger is unchanged either way)

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

M4 Phase 2 additions:
- The margin model (`nflelo.ml.models.margin`), fit once per season on the same
  frame and frozen in experiments/live/margin_<season>.json, fills the ledger's
  `spread_model` for new rows (older rows stay empty). The win probability is
  still A4s, computed exactly as before: the 2026 live record started with A4s
  and stays frozen for the season. `model_version` becomes "<A4s>+<margin>".
- The playoff simulation (`nflelo.ml.sim`, 20,000 seasons, seed = the run's Unix
  time) runs after the ledger append and adds 32 rows to the append-only
  experiments/live/sim_<season>.csv.

M3b shadow model (context/ml.md, "Shadow deployment"):
- C2d (elo_logit + Kalman kf_margin + qb_delta_diff), fit once per season on REG
  2001..season-1 with the M3b protocol and frozen in experiments/live/shadow_<season>.json
  with kalman.TUNED. The first fit refits 2019 through the same code path and stops
  unless it reproduces the logged DEV fit (m3b_C2d).
- Each run predicts every unplayed game with the Kalman state as of now (the A4s
  as-of rule) and appends the pre-kickoff rows to experiments/live/<season>_shadow.csv,
  with the same run_at_utc as the main ledger rows. The main ledger's rows and
  schema do not depend on it, and a shadow failure never blocks the A4s record.

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
from nflelo.ml.features import kalman as kf, opponent_adjust as oa, qb  # noqa: E402
from nflelo.ml.models import logistic, margin as mm  # noqa: E402
from nflelo.ml.sim import season as sim, state as sim_state  # noqa: E402

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


# --------------------------------------------------------------------------- the season margin model

def margin_spec() -> dict:
    """Everything that defines the margin model; a saved fit is reused only when this matches."""
    return {"model": "margin", "features": mm.FEATURES, "mean": "weighted least squares",
            "sigma": "weighted residual SD of the training fit", "shape": mm.CHOSEN_SHAPE,
            "keynum_fit_seasons": list(mm.KEYNUM_FIT_SEASONS), "kmax": mm.KMAX, "ratings": oa.TUNED.to_dict(),
            "qb": qb.tuned_config().to_dict(), "train_start": logistic.TRAIN_START,
            "season_half_life": logistic.SEASON_HALF_LIFE}


def margin_reference(season: int) -> tuple[dict, str] | tuple[None, None]:
    """The DEV walk-forward fit of `season` logged by scripts/ml_m4.py margin (for the reproduction check)."""
    for path in sorted(glob.glob(str(registry.RUNS_DIR / f"*_m4_margin_{mm.CHOSEN_SHAPE}.json")), reverse=True):
        rec = json.loads(Path(path).read_text())
        for c in rec.get("coefficients") or []:
            if int(c["season"]) == season:
                return c, str(Path(path).relative_to(config.ROOT))
    return None, None


def fit_margin_model(season: int, pbp: pd.DataFrame, sched: pd.DataFrame, games: pd.DataFrame) -> dict:
    """Fit the margin model for `season` on REG 2001..season-1 through the M3 frame (ml_m3.build_frame)."""
    last = season - 1
    g, p, s = (games[games["season"] <= last].reset_index(drop=True),
               pbp[pbp["game_id"].str[:4].astype(int) <= last].reset_index(drop=True),
               sched[sched["season"] <= last].reset_index(drop=True))
    frame, _ = ml_m3.build_frame(g, p, s, last_season=last)
    sc = s.drop_duplicates("game_id").set_index("game_id")
    frame["margin"] = (sc.loc[frame["game_id"], "home_score"] - sc.loc[frame["game_id"], "away_score"]).to_numpy(float)
    feats = mm.FEATURES

    # Reproduction check: the last DEV season (2019) refit on this frame must match the logged DEV fit.
    _, fits = mm.walk_forward(frame, feats, (2019, 2019))
    got = fits[0].to_dict()
    ref, ref_src = margin_reference(2019)
    repro = {"season": 2019, "refit": got, "reference_file": ref_src}
    if ref is None:
        print("note: no logged DEV margin fit for 2019; reproduction check skipped")
        repro["status"] = "skipped (no reference)"
    else:
        worst = max([abs(got["intercept"] - ref["intercept"]), abs(got["sigma"] - ref["sigma"])]
                    + [abs(got["coef"][f] - ref["coef"][f]) for f in feats])
        repro["max_abs_diff"] = worst
        if worst > COEF_TOLERANCE:
            raise SystemExit(f"margin refit of 2019 does not reproduce {ref_src}: max |diff| {worst:.2e}. Stopping.")
        repro["status"] = "reproduced"
        print(f"margin reproduction check: 2019 fit matches {ref_src} (max |diff| {worst:.1e})")

    train = frame[(frame["season"] >= logistic.TRAIN_START) & (frame["season"] < season)]
    fit = mm.fit_margin(train[feats].to_numpy(float), train["margin"].to_numpy(float),
                        logistic.season_weights(train["season"], season), feats)
    fit.season = season
    kn = mm.keynumbers_from_frame(frame) if mm.CHOSEN_SHAPE == "keynum" else None
    fp = registry.data_fingerprint(games_csv=False, ml_datasets=("pbp", "schedules"),
                                   seasons=range(FIRST_DATA_SEASON, last + 1))
    return {
        "version": mm.version(fit, mm.CHOSEN_SHAPE, kn, season), "season": season, "spec": margin_spec(),
        "fit": fit.to_dict(), "shape": mm.CHOSEN_SHAPE, "keynum": kn.to_dict() if kn is not None else None,
        "train": {"seasons": [logistic.TRAIN_START, last], "game_type": "REG", "n_games": int(len(train)),
                  "games_hash": registry.games_hash(train["game_id"]),
                  "weights": f"0.5 ** (({season} - 1 - season) / {logistic.SEASON_HALF_LIFE})"},
        "reproduction_check": repro, "data": fp, "git": registry.git_commit(),
        "fitted_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def margin_path(season: int, root: Path = live.LIVE_DIR) -> Path:
    return root / f"margin_{int(season)}.json"


def margin_model(season: int, pbp, sched, games, refit: bool) -> dict:
    path = margin_path(season)
    if path.exists() and not refit:
        m = json.loads(path.read_text())
        if m.get("spec") != margin_spec():
            raise SystemExit(f"{path} was fit with a different spec; rerun with --refit after review")
        print(f"margin model: reusing {path.relative_to(config.ROOT)} ({m['version']}, fit {m['fitted_at_utc']})")
        return m
    m = fit_margin_model(season, pbp, sched, games)
    path.write_text(json.dumps(m, indent=2) + "\n")
    print(f"margin model: fit {m['version']} on REG {m['train']['seasons'][0]}-{m['train']['seasons'][1]} "
          f"(n = {m['train']['n_games']}), sigma {m['fit']['sigma']:.2f}, shape {m['shape']}, "
          f"saved to {path.relative_to(config.ROOT)}")
    return m


def margin_objects(m: dict) -> tuple[mm.MarginFit, mm.KeyNumbers | None]:
    return mm.MarginFit.from_dict(m["fit"]), (mm.KeyNumbers.from_dict(m["keynum"]) if m.get("keynum") else None)


def combined_version(model: dict, margin: dict | None) -> str:
    return model["model_version"] if margin is None else f"{model['model_version']}+{margin['version']}"


# --------------------------------------------------------------------------- the shadow model (M3b C2d)

SHADOW_REFERENCE_SEASON = 2019   # the last DEV refit, logged by scripts/ml_m3b.py dev (m3b_C2d)


def shadow_spec() -> dict:
    """Everything that defines the shadow model; a saved fit is reused only when this matches."""
    return {"model": "C2d", "role": "shadow: logged and scored next to A4s, never shown as the model's pick",
            "features": live_features.SHADOW_FEATURES, "kind": "logistic, unpenalized", "kalman": kf.TUNED.to_dict(),
            "qb": qb.tuned_config().to_dict(), "train_start": logistic.TRAIN_START,
            "season_half_life": logistic.SEASON_HALF_LIFE, "ties": "two half-weight rows",
            "elo_config": config.DEFAULT_CONFIG.to_dict(), "qb_starter": "actual (M3-D1)",
            "hfa": "Elo online HFA frozen before the week's first game",
            "kalman_asof": "completed REG/POST games before the feature week's as_of (the A4s rule), then the "
                           "filter's time update to that week"}


def shadow_reference(season: int = SHADOW_REFERENCE_SEASON) -> tuple[dict, str] | tuple[None, None]:
    """The logged DEV walk-forward fit of `season` for C2d (scripts/ml_m3b.py dev, label dev)."""
    for path in sorted(glob.glob(str(registry.RUNS_DIR / "*_m3b_C2d.json")), reverse=True):
        rec = json.loads(Path(path).read_text())
        if rec.get("label") != "dev":
            continue
        for c in rec.get("coefficients") or []:
            if int(c["season"]) == season:
                return ({k: float(c[k]) for k in ["intercept", *live_features.SHADOW_FEATURES]},
                        str(Path(path).relative_to(config.ROOT)))
    return None, None


def shadow_frame(season: int, pbp: pd.DataFrame, sched: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """The M3 frame for REG 2001..season-1 plus kf_margin/kf_sd, built as scripts/ml_m3b.py build_frame does."""
    last = season - 1
    g, p, s = (games[games["season"] <= last].reset_index(drop=True),
               pbp[pbp["game_id"].str[:4].astype(int) <= last].reset_index(drop=True),
               sched[sched["season"] <= last].reset_index(drop=True))
    frame, _ = ml_m3.build_frame(g, p, s, last_season=last)
    reg = s.set_index("game_id").loc[frame["game_id"]].reset_index()
    k = kf.build_features(p, s, reg, kf.TUNED)
    frame["kf_margin"], frame["kf_sd"] = k["kf_margin"].to_numpy(), k["kf_sd"].to_numpy()
    if frame[["kf_margin", "kf_sd"]].isna().any().any():
        raise SystemExit("NaN in Kalman features")
    return frame


def fit_shadow_model(season: int, pbp: pd.DataFrame, sched: pd.DataFrame, games: pd.DataFrame) -> dict:
    """Fit C2d for `season` on REG 2001..season-1 with the M3b protocol, after checking it reproduces the DEV fit."""
    frame = shadow_frame(season, pbp, sched, games)
    feats = live_features.SHADOW_FEATURES
    ref_season = SHADOW_REFERENCE_SEASON
    _, coefs = logistic.walk_forward(frame, feats, (ref_season, ref_season), train_start=logistic.TRAIN_START)
    got = {k: float(coefs.iloc[-1][k]) for k in ["intercept", *feats]}
    ref, ref_src = shadow_reference(ref_season)
    if ref is None:
        raise SystemExit(f"no logged m3b_C2d DEV fit for {ref_season} in {registry.RUNS_DIR}; cannot check the "
                         "shadow fit. Stopping.")
    worst = max(abs(got[k] - ref[k]) for k in got)
    repro = {"season": ref_season, "refit": got, "reference": ref, "reference_file": ref_src,
             "n_train": int(coefs.iloc[-1]["n_train"]), "max_abs_diff": worst}
    if worst > COEF_TOLERANCE:
        raise SystemExit(f"shadow refit of {ref_season} does not reproduce {ref_src}: max |diff| {worst:.2e} "
                         f"(got {got}, expected {ref}). Stopping.")
    repro["status"] = "reproduced"
    print(f"shadow reproduction check: {ref_season} C2d coefficients match {ref_src} (max |diff| {worst:.1e})")

    train = frame[(frame["season"] >= logistic.TRAIN_START) & (frame["season"] < season)]
    m = logistic.fit_logistic(train[feats].to_numpy(float), train["y"].to_numpy(float),
                              logistic.season_weights(train["season"], season))
    coef = {f: float(c) for f, c in zip(feats, m.coef_[0])}
    blob = json.dumps({"intercept": float(m.intercept_[0]), "coef": coef, "kalman": kf.TUNED.to_dict()},
                      sort_keys=True)
    last = season - 1
    fp = registry.data_fingerprint(games_csv=False, ml_datasets=("pbp", "schedules"),
                                   seasons=range(FIRST_DATA_SEASON, last + 1))
    return {
        "model_version": f"C2d-{season}-{hashlib.sha256(blob.encode()).hexdigest()[:8]}", "season": season,
        "spec": shadow_spec(), "features": feats, "intercept": float(m.intercept_[0]), "coef": coef,
        "kalman": kf.TUNED.to_dict(),
        "train": {"seasons": [logistic.TRAIN_START, last], "game_type": "REG", "n_games": int(len(train)),
                  "games_hash": registry.games_hash(train["game_id"]),
                  "weights": f"0.5 ** (({season} - 1 - season) / {logistic.SEASON_HALF_LIFE})"},
        "reproduction_check": repro, "data": fp, "git": registry.git_commit(),
        "fitted_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def shadow_model(season: int, pbp, sched, games, refit: bool, root: Path = live.LIVE_DIR) -> dict:
    path = live.shadow_model_path(season, root)
    if path.exists() and not refit:
        m = json.loads(path.read_text())
        if m.get("spec") != shadow_spec():
            raise SystemExit(f"{path} was fit with a different spec; rerun with --refit-shadow after review")
        print(f"shadow model: reusing {path.relative_to(config.ROOT)} ({m['model_version']}, fit {m['fitted_at_utc']})")
        return m
    m = fit_shadow_model(season, pbp, sched, games)
    path.write_text(json.dumps(m, indent=2) + "\n")
    print(f"shadow model: fit {m['model_version']} on REG {m['train']['seasons'][0]}-{m['train']['seasons'][1]} "
          f"(n = {m['train']['n_games']}), saved to {path.relative_to(config.ROOT)}")
    return m


def shadow_predict(shadow: dict, pred: pd.DataFrame, pbp, sched, season: int, now: pd.Timestamp) -> pd.DataFrame:
    """The shadow model's home win probability for every game in `pred`, plus kf_margin and kf_sd.

    Reads Elo and the QB term from `pred` (the A4s features, computed once) and never writes to it.
    """
    if shadow["kalman"] != kf.TUNED.to_dict():
        raise ValueError("the saved shadow model was fit with other Kalman knobs than kalman.TUNED")
    todo = sched[(sched["season"] == season) & (sched["game_type"] == "REG")].drop_duplicates("game_id")
    todo = todo.set_index("game_id").loc[pred.index].reset_index()
    k = live_features.kalman_asof(pbp, sched, todo, now, kf.TUNED)
    x = pd.DataFrame({"elo_logit": pred["elo_logit"].to_numpy(float), "kf_margin": k["kf_margin"].to_numpy(float),
                      "qb_delta_diff": pred["qb_delta_diff"].to_numpy(float)}, index=pred.index)
    return pd.DataFrame({"p_home_shadow": live_features.predict(shadow, x), "kf_margin": x["kf_margin"],
                         "kf_sd": k["kf_sd"].to_numpy(float), "kickoff_utc": pred["kickoff_utc"]}, index=pred.index)


def shadow_rows(sp: pd.DataFrame, now: pd.Timestamp, version: str, dhash: str) -> pd.DataFrame:
    ahead = sp[sp["kickoff_utc"] > now]
    rows = pd.DataFrame({
        "run_at_utc": live.utc_iso(now), "game_id": ahead.index.to_numpy(),
        "kickoff_utc": [live.utc_iso(k) for k in ahead["kickoff_utc"]], "model_version": version,
        "p_home_shadow": ahead["p_home_shadow"].astype(float).round(4).to_numpy(),
        "kf_margin": ahead["kf_margin"].astype(float).round(2).to_numpy(),
        "kf_sd": ahead["kf_sd"].astype(float).round(2).to_numpy(), "data_hash": dhash,
    })
    return rows.sort_values(["kickoff_utc", "game_id"], kind="stable").reset_index(drop=True)[live.SHADOW_COLUMNS]


def record_run(pred: pd.DataFrame, now: pd.Timestamp, model: dict, dhash: str, version: str, ledger: Path,
               shadow: dict | None = None, shadow_pred: pd.DataFrame | None = None, season: int = SEASON
               ) -> tuple[pd.DataFrame, int, int]:
    """Append the run to the main ledger and, when there is a shadow prediction, to the shadow ledger.

    The main ledger's rows come from `pred` alone, so they are the same bytes with or without the
    shadow (tested). A shadow problem never blocks the A4s record: bad shadow rows are reported
    and skipped. Returns (main rows, main rows written, shadow rows written).
    """
    rows = ledger_rows(pred, now, model, dhash, version)
    srows = None
    if shadow is not None and shadow_pred is not None:
        srows = shadow_rows(shadow_pred, now, shadow["model_version"], dhash)
        try:
            live.check_shadow_rows(srows)
        except live.LedgerError as e:
            print(f"WARNING: shadow rows refused ({e}); the A4s ledger is unaffected", file=sys.stderr)
            srows = None
    n = live.append_rows(rows, ledger)
    ns = 0
    if srows is not None:
        spath = ledger.parent / live.shadow_path(season).name
        try:
            ns = live.append_shadow(srows, spath)
        except live.LedgerError as e:
            print(f"WARNING: shadow ledger append refused ({e}); the A4s ledger is unaffected", file=sys.stderr)
    return rows, n, ns


# --------------------------------------------------------------------------- predict

def model_columns(model: dict, margin: dict | None, feats: pd.DataFrame) -> pd.DataFrame:
    """The model's numbers for each game: A4s win probability (unchanged from Phase 1) and the margin spread.

    The probability depends only on the A4s model and the features; the margin
    model never touches it (tests/ml/test_margin.py checks this bit for bit).
    """
    out = pd.DataFrame(index=feats.index)
    out["p_home_model"] = live_features.predict(model, feats)
    out["spread_model"] = np.nan if margin is None else mm.spread(margin_objects(margin)[0], feats)
    return out


def played_before(games: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Games played by `now` (by Eastern calendar day, as the Elo pipeline records them)."""
    played = games[pd.to_datetime(games["date"]) < now.tz_convert("America/New_York").tz_localize(None).normalize()
                   + pd.Timedelta(days=1)]
    return played[played["home_score"].notna()]


def predict(season: int, now: pd.Timestamp, pbp, sched, games, model: dict, margin: dict | None = None) -> pd.DataFrame:
    """One row per unplayed REG game of the season: model, Elo and (added last) market numbers."""
    todo = sched[(sched["season"] == season) & (sched["game_type"] == "REG") & sched["home_score"].isna()]
    if todo.empty:
        return pd.DataFrame()
    elo = live_features.elo_state(played_before(games, now), season)

    # 1. the model, from market-free inputs
    feats = live_features.build(pbp, sched, todo, now, elo)
    out = feats.copy()
    cols = model_columns(model, margin, feats)
    out["p_home_model"] = cols["p_home_model"]
    out["spread_model"] = cols["spread_model"]
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


def ledger_rows(pred: pd.DataFrame, now: pd.Timestamp, model: dict, dhash: str, version: str | None = None) -> pd.DataFrame:
    ahead = pred[pred["kickoff_utc"] > now]
    r4 = lambda s: s.astype(float).round(4)  # noqa: E731
    rows = pd.DataFrame({
        "run_at_utc": live.utc_iso(now), "game_id": ahead.index.to_numpy(),
        "kickoff_utc": [live.utc_iso(k) for k in ahead["kickoff_utc"]],
        "model_version": version or model["model_version"], "p_home_model": r4(ahead["p_home_model"]).to_numpy(),
        "spread_model": (ahead["spread_model"].astype(float).round(1).to_numpy() if "spread_model" in ahead
                         else np.nan),
        "p_home_elo": r4(ahead["p_home_elo"]).to_numpy(), "spread_elo": ahead["spread_elo"].round(1).to_numpy(),
        "p_home_market": r4(ahead["p_home_market"]).to_numpy(), "spread_market": ahead["spread_market"].to_numpy(),
        "home_qb_id": ahead["home_qb_id"].to_numpy(object), "away_qb_id": ahead["away_qb_id"].to_numpy(object),
        "qb_source": ahead["qb_source"].to_numpy(object), "data_hash": dhash,
    })
    return rows.sort_values(["kickoff_utc", "game_id"], kind="stable").reset_index(drop=True)[live.LEDGER_COLUMNS]


def latest_payload(pred: pd.DataFrame, now, model: dict, dhash: str, logged: set, next_week,
                   version: str | None = None) -> dict:
    def f(x, d=4):
        return None if x is None or pd.isna(x) else round(float(x), d)
    games = []
    for gid, r in pred.sort_values(["kickoff_utc"], kind="stable").iterrows():
        games.append({
            "game_id": gid, "week": int(r["game_week"]), "feature_week": int(r["feature_week"]),
            "kickoff_utc": live.utc_iso(r["kickoff_utc"]), "home_team": r["home_team"], "away_team": r["away_team"],
            "p_home_model": f(r["p_home_model"]), "spread_model": f(r.get("spread_model"), 1),
            "elo_logit": f(r["elo_logit"], 5),
            "adj_epa_margin": f(r["adj_epa_margin"], 5), "qb_delta_home": f(r["qb_delta_home"], 5),
            "qb_delta_away": f(r["qb_delta_away"], 5), "qb_delta_diff": f(r["qb_delta_diff"], 5),
            **{f"{s}_qb_{k}": (None if r[f"{s}_qb_{k}"] is None or pd.isna(r[f"{s}_qb_{k}"]) else r[f"{s}_qb_{k}"])
               for s in ("home", "away") for k in ("id", "name", "source")},
            "logged": gid in logged,
        })
    return {"season": SEASON, "run_at_utc": live.utc_iso(now), "model_version": version or model["model_version"],
            "data_hash": dhash, "next_feature_week": next_week,
            "note": "Every unplayed game at run time. Only games with logged = true were before kickoff and "
                    "went into the ledger; the others had already kicked off. No market data here.",
            "games": games}


# --------------------------------------------------------------------------- playoff simulation

def sim_rows(out: pd.DataFrame, now, n_sims: int, tau: float, shape: str, seed: int, version: str) -> pd.DataFrame:
    r = out.reset_index()
    rows = pd.DataFrame({"run_at_utc": live.utc_iso(now), "team": r["team"]})
    for c in live.SIM_PROBS:
        rows[c] = r[c].round(4)
    for c in ("wins_mean", "wins_p10", "wins_p90"):
        rows[c] = r[c].round(2)
    rows["n_sims"], rows["tau_rest"], rows["shape"], rows["seed"], rows["model_version"] = n_sims, tau, shape, seed, version
    return rows[live.SIM_COLUMNS]


def run_sim(season: int, now: pd.Timestamp, pbp, sched, games, margin: dict, version: str, out_dir: Path,
            n_sims: int = sim.DEFAULT_SIMS) -> pd.DataFrame:
    """Simulate the rest of the season from the current state and append the run to sim_<season>.csv."""
    import time
    fit, kn = margin_objects(margin)
    t0 = time.perf_counter()
    inp, _, _ = sim_state.state_at(pbp, sched, played_before(games, now), season, now, fit, margin["shape"], kn)
    t1 = time.perf_counter()
    seed = int(now.timestamp())
    tm: dict = {}
    out = sim.simulate(inp, n_sims, tau=sim.TAU_REST, seed=seed, timings=tm)
    t2 = time.perf_counter()
    rows = sim_rows(out, now, n_sims, sim.TAU_REST, margin["shape"], seed, version)
    path = out_dir / live.sim_path(season).name
    live.append_sim(rows, path)
    print(f"simulation: {n_sims:,} seasons, {tm['remaining_games']} games left, tau_rest {sim.TAU_REST:g}, seed {seed}; "
          f"state {t1 - t0:.1f}s, simulate {t2 - t1:.1f}s (draws {tm['draw_s']:.1f}s, tiebreakers {tm['seed_s']:.1f}s, "
          f"bracket {tm['bracket_s']:.1f}s); appended {len(rows)} rows to {path}")
    if not live.publish_flags(out_dir if (out_dir / live.PUBLISH_FILE).exists() else live.LIVE_DIR)["publish_sim"]:
        print("Playoff odds are recorded but not shown on the site: publish_sim is false in "
              "experiments/live/publish.json (it opens after the M4 sign-off).")
    o = out.sort_values(["playoffs", "win_sb"], ascending=False)
    fmt = lambda t, r: f"{t} {100 * r.playoffs:.0f}% (SB {100 * r.win_sb:.1f}%)"  # noqa: E731
    print("Playoff odds, top 5:    " + "; ".join(fmt(t, r) for t, r in o.head(5).iterrows()))
    print("Playoff odds, bottom 5: " + "; ".join(fmt(t, r) for t, r in o.tail(5).iterrows()))
    return out


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
    ap.add_argument("--refit", action="store_true", help="refit the season A4s model even if one is saved")
    ap.add_argument("--refit-margin", action="store_true", help="refit the season margin model even if one is saved")
    ap.add_argument("--fit-margin-only", action="store_true", help="fit and save the margin model, then stop")
    ap.add_argument("--no-shadow", action="store_true", help="skip the shadow model (C2d)")
    ap.add_argument("--refit-shadow", action="store_true", help="refit the season shadow model even if one is saved")
    ap.add_argument("--fit-shadow-only", action="store_true", help="fit and save the shadow model, then stop")
    ap.add_argument("--sims", type=int, default=sim.DEFAULT_SIMS, help="simulated seasons (default 20,000)")
    a = ap.parse_args()
    season = SEASON
    default_ledger = live.ledger_path(season)
    ledger = a.ledger or default_ledger
    if a.now and ledger.resolve() == default_ledger.resolve():
        raise SystemExit("--now writes made-up run times; point --ledger somewhere other than the real ledger")
    now = (live.to_utc(a.now) if a.now else pd.Timestamp.now(tz="UTC")).floor("s")

    refresh(season, a.offline)
    pbp, sched, games = load(season)
    if a.fit_margin_only:
        margin_model(season, pbp, sched, games, a.refit_margin)
        return 0
    if a.fit_shadow_only:
        shadow_model(season, pbp, sched, games, a.refit_shadow)
        return 0
    check_elo_fresh(sched, games, export_site.load_elo(), season)
    model = season_model(season, pbp, sched, games, a.refit)
    margin = margin_model(season, pbp, sched, games, a.refit_margin)
    shadow = None if a.no_shadow else shadow_model(season, pbp, sched, games, a.refit_shadow)
    version = combined_version(model, margin)
    dhash = data_hash(season)

    pred = predict(season, now, pbp, sched, games, model, margin)
    if pred.empty:
        print(f"No unplayed {season} REG games; nothing to predict.")
        return 0
    sp = None
    if shadow is not None:
        try:
            sp = shadow_predict(shadow, pred, pbp, sched, season, now)
        except Exception as e:  # a shadow bug must not block the A4s record
            print(f"WARNING: shadow model failed ({e!r}); the A4s ledger is unaffected", file=sys.stderr)
    rows, n, ns = record_run(pred, now, model, dhash, version, ledger, shadow, sp, season)
    nxt = live_features.next_week(sched[sched["season"] == season], now)
    latest = latest_payload(pred, now, model, dhash, set(rows["game_id"]), nxt, version)
    lp = ledger.parent / live.latest_path(season).name
    lp.write_text(json.dumps(latest, indent=1) + "\n")
    print(f"run {live.utc_iso(now)}: appended {n} rows to {ledger}; wrote {lp}")
    if shadow is not None:
        spath = ledger.parent / live.shadow_path(season).name
        print(f"shadow {shadow['model_version']}: appended {ns} rows to {spath}")
        if sp is not None and ns:
            d = (sp["p_home_shadow"] - pred["p_home_model"]).abs()
            print(f"  shadow vs A4s, |p difference| over unplayed games: mean {d.mean():.3f}, max {d.max():.3f}")
    summarize(pred, rows, season, sched)
    run_sim(season, now, pbp, sched, games, margin, version, ledger.parent, a.sims)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
