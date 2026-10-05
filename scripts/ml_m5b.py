"""M5b on-field player ratings (Bayesian RAPM): cleaning, tuning, dev evaluation, leakage, sign-off.

LICENSE: CC BY-SA 4.0. This script reads nflverse participation data (NFL Next
Gen Stats via nflverse 2016-2022, FTN Data via nflverse 2023+), so everything it
writes (ratings, predictions, run records' derived numbers) is CC BY-SA 4.0.
Nothing here feeds A4s or any CC BY artifact. Spec: context/ml-m5b-method.md.

    python scripts/ml_m5b.py clean            # cleaning report 2016-2025 (counts only; nothing is scored)
    python scripts/ml_m5b.py tune             # tune on the dev predictions (2018, 2019); logs m5b_tune
    python scripts/ml_m5b.py dev              # dev table + secondaries + sanity at the frozen settings; logs runs
    python scripts/ml_m5b.py leakcheck        # real-data leakage check for season 2019, with the positive control
    python scripts/ml_m5b.py signoff-dry-run  # the sign-off code path on dev 2018-2019, no logging; must reproduce `dev`
    python scripts/ml_m5b.py signoff          # the ONE pre-registered holdout run, 2020-2025 (refuses if one exists)
    add --no-log to skip the registry (dev and tune)

Season-ahead evaluation (spec section 4). Ratings for season S are fit on the
plays of seasons before S and predict every kept play of S (REG and POST)
from the 22 players on the field plus the situation terms. Baselines on the
same plays, each fit on the same training plays with the same season weights
(decay ** (S-1-T)) and the same situation terms:
  team       M3 opponent-adjust ratings as of S week 1 (TUNED knobs): mu + O[off] + D[def] + h * home,
             plus the situation terms fit on the training residuals (each training season T against
             the ratings as of T+1 week 1, i.e. the end of T);
  box        M5 box-score values at the start of S summed over the 11 offensive and the 11 defensive
             players (two coefficients) plus the situation terms, least squares on the training plays
             with each training season's own start-of-season values;
  intercept  the weighted mean EPA of the training plays.
Metric: play-level EPA MSE. CIs: paired bootstrap resampling games (2,000 reps, seed 20261003).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

import ml_m3  # noqa: E402
from nflelo import config  # noqa: E402
from nflelo import data as elo_data  # noqa: E402
from nflelo.ml import asof as asof_mod  # noqa: E402
from nflelo.ml import data as mldata  # noqa: E402
from nflelo.ml import registry  # noqa: E402
from nflelo.ml.eval import baselines, bootstrap, windows  # noqa: E402
from nflelo.ml.features import opponent_adjust as oa  # noqa: E402
from nflelo.ml.features import roster as rf  # noqa: E402
from nflelo.ml.models import logistic  # noqa: E402
from nflelo.ml.players import credit as cr  # noqa: E402
from nflelo.ml.players import inputs as m5in  # noqa: E402
from nflelo.ml.players import lineup as lu  # noqa: E402
from nflelo.ml.players import participation as pt  # noqa: E402
from nflelo.ml.players import positions as pos  # noqa: E402
from nflelo.ml.players import rapm  # noqa: E402
from nflelo.ml.players import value as pv  # noqa: E402

FIRST = 2016                      # participation starts here
DEV = windows.M5B_DEV             # (2018, 2019)
HOLDOUT = windows.HOLDOUT         # (2020, 2025)
OUT = mldata.ML_DIR / "m5b"
SEED = 20261003
REPS = 2000
LAM_GRID = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 32000.0, 64000.0, 128000.0, 256000.0,
            512000.0, 1024000.0)
DECAY_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
BOX_SCALE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
SANITY_MIN_SNAPS = 300
Y2Y_MIN_SNAPS = 200
LICENSE = {"license": "CC BY-SA 4.0", "note": pt.LICENSE_NOTE}
MODELS = ("rapm", "team", "box", "intercept")
PBP_EXTRA = sorted(set(pt.PBP_COLUMNS) | set(ml_m3.PBP_COLUMNS))


# --------------------------------------------------------------------------- inputs

@dataclass
class Ctx:
    """Everything the evaluation needs for seasons FIRST..last. Built from data through `last` only."""
    last: int
    inp: m5in.M5Inputs
    plays: pd.DataFrame
    data: rapm.RapmData
    tab: pd.DataFrame                 # M3 rating table
    _priors: dict = field(default_factory=dict)
    _ratings: dict = field(default_factory=dict)
    _box: dict = field(default_factory=dict)

    def positions_before(self, S: int) -> pd.DataFrame:
        """Position table from seasons before S only (plus the players-table fallback, season -1)."""
        return self.inp.positions[self.inp.positions["season"] <= S - 1]

    def box_inputs(self, S: int) -> tuple[pd.DataFrame, pv.BoxValues]:
        """M5 player-game credits rebuilt from data before S only.

        `credit.lookup_groups` falls back to a player's LATEST position-table
        entry when his season has none, which can be a later season; the
        real-data leakage check caught that moving the start-of-S box values.
        Rebuilding the credits per S from play-by-play, depth charts and
        position tables cut at S-1 removes the path.
        """
        if S not in self._box:
            inp = self.inp
            pbp = inp.pbp[inp.pbp["season"] < S]
            depth = inp.depth[inp.depth["season"] < S] if len(inp.depth) else inp.depth
            pg = cr.player_games(pbp, inp.sched, self.positions_before(S), depth, inp.availability)
            self._box[S] = (pg, pv.BoxValues(pg, inp.sched))
        return self._box[S]

    def priors(self, S: int) -> pd.DataFrame:
        """Prior table for the season-S fit: group as of S-1 and the box value at the start of S (data < S only)."""
        if S in self._priors:
            return self._priors[S]
        cfg = pv.TUNED
        p, bv = self.box_inputs(S)
        as_of = int(np.iinfo(np.int64).max)  # every credited row is before S already
        ord_now = int(bv.ord.max()) + 1 if len(bv.ord) else 1
        R = pv.replacement_levels(p, [S], cfg.early_games)
        bs = bv.sums_at(as_of, ord_now, S, cfg.half_life, cfg.min_weight)
        vals = pd.DataFrame({"v_box": bs.values(cfg.kd, R[S]), "n_box": bs.n, "box_group": bs.group},
                            index=pd.Index(bs.players, name="gsis_id"))
        ids = pd.Index(sorted(set(vals.index) | set(self.data.index.ids)), name="gsis_id")
        grp = cr.lookup_groups(self.positions_before(S), np.full(len(ids), S - 1), ids.to_numpy(object))
        out = pd.DataFrame({"group": grp}, index=ids).join(vals)
        out["v_box"] = out["v_box"].fillna(0.0)
        out["n_box"] = out["n_box"].fillna(0.0)
        self._priors[S] = out
        return out

    def team_ratings(self, keys) -> pd.DataFrame:
        keys = sorted(set(keys) - set(self._ratings))
        if keys:
            r = oa.compute_ratings(self.tab, self.inp.sched, keys, oa.TUNED, ("all",))
            for k, g in r.groupby(["season", "week"]):
                self._ratings[(int(k[0]), int(k[1]))] = g
        return self._ratings


def build_ctx(last: int, pbp_override=None, sched_override=None, tables_override=None, part_override=None,
              use_cache: bool = True) -> Ctx:
    """Load M5 inputs 1999..last and participation plays FIRST..last (overrides: the leakage check)."""
    inp = m5in.load(last, extra_pbp_columns=PBP_EXTRA)
    if pbp_override is not None:
        inp.pbp = pbp_override
    if sched_override is not None:
        inp.sched = sched_override
    if tables_override is not None:
        inp.depth, inp.injuries, inp.rosters = (tables_override["depth"], tables_override["injuries"],
                                                tables_override["rosters"])
        inp.positions = cr.position_table(inp.depth, inp.rosters, inp.players)
    if part_override is None and pbp_override is None:
        plays = pt.load_plays(range(FIRST, last + 1), use_cache=use_cache)
    else:
        parts = []
        reps = []
        for S in range(FIRST, last + 1):
            part = part_override[S] if part_override is not None else mldata.load_participation([S])
            pb = inp.pbp[inp.pbp["season"] == S]
            x = pt.build_plays(pb, part, inp.sched[inp.sched["season"] == S], inp.players)
            reps.extend(x.attrs["report"])
            parts.append(x)
        plays = pd.concat(parts, ignore_index=True)
        plays.attrs["report"] = reps
    tab = oa.rating_table(inp.pbp, inp.sched, oa.TUNED)
    return Ctx(last, inp, plays, rapm.RapmData(plays), tab)


def first_week(sched: pd.DataFrame, S: int) -> int:
    return int(sched.loc[sched["season"] == S, "week"].min())


# --------------------------------------------------------------------------- models

def train_weights(season: np.ndarray, S: int, decay: float) -> np.ndarray:
    w = np.where(season < S, np.power(float(decay), np.maximum(S - 1 - season, 0).astype(float)), 0.0)
    return w


def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return beta


def team_offsets(ctx: Ctx, rows: np.ndarray, S_of_ratings) -> np.ndarray:
    """mu + O[off] + D[def] + h * home for the given play rows, from ratings as of (season, week) keys."""
    plays = ctx.data.plays.iloc[rows]
    out = np.zeros(len(rows))
    keys = S_of_ratings(plays["season"].to_numpy(int))
    rat = ctx.team_ratings(set(keys))
    for k in set(keys):
        m = np.array([kk == k for kk in keys])
        r = rat[k].set_index("team")
        mu, h = float(r["mu"].iloc[0]), float(r["h"].iloc[0])
        p = plays[m]
        O = r["off"].reindex(p["off_team"]).fillna(0.0).to_numpy()
        D = r["def"].reindex(p["def_team"]).fillna(0.0).to_numpy()
        out[m] = mu + O + D + h * p["home"].to_numpy(float)
    return out


def box_sums(ctx: Ctx, rows: np.ndarray, priors_for) -> np.ndarray:
    """(n x 2): sum of the 11 offensive players' box values (offense groups) and the 11 defenders' (defense groups)."""
    plays = ctx.data.plays.iloc[rows]
    out = np.zeros((len(rows), 2))
    for T in sorted(set(plays["season"].astype(int))):
        m = (plays["season"] == T).to_numpy()
        pr = priors_for(T)
        side_ok = pr["group"].map(lambda g: pos.SIDE.get(g) if isinstance(g, str) else None)
        v_off = pr["v_box"].where(side_ok == "off", 0.0)
        v_def = pr["v_box"].where(side_ok == "def", 0.0)
        o = plays.loc[m, pt.OFF_COLS].to_numpy(object)
        d = plays.loc[m, pt.DEF_COLS].to_numpy(object)
        out[m, 0] = v_off.reindex(o.ravel()).fillna(0.0).to_numpy().reshape(o.shape).sum(axis=1)
        out[m, 1] = v_def.reindex(d.ravel()).fillna(0.0).to_numpy().reshape(d.shape).sum(axis=1)
    return out


def predict_season(ctx: Ctx, S: int, cfg: rapm.RapmConfig, models=MODELS) -> dict:
    """Season-ahead predictions for every kept play of S, for each model. Uses plays of seasons < S to fit."""
    d = ctx.data
    test = d.rows(S)
    train = np.flatnonzero(d.season < S)
    w = train_weights(d.season[train], S, cfg.decay)
    keep = w > 0
    train, w = train[keep], w[keep]
    fc = rapm.fixed_columns(cfg.situation)
    Ztr, Zte = d.Z[train][:, fc], d.Z[test][:, fc]
    y = d.y[train]
    out = {}
    if "rapm" in models:
        fit = d.fit(S, ctx.priors(S), cfg)
        out["rapm"] = d.predict_rows(fit, test)
        out["_fit"] = fit
    if "team" in models:
        fw = first_week(ctx.inp.sched, S)
        # training season T is offset by the ratings as of T+1 week 1 (the end of T); S by the ratings as of S week 1
        key_of = lambda seasons: [(int(t) + 1, first_week(ctx.inp.sched, int(t) + 1)) for t in seasons]  # noqa: E731
        off_tr = team_offsets(ctx, train, key_of)
        off_te = team_offsets(ctx, test, lambda seasons: [(S, fw)] * len(seasons))
        g = _wls(Ztr, y - off_tr, w)
        out["team"] = off_te + Zte @ g
    if "box" in models:
        btr = box_sums(ctx, train, ctx.priors)
        bte = box_sums(ctx, test, lambda T: ctx.priors(S))
        g = _wls(np.column_stack([btr, Ztr]), y, w)
        out["box"] = np.column_stack([bte, Zte]) @ g
        out["_box_coef"] = g[:2]
    if "intercept" in models:
        out["intercept"] = np.full(len(test), float(np.sum(w * y) / np.sum(w)))
    return out


# --------------------------------------------------------------------------- scoring

def margin_table(plays: pd.DataFrame, preds: dict, sched: pd.DataFrame) -> pd.DataFrame:
    """Per game: actual and predicted home-minus-away offensive EPA per play (kept plays)."""
    info = sched.drop_duplicates("game_id").set_index("game_id")
    home = pos.map_teams(info["home_team"], info["season"])
    p = plays[["game_id", "off_team", "epa"]].copy()
    for m in MODELS:
        p[m] = preds[m]
    g = p.groupby(["game_id", "off_team"]).mean(numeric_only=True)
    rows = []
    for gid, x in g.groupby(level=0):
        x = x.droplevel(0)
        h = home.get(gid)
        if h not in x.index or len(x) != 2:
            continue
        a = [t for t in x.index if t != h][0]
        rec = {"game_id": gid, "actual": x.at[h, "epa"] - x.at[a, "epa"]}
        for m in MODELS:
            rec[m] = x.at[h, m] - x.at[a, m]
        rows.append(rec)
    return pd.DataFrame(rows)


def score(plays: pd.DataFrame, preds: dict, sched: pd.DataFrame) -> dict:
    y = plays["epa"].to_numpy(float)
    games = plays["game_id"].to_numpy(object)
    err = {m: (y - preds[m]) ** 2 for m in MODELS}
    res = {"n_plays": int(len(y)), "n_games": int(len(set(games))),
           "mse": {m: float(err[m].mean()) for m in MODELS}, "vs": {}}
    for b in ("team", "box", "intercept"):
        res["vs"][b] = bootstrap.cluster_bootstrap_diff(err["rapm"] - err[b], games, REPS, SEED)
    for a_, b_ in (("team", "intercept"), ("box", "intercept"), ("box", "team")):
        res["vs"][f"{a_}_vs_{b_}"] = bootstrap.cluster_bootstrap_diff(err[a_] - err[b_], games, REPS, SEED)
    mt = margin_table(plays, preds, sched)
    me = {m: (mt["actual"] - mt[m]) ** 2 for m in MODELS}
    res["margin"] = {"n_games": int(len(mt)), "mse": {m: float(me[m].mean()) for m in MODELS},
                     "vs": {b: bootstrap.paired_bootstrap_diff(me["rapm"] - me[b], REPS, SEED)
                            for b in ("team", "box", "intercept")}}
    return res


def evaluate(ctx: Ctx, seasons, cfg: rapm.RapmConfig) -> tuple[dict, pd.DataFrame]:
    """Season-ahead predictions and scores, pooled over `seasons` and per season."""
    parts, per = [], {}
    fits, boxc = {}, {}
    for S in seasons:
        pr = predict_season(ctx, S, cfg)
        fits[S] = pr.pop("_fit")
        boxc[S] = pr.pop("_box_coef").tolist()
        rows = ctx.data.rows(S)
        f = ctx.data.plays.iloc[rows][["game_id", "play_id", "season", "week", "season_type", "off_team",
                                       "def_team", "epa"]].copy()
        for m in MODELS:
            f[m] = pr[m]
        parts.append(f)
    frame = pd.concat(parts, ignore_index=True)
    preds = {m: frame[m].to_numpy() for m in MODELS}
    res = {"seasons": list(seasons), "cfg": cfg.to_dict(), "pooled": score(frame, preds, ctx.inp.sched),
           "box_coef": boxc,
           "fits": {S: {"sigma2": f.sigma2, "n_train": f.n_train, "gamma": f.gamma,
                        "active_players": int(f.active.sum())} for S, f in fits.items()}}
    for S in seasons:
        m = frame["season"] == S
        sub = frame[m]
        per[S] = score(sub, {k: sub[k].to_numpy() for k in MODELS}, ctx.inp.sched)
    res["per_season"] = per
    res["games_hash"] = registry.games_hash(frame["game_id"].unique())
    res["plays_hash"] = registry.games_hash([f"{g}:{p}" for g, p in zip(frame["game_id"], frame["play_id"])])
    return res, frame


def fmt_ci(c: dict, nd: int = 4) -> str:
    return f"{c['diff']:+.{nd}f} [{c['ci_low']:+.{nd}f}, {c['ci_high']:+.{nd}f}]"


def print_eval(res: dict, title: str) -> None:
    p = res["pooled"]
    print(f"\n{title}: {p['n_plays']:,} plays in {p['n_games']} games (seasons {res['seasons']})")
    print(f"  {'model':<10}{'MSE':>9}   RAPM minus model, 95% game-cluster CI")
    for m in MODELS:
        tail = fmt_ci(p["vs"][m], 5) if m != "rapm" else ""
        print(f"  {m:<10}{p['mse'][m]:>9.5f}   {tail}")
    for k in ("team_vs_intercept", "box_vs_intercept", "box_vs_team"):
        print(f"  ({k}: {fmt_ci(p['vs'][k], 5)})")
    for S, r in res["per_season"].items():
        print(f"  {S}: " + "  ".join(f"{m} {r['mse'][m]:.5f}" for m in MODELS)
              + "  | RAPM-team " + fmt_ci(r["vs"]["team"], 5) + "  RAPM-box " + fmt_ci(r["vs"]["box"], 5))
    mg = p["margin"]
    print(f"  Team-game EPA margin MSE ({mg['n_games']} games): "
          + "  ".join(f"{m} {mg['mse'][m]:.5f}" for m in MODELS))
    print("    RAPM minus team " + fmt_ci(mg["vs"]["team"]) + "; minus box " + fmt_ci(mg["vs"]["box"])
          + "; minus intercept " + fmt_ci(mg["vs"]["intercept"]))


# --------------------------------------------------------------------------- stages: clean

def stage_clean() -> int:
    """Cleaning counts for 2016-2025 and players per season, position and source era. Nothing is scored."""
    plays = pt.load_plays(range(FIRST, HOLDOUT[1] + 1))
    rep = pt.cleaning_report(plays)
    print("Participation cleaning (kept = exactly 11 + 11 distinct GSIS IDs after the side repair):")
    print(rep.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    sn = pt.snaps(plays)
    players = mldata.load_players()
    yrs = range(FIRST, HOLDOUT[1] + 1)
    sched = asof_mod.add_asof(mldata.load_schedules(yrs))
    tab = cr.position_table(mldata.load_depth_charts(yrs, sched), mldata.load_rosters_weekly(yrs), players)
    sn["group"] = cr.lookup_groups(tab, sn["season"].to_numpy(int), sn["gsis_id"].to_numpy(object))
    sn["group"] = sn["group"].fillna("unknown")
    sn["era"] = sn["season"].map(pt.era)
    per = sn.groupby(["season", "era"]).agg(players=("gsis_id", "nunique")).reset_index()
    print("\nDistinct players on the field per season:")
    print(per.to_string(index=False))
    t = sn.groupby(["season", "group"])["gsis_id"].nunique().unstack(fill_value=0)
    print("\nPlayers per season and position group (group from that season's depth charts, then rosters):")
    print(t.to_string())
    e = sn.groupby(["era", "group"])["gsis_id"].nunique().unstack(fill_value=0)
    snaps_per = sn.groupby(["season"])["snaps"].sum() / plays.groupby("season").size()
    print("\nBy source era (distinct players):")
    print(e.to_string())
    print("\nOn-field player-snaps per kept play (must be 22): " + ", ".join(f"{s} {v:.2f}" for s, v in snaps_per.items()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cleaning.json").write_text(json.dumps({**LICENSE, "report": rep.to_dict("records"),
                                                    "players_per_season": per.to_dict("records")},
                                                   indent=1, default=float) + "\n")
    return 0


# --------------------------------------------------------------------------- stages: tune

def tune_objective(ctx: Ctx, cfg: rapm.RapmConfig, seasons=DEV) -> float:
    se, n = 0.0, 0
    for S in range(seasons[0], seasons[1] + 1):
        fit = ctx.data.fit(S, ctx.priors(S), cfg)
        rows = ctx.data.rows(S)
        e = ctx.data.y[rows] - ctx.data.predict_rows(fit, rows)
        se += float(e @ e)
        n += len(rows)
    return se / n


def stage_tune(log: bool) -> int:
    windows.check(DEV)
    t0 = time.perf_counter()
    ctx = build_ctx(DEV[1])
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    path = []
    cache = {}

    def obj(c: rapm.RapmConfig) -> float:
        k = json.dumps(c.to_dict(), sort_keys=True)
        if k not in cache:
            cache[k] = tune_objective(ctx, c)
            path.append({"cfg": c.to_dict(), "mse": cache[k]})
        return cache[k]

    cfg = rapm.RapmConfig()
    best = obj(cfg)
    print(f"Start {cfg.to_dict()}: MSE {best:.6f}")
    for sweep in range(4):
        changed = False
        for dec in DECAY_GRID:
            c = rapm.RapmConfig(cfg.lam, cfg.lam_other, dec, cfg.situation, cfg.box_scale)
            v = obj(c)
            if v < best - 1e-12:
                best, cfg, changed = v, c, True
        for g in rapm.GROUPS:
            for lam in LAM_GRID:
                c = cfg.with_lam(**{g: lam})
                v = obj(c)
                if v < best - 1e-12:
                    best, cfg, changed = v, c, True
        for b in BOX_SCALE_GRID:
            c = rapm.RapmConfig(cfg.lam, cfg.lam_other, cfg.decay, cfg.situation, b)
            v = obj(c)
            if v < best - 1e-12:
                best, cfg, changed = v, c, True
        for sit in (True, False):
            c = rapm.RapmConfig(cfg.lam, cfg.lam_other, cfg.decay, sit, cfg.box_scale)
            v = obj(c)
            if v < best - 1e-12:
                best, cfg, changed = v, c, True
        print(f"Sweep {sweep + 1}: MSE {best:.6f}  decay {cfg.decay}  box_scale {cfg.box_scale}  situation {cfg.situation}  "
              + " ".join(f"{g} {int(v)}" for g, v in cfg.lamd.items()))
        if not changed:
            break
    sit_off = obj(rapm.RapmConfig(cfg.lam, cfg.lam_other, cfg.decay, not cfg.situation, cfg.box_scale))
    edges = [g for g, v in cfg.lamd.items() if v in (LAM_GRID[0], LAM_GRID[-1])]
    edges += [k for k, v, grid in (("decay", cfg.decay, DECAY_GRID), ("box_scale", cfg.box_scale, BOX_SCALE_GRID))
              if v in (grid[0], grid[-1])]
    diag = {f"box_scale_{b}": obj(rapm.RapmConfig(cfg.lam, cfg.lam_other, cfg.decay, cfg.situation, b))
            for b in (0.0, 1.0)}
    diag["all_lam_equal_best"] = min(obj(rapm.RapmConfig(tuple((g, lam) for g in rapm.GROUPS), cfg.lam_other,
                                                         cfg.decay, cfg.situation, cfg.box_scale)) for lam in LAM_GRID)
    t_tune = time.perf_counter() - t0
    print(f"\nChosen: {cfg.to_dict()}  dev MSE {best:.6f}")
    print(f"Situation terms {'on' if cfg.situation else 'off'}; the other choice scores {sit_off:.6f} "
          f"({sit_off - best:+.6f}).")
    print(f"Ridge strengths at a grid edge: {edges or 'none'}")
    print("Diagnostics: "
          + ", ".join(f"{k} {v:.6f} ({v - best:+.6f})" for k, v in diag.items()))
    print(f"{len(cache)} settings scored; timing: load {t_load:.0f}s, tuning {t_tune:.0f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {**LICENSE, "chosen": cfg.to_dict(), "mse": best, "situation_other": sit_off, "edges": edges,
           "diagnostics": diag, "path": path, "timing": {"load_s": t_load, "tune_s": t_tune}}
    (OUT / "tuning.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    if log:
        registry.log_run("m5b_tune", label="m5bdev", seasons=DEV, game_ids=ctx.data.plays.loc[
            ctx.data.season >= DEV[0], "game_id"].unique(), metrics={"n": int((ctx.data.season >= DEV[0]).sum()),
                                                                   "mse": best},
            params={"chosen": cfg.to_dict(), "grid": {"lam": LAM_GRID, "decay": DECAY_GRID, "box_scale": BOX_SCALE_GRID}},
            data=fingerprint(DEV[1]), notes="M5b tuning on the dev season-ahead predictions (play-level EPA MSE). "
            "CC BY-SA 4.0 (participation-derived).",
            extra={"window": {"seasons": list(DEV), "game_type": "REG+POST"}, "license": LICENSE,
                   "tuning": {k: v for k, v in rec.items() if k not in ("license", "note")}})
    return 0


# --------------------------------------------------------------------------- secondaries

def ratings_through(ctx: Ctx, T: int, cfg: rapm.RapmConfig, sd: bool = False) -> rapm.RapmFit:
    """Ratings fit on seasons FIRST..T (the ones used to predict T+1)."""
    return ctx.data.fit(T + 1, ctx.priors(T + 1), cfg, sd=sd)


def year_to_year(ctx: Ctx, cfg: rapm.RapmConfig, seasons) -> pd.DataFrame:
    """Correlation of ratings fit through T and through T+1, by group, for players with enough snaps in both."""
    sn = pt.snaps(ctx.data.plays)
    sn["key"] = sn["side"] + "|" + sn["gsis_id"]
    rows = []
    for label, c in (("chained", cfg), ("single-season", rapm.RapmConfig(cfg.lam, cfg.lam_other, 0.0,
                                                                         cfg.situation, cfg.box_scale))):
        rt = {T: ratings_through(ctx, T, c).ratings() for T in seasons}
        for T in seasons[:-1]:
            a, b = rt[T], rt[T + 1]
            a = a.assign(key=a["side"] + "|" + a["gsis_id"]).set_index("key")
            b = b.assign(key=b["side"] + "|" + b["gsis_id"]).set_index("key")
            s1 = sn[sn["season"] == T].set_index("key")["snaps"]
            s2 = sn[sn["season"] == T + 1].set_index("key")["snaps"]
            keys = s1[s1 >= Y2Y_MIN_SNAPS].index.intersection(s2[s2 >= Y2Y_MIN_SNAPS].index)
            g = ctx.priors(T + 2)["group"].reindex([k[4:] for k in keys]).to_numpy(object)
            x = pd.DataFrame({"a": a["value"].reindex(keys).to_numpy(), "b": b["value"].reindex(keys).to_numpy(),
                              "pa": a["prior_value"].reindex(keys).to_numpy(),
                              "pb": b["prior_value"].reindex(keys).to_numpy(), "group": g})
            for grp, z in x.groupby("group"):
                if len(z) >= 10:
                    rows.append({"fit": label, "pair": f"{T}->{T + 1}", "group": grp, "n": len(z),
                                 "corr": float(np.corrcoef(z["a"], z["b"])[0, 1]),
                                 "corr_minus_prior": float(np.corrcoef(z["a"] - z["pa"], z["b"] - z["pb"])[0, 1])})
    return pd.DataFrame(rows)


def sanity(ctx: Ctx, cfg: rapm.RapmConfig, T: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ratings through T with SDs, plus T snaps, team and name; and how far each group moved from its prior."""
    fit = ratings_through(ctx, T, cfg, sd=True)
    r = fit.ratings()
    sn = pt.snaps(ctx.data.plays)
    sT = sn[sn["season"] == T].set_index(["side", "gsis_id"])["snaps"]
    r["snaps_T"] = sT.reindex(pd.MultiIndex.from_frame(r[["side", "gsis_id"]])).fillna(0).to_numpy()
    allsn = sn.groupby(["side", "gsis_id"])["snaps"].sum()
    r["snaps_all"] = allsn.reindex(pd.MultiIndex.from_frame(r[["side", "gsis_id"]])).fillna(0).to_numpy()
    names = ctx.inp.players.drop_duplicates("gsis_id").set_index("gsis_id")["display_name"]
    r["name"] = r["gsis_id"].map(names)
    pl = ctx.data.plays[ctx.data.plays["season"] == T]
    tm = pd.concat([pl[["off_team"] + pt.OFF_COLS].melt(id_vars="off_team").rename(columns={"off_team": "team"}),
                    pl[["def_team"] + pt.DEF_COLS].melt(id_vars="def_team").rename(columns={"def_team": "team"})])
    team = tm.groupby("value")["team"].agg(lambda s: s.value_counts().index[0])
    r["team"] = r["gsis_id"].map(team)
    r["side_ok"] = [pos.SIDE.get(g) == s if isinstance(g, str) else False for g, s in zip(r["group"], r["side"])]
    main = r[r["side_ok"] & r["active"]]
    mv = []
    for g, x in main[main["snaps_T"] >= SANITY_MIN_SNAPS].groupby("group"):
        mv.append({"group": g, "n": len(x), "mean_abs_move": float((x["value"] - x["prior_value"]).abs().mean()),
                   "sd_value": float(x["value"].std()), "sd_prior": float(x["prior_value"].std()),
                   "median_sd": float(x["sd"].median()),
                   "prior_sd": float(np.sqrt(fit.sigma2 / x["lam"].iloc[0])),
                   "sd_ratio": float(x["sd"].median() / np.sqrt(fit.sigma2 / x["lam"].iloc[0]))})
    return r, pd.DataFrame(mv)


def print_sanity(r: pd.DataFrame, mv: pd.DataFrame, T: int) -> None:
    main = r[r["side_ok"] & r["active"] & (r["snaps_T"] >= SANITY_MIN_SNAPS)]
    print(f"\nSanity: ratings fit through {T} (value = EPA per play on the field, + is good for his team), "
          f"players with >= {SANITY_MIN_SNAPS} snaps in {T}:")
    for g in rapm.GROUPS:
        x = main[main["group"] == g].sort_values("value", ascending=False)
        if x.empty:
            continue
        f = lambda z: "; ".join(f"{n} {t} {v:+.3f}±{s:.3f} ({int(sn_)})"  # noqa: E731
                                for n, t, v, s, sn_ in zip(z["name"], z["team"], z["value"], z["sd"], z["snaps_T"]))
        print(f"  {g} (n {len(x)}) top 10: {f(x.head(10))}")
        print(f"  {g} bottom 5: {f(x.tail(5).iloc[::-1])}")
    print("\nHow far ratings moved from their priors (players above), by group:")
    print(mv.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# --------------------------------------------------------------------------- game-model descriptive

def lineup_inputs(ctx: Ctx, fits: dict) -> tuple:
    """Participation-based on-field rows and expected shares, and RAPM values per (season, week)."""
    sched = ctx.inp.sched
    gs = pt.game_shares(ctx.data.plays)
    info = sched.drop_duplicates("game_id").set_index("game_id")
    ords = oa.week_ordinals(sched)
    gs["season"] = gs["game_id"].map(info["season"]).astype(int)
    gs["week"] = gs["game_id"].map(info["week"]).astype(int)
    gs["game_type"] = gs["game_id"].map(info["game_type"])
    gs["kick_ns"] = gs["game_id"].map(pd.Series(pt.te._utc_ns(info["kickoff"]), index=info.index))
    gs["ord"] = ords.reindex(pd.MultiIndex.from_frame(gs[["season", "week"]])).to_numpy()
    gs["group"] = None
    for S in sorted(set(gs["season"])):
        m = gs["season"] == S
        gs.loc[m, "group"] = ctx.priors(S)["group"].reindex(gs.loc[m, "gsis_id"]).to_numpy(object)
    gs["inv"] = 1.0
    onfield = gs[["game_id", "season", "week", "kick_ns", "ord", "team", "gsis_id", "group", "side", "share"]]
    onfield = onfield.sort_values("kick_ns", kind="stable").reset_index(drop=True)
    base = lu.attach_statuses(lu.chart_players(ctx.inp.depth), ctx.inp.injuries, ctx.inp.rosters,
                              ctx.inp.roster_seasons)
    base = base[base["season"] >= FIRST].reset_index(drop=True)
    pg0 = ctx.box_inputs(FIRST)[0]  # credits before 2016: the 2009-2015 estimation seasons
    probs = lu.estimate_status_probs(lu.attach_statuses(lu.chart_players(ctx.inp.depth), ctx.inp.injuries,
                                                        ctx.inp.rosters, ctx.inp.roster_seasons), pg0[pg0["inv"] > 0])
    lcfg = lu.LineupConfig()
    base["p_play"] = lu.play_probability(base, probs, lcfg)
    defaults = {S: lu.default_shares(gs, base, S) for S in sorted(set(base["season"]))}
    base["share_exp"] = lu.expected_shares(gs, base, sched, pv.TUNED.half_life, defaults)
    state = rf.RosterState(sched, gs, onfield, {lcfg: base}, probs, {})
    vals = []
    for S, fit in fits.items():
        r = fit.ratings()
        r = r[[pos.SIDE.get(g) == s if isinstance(g, str) else False for g, s in zip(r["group"], r["side"])]]
        v = r.drop_duplicates("gsis_id").set_index("gsis_id")["value"]
        for W in sorted(set(sched.loc[(sched["season"] == S) & (sched["game_type"] == "REG"), "week"])):
            vals.append(pd.DataFrame({"season": S, "week": int(W), "gsis_id": v.index, "v_rapm": v.to_numpy()}))
    return state, pd.concat(vals, ignore_index=True), lcfg


def normalize_expected(terms: pd.DataFrame, state, lcfg) -> pd.DataFrame:
    """Rescale the expected lineup value to the slots actually filled (non-QB: 10 on offense, 11 on defense).

    The chart lists up to three players per slot, so the expected shares
    (p_play x share_exp) of a team-week cover 6 to 15 defensive slots instead
    of 11. Left alone, that coverage noise enters exp - rem and correlates with
    team quality (it gave the coefficient the wrong sign on dev). The
    remembered lineup uses actual on-field shares, which already sum to 10 and 11.
    """
    lin = state.lineups[lcfg]
    x = lin[lin["group"] != "QB"].assign(ps=lambda d: d["p_play"] * d["share_exp"],
                                        sd=lambda d: np.where(d["group"].map(pos.SIDE) == "def", "def", "off"))
    cov = x.groupby(["season", "week", "team", "sd"])["ps"].sum().unstack(fill_value=0.0)
    t = terms.copy()
    key = pd.MultiIndex.from_frame(t[["season", "week", "team"]])
    for part, slots in (("off", 10.0), ("def", 11.0)):
        c = cov[part].reindex(key).to_numpy() if part in cov.columns else np.full(len(t), np.nan)
        scale = np.where(np.isfinite(c) & (c > 0), slots / np.where(c > 0, c, 1.0), 1.0)
        t[f"exp_{part}"] = t[f"exp_{part}"].to_numpy() * scale
        t[f"cov_{part}"] = c
    return t


def game_model(ctx: Ctx, cfg: rapm.RapmConfig, window: tuple[int, int], allow_holdout: bool) -> dict:
    """Descriptive: A4s + rapm_lineup_delta_diff vs A4s on REG games with moneylines in `window`."""
    games = elo_data.load_games()
    games = games[games["season"] <= window[1]].reset_index(drop=True)
    frame, _ = ml_m3.build_frame(games, ctx.inp.pbp, ctx.inp.sched, last_season=window[1])
    feats = ["elo_logit", "adj_epa_margin", "qb_delta_diff"]
    p_a4s, _ = logistic.walk_forward(frame, feats, (FIRST, window[1]), train_start=logistic.TRAIN_START)
    frame["p_a4s"] = p_a4s
    fits = {S: ctx.data.fit(S, ctx.priors(S), cfg) for S in range(FIRST, window[1] + 1)}
    state, vals, lcfg = lineup_inputs(ctx, fits)
    reg = ctx.inp.sched[(ctx.inp.sched["game_type"] == "REG") & ctx.inp.sched["season"].between(FIRST, window[1])
                        & ctx.inp.sched["home_score"].notna()]
    terms = normalize_expected(rf.team_lineup_terms(state, vals, "v_rapm", reg, lcfg, oa.TUNED), state, lcfg)
    f = rf.features_from_terms(terms, reg)
    frame = frame.merge(f[["lineup_delta_diff"]].rename(columns={"lineup_delta_diff": "rapm_lineup_delta_diff"}),
                        left_on="game_id", right_index=True, how="left")
    sub = frame[frame["season"] >= FIRST].copy()
    sub["rapm_lineup_delta_diff"] = sub["rapm_lineup_delta_diff"].fillna(0.0)
    lo = np.log(sub["p_a4s"] / (1 - sub["p_a4s"])).to_numpy()
    x = sub["rapm_lineup_delta_diff"].to_numpy()
    y = sub["y"].to_numpy(float)
    season = sub["season"].to_numpy(int)
    p_new = np.full(len(sub), np.nan)
    coefs = {}
    for S in range(window[0], window[1] + 1):
        tr, te_ = season < S, season == S

        def nll(b):
            p = 1 / (1 + np.exp(-(lo[tr] + b * x[tr])))
            p = np.clip(p, 1e-12, 1 - 1e-12)
            return -np.sum(y[tr] * np.log(p) + (1 - y[tr]) * np.log(1 - p))
        b = float(minimize_scalar(nll, bounds=(-200, 200), method="bounded").x)
        coefs[S] = b
        p_new[te_] = 1 / (1 + np.exp(-(lo[te_] + b * x[te_])))
    sub["p_new"] = p_new
    base = baselines.baseline_frame(games, window, allow_holdout=allow_holdout, require_market=True)
    ev = sub[sub["game_id"].isin(base["game_id"])]
    if len(ev) != len(base):
        raise SystemExit(f"game model: {len(ev)} of {len(base)} scored games have features")
    yv = ev["y"].to_numpy()
    ci = bootstrap.paired_bootstrap(yv, ev["p_new"].to_numpy(), ev["p_a4s"].to_numpy(), "brier", REPS, SEED)
    return {"n": int(len(ev)), "games_hash": registry.games_hash(ev["game_id"]), "game_ids": sorted(ev["game_id"]),
            "brier_a4s": float(np.mean((yv - ev["p_a4s"]) ** 2)), "brier_new": float(np.mean((yv - ev["p_new"]) ** 2)),
            "vs_a4s": ci, "coef_by_season": coefs, "feature_sd": float(ev["rapm_lineup_delta_diff"].std()),
            "corr_market_departure": float(np.corrcoef(
                ev["rapm_lineup_delta_diff"],
                base.set_index("game_id").loc[ev["game_id"], baselines.MARKET_COL].to_numpy() - ev["p_a4s"])[0, 1])}


# --------------------------------------------------------------------------- stages: dev / signoff

def fingerprint(last: int) -> dict:
    yrs = list(range(FIRST, last + 1))
    out = m5in.fingerprint(last)
    out.update({f"data/raw/ml/{k}": v for k, v in mldata.manifest_hashes(("participation",), yrs).items()})
    out["data/games.csv"] = registry.sha256_path(config.GAMES_CSV)
    return out


def run_window(window: tuple[int, int], allow_holdout: bool, cfg: rapm.RapmConfig, secondaries: bool = True
               ) -> tuple[dict, pd.DataFrame, Ctx]:
    windows.check(window, allow_holdout)
    timing = {}
    t0 = time.perf_counter()
    ctx = build_ctx(window[1])
    timing["load_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    res, frame = evaluate(ctx, range(window[0], window[1] + 1), cfg)
    timing["evaluate_s"] = time.perf_counter() - t0
    if secondaries:
        t0 = time.perf_counter()
        res["game_model"] = game_model(ctx, cfg, window, allow_holdout)
        timing["game_model_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        y2y = year_to_year(ctx, cfg, list(range(FIRST, window[1] + 1)))
        res["year_to_year"] = y2y.to_dict("records")
        timing["y2y_s"] = time.perf_counter() - t0
    res["timing"] = timing
    return res, frame, ctx


def print_secondaries(res: dict) -> None:
    gm = res["game_model"]
    print(f"\nGame model (descriptive): REG games with moneylines, n = {gm['n']} (hash {gm['games_hash']})")
    print(f"  A4s {gm['brier_a4s']:.4f}; A4s + rapm_lineup_delta_diff {gm['brier_new']:.4f}; "
          f"diff {fmt_ci(gm['vs_a4s'])}; coefficient by season "
          + ", ".join(f"{k} {v:+.2f}" for k, v in gm["coef_by_season"].items())
          + f"; feature SD {gm['feature_sd']:.4f}; corr with market minus A4s {gm['corr_market_departure']:+.2f}")
    y = pd.DataFrame(res["year_to_year"])
    if len(y):
        print("\nYear-to-year correlation of player values (both seasons >= "
              f"{Y2Y_MIN_SNAPS} snaps; n in parentheses):")
        for fit_label, z in y.groupby("fit"):
            t = z.pivot(index="group", columns="pair", values="corr").round(2)
            n = z.pivot(index="group", columns="pair", values="n")
            print(f"  {fit_label} fits:")
            for g in t.index:
                print(f"    {g:<3} " + "  ".join(f"{p} {t.at[g, p]:+.2f} ({int(n.at[g, p])})" for p in t.columns
                                            if not np.isnan(t.at[g, p])))
            t2 = z.pivot(index="group", columns="pair", values="corr_minus_prior").round(2)
            print("    rating minus prior: " + "; ".join(
                f"{g} " + "/".join(f"{t2.at[g, p]:+.2f}" for p in t2.columns if not np.isnan(t2.at[g, p]))
                for g in t2.index))


def stage_dev(log: bool) -> int:
    cfg = rapm.TUNED
    print(f"M5b dev evaluation, frozen settings: {cfg.to_dict()}")
    res, frame, ctx = run_window(DEV, False, cfg)
    print_eval(res, "Season-ahead play-level EPA MSE (dev)")
    print_secondaries(res)
    T = DEV[1]
    t0 = time.perf_counter()
    r, mv = sanity(ctx, cfg, T)
    res["timing"]["sanity_s"] = time.perf_counter() - t0
    print_sanity(r, mv, T)
    res["ol_movement"] = mv.to_dict("records")
    print("\nTiming: " + ", ".join(f"{k} {v:.0f}s" for k, v in res["timing"].items()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dev.json").write_text(json.dumps({**LICENSE, **res}, indent=1, default=float) + "\n")
    rr = r.copy()
    rr.attrs = {}
    rr.to_parquet(OUT / f"ratings_through_{T}.parquet", index=False)
    (OUT / "LICENSE.txt").write_text(pt.LICENSE_NOTE + "\nEvery file in this folder is CC BY-SA 4.0.\n")
    frame.to_parquet(OUT / "dev_predictions.parquet", index=False)
    if log:
        log_runs(res, frame, "m5bdev", DEV, cfg, ctx, prefix="m5b_dev")
    return 0


def log_runs(res: dict, frame: pd.DataFrame, label: str, window, cfg, ctx, prefix: str, signoff=None) -> None:
    fp = fingerprint(window[1])
    p = res["pooled"]
    gids = frame["game_id"].unique()
    common = {"window": {"seasons": list(window), "game_type": "REG+POST"}, "license": LICENSE,
              "plays_hash": res["plays_hash"]}
    if signoff:
        common["signoff"] = signoff
    notes = {"rapm": "Bayesian RAPM, season-ahead (ratings through S-1).",
             "team": "Baseline (a): M3 opponent-adjust team ratings as of S week 1 + situation terms.",
             "box": "Baseline (b): M5 box values summed over the 22 on-field players, refit on the training seasons.",
             "intercept": "Reference: training-play mean EPA."}
    for m in MODELS:
        extra = dict(common)
        extra["comparisons"] = {k: v for k, v in p["vs"].items()} if m == "rapm" else None
        extra["margin"] = {"mse": p["margin"]["mse"][m], "n_games": p["margin"]["n_games"]}
        if m == "rapm":
            extra["result"] = {k: v for k, v in res.items() if k not in ("year_to_year",)}
        registry.log_run(f"{prefix}_{m}", label=label, seasons=window, game_ids=gids,
                         metrics={"n": p["n_plays"], "mse": p["mse"][m], "margin_mse": p["margin"]["mse"][m]},
                         features=[], params={"rapm": cfg.to_dict(), "ratings": oa.TUNED.to_dict(),
                                              "box_values": pv.TUNED.to_dict()},
                         data=fp, notes=notes[m] + " CC BY-SA 4.0 (participation-derived).", extra=extra)
    if "game_model" in res:
        gm = res["game_model"]
        registry.log_run(f"{prefix}_game_model", label=label, seasons=window, game_ids=gm["game_ids"],
                         metrics={"n": gm["n"], "brier": gm["brier_new"], "brier_a4s": gm["brier_a4s"]},
                         features=["elo_logit", "adj_epa_margin", "qb_delta_diff", "rapm_lineup_delta_diff"],
                         params={"rapm": cfg.to_dict(), "offset": "A4s walk-forward logit, trained 2001..S-1"},
                         data=fp, notes="Descriptive only: A4s logit + b * rapm_lineup_delta_diff (b fit on REG "
                         f"{FIRST}..S-1). CC BY-SA 4.0.",
                         extra={**common, "window": {"seasons": list(window), "game_type": "REG"},
                                "result": {k: v for k, v in gm.items() if k != "game_ids"}})
    print(f"\nRuns written to {registry.RUNS_DIR.relative_to(config.ROOT)}/ (label {label})")


def holdout_signoff_runs() -> list[str]:
    hits = []
    for path in glob.glob(str(registry.RUNS_DIR / "*_m5b_signoff_*.json")):
        if json.loads(Path(path).read_text()).get("holdout"):
            hits.append(Path(path).name)
    return sorted(hits)


def _max_diff(a, b) -> float:
    if isinstance(a, dict) and isinstance(b, dict):
        keys = (set(a) & set(b)) - {"timing", "license", "note"}
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


def pass_rules(res: dict) -> dict:
    p = res["pooled"]["vs"]
    m = res["pooled"]["margin"]["vs"]
    return {"primary_vs_team_ci_high": p["team"]["ci_high"], "primary_vs_box_ci_high": p["box"]["ci_high"],
            "primary_pass": bool(p["team"]["ci_high"] < 0 and p["box"]["ci_high"] < 0),
            "secondary_margin_vs_team_pass": bool(m["team"]["ci_high"] < 0),
            "secondary_margin_vs_box_pass": bool(m["box"]["ci_high"] < 0),
            "descriptive_game_model_diff": res["game_model"]["vs_a4s"]["diff"],
            "descriptive_game_model_ci": [res["game_model"]["vs_a4s"]["ci_low"], res["game_model"]["vs_a4s"]["ci_high"]]}


def stage_signoff(window: tuple[int, int], allow_holdout: bool, log: bool) -> int:
    """The pre-registered sign-off (context/ml.md, "M5b holdout pre-registration"). With the dev window and no
    logging it is the dry run, which must reproduce `dev.json` exactly."""
    label = "holdout" if windows.touches_holdout(window) else "m5bdev"
    if label == "holdout":
        prior = holdout_signoff_runs()
        if prior:
            raise SystemExit(f"an M5b holdout sign-off is already recorded ({prior}); it is run once only")
    cfg = rapm.TUNED
    print(f"== M5b sign-off ({label}), seasons {window[0]}-{window[1]} season-ahead, frozen {cfg.to_dict()} ==")
    res, frame, ctx = run_window(window, allow_holdout, cfg)
    print_eval(res, f"Season-ahead play-level EPA MSE ({label})")
    print_secondaries(res)
    rules = pass_rules(res)
    print("\nPre-registered rules:")
    for k, v in rules.items():
        print(f"  {k:<36} {v}")
    print(f"M5b PRIMARY: {'PASS' if rules['primary_pass'] else 'FAIL'} (RAPM beats both baselines, "
          "game-cluster CI upper bounds < 0).")
    result = {**LICENSE, **res, "rules": rules, "label": label}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"signoff_{label}.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    if label != "holdout":
        ref = OUT / "dev.json"
        if not ref.exists():
            print("\nDry run: run `ml_m5b.py dev` first to compare against it.")
            return 1
        old = json.loads(ref.read_text())
        keys = ("pooled", "per_season", "game_model", "year_to_year", "games_hash", "plays_hash", "fits")
        d = max(_max_diff(json.loads(json.dumps(res[k], default=float)), old[k]) for k in keys)
        print(f"\nDry-run reproduction of dev.json: max |diff| {d:.1e} -> {'REPRODUCED' if d == 0 else 'NOT REPRODUCED'}")
        return 0 if d == 0 else 1
    if log:
        log_runs(res, frame, label, window, cfg, ctx, prefix="m5b_signoff", signoff="m5b_signoff")
    return 0


# --------------------------------------------------------------------------- stage: leakcheck

def stage_leakcheck(S: int = DEV[1], seed: int = 0) -> int:
    """Real-data check: corrupt all of season S (play-by-play, participation, depth charts, injuries, rosters),
    rebuild every input from raw, refit, and predict the CLEAN season-S plays. Must be identical; the positive
    control (training on S too) must change."""
    windows.check((S, S))
    cfg = rapm.TUNED
    t0 = time.perf_counter()
    clean_parts = {T: mldata.load_participation([T]) for T in range(FIRST, S + 1)}
    clean_ctx = build_ctx(S, part_override=clean_parts)
    sched = clean_ctx.inp.sched
    as_of = sched.loc[sched["season"] == S, "as_of"].min()
    p2, s2 = asof_mod.corrupt_from(clean_ctx.inp.pbp, sched, as_of, seed)
    rng = np.random.default_rng(seed)
    late = (p2["season"] >= S).to_numpy()
    for c in ("down", "ydstogo", "yardline_100"):
        p2.loc[late, c] = rng.permutation(p2.loc[late, c].to_numpy())
    parts = dict(clean_parts)
    q = parts[S].copy()
    for c in ("players_on_play", "offense_players", "defense_players"):
        q[c] = rng.permutation(q[c].to_numpy(object))
    parts[S] = q
    tables = rf.corrupt_aux(clean_ctx.inp.tables(), S - 1, 99, seed)
    dirty_ctx = build_ctx(S, pbp_override=p2, sched_override=s2, tables_override=tables, part_override=parts)
    target = clean_ctx.data.plays[clean_ctx.data.plays["season"] == S]
    preds = {}
    for name, c in (("clean", clean_ctx), ("dirty", dirty_ctx)):
        preds[name] = predict_on_clean(c, clean_ctx, S, cfg)
        preds[name]["rapm"] = c.data.fit(S, c.priors(S), cfg).predict(target)
    out = {}
    for m in MODELS:
        d = np.abs(preds["clean"][m] - preds["dirty"][m])
        out[m] = {"max_abs_diff": float(d.max()), "changed": int((d > 1e-12).sum()), "leak_free": bool((d <= 1e-12).all())}
    # positive control: train on season S as well (dirty vs clean must then differ)
    ctrl_c = rapm.season_ahead(clean_ctx.data.plays, S, clean_ctx.priors(S), cfg, leaky=True).predict(target)
    ctrl_d = rapm.season_ahead(dirty_ctx.data.plays, S, dirty_ctx.priors(S), cfg, leaky=True).predict(target)
    dc = np.abs(ctrl_c - ctrl_d)
    out["positive_control"] = {"max_abs_diff": float(dc.max()), "changed": int((dc > 1e-12).sum()),
                               "leak_free": bool((dc <= 1e-12).all())}
    pr_diff = (clean_ctx.priors(S)[["v_box"]].join(dirty_ctx.priors(S)[["v_box"]], rsuffix="_d", how="outer")
               .fillna(0.0))
    out["box_priors_max_abs_diff"] = float((pr_diff["v_box"] - pr_diff["v_box_d"]).abs().max())
    common = clean_ctx.priors(S).index.intersection(dirty_ctx.priors(S).index)
    gc = clean_ctx.priors(S)["group"].reindex(common).fillna("")
    out["groups_changed"] = int((gc != dirty_ctx.priors(S)["group"].reindex(common).fillna("")).sum())
    print(f"Leakage check, season {S} ({len(target):,} clean plays; everything from {S} corrupted):")
    for k, v in out.items():
        print(f"  {k:<18} {v}")
    ok = all(out[m]["leak_free"] for m in MODELS) and not out["positive_control"]["leak_free"]
    print(f"Result: {'LEAK-FREE (control fails as it must)' if ok else 'PROBLEM'}; {time.perf_counter() - t0:.0f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "leakcheck.json").write_text(json.dumps({**LICENSE, "season": S, **out}, indent=1, default=float) + "\n")
    return 0 if ok else 1


def predict_on_clean(fit_ctx: Ctx, clean_ctx: Ctx, S: int, cfg) -> dict:
    """Baseline predictions for the CLEAN season-S plays with every fitted piece taken from `fit_ctx`."""
    d = fit_ctx.data
    train = np.flatnonzero(d.season < S)
    w = train_weights(d.season[train], S, cfg.decay)
    keep = w > 0
    train, w = train[keep], w[keep]
    fc = rapm.fixed_columns(cfg.situation)
    y = d.y[train]
    test_plays = clean_ctx.data.plays[clean_ctx.data.plays["season"] == S]
    Zte = rapm.fixed_matrix(test_plays)[:, fc]
    out = {}
    fw = first_week(fit_ctx.inp.sched, S)
    key_of = lambda seasons: [(int(t) + 1, first_week(fit_ctx.inp.sched, int(t) + 1)) for t in seasons]  # noqa: E731
    g = _wls(d.Z[train][:, fc], y - team_offsets(fit_ctx, train, key_of), w)
    rat = fit_ctx.team_ratings({(S, fw)})[(S, fw)].set_index("team")
    off = (float(rat["mu"].iloc[0]) + rat["off"].reindex(test_plays["off_team"]).fillna(0.0).to_numpy()
           + rat["def"].reindex(test_plays["def_team"]).fillna(0.0).to_numpy()
           + float(rat["h"].iloc[0]) * test_plays["home"].to_numpy(float))
    out["team"] = off + Zte @ g
    btr = box_sums(fit_ctx, train, fit_ctx.priors)
    gb = _wls(np.column_stack([btr, d.Z[train][:, fc]]), y, w)
    pr = fit_ctx.priors(S)
    side_ok = pr["group"].map(lambda x: pos.SIDE.get(x) if isinstance(x, str) else None)
    v_off, v_def = pr["v_box"].where(side_ok == "off", 0.0), pr["v_box"].where(side_ok == "def", 0.0)
    o, dd = test_plays[pt.OFF_COLS].to_numpy(object), test_plays[pt.DEF_COLS].to_numpy(object)
    bte = np.column_stack([v_off.reindex(o.ravel()).fillna(0.0).to_numpy().reshape(o.shape).sum(axis=1),
                           v_def.reindex(dd.ravel()).fillna(0.0).to_numpy().reshape(dd.shape).sum(axis=1)])
    out["box"] = np.column_stack([bte, Zte]) @ gb
    out["intercept"] = np.full(len(test_plays), float(np.sum(w * y) / np.sum(w)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["clean", "tune", "dev", "leakcheck", "signoff-dry-run", "signoff"])
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    if a.stage == "clean":
        return stage_clean()
    if a.stage == "tune":
        return stage_tune(not a.no_log)
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
