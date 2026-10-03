"""Opponent-adjusted EPA ratings: one weighted ridge regression per (season, week).

Model (context/ml-m3-method.md, section 4). Every scrimmage play is one equation

    EPA(play) = mu + O[offense] + D[defense] + h * (offense is home) + noise

fit by weighted ridge regression, once per (season, week), on every play that
kicked off before that week's `as_of` (the week's earliest kickoff, see
`nflelo.ml.asof`). The penalty `lam * (sum O^2 + sum D^2)` pulls every rating
toward 0, the league average; `mu` and `h` are not penalized. `lam` is in
units of full-weight plays: it equals sigma^2 / tau^2, play noise over the
true spread of team ratings (the Bayesian-prior reading of ridge).

Weights. Each play gets

    w = 0.5 ** (weeks_ago / half_life) * (rho if the play is from last season else 1)

`weeks_ago` counts game weeks on a timeline that skips the off-season (see
`week_ordinals`); the most recent week before `as_of` has weeks_ago = 0. Only
this season and last season enter a fit, so in week 1 the ratings are last
season's, shrunk by the decay, `rho`, and the prior.

Speed. Plays are first summed to one row per (game, offense). Because every
play in that row has the same design row and the same weight, the weighted
normal equations of the play-level problem are exactly

    X' diag(w * n) X beta = X' (w * sum_epa)

with X built from the aggregated rows (a test checks the equivalence). The
design is sparse (one 1 per offense, defense, intercept, and home flag).

Plays used are the M1 ones (`team_efficiency.clean_plays`): real pass and run
snaps with an EPA, no two-point tries, garbage time (win probability outside
0.05 to 0.95) removed, REG and POST games both counted, betting-market columns
dropped first (decision D3). Teams are franchise IDs.

Features (one row per game, indexed by game_id):

    adj_epa_margin  = (O[home] + D[away]) - (O[away] + D[home])     # all plays
    adj_pass_margin, adj_rush_margin                                  # split fits (ladder step A3)

A team with no rating yet (an expansion team before its first game) counts as
league average (0).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from ...teams import franchise
from .. import asof as asof_mod
from ..data import ML_DIR
from . import team_efficiency as te

# kind -> (play-count column, EPA-sum column) in the team-game table
KINDS = {"all": ("n_all", "epa_all"), "pass": ("n_pass", "epa_pass"), "rush": ("n_rush", "epa_rush")}
FEATURES = {"all": "adj_epa_margin", "pass": "adj_pass_margin", "rush": "adj_rush_margin"}
CACHE_DIR = ML_DIR / "ratings"
CACHE_VERSION = 1  # bump when the fitting code changes, so stale caches are ignored


@dataclass(frozen=True)
class RatingConfig:
    lam: float = 400.0        # ridge penalty, in full-weight plays (sigma^2 / tau^2)
    half_life: float = 8.0    # weeks: a play this many game weeks older counts half
    rho: float = 0.5          # extra off-season discount on last season's plays
    garbage_low: float = 0.05
    garbage_high: float = 0.95

    def efficiency(self) -> te.EfficiencyConfig:
        return te.EfficiencyConfig(garbage_low=self.garbage_low, garbage_high=self.garbage_high)

    def to_dict(self) -> dict:
        return asdict(self)


# Chosen by scripts/ml_tune_ratings.py on 2000-2005 only (next-week EPA-margin target,
# decision M3-D2). See context/ml.md, "M3 build notes", for the grid and the run file.
TUNED = RatingConfig(lam=100.0, half_life=48.0, rho=0.25)


# --------------------------------------------------------------------------- timeline

def week_ordinals(sched: pd.DataFrame) -> pd.Series:
    """(season, week) -> position on a timeline of game weeks that skips the off-season.

    Weeks are ranked within each season (REG and POST together), and seasons are
    laid end to end, so last season's final week sits one step before this
    season's week 1.
    """
    wk = sched[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).copy()
    wk["rank"] = wk.groupby("season").cumcount() + 1
    per = wk.groupby("season")["rank"].max()
    offset = per.cumsum().shift(fill_value=0)
    wk["ord"] = wk["season"].map(offset).to_numpy() + wk["rank"].to_numpy()
    return wk.set_index(["season", "week"])["ord"].astype(int)


def week_asof(sched: pd.DataFrame) -> pd.Series:
    """(season, week) -> the week's as_of (earliest kickoff), as UTC nanoseconds."""
    if "as_of" not in sched.columns:
        sched = asof_mod.add_asof(sched)
    a = sched.groupby(["season", "week"])["as_of"].min()
    return pd.Series(te._utc_ns(a), index=a.index)


def decay_weights(ord_now: int, ord_rows: np.ndarray, season_now: int, season_rows: np.ndarray,
                  half_life: float, rho: float) -> np.ndarray:
    """0.5 ** (weeks_ago / half_life) * rho ** (seasons ago). Most recent week: weeks_ago = 0."""
    weeks_ago = np.maximum(ord_now - ord_rows - 1, 0)
    return 0.5 ** (weeks_ago / half_life) * rho ** np.maximum(season_now - season_rows, 0)


# --------------------------------------------------------------------------- team-game table

def rating_table(pbp: pd.DataFrame, sched: pd.DataFrame, cfg: RatingConfig = RatingConfig()) -> pd.DataFrame:
    """One row per (game, offense) with play counts and EPA sums, plus the home flag and timeline position."""
    if "as_of" not in sched.columns:
        sched = asof_mod.add_asof(sched)
    tab = te.team_game_table(te.clean_plays(pbp, cfg.efficiency()), sched)
    info = sched.drop_duplicates("game_id").set_index("game_id")
    home_fr = pd.Series([franchise(c, int(s)) for c, s in zip(info["home_team"], info["season"])], index=info.index)
    neutral = (info["location"] == "Neutral") if "location" in info.columns else pd.Series(False, index=info.index)
    tab["home"] = ((tab["off"] == tab["game_id"].map(home_fr)) & ~tab["game_id"].map(neutral).fillna(False)
                   .astype(bool)).astype(float)
    tab["week"] = tab["game_id"].map(info["week"]).astype(int)
    ords = week_ordinals(sched)
    tab["ord"] = ords.reindex(pd.MultiIndex.from_frame(tab[["season", "week"]])).to_numpy()
    tab["kick_ns"] = te._utc_ns(tab["kickoff"])
    return tab.sort_values("kick_ns", kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- the ridge fit

def fit_ridge(off: np.ndarray, deff: np.ndarray, home: np.ndarray, n: np.ndarray, s: np.ndarray,
              w: np.ndarray, n_teams: int, lam: float) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Weighted ridge on aggregated rows. Returns (mu, h, O, D).

    off, deff: team codes 0..n_teams-1; home: 0/1; n: plays in the row; s: EPA
    sum of the row; w: per-play weight of the row. Equivalent to the play-level
    fit where each of the row's n plays has weight w.
    """
    m = len(n)
    p = 2 + 2 * n_teams
    r = np.arange(m)
    rows = np.concatenate([r, r, r, r])
    cols = np.concatenate([np.zeros(m, int), np.ones(m, int), 2 + off, 2 + n_teams + deff])
    vals = np.concatenate([np.ones(m), home.astype(float), np.ones(m), np.ones(m)])
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(m, p))
    W = w * n
    A = (X.T @ X.multiply(W[:, None])).toarray()
    b = X.T @ (w * s)
    pen = np.full(p, float(lam))
    pen[0] = 0.0
    pen[1] = 1e-9 * max(W.sum(), 1.0)  # keeps h identifiable when no row is a home offense
    A[np.diag_indices(p)] += pen
    beta = np.linalg.solve(A, b)
    return float(beta[0]), float(beta[1]), beta[2:2 + n_teams], beta[2 + n_teams:]


def compute_ratings(tab: pd.DataFrame, sched: pd.DataFrame, keys: Iterable[tuple[int, int]],
                    cfg: RatingConfig = RatingConfig(), kinds: Iterable[str] = ("all",)) -> pd.DataFrame:
    """Ratings for each (season, week) key, fit only on rows that kicked off before the week's as_of.

    Returns long rows: kind, season, week, team, off, def, mu, h, w_off, w_def
    (w_* are the weighted play counts behind each rating).
    """
    kinds = tuple(kinds)
    asofs = week_asof(sched)
    ords = week_ordinals(sched)
    kick = tab["kick_ns"].to_numpy()
    seas = tab["season"].to_numpy()
    tord = tab["ord"].to_numpy()
    off_all, def_all = tab["off"].to_numpy(), tab["def"].to_numpy()
    home_all = tab["home"].to_numpy()
    cols = {k: (tab[KINDS[k][0]].to_numpy(float), tab[KINDS[k][1]].to_numpy(float)) for k in kinds}
    out = []
    for S, W in sorted(set((int(a), int(b)) for a, b in keys)):
        end = np.searchsorted(kick, asofs[(S, W)], side="left")  # tab is sorted by kickoff
        sl = slice(0, end)
        m = (seas[sl] >= S - 1) & (seas[sl] <= S)
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        teams, inv = np.unique(np.concatenate([off_all[idx], def_all[idx]]), return_inverse=True)
        o_code, d_code = inv[:len(idx)], inv[len(idx):]
        w = decay_weights(int(ords[(S, W)]), tord[idx], S, seas[idx], cfg.half_life, cfg.rho)
        for k in kinds:
            n, s = cols[k][0][idx], cols[k][1][idx]
            keep = n > 0
            mu, h, O, D = fit_ridge(o_code[keep], d_code[keep], home_all[idx][keep], n[keep], s[keep],
                                    w[keep], len(teams), cfg.lam)
            wn = w[keep] * n[keep]
            w_off = np.bincount(o_code[keep], weights=wn, minlength=len(teams))
            w_def = np.bincount(d_code[keep], weights=wn, minlength=len(teams))
            out.append(pd.DataFrame({"kind": k, "season": S, "week": W, "team": teams, "off": O, "def": D,
                                     "mu": mu, "h": h, "w_off": w_off, "w_def": w_def}))
    if not out:
        return pd.DataFrame(columns=["kind", "season", "week", "team", "off", "def", "mu", "h", "w_off", "w_def"])
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- caching

def _cache_path(cfg: RatingConfig, kinds: tuple[str, ...], keys: list[tuple[int, int]], data_key: str,
                cache_dir: Path) -> Path:
    blob = json.dumps({"v": CACHE_VERSION, "cfg": cfg.to_dict(), "kinds": kinds, "keys": keys,
                       "data": data_key}, sort_keys=True)
    h = hashlib.sha256(blob.encode()).hexdigest()[:16]
    return cache_dir / f"ratings_{'-'.join(kinds)}_{h}.parquet"


def cached_ratings(tab: pd.DataFrame, sched: pd.DataFrame, keys: Iterable[tuple[int, int]],
                   cfg: RatingConfig, kinds: Iterable[str], data_key: str,
                   cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """`compute_ratings`, cached on disk under `data/raw/ml/ratings/` (gitignored).

    The cache key covers the config, kinds, the (season, week) keys, the code
    version, and `data_key` (pass a hash of the input files, e.g. from the
    manifest), so changed data or settings never reuse a stale file.
    """
    kinds = tuple(kinds)
    keys = sorted(set((int(a), int(b)) for a, b in keys))
    path = _cache_path(cfg, kinds, keys, data_key, cache_dir)
    if path.exists():
        return pd.read_parquet(path)
    r = compute_ratings(tab, sched, keys, cfg, kinds)
    cache_dir.mkdir(parents=True, exist_ok=True)
    r.to_parquet(path, index=False)
    return r


# --------------------------------------------------------------------------- features

def _lookup(ratings: pd.DataFrame, kind: str, season: np.ndarray, week: np.ndarray, team: np.ndarray,
            col: str) -> np.ndarray:
    r = ratings[ratings["kind"] == kind].set_index(["season", "week", "team"])[col]
    idx = pd.MultiIndex.from_arrays([season, week, team])
    return r.reindex(idx).fillna(0.0).to_numpy()  # unrated team = league average


def matchup_margin(ratings: pd.DataFrame, games: pd.DataFrame, kind: str = "all") -> np.ndarray:
    """(O[home] + D[away]) - (O[away] + D[home]) for each game, from the ratings of the game's (season, week)."""
    s, w = games["season"].to_numpy(int), games["week"].to_numpy(int)
    h = np.array([franchise(c, int(y)) for c, y in zip(games["home_team"], s)], dtype=object)
    a = np.array([franchise(c, int(y)) for c, y in zip(games["away_team"], s)], dtype=object)
    oh, dh = _lookup(ratings, kind, s, w, h, "off"), _lookup(ratings, kind, s, w, h, "def")
    oa, da = _lookup(ratings, kind, s, w, a, "off"), _lookup(ratings, kind, s, w, a, "def")
    return (oh + da) - (oa + dh)


def feature_names(split: bool = False) -> list[str]:
    return [FEATURES["pass"], FEATURES["rush"]] if split else [FEATURES["all"]]


def build_features(pbp: pd.DataFrame, sched: pd.DataFrame, games: pd.DataFrame | None = None,
                   cfg: RatingConfig = TUNED, split: bool = False,
                   ratings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Opponent-adjusted matchup margins for each game in `games` (default: all of `sched`), indexed by game_id.

    With `split`, returns the pass and rush margins (separate fits) instead of
    the combined one. Pass precomputed `ratings` (from `compute_ratings` or
    `cached_ratings` on the same data) to skip the fits; otherwise only the
    (season, week) keys the games need are fit, from `pbp` and `sched` alone.
    """
    if "as_of" not in sched.columns:
        sched = asof_mod.add_asof(sched)
    games = sched if games is None else games
    kinds = ("pass", "rush") if split else ("all",)
    if ratings is None:
        keys = list(zip(games["season"].astype(int), games["week"].astype(int)))
        ratings = compute_ratings(rating_table(pbp, sched, cfg), sched, keys, cfg, kinds)
    out = pd.DataFrame({FEATURES[k]: matchup_margin(ratings, games, k) for k in kinds},
                       index=games["game_id"].to_numpy())
    out.index.name = "game_id"
    return out


# --------------------------------------------------------------------------- tuning target (M3-D2)

def epa_margin_targets(tab: pd.DataFrame, sched: pd.DataFrame, seasons: tuple[int, int],
                       game_types: tuple[str, ...] = ("REG",)) -> pd.DataFrame:
    """Per game: home offense EPA/play minus away offense EPA/play (the M3-D2 tuning target).

    Uses the same cleaned plays as the ratings. Returns game_id, season, week,
    home, away, neutral, margin.
    """
    g = sched[sched["season"].between(*seasons) & sched["game_type"].isin(game_types)].copy()
    g["home"] = [franchise(c, int(s)) for c, s in zip(g["home_team"], g["season"])]
    g["away"] = [franchise(c, int(s)) for c, s in zip(g["away_team"], g["season"])]
    g["neutral"] = (g["location"] == "Neutral").astype(float) if "location" in g.columns else 0.0
    rate = tab.assign(r=tab["epa_all"] / tab["n_all"]).set_index(["game_id", "off"])["r"]
    hr = rate.reindex(pd.MultiIndex.from_arrays([g["game_id"], g["home"]])).to_numpy()
    ar = rate.reindex(pd.MultiIndex.from_arrays([g["game_id"], g["away"]])).to_numpy()
    g["margin"] = hr - ar
    g = g[np.isfinite(g["margin"])]
    return g[["game_id", "season", "week", "home", "away", "neutral", "margin"]].reset_index(drop=True)


def score_ratings(ratings: pd.DataFrame, targets: pd.DataFrame) -> dict:
    """Mean squared error of predicted EPA margins (matchup margin + h off neutral sites) on `targets`."""
    r = ratings[ratings["kind"] == "all"]
    s, w = targets["season"].to_numpy(int), targets["week"].to_numpy(int)
    get = lambda team, col: _lookup(r, "all", s, w, targets[team].to_numpy(object), col)  # noqa: E731
    hk = r.drop_duplicates(["season", "week"]).set_index(["season", "week"])["h"]
    h = hk.reindex(pd.MultiIndex.from_arrays([s, w])).fillna(0.0).to_numpy()
    pred = (get("home", "off") + get("away", "def")) - (get("away", "off") + get("home", "def"))
    pred = pred + h * (1 - targets["neutral"].to_numpy(float))
    y = targets["margin"].to_numpy(float)
    mse = float(np.mean((y - pred) ** 2))
    return {"n": int(len(y)), "mse": mse, "mse_zero": float(np.mean(y ** 2)),
            "r2": 1 - mse / float(np.mean((y - y.mean()) ** 2)), "corr": float(np.corrcoef(y, pred)[0, 1])}
