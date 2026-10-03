"""Game context from the schedule, and the Elo feature.

Context (context/ml-m3-method.md, section 3):

    rest_diff   home minus away days of rest (schedule `home_rest`, `away_rest`),
                capped at +/-7 so a bye week doesn't count as huge; missing = 0
    neutral     1 for a neutral-site game (schedule `location` == "Neutral")

Both are fixed when the schedule is published, long before the week starts.

Elo (feature 1):

    elo_logit   Elo v2's log-odds for the home team:
                (home_pre - away_pre + HFA) * ln(10) / 400, with HFA = 0 at a
                neutral site.

Elo v2's ratings are pre-game and change only when a team plays, so a team's
rating at kickoff equals its rating at the week's `as_of`. The online
home-field edge is league-wide and moves after every game, including that
week's Thursday game, so here it is frozen at its value before the week's
first game (spec section 7). That is the only difference from Elo's own
`p_home`.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ... import config
from ...config import EloConfig
from ...data import nflverse_to_games
from ...elo import run_elo
from .. import asof as asof_mod

REST_CAP = 7.0
CONTEXT_FEATURES = ["rest_diff", "neutral"]
ELO_FEATURE = "elo_logit"
LN10_400 = math.log(10.0) / 400.0


def build_features(pbp: pd.DataFrame, sched: pd.DataFrame, games: pd.DataFrame | None = None) -> pd.DataFrame:
    """rest_diff and neutral for each game in `games` (default: all of `sched`), indexed by game_id."""
    games = sched if games is None else games
    hr = pd.to_numeric(games["home_rest"], errors="coerce") if "home_rest" in games else pd.Series(np.nan, index=games.index)
    ar = pd.to_numeric(games["away_rest"], errors="coerce") if "away_rest" in games else pd.Series(np.nan, index=games.index)
    rest = np.clip((hr - ar).fillna(0.0).to_numpy(float), -REST_CAP, REST_CAP)
    loc = games["location"] if "location" in games.columns else pd.Series("Home", index=games.index)
    out = pd.DataFrame({"rest_diff": rest, "neutral": (loc == "Neutral").to_numpy(float)},
                       index=games["game_id"].to_numpy())
    out.index.name = "game_id"
    return out


# --------------------------------------------------------------------------- Elo

def hfa_before(elo_out: pd.DataFrame, cfg: EloConfig) -> np.ndarray:
    """The online HFA in force before each game of a `run_elo` output (same order).

    Rebuilt from the update rule in `nflelo.elo.run_elo`; for non-neutral games it
    equals the output's `hfa_used` (a test checks this).
    """
    if cfg.hfa_mode != "online":
        return np.full(len(elo_out), float(cfg.hfa))
    hs, as_ = elo_out["home_score"].to_numpy(float), elo_out["away_score"].to_numpy(float)
    actual = np.where(hs > as_, 1.0, np.where(hs == as_, 0.5, 0.0))
    step = cfg.k_hfa * (actual - elo_out["p_home"].to_numpy(float))
    if cfg.hfa_mov:
        step = step * elo_out["mov"].to_numpy(float)
    live = elo_out["updated"].to_numpy(bool) & ~elo_out["neutral"].to_numpy(bool)
    step = np.where(live, step, 0.0)
    return cfg.hfa_init + np.concatenate([[0.0], np.cumsum(step)[:-1]])


def elo_features_from_games(games: pd.DataFrame, cfg: EloConfig = config.DEFAULT_CONFIG) -> pd.DataFrame:
    """elo_logit for every game in an `nflelo.data` games frame (1970+), indexed by game_id.

    The week's HFA is the HFA before the first game of that (season, week) in
    Elo's own (chronological) order.
    """
    out, _ = run_elo(games, cfg)
    hb = hfa_before(out, cfg)
    first = pd.Series(np.arange(len(out))).groupby([out["season"].to_numpy(), out["week"].to_numpy()]).transform("min")
    hfa_week = hb[first.to_numpy()]
    neutral = out["neutral"].to_numpy(bool)
    d = out["home_pre"].to_numpy(float) - out["away_pre"].to_numpy(float) + np.where(neutral, 0.0, hfa_week)
    res = pd.DataFrame({ELO_FEATURE: d * LN10_400, "hfa_week": hfa_week}, index=out["game_id"].to_numpy())
    res.index.name = "game_id"
    return res


def build_elo_features(pbp: pd.DataFrame, sched: pd.DataFrame, games: pd.DataFrame | None = None,
                       history: pd.DataFrame | None = None, cfg: EloConfig = config.DEFAULT_CONFIG) -> pd.DataFrame:
    """elo_logit for each game in `games`, in the (pbp, sched, games) builder form used by the leakage check.

    `sched` is an nflverse schedule with scores; `history` (optional) holds
    earlier games in `nflelo.data` form to warm the ratings up. `pbp` is unused.
    """
    if "kickoff" not in sched.columns:
        sched = asof_mod.add_asof(sched)
    games = sched if games is None else games
    s = sched.sort_values("kickoff", kind="stable")
    g = nflverse_to_games(s)
    # nflverse_to_games sorts by (date, game_id); keep true kickoff order within a day
    g = g.set_index("game_id").loc[[i for i in s["game_id"] if i in set(g["game_id"])]].reset_index()
    if history is not None:
        g = pd.concat([history, g], ignore_index=True)
    feats = elo_features_from_games(g, cfg)
    return feats.loc[games["game_id"].to_numpy(), [ELO_FEATURE]]
