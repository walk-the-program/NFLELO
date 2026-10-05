"""Simulation inputs "as of now", from the same live feature code the ledger uses.

`state_at` builds the A4s features of every unplayed REG game with
`live_features.build` (so the means match the ledger's spreads), the Elo state,
and each team's opponent-adjusted rating (offense minus defense) at the week
the future games are predicted with. It is used by the weekly run and, through
`as_of_view`, by the backtests: `as_of_view` rolls a past season back to a
moment in time (later scores removed, starters listed only for the next week,
as nflverse does), so a backtest sees what the live job would have seen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import asof as asof_mod
from .. import live_features
from ..features import opponent_adjust as oa
from ..models import margin as mm
from ...teams import franchise
from . import season as sim

SCORE_COLS = ("home_score", "away_score", "result", "total")
QB_COLS = ("home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name")


def as_of_view(sched: pd.DataFrame, now) -> pd.DataFrame:
    """The schedule as it looked at `now`: scores of games not yet kicked off removed, and starting QBs
    listed only for the next week (the week whose as_of is the first after `now`), as nflverse does."""
    s = sched if "as_of" in sched.columns else asof_mod.add_asof(sched)
    s = s.copy()
    later = (s["kickoff"] >= pd.Timestamp(now)).to_numpy()
    for c in SCORE_COLS:
        if c in s.columns:
            s.loc[later, c] = np.nan
    season = int(s.loc[later, "season"].min()) if later.any() else None
    if season is not None:
        nxt = live_features.next_week(s[s["season"] == season], now)
        beyond = later & ((s["season"] > season) | ((s["season"] == season) & (s["week"] > (nxt or 0))))
        for c in QB_COLS:
            if c in s.columns:
                s[c] = s[c].astype(object)
                s.loc[beyond, c] = None
    return s


def team_adj(tab: pd.DataFrame, sched: pd.DataFrame, season: int, week: int) -> dict:
    """Offense minus defense (EPA per play) per franchise, from the ratings fit as of (season, week)."""
    r = oa.compute_ratings(tab, sched, [(int(season), int(week))], oa.TUNED, ("all",))
    r = r[r["kind"] == "all"]
    return {t: float(o - d) for t, o, d in zip(r["team"], r["off"], r["def"])}


def state_at(pbp: pd.DataFrame, sched: pd.DataFrame, elo_games: pd.DataFrame, season: int, now,
             fit: mm.MarginFit, shape: str, kn: mm.KeyNumbers | None,
             tab: pd.DataFrame | None = None) -> tuple[sim.SimInputs, pd.DataFrame, dict]:
    """(SimInputs, live features of the unplayed games, team adj ratings) for `season` at `now`.

    `sched` has every season the features need (1999 on), with results only for
    games before `now`; `elo_games` are the `nflelo.data` games played before `now`.
    """
    if "as_of" not in sched.columns:
        sched = asof_mod.add_asof(sched)
    reg = sched[(sched["season"] == season) & (sched["game_type"] == "REG")]
    todo = reg[reg["home_score"].isna()]
    if todo.empty:
        raise ValueError(f"no unplayed {season} REG games at {now}")
    elo = live_features.elo_state(elo_games, season)
    feats = live_features.build(pbp, sched, todo, now, elo)
    tab = oa.rating_table(pbp, sched, oa.TUNED) if tab is None else tab
    adj = team_adj(tab, sched, season, int(feats["feature_week"].max()))
    games = reg.copy()
    inp = sim.build_inputs(games, feats, elo, adj, fit, shape, kn)
    return inp, feats, adj


def check_consistency(inp: sim.SimInputs, feats: pd.DataFrame, adj: dict, fit: mm.MarginFit) -> float:
    """Largest gap between a future game's adj_epa_margin feature and adj_home - adj_away (should be ~0)."""
    wk = feats["feature_week"].max()
    f = feats[feats["feature_week"] == wk]
    s = int(inp.season)
    gap = [abs(r.adj_epa_margin - (adj[franchise(r.home_team, s)] - adj[franchise(r.away_team, s)]))
           for r in f.itertuples()]
    return float(max(gap)) if gap else 0.0
