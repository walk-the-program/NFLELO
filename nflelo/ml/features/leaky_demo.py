"""A deliberately LEAKY feature, for tests and the notebook. Never use it in a model.

It reports each team's offense and defense EPA per play averaged over the
WHOLE season, the classic mistake: for a week-3 game the average quietly
includes weeks 3 to 18, among them the very game being predicted. The leakage
test must catch this builder; if it ever passes, the test is broken.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ... import teams
from .. import asof as asof_mod
from . import team_efficiency as te

LEAKY_COLUMNS = [f"{side}_{k}_epa_all" for side in te.SIDES for k in ("off", "def")]


def build_leaky_season_average(pbp: pd.DataFrame, sched: pd.DataFrame, games: pd.DataFrame | None = None,
                               cfg: te.EfficiencyConfig = te.EfficiencyConfig()) -> pd.DataFrame:
    if "as_of" not in sched.columns:
        sched = asof_mod.add_asof(sched)
    games = sched if games is None else games
    tab = te.team_game_table(te.clean_plays(pbp, cfg), sched)
    out = {}
    for k in ("off", "def"):
        season = tab.groupby(["season", k])[["epa_all", "n_all"]].sum()
        avg = (season["epa_all"] / season["n_all"]).to_dict()
        for side in te.SIDES:
            out[f"{side}_{k}_epa_all"] = np.array(
                [avg.get((int(s), teams.franchise(c, int(s))), np.nan)
                 for c, s in zip(games[f"{side}_team"], games["season"])])
    res = pd.DataFrame(out, index=games["game_id"].to_numpy())
    res.index.name = "game_id"
    return res[LEAKY_COLUMNS]
