"""Load every table M5 needs for a season range (cached nflverse files only; see nflelo.ml.data).

Never loads participation: that is validation-only and lives in players/validate.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .. import asof as asof_mod
from .. import data as mldata
from . import credit as cr

FIRST_SEASON = 1999


@dataclass
class M5Inputs:
    sched: pd.DataFrame        # with kickoff and as_of
    pbp: pd.DataFrame          # credit.PBP_COLUMNS (market columns dropped)
    depth: pd.DataFrame        # normalized depth charts (2005+, 2004 dropped)
    injuries: pd.DataFrame
    rosters: pd.DataFrame
    players: pd.DataFrame
    positions: pd.DataFrame    # credit.position_table
    availability: pd.DataFrame  # credit.component_availability (era gaps), fixed per season
    roster_seasons: list        # lineup.roster_status_seasons: seasons with week-accurate roster statuses

    def tables(self) -> dict:
        """The weekly tables, as `features.roster.roster_leakage_check` expects them."""
        return {"depth": self.depth, "injuries": self.injuries, "rosters": self.rosters}


def load(last_season: int, first_season: int = FIRST_SEASON, extra_pbp_columns: list[str] | None = None) -> M5Inputs:
    """Seasons first_season..last_season of every M5 input. The caller decides the window; nothing is scored here."""
    years = range(first_season, last_season + 1)
    sched = asof_mod.add_asof(mldata.load_schedules(years))
    cols = list(dict.fromkeys(cr.PBP_COLUMNS + list(extra_pbp_columns or [])))
    pbp = mldata.load_pbp(years, columns=cols)
    depth = mldata.load_depth_charts(years, sched)
    rosters = mldata.load_rosters_weekly(years)
    injuries = mldata.load_injuries(years)
    players = mldata.load_players()
    positions = cr.position_table(depth, rosters, players)
    avail = cr.component_availability(pbp)
    from . import lineup as lu
    roster_ok = lu.roster_status_seasons(rosters, pbp)
    return M5Inputs(sched, pbp, depth, injuries, rosters, players, positions, avail, roster_ok)


def data_seasons(last_season: int, first_season: int = FIRST_SEASON) -> dict:
    """{dataset: [seasons]} actually read by `load`, for registry fingerprints."""
    yrs = list(range(first_season, last_season + 1))
    return {"pbp": yrs, "schedules": yrs, "depth_charts": [y for y in yrs if y >= 2005 and y != 2004],
            "rosters_weekly": [y for y in yrs if y >= 2002], "injuries": [y for y in yrs if y >= 2009],
            "players": [mldata.SNAPSHOT_SEASON]}


def fingerprint(last_season: int, first_season: int = FIRST_SEASON) -> dict:
    out = {}
    for ds, yrs in data_seasons(last_season, first_season).items():
        out.update({f"data/raw/ml/{k}": v for k, v in mldata.manifest_hashes((ds,), yrs).items()})
    return out
