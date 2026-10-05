"""Position groups, depth-chart slot names, and team-code aliases used across M5.

Groups (context/ml-m5-method.md, section 3):

    offense  QB, RB (FB included), WR, TE, OL
    defense  DL, LB, CB, S
    kicking  K (P is tracked but carries no value)

`group_of(code)` maps any position code nflverse uses (rosters, players table,
2001-2024 depth-chart `depth_position`, 2025+ ESPN `pos_abb`) to a group, or
None. A generic "DB" maps to CB; callers that know better (a depth-chart slot
such as SS) should prefer that.
"""
from __future__ import annotations

import pandas as pd

from ...teams import franchise

OFFENSE = ("QB", "RB", "WR", "TE", "OL")
DEFENSE = ("DL", "LB", "CB", "S")
GROUPS = OFFENSE + DEFENSE + ("K",)
SIDE = {**{g: "off" for g in OFFENSE}, **{g: "def" for g in DEFENSE}, "K": "off"}

_CODES = {
    "QB": ["QB"],
    "RB": ["RB", "HB", "H-B", "TB", "FB", "RB/FB"],
    "WR": ["WR", "LWR", "RWR", "FL", "SE", "SWR", "X", "Z", "SL", "WR/KR", "KR/WR"],
    "TE": ["TE", "TE/HB", "TE/FB", "Y"],
    "OL": ["LT", "LG", "C", "RG", "RT", "T", "G", "OT", "OG", "LOT", "ROT", "OC", "OL", "LOG", "ROG"],
    "DL": ["DE", "DT", "NT", "LDE", "RDE", "LDT", "RDT", "DL", "LE", "RE", "NG", "DT/DE", "EDGE", "LNT",
           "RNT", "LEO", "UT", "END", "N", "DE/LB"],
    "LB": ["OLB", "MLB", "ILB", "WLB", "SLB", "LB", "WILL", "SAM", "MIKE", "LOLB", "ROLB", "LILB", "RILB",
           "JACK", "LLB", "RLB", "BUCK", "MAC", "RUSH", "WOLB", "SOLB", "BLB", "JLB", "MILB", "MOLB", "$LB",
           "WILB", "WIL", "OTTO", "MO", "J", "LB/DE"],
    "CB": ["LCB", "RCB", "CB", "NB", "NCB", "DB", "RCB2", "LCB2", "NKL", "NICK"],
    "S": ["SS", "FS", "S", "SAF", "DS", "WS"],
    "K": ["K", "PK"],
}
CODE_TO_GROUP = {c: g for g, codes in _CODES.items() for c in codes}
# On the depth chart, a defensive "LT"/"RT" is a tackle on the line (old 3-4 charts), not an offensive tackle.
DEFENSE_OVERRIDES = {"LT": "DL", "RT": "DL", "T": "DL", "DT": "DL"}
# A fullback is in the RB group for credit and values, but his slot does not count as an RB slot.
FULLBACK_CODES = frozenset({"FB", "RB/FB"})

# Base-formation slots per group when a team-week has no usable depth chart (before 2005).
DEFAULT_SLOTS = {"QB": 1, "RB": 1, "WR": 3, "TE": 1, "OL": 5, "DL": 4, "LB": 3, "CB": 2, "S": 2, "K": 1}

# Weekly-roster team codes that nflelo.teams does not know.
TEAM_ALIASES = {"ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "SL": "LA"}


def group_of(code, side: str | None = None) -> str | None:
    """Position group of a code (case-insensitive), or None. `side` "def" applies the defensive overrides."""
    if code is None or (not isinstance(code, str) and pd.isna(code)):
        return None
    c = str(code).strip().upper()
    if side == "def" and c in DEFENSE_OVERRIDES:
        return DEFENSE_OVERRIDES[c]
    return CODE_TO_GROUP.get(c)


def team_id(code, season: int) -> str | None:
    """Franchise ID for any nflverse team code (including the weekly-roster aliases), or None if unknown."""
    if code is None or (not isinstance(code, str) and pd.isna(code)) or not str(code).strip():
        return None
    c = str(code).strip().upper()
    c = TEAM_ALIASES.get(c, c)
    try:
        return franchise(c, int(season))
    except ValueError:
        return None


def map_teams(codes: pd.Series, seasons: pd.Series) -> pd.Series:
    """Vectorized `team_id` over (code, season) pairs (memoized on distinct pairs)."""
    pairs = pd.DataFrame({"c": codes.astype("string"), "s": seasons.astype(int)})
    uniq = pairs.drop_duplicates()
    lut = {(c, s): team_id(c, s) for c, s in zip(uniq["c"], uniq["s"]) if not pd.isna(c)}
    return pd.Series([lut.get((c, s)) if not pd.isna(c) else None for c, s in zip(pairs["c"], pairs["s"])],
                     index=codes.index, dtype=object)
