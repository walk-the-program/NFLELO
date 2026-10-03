"""One canonical franchise ID per franchise across all eras.

IDs are the current nflverse abbreviations. Raw codes come from three places:
FiveThirtyEight's game file (the 1970-1998 source; see `franchise_538`), the
legacy spreadsheet (Pro-Football-Reference style, e.g. GNB, RAM, SDG; now used
only for validation), and nflverse (e.g. GB, STL, SD). Codes that are
ambiguous across eras are resolved by season.

Decisions
- Relocations keep one history: Raiders (OAK/RAI/LAR-in-LA/LVR) -> LV,
  Chargers (SDG/SD) -> LAC, Rams (RAM/STL 1995+/LAR/LA) -> LA, Oilers (HOU/OTI
  before 1999) -> TEN, Baltimore Colts (BAL before 1996) -> IND, Cardinals
  (STL before 1995, PHO) -> ARI, Boston Patriots (BOS) -> NE.
- Browns: CLE is one continuous franchise across the 1996-1998 hiatus. This
  follows the legacy script and the NFL's own record-keeping.
- Ravens: BAL from 1996 is a new franchise (an expansion-style entry that
  starts at the expansion rating), not a continuation of the Colts.
- Texans: HOU from 1999 on is a new franchise (first games in 2002).
"""
from __future__ import annotations

import pandas as pd

FRANCHISES = frozenset(
    "ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LV LAC LA "
    "MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS".split()
)

# Codes whose meaning does not depend on the season.
_STATIC = {
    "ARI": "ARI", "PHO": "ARI", "PHX": "ARI",
    "ATL": "ATL", "BUF": "BUF", "CAR": "CAR", "CHI": "CHI", "CIN": "CIN",
    "CLE": "CLE", "DAL": "DAL", "DEN": "DEN", "DET": "DET",
    "GB": "GB", "GNB": "GB",
    "IND": "IND", "CLT": "IND",
    "JAX": "JAX", "JAC": "JAX",
    "KC": "KC", "KAN": "KC",
    "LV": "LV", "LVR": "LV", "OAK": "LV", "RAI": "LV",
    "LAC": "LAC", "SD": "LAC", "SDG": "LAC",
    "LA": "LA", "LAR": "LA", "RAM": "LA",
    "MIA": "MIA", "MIN": "MIN",
    "NE": "NE", "NWE": "NE", "BOS": "NE",
    "NO": "NO", "NOR": "NO",
    "NYG": "NYG", "NYJ": "NYJ", "PHI": "PHI", "PIT": "PIT", "SEA": "SEA",
    "SF": "SF", "SFO": "SF",
    "TB": "TB", "TAM": "TB",
    "TEN": "TEN", "OTI": "TEN",
    "WAS": "WAS", "WSH": "WAS",
}


def franchise(code: str, season: int) -> str:
    """Map a raw team code to its canonical franchise ID for a given season."""
    if code is None or (not isinstance(code, str) and pd.isna(code)):
        raise ValueError("missing team code")
    code = str(code).strip().upper()

    if code == "STL":
        # St. Louis Cardinals through 1994; St. Louis Rams 1995-2015.
        return "ARI" if season < 1995 else "LA"
    if code == "HOU":
        # Houston Oilers (pre-1999) became the Titans; the Texans began in 2002.
        return "TEN" if season < 1999 else "HOU"
    if code == "BAL":
        # Baltimore Colts left after 1983; the Ravens are a separate franchise from 1996.
        return "IND" if season < 1996 else "BAL"
    if code in _STATIC:
        return _STATIC[code]
    raise ValueError(f"unknown team code {code!r} (season {season})")


# FiveThirtyEight's file already uses one code per franchise across all eras
# (checked on the 1970-2022 file): Oilers and Titans are TEN, Colts IND, Ravens
# BAL (1996 on), Texans HOU (2002 on), Cardinals ARI, Rams LAR, Raiders OAK,
# Chargers LAC, Washington WSH. The Browns (CLE) are absent in 1996-1998.
_CODES_538 = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BUF": "BUF", "CAR": "CAR",
    "CHI": "CHI", "CIN": "CIN", "CLE": "CLE", "DAL": "DAL", "DEN": "DEN",
    "DET": "DET", "GB": "GB", "HOU": "HOU", "IND": "IND", "JAX": "JAX",
    "KC": "KC", "LAC": "LAC", "LAR": "LA", "MIA": "MIA", "MIN": "MIN",
    "NE": "NE", "NO": "NO", "NYG": "NYG", "NYJ": "NYJ", "OAK": "LV",
    "PHI": "PHI", "PIT": "PIT", "SEA": "SEA", "SF": "SF", "TB": "TB",
    "TEN": "TEN", "WSH": "WAS",
}
# First season each franchise-consistent code is valid. A code that shows up
# earlier means the file uses a different convention, which we want to know about.
_FIRST_SEASON_538 = {"BAL": 1996, "HOU": 2002, "CAR": 1995, "JAX": 1995, "SEA": 1976, "TB": 1976}


def franchise_538(code: str, season: int) -> str:
    """Map a FiveThirtyEight team code to a canonical franchise ID.

    The mapping does not depend on era, but it is strict: unknown codes raise,
    and a code used before its franchise existed (BAL before 1996, HOU before
    2002, ...) raises instead of being quietly re-mapped, because that would
    mean the file's convention differs from what we verified.
    """
    code = str(code).strip().upper()
    if code not in _CODES_538:
        raise ValueError(f"unknown FiveThirtyEight team code {code!r} (season {season})")
    if season < _FIRST_SEASON_538.get(code, 0):
        raise ValueError(f"FiveThirtyEight code {code!r} before its franchise existed (season {season})")
    return _CODES_538[code]
