"""Display metadata per current franchise: full name, conference, division.

Keyed by the canonical franchise IDs in teams.py. This is the 2026 alignment;
it is applied to the whole history (franchises are tracked as one line even
across the 2002 realignment). Names are plain facts; no logos.
"""
from __future__ import annotations

# franchise ID -> (full name, conference, division)
TEAM_META: dict[str, tuple[str, str, str]] = {
    "BUF": ("Buffalo Bills", "AFC", "East"),
    "MIA": ("Miami Dolphins", "AFC", "East"),
    "NE": ("New England Patriots", "AFC", "East"),
    "NYJ": ("New York Jets", "AFC", "East"),
    "BAL": ("Baltimore Ravens", "AFC", "North"),
    "CIN": ("Cincinnati Bengals", "AFC", "North"),
    "CLE": ("Cleveland Browns", "AFC", "North"),
    "PIT": ("Pittsburgh Steelers", "AFC", "North"),
    "HOU": ("Houston Texans", "AFC", "South"),
    "IND": ("Indianapolis Colts", "AFC", "South"),
    "JAX": ("Jacksonville Jaguars", "AFC", "South"),
    "TEN": ("Tennessee Titans", "AFC", "South"),
    "DEN": ("Denver Broncos", "AFC", "West"),
    "KC": ("Kansas City Chiefs", "AFC", "West"),
    "LV": ("Las Vegas Raiders", "AFC", "West"),
    "LAC": ("Los Angeles Chargers", "AFC", "West"),
    "DAL": ("Dallas Cowboys", "NFC", "East"),
    "NYG": ("New York Giants", "NFC", "East"),
    "PHI": ("Philadelphia Eagles", "NFC", "East"),
    "WAS": ("Washington Commanders", "NFC", "East"),
    "CHI": ("Chicago Bears", "NFC", "North"),
    "DET": ("Detroit Lions", "NFC", "North"),
    "GB": ("Green Bay Packers", "NFC", "North"),
    "MIN": ("Minnesota Vikings", "NFC", "North"),
    "ATL": ("Atlanta Falcons", "NFC", "South"),
    "CAR": ("Carolina Panthers", "NFC", "South"),
    "NO": ("New Orleans Saints", "NFC", "South"),
    "TB": ("Tampa Bay Buccaneers", "NFC", "South"),
    "ARI": ("Arizona Cardinals", "NFC", "West"),
    "LA": ("Los Angeles Rams", "NFC", "West"),
    "SF": ("San Francisco 49ers", "NFC", "West"),
    "SEA": ("Seattle Seahawks", "NFC", "West"),
}

# Display order: AFC then NFC, East/North/South/West, alphabetical by ID within a division.
TEAM_ORDER: list[str] = list(TEAM_META)


def full_name(team: str) -> str:
    return TEAM_META[team][0]


def conference(team: str) -> str:
    return TEAM_META[team][1]


def division(team: str) -> str:
    """e.g. 'AFC East'."""
    return f"{TEAM_META[team][1]} {TEAM_META[team][2]}"
