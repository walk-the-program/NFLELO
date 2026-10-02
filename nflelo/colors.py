"""Team colors keyed by canonical franchise ID (see teams.py).

Taken from the legacy script's `NFLConfig.COLORS['team_colors']` palette and
remapped to the current nflverse codes (KAN -> KC, GNB -> GB, LVR -> LV,
LAR -> LA, NWE -> NE, NOR -> NO, SFO -> SF, TAM -> TB). No logos.
"""
from __future__ import annotations

TEAM_COLORS: dict[str, str] = {
    "ARI": "#97233f", "ATL": "#a71930", "BAL": "#241773", "BUF": "#00338d",
    "CAR": "#0085ca", "CHI": "#0b162a", "CIN": "#fb4f14", "CLE": "#311d00",
    "DAL": "#003594", "DEN": "#fb4f14", "DET": "#0076b6", "GB": "#203731",
    "HOU": "#03202f", "IND": "#002c5f", "JAX": "#006778", "KC": "#e31837",
    "LV": "#000000", "LAC": "#0080c6", "LA": "#003594", "MIA": "#008e97",
    "MIN": "#4f2683", "NE": "#002244", "NO": "#d3bc8d", "NYG": "#0b2265",
    "NYJ": "#125740", "PHI": "#004c54", "PIT": "#ffb612", "SF": "#aa0000",
    "SEA": "#002244", "TB": "#d50a0a", "TEN": "#0c2340", "WAS": "#5a1414",
}

FALLBACK = "#6b7280"


def team_color(code: str) -> str:
    """Hex color for a franchise ID (neutral grey if unknown)."""
    return TEAM_COLORS.get(code, FALLBACK)
