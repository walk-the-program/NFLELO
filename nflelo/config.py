"""Paths, evaluation windows, and Elo configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = ROOT / "outputs"
LEGACY_XLSX = ROOT / "legacy_2025" / "NFLELO_data.xlsx"
GAMES_CSV = DATA_DIR / "games.csv"

# Seasons 1970-1998 come from the legacy spreadsheet, 1999+ from nflverse.
LEGACY_LAST_SEASON = 1998
NFLVERSE_FIRST_SEASON = 1999

# Held-out protocol. Ratings warm up from 1970; tuning is scored on
# TUNE_SEASONS, the final report on TEST_SEASONS (REG games only).
TUNE_SEASONS = (1980, 2009)
TEST_SEASONS = (2010, 2025)


@dataclass(frozen=True)
class EloConfig:
    start: float = 1500.0          # initial rating and regression target
    k: float = 20.0
    lam: float = 0.15              # season regression toward `start`
    hfa_mode: str = "fixed"        # "fixed" or "online"
    hfa: float = 55.0              # used when hfa_mode == "fixed"
    hfa_init: float = 65.0         # 1970 value when hfa_mode == "online"
    k_hfa: float = 1.0             # online HFA learning rate
    hfa_mov: bool = False          # scale the online HFA update by the MOV multiplier
    mov_mode: str = "winner"       # "winner": 538 (winner minus loser); "abs": legacy
    mov_cap: float | None = None   # cap on the MOV multiplier (legacy used 2.0)
    expansion_start: float = 1300.0  # rating of franchises that first appear after the first season
    include_playoffs: bool = True  # update ratings on playoff games

    def with_(self, **kw) -> "EloConfig":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)


# Legacy 2025 model shifted from a 1000 to a 1500 scale. Ratings only enter
# the model as differences, so the shift changes nothing.
LEGACY_CONFIG = EloConfig(
    start=1500.0, k=20.0, lam=0.15, hfa_mode="fixed", hfa=55.0,
    mov_mode="abs", mov_cap=2.0, expansion_start=1500.0, include_playoffs=False,
)

# Best online-HFA config from scripts/tune.py on the 1980-2009 tuning window.
# It ties the overall winner (fixed HFA 65) within 0.0001 Brier there; online
# HFA was preferred because a fixed 65 implies a ~59% home win rate, while
# the 2020s rate is ~53.6%. Decision recorded in CONTEXT.md, 2026-10-02.
DEFAULT_CONFIG = EloConfig(
    k=20.0, lam=0.40, hfa_mode="online", hfa_init=65.0, k_hfa=0.5,
    mov_cap=None, expansion_start=1300.0, include_playoffs=False,
)
