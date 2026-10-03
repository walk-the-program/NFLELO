"""Evaluation windows (decision D2).

- DEV, 2006-2019: tune and compare here as often as you like.
- HOLDOUT, 2020-2025: scored only at milestone sign-off. Selecting any holdout
  season raises unless the caller passes `allow_holdout=True`, and the
  registry flags every such run.
- REPRODUCTION, 2010-2025: the Elo v2 test window from `nflelo.config`, used
  only to prove the harness reproduces the known Elo and market numbers. It
  overlaps the holdout, so it needs `allow_holdout=True` too.
"""
from __future__ import annotations

import pandas as pd

from ... import config

DEV = (2006, 2019)
HOLDOUT = (2020, 2025)
REPRODUCTION = tuple(config.TEST_SEASONS)

LABELS = {"dev": DEV, "holdout": HOLDOUT, "reproduction": REPRODUCTION}


class HoldoutError(RuntimeError):
    """Raised when holdout seasons are scored without `allow_holdout=True`."""


def touches_holdout(seasons: tuple[int, int]) -> bool:
    lo, hi = int(seasons[0]), int(seasons[1])
    return lo <= HOLDOUT[1] and hi >= HOLDOUT[0]


def check(seasons: tuple[int, int], allow_holdout: bool = False) -> bool:
    """Validate a season window. Returns True if it touches the holdout (allowed only when asked)."""
    lo, hi = int(seasons[0]), int(seasons[1])
    if lo > hi:
        raise ValueError(f"empty window {seasons}")
    hit = touches_holdout((lo, hi))
    if hit and not allow_holdout:
        raise HoldoutError(f"window {lo}-{hi} overlaps the locked holdout {HOLDOUT[0]}-{HOLDOUT[1]}; "
                           "pass allow_holdout=True only for a milestone sign-off or a reproduction run")
    return hit


def select(df: pd.DataFrame, seasons: tuple[int, int], allow_holdout: bool = False,
           game_types: tuple[str, ...] = ("REG",)) -> pd.DataFrame:
    """Rows of `df` in the inclusive season window with the given game types (REG by default)."""
    check(seasons, allow_holdout)
    m = df["season"].between(int(seasons[0]), int(seasons[1]))
    if game_types:
        m &= df["game_type"].isin(game_types)
    return df[m]
