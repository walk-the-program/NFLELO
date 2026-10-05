"""Evaluation windows (decision D2).

- DEV, 2006-2019: tune and compare here as often as you like.
- HOLDOUT, 2020-2025: scored only at milestone sign-off. Selecting any holdout
  season raises unless the caller passes `allow_holdout=True`, and the
  registry flags every such run.
- REPRODUCTION, 2010-2025: the Elo v2 test window from `nflelo.config`, used
  only to prove the harness reproduces the known Elo and market numbers. It
  overlaps the holdout, so it needs `allow_holdout=True` too.
- TUNE, 2000-2005: where the M3 rating knobs are tuned (decision M3-D2), on
  next-week EPA rather than game outcomes. Runs labelled "tune" carry a tuning
  loss, not a Brier. (M5 tunes player knobs on 2001-2008 under the same label.)
- M5_DEV, 2012-2019: the M5 evaluation window (decision M5-D3), a sub-window
  of DEV. Lineup features need injury reports (2009+) and a few training
  seasons, so M5 is scored here, against A4s on the same games. Guarded like
  DEV: it can never include a holdout season.
- M5B_DEV, 2018-2019: the M5b development window (context/ml-m5b-method.md,
  section 4): season-ahead play predictions for 2018 (ratings from 2016-2017)
  and 2019 (from 2016-2018). Participation starts in 2016, so these are the
  only DEV seasons with training data behind them. Guarded like DEV.
- M6_DEV, 2018-2019: the M6 play-model development window
  (context/ml-m6-method.md, section 6), the same season-ahead seasons as
  M5B_DEV. Guarded like DEV.
"""
from __future__ import annotations

import pandas as pd

from ... import config

DEV = (2006, 2019)
HOLDOUT = (2020, 2025)
REPRODUCTION = tuple(config.TEST_SEASONS)
TUNE = (2000, 2005)
M5_DEV = (2012, 2019)
M5B_DEV = (2018, 2019)
M6_DEV = (2018, 2019)

LABELS = {"dev": DEV, "holdout": HOLDOUT, "reproduction": REPRODUCTION, "tune": TUNE, "m5dev": M5_DEV, "m5bdev": M5B_DEV,
          "m6dev": M6_DEV}
# Labels whose runs may never touch the holdout.
NON_HOLDOUT_LABELS = frozenset({"dev", "tune", "m5dev", "m5bdev", "m6dev"})


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
