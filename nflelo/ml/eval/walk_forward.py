"""Walk-forward splits: train only on the past, the way the model is used.

- `season_splits`: to predict season S, train on seasons before S.
- `week_splits`: within a season, week W is predicted with everything that
  kicked off before W's `as_of` (the earliest kickoff of week W; see
  `nflelo.ml.asof`). That includes earlier weeks of the same season and all
  prior seasons.
- `walk_forward_predict`: run a fit/predict function over season splits and
  collect out-of-sample predictions.

Random K-fold cross-validation is wrong here: it trains on games from the
future of the games it scores.
"""
from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import pandas as pd


def season_splits(df: pd.DataFrame, test_seasons: tuple[int, int], min_train_season: int | None = None
                  ) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (season, train_mask, test_mask) for each season S in the window.

    train = rows with season < S (and >= min_train_season if given); test = rows with season == S.
    """
    season = df["season"].to_numpy()
    for s in range(int(test_seasons[0]), int(test_seasons[1]) + 1):
        train = season < s
        if min_train_season is not None:
            train &= season >= int(min_train_season)
        test = season == s
        if test.any():
            yield s, train, test


def week_splits(df: pd.DataFrame, season: int) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (week, train_mask, test_mask) for each week of `season`.

    `df` needs season, week, kickoff, and as_of. train = rows that kicked off
    before the week's as_of; test = rows of that week.
    """
    for col in ("kickoff", "as_of"):
        if col not in df.columns:
            raise ValueError(f"week_splits needs a {col!r} column (see nflelo.ml.asof.add_asof)")
    in_season = (df["season"] == int(season)).to_numpy()
    kick = df["kickoff"]
    for w in sorted(df.loc[in_season, "week"].unique()):
        test = in_season & (df["week"] == w).to_numpy()
        as_of = df.loc[test, "as_of"].min()
        train = (kick < as_of).to_numpy()
        yield int(w), train, test


def walk_forward_predict(df: pd.DataFrame, test_seasons: tuple[int, int],
                         fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
                         min_train_season: int | None = None) -> pd.Series:
    """Out-of-sample predictions for each test season, trained on seasons before it.

    `fit_predict(train_df, test_df)` returns one prediction per test row.
    Returns a Series aligned to `df.index` (NaN outside the test window).
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, train, test in season_splits(df, test_seasons, min_train_season):
        out.loc[test] = np.asarray(fit_predict(df[train], df[test]), dtype=float)
    return out
