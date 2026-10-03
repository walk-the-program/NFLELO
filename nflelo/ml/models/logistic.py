"""Walk-forward logistic regression for home win probability (context/ml-m3-method.md, sections 2 and 6).

    log(p / (1 - p)) = b0 + b1 * x1 + ... + bk * xk

Protocol: to predict season S, fit on REG games from `train_start` (2001)
through S - 1, then predict every game of S. Refit once per season; the
features themselves are already as of each game's week.

- Ties (y = 0.5) enter as two half-weight rows, one win and one loss, which
  matches how the harness scores them.
- Training rows are weighted by season, 0.5 ** ((S - 1 - season) / half_life)
  with half_life = 8 seasons, so the intercept can follow the falling
  home-field edge.
- No penalty: with four or five features and thousands of games, the plain
  maximum-likelihood fit is the readable one.

`walk_forward` takes any `fitter(X, y, w) -> model with predict_proba`, so the
same protocol runs the boosted-tree comparison (ladder step A6).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

TRAIN_START = 2001
SEASON_HALF_LIFE = 8.0


def expand_ties(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows with y = 0.5 become a win and a loss, each with half the weight."""
    X, y, w = np.asarray(X, float), np.asarray(y, float), np.asarray(w, float)
    tie = np.isclose(y, 0.5)
    if not tie.any():
        return X, y, w
    Xt, wt = X[tie], w[tie] * 0.5
    return (np.vstack([X[~tie], Xt, Xt]), np.concatenate([y[~tie], np.ones(len(Xt)), np.zeros(len(Xt))]),
            np.concatenate([w[~tie], wt, wt]))


def season_weights(seasons, target_season: int, half_life: float = SEASON_HALF_LIFE) -> np.ndarray:
    """Weight 1 for the season just before the target, halving every `half_life` seasons further back."""
    s = np.asarray(seasons, float)
    return 0.5 ** ((target_season - 1 - s) / half_life)


def fit_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> LogisticRegression:
    """Unpenalized weighted logistic regression; y may contain ties (0.5)."""
    X2, y2, w2 = expand_ties(X, y, w)
    m = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=5000, tol=1e-10)  # C=inf: no penalty
    m.fit(X2, y2.astype(int), sample_weight=w2)
    return m


def walk_forward(frame: pd.DataFrame, features: list[str], test_seasons: tuple[int, int],
                 train_start: int = TRAIN_START, half_life: float = SEASON_HALF_LIFE,
                 fitter: Callable = fit_logistic, y_col: str = "y") -> tuple[pd.Series, pd.DataFrame]:
    """Out-of-sample home-win probabilities for each season in `test_seasons`.

    `frame` holds REG games with season, `y_col`, and the feature columns (no
    NaN). Returns (predictions aligned to frame.index, NaN outside the test
    seasons; one row per test season with the fitted intercept and
    coefficients when the model has them, plus the training size).
    """
    X_all = frame[features].to_numpy(float)
    if np.isnan(X_all).any():
        bad = [f for f in features if frame[f].isna().any()]
        raise ValueError(f"NaN in features {bad}")
    y_all = frame[y_col].to_numpy(float)
    season = frame["season"].to_numpy(int)
    pred = pd.Series(np.nan, index=frame.index, dtype=float)
    rows = []
    for S in range(int(test_seasons[0]), int(test_seasons[1]) + 1):
        train = (season >= train_start) & (season < S)
        test = season == S
        if not test.any():
            continue
        if not train.any():
            raise ValueError(f"no training games before season {S}")
        model = fitter(X_all[train], y_all[train], season_weights(season[train], S, half_life))
        pred[test] = model.predict_proba(X_all[test])[:, 1]
        rec = {"season": S, "n_train": int(train.sum())}
        if hasattr(model, "coef_"):
            rec["intercept"] = float(model.intercept_[0])
            rec.update({f: float(c) for f, c in zip(features, model.coef_[0])})
        rows.append(rec)
    return pred, pd.DataFrame(rows)
