"""Stacking on the log-odds scale (context/ml-m8-method.md, component 2).

    logit P = c + sum_j w_j * logit(p_j),     w_j >= 0

where p_j are base models' out-of-sample predictions. For season S the weights
are fit on the base models' walk-forward predictions of seasons
`first_season`..S-1 only (each of those was itself made from earlier seasons),
with A4s's season-decay weights, by minimizing

    weighted mean log loss + lam * sum_j (w_j - 1/K)^2

The small ridge pulls the weights toward equal weights 1/K (K base models), so
two nearly identical bases split the weight instead of swinging between
corners. The intercept is free. Ties count as y = 0.5 in the log loss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from . import logistic

STACK_LAM = 0.001          # fixed before any DEV result; not tuned
STACK_FIRST_SEASON = 2003  # base walk-forward predictions start here (2002 would rest on a single season)
_EPS = 1e-12


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def fit_stack(L: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float = STACK_LAM) -> tuple[float, np.ndarray]:
    """(intercept, non-negative weights) for base log-odds L (n x K)."""
    L, y, w = np.asarray(L, float), np.asarray(y, float), np.asarray(w, float)
    K = L.shape[1]
    wn = w / w.sum()

    def obj(theta):
        z = theta[0] + L @ theta[1:]
        p = 1.0 / (1.0 + np.exp(-z))
        ll = -np.sum(wn * (y * np.log(p + _EPS) + (1 - y) * np.log(1 - p + _EPS)))
        g_z = wn * (p - y)
        pen = lam * np.sum((theta[1:] - 1.0 / K) ** 2)
        grad = np.r_[g_z.sum(), L.T @ g_z + 2 * lam * (theta[1:] - 1.0 / K)]
        return ll + pen, grad
    x0 = np.r_[0.0, np.full(K, 1.0 / K)]
    r = optimize.minimize(obj, x0, jac=True, method="L-BFGS-B", bounds=[(None, None)] + [(0.0, None)] * K,
                          options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10})
    return float(r.x[0]), r.x[1:].copy()


def predict(intercept: float, weights: np.ndarray, L: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(intercept + np.asarray(L, float) @ np.asarray(weights, float))))


def walk_forward(base: pd.DataFrame, y: np.ndarray, season: np.ndarray, test_seasons: tuple[int, int],
                 first_season: int = STACK_FIRST_SEASON, half_life: float = logistic.SEASON_HALF_LIFE,
                 lam: float = STACK_LAM) -> tuple[pd.Series, pd.DataFrame]:
    """Stacked P(home win) for each test season; `base` holds the base models' probabilities (one column each).

    Rows of a training season must have every base prediction (no NaN); test rows too.
    """
    names = list(base.columns)
    L_all = logit(base.to_numpy(float))
    y, season = np.asarray(y, float), np.asarray(season, int)
    pred = pd.Series(np.nan, index=base.index, dtype=float)
    rows = []
    for S in range(int(test_seasons[0]), int(test_seasons[1]) + 1):
        train = (season >= first_season) & (season < S)
        test = season == S
        if not test.any():
            continue
        if np.isnan(L_all[train]).any() or np.isnan(L_all[test]).any():
            raise ValueError(f"missing base predictions for the stack of season {S}")
        c, wts = fit_stack(L_all[train], y[train], logistic.season_weights(season[train], S, half_life), lam)
        pred[test] = predict(c, wts, L_all[test])
        rows.append({"season": S, "n_train": int(train.sum()), "intercept": c, **dict(zip(names, map(float, wts)))})
    return pred, pd.DataFrame(rows)
