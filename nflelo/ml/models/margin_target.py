"""Margin-target game models (context/ml-m8-method.md, component 1).

A win/loss logistic learns from one bit per game. A margin model learns from a
less noisy number: the point margin, the game's EPA margin, or a blend. The
model is a weighted linear regression

    target = b0 + b1 * x1 + ... + bk * xk          (mu = the predicted target)

fit with A4s's protocol (walk-forward, season S fit on REG 2001..S-1, weights
0.5 ** ((S - 1 - season) / 8)). Two ways to turn mu into P(home win):

    normal   P = Phi((a + b * mu) / sigma), where margin = a + b * mu + e is fit
             by weighted least squares on the training games and sigma is the
             weighted RMS of e (for the margin target a = 0 and b = 1 exactly)
    logit    logit P = c0 + c1 * mu, a weighted logistic calibration of mu on the
             training games' win/loss (ties as two half rows)

Targets (columns of the frame, built by scripts/ml_m8.py):

    margin   home score minus away score
    epa      scale * (home offense EPA per play - away offense EPA per play), the
             raw game EPA margin of the M1 clean plays in points (scale = the
             Kalman filter's tuned 39.9 points per EPA/play)
    blend    w * margin + (1 - w) * epa, with w tuned on 2001-2005 only

The target is an outcome, so it is read only for training seasons (< S).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import logistic

METHODS = ("normal", "logit")


@dataclass
class MTFit:
    beta: np.ndarray          # intercept first
    conv_a: float
    conv_b: float
    sigma: float
    cal_a: float
    cal_b: float

    def mu(self, X: np.ndarray) -> np.ndarray:
        return self.beta[0] + np.asarray(X, float) @ self.beta[1:]

    def predict_proba(self, X: np.ndarray, method: str = "normal") -> np.ndarray:
        mu = self.mu(X)
        if method == "normal":
            return stats.norm.cdf((self.conv_a + self.conv_b * mu) / self.sigma)
        if method == "logit":
            return 1.0 / (1.0 + np.exp(-(self.cal_a + self.cal_b * mu)))
        raise ValueError(f"method must be one of {METHODS}")


def _wls(X: np.ndarray, t: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xd = np.column_stack([np.ones(len(t)), X])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(Xd * sw[:, None], t * sw, rcond=None)
    return beta


def fit(X: np.ndarray, target: np.ndarray, margin: np.ndarray, y: np.ndarray, w: np.ndarray) -> MTFit:
    """Fit the regression on rows with a finite target, then the two probability maps on every row."""
    X, target, margin, y, w = (np.asarray(a, float) for a in (X, target, margin, y, w))
    if X.ndim == 1:
        X = X[:, None]
    ok = np.isfinite(target)
    beta = _wls(X[ok], target[ok], w[ok])
    mu = beta[0] + X @ beta[1:]
    a, b = _wls(mu[:, None], margin, w)
    r = margin - (a + b * mu)
    sigma = float(np.sqrt(np.sum(w * r * r) / np.sum(w)))
    cal = logistic.fit_logistic(mu[:, None], y, w)
    return MTFit(beta, float(a), float(b), sigma, float(cal.intercept_[0]), float(cal.coef_[0][0]))


def walk_forward(frame: pd.DataFrame, features: list[str], test_seasons: tuple[int, int], target: str,
                 method: str = "normal", train_start: int = logistic.TRAIN_START,
                 half_life: float = logistic.SEASON_HALF_LIFE) -> tuple[pd.Series, pd.DataFrame]:
    """Out-of-sample P(home win) for each season in `test_seasons` (NaN elsewhere), and one row per refit.

    `frame` needs season, y, margin, the `target` column and the features (no NaN in features).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    X_all = frame[features].to_numpy(float)
    if np.isnan(X_all).any():
        raise ValueError(f"NaN in features {[f for f in features if frame[f].isna().any()]}")
    t_all = frame[target].to_numpy(float)
    m_all, y_all = frame["margin"].to_numpy(float), frame["y"].to_numpy(float)
    season = frame["season"].to_numpy(int)
    pred = pd.Series(np.nan, index=frame.index, dtype=float)
    rows = []
    for S in range(int(test_seasons[0]), int(test_seasons[1]) + 1):
        train = (season >= train_start) & (season < S)
        test = season == S
        if not test.any():
            continue
        f = fit(X_all[train], t_all[train], m_all[train], y_all[train],
                logistic.season_weights(season[train], S, half_life))
        pred[test] = f.predict_proba(X_all[test], method)
        rows.append({"season": S, "n_train": int(train.sum()), "intercept": float(f.beta[0]),
                     **{c: float(v) for c, v in zip(features, f.beta[1:])},
                     "conv_a": f.conv_a, "conv_b": f.conv_b, "sigma": f.sigma, "cal_a": f.cal_a, "cal_b": f.cal_b})
    return pred, pd.DataFrame(rows)
