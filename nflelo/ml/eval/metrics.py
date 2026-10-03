"""Scoring rules.

Win probability: Brier (primary), log loss, and accuracy come straight from
`nflelo.evaluate.metrics`, so every number here matches the Elo reports (ties
count as 0.5). Margin and total: mean absolute error and CRPS.
"""
from __future__ import annotations

import math

import numpy as np

from ...evaluate import metrics as _elo_metrics
from ...evaluate import outcome  # noqa: F401  (re-exported: home result 1 / 0.5 / 0)


def win_metrics(y, p) -> dict:
    """{"n", "brier", "logloss", "accuracy"} on identical arrays (ties y=0.5 count in Brier and log loss)."""
    return _elo_metrics(np.asarray(y, float), np.asarray(p, float))


def brier_per_game(y, p) -> np.ndarray:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return (y - p) ** 2


def logloss_per_game(y, p) -> np.ndarray:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


PER_GAME = {"brier": brier_per_game, "logloss": logloss_per_game}


def margin_mae(actual, predicted) -> float:
    """Mean absolute error of a point prediction (margin or total)."""
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.mean(np.abs(a - p)))


def _norm_pdf(z):
    return np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _norm_cdf(z):
    from scipy.special import ndtr  # scipy ships with scikit-learn and statsmodels
    return ndtr(z)


def crps_normal_per_game(actual, mu, sigma) -> np.ndarray:
    """CRPS of a Normal(mu, sigma) forecast, closed form (Gneiting and Raftery 2007). Lower is better."""
    y, m, s = (np.asarray(v, float) for v in (actual, mu, sigma))
    s = np.broadcast_to(s, np.broadcast(y, m).shape)
    if np.any(s <= 0):
        raise ValueError("sigma must be positive")
    z = (y - m) / s
    return s * (z * (2 * _norm_cdf(z) - 1) + 2 * _norm_pdf(z) - 1 / math.sqrt(math.pi))


def crps_normal(actual, mu, sigma) -> float:
    return float(np.mean(crps_normal_per_game(actual, mu, sigma)))


def crps_samples_per_game(actual, samples) -> np.ndarray:
    """CRPS from a sample (ensemble) forecast: samples has shape (n_games, n_draws).

    Uses the energy form E|X - y| - 0.5 E|X - X'|, computed exactly via sorting.
    Works for any distribution, including discrete margins with key numbers.
    """
    y = np.asarray(actual, float)
    x = np.sort(np.asarray(samples, float), axis=1)
    m = x.shape[1]
    term1 = np.mean(np.abs(x - y[:, None]), axis=1)
    i = np.arange(1, m + 1)
    # E|X - X'| over all ordered pairs = 2/m^2 * sum_i (2i - m - 1) x_(i)
    term2 = (2.0 / (m * m)) * np.sum((2 * i - m - 1) * x, axis=1)
    return term1 - 0.5 * term2


def crps_samples(actual, samples) -> float:
    return float(np.mean(crps_samples_per_game(actual, samples)))
