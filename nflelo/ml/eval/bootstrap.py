"""Paired bootstrap confidence intervals on metric differences.

Both models are scored on the same games. Each replicate resamples games with
replacement and recomputes mean(loss_a) - mean(loss_b). An improvement counts
only if the 95% interval excludes zero. Seeded, so results are reproducible.
"""
from __future__ import annotations

import numpy as np

from .metrics import PER_GAME


def paired_bootstrap(y, p_a, p_b, metric: str = "brier", reps: int = 2000, seed: int = 20261003,
                     level: float = 0.95, chunk: int = 250) -> dict:
    """CI for metric(a) - metric(b). Negative means model a scores better (lower loss)."""
    if metric not in PER_GAME:
        raise ValueError(f"metric must be one of {sorted(PER_GAME)}")
    y, a, b = (np.asarray(v, float) for v in (y, p_a, p_b))
    if not (len(y) == len(a) == len(b)):
        raise ValueError("y, p_a, p_b must be the same length (paired games)")
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("predictions contain NaN; score both models on identical games")
    d = PER_GAME[metric](y, a) - PER_GAME[metric](y, b)
    return paired_bootstrap_diff(d, reps=reps, seed=seed, level=level, chunk=chunk) | {"metric": metric}


def paired_bootstrap_diff(d, reps: int = 2000, seed: int = 20261003, level: float = 0.95,
                          chunk: int = 250) -> dict:
    """Bootstrap CI for the mean of per-game loss differences `d`."""
    d = np.asarray(d, float)
    n = len(d)
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    for start in range(0, reps, chunk):
        k = min(chunk, reps - start)
        idx = rng.integers(0, n, size=(k, n))
        means[start:start + k] = d[idx].mean(axis=1)
    alpha = (1 - level) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return {"n": int(n), "diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "level": level, "reps": int(reps), "seed": int(seed),
            "excludes_zero": bool(lo > 0 or hi < 0), "boot_se": float(means.std(ddof=1))}
