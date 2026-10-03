"""Calibration: do 70% predictions win about 70% of the time?

`reliability_table` bins predictions and compares the mean prediction with
the observed home-win rate per bin (ties count 0.5). `ece` is the expected
calibration error: the game-weighted average gap between the two.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def reliability_table(y, p, bins: int = 10, strategy: str = "uniform") -> pd.DataFrame:
    """One row per non-empty bin: bin edges, games, mean predicted, observed rate, and the gap."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if strategy == "uniform":
        edges = np.linspace(0, 1, bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = which == b
        if m.any():
            rows.append({"lo": edges[b], "hi": edges[b + 1], "n": int(m.sum()),
                         "mean_pred": float(p[m].mean()), "observed": float(y[m].mean())})
    t = pd.DataFrame(rows)
    t["gap"] = t["observed"] - t["mean_pred"]
    return t


def ece(y, p, bins: int = 10, strategy: str = "uniform") -> float:
    """Expected calibration error (weights each bin by its share of games). Lower is better."""
    t = reliability_table(y, p, bins, strategy)
    return float((t["n"] * t["gap"].abs()).sum() / t["n"].sum())
