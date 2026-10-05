"""M6 metrics for 53-bin yard distributions (LICENSE: CC BY-SA 4.0 when applied to M6 outputs).

CRPS (the primary metric, spec section 6). For a play with predicted bin
probabilities p (already folded onto the bins possible from its yard line)
and observed bin b, with F the predicted CDF and H the observed step,

    CRPS = sum over thresholds k of (F_k - H_k)^2,   F_k = p_0 + ... + p_k,   H_k = 1 if b <= k else 0,

summed over the thresholds between consecutive POSSIBLE bins (k = every
possible non-touchdown bin). Between -9 and +40 the thresholds are one yard
apart, so CRPS is in yards: a point forecast that is off by d yards scores d.
The impossible bins between a play's last possible yard and the touchdown
are skipped, so a touchdown from the 3 is one threshold beyond "2 yards",
not 49. Lower is better. This is the discrete ranked probability score the
Big Data Bowl rushing contest used (there divided by 199).

Also: bin log loss; derived event probabilities P(first down or TD), P(20+
yards), P(loss) with reliability tables, ECE and the one-sided null check
(`eval.calibration.ece_null_range`: the 95th percentile a perfectly
calibrated forecaster shows at this n); turnover Brier. A calibration check
passes iff ECE <= that p95 OR ECE <= 0.010 (`ECE_FLOOR`, decided 2026-10-05
before any holdout number); the null-only verdict is kept as `ok_null`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..eval import calibration
from . import data as pdata

EVENTS = {"first": "ev_first", "20plus": "ev_20", "loss": "ev_loss"}


def crps(p: np.ndarray, b: np.ndarray, yardline: np.ndarray) -> np.ndarray:
    """Per-play discrete CRPS (see the module docstring). p: (n, 53) folded; b: observed bins."""
    p = np.asarray(p, float)
    b = np.asarray(b, int)
    F = np.cumsum(p, axis=1)[:, :pdata.TD_BIN]
    H = (b[:, None] <= np.arange(pdata.TD_BIN)[None, :]).astype(float)
    m = pdata.possible_mask(yardline)[:, :pdata.TD_BIN]
    return ((F - H) ** 2 * m).sum(axis=1)


def logloss(p: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, float)
    return -np.log(np.clip(p[np.arange(len(p)), np.asarray(b, int)], eps, 1.0))


def event_probs(p: np.ndarray, ydstogo: np.ndarray, yardline: np.ndarray) -> dict:
    """P(first down or TD), P(20+ yards), P(loss) from folded bin probabilities."""
    p = np.asarray(p, float)
    yk = np.concatenate([pdata.BIN_YARDS[:pdata.BIG_BIN], [np.inf]])   # 41+ counts as reaching any distance
    nontd = p[:, :pdata.TD_BIN]
    td = p[:, pdata.TD_BIN]
    ytg = np.asarray(ydstogo, float)[:, None]
    first = (nontd * (yk[None, :] >= ytg)).sum(axis=1) + td
    p20 = nontd[:, 30:].sum(axis=1) + td * (np.asarray(yardline, float) >= 20)
    loss = nontd[:, :10].sum(axis=1)
    return {"first": np.clip(first, 0, 1), "20plus": np.clip(p20, 0, 1), "loss": np.clip(loss, 0, 1)}


def expected_yards(p: np.ndarray, yardline: np.ndarray, big: float = 55.0, low: float = -12.0) -> np.ndarray:
    """Mean of the distribution (bin 0 at `low`, the 41+ bin at `big`, a TD at the yard line). Descriptive only."""
    yk = pdata.BIN_YARDS.copy()
    yk[0], yk[-1] = low, big
    y = np.asarray(yardline, float)
    return (np.asarray(p)[:, :pdata.TD_BIN] * yk[None, :]).sum(axis=1) + np.asarray(p)[:, pdata.TD_BIN] * y


def per_play(p: np.ndarray, p_tov: np.ndarray, table: pd.DataFrame) -> pd.DataFrame:
    """Per-play losses (for paired, game-clustered bootstraps): crps, logloss, tov (squared error)."""
    yl = table["yardline_100"].to_numpy(float)
    b = table["bin"].to_numpy(int)
    return pd.DataFrame({"crps": crps(p, b, yl), "logloss": logloss(p, b),
                         "tov": (np.asarray(p_tov, float) - table["turnover"].to_numpy(float)) ** 2},
                        index=table.index)


# Practical floor for every M6 pass/fail calibration rule (decided 2026-10-05, before any holdout number): at
# n ~ 55k-180k plays the calibrated-null band is ~0.005 wide, so the null test alone flags miscalibration far too
# small to matter (the baseline fails it on dev). A check passes if ECE <= null p95 OR ECE <= ECE_FLOOR.
ECE_FLOOR = 0.010


def calibration_check(y, prob, reps: int = 500, seed: int = 20261003) -> dict:
    """ECE (10 uniform bins). ok iff ECE <= the one-sided null's 95th percentile at this n OR ECE <= ECE_FLOOR;
    `ok_null` keeps the null-only result (descriptive)."""
    e = calibration.ece(y, prob)
    null = calibration.ece_null_range(prob, reps=reps, seed=seed)
    ok_null = bool(e <= null["p95"])
    return {"ece": float(e), "null_p95": null["p95"], "null_median": null["median"], "floor": ECE_FLOOR,
            "ok_null": ok_null, "ok": bool(ok_null or e <= ECE_FLOOR),
            "mean_pred": float(np.mean(prob)), "observed": float(np.mean(y))}


def summary(p: np.ndarray, p_tov: np.ndarray, table: pd.DataFrame, calib: bool = True, reps: int = 500) -> dict:
    """Pooled metrics: CRPS, bin log loss, turnover Brier, and (with `calib`) the derived-event calibration."""
    pp = per_play(p, p_tov, table)
    out = {"n": int(len(table)), "games": int(table["game_id"].nunique()), "crps": float(pp["crps"].mean()),
           "logloss": float(pp["logloss"].mean()), "tov_brier": float(pp["tov"].mean())}
    ev = event_probs(p, table["ydstogo"].to_numpy(float), table["yardline_100"].to_numpy(float))
    for k, col in EVENTS.items():
        y = table[col].to_numpy(float)
        out[f"brier_{k}"] = float(np.mean((ev[k] - y) ** 2))
        if calib:
            out[f"cal_{k}"] = calibration_check(y, ev[k], reps=reps)
    if calib:
        out["cal_tov"] = calibration_check(table["turnover"].to_numpy(float), p_tov, reps=reps)
    return out


def reliability(table: pd.DataFrame, p: np.ndarray, event: str = "first", bins: int = 10) -> pd.DataFrame:
    ev = event_probs(p, table["ydstogo"].to_numpy(float), table["yardline_100"].to_numpy(float))
    return calibration.reliability_table(table[EVENTS[event]].to_numpy(float), ev[event], bins)
