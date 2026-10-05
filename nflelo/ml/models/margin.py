"""Margin model (M4 Phase 2, context/ml-m4-method.md section 4): the distribution of home score minus away score.

Mean. A weighted least-squares line on the A4s features,

    mean = b0 + b1 * elo_logit + b2 * adj_epa_margin + b3 * qb_delta_diff

fit walk-forward exactly like M3: to predict season S, fit on REG games from
2001 through S - 1 with season-decay weights 0.5 ** ((S - 1 - season) / 8).
The mean is our spread (positive = home favored, the nflverse `spread_line`
convention used by the site).

Spread around the mean. `sigma` for season S is the season-weighted standard
deviation of the training residuals (same rows, same weights). Two shapes:

- "normal": Normal(mean, sigma), rounded to whole points, so ties exist and
  every score margin is an integer, as in football.
- "keynum": the same rounded normal, with each margin k reweighted by a
  factor r(k) and renormalized: P(k) proportional to q(k; mean, sigma) * r(k).
  The factors capture football's key numbers (3, 7, 10, 6, 14, 4 come up more
  often than a bell curve says; ties come up much less). They are fit once,
  on 2001-2005 only (never the DEV or holdout years), by iterative
  proportional fitting so that the expected count of every |k| matches the
  observed count. r is symmetric (r(k) = r(-k)) and is 1 beyond |k| = KMAX.

Win probability from a margin distribution: P(margin > 0) + 0.5 * P(margin = 0),
the same tie convention as the Brier scores everywhere else.

Inputs are market-free: the feature list is checked against the market-column
ban (decision D3) before any fit or prediction.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import ndtr

from ..data import is_market_column
from .logistic import SEASON_HALF_LIFE, TRAIN_START, season_weights

FEATURES = ["elo_logit", "adj_epa_margin", "qb_delta_diff"]
GRID = np.arange(-80, 81)          # every NFL margin ever recorded fits comfortably
KMAX = 20                          # reweighting factors are fit for |k| <= KMAX, 1 beyond (83% of 2001-2005 margins; per-margin counts are thin beyond)
KEYNUM_FIT_SEASONS = (2001, 2005)  # the only seasons the key-number factors ever see
SHAPES = ("normal", "keynum")
CHOSEN_SHAPE = "keynum"            # the DEV choice (scripts/ml_m4.py margin); see context/ml.md, M4 Phase 2 notes


def check_features(features) -> None:
    bad = [f for f in features if is_market_column(f)]
    if bad:
        raise ValueError(f"market columns can't be margin-model features (decision D3): {bad}")


# --------------------------------------------------------------------------- the mean and sigma

@dataclass
class MarginFit:
    """One season's fitted mean line and sigma."""
    features: list[str]
    intercept: float
    coef: dict
    sigma: float
    n_train: int = 0
    season: int | None = None

    def mean(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        return self.intercept + X @ np.array([self.coef[f] for f in self.features], float)

    def to_dict(self) -> dict:
        return {"features": list(self.features), "intercept": float(self.intercept),
                "coef": {f: float(self.coef[f]) for f in self.features}, "sigma": float(self.sigma),
                "n_train": int(self.n_train), "season": self.season}

    @classmethod
    def from_dict(cls, d: dict) -> "MarginFit":
        return cls(list(d["features"]), float(d["intercept"]), {k: float(v) for k, v in d["coef"].items()},
                   float(d["sigma"]), int(d.get("n_train", 0)), d.get("season"))


def fit_margin(X, y, w, features) -> MarginFit:
    """Weighted least squares for the mean, and the weighted residual SD (with a degrees-of-freedom correction)."""
    check_features(features)
    X, y, w = np.asarray(X, float), np.asarray(y, float), np.asarray(w, float)
    A = np.column_stack([np.ones(len(X)), X])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    resid = y - A @ beta
    n_eff = w.sum() ** 2 / (w ** 2).sum()
    dof = max(n_eff - A.shape[1], 1.0) / n_eff
    sigma = float(np.sqrt((w * resid ** 2).sum() / w.sum() / dof))
    return MarginFit(list(features), float(beta[0]), {f: float(b) for f, b in zip(features, beta[1:])}, sigma,
                     int(len(X)))


def walk_forward(frame: pd.DataFrame, features: list[str], test_seasons: tuple[int, int],
                 train_start: int = TRAIN_START, half_life: float = SEASON_HALF_LIFE,
                 y_col: str = "margin") -> tuple[pd.DataFrame, list[MarginFit]]:
    """Out-of-sample mean and sigma for every game of each test season (NaN elsewhere), plus each season's fit."""
    check_features(features)
    X_all = frame[features].to_numpy(float)
    if np.isnan(X_all).any():
        raise ValueError(f"NaN in features {[f for f in features if frame[f].isna().any()]}")
    y_all = frame[y_col].to_numpy(float)
    season = frame["season"].to_numpy(int)
    out = pd.DataFrame({"mu": np.nan, "sigma": np.nan}, index=frame.index)
    fits = []
    for S in range(int(test_seasons[0]), int(test_seasons[1]) + 1):
        train = (season >= train_start) & (season < S)
        test = season == S
        if not test.any():
            continue
        if not train.any():
            raise ValueError(f"no training games before season {S}")
        f = fit_margin(X_all[train], y_all[train], season_weights(season[train], S, half_life), features)
        f.season = S
        out.loc[test, "mu"] = f.mean(X_all[test])
        out.loc[test, "sigma"] = f.sigma
        fits.append(f)
    return out, fits


# --------------------------------------------------------------------------- shapes

def normal_pmf(mu, sigma, grid: np.ndarray = GRID) -> np.ndarray:
    """Rounded normal: P(k) = Phi((k + .5 - mu) / sigma) - Phi((k - .5 - mu) / sigma), one row per game."""
    mu = np.atleast_1d(np.asarray(mu, float))[:, None]
    s = np.broadcast_to(np.atleast_1d(np.asarray(sigma, float)), (mu.shape[0],))[:, None]
    hi = ndtr((grid[None, :] + 0.5 - mu) / s)
    lo = ndtr((grid[None, :] - 0.5 - mu) / s)
    p = hi - lo
    p[:, 0] += lo[:, 0]             # fold the tails into the end points so each row sums to 1
    p[:, -1] += 1.0 - hi[:, -1]
    return p


@dataclass
class KeyNumbers:
    """Reweighting factors r(|k|) for |k| = 0..KMAX (1 beyond)."""
    r: np.ndarray = field(default_factory=lambda: np.ones(KMAX + 1))

    def full(self, grid: np.ndarray = GRID) -> np.ndarray:
        a = np.abs(grid)
        out = np.ones(len(grid))
        m = a <= KMAX
        out[m] = self.r[a[m]]
        return out

    def to_dict(self) -> dict:
        return {"kmax": KMAX, "r_abs": [round(float(x), 6) for x in self.r], "fit_seasons": list(KEYNUM_FIT_SEASONS)}

    @classmethod
    def from_dict(cls, d: dict) -> "KeyNumbers":
        r = np.asarray(d["r_abs"], float)
        if len(r) != KMAX + 1:
            raise ValueError(f"expected {KMAX + 1} factors, got {len(r)}")
        return cls(r)


def keynum_pmf(mu, sigma, kn: KeyNumbers, grid: np.ndarray = GRID) -> np.ndarray:
    p = normal_pmf(mu, sigma, grid) * kn.full(grid)[None, :]
    return p / p.sum(axis=1, keepdims=True)


def fit_keynumbers(margins, mu, sigma, iters: int = 200, tol: float = 1e-10) -> KeyNumbers:
    """Iterative proportional fitting: choose r so expected counts of each |k| <= KMAX match observed counts."""
    y = np.asarray(margins, float).round().astype(int)
    q = normal_pmf(mu, sigma)
    a = np.abs(GRID)
    obs = np.array([(np.abs(y) == k).sum() for k in range(KMAX + 1)], float)
    r = np.ones(KMAX + 1)
    for _ in range(iters):
        full = np.ones(len(GRID))
        full[a <= KMAX] = r[a[a <= KMAX]]
        p = q * full[None, :]
        p /= p.sum(axis=1, keepdims=True)
        exp = np.array([p[:, a == k].sum() for k in range(KMAX + 1)])
        new = r * np.where(exp > 0, obs / np.maximum(exp, 1e-300), 1.0)
        new = np.where(obs > 0, new, 1e-6)  # an |k| never seen gets (almost) no mass; none is within KMAX in practice
        if np.max(np.abs(new - r)) < tol:
            r = new
            break
        r = new
    return KeyNumbers(r)


def keynumbers_from_frame(frame: pd.DataFrame, features: list[str] = FEATURES, y_col: str = "margin",
                          seasons: tuple[int, int] = KEYNUM_FIT_SEASONS) -> KeyNumbers:
    """Key-number factors from `seasons` only (2001-2005): an unweighted in-sample mean line and sigma on those
    games give the normal each margin is compared with. Nothing outside `seasons` is read."""
    t = frame[frame["season"].between(*seasons)]
    if t.empty:
        raise ValueError(f"no games in {seasons}")
    f = fit_margin(t[features].to_numpy(float), t[y_col].to_numpy(float), np.ones(len(t)), features)
    return fit_keynumbers(t[y_col].to_numpy(float), f.mean(t[features].to_numpy(float)), f.sigma)


def pmf(shape: str, mu, sigma, kn: KeyNumbers | None = None) -> np.ndarray:
    if shape == "normal":
        return normal_pmf(mu, sigma)
    if shape == "keynum":
        if kn is None:
            raise ValueError("the key-number shape needs fitted factors")
        return keynum_pmf(mu, sigma, kn)
    raise ValueError(f"shape must be one of {SHAPES}")


def win_prob(p: np.ndarray, grid: np.ndarray = GRID) -> np.ndarray:
    """P(margin > 0) + 0.5 * P(margin = 0) for each row of a pmf over `grid`."""
    return p[:, grid > 0].sum(axis=1) + 0.5 * p[:, grid == 0].sum(axis=1)


def crps_discrete_per_game(actual, p: np.ndarray, grid: np.ndarray = GRID) -> np.ndarray:
    """CRPS of an integer-valued forecast: sum over the grid of (F(k) - 1{y <= k})^2 (exact for step CDFs)."""
    y = np.asarray(actual, float)
    F = np.cumsum(p, axis=1)
    ind = (y[:, None] <= grid[None, :]).astype(float)
    return ((F - ind) ** 2).sum(axis=1)


def median(p: np.ndarray, grid: np.ndarray = GRID) -> np.ndarray:
    F = np.cumsum(p, axis=1)
    return grid[np.argmax(F >= 0.5, axis=1)]


# --------------------------------------------------------------------------- sampling

def sample(shape: str, mu: np.ndarray, sigma: float, rng: np.random.Generator,
           kn: KeyNumbers | None = None) -> np.ndarray:
    """Integer margins, one per entry of `mu` (any shape). Exact draws from the chosen shape.

    normal: round(mu + sigma * z). keynum: rejection sampling from the rounded
    normal, accepting k with probability r(k) / max(r); the accepted draws
    follow q(k) r(k) / sum(q r) exactly.
    """
    mu = np.asarray(mu, float)
    sigma = float(sigma)
    k = np.rint(mu + sigma * rng.standard_normal(mu.shape))
    if shape == "normal":
        return k.astype(np.int16)
    if shape != "keynum" or kn is None:
        raise ValueError("keynum sampling needs fitted factors")
    rmax = float(kn.r.max())
    acc_r = kn.r / rmax
    flat_mu = mu.ravel()
    out = np.zeros(flat_mu.shape, np.int16)
    idx = np.arange(flat_mu.size)
    kk = k.ravel()
    for _ in range(500):
        a = np.abs(kk).astype(int)
        acc = np.where(a <= KMAX, acc_r[np.minimum(a, KMAX)], 1.0 / rmax)
        ok = rng.uniform(size=acc.shape) < acc
        out[idx[ok]] = kk[ok]
        idx = idx[~ok]
        if idx.size == 0:
            return out.reshape(mu.shape)
        kk = np.rint(flat_mu[idx] + sigma * rng.standard_normal(idx.size))
    raise RuntimeError("rejection sampling did not finish")


# --------------------------------------------------------------------------- frozen season model

def version(fit: MarginFit, shape: str, kn: KeyNumbers | None, season: int) -> str:
    blob = json.dumps({"fit": fit.to_dict(), "shape": shape, "kn": kn.to_dict() if kn is not None else None},
                      sort_keys=True)
    return f"M-{season}-{hashlib.sha256(blob.encode()).hexdigest()[:8]}"


def spread(fit: MarginFit, feats: pd.DataFrame) -> np.ndarray:
    """The model spread (mean margin, positive = home favored) for a frame with the feature columns."""
    check_features(fit.features)
    return fit.mean(feats[fit.features].to_numpy(float))
