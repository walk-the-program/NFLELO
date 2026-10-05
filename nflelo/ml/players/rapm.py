"""Bayesian RAPM: on-field player ratings from who was on the field (context/ml-m5b-method.md, section 2).

LICENSE: CC BY-SA 4.0. The ratings, fits and any file written from them are
derived from nflverse participation data (NFL Next Gen Stats via nflverse
2016-2022, FTN Data via nflverse 2023+). Only M5b modules may import this;
nothing here may flow into A4s or any CC BY artifact.

Model. Every kept scrimmage play (players/participation.py) is one equation

    EPA = mu + sum_{11 offense} O[p] + sum_{11 defense} D[p] + h * home + situation terms + noise

Each (player, side) pair has its own coefficient. D is EPA allowed, so a good
defender has a negative D and his reported value is -D. Situation terms
(`fixed_matrix`): down (2, 3, 4 vs 1), distance bucket (3-6, 7-10, 11+ vs
1-2), field zone by yards to the end zone (11-20, 21-50, 51-80, 81+ vs 1-10),
and pass vs run. mu, h and the situation terms are not penalized.

Prior (ridge toward a prior mean, not zero). Each player coefficient has
prior N(m_p, sigma^2 / lam_g): m_p is his M5 box-score value (EPA per
on-field play above a replacement player at his position, a CC BY input),
times `box_scale`, with the sign flipped for defenders; a player with no box
record gets 0, the replacement level. lam_g (in plays) is the ridge strength
of his position group, so linemen, who are on the field together, can be
held close to their prior while receivers move freely. A player seen on the
"wrong" side of the ball (a lineman at fullback) gets prior 0 and `lam_other`.

Season chaining. Ratings for season S are fit on seasons before S only.
The prior for S is the posterior through S-1 with its evidence discounted
toward the box prior: in precision form, with H_T = A_T'A_T and g_T = A_T'y_T
the per-season sufficient statistics of the design A_T = [players | fixed],

    H(S) = sum_{T < S} decay**(S-1-T) * H_T,   g(S) likewise,

which is exactly "last season's posterior, decayed toward the box prior"
carried with its full covariance (the chain H(S+1) = decay * H(S) + H_S).
decay = 1 keeps all evidence; decay = 0 uses season S-1 alone.

Solve. With delta = beta - m (the ridge-with-offset form),
(H + Lambda) [delta; gamma] = g - H [m; 0], solved by conjugate gradients
with a Jacobi preconditioner (`solver="cg"`, scipy.sparse.linalg.cg) or a
dense Cholesky (`solver="direct"`, used to check CG and for posterior SDs).
Only players seen in training enter the solve; everyone else stays at his prior.

Uncertainty. sigma^2 is the weighted residual variance; the posterior
covariance is sigma^2 (H + Lambda)^-1 (exact for this Gaussian model), and
its diagonal gives each player's SD. Unseen players have the prior SD
sigma / sqrt(lam_g).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
from scipy import linalg, sparse
from scipy.sparse.linalg import cg

from . import participation as pt
from . import positions as pos

LICENSE_NOTE = pt.LICENSE_NOTE
GROUPS = ("QB", "RB", "WR", "TE", "OL", "DL", "LB", "CB", "S")
SIT_NAMES = ["home", "down2", "down3", "down4", "dist3_6", "dist7_10", "dist11p", "zone11_20", "zone21_50",
             "zone51_80", "zone81p", "pass"]
FIXED_NAMES = ["intercept"] + SIT_NAMES
BASE_FIXED = ["intercept", "home"]


@dataclass(frozen=True)
class RapmConfig:
    lam: tuple = tuple(sorted({g: 4000.0 for g in GROUPS}.items()))  # (group, ridge strength in plays)
    lam_other: float = 20000.0    # unknown group, or a player on the other side of the ball
    decay: float = 0.5            # weight of each older season's evidence, per season
    situation: bool = True
    box_scale: float = 1.0        # prior mean = box_scale * M5 box value

    @property
    def lamd(self) -> dict:
        return dict(self.lam)

    def with_lam(self, **kw) -> "RapmConfig":
        return replace(self, lam=tuple(sorted((self.lamd | kw).items())))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["lam"] = self.lamd
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RapmConfig":
        d = dict(d)
        d["lam"] = tuple(sorted(dict(d["lam"]).items()))
        return cls(**d)


# Chosen by `scripts/ml_m5b.py tune` on the dev predictions (2018 from 2016-17, 2019 from 2016-18),
# play-level EPA MSE, coordinate descent over the group ridge strengths, decay, box_scale and the
# situation switch; see context/ml.md, "M5b build notes". Frozen for the holdout pre-registration.
# DL at the top of the grid means "the data do not move DL off their prior" (a prior-only group).
TUNED = RapmConfig(lam=tuple(sorted({"QB": 500.0, "RB": 1000.0, "WR": 4000.0, "TE": 2000.0, "OL": 1000.0,
                                     "DL": 1024000.0, "LB": 2000.0, "CB": 32000.0, "S": 4000.0}.items())),
                   lam_other=20000.0, decay=1.0, situation=True, box_scale=0.25)


# --------------------------------------------------------------------------- design pieces

def fixed_matrix(plays: pd.DataFrame) -> np.ndarray:
    """(n x 13) unpenalized columns, in FIXED_NAMES order."""
    n = len(plays)
    down = plays["down"].to_numpy(int)
    ytg = plays["ydstogo"].to_numpy(float)
    yl = plays["yardline_100"].to_numpy(float)
    cols = [np.ones(n), plays["home"].to_numpy(float),
            down == 2, down == 3, down >= 4,
            (ytg >= 3) & (ytg <= 6), (ytg >= 7) & (ytg <= 10), ytg >= 11,
            (yl > 10) & (yl <= 20), (yl > 20) & (yl <= 50), (yl > 50) & (yl <= 80), yl > 80,
            plays["is_pass"].to_numpy(float)]
    return np.column_stack([np.asarray(c, float) for c in cols])


def fixed_columns(situation: bool) -> np.ndarray:
    return np.arange(len(FIXED_NAMES)) if situation else np.array([FIXED_NAMES.index(c) for c in BASE_FIXED])


def prior_vectors(ids: np.ndarray, side: np.ndarray, priors: pd.DataFrame, cfg: RapmConfig
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prior mean m, ridge strength lam, and group for each (id, side) column.

    `priors` is indexed by gsis_id with columns `group` (as of the last training
    season) and `v_box` (M5 box value; positive = good). Missing players: group
    None, v_box 0.
    """
    pr = priors.reindex(pd.Index(ids, dtype=object))
    grp = pr["group"].to_numpy(object)
    v = pr["v_box"].fillna(0.0).to_numpy(float)
    gside = np.array([pos.SIDE.get(g) if isinstance(g, str) else None for g in grp], dtype=object)
    match = gside == side
    sign = np.where(side == "off", 1.0, -1.0)
    m = np.where(match, cfg.box_scale * sign * v, 0.0)
    lamd = cfg.lamd
    lam = np.array([lamd[g] if ok and g in lamd else cfg.lam_other for g, ok in zip(grp, match)], float)
    return m, lam, grp


# --------------------------------------------------------------------------- data and fits

@dataclass
class RapmFit:
    """Season-ahead ratings for season S (fit on seasons before S). CC BY-SA 4.0."""
    season: int
    cfg: RapmConfig
    keys: np.ndarray            # global column keys "off|id" / "def|id"
    beta: np.ndarray            # rating per key (prior mean where unseen)
    prior: np.ndarray           # prior mean per key
    lam: np.ndarray
    group: np.ndarray
    active: np.ndarray          # bool: seen in training
    wsnaps: np.ndarray          # weighted plays on the field in training (diag of H)
    gamma: dict                 # fixed effects by name
    sigma2: float
    n_train: float              # weighted training plays
    sd: np.ndarray | None = None
    priors: pd.DataFrame | None = None
    solver_info: dict | None = None

    def ratings(self) -> pd.DataFrame:
        """One row per (player, side): rating (O or D), value (O or -D), prior value, SD, weighted snaps."""
        side = np.array([k[:3] for k in self.keys], dtype=object)
        sign = np.where(side == "off", 1.0, -1.0)
        out = pd.DataFrame({"gsis_id": [k[4:] for k in self.keys], "side": side, "group": self.group,
                            "rating": self.beta, "value": sign * self.beta, "prior_value": sign * self.prior,
                            "sd": self.sd if self.sd is not None else np.nan, "wsnaps": self.wsnaps,
                            "active": self.active, "lam": self.lam})
        out.attrs["license"] = LICENSE_NOTE
        out.attrs["season_ahead"] = self.season
        return out

    def predict(self, plays: pd.DataFrame) -> np.ndarray:
        """Predicted EPA for each play from its 22 players and situation (unseen players at their prior)."""
        ko, kd = pt.player_keys(plays)
        keys = np.concatenate([ko, kd], axis=1)
        uk, inv = np.unique(keys.ravel(), return_inverse=True)
        b = pd.Series(self.beta, index=self.keys).reindex(uk).to_numpy(float, copy=True)
        miss = np.isnan(b)
        if miss.any():
            ids = np.array([k[4:] for k in uk[miss]], dtype=object)
            sd_ = np.array([k[:3] for k in uk[miss]], dtype=object)
            pri = self.priors if self.priors is not None else pd.DataFrame(columns=["group", "v_box"])
            b[miss] = prior_vectors(ids, sd_, pri, self.cfg)[0]
        player = b[inv].reshape(keys.shape).sum(axis=1)
        Z = fixed_matrix(plays)
        g = np.array([self.gamma.get(c, 0.0) for c in FIXED_NAMES])
        return player + Z @ g


class RapmData:
    """Design and per-season sufficient statistics for every play given (any seasons). CC BY-SA 4.0.

    `fit(S, ...)` reads only the statistics of seasons before S, so a data
    object built on all seasons can serve every season-ahead fit.
    """

    def __init__(self, plays: pd.DataFrame):
        self.plays = plays.reset_index(drop=True)
        self.index = pt.player_index(self.plays)
        self.P = len(self.index)
        self.X = pt.design(self.plays, self.index)
        self.Z = fixed_matrix(self.plays)
        self.y = self.plays["epa"].to_numpy(float)
        self.season = self.plays["season"].to_numpy(int)
        self._stats: dict = {}
        self._warm: dict = {}

    def seasons(self) -> list[int]:
        return sorted(set(self.season.tolist()))

    def rows(self, S: int) -> np.ndarray:
        return np.flatnonzero(self.season == S)

    def stats(self, T: int):
        if T not in self._stats:
            r = self.rows(T)
            A = sparse.hstack([self.X[r], sparse.csr_matrix(self.Z[r])]).tocsr()
            y = self.y[r]
            self._stats[T] = ((A.T @ A).tocsr(), np.asarray(A.T @ y).ravel(), float(y @ y), float(len(r)))
        return self._stats[T]

    def combined(self, S: int, decay: float):
        H = None
        g = None
        yy = n = 0.0
        used = []
        for T in self.seasons():
            if T >= S:
                continue
            w = float(decay) ** (S - 1 - T)
            if w <= 0:
                continue
            Ht, gt, yyt, nt = self.stats(T)
            H = Ht * w if H is None else H + Ht * w
            g = gt * w if g is None else g + gt * w
            yy += w * yyt
            n += w * nt
            used.append((T, w))
        return H, g, yy, n, used

    def fit(self, S: int, priors: pd.DataFrame, cfg: RapmConfig, sd: bool = False, solver: str = "cg",
            tol: float = 1e-10, warm: bool = True) -> RapmFit:
        """Season-ahead fit for season S from seasons < S, shrunk toward `priors` (see `prior_vectors`)."""
        m, lam, grp = prior_vectors(self.index.ids, self.index.side, priors, cfg)
        F = len(FIXED_NAMES)
        H, g, yy, nw, used = self.combined(S, cfg.decay)
        if H is None:  # no training data: the prior is the rating
            return RapmFit(S, cfg, self.index.keys, m.copy(), m, lam, grp, np.zeros(self.P, bool), np.zeros(self.P),
                           {c: 0.0 for c in FIXED_NAMES}, np.nan, 0.0,
                           np.full(self.P, np.nan) if not sd else np.full(self.P, np.nan), priors,
                           {"used": [], "iterations": 0})
        diag = H.diagonal()
        active = diag[:self.P] > 0
        act = np.flatnonzero(active)
        fcols = fixed_columns(cfg.situation)
        cols = np.concatenate([act, self.P + fcols])
        Hs = H[cols][:, cols].tocsr()
        mfull = np.concatenate([m, np.zeros(F)])
        rhs = g[cols] - H[cols] @ mfull
        pen = np.concatenate([lam[act], np.full(len(fcols), 1e-9 * max(nw, 1.0))])
        A = (Hs + sparse.diags(pen)).tocsr()
        info = {"used": used, "solver": solver}
        if solver == "direct" or sd:
            Ad = A.toarray()
            # numpy's Cholesky: scipy's lower-triangular potrf (Accelerate build) fails spuriously on these systems
            L = np.linalg.cholesky(Ad)
            x = linalg.solve_triangular(L.T, linalg.solve_triangular(L, rhs, lower=True), lower=False)
            res = np.abs(Ad @ x - rhs).max() / max(np.abs(rhs).max(), 1e-300)
            if not res < 1e-8:
                raise RuntimeError(f"direct solve residual {res:.1e}")
            info["iterations"] = 0
        else:
            key = (S, cfg.situation, tuple(T for T, _ in used))
            x0 = self._warm.get(key) if warm else None
            Minv = sparse.diags(1.0 / A.diagonal())
            x, it = _cg(A, rhs, x0, Minv, tol)
            info["iterations"] = it
            if warm:
                self._warm[key] = x
        k = len(act)
        beta = m.copy()
        beta[act] += x[:k]
        gfull = np.zeros(F)
        gfull[fcols] = x[k:]
        theta = np.concatenate([beta, gfull])
        rss = yy - 2 * theta @ g + theta @ (H @ theta)
        sigma2 = float(rss / nw)
        sdv = None
        if sd:
            Linv = linalg.solve_triangular(L, np.eye(len(cols)), lower=True)
            var = (Linv ** 2).sum(axis=0)[:k] * sigma2
            sdv = np.sqrt(sigma2 / lam)
            sdv[act] = np.sqrt(var)
        return RapmFit(S, cfg, self.index.keys, beta, m, lam, grp, active, diag[:self.P],
                       {c_: float(v) for c_, v in zip(FIXED_NAMES, gfull)}, sigma2, nw, sdv, priors, info)

    def predict_rows(self, fit: RapmFit, rows: np.ndarray) -> np.ndarray:
        """Fast prediction for plays of this data object (global index)."""
        g = np.array([fit.gamma[c] for c in FIXED_NAMES])
        return np.asarray(self.X[rows] @ fit.beta).ravel() + self.Z[rows] @ g


def _cg(A, b, x0, Minv, tol: float) -> tuple[np.ndarray, int]:
    it = [0]

    def cb(_):
        it[0] += 1
    x, status = cg(A, b, x0=x0, rtol=tol, atol=0.0, maxiter=20000, M=Minv, callback=cb)
    if status != 0:
        raise RuntimeError(f"conjugate gradients did not converge (status {status})")
    return x, it[0]


# --------------------------------------------------------------------------- season-ahead API

def season_ahead(plays: pd.DataFrame, S: int, priors: pd.DataFrame, cfg: RapmConfig = TUNED,
                 leaky: bool = False, sd: bool = False) -> RapmFit:
    """Ratings for season S from plays of seasons before S only. `leaky=True` (the positive control
    for the leakage check) also trains on season S itself."""
    train = plays[plays["season"] <= S] if leaky else plays[plays["season"] < S]
    fit_S = S + 1 if leaky else S
    return RapmData(train).fit(fit_S, priors, cfg, sd=sd)


def corrupt_plays(plays: pd.DataFrame, first_season: int, seed: int = 0) -> pd.DataFrame:
    """Copy with every play of `first_season` and later scrambled: EPA, situation and all 22 IDs shuffled."""
    rng = np.random.default_rng(seed)
    p = plays.copy()
    late = (p["season"] >= first_season).to_numpy()
    k = int(late.sum())
    if not k:
        return p
    for c in ["epa", "down", "ydstogo", "yardline_100", "is_pass", "home"]:
        p.loc[late, c] = rng.permutation(p.loc[late, c].to_numpy())
    ids = p.loc[late, pt.OFF_COLS + pt.DEF_COLS].to_numpy(object)
    flat = rng.permutation(ids.ravel())
    p.loc[late, pt.OFF_COLS + pt.DEF_COLS] = flat.reshape(ids.shape)
    p.loc[late, "epa"] = p.loc[late, "epa"].to_numpy(float) + rng.normal(0, 1, k)
    return p


def leakage_check(clean: pd.DataFrame, dirty: pd.DataFrame, S: int, priors: pd.DataFrame,
                  cfg: RapmConfig = TUNED, leaky: bool = False) -> dict:
    """Fit on `clean` and on `dirty` (season S onward corrupted), predict the CLEAN season-S plays both times.

    Leak-free iff the predictions are identical. `leaky=True` trains on S too
    (the positive control), which must fail.
    """
    target = clean[clean["season"] == S]
    a = season_ahead(clean, S, priors, cfg, leaky).predict(target)
    b = season_ahead(dirty, S, priors, cfg, leaky).predict(target)
    diff = np.abs(a - b)
    return {"season": S, "plays": int(len(target)), "max_abs_diff": float(diff.max()) if len(diff) else 0.0,
            "changed": int((diff > 1e-12).sum()), "leak_free": bool((diff <= 1e-12).all()), "leaky_control": leaky}
