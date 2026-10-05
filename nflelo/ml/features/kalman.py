"""State-space team strength: a Kalman filter on point margins and EPA margins (context/ml-m3b-method.md, C2).

The model
---------
Each franchise i has a latent strength s_i, in points above an average team.
Home field is one more state, h (points). Strengths move in three ways:

    within a season   s_i <- s_i + w,  w ~ N(0, q_week)  per game week elapsed (a random walk)
    between seasons   s_i <- gamma * s_i + u,  u ~ N(0, q_season)   (mean reversion plus new noise)
    home field        h   <- h + v,    v ~ N(0, q_hfa)   per game week (a slowly drifting state)

Every played game (REG and POST) gives two measurements of d = s_home - s_away:

    margin            y_m = d + h * (not neutral) + e_m              e_m ~ N(0, sigma_m^2)
    EPA margin        y_e = scale * (epa_margin - epa_home * (not neutral)) = d + e_e,   e_e ~ N(0, sigma_e^2)

with corr(e_m, e_e) = rho (the two share the game's luck). `epa_margin` is the
game's home offense EPA per play minus the away offense EPA per play (the M1
clean plays: real snaps with an EPA, garbage time out). It is *raw*, not
opponent-adjusted: the filter adjusts for the opponent itself, because the
measurement is about the strength difference of the two teams that met.
Feeding an already-adjusted margin would count the opponent twice. `scale`
turns EPA per play into points; `epa_home` is the league's home EPA edge per
play (a fixed constant from 1999-2005), so the EPA measurement carries no home
term and home field is learned from the scores alone.

Why one strength per team and not offense and defense: both measurements are
margins, which only see offense minus defense. A split would need points
scored (totals), which the win-probability target does not use.

Why home field is a drifting state, not a constant: the home edge has fallen
from about 3 points in the early 2000s to about 1.5 in the 2020s (CONTEXT.md,
finding 2; Elo v2 learns it online for the same reason). As a state it is
re-estimated from every game's margin residual; `q_hfa` sets how fast it may
move, tuned with everything else. With q_hfa = 0 it would be a learned
constant (a running average since 1999). The tuned q_hfa is small (about
0.08 points of drift per season), so in practice home field is a slowly
updated estimate; the logistic models built on `kf_margin` also refit their
intercept every season, which absorbs any remaining drift.

The filter
----------
State x = (s_1..s_32, h), covariance P (33 x 33). Each game week, in timeline
order: (1) the time update (drift, or the season transition at a new season);
(2) a prediction for every game of the week from the state *before any game of
that week* (the M1 as-of rule: a Thursday result never reaches that Sunday);
(3) the measurement update, one game at a time in kickoff order, with both
measurements jointly (2 x 2 innovation covariance). A game with no
play-by-play uses the margin alone.

Outputs per game: `kf_margin` = predicted home margin (with home field unless
neutral) and `kf_sd` = SD of that prediction from the state uncertainty alone
(sqrt(a' P a)); the margin's own noise sigma_m comes on top. As a standalone
forecaster, P(home win) = Phi(kf_margin / sqrt(kf_sd^2 + sigma_m^2)).

Tuning (`tune`): maximum likelihood of the next game's measurements under
their pre-week predictive distribution, summed over REG and POST games
2000-2005 (1999 is burn-in). DEV (2006-2019) is never used, as in M3-D2. The
objective is the joint density of both measurements; the margin-only
likelihood, N(margin; kf_margin, kf_sd^2 + sigma_m^2), is reported next to it
(see `tune` for why it cannot be the objective by itself). The frozen result
is `TUNED`.

Start: 1999 week 1, every team at 0 with variance `p0`; home field at `h0`
with variance `ph0`. A franchise that has not played yet (Houston before 2002)
stays at 0 and only gains variance until its first game.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ...teams import FRANCHISES, franchise
from .. import asof as asof_mod
from . import opponent_adjust as oa
from . import team_efficiency as te

FEATURES = ["kf_margin", "kf_sd"]
TEAMS = sorted(FRANCHISES)
TUNE_SEASONS = (2000, 2005)
BURN_IN_FIRST = 1999


@dataclass(frozen=True)
class KalmanConfig:
    q_week: float = 0.5        # weekly drift variance of a team's strength (points^2)
    gamma: float = 0.7         # between-season reversion: strength is multiplied by gamma
    q_season: float = 6.0      # extra off-season variance (points^2)
    sigma_m: float = 13.5      # margin noise SD (points)
    sigma_e: float = 9.0       # EPA-measurement noise SD (points)
    rho: float = 0.5           # correlation of the two measurement noises
    scale: float = 60.0        # points per EPA-per-play margin
    q_hfa: float = 0.001       # weekly drift variance of home field (points^2)
    epa_home: float = 0.0      # league home EPA-per-play edge removed from the EPA measurement (fixed, 1999-2005)
    p0: float = 36.0           # initial team variance, 1999 week 1 (6 points SD)
    h0: float = 0.0            # initial home field (points)
    ph0: float = 9.0           # initial home-field variance
    garbage_low: float = 0.05
    garbage_high: float = 0.95

    def to_dict(self) -> dict:
        return asdict(self)


# Tuned by `tune` (scripts/ml_m3b.py tune): joint maximum likelihood of each game's margin and EPA margin,
# REG and POST 2000-2005, 1999 burn-in, four starts that agree; see context/ml.md, "M3b build notes".
TUNED = KalmanConfig(q_week=0.72158, gamma=0.48317, q_season=14.431, sigma_m=12.361, sigma_e=13.52,
                     rho=0.80709, scale=39.921, q_hfa=0.00034383, epa_home=0.0085843)

TUNED_KEYS = ("q_week", "gamma", "q_season", "sigma_m", "sigma_e", "rho", "scale", "q_hfa")


# --------------------------------------------------------------------------- inputs

def game_table(pbp: pd.DataFrame, sched: pd.DataFrame, cfg: KalmanConfig = TUNED) -> pd.DataFrame:
    """One row per played REG/POST game in kickoff order: teams, neutral flag, margin, raw EPA margin, timeline.

    `epa_margin` is home offense EPA per play minus away offense EPA per play
    (NaN when the game has no play-by-play).
    """
    if "as_of" not in sched.columns:
        sched = asof_mod.add_asof(sched)
    s = sched[sched["game_type"].isin(asof_mod.REG_POST_TYPES)].drop_duplicates("game_id")
    s = s[s["home_score"].notna() & s["away_score"].notna()].copy()
    seasons = s["season"].astype(int).to_numpy()
    out = pd.DataFrame({
        "game_id": s["game_id"].to_numpy(), "season": seasons, "week": s["week"].astype(int).to_numpy(),
        "home": [franchise(c, int(y)) for c, y in zip(s["home_team"], seasons)],
        "away": [franchise(c, int(y)) for c, y in zip(s["away_team"], seasons)],
        "neutral": ((s["location"] == "Neutral") if "location" in s.columns
                    else pd.Series(False, index=s.index)).to_numpy(bool),
        "margin": (s["home_score"].astype(float) - s["away_score"].astype(float)).to_numpy(),
        "kick_ns": te._utc_ns(s["kickoff"]),
    })
    ords = oa.week_ordinals(sched)
    out["ord"] = ords.reindex(pd.MultiIndex.from_frame(out[["season", "week"]])).to_numpy()
    plays = te.clean_plays(pbp, te.EfficiencyConfig(garbage_low=cfg.garbage_low, garbage_high=cfg.garbage_high))
    plays = plays[plays["game_id"].isin(set(out["game_id"]))]
    tab = plays.groupby(["game_id", "posteam"], sort=False)["epa"].agg(["size", "sum"]).reset_index()
    info = out.set_index("game_id")
    tab["season"] = tab["game_id"].map(info["season"])
    tab["team"] = [franchise(c, int(y)) for c, y in zip(tab["posteam"], tab["season"])]
    tab["rate"] = tab["sum"] / tab["size"]
    home_rate = tab[tab["team"].to_numpy() == tab["game_id"].map(info["home"]).to_numpy()].set_index("game_id")["rate"]
    away_rate = tab[tab["team"].to_numpy() == tab["game_id"].map(info["away"]).to_numpy()].set_index("game_id")["rate"]
    out["epa_margin"] = out["game_id"].map(home_rate) - out["game_id"].map(away_rate)
    return out.sort_values(["kick_ns", "game_id"], kind="stable").reset_index(drop=True)


def epa_home_edge(gt: pd.DataFrame, seasons: tuple[int, int] = (BURN_IN_FIRST, TUNE_SEASONS[1])) -> float:
    """Mean home-minus-away EPA per play over non-neutral games in `seasons` (the fixed `epa_home`)."""
    m = gt["season"].between(*seasons) & ~gt["neutral"] & gt["epa_margin"].notna()
    return float(gt.loc[m, "epa_margin"].mean())


# --------------------------------------------------------------------------- the filter

@dataclass
class FilterResult:
    pred_mean: np.ndarray      # kf_margin per game (gt order)
    pred_var: np.ndarray       # kf_sd^2 per game
    post_mean: np.ndarray      # predicted margin after the game's own week (positive-control use only)
    teams: list
    snapshots: list | None     # (season, week, x, diag P) at the start of each week when requested
    x_final: np.ndarray | None = None   # state after the last game (no further time update)
    P_final: np.ndarray | None = None
    epa_pred: tuple | None = None       # (mean, var, cov with the margin row) of the EPA measurement's d part


def run_filter(gt: pd.DataFrame, cfg: KalmanConfig, snapshots: bool = False) -> FilterResult:
    """Run the filter over `gt` (a `game_table`) in timeline order. See the module docstring."""
    T = len(TEAMS)
    H = T
    code = {t: i for i, t in enumerate(TEAMS)}
    home = np.array([code[t] for t in gt["home"]], dtype=int)
    away = np.array([code[t] for t in gt["away"]], dtype=int)
    nn = (~gt["neutral"].to_numpy(bool)).astype(float)
    y_m = gt["margin"].to_numpy(float)
    e = gt["epa_margin"].to_numpy(float)
    y_e = cfg.scale * (e - cfg.epa_home * nn)
    has_e = ~np.isnan(e)
    seasons = gt["season"].to_numpy(int)
    ords = gt["ord"].to_numpy(int)
    sm2, se2 = cfg.sigma_m ** 2, cfg.sigma_e ** 2
    cov = cfg.rho * cfg.sigma_m * cfg.sigma_e
    R2 = np.array([[sm2, cov], [cov, se2]])

    x = np.zeros(T + 1)
    x[H] = cfg.h0
    P = np.diag(np.r_[np.full(T, cfg.p0), cfg.ph0]).astype(float)
    n = len(gt)
    pm, pv, post = np.empty(n), np.empty(n), np.empty(n)
    pe_mean, pe_var, pe_cov = np.empty(n), np.empty(n), np.empty(n)   # EPA-row predictive (b'x, b'Pb, a'Pb)
    snaps = [] if snapshots else None
    diag_t = np.arange(T)
    if n == 0:
        return FilterResult(pm, pv, post, TEAMS, snaps, x.copy(), P.copy(), (pe_mean, pe_var, pe_cov))

    # week blocks: contiguous runs of the same timeline position
    starts = np.flatnonzero(np.r_[True, ords[1:] != ords[:-1]])
    ends = np.r_[starts[1:], n]
    prev_season, prev_ord = None, None
    for b0, b1 in zip(starts, ends):
        S, o = int(seasons[b0]), int(ords[b0])
        # (1) time update
        if prev_season is not None:
            if S != prev_season:
                x[:T] *= cfg.gamma
                P[:T, :] *= cfg.gamma
                P[:, :T] *= cfg.gamma
                P[diag_t, diag_t] += cfg.q_season
                P[H, H] += cfg.q_hfa
            else:
                dt = max(o - prev_ord, 1)
                P[diag_t, diag_t] += cfg.q_week * dt
                P[H, H] += cfg.q_hfa * dt
        prev_season, prev_ord = S, o
        if snaps is not None:
            snaps.append((S, int(gt["week"].iat[b0]), x.copy(), np.diag(P).copy()))
        # (2) predictions for the whole week, before any of its games
        i, j, h = home[b0:b1], away[b0:b1], nn[b0:b1]
        pm[b0:b1] = x[i] - x[j] + h * x[H]
        pv[b0:b1] = (P[i, i] + P[j, j] - 2 * P[i, j]
                     + h * (P[H, H] + 2 * P[i, H] - 2 * P[j, H]))
        pe_mean[b0:b1] = x[i] - x[j]
        pe_var[b0:b1] = P[i, i] + P[j, j] - 2 * P[i, j]
        pe_cov[b0:b1] = pe_var[b0:b1] + h * (P[i, H] - P[j, H])
        # (3) measurement updates, one game at a time
        for g in range(b0, b1):
            gi, gj, gh = home[g], away[g], nn[g]
            pa = P[:, gi] - P[:, gj] + gh * P[:, H]     # P a   (margin row)
            pb = P[:, gi] - P[:, gj]                     # P b   (EPA row: no home term)
            d = x[gi] - x[gj]
            if has_e[g]:
                Ph = np.stack([pa, pb], axis=1)          # (T+1) x 2
                S2 = np.array([[pa[gi] - pa[gj] + gh * pa[H], pb[gi] - pb[gj] + gh * pb[H]],
                               [pa[gi] - pa[gj], pb[gi] - pb[gj]]]) + R2
                v = np.array([y_m[g] - (d + gh * x[H]), y_e[g] - d])
                K = Ph @ np.linalg.inv(S2)
                x += K @ v
                P -= K @ Ph.T
            else:
                s = pa[gi] - pa[gj] + gh * pa[H] + sm2
                x += pa * ((y_m[g] - (d + gh * x[H])) / s)
                P -= np.outer(pa, pa) / s
            P = 0.5 * (P + P.T)
        post[b0:b1] = x[i] - x[j] + h * x[H]
    return FilterResult(pm, pv, post, TEAMS, snaps, x.copy(), P.copy(), (pe_mean, pe_var, pe_cov))


def log_likelihood(gt: pd.DataFrame, res: FilterResult, cfg: KalmanConfig,
                   seasons: tuple[int, int] = TUNE_SEASONS) -> tuple[float, int]:
    """Sum of log N(margin; kf_margin, kf_sd^2 + sigma_m^2) over games in `seasons`, and the game count."""
    m = gt["season"].between(*seasons).to_numpy()
    var = res.pred_var[m] + cfg.sigma_m ** 2
    r = gt["margin"].to_numpy(float)[m] - res.pred_mean[m]
    return float(np.sum(-0.5 * (np.log(2 * np.pi * var) + r * r / var))), int(m.sum())


def joint_log_likelihood(gt: pd.DataFrame, res: FilterResult, cfg: KalmanConfig,
                         seasons: tuple[int, int] = TUNE_SEASONS) -> tuple[float, int]:
    """Sum of the pre-week predictive log density of BOTH measurements (margin and scaled EPA margin).

    Games without play-by-play contribute the margin term alone. This is the
    proper state-space likelihood: unlike the margin-only one, it pins down the
    EPA measurement's scale, noise, and correlation.
    """
    m = gt["season"].between(*seasons).to_numpy()
    nn = (~gt["neutral"].to_numpy(bool)).astype(float)[m]
    ym = gt["margin"].to_numpy(float)[m]
    e = gt["epa_margin"].to_numpy(float)[m]
    ye = cfg.scale * (e - cfg.epa_home * nn)
    mm, vm = res.pred_mean[m], res.pred_var[m] + cfg.sigma_m ** 2
    me, ve, c = (a[m] for a in res.epa_pred)
    ve = ve + cfg.sigma_e ** 2
    c = c + cfg.rho * cfg.sigma_m * cfg.sigma_e
    has = ~np.isnan(e)
    r1 = ym - mm
    ll = np.where(has, 0.0, -0.5 * (np.log(2 * np.pi * vm) + r1 * r1 / vm))
    r2 = np.where(has, ye - me, 0.0)
    det = vm * ve - c * c
    q = (ve * r1 * r1 - 2 * c * r1 * r2 + vm * r2 * r2) / det
    ll = ll + np.where(has, -0.5 * (2 * np.log(2 * np.pi) + np.log(np.where(has, det, 1.0)) + q), 0.0)
    # density of the observed EPA margin itself (y_e = scale * epa), so the scale cannot shrink the units away
    ll = ll + np.where(has, np.log(cfg.scale), 0.0)
    return float(ll.sum()), int(m.sum())


def win_prob(kf_margin, kf_sd, sigma_m: float) -> np.ndarray:
    """Standalone forecaster: P(home win) = Phi(kf_margin / sqrt(kf_sd^2 + sigma_m^2))."""
    return stats.norm.cdf(np.asarray(kf_margin, float) / np.sqrt(np.asarray(kf_sd, float) ** 2 + sigma_m ** 2))


# --------------------------------------------------------------------------- tuning (2000-2005 only)

_LO = {"q_week": 1e-4, "gamma": 0.05, "q_season": 1e-3, "sigma_m": 5.0, "sigma_e": 1.0, "rho": -0.95,
       "scale": 5.0, "q_hfa": 0.0}
_HI = {"q_week": 10.0, "gamma": 1.0, "q_season": 100.0, "sigma_m": 25.0, "sigma_e": 40.0, "rho": 0.98,
       "scale": 200.0, "q_hfa": 0.5}


START_POINTS = (  # multi-start: the likelihood has local optima
    dict(q_week=0.5, gamma=0.7, q_season=6.0, sigma_m=13.5, sigma_e=9.0, rho=0.5, scale=60.0, q_hfa=0.001),
    dict(q_week=0.1, gamma=0.8, q_season=2.0, sigma_m=13.0, sigma_e=6.0, rho=0.2, scale=30.0, q_hfa=0.0),
    dict(q_week=2.0, gamma=0.4, q_season=15.0, sigma_m=12.0, sigma_e=25.0, rho=0.8, scale=90.0, q_hfa=0.05),
    dict(q_week=0.3, gamma=0.65, q_season=8.0, sigma_m=13.5, sigma_e=10.0, rho=0.0, scale=45.0, q_hfa=0.001),
)


def tune(gt: pd.DataFrame, seasons: tuple[int, int] = TUNE_SEASONS, starts=START_POINTS) -> dict:
    """Maximum likelihood over `seasons` (default 2000-2005) of each game's pre-week predictive density.

    Only games through `seasons[1]` are filtered, so DEV and the holdout are
    never read. The objective is the joint density of the game's two
    measurements (margin and EPA margin, `joint_log_likelihood`). The
    margin-only likelihood (`log_likelihood`) is reported next to it: on its
    own it cannot pin down the EPA measurement's scale, noise and correlation
    (they trade off along a flat ridge, and different starts land in different
    places), while the joint one gives the same answer from every start.
    `epa_home` is fixed first from 1999-2005. Bounded L-BFGS-B from each of
    `starts`; the best joint likelihood wins.
    """
    gt = gt[gt["season"] <= seasons[1]].reset_index(drop=True)
    base = replace(KalmanConfig(), epa_home=epa_home_edge(gt, (BURN_IN_FIRST, seasons[1])))
    bounds = [(_LO[k], _HI[k]) for k in TUNED_KEYS]
    calls = {"n": 0}

    def nll(v, fn):
        calls["n"] += 1
        cfg = replace(base, **dict(zip(TUNED_KEYS, map(float, v))))
        return -fn(gt, run_filter(gt, cfg), cfg, seasons)[0]

    t0 = time.perf_counter()
    runs = []
    for st in starts:
        x0 = np.array([st[k] for k in TUNED_KEYS], float)
        r = optimize.minimize(nll, x0, args=(joint_log_likelihood,), method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": 400, "eps": 1e-4, "ftol": 1e-11})
        cfg = replace(base, **dict(zip(TUNED_KEYS, map(float, r.x))))
        runs.append({"config": cfg, "joint": -float(r.fun), "converged": bool(r.success),
                     "margin": log_likelihood(gt, run_filter(gt, cfg), cfg, seasons)[0]})
    best = max(runs, key=lambda d: d["joint"])
    cfg = best["config"]
    # the margin-only maximum from the same starts, for reference (flat; not used)
    m_only = []
    for st in starts:
        x0 = np.array([st[k] for k in TUNED_KEYS], float)
        r = optimize.minimize(nll, x0, args=(log_likelihood,), method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": 300, "eps": 1e-4, "ftol": 1e-10})
        m_only.append({"loglik": -float(r.fun), **{k: float(v) for k, v in zip(TUNED_KEYS, r.x)}})
    res = run_filter(gt, cfg)
    ll, n = log_likelihood(gt, res, cfg, seasons)
    msk = gt["season"].between(*seasons).to_numpy()
    y = gt["margin"].to_numpy(float)[msk]
    sd0 = float(np.std(y))
    ll_const = float(np.sum(stats.norm.logpdf(y, loc=float(np.mean(y)), scale=sd0)))
    resid = y - res.pred_mean[msk]
    vals = [getattr(cfg, k) for k in TUNED_KEYS]
    return {"config": cfg, "objective": "joint", "joint_loglik": best["joint"], "loglik": ll, "n_games": n,
            "loglik_per_game": ll / n, "loglik_constant_per_game": ll_const / n,
            "rmse": float(np.sqrt(np.mean(resid ** 2))), "rmse_constant": sd0,
            "starts": [{"joint": d["joint"], "margin": d["margin"], "converged": d["converged"],
                        **{k: getattr(d["config"], k) for k in TUNED_KEYS}} for d in runs],
            "margin_only_optima": m_only, "evaluations": calls["n"], "seconds": time.perf_counter() - t0,
            "at_bounds": [k for k, v in zip(TUNED_KEYS, vals) if math.isclose(v, _LO[k], abs_tol=1e-6)
                          or math.isclose(v, _HI[k], abs_tol=1e-6)]}


def profile(gt: pd.DataFrame, cfg: KalmanConfig, key: str, values, seasons: tuple[int, int] = TUNE_SEASONS
            ) -> pd.DataFrame:
    """Log likelihood per game over `seasons` with one knob varied and the rest held at `cfg`."""
    gt = gt[gt["season"] <= seasons[1]].reset_index(drop=True)
    rows = []
    for v in values:
        c = replace(cfg, **{key: float(v)})
        ll, n = log_likelihood(gt, run_filter(gt, c), c, seasons)
        rows.append({key: float(v), "loglik_per_game": ll / n})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- features

def build_features(pbp: pd.DataFrame, sched: pd.DataFrame, games: pd.DataFrame | None = None,
                   cfg: KalmanConfig = TUNED, positive_control: bool = False) -> pd.DataFrame:
    """kf_margin and kf_sd for each game in `games` (default: every played game of `sched`), indexed by game_id.

    Built only from games that kicked off before each game's as_of. Games that
    are scheduled but not played yet get the prediction from the latest state
    (the drift since then is not added). `positive_control=True` returns the
    margin predicted *after* the game's own week was absorbed: a deliberate
    leak that `leakage_check` must catch.
    """
    if "as_of" not in sched.columns:
        sched = asof_mod.add_asof(sched)
    games = sched if games is None else games
    gt = game_table(pbp, sched, cfg)
    res = run_filter(gt, cfg)
    mean = res.post_mean if positive_control else res.pred_mean
    known = pd.DataFrame({"kf_margin": mean, "kf_sd": np.sqrt(res.pred_var)}, index=gt["game_id"].to_numpy())
    missing = [g for g in games["game_id"] if g not in known.index]
    if missing:
        known = pd.concat([known, _unplayed(gt, cfg, sched, missing)])
    out = known.loc[games["game_id"].to_numpy()]
    out.index.name = "game_id"
    return out


def _unplayed(gt: pd.DataFrame, cfg: KalmanConfig, sched: pd.DataFrame, ids) -> pd.DataFrame:
    """Predictions for scheduled games with no result yet, from the state after every game before their as_of."""
    s = sched.drop_duplicates("game_id").set_index("game_id").loc[ids]
    code = {t: i for i, t in enumerate(TEAMS)}
    rows = {}
    for gid, r in s.iterrows():
        as_of = int(te._utc_ns(pd.Series([r["as_of"]]))[0])
        sub = gt[gt["kick_ns"] < as_of].reset_index(drop=True)
        fr = run_filter(sub, cfg)
        S = int(r["season"])
        a = np.zeros(len(TEAMS) + 1)
        a[code[franchise(r["home_team"], S)]] += 1.0
        a[code[franchise(r["away_team"], S)]] -= 1.0
        a[-1] = 0.0 if r.get("location", "Home") == "Neutral" else 1.0
        rows[gid] = (float(a @ fr.x_final), float(np.sqrt(a @ fr.P_final @ a)))
    return pd.DataFrame(rows, index=FEATURES).T
