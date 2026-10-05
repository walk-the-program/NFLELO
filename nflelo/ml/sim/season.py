"""Monte Carlo playoff odds (M4 Phase 2, context/ml-m4-method.md section 5).

For each of `n_sims` simulated seasons (20,000 by default, seeded):

1. Every team draws one strength shock, Normal(0, tau_rest) points, held for
   the rest of that simulated season (decision M4-D3). Ratings are estimates
   and teams change; without the shock, favorites lock up spots too early.
2. Every remaining REG game is played from the margin model: mean = the
   model's spread plus (home shock - away shock), then an integer margin from
   the chosen shape. A margin of 0 is a tie (half a win each).
3. The standings are seeded with the NFL tiebreakers (`tiebreak`): 7 seeds per
   conference from 2020, 6 before.
4. The bracket is played with the same model: the higher seed hosts and gets
   the home edge the model learned in training (intercept plus Elo's home-field
   term); the Super Bowl is neutral (no home edge). A playoff tie is settled by
   a coin flip, standing in for overtime.

Team strength. The margin model's mean splits into a home edge plus a
per-team strength:

    mean(h, a) = b0 + b1 * c * HFA + s_h - s_a,   s_t = b1 * c * Elo_t + b2 * adj_t + b3 * qb_delta_t

with c = ln(10) / 400, adj_t = offense minus defense from the opponent-adjusted
EPA ratings, and qb_delta_t the QB term of the team's current listed starter
(the starter nflverse lists for its next game, else its last starter). That
starter carries forward to every future game: a known long-term injury is
included, a return from injury isn't anticipated (spec section 5).

Remaining REG game means use each game's own A4s features, exactly as the
ledger's spread, except that the QB term uses the current listed starter of
both teams. Playoff matchups, which have no features yet, use the strengths.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features import context
from ..models import margin as mm
from . import tiebreak

OUTPUT_COLUMNS = ["playoffs", "division", "seed1", "reach_sb", "win_sb", "wins_mean", "wins_p10", "wins_p90"]
DEFAULT_SIMS = 20_000
TAU_REST = 4.5       # season-long strength shock SD in points, tuned on DEV 2006-2019 (scripts/ml_m4.py tune; one-SE rule)


@dataclass
class SimInputs:
    season: int
    static: tiebreak.SeasonStatic          # every REG game of the season
    res_done: np.ndarray                    # [G] home result for completed games (1 / .5 / 0), NaN otherwise
    margin_done: np.ndarray                 # [G] home margin for completed games, NaN otherwise
    mu: np.ndarray                          # [G] model mean for remaining games, NaN for completed ones
    strength: np.ndarray                    # [T] team strengths s_t (points)
    home_edge: float                        # playoff home edge (points)
    sigma: float
    shape: str = "normal"
    kn: mm.KeyNumbers | None = None
    cancelled: np.ndarray | None = None     # [G] games that will never be played

    @property
    def remaining(self) -> np.ndarray:
        rem = np.isnan(self.res_done)
        if self.cancelled is not None:
            rem &= ~self.cancelled
        return rem


# --------------------------------------------------------------------------- inputs

def team_strengths(fit: mm.MarginFit, elo_ratings: dict, adj: dict, qb_delta: dict, teams: list[str]) -> np.ndarray:
    b = fit.coef
    return np.array([b["elo_logit"] * context.LN10_400 * elo_ratings[t] + b["adj_epa_margin"] * adj.get(t, 0.0)
                     + b["qb_delta_diff"] * qb_delta.get(t, 0.0) for t in teams], float)


def current_qb_delta(feats: pd.DataFrame) -> dict:
    """Each team's QB term from its next game (the listed starter when nflverse lists one)."""
    rows = []
    for gid, r in feats.iterrows():
        rows.append((r["home_team"], int(r["game_week"]), gid, float(r["qb_delta_home"])))
        rows.append((r["away_team"], int(r["game_week"]), gid, float(r["qb_delta_away"])))
    t = pd.DataFrame(rows, columns=["team", "week", "game_id", "d"]).sort_values(["team", "week", "game_id"])
    return t.groupby("team")["d"].first().to_dict()


def build_inputs(season_games: pd.DataFrame, feats: pd.DataFrame, elo: dict, adj: dict, fit: mm.MarginFit,
                 shape: str, kn: mm.KeyNumbers | None) -> SimInputs:
    """Simulation inputs from one season's REG games (scores where played), the live features of the unplayed
    games (`live_features.build`), the Elo state (`live_features.elo_state`), and team adj = off - def ratings."""
    g = season_games.sort_values("game_id").reset_index(drop=True)
    season = int(g["season"].iloc[0])
    st = tiebreak.SeasonStatic(g, season)
    hs, as_ = g["home_score"].to_numpy(float), g["away_score"].to_numpy(float)
    res = np.where(np.isnan(hs), np.nan, np.where(hs > as_, 1.0, np.where(hs < as_, 0.0, 0.5)))
    todo = np.isnan(res)
    missing = sorted(set(g.loc[todo, "game_id"]) - set(feats.index))
    if missing:
        raise ValueError(f"{len(missing)} unplayed games have no features, e.g. {missing[:3]}")
    qd = current_qb_delta(feats)
    f = feats.loc[g.loc[todo, "game_id"]].copy()
    f["qb_delta_diff"] = [qd[h] - qd[a] for h, a in zip(f["home_team"], f["away_team"])]
    mu = np.full(len(g), np.nan)
    mu[todo] = fit.mean(f[fit.features].to_numpy(float))
    strength = team_strengths(fit, elo["ratings"], adj, qd, st.teams)
    edge = fit.intercept + fit.coef["elo_logit"] * context.LN10_400 * float(elo["hfa_now"])
    return SimInputs(season, st, res, np.where(todo, np.nan, hs - as_), mu, strength, float(edge), fit.sigma, shape, kn)


# --------------------------------------------------------------------------- one batch of seasons

def _play(inp: SimInputs, h: np.ndarray, a: np.ndarray, shocks: np.ndarray, neutral: bool,
          rng: np.random.Generator) -> np.ndarray:
    """Winners of one playoff game per sim (team indices). h, a: [N] team indices."""
    n = np.arange(len(h))
    mean = inp.strength[h] - inp.strength[a] + shocks[n, h] - shocks[n, a] + (0.0 if neutral else inp.home_edge)
    k = mm.sample(inp.shape, mean, inp.sigma, rng, inp.kn).astype(float)
    tie = k == 0
    k[tie] = np.where(rng.uniform(size=int(tie.sum())) < 0.5, 1.0, -1.0)
    return np.where(k > 0, h, a)


def _bracket(inp: SimInputs, seeds: np.ndarray, shocks: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """seeds [N, 2, n] team indices. Returns (conference champions [N, 2], SB winner [N])."""
    N, _, n = seeds.shape
    byes = 2 if n == 6 else 1
    champs = np.zeros((N, 2), np.int64)
    rows = np.arange(N)
    for c in range(2):
        s = seeds[:, c, :]                                   # [N, n], column j = seed j+1
        alive = [np.full(N, j + 1) for j in range(byes)]     # seed numbers still alive
        for hi in range(byes + 1, (n + byes) // 2 + 1):      # wild card: hi hosts n + byes + 1 - hi
            lo = n + byes + 1 - hi
            w = _play(inp, s[:, hi - 1], s[:, lo - 1], shocks, False, rng)
            alive.append(np.where(w == s[:, hi - 1], hi, lo))
        r = np.sort(np.stack(alive, axis=1), axis=1)          # 4 seeds left, ascending
        team = lambda k: s[rows, r[:, k] - 1]                 # noqa: E731
        w1 = _play(inp, team(0), team(3), shocks, False, rng)  # #1 hosts the lowest remaining seed
        w2 = _play(inp, team(1), team(2), shocks, False, rng)
        sd = lambda t: np.argmax(s == t[:, None], axis=1)     # noqa: E731  seed index of a team
        h = np.where(sd(w1) < sd(w2), w1, w2)
        a = np.where(sd(w1) < sd(w2), w2, w1)
        champs[:, c] = _play(inp, h, a, shocks, False, rng)
    sb = _play(inp, champs[:, 0], champs[:, 1], shocks, True, rng)
    return champs, sb


def _seed_all(inp: SimInputs, res: np.ndarray, margin: np.ndarray, rng) -> tuple[np.ndarray, dict]:
    st = inp.static
    n = tiebreak.n_seeds(inp.season)
    stats = tiebreak.team_stats(st, res, margin)
    N = res.shape[0]
    seeds = np.zeros((N, 2, n), np.int64)
    for i in range(N):
        s = tiebreak.Standings(st, res[i], margin[i], {k: v[i] for k, v in stats.items()}, rng)
        for c, conf in enumerate(tiebreak.CONFS):
            seeds[i, c] = s.seed_conference(conf, n)
    return seeds, stats


def simulate(inp: SimInputs, n_sims: int = DEFAULT_SIMS, tau: float = 0.0, seed: int = 20261004,
             chunk: int = 5000, timings: dict | None = None) -> pd.DataFrame:
    """Per-team probabilities and win ranges over `n_sims` simulated seasons. Deterministic for a given seed."""
    st = inp.static
    T, rem = st.T, inp.remaining
    hr, ar, mu = st.home[rem], st.away[rem], inp.mu[rem]
    if np.isnan(mu).any():
        raise ValueError("a remaining game has no model mean")
    counts = {k: np.zeros(T) for k in ("playoffs", "division", "seed1", "reach_sb", "win_sb")}
    wins_all = []
    t_draw = t_seed = t_bracket = 0.0
    base = np.random.SeedSequence(seed)
    for b, child in enumerate(base.spawn((n_sims + chunk - 1) // chunk)):
        N = min(chunk, n_sims - b * chunk)
        rng = np.random.default_rng(child)
        t0 = time.perf_counter()
        shocks = tau * rng.standard_normal((N, T))
        M = mu[None, :] + shocks[:, hr] - shocks[:, ar]
        k = mm.sample(inp.shape, M, inp.sigma, rng, inp.kn).astype(float) if len(mu) else np.zeros((N, 0))
        res = np.tile(inp.res_done, (N, 1))
        mar = np.tile(inp.margin_done, (N, 1))
        res[:, rem] = np.where(k > 0, 1.0, np.where(k < 0, 0.0, 0.5))
        mar[:, rem] = k
        t1 = time.perf_counter()
        seeds, stats = _seed_all(inp, res, mar, rng)
        t2 = time.perf_counter()
        champs, sb = _bracket(inp, seeds, shocks, rng)
        t3 = time.perf_counter()
        t_draw, t_seed, t_bracket = t_draw + t1 - t0, t_seed + t2 - t1, t_bracket + t3 - t2
        for c in range(2):
            np.add.at(counts["playoffs"], seeds[:, c, :].ravel(), 1)
            np.add.at(counts["division"], seeds[:, c, :4].ravel(), 1)
            np.add.at(counts["seed1"], seeds[:, c, 0], 1)
            np.add.at(counts["reach_sb"], champs[:, c], 1)
        np.add.at(counts["win_sb"], sb, 1)
        wins_all.append(stats["wins"])
    wins = np.vstack(wins_all)
    out = pd.DataFrame({k: v / n_sims for k, v in counts.items()}, index=pd.Index(st.teams, name="team"))
    out["wins_mean"] = wins.mean(axis=0)
    out["wins_p10"], out["wins_p90"] = np.percentile(wins, [10, 90], axis=0)
    if timings is not None:
        timings.update({"draw_s": t_draw, "seed_s": t_seed, "bracket_s": t_bracket, "n_sims": n_sims,
                        "remaining_games": int(rem.sum())})
    return out[OUTPUT_COLUMNS]
