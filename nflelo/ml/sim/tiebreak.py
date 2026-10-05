"""NFL standings tiebreakers and playoff seeding (M4 Phase 2, context/ml-m4-method.md section 5).

The procedure follows the NFL's published rules (https://www.nfl.com/standings/tie-breaking-procedures,
read 2026-10-04). Tie games count as half a win and half a loss.

Division ties
    Two clubs: head-to-head; division record; common games; conference record;
    strength of victory; strength of schedule; combined ranking (conference) in
    points scored and allowed; combined ranking (all teams); net points in
    common games; net points in all games; net touchdowns; coin toss.
    Three or more: the same list, with head-to-head over games among the clubs.
    If two clubs remain after a step eliminates others, restart at step 1 of the
    two-club list; if three or more remain, restart at step 1 of the three-club list.

Wild-card ties (clubs from different divisions)
    Step 0 (three or more clubs): keep only the highest-ranked club of each
    division, using the division ranking computed once (it stays fixed).
    Two clubs: head-to-head if they played; conference record; common games
    (minimum four); strength of victory; strength of schedule; combined ranking
    (conference); combined ranking (all); net points in conference games; net
    points in all games; net touchdowns; coin toss.
    Three or more: head-to-head sweep (only if one club beat each of the others,
    or lost to each of the others); then the two-club list from conference
    record on. Two left: restart at two-club step 1; three or more left: restart
    at the sweep. Clubs from one division use the division tiebreaker.

Only one club advances per application: once a club is selected, the others
start over. Division winners are seeded 1-4 with the wild-card tiebreakers;
the wild cards are picked one at a time, repeating step 0 each time.

Points. With actual scores (`hpts`/`apts`), the combined-ranking steps are
exact (tied ranks share the better rank, as the NFL states). In simulations
only the margin is known, so both combined-ranking steps are approximated by
the rank of net points (conference teams, then all teams); net-points steps are
exact either way. Net touchdowns are never available: that step is skipped and
the coin toss follows. Every decision can be logged with the step that decided
it (`Standings.log`), so the historical check can name the step behind each seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...meta import conference as team_conference
from ...meta import division as team_division
from ...teams import franchise

EPS = 1e-9
CONFS = ("AFC", "NFC")


def n_seeds(season: int) -> int:
    """Playoff clubs per conference: 7 from 2020, 6 from the 2002 realignment through 2019."""
    if season < 2002:
        raise ValueError("the tiebreakers here cover the 2002+ league of 8 four-team divisions")
    return 7 if season >= 2020 else 6


# --------------------------------------------------------------------------- the schedule's structure

class SeasonStatic:
    """Everything about a season's REG schedule that doesn't depend on results: teams, divisions, games."""

    def __init__(self, games: pd.DataFrame, season: int):
        g = games[games["season"] == season] if "season" in games.columns else games
        self.season = int(season)
        home = [franchise(c, self.season) for c in g["home_team"]]
        away = [franchise(c, self.season) for c in g["away_team"]]
        self.teams = sorted(set(home) | set(away))
        if len(self.teams) != 32:
            raise ValueError(f"{season}: expected 32 teams, got {len(self.teams)}")
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.T = len(self.teams)
        self.game_ids = g["game_id"].astype(str).to_numpy() if "game_id" in g.columns else np.arange(len(g)).astype(str)
        self.home = np.array([self.idx[t] for t in home], dtype=np.int64)
        self.away = np.array([self.idx[t] for t in away], dtype=np.int64)
        self.G = len(self.home)
        self.conf = np.array([team_conference(t) for t in self.teams])
        self.div = np.array([team_division(t) for t in self.teams])
        self.conf_game = self.conf[self.home] == self.conf[self.away]
        self.div_game = self.div[self.home] == self.div[self.away]
        self.divisions = {d: [i for i in range(self.T) if self.div[i] == d] for d in sorted(set(self.div))}
        self.conf_teams = {c: [i for i in range(self.T) if self.conf[i] == c] for c in CONFS}
        # incidence: +1 where the team is home, -1 where away (for margins); and plain membership
        self.H = np.zeros((self.G, self.T))
        self.A = np.zeros((self.G, self.T))
        self.H[np.arange(self.G), self.home] = 1.0
        self.A[np.arange(self.G), self.away] = 1.0
        self.opp_sets = [frozenset(self.away[self.home == t]) | frozenset(self.home[self.away == t])
                         for t in range(self.T)]
        self._common: dict = {}

    def common_games(self, group: tuple[int, ...]) -> list[tuple[np.ndarray, np.ndarray]]:
        """For each club in `group`: (game indices vs the group's common opponents, +1 home / -1 away)."""
        key = tuple(sorted(group))
        if key not in self._common:
            common = frozenset.intersection(*[self.opp_sets[t] for t in key]) - frozenset(key)
            cm = np.isin(np.arange(self.T), list(common))
            res = []
            for t in key:
                gi = np.flatnonzero(((self.home == t) & cm[self.away]) | ((self.away == t) & cm[self.home]))
                res.append((gi, np.where(self.home[gi] == t, 1.0, -1.0)))
            self._common[key] = dict(zip(key, res))
        d = self._common[key]
        return [d[t] for t in group]


# --------------------------------------------------------------------------- vectorized team stats

def team_stats(st: SeasonStatic, res: np.ndarray, margin: np.ndarray, hpts=None, apts=None) -> dict:
    """Per-team stats for N result sets at once. `res` is [N, G] home results (1, .5, 0; NaN = not played,
    e.g. the cancelled 2022 BUF-CIN game), `margin` [N, G] home minus away. Returns [N, T] arrays."""
    res = np.atleast_2d(np.asarray(res, float))
    margin = np.atleast_2d(np.asarray(margin, float))
    played = ~np.isnan(res)
    r = np.where(played, res, 0.0)
    m = np.where(played, margin, 0.0)
    P = played.astype(float)
    H, A = st.H, st.A
    wins = r @ H + (P - r) @ A                     # ties are half wins
    gp = P @ (H + A)
    pct = np.divide(wins, gp, out=np.zeros_like(wins), where=gp > 0)

    def sub(mask):
        rr, pp = r * mask, P * mask
        w = rr @ H + (pp - rr) @ A
        n = pp @ (H + A)
        return np.divide(w, n, out=np.zeros_like(w), where=n > 0), n

    div_pct, div_n = sub(st.div_game[None, :])
    conf_pct, conf_n = sub(st.conf_game[None, :])
    # strength of schedule: combined record of every opponent, once per game played
    opp_w = wins[:, st.away] * P, wins[:, st.home] * P     # [N, G]: opponent's wins for home / away club
    opp_n = gp[:, st.away] * P, gp[:, st.home] * P
    sos_num = opp_w[0] @ H + opp_w[1] @ A
    sos_den = opp_n[0] @ H + opp_n[1] @ A
    sos = np.divide(sos_num, sos_den, out=np.zeros_like(sos_num), where=sos_den > 0)
    # strength of victory: combined record of the opponents beaten (a tie is not a victory)
    hw, aw = (r == 1.0) & played, (r == 0.0) & played
    sov_num = (opp_w[0] * hw) @ H + (opp_w[1] * aw) @ A
    sov_den = (opp_n[0] * hw) @ H + (opp_n[1] * aw) @ A
    sov = np.divide(sov_num, sov_den, out=np.zeros_like(sov_num), where=sov_den > 0)
    net = m @ H - m @ A
    net_conf = (m * st.conf_game) @ H - (m * st.conf_game) @ A
    out = {"wins": wins, "gp": gp, "pct": pct, "div_pct": div_pct, "conf_pct": conf_pct, "sos": sos, "sov": sov,
           "net": net, "net_conf": net_conf}
    if hpts is not None:
        hp = np.where(played, np.atleast_2d(np.asarray(hpts, float)), 0.0)
        ap = np.where(played, np.atleast_2d(np.asarray(apts, float)), 0.0)
        out["pf"] = hp @ H + ap @ A
        out["pa"] = ap @ H + hp @ A
    return out


def _min_rank(values: np.ndarray, higher_better: bool) -> np.ndarray:
    """Rank 1 = best; tied values share the better rank ('min' method)."""
    v = -values if higher_better else values
    order = np.sort(v)
    return np.searchsorted(order, v, side="left") + 1


# --------------------------------------------------------------------------- one result set

@dataclass
class Standings:
    """One season's results (actual or simulated) and the tiebreaking logic on top of them."""
    st: SeasonStatic
    res: np.ndarray                     # [G] home results 1 / .5 / 0, NaN = not played
    margin: np.ndarray                  # [G]
    stats: dict                         # one row of team_stats (each [T])
    rng: np.random.Generator
    log: list | None = None             # append (context, group, step, chosen) when given a list
    _cache: dict = field(default_factory=dict)

    @classmethod
    def from_results(cls, st: SeasonStatic, res, margin, hpts=None, apts=None, rng=None, log=None) -> "Standings":
        s = team_stats(st, res, margin, hpts, apts)
        return cls(st, np.asarray(res, float), np.asarray(margin, float), {k: v[0] for k, v in s.items()},
                   rng or np.random.default_rng(0), log)

    # ---- small helpers
    def pct(self, t: int) -> float:
        return float(self.stats["pct"][t])

    def _record(self, gi: np.ndarray, side: np.ndarray) -> tuple[float, float]:
        r = self.res[gi]
        ok = ~np.isnan(r)
        pts = np.where(side[ok] > 0, r[ok], 1.0 - r[ok])
        return float(pts.sum()), float(ok.sum())

    def _h2h_pct(self, group) -> dict | None:
        """W-L-T percentage in games among the group (None if any club played none of them)."""
        gs = np.array(group)
        m = np.isin(self.st.home, gs) & np.isin(self.st.away, gs) & ~np.isnan(self.res)
        gi = np.flatnonzero(m)
        out = {}
        for t in group:
            mine = gi[(self.st.home[gi] == t) | (self.st.away[gi] == t)]
            if not len(mine):
                return None
            w, n = self._record(mine, np.where(self.st.home[mine] == t, 1.0, -1.0))
            out[t] = w / n
        return out

    def _pair(self, t: int, u: int) -> tuple[float, float]:
        m = (((self.st.home == t) & (self.st.away == u)) | ((self.st.home == u) & (self.st.away == t))) & ~np.isnan(self.res)
        gi = np.flatnonzero(m)
        return self._record(gi, np.where(self.st.home[gi] == t, 1.0, -1.0)) if len(gi) else (0.0, 0.0)

    def _ranking(self, key: str, scope: str) -> np.ndarray:
        """Combined ranking (lower is better) over conference teams or all teams; exact with points, net points otherwise."""
        if (key, scope) not in self._cache:
            T = self.st.T
            comb = np.full(T, np.nan)
            pools = [self.st.conf_teams[c] for c in CONFS] if scope == "conf" else [list(range(T))]
            for pool in pools:
                pool = np.array(pool)
                if "pf" in self.stats:
                    comb[pool] = (_min_rank(self.stats["pf"][pool], True) + _min_rank(self.stats["pa"][pool], False))
                else:
                    comb[pool] = _min_rank(self.stats["net"][pool], True)
            self._cache[(key, scope)] = comb
        return self._cache[(key, scope)]

    # ---- steps: each returns {team: value} (higher is better) or None when not applicable
    def s_h2h(self, group):
        return self._h2h_pct(group)

    def s_h2h_if_played(self, group):
        t, u = group
        w, n = self._pair(t, u)
        return None if n == 0 else {t: w / n, u: 1.0 - w / n}

    def s_sweep(self, group):
        beat_all, lost_all = [], []
        for t in group:
            recs = [self._pair(t, u) for u in group if u != t]
            if all(n > 0 and w == n for w, n in recs):
                beat_all.append(t)
            if all(n > 0 and w == 0 for w, n in recs):
                lost_all.append(t)
        if beat_all:
            return {t: 1.0 if t in beat_all else 0.0 for t in group}
        if lost_all:
            return {t: 0.0 if t in lost_all else 1.0 for t in group}
        return None

    def s_div(self, group):
        return {t: float(self.stats["div_pct"][t]) for t in group}

    def s_conf(self, group):
        return {t: float(self.stats["conf_pct"][t]) for t in group}

    def _common(self, group, minimum: int):
        out = {}
        for t, (gi, side) in zip(group, self.st.common_games(tuple(group))):
            w, n = self._record(gi, side)
            if n < minimum or n == 0:
                return None
            out[t] = w / n
        return out

    def s_common(self, group):
        return self._common(group, 1)

    def s_common4(self, group):
        return self._common(group, 4)

    def s_sov(self, group):
        return {t: float(self.stats["sov"][t]) for t in group}

    def s_sos(self, group):
        return {t: float(self.stats["sos"][t]) for t in group}

    def s_rank_conf(self, group):
        r = self._ranking("comb", "conf")
        return {t: -float(r[t]) for t in group}

    def s_rank_all(self, group):
        r = self._ranking("comb", "all")
        return {t: -float(r[t]) for t in group}

    def s_net_common(self, group):
        out = {}
        for t, (gi, side) in zip(group, self.st.common_games(tuple(group))):
            mm = self.margin[gi]
            ok = ~np.isnan(self.res[gi])
            out[t] = float((mm[ok] * side[ok]).sum())
        return out

    def s_net_conf(self, group):
        return {t: float(self.stats["net_conf"][t]) for t in group}

    def s_net(self, group):
        return {t: float(self.stats["net"][t]) for t in group}

    # ---- the procedures
    DIV2 = ("h2h", "div", "common", "conf", "sov", "sos", "rank_conf", "rank_all", "net_common", "net")
    DIV3 = DIV2
    WC2 = ("h2h_if_played", "conf", "common4", "sov", "sos", "rank_conf", "rank_all", "net_conf", "net")
    WC3 = ("sweep", "conf", "common4", "sov", "sos", "rank_conf", "rank_all", "net_conf", "net")
    POINTS = {"rank_conf", "rank_all", "net_common", "net_conf", "net"}

    def _break(self, group: list[int], kind: str) -> int:
        """Select the one club that wins the tie among `group` (kind 'div' or 'wc')."""
        group = list(group)
        context = (kind, tuple(self.st.teams[t] for t in group))
        while len(group) > 1:
            steps = (self.DIV2 if len(group) == 2 else self.DIV3) if kind == "div" else \
                    (self.WC2 if len(group) == 2 else self.WC3)
            restarted = False
            for step in steps:
                vals = getattr(self, "s_" + step)(group)
                if vals is None:
                    continue
                best_v = max(vals.values())
                best = [t for t in group if vals[t] >= best_v - EPS]
                if len(best) == 1:
                    self._note(context, step, best[0])
                    return best[0]
                if len(best) < len(group):
                    group = best
                    restarted = True
                    break
            if not restarted:
                pick = int(self.rng.choice(group))           # net touchdowns are unavailable: coin toss
                self._note(context, "coin", pick)
                return pick
        return group[0]

    def _note(self, context, step, chosen):
        if self.log is not None:
            self.log.append({"kind": context[0], "group": context[1], "step": step,
                             "chosen": self.st.teams[chosen], "points_based": step in self.POINTS or step == "coin"})

    def _order(self, teams: list[int], pick) -> list[int]:
        """Order clubs by percentage; clubs on the same percentage are ordered by repeated `pick(group)`."""
        p = {t: round(self.pct(t), 9) for t in teams}
        out = []
        for v in sorted(set(p.values()), reverse=True):
            group = [t for t in teams if p[t] == v]
            while group:
                t = group[0] if len(group) == 1 else pick(group)
                out.append(t)
                group.remove(t)
        return out

    def division_order(self, div: str) -> list[int]:
        if div not in self._cache:
            self._cache[div] = self._order(self.st.divisions[div], lambda g: self._break(g, "div"))
        return self._cache[div]

    def _wc_pick(self, group: list[int]) -> int:
        """Wild-card style pick: step 0 (best-ranked club per division), then the wild-card tiebreaker."""
        best_per_div = {}
        for t in group:
            d = self.st.div[t]
            rank = self.division_order(d).index(t)
            if d not in best_per_div or rank < best_per_div[d][0]:
                best_per_div[d] = (rank, t)
        reduced = [t for t in group if best_per_div[self.st.div[t]][1] == t]
        if len(reduced) == 1:
            if len(group) > 1:
                self._note(("wc", tuple(self.st.teams[t] for t in group)), "division_rank", reduced[0])
            return reduced[0]
        return self._break(reduced, "wc")

    def seed_conference(self, conf: str, n: int) -> list[int]:
        """Team indices for seeds 1..n."""
        divs = sorted({self.st.div[t] for t in self.st.conf_teams[conf]})
        winners = [self.division_order(d)[0] for d in divs]
        seeds = self._order(winners, self._wc_pick)
        rest = [t for t in self.st.conf_teams[conf] if t not in winners]
        while len(seeds) < n:
            p = {t: round(self.pct(t), 9) for t in rest}
            top = max(p.values())
            group = [t for t in rest if p[t] == top]
            pick = group[0] if len(group) == 1 else self._wc_pick(group)
            seeds.append(pick)
            rest.remove(pick)
        return seeds

    def seeds(self, n: int | None = None) -> dict:
        n = n or n_seeds(self.st.season)
        return {c: self.seed_conference(c, n) for c in CONFS}


# --------------------------------------------------------------------------- actual seasons

def actual_standings(sched: pd.DataFrame, season: int, log: list | None = None, seed: int = 0) -> Standings:
    """Standings from a season's actual REG results (scores known; unplayed or cancelled games count as not played)."""
    g = sched[(sched["season"] == season) & (sched["game_type"] == "REG")].sort_values("game_id").reset_index(drop=True)
    st = SeasonStatic(g, season)
    hs, as_ = g["home_score"].to_numpy(float), g["away_score"].to_numpy(float)
    res = np.where(np.isnan(hs), np.nan, np.where(hs > as_, 1.0, np.where(hs < as_, 0.0, 0.5)))
    return Standings.from_results(st, res, hs - as_, hs, as_, np.random.default_rng(seed), log)


def actual_seeds(sched: pd.DataFrame, season: int, log: list | None = None) -> dict:
    """{conf: [team for seed 1..n]} computed by this module from the season's actual results."""
    s = actual_standings(sched, season, log)
    return {c: [s.st.teams[t] for t in v] for c, v in s.seeds().items()}
