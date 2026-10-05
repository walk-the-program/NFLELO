"""Player values: recency-weighted, shrunk box-score values, replacement levels, and the with/without ridge.

Unit (context/ml-m5-method.md, section 3): EPA per on-field play above a
replacement-level player at the position. A player's contribution to a lineup
is (expected share of team plays) x value, in EPA per team play, so values add
up across a lineup.

Box-score value, as of a week's `as_of`, from credit.player_games rows of games
that kicked off before it (any team; values follow players). For each credit
component c of his group, with w = 0.5 ** (weeks_ago / half_life) on the
game-week timeline (the same timeline as the team ratings):

    num_c = sum w * credit_c          (credit per team play, only where c is recorded)
    den_c = sum w * share             (weighted on-field games, only where c is recorded)
    value_c = (num_c + k_g * R_c) / (den_c + k_g)              # shrunk toward replacement
    v_box = sum_c (value_c - R_c) = sum_c (num_c - R_c * den_c) / (den_c + k_g)

k_g is the prior strength (pseudo full games) of his position group. R_c is
replacement level: the pooled per-play rate of players in their first
`early_games` full-game equivalents of career (players who debuted after the
first data season), from seasons before the week's season only. This is the
same idea as the QB prior in M3.

With/without layer (`ridge_values`): team-game opponent-adjusted EPA residuals
explained by the players on the field,

    y = EPA/play - (mu + D[opponent] + h * home)   (offense; defense uses O[opponent], sign flipped)
    y = c + sum_i share_i * e_i + noise,   e_i ~ N(b_i, 1 / lambda_i)

fit by weighted ridge (weights w * plays) on the team-games of the last three
seasons before `as_of`. b_i is the box value (negated for defenders, since
their rows measure EPA allowed), and lambda_i = lam * (1 + n_i / k_g) grows
with the player's box evidence n_i: where box scores are informative the
with/without data barely moves the value; where they are blind (linemen,
n_i = 0) the with/without data is all there is, shrunk hard toward 0
(replacement-level box value). The ratings (mu, D, O, h) are the M3 ones as of
each past game's week, so the residuals use only earlier games.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ..features.opponent_adjust import decay_weights, week_asof, week_ordinals
from . import positions as pos
from .credit import ALL_COMPONENTS, COMPONENTS

DEFAULT_K = {"QB": 4.0, "RB": 4.0, "WR": 4.0, "TE": 4.0, "OL": 4.0, "DL": 8.0, "LB": 8.0, "CB": 8.0, "S": 8.0,
             "K": 8.0}


@dataclass(frozen=True)
class ValueConfig:
    half_life: float = 16.0                       # weeks on the game-week timeline
    k: tuple = tuple(sorted(DEFAULT_K.items()))  # (group, pseudo full games) pairs
    early_games: float = 8.0                      # full-game equivalents that define "early career"
    lam: float = 2000.0                           # with/without ridge strength, in plays
    ridge_seasons: int = 3                        # seasons of team-games in each ridge fit
    min_weight: float = 1e-3                      # players with less weighted history are dropped

    @property
    def kd(self) -> dict:
        return dict(self.k)

    def with_k(self, **k) -> "ValueConfig":
        d = self.kd | k
        return ValueConfig(self.half_life, tuple(sorted(d.items())), self.early_games, self.lam,
                           self.ridge_seasons, self.min_weight)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["k"] = self.kd
        return d


# Chosen by scripts/ml_tune_players.py on 2001-2008 (no game outcomes; see context/ml.md, "M5 build notes"):
# half-life and k per group on next-4-games credited EPA, lambda on next-4 team-game EPA residuals (2006-2008).
TUNED = ValueConfig(half_life=32.0,
                    k=tuple(sorted({"QB": 4.0, "RB": 16.0, "WR": 16.0, "TE": 16.0, "OL": 4.0, "DL": 8.0, "LB": 8.0,
                                    "CB": 16.0, "S": 32.0, "K": 16.0}.items())),
                    lam=4000.0)


# --------------------------------------------------------------------------- replacement levels

def replacement_levels(pg: pd.DataFrame, seasons, early_games: float = 8.0) -> dict[int, pd.DataFrame]:
    """Season S -> (group x component) replacement rate, from early-career rows of seasons before S."""
    p = pg.sort_values("kick_ns", kind="stable")
    prior = p.groupby("gsis_id")["share"].cumsum() - p["share"]
    first_season = int(p["season"].min()) if len(p) else 0
    debut = p.groupby("gsis_id")["season"].transform("min")
    early = p[(prior < early_games) & (debut > first_season)]
    out = {}
    for S in sorted({int(s) for s in seasons}):
        e = early[early["season"] < S]
        if e.empty:
            e = p[p["season"] < S]
        rows = {}
        for g, comps in COMPONENTS.items():
            eg = e[e["group"] == g]
            rows[g] = {c: (float((eg[f"c_{c}"] * eg[f"a_{c}"]).sum() / (eg["share"] * eg[f"a_{c}"]).sum())
                           if (eg["share"] * eg[f"a_{c}"]).sum() > 0 else 0.0) for c in comps}
        out[S] = pd.DataFrame(rows).T.reindex(columns=list(ALL_COMPONENTS)).fillna(0.0)
    return out


# --------------------------------------------------------------------------- box values

@dataclass
class BoxSums:
    """Weighted sums at one week: per player, num and den for every component, plus n (weighted games)."""
    players: pd.Index
    group: np.ndarray
    num: np.ndarray   # (P, C)
    den: np.ndarray   # (P, C)
    n: np.ndarray     # (P,)

    def values(self, k: dict, R: pd.DataFrame) -> np.ndarray:
        """v_box per player (sum over his group's components of the shrunk value above replacement)."""
        v = np.zeros(len(self.players))
        for g, comps in COMPONENTS.items():
            m = self.group == g
            if not m.any() or not comps:
                continue
            for c in comps:
                j = ALL_COMPONENTS.index(c)
                r = R.at[g, c] if g in R.index else 0.0
                v[m] += (self.num[m, j] - r * self.den[m, j]) / (self.den[m, j] + k.get(g, 4.0))
        return v

    def rates(self, k: dict, R: pd.DataFrame) -> np.ndarray:
        """Shrunk per-play rate including replacement (what the tuning target is compared with), per component."""
        out = np.zeros_like(self.num)
        for g, comps in COMPONENTS.items():
            m = self.group == g
            for c in comps:
                j = ALL_COMPONENTS.index(c)
                r = R.at[g, c] if g in R.index else 0.0
                out[m, j] = (self.num[m, j] + k.get(g, 4.0) * r) / (self.den[m, j] + k.get(g, 4.0))
        return out


class BoxValues:
    """Precomputed arrays for computing box sums at many (season, week) keys quickly."""

    def __init__(self, pg: pd.DataFrame, sched: pd.DataFrame):
        p = pg.sort_values("kick_ns", kind="stable").reset_index(drop=True)
        self.pg = p
        self.kick = p["kick_ns"].to_numpy()
        self.ord = p["ord"].to_numpy(float)
        self.season = p["season"].to_numpy(int)
        self.ids = p["gsis_id"].to_numpy(object)
        self.group = p["group"].to_numpy(object)
        self.share = p["share"].to_numpy(float)
        self.C = np.column_stack([p[f"c_{c}"].to_numpy(float) * p[f"a_{c}"].to_numpy(float) for c in ALL_COMPONENTS])
        self.A = np.column_stack([p[f"a_{c}"].to_numpy(float) for c in ALL_COMPONENTS])
        self.asofs, self.ords = week_asof(sched), week_ordinals(sched)

    def sums(self, S: int, W: int, half_life: float, min_weight: float = 1e-3) -> BoxSums:
        end = np.searchsorted(self.kick, self.asofs[(S, W)], side="left")
        w = decay_weights(int(self.ords[(S, W)]), self.ord[:end], S, self.season[:end], half_life, 1.0)
        keep = w * self.share[:end] > 0
        keep &= w > min_weight * 1e-3
        idx = np.flatnonzero(keep)
        players, code = np.unique(self.ids[idx], return_inverse=True)
        P = len(players)
        ws = (w[idx] * self.share[idx])
        n = np.bincount(code, weights=ws, minlength=P)
        num = np.column_stack([np.bincount(code, weights=w[idx] * self.C[idx, j], minlength=P)
                               for j in range(len(ALL_COMPONENTS))])
        den = np.column_stack([np.bincount(code, weights=ws * self.A[idx, j], minlength=P)
                               for j in range(len(ALL_COMPONENTS))])
        # group = the group of the player's most recent row
        last = np.zeros(P, dtype=int)
        last[code] = idx  # idx is increasing, so the last assignment is the latest row
        grp = self.group[last]
        sel = n > min_weight
        return BoxSums(pd.Index(players[sel]), grp[sel], num[sel], den[sel], n[sel])


def box_values(bv: BoxValues, keys, cfg: ValueConfig, R: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Long frame (season, week, gsis_id, group, n, v_box) at each (season, week) key."""
    out = []
    for S, W in sorted(set((int(a), int(b)) for a, b in keys)):
        bs = bv.sums(S, W, cfg.half_life, cfg.min_weight)
        v = bs.values(cfg.kd, R[S])
        out.append(pd.DataFrame({"season": S, "week": W, "gsis_id": bs.players, "group": bs.group,
                                 "n": bs.n, "v_box": v}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["season", "week", "gsis_id", "group", "n", "v_box"])


# --------------------------------------------------------------------------- with/without ridge

GROUP_CODE = {g: i + 1 for i, g in enumerate(pos.GROUPS)}
CODE_GROUP = {v: k for k, v in GROUP_CODE.items()}

def residual_rows(tab: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Team-game rows (offense and defense) with opponent-adjusted EPA residuals per play.

    `tab` is opponent_adjust.rating_table (one row per game and offense, REG and
    POST); `ratings` holds the M3 ratings (kind "all") at each (season, week),
    fit on games before that week. Returns game_id, season, week, kick_ns, ord,
    team, side ("off"/"def"), plays, y.
    """
    r = ratings[ratings["kind"] == "all"].set_index(["season", "week", "team"])
    mu_h = ratings[ratings["kind"] == "all"].drop_duplicates(["season", "week"]).set_index(["season", "week"])[["mu", "h"]]
    t = tab[tab["n_all"] > 0].copy()
    key_o = pd.MultiIndex.from_arrays([t["season"], t["week"], t["off"]])
    key_d = pd.MultiIndex.from_arrays([t["season"], t["week"], t["def"]])
    mh = mu_h.reindex(pd.MultiIndex.from_arrays([t["season"], t["week"]]))
    mu, h = mh["mu"].to_numpy(float), mh["h"].to_numpy(float)
    O_off = r["off"].reindex(key_o).fillna(0.0).to_numpy()
    D_def = r["def"].reindex(key_d).fillna(0.0).to_numpy()
    rate = (t["epa_all"] / t["n_all"]).to_numpy(float)
    home = t["home"].to_numpy(float)
    base = pd.DataFrame({"game_id": t["game_id"].to_numpy(), "season": t["season"].to_numpy(int),
                         "week": t["week"].to_numpy(int), "kick_ns": t["kick_ns"].to_numpy(), "ord": t["ord"].to_numpy(),
                         "plays": t["n_all"].to_numpy(float)})
    off = base.assign(team=t["off"].to_numpy(object), side="off", y=rate - (mu + D_def + h * home))
    de = base.assign(team=t["def"].to_numpy(object), side="def", y=rate - (mu + O_off + h * home))
    out = pd.concat([off, de], ignore_index=True)
    out = out[np.isfinite(out["y"])]
    return out.sort_values("kick_ns", kind="stable").reset_index(drop=True)


class RidgeData:
    """Team-game residual rows and their lineups (shares) as one sparse matrix per side."""

    def __init__(self, rows: pd.DataFrame, onfield: pd.DataFrame, sched: pd.DataFrame):
        self.asofs, self.ords = week_asof(sched), week_ordinals(sched)
        self.sides = {}
        for side in ("off", "def"):
            rr = rows[rows["side"] == side].reset_index(drop=True)
            of = onfield[(onfield["side"] == side) & (onfield["group"] != "K") & (onfield["share"] > 0)]
            rid = pd.MultiIndex.from_frame(rr[["game_id", "team"]])
            ri = rid.get_indexer(pd.MultiIndex.from_frame(of[["game_id", "team"]]))
            of = of[ri >= 0]
            ri = ri[ri >= 0]
            players, pc = np.unique(of["gsis_id"].to_numpy(object), return_inverse=True)
            X = sparse.csr_matrix((of["share"].to_numpy(float), (ri, pc)), shape=(len(rr), len(players)))
            # Group codes as a parallel matrix, so a fit reads each player's group from rows before as_of only.
            gcode = np.array([GROUP_CODE.get(g, 0) for g in of["group"]], float)
            G = sparse.csr_matrix((gcode, (ri, pc)), shape=(len(rr), len(players)))
            self.sides[side] = {"rows": rr, "X": X, "G": G, "players": pd.Index(players)}

    def fit(self, S: int, W: int, prior: pd.DataFrame, cfg: ValueConfig, side: str) -> pd.DataFrame:
        """Ridge fit at (S, W) for one side. `prior` has gsis_id, group, n, v_box at (S, W). Returns gsis_id, e."""
        d = self.sides[side]
        rr, X = d["rows"], d["X"]
        end = np.searchsorted(rr["kick_ns"].to_numpy(), self.asofs[(S, W)], side="left")
        seas = rr["season"].to_numpy(int)[:end]
        m = seas >= S - cfg.ridge_seasons + 1
        if not m.any():
            return pd.DataFrame(columns=["gsis_id", "e"])
        idx = np.flatnonzero(m)
        w = decay_weights(int(self.ords[(S, W)]), rr["ord"].to_numpy(float)[idx], S, seas[idx], cfg.half_life, 1.0)
        om = w * rr["plays"].to_numpy(float)[idx]
        Xs = X[idx]
        cols = np.unique(Xs.indices)
        Xs = Xs[:, cols]
        players = d["players"][cols]
        # each player's group = his group in his latest row inside the window (rows are in kickoff order)
        Gs = d["G"][idx][:, cols].tocsc()
        group = np.empty(len(cols), dtype=object)
        for j in range(len(cols)):
            col = Gs.data[Gs.indptr[j]:Gs.indptr[j + 1]]
            rws = Gs.indices[Gs.indptr[j]:Gs.indptr[j + 1]]
            group[j] = CODE_GROUP.get(int(col[np.argmax(rws)]), "OL") if len(col) else "OL"
        pr = prior.set_index("gsis_id").reindex(players)
        sign = 1.0 if side == "off" else -1.0
        b = sign * pr["v_box"].fillna(0.0).to_numpy(float)
        n = pr["n"].fillna(0.0).to_numpy(float)
        kk = np.array([cfg.kd.get(g, 4.0) for g in group], float)
        lam = cfg.lam * (1.0 + n / kk)
        y = rr["y"].to_numpy(float)[idx]
        r = y - Xs @ b
        Z = sparse.hstack([sparse.csr_matrix(np.ones((len(idx), 1))), Xs]).tocsc()
        A = (Z.T @ Z.multiply(om[:, None])).tocsc()
        pen = np.concatenate([[1e-9 * om.sum()], lam])
        A = A + sparse.diags(pen)
        rhs = Z.T @ (om * r)
        beta = spsolve(A, rhs)
        e = b + beta[1:]
        out = pd.DataFrame({"gsis_id": players, "e": e, "group": group})
        out.attrs["intercept"] = float(beta[0])
        return out


def ridge_values(rd: RidgeData, box: pd.DataFrame, keys, cfg: ValueConfig) -> pd.DataFrame:
    """With/without values at each key: season, week, gsis_id, v_ww (positive = good for his team).

    Players not in the ridge window keep their box value; kickers always do.
    """
    out = []
    for S, W in sorted(set((int(a), int(b)) for a, b in keys)):
        pr = box[(box["season"] == S) & (box["week"] == W)]
        vals = pr.set_index("gsis_id")["v_box"].copy()
        for side, sign in (("off", 1.0), ("def", -1.0)):
            f = rd.fit(S, W, pr, cfg, side)
            if len(f):
                f = f.drop_duplicates("gsis_id")
                ser = pd.Series(sign * f["e"].to_numpy(float), index=f["gsis_id"].to_numpy())
                vals = pd.concat([vals[~vals.index.isin(ser.index)], ser])
        out.append(pd.DataFrame({"season": S, "week": W, "gsis_id": vals.index.to_numpy(), "v_ww": vals.to_numpy()}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["season", "week", "gsis_id", "v_ww"])
