"""Quarterback adjustment: QB value, a replacement-level prior, and qb_delta (context/ml-m3-method.md, section 5).

QB value, as of a week's `as_of`. Every dropback (pass, sack, or scramble:
nflfastR's `qb_dropback`, credited to `passer_id`, which includes scramblers)
from games that kicked off before `as_of`, over the QB's whole recorded career,
gets the rating system's weight

    w = 0.5 ** (weeks_ago / half_life) * rho ** (seasons ago)

(same timeline and knobs as `opponent_adjust`). With n = weighted dropbacks
and raw = weighted EPA per dropback, the value is shrunk toward a prior:

    qb_value = (n * raw + k * prior) / (n + k)

`prior` is replacement level for season S: the pooled EPA per dropback of
QBs' first 100 career dropbacks, using seasons before S only. QBs already
playing in the first season of the data (1999) are left out, since their
careers started earlier; if that leaves no one (S = 2000), every QB counts.

qb_delta for a team = qb_value(this week's starter) minus the weighted average
qb_value over the team's own dropbacks inside the rating window (this season
and last, same weights as the team ratings). That second term is the QB play
the team's EPA rating "remembers", so the usual starter gives a delta near 0
and a backup gives a negative one.

Who starts (decision M3-D1):
- `starter="actual"` (ladder A4): the schedule's `home_qb_id` / `away_qb_id`,
  the one fact treated as known before kickoff. Only the identity is used;
  his value comes only from earlier games.
- `starter="last"` (ladder A4b): the team's starter in its most recent game
  before `as_of`, i.e. what is known on Wednesday. This version obeys the
  strict as-of rule with no exception.

A starter with no recorded dropbacks gets the prior. A team with no dropbacks
in the window (an expansion team in week 1) or a missing starter ID gets
qb_delta = 0. Garbage time is excluded, as for the team ratings.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ...teams import franchise
from .. import asof as asof_mod
from . import opponent_adjust as oa
from . import team_efficiency as te

ROOKIE_DROPBACKS = 100
STARTERS = ("actual", "last")
FEATURES = ["qb_delta_home", "qb_delta_away"]


@dataclass(frozen=True)
class QBConfig:
    k: float = 200.0          # pseudo-dropbacks of prior belief
    half_life: float = 8.0    # weeks (set equal to the team ratings' half-life)
    rho: float = 0.5          # off-season discount per season (same as the team ratings)
    rookie_dropbacks: int = ROOKIE_DROPBACKS
    garbage_low: float = 0.05
    garbage_high: float = 0.95

    @classmethod
    def from_ratings(cls, r: oa.RatingConfig, k: float) -> "QBConfig":
        return cls(k=k, half_life=r.half_life, rho=r.rho, garbage_low=r.garbage_low, garbage_high=r.garbage_high)

    def to_dict(self) -> dict:
        return asdict(self)


# k chosen by scripts/ml_tune_ratings.py on 2000-2005 only (decision M3-D2), with the
# tuned half-life and rho of opponent_adjust.TUNED.
TUNED_K = 100.0


def tuned_config() -> QBConfig:
    return QBConfig.from_ratings(oa.TUNED, TUNED_K)


# --------------------------------------------------------------------------- tables

def dropback_plays(pbp: pd.DataFrame, sched: pd.DataFrame, cfg: QBConfig = QBConfig()) -> pd.DataFrame:
    """Play-level dropbacks: game_id, qb, team (franchise), season, kickoff, epa, in career order."""
    if "as_of" not in sched.columns:
        sched = asof_mod.add_asof(sched)
    plays = te.clean_plays(pbp, te.EfficiencyConfig(garbage_low=cfg.garbage_low, garbage_high=cfg.garbage_high))
    flag = plays["qb_dropback"] if "qb_dropback" in plays.columns else plays["pass"]
    qb_col = "passer_id" if "passer_id" in plays.columns else "passer_player_id"
    db = plays[(flag.fillna(0) == 1) & plays[qb_col].notna()]
    info = sched.drop_duplicates("game_id").set_index("game_id")
    out = pd.DataFrame({"game_id": db["game_id"].to_numpy(), "qb": db[qb_col].astype(str).to_numpy(),
                        "posteam": db["posteam"].to_numpy(), "epa": db["epa"].to_numpy(float),
                        "order": np.arange(len(db))})
    if "play_id" in db.columns:
        out["order"] = db["play_id"].to_numpy(float)
    if "cpoe" in db.columns:  # M3b C3a only; the M3 aggregation ignores it
        out["cpoe"] = pd.to_numeric(db["cpoe"], errors="coerce").to_numpy(float)
    out["season"] = out["game_id"].map(info["season"]).astype(int)
    out["week"] = out["game_id"].map(info["week"]).astype(int)
    out["kickoff"] = out["game_id"].map(info["kickoff"])
    if out["kickoff"].isna().any():
        raise ValueError("play-by-play contains games that are not in the schedule")
    out["team"] = [franchise(c, int(s)) for c, s in zip(out["posteam"], out["season"])]
    out["kick_ns"] = te._utc_ns(out["kickoff"])
    out = out.sort_values(["kick_ns", "game_id", "order"], kind="stable").reset_index(drop=True)
    out["career_n"] = out.groupby("qb").cumcount() + 1
    return out


def qb_game_table(db: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team, qb): dropbacks n and EPA sum, with season, week, timeline position, kickoff."""
    g = db.groupby(["game_id", "team", "qb"], sort=False).agg(
        n=("epa", "size"), epa=("epa", "sum"), season=("season", "first"), week=("week", "first"),
        kick_ns=("kick_ns", "first")).reset_index()
    ords = oa.week_ordinals(sched)
    g["ord"] = ords.reindex(pd.MultiIndex.from_frame(g[["season", "week"]])).to_numpy()
    return g.sort_values("kick_ns", kind="stable").reset_index(drop=True)


def replacement_priors(db: pd.DataFrame, seasons, rookie_dropbacks: int = ROOKIE_DROPBACKS) -> dict[int, float]:
    """Season -> replacement-level EPA per dropback, from seasons strictly before it."""
    first_season = int(db["season"].min()) if len(db) else 0
    debut = db.groupby("qb")["season"].min()
    early = db[db["career_n"] <= rookie_dropbacks]
    early_debut = early["qb"].map(debut).to_numpy()
    out = {}
    for S in sorted({int(s) for s in seasons}):
        past = early["season"].to_numpy() < S
        m = past & (early_debut > first_season)
        if not m.any():
            m = past
        out[S] = float(early["epa"].to_numpy()[m].mean()) if m.any() else 0.0
    return out


# --------------------------------------------------------------------------- values at a week

@dataclass
class WeekState:
    """Per-QB weighted sums at one (season, week), plus the team-window QB mix and the prior."""
    qb_codes: np.ndarray      # index into `qbs`
    n: np.ndarray             # weighted dropbacks per QB code
    e: np.ndarray             # weighted EPA per QB code
    team_mix: pd.DataFrame    # team, qb code, weighted dropbacks inside the rating window
    prior: float

    def values(self, k: float) -> np.ndarray:
        return (self.e + k * self.prior) / (self.n + k)


def week_states(qg: pd.DataFrame, sched: pd.DataFrame, keys, cfg: QBConfig, priors: dict[int, float]
                ) -> tuple[dict[tuple[int, int], WeekState], pd.Index]:
    """WeekState for each (season, week) key, using only games before the week's as_of."""
    qbs = pd.Index(pd.unique(qg["qb"]))
    code = qbs.get_indexer(qg["qb"])
    asofs, ords = oa.week_asof(sched), oa.week_ordinals(sched)
    kick, seas, tord = qg["kick_ns"].to_numpy(), qg["season"].to_numpy(), qg["ord"].to_numpy()
    n, e, team = qg["n"].to_numpy(float), qg["epa"].to_numpy(float), qg["team"].to_numpy(object)
    out = {}
    for S, W in sorted(set((int(a), int(b)) for a, b in keys)):
        end = np.searchsorted(kick, asofs[(S, W)], side="left")
        w = oa.decay_weights(int(ords[(S, W)]), tord[:end], S, seas[:end], cfg.half_life, cfg.rho)
        N = np.bincount(code[:end], weights=w * n[:end], minlength=len(qbs))
        E = np.bincount(code[:end], weights=w * e[:end], minlength=len(qbs))
        win = seas[:end] >= S - 1
        mix = pd.DataFrame({"team": team[:end][win], "qb": code[:end][win], "w": (w * n[:end])[win]})
        mix = mix.groupby(["team", "qb"], as_index=False)["w"].sum()
        out[(S, W)] = WeekState(code, N, E, mix, priors.get(S, 0.0))
    return out, qbs


def team_remembered(state: WeekState, values: np.ndarray) -> pd.Series:
    """Team -> weighted average QB value over the team's dropbacks in the rating window."""
    m = state.team_mix
    if m.empty:
        return pd.Series(dtype=float)
    v = values[m["qb"].to_numpy()] * m["w"].to_numpy()
    num = pd.Series(v).groupby(m["team"].to_numpy()).sum()
    den = m.groupby("team")["w"].sum()
    return num / den


# --------------------------------------------------------------------------- starters

def team_game_starters(sched: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (game, side) with franchise, kickoff (ns), and the schedule's starting QB."""
    if "kickoff" not in sched.columns:
        sched = asof_mod.add_asof(sched)
    parts = []
    for side in ("home", "away"):
        col = f"{side}_qb_id"
        qb = sched[col] if col in sched.columns else pd.Series(np.nan, index=sched.index)
        parts.append(pd.DataFrame({
            "game_id": sched["game_id"].to_numpy(), "side": side,
            "team": [franchise(c, int(s)) for c, s in zip(sched[f"{side}_team"], sched["season"])],
            "kick_ns": te._utc_ns(sched["kickoff"]), "qb": qb.to_numpy(object)}))
    return pd.concat(parts, ignore_index=True).sort_values("kick_ns", kind="stable").reset_index(drop=True)


def last_starters(sched: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """For each game and side, the team's starter in its most recent game that kicked off before the game's as_of."""
    long = team_game_starters(sched)
    asof_ns = te._utc_ns(games["as_of"])
    out = {}
    for side in ("home", "away"):
        teams = np.array([franchise(c, int(s)) for c, s in zip(games[f"{side}_team"], games["season"])], dtype=object)
        res = np.full(len(games), None, dtype=object)
        for t in np.unique(teams):
            rows = long[long["team"] == t]
            qi = np.flatnonzero(teams == t)
            pos = np.searchsorted(rows["kick_ns"].to_numpy(), asof_ns[qi], side="left") - 1
            ok = pos >= 0
            res[qi[ok]] = rows["qb"].to_numpy(object)[pos[ok]]
        out[f"{side}_qb_id"] = res
    return pd.DataFrame(out, index=games["game_id"].to_numpy())


# --------------------------------------------------------------------------- features

def qb_deltas(states: dict, qbs: pd.Index, games: pd.DataFrame, starters: pd.DataFrame, k: float,
              detail: bool = False) -> pd.DataFrame:
    """qb_delta_home/away for each game from precomputed week states. `starters` is indexed by game_id."""
    cols = {f: np.zeros(len(games)) for f in FEATURES}
    extra = {f"qb_value_{s}": np.full(len(games), np.nan) for s in ("home", "away")} if detail else {}
    seasons, weeks = games["season"].to_numpy(int), games["week"].to_numpy(int)
    gids = games["game_id"].to_numpy()
    for (S, W), idx in pd.Series(np.arange(len(games))).groupby([seasons, weeks]).groups.items():
        st = states[(int(S), int(W))]
        vals = st.values(k)
        rem = team_remembered(st, vals)
        for side in ("home", "away"):
            for i in np.asarray(idx):
                qb = starters.at[gids[i], f"{side}_qb_id"]
                team = franchise(games[f"{side}_team"].iloc[i], int(S))
                if qb is None or (isinstance(qb, float) and np.isnan(qb)) or pd.isna(qb):
                    continue
                c = qbs.get_indexer([str(qb)])[0]
                v = vals[c] if c >= 0 else st.prior
                if detail:
                    extra[f"qb_value_{side}"][i] = v
                if team in rem.index:
                    cols[f"qb_delta_{side}"][i] = v - rem[team]
    out = pd.DataFrame({**cols, **extra}, index=gids)
    out.index.name = "game_id"
    return out


def build_features(pbp: pd.DataFrame, sched: pd.DataFrame, games: pd.DataFrame | None = None,
                   cfg: QBConfig | None = None, starter: str = "actual", detail: bool = False) -> pd.DataFrame:
    """qb_delta_home and qb_delta_away for each game in `games` (default: all of `sched`), indexed by game_id.

    `starter="actual"` uses the schedule's starting QBs (M3-D1); `"last"` uses
    each team's starter from its previous game (strict Wednesday rule).
    """
    if starter not in STARTERS:
        raise ValueError(f"starter must be one of {STARTERS}")
    cfg = cfg or tuned_config()
    if "as_of" not in sched.columns:
        sched = asof_mod.add_asof(sched)
    games = sched if games is None else games
    if "as_of" not in games.columns:
        games = games.merge(sched[["game_id", "kickoff", "as_of"]], on="game_id", how="left")
    db = dropback_plays(pbp, sched, cfg)
    qg = qb_game_table(db, sched)
    keys = list(zip(games["season"].astype(int), games["week"].astype(int)))
    priors = replacement_priors(db, {s for s, _ in keys}, cfg.rookie_dropbacks)
    states, qbs = week_states(qg, sched, keys, cfg, priors)
    if starter == "actual":
        cols = [c for c in ("home_qb_id", "away_qb_id") if c in games.columns]
        if len(cols) < 2:
            raise ValueError("starter='actual' needs home_qb_id and away_qb_id in the schedule")
        st = games.set_index("game_id")[["home_qb_id", "away_qb_id"]]
    else:
        st = last_starters(sched, games)
    return qb_deltas(states, qbs, games, st, cfg.k, detail=detail)


# --------------------------------------------------------------------------- tuning target (M3-D2)

def starter_targets(qg: pd.DataFrame, sched: pd.DataFrame, seasons: tuple[int, int]) -> pd.DataFrame:
    """Per REG game and side: the schedule starter's own dropbacks and EPA in that game (the k-tuning target)."""
    g = sched[sched["season"].between(*seasons) & (sched["game_type"] == "REG")]
    long = team_game_starters(g).merge(g[["game_id", "season", "week"]], on="game_id")
    t = long.merge(qg[["game_id", "team", "qb", "n", "epa"]], on=["game_id", "team", "qb"], how="inner")
    return t[t["n"] > 0].reset_index(drop=True)


def score_k(states: dict, qbs: pd.Index, targets: pd.DataFrame, k: float) -> dict:
    """Dropback-weighted MSE of the starter's per-game EPA per dropback, predicted by his value before the week."""
    pred = np.empty(len(targets))
    for (S, W), idx in targets.groupby(["season", "week"]).groups.items():
        st = states[(int(S), int(W))]
        vals = st.values(k)
        c = qbs.get_indexer(targets.loc[idx, "qb"].astype(str))
        pred[targets.index.get_indexer(idx)] = np.where(c >= 0, vals[np.maximum(c, 0)], st.prior)
    n = targets["n"].to_numpy(float)
    y = targets["epa"].to_numpy(float) / n
    return {"n_games": int(len(y)), "dropbacks": int(n.sum()),
            "wmse": float(np.sum(n * (y - pred) ** 2) / n.sum())}


# =========================================================================== M3b (C3): QB value extras
#
# Everything below is off by default. `build_features(..., extras=None)` (or
# `extras=QBExtras()`) runs the M3 code path above untouched, so A4s's
# qb_delta is byte-identical (tests/ml/test_m3b.py checks it). With extras:
#
# (a) CPOE composite. value = epa_value + cpoe_weight * (cpoe_value - cpoe_prior),
#     where cpoe_value is the QB's recency-weighted CPOE (percentage points,
#     nflfastR's `cpoe`, pass attempts with a value; 2006+) shrunk toward
#     replacement CPOE with `cpoe_k` pseudo-attempts, and cpoe_prior is that
#     replacement level (first 100 CPOE attempts of QBs, seasons before S). A QB
#     with no CPOE record, and everyone before 2006, adds 0.
# (b) Experience prior. The shrinkage target becomes replacement level plus an
#     offset for the QB's career dropbacks before as_of (unweighted; buckets
#     EXP_EDGES). QBs already playing in 1999, whose careers began before the
#     data, go in the top bucket. Draft round is NOT used: nflverse's players
#     table takes its draft fields from Pro-Football-Reference (nflverse-players
#     CONTRIBUTING.md), which this project avoids.
# (c) Aging. The data part of the value moves along a piecewise-linear age curve
#     F (slope `before` per year up to `peak`, `after` beyond) from the
#     weighted-average age of the QB's past dropbacks to his age at as_of:
#     value += n / (n + k) * (F(age_now) - F(age_data)). Birth dates come from
#     the nflverse players table (GSIS basic info, CC BY 4.0); a QB without one
#     gets no adjustment.

EXP_EDGES = (0, 100, 500, 2000)   # career-dropback buckets: [0,100), [100,500), [500,2000), [2000, inf)
CPOE_ROOKIE_ATTEMPTS = 100


@dataclass(frozen=True)
class QBExtras:
    cpoe_weight: float = 0.0            # (a) EPA per dropback per CPOE point; 0 = off
    cpoe_k: float = 100.0               # (a) pseudo-attempts of replacement CPOE
    experience: tuple = ()              # (b) prior offsets per EXP_EDGES bucket (EPA/dropback); () = off
    aging: tuple = ()                   # (c) (peak_age, slope_before, slope_after) per year; () = off

    def active(self) -> bool:
        return bool(self.cpoe_weight) or bool(self.experience) or bool(self.aging)

    def to_dict(self) -> dict:
        return {"cpoe_weight": self.cpoe_weight, "cpoe_k": self.cpoe_k, "experience": list(self.experience),
                "aging": list(self.aging), "exp_edges": list(EXP_EDGES)}


def birth_dates(players: pd.DataFrame) -> pd.Series:
    """gsis_id -> birth date (Timestamp) from the nflverse players table; rows without one are dropped."""
    b = pd.to_datetime(players["birth_date"], errors="coerce")
    out = pd.Series(b.to_numpy(), index=players["gsis_id"].astype(str).to_numpy())
    return out[out.notna() & ~out.index.duplicated()]


def qb_game_table_ext(db: pd.DataFrame, sched: pd.DataFrame, births: pd.Series | None = None) -> pd.DataFrame:
    """`qb_game_table` plus CPOE sums and counts per (game, team, qb) and the QB's age at kickoff (years)."""
    d = db.copy()
    if "cpoe" not in d.columns:
        d["cpoe"] = np.nan
    d["has_c"] = d["cpoe"].notna().astype(float)
    d["c"] = d["cpoe"].fillna(0.0)
    g = d.groupby(["game_id", "team", "qb"], sort=False).agg(
        n=("epa", "size"), epa=("epa", "sum"), nc=("has_c", "sum"), cs=("c", "sum"),
        season=("season", "first"), week=("week", "first"), kick_ns=("kick_ns", "first")).reset_index()
    ords = oa.week_ordinals(sched)
    g["ord"] = ords.reindex(pd.MultiIndex.from_frame(g[["season", "week"]])).to_numpy()
    g = g.sort_values("kick_ns", kind="stable").reset_index(drop=True)
    g["age"] = np.nan
    if births is not None and len(births):
        b = g["qb"].map(births)
        ok = b.notna().to_numpy()
        if ok.any():
            born_ns = pd.to_datetime(b[ok]).dt.tz_localize("UTC").dt.as_unit("ns").astype("int64").to_numpy()
            g.loc[ok, "age"] = (g.loc[ok, "kick_ns"].to_numpy() - born_ns) / (365.2425 * 86400e9)
    return g


def cpoe_priors(db: pd.DataFrame, seasons, rookie_attempts: int = CPOE_ROOKIE_ATTEMPTS) -> dict[int, float]:
    """Season -> replacement CPOE: pooled CPOE of QBs' first `rookie_attempts` CPOE attempts, seasons before S.

    QBs whose CPOE record starts in its first season (2006, or the data's first
    season with CPOE) are left out, since their careers started earlier; if that
    leaves no one, everyone counts. No CPOE data before S gives 0.
    """
    if "cpoe" not in db.columns:
        return {int(s): 0.0 for s in seasons}
    c = db[db["cpoe"].notna()].copy()
    if c.empty:
        return {int(s): 0.0 for s in seasons}
    c["cn"] = c.groupby("qb").cumcount() + 1
    first = int(c["season"].min())
    debut = c.groupby("qb")["season"].min()
    early = c[c["cn"] <= rookie_attempts]
    early_debut = early["qb"].map(debut).to_numpy()
    out = {}
    for S in sorted({int(s) for s in seasons}):
        past = early["season"].to_numpy() < S
        m = past & (early_debut > first)
        if not m.any():
            m = past
        out[S] = float(early["cpoe"].to_numpy()[m].mean()) if m.any() else 0.0
    return out


@dataclass
class WeekStateExt:
    base: WeekState
    nc: np.ndarray            # weighted CPOE attempts per QB code
    ec: np.ndarray            # weighted CPOE sum per QB code
    career: np.ndarray        # unweighted career dropbacks before as_of
    veteran: np.ndarray       # bool: QB already playing in the data's first season
    age_w: np.ndarray         # weighted sum of (dropbacks x age at the game) over games with a known age
    n_age: np.ndarray         # weighted dropbacks with a known age
    age_now: np.ndarray       # age at as_of (NaN if unknown)
    cprior: float

    def values(self, k: float, extras: QBExtras) -> np.ndarray:
        st = self.base
        prior = np.full(len(st.n), st.prior)
        if extras.experience:
            b = np.searchsorted(np.asarray(EXP_EDGES[1:], float), self.career, side="right")
            b = np.where(self.veteran, len(EXP_EDGES) - 1, b)
            prior = prior + np.asarray(extras.experience, float)[b]
        v = (st.e + k * prior) / (st.n + k)
        if extras.aging:
            peak, before, after = (float(a) for a in extras.aging)

            def F(a):
                return before * (np.minimum(a, peak) - peak) + after * (np.maximum(a, peak) - peak)
            with np.errstate(invalid="ignore", divide="ignore"):
                a_data = self.age_w / self.n_age
            adj = F(self.age_now) - F(a_data)
            ok = np.isfinite(adj) & (self.n_age > 0)
            v = v + np.where(ok, st.n / (st.n + k) * np.where(ok, adj, 0.0), 0.0)
        if extras.cpoe_weight:
            cv = (self.ec + extras.cpoe_k * self.cprior) / (self.nc + extras.cpoe_k)
            v = v + extras.cpoe_weight * (cv - self.cprior)
        return v


def week_states_ext(qg: pd.DataFrame, sched: pd.DataFrame, keys, cfg: QBConfig, priors: dict[int, float],
                    cpriors: dict[int, float], births: pd.Series | None = None) -> tuple[dict, pd.Index]:
    """`week_states` plus the extras' sums, for each (season, week) key, from games before the week's as_of."""
    base, qbs = week_states(qg, sched, keys, cfg, priors)
    code = qbs.get_indexer(qg["qb"])
    asofs, ords = oa.week_asof(sched), oa.week_ordinals(sched)
    kick, seas, tord = qg["kick_ns"].to_numpy(), qg["season"].to_numpy(), qg["ord"].to_numpy()
    n, nc, cs = qg["n"].to_numpy(float), qg["nc"].to_numpy(float), qg["cs"].to_numpy(float)
    age = qg["age"].to_numpy(float)
    has_age = np.isfinite(age)
    first = int(qg["season"].min()) if len(qg) else 0
    vet = np.zeros(len(qbs), bool)
    vet[np.unique(code[seas == first])] = True
    # birth date per QB, as UTC nanoseconds (NaN if unknown); a static fact, never an outcome
    born = np.full(len(qbs), np.nan)
    if births is not None and len(births):
        b = pd.Series(qbs.astype(str), index=qbs).map(births)
        ok = b.notna().to_numpy()
        if ok.any():
            born[ok] = pd.to_datetime(b[ok]).dt.tz_localize("UTC").dt.as_unit("ns").astype("int64").to_numpy()
    out = {}
    for S, W in sorted(set((int(a), int(b)) for a, b in keys)):
        end = np.searchsorted(kick, asofs[(S, W)], side="left")
        w = oa.decay_weights(int(ords[(S, W)]), tord[:end], S, seas[:end], cfg.half_life, cfg.rho)
        c = code[:end]
        L = len(qbs)
        NC = np.bincount(c, weights=w * nc[:end], minlength=L)
        EC = np.bincount(c, weights=w * cs[:end], minlength=L)
        CAR = np.bincount(c, weights=n[:end], minlength=L)
        ha = has_age[:end]
        AW = np.bincount(c[ha], weights=(w * n[:end] * age[:end])[ha], minlength=L)
        NA = np.bincount(c[ha], weights=(w * n[:end])[ha], minlength=L)
        now = (asofs[(S, W)] - born) / (365.2425 * 86400e9)
        out[(S, W)] = WeekStateExt(base[(S, W)], NC, EC, CAR, vet, AW, NA, now, cpriors.get(S, 0.0))
    return out, qbs


def qb_deltas_ext(states: dict, qbs: pd.Index, games: pd.DataFrame, starters: pd.DataFrame, k: float,
                  extras: QBExtras, unknown_starter_value: str = "prior") -> pd.DataFrame:
    """`qb_deltas` with the extras' values. A starter never seen before gets his prior (experience bucket 0)."""
    cols = {f: np.zeros(len(games)) for f in FEATURES}
    seasons, weeks = games["season"].to_numpy(int), games["week"].to_numpy(int)
    gids = games["game_id"].to_numpy()
    for (S, W), idx in pd.Series(np.arange(len(games))).groupby([seasons, weeks]).groups.items():
        st = states[(int(S), int(W))]
        vals = st.values(k, extras)
        rem = team_remembered(st.base, vals)
        new_prior = st.base.prior + (float(extras.experience[0]) if extras.experience else 0.0)
        for side in ("home", "away"):
            for i in np.asarray(idx):
                qb = starters.at[gids[i], f"{side}_qb_id"]
                team = franchise(games[f"{side}_team"].iloc[i], int(S))
                if qb is None or (isinstance(qb, float) and np.isnan(qb)) or pd.isna(qb):
                    continue
                c = qbs.get_indexer([str(qb)])[0]
                v = vals[c] if c >= 0 else new_prior
                if team in rem.index:
                    cols[f"qb_delta_{side}"][i] = v - rem[team]
    out = pd.DataFrame(cols, index=gids)
    out.index.name = "game_id"
    return out


def build_features_ext(pbp: pd.DataFrame, sched: pd.DataFrame, games: pd.DataFrame | None = None,
                       cfg: QBConfig | None = None, starter: str = "actual", extras: QBExtras | None = None,
                       births: pd.Series | None = None) -> pd.DataFrame:
    """qb_delta_home/away with the M3b extras. With no active extras this IS `build_features` (same code path)."""
    if extras is None or not extras.active():
        return build_features(pbp, sched, games, cfg, starter)
    if starter not in STARTERS:
        raise ValueError(f"starter must be one of {STARTERS}")
    cfg = cfg or tuned_config()
    if "as_of" not in sched.columns:
        sched = asof_mod.add_asof(sched)
    games = sched if games is None else games
    if "as_of" not in games.columns:
        games = games.merge(sched[["game_id", "kickoff", "as_of"]], on="game_id", how="left")
    st = ext_states(pbp, sched, games, cfg, births)
    starters = (games.set_index("game_id")[["home_qb_id", "away_qb_id"]] if starter == "actual"
                else last_starters(sched, games))
    return qb_deltas_ext(st["states"], st["qbs"], games, starters, cfg.k, extras)


def ext_states(pbp, sched, games, cfg: QBConfig, births=None, keys=None) -> dict:
    """The dropback table, per-game table, and extended week states for every (season, week) of `games`."""
    db = dropback_plays(pbp, sched, cfg)
    qg = qb_game_table_ext(db, sched, births)
    keys = keys if keys is not None else list(zip(games["season"].astype(int), games["week"].astype(int)))
    seasons = {s for s, _ in keys}
    priors = replacement_priors(db, seasons, cfg.rookie_dropbacks)
    cpriors = cpoe_priors(db, seasons)
    states, qbs = week_states_ext(qg, sched, keys, cfg, priors, cpriors, births)
    return {"db": db, "qg": qg, "states": states, "qbs": qbs}


def score_values(states: dict, qbs: pd.Index, targets: pd.DataFrame, k: float, extras: QBExtras) -> dict:
    """`score_k` for the extended values: dropback-weighted MSE of the starter's next-game EPA per dropback."""
    pred = np.empty(len(targets))
    for (S, W), idx in targets.groupby(["season", "week"]).groups.items():
        st = states[(int(S), int(W))]
        vals = st.values(k, extras)
        new_prior = st.base.prior + (float(extras.experience[0]) if extras.experience else 0.0)
        c = qbs.get_indexer(targets.loc[idx, "qb"].astype(str))
        pred[targets.index.get_indexer(idx)] = np.where(c >= 0, vals[np.maximum(c, 0)], new_prior)
    n = targets["n"].to_numpy(float)
    y = targets["epa"].to_numpy(float) / n
    return {"n_games": int(len(y)), "dropbacks": int(n.sum()),
            "wmse": float(np.sum(n * (y - pred) ** 2) / n.sum())}


# Frozen by `scripts/ml_m3b.py tune` (experience and aging on 2000-2005, CPOE on 2007-2008); see context/ml.md.
TUNED_EXTRAS = QBExtras(cpoe_weight=0.01, cpoe_k=400.0, experience=(-0.0541, 0.0159, 0.1412, 0.0648),
                        aging=(31.0, 0.14, 0.0))
# (b) re-tuned on 2000-2005 starter-games where the team's starter changed from its previous game (variant 'd').
TUNED_EXPERIENCE_CHANGED: tuple = (-0.0651, 0.08, 0.0329, 0.0318)
