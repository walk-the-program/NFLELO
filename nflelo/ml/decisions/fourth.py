"""M7a fourth-down valuation: go for it, kick a field goal, or punt, in win probability.

LICENSE: CC BY-SA 4.0. The "go" option uses the M6 call-conditioned yards
model, which is built on nflverse participation data (CC BY-SA 4.0), so every
valuation, recommendation and audit number from this module is CC BY-SA 4.0.
The WP model (`nflelo.ml.wp.model`) and the kicking models
(`nflelo.ml.decisions.kicking`) are CC BY on their own. Spec:
context/ml-m7-method.md, section 3 (Part A).

For each regulation fourth down of a REG game, as of the snap:

GO      The M6 call-view GBM (season-ahead: trained on 2016..S-1 with the M6
        protocol) gives a 53-bin yards distribution for a run and for a pass.
        The pre-snap structure (personnel, formation, box) of a play that did
        not happen is unknown, so each call's distribution is the average over
        N_CONFIGS structure configurations drawn from the training seasons'
        third- and fourth-down plays of that call at that distance bucket.
        Run and pass are mixed by the EMPIRICAL GO-FOR-IT RUN SHARE BY DISTANCE
        (`run_share`: actual fourth-down attempts in the WP training seasons,
        nflfastR `pass` flag, buckets 1, 2, 3, 4-5, 6-9, 10+), not by the
        team's own mix. Conversion = P(yards >= ydstogo or TD). Each yards bin
        leads to a state: a conversion is the team's 1st and 10 (or goal) at the
        new spot; a TD is +7 (extra point assumed) and the opponent's ball after
        the kickoff; a failure is the opponent's 1st and 10 at the spot (a loss
        behind the goal line is a safety: +2 to the opponent, who gets the ball
        at its SAFETY_OPP_YL). Turnovers are not modelled separately (an
        interception is a failure at the line).
FG      make (kicking.FGModel): +3, then the opponent's ball after the kickoff
        at the kickoff start as of the snap (`kicking.kickoff_asof`: this season's
        running mean from earlier weeks, else the previous season's); miss:
        the opponent's ball at the spot of the kick (8 yards behind the line) or
        its 20, whichever is farther from its goal.
PUNT    kicking.PuntModel: the expectation over the receiver's start-spot classes
        of the opponent's 1st-and-10 WP; a return TD is -7 and the team receives
        the kickoff; a kept ball is the team's 1st and 10 at the mean kept spot.
Time    each outcome subtracts its median snap-to-next-snap clock time
        (`kicking.durations`, last three training seasons: go converted / TD /
        failed, FG made / missed, punt), within the half.

WP of a state is the season-ahead WP model from the possession team's point of
view; an opponent-ball state is 1 minus the opponent's WP (scores, timeouts,
home and the pregame A4s probability flipped; the second-half kickoff goes to
the opponent iff the team does not receive it). The team's kicker and punter
are the last ones it used before the snap; their values use earlier games only.

Engine versions (`ENGINE_VERSION`, default "v1"): "v1" is the engine above, exactly as in the M7a dev
and holdout runs. "v2" (M7a-v2, context/ml.md "M7a-v2 build notes") changes only the go-for-it
conversion: the run/pass mix by distance x field zone (optionally with an as-of team offset) and,
optionally, fourth-down conversion offsets per call and distance. See the "engine v2" section below.

Uncertainty: `bootstrap` refits the WP model, the M6 bins model, the FG model
and the punt model on game-resampled training data (the same game draw for all
four in a replicate; hyperparameters, tree counts, K, run shares, structure
configurations, kickoff spot and durations fixed at the point fit). The call
is a "toss-up" when the 90% band of (best minus second-best WP), with best
and second fixed by the point estimate, includes 0.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ..wp import model as wpm
from . import kicking as kk
from ..plays import data as pdata
from ..plays import gbm
from ..plays import metrics as pm

LICENSE = "CC BY-SA 4.0"
LICENSE_NOTE = ("CC BY-SA 4.0: M7a fourth-down valuations use the M6 play model, built on nflverse participation "
                "data (NFL Next Gen Stats via nflverse 2016-2022, FTN Data via nflverse 2023+). "
                "Data: nflverse (CC BY 4.0).")
OPTIONS = ("go", "fg", "punt")
SEED = 20261003
FIRST_M6 = 2016
N_CONFIGS = 16
BUCKETS = ((1, 1), (2, 2), (3, 3), (4, 5), (6, 9), (10, 99))
TD_POINTS = 7.0
FG_POINTS = 3.0
SAFETY_OPP_YL = 60.0
LOW_YARDS, BIG_YARDS = -12.0, 55.0

PBP_COLUMNS = sorted(set(wpm.PBP_COLUMNS) | set(kk.PBP_COLUMNS) | {"pass", "desc"})


def bucket(ydstogo) -> np.ndarray:
    y = np.asarray(ydstogo, float)
    out = np.full(len(y), len(BUCKETS) - 1, int)
    for i, (a, b) in enumerate(BUCKETS):
        out[(y >= a) & (y <= b)] = i
    return out


# --------------------------------------------------------------------------- decisions

def team_player(pbp: pd.DataFrame, rows: pd.DataFrame, kind: str) -> np.ndarray:
    """The team's kicker ("k") or punter ("p") as of each row: the last one it used on an earlier play."""
    if kind == "k":
        ev = pbp[pbp["play_type"].isin(["field_goal", "extra_point"]) & pbp["kicker_player_id"].notna()]
        col = "kicker_player_id"
    else:
        ev = pbp[(pbp["play_type"] == "punt") & pbp["punter_player_id"].notna()]
        col = "punter_player_id"
    order = lambda d: kk.game_key(d["season"], d["week"]) * 100000 + d["play_id"].to_numpy(np.int64)  # noqa: E731
    e = pd.DataFrame({"team": ev["posteam"].astype(object).to_numpy(), "order": order(ev),
                      "player": ev[col].astype(object).to_numpy()}).sort_values("order")
    q = pd.DataFrame({"team": rows["posteam"].astype(object).to_numpy(), "order": order(rows),
                      "_row": np.arange(len(rows))}).sort_values("order")
    m = pd.merge_asof(q, e, on="order", by="team", allow_exact_matches=False, direction="backward")
    out = np.full(len(rows), None, dtype=object)
    out[m["_row"].to_numpy(int)] = m["player"].to_numpy(object)
    return out


def decision_table(pbp: pd.DataFrame, a4s_home: pd.Series, seasons=None) -> pd.DataFrame:
    """Regulation fourth downs of REG games with a go / FG / punt choice (no penalties, no kneels).

    Pre-snap inputs (state, weather, the team's kicker and punter as of the snap, A4s) plus the
    observed choice and, for the audit only, its realized outcome (converted, made, game won).
    Tie games and games without an A4s probability are dropped (as in the WP model).
    """
    p = pbp[(pbp["season_type"] == "REG") & (pbp["qtr"] <= 4) & (pbp["down"] == 4) & pbp["posteam"].notna()
            & pbp["play_type"].isin(["run", "pass", "field_goal", "punt"]) & pbp["yardline_100"].notna()
            & pbp["score_differential"].notna()]
    if seasons is not None:
        p = p[p["season"].isin(list(seasons))]
    p = p[p["game_id"].isin(a4s_home.index) & p["result"].notna() & (p["result"] != 0)]
    st = wpm.state_table(p.assign(down=4.0), a4s_home)
    st = st.set_index(["game_id", "play_id"]).loc[list(zip(p["game_id"], p["play_id"]))].reset_index()
    out = st.drop(columns=["y"]).copy()
    out["home_team"] = p["home_team"].to_numpy(object)
    out["choice"] = p["play_type"].map({"run": "go", "pass": "go", "field_goal": "fg", "punt": "punt"}).to_numpy()
    w = kk.weather(p)
    out["indoor"], out["wind_out"], out["cold"] = w["indoor"].to_numpy(), w["wind_out"].to_numpy(), w["cold"].to_numpy()
    roof = p["roof"].astype("string").str.lower()
    out["roof_open"] = (roof == "open").astype(float).to_numpy()
    out["temp"] = pd.to_numeric(p["temp"], errors="coerce").to_numpy(float)
    out["wind"] = pd.to_numeric(p["wind"], errors="coerce").to_numpy(float)
    out["gkey"] = kk.game_key(p["season"], p["week"])
    out["kicker"] = team_player(pbp, p, "k")
    out["punter"] = team_player(pbp, p, "p")
    out["desc"] = p["desc"].astype(str).to_numpy() if "desc" in p.columns else ""
    # realized outcomes (audit and consistency check only; never an input)
    out["out_converted"] = np.where(out["choice"] == "go", p["fourth_down_converted"].fillna(0).to_numpy(float), np.nan)
    out["out_made"] = np.where(out["choice"] == "fg", (p["field_goal_result"] == "made").astype(float).to_numpy(),
                               np.nan)
    res = p["result"].to_numpy(float)
    home = out["home"].to_numpy(float) == 1
    out["out_won"] = np.where(home, res > 0, res < 0).astype(float)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- components

@dataclass
class M6Yards:
    """M6 call-view bins classifier (CC BY-SA 4.0): folded 53-bin yards distribution."""
    clf: HistGradientBoostingClassifier

    def predict_X(self, X: pd.DataFrame, yardline: np.ndarray) -> np.ndarray:
        P = np.zeros((len(X), pdata.N_BINS))
        P[:, self.clf.classes_.astype(int)] = self.clf.predict_proba(X)
        return pdata.fold(P, yardline)


def run_share(pbp: pd.DataFrame, seasons) -> np.ndarray:
    """Empirical run share of fourth-down go-for-it attempts by distance bucket (nflfastR `pass` flag)."""
    p = pbp[(pbp["down"] == 4) & (pbp["qtr"] <= 4) & pbp["play_type"].isin(["run", "pass"])
            & pbp["season"].isin(list(seasons))]
    b = bucket(p["ydstogo"])
    is_run = (p["pass"].fillna(0) != 1).to_numpy(float)
    return np.array([is_run[b == i].mean() if (b == i).any() else 0.5 for i in range(len(BUCKETS))])


def structure_configs(train: pd.DataFrame, n: int = N_CONFIGS, seed: int = SEED) -> np.ndarray:
    """(2 calls, buckets, n, len(STRUCTURE)) pre-snap structures drawn from training 3rd/4th downs."""
    rng = np.random.default_rng(seed)
    t = train[train["down"] >= 3]
    b = bucket(t["ydstogo"])
    out = np.zeros((2, len(BUCKETS), n, len(pdata.STRUCTURE)))
    for call in (0, 1):
        for i in range(len(BUCKETS)):
            pool = t[(t["is_pass"] == call) & (b == i)]
            if len(pool) == 0:
                pool = t[t["is_pass"] == call]
            out[call, i] = pool[pdata.STRUCTURE].to_numpy(float)[rng.integers(0, len(pool), n)]
    return out


# --------------------------------------------------------------------------- engine v2 (M7a-v2)
#
# The v1 engine (the default, `ENGINE_VERSION`) is exactly the engine of the M7a dev and holdout runs.
# v2 changes only the go-for-it conversion: the run/pass mix and, optionally, a fourth-down offset on
# each call's conversion probability. Everything else (WP, FG, punt, structure configurations, the state
# construction) is shared. Spec and dev evidence: context/ml.md, "M7a-v2 build notes".
#
#   mix "v1"         the v1 share: league fourth-down go run share by v1 distance bucket (BUCKETS)
#   mix "cell"       REG fourth-down go attempts of seasons first..S-1, season-decay weights (half-life
#                    `half_life` seasons), run share by distance bucket (V2_DIST) x field zone (V2_ZONES):
#                    the bucket share shrunk to the overall share (k_bucket attempts), the cell share
#                    shrunk to its bucket share (k_cell attempts)
#   mix "cell_team"  "cell" plus the team's logit offset from its own go attempts in earlier games of
#                    season S (as of the snap: earlier weeks only), one Newton step with prior precision
#                    k_team: theta = sum(run - share) / (sum share (1 - share) + k_team)
#   offsets          beta[call, distance bucket] added to the logit of each call's engine conversion
#                    probability; estimated from out-of-fold engine predictions on the actual fourth-down go
#                    attempts of the M6 training seasons first_m6..S-1 (leave one training season out:
#                    M6 bins model refit on the other training seasons with the season's chosen M6 settings
#                    and tree count, structure configurations from that fold), penalized logistic with
#                    prior N(0, offset_prior_sd^2), one cell at a time. The call is nflfastR `pass`.
#                    The offset rescales the call's yards distribution: converting bins by P'/P, the
#                    others by (1 - P')/(1 - P), so the WP valuation and the conversion agree.
# Shares, team offsets and conversion offsets are fixed in the bootstrap replicates (like v1's run share).

ENGINE_VERSIONS = ("v1", "v2")
ENGINE_VERSION = "v1"                                  # the default: every M7a v1 result reproduces exactly
V2_DIST = ((1, 1), (2, 2), (3, 5), (6, 99))
V2_ZONES = ((1, 5), (6, 20), (21, 50), (51, 99))       # yards to the opponent's goal
SHARE_CLIP = (0.005, 0.995)


@dataclass(frozen=True)
class V2Settings:
    mix: str = "cell"                    # "v1", "cell" or "cell_team"
    offsets: bool = False
    half_life: float | None = 2.0        # seasons (None: equal weights)
    k_bucket: float = 0.0
    k_cell: float = 100.0
    k_team: float = 10.0
    offset_prior_sd: float = 0.25

    def to_dict(self) -> dict:
        return {"mix": self.mix, "offsets": self.offsets, "half_life": self.half_life, "k_bucket": self.k_bucket,
                "k_cell": self.k_cell, "k_team": self.k_team, "offset_prior_sd": self.offset_prior_sd}


# Mix settings (half_life 2, k_bucket 0, k_cell 100, k_team 10) were chosen on TRAINING seasons only: each
# season 2010-2017 predicted from 2006..s-1, log loss of the call on REG fourth-down go attempts. The
# offset prior sd (0.25 logit) is fixed a priori. Dev 2018-2019 scores the candidates; it tunes nothing.
V2_CANDIDATES = {
    "v2a": V2Settings(mix="cell"),
    "v2b": V2Settings(mix="cell_team"),
    "v2d": V2Settings(mix="v1", offsets=True),
    "v2ad": V2Settings(mix="cell", offsets=True),
    "v2bd": V2Settings(mix="cell_team", offsets=True),
}
V2_COMPLEXITY = {"v2a": 1, "v2b": 2, "v2d": 3, "v2ad": 4, "v2bd": 5}
# Chosen on dev by the CORRECTED rule (2026-10-05, before any 2026 data): within one SE of the best Brier, both
# 4th-and-1/2 selection gaps under 0.02, lowest ECE; ties simplest. The original rule (simplest within one SE)
# picked v2a, which does not address the diagnosed cause (context/ml.md, M7a-v2 build notes).
V2_CHOSEN = "v2ad"


def _index(v, bounds) -> np.ndarray:
    v = np.asarray(v, float)
    out = np.full(len(v), len(bounds) - 1, int)
    for i, (a, b) in enumerate(bounds):
        out[(v >= a) & (v <= b)] = i
    return out


def v2_bucket(ydstogo) -> np.ndarray:
    return _index(ydstogo, V2_DIST)


def v2_zone(yardline_100) -> np.ndarray:
    return _index(yardline_100, V2_ZONES)


def go_attempts(pbp: pd.DataFrame, seasons=None) -> pd.DataFrame:
    """REG regulation fourth-down go-for-it attempts (run or pass plays) with the call (nflfastR `pass`)."""
    p = pbp[(pbp["season_type"] == "REG") & (pbp["down"] == 4) & (pbp["qtr"] <= 4)
            & pbp["play_type"].isin(["run", "pass"]) & pbp["ydstogo"].notna() & pbp["yardline_100"].notna()
            & pbp["posteam"].notna()]
    if seasons is not None:
        p = p[p["season"].isin(list(seasons))]
    out = pd.DataFrame({"game_id": p["game_id"].to_numpy(object), "play_id": p["play_id"].to_numpy(np.int64),
                        "season": p["season"].to_numpy(int), "week": p["week"].to_numpy(int),
                        "posteam": p["posteam"].astype(object).to_numpy(),
                        "ydstogo": p["ydstogo"].to_numpy(float), "yardline_100": p["yardline_100"].to_numpy(float),
                        "run": (p["pass"].fillna(0) != 1).to_numpy(float)})
    out["gkey"] = kk.game_key(out["season"], out["week"])
    out["dbin"], out["zone"] = v2_bucket(out["ydstogo"]), v2_zone(out["yardline_100"])
    return out.reset_index(drop=True)


def attempts_hash(att: pd.DataFrame) -> str:
    """Content hash of the attempts a table was fit on (keys, situation and call)."""
    import hashlib
    cols = ["game_id", "play_id", "ydstogo", "yardline_100", "run"]
    a = att.sort_values(["game_id", "play_id"])[cols].reset_index(drop=True)
    a["play_id"] = a["play_id"].astype(np.int64)
    a[["ydstogo", "yardline_100", "run"]] = a[["ydstogo", "yardline_100", "run"]].astype(float)
    a["game_id"] = a["game_id"].astype(str)
    return hashlib.sha256(pd.util.hash_pandas_object(a, index=False).to_numpy().tobytes()).hexdigest()[:16]


@dataclass
class MixTable:
    """Run share of fourth-down go attempts by V2_DIST bucket x V2_ZONES zone (fit on seasons < S)."""
    S: int
    overall: float
    bucket: np.ndarray                   # (len(V2_DIST),)
    cell: np.ndarray                     # (len(V2_DIST), len(V2_ZONES))
    n_cell: np.ndarray
    info: dict = field(default_factory=dict)

    def share(self, ydstogo, yardline_100) -> np.ndarray:
        return self.cell[v2_bucket(ydstogo), v2_zone(yardline_100)]


def fit_mix(att: pd.DataFrame, S: int, settings: V2Settings, first: int = wpm.FIRST_TRAIN) -> MixTable:
    """The cell run-share table from attempts of seasons first..S-1 only."""
    tr = att[(att["season"] >= first) & (att["season"] <= S - 1)]
    s = tr["season"].to_numpy(int)
    w = np.ones(len(tr)) if settings.half_life is None else 0.5 ** ((S - 1 - s) / settings.half_life)
    r = tr["run"].to_numpy(float)
    b, z = tr["dbin"].to_numpy(int), tr["zone"].to_numpy(int)
    overall = float((w * r).sum() / w.sum())
    nb, nz = len(V2_DIST), len(V2_ZONES)
    bucket = np.array([((w * r)[b == i].sum() + settings.k_bucket * overall) / (w[b == i].sum() + settings.k_bucket)
                       if (w[b == i].sum() + settings.k_bucket) > 0 else overall for i in range(nb)])
    cell, n_cell = np.zeros((nb, nz)), np.zeros((nb, nz), int)
    for i in range(nb):
        for j in range(nz):
            m = (b == i) & (z == j)
            den = w[m].sum() + settings.k_cell
            cell[i, j] = ((w * r)[m].sum() + settings.k_cell * bucket[i]) / den if den > 0 else bucket[i]
            n_cell[i, j] = int(m.sum())
    return MixTable(S, overall, bucket, np.clip(cell, *SHARE_CLIP), n_cell,
                    {"seasons": [int(first), int(S - 1)], "n": int(len(tr)), "hash": attempts_hash(tr)})


def team_theta(att_S: pd.DataFrame, mix: MixTable, dec: pd.DataFrame, k_team: float) -> np.ndarray:
    """Each decision's team logit offset from the team's go attempts in EARLIER games of the same season."""
    out = np.zeros(len(dec))
    if att_S is None or not len(att_S) or not len(dec):
        return out
    a = att_S.copy()
    sh = mix.share(a["ydstogo"], a["yardline_100"])
    a["res"], a["inf"] = a["run"].to_numpy(float) - sh, sh * (1 - sh)
    g = a.groupby(["posteam", "gkey"], sort=True)[["res", "inf"]].sum().reset_index()
    g[["cres", "cinf"]] = g.groupby("posteam")[["res", "inf"]].cumsum()
    q = pd.DataFrame({"posteam": dec["posteam"].astype(object).to_numpy(), "gkey": dec["gkey"].to_numpy(np.int64),
                      "_row": np.arange(len(dec))}).sort_values("gkey")
    m = pd.merge_asof(q, g[["posteam", "gkey", "cres", "cinf"]].sort_values("gkey"), on="gkey", by="posteam",
                      allow_exact_matches=False, direction="backward")
    out[m["_row"].to_numpy(int)] = (m["cres"].fillna(0.0) / (m["cinf"].fillna(0.0) + k_team)).to_numpy(float)
    return out


@dataclass
class Offsets:
    """Fourth-down logit offsets on each call's engine conversion probability, beta[call (0 run, 1 pass), V2_DIST]."""
    beta: np.ndarray
    n: np.ndarray
    info: dict = field(default_factory=dict)


def fit_offset_cells(y, p, call, dbin, prior_sd: float, iters: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Penalized logistic offset per (call, bucket): maximize sum loglik(y | logit p + beta) - beta^2 / (2 sd^2)."""
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    call, dbin = np.asarray(call, int), np.asarray(dbin, int)
    lg = np.log(p / (1 - p))
    beta, n = np.zeros((2, len(V2_DIST))), np.zeros((2, len(V2_DIST)), int)
    tau = 1.0 / prior_sd ** 2
    for c in (0, 1):
        for i in range(len(V2_DIST)):
            m = (call == c) & (dbin == i)
            n[c, i] = int(m.sum())
            bt = 0.0
            for _ in range(iters):
                q = 1 / (1 + np.exp(-(lg[m] + bt)))
                g = (y[m] - q).sum() - tau * bt
                h = (q * (1 - q)).sum() + tau
                bt += g / h
            beta[c, i] = bt
    return beta, n


def shift_conversion(P: np.ndarray, beta: np.ndarray, ytg: np.ndarray, yl: np.ndarray) -> np.ndarray:
    """Rescale folded yards distributions so P(first down or TD) moves by `beta` on the logit scale."""
    yk = np.concatenate([pdata.BIN_YARDS[:pdata.BIG_BIN], [np.inf]])
    conv = np.concatenate([yk[None, :] >= np.asarray(ytg, float)[:, None], np.ones((len(P), 1), bool)], axis=1)
    pc = np.clip((P * conv).sum(axis=1), 1e-9, 1 - 1e-9)
    pn = 1 / (1 + np.exp(-(np.log(pc / (1 - pc)) + np.asarray(beta, float))))
    return np.where(conv, P * (pn / pc)[:, None], P * ((1 - pn) / (1 - pc))[:, None])


@dataclass
class EngineV2:
    settings: V2Settings
    mix: MixTable | None                  # None when settings.mix == "v1"
    offsets: Offsets | None
    att_S: pd.DataFrame | None = None     # season S attempts (team offsets read earlier games only)
    info: dict = field(default_factory=dict)
    v1_share: np.ndarray | None = None    # mix "v1": the v1 share table, frozen with the engine

    def share(self, dec: pd.DataFrame, v1_share: np.ndarray) -> np.ndarray:
        if self.settings.mix == "v1":
            return (self.v1_share if self.v1_share is not None else v1_share)[bucket(dec["ydstogo"])]
        s = self.mix.share(dec["ydstogo"], dec["yardline_100"])
        if self.settings.mix == "cell_team":
            th = team_theta(self.att_S, self.mix, dec, self.settings.k_team)
            s = 1 / (1 + np.exp(-(np.log(s / (1 - s)) + th)))
        return np.clip(s, *SHARE_CLIP)

    def to_json(self) -> dict:
        out = {"settings": self.settings.to_dict(), "info": self.info}
        if self.v1_share is not None:
            out["v1_share"] = {"buckets": [list(b) for b in BUCKETS], "share": self.v1_share.tolist()}
        if self.mix is not None:
            out["mix"] = {"S": self.mix.S, "overall": self.mix.overall, "bucket": self.mix.bucket.tolist(),
                          "cell": self.mix.cell.tolist(), "n_cell": self.mix.n_cell.tolist(), "info": self.mix.info}
        if self.offsets is not None:
            out["offsets"] = {"beta": self.offsets.beta.tolist(), "n": self.offsets.n.tolist(),
                              "info": self.offsets.info}
        return out

    @classmethod
    def from_json(cls, d: dict) -> "EngineV2":
        st = V2Settings(**d["settings"])
        mx = d.get("mix")
        mix = (MixTable(int(mx["S"]), float(mx["overall"]), np.array(mx["bucket"], float), np.array(mx["cell"], float),
                        np.array(mx["n_cell"], int), mx.get("info", {})) if mx else None)
        of = d.get("offsets")
        offs = Offsets(np.array(of["beta"], float), np.array(of["n"], int), of.get("info", {})) if of else None
        v1 = np.array(d["v1_share"]["share"], float) if d.get("v1_share") else None
        return cls(st, mix, offs, None, d.get("info", {}), v1)


@dataclass
class Components:
    S: int
    wp: wpm.WPModel
    yards: M6Yards
    fg: kk.FGModel
    punt: kk.PuntModel
    ko_spot: float                    # previous season's mean kickoff start (descriptive; the fallback)
    dur: dict
    run_share: np.ndarray
    configs: np.ndarray
    ko_table: pd.DataFrame | None = None
    info: dict = field(default_factory=dict)
    engine_version: str = ENGINE_VERSION
    v2: EngineV2 | None = None

    def kickoff(self, dec: pd.DataFrame) -> np.ndarray:
        """Kickoff start as of each decision (earlier weeks of its season only; see kicking.kickoff_asof)."""
        return kk.kickoff_asof(self.ko_table, dec["gkey"].to_numpy(np.int64))


@dataclass
class Tables:
    """Training tables of the point fit (kept for the bootstrap refits)."""
    wp: pd.DataFrame
    m6: pd.DataFrame
    fg: pd.DataFrame
    punt: pd.DataFrame


def oof_conversion(S: int, pbp: pd.DataFrame, a4s_home: pd.Series, m6_table: pd.DataFrame, ratings: pd.DataFrame,
                   m6_params: dict, n_iter: int, first_m6: int = FIRST_M6, log=None) -> pd.DataFrame:
    """Out-of-fold engine conversion (run call and pass call) on the actual fourth-down go attempts of the M6
    training seasons first_m6..S-1: for each training season T, the M6 bins model is refit on the OTHER
    training seasons (< S) with the given settings and tree count, with structure configurations from that fold.
    Nothing from season S or later is read."""
    feats = pdata.check_features(pdata.FEATURES["call"])
    train = m6_table[m6_table["season"].between(first_m6, S - 1)]
    past = pbp[pbp["season"] <= S - 1]
    dec = decision_table(past, a4s_home, seasons=range(first_m6, S))
    go = dec[dec["choice"] == "go"]
    calls = go_attempts(past, range(first_m6, S))[["game_id", "play_id", "run"]]
    out = []
    for T in range(first_m6, S):
        tr = train[train["season"] != T]
        d = go[go["season"] == T].reset_index(drop=True)
        if not len(tr) or not len(d):
            continue
        t0 = time.perf_counter()
        clf = gbm._clf({**gbm.FIXED, **m6_params}, int(n_iter), False).fit(tr[feats], tr["bin"])
        fr = _go_frames(d, structure_configs(tr), ratings)
        y = M6Yards(clf)
        n, k = len(d), fr["k"]
        ytg, yl = d["ydstogo"].to_numpy(float), d["yardline_100"].to_numpy(float)
        pr = y.predict_X(fr[0], fr["yl"]).reshape(n, k, -1).mean(axis=1)
        pp = y.predict_X(fr[1], fr["yl"]).reshape(n, k, -1).mean(axis=1)
        out.append(d[["game_id", "play_id", "season", "week", "ydstogo", "yardline_100", "out_converted"]].assign(
            p_run=pm.event_probs(pr, ytg, yl)["first"], p_pass=pm.event_probs(pp, ytg, yl)["first"]))
        if log:
            log(f"  v2 offsets {S}: fold {T} ({len(tr):,} M6 plays, {n} attempts, {time.perf_counter() - t0:.0f}s)")
    o = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["game_id", "play_id", "season", "week", "ydstogo", "yardline_100", "out_converted", "p_run", "p_pass"])
    o = o.merge(calls, on=["game_id", "play_id"], how="inner", validate="one_to_one")
    return o


def fit_offsets(oof: pd.DataFrame, prior_sd: float) -> Offsets:
    run = oof["run"].to_numpy(float) == 1
    p = np.where(run, oof["p_run"].to_numpy(float), oof["p_pass"].to_numpy(float))
    beta, n = fit_offset_cells(oof["out_converted"].to_numpy(float), p, (~run).astype(int),
                               v2_bucket(oof["ydstogo"]), prior_sd)
    seasons = sorted(set(oof["season"].astype(int))) if len(oof) else []
    return Offsets(beta, n, {"seasons": [seasons[0], seasons[-1]] if seasons else [], "n": int(len(oof)),
                             "hash": attempts_hash(oof.assign(run=oof["run"])) if len(oof) else None,
                             "prior_sd": prior_sd})


def fit_v2(S: int, pbp: pd.DataFrame, settings: V2Settings, m6_table: pd.DataFrame | None = None,
           a4s_home: pd.Series | None = None, ratings: pd.DataFrame | None = None, m6_params: dict | None = None,
           n_iter: int | None = None, first_wp: int = wpm.FIRST_TRAIN, first_m6: int = FIRST_M6,
           offsets: Offsets | None = None, log=None) -> EngineV2:
    """The v2 engine for season S from seasons before S (team offsets: earlier games of S only)."""
    att = go_attempts(pbp[pbp["season"] <= S])
    mix = fit_mix(att, S, settings, first_wp) if settings.mix != "v1" else None
    if settings.offsets and offsets is None:
        if any(v is None for v in (m6_table, a4s_home, ratings, m6_params, n_iter)):
            raise ValueError("v2 offsets need m6_table, a4s_home, ratings, m6_params and n_iter")
        offsets = fit_offsets(oof_conversion(S, pbp, a4s_home, m6_table, ratings, m6_params, n_iter, first_m6, log),
                              settings.offset_prior_sd)
    return EngineV2(settings, mix, offsets if settings.offsets else None,
                    att[att["season"] == S].reset_index(drop=True) if settings.mix == "cell_team" else None,
                    {"S": int(S), "first_wp": int(first_wp), "first_m6": int(first_m6)},
                    run_share(pbp, range(first_wp, S)) if settings.mix == "v1" else None)


def attach_season(v2: EngineV2, pbp: pd.DataFrame, S: int) -> EngineV2:
    """A frozen v2 engine for season S: the fitted tables as they are, plus S's attempts for the as-of team offsets."""
    att = go_attempts(pbp[pbp["season"] == S]) if v2.settings.mix == "cell_team" else None
    return EngineV2(v2.settings, v2.mix, v2.offsets, att, {**v2.info, "S": int(S)}, v2.v1_share)


def with_engine(comp: Components, v2: EngineV2 | None) -> Components:
    """The same fitted components with another go-for-it engine (None: v1)."""
    from dataclasses import replace
    return replace(comp, engine_version="v1" if v2 is None else "v2", v2=v2)


def fit_components(S: int, pbp: pd.DataFrame, wp_table: pd.DataFrame, m6_table: pd.DataFrame,
                   first_wp: int = wpm.FIRST_TRAIN, first_m6: int = FIRST_M6, m6_grid=gbm.GRID,
                   wp_calibrate: bool = wpm.CALIBRATE, wp: wpm.WPModel | None = None, fg: tuple | None = None,
                   punt: tuple | None = None, engine_version: str = ENGINE_VERSION,
                   v2_settings: V2Settings | None = None, v2: EngineV2 | None = None,
                   a4s_home: pd.Series | None = None, ratings: pd.DataFrame | None = None,
                   log=None, m6: gbm.GBMModel | None = None,
                   configs: np.ndarray | None = None) -> tuple[Components, Tables]:
    """Every component for season S, fit on seasons before S only (see the module docstring).

    `wp`, `fg` (model, rows) and `punt` (model, rows) may be passed in when already fit for S
    with the same functions (the dev script scores them first). `engine_version` "v1" (default) is the
    M7a engine; "v2" uses `v2` (a frozen engine; S's attempts are attached for the team offsets) or fits
    one with `v2_settings` (default: V2_CANDIDATES[V2_CHOSEN]; offsets need `a4s_home` and `ratings`).

    `m6` (the call-view GBM from `gbm.fit_season_ahead` on first_m6..S-1) and `configs`
    (`structure_configs` of that training table) may be passed in too (the site exporter caches them).
    With `wp`, `m6` and `configs` given, `wp_table` and `m6_table` may be None; the returned Tables
    then hold None for them and cannot drive a bootstrap."""
    if engine_version not in ENGINE_VERSIONS:
        raise ValueError(f"engine_version must be one of {ENGINE_VERSIONS}")
    t0 = time.perf_counter()
    wp = wp if wp is not None else wpm.fit_season_ahead(wp_table, S, first_wp, calibrate=wp_calibrate)
    m6_train = m6_table[m6_table["season"].between(first_m6, S - 1)] if m6_table is not None else None
    g = m6 if m6 is not None else gbm.fit_season_ahead(m6_train, "call", grid=m6_grid)
    configs = configs if configs is not None else structure_configs(m6_train)
    fg_model, fg_x = fg if fg is not None else kk.fit_fg_season_ahead(kk.fg_table(pbp[pbp["season"] <= S]), S, first_wp)
    punt_model, punt_x = (punt if punt is not None
                          else kk.fit_punt_season_ahead(kk.punt_table(pbp[pbp["season"] <= S]), S, first_wp))
    ko_table = kk.kickoff_table(pbp[pbp["season"] <= S])
    comp = Components(S, wp, M6Yards(g.bins), fg_model, punt_model,
                      float(ko_table.loc[ko_table["season"] == S - 1, "start"].mean()),
                      kk.durations(pbp, range(max(first_wp, S - 3), S)), run_share(pbp, range(first_wp, S)),
                      configs, ko_table,
                      {"wp": wp.info, "m6": {"chosen": g.info.get("chosen"), "n_iter": g.info["n_iter"],
                                             "seconds": g.info.get("total_seconds")},
                       "fg": fg_model.info, "punt": punt_model.info, "seconds": time.perf_counter() - t0})
    comp.info.update({"ko_spot": comp.ko_spot, "durations": comp.dur, "run_share": comp.run_share.tolist()})
    wp_train = wp_table[wp_table["season"].between(first_wp, S - 1)] if wp_table is not None else None
    tables = Tables(wp_train, m6_train,
                    fg_x[fg_x["season"].between(first_wp, S - 1)], punt_x[punt_x["season"].between(first_wp, S - 1)])
    if engine_version == "v2":
        t1 = time.perf_counter()
        if v2 is not None:
            eng = attach_season(v2, pbp, S)
        else:
            eng = fit_v2(S, pbp, v2_settings or V2_CANDIDATES[V2_CHOSEN], m6_table, a4s_home, ratings,
                         g.info.get("chosen"), g.info["n_iter"], first_wp, first_m6, log=log)
        comp = with_engine(comp, eng)
        comp.info = {**comp.info, "engine": {"version": "v2", **eng.to_json(), "seconds": time.perf_counter() - t1}}
    return comp, tables


# --------------------------------------------------------------------------- valuation

def _go_frames(dec: pd.DataFrame, configs: np.ndarray, ratings: pd.DataFrame) -> dict:
    """M6 feature matrices for the run (0) and pass (1) calls of every decision, one row per structure config."""
    n = len(dec)
    sit = pd.DataFrame({"down": 4.0, "ydstogo": dec["ydstogo"].to_numpy(float),
                        "yardline_100": dec["yardline_100"].to_numpy(float),
                        "score_diff": dec["score_diff"].to_numpy(float), "half_secs": dec["half_secs"].to_numpy(float),
                        "game_secs": dec["game_secs"].to_numpy(float), "off_timeouts": dec["pos_timeouts"].to_numpy(float),
                        "def_timeouts": dec["def_timeouts"].to_numpy(float), "home": dec["home"].to_numpy(float),
                        "roof_indoor": dec["indoor"].to_numpy(float), "roof_open": dec["roof_open"].to_numpy(float),
                        "temp": dec["temp"].to_numpy(float), "wind": dec["wind"].to_numpy(float)})
    rat = pdata.rating_features(pd.DataFrame({"season": dec["season"].to_numpy(int), "week": dec["week"].to_numpy(int),
                                              "off_team": dec["posteam"].to_numpy(object),
                                              "def_team": dec["defteam"].to_numpy(object)}), ratings)
    b = bucket(dec["ydstogo"])
    feats = pdata.check_features(pdata.FEATURES["call"])
    out = {}
    for call in (0, 1):
        struct = configs[call, b]                                        # (n, N_CONFIGS, n_struct)
        k = struct.shape[1]
        frame = pd.DataFrame(np.repeat(sit.to_numpy(float), k, axis=0), columns=sit.columns)
        frame[pdata.STRUCTURE] = struct.reshape(n * k, -1)
        frame[pdata.RATINGS] = np.repeat(rat[pdata.RATINGS].to_numpy(float), k, axis=0)
        frame["is_pass"] = float(call)
        out[call] = frame[feats]
    out["yl"] = np.repeat(dec["yardline_100"].to_numpy(float), configs.shape[2])
    out["k"] = configs.shape[2]
    return out


def go_inputs(dec: pd.DataFrame, comp: Components, ratings: pd.DataFrame) -> dict:
    """M6 feature matrices for the run and pass calls of every decision (N_CONFIGS structures each)."""
    if getattr(comp, "engine_version", "v1") == "v2":
        out = _go_frames(dec, comp.configs, ratings)
        out["share_run"] = comp.v2.share(dec, comp.run_share)
        if comp.v2.offsets is not None:
            out["beta"] = comp.v2.offsets.beta[:, v2_bucket(dec["ydstogo"])]          # (2 calls, n)
            out["ytg"] = dec["ydstogo"].to_numpy(float)
            out["yl_row"] = dec["yardline_100"].to_numpy(float)
        return out
    n = len(dec)
    sit = pd.DataFrame({"down": 4.0, "ydstogo": dec["ydstogo"].to_numpy(float),
                        "yardline_100": dec["yardline_100"].to_numpy(float),
                        "score_diff": dec["score_diff"].to_numpy(float), "half_secs": dec["half_secs"].to_numpy(float),
                        "game_secs": dec["game_secs"].to_numpy(float), "off_timeouts": dec["pos_timeouts"].to_numpy(float),
                        "def_timeouts": dec["def_timeouts"].to_numpy(float), "home": dec["home"].to_numpy(float),
                        "roof_indoor": dec["indoor"].to_numpy(float), "roof_open": dec["roof_open"].to_numpy(float),
                        "temp": dec["temp"].to_numpy(float), "wind": dec["wind"].to_numpy(float)})
    rat = pdata.rating_features(pd.DataFrame({"season": dec["season"].to_numpy(int), "week": dec["week"].to_numpy(int),
                                              "off_team": dec["posteam"].to_numpy(object),
                                              "def_team": dec["defteam"].to_numpy(object)}), ratings)
    b = bucket(dec["ydstogo"])
    feats = pdata.check_features(pdata.FEATURES["call"])
    out = {}
    for call in (0, 1):
        struct = comp.configs[call, b]                                   # (n, N_CONFIGS, n_struct)
        k = struct.shape[1]
        frame = pd.DataFrame(np.repeat(sit.to_numpy(float), k, axis=0), columns=sit.columns)
        frame[pdata.STRUCTURE] = struct.reshape(n * k, -1)
        frame[pdata.RATINGS] = np.repeat(rat[pdata.RATINGS].to_numpy(float), k, axis=0)
        frame["is_pass"] = float(call)
        out[call] = frame[feats]
    out["yl"] = np.repeat(dec["yardline_100"].to_numpy(float), comp.configs.shape[2])
    out["k"] = comp.configs.shape[2]
    out["share_run"] = comp.run_share[b]
    return out


def go_distribution(yards: M6Yards, gi: dict, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mixture, run, pass) folded 53-bin distributions, each averaged over the structure configurations."""
    k = gi["k"]
    pr = yards.predict_X(gi[0], gi["yl"]).reshape(n, k, -1).mean(axis=1)
    pp = yards.predict_X(gi[1], gi["yl"]).reshape(n, k, -1).mean(axis=1)
    if "beta" in gi:                                                     # v2 fourth-down offsets
        pr = shift_conversion(pr, gi["beta"][0], gi["ytg"], gi["yl_row"])
        pp = shift_conversion(pp, gi["beta"][1], gi["ytg"], gi["yl_row"])
    s = gi["share_run"][:, None]
    return s * pr + (1 - s) * pp, pr, pp


def _bin_yards(yl: np.ndarray) -> np.ndarray:
    """(n, 52) representative yards of each non-TD bin."""
    y = np.broadcast_to(pdata.BIN_YARDS[None, :pdata.TD_BIN], (len(yl), pdata.TD_BIN)).copy()
    y[:, pdata.LOSS_BIN] = LOW_YARDS
    y[:, pdata.BIG_BIN] = np.minimum(BIG_YARDS, yl - 1)
    return y


class StateBatch:
    """Collects post-decision states and scores them with one WP call."""

    def __init__(self, dec: pd.DataFrame):
        self.d = {c: dec[c].to_numpy(float) for c in ("score_diff", "game_secs", "half_secs", "half2",
                                                      "pos_timeouts", "def_timeouts", "receive_2h_ko", "home",
                                                      "a4s_prob")}
        self.n = len(dec)
        self.parts = []

    def add(self, key, idx, offense, yl, d_sd, dt, weight):
        """States for decisions `idx`: offense (bool array) = the deciding team has the ball at `yl`
        (its yards to go), else the opponent has it at `yl` (the opponent's yards to go)."""
        idx = np.asarray(idx, int)
        b = lambda v, t: np.broadcast_to(np.asarray(v, t), idx.shape)  # noqa: E731
        self.parts.append((key, idx, b(offense, bool), b(yl, float), b(d_sd, float), b(dt, float), b(weight, float)))

    def evaluate(self, wp: wpm.WPModel) -> dict:
        idx = np.concatenate([p[1] for p in self.parts])
        off = np.concatenate([p[2] for p in self.parts])
        yl = np.clip(np.concatenate([p[3] for p in self.parts]), 1, 99)
        dsd = np.concatenate([p[4] for p in self.parts])
        dt = np.concatenate([p[5] for p in self.parts])
        d = {k: v[idx] for k, v in self.d.items()}
        sd = d["score_diff"] + dsd
        first_half = d["half2"] == 0
        st = pd.DataFrame({
            "score_diff": np.where(off, sd, -sd),
            "game_secs": np.maximum(d["game_secs"] - dt, 0.0),
            "half_secs": np.maximum(d["half_secs"] - dt, 0.0),
            "half2": d["half2"], "down": 1.0, "ydstogo": np.minimum(10.0, yl), "yardline_100": yl,
            "pos_timeouts": np.where(off, d["pos_timeouts"], d["def_timeouts"]),
            "def_timeouts": np.where(off, d["def_timeouts"], d["pos_timeouts"]),
            "receive_2h_ko": np.where(off, d["receive_2h_ko"], first_half * (1 - d["receive_2h_ko"])),
            "home": np.where(off, d["home"], 1 - d["home"]),
            "a4s_prob": np.where(off, d["a4s_prob"], 1 - d["a4s_prob"])})
        p = wp.predict(wpm.add_derived(st))
        mine = np.where(off, p, 1 - p)
        out, start = {}, 0
        for key, i, _, _, _, _, w in self.parts:
            m = len(i)
            v = np.zeros(self.n)
            np.add.at(v, i, w * mine[start:start + m])
            out[key] = out.get(key, 0.0) + v
            start += m
        return out


def option_values(comp: Components, dec: pd.DataFrame, gi: dict, wp: wpm.WPModel | None = None,
                  yards: M6Yards | None = None, fg: kk.FGModel | None = None,
                  punt: kk.PuntModel | None = None) -> pd.DataFrame:
    """WP after each option for every decision (the model parts can be swapped for bootstrap replicates)."""
    wp, yards = wp or comp.wp, yards or comp.yards
    fg, punt = fg or comp.fg, punt or comp.punt
    n = len(dec)
    i = np.arange(n)
    yl = dec["yardline_100"].to_numpy(float)
    ytg = dec["ydstogo"].to_numpy(float)
    ko = comp.kickoff(dec)
    sb = StateBatch(dec)
    # GO
    p_go, p_run, p_pass = go_distribution(yards, gi, n)
    ys = _bin_yards(yl)
    du = comp.dur
    for k in range(pdata.TD_BIN):
        new = yl - ys[:, k]
        ok = ys[:, k] >= ytg
        safety = new >= 100
        w = p_go[:, k]
        sb.add("go", i[ok], True, new[ok], 0.0, du["go_conv"], w[ok])
        f = ~ok & ~safety
        sb.add("go", i[f], False, 100 - new[f], 0.0, du["go_fail"], w[f])
        sb.add("go", i[safety], False, SAFETY_OPP_YL, -2.0, du["go_fail"], w[safety])
    sb.add("go", i, False, ko, TD_POINTS, du["go_td"], p_go[:, pdata.TD_BIN])
    # FG
    fgx = pd.concat([kk.fg_inputs(dec["yardline_100"].reset_index(drop=True)),
                     dec[["indoor", "wind_out", "cold"]].reset_index(drop=True)], axis=1)
    fgx["kval"] = fg.value(dec["kicker"].to_numpy(object), dec["gkey"].to_numpy(np.int64))
    fgx["lg"] = fg.league(dec["gkey"].to_numpy(np.int64))
    p_fg = fg.predict(fgx)
    sb.add("fg", i, False, ko, FG_POINTS, du["fg_make"], p_fg)
    sb.add("fg", i, False, 100 - np.maximum(yl + kk.MISS_SPOT_BACK, 20.0), 0.0, du["fg_miss"], 1 - p_fg)
    # PUNT
    px = dec[["yardline_100", "indoor", "wind_out"]].reset_index(drop=True).copy()
    px["pval"] = punt.value(dec["punter"].to_numpy(object), dec["gkey"].to_numpy(np.int64))
    px["lgp"] = punt.league(dec["gkey"].to_numpy(np.int64))
    P = punt.predict(px)
    dt = du["punt"]
    for c in range(1, kk.N_PUNT - 1):
        sb.add("punt", i, False, np.full(n, punt.class_r[c]), 0.0, dt, P[:, c])
    sb.add("punt", i, True, ko, -TD_POINTS, dt, P[:, kk.TD_CLASS])
    sb.add("punt", i, True, np.full(n, punt.keep_yl), 0.0, dt, P[:, kk.KEEP_CLASS])
    v = sb.evaluate(wp)
    ev = lambda p: pm.event_probs(p, ytg, yl)["first"]  # noqa: E731
    return pd.DataFrame({"wp_go": v["go"], "wp_fg": v["fg"], "wp_punt": v["punt"], "p_conv": ev(p_go),
                         "p_conv_run": ev(p_run), "p_conv_pass": ev(p_pass), "share_run": gi["share_run"],
                         "p_fg": p_fg, "kval": fgx["kval"].to_numpy(), "pval": px["pval"].to_numpy(), "ko_start": ko},
                        index=dec.index)


def recommend(vals: pd.DataFrame) -> pd.DataFrame:
    W = vals[["wp_go", "wp_fg", "wp_punt"]].to_numpy(float)
    order = np.argsort(-W, axis=1)
    best, second = order[:, 0], order[:, 1]
    r = np.arange(len(W))
    return pd.DataFrame({"best": np.array(OPTIONS)[best], "second": np.array(OPTIONS)[second],
                         "margin": W[r, best] - W[r, second], "wp_best": W[r, best]}, index=vals.index)


# --------------------------------------------------------------------------- bootstrap

def game_weights(game_ids, rng: np.random.Generator) -> pd.Series:
    """Multinomial game counts: a bootstrap resample of games, as weights (same draw for every table)."""
    g = pd.Index(pd.unique(np.asarray(game_ids, dtype=object)))
    return pd.Series(rng.multinomial(len(g), np.full(len(g), 1.0 / len(g))).astype(float), index=g)


def refit(comp: Components, tables: Tables, w: pd.Series, first_wp: int = wpm.FIRST_TRAIN) -> tuple:
    """One bootstrap replicate of the four models on game-weighted training tables."""
    def sw(t):
        return w.reindex(t["game_id"]).fillna(0.0).to_numpy(float)
    wp = wpm.fit_final(tables.wp, comp.wp.info["n_iter"], comp.wp.iso, sample_weight=sw(tables.wp))
    m6 = tables.m6
    clf = gbm._clf({**gbm.FIXED, **comp.info["m6"]["chosen"]}, comp.info["m6"]["n_iter"], False)
    clf.fit(m6[pdata.FEATURES["call"]], m6["bin"], sample_weight=sw(m6))   # DataFrame, as in M6
    fg = kk.refit_fg(comp.fg, tables.fg.reset_index(drop=True), comp.S, first_wp, sw(tables.fg))
    punt = kk.refit_punt(comp.punt, tables.punt.reset_index(drop=True), comp.S, first_wp, sw(tables.punt))
    return wp, M6Yards(clf), fg, punt


def bootstrap(comp: Components, tables: Tables, dec: pd.DataFrame, gi: dict, B: int, seed: int = SEED,
              first_wp: int = wpm.FIRST_TRAIN, log=None) -> np.ndarray:
    """(B, n, 3) option WPs from B game-resampled refits of the component models."""
    rng = np.random.default_rng(seed + comp.S)
    games = pd.unique(pd.concat([tables.wp["game_id"], tables.m6["game_id"], tables.fg["game_id"],
                                 tables.punt["game_id"]]).to_numpy(object))
    out = np.zeros((B, len(dec), 3))
    for b in range(B):
        t0 = time.perf_counter()
        w = game_weights(games, rng)
        wp, yards, fg, punt = refit(comp, tables, w, first_wp)
        v = option_values(comp, dec, gi, wp, yards, fg, punt)
        out[b] = v[["wp_go", "wp_fg", "wp_punt"]].to_numpy(float)
        if log:
            log(f"  bootstrap {comp.S} {b + 1}/{B} ({time.perf_counter() - t0:.1f}s)")
    return out


def bands(point: pd.DataFrame, rec: pd.DataFrame, boot: np.ndarray, level: float = 0.90) -> pd.DataFrame:
    """90% band of (best minus second) with best/second fixed by the point estimate; toss-up if it includes 0."""
    k = {o: j for j, o in enumerate(OPTIONS)}
    bi = rec["best"].map(k).to_numpy(int)
    si = rec["second"].map(k).to_numpy(int)
    r = np.arange(len(rec))
    d = boot[:, r, bi] - boot[:, r, si]
    a = (1 - level) / 2
    lo, hi = np.quantile(d, [a, 1 - a], axis=0)
    sd = boot.std(axis=0, ddof=1)
    return pd.DataFrame({"band_lo": lo, "band_hi": hi, "tossup": (lo <= 0) & (hi >= 0),
                         "sd_go": sd[:, 0], "sd_fg": sd[:, 1], "sd_punt": sd[:, 2]}, index=rec.index)


def value_season(comp: Components, dec: pd.DataFrame, ratings: pd.DataFrame, tables: Tables | None = None,
                 B: int = 0, seed: int = SEED, first_wp: int = wpm.FIRST_TRAIN, log=None) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Valuation of every decision row (one season, components fit for that season)."""
    gi = go_inputs(dec, comp, ratings)
    vals = option_values(comp, dec, gi)
    rec = recommend(vals)
    out = pd.concat([dec.reset_index(drop=True), vals.reset_index(drop=True), rec.reset_index(drop=True)], axis=1)
    W = out[["wp_go", "wp_fg", "wp_punt"]]
    out["wp_choice"] = W.to_numpy()[np.arange(len(out)), out["choice"].map({o: j for j, o in enumerate(OPTIONS)})]
    out["lost"] = out["wp_best"] - out["wp_choice"]
    out["matched"] = out["choice"] == out["best"]
    boot = None
    if B > 0 and tables is not None:
        boot = bootstrap(comp, tables, dec.reset_index(drop=True), gi, B, seed, first_wp, log)
        out = pd.concat([out, bands(vals, rec.reset_index(drop=True), boot)], axis=1)
    out.attrs["license"] = LICENSE_NOTE
    return out, boot
