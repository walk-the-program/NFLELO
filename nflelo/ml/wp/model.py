"""M7a win-probability model: monotone gradient boosting plus isotonic calibration.

LICENSE: CC BY 4.0. Inputs are nflverse play-by-play (CC BY 4.0) and the A4s
pregame probability (built from play-by-play, schedules and Elo; CC BY). No
participation data and nothing from M6 enters this module, so the WP model and
its outputs stay CC BY (the fourth-down outputs that combine it with M6 are
CC BY-SA; see `nflelo/ml/decisions/fourth.py`). Spec: context/ml-m7-method.md,
section 2.

Target: the team with the ball (posteam) wins the game. Tie games are
EXCLUDED from training and scoring (2-3 a season; a tie is neither a win nor a
loss, and a 0.5 label would need a regression model). Overtime plays are
excluded too (the OT clock and rules differ), as are postseason games (A4s is
fit on REG games only).

Inputs, all from the possession team's point of view and all known at the snap:

    score_diff       posteam score minus defteam score
    game_secs        seconds left in regulation
    half_secs        seconds left in the half
    half2            1 in the second half
    down, ydstogo
    yardline_100     yards to the opponent's goal line
    pos_timeouts, def_timeouts
    receive_2h_ko    1 if posteam receives the second-half kickoff (first half only)
    home             1 if posteam is the home team
    a4s_prob         the pregame A4s probability that posteam wins (walk-forward: for
                     season S, the A4s fit on seasons before S; features as of the week)
    diff_time_ratio  score_diff / exp(-4 * elapsed share)   (the lead matters more late)
    strength_time    logit(a4s_prob) * exp(-4 * elapsed share)  (pregame strength fades)

Market columns (spread, vegas_wp, moneylines, totals) and nflfastR's own `wp`
are NEVER inputs: `check_features` rejects them and a test scrambles them to
prove the predictions do not move. nflfastR `wp` and `vegas_wp` are benchmarks
only (`benchmark_columns`).

Monotone constraints (+1 increasing, -1 decreasing in the posteam win
probability), each a statement that can never be false in football:
    score_diff +1, diff_time_ratio +1   (more points never hurt; for fixed time the
                                        ratio is increasing in score_diff)
    a4s_prob +1, strength_time +1       (a stronger team never has a lower WP; the
                                        time factor is positive)
    yardline_100 -1                     (closer to the opponent's goal is better)
    down -1, ydstogo -1                 (fewer downs left or more to go never helps)
    pos_timeouts +1, def_timeouts -1    (a timeout is an option, never an obligation)
Time remaining, half, home and receive_2h_ko are free: their sign depends on
the score (more time helps the trailing team and hurts the leader).

Fitting (season-ahead, for target season S): an inner model is fit on
2006..S-2 with early stopping on season S-1 (validation log loss), which picks
the number of trees; an isotonic map is fit on the inner model's S-1
predictions; the final model is refit on 2006..S-1 with that number of trees
and the S-1 isotonic map is applied to it. The isotonic map is interpolated
linearly between its block centers (`SmoothIsotonic`), so it is strictly
increasing inside the data range: a plain isotonic step function would map
nearby WPs onto the same flat step and erase the small differences that
fourth-down decisions are made of.

CALIBRATE (dev decision, 2026-10-05): the isotonic step is built and always
reported, but the frozen model is the RAW boosted model. On every dev season
2016-2019 the isotonic map made the season-ahead Brier worse (+0.0007 to
+0.0015) and the ECE worse in three of four; so did a three-season and a
cross-fitted (5-fold by game) isotonic map. A log-loss GBM is close to
calibrated already, and one season of play outcomes is only about 256 game
outcomes, so the map mostly learns that season's noise.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

LICENSE = "CC BY 4.0"
FIRST_TRAIN = 2006
SEED = 20261003

BASE = ["score_diff", "game_secs", "half_secs", "half2", "down", "ydstogo", "yardline_100",
        "pos_timeouts", "def_timeouts", "receive_2h_ko", "home", "a4s_prob"]
DERIVED = ["diff_time_ratio", "strength_time"]
FEATURES = BASE + DERIVED
MONOTONE = {"score_diff": 1, "diff_time_ratio": 1, "a4s_prob": 1, "strength_time": 1, "yardline_100": -1,
            "down": -1, "ydstogo": -1, "pos_timeouts": 1, "def_timeouts": -1}
PARAMS = {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 500, "l2_regularization": 1.0,
          "max_bins": 255}
MAX_ITER = 1000
CALIBRATE = False                 # the frozen model is the raw GBM (see the module docstring)

# Never an input: betting-market information, nflfastR's own win probability / EP, or the game's result.
BANNED_PATTERNS = (r"vegas", r"spread", r"moneyline", r"total_line", r"market", r"odds", r"(^|_)wp($|_)",
                   r"wpa", r"(^|_)epa?($|_)", r"result", r"final", r"(^|_)win", r"_score_post", r"^y$")
_BANNED = re.compile("|".join(BANNED_PATTERNS), re.IGNORECASE)

SNAP_TYPES = ("pass", "run", "punt", "field_goal", "no_play", "qb_kneel", "qb_spike")
PBP_COLUMNS = ["game_id", "play_id", "season", "season_type", "week", "home_team", "away_team", "posteam",
               "defteam", "down", "ydstogo", "yardline_100", "qtr", "half_seconds_remaining",
               "game_seconds_remaining", "score_differential", "posteam_timeouts_remaining",
               "defteam_timeouts_remaining", "play_type", "home_opening_kickoff", "result"]
INFO = ["game_id", "play_id", "season", "week", "qtr", "posteam", "defteam", "play_type"]


def is_banned(name: str) -> bool:
    return bool(_BANNED.search(str(name)))


def check_features(cols: Iterable[str]) -> list[str]:
    """Raise unless every column is a WP feature and none is market / nflfastR-WP / outcome information."""
    cols = list(cols)
    bad = [c for c in cols if is_banned(c)]
    if bad:
        raise ValueError(f"banned market / nflfastR-WP / outcome columns among the WP features: {bad}")
    extra = [c for c in cols if c not in FEATURES]
    if extra:
        raise ValueError(f"columns outside the WP feature list: {extra}")
    return cols


# --------------------------------------------------------------------------- the state table

def elapsed_factor(game_secs) -> np.ndarray:
    """exp(-4 * elapsed share of regulation): 1 at kickoff, about 0.018 at the end."""
    g = np.clip(np.asarray(game_secs, float), 0, 3600)
    return np.exp(-4.0 * (3600.0 - g) / 3600.0)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add diff_time_ratio and strength_time to a frame holding the BASE columns (returns a copy)."""
    out = df.copy()
    f = elapsed_factor(out["game_secs"])
    out["diff_time_ratio"] = out["score_diff"].to_numpy(float) / f
    p = np.clip(out["a4s_prob"].to_numpy(float), 1e-4, 1 - 1e-4)
    out["strength_time"] = np.log(p / (1 - p)) * f
    return out


def opening_receiver(pbp: pd.DataFrame) -> pd.Series:
    """Per game: the team that received the opening kickoff (nflverse `home_opening_kickoff` = 1 means home)."""
    g = pbp.drop_duplicates("game_id").set_index("game_id")
    hok = g["home_opening_kickoff"]
    return pd.Series(np.where(hok == 1, g["home_team"], np.where(hok == 0, g["away_team"], None)), index=g.index)


def state_table(pbp: pd.DataFrame, a4s_home: pd.Series) -> pd.DataFrame:
    """One row per REG regulation snap with a possession team, from the possession team's point of view.

    pbp       play-by-play with PBP_COLUMNS (market columns are not needed and not read)
    a4s_home  pregame A4s probability that the HOME team wins, indexed by game_id

    Returns INFO + BASE + DERIVED + `y` (posteam won; ties excluded). Plays from games
    without an A4s probability are dropped.
    """
    p = pbp[(pbp["season_type"] == "REG") & (pbp["qtr"] <= 4) & pbp["posteam"].notna() & pbp["down"].notna()
            & pbp["yardline_100"].notna() & pbp["score_differential"].notna()
            & pbp["play_type"].isin(SNAP_TYPES)]
    p = p[p["game_id"].isin(a4s_home.index)]
    p = p[p["result"].notna() & (p["result"] != 0)]
    home = (p["posteam"] == p["home_team"]).to_numpy()
    out = p[INFO].copy()
    out["score_diff"] = p["score_differential"].to_numpy(float)
    out["game_secs"] = p["game_seconds_remaining"].to_numpy(float)
    out["half_secs"] = p["half_seconds_remaining"].to_numpy(float)
    out["half2"] = (p["qtr"] >= 3).astype(float).to_numpy()
    out["down"] = p["down"].to_numpy(float)
    out["ydstogo"] = p["ydstogo"].to_numpy(float)
    out["yardline_100"] = p["yardline_100"].to_numpy(float)
    out["pos_timeouts"] = p["posteam_timeouts_remaining"].fillna(3).to_numpy(float)
    out["def_timeouts"] = p["defteam_timeouts_remaining"].fillna(3).to_numpy(float)
    rec = opening_receiver(pbp).reindex(p["game_id"]).to_numpy(object)
    out["receive_2h_ko"] = ((p["qtr"] <= 2).to_numpy() & (p["posteam"].to_numpy(object) != rec)
                            & pd.notna(rec)).astype(float)
    out["home"] = home.astype(float)
    ph = a4s_home.reindex(p["game_id"]).to_numpy(float)
    out["a4s_prob"] = np.where(home, ph, 1 - ph)
    out = add_derived(out)
    res = p["result"].to_numpy(float)
    out["y"] = np.where(home, res > 0, res < 0).astype(float)
    return out.reset_index(drop=True)


def benchmark_columns(pbp_market: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """nflfastR `wp` and `vegas_wp` for the table's plays (BENCHMARKS ONLY; never a feature).

    `pbp_market` is play-by-play loaded with keep_market=True; only game_id, play_id, wp and
    vegas_wp are read. Returns a frame aligned to `table` with columns bench_wp, bench_vegas_wp.
    """
    b = pbp_market[["game_id", "play_id", "wp", "vegas_wp"]].drop_duplicates(["game_id", "play_id"])
    j = table[["game_id", "play_id"]].merge(b, on=["game_id", "play_id"], how="left")
    return pd.DataFrame({"bench_wp": j["wp"].to_numpy(float), "bench_vegas_wp": j["vegas_wp"].to_numpy(float)},
                        index=table.index)


# --------------------------------------------------------------------------- the model

@dataclass
class SmoothIsotonic:
    """Isotonic calibration, interpolated linearly between block centers (strictly increasing in range)."""
    x: np.ndarray
    y: np.ndarray

    @classmethod
    def fit(cls, raw: np.ndarray, y: np.ndarray) -> "SmoothIsotonic":
        raw, y = np.asarray(raw, float), np.asarray(y, float)
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, y)
        fitted = iso.predict(raw)
        blocks = pd.DataFrame({"raw": raw, "cal": np.round(fitted, 12)}).groupby("cal", sort=True)["raw"].mean()
        xs, ys = blocks.to_numpy(float), blocks.index.to_numpy(float)
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        keep = np.concatenate([[True], np.diff(xs) > 0])           # centers are increasing by construction
        xs, ys = xs[keep], ys[keep]
        xs = np.concatenate([[0.0], xs, [1.0]])
        ys = np.concatenate([[min(0.0, ys[0])], ys, [max(1.0, ys[-1])]])
        return cls(xs, ys)

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        return np.clip(np.interp(np.asarray(raw, float), self.x, self.y), 1e-6, 1 - 1e-6)


def monotone_vector(features=FEATURES) -> list[int]:
    return [MONOTONE.get(f, 0) for f in features]


def _clf(max_iter: int, early: bool) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=max_iter, early_stopping=early, n_iter_no_change=20,
                                          scoring="loss", monotonic_cst=monotone_vector(), random_state=SEED,
                                          **PARAMS)


@dataclass
class WPModel:
    gbm: HistGradientBoostingClassifier
    iso: SmoothIsotonic | None
    info: dict = field(default_factory=dict)

    def predict_raw(self, states: pd.DataFrame) -> np.ndarray:
        X = states[check_features(FEATURES)].to_numpy(float)
        return self.gbm.predict_proba(X)[:, 1]

    def predict(self, states: pd.DataFrame) -> np.ndarray:
        r = self.predict_raw(states)
        return self.iso(r) if self.iso is not None else r


def _xy(t: pd.DataFrame):
    return t[check_features(FEATURES)].to_numpy(float), t["y"].to_numpy(int)


def fit_final(train: pd.DataFrame, n_iter: int, iso: SmoothIsotonic | None, sample_weight=None) -> WPModel:
    """Refit with a fixed number of trees (the season-ahead refit and the bootstrap replicates)."""
    t0 = time.perf_counter()
    X, y = _xy(train)
    m = _clf(int(n_iter), False).fit(X, y, sample_weight=sample_weight)
    return WPModel(m, iso, {"n_iter": int(n_iter), "n_train": int(len(train)), "seconds": time.perf_counter() - t0})


def fit_season_ahead(table: pd.DataFrame, S: int, first: int = FIRST_TRAIN, calibrate: bool = True) -> WPModel:
    """WP model for season S: trees chosen by early stopping on S-1, isotonic on S-1, refit on first..S-1.

    `calibrate=False` returns the same GBM without the isotonic map (`without_calibration` drops it later)."""
    t0 = time.perf_counter()
    inner = table[table["season"].between(first, S - 2)]
    val = table[table["season"] == S - 1]
    if inner.empty or val.empty:
        raise ValueError(f"need training seasons {first}..{S - 2} and validation season {S - 1}")
    Xi, yi = _xy(inner)
    Xv, yv = _xy(val)
    m_in = _clf(MAX_ITER, True).fit(Xi, yi, X_val=Xv, y_val=yv)
    n_iter = int(m_in.n_iter_)
    raw_v = m_in.predict_proba(Xv)[:, 1]
    iso = SmoothIsotonic.fit(raw_v, yv) if calibrate else None
    final = fit_final(table[table["season"].between(first, S - 1)], n_iter, iso)
    final.info.update({"season": int(S), "trained_on": [int(first), int(S - 1)], "iso_season": int(S - 1),
                       "inner_n_train": int(len(inner)), "val_n": int(len(val)),
                       "total_seconds": time.perf_counter() - t0, "params": dict(PARAMS), "calibrated": bool(calibrate)})
    return final


def without_calibration(m: WPModel) -> WPModel:
    """The same fitted GBM with the isotonic map removed (the frozen M7a WP model when CALIBRATE is False)."""
    return WPModel(m.gbm, None, {**m.info, "calibrated": False})
