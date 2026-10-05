"""M7a kicking components: field-goal make probability and the net-punt distribution.

LICENSE: CC BY 4.0. Built from nflverse play-by-play only (no participation
data, nothing from M6), so these models and their outputs are CC BY. They
become part of CC BY-SA outputs only when `fourth.py` combines them with M6.
Spec: context/ml-m7-method.md, section 3.

Field goals
-----------
One row per field-goal attempt (REG and POST, `play_type == "field_goal"`;
blocked kicks count as misses). Inputs, all known before the snap:
    dist      kick distance = yardline_100 + 18 (the snap is 8 yards back, plus the end zone;
              nflverse's `kick_distance` is yardline_100 + 18 on 99% of attempts and is never read)
    indoor    dome or closed roof
    wind_out  wind speed outdoors (0 indoors; missing = 8 mph)
    cold      max(0, 50 - temperature) / 10 outdoors (0 indoors or missing)
    kval      the kicker's shrunk value: the sum over his attempts in EARLIER GAMES of
              (made - base make probability) / (attempts + K). The base curve is a
              distance-only logistic fit on the training seasons. K is chosen from
              K_FG_GRID by validation log loss on season S-1 (fit on seasons before S-1).
    lg        the league's shrunk residual so far THIS season (all attempts in earlier
              weeks of the same season / (attempts + K_LEAGUE)): kicking levels move from
              year to year (2019 was a bad year), and this follows them as of the week.
Model: logistic regression on dist, (dist - 40)+, (dist - 50)+, indoor, wind_out,
cold, kval, lg with season-decay weights (half-life FG_HALF_LIFE seasons). Beyond
MAX_FG_DIST yards the make probability is 0, and the distance part of the logit is
projected onto a non-increasing curve (`distance_logit`; the real fits are
already monotone, `info["monotone_as_fit"]`). The hinge at 40 and the league term
were chosen on 2010-2017 season-ahead calibration (ECE 0.0152 -> 0.0072; Brier
0.1185 -> 0.1183) before the dev window was scored with them.

Punts
-----
One row per punt (`play_type == "punt"`). The outcome is where the receiving
team starts: its yardline_100 at its next snap (the next play with a
possession team and a down, so returns, fair catches, touchbacks, penalties and
muffs are all included). Classes, ordered from best to worst for the receiver:
    0              punt-return touchdown
    1..21          receiver yardline_100 in R_BINS (1-5, ..., 76-79, 80 (touchback), 81-85, ..., 96-99)
    22             kicking team keeps the ball (muff recovered, or a penalty that gives a first down)
Punts with no next snap (end of half or game) are dropped. Inputs: yardline_100
(the punt spot), indoor, wind_out, and the punter's shrunk value (sum over his
punts in earlier games of (r - expected r at the spot) / (punts + K); the
expected r is the by-spot mean of the training seasons). Model: multiclass
HistGradientBoosting, trees chosen by early stopping on S-1, refit on first..S-1,
training rows weighted by season decay (half-life PUNT_HALF_LIFE = 2 seasons).
Punting improves steadily (punts downed inside the 20 rose from 0.33 of punts in
2010 to 0.40 in 2019, touchbacks fell), and an unweighted model lags the trend.
The half-life was chosen on 2010-2017 season-ahead calibration of P(inside the
20) and P(at or beyond the 40, or a TD) (ECE 0.0167 / 0.0127 unweighted, 0.0142 /
0.0099 at 2 seasons; CRPS 6.016 -> 6.019) before the dev window was scored. Larger
trees, a one-season half-life, an in-season league term and a kernel-smoothed
by-spot distribution were tried on the same seasons and were no better.
Baseline: the by-spot empirical distribution (5-yard spot bins, smoothed toward
all punts with strength BASELINE_K punts). Metric: ordered CRPS in yards
(`crps_ordered`), with class values 0 (TD), the bin midpoints, and 100 (keep).

Kickoffs and clock
------------------
`kickoff_asof` is where the receiving team starts after a kickoff (used after a
made field goal or a touchdown), AS OF the decision: the running mean of the
current season's kickoff starts (yardline_100 at the receiver's first snap;
kicking-team recoveries excluded) from games in earlier weeks, falling back to
the previous season's mean in week 1 or while fewer than KO_MIN kickoffs have
been played. This follows in-season rule changes such as the 2024 dynamic
kickoff and the 2025 touchback move (decided 2026-10-05, before any holdout
number; a fixed previous-season mean was a defect for those seasons). `durations` are the
median clock seconds from a fourth-down snap to the next snap, by option AND
outcome (go: converted without a TD, TD, failed; FG: made, missed; punt), over
the last three training seasons: a converted run keeps the clock moving, a
failure stops it, which matters late in a half.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

LICENSE = "CC BY 4.0"
SEED = 20261003
FIRST_TRAIN = 2006

FG_DIST_OFFSET = 18.0
MISS_SPOT_BACK = 8.0          # a missed kick goes over at the spot of the kick (8 yards behind the line), or the 20
MAX_FG_DIST = 68.0
FG_HALF_LIFE = 4.0
K_FG_GRID = (50.0, 100.0, 200.0, 400.0, 800.0)
K_PUNT_GRID = (10.0, 30.0, 100.0, 300.0)
WIND_FILL = 8.0
K_LEAGUE = 200.0
FG_FEATURES = ["dist", "dist_over40", "dist_over50", "indoor", "wind_out", "cold", "kval", "lg"]

R_BINS = [(lo, lo + 4) for lo in range(1, 76, 5)] + [(76, 79), (80, 80), (81, 85), (86, 90), (91, 95), (96, 99)]
N_PUNT = len(R_BINS) + 2
TD_CLASS, KEEP_CLASS = 0, N_PUNT - 1
PUNT_VALUES = np.array([0.0] + [(a + b) / 2 for a, b in R_BINS] + [100.0])
PUNT_FEATURES = ["yardline_100", "indoor", "wind_out", "pval"]
PUNT_HALF_LIFE = 2.0              # season-decay half-life for the punt model's training rows (None: no decay)
PUNT_PARAMS = {"learning_rate": 0.05, "max_leaf_nodes": 7, "min_samples_leaf": 100, "l2_regularization": 1.0}
BASELINE_K = 50.0

PBP_COLUMNS = ["game_id", "play_id", "season", "season_type", "week", "qtr", "posteam", "defteam", "down",
               "ydstogo", "yardline_100", "play_type", "field_goal_result", "kicker_player_id", "punter_player_id",
               "return_touchdown", "td_team", "roof", "temp", "wind", "game_seconds_remaining",
               "half_seconds_remaining", "own_kickoff_recovery", "touchdown", "fourth_down_converted"]


# --------------------------------------------------------------------------- shared helpers

def game_key(season, week) -> np.ndarray:
    """Game order for as-of player values: one game per team per (season, week)."""
    return np.asarray(season, np.int64) * 100 + np.asarray(week, np.int64)


def weather(df: pd.DataFrame) -> pd.DataFrame:
    roof = df["roof"].astype("string").str.lower()
    indoor = roof.isin(["dome", "closed"]).to_numpy()
    wind = pd.to_numeric(df["wind"], errors="coerce").fillna(WIND_FILL).to_numpy(float)
    temp = pd.to_numeric(df["temp"], errors="coerce").to_numpy(float)
    cold = np.where(np.isnan(temp), 0.0, np.maximum(0.0, 50.0 - temp) / 10.0)
    return pd.DataFrame({"indoor": indoor.astype(float), "wind_out": np.where(indoor, 0.0, wind),
                         "cold": np.where(indoor, 0.0, cold)}, index=df.index)


def next_snap(pbp: pd.DataFrame) -> pd.DataFrame:
    """For every row: the next snap in the same game (a later row with a possession team and a down).

    Returns columns n_posteam, n_yardline_100, n_game_secs, n_qtr aligned to pbp's index (NaN if none).
    """
    p = pbp.sort_values(["game_id", "play_id"])
    snap = p["posteam"].notna() & p["down"].notna()
    cols = {"posteam": "n_posteam", "yardline_100": "n_yardline_100", "game_seconds_remaining": "n_game_secs",
            "qtr": "n_qtr"}
    nxt = p[list(cols)].where(snap).rename(columns=cols)
    nxt = nxt.groupby(p["game_id"]).shift(-1)
    nxt = nxt.groupby(p["game_id"]).bfill()
    return nxt.reindex(pbp.index)


def player_values(events: pd.DataFrame, query: pd.DataFrame, K: float) -> np.ndarray:
    """Shrunk value per query row: sum of the player's residuals in EARLIER games / (count + K).

    events: player, gkey, resid (one row per kick); query: player, gkey. Only events with
    gkey strictly below the query's gkey count (prior games only). Unknown players get 0.
    """
    if len(query) == 0:
        return np.zeros(0)
    e = events.dropna(subset=["player"]).groupby(["player", "gkey"])["resid"].agg(["sum", "count"]).reset_index()
    e = e.sort_values(["player", "gkey"])
    e["csum"] = e.groupby("player")["sum"].cumsum()
    e["ccount"] = e.groupby("player")["count"].cumsum()
    q = query[["player", "gkey"]].copy()
    q["_row"] = np.arange(len(q))
    q["player"] = q["player"].astype(object)
    e["player"] = e["player"].astype(object)
    m = pd.merge_asof(q.dropna(subset=["player"]).sort_values("gkey"), e[["player", "gkey", "csum", "ccount"]]
                      .sort_values("gkey"), on="gkey", by="player", allow_exact_matches=False, direction="backward")
    out = np.zeros(len(q))
    den = (m["ccount"].fillna(0.0) + K).to_numpy(float)
    v = np.divide(m["csum"].fillna(0.0).to_numpy(float), den, out=np.zeros(len(den)), where=den > 0)
    out[m["_row"].to_numpy(int)] = v
    return out


def league_values(events: pd.DataFrame, gkeys, K: float = K_LEAGUE) -> np.ndarray:
    """League-wide shrunk residual so far THIS season: events of the same season in earlier weeks only."""
    e = pd.DataFrame({"player": (events["gkey"].to_numpy(np.int64) // 100), "gkey": events["gkey"].to_numpy(np.int64),
                      "resid": events["resid"].to_numpy(float)})
    g = np.asarray(gkeys, np.int64)
    return player_values(e, pd.DataFrame({"player": g // 100, "gkey": g}), K)


def season_weights(season, S: int, half_life: float) -> np.ndarray:
    return 0.5 ** ((S - 1 - np.asarray(season, float)) / half_life)


def _logloss(y, p) -> float:
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# --------------------------------------------------------------------------- field goals

def fg_table(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per field-goal attempt (REG and POST): inputs (no kval yet) and `made`."""
    p = pbp[(pbp["play_type"] == "field_goal") & pbp["yardline_100"].notna() & pbp["field_goal_result"].notna()]
    out = p[["game_id", "play_id", "season", "week", "season_type", "posteam"]].copy()
    out["player"] = p["kicker_player_id"].astype(object).to_numpy()
    out["gkey"] = game_key(p["season"], p["week"])
    out["yardline_100"] = p["yardline_100"].to_numpy(float)
    out = pd.concat([out, fg_inputs(out["yardline_100"]), weather(p)], axis=1)
    out["made"] = (p["field_goal_result"] == "made").astype(float).to_numpy()
    return out.reset_index(drop=True)


def fg_inputs(yardline_100) -> pd.DataFrame:
    yl = pd.Series(np.asarray(yardline_100, float)) if not isinstance(yardline_100, pd.Series) else yardline_100
    d = yl.to_numpy(float) + FG_DIST_OFFSET
    return pd.DataFrame({"dist": d, "dist_over40": np.maximum(d - 40.0, 0.0), "dist_over50": np.maximum(d - 50.0, 0.0)},
                        index=yl.index)


@dataclass
class FGModel:
    clf: LogisticRegression
    base: LogisticRegression
    K: float
    events: pd.DataFrame              # player, gkey, resid: every attempt the model may draw kicker values from
    info: dict = field(default_factory=dict)

    def value(self, players, gkeys) -> np.ndarray:
        """Shrunk kicker value as of each (player, game): attempts in earlier games only."""
        return player_values(self.events, pd.DataFrame({"player": np.asarray(players, object),
                                                        "gkey": np.asarray(gkeys, np.int64)}), self.K)

    def league(self, gkeys) -> np.ndarray:
        """League FG residual so far this season (earlier weeks only)."""
        return league_values(self.events, gkeys)

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        """Make probability; the distance part of the logit is projected onto a non-increasing curve
        (its running minimum from short to long), so a longer kick is never more likely to be made."""
        X = rows[FG_FEATURES].to_numpy(float)
        coef = dict(zip(FG_FEATURES, self.clf.coef_[0]))
        dist_cols = [c for c in FG_FEATURES if c.startswith("dist")]
        other = [c for c in FG_FEATURES if c not in dist_cols]
        d = rows["dist"].to_numpy(float)
        z = self.clf.intercept_[0] + X[:, [FG_FEATURES.index(c) for c in other]] @ np.array([coef[c] for c in other])
        z = z + distance_logit(coef, d)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.where(d > MAX_FG_DIST, 0.0, p)

    def base_p(self, rows: pd.DataFrame) -> np.ndarray:
        return self.base.predict_proba(rows[["dist", "dist_over50"]].to_numpy(float))[:, 1]


DIST_GRID = np.arange(0.0, 80.5, 0.5)


def distance_logit(coef: dict, dist) -> np.ndarray:
    """The fitted distance part of the FG logit, made non-increasing in distance (running minimum)."""
    g = DIST_GRID
    f = coef.get("dist", 0.0) * g + coef.get("dist_over40", 0.0) * np.maximum(g - 40, 0) \
        + coef.get("dist_over50", 0.0) * np.maximum(g - 50, 0)
    return np.interp(np.asarray(dist, float), g, np.minimum.accumulate(f))


def _fit_base(rows: pd.DataFrame) -> LogisticRegression:
    return LogisticRegression(C=1e6, max_iter=2000).fit(rows[["dist", "dist_over50"]].to_numpy(float),
                                                       rows["made"].to_numpy(int))


def _fit_fg(rows: pd.DataFrame, S: int, sample_weight=None) -> LogisticRegression:
    w = season_weights(rows["season"], S, FG_HALF_LIFE)
    if sample_weight is not None:
        w = w * sample_weight
    return LogisticRegression(C=1e6, max_iter=5000).fit(rows[FG_FEATURES].to_numpy(float), rows["made"].to_numpy(int),
                                                       sample_weight=w)


def kicker_events(fg: pd.DataFrame, base: LogisticRegression, upto_season: int) -> pd.DataFrame:
    """Every attempt through `upto_season` as (player, gkey, made - base make probability)."""
    ev = fg[fg["season"] <= upto_season]
    resid = ev["made"].to_numpy(float) - base.predict_proba(ev[["dist", "dist_over50"]].to_numpy(float))[:, 1]
    return pd.DataFrame({"player": ev["player"].to_numpy(object), "gkey": ev["gkey"].to_numpy(np.int64),
                         "resid": resid})


def with_kval(fg: pd.DataFrame, base: LogisticRegression, K: float, upto_season: int) -> pd.DataFrame:
    """fg rows with `kval`: each row sees only the kicker's attempts in strictly earlier games."""
    out = fg.copy()
    ev = kicker_events(fg, base, upto_season)
    out["kval"] = player_values(ev, out[["player", "gkey"]], K)
    out["lg"] = league_values(ev, out["gkey"])
    return out


def fit_fg_season_ahead(fg: pd.DataFrame, S: int, first: int = FIRST_TRAIN) -> tuple[FGModel, pd.DataFrame]:
    """FG model for season S. Returns (model, fg rows of every season <= S with the model's `kval`).

    Tuning: base curve on first..S-2, each K in K_FG_GRID fit on first..S-2 and scored on S-1.
    Final: base on first..S-1, chosen K, fit on first..S-1. Season-S rows only read attempts from
    earlier games (player_values) and outcomes of seasons <= S-1 never feed S's own rows beyond that.
    """
    t0 = time.perf_counter()
    fg = fg[fg["season"] <= S]
    tr_in = fg[fg["season"].between(first, S - 2)]
    base_in = _fit_base(tr_in)
    scores = {}
    for K in K_FG_GRID:
        x = with_kval(fg[fg["season"] <= S - 1], base_in, K, S - 1)
        inner, val = x[x["season"].between(first, S - 2)], x[x["season"] == S - 1]
        m = _fit_fg(inner, S - 1)
        scores[K] = _logloss(val["made"], m.predict_proba(val[FG_FEATURES].to_numpy(float))[:, 1])
    K = min(scores, key=scores.get)
    base = _fit_base(fg[fg["season"].between(first, S - 1)])
    x = with_kval(fg, base, K, S)
    events = kicker_events(fg, base, S)
    clf = _fit_fg(x[x["season"].between(first, S - 1)], S)
    info = {"season": int(S), "K": float(K), "val_logloss": {str(k): v for k, v in scores.items()},
            "coef": dict(zip(FG_FEATURES, map(float, clf.coef_[0]))), "intercept": float(clf.intercept_[0]),
            "n_train": int(x["season"].between(first, S - 1).sum()), "seconds": time.perf_counter() - t0}
    c = info["coef"]
    info["monotone_as_fit"] = bool(c["dist"] < 0 and c["dist"] + c["dist_over40"] < 0
                                   and c["dist"] + c["dist_over40"] + c["dist_over50"] < 0)
    return FGModel(clf, base, K, events, info), x


def refit_fg(model: FGModel, x: pd.DataFrame, S: int, first: int, sample_weight) -> FGModel:
    """Bootstrap replicate: same K and kval inputs, coefficients refit with game-resampling weights."""
    tr = x["season"].between(first, S - 1).to_numpy()
    clf = _fit_fg(x[tr], S, sample_weight=sample_weight[tr])
    return FGModel(clf, model.base, model.K, model.events, {"replicate": True})


# --------------------------------------------------------------------------- punts

def r_to_class(r) -> np.ndarray:
    r = np.asarray(r, float)
    out = np.full(len(r), -1, int)
    for i, (a, b) in enumerate(R_BINS):
        out[(r >= a) & (r <= b)] = i + 1
    return out


def punt_table(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per punt with a next snap: inputs (no pval yet), the class, r, and the kept-ball spot."""
    nxt = next_snap(pbp)
    p = pbp[(pbp["play_type"] == "punt") & pbp["yardline_100"].notna()]
    n = nxt.loc[p.index]
    rtd = ((p["return_touchdown"].fillna(0) == 1) & (p["td_team"] == p["defteam"])).to_numpy()
    rec = (n["n_posteam"] == p["defteam"]).to_numpy()
    keep = (n["n_posteam"] == p["posteam"]).to_numpy() & ~rtd
    r = np.where(rec, n["n_yardline_100"].to_numpy(float), np.nan)
    cls = np.where(rtd, TD_CLASS, np.where(keep, KEEP_CLASS, r_to_class(np.nan_to_num(r, nan=-1))))
    ok = rtd | keep | (rec & (cls > 0))
    out = p[["game_id", "play_id", "season", "week", "season_type", "posteam", "defteam"]].copy()
    out["player"] = p["punter_player_id"].astype(object).to_numpy()
    out["gkey"] = game_key(p["season"], p["week"])
    out["yardline_100"] = p["yardline_100"].to_numpy(float)
    out = pd.concat([out, weather(p)[["indoor", "wind_out"]]], axis=1)
    out["cls"] = cls
    out["r"] = r
    out["keep_yl"] = np.where(keep, n["n_yardline_100"].to_numpy(float), np.nan)
    return out[ok].reset_index(drop=True)


def spot_bin(yardline_100) -> np.ndarray:
    return np.clip((np.asarray(yardline_100, float) - 1) // 5, 0, 19).astype(int)


def expected_r(train: pd.DataFrame) -> np.ndarray:
    """By-spot (5-yard bins) mean receiver yardline_100 on returned/fair-caught/downed/touchback punts."""
    t = train[train["cls"].between(1, N_PUNT - 2)]
    g = t.groupby(spot_bin(t["yardline_100"]))["r"].mean()
    return g.reindex(range(20)).interpolate(limit_direction="both").to_numpy(float)


def punter_events(pt: pd.DataFrame, exp_r: np.ndarray, upto_season: int) -> pd.DataFrame:
    """Every returned / fair-caught / downed / touchback punt through `upto_season` as (player, gkey, r - expected r)."""
    ev = pt[(pt["season"] <= upto_season) & pt["cls"].between(1, N_PUNT - 2)]
    resid = ev["r"].to_numpy(float) - exp_r[spot_bin(ev["yardline_100"])]
    return pd.DataFrame({"player": ev["player"].to_numpy(object), "gkey": ev["gkey"].to_numpy(np.int64),
                         "resid": resid})


def with_pval(pt: pd.DataFrame, exp_r: np.ndarray, K: float, upto_season: int) -> pd.DataFrame:
    out = pt.copy()
    ev = punter_events(pt, exp_r, upto_season)
    out["pval"] = player_values(ev, out[["player", "gkey"]], K)
    out["lgp"] = league_values(ev, out["gkey"])
    return out


def punt_weights(season, S: int) -> np.ndarray | None:
    return None if PUNT_HALF_LIFE is None else season_weights(season, S, PUNT_HALF_LIFE)


def crps_ordered(P: np.ndarray, cls: np.ndarray, values: np.ndarray = PUNT_VALUES) -> np.ndarray:
    """Per-row CRPS for ordered classes with representative values (sum of (F_k - H_k)^2 * gap_k)."""
    P = np.asarray(P, float)
    F = np.cumsum(P, axis=1)[:, :-1]
    H = (np.asarray(cls, int)[:, None] <= np.arange(P.shape[1] - 1)[None, :]).astype(float)
    return ((F - H) ** 2 * np.diff(values)[None, :]).sum(axis=1)


def _punt_clf(max_iter: int, early: bool) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=max_iter, early_stopping=early, n_iter_no_change=20,
                                          scoring="loss", random_state=SEED, **PUNT_PARAMS)


@dataclass
class PuntModel:
    clf: HistGradientBoostingClassifier
    exp_r: np.ndarray
    K: float
    keep_yl: float                    # mean kicking-team yardline_100 after it keeps the ball
    class_r: np.ndarray               # representative receiver yardline_100 per class (training means)
    events: pd.DataFrame              # player, gkey, resid: every punt the model may draw punter values from
    info: dict = field(default_factory=dict)

    def value(self, players, gkeys) -> np.ndarray:
        """Shrunk punter value as of each (player, game): punts in earlier games only."""
        return player_values(self.events, pd.DataFrame({"player": np.asarray(players, object),
                                                        "gkey": np.asarray(gkeys, np.int64)}), self.K)

    def league(self, gkeys) -> np.ndarray:
        """League net-punt residual so far this season (earlier weeks only)."""
        return league_values(self.events, gkeys)

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        P = np.zeros((len(rows), N_PUNT))
        P[:, self.clf.classes_.astype(int)] = self.clf.predict_proba(rows[PUNT_FEATURES].to_numpy(float))
        return P


def class_values(train: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Mean receiver yardline_100 per r class (bin midpoint if unseen) and the mean kept-ball spot."""
    g = train[train["cls"].between(1, N_PUNT - 2)].groupby("cls")["r"].mean()
    cr = PUNT_VALUES.copy()
    for c, v in g.items():
        cr[int(c)] = v
    keep = train.loc[train["cls"] == KEEP_CLASS, "keep_yl"].mean()
    return cr, float(keep) if np.isfinite(keep) else 50.0


def fit_punt_season_ahead(pt: pd.DataFrame, S: int, first: int = FIRST_TRAIN) -> tuple[PuntModel, pd.DataFrame]:
    """Punt model for season S (K and trees chosen on S-1, refit on first..S-1). Returns (model, rows <= S with pval)."""
    t0 = time.perf_counter()
    pt = pt[pt["season"] <= S]
    exp_in = expected_r(pt[pt["season"].between(first, S - 2)])
    scores, iters = {}, {}
    for K in K_PUNT_GRID:
        x = with_pval(pt[pt["season"] <= S - 1], exp_in, K, S - 1)
        inner, val = x[x["season"].between(first, S - 2)], x[x["season"] == S - 1]
        vv = val[val["cls"].isin(np.unique(inner["cls"]))]
        m = _punt_clf(1000, True).fit(inner[PUNT_FEATURES].to_numpy(float), inner["cls"].to_numpy(int),
                                      sample_weight=punt_weights(inner["season"], S - 1),
                                      X_val=vv[PUNT_FEATURES].to_numpy(float), y_val=vv["cls"].to_numpy(int))
        P = np.zeros((len(val), N_PUNT))
        P[:, m.classes_.astype(int)] = m.predict_proba(val[PUNT_FEATURES].to_numpy(float))
        scores[K] = float(crps_ordered(P, val["cls"].to_numpy(int)).mean())
        iters[K] = int(m.n_iter_)
    K = min(scores, key=scores.get)
    exp_r = expected_r(pt[pt["season"].between(first, S - 1)])
    x = with_pval(pt, exp_r, K, S)
    tr = x[x["season"].between(first, S - 1)]
    clf = _punt_clf(iters[K], False).fit(tr[PUNT_FEATURES].to_numpy(float), tr["cls"].to_numpy(int),
                                         sample_weight=punt_weights(tr["season"], S))
    cr, keep = class_values(tr)
    info = {"season": int(S), "K": float(K), "val_crps": {str(k): v for k, v in scores.items()}, "n_iter": iters[K],
            "n_train": int(len(tr)), "seconds": time.perf_counter() - t0}
    return PuntModel(clf, exp_r, K, keep, cr, punter_events(pt, exp_r, S), info), x


def refit_punt(model: PuntModel, x: pd.DataFrame, S: int, first: int, sample_weight) -> PuntModel:
    tr = x["season"].between(first, S - 1).to_numpy()
    w = punt_weights(x.loc[tr, "season"], S)
    sw = sample_weight[tr] if w is None else sample_weight[tr] * w
    clf = _punt_clf(model.info["n_iter"], False).fit(x.loc[tr, PUNT_FEATURES].to_numpy(float),
                                                     x.loc[tr, "cls"].to_numpy(int), sample_weight=sw)
    return PuntModel(clf, model.exp_r, model.K, model.keep_yl, model.class_r, model.events, {"replicate": True})


@dataclass
class PuntBaseline:
    table: np.ndarray                 # (20 spot bins, N_PUNT)

    @classmethod
    def fit(cls, train: pd.DataFrame, k: float = BASELINE_K) -> "PuntBaseline":
        allp = np.bincount(train["cls"].to_numpy(int), minlength=N_PUNT).astype(float)
        allp /= allp.sum()
        sb = spot_bin(train["yardline_100"])
        T = np.zeros((20, N_PUNT))
        for b in range(20):
            c = np.bincount(train["cls"].to_numpy(int)[sb == b], minlength=N_PUNT).astype(float)
            T[b] = (c + k * allp) / (c.sum() + k)
        return cls(T)

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.table[spot_bin(rows["yardline_100"])]


# --------------------------------------------------------------------------- kickoffs and the clock

KO_MIN = 50


def kickoff_table(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per kickoff the receiving team kept: season, gkey, and its start (yardline_100 at its first snap)."""
    nxt = next_snap(pbp)
    k = pbp[pbp["play_type"] == "kickoff"]
    n = nxt.loc[k.index]
    # nflverse puts the RECEIVING team in posteam on kickoffs
    rec = ((n["n_posteam"] == k["posteam"]) & (k["own_kickoff_recovery"].fillna(0) == 0)
           & n["n_yardline_100"].notna()).to_numpy()
    return pd.DataFrame({"season": k["season"].to_numpy(int)[rec],
                         "gkey": game_key(k["season"], k["week"])[rec],
                         "start": n["n_yardline_100"].to_numpy(float)[rec]})


def kickoff_spot(pbp: pd.DataFrame, season: int) -> float:
    """Mean kickoff start over a whole season (the fallback, and descriptive)."""
    t = kickoff_table(pbp)
    return float(t.loc[t["season"] == season, "start"].mean())


def kickoff_asof(ko: pd.DataFrame, gkeys, min_n: int = KO_MIN) -> np.ndarray:
    """Kickoff start as of each game: the mean of this season's kickoffs in EARLIER weeks (gkey strictly
    below), or the previous season's mean in week 1 or while fewer than `min_n` kickoffs have been played."""
    g = np.asarray(gkeys, np.int64)
    out = np.empty(len(g))
    season = g // 100
    by_season = ko.groupby("season")["start"].mean()
    for S in np.unique(season):
        prev = by_season.get(S - 1, np.nan)
        if not np.isfinite(prev):                 # no previous season in the data: the latest earlier one
            earlier = by_season[by_season.index < S]
            prev = float(earlier.iloc[-1]) if len(earlier) else float(ko["start"].mean())
        cur = ko[ko["season"] == S].sort_values("gkey")
        keys = cur["gkey"].to_numpy(np.int64)
        csum = np.concatenate([[0.0], np.cumsum(cur["start"].to_numpy(float))])
        m = season == S
        n_before = np.searchsorted(keys, g[m], side="left")       # kickoffs with gkey < query
        mean = np.divide(csum[n_before], n_before, out=np.zeros(len(n_before)), where=n_before > 0)
        out[m] = np.where(n_before >= min_n, mean, prev)
    return out


DURATION_KEYS = ("go_conv", "go_td", "go_fail", "fg_make", "fg_miss", "punt")


def durations(pbp: pd.DataFrame, seasons) -> dict:
    """Median clock seconds from a regulation fourth-down snap to the next snap (same half), by option and outcome."""
    nxt = next_snap(pbp)
    p = pbp[(pbp["down"] == 4) & (pbp["qtr"] <= 4) & pbp["season"].isin(list(seasons))]
    n = nxt.loc[p.index]
    same_half = ((p["qtr"] <= 2) == (n["n_qtr"] <= 2)) & (n["n_qtr"] <= 4)
    dt = (p["game_seconds_remaining"] - n["n_game_secs"]).where(same_half)
    go = p["play_type"].isin(["run", "pass"])
    td = (p["touchdown"].fillna(0) == 1) & (p["td_team"] == p["posteam"])
    conv = p["fourth_down_converted"].fillna(0) == 1
    made = p["field_goal_result"] == "made"
    key = pd.Series(np.select([go & td, go & conv, go, (p["play_type"] == "field_goal") & made,
                               p["play_type"] == "field_goal", p["play_type"] == "punt"],
                              list(DURATION_KEYS), default=""), index=p.index)
    d = dt[key != ""].groupby(key[key != ""]).median()
    overall = float(dt.median()) if dt.notna().any() else 6.0
    return {k: float(d[k]) if k in d.index and np.isfinite(d[k]) else overall for k in DURATION_KEYS}
