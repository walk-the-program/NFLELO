"""Features "as of now" for the season's unplayed games (M4 Phase 1, context/ml-m4-method.md section 2).

Same three A4s features as the M3 model, built by the same functions:

    elo_logit        (home rating - away rating + HFA) * ln(10) / 400, HFA 0 at a neutral site
    adj_epa_margin   opponent-adjusted EPA margin from `opponent_adjust` (TUNED knobs)
    qb_delta_diff    qb_delta_home - qb_delta_away from `qb` (k = 100)

Which week's information a game gets ("feature week"):
- Let W_next be the first week whose as_of (earliest kickoff) is after `now`.
- A game in an earlier week (the week in progress) keeps its own week: exactly
  the training protocol, information before that week's first kickoff.
- A game in W_next or later uses W_next's state, i.e. everything played so
  far. Nothing about the future is known, so every later week sees the same
  information; the decay weights don't age the data by weeks that haven't
  happened yet.

Elo: ratings are the current ones (a team's rating doesn't change until it
plays). HFA is the online HFA frozen before the feature week's first game when
that game has been played, otherwise the current HFA.

Starting QB: the schedule's `home_qb_id` / `away_qb_id` when nflverse lists
one (source "nflverse"); otherwise the team's starter in its most recent game
that kicked off before `now` (source "last_starter").

Market columns are dropped from every input on entry and the output is checked
for them (decision D3): betting lines can't reach these features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..config import EloConfig
from ..elo import run_elo
from ..teams import franchise
from . import asof as asof_mod
from .data import drop_market_columns
from .features import context
from .features import kalman
from .features import opponent_adjust as oa
from .features import qb
from .features import team_efficiency as te

FEATURES = ["elo_logit", "adj_epa_margin", "qb_delta_diff"]


# --------------------------------------------------------------------------- Elo state

def elo_state(games: pd.DataFrame, season: int, cfg: EloConfig = config.DEFAULT_CONFIG) -> dict:
    """Ratings for `season`'s next games (full precision), current HFA, and the HFA frozen before each played week.

    `games` are the games played so far. If none of them is from `season` yet,
    the off-season regression toward the mean is applied, as Elo does at a
    team's first game of a new season.
    """
    out, state = run_elo(games, cfg)
    hb = context.hfa_before(out, cfg)
    first = pd.Series(np.arange(len(out))).groupby([out["season"].to_numpy(), out["week"].to_numpy()]).min()
    hfa_week = {(int(s), int(w)): float(hb[i]) for (s, w), i in first.items()}
    ratings = dict(state["ratings"])
    if int(out["season"].max()) < int(season):
        ratings = {t: (1.0 - cfg.lam) * r + cfg.lam * cfg.start for t, r in ratings.items()}
    return {"ratings": ratings, "hfa_now": float(state["hfa"]), "hfa_week": hfa_week}


# --------------------------------------------------------------------------- weeks and starters

def next_week(season_sched: pd.DataFrame, now) -> int | None:
    """First week of the season whose as_of is after `now` (None when every week has started)."""
    s = season_sched if "as_of" in season_sched.columns else asof_mod.add_asof(season_sched)
    a = s.groupby("week")["as_of"].min()
    later = a[a > pd.Timestamp(now)]
    return int(later.index.min()) if len(later) else None


def feature_weeks(games: pd.DataFrame, season_sched: pd.DataFrame, now) -> np.ndarray:
    w = games["week"].to_numpy(int)
    nxt = next_week(season_sched, now)
    return w if nxt is None else np.where(w < nxt, w, nxt)


def resolve_starters(sched: pd.DataFrame, games: pd.DataFrame, now) -> pd.DataFrame:
    """Starting QB per game and side, its name, and where it came from. Indexed by game_id."""
    s = sched if "kickoff" in sched.columns else asof_mod.add_asof(sched)
    now = pd.Timestamp(now)
    past = s[s["kickoff"] < now]
    long = []
    for side in ("home", "away"):
        name_col = f"{side}_qb_name"
        long.append(pd.DataFrame({
            "team": [franchise(c, int(y)) for c, y in zip(past[f"{side}_team"], past["season"])],
            "kick": past["kickoff"].to_numpy(), "qb": past[f"{side}_qb_id"].to_numpy(object),
            "name": past[name_col].to_numpy(object) if name_col in past else None}))
    long = pd.concat(long, ignore_index=True)
    long = long[long["qb"].notna()].sort_values("kick", kind="stable")
    last = long.groupby("team").tail(1).set_index("team")

    rows = {}
    for r in games.itertuples():
        rec = {}
        for side in ("home", "away"):
            qid = getattr(r, f"{side}_qb_id", None)
            name = getattr(r, f"{side}_qb_name", None)
            team = franchise(getattr(r, f"{side}_team"), int(r.season))
            if qid is not None and not pd.isna(qid):
                rec.update({f"{side}_qb_id": str(qid), f"{side}_qb_name": None if pd.isna(name) else name,
                            f"{side}_qb_source": "nflverse"})
            elif team in last.index:
                nm = last.at[team, "name"]
                rec.update({f"{side}_qb_id": str(last.at[team, "qb"]), f"{side}_qb_name": None if pd.isna(nm) else nm,
                            f"{side}_qb_source": "last_starter"})
            else:
                rec.update({f"{side}_qb_id": None, f"{side}_qb_name": None, f"{side}_qb_source": "none"})
        rows[r.game_id] = rec
    return pd.DataFrame.from_dict(rows, orient="index")


def game_qb_source(home: str, away: str) -> str:
    return home if home == away else "mixed"


# --------------------------------------------------------------------------- features

def build(pbp: pd.DataFrame, sched: pd.DataFrame, games: pd.DataFrame, now, elo: dict) -> pd.DataFrame:
    """Features for `games` (unplayed rows of `sched`) as of `now`, indexed by game_id.

    `pbp` and `sched` cover every season the QB values and ratings need (1999 on
    for the live model). `elo` is `elo_state(...)` over games played before `now`.
    Returns the three model features, the QB deltas per side, the feature week,
    and the starters with their source.
    """
    sched = drop_market_columns(sched)          # D3: no betting line can reach a feature
    games = drop_market_columns(games)
    if "as_of" not in sched.columns:
        sched = asof_mod.add_asof(sched)
    season = int(games["season"].iloc[0])
    if (games["season"] != season).any():
        raise ValueError("build() takes one season's games at a time")
    season_sched = sched[sched["season"] == season]
    keyed = games[["game_id", "season", "week", "home_team", "away_team"]
                  + [c for c in ("location", "home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name")
                     if c in games.columns]].copy()
    keyed["game_week"] = keyed["week"].to_numpy(int)
    keyed["week"] = feature_weeks(keyed, season_sched, now)
    wk_asof = season_sched.groupby("week")["as_of"].min()
    keyed["as_of"] = wk_asof.reindex(keyed["week"]).to_numpy()

    starters = resolve_starters(sched, games, now)
    for side in ("home", "away"):
        keyed[f"{side}_qb_id"] = starters.loc[keyed["game_id"], f"{side}_qb_id"].to_numpy(object)

    # opponent-adjusted EPA margin at the feature week
    tab = oa.rating_table(pbp, sched, oa.TUNED)
    keys = sorted(set(zip(keyed["season"].astype(int), keyed["week"].astype(int))))
    ratings = oa.compute_ratings(tab, sched, keys, oa.TUNED, ("all",))
    adj = oa.matchup_margin(ratings, keyed, "all")

    # QB deltas: the resolved starter, valued with data before the feature week's as_of
    qd = qb.build_features(pbp, sched, keyed, qb.tuned_config(), starter="actual")

    # Elo log-odds from current ratings and the feature week's HFA
    h = np.array([franchise(c, season) for c in keyed["home_team"]], dtype=object)
    a = np.array([franchise(c, season) for c in keyed["away_team"]], dtype=object)
    missing = sorted({t for t in np.concatenate([h, a]) if t not in elo["ratings"]})
    if missing:
        raise ValueError(f"no Elo rating for {missing}")
    hfa = np.array([elo["hfa_week"].get((season, int(w)), elo["hfa_now"]) for w in keyed["week"]], float)
    neutral = (keyed["location"] == "Neutral").to_numpy() if "location" in keyed else np.zeros(len(keyed), bool)
    diff = (np.array([elo["ratings"][t] for t in h]) - np.array([elo["ratings"][t] for t in a])
            + np.where(neutral, 0.0, hfa))

    out = pd.DataFrame({
        "season": season, "game_week": keyed["game_week"].to_numpy(int), "feature_week": keyed["week"].to_numpy(int),
        "home_team": h, "away_team": a,
        "elo_logit": diff * context.LN10_400, "hfa_used": np.where(neutral, 0.0, hfa),
        "adj_epa_margin": adj,
        "qb_delta_home": qd.loc[keyed["game_id"], "qb_delta_home"].to_numpy(float),
        "qb_delta_away": qd.loc[keyed["game_id"], "qb_delta_away"].to_numpy(float),
    }, index=keyed["game_id"].to_numpy())
    out["qb_delta_diff"] = out["qb_delta_home"] - out["qb_delta_away"]
    out.index.name = "game_id"
    te.assert_no_market_columns(out[FEATURES + ["qb_delta_home", "qb_delta_away"]])
    if out[FEATURES].isna().any().any():
        raise ValueError(f"NaN in live features: {out[FEATURES].isna().sum().to_dict()}")
    st = starters.loc[out.index]
    for c in st.columns:
        out[c] = st[c].to_numpy(object)
    out["qb_source"] = [game_qb_source(x, y) for x, y in zip(out["home_qb_source"], out["away_qb_source"])]
    return out


# --------------------------------------------------------------------------- the shadow model's Kalman feature

SHADOW_FEATURES = ["elo_logit", "kf_margin", "qb_delta_diff"]   # M3b C2d (context/ml.md, "Shadow deployment")


def kalman_asof(pbp: pd.DataFrame, sched: pd.DataFrame, games: pd.DataFrame, now,
                cfg: kalman.KalmanConfig = kalman.TUNED) -> pd.DataFrame:
    """kf_margin and kf_sd for `games` (unplayed rows of `sched`, one season) as of `now`, indexed by game_id.

    Same as-of rule as `build`: each game gets its feature week (its own week
    while that week is in progress, otherwise the first week that has not
    started), and the filter absorbs only completed REG/POST games that kicked
    off before that week's as_of (and before `now`). It then predicts the way
    the filter predicts that week in training (`kalman.predict_at`).
    """
    sched = drop_market_columns(sched)          # D3
    games = drop_market_columns(games)
    if "as_of" not in sched.columns:
        sched = asof_mod.add_asof(sched)
    season = int(games["season"].iloc[0])
    if (games["season"] != season).any():
        raise ValueError("kalman_asof() takes one season's games at a time")
    season_sched = sched[sched["season"] == season]
    fw = feature_weeks(games, season_sched, now)
    wk_asof = season_sched.groupby("week")["as_of"].min()
    ords = oa.week_ordinals(sched)
    gt = kalman.game_table(pbp, sched, cfg)
    now = pd.Timestamp(now)
    parts = []
    for w in sorted(set(fw.tolist())):
        cutoff = min(pd.Timestamp(wk_asof[w]), now)
        parts.append(kalman.predict_at(gt, cfg, games[fw == w], cutoff, season, int(ords[(season, int(w))])))
    out = pd.concat(parts).loc[games["game_id"].to_numpy()]
    te.assert_no_market_columns(out)
    if out.isna().any().any():
        raise ValueError(f"NaN in live Kalman features: {out.isna().sum().to_dict()}")
    return out


def predict(model: dict, feats: pd.DataFrame) -> np.ndarray:
    """Home win probability from a saved logistic model (intercept plus one coefficient per feature)."""
    X = feats[model["features"]].to_numpy(float)
    z = model["intercept"] + X @ np.array([model["coef"][f] for f in model["features"]], float)
    return 1.0 / (1.0 + np.exp(-z))
