"""The pregame lineup: who is expected to play, with a probability (context/ml-m5-method.md, section 4).

Layers, applied in order to every player on a team's depth chart for the week
(offense and defense ranks 1-3, plus the place kicker):

1. Depth chart (2005+, week labels already shifted, 2004 dropped; see
   `data.load_depth_charts`): lists the expected starters (rank 1) and backups.
2. Weekly roster status: anything other than ACT or INA (reserve/IR, PUP,
   suspended, cut, practice squad, ...) means he cannot play: p = 0. ONLY in
   seasons where the statuses are week-accurate (`roster_status_seasons`:
   2016+ in nflverse). Before 2016 the weekly "status" is effectively the
   season-end status (future information), so the layer is switched off.
3. Final injury report status (2009+): Out, Doubtful, Questionable, Probable
   (until 2015), or listed on the practice report without a game status.
   Each status maps to a probability of playing, estimated by position group
   from 2009-2015 only (`estimate_status_probs`).
4. Game-day inactives (INA on the weekly roster; 2020+, partial in 2019):
   p = 0. Skipped when `friday_only=True` (decision M5-D2: the Sunday-morning
   run cannot know the inactive list for early games; ladder step B3f).

Every layer is a pregame fact for the game's own week (decision M5-D2); nothing
from later weeks is read. `binary=True` rounds each probability to 0 or 1
(ladder step B5: a Questionable player counts as fully in, Doubtful as out).

Expected share of plays (`expected_shares`): a player's recency-weighted average
usage-based share over his earlier games (credit.player_games), shrunk toward
the average share of players at his group and depth rank in earlier seasons.
A starting lineman counts as 1 (linemen have no usage stats); backup linemen 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.opponent_adjust import decay_weights, week_asof, week_ordinals
from . import positions as pos

STATUSES = ("out", "doubtful", "questionable", "probable", "practice", "none")
ESTIMATION_SEASONS = (2009, 2015)
SHRINK_N = 20.0   # pseudo-starters pulling a small (status, group) cell toward the pooled estimate
SHARE_K = 2.0     # pseudo-games of the (group, rank) default in a player's expected share
LINEUP_COLUMNS = ["season", "week", "team", "gsis_id", "group", "side", "rank", "slot", "status", "roster_status",
                  "p_play"]


@dataclass(frozen=True)
class LineupConfig:
    friday_only: bool = False   # skip the game-day inactive layer (B3f)
    binary: bool = False        # round probabilities to 0/1 (B5)


# --------------------------------------------------------------------------- the layers

def _report_status(inj: pd.DataFrame) -> pd.DataFrame:
    """(season, week, gsis_id) -> status category (out/doubtful/questionable/probable/practice)."""
    if inj is None or inj.empty:
        return pd.DataFrame(columns=["season", "week", "gsis_id", "status"])
    r = inj[inj["game_type"] == "REG"][["season", "week", "gsis_id", "report_status"]].copy()
    rs = r["report_status"].astype("string").str.strip().str.lower().fillna("").to_numpy(object)
    r["status"] = np.select([rs == "out", rs == "doubtful", rs == "questionable", rs == "probable"],
                            ["out", "doubtful", "questionable", "probable"], "practice")
    order = {s: i for i, s in enumerate(STATUSES)}
    r["o"] = r["status"].map(order)
    r = r.sort_values("o").drop_duplicates(["season", "week", "gsis_id"], keep="first")
    return r[["season", "week", "gsis_id", "status"]]


def _roster_status(rosters: pd.DataFrame) -> pd.DataFrame:
    """(season, week, gsis_id) -> roster status code (ACT, INA, RES, ...)."""
    if rosters is None or rosters.empty:
        return pd.DataFrame(columns=["season", "week", "gsis_id", "roster_status"])
    r = rosters[rosters["game_type"] == "REG"][["season", "week", "gsis_id", "status"]].copy()
    r["roster_status"] = [x.strip().upper() if isinstance(x, str) and x.strip() else None for x in r["status"]]
    # A player listed twice in a week (traded) keeps the ACT row if there is one.
    r["o"] = (r["roster_status"] != "ACT").astype(int)
    r = r.sort_values("o").drop_duplicates(["season", "week", "gsis_id"], keep="first")
    return r[["season", "week", "gsis_id", "roster_status"]]


def chart_players(depth: pd.DataFrame, max_rank: int = 3) -> pd.DataFrame:
    """One row per (season, week, team, player) from the chart: offense/defense ranks 1..max_rank, rank-1 kicker."""
    d = depth[depth["group"].notna()]
    d = d[(d["side"].isin(["off", "def"]) & (d["rank"] <= max_rank)) | ((d["group"] == "K") & (d["rank"] == 1))]
    d = d.sort_values(["rank"], kind="stable").drop_duplicates(["season", "week", "team", "gsis_id"], keep="first")
    out = d[["season", "week", "team", "gsis_id", "group", "rank", "slot"]].copy()
    out["side"] = out["group"].map(pos.SIDE)
    out["rank"] = out["rank"].astype(int)
    return out.reset_index(drop=True)


def roster_status_seasons(rosters: pd.DataFrame, pbp: pd.DataFrame, max_played: float = 0.02,
                          min_rows: int = 50) -> list[int]:
    """Seasons whose weekly-roster statuses are week-accurate (a fixed data-quality table, like the era gaps).

    Test: among REG player-weeks marked RES (reserve/IR) or CUT, the share
    whose player is credited on any play of his team's game that same week.
    A week-accurate status almost never has the player playing; a status
    copied from the end of the season often does. In nflverse, 2002-2015
    fail badly (35-45% of "RES" player-weeks played that week: the status is
    effectively the season-end status, which is FUTURE information), and
    2016+ pass (under 0.5%). Only seasons that pass may use the roster layer.
    """
    id_cols = [c for c in pbp.columns if c.endswith("_player_id") or c == "passer_id"]
    long = pbp[["season", "week"] + id_cols].melt(id_vars=["season", "week"], value_name="gsis_id")
    played = long[["season", "week", "gsis_id"]].dropna().drop_duplicates().assign(pl=1.0)
    r = rosters[(rosters["game_type"] == "REG") & rosters["status"].isin(["RES", "CUT"])][["season", "week", "gsis_id"]]
    x = r.merge(played, on=["season", "week", "gsis_id"], how="left").fillna({"pl": 0.0})
    rate = x.groupby("season")["pl"].agg(["mean", "size"])
    return sorted(int(s) for s, row in rate.iterrows() if row["size"] >= min_rows and row["mean"] <= max_played)


def attach_statuses(chart: pd.DataFrame, injuries: pd.DataFrame | None, rosters: pd.DataFrame | None,
                    roster_seasons=None) -> pd.DataFrame:
    """Chart rows plus `status` (injury report category or "none") and `roster_status` (or NA).

    `roster_seasons` (from `roster_status_seasons`): roster statuses are kept
    only in these seasons; elsewhere they are not week-accurate and are
    dropped (NA), so the roster and inactive layers do nothing there.
    """
    out = chart.merge(_report_status(injuries), on=["season", "week", "gsis_id"], how="left")
    out["status"] = out["status"].fillna("none")
    out = out.merge(_roster_status(rosters), on=["season", "week", "gsis_id"], how="left")
    if roster_seasons is not None:
        out.loc[~out["season"].isin(list(roster_seasons)), "roster_status"] = None
    return out


# --------------------------------------------------------------------------- probabilities

def estimate_status_probs(lineup: pd.DataFrame, played: pd.DataFrame,
                          seasons: tuple[int, int] = ESTIMATION_SEASONS) -> pd.DataFrame:
    """P(plays | status, group) for rank-1 chart players, estimated on `seasons` (2009-2015) only.

    `played` holds (season, week, gsis_id) of every player with recorded usage
    in a game (credit.player_games rows with inv > 0). Recorded usage
    understates playing (a starter can play without a stat), so the estimate is
    the ratio P(usage | status) / P(usage | healthy: status "none", roster ACT),
    capped at 1. Linemen have no usage, so they get the pooled ratio over all
    other groups. Small cells are shrunk toward the pooled ratio. Returns rows
    (status, group, n, p).
    """
    lu = lineup[lineup["season"].between(*seasons) & (lineup["rank"] == 1)]
    lu = lu[lu["roster_status"].isna() | lu["roster_status"].isin(["ACT", "INA"])]
    lu = lu[lu["roster_status"] != "INA"]
    pl = played[["season", "week", "gsis_id"]].drop_duplicates().assign(pl=1.0)
    x = lu.merge(pl, on=["season", "week", "gsis_id"], how="left").fillna({"pl": 0.0})
    x = x[x["group"] != "OL"]
    healthy = x[(x["status"] == "none")].groupby("group")["pl"].mean()
    rows = []
    pooled = {}
    for st in STATUSES:
        s = x[x["status"] == st]
        r = (s.groupby("group")["pl"].mean() / healthy).clip(upper=1.0)
        n = s.groupby("group").size()
        tot_n = n.sum()
        pooled[st] = float((r * n).sum() / tot_n) if tot_n else 1.0
        for g in pos.GROUPS:
            if g == "OL":
                rows.append({"status": st, "group": g, "n": 0, "p": pooled[st]})
                continue
            ng = int(n.get(g, 0))
            pg_ = float(r.get(g, pooled[st])) if ng else pooled[st]
            rows.append({"status": st, "group": g, "n": ng,
                         "p": (ng * pg_ + SHRINK_N * pooled[st]) / (ng + SHRINK_N)})
    out = pd.DataFrame(rows)
    out.loc[out["status"] == "none", "p"] = 1.0
    return out


# Fallback probabilities (context/m5-data-scope.md, 2016-2025 absence rates) for windows with no
# estimation data, e.g. the 2001-2008 tuning seasons, which have no injury reports anyway.
SCOPE_PROBS = {"out": 0.0, "doubtful": 0.01, "questionable": 0.67, "probable": 0.95, "practice": 0.91, "none": 1.0}


def default_probs() -> pd.DataFrame:
    """Status probabilities from the data-scope measurements, the same for every group."""
    return pd.DataFrame([{"status": s, "group": g, "n": 0, "p": p} for s, p in SCOPE_PROBS.items() for g in pos.GROUPS])


def play_probability(lineup: pd.DataFrame, probs: pd.DataFrame, cfg: LineupConfig = LineupConfig()) -> np.ndarray:
    """p_play per lineup row from the layers (roster status, injury report, inactives)."""
    pt = probs.set_index(["status", "group"])["p"]
    p = pt.reindex(pd.MultiIndex.from_frame(lineup[["status", "group"]])).fillna(1.0).to_numpy(float)
    rs = lineup["roster_status"].astype(object).to_numpy()
    known = np.array([isinstance(x, str) for x in rs])
    unavailable = known & ~np.isin(rs.astype(str), ["ACT", "INA"])
    p = np.where(unavailable, 0.0, p)
    if not cfg.friday_only:
        p = np.where(known & (rs.astype(str) == "INA"), 0.0, p)
    if cfg.binary:
        p = (p >= 0.5).astype(float)
    return p


def build_lineups(depth: pd.DataFrame, injuries: pd.DataFrame | None, rosters: pd.DataFrame | None,
                  probs: pd.DataFrame, cfg: LineupConfig = LineupConfig()) -> pd.DataFrame:
    """Pregame lineups (LINEUP_COLUMNS) for every team-week on the chart."""
    lu = attach_statuses(chart_players(depth), injuries, rosters)
    lu["p_play"] = play_probability(lu, probs, cfg)
    return lu[LINEUP_COLUMNS]


def starters_out(lineup: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """(season, week, team) -> number of rank-1 non-QB starters with p_play below `threshold`."""
    s = lineup[(lineup["rank"] == 1) & (lineup["group"] != "QB") & (lineup["group"] != "K")]
    return (s["p_play"] < threshold).groupby([s["season"], s["week"], s["team"]]).sum()


# --------------------------------------------------------------------------- expected shares

def default_shares(pg: pd.DataFrame, lineup: pd.DataFrame, season: int) -> pd.Series:
    """(group, rank) -> mean usage share of chart players who played, from REG seasons before `season`."""
    past = lineup[lineup["season"] < season][["season", "week", "team", "gsis_id", "rank"]]
    u = pg[(pg["season"] < season) & (pg["game_type"] == "REG") & (pg["inv"] > 0)]
    x = u.merge(past, on=["season", "week", "team", "gsis_id"], how="inner")
    out = x.groupby(["group", "rank"])["share"].mean()
    return out


def expected_shares(pg: pd.DataFrame, lineup_rows: pd.DataFrame, sched: pd.DataFrame, half_life: float,
                    defaults: dict[int, pd.Series]) -> np.ndarray:
    """Expected share of team plays for each lineup row, from the player's games before the week's as_of.

    s = (sum w * share + SHARE_K * default(group, rank)) / (sum w + SHARE_K), over his
    earlier games with usage (any team); w = 0.5 ** (weeks_ago / half_life) on the
    game-week timeline. Linemen: 1 at rank 1, else 0.
    """
    asofs, ords = week_asof(sched), week_ordinals(sched)
    used = pg[pg["inv"] > 0].sort_values("kick_ns", kind="stable")
    kick = used["kick_ns"].to_numpy()
    ids = used["gsis_id"].to_numpy(object)
    sh = used["share"].to_numpy(float)
    tord = used["ord"].to_numpy(float)
    seas = used["season"].to_numpy(int)
    out = np.zeros(len(lineup_rows))
    lr = lineup_rows.reset_index(drop=True)
    for (S, W), idx in lr.groupby(["season", "week"]).indices.items():
        end = np.searchsorted(kick, asofs[(int(S), int(W))], side="left")
        w = decay_weights(int(ords[(int(S), int(W))]), tord[:end], int(S), seas[:end], half_life, 1.0)
        players = pd.Index(pd.unique(lr.loc[idx, "gsis_id"]))
        code = players.get_indexer(ids[:end])
        m = code >= 0
        sw = np.bincount(code[m], weights=w[m] * sh[:end][m], minlength=len(players))
        ww = np.bincount(code[m], weights=w[m], minlength=len(players))
        dflt = defaults.get(int(S), pd.Series(dtype=float))
        sub = lr.loc[idx]
        d = dflt.reindex(pd.MultiIndex.from_frame(sub[["group", "rank"]])).to_numpy(float)
        d = np.where(np.isnan(d), np.where(sub["rank"].to_numpy() == 1, 0.8, 0.2), d)
        c = players.get_indexer(sub["gsis_id"])
        s = (sw[c] + SHARE_K * d) / (ww[c] + SHARE_K)
        ol = (sub["group"] == "OL").to_numpy()
        s = np.where(ol, (sub["rank"].to_numpy() == 1).astype(float), s)
        out[idx] = s
    return out


def ol_games(lineup: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """Linemen's on-field rows for games already played: rank-1 OL with share = p_play (pregame, all layers).

    Linemen have no individual usage, so their 'played' rows for the remembered
    lineup and the with/without layer are the pregame lineup itself.
    """
    from ..features.team_efficiency import _utc_ns
    ol = lineup[(lineup["group"] == "OL") & (lineup["rank"] == 1) & (lineup["p_play"] > 0)]
    reg = sched[sched["game_type"] == "REG"]
    parts = []
    for side in ("home", "away"):
        parts.append(pd.DataFrame({"game_id": reg["game_id"].to_numpy(), "season": reg["season"].to_numpy(int),
                                   "week": reg["week"].to_numpy(int),
                                   "team": pos.map_teams(reg[f"{side}_team"], reg["season"]).to_numpy(object),
                                   "kick_ns": _utc_ns(reg["kickoff"])}))
    tg = pd.concat(parts, ignore_index=True)
    out = ol.merge(tg, on=["season", "week", "team"])
    out = out.rename(columns={"p_play": "share"})
    out["side"] = "off"
    ords = week_ordinals(sched)
    out["ord"] = ords.reindex(pd.MultiIndex.from_frame(out[["season", "week"]])).to_numpy()
    return out[["game_id", "season", "week", "kick_ns", "ord", "team", "gsis_id", "group", "side", "share"]]
