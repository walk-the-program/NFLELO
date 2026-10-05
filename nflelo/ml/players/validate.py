"""Validation against participation data (CC BY-SA). VALIDATION ONLY (decision M5-D1 (a)).

This is the only module allowed to read participation. Nothing here feeds a
model: it measures how good the CC BY estimates are.

- `onfield_shares`: each player's actual share of his team's clean scrimmage
  plays (same plays as credit.py), from participation.
- `share_errors`: usage-based playing-time estimates (credit.player_games
  `share`, and lineup.ol_games for linemen) against those actual shares, by group.
- `absence_table`: rank-1 depth-chart starters, pregame flags (p_play < 0.5)
  against actual absences (zero snaps), by group, with and without inactives.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import team_efficiency as te
from . import positions as pos


def onfield_shares(participation: pd.DataFrame, pbp: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """(game_id, team, gsis_id, side) -> plays on field and share of the team's clean scrimmage plays (REG)."""
    plays = te.clean_plays(pbp, te.EfficiencyConfig())
    reg = set(sched.loc[sched["game_type"] == "REG", "game_id"])
    plays = plays[plays["game_id"].isin(reg)][["game_id", "play_id", "posteam", "defteam"]]
    part = participation.rename(columns={"nflverse_game_id": "game_id"})
    j = plays.merge(part[["game_id", "play_id", "offense_players", "defense_players"]], on=["game_id", "play_id"])
    covered = set(j["game_id"])
    season = j["game_id"].str.slice(0, 4).astype(int)
    rows = []
    for side, team_col, col in (("off", "posteam", "offense_players"), ("def", "defteam", "defense_players")):
        t = pd.DataFrame({"game_id": j["game_id"].to_numpy(), "team": pos.map_teams(j[team_col], season).to_numpy(),
                          "pid": j[col].fillna("").str.split(";")})
        tot = t.groupby(["game_id", "team"]).size().rename("team_plays")
        e = t.explode("pid")
        e = e[e["pid"].notna() & (e["pid"] != "")]
        c = e.groupby(["game_id", "team", "pid"]).size().rename("on").reset_index()
        c = c.join(tot, on=["game_id", "team"])
        c["side"] = side
        rows.append(c)
    out = pd.concat(rows, ignore_index=True).rename(columns={"pid": "gsis_id"})
    out["actual"] = out["on"] / out["team_plays"]
    out.attrs["games_covered"] = covered
    return out


def share_errors(est: pd.DataFrame, actual: pd.DataFrame, groups_of: pd.Series) -> pd.DataFrame:
    """By group: n player-games, MAE, bias (estimate - actual), correlation, share of rows with estimate 0 but played.

    `est` has game_id, team, gsis_id, share (usage-based estimate); `actual` from
    `onfield_shares`; both restricted to the same games. Player-games present on
    either side count (a missing side is 0). `groups_of` maps gsis_id -> group.
    """
    games = actual.attrs.get("games_covered", set(actual["game_id"]))
    e = est[est["game_id"].isin(games)][["game_id", "team", "gsis_id", "share"]]
    a = actual[["game_id", "team", "gsis_id", "actual"]]
    m = e.merge(a, on=["game_id", "team", "gsis_id"], how="outer").fillna({"share": 0.0, "actual": 0.0})
    m["group"] = m["gsis_id"].map(groups_of)
    m = m[m["group"].notna() & (m["group"] != "K")]
    rows = []
    for g, x in m.groupby("group"):
        played = x["actual"] > 0
        rows.append({"group": g, "n": int(len(x)), "mae": float((x["share"] - x["actual"]).abs().mean()),
                     "bias": float((x["share"] - x["actual"]).mean()),
                     "corr": float(np.corrcoef(x["share"], x["actual"])[0, 1]) if len(x) > 2 else np.nan,
                     "mae_regulars": float((x.loc[x["actual"] >= 0.5, "share"] - x.loc[x["actual"] >= 0.5, "actual"])
                                           .abs().mean()),
                     "zero_est_but_played": float(((x["share"] == 0) & played).sum() / max(played.sum(), 1))})
    return pd.DataFrame(rows)


def absence_table(lineup: pd.DataFrame, actual: pd.DataFrame, sched: pd.DataFrame, threshold: float = 0.5
                  ) -> pd.DataFrame:
    """Rank-1 starters (REG): caught = share of actual absences flagged; false_alarm = share of flags who played.

    `lineup` rows need season, week, team, gsis_id, group, rank, p_play. Actual
    absence: zero plays in the participation data of a covered game.
    """
    games = actual.attrs.get("games_covered", set(actual["game_id"]))
    reg = sched[(sched["game_type"] == "REG") & sched["game_id"].isin(games)]
    tg = pd.concat([pd.DataFrame({"game_id": reg["game_id"], "season": reg["season"], "week": reg["week"],
                                  "team": pos.map_teams(reg[f"{s}_team"], reg["season"])}) for s in ("home", "away")])
    st = lineup[lineup["rank"] == 1].merge(tg, on=["season", "week", "team"])
    on = actual.groupby(["game_id", "gsis_id"])["on"].sum()
    st["absent"] = on.reindex(pd.MultiIndex.from_frame(st[["game_id", "gsis_id"]])).fillna(0).to_numpy() == 0
    st["flag"] = st["p_play"] < threshold
    rows = []
    for g, x in list(st.groupby("group")) + [("ALL", st)]:
        ab, fl = x["absent"], x["flag"]
        rows.append({"group": g, "starters": int(len(x)), "absent": int(ab.sum()),
                     "absent_rate": float(ab.mean()), "flagged": int(fl.sum()),
                     "caught": float((ab & fl).sum() / max(ab.sum(), 1)),
                     "false_alarm": float((fl & ~ab).sum() / max(fl.sum(), 1)),
                     "expected_absent": float((1 - x["p_play"]).sum())})
    return pd.DataFrame(rows)
