"""Point-in-time rule.

A game's prediction time, `as_of`, is the earliest kickoff among the REG and
POST games in the same (season, week). That matches the Wednesday pipeline:
week N is predicted once, before the week's first game, using everything
through week N-1. A Thursday game's result is therefore never used to predict
that Sunday's games.

Kickoff = `gameday` + `gametime` read as America/New_York wall-clock time. A
missing `gametime` falls back to 00:00 ET on game day, which is the
conservative choice: it makes the game look earlier, so fewer things count as
"known before it". (Every 1999 game lacks a gametime in nflverse.)

Usage rule everywhere in nflelo.ml: a feature for a game may only use rows
whose game kickoff is strictly before that game's `as_of`.

`leakage_check` turns the rule into a test: rebuild features after scrambling
everything at or after each game's `as_of`; an honest builder gives identical
features.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

TZ = "America/New_York"
# nflverse game_type values. Postseason rounds each have their own week number.
REG_POST_TYPES = frozenset({"REG", "WC", "DIV", "CON", "SB", "POST"})


def kickoff_times(sched: pd.DataFrame) -> pd.Series:
    """Timezone-aware kickoff (ET) for each schedule row."""
    day = pd.to_datetime(sched["gameday"]).dt.normalize()
    t = sched["gametime"].astype("string").str.strip()
    t = t.where(t.str.match(r"^\d{1,2}:\d{2}$", na=False), "00:00")  # missing or malformed -> midnight
    hhmm = t.str.split(":", expand=True).astype(int)
    naive = day + pd.to_timedelta(hhmm[0], unit="h") + pd.to_timedelta(hhmm[1], unit="m")
    return naive.dt.tz_localize(TZ, ambiguous=True, nonexistent="shift_forward")


def add_asof(sched: pd.DataFrame) -> pd.DataFrame:
    """Add `kickoff` and `as_of` columns to a schedule frame (needs season, week, game_type, gameday, gametime).

    `as_of` is the earliest REG/POST kickoff in the row's (season, week). Rows of
    any other game type (nflverse has none today) get the as_of of their week,
    or their own kickoff if the week has no REG/POST game.
    """
    out = sched.copy()
    out["kickoff"] = kickoff_times(out)
    gt = out["game_type"] if "game_type" in out.columns else pd.Series("REG", index=out.index)
    eligible = out[gt.isin(REG_POST_TYPES)]
    first = eligible.groupby(["season", "week"])["kickoff"].min()
    keys = pd.MultiIndex.from_frame(out[["season", "week"]])
    out["as_of"] = first.reindex(keys).to_numpy()
    out["as_of"] = out["as_of"].fillna(out["kickoff"])
    out["as_of"] = out[["as_of", "kickoff"]].min(axis=1)  # never later than the game itself
    return out


def attach_kickoff(plays: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """Add each play's game `kickoff` (looked up by game_id). Plays from unknown games raise."""
    k = sched.drop_duplicates("game_id").set_index("game_id")["kickoff"]
    out = plays.copy()
    out["kickoff"] = out["game_id"].map(k)
    missing = out.loc[out["kickoff"].isna(), "game_id"].unique()
    if len(missing):
        raise ValueError(f"{len(missing)} play game_ids are not in the schedule, e.g. {list(missing[:3])}")
    return out


def before(df: pd.DataFrame, as_of, col: str = "kickoff") -> pd.DataFrame:
    """Rows whose `col` (a kickoff time) is strictly before `as_of`."""
    return df[df[col] < pd.Timestamp(as_of)]


def games_before(games: pd.DataFrame, as_of) -> pd.DataFrame:
    """Games (with a `kickoff` column) that were already played when `as_of` arrived."""
    return before(games, as_of, "kickoff")


# --------------------------------------------------------------------------- leakage check

# Numeric play-level columns that carry outcome information and get scrambled.
_PLAY_NOISE_COLS = ("epa", "wp", "success", "yards_gained", "wpa")
_SCHED_NOISE_COLS = ("home_score", "away_score", "result", "total")
# Who threw the ball on late plays is outcome information too; shuffled among late plays.
_PLAY_ID_COLS = ("passer_id", "passer_player_id")
# The schedule's starting QBs. Kept by default: decision M3-D1 treats the starter's
# identity as known before kickoff. `scramble_starters=True` removes that exception.
_STARTER_COLS = ("home_qb_id", "away_qb_id")


def corrupt_from(pbp: pd.DataFrame, sched: pd.DataFrame, as_of, seed: int = 0,
                 scramble_starters: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copies of (pbp, sched) with everything from games kicking off at or after `as_of` scrambled.

    Play outcomes (EPA, win probability, success, yards) are replaced with
    noise, pass and run labels are shuffled, passer IDs are shuffled among the
    late plays, and scores are randomized. With `scramble_starters`, the
    schedule's starting-QB IDs of late games are shuffled as well. Games
    before `as_of` are untouched. `sched` must have a `kickoff` column.
    """
    rng = np.random.default_rng(seed)
    cutoff = pd.Timestamp(as_of)
    late_games = set(sched.loc[sched["kickoff"] >= cutoff, "game_id"])
    p = pbp.copy()
    m = p["game_id"].isin(late_games).to_numpy()
    k = int(m.sum())
    if k:
        for c in _PLAY_NOISE_COLS:
            if c in p.columns:
                if c == "wp":
                    vals = rng.uniform(0, 1, k)
                elif c == "success":
                    vals = rng.integers(0, 2, k).astype(float)
                else:
                    vals = rng.normal(0, 5, k)
                p[c] = p[c].astype(float)
                p.loc[m, c] = vals
        if "play_type" in p.columns:
            p.loc[m, "play_type"] = rng.choice(["pass", "run"], k)
        for c in ("pass", "rush"):
            if c in p.columns:
                p[c] = p[c].astype(float)
        if "pass" in p.columns:
            p.loc[m, "pass"] = (p.loc[m, "play_type"] == "pass").astype(float).to_numpy()
        if "rush" in p.columns:
            p.loc[m, "rush"] = (p.loc[m, "play_type"] == "run").astype(float).to_numpy()
        for c in _PLAY_ID_COLS:
            if c in p.columns:
                p[c] = p[c].astype(object)
                p.loc[m, c] = rng.permutation(p.loc[m, c].to_numpy(object))
    s = sched.copy()
    sm = s["game_id"].isin(late_games).to_numpy()
    for c in _SCHED_NOISE_COLS:
        if c in s.columns and sm.any():
            s[c] = s[c].astype(float)
            s.loc[sm, c] = rng.integers(0, 60, int(sm.sum())).astype(float)
    if scramble_starters and sm.any():
        cols = [c for c in _STARTER_COLS if c in s.columns]
        if cols:
            pool = rng.permutation(np.concatenate([s.loc[sm, c].to_numpy(object) for c in cols]))
            for i, c in enumerate(cols):
                s[c] = s[c].astype(object)
                s.loc[sm, c] = pool[i * int(sm.sum()):(i + 1) * int(sm.sum())]
    return p, s


def leakage_check(builder: Callable[..., pd.DataFrame], pbp: pd.DataFrame, sched: pd.DataFrame,
                  game_ids, seed: int = 0, scramble_starters: bool = False) -> pd.DataFrame:
    """For each game, rebuild features after corrupting all data at or after its as_of.

    `builder(pbp, sched, games)` must return a frame indexed by game_id. Returns
    one row per game: game_id, as_of, the number of feature values that
    changed, and `leak_free` (True when nothing changed). An honest builder is
    leak-free for every game. Starting-QB IDs are kept unless
    `scramble_starters` (see `corrupt_from` and decision M3-D1).
    """
    if "as_of" not in sched.columns:
        sched = add_asof(sched)
    rows = []
    for gid in game_ids:
        game = sched[sched["game_id"] == gid]
        if len(game) != 1:
            raise ValueError(f"game {gid!r} is not in the schedule exactly once")
        as_of = game["as_of"].iloc[0]
        clean = builder(pbp, sched, game).loc[[gid]]
        p2, s2 = corrupt_from(pbp, sched, as_of, seed, scramble_starters)
        dirty = builder(p2, s2, s2[s2["game_id"] == gid]).loc[[gid]]
        a, b = clean.to_numpy(float), dirty.loc[:, clean.columns].to_numpy(float)
        same = np.isclose(a, b, rtol=0, atol=1e-12) | (np.isnan(a) & np.isnan(b))
        rows.append({"game_id": gid, "as_of": as_of, "changed": int((~same).sum()),
                     "leak_free": bool(same.all())})
    return pd.DataFrame(rows)
