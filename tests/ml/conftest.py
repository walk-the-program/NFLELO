"""Small synthetic schedule and play-by-play frames, so the ML tests run offline in seconds.

Two seasons (2022, 2023) of six teams, six REG weeks each plus one playoff
week. Every REG week has a Thursday night game, a Sunday 1 pm game, and a
Monday night game, so the as-of rule (earliest kickoff of the week) matters.
Play-by-play carries betting-market columns on purpose: feature code must
drop them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

TEAMS = ["KC", "BUF", "NE", "MIA", "DAL", "PHI"]
SEASONS = (2022, 2023)
FIRST_THURSDAY = {2022: "2022-09-08", 2023: "2023-09-07"}
SLOTS = [(0, "20:15"), (3, "13:00"), (4, "20:15")]  # Thu, Sun, Mon (days after Thursday)
N_REG_WEEKS = 6


def make_schedule() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(1)
    for season in SEASONS:
        thu = pd.Timestamp(FIRST_THURSDAY[season])
        for week in range(1, N_REG_WEEKS + 1):
            order = TEAMS[week % 6:] + TEAMS[:week % 6]
            pairs = [(order[0], order[3]), (order[1], order[4]), (order[2], order[5])]
            for (home, away), (offset, t) in zip(pairs, SLOTS):
                day = thu + pd.Timedelta(days=7 * (week - 1) + offset)
                rows.append((season, "REG", week, day, t, home, away))
        # one playoff game the following week (Saturday)
        day = thu + pd.Timedelta(days=7 * N_REG_WEEKS + 2)
        rows.append((season, "WC", 19, day, "16:30", "KC", "BUF"))
    df = pd.DataFrame(rows, columns=["season", "game_type", "week", "day", "gametime", "home_team", "away_team"])
    df["gameday"] = df["day"].dt.strftime("%Y-%m-%d")
    df["game_id"] = [f"{s}_{w:02d}_{a}_{h}" for s, w, a, h in
                     zip(df["season"], df["week"], df["away_team"], df["home_team"])]
    df["home_score"] = rng.integers(3, 40, len(df)).astype(float)
    df["away_score"] = rng.integers(3, 40, len(df)).astype(float)
    df["result"] = df["home_score"] - df["away_score"]
    df["total"] = df["home_score"] + df["away_score"]
    df["spread_line"] = rng.normal(0, 4, len(df)).round(1)
    df["total_line"] = rng.normal(44, 3, len(df)).round(1)
    df["home_moneyline"] = -150.0
    df["away_moneyline"] = 130.0
    return df.drop(columns="day")[["game_id", "season", "game_type", "week", "gameday", "gametime",
                                   "home_team", "away_team", "home_score", "away_score", "result", "total",
                                   "spread_line", "total_line", "home_moneyline", "away_moneyline"]]


def make_pbp(sched: pd.DataFrame, plays_per_game: int = 120, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for g in sched.itertuples(index=False):
        n = plays_per_game
        home_has = np.arange(n) % 2 == 0
        play_type = rng.choice(["pass", "run", "no_play", "punt", "qb_kneel"], n, p=[0.5, 0.35, 0.06, 0.06, 0.03])
        scramble = (play_type == "run") & (rng.uniform(size=n) < 0.05)
        epa = rng.normal(0.0, 1.3, n)
        epa[np.isin(play_type, ["qb_kneel"])] = -0.3
        parts.append(pd.DataFrame({
            "game_id": g.game_id, "season": g.season,
            "season_type": "REG" if g.game_type == "REG" else "POST", "week": g.week,
            "home_team": g.home_team, "away_team": g.away_team,
            "posteam": np.where(home_has, g.home_team, g.away_team),
            "defteam": np.where(home_has, g.away_team, g.home_team),
            "play_type": play_type,
            "pass": ((play_type == "pass") | scramble).astype(float),
            "rush": ((play_type == "run") & ~scramble).astype(float),
            "epa": epa,
            "success": (epa > 0).astype(float),
            "wp": rng.uniform(0.01, 0.99, n),
            "two_point_attempt": (rng.uniform(size=n) < 0.01).astype(float),
            "yards_gained": rng.integers(-5, 30, n).astype(float),
            "spread_line": g.spread_line, "total_line": g.total_line,
            "vegas_wp": rng.uniform(0, 1, n), "vegas_home_wp": rng.uniform(0, 1, n),
        }))
    return pd.concat(parts, ignore_index=True)


@pytest.fixture(scope="session")
def synth_sched() -> pd.DataFrame:
    return make_schedule()


@pytest.fixture(scope="session")
def synth_pbp(synth_sched) -> pd.DataFrame:
    return make_pbp(synth_sched)
