"""Team efficiency features, built strictly as of each game's `as_of` time.

For every game and each side (home, away) we report, from play-by-play of games
that kicked off before `as_of`:

    off_epa_all / off_epa_pass / off_epa_rush   offense EPA per play
    off_sr                                      offense success rate
    def_epa_all / def_epa_pass / def_epa_rush   EPA per play allowed
    def_sr                                      success rate allowed
    off_plays / def_plays                       scrimmage plays counted this season
    off_plays_prev / def_plays_prev             scrimmage plays counted from last season

Every rate pools the current season to date with the previous season, the
previous season's plays weighted by `prev_weight` (default 0.5):

    rate = (cur_sum + w * prev_sum) / (cur_n + w * prev_n)

Plays used: real scrimmage snaps, meaning `play_type` is pass or run (this drops
penalties that wipe out the play, kneels, spikes, kicks, and timeouts), the
play has an EPA, and it is not a two-point try. Pass versus rush follows
nflfastR's `pass` and `rush` flags, so scrambles and sacks count as dropbacks.
Garbage time is excluded (win probability below 0.05 or above 0.95; plays with
no win probability are kept). Every play from both REG and POST games counts.
All betting-market columns are dropped before anything else happens
(decision D3). Teams are mapped to franchise IDs so a relocated team keeps its
history.

Design: plays are aggregated to one row per (game, offense) first. That step
looks at one game at a time, so it cannot leak across games. The only place
time matters is `_rolling_sums`, which sums team-games with kickoff < as_of.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...teams import franchise
from .. import asof as asof_mod
from ..data import drop_market_columns

SIDES = ("home", "away")
RATE_NAMES = ("epa_all", "epa_pass", "epa_rush", "sr")
# Accumulated per team-game: plays, EPA sum, pass plays, pass EPA, rush plays, rush EPA, success plays, successes.
# n_all counts every kept snap; a rare snap flagged neither pass nor rush counts only in the "all" rates.
_ACC = ["n_all", "epa_all", "n_pass", "epa_pass", "n_rush", "epa_rush", "n_sr", "sr"]
PLAY_COLUMNS = ["game_id", "posteam", "defteam", "play_type", "pass", "rush", "epa", "success", "wp"]
FORBIDDEN_FEATURE_PATTERN = r"spread|total_line|moneyline|vegas"


@dataclass(frozen=True)
class EfficiencyConfig:
    prev_weight: float = 0.5
    garbage_low: float = 0.05
    garbage_high: float = 0.95

    def to_dict(self) -> dict:
        return {"prev_weight": self.prev_weight, "garbage_low": self.garbage_low,
                "garbage_high": self.garbage_high}


def feature_names() -> list[str]:
    stats = [f"{k}_{r}" for k in ("off", "def") for r in RATE_NAMES]
    counts = [f"{k}_{c}" for k in ("off", "def") for c in ("plays", "plays_prev")]
    return [f"{s}_{n}" for s in SIDES for n in stats + counts]


def clean_plays(pbp: pd.DataFrame, cfg: EfficiencyConfig = EfficiencyConfig()) -> pd.DataFrame:
    """Market columns dropped; scrimmage passes and runs with EPA; garbage time removed."""
    p = drop_market_columns(pbp)
    missing = [c for c in PLAY_COLUMNS if c not in p.columns]
    if missing:
        raise ValueError(f"play-by-play is missing columns {missing}")
    keep = (p["play_type"].isin(["pass", "run"]) & p["epa"].notna()
            & p["posteam"].notna() & (p["posteam"] != "") & p["defteam"].notna() & (p["defteam"] != ""))
    if "two_point_attempt" in p.columns:
        keep &= p["two_point_attempt"].fillna(0) != 1
    p = p[keep]
    garbage = (p["wp"] < cfg.garbage_low) | (p["wp"] > cfg.garbage_high)
    return p[~garbage.fillna(False)]


def team_game_table(plays: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, offense): sums the rolling step adds up. `sched` needs season and kickoff."""
    p = plays[["game_id", "posteam", "defteam", "pass", "rush", "epa", "success"]].copy()
    # A snap is a dropback (pass) or a designed run (rush); the flags are exclusive in nflfastR.
    p["is_pass"] = (p["pass"].fillna(0) == 1).astype(float)
    p["is_rush"] = ((p["rush"].fillna(0) == 1) & (p["is_pass"] == 0)).astype(float)
    p["epa_pass"] = p["epa"] * p["is_pass"]
    p["epa_rush"] = p["epa"] * p["is_rush"]
    p["has_sr"] = p["success"].notna().astype(float)
    p["sr"] = p["success"].fillna(0).astype(float)
    g = p.groupby(["game_id", "posteam", "defteam"], sort=False).agg(
        n_all=("epa", "size"), epa_all=("epa", "sum"),
        n_pass=("is_pass", "sum"), epa_pass=("epa_pass", "sum"),
        n_rush=("is_rush", "sum"), epa_rush=("epa_rush", "sum"),
        n_sr=("has_sr", "sum"), sr=("sr", "sum")).reset_index()
    info = sched.drop_duplicates("game_id").set_index("game_id")[["season", "kickoff"]]
    g = g.join(info, on="game_id")
    if g["kickoff"].isna().any():
        raise ValueError("play-by-play contains games that are not in the schedule")
    seasons = g["season"].to_numpy()
    g["off"] = [franchise(c, int(s)) for c, s in zip(g["posteam"], seasons)]
    g["def"] = [franchise(c, int(s)) for c, s in zip(g["defteam"], seasons)]
    return g


def _utc_ns(ts: pd.Series) -> np.ndarray:
    """Timezone-aware timestamps as int64 nanoseconds since the epoch (resolution-proof)."""
    return ts.dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").to_numpy()


def _rolling_sums(tab: pd.DataFrame, team_col: str, q_team: np.ndarray, q_season: np.ndarray,
                  q_asof: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each query, sums of `_ACC` over this team's rows from the same season and the
    previous season with kickoff strictly before the query's as_of. Returns (cur, prev), each (n_q, 8)."""
    cur = np.zeros((len(q_team), len(_ACC)))
    prev = np.zeros_like(cur)
    t = tab.sort_values("kickoff", kind="stable")
    for team, rows in t.groupby(team_col, sort=False):
        qi = np.flatnonzero(q_team == team)
        if len(qi) == 0:
            continue
        kick = _utc_ns(rows["kickoff"])
        seas = rows["season"].to_numpy()
        acc = np.vstack([np.zeros(len(_ACC)), np.cumsum(rows[_ACC].to_numpy(float), axis=0)])
        idx = np.searchsorted(kick, q_asof[qi], side="left")          # rows [0, idx) kicked off before as_of
        start_cur = np.searchsorted(seas, q_season[qi], side="left")  # seasons are contiguous in time
        start_prev = np.searchsorted(seas, q_season[qi] - 1, side="left")
        end_prev = np.minimum(idx, start_cur)
        cur[qi] = acc[idx] - acc[np.minimum(idx, start_cur)]
        prev[qi] = acc[end_prev] - acc[np.minimum(end_prev, start_prev)]
    return cur, prev


def _rates(cur: np.ndarray, prev: np.ndarray, w: float, prefix: str) -> dict[str, np.ndarray]:
    out = {}
    pairs = {"epa_all": ("n_all", "epa_all"), "epa_pass": ("n_pass", "epa_pass"),
             "epa_rush": ("n_rush", "epa_rush"), "sr": ("n_sr", "sr")}
    ix = {n: i for i, n in enumerate(_ACC)}
    with np.errstate(invalid="ignore", divide="ignore"):
        for name, (n, s) in pairs.items():
            den = cur[:, ix[n]] + w * prev[:, ix[n]]
            num = cur[:, ix[s]] + w * prev[:, ix[s]]
            out[f"{prefix}_{name}"] = np.where(den > 0, num / den, np.nan)
    out[f"{prefix}_plays"] = cur[:, ix["n_all"]]
    out[f"{prefix}_plays_prev"] = prev[:, ix["n_all"]]
    return out


def build_features(pbp: pd.DataFrame, sched: pd.DataFrame, games: pd.DataFrame | None = None,
                   cfg: EfficiencyConfig = EfficiencyConfig()) -> pd.DataFrame:
    """Features for each game in `games` (default: every game in `sched`), indexed by game_id.

    `sched` needs game_id, season, week, gameday, gametime, home_team, away_team
    (kickoff and as_of are added if absent). `pbp` may contain anything,
    including later games and market columns; only games that kicked off
    before each game's `as_of` can influence that game's row.
    """
    if "as_of" not in sched.columns:
        sched = asof_mod.add_asof(sched)
    games = sched if games is None else games
    if "as_of" not in games.columns:
        games = games.merge(sched[["game_id", "kickoff", "as_of"]], on="game_id", how="left")
    plays = clean_plays(pbp, cfg)
    tab = team_game_table(plays, sched)

    seasons = games["season"].to_numpy()
    cols: dict[str, np.ndarray] = {}
    q_asof = _utc_ns(games["as_of"])
    for side in SIDES:
        q_team = np.array([franchise(c, int(s)) for c, s in zip(games[f"{side}_team"], seasons)], dtype=object)
        for k, team_col in (("off", "off"), ("def", "def")):
            cur, prev = _rolling_sums(tab, team_col, q_team, seasons, q_asof)
            cols.update(_rates(cur, prev, cfg.prev_weight, f"{side}_{k}"))
    out = pd.DataFrame(cols, index=games["game_id"].to_numpy())
    out.index.name = "game_id"
    return out[feature_names()]


def assert_no_market_columns(features: pd.DataFrame) -> None:
    import re
    bad = [c for c in features.columns if re.search(FORBIDDEN_FEATURE_PATTERN, c, re.I)]
    if bad:
        raise AssertionError(f"betting-market information in feature columns: {bad}")
