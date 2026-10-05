"""Participation for M5b: who was on the field, cleaned to one row per scrimmage play.

LICENSE: CC BY-SA 4.0. Participation is NFL Next Gen Stats via nflverse
(2016-2022) and FTN Data via nflverse (2023+). Every table this module
returns, and anything derived from it (the M5b ratings, fits and files), is
CC BY-SA 4.0 and carries that note. Only the M5b modules (this file,
`players/rapm.py`, `scripts/ml_m5b.py`) may import it; nothing derived from
it may flow into A4s or any CC BY artifact (tests/ml/test_m5.py enforces the
import boundary).

Cleaning (context/ml-m5b-method.md, section 2; facts in context/m5-data-scope.md):

1. Plays: the M1 scrimmage plays (`team_efficiency.clean_plays`: pass and
   run snaps with an EPA, no two-point tries, garbage time removed, betting
   columns dropped), REG and POST.
2. Join to participation on (game_id, play_id).
3. Sides. `offense_players` / `defense_players` are GSIS ID lists. On some
   plays one list has 10 IDs (2016: a player with no GSIS mapping) or 12
   (2018-2019: an extra ID that is not one of the 22 players on the play).
   `players_on_play` lists all 22 (NFL `nfl_id`s through 2022, GSIS IDs from
   2023), and every `nfl_id` maps to a GSIS ID through the players table
   (CC BY). The repair: a player's team in a game is the side he is listed
   on most often in that game (offense = posteam, defense = defteam); the
   play's 22 are split by team; a player never listed in that game goes to
   the side that is short. On plays whose lists already hold 11 + 11, the
   repair reproduces them exactly (checked on every such play 2016-2019);
   it only changes plays the raw lists would drop.
4. Keep plays with exactly 11 distinct offensive and 11 distinct defensive
   GSIS IDs. `cleaning_report` gives the raw and repaired drop rate per season.

Output (`build_plays`), one row per kept play: game_id, play_id, season,
week, season_type, kick_ns, off_team, def_team (franchise IDs), home
(1 if the offense is the home team at a non-neutral site), epa, down,
ydstogo, yardline_100, is_pass, source ("raw" or "repaired"), o1..o11 and
d1..d11 (GSIS IDs, sorted). `design` turns plays into the sparse
plays-by-players matrix (offense and defense blocks).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from .. import asof as asof_mod
from .. import data as mldata
from ..features import team_efficiency as te
from . import positions as pos

LICENSE = mldata.PARTICIPATION_LICENSE
LICENSE_NOTE = ("CC BY-SA 4.0. Derived from nflverse participation data: NFL Next Gen Stats via nflverse "
                "(2016-2022), FTN Data via nflverse (2023+).")
OFF_COLS = [f"o{i}" for i in range(1, 12)]
DEF_COLS = [f"d{i}" for i in range(1, 12)]
PBP_COLUMNS = ["game_id", "play_id", "season", "season_type", "week", "posteam", "defteam", "home_team",
               "play_type", "pass", "rush", "epa", "success", "wp", "two_point_attempt", "down", "ydstogo",
               "yardline_100"]
PLAY_COLUMNS = ["game_id", "play_id", "season", "week", "season_type", "kick_ns", "off_team", "def_team", "home",
                "epa", "down", "ydstogo", "yardline_100", "is_pass", "source"] + OFF_COLS + DEF_COLS
CACHE_DIR = mldata.ML_DIR / "m5b"
CACHE_VERSION = 1


def era(season: int) -> str:
    """Participation source for a season: "ngs" (2016-2022) or "ftn" (2023+)."""
    return "ftn" if int(season) >= mldata.PARTICIPATION_FTN_FIRST else "ngs"


def nfl_to_gsis(players: pd.DataFrame) -> dict[str, str]:
    """nfl_id (as text) -> gsis_id, from the nflverse players table."""
    p = players[players["nfl_id"].notna() & players["gsis_id"].notna()]
    ids = p["nfl_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    return dict(zip(ids, p["gsis_id"].astype(str)))


def _split(x) -> list[str]:
    return [i for i in x.split(";") if i] if isinstance(x, str) else []


def _to_gsis(tokens: list[str], lut: dict[str, str]) -> set[str]:
    """players_on_play tokens to GSIS IDs (already GSIS from 2023; nfl_ids before). Unknown tokens stay marked."""
    return {t if t.startswith("00-") else lut.get(t, "?" + t) for t in tokens}


def resolve_sides(j: pd.DataFrame, lut: dict[str, str]) -> tuple[list, list, np.ndarray, np.ndarray]:
    """For each joined play: (offense IDs, defense IDs, ok_raw, ok_repaired). See the module docstring, step 3."""
    O = [set(_split(x)) for x in j["offense_players"]]
    D = [set(_split(x)) for x in j["defense_players"]]
    P = [_to_gsis(_split(x), lut) for x in j["players_on_play"]]
    g, pt, dt = j["game_id"].to_numpy(object), j["posteam"].to_numpy(object), j["defteam"].to_numpy(object)
    votes: dict = {}
    for gi, a, b, o, d in zip(g, pt, dt, O, D):
        for i in o:
            votes.setdefault((gi, i), Counter())[a] += 1
        for i in d:
            votes.setdefault((gi, i), Counter())[b] += 1
    team = {k: v.most_common(1)[0][0] for k, v in votes.items()}
    off_out, def_out = [], []
    ok_raw = np.zeros(len(j), bool)
    ok_fix = np.zeros(len(j), bool)
    for k, (gi, a, b, o, d, pp) in enumerate(zip(g, pt, dt, O, D, P)):
        raw = len(o) == 11 and len(d) == 11 and not (o & d)
        ok_raw[k] = raw
        if raw:
            off_out.append(o)
            def_out.append(d)
            ok_fix[k] = True
            continue
        off = de = None
        if len(pp) == 22 and not any(x.startswith("?") for x in pp):
            off = {i for i in pp if team.get((gi, i)) == a}
            de = {i for i in pp if team.get((gi, i)) == b}
            rest = pp - off - de
            if rest:
                if len(off) + len(rest) == 11 and len(de) == 11:
                    off = off | rest
                elif len(de) + len(rest) == 11 and len(off) == 11:
                    de = de | rest
            ok_fix[k] = len(off) == 11 and len(de) == 11
        off_out.append(off if ok_fix[k] else o)
        def_out.append(de if ok_fix[k] else d)
    return off_out, def_out, ok_raw, ok_fix


def build_plays(pbp: pd.DataFrame, participation: pd.DataFrame, sched: pd.DataFrame,
                players: pd.DataFrame) -> pd.DataFrame:
    """Kept plays (PLAY_COLUMNS) with the cleaning counts per season in `.attrs["report"]`. CC BY-SA 4.0."""
    if "as_of" not in sched.columns:
        sched = asof_mod.add_asof(sched)
    plays = te.clean_plays(pbp)
    part = participation.rename(columns={"nflverse_game_id": "game_id"}).copy()
    part["play_id"] = pd.to_numeric(part["play_id"]).astype("int64")
    plays = plays.copy()
    plays["play_id"] = pd.to_numeric(plays["play_id"]).astype("int64")
    keep_cols = ["game_id", "play_id", "players_on_play", "offense_players", "defense_players"]
    j = plays.merge(part[keep_cols].drop_duplicates(["game_id", "play_id"]), on=["game_id", "play_id"],
                    how="left", indicator=True).reset_index(drop=True)
    joined = (j["_merge"] == "both").to_numpy()
    off, de, ok_raw, ok_fix = resolve_sides(j, nfl_to_gsis(players))
    ok_raw &= joined
    ok_fix &= joined
    info = sched.drop_duplicates("game_id").set_index("game_id")
    season = j["season"].astype(int)
    report = []
    for S, idx in j.groupby(season).indices.items():
        n = len(idx)
        report.append({"season": int(S), "era": era(S), "clean_plays": int(n), "joined": int(joined[idx].sum()),
                       "raw_11_11": int(ok_raw[idx].sum()), "kept": int(ok_fix[idx].sum()),
                       "repaired": int((ok_fix[idx] & ~ok_raw[idx]).sum()),
                       "raw_drop_rate": float(1 - ok_raw[idx].mean()), "drop_rate": float(1 - ok_fix[idx].mean()),
                       "games": int(j.loc[idx, "game_id"].nunique())})
    k = np.flatnonzero(ok_fix)
    out = pd.DataFrame({"game_id": j["game_id"].to_numpy()[k], "play_id": j["play_id"].to_numpy()[k],
                        "season": season.to_numpy()[k], "week": j["week"].astype(int).to_numpy()[k],
                        "season_type": j["season_type"].to_numpy()[k]})
    out["kick_ns"] = te._utc_ns(info["kickoff"].reindex(out["game_id"]).reset_index(drop=True))
    out["off_team"] = pos.map_teams(pd.Series(j["posteam"].to_numpy()[k]), out["season"]).to_numpy(object)
    out["def_team"] = pos.map_teams(pd.Series(j["defteam"].to_numpy()[k]), out["season"]).to_numpy(object)
    neutral = (info["location"] == "Neutral") if "location" in info.columns else pd.Series(False, index=info.index)
    home_code = j["home_team"].to_numpy(object)[k]
    out["home"] = ((j["posteam"].to_numpy(object)[k] == home_code)
                   & ~neutral.reindex(out["game_id"]).fillna(False).astype(bool).to_numpy()).astype(float)
    out["epa"] = j["epa"].to_numpy(float)[k]
    out["down"] = j["down"].fillna(1).astype(int).to_numpy()[k]
    out["ydstogo"] = j["ydstogo"].fillna(10).astype(float).to_numpy()[k]
    out["yardline_100"] = j["yardline_100"].fillna(50).astype(float).to_numpy()[k]
    out["is_pass"] = (j["pass"].fillna(0) == 1).astype(float).to_numpy()[k]
    out["source"] = np.where(ok_raw[k], "raw", "repaired")
    oa = np.array([sorted(off[i]) for i in k], dtype=object).reshape(len(k), 11) if len(k) else np.empty((0, 11))
    da = np.array([sorted(de[i]) for i in k], dtype=object).reshape(len(k), 11) if len(k) else np.empty((0, 11))
    for c in range(11):
        out[OFF_COLS[c]] = oa[:, c]
        out[DEF_COLS[c]] = da[:, c]
    out = out.sort_values(["kick_ns", "game_id", "play_id"], kind="stable").reset_index(drop=True)
    out.attrs["report"] = report
    out.attrs["license"] = LICENSE_NOTE
    return out


def cleaning_report(plays: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(plays.attrs.get("report", []))


# --------------------------------------------------------------------------- cached loading

def _cache_key(season: int) -> str:
    hashes = {}
    for ds, s in (("pbp", season), ("schedules", season), ("participation", season),
                  ("players", mldata.SNAPSHOT_SEASON)):
        hashes.update(mldata.manifest_hashes((ds,), [s]))
    blob = json.dumps({"v": CACHE_VERSION, "h": hashes}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load_plays(seasons: Iterable[int], cache_dir: Path = CACHE_DIR, use_cache: bool = True) -> pd.DataFrame:
    """Cleaned plays for the given seasons (2016+), cached per season under data/raw/ml/m5b/. CC BY-SA 4.0.

    The cache key covers the manifest hashes of the season's play-by-play,
    schedule and participation files and of the players table.
    """
    seasons = [int(s) for s in seasons if int(s) >= mldata.FIRST_SEASON["participation"]]
    parts, reports = [], []
    players = None
    for S in seasons:
        for ds in ("pbp", "schedules", "participation"):
            mldata.ensure(ds, [S])
        mldata.ensure("players", [mldata.SNAPSHOT_SEASON])
        path = cache_dir / f"plays_{S}_{_cache_key(S)}.parquet"
        meta = path.with_suffix(".json")
        if use_cache and path.exists() and meta.exists():
            df = pd.read_parquet(path)
            rep = json.loads(meta.read_text())["report"]
        else:
            if players is None:
                players = mldata.load_players()
            sched = asof_mod.add_asof(mldata.load_schedules([S]))
            pbp = mldata.load_pbp([S], columns=PBP_COLUMNS)
            part = mldata.load_participation([S])
            df = build_plays(pbp, part, sched, players)
            rep = df.attrs["report"]
            if use_cache:
                cache_dir.mkdir(parents=True, exist_ok=True)
                df.to_parquet(path, index=False)
                meta.write_text(json.dumps({"license": LICENSE_NOTE, "season": S, "report": rep}, indent=1) + "\n")
        parts.append(df)
        reports.extend(rep)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=PLAY_COLUMNS)
    out.attrs["report"] = reports
    out.attrs["license"] = LICENSE_NOTE
    return out


# --------------------------------------------------------------------------- design

@dataclass
class PlayerIndex:
    """Columns of the player design: one per (side, gsis_id), sorted, so any subset keeps its order."""
    keys: np.ndarray   # "off|<id>" / "def|<id>"
    ids: np.ndarray
    side: np.ndarray

    def __len__(self) -> int:
        return len(self.keys)

    def get(self, keys) -> np.ndarray:
        return pd.Index(self.keys).get_indexer(keys)


def player_keys(plays: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(n x 11) arrays of "off|id" and "def|id" keys."""
    o = plays[OFF_COLS].to_numpy(object).astype(str)
    d = plays[DEF_COLS].to_numpy(object).astype(str)
    return np.char.add("off|", o), np.char.add("def|", d)


def player_index(plays: pd.DataFrame) -> PlayerIndex:
    ko, kd = player_keys(plays)
    keys = np.unique(np.concatenate([ko.ravel(), kd.ravel()]))
    side = np.array([k[:3] for k in keys], dtype=object)
    ids = np.array([k[4:] for k in keys], dtype=object)
    return PlayerIndex(keys, ids, side)


def design(plays: pd.DataFrame, index: PlayerIndex) -> sparse.csr_matrix:
    """Plays x players 0/1 matrix (22 ones per row). Every player must be in `index`."""
    ko, kd = player_keys(plays)
    cols = index.get(np.concatenate([ko, kd], axis=1).ravel())
    if (cols < 0).any():
        raise ValueError("plays contain players missing from the index")
    n = len(plays)
    rows = np.repeat(np.arange(n), 22)
    return sparse.csr_matrix((np.ones(n * 22), (rows, cols)), shape=(n, len(index)))


def snaps(plays: pd.DataFrame) -> pd.DataFrame:
    """(season, side, gsis_id) -> plays on the field, and (season, team, side) shares (CC BY-SA 4.0)."""
    rows = []
    for side, cols, tcol in (("off", OFF_COLS, "off_team"), ("def", DEF_COLS, "def_team")):
        x = plays[["season", tcol] + cols].melt(id_vars=["season", tcol], value_name="gsis_id")
        c = x.groupby(["season", "gsis_id"]).size().rename("snaps").reset_index()
        rows.append(c.assign(side=side))
    return pd.concat(rows, ignore_index=True)


def game_shares(plays: pd.DataFrame) -> pd.DataFrame:
    """Actual on-field share of each player in each game (kept plays): game_id, team, side, gsis_id, share."""
    out = []
    for side, cols, tcol in (("off", OFF_COLS, "off_team"), ("def", DEF_COLS, "def_team")):
        tot = plays.groupby(["game_id", tcol]).size().rename("team_plays")
        x = plays[["game_id", tcol] + cols].melt(id_vars=["game_id", tcol], value_name="gsis_id")
        c = x.groupby(["game_id", tcol, "gsis_id"]).size().rename("on").reset_index()
        c = c.join(tot, on=["game_id", tcol]).rename(columns={tcol: "team"})
        c["share"] = c["on"] / c["team_plays"]
        out.append(c.assign(side=side))
    return pd.concat(out, ignore_index=True)
