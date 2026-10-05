"""Cached nflverse loaders for the ML work: per-season parquet files, a hash manifest, schema checks.

Layout (all under `data/raw/ml/`, which is gitignored):

    data/raw/ml/<dataset>/<season>.parquet
    data/raw/ml/manifest.json      # {"<dataset>/<season>": {sha256, rows, fetched_at}}

Datasets: "pbp" (1999+) and "schedules", plus the M5 player datasets (all CC BY):
"player_stats" (weekly, 1999+), "depth_charts" (2001+; two formats, see
`load_depth_charts`), "injuries" (2009+), "rosters_weekly" (2002+), and
"players" (one snapshot, stored as season 0). "participation" (2016+) is
CC BY-SA 4.0. M5 reads it only for validation (`load_participation_for_validation`,
decision M5-D1 (a)). M5b (on-field ratings, approved under D1) reads it through
`load_participation`, and only the M5b modules may call that (a test enforces it):
nothing derived from participation may feed A4s or any CC BY artifact. Never add snap counts or any
Pro-Football-Reference-derived dataset (see context/ml.md section 3); FTN
data is out of scope.

Command line (run from the repo root):

    python -m nflelo.ml.data status              # cached seasons, rows, hash check
    python -m nflelo.ml.data fetch 1999 2025     # fill the cache (skips cached seasons)
    python -m nflelo.ml.data refresh 2025        # re-pull one season, report hash changes

Betting-market columns are stripped from play-by-play on load unless you
explicitly ask otherwise (decision D3: the market is a benchmark, never a feature).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .. import config
from ..teams import franchise

ML_DIR = config.RAW_DIR / "ml"
MANIFEST_NAME = "manifest.json"

FIRST_PBP_SEASON = 1999

# Columns that carry betting-market information. Matched case-insensitively.
# (`total` in play-by-play is the game's FINAL points, not the line. It is an
# outcome, not market data, and feature code never reads it.)
MARKET_EXACT = frozenset({"spread_line", "total_line", "vegas_wp", "vegas_home_wp",
                          "vegas_wpa", "vegas_home_wpa"})
MARKET_PATTERN = re.compile(r"vegas|moneyline|spread|total_line|over_odds|under_odds", re.I)

PBP_REQUIRED = ["game_id", "season", "season_type", "week", "home_team", "away_team",
                "posteam", "defteam", "play_type", "epa", "success", "wp", "pass", "rush"]
SCHEDULE_REQUIRED = ["game_id", "season", "game_type", "week", "gameday", "gametime",
                     "home_team", "away_team", "home_score", "away_score"]

# Scrimmage plays (pass or run) per game, both teams together. Real games run
# roughly 100 to 175; the bounds only catch truncated or duplicated files.
PLAYS_PER_GAME_BOUNDS = (80, 220)


class DataError(RuntimeError):
    """A cached or downloaded file failed a schema check."""


def is_market_column(name: str) -> bool:
    return name.lower() in MARKET_EXACT or bool(MARKET_PATTERN.search(name))


def drop_market_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove every betting-market column (decision D3)."""
    return df.drop(columns=[c for c in df.columns if is_market_column(c)])


# --------------------------------------------------------------------------- manifest

def _manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def read_manifest(root: Path = ML_DIR) -> dict:
    p = _manifest_path(root)
    return json.loads(p.read_text()) if p.exists() else {}


def _write_manifest(manifest: dict, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _manifest_path(root).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_path(dataset: str, season: int, root: Path = ML_DIR) -> Path:
    return root / dataset / f"{int(season)}.parquet"


def manifest_key(dataset: str, season: int) -> str:
    return f"{dataset}/{int(season)}"


def manifest_hashes(datasets: Iterable[str] = ("pbp", "schedules"),
                    seasons: Iterable[int] | None = None, root: Path = ML_DIR) -> dict:
    """{key: sha256} for the cached files a run depends on (used by the registry)."""
    seasons = None if seasons is None else {int(s) for s in seasons}
    out = {}
    for key, rec in read_manifest(root).items():
        ds, season = key.split("/")
        if ds in datasets and (seasons is None or int(season) in seasons):
            out[key] = rec["sha256"]
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- fetching

def _fetch_pbp(season: int):
    import nflreadpy
    return nflreadpy.load_pbp([season])


def _fetch_schedules(season: int):
    import nflreadpy
    return nflreadpy.load_schedules([season])


def _fetch_player_stats(season: int):
    import nflreadpy
    return nflreadpy.load_player_stats([season], summary_level="week")


def _fetch_depth_charts(season: int):
    import nflreadpy
    return nflreadpy.load_depth_charts([season])


def _fetch_injuries(season: int):
    import nflreadpy
    return nflreadpy.load_injuries([season])


def _fetch_rosters_weekly(season: int):
    import nflreadpy
    return nflreadpy.load_rosters_weekly([season])


def _fetch_players(season: int):
    import nflreadpy
    if int(season) != SNAPSHOT_SEASON:
        raise ValueError(f"players is a single snapshot; use season {SNAPSHOT_SEASON}")
    return nflreadpy.load_players()


def _fetch_participation(season: int):
    import nflreadpy
    return nflreadpy.load_participation([season])


FETCHERS: dict[str, Callable[[int], object]] = {
    "pbp": _fetch_pbp,
    "schedules": _fetch_schedules,
    "player_stats": _fetch_player_stats,
    "depth_charts": _fetch_depth_charts,
    "injuries": _fetch_injuries,
    "rosters_weekly": _fetch_rosters_weekly,
    "players": _fetch_players,
    "participation": _fetch_participation,
}
# First season each dataset exists in nflverse. "players" is one snapshot stored as season 0.
SNAPSHOT_SEASON = 0
FIRST_SEASON = {"pbp": FIRST_PBP_SEASON, "player_stats": 1999, "depth_charts": 2001, "injuries": 2009,
                "rosters_weekly": 2002, "participation": 2016}
# Datasets whose files carry a `season` column that must equal the requested season.
# (2025+ depth charts are ESPN snapshots with a null season; they are checked by date instead.)
_SEASON_CHECKED = ("pbp", "schedules", "player_stats", "injuries", "rosters_weekly")


def _unique_ints(col) -> set[int]:
    """Distinct values of a polars or pandas column, as Python ints."""
    vals = col.unique()
    vals = vals.to_list() if hasattr(vals, "to_list") else list(vals)
    return {int(v) for v in vals}


def _store(dataset: str, season: int, frame, root: Path) -> dict:
    """Write a polars (or pandas) frame as parquet, update the manifest, return the record."""
    path = cache_path(dataset, season, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    if hasattr(frame, "write_parquet"):
        frame.write_parquet(tmp)
    else:
        frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    rec = {"sha256": sha256_file(path), "rows": int(len(frame)),
           "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    manifest = read_manifest(root)
    manifest[manifest_key(dataset, season)] = rec
    _write_manifest(manifest, root)
    return rec


def fetch(dataset: str, season: int, root: Path = ML_DIR) -> dict:
    """Download one season of one dataset into the cache and return its manifest record."""
    if dataset not in FETCHERS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(FETCHERS)}")
    if dataset == "pbp" and season < FIRST_PBP_SEASON:
        raise ValueError(f"nflverse play-by-play starts in {FIRST_PBP_SEASON}")
    if dataset in FIRST_SEASON and season < FIRST_SEASON[dataset]:
        raise ValueError(f"nflverse {dataset} starts in {FIRST_SEASON[dataset]}")
    frame = FETCHERS[dataset](int(season))
    if len(frame) == 0:
        raise DataError(f"{dataset} {season}: download returned no rows")
    if dataset in _SEASON_CHECKED:
        col = frame["season"]
        col = col.drop_nulls() if hasattr(col, "drop_nulls") else col.dropna()
        got = _unique_ints(col)
        if got != {int(season)}:
            raise DataError(f"{dataset} {season}: download contained seasons {sorted(got)}")
    return _store(dataset, season, frame, root)


def refresh_one(dataset: str, season: int, root: Path = ML_DIR) -> dict:
    """Re-pull one season of one dataset and report whether its content hash changed."""
    old = read_manifest(root).get(manifest_key(dataset, season))
    rec = fetch(dataset, season, root)
    return {"dataset": dataset, "season": int(season), "changed": old is None or old["sha256"] != rec["sha256"],
            "was_cached": old is not None, "old_sha256": old and old["sha256"],
            "new_sha256": rec["sha256"], "rows": rec["rows"],
            "old_rows": old and old["rows"]}


def refresh(season: int, datasets: Iterable[str] = ("pbp", "schedules"), root: Path = ML_DIR) -> list[dict]:
    """Re-pull one season (every dataset by default) and report, per dataset, whether its hash changed.

    nflverse sometimes corrects past play-by-play. A changed hash on an old
    season is the signal to re-run experiments that used it. Note that parquet
    bytes can also change when nflverse re-exports identical rows, so a
    changed hash means "look", not "the numbers moved".
    """
    return [refresh_one(ds, season, root) for ds in datasets
            if not (ds == "pbp" and int(season) < FIRST_PBP_SEASON)]


def ensure(dataset: str, seasons: Iterable[int], root: Path = ML_DIR) -> list[int]:
    """Fetch any season that is missing from the cache. Returns the seasons fetched."""
    fetched = []
    for s in seasons:
        if not cache_path(dataset, s, root).exists():
            fetch(dataset, s, root)
            fetched.append(int(s))
    return fetched


def verify_cache(root: Path = ML_DIR) -> list[str]:
    """Compare cached files against the manifest. Returns human-readable problems."""
    problems = []
    for key, rec in read_manifest(root).items():
        ds, season = key.split("/")
        p = cache_path(ds, int(season), root)
        if not p.exists():
            problems.append(f"{key}: file missing")
        elif sha256_file(p) != rec["sha256"]:
            problems.append(f"{key}: hash differs from manifest")
    return problems


# --------------------------------------------------------------------------- schema checks

@dataclass
class Report:
    dataset: str
    season: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def unmapped_team_codes(codes: Iterable[str], season: int) -> list[str]:
    """Team codes that `nflelo.teams.franchise` does not know (reported, never dropped)."""
    bad = []
    for c in sorted({c for c in codes if isinstance(c, str) and c.strip()}):  # blanks are "no team", not a code
        try:
            franchise(c, int(season))
        except ValueError:
            bad.append(c)
    return bad


def validate_pbp(df: pd.DataFrame, season: int,
                 bounds: tuple[int, int] = PLAYS_PER_GAME_BOUNDS) -> Report:
    rep = Report("pbp", int(season))
    missing = [c for c in PBP_REQUIRED if c not in df.columns]
    if missing:
        rep.errors.append(f"missing required columns: {missing}")
        return rep
    if set(df["season"].unique()) != {int(season)}:
        rep.errors.append(f"unexpected seasons in file: {sorted(df['season'].unique())}")
    for c in ("epa", "wp", "pass", "rush", "week"):
        if not pd.api.types.is_numeric_dtype(df[c]):
            rep.errors.append(f"column {c!r} should be numeric, got {df[c].dtype}")
    if rep.errors:
        return rep
    scrim = df[df["play_type"].isin(["pass", "run"])]
    per_game = scrim.groupby("game_id").size()
    rep.stats = {"rows": int(len(df)), "games": int(df["game_id"].nunique()),
                 "scrimmage_plays": int(len(scrim)),
                 "plays_per_game_min": int(per_game.min()), "plays_per_game_median": float(per_game.median()),
                 "plays_per_game_max": int(per_game.max())}
    bad_games = per_game[(per_game < bounds[0]) | (per_game > bounds[1])]
    if len(bad_games):
        rep.errors.append(f"implausible plays per game {bounds}: {bad_games.to_dict()}")
    # Every game_id should embed the season it claims.
    wrong = df.loc[~df["game_id"].astype(str).str.startswith(f"{int(season)}_"), "game_id"].unique()
    if len(wrong):
        rep.errors.append(f"game_ids not from season {season}: {list(wrong[:5])}")
    codes = set(df["home_team"].dropna()) | set(df["away_team"].dropna()) \
        | set(df["posteam"].dropna()) | set(df["defteam"].dropna())
    bad = unmapped_team_codes(codes, season)
    if bad:
        rep.errors.append(f"team codes not known to nflelo.teams: {bad}")
    n_epa = scrim["epa"].notna().mean()
    rep.stats["epa_coverage"] = float(n_epa)
    if n_epa < 0.97:
        rep.warnings.append(f"EPA missing on {1 - n_epa:.1%} of scrimmage plays")
    return rep


def validate_schedules(df: pd.DataFrame, season: int) -> Report:
    rep = Report("schedules", int(season))
    missing = [c for c in SCHEDULE_REQUIRED if c not in df.columns]
    if missing:
        rep.errors.append(f"missing required columns: {missing}")
        return rep
    if set(df["season"].unique()) != {int(season)}:
        rep.errors.append(f"unexpected seasons in file: {sorted(df['season'].unique())}")
    if df["game_id"].duplicated().any():
        rep.errors.append("duplicate game_id values")
    for c in ("week", "home_score", "away_score"):
        if not pd.api.types.is_numeric_dtype(df[c]):
            rep.errors.append(f"column {c!r} should be numeric, got {df[c].dtype}")
    bad = unmapped_team_codes(set(df["home_team"]) | set(df["away_team"]), season)
    if bad:
        rep.errors.append(f"team codes not known to nflelo.teams: {bad}")
    played = df["home_score"].notna()
    rep.stats = {"games": int(len(df)), "played": int(played.sum()),
                 "missing_gametime": int(df["gametime"].isna().sum())}
    if rep.stats["missing_gametime"]:
        rep.warnings.append(f"{rep.stats['missing_gametime']} games have no kickoff time "
                            "(as-of falls back to 00:00 ET on game day)")
    return rep


# Required columns of the M5 datasets (as nflverse publishes them).
PLAYER_STATS_REQUIRED = ["player_id", "season", "week", "season_type", "team", "position"]
DEPTH_WEEKLY_REQUIRED = ["season", "club_code", "week", "game_type", "depth_team", "formation", "gsis_id",
                         "depth_position", "position"]
DEPTH_SNAPSHOT_REQUIRED = ["dt", "team", "gsis_id", "pos_grp", "pos_abb", "pos_slot", "pos_rank"]
INJURIES_REQUIRED = ["season", "game_type", "team", "week", "gsis_id", "position", "report_status",
                     "practice_status"]
ROSTERS_REQUIRED = ["season", "week", "game_type", "team", "gsis_id", "status", "position"]
PLAYERS_REQUIRED = ["gsis_id", "position", "position_group", "display_name"]
PARTICIPATION_REQUIRED = ["nflverse_game_id", "play_id", "possession_team", "offense_players", "defense_players"]


def _validate_columns(dataset: str, required: list[str]):
    def check(df: pd.DataFrame, season: int) -> Report:
        rep = Report(dataset, int(season))
        missing = [c for c in required if c not in df.columns]
        if missing:
            rep.errors.append(f"missing required columns: {missing}")
            return rep
        if "season" in df.columns and dataset in _SEASON_CHECKED:
            got = set(pd.to_numeric(df["season"], errors="coerce").dropna().astype(int).unique())
            if got != {int(season)}:
                rep.errors.append(f"unexpected seasons in file: {sorted(got)}")
        if "gsis_id" in required and df["gsis_id"].notna().mean() < 0.9:
            rep.errors.append(f"gsis_id missing on {1 - df['gsis_id'].notna().mean():.1%} of rows")
        rep.stats = {"rows": int(len(df))}
        return rep
    return check


def validate_depth_charts(df: pd.DataFrame, season: int) -> Report:
    """Either the 2001-2024 weekly format or the 2025+ ESPN snapshot format."""
    weekly = all(c in df.columns for c in DEPTH_WEEKLY_REQUIRED) and df["season"].notna().any()
    rep = _validate_columns("depth_charts", DEPTH_WEEKLY_REQUIRED if weekly else DEPTH_SNAPSHOT_REQUIRED)(df, season)
    if rep.ok and weekly:
        got = set(df["season"].dropna().astype(int).unique())
        if got != {int(season)}:
            rep.errors.append(f"unexpected seasons in file: {sorted(got)}")
        bad = sorted(set(df["depth_team"].dropna().astype(str)) - {"1", "2", "3", "4", "5"})
        if bad:
            rep.errors.append(f"unexpected depth_team values: {bad[:5]}")
    rep.stats["format"] = "weekly" if weekly else "snapshot"
    return rep


_VALIDATORS = {"pbp": validate_pbp, "schedules": validate_schedules,
               "player_stats": _validate_columns("player_stats", PLAYER_STATS_REQUIRED),
               "depth_charts": validate_depth_charts,
               "injuries": _validate_columns("injuries", INJURIES_REQUIRED),
               "rosters_weekly": _validate_columns("rosters_weekly", ROSTERS_REQUIRED),
               "players": _validate_columns("players", PLAYERS_REQUIRED),
               "participation": _validate_columns("participation", PARTICIPATION_REQUIRED)}


def _raise_if_bad(rep: Report) -> None:
    if not rep.ok:
        raise DataError(f"{rep.dataset} {rep.season}: " + "; ".join(rep.errors))


# --------------------------------------------------------------------------- loading

def _load(dataset: str, seasons: Iterable[int], columns: list[str] | None,
          validate: bool, root: Path) -> pd.DataFrame:
    seasons = [int(s) for s in seasons]
    ensure(dataset, seasons, root)
    parts = []
    for s in seasons:
        path = cache_path(dataset, s, root)
        # Validation needs the required columns even when the caller wants fewer.
        need = None
        if columns is not None:
            required = {"pbp": PBP_REQUIRED, "schedules": SCHEDULE_REQUIRED,
                        "player_stats": PLAYER_STATS_REQUIRED, "injuries": INJURIES_REQUIRED,
                        "rosters_weekly": ROSTERS_REQUIRED,
                        "participation": PARTICIPATION_REQUIRED}.get(dataset, [])
            need = sorted(set(columns) | (set(required) if validate else set()))
        df = pd.read_parquet(path, columns=need)
        if validate and dataset in _VALIDATORS:
            _raise_if_bad(_VALIDATORS[dataset](df, s))
        parts.append(df)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out


def load_pbp(seasons: Iterable[int], columns: list[str] | None = None, validate: bool = True,
             keep_market: bool = False, root: Path = ML_DIR) -> pd.DataFrame:
    """Play-by-play for the given seasons (pandas), fetching any missing season.

    Market columns are dropped unless `keep_market=True`; there is no reason to
    ask for them in feature code (decision D3).
    """
    if columns is not None and not keep_market:
        bad = [c for c in columns if is_market_column(c)]
        if bad:
            raise ValueError(f"market columns requested without keep_market=True: {bad}")
    df = _load("pbp", seasons, columns, validate, root)
    if not keep_market:
        df = drop_market_columns(df)
    if columns is not None:
        df = df[[c for c in columns if c in df.columns]]
    return df


def load_schedules(seasons: Iterable[int], validate: bool = True, root: Path = ML_DIR) -> pd.DataFrame:
    """Schedules and results (pandas). Market columns are kept here (they feed the benchmark only)."""
    return _load("schedules", seasons, None, validate, root)


# --------------------------------------------------------------------------- M5 player datasets

def load_player_stats(seasons: Iterable[int], columns: list[str] | None = None, validate: bool = True,
                      root: Path = ML_DIR) -> pd.DataFrame:
    """Weekly player stats (one row per player-game with any recorded stat), REG and POST."""
    return _load("player_stats", seasons, columns, validate, root)


def load_injuries(seasons: Iterable[int], validate: bool = True, root: Path = ML_DIR) -> pd.DataFrame:
    """Injury reports (2009+): the final game status (`report_status`) and practice status per player-week.

    Seasons and weeks are cast to int; rows without a gsis_id are dropped.
    """
    df = _load("injuries", [s for s in seasons if s >= FIRST_SEASON["injuries"]], None, validate, root)
    if df.empty:
        return df
    df = df[df["gsis_id"].notna()].copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    return df.reset_index(drop=True)


def load_rosters_weekly(seasons: Iterable[int], validate: bool = True, root: Path = ML_DIR) -> pd.DataFrame:
    """Weekly rosters (2002+): team, position, and roster status (ACT, RES, INA, PUP, ...) per player-week."""
    df = _load("rosters_weekly", [s for s in seasons if s >= FIRST_SEASON["rosters_weekly"]], None, validate, root)
    return df[df["gsis_id"].notna()].reset_index(drop=True) if len(df) else df


def load_players(validate: bool = True, root: Path = ML_DIR) -> pd.DataFrame:
    """The nflverse players table (one snapshot): gsis_id, position, names, draft info."""
    return _load("players", [SNAPSHOT_SEASON], None, validate, root)


def load_participation_for_validation(seasons: Iterable[int], validate: bool = True,
                                      root: Path = ML_DIR) -> pd.DataFrame:
    """Participation (CC BY-SA, 2016+). VALIDATION ONLY (decision M5-D1 (a)).

    Allowed uses: checking how good the usage-based playing-time estimates and
    the pregame lineups are. Never a model input, never used to fit anything
    that feeds a model. Only `nflelo.ml.players.validate` may call this
    (a test enforces it).
    """
    return _load("participation", [s for s in seasons if s >= FIRST_SEASON["participation"]], None, validate, root)


PARTICIPATION_LICENSE = "CC BY-SA 4.0"
PARTICIPATION_CREDIT = {"ngs": "NFL Next Gen Stats via nflverse (2016-2022)", "ftn": "FTN Data via nflverse (2023+)"}
PARTICIPATION_FTN_FIRST = 2023  # nflverse switched the participation source from NGS to FTN in 2023


def load_participation(seasons: Iterable[int], validate: bool = True, root: Path = ML_DIR,
                       columns: list[str] | None = None) -> pd.DataFrame:
    """Participation (who was on the field, every play; 2016+) for M5b on-field ratings. CC BY-SA 4.0.

    Approved for M5b under decision D1 (2026-10-03). Only the M5b modules
    (`nflelo.ml.players.participation`, `nflelo.ml.players.rapm`,
    `scripts/ml_m5b.py`) may call this; anything derived from it is CC BY-SA
    and must never flow into A4s or any CC BY artifact (a test enforces the
    import boundary). Credit: NFL Next Gen Stats via nflverse for 2016-2022,
    FTN Data via nflverse for 2023 onward. Cached per season with a manifest
    entry, like every other dataset here. M6 (`nflelo.ml.plays`, also CC BY-SA)
    reads it too, with `columns` limited to the pre-snap fields.
    """
    seasons = [int(s) for s in seasons if int(s) >= FIRST_SEASON["participation"]]
    df = _load("participation", seasons, columns, validate, root)
    return df


DEPTH_SNAPSHOT_FIRST = 2025   # from 2025 on, nflverse publishes ESPN daily snapshots instead of weekly charts
DEPTH_DROP = frozenset({2004})  # the 2004 charts are frozen (the same chart every week)
DEPTH_FIRST_RELIABLE = 2005     # 2001-2003 exist but are weak (context/m5-data-scope.md)
DEPTH_COLUMNS = ["season", "week", "team", "side", "slot", "group", "rank", "gsis_id", "source", "snapshot_ns"]


def _depth_side(formation: pd.Series) -> pd.Series:
    f = formation.astype("string").str.strip().str.lower()
    return pd.Series(np.select([f.str.startswith("off"), f.str.startswith("def")], ["off", "def"], "st"),
                     index=formation.index)


def depth_week_shift(dc: pd.DataFrame, sched: pd.DataFrame) -> dict[int, int]:
    """Season -> week-label shift of the weekly depth charts (REG).

    In 2007-2013 and 2015-2024 the chart labelled week L is the chart for
    schedule week L-1 (context/m5-data-scope.md). The rule: when a season's
    chart has one more REG week than the schedule, shift by one. Any other
    difference raises.
    """
    dmax = dc[dc["game_type"] == "REG"].groupby("season")["week"].max()
    smax = sched[sched["game_type"] == "REG"].groupby("season")["week"].max()
    out = {}
    for s, w in dmax.items():
        if s not in smax.index:
            continue
        d = int(w) - int(smax[s])
        if d not in (0, 1):
            raise DataError(f"depth charts {s}: {int(w)} REG weeks vs {int(smax[s])} in the schedule")
        out[int(s)] = d
    return out


def normalize_weekly_depth(dc: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """2001-2024 weekly charts -> DEPTH_COLUMNS, REG only, weeks shifted to schedule weeks."""
    from .players import positions as pos
    d = dc[(dc["game_type"] == "REG") & dc["gsis_id"].notna()].copy()
    d["season"] = d["season"].astype(int)
    shift = depth_week_shift(d, sched)
    d["week"] = d["week"].astype(int) - d["season"].map(shift).fillna(0).astype(int)
    d = d[d["week"] >= 1]
    d["team"] = pos.map_teams(d["club_code"], d["season"])
    d["side"] = _depth_side(d["formation"])
    d["slot"] = d["depth_position"].astype("string").str.strip().str.upper()
    grp = [pos.group_of(sl, side) or pos.group_of(p, side) for sl, p, side in zip(d["slot"], d["position"], d["side"])]
    d["group"] = grp
    # Special teams: only the place kicker is kept (as group K).
    st = d["side"] == "st"
    d.loc[st, "group"] = np.where(d.loc[st, "slot"].isin(["K", "PK"]), "K", None)
    d["rank"] = pd.to_numeric(d["depth_team"], errors="coerce").astype("Int64")
    d["source"] = "weekly"
    d["snapshot_ns"] = np.int64(0)
    return d[DEPTH_COLUMNS]


def normalize_snapshot_depth(dc: pd.DataFrame, sched: pd.DataFrame, season: int) -> pd.DataFrame:
    """2025+ ESPN snapshots -> DEPTH_COLUMNS: for each REG team-game, the team's last snapshot strictly before kickoff."""
    from . import asof as asof_mod
    from .players import positions as pos
    from .features.team_efficiency import _utc_ns
    s = sched[(sched["season"] == int(season)) & (sched["game_type"] == "REG")]
    if "kickoff" not in s.columns:
        s = asof_mod.add_asof(s)
    d = dc[dc["gsis_id"].notna()].copy()
    d["snap_ns"] = _utc_ns(pd.to_datetime(d["dt"], utc=True))
    d["team"] = pos.map_teams(d["team"], pd.Series(int(season), index=d.index))
    snaps = d[["team", "snap_ns"]].drop_duplicates().sort_values("snap_ns")
    parts = []
    for side_col in ("home_team", "away_team"):
        tg = pd.DataFrame({"team": pos.map_teams(s[side_col], s["season"]).to_numpy(),
                           "week": s["week"].astype(int).to_numpy(), "kick_ns": _utc_ns(s["kickoff"])})
        parts.append(tg)
    tg = pd.concat(parts, ignore_index=True).sort_values("kick_ns")
    tg = tg[tg["team"].notna()]
    tg["team"] = tg["team"].astype(str)
    snaps = snaps[snaps["team"].notna()].assign(team=lambda x: x["team"].astype(str))
    d = d[d["team"].notna()].assign(team=lambda x: x["team"].astype(str))
    tg["kick_key"] = tg["kick_ns"] - 1  # strictly before kickoff
    j = pd.merge_asof(tg, snaps.rename(columns={"snap_ns": "snap"}), left_on="kick_key", right_on="snap",
                      by="team", direction="backward")
    j = j[j["snap"].notna()]
    j["snap"] = j["snap"].astype("int64")
    out = j.merge(d, left_on=["team", "snap"], right_on=["team", "snap_ns"])
    grp = out["pos_grp"].astype("string")
    out["side"] = np.select([grp.str.contains(" D", regex=False), grp.str.contains("Special", regex=False)],
                            ["def", "st"], "off")
    out["slot"] = out["pos_abb"].astype("string").str.strip().str.upper()
    out["group"] = [pos.group_of(sl, side) for sl, side in zip(out["slot"], out["side"])]
    st = out["side"] == "st"
    out.loc[st, "group"] = np.where(out.loc[st, "slot"].isin(["K", "PK"]), "K", None)
    out["rank"] = pd.to_numeric(out["pos_rank"], errors="coerce").astype("Int64")
    out["season"] = int(season)
    out["source"] = "snapshot"
    out["snapshot_ns"] = out["snap"]
    return out[DEPTH_COLUMNS]


def load_depth_charts(seasons: Iterable[int], sched: pd.DataFrame, validate: bool = True,
                      min_season: int = DEPTH_FIRST_RELIABLE, root: Path = ML_DIR) -> pd.DataFrame:
    """Depth charts normalized to one row per (season, schedule week, team, side, slot, rank, player).

    Columns: season, week (the SCHEDULE week the chart is for), team (franchise
    ID), side (off/def/st), slot (depth_position or ESPN pos_abb), group (see
    `players.positions`; special teams keep only K), rank (1 = starter),
    gsis_id, source ("weekly" or "snapshot"), snapshot_ns (UTC ns of the ESPN
    snapshot, 0 for weekly charts).

    - 2001-2024 weekly charts: REG only, week labels shifted by
      `depth_week_shift`; 2004 is dropped (frozen), and seasons before
      `min_season` (2005 by default; 2001-2003 are weak) are skipped.
    - 2025+: ESPN daily snapshots; each team-game gets the team's last
      snapshot strictly before kickoff, so the chart is timestamped pregame.
    `sched` must include the requested seasons (game_type, week, gameday, gametime, teams).
    """
    wanted = [int(s) for s in seasons if int(s) >= max(min_season, FIRST_SEASON["depth_charts"])
              and int(s) not in DEPTH_DROP]
    parts = []
    for s in wanted:
        raw = _load("depth_charts", [s], None, validate, root)
        if s >= DEPTH_SNAPSHOT_FIRST or raw["season"].isna().all():
            parts.append(normalize_snapshot_depth(raw, sched, s))
        else:
            parts.append(normalize_weekly_depth(raw, sched))
    if not parts:
        return pd.DataFrame(columns=DEPTH_COLUMNS)
    out = pd.concat(parts, ignore_index=True)
    bad = out["team"].isna().sum()
    if bad:
        raise DataError(f"depth charts: {bad} rows with team codes not known to nflelo.teams")
    return out


def pbp_row_counts(root: Path = ML_DIR) -> dict[int, int]:
    """Row count per cached pbp season, from the manifest."""
    return {int(k.split("/")[1]): v["rows"] for k, v in sorted(read_manifest(root).items())
            if k.startswith("pbp/")}


def status(root: Path = ML_DIR) -> pd.DataFrame:
    """One row per cached file: dataset, season, rows, fetched_at, and whether the file still matches its hash."""
    bad = {p.split(":")[0] for p in verify_cache(root)}
    rows = [{"dataset": k.split("/")[0], "season": int(k.split("/")[1]), "rows": v["rows"],
             "fetched_at": v["fetched_at"], "hash_ok": k not in bad}
            for k, v in sorted(read_manifest(root).items())]
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m nflelo.ml.data", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    f = sub.add_parser("fetch")
    f.add_argument("first", type=int)
    f.add_argument("last", type=int)
    r = sub.add_parser("refresh")
    r.add_argument("season", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "status":
        st = status()
        print(st.to_string(index=False) if len(st) else "cache is empty")
        return 0 if st.empty or st["hash_ok"].all() else 1
    if a.cmd == "fetch":
        for ds in ("schedules", "pbp"):
            got = ensure(ds, [s for s in range(a.first, a.last + 1) if ds != "pbp" or s >= FIRST_PBP_SEASON])
            print(f"{ds}: fetched {got or 'nothing (all cached)'}")
        return 0
    for res in refresh(a.season):
        print(f"{res['dataset']} {res['season']}: rows {res['old_rows']} -> {res['rows']}, "
              f"{'CHANGED' if res['changed'] else 'unchanged'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
