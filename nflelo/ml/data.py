"""Cached nflverse loaders for the ML work: per-season parquet files, a hash manifest, schema checks.

Layout (all under `data/raw/ml/`, which is gitignored):

    data/raw/ml/<dataset>/<season>.parquet
    data/raw/ml/manifest.json      # {"<dataset>/<season>": {sha256, rows, fetched_at}}

Datasets: "pbp" (1999+) and "schedules". Never add snap counts or any
Pro-Football-Reference-derived dataset (see context/ml.md section 3);
participation and FTN data are out of scope for Stage A.

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


FETCHERS: dict[str, Callable[[int], object]] = {
    "pbp": _fetch_pbp,
    "schedules": _fetch_schedules,
}


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
    frame = FETCHERS[dataset](int(season))
    if len(frame) == 0:
        raise DataError(f"{dataset} {season}: download returned no rows")
    if dataset in ("pbp", "schedules"):
        got = _unique_ints(frame["season"])
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


_VALIDATORS = {"pbp": validate_pbp, "schedules": validate_schedules}


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
            required = {"pbp": PBP_REQUIRED, "schedules": SCHEDULE_REQUIRED}.get(dataset, [])
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
