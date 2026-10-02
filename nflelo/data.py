"""Build one row per game from the legacy spreadsheet (1970-1998) and nflverse (1999+)."""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from . import config
from .teams import franchise

NFLVERSE_CSV_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"

COLUMNS = [
    "game_id", "season", "date", "week", "game_type", "home", "away",
    "home_score", "away_score", "neutral", "source",
    "home_moneyline", "away_moneyline", "spread_line",
    "home_qb_id", "away_qb_id", "home_rest", "away_rest",
]


def season_from_date(date: pd.Timestamp) -> int:
    """January and February games belong to the previous year's season."""
    return date.year - 1 if date.month in (1, 2) else date.year


# --------------------------------------------------------------------------- legacy

def dedupe_legacy_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse the legacy two-rows-per-game format to one row per game.

    Each game appears once from each team's view. Hosting is NaN for the home
    side, '@' for the away side, and 'N' for both sides of a neutral game. We
    keep the home (or, for neutral games, the first-listed) row.
    """
    df = raw.copy()
    df["Hosting"] = df["Hosting"].fillna("").astype(str).str.strip().str.upper()
    df["_pair"] = [
        "|".join(sorted((str(a), str(b)))) for a, b in zip(df["Team"], df["Opp"])
    ]
    df = df[df["Hosting"] != "@"]
    df = df.sort_values(["Date", "GameID"], kind="stable")
    return df.drop_duplicates(subset=["Date", "_pair"], keep="first").drop(columns="_pair")


def load_legacy_games(path: Path = config.LEGACY_XLSX) -> pd.DataFrame:
    """Legacy spreadsheet as unified-schema games (regular season only)."""
    raw = pd.read_excel(path)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw["Result"] = raw["Result"].astype("string").str.replace(" (OT)", "", regex=False)
    raw = raw.dropna(subset=["Team", "Opp", "Date", "Result"])
    parsed = raw["Result"].str.strip().str.extract(r"^([WLT])\s+(\d+)-(\d+)$")
    raw = raw[parsed[0].notna()].copy()
    raw["_team_pts"] = parsed.loc[raw.index, 1].astype(int)
    raw["_opp_pts"] = parsed.loc[raw.index, 2].astype(int)

    g = dedupe_legacy_rows(raw)
    season = g["Date"].map(season_from_date).astype(int)
    home = [franchise(c, s) for c, s in zip(g["Team"], season)]
    away = [franchise(c, s) for c, s in zip(g["Opp"], season)]
    out = pd.DataFrame({
        "season": season.to_numpy(),
        "date": g["Date"].to_numpy(),
        "week": g["Week"].astype(int).to_numpy(),
        "game_type": "REG",
        "home": home,
        "away": away,
        "home_score": g["_team_pts"].to_numpy(),
        "away_score": g["_opp_pts"].to_numpy(),
        "neutral": (g["Hosting"] == "N").to_numpy(),
        "source": "legacy",
    })
    out["game_id"] = (
        out["season"].astype(str) + "_" + out["week"].astype(str).str.zfill(2)
        + "_" + out["away"] + "_" + out["home"]
    )
    return _finalize(out)


# --------------------------------------------------------------------------- nflverse

def fetch_nflverse_schedules(raw_dir: Path = config.RAW_DIR, refresh: bool = True) -> pd.DataFrame:
    """Download the nflverse schedule file, caching it under data/raw.

    Tries nflreadpy first, then the plain CSV. If both fail and a cached copy
    exists it is used with a warning; otherwise the error is raised.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / "schedules.csv"
    if not refresh and cache.exists():
        return pd.read_csv(cache)

    errors = []
    df = None
    try:
        import nflreadpy
        df = nflreadpy.load_schedules().to_pandas()
    except Exception as exc:  # noqa: BLE001 - any failure falls through to the CSV
        errors.append(f"nflreadpy: {exc!r}")
        try:
            df = pd.read_csv(NFLVERSE_CSV_URL)
        except Exception as exc2:  # noqa: BLE001
            errors.append(f"csv: {exc2!r}")
    if df is not None:
        df.to_csv(cache, index=False)
        return df
    if cache.exists():
        warnings.warn(f"nflverse download failed ({'; '.join(errors)}); using cached {cache}")
        return pd.read_csv(cache)
    raise RuntimeError(f"nflverse download failed both ways: {'; '.join(errors)}")


def nflverse_to_games(sched: pd.DataFrame) -> pd.DataFrame:
    """Unified-schema rows for completed nflverse games."""
    s = sched.dropna(subset=["home_score", "away_score"]).copy()
    out = pd.DataFrame({
        "game_id": s["game_id"].to_numpy(),
        "season": s["season"].astype(int).to_numpy(),
        "date": pd.to_datetime(s["gameday"]).to_numpy(),
        "week": s["week"].astype(int).to_numpy(),
        "game_type": s["game_type"].to_numpy(),
        "home": [franchise(c, int(y)) for c, y in zip(s["home_team"], s["season"])],
        "away": [franchise(c, int(y)) for c, y in zip(s["away_team"], s["season"])],
        "home_score": s["home_score"].astype(int).to_numpy(),
        "away_score": s["away_score"].astype(int).to_numpy(),
        "neutral": (s["location"] == "Neutral").to_numpy(),
        "source": "nflverse",
    })
    for col in ["home_moneyline", "away_moneyline", "spread_line",
                "home_qb_id", "away_qb_id", "home_rest", "away_rest"]:
        out[col] = s[col].to_numpy() if col in s else pd.NA
    return _finalize(out)


# --------------------------------------------------------------------------- combine

def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    for col in COLUMNS:
        if col not in df:
            df[col] = pd.NA
    df = df[COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "game_id"], kind="stable").reset_index(drop=True)


def build_games(legacy: pd.DataFrame, nflverse: pd.DataFrame) -> pd.DataFrame:
    """Legacy REG games through 1998, nflverse (REG + playoffs) from 1999."""
    old = legacy[legacy["season"] <= config.LEGACY_LAST_SEASON]
    new = nflverse[nflverse["season"] >= config.NFLVERSE_FIRST_SEASON]
    games = pd.concat([old, new], ignore_index=True)
    return games.sort_values(["season", "date", "game_id"], kind="stable").reset_index(drop=True)


def load_games(csv: Path = config.GAMES_CSV) -> pd.DataFrame:
    return pd.read_csv(csv, parse_dates=["date"])


# --------------------------------------------------------------------------- validation

def validate_legacy_vs_nflverse(legacy: pd.DataFrame, nflverse: pd.DataFrame,
                                seasons: tuple[int, int] = (1999, 2025)) -> dict:
    """Compare legacy and nflverse REG games on (season, week, franchise pair, scores).

    Match rate is exact score agreement divided by the number of completed
    nflverse REG games in the window. Returns counts plus a mismatch table.
    """
    lo, hi = seasons

    def keyed(df: pd.DataFrame) -> pd.DataFrame:
        d = df[(df["game_type"] == "REG") & df["season"].between(lo, hi)].copy()
        swap = d["home"] > d["away"]
        d["a"] = d["home"].where(~swap, d["away"])
        d["b"] = d["away"].where(~swap, d["home"])
        d["a_score"] = d["home_score"].where(~swap, d["away_score"])
        d["b_score"] = d["away_score"].where(~swap, d["home_score"])
        return d[["season", "week", "a", "b", "a_score", "b_score", "home", "date"]]

    L, N = keyed(legacy), keyed(nflverse)
    m = N.merge(L, on=["season", "week", "a", "b"], how="outer",
                suffixes=("_nfl", "_leg"), indicator=True)
    both = m["_merge"] == "both"
    scores_eq = both & (m["a_score_nfl"] == m["a_score_leg"]) & (m["b_score_nfl"] == m["b_score_leg"])
    home_eq = both & (m["home_nfl"] == m["home_leg"])
    kind = pd.Series("score_differs", index=m.index)
    kind[m["_merge"] == "left_only"] = "only_nflverse"
    kind[m["_merge"] == "right_only"] = "only_legacy"
    mism = m[~scores_eq].assign(kind=kind[~scores_eq])
    n_nfl = len(N)
    sample_cols = ["kind", "season", "week", "a", "b", "a_score_nfl", "b_score_nfl",
                   "a_score_leg", "b_score_leg", "date_nfl", "date_leg"]
    return {
        "seasons": seasons,
        "n_nflverse": n_nfl,
        "n_legacy": len(L),
        "n_key_matched": int(both.sum()),
        "n_exact": int(scores_eq.sum()),
        "match_rate": float(scores_eq.sum() / n_nfl),
        "home_agrees_rate": float(home_eq.sum() / max(both.sum(), 1)),
        "mismatch_counts": mism["kind"].value_counts().to_dict(),
        "mismatches": mism.sort_values(["season", "week"])[sample_cols].reset_index(drop=True),
    }


def season_franchise_check(games: pd.DataFrame) -> pd.DataFrame:
    """Per season: franchises seen, games, and min/max games per franchise (REG only).

    A mis-mapped team code shows up as a wrong franchise count or lopsided
    games per franchise, so this is the evidence for seasons that have no
    nflverse overlap.
    """
    reg = games[games["game_type"] == "REG"]
    rows = []
    for season, g in reg.groupby("season"):
        per_team = pd.concat([g["home"], g["away"]]).value_counts()
        rows.append({"season": season, "games": len(g), "franchises": len(per_team),
                     "min_games": int(per_team.min()), "max_games": int(per_team.max())})
    return pd.DataFrame(rows)
