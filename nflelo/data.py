"""Build one row per game: FiveThirtyEight's game file for 1970-1998, nflverse for 1999+.

The 2025 legacy spreadsheet is no longer a source. It is loaded only to
validate the FiveThirtyEight file and for the legacy-reproduction check.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from . import config
from .teams import franchise, franchise_538

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


# --------------------------------------------------------------------------- FiveThirtyEight

# 538's `playoff` column: w wild card, d divisional, c conference championship, s Super Bowl.
# The 1982 strike season's 16-team tournament (8 first-round games) and the
# 1990+ 12-team format's first round are both mapped to WC.
PLAYOFF_CODES_538 = {"w": "WC", "d": "DIV", "c": "CON", "s": "SB"}
# Playoff week = last regular-season week + this offset (nflverse numbers rounds the same way).
PLAYOFF_WEEK_OFFSET = {"WC": 1, "DIV": 2, "CON": 3, "SB": 4}


def calendar_weeks(dates: pd.Series) -> pd.Series:
    """Week number of each date within a season, counting Tuesday-to-Monday weeks.

    Week 1 starts on the Tuesday on or before the season's first game, so
    Thursday, Sunday, and Monday games share a week. This reproduces the
    legacy spreadsheet's week numbers for every 1970-1998 regular-season game.
    """
    first = dates.min()
    anchor = first - pd.Timedelta(days=(first.weekday() - 1) % 7)
    return ((dates - anchor).dt.days // 7 + 1).astype(int)


def load_538_games(path: Path = config.ELO538_CSV,
                   seasons: tuple[int, int] = (1970, config.FIVETHIRTYEIGHT_LAST_SEASON)) -> pd.DataFrame:
    """FiveThirtyEight game file as unified-schema games, REG and playoffs.

    `team1` is the home team unless `neutral` is 1 (verified against nflverse
    and the legacy spreadsheet). The file has no week, so REG weeks are
    calendar weeks (see calendar_weeks) and playoff weeks follow the last REG week.
    """
    raw = pd.read_csv(path, dtype={"playoff": "string"})
    raw = raw[raw["season"].between(*seasons)].dropna(subset=["score1", "score2"]).copy()
    raw["date"] = pd.to_datetime(raw["date"])
    codes = raw["playoff"].fillna("")
    bad = set(codes) - set(PLAYOFF_CODES_538) - {""}
    if bad:
        raise ValueError(f"unknown FiveThirtyEight playoff codes: {sorted(bad)}")
    raw["game_type"] = codes.map(PLAYOFF_CODES_538).fillna("REG")
    raw["home"] = [franchise_538(c, int(y)) for c, y in zip(raw["team1"], raw["season"])]
    raw["away"] = [franchise_538(c, int(y)) for c, y in zip(raw["team2"], raw["season"])]

    raw["week"] = 0
    for _, idx in raw.groupby("season").groups.items():
        sub = raw.loc[idx]
        reg = sub["game_type"] == "REG"
        reg_weeks = calendar_weeks(sub.loc[reg, "date"])
        raw.loc[reg_weeks.index, "week"] = reg_weeks
        last = int(reg_weeks.max())
        po = sub.loc[~reg, "game_type"].map(PLAYOFF_WEEK_OFFSET)
        raw.loc[po.index, "week"] = last + po

    out = pd.DataFrame({
        "season": raw["season"].astype(int).to_numpy(),
        "date": raw["date"].to_numpy(),
        "week": raw["week"].astype(int).to_numpy(),
        "game_type": raw["game_type"].to_numpy(),
        "home": raw["home"].to_numpy(),
        "away": raw["away"].to_numpy(),
        "home_score": raw["score1"].astype(int).to_numpy(),
        "away_score": raw["score2"].astype(int).to_numpy(),
        "neutral": (raw["neutral"] == 1).to_numpy(),
        "source": "fivethirtyeight",
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


def build_games(elo538: pd.DataFrame, nflverse: pd.DataFrame) -> pd.DataFrame:
    """FiveThirtyEight (REG + playoffs) through 1998, nflverse (REG + playoffs) from 1999."""
    old = elo538[elo538["season"] <= config.FIVETHIRTYEIGHT_LAST_SEASON]
    new = nflverse[nflverse["season"] >= config.NFLVERSE_FIRST_SEASON]
    games = pd.concat([old, new], ignore_index=True)
    return games.sort_values(["season", "date", "game_id"], kind="stable").reset_index(drop=True)


def load_games(csv: Path = config.GAMES_CSV) -> pd.DataFrame:
    return pd.read_csv(csv, parse_dates=["date"])


# --------------------------------------------------------------------------- validation

def validate_games(a: pd.DataFrame, b: pd.DataFrame, seasons: tuple[int, int],
                   game_types: tuple[str, ...] = ("REG",)) -> dict:
    """Compare two unified-schema game tables on a season window.

    Games are keyed by (season, game_type, franchise pair, n-th meeting of that
    pair in that season, ordered by date), so sources without week numbers
    can be compared. Match rate is exact agreement on that key plus both
    scores, divided by the number of games in `a`. Home/away agreement is
    measured on matched games that are non-neutral in both sources (the home
    side of a neutral game is arbitrary); neutral agreement and date
    agreement are reported on all matched games.
    """
    lo, hi = seasons

    def keyed(df: pd.DataFrame) -> pd.DataFrame:
        d = df[df["game_type"].isin(game_types) & df["season"].between(lo, hi)].copy()
        swap = d["home"] > d["away"]
        d["a"] = d["home"].where(~swap, d["away"])
        d["b"] = d["away"].where(~swap, d["home"])
        d["a_score"] = d["home_score"].where(~swap, d["away_score"])
        d["b_score"] = d["away_score"].where(~swap, d["home_score"])
        d["neutral"] = d["neutral"].fillna(False).astype(bool)
        d = d.sort_values(["date", "game_id"], kind="stable")
        d["nth"] = d.groupby(["season", "game_type", "a", "b"]).cumcount()
        return d[["season", "game_type", "a", "b", "nth", "a_score", "b_score", "home", "neutral", "date"]]

    A, B = keyed(a), keyed(b)
    key = ["season", "game_type", "a", "b", "nth"]
    m = A.merge(B, on=key, how="outer", suffixes=("_a", "_b"), indicator=True)
    both = m["_merge"] == "both"
    scores_eq = both & (m["a_score_a"] == m["a_score_b"]) & (m["b_score_a"] == m["b_score_b"])
    non_neutral = both & ~m["neutral_a"].fillna(False).astype(bool) & ~m["neutral_b"].fillna(False).astype(bool)
    home_eq = non_neutral & (m["home_a"] == m["home_b"])
    neutral_eq = both & (m["neutral_a"] == m["neutral_b"])
    date_eq = both & (m["date_a"] == m["date_b"])
    kind = pd.Series("score_differs", index=m.index)
    kind[m["_merge"] == "left_only"] = "only_in_a"
    kind[m["_merge"] == "right_only"] = "only_in_b"
    mism = m[~scores_eq].assign(kind=kind[~scores_eq])
    flag_bad = both & scores_eq & ((m["neutral_a"] != m["neutral_b"]) | (non_neutral & ~home_eq))
    flag_cols = ["season", "game_type", "a", "b", "home_a", "home_b", "neutral_a", "neutral_b", "date_a"]
    sample_cols = ["kind", "season", "game_type", "a", "b", "a_score_a", "b_score_a",
                   "a_score_b", "b_score_b", "date_a", "date_b"]
    return {
        "seasons": seasons,
        "game_types": game_types,
        "n_a": len(A),
        "n_b": len(B),
        "n_key_matched": int(both.sum()),
        "n_exact": int(scores_eq.sum()),
        "match_rate": float(scores_eq.sum() / max(len(A), 1)),
        "n_non_neutral_matched": int(non_neutral.sum()),
        "home_agrees_rate": float(home_eq.sum() / max(non_neutral.sum(), 1)),
        "neutral_agrees_rate": float(neutral_eq.sum() / max(both.sum(), 1)),
        "date_agrees_rate": float(date_eq.sum() / max(both.sum(), 1)),
        "mismatch_counts": mism["kind"].value_counts().to_dict(),
        "flag_mismatches": m[flag_bad].sort_values(["season", "date_a"])[flag_cols].reset_index(drop=True),
        "mismatches": mism.sort_values(["season", "date_a"])[sample_cols].reset_index(drop=True),
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
