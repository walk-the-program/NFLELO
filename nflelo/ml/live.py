"""Live predictions (M4 Phase 1): the prediction ledger, which row counts, scoring, and projections.

Spec: context/ml-m4-method.md, sections 2, 3 and 6.

The ledger is `experiments/live/<season>.csv`, append-only and committed to git.
One row per (game, prediction run), with the columns in LEDGER_COLUMNS. Rules:

- A row whose `run_at_utc` is at or after the game's kickoff is refused
  (`append_rows` raises and writes nothing).
- Rows are only ever appended, in run order; existing rows are never rewritten.
- The scored prediction for a game is its last row before kickoff (M4-D2).
  The Wednesday row (the last pre-kickoff row written on a Wednesday, Eastern
  time) is scored separately, to measure what late QB news is worth.
- The market columns (`p_home_market`, `spread_market`) are written after the
  model has predicted and are for scoring only. `read_ledger` drops them
  unless asked (`market=True`), and nothing in the feature code reads the
  ledger at all (decision D3; tests/ml/test_live.py checks both).

This module needs only pandas and numpy: scripts/export_site.py imports it, and
the plain site build installs only requirements.txt (decision D5). The feature
side lives in `nflelo.ml.live_features`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..evaluate import metrics, outcome
from ..teams import franchise

LIVE_DIR = config.ROOT / "experiments" / "live"
LEDGER_COLUMNS = ["run_at_utc", "game_id", "kickoff_utc", "model_version", "p_home_model", "spread_model",
                  "p_home_elo", "spread_elo", "p_home_market", "spread_market", "home_qb_id", "away_qb_id",
                  "qb_source", "data_hash"]
MARKET_COLUMNS = ("p_home_market", "spread_market")
REQUIRED = ("run_at_utc", "game_id", "kickoff_utc", "model_version", "p_home_model", "data_hash")
QB_SOURCES = ("nflverse", "last_starter", "mixed", "none")
ET = "America/New_York"

# The first week whose predictions could be committed before kickoff: the live record starts here.
# 2026: week 5, first kickoff Thursday 2026-10-08. Weeks 1-4 were never predicted in advance.
LIVE_FROM_WEEK = {2026: 5}

# "QB change" tag on the site: |qb_delta_diff| at or above this, in EPA per dropback.
# With the 2026 fit's QB coefficient (about 3.7 log-odds per EPA per dropback) 0.08 moves an even
# game by about 7 percentage points; it is about 1.8 standard deviations of the DEV distribution.
QB_CHANGE_THRESHOLD = 0.08

REPO_URL = "https://github.com/walk-the-program/NFLELO"

# The M3 holdout sign-off (experiments/runs/20261003T215950Z_m3_signoff_*.json), quoted on the site.
HOLDOUT = {"seasons": [2020, 2025], "n": 1615, "model_brier": 0.2195, "elo_brier": 0.2231, "market_brier": 0.2096}


class LedgerError(ValueError):
    """A ledger write broke a rule (late row, missing field, column mismatch, out-of-order run)."""


# --------------------------------------------------------------------------- paths and time

def ledger_path(season: int, root: Path = LIVE_DIR) -> Path:
    return root / f"{int(season)}.csv"


def model_path(season: int, root: Path = LIVE_DIR) -> Path:
    return root / f"model_{int(season)}.json"


def latest_path(season: int, root: Path = LIVE_DIR) -> Path:
    return root / f"latest_{int(season)}.json"


def ledger_url(season: int) -> str:
    return f"{REPO_URL}/blob/main/experiments/live/{int(season)}.csv"


def to_utc(x):
    """Timestamp(s) as tz-aware UTC. Naive input is read as UTC."""
    if isinstance(x, (pd.Series, pd.Index, list, np.ndarray)):
        return pd.to_datetime(pd.Series(x) if not isinstance(x, pd.Series) else x, utc=True)
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def utc_iso(x) -> str:
    """'YYYY-MM-DDTHH:MM:SSZ' for one timestamp."""
    return to_utc(x).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- writing

def check_rows(rows: pd.DataFrame) -> None:
    """Raise LedgerError unless every row is complete and was written before its game's kickoff."""
    if list(rows.columns) != LEDGER_COLUMNS:
        raise LedgerError(f"ledger rows need exactly these columns, in order: {LEDGER_COLUMNS}; got {list(rows.columns)}")
    for c in REQUIRED:
        if rows[c].isna().any():
            raise LedgerError(f"missing {c} in {int(rows[c].isna().sum())} rows")
    run_at, kick = to_utc(rows["run_at_utc"]), to_utc(rows["kickoff_utc"])
    late = (run_at >= kick).to_numpy()
    if late.any():
        bad = rows.loc[late, ["game_id", "run_at_utc", "kickoff_utc"]].to_dict("records")
        raise LedgerError(f"{int(late.sum())} rows were made at or after kickoff, e.g. {bad[:3]}")
    if rows["run_at_utc"].nunique() > 1:
        raise LedgerError("one append is one prediction run: all rows must share run_at_utc")
    if rows["game_id"].duplicated().any():
        raise LedgerError("a run has at most one row per game")
    for c in ("p_home_model", "p_home_elo", "p_home_market"):
        p = pd.to_numeric(rows[c], errors="coerce")
        if ((p < 0) | (p > 1)).any():
            raise LedgerError(f"{c} outside [0, 1]")
    bad_src = set(rows["qb_source"].dropna()) - set(QB_SOURCES)
    if bad_src:
        raise LedgerError(f"unknown qb_source values {sorted(bad_src)}")


def append_rows(rows: pd.DataFrame, path: Path) -> int:
    """Append one prediction run to the ledger. Returns the number of rows written.

    Nothing is written if any row breaks a rule. The file is opened in append
    mode, so earlier rows are never touched. A new run must be later than every
    run already in the file.
    """
    if rows.empty:
        return 0
    check_rows(rows)
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        with open(path) as f:
            header = f.readline().rstrip("\n").split(",")
        if header != LEDGER_COLUMNS:
            raise LedgerError(f"{path} has columns {header}, expected {LEDGER_COLUMNS}")
        prev = read_ledger(path)
        if len(prev) and to_utc(rows["run_at_utc"].iloc[0]) <= prev["run_at_utc"].max():
            raise LedgerError(f"run_at {rows['run_at_utc'].iloc[0]} is not later than the last run in {path}")
        rows.to_csv(path, mode="a", header=False, index=False, lineterminator="\n")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(path, index=False, lineterminator="\n")
    return int(len(rows))


# --------------------------------------------------------------------------- reading and selection

def read_ledger(path: Path, market: bool = False) -> pd.DataFrame:
    """The ledger with parsed UTC times. Market columns are dropped unless `market=True` (D3)."""
    df = pd.read_csv(path, dtype={"game_id": str, "model_version": str, "home_qb_id": str, "away_qb_id": str,
                                  "qb_source": str, "data_hash": str})
    df["run_at_utc"] = to_utc(df["run_at_utc"])
    df["kickoff_utc"] = to_utc(df["kickoff_utc"])
    if not market:
        df = df.drop(columns=[c for c in MARKET_COLUMNS if c in df.columns])
    return df


def _pre_kickoff(ledger: pd.DataFrame) -> pd.DataFrame:
    # The writer already refuses late rows; filter again so a hand-edited file can't sneak one in.
    return ledger[ledger["run_at_utc"] < ledger["kickoff_utc"]]


def latest_before_kickoff(ledger: pd.DataFrame) -> pd.DataFrame:
    """One row per game: its last prediction made before kickoff (the scored one, M4-D2)."""
    ok = _pre_kickoff(ledger).sort_values(["run_at_utc", "game_id"], kind="stable")
    return ok.groupby("game_id", sort=False).tail(1).sort_values(["kickoff_utc", "game_id"]).reset_index(drop=True)


def wednesday_rows(ledger: pd.DataFrame) -> pd.DataFrame:
    """One row per game: its last pre-kickoff prediction written on a Wednesday (Eastern time)."""
    ok = _pre_kickoff(ledger)
    wed = ok[ok["run_at_utc"].dt.tz_convert(ET).dt.dayofweek == 2]
    return latest_before_kickoff(wed)


# --------------------------------------------------------------------------- scoring

def results_frame(sched: pd.DataFrame) -> pd.DataFrame:
    """Completed games: game_id, season, week, y (home 1 / 0.5 / 0), home margin."""
    done = sched[sched["home_score"].notna() & sched["away_score"].notna()].copy()
    return pd.DataFrame({"game_id": done["game_id"].astype(str).to_numpy(), "season": done["season"].astype(int).to_numpy(),
                         "week": done["week"].astype(int).to_numpy(), "y": outcome(done),
                         "margin": (done["home_score"] - done["away_score"]).astype(float).to_numpy()})


def _m(y, p) -> dict:
    m = metrics(y, p)
    return {"n": m["n"], "brier": round(m["brier"], 4), "accuracy": round(m["accuracy"], 4)}


def ats_elo(rows: pd.DataFrame) -> dict:
    """Elo's record against the market spread. Elo takes the home side when its spread is above
    the market's, the away side when below, and passes when equal. Positive spreads favor home,
    so the home side covers when the home margin beats the market spread; equal is a push."""
    return ats(rows, "spread_elo")


def ats(rows: pd.DataFrame, col: str) -> dict:
    """Record against the market spread for the spread in `col` (see `ats_elo`); rows without one are skipped."""
    r = rows[rows[col].notna() & rows["spread_market"].notna()]
    side = np.sign(r[col].to_numpy(float) - r["spread_market"].to_numpy(float))
    cover = np.sign(r["margin"].to_numpy(float) - r["spread_market"].to_numpy(float))
    picked = side != 0
    return {"w": int(((side == cover) & picked & (cover != 0)).sum()),
            "l": int(((side == -cover) & picked & (cover != 0)).sum()),
            "push": int((picked & (cover == 0)).sum()), "no_pick": int((~picked).sum())}


def score(selected: pd.DataFrame, results: pd.DataFrame, from_week: int) -> dict:
    """Brier and accuracy for model, Elo and market on completed games from `from_week` on.

    `selected` is one ledger row per game (with market columns). The market is
    scored only where it has a probability, and the model and Elo are also
    reported on exactly those games, so the three are comparable.
    """
    rows = selected.merge(results, on="game_id", how="inner")
    rows = rows[rows["week"] >= from_week]
    out = {"from_week": int(from_week), "n": int(len(rows))}
    if rows.empty:
        return out
    y = rows["y"].to_numpy(float)
    out["model"] = _m(y, rows["p_home_model"].to_numpy(float))
    out["elo"] = _m(y, rows["p_home_elo"].to_numpy(float))
    mk = rows[rows["p_home_market"].notna()]
    if len(mk):
        ym = mk["y"].to_numpy(float)
        out["market_games"] = {"n": int(len(mk)), "market": _m(ym, mk["p_home_market"].to_numpy(float)),
                               "model": _m(ym, mk["p_home_model"].to_numpy(float)),
                               "elo": _m(ym, mk["p_home_elo"].to_numpy(float))}
    out["ats_elo"] = ats_elo(rows)
    # The model's spread exists only for rows written after the margin model went live (M4 Phase 2).
    with_spread = rows[rows["spread_model"].notna()] if "spread_model" in rows else rows.iloc[:0]
    out["ats_model"] = {**ats(with_spread, "spread_model"), "games": int(len(with_spread))}
    out["weeks"] = sorted(int(w) for w in rows["week"].unique())
    return out


def live_record(ledger_with_market: pd.DataFrame, results: pd.DataFrame, from_week: int) -> dict:
    """The live scorecard: the last pre-kickoff rows, plus the Wednesday rows scored separately."""
    return {"final": score(latest_before_kickoff(ledger_with_market), results, from_week),
            "wednesday": score(wednesday_rows(ledger_with_market), results, from_week)}


# --------------------------------------------------------------------------- projections

def projected_wins(games: pd.DataFrame, p_model: pd.Series, p_elo: pd.Series) -> pd.DataFrame:
    """Projected final wins per team: wins so far, plus half a win per tie, plus the sum of the
    team's win probabilities in its remaining games.

    `games`: one season's REG games with game_id, season, week, home_team, away_team,
    home_score, away_score (NaN while unplayed). `p_model` and `p_elo` are home win
    probabilities indexed by game_id for the unplayed games. Returns one row per
    team with w, l, t, remaining, proj_model, proj_elo, and `games`, the team's
    remaining games (its own win probabilities).
    """
    g = games.copy()
    g["home"] = [franchise(c, int(s)) for c, s in zip(g["home_team"], g["season"])]
    g["away"] = [franchise(c, int(s)) for c, s in zip(g["away_team"], g["season"])]
    done = g["home_score"].notna() & g["away_score"].notna()
    teams = sorted(set(g["home"]) | set(g["away"]))
    rec = {t: {"team": t, "w": 0, "l": 0, "t": 0, "remaining": 0, "proj_model": 0.0, "proj_elo": 0.0, "games": []}
           for t in teams}
    for r in g[done].itertuples():
        y = 1.0 if r.home_score > r.away_score else 0.0 if r.home_score < r.away_score else 0.5
        for team, res in ((r.home, y), (r.away, 1.0 - y)):
            rec[team]["w" if res == 1.0 else "l" if res == 0.0 else "t"] += 1
    todo = g[~done].sort_values(["week", "gameday", "gametime", "game_id"] if "gameday" in g else ["week", "game_id"],
                                kind="stable")
    missing = [gid for gid in todo["game_id"] if gid not in p_model.index or pd.isna(p_model.get(gid))]
    if missing:
        raise ValueError(f"{len(missing)} unplayed games have no model probability, e.g. {missing[:3]}")
    for r in todo.itertuples():
        pm, pe = float(p_model[r.game_id]), float(p_elo[r.game_id])
        for team, opp, home, a, b in ((r.home, r.away, True, pm, pe), (r.away, r.home, False, 1 - pm, 1 - pe)):
            rec[team]["remaining"] += 1
            rec[team]["proj_model"] += a
            rec[team]["proj_elo"] += b
            rec[team]["games"].append({"game_id": r.game_id, "week": int(r.week), "opp": opp, "home": home,
                                       "neutral": bool(getattr(r, "location", "") == "Neutral"),
                                       "date": getattr(r, "gameday", None), "p_model": round(a, 4), "p_elo": round(b, 4)})
    rows = []
    for t in teams:
        x = rec[t]
        base = x["w"] + 0.5 * x["t"]
        rows.append({**x, "proj_model": round(base + x["proj_model"], 2), "proj_elo": round(base + x["proj_elo"], 2)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- playoff-odds history

# experiments/live/sim_<season>.csv: append-only, one row per (simulation run, team).
SIM_COLUMNS = ["run_at_utc", "team", "playoffs", "division", "seed1", "reach_sb", "win_sb", "wins_mean",
               "wins_p10", "wins_p90", "n_sims", "tau_rest", "shape", "seed", "model_version"]
SIM_PROBS = ("playoffs", "division", "seed1", "reach_sb", "win_sb")


# Site display gate (experiments/live/publish.json, committed). Missing file or key means "don't publish".
PUBLISH_FILE = "publish.json"


def publish_flags(root: Path = LIVE_DIR) -> dict:
    """{"publish_sim": bool}: whether the site may show the simulation (playoff odds, win ranges)."""
    p = Path(root) / PUBLISH_FILE
    d = json.loads(p.read_text()) if p.exists() else {}
    return {"publish_sim": bool(d.get("publish_sim", False))}


def sim_path(season: int, root: Path = LIVE_DIR) -> Path:
    return root / f"sim_{int(season)}.csv"


def append_sim(rows: pd.DataFrame, path: Path) -> int:
    """Append one simulation run (32 rows sharing run_at_utc). Refuses a run not later than the last one."""
    if rows.empty:
        return 0
    if list(rows.columns) != SIM_COLUMNS:
        raise LedgerError(f"simulation rows need exactly these columns, in order: {SIM_COLUMNS}")
    if rows["run_at_utc"].nunique() != 1 or rows["team"].duplicated().any():
        raise LedgerError("one append is one simulation run: one run_at_utc, one row per team")
    for c in SIM_PROBS:
        if ((rows[c] < 0) | (rows[c] > 1)).any():
            raise LedgerError(f"{c} outside [0, 1]")
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        with open(path) as f:
            header = f.readline().rstrip("\n").split(",")
        if header != SIM_COLUMNS:
            raise LedgerError(f"{path} has columns {header}, expected {SIM_COLUMNS}")
        prev = read_sim(path)
        if len(prev) and to_utc(rows["run_at_utc"].iloc[0]) <= prev["run_at_utc"].max():
            raise LedgerError(f"run_at {rows['run_at_utc'].iloc[0]} is not later than the last run in {path}")
        rows.to_csv(path, mode="a", header=False, index=False, lineterminator="\n")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(path, index=False, lineterminator="\n")
    return int(len(rows))


def read_sim(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"team": str, "shape": str, "model_version": str})
    df["run_at_utc"] = to_utc(df["run_at_utc"])
    return df


def sim_baseline(runs: pd.Series, current) -> pd.Timestamp | None:
    """The run to measure change against: the latest Wednesday (Eastern) run made more than a day before
    `current` (so Wednesday compares with last Wednesday, and Sunday with the Wednesday before it)."""
    cur = to_utc(current)
    r = pd.Series(sorted(set(to_utc(runs))))
    r = r[(r < cur - pd.Timedelta(days=1)) & (r.dt.tz_convert(ET).dt.dayofweek == 2)]
    return None if r.empty else r.iloc[-1]


# --------------------------------------------------------------------------- latest run

def read_latest(season: int, root: Path = LIVE_DIR) -> dict | None:
    p = latest_path(season, root)
    return json.loads(p.read_text()) if p.exists() else None


def qb_change(game: dict, threshold: float = QB_CHANGE_THRESHOLD) -> dict | None:
    """The 'QB change' tag for one game of the latest run, or None.

    Fires when |qb_delta_diff| reaches the threshold; names the side whose own
    delta is larger in size (the starter the team's rating doesn't remember).
    """
    d = game.get("qb_delta_diff")
    if d is None or abs(d) < threshold:
        return None
    side = "home" if abs(game.get("qb_delta_home") or 0.0) >= abs(game.get("qb_delta_away") or 0.0) else "away"
    return {"team": game[f"{side}_team"], "name": game.get(f"{side}_qb_name"), "side": side,
            "delta": round(float(game[f"qb_delta_{side}"]), 3), "source": game.get(f"{side}_qb_source")}
