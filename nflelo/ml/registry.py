"""Experiment registry: one JSON file per scored run, committed under `experiments/runs/`.

Each record holds the run name, the window label (dev, holdout, or
reproduction), the seasons and game set, n, metrics, features, params, the
git commit, the data fingerprint, the holdout flag, and a UTC timestamp.
`games_hash` is a hash of the sorted game_ids scored, so two rows of the
leaderboard are comparable only when their hashes match.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from .. import config
from . import data as mldata
from .eval import windows

RUNS_DIR = config.ROOT / "experiments" / "runs"


def git_commit(root: Path = config.ROOT) -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    sha = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain", "--", ".", ":!experiments/runs")  # the run files themselves don't count
    return {"sha": sha, "dirty": bool(dirty) if dirty is not None else None}


def sha256_path(path: Path) -> str | None:
    return mldata.sha256_file(path) if path.exists() else None


def data_fingerprint(games_csv: bool = True, ml_datasets: Iterable[str] = (),
                     seasons: Iterable[int] | None = None) -> dict:
    """Hashes of the inputs a run used: data/games.csv and/or cached nflverse files from the manifest."""
    out = {}
    if games_csv:
        out["data/games.csv"] = sha256_path(config.GAMES_CSV)
    ml_datasets = tuple(ml_datasets)
    if ml_datasets:
        out.update({f"data/raw/ml/{k}": v for k, v in mldata.manifest_hashes(ml_datasets, seasons).items()})
    return out


def games_hash(game_ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, game_ids))).encode()).hexdigest()[:16]


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in name)


def log_run(name: str, *, label: str, seasons: tuple[int, int], game_ids: Iterable[str], metrics: dict,
            features: list[str] | None = None, params: dict | None = None, data: dict | None = None,
            notes: str = "", extra: dict | None = None, runs_dir: Path = RUNS_DIR) -> Path:
    """Write one run record and return its path.

    `label` is "dev", "holdout", or "reproduction". The holdout flag is set
    from the seasons themselves, so a run cannot touch 2020-2025 unflagged.
    """
    if label not in windows.LABELS:
        raise ValueError(f"label must be one of {sorted(windows.LABELS)}")
    holdout = windows.touches_holdout(seasons)
    if label == "dev" and holdout:
        raise windows.HoldoutError(f"a 'dev' run cannot include holdout seasons: {seasons}")
    ids = list(game_ids)
    now = datetime.now(timezone.utc)
    rec = {
        "name": name,
        "label": label,
        "window": {"seasons": [int(seasons[0]), int(seasons[1])], "game_type": "REG"},
        "holdout": holdout,
        "n": int(metrics.get("n", len(ids))),
        "games_hash": games_hash(ids),
        "metrics": metrics,
        "features": features or [],
        "params": params or {},
        "data": data if data is not None else data_fingerprint(),
        "git": git_commit(),
        "timestamp": now.isoformat(timespec="seconds"),
        "notes": notes,
    }
    if extra:
        rec.update(extra)
    runs_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{now.strftime('%Y%m%dT%H%M%SZ')}_{_slug(name)}"
    path = runs_dir / f"{stem}.json"
    i = 1
    while path.exists():
        path = runs_dir / f"{stem}-{i}.json"
        i += 1
    path.write_text(json.dumps(rec, indent=2, sort_keys=False, default=float) + "\n")
    return path


def load_runs(runs_dir: Path = RUNS_DIR) -> list[dict]:
    out = []
    for p in sorted(runs_dir.glob("*.json")):
        rec = json.loads(p.read_text())
        rec["_file"] = p.name
        out.append(rec)
    return out


def leaderboard(label: str, runs_dir: Path = RUNS_DIR, latest_only: bool = True) -> pd.DataFrame:
    """Runs with this label, sorted by Brier. With `latest_only`, keep the newest run per (name, games_hash)."""
    rows = []
    for r in load_runs(runs_dir):
        if r.get("label") != label:
            continue
        m = r.get("metrics", {})
        rows.append({"name": r["name"], "seasons": "-".join(map(str, r["window"]["seasons"])), "n": r["n"],
                     "brier": m.get("brier"), "logloss": m.get("logloss"), "accuracy": m.get("accuracy"),
                     "ece": m.get("ece"), "games": r["games_hash"][:8], "holdout": r["holdout"],
                     "commit": (r.get("git") or {}).get("sha", "")[:7] if (r.get("git") or {}).get("sha") else "",
                     "timestamp": r["timestamp"]})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    if latest_only:
        t = t.sort_values("timestamp").drop_duplicates(["name", "games"], keep="last")
    return t.sort_values("brier").reset_index(drop=True)
