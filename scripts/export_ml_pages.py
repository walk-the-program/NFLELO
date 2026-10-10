"""Site data for four model pages: the play explorer (M6), win probability (M7a), fourth downs (M7a-v2),
and early-down play calling (M7b).

    python scripts/export_ml_pages.py plays          # static, refreshed yearly: site/data/plays.json
    python scripts/export_ml_pages.py playcalling    # static, refreshed yearly: site/data/playcalling.json
    python scripts/export_ml_pages.py winprob        # live, weekly:             site/data/winprob.json
    python scripts/export_ml_pages.py fourth         # live, weekly (+ static chart): site/data/fourth.json
    python scripts/export_ml_pages.py live           # winprob then fourth, fail-soft (the weekly Action step)
    python scripts/export_ml_pages.py all            # all four
Options: --no-refresh (skip the refresh of this season's schedule and play-by-play; the weekly Action passes it
because scripts/ml_predict.py has just refreshed them), --budget SECONDS (live: skip a part that would start after
this many seconds), --season (default SEASON).

The data contract for the page builder is context/ml-pages.md.

Rules this script keeps:
- Nothing is scored on 2020-2025 here. Holdout numbers are LOADED from the run registry
  (experiments/runs) and the saved sign-off outputs; nothing new is computed on those seasons except the
  descriptive 2025 team play-calling view (observed play calls, no model evaluation).
- No market or betting fields in any output: no lines, spreads, odds, picks or EV. (Tests enforce it.)
- Live parts use only completed games (final in the schedule and already in data/games.csv).
- Every JSON is written atomically (temp file, then rename), so a killed run never leaves a broken file.
- Output is deterministic (no timestamps or run times in the files): a weekly run with no new games
  leaves them byte-identical, so it adds nothing to the commit.

Licenses: winprob.json is CC BY 4.0 (play-by-play and A4s only). plays.json, fourth.json and
playcalling.json are CC BY-SA 4.0 (participation data: NFL Next Gen Stats via nflverse 2016-2022,
FTN Data via nflverse 2023+).

Caches (gitignored, under data/raw/ml/pages/; the weekly Action keeps data/raw between runs):
    wp_<S>.joblib           the M7a WP model for season S (fit on 2006..S-1), refit if missing
    m6call_<S>.joblib       the M6 call-view GBM for season S (fit on 2016..S-1) and its structure configurations
    fourth_live_<S>.parquet valued fourth downs of S, by game; only new games are valued each week
A cache is rebuilt when the season, the scikit-learn version or CACHE_VERSION changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from nflelo import config  # noqa: E402
from nflelo.ml import data as mldata  # noqa: E402
from nflelo.ml import registry  # noqa: E402
from nflelo.ml.decisions import fourth as fd  # noqa: E402
from nflelo.ml.decisions import kicking as kk  # noqa: E402
from nflelo.ml.decisions import policy as pol  # noqa: E402
from nflelo.ml.features import opponent_adjust as oa  # noqa: E402
from nflelo.ml.plays import data as pdata  # noqa: E402
from nflelo.ml.plays import gbm  # noqa: E402
from nflelo.ml.plays import metrics as pm  # noqa: E402
from nflelo.ml.wp import model as wpm  # noqa: E402

SEASON = 2026                       # the live season (scripts/ml_predict.py SEASON)
SITE_DATA = config.ROOT / "site" / "data"
CACHE_DIR = mldata.ML_DIR / "pages"
RUNS_DIR = registry.RUNS_DIR
M7B_DIR = mldata.ML_DIR / "m7b"
FROZEN = config.ROOT / "experiments" / "m7a_v2" / "frozen_2026.json"
FILES = {"plays": "plays.json", "winprob": "winprob.json", "fourth": "fourth.json", "playcalling": "playcalling.json"}
SCHEMA = 1
CACHE_VERSION = 1
SEED = 20261003
FIRST_PBP = 1999
# The win-probability page is CC BY: its code path (load_live, wp_model, wp_states, game_series, build_winprob)
# reads play-by-play and A4s only, never M6, M7b or the fourth-down module (tests check the source).
WP_PBP_COLUMNS = list(wpm.PBP_COLUMNS) + ["desc"]

# Toss-up rule without the bootstrap: |best - second best| below this WP margin. 0.02 is the cut that best
# reproduces the bootstrap toss-up flags on the dev seasons 2018-2019 (77% agreement; data/raw/ml/m7a/
# fourth_m7adev.parquet, the v1 engine's 90% bands from 50 refits). No 2020-2025 data was used to pick it.
TOSSUP_MARGIN = 0.02

CC_BY = {"license": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"}
CC_BY_SA = {"license": "CC BY-SA 4.0", "url": "https://creativecommons.org/licenses/by-sa/4.0/"}
CREDIT_BY = ("Data: nflverse play-by-play and schedules (CC BY 4.0). Model: NFLELO M7a win probability "
             "(CC BY 4.0).")
CREDIT_SA = ("Data: nflverse (CC BY 4.0); participation data from NFL Next Gen Stats via nflverse (2016-2022) and "
             "FTN Data via nflverse (2023+). Numbers on this page derive from participation data and are released "
             "under CC BY-SA 4.0.")
REPO = "https://github.com/walk-the-program/NFLELO"

# Fields that must never appear in these files (Vegas is a benchmark elsewhere on the site, never here).
BANNED_KEYS = ("vegas", "spread", "moneyline", "odds", "market", "line", "pick", "units", "ats", "ev_bet", "bet")


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- small helpers

def rnd(x, nd: int = 4):
    """Round for JSON; NaN/inf -> None; numpy scalars -> python."""
    if x is None:
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    x = float(x)
    if not np.isfinite(x):
        return None
    return round(x, nd)


def write_json(name: str, obj, site_data: Path | None = None) -> int:
    """Compact JSON, written atomically. Returns the bytes written."""
    d = SITE_DATA if site_data is None else Path(site_data)
    d.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n"
    path = d / name
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
    return len(text.encode())


def read_json(name: str, site_data: Path | None = None) -> dict | None:
    p = (SITE_DATA if site_data is None else Path(site_data)) / name
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def run_record(name: str, runs_dir: Path | None = None) -> dict:
    """The latest registry record with exactly this run name (files are <timestamp>_<name>.json)."""
    d = RUNS_DIR if runs_dir is None else Path(runs_dir)
    hits = sorted(p for p in d.glob(f"*_{name}.json") if p.name.split("_", 1)[1] == f"{name}.json")
    if not hits:
        raise FileNotFoundError(f"no registry run named {name} in {d}")
    rec = json.loads(hits[-1].read_text())
    rec["_file"] = f"experiments/runs/{hits[-1].name}"
    return rec


def receipt(rec: dict) -> dict:
    g = rec.get("git") or {}
    return {"run": rec["_file"], "commit": (g.get("sha") or "")[:7] or None,
            "window": (rec.get("window") or {}).get("seasons"), "logged_at": rec.get("timestamp")}


def ci(d: dict | None, nd: int = 4) -> dict | None:
    if not d:
        return None
    return {"diff": rnd(d["diff"], nd), "lo": rnd(d["ci_low"], nd), "hi": rnd(d["ci_high"], nd),
            "level": d.get("level", 0.95), "excludes_zero": bool(d.get("excludes_zero")),
            "n": d.get("n"), "clusters": d.get("clusters")}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_load(path: Path) -> dict | None:
    try:
        blob = joblib.load(path)
    except Exception:          # missing, unreadable or from another library version
        return None
    if not isinstance(blob, dict) or blob.get("sklearn") != sklearn.__version__ \
            or blob.get("cache_version") != CACHE_VERSION:
        return None
    return blob


def cache_save(path: Path, blob: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {**blob, "sklearn": sklearn.__version__, "cache_version": CACHE_VERSION}
    tmp = path.with_name(path.name + ".tmp")
    joblib.dump(blob, tmp)
    os.replace(tmp, path)


def banned_keys(obj, path: str = "") -> list[str]:
    """Every dict key (or columnar field name) in `obj` that names a market or betting field."""
    def bad(name) -> bool:
        parts = str(name).lower().replace("-", "_").split("_")
        return any(b in parts or str(name).lower() == b for b in BANNED_KEYS)

    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if bad(k):
                out.append(f"{path}/{k}")
            if k == "columns" and isinstance(v, list):          # columnar tables name their fields here
                out.extend(f"{path}/columns/{c}" for c in v if bad(c))
            out.extend(banned_keys(v, f"{path}/{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:50]):
            out.extend(banned_keys(v, f"{path}[{i}]"))
    return out


def checked(obj: dict) -> dict:
    bad = banned_keys(obj)
    if bad:
        raise ValueError(f"market/betting fields in a page file: {bad[:5]}")
    return obj


def clock(qtr, game_secs) -> str:
    """'Q4 1:45' from the quarter and regulation seconds left (same rule as scripts/ml_m7a.py)."""
    qs = int(max(0, game_secs - (4 - int(qtr)) * 900))
    return f"Q{int(qtr)} {qs // 60}:{qs % 60:02d}"


def short(text, n: int = 120) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[:n - 1].rsplit(" ", 1)[0] + "…"


# --------------------------------------------------------------------------- live inputs

@dataclass
class Live:
    """Everything the live parts read, built once per run from data through now (completed games only)."""
    S: int
    games: pd.DataFrame
    pbp3: pd.DataFrame            # ml_m3 columns (A4s, M3 ratings)
    sched: pd.DataFrame
    a4s: pd.Series                # pregame A4s home-win probability, completed REG games 2006..S
    held_back: list               # final in nflverse, not yet in data/games.csv (picked up next run)
    pbp: pd.DataFrame | None = None   # fourth.PBP_COLUMNS, 1999..S (loaded on demand)
    seconds: dict = field(default_factory=dict)

    def final_games(self) -> pd.DataFrame:
        """Completed REG games of S that the live parts may use: final and with an A4s probability."""
        return final_games(self.sched, self.S, self.a4s.index)

    def full_pbp(self) -> pd.DataFrame:
        if self.pbp is None:
            t0 = time.perf_counter()
            self.pbp = mldata.load_pbp(range(FIRST_PBP, self.S + 1), columns=fd.PBP_COLUMNS)
            self.seconds["pbp_full"] = time.perf_counter() - t0
        return self.pbp


def final_games(sched: pd.DataFrame, S: int, a4s_index) -> pd.DataFrame:
    s = sched[(sched["season"] == S) & (sched["game_type"] == "REG") & sched["home_score"].notna()
              & sched["away_score"].notna()]
    return s[s["game_id"].isin(set(a4s_index))].reset_index(drop=True)


def load_live(S: int, no_refresh: bool) -> Live:
    import ml_m3
    import ml_m7a
    t = {}
    if not no_refresh:
        t0 = time.perf_counter()
        for r in mldata.refresh(S):
            log(f"nflverse {r['dataset']} {r['season']}: rows {r['old_rows']} -> {r['rows']}, "
                f"{'changed' if r['changed'] else 'unchanged'}")
        t["refresh"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    games, pbp3, sched = ml_m3.load_inputs(last_season=S)
    sched = sched.copy()
    # A game nflverse shows as final but data/games.csv does not have yet (build.py has not run since) has no
    # Elo input for A4s: treat it as not played this run; the next run (after build.py) picks it up.
    late = (sched["season"] == S) & sched["home_score"].notna() & ~sched["game_id"].isin(set(games["game_id"]))
    held = sorted(sched.loc[late, "game_id"])
    sched.loc[late, ["home_score", "away_score", "result"]] = np.nan
    t["inputs"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    a4s = ml_m7a.a4s_pregame(S, games, pbp3, sched)          # walk-forward: season s from A4s fit on < s
    t["a4s"] = time.perf_counter() - t0
    if held:
        log(f"held back until data/games.csv has them: {held}")
    return Live(S, games, pbp3, sched, a4s, held, seconds=t)


# --------------------------------------------------------------------------- cached models

def wp_model(L: Live) -> tuple[wpm.WPModel, dict]:
    """The M7a WP model for season S: trees by early stopping on S-1, refit on 2006..S-1, no isotonic map
    (wpm.CALIBRATE is False: the frozen model is the raw GBM), exactly as scripts/ml_m7a.py season_work."""
    path = CACHE_DIR / f"wp_{L.S}.joblib"
    blob = cache_load(path)
    if blob is not None and blob.get("season") == L.S:
        return blob["model"], blob["info"]
    log(f"fitting the WP model for {L.S} (2006..{L.S - 1}); cached afterwards in {path.name}")
    t0 = time.perf_counter()
    pbp = mldata.load_pbp(range(wpm.FIRST_TRAIN, L.S), columns=WP_PBP_COLUMNS)
    table = wpm.state_table(pbp, L.a4s)
    m = wpm.fit_season_ahead(table, L.S, calibrate=wpm.CALIBRATE)
    info = {"season": L.S, "trained_on": [wpm.FIRST_TRAIN, L.S - 1], "n_train": int(m.info["n_train"]),
            "n_iter": int(m.info["n_iter"]), "calibrated": bool(wpm.CALIBRATE),
            "fit_seconds": round(time.perf_counter() - t0, 1)}
    cache_save(path, {"season": L.S, "model": m, "info": info})
    return m, info


def frozen_artifact() -> dict:
    return json.loads(FROZEN.read_text())


def m6_bundle(S: int, table: pd.DataFrame | None = None) -> dict:
    """The M6 call-view GBM for season S by the M6 protocol (gbm.fit_season_ahead on 2016..S-1: the grid is
    scored on S-1, the winner refit on every training season) plus fourth.structure_configs of its training
    table. This is the model the fourth-down engine uses for S; the frozen M7a-v2 record lists its settings."""
    path = CACHE_DIR / f"m6call_{S}.joblib"
    blob = cache_load(path)
    if blob is not None and blob.get("season") == S:
        return blob
    t0 = time.perf_counter()
    if table is None:
        import ml_m6
        log(f"building the M6 play table 2016..{S - 1} (participation data)")
        table = ml_m6.load_world(S - 1, priors={}).table
    train = table[table["season"].between(fd.FIRST_M6, S - 1)]
    log(f"fitting the M6 call-view GBM for {S} on {len(train):,} plays; cached afterwards in {path.name}")
    g = gbm.fit_season_ahead(train, "call")
    info = {"season": S, "trained_on": [fd.FIRST_M6, S - 1], "n_train": int(len(train)),
            "chosen": g.info.get("chosen"), "n_iter": int(g.info["n_iter"]), "tov_iter": int(g.info["tov_iter"]),
            "val_crps": [rnd(v, 5) for v in g.info.get("val_crps", [])],
            "fit_seconds": round(time.perf_counter() - t0, 1)}
    try:
        fr = frozen_artifact().get("m6_for_offsets", {}) if S == SEASON else {}
        info["matches_frozen_record"] = (bool(fr) and fr.get("chosen") == info["chosen"]
                                         and int(fr.get("n_iter", -1)) == info["n_iter"])
    except OSError:
        info["matches_frozen_record"] = None
    blob = {"season": S, "model": g, "configs": fd.structure_configs(train), "info": info}
    cache_save(path, blob)
    return blob


# =========================================================================== 1. plays.json (M6)

PERSONNEL = {"11": (1.0, 1.0, 3.0), "12": (1.0, 2.0, 2.0), "21": (2.0, 1.0, 2.0), "13": (1.0, 3.0, 1.0)}
PERSONNEL_LABELS = {"11": "1 RB, 1 TE, 3 WR", "12": "1 RB, 2 TE, 2 WR", "21": "2 RB, 1 TE, 2 WR",
                    "13": "1 RB, 3 TE, 1 WR"}
GRID_DOWNS = (1, 2, 3, 4)
GRID_YTG = (1, 2, 3, 5, 7, 10, 15)
GRID_YL = (5, 10, 20, 35, 50, 65, 80, 95)
CALLS = ("run", "pass")
K_DRAWS = 32
MIN_POOL = 200
DRAWN = ["def_dl", "def_lb", "def_db", "form_shotgun", "form_pistol", "form_under_center", "box"]
P_MIN = 0.0005              # bins below this at either end of the stored range are dropped (sparse)
NEUTRAL_CLOCK = {"half_secs": 450.0, "game_secs": 2250.0}   # 7:30 left in the 2nd quarter


def _ytg_bucket(v) -> np.ndarray:
    return np.digitize(np.asarray(v, float), [3, 6, 10, 11])          # 1-2, 3-5, 6-9, 10, 11+


def _zone(v) -> np.ndarray:
    return np.digitize(np.asarray(v, float), [6, 11, 21, 51, 81])     # 1-5, 6-10, 11-20, 21-50, 51-80, 81-99


def structure_pool(train: pd.DataFrame, pers: str, call: int, down: int, ytg: float, yl: float) -> pd.DataFrame:
    """Training plays with this offensive personnel and call, in the most specific matching situation with at
    least MIN_POOL plays: (down, distance bucket, field zone), then (down, distance), then (distance), then all."""
    rb, te, wr = PERSONNEL[pers]
    base = train[(train["off_rb"] == rb) & (train["off_te"] == te) & (train["off_wr"] == wr)
                 & (train["off_ol"] == 5) & (train["is_pass"] == call)]
    yb, zn = _ytg_bucket([ytg])[0], _zone([yl])[0]
    by = _ytg_bucket(base["ydstogo"])
    bz = _zone(base["yardline_100"])
    for m in ((base["down"] == down) & (by == yb) & (bz == zn), (base["down"] == down) & (by == yb), by == yb):
        if int(m.sum()) >= MIN_POOL:
            return base[m]
    return base


def plays_grid(train: pd.DataFrame) -> pd.DataFrame:
    """Every grid cell (impossible ones dropped: ydstogo > yardline_100)."""
    rows = [{"down": d, "ydstogo": t, "yardline_100": y, "call": c, "personnel": p}
            for d in GRID_DOWNS for t in GRID_YTG for y in GRID_YL for c in CALLS for p in PERSONNEL if t <= y]
    return pd.DataFrame(rows)


def plays_inputs(train: pd.DataFrame, grid: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """K_DRAWS feature rows per cell: neutral situation, the cell's personnel and call, the defensive personnel,
    formation and box drawn from matching training plays (structure_pool), home alternating 0/1."""
    rng = np.random.default_rng(seed)
    outdoor = train[(train["roof_indoor"] == 0) & (train["roof_open"] == 0)]
    temp, wind = float(outdoor["temp"].median()), float(outdoor["wind"].median())
    neutral = {"score_diff": 0.0, **NEUTRAL_CLOCK, "off_timeouts": 3.0, "def_timeouts": 3.0, "roof_indoor": 0.0,
               "roof_open": 0.0, "temp": temp, "wind": wind, "off_ol": 5.0,
               **{c: 0.0 for c in pdata.RATINGS}}
    parts = []
    for i, c in grid.iterrows():
        call = 1 if c["call"] == "pass" else 0
        pool = structure_pool(train, c["personnel"], call, c["down"], c["ydstogo"], c["yardline_100"])
        draw = pool[DRAWN].to_numpy(float)[rng.integers(0, len(pool), K_DRAWS)]
        f = pd.DataFrame(draw, columns=DRAWN)
        rb, te, wr = PERSONNEL[c["personnel"]]
        f = f.assign(cell=i, down=float(c["down"]), ydstogo=float(c["ydstogo"]), yardline_100=float(c["yardline_100"]),
                     off_rb=rb, off_te=te, off_wr=wr, is_pass=float(call),
                     home=(np.arange(K_DRAWS) % 2).astype(float), pool_n=len(pool), **neutral)
        parts.append(f)
    X = pd.concat(parts, ignore_index=True)
    return X, {"temp": temp, "wind": wind}


def n_similar(train: pd.DataFrame, grid: pd.DataFrame, band: float = 5.0) -> np.ndarray:
    """Training plays with the cell's down, distance, call and personnel within +/- band yards of its yard line."""
    out = np.zeros(len(grid), int)
    for p, (rb, te, wr) in PERSONNEL.items():
        tp = train[(train["off_rb"] == rb) & (train["off_te"] == te) & (train["off_wr"] == wr)
                   & (train["off_ol"] == 5)]
        g = {k: np.sort(v["yardline_100"].to_numpy(float))
             for k, v in tp.groupby([tp["down"].astype(int), tp["ydstogo"].astype(int), tp["is_pass"].astype(int)])}
        for i in grid.index[grid["personnel"] == p]:
            c = grid.loc[i]
            ys = g.get((int(c["down"]), int(c["ydstogo"]), int(c["call"] == "pass")))
            if ys is not None:
                y = float(c["yardline_100"])
                out[i] = int(np.searchsorted(ys, y + band, "right") - np.searchsorted(ys, y - band, "left"))
    return out


def sparse_bins(p: np.ndarray) -> tuple[int, list]:
    """(first bin index, rounded probabilities of the non-TD bins from there): leading and trailing bins under
    P_MIN are dropped; the touchdown bin is reported separately."""
    q = p[:pdata.TD_BIN]
    keep = np.flatnonzero(q >= P_MIN)
    if not len(keep):
        return 0, []
    lo, hi = int(keep[0]), int(keep[-1])
    return lo, [rnd(v, 4) for v in q[lo:hi + 1]]


def m6_headline(runs_dir: Path | None = None) -> dict:
    out = {}
    for view in ("call", "situation"):
        gb, bl = run_record(f"m6_signoff_{view}_gbm", runs_dir), run_record(f"m6_signoff_{view}_baseline", runs_dir)
        cal = gb["result"]["cal_first"]
        out[view] = {"plays": gb["n"], "crps_model": rnd(gb["metrics"]["crps"]), "crps_baseline": rnd(bl["metrics"]["crps"]),
                     "crps_model_minus_baseline": ci(gb["result"]["vs_baseline"]["crps"]),
                     "p_first_ece": rnd(cal["ece"]), "p_first_ece_floor": cal["floor"],
                     "p_first_calibration_ok": bool(cal["ok"]), "receipt": receipt(gb)}
    return {"window": "2020-2025, season-ahead, REG + POST (locked holdout, scored once)",
            "verdict": "Pass", "metric": "CRPS in yards (lower is better)", **out,
            "plain": ("On the locked 2020-2025 seasons the model's yard distributions beat the historical "
                      "down-distance-field baseline in every season and both data eras; the gain is small because "
                      "most of a play's outcome is not predictable before the snap.")}


def build_plays(S: int = SEASON, table: pd.DataFrame | None = None, runs_dir: Path | None = None) -> dict:
    if table is None:
        import ml_m6
        table = ml_m6.load_world(S - 1, priors={}).table
    train = table[table["season"].between(fd.FIRST_M6, S - 1)].reset_index(drop=True)
    bundle = m6_bundle(S, table)
    model = bundle["model"]
    grid = plays_grid(train)
    X, nt = plays_inputs(train, grid)
    P, tov = model.predict(X)
    cell = X["cell"].to_numpy(int)
    k = np.bincount(cell, minlength=len(grid)).astype(float)
    Pc = np.zeros((len(grid), pdata.N_BINS))
    np.add.at(Pc, cell, P)
    Pc /= k[:, None]
    tovc = np.bincount(cell, weights=tov, minlength=len(grid)) / k
    pool_n = X.groupby("cell")["pool_n"].first().reindex(range(len(grid))).to_numpy(int)
    ytg, yl = grid["ydstogo"].to_numpy(float), grid["yardline_100"].to_numpy(float)
    ev = pm.event_probs(Pc, ytg, yl)
    ey = pm.expected_yards(Pc, yl)
    sim = n_similar(train, grid)
    cells = []
    for i, c in grid.iterrows():
        lo, ps = sparse_bins(Pc[i])
        cells.append({"down": int(c["down"]), "ydstogo": int(c["ydstogo"]), "yardline_100": int(c["yardline_100"]),
                      "call": c["call"], "personnel": c["personnel"], "exp_yards": rnd(ey[i], 2),
                      "p_first_or_td": rnd(ev["first"][i]), "p_20plus": rnd(ev["20plus"][i]),
                      "p_loss": rnd(ev["loss"][i]), "p_td": rnd(Pc[i, pdata.TD_BIN]), "p_turnover": rnd(tovc[i]),
                      "bins_from": lo, "bins": ps, "n_similar": int(sim[i]), "structure_pool": int(pool_n[i])})
    info = bundle["info"]
    obj = {
        "schema": SCHEMA, "page": "plays", "model": "M6 play outcome model, call view (gradient-boosted trees)",
        "season_model": S, "trained_on": info["trained_on"], "n_train_plays": info["n_train"],
        "settings": {"chosen": info["chosen"], "n_iter": info["n_iter"], "tov_iter": info["tov_iter"],
                     "protocol": ("M6 protocol: the 4-setting grid is fit on 2016..S-2 and scored on S-1 (CRPS); the "
                                  "winner is refit on 2016..S-1 with its early-stopped tree count. The same fitted "
                                  "model drives the 2026 fourth-down engine."),
                     "matches_frozen_record": info.get("matches_frozen_record")},
        "refresh": "yearly (after each season's participation release)",
        "license": {**CC_BY_SA, "credit": CREDIT_SA},
        "headline": m6_headline(runs_dir),
        "bins": {"count": pdata.N_BINS, "labels": pdata.BIN_LABELS,
                 "yards": [int(v) for v in pdata.BIN_YARDS], "td_index": pdata.TD_BIN,
                 "note": ("bin 0 = 10+ yards lost, bins 1-50 = -9..+40 yards, bin 51 = 41+ yards short of the end zone, "
                          "bin 52 = touchdown (reported as p_td, not in `bins`). Bins that are impossible from the "
                          "yard line are folded (gains past the goal line become touchdowns).")},
        "grid": {"down": list(GRID_DOWNS), "ydstogo": list(GRID_YTG), "yardline_100": list(GRID_YL),
                 "call": list(CALLS), "personnel": PERSONNEL_LABELS, "dropped": "ydstogo > yardline_100 (impossible)",
                 "cells": len(cells)},
        "held_neutral": {
            "score_diff": 0, "clock": "7:30 left in the 2nd quarter (half_secs 450, game_secs 2250)",
            "timeouts": "3 each", "roof": "outdoors", "temp_f": rnd(nt["temp"], 1), "wind_mph": rnd(nt["wind"], 1),
            "team_ratings": "league average (0) for the offense's pass/rush and the defense's pass/rush ratings",
            "home": "averaged: half the draws home, half away",
            "structure": (f"defensive personnel, formation and box count are not fixed: each cell averages the model "
                          f"over {K_DRAWS} draws of (defensive personnel, formation, box) from 2016-{S - 1} plays with "
                          f"the same offensive personnel and call in the most specific matching situation with at "
                          f"least {MIN_POOL} plays (down x distance bucket x field zone, else down x distance, else "
                          f"distance, else all); `structure_pool` is that pool's size"),
        },
        "fields": {"exp_yards": "expected yards (descriptive; 10+ lost counted as -12, 41+ as 55, a TD as the yard line)",
                   "p_first_or_td": "P(gain reaches the line to gain, or a touchdown)",
                   "p_20plus": "P(20+ yards)", "p_loss": "P(lost yards)", "p_td": "P(touchdown)",
                   "p_turnover": "P(interception or lost fumble), separate model",
                   "n_similar": (f"2016-{S - 1} plays with the same down, distance, call and personnel within 5 yards "
                                 "of the yard line: thin support means read the cell with care")},
        "cells": cells,
        "caveats": [
            "A prediction for the snap, not a promise: most of a play's outcome cannot be known before it.",
            "Held-neutral inputs: real games differ (score, clock, weather, the two teams).",
            "Cells with small n_similar (rare situations, e.g. 1st and 3 outside goal-to-go) rest on the model's "
            "smoothing more than on direct data.",
            "Run versus pass here is the called play type; it is not a recommendation (see the play-calling page).",
        ],
    }
    return checked(obj)


# =========================================================================== 2. winprob.json (M7a WP)

def wp_headline(runs_dir: Path | None = None) -> dict:
    w = run_record("m7a_signoff_wp", runs_dir)
    b = run_record("m7a_signoff_bench_nflfastr_wp", runs_dir)
    return {"window": "2020-2025 REG, season-ahead (locked holdout, scored once)", "plays": w["n"],
            "brier_ours": rnd(w["metrics"]["brier"]), "brier_nflfastr": rnd(b["metrics"]["brier"]),
            "ours_minus_nflfastr": ci(w["diffs"]["m7a_wp-nflfastr_wp"]["brier"]),
            "ece_ours": rnd(w["metrics"]["ece"]), "verdict": "Pass", "receipt": receipt(w),
            "plain": ("Our win probability beats nflfastR's public model on 2020-2025 by about 0.01 Brier, with a "
                      "95% interval that excludes zero, and uses no betting-market input.")}


def wp_states(pbp: pd.DataFrame, a4s: pd.Series, game_ids) -> pd.DataFrame:
    """WP model states for the given (completed) games. Ties are kept: their result is set to a placeholder
    only so wpm.state_table keeps the rows (its label column `y` is never used here)."""
    p = pbp[pbp["game_id"].isin(set(game_ids))].copy()
    p["result"] = p["result"].where(p["result"] != 0, 1.0)
    return wpm.state_table(p, a4s)


def game_series(st: pd.DataFrame, p_pos: np.ndarray, final_home: float) -> dict:
    """One game's home-WP series (one point per snap, then the final), its excitement and biggest swing."""
    st = st.assign(p_home=np.where(st["home"].to_numpy(float) == 1, p_pos, 1 - p_pos)).sort_values("play_id")
    t = np.maximum.accumulate(3600.0 - st["game_secs"].to_numpy(float))
    wp = st["p_home"].to_numpy(float)
    t_all = np.concatenate([t, [max(3600.0, t[-1])]])
    wp_all = np.concatenate([wp, [final_home]])
    d = np.diff(wp_all)
    k = int(np.argmax(np.abs(d)))
    row = st.iloc[k]
    winner_min = None
    if final_home in (0.0, 1.0):
        w = wp_all if final_home == 1.0 else 1 - wp_all
        winner_min = rnd(w.min(), 3)
    return {"t": [int(v) for v in np.round(t_all)], "wp": [rnd(v, 3) for v in wp_all],
            "excitement": rnd(np.abs(d).sum(), 3), "winner_min_wp": winner_min,
            "swing": {"delta_home": rnd(d[k], 3), "abs": rnd(abs(d[k]), 3), "qtr": int(row["qtr"]),
                      "clock": clock(row["qtr"], row["game_secs"]), "offense": row["posteam"],
                      "desc": short(row.get("desc", "")), "play_id": int(row["play_id"]),
                      "wp_before": rnd(wp_all[k], 3), "wp_after": rnd(wp_all[k + 1], 3)}}


def build_winprob(L: Live, runs_dir: Path | None = None) -> dict:
    wp, info = wp_model(L)
    fin = L.final_games()
    pbpS = mldata.load_pbp([L.S], columns=WP_PBP_COLUMNS)
    st = wp_states(pbpS, L.a4s, fin["game_id"])
    st = st.merge(pbpS[["game_id", "play_id", "desc"]].drop_duplicates(["game_id", "play_id"]),
                  on=["game_id", "play_id"], how="left")
    p = wp.predict(st) if len(st) else np.zeros(0)
    st = st.assign(_p=p)
    games, listing = [], []
    for g in fin.sort_values(["week", "gameday", "game_id"]).itertuples():
        s = st[st["game_id"] == g.game_id]
        if s.empty:
            continue
        hs, as_ = float(g.home_score), float(g.away_score)
        final_home = 1.0 if hs > as_ else 0.0 if hs < as_ else 0.5
        ser = game_series(s.drop(columns=["_p"]), s["_p"].to_numpy(float), final_home)
        ot = bool((pbpS.loc[pbpS["game_id"] == g.game_id, "qtr"] > 4).any())
        entry = {"game_id": g.game_id, "week": int(g.week), "date": str(g.gameday), "home": g.home_team,
                 "away": g.away_team, "home_score": int(hs), "away_score": int(as_), "overtime": ot,
                 "tie": final_home == 0.5, "pregame_home": rnd(L.a4s[g.game_id], 3), **ser}
        games.append(entry)
        listing.append({k: entry[k] for k in ("game_id", "week", "home", "away", "home_score", "away_score",
                                              "excitement", "winner_min_wp", "overtime")}
                       | {"biggest_swing": entry["swing"]["abs"]})
    listing.sort(key=lambda r: (-(r["excitement"] or 0), r["game_id"]))
    for i, r in enumerate(listing, 1):
        r["rank"] = i
    weeks = sorted({g["week"] for g in games})
    obj = {
        "schema": SCHEMA, "page": "winprob", "season": L.S,
        "model": "M7a win probability (monotone gradient boosting; no betting-market inputs)",
        "model_fit": {**info, "inputs": ("score, clock, down, distance, field position, timeouts, second-half kickoff, "
                                         "home, and the pregame A4s win probability (our game model)")},
        "refresh": "weekly (Wednesday and Sunday Action runs)",
        "updated_through": {"week": max(weeks) if weeks else None, "games": len(games),
                            "last_game_date": max((g["date"] for g in games), default=None),
                            "held_back": L.held_back},
        "license": {**CC_BY, "credit": CREDIT_BY},
        "headline": wp_headline(runs_dir),
        "definitions": {
            "t": "seconds of regulation elapsed at the snap (0 = kickoff, 3600 = end of regulation); "
                 "never decreases within a game",
            "wp": "our model's home-team win probability at each snap (the state left by the previous play); "
                  "the last point is the final result: 1 home win, 0 home loss, 0.5 tie",
            "excitement": "sum of absolute changes in home WP from snap to snap, including the final step",
            "swing": "the largest single change; the play is the snap that began it (a following kickoff or "
                     "extra point is part of the same change)",
            "winner_min_wp": "lowest WP the eventual winner had (comeback measure; null for a tie)",
            "pregame_home": "A4s pregame home win probability (logged before kickoff in the live ledger)"},
        "games_by_excitement": listing,
        "games": games,
        "caveats": [
            "Overtime is not modelled: the series stops at the end of regulation and jumps to the result.",
            "One point per snap; kickoffs, extra points and penalties without a snap are folded into the next snap.",
            "Win probability is a model estimate of how often teams in that spot win, not a certainty.",
        ],
    }
    return checked(obj)


# =========================================================================== 3. fourth.json (M7a-v2)

PRESETS = (
    {"key": "tied_q2", "label": "Tied, 7:30 left in the 2nd quarter", "qtr": 2, "score_diff": 0.0,
     "game_secs": 2250.0, "half_secs": 450.0, "timeouts": 3.0,
     "average_over": "home/away and who receives the second-half kickoff"},
    {"key": "down4_early_q4", "label": "Down 4, 13:00 left in the 4th quarter", "qtr": 4, "score_diff": -4.0,
     "game_secs": 780.0, "half_secs": 780.0, "timeouts": 3.0, "average_over": "home/away"},
    {"key": "up3_late_q4", "label": "Up 3, 4:00 left in the 4th quarter", "qtr": 4, "score_diff": 3.0,
     "game_secs": 240.0, "half_secs": 240.0, "timeouts": 3.0, "average_over": "home/away"},
)
CHART_YL = tuple(range(1, 100))
CHART_YTG = tuple(range(1, 11))
NEUTRAL_WEATHER = {"temp": 60.0, "wind": 5.0}
PLAY_COLUMNS = ("game_id", "play_id", "week", "team", "opponent", "home", "qtr", "clock", "score_diff", "ydstogo",
                "yardline_100", "choice", "recommended", "second", "margin", "wp_go", "wp_fg", "wp_punt", "p_convert",
                "p_fg_make", "wp_given_up", "matched", "tossup", "desc")
LIVE_COLS = ["game_id", "play_id", "season", "week", "qtr", "posteam", "defteam", "home", "score_diff", "game_secs",
             "ydstogo", "yardline_100", "choice", "best", "second", "margin", "wp_go", "wp_fg", "wp_punt", "p_conv",
             "p_fg", "wp_choice", "lost", "matched", "desc"]


def chart_decisions(S: int, preset: dict) -> pd.DataFrame:
    """Synthetic fourth downs for one preset: every yard line x distance (ydstogo <= yardline_100), one row per
    averaging variant. Teams are unnamed (league-average ratings), the kicker and punter unknown (league value),
    outdoors at 60F with a 5 mph wind, the A4s pregame probability 0.5 (even teams), as of week 1 of S."""
    first_half = preset["qtr"] <= 2
    variants = [(h, r) for h in (0.0, 1.0) for r in ((0.0, 1.0) if first_half else (0.0,))]
    rows = []
    for yl in CHART_YL:
        for ytg in CHART_YTG:
            if ytg > yl:
                continue
            for h, rec in variants:
                rows.append({"yardline_100": float(yl), "ydstogo": float(ytg), "home": h, "receive_2h_ko": rec})
    d = pd.DataFrame(rows)
    n = len(d)
    w = pd.DataFrame({"roof": ["outdoors"] * n, "temp": NEUTRAL_WEATHER["temp"], "wind": NEUTRAL_WEATHER["wind"]})
    wx = kk.weather(w)
    d = d.assign(game_id=f"{S}_chart_{preset['key']}", play_id=np.arange(n), season=S, week=1, qtr=preset["qtr"],
                 posteam="AVG_OFF", defteam="AVG_DEF", play_type="chart", score_diff=preset["score_diff"],
                 game_secs=preset["game_secs"], half_secs=preset["half_secs"], half2=float(not first_half), down=4.0,
                 pos_timeouts=preset["timeouts"], def_timeouts=preset["timeouts"], a4s_prob=0.5, home_team="",
                 choice="punt", indoor=wx["indoor"].to_numpy(), wind_out=wx["wind_out"].to_numpy(),
                 cold=wx["cold"].to_numpy(), roof_open=0.0, temp=NEUTRAL_WEATHER["temp"], wind=NEUTRAL_WEATHER["wind"],
                 gkey=int(kk.game_key(S, 1)), kicker=None, punter=None, desc="")
    return wpm.add_derived(d)


def build_chart(comp: fd.Components, S: int) -> dict:
    """Static decision chart: recommended option and WP margin by yard line x distance for three presets.
    Point estimates WITHOUT the bootstrap uncertainty band."""
    empty = pd.DataFrame(columns=["kind", "season", "week", "team", "off", "def"])
    out = []
    for pr in PRESETS:
        d = chart_decisions(S, pr)
        v = fd.option_values(comp, d, fd.go_inputs(d, comp, empty))
        v = pd.concat([d[["yardline_100", "ydstogo"]].reset_index(drop=True), v.reset_index(drop=True)], axis=1)
        m = v.groupby(["yardline_100", "ydstogo"], sort=True)[["wp_go", "wp_fg", "wp_punt", "p_conv", "p_fg"]].mean()
        rec = fd.recommend(m)
        m = m.join(rec).reset_index()
        out.append({"key": pr["key"], "label": pr["label"],
                    "state": {"qtr": pr["qtr"], "score_diff": pr["score_diff"], "clock": clock(pr["qtr"], pr["game_secs"]),
                              "timeouts_each": pr["timeouts"], "averaged_over": pr["average_over"]},
                    "columns": ["yardline_100", "ydstogo", "best", "margin", "wp_go", "wp_fg", "wp_punt", "p_conv", "p_fg"],
                    "yardline_100": m["yardline_100"].astype(int).tolist(), "ydstogo": m["ydstogo"].astype(int).tolist(),
                    "best": m["best"].tolist(), "margin": [rnd(x, 3) for x in m["margin"]],
                    "wp_go": [rnd(x, 3) for x in m["wp_go"]], "wp_fg": [rnd(x, 3) for x in m["wp_fg"]],
                    "wp_punt": [rnd(x, 3) for x in m["wp_punt"]], "p_conv": [rnd(x, 3) for x in m["p_conv"]],
                    "p_fg": [rnd(x, 3) for x in m["p_fg"]],
                    "close_call": [bool(x < TOSSUP_MARGIN) for x in m["margin"]]})
    return {"label": "Point estimates without the uncertainty band (no bootstrap)",
            "as_of": f"week 1 of {S}: league-average teams, kicker and punter; outdoors, 60F, 5 mph wind; "
                     "even teams (A4s 0.5)",
            "close_call_margin": TOSSUP_MARGIN, "presets": out}


def engine_fingerprint(wp_info: dict, m6_info: dict) -> str:
    blob = json.dumps({"frozen": sha256(FROZEN), "wp": wp_info, "m6": {k: m6_info.get(k) for k in
                                                                      ("season", "chosen", "n_iter", "n_train")},
                       "tossup": TOSSUP_MARGIN, "cache": CACHE_VERSION}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def fourth_components(L: Live) -> tuple[fd.Components, dict, str]:
    """The 2026 engine: the cached WP and M6 models, FG and punt models refit season-ahead every run (their as-of
    kicker and punter values read earlier games of S), the frozen M7a-v2 tables (candidate v2ad)."""
    import ml_m7a
    t0 = time.perf_counter()
    wp, wp_info = wp_model(L)
    b = m6_bundle(L.S)
    pbp = L.full_pbp()
    pS = pbp[pbp["season"] <= L.S]
    fg = kk.fit_fg_season_ahead(kk.fg_table(pS), L.S, wpm.FIRST_TRAIN)
    punt = kk.fit_punt_season_ahead(kk.punt_table(pS), L.S, wpm.FIRST_TRAIN)
    art, engine = ml_m7a.load_frozen(FROZEN)
    comp, _ = fd.fit_components(L.S, pbp, None, None, wp=wp, fg=fg, punt=punt, m6=b["model"], configs=b["configs"],
                                engine_version="v2", v2=engine)
    L.seconds["components"] = time.perf_counter() - t0
    info = {"wp": wp_info, "m6": b["info"], "engine": {"name": art["engine"], "candidate": art["candidate"],
                                                        "frozen_at": art["frozen_at"],
                                                        "frozen_file": "experiments/m7a_v2/frozen_2026.json",
                                                        "frozen_sha256": sha256(FROZEN)[:16]}}
    return comp, info, engine_fingerprint(wp_info, b["info"])


def value_live(L: Live, comp: fd.Components, fp: str, cache: Path | None = None) -> tuple[pd.DataFrame, int]:
    """Valued fourth downs of every completed game of S; only games not in the cache are valued.
    Returns (rows, number of newly valued games)."""
    cache = CACHE_DIR / f"fourth_live_{L.S}.parquet" if cache is None else cache
    old = None
    try:
        old = pd.read_parquet(cache)
        if old.empty or (old["fp"] != fp).any():
            old = None
    except Exception:
        old = None
    fin = set(L.final_games()["game_id"])
    done = set(old["game_id"]) if old is not None else set()
    todo = sorted(fin - done)
    new = []
    if todo:
        pbp = L.full_pbp()
        # the full play-by-play: each team's kicker and punter are the last ones it used, possibly in earlier games
        dec = fd.decision_table(pbp, L.a4s, seasons=[L.S])
        dec = dec[dec["game_id"].isin(todo)]
        if len(dec):
            keys = sorted(set(zip(dec["season"].astype(int), dec["week"].astype(int))))
            tab = oa.rating_table(L.pbp3, L.sched, oa.TUNED)
            ratings = pdata.week_ratings(tab, L.sched, keys)
            vals, _ = fd.value_season(comp, dec.reset_index(drop=True), ratings)
            new.append(vals[LIVE_COLS].assign(fp=fp))
    rows = pd.concat(([old] if old is not None else []) + new, ignore_index=True) if (old is not None or new) \
        else pd.DataFrame(columns=LIVE_COLS + ["fp"])
    rows = rows[rows["game_id"].isin(fin)].sort_values(["week", "game_id", "play_id"]).reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + ".tmp")
    rows.to_parquet(tmp, index=False)
    os.replace(tmp, cache)
    return rows, len(todo)


def team_table(rows: pd.DataFrame, fin: pd.DataFrame) -> list[dict]:
    games = pd.concat([fin["home_team"], fin["away_team"]]).value_counts()
    out = []
    for team, g in rows.groupby("posteam"):
        said_go = g["best"] == "go"
        clear_go = said_go & (g["margin"] >= TOSSUP_MARGIN)
        went = g["choice"] == "go"
        n_games = int(games.get(team, 0))
        out.append({"team": team, "games": n_games, "fourth_downs": int(len(g)),
                    "model_said_go": int(said_go.sum()), "went_when_model_said_go": int((said_go & went).sum()),
                    "go_rate_when_model_said_go": rnd((said_go & went).sum() / said_go.sum(), 3) if said_go.any() else None,
                    "model_said_go_clear": int(clear_go.sum()),
                    "go_rate_when_model_said_go_clear": (rnd((clear_go & went).sum() / clear_go.sum(), 3)
                                                         if clear_go.any() else None),
                    "went_when_model_said_kick": int((~said_go & went).sum()),
                    "matched_share": rnd(g["matched"].mean(), 3),
                    "wp_lost_total": rnd(g["lost"].sum(), 3),
                    "wp_lost_per_game": rnd(g["lost"].sum() / n_games, 4) if n_games else None})
    out.sort(key=lambda r: r["team"])
    return out


def fourth_status() -> dict:
    try:
        from export_research import FORWARD_TEST
        ft = dict(FORWARD_TEST)
    except Exception:
        ft = {}
    return {"forward_test": "pending",
            "plain": ("This engine's go-for-it conversion component is under a pre-registered forward test on the 2026 "
                      "season. It was frozen before any 2026 game and will be scored once, with no refit, after the "
                      "2026 participation data is released (earliest 2027-02-15). Until then its recommendations are "
                      "a model's view, not a validated tool."),
            "earliest_run": "2027-02-15", "pooled_2026_2027_secondary_earliest": "2028-02-15",
            "frozen_commit": ft.get("frozen_commit"), "registered_commit": ft.get("registered_commit"),
            "why": ("The first engine (v1) failed its 2020-2025 conversion calibration test (ECE 0.027 against a limit "
                    "of 0.022; it under-predicted 4th and 2). The fix (v2ad: run/pass mix by field zone plus "
                    "fourth-down conversion offsets) was built on development seasons only (dev ECE 0.019 vs 0.033)."),
            "components_status": {"win_probability": "passed 2020-2025", "field_goal": "passed 2020-2025",
                                  "punt": "passed 2020-2025", "conversion_v1": "failed 2020-2025",
                                  "conversion_v2ad": "forward test pending (2026)"},
            "doc": "context/ml.md (M7a-v2 forward test)"}


def fourth_audit_headline(runs_dir: Path | None = None) -> dict:
    a = run_record("m7a_signoff_audit", runs_dir)
    m = a["metrics"]
    return {"window": "2020-2025 REG (locked holdout), the v1 engine with 50-refit bands", "fourth_downs": m["n"],
            "matched_share": rnd(m["matched_share"], 3), "matched_share_clear_calls": rnd(m["matched_share_clear_calls"], 3),
            "tossup_share": rnd(m["tossup_share"], 3), "wp_lost_total": rnd(m["wp_lost_total"], 1),
            "wp_lost_per_team_game": rnd(m["wp_lost_per_team_game"], 3),
            "wp_lost_by_choice": {k: rnd(v, 1) for k, v in m["wp_lost_by_choice"].items()},
            "label": "model-estimated", "receipt": receipt(a)}


def build_fourth(L: Live, runs_dir: Path | None = None, keep_chart: dict | None = None) -> dict:
    t0 = time.perf_counter()
    comp, info, fp = fourth_components(L)
    t1 = time.perf_counter()
    if keep_chart and keep_chart.get("fingerprint") == fp and keep_chart.get("season") == L.S:
        chart = keep_chart
    else:
        chart = {**build_chart(comp, L.S), "fingerprint": fp, "season": L.S}
    t_chart = time.perf_counter() - t1
    t2 = time.perf_counter()
    rows, n_new = value_live(L, comp, fp)
    t_live = time.perf_counter() - t2
    fin = L.final_games()
    tossup = rows["margin"] < TOSSUP_MARGIN
    plays = [[r.game_id, int(r.play_id), int(r.week), r.posteam, r.defteam, bool(r.home == 1), int(r.qtr),
              clock(r.qtr, r.game_secs), int(r.score_diff), int(r.ydstogo), int(r.yardline_100), r.choice, r.best,
              r.second, rnd(r.margin, 3), rnd(r.wp_go, 3), rnd(r.wp_fg, 3), rnd(r.wp_punt, 3), rnd(r.p_conv, 3),
              rnd(r.p_fg, 3), rnd(r.lost, 3), bool(r.matched), bool(t), short(r.desc, 90)]
             for r, t in zip(rows.itertuples(), tossup)]
    n_tg = 2 * len(fin)
    summary = {"fourth_downs": len(plays), "games": int(rows["game_id"].nunique()) if len(rows) else 0,
               "matched_share": rnd(rows["matched"].mean(), 3) if len(rows) else None,
               "tossup_share": rnd(tossup.mean(), 3) if len(rows) else None,
               "wp_lost_total": rnd(rows["lost"].sum(), 2) if len(rows) else 0.0,
               "wp_lost_per_team_game": rnd(rows["lost"].sum() / n_tg, 4) if n_tg else None,
               "choices": {k: int(v) for k, v in rows["choice"].value_counts().items()},
               "recommended": {k: int(v) for k, v in rows["best"].value_counts().items()}}
    obj = {
        "schema": SCHEMA, "page": "fourth", "season": L.S,
        "model": "M7a-v2 fourth-down engine, candidate v2ad, frozen for 2026 (go / field goal / punt in win probability)",
        "engine": info,
        "refresh": "live part weekly (Wednesday and Sunday Action runs); chart once per season",
        "license": {**CC_BY_SA, "credit": CREDIT_SA},
        "label": "model-estimated",
        "status": fourth_status(),
        "headline": {"holdout_audit_v1": fourth_audit_headline(runs_dir), "wp_model": wp_headline(runs_dir)},
        "tossup_rule": {"margin_below": TOSSUP_MARGIN,
                        "plain": ("A call is a toss-up when the best option beats the second best by less than 0.02 "
                                  "win probability. Fixed threshold, no bootstrap band: 0.02 is the cut that best "
                                  "matched the bootstrap toss-up flags on the 2018-2019 development seasons.")},
        "chart": chart,
        "live": {"label": "model-estimated", "updated_through": {
                     "week": int(rows["week"].max()) if len(rows) else None, "games": len(fin),
                     "last_game_date": str(fin["gameday"].max()) if len(fin) else None, "held_back": L.held_back},
                 "summary": summary, "teams": team_table(rows, fin),
                 "plays": {"columns": list(PLAY_COLUMNS), "rows": plays}},
        "definitions": {
            "choice": "what the team did: go (run or pass), fg, or punt",
            "recommended": "the option with the highest model WP; `second` is the runner-up",
            "margin": "WP of the best option minus the second best",
            "wp_go / wp_fg / wp_punt": "model win probability for the team after each option",
            "wp_given_up": "model WP of the best option minus the WP of the team's choice (0 when they matched)",
            "p_convert": "model P(conversion) if the team goes for it (run/pass mix by distance and field zone)",
            "p_fg_make": "model P(make) for a field goal try from this spot with the team's kicker",
            "tossup": f"margin below {TOSSUP_MARGIN} (see tossup_rule)",
            "go_rate_when_model_said_go": "share of the team's model-says-go fourth downs on which it went for it"},
        "caveats": [
            "Every number here is model-estimated: what would have happened under the other choice is never observed.",
            "Teams know things the model does not (injuries, matchups, weather on the field, the game plan).",
            "The conversion component is under a pre-registered forward test until at least 2027-02-15.",
            "The chart shows point estimates without an uncertainty band; close calls (margin under 0.02) are toss-ups.",
            "Tie games and overtime are excluded; penalties and kneel-downs on fourth down are not decisions here.",
        ],
    }
    log(f"fourth: components {L.seconds.get('components', 0):.1f}s, chart {t_chart:.1f}s, live valuation "
        f"{t_live:.1f}s ({n_new} new games), total {time.perf_counter() - t0:.1f}s")
    return checked(obj)


# =========================================================================== 4. playcalling.json (M7b)

DD_LABELS = {"1-10+": "1st down, 10 or more to go", "1-short": "1st down, under 10 to go (mostly goal to go)",
             "2-short": "2nd down, 1-3 to go", "2-mid": "2nd down, 4-7 to go", "2-long": "2nd down, 8+ to go"}
ZONE_LABELS = {"own 1-20": "own 1-20 (yardline_100 81-99)", "own 21-50": "own 21-50 (yardline_100 50-80)",
               "opp 49-21": "opponent 49-21 (yardline_100 21-49)", "red zone": "red zone (yardline_100 1-20)"}
SCORE_LABELS = {"trail 4+": "trailing by 4 or more", "within 3": "within 3 points", "lead 4+": "leading by 4 or more"}
GROUP_LABELS = {"11": "11 personnel (1 RB, 1 TE)", "12": "12 personnel (1 RB, 2 TE)", "21": "21 personnel (2 RB, 1 TE)",
                "13": "13 personnel (1 RB, 3 TE)", "10": "10 personnel (1 RB, 0 TE)",
                "oth": "other groupings pooled (22, 20, 6+ linemen, empty backfield, ...)"}


def m7b_headline(runs_dir: Path | None = None) -> dict:
    o = run_record("m7b_signoff_ope", runs_dir)
    pl = run_record("m7b_signoff_placebo", runs_dir)
    t10 = run_record("m7b_signoff_ope_thr10", runs_dir)
    tr10 = run_record("m7b_signoff_ope_trim10", runs_dir)
    pol_run = run_record("m7b_signoff_policy", runs_dir)
    m = o["metrics"]
    per = {s: {"ope_epa": ci(v["ope"]), "placebo_epa": ci(v["placebo"]), "clear_share": rnd(v.get("clear_share"), 3),
               "plays": v["ope"]["n"]} for s, v in sorted(pol_run.get("per_season", {}).items())}
    return {"window": "2020-2025, season-ahead, REG + POST early downs (locked holdout, scored once)",
            "verdict": "Pass", "plays": m["n"],
            "ope_epa_per_play": ci(m["ope_epa"]), "ope_success_rate": ci(m["ope_succ"]),
            "placebo_epa_per_play": ci(pl["metrics"]["ope_epa"]),
            "sensitivity": {"overlap_threshold_0.10": ci(t10["metrics"]["ope_epa"]),
                            "weights_trimmed_at_10": ci(tr10["metrics"]["ope_epa"])},
            "share_of_plays_policy_applies": rnd(m["share_policy_applies"], 3),
            "epa_per_affected_play": rnd(m["epa_per_affected_play"], 3),
            "rule": "pass if the lower end of the OPE interval is above 0 and the placebo interval includes 0",
            "per_season": per, "receipt": receipt(o),
            "plain": ("Following the recommendations would have gained about 0.05 EPA per early-down play on the "
                      "unseen 2020-2025 seasons (95% interval +0.038 to +0.067); a placebo with shuffled actions "
                      "shows no gain.")}


def build_playcalling(runs_dir: Path | None = None, m7b_dir: Path | None = None) -> dict:
    d = M7B_DIR if m7b_dir is None else Path(m7b_dir)
    recs = pd.read_parquet(d / "recs_holdout.parquet")
    cells = pd.read_parquet(d / "cells_holdout.parquet")
    recs, cells = recs[recs["variant"] == "main"], cells[cells["variant"] == "main"]
    pol_run = run_record("m7b_signoff_policy", runs_dir)
    # the saved outputs must be the logged run's: clear-cell counts per season agree
    logged = {int(s): v["clear"] for s, v in pol_run["metrics"]["recs"]["main"].items()}
    saved = recs.groupby("season")["clear"].sum().astype(int).to_dict()
    if logged != saved:
        raise ValueError(f"saved M7b recommendations do not match the logged run: {saved} vs {logged}")
    seasons = sorted(int(s) for s in recs["season"].unique())
    cand = cells[cells["candidate"]]
    out_cells = []
    for c in pol.CELLS:
        dd, zone, score = [x.strip() for x in c.split("|")]
        by = {}
        for S in seasons:
            r = recs[(recs["cell"] == c) & (recs["season"] == S)]
            if r.empty:
                continue
            r = r.iloc[0]
            cs = cand[(cand["cell"] == c) & (cand["season"] == S)].sort_values("gain", ascending=False)
            by[str(S)] = {"best": r["best"] if isinstance(r["best"], str) and r["best"] else None,
                          "clear": bool(r["clear"]), "label": "clear best" if r["clear"] else "no clear best",
                          "gain": rnd(r["gain"]), "gain_lo": rnd(r["gain_lo"]), "gain_hi": rnd(r["gain_hi"]),
                          "n_train_plays": int(r["n_cell"]), "candidates": int(r["candidates"]),
                          "actions": [[a.action, rnd(a.gain), rnd(a.gain_lo), rnd(a.gain_hi), rnd(a.share_called, 3)]
                                      for a in cs.itertuples()]}
        out_cells.append({"cell": c, "down_distance": dd, "zone": zone, "score": score, "by_season": by})
    obj = {
        "schema": SCHEMA, "page": "playcalling",
        "model": "M7b early-down play type x personnel: propensity GBM, cross-fit AIPW, off-policy evaluation",
        "refresh": "yearly (static: from the logged 2020-2025 sign-off run and its saved outputs)",
        "license": {**CC_BY_SA, "credit": CREDIT_SA},
        "headline": m7b_headline(runs_dir),
        "definitions": {
            "sample": ("1st and 2nd down scrimmage runs and passes with an EPA, REG and POST, outside the last two "
                       "minutes of each half, garbage time out (nflfastR WP outside 0.05-0.95)"),
            "cells": {"count": len(pol.CELLS), "down_distance": DD_LABELS, "zone": ZONE_LABELS, "score": SCORE_LABELS},
            "actions": {"groups": GROUP_LABELS, "calls": {"run": "designed run", "pass": "pass (sacks and scrambles count)"},
                        "list": list(pol.ACTIONS)},
            "gain": ("estimated EPA per play of calling the action instead of the teams' current mix, on the same plays "
                     "(doubly robust AIPW); gain_lo/gain_hi are the 95% game-cluster bootstrap interval"),
            "best": "the candidate action with the highest gain",
            "clear": ("true when the best candidate's interval lies above 0 (it beats the current mix); otherwise the "
                      "cell is labeled 'no clear best'"),
            "candidate": (f"eligible (propensity >= {pol.OVERLAP}) on at least {int(pol.CAND_SHARE * 100)}% of the "
                          f"cell's plays and actually called at least {pol.MIN_TAKEN} times there"),
            "actions_row": "[action, gain, gain_lo, gain_hi, share of the cell's training plays that called it]",
            "by_season": ("season S's recommendations are estimated from seasons 2016..S-1 only (cross-fit by season), "
                          "then applied to S"),
            "n_train_plays": "training plays in the cell (seasons 2016..S-1)"},
        "seasons": seasons,
        "cells": out_cells,
        "team_view_2025": team_view_2025(d, recs),
        "caveats": [
            "Unmeasured confounding: teams know things the data does not (injuries, the game plan, the opponent's "
            "tendencies). The method removes measured selection, not all of it.",
            "Defenses adapt: if a team always took the 'best' call, defenses would adjust. The estimates describe "
            "marginal shifts from today's tendencies, not a fixed strategy.",
            "The placebo's upper end (+0.012) hints at a small leftover bias, perhaps 0.005 EPA per play, well below "
            "the +0.052 estimated gain.",
            "Cells marked 'no clear best' are not evidence that every action is equal, only that the data cannot rank "
            "them.",
        ],
    }
    return checked(obj)


def team_view_2025(m7b_dir: Path, recs: pd.DataFrame, season: int = 2025) -> dict:
    """Descriptive: each team's 2025 REG early-down pass rate in the cells where the 2025 recommendation (fit on
    2016-2024) is a clear pass, against the league. Observed play calls only; no model is scored."""
    plays = pd.read_parquet(m7b_dir / "plays_holdout.parquet",
                            columns=["season", "season_type", "off_team", "cell", "action"])
    p = plays[(plays["season"] == season) & (plays["season_type"] == "REG")].copy()
    r = recs[(recs["season"] == season) & recs["clear"]]
    pass_cells = r[r["best"].str.endswith("_pass")].set_index("cell")["best"]
    act = np.array(pol.ACTIONS, dtype=object)[p["action"].to_numpy(int)]
    p["is_pass"] = np.char.endswith(act.astype(str), "_pass")
    p["match"] = act == p["cell"].map(pass_cells).to_numpy(object)
    q = p[p["cell"].isin(pass_cells.index)]
    lg_rate, lg_all = float(q["is_pass"].mean()), float(p["is_pass"].mean())
    teams = []
    for team, g in q.groupby("off_team"):
        allp = p[p["off_team"] == team]
        teams.append({"team": team, "plays": int(len(g)), "pass_rate": rnd(g["is_pass"].mean(), 3),
                      "vs_league": rnd(g["is_pass"].mean() - lg_rate, 3),
                      "exact_match_rate": rnd(g["match"].mean(), 3),
                      "early_down_pass_rate_all_cells": rnd(allp["is_pass"].mean(), 3)})
    teams.sort(key=lambda t: -(t["pass_rate"] or 0))
    return {"season": season, "game_type": "REG", "label": "descriptive (observed 2025 play calls)",
            "pass_cells": [{"cell": c, "recommended": b} for c, b in pass_cells.items()],
            "league": {"plays": int(len(q)), "pass_rate": rnd(lg_rate, 3),
                       "exact_match_rate": rnd(q["match"].mean(), 3),
                       "early_down_pass_rate_all_cells": rnd(lg_all, 3)},
            "teams": teams,
            "fields": {"pass_rate": "team's pass share in the cells where 2025's clear recommendation is a pass",
                       "vs_league": "pass_rate minus the league's in the same cells (mix of cells differs by team)",
                       "exact_match_rate": "share of those plays that used the recommended personnel and call",
                       "early_down_pass_rate_all_cells": "team's pass share on every early down in the sample"}}


# =========================================================================== orchestration

def run_part(name: str, fn) -> tuple[bool, float]:
    """Run one export; on failure print a GitHub warning and the traceback, never raise."""
    t0 = time.perf_counter()
    try:
        n = fn()
        dt = time.perf_counter() - t0
        log(f"{name}: wrote {n:,} bytes in {dt:.1f}s")
        return True, dt
    except Exception as e:  # noqa: BLE001  (fail soft: the main site update must never depend on this)
        dt = time.perf_counter() - t0
        log(f"::warning title=export_ml_pages {name} failed::{type(e).__name__}: {e}")
        traceback.print_exc()
        return False, dt


def run_live(S: int = SEASON, no_refresh: bool = False, budget: float | None = None, parts=("winprob", "fourth")) -> int:
    """The weekly step: winprob then fourth, each fail-soft. Returns 0 if both wrote, else 1 (never raises)."""
    t0 = time.perf_counter()
    state = {}

    def inputs():
        if "L" not in state:
            state["L"] = load_live(S, no_refresh)
        return state["L"]

    ok = True
    for part in parts:
        if budget is not None and time.perf_counter() - t0 > budget:
            log(f"::warning title=export_ml_pages::time budget {budget:.0f}s used up; {part} skipped this run")
            ok = False
            continue
        if part == "winprob":
            good, _ = run_part("winprob", lambda: write_json(FILES["winprob"], build_winprob(inputs())))
        else:
            good, _ = run_part("fourth", lambda: write_json(
                FILES["fourth"], build_fourth(inputs(), keep_chart=(read_json(FILES["fourth"]) or {}).get("chart"))))
        ok &= good
    log(f"live parts: {'ok' if ok else 'with warnings'} in {time.perf_counter() - t0:.0f}s")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["plays", "playcalling", "winprob", "fourth", "live", "all"])
    ap.add_argument("--no-refresh", action="store_true",
                    help="skip the nflverse refresh of this season (missing files are still fetched)")
    ap.add_argument("--budget", type=float, default=None, help="live: skip parts started after this many seconds")
    ap.add_argument("--season", type=int, default=SEASON)
    a = ap.parse_args()
    S = a.season
    if a.what == "live":
        return run_live(S, a.no_refresh, a.budget)
    ok = True
    if a.what in ("plays", "all"):
        ok &= run_part("plays", lambda: write_json(FILES["plays"], build_plays(S)))[0]
    if a.what in ("playcalling", "all"):
        ok &= run_part("playcalling", lambda: write_json(FILES["playcalling"], build_playcalling()))[0]
    if a.what in ("winprob", "fourth", "all"):
        parts = ("winprob", "fourth") if a.what == "all" else (a.what,)
        ok &= run_live(S, a.no_refresh, a.budget, parts) == 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
