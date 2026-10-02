"""Metrics, grid tuning, and held-out comparison against legacy and the market."""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from . import config
from .config import EloConfig
from .elo import run_elo

# --------------------------------------------------------------------------- metrics

def outcome(df: pd.DataFrame) -> np.ndarray:
    """Home result: 1 win, 0.5 tie, 0 loss."""
    return np.where(df["home_score"] > df["away_score"], 1.0,
                    np.where(df["home_score"] == df["away_score"], 0.5, 0.0))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Brier, log loss (ties count 0.5, as in the legacy script), accuracy (ties excluded)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    pc = np.clip(p, 1e-12, 1 - 1e-12)
    decided = y != 0.5
    return {
        "n": int(len(y)),
        "brier": float(np.mean((y - p) ** 2)),
        "logloss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
        "accuracy": float(np.mean((p[decided] > 0.5) == (y[decided] == 1.0))),
    }


def window(elo: pd.DataFrame, seasons: tuple[int, int]) -> pd.DataFrame:
    """REG games in an inclusive season window."""
    return elo[(elo["game_type"] == "REG") & elo["season"].between(*seasons)]


def score(elo: pd.DataFrame, seasons: tuple[int, int]) -> dict:
    w = window(elo, seasons)
    return metrics(outcome(w), w["p_home"].to_numpy())


def market_prob(games: pd.DataFrame) -> pd.Series:
    """Home win probability from moneylines with the vig removed (NaN if missing)."""
    def implied(ml: pd.Series) -> pd.Series:
        ml = pd.to_numeric(ml, errors="coerce")
        return np.where(ml < 0, -ml / (-ml + 100.0), 100.0 / (ml + 100.0))
    ph, pa = implied(games["home_moneyline"]), implied(games["away_moneyline"])
    return pd.Series(ph / (ph + pa), index=games.index)


# --------------------------------------------------------------------------- tuning

def default_grid() -> list[EloConfig]:
    """Full grid; the first five axes are the required ones, the last two are extras."""
    hfa_variants = (
        [dict(hfa_mode="fixed", hfa=h) for h in (35, 45, 55, 65)]
        + [dict(hfa_mode="online", hfa_init=65.0, k_hfa=kh, hfa_mov=mv)
           for kh in (0.5, 1, 2, 4) for mv in (False, True)]
    )
    cfgs = []
    for k, lam, hv, po, ex, cap in itertools.product(
        (15, 20, 25, 30),
        (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50),
        hfa_variants,
        (True, False),
        (1300.0, 1500.0),
        (None, 2.0),
    ):
        cfgs.append(EloConfig(k=float(k), lam=lam, include_playoffs=po,
                              expansion_start=ex, mov_cap=cap, **hv))
    return cfgs


def tune(games: pd.DataFrame, grid: list[EloConfig],
         tune_seasons: tuple[int, int] = config.TUNE_SEASONS) -> pd.DataFrame:
    """Score every config on the tuning window. Only games through the window end are used."""
    g = games[games["season"] <= tune_seasons[1]]
    rows = []
    for cfg in grid:
        elo, _ = run_elo(g, cfg)
        rows.append({**cfg.to_dict(), **score(elo, tune_seasons)})
    return pd.DataFrame(rows).sort_values("brier").reset_index(drop=True)


def config_from_row(row: pd.Series) -> EloConfig:
    fields = EloConfig.__dataclass_fields__
    kw = {}
    for name in fields:
        v = row[name]
        if name == "mov_cap":
            v = None if pd.isna(v) else float(v)
        elif name in ("hfa_mode", "mov_mode"):
            v = str(v)
        elif name in ("hfa_mov", "include_playoffs"):
            v = bool(v)
        else:
            v = float(v)
        kw[name] = v
    return EloConfig(**kw)


def best_online(res: pd.DataFrame) -> EloConfig | None:
    """Best online-HFA config in a tuning table (tune-window score only), for reference."""
    online = res[res["hfa_mode"] == "online"]
    return config_from_row(online.iloc[0]) if len(online) else None


# --------------------------------------------------------------------------- test comparison

def compare_on_test(games: pd.DataFrame, models: dict[str, EloConfig],
                    seasons: tuple[int, int] = config.TEST_SEASONS) -> dict:
    """Score each config on the test window, and with the market on games that have moneylines.

    Returns {"all": table over all test REG games, "market": table over test REG
    games with moneylines (identical games for every row), "paired": tuned-vs-market
    Brier difference stats for the first model}.
    """
    runs = {name: window(run_elo(games[games["season"] <= seasons[1]], cfg)[0], seasons)
            for name, cfg in models.items()}
    first = next(iter(runs.values()))
    y_all = outcome(first)
    all_rows = [{"model": n, **metrics(y_all, r["p_home"].to_numpy())} for n, r in runs.items()]

    mp = market_prob(first)
    has = mp.notna().to_numpy()
    y_m = y_all[has]
    mkt_rows = [{"model": n, **metrics(y_m, r["p_home"].to_numpy()[has])} for n, r in runs.items()]
    mkt_rows.append({"model": "market (vig removed)", **metrics(y_m, mp.to_numpy()[has])})

    name0 = next(iter(runs))
    diff = (y_m - runs[name0]["p_home"].to_numpy()[has]) ** 2 - (y_m - mp.to_numpy()[has]) ** 2
    paired = {
        "model": name0, "n": int(has.sum()),
        "brier_diff_vs_market": float(diff.mean()),
        "se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
    }
    return {"all": pd.DataFrame(all_rows), "market": pd.DataFrame(mkt_rows), "paired": paired,
            "market_seasons": (int(first.loc[has, "season"].min()), int(first.loc[has, "season"].max()))}


def legacy_reproduction(legacy_games: pd.DataFrame) -> dict:
    """Legacy parameters on legacy-xlsx-only data, all REG games 1970-2025."""
    elo, _ = run_elo(legacy_games, config.LEGACY_CONFIG)
    return score(elo, (int(elo["season"].min()), int(elo["season"].max())))


def timed(fn, *a, **kw):
    t = time.time()
    out = fn(*a, **kw)
    return out, time.time() - t
