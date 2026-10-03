"""Baselines inside the harness.

Prediction columns (all start with `p_`, all are home-win probabilities):

- `p_coin_flip`: 0.5 for every game.
- `p_home_rate`: one constant per season S, the home-win rate (ties 0.5) of
  REG games in the `lookback` seasons before S. Nothing from S or later.
- `p_elo_v2`: Elo v2 (`nflelo.config.DEFAULT_CONFIG`) pre-game probability,
  computed exactly as `nflelo.evaluate.compare_on_test` does.

Benchmark column (decision D3): `bench_market`, the vig-removed moneyline
probability. It is a yardstick, never an input: it lives under its own
`bench_` prefix so `prediction_columns` and any feature matrix exclude it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ... import config
from ...config import EloConfig
from ...elo import run_elo
from ...evaluate import market_prob, outcome
from . import windows

PRED_PREFIX = "p_"
BENCH_PREFIX = "bench_"
MARKET_COL = "bench_market"
BASELINES = ("p_coin_flip", "p_home_rate", "p_elo_v2")
HOME_RATE_LOOKBACK = 10  # seasons; the home edge has fallen from ~60% (1990s) to ~53% (2020s)


def coin_flip(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.5, index=df.index, name="p_coin_flip")


def home_rate(df: pd.DataFrame, history: pd.DataFrame, lookback: int | None = HOME_RATE_LOOKBACK) -> pd.Series:
    """Per-season constant home-win rate learned only from REG games of earlier seasons in `history`."""
    reg = history[history["game_type"] == "REG"]
    y = pd.Series(outcome(reg), index=reg.index)
    out = pd.Series(np.nan, index=df.index, name="p_home_rate")
    for s in sorted(df["season"].unique()):
        m = reg["season"] < s
        if lookback is not None:
            m &= reg["season"] >= s - lookback
        if not m.any():
            raise ValueError(f"no REG games before season {s} to learn a home-win rate from")
        out[df["season"] == s] = float(y[m].mean())
    return out


def elo_v2(games: pd.DataFrame, seasons: tuple[int, int], cfg: EloConfig = config.DEFAULT_CONFIG) -> pd.Series:
    """Elo pre-game home-win probability for every game through `seasons[1]` (index matches `games`).

    Elo is sequential, so a game's probability uses only earlier games; the
    season cut only saves work. Note Elo updates game by game, so a Sunday game
    does see that week's Thursday result (finer than the ML as-of rule).
    """
    elo, _ = run_elo(games[games["season"] <= int(seasons[1])], cfg)
    return elo["p_home"].rename("p_elo_v2")


def market(df: pd.DataFrame) -> pd.Series:
    """Vig-removed moneyline probability. BENCHMARK ONLY (decision D3)."""
    return market_prob(df).rename(MARKET_COL)


def baseline_frame(games: pd.DataFrame, seasons: tuple[int, int], allow_holdout: bool = False,
                   require_market: bool = True, home_rate_lookback: int | None = HOME_RATE_LOOKBACK,
                   cfg: EloConfig = config.DEFAULT_CONFIG) -> pd.DataFrame:
    """One row per REG game in the window, with the outcome, every baseline, and the market benchmark.

    `games` is `nflelo.data.load_games()` (1970 onward; needed for Elo warm-up
    and the home-rate history). With `require_market`, only games that have
    moneylines are kept, so every column is scored on identical games.
    """
    elo_p = elo_v2(games, seasons, cfg)
    w = windows.select(games, seasons, allow_holdout)
    out = w[["game_id", "season", "week", "game_type", "home", "away", "home_score", "away_score"]].copy()
    out["y"] = outcome(w)
    out["p_coin_flip"] = coin_flip(w)
    out["p_home_rate"] = home_rate(w, games, home_rate_lookback)
    out["p_elo_v2"] = elo_p.loc[w.index]
    out[MARKET_COL] = market(w)
    if require_market:
        out = out[out[MARKET_COL].notna()]
    return out


def prediction_columns(df: pd.DataFrame) -> list[str]:
    """Model prediction columns (`p_*`). Never includes the market benchmark."""
    return [c for c in df.columns if c.startswith(PRED_PREFIX)]
