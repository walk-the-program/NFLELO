"""Elo engine: MOV-scaled updates, season regression, expansion starts, fixed or online HFA."""
from __future__ import annotations

import math

import pandas as pd

from .config import EloConfig

OUTPUT_COLUMNS = [
    "home_pre", "away_pre", "hfa_used", "p_home", "mov", "updated",
    "home_post", "away_post",
]


def expected_home(home_rating: float, away_rating: float, hfa: float) -> float:
    """Home win probability. `hfa` is added to the home rating."""
    return 1.0 / (1.0 + 10.0 ** (-(home_rating + hfa - away_rating) / 400.0))


def mov_multiplier(margin: float, winner_diff: float, cfg: EloConfig) -> float:
    """Margin-of-victory multiplier.

    margin: absolute score margin. winner_diff: winner rating minus loser
    rating, including home-field advantage. Ties return 1.0.
    """
    if margin == 0:
        return 1.0
    d = abs(winner_diff) if cfg.mov_mode == "abs" else winner_diff
    m = math.log(margin + 1.0) * 2.2 / (d * 0.001 + 2.2)
    return min(m, cfg.mov_cap) if cfg.mov_cap is not None else m


def run_elo(games: pd.DataFrame, cfg: EloConfig) -> tuple[pd.DataFrame, dict]:
    """Run Elo over games in chronological order.

    games needs: season, home, away, home_score, away_score, neutral, game_type
    (already sorted by date). Returns (games with OUTPUT_COLUMNS added, final
    state {"ratings", "hfa"}). Playoff games are always predicted; they update
    ratings (and online HFA) only when cfg.include_playoffs is true.
    """
    ratings: dict[str, float] = {}
    online = cfg.hfa_mode == "online"
    hfa = cfg.hfa_init if online else cfg.hfa
    first_season = int(games["season"].iloc[0])
    current_season = first_season
    start, lam, k = cfg.start, cfg.lam, cfg.k

    cols = {c: [] for c in OUTPUT_COLUMNS}
    for season, home, away, hs, as_, neutral, gtype in zip(
        games["season"].tolist(), games["home"].tolist(), games["away"].tolist(),
        games["home_score"].tolist(), games["away_score"].tolist(),
        games["neutral"].tolist(), games["game_type"].tolist(),
    ):
        if season != current_season:
            for t in ratings:
                ratings[t] = (1.0 - lam) * ratings[t] + lam * start
            current_season = season
        for t in (home, away):
            if t not in ratings:
                ratings[t] = start if season == first_season else cfg.expansion_start

        rh, ra = ratings[home], ratings[away]
        h = 0.0 if neutral else hfa
        p = expected_home(rh, ra, h)
        update = gtype == "REG" or cfg.include_playoffs

        mov = 1.0
        if update:
            actual = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
            home_eff = rh + h
            winner_diff = home_eff - ra if hs >= as_ else ra - home_eff
            mov = mov_multiplier(abs(hs - as_), winner_diff, cfg)
            delta = k * mov * (actual - p)
            ratings[home] = rh + delta
            ratings[away] = ra - delta
            if online and not neutral:
                hfa += cfg.k_hfa * (actual - p) * (mov if cfg.hfa_mov else 1.0)

        cols["home_pre"].append(rh)
        cols["away_pre"].append(ra)
        cols["hfa_used"].append(h)
        cols["p_home"].append(p)
        cols["mov"].append(mov)
        cols["updated"].append(update)
        cols["home_post"].append(ratings[home])
        cols["away_post"].append(ratings[away])

    out = games.reset_index(drop=True).assign(**cols)
    return out, {"ratings": ratings, "hfa": hfa}
