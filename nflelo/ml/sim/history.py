"""Actual playoff fields and seeds, for validating the tiebreakers and scoring the simulation backtest.

Ground truth comes from nflverse's playoff games wherever the bracket fixes it:
the teams that played in the postseason, which ones had a bye, the wild-card
pairings (seeds summing to n + byes + 1, higher seed at home), the #1 seed
hosting the lowest remaining seed in the divisional round, and higher seeds
hosting through the conference championship. Every seeding consistent with
those facts is enumerated. Where more than one remains (33 of the 48
conference-seasons 2002-2025, e.g. #1 vs #2 before 2020), the public record
decides: `data/sources/playoff_seeds_2002_2025.csv`, from the `seed` column of
nflverse's nfldata `data/standings.csv` (see data/sources/README.md). The
public seeding must itself be one the bracket allows, or loading fails.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ... import config
from ...meta import conference
from ...teams import franchise
from . import tiebreak

SEEDS_CSV = config.ROOT / "data" / "sources" / "playoff_seeds_2002_2025.csv"


def public_seeds(path=SEEDS_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def _post(sched: pd.DataFrame, season: int) -> pd.DataFrame:
    p = sched[(sched["season"] == season) & (sched["game_type"].isin(["WC", "DIV", "CON", "SB"]))].copy()
    p["h"] = [franchise(c, season) for c in p["home_team"]]
    p["a"] = [franchise(c, season) for c in p["away_team"]]
    p["winner"] = np.where(p["home_score"] > p["away_score"], p["h"], p["a"])
    return p


def bracket_seedings(sched: pd.DataFrame, season: int) -> dict:
    """{conf: [seeding dicts team -> seed consistent with the season's playoff games]}."""
    n = tiebreak.n_seeds(season)
    byes_n = 2 if n == 6 else 1
    post = _post(sched, season)
    out = {}
    for conf in tiebreak.CONFS:
        g = post[np.array([conference(h) == conf for h in post["h"]], bool) & (post["game_type"] != "SB").to_numpy()]
        teams = sorted(set(g["h"]) | set(g["a"]))
        if len(teams) != n:
            raise ValueError(f"{season} {conf}: {len(teams)} playoff teams in nflverse, expected {n}")
        wc = g[g["game_type"] == "WC"]
        byes = set(teams) - set(wc["h"]) - set(wc["a"])
        ok = []
        for perm in itertools.permutations(range(1, n + 1)):
            seed = dict(zip(teams, perm))
            if sorted(seed[t] for t in byes) != list(range(1, byes_n + 1)):
                continue
            if any(seed[r.h] + seed[r.a] != n + byes_n + 1 or seed[r.h] > seed[r.a] for r in wc.itertuples()):
                continue
            remaining = sorted([seed[t] for t in byes] + [seed[w] for w in wc["winner"]])
            div = g[g["game_type"] == "DIV"]
            if any(seed[r.h] > seed[r.a] or (seed[r.h] == 1 and seed[r.a] != remaining[-1]) for r in div.itertuples()):
                continue
            if any(seed[r.h] > seed[r.a] for r in g[g["game_type"] == "CON"].itertuples()):
                continue
            ok.append(seed)
        out[conf] = ok
    return out


def true_seeds(sched: pd.DataFrame, season: int, public: pd.DataFrame | None = None) -> dict:
    """{conf: {"seeds": [team for seed 1..n], "from": "bracket" or "public record", "options": k}}."""
    public = public_seeds() if public is None else public
    out = {}
    for conf, options in bracket_seedings(sched, season).items():
        rows = public[(public["season"] == season) & (public["conf"] == conf)].sort_values("seed")
        pub = dict(zip(rows["team"], rows["seed"].astype(int)))
        if len(options) == 1:
            seeding, src = options[0], "bracket"
            if pub and pub != seeding:
                raise ValueError(f"{season} {conf}: public seeds {pub} contradict the bracket {seeding}")
        else:
            if pub not in options:
                raise ValueError(f"{season} {conf}: public seeds {pub} are not among the {len(options)} the bracket allows")
            seeding, src = pub, "public record"
        out[conf] = {"seeds": [t for t, _ in sorted(seeding.items(), key=lambda kv: kv[1])], "from": src,
                     "options": len(options)}
    return out


def reproduce(sched: pd.DataFrame, seasons=range(2006, 2026)) -> pd.DataFrame:
    """One row per (season, conf): our seeds from actual results vs the truth, and the steps that decided ties."""
    public = public_seeds()
    rows = []
    for S in seasons:
        truth = true_seeds(sched, S, public)
        log: list = []
        ours = tiebreak.actual_seeds(sched, S, log)
        pts = [f"{x['step']}: {'/'.join(x['group'])} -> {x['chosen']}" for x in log if x["points_based"]]
        for conf in tiebreak.CONFS:
            t = truth[conf]["seeds"]
            rows.append({"season": S, "conf": conf, "match": ours[conf] == t, "ours": " ".join(ours[conf]),
                         "truth": " ".join(t), "truth_from": truth[conf]["from"], "bracket_options": truth[conf]["options"],
                         "points_or_coin_steps": "; ".join(pts)})
    return pd.DataFrame(rows)


def outcomes(sched: pd.DataFrame, season: int) -> pd.DataFrame:
    """Actual season outcomes per team: made playoffs, won division, #1 seed, reached and won the Super Bowl."""
    truth = true_seeds(sched, season)
    post = _post(sched, season)
    sb = post[post["game_type"] == "SB"]
    rows = []
    teams = sorted({franchise(c, season) for c in sched.loc[(sched["season"] == season) & (sched["game_type"] == "REG"),
                                                             "home_team"]})
    seeds = {t: i + 1 for c in truth.values() for i, t in enumerate(c["seeds"])}
    for t in teams:
        s = seeds.get(t)
        rows.append({"team": t, "playoffs": s is not None, "division": s is not None and s <= 4,
                     "seed1": s == 1, "reach_sb": bool(len(sb)) and t in set(sb["h"]) | set(sb["a"]),
                     "win_sb": bool(len(sb)) and t in set(sb["winner"])})
    return pd.DataFrame(rows).set_index("team")
