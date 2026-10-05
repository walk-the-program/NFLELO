"""Lineup features for the game model (context/ml-m5-method.md, section 5).

For each team before each game, with player values v as of the game's week
(players.value; box-score or with/without):

    expected   = sum over the pregame lineup (players.lineup) of p_play * expected share * v
    remembered = weighted average, over the team's games in the rating window (this
                 season and last, game weights 0.5 ** (weeks_ago / 48) * 0.25 ** seasons_ago,
                 exactly the M3 rating weights), of sum over who was on the field of share * v
    last_season = plain average over the team's games last season of sum share * v

Both sums leave out the QB (he has his own term, qb_delta) and use the SAME
current values, so the difference measures who is (not) playing, not how
values changed. Features, home minus away:

    lineup_delta_diff        (expected - remembered), all non-QB players
    lineup_delta_off_diff    the same, offense and kicker only (ladder step B4)
    lineup_delta_def_diff    the same, defense only (B4)
    preseason_change_diff    (expected - last_season): this season's roster against last
                             season's, valued with today's values, so free agency, trades,
                             retirements and rookies (at replacement level) show up
    preseason_change_diff_wk preseason_change_diff * (min(week, 10) - 1) / 9: lets the
                             model learn how fast the preseason term fades
    starters_out_home/away   rank-1 non-QB starters with p_play < 0.5 (reporting only)

"Who was on the field" in past games: usage-based shares for QB, skill players,
defenders and kickers (credit.player_games), and the pregame lineup for
linemen (lineup.ol_games). The current game's own usage is never read; the
`leaky=True` variant reads it on purpose (the positive control for the
leakage check).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .. import asof as asof_mod
from ..players import credit as cr
from ..players import lineup as lu
from ..players import positions as pos
from ..players import value as pv
from . import opponent_adjust as oa
from .team_efficiency import _utc_ns

FEATURES = ["lineup_delta_diff", "lineup_delta_off_diff", "lineup_delta_def_diff", "preseason_change_diff",
            "preseason_change_diff_wk"]
FADE_WEEKS = 10


@dataclass(frozen=True)
class RosterConfig:
    values: str = "box"                                  # "box" or "ww" (with/without)
    lineup: lu.LineupConfig = lu.LineupConfig()
    value_cfg: pv.ValueConfig = pv.ValueConfig()
    rating: oa.RatingConfig = oa.TUNED                   # remembered-lineup weights = the rating weights
    leaky: bool = False                                  # positive control: use the game's actual players


@dataclass
class RosterState:
    """Everything the features need, precomputed once for a season range."""
    sched: pd.DataFrame
    pg: pd.DataFrame                 # credit.player_games
    onfield: pd.DataFrame            # pg rows plus linemen rows (share = pregame p_play)
    lineups: dict                    # LineupConfig -> lineup rows (LINEUP_COLUMNS + share_exp)
    probs: pd.DataFrame              # status probabilities (fixed table)
    R: dict                          # season -> replacement levels
    values: dict = field(default_factory=dict)   # (kind, value_cfg) -> long values frame


def prepare(pbp: pd.DataFrame, sched: pd.DataFrame, depth: pd.DataFrame, injuries: pd.DataFrame | None,
            rosters: pd.DataFrame | None, positions: pd.DataFrame, probs: pd.DataFrame | None = None,
            availability: pd.DataFrame | None = None, value_cfg: pv.ValueConfig = pv.ValueConfig(),
            lineup_cfgs=(lu.LineupConfig(),), roster_seasons=None) -> RosterState:
    """Credits, lineups (for each lineup variant), expected shares and replacement levels.

    `probs` (status -> probability by group) is a FIXED table estimated once on
    2009-2015 (spec section 4); pass it explicitly. If None it is estimated
    here from `lineup.ESTIMATION_SEASONS`, which is only appropriate when those
    seasons are entirely before the games being predicted. `availability` (the
    era-gap table) is likewise a fixed property of the data; pass it to keep a
    leakage check exact. `roster_seasons` (lineup.roster_status_seasons) are
    the seasons whose weekly-roster statuses may be used; None means all
    (only safe on data known to be week-accurate, e.g. synthetic tests).
    """
    if "as_of" not in sched.columns:
        sched = asof_mod.add_asof(sched)
    pg = cr.player_games(pbp, sched, positions, depth, availability)
    base = lu.attach_statuses(lu.chart_players(depth), injuries, rosters, roster_seasons)
    if probs is None:
        probs = lu.estimate_status_probs(base, pg[pg["inv"] > 0])
    full = base.copy()
    full["p_play"] = lu.play_probability(full, probs, lu.LineupConfig())
    ol = lu.ol_games(full, sched)  # past games: every layer known by now (inactives included)
    onfield = pd.concat([pg[["game_id", "season", "week", "kick_ns", "ord", "team", "gsis_id", "group", "side",
                             "share"]], ol], ignore_index=True).sort_values("kick_ns", kind="stable")
    seasons = sorted(set(base["season"].astype(int)))
    defaults = {S: lu.default_shares(pg, full, S) for S in seasons}
    share_exp = lu.expected_shares(pg, base, sched, value_cfg.half_life, defaults)
    lineups = {}
    for c in lineup_cfgs:
        x = base.copy()
        x["p_play"] = lu.play_probability(x, probs, c)
        x["share_exp"] = share_exp
        lineups[c] = x
    R = pv.replacement_levels(pg, sorted(set(pg["season"]) | set(seasons)), value_cfg.early_games)
    return RosterState(sched, pg, onfield.reset_index(drop=True), lineups, probs, R)


def compute_values(state: RosterState, keys, cfg: pv.ValueConfig, ratings: pd.DataFrame | None = None,
                   tab: pd.DataFrame | None = None, with_ridge: bool = True) -> pd.DataFrame:
    """Box (and with/without) values at each key: season, week, gsis_id, group, n, v_box[, v_ww]."""
    bv = pv.BoxValues(state.pg, state.sched)
    box = pv.box_values(bv, keys, cfg, state.R)
    if not with_ridge:
        return box
    if ratings is None or tab is None:
        raise ValueError("the with/without layer needs the M3 rating table and ratings")
    rows = pv.residual_rows(tab, ratings)
    rows = rows[rows["game_id"].isin(state.sched.loc[state.sched["game_type"] == "REG", "game_id"])]
    rd = pv.RidgeData(rows, state.onfield, state.sched)
    ww = pv.ridge_values(rd, box, keys, cfg)
    out = box.merge(ww, on=["season", "week", "gsis_id"], how="outer")
    out["v_box"] = out["v_box"].fillna(0.0)
    out["v_ww"] = out["v_ww"].fillna(out["v_box"])
    if "group" in out.columns:  # players with no box row (linemen) are linemen
        out["group"] = out["group"].fillna("OL")
    return out


# --------------------------------------------------------------------------- team sums

def _team_codes(games: pd.DataFrame, side: str) -> np.ndarray:
    return pos.map_teams(games[f"{side}_team"], games["season"]).to_numpy(object)


def team_lineup_terms(state: RosterState, values: pd.DataFrame, vcol: str, games: pd.DataFrame,
                      lcfg: lu.LineupConfig, rating: oa.RatingConfig, leaky: bool = False) -> pd.DataFrame:
    """Per game and side: expected, remembered and last-season lineup values (all, off, def), and starters out."""
    sched = state.sched
    asofs, ords = oa.week_asof(sched), oa.week_ordinals(sched)
    of = state.onfield[state.onfield["group"] != "QB"]
    of_kick = of["kick_ns"].to_numpy()
    lineup = state.lineups[lcfg]
    lineup = lineup[lineup["group"] != "QB"]
    out_rows = []
    vals_by_key = {k: g.set_index("gsis_id")[vcol] for k, g in values.groupby(["season", "week"])}
    tg = pd.DataFrame({"game_id": np.concatenate([games["game_id"].to_numpy()] * 2),
                       "season": np.concatenate([games["season"].to_numpy(int)] * 2),
                       "week": np.concatenate([games["week"].to_numpy(int)] * 2),
                       "side": ["home"] * len(games) + ["away"] * len(games),
                       "team": np.concatenate([_team_codes(games, "home"), _team_codes(games, "away")])})
    # team-games already played, for the remembered and last-season denominators
    played = of.drop_duplicates(["game_id", "team"])[["game_id", "team", "season", "kick_ns", "ord"]]
    for (S, W), idx in tg.groupby(["season", "week"]).indices.items():
        S, W = int(S), int(W)
        v = vals_by_key.get((S, W), pd.Series(dtype=float))
        sub = tg.iloc[idx]
        teams = set(sub["team"])
        # expected lineup
        if leaky:  # positive control: the players who actually played in this very game
            g_ids = set(sub["game_id"])
            x = of[of["game_id"].isin(g_ids)]
            x = x.assign(contrib=x["share"].to_numpy() * x["gsis_id"].map(v).fillna(0.0).to_numpy())
        else:
            x = lineup[(lineup["season"] == S) & (lineup["week"] == W) & lineup["team"].isin(teams)]
            x = x.assign(contrib=(x["p_play"] * x["share_exp"]).to_numpy() * x["gsis_id"].map(v).fillna(0.0).to_numpy())
        x = x.assign(sd=np.where(x["group"].map(pos.SIDE) == "def", "def", "off"))
        exp = x.groupby(["team", "sd"])["contrib"].sum().unstack(fill_value=0.0)
        # remembered: games before as_of in seasons S-1..S, M3 rating weights
        end = np.searchsorted(of_kick, asofs[(S, W)], side="left")
        past = of.iloc[:end]
        past = past[(past["season"] >= S - 1) & past["team"].isin(teams)]
        w = oa.decay_weights(int(ords[(S, W)]), past["ord"].to_numpy(float), S, past["season"].to_numpy(int),
                             rating.half_life, rating.rho)
        contrib = past["share"].to_numpy() * past["gsis_id"].map(v).fillna(0.0).to_numpy()
        sd = np.where(past["side"].to_numpy() == "def", "def", "off")
        rem_num = pd.DataFrame({"team": past["team"].to_numpy(), "sd": sd, "x": w * contrib}).groupby(
            ["team", "sd"])["x"].sum().unstack(fill_value=0.0)
        pp = played[(played["kick_ns"] < asofs[(S, W)]) & (played["season"] >= S - 1) & played["team"].isin(teams)]
        pw = oa.decay_weights(int(ords[(S, W)]), pp["ord"].to_numpy(float), S, pp["season"].to_numpy(int),
                              rating.half_life, rating.rho)
        rem_den = pd.Series(pw, index=pp["team"].to_numpy()).groupby(level=0).sum()
        last = past[past["season"] == S - 1]
        lc = last["share"].to_numpy() * last["gsis_id"].map(v).fillna(0.0).to_numpy()
        lsd = np.where(last["side"].to_numpy() == "def", "def", "off")
        last_num = pd.DataFrame({"team": last["team"].to_numpy(), "sd": lsd, "x": lc}).groupby(
            ["team", "sd"])["x"].sum().unstack(fill_value=0.0)
        last_den = pp[pp["season"] == S - 1].groupby("team").size()
        so = lu.starters_out(lineup[(lineup["season"] == S) & (lineup["week"] == W)]) if not leaky else None
        for i in idx:
            t = tg.at[i, "team"]
            rec = {"row": i}
            for part in ("off", "def"):
                e = float(exp.at[t, part]) if t in exp.index and part in exp.columns else 0.0
                rn = float(rem_num.at[t, part]) if t in rem_num.index and part in rem_num.columns else 0.0
                rd = float(rem_den.get(t, 0.0))
                ln = float(last_num.at[t, part]) if t in last_num.index and part in last_num.columns else 0.0
                ld = float(last_den.get(t, 0))
                rec[f"exp_{part}"] = e
                rec[f"rem_{part}"] = rn / rd if rd > 0 else e
                rec[f"last_{part}"] = ln / ld if ld > 0 else e
            rec["has_lineup"] = bool(t in exp.index)
            rec["starters_out"] = int(so.get((S, W, t), 0)) if so is not None else 0
            out_rows.append(rec)
    res = pd.DataFrame(out_rows).set_index("row").sort_index()
    return pd.concat([tg, res], axis=1)


def features_from_terms(terms: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Home-minus-away features (FEATURES) from `team_lineup_terms`, indexed by game_id."""
    t = terms.copy()
    for part in ("off", "def"):
        t[f"delta_{part}"] = np.where(t["has_lineup"], t[f"exp_{part}"] - t[f"rem_{part}"], 0.0)
        t[f"pre_{part}"] = np.where(t["has_lineup"], t[f"exp_{part}"] - t[f"last_{part}"], 0.0)
    h = t[t["side"] == "home"].set_index("game_id")
    a = t[t["side"] == "away"].set_index("game_id")
    gid = games["game_id"].to_numpy()
    h, a = h.loc[gid], a.loc[gid]
    out = pd.DataFrame(index=gid)
    out["lineup_delta_off_diff"] = (h["delta_off"] - a["delta_off"]).to_numpy()
    out["lineup_delta_def_diff"] = (h["delta_def"] - a["delta_def"]).to_numpy()
    out["lineup_delta_diff"] = out["lineup_delta_off_diff"] + out["lineup_delta_def_diff"]
    pre = ((h["pre_off"] + h["pre_def"]) - (a["pre_off"] + a["pre_def"])).to_numpy()
    out["preseason_change_diff"] = pre
    wk = games["week"].to_numpy(float)
    out["preseason_change_diff_wk"] = pre * (np.minimum(wk, FADE_WEEKS) - 1) / (FADE_WEEKS - 1)
    out["starters_out_home"] = h["starters_out"].to_numpy()
    out["starters_out_away"] = a["starters_out"].to_numpy()
    out.index.name = "game_id"
    return out


def build_features(pbp: pd.DataFrame, sched: pd.DataFrame, games: pd.DataFrame, depth: pd.DataFrame,
                   injuries: pd.DataFrame | None, rosters: pd.DataFrame | None, positions: pd.DataFrame,
                   probs: pd.DataFrame, availability: pd.DataFrame | None = None,
                   cfg: RosterConfig = RosterConfig(), roster_seasons=None) -> pd.DataFrame:
    """End to end from raw tables (used by the leakage checks): credits, lineups, values, features."""
    if "as_of" not in sched.columns:
        sched = asof_mod.add_asof(sched)
    if "as_of" not in games.columns:
        games = games.merge(sched[["game_id", "kickoff", "as_of"]], on="game_id", how="left")
    st = prepare(pbp, sched, depth, injuries, rosters, positions, probs, availability, cfg.value_cfg,
                 (cfg.lineup,), roster_seasons)
    keys = sorted(set(zip(games["season"].astype(int), games["week"].astype(int))))
    if cfg.values == "ww":
        tab = oa.rating_table(pbp, sched, cfg.rating)
        rkeys = sorted(set(zip(tab["season"].astype(int), tab["week"].astype(int))))
        ratings = oa.compute_ratings(tab, sched, rkeys, cfg.rating, ("all",))
        vals = compute_values(st, keys, cfg.value_cfg, ratings, tab, with_ridge=True)
        vcol = "v_ww"
    else:
        vals = compute_values(st, keys, cfg.value_cfg, with_ridge=False)
        vcol = "v_box"
    terms = team_lineup_terms(st, vals, vcol, games, cfg.lineup, cfg.rating, leaky=cfg.leaky)
    return features_from_terms(terms, games)[FEATURES]


# --------------------------------------------------------------------------- leakage check

def corrupt_aux(tables: dict, season: int, week: int, seed: int = 0) -> dict:
    """Copies of the weekly tables with every row AFTER (season, week) scrambled (player IDs and statuses shuffled).

    The game's own week is left alone: its depth chart, injury report and
    inactives are pregame facts (decision M5-D2). Later weeks are not.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for name, df in tables.items():
        if df is None or df.empty:
            out[name] = df
            continue
        d = df.copy()
        late = ((d["season"] > season) | ((d["season"] == season) & (d["week"] > week))).to_numpy()
        k = int(late.sum())
        if k:
            for c in ("gsis_id", "status", "report_status", "roster_status", "group", "rank"):
                if c in d.columns:
                    d[c] = d[c].astype(object)
                    d.loc[late, c] = rng.permutation(d.loc[late, c].to_numpy(object))
        out[name] = d
    return out


def roster_leakage_check(builder, pbp: pd.DataFrame, sched: pd.DataFrame, tables: dict, game_ids,
                         seed: int = 0) -> pd.DataFrame:
    """`asof.leakage_check` extended to the weekly tables.

    For each game: rebuild features after (1) `asof.corrupt_from` scrambles all
    play-by-play and scores at or after the game's as_of (including every
    player ID column on those plays) and (2) `corrupt_aux` scrambles depth
    charts, injuries and rosters of later weeks. `builder(pbp, sched, games,
    tables)` returns features indexed by game_id. Identical features = leak-free.
    """
    if "as_of" not in sched.columns:
        sched = asof_mod.add_asof(sched)
    rows = []
    for gid in game_ids:
        game = sched[sched["game_id"] == gid]
        S, W, as_of = int(game["season"].iloc[0]), int(game["week"].iloc[0]), game["as_of"].iloc[0]
        clean = builder(pbp, sched, game, tables).loc[[gid]]
        p2, s2 = asof_mod.corrupt_from(pbp, sched, as_of, seed)
        t2 = corrupt_aux(tables, S, W, seed)
        dirty = builder(p2, s2, s2[s2["game_id"] == gid], t2).loc[[gid]]
        a, b = clean.to_numpy(float), dirty.loc[:, clean.columns].to_numpy(float)
        same = np.isclose(a, b, rtol=0, atol=1e-12) | (np.isnan(a) & np.isnan(b))
        rows.append({"game_id": gid, "as_of": as_of, "changed": int((~same).sum()), "leak_free": bool(same.all())})
    return pd.DataFrame(rows)
