"""Game context from the schedule, and the Elo feature.

Context (context/ml-m3-method.md, section 3):

    rest_diff   home minus away days of rest (schedule `home_rest`, `away_rest`),
                capped at +/-7 so a bye week doesn't count as huge; missing = 0
    neutral     1 for a neutral-site game (schedule `location` == "Neutral")

Both are fixed when the schedule is published, long before the week starts.

Elo (feature 1):

    elo_logit   Elo v2's log-odds for the home team:
                (home_pre - away_pre + HFA) * ln(10) / 400, with HFA = 0 at a
                neutral site.

Elo v2's ratings are pre-game and change only when a team plays, so a team's
rating at kickoff equals its rating at the week's `as_of`. The online
home-field edge is league-wide and moves after every game, including that
week's Thursday game, so here it is frozen at its value before the week's
first game (spec section 7). That is the only difference from Elo's own
`p_home`.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ... import config
from ...config import EloConfig
from ...data import nflverse_to_games
from ...elo import run_elo
from .. import asof as asof_mod

REST_CAP = 7.0
CONTEXT_FEATURES = ["rest_diff", "neutral"]
ELO_FEATURE = "elo_logit"
LN10_400 = math.log(10.0) / 400.0


def build_features(pbp: pd.DataFrame, sched: pd.DataFrame, games: pd.DataFrame | None = None) -> pd.DataFrame:
    """rest_diff and neutral for each game in `games` (default: all of `sched`), indexed by game_id."""
    games = sched if games is None else games
    hr = pd.to_numeric(games["home_rest"], errors="coerce") if "home_rest" in games else pd.Series(np.nan, index=games.index)
    ar = pd.to_numeric(games["away_rest"], errors="coerce") if "away_rest" in games else pd.Series(np.nan, index=games.index)
    rest = np.clip((hr - ar).fillna(0.0).to_numpy(float), -REST_CAP, REST_CAP)
    loc = games["location"] if "location" in games.columns else pd.Series("Home", index=games.index)
    out = pd.DataFrame({"rest_diff": rest, "neutral": (loc == "Neutral").to_numpy(float)},
                       index=games["game_id"].to_numpy())
    out.index.name = "game_id"
    return out


# --------------------------------------------------------------------------- Elo

def hfa_before(elo_out: pd.DataFrame, cfg: EloConfig) -> np.ndarray:
    """The online HFA in force before each game of a `run_elo` output (same order).

    Rebuilt from the update rule in `nflelo.elo.run_elo`; for non-neutral games it
    equals the output's `hfa_used` (a test checks this).
    """
    if cfg.hfa_mode != "online":
        return np.full(len(elo_out), float(cfg.hfa))
    hs, as_ = elo_out["home_score"].to_numpy(float), elo_out["away_score"].to_numpy(float)
    actual = np.where(hs > as_, 1.0, np.where(hs == as_, 0.5, 0.0))
    step = cfg.k_hfa * (actual - elo_out["p_home"].to_numpy(float))
    if cfg.hfa_mov:
        step = step * elo_out["mov"].to_numpy(float)
    live = elo_out["updated"].to_numpy(bool) & ~elo_out["neutral"].to_numpy(bool)
    step = np.where(live, step, 0.0)
    return cfg.hfa_init + np.concatenate([[0.0], np.cumsum(step)[:-1]])


def elo_features_from_games(games: pd.DataFrame, cfg: EloConfig = config.DEFAULT_CONFIG) -> pd.DataFrame:
    """elo_logit for every game in an `nflelo.data` games frame (1970+), indexed by game_id.

    The week's HFA is the HFA before the first game of that (season, week) in
    Elo's own (chronological) order.
    """
    out, _ = run_elo(games, cfg)
    hb = hfa_before(out, cfg)
    first = pd.Series(np.arange(len(out))).groupby([out["season"].to_numpy(), out["week"].to_numpy()]).transform("min")
    hfa_week = hb[first.to_numpy()]
    neutral = out["neutral"].to_numpy(bool)
    d = out["home_pre"].to_numpy(float) - out["away_pre"].to_numpy(float) + np.where(neutral, 0.0, hfa_week)
    res = pd.DataFrame({ELO_FEATURE: d * LN10_400, "hfa_week": hfa_week}, index=out["game_id"].to_numpy())
    res.index.name = "game_id"
    return res


def build_elo_features(pbp: pd.DataFrame, sched: pd.DataFrame, games: pd.DataFrame | None = None,
                       history: pd.DataFrame | None = None, cfg: EloConfig = config.DEFAULT_CONFIG) -> pd.DataFrame:
    """elo_logit for each game in `games`, in the (pbp, sched, games) builder form used by the leakage check.

    `sched` is an nflverse schedule with scores; `history` (optional) holds
    earlier games in `nflelo.data` form to warm the ratings up. `pbp` is unused.
    """
    if "kickoff" not in sched.columns:
        sched = asof_mod.add_asof(sched)
    games = sched if games is None else games
    s = sched.sort_values("kickoff", kind="stable")
    g = nflverse_to_games(s)
    # nflverse_to_games sorts by (date, game_id); keep true kickoff order within a day
    g = g.set_index("game_id").loc[[i for i in s["game_id"] if i in set(g["game_id"])]].reset_index()
    if history is not None:
        g = pd.concat([history, g], ignore_index=True)
    feats = elo_features_from_games(g, cfg)
    return feats.loc[games["game_id"].to_numpy(), [ELO_FEATURE]]


# =========================================================================== M8 context and early-season features
#
# Everything below is new in M8 (context/ml-m8-method.md, components 4 and 5) and is not used by
# `build_features` above, so A4s and C2d are unchanged. All of it comes from schedule facts that are
# published before the season (venue, kickoff time, rest days, surface) plus `data/sources/stadiums.csv`,
# except the weather columns (`temp`, `wind`), which nflverse fills in only after a game is played: they
# are built for a descriptive DEV test and `asof.corrupt_from` scrambles them, so `leakage_check` flags them.
#
# A team's base for a season is the venue of most of its non-neutral home games that season (ties: the
# venue of its first such game), so relocations and temporary homes (Saints 2005, Bears 2002) follow the
# schedule. Every feature is "home minus away" unless named otherwise:
#
#   travel_diff        (home km - away km) / 1000, great-circle distance from the team's base to the venue
#   tz_diff            |home time-zone shift| - |away shift|, hours, at kickoff (DST-aware)
#   body_early_diff    early(home) - early(away); early = kickoff before 11:00 on the team's base clock
#   body_late_diff     late(home) - late(away);  late = kickoff at or after 20:00 on the base clock
#   short_diff         (home rest <= 5 days) - (away rest <= 5)
#   bye_diff           (home rest >= 13) - (away rest >= 13)
#   alt_diff           exposed(home) - exposed(away); exposed = venue >= 1000 m and base below it
#   indoor             venue roof type is dome or retractable (static, from stadiums.csv)
#   turf               venue surface is artificial (schedule `surface`; grass and dessograss count as grass)
#   surface_mm_diff    mismatch(home) - mismatch(away); mismatch = venue surface type differs from the base's
#   dome_cold_diff     indoor-based team at an outdoor venue in week 13 or later, home minus away
#   wind15, cold32     NOT DEPLOYABLE (filled after the game): wind >= 15 mph, temperature <= 32 F, outdoor only
#
# Early season (component 4):
#
#   qb_new_diff        new(home) - new(away); new = the game's starter differs from the franchise's main
#                      starter last season (most REG starts in S-1); 0 for a franchise with no S-1 games
#   coach_new_diff     new(home) - new(away); new = the game's head coach differs from the franchise's head
#                      coach in its last game of season S-1
#   weeks_played       week - 1 (REG), for the "x weeks played" interactions built by the caller

from pathlib import Path  # noqa: E402

from ...teams import franchise as _franchise  # noqa: E402

STADIUMS_CSV = config.ROOT / "data" / "sources" / "stadiums.csv"
M8_CONTEXT = ["travel_diff", "tz_diff", "body_early_diff", "body_late_diff", "short_diff", "bye_diff", "alt_diff",
              "indoor", "turf", "surface_mm_diff", "dome_cold_diff"]
M8_WEATHER = ["wind15", "cold32"]
M8_EARLY = ["qb_new_diff", "coach_new_diff", "weeks_played"]
ALT_THRESHOLD_M = 1000.0
GRASS = frozenset({"grass", "dessograss"})
EARTH_KM = 6371.0


def load_stadiums(path: Path = STADIUMS_CSV) -> pd.DataFrame:
    s = pd.read_csv(path)
    if s["stadium_id"].duplicated().any():
        raise ValueError("duplicate stadium_id in stadiums.csv")
    return s.set_index("stadium_id")


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2, float) - np.asarray(lon1, float))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _surface_type(s: pd.Series) -> pd.Series:
    """'grass' or 'turf' from the schedule's surface string; missing stays missing."""
    t = s.astype("string").str.strip().str.lower()
    return t.where(t.isna() | (t == ""), np.where(t.isin(GRASS), "grass", "turf")).replace("", pd.NA)


def team_bases(sched: pd.DataFrame) -> pd.DataFrame:
    """(franchise, season) -> base stadium_id and the base's usual surface type (from its home games)."""
    s = sched[(sched["location"] != "Neutral") & sched["stadium_id"].notna()].copy()
    s = s.sort_values(["season", "gameday", "gametime"], kind="stable")
    s["fr"] = [_franchise(c, int(y)) for c, y in zip(s["home_team"], s["season"])]
    s["surf"] = _surface_type(s["surface"]) if "surface" in s.columns else pd.Series(pd.NA, index=s.index)
    rows = []
    for (fr, season), g in s.groupby(["fr", "season"], sort=True):
        counts = g["stadium_id"].value_counts()
        top = counts[counts == counts.max()].index
        base = next(x for x in g["stadium_id"] if x in set(top))  # tie: the earliest game's venue
        surf = g.loc[g["stadium_id"] == base, "surf"].dropna()
        rows.append({"fr": fr, "season": int(season), "base": base,
                     "base_surface": surf.mode().iloc[0] if len(surf) else pd.NA})
    return pd.DataFrame(rows).set_index(["fr", "season"])


def _base_for(bases: pd.DataFrame, fr: np.ndarray, season: np.ndarray) -> pd.DataFrame:
    """Base row for each (franchise, season); a season with no home game falls back to the latest earlier one."""
    out = bases.reindex(pd.MultiIndex.from_arrays([fr, season])).reset_index(drop=True)
    miss = out["base"].isna().to_numpy()
    for i in np.flatnonzero(miss):
        prev = bases.loc[fr[i]] if fr[i] in bases.index.get_level_values(0) else None
        if prev is not None:
            prev = prev[prev.index < season[i]]
            if len(prev):
                out.iloc[i] = prev.iloc[-1]
    return out


def _local_hours(kick: pd.Series, tz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kickoff clock hour (fractional) in each row's time zone, and the UTC offset (hours) of that zone."""
    hour = np.full(len(kick), np.nan)
    off = np.full(len(kick), np.nan)
    for z in pd.unique(tz):
        if not isinstance(z, str):
            continue
        m = tz == z
        loc = kick[m].dt.tz_convert(z)
        hour[m] = loc.dt.hour.to_numpy() + loc.dt.minute.to_numpy() / 60.0
        off[m] = np.array([t.utcoffset().total_seconds() / 3600.0 for t in loc])
    return hour, off


def m8_context_features(pbp: pd.DataFrame | None, sched: pd.DataFrame, games: pd.DataFrame | None = None,
                        stadiums: pd.DataFrame | None = None, weather: bool = False) -> pd.DataFrame:
    """M8 context features (M8_CONTEXT, plus M8_WEATHER when `weather`) for each game, indexed by game_id.

    Builder form (pbp, sched, games) for `asof.leakage_check`; `pbp` is unused.
    """
    stadiums = load_stadiums() if stadiums is None else stadiums
    if "kickoff" not in sched.columns:
        sched = asof_mod.add_asof(sched)
    games = sched if games is None else games
    if "kickoff" not in games.columns:
        games = games.merge(sched[["game_id", "kickoff"]], on="game_id", how="left")
    unknown = sorted(set(games["stadium_id"].dropna()) - set(stadiums.index))
    if unknown:
        raise ValueError(f"stadium_id not in stadiums.csv: {unknown}")
    bases = team_bases(sched)
    season = games["season"].astype(int).to_numpy()
    venue = stadiums.reindex(games["stadium_id"].to_numpy())
    kick = pd.Series(pd.to_datetime(games["kickoff"]).to_numpy(), index=range(len(games)))
    if kick.dt.tz is None:
        kick = kick.dt.tz_localize("UTC")
    has_time = (games["gametime"].astype("string").str.match(r"^\d{1,2}:\d{2}$", na=False).to_numpy()
                if "gametime" in games.columns else np.ones(len(games), bool))
    _, v_off = _local_hours(kick, venue["tz"].to_numpy(object))
    v_indoor = venue["roof_type"].isin(["dome", "retractable"]).to_numpy()
    v_surf = (_surface_type(games["surface"]).to_numpy(object) if "surface" in games.columns
              else np.full(len(games), pd.NA, dtype=object))
    side = {}
    for sd in ("home", "away"):
        fr = np.array([_franchise(c, int(y)) for c, y in zip(games[f"{sd}_team"], season)], dtype=object)
        b = _base_for(bases, fr, season)
        bst = stadiums.reindex(b["base"].to_numpy())
        km = _haversine_km(bst["lat"].to_numpy(float), bst["lon"].to_numpy(float),
                           venue["lat"].to_numpy(float), venue["lon"].to_numpy(float))
        hour, b_off = _local_hours(kick, bst["tz"].to_numpy(object))
        shift = v_off - b_off
        b_indoor = bst["roof_type"].isin(["dome", "retractable"]).to_numpy()
        b_surf = b["base_surface"].to_numpy(object)
        mm = np.array([isinstance(x, str) and isinstance(y, str) and x != y for x, y in zip(v_surf, b_surf)])
        rest = (pd.to_numeric(games[f"{sd}_rest"], errors="coerce").to_numpy(float)
                if f"{sd}_rest" in games.columns else np.full(len(games), np.nan))
        side[sd] = {
            "km": np.nan_to_num(km, nan=0.0), "tz": np.nan_to_num(np.abs(shift), nan=0.0),
            "early": (has_time & (hour < 11.0)).astype(float), "late": (has_time & (hour >= 20.0)).astype(float),
            "short": (rest <= 5).astype(float), "bye": (rest >= 13).astype(float),
            "alt": ((venue["elevation_m"].to_numpy(float) >= ALT_THRESHOLD_M)
                    & (bst["elevation_m"].to_numpy(float) < ALT_THRESHOLD_M)).astype(float),
            "mm": mm.astype(float),
            "dome_cold": (b_indoor & ~v_indoor & (games["week"].to_numpy(int) >= 13)).astype(float),
        }
    h, a = side["home"], side["away"]
    out = pd.DataFrame({
        "travel_diff": (h["km"] - a["km"]) / 1000.0, "tz_diff": h["tz"] - a["tz"],
        "body_early_diff": h["early"] - a["early"], "body_late_diff": h["late"] - a["late"],
        "short_diff": h["short"] - a["short"], "bye_diff": h["bye"] - a["bye"], "alt_diff": h["alt"] - a["alt"],
        "indoor": v_indoor.astype(float),
        "turf": np.array([isinstance(x, str) and x == "turf" for x in v_surf], dtype=float),
        "surface_mm_diff": h["mm"] - a["mm"], "dome_cold_diff": h["dome_cold"] - a["dome_cold"],
    }, index=games["game_id"].to_numpy())
    if weather:
        outdoor = ~v_indoor
        wind = pd.to_numeric(games["wind"], errors="coerce").to_numpy(float) if "wind" in games else np.nan
        temp = pd.to_numeric(games["temp"], errors="coerce").to_numpy(float) if "temp" in games else np.nan
        out["wind15"] = (outdoor & (np.nan_to_num(wind, nan=0.0) >= 15)).astype(float)
        out["cold32"] = (outdoor & (np.nan_to_num(temp, nan=60.0) <= 32)).astype(float)
    out.index.name = "game_id"
    return out


def _last_season_facts(sched: pd.DataFrame) -> tuple[dict, dict]:
    """(franchise, season) -> main starter (most REG starts) and head coach in the franchise's last game."""
    s = sched[sched["game_type"].isin(asof_mod.REG_POST_TYPES)].copy()
    if "kickoff" not in s.columns:
        s = asof_mod.add_asof(s)
    s = s.sort_values("kickoff", kind="stable")
    parts = []
    for sd in ("home", "away"):
        parts.append(pd.DataFrame({
            "fr": [_franchise(c, int(y)) for c, y in zip(s[f"{sd}_team"], s["season"])],
            "season": s["season"].astype(int).to_numpy(), "reg": (s["game_type"] == "REG").to_numpy(),
            "qb": s[f"{sd}_qb_id"].to_numpy(object) if f"{sd}_qb_id" in s else None,
            "coach": s[f"{sd}_coach"].to_numpy(object) if f"{sd}_coach" in s else None,
            "kick": s["kickoff"].to_numpy()}))
    long = pd.concat(parts, ignore_index=True).sort_values("kick", kind="stable")
    main_qb, last_coach = {}, {}
    for (fr, S), g in long.groupby(["fr", "season"]):
        q = g.loc[g["reg"] & g["qb"].notna(), "qb"]
        if len(q):
            c = q.value_counts()
            top = set(c[c == c.max()].index)
            main_qb[(fr, int(S))] = next(x for x in q[::-1] if x in top)  # tie: the later starter
        cc = g["coach"].dropna()
        if len(cc):
            last_coach[(fr, int(S))] = cc.iloc[-1]
    return main_qb, last_coach


def m8_early_features(pbp: pd.DataFrame | None, sched: pd.DataFrame, games: pd.DataFrame | None = None
                      ) -> pd.DataFrame:
    """qb_new_diff, coach_new_diff and weeks_played for each game, indexed by game_id (builder form).

    Uses only season S-1 facts plus this game's own starter and head coach (pre-game facts: the
    starter as in M3-D1, the coach from the schedule).
    """
    games = sched if games is None else games
    main_qb, last_coach = _last_season_facts(sched)  # only (franchise, S-1) entries are read for season S
    season = games["season"].astype(int).to_numpy()
    out = {}
    for sd in ("home", "away"):
        fr = [_franchise(c, int(y)) for c, y in zip(games[f"{sd}_team"], season)]
        qb = games[f"{sd}_qb_id"].to_numpy(object) if f"{sd}_qb_id" in games else np.full(len(games), None)
        co = games[f"{sd}_coach"].to_numpy(object) if f"{sd}_coach" in games else np.full(len(games), None)
        nq, nc = np.zeros(len(games)), np.zeros(len(games))
        for i, (f, S) in enumerate(zip(fr, season)):
            m = main_qb.get((f, int(S) - 1))
            if m is not None and isinstance(qb[i], str):
                nq[i] = float(qb[i] != m)
            c = last_coach.get((f, int(S) - 1))
            if c is not None and isinstance(co[i], str):
                nc[i] = float(co[i] != c)
        out[sd] = (nq, nc)
    res = pd.DataFrame({"qb_new_diff": out["home"][0] - out["away"][0],
                        "coach_new_diff": out["home"][1] - out["away"][1],
                        "weeks_played": np.clip(games["week"].to_numpy(int) - 1, 0, None).astype(float)},
                       index=games["game_id"].to_numpy())
    res.index.name = "game_id"
    return res
