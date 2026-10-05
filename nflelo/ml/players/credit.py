"""Per-player, per-game credited EPA by position (context/ml-m5-method.md, section 3).

Output of `player_games`: one row per (game, team, player) with

    group, side          position group (players.positions) and off/def
    inv                  involvement count in the game (usage: carries, targets, credited plays)
    share                estimated share of the team's plays he was on the field for (usage-based,
                         decision M5-D1 (a): no snap counts, no participation)
    team_plays           the team's clean scrimmage plays on his side of the ball
    c_<component>        credited EPA in the game for each component, per TEAM play
    a_<component>        1 if that component is recorded in the game's season (era gaps), else 0

Credit by group (all EPA from the offense's point of view is flipped for defenders,
so a positive credit is always good for the player's team):

    QB      db      EPA on his dropbacks (passes, sacks, scrambles)
    RB      rush    EPA per carry minus the team's OTHER backs' EPA per carry in the same game
            rb_recv EPA per target minus the other backs' EPA per target in the same game
            (the other backs' average is shrunk toward the league RB average of earlier seasons)
    WR, TE  recv    EPA per target minus his QB's average EPA per target to OTHER receivers,
                    from the QB's earlier games only (shrunk toward the league average of earlier seasons)
    DL, LB  sack, qbhit, tfl, ff   -EPA of plays where he is credited with a sack (halves split),
                    a QB hit (not a sack), a tackle for loss (not a sack), or a forced fumble
    CB, S   int, pd -EPA of plays where he is credited with an interception or a pass defensed
                    (pass-defensed credit only on non-interception plays)
    K       fg      3 * (made - expected) per attempt, expected = league make rate by distance band
                    in the three earlier seasons
    OL      none    (values come only from the with/without layer in value.py)

Co-credited plays split the credit equally. Plays are the M1/M3 clean plays
(`team_efficiency.clean_plays`: real pass and run snaps with EPA, no two-point
tries, garbage time removed), and betting-market columns are dropped first.
League baselines always come from EARLIER seasons (the first data season,
1999, uses itself), so a game's credit uses nothing from later games.

Era gaps (`component_availability`): a component counts only in seasons where
its stat is recorded, detected from the data (fill rate of the credited ID and
event rate per game). In nflverse, incomplete-pass targets are missing in
2003-2008 (so receiving credit is unavailable), QB hits in 2003-2005, and
tackles for loss in 2003-2007.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import team_efficiency as te
from ..features.opponent_adjust import week_ordinals
from . import positions as pos

COMPONENTS = {
    "QB": ("db",), "RB": ("rush", "rb_recv"), "WR": ("recv",), "TE": ("recv",), "OL": (),
    "DL": ("sack", "qbhit", "tfl", "ff"), "LB": ("sack", "qbhit", "tfl", "ff"),
    "CB": ("int", "pd"), "S": ("int", "pd"), "K": ("fg",),
}
ALL_COMPONENTS = ("db", "rush", "rb_recv", "recv", "sack", "qbhit", "tfl", "ff", "int", "pd", "fg")
DEF_COLS = {
    "sack": ["sack_player_id", "half_sack_1_player_id", "half_sack_2_player_id"],
    "qbhit": ["qb_hit_1_player_id", "qb_hit_2_player_id"],
    "tfl": ["tackle_for_loss_1_player_id", "tackle_for_loss_2_player_id"],
    "ff": ["forced_fumble_player_1_player_id", "forced_fumble_player_2_player_id"],
    "int": ["interception_player_id"],
    "pd": ["pass_defense_1_player_id", "pass_defense_2_player_id"],
}
TACKLE_COLS = ["solo_tackle_1_player_id", "solo_tackle_2_player_id", "assist_tackle_1_player_id",
               "assist_tackle_2_player_id", "assist_tackle_3_player_id", "assist_tackle_4_player_id",
               "tackle_with_assist_1_player_id", "tackle_with_assist_2_player_id"]
PBP_COLUMNS = (["game_id", "season", "week", "posteam", "defteam", "play_type", "pass", "rush", "epa", "success",
                "wp", "two_point_attempt", "qb_dropback", "qb_scramble", "pass_attempt", "rush_attempt", "sack",
                "qb_hit", "tackled_for_loss", "fumble_forced", "interception", "incomplete_pass", "passer_id",
                "passer_player_id", "rusher_player_id", "receiver_player_id", "forced_fumble_player_1_team",
                "forced_fumble_player_2_team", "field_goal_attempt", "extra_point_attempt", "field_goal_result",
                "kick_distance", "kicker_player_id", "play_id"]
               + [c for cols in DEF_COLS.values() for c in cols] + TACKLE_COLS)
FG_BANDS = (0, 30, 40, 50, 100)
BASE_K_QB = 100.0    # pseudo-targets of league average in a QB's per-target baseline
BASE_K_RB = 10.0     # pseudo-carries (and pseudo-targets) in the other-backs baseline
LEAGUE_LOOKBACK = 3  # seasons of league averages behind each baseline


# --------------------------------------------------------------------------- availability (era gaps)

def component_availability(pbp: pd.DataFrame) -> pd.DataFrame:
    """Season x component: 1 if the component's stat is recorded that season, else 0 (detected from the data)."""
    p = pbp
    rows = []
    for S, d in p.groupby("season"):
        g = max(d["game_id"].nunique(), 1)

        def fill(mask, col):
            m = mask.fillna(False).astype(bool) if hasattr(mask, "fillna") else mask
            return float(d.loc[m, col].notna().mean()) if m.any() and col in d.columns else 0.0

        inc = d["incomplete_pass"] == 1 if "incomplete_pass" in d.columns else pd.Series(False, index=d.index)
        hit = (d["qb_hit"] == 1) & (d["sack"] != 1)
        tfl = (d["tackled_for_loss"] == 1) & (d["sack"] != 1)
        sack_fill = float(((d["sack"] == 1) & (d["sack_player_id"].notna() | d["half_sack_1_player_id"].notna()))
                          .sum() / max((d["sack"] == 1).sum(), 1))
        recv = fill(inc, "receiver_player_id") >= 0.5
        rows.append({"season": int(S), "db": 1, "rush": 1, "fg": 1,
                     "recv": int(recv), "rb_recv": int(recv),
                     "sack": int(sack_fill >= 0.5),
                     "qbhit": int(hit.sum() / g >= 1.0 and fill(hit, "qb_hit_1_player_id") >= 0.5),
                     "tfl": int(tfl.sum() / g >= 0.5 and fill(tfl, "tackle_for_loss_1_player_id") >= 0.5),
                     "ff": int(fill(d["fumble_forced"] == 1, "forced_fumble_player_1_player_id") >= 0.5),
                     "int": int(fill(d["interception"] == 1, "interception_player_id") >= 0.5),
                     "pd": int(d["pass_defense_1_player_id"].notna().sum() / g >= 2.0)})
    return pd.DataFrame(rows).set_index("season")[list(ALL_COMPONENTS)]


# --------------------------------------------------------------------------- positions

def position_table(depth: pd.DataFrame | None, rosters: pd.DataFrame | None,
                   players: pd.DataFrame | None) -> pd.DataFrame:
    """(season, gsis_id) -> group, plus a season-free fallback (season = -1).

    Priority: the player's most common group on that season's depth charts
    (offense/defense slots, and K), then his most common weekly-roster
    position that season, then the players table. A generic "DB" becomes CB
    unless a depth chart says S.
    """
    parts = []
    if depth is not None and len(depth):
        d = depth[depth["group"].notna()]
        m = d.groupby(["season", "gsis_id"])["group"].agg(lambda s: s.value_counts().index[0]).reset_index()
        parts.append(m.assign(pri=0))
    if rosters is not None and len(rosters):
        r = rosters[["season", "gsis_id", "position"]].copy()
        r["group"] = [pos.group_of(x) for x in r["position"]]
        r = r[r["group"].notna()]
        m = r.groupby(["season", "gsis_id"])["group"].agg(lambda s: s.value_counts().index[0]).reset_index()
        parts.append(m.assign(pri=1))
    if players is not None and len(players):
        p = players[["gsis_id", "position"]].dropna().copy()
        p["group"] = [pos.group_of(x) for x in p["position"]]
        parts.append(p[p["group"].notna()][["gsis_id", "group"]].assign(season=-1, pri=2))
    if not parts:
        return pd.DataFrame(columns=["season", "gsis_id", "group"])
    out = pd.concat(parts, ignore_index=True).sort_values("pri")
    out = out.drop_duplicates(["season", "gsis_id"], keep="first")
    return out[["season", "gsis_id", "group"]].reset_index(drop=True)


def lookup_groups(tab: pd.DataFrame, seasons: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Group for each (season, id): that season's entry, else the nearest earlier season's, else the fallback."""
    by_season = tab[tab["season"] >= 0]
    exact = by_season.set_index(["season", "gsis_id"])["group"]
    key = pd.MultiIndex.from_arrays([seasons.astype(int), ids.astype(object)])
    out = exact.reindex(key).to_numpy(object)
    miss = pd.isna(out)
    if miss.any():
        # nearest season with an entry (prefer earlier), then the players-table fallback
        latest = by_season.sort_values("season").groupby("gsis_id")["group"].last()
        out[miss] = pd.Series(ids[miss]).map(latest).to_numpy(object)
        miss = pd.isna(out)
        fb = tab[tab["season"] == -1].set_index("gsis_id")["group"]
        out[miss] = pd.Series(ids[miss]).map(fb).to_numpy(object)
    return out


# --------------------------------------------------------------------------- helpers

def _league_mean(values: pd.Series, seasons: pd.Series, lookback: int = LEAGUE_LOOKBACK) -> dict[int, float]:
    """Season S -> mean of `values` over seasons S-lookback..S-1 (the first season uses itself)."""
    by = pd.DataFrame({"v": values.to_numpy(float), "s": seasons.to_numpy(int)}).groupby("s")["v"].agg(["sum", "count"])
    out = {}
    for S in by.index:
        prior = by[(by.index < S) & (by.index >= S - lookback)]
        src = prior if prior["count"].sum() > 0 else by.loc[[S]]
        out[int(S)] = float(src["sum"].sum() / src["count"].sum())
    return out


def _explode_credit(plays: pd.DataFrame, cols: list[str], value: np.ndarray, weights: dict | None = None
                    ) -> pd.DataFrame:
    """Long rows (row index, player, weight) for every non-null ID in `cols`; weight splits the play equally."""
    ids = plays[cols].to_numpy(object)
    present = ~pd.isna(ids)
    if weights:
        w = np.column_stack([np.where(present[:, j], weights.get(c, 1.0), 0.0) for j, c in enumerate(cols)])
        tot = w.sum(axis=1, keepdims=True)
        w = np.divide(w, tot, out=np.zeros_like(w), where=tot > 0)
    else:
        n = present.sum(axis=1, keepdims=True)
        w = np.divide(present.astype(float), n, out=np.zeros(present.shape), where=n > 0)
    r, c = np.nonzero(present)
    return pd.DataFrame({"row": r, "player": ids[r, c].astype(str), "w": w[r, c], "val": value[r] * w[r, c]})


# --------------------------------------------------------------------------- slots and shares

def team_week_slots(depth: pd.DataFrame | None) -> pd.DataFrame:
    """(season, week, team, group) -> number of rank-1 depth-chart slots (fullbacks not counted as RB slots)."""
    if depth is None or depth.empty:
        return pd.DataFrame(columns=["season", "week", "team", "group", "slots"])
    d = depth[(depth["rank"] == 1) & depth["group"].notna() & depth["side"].isin(["off", "def"])]
    d = d[~((d["group"] == "RB") & d["slot"].isin(pos.FULLBACK_CODES))]
    d = d.drop_duplicates(["season", "week", "team", "gsis_id"])
    return d.groupby(["season", "week", "team", "group"]).size().rename("slots").reset_index()


def waterfill(inv: np.ndarray, slots: float) -> np.ndarray:
    """Shares proportional to involvement, capped at 1, scaled so they sum to `slots` when possible."""
    inv = np.asarray(inv, float)
    tot = inv.sum()
    if tot <= 0 or slots <= 0:
        return np.zeros_like(inv)
    if (inv > 0).sum() <= slots:
        return (inv > 0).astype(float)
    share = np.zeros_like(inv)
    free = inv > 0
    remaining = float(slots)
    for _ in range(len(inv)):
        c = remaining / inv[free].sum()
        s = c * inv
        over = free & (s >= 1)
        if not over.any():
            share[free] = s[free]
            break
        share[over] = 1.0
        remaining -= over.sum()
        free &= ~over
        if remaining <= 0 or not free.any():
            break
    return share


def _shares(pg: pd.DataFrame, slots: pd.DataFrame) -> np.ndarray:
    """Usage-based on-field share per player-game (water-filled within team-game-group)."""
    sl = slots.set_index(["season", "week", "team", "group"])["slots"]
    key = pd.MultiIndex.from_frame(pg[["season", "week", "team", "group"]])
    n_slots = sl.reindex(key).to_numpy(float)
    default = pg["group"].map(pos.DEFAULT_SLOTS).to_numpy(float)
    n_slots = np.where(np.isnan(n_slots), default, n_slots)
    n_slots = np.where(pg["group"].isin(["QB", "K"]).to_numpy(), 1.0, n_slots)
    out = np.zeros(len(pg))
    gk = pg.groupby(["game_id", "team", "group"], sort=False).indices
    inv = pg["inv"].to_numpy(float)
    for _, idx in gk.items():
        out[idx] = waterfill(inv[idx], n_slots[idx[0]])
    return out


# --------------------------------------------------------------------------- main builder

def player_games(pbp: pd.DataFrame, sched: pd.DataFrame, positions: pd.DataFrame,
                 depth: pd.DataFrame | None = None, availability: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (game, team, player) with usage, estimated share, and credited EPA by component.

    `sched` needs kickoff (asof.add_asof) for every game in `pbp`; `positions`
    is `position_table(...)`; `depth` (normalized, from data.load_depth_charts)
    gives the slot counts behind the shares (defaults are used without it).
    Players whose group is unknown are dropped; OL rows appear only through
    lineup.py (they have no individual stats).
    """
    if "kickoff" not in sched.columns:
        raise ValueError("sched needs kickoff (asof.add_asof)")
    avail = availability if availability is not None else component_availability(pbp)
    info = sched.drop_duplicates("game_id").set_index("game_id")
    plays = te.clean_plays(pbp, te.EfficiencyConfig()).copy()
    plays = plays[plays["game_id"].isin(info.index)]
    plays["season"] = plays["game_id"].map(info["season"]).astype(int)
    plays["week"] = plays["game_id"].map(info["week"]).astype(int)
    plays["kick_ns"] = te._utc_ns(plays["game_id"].map(info["kickoff"]))
    plays["off"] = pos.map_teams(plays["posteam"], plays["season"]).to_numpy(object)
    plays["def"] = pos.map_teams(plays["defteam"], plays["season"]).to_numpy(object)
    plays = plays.reset_index(drop=True)
    epa = plays["epa"].to_numpy(float)
    f = lambda c: (plays[c].fillna(0).to_numpy(float) == 1) if c in plays.columns else np.zeros(len(plays), bool)  # noqa: E731

    n_off = plays.groupby(["game_id", "off"]).size()
    n_def = plays.groupby(["game_id", "def"]).size()

    recs = []  # (row, player, side, component, credit, inv)

    # ---- QB dropbacks
    qb_col = "passer_id" if "passer_id" in plays.columns else "passer_player_id"
    db = f("qb_dropback") & plays[qb_col].notna().to_numpy()
    recs.append(pd.DataFrame({"row": np.flatnonzero(db), "player": plays.loc[db, qb_col].astype(str).to_numpy(),
                              "side": "off", "comp": "db", "val": epa[db], "inv": 1.0}))

    # ---- carries (designed runs; scrambles belong to the QB's dropbacks)
    carry = (f("rush_attempt") & ~f("qb_scramble") & (plays["play_type"] == "run").to_numpy()
             & plays["rusher_player_id"].notna().to_numpy())
    # ---- targets (pass attempts that are not sacks, with a receiver)
    target = f("pass_attempt") & ~f("sack") & plays["receiver_player_id"].notna().to_numpy()

    groups_of = lambda ids, seas: lookup_groups(positions, np.asarray(seas), np.asarray(ids, object))  # noqa: E731
    c_rows = np.flatnonzero(carry)
    c_tab = pd.DataFrame({"row": c_rows, "player": plays.loc[carry, "rusher_player_id"].astype(str).to_numpy(),
                          "game_id": plays.loc[carry, "game_id"].to_numpy(), "team": plays.loc[carry, "off"].to_numpy(),
                          "season": plays.loc[carry, "season"].to_numpy(), "epa": epa[carry]})
    c_tab["group"] = groups_of(c_tab["player"], c_tab["season"])
    t_rows = np.flatnonzero(target)
    t_tab = pd.DataFrame({"row": t_rows, "player": plays.loc[target, "receiver_player_id"].astype(str).to_numpy(),
                          "qb": plays.loc[target, "passer_player_id"].astype("string").to_numpy(object),
                          "game_id": plays.loc[target, "game_id"].to_numpy(), "team": plays.loc[target, "off"].to_numpy(),
                          "season": plays.loc[target, "season"].to_numpy(), "kick_ns": plays.loc[target, "kick_ns"].to_numpy(),
                          "epa": epa[target]})
    t_tab["group"] = groups_of(t_tab["player"], t_tab["season"])

    # usage for skill players: every carry and target (any group; QB carries count toward QB usage)
    recs.append(pd.DataFrame({"row": c_tab["row"], "player": c_tab["player"], "side": "off", "comp": "_use",
                              "val": 0.0, "inv": 1.0}))
    recs.append(pd.DataFrame({"row": t_tab["row"], "player": t_tab["player"], "side": "off", "comp": "_use",
                              "val": 0.0, "inv": 1.0}))

    # ---- RB rushing and receiving: relative to the team's other backs in the same game
    def other_backs(tab: pd.DataFrame, comp: str) -> pd.DataFrame:
        rb = tab[tab["group"] == "RB"]
        if rb.empty:
            return rb.assign(val=[], comp=comp)
        league = _league_mean(rb["epa"], rb["season"])
        gp = rb.groupby(["game_id", "team", "player"]).agg(n=("epa", "size"), s=("epa", "sum")).reset_index()
        tot = gp.groupby(["game_id", "team"])[["n", "s"]].transform("sum")
        L = rb.drop_duplicates("game_id").set_index("game_id")["season"].map(league)
        lg = gp["game_id"].map(L).to_numpy(float)
        base = (tot["s"] - gp["s"] + BASE_K_RB * lg) / (tot["n"] - gp["n"] + BASE_K_RB)
        gp["base"] = base.to_numpy(float)
        r = rb.merge(gp[["game_id", "team", "player", "base"]], on=["game_id", "team", "player"])
        return pd.DataFrame({"row": r["row"], "player": r["player"], "side": "off", "comp": comp,
                             "val": r["epa"] - r["base"], "inv": 0.0})
    recs.append(other_backs(c_tab, "rush"))
    recs.append(other_backs(t_tab, "rb_recv"))

    # ---- WR/TE receiving: relative to his QB's average per target to OTHER receivers, from earlier games
    wt = t_tab
    league = _league_mean(wt["epa"], wt["season"])
    g_qr = wt[wt["qb"].notna()].groupby(["qb", "player", "game_id"]).agg(
        n=("epa", "size"), s=("epa", "sum"), kick_ns=("kick_ns", "first")).reset_index()
    g_q = g_qr.groupby(["qb", "game_id"]).agg(n=("n", "sum"), s=("s", "sum"), kick_ns=("kick_ns", "first")).reset_index()
    g_q = g_q.sort_values(["qb", "kick_ns"], kind="stable")
    g_q["N_prior"] = g_q.groupby("qb")["n"].cumsum() - g_q["n"]
    g_q["S_prior"] = g_q.groupby("qb")["s"].cumsum() - g_q["s"]
    g_qr = g_qr.sort_values(["qb", "player", "kick_ns"], kind="stable")
    g_qr["n_prior"] = g_qr.groupby(["qb", "player"])["n"].cumsum() - g_qr["n"]
    g_qr["s_prior"] = g_qr.groupby(["qb", "player"])["s"].cumsum() - g_qr["s"]
    g_qr = g_qr.merge(g_q[["qb", "game_id", "N_prior", "S_prior"]], on=["qb", "game_id"])
    seas_of = wt.drop_duplicates("game_id").set_index("game_id")["season"]
    lg = g_qr["game_id"].map(seas_of).map(league).to_numpy(float)
    g_qr["base"] = ((g_qr["S_prior"] - g_qr["s_prior"] + BASE_K_QB * lg)
                    / (g_qr["N_prior"] - g_qr["n_prior"] + BASE_K_QB))
    w = wt[wt["group"].isin(["WR", "TE"])].merge(g_qr[["qb", "player", "game_id", "base"]],
                                                  on=["qb", "player", "game_id"], how="left")
    no_qb = w["base"].isna()
    w.loc[no_qb, "base"] = w.loc[no_qb, "season"].map(league).to_numpy(float)
    recs.append(pd.DataFrame({"row": w["row"], "player": w["player"], "side": "off", "comp": "recv",
                              "val": w["epa"] - w["base"], "inv": 0.0}))

    # ---- defense: -EPA of credited plays
    neg = -epa
    sack = f("sack")
    masks = {"sack": sack, "qbhit": f("qb_hit") & ~sack, "tfl": f("tackled_for_loss") & ~sack,
             "ff": f("fumble_forced"), "int": f("interception"), "pd": ~f("interception")}
    weights = {"sack": {"sack_player_id": 1.0, "half_sack_1_player_id": 0.5, "half_sack_2_player_id": 0.5}}
    for comp, cols in DEF_COLS.items():
        cols = [c for c in cols if c in plays.columns]
        m = masks[comp]
        sub = plays.loc[m, cols].copy()
        if comp == "ff":  # only forced fumbles by the defense
            for i, c in enumerate(cols, start=1):
                tcol = f"forced_fumble_player_{i}_team"
                if tcol in plays.columns:
                    own = pos.map_teams(plays.loc[m, tcol], plays.loc[m, "season"]).to_numpy(object)
                    sub.loc[own != plays.loc[m, "def"].to_numpy(object), c] = None
        ex = _explode_credit(sub, cols, neg[m], weights.get(comp))
        ex["row"] = np.flatnonzero(m)[ex["row"].to_numpy()]
        recs.append(ex.assign(side="def", comp=comp, inv=0.0)[["row", "player", "side", "comp", "val", "inv"]])
    # defensive involvement: distinct plays where the player is credited with anything (tackles included)
    inv_cols = [c for c in TACKLE_COLS + [c for cols in DEF_COLS.values() for c in cols] if c in plays.columns]
    ids = plays[inv_cols].to_numpy(object)
    r, c = np.nonzero(~pd.isna(ids))
    dinv = pd.DataFrame({"row": r, "player": ids[r, c].astype(str)}).drop_duplicates()
    recs.append(dinv.assign(side="def", comp="_use", val=0.0, inv=1.0))

    long = pd.concat(recs, ignore_index=True)
    long["game_id"] = plays["game_id"].to_numpy()[long["row"].to_numpy()]
    long["team"] = np.where(long["side"] == "off", plays["off"].to_numpy(object)[long["row"].to_numpy()],
                            plays["def"].to_numpy(object)[long["row"].to_numpy()])
    long["season"] = plays["season"].to_numpy()[long["row"].to_numpy()]
    long["group"] = groups_of(long["player"], long["season"])
    long = long[long["group"].notna()]
    # A component counts only for the groups it belongs to (spec table); usage counts for the player's own side.
    ok = [(cmp == "_use" and pos.SIDE.get(g) == sd) or cmp in COMPONENTS.get(g, ())
          for cmp, g, sd in zip(long["comp"], long["group"], long["side"])]
    long = long[np.asarray(ok, bool)]

    # ---- kickers: field goals over the expected make rate by distance band
    k = kicker_rows(pbp, sched, info)
    pg = long.pivot_table(index=["game_id", "team", "player", "group"], columns="comp", values="val",
                          aggfunc="sum", fill_value=0.0)
    inv = long.groupby(["game_id", "team", "player", "group"])["inv"].sum()
    pg = pg.join(inv).reset_index()
    if len(k):
        pg = pd.concat([pg, k], ignore_index=True).fillna({"fg": 0.0, "inv": 0.0})
        pg = pg.groupby(["game_id", "team", "player", "group"], as_index=False).sum(numeric_only=True)
    for c in ALL_COMPONENTS:
        if c not in pg.columns:
            pg[c] = 0.0
        pg[c] = pg[c].fillna(0.0)
    pg["inv"] = pg["inv"].fillna(0.0)
    pg = pg.rename(columns={"player": "gsis_id"})
    pg["season"] = pg["game_id"].map(info["season"]).astype(int)
    pg["week"] = pg["game_id"].map(info["week"]).astype(int)
    pg["game_type"] = pg["game_id"].map(info["game_type"]) if "game_type" in info.columns else "REG"
    pg["kick_ns"] = te._utc_ns(pg["game_id"].map(info["kickoff"]))
    pg["side"] = pg["group"].map(pos.SIDE)
    key_off = pd.MultiIndex.from_frame(pg[["game_id", "team"]])
    tp_off = n_off.reindex(key_off).to_numpy(float)
    tp_def = n_def.reindex(key_off).to_numpy(float)
    pg["team_plays"] = np.where(pg["side"] == "def", tp_def, tp_off)
    # Linemen have no individual usage; their on-field rows come from the pregame lineup (lineup.py).
    pg = pg[(pg["team_plays"] > 0) & (pg["group"] != "OL")].reset_index(drop=True)
    for c in ALL_COMPONENTS:
        pg[f"c_{c}"] = pg.pop(c) / pg["team_plays"]
    # Era gaps: component recorded in this season?
    av = avail.reindex(pg["season"].to_numpy())
    for c in ALL_COMPONENTS:
        pg[f"a_{c}"] = av[c].fillna(1).to_numpy(float)
    slots = team_week_slots(depth)
    pg["share"] = _shares(pg, slots)
    ords = week_ordinals(sched)
    pg["ord"] = ords.reindex(pd.MultiIndex.from_frame(pg[["season", "week"]])).to_numpy()
    cols = (["game_id", "season", "week", "game_type", "kick_ns", "ord", "team", "gsis_id", "group", "side",
             "inv", "share", "team_plays"] + [f"c_{c}" for c in ALL_COMPONENTS] + [f"a_{c}" for c in ALL_COMPONENTS])
    return pg[cols].sort_values(["kick_ns", "game_id", "team", "gsis_id"], kind="stable").reset_index(drop=True)


def kicker_rows(pbp: pd.DataFrame, sched: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """Kickers: one row per (game, team, kicker) with fg credit (points over expected; per team play later)."""
    need = ["game_id", "posteam", "field_goal_attempt", "kicker_player_id", "kick_distance", "field_goal_result"]
    if any(c not in pbp.columns for c in need):
        return pd.DataFrame()
    fga = pbp["field_goal_attempt"].fillna(0) == 1
    pat = pbp["extra_point_attempt"].fillna(0) == 1 if "extra_point_attempt" in pbp.columns else fga & False
    k = pbp[(fga | pat) & pbp["kicker_player_id"].notna() & pbp["game_id"].isin(info.index)][need].copy()
    k["is_fg"] = fga[k.index].to_numpy()
    k["season"] = k["game_id"].map(info["season"]).astype(int)
    k["made"] = (k["field_goal_result"] == "made").astype(float)
    k["band"] = np.digitize(k["kick_distance"].fillna(40).to_numpy(float), FG_BANDS[1:-1])
    exp = np.zeros(len(k))
    for b in range(len(FG_BANDS) - 1):
        m = ((k["band"] == b) & k["is_fg"]).to_numpy()
        if m.any():
            rate = _league_mean(k.loc[m, "made"], k.loc[m, "season"])
            exp[m] = k.loc[m, "season"].map(rate).to_numpy(float)
    k["val"] = np.where(k["is_fg"], 3.0 * (k["made"] - exp), 0.0)  # PATs count as usage only
    k["team"] = pos.map_teams(k["posteam"], k["season"]).to_numpy(object)
    out = k.groupby(["game_id", "team", "kicker_player_id"]).agg(fg=("val", "sum"), inv=("val", "size")).reset_index()
    out = out.rename(columns={"kicker_player_id": "player"})
    out["group"] = "K"
    return out
