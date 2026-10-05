"""M5 player values and lineups: leakage checks, credit sums, lineup rules, loaders. Offline, synthetic data.

Synthetic world: the conftest schedule (6 teams, 2022-2023, six REG weeks and
a playoff game) plus a roster per team (QB, two RBs, three WRs, a TE, five
OL, four DL, three LB, two CB, two S, a K) and play-by-play with every credited
player ID nflverse uses. KC's WR1 is ruled Out in 2023 week 4 (and has no plays
that week), and BUF's LB1 is on reserve (RES) from 2023 week 3.
"""
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nflelo.ml import asof, registry
from nflelo.ml import data as D
from nflelo.ml.eval import windows
from nflelo.ml.features import roster as rf
from nflelo.ml.players import credit as cr, lineup as lu, positions as pos, value as pv

ROOT = Path(__file__).resolve().parents[2]
SLOTS = {"QB": ["QB1"], "RB": ["RB1", "RB2"], "WR": ["WR1", "WR2", "WR3"], "TE": ["TE1"],
         "OL": ["LT", "LG", "C", "RG", "RT"], "DL": ["DE1", "DT1", "DT2", "DE2"], "LB": ["LB1", "LB2", "LB3"],
         "CB": ["CB1", "CB2"], "S": ["FS", "SS"], "K": ["K1"]}
SLOT_CODE = {"QB1": "QB", "RB1": "RB", "RB2": "RB", "WR1": "WR", "WR2": "WR", "WR3": "WR", "TE1": "TE",
             "LT": "LT", "LG": "LG", "C": "C", "RG": "RG", "RT": "RT", "DE1": "LDE", "DT1": "LDT", "DT2": "RDT",
             "DE2": "RDE", "LB1": "MLB", "LB2": "WLB", "LB3": "SLB", "CB1": "LCB", "CB2": "RCB", "FS": "FS",
             "SS": "SS", "K1": "K"}
VCFG = pv.ValueConfig(half_life=8.0, lam=500.0)


def pid(team, slot):
    return f"{team}_{slot}"


def roster_of(team):
    return {s: pid(team, s) for g in SLOTS.values() for s in g}


def add_players(pbp: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Credited player IDs and play-event flags on the conftest play-by-play, plus field-goal rows."""
    rng = np.random.default_rng(seed)
    p = pbp.copy()
    n = len(p)
    off, de = p["posteam"].to_numpy(), p["defteam"].to_numpy()
    pt = p["play_type"].to_numpy()
    is_pass, is_run = pt == "pass", pt == "run"
    p["passer_player_id"] = p["passer_id"]
    p["sack"] = (is_pass & (rng.uniform(size=n) < 0.06)).astype(float)
    p["qb_hit"] = (is_pass & ((p["sack"] == 1) | (rng.uniform(size=n) < 0.08))).astype(float)
    p["pass_attempt"] = is_pass.astype(float)
    p["rush_attempt"] = is_run.astype(float)
    p["qb_scramble"] = 0.0
    tgt = is_pass & (p["sack"] != 1)
    p["incomplete_pass"] = (tgt & (rng.uniform(size=n) < 0.35)).astype(float)
    p["interception"] = (tgt & (p["incomplete_pass"] == 1) & (rng.uniform(size=n) < 0.08)).astype(float)
    p["complete_pass"] = (tgt & (p["incomplete_pass"] == 0)).astype(float)
    p["tackled_for_loss"] = (is_run & (rng.uniform(size=n) < 0.1)).astype(float)
    p["fumble_forced"] = ((is_pass | is_run) & (rng.uniform(size=n) < 0.01)).astype(float)
    recv_pool = ["WR1", "WR1", "WR2", "WR2", "WR3", "TE1", "RB1"]
    rb_pool = ["RB1", "RB1", "RB1", "RB2"]
    rec = np.where(tgt, [pid(t, s) for t, s in zip(off, rng.choice(recv_pool, n))], None).astype(object)
    rush = np.where(is_run, [pid(t, s) for t, s in zip(off, rng.choice(rb_pool, n))], None).astype(object)
    # KC's WR1 is out in 2023 week 4: no plays for him that week
    out = (p["season"] == 2023) & (p["week"] == 4) & (rec == "KC_WR1")
    rec[out.to_numpy()] = "KC_WR2"
    p["receiver_player_id"] = rec
    p["rusher_player_id"] = rush
    rushers = ["DE1", "DE2", "DT1", "LB1", "LB3"]
    cover = ["CB1", "CB2", "FS", "SS", "LB2"]
    tack = ["LB1", "LB2", "LB3", "DT1", "DT2", "DE1", "CB1", "SS", "FS"]
    pick = lambda pool: np.array([pid(t, s) for t, s in zip(de, rng.choice(pool, n))], dtype=object)  # noqa: E731
    p["sack_player_id"] = np.where(p["sack"] == 1, pick(rushers), None)
    p["half_sack_1_player_id"] = None
    p["half_sack_2_player_id"] = None
    p["qb_hit_1_player_id"] = np.where(p["qb_hit"] == 1, pick(rushers), None)
    p["qb_hit_2_player_id"] = None
    p["tackle_for_loss_1_player_id"] = np.where(p["tackled_for_loss"] == 1, pick(tack), None)
    p["tackle_for_loss_2_player_id"] = None
    p["forced_fumble_player_1_player_id"] = np.where(p["fumble_forced"] == 1, pick(tack), None)
    p["forced_fumble_player_1_team"] = np.where(p["fumble_forced"] == 1, de, None)
    p["forced_fumble_player_2_player_id"] = None
    p["forced_fumble_player_2_team"] = None
    p["interception_player_id"] = np.where(p["interception"] == 1, pick(cover), None)
    pd_ = (p["incomplete_pass"] == 1) & (rng.uniform(size=n) < 0.4)
    p["pass_defense_1_player_id"] = np.where(pd_, pick(cover), None)
    p["pass_defense_2_player_id"] = None
    p["solo_tackle_1_player_id"] = np.where(is_run | (p["complete_pass"] == 1), pick(tack), None)
    for c in cr.TACKLE_COLS[1:]:
        p[c] = None
    # BUF's LB1 is on reserve from 2023 week 3 and KC's WR3 is inactive in 2023 week 5: no plays for them
    id_cols = [c for c in p.columns if c.endswith("_player_id")]
    gone = {"BUF_LB1": ((p["season"] == 2023) & (p["week"] >= 3), "BUF_LB2"),
            "KC_WR3": ((p["season"] == 2023) & (p["week"] == 5), "KC_WR2")}
    for who, (mask, sub) in gone.items():
        for c in id_cols:
            hit = mask & (p[c] == who)
            p.loc[hit, c] = sub
    p["field_goal_attempt"] = 0.0
    p["extra_point_attempt"] = 0.0
    p["field_goal_result"] = None
    p["kick_distance"] = np.nan
    p["kicker_player_id"] = None
    # field goals: three per team-game, appended as their own rows
    g = p.drop_duplicates(["game_id", "posteam"])[["game_id", "season", "season_type", "week", "home_team",
                                                   "away_team", "posteam", "defteam"]]
    fg = g.loc[g.index.repeat(3)].copy()
    m = len(fg)
    fg["play_type"] = "field_goal"
    fg["field_goal_attempt"] = 1.0
    fg["kick_distance"] = rng.integers(20, 56, m).astype(float)
    fg["field_goal_result"] = np.where(rng.uniform(size=m) < 0.8, "made", "missed")
    fg["kicker_player_id"] = [pid(t, "K1") for t in fg["posteam"]]
    fg["epa"] = rng.normal(0, 1, m)
    fg["wp"] = 0.5
    fg["play_id"] = 1000.0 + np.arange(m)
    out = pd.concat([p, fg], ignore_index=True)
    return out


def make_tables(sched: pd.DataFrame):
    """Normalized depth charts, injuries, weekly rosters, and the position table."""
    reg = sched[sched["game_type"] == "REG"]
    dep, inj, ros = [], [], []
    for (S, W), _ in reg.groupby(["season", "week"]):
        for team in ["KC", "BUF", "NE", "MIA", "DAL", "PHI"]:
            for g, slots in SLOTS.items():
                for i, s in enumerate(slots):
                    rank = 2 if s in ("RB2", "WR3") else 1
                    side = "st" if g == "K" else pos.SIDE[g]
                    dep.append((S, W, team, side, SLOT_CODE[s], g, rank, pid(team, s), "weekly", 0))
                    status = "ACT"
                    if team == "BUF" and s == "LB1" and S == 2023 and W >= 3:
                        status = "RES"
                    if team == "KC" and s == "WR3" and S == 2023 and W == 5:
                        status = "INA"
                    ros.append((S, W, "REG", team, pid(team, s), status, g))
            if team == "KC" and S == 2023 and W == 4:
                inj.append((S, W, "REG", team, "KC_WR1", "WR", "Out", "Did Not Participate"))
            if team == "NE" and S == 2023 and W == 4:
                inj.append((S, W, "REG", team, "NE_CB1", "CB", "Questionable", "Limited"))
    depth = pd.DataFrame(dep, columns=D.DEPTH_COLUMNS)
    depth["rank"] = depth["rank"].astype("Int64")
    injuries = pd.DataFrame(inj, columns=["season", "week", "game_type", "team", "gsis_id", "position",
                                          "report_status", "practice_status"])
    rosters = pd.DataFrame(ros, columns=["season", "week", "game_type", "team", "gsis_id", "status", "position"])
    positions = cr.position_table(depth, rosters, None)
    return depth, injuries, rosters, positions


@pytest.fixture(scope="module")
def world(synth_sched, synth_pbp):
    sched = asof.add_asof(synth_sched)
    pbp = add_players(synth_pbp)
    depth, injuries, rosters, positions = make_tables(sched)
    avail = cr.component_availability(pbp)
    return {"sched": sched, "pbp": pbp, "depth": depth, "injuries": injuries, "rosters": rosters,
            "positions": positions, "avail": avail, "probs": lu.default_probs()}


def _gid(sched, season, week, slot):
    rows = sched[(sched["season"] == season) & (sched["week"] == week) & (sched["game_type"] == "REG")]
    return rows.sort_values("kickoff")["game_id"].iloc[slot]


# --------------------------------------------------------------------------- loaders and small rules

def test_waterfill_caps_and_sums():
    s = cr.waterfill(np.array([10.0, 5.0, 1.0, 0.0]), 2.0)
    assert np.isclose(s.sum(), 2.0) and s.max() <= 1.0 and s[3] == 0.0
    assert s[0] == 1.0 and s[1] > s[2] > 0
    assert np.allclose(cr.waterfill(np.array([3.0, 1.0]), 3.0), [1.0, 1.0])   # fewer players than slots
    assert np.allclose(cr.waterfill(np.array([0.0, 0.0]), 2.0), 0.0)


def test_depth_week_shift_rule_and_normalization(synth_sched):
    sched = asof.add_asof(synth_sched)
    rows = []
    for w in range(1, 8):  # one more REG week than the schedule's six -> shift by one
        rows.append({"season": 2022, "club_code": "KC", "week": w, "game_type": "REG", "depth_team": "1",
                     "formation": "Offense", "gsis_id": f"P{w}", "depth_position": "WR", "position": "WR"})
    dc = pd.DataFrame(rows)
    assert D.depth_week_shift(dc, sched) == {2022: 1}
    n = D.normalize_weekly_depth(dc, sched)
    assert sorted(n["week"]) == [1, 2, 3, 4, 5, 6]            # chart week 1 (preseason) dropped
    assert n.set_index("week").loc[1, "gsis_id"] == "P2"       # chart labelled week 2 is for schedule week 1
    assert set(n["group"]) == {"WR"} and set(n["team"]) == {"KC"}
    bad = dc.assign(week=dc["week"] + 5)
    with pytest.raises(D.DataError):
        D.depth_week_shift(bad, sched)
    assert 2004 in D.DEPTH_DROP and D.DEPTH_FIRST_RELIABLE == 2005


def test_snapshot_depth_uses_last_snapshot_before_kickoff(synth_sched):
    sched = asof.add_asof(synth_sched).assign(season=lambda d: d["season"])
    g = sched[(sched["season"] == 2023) & (sched["week"] == 2) & ((sched["home_team"] == "KC") |
                                                                   (sched["away_team"] == "KC"))].iloc[0]
    kick = g["kickoff"].tz_convert("UTC")
    snaps = [(kick - pd.Timedelta(days=2), "OLD"), (kick - pd.Timedelta(hours=3), "NEW"),
             (kick + pd.Timedelta(hours=1), "AFTER")]
    dc = pd.DataFrame([{"dt": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "team": "KC", "gsis_id": who, "pos_grp": "3WR 1TE",
                        "pos_abb": "WR", "pos_slot": 1, "pos_rank": 1, "season": None} for t, who in snaps])
    out = D.normalize_snapshot_depth(dc, sched, 2023)
    wk2 = out[(out["week"] == 2) & (out["team"] == "KC")]
    assert list(wk2["gsis_id"]) == ["NEW"]
    assert (out["source"] == "snapshot").all() and set(out["group"]) == {"WR"}


def test_group_mapping():
    assert pos.group_of("LDE") == "DL" and pos.group_of("OLB") == "LB" and pos.group_of("SS") == "S"
    assert pos.group_of("LT") == "OL" and pos.group_of("LT", side="def") == "DL"
    assert pos.group_of("FB") == "RB" and pos.group_of("DB") == "CB" and pos.group_of("P") is None
    assert pos.team_id("ARZ", 2010) == "ARI" and pos.team_id("SL", 2010) == "LA" and pos.team_id("XXX", 2010) is None


def test_m5_dev_window_is_guarded(tmp_path):
    assert windows.M5_DEV == (2012, 2019) and not windows.touches_holdout(windows.M5_DEV)
    with pytest.raises(windows.HoldoutError):
        registry.log_run("x", label="m5dev", seasons=(2018, 2020), game_ids=["a"], metrics={"n": 1},
                         data={}, runs_dir=tmp_path)


def test_participation_is_read_only_by_the_validation_module():
    """Decision M5-D1 (a): participation (CC BY-SA) may only be read for validation."""
    allowed = {ROOT / "nflelo/ml/data.py", ROOT / "nflelo/ml/players/validate.py", ROOT / "scripts/ml_m5.py"}
    hits = []
    for path in list((ROOT / "nflelo").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        if path in allowed:
            continue
        if re.search(r"load_participation|['\"]participation['\"]|validate import|players\.validate", path.read_text()):
            hits.append(str(path.relative_to(ROOT)))
    assert not hits, f"participation referenced outside the validation code: {hits}"
    src = (ROOT / "scripts/ml_m5.py").read_text()
    tree = ast.parse(src)
    users = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
             and re.search(r"load_participation|players import validate", ast.get_source_segment(src, n))}
    assert users == {"validate"}, users


# --------------------------------------------------------------------------- credit

def test_credit_signs_sums_and_groups(world):
    pg = cr.player_games(world["pbp"], world["sched"], world["positions"], world["depth"], world["avail"])
    assert not (pg["group"] == "OL").any()
    raw = lambda c: (pg[f"c_{c}"] * pg["team_plays"])  # noqa: E731
    # defensive credit is -EPA of the credited plays: sack credit sums to -EPA of credited sacks (clean plays)
    plays = world["pbp"]
    clean = cr.te.clean_plays(plays)
    sk = clean[(clean["sack"] == 1) & clean["sack_player_id"].notna()]
    assert np.isclose(raw("sack").sum(), -sk["epa"].sum())
    # components are credited only to their groups
    for c in ("sack", "qbhit", "tfl", "ff"):
        assert (pg.loc[~pg["group"].isin(["DL", "LB"]), f"c_{c}"] == 0).all()
    for c in ("int", "pd"):
        assert (pg.loc[~pg["group"].isin(["CB", "S"]), f"c_{c}"] == 0).all()
    assert (pg.loc[pg["group"] != "K", "c_fg"] == 0).all() and (pg.loc[pg["group"] == "K", "share"] == 1).all()
    # shares: at most 1, QB share ~1, KC_WR1 absent in 2023 week 4
    assert pg["share"].between(0, 1).all()
    wk = pg[(pg["season"] == 2023) & (pg["week"] == 4)]
    assert "KC_WR1" not in set(wk["gsis_id"])
    s = pg[pg["game_type"] == "REG"].groupby(["game_id", "team", "group"])["share"].sum()
    assert np.allclose(s.xs("WR", level="group"), 2.0)  # two rank-1 WR slots on the chart (WR3 is rank 2)


def test_receiver_baseline_uses_only_earlier_games(world):
    """Changing a later game's EPA must not change an earlier game's credits (the QB baseline is prior-only)."""
    pbp, sched = world["pbp"], world["sched"]
    a = cr.player_games(pbp, sched, world["positions"], world["depth"], world["avail"])
    late = pbp["game_id"].isin(sched.loc[(sched["season"] == 2023) & (sched["week"] >= 4), "game_id"])
    b = cr.player_games(pbp.assign(epa=np.where(late, pbp["epa"] + 3.0, pbp["epa"])), sched, world["positions"],
                        world["depth"], world["avail"])
    early = a["season"].eq(2022) | (a["season"].eq(2023) & a["week"].lt(4))
    cols = [f"c_{c}" for c in ("recv", "rb_recv", "rush", "db", "sack")]
    assert np.allclose(a.loc[early, cols].to_numpy(), b.loc[early.to_numpy(), cols].to_numpy())


def test_component_availability_detects_missing_stats(world):
    p = world["pbp"].copy()
    p.loc[p["season"] == 2022, "qb_hit_1_player_id"] = None
    p.loc[(p["season"] == 2022) & (p["incomplete_pass"] == 1), "receiver_player_id"] = None
    av = cr.component_availability(p)
    assert av.loc[2022, "qbhit"] == 0 and av.loc[2022, "recv"] == 0
    assert av.loc[2023, "qbhit"] == 1 and av.loc[2023, "recv"] == 1


# --------------------------------------------------------------------------- lineups

def test_lineup_layers(world):
    base = lu.attach_statuses(lu.chart_players(world["depth"]), world["injuries"], world["rosters"])
    probs = world["probs"]
    full = base.assign(p=lu.play_probability(base, probs, lu.LineupConfig()))
    fri = base.assign(p=lu.play_probability(base, probs, lu.LineupConfig(friday_only=True)))
    binr = base.assign(p=lu.play_probability(base, probs, lu.LineupConfig(binary=True)))
    get = lambda df, S, W, who: float(df[(df["season"] == S) & (df["week"] == W) & (df["gsis_id"] == who)]["p"].iloc[0])  # noqa: E731
    assert get(full, 2023, 4, "KC_WR1") == 0.0                     # Out
    assert get(full, 2023, 4, "NE_CB1") == pytest.approx(lu.SCOPE_PROBS["questionable"])
    assert get(binr, 2023, 4, "NE_CB1") == 1.0                     # B5: Questionable counts as in
    assert get(full, 2023, 3, "BUF_LB1") == 0.0                    # RES
    assert get(full, 2023, 5, "KC_WR3") == 0.0 and get(fri, 2023, 5, "KC_WR3") == 1.0   # INA only without friday_only
    assert get(full, 2023, 2, "BUF_LB1") == 1.0
    # roster statuses are ignored outside the seasons where they are week-accurate
    b2 = lu.attach_statuses(lu.chart_players(world["depth"]), world["injuries"], world["rosters"], roster_seasons=[2022])
    p2 = b2.assign(p=lu.play_probability(b2, probs))
    assert get(p2, 2023, 3, "BUF_LB1") == 1.0 and get(p2, 2023, 4, "KC_WR1") == 0.0
    so = lu.starters_out(full.rename(columns={"p": "p_play"}))
    assert so[(2023, 4, "KC")] == 1 and so[(2023, 3, "BUF")] == 1


def test_roster_status_seasons_rejects_season_end_statuses(world):
    ros = world["rosters"].copy()
    pbp = world["pbp"]
    # 2022: mark players RES in weeks they actually played -> not week-accurate
    played = pbp[pbp["season"] == 2022][["week", "rusher_player_id"]].dropna().drop_duplicates()
    m = (ros["season"] == 2022) & ros["gsis_id"].isin(played["rusher_player_id"])
    ros.loc[m, "status"] = "RES"
    ok = lu.roster_status_seasons(ros, pbp, min_rows=1)
    assert 2022 not in ok and 2023 in ok


def test_status_probabilities_estimate_ratio(world):
    rows = []
    for i in range(200):
        rows.append({"season": 2010, "week": 1 + i % 17, "team": "KC", "gsis_id": f"Q{i}", "group": "WR", "rank": 1,
                     "status": "questionable", "roster_status": "ACT"})
        rows.append({"season": 2010, "week": 1 + i % 17, "team": "KC", "gsis_id": f"H{i}", "group": "WR", "rank": 1,
                     "status": "none", "roster_status": "ACT"})
    lineup = pd.DataFrame(rows)
    played = pd.DataFrame({"season": 2010, "week": lineup["week"], "gsis_id": lineup["gsis_id"]})
    keep = [g.startswith("H") and int(g[1:]) % 10 != 0 or g.startswith("Q") and int(g[1:]) % 2 == 0
            for g in played["gsis_id"]]
    pr = lu.estimate_status_probs(lineup, played[keep], seasons=(2010, 2010)).set_index(["status", "group"])["p"]
    # P(usage | Q) = 0.5, P(usage | healthy) = 0.9 -> 0.556 for WR (200 starters, light shrinkage)
    assert pr[("questionable", "WR")] == pytest.approx(0.5 / 0.9, abs=0.01)
    assert pr[("none", "WR")] == 1.0


# --------------------------------------------------------------------------- values

def test_box_values_shrink_and_follow_players(world):
    st = rf.prepare(world["pbp"], world["sched"], world["depth"], world["injuries"], world["rosters"],
                    world["positions"], world["probs"], world["avail"], VCFG)
    bv = pv.BoxValues(st.pg, st.sched)
    bs = bv.sums(2023, 3, VCFG.half_life)
    k_small = bs.values({g: 0.01 for g in pos.GROUPS}, st.R[2023])
    k_big = bs.values({g: 1e6 for g in pos.GROUPS}, st.R[2023])
    assert np.abs(k_big).max() < 1e-4 < np.abs(k_small).max()        # huge prior -> replacement level (0)
    assert np.abs(bs.values(VCFG.kd, st.R[2023])).max() <= np.abs(k_small).max() + 1e-12
    # nothing from the week itself or later: the 2023 week 3 sums equal sums on data truncated before it
    cut = st.pg[st.pg["kick_ns"] < bv.asofs[(2023, 3)]]
    bs2 = pv.BoxValues(cut, st.sched).sums(2023, 3, VCFG.half_life)
    assert np.allclose(bs.num, bs2.num) and list(bs.players) == list(bs2.players)


# --------------------------------------------------------------------------- leakage (every builder, plus the positive control)

def _builder(world, cfg):
    def b(pbp, sched, games, tables):
        return rf.build_features(pbp, sched, games, tables["depth"], tables["injuries"], tables["rosters"],
                                 world["positions"], world["probs"], world["avail"], cfg)
    return b


@pytest.fixture(scope="module")
def targets(world):
    s = world["sched"]
    return [_gid(s, 2023, 1, 0), _gid(s, 2023, 3, 1), _gid(s, 2023, 4, 0), _gid(s, 2023, 4, 2), _gid(s, 2023, 5, 1)]


@pytest.mark.parametrize("name,cfg", [
    ("box", rf.RosterConfig(values="box", value_cfg=VCFG)),
    ("ww", rf.RosterConfig(values="ww", value_cfg=VCFG)),
    ("ww_friday", rf.RosterConfig(values="ww", value_cfg=VCFG, lineup=lu.LineupConfig(friday_only=True))),
    ("ww_binary", rf.RosterConfig(values="ww", value_cfg=VCFG, lineup=lu.LineupConfig(binary=True))),
])
def test_roster_builders_are_leak_free(world, targets, name, cfg):
    tables = {"depth": world["depth"], "injuries": world["injuries"], "rosters": world["rosters"]}
    res = rf.roster_leakage_check(_builder(world, cfg), world["pbp"], world["sched"], tables, targets)
    assert res["leak_free"].all(), f"{name}\n{res.to_string()}"
    feats = _builder(world, cfg)(world["pbp"], world["sched"],
                                 world["sched"][world["sched"]["game_id"].isin(targets)], tables)
    assert feats.notna().all().all()
    assert (feats[["lineup_delta_diff", "preseason_change_diff"]].abs() > 1e-9).any().all(), \
        f"{name}: features all zero; the check would pass trivially"


def test_roster_positive_control_fails(world, targets):
    tables = {"depth": world["depth"], "injuries": world["injuries"], "rosters": world["rosters"]}
    cfg = rf.RosterConfig(values="box", value_cfg=VCFG, leaky=True)
    res = rf.roster_leakage_check(_builder(world, cfg), world["pbp"], world["sched"], tables, targets[1:])
    assert not res["leak_free"].any(), res.to_string()


def test_later_weekly_tables_cannot_reach_a_game(world, targets):
    """Scrambling only later weeks' depth charts, injuries and rosters leaves the features unchanged."""
    cfg = rf.RosterConfig(values="box", value_cfg=VCFG)
    tables = {"depth": world["depth"], "injuries": world["injuries"], "rosters": world["rosters"]}
    g = world["sched"][world["sched"]["game_id"] == targets[2]]
    a = _builder(world, cfg)(world["pbp"], world["sched"], g, tables)
    b = _builder(world, cfg)(world["pbp"], world["sched"], g, rf.corrupt_aux(tables, 2023, 4, seed=3))
    assert np.allclose(a.to_numpy(), b.to_numpy())
    # ...while this week's own table (KC_WR1 Out) does change them
    t2 = dict(tables, injuries=tables["injuries"].iloc[0:0])
    c = _builder(world, cfg)(world["pbp"], world["sched"], world["sched"][world["sched"]["game_id"].isin(
        world["sched"].loc[(world["sched"]["season"] == 2023) & (world["sched"]["week"] == 4), "game_id"])], t2)
    d = _builder(world, cfg)(world["pbp"], world["sched"], world["sched"][world["sched"]["game_id"].isin(c.index)],
                             tables)
    assert not np.allclose(c.to_numpy(), d.to_numpy())


def test_corrupt_from_shuffles_credited_ids(world):
    s = world["sched"]
    as_of = s.loc[(s["season"] == 2023) & (s["week"] == 4), "as_of"].min()
    p2, _ = asof.corrupt_from(world["pbp"], s, as_of, seed=1)
    late = world["pbp"]["game_id"].isin(s.loc[s["kickoff"] >= as_of, "game_id"]).to_numpy()
    for c in ("receiver_player_id", "sack_player_id", "solo_tackle_1_player_id", "kicker_player_id"):
        assert (p2.loc[~late, c].fillna("") == world["pbp"].loc[~late, c].fillna("")).all()
        assert (p2.loc[late, c].fillna("") != world["pbp"].loc[late, c].fillna("")).any()


def test_ridge_reads_groups_from_rows_before_asof(world):
    """A later position change must not reach an earlier with/without fit (it sets the prior strength)."""
    st = rf.prepare(world["pbp"], world["sched"], world["depth"], world["injuries"], world["rosters"],
                    world["positions"], world["probs"], world["avail"], VCFG)
    from nflelo.ml.features import opponent_adjust as oa
    tab = oa.rating_table(world["pbp"], st.sched, oa.TUNED)
    keys = sorted(set(zip(tab["season"].astype(int), tab["week"].astype(int))))
    ratings = oa.compute_ratings(tab, st.sched, keys, oa.TUNED, ("all",))
    rows = pv.residual_rows(tab, ratings)
    box = pv.box_values(pv.BoxValues(st.pg, st.sched), [(2023, 3)], VCFG, st.R)
    of2 = st.onfield.copy()
    late = of2["season"].eq(2023) & of2["week"].ge(3) & of2["gsis_id"].eq("KC_WR1")
    of2.loc[late, "group"] = "TE"
    a = pv.RidgeData(rows, st.onfield, st.sched).fit(2023, 3, box, VCFG, "off")
    b = pv.RidgeData(rows, of2, st.sched).fit(2023, 3, box, VCFG, "off")
    assert np.allclose(a["e"].to_numpy(), b["e"].to_numpy()) and list(a["group"]) == list(b["group"])
