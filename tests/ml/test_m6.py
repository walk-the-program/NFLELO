"""M6 play outcome model (CC BY-SA 4.0 code paths): bins, CRPS, allowlist, table builder, models, leakage.

Offline and synthetic: a small raw world (kept plays, play-by-play, participation with post-snap columns,
team ratings) built here, plus the conftest schedule and play-by-play for the rating as-of check.
"""
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nflelo.ml import asof as asof_mod
from nflelo.ml import registry
from nflelo.ml.eval import windows
from nflelo.ml.features import opponent_adjust as oa
from nflelo.ml.players import participation as pt
from nflelo.ml.plays import baseline as bl
from nflelo.ml.plays import data as pdata
from nflelo.ml.plays import gbm
from nflelo.ml.plays import metrics as pm
from nflelo.ml.plays import net

ROOT = Path(__file__).resolve().parents[2]
TEAMS = ["T0", "T1", "T2", "T3"]
POST_SNAP = ["number_of_pass_rushers", "defense_coverage_type", "defense_man_zone_type", "was_pressure", "route",
             "time_to_throw", "ngs_air_yards"]


# --------------------------------------------------------------------------- synthetic world

def raw_world(seasons=(2016, 2017, 2018), games_per_season=24, plays_per_game=60, seed=0):
    rng = np.random.default_rng(seed)
    kept, pbp, part, rat = [], [], [], []
    for S in seasons:
        for gi in range(games_per_season):
            week = gi // 2 + 1
            a, b = rng.choice(4, 2, replace=False)
            gid = f"{S}_{week:02d}_{TEAMS[a]}_{TEAMS[b]}_{gi}"
            n = plays_per_game
            pid = np.arange(1, n + 1)
            off = np.where(pid % 2 == 0, TEAMS[a], TEAMS[b])
            de = np.where(pid % 2 == 0, TEAMS[b], TEAMS[a])
            k = pd.DataFrame({"game_id": gid, "play_id": pid, "season": S, "week": week, "season_type": "REG",
                              "kick_ns": np.int64(S * 10 ** 6 + gi), "off_team": off, "def_team": de,
                              "home": (off == TEAMS[b]).astype(float), "epa": rng.normal(0, 1, n)})
            # 11 distinct IDs per side out of a 14-man unit, sorted like the cleaner does
            for side, n_ in (("o", 11), ("d", 11)):
                cols = [f"{side}{c + 1}" for c in range(n_)]
                team = off if side == "o" else de
                k[cols] = np.array([sorted(f"{t}_{side}{j:02d}" for j in rng.choice(14, 11, replace=False))
                                    for t in team], dtype=object)
            kept.append(k)
            is_pass = (rng.uniform(size=n) < 0.6).astype(float)
            yl = rng.integers(1, 100, n).astype(float)
            shotgun = np.where(is_pass == 1, rng.uniform(size=n) < 0.8, rng.uniform(size=n) < 0.3).astype(float)
            yards = np.round(np.where(is_pass == 1, rng.choice([0, 0, 0, 5, 8, 12, 25], n), rng.normal(4, 4, n)))
            yards = np.minimum(yards, yl)
            td = (yards >= yl).astype(float)
            pbp.append(pd.DataFrame({
                "game_id": gid, "play_id": pid.astype(float), "season": S, "down": rng.integers(1, 5, n).astype(float),
                "ydstogo": rng.integers(1, 16, n).astype(float), "yardline_100": yl,
                "score_differential": rng.integers(-14, 15, n).astype(float),
                "half_seconds_remaining": rng.integers(0, 1800, n).astype(float),
                "game_seconds_remaining": rng.integers(0, 3600, n).astype(float),
                "posteam_timeouts_remaining": 3.0, "defteam_timeouts_remaining": 3.0,
                "roof": rng.choice(["outdoors", "dome", "closed", "open"], n),
                "temp": rng.normal(60, 10, n), "wind": rng.uniform(0, 15, n),
                "pass": is_pass, "shotgun": shotgun, "yards_gained": yards, "touchdown": td,
                "td_team": np.where(td == 1, off, None), "posteam": off,
                "interception": (rng.uniform(size=n) < 0.02 * is_pass).astype(float),
                "fumble_lost": (rng.uniform(size=n) < 0.01).astype(float)}))
            form = np.where(shotgun == 1, "SHOTGUN", rng.choice(["SINGLEBACK", "I_FORM"], n))
            form = np.where(rng.uniform(size=n) < 0.03, None, form)   # a few missing formations
            q = pd.DataFrame({"nflverse_game_id": gid, "play_id": pid, "offense_formation": form,
                              "offense_personnel": rng.choice(["1 RB, 1 TE, 3 WR", "1 RB, 2 TE, 2 WR",
                                                               "6 OL, 2 RB, 1 TE, 1 WR"], n),
                              "defense_personnel": rng.choice(["4 DL, 2 LB, 5 DB", "4 DL, 3 LB, 4 DB"], n),
                              "defenders_in_box": rng.integers(5, 9, n).astype(float)})
            for c in POST_SNAP:
                q[c] = rng.integers(0, 6, n)
            part.append(q)
        for wk in range(1, games_per_season // 2 + 1):
            for kind in ("pass", "rush"):
                rat.append(pd.DataFrame({"kind": kind, "season": S, "week": wk, "team": TEAMS,
                                         "off": rng.normal(0, 0.1, 4), "def": rng.normal(0, 0.1, 4),
                                         "mu": 0.0, "h": 0.0, "w_off": 1.0, "w_def": 1.0}))
    return (pd.concat(kept, ignore_index=True), pd.concat(pbp, ignore_index=True),
            pd.concat(part, ignore_index=True), pd.concat(rat, ignore_index=True))


@pytest.fixture(scope="module")
def world():
    return raw_world()


@pytest.fixture(scope="module")
def table(world):
    return pdata.build_table(*world)


def corrupt_outcomes(pbp, part, S, seed=1):
    rng = np.random.default_rng(seed)
    p, q = pbp.copy(), part.copy()
    m = (p["season"] == S).to_numpy()
    p.loc[m, "yards_gained"] = rng.integers(-15, 60, int(m.sum())).astype(float)
    for c in ("touchdown", "interception", "fumble_lost"):
        p.loc[m, c] = rng.integers(0, 2, int(m.sum())).astype(float)
    qm = q["nflverse_game_id"].str.startswith(f"{S}_").to_numpy()
    for c in POST_SNAP:
        q.loc[qm, c] = rng.permutation(q.loc[qm, c].to_numpy())
    return p, q


# --------------------------------------------------------------------------- bins, fold, CRPS

def test_yards_to_bin_and_fold():
    yl = np.array([50, 50, 50, 50, 50, 50, 5, 95, 3])
    yards = np.array([-12, -9, 0, 40, 47, 60, 7, -10, 3])
    td = np.array([0, 0, 0, 0, 0, 1, 0, 0, 1], bool)
    b, counted = pdata.yards_to_bin(yards, td, yl)
    assert b.tolist() == [0, 1, 10, 50, 51, 52, 14, 5, 52]
    assert counted.tolist() == [-12, -9, 0, 40, 47, 50, 4, -5, 3]
    f = pdata.fold_index(np.array([5, 95, 50]))
    assert (f[0, :15] == np.arange(15)).all() and (f[0, 15:] == 52).all()          # 5+ yards from the 5 is a TD
    assert (f[1, :5] == 5).all() and (f[1, 5:] == np.arange(5, 53)).all()          # safety floor from the 95
    assert (f[2] == np.arange(53)).all()
    p = np.full((3, 53), 1 / 53)
    q = pdata.fold(p, np.array([5, 95, 50]), floor=0.0)
    assert np.allclose(q.sum(axis=1), 1) and q[0, 20] == 0 and np.isclose(q[0, 52], 38 / 53)
    assert (q[1, :5] == 0).all() and np.isclose(q[1, 5], 6 / 53)


def test_crps_hand_computed():
    p = np.zeros((4, 53))
    p[0, 10] = p[0, 12] = 0.5                 # half on 0 yards, half on 2 yards; observed 1 yard
    p[1, 10] = 1.0                            # point forecast 0 yards; observed 3 yards
    p[2, 10] = 1.0                            # from the 3: 0 yards forecast, a TD observed (3 yards)
    p[3, 13] = 1.0                            # perfect
    b = np.array([11, 13, 52, 13])
    yl = np.array([50, 50, 3, 50])
    c = pm.crps(p, b, yl)
    assert np.allclose(c, [0.5, 3.0, 3.0, 0.0])
    ll = pm.logloss(p, b)
    assert np.isclose(ll[3], 0.0) and ll[0] > 20


def test_event_probs():
    p = np.zeros((2, 53))
    p[0, [5, 15, 31, 52]] = 0.25              # -5, +5, +21, TD
    p[1, [9, 51]] = 0.5                       # -1, 41+
    ev = pm.event_probs(p, np.array([5, 50]), np.array([60, 80]))
    assert np.allclose(ev["first"], [0.75, 0.5]) and np.allclose(ev["20plus"], [0.5, 0.5])
    assert np.allclose(ev["loss"], [0.25, 0.5])
    ev = pm.event_probs(p, np.array([5, 50]), np.array([10, 80]))
    assert np.isclose(ev["20plus"][0], 0.25)  # a TD from the 10 is not a 20-yard gain


# --------------------------------------------------------------------------- allowlist / banned columns

def test_banned_post_snap_fields_are_never_features(table):
    for view in pdata.VIEWS:
        assert pdata.check_features(pdata.FEATURES[view]) == pdata.FEATURES[view]
        assert not [c for c in pdata.FEATURES[view] if pdata.is_banned(c)]
    banned = ["number_of_pass_rushers", "defense_coverage_type", "defense_man_zone_type", "was_pressure", "route",
              "time_to_throw", "air_yards", "ngs_air_yards", "yards_after_catch", "epa", "qb_epa", "wpa",
              "field_goal_result", "yards_gained", "touchdown", "interception", "fumble_lost", "turnover", "yards",
              "bin", "own_yards", "first_down", "sack"]
    for c in banned:
        assert pdata.is_banned(c), c
        with pytest.raises(ValueError):
            pdata.check_features(pdata.FEATURES["call"] + [c])
    with pytest.raises(ValueError):
        pdata.check_features(["down", "kick_ns"])                  # not on the allowlist either
    # the builder never reads a post-snap participation column
    assert not [c for c in pdata.PART_COLUMNS if pdata.is_banned(c)]
    assert not set(POST_SNAP) & set(table.columns)
    # every model input frame is exactly the allowlist
    assert set(pdata.FEATURES["call"]) <= set(table.columns)


def test_personnel_formation_parsing():
    assert pdata.parse_personnel("1 RB, 1 TE, 3 WR", "off") == {"rb": 1, "te": 1, "wr": 3, "ol": 5}
    assert pdata.parse_personnel("6 OL, 2 RB, 1 TE, 1 WR", "off") == {"rb": 2, "te": 1, "wr": 1, "ol": 6}
    assert pdata.parse_personnel("1 C, 2 G, 1 QB, 1 RB, 1 FB, 2 T, 1 TE, 2 WR", "off") == \
        {"rb": 2, "te": 1, "wr": 2, "ol": 5}
    assert pdata.parse_personnel("3 CB, 2 DE, 2 DT, 1 FS, 1 MLB, 1 OLB, 1 SS", "def") == {"dl": 4, "lb": 2, "db": 5}
    assert pdata.parse_personnel("4 DL, 2 LB, 5 DB", "def") == {"dl": 4, "lb": 2, "db": 5}
    assert np.isnan(pdata.parse_personnel(None, "def")["dl"])
    f = pdata.harmonize_formation(pd.Series(["SHOTGUN", "EMPTY", "I_FORM", "UNDER CENTER", "PISTOL", "WILDCAT", None]))
    assert f.tolist()[:6] == ["shotgun", "shotgun", "under_center", "under_center", "pistol", "other"]
    g = pdata.impute_formation(f, pd.Series([0, 0, 0, 0, 0, 0, 1.0]))
    assert g.iloc[-1] == "shotgun" and g.iloc[0] == "shotgun"


def test_build_table_fills_missing_without_flags(world, table):
    kept, pbp, part, rat = world
    assert len(table) == len(kept)
    # only the weather fields can be missing (indoors); missingness elsewhere could leak (NGS formation)
    na = [c for c in pdata.FEATURES["call"] if table[c].isna().any()]
    assert set(na) <= {"temp", "wind"}
    form_sum = table[["form_shotgun", "form_pistol", "form_under_center"]].sum(axis=1)
    assert (form_sum == 1).all()                 # synthetic formations are all shotgun/under center after imputation
    # missing formations follow play-by-play's shotgun flag
    j = table.merge(pbp[["game_id", "play_id", "shotgun"]].assign(play_id=lambda d: d["play_id"].astype("int64")),
                    on=["game_id", "play_id"]).merge(
        part[["nflverse_game_id", "play_id", "offense_formation"]].rename(columns={"nflverse_game_id": "game_id"}),
        on=["game_id", "play_id"])
    miss = j["offense_formation"].isna()
    assert miss.any() and (j.loc[miss, "form_shotgun"] == j.loc[miss, "shotgun"]).all()
    assert table["bin"].between(0, 52).all() and set(np.unique(table["td"])) <= {0.0, 1.0}
    # bins are always possible from the play's own yard line
    assert pdata.possible_mask(table["yardline_100"].to_numpy())[np.arange(len(table)), table["bin"]].all()


# --------------------------------------------------------------------------- leakage

def test_corrupting_season_outcomes_leaves_its_features_unchanged(world, table):
    kept, pbp, part, rat = world
    S = 2018
    p2, q2 = corrupt_outcomes(pbp, part, S)
    dirty = pdata.build_table(kept, p2, q2, rat)
    m = (table["season"] == S).to_numpy()
    cols = pdata.FEATURES["call"] + pt.OFF_COLS + pt.DEF_COLS
    pd.testing.assert_frame_equal(table.loc[m, cols], dirty.loc[m, cols])
    assert (table.loc[m, "bin"] != dirty.loc[m, "bin"]).mean() > 0.5        # the corruption is real
    pd.testing.assert_frame_equal(table.loc[~m], dirty.loc[~m])
    # positive control: a feature built from the play's own yards changes
    yards_fn = {"own_yards": lambda j: j["yards_gained"]}
    a = pdata.build_table(kept, pbp, part, rat, extra_features=yards_fn)
    b = pdata.build_table(kept, p2, q2, rat, extra_features=yards_fn)
    assert (a.loc[m, "own_yards"] != b.loc[m, "own_yards"]).any()


def _season_ahead_preds(t, S, models=("baseline", "gbm", "N0", "N1")):
    train, test = t[t["season"] < S], t[t["season"] == S]
    inner, val = train[train["season"] < S - 1], train[train["season"] == S - 1]
    out = {}
    if "baseline" in models:
        k, _ = bl.tune_k(inner, val, call=True, grid=(5.0, 50.0))
        out["baseline"] = bl.Baseline.fit(train, True, k).predict(test)
    if "gbm" in models:
        out["gbm"] = gbm.fit_season_ahead(train, "call", grid=({"learning_rate": 0.1, "max_leaf_nodes": 7},)).predict(test)
    for v in [m for m in models if m.startswith("N")]:
        cfg = net.NetConfig(variant=v, hidden=16, phi=8, emb_dim=4, max_epochs=2, patience=1)
        model, _ = net.fit_season_ahead(train, "call", [cfg])
        out[v] = model.predict(test)
    return out


def test_season_ahead_predictions_ignore_the_target_season(world, table):
    """Corrupt season S entirely (every outcome and post-snap field), rebuild, refit: S predictions unchanged."""
    pytest.importorskip("torch")  # requirements-m6.txt
    kept, pbp, part, rat = world
    S = 2018
    p2, q2 = corrupt_outcomes(pbp, part, S)
    dirty = pdata.build_table(kept, p2, q2, rat)
    a = _season_ahead_preds(table, S)
    b = _season_ahead_preds(dirty, S)
    for m in a:
        assert np.array_equal(a[m][0], b[m][0]) and np.array_equal(a[m][1], b[m][1]), m
    # positive control: a model trained on S too (or fed the play's own yards) changes
    c1 = bl.Baseline.fit(table, True, 5.0).predict(table[table["season"] == S])[0]
    c2 = bl.Baseline.fit(dirty, True, 5.0).predict(dirty[dirty["season"] == S])[0]
    assert np.abs(c1 - c2).max() > 1e-6


def test_rating_features_use_only_games_before_the_week(synth_pbp, synth_sched):
    sched = asof_mod.add_asof(synth_sched)
    tab = oa.rating_table(synth_pbp, sched, oa.TUNED)
    S, W = 2023, 4
    games = sched[(sched["season"] == S) & (sched["week"] == W)]
    plays = pd.DataFrame({"season": S, "week": W, "off_team": games["home_team"].to_numpy(),
                          "def_team": games["away_team"].to_numpy()})
    a = pdata.rating_features(plays, pdata.week_ratings(tab, sched, [(S, W)]))
    p2, s2 = asof_mod.corrupt_from(synth_pbp, sched, games["as_of"].min(), seed=3)
    tab2 = oa.rating_table(p2, s2, oa.TUNED)
    b = pdata.rating_features(plays, pdata.week_ratings(tab2, s2, [(S, W)]))
    pd.testing.assert_frame_equal(a, b)
    assert (a.abs() > 0).any().any()
    # control: the next week's ratings include this week's (corrupted) games
    nxt = plays.assign(week=W + 1)
    c = pdata.rating_features(nxt, pdata.week_ratings(tab, sched, [(S, W + 1)]))
    d = pdata.rating_features(nxt, pdata.week_ratings(tab2, s2, [(S, W + 1)]))
    assert (c - d).abs().to_numpy().max() > 1e-9


# --------------------------------------------------------------------------- models

def test_baseline_smoothing_and_fold(table):
    t = table[table["season"] < 2018]
    small = bl.Baseline.fit(t, False, 1e-9)
    big = bl.Baseline.fit(t, False, 1e9)
    p_small, _ = small.predict(t.iloc[:200])
    p_big, _ = big.predict(t.iloc[:200])
    yl = t["yardline_100"].to_numpy().astype(int)[:200]
    root = pdata.fold(np.stack([big.levels[0][0] @ bl.FOLD[y] for y in yl]), yl)
    assert np.allclose(p_big, root, atol=1e-6)              # huge k: every cell is the root distribution
    assert np.abs(p_small - p_big).max() > 0.01              # tiny k: the cell's own history
    assert (p_small[~pdata.possible_mask(yl)] == 0).all()
    assert np.allclose(p_small.sum(axis=1), 1)


def test_net_is_permutation_invariant_and_deterministic(table):
    pytest.importorskip("torch")  # requirements-m6.txt
    tr = table[table["season"] < 2018]
    te = table[table["season"] == 2018].iloc[:300]
    cfg = net.NetConfig(variant="N1", hidden=16, phi=8, emb_dim=4)
    m1 = net.fit(tr, "call", cfg, epochs=2)
    m2 = net.fit(tr, "call", cfg, epochs=2)
    p1, t1 = m1.predict(te)
    p2, t2 = m2.predict(te)
    assert np.array_equal(p1, p2) and np.array_equal(t1, t2)
    rng = np.random.default_rng(0)
    sh = te.copy()
    sh[pt.OFF_COLS] = np.array([rng.permutation(r) for r in te[pt.OFF_COLS].to_numpy(object)])
    sh[pt.DEF_COLS] = np.array([rng.permutation(r) for r in te[pt.DEF_COLS].to_numpy(object)])
    p3, _ = m1.predict(sh)
    assert np.allclose(p1, p3, atol=1e-6)
    assert np.allclose(p1.sum(axis=1), 1) and (p1[~pdata.possible_mask(te["yardline_100"].to_numpy())] == 0).all()
    # N2: prior features enter per player; unseen players are the zero embedding
    priors = {S: pd.DataFrame({"box": 0.01, "rapm": 0.0, "has_box": 1.0},
                              index=np.unique(np.concatenate([np.char.add("off|", table[pt.OFF_COLS].to_numpy(str).ravel()),
                                                              np.char.add("def|", table[pt.DEF_COLS].to_numpy(str).ravel())])))
              for S in (2016, 2017, 2018)}
    m4 = net.fit(tr, "call", net.NetConfig(variant="N2", hidden=16, phi=8, emb_dim=4), priors=priors, epochs=1)
    p4, _ = m4.predict(te, priors)
    assert np.allclose(p4.sum(axis=1), 1)


def test_gbm_fold_and_turnover(table):
    tr = table[table["season"] < 2018]
    te = table[table["season"] == 2018]
    m = gbm.fit(tr, "situation", {"learning_rate": 0.1, "max_leaf_nodes": 7}, n_iter=5, tov_iter=5)
    p, tov = m.predict(te)
    assert p.shape == (len(te), 53) and np.allclose(p.sum(axis=1), 1)
    assert (p[~pdata.possible_mask(te["yardline_100"].to_numpy())] == 0).all()
    assert ((tov > 0) & (tov < 1)).all()
    with pytest.raises(ValueError):
        pdata.check_features(list(m.bins.feature_names_in_) + ["epa"])


# --------------------------------------------------------------------------- boundaries

M6_FILES = {"nflelo/ml/plays/__init__.py", "nflelo/ml/plays/data.py", "nflelo/ml/plays/metrics.py",
            "nflelo/ml/plays/baseline.py", "nflelo/ml/plays/gbm.py", "nflelo/ml/plays/net.py", "scripts/ml_m6.py"}


def test_m6_stays_inside_its_license_boundary():
    """M6 is CC BY-SA 4.0: its modules say so, and nothing outside M6 imports them."""
    for p in M6_FILES:
        assert "CC BY-SA 4.0" in (ROOT / p).read_text(), p
    hits = []
    for path in list((ROOT / "nflelo").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        rel = str(path.relative_to(ROOT))
        if rel in M6_FILES:
            continue
        tree = ast.parse(path.read_text())
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                mod = getattr(n, "module", None) or ""
                names = [a.name for a in n.names]
                if re.search(r"(^|\.)plays($|\.)", mod) or any(re.search(r"(^|\.)plays($|\.)", x) for x in names):
                    hits.append(rel)
    assert not hits, f"M6 (CC BY-SA) imported outside M6: {hits}"


def test_m6_dev_window_is_guarded(tmp_path):
    assert windows.M6_DEV == (2018, 2019) and not windows.touches_holdout(windows.M6_DEV)
    with pytest.raises(windows.HoldoutError):
        registry.log_run("x", label="m6dev", seasons=(2019, 2020), game_ids=["a"], metrics={"n": 1},
                         data={}, runs_dir=tmp_path)


def test_calibration_rule_has_a_practical_floor():
    """M6 calibration checks pass if ECE <= null p95 OR ECE <= 0.010 (decided 2026-10-05, before any holdout)."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0.1, 0.5, 60000)
    y = (rng.uniform(size=p.size) < np.clip(p + 0.006, 0, 1)).astype(float)   # off by 0.6 points
    c = pm.calibration_check(y, p, reps=50)
    assert pm.ECE_FLOOR == 0.010 and not c["ok_null"] and c["ok"] and c["ece"] <= 0.010
    y2 = (rng.uniform(size=p.size) < np.clip(p + 0.03, 0, 1)).astype(float)   # off by 3 points
    c2 = pm.calibration_check(y2, p, reps=50)
    assert not c2["ok_null"] and not c2["ok"]
