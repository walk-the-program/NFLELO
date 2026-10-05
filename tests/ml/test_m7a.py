"""M7a: WP model (CC BY), kicking models (CC BY), fourth-down valuation (CC BY-SA 4.0 code path), leakage.

Offline and synthetic: a small simulated league (four teams, 2012-2018, eight weeks) with coherent drives,
field goals, punts, kickoffs and fourth-down attempts, a synthetic M6 play table and synthetic ratings.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nflelo import config
from nflelo.ml import registry
from nflelo.ml.decisions import fourth as fd
from nflelo.ml.decisions import kicking as kk
from nflelo.ml.eval import windows
from nflelo.ml.plays import data as pdata
from nflelo.ml.wp import model as wpm

sys.path.insert(0, str(config.ROOT / "scripts"))

TEAMS = ["T0", "T1", "T2", "T3"]
SEASONS = range(2012, 2019)
WEEKS = 8
GRID1 = ({"learning_rate": 0.1, "max_leaf_nodes": 7},)


# --------------------------------------------------------------------------- synthetic league

def simulate_game(rng, S, w, home, away, n_plays=90):
    hok = int(rng.integers(0, 2))
    pos = home if hok == 1 else away
    score = {home: 0, away: 0}
    rows = []
    yl, down, ytg = 75.0, 1, 10
    pid = 1
    k_id = {t: f"{t}_K{1 + (S % 2)}" for t in (home, away)}
    p_id = {t: f"{t}_P1" for t in (home, away)}
    other = lambda t: away if t == home else home  # noqa: E731
    roof = rng.choice(["outdoors", "dome"])
    temp, wind = float(rng.normal(60, 10)), float(rng.uniform(0, 15))

    def add(play_type, **kw):
        nonlocal pid
        secs = max(3600 - pid * 3600 / (n_plays + 6), 1.0)
        qtr = int(min(4, 1 + (3600 - secs) // 900))
        base = {"game_id": f"{S}_{w:02d}_{away}_{home}", "play_id": float(pid), "season": S, "season_type": "REG",
                "week": w, "qtr": float(qtr), "home_team": home, "away_team": away, "posteam": pos,
                "defteam": other(pos), "down": float(down), "ydstogo": float(min(ytg, yl)), "yardline_100": float(yl),
                "game_seconds_remaining": float(round(secs)),
                "half_seconds_remaining": float(round(secs - 1800 if qtr <= 2 else secs)),
                "score_differential": float(score[pos] - score[other(pos)]), "posteam_timeouts_remaining": 3.0,
                "defteam_timeouts_remaining": 3.0, "play_type": play_type, "home_opening_kickoff": float(hok),
                "field_goal_result": None, "kicker_player_id": None, "punter_player_id": None,
                "return_touchdown": 0.0, "td_team": None, "touchdown": 0.0, "fourth_down_converted": 0.0,
                "own_kickoff_recovery": 0.0, "roof": roof, "temp": temp, "wind": wind,
                "pass": float(play_type == "pass"), "desc": f"{play_type} {pid}", "yards_gained": 0.0,
                "vegas_wp": float(rng.uniform()), "spread_line": float(rng.normal(0, 4)), "wp": float(rng.uniform())}
        base.update(kw)
        rows.append(base)
        pid += 1

    def kickoff(receiver):
        nonlocal pos, yl, down, ytg
        pos = receiver
        add("kickoff", down=np.nan, ydstogo=np.nan, yardline_100=35.0)
        yl, down, ytg = float(rng.choice([75, 75, 70, 80])), 1, 10

    kickoff(pos)
    while pid < n_plays:
        if down < 4:
            pt = "pass" if rng.uniform() < 0.6 else "run"
            yards = float(rng.integers(-3, 14))
            if yards >= yl:
                add(pt, yards_gained=yl, touchdown=1.0, td_team=pos)
                score[pos] += 7
                kickoff(other(pos))
                continue
            add(pt, yards_gained=yards)
            yl -= yards
            if yards >= ytg:
                down, ytg = 1, 10
            else:
                down, ytg = down + 1, ytg - yards
            yl = min(yl, 99.0)
            if yl >= 99.0:
                down, ytg = 1, 10
            continue
        # fourth down
        u = rng.uniform()
        if yl <= 38 and u < 0.85:
            made = rng.uniform() < 1 / (1 + np.exp(-(6.0 - 0.11 * (yl + 18))))
            add("field_goal", field_goal_result="made" if made else "missed", kicker_player_id=k_id[pos])
            if made:
                score[pos] += 3
                kickoff(other(pos))
            else:
                yl, pos, down, ytg = 100 - max(yl + 8, 20), other(pos), 1, 10
        elif u < 0.8 or yl > 60:
            add("punt", punter_player_id=p_id[pos])
            land = yl - float(rng.normal(42, 8))
            yl, pos, down, ytg = (80.0 if land <= 0 else float(np.clip(100 - land, 1, 99))), other(pos), 1, 10
        else:
            pt = "pass" if rng.uniform() < 0.6 else "run"
            conv = rng.uniform() < 0.5
            yards = float(ytg + rng.integers(0, 6)) if conv else float(rng.integers(-2, max(ytg, 1)))
            yards = min(yards, yl - 1) if not conv else min(yards, yl)
            add(pt, yards_gained=yards, fourth_down_converted=float(conv))
            yl -= yards
            if conv:
                down, ytg = 1, 10
            else:
                yl, pos, down, ytg = 100 - yl, other(pos), 1, 10
            yl = float(np.clip(yl, 1, 99))
    res = score[home] - score[away]
    for r in rows:
        r["result"] = float(res if res != 0 else 1)
    return rows


def synth_league(seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for S in SEASONS:
        for w in range(1, WEEKS + 1):
            order = TEAMS[w % 4:] + TEAMS[:w % 4]
            for home, away in ((order[0], order[1]), (order[2], order[3])):
                rows.extend(simulate_game(rng, S, w, home, away))
    return pd.DataFrame(rows)


def synth_m6(pbp: pd.DataFrame, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = pbp[pbp["play_type"].isin(["run", "pass"]) & (pbp["season"] >= 2016)].reset_index(drop=True)
    t = pd.DataFrame({"game_id": p["game_id"], "play_id": p["play_id"], "season": p["season"], "week": p["week"],
                      "season_type": "REG", "down": p["down"], "ydstogo": p["ydstogo"],
                      "yardline_100": p["yardline_100"], "score_diff": p["score_differential"],
                      "half_secs": p["half_seconds_remaining"], "game_secs": p["game_seconds_remaining"],
                      "off_timeouts": 3.0, "def_timeouts": 3.0, "home": (p["posteam"] == p["home_team"]).astype(float),
                      "roof_indoor": (p["roof"] == "dome").astype(float), "roof_open": 0.0, "temp": p["temp"],
                      "wind": p["wind"], "is_pass": p["pass"]})
    for c in pdata.STRUCTURE:
        t[c] = rng.integers(0, 3, len(t)).astype(float)
    for c in pdata.RATINGS:
        t[c] = rng.normal(0, 0.1, len(t))
    td = (p["touchdown"] == 1).to_numpy()
    b, yds = pdata.yards_to_bin(p["yards_gained"].to_numpy(float), td, p["yardline_100"].to_numpy(float))
    t["bin"], t["yards"] = b, yds
    t["ev_first"] = ((yds >= t["ydstogo"]) | td).astype(float)
    t["turnover"] = (rng.uniform(size=len(t)) < 0.03).astype(float)
    return t


def synth_ratings(seed=2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for S in SEASONS:
        for w in range(1, WEEKS + 1):
            for kind in ("pass", "rush"):
                rows.append(pd.DataFrame({"kind": kind, "season": S, "week": w, "team": TEAMS,
                                          "off": rng.normal(0, 0.1, 4), "def": rng.normal(0, 0.1, 4)}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def league():
    pbp = synth_league()
    rng = np.random.default_rng(3)
    a4s = pd.Series(rng.uniform(0.2, 0.8, pbp["game_id"].nunique()), index=pbp["game_id"].unique())
    return {"pbp": pbp, "a4s": a4s, "wp": wpm.state_table(pbp, a4s), "m6": synth_m6(pbp), "ratings": synth_ratings()}


# --------------------------------------------------------------------------- WP model

def test_wp_feature_list_bans_market_and_nflfastr_columns():
    for bad in ("vegas_wp", "vegas_home_wp", "spread_line", "wp", "home_wp", "away_wp", "away_moneyline",
                "total_line", "market_prob", "result", "y", "ep", "epa", "wpa"):
        assert wpm.is_banned(bad), bad
        with pytest.raises(ValueError):
            wpm.check_features(wpm.FEATURES + [bad])
    assert wpm.check_features(wpm.FEATURES) == wpm.FEATURES
    assert not any(wpm.is_banned(f) for f in wpm.FEATURES)
    assert set(wpm.MONOTONE) <= set(wpm.FEATURES)
    with pytest.raises(ValueError):
        wpm.check_features(["score_diff", "something_else"])


def test_wp_predictions_ignore_market_columns(league):
    """Scrambling vegas_wp, spread_line and nflfastR's wp in play-by-play changes nothing."""
    pbp = league["pbp"]
    t = league["wp"]
    assert not any(wpm.is_banned(c) for c in t.columns if c not in ("y",))
    m = wpm.fit_season_ahead(t, 2018, first=2012)
    a = m.predict(t[t["season"] == 2018])
    rng = np.random.default_rng(9)
    p2 = pbp.copy()
    for c in ("vegas_wp", "spread_line", "wp"):
        p2[c] = rng.permutation(p2[c].to_numpy())
    t2 = wpm.state_table(p2, league["a4s"])
    m2 = wpm.fit_season_ahead(t2, 2018, first=2012)
    assert np.array_equal(a, m2.predict(t2[t2["season"] == 2018]))
    assert m.gbm.n_features_in_ == len(wpm.FEATURES)


def test_wp_monotone_constraints_hold(league):
    m = wpm.without_calibration(wpm.fit_season_ahead(league["wp"], 2018, first=2012))
    base = league["wp"].sample(200, random_state=1).reset_index(drop=True)
    for feat, sign, grid in (("score_diff", 1, np.arange(-21, 22, 3)), ("a4s_prob", 1, np.linspace(0.05, 0.95, 10)),
                             ("yardline_100", -1, np.arange(1, 100, 7))):
        P = []
        for v in grid:
            s = base.copy()
            s[feat] = float(v)
            P.append(m.predict(wpm.add_derived(s[wpm.BASE])))
        d = np.diff(np.array(P), axis=0) * sign
        assert (d >= -1e-12).all(), feat


def test_smooth_isotonic_is_increasing_and_bounded():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, 5000)
    y = (rng.uniform(size=5000) < raw ** 1.3).astype(float)
    iso = wpm.SmoothIsotonic.fit(raw, y)
    g = np.linspace(0.01, 0.99, 200)
    v = iso(g)
    assert (np.diff(v) >= 0).all() and (np.diff(v) > 0).mean() > 0.9 and v.min() > 0 and v.max() < 1


def test_wp_season_ahead_ignores_the_target_season(league):
    t = league["wp"]
    S = 2018
    a = wpm.fit_season_ahead(t, S, first=2012).predict(t[t["season"] == S])
    t2 = t.copy()
    m = (t2["season"] == S).to_numpy()
    t2.loc[m, "y"] = 1 - t2.loc[m, "y"]                        # every season-S outcome flipped
    b = wpm.fit_season_ahead(t2, S, first=2012).predict(t2[t2["season"] == S])
    assert np.array_equal(a, b)
    # positive control: a model trained through S sees the flipped outcomes
    m1 = wpm.fit_final(t[t["season"].between(2012, S)], 30, None).predict(t[t["season"] == S])
    m2 = wpm.fit_final(t2[t2["season"].between(2012, S)], 30, None).predict(t2[t2["season"] == S])
    assert np.abs(m1 - m2).max() > 1e-3


# --------------------------------------------------------------------------- kicking

def test_player_values_use_earlier_games_only():
    ev = pd.DataFrame({"player": ["a", "a", "a", "b"], "gkey": [201801, 201802, 201803, 201801],
                       "resid": [0.2, -0.1, 0.5, 1.0]})
    q = pd.DataFrame({"player": ["a", "a", "a", "c"], "gkey": [201801, 201803, 201810, 201805]})
    v = kk.player_values(ev, q, K=2.0)
    assert np.allclose(v, [0.0, (0.2 - 0.1) / (2 + 2), (0.2 - 0.1 + 0.5) / (3 + 2), 0.0])
    ev2 = ev.copy()
    ev2.loc[ev2["gkey"] >= 201803, "resid"] = 99.0             # corrupt from game 201803 on
    assert np.allclose(kk.player_values(ev2, q.iloc[:2], 2.0), v[:2])
    # league value: same season, earlier weeks only
    lv = kk.league_values(ev, [201801, 201803, 201901], K=0.0)
    assert np.allclose(lv, [0.0, (0.2 - 0.1 + 1.0) / 3, 0.0])


def test_punt_classes_and_ordered_crps():
    assert kk.r_to_class([1, 5, 6, 80, 79, 99]).tolist() == [1, 1, 2, 17, 16, 21]
    assert kk.N_PUNT == 23 and kk.PUNT_VALUES[kk.KEEP_CLASS] == 100.0
    P = np.zeros((1, kk.N_PUNT))
    P[0, 17] = 1.0
    assert kk.crps_ordered(P, np.array([17]))[0] == 0.0
    # all mass one class too low: CRPS = the distance between the two class values
    assert np.isclose(kk.crps_ordered(P, np.array([18]))[0], kk.PUNT_VALUES[18] - kk.PUNT_VALUES[17])


def test_punt_table_reads_the_next_snap(league):
    pt = kk.punt_table(league["pbp"])
    assert len(pt) > 200 and pt["cls"].between(0, kk.N_PUNT - 1).all()
    tb = pt[pt["r"] == 80]
    assert len(tb) and (tb["cls"] == 17).all()


def _corrupt_from_week(df: pd.DataFrame, S: int, w: int, cols, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    m = ((out["season"] == S) & (out["week"] >= w)).to_numpy()
    for c in cols:
        out.loc[m, c] = rng.permutation(out.loc[m, c].to_numpy())
    return out


def test_fg_and_punt_models_are_season_ahead_and_as_of(league):
    pbp = league["pbp"]
    S, w = 2018, 5
    fg = kk.fg_table(pbp)
    fg2 = fg.copy()
    mm = ((fg2["season"] == S) & (fg2["week"] >= w)).to_numpy()
    fg2.loc[mm, "made"] = 1 - fg2.loc[mm, "made"]
    m1, x1 = kk.fit_fg_season_ahead(fg, S, 2012)
    m2, x2 = kk.fit_fg_season_ahead(fg2, S, 2012)
    sel = lambda x: x[(x["season"] == S) & (x["week"] == w)]  # noqa: E731
    assert len(sel(x1)) and np.array_equal(m1.predict(sel(x1)), m2.predict(sel(x2)))
    # positive control: a kicker value that also reads the same game moves
    ev1 = kk.kicker_events(fg, m1.base, S)
    ev2 = kk.kicker_events(fg2, m1.base, S)
    q = sel(x1)[["player", "gkey"]].assign(gkey=lambda d: d["gkey"] + 1)
    assert np.abs(kk.player_values(ev1, q, m1.K) - kk.player_values(ev2, q, m1.K)).max() > 0
    pt = kk.punt_table(pbp)
    pt2 = _corrupt_from_week(pt, S, w, ["cls", "r", "keep_yl"])
    p1, y1 = kk.fit_punt_season_ahead(pt, S, 2012)
    p2, y2 = kk.fit_punt_season_ahead(pt2, S, 2012)
    assert len(sel(y1)) and np.array_equal(p1.predict(sel(y1)), p2.predict(sel(y2)))


def test_kickoff_start_is_as_of_the_week():
    """This season's running mean from earlier weeks; the previous season's mean in week 1 or with few kickoffs."""
    rows = [(2017, 201700 + w, 75.0) for w in range(1, 18) for _ in range(10)]
    rows += [(2018, 201800 + w, 70.0 if w < 5 else 60.0) for w in range(1, 18) for _ in range(10)]
    ko = pd.DataFrame(rows, columns=["season", "gkey", "start"])
    v = kk.kickoff_asof(ko, [201801, 201805, 201806, 201807], min_n=50)
    assert np.allclose(v, [75.0, 75.0, (40 * 70 + 10 * 60) / 50, (40 * 70 + 20 * 60) / 60])  # week 6: 50 earlier
    ko2 = ko.copy()
    ko2.loc[ko2["gkey"] >= 201806, "start"] = 5.0                         # corrupt from week 6 on
    assert np.allclose(kk.kickoff_asof(ko2, [201801, 201805, 201806], min_n=50), v[:3])
    assert not np.allclose(kk.kickoff_asof(ko2, [201807], min_n=50), v[3:])   # control: week 7 sees week 6


def test_durations_and_kickoff_spot(league):
    d = kk.durations(league["pbp"], [2016, 2017])
    assert set(d) == set(kk.DURATION_KEYS) and all(np.isfinite(v) and v >= 0 for v in d.values())
    assert 60 < kk.kickoff_spot(league["pbp"], 2017) < 90
    t = kk.kickoff_table(league["pbp"])
    v = kk.kickoff_asof(t, [201801, 201808], min_n=5)
    assert np.isclose(v[0], t.loc[t["season"] == 2017, "start"].mean())
    assert np.isclose(v[1], t.loc[(t["season"] == 2018) & (t["gkey"] < 201808), "start"].mean())


# --------------------------------------------------------------------------- fourth downs

@pytest.fixture(scope="module")
def fitted(league):
    comp, tables = fd.fit_components(2018, league["pbp"], league["wp"], league["m6"], first_wp=2012,
                                     m6_grid=GRID1)
    dec = fd.decision_table(league["pbp"], league["a4s"], seasons=[2018])
    return comp, tables, dec


def test_decision_table_and_option_values(fitted):
    comp, tables, dec = fitted
    assert set(dec["choice"]) <= set(fd.OPTIONS) and len(dec) > 50
    assert dec["kicker"].notna().mean() > 0.5 and dec["punter"].notna().mean() > 0.5
    out, boot = fd.value_season(comp, dec, synth_ratings(), tables, B=2)
    for c in ("wp_go", "wp_fg", "wp_punt", "p_conv", "p_fg"):
        assert out[c].between(0, 1).all(), c
    assert (out["lost"] >= -1e-12).all() and boot.shape == (2, len(dec), 3)
    assert set(out["best"]) <= set(fd.OPTIONS) and out["tossup"].dtype == bool
    far = out["yardline_100"] + kk.FG_DIST_OFFSET > kk.MAX_FG_DIST
    assert (out.loc[far, "p_fg"] == 0).all()


def test_bands_mark_tossups():
    rec = pd.DataFrame({"best": ["go", "punt"], "second": ["punt", "fg"]})
    boot = np.zeros((100, 2, 3))
    rng = np.random.default_rng(0)
    boot[:, 0, 0] = 0.50 + rng.normal(0, 0.02, 100)            # go vs punt: overlapping
    boot[:, 0, 2] = 0.49 + rng.normal(0, 0.02, 100)
    boot[:, 1, 2] = 0.60                                       # punt clearly over fg
    boot[:, 1, 1] = 0.40
    b = fd.bands(None, rec, boot)
    assert bool(b["tossup"].iloc[0]) and not bool(b["tossup"].iloc[1])


def test_fourth_down_valuations_ignore_season_from_the_week(league, fitted):
    """Corrupt season S from week w on (outcomes of weeks >= w, states of weeks > w): week-w valuations unchanged."""
    import ml_m7a
    comp, _, dec = fitted
    S, w = 2018, 5
    rat = synth_ratings()
    v0, _ = fd.value_season(comp, dec[dec["week"] == w].reset_index(drop=True), rat)
    pbp_c = ml_m7a.corrupt_m7a(league["pbp"], S, w, seed=1)
    wp_c = wpm.state_table(pbp_c, league["a4s"])
    m6_c = _corrupt_from_week(league["m6"], S, w, ["bin", "yards", "ev_first"])
    comp_c, _ = fd.fit_components(S, pbp_c, wp_c, m6_c, first_wp=2012, m6_grid=GRID1)
    d1 = fd.decision_table(pbp_c, league["a4s"], seasons=[S])
    d1 = d1[d1["week"] == w].reset_index(drop=True)
    v1, _ = fd.value_season(comp_c, d1, rat)
    assert v0[["game_id", "play_id"]].equals(v1[["game_id", "play_id"]])
    cols = ["wp_go", "wp_fg", "wp_punt", "p_conv", "p_fg", "kval", "pval", "ko_start"]
    assert np.array_equal(v0[cols].to_numpy(float), v1[cols].to_numpy(float))
    assert (pbp_c.loc[pbp_c["season"] == S, "result"] != league["pbp"].loc[league["pbp"]["season"] == S,
                                                                            "result"]).mean() > 0.3
    # positive control: a WP model trained through S on the corrupted outcomes moves the valuations
    wp_pc = wpm.fit_final(wp_c[wp_c["season"].between(2012, S)], comp_c.wp.info["n_iter"], None)
    v_pc = fd.option_values(comp_c, d1, fd.go_inputs(d1, comp_c, rat), wp=wp_pc)
    assert np.abs(v_pc["wp_go"].to_numpy() - v1["wp_go"].to_numpy()).max() > 1e-6


def test_m7a_dev_window_is_guarded(tmp_path):
    assert windows.M7A_DEV == (2016, 2019) and not windows.touches_holdout(windows.M7A_DEV)
    with pytest.raises(windows.HoldoutError):
        registry.log_run("x", label="m7adev", seasons=(2019, 2020), game_ids=["a"], metrics={"n": 1},
                         data={}, runs_dir=tmp_path)


def test_signoff_refuses_without_a_reproduced_dry_run(tmp_path, monkeypatch):
    import ml_m7a
    monkeypatch.setattr(ml_m7a, "OUT", tmp_path)
    # isolate from the real registry (a recorded holdout sign-off would trip the run-once check first)
    monkeypatch.setattr(ml_m7a, "holdout_signoff_runs", lambda: [])
    with pytest.raises(SystemExit):
        ml_m7a.stage_signoff(ml_m7a.HOLDOUT, allow_holdout=True, log_=False)     # no dev.json
    (tmp_path / "dev.json").write_text('{"boot": 2, "primary": {}}')
    with pytest.raises(SystemExit, match="dry run"):
        ml_m7a.stage_signoff(ml_m7a.HOLDOUT, allow_holdout=True, log_=False)
    (tmp_path / "signoff_m7adev.json").write_text('{"reproduced": true, "git": {"sha": "abc", "dirty": true}}')
    with pytest.raises(SystemExit, match="clean tree"):
        ml_m7a.stage_signoff(ml_m7a.HOLDOUT, allow_holdout=True, log_=False)
    monkeypatch.setattr(ml_m7a, "holdout_signoff_runs", lambda: ["x_m7a_signoff_wp.json"])
    with pytest.raises(SystemExit, match="run once only"):
        ml_m7a.stage_signoff(ml_m7a.HOLDOUT, allow_holdout=True, log_=False)
