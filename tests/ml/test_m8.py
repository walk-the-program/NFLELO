"""M8: QB v2, early-season and context features, margin-target models and stacking (offline, synthetic frames).

- A4s is unchanged: QB v2 switched off is the M3 code path, and carrying `defteam` changes no M3 value.
- Every new builder passes `asof.leakage_check`; weather (filled after the game) and the filter's post-week
  margin are controls that must fail.
- Partial QB-games (share under 50%, or a starter who left early) are flagged as specified.
- Margin-target and stack walk-forwards never read the season they predict.
- stadiums.csv is well formed and covers every venue in the cached schedules.
"""
import zoneinfo

import numpy as np
import pandas as pd
import pytest

from nflelo.ml import asof
from nflelo.ml import data as mldata
from nflelo.ml.features import context, kalman as kf, qb
from nflelo.ml.models import margin_target as mt, stack

QCFG = qb.QBConfig(k=50.0, half_life=4.0, rho=0.5)
V2 = {"a": qb.QBv2(lam_d=100.0), "b": qb.QBv2(exit_weight=0.0), "ab": qb.QBv2(lam_d=100.0, exit_weight=0.25)}
HOME_STADIUM = {"KC": "KAN00", "BUF": "BUF00", "NE": "BOS00", "MIA": "MIA00", "DAL": "DAL00", "PHI": "PHI00"}


@pytest.fixture(scope="module")
def sched(synth_sched):
    s = synth_sched.copy()
    s["stadium_id"] = s["home_team"].map(HOME_STADIUM)
    s.loc[s["location"] == "Neutral", "stadium_id"] = "LON00"
    s["surface"] = np.where(s["home_team"].isin(["DAL", "MIA"]), "fieldturf", "grass")
    rng = np.random.default_rng(31)
    s["temp"] = rng.integers(20, 90, len(s)).astype(float)
    s["wind"] = rng.integers(0, 25, len(s)).astype(float)
    for side in ("home", "away"):
        coach = s[f"{side}_team"] + " coach"
        s[f"{side}_coach"] = coach.where(~((s[f"{side}_team"] == "NE") & (s["season"] == 2023)), "NE new coach")
    return asof.add_asof(s)


def _targets(sched):
    out = []
    for season, week, slot in ((2023, 1, 0), (2023, 1, 1), (2023, 3, 0), (2023, 4, 2), (2022, 5, 2)):
        rows = sched[(sched["season"] == season) & (sched["week"] == week)].sort_values("kickoff")
        out.append(rows["game_id"].iloc[slot])
    out.append(sched.loc[(sched["season"] == 2023) & (sched["game_type"] == "WC"), "game_id"].iloc[0])
    return out


# --------------------------------------------------------------------------- A4s unchanged

@pytest.mark.parametrize("starter", ["actual", "last"])
def test_qb_v2_off_is_byte_identical(synth_pbp, sched, starter):
    games = sched[sched["season"] == 2023]
    m3 = qb.build_features(synth_pbp, sched, games, QCFG, starter)
    assert qb.build_features_v2(synth_pbp, sched, games, QCFG, starter, qb.QBv2()).equals(m3)
    assert qb.build_features_v2(synth_pbp, sched, games, QCFG, starter, None).equals(m3)
    # the defteam column dropback_plays now carries is ignored by the M3 aggregation
    db = qb.dropback_plays(synth_pbp, sched, QCFG)
    assert "defteam" in db.columns
    assert qb.qb_game_table(db, sched).equals(qb.qb_game_table(db.drop(columns="defteam"), sched))


def test_v2_states_with_nothing_on_equal_m3_values(synth_pbp, sched):
    keys = [(2023, 3), (2023, 5)]
    st = qb.v2_states(synth_pbp, sched, sched, QCFG, qb.QBv2(), keys=keys)
    db = qb.dropback_plays(synth_pbp, sched, QCFG)
    priors = qb.replacement_priors(db, {2023}, QCFG.rookie_dropbacks)
    base, qbs = qb.week_states(qb.qb_game_table(db, sched), sched, keys, QCFG, priors)
    for k in keys:
        a = pd.Series(st["states"][k].values(QCFG.k), index=st["qbs"]).sort_index()
        b = pd.Series(base[k].values(QCFG.k), index=qbs).sort_index()
        assert np.array_equal(a.to_numpy(), b.to_numpy())


def test_v2_changes_values(synth_pbp, sched):
    keys = [(2023, 5)]
    v0 = qb.v2_states(synth_pbp, sched, sched, QCFG, qb.QBv2(), keys=keys)["states"][(2023, 5)].vals
    for v in V2.values():
        st = qb.v2_states(synth_pbp, sched, sched, QCFG, v, keys=keys)["states"][(2023, 5)]
        assert not np.allclose(st.vals, v0)
    st = qb.v2_states(synth_pbp, sched, sched, QCFG, V2["a"], keys=keys)["states"][(2023, 5)]
    assert st.d_eff and all(np.isfinite(list(st.d_eff.values())))


def test_partial_flags_toy():
    sched = pd.DataFrame({"game_id": ["2023_01_B_A"], "season": [2023], "game_type": ["REG"], "week": [1],
                          "gameday": ["2023-09-10"], "gametime": ["13:00"], "home_team": ["KC"], "away_team": ["BUF"],
                          "location": ["Home"], "home_qb_id": ["S"], "away_qb_id": ["T"]})
    sched = asof.add_asof(sched)
    db = pd.DataFrame({"game_id": "2023_01_B_A", "qb": ["S"] * 10 + ["R"] * 6 + ["T"] * 12,
                       "team": ["KC"] * 16 + ["BUF"] * 12, "defteam": ["BUF"] * 16 + ["KC"] * 12,
                       "epa": 0.1, "order": list(range(1, 17)) + list(range(1, 13)), "season": 2023, "week": 1})
    db["kick_ns"] = 0
    g = qb.qb_game_table_v2(db, sched).set_index("qb")
    assert g.loc["S", "share"] == pytest.approx(10 / 16) and g.loc["S", "after"] == 6 and g.loc["S", "is_starter"]
    flags = pd.Series(qb.partial_flags(g.reset_index(), qb.QBv2(exit_weight=0.0)), index=g.index)
    assert flags["S"] and flags["R"] and not flags["T"]  # S left early, R under 50%, T played it all
    assert g.loc["S", "home"] == 1.0 and g.loc["T", "home"] == 0.0


# --------------------------------------------------------------------------- leakage

@pytest.mark.parametrize("v", list(V2))
@pytest.mark.parametrize("starter,scramble", [("actual", False), ("last", True)])
def test_qb_v2_is_leak_free(synth_pbp, sched, v, starter, scramble):
    r = asof.leakage_check(lambda p, s, g: qb.build_features_v2(p, s, g, QCFG, starter, V2[v]),
                           synth_pbp, sched, _targets(sched), scramble_starters=scramble)
    assert r["leak_free"].all(), r


def test_context_and_early_features_are_leak_free(synth_pbp, sched):
    for b in (lambda p, s, g: context.m8_context_features(p, s, g),
              lambda p, s, g: context.m8_early_features(p, s, g),
              lambda p, s, g: kf.build_features(p, s, g, kf.TUNED_CARRYOVER)):
        r = asof.leakage_check(b, synth_pbp, sched, _targets(sched))
        assert r["leak_free"].all(), r


def test_weather_is_flagged_as_a_leak(synth_pbp, sched):
    def b(p, s, g):
        return pd.DataFrame({"temp": pd.to_numeric(g["temp"]).to_numpy(), "wind": pd.to_numeric(g["wind"]).to_numpy()},
                            index=g["game_id"].to_numpy())
    r = asof.leakage_check(b, synth_pbp, sched, _targets(sched))
    assert not r["leak_free"].any()


# --------------------------------------------------------------------------- context values

def test_context_values(sched):
    f = context.m8_context_features(None, sched)
    g = sched.set_index("game_id")
    home = g["location"] != "Neutral"
    # a home team at its own stadium travels 0 km; Arlington TX to Orchard Park NY is about 1,950 km
    dal_at_buf = g.index[(g["home_team"] == "BUF") & (g["away_team"] == "DAL")][0]
    assert f.loc[dal_at_buf, "travel_diff"] == pytest.approx(-1.95, abs=0.05)
    assert (f.loc[home[home].index, "travel_diff"] <= 0).all()
    neutral = g.index[~home][0]
    assert f.loc[neutral, "tz_diff"] in (-1.0, 0.0, 1.0)  # both teams cross the Atlantic: shifts nearly cancel
    assert set(f["indoor"].unique()) <= {0.0, 1.0}
    assert f.loc[g.index[g["home_team"] == "DAL"], "indoor"].eq(1.0).all()  # AT&T Stadium is retractable
    e = context.m8_early_features(None, sched)
    ne23 = g.index[(g["season"] == 2023) & (g["home_team"] == "NE") & (g["game_type"] == "REG")]
    assert (e.loc[ne23, "coach_new_diff"] == 1.0).all()
    assert (e.loc[g.index[g["season"] == 2022], ["qb_new_diff", "coach_new_diff"]] == 0).all().all()
    kc_backup = g.index[(g["season"] == 2023) & (g["home_team"] == "KC") & g["week"].isin([3, 4])]
    assert (e.loc[kc_backup, "qb_new_diff"] == 1.0).all()


def test_team_base_follows_relocation(sched):
    s = sched.copy()
    s.loc[(s["home_team"] == "KC") & (s["season"] == 2023), "stadium_id"] = "DEN00"
    b = context.team_bases(s)
    assert b.loc[("KC", 2022), "base"] == "KAN00" and b.loc[("KC", 2023), "base"] == "DEN00"


def test_stadiums_csv_is_well_formed():
    s = context.load_stadiums()
    assert s.index.is_unique and len(s) >= 67
    assert s["lat"].between(-60, 70).all() and s["lon"].between(-180, 180).all()
    assert s["roof_type"].isin(["outdoor", "dome", "retractable"]).all()
    for z in s["tz"]:
        zoneinfo.ZoneInfo(z)
    assert s["source"].str.len().gt(0).all()
    assert (s.loc[["DEN00", "MEX00"], "elevation_m"] >= context.ALT_THRESHOLD_M).all()
    assert (s.drop(index=["DEN00", "DEN99", "MEX00"])["elevation_m"] < context.ALT_THRESHOLD_M).all()


def test_stadiums_cover_cached_schedules():
    try:
        sched = mldata.load_schedules(range(1999, 2026))
    except Exception:  # noqa: BLE001 - no cache on this machine
        pytest.skip("no cached schedules")
    assert set(sched["stadium_id"].dropna()) <= set(context.load_stadiums().index)


# --------------------------------------------------------------------------- models

def _toy_frame(seed=3):
    rng = np.random.default_rng(seed)
    n = 1200
    season = np.repeat(np.arange(2001, 2007), n // 6)
    x = rng.normal(0, 1, (n, 2))
    margin = np.round(6 * x[:, 0] + 2 * x[:, 1] + 2 + rng.normal(0, 13, n))
    y = np.where(margin > 0, 1.0, np.where(margin == 0, 0.5, 0.0))
    return pd.DataFrame({"season": season, "x1": x[:, 0], "x2": x[:, 1], "margin": margin, "y": y,
                         "epa": margin * 0.8 + rng.normal(0, 5, n)})


def test_margin_target_is_season_ahead():
    f = _toy_frame()
    for target in ("margin", "epa"):
        for method in mt.METHODS:
            p, c = mt.walk_forward(f, ["x1", "x2"], (2004, 2006), target, method)
            g = f.copy()
            last = g["season"] == 2006
            g.loc[last, ["margin", "epa"]] = -g.loc[last, ["margin", "epa"]]
            g.loc[last, "y"] = 1 - g.loc[last, "y"]
            p2, _ = mt.walk_forward(g, ["x1", "x2"], (2004, 2006), target, method)
            assert np.allclose(p[last], p2[last]) and p[last].between(0, 1).all()
            assert p[f["season"] < 2004].isna().all()
    fit = mt.fit(f[["x1", "x2"]], f["margin"], f["margin"], f["y"], np.ones(len(f)))
    assert fit.conv_a == pytest.approx(0, abs=1e-9) and fit.conv_b == pytest.approx(1, abs=1e-9)
    assert 11 < fit.sigma < 15 and fit.beta[1] > 0


def test_stack_weights_nonnegative_and_season_ahead():
    f = _toy_frame()
    rng = np.random.default_rng(9)
    z = 0.4 * f["x1"] + 0.13 * f["x2"]
    base = pd.DataFrame({"good": 1 / (1 + np.exp(-z)), "noise": 1 / (1 + np.exp(-rng.normal(0, 1, len(f)))),
                         "twin": 1 / (1 + np.exp(-z))})
    p, w = stack.walk_forward(base, f["y"].to_numpy(), f["season"].to_numpy(), (2004, 2006), first_season=2001)
    assert (w[["good", "noise", "twin"]] >= 0).all().all()
    assert (w["noise"] < w["good"]).all()
    assert np.allclose(w["good"], w["twin"], atol=1e-4)  # identical bases split the weight (ridge to equal)
    y2 = f["y"].to_numpy().copy()
    last = (f["season"] == 2006).to_numpy()
    y2[last] = 1 - y2[last]
    p2, _ = stack.walk_forward(base, y2, f["season"].to_numpy(), (2004, 2006), first_season=2001)
    assert np.allclose(p[last], p2[last])


def test_carryover_config_changes_only_carryover():
    a, b = kf.TUNED.to_dict(), kf.TUNED_CARRYOVER.to_dict()
    assert {k for k in a if a[k] != b[k]} <= set(kf.CARRYOVER_KEYS)
