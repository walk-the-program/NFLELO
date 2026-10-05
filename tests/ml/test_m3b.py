"""M3b: the Kalman team-strength filter and the QB-value extras (offline, synthetic frames from conftest.py).

- The new builders pass `asof.leakage_check`; the filter's post-week margin (a
  deliberate leak) fails it.
- With no extras, the QB builder is byte-identical to M3's (A4s unchanged).
- Kalman sanity on a two-team toy: a win moves the two teams apart by equal
  amounts, uncertainty shrinks, and the EPA measurement adds information.
"""
import numpy as np
import pandas as pd
import pytest

from nflelo.ml import asof
from nflelo.ml.features import kalman as kf
from nflelo.ml.features import qb

QCFG = qb.QBConfig(k=50.0, half_life=4.0, rho=0.5)
KCFG = kf.KalmanConfig(q_week=0.5, gamma=0.6, q_season=10.0, sigma_m=13.0, sigma_e=12.0, rho=0.7, scale=40.0,
                       q_hfa=0.01, epa_home=0.01)
EXTRAS = {"abc": qb.QBExtras(cpoe_weight=0.01, cpoe_k=100.0, experience=(-0.05, 0.0, 0.05, 0.03),
                             aging=(30.0, 0.05, -0.02)),
          "d": qb.QBExtras(experience=(-0.06, 0.08, 0.03, 0.03))}


@pytest.fixture(scope="module")
def sched(synth_sched):
    return asof.add_asof(synth_sched)


@pytest.fixture(scope="module")
def pbp(synth_pbp):
    p = synth_pbp.copy()
    rng = np.random.default_rng(77)
    att = (p["play_type"] == "pass").to_numpy() & (rng.uniform(size=len(p)) < 0.9)
    p["cpoe"] = np.where(att, rng.normal(0, 40, len(p)), np.nan)
    return p


@pytest.fixture(scope="module")
def births(sched):
    ids = pd.unique(pd.concat([sched["home_qb_id"], sched["away_qb_id"]]))
    ids = list(ids) + [t + "_QB2" for t in ["KC", "BUF", "NE", "MIA", "DAL", "PHI"]]
    rng = np.random.default_rng(5)
    return pd.Series(pd.to_datetime("1990-01-01") + pd.to_timedelta(rng.integers(0, 4000, len(ids)), unit="D"),
                     index=pd.Index(ids).astype(str)).loc[lambda s: ~s.index.duplicated()]


def _targets(sched):
    out = []
    for season, week, slot in ((2023, 1, 0), (2023, 1, 1), (2023, 3, 0), (2023, 4, 2), (2022, 5, 2)):
        rows = sched[(sched["season"] == season) & (sched["week"] == week)].sort_values("kickoff")
        out.append(rows["game_id"].iloc[slot])
    out.append(sched.loc[(sched["season"] == 2023) & (sched["game_type"] == "WC"), "game_id"].iloc[0])
    return out


# --------------------------------------------------------------------------- leakage

def test_kalman_is_leak_free(pbp, sched):
    r = asof.leakage_check(lambda p, s, g: kf.build_features(p, s, g, KCFG), pbp, sched, _targets(sched))
    assert r["leak_free"].all(), r


def test_kalman_positive_control_fails(pbp, sched):
    r = asof.leakage_check(lambda p, s, g: kf.build_features(p, s, g, KCFG, positive_control=True),
                           pbp, sched, _targets(sched))
    assert not r["leak_free"].any()


@pytest.mark.parametrize("variant", ["abc", "d"])
@pytest.mark.parametrize("starter,scramble", [("actual", False), ("last", True)])
def test_qb_extras_are_leak_free(pbp, sched, births, variant, starter, scramble):
    ex = EXTRAS[variant]
    r = asof.leakage_check(lambda p, s, g: qb.build_features_ext(p, s, g, QCFG, starter, ex, births),
                           pbp, sched, _targets(sched), scramble_starters=scramble)
    assert r["leak_free"].all(), r


def test_corrupt_scrambles_cpoe(pbp, sched):
    gid = _targets(sched)[2]
    as_of = sched.loc[sched["game_id"] == gid, "as_of"].iloc[0]
    p2, _ = asof.corrupt_from(pbp, sched, as_of, seed=0)
    late = pbp["game_id"].isin(set(sched.loc[sched["kickoff"] >= as_of, "game_id"])).to_numpy()
    a, b = pbp.loc[late, "cpoe"].to_numpy(float), p2.loc[late, "cpoe"].to_numpy(float)
    assert not np.allclose(np.nan_to_num(a), np.nan_to_num(b))
    assert np.array_equal(np.nan_to_num(pbp.loc[~late, "cpoe"].to_numpy(float)),
                          np.nan_to_num(p2.loc[~late, "cpoe"].to_numpy(float)))


# --------------------------------------------------------------------------- A4s unchanged

@pytest.mark.parametrize("starter", ["actual", "last"])
def test_qb_defaults_are_byte_identical(pbp, synth_pbp, sched, starter):
    games = sched[sched["season"] == 2023]
    m3 = qb.build_features(synth_pbp, sched, games, QCFG, starter)
    for extras in (None, qb.QBExtras()):
        ext = qb.build_features_ext(pbp, sched, games, QCFG, starter, extras)   # pbp carries cpoe here
        assert ext.equals(m3)


def test_inactive_extras_reproduce_m3_values(pbp, sched):
    keys = [(2023, 3), (2023, 5)]
    st = qb.ext_states(pbp, sched, sched, QCFG, keys=keys)
    db = qb.dropback_plays(pbp, sched, QCFG)
    priors = qb.replacement_priors(db, {2023}, QCFG.rookie_dropbacks)
    base, qbs = qb.week_states(qb.qb_game_table(db, sched), sched, keys, QCFG, priors)
    for k in keys:
        assert np.array_equal(st["states"][k].values(QCFG.k, qb.QBExtras()), base[k].values(QCFG.k))


def test_extras_change_values(pbp, sched, births):
    keys = [(2023, 5)]
    st = qb.ext_states(pbp, sched, sched, QCFG, births, keys=keys)["states"][(2023, 5)]
    v0 = st.values(QCFG.k, qb.QBExtras())
    for ex in (qb.QBExtras(cpoe_weight=0.01), qb.QBExtras(experience=(0.0, 0.1, 0.1, 0.1)),
               qb.QBExtras(aging=(25.0, 0.0, -0.05))):
        assert not np.allclose(st.values(QCFG.k, ex), v0)
    assert np.allclose(st.values(QCFG.k, qb.QBExtras(aging=(30.0, 0.0, 0.0))), v0)
    assert (st.career >= st.base.n - 1e-9).all()  # unweighted career count >= decayed count


def test_birth_dates_parse():
    players = pd.DataFrame({"gsis_id": ["a", "b", "c"], "birth_date": ["1990-05-01", None, "bad"]})
    b = qb.birth_dates(players)
    assert list(b.index) == ["a"]


# --------------------------------------------------------------------------- Kalman sanity

def _toy(margins, epa=None, neutral=False):
    """Two teams, one game per week in one season: KC hosts BUF every week."""
    n = len(margins)
    return pd.DataFrame({"game_id": [f"g{i}" for i in range(n)], "season": 2022, "week": np.arange(1, n + 1),
                         "home": "KC", "away": "BUF", "neutral": neutral, "margin": np.asarray(margins, float),
                         "kick_ns": np.arange(n) * 7, "ord": np.arange(1, n + 1),
                         "epa_margin": np.full(n, np.nan) if epa is None else np.asarray(epa, float)})


def test_toy_update_is_symmetric_and_shrinks_uncertainty():
    cfg = kf.KalmanConfig(q_week=0.0, q_hfa=0.0, h0=0.0, ph0=1e-9, sigma_m=13.0)
    res = kf.run_filter(_toy([14.0, 0.0]), cfg, snapshots=True)
    i, j = kf.TEAMS.index("KC"), kf.TEAMS.index("BUF")
    x1, d1 = res.snapshots[1][2], res.snapshots[1][3]
    assert x1[i] > 0 and np.isclose(x1[i], -x1[j])
    gain = 2 * cfg.p0 / (2 * cfg.p0 + cfg.sigma_m ** 2)
    assert np.isclose(x1[i] - x1[j], gain * 14.0)
    assert d1[i] < cfg.p0
    assert np.isclose(res.pred_mean[0], 0.0) and np.isclose(res.pred_mean[1], gain * 14.0)


def test_toy_epa_measurement_adds_information():
    cfg = kf.KalmanConfig(q_week=0.0, q_hfa=0.0, ph0=1e-9, sigma_m=13.0, sigma_e=10.0, rho=0.5, scale=40.0,
                          epa_home=0.0)
    without = kf.run_filter(_toy([10.0, 0.0]), cfg)
    with_e = kf.run_filter(_toy([10.0, 0.0], epa=[0.25, 0.0]), cfg)
    assert with_e.pred_var[1] < without.pred_var[1]


def test_season_transition_reverts_toward_zero():
    cfg = kf.KalmanConfig(q_week=0.0, q_hfa=0.0, ph0=1e-9, gamma=0.5)
    first = kf.run_filter(_toy([20.0]), cfg)
    t = pd.concat([_toy([20.0]), _toy([0.0]).assign(season=2023, game_id="h0", ord=2)], ignore_index=True)
    res = kf.run_filter(t, cfg, snapshots=True)
    assert np.allclose(res.snapshots[1][2][:-1], 0.5 * first.x_final[:-1])
    assert np.isclose(res.pred_mean[1], 0.5 * (first.x_final[kf.TEAMS.index("KC")]
                                               - first.x_final[kf.TEAMS.index("BUF")]) + first.x_final[-1])


def test_likelihoods_and_win_prob(pbp, sched):
    gt = kf.game_table(pbp, sched, KCFG)
    res = kf.run_filter(gt, KCFG)
    ll, n = kf.log_likelihood(gt, res, KCFG, (2022, 2023))
    jl, _ = kf.joint_log_likelihood(gt, res, KCFG, (2022, 2023))
    assert n == len(gt) and np.isfinite(ll) and np.isfinite(jl)
    assert np.isclose(kf.win_prob(0.0, 2.0, 13.0), 0.5)
    assert 0.5 < kf.win_prob(3.0, 2.0, 13.0) < 1.0
    assert (res.pred_var > 0).all()


def test_unplayed_games_get_a_prediction(pbp, sched):
    s = sched.copy()
    last = s[(s["season"] == 2023) & (s["week"] == 6)]["game_id"]
    s.loc[s["game_id"].isin(last), ["home_score", "away_score"]] = np.nan
    f = kf.build_features(pbp, s, s[s["game_id"].isin(last)], KCFG)
    full = kf.build_features(pbp, sched, sched[sched["game_id"].isin(last)], KCFG)
    assert np.allclose(f["kf_margin"], full["kf_margin"])
