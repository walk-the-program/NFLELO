"""Margin model (M4 Phase 2): leakage, walk-forward discipline, the market ban, the shapes, and the live wiring.

The live win probability stays A4s for the 2026 season (Walker's decision): the
margin model adds `spread_model` and must not move `p_home_model` by a single bit.
"""
import json
import sys

import numpy as np
import pandas as pd
import pytest

from nflelo import config
from nflelo.ml import asof, live, live_features
from nflelo.ml.data import is_market_column
from nflelo.ml.eval import metrics
from nflelo.ml.features import context, opponent_adjust as oa, qb
from nflelo.ml.models import margin as mm

sys.path.insert(0, str(config.ROOT / "scripts"))

CFG = oa.RatingConfig(lam=50.0, half_life=4.0, rho=0.5)
QCFG = qb.QBConfig(k=50.0, half_life=4.0, rho=0.5)
FIT = mm.MarginFit(mm.FEATURES, 0.4, {"elo_logit": 6.0, "adj_epa_margin": 9.0, "qb_delta_diff": 27.0}, 13.5)


# --------------------------------------------------------------------------- leakage and walk-forward

def margin_builder(p, s, g):
    """The margin model's spread for each game, from the same three feature builders as A4s."""
    adj = oa.build_features(p, s, g, cfg=CFG)
    q = qb.build_features(p, s, g, cfg=QCFG, starter="actual")
    elo = context.build_elo_features(p, s, g)
    f = pd.DataFrame({"elo_logit": elo["elo_logit"], "adj_epa_margin": adj["adj_epa_margin"],
                      "qb_delta_diff": q["qb_delta_home"] - q["qb_delta_away"]})
    return pd.DataFrame({"spread": mm.spread(FIT, f)}, index=f.index)


def test_margin_spread_is_leak_free(synth_pbp, synth_sched):
    s = asof.add_asof(synth_sched)
    reg = s[s["game_type"] == "REG"].sort_values("kickoff")
    targets = [reg[(reg["season"] == 2023) & (reg["week"] == w)]["game_id"].iloc[k] for w, k in ((1, 0), (3, 1), (5, 2))]
    res = asof.leakage_check(margin_builder, synth_pbp, s, targets)
    assert res["leak_free"].all(), res


def _frame(seed=0, seasons=range(2001, 2020), per=256):
    rng = np.random.default_rng(seed)
    s = np.repeat(np.arange(seasons.start, seasons.stop), per)
    X = rng.normal(0, [0.4, 0.15, 0.04], (len(s), 3))
    margin = np.rint(0.5 + X @ np.array([6.0, 9.0, 27.0]) + rng.normal(0, 13.5, len(s)))
    df = pd.DataFrame(X, columns=mm.FEATURES)
    df["season"], df["margin"] = s, margin
    return df


def test_walk_forward_never_sees_the_test_season_or_later():
    df = _frame()
    a, _ = mm.walk_forward(df, mm.FEATURES, (2010, 2012))
    df2 = df.copy()
    df2.loc[df2["season"] >= 2012, "margin"] *= -1
    b, _ = mm.walk_forward(df2, mm.FEATURES, (2010, 2012))
    m = df["season"] == 2012
    np.testing.assert_allclose(a.loc[m, "mu"], b.loc[m, "mu"])
    np.testing.assert_allclose(a.loc[m, "sigma"], b.loc[m, "sigma"])
    df3 = df.copy()
    df3.loc[df3["season"] == 2011, "margin"] *= -1
    c, _ = mm.walk_forward(df3, mm.FEATURES, (2010, 2012))
    assert not np.allclose(a.loc[m, "mu"], c.loc[m, "mu"])


def test_fit_recovers_the_line_and_sigma():
    df = _frame(per=4000, seasons=range(2001, 2004))
    f = mm.fit_margin(df[mm.FEATURES], df["margin"], np.ones(len(df)), mm.FEATURES)
    assert f.coef["elo_logit"] == pytest.approx(6.0, abs=0.5) and f.sigma == pytest.approx(13.5, abs=0.3)


def test_market_columns_are_banned():
    assert not [f for f in mm.FEATURES if is_market_column(f)]
    df = _frame()
    df["spread_line"] = df["margin"] + 1
    with pytest.raises(ValueError, match="market"):
        mm.fit_margin(df[["elo_logit", "spread_line"]], df["margin"], np.ones(len(df)), ["elo_logit", "spread_line"])
    with pytest.raises(ValueError, match="market"):
        mm.walk_forward(df, ["elo_logit", "spread_line"], (2010, 2010))
    with pytest.raises(ValueError, match="market"):
        mm.spread(mm.MarginFit(["spread_line"], 0.0, {"spread_line": 1.0}, 13.0), df)


def test_keynumbers_fit_reads_only_2001_2005():
    df = _frame()
    a = mm.keynumbers_from_frame(df)
    df2 = df.copy()
    df2.loc[df2["season"] > 2005, "margin"] = 3.0                 # wreck every later season
    b = mm.keynumbers_from_frame(df2)
    np.testing.assert_allclose(a.r, b.r)


# --------------------------------------------------------------------------- shapes

def test_pmfs_sum_to_one_and_win_prob_is_symmetric():
    kn = mm.KeyNumbers(np.where(np.arange(mm.KMAX + 1) == 3, 2.5, 1.0))
    for p in (mm.normal_pmf([0.0, 3.0, -7.5], 13.5), mm.keynum_pmf([0.0, 3.0, -7.5], 13.5, kn)):
        np.testing.assert_allclose(p.sum(axis=1), 1.0)
        assert mm.win_prob(p)[0] == pytest.approx(0.5)
        assert mm.win_prob(p)[1] > 0.5 > mm.win_prob(p)[2]
    p = mm.keynum_pmf([0.0], 13.5, kn)[0]
    q = mm.normal_pmf([0.0], 13.5)[0]
    k3 = mm.GRID == 3
    assert p[k3] / q[k3] > 2.0                                    # 3 is boosted against the bell curve


def test_ipf_matches_observed_counts():
    rng = np.random.default_rng(1)
    mu = rng.normal(0, 5, 3000)
    y = np.rint(mu + rng.normal(0, 13.5, 3000))
    y[rng.uniform(size=3000) < 0.1] = 3.0                         # extra mass on a key number
    kn = mm.fit_keynumbers(y, mu, 13.5)
    p = mm.keynum_pmf(mu, 13.5, kn)
    a = np.abs(mm.GRID)
    for k in (0, 3, 7, 10):
        assert p[:, a == k].sum() == pytest.approx((np.abs(y) == k).sum(), rel=1e-6)


def test_sampling_matches_the_pmf():
    kn = mm.KeyNumbers(np.where(np.arange(mm.KMAX + 1) == 3, 2.5, np.where(np.arange(mm.KMAX + 1) == 0, 0.1, 1.0)))
    rng = np.random.default_rng(4)
    draws = mm.sample("keynum", np.full(200_000, 2.0), 13.5, rng, kn)
    p = mm.keynum_pmf([2.0], 13.5, kn)[0]
    for k in (-3, 0, 3, 7):
        assert (draws == k).mean() == pytest.approx(p[mm.GRID == k][0], abs=0.003)
    n = mm.sample("normal", np.full(200_000, 2.0), 13.5, rng)
    assert (n == 3).mean() == pytest.approx(mm.normal_pmf([2.0], 13.5)[0][mm.GRID == 3][0], abs=0.003)


def test_discrete_crps_against_the_sample_formula():
    rng = np.random.default_rng(2)
    p = mm.normal_pmf([1.0], 10.0)
    draws = rng.choice(mm.GRID, size=(1, 40_000), p=p[0])
    for y in (-14.0, 0.0, 3.0):
        a = mm.crps_discrete_per_game(np.array([y]), p)[0]
        b = metrics.crps_samples_per_game(np.array([y]), draws)[0]
        assert a == pytest.approx(b, rel=0.02)


# --------------------------------------------------------------------------- live wiring: A4s untouched

def _model_2026():
    path = live.model_path(2026)
    if not path.exists():
        pytest.skip("needs experiments/live/model_2026.json")
    return json.loads(path.read_text())


def test_a4s_probability_is_unchanged_by_the_margin_model(synth_pbp, synth_sched):
    import ml_predict
    model = _model_2026()
    s = asof.add_asof(synth_sched)
    todo = s[(s["season"] == 2023) & (s["week"] >= 5) & (s["game_type"] == "REG")]
    now = todo["as_of"].min() - pd.Timedelta(hours=1)
    s = s.copy()
    s.loc[s["kickoff"] >= now, ["home_score", "away_score", "result", "total"]] = np.nan
    elo = {"ratings": {t: 1500.0 + 10 * i for i, t in enumerate(["KC", "BUF", "NE", "MIA", "DAL", "PHI"])},
           "hfa_now": 50.0, "hfa_week": {}}
    feats = live_features.build(synth_pbp, s, s.loc[todo.index], now, elo)
    reference = live_features.predict(model, feats)                 # the Phase 1 computation
    margin = {"fit": FIT.to_dict(), "shape": "keynum", "keynum": mm.KeyNumbers().to_dict(), "version": "M-test"}
    with_margin = ml_predict.model_columns(model, margin, feats)
    without = ml_predict.model_columns(model, None, feats)
    assert np.array_equal(with_margin["p_home_model"].to_numpy(), reference)          # bit for bit
    assert np.array_equal(without["p_home_model"].to_numpy(), reference)
    assert without["spread_model"].isna().all() and with_margin["spread_model"].notna().all()
    np.testing.assert_allclose(with_margin["spread_model"], FIT.mean(feats[mm.FEATURES].to_numpy(float)))
    assert ml_predict.combined_version(model, margin) == model["model_version"] + "+M-test"


def test_the_2026_a4s_model_is_frozen():
    import hashlib
    m = _model_2026()
    assert m["model_version"] == "A4s-2026-c6b394e8"
    blob = json.dumps({"intercept": m["intercept"], "coef": m["coef"]}, sort_keys=True)
    assert m["model_version"].endswith(hashlib.sha256(blob.encode()).hexdigest()[:8])
    latest = live.read_latest(2026)
    if latest:                                                    # every logged probability comes from this fit
        g = pd.DataFrame(latest["games"]).set_index("game_id")
        p = live_features.predict(m, g)
        np.testing.assert_allclose(p, g["p_home_model"], atol=6e-5)


def test_margin_2026_file_matches_its_version():
    import ml_predict
    path = ml_predict.margin_path(2026)
    if not path.exists():
        pytest.skip("needs experiments/live/margin_2026.json")
    m = json.loads(path.read_text())
    fit, kn = ml_predict.margin_objects(m)
    assert mm.version(fit, m["shape"], kn, 2026) == m["version"]
    assert m["shape"] == mm.CHOSEN_SHAPE and m["spec"] == ml_predict.margin_spec()
    assert m["train"]["seasons"] == [2001, 2025]
