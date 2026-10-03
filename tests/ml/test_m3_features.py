"""M3 feature builders: leakage checks, the market ban, ridge sanity, QB and context details.

Offline, on the synthetic frames in conftest.py. Every new builder must pass
`asof.leakage_check`; the full-season-average positive control must still fail.
The starting QB's identity is the one allowed exception (decision M3-D1): the
actual-starter builder is checked with starter IDs kept, and a second test
shows that scrambling them is exactly what it is sensitive to.
"""
import numpy as np
import pandas as pd
import pytest

from nflelo.ml import asof
from nflelo.ml.features import context, opponent_adjust as oa, qb
from nflelo.ml.features import team_efficiency as te
from nflelo.ml.features.leaky_demo import build_leaky_season_average

CFG = oa.RatingConfig(lam=50.0, half_life=4.0, rho=0.5)
QCFG = qb.QBConfig(k=50.0, half_life=4.0, rho=0.5)


@pytest.fixture(scope="module")
def sched(synth_sched):
    return asof.add_asof(synth_sched)


def _gid(sched, season, week, slot):
    rows = sched[(sched["season"] == season) & (sched["week"] == week)].sort_values("kickoff")
    return rows["game_id"].iloc[slot]


@pytest.fixture(scope="module")
def targets(sched):
    return [
        _gid(sched, 2023, 1, 0),   # week 1 Thursday: only last season is known
        _gid(sched, 2023, 1, 1),
        _gid(sched, 2023, 3, 0),   # KC's backup starts weeks 3-4 of 2023
        _gid(sched, 2023, 4, 1),
        _gid(sched, 2023, 4, 2),
        _gid(sched, 2022, 5, 2),
        sched.loc[(sched["season"] == 2023) & (sched["game_type"] == "WC"), "game_id"].iloc[0],
    ]


def adj_builder(split):
    return lambda p, s, g: oa.build_features(p, s, g, cfg=CFG, split=split)


def qb_builder(starter):
    return lambda p, s, g: qb.build_features(p, s, g, cfg=QCFG, starter=starter)


def elo_builder(p, s, g):
    return context.build_elo_features(p, s, g)


# --------------------------------------------------------------------------- leakage

@pytest.mark.parametrize("name,builder", [
    ("adjusted", adj_builder(False)),
    ("adjusted_split", adj_builder(True)),
    ("qb_actual_starter", qb_builder("actual")),
    ("qb_last_starter", qb_builder("last")),
    ("context", context.build_features),
    ("elo", elo_builder),
])
def test_m3_builders_are_leak_free(synth_pbp, sched, targets, name, builder):
    res = asof.leakage_check(builder, synth_pbp, sched, targets)
    assert res["leak_free"].all(), f"{name}\n{res.to_string()}"
    feats = builder(synth_pbp, sched, sched[sched["game_id"].isin(targets)])
    assert feats.notna().all().all()
    if name != "context":
        assert (feats.abs() > 1e-9).any().all(), f"{name} features are all zero; the check would pass trivially"


def test_positive_control_still_fails(synth_pbp, sched, targets):
    res = asof.leakage_check(build_leaky_season_average, synth_pbp, sched, targets[3:6])
    assert not res["leak_free"].any()


@pytest.mark.parametrize("builder", [adj_builder(False), qb_builder("actual"), elo_builder])
def test_checks_see_data_before_as_of(synth_pbp, sched, builder):
    """Corrupting from one week earlier must move the features, or the check proves nothing."""
    gid = _gid(sched, 2023, 4, 1)
    game = sched[sched["game_id"] == gid]
    clean = builder(synth_pbp, sched, game)
    p2, s2 = asof.corrupt_from(synth_pbp, sched, game["as_of"].iloc[0] - pd.Timedelta(days=7), seed=5)
    dirty = builder(p2, s2, s2[s2["game_id"] == gid])
    assert not np.allclose(clean.to_numpy(float), dirty.to_numpy(float))


def test_actual_starter_is_the_only_exception(synth_pbp, sched, targets):
    """Scrambling late starting-QB IDs moves A4 (identity is used, M3-D1) but never A4b (strict rule)."""
    a4 = asof.leakage_check(qb_builder("actual"), synth_pbp, sched, targets, scramble_starters=True)
    a4b = asof.leakage_check(qb_builder("last"), synth_pbp, sched, targets, scramble_starters=True)
    assert not a4["leak_free"].all()
    assert a4b["leak_free"].all(), a4b.to_string()


def test_qb_value_ignores_the_game_itself(synth_pbp, sched):
    """The starter's own dropbacks in this game (and later) never enter his value."""
    gid = _gid(sched, 2023, 4, 2)  # KC backup at home
    game = sched[sched["game_id"] == gid]
    clean = qb.build_features(synth_pbp, sched, game, QCFG, detail=True)
    p2 = synth_pbp.copy()
    m = (p2["game_id"] == gid) & (p2["passer_id"] == "KC_QB2")
    p2.loc[m, "epa"] = 50.0
    after = qb.build_features(p2, sched, game, QCFG, detail=True)
    pd.testing.assert_frame_equal(clean, after)


# --------------------------------------------------------------------------- market ban (D3)

@pytest.mark.parametrize("builder", [adj_builder(True), qb_builder("actual"), context.build_features, elo_builder])
def test_builders_ignore_and_never_emit_market_columns(synth_pbp, sched, builder):
    games = sched[sched["season"] == 2023]
    f = builder(synth_pbp, sched, games)
    te.assert_no_market_columns(f)
    mk = [c for c in synth_pbp.columns if "vegas" in c or "line" in c]
    sk = [c for c in sched.columns if "line" in c or "moneyline" in c]
    assert mk and sk
    g = builder(synth_pbp.drop(columns=mk), sched.drop(columns=sk), games.drop(columns=sk))
    pd.testing.assert_frame_equal(f, g)


# --------------------------------------------------------------------------- ridge sanity

def _play_level_ridge(off, deff, home, y, w, n_teams, lam):
    p = 2 + 2 * n_teams
    X = np.zeros((len(y), p))
    X[:, 0] = 1
    X[:, 1] = home
    X[np.arange(len(y)), 2 + off] = 1
    X[np.arange(len(y)), 2 + n_teams + deff] = 1
    pen = np.r_[0.0, 1e-9 * w.sum(), np.full(2 * n_teams, lam)]
    return np.linalg.solve(X.T @ (X * w[:, None]) + np.diag(pen), X.T @ (w * y))


def test_aggregated_fit_equals_play_level_fit():
    rng = np.random.default_rng(0)
    T, games = 6, 40
    off = rng.integers(0, T, games)
    deff = (off + rng.integers(1, T, games)) % T
    home = rng.integers(0, 2, games).astype(float)
    n = rng.integers(20, 70, games)
    w = rng.uniform(0.2, 1.0, games)
    plays_y = [rng.normal(0.05 * (o - d), 1.3, k) for o, d, k in zip(off, deff, n)]
    mu, h, O, D = oa.fit_ridge(off, deff, home, n.astype(float), np.array([y.sum() for y in plays_y]), w, T, 30.0)
    rep = lambda a: np.repeat(a, n)  # noqa: E731
    beta = _play_level_ridge(rep(off), rep(deff), rep(home), np.concatenate(plays_y), rep(w), T, 30.0)
    np.testing.assert_allclose([mu, h, *O, *D], beta, atol=1e-10)


def test_ridge_recovers_known_ratings_and_shrinks():
    rng = np.random.default_rng(1)
    T, games = 8, 4000
    O_true = np.linspace(-0.15, 0.15, T)
    D_true = rng.permutation(np.linspace(-0.1, 0.1, T))
    off = rng.integers(0, T, games)
    deff = (off + rng.integers(1, T, games)) % T
    home = rng.integers(0, 2, games).astype(float)
    n = np.full(games, 400.0)
    mean = 0.01 + O_true[off] + D_true[deff] + 0.03 * home
    s = n * mean + rng.normal(0, 1.3 * np.sqrt(n))
    w = np.ones(games)
    mu, h, O, D = oa.fit_ridge(off, deff, home, n, s, w, T, 1.0)
    # Ratings are identified up to a shift between mu, O and D; compare centered values.
    np.testing.assert_allclose(O - O.mean(), O_true - O_true.mean(), atol=0.01)
    np.testing.assert_allclose(D - D.mean(), D_true - D_true.mean(), atol=0.01)
    assert abs(h - 0.03) < 0.01
    # A huge penalty pulls every rating to (nearly) zero and mu to the weighted mean.
    mu2, _, O2, D2 = oa.fit_ridge(off, deff, home, n, s, w, T, 1e9)
    assert np.abs(O2).max() < 1e-4 and np.abs(D2).max() < 1e-4
    assert abs(mu2 + 0.03 * home.mean() - s.sum() / n.sum()) < 2e-3


def test_week_one_is_last_season_shrunk_by_rho(synth_pbp, sched):
    tab = oa.rating_table(synth_pbp, sched, CFG)
    full = oa.compute_ratings(tab, sched, [(2023, 1)], CFG)
    weak = oa.compute_ratings(tab, sched, [(2023, 1)], CFG.__class__(lam=CFG.lam, half_life=CFG.half_life, rho=0.01))
    assert full["off"].abs().mean() > 5 * weak["off"].abs().mean()


def test_week_ordinals_skip_the_offseason(sched):
    o = oa.week_ordinals(sched)
    assert o[(2023, 1)] - o[(2022, 19)] == 1     # playoff week 19 is 2022's last week
    assert o[(2022, 2)] - o[(2022, 1)] == 1


def test_matchup_margin_formula(sched):
    r = pd.DataFrame({"kind": "all", "season": 2023, "week": 2, "team": ["KC", "BUF"],
                      "off": [0.10, -0.02], "def": [-0.05, 0.03]})
    g = pd.DataFrame({"season": [2023], "week": [2], "home_team": ["KC"], "away_team": ["BUF"]})
    # (O_KC + D_BUF) - (O_BUF + D_KC) = (0.10 + 0.03) - (-0.02 - 0.05) = 0.20
    assert oa.matchup_margin(r, g)[0] == pytest.approx(0.20)


# --------------------------------------------------------------------------- QB details

def test_qb_value_formula_and_prior(synth_pbp, sched):
    db = qb.dropback_plays(synth_pbp, sched, QCFG)
    qg = qb.qb_game_table(db, sched)
    priors = qb.replacement_priors(db, [2023], QCFG.rookie_dropbacks)
    states, qbs = qb.week_states(qg, sched, [(2023, 3)], QCFG, priors)
    st = states[(2023, 3)]
    # Hand computation for BUF_QB1.
    a = sched.loc[(sched["season"] == 2023) & (sched["week"] == 3), "as_of"].min()
    past = qg[qg["kick_ns"] < te._utc_ns(pd.Series([a]))[0]]
    past = past[past["qb"] == "BUF_QB1"]
    ords = oa.week_ordinals(sched)
    w = oa.decay_weights(int(ords[(2023, 3)]), past["ord"].to_numpy(), 2023, past["season"].to_numpy(),
                         QCFG.half_life, QCFG.rho)
    N, E = (w * past["n"]).sum(), (w * past["epa"]).sum()
    want = (E + QCFG.k * priors[2023]) / (N + QCFG.k)
    assert st.values(QCFG.k)[qbs.get_loc("BUF_QB1")] == pytest.approx(want)
    # A QB with no earlier dropbacks sits exactly at the prior; k -> infinity sends everyone there.
    assert qb.WeekState(st.qb_codes, np.zeros_like(st.n), np.zeros_like(st.e), st.team_mix, 0.1).values(5)[0] == 0.1
    assert np.allclose(st.values(1e12), st.prior)


def test_replacement_prior_uses_only_earlier_seasons(synth_pbp, sched):
    db = qb.dropback_plays(synth_pbp, sched, QCFG)
    before = qb.replacement_priors(db, [2023])
    db2 = db.copy()
    db2.loc[db2["season"] == 2023, "epa"] += 10.0
    assert qb.replacement_priors(db2, [2023]) == before


def test_last_starter_is_previous_game(sched):
    wk5 = sched[(sched["season"] == 2023) & (sched["week"] == 5)]
    ls = qb.last_starters(sched, wk5)
    kc = wk5[(wk5["home_team"] == "KC") | (wk5["away_team"] == "KC")]
    side = "home" if kc["home_team"].iloc[0] == "KC" else "away"
    assert ls.loc[kc["game_id"].iloc[0], f"{side}_qb_id"] == "KC_QB2"   # backup started week 4
    assert kc[f"{side}_qb_id"].iloc[0] == "KC_QB1"                    # starter is back in week 5


def test_backup_start_is_bad_news_and_starter_return_good_news(synth_pbp, sched):
    """Make KC's backup clearly worse than the starter. His start gets a big negative delta. After it,
    the team's rating "remembers" the backup's weeks, so the starter's return is a positive delta."""
    p = synth_pbp.copy()
    p.loc[p["passer_id"] == "KC_QB2", "epa"] = -1.5
    p.loc[p["passer_id"] == "KC_QB1", "epa"] = 0.5
    g = sched[(sched["season"] == 2023) & sched["week"].isin([3, 6])]
    f = qb.build_features(p, sched, g, QCFG)
    kc = g[(g["home_team"] == "KC") | (g["away_team"] == "KC")]
    d = {int(r.week): f.loc[r.game_id, "qb_delta_home" if r.home_team == "KC" else "qb_delta_away"]
         for r in kc.itertuples()}
    assert d[3] < -0.3
    assert d[6] > 0.1


# --------------------------------------------------------------------------- context and Elo

def test_context_rest_cap_and_neutral():
    g = pd.DataFrame({"game_id": ["a", "b", "c"], "home_rest": [14, 6, np.nan], "away_rest": [6, 13, 7],
                      "location": ["Home", "Neutral", "Home"]})
    f = context.build_features(None, g, g)
    assert f["rest_diff"].tolist() == [7.0, -7.0, 0.0]
    assert f["neutral"].tolist() == [0.0, 1.0, 0.0]


def test_elo_hfa_rebuild_matches_engine_and_week_start(sched):
    from nflelo import config
    from nflelo.data import nflverse_to_games
    from nflelo.elo import run_elo
    g = nflverse_to_games(sched)
    out, _ = run_elo(g, config.DEFAULT_CONFIG)
    hb = context.hfa_before(out, config.DEFAULT_CONFIG)
    live = ~out["neutral"].to_numpy(bool)
    np.testing.assert_allclose(hb[live], out["hfa_used"].to_numpy()[live])
    f = context.elo_features_from_games(g)
    # The first game of each week uses exactly Elo's own log-odds.
    first = out.groupby(["season", "week"]).head(1)
    lo = np.log(first["p_home"] / (1 - first["p_home"])).to_numpy()
    np.testing.assert_allclose(f.loc[first["game_id"], "elo_logit"].to_numpy(), lo, atol=1e-9)
    # Later games in the week use the week-start HFA, not the post-Thursday one.
    assert f.groupby(out["season"].astype(str).to_numpy() + out["week"].astype(str).to_numpy())["hfa_week"].nunique().max() == 1


def test_qb_deltas_are_side_symmetric(synth_pbp, sched):
    """Swapping home and away in the schedule swaps qb_delta_home and qb_delta_away exactly (no side bug)."""
    g = sched[sched["season"] == 2023]
    f = qb.build_features(synth_pbp, sched, g, QCFG)
    swap = {"home_team": "away_team", "away_team": "home_team", "home_qb_id": "away_qb_id",
            "away_qb_id": "home_qb_id", "home_score": "away_score", "away_score": "home_score"}
    s2 = sched.rename(columns=swap)
    f2 = qb.build_features(synth_pbp, s2, s2[s2["season"] == 2023], QCFG)
    np.testing.assert_allclose(f["qb_delta_home"], f2["qb_delta_away"])
    np.testing.assert_allclose(f["qb_delta_away"], f2["qb_delta_home"])
    assert (f.abs() > 1e-9).any().all()
