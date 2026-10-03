"""Evaluation harness: windows, walk-forward splits, metrics, bootstrap, calibration, baselines, registry."""
import json

import numpy as np
import pandas as pd
import pytest

from nflelo import evaluate
from nflelo.ml import asof, registry
from nflelo.ml.eval import baselines, bootstrap, calibration, metrics, walk_forward, windows


# ------------------------------------------------------------------ windows

def test_holdout_is_locked():
    assert windows.DEV == (2006, 2019) and windows.HOLDOUT == (2020, 2025)
    assert windows.check(windows.DEV) is False
    for w in [(2019, 2020), (2020, 2020), (2025, 2026), (2010, 2025)]:
        with pytest.raises(windows.HoldoutError):
            windows.check(w)
        assert windows.check(w, allow_holdout=True) is True
    df = pd.DataFrame({"season": [2019, 2020], "game_type": ["REG", "REG"]})
    with pytest.raises(windows.HoldoutError):
        windows.select(df, (2019, 2020))
    assert len(windows.select(df, (2006, 2019))) == 1


# ------------------------------------------------------------------ walk-forward

def test_season_splits_train_only_on_the_past():
    df = pd.DataFrame({"season": np.repeat(np.arange(2000, 2010), 3)})
    seen = []
    for s, train, test in walk_forward.season_splits(df, (2005, 2009)):
        assert (df.loc[train, "season"] < s).all() and (df.loc[test, "season"] == s).all()
        assert not (train & test).any()
        seen.append(s)
    assert seen == [2005, 2006, 2007, 2008, 2009]


def test_week_splits_respect_as_of(synth_sched):
    s = asof.add_asof(synth_sched)
    for week, train, test in walk_forward.week_splits(s, 2023):
        cutoff = s.loc[test, "as_of"].min()
        assert (s.loc[train, "kickoff"] < cutoff).all()
        assert not (train & test).any()
        if week >= 2:
            # The previous week's Monday game is in, this week's Thursday game is out.
            assert (s.loc[train & (s["season"] == 2023).to_numpy(), "week"] < week).all()


def test_walk_forward_predict_uses_prior_seasons_only():
    df = pd.DataFrame({"season": np.repeat([2000, 2001, 2002], 4), "x": np.arange(12.0)})
    pred = walk_forward.walk_forward_predict(df, (2001, 2002), lambda tr, te_: np.full(len(te_), tr["x"].max()))
    assert pred[df["season"] == 2000].isna().all()
    assert (pred[df["season"] == 2001] == 3.0).all() and (pred[df["season"] == 2002] == 7.0).all()


# ------------------------------------------------------------------ metrics

def test_win_metrics_match_elo_module():
    rng = np.random.default_rng(0)
    y = rng.choice([0.0, 0.5, 1.0], 500, p=[0.45, 0.01, 0.54])
    p = rng.uniform(0.05, 0.95, 500)
    assert metrics.win_metrics(y, p) == evaluate.metrics(y, p)
    assert metrics.brier_per_game(y, p).mean() == pytest.approx(evaluate.metrics(y, p)["brier"])
    assert metrics.logloss_per_game(y, p).mean() == pytest.approx(evaluate.metrics(y, p)["logloss"])


def test_margin_mae_and_crps():
    assert metrics.margin_mae([3, -7, 10], [0, 0, 0]) == pytest.approx(20 / 3)
    # CRPS of a point mass equals absolute error; a tiny sigma approaches it.
    assert metrics.crps_normal([3.0], [0.0], 1e-6) == pytest.approx(3.0, abs=1e-5)
    # Known value: CRPS(N(0,1), 0) = 2*phi(0) - 1/sqrt(pi) = 0.2337...
    assert metrics.crps_normal([0.0], [0.0], 1.0) == pytest.approx(0.23369497725, abs=1e-9)
    # Sample-based CRPS converges to the closed form.
    rng = np.random.default_rng(1)
    y = np.array([0.0, 5.0, -10.0])
    draws = rng.normal(2.0, 13.5, size=(3, 40000))
    assert metrics.crps_samples(y, draws) == pytest.approx(metrics.crps_normal(y, 2.0, 13.5), rel=0.01)


# ------------------------------------------------------------------ bootstrap and calibration

def test_paired_bootstrap_is_seeded_and_sensible():
    rng = np.random.default_rng(2)
    p_true = rng.uniform(0.2, 0.8, 3000)
    y = (rng.uniform(size=3000) < p_true).astype(float)
    a = bootstrap.paired_bootstrap(y, p_true, np.full(3000, 0.5), reps=500, seed=7)
    b = bootstrap.paired_bootstrap(y, p_true, np.full(3000, 0.5), reps=500, seed=7)
    assert a == b
    assert a["diff"] < 0 and a["ci_high"] < 0 and a["excludes_zero"]
    same = bootstrap.paired_bootstrap(y, p_true, p_true, reps=200)
    assert same["diff"] == 0 and not same["excludes_zero"]
    with pytest.raises(ValueError):
        bootstrap.paired_bootstrap(y, p_true, np.full(3000, np.nan))


def test_calibration_ece():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 200000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    assert calibration.ece(y, p) < 0.01
    assert calibration.ece(y, np.clip(p + 0.1, 0, 1)) > 0.08
    t = calibration.reliability_table(y, p, bins=10)
    assert len(t) == 10 and t["n"].sum() == p.size
    assert calibration.reliability_table(y, p, bins=5, strategy="quantile")["n"].sum() == p.size


# ------------------------------------------------------------------ baselines

def _games():
    rows = []
    for s in range(2000, 2008):
        for i in range(10):
            home_wins = i < (8 if s < 2004 else 4)
            rows.append({"game_id": f"{s}_{i}", "season": s, "week": 1, "game_type": "REG",
                         "home": "KC", "away": "BUF", "home_score": 21 if home_wins else 10,
                         "away_score": 10 if home_wins else 21, "neutral": False,
                         "home_moneyline": -120.0, "away_moneyline": 100.0})
    return pd.DataFrame(rows)


def test_home_rate_uses_only_earlier_seasons():
    g = _games()
    target = g[g["season"] == 2005]
    r = baselines.home_rate(target, g, lookback=2)
    assert np.allclose(r, 0.6)  # seasons 2003 (home won 8 of 10) and 2004 (4 of 10)
    # Rewriting season 2005 and later changes nothing for 2005.
    g2 = g.copy()
    late = g2["season"] >= 2005
    g2.loc[late, ["home_score", "away_score"]] = [50, 0]
    pd.testing.assert_series_equal(r, baselines.home_rate(g2[g2["season"] == 2005], g2, lookback=2))
    assert np.allclose(baselines.home_rate(target, g, lookback=None), (4 * 0.8 + 0.4) / 5)


def test_market_never_counts_as_a_prediction():
    g = _games()
    df = baselines.baseline_frame(g, (2004, 2005), require_market=True, home_rate_lookback=2)
    assert baselines.MARKET_COL in df.columns
    assert baselines.prediction_columns(df) == ["p_coin_flip", "p_home_rate", "p_elo_v2"]
    assert not [c for c in baselines.prediction_columns(df) if "market" in c or "moneyline" in c]


# ------------------------------------------------------------------ registry

def test_registry_writes_flags_and_ranks(tmp_path):
    ids = ["a", "b", "c"]
    registry.log_run("m1", label="dev", seasons=(2006, 2019), game_ids=ids, metrics={"n": 3, "brier": 0.22},
                     data={}, runs_dir=tmp_path)
    registry.log_run("m2", label="dev", seasons=(2006, 2019), game_ids=ids, metrics={"n": 3, "brier": 0.21},
                     data={}, runs_dir=tmp_path)
    p = registry.log_run("m3", label="holdout", seasons=(2020, 2025), game_ids=ids, metrics={"n": 3, "brier": 0.2},
                         data={}, runs_dir=tmp_path)
    rec = json.loads(p.read_text())
    assert rec["holdout"] is True and rec["label"] == "holdout"
    for key in ("name", "git", "data", "features", "params", "window", "n", "metrics", "timestamp", "games_hash"):
        assert key in rec
    with pytest.raises(windows.HoldoutError):
        registry.log_run("bad", label="dev", seasons=(2015, 2021), game_ids=ids, metrics={}, data={},
                         runs_dir=tmp_path)
    lb = registry.leaderboard("dev", runs_dir=tmp_path)
    assert list(lb["name"]) == ["m2", "m1"]
    assert registry.leaderboard("reproduction", runs_dir=tmp_path).empty
