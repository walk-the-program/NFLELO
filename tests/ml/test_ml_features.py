"""Team efficiency features: the market ban (D3), play filters, and the pooled-rate arithmetic."""
import re

import numpy as np
import pandas as pd
import pytest

from nflelo.ml import asof
from nflelo.ml.data import drop_market_columns, is_market_column
from nflelo.ml.features import team_efficiency as te

FORBIDDEN = re.compile(r"spread|total_line|moneyline|vegas", re.I)


@pytest.fixture(scope="module")
def sched(synth_sched):
    return asof.add_asof(synth_sched)


@pytest.fixture(scope="module")
def feats(synth_pbp, sched):
    return te.build_features(synth_pbp, sched)


def test_no_feature_column_carries_market_information(feats):
    assert not [c for c in te.feature_names() if FORBIDDEN.search(c)]
    assert not [c for c in feats.columns if FORBIDDEN.search(c)]
    te.assert_no_market_columns(feats)


def test_market_guard_raises_on_market_columns():
    for bad in ("spread_line", "home_total_line", "home_moneyline", "vegas_home_wp"):
        with pytest.raises(AssertionError):
            te.assert_no_market_columns(pd.DataFrame({bad: [1.0], "home_off_epa_all": [0.1]}))


def test_clean_plays_drops_market_columns_and_non_plays(synth_pbp):
    p = te.clean_plays(synth_pbp)
    assert not [c for c in p.columns if FORBIDDEN.search(c)]
    assert set(p["play_type"]) <= {"pass", "run"}
    assert (p["two_point_attempt"] != 1).all()
    assert p["wp"].between(0.05, 0.95).all()
    assert p["epa"].notna().all()


def test_market_column_detection():
    for c in ("spread_line", "total_line", "vegas_wp", "vegas_home_wp", "home_moneyline", "away_moneyline"):
        assert is_market_column(c)
    for c in ("epa", "wp", "total", "total_home_score", "posteam", "result"):
        assert not is_market_column(c)
    df = pd.DataFrame(columns=["epa", "spread_line", "vegas_wp", "wp"])
    assert list(drop_market_columns(df).columns) == ["epa", "wp"]


def _manual_rate(pbp, sched, team, season, as_of, side, w, kind="all"):
    p = te.clean_plays(pbp)
    k = sched.set_index("game_id")[["season", "kickoff"]]
    p = p.join(k, on="game_id", rsuffix="_g")
    col = "posteam" if side == "off" else "defteam"
    p = p[(p[col] == team) & (p["kickoff"] < as_of)]
    if kind == "pass":
        p = p[p["pass"] == 1]
    elif kind == "rush":
        p = p[(p["rush"] == 1) & (p["pass"] != 1)]
    cur = p[p["season_g"] == season]
    prev = p[p["season_g"] == season - 1]
    return (cur["epa"].sum() + w * prev["epa"].sum()) / (len(cur) + w * len(prev)), len(cur), len(prev)


@pytest.mark.parametrize("w", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("kind", ["all", "pass", "rush"])
def test_pooled_rate_matches_hand_computation(synth_pbp, sched, w, kind):
    game = sched[(sched["season"] == 2023) & (sched["week"] == 3)].iloc[[1]]
    f = te.build_features(synth_pbp, sched, game, te.EfficiencyConfig(prev_weight=w)).iloc[0]
    for side in te.SIDES:
        team = game[f"{side}_team"].iloc[0]
        for unit in ("off", "def"):
            rate, n_cur, n_prev = _manual_rate(synth_pbp, sched, team, 2023, game["as_of"].iloc[0], unit, w, kind)
            assert f[f"{side}_{unit}_epa_{kind}"] == pytest.approx(rate, abs=1e-12)
            if kind == "all":
                assert f[f"{side}_{unit}_plays"] == n_cur
                assert f[f"{side}_{unit}_plays_prev"] == n_prev


def test_success_rate_and_first_game_is_nan(synth_pbp, sched, feats):
    first = sched.sort_values("kickoff")["game_id"].iloc[0]
    assert np.isnan(feats.loc[first, "home_off_epa_all"])
    assert feats.loc[first, "home_off_plays"] == 0
    assert feats["home_off_sr"].dropna().between(0, 1).all()


def test_relocated_franchise_keeps_its_history(synth_pbp, synth_sched):
    """Raiders: OAK plays in 2019 feed LV's 2020 week 1 features through the franchise map."""
    s = synth_sched.copy()
    p = synth_pbp.copy()
    s["season"] = s["season"].map({2022: 2019, 2023: 2020})
    s["gameday"] = (pd.to_datetime(s["gameday"]) - pd.DateOffset(years=3)).dt.strftime("%Y-%m-%d")
    p["season"] = p["season"].map({2022: 2019, 2023: 2020})
    for frame, cols in ((s, ["home_team", "away_team"]), (p, ["home_team", "away_team", "posteam", "defteam"])):
        for c in cols:
            frame.loc[(frame["season"] == 2019) & (frame[c] == "KC"), c] = "OAK"
            frame.loc[(frame["season"] == 2020) & (frame[c] == "KC"), c] = "LV"
    s = asof.add_asof(s)
    g = s[(s["season"] == 2020) & (s["week"] == 1) & ((s["home_team"] == "LV") | (s["away_team"] == "LV"))]
    f = te.build_features(p, s, g).iloc[0]
    side = "home" if g["home_team"].iloc[0] == "LV" else "away"
    assert f[f"{side}_off_plays"] == 0 and f[f"{side}_off_plays_prev"] > 0
    assert not np.isnan(f[f"{side}_off_epa_all"])
