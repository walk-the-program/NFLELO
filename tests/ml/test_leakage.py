"""The leakage test: features must not change when everything at or after as_of is scrambled.

Runs offline on the synthetic frames in conftest.py. The positive control (a
full-season average) must FAIL the same check; if it passes, the check is broken.
"""
import numpy as np
import pandas as pd
import pytest

from nflelo.ml import asof
from nflelo.ml.features import team_efficiency as te
from nflelo.ml.features.leaky_demo import build_leaky_season_average


@pytest.fixture(scope="module")
def sched(synth_sched):
    return asof.add_asof(synth_sched)


def _gid(sched, season, week, slot):
    """slot 0 = Thursday, 1 = Sunday, 2 = Monday (schedule order within a week)."""
    rows = sched[(sched["season"] == season) & (sched["week"] == week)].sort_values("kickoff")
    return rows["game_id"].iloc[slot]


@pytest.fixture(scope="module")
def targets(sched):
    return {
        "week1_thursday_next_season": _gid(sched, 2023, 1, 0),   # only last season is known
        "week1_sunday_next_season": _gid(sched, 2023, 1, 1),
        "midseason_sunday": _gid(sched, 2023, 4, 1),
        "midseason_thursday": _gid(sched, 2023, 4, 0),
        "midseason_monday": _gid(sched, 2022, 5, 2),
        "playoff": sched.loc[(sched["season"] == 2023) & (sched["game_type"] == "WC"), "game_id"].iloc[0],
    }


def test_honest_features_are_leak_free(synth_pbp, sched, targets):
    res = asof.leakage_check(te.build_features, synth_pbp, sched, list(targets.values()))
    assert res["leak_free"].all(), res.to_string()
    # And the features are real numbers, not an all-NaN frame that would pass trivially.
    feats = te.build_features(synth_pbp, sched, sched[sched["game_id"].isin(targets.values())])
    assert feats[[c for c in feats.columns if c.endswith("epa_all")]].notna().all().all()


def test_check_is_sensitive_to_data_before_as_of(synth_pbp, sched, targets):
    """Corrupting data from a week EARLIER must change the features, or the check proves nothing."""
    gid = targets["midseason_sunday"]
    game = sched[sched["game_id"] == gid]
    clean = te.build_features(synth_pbp, sched, game)
    earlier = game["as_of"].iloc[0] - pd.Timedelta(days=7)
    p2, s2 = asof.corrupt_from(synth_pbp, sched, earlier, seed=5)
    dirty = te.build_features(p2, s2, s2[s2["game_id"] == gid])
    assert not np.allclose(clean.to_numpy(float), dirty.to_numpy(float), equal_nan=True)


def test_thursday_result_does_not_reach_that_weeks_sunday(synth_pbp, sched):
    thu, sun = _gid(sched, 2023, 4, 0), _gid(sched, 2023, 4, 1)
    nxt = _gid(sched, 2023, 5, 1)
    p2 = synth_pbp.copy()
    m = p2["game_id"] == thu
    p2.loc[m, "epa"] = p2.loc[m, "epa"] + 10.0  # a wildly different Thursday game
    games = sched[sched["game_id"].isin([sun, nxt])]
    before = te.build_features(synth_pbp, sched, games)
    after = te.build_features(p2, sched, games)
    pd.testing.assert_series_equal(before.loc[sun], after.loc[sun])
    # The Thursday game IS known by the next week, so the next week's features move
    # (if one of its teams played on Thursday).
    thu_teams = set(sched.loc[sched["game_id"] == thu, ["home_team", "away_team"]].iloc[0])
    nxt_teams = set(sched.loc[sched["game_id"] == nxt, ["home_team", "away_team"]].iloc[0])
    if thu_teams & nxt_teams:
        assert not before.loc[nxt].equals(after.loc[nxt])


def test_leaky_season_average_fails_the_check(synth_pbp, sched, targets):
    """Positive control: a full-season average must be caught."""
    gids = [targets["midseason_sunday"], targets["midseason_thursday"], targets["midseason_monday"]]
    res = asof.leakage_check(build_leaky_season_average, synth_pbp, sched, gids)
    assert not res["leak_free"].any(), res.to_string()
    assert (res["changed"] > 0).all()


def test_corrupt_from_leaves_earlier_games_untouched(synth_pbp, sched):
    cut = sched.loc[(sched["season"] == 2023) & (sched["week"] == 3), "as_of"].iloc[0]
    p2, s2 = asof.corrupt_from(synth_pbp, sched, cut, seed=3)
    early = set(sched.loc[sched["kickoff"] < cut, "game_id"])
    m = synth_pbp["game_id"].isin(early)
    pd.testing.assert_frame_equal(synth_pbp[m], p2[m], check_dtype=False)
    assert not np.allclose(synth_pbp.loc[~m, "epa"], p2.loc[~m, "epa"])
    late = ~sched["game_id"].isin(early)
    assert not np.allclose(sched.loc[late, "home_score"], s2.loc[late, "home_score"])
