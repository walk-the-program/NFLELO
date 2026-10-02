import math

import pandas as pd
import pytest

from nflelo.config import EloConfig
from nflelo.elo import expected_home, mov_multiplier, run_elo


def games(rows):
    df = pd.DataFrame(rows, columns=["season", "home", "away", "home_score", "away_score",
                                     "neutral", "game_type"])
    df["game_id"] = range(len(df))
    return df


def test_expected_prob_symmetry():
    p = expected_home(1600, 1500, 0.0)
    q = expected_home(1500, 1600, 0.0)
    assert p + q == pytest.approx(1.0)
    assert expected_home(1500, 1500, 0.0) == 0.5


def test_hfa_raises_home_probability():
    assert expected_home(1500, 1500, 55) == pytest.approx(1 / (1 + 10 ** (-55 / 400)))
    assert expected_home(1500, 1500, 55) > 0.5


def test_neutral_game_uses_zero_hfa():
    out, _ = run_elo(games([(2000, "A", "B", 20, 10, True, "REG"),
                            (2000, "A", "B", 20, 10, False, "REG")]), EloConfig(hfa=55.0))
    assert out["hfa_used"].tolist() == [0.0, 55.0]
    assert out["p_home"].iloc[0] == 0.5


def test_updates_are_zero_sum():
    rows = [(2000, "A", "B", 24, 17, False, "REG"), (2000, "B", "C", 3, 30, False, "REG"),
            (2000, "C", "A", 10, 10, False, "REG")]
    out, state = run_elo(games(rows), EloConfig())
    for r in out.itertuples():
        assert (r.home_post - r.home_pre) + (r.away_post - r.away_pre) == pytest.approx(0.0)
    assert sum(state["ratings"].values()) == pytest.approx(3 * 1500)


def test_mov_multiplier_values():
    cfg = EloConfig()
    assert mov_multiplier(0, 100, cfg) == 1.0
    assert mov_multiplier(1, 0, cfg) == pytest.approx(math.log(2))
    assert mov_multiplier(10, 0, cfg) == pytest.approx(math.log(11))
    # a favorite (positive winner-minus-loser diff) gets a smaller multiplier
    assert mov_multiplier(10, 200, cfg) == pytest.approx(math.log(11) * 2.2 / (0.2 + 2.2))
    assert mov_multiplier(10, 200, cfg) < mov_multiplier(10, -200, cfg)


def test_legacy_mov_uses_abs_diff_and_cap():
    cfg = EloConfig(mov_mode="abs", mov_cap=2.0)
    assert mov_multiplier(10, -200, cfg) == mov_multiplier(10, 200, cfg)
    assert mov_multiplier(60, 0, cfg) == 2.0


def test_tie_counts_half_and_equal_teams_move_by_hfa_only():
    out, _ = run_elo(games([(2000, "A", "B", 10, 10, True, "REG")]), EloConfig())
    assert out["home_post"].iloc[0] == 1500.0  # tie between equal teams, no HFA: no change


def test_season_regression_toward_start():
    cfg = EloConfig(lam=0.25)
    out, _ = run_elo(games([(2000, "A", "B", 30, 0, True, "REG"),
                            (2001, "A", "C", 30, 0, True, "REG")]), cfg)
    end_2000 = out["home_post"].iloc[0]
    assert out["home_pre"].iloc[1] == pytest.approx(0.75 * end_2000 + 0.25 * 1500)


def test_expansion_team_starts_at_expansion_rating():
    cfg = EloConfig(expansion_start=1300.0)
    out, _ = run_elo(games([(2000, "A", "B", 20, 10, True, "REG"),
                            (2001, "A", "N", 20, 10, True, "REG")]), cfg)
    assert out["away_pre"].iloc[1] == 1300.0
    assert out["home_pre"].iloc[0] == 1500.0  # first-season teams use `start`


def test_playoffs_flag():
    rows = [(2000, "A", "B", 20, 10, False, "REG"), (2000, "A", "B", 20, 10, False, "WC")]
    on, _ = run_elo(games(rows), EloConfig(include_playoffs=True))
    off, _ = run_elo(games(rows), EloConfig(include_playoffs=False))
    assert on["updated"].tolist() == [True, True]
    assert off["updated"].tolist() == [True, False]
    assert off["home_post"].iloc[1] == off["home_pre"].iloc[1]
    assert off["p_home"].iloc[1] > 0.5  # still predicted


def test_online_hfa_moves_with_surprise_and_skips_neutral():
    cfg = EloConfig(hfa_mode="online", hfa_init=65.0, k_hfa=2.0)
    rows = [(2000, "A", "B", 10, 20, False, "REG"),   # home loses: HFA falls
            (2000, "C", "D", 10, 20, True, "REG"),    # neutral: no HFA change
            (2000, "E", "F", 30, 10, False, "REG")]   # home wins: HFA rises
    out, state = run_elo(games(rows), cfg)
    p0 = expected_home(1500, 1500, 65.0)
    assert out["hfa_used"].tolist()[0] == 65.0
    assert out["hfa_used"].iloc[1] == 0.0
    h1 = 65.0 + 2.0 * (0 - p0)
    assert out["hfa_used"].iloc[2] == pytest.approx(h1)  # neutral game left HFA alone
    assert state["hfa"] == pytest.approx(h1 + 2.0 * (1 - expected_home(1500, 1500, h1)))


def test_fixed_hfa_never_changes():
    rows = [(2000, "A", "B", 10, 20, False, "REG")] * 5
    _, state = run_elo(games(rows), EloConfig(hfa_mode="fixed", hfa=55.0))
    assert state["hfa"] == 55.0
