"""NFL tiebreakers: hand-built two- and three-club cases for each step, and the historical reproduction.

Each case is a tiny league: the games under test plus 0-0 filler ties so all 32
franchises exist. Tie games count as half a win; the filler teams never play the
clubs under test, so they change nothing the steps look at.
"""
import numpy as np
import pandas as pd
import pytest

from nflelo.meta import TEAM_ORDER
from nflelo.ml.sim import tiebreak as tb

SEASON = 2023


def league(games, seed=0, points=True, log=None):
    """games: (home, away, home_pts, away_pts). Returns Standings (with points unless `points` is False)."""
    used = {t for g in games for t in g[:2]}
    rest = [t for t in TEAM_ORDER if t not in used]
    filler = [(rest[i], rest[i + 1], 0, 0) for i in range(0, len(rest) - 1, 2)]
    if len(rest) % 2:
        filler.append((rest[-1], rest[0], 0, 0))
    allg = list(games) + filler
    df = pd.DataFrame(allg, columns=["home_team", "away_team", "home_score", "away_score"])
    df["season"], df["game_type"] = SEASON, "REG"
    df["game_id"] = [f"{SEASON}_{i:03d}" for i in range(len(df))]
    st = tb.SeasonStatic(df, SEASON)
    hs, as_ = df["home_score"].to_numpy(float), df["away_score"].to_numpy(float)
    res = np.where(np.isnan(hs), np.nan, np.where(hs > as_, 1.0, np.where(hs < as_, 0.0, 0.5)))
    return tb.Standings.from_results(st, res, hs - as_, hs if points else None, as_ if points else None,
                                     np.random.default_rng(seed), log)


def W(a, b, pa=20, pb=10):
    """a beats b at a's home."""
    return (a, b, pa, pb)


def idx(s, *teams):
    return [s.st.idx[t] for t in teams]


def pick(s, kind, *teams):
    log = []
    s.log = log
    t = s._break(idx(s, *teams), kind)
    return s.st.teams[t], log[-1]["step"] if log else None


# --------------------------------------------------------------------------- division, two clubs

def test_div2_head_to_head():
    s = league([W("BUF", "MIA"), W("KC", "BUF"), W("MIA", "HOU")])
    assert pick(s, "div", "BUF", "MIA") == ("BUF", "h2h")


def test_div2_division_record():
    s = league([W("BUF", "MIA"), W("MIA", "BUF"), W("BUF", "NE"), W("KC", "BUF"), W("NYJ", "MIA"), W("MIA", "KC")])
    assert pick(s, "div", "BUF", "MIA") == ("BUF", "div")


def test_div2_common_games():
    s = league([W("BUF", "MIA"), W("MIA", "BUF"), W("BUF", "KC"), W("BUF", "HOU"), W("PHI", "BUF"),
                W("KC", "MIA"), W("MIA", "HOU"), W("MIA", "DAL")])
    assert pick(s, "div", "BUF", "MIA") == ("BUF", "common")


def test_div2_conference_record():
    s = league([W("BUF", "MIA"), W("MIA", "BUF"), W("BUF", "KC"), W("DAL", "BUF"), W("MIA", "SEA"), W("HOU", "MIA")])
    assert s.s_common(idx(s, "BUF", "MIA")) is None            # no common opponents: step skipped
    assert pick(s, "div", "BUF", "MIA") == ("BUF", "conf")


def test_div2_strength_of_victory():
    s = league([W("BUF", "MIA"), W("MIA", "BUF"), W("BUF", "KC"), W("DAL", "BUF"), W("MIA", "HOU"), W("PHI", "MIA"),
                W("KC", "NO"), W("KC", "ATL"), W("NO", "HOU")])
    assert s.stats["sov"][s.st.idx["BUF"]] > s.stats["sov"][s.st.idx["MIA"]]
    assert pick(s, "div", "BUF", "MIA") == ("BUF", "sov")


def symmetric(buf_pts=(20, 10), mia_pts=(20, 10)):
    """BUF and MIA identical through strength of schedule; only the points differ."""
    return [("BUF", "MIA", 17, 14), ("MIA", "BUF", 17, 14), ("BUF", "NYJ", *buf_pts), ("MIA", "NE", *mia_pts),
            ("NE", "NYJ", 10, 7), ("NYJ", "NE", 10, 7)]


def test_div2_points_steps_then_coin():
    s = league(symmetric((30, 3), (20, 10)))
    for k in ("h2h", "div", "conf", "sov", "sos"):
        v = getattr(s, "s_" + k)(idx(s, "BUF", "MIA"))
        assert v is None or len(set(np.round(list(v.values()), 9))) == 1, k
    assert pick(s, "div", "BUF", "MIA") == ("BUF", "rank_conf")
    # Margins only (simulations): the combined ranking falls back to net points, same answer here.
    s2 = league(symmetric((30, 3), (20, 10)), points=False)
    assert pick(s2, "div", "BUF", "MIA") == ("BUF", "rank_conf")
    # Identical everything: net touchdowns are unknown, so a coin toss decides, reproducibly per seed.
    s3 = league(symmetric(), seed=1)
    team, step = pick(s3, "div", "BUF", "MIA")
    assert step == "coin" and team in ("BUF", "MIA")
    assert pick(league(symmetric(), seed=1), "div", "BUF", "MIA")[0] == team


def test_net_points_steps():
    s = league(symmetric((30, 3), (20, 10)))
    g = idx(s, "BUF", "MIA")
    assert s.s_net(g)[g[0]] == 3 - 3 + 27 and s.s_net(g)[g[1]] == -3 + 3 + 10
    assert s.s_net_conf(g) == s.s_net(g)                       # every game here is an AFC game
    assert s.s_net_common(g) == {g[0]: 0.0, g[1]: 0.0}         # no common opponents
    r = s._ranking("comb", "all")
    assert r[g[0]] < r[g[1]]


def test_combined_ranking_shares_tied_ranks():
    assert tb._min_rank(np.array([30.0, 30.0, 10.0]), True).tolist() == [1, 1, 3]
    assert tb._min_rank(np.array([7.0, 3.0, 3.0]), False).tolist() == [3, 1, 1]


# --------------------------------------------------------------------------- division, three clubs

def test_div3_head_to_head_among_clubs_then_two_club_restart():
    # Among the three: BUF 3-1, MIA 2-2, NE 1-3; everyone 4-3 overall. BUF wins the three-club head-to-head,
    # then MIA and NE start over at two-club step 1 (MIA won both of their games).
    g = [W("BUF", "MIA"), ("MIA", "BUF", 10, 20), W("BUF", "NE"), W("NE", "BUF"), W("MIA", "NE"), ("NE", "MIA", 10, 20),
         W("BUF", "DAL"), W("PHI", "BUF"), W("NYG", "BUF"),
         W("MIA", "WAS"), W("MIA", "CHI"), W("DET", "MIA"),
         W("NE", "GB"), W("NE", "MIN"), W("NE", "SEA")]
    s = league(g)
    assert len({round(s.pct(t), 9) for t in idx(s, "BUF", "MIA", "NE")}) == 1
    log = []
    s.log = log
    assert [s.st.teams[t] for t in s.division_order("AFC East")] == ["BUF", "MIA", "NE", "NYJ"]
    assert [(x["group"], x["step"], x["chosen"]) for x in log] == [(("BUF", "MIA", "NE"), "h2h", "BUF"),
                                                                   (("MIA", "NE"), "h2h", "MIA")]


# --------------------------------------------------------------------------- wild card, two clubs

def test_wc2_head_to_head_if_played():
    s = league([W("BAL", "HOU"), W("KC", "BAL"), W("HOU", "DAL")])
    assert pick(s, "wc", "BAL", "HOU") == ("BAL", "h2h_if_played")


def test_wc2_skips_head_to_head_when_not_played():
    s = league([W("BAL", "KC"), W("DAL", "BAL"), W("HOU", "PHI"), W("LV", "HOU")])
    assert s.s_h2h_if_played(idx(s, "BAL", "HOU")) is None
    assert pick(s, "wc", "BAL", "HOU") == ("BAL", "conf")


def test_wc2_common_games_need_four():
    base = [W("BAL", "NE"), W("BAL", "NYJ"), W("MIA", "BAL"), W("HOU", "NE"), W("NYJ", "HOU"), W("MIA", "HOU")]
    s = league(base)
    assert s.s_common4(idx(s, "BAL", "HOU")) is None          # three common games: step skipped
    assert s.s_common(idx(s, "BAL", "HOU")) is not None
    s4 = league(base + [W("BAL", "BUF"), W("BUF", "HOU")])
    v = s4.s_common4(idx(s4, "BAL", "HOU"))
    assert v is not None and v[s4.st.idx["BAL"]] > v[s4.st.idx["HOU"]]


def test_wc2_strength_of_schedule():
    # Same records everywhere; BAL's opponents won more games elsewhere.
    s = league([W("BAL", "NYJ"), W("DAL", "BAL"), W("HOU", "NE"), W("PHI", "HOU"),
                W("DAL", "SEA"), W("NYJ", "SF")])
    v = s.s_sos(idx(s, "BAL", "HOU"))
    assert v[s.st.idx["BAL"]] > v[s.st.idx["HOU"]]
    team, step = pick(s, "wc", "BAL", "HOU")
    assert team == "BAL" and step in ("sov", "sos")


# --------------------------------------------------------------------------- wild card, three or more clubs

def test_wc3_head_to_head_sweep_then_two_club():
    s = league([W("BAL", "HOU"), W("BAL", "KC"), W("DAL", "BAL"), W("PHI", "BAL"),
                W("HOU", "KC"), W("HOU", "NYG"), W("WAS", "HOU"),
                W("KC", "ATL"), W("KC", "CAR")])
    order = [s.st.teams[t] for t in s._order(idx(s, "BAL", "HOU", "KC"), s._wc_pick)]
    assert order == ["BAL", "HOU", "KC"]


def test_wc3_club_that_lost_to_each_is_eliminated():
    # KC lost to BAL and HOU; BAL and HOU never met, so nobody swept. KC drops out, then BAL vs HOU (conference record).
    s = league([W("BAL", "KC"), W("HOU", "KC"), W("BAL", "NE"), W("DAL", "BAL"), W("PHI", "BAL"),
                W("HOU", "NYG"), W("NE", "HOU"), W("WAS", "HOU"), W("KC", "ATL"), W("KC", "CAR")])
    assert s.s_sweep(idx(s, "BAL", "HOU", "KC")) == {s.st.idx["BAL"]: 1.0, s.st.idx["HOU"]: 1.0, s.st.idx["KC"]: 0.0}
    assert s.st.teams[s._wc_pick(idx(s, "BAL", "HOU", "KC"))] == "BAL"


def test_wc_step0_keeps_only_the_best_club_of_a_division():
    # PIT and BAL (same division) and HOU tied. PIT beat BAL twice, so BAL is out at step 0 even though
    # BAL beat HOU; then PIT vs HOU is a two-club tie (HOU won their game).
    s = league([W("PIT", "BAL"), W("PIT", "BAL"), W("HOU", "PIT"), W("PIT", "PHI"), W("WAS", "PIT"),
                W("BAL", "HOU"), W("BAL", "DAL"), W("BAL", "NYG"),
                W("HOU", "SEA"), W("HOU", "SF"), W("NO", "HOU")])
    assert len({round(s.pct(t), 9) for t in idx(s, "PIT", "BAL", "HOU")}) == 1
    assert [s.st.teams[t] for t in s.division_order("AFC North")][:2] == ["PIT", "BAL"]
    assert s.st.teams[s._wc_pick(idx(s, "PIT", "BAL", "HOU"))] == "HOU"
    # Without step 0, PIT would have won on conference record (2-1 against 1-1 and 1-2).
    assert s.st.teams[s._break(idx(s, "PIT", "BAL", "HOU"), "wc")] == "PIT"


def test_wc_restarts_at_the_sweep_when_three_remain():
    # Four clubs from four divisions. No sweep among four; KC has the worst conference record and drops.
    # With three left the procedure restarts at the sweep, and BUF beat both others. (Continuing to the next
    # steps instead would reach strength of victory, where BAL is best.)
    g = [W("BUF", "BAL"), W("BUF", "HOU"), W("NE", "BUF"), W("NYJ", "BUF"),
         W("BAL", "MIA"), W("BAL", "DAL"), W("PHI", "BAL"),
         W("HOU", "NYJ"), W("HOU", "WAS"), W("NYG", "HOU"),
         W("LV", "KC"), W("KC", "ATL"), W("KC", "CAR"), W("TB", "KC"),
         W("MIA", "ATL"), W("MIA", "CAR"), W("MIA", "SEA"), W("DAL", "NO"), W("DAL", "SF"), W("DAL", "LA")]
    s = league(g)
    four = idx(s, "BUF", "BAL", "HOU", "KC")
    assert len({round(s.pct(t), 9) for t in four}) == 1
    sov = s.s_sov(idx(s, "BUF", "BAL", "HOU"))
    assert max(sov, key=sov.get) == s.st.idx["BAL"]
    assert s.st.teams[s._wc_pick(four)] == "BUF"


def test_n_seeds():
    assert tb.n_seeds(2019) == 6 and tb.n_seeds(2020) == 7 and tb.n_seeds(2002) == 6
    with pytest.raises(ValueError):
        tb.n_seeds(2001)


def test_ties_count_half_a_win():
    s = league([("BUF", "MIA", 10, 10), W("BUF", "NE")])
    i = s.st.idx["BUF"]
    assert s.stats["wins"][i] == 1.5 and s.pct(i) == pytest.approx(0.75)


def test_cancelled_game_is_not_played():
    s = league([W("BUF", "MIA"), ("BUF", "NE", np.nan, np.nan)])
    i = s.st.idx["BUF"]
    assert s.stats["gp"][i] == 1 and s.pct(i) == 1.0


# --------------------------------------------------------------------------- history

def _cached_schedules():
    from nflelo.ml import data as mldata
    years = range(2002, 2026)
    if not all(mldata.cache_path("schedules", y).exists() for y in years):
        return None
    return mldata.load_schedules(years)


def test_reproduces_every_actual_playoff_field_and_seeding():
    sched = _cached_schedules()
    if sched is None:
        pytest.skip("needs the cached nflverse schedules 2002-2025")
    from nflelo.ml.sim import history
    rep = history.reproduce(sched, range(2006, 2026))
    assert len(rep) == 40
    bad = rep[~rep["match"]]
    assert bad.empty, bad.to_string()
