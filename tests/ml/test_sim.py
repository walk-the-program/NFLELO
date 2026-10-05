"""Season simulation: seeded reproducibility, determinism with tau_rest = 0 and certain outcomes, and the bracket."""
import numpy as np
import pandas as pd
import pytest

from nflelo.meta import TEAM_ORDER, division
from nflelo.ml.models import margin as mm
from nflelo.ml.sim import season as sim
from nflelo.ml.sim import tiebreak as tb

SEASON = 2023


def schedule() -> pd.DataFrame:
    """A 14-game synthetic season: division double round-robin, one same-conference division, one other-conference division."""
    divs = sorted({division(t) for t in TEAM_ORDER})
    members = {d: [t for t in TEAM_ORDER if division(t) == d] for d in divs}
    afc, nfc = [d for d in divs if d.startswith("AFC")], [d for d in divs if d.startswith("NFC")]
    games = []
    for d in divs:
        m = members[d]
        for i in range(4):
            for j in range(i + 1, 4):
                games += [(m[i], m[j]), (m[j], m[i])]
    for a, b in [(afc[0], afc[1]), (afc[2], afc[3]), (nfc[0], nfc[1]), (nfc[2], nfc[3])] + list(zip(afc, nfc)):
        for i, x in enumerate(members[a]):
            for j, y in enumerate(members[b]):
                games.append((x, y) if (i + j) % 2 == 0 else (y, x))
    df = pd.DataFrame(games, columns=["home_team", "away_team"])
    df["season"], df["game_type"] = SEASON, "REG"
    df["game_id"] = [f"{SEASON}_{i:03d}" for i in range(len(df))]
    return df


def inputs(done_frac=0.5, tau_strength=None, sigma=13.0, shape="normal", kn=None, certain=False, seed=0):
    g = schedule().sort_values("game_id").reset_index(drop=True)
    st = tb.SeasonStatic(g, SEASON)
    rng = np.random.default_rng(seed)
    strength = (np.arange(32) * 1000.0) if certain else rng.normal(0, 4, 32)
    done = np.arange(len(g)) < int(done_frac * len(g))
    margin = np.where(done, rng.integers(-20, 21, len(g)).astype(float), np.nan)
    res = np.where(done, np.where(margin > 0, 1.0, np.where(margin < 0, 0.0, 0.5)), np.nan)
    mu = np.where(done, np.nan, 1.5 + strength[st.home] - strength[st.away])
    return sim.SimInputs(SEASON, st, res, margin, mu, strength, 1.5, sigma, shape, kn)


def test_seeded_runs_are_reproducible():
    inp = inputs()
    a = sim.simulate(inp, 3000, tau=3.0, seed=7, chunk=1000)
    b = sim.simulate(inp, 3000, tau=3.0, seed=7, chunk=1000)
    pd.testing.assert_frame_equal(a, b)
    c = sim.simulate(inp, 3000, tau=3.0, seed=8, chunk=1000)
    assert not a.equals(c)


def test_probabilities_add_up():
    out = sim.simulate(inputs(), 2000, tau=2.0, seed=1)
    n = tb.n_seeds(SEASON)
    assert out["playoffs"].sum() == pytest.approx(2 * n)
    assert out["division"].sum() == pytest.approx(8)
    assert out["seed1"].sum() == pytest.approx(2)
    assert out["reach_sb"].sum() == pytest.approx(2)
    assert out["win_sb"].sum() == pytest.approx(1)
    assert (out["wins_p10"] <= out["wins_mean"]).all() and (out["wins_mean"] <= out["wins_p90"]).all()
    assert out["wins_mean"].sum() == pytest.approx(len(schedule()))   # one win per game, ties as halves


def test_tau_zero_and_certain_outcomes_are_deterministic():
    inp = inputs(certain=True, sigma=1.0)
    out = sim.simulate(inp, 500, tau=0.0, seed=3)
    for c in ("playoffs", "division", "seed1", "reach_sb", "win_sb"):
        assert set(np.unique(out[c])) <= {0.0, 1.0}, c
    assert (out["wins_p10"] == out["wins_p90"]).all()
    # the same seeding as the tiebreakers on the one possible final table
    st = inp.static
    rem = inp.remaining
    res, mar = inp.res_done.copy(), inp.margin_done.copy()
    res[rem] = np.where(inp.mu[rem] > 0, 1.0, 0.0)
    mar[rem] = inp.mu[rem]
    s = tb.Standings.from_results(st, res, mar)
    seeds = s.seeds()
    field = {st.teams[t] for c in seeds.values() for t in c}
    assert set(out.index[out["playoffs"] == 1.0]) == field
    assert set(out.index[out["seed1"] == 1.0]) == {st.teams[seeds[c][0]] for c in seeds}
    # strengths rise with the team index, so the strongest playoff team wins every playoff game
    best = max(field, key=lambda t: inp.strength[st.idx[t]])
    assert out.loc[best, "win_sb"] == 1.0
    # a different seed changes nothing when nothing is random
    pd.testing.assert_frame_equal(out, sim.simulate(inp, 500, tau=0.0, seed=99))


def test_bracket_pairings_and_home_field(monkeypatch):
    """Wild card 2v7, 3v6, 4v5 (higher seed at home); #1 hosts the lowest seed left; Super Bowl neutral."""
    calls = []

    def fake_play(inp, h, a, shocks, neutral, rng):
        calls.append((h.copy(), a.copy(), neutral))
        return h.copy()                                       # the home (higher) seed always wins
    monkeypatch.setattr(sim, "_play", fake_play)
    inp = inputs()
    seeds = np.array([[[10, 11, 12, 13, 14, 15, 16], [20, 21, 22, 23, 24, 25, 26]]])
    champs, sb = sim._bracket(inp, seeds, np.zeros((1, 32)), np.random.default_rng(0))
    afc = calls[:6]
    assert [(int(h[0]), int(a[0])) for h, a, _ in afc[:3]] == [(11, 16), (12, 15), (13, 14)]
    assert (int(afc[3][0][0]), int(afc[3][1][0])) == (10, 13)          # #1 hosts #4, the lowest left
    assert (int(afc[4][0][0]), int(afc[4][1][0])) == (11, 12)
    assert (int(afc[5][0][0]), int(afc[5][1][0])) == (10, 11)
    assert champs.tolist() == [[10, 20]] and calls[-1][2] is True and not any(c[2] for c in calls[:-1])
    calls.clear()

    def upsets(inp, h, a, shocks, neutral, rng):
        calls.append((h.copy(), a.copy(), neutral))
        return a.copy()                                       # the visitor always wins
    monkeypatch.setattr(sim, "_play", upsets)
    sim._bracket(inp, seeds, np.zeros((1, 32)), np.random.default_rng(0))
    # wild-card winners are seeds 7, 6, 5; #1 hosts #7, then #5 hosts #6
    assert (int(calls[3][0][0]), int(calls[3][1][0])) == (10, 16)
    assert (int(calls[4][0][0]), int(calls[4][1][0])) == (14, 15)


def test_six_seed_bracket_before_2020(monkeypatch):
    calls = []
    monkeypatch.setattr(sim, "_play", lambda inp, h, a, s, n, r: (calls.append((h.copy(), a.copy())), h.copy())[1])
    inp = inputs()
    seeds = np.array([[[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]]])
    sim._bracket(inp, seeds, np.zeros((1, 32)), np.random.default_rng(0))
    assert [(int(h[0]), int(a[0])) for h, a in calls[:2]] == [(12, 15), (13, 14)]   # 3v6, 4v5; 1 and 2 rest
    assert [(int(h[0]), int(a[0])) for h, a in calls[2:4]] == [(10, 13), (11, 12)]


def test_shocks_widen_the_odds():
    inp = inputs(done_frac=0.25)
    a = sim.simulate(inp, 4000, tau=0.0, seed=5)
    b = sim.simulate(inp, 4000, tau=6.0, seed=5)
    spread = lambda o: float(((o["playoffs"] - 0.5).abs()).mean())  # noqa: E731
    assert spread(b) < spread(a)                              # less certain with shocks
    assert (b["wins_p90"] - b["wins_p10"]).mean() > (a["wins_p90"] - a["wins_p10"]).mean()


def test_keynum_shape_runs_and_ties_count_half():
    kn = mm.KeyNumbers(np.where(np.arange(mm.KMAX + 1) == 0, 40.0, 1.0))   # ties common enough to show up
    inp = inputs(shape="keynum", kn=kn, done_frac=0.0)
    out = sim.simulate(inp, 1000, tau=1.0, seed=2)
    assert out["wins_mean"].sum() == pytest.approx(len(schedule()))
    assert (np.round(out["wins_p10"] * 2) % 2 == 1).any() or (np.round(out["wins_p90"] * 2) % 2 == 1).any()
