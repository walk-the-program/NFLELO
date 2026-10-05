"""M5b on-field ratings (CC BY-SA 4.0 code paths): participation cleaning, the RAPM solver, leakage.

Offline and synthetic. Two kinds of synthetic data:
- `lineup_plays`: plays already in the cleaned format (participation.PLAY_COLUMNS),
  with known player effects, for the solver tests;
- `raw_world`: the conftest schedule and play-by-play plus a raw participation
  table and a players table, for the cleaning and leakage tests (the full
  path from raw files to season-ahead predictions).
"""
import numpy as np
import pandas as pd
import pytest

from nflelo.ml.players import participation as pt
from nflelo.ml.players import rapm


GROUP_OFF = ["QB"] + ["RB"] * 2 + ["WR"] * 4 + ["TE"] * 2 + ["OL"] * 6   # 15 offensive players per team
GROUP_DEF = ["DL"] * 6 + ["LB"] * 4 + ["CB"] * 3 + ["S"] * 2               # 15 defensive players per team


def _pick(rng, n_total, k, core):
    """k of n_total roster spots: the `core` always, the rest drawn from the others."""
    others = [i for i in range(n_total) if i not in core]
    return list(core) + list(rng.choice(others, k - len(core), replace=False))


def lineup_plays(n_plays=40000, n_teams=8, seasons=(2020,), seed=0, together=False, true=None):
    """Cleaned-format plays with known effects. Returns (plays, true effects keyed 'off|id'/'def|id', fixed)."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    if true is None:
        true = {}
        for t in teams:
            for i in range(15):
                true[f"off|{t}_o{i}"] = rng.normal(0, 0.08)
                true[f"def|{t}_d{i}"] = rng.normal(0, 0.08)
        if together:
            true["off|T0_o13"], true["off|T0_o14"] = 0.30, 0.0
            true["off|T0_o11"], true["off|T0_o12"] = 0.0, 0.0
    fixed = {"intercept": 0.05, "home": 0.03, "down3": -0.10, "pass": 0.08}
    rows = []
    per = n_plays // len(seasons)
    for S in seasons:
        for k in range(per):
            a, b = rng.choice(n_teams, 2, replace=False)
            ot, dt = teams[a], teams[b]
            if together and ot == "T0":
                # o13 and o14 always together; o11 and o12 always together; one pair is on each play
                pair = [13, 14] if rng.uniform() < 0.5 else [11, 12]
                rest = list(rng.choice([i for i in range(11)], 9, replace=False))
                o = rest + pair
            else:
                o = _pick(rng, 15, 11, core=[])
            d = _pick(rng, 15, 11, core=[])
            oid = sorted(f"{ot}_o{i}" for i in o)
            did = sorted(f"{dt}_d{i}" for i in d)
            down = int(rng.integers(1, 5))
            is_pass = float(rng.uniform() < 0.55)
            home = float(rng.uniform() < 0.5)
            mu = (fixed["intercept"] + fixed["home"] * home + fixed["down3"] * (down == 3)
                  + fixed["pass"] * is_pass + sum(true[f"off|{x}"] for x in oid) + sum(true[f"def|{x}"] for x in did))
            rows.append([f"{S}_{k // 120:03d}_{ot}_{dt}", k, S, 1 + (k // 120) % 17, "REG", k, ot, dt, home,
                         mu + rng.normal(0, 1.0), down, 8.0, 50.0, is_pass, "raw"] + oid + did)
    plays = pd.DataFrame(rows, columns=pt.PLAY_COLUMNS)
    return plays, true, fixed


def priors_from(true: dict, keys=(), groups=None, scale=0.0) -> pd.DataFrame:
    """Priors table: every player with his group; v_box = scale * true value (0 = replacement)."""
    ids, grp, v = [], [], []
    for k, val in true.items():
        side, pid = k.split("|")
        idx = int(pid.split("_")[1][1:])
        g = GROUP_OFF[idx] if side == "off" else GROUP_DEF[idx]
        ids.append(pid)
        grp.append(groups.get(pid, g) if groups else g)
        v.append(scale * (val if side == "off" else -val))
    return pd.DataFrame({"group": grp, "v_box": v}, index=pd.Index(ids, name="gsis_id"))


# --------------------------------------------------------------------------- solver

@pytest.fixture(scope="module")
def recovery():
    plays, true, fixed = lineup_plays(n_plays=60000, seasons=(2020, 2021), seed=1)
    cfg = rapm.RapmConfig(lam=tuple((g, 20.0) for g in rapm.GROUPS), decay=1.0, box_scale=0.0)
    data = rapm.RapmData(plays)
    fit = data.fit(2022, priors_from(true), cfg)
    return plays, true, fixed, data, fit, cfg


def test_rapm_recovers_known_effects(recovery):
    plays, true, fixed, data, fit, cfg = recovery
    r = pd.Series(fit.beta, index=fit.keys)
    t = pd.Series(true).reindex(r.index)
    # mu absorbs the mean of the eleven effects per side, so compare centred values per side and team
    for side in ("off", "def"):
        m = r.index.str.startswith(side)
        corr = np.corrcoef(r[m], t[m])[0, 1]
        assert corr > 0.9, (side, corr)
        assert np.abs((r[m] - r[m].mean()) - (t[m] - t[m].mean())).max() < 0.08
    assert fit.gamma["down3"] == pytest.approx(fixed["down3"], abs=0.03)
    assert fit.gamma["pass"] == pytest.approx(fixed["pass"], abs=0.03)
    assert fit.gamma["home"] == pytest.approx(fixed["home"], abs=0.03)
    assert fit.sigma2 == pytest.approx(1.0, abs=0.03)


def test_cg_matches_direct_and_fast_prediction(recovery):
    plays, true, fixed, data, fit, cfg = recovery
    direct = data.fit(2022, priors_from(true), cfg, solver="direct")
    assert np.abs(direct.beta - fit.beta).max() < 1e-6
    rows = np.arange(500)
    assert np.allclose(data.predict_rows(fit, rows), fit.predict(plays.iloc[rows]), atol=1e-10)


def test_priors_split_always_together_players():
    plays, true, _ = lineup_plays(n_plays=40000, seed=2, together=True)
    data = rapm.RapmData(plays)
    cfg = rapm.RapmConfig(lam=tuple((g, 300.0) for g in rapm.GROUPS), decay=1.0, box_scale=1.0)
    pr = priors_from(true, scale=0.0)                     # every prior at replacement
    flat = data.fit(2021, pr, cfg, sd=True)
    r = pd.Series(flat.beta, index=flat.keys)
    a, b = r["off|T0_o13"], r["off|T0_o14"]
    # data identify only the pair's sum relative to the other pair; equal priors split it evenly
    assert a == pytest.approx(b, abs=1e-9)
    gap = (a + b) - (r["off|T0_o11"] + r["off|T0_o12"])
    assert gap == pytest.approx(0.30, abs=0.06)
    # an informative prior on o13 (box value 0.30) puts the effect on him
    pr2 = pr.copy()
    pr2.loc["T0_o13", "v_box"] = 0.30
    inf = data.fit(2021, pr2, cfg)
    r2 = pd.Series(inf.beta, index=inf.keys)
    assert r2["off|T0_o13"] - r2["off|T0_o14"] > 0.2
    assert ((r2["off|T0_o13"] + r2["off|T0_o14"]) - (r2["off|T0_o11"] + r2["off|T0_o12"])) == pytest.approx(0.30,
                                                                                                         abs=0.06)
    # the inseparable pair is more uncertain than a rotating teammate with similar snaps
    sd = pd.Series(flat.sd, index=flat.keys)
    assert sd["off|T0_o13"] > 1.3 * sd["off|T0_o3"]


def test_unseen_players_keep_their_prior_and_chaining_decays():
    plays, true, _ = lineup_plays(n_plays=30000, seasons=(2020, 2021), seed=3)
    pr = priors_from(true, scale=0.5)
    cfg = rapm.RapmConfig(lam=tuple((g, 500.0) for g in rapm.GROUPS), decay=0.5, box_scale=1.0)
    fit = rapm.season_ahead(plays[plays["season"] == 2020], 2021, pr, cfg)
    newp = plays[plays["season"] == 2021].head(3).copy()
    newp[pt.OFF_COLS[0]] = "ROOKIE"
    pr2 = pd.concat([pr, pd.DataFrame({"group": ["WR"], "v_box": [0.2]}, index=["ROOKIE"])])
    fit.priors = pr2
    base = fit.predict(plays[plays["season"] == 2021].head(3))
    b0 = pd.Series(fit.beta, index=fit.keys)
    old = plays[plays["season"] == 2021].head(3)[pt.OFF_COLS[0]].map(lambda x: b0[f"off|{x}"]).to_numpy()
    assert np.allclose(fit.predict(newp), base - old + 0.2)
    # decay 0 uses only the last season; decay 1 weights both equally
    data = rapm.RapmData(plays)
    used0 = data.fit(2022, pr, rapm.RapmConfig(decay=0.0)).solver_info["used"]
    used5 = data.fit(2022, pr, rapm.RapmConfig(decay=0.5)).solver_info["used"]
    assert used0 == [(2021, 1.0)] and used5 == [(2020, 0.5), (2021, 1.0)]


# --------------------------------------------------------------------------- raw participation world

def raw_world(sched, pbp, seed=5):
    """Schedule, play-by-play, participation and players tables from the conftest world."""
    rng = np.random.default_rng(seed)
    sched, pbp = sched.copy(), pbp.copy()
    pbp["down"] = rng.integers(1, 5, len(pbp)).astype(float)
    pbp["ydstogo"] = rng.integers(1, 15, len(pbp)).astype(float)
    pbp["yardline_100"] = rng.integers(1, 99, len(pbp)).astype(float)
    teams = sorted(set(sched["home_team"]))
    roster = {t: ([f"00-{t}O{i:02d}" for i in range(15)], [f"00-{t}D{i:02d}" for i in range(15)]) for t in teams}
    allp = [p for t in teams for side in roster[t] for p in side]
    players = pd.DataFrame({"gsis_id": allp, "nfl_id": [str(50000 + i) for i in range(len(allp))],
                            "position": "WR"})
    g2n = dict(zip(players["gsis_id"], players["nfl_id"]))
    eff = {p: rng.normal(0, 0.1) for p in allp}
    part_rows, epa = [], pbp["epa"].to_numpy().copy()
    for i, r in enumerate(pbp.itertuples(index=False)):
        o = sorted(rng.choice(roster[r.posteam][0], 11, replace=False))
        d = sorted(rng.choice(roster[r.defteam][1], 11, replace=False))
        epa[i] = epa[i] * 0.5 + sum(eff[x] for x in o) - sum(eff[x] for x in d)
        on = [g2n[x] for x in o + d] if r.season == 2022 else o + d   # nfl_ids before the "FTN" season
        ol, dl = list(o), list(d)
        u = rng.uniform()
        if u < 0.04:
            ol = ol[1:]                     # one offensive player without a GSIS ID
        elif u < 0.08:
            ol = ol + ["00-BOGUS"]          # a 12th ID that is not on the play
        part_rows.append({"nflverse_game_id": r.game_id, "play_id": r.play_id, "possession_team": r.posteam,
                          "players_on_play": ";".join(on), "offense_players": ";".join(ol),
                          "defense_players": ";".join(dl), "truth_o": o, "truth_d": d})
    pbp["epa"] = epa
    part = pd.DataFrame(part_rows)
    return sched, pbp, part, players


@pytest.fixture(scope="module")
def world(synth_sched, synth_pbp):
    return raw_world(synth_sched, synth_pbp)


def test_cleaning_repairs_sides_exactly(world):
    sched, pbp, part, players = world
    plays = pt.build_plays(pbp, part.drop(columns=["truth_o", "truth_d"]), sched, players)
    rep = pt.cleaning_report(plays)
    assert (rep["drop_rate"] == 0).all() and (rep["raw_drop_rate"] > 0.05).all()
    truth = part.set_index(["nflverse_game_id", "play_id"])
    t = truth.loc[list(zip(plays["game_id"], plays["play_id"].astype(float)))]
    assert all(sorted(a) == list(b) for a, b in zip(t["truth_o"], plays[pt.OFF_COLS].to_numpy()))
    assert all(sorted(a) == list(b) for a, b in zip(t["truth_d"], plays[pt.DEF_COLS].to_numpy()))
    assert set(plays["source"]) == {"raw", "repaired"}
    # garbage time and non-scrimmage plays never enter
    assert plays["epa"].notna().all() and len(plays) < len(pbp)
    X = pt.design(plays, pt.player_index(plays))
    assert (np.asarray(X.sum(axis=1)).ravel() == 22).all()


def _corrupt_raw(pbp, part, sched, S, seed=0):
    """Scramble season S onward in the raw files: EPA, situation and every on-field ID list."""
    rng = np.random.default_rng(seed)
    p, q = pbp.copy(), part.copy()
    late = (p["season"] >= S).to_numpy()
    for c in ("epa", "down", "ydstogo", "yardline_100"):
        p.loc[late, c] = rng.permutation(p.loc[late, c].to_numpy())
    p.loc[late, "epa"] += rng.normal(0, 1, late.sum())
    ql = q["nflverse_game_id"].str.slice(0, 4).astype(int).to_numpy() >= S
    for c in ("players_on_play", "offense_players", "defense_players"):
        q.loc[ql, c] = rng.permutation(q.loc[ql, c].to_numpy())
    return p, q


def test_season_ahead_is_leak_free_and_the_control_fails(world):
    sched, pbp, part, players = world
    part = part.drop(columns=["truth_o", "truth_d"])
    S = 2023
    clean = pt.build_plays(pbp, part, sched, players)
    p2, q2 = _corrupt_raw(pbp, part, sched, S)
    dirty = pt.build_plays(p2, q2, sched, players)
    pr = pd.DataFrame({"group": "WR", "v_box": 0.0}, index=pd.Index(sorted(players["gsis_id"]), name="gsis_id"))
    cfg = rapm.RapmConfig(lam=tuple((g, 200.0) for g in rapm.GROUPS))
    ok = rapm.leakage_check(clean, dirty, S, pr, cfg)
    assert ok["leak_free"] and ok["plays"] > 0, ok
    bad = rapm.leakage_check(clean, dirty, S, pr, cfg, leaky=True)
    assert not bad["leak_free"] and bad["changed"] > 0.9 * bad["plays"], bad
    # the plays-level corruption helper behaves the same way
    ok2 = rapm.leakage_check(clean, rapm.corrupt_plays(clean, S), S, pr, cfg)
    assert ok2["leak_free"]


def test_m5b_outputs_carry_the_share_alike_license():
    plays, true, _ = lineup_plays(n_plays=3000, seed=4)
    fit = rapm.RapmData(plays).fit(2021, priors_from(true), rapm.RapmConfig())
    assert "CC BY-SA 4.0" in fit.ratings().attrs["license"]
    for mod in (pt, rapm):
        assert "CC BY-SA 4.0" in mod.__doc__


def test_cluster_bootstrap_resamples_whole_games():
    from nflelo.ml.eval import bootstrap
    rng = np.random.default_rng(0)
    games = np.repeat(np.arange(200), 50)
    shared = rng.normal(0, 1, 200)[games]           # rows inside a game move together
    d = shared + rng.normal(0, 0.1, len(games))
    c = bootstrap.cluster_bootstrap_diff(d, games, reps=500, seed=1)
    naive = bootstrap.paired_bootstrap_diff(d, reps=500, seed=1)
    assert c["clusters"] == 200 and c["diff"] == pytest.approx(d.mean())
    assert c["boot_se"] > 4 * naive["boot_se"]       # ignoring the clustering would understate the error
    one = bootstrap.cluster_bootstrap_diff(d[:200], np.arange(200), reps=300, seed=2)
    assert one["boot_se"] == pytest.approx(bootstrap.paired_bootstrap_diff(d[:200], reps=300, seed=2)["boot_se"],
                                           rel=0.25)
