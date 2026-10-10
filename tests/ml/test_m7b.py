"""M7b early-down play type x personnel (CC BY-SA 4.0 code paths): actions, cells, tendencies, AIPW, policy,
OPE, placebo, season-ahead leakage, license boundary.

Offline and synthetic: a small simulated league (four teams, 2016-2019) whose play caller picks heavy personnel
near the goal line (where every play gains less) and passes more when trailing, with a known effect per action.
"""
import ast
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nflelo import config
from nflelo.ml import registry
from nflelo.ml.decisions import policy as pol
from nflelo.ml.eval import windows

sys.path.insert(0, str(config.ROOT / "scripts"))
warnings.filterwarnings("ignore", message="X does not have valid feature names")

ROOT = Path(__file__).resolve().parents[2]
TEAMS = ["T0", "T1", "T2", "T3"]
PERS = {"11": (1, 1, 3), "12": (1, 2, 2), "13": (1, 3, 1), "21": (2, 1, 2), "10": (1, 0, 4)}
# true effect of each called action on EPA (situation effects are separate): passing is worth +0.3
TAU = {"11_run": 0.0, "11_pass": 0.3, "12_run": 0.0, "12_pass": 0.3, "13_run": 0.0}
USED = list(TAU)


def sim_table(seed=0, seasons=(2016, 2017, 2018, 2019), games=80, plays=70):
    rng = np.random.default_rng(seed)
    team_bias = {t: b for t, b in zip(TEAMS, (-0.6, -0.2, 0.2, 0.6))}   # persistent pass tendency
    rows = []
    for S in seasons:
        for gi in range(games):
            week = gi // 2 + 1
            a, b = rng.choice(4, 2, replace=False)
            gid = f"{S}_{week:02d}_{TEAMS[a]}_{TEAMS[b]}_{gi}"
            n = plays
            off = np.where(np.arange(n) % 2 == 0, TEAMS[a], TEAMS[b])
            de = np.where(np.arange(n) % 2 == 0, TEAMS[b], TEAMS[a])
            down = rng.choice([1, 2, 3], n, p=[0.45, 0.35, 0.2]).astype(float)
            ytg = np.where(down == 1, np.where(rng.uniform(size=n) < 0.9, 10, rng.integers(1, 10, n)),
                           rng.integers(1, 15, n)).astype(float)
            yl = rng.integers(1, 100, n).astype(float)
            sd = rng.integers(-14, 15, n).astype(float)
            hs = rng.integers(0, 1800, n).astype(float)
            rz = yl <= 20
            bias = np.array([team_bias[t] for t in off])
            # logits: 11_run, 11_pass, 12_run, 12_pass, 13_run (heavy at the goal line; pass when trailing)
            L = np.stack([np.zeros(n), 0.2 + bias - 0.05 * sd, -0.3 + 0.0 * bias, -0.4 + bias - 0.05 * sd,
                          -2.5 + 3.0 * rz], axis=1)
            P = np.exp(L) / np.exp(L).sum(axis=1, keepdims=True)
            k = np.array([rng.choice(5, p=p) for p in P])
            act = np.array(USED)[k]
            grp = np.array([x.split("_")[0] for x in act])
            base = -0.8 * rz - 0.15 * (down == 2) + 0.02 * (yl < 50)
            epa = base + np.array([TAU[x] for x in act]) + rng.normal(0, 1.0, n)
            rows.append(pd.DataFrame({
                "game_id": gid, "play_id": np.arange(1, n + 1), "season": S, "week": week, "season_type": "REG",
                "off_team": off, "def_team": de, "down": down, "ydstogo": ytg, "yardline_100": yl,
                "score_diff": sd, "half_secs": hs, "game_secs": hs + 1800 * rng.integers(0, 2, n),
                "off_timeouts": 3.0, "def_timeouts": 3.0, "home": (off == TEAMS[b]).astype(float),
                "roof_indoor": 0.0, "roof_open": 0.0, "temp": np.where(rng.uniform(size=n) < 0.2, np.nan, 60.0),
                "wind": 5.0, "off_rb": [PERS[g][0] for g in grp], "off_te": [PERS[g][1] for g in grp],
                "off_wr": [PERS[g][2] for g in grp], "off_ol": 5.0,
                "is_pass": np.char.endswith(act.astype(str), "pass").astype(float),
                "off_pass_o": 0.0, "off_rush_o": 0.0, "def_pass_d": rng.normal(0, 0.05), "def_rush_d": 0.0,
                "def_dl": 4.0, "def_lb": 2.0, "def_db": 5.0, "form_shotgun": 1.0, "form_pistol": 0.0,
                "form_under_center": 0.0, "box": 6.0, "epa": epa}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def league():
    t = sim_table()
    return {"table": t, "frame": pol.prepare(t)}


@pytest.fixture(scope="module")
def work19(league):
    return pol.season_work(league["frame"], 2019, ("main",), reps=200)


# --------------------------------------------------------------------------- definitions

def test_actions_cells_and_sample():
    t = pd.DataFrame({"off_rb": [1, 1, 2, 1, 1, 2, 1, 0], "off_te": [1, 2, 1, 3, 0, 2, 1, 1],
                      "off_wr": [3, 2, 2, 1, 4, 1, 2, 4], "off_ol": [5, 5, 5, 5, 5, 5, 6, 5],
                      "is_pass": [1, 0, 1, 0, 1, 0, 0, 1]})
    assert list(pol.personnel_group(t)) == ["11", "12", "21", "13", "10", "oth", "oth", "oth"]
    assert [pol.ACTIONS[i] for i in pol.action_codes(t)] == ["11_pass", "12_run", "21_pass", "13_run", "10_pass",
                                                             "oth_run", "oth_run", "oth_pass"]
    assert len(pol.ACTIONS) == 12 and len(pol.CELLS) == 60
    s = pd.DataFrame({"down": [1, 1, 2, 2, 2, 1], "ydstogo": [10, 3, 2, 5, 12, 15],
                      "yardline_100": [85, 5, 60, 30, 75, 50], "score_diff": [0, -7, 3, 10, -3, 4]})
    assert list(pol.cell_labels(s)) == ["1-10+ | own 1-20 | within 3", "1-short | red zone | trail 4+",
                                        "2-short | own 21-50 | within 3", "2-mid | opp 49-21 | lead 4+",
                                        "2-long | own 21-50 | within 3", "1-10+ | own 21-50 | lead 4+"]
    m = pd.DataFrame({"down": [1, 2, 3, 1, 2], "half_secs": [600, 121, 600, 120, 30]})
    assert list(pol.sample_mask(m)) == [True, True, False, False, False]


def test_features_are_pre_snap_and_never_the_structure():
    assert pol.check_features(pol.FEATURES) == pol.FEATURES
    assert not set(pol.FEATURES) & pol.MEDIATORS
    for bad in ("off_te", "box", "def_db", "form_shotgun"):          # the action itself or downstream of it
        with pytest.raises(ValueError, match="never features"):
            pol.check_features(pol.FEATURES + [bad])
    for bad in ("epa", "success", "yards_gained", "was_pressure"):
        with pytest.raises(ValueError, match="banned"):
            pol.check_features(pol.FEATURES + [bad])
    with pytest.raises(ValueError, match="outside"):
        pol.check_features(pol.FEATURES + ["off_team_code"])


def test_tendencies_are_as_of_the_week_start(league):
    f = league["frame"]
    S, w = 2018, 6
    g = f.copy()
    m = ((g["season"] == S) & (g["week"] >= w)).to_numpy()
    g.loc[m, "action"] = np.random.default_rng(1).permutation(g.loc[m, "action"].to_numpy())
    g = pol.with_tendencies(g)
    keep = ~((f["season"] == S) & (f["week"] > w)).to_numpy() & ~(f["season"] > S).to_numpy()
    assert np.array_equal(f.loc[keep, pol.TENDENCY].to_numpy(), g.loc[keep, pol.TENDENCY].to_numpy())
    nxt = ((f["season"] == S) & (f["week"] == w + 1)).to_numpy()
    assert np.abs(f.loc[nxt, pol.TENDENCY].to_numpy() - g.loc[nxt, pol.TENDENCY].to_numpy()).max() > 1e-6
    # week 1 = the prior: last season's team mix shrunk halfway to last season's league mix
    T = "T0"
    r = f[(f["season"] == S) & (f["week"] == 1) & (f["off_team"] == T)]
    if len(r):
        prev = f[f["season"] == S - 1]
        team = np.bincount(prev.loc[prev["off_team"] == T, "action"], minlength=12) / (prev["off_team"] == T).sum()
        lg = np.bincount(prev["action"], minlength=12) / len(prev)
        assert np.allclose(r[pol.TENDENCY].iloc[0].to_numpy(), 0.5 * team + 0.5 * lg)
    # the tendency rows sum to one
    assert np.allclose(f[pol.TENDENCY].sum(axis=1), 1.0)


def test_pseudo_outcome_identities():
    rng = np.random.default_rng(0)
    n = 500
    e = rng.dirichlet(np.ones(12), n)
    mu = rng.normal(0, 0.2, (n, 12))
    a = np.array([rng.choice(12, p=p) for p in e])
    y = rng.normal(0, 1, n)
    g = pol.pseudo(e, mu, a, y)
    rows = np.arange(n)
    assert np.allclose(g[rows, a], mu[rows, a] + (y - mu[rows, a]) / e[rows, a])
    other = np.ones((n, 12), bool)
    other[rows, a] = False
    assert np.allclose(g[other], mu[other])
    # observed policy's DR value = sum_a e_a mu_a + (y - mu_A)
    assert np.allclose(pol.observed_value(e, g), (e * mu).sum(axis=1) + y - mu[rows, a])
    # trimming floors the propensity in the IPW term only
    gt = pol.pseudo(e, mu, a, y, floor=0.5)
    assert np.allclose(gt[rows, a], mu[rows, a] + (y - mu[rows, a]) / np.maximum(e[rows, a], 0.5))
    # a policy that keeps every call contributes nothing; ope_diff is G - y where it applies
    assert np.all(pol.ope_diff(np.full(n, -1), g, y) == 0)
    rec = np.where(rows % 2 == 0, 1, -1)
    d = pol.ope_diff(rec, g, y)
    assert np.allclose(d[rows % 2 == 0], g[rows % 2 == 0, 1] - y[rows % 2 == 0]) and np.all(d[rows % 2 == 1] == 0)


# --------------------------------------------------------------------------- causal estimates

def test_aipw_removes_the_goal_line_confounding(league, work19):
    """Heavy personnel (13) is called near the goal line, where every play gains less. The raw comparison
    makes it look bad; the cross-fit AIPW contrast with 11 personnel runs is close to the true 0."""
    f = league["frame"]
    tr = f[f["season"] < 2019].reset_index(drop=True)
    i13, i11 = pol.ACTIONS.index("13_run"), pol.ACTIONS.index("11_run")
    a, y = tr["action"].to_numpy(), tr["y_epa"].to_numpy()
    naive = y[a == i13].mean() - y[a == i11].mean()
    oof = work19["oof"]
    g = pol.pseudo(oof["e"], oof["mu_epa"], a, y)
    both = (oof["e"][:, i13] >= pol.OVERLAP) & (oof["e"][:, i11] >= pol.OVERLAP)
    dr = (g[both, i13] - g[both, i11]).mean()
    assert naive < -0.25
    assert abs(dr) < 0.15, dr
    # the pass effect (+0.3 for 11 personnel) is recovered too
    i11p = pol.ACTIONS.index("11_pass")
    both = (oof["e"][:, i11p] >= pol.OVERLAP) & (oof["e"][:, i11] >= pol.OVERLAP)
    assert abs((g[both, i11p] - g[both, i11]).mean() - 0.3) < 0.1


def test_policy_and_ope_find_the_true_gain(league, work19):
    v = work19["variants"]["main"]
    recs = v["recs"]
    assert recs["clear"].any()
    assert set(recs.loc[recs["clear"], "best"]) <= {"11_pass", "12_pass"}
    test = work19["test"]
    rec, a = v["rec"], test["action"].to_numpy()
    tau = np.array([TAU.get(x, 0.0) for x in pol.ACTIONS])
    truth = np.where(rec >= 0, tau[np.maximum(rec, 0)] - tau[a], 0.0).mean()
    est = v["d_epa"].mean()
    assert truth > 0.02 and abs(est - truth) < 0.05, (est, truth)
    # the observed policy's DR value matches the observed mean
    assert abs(v["obs_epa"].mean() - test["y_epa"].mean()) < 0.02
    # every recommended action is eligible where the policy applies
    on = rec >= 0
    assert np.all(work19["pred"]["e"][np.flatnonzero(on), rec[on]] >= pol.OVERLAP)


def test_placebo_shows_no_gain(league):
    fp = pol.shuffle_within_cells(league["frame"], seed=3)
    f = league["frame"]
    # actions only move within (season, cell)
    for k in ("season", "cell"):
        assert fp[k].equals(f[k])
    assert (pd.crosstab([f["season"], f["cell"]], f["action"]).to_numpy()
            == pd.crosstab([fp["season"], fp["cell"]], fp["action"]).to_numpy()).all()
    w = pol.season_work(fp, 2019, ("main",), reps=200)
    d = w["variants"]["main"]["d_epa"]
    assert abs(d.mean()) < 0.03, d.mean()


def test_cell_table_and_recommend():
    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame({"cell": pol.CELLS[0], "game_id": [f"g{i // 40}" for i in range(n)],
                       "action": rng.choice([0, 1], n)})
    e = np.zeros((n, 12))
    e[:, 0], e[:, 1] = 0.5, 0.5
    y = rng.normal(0, 1, n)
    g = np.repeat(y[:, None], 12, axis=1)
    g[:, 1] += 0.5                                  # action 1 is worth +0.5 over the current mix
    cells = pol.cell_table(df, e, g, y, reps=300)
    c = cells.set_index("action")
    assert c.loc["11_pass", "candidate"] and c.loc["11_run", "candidate"] and not c.loc["12_run", "candidate"]
    assert np.isclose(c.loc["11_pass", "gain"], 0.5) and c.loc["11_pass", "gain_lo"] > 0.4
    r = pol.recommend(cells)
    assert r.iloc[0]["best"] == "11_pass" and r.iloc[0]["clear"]
    # no gain anywhere: no clear best
    r0 = pol.recommend(pol.cell_table(df, e, np.repeat(y[:, None], 12, axis=1), y, reps=300))
    assert not r0.iloc[0]["clear"]
    # the policy only applies where the recommended action is eligible
    e2 = e.copy()
    e2[:10, 1] = 0.01
    rec = pol.apply_policy(df, r, e2)
    assert np.all(rec[:10] == -1) and np.all(rec[10:] == 1)


# --------------------------------------------------------------------------- leakage

def test_season_ahead_leakage(league, work19):
    """Corrupt season S (all outcomes; calls and downstream structure from week w): the S nuisance
    predictions and recommendations for week w are unchanged; a fit through S moves."""
    import ml_m7b
    S, w = 2019, 9
    t = league["table"]
    fc = pol.prepare(ml_m7b.corrupt_table(t, S, w, seed=0))
    assert fc[["game_id", "play_id"]].equals(league["frame"][["game_id", "play_id"]])
    assert (fc.loc[fc["season"] == S, "y_epa"].to_numpy() != league["frame"].loc[league["frame"]["season"] == S,
                                                                                 "y_epa"].to_numpy()).mean() > 0.9
    cw = pol.season_work(fc, S, ("main",), reps=200)
    tm = (work19["test"]["week"] == w).to_numpy()
    for k in ("e", "mu_epa", "mu_succ"):
        assert np.array_equal(work19["pred"][k][tm], cw["pred"][k][tm])
    assert work19["variants"]["main"]["recs"].fillna(-9).equals(cw["variants"]["main"]["recs"].fillna(-9))
    nxt = (work19["test"]["week"] == w + 1).to_numpy()
    assert np.abs(work19["test"].loc[nxt, pol.TENDENCY].to_numpy()
                  - cw["test"].loc[nxt, pol.TENDENCY].to_numpy()).max() > 0
    through = pol.fit_nuisance(fc[fc["season"] <= S].reset_index(drop=True))
    pc = pol.predict_nuisance(through, cw["test"][tm])
    assert np.abs(pc["mu_epa"] - work19["pred"]["mu_epa"][tm]).max() > 1e-6


def test_m7b_dev_window_is_guarded(tmp_path):
    assert windows.M7B_DEV == (2018, 2019) and not windows.touches_holdout(windows.M7B_DEV)
    with pytest.raises(windows.HoldoutError):
        registry.log_run("x", label="m7bdev", seasons=(2019, 2020), game_ids=["a"], metrics={"n": 1},
                         data={}, runs_dir=tmp_path)


def test_signoff_refuses_without_a_reproduced_dry_run(tmp_path, monkeypatch):
    import ml_m7b
    monkeypatch.setattr(ml_m7b, "OUT", tmp_path)
    # isolate from the real registry (a recorded holdout sign-off would trip the run-once check first)
    monkeypatch.setattr(ml_m7b, "holdout_signoff_runs", lambda: [])
    with pytest.raises(SystemExit):
        ml_m7b.stage_signoff(ml_m7b.HOLDOUT, allow_holdout=True, log_=False)     # no dev.json
    (tmp_path / "dev.json").write_text('{"primary": {}}')
    with pytest.raises(SystemExit, match="dry run"):
        ml_m7b.stage_signoff(ml_m7b.HOLDOUT, allow_holdout=True, log_=False)
    (tmp_path / "signoff_m7bdev.json").write_text('{"reproduced": true, "git": {"sha": "abc", "dirty": true}}')
    with pytest.raises(SystemExit, match="clean tree"):
        ml_m7b.stage_signoff(ml_m7b.HOLDOUT, allow_holdout=True, log_=False)
    monkeypatch.setattr(ml_m7b, "holdout_signoff_runs", lambda: ["x_m7b_signoff_ope.json"])
    with pytest.raises(SystemExit, match="run once only"):
        ml_m7b.stage_signoff(ml_m7b.HOLDOUT, allow_holdout=True, log_=False)


M7B_FILES = {"nflelo/ml/decisions/policy.py", "scripts/ml_m7b.py"}
# The model-page exporter publishes CC BY-SA pages (plays, fourth, playcalling) and one CC BY page (winprob);
# tests/test_export_ml_pages.py checks that the winprob code path never touches M6, M7b or the fourth-down module.
SITE_EXPORTERS = {"scripts/export_ml_pages.py"}


def test_m7b_license_boundary():
    """M7b is CC BY-SA 4.0 (participation personnel): its files say so, and nothing outside M7b imports the
    policy module (in particular not the CC BY WP and kicking modules, A4s or the live pipeline)."""
    for p in M7B_FILES:
        assert "CC BY-SA 4.0" in (ROOT / p).read_text(), p
    hits = []
    for path in list((ROOT / "nflelo").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        rel = str(path.relative_to(ROOT))
        if rel in M7B_FILES or rel in SITE_EXPORTERS:
            continue
        tree = ast.parse(path.read_text())
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                mod = getattr(n, "module", None) or ""
                names = [a.name for a in n.names]
                if re.search(r"(^|\.)policy($|\.)", mod) or any(re.search(r"(^|\.)policy($|\.)", x) for x in names):
                    hits.append(rel)
    assert not hits, f"the CC BY-SA M7b policy module is imported outside M7b: {hits}"
