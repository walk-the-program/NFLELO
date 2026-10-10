"""Model-page exporter (scripts/export_ml_pages.py): schemas, no market fields, completed plays only,
WP series sanity, play-grid probabilities, and the fail-soft weekly step.

Unit tests run offline on synthetic frames. The committed site/data files are checked against the data
contract (context/ml-pages.md) when they exist.
"""
import json
import re
import sys

import numpy as np
import pandas as pd
import pytest

from nflelo import config
from nflelo.ml.plays import data as pdata

sys.path.insert(0, str(config.ROOT / "scripts"))
import export_ml_pages as em  # noqa: E402

SITE = config.ROOT / "site" / "data"

# The data contract: required top-level keys per page file (context/ml-pages.md).
REQUIRED = {
    "plays.json": ["schema", "page", "license", "headline", "bins", "grid", "held_neutral", "fields", "cells",
                   "caveats"],
    "winprob.json": ["schema", "page", "season", "license", "headline", "updated_through", "definitions",
                     "games_by_excitement", "games", "caveats"],
    "fourth.json": ["schema", "page", "season", "license", "label", "status", "headline", "tossup_rule", "chart",
                    "live", "definitions", "caveats"],
    "playcalling.json": ["schema", "page", "license", "headline", "definitions", "seasons", "cells",
                         "team_view_2025", "caveats"],
}
LICENSES = {"plays.json": "CC BY-SA 4.0", "winprob.json": "CC BY 4.0", "fourth.json": "CC BY-SA 4.0",
            "playcalling.json": "CC BY-SA 4.0"}


def committed(name):
    p = SITE / name
    if not p.exists():
        pytest.skip(f"{name} not generated")
    return json.loads(p.read_text())


# --------------------------------------------------------------------------- contract

def test_season_matches_the_live_pipeline():
    import ml_predict
    assert em.SEASON == ml_predict.SEASON


@pytest.mark.parametrize("name", sorted(REQUIRED))
def test_committed_files_follow_the_contract(name):
    d = committed(name)
    missing = [k for k in REQUIRED[name] if k not in d]
    assert not missing, missing
    assert d["schema"] == em.SCHEMA and d["page"] == name.split(".")[0]
    assert d["license"]["license"] == LICENSES[name] and d["license"]["credit"]
    assert d["caveats"] and all(isinstance(c, str) and c for c in d["caveats"])


@pytest.mark.parametrize("name", sorted(REQUIRED))
def test_no_market_or_betting_fields(name):
    d = committed(name)
    assert em.banned_keys(d) == []
    text = json.dumps(d).lower()
    for word in ("vegas", "moneyline", "spread_line", "sportsbook", "against the spread", "units"):
        assert word not in text, word


def test_banned_key_check_catches_market_fields():
    assert em.banned_keys({"a": [{"vegas_wp": 1}], "b": {"spread": 2, "yardline_100": 3}}) == ["/a[0]/vegas_wp",
                                                                                             "/b/spread"]
    assert em.banned_keys({"yardline_100": 1, "p_first_or_td": 2, "best": 3}) == []
    assert em.banned_keys({"t": {"columns": ["game_id", "vegas_wp"], "rows": []}}) == ["/t/columns/vegas_wp"]
    with pytest.raises(ValueError):
        em.checked({"home_moneyline": -110})


def test_winprob_code_path_stays_cc_by():
    """winprob.json is CC BY: its functions read play-by-play and A4s only, never M6, M7b or the fourth-down
    module (CC BY-SA). The other three pages are CC BY-SA and say so."""
    import inspect
    for fn in (em.load_live, em.wp_model, em.wp_states, em.game_series, em.build_winprob, em.wp_headline):
        src = inspect.getsource(fn)
        for token in ("fd.", "pdata.", "gbm.", "pm.", "pol.", "kk.", "m6_bundle", "ml_m6", "full_pbp", "participation"):
            pat = r"(?<![\w.])" + re.escape(token) if token.endswith(".") else re.escape(token)   # wpm. is fine
            assert not re.search(pat, src), (fn.__name__, token)
    assert set(em.WP_PBP_COLUMNS) <= set(em.wpm.PBP_COLUMNS) | {"desc"}


# --------------------------------------------------------------------------- plays.json

def test_sparse_bins_keep_the_mass():
    p = np.zeros(pdata.N_BINS)
    p[10:30] = 0.045
    p[5] = 0.0002                          # trimmed (under P_MIN at the low end)
    p[pdata.TD_BIN] = 1 - p.sum()
    lo, ps = em.sparse_bins(p)
    assert lo == 10 and len(ps) == 20
    assert abs(sum(ps) + p[pdata.TD_BIN] - 1) < 0.001


def test_plays_grid_drops_impossible_cells():
    g = em.plays_grid(pd.DataFrame())
    assert (g["ydstogo"] <= g["yardline_100"]).all()
    assert len(g) == sum(1 for t in em.GRID_YTG for y in em.GRID_YL if t <= y) * 4 * 2 * 4


def test_committed_plays_cells_sum_to_one():
    d = committed("plays.json")
    assert d["grid"]["cells"] == len(d["cells"]) > 1000
    for c in d["cells"]:
        total = sum(c["bins"]) + c["p_td"]
        assert abs(total - 1) < 0.01, (c["down"], c["ydstogo"], c["yardline_100"], total)
        assert c["ydstogo"] <= c["yardline_100"]
        for k in ("p_first_or_td", "p_20plus", "p_loss", "p_td", "p_turnover"):
            assert 0 <= c[k] <= 1, k
        hi = c["bins_from"] + len(c["bins"]) - 1
        assert hi < pdata.TD_BIN and pdata.BIN_YARDS[hi] < c["yardline_100"] or hi == pdata.BIG_BIN
    assert len(json.dumps(d)) < 1_500_000


# --------------------------------------------------------------------------- winprob.json

def _states(n=40, seed=0):
    rng = np.random.default_rng(seed)
    secs = np.sort(rng.uniform(0, 3600, n))[::-1].copy()
    secs[10] = secs[9] + 5                       # a clock glitch: time must still never run backwards
    return pd.DataFrame({"play_id": np.arange(n) * 10 + 1, "game_secs": secs, "home": (np.arange(n) // 5) % 2,
                         "qtr": np.clip(4 - secs // 900, 1, 4), "posteam": "A", "desc": "a play"})


def test_game_series_is_bounded_ordered_and_scored():
    st = _states()
    p = np.random.default_rng(1).uniform(0.05, 0.95, len(st))
    s = em.game_series(st, p, 1.0)
    t, wp = np.array(s["t"]), np.array(s["wp"], float)
    assert len(t) == len(wp) == len(st) + 1 and (np.diff(t) >= 0).all() and t[-1] == 3600
    assert ((wp >= 0) & (wp <= 1)).all() and wp[-1] == 1.0
    assert abs(s["excitement"] - np.abs(np.diff(wp)).sum()) < 0.01
    assert s["swing"]["abs"] == pytest.approx(np.abs(np.diff(wp)).max(), abs=0.002)
    home_p = np.where(st["home"] == 1, p, 1 - p)
    assert wp[0] == pytest.approx(home_p[0], abs=0.001)          # home view, not the offense's


def test_committed_winprob_series():
    d = committed("winprob.json")
    assert len(d["games"]) == d["updated_through"]["games"] == len(d["games_by_excitement"])
    ex = [g["excitement"] for g in d["games_by_excitement"]]
    assert ex == sorted(ex, reverse=True)
    for g in d["games"]:
        t, wp = np.array(g["t"]), np.array(g["wp"], float)
        assert len(t) == len(wp) >= 2 and (np.diff(t) >= 0).all() and t.min() >= 0
        assert ((wp >= 0) & (wp <= 1)).all()
        final = 1.0 if g["home_score"] > g["away_score"] else 0.0 if g["home_score"] < g["away_score"] else 0.5
        assert wp[-1] == final
        assert len(t) <= 260                                   # about one point per snap or fewer


# --------------------------------------------------------------------------- fourth.json

def _live(S=2018):
    sched = pd.DataFrame({"game_id": ["G1", "G2", "G3", "G4"], "season": S, "game_type": "REG", "week": [1, 1, 2, 2],
                          "home_team": "H", "away_team": "A", "gameday": "2018-09-09",
                          "home_score": [20.0, np.nan, 17.0, 21.0], "away_score": [10.0, np.nan, 14.0, 3.0]})
    a4s = pd.Series(0.5, index=["G1", "G2", "G4"])           # G3 final but held back (no A4s yet)
    return em.Live(S, pd.DataFrame(), pd.DataFrame(), sched, a4s, ["G3"], pbp=pd.DataFrame({"x": [1]}))


def test_final_games_are_completed_and_ready():
    assert list(_live().final_games()["game_id"]) == ["G1", "G4"]


def test_live_fourth_rows_only_for_completed_games_and_only_new_ones(monkeypatch, tmp_path):
    L = _live()
    calls = []

    def fake_dec(pbp, a4s, seasons=None):      # fourth downs exist for every game, finished or not
        return pd.DataFrame({"game_id": ["G1", "G2", "G3", "G4"], "play_id": [5, 6, 7, 8], "season": 2018,
                             "week": [1, 1, 2, 2]})

    def fake_value(comp, dec, ratings, *a, **k):
        calls.append(sorted(dec["game_id"]))
        v = pd.DataFrame({c: 0.0 for c in em.LIVE_COLS}, index=dec.index)
        v["game_id"], v["play_id"], v["week"] = dec["game_id"], dec["play_id"], dec["week"]
        v[["choice", "best", "second", "posteam", "defteam", "desc"]] = ["go", "go", "punt", "H", "A", ""]
        return v, None
    monkeypatch.setattr(em.fd, "decision_table", fake_dec)
    monkeypatch.setattr(em.fd, "value_season", fake_value)
    monkeypatch.setattr(em.oa, "rating_table", lambda *a, **k: None)
    monkeypatch.setattr(em.pdata, "week_ratings", lambda *a, **k: None)
    cache = tmp_path / "fourth_live.parquet"
    rows, n_new = em.value_live(L, None, "fp1", cache)
    assert sorted(set(rows["game_id"])) == ["G1", "G4"] and n_new == 2 and calls == [["G1", "G4"]]
    # next week: G2 has finished; only it is valued, the cached rows are kept
    L.sched.loc[L.sched["game_id"] == "G2", ["home_score", "away_score"]] = [24.0, 21.0]
    rows, n_new = em.value_live(L, None, "fp1", cache)
    assert sorted(set(rows["game_id"])) == ["G1", "G2", "G4"] and n_new == 1 and calls[-1] == ["G2"]
    # a changed engine fingerprint revalues everything
    rows, n_new = em.value_live(L, None, "fp2", cache)
    assert n_new == 3 and (rows["fp"] == "fp2").all()


def test_committed_fourth_rows_are_completed_plays():
    d = committed("fourth.json")
    live = d["live"]
    assert live["label"] == "model-estimated" and d["label"] == "model-estimated"
    assert "2027-02-15" in d["status"]["plain"] and d["status"]["forward_test"] == "pending"
    sched_p = config.ROOT / "data" / "raw" / "ml" / "schedules" / f"{d['season']}.parquet"
    final = None
    if sched_p.exists():
        s = pd.read_parquet(sched_p)
        final = set(s.loc[s["home_score"].notna(), "game_id"])
    cols = live["plays"]["columns"]
    for row in live["plays"]["rows"]:
        assert len(row) == len(cols)
        p = dict(zip(cols, row))
        assert p["choice"] in ("go", "fg", "punt") and p["recommended"] in ("go", "fg", "punt")
        assert p["wp_given_up"] >= -1e-9 and 0 <= p["wp_go"] <= 1
        cut = d["tossup_rule"]["margin_below"]
        if abs(p["margin"] - cut) > 0.0006:                   # margins are rounded to 3 decimals
            assert p["tossup"] == (p["margin"] < cut)
        if final is not None:
            assert p["game_id"] in final
    for pr in d["chart"]["presets"]:
        n = len(pr["yardline_100"])
        assert n == sum(1 for y in em.CHART_YL for t in em.CHART_YTG if t <= y)
        assert all(len(pr[c]) == n for c in pr["columns"])
    assert "without" in d["chart"]["label"].lower()


# --------------------------------------------------------------------------- fail-soft

def test_live_step_fails_soft(monkeypatch, tmp_path):
    monkeypatch.setattr(em, "SITE_DATA", tmp_path)
    (tmp_path / "winprob.json").write_text('{"old": true}\n')
    monkeypatch.setattr(em, "load_live", lambda S, no_refresh: "inputs")

    def boom(L, *a, **k):
        raise RuntimeError("nflverse is down")
    monkeypatch.setattr(em, "build_winprob", boom)
    monkeypatch.setattr(em, "build_fourth", lambda L, *a, **k: {"page": "fourth", "ok": L == "inputs"})
    assert em.run_live(2026) == 1                                     # reported, not raised
    assert json.loads((tmp_path / "winprob.json").read_text()) == {"old": True}     # untouched
    assert json.loads((tmp_path / "fourth.json").read_text()) == {"page": "fourth", "ok": True}
    assert not list(tmp_path.glob("*.tmp"))

    def inputs_fail(S, no_refresh):
        raise OSError("no network")
    monkeypatch.setattr(em, "load_live", inputs_fail)
    assert em.run_live(2026) == 1                                     # even the shared inputs failing is soft


def test_budget_skips_remaining_parts(monkeypatch, tmp_path):
    monkeypatch.setattr(em, "SITE_DATA", tmp_path)
    monkeypatch.setattr(em, "load_live", lambda S, no_refresh: "inputs")
    monkeypatch.setattr(em, "build_winprob", lambda L, *a, **k: {"page": "winprob"})
    monkeypatch.setattr(em, "build_fourth", lambda L, *a, **k: {"page": "fourth"})
    assert em.run_live(2026, budget=-1) == 1
    assert not (tmp_path / "winprob.json").exists() and not (tmp_path / "fourth.json").exists()
