"""M4 Phase 1: the prediction ledger, which row counts, scoring, projections, the live features'
market ban, and the site exporter with and without a ledger."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nflelo import config
from nflelo.ml import asof, live, live_features
from nflelo.ml.data import is_market_column

sys.path.insert(0, str(config.ROOT / "scripts"))

T0 = pd.Timestamp("2026-10-07T14:00:00Z")       # a Wednesday, 10:00 ET


def rows(run_at, games: dict, **over) -> pd.DataFrame:
    """Ledger rows for one run: games = {game_id: kickoff}."""
    n = len(games)
    df = pd.DataFrame({
        "run_at_utc": live.utc_iso(run_at), "game_id": list(games),
        "kickoff_utc": [live.utc_iso(k) for k in games.values()], "model_version": "A4s-test",
        "p_home_model": over.get("p_home_model", [0.6] * n), "spread_model": np.nan,
        "p_home_elo": over.get("p_home_elo", [0.55] * n), "spread_elo": over.get("spread_elo", [2.0] * n),
        "p_home_market": over.get("p_home_market", [0.58] * n), "spread_market": over.get("spread_market", [3.0] * n),
        "home_qb_id": "00-1", "away_qb_id": "00-2", "qb_source": "nflverse", "data_hash": "abc"})
    return df[live.LEDGER_COLUMNS]


KICK = {"2026_05_A": pd.Timestamp("2026-10-09T00:15:00Z"),   # Thursday night
        "2026_05_B": pd.Timestamp("2026-10-11T17:00:00Z")}   # Sunday 1 pm


# --------------------------------------------------------------------------- writing

def test_post_kickoff_rows_are_rejected_and_nothing_is_written(tmp_path):
    path = tmp_path / "ledger.csv"
    at_kick = rows(KICK["2026_05_A"], {"2026_05_A": KICK["2026_05_A"]})
    with pytest.raises(live.LedgerError, match="at or after kickoff"):
        live.append_rows(at_kick, path)
    late = rows(KICK["2026_05_A"] + pd.Timedelta(minutes=1), {"2026_05_A": KICK["2026_05_A"],
                                                              "2026_05_B": KICK["2026_05_B"]})
    with pytest.raises(live.LedgerError):
        live.append_rows(late, path)   # one bad row sinks the whole run
    assert not path.exists()
    ok = rows(KICK["2026_05_A"] - pd.Timedelta(seconds=1), {"2026_05_A": KICK["2026_05_A"]})
    assert live.append_rows(ok, path) == 1


def test_ledger_is_append_only_and_runs_move_forward(tmp_path):
    path = tmp_path / "ledger.csv"
    live.append_rows(rows(T0, KICK), path)
    first = path.read_bytes()
    live.append_rows(rows(T0 + pd.Timedelta(days=1), KICK), path)
    assert path.read_bytes().startswith(first)
    assert len(live.read_ledger(path)) == 4
    with pytest.raises(live.LedgerError, match="not later"):
        live.append_rows(rows(T0, KICK), path)
    with pytest.raises(live.LedgerError, match="columns"):
        live.append_rows(rows(T0 + pd.Timedelta(days=2), KICK).drop(columns="data_hash"), path)
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(live.LedgerError, match="has columns"):
        live.append_rows(rows(T0, KICK), bad)


def test_ml_predict_only_logs_games_that_have_not_kicked_off(tmp_path):
    import ml_predict
    now = pd.Timestamp("2026-10-11T17:00:00Z")
    kick = pd.to_datetime(["2026-10-09T00:15:00Z", "2026-10-11T17:00:00Z", "2026-10-11T20:25:00Z"], utc=True)
    pred = pd.DataFrame({"kickoff_utc": kick, "p_home_model": [0.5, 0.6, 0.7], "p_home_elo": 0.5, "spread_elo": 1.0,
                         "p_home_market": 0.5, "spread_market": 1.5, "home_qb_id": "x", "away_qb_id": "y",
                         "qb_source": "nflverse"}, index=["g_thu", "g_now", "g_late"])
    r = ml_predict.ledger_rows(pred, now, {"model_version": "A4s-test"}, "abc")
    assert list(r["game_id"]) == ["g_late"]    # kickoff equal to now is excluded too
    assert live.append_rows(r, tmp_path / "l.csv") == 1


# --------------------------------------------------------------------------- selection

def test_latest_before_kickoff_and_wednesday_selection(tmp_path):
    path = tmp_path / "ledger.csv"
    live.append_rows(rows(T0, KICK, p_home_model=[0.61, 0.62]), path)                         # Wed
    live.append_rows(rows(T0 + pd.Timedelta(hours=6), KICK, p_home_model=[0.63, 0.64]), path)   # Wed afternoon
    sun = pd.Timestamp("2026-10-11T12:00:00Z")                                                  # Sun 8 am ET
    live.append_rows(rows(sun, {"2026_05_B": KICK["2026_05_B"]}, p_home_model=[0.70]), path)
    # A row forged into the file after kickoff must never be selected.
    forged = rows(KICK["2026_05_B"] + pd.Timedelta(hours=1), {"2026_05_B": KICK["2026_05_B"]}, p_home_model=[0.99])
    forged.to_csv(path, mode="a", header=False, index=False)
    led = live.read_ledger(path)
    final = live.latest_before_kickoff(led).set_index("game_id")["p_home_model"].to_dict()
    assert final == {"2026_05_A": 0.63, "2026_05_B": 0.70}
    wed = live.wednesday_rows(led).set_index("game_id")["p_home_model"].to_dict()
    assert wed == {"2026_05_A": 0.63, "2026_05_B": 0.64}   # the later Wednesday run, not Sunday's
    only_sun = tmp_path / "sun.csv"
    live.append_rows(rows(sun, {"2026_05_B": KICK["2026_05_B"]}), only_sun)
    assert live.wednesday_rows(live.read_ledger(only_sun)).empty


def test_read_ledger_hides_market_columns_unless_asked(tmp_path):
    path = tmp_path / "ledger.csv"
    live.append_rows(rows(T0, KICK), path)
    plain = live.read_ledger(path)
    assert not set(live.MARKET_COLUMNS) & set(plain.columns)
    assert set(live.MARKET_COLUMNS) <= set(live.read_ledger(path, market=True).columns)


# --------------------------------------------------------------------------- scoring and projections

def test_scoring_brier_accuracy_and_ats(tmp_path):
    path = tmp_path / "ledger.csv"
    games = {"2026_04_X": pd.Timestamp("2026-10-04T17:00:00Z"), **KICK}
    live.append_rows(rows(pd.Timestamp("2026-10-01T14:00:00Z"), games, p_home_model=[0.9, 0.8, 0.3],
                          p_home_elo=[0.5, 0.6, 0.6], p_home_market=[0.7, np.nan, 0.4],
                          spread_elo=[1.0, 5.0, -2.0], spread_market=[3.0, 3.0, -2.0]), path)
    results = pd.DataFrame({"game_id": ["2026_04_X", "2026_05_A", "2026_05_B"], "season": 2026, "week": [4, 5, 5],
                            "y": [1.0, 1.0, 0.0], "margin": [7.0, 3.0, -10.0]})
    s = live.live_record(live.read_ledger(path, market=True), results, from_week=5)["final"]
    assert s["n"] == 2 and s["weeks"] == [5]                       # week 4 is before the live start
    assert s["model"]["brier"] == pytest.approx(((1 - 0.8) ** 2 + 0.3 ** 2) / 2, abs=1e-4)
    assert s["model"]["accuracy"] == 1.0 and s["elo"]["accuracy"] == 0.5
    assert s["market_games"]["n"] == 1 and s["market_games"]["market"]["brier"] == pytest.approx(0.16, abs=1e-4)
    # A: Elo 5.0 > line 3.0 -> takes home; margin 3 == line -> push. B: Elo equals the line -> no pick.
    assert s["ats_elo"] == {"w": 0, "l": 0, "push": 1, "no_pick": 1}
    assert s["ats_model"] == {"w": 0, "l": 0, "push": 0, "no_pick": 0, "games": 0}   # Phase 1 rows have no spread


def test_model_ats_counts_only_rows_with_a_spread(tmp_path):
    path = tmp_path / "ledger.csv"
    r = rows(pd.Timestamp("2026-10-07T14:00:00Z"), KICK, spread_market=[3.0, -2.0])
    r["spread_model"] = [6.5, np.nan]                              # A: model likes home more than Vegas
    live.append_rows(r, path)
    results = pd.DataFrame({"game_id": ["2026_05_A", "2026_05_B"], "season": 2026, "week": [5, 5],
                            "y": [1.0, 0.0], "margin": [7.0, -10.0]})
    s = live.live_record(live.read_ledger(path, market=True), results, from_week=5)["final"]
    assert s["ats_model"] == {"w": 1, "l": 0, "push": 0, "no_pick": 0, "games": 1}   # 7 beats the 3-point line


def sim_rows(run_at, val=0.5):
    teams = ["KC", "BUF"]
    df = pd.DataFrame({"run_at_utc": live.utc_iso(run_at), "team": teams})
    for c in live.SIM_PROBS:
        df[c] = val
    df["wins_mean"], df["wins_p10"], df["wins_p90"] = 9.0, 7.0, 11.0
    df["n_sims"], df["tau_rest"], df["shape"], df["seed"], df["model_version"] = 100, 0.5, "keynum", 1, "x"
    return df[live.SIM_COLUMNS]


def test_simulation_history_is_append_only(tmp_path):
    path = tmp_path / "sim.csv"
    live.append_sim(sim_rows(T0), path)
    before = path.read_bytes()
    live.append_sim(sim_rows(T0 + pd.Timedelta(days=4), 0.6), path)
    assert path.read_bytes().startswith(before)
    with pytest.raises(live.LedgerError):
        live.append_sim(sim_rows(T0 + pd.Timedelta(days=1)), path)         # not later than the last run
    bad = sim_rows(T0 + pd.Timedelta(days=9))
    bad.loc[0, "playoffs"] = 1.2
    with pytest.raises(live.LedgerError):
        live.append_sim(bad, path)
    assert len(live.read_sim(path)) == 4


def test_simulation_baseline_is_the_previous_wednesday_run():
    wed = pd.Timestamp("2026-10-14T14:00:00Z")                    # Wednesday 10:00 ET
    runs = pd.Series([wed - pd.Timedelta(days=7), wed - pd.Timedelta(days=3), wed, wed + pd.Timedelta(days=4)])
    assert live.sim_baseline(runs, wed) == wed - pd.Timedelta(days=7)          # Wednesday vs last Wednesday
    assert live.sim_baseline(runs, wed + pd.Timedelta(days=4)) == wed          # Sunday vs this Wednesday
    assert live.sim_baseline(runs[:1], runs[0]) is None                         # first run: nothing to compare


def test_projected_wins_counts_ties_as_half():
    g = pd.DataFrame({
        "game_id": ["g1", "g2", "g3", "g4", "g5"], "season": 2026, "week": [1, 1, 2, 3, 3],
        "home_team": ["KC", "BUF", "KC", "BUF", "MIA"], "away_team": ["BUF", "MIA", "MIA", "KC", "NE"],
        "home_score": [24, 17, np.nan, np.nan, np.nan], "away_score": [20, 17, np.nan, np.nan, np.nan]})
    p_model = pd.Series({"g3": 0.7, "g4": 0.4, "g5": 0.5})
    p_elo = pd.Series({"g3": 0.6, "g4": 0.5, "g5": 0.55})
    out = live.projected_wins(g, p_model, p_elo).set_index("team")
    assert (out.loc["KC", ["w", "l", "t"]] == [1, 0, 0]).all()
    assert (out.loc["BUF", ["w", "l", "t"]] == [0, 1, 1]).all()
    assert out.loc["KC", "proj_model"] == pytest.approx(1 + 0.7 + 0.6)          # g3 home, g4 away (1 - 0.4)
    assert out.loc["BUF", "proj_model"] == pytest.approx(0 + 0.5 + 0.4)         # the tie is half a win
    assert out.loc["MIA", "proj_elo"] == pytest.approx(0.5 + 0.4 + 0.55)
    assert out["proj_model"].sum() == pytest.approx(len(g))                     # every game hands out one win
    assert out.loc["KC", "remaining"] == 2 and [x["opp"] for x in out.loc["KC", "games"]] == ["MIA", "BUF"]
    with pytest.raises(ValueError, match="no model probability"):
        live.projected_wins(g, p_model.drop("g5"), p_elo)


def test_qb_change_tag_threshold():
    base = {"home_team": "KC", "away_team": "BUF", "home_qb_name": "Backup", "away_qb_name": "Starter",
            "qb_delta_home": -0.09, "qb_delta_away": 0.0}
    assert live.qb_change({**base, "qb_delta_diff": -0.09})["name"] == "Backup"
    assert live.qb_change({**base, "qb_delta_diff": -0.05}) is None


# --------------------------------------------------------------------------- live features: D3

def _live_inputs(synth_sched):
    s = asof.add_asof(synth_sched)
    wk4 = s[(s["season"] == 2023) & (s["week"] == 4)]
    now = wk4["as_of"].min() - pd.Timedelta(hours=1)
    s = s.copy()
    s.loc[s["kickoff"] >= now, ["home_score", "away_score", "result", "total"]] = np.nan
    later = (s["season"] == 2023) & (s["week"] >= 5)
    s.loc[later, ["home_qb_id", "away_qb_id"]] = np.nan              # nflverse hasn't listed them yet
    todo = s[(s["season"] == 2023) & (s["game_type"] == "REG") & s["home_score"].isna()]
    elo = {"ratings": {t: 1500.0 + 10 * i for i, t in enumerate(["KC", "BUF", "NE", "MIA", "DAL", "PHI"])},
           "hfa_now": 50.0, "hfa_week": {}}
    return s, todo, now, elo


def test_live_features_ignore_the_market_and_never_emit_it(synth_pbp, synth_sched):
    s, todo, now, elo = _live_inputs(synth_sched)
    a = live_features.build(synth_pbp, s, todo, now, elo)
    rng = np.random.default_rng(7)
    s2 = s.copy()
    for c in ("spread_line", "total_line", "home_moneyline", "away_moneyline"):
        s2[c] = rng.normal(0, 300, len(s2))
    b = live_features.build(synth_pbp, s2, s2.loc[todo.index], now, elo)
    pd.testing.assert_frame_equal(a, b)
    assert not [c for c in a.columns if is_market_column(c) or c in live.MARKET_COLUMNS]


def test_feature_code_never_reads_the_ledger():
    ml = config.ROOT / "nflelo" / "ml"
    sources = [ml / "live_features.py", *sorted((ml / "features").glob("*.py"))]
    for p in sources:
        text = p.read_text()
        for banned in ("read_ledger", "p_home_market", "spread_market", "experiments/live", "from . import live",
                       "from .live import", "ml.live import"):
            assert banned not in text, f"{p.name} mentions {banned!r}"


def test_live_features_weeks_and_starters(synth_pbp, synth_sched):
    s, todo, now, elo = _live_inputs(synth_sched)
    f = live_features.build(synth_pbp, s, todo, now, elo)
    assert set(f["feature_week"]) == {4}                    # every future week sees the same information
    wk4 = f[f["game_week"] == 4]
    assert (wk4["qb_source"] == "nflverse").all()
    later = f[f["game_week"] >= 5]
    assert (later["qb_source"] == "last_starter").all()
    kc = later[(later["home_team"] == "KC") | (later["away_team"] == "KC")]
    sides = np.where(kc["home_team"] == "KC", kc["home_qb_id"], kc["away_qb_id"])
    assert set(sides) == {"KC_QB2"}                         # the backup started KC's last game (week 3)
    assert f[live_features.FEATURES].notna().all().all()


# --------------------------------------------------------------------------- exporter

INPUTS = [config.OUT_DIR / "elo_games.csv", config.OUT_DIR / "ratings_current.json", config.RAW_DIR / "schedules.csv"]
SITE_FILES = ["meta.json", "ladder.json", "upcoming.json", "history.json", "luck.json", "tapestry.json",
              "records.json", "scorecard.json", "headlines.json"]
needs_inputs = pytest.mark.skipif(not all(p.exists() for p in INPUTS), reason="needs outputs/ and data/raw/schedules.csv")


def write_ledger(live_dir: Path, schedule: pd.DataFrame, season: int, unplayed_only: bool = True):
    """A one-run ledger (plus latest_<season>.json) for the season's games, made a day before the first kickoff.
    Returns (run_at, ledger rows), or None when there are no games to log."""
    reg = schedule[(schedule["season"] == season) & (schedule["game_type"] == "REG")]
    todo = reg[reg["home_score"].isna()] if unplayed_only else reg
    if todo.empty:
        return None
    kick = asof.add_asof(todo).set_index("game_id")["kickoff"].dt.tz_convert("UTC")
    run_at = kick.min() - pd.Timedelta(days=1)
    led = rows(run_at, kick.to_dict(), p_home_model=list(np.linspace(0.3, 0.7, len(kick))),
               p_home_elo=[0.5] * len(kick), spread_elo=[1.0] * len(kick),
               p_home_market=[0.55] * len(kick), spread_market=[2.5] * len(kick))
    live.append_rows(led, live.ledger_path(season, live_dir))
    latest = {"run_at_utc": live.utc_iso(run_at), "model_version": "A4s-test",
              "games": [{"game_id": g, "home_team": "KC", "away_team": "BUF", "p_home_model": p, "qb_delta_diff": 0.1,
                         "qb_delta_home": 0.1, "qb_delta_away": 0.0, "home_qb_name": "Somebody"}
                        for g, p in zip(led["game_id"], led["p_home_model"])]}
    live.latest_path(season, live_dir).write_text(json.dumps(latest))
    return run_at, led


def write_sim_history(live_dir: Path, season: int, run_at: pd.Timestamp, teams: list[str]) -> pd.Timestamp:
    """Two Wednesday simulation runs a week apart (all probabilities 0.25, then 0.5). Returns the later run time."""
    wed = run_at.floor("D") - pd.Timedelta(days=(run_at.dayofweek - 2) % 7) + pd.Timedelta(hours=14)
    for k, at in enumerate((wed - pd.Timedelta(days=7), wed)):
        df = pd.DataFrame({"run_at_utc": live.utc_iso(at), "team": teams})
        for c in live.SIM_PROBS:
            df[c] = 0.25 + 0.25 * k
        df["wins_mean"], df["wins_p10"], df["wins_p90"] = 8.5, 6.0, 11.0
        df["n_sims"], df["tau_rest"], df["shape"], df["seed"], df["model_version"] = 20000, 0.5, "keynum", 1, "A4s+M"
        live.append_sim(df[live.SIM_COLUMNS], live.sim_path(season, live_dir))
    return wed


def season_teams(schedule: pd.DataFrame, season: int) -> list[str]:
    reg = schedule[(schedule["season"] == season) & (schedule["game_type"] == "REG")]
    return sorted(set(reg["home_team"]) | set(reg["away_team"]))


@needs_inputs
def test_exporter_ml_files_never_change_the_non_ml_site_data(tmp_path, monkeypatch):
    """Same environment, same inputs: exporting with the ML files (ledger, latest run, open simulation) and
    without them gives byte-identical non-ML JSON. Only ml.json differs, and only the ML run has it."""
    import export_site
    schedule, season = export_site.load_schedule(), export_site.load_ratings()["season"]
    with_ml, without_ml = tmp_path / "live_with", tmp_path / "live_without"
    with_ml.mkdir()
    without_ml.mkdir()
    made = write_ledger(with_ml, schedule, season, unplayed_only=False)
    assert made is not None, f"no {season} REG games in the schedule"
    (with_ml / live.PUBLISH_FILE).write_text(json.dumps({"publish_sim": True}))
    write_sim_history(with_ml, season, made[0], season_teams(schedule, season))

    out, sizes = {}, {}
    for name, live_dir in (("with", with_ml), ("without", without_ml)):
        out[name] = tmp_path / f"data_{name}"
        out[name].mkdir()
        (out[name] / "ml.json").write_text("{}")         # a stale file from an earlier run
        monkeypatch.setattr(export_site, "SITE_DATA", out[name])
        monkeypatch.setattr(export_site, "LIVE_DIR", live_dir)
        sizes[name] = export_site.export()

    assert set(sizes["without"]) == set(SITE_FILES)
    assert not (out["without"] / "ml.json").exists()     # the stale file is removed when there is no ledger
    assert set(sizes["with"]) == set(SITE_FILES) | {"ml.json"}
    ml = json.loads((out["with"] / "ml.json").read_text())
    assert ml["ledger_rows"] == len(made[1]) and ml["sim_published"] is True and ml["playoff_odds"] is not None
    for name in SITE_FILES:
        assert (out["with"] / name).read_bytes() == (out["without"] / name).read_bytes(), name


@needs_inputs
def test_exporter_with_ledger_writes_valid_ml_json(tmp_path, monkeypatch):
    import export_site
    elo, ratings, schedule = export_site.load_elo(), export_site.load_ratings(), export_site.load_schedule()
    season = ratings["season"]
    upcoming = export_site.build_upcoming(schedule, ratings, elo)
    reg = schedule[(schedule["season"] == season) & (schedule["game_type"] == "REG")]
    live_dir = tmp_path / "live"
    made = write_ledger(live_dir, schedule, season)
    if made is None:
        pytest.skip("no unplayed games")
    run_at, led = made
    out = tmp_path / "data"
    monkeypatch.setattr(export_site, "SITE_DATA", out)
    assert export_site.export_ml(schedule, ratings, elo, upcoming, live_dir) > 0
    ml = json.loads((out / "ml.json").read_text())
    assert ml["season"] == season and ml["live_from_week"] == live.LIVE_FROM_WEEK.get(season, 1)
    assert ml["ledger_rows"] == len(led) and ml["games_logged"] == len(led)
    assert ml["ledger_url"].endswith(f"experiments/live/{season}.csv")
    assert len(ml["rest"]["teams"]) == 32 and ml["rest"]["model_fallback_to_elo"] == 0
    for key in ("proj_model", "proj_elo"):
        assert sum(t[key] for t in ml["rest"]["teams"]) == pytest.approx(len(reg), abs=0.2)  # one win per game
    wk = ml["week"]
    assert wk["week"] == upcoming["week"] and len(wk["games"]) == len(upcoming["games"])
    assert all(g["p_home_model"] is not None and g["qb_change"]["name"] == "Somebody" for g in wk["games"])
    assert len(wk["flagged"]) == min(3, len(wk["games"])) and wk["headline"]
    assert ml["live"]["final"]["n"] == 0                   # nothing from the live weeks is final yet
    assert ml["playoff_odds"] is None                      # no simulation history in this directory
    json.dumps(ml, allow_nan=False)

    # with two simulation runs: odds per conference, change since the earlier Wednesday run, win ranges
    (live_dir / live.PUBLISH_FILE).write_text(json.dumps({"publish_sim": True}))
    wed = write_sim_history(live_dir, season, run_at, sorted({t["team"] for t in ml["rest"]["teams"]}))
    export_site.export_ml(schedule, ratings, elo, upcoming, live_dir)
    ml = json.loads((out / "ml.json").read_text())
    po = ml["playoff_odds"]
    assert po["runs"] == 2 and po["baseline_run_utc"] == live.utc_iso(wed - pd.Timedelta(days=7))
    assert sorted(po["conferences"]) == ["AFC", "NFC"] and sum(len(v) for v in po["conferences"].values()) == 32
    row = po["conferences"]["AFC"][0]
    assert row["playoffs"] == 0.5 and row["d_playoffs"] == pytest.approx(0.25) and row["div"].startswith("AFC")
    assert all(t["wins_p10"] == 6.0 and t["wins_p90"] == 11.0 for t in ml["rest"]["teams"])
    assert ml["sim_published"] is True
    json.dumps(ml, allow_nan=False)

    # gated: the simulation history is there, but publish_sim is false -> nothing shown
    (live_dir / live.PUBLISH_FILE).write_text(json.dumps({"publish_sim": False}))
    export_site.export_ml(schedule, ratings, elo, upcoming, live_dir)
    ml = json.loads((out / "ml.json").read_text())
    assert ml["sim_published"] is False and ml["playoff_odds"] is None
    assert all(t["wins_p10"] is None and t["wins_p90"] is None for t in ml["rest"]["teams"])
    (live_dir / live.PUBLISH_FILE).unlink()
    export_site.export_ml(schedule, ratings, elo, upcoming, live_dir)
    assert json.loads((out / "ml.json").read_text())["playoff_odds"] is None      # no file: not published


def test_committed_publish_flag_is_valid():
    """experiments/live/publish.json exists and holds a real boolean (opened 2026-10-05 after the M4 holdout)."""
    path = live.LIVE_DIR / live.PUBLISH_FILE
    assert path.exists(), path
    d = json.loads(path.read_text())
    assert isinstance(d.get("publish_sim"), bool), "publish_sim must be a JSON boolean, not a string or number"
    assert live.publish_flags()["publish_sim"] is d["publish_sim"]


@needs_inputs
def test_committed_publish_flag_shows_odds_only_with_simulation_history(tmp_path, monkeypatch):
    """With the committed flag: no simulation history -> no odds or win ranges, even when the gate is open;
    with history -> odds and win ranges exactly when the gate is open."""
    import export_site
    elo, ratings, schedule = export_site.load_elo(), export_site.load_ratings(), export_site.load_schedule()
    season = ratings["season"]
    upcoming = export_site.build_upcoming(schedule, ratings, elo)
    live_dir = tmp_path / "live"
    made = write_ledger(live_dir, schedule, season, unplayed_only=False)
    assert made is not None, f"no {season} REG games in the schedule"
    (live_dir / live.PUBLISH_FILE).write_bytes((live.LIVE_DIR / live.PUBLISH_FILE).read_bytes())
    is_open = live.publish_flags(live_dir)["publish_sim"]
    assert is_open is live.publish_flags()["publish_sim"]
    out = tmp_path / "data"
    monkeypatch.setattr(export_site, "SITE_DATA", out)

    export_site.export_ml(schedule, ratings, elo, upcoming, live_dir)
    ml = json.loads((out / "ml.json").read_text())
    assert ml["sim_published"] is is_open
    assert ml["playoff_odds"] is None                      # no sim_<season>.csv yet
    assert all(t["wins_p10"] is None and t["wins_p90"] is None for t in ml["rest"]["teams"])

    write_sim_history(live_dir, season, made[0], season_teams(schedule, season))
    export_site.export_ml(schedule, ratings, elo, upcoming, live_dir)
    ml = json.loads((out / "ml.json").read_text())
    assert ml["sim_published"] is is_open
    if is_open:
        assert ml["playoff_odds"] is not None and ml["playoff_odds"]["runs"] == 2
        assert all(t["wins_p10"] == 6.0 and t["wins_p90"] == 11.0 for t in ml["rest"]["teams"])
    else:
        assert ml["playoff_odds"] is None
        assert all(t["wins_p10"] is None and t["wins_p90"] is None for t in ml["rest"]["teams"])
