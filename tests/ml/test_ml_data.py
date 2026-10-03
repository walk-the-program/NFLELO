"""Cache layer: manifest hashes, refresh, fetch-on-miss, schema checks. Fetchers are faked; no network."""
import json

import pandas as pd
import pytest

from nflelo.ml import data as D


def _pbp(season, n_games=2, plays=120, team="KC"):
    rows = []
    for g in range(n_games):
        gid = f"{season}_01_{team}_BUF{g}"
        for i in range(plays):
            off, de = (team, "BUF") if i % 2 else ("BUF", team)
            rows.append({"game_id": gid, "season": season, "season_type": "REG", "week": 1,
                         "home_team": team, "away_team": "BUF", "posteam": off, "defteam": de,
                         "play_type": "pass" if i % 3 else "run", "epa": 0.1, "success": 1.0, "wp": 0.5,
                         "pass": float(i % 3 != 0), "rush": float(i % 3 == 0), "spread_line": 3.0,
                         "vegas_wp": 0.6})
    return pd.DataFrame(rows)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    calls = []
    state = {"epa": 0.1}

    def fetch_pbp(season):
        calls.append(season)
        df = _pbp(season)
        df["epa"] = state["epa"]
        return df
    monkeypatch.setitem(D.FETCHERS, "pbp", fetch_pbp)
    return tmp_path, calls, state


def test_fetch_on_miss_writes_parquet_and_manifest(fake):
    root, calls, _ = fake
    df = D.load_pbp([2023], root=root)
    assert calls == [2023]
    assert (root / "pbp" / "2023.parquet").exists()
    man = json.loads((root / "manifest.json").read_text())
    rec = man["pbp/2023"]
    assert rec["rows"] == len(df) == 240 and len(rec["sha256"]) == 64 and rec["fetched_at"]
    D.load_pbp([2023], root=root)
    assert calls == [2023]  # second load is served from the cache
    assert D.verify_cache(root) == []


def test_loaded_pbp_has_no_market_columns_and_market_requests_fail(fake):
    root, _, _ = fake
    df = D.load_pbp([2023], root=root)
    assert "spread_line" not in df.columns and "vegas_wp" not in df.columns
    with pytest.raises(ValueError):
        D.load_pbp([2023], columns=["epa", "spread_line"], root=root)
    assert "spread_line" in D.load_pbp([2023], keep_market=True, root=root).columns


def test_refresh_reports_hash_changes(fake):
    root, _, state = fake
    D.load_pbp([2023], root=root)
    same = D.refresh_one("pbp", 2023, root)
    assert same["was_cached"] and not same["changed"]
    state["epa"] = 0.2  # nflverse corrected history
    moved = D.refresh_one("pbp", 2023, root)
    assert moved["changed"] and moved["old_sha256"] != moved["new_sha256"]


def test_verify_cache_detects_tampering(fake):
    root, _, _ = fake
    D.load_pbp([2023], root=root)
    _pbp(2023).head(10).to_parquet(root / "pbp" / "2023.parquet")
    assert D.verify_cache(root) == ["pbp/2023: hash differs from manifest"]


def test_unmapped_team_codes_are_reported_not_dropped():
    df = _pbp(2023, team="XXX")
    rep = D.validate_pbp(df, 2023)
    assert not rep.ok and any("XXX" in e for e in rep.errors)
    assert len(df) == 240  # validation never filters rows
    assert D.unmapped_team_codes(["KC", "OAK", "XXX", "", None], 2019) == ["XXX"]


def test_implausible_play_counts_and_wrong_season_are_errors():
    rep = D.validate_pbp(_pbp(2023, plays=30), 2023)
    assert any("implausible plays per game" in e for e in rep.errors)
    rep = D.validate_pbp(_pbp(2023), 2022)
    assert not rep.ok
    assert D.validate_pbp(_pbp(2023), 2023).ok


def test_missing_columns_are_errors():
    rep = D.validate_pbp(_pbp(2023).drop(columns=["epa"]), 2023)
    assert not rep.ok and "epa" in rep.errors[0]


def test_bad_loader_raises(fake, monkeypatch):
    root, _, _ = fake
    monkeypatch.setitem(D.FETCHERS, "pbp", lambda s: _pbp(s, team="XXX"))
    with pytest.raises(D.DataError):
        D.load_pbp([2024], root=root)


def test_pbp_before_1999_is_refused(tmp_path):
    with pytest.raises(ValueError):
        D.fetch("pbp", 1998, tmp_path)
    assert D.refresh(1998, root=tmp_path, datasets=("pbp",)) == []
