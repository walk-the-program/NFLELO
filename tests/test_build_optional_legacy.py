"""The build must run without the Pro-Football-Reference-derived legacy spreadsheet."""
import sys

import pandas as pd
import pytest

from nflelo import config, data

sys.path.insert(0, str(config.ROOT / "scripts"))
import build  # noqa: E402


@pytest.fixture
def no_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LEGACY_XLSX", tmp_path / "missing.xlsx")


@pytest.fixture(scope="module")
def sources():
    if not config.ELO538_CSV.exists() or not (config.RAW_DIR / "schedules.csv").exists():
        pytest.skip("538 file or cached nflverse schedules not present")
    nfl = data.nflverse_to_games(data.fetch_nflverse_schedules(refresh=False))
    return data.load_538_games(seasons=(1970, 2022)), nfl


def test_missing_legacy_returns_none(no_legacy):
    assert build.load_legacy_if_present() is None
    assert build.legacy_reproduction(None) == (None, None)


def test_validations_and_report_without_legacy(no_legacy, sources, tmp_path):
    elo538, nfl = sources
    validations = build.run_validations(elo538, nfl, build.load_legacy_if_present())
    assert [v["b_name"] for _, v in validations] == ["nflverse", "nflverse"]
    assert all(v["match_rate"] > 0.99 for _, v in validations)

    games = data.build_games(elo538, nfl)
    season_check = data.season_franchise_check(games)
    elo = pd.DataFrame({"model": ["m"], "n": [1], "brier": [0.2], "logloss": [0.6], "accuracy": [0.6]})
    test = {"all": elo, "market": elo, "market_seasons": (2006, 2025),
            "paired": {"model": "m", "n": 1, "brier_diff_vs_market": 0.0, "se": 0.0}}
    decade = pd.DataFrame({"decade": ["1970s"], "games": [1], "actual_home_win_rate": [0.5], "chosen_model_hfa": [60.0]})
    out = tmp_path / "report.md"
    build.write_report(out, validations, season_check, None, None, config.DEFAULT_CONFIG, test, decade, None)
    text = out.read_text()
    assert text.count(build.LEGACY_MISSING_NOTE) == 2   # validation section and reproduction section
    assert "FiveThirtyEight vs nflverse" in text and "vs legacy spreadsheet" not in text


@pytest.mark.skipif(not config.LEGACY_XLSX.exists(), reason="legacy spreadsheet not present")
def test_legacy_validation_runs_when_present(sources):
    elo538, nfl = sources
    legacy = build.load_legacy_if_present()
    names = [v["b_name"] for _, v in build.run_validations(elo538, nfl, legacy)]
    assert names == ["legacy spreadsheet", "nflverse", "nflverse"]
