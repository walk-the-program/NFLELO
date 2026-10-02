import pandas as pd

from nflelo import data


def legacy_rows():
    # Two games, each listed from both sides; game 2 is on a neutral field.
    return pd.DataFrame({
        "GameID": [1, 2, 3, 4],
        "Team": ["GNB", "CHI", "NOR", "SEA"],
        "Date": pd.to_datetime(["1990-09-09", "1990-09-09", "1990-09-16", "1990-09-16"]),
        "Hosting": [None, "@", "N", "N"],
        "Opp": ["CHI", "GNB", "SEA", "NOR"],
        "Result": ["W 24-17", "L 17-24", "L 10-20", "W 20-10"],
        "Week": [1, 1, 2, 2],
    })


def test_dedupe_legacy_two_row_format():
    d = data.dedupe_legacy_rows(legacy_rows())
    assert len(d) == 2
    first = d[d["Team"] == "GNB"].iloc[0]  # home row kept, away row dropped
    assert first["Opp"] == "CHI"
    assert "CHI" not in d["Team"].tolist()


def test_season_from_date():
    assert data.season_from_date(pd.Timestamp("2024-01-14")) == 2023
    assert data.season_from_date(pd.Timestamp("2024-09-08")) == 2024


def test_legacy_loader_roundtrip(tmp_path):
    raw = legacy_rows()
    raw["Day"], raw["Game Number"] = "Sun", 1
    p = tmp_path / "legacy.xlsx"
    raw.to_excel(p, index=False)
    g = data.load_legacy_games(p)
    assert len(g) == 2
    gb = g[g["home"] == "GB"].iloc[0]
    assert (gb["away"], gb["home_score"], gb["away_score"], gb["season"]) == ("CHI", 24, 17, 1990)
    assert g["neutral"].sum() == 1
    assert (g["game_type"] == "REG").all()


def test_build_games_splits_sources_at_1999():
    leg = pd.DataFrame({"season": [1998, 1999], "game_type": "REG", "date": pd.to_datetime(["1998-09-06", "1999-09-12"]),
                        "game_id": ["a", "b"], "source": "legacy"})
    nfl = pd.DataFrame({"season": [1998, 1999, 1999], "game_type": ["REG", "REG", "WC"],
                        "date": pd.to_datetime(["1998-09-06", "1999-09-12", "2000-01-08"]),
                        "game_id": ["c", "d", "e"], "source": "nflverse"})
    g = data.build_games(leg, nfl)
    assert g["game_id"].tolist() == ["a", "d", "e"]
