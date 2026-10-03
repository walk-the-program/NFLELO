import pandas as pd
import pytest

from nflelo import config, data


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


def csv538(tmp_path, rows):
    p = tmp_path / "e.csv"
    cols = "date,season,neutral,playoff,team1,team2,score1,score2"
    p.write_text(cols + "\n" + "\n".join(rows) + "\n")
    return p


def test_538_loader_maps_types_teams_and_weeks(tmp_path):
    # 538 never uses HOU before 2002 (Oilers are TEN); seeing it would mean a different convention.
    with pytest.raises(ValueError):
        data.load_538_games(csv538(tmp_path, ["1990-09-09,1990,0,,HOU,ARI,10,20"]))

    p = csv538(tmp_path, [
        "1990-09-09,1990,0,,TEN,ARI,10,20",       # Oilers (TEN), Cardinals (ARI); Sunday, week 1
        "1990-09-10,1990,0,,LAR,OAK,17,17",       # Monday, still week 1; tie
        "1990-09-16,1990,0,,IND,WSH,3,0",         # week 2
        "1991-01-05,1990,0,w,LAR,OAK,3,0",        # playoffs: week = last REG week + offset
        "1991-01-13,1990,0,d,IND,TEN,3,0",
        "1991-01-27,1990,0,c,IND,TEN,3,0",
        "1991-02-03,1990,1,s,LAR,IND,24,17",      # neutral Super Bowl
    ])
    g = data.load_538_games(p)
    assert len(g) == 7 and (g["source"] == "fivethirtyeight").all()
    reg = g[g["game_type"] == "REG"]
    assert reg[["home", "away", "week"]].values.tolist() == [["TEN", "ARI", 1], ["LA", "LV", 1], ["IND", "WAS", 2]]
    assert g["game_type"].tolist()[3:] == ["WC", "DIV", "CON", "SB"]
    assert g["week"].tolist()[3:] == [3, 4, 5, 6]
    assert g["neutral"].tolist() == [False] * 6 + [True]
    assert g["game_id"].is_unique


def test_calendar_weeks_tuesday_anchor():
    d = pd.Series(pd.to_datetime(["1993-09-05", "1993-09-06", "1993-09-12", "1993-10-14", "1994-01-03"]))
    assert data.calendar_weeks(d).tolist() == [1, 1, 2, 7, 18]


def test_unknown_playoff_code_raises(tmp_path):
    with pytest.raises(ValueError):
        data.load_538_games(csv538(tmp_path, ["1990-01-05,1990,0,x,LAR,OAK,3,0"]))


def test_build_games_splits_sources_at_1999():
    old = pd.DataFrame({"season": [1998, 1999], "game_type": "REG", "date": pd.to_datetime(["1998-09-06", "1999-09-12"]),
                        "game_id": ["a", "b"], "source": "fivethirtyeight"})
    nfl = pd.DataFrame({"season": [1998, 1999, 1999], "game_type": ["REG", "REG", "WC"],
                        "date": pd.to_datetime(["1998-09-06", "1999-09-12", "2000-01-08"]),
                        "game_id": ["c", "d", "e"], "source": "nflverse"})
    g = data.build_games(old, nfl)
    assert g["game_id"].tolist() == ["a", "d", "e"]


def test_validate_games_flags_score_and_missing_games():
    def mk(rows):
        return pd.DataFrame(rows, columns=["season", "game_type", "home", "away", "home_score", "away_score",
                                           "neutral", "date", "game_id"]).assign(date=lambda d: pd.to_datetime(d["date"]))
    a = mk([(1990, "REG", "GB", "CHI", 24, 17, False, "1990-09-09", "1"),
            (1990, "REG", "GB", "CHI", 10, 3, False, "1990-11-11", "2"),    # second meeting
            (1990, "REG", "NYJ", "NE", 7, 6, False, "1990-09-09", "3")])    # score differs
    b = mk([(1990, "REG", "CHI", "GB", 17, 24, False, "1990-09-09", "x"),   # home/away flipped
            (1990, "REG", "GB", "CHI", 10, 3, False, "1990-11-11", "y"),
            (1990, "REG", "NYJ", "NE", 7, 3, False, "1990-09-09", "z"),
            (1990, "REG", "DAL", "PHI", 7, 3, False, "1990-09-09", "w")])   # only in b
    v = data.validate_games(a, b, (1990, 1990))
    assert (v["n_a"], v["n_b"], v["n_key_matched"], v["n_exact"]) == (3, 4, 3, 2)
    assert v["mismatch_counts"] == {"score_differs": 1, "only_in_b": 1}
    assert v["home_agrees_rate"] == pytest.approx(2 / 3)   # the flipped first game disagrees


# --- the built dataset, when present (data/games.csv is regenerated by scripts/build.py)

@pytest.fixture(scope="module")
def games():
    if not config.GAMES_CSV.exists():
        pytest.skip("data/games.csv not built")
    return data.load_games()


def test_no_duplicate_games_and_clean_boundary(games):
    assert not games.duplicated(["season", "date", "home", "away"]).any()
    assert games["game_id"].is_unique
    sources = games.groupby("season")["source"].agg(lambda s: set(s))
    assert all(s == {"fivethirtyeight"} for s in sources.loc[:1998])
    assert all(s == {"nflverse"} for s in sources.loc[1999:])
    # 1998 playoffs end in January 1999 and belong to the 1998 season, not nflverse's 1999.
    last98 = games[games["season"] == 1998]["date"].max()
    first99 = games[games["season"] == 1999]["date"].min()
    assert last98 < first99


def test_pre_1999_playoffs_present(games):
    po = games[(games["season"] <= 1998) & (games["game_type"] != "REG")]
    assert len(po) == 269
    assert po.groupby("season").size().between(7, 15).all()
    assert set(po["game_type"]) == {"WC", "DIV", "CON", "SB"}
    assert (po[po["game_type"] == "SB"].groupby("season").size() == 1).all()


def expected_games_per_team(season):
    """Regular-season games per franchise: 14 in 1970-77, 16 after, minus strike seasons."""
    return {1982: 9, 1987: 15}.get(season, 14 if season <= 1977 else 16)


def test_franchise_game_counts_sane(games):
    reg = games[games["game_type"] == "REG"]
    for season, g in reg.groupby("season"):
        counts = pd.concat([g["home"], g["away"]]).value_counts()
        if season <= 1998:
            # Every franchise plays the same number of games; a mis-mapped code would break this.
            assert counts.nunique() == 1 and counts.iloc[0] == expected_games_per_team(season), season
        elif season <= 2025:
            # nflverse years: 16, then 17 from 2021; 2022 had one cancelled game.
            assert counts.max() - counts.min() <= 1, season
    teams = reg.groupby("season").apply(lambda g: len(set(g["home"]) | set(g["away"])), include_groups=False)
    assert (teams.loc[2002:] == 32).all()
    assert (teams.loc[1970], teams.loc[1976], teams.loc[1995], teams.loc[1999]) == (26, 28, 30, 31)
    assert teams.loc[1996] == 30 and not (reg["season"].between(1996, 1998) & ((reg["home"] == "CLE") | (reg["away"] == "CLE"))).any()
