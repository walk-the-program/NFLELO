import pytest

from nflelo.teams import FRANCHISES, franchise


@pytest.mark.parametrize("code,season,expected", [
    ("STL", 1975, "ARI"),   # St. Louis Cardinals
    ("STL", 2005, "LA"),    # St. Louis Rams
    ("OAK", 1985, "LV"),
    ("RAI", 1985, "LV"),
    ("SD", 2010, "LAC"),
    ("SDG", 1990, "LAC"),
    ("HOU", 1990, "TEN"),   # Houston Oilers
    ("HOU", 2005, "HOU"),   # Houston Texans
    ("BAL", 1980, "IND"),   # Baltimore Colts
    ("BAL", 2000, "BAL"),   # Ravens
    ("RAM", 1975, "LA"),
    ("PHO", 1996, "ARI"),
    ("BOS", 1970, "NE"),
    ("CLE", 1985, "CLE"),
    ("CLE", 2005, "CLE"),
    ("LAR", 2018, "LA"),
    ("LA", 2016, "LA"),
    ("OTI", 1998, "TEN"),
])
def test_franchise_mapping(code, season, expected):
    assert franchise(code, season) == expected


def test_all_outputs_are_canonical():
    for code in ["GNB", "KAN", "NWE", "NOR", "SFO", "TAM", "WSH", "JAC", "LVR", "CLT"]:
        assert franchise(code, 2000) in FRANCHISES


def test_unknown_code_raises():
    with pytest.raises(ValueError):
        franchise("XYZ", 2000)
