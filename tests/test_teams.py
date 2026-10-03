import pytest

from nflelo.teams import FRANCHISES, franchise, franchise_538


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


@pytest.mark.parametrize("code,season,expected", [
    ("TEN", 1975, "TEN"),   # 538 uses TEN for the Houston Oilers
    ("IND", 1975, "IND"),   # Baltimore Colts
    ("BAL", 1996, "BAL"),   # Ravens
    ("HOU", 2002, "HOU"),   # Texans
    ("ARI", 1980, "ARI"),   # St. Louis Cardinals
    ("LAR", 1985, "LA"),
    ("OAK", 1985, "LV"),
    ("LAC", 1990, "LAC"),
    ("WSH", 1990, "WAS"),
    ("CLE", 1985, "CLE"),
])
def test_franchise_538(code, season, expected):
    assert franchise_538(code, season) == expected


def test_franchise_538_all_codes_canonical():
    from nflelo.teams import _CODES_538
    assert set(_CODES_538.values()) == FRANCHISES
    assert len(_CODES_538) == 32


@pytest.mark.parametrize("code,season", [("BAL", 1990), ("HOU", 1990), ("STL", 1990), ("GNB", 1990), ("XYZ", 2000)])
def test_franchise_538_rejects_other_conventions(code, season):
    with pytest.raises(ValueError):
        franchise_538(code, season)
