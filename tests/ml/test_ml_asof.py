"""The as-of rule: kickoff parsing, the earliest-kickoff-of-the-week cutoff, and its edge cases."""
import pandas as pd

from nflelo.ml import asof


def _sched(rows):
    return pd.DataFrame(rows, columns=["game_id", "season", "game_type", "week", "gameday", "gametime"])


def test_kickoff_is_eastern_and_missing_time_is_midnight():
    s = _sched([("a", 1999, "REG", 1, "1999-09-12", None), ("b", 2023, "REG", 1, "2023-09-07", "20:20")])
    k = asof.kickoff_times(s)
    assert str(k.dt.tz) == asof.TZ
    assert k.iloc[0] == pd.Timestamp("1999-09-12 00:00", tz=asof.TZ)
    assert k.iloc[1] == pd.Timestamp("2023-09-07 20:20", tz=asof.TZ)
    assert k.iloc[1].tz_convert("UTC") == pd.Timestamp("2023-09-08 00:20", tz="UTC")  # EDT is UTC-4


def test_as_of_is_earliest_kickoff_of_the_week():
    s = asof.add_asof(_sched([
        ("thu", 2023, "REG", 1, "2023-09-07", "20:20"),
        ("sun", 2023, "REG", 1, "2023-09-10", "13:00"),
        ("mon", 2023, "REG", 1, "2023-09-11", "20:15"),
        ("w2", 2023, "REG", 2, "2023-09-14", "20:15"),
        ("wc", 2023, "WC", 19, "2024-01-13", "16:30"),
        ("wc2", 2023, "WC", 19, "2024-01-14", "13:00"),
    ]))
    a = s.set_index("game_id")["as_of"]
    thu = pd.Timestamp("2023-09-07 20:20", tz=asof.TZ)
    assert a["thu"] == a["sun"] == a["mon"] == thu
    assert a["w2"] == pd.Timestamp("2023-09-14 20:15", tz=asof.TZ)
    assert a["wc"] == a["wc2"] == pd.Timestamp("2024-01-13 16:30", tz=asof.TZ)
    # The rule is "strictly before": the Thursday game itself is not known at as_of.
    assert asof.games_before(s, a["sun"]).empty
    assert set(asof.games_before(s, a["w2"])["game_id"]) == {"thu", "sun", "mon"}


def test_as_of_never_after_kickoff_and_missing_times_are_conservative():
    s = asof.add_asof(_sched([
        ("a", 1999, "REG", 1, "1999-09-12", None),
        ("b", 1999, "REG", 1, "1999-09-13", None),
    ]))
    assert (s["as_of"] <= s["kickoff"]).all()
    assert (s["as_of"] == pd.Timestamp("1999-09-12 00:00", tz=asof.TZ)).all()


def test_non_reg_post_rows_do_not_set_the_week_cutoff():
    s = asof.add_asof(_sched([
        ("pre", 2023, "PRE", 1, "2023-09-01", "19:00"),
        ("reg", 2023, "REG", 1, "2023-09-07", "20:20"),
    ]))
    a = s.set_index("game_id")["as_of"]
    assert a["reg"] == pd.Timestamp("2023-09-07 20:20", tz=asof.TZ)
    assert a["pre"] <= s.set_index("game_id").loc["pre", "kickoff"]


def test_attach_kickoff_rejects_unknown_games():
    s = asof.add_asof(_sched([("a", 2023, "REG", 1, "2023-09-07", "20:20")]))
    import pytest
    with pytest.raises(ValueError):
        asof.attach_kickoff(pd.DataFrame({"game_id": ["a", "zzz"]}), s)
    assert asof.attach_kickoff(pd.DataFrame({"game_id": ["a"]}), s)["kickoff"].notna().all()
