"""The home hero follows the next kickoff slot: the exporter's kickoff_utc and the slot-picking function in site/slot.js."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import export_site  # noqa: E402

needs_inputs = pytest.mark.skipif(
    not (ROOT / "outputs").exists() or not (ROOT / "data" / "raw" / "schedules.csv").exists(),
    reason="needs outputs/ and data/raw/schedules.csv")


def games(*rows):
    return pd.DataFrame(rows, columns=["gameday", "gametime"])


def test_kickoff_utc_follows_daylight_saving():
    out = export_site.kickoff_utc_iso(games(
        ("2026-10-25", "13:00"),     # EDT: UTC-4
        ("2026-11-08", "13:00"),     # EST: UTC-5 (clocks changed 2026-11-01)
        ("2026-10-11", "20:20"),     # evening ET is the next UTC day
        ("2026-03-01", "09:30"),     # before the March change
        ("2026-03-15", "09:30")))
    assert out.tolist() == ["2026-10-25T17:00:00Z", "2026-11-08T18:00:00Z", "2026-10-12T00:20:00Z",
                            "2026-03-01T14:30:00Z", "2026-03-15T13:30:00Z"]


def test_kickoff_utc_is_none_without_a_usable_time():
    out = export_site.kickoff_utc_iso(games(("2026-10-25", None), ("2026-10-25", ""), ("2026-10-25", "TBD")))
    assert out.isna().all()


@needs_inputs
def test_every_upcoming_game_has_a_kickoff_utc():
    schedule, ratings, elo = export_site.load_schedule(), export_site.load_ratings(), export_site.load_elo()
    up = export_site.build_upcoming(schedule, ratings, elo)
    assert up["games"], "no upcoming games to check"
    for g in up["games"]:
        assert g["kickoff_utc"] and g["kickoff_utc"].endswith("Z"), g["game_id"]
        t = pd.Timestamp(g["kickoff_utc"]).tz_convert("America/New_York")
        assert str(t.date()) == g["date"] and t.strftime("%H:%M") == g["time"], g["game_id"]


def test_committed_upcoming_json_carries_kickoff_utc():
    up = json.loads((ROOT / "site" / "data" / "upcoming.json").read_text())
    assert all(g.get("kickoff_utc") for g in up["games"])


# ---- site/slot.js, run under node ----

NODE = shutil.which("node")
SLOT_JS = ROOT / "site" / "slot.js"

PROBE = """
const s = require(process.argv[1]);
const games = JSON.parse(process.argv[2]), now = Date.parse(process.argv[3]);
const slot = s.pickSlot(games, now);
console.log(JSON.stringify(slot && { start: new Date(slot.startMs).toISOString(), live: slot.live, ids: slot.games.map(g => g.id),
  label: s.slotLabel(slot, process.argv[4] || 'America/Chicago') }));
"""


def pick(games_, now, tz="America/Chicago"):
    out = subprocess.run([NODE, "-e", PROBE, str(SLOT_JS), json.dumps(games_), now, tz],
                         capture_output=True, text=True, check=True).stdout.strip()
    return json.loads(out)


WEEK = [
    {"id": "thu", "kickoff_utc": "2026-10-09T00:15:00Z"},
    {"id": "lon", "kickoff_utc": "2026-10-11T13:30:00Z"},
    {"id": "e1", "kickoff_utc": "2026-10-11T17:00:00Z"},
    {"id": "e2", "kickoff_utc": "2026-10-11T17:00:00Z"},
    {"id": "e3", "kickoff_utc": "2026-10-11T17:00:00Z"},
    {"id": "l1", "kickoff_utc": "2026-10-11T20:25:00Z"},
    {"id": "snf", "kickoff_utc": "2026-10-12T00:20:00Z"},
    {"id": "mnf", "kickoff_utc": "2026-10-13T00:15:00Z"},
    {"id": "tbd", "kickoff_utc": None},
]


NO_LONDON = [g for g in WEEK if g["id"] != "lon"]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
class TestSlotPicking:
    def test_before_a_multi_game_slot_it_is_next_up(self):
        r = pick(NO_LONDON, "2026-10-11T16:59:59Z")
        assert r["start"] == "2026-10-11T17:00:00.000Z" and r["live"] is False
        assert r["ids"] == ["e1", "e2", "e3"] and r["label"] == "Next up · Sun 12:00 PM CDT"

    def test_during_the_slot_it_is_live(self):
        r = pick(WEEK, "2026-10-11T18:30:00Z")
        assert r["live"] is True and r["ids"] == ["e1", "e2", "e3"]
        assert r["label"] == "Live now · kicked off 12:00 PM CDT"

    def test_a_slot_stays_up_until_three_and_a_half_hours_after_kickoff(self):
        assert pick(WEEK, "2026-10-11T20:29:59Z")["ids"] == ["e1", "e2", "e3"]       # 17:00Z + 3.5 h = 20:30Z
        assert pick(WEEK, "2026-10-11T20:30:00Z")["ids"] == ["l1"]

    def test_an_earlier_game_still_in_progress_stays_up(self):
        r = pick(WEEK, "2026-10-11T16:59:59Z")                 # London kicked off 13:30Z; 3.5 h later is 17:00Z
        assert r["ids"] == ["lon"] and r["live"] is True
        assert pick(WEEK, "2026-10-11T17:00:00Z")["ids"] == ["e1", "e2", "e3"]

    def test_a_single_game_slot(self):
        r = pick(WEEK, "2026-10-12T12:00:00Z")
        assert r["ids"] == ["mnf"] and r["live"] is False

    def test_after_the_last_game_there_is_no_slot(self):
        assert pick(WEEK, "2026-10-13T03:45:00Z") is None

    def test_games_without_a_kickoff_are_skipped_and_empty_input_is_safe(self):
        assert pick([{"id": "x", "kickoff_utc": None}], "2026-10-11T00:00:00Z") is None
        assert pick([], "2026-10-11T00:00:00Z") is None

    def test_input_order_does_not_matter(self):
        assert pick(list(reversed(WEEK)), "2026-10-11T16:00:00Z")["start"] == "2026-10-11T13:30:00.000Z"

    def test_label_uses_the_requested_zone_and_its_abbreviation(self):
        assert pick(NO_LONDON, "2026-10-11T16:00:00Z", "America/Los_Angeles")["label"] == "Next up · Sun 10:00 AM PDT"
