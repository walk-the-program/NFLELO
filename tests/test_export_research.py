"""The models page data: scripts/export_research.py and its guarded call from scripts/export_site.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import export_research  # noqa: E402

EXPECTED_IDS = ["M3", "M4-margin", "M4-odds", "M5", "M5b", "M3b", "M6", "M7a-wp", "M7a-4th", "M7b"]

needs_registry = pytest.mark.skipif(not export_research.RUNS_DIR.exists(), reason="needs experiments/runs")


@pytest.fixture
def research(tmp_path):
    return export_research.build_research(site_data=tmp_path)


@needs_registry
def test_output_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    export_research.export_research(a)
    export_research.export_research(b)
    assert (a / "research.json").read_bytes() == (b / "research.json").read_bytes()


@needs_registry
def test_every_model_id_is_present_once_in_scoreboard_order(research):
    assert [m["id"] for m in research["models"]] == EXPECTED_IDS
    assert export_research.MODEL_IDS == EXPECTED_IDS


@needs_registry
def test_verdict_counts_match_the_metadata(research):
    meta = export_research.METADATA
    counts = research["counts"]
    for v in export_research.VERDICT_ORDER:
        assert counts[v] == sum(1 for m in meta if m["verdict"] == v)
        assert counts[v] == sum(1 for m in research["models"] if m["verdict"] == v)
    assert counts["total"] == len(meta) == 10
    assert counts["holdout_tested"] == sum(1 for m in meta if m["tested_holdout"])
    assert counts["holdout_pass"] + counts["holdout_fail"] == counts["holdout_tested"]
    assert counts["pass"] + counts["fail"] + counts["negative"] + counts["shadow"] == counts["total"]
    # the hero sentence is counted from the data
    assert research["headlines"]["hero"].startswith(f"{counts['holdout_tested']} models tested once")
    assert f"{counts['holdout_pass']} passed" in research["headlines"]["hero"]
    # only models with a holdout run have frozen rules; every verdict has a label
    for m in research["models"]:
        assert m["verdict_label"] == export_research.VERDICT_LABELS[m["verdict"]]
        assert bool(m["rules_commit"]) == m["tested_holdout"], m["id"]


@needs_registry
def test_numbers_come_from_the_registry_and_signoff_runs_are_holdout(research):
    runs = export_research.RUNS_DIR
    for m in research["models"]:
        p = m["primary"]
        d = json.loads((runs / Path(p["source"]).name).read_text())
        assert bool(d["holdout"]) == m["tested_holdout"], m["id"]
        if m["tested_holdout"]:
            assert "signoff" in d["name"] and d["window"]["seasons"] == [2020, 2025], m["id"]
        if p["kind"] == "difference":
            assert p["ci_low"] <= p["estimate"] <= p["ci_high"]
            assert p["excludes_zero"] == (p["ci_low"] > 0 or p["ci_high"] < 0)
    by_id = {m["id"]: m for m in research["models"]}
    assert by_id["M3"]["primary"]["text"] == "-0.0035 [-0.0063, -0.0007]"
    assert by_id["M4-margin"]["primary"]["excludes_zero"] is False      # the failed primary
    assert by_id["M7a-4th"]["primary"]["estimate"] > by_id["M7a-4th"]["primary"]["limit"]


@needs_registry
def test_page_copy_has_no_em_or_en_dashes_or_betting_words(research):
    text = json.dumps(research, ensure_ascii=False)
    assert "—" not in text and "–" not in text
    for banned in ("units", "ATS", "against the spread", "betting pick"):
        assert banned not in text, banned


@needs_registry
def test_chart_matches_the_game_model_signoff(research):
    c = research["chart"]
    rows = {r["id"]: r for r in c["rows"]}
    assert c["n"] == 1615 and c["window"] == [2020, 2025]
    assert rows["market"]["brier"] < rows["model"]["brier"] < rows["elo"]["brier"]
    assert rows["elo"]["whisker_low"] < rows["elo"]["brier"] < rows["elo"]["whisker_high"]
    assert rows["market"]["whisker_low"] < rows["market"]["brier"] < rows["market"]["whisker_high"]
    assert c["model_beats_elo_seasons"] == 6 == c["seasons_total"] == len(c["seasons"])


def test_live_summary_reads_ml_json(tmp_path):
    assert export_research.build_live(tmp_path) is None
    (tmp_path / "ml.json").write_text(json.dumps({
        "season": 2026, "live_from_week": 5, "live_from_date": "2026-10-08", "last_run_utc": "x", "ledger_url": "u",
        "ledger_rows": 9,
        "live": {"final": {"n": 3, "model": {"brier": 0.2}, "elo": {"brier": 0.21}}},
        "shadow": {"n": 3, "shadow": {"brier": 0.19}, "model": {"brier": 0.2}, "brier_diff": -0.01, "ledger_url": "s"}}))
    live = export_research.build_live(tmp_path)
    assert live["model"] == {"n_scored": 3, "brier": 0.2, "elo_brier": 0.21}
    assert live["shadow"]["n_scored"] == 3 and live["shadow"]["brier"] == 0.19
    (tmp_path / "ml.json").write_text("not json")
    assert export_research.build_live(tmp_path) is None


def test_forward_test_status_is_recorded():
    assert export_research.FORWARD_TEST["status"] == "Frozen at 5a0b850, runs after 2027-02-15"


def test_missing_registry_run_raises_a_clear_error(tmp_path):
    with pytest.raises(export_research.RunMissing):
        export_research.load_run("m3_signoff_A4s", tmp_path)


def test_export_site_survives_export_research_failing(tmp_path, monkeypatch, capsys):
    import export_site

    def boom(*a, **k):
        raise RuntimeError("registry is gone")

    monkeypatch.setattr(export_site, "SITE_DATA", tmp_path)
    monkeypatch.setattr(export_research, "export_research", boom)
    assert export_site.export_research_safely() is None
    assert "research.json was not written" in capsys.readouterr().err
    assert not (tmp_path / "research.json").exists()


INPUTS = [export_research.config.OUT_DIR / "elo_games.csv", export_research.config.OUT_DIR / "ratings_current.json",
          export_research.config.RAW_DIR / "schedules.csv"]


@pytest.mark.skipif(not all(p.exists() for p in INPUTS), reason="needs outputs/ and data/raw/schedules.csv")
def test_full_export_still_succeeds_when_export_research_raises(tmp_path, monkeypatch):
    import export_site
    monkeypatch.setattr(export_site, "SITE_DATA", tmp_path)

    def boom(*a, **k):
        raise RuntimeError("registry is gone")

    monkeypatch.setattr(export_research, "export_research", boom)
    sizes = export_site.export()
    assert "ladder.json" in sizes and (tmp_path / "ladder.json").exists()
    assert not (tmp_path / "research.json").exists()
