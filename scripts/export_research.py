"""Export site/data/research.json: every NFLELO model and its pre-registered 2020-2025 result.

    python scripts/export_research.py

Feeds the public page site/models.html ("The models and their receipts").

Numbers come from the committed run registry (experiments/runs/*.json, sign-off runs have
`holdout: true` and names containing `signoff`). The registry is the source of truth for every
estimate and confidence interval. The few facts the registry does not hold are marked
`context/ml-results.md` in their `source` field. The hand-maintained part is the METADATA table
below: plain-English names, one-sentence descriptions, verdicts, readings, method docs and commits.

The live 2026 record and the shadow model are summarised from site/data/ml.json when it exists.
Output is deterministic: no timestamps, same inputs give identical bytes.
`scripts/export_site.py` calls export_research() at the end of export(), guarded so that a failure
here never breaks the main site export.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflelo import config  # noqa: E402

RUNS_DIR = config.ROOT / "experiments" / "runs"
SITE_DATA = config.ROOT / "site" / "data"
REPO_URL = "https://github.com/walk-the-program/NFLELO"
RESULTS_DOC = "context/ml-results.md"
SCHEMA = 1

VERDICT_LABELS = {
    "pass": "Pass",
    "fail": "Fail",
    "negative": "Negative result",
    "shadow": "Not tested, shadow live",
}
VERDICT_ORDER = ("pass", "fail", "negative", "shadow")

# The M7a-v2 conversion fix is frozen and waits for the 2026 season; it is a status line, not a card.
FORWARD_TEST = {
    "name": "Fourth-down conversion fix (M7a-v2)",
    "status": "Frozen at 5a0b850, runs after 2027-02-15",
    "frozen_commit": "5a0b850",
    "registered_commit": "201ad8e",
    "runs_after": "2027-02-15",
    "text": ("The corrected fourth-down conversion engine was frozen before any 2026 game was played. "
             "It is scored once, on the 2026 season, after the 2027-02-15 data release, with no refit."),
    "dev_note": ("On development seasons its conversion calibration error was 0.019 against 0.033 for the first "
                 "version, and the 4th-and-2 gap was +0.004 against +0.034."),
    "dev_source": RESULTS_DOC,
    "doc": "context/ml.md",
}

# --------------------------------------------------------------------------- metadata (hand-maintained)
# One entry per model, in scoreboard order. `rules_commit` is the commit that wrote the pass/fail rules
# before the run; `result_commit` is the commit that recorded the result. Both are None when a model had
# no holdout run (its rules were never frozen for one).

METADATA = [
    {"id": "M3", "short": "M3", "name": "Game model",
     "what": "Win chance for each game from Elo, opponent-adjusted efficiency (EPA, expected points added) and the starting quarterback.",
     "verdict": "pass", "tested_holdout": True, "license": "CC BY",
     "reading": "It beat Elo in all six seasons. The market is still ahead. Its calibration error missed a bar that was too tight for 1,615 games.",
     "method_doc": "context/ml-m3-method.md", "rules_commit": "eb65d61", "result_commit": "5a66c9a"},
    {"id": "M4-margin", "short": "M4", "name": "Margin model",
     "what": "The full spread of possible final margins, aware that 3 and 7 are common margins.",
     "verdict": "fail", "tested_holdout": True, "license": "CC BY",
     "reading": "Its error was lower than Elo's spread, but the interval includes zero, so the edge is not proven. It still drives the site's spreads and the simulation.",
     "method_doc": "context/ml-m4-method.md", "rules_commit": "e51e890", "result_commit": "816c327"},
    {"id": "M4-odds", "short": "M4", "name": "Playoff odds",
     "what": "20,000 simulated seasons with the real NFL tiebreakers and a season-long strength shock for each team.",
     "verdict": "pass", "tested_holdout": True, "license": "CC BY",
     "reading": "Adding the strength shock improved the made-playoffs forecast, and the odds were calibrated. Live on the site.",
     "method_doc": "context/ml-m4-method.md", "rules_commit": "e51e890", "result_commit": "816c327"},
    {"id": "M5", "short": "M5", "name": "Lineup values",
     "what": "A value for each player from box-score stats, and the probable lineup each team will field.",
     "verdict": "negative", "tested_holdout": False, "license": "CC BY",
     "reading": "Knowing who plays added nothing measurable to the game model on development seasons, so it was never sent to the holdout.",
     "method_doc": "context/ml-m5-method.md", "rules_commit": None, "result_commit": "a91d8a3"},
    {"id": "M5b", "short": "M5b", "name": "On-field player ratings",
     "what": "A plus-minus rating for every player, from who was on the field for each of 177,545 plays (2016 on).",
     "verdict": "fail", "tested_holdout": True, "license": "CC BY-SA",
     "reading": "Beat team ratings, but lost to simply adding up box-score values. Public data cannot split credit among the 22 players better than that. Kept as a descriptive view.",
     "method_doc": "context/ml-m5b-method.md", "rules_commit": "8956c97", "result_commit": "e83f7ac"},
    {"id": "M3b", "short": "M3b", "name": "Kalman strength model",
     "what": "A team-strength rating that carries its own uncertainty, combined with Elo and the quarterback term.",
     "verdict": "shadow", "tested_holdout": False, "license": "CC BY",
     "reading": "It narrowly missed the bar on development seasons, so no holdout was spent on it. It is logged before kickoff next to the live model as a clean forward test.",
     "method_doc": "context/ml-m3b-method.md", "rules_commit": None, "result_commit": "ca705de"},
    {"id": "M6", "short": "M6", "name": "Play outcome model",
     "what": "The whole distribution of yards a play can gain, from the pre-snap situation and personnel.",
     "verdict": "pass", "tested_holdout": True, "license": "CC BY-SA",
     "reading": "More accurate than the situation-only baseline in every season and in both data eras. Knowing the individual players adds only a small amount.",
     "method_doc": "context/ml-m6-method.md", "rules_commit": "bc0bff7", "result_commit": "fe61c0e"},
    {"id": "M7a-wp", "short": "M7a", "name": "Win-probability model",
     "what": "Win chance at every snap of a game, built without any betting-market input.",
     "verdict": "pass", "tested_holdout": True, "license": "CC BY",
     "reading": "More accurate than the public nflfastR model. The Vegas-based probability is still better.",
     "method_doc": "context/ml-m7-method.md", "rules_commit": "a4a24b1", "result_commit": "4e509da"},
    {"id": "M7a-4th", "short": "M7a", "name": "Fourth-down tool",
     "what": "Go for it, kick a field goal, or punt, compared in win probability with an uncertainty band.",
     "verdict": "fail", "tested_holdout": True, "license": "CC BY-SA",
     "reading": "The field-goal and punt parts passed. The go-for-it conversion estimate was off, so the tool failed. A corrected version is frozen for 2026.",
     "method_doc": "context/ml-m7-method.md", "rules_commit": "a4a24b1", "result_commit": "4e509da"},
    {"id": "M7b", "short": "M7b", "name": "Early-down play calling",
     "what": "Run or pass, and from which personnel group, on first and second down, using causal methods.",
     "verdict": "pass", "tested_holdout": True, "license": "CC BY-SA",
     "reading": "The recommended mix beats what teams actually did, and the placebo check came back near zero. Defenses would adapt, so treat it as a direction, not a guarantee.",
     "method_doc": "context/ml-m7-method.md", "rules_commit": "92ae084", "result_commit": "7b3e703"},
]

MODEL_IDS = [m["id"] for m in METADATA]


# --------------------------------------------------------------------------- registry access

class RunMissing(RuntimeError):
    pass


def load_run(name: str, runs_dir: Path | None = None, *, holdout: bool | None = None) -> tuple[dict, str]:
    """The latest registry run called `name`, and its file name. `holdout` is checked when given."""
    runs_dir = RUNS_DIR if runs_dir is None else runs_dir
    for f in sorted(runs_dir.glob(f"*_{name}.json"), reverse=True):
        d = json.loads(f.read_text())
        if d.get("name") != name:
            continue
        if holdout is not None and bool(d.get("holdout")) is not holdout:
            raise RunMissing(f"run {f.name}: holdout flag is {d.get('holdout')}, expected {holdout}")
        return d, f.name
    raise RunMissing(f"no registry run named {name!r} in {runs_dir}")


def src(file_name: str) -> str:
    return f"experiments/runs/{file_name}"


def sgn(x: float, d: int) -> str:
    """Signed fixed-point text with an ASCII minus; a value that rounds to zero is shown as +0."""
    r = round(x, d)
    return ("+" if r >= 0 else "-") + f"{abs(r):.{d}f}"


def interval(est: float, lo: float, hi: float, d: int) -> str:
    return f"{sgn(est, d)} [{sgn(lo, d)}, {sgn(hi, d)}]"


def comma(n: int) -> str:
    return f"{int(n):,}"


def paired(label: str, cmp_: dict, d: int, *, n_unit: str, window: str, file_name: str, better: str = "lower") -> dict:
    """A primary metric that is a paired difference with a bootstrap interval."""
    est, lo, hi = cmp_["diff"], cmp_["ci_low"], cmp_["ci_high"]
    return {"label": label, "better": better, "estimate": est, "ci_low": lo, "ci_high": hi, "decimals": d,
            "kind": "difference", "text": interval(est, lo, hi, d), "n": cmp_["n"], "n_unit": n_unit,
            "window": window, "excludes_zero": bool(cmp_["excludes_zero"]), "level": cmp_.get("level", 0.95),
            "source": src(file_name)}


# --------------------------------------------------------------------------- per-model numbers

def model_numbers(mid: str, runs_dir: Path | None) -> dict:
    """The registry-backed numbers for one model: primary metric, a few `also` facts, chart rows."""
    reg = "regular seasons 2020 to 2025"

    if mid == "M3":
        a, fa = load_run("m3_signoff_A4s", runs_dir, holdout=True)
        e, fe = load_run("m3_signoff_elo_v2", runs_dir, holdout=True)
        m, fm = load_run("m3_signoff_market", runs_dir, holdout=True)
        c = a["comparisons"]["A4s_minus_elo_brier"]
        prim = paired("Brier score, model minus Elo (lower is better)", c, 4, n_unit="games", window=reg, file_name=fa)
        also = [{"text": f"Brier {a['metrics']['brier']:.4f} against Elo {e['metrics']['brier']:.4f} and the Vegas market "
                         f"{m['metrics']['brier']:.4f}", "source": src(fa)}]
        return {"primary": prim, "also": also}

    if mid == "M4-margin":
        d, f = load_run("m4_signoff_margin", runs_dir, holdout=True)
        r = d["result"]["mae"]
        prim = paired("Average margin error in points, model minus Elo's spread (lower is better)", r["model_minus_elo"], 3,
                      n_unit="games", window=reg, file_name=f)
        also = [{"text": f"Error {r['model']:.3f} points against Elo {r['elo']:.3f} and the market line {r['market']:.3f}",
                 "source": src(f)}]
        return {"primary": prim, "also": also}

    if mid == "M4-odds":
        d, f = load_run("m4_signoff_playoff_odds", runs_dir, holdout=True)
        pm = d["result"]["metrics"]["playoffs"]
        prim = paired("Made-playoffs Brier score, with strength shocks minus without (lower is better)", pm["diff"], 4,
                      n_unit="team forecasts", window=reg + ", 4 starting weeks each", file_name=f)
        also = [{"text": f"Brier {pm['brier']:.4f} against {pm['brier_tau0']:.4f} without the shocks",
                 "source": src(f)},
                {"text": "The tiebreaker code reproduces all 48 actual conference seedings, 2002 to 2025",
                 "source": RESULTS_DOC}]
        return {"primary": prim, "also": also}

    if mid == "M5":
        d, f = load_run("m5_B3", runs_dir)
        if d.get("holdout"):
            raise RunMissing("m5_B3 should be a development run")
        b1, f1 = load_run("m5_B1", runs_dir)
        b2, f2 = load_run("m5_B2", runs_dir)
        prim = paired("Brier score, best lineup model minus the game model (lower is better)", d["comparisons"]["vs_B0_brier"], 4,
                      n_unit="games", window="development seasons 2012 to 2019, no holdout run", file_name=f)
        also = [{"text": (f"Smaller steps: {sgn(b1['comparisons']['vs_B0_brier']['diff'], 4)} and "
                          f"{sgn(b2['comparisons']['vs_B0_brier']['diff'], 4)}; every interval includes zero"),
                 "source": src(f1)}]
        return {"primary": prim, "also": also}

    if mid == "M5b":
        d, f = load_run("m5b_signoff_rapm", runs_dir, holdout=True)
        c = d["comparisons"]
        prim = paired("Play-level squared error, RAPM minus box-score values (lower is better)", c["box"], 4,
                      n_unit="plays", window="seasons 2020 to 2025, forecast one season ahead", file_name=f)
        also = [{"text": f"Against team ratings: {interval(c['team']['diff'], c['team']['ci_low'], c['team']['ci_high'], 4)}",
                 "source": src(f)}]
        return {"primary": prim, "also": also}

    if mid == "M3b":
        d, f = load_run("m3b_C2d", runs_dir)
        if d.get("holdout"):
            raise RunMissing("m3b_C2d should be a development run")
        prim = paired("Brier score, Kalman candidate minus the live model (lower is better)", d["comparisons"]["vs_a4s"], 4,
                      n_unit="games", window="development seasons 2006 to 2019, no holdout run", file_name=f)
        ll = d["comparisons"]["vs_a4s_logloss"]
        also = [{"text": f"Log loss, the stricter score: {interval(ll['diff'], ll['ci_low'], ll['ci_high'], 4)}",
                 "source": src(f)}]
        return {"primary": prim, "also": also}

    if mid == "M6":
        sit, fs = load_run("m6_signoff_situation_gbm", runs_dir, holdout=True)
        call, fc = load_run("m6_signoff_call_gbm", runs_dir, holdout=True)
        cs, cc = sit["result"]["vs_baseline"]["crps"], call["result"]["vs_baseline"]["crps"]
        prim = paired("Yards error (CRPS), model minus the situation baseline (lower is better)", cs, 4,
                      n_unit="plays", window="seasons 2020 to 2025, forecast one season ahead", file_name=fs)
        also = [{"text": f"With the play call known: {interval(cc['diff'], cc['ci_low'], cc['ci_high'], 4)}",
                 "source": src(fc)}]
        return {"primary": prim, "also": also}

    if mid == "M7a-wp":
        d, f = load_run("m7a_signoff_wp", runs_dir, holdout=True)
        cmpd = d["diffs"]
        vg = cmpd["m7a_wp-vegas_wp"]["brier"]
        prim = paired("Brier score, our model minus nflfastR (lower is better)", cmpd["m7a_wp-nflfastr_wp"]["brier"], 4,
                      n_unit="plays", window="regular seasons 2020 to 2025", file_name=f)
        also = [{"text": f"Against the Vegas-based probability: {interval(vg['diff'], vg['ci_low'], vg['ci_high'], 4)} (Vegas is better)",
                 "source": src(f)}]
        return {"primary": prim, "also": also}

    if mid == "M7a-4th":
        d, f = load_run("m7a_signoff_conversion", runs_dir, holdout=True)
        fg, ffg = load_run("m7a_signoff_fg", runs_dir, holdout=True)
        pt, fpt = load_run("m7a_signoff_punt", runs_dir, holdout=True)
        cal = d["metrics"]["engine_on_attempts"]["cal"]
        pc = pt["metrics"]["crps_vs_baseline"]
        prim = {"label": "Go-for-it conversion calibration error (ECE) against its limit (lower is better)",
                "better": "lower", "estimate": cal["ece"], "ci_low": None, "ci_high": None, "limit": cal["null_p95"],
                "decimals": 3, "kind": "calibration", "level": None,
                "text": f"{cal['ece']:.3f} against a limit of {cal['null_p95']:.3f}",
                "n": d["metrics"]["n"], "n_unit": "go-for-it attempts", "window": "regular seasons 2020 to 2025",
                "excludes_zero": None, "source": src(f)}
        also = [{"text": f"Field goal make rate calibration {fg['metrics']['ece']:.3f}: passed", "source": src(ffg)},
                {"text": f"Punts: {interval(pc['diff'], pc['ci_low'], pc['ci_high'], 3)} yards of error against the by-spot baseline: passed",
                 "source": src(fpt)}]
        return {"primary": prim, "also": also}

    if mid == "M7b":
        d, f = load_run("m7b_signoff_ope", runs_dir, holdout=True)
        pl, fpl = load_run("m7b_signoff_placebo", runs_dir, holdout=True)
        o = d["metrics"]["ope_epa"]
        p = pl["metrics"]["ope_epa"]
        prim = paired("EPA per early-down play, recommended mix minus observed (higher is better)", o, 3,
                      n_unit="plays", window="seasons 2020 to 2025, forecast one season ahead", file_name=f, better="higher")
        also = [{"text": f"Placebo check, should be about zero: {interval(p['diff'], p['ci_low'], p['ci_high'], 3)}",
                 "source": src(fpl)}]
        return {"primary": prim, "also": also}

    raise KeyError(mid)


# --------------------------------------------------------------------------- chart, findings, live

def build_chart(runs_dir: Path | None) -> dict:
    """The game-model holdout comparison: Elo, the model and the market, 2020 to 2025.

    The registry holds paired intervals for the gaps, not intervals for each score. Each whisker is the
    model's score plus the 95% interval of that row's gap to the model (paired bootstrap, 2,000 reps).
    """
    a, fa = load_run("m3_signoff_A4s", runs_dir, holdout=True)
    e, _ = load_run("m3_signoff_elo_v2", runs_dir, holdout=True)
    m, _ = load_run("m3_signoff_market", runs_dir, holdout=True)
    cmp_ = a["comparisons"]
    model_b = a["metrics"]["brier"]
    # row minus model = minus (model minus row), so the interval flips sign and swaps ends
    ce, cm = cmp_["A4s_minus_elo_brier"], cmp_["A4s_minus_market_brier"]
    rows = []
    for rid, label, series, b, c in (("elo", "Elo v2", "s1", e["metrics"]["brier"], ce),
                                     ("model", "NFLELO game model", "s3", model_b, None),
                                     ("market", "Vegas market (benchmark only)", "s2", m["metrics"]["brier"], cm)):
        row = {"id": rid, "label": label, "series": series, "brier": b, "logloss": None}
        if c is not None:
            row["gap"] = b - model_b
            row["gap_ci_low"] = -c["ci_high"]
            row["gap_ci_high"] = -c["ci_low"]
            row["whisker_low"] = model_b + row["gap_ci_low"]
            row["whisker_high"] = model_b + row["gap_ci_high"]
        rows.append(row)
    ll = {"elo": e["metrics"]["logloss"], "model": a["metrics"]["logloss"], "market": m["metrics"]["logloss"]}
    for r in rows:
        r["logloss"] = ll[r["id"]]
    seasons = [{"season": s["season"], "n": s["n"], "elo": s["brier_elo_v2"], "model": s["brier_A4s"],
                "market": s["brier_market"]} for s in a["per_season"]]
    return {"metric": "Brier score", "n": a["metrics"]["n"], "window": a["window"]["seasons"], "rows": rows,
            "seasons": seasons, "model_beats_elo_seasons": sum(1 for s in seasons if s["model"] < s["elo"]),
            "seasons_total": len(seasons), "source": src(fa),
            "whisker_note": "Whiskers show the 95% interval of each row's gap to the model (paired bootstrap)."}


def build_findings(runs_dir: Path | None) -> list[dict]:
    a, fa = load_run("m3_signoff_A4s", runs_dir, holdout=True)
    m, _ = load_run("m3_signoff_market", runs_dir, holdout=True)
    audit, fau = load_run("m7a_signoff_audit", runs_dir, holdout=True)
    wins = audit["metrics"]["wp_lost_per_team_game"]
    return [
        {"n": 1, "title": "The quarterback is most of the story.",
         "text": "In the game model, nearly all of the gain over Elo comes from knowing the starting quarterback, and it depends on knowing who actually starts.",
         "source": RESULTS_DOC},
        {"n": 2, "title": "Player-level beats team-level, but public data limits individual credit.",
         "text": ("Box-score credit summed over the players on the field beats team ratings when predicting the next season. "
                  "On-field plus-minus does not beat box credit, and lineup effects do not move game predictions."),
         "source": RESULTS_DOC},
        {"n": 3, "title": "Elo's edge is memory.",
         "text": ("Elo carries what it learned across seasons. On development seasons our features alone tie Elo overall: "
                  "Elo wins in weeks 1 to 4, and our features win in weeks 10 to 18."),
         "source": RESULTS_DOC},
        {"n": 4, "title": "Teams are still too conservative.",
         "text": (f"The fourth-down audit estimates about {wins:.2f} wins per team-game given up, mostly by punting "
                  "(model-estimated). The early-down policy says pass more, especially from 12 personnel on 1st and 10."),
         "source": src(fau)},
        {"n": 5, "title": "The market is still better at game prediction.",
         "text": (f"Brier {m['metrics']['brier']:.4f} against our {a['metrics']['brier']:.4f} on 2020 to 2025. "
                  "Our models are independent of the market by design, so the edge we measured over Elo and nflfastR is real."),
         "source": src(fa)},
    ]


def build_live(site_data: Path) -> dict | None:
    """The live 2026 record: the model and the shadow, from ml.json. None when ml.json does not exist."""
    path = site_data / "ml.json"
    if not path.exists():
        return None
    try:
        ml = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    final = (ml.get("live") or {}).get("final") or {}
    sh = ml.get("shadow") or {}
    return {
        "season": ml.get("season"),
        "from_week": ml.get("live_from_week"),
        "from_date": ml.get("live_from_date"),
        "last_run_utc": ml.get("last_run_utc"),
        "ledger_url": ml.get("ledger_url"),
        "ledger_rows": ml.get("ledger_rows"),
        "model": {"n_scored": int(final.get("n") or 0),
                  "brier": (final.get("model") or {}).get("brier"),
                  "elo_brier": (final.get("elo") or {}).get("brier")},
        "shadow": None if not sh else {
            "name": "Kalman strength model (M3b)",
            "n_scored": int(sh.get("n") or 0),
            "brier": (sh.get("shadow") or {}).get("brier"),
            "model_brier_same_games": (sh.get("model") or {}).get("brier"),
            "brier_diff": sh.get("brier_diff"),
            "ledger_url": sh.get("ledger_url"),
        },
    }


# --------------------------------------------------------------------------- assembly

def count_verdicts(models: list[dict]) -> dict:
    counts = {v: sum(1 for m in models if m["verdict"] == v) for v in VERDICT_ORDER}
    counts["total"] = len(models)
    counts["holdout_tested"] = sum(1 for m in models if m["tested_holdout"])
    counts["holdout_pass"] = sum(1 for m in models if m["tested_holdout"] and m["verdict"] == "pass")
    counts["holdout_fail"] = sum(1 for m in models if m["tested_holdout"] and m["verdict"] == "fail")
    return counts


def headlines(counts: dict, chart: dict) -> dict:
    t, p, f = counts["holdout_tested"], counts["holdout_pass"], counts["holdout_fail"]
    hero = f"{t} models tested once on seasons they never saw. {p} passed, {f} failed."
    parts = [f"{counts['pass']} passed", f"{counts['fail']} failed"]
    if counts["negative"]:
        parts.append(f"{counts['negative']} negative result" + ("" if counts["negative"] == 1 else "s"))
    if counts["shadow"]:
        parts.append(f"{counts['shadow']} not yet tested")
    scoreboard = ", ".join(parts) + "."
    gap = next(r for r in chart["rows"] if r["id"] == "market")["gap"]
    beat = chart["model_beats_elo_seasons"]
    n_s = chart["seasons_total"]
    season_text = f"all {n_s} seasons" if beat == n_s else f"{beat} of {n_s} seasons"
    game = f"The game model beat Elo in {season_text} and finished {abs(gap):.4f} Brier behind the market."
    extra = counts["negative"] + counts["shadow"]
    deck = (f"{extra} more are reported as development-only results: {counts['negative']} negative "
            f"and {counts['shadow']} still running live as a shadow model.")
    return {"hero": hero, "hero_deck": deck, "scoreboard": scoreboard, "chart": game}


def build_research(runs_dir: Path | None = None, site_data: Path | None = None) -> dict:
    site_data = SITE_DATA if site_data is None else site_data
    models = []
    for meta in METADATA:
        nums = model_numbers(meta["id"], runs_dir)
        models.append({**meta, "verdict_label": VERDICT_LABELS[meta["verdict"]], **nums})
    counts = count_verdicts(models)
    chart = build_chart(runs_dir)
    return {
        "schema": SCHEMA,
        "title": "The models and their receipts",
        "holdout_seasons": [2020, 2025],
        "repo": REPO_URL,
        "headlines": headlines(counts, chart),
        "counts": counts,
        "verdict_labels": VERDICT_LABELS,
        "models": models,
        "chart": chart,
        "findings": build_findings(runs_dir),
        "live": build_live(site_data),
        "forward_test": FORWARD_TEST,
        "sources": {"registry": "experiments/runs", "scoreboard_doc": RESULTS_DOC},
    }


def export_research(site_data: Path | None = None, runs_dir: Path | None = None) -> int:
    """Write research.json into `site_data` (default site/data). Returns the bytes written."""
    site_data = SITE_DATA if site_data is None else Path(site_data)
    obj = build_research(runs_dir, site_data)
    site_data.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, separators=(",", ":"), allow_nan=False)
    (site_data / "research.json").write_text(text + "\n")
    return len(text) + 1


if __name__ == "__main__":
    size = export_research()
    print(f"site/data/research.json: {size / 1024:.0f} KB")
