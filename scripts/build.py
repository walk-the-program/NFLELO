"""One command: pull data, validate it, run Elo, write outputs.

    python scripts/build.py            # download fresh nflverse data
    python scripts/build.py --offline  # reuse data/raw/schedules.csv

Uses the vendored FiveThirtyEight file in data/sources/ for 1970-1998.
Writes data/games.csv, outputs/elo_games.csv, outputs/ratings_current.json,
outputs/model_report.md, outputs/dashboards/team_summary.png, site/data/*.json.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflelo import config, data, evaluate  # noqa: E402
from nflelo.elo import run_elo  # noqa: E402

CFG_COLS = ["k", "lam", "hfa_mode", "hfa", "hfa_init", "k_hfa", "hfa_mov", "mov_mode",
            "mov_cap", "expansion_start", "include_playoffs"]


def md_table(df: pd.DataFrame, floats: int = 4) -> str:
    cols = list(df.columns)
    metric = {"brier", "logloss", "accuracy", "actual_home_win_rate"}

    def fmt(col, v):
        if not isinstance(v, float):
            return str(v)
        return f"{v:.{floats}f}" if col in metric else f"{v:.1f}" if col.endswith("hfa") else f"{v:g}"

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(c, v) for c, v in zip(cols, row)) + " |"
              for row in df.itertuples(index=False)]
    return "\n".join(lines)


def current_ratings(elo: pd.DataFrame, cfg: config.EloConfig, state: dict) -> dict:
    """Current rating per franchise, rank, and change since the previous week."""
    home = elo[["season", "week", "date", "game_id", "home", "home_post"]]
    away = elo[["season", "week", "date", "game_id", "away", "away_post"]]
    long = pd.concat([home.set_axis(["season", "week", "date", "game_id", "team", "post"], axis=1),
                      away.set_axis(["season", "week", "date", "game_id", "team", "post"], axis=1)])
    long = long.sort_values(["date", "game_id"], kind="stable")
    cur = long.groupby("team").tail(1).set_index("team")
    as_of = long["date"].max()
    prev = long[long["date"] <= as_of - pd.Timedelta(days=7)].groupby("team").tail(1).set_index("team")["post"]

    tbl = pd.DataFrame({"rating": cur["post"], "last_game_date": cur["date"]})
    tbl["prev_rating"] = prev.reindex(tbl.index).fillna(tbl["rating"])
    tbl["rank"] = tbl["rating"].rank(ascending=False, method="first").astype(int)
    tbl["prev_rank"] = tbl["prev_rating"].rank(ascending=False, method="first").astype(int)
    tbl = tbl.sort_values("rank")
    teams = [{
        "franchise": t, "rating": round(r.rating, 1), "rank": int(r["rank"]),
        "last_game_date": r.last_game_date.strftime("%Y-%m-%d"),
        "rating_change": round(r.rating - r.prev_rating, 1),
        "rank_change": int(r.prev_rank - r["rank"]),
    } for t, r in tbl.iterrows()]
    return {
        "as_of": as_of.strftime("%Y-%m-%d"),
        "season": int(cur["season"].max()), "week": int(cur.loc[cur["date"] == as_of, "week"].max()),
        "change_is_since": "ratings 7 days before as_of (teams with no earlier game show 0)",
        "hfa_current": round(state["hfa"], 1),
        "config": cfg.to_dict(),
        "teams": teams,
    }


def hfa_by_decade(elo: pd.DataFrame, alt: pd.DataFrame | None = None) -> pd.DataFrame:
    """Actual home win rate by decade next to the HFA (Elo points) each model used."""
    def prep(df):
        g = df[(df["game_type"] == "REG") & ~df["neutral"].astype(bool)].copy()
        g["decade"] = (g["season"] // 10 * 10).astype(str) + "s"
        g["y"] = evaluate.outcome(g)
        return g
    g = prep(elo)
    t = g.groupby("decade").agg(games=("y", "size"), actual_home_win_rate=("y", "mean"),
                                chosen_model_hfa=("hfa_used", "mean"))
    if alt is not None:
        t["online_alt_hfa"] = prep(alt).groupby("decade")["hfa_used"].mean()
    return t.reset_index()


LEGACY_MISSING_NOTE = "Legacy spreadsheet not present; skipped legacy validation."


def load_legacy_if_present() -> pd.DataFrame | None:
    """The archived 2025 spreadsheet, or None when it is absent. The build never depends on it."""
    if not config.LEGACY_XLSX.exists():
        print(LEGACY_MISSING_NOTE)
        return None
    return data.load_legacy_games()


def run_validations(elo538: pd.DataFrame, nflverse: pd.DataFrame, legacy: pd.DataFrame | None) -> list:
    """538 vs the legacy spreadsheet (when present) and vs nflverse; prints one line per check."""
    last = config.FIVETHIRTYEIGHT_LAST_SEASON
    specs = []
    if legacy is not None:
        specs.append((f"FiveThirtyEight vs legacy spreadsheet, 1970-{last} regular season", legacy,
                      "legacy spreadsheet", (1970, last), ("REG",)))
    specs += [
        (f"FiveThirtyEight vs nflverse, {last + 1}-2022 regular season", nflverse, "nflverse",
         (last + 1, 2022), ("REG",)),
        (f"FiveThirtyEight vs nflverse, {last + 1}-2022 playoffs", nflverse, "nflverse",
         (last + 1, 2022), ("WC", "DIV", "CON", "SB")),
    ]
    validations = []
    for title, other, name, seasons, types in specs:
        v = data.validate_games(elo538, other, seasons, types)
        v["b_name"] = name
        validations.append((title, v))
        print(f"validation {title}: match rate {v['match_rate']:.4%} ({v['n_exact']}/{v['n_a']}), "
              f"home/away {v['home_agrees_rate']:.4%}, mismatches {v['mismatch_counts'] or 'none'}")
        if not v["mismatches"].empty:
            print(v["mismatches"].head(8).to_string(index=False))
    return validations


def legacy_reproduction(legacy: pd.DataFrame | None) -> tuple[dict | None, dict | None]:
    """Legacy parameters on the legacy spreadsheet alone, at lambda 0.15 and 0.20; (None, None) if absent."""
    if legacy is None:
        return None, None
    span = (int(legacy.season.min()), int(legacy.season.max()))
    repro = evaluate.score(run_elo(legacy, config.LEGACY_CONFIG)[0], span)
    repro_best = evaluate.score(run_elo(legacy, config.LEGACY_CONFIG.with_(lam=0.20))[0], span)
    print(f"legacy reproduction Brier {repro['brier']:.4f} (lambda 0.20: {repro_best['brier']:.4f})")
    return repro, repro_best


def write_report(path, validations, season_check, repro, repro_best, cfg, test, decade, tuning):
    L = []
    a = L.append
    a("# Elo v2 model report\n")
    a("Generated by `scripts/build.py`. Ratings warm up from 1970; tuning scored 1980-2009; test 2010-2025 (REG only).\n")
    a("## 1. Data validation\n")
    a("1970-1998 comes from FiveThirtyEight's game file (CC BY 4.0), 1999 on from nflverse. Games are matched on "
      "(season, game type, franchise pair, n-th meeting that season) and scores compared. Match rate is exact "
      "agreement on both scores divided by the games in the FiveThirtyEight file for that window. Home/away is "
      "compared on matched games that are non-neutral in both sources.\n")
    if not any(v["b_name"] == "legacy spreadsheet" for _, v in validations):
        a(LEGACY_MISSING_NOTE + "\n")
    for title, v in validations:
        lo, hi = v["seasons"]
        a(f"### {title}\n")
        a(f"Seasons {lo}-{hi}, game types {', '.join(v['game_types'])}.\n")
        a(f"- Games: {v['n_a']:,} in FiveThirtyEight, {v['n_b']:,} in {v['b_name']}")
        a(f"- Matched on key: {v['n_key_matched']:,}; exact score match: {v['n_exact']:,}")
        a(f"- **Match rate: {v['match_rate']:.4%}**; home/away agreement on {v['n_non_neutral_matched']:,} "
          f"non-neutral matched games: {v['home_agrees_rate']:.4%}; neutral-flag agreement: "
          f"{v['neutral_agrees_rate']:.4%}; date agreement: {v['date_agrees_rate']:.4%}")
        if v["mismatch_counts"]:
            a(f"- Score/key mismatches by kind: {v['mismatch_counts']}\n")
            a(md_table(v["mismatches"].head(15).astype(str)) + "\n")
        else:
            a("- Score/key mismatches: none.\n")
        fm = v["flag_mismatches"]
        if len(fm):
            a(f"Home/neutral-flag disagreements on score-matched games ({len(fm)}; these are relocated or "
              "venue-swapped games where the two sources label the host differently):\n")
            a(md_table(fm.head(15).astype(str)) + "\n")
    a("The nflverse overlap exercises the 1999+ code mapping and the FiveThirtyEight loader. The pre-1999 mapping "
      "rules (Oilers, Colts, Cardinals, Rams, Raiders, the Browns gap) are checked by the legacy comparison "
      "(when present), which uses an independent set of codes, and by franchises and games per franchise by season:\n")
    sc = season_check[season_check["season"].isin([1970, 1975, 1976, 1982, 1987, 1994, 1995, 1996, 1998, 1999, 2002])]
    a(md_table(sc) + "\n")

    a("## 2. Legacy reproduction (sanity check)\n")
    if repro is None:
        a(LEGACY_MISSING_NOTE + "\n")
    else:
        a("Legacy parameters (K 20, HFA 55, MOV with the 2.0 cap and absolute rating difference, no playoffs, start "
          "1500 instead of 1000) on legacy xlsx data only, all REG games 1970-2025:\n")
        a(f"- lambda 0.15 (legacy default): Brier **{repro['brier']:.4f}**, log loss {repro['logloss']:.4f}, "
          f"accuracy {repro['accuracy']:.4f}, n={repro['n']:,}")
        a(f"- lambda 0.20 (the legacy sensitivity-grid best, the source of the quoted 0.2201 / 0.6326): "
          f"Brier **{repro_best['brier']:.4f}**, log loss {repro_best['logloss']:.4f}\n")


    a("## 3. Tuning\n")
    if tuning is None:
        a("No tuning results found. Run `scripts/tune.py`.\n")
    else:
        res, meta = tuning
        a(f"{meta['n_configs']} configs scored on REG games {config.TUNE_SEASONS[0]}-{config.TUNE_SEASONS[1]}; "
          f"runtime {meta['runtime_seconds']:.0f}s. Top 10 by Brier:\n")
        show = res.head(10)[["k", "lam", "hfa_mode", "hfa", "k_hfa", "hfa_mov", "include_playoffs",
                             "expansion_start", "mov_cap", "brier", "logloss", "accuracy"]].copy()
        show["hfa"] = show["hfa"].where(show["hfa_mode"] == "fixed", "learned").astype(str)
        show["k_hfa"] = show["k_hfa"].where(show["hfa_mode"] == "online", "-").astype(str)
        show["mov_cap"] = show["mov_cap"].fillna(-1).map(lambda x: "none" if x < 0 else f"{x:g}")
        a(md_table(show) + "\n")
        a("The grid adds two axes beyond the required ones: MOV multiplier cap (none or 2.0) and scaling the "
          "online HFA update by the MOV multiplier. The top configs differ by about 0.0001 Brier, which is "
          "within noise, so the single winner is a weak signal; the choice follows the protocol (best tune-window "
          "Brier) rather than judgment.\n")
        ob = res[res["hfa_mode"] == "online"].iloc[0]
        a(f"Best online-HFA config on the tune window: K {ob['k']:g}, lambda {ob['lam']:g}, k_hfa {ob['k_hfa']:g}, "
          f"hfa_mov {bool(ob['hfa_mov'])}, playoffs {bool(ob['include_playoffs'])}, Brier {ob['brier']:.4f}.\n")
        best = evaluate.config_from_row(res.iloc[0])
        if best != cfg:
            a("`DEFAULT_CONFIG` is the best online-HFA config rather than the overall winner. The two tie within "
              "noise on the tune window, and a fixed HFA cannot track the decline in home-field advantage "
              "(see section 5).\n")
    a("Chosen config (`DEFAULT_CONFIG`):\n")
    a("```\n" + json.dumps(cfg.to_dict(), indent=2) + "\n```\n")

    a(f"## 4. Held-out test {config.TEST_SEASONS[0]}-{config.TEST_SEASONS[1]}\n")
    a("All REG games in the window:\n")
    a(md_table(test["all"]) + "\n")
    ms = test["market_seasons"]
    a(f"Same table restricted to REG games that have moneylines ({ms[0]}-{ms[1]}); every row scores the identical games:\n")
    a(md_table(test["market"]) + "\n")
    p = test["paired"]
    a(f"Paired Brier difference, {p['model']} minus market, over {p['n']:,} games: "
      f"{p['brier_diff_vs_market']:+.4f} (standard error {p['se']:.4f}). Negative favors Elo.\n")

    a("## 5. Home-field advantage by decade (REG, non-neutral)\n")
    a("Mean HFA in Elo points used per game. `online_alt_hfa` is the best online-HFA config from the tuning "
      "table (currently the same config as the default).\n")
    a(md_table(decade) + "\n")
    path.write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use the cached nflverse file if present")
    args = ap.parse_args()
    cfg = config.DEFAULT_CONFIG

    nflverse = data.nflverse_to_games(data.fetch_nflverse_schedules(refresh=not args.offline))
    elo538 = data.load_538_games(seasons=(1970, 2022))  # 1999-2022 rows are for validation only
    games = data.build_games(elo538, nflverse)
    config.DATA_DIR.mkdir(exist_ok=True)
    games.to_csv(config.GAMES_CSV, index=False)
    print(f"data/games.csv: {len(games):,} games, seasons {games.season.min()}-{games.season.max()}")

    legacy = load_legacy_if_present()
    validations = run_validations(elo538, nflverse, legacy)
    season_check = data.season_franchise_check(games)
    repro, repro_best = legacy_reproduction(legacy)

    elo, state = run_elo(games, cfg)
    config.OUT_DIR.mkdir(exist_ok=True)
    out = elo.drop(columns=["source"]).copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    for c in ["home_pre", "away_pre", "home_post", "away_post"]:
        out[c] = out[c].round(2)
    out["p_home"] = out["p_home"].round(4)
    out["hfa_used"] = out["hfa_used"].round(2)
    out["mov"] = out["mov"].round(3)
    out.to_csv(config.OUT_DIR / "elo_games.csv", index=False)

    ratings = current_ratings(elo, cfg, state)
    (config.OUT_DIR / "ratings_current.json").write_text(json.dumps(ratings, indent=2))

    models = {"Elo v2 (default)": cfg, "legacy (scaled)": config.LEGACY_CONFIG}
    tuning, alt_elo = None, None
    tp, mp = config.OUT_DIR / "tuning_results.csv", config.OUT_DIR / "tuning_best.json"
    if tp.exists() and mp.exists():
        tuning = (pd.read_csv(tp), json.loads(mp.read_text()))
        online = evaluate.best_online(tuning[0])
        if online is not None:
            models["best online-HFA (tune-selected)"] = online
            alt_elo = run_elo(games, online)[0]
    test = evaluate.compare_on_test(games, models)
    write_report(config.OUT_DIR / "model_report.md", validations, season_check, repro, repro_best,
                 cfg, test, hfa_by_decade(elo, alt_elo), tuning)
    print("test (all REG 2010-2025):")
    print(test["all"].to_string(index=False))
    print("test (moneyline games):")
    print(test["market"].to_string(index=False))
    print(f"as of {ratings['as_of']} (season {ratings['season']} week {ratings['week']}); "
          f"wrote outputs/elo_games.csv, ratings_current.json, model_report.md")
    import dashboard  # noqa: E402  (scripts/ is on sys.path when run as a script)
    dashboard.render()
    import export_site  # noqa: E402  (writes site/data/*.json for the website)
    export_site.export()


if __name__ == "__main__":
    main()
