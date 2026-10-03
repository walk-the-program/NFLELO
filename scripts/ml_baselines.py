"""Score the M2 baselines and log them in the experiment registry.

    python scripts/ml_baselines.py            # reproduction check + DEV baselines
    python scripts/ml_baselines.py --no-log   # print only, write no run files

(a) Reproduction, REG 2010-2025 (labelled "reproduction", allow_holdout=True):
    the harness must match `nflelo.evaluate.compare_on_test` to within 1e-4 on
    identical games: Elo v2 on all test games, and Elo v2 and the market on the
    games with moneylines. Exits 1 if it does not.
(b) DEV, REG 2006-2019: the four baselines on identical games (those with
    moneylines), plus a paired bootstrap CI for Elo minus market Brier.

Reads data/games.csv only (run scripts/build.py first); no network access.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflelo import config, data, evaluate  # noqa: E402
from nflelo.ml import registry  # noqa: E402
from nflelo.ml.eval import baselines, bootstrap, calibration, metrics, windows  # noqa: E402

TOL = 1e-4

NOTES = {
    "p_coin_flip": "50/50 for every game.",
    "p_home_rate": "Constant home-win rate (ties 0.5) from REG games of the 10 seasons before each season.",
    "p_elo_v2": "Elo v2 DEFAULT_CONFIG pre-game p_home. Its parameters were tuned on 1980-2009, so "
                "2006-2009 are in-sample for Elo (slightly optimistic on DEV).",
    baselines.MARKET_COL: "Vig-removed moneyline probability. Benchmark only, never a feature (D3).",
}
RUN_NAMES = {"p_coin_flip": "coin_flip", "p_home_rate": "home_rate", "p_elo_v2": "elo_v2",
             baselines.MARKET_COL: "market"}


def score(df, col) -> dict:
    y, p = df["y"].to_numpy(), df[col].to_numpy()
    return {**metrics.win_metrics(y, p), "ece": calibration.ece(y, p)}


def params_for(col: str) -> dict:
    if col == "p_elo_v2":
        return {"elo_config": config.DEFAULT_CONFIG.to_dict()}
    if col == "p_home_rate":
        return {"lookback_seasons": baselines.HOME_RATE_LOOKBACK}
    return {}


def reproduction(games, log: bool) -> bool:
    seasons = windows.REPRODUCTION
    print(f"== (a) Reproduction, REG {seasons[0]}-{seasons[1]} (allow_holdout=True) ==")
    ref = evaluate.compare_on_test(games, {"elo_v2": config.DEFAULT_CONFIG}, seasons)
    ref_all = ref["all"].set_index("model")
    ref_mkt = ref["market"].set_index("model")

    full = baselines.baseline_frame(games, seasons, allow_holdout=True, require_market=False)
    mkt = full[full[baselines.MARKET_COL].notna()]
    checks = [
        ("elo_v2", "all test games", full, "p_elo_v2", ref_all.loc["elo_v2"]),
        ("elo_v2_market_games", "games with moneylines", mkt, "p_elo_v2", ref_mkt.loc["elo_v2"]),
        ("market", "games with moneylines", mkt, baselines.MARKET_COL, ref_mkt.loc["market (vig removed)"]),
    ]
    ok = True
    print(f"{'run':<22}{'games':<24}{'n':>6}{'harness':>10}{'reference':>11}{'diff':>11}  status")
    for name, desc, df, col, r in checks:
        m = score(df, col)
        diff = m["brier"] - r["brier"]
        good = abs(diff) <= TOL and m["n"] == int(r["n"])
        ok &= good
        print(f"{name:<22}{desc:<24}{m['n']:>6}{m['brier']:>10.4f}{r['brier']:>11.4f}{diff:>11.1e}  "
              f"{'OK' if good else 'MISMATCH'}")
        if log:
            registry.log_run(name, label="reproduction", seasons=seasons, game_ids=df["game_id"], metrics=m,
                             params=params_for(col), notes=f"{NOTES[col]} Scored on {desc}. "
                             f"Reference: evaluate.compare_on_test Brier {r['brier']:.6f}, n {int(r['n'])}.")
    print("reproduction:", "PASS" if ok else "FAIL")
    return ok


def dev(games, log: bool) -> None:
    seasons = windows.DEV
    print(f"\n== (b) DEV, REG {seasons[0]}-{seasons[1]}: identical games (those with moneylines) ==")
    df = baselines.baseline_frame(games, seasons, require_market=True)
    n_all = len(windows.select(games, seasons))
    print(f"{len(df)} of {n_all} REG games have moneylines; all four baselines are scored on those {len(df)}.")
    cols = list(baselines.BASELINES) + [baselines.MARKET_COL]
    results = {c: score(df, c) for c in cols}
    ci = bootstrap.paired_bootstrap(df["y"], df["p_elo_v2"], df[baselines.MARKET_COL], metric="brier")
    print(f"{'model':<12}{'n':>6}{'brier':>9}{'logloss':>9}{'acc':>8}{'ece':>8}")
    for c in sorted(cols, key=lambda c: results[c]["brier"]):
        m = results[c]
        print(f"{RUN_NAMES[c]:<12}{m['n']:>6}{m['brier']:>9.4f}{m['logloss']:>9.4f}{m['accuracy']:>8.3f}{m['ece']:>8.4f}")
    print(f"\nElo v2 minus market, Brier: {ci['diff']:+.4f}, 95% paired bootstrap CI "
          f"[{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}] ({ci['reps']} reps, seed {ci['seed']}); "
          f"{'excludes' if ci['excludes_zero'] else 'includes'} zero.")
    print("Note: Elo v2 was tuned on 1980-2009, so 2006-2009 are in-sample for Elo.")
    if log:
        for c in cols:
            extra = {"comparisons": {"vs_market_brier": ci}} if c == "p_elo_v2" else None
            registry.log_run(RUN_NAMES[c], label="dev", seasons=seasons, game_ids=df["game_id"],
                             metrics=results[c], params=params_for(c), notes=NOTES[c], extra=extra)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-log", action="store_true", help="print results without writing run files")
    a = ap.parse_args()
    games = data.load_games()
    ok = reproduction(games, log=not a.no_log)
    if not ok:
        print("Reproduction failed; DEV baselines not scored.", file=sys.stderr)
        return 1
    dev(games, log=not a.no_log)
    if not a.no_log:
        print(f"\nRuns written to {registry.RUNS_DIR.relative_to(config.ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
