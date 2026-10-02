"""Grid-tune the Elo config on the 1980-2009 window, then report once on 2010-2025.

Run scripts/build.py first so data/games.csv exists. Writes
outputs/tuning_results.csv (every config, best first) and outputs/tuning_best.json.
Copy the winning config into DEFAULT_CONFIG in nflelo/config.py.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflelo import config, data, evaluate  # noqa: E402


def main() -> None:
    games = data.load_games()
    grid = evaluate.default_grid()
    print(f"Tuning {len(grid)} configs on seasons {config.TUNE_SEASONS} ...")
    t0 = time.time()
    res = evaluate.tune(games, grid)
    runtime = time.time() - t0
    config.OUT_DIR.mkdir(exist_ok=True)
    res.to_csv(config.OUT_DIR / "tuning_results.csv", index=False)

    best = evaluate.config_from_row(res.iloc[0])
    print(f"Runtime {runtime:.1f}s. Best on tune window:")
    print(res.head(10)[["k", "lam", "hfa_mode", "hfa", "k_hfa", "hfa_mov", "include_playoffs",
                        "expansion_start", "mov_cap", "brier", "logloss", "accuracy"]].to_string())

    models = {"tuned": best, "legacy (scaled)": config.LEGACY_CONFIG}
    online = evaluate.best_online(res)
    if online is not None:
        models["best online-HFA (tune-selected)"] = online
    test = evaluate.compare_on_test(games, models)
    print("\nTest, all REG games", config.TEST_SEASONS)
    print(test["all"].to_string(index=False))
    print("\nTest, REG games with moneylines (same games)")
    print(test["market"].to_string(index=False))
    print(test["paired"])
    with open(config.OUT_DIR / "tuning_best.json", "w") as f:
        json.dump({"config": best.to_dict(), "tune_window": config.TUNE_SEASONS,
                   "tune_metrics": res.iloc[0][["n", "brier", "logloss", "accuracy"]].to_dict(),
                   "n_configs": len(grid), "runtime_seconds": runtime}, f, indent=2, default=float)


if __name__ == "__main__":
    main()
