"""Print the experiment leaderboard for one window, sorted by Brier.

    python scripts/ml_leaderboard.py --window dev
    python scripts/ml_leaderboard.py --window reproduction
    python scripts/ml_leaderboard.py --window holdout --all   # every run, not just the newest per model

Rows are comparable only when their `games` hash matches (same scored games).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from nflelo.ml import registry  # noqa: E402
from nflelo.ml.eval import windows  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", choices=sorted(windows.LABELS), default="dev")
    ap.add_argument("--all", action="store_true", help="show every run instead of the newest per (model, game set)")
    a = ap.parse_args()
    t = registry.leaderboard(a.window, latest_only=not a.all)
    lo, hi = windows.LABELS[a.window]
    print(f"Leaderboard: {a.window} ({lo}-{hi}, REG), sorted by Brier")
    if t.empty:
        print("no runs logged for this window yet")
        return 0
    t = t.drop(columns=["seasons"]) if t["seasons"].nunique() == 1 else t
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
