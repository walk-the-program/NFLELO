"""Render the "Team Summary" dashboard PNG from the Elo v2 outputs.

    python scripts/dashboard.py                # writes outputs/dashboards/team_summary.png
    python scripts/dashboard.py --show-tables  # also prints the four tables behind the panels

Reads outputs/elo_games.csv and outputs/ratings_current.json (both written by
scripts/build.py). Regular-season games only for records and rating history.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflelo import config  # noqa: E402
from nflelo.colors import team_color  # noqa: E402

OUT_PATH = config.OUT_DIR / "dashboards" / "team_summary.png"
AVG = 1500.0  # league-average rating
TOP_N = 10

BG = "#f3f4f6"       # figure background
PANEL = "#ffffff"
INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"
UP = "#15803d"
DOWN = "#b91c1c"
FLAT = "#9ca3af"
LEADER = {"arrowstyle": "-", "color": "#9ca3af", "lw": 0.6, "shrinkA": 0, "shrinkB": 3}


# --------------------------------------------------------------------------- tables

def load_inputs() -> tuple[pd.DataFrame, dict]:
    """Per-game Elo output and the current-ratings JSON."""
    games = pd.read_csv(config.OUT_DIR / "elo_games.csv")
    ratings = json.loads((config.OUT_DIR / "ratings_current.json").read_text())
    return games, ratings


def team_games(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team per REG game: team, season, date, post rating, score (1 / 0.5 / 0)."""
    reg = games[games["game_type"] == "REG"]
    diff = (reg["home_score"] - reg["away_score"]).to_numpy()
    home_pts = np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, 0.5))
    cols = ["season", "date", "game_id"]
    home = reg[cols].assign(team=reg["home"], post=reg["home_post"], pts=home_pts)
    away = reg[cols].assign(team=reg["away"], post=reg["away_post"], pts=1.0 - home_pts)
    return pd.concat([home, away], ignore_index=True).sort_values(["date", "game_id"], kind="stable")


def build_tables(games: pd.DataFrame, ratings: dict) -> dict[str, pd.DataFrame]:
    """The four DataFrames behind the panels: ladder, scatter, swings, avg_elo."""
    ladder = pd.DataFrame(ratings["teams"]).sort_values("rank").reset_index(drop=True)

    long = team_games(games)
    by_team = long.groupby("team")
    allt = pd.DataFrame({
        "games": by_team.size(),
        "win_pct": by_team["pts"].mean(),
        "avg_elo": by_team["post"].mean(),
    })
    scatter = allt.reset_index().rename(columns={"team": "franchise"})

    # Peak and low post-game rating, with the season each was first reached.
    hi = long.loc[by_team["post"].idxmax()].set_index("team")
    lo = long.loc[by_team["post"].idxmin()].set_index("team")
    sw = pd.DataFrame({
        "low": lo["post"], "low_season": lo["season"],
        "peak": hi["post"], "peak_season": hi["season"],
    })
    sw["range"] = sw["peak"] - sw["low"]
    swings = sw.sort_values("range", ascending=False).head(TOP_N).rename_axis("franchise").reset_index()

    avg_elo = (allt[["avg_elo", "games"]].sort_values("avg_elo", ascending=False).head(TOP_N)
               .rename_axis("franchise").reset_index())
    return {"ladder": ladder, "scatter": scatter, "swings": swings, "avg_elo": avg_elo}


# --------------------------------------------------------------------------- helpers

def style_axes(ax, title: str, subtitle: str = "") -> None:
    """Shared look: white card, light grid, left-aligned title and a muted subtitle."""
    ax.set_facecolor(PANEL)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_axisbelow(True)
    ax.annotate(title, (0, 1), xytext=(0, 22), textcoords="offset points", xycoords="axes fraction",
                fontsize=13, fontweight="bold", color=INK, va="bottom")
    if subtitle:
        ax.annotate(subtitle, (0, 1), xytext=(0, 8), textcoords="offset points", xycoords="axes fraction",
                    fontsize=9, color=MUTED, va="bottom")


def delta_text(value: float, fmt: str) -> tuple[str, str]:
    """Arrow text and color for a change value (green up, red down, grey flat)."""
    if value > 0:
        return f"▲ {fmt.format(value)}", UP
    if value < 0:
        return f"▼ {fmt.format(abs(value))}", DOWN
    return "–", FLAT


def place_labels(ax, xs, ys, labels, fontsize: float = 8.5) -> None:
    """Label scatter points with the offset that overlaps the fewest other labels, markers, or edges.

    Offsets are tried nearest first; a label that has to move far gets a thin leader line.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    near = [(5, 4), (5, -11), (-5, 4), (-5, -11), (0, 8), (0, -13), (12, -3), (-12, -3)]
    ring = [(r * np.cos(a), r * np.sin(a)) for r in (18, 28, 40, 54)
            for a in np.linspace(0, 2 * np.pi, 12, endpoint=False) + 0.3]
    offsets = near + [(round(x), round(y)) for x, y in ring]
    frame = ax.get_window_extent(renderer)
    placed: list = []   # window-space boxes of accepted labels
    dots = [(x - 7, y - 7, x + 7, y + 7) for x, y in pts]   # marker footprints (pixels)
    order = np.argsort(pts[:, 1])[::-1]   # top-down so crowded areas resolve deterministically
    for i in order:
        best, best_hits, best_off = None, 10**6, offsets[0]
        for dx, dy in offsets:
            tr = offset_copy(ax.transData, fig=fig, x=dx, y=dy, units="points")
            t = ax.text(xs[i], ys[i], labels[i], transform=tr, fontsize=fontsize, fontweight="bold", color=INK,
                        ha="left" if dx >= 0 else "right", va="bottom", zorder=6)
            bb = t.get_window_extent(renderer).expanded(1.05, 1.15)
            hits = sum(bb.overlaps(p) for p in placed)
            hits += sum(1 for j, d in enumerate(dots) if j != i and
                        bb.x0 < d[2] and bb.x1 > d[0] and bb.y0 < d[3] and bb.y1 > d[1])
            hits += 3 * (bb.x0 < frame.x0 or bb.x1 > frame.x1 or bb.y0 < frame.y0 or bb.y1 > frame.y1)
            if hits < best_hits:
                if best is not None:
                    best.remove()
                best, best_hits, best_off = t, hits, (dx, dy)
            else:
                t.remove()
            if best_hits == 0:
                break
        placed.append(best.get_window_extent(renderer).expanded(1.05, 1.15))
        if max(abs(best_off[0]), abs(best_off[1])) > 11:   # leader line from the dot to the label
            ax.annotate("", (xs[i], ys[i]), xytext=best_off, textcoords="offset points",
                        arrowprops=LEADER, zorder=5)


# --------------------------------------------------------------------------- panels

def panel_ladder(ax, ladder: pd.DataFrame) -> None:
    """Power Ladder: all teams, rank 1 on top, bars diverging from the 1500 average."""
    n = len(ladder)
    y = np.arange(n)[::-1]
    vals = ladder["rating"].to_numpy()
    colors = [team_color(t) for t in ladder["franchise"]]
    ax.barh(y, vals - AVG, left=AVG, color=colors, height=0.72, zorder=3)
    lo, hi = vals.min(), vals.max()
    span = max(hi - AVG, AVG - lo)
    ax.set_xlim(AVG - span * 1.28, AVG + span * 1.28)
    ax.set_ylim(-0.7, n + 0.2)
    ax.axvline(AVG, color=INK, lw=1, zorder=4)
    ax.text(AVG + 3, n - 0.3, "league avg 1500", ha="left", va="bottom", fontsize=8, color=MUTED)
    ax.set_yticks(y, ladder["franchise"], fontsize=10, fontweight="bold", color=INK)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_xlabel("Elo rating (1500 = league average)", fontsize=9, color=MUTED)
    style_axes(ax, "Power Ladder", "Current rating, with change vs. 7 days earlier")
    for yi, (_, r) in zip(y, ladder.iterrows()):
        above = r["rating"] >= AVG
        ax.text(r["rating"] + (3 if above else -3), yi, f"{r['rating']:.0f}", ha="left" if above else "right",
                va="center", fontsize=9, color=INK, zorder=5)
        ax.text(-0.14, yi, str(int(r["rank"])), transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=8.5, color=MUTED)
        for x, txt, col in ((1.045, *delta_text(r["rating_change"], "{:.1f}")),
                            (1.19, *delta_text(r["rank_change"], "{:d}"))):
            ax.text(x, yi, txt, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8.5,
                    color=col, fontweight="bold", clip_on=False)
    for x, head in ((1.045, "Elo Δ"), (1.19, "Rank Δ")):
        ax.text(x, n - 0.3, head, transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=8,
                color=MUTED, clip_on=False)


def panel_scatter(ax, scatter: pd.DataFrame) -> None:
    """All-time REG win % vs. all-time average Elo, one dot per franchise."""
    x, yv = scatter["win_pct"].to_numpy(), scatter["avg_elo"].to_numpy()
    colors = [team_color(t) for t in scatter["franchise"]]
    ax.scatter(x, yv, s=48, c=colors, edgecolor="white", linewidth=1, zorder=4)
    slope, icept = np.polyfit(x, yv, 1)
    r = np.corrcoef(x, yv)[0, 1]
    xr = np.array([x.min() - 0.01, x.max() + 0.01])
    ax.plot(xr, slope * xr + icept, color=MUTED, lw=1, ls="--", zorder=2)
    ax.axhline(AVG, color=GRID, lw=1, zorder=1)
    ax.axvline(0.5, color=GRID, lw=1, zorder=1)
    ax.set_xlim(x.min() - 0.012, x.max() + 0.012)
    pad = (yv.max() - yv.min()) * 0.08
    ax.set_ylim(yv.min() - pad, yv.max() + pad)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_xlabel("All-time regular-season win % (ties = half)", fontsize=9, color=MUTED)
    ax.set_ylabel("All-time average Elo", fontsize=9, color=MUTED)
    style_axes(ax, "Elo vs. Winning, since 1970", "One dot per franchise; both axes cover every regular-season game")
    ax.text(0.99, 0.04, f"r = {r:.2f}", transform=ax.transAxes, ha="right", fontsize=9, color=MUTED)
    place_labels(ax, x, yv, list(scatter["franchise"]))


def panel_swings(ax, swings: pd.DataFrame) -> None:
    """Range bars from each franchise's lowest to highest post-game rating, biggest on top."""
    n = len(swings)
    y = np.arange(n)[::-1]
    colors = [team_color(t) for t in swings["franchise"]]
    ax.barh(y, swings["range"], left=swings["low"], color=colors, height=0.62, zorder=3)
    lo, hi = swings["low"].min(), swings["peak"].max()
    span = hi - lo
    ax.set_xlim(lo - span * 0.2, hi + span * 0.36)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(y, swings["franchise"], fontsize=10, fontweight="bold", color=INK)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.axvline(AVG, color=MUTED, lw=0.8, ls=":", zorder=2)
    ax.set_xlabel("Post-game Elo rating (dotted line = 1500)", fontsize=9, color=MUTED)
    style_axes(ax, "Biggest Rises and Falls (Peak to Low)",
               "Lowest to highest post-game rating since 1970, with the season of each extreme")
    for yi, (_, r) in zip(y, swings.iterrows()):
        ax.text(r["low"] - 4, yi, f"{r['low_season']} · {r['low']:.0f}", ha="right", va="center",
                fontsize=8.5, color=INK)
        ax.text(r["peak"] + 4, yi, f"{r['peak']:.0f} · {r['peak_season']}", ha="left", va="center",
                fontsize=8.5, color=INK)
        ax.text(0.995, yi, f"{r['range']:.0f}", transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=8.5, color=MUTED, fontweight="bold")
    ax.text(0.995, n - 0.35, "range", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=8, color=MUTED)


def panel_avg_elo(ax, avg_elo: pd.DataFrame) -> None:
    """Top franchises by all-time average Elo, bars drawn from 1500."""
    n = len(avg_elo)
    y = np.arange(n)[::-1]
    colors = [team_color(t) for t in avg_elo["franchise"]]
    ax.barh(y, avg_elo["avg_elo"] - AVG, left=AVG, color=colors, height=0.62, zorder=3)
    hi = avg_elo["avg_elo"].max()
    ax.set_xlim(AVG, AVG + (hi - AVG) * 1.15)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(y, avg_elo["franchise"], fontsize=10, fontweight="bold", color=INK)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_xlabel("Average post-game Elo, every regular-season game since 1970 (bars start at 1500)",
                  fontsize=9, color=MUTED)
    style_axes(ax, "Highest Average Elo, All-Time", "Top 10 franchises")
    for yi, v in zip(y, avg_elo["avg_elo"]):
        ax.text(v + 0.6, yi, f"{v:.1f}", ha="left", va="center", fontsize=9, color=INK)


# --------------------------------------------------------------------------- figure

def footer_text(ratings: dict) -> str:
    cfg = ratings["config"]
    hfa = "learned home-field advantage" if cfg["hfa_mode"] == "online" else "fixed home-field advantage"
    return (f"Data: nflverse (CC BY 4.0); 1970–1998 results: FiveThirtyEight (CC BY 4.0) · Model: Elo v2 "
            f"(K {cfg['k']:g}, λ {cfg['lam']:.2f}, {hfa}, currently {ratings['hfa_current']:g} pts) "
            f"· github.com/walk-the-program/NFLELO")


def render(out_path: Path = OUT_PATH, tables: dict | None = None, ratings: dict | None = None) -> Path:
    """Draw the dashboard and save it as a PNG."""
    if tables is None or ratings is None:
        games, ratings = load_inputs()
        tables = build_tables(games, ratings)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
    fig = plt.figure(figsize=(16, 11), facecolor=BG)
    fig.text(0.075, 0.965, "NFL Elo — Team Summary", fontsize=24, fontweight="bold", color=INK, va="center")
    fig.text(0.075, 0.928, f"Through {ratings['season']} Week {ratings['week']} · as of {ratings['as_of']}",
             fontsize=12, color=MUTED, va="center")

    left = GridSpec(1, 1, figure=fig, left=0.075, right=0.355, top=0.835, bottom=0.07)
    right = GridSpec(3, 1, figure=fig, left=0.54, right=0.965, top=0.835, bottom=0.075, hspace=0.68,
                     height_ratios=[1.8, 1, 0.85])
    panel_ladder(fig.add_subplot(left[0]), tables["ladder"])
    panel_scatter(fig.add_subplot(right[0]), tables["scatter"])
    panel_swings(fig.add_subplot(right[1]), tables["swings"])
    panel_avg_elo(fig.add_subplot(right[2]), tables["avg_elo"])
    fig.text(0.075, 0.022, footer_text(ratings), fontsize=8.5, color=MUTED, va="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--show-tables", action="store_true", help="print the four tables behind the panels")
    args = ap.parse_args(argv)
    games, ratings = load_inputs()
    tables = build_tables(games, ratings)
    if args.show_tables:
        with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_rows", 50):
            for name, df in tables.items():
                print(f"\n== {name} ({len(df)} rows) ==")
                print(df.round(3).to_string(index=False))
    path = render(tables=tables, ratings=ratings)
    print(f"wrote {path.relative_to(config.ROOT)} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
