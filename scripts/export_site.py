"""Export the JSON files the static site reads (site/data/*.json).

    python scripts/export_site.py

Reads outputs/elo_games.csv, outputs/ratings_current.json, outputs/model_report.md
and the cached nflverse schedule data/raw/schedules.csv (upcoming games and lines).
Writes meta, ladder, upcoming, history, luck, tapestry, records, scorecard and headlines JSON.
Output is deterministic (no timestamps; same inputs give identical bytes).
`scripts/build.py` calls export() at the end of every run.

Conventions
- Ratings: 1500 is average. Win probability p = 1/(1+10^(-(home+hfa-away)/400)).
- Elo spread = (home + hfa - away) / 25, positive when the HOME team is favored.
  nflverse `spread_line` uses the same sign (verified against results: games with
  spread_line > 7 average a home margin of about +11), so the two are comparable
  and the difference is elo_spread - vegas_spread.
- History downsampling: none needed. One point per team per regular-season game
  week (post-game rating), 1970 to now, about 30k points (~0.4 MB). Playoff games
  do not update ratings in Elo v2, so they add no points.
- The season in progress is partial: it is excluded from season-end records and
  the scorecard, and marked partial in the tapestry.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflelo import config, meta as team_meta  # noqa: E402
from nflelo.colors import team_color  # noqa: E402
from nflelo.evaluate import market_prob, metrics  # noqa: E402

SITE_DATA = config.ROOT / "site" / "data"
SCHEDULES_CSV = config.RAW_DIR / "schedules.csv"
REPORT_MD = config.OUT_DIR / "model_report.md"
CREDIT = "Data: nflverse (CC-BY 4.0)"
REPO_URL = "https://github.com/walk-the-program/NFLELO"
SPARK_GAMES = 17          # ladder sparkline length (last N regular-season games)
TOP_N = 10
DISAGREEMENTS = 3
MIN_SEASON_GAMES = 200    # fewest market games for a season's market Brier to be reported
LUCK_SEASONS_BACK = (1, 0)  # previous season (full) and the current one (to date)


# --------------------------------------------------------------------------- loading

def load_elo() -> pd.DataFrame:
    elo = pd.read_csv(config.OUT_DIR / "elo_games.csv", parse_dates=["date"])
    elo["neutral"] = elo["neutral"].astype(bool)
    elo["updated"] = elo["updated"].astype(bool)
    return elo


def load_ratings() -> dict:
    return json.loads((config.OUT_DIR / "ratings_current.json").read_text())


def load_schedule() -> pd.DataFrame:
    return pd.read_csv(SCHEDULES_CSV)


def write_json(name: str, obj) -> int:
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, separators=(",", ":"), allow_nan=False)
    (SITE_DATA / name).write_text(text + "\n")
    return len(text) + 1


def clean(x):
    """NaN -> None and numpy scalars -> python, for JSON."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, np.generic):
        return x.item()
    return x


# --------------------------------------------------------------------------- derived tables

def team_game_rows(elo: pd.DataFrame) -> pd.DataFrame:
    """One row per team per REG game: pre/post rating, the team's win probability, result."""
    reg = elo[elo["game_type"] == "REG"]
    diff = (reg["home_score"] - reg["away_score"]).to_numpy()
    home_pts = np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, 0.5))
    base = ["season", "week", "date", "game_id"]
    home = reg[base].assign(team=reg["home"], pre=reg["home_pre"], post=reg["home_post"],
                            p=reg["p_home"], pts=home_pts)
    away = reg[base].assign(team=reg["away"], pre=reg["away_pre"], post=reg["away_post"],
                            p=1.0 - reg["p_home"], pts=1.0 - home_pts)
    long = pd.concat([home, away], ignore_index=True)
    return long.sort_values(["date", "game_id", "team"], kind="stable").reset_index(drop=True)


def record(pts: pd.Series) -> tuple[int, int, int]:
    """(wins, losses, ties) from a series of 1 / 0.5 / 0 results."""
    return int((pts == 1.0).sum()), int((pts == 0.0).sum()), int((pts == 0.5).sum())


def season_end(long: pd.DataFrame) -> pd.DataFrame:
    """Per team-season: end-of-REG-season rating and record."""
    rows = []
    for (team, season), g in long.groupby(["team", "season"], sort=True):
        w, l, t = record(g["pts"])
        rows.append({"team": team, "season": int(season), "rating": float(g["post"].iloc[-1]),
                     "w": w, "l": l, "t": t})
    return pd.DataFrame(rows)


def season_is_partial(schedule: pd.DataFrame, season: int) -> bool:
    s = schedule[schedule["season"] == season]
    return bool(s["home_score"].isna().any())


def legacy_brier() -> float | None:
    """Legacy-model Brier on the test window, parsed from the model report (None if absent)."""
    if not REPORT_MD.exists():
        return None
    m = re.search(r"^\| legacy \(scaled\) \| \d+ \| ([0-9.]+) \|", REPORT_MD.read_text(), re.M)
    return float(m.group(1)) if m else None


# --------------------------------------------------------------------------- sections

def build_scorecard(elo: pd.DataFrame, hfa_now: float, partial_season: int | None) -> tuple[dict, dict]:
    """Per-season Brier (Elo vs market) and learned HFA, plus the test-window headline numbers."""
    reg = elo[elo["game_type"] == "REG"].copy()
    reg["y"] = np.where(reg["home_score"] > reg["away_score"], 1.0,
                        np.where(reg["home_score"] == reg["away_score"], 0.5, 0.0))
    reg["p_mkt"] = market_prob(reg)

    seasons = []
    for season, g in reg.groupby("season"):
        if season == partial_season:
            continue
        row = {"season": int(season), "n": int(len(g)),
               "hfa_pts": round(float(g["hfa_used"].mean()), 1)}
        non_neutral = g[~g["neutral"]]
        row["home_win_rate"] = round(float(non_neutral["y"].mean()), 4)
        row["hfa_win_pct"] = round(float(np.mean(1 / (1 + 10 ** (-non_neutral["hfa_used"] / 400)))), 4)
        m = g[g["p_mkt"].notna()]
        if len(m) >= MIN_SEASON_GAMES:
            row["n_market"] = int(len(m))
            row["elo_brier"] = round(float(np.mean((m["y"] - m["p_home"]) ** 2)), 4)
            row["market_brier"] = round(float(np.mean((m["y"] - m["p_mkt"]) ** 2)), 4)
        else:
            row["n_market"] = row["elo_brier"] = row["market_brier"] = None
        seasons.append(row)

    lo, hi = config.TEST_SEASONS
    test = reg[reg["season"].between(lo, hi)]
    mt = test[test["p_mkt"].notna()]
    elo_all = metrics(test["y"].to_numpy(), test["p_home"].to_numpy())
    elo_m = metrics(mt["y"].to_numpy(), mt["p_home"].to_numpy())
    mkt_m = metrics(mt["y"].to_numpy(), mt["p_mkt"].to_numpy())
    se = float(np.std((mt["y"] - mt["p_home"]) ** 2 - (mt["y"] - mt["p_mkt"]) ** 2, ddof=1) / np.sqrt(len(mt)))
    headline = {
        "test_seasons": [lo, hi],
        "tune_seasons": list(config.TUNE_SEASONS),
        "n_all": elo_all["n"], "n_market": mkt_m["n"],
        "elo_brier": round(elo_all["brier"], 4), "elo_accuracy": round(elo_all["accuracy"], 4),
        "elo_brier_market_games": round(elo_m["brier"], 4),
        "elo_accuracy_market_games": round(elo_m["accuracy"], 4),
        "market_brier": round(mkt_m["brier"], 4), "market_accuracy": round(mkt_m["accuracy"], 4),
        "legacy_brier": legacy_brier(),
        "brier_gap_vs_market": round(elo_m["brier"] - mkt_m["brier"], 4),
        "brier_gap_se": round(se, 4),
    }
    return {"seasons": seasons, "hfa_now_pts": hfa_now}, headline


def build_meta(ratings: dict, headline: dict, partial: bool, upcoming: dict) -> dict:
    hfa = ratings["hfa_current"]
    cfg = ratings["config"]
    return {
        "as_of": ratings["as_of"],
        "season": ratings["season"],
        "week": ratings["week"],
        "season_partial": partial,
        "hfa_pts": hfa,
        "hfa_win_pct": round(1 / (1 + 10 ** (-hfa / 400)), 4),
        "config": {k: cfg[k] for k in ("k", "lam", "hfa_init", "k_hfa", "expansion_start")},
        "scorecard": headline,
        "upcoming_week": upcoming["week"],
        "credit": CREDIT,
        "repo": REPO_URL,
    }


def build_ladder(ratings: dict, long: pd.DataFrame, season: int) -> list[dict]:
    teams = []
    for t in ratings["teams"]:
        code = t["franchise"]
        g = long[long["team"] == code]
        cur = g[g["season"] == season]
        w, l, tie = record(cur["pts"])
        tail = g.tail(SPARK_GAMES)
        teams.append({
            "team": code,
            "name": team_meta.full_name(code),
            "conf": team_meta.conference(code),
            "div": team_meta.division(code),
            "color": team_color(code),
            "rating": t["rating"],
            "rank": t["rank"],
            "rating_change": t["rating_change"],
            "rank_change": t["rank_change"],
            "w": w, "l": l, "t": tie,
            "spark": [round(float(x), 1) for x in tail["post"]],
            "spark_from": int((tail["season"] == season).to_numpy().argmax()) if len(cur) else len(tail),
        })
    return teams


def elo_pick_table(sched: pd.DataFrame, ratings_by_team: dict, hfa: float) -> pd.DataFrame:
    """Add Elo win probability and spread (home-favored positive) to scheduled games."""
    s = sched.copy()
    neutral = (s["location"] == "Neutral").to_numpy()
    s["hfa"] = np.where(neutral, 0.0, hfa)
    diff = (s["home_team"].map(ratings_by_team) + s["hfa"] - s["away_team"].map(ratings_by_team))
    s["elo_diff"] = diff
    s["p_home"] = 1 / (1 + 10 ** (-diff / 400))
    s["elo_spread"] = diff / 25.0
    return s


def build_upcoming(schedule: pd.DataFrame, ratings: dict, elo: pd.DataFrame) -> dict:
    """The next unplayed week: unplayed games with Elo pick, Elo spread, and the Vegas line.

    Uses the earliest week of the current season that still has unplayed games, so a
    Thursday as_of shows the rest of that week (played games are in the ratings already).
    """
    season = ratings["season"]
    sched = schedule[schedule["season"] == season]
    unplayed = sched[sched["home_score"].isna() & sched["away_score"].isna()]
    if unplayed.empty:
        return {"season": season, "week": None, "games": [], "played": 0, "total": 0, "has_lines": False,
                "flagged": []}
    week = int(unplayed["week"].min())
    wk = sched[(sched["week"] == week)]
    todo = unplayed[unplayed["week"] == week].copy()

    # Current ratings at full precision: last post-game rating per team.
    long = pd.concat([
        elo[["date", "game_id", "home", "home_post"]].set_axis(["date", "game_id", "team", "post"], axis=1),
        elo[["date", "game_id", "away", "away_post"]].set_axis(["date", "game_id", "team", "post"], axis=1),
    ]).sort_values(["date", "game_id"], kind="stable")
    current = long.groupby("team")["post"].last().to_dict()

    todo = elo_pick_table(todo, current, ratings["hfa_current"])
    todo["p_mkt"] = market_prob(todo)
    todo["vegas_spread"] = pd.to_numeric(todo["spread_line"], errors="coerce")
    todo["diff"] = todo["elo_spread"] - todo["vegas_spread"]
    todo = todo.sort_values(["gameday", "gametime", "game_id"], kind="stable")

    flagged = []
    if todo["diff"].notna().any():
        flagged = todo.loc[todo["diff"].abs().sort_values(ascending=False, kind="stable").index[:DISAGREEMENTS],
                           "game_id"].tolist()
    games = []
    for r in todo.itertuples():
        games.append({
            "game_id": r.game_id,
            "date": r.gameday,
            "time": clean(r.gametime),
            "away": r.away_team, "home": r.home_team,
            "neutral": bool(r.location == "Neutral"),
            "p_home": round(float(r.p_home), 4),
            "elo_spread": round(float(r.elo_spread), 1),
            "vegas_spread": clean(r.vegas_spread),
            "diff": None if pd.isna(r.diff) else round(float(r.diff), 1),
            "p_market_home": None if pd.isna(r.p_mkt) else round(float(r.p_mkt), 4),
            "flagged": r.game_id in flagged,
        })
    return {"season": season, "week": week, "played": int(len(wk) - len(todo)), "total": int(len(wk)),
            "has_lines": bool(todo["vegas_spread"].notna().any()),
            "spread_note": "Spreads are from the home team's side, positive when the home team is favored.",
            "flagged": flagged, "games": games}


def build_history(long: pd.DataFrame, ends: pd.DataFrame, partial_season: int | None) -> dict:
    out = []
    for code in team_meta.TEAM_ORDER:
        g = long[long["team"] == code]
        done = ends[(ends["team"] == code) & (ends["season"] != partial_season)]
        w, l, t = record(g["pts"])
        item = {
            "team": code, "name": team_meta.full_name(code), "color": team_color(code),
            "first_season": int(g["season"].min()),
            "all_time": {"w": w, "l": l, "t": t, "games": int(len(g))},
            "s": [int(x) for x in g["season"]], "w": [int(x) for x in g["week"]],
            "r": [round(float(x), 1) for x in g["post"]],
        }
        for key, row in (("best", done.loc[done["rating"].idxmax()]), ("worst", done.loc[done["rating"].idxmin()])):
            item[key] = {"season": int(row["season"]), "rating": round(float(row["rating"]), 1),
                         "w": int(row["w"]), "l": int(row["l"]), "t": int(row["t"])}
        out.append(item)
    return {"teams": out}


def build_luck(long: pd.DataFrame, season: int) -> dict:
    seasons = {}
    for back in LUCK_SEASONS_BACK:
        s = season - back
        g = long[long["season"] == s]
        if g.empty:
            continue
        rows = []
        for code, tg in g.groupby("team"):
            actual, expected = float(tg["pts"].sum()), float(tg["p"].sum())
            rows.append({"team": code, "games": int(len(tg)), "actual": actual,
                         "expected": round(expected, 2), "luck": round(actual - expected, 2)})
        rows.sort(key=lambda r: (-r["luck"], r["team"]))
        seasons[str(s)] = {"season": s, "teams": rows}
    return {"seasons": seasons,
            "note": "Luck is actual wins minus expected wins, where expected wins is the sum of each game's pre-game win probability."}


def build_tapestry(ends: pd.DataFrame, partial_season: int | None) -> dict:
    seasons = list(range(int(ends["season"].min()), int(ends["season"].max()) + 1))
    pivot = ends.pivot(index="team", columns="season", values="rating").reindex(
        index=team_meta.TEAM_ORDER, columns=seasons)
    values = [[None if pd.isna(v) else round(float(v), 1) for v in row] for row in pivot.to_numpy()]
    parity = [round(float(np.std(pivot[s].dropna().to_numpy())), 1) for s in seasons]
    return {"seasons": seasons, "teams": team_meta.TEAM_ORDER, "values": values, "parity": parity,
            "teams_count": [int(pivot[s].notna().sum()) for s in seasons],
            "partial_season": partial_season}


def game_label(r) -> dict:
    return {"season": int(r.season), "week": int(r.week), "type": r.game_type, "date": r.date.strftime("%Y-%m-%d"),
            "home": r.home, "away": r.away, "home_score": int(r.home_score), "away_score": int(r.away_score)}


def build_records(elo: pd.DataFrame, ends: pd.DataFrame, partial_season: int | None) -> dict:
    # Biggest upsets: lowest pre-game win probability among winners (all games incl. playoffs, ties excluded).
    decided = elo[elo["home_score"] != elo["away_score"]].copy()
    home_won = decided["home_score"] > decided["away_score"]
    decided["winner"] = np.where(home_won, decided["home"], decided["away"])
    decided["p_win"] = np.where(home_won, decided["p_home"], 1.0 - decided["p_home"])
    upsets = []
    for r in decided.sort_values(["p_win", "date", "game_id"], kind="stable").head(TOP_N).itertuples():
        upsets.append({**game_label(r), "winner": r.winner, "p_win": round(float(r.p_win), 4)})

    # Biggest single-game rating changes (the winner's gain; the loser's loss is the same size).
    upd = decided[decided["updated"]].copy()
    upd["gain"] = np.where(home_won[upd.index], upd["home_post"] - upd["home_pre"], upd["away_post"] - upd["away_pre"])
    swings = []
    for r in upd.sort_values(["gain", "date", "game_id"], ascending=[False, True, True], kind="stable").head(TOP_N).itertuples():
        swings.append({**game_label(r), "winner": r.winner, "gain": round(float(r.gain), 1),
                       "p_win": round(float(r.p_win), 4)})

    done = ends[ends["season"] != partial_season]

    def seasons_list(df):
        return [{"team": r.team, "season": int(r.season), "rating": round(float(r.rating), 1),
                 "w": int(r.w), "l": int(r.l), "t": int(r.t)} for r in df.itertuples()]

    best = done.sort_values(["rating", "season", "team"], ascending=[False, True, True]).head(TOP_N)
    worst = done.sort_values(["rating", "season", "team"], ascending=[True, True, True]).head(TOP_N)
    return {"upsets": upsets, "best_seasons": seasons_list(best), "worst_seasons": seasons_list(worst),
            "swings": swings}


# --------------------------------------------------------------------------- headlines
# One short, computed sentence per section (headlines.json). Nothing here is hard-coded
# to a team or number; each function falls back gracefully when its input is empty.

def subj(team: str) -> str:
    return team_meta.subject(team)


def cap(text: str) -> str:
    return text[0].upper() + text[1:]


def verb(team: str, singular: str, plural: str) -> str:
    return plural if team_meta.is_shared_city(team) else singular


def points(x: float) -> str:
    return f"{x:.1f} " + ("point" if f"{x:.1f}" == "1.0" else "points")


def games(x: float) -> str:
    return f"{x:.1f} " + ("game" if f"{x:.1f}" == "1.0" else "games")


def headline_ladder(teams: list[dict]) -> str:
    first, second = sorted(teams, key=lambda t: t["rank"])[:2]
    gap = round(first["rating"] - second["rating"], 1)
    a, b = first["team"], second["team"]
    if gap == 0:
        return f"{cap(subj(a))} and {subj(b)} are level at the top, both at {first['rating']:.1f}."
    if gap < 5:
        return f"{cap(subj(a))} {verb(a, 'edges', 'edge')} {subj(b)} by {points(gap)} at the top."
    return f"{cap(subj(a))} {verb(a, 'leads', 'lead')} {subj(b)} by {points(gap)} at the top."


def headline_week(upcoming: dict) -> str:
    games_ = upcoming["games"]
    if not games_:
        return "No games are left on the schedule, so there are no picks this week."
    by_id = {g["game_id"]: g for g in games_}
    if upcoming["flagged"]:
        g = by_id[upcoming["flagged"][0]]
        game = f"{subj(g['away'])} at {subj(g['home'])}"
        elo_home, vegas_home = g["elo_spread"] > 0, g["vegas_spread"] > 0
        if elo_home != vegas_home and abs(g["elo_spread"]) >= 0.05 and abs(g["vegas_spread"]) >= 0.05:
            return f"Elo and Vegas split on {game}."
        return f"Elo and Vegas are {points(abs(g['diff']))} apart on {game}."
    g = max(games_, key=lambda x: (abs(x["p_home"] - 0.5), x["game_id"]))
    fav, dog = (g["home"], g["away"]) if g["p_home"] >= 0.5 else (g["away"], g["home"])
    return f"Elo's strongest pick is {subj(fav)} over {subj(dog)}, at {max(g['p_home'], 1 - g['p_home']) * 100:.0f}%."


MIN_STRETCH_GAMES = 10   # "highest since" needs at least this many games of history behind it


def headline_explorer(item: dict, season: int, rank: int) -> str:
    """Where the team's current rating sits against its own history.

    "Highest (lowest) rating since X" when the rating has not been beaten for a real stretch of
    games; otherwise the distance from average and the league rank.
    """
    team = item["team"]
    r, seasons, weeks = item["r"], item["s"], item["w"]
    now = r[-1]
    high = now >= 1500
    beaten = [i for i in range(len(r) - 1) if (r[i] > now if high else r[i] < now)]
    plural = team_meta.is_shared_city(team)
    be, its = ("are", "their") if plural else ("is", "its")
    word = "highest" if high else "lowest"
    if not beaten and len(r) > MIN_STRETCH_GAMES:
        return f"{cap(subj(team))} {be} at {its} {word} rating since 1970, at {now:.1f}."
    if beaten and len(r) - 1 - beaten[-1] >= MIN_STRETCH_GAMES:
        i = beaten[-1]
        since = f"Week {weeks[i]} of {seasons[i]}" if seasons[i] == season else str(seasons[i])
        return f"{cap(subj(team))} {be} at {its} {word} rating since {since}, at {now:.1f}."
    side = "above" if high else "below"
    return f"{cap(subj(team))} {be} {points(abs(now - 1500))} {side} average, {ordinal(rank)} of 32."


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def headline_luck(season_block: dict, current: bool) -> str:
    rows = season_block["teams"]
    top = round(rows[0]["luck"], 1)
    if top <= 0:
        return "Nobody has won more games than Elo expected."
    leaders = [r["team"] for r in rows if round(r["luck"], 1) == top]
    names = [subj(t) for t in leaders]
    year = season_block["season"]
    if len(leaders) > 1:
        who = ", ".join(names[:-1]) + " and " + names[-1]
        tail = "so far" if current else f"in {year}"
        return f"{cap(who)} each {'have won' if current else 'won'} {games(top).replace('game', 'more game')} than expected {tail}."
    t = leaders[0]
    have = ("have" if team_meta.is_shared_city(t) else "has") if current else None
    more = games(top).replace("game", "more game")
    if current:
        return f"{cap(subj(t))} {have} won {more} than expected."
    return f"{cap(subj(t))} won {more} than expected in {year}."


def headline_history(tapestry: dict) -> str:
    partial = tapestry["partial_season"]
    pairs = [(y, p) for y, p in zip(tapestry["seasons"], tapestry["parity"]) if y != partial]
    most = min(pairs, key=lambda x: (x[1], x[0]))     # ties go to the earlier season
    least = max(pairs, key=lambda x: (x[1], -x[0]))
    first = pairs[0][0]
    if most[0] == least[0]:
        return f"Every season since {first} looks the same on parity."
    return f"{most[0]} was the most balanced season since {first}; {least[0]} was the least."


def headline_records(records: dict) -> str:
    best = records["best_seasons"][0]
    t = best["team"]
    return f"The {best['season']} {team_meta.nickname(t)} still own the top rating, {best['rating']:.0f}."


def headline_scorecard(card: dict) -> str:
    gap = card["brier_gap_vs_market"]
    if gap > 0:
        return f"Vegas still wins, by {gap:.4f} Brier."
    if gap < 0:
        return f"Elo beats Vegas, by {abs(gap):.4f} Brier."
    return "Elo and Vegas are level on Brier."


def build_headlines(ladder: list[dict], upcoming: dict, history: dict, luck: dict, tapestry: dict,
                    records: dict, card: dict, season: int, partial: bool) -> dict:
    ranks = {t["team"]: t["rank"] for t in ladder}
    return {
        "ladder": headline_ladder(ladder),
        "week": headline_week(upcoming),
        "explorer": {t["team"]: headline_explorer(t, season, ranks[t["team"]]) for t in history["teams"]},
        "luck": {k: headline_luck(v, partial and v["season"] == season) for k, v in luck["seasons"].items()},
        "history": headline_history(tapestry),
        "records": headline_records(records),
        "scorecard": headline_scorecard(card),
    }


# --------------------------------------------------------------------------- main

def export() -> dict[str, int]:
    elo, ratings, schedule = load_elo(), load_ratings(), load_schedule()
    season = ratings["season"]
    long = team_game_rows(elo)
    ends = season_end(long)
    partial = season_is_partial(schedule, season)
    partial_season = season if partial else None

    upcoming = build_upcoming(schedule, ratings, elo)
    scorecard, headline = build_scorecard(elo, ratings["hfa_current"], partial_season)
    files = {
        "meta.json": build_meta(ratings, headline, partial, upcoming),
        "ladder.json": {"teams": build_ladder(ratings, long, season)},
        "upcoming.json": upcoming,
        "history.json": build_history(long, ends, partial_season),
        "luck.json": build_luck(long, season),
        "tapestry.json": build_tapestry(ends, partial_season),
        "records.json": build_records(elo, ends, partial_season),
        "scorecard.json": scorecard,
    }
    files["headlines.json"] = build_headlines(
        files["ladder.json"]["teams"], upcoming, files["history.json"], files["luck.json"], files["tapestry.json"],
        files["records.json"], headline, season, partial)
    sizes = {name: write_json(name, obj) for name, obj in files.items()}
    total = sum(sizes.values())
    print(f"site/data: {len(sizes)} files, {total / 1024:.0f} KB total "
          f"({', '.join(f'{n} {s / 1024:.0f}K' for n, s in sizes.items())})")
    return sizes


if __name__ == "__main__":
    export()
