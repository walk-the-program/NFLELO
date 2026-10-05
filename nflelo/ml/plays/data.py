"""M6 play table: pre-snap features, yard-bin targets, and the banned-columns allowlist.

LICENSE: CC BY-SA 4.0. Built on nflverse participation data (NFL Next Gen
Stats via nflverse 2016-2022, FTN Data via nflverse 2023+), so this table,
every model fit on it and every number derived from it are CC BY-SA 4.0.
Nothing here may flow into A4s or any CC BY artifact (tests enforce the import
boundary). Spec: context/ml-m6-method.md.

One row per kept scrimmage play from the M5b cleaning
(`players.participation.build_plays`: M1 scrimmage snaps, REG and POST,
garbage time out, exactly 11 + 11 GSIS IDs after the side repair), joined to
play-by-play (situation and outcome) and to the participation pre-snap
fields. Only the columns in `PART_COLUMNS` are ever read from participation;
the post-snap ones (pass rushers, coverage, man/zone, pressure, routes, time
to throw, NGS air yards) are never touched.

Features: an explicit ALLOWLIST (`FEATURES[view]`). `check_features` rejects
any column outside it or matching a banned post-snap / outcome pattern; every
model calls it on its input columns.

    situation  down, ydstogo, yardline_100, score_diff, half_secs, game_secs,
               off_timeouts, def_timeouts, home, roof_indoor, roof_open, temp, wind
    structure  off_rb, off_te, off_wr, off_ol (personnel counts), def_dl, def_lb, def_db,
               form_shotgun, form_pistol, form_under_center (harmonized formation), box
    ratings    off_pass_o, off_rush_o (offense's M3 adjusted pass/rush offense),
               def_pass_d, def_rush_d (defense's adjusted pass/rush EPA allowed),
               as of the play's week start (the M3 as-of machinery)
    call       is_pass (call-conditioned view only; nflfastR `pass`, so sacks and scrambles are passes)

Source shift (2023, NGS -> FTN). On scrimmage plays the box count keeps its
scale (mean about 6.3-6.4 in both eras), but the vocabularies change:
formation goes from SHOTGUN / EMPTY / SINGLEBACK / I_FORM / PISTOL / JUMBO /
WILDCAT to SHOTGUN / UNDER CENTER / PISTOL, and personnel from group counts
("1 RB, 1 TE, 3 WR"; "6 OL" only when there is an extra lineman) to roster
positions ("1 C, 2 G, 2 T, 1 QB, 1 RB, 1 TE, 3 WR"; defense CB / DE / DT / NT /
ILB / MLB / OLB / FS / SS). Both are mapped to one era-free encoding here
(`harmonize_formation`, `parse_personnel`); `field_report` measures what is
left of the shift. FTN marks a missing box count as 0, read here as missing.
Missing pre-snap fields are filled, never flagged (see BASE_PERSONNEL / BOX_FILL / impute_formation).

Target (spec section 4): yards gained in 53 bins: bin 0 = -10 or worse,
bins 1..50 = -9..+40, bin 51 = 41+ short of the end zone, bin 52 =
touchdown. A play at `yardline_100 = y` can gain at most y yards: every bin
at y yards or more is folded into the touchdown bin, and losses beyond the
own goal line (a safety) into the lowest possible bin (`fold_index`).
Incomplete passes are 0 yards, sacks are negative yards. The separate
turnover flag is an interception or a lost fumble.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from ..features import opponent_adjust as oa
from ..players import participation as pt

LICENSE = "CC BY-SA 4.0"
LICENSE_NOTE = pt.LICENSE_NOTE + " M6 play outcome model (context/ml-m6-method.md)."

# --------------------------------------------------------------------------- bins

N_BINS = 53
LOSS_BIN, BIG_BIN, TD_BIN = 0, 51, 52
# Representative yards of each non-TD bin (bin 0 = "-10 or worse", bin 51 = "41+"); the TD bin's yards = yardline_100.
BIN_YARDS = np.array([-10] + list(range(-9, 41)) + [41], dtype=float)
BIN_LABELS = ["<=-10"] + [str(y) for y in range(-9, 41)] + ["41+", "TD"]


def yards_to_bin(yards: np.ndarray, td: np.ndarray, yardline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(bin, yards actually counted). Non-TD yards are clipped to the field: [y - 100, y - 1]; a TD counts y yards."""
    y = np.asarray(yardline, float)
    yd = np.clip(np.asarray(yards, float), y - 100, y - 1)
    td = np.asarray(td, bool)  # an offensive TD is the TD flag only (not yards >= y: two-point-like quirks)
    b = np.where(yd <= -10, LOSS_BIN, np.where(yd >= 41, BIG_BIN, yd + 10)).astype(int)
    b = np.where(td, TD_BIN, b)
    return b, np.where(td, y, yd)


def fold_index(yardline: np.ndarray) -> np.ndarray:
    """(n, 53) destination bin of each bin for a play at `yardline` (identity where the bin is possible)."""
    y = np.asarray(yardline, float)[:, None]
    k = np.arange(N_BINS)[None, :]
    yk = np.concatenate([BIN_YARDS, [np.inf]])[None, :]          # TD bin never folds
    low = np.where(y <= 90, LOSS_BIN, y - 90).astype(int)        # lowest possible bin (a safety at y - 100 yards)
    out = np.broadcast_to(k, (len(y), N_BINS)).copy()
    upper = (k < TD_BIN) & (yk >= y)
    lower = (k < TD_BIN) & ~upper & (yk < y - 100)
    out = np.where(upper, TD_BIN, out)
    out = np.where(lower, np.broadcast_to(low, out.shape), out)
    return out


def possible_mask(yardline: np.ndarray) -> np.ndarray:
    """(n, 53) True where the bin can happen from this yard line."""
    f = fold_index(yardline)
    return f == np.arange(N_BINS)[None, :]


def fold(p: np.ndarray, yardline: np.ndarray, floor: float = 1e-7) -> np.ndarray:
    """Fold a 53-bin distribution onto the bins possible from each play's yard line, floor them, renormalize."""
    p = np.asarray(p, float)
    f = fold_index(yardline)
    out = np.zeros_like(p)
    rows = np.repeat(np.arange(len(p)), N_BINS)
    np.add.at(out, (rows, f.ravel()), p.ravel())
    m = f == np.arange(N_BINS)[None, :]
    out = np.where(m, np.maximum(out, floor), 0.0)
    return out / out.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- allowlist

SITUATION = ["down", "ydstogo", "yardline_100", "score_diff", "half_secs", "game_secs", "off_timeouts",
             "def_timeouts", "home", "roof_indoor", "roof_open", "temp", "wind"]
STRUCTURE = ["off_rb", "off_te", "off_wr", "off_ol", "def_dl", "def_lb", "def_db",
             "form_shotgun", "form_pistol", "form_under_center", "box"]
RATINGS = ["off_pass_o", "off_rush_o", "def_pass_d", "def_rush_d"]
CALL = ["is_pass"]
VIEWS = ("situation", "call")
FEATURES = {"situation": SITUATION + STRUCTURE + RATINGS, "call": SITUATION + STRUCTURE + RATINGS + CALL}
ALLOWLIST = frozenset(FEATURES["call"])

# Post-snap or outcome fields: never a feature. Matched case-insensitively against column names.
BANNED_PATTERNS = (r"pass_rusher", r"rushers", r"coverage", r"man_zone", r"pressure", r"route", r"time_to_throw",
                   r"air_yards", r"yards_after_catch", r"(^|_)yac", r"(^|_)epa($|_)", r"(^|_)wpa($|_)", r"_result$",
                   r"yards_gained", r"touchdown", r"interception", r"fumble", r"sack", r"complete", r"first_down",
                   r"success", r"turnover", r"^yards$", r"^bin$", r"^td$", r"^ev_", r"penalty", r"qb_hit",
                   r"tackle", r"scramble", r"own_yards")
_BANNED = re.compile("|".join(BANNED_PATTERNS), re.IGNORECASE)


def is_banned(name: str) -> bool:
    return bool(_BANNED.search(str(name)))


def check_features(cols: Iterable[str]) -> list[str]:
    """Raise unless every column is on the allowlist and none matches a banned pattern. Returns the list."""
    cols = list(cols)
    bad = [c for c in cols if is_banned(c)]
    if bad:
        raise ValueError(f"banned post-snap/outcome columns among the features: {bad}")
    extra = [c for c in cols if c not in ALLOWLIST]
    if extra:
        raise ValueError(f"columns outside the M6 pre-snap allowlist: {extra}")
    return cols


# --------------------------------------------------------------------------- sources

KEYS = ["game_id", "play_id"]
PBP_SITUATION = ["down", "ydstogo", "yardline_100", "score_differential", "half_seconds_remaining",
                 "game_seconds_remaining", "posteam_timeouts_remaining", "defteam_timeouts_remaining",
                 "roof", "temp", "wind", "pass", "shotgun"]
PBP_OUTCOME = ["yards_gained", "touchdown", "td_team", "posteam", "interception", "fumble_lost"]
PBP_COLUMNS = KEYS + PBP_SITUATION + PBP_OUTCOME
# The only participation columns M6 reads (all pre-snap). Post-snap columns are never loaded.
PART_COLUMNS = ["nflverse_game_id", "play_id", "offense_formation", "offense_personnel", "defense_personnel",
                "defenders_in_box"]
OUTCOMES = ["yards", "bin", "td", "turnover", "ev_first", "ev_20", "ev_loss"]
INFO = ["game_id", "play_id", "season", "week", "season_type", "kick_ns", "off_team", "def_team"]

OFF_GROUP = {"RB": "rb", "FB": "rb", "HB": "rb", "TE": "te", "WR": "wr",
             "OL": "ol", "C": "ol", "G": "ol", "T": "ol", "OT": "ol", "OG": "ol"}
DEF_GROUP = {"DL": "dl", "DE": "dl", "DT": "dl", "NT": "dl",
             "LB": "lb", "ILB": "lb", "MLB": "lb", "OLB": "lb",
             "DB": "db", "CB": "db", "S": "db", "FS": "db", "SS": "db"}
FORMATION = {"SHOTGUN": "shotgun", "EMPTY": "shotgun", "PISTOL": "pistol",
             "SINGLEBACK": "under_center", "I_FORM": "under_center", "JUMBO": "under_center",
             "UNDER CENTER": "under_center", "UNDER_CENTER": "under_center", "WILDCAT": "other"}


def parse_personnel(s, side: str) -> dict:
    """Group counts from a personnel string of either era. Offense: rb, te, wr, ol; defense: dl, lb, db.

    NGS (2016-2022) lists offensive linemen only when there are more than five
    ("6 OL, ..."), so a missing OL token means 5. FTN (2023+) lists every
    lineman by position (C, G, T). Missing strings give NaN counts.
    """
    groups = ("rb", "te", "wr", "ol") if side == "off" else ("dl", "lb", "db")
    if not isinstance(s, str) or not s.strip():
        return {g: np.nan for g in groups}
    lut = OFF_GROUP if side == "off" else DEF_GROUP
    out = {g: 0.0 for g in groups}
    saw_line = False
    for tok in s.split(","):
        parts = tok.strip().split(" ", 1)
        if len(parts) != 2:
            continue
        try:
            n = float(parts[0])
        except ValueError:
            continue
        g = lut.get(parts[1].strip().upper())
        if g in out:
            out[g] += n
            saw_line |= g == "ol"
    if side == "off" and not saw_line:
        out["ol"] = 5.0
    return out


def personnel_counts(strings: pd.Series, side: str) -> pd.DataFrame:
    uniq = pd.Series(strings.dropna().unique())
    table = pd.DataFrame([parse_personnel(s, side) for s in uniq], index=uniq.to_numpy())
    groups = ("rb", "te", "wr", "ol") if side == "off" else ("dl", "lb", "db")
    if table.empty:
        return pd.DataFrame({g: np.full(len(strings), np.nan) for g in groups}, index=strings.index)
    return table.reindex(strings.to_numpy()).set_axis(strings.index)


# Missing pre-snap fields are filled, never flagged: in the NGS seasons the formation is missing on
# about 1% of plays and 83-89% of those are fumbles (2016-2018; strip sacks), so "formation missing"
# would be an outcome leak. Formation falls back to play-by-play's own `shotgun` flag (CC BY; agrees
# with the participation formation on 98.6-99.7% of plays), box count to the typical box for the
# formation, personnel to 11 personnel against nickel.
BASE_PERSONNEL = {"off_rb": 1.0, "off_te": 1.0, "off_wr": 3.0, "off_ol": 5.0, "def_dl": 4.0, "def_lb": 2.0,
                  "def_db": 5.0}
BOX_FILL = {"under_center": 7.0, "other": 6.0}


def impute_formation(form: pd.Series, shotgun: pd.Series) -> pd.Series:
    """Missing harmonized formation -> shotgun (pbp `shotgun` = 1, which includes pistol) or under_center."""
    fill = pd.Series(np.where(shotgun.fillna(0).to_numpy() == 1, "shotgun", "under_center"), index=form.index)
    return form.where(form.notna(), fill)


def harmonize_formation(s: pd.Series) -> pd.Series:
    """Era-free formation: shotgun (incl. NGS EMPTY), pistol, under_center, other (wildcat), or NaN."""
    return s.astype("string").str.strip().str.upper().map(FORMATION).astype(object)


# --------------------------------------------------------------------------- ratings

def rating_keys(plays: pd.DataFrame) -> list[tuple[int, int]]:
    return sorted(set(zip(plays["season"].astype(int), plays["week"].astype(int))))


def week_ratings(tab: pd.DataFrame, sched: pd.DataFrame, keys) -> pd.DataFrame:
    """M3 opponent-adjusted pass and rush ratings (oa.TUNED) as of each (season, week)'s start (its as_of)."""
    return oa.compute_ratings(tab, sched, keys, oa.TUNED, ("pass", "rush"))


def rating_features(plays: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Offense's adjusted pass/rush O and defense's adjusted pass/rush D as of the play's (season, week)."""
    s, w = plays["season"].to_numpy(int), plays["week"].to_numpy(int)
    o, d = plays["off_team"].to_numpy(object), plays["def_team"].to_numpy(object)
    return pd.DataFrame({"off_pass_o": oa._lookup(ratings, "pass", s, w, o, "off"),
                         "off_rush_o": oa._lookup(ratings, "rush", s, w, o, "off"),
                         "def_pass_d": oa._lookup(ratings, "pass", s, w, d, "def"),
                         "def_rush_d": oa._lookup(ratings, "rush", s, w, d, "def")}, index=plays.index)


# --------------------------------------------------------------------------- the table

def _ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["play_id"] = pd.to_numeric(out["play_id"]).astype("int64")
    return out


def build_table(kept: pd.DataFrame, pbp: pd.DataFrame, part: pd.DataFrame, ratings: pd.DataFrame,
                extra_features: dict | None = None) -> pd.DataFrame:
    """The M6 play table (CC BY-SA 4.0): INFO + player IDs + every allowlisted feature + OUTCOMES.

    kept      M5b kept plays (`participation.build_plays` / `load_plays`)
    pbp       play-by-play with PBP_COLUMNS
    part      participation; only PART_COLUMNS are read
    ratings   `week_ratings` output covering every play's (season, week)
    extra_features  {name: fn(joined frame) -> array}: ONLY for the leakage test's positive control.
    """
    k = _ids(kept)
    p = _ids(pbp[[c for c in PBP_COLUMNS if c in pbp.columns]]).drop_duplicates(KEYS)
    pa = _ids(part[PART_COLUMNS].rename(columns={"nflverse_game_id": "game_id"})).drop_duplicates(KEYS)
    j = (k[INFO + ["home"] + pt.OFF_COLS + pt.DEF_COLS]
         .merge(p.drop(columns=["posteam"]).rename(columns={c: f"pbp_{c}" for c in PBP_SITUATION}),
                on=KEYS, how="left", validate="one_to_one")
         .merge(p[KEYS + ["posteam"]], on=KEYS, how="left")
         .merge(pa, on=KEYS, how="left", validate="one_to_one"))
    out = j[INFO + pt.OFF_COLS + pt.DEF_COLS].copy()
    out["down"] = j["pbp_down"].fillna(1).astype(float)
    out["ydstogo"] = j["pbp_ydstogo"].fillna(10).astype(float)
    out["yardline_100"] = j["pbp_yardline_100"].fillna(50).clip(1, 99).astype(float)
    out["score_diff"] = j["pbp_score_differential"].astype(float)
    out["half_secs"] = j["pbp_half_seconds_remaining"].astype(float)
    out["game_secs"] = j["pbp_game_seconds_remaining"].astype(float)
    out["off_timeouts"] = j["pbp_posteam_timeouts_remaining"].astype(float)
    out["def_timeouts"] = j["pbp_defteam_timeouts_remaining"].astype(float)
    out["home"] = k["home"].to_numpy(float)
    roof = j["pbp_roof"].astype("string").str.lower()
    out["roof_indoor"] = roof.isin(["dome", "closed"]).astype(float).to_numpy()
    out["roof_open"] = (roof == "open").astype(float).to_numpy()
    out["temp"] = pd.to_numeric(j["pbp_temp"], errors="coerce").astype(float)
    out["wind"] = pd.to_numeric(j["pbp_wind"], errors="coerce").astype(float)
    oc = personnel_counts(j["offense_personnel"], "off")
    dc = personnel_counts(j["defense_personnel"], "def")
    for g in ("rb", "te", "wr", "ol"):
        out[f"off_{g}"] = oc[g].to_numpy(float)
    for g in ("dl", "lb", "db"):
        out[f"def_{g}"] = dc[g].to_numpy(float)
    for g, v in BASE_PERSONNEL.items():                       # missing personnel (a handful of plays)
        out[g] = out[g].fillna(v)
    form = impute_formation(harmonize_formation(j["offense_formation"]), j["pbp_shotgun"])
    for f in ("shotgun", "pistol", "under_center"):
        out[f"form_{f}"] = (form == f).astype(float).to_numpy()
    box = pd.to_numeric(j["defenders_in_box"], errors="coerce").astype(float)
    box = box.where(box > 0)                                  # FTN codes a missing box count as 0
    out["box"] = box.fillna(pd.Series(np.where(form == "under_center", BOX_FILL["under_center"],
                                               BOX_FILL["other"]), index=box.index)).to_numpy()
    out = pd.concat([out, rating_features(out, ratings)], axis=1)
    out["is_pass"] = (j["pbp_pass"].fillna(0) == 1).astype(float).to_numpy()
    # targets
    yl = out["yardline_100"].to_numpy(float)
    td = ((j["touchdown"].fillna(0) == 1) & (j["td_team"] == j["posteam"])).to_numpy()
    b, yds = yards_to_bin(j["yards_gained"].fillna(0).to_numpy(float), td, yl)
    out["yards"] = yds
    out["bin"] = b
    out["td"] = td.astype(float)
    out["turnover"] = ((j["interception"].fillna(0) == 1) | (j["fumble_lost"].fillna(0) == 1)).astype(float).to_numpy()
    ytg = out["ydstogo"].to_numpy(float)
    out["ev_first"] = ((yds >= ytg) | td).astype(float)
    out["ev_20"] = (yds >= 20).astype(float)
    out["ev_loss"] = (yds < 0).astype(float)
    for name, fn in (extra_features or {}).items():
        out[name] = np.asarray(fn(j), float)
    out.attrs["license"] = LICENSE_NOTE
    return out.reset_index(drop=True)


def era(season) -> np.ndarray:
    return np.where(np.asarray(season, int) >= 2023, "ftn", "ngs")


def field_report(kept: pd.DataFrame, part: pd.DataFrame) -> pd.DataFrame:
    """Per-season distributions of the pre-snap participation fields on kept plays (no outcome is read).

    Formation shares (harmonized and raw vocabulary size), box count mean / SD /
    missing share, personnel group means, the most common offensive grouping
    share (11 personnel), and missing shares.
    """
    k = _ids(kept[["game_id", "play_id", "season"]])
    pa = _ids(part[PART_COLUMNS].rename(columns={"nflverse_game_id": "game_id"})).drop_duplicates(KEYS)
    j = k.merge(pa, on=KEYS, how="left")
    rows = []
    for S, g in j.groupby("season"):
        form = harmonize_formation(g["offense_formation"])
        box = pd.to_numeric(g["defenders_in_box"], errors="coerce").astype(float)
        box = box.where(box > 0)
        oc = personnel_counts(g["offense_personnel"], "off")
        dc = personnel_counts(g["defense_personnel"], "def")
        eleven = (oc["rb"] == 1) & (oc["te"] == 1) & (oc["wr"] == 3)
        rows.append({"season": int(S), "era": str(era([S])[0]), "plays": int(len(g)),
                     "raw_formations": int(g["offense_formation"].nunique()),
                     "form_missing": float(form.isna().mean()),
                     "shotgun": float((form == "shotgun").mean()), "pistol": float((form == "pistol").mean()),
                     "under_center": float((form == "under_center").mean()), "other": float((form == "other").mean()),
                     "box_missing": float(box.isna().mean()), "box_mean": float(box.mean()), "box_sd": float(box.std()),
                     "box_8plus": float((box >= 8).mean()),
                     "pers_missing": float(oc["rb"].isna().mean()),
                     "rb": float(oc["rb"].mean()), "te": float(oc["te"].mean()), "wr": float(oc["wr"].mean()),
                     "ol": float(oc["ol"].mean()), "p11": float(eleven.mean()),
                     "dl": float(dc["dl"].mean()), "lb": float(dc["lb"].mean()), "db": float(dc["db"].mean()),
                     "nickel_or_more": float((dc["db"] >= 5).mean())})
    return pd.DataFrame(rows)
