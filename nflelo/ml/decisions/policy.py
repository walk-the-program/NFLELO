"""M7b early-down play type x personnel: propensities, cross-fit AIPW, recommendations, off-policy evaluation.

LICENSE: CC BY-SA 4.0. The actions are offensive personnel groupings from
nflverse participation data (NFL Next Gen Stats via nflverse 2016-2022, FTN
Data via nflverse 2023+), read through the M6 play table, so every model fit
here, every estimate, recommendation and evaluation number is CC BY-SA 4.0.
Nothing here may flow into A4s or any CC BY artifact (tests enforce the
import boundary). Spec: context/ml-m7-method.md, section 4 (Part B).

Why this is causal (context/ml.md, section 7): teams pick personnel and play
type BECAUSE of the situation, so raw averages by action mix the effect of
the call with the effect of the spot. Heavy personnel shows up at the goal
line, where plays gain few yards whatever is called.

Sample. M6 kept plays (M5b cleaning: scrimmage runs and passes with an EPA,
REG and POST, garbage time out: nflfastR wp outside [0.05, 0.95]), 1st and
2nd down, outside the last two minutes of each half (half_secs > 120).

Actions (12). Offensive personnel grouping x play type. Grouping from the M6
harmonized counts: "RB TE" with exactly five linemen and five skill players,
kept for 11, 12, 21, 13 and 10; everything else (22, 20, 6+ linemen, empty
backfields, ...) is pooled as "oth". Play type: nflfastR `pass` (sacks and
scrambles are passes) or run.

Outcomes. Play EPA (primary) and success (EPA > 0, nflfastR's definition; secondary).

Situation cells (5 x 4 x 3 = 60), for reporting and recommendations:
    down-distance  1-10+ (1st, 10 or more to go), 1-short (1st, under 10: mostly goal to go),
                   2-short (2nd, 1-3), 2-mid (2nd, 4-7), 2-long (2nd, 8+)
    field zone     own 1-20 (yardline_100 81-99), own 21-50 (50-80), opp 49-21 (21-49), red zone (1-20)
    score state    trailing by 4+, within 3, leading by 4+

Nuisance features (pre-snap, as of the snap): the M6 allowlist's situation
fields and the M3 ratings as of the week start (the offense's pass / rush
offense and the opponent's pass / rush defense), plus the offense's
team-season tendencies as of the week start (`tendencies`). The M6
structure fields (personnel counts, defensive personnel, formation, box)
are NEVER features here: offensive personnel IS the action, and the
defense's personnel, the box and the formation respond to it (they are
downstream of the choice). Conditioning on them would block part of the
effect we want, and under a counterfactual action they are unknown.

Nuisances. Propensity: multiclass GBM P(action | x). Outcome: GBM
E[Y | x, action] with the action as a categorical input (EPA: squared
error; success: log loss). Early stopping on a fixed 10% of the training
games (by game_id hash), seeded, deterministic.

AIPW. Pseudo-outcome of action a for play i:
    G[i, a] = mu_a(x_i) + 1[A_i = a] (Y_i - mu_a(x_i)) / e_a(x_i)
Overlap: a is eligible for play i only if e_a(x_i) >= OVERLAP (0.05).
Cross-fitting by season: to estimate the cells used for target season S,
each training season T in FIRST..S-1 gets nuisances fit on the other
training seasons. The target season S gets nuisances fit on all of
FIRST..S-1 (season-ahead).

Cells and policy. For cell c and action a, P = the cell's training plays on
which a is eligible. gain(c, a) = mean over P of (G[i, a] - Y_i): the
estimated EPA of calling a instead of the current mix, on the same plays.
An action is a candidate if it is eligible on at least CAND_SHARE of the
cell's plays and was actually called at least MIN_TAKEN times there. The
recommendation is the candidate with the highest gain; it is CLEAR only if
the lower end of its 95% game-cluster bootstrap CI is above 0 (it beats the
current mix); otherwise the cell is "no clear best".

Off-policy evaluation of season S (season-ahead): the recommended policy
calls the cell's clear recommendation a* when it is eligible for the play
(by S's propensity), and otherwise does what the team did. Per play
d_i = 1[policy deviates or applies] (G[i, a*] - Y_i); its mean is the EPA per
early-down play gained over the observed policy (game-cluster CI). The
observed policy's own DR value, sum_a e_a G[i, a], must match mean(Y).

Game theory. If a team always took the "best" action, the defense would
adjust. These estimates describe marginal shifts from current tendencies
(in the mix teams and defenses actually played against), not a fixed
strategy to run every time.
"""
from __future__ import annotations

import time
import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from ..plays import data as pdata

LICENSE = "CC BY-SA 4.0"
LICENSE_NOTE = pdata.LICENSE_NOTE + " M7b early-down play type x personnel (context/ml-m7-method.md, section 4)."

FIRST = 2016                      # first participation season
SEED = 20261003
GROUPS = ("11", "12", "21", "13", "10", "oth")
CALLS = ("run", "pass")
ACTIONS = tuple(f"{g}_{c}" for g in GROUPS for c in CALLS)
N_ACTIONS = len(ACTIONS)
OVERLAP = 0.05                    # an action is eligible for a play only if its propensity is at least this
OVERLAP_ALT = 0.10                # sensitivity
TRIM_FLOOR = 0.10                 # trimming variant: propensities floored here in the IPW term (weights <= 10)
CAND_SHARE = 0.5                  # a cell candidate is eligible on at least half the cell's plays ...
MIN_TAKEN = 30                    # ... and was called at least this often there
TEND_M = 25                       # tendency shrinkage (plays) toward the prior mix
PRIOR_LEAGUE = 0.5                # the prior: last season's team mix shrunk halfway to last season's league mix
VAL_MOD = 10                      # 1 game in VAL_MOD (by game_id hash) is the early-stopping set
REPS = 2000

SITUATION = list(pdata.SITUATION)
RATINGS = list(pdata.RATINGS)
TENDENCY = [f"tend_{a}" for a in ACTIONS]
FEATURES = SITUATION + RATINGS + TENDENCY
MEDIATORS = frozenset(pdata.STRUCTURE)       # the action itself, or downstream of it: never a feature
PROP_PARAMS = {"learning_rate": 0.05, "max_leaf_nodes": 7, "min_samples_leaf": 500, "l2_regularization": 1.0}
OUT_PARAMS = {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 400, "l2_regularization": 1.0}
MAX_ITER = 500

DD = ("1-10+", "1-short", "2-short", "2-mid", "2-long")
ZONES = ("own 1-20", "own 21-50", "opp 49-21", "red zone")
SCORES = ("trail 4+", "within 3", "lead 4+")
CELLS = tuple(f"{d} | {z} | {s}" for d in DD for z in ZONES for s in SCORES)


# --------------------------------------------------------------------------- features

def check_features(cols) -> list[str]:
    """Raise unless every column is pre-snap and allowed: M6 situation / ratings fields or a tendency.

    The M6 structure fields are rejected (mediators), as is anything matching an M6 banned
    post-snap / outcome pattern.
    """
    cols = list(cols)
    bad = [c for c in cols if pdata.is_banned(c)]
    if bad:
        raise ValueError(f"banned post-snap/outcome columns among the features: {bad}")
    med = [c for c in cols if c in MEDIATORS]
    if med:
        raise ValueError(f"personnel / formation / box are the action or downstream of it, never features: {med}")
    extra = [c for c in cols if c not in pdata.ALLOWLIST and c not in TENDENCY]
    if extra:
        raise ValueError(f"columns outside the M7b allowlist: {extra}")
    return cols


def personnel_group(t: pd.DataFrame) -> np.ndarray:
    rb, te, wr, ol = (t[c].to_numpy(float) for c in ("off_rb", "off_te", "off_wr", "off_ol"))
    std = (ol == 5) & (rb + te + wr == 5)
    code = np.char.add(rb.astype(int).astype(str), te.astype(int).astype(str))
    return np.where(std & np.isin(code, GROUPS[:-1]), code, "oth")


def action_codes(t: pd.DataFrame) -> np.ndarray:
    g = personnel_group(t)
    call = np.where(t["is_pass"].to_numpy(float) == 1, "pass", "run")
    lut = {a: i for i, a in enumerate(ACTIONS)}
    return np.array([lut[f"{x}_{c}"] for x, c in zip(g, call)], dtype=int)


def sample_mask(t: pd.DataFrame) -> np.ndarray:
    """1st and 2nd down outside the last two minutes of each half (garbage time is already out of the M6 table)."""
    return (t["down"].isin([1, 2]) & (t["half_secs"] > 120)).to_numpy()


def cell_labels(t: pd.DataFrame) -> np.ndarray:
    d, y = t["down"].to_numpy(int), t["ydstogo"].to_numpy(float)
    dd = np.where(d == 1, np.where(y >= 10, DD[0], DD[1]),
                  np.where(y <= 3, DD[2], np.where(y <= 7, DD[3], DD[4])))
    yl = t["yardline_100"].to_numpy(float)
    z = np.where(yl >= 81, ZONES[0], np.where(yl >= 50, ZONES[1], np.where(yl >= 21, ZONES[2], ZONES[3])))
    s = t["score_diff"].to_numpy(float)
    sc = np.where(s <= -4, SCORES[0], np.where(s >= 4, SCORES[2], SCORES[1]))
    return np.char.add(np.char.add(np.char.add(np.char.add(dd.astype(str), " | "), z.astype(str)), " | "), sc.astype(str))


def tendencies(df: pd.DataFrame, m: float = TEND_M, prior_league: float = PRIOR_LEAGUE) -> pd.DataFrame:
    """The offense's early-down action mix as of the start of the play's week (TENDENCY columns).

    For team k, season S, week w: counts of k's sample plays in weeks < w of S, shrunk with
    weight `m` toward a prior: k's full mix in S-1 shrunk toward the league's S-1 mix (weight
    `prior_league`) when S-1 is in the data, else the league's mix in weeks < w of S (else
    uniform). Only earlier weeks and earlier seasons are read. Settings (m 25, prior_league 0.5,
    and the propensity GBM's PROP_PARAMS) were chosen on dev 2018-2019 by the season-ahead
    propensity log loss and per-action ECE only (m in 10-500, prior_league 0 / 0.5, two GBM
    settings), never by any EPA or OPE number: team mixes carry over between seasons less than a
    full-weight prior assumes, and this season's earlier weeks are worth a lot.
    Needs off_team, season, week, action.
    """
    K = N_ACTIONS
    key = df[["off_team", "season", "week"]].copy()
    oh = np.eye(K)[df["action"].to_numpy(int)]
    wk = pd.concat([key.reset_index(drop=True), pd.DataFrame(oh, columns=ACTIONS)], axis=1)
    wk = wk.groupby(["off_team", "season", "week"], sort=True)[list(ACTIONS)].sum()
    # cumulative counts before each week, per team-season
    cum = wk.groupby(level=[0, 1]).cumsum() - wk
    # team full-season mixes (for the next season's prior)
    full = wk.groupby(level=[0, 1]).sum()
    full = full.div(full.sum(axis=1), axis=0)
    # league counts before each week of a season
    lg = wk.groupby(level=[1, 2]).sum().sort_index()
    lg_cum = lg.groupby(level=0).cumsum() - lg
    lg_n = lg_cum.sum(axis=1)
    lg_mix = lg_cum.div(lg_n.where(lg_n > 0), axis=0).fillna(1.0 / K)
    idx = pd.MultiIndex.from_frame(key)
    c = cum.reindex(idx).to_numpy(float)
    prev = pd.MultiIndex.from_arrays([key["off_team"], key["season"] - 1])
    prior = full.reindex(prev).to_numpy(float)
    lgp = lg_mix.reindex(pd.MultiIndex.from_arrays([key["season"], key["week"]])).to_numpy(float)
    if prior_league > 0:              # shrink last season's team mix toward last season's league mix
        lg_full = wk.groupby(level=1).sum()
        lg_full = lg_full.div(lg_full.sum(axis=1), axis=0)
        lprev = lg_full.reindex(key["season"] - 1).to_numpy(float)
        prior = np.where(np.isnan(lprev), prior, (1 - prior_league) * prior + prior_league * lprev)
    prior = np.where(np.isnan(prior).any(axis=1, keepdims=True), lgp, prior)
    n = c.sum(axis=1, keepdims=True)
    out = (c + m * prior) / (n + m)
    return pd.DataFrame(out, columns=TENDENCY, index=df.index)


def prepare(table: pd.DataFrame) -> pd.DataFrame:
    """The M7b frame: the sample, action, cell, outcomes and tendencies. `table` = M6 table + `epa`."""
    t = table[sample_mask(table)].copy()
    t["action"] = action_codes(t)
    t["cell"] = cell_labels(t)
    t["y_epa"] = t["epa"].to_numpy(float)
    t["y_succ"] = (t["epa"].to_numpy(float) > 0).astype(float)
    t = t.sort_values(["season", "week", "game_id", "play_id"], kind="stable").reset_index(drop=True)
    return with_tendencies(t)


def with_tendencies(t: pd.DataFrame) -> pd.DataFrame:
    t = t.drop(columns=[c for c in TENDENCY if c in t.columns])
    out = pd.concat([t, tendencies(t)], axis=1)
    out.attrs["license"] = LICENSE_NOTE
    return out


# --------------------------------------------------------------------------- nuisances

def _val_games(game_ids) -> np.ndarray:
    g = np.asarray(game_ids, dtype=object)
    u, inv = np.unique(g, return_inverse=True)
    h = np.array([zlib.crc32(str(x).encode()) % VAL_MOD == 0 for x in u])
    return h[inv]


@dataclass
class Nuisance:
    prop: HistGradientBoostingClassifier
    epa: HistGradientBoostingRegressor
    succ: HistGradientBoostingClassifier
    info: dict = field(default_factory=dict)


def _xa(df: pd.DataFrame, a) -> np.ndarray:
    X = df[FEATURES].to_numpy(float)
    act = np.full((len(df), 1), a, float) if np.isscalar(a) else np.asarray(a, float)[:, None]
    return np.hstack([X, act])


def fit_nuisance(train: pd.DataFrame) -> Nuisance:
    """Propensity, EPA and success models on the training plays (early stopping on 10% of games)."""
    check_features(FEATURES)
    t0 = time.perf_counter()
    v = _val_games(train["game_id"])
    if v.all() or not v.any():
        v = np.zeros(len(train), bool)
        v[::VAL_MOD] = True
    tr, va = train[~v], train[v]
    a_tr, a_va = tr["action"].to_numpy(int), va["action"].to_numpy(int)
    common = dict(max_iter=MAX_ITER, early_stopping=True, n_iter_no_change=20, random_state=SEED)
    seen = np.isin(a_va, np.unique(a_tr))
    prop = HistGradientBoostingClassifier(**PROP_PARAMS, **common).fit(
        tr[FEATURES].to_numpy(float), a_tr, X_val=va[FEATURES].to_numpy(float)[seen], y_val=a_va[seen])
    t1 = time.perf_counter()
    cat = [len(FEATURES)]
    epa = HistGradientBoostingRegressor(**OUT_PARAMS, **common, categorical_features=cat).fit(
        _xa(tr, a_tr), tr["y_epa"].to_numpy(float), X_val=_xa(va, a_va), y_val=va["y_epa"].to_numpy(float))
    succ = HistGradientBoostingClassifier(**OUT_PARAMS, **common, categorical_features=cat).fit(
        _xa(tr, a_tr), tr["y_succ"].to_numpy(int), X_val=_xa(va, a_va), y_val=va["y_succ"].to_numpy(int))
    t2 = time.perf_counter()
    return Nuisance(prop, epa, succ, {"n_train": int(len(train)), "seasons": sorted(map(int, train["season"].unique())),
                                      "prop_iter": int(prop.n_iter_), "epa_iter": int(epa.n_iter_),
                                      "succ_iter": int(succ.n_iter_), "prop_seconds": t1 - t0,
                                      "outcome_seconds": t2 - t1})


def predict_nuisance(nu: Nuisance, df: pd.DataFrame) -> dict:
    """{"e": (n, 12) propensities, "mu_epa": (n, 12), "mu_succ": (n, 12)} for every action."""
    n = len(df)
    e = np.zeros((n, N_ACTIONS))
    if n:
        e[:, nu.prop.classes_.astype(int)] = nu.prop.predict_proba(df[FEATURES].to_numpy(float))
    mu_e, mu_s = np.zeros((n, N_ACTIONS)), np.zeros((n, N_ACTIONS))
    for a in range(N_ACTIONS):
        if n:
            X = _xa(df, a)
            mu_e[:, a] = nu.epa.predict(X)
            mu_s[:, a] = nu.succ.predict_proba(X)[:, 1]
    return {"e": e, "mu_epa": mu_e, "mu_succ": mu_s}


def cross_fit(train: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Out-of-fold nuisance predictions for the training plays, folds = seasons (each season's nuisances
    are fit on the other training seasons). Needs at least two training seasons."""
    seasons = sorted(train["season"].unique())
    if len(seasons) < 2:
        raise ValueError("cross-fitting by season needs at least two training seasons")
    out = {k: np.zeros((len(train), N_ACTIONS)) for k in ("e", "mu_epa", "mu_succ")}
    info = []
    s = train["season"].to_numpy(int)
    for T in seasons:
        m = s == T
        nu = fit_nuisance(train[~m])
        p = predict_nuisance(nu, train[m])
        for k in out:
            out[k][m] = p[k]
        info.append({"fold": int(T), **nu.info})
    return out, info


def pseudo(e: np.ndarray, mu: np.ndarray, a_obs: np.ndarray, y: np.ndarray, floor: float | None = None) -> np.ndarray:
    """AIPW pseudo-outcomes G (n, 12). With `floor`, propensities are floored there in the IPW term (trimming)."""
    a_obs = np.asarray(a_obs, int)
    n = len(a_obs)
    rows = np.arange(n)
    ea = e[rows, a_obs]
    if floor is not None:
        ea = np.maximum(ea, floor)
    g = mu.copy()
    g[rows, a_obs] += (np.asarray(y, float) - mu[rows, a_obs]) / np.maximum(ea, 1e-6)
    return g


# --------------------------------------------------------------------------- cells and policy

def _cluster_boot_many(vals: np.ndarray, mask: np.ndarray, cell_idx: np.ndarray, n_cells: int, games,
                       reps: int = REPS, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Game-cluster bootstrap 95% CIs of mean(vals[:, a]) over plays with mask[:, a], per (cell, action).

    vals, mask: (n, A). Returns lo, hi of shape (n_cells, A) (NaN where empty)."""
    codes, _ = pd.factorize(np.asarray(games, dtype=object))
    G = int(codes.max()) + 1
    A = vals.shape[1]
    K = n_cells * A
    key = (cell_idx[:, None] * A + np.arange(A)[None, :])
    w = mask.astype(float)
    S = np.zeros((G, K))
    N = np.zeros((G, K))
    np.add.at(S, (np.repeat(codes, A), key.ravel()), (np.where(mask, vals, 0.0)).ravel())
    np.add.at(N, (np.repeat(codes, A), key.ravel()), w.ravel())
    rng = np.random.default_rng(seed)
    lo, hi = np.full(K, np.nan), np.full(K, np.nan)
    means = np.empty((reps, K))
    chunk = 250
    for st in range(0, reps, chunk):
        k = min(chunk, reps - st)
        idx = rng.integers(0, G, size=(k, G))
        W = np.stack([np.bincount(r, minlength=G) for r in idx]).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            means[st:st + k] = (W @ S) / (W @ N)
    ok = N.sum(axis=0) > 0
    lo[ok] = np.nanquantile(means[:, ok], 0.025, axis=0)
    hi[ok] = np.nanquantile(means[:, ok], 0.975, axis=0)
    return lo.reshape(n_cells, A), hi.reshape(n_cells, A)


def cell_table(df: pd.DataFrame, e: np.ndarray, g: np.ndarray, y: np.ndarray, thr: float = OVERLAP,
               reps: int = REPS, seed: int = SEED) -> pd.DataFrame:
    """Per (cell, action): eligibility, DR mean EPA, observed mean on the same plays, gain = DR - observed,
    with game-cluster CIs, and the candidate flag."""
    cell = df["cell"].to_numpy(object)
    ci = pd.Index(CELLS).get_indexer(cell)
    if (ci < 0).any():
        raise ValueError("unknown cell labels")
    a_obs = df["action"].to_numpy(int)
    y = np.asarray(y, float)
    elig = e >= thr
    gain = g - y[:, None]
    lo_g, hi_g = _cluster_boot_many(gain, elig, ci, len(CELLS), df["game_id"], reps, seed)
    lo_d, hi_d = _cluster_boot_many(g, elig, ci, len(CELLS), df["game_id"], reps, seed)
    n_cell = np.bincount(ci, minlength=len(CELLS))
    rows = []
    taken = np.zeros((len(CELLS), N_ACTIONS))
    np.add.at(taken, (ci, a_obs), 1.0)
    for c in range(len(CELLS)):
        m = ci == c
        if not m.any():
            continue
        for a in range(N_ACTIONS):
            p = m & elig[:, a]
            ne = int(p.sum())
            nt = int((p & (a_obs == a)).sum())
            rows.append({"cell": CELLS[c], "action": ACTIONS[a], "n_cell": int(n_cell[c]),
                         "share_called": float(taken[c, a] / n_cell[c]), "n_elig": ne,
                         "share_elig": ne / int(n_cell[c]), "n_taken_elig": nt,
                         "dr_mean": float(g[p, a].mean()) if ne else np.nan, "dr_lo": lo_d[c, a], "dr_hi": hi_d[c, a],
                         "obs_mean": float(y[p].mean()) if ne else np.nan,
                         "gain": float(gain[p, a].mean()) if ne else np.nan, "gain_lo": lo_g[c, a],
                         "gain_hi": hi_g[c, a],
                         "candidate": bool(ne / n_cell[c] >= CAND_SHARE and nt >= MIN_TAKEN)})
    return pd.DataFrame(rows)


def recommend(cells: pd.DataFrame) -> pd.DataFrame:
    """Per cell: the candidate with the highest gain; clear iff its CI lower bound is above 0."""
    rows = []
    for c, g in cells.groupby("cell", sort=False):
        cand = g[g["candidate"]]
        if cand.empty:
            rows.append({"cell": c, "n_cell": int(g["n_cell"].iloc[0]), "best": None, "gain": np.nan,
                         "gain_lo": np.nan, "gain_hi": np.nan, "dr_mean": np.nan, "obs_mean": np.nan,
                         "candidates": 0, "clear": False})
            continue
        b = cand.loc[cand["gain"].idxmax()]
        rows.append({"cell": c, "n_cell": int(b["n_cell"]), "best": b["action"], "gain": b["gain"],
                     "gain_lo": b["gain_lo"], "gain_hi": b["gain_hi"], "dr_mean": b["dr_mean"],
                     "obs_mean": b["obs_mean"], "candidates": int(len(cand)), "clear": bool(b["gain_lo"] > 0)})
    return pd.DataFrame(rows)


def apply_policy(df: pd.DataFrame, recs: pd.DataFrame, e: np.ndarray, thr: float = OVERLAP) -> np.ndarray:
    """Recommended action per play (index into ACTIONS), or -1 where the policy keeps the team's own call."""
    clear = recs[recs["clear"]].set_index("cell")["best"]
    best = df["cell"].map(clear).to_numpy(object)
    lut = {a: i for i, a in enumerate(ACTIONS)}
    a = np.array([lut.get(x, -1) if isinstance(x, str) else -1 for x in best], dtype=int)
    ok = a >= 0
    rows = np.flatnonzero(ok)
    ok[rows] = e[rows, a[rows]] >= thr
    return np.where(ok, a, -1)


def ope_diff(rec: np.ndarray, g: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-play DR value of the recommended policy minus the observed outcome (0 where it keeps the team's call)."""
    y = np.asarray(y, float)
    rows = np.arange(len(y))
    on = rec >= 0
    d = np.zeros(len(y))
    d[on] = g[rows[on], rec[on]] - y[on]
    return d


def observed_value(e: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Per-play DR value of the observed (propensity) policy: sum_a e_a G_a. Its mean should match mean(Y)."""
    return (e * g).sum(axis=1)


def shuffle_within_cells(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Placebo: permute the actions within (season, cell); outcomes and situations stay. Tendencies are rebuilt."""
    rng = np.random.default_rng(seed)
    a = df["action"].to_numpy(int).copy()
    for _, idx in df.groupby(["season", "cell"], sort=True).indices.items():
        a[idx] = a[idx][rng.permutation(len(idx))]
    return with_tendencies(df.assign(action=a))


# --------------------------------------------------------------------------- one target season

VARIANTS = {"main": {"thr": OVERLAP, "floor": None}, "thr10": {"thr": OVERLAP_ALT, "floor": None},
            "trim10": {"thr": OVERLAP, "floor": TRIM_FLOOR}}


def season_work(frame: pd.DataFrame, S: int, variants=tuple(VARIANTS), reps: int = REPS, seed: int = SEED) -> dict:
    """Everything for target season S, season-ahead: nuisances cross-fit on FIRST..S-1 (cells and
    recommendations), nuisances fit on all of FIRST..S-1 for S, and the OPE of S under each variant."""
    t0 = time.perf_counter()
    s = frame["season"].to_numpy(int)
    train = frame[(s >= FIRST) & (s < S)].reset_index(drop=True)
    test = frame[s == S].reset_index(drop=True)
    oof, fold_info = cross_fit(train)
    t1 = time.perf_counter()
    nu = fit_nuisance(train)
    pt = predict_nuisance(nu, test)
    t2 = time.perf_counter()
    a_tr, a_te = train["action"].to_numpy(int), test["action"].to_numpy(int)
    out = {"S": int(S), "test": test, "train_keys": train[["game_id", "play_id", "season", "cell", "action"]],
           "pred": pt, "oof": oof, "nuisance": nu.info, "folds": fold_info, "variants": {}}
    for v in variants:
        cfg = VARIANTS[v]
        g_tr = pseudo(oof["e"], oof["mu_epa"], a_tr, train["y_epa"], cfg["floor"])
        cells = cell_table(train, oof["e"], g_tr, train["y_epa"].to_numpy(float), cfg["thr"], reps, seed)
        recs = recommend(cells)
        g_te = pseudo(pt["e"], pt["mu_epa"], a_te, test["y_epa"], cfg["floor"])
        gs_te = pseudo(pt["e"], pt["mu_succ"], a_te, test["y_succ"], cfg["floor"])
        rec = apply_policy(test, recs, pt["e"], cfg["thr"])
        out["variants"][v] = {"cells": cells, "recs": recs, "rec": rec,
                              "d_epa": ope_diff(rec, g_te, test["y_epa"]),
                              "d_succ": ope_diff(rec, gs_te, test["y_succ"]),
                              "obs_epa": observed_value(pt["e"], g_te), "obs_succ": observed_value(pt["e"], gs_te),
                              "g_epa": g_te, "g_succ": gs_te}
    out["seconds"] = {"cross_fit": t1 - t0, "fit_predict": t2 - t1, "cells_ope": time.perf_counter() - t2}
    return out


def overlap_stats(e: np.ndarray, a_obs: np.ndarray, thr: float) -> dict:
    elig = e >= thr
    k = elig.sum(axis=1)
    rows = np.arange(len(a_obs))
    return {"thr": thr, "n": int(len(e)), "share_2plus": float((k >= 2).mean()), "mean_eligible": float(k.mean()),
            "share_pairs_eligible": float(elig.mean()), "share_observed_eligible": float(elig[rows, a_obs].mean()),
            "share_eligible_by_action": {a: float(elig[:, i].mean()) for i, a in enumerate(ACTIONS)}}
