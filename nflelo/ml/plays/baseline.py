"""M6 baseline: the historical yard distribution of the play's situation cell (CC BY-SA 4.0 when fit on M6 data).

Cells: down x distance bucket x field zone (x play type in the call-conditioned
view), from the training seasons. Each cell is smoothed toward its parent,

    p(cell) = (counts(cell) + k * p(parent)) / (n(cell) + k)

along down x distance x zone -> down x distance -> down -> all plays (the
root gets half a pseudo-count per yard value), with play type kept at every
level in the call view. Distributions are kept in raw yards (-10 or worse up
to 99, a touchdown counted at its yards) and folded onto the 53 bins of each
target play's own yard line, so a cell that spans the 6 to the 10 still
respects the goal line exactly. The turnover rate uses the same cells and
smoothing. `k` is chosen from `K_GRID` on the last training season.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data as pdata

R_MIN, R_MAX = -10, 99
NR = R_MAX - R_MIN + 1
DIST_EDGES = (1, 2, 3, 4, 7, 10, 11, 16)          # buckets: 1, 2, 3, 4-6, 7-9, 10, 11-15, 16+
ZONE_EDGES = (1, 6, 11, 21, 41, 61, 81, 91)       # yardline_100: 1-5, 6-10, 11-20, 21-40, 41-60, 61-80, 81-90, 91-99
K_GRID = (20.0, 80.0, 320.0, 1280.0, 5120.0)


def dist_bucket(ytg) -> np.ndarray:
    return np.searchsorted(np.asarray(DIST_EDGES), np.asarray(ytg, float), side="right") - 1


def zone_bucket(yl) -> np.ndarray:
    return np.searchsorted(np.asarray(ZONE_EDGES), np.asarray(yl, float), side="right") - 1


def raw_index(table: pd.DataFrame) -> np.ndarray:
    return (np.clip(np.round(table["yards"].to_numpy(float)), R_MIN, R_MAX) - R_MIN).astype(int)


def _fold_matrices() -> np.ndarray:
    """(100, NR, 53): raw yards -> bin for a play at yard line y (index y; 0 unused)."""
    M = np.zeros((100, NR, pdata.N_BINS))
    r = np.arange(R_MIN, R_MAX + 1)
    for y in range(1, 100):
        b = np.where(r >= y, pdata.TD_BIN, np.where(r <= -10, pdata.LOSS_BIN, np.where(r >= 41, pdata.BIG_BIN, r + 10)))
        low = pdata.LOSS_BIN if y <= 90 else y - 90
        b = np.where((r < y - 100) | ((b < low) & (b != pdata.TD_BIN)), low, b)
        M[y, np.arange(NR), b] = 1.0
    return M


FOLD = _fold_matrices()


def _keys(table: pd.DataFrame, call: bool) -> list[np.ndarray]:
    d = table["down"].to_numpy(int)
    pt_ = table["is_pass"].to_numpy(int) if call else np.zeros(len(table), int)
    db = dist_bucket(table["ydstogo"])
    zb = zone_bucket(table["yardline_100"])
    return [pt_, pt_ * 10 + d, (pt_ * 10 + d) * 10 + db, ((pt_ * 10 + d) * 10 + db) * 10 + zb]


@dataclass
class Baseline:
    call: bool
    k: float
    levels: list            # per level: dict key -> (raw distribution (NR,), turnover rate)

    @classmethod
    def fit(cls, table: pd.DataFrame, call: bool, k: float) -> "Baseline":
        r = raw_index(table)
        tov = table["turnover"].to_numpy(float)
        keys = _keys(table, call)
        root_c = np.bincount(r, minlength=NR).astype(float) + 0.5
        root = (root_c / root_c.sum(), (tov.sum() + 0.5) / (len(tov) + 1.0))
        levels = []
        prev = None
        for lvl, key in enumerate(keys):
            cur = {}
            codes, uniq = pd.factorize(key)
            C = len(uniq)
            cnt = np.zeros((C, NR))
            np.add.at(cnt, (codes, r), 1.0)
            n = cnt.sum(axis=1)
            t = np.bincount(codes, weights=tov, minlength=C)
            for i, u in enumerate(uniq):
                if lvl == 0:
                    pd_, pt_ = root
                else:
                    pk = u // 10
                    pd_, pt_ = prev.get(pk, (None, None))
                    if pd_ is None:
                        pd_, pt_ = root
                cur[int(u)] = ((cnt[i] + k * pd_) / (n[i] + k), (t[i] + k * pt_) / (n[i] + k))
            prev = cur
            levels.append(cur)
        return cls(call, float(k), [root] + levels)

    def _lookup(self, table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        keys = _keys(table, self.call)
        n = len(table)
        dist = np.empty((n, NR))
        tov = np.empty(n)
        root = self.levels[0]
        # deepest level available for each play, falling back to coarser cells (or the root)
        uniq, inv = np.unique(np.column_stack(keys), axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        for i, row in enumerate(uniq):
            val = root
            for lvl in range(len(keys)):
                v = self.levels[lvl + 1].get(int(row[lvl]))
                if v is None:
                    break
                val = v
            m = inv == i
            dist[m] = val[0]
            tov[m] = val[1]
        return dist, tov

    def predict(self, table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(n, 53) folded bin probabilities and the turnover probability."""
        dist, tov = self._lookup(table)
        yl = table["yardline_100"].to_numpy(int)
        p = np.empty((len(table), pdata.N_BINS))
        for y in np.unique(yl):
            m = yl == y
            p[m] = dist[m] @ FOLD[y]
        return pdata.fold(p, yl), tov


def tune_k(train: pd.DataFrame, val: pd.DataFrame, call: bool, grid=K_GRID) -> tuple[float, dict]:
    """Smoothing strength with the lowest validation CRPS."""
    from .metrics import crps
    res = {}
    for k in grid:
        p, _ = Baseline.fit(train, call, k).predict(val)
        res[float(k)] = float(crps(p, val["bin"].to_numpy(int), val["yardline_100"].to_numpy(float)).mean())
    return min(res, key=res.get), res
