"""M6 gradient-boosted multiclass model (scikit-learn HistGradientBoosting). CC BY-SA 4.0 when fit on M6 data.

One multiclass model over the 53 yard bins and one binary model for the
turnover flag, both on the view's allowlisted features (`data.FEATURES`).
The bin probabilities are folded onto the bins possible from each play's yard
line (`data.fold`). Missing values (temperature and wind indoors) are handled
natively by the trees.

Tuning (spec section 5): a fixed small grid. For a target season S, each
setting is fit on the training seasons before S-1 with early stopping on
season S-1 (validation log loss); the setting with the lowest validation CRPS
wins, and the final model is refit on every training season with that
setting and its early-stopped number of iterations.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import data as pdata
from .metrics import crps

GRID = ({"learning_rate": 0.1, "max_leaf_nodes": 7}, {"learning_rate": 0.1, "max_leaf_nodes": 15},
        {"learning_rate": 0.05, "max_leaf_nodes": 7}, {"learning_rate": 0.05, "max_leaf_nodes": 15})
FIXED = {"min_samples_leaf": 200, "l2_regularization": 1.0, "max_bins": 255}
TOV_PARAMS = {"learning_rate": 0.05, "max_leaf_nodes": 7, "min_samples_leaf": 200, "l2_regularization": 1.0}
MAX_ITER = 400
SEED = 20261003


@dataclass
class GBMModel:
    view: str
    params: dict
    bins: HistGradientBoostingClassifier
    tov: HistGradientBoostingClassifier
    info: dict = field(default_factory=dict)

    def predict(self, table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        feats = pdata.FEATURES[self.view]
        X = table[feats]
        P = np.zeros((len(table), pdata.N_BINS))
        P[:, self.bins.classes_.astype(int)] = self.bins.predict_proba(X)
        return pdata.fold(P, table["yardline_100"].to_numpy(float)), self.tov.predict_proba(X)[:, 1]


def _clf(params: dict, max_iter: int, early: bool) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=max_iter, early_stopping=early, n_iter_no_change=10,
                                          random_state=SEED, **params)


def fit(train: pd.DataFrame, view: str, params: dict, val: pd.DataFrame | None = None,
        n_iter: int | None = None, tov_iter: int | None = None) -> GBMModel:
    feats = pdata.check_features(pdata.FEATURES[view])
    t0 = time.perf_counter()
    p = {**FIXED, **params}
    if val is not None:
        vb = val[val["bin"].isin(np.unique(train["bin"]))]   # early-stopping loss needs bins seen in training
        b = _clf(p, MAX_ITER, True).fit(train[feats], train["bin"], X_val=vb[feats], y_val=vb["bin"])
        t = _clf(TOV_PARAMS, MAX_ITER, True).fit(train[feats], train["turnover"].astype(int),
                                                 X_val=val[feats], y_val=val["turnover"].astype(int))
    else:
        b = _clf(p, int(n_iter), False).fit(train[feats], train["bin"])
        t = _clf(TOV_PARAMS, int(tov_iter), False).fit(train[feats], train["turnover"].astype(int))
    return GBMModel(view, params, b, t, {"n_iter": int(b.n_iter_), "tov_iter": int(t.n_iter_),
                                         "seconds": time.perf_counter() - t0, "n_train": int(len(train))})


def fit_season_ahead(train: pd.DataFrame, view: str, grid=GRID) -> GBMModel:
    """Grid on the last training season (fit on the earlier ones), refit the winner on all training seasons."""
    last = int(train["season"].max())
    inner, val = train[train["season"] < last], train[train["season"] == last]
    t0 = time.perf_counter()
    scores, fits = [], []
    for params in grid:
        m = fit(inner, view, params, val=val)
        p, _ = m.predict(val)
        scores.append(float(crps(p, val["bin"].to_numpy(int), val["yardline_100"].to_numpy(float)).mean()))
        fits.append(m)
    k = int(np.argmin(scores))
    best = fits[k]
    final = fit(train, view, grid[k], n_iter=best.info["n_iter"], tov_iter=best.info["tov_iter"])
    final.info.update({"grid": [dict(g) for g in grid], "val_crps": scores, "chosen": dict(grid[k]),
                       "total_seconds": time.perf_counter() - t0})
    return final
