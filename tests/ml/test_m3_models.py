"""M3 logistic model: tie handling, season weights, walk-forward discipline, and A0 reproducing Elo."""
import numpy as np
import pandas as pd
import pytest

from nflelo.ml.models import logistic


def test_ties_become_two_half_rows():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 0.5, 0.0])
    w = np.array([1.0, 0.8, 1.0])
    X2, y2, w2 = logistic.expand_ties(X, y, w)
    assert len(y2) == 4 and sorted(y2.tolist()) == [0, 0, 1, 1]
    assert w2.sum() == pytest.approx(w.sum())
    assert sorted(w2[X2[:, 0] == 2.0].tolist()) == [0.4, 0.4]


def test_season_weights_half_life():
    w = logistic.season_weights([2018, 2010, 2002], 2019, 8.0)
    np.testing.assert_allclose(w, [1.0, 0.5, 0.25])


def _synthetic(seasons=range(2001, 2020), per=260, seed=0, slope=1.0, icpt=0.0):
    rng = np.random.default_rng(seed)
    s = np.repeat(np.arange(seasons.start, seasons.stop), per)
    x = rng.normal(0, 0.8, len(s))
    p = 1 / (1 + np.exp(-(icpt + slope * x)))
    y = (rng.uniform(size=len(s)) < p).astype(float)
    y[rng.uniform(size=len(s)) < 0.004] = 0.5
    return pd.DataFrame({"season": s, "elo_logit": x, "y": y, "p_true": p})


def test_a0_on_elo_logit_reproduces_elo():
    """If the outcomes follow Elo's own probabilities, logistic on elo_logit gives back Elo (slope 1, intercept 0)."""
    df = _synthetic(per=2000)
    m = logistic.fit_logistic(df[["elo_logit"]].to_numpy(), df["y"].to_numpy(), np.ones(len(df)))
    assert m.coef_[0, 0] == pytest.approx(1.0, abs=0.05)
    assert m.intercept_[0] == pytest.approx(0.0, abs=0.03)
    pred, coefs = logistic.walk_forward(df, ["elo_logit"], (2006, 2019))
    dev = df["season"].between(2006, 2019)
    brier_model = np.mean((df.loc[dev, "y"] - pred[dev]) ** 2)
    brier_elo = np.mean((df.loc[dev, "y"] - df.loc[dev, "p_true"]) ** 2)
    assert abs(brier_model - brier_elo) < 0.001
    assert list(coefs["season"]) == list(range(2006, 2020)) and pred[~dev].isna().all()


def test_walk_forward_never_sees_the_test_season_or_later():
    df = _synthetic()
    pred, _ = logistic.walk_forward(df, ["elo_logit"], (2010, 2012))
    df2 = df.copy()
    df2.loc[df2["season"] >= 2012, "y"] = 1 - df2.loc[df2["season"] >= 2012, "y"]
    pred2, _ = logistic.walk_forward(df2, ["elo_logit"], (2010, 2012))
    m = df["season"] == 2012
    np.testing.assert_allclose(pred[m], pred2[m])        # 2012 is fit on 2001-2011 only
    df3 = df.copy()
    df3.loc[df3["season"] == 2011, "y"] = 1 - df3.loc[df3["season"] == 2011, "y"]
    pred3, _ = logistic.walk_forward(df3, ["elo_logit"], (2010, 2012))
    assert not np.allclose(pred[m], pred3[m])            # ... and 2011 does matter


def test_walk_forward_rejects_nan_features():
    df = _synthetic()
    df.loc[5, "elo_logit"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        logistic.walk_forward(df, ["elo_logit"], (2010, 2010))
