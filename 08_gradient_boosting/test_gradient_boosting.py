# Tests for Project 8.   Run:  ..\.venv\Scripts\python -m pytest -q
import numpy as np                                # arrays
import pandas as pd                               # tables
import pytest                                     # test runner
from sklearn.ensemble import GradientBoostingRegressor   # reference implementation

import gradient_boosting as gb                    # the project code


@pytest.fixture(scope="module")
def df():
    raw = pd.read_csv(gb.SAMPLE_DATA, parse_dates=["Date"], low_memory=False)
    return gb.add_features(raw)


def test_features_keep_only_open_days_and_add_calendar(df):
    assert (df["Sales"] > 0).all() and (df["Open"] == 1).all()        # closed days removed
    train, hold = gb.time_split(df)                               # store averages are added at the split
    assert set(gb.FEATURES) <= set(hold.columns) and "Customers" not in gb.FEATURES   # no leak in the feature list
    assert df["promo2_active"].isin([0, 1]).all() and df["promo2_active"].sum() > 0


def test_time_split_holds_out_the_last_six_weeks(df):
    train, hold = gb.time_split(df)
    assert train["Date"].max() < hold["Date"].min()
    assert (hold["Date"].max() - hold["Date"].min()).days <= gb.HOLDOUT_DAYS


def test_rmspe_by_hand():
    assert np.isclose(gb.rmspe([100, 200], [110, 180]), np.sqrt((0.1 ** 2 + 0.1 ** 2) / 2))   # both 10% off


def test_scratch_boosting_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.random((500, 4)); y = np.sin(6 * X[:, 0]) + X[:, 1] ** 2 + rng.normal(0, 0.1, 500)
    ours = gb.ScratchGradientBoosting(n_trees=40, learning_rate=0.1, max_depth=3).fit(X, y)
    ref = GradientBoostingRegressor(n_estimators=40, learning_rate=0.1, max_depth=3, random_state=0).fit(X, y)
    assert np.allclose(ours.predict(X), ref.predict(X))   # identical: same init, same residual trees
    assert ours.train_loss[-1] < ours.train_loss[0]       # each tree reduces the error


def test_model_beats_the_naive_baseline(df):
    train, hold = gb.time_split(df)
    base = gb.baselines(train, hold)["store_weekday_promo_median"]
    m = gb.hgb(lr=0.1, iters=500).fit(train[gb.FEATURES], train["log_sales"])
    assert gb.rmspe(hold["Sales"], np.expm1(m.predict(hold[gb.FEATURES]))) < base
