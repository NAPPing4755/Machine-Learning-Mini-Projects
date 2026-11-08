# Tests for Project 5.   Run:  ..\.venv\Scripts\python -m pytest -q
import numpy as np                                # arrays
import pytest                                     # test runner
from sklearn.tree import DecisionTreeRegressor    # reference implementation

import decision_trees as dt                       # the project code


@pytest.fixture(scope="module")
def trips():
    return dt.clean(dt.load_trips(dt.SAMPLE_DATA))   # (clean_df, removed)


def test_cleaning_removes_impossible_trips_and_reports_why(trips):
    df, removed = trips
    assert df["duration_min"].between(1, 180).all() and df["fare_amount"].between(3, 500).all()
    assert (df["trip_distance"] / (df["duration_min"] / 60) <= 80).all()   # no 300-mph taxis
    assert sum(removed.values()) > 0 and set(removed) >= {"fare not $3-$500", "duration not 1-180 minutes"}


def test_time_split_has_no_overlap(trips):
    train, test = dt.time_split(trips[0])
    assert train["tpep_pickup_datetime"].max() < test["tpep_pickup_datetime"].min()


def test_best_split_finds_the_obvious_cut():
    X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]]); y = np.array([1, 1, 1, 9, 9, 9.0])
    j, t, gain = dt.best_split(X, y)
    assert j == 0 and t == 6.5 and np.isclose(gain, ((y - y.mean()) ** 2).sum())   # removes ALL the error


def test_scratch_tree_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.random((400, 4)); y = 3 * (X[:, 0] > 0.5) + X[:, 1] ** 2 + rng.normal(0, 0.1, 400)
    ours = dt.ScratchRegressionTree(max_depth=4).fit(X, y).predict(X)
    ref = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y).predict(X)
    assert np.allclose(ours, ref)                 # same splits, same leaf means


def test_trees_beat_the_baselines_on_the_sample(trips):
    train, test = dt.time_split(trips[0])
    base = dt.baselines(train, test)
    m = DecisionTreeRegressor(max_depth=8, random_state=0).fit(train[dt.FEATURES], train["duration_min"])
    from sklearn.metrics import root_mean_squared_error
    rmse = root_mean_squared_error(test["duration_min"], m.predict(test[dt.FEATURES]))
    assert rmse < base["predict_mean"]["rmse"]    # a tree must beat "always predict the average"
