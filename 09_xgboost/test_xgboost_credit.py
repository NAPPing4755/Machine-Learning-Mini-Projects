# Tests for Project 9.   Run once:  ..\.venv\Scripts\python make_sample.py      then:  ..\.venv\Scripts\python -m pytest -q
import numpy as np                                # arrays
import pandas as pd                               # tables
import pytest                                     # test runner
import xgboost as xgb                             # reference implementation

import xgboost_credit as xc                       # the project code


def test_employment_sentinel_is_turned_into_a_flag():
    app = pd.DataFrame({"DAYS_EMPLOYED": [-1000, 365243], "AMT_CREDIT": [1.0, 1.0], "AMT_INCOME_TOTAL": [1.0, 1.0],
                        "AMT_ANNUITY": [1.0, 1.0], "AMT_GOODS_PRICE": [1.0, 1.0], "EXT_SOURCE_1": [0.5, 0.5],
                        "EXT_SOURCE_2": [0.5, 0.5], "EXT_SOURCE_3": [0.5, 0.5], "DAYS_BIRTH": [-10000, -20000]})
    a = xc.clean_application(app)
    assert a["DAYS_EMPLOYED_SENTINEL"].tolist() == [0, 1] and np.isnan(a["DAYS_EMPLOYED"].iloc[1])   # code, not 1,000 years
    assert a["DAYS_EMPLOYED"].iloc[0] == -1000    # real values untouched


def test_aggregate_gives_one_row_per_applicant():
    df = pd.DataFrame({"SK_ID_CURR": [1, 1, 2], "AMT": [10.0, 30.0, 5.0]})
    agg = xc.aggregate(df, "SK_ID_CURR", "X")
    assert agg.loc[1, "X_AMT_mean"] == 20 and agg.loc[1, "X_AMT_max"] == 30 and agg.loc[2, "X_COUNT"] == 1


def test_gain_formula_by_hand():
    # one leaf with G=4, H=4 split into (G=4,H=2) and (G=0,H=2), lambda=0: 0.5*(16/2 + 0 - 16/4) = 2
    assert np.isclose(xc.split_gain(4, 2, 0, 2, lam=0.0), 2.0)
    assert np.isclose(xc.leaf_weight(4, 3, lam=1.0), -1.0)        # -G/(H+lambda)


def test_scratch_stumps_match_xgboost():
    rng = np.random.default_rng(0)
    X = rng.random((400, 3)); y = (X[:, 0] + 0.3 * rng.normal(size=400) > 0.5).astype(float)
    ours = xc.ScratchXGBStumps(n_rounds=8, eta=0.3, lam=1.0).fit(X, y)
    ref = xgb.train({"objective": "binary:logistic", "max_depth": 1, "eta": 0.3, "lambda": 1.0, "min_child_weight": 0,
                     "tree_method": "exact", "base_score": 0.5}, xgb.DMatrix(X, y), 8)
    assert np.allclose(ours.predict_proba(X), ref.predict(xgb.DMatrix(X)), atol=1e-5)   # same splits, same leaves


def test_full_pipeline_on_the_sample():
    df = xc.build(xc.SAMPLE_DIR)
    assert df["SK_ID_CURR"].is_unique and df.shape[1] > 300         # one row per applicant, history features added
    X, y = df.drop(columns=["TARGET", "SK_ID_CURR"]), df["TARGET"]
    b, it = xc.fit_xgb(X.iloc[:7000], y.iloc[:7000], X.iloc[7000:8500], y.iloc[7000:8500], rounds=300)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y.iloc[8500:], xc.predict(b, X.iloc[8500:])) > 0.65   # learns something real
