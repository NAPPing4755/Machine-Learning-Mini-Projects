# Tests for Project 10.   Run once:  ..\.venv\Scripts\python make_sample.py      then:  ..\.venv\Scripts\python -m pytest -q
import numpy as np                                # arrays
import pandas as pd                               # tables
import pytest                                     # test runner

import capstone as cp                             # the project code


@pytest.fixture(scope="module")
def raw():
    return cp.load(cp.SAMPLE_DIR)


def test_load_joins_identity_and_encodes_text(raw):
    assert len(raw) == 30000 and "DeviceType" in raw.columns        # identity columns attached
    assert raw["TransactionDT"].is_monotonic_increasing             # sorted by time
    assert all(np.issubdtype(t, np.number) for t in raw.dtypes)     # every column numeric (text -> codes)


def test_uid_features_identify_repeat_clients(raw):
    fe = cp.add_features(raw)
    assert (fe["uid_count"] >= 1).all() and fe["uid_count"].max() > 5   # some clients have many transactions
    one = pd.DataFrame({"TransactionDT": [86400 * 10, 86400 * 12], "D1": [3.0, 5.0], "card1": [1.0, 1.0],
                        "addr1": [2.0, 2.0], "TransactionAmt": [10.0, 30.0], "P_emaildomain": [1.0, 1.0]})
    out = cp.add_features(one)
    assert out["card_start_day"].nunique() == 1 and out["uid_count"].tolist() == [2, 2]   # same person, 2 days apart


def test_time_split_and_feature_list(raw):
    tr, te = cp.time_split(raw)
    assert tr["TransactionDT"].max() <= te["TransactionDT"].min()
    assert "isFraud" not in cp.features(raw) and "TransactionID" not in cp.features(raw)


def test_lineup_runs_on_the_sample(raw):
    fe = cp.add_features(raw)
    tr, te = cp.time_split(fe)
    rows = cp.compare_all(tr, te, cp.features(fe)[:60])           # a slice of features keeps the test fast
    assert len(rows) == 9 and all(0.4 < r["roc_auc"] <= 1.0 for r in rows)
