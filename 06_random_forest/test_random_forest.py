# Tests for Project 6.   Run:  ..\.venv\Scripts\python -m pytest -q
import numpy as np                                # arrays
import pytest                                     # test runner
from sklearn.metrics import average_precision_score   # rare-class metric
from sklearn.tree import DecisionTreeClassifier   # single-tree baseline

import random_forest as rf                        # the project code


@pytest.fixture(scope="module")
def split():
    df = rf.load(rf.SAMPLE_DATA)
    tr, te = rf.time_split(df)
    return tr[rf.FEATURES], tr["Class"].values, te[rf.FEATURES], te["Class"].values, te["Amount"].values


def test_load_dedupes_and_sorts_by_time():
    df = rf.load(rf.SAMPLE_DATA)
    assert not df.duplicated().any() and df["Time"].is_monotonic_increasing


def test_time_split_trains_on_the_past(split):
    df = rf.load(rf.SAMPLE_DATA)
    tr, te = rf.time_split(df)
    assert tr["Time"].max() <= te["Time"].min() and tr["Class"].sum() > 0 and te["Class"].sum() > 0


def test_scratch_forest_beats_a_single_tree_and_oob_is_sane(split):
    Xtr, ytr, Xte, yte, amt = split
    forest = rf.ScratchRandomForest(n_trees=30).fit(Xtr, ytr)
    tree = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
    ap_forest = average_precision_score(yte, forest.predict_proba(Xte))
    ap_tree = average_precision_score(yte, tree.predict_proba(Xte)[:, 1])
    assert ap_forest > ap_tree                    # averaging many noisy trees beats one
    assert 0.3 < average_precision_score(ytr, forest.oob_proba()) <= 1.0   # out-of-bag score is a real estimate


def test_metrics_budget_logic():
    y = np.array([0] * 990 + [1] * 10); p = np.r_[np.zeros(990), np.ones(10)]   # a perfect ranking
    amt = np.ones(1000)
    m = rf.metrics(y, p, amt)
    assert m["precision_at_0.1pct"] == 1.0 and m["recall_at_0.1pct"] == 0.1   # 1 review = 1 of 10 frauds


def test_noise_column_fools_impurity_importance_more(split):
    Xtr, ytr, Xte, yte, _ = split
    res, _, _ = rf.importance_trap(Xtr, ytr, Xte, yte)
    assert res["mdi_rank_of_noise"] < res["permutation_rank_of_noise"]   # MDI ranks pure noise higher
