# Tests for Project 7.   Run:  ..\.venv\Scripts\python -m pytest -q
import numpy as np                                # arrays
import pytest                                     # test runner
from sklearn.ensemble import AdaBoostClassifier   # reference implementation
from sklearn.metrics import roc_auc_score         # metric
from sklearn.tree import DecisionTreeClassifier   # stumps

import adaboost as ab                             # the project code


@pytest.fixture(scope="module")
def data():
    df = ab.load(ab.SAMPLE_DATA)
    X, y = df[ab.LOW + ab.HIGH], df["signal"].values.astype(int)
    return X.iloc[:15000], y[:15000], X.iloc[15000:], y[15000:]


def test_load_has_28_features_and_binary_label(data):
    Xtr, ytr, _, _ = data
    assert Xtr.shape[1] == 28 and set(np.unique(ytr)) == {0, 1}


def test_scratch_adaboost_matches_sklearn(data):
    Xtr, ytr, Xte, _ = data
    ours = ab.ScratchAdaBoost(n_rounds=30).fit(Xtr, ytr)
    ref = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=30, learning_rate=1.0,
                             random_state=0).fit(Xtr, ytr)
    assert np.allclose(ours.alphas, ref.estimator_weights_[:len(ours.alphas)])   # same votes, round by round
    assert np.mean(ours.predict(Xte) == ref.predict(Xte)) > 0.999                # same decisions


def test_weights_stay_a_distribution_and_mistakes_get_heavier(data):
    Xtr, ytr, _, _ = data
    m = ab.ScratchAdaBoost(n_rounds=1).fit(Xtr, ytr)
    w = m.weights_history[0]
    wrong = m.stumps[0].predict(np.asarray(Xtr)) != ytr
    assert np.isclose(w.sum(), 1.0) and w[wrong].min() > w[~wrong].max()   # misclassified points weigh more
    assert np.isclose(w[wrong].sum(), 0.5)        # classic AdaBoost fact: after reweighting, errors hold half the weight


def test_boosting_beats_one_stump(data):
    Xtr, ytr, Xte, yte = data
    one = roc_auc_score(yte, ab.ScratchAdaBoost(n_rounds=1).fit(Xtr, ytr).decision_function(Xte))
    many = roc_auc_score(yte, ab.ScratchAdaBoost(n_rounds=100).fit(Xtr, ytr).decision_function(Xte))
    assert many > one + 0.1                       # many weak learners -> one strong learner
