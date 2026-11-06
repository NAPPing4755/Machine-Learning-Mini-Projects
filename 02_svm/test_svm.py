# Tests for Project 2.   Run:  ..\.venv\Scripts\python -m pytest -q
import numpy as np                                # arrays
import pytest                                     # test runner
from sklearn.svm import SVC                       # reference implementation

import svm                                        # the project code


@pytest.fixture(scope="module")
def data():
    return svm.load_data(svm.SAMPLE_DATA)         # 3,000 train + 1,000 test images


def test_load_scales_pixels_and_keeps_shapes(data):
    Xtr, ytr, Xte, yte = data
    assert Xtr.shape == (3000, 784) and Xte.shape == (1000, 784)      # 28x28 images, flattened
    assert 0.0 <= Xtr.min() and Xtr.max() <= 1.0                       # scaled to [0, 1]
    assert set(np.unique(ytr)) == set(range(10))                       # all ten classes present


def _two_blobs(n=300, seed=0):
    rng = np.random.default_rng(seed)             # reproducible random data
    X = np.vstack([rng.normal(-1, 1, (n, 2)), rng.normal(1.5, 1, (n, 2))])   # two overlapping clouds
    y = np.r_[-np.ones(n), np.ones(n)]            # labels -1 and +1 (the SVM convention)
    return X, y


def test_hinge_objective_by_hand():
    X = np.array([[1.0, 0.0], [0.0, 1.0]]); y = np.array([1.0, -1.0]); w = np.array([2.0, 0.0])
    # margins: 1*2 = 2 (no loss), -1*0 = 0 (loss 1) -> mean hinge 0.5; penalty 0.5*0.1*4 = 0.2
    assert np.isclose(svm.hinge_objective(w, 0.0, X, y, lam=0.1), 0.7)


def test_from_scratch_svm_reaches_sklearns_objective():
    X, y = _two_blobs()
    lam = 0.01                                    # our regularisation strength
    w, b, hist = svm.fit_linear_svm(X, y, lam=lam, epochs=3000)
    ref = SVC(kernel="linear", C=1 / (lam * len(y))).fit(X, y)   # the same objective in sklearn's units
    ours = svm.hinge_objective(w, b, X, y, lam)
    theirs = svm.hinge_objective(ref.coef_[0], ref.intercept_[0], X, y, lam)
    assert ours <= theirs * 1.01                  # within 1% of the exact optimum
    agree = np.mean(np.sign(X @ w + b) == ref.predict(X))
    assert agree > 0.98                           # nearly identical decisions
    assert hist[-1] < hist[0]                     # it learned


def test_linear_svm_beats_chance_on_images(data):
    Xtr, ytr, Xte, yte = data
    _, acc, _ = svm.linear_svm(Xtr, ytr, Xte, yte)
    assert acc > 0.75                             # 10 classes: chance is 0.10


def test_tuning_picks_from_the_grid_and_shows_overfitting(data):
    Xtr, ytr, _, _ = data
    best, grid, train_acc = svm.tune_rbf(Xtr, ytr, Cs=(1, 100), gammas=(0.003, 3.0), n_fit=1500, n_val=1000)
    assert best["C"] in (1, 100) and best["gamma"] in (0.003, 3.0)
    assert train_acc[1, 1] > 0.99 and grid[1, 1] < 0.5   # huge gamma: memorises training, fails validation
