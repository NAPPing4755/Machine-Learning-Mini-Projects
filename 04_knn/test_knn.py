# Tests for Project 4.   Run:  ..\.venv\Scripts\python -m pytest -q
import numpy as np                                # arrays
from sklearn.neighbors import KNeighborsClassifier   # reference implementation

import knn                                        # the project code


def test_load_sample():
    X, y = knn.load_data(knn.SAMPLE_DATA)
    assert X.shape == (5000, 54) and set(np.unique(y)) <= set(range(1, 8))   # 54 features, classes 1..7


def test_pairwise_distances_match_brute_force():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(5, 3)), rng.normal(size=(7, 3))
    slow = np.array([[((a - b) ** 2).sum() for b in B] for a in A])   # the definition, with loops
    assert np.allclose(knn.pairwise_sq_dists(A, B), slow)


def test_scratch_knn_matches_sklearn_exactly_on_continuous_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 6)); y = rng.integers(1, 5, size=1500)   # no two distances are exactly equal
    ours = knn.ScratchKNN(k=5).fit(X[:1000], y[:1000]).predict(X[1000:])
    ref = KNeighborsClassifier(5).fit(X[:1000], y[:1000]).predict(X[1000:])
    assert (ours == ref).all()                    # same neighbours, same vote, same tie rule


def test_scratch_knn_on_covertype_differs_only_by_distance_ties():
    X, y = knn.load_data(knn.SAMPLE_DATA)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)       # scaled, as in the project
    ours = knn.ScratchKNN(k=5).fit(X[:4000], y[:4000]).predict(X[4000:])
    ref = KNeighborsClassifier(5).fit(X[:4000], y[:4000]).predict(X[4000:])
    assert np.mean(ours == ref) > 0.95            # 0/1 soil columns create exact distance ties at the k-th place


def test_scaling_helps_a_lot():
    X, y = knn.load_data(knn.SAMPLE_DATA)
    res, _ = knn.scaling_effect(X[:4000], y[:4000], X[4000:], y[4000:])
    assert res["scaled"] > res["raw"]             # raw metres/degrees mix: elevation dominates the distance


def test_distances_concentrate_in_high_dimensions():
    c = knn.distance_concentration(dims=(2, 1000), n=500)
    assert c[2] > 10 and c[1000] < 1.5            # in 1000-D the farthest point is barely farther than the nearest
