# Tests for Project 3.   Run:  ..\.venv\Scripts\python -m pytest -q
import numpy as np                                # arrays
from scipy import sparse                          # sparse matrices
from sklearn.naive_bayes import MultinomialNB     # reference implementation

import naive_bayes as nb                          # the project code


def test_load_builds_text_and_target():
    df = nb.load_emails(nb.SAMPLE_DATA)
    assert len(df) > 1900 and set(df["spam"]) == {0, 1}           # sample minus duplicates, binary target
    assert not df["text"].duplicated().any() and df["mailbox"].nunique() == 6   # deduplicated, all mailboxes


def test_scratch_nb_matches_sklearn_exactly():
    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.poisson(0.3, size=(200, 40)))       # random word counts
    y = (rng.random(200) < 0.4).astype(int)
    ours = nb.ScratchMultinomialNB(alpha=0.5).fit(X, y)
    ref = MultinomialNB(alpha=0.5).fit(X, y)
    assert np.allclose(ours.feature_log_prob_, ref.feature_log_prob_)   # same log P(word | class)
    assert np.allclose(ours.predict_proba(X), ref.predict_proba(X))     # same probabilities
    assert (ours.predict(X) == ref.predict(X)).all()


def test_log_space_avoids_underflow():
    X = sparse.csr_matrix(np.full((1, 5), 400))   # a "long e-mail": 2,000 words -> product of probabilities = 0.0
    y_fit = sparse.csr_matrix(np.array([[5, 1, 1, 1, 1], [1, 1, 1, 1, 5]]))
    m = nb.ScratchMultinomialNB().fit(y_fit, np.array([0, 1]))
    p = m.predict_proba(X)
    assert np.isfinite(p).all() and np.isclose(p.sum(), 1.0)      # still a valid probability


def test_models_work_on_the_sample():
    df = nb.load_emails(nb.SAMPLE_DATA)
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(df, test_size=0.25, stratify=df["spam"], random_state=0)
    res, *_ = nb.compare_models(tr, te)
    assert res["MultinomialNB (counts)"]["accuracy"] > 0.9        # text NB works
    assert res["GaussianNB (2k counts)"]["accuracy"] < res["MultinomialNB (counts)"]["accuracy"]   # wrong distribution
