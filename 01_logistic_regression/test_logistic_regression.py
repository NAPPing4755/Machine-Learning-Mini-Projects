# Tests for Project 1. They check YOUR functions in logistic_regression.py.
# Run one step at a time, e.g.:   ..\.venv\Scripts\python -m pytest -q -k step1
# A step is done when all of its tests pass. Read a failing test to see exactly what is expected.

import warnings                                   # to turn numpy overflow warnings into test failures

import numpy as np                                # arrays and maths
import pandas as pd                               # tables (DataFrames)
import pytest                                     # the test runner

import logistic_regression as lr                  # YOUR code: the file you are writing

SAMPLE = lr.SAMPLE_DATA                           # the 20,000-row sample in data/ (fast to load)


@pytest.fixture(scope="module")                   # load the sample once and share it between tests
def raw():
    return lr.load_loans(SAMPLE)                  # uses your step 1 function


# ------------------------------------------------------------------ STEP 1: load the data and build the target
def test_step1_load_keeps_only_the_listed_columns(raw):
    assert list(raw.columns) == lr.RAW_COLUMNS                          # exactly the columns we asked for
    assert len(raw) == 20000                                             # every sample row was read


def test_step1_issue_date_is_a_real_date(raw):
    assert pd.api.types.is_datetime64_any_dtype(raw["issue_d"])          # "Dec-2015" text became a date
    assert raw["issue_d"].dt.year.between(2011, 2016).all()              # the sample covers 2011-2016


def test_step1_nrows_limits_the_rows_read():
    assert len(lr.load_loans(SAMPLE, nrows=100)) == 100                  # nrows works (handy on the full file)


def test_step1_target_keeps_only_finished_36_month_loans(raw):
    df = lr.make_target(raw)
    assert set(df["loan_status"]) == {"Fully Paid", "Charged Off"}      # unfinished loans are gone
    assert set(df["term"].str.strip()) == {"36 months"}                 # only 36-month loans
    assert set(df["default"]) == {0, 1}                                 # a 0/1 target column exists
    assert (df["default"] == (df["loan_status"] == "Charged Off")).all()  # 1 exactly when charged off
    assert 0.10 < df["default"].mean() < 0.20                           # realistic default rate (about 14%)
    assert len(raw) == 20000                                            # the input DataFrame was not modified


# ------------------------------------------------------------------ STEP 2: clean features and split by time
def test_step2_emp_length_becomes_years():
    s = pd.Series(["10+ years", "< 1 year", "1 year", "5 years", np.nan])
    out = lr.clean_emp_length(s)
    assert out.iloc[:4].tolist() == [10.0, 0.0, 1.0, 5.0]               # text -> number of years
    assert np.isnan(out.iloc[4])                                        # missing stays missing (imputed later)


def test_step2_credit_history_in_years():
    issue = pd.to_datetime(pd.Series(["Dec-2015", "Jan-2014"]), format="%b-%Y")
    first = pd.Series(["Dec-2005", "Jan-2013"])
    out = lr.credit_history_years(issue, first)
    assert out.round(1).tolist() == [10.0, 1.0]                         # 10 years and 1 year of credit history


def test_step2_add_features_adds_columns_without_touching_input(raw):
    df = lr.make_target(raw)
    out = lr.add_features(df)
    assert {"emp_years", "credit_history_years"} <= set(out.columns)    # new columns exist
    assert "emp_years" not in df.columns                                # the input was not modified
    assert out["credit_history_years"].min() >= 0                       # nobody's history starts after the loan


def test_step2_time_split_trains_on_the_past_and_tests_on_the_future(raw):
    df = lr.add_features(lr.make_target(raw))
    train, test = lr.time_split(df)
    assert set(train["issue_d"].dt.year) == set(lr.TRAIN_YEARS)        # 2012-2014 only
    assert set(test["issue_d"].dt.year) == set(lr.TEST_YEARS)           # 2015 only
    assert train["issue_d"].max() < test["issue_d"].min()               # no overlap: the past predicts the future


# ------------------------------------------------------------------ STEP 3: the preprocessing pipeline
def test_step3_preprocessor_turns_the_table_into_numbers(raw):
    train, test = lr.time_split(lr.add_features(lr.make_target(raw)))
    prep = lr.build_preprocessor()
    Xtr = prep.fit_transform(train)                                     # learn medians/means/categories on TRAIN
    Xte = prep.transform(test)                                          # only apply them to TEST
    Xtr = Xtr.toarray() if hasattr(Xtr, "toarray") else Xtr            # one-hot output may be sparse
    Xte = Xte.toarray() if hasattr(Xte, "toarray") else Xte
    assert Xtr.shape[1] == Xte.shape[1]                                 # same columns for train and test
    assert not np.isnan(Xtr).any() and not np.isnan(Xte).any()          # every missing value was imputed
    n_plain = len(lr.NUMERIC_FEATURES)                                  # scaled numeric columns come first
    assert np.allclose(Xtr[:, :n_plain].mean(axis=0), 0, atol=1e-6)     # standardised: mean 0 on train
    assert Xtr.shape[1] > n_plain + 10                                  # one-hot added many category columns


def test_step3_no_leaky_columns_are_used():
    used = set(lr.NUMERIC_FEATURES) | set(lr.CATEGORICAL_FEATURES)
    assert not used & set(lr.LEAKY_COLUMNS)                             # nothing known only AFTER the loan


# ------------------------------------------------------------------ STEP 4: logistic regression from scratch
def test_step4_sigmoid_values_and_no_overflow():
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                  # any overflow warning fails the test
        out = lr.sigmoid(np.array([-1000.0, -2.0, 0.0, 2.0, 1000.0]))
    assert np.allclose(out, [0.0, 0.11920292, 0.5, 0.88079708, 1.0])


def test_step4_log_loss_matches_sklearn():
    from sklearn.metrics import log_loss as sk
    y = np.array([0, 1, 1, 0, 1]); p = np.array([0.1, 0.8, 0.6, 0.3, 0.99])
    assert np.isclose(lr.log_loss(y, p), sk(y, p))                      # same number as scikit-learn
    assert np.isfinite(lr.log_loss(np.array([1, 0]), np.array([0.0, 1.0])))   # p of exactly 0 or 1 is clipped


def test_step4_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3)); y = (rng.random(50) < 0.4).astype(float)
    w = rng.normal(size=3); b = 0.3; l2 = 0.1

    def loss(w_, b_):                                                   # the exact loss the gradient is of
        return lr.log_loss(y, lr.sigmoid(X @ w_ + b_)) + 0.5 * l2 * w_ @ w_

    gw, gb = lr.gradients(X, y, w, b, l2)
    h = 1e-6
    num_w = [(loss(w + h * e, b) - loss(w - h * e, b)) / (2 * h) for e in np.eye(3)]
    num_b = (loss(w, b + h) - loss(w, b - h)) / (2 * h)
    assert np.allclose(gw, num_w, atol=1e-6) and np.isclose(gb, num_b, atol=1e-6)


def test_step4_gradient_descent_matches_sklearn():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 4)); true_w = np.array([1.5, -2.0, 0.0, 0.7])
    y = (rng.random(500) < lr.sigmoid(X @ true_w - 0.5)).astype(int)
    l2 = 0.01
    w, b, history = lr.fit_gd(X, y, lr=0.5, epochs=3000, l2=l2)
    ref = LogisticRegression(C=1 / (l2 * len(y)), max_iter=5000).fit(X, y)   # the same objective in sklearn's terms
    assert np.allclose(w, ref.coef_[0], atol=1e-3) and np.isclose(b, ref.intercept_[0], atol=1e-3)
    assert len(history) == 3000 and history[-1] < history[0]           # the loss went down
    assert np.allclose(lr.predict_proba(X, w, b), ref.predict_proba(X)[:, 1], atol=1e-3)


# ------------------------------------------------------------------ STEP 5: the real model and honest evaluation
@pytest.fixture(scope="module")
def split(raw):
    return lr.time_split(lr.add_features(lr.make_target(raw)))


def test_step5_model_trains_and_beats_random(split):
    train, test = split
    model = lr.train_model(train, C=1.0)
    p = model.predict_proba(test)[:, 1]                                 # probability of default per test loan
    m = lr.evaluate(test["default"], p)
    assert set(m) >= {"roc_auc", "pr_auc", "brier", "log_loss", "default_rate"}
    assert 0.62 < m["roc_auc"] < 0.80                                   # useful, and not suspiciously perfect
    assert m["pr_auc"] > m["default_rate"]                              # better than guessing at the base rate


def test_step5_class_weight_balanced_raises_predicted_risk(split):
    train, test = split
    p_plain = lr.train_model(train).predict_proba(test)[:, 1]
    p_bal = lr.train_model(train, class_weight="balanced").predict_proba(test)[:, 1]
    assert p_bal.mean() > 2 * p_plain.mean()                            # re-weighting distorts probabilities


def test_step5_best_threshold_follows_the_costs():
    y = np.array([0, 0, 0, 0, 1, 1]); p = np.array([0.1, 0.2, 0.3, 0.45, 0.4, 0.9])
    t_expensive_miss = lr.best_threshold(y, p, cost_fn=10, cost_fp=1)   # missing a default costs 10x
    t_cheap_miss = lr.best_threshold(y, p, cost_fn=1, cost_fp=10)       # rejecting a good loan costs 10x
    assert t_expensive_miss <= 0.4 < t_cheap_miss                       # expensive misses -> flag more loans


# ------------------------------------------------------------------ STEP 6: plots and the "break it" experiment
def test_step6_plots_are_written(split, tmp_path):
    train, test = split
    model = lr.train_model(train)
    p = model.predict_proba(test)[:, 1]
    lr.plot_roc_pr(test["default"], p, tmp_path / "roc_pr.png")
    lr.plot_calibration(test["default"], p, tmp_path / "calibration.png")
    lr.plot_coefficients(model, tmp_path / "coefficients.png")
    for name in ("roc_pr.png", "calibration.png", "coefficients.png"):
        assert (tmp_path / name).stat().st_size > 5000                  # a real image, not an empty file


def test_step6_leakage_makes_the_model_look_too_good(split):
    train, test = split
    out = lr.leakage_experiment(train, test)
    assert out["honest_auc"] < 0.80 < out["leaky_auc"]                  # leaked future data inflates the score


# ------------------------------------------------------------------ STEP 7: tuning without touching the test year
def test_step7_ratio_features():
    df = pd.DataFrame({"loan_amnt": [10000.0], "installment": [300.0], "revol_bal": [5000.0],
                       "annual_inc": [50000.0]})
    out = lr.add_ratio_features(df)
    assert out[lr.RATIO_FEATURES].iloc[0].round(3).tolist() == [0.2, 0.072, 0.1]   # 10k/50k, 3.6k/50k, 5k/50k
    assert "loan_to_income" not in df.columns                                      # input untouched


def test_step7_tune_C_uses_only_training_years(split):
    train, test = split
    best, scores = lr.tune_C(lr.add_ratio_features(train), grid=(0.01, 1.0))
    assert best in (0.01, 1.0) and set(scores) == {0.01, 1.0}                       # picked from the grid
    assert all(0.55 < a < 0.80 for a in scores.values())                             # sane validation AUCs
    model = lr.train_tuned_model(lr.add_ratio_features(train), best)
    auc = lr.evaluate(test["default"], model.predict_proba(lr.add_ratio_features(test))[:, 1])["roc_auc"]
    assert 0.6 < auc < 0.8                                                           # works on the test year
