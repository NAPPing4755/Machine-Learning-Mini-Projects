# =====================================================================================================
# Project 1 - Logistic Regression: will this Lending Club loan default?
# -----------------------------------------------------------------------------------------------------
# Every line has a comment saying WHY it is there. Read it top to bottom alongside README.md.
# Tests, step by step:   ..\.venv\Scripts\python -m pytest -q -k step1   (then step2, step3 ...)
# Full run on all 2.26M loans:   ..\.venv\Scripts\python logistic_regression.py   (--sample for a quick run)
# =====================================================================================================

import argparse                                   # reads command-line options such as --sample
import json                                       # saves the final metrics to a .json file
from pathlib import Path                          # builds file paths that work on any operating system

import matplotlib                                 # the plotting library
matplotlib.use("Agg")                             # draw straight to image files (no pop-up windows)
import matplotlib.pyplot as plt                   # the plotting functions themselves
import numpy as np                                # fast maths on arrays (used for the from-scratch model)
import pandas as pd                               # tables of data (DataFrames)
from sklearn.calibration import calibration_curve                   # predicted vs actual default rate
from sklearn.compose import ColumnTransformer                       # different preprocessing per column group
from sklearn.impute import SimpleImputer                            # fills in missing values
from sklearn.linear_model import LogisticRegression                 # scikit-learn's logistic regression
from sklearn.metrics import (average_precision_score,               # PR-AUC: ranking quality on rare defaults
                             brier_score_loss,                      # mean squared error of probabilities
                             log_loss as sk_log_loss,               # sklearn's log loss (renamed: we write our own)
                             precision_recall_curve,                # points for the precision-recall plot
                             roc_auc_score,                         # ROC-AUC: overall ranking quality
                             roc_curve)                             # points for the ROC plot
from sklearn.pipeline import Pipeline                               # chains steps: preprocess -> model
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,   # log transform, one-hot encoding
                                   SplineTransformer, StandardScaler)    # smooth curves (step 7), scaling

# ----------------------------------------------------------------------------------------------- settings
HERE = Path(__file__).resolve().parent                                          # the folder this file lives in
FULL_DATA = Path(r"D:\AI Projects\ML_Mini_Projects\_datasets\lending_club\accepted_2007_to_2018Q4.csv.gz")  # 2.26M loans
SAMPLE_DATA = HERE / "data" / "lending_club_sample.csv"                        # 20,000 rows for fast tests
OUTPUTS = HERE / "outputs"                                                      # plots and metrics go here

RAW_COLUMNS = ["loan_status", "issue_d", "term", "loan_amnt", "int_rate", "installment", "grade",  # columns to read
               "emp_length", "home_ownership", "annual_inc", "verification_status", "purpose", "dti",  # (151 exist;
               "delinq_2yrs", "earliest_cr_line", "fico_range_low", "inq_last_6mths", "open_acc",       # reading only
               "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies",  # these saves
               "application_type", "total_pymnt", "recoveries", "last_fico_range_high",                # memory)
               "sub_grade", "addr_state"]                                                              # step 7 extras

LEAKY_COLUMNS = ["total_pymnt", "recoveries", "last_fico_range_high"]  # known only AFTER the loan ends: never a feature

NUMERIC_FEATURES = ["loan_amnt", "int_rate", "installment", "annual_inc", "dti", "fico_range_low",  # numbers known
                    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",  # on the day
                    "total_acc", "mort_acc", "pub_rec_bankruptcies", "emp_years", "credit_history_years"]  # of applying
LOG_FEATURES = ["annual_inc", "revol_bal"]        # heavily skewed money amounts: log-transform them before scaling
CATEGORICAL_FEATURES = ["grade", "home_ownership", "verification_status", "purpose", "application_type"]  # text labels

RATIO_FEATURES = ["loan_to_income", "pay_to_income", "revol_to_income"]   # step 7: affordability ratios
EXTRA_CATEGORICAL = ["sub_grade", "addr_state"]   # step 7: finer grade (A1..G5) and US state

TRAIN_YEARS = (2012, 2013, 2014)                  # learn from loans issued in these years ...
TEST_YEARS = (2015,)                              # ... and test on the following year, like real life


# ========================================================================= STEP 1: load data, build the target
def load_loans(path, nrows=None):
    """Read the Lending Club CSV (only RAW_COLUMNS) and turn issue_d ("Dec-2015") into a real date."""
    df = pd.read_csv(path,                        # pandas reads .csv and .csv.gz the same way
                     usecols=RAW_COLUMNS,         # only the 28 columns we need, not all 151 (saves memory and time)
                     nrows=nrows,                 # None = every row; a number = just the first rows (quick tests)
                     low_memory=False)            # infer each column's type in one pass (no "mixed types" warning)
    df = df[RAW_COLUMNS]                          # put columns in OUR order (read_csv keeps the file's order)
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")   # "Dec-2015" -> a real date we can sort/split
    return df                                     # hand the table back to the caller


def make_target(df):
    """Keep only FINISHED 36-month loans and add the target column `default` (1 = Charged Off, 0 = Fully Paid)."""
    finished = df["loan_status"].isin(["Fully Paid", "Charged Off"])   # only loans whose outcome is known
    short = df["term"].str.strip() == "36 months"                     # raw text is " 36 months": strip the space
    out = df[finished & short].copy()             # keep rows passing BOTH tests; copy so the caller's df is untouched
    out["default"] = (out["loan_status"] == "Charged Off").astype(int)  # True/False -> 1/0: the thing we predict
    return out                                    # every row now has a known 0/1 outcome


# ========================================================================= STEP 2: clean features, split by time
def clean_emp_length(s):
    """'10+ years' -> 10.0, '< 1 year' -> 0.0, '3 years' -> 3.0, missing stays NaN."""
    s = s.replace({"< 1 year": "0 years"})        # otherwise the digit search below would read "< 1" as 1
    years = s.str.extract(r"(\d+)")[0]            # first run of digits as text: "10+ years" -> "10"
    return years.astype(float)                    # text -> number; float because NaN (missing) is a float


def credit_history_years(issue_d, earliest_cr_line):
    """Years between the borrower's first credit line and the loan date (a longer history = more evidence)."""
    first = pd.to_datetime(earliest_cr_line, format="%b-%Y")   # "Aug-2003" -> a date, same format as issue_d
    return (issue_d - first).dt.days / 365.25     # difference in days -> years (365.25 averages in leap years)


def add_features(df):
    """Return a COPY of df with two new columns: emp_years and credit_history_years."""
    out = df.copy()                               # never modify the caller's table
    out["emp_years"] = clean_emp_length(out["emp_length"])          # messy text -> years employed
    out["credit_history_years"] = credit_history_years(out["issue_d"], out["earliest_cr_line"])  # history length
    return out                                    # the original columns plus the two new ones


def time_split(df):
    """Split by issue year: train = TRAIN_YEARS, test = TEST_YEARS. Returns (train, test)."""
    year = df["issue_d"].dt.year                  # the year each loan was issued
    train = df[year.isin(TRAIN_YEARS)].copy()     # the past (2012-2014): what the model learns from
    test = df[year.isin(TEST_YEARS)].copy()       # the future (2015): what it is judged on, never seen in training
    return train, test                            # a bank works exactly like this: train on old loans, use on new ones


# ========================================================================= STEP 3: preprocessing pipeline
def build_preprocessor():
    """A ColumnTransformer that turns the loan table into a matrix of numbers the model can use."""
    plain = [c for c in NUMERIC_FEATURES if c not in LOG_FEATURES]     # numbers that need no log transform
    numeric = Pipeline([                          # the recipe for ordinary numeric columns
        ("impute", SimpleImputer(strategy="median")),   # fill gaps with the TRAIN median (robust to outliers)
        ("scale", StandardScaler()),              # (x - mean) / std: every feature on a similar scale
    ])
    logged = Pipeline([                           # the recipe for skewed money columns
        ("impute", SimpleImputer(strategy="median")),   # fill gaps first (log of NaN is still NaN)
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),  # log(1+x) squashes huge values;
                                                  # feature_names_out keeps column names for the coefficient plot
        ("scale", StandardScaler()),              # then standardise like the others
    ])
    categorical = Pipeline([                      # the recipe for text columns
        ("impute", SimpleImputer(strategy="most_frequent")),   # a missing category becomes the commonest one
        ("onehot", OneHotEncoder(handle_unknown="ignore")),    # one 0/1 column per category; unseen -> all zeros
    ])
    return ColumnTransformer([                    # apply each recipe to its own columns, then glue the results
        ("num", numeric, plain),                  # numeric columns come first (the tests rely on this order)
        ("log", logged, LOG_FEATURES),            # then the logged money columns
        ("cat", categorical, CATEGORICAL_FEATURES),   # then the one-hot columns
    ])                                            # any column not listed (e.g. loan_status, leaky ones) is DROPPED


# ========================================================================= STEP 4: logistic regression from scratch
def sigmoid(z):
    """1 / (1 + e^-z), computed so that huge |z| never overflows."""
    z = np.asarray(z, dtype=float)                # accept lists or arrays; make sure we do float maths
    out = np.empty_like(z)                        # an empty array of the same shape for the answers
    pos = z >= 0                                  # True where z is zero or positive
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))      # z >= 0: exp(-z) <= 1, can never overflow
    ez = np.exp(z[~pos])                          # z < 0: exp(z) is tiny, also safe
    out[~pos] = ez / (1.0 + ez)                   # the same sigmoid, rearranged (multiply top and bottom by e^z)
    return out                                    # probabilities between 0 and 1


def log_loss(y, p, eps=1e-15):
    """Mean binary cross-entropy: -mean( y*log(p) + (1-y)*log(1-p) )."""
    p = np.clip(p, eps, 1 - eps)                  # keep p away from exactly 0 or 1, because log(0) = -infinity
    losses = y * np.log(p) + (1 - y) * np.log(1 - p)   # per row: log of the probability given to the TRUE class
    return float(-np.mean(losses))                # average, flip the sign (lower = better), plain Python float


def gradients(X, y, w, b, l2=0.0):
    """Gradients of  log_loss + (l2/2)*||w||^2  with respect to w and b. Returns (grad_w, grad_b)."""
    p = sigmoid(X @ w + b)                        # current predicted probability for every row
    err = p - y                                   # the calculus result: dLoss/dz = prediction - truth
    n = len(y)                                    # number of rows, to average rather than sum
    grad_w = X.T @ err / n + l2 * w               # chain rule (dz/dw = x), averaged, plus the L2 penalty's gradient
    grad_b = float(err.mean())                    # dz/db = 1; the bias is not penalised
    return grad_w, grad_b                         # the direction of steepest INCREASE of the loss


def fit_gd(X, y, lr=0.1, epochs=2000, l2=0.0):
    """Batch gradient descent. Returns (w, b, history) where history[i] is the loss after epoch i."""
    w = np.zeros(X.shape[1])                      # start with every weight at 0 (fine: the loss is convex)
    b = 0.0                                       # and the bias at 0, i.e. p = 0.5 for everyone
    history = []                                  # the loss after each epoch, to check it goes down
    for _ in range(epochs):                       # one epoch = one pass over ALL rows ("batch" gradient descent)
        grad_w, grad_b = gradients(X, y, w, b, l2)   # which way is uphill right now?
        w = w - lr * grad_w                       # step downhill, scaled by the learning rate
        b = b - lr * grad_b                       # same for the bias
        loss = log_loss(y, sigmoid(X @ w + b)) + 0.5 * l2 * float(w @ w)   # the full objective we are minimising
        history.append(loss)                      # record it for the learning curve
    return w, b, history                          # the trained model and how training went


def predict_proba(X, w, b):
    """Probability of class 1 for every row."""
    return sigmoid(X @ w + b)                     # linear score -> squashed into a probability


# ========================================================================= STEP 5: the real model, honest evaluation
def train_model(train, C=1.0, class_weight=None):
    """Pipeline(preprocessor -> LogisticRegression) fitted on the train DataFrame and its `default` column."""
    model = Pipeline([                            # preprocessing and model travel together as one object
        ("prep", build_preprocessor()),           # table -> clean numeric matrix (learned on train only)
        ("clf", LogisticRegression(C=C,           # C = 1/regularisation strength: small C = simpler model
                                   class_weight=class_weight,   # None, or "balanced" to up-weight rare defaults
                                   max_iter=1000)),   # the default of 100 iterations often stops too early here
    ])
    model.fit(train, train["default"])            # learn medians/means/categories AND weights from the past
    return model                                  # ready for model.predict_proba(new_loans)


def evaluate(y, p):
    """Dict with roc_auc, pr_auc, brier, log_loss and default_rate (all plain floats)."""
    return {
        "roc_auc": float(roc_auc_score(y, p)),    # ranking quality: 0.5 = coin flip, 1.0 = perfect
        "pr_auc": float(average_precision_score(y, p)),   # precision/recall on the rare class; compare to base rate
        "brier": float(brier_score_loss(y, p)),   # mean squared error of the probabilities (lower = better)
        "log_loss": float(sk_log_loss(y, p)),     # the training objective measured on unseen data
        "default_rate": float(np.mean(y)),        # the base rate: what PR-AUC must beat
    }


def best_threshold(y, p, cost_fn, cost_fp):
    """The cut-off t in 0.01..0.99 that minimises total cost when we flag every loan with p >= t."""
    y, p = np.asarray(y), np.asarray(p)           # plain arrays: fast element-wise comparisons
    best_t, best_cost = 0.5, float("inf")         # start with "no answer yet"
    for t in np.linspace(0.01, 0.99, 99):         # try every threshold 0.01, 0.02, ..., 0.99
        missed = np.sum((p < t) & (y == 1))       # defaults we approved (false negatives)
        false_alarms = np.sum((p >= t) & (y == 0))    # good loans we rejected (false positives)
        cost = cost_fn * missed + cost_fp * false_alarms   # what this threshold would cost the bank
        if cost < best_cost:                      # cheaper than anything so far?
            best_t, best_cost = float(t), cost    # remember it
    return best_t                                 # the cheapest cut-off


# ========================================================================= STEP 6: plots and "break it"
def plot_roc_pr(y, p, path):
    """Two plots side by side: ROC curve (with the diagonal) and precision-recall (with the base rate)."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))   # one figure, two panels
    fpr, tpr, _ = roc_curve(y, p)                 # false/true positive rate at every threshold
    ax[0].plot(fpr, tpr, label=f"model (AUC {roc_auc_score(y, p):.3f})")   # the model's curve
    ax[0].plot([0, 1], [0, 1], "--", color="grey", label="random (AUC 0.5)")   # what guessing looks like
    ax[0].set(xlabel="False positive rate (good loans flagged)", ylabel="True positive rate (defaults caught)",
              title="ROC curve")                  # axis labels say what the rates mean in loan terms
    ax[0].legend()                                # show which line is which
    prec, rec, _ = precision_recall_curve(y, p)   # precision and recall at every threshold
    ax[1].plot(rec, prec, label=f"model (AP {average_precision_score(y, p):.3f})")   # the model's curve
    ax[1].axhline(np.mean(y), ls="--", color="grey", label=f"base rate ({np.mean(y):.3f})")   # guessing level
    ax[1].set(xlabel="Recall (share of defaults caught)", ylabel="Precision (flagged loans that default)",
              title="Precision-recall curve")     # the honest view when positives are rare
    ax[1].legend()                                # legend for the second panel
    fig.tight_layout()                            # stop labels overlapping
    fig.savefig(path, dpi=120)                    # write the image file
    plt.close(fig)                                # free the memory (important when making many plots)


def plot_calibration(y, p, path):
    """Reliability diagram: in each probability bin, the predicted default rate vs the real one."""
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10)   # 10 bins: mean prediction vs actual default rate
    fig, ax = plt.subplots(figsize=(5.5, 5.5))    # a square plot: both axes are probabilities
    ax.plot(mean_pred, frac_pos, "o-", label="model")   # one dot per bin
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfectly calibrated")   # predicted = actual
    ax.set(xlabel="Predicted probability of default", ylabel="Actual default rate",
           title="Calibration (reliability diagram)")   # dots on the line = honest probabilities
    ax.legend()                                   # which line is which
    fig.tight_layout()                            # tidy margins
    fig.savefig(path, dpi=120)                    # write the image
    plt.close(fig)                                # free memory


def plot_coefficients(model, path, top=20):
    """Horizontal bar chart of the `top` largest coefficients (by absolute value), with feature names."""
    names = model.named_steps["prep"].get_feature_names_out()   # e.g. "num__int_rate", "cat__grade_G"
    coef = model.named_steps["clf"].coef_[0]      # one weight per feature (comparable, because features are scaled)
    idx = np.argsort(np.abs(coef))[-top:]         # positions of the `top` biggest effects, smallest first
    colors = ["tab:red" if c > 0 else "tab:green" for c in coef[idx]]   # red raises risk, green lowers it
    fig, ax = plt.subplots(figsize=(8, 7))        # tall enough for 20 labels
    ax.barh(names[idx], coef[idx], color=colors)  # one bar per feature
    ax.axvline(0, color="black", lw=0.8)          # the zero line: no effect
    ax.set(xlabel="Coefficient (change in log-odds of default per 1 std)", title=f"Top {top} coefficients")
    fig.tight_layout()                            # make room for long feature names
    fig.savefig(path, dpi=120)                    # write the image
    plt.close(fig)                                # free memory


def leakage_experiment(train, test):
    """Train once honestly and once WITH LEAKY_COLUMNS added; return {"honest_auc": ..., "leaky_auc": ...}."""
    honest = train_model(train)                   # the real model: application-time features only
    honest_auc = roc_auc_score(test["default"], honest.predict_proba(test)[:, 1])   # its true score
    leaky_prep = ColumnTransformer([              # the same idea, but the numeric group includes FUTURE columns
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
         NUMERIC_FEATURES + LEAKY_COLUMNS),       # total_pymnt, recoveries, last FICO: known only after the loan
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),   # categories as before
    ])
    leaky = Pipeline([("prep", leaky_prep), ("clf", LogisticRegression(max_iter=1000))])   # same model type
    leaky.fit(train, train["default"])            # it will "learn" that low repayments mean default (circular!)
    leaky_auc = roc_auc_score(test["default"], leaky.predict_proba(test)[:, 1])   # a dazzling, meaningless score
    return {"honest_auc": float(honest_auc), "leaky_auc": float(leaky_auc)}   # both, to compare in the README


# ========================================================================= STEP 7: tuning (honestly)
def add_ratio_features(df):
    """Affordability ratios a credit analyst would compute. Returns a COPY with RATIO_FEATURES added."""
    out = df.copy()                               # never modify the caller's table
    income = out["annual_inc"].clip(lower=1000)   # floor tiny/zero incomes so a ratio can't explode to infinity
    out["loan_to_income"] = out["loan_amnt"] / income             # how big the loan is relative to income
    out["pay_to_income"] = out["installment"] * 12 / income        # yearly repayments as a share of income
    out["revol_to_income"] = out["revol_bal"] / income             # existing card debt relative to income
    return out                                    # original columns + the three ratios


def build_tuned_preprocessor():
    """Like build_preprocessor, plus ratio features, sub_grade/state, and SPLINES on every numeric column."""
    numeric = NUMERIC_FEATURES + RATIO_FEATURES   # the original numbers plus the three ratios
    num_pipe = Pipeline([                         # recipe for numbers
        ("impute", SimpleImputer(strategy="median")),   # fill gaps with the train median
        ("scale", StandardScaler()),              # put every feature on the same scale first
        ("spline", SplineTransformer(n_knots=5, degree=3)),   # each number -> 7 smooth "bump" columns, so the model
                                                  # can learn CURVES (risk rising fast, then flattening) not just lines
    ])
    cat_pipe = Pipeline([                         # recipe for categories
        ("impute", SimpleImputer(strategy="most_frequent")),   # missing -> commonest category
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),   # categories with < 50 loans are
                                                  # grouped together, so the model doesn't fit noise
    ])
    return ColumnTransformer([                    # glue both groups side by side
        ("num", num_pipe, numeric),               # spline-expanded numbers
        ("cat", cat_pipe, CATEGORICAL_FEATURES + EXTRA_CATEGORICAL),   # one-hot categories incl. sub_grade, state
    ])


def tune_C(train, grid=(0.001, 0.003, 0.01, 0.1, 1.0)):
    """Pick C on a VALIDATION year carved out of the training years. The test year is never touched.

    Returns (best_C, {C: validation_auc}).
    """
    year = train["issue_d"].dt.year               # issue year of every training loan
    fit_part = train[year < max(TRAIN_YEARS)]     # 2012-2013: fit candidate models here
    val_part = train[year == max(TRAIN_YEARS)]    # 2014: score them here (the "future" relative to fit_part)
    scores = {}                                   # validation AUC for every C we try
    for C in grid:                                # try each regularisation strength
        model = Pipeline([("prep", build_tuned_preprocessor()),   # same preprocessing as the final model
                          ("clf", LogisticRegression(C=C, max_iter=3000))])   # more columns -> more iterations
        model.fit(fit_part, fit_part["default"])  # train on the earlier years only
        p = model.predict_proba(val_part)[:, 1]   # predict the later year
        scores[C] = float(roc_auc_score(val_part["default"], p))   # how well did this C generalise?
    best_C = max(scores, key=scores.get)          # the C with the highest validation AUC
    return best_C, scores                         # the winner, and every score for the README


def train_tuned_model(train, C):
    """The tuned pipeline, refitted on ALL training years with the chosen C."""
    model = Pipeline([("prep", build_tuned_preprocessor()),   # richer features + splines
                      ("clf", LogisticRegression(C=C, max_iter=3000))])   # the C picked on validation
    model.fit(train, train["default"])            # now use all of 2012-2014 (more data = better estimates)
    return model                                  # evaluated ONCE on 2015 in main()


# ========================================================================= run everything
def main():
    """Full run: load -> clean -> split -> train -> evaluate -> plots -> metrics.json."""
    ap = argparse.ArgumentParser()                                          # command-line options
    ap.add_argument("--sample", action="store_true", help="use the 20k-row sample instead of the full data")
    args = ap.parse_args()                                                  # read the options
    OUTPUTS.mkdir(exist_ok=True)                                            # make sure outputs/ exists
    path = SAMPLE_DATA if args.sample else FULL_DATA                        # which file to read
    df = add_features(make_target(load_loans(path)))                        # steps 1 and 2
    train, test = time_split(df)                                            # past -> future split
    print(f"train {len(train):,} loans ({train['default'].mean():.1%} default), "
          f"test {len(test):,} loans ({test['default'].mean():.1%} default)")
    results = {}                                                            # every metric goes in here
    for name, cw in (("plain", None), ("balanced", "balanced")):            # compare both class weightings
        model = train_model(train, C=1.0, class_weight=cw)                  # step 5
        p = model.predict_proba(test)[:, 1]                                 # probability of default
        results[name] = evaluate(test["default"], p)                        # step 5
        results[name]["mean_predicted"] = float(p.mean())                   # compare with the real default rate
        results[name]["threshold_fn5_fp1"] = best_threshold(test["default"], p, cost_fn=5, cost_fp=1)
        print(name, {k: round(v, 4) for k, v in results[name].items()})   # show progress
        if name == "plain":                                                 # plots for the honest model
            plot_roc_pr(test["default"], p, OUTPUTS / "roc_pr.png")
            plot_calibration(test["default"], p, OUTPUTS / "calibration.png")
            plot_coefficients(model, OUTPUTS / "coefficients.png")
    results["leakage"] = leakage_experiment(train, test)                    # step 6: break it
    train_r, test_r = add_ratio_features(train), add_ratio_features(test)   # step 7: richer features
    best_C, val_scores = tune_C(train_r)                                    # choose C on 2014, not on 2015
    print("validation AUC by C:", {c: round(a, 4) for c, a in val_scores.items()}, "-> best C", best_C)
    tuned = train_tuned_model(train_r, best_C)                              # refit on all training years
    p_tuned = tuned.predict_proba(test_r)[:, 1]                             # the ONE look at the test year
    results["tuned"] = evaluate(test["default"], p_tuned)                   # same metrics as before
    results["tuned"]["C"] = best_C                                          # remember which C won
    results["tuned"]["validation_auc_by_C"] = {str(c): a for c, a in val_scores.items()}   # json needs str keys
    print("tuned", {k: v for k, v in results["tuned"].items() if k != "validation_auc_by_C"})
    print("leakage", results["leakage"])
    (OUTPUTS / "metrics.json").write_text(json.dumps(results, indent=2))   # save for the README
    print(f"saved plots and metrics to {OUTPUTS}")


if __name__ == "__main__":                        # only runs when you start this file directly
    main()
