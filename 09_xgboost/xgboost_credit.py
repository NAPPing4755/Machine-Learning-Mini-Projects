# =====================================================================================================
# Project 9 - XGBoost: Home Credit default risk (307,511 applicants, 7 linked tables, 2.7 GB)
# -----------------------------------------------------------------------------------------------------
# Every line has a comment saying WHY it is there. Read it alongside README.md.
# Sample:     ..\.venv\Scripts\python make_sample.py      (once: competition data can't be redistributed)
# Tests:      ..\.venv\Scripts\python -m pytest -q
# Full run:   ..\.venv\Scripts\python xgboost_credit.py   (about 20 minutes; --quick for the sample)
# =====================================================================================================

import argparse                                   # command-line options
import json                                       # save metrics
from pathlib import Path                          # file paths

import matplotlib                                 # plotting
matplotlib.use("Agg")                             # write images to files
import matplotlib.pyplot as plt                   # plotting functions
import numpy as np                                # arrays and maths
import pandas as pd                               # tables
import xgboost as xgb                             # the library that dominated tabular Kaggle for years
from sklearn.metrics import brier_score_loss, roc_auc_score   # metrics
from sklearn.model_selection import train_test_split   # stratified split

HERE = Path(__file__).resolve().parent                                        # this project's folder
DATA_DIR = HERE.parent / "_datasets" / "home_credit"                          # the 7 Kaggle tables
SAMPLE_DIR = HERE / "data"                                                    # the same 7 tables for 10k applicants
OUTPUTS = HERE / "outputs"                                                    # plots and metrics
TABLES = ["bureau", "bureau_balance", "previous_application", "installments_payments", "POS_CASH_balance",
          "credit_card_balance"]                  # the 6 history tables around application_train


# ========================================================================= loading and feature engineering
def read(data_dir, name):
    return pd.read_csv(data_dir / f"{name}.csv")  # one table


def clean_application(app):
    """The main table, with its traps fixed and a few ratios a credit analyst would compute. Returns a COPY."""
    a = app.copy()                                # never modify the caller's table
    a["DAYS_EMPLOYED_SENTINEL"] = (a["DAYS_EMPLOYED"] == 365243).astype(int)   # "employed 1,000 years" is a CODE
    a.loc[a["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan   # (pensioners/unemployed), not a duration
    a["CREDIT_TO_INCOME"] = a["AMT_CREDIT"] / a["AMT_INCOME_TOTAL"]   # how big the loan is vs income
    a["ANNUITY_TO_INCOME"] = a["AMT_ANNUITY"] / a["AMT_INCOME_TOTAL"]   # yearly repayment burden
    a["CREDIT_TO_GOODS"] = a["AMT_CREDIT"] / a["AMT_GOODS_PRICE"]   # borrowing more than the item costs?
    a["TERM_YEARS"] = a["AMT_CREDIT"] / a["AMT_ANNUITY"]   # roughly how many years to repay
    a["EXT_SOURCE_MEAN"] = a[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)   # external scores
    a["AGE_YEARS"] = -a["DAYS_BIRTH"] / 365.25     # days are stored as negative numbers before the application
    for c in a.select_dtypes("object").columns:   # text columns -> pandas categories (XGBoost handles them natively)
        a[c] = a[c].astype("category")
    return a


def aggregate(df, key, prefix):
    """Many rows per applicant -> one row: mean/max/min of every number, plus a row count."""
    num = df.select_dtypes("number").drop(columns=[c for c in ("SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU") if c in df and c != key], errors="ignore")
    agg = num.groupby(key).agg(["mean", "max", "min"])   # 3 summaries per numeric column
    agg.columns = [f"{prefix}_{c}_{s}" for c, s in agg.columns]   # flat, readable names
    agg[f"{prefix}_COUNT"] = df.groupby(key).size()   # how many records (e.g. previous loans) exist
    return agg


def history_features(tables):
    """Summarise each history table per applicant (SK_ID_CURR)."""
    bb = aggregate(tables["bureau_balance"].assign(LATE=lambda d: d["STATUS"].isin(list("12345")).astype(int)),
                   "SK_ID_BUREAU", "BB")          # monthly status of each credit-bureau loan -> per loan
    bureau = tables["bureau"].join(bb, on="SK_ID_BUREAU")   # attach to the loan, then summarise per applicant
    bureau["ACTIVE"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(int)   # still open elsewhere?
    prev = tables["previous_application"].assign(REFUSED=lambda d: (d["NAME_CONTRACT_STATUS"] == "Refused").astype(int))
    inst = tables["installments_payments"].assign(
        PAID_LATE_DAYS=lambda d: (d["DAYS_ENTRY_PAYMENT"] - d["DAYS_INSTALMENT"]).clip(lower=0),   # days late
        PAID_SHORT=lambda d: (d["AMT_INSTALMENT"] - d["AMT_PAYMENT"]).clip(lower=0))   # amount missing
    parts = {"bureau": aggregate(bureau, "SK_ID_CURR", "BUREAU"),   # other lenders (credit bureau)
             "previous_application": aggregate(prev, "SK_ID_CURR", "PREV"),   # earlier Home Credit applications
             "installments_payments": aggregate(inst, "SK_ID_CURR", "INST"),   # repayment behaviour
             "POS_CASH_balance": aggregate(tables["POS_CASH_balance"], "SK_ID_CURR", "POS"),   # point-of-sale loans
             "credit_card_balance": aggregate(tables["credit_card_balance"], "SK_ID_CURR", "CC")}   # card usage
    return parts


def build(data_dir, history=("bureau", "previous_application", "installments_payments", "POS_CASH_balance",
                             "credit_card_balance")):
    """application_train + the chosen history summaries, one row per applicant."""
    app = clean_application(read(data_dir, "application_train"))   # the main table
    tables = {t: read(data_dir, t) for t in TABLES}   # all history tables
    parts = history_features(tables)              # summaries
    for name in history:                          # join the requested ones
        app = app.join(parts[name], on="SK_ID_CURR")
    return app


# ========================================================================= the XGBoost split formula, from scratch
def leaf_weight(G, H, lam):
    """Optimal leaf value for a 2nd-order approximation of the loss: w = -G / (H + lambda)."""
    return -G / (H + lam)                         # G = sum of gradients, H = sum of hessians in the leaf


def split_gain(GL, HL, GR, HR, lam, gamma=0.0):
    """XGBoost's gain: how much a split improves the regularised objective (> 0 = worth it)."""
    score = lambda G, H: G * G / (H + lam)        # quality of a leaf
    return 0.5 * (score(GL, HL) + score(GR, HR) - score(GL + GR, HL + HR)) - gamma   # gamma = price of a new leaf


class ScratchXGBStumps:
    """Logistic-loss boosting with depth-1 trees chosen by XGBoost's exact gain formula."""

    def __init__(self, n_rounds=20, eta=0.3, lam=1.0):
        self.n_rounds, self.eta, self.lam = n_rounds, eta, lam   # settings named as in XGBoost

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)   # plain arrays
        margin = np.zeros(len(y))                 # log-odds; 0 = probability 0.5 (XGBoost's base_score=0.5)
        self.stumps = []                          # (feature, threshold, left value, right value)
        for _ in range(self.n_rounds):
            p = 1 / (1 + np.exp(-margin))         # current probabilities
            g, h = p - y, p * (1 - p)             # 1st and 2nd derivative of log loss w.r.t. the margin
            best = (-np.inf, None)                # best gain so far
            for j in range(X.shape[1]):           # every feature ...
                order = np.argsort(X[:, j]); xs = X[order, j]
                Gc, Hc = np.cumsum(g[order]), np.cumsum(h[order])   # running sums: each split O(1)
                for i in range(1, len(xs)):       # ... every threshold between distinct values
                    if xs[i] == xs[i - 1]:
                        continue
                    gain = split_gain(Gc[i - 1], Hc[i - 1], Gc[-1] - Gc[i - 1], Hc[-1] - Hc[i - 1], self.lam)
                    if gain > best[0]:
                        best = (gain, (j, (xs[i - 1] + xs[i]) / 2, Gc[i - 1], Hc[i - 1], Gc[-1], Hc[-1]))
            j, t, GL, HL, G, H = best[1]          # the winning split
            wl, wr = leaf_weight(GL, HL, self.lam), leaf_weight(G - GL, H - HL, self.lam)   # optimal leaf values
            self.stumps.append((j, t, self.eta * wl, self.eta * wr))   # shrunk by eta (the learning rate)
            margin += np.where(X[:, j] < t, self.eta * wl, self.eta * wr)   # update everyone's log-odds
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)                  # plain array
        m = sum(np.where(X[:, j] < t, l, r) for j, t, l, r in self.stumps)   # add up all stumps
        return 1 / (1 + np.exp(-m))               # log-odds -> probability


# ========================================================================= experiments
PARAMS = {"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",   # histogram trees
          "eta": 0.05, "max_depth": 6, "min_child_weight": 5, "subsample": 0.8,   # row sampling per tree
          "colsample_bytree": 0.6, "lambda": 1.0, "alpha": 0.0, "nthread": -1, "seed": 0}   # column sampling, L2/L1


def fit_xgb(Xtr, ytr, Xva, yva, params=None, rounds=5000):
    """Train with early stopping on the VALIDATION set; return (booster, best_iteration)."""
    p = {**PARAMS, **(params or {})}              # defaults + overrides
    dtr = xgb.DMatrix(Xtr, ytr, enable_categorical=True)   # XGBoost's data format (keeps categories)
    dva = xgb.DMatrix(Xva, yva, enable_categorical=True)
    b = xgb.train(p, dtr, rounds, evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=False)
    return b, b.best_iteration                    # stops 100 rounds after the validation AUC peaked


def predict(b, X):
    return b.predict(xgb.DMatrix(X, enable_categorical=True), iteration_range=(0, b.best_iteration + 1))   # best trees


def table_ablation(data_dir, ytr_idx, yva_idx, yte_idx):
    """How much does each history table add? Main table only -> + bureau -> + everything."""
    out = {}                                      # results
    for name, hist in (("application only", ()), ("+ credit bureau", ("bureau",)),
                       ("+ all 6 history tables", ("bureau", "previous_application", "installments_payments",
                                                   "POS_CASH_balance", "credit_card_balance"))):
        df = build(data_dir, hist)                # rebuild with these tables
        X, y = df.drop(columns=["TARGET", "SK_ID_CURR"]), df["TARGET"]   # features / label
        b, it = fit_xgb(X.iloc[ytr_idx], y.iloc[ytr_idx], X.iloc[yva_idx], y.iloc[yva_idx])
        out[name] = {"features": X.shape[1], "trees": it + 1,
                     "test_auc": round(roc_auc_score(y.iloc[yte_idx], predict(b, X.iloc[yte_idx])), 4)}
        print("  ", name, out[name])
    return out


def param_sweep(Xtr, ytr, Xva, yva, Xte, yte):
    """One knob at a time around the defaults."""
    grid = {"max_depth": [3, 6, 9], "min_child_weight": [1, 5, 50], "colsample_bytree": [0.3, 0.6, 1.0],
            "subsample": [0.5, 0.8, 1.0], "lambda": [0.0, 1.0, 10.0]}   # each list includes the default
    out = {}                                      # knob=value -> test AUC
    for knob, values in grid.items():
        for v in values:
            b, it = fit_xgb(Xtr, ytr, Xva, yva, {knob: v}, rounds=3000)
            out[f"{knob}={v}"] = {"trees": it + 1, "test_auc": round(roc_auc_score(yte, predict(b, Xte)), 4)}
    return out


def imbalance(Xtr, ytr, Xva, yva, Xte, yte):
    """scale_pos_weight (= negatives / positives ~ 11) vs none: ranking vs probability quality."""
    out = {}                                      # results
    for name, w in (("none", 1.0), ("scale_pos_weight=11", float((ytr == 0).sum() / (ytr == 1).sum()))):
        b, it = fit_xgb(Xtr, ytr, Xva, yva, {"scale_pos_weight": w})
        p = predict(b, Xte)                       # predicted default probability
        out[name] = {"auc": round(roc_auc_score(yte, p), 4), "brier": round(brier_score_loss(yte, p), 4),
                     "mean_predicted": round(float(p.mean()), 4), "true_rate": round(float(yte.mean()), 4)}
    return out


def shap_summary(b, X, path_bar, path_one):
    """TreeSHAP, built into XGBoost: each feature's push on each applicant's log-odds."""
    contrib = b.predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True,
                        iteration_range=(0, b.best_iteration + 1))   # last column = the base value
    names = list(X.columns)                       # feature names
    imp = pd.Series(np.abs(contrib[:, :-1]).mean(axis=0), index=names).sort_values()   # mean |push| per feature
    fig, ax = plt.subplots(figsize=(8, 7)); imp.tail(20).plot.barh(ax=ax)   # top 20
    ax.set(xlabel="mean |SHAP value| (log-odds)", title="What drives default risk (top 20)")
    fig.tight_layout(); fig.savefig(path_bar, dpi=120); plt.close(fig)
    i = int(np.argmax(contrib.sum(axis=1)))       # the riskiest applicant in the set
    one = pd.Series(contrib[i, :-1], index=names)  # their per-feature pushes
    top = one.reindex(one.abs().sort_values().tail(12).index)   # the 12 biggest
    fig, ax = plt.subplots(figsize=(8, 5.5)); ax.barh(top.index, top.values, color=["tab:red" if v > 0 else "tab:green" for v in top.values])
    ax.set(xlabel="push on log-odds of default", title="Why this applicant is high risk (red = riskier)")
    fig.tight_layout(); fig.savefig(path_one, dpi=120); plt.close(fig)
    return imp.tail(10)[::-1].round(4).to_dict()  # top 10 for the README


# ========================================================================= run everything
def main():
    ap = argparse.ArgumentParser()                # options
    ap.add_argument("--quick", action="store_true")   # the 10k-applicant sample
    args = ap.parse_args()
    OUTPUTS.mkdir(exist_ok=True)                  # outputs/
    data_dir = SAMPLE_DIR if args.quick else DATA_DIR
    df = build(data_dir)                          # all tables, one row per applicant
    X, y = df.drop(columns=["TARGET", "SK_ID_CURR"]), df["TARGET"]   # features / label
    idx = np.arange(len(df))                      # positions
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=0)   # 80% / 20% test
    tr, va = train_test_split(tr, test_size=0.125, stratify=y.iloc[tr], random_state=0)   # 70 / 10 / 20
    print(f"{len(df):,} applicants, {X.shape[1]} features, default rate {y.mean():.3f}")
    res = {"rows": {"train": len(tr), "validation": len(va), "test": len(te)}, "features": X.shape[1]}
    res["employment_sentinel"] = int(df["DAYS_EMPLOYED_SENTINEL"].sum())   # the 365,243-day trap
    Xtr, ytr, Xva, yva, Xte, yte = X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va], X.iloc[te], y.iloc[te]
    b, it = fit_xgb(Xtr, ytr, Xva, yva)           # 1. the main model
    p = predict(b, Xte)
    res["main"] = {"trees": it + 1, "test_auc": round(roc_auc_score(yte, p), 4)}
    print("main", res["main"])
    res["tables"] = table_ablation(data_dir, tr, va, te)   # 2. value of each table
    res["params"] = param_sweep(Xtr, ytr, Xva, yva, Xte, yte)   # 3. the knobs (illustrative)
    print("params", res["params"])
    res["imbalance"] = imbalance(Xtr, ytr, Xva, yva, Xte, yte)   # 4. scale_pos_weight
    print("imbalance", res["imbalance"])
    res["shap_top10"] = shap_summary(b, Xte.iloc[:20000], OUTPUTS / "shap_top20.png", OUTPUTS / "shap_one_applicant.png")
    print("shap", res["shap_top10"])
    num = Xtr.select_dtypes("number").columns[:6]  # 5. from-scratch XGBoost stumps vs the library (small data)
    sub = Xtr[num].fillna(0).iloc[:3000]; ysub = ytr.iloc[:3000].values
    ours = ScratchXGBStumps(n_rounds=10, eta=0.3, lam=1.0).fit(sub, ysub)
    ref = xgb.train({"objective": "binary:logistic", "max_depth": 1, "eta": 0.3, "lambda": 1.0, "min_child_weight": 0,
                     "tree_method": "exact", "base_score": 0.5}, xgb.DMatrix(sub, ysub), 10)
    res["scratch_max_prob_diff"] = float(np.max(np.abs(ours.predict_proba(sub) - ref.predict(xgb.DMatrix(sub)))))
    print("scratch diff", res["scratch_max_prob_diff"])
    (OUTPUTS / "metrics.json").write_text(json.dumps(res, indent=2))
    print("saved to", OUTPUTS)


if __name__ == "__main__":                        # run only when started directly
    main()
