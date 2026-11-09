# =====================================================================================================
# Project 10 - Capstone: every algorithm in this series on one hard problem (IEEE-CIS fraud, 590,540 transactions)
# -----------------------------------------------------------------------------------------------------
# Every line has a comment saying WHY it is there. Read it alongside README.md.
# Sample:     ..\.venv\Scripts\python make_sample.py      (once: competition data can't be redistributed)
# Tests:      ..\.venv\Scripts\python -m pytest -q
# Full run:   ..\.venv\Scripts\python capstone.py         (about 30 minutes; --quick for the sample)
# =====================================================================================================

import argparse                                   # command-line options
import json                                       # save metrics
import time                                       # training and prediction time per algorithm
from pathlib import Path                          # file paths

import matplotlib                                 # plotting
matplotlib.use("Agg")                             # write images to files
import matplotlib.pyplot as plt                   # plotting functions
import numpy as np                                # arrays and maths
import pandas as pd                               # tables
import xgboost as xgb                             # Project 9's library
from sklearn.decomposition import PCA             # compress features for KNN and SVM (Projects 2 and 4)
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer          # fill gaps for models that can't handle NaN
from sklearn.linear_model import LogisticRegression   # Project 1
from sklearn.metrics import average_precision_score, roc_auc_score   # rare-class metrics
from sklearn.naive_bayes import GaussianNB        # Project 3
from sklearn.neighbors import KNeighborsClassifier   # Project 4
from sklearn.pipeline import make_pipeline        # preprocessing + model
from sklearn.preprocessing import StandardScaler  # scaling for distance/linear models
from sklearn.svm import LinearSVC                 # Project 2 (linear: kernel SVMs can't take 470k rows)
from sklearn.tree import DecisionTreeClassifier   # Project 5

HERE = Path(__file__).resolve().parent                                        # this project's folder
DATA_DIR = HERE.parent / "_datasets" / "ieee_fraud"                           # Kaggle's train files
SAMPLE_DIR = HERE / "data"                                                    # 30k transactions for tests
OUTPUTS = HERE / "outputs"                                                    # plots and metrics


# ========================================================================= data
def load(data_dir):
    """Transactions joined with device/identity info (only ~24% of transactions have it), sorted by time."""
    t = pd.read_csv(data_dir / "train_transaction.csv")   # 394 columns: amount, card, address, C/D/M/V features
    i = pd.read_csv(data_dir / "train_identity.csv")      # 41 columns: device type, browser, id_xx
    df = t.merge(i, on="TransactionID", how="left").sort_values("TransactionDT").reset_index(drop=True)
    for c in df.columns:                          # text -> category codes (NaN stays NaN as code -1 -> NaN below)
        if df[c].dtype == object or str(df[c].dtype).startswith(("str", "string")):
            codes = df[c].astype("category").cat.codes.astype("float32")
            df[c] = codes.where(codes >= 0)       # -1 means missing: keep it as NaN
    num = df.select_dtypes("number").columns      # everything is numeric now
    df[num] = df[num].astype("float32")           # halve the memory (590k x 434 floats)
    return df


def add_features(df):
    """The famous IEEE-CIS insight: fraud is committed by PEOPLE, so identify the person behind each card.

    D1 = days since the card was first used; TransactionDT/86400 - D1 = the day the card started, which is
    constant for one card holder. card1 + addr1 + that start day approximates one client ("uid").
    """
    out = df.copy()                               # never modify the caller's table
    day = out["TransactionDT"] / 86400            # transaction day
    out["card_start_day"] = np.floor(day - out["D1"])   # same value for every transaction of one card holder
    part = lambda c: out[c].fillna(-1).astype("int64").astype(str)   # missing -> "-1" (pandas would keep NaN,
    uid = part("card1") + "_" + part("addr1") + "_" + part("card_start_day")   # and a NaN id groups nothing)
    out["uid_count"] = uid.map(uid.value_counts()).astype("float32")   # how active this client is
    grp = out.groupby(uid)["TransactionAmt"]      # per-client spending pattern
    out["uid_amt_mean"] = grp.transform("mean").astype("float32")    # their usual amount
    out["amt_vs_uid_mean"] = (out["TransactionAmt"] / out["uid_amt_mean"]).astype("float32")   # unusual for them?
    out["uid_P_email_n"] = out.groupby(uid)["P_emaildomain"].transform("nunique").astype("float32")   # e-mails used
    out["amt_cents"] = (out["TransactionAmt"] - np.floor(out["TransactionAmt"])).astype("float32")   # 0.99 vs .00:
    out["hour"] = ((out["TransactionDT"] / 3600) % 24).astype("float32")   # foreign currency conversions show
    return out


def time_split(df, share=0.8):
    """First 80% of the 6 months to train, the last 20% to test (Kaggle's own test set was also later in time)."""
    cut = int(len(df) * share)                    # rows are sorted by time
    return df.iloc[:cut], df.iloc[cut:]           # (train, test)


def features(df):
    return [c for c in df.columns if c not in ("isFraud", "TransactionID", "TransactionDT")]   # ids/time excluded


# ========================================================================= the line-up
def lineup(n_rows):
    """Every algorithm from Projects 1-9, each with the preprocessing it needs. Slow ones get a subsample."""
    imp = lambda: SimpleImputer(strategy="median")   # most sklearn models can't take NaN
    sub = min(n_rows, 100_000)                    # cap for algorithms that don't scale (stated in the results)
    return [   # (name, model, max training rows)
        ("1 Logistic regression", make_pipeline(imp(), StandardScaler(), LogisticRegression(C=0.1, max_iter=500)), n_rows),
        ("2 Linear SVM", make_pipeline(imp(), StandardScaler(), LinearSVC(C=0.01, dual=False)), sub),
        ("3 Gaussian naive Bayes", make_pipeline(imp(), StandardScaler(), GaussianNB()), n_rows),
        ("4 KNN (k=25, PCA-30)", make_pipeline(imp(), StandardScaler(), PCA(30, random_state=0),
                                                KNeighborsClassifier(25, weights="distance", n_jobs=-1)), min(n_rows, 50_000)),
        ("5 Decision tree", DecisionTreeClassifier(min_samples_leaf=100, random_state=0), n_rows),
        ("6 Random forest", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                                   n_jobs=-1, random_state=0), n_rows),
        ("7 AdaBoost (depth-2)", make_pipeline(imp(), AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),   # AdaBoost
                                               n_estimators=300, learning_rate=0.5, random_state=0)), sub),   # rejects NaN
        ("8 Hist gradient boosting", HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, max_leaf_nodes=127,
                                                                    random_state=0), n_rows),
        ("9 XGBoost", xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=9, subsample=0.8,
                                        colsample_bytree=0.5, tree_method="hist", eval_metric="auc", n_jobs=-1,
                                        random_state=0), n_rows),
    ]


def score_model(model, X, y):
    """Probability-like scores whatever the model offers (LinearSVC has no predict_proba)."""
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)


def compare_all(train, test, cols):
    """Same split, same features, same metrics for every algorithm."""
    Xtr, ytr, Xte, yte = train[cols], train["isFraud"].values, test[cols], test["isFraud"].values
    rows = []                                     # one result per algorithm
    for name, model, n in lineup(len(train)):
        idx = np.arange(len(train)) if n >= len(train) else np.sort(np.random.default_rng(0).choice(len(train), n, replace=False))
        t = time.perf_counter()                   # clock
        model.fit(Xtr.iloc[idx], ytr[idx])        # train (on a subsample if the algorithm can't scale)
        fit_s = time.perf_counter() - t
        t = time.perf_counter()
        s = score_model(model, Xte, yte)          # score every test transaction
        rows.append({"model": name, "train_rows": int(len(idx)), "fit_seconds": round(fit_s, 1),
                     "predict_seconds": round(time.perf_counter() - t, 1),
                     "roc_auc": round(roc_auc_score(yte, s), 4), "pr_auc": round(average_precision_score(yte, s), 4)})
        print("  ", rows[-1])
    return rows


def feature_engineering_gain(train_raw, test_raw, train_fe, test_fe):
    """XGBoost with and without the client-identification features."""
    out = {}                                      # results
    for name, tr, te in (("raw features", train_raw, test_raw), ("+ client (uid) features", train_fe, test_fe)):
        cols = features(tr)
        m = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=9, subsample=0.8, colsample_bytree=0.5,
                              tree_method="hist", n_jobs=-1, random_state=0).fit(tr[cols], tr["isFraud"])
        s = m.predict_proba(te[cols])[:, 1]       # test probabilities
        out[name] = {"features": len(cols), "roc_auc": round(roc_auc_score(te["isFraud"], s), 4),
                     "pr_auc": round(average_precision_score(te["isFraud"], s), 4)}
    return out


def random_vs_time(df):
    """The same XGBoost scored on a random 20% vs the real last 20%: why time-based validation matters."""
    cols = features(df)
    rnd = np.random.default_rng(0).random(len(df)) < 0.2   # a random 20% "test"
    out = {}
    for name, tr, te in (("time split (last 20%)", *time_split(df)), ("random split", df[~rnd], df[rnd])):
        m = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=9, subsample=0.8, colsample_bytree=0.5,
                              tree_method="hist", n_jobs=-1, random_state=0).fit(tr[cols], tr["isFraud"])
        out[name] = round(roc_auc_score(te["isFraud"], m.predict_proba(te[cols])[:, 1]), 4)
    return out


# ========================================================================= plots
def plot_lineup(rows, path):
    r = sorted(rows, key=lambda x: x["roc_auc"])  # worst to best
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))   # two panels
    ax[0].barh([x["model"] for x in r], [x["roc_auc"] for x in r]); ax[0].set(xlim=(0.5, 1.0), xlabel="test ROC-AUC")
    ax[1].barh([x["model"] for x in r], [x["pr_auc"] for x in r], color="tab:orange"); ax[1].set(xlabel="test PR-AUC")
    ax[1].set_yticklabels([])                     # names already on the left panel
    fig.suptitle("Nine algorithms, one fraud problem (same split, same features)"); fig.tight_layout()
    fig.savefig(path, dpi=120); plt.close(fig)


# ========================================================================= run everything
def main():
    ap = argparse.ArgumentParser()                # options
    ap.add_argument("--quick", action="store_true")   # the 30k sample
    args = ap.parse_args()
    OUTPUTS.mkdir(exist_ok=True)                  # outputs/
    raw = load(SAMPLE_DIR if args.quick else DATA_DIR)   # transactions + identity
    fe = add_features(raw)                        # + client-identification features
    tr_raw, te_raw = time_split(raw); tr, te = time_split(fe)   # identical rows in both versions
    print(f"train {len(tr):,}  test {len(te):,}  fraud rate {raw['isFraud'].mean():.4f}  features {len(features(fe))}")
    res = {"rows": {"train": len(tr), "test": len(te)}, "fraud_rate": round(float(raw["isFraud"].mean()), 4)}
    res["lineup"] = compare_all(tr, te, features(fe))   # 1. every algorithm
    res["feature_engineering"] = feature_engineering_gain(tr_raw, te_raw, tr, te)   # 2. the uid insight
    print("fe", res["feature_engineering"])
    res["splits"] = random_vs_time(fe)            # 3. validation design
    print("splits", res["splits"])
    plot_lineup(res["lineup"], OUTPUTS / "lineup.png")
    (OUTPUTS / "metrics.json").write_text(json.dumps(res, indent=2))
    print("saved to", OUTPUTS)


if __name__ == "__main__":                        # run only when started directly
    main()
