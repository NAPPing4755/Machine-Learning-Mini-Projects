# =====================================================================================================
# Project 6 - Random Forest: catching card fraud (284,807 real transactions, 0.17% fraud)
# -----------------------------------------------------------------------------------------------------
# Every line has a comment saying WHY it is there. Read it alongside README.md.
# Tests:      ..\.venv\Scripts\python -m pytest -q
# Full run:   ..\.venv\Scripts\python random_forest.py      (about 10 minutes; --quick for the sample)
# =====================================================================================================

import argparse                                   # command-line options
import json                                       # save metrics
from pathlib import Path                          # file paths

import matplotlib                                 # plotting
matplotlib.use("Agg")                             # write images to files
import matplotlib.pyplot as plt                   # plotting functions
import numpy as np                                # arrays and maths
import pandas as pd                               # tables
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier   # bagged trees, and random forests
from sklearn.inspection import permutation_importance   # importance measured by breaking a feature on test data
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score   # rare-class metrics
from sklearn.tree import DecisionTreeClassifier   # the building block

HERE = Path(__file__).resolve().parent                                        # this project's folder
FULL_DATA = HERE.parent / "_datasets" / "creditcard" / "creditcard.csv"       # all transactions
SAMPLE_DATA = HERE / "data" / "creditcard_sample.csv"                         # all frauds + 9,000 normal ones
OUTPUTS = HERE / "outputs"                                                    # plots and metrics
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]   # V1-V28: anonymised (PCA) by the bank; Amount in euros


# ========================================================================= data
def load(path):
    """Transactions sorted by time, exact duplicates removed."""
    df = pd.read_csv(path)                        # one row per card transaction
    df = df.drop_duplicates()                     # 1,081 exact copies: they would sit in train AND test
    return df.sort_values("Time").reset_index(drop=True)   # Time = seconds since the first transaction


def time_split(df, train_share=0.75):
    """First 75% of the 48 hours for training, the rest for testing (fraud patterns are judged on the future)."""
    cut = df["Time"].quantile(train_share)        # the moment that splits the rows 75/25
    return df[df["Time"] <= cut], df[df["Time"] > cut]   # (train, test)


# ========================================================================= a random forest from scratch
class ScratchRandomForest:
    """Many deep trees, each trained on a BOOTSTRAP sample (rows drawn with replacement) and allowed to look at only
    a random subset of features at every split. Prediction = the average of the trees' probabilities."""

    def __init__(self, n_trees=50, max_features="sqrt", min_samples_leaf=1, seed=0):
        self.n_trees, self.max_features, self.min_samples_leaf = n_trees, max_features, min_samples_leaf   # settings
        self.rng = np.random.default_rng(seed)    # reproducible bootstraps

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)       # plain arrays
        n = len(y)                                # rows
        self.trees, self.oob_sum, self.oob_cnt = [], np.zeros(n), np.zeros(n)   # trees + out-of-bag tallies
        for t in range(self.n_trees):             # grow each tree independently
            idx = self.rng.integers(0, n, n)      # bootstrap: n draws WITH replacement (~63% unique rows)
            tree = DecisionTreeClassifier(max_features=self.max_features,   # random feature subset at EVERY split
                                          min_samples_leaf=self.min_samples_leaf,
                                          random_state=int(self.rng.integers(1 << 30)))   # its own randomness
            tree.fit(X[idx], y[idx])              # a deep, overfitted, low-bias tree
            self.trees.append(tree)               # keep it
            oob = np.setdiff1d(np.arange(n), idx) # rows this tree never saw (~37%)
            if len(oob):                          # score them: a free validation set
                self.oob_sum[oob] += self._p1(tree, X[oob])   # accumulate this tree's fraud probability
                self.oob_cnt[oob] += 1            # how many trees voted on each row
        return self

    @staticmethod
    def _p1(tree, X):
        p = tree.predict_proba(X)                 # columns follow tree.classes_
        return p[:, list(tree.classes_).index(1)] if 1 in tree.classes_ else np.zeros(len(X))   # P(fraud)

    def oob_proba(self):
        """Each row's fraud probability averaged over only the trees that did NOT train on it."""
        return np.divide(self.oob_sum, self.oob_cnt, out=np.zeros_like(self.oob_sum), where=self.oob_cnt > 0)

    def predict_proba(self, X):
        return np.mean([self._p1(t, np.asarray(X)) for t in self.trees], axis=0)   # average of all trees


# ========================================================================= experiments
def metrics(y, p, amount):
    """Rare-event metrics: ranking quality, plus what an investigation team with a fixed budget would catch."""
    order = np.argsort(-p)                        # most suspicious first
    top = order[: max(1, int(0.001 * len(p)))]    # the team can review 0.1% of transactions (~70 a day here)
    return {"roc_auc": round(roc_auc_score(y, p), 4),   # overall ranking (looks great even for weak models)
            "pr_auc": round(average_precision_score(y, p), 4),   # precision vs recall: the honest rare-class score
            "precision_at_0.1pct": round(float(y[top].mean()), 4),   # of the cases reviewed, share really fraud
            "recall_at_0.1pct": round(float(y[top].sum() / y.sum()), 4),   # share of all fraud caught
            "fraud_euros_caught_at_0.1pct": round(float(amount[top][y[top] == 1].sum() / amount[y == 1].sum()), 4)}


def compare_ensembles(Xtr, ytr, Xte, yte, amt, n_trees=300):
    """One tree vs bagging (bootstrap only) vs random forest (bootstrap + random features)."""
    models = {                                    # the same building block, three ways
        "single deep tree": DecisionTreeClassifier(random_state=0),
        "bagging (bootstrap rows)": BaggingClassifier(DecisionTreeClassifier(), n_estimators=n_trees, n_jobs=-1, random_state=0),
        "random forest (+ random features)": RandomForestClassifier(n_estimators=n_trees, n_jobs=-1, random_state=0),
    }
    out = {}                                      # results per model
    for name, m in models.items():
        p = m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]   # train, then P(fraud) for each test transaction
        out[name] = metrics(yte, p, amt)          # score
        print(" ", name, out[name])
    return out


def trees_curve(Xtr, ytr, Xte, yte, sizes=(1, 5, 10, 25, 50, 100, 200, 400)):
    """More trees never overfit: OOB and test scores rise, then flatten."""
    rows = []                                     # one row per forest size
    for n in sizes:
        m = RandomForestClassifier(n_estimators=n, oob_score=n >= 25, n_jobs=-1, random_state=0).fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]            # test probabilities
        oob = m.oob_decision_function_[:, 1] if n >= 25 else None   # out-of-bag probabilities (need enough trees)
        rows.append({"trees": n, "test_pr_auc": round(average_precision_score(yte, p), 4),
                     "oob_pr_auc": None if oob is None else round(average_precision_score(ytr, np.nan_to_num(oob)), 4)})
        print("  trees", rows[-1])
    return rows


def max_features_sweep(Xtr, ytr, Xte, yte, options=(1, 3, "sqrt", 10, 0.5, None)):
    """How many features each split may consider: fewer = more diverse (less correlated) trees."""
    return {str(f): round(average_precision_score(
        yte, RandomForestClassifier(n_estimators=200, max_features=f, n_jobs=-1, random_state=0)
        .fit(Xtr, ytr).predict_proba(Xte)[:, 1]), 4) for f in options}


def class_weight_sweep(Xtr, ytr, Xte, yte, amt):
    """Does re-weighting the rare class help a forest?"""
    out = {}                                      # results per setting
    for cw in (None, "balanced", "balanced_subsample"):
        m = RandomForestClassifier(n_estimators=300, class_weight=cw, min_samples_leaf=1 if cw is None else 5,
                                   n_jobs=-1, random_state=0)   # weighted forests need bigger leaves to not overfit
        out[str(cw)] = metrics(yte, m.fit(Xtr, ytr).predict_proba(Xte)[:, 1], amt)
    return out


def importance_trap(Xtr, ytr, Xte, yte, seed=0):
    """Impurity importance (built-in) vs permutation importance, with a planted pure-noise column."""
    rng = np.random.default_rng(seed)             # reproducible noise
    A, B = Xtr.copy(), Xte.copy()                 # don't modify the caller's tables
    A["random_noise"] = rng.random(len(A))        # a continuous column with NO relation to fraud
    B["random_noise"] = rng.random(len(B))
    m = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0).fit(A, ytr)
    mdi = pd.Series(m.feature_importances_, index=A.columns)   # sum of impurity decrease per feature (on TRAIN)
    perm = permutation_importance(m, B, yte, scoring="average_precision", n_repeats=5, random_state=0, n_jobs=-1)
    perm = pd.Series(perm.importances_mean, index=B.columns)   # drop in test PR-AUC when the column is shuffled
    rank = lambda s: int(s.rank(ascending=False)["random_noise"])   # where the noise column lands
    return {"mdi_rank_of_noise": rank(mdi), "permutation_rank_of_noise": rank(perm),
            "mdi_top5": mdi.nlargest(5).round(4).to_dict(), "permutation_top5": perm.nlargest(5).round(4).to_dict(),
            "noise_mdi": round(float(mdi["random_noise"]), 4), "noise_permutation": round(float(perm["random_noise"]), 4)}, mdi, perm


# ========================================================================= plots
def plot_trees(rows, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))      # one plot
    ax.plot([r["trees"] for r in rows], [r["test_pr_auc"] for r in rows], "o-", label="test PR-AUC")
    oob = [(r["trees"], r["oob_pr_auc"]) for r in rows if r["oob_pr_auc"] is not None]
    ax.plot(*zip(*oob), "s--", label="out-of-bag PR-AUC (training data, free)")
    ax.set_xscale("log"); ax.set(xlabel="number of trees", ylabel="PR-AUC", title="More trees: better, then flat - never worse")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_pr(y, curves, path):
    fig, ax = plt.subplots(figsize=(6.5, 5))      # one plot
    for name, p in curves.items():                # one curve per model
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    ax.axhline(y.mean(), ls="--", color="grey", label=f"random ({y.mean():.4f})")
    ax.set(xlabel="recall (share of fraud caught)", ylabel="precision (alerts that are fraud)", title="Precision-recall")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_importance(mdi, perm, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))   # two panels
    for a, s, title in ((ax[0], mdi, "Impurity importance (train)"), (ax[1], perm, "Permutation importance (test)")):
        top = s.nlargest(12).sort_values()        # the 12 biggest
        a.barh(top.index, top.values, color=["tab:red" if i == "random_noise" else "tab:blue" for i in top.index])
        a.set_title(title)
    fig.suptitle("A pure-noise column (red) fools impurity importance, not permutation importance")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ========================================================================= run everything
def main():
    ap = argparse.ArgumentParser()                # options
    ap.add_argument("--quick", action="store_true")   # the sample
    args = ap.parse_args()
    OUTPUTS.mkdir(exist_ok=True)                  # outputs/
    df = load(SAMPLE_DATA if args.quick else FULL_DATA)   # transactions
    train, test = time_split(df)                  # earlier hours -> later hours
    Xtr, ytr, Xte, yte = train[FEATURES], train["Class"].values, test[FEATURES], test["Class"].values
    amt = test["Amount"].values                   # euros, to value what is caught
    print(f"train {len(train):,} ({ytr.sum()} fraud)   test {len(test):,} ({yte.sum()} fraud)")
    res = {"rows": {"train": len(train), "test": len(test), "train_fraud": int(ytr.sum()), "test_fraud": int(yte.sum())}}
    res["accuracy_of_always_legit"] = round(1 - yte.mean(), 5)   # the useless model's accuracy
    res["ensembles"] = compare_ensembles(Xtr, ytr, Xte, yte, amt)   # 1. tree vs bagging vs forest
    ours = ScratchRandomForest(n_trees=100 if len(train) > 50000 else 30).fit(Xtr, ytr)   # 2. from scratch
    res["scratch_forest"] = metrics(yte, ours.predict_proba(Xte), amt)
    res["scratch_forest"]["oob_pr_auc"] = round(average_precision_score(ytr, ours.oob_proba()), 4)
    print(" scratch", res["scratch_forest"])
    res["trees_curve"] = trees_curve(Xtr, ytr, Xte, yte)   # 3. number of trees
    res["max_features"] = max_features_sweep(Xtr, ytr, Xte, yte)   # 4. feature subsampling
    print(" max_features", res["max_features"])
    res["class_weight"] = class_weight_sweep(Xtr, ytr, Xte, yte, amt)   # 5. re-weighting
    print(" class_weight", res["class_weight"])
    res["importance_trap"], mdi, perm = importance_trap(Xtr, ytr, Xte, yte)   # 6. break it
    print(" importance", res["importance_trap"])
    tree_p = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]   # for the PR plot
    rf_p = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    plot_pr(yte, {"single tree": tree_p, "random forest": rf_p}, OUTPUTS / "pr_curves.png")   # plots
    plot_trees(res["trees_curve"], OUTPUTS / "trees_curve.png")
    plot_importance(mdi, perm, OUTPUTS / "importance_trap.png")
    (OUTPUTS / "metrics.json").write_text(json.dumps(res, indent=2))
    print("saved to", OUTPUTS)


if __name__ == "__main__":                        # run only when started directly
    main()
