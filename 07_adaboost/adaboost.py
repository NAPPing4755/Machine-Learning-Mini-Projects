# =====================================================================================================
# Project 7 - AdaBoost: finding Higgs bosons in 11 million simulated particle collisions (UCI HIGGS)
# -----------------------------------------------------------------------------------------------------
# Every line has a comment saying WHY it is there. Read it alongside README.md.
# Tests:      ..\.venv\Scripts\python -m pytest -q
# Full run:   ..\.venv\Scripts\python adaboost.py       (about 20 minutes; --quick for the sample)
# =====================================================================================================

import argparse                                   # command-line options
import json                                       # save metrics
from pathlib import Path                          # file paths

from joblib import Parallel, delayed              # AdaBoost itself is sequential: run separate models in parallel

import matplotlib                                 # plotting
matplotlib.use("Agg")                             # write images to files
import matplotlib.pyplot as plt                   # plotting functions
import numpy as np                                # arrays and maths
import pandas as pd                               # tables
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier   # boosting, and a bagging comparison
from sklearn.metrics import accuracy_score, roc_auc_score   # metrics
from sklearn.tree import DecisionTreeClassifier   # the weak learner (a stump = depth-1 tree)

HERE = Path(__file__).resolve().parent                                    # this project's folder
FULL_DATA = HERE.parent / "_datasets" / "higgs" / "HIGGS.csv.gz"          # 11,000,000 collisions
SAMPLE_DATA = HERE / "data" / "higgs_sample.csv.gz"                       # 20,000 collisions for tests
OUTPUTS = HERE / "outputs"                                                # plots and metrics
LOW = ["lepton_pT", "lepton_eta", "lepton_phi", "missing_energy", "missing_energy_phi"] + \
      [f"jet{j}_{q}" for j in range(1, 5) for q in ("pt", "eta", "phi", "btag")]   # 21 raw detector measurements
HIGH = ["m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"]      # 7 masses physicists derive from them
COLUMNS = ["signal"] + LOW + HIGH                 # the file has no header: column 0 = label, then 28 features


# ========================================================================= data
def load(path, nrows=None, skiprows=None):
    """HIGGS rows as a DataFrame (label first). The last 500,000 rows of the full file are the official test set."""
    return pd.read_csv(path, header=None, names=COLUMNS, nrows=nrows, skiprows=skiprows, dtype=np.float32)


# ========================================================================= AdaBoost from scratch
class ScratchAdaBoost:
    """Discrete AdaBoost (SAMME, two classes) with stumps.

    Each round: fit a stump on WEIGHTED data -> measure its weighted error -> give it a vote
    alpha = lr * log((1 - err) / err) -> increase the weight of the points it got wrong -> renormalise.
    """

    def __init__(self, n_rounds=50, learning_rate=1.0, max_depth=1):
        self.n_rounds, self.lr, self.max_depth = n_rounds, learning_rate, max_depth   # settings

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)       # plain arrays; y in {0, 1}
        w = np.full(len(y), 1.0 / len(y))         # start: every point equally important
        self.stumps, self.alphas, self.weights_history = [], [], []   # the ensemble and a record of the weights
        for _ in range(self.n_rounds):            # add one weak learner per round
            stump = DecisionTreeClassifier(max_depth=self.max_depth, random_state=0)   # a weak learner
            stump.fit(X, y, sample_weight=w)      # it tries hardest on the currently heavy (hard) points
            wrong = stump.predict(X) != y         # which points it misclassifies
            err = w[wrong].sum() / w.sum()        # WEIGHTED error rate
            if err >= 0.5:                        # no better than a coin flip on the weighted data: stop
                break
            alpha = self.lr * np.log((1 - err) / max(err, 1e-12))   # accurate stumps get a big say
            w = w * np.exp(alpha * wrong)         # mistakes get heavier (correct points are unchanged)
            w = w / w.sum()                       # renormalise to a probability distribution
            self.stumps.append(stump); self.alphas.append(alpha)   # keep the stump and its vote
            self.weights_history.append(w.copy()) # to study where the weight goes
        return self

    def decision_function(self, X):
        votes = [a * (2 * s.predict(np.asarray(X)) - 1) for s, a in zip(self.stumps, self.alphas)]   # +alpha / -alpha
        return np.sum(votes, axis=0)              # the weighted vote; > 0 means "signal"

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)   # the sign of the vote


# ========================================================================= experiments
def boosting_curve(Xtr, ytr, Xte, yte, n=800, lr=0.5):
    """Train and test AUC after every round (staged predictions: one fit, every ensemble size)."""
    m = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=n, learning_rate=lr, random_state=0)
    m.fit(Xtr, ytr)                               # one fit; sklearn can replay the ensemble round by round
    test = [roc_auc_score(yte, s) for s in m.staged_decision_function(Xte)]   # AUC after 1, 2, ... n stumps
    train = [roc_auc_score(ytr, s) for s in m.staged_decision_function(Xtr)]
    return m, train, test                         # model + both curves


def _fit_auc(Xtr, ytr, Xte, yte, depth, lr, n):
    """One AdaBoost model's test AUC (a separate function so joblib can run several at once)."""
    m = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth), n_estimators=n, learning_rate=lr, random_state=0)
    return round(roc_auc_score(yte, m.fit(Xtr, ytr).decision_function(Xte)), 4)   # fit, then score


def depth_and_rate(Xtr, ytr, Xte, yte, n=300):
    """Weak learner depth and learning rate together (9 independent models, fitted in parallel)."""
    grid = [(d, lr) for d in (1, 2, 3) for lr in (0.1, 0.5, 1.0)]   # stump..depth-3 tree x small..large steps
    aucs = Parallel(n_jobs=-1)(delayed(_fit_auc)(Xtr, ytr, Xte, yte, d, lr, n) for d, lr in grid)   # one core each
    return {f"depth{d}_lr{lr}": a for (d, lr), a in zip(grid, aucs)}


def feature_sets(Xtr, ytr, Xte, yte, n=300):
    """Raw detector readings vs the physicists' engineered masses vs both."""
    sets = (("21 low-level (raw)", LOW), ("7 high-level (engineered)", HIGH), ("all 28", LOW + HIGH))   # 3 views
    aucs = Parallel(n_jobs=3)(delayed(_fit_auc)(Xtr[c], ytr, Xte[c], yte, 2, 0.5, n) for _, c in sets)   # in parallel
    return {name: a for (name, _), a in zip(sets, aucs)}


def label_noise(Xtr, ytr, Xte, yte, flip=0.1, n=200, seed=0):
    """Flip 10% of training labels. AdaBoost's exponential loss keeps chasing them; a random forest shrugs."""
    rng = np.random.default_rng(seed)             # reproducible
    noisy = ytr.copy(); flipped = rng.random(len(ytr)) < flip   # which labels get flipped
    noisy[flipped] = 1 - noisy[flipped]           # signal <-> background
    out = {}                                      # results
    for name, y in (("clean labels", ytr), (f"{int(flip * 100)}% flipped", noisy)):
        ada = ScratchAdaBoost(n_rounds=n, learning_rate=0.5, max_depth=2).fit(Xtr, y)   # our version: we see weights
        rfm = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=0).fit(Xtr, y)
        out[name] = {"adaboost_auc": round(roc_auc_score(yte, ada.decision_function(Xte)), 4),
                     "forest_auc": round(roc_auc_score(yte, rfm.predict_proba(Xte)[:, 1]), 4)}
        if name != "clean labels":                # where did AdaBoost put its attention?
            w = ada.weights_history[-1]           # final sample weights
            out[name]["weight_share_on_flipped"] = round(float(w[flipped].sum()), 4)   # vs 10% if weights stayed even
    return out


# ========================================================================= plots
def plot_curve(train, test, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))      # one plot
    r = np.arange(1, len(test) + 1)               # rounds
    ax.plot(r, train, label="train"); ax.plot(r, test, label="test")
    ax.set_xscale("log"); ax.set(xlabel="boosting rounds (stumps)", ylabel="ROC-AUC", title="AdaBoost: weak learners add up")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_weights(model, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))      # one plot
    for t in (0, 9, 49, len(model.weights_history) - 1):   # after round 1, 10, 50 and the last
        w = model.weights_history[t] * len(model.weights_history[t])   # 1.0 = the starting weight
        ax.hist(np.log10(w), bins=80, histtype="step", label=f"after round {t + 1}")
    ax.set(xlabel="log10(weight / starting weight)", ylabel="collisions", title="Weights spread out: hard points get heavy")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ========================================================================= run everything
def main():
    ap = argparse.ArgumentParser()                # options
    ap.add_argument("--quick", action="store_true")   # the 20k sample
    args = ap.parse_args()
    OUTPUTS.mkdir(exist_ok=True)                  # outputs/
    if args.quick:                                # sample: 15k train, 5k test
        df = load(SAMPLE_DATA); train, test = df.iloc[:15000], df.iloc[15000:]
    else:                                         # full: first 500k rows to train, the official last 500k to test
        train = load(FULL_DATA, nrows=500_000)
        test = load(FULL_DATA, skiprows=10_500_000)
    Xtr, ytr = train[LOW + HIGH], train["signal"].values.astype(int)   # features, labels
    Xte, yte = test[LOW + HIGH], test["signal"].values.astype(int)
    print(f"train {len(train):,}  test {len(test):,}  signal share {ytr.mean():.3f}")
    res = {"rows": {"train": len(train), "test": len(test)}}
    stump = DecisionTreeClassifier(max_depth=1).fit(Xtr, ytr)   # 1. how weak is one stump?
    res["single_stump"] = {"auc": round(roc_auc_score(yte, stump.predict_proba(Xte)[:, 1]), 4),
                           "accuracy": round(accuracy_score(yte, stump.predict(Xte)), 4),
                           "feature": (LOW + HIGH)[int(np.argmax(stump.feature_importances_))]}
    print("stump", res["single_stump"])
    model, train_curve, test_curve = boosting_curve(Xtr, ytr, Xte, yte)   # 2. adding stumps
    res["boosting_curve"] = {str(r): {"train": round(train_curve[r - 1], 4), "test": round(test_curve[r - 1], 4)}
                             for r in (1, 10, 50, 100, 200, 400, 800) if r <= len(test_curve)}
    print("curve", res["boosting_curve"])
    n_small = 100_000 if len(train) > 100_000 else len(train)   # experiments 3-5 use 100k training rows: AdaBoost
    Xs, ys = Xtr[:n_small], ytr[:n_small]         # can't use many cores, and the comparisons hold at this size
    res["depth_and_rate"] = depth_and_rate(Xs, ys, Xte, yte)   # 3. tuning (illustrative: scored on test)
    print("depth/rate", res["depth_and_rate"])
    res["feature_sets"] = feature_sets(Xs, ys, Xte, yte)   # 4. raw vs engineered physics features
    print("features", res["feature_sets"])
    ours = ScratchAdaBoost(n_rounds=100).fit(Xtr[:n_small], ytr[:n_small])
    ref = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100, learning_rate=1.0, random_state=0)
    ref.fit(Xtr[:n_small], ytr[:n_small])         # the same algorithm in scikit-learn
    res["scratch_agreement"] = float(np.mean(ours.predict(Xte) == ref.predict(Xte)))
    res["label_noise"] = label_noise(Xtr[:n_small], ytr[:n_small], Xte, yte)
    print("scratch agreement", res["scratch_agreement"], "noise", res["label_noise"])
    plot_curve(train_curve, test_curve, OUTPUTS / "boosting_curve.png")   # plots
    plot_weights(ours, OUTPUTS / "weights.png")
    (OUTPUTS / "metrics.json").write_text(json.dumps(res, indent=2))
    print("saved to", OUTPUTS)


if __name__ == "__main__":                        # run only when started directly
    main()
