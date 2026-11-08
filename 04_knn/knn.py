# =====================================================================================================
# Project 4 - K-Nearest Neighbours: which tree grows here? (581,012 forest patches, Colorado)
# -----------------------------------------------------------------------------------------------------
# Every line has a comment saying WHY it is there. Read it alongside README.md.
# Tests:      ..\.venv\Scripts\python -m pytest -q
# Full run:   ..\.venv\Scripts\python knn.py            (about 10 minutes; --quick for the 5k sample)
# =====================================================================================================

import argparse                                   # command-line options
import json                                       # save metrics
import time                                       # KNN's cost is at PREDICTION time - we measure it
from pathlib import Path                          # file paths

import matplotlib                                 # plotting
matplotlib.use("Agg")                             # write images to files
import matplotlib.pyplot as plt                   # plotting functions
import numpy as np                                # arrays and maths
from sklearn.metrics import accuracy_score, balanced_accuracy_score   # overall and per-class-averaged accuracy
from sklearn.model_selection import train_test_split   # stratified split
from sklearn.neighbors import KNeighborsClassifier    # scikit-learn's KNN
from sklearn.preprocessing import StandardScaler  # (x - mean) / std for every feature

HERE = Path(__file__).resolve().parent                                        # this project's folder
FULL_DATA = HERE.parent / "_datasets" / "covertype" / "covertype.npz"         # all 581,012 rows
SAMPLE_DATA = HERE / "data" / "covertype_sample.npz"                          # 5,000 rows for tests
OUTPUTS = HERE / "outputs"                                                    # plots and metrics
CLASSES = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"]
NUMERIC = list(range(10))                         # columns 0-9: elevation, slope, distances, hillshade ...
                                                  # columns 10-53: 4 wilderness-area flags + 40 soil-type flags (0/1)


# ========================================================================= data
def load_data(path):
    """X (float32) and y (1..7)."""
    d = np.load(path)                             # compressed numpy arrays
    return d["X"], d["y"]                         # features and tree type


# ========================================================================= KNN from scratch
def pairwise_sq_dists(A, B):
    """Squared Euclidean distance between every row of A and every row of B, without a Python loop.

    ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b  -> one matrix multiplication does the expensive part.
    """
    a2 = (A ** 2).sum(axis=1)[:, None]            # ||a||^2 as a column (one per query row)
    b2 = (B ** 2).sum(axis=1)[None, :]            # ||b||^2 as a row (one per training row)
    d = a2 + b2 - 2.0 * A @ B.T                   # the identity above, for all pairs at once
    return np.maximum(d, 0.0)                     # rounding can give tiny negatives: clip to 0


class ScratchKNN:
    """Store the training data; to predict, find the k closest training rows and take a majority vote."""

    def __init__(self, k=5):
        self.k = k                                # how many neighbours vote

    def fit(self, X, y):
        self.X, self.y = np.asarray(X, dtype=np.float64), np.asarray(y)   # "training" = memorising: that's all
        self.classes_ = np.unique(self.y)         # the possible labels
        return self                               # sklearn convention

    def predict(self, X, chunk=2000):
        X = np.asarray(X, dtype=np.float64)       # same precision as the stored data
        out = np.empty(len(X), dtype=self.y.dtype)   # one prediction per query
        for s in range(0, len(X), chunk):         # in chunks: a full distance matrix would not fit in memory
            d = pairwise_sq_dists(X[s:s + chunk], self.X)   # distances from this chunk to every training row
            nn = np.argpartition(d, self.k - 1, axis=1)[:, :self.k]   # the k smallest per row (no full sort needed)
            votes = self.y[nn]                    # the labels of those neighbours
            counts = (votes[:, :, None] == self.classes_[None, None, :]).sum(axis=1)   # votes per class
            out[s:s + chunk] = self.classes_[counts.argmax(axis=1)]   # most votes wins; on a tie argmax picks the
                                                  # SMALLEST label - exactly scikit-learn's rule
        return out                                # predicted tree type per row


# ========================================================================= experiments
def scaling_effect(Xtr, ytr, Xva, yva, k=5):
    """KNN with raw features vs standardised features. Raw: elevation (metres) drowns out everything else."""
    raw = accuracy_score(yva, KNeighborsClassifier(k).fit(Xtr, ytr).predict(Xva))   # distances in raw units
    sc = StandardScaler().fit(Xtr[:, NUMERIC])   # learn means/stds on TRAIN only
    def scale(X):                                 # standardise the 10 numeric columns, keep the 0/1 flags
        out = X.copy(); out[:, NUMERIC] = sc.transform(X[:, NUMERIC]); return out
    scaled = accuracy_score(yva, KNeighborsClassifier(k).fit(scale(Xtr), ytr).predict(scale(Xva)))
    return {"raw": raw, "scaled": scaled}, scale  # also return the scaler function for the rest of the run


def k_sweep(Xtr, ytr, Xva, yva, ks=(1, 3, 5, 11, 21, 51)):
    """Validation accuracy for each k, with uniform votes and distance-weighted votes."""
    res = {}                                      # (k, weights) -> accuracy
    for w in ("uniform", "distance"):             # every neighbour counts equally / closer ones count more
        for k in ks:                              # small k = flexible (noisy), large k = smooth (blurry)
            m = KNeighborsClassifier(n_neighbors=k, weights=w, n_jobs=-1).fit(Xtr, ytr)   # all CPU cores
            res[f"{w}_k{k}"] = accuracy_score(yva, m.predict(Xva))   # score on validation
    return res


def learning_curve(Xtr, ytr, Xte, yte, sizes, k, weights):
    """More training data: accuracy keeps rising, and so does the time to predict."""
    rows = []                                     # one row per training size
    for n in sizes:                               # e.g. 10k -> 400k
        m = KNeighborsClassifier(n_neighbors=k, weights=weights, n_jobs=-1).fit(Xtr[:n], ytr[:n])   # instant
        t = time.perf_counter()                   # the slow part is predicting
        pred = m.predict(Xte)                     # search n stored rows for each test row
        rows.append({"n": n, "accuracy": round(accuracy_score(yte, pred), 4),
                     "balanced_accuracy": round(balanced_accuracy_score(yte, pred), 4),   # rare classes count equally
                     "predict_seconds": round(time.perf_counter() - t, 1)})
        print(" curve", rows[-1])                 # progress
    return rows


def algorithm_timing(Xtr, ytr, Xq, k=5):
    """Brute force vs KD-tree vs ball tree: same answers, different speed, depending on dimensions."""
    res = {}                                      # (features, algorithm) -> seconds
    for name, cols in (("10 numeric features", NUMERIC), ("all 54 features", list(range(Xtr.shape[1])))):
        for algo in ("brute", "kd_tree", "ball_tree"):   # the three search strategies
            m = KNeighborsClassifier(n_neighbors=k, algorithm=algo).fit(Xtr[:, cols], ytr)   # build the index
            t = time.perf_counter()               # time the queries only
            m.predict(Xq[:, cols])                # one search per query row
            res[f"{name} | {algo}"] = round(time.perf_counter() - t, 2)
    return res


def noise_features(Xtr, ytr, Xva, yva, k, extra=(0, 10, 50, 200)):
    """Curse of dimensionality: add random useless features and watch KNN fall apart."""
    rng = np.random.default_rng(0)                # reproducible noise
    res = {}                                      # number of noise columns -> accuracy
    for e in extra:                               # 0, 10, 50, 200 junk columns
        A = np.hstack([Xtr, rng.normal(size=(len(Xtr), e))]) if e else Xtr   # standard-normal junk, same scale
        B = np.hstack([Xva, rng.normal(size=(len(Xva), e))]) if e else Xva   # as the scaled real features
        res[e] = accuracy_score(yva, KNeighborsClassifier(k, n_jobs=-1).fit(A, ytr).predict(B))
    return res


def distance_concentration(dims=(2, 10, 50, 200, 1000), n=2000):
    """In high dimensions the nearest and farthest points are almost equally far: 'near' stops meaning anything."""
    rng = np.random.default_rng(0)                # reproducible
    out = {}                                      # dimensions -> (farthest / nearest) distance ratio
    for d in dims:                                # more and more dimensions
        X = rng.random((n, d))                    # random points in a unit cube
        dist = np.sqrt(pairwise_sq_dists(X[:1], X[1:]))[0]   # distances from one point to all others
        out[d] = round(float(dist.max() / dist.min()), 2)   # 1.0 would mean "everything is equally far"
    return out


# ========================================================================= plots
def plot_k(res, ks, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))      # one plot
    for w in ("uniform", "distance"):             # one line per voting rule
        ax.plot(ks, [res[f"{w}_k{k}"] for k in ks], "o-", label=f"{w} votes")
    ax.set_xscale("log"); ax.set_xticks(ks, [str(k) for k in ks])   # k spans 1..51: log axis
    ax.set(xlabel="k (number of neighbours)", ylabel="validation accuracy", title="Choosing k")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_curve(rows, path):
    n = [r["n"] for r in rows]                    # training sizes
    fig, ax1 = plt.subplots(figsize=(7, 4.5))     # two y axes
    ax1.plot(n, [r["accuracy"] for r in rows], "o-", color="tab:blue"); ax1.set(xlabel="training rows", ylabel="test accuracy")
    ax2 = ax1.twinx(); ax2.plot(n, [r["predict_seconds"] for r in rows], "s--", color="tab:red"); ax2.set_ylabel("seconds to predict")
    ax1.set_xscale("log"); ax1.set_title("KNN keeps improving with data - and keeps slowing down")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_noise(noise, conc, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))   # two panels
    ax[0].plot(list(noise), list(noise.values()), "o-"); ax[0].set(xlabel="random noise features added",
                                                                 ylabel="validation accuracy", title="Useless features hurt KNN")
    ax[1].plot(list(conc), list(conc.values()), "o-"); ax[1].set_xscale("log")
    ax[1].set(xlabel="dimensions", ylabel="farthest / nearest distance", title="Distances concentrate in high dimensions")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ========================================================================= run everything
def main():
    ap = argparse.ArgumentParser()                # options
    ap.add_argument("--quick", action="store_true")   # the 5k sample
    args = ap.parse_args()
    OUTPUTS.mkdir(exist_ok=True)                  # outputs/
    X, y = load_data(SAMPLE_DATA if args.quick else FULL_DATA)   # features and labels
    big = len(y) > 100000                         # full data?
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)   # 80/20
    Xfit, Xva, yfit, yva = train_test_split(Xtr, ytr, test_size=0.1 if big else 0.25, stratify=ytr, random_state=0)
    n_tune = 50000 if big else len(yfit)          # tune on 50k rows: KNN prediction is slow, and the ranking holds
    Xva, yva = Xva[:10000], yva[:10000]           # 10k validation rows are plenty to compare settings
    print(f"train {len(ytr):,}  test {len(yte):,}  (tuning on {n_tune:,} / validating on {len(yva):,})")
    res = {}                                      # every result
    res["scaling"], scale = scaling_effect(Xfit[:n_tune], yfit[:n_tune], Xva, yva)   # 1. raw vs scaled
    print("scaling", res["scaling"])
    Xtr_s, Xte_s, Xfit_s, Xva_s = scale(Xtr), scale(Xte), scale(Xfit), scale(Xva)   # use scaled features from now on
    ks = (1, 3, 5, 11, 21, 51)                    # 2. choose k and the voting rule
    res["k_sweep"] = k_sweep(Xfit_s[:n_tune], yfit[:n_tune], Xva_s, yva, ks)
    best = max(res["k_sweep"], key=res["k_sweep"].get)   # e.g. "distance_k5"
    weights, k = best.split("_k")[0], int(best.split("_k")[1])
    print("k sweep", {a: round(b, 4) for a, b in res["k_sweep"].items()}, "-> best", best)
    n_test = 20000 if big else len(yte)           # 3. learning curve on 20k test rows (prediction cost!)
    sizes = [10000, 30000, 100000, 300000, len(ytr)] if big else [500, 1000, 2000, len(ytr)]
    res["learning_curve"] = learning_curve(Xtr_s, ytr, Xte_s[:n_test], yte[:n_test], sizes, k, weights)
    res["timing"] = algorithm_timing(Xtr_s[:100000] if big else Xtr_s, ytr[:100000] if big else ytr, Xte_s[:2000])   # 4.
    print("timing", res["timing"])
    res["noise"] = noise_features(Xfit_s[:n_tune], yfit[:n_tune], Xva_s, yva, k)   # 5. curse of dimensionality
    res["concentration"] = distance_concentration()
    print("noise", res["noise"], "concentration", res["concentration"])
    sk = KNeighborsClassifier(n_neighbors=k).fit(Xfit_s[:5000], yfit[:5000])      # 6. scratch vs sklearn
    ours = ScratchKNN(k).fit(Xfit_s[:5000], yfit[:5000])
    res["scratch_agreement"] = float(np.mean(ours.predict(Xva_s[:2000]) == sk.predict(Xva_s[:2000])))
    print("scratch agreement", res["scratch_agreement"])
    plot_k(res["k_sweep"], ks, OUTPUTS / "k_sweep.png")   # plots
    plot_curve(res["learning_curve"], OUTPUTS / "learning_curve.png")
    plot_noise(res["noise"], res["concentration"], OUTPUTS / "curse_of_dimensionality.png")
    res["best"] = {"k": k, "weights": weights}    # remember the winner
    (OUTPUTS / "metrics.json").write_text(json.dumps(res, indent=2, default=str))
    print("saved to", OUTPUTS)


if __name__ == "__main__":                        # run only when started directly
    main()
