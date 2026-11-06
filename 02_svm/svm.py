# =====================================================================================================
# Project 2 - Support Vector Machines: classifying 70,000 clothing images (Fashion-MNIST)
# -----------------------------------------------------------------------------------------------------
# Every line has a comment saying WHY it is there. Read it alongside README.md.
# Tests:      ..\.venv\Scripts\python -m pytest -q
# Full run:   ..\.venv\Scripts\python svm.py            (about 10 minutes; --quick for the 3k-image sample)
# =====================================================================================================

import argparse                                   # command-line options (--quick)
import json                                       # save metrics
import time                                       # measure training time (SVMs get slow - we prove it)
from pathlib import Path                          # file paths that work everywhere

import matplotlib                                 # plotting
matplotlib.use("Agg")                             # write images to files, no windows
import matplotlib.pyplot as plt                   # plotting functions
import numpy as np                                # arrays and maths
from sklearn.decomposition import PCA             # compress 784 pixels into fewer, informative numbers
from sklearn.kernel_approximation import Nystroem # approximate the RBF kernel so a LINEAR SVM can use it fast
from sklearn.metrics import accuracy_score, confusion_matrix   # how many correct; which classes get confused
from sklearn.pipeline import make_pipeline        # chain preprocessing + model
from sklearn.svm import SVC, LinearSVC            # kernel SVM (exact) and linear SVM (fast)

HERE = Path(__file__).resolve().parent                                            # this project's folder
FULL_DATA = HERE.parent / "_datasets" / "fashion_mnist" / "fashion_mnist.npz"     # all 70,000 images
SAMPLE_DATA = HERE / "data" / "fashion_mnist_sample.npz"                          # 3,000 + 1,000 for tests
OUTPUTS = HERE / "outputs"                                                        # plots and metrics
CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",                     # label 0..9 -> name
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# ========================================================================= data
def load_data(path):
    """Return X_train, y_train, X_test, y_test with pixels scaled to [0, 1]."""
    d = np.load(path)                             # .npz = several numpy arrays in one compressed file
    if "X" in d:                                  # the full file: 70,000 rows, official split = first 60k / last 10k
        X, y = d["X"], d["y"]                     # all images and labels
        Xtr, ytr, Xte, yte = X[:60000], y[:60000], X[60000:], y[60000:]   # the standard benchmark split
    else:                                         # the sample file already stores train and test separately
        Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    return (Xtr / 255.0, ytr, Xte / 255.0, yte)   # 0..255 -> 0..1: SVM distances need comparable scales


# ========================================================================= a linear SVM from scratch
def hinge_objective(w, b, X, y, lam):
    """lam/2 * ||w||^2 + mean(max(0, 1 - y * (Xw + b))), with y in {-1, +1}."""
    margins = y * (X @ w + b)                     # > 1: correct and outside the margin; < 0: wrong side
    hinge = np.maximum(0.0, 1.0 - margins)        # zero penalty once a point is safely past the margin
    return float(0.5 * lam * w @ w + hinge.mean())   # wide margin (small w) vs few violations, traded by lam


def hinge_subgradient(w, b, X, y, lam):
    """Subgradient of hinge_objective (the hinge has a kink at margin 1, so 'sub'-gradient)."""
    margins = y * (X @ w + b)                     # same margins as above
    active = margins < 1.0                        # only points inside the margin or misclassified push on w
    n = len(y)                                    # average over all rows
    grad_w = lam * w - (X[active].T @ y[active]) / n   # regulariser pulls w to 0; violators pull towards y*x
    grad_b = -float(y[active].sum()) / n          # the bias is not regularised
    return grad_w, grad_b                         # direction of steepest increase


def fit_linear_svm(X, y, lam=0.01, epochs=500, lr0=1.0):
    """Full-batch subgradient descent with a decaying step (lr0 / sqrt(t)). Returns (w, b, history)."""
    w, b, history = np.zeros(X.shape[1]), 0.0, []  # start from the zero hyperplane
    best = (w.copy(), b, np.inf)                  # subgradient steps can bounce: keep the best point seen
    for t in range(1, epochs + 1):                # one full pass per epoch
        gw, gb = hinge_subgradient(w, b, X, y, lam)   # which way is uphill
        step = lr0 / np.sqrt(t)                   # shrinking steps let it settle into the kink-y minimum
        w, b = w - step * gw, b - step * gb       # step downhill
        obj = hinge_objective(w, b, X, y, lam)    # how good is this point
        history.append(obj)                       # learning curve
        if obj < best[2]:                         # new best?
            best = (w.copy(), b, obj)             # remember it
    return best[0], best[1], history              # the best hyperplane found


# ========================================================================= experiments on the real data
def linear_svm(Xtr, ytr, Xte, yte, C=0.1):
    """A linear SVM on raw pixels (one-vs-rest over 10 classes). Fast even on 60,000 images."""
    t = time.perf_counter()                       # start the clock
    model = LinearSVC(C=C, dual=False, max_iter=2000).fit(Xtr, ytr)   # dual=False: faster when rows >> columns
    secs = time.perf_counter() - t                # training time
    return model, accuracy_score(yte, model.predict(Xte)), secs   # model, test accuracy, seconds


def rbf_scaling(Xtr, ytr, Xte, yte, sizes, pca_dims=50):
    """Train exact RBF SVMs on growing subsets: accuracy rises, but training time grows ~quadratically."""
    pca = PCA(n_components=pca_dims, random_state=0).fit(Xtr)   # 784 pixels -> 50 numbers (keeps ~86% variance)
    Ptr, Pte = pca.transform(Xtr), pca.transform(Xte)   # compress train and test the same way
    rows = []                                     # one result per subset size
    for n in sizes:                               # e.g. 2,500 -> 5,000 -> 10,000 -> 20,000
        t = time.perf_counter()                   # clock
        m = SVC(kernel="rbf", C=10, gamma="scale").fit(Ptr[:n], ytr[:n])   # exact kernel SVM on n images
        secs = time.perf_counter() - t            # training time for this n
        acc = accuracy_score(yte, m.predict(Pte)) # test accuracy
        rows.append({"n": n, "seconds": round(secs, 1), "accuracy": round(acc, 4), "support_vectors": int(m.n_support_.sum())})
        print(" rbf", rows[-1])                   # progress
    return rows                                   # for the README table and the timing plot


def tune_rbf(Xtr, ytr, Cs=(1, 10, 100), gammas=(0.003, 0.03, 0.3), n_fit=8000, n_val=4000, pca_dims=50):
    """Grid over C and gamma, scored on a VALIDATION slice of the training set (the test set stays unseen)."""
    pca = PCA(n_components=pca_dims, random_state=0).fit(Xtr[:n_fit])   # fit PCA on the fitting slice only
    Pf, Pv = pca.transform(Xtr[:n_fit]), pca.transform(Xtr[n_fit:n_fit + n_val])   # fit and validation slices
    yf, yv = ytr[:n_fit], ytr[n_fit:n_fit + n_val]   # their labels
    grid = np.zeros((len(Cs), len(gammas)))       # validation accuracy for every (C, gamma)
    train_acc = np.zeros_like(grid)               # training accuracy, to SEE overfitting
    for i, C in enumerate(Cs):                    # each penalty for margin violations
        for j, g in enumerate(gammas):            # each kernel width
            m = SVC(kernel="rbf", C=C, gamma=g).fit(Pf, yf)   # one model per cell
            grid[i, j] = accuracy_score(yv, m.predict(Pv))     # how it generalises
            train_acc[i, j] = accuracy_score(yf, m.predict(Pf))   # how well it memorises
    i, j = np.unravel_index(grid.argmax(), grid.shape)   # the best cell
    return {"C": Cs[i], "gamma": gammas[j]}, grid, train_acc   # best params + both grids for the heatmap


def final_rbf(Xtr, ytr, Xte, yte, C, gamma, n=30000, pca_dims=50):
    """The tuned exact RBF SVM on a large subset (all 60k would take very long: see the scaling result)."""
    model = make_pipeline(PCA(n_components=pca_dims, random_state=0), SVC(kernel="rbf", C=C, gamma=gamma))
    t = time.perf_counter()                       # clock
    model.fit(Xtr[:n], ytr[:n])                   # PCA + SVM trained together
    secs = time.perf_counter() - t                # training time
    pred = model.predict(Xte)                     # predictions on all 10,000 test images
    return model, accuracy_score(yte, pred), secs, pred   # keep predictions for the confusion matrix


def nystroem_svm(Xtr, ytr, Xte, yte, gamma, C=1.0, components=3000, pca_dims=50):
    """Kernel APPROXIMATION: map images into 3,000 random-feature dimensions, then a fast linear SVM on ALL 60k."""
    model = make_pipeline(PCA(n_components=pca_dims, random_state=0),            # compress first
                          Nystroem(kernel="rbf", gamma=gamma, n_components=components, random_state=0),  # approx kernel
                          LinearSVC(C=C, dual=False, max_iter=3000))             # linear SVM in that space
    t = time.perf_counter()                       # clock
    model.fit(Xtr, ytr)                           # scales linearly with the number of images
    secs = time.perf_counter() - t                # training time
    return model, accuracy_score(yte, model.predict(Xte)), secs   # model, accuracy, seconds


# ========================================================================= plots
def plot_confusion(yte, pred, path):
    """Which classes get mistaken for which (rows = truth, columns = prediction)."""
    cm = confusion_matrix(yte, pred, normalize="true")   # each row sums to 1: share of that class
    fig, ax = plt.subplots(figsize=(8, 7))        # square-ish figure
    im = ax.imshow(cm, cmap="Blues")              # darker = more
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right"); ax.set_yticks(range(10), CLASSES)   # class names
    for i in range(10):                           # write the percentage in every cell
        for j in range(10):
            ax.text(j, i, f"{cm[i, j]:.0%}" if cm[i, j] >= 0.01 else "", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
    ax.set(xlabel="Predicted", ylabel="True", title="Confusion matrix (tuned RBF SVM)")   # labels
    fig.colorbar(im); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)   # save and free memory


def plot_grid(grid, train_acc, Cs, gammas, path):
    """Validation vs training accuracy over (C, gamma): the overfitting picture."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))   # two heatmaps side by side
    for k, (g, title) in enumerate(((train_acc, "Training accuracy"), (grid, "Validation accuracy"))):
        im = ax[k].imshow(g, cmap="viridis", vmin=0.5, vmax=1.0)   # same colour scale on both
        ax[k].set_xticks(range(len(gammas)), [str(x) for x in gammas]); ax[k].set_yticks(range(len(Cs)), [str(x) for x in Cs])
        ax[k].set(xlabel="gamma (kernel width: larger = more wiggly)", ylabel="C (penalty for errors)", title=title)
        for i in range(len(Cs)):                  # annotate every cell
            for j in range(len(gammas)):
                ax[k].text(j, i, f"{g[i, j]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def plot_scaling(rows, path):
    """Training time and accuracy against the number of training images."""
    n = [r["n"] for r in rows]                    # x axis
    fig, ax1 = plt.subplots(figsize=(7, 4.5))     # one plot, two y axes
    ax1.plot(n, [r["seconds"] for r in rows], "o-", color="tab:red"); ax1.set(xlabel="training images", ylabel="seconds")
    ax2 = ax1.twinx(); ax2.plot(n, [r["accuracy"] for r in rows], "s--", color="tab:blue"); ax2.set_ylabel("test accuracy")
    ax1.set_title("Exact RBF SVM: time grows much faster than accuracy"); fig.tight_layout()
    fig.savefig(path, dpi=120); plt.close(fig)    # save and free memory


# ========================================================================= run everything
def main():
    ap = argparse.ArgumentParser()                # options
    ap.add_argument("--quick", action="store_true", help="use the 3k-image sample (for a fast check)")
    args = ap.parse_args()                        # read them
    OUTPUTS.mkdir(exist_ok=True)                  # make sure outputs/ exists
    Xtr, ytr, Xte, yte = load_data(SAMPLE_DATA if args.quick else FULL_DATA)   # pixels in [0, 1]
    big = len(Xtr) > 10000                        # full data or the sample?
    print(f"train {len(Xtr):,} images, test {len(Xte):,}")
    res = {}                                      # every result
    _, acc, secs = linear_svm(Xtr, ytr, Xte, yte) # 1. linear baseline on raw pixels
    res["linear"] = {"accuracy": acc, "seconds": round(secs, 1)}; print("linear", res["linear"])
    sizes = [2500, 5000, 10000, 20000] if big else [500, 1000, 2000]   # 2. how exact kernel SVMs scale
    res["rbf_scaling"] = rbf_scaling(Xtr, ytr, Xte, yte, sizes)
    Cs, gammas = (1, 10, 100), (0.003, 0.03, 0.3) # 3. tune on a validation slice
    best, grid, train_acc = tune_rbf(Xtr, ytr, Cs, gammas, n_fit=8000 if big else 1500, n_val=4000 if big else 1000)
    res["tuning"] = {"best": best, "val_grid": grid.round(4).tolist(), "train_grid": train_acc.round(4).tolist()}
    print("best", best)
    _, acc, secs, pred = final_rbf(Xtr, ytr, Xte, yte, best["C"], best["gamma"], n=30000 if big else len(Xtr))   # 4.
    res["rbf_final"] = {"accuracy": acc, "seconds": round(secs, 1), "train_images": 30000 if big else len(Xtr)}
    print("rbf final", res["rbf_final"])
    _, acc, secs = nystroem_svm(Xtr, ytr, Xte, yte, best["gamma"], components=3000 if big else 500)   # 5. approx
    res["nystroem"] = {"accuracy": acc, "seconds": round(secs, 1), "train_images": len(Xtr)}; print("nystroem", res["nystroem"])
    plot_confusion(yte, pred, OUTPUTS / "confusion.png")         # plots for the README
    plot_grid(grid, train_acc, Cs, gammas, OUTPUTS / "c_gamma_grid.png")
    plot_scaling(res["rbf_scaling"], OUTPUTS / "scaling.png")
    (OUTPUTS / "metrics.json").write_text(json.dumps(res, indent=2))   # save all numbers
    print("saved to", OUTPUTS)


if __name__ == "__main__":                        # run only when started directly
    main()
