# =====================================================================================================
# Project 5 - Decision Trees: NYC taxi trips (2.96M rides, January 2024)
#   Regressor : how long will this trip take?           Classifier: will a card-paying passenger tip?
# -----------------------------------------------------------------------------------------------------
# Every line has a comment saying WHY it is there. Read it alongside README.md.
# Tests:      ..\.venv\Scripts\python -m pytest -q
# Full run:   ..\.venv\Scripts\python decision_trees.py        (about 5 minutes; --quick for the sample)
# =====================================================================================================

import argparse                                   # command-line options
import json                                       # save metrics
from pathlib import Path                          # file paths

import matplotlib                                 # plotting
matplotlib.use("Agg")                             # write images to files
import matplotlib.pyplot as plt                   # plotting functions
import numpy as np                                # arrays and maths
import pandas as pd                               # tables
from sklearn.linear_model import LinearRegression # a straight-line baseline to beat
from sklearn.metrics import mean_absolute_error, roc_auc_score, root_mean_squared_error   # error metrics
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree   # the models + a drawing tool

HERE = Path(__file__).resolve().parent                                            # this project's folder
FULL_DATA = HERE.parent / "_datasets" / "nyc_taxi" / "yellow_tripdata_2024-01.parquet"   # all trips
SAMPLE_DATA = HERE / "data" / "taxi_sample.parquet"                               # 20,000 trips for tests
OUTPUTS = HERE / "outputs"                                                        # plots and metrics
FEATURES = ["trip_distance", "PULocationID", "DOLocationID", "hour", "weekday", "passenger_count", "RatecodeID"]
TIP_FEATURES = ["fare_amount", "trip_distance", "duration_min", "PULocationID", "DOLocationID", "hour", "weekday",
                "passenger_count"]        # known when the ride ends, before the passenger decides on a tip
TEST_FROM = pd.Timestamp("2024-01-25")            # train on Jan 1-24, test on Jan 25-31 (the future)


# ========================================================================= data
def load_trips(path):
    """Read the raw TLC file and add duration (minutes), hour and weekday."""
    df = pd.read_parquet(path)                    # columnar file: fast to read
    df["duration_min"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60   # target
    df["hour"] = df["tpep_pickup_datetime"].dt.hour          # rush hour matters
    df["weekday"] = df["tpep_pickup_datetime"].dt.dayofweek  # 0 = Monday ... 6 = Sunday
    return df                                     # raw + derived columns


def clean(df):
    """Drop impossible records. Returns (clean_df, {reason: rows_removed}) so nothing disappears silently."""
    rules = {                                     # each rule: rows that FAIL it are removed
        "pickup outside January 2024": df["tpep_pickup_datetime"].between("2024-01-01", "2024-01-31 23:59:59"),
        "duration not 1-180 minutes": df["duration_min"].between(1, 180),   # 0 s trips or 3-day "rides"
        "distance not 0.1-100 miles": df["trip_distance"].between(0.1, 100),   # 312,722-mile trips exist
        "fare not $3-$500": df["fare_amount"].between(3, 500),     # negative fares are refunds/corrections
        "missing passenger count / rate code": df["passenger_count"].notna() & df["RatecodeID"].notna(),
        "speed above 80 mph": df["trip_distance"] / (df["duration_min"] / 60) <= 80,   # GPS/meter glitches
    }
    keep = pd.Series(True, index=df.index)        # start by keeping everything
    removed = {}                                  # count per reason (in order, so each row is counted once)
    for reason, ok in rules.items():
        removed[reason] = int((keep & ~ok).sum()) # rows still kept that this rule would drop
        keep &= ok                                # apply the rule
    return df[keep].copy(), removed               # clean copy + the audit trail


def time_split(df):
    """Train on Jan 1-24, test on Jan 25-31."""
    is_test = df["tpep_pickup_datetime"] >= TEST_FROM   # the last week of the month
    return df[~is_test], df[is_test]              # (train, test)


# ========================================================================= a regression tree from scratch (CART)
def best_split(X, y, min_leaf=1):
    """The (feature, threshold) that most reduces the sum of squared errors. Returns (feature, threshold, gain)."""
    n, best = len(y), (None, None, 0.0)           # no split found yet
    parent_sse = ((y - y.mean()) ** 2).sum()      # error if we just predicted the mean for everyone
    for j in range(X.shape[1]):                   # try every feature
        order = np.argsort(X[:, j], kind="stable")   # sort rows by this feature
        xs, ys = X[order, j], y[order]            # sorted values and their targets
        csum, csq = np.cumsum(ys), np.cumsum(ys ** 2)   # running sums make every split O(1) to evaluate
        for i in range(min_leaf, n - min_leaf + 1):   # split between row i-1 and row i
            if xs[i] == xs[i - 1]:                # can't split between equal values
                continue
            nl, nr = i, n - i                     # left and right sizes
            sl, sr = csum[i - 1], csum[-1] - csum[i - 1]   # sums of y on each side
            ql, qr = csq[i - 1], csq[-1] - csq[i - 1]      # sums of y^2 on each side
            sse = (ql - sl ** 2 / nl) + (qr - sr ** 2 / nr)   # SSE = sum(y^2) - (sum y)^2 / n, per side
            gain = parent_sse - sse               # how much error this split removes
            if gain > best[2] + 1e-12:            # strictly better than anything so far
                best = (j, (xs[i - 1] + xs[i]) / 2, gain)   # threshold halfway between the two values (like sklearn)
    return best


class ScratchRegressionTree:
    """Grow the tree recursively: split while depth allows and a split reduces error; leaves predict the mean."""

    def __init__(self, max_depth=3, min_leaf=1):
        self.max_depth, self.min_leaf = max_depth, min_leaf   # stopping rules (the main overfitting controls)

    def fit(self, X, y):
        self.root = self._grow(np.asarray(X, float), np.asarray(y, float), 0)   # build from the top
        return self

    def _grow(self, X, y, depth):
        if depth == self.max_depth or len(y) < 2 * self.min_leaf:   # too deep or too few rows: make a leaf
            return {"value": y.mean()}
        j, t, gain = best_split(X, y, self.min_leaf)   # the best question to ask here
        if j is None:                             # no split reduces the error: leaf
            return {"value": y.mean()}
        left = X[:, j] <= t                       # rows answering "yes"
        return {"feature": j, "threshold": t,     # an internal node: a question and two subtrees
                "left": self._grow(X[left], y[left], depth + 1), "right": self._grow(X[~left], y[~left], depth + 1)}

    def _predict_one(self, x, node):
        while "value" not in node:                # walk down until a leaf
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node["value"]                      # the leaf's mean

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in np.asarray(X, float)])   # one walk per row


# ========================================================================= experiments
def depth_sweep(train, test, depths=(1, 2, 4, 6, 8, 10, 14, 18, 24, None)):
    """Train vs test error as the tree grows: the textbook bias-variance picture on real data."""
    Xtr, ytr, Xte, yte = train[FEATURES], train["duration_min"], test[FEATURES], test["duration_min"]
    rows = []                                     # one row per depth
    for d in depths:                              # None = grow until every leaf is pure
        m = DecisionTreeRegressor(max_depth=d, random_state=0).fit(Xtr, ytr)   # fit
        rows.append({"depth": "full" if d is None else d, "leaves": int(m.get_n_leaves()),
                     "train_rmse": round(root_mean_squared_error(ytr, m.predict(Xtr)), 3),   # error on seen data
                     "test_rmse": round(root_mean_squared_error(yte, m.predict(Xte)), 3)})   # error on the future
        print(" depth", rows[-1])
    return rows


def leaf_sweep(train, test, leaves=(1, 10, 50, 200, 1000, 5000)):
    """min_samples_leaf: each leaf must average at least this many trips (smoother predictions)."""
    Xtr, ytr, Xte, yte = train[FEATURES], train["duration_min"], test[FEATURES], test["duration_min"]
    return {m: round(root_mean_squared_error(yte, DecisionTreeRegressor(min_samples_leaf=m, random_state=0)
                                             .fit(Xtr, ytr).predict(Xte)), 3) for m in leaves}


def baselines(train, test):
    """What a tree has to beat: predicting the average, and a straight line."""
    yte = test["duration_min"]                    # the truth
    mean_pred = np.full(len(yte), train["duration_min"].mean())   # same guess for every trip
    lin = LinearRegression().fit(train[FEATURES], train["duration_min"])   # duration = a*distance + b*hour + ...
    return {"predict_mean": {"rmse": root_mean_squared_error(yte, mean_pred), "mae": mean_absolute_error(yte, mean_pred)},
            "linear_regression": {"rmse": root_mean_squared_error(yte, lin.predict(test[FEATURES])),
                                  "mae": mean_absolute_error(yte, lin.predict(test[FEATURES]))}}


def tip_classifier(train, test):
    """Will a passenger tip? Done twice: on ALL trips (with payment type), then honestly on card trips only."""
    out = {}                                      # results
    for label, tr, te, feats in (
            ("all trips, payment_type as a feature", train, test, TIP_FEATURES + ["payment_type"]),
            ("card trips only (honest)", train[train["payment_type"] == 1], test[test["payment_type"] == 1], TIP_FEATURES)):
        ytr, yte = (tr["tip_amount"] > 0).astype(int), (te["tip_amount"] > 0).astype(int)   # tipped or not
        m = DecisionTreeClassifier(max_depth=10, min_samples_leaf=200, random_state=0).fit(tr[feats], ytr)
        p = m.predict_proba(te[feats])[:, 1]      # probability of a tip
        imp = dict(zip(feats, m.feature_importances_.round(3)))   # which features the tree used
        out[label] = {"auc": round(roc_auc_score(yte, p), 4), "tip_rate": round(float(yte.mean()), 4),
                      "rows": len(te), "top_feature": max(imp, key=imp.get), "importances": imp}
        print(" tip", label, {k: v for k, v in out[label].items() if k != "importances"})
    return out


# ========================================================================= plots
def plot_depth(rows, path):
    x = [str(r["depth"]) for r in rows]           # depth labels (including "full")
    fig, ax = plt.subplots(figsize=(8, 4.5))      # one plot
    ax.plot(x, [r["train_rmse"] for r in rows], "o-", label="train (Jan 1-24)")
    ax.plot(x, [r["test_rmse"] for r in rows], "s-", label="test (Jan 25-31)")
    ax.set(xlabel="max_depth", ylabel="RMSE (minutes)", title="Deeper trees memorise: train error -> 0, test error rises")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_small_tree(train, path):
    m = DecisionTreeRegressor(max_depth=3, random_state=0).fit(train[FEATURES], train["duration_min"])   # readable size
    fig, ax = plt.subplots(figsize=(18, 7))       # wide: 8 leaves
    plot_tree(m, feature_names=FEATURES, filled=True, rounded=True, precision=1, fontsize=9, ax=ax)   # the questions
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_importance(model, path):
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()   # total error reduction per feature
    fig, ax = plt.subplots(figsize=(7, 4)); imp.plot.barh(ax=ax)   # horizontal bars
    ax.set(xlabel="importance (share of total error reduction)", title="What the duration tree relies on")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ========================================================================= run everything
def main():
    ap = argparse.ArgumentParser()                # options
    ap.add_argument("--quick", action="store_true")   # the 20k-trip sample
    args = ap.parse_args()
    OUTPUTS.mkdir(exist_ok=True)                  # outputs/
    df, removed = clean(load_trips(SAMPLE_DATA if args.quick else FULL_DATA))   # load + clean (with an audit trail)
    print(f"{len(df):,} clean trips; removed:", removed)
    train, test = time_split(df)                  # past -> future
    print(f"train {len(train):,}  test {len(test):,}")
    res = {"cleaning": removed, "rows": {"train": len(train), "test": len(test)}}
    res["baselines"] = baselines(train, test)     # 1. what to beat
    print("baselines", res["baselines"])
    res["depth_sweep"] = depth_sweep(train, test) # 2. over/underfitting
    best = min(res["depth_sweep"], key=lambda r: r["test_rmse"])   # (in a real project: pick on validation data)
    res["leaf_sweep"] = leaf_sweep(train, test)   # 3. another way to stop overfitting
    print("leaf sweep", res["leaf_sweep"])
    final = DecisionTreeRegressor(max_depth=None if best["depth"] == "full" else best["depth"], random_state=0)
    final.fit(train[FEATURES], train["duration_min"])   # the chosen tree
    pred = final.predict(test[FEATURES])          # final predictions
    res["final_tree"] = {"depth": best["depth"], "rmse": root_mean_squared_error(test["duration_min"], pred),
                         "mae": mean_absolute_error(test["duration_min"], pred)}
    print("final", res["final_tree"])
    res["tips"] = tip_classifier(train, test)     # 4. the classifier, and the recording trap
    plot_depth(res["depth_sweep"], OUTPUTS / "depth_sweep.png")   # plots
    plot_small_tree(train, OUTPUTS / "tree_depth3.png")
    plot_importance(final, OUTPUTS / "importance.png")
    (OUTPUTS / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    print("saved to", OUTPUTS)


if __name__ == "__main__":                        # run only when started directly
    main()
