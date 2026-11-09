# =====================================================================================================
# Project 8 - Gradient Boosting: forecasting daily sales for 1,115 Rossmann drugstores (1M store-days)
# -----------------------------------------------------------------------------------------------------
# Every line has a comment saying WHY it is there. Read it alongside README.md.
# Tests:      ..\.venv\Scripts\python -m pytest -q
# Full run:   ..\.venv\Scripts\python gradient_boosting.py      (about 10 minutes; --quick for the sample)
# =====================================================================================================

import argparse                                   # command-line options
import json                                       # save metrics
import time                                       # compare training speeds
from pathlib import Path                          # file paths

import matplotlib                                 # plotting
matplotlib.use("Agg")                             # write images to files
import matplotlib.pyplot as plt                   # plotting functions
import numpy as np                                # arrays and maths
import pandas as pd                               # tables
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor   # classic and histogram GB
from sklearn.tree import DecisionTreeRegressor    # the trees we boost from scratch

HERE = Path(__file__).resolve().parent                                        # this project's folder
DATA_DIR = HERE.parent / "_datasets" / "rossmann"                             # train.csv + store.csv from Kaggle
SAMPLE_DATA = HERE / "data" / "rossmann_sample.csv"                           # 30 stores, all days, already merged
OUTPUTS = HERE / "outputs"                                                    # plots and metrics
FEATURES = ["Store", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday", "StoreType", "Assortment",
            "CompetitionDistance", "competition_open_months", "Promo2", "promo2_active", "day", "month",
            "year", "weekofyear", "dayofyear", "store_avg", "store_dow_avg"]   # all known IN ADVANCE for a future
                                                  # date (Customers is not!)
CATEGORICAL = ["StateHoliday", "StoreType", "Assortment"]   # codes, not quantities (Store has 1,115 values: too many
                                                  # for native categories, so it gets target encoding instead)
HOLDOUT_DAYS = 42                                 # the last 6 weeks, as long as Kaggle's own test period


# ========================================================================= data
def load(data_dir=DATA_DIR):
    """Daily sales merged with store information."""
    train = pd.read_csv(data_dir / "train.csv", low_memory=False, parse_dates=["Date"])   # 1,017,209 store-days
    stores = pd.read_csv(data_dir / "store.csv")  # one row per store: type, assortment, competition, promo2
    return train.merge(stores, on="Store", how="left")   # attach store facts to every day


def add_features(df):
    """Calendar, competition and promotion features; keep only open days with sales. Returns a COPY."""
    out = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()   # closed days have 0 sales by definition (Kaggle
                                                  # ignores them in scoring), so the model shouldn't learn them
    d = out["Date"]                               # the calendar date
    out["day"], out["month"], out["year"] = d.dt.day, d.dt.month, d.dt.year   # paydays, seasons, trend
    out["weekofyear"], out["dayofyear"] = d.dt.isocalendar().week.astype(int), d.dt.dayofyear   # Christmas etc.
    opened = pd.to_datetime(dict(year=out["CompetitionOpenSinceYear"].fillna(1900), month=out["CompetitionOpenSinceMonth"].fillna(1), day=1))
    out["competition_open_months"] = ((d - opened).dt.days / 30.4).clip(lower=0)   # how long a competitor has existed
    out.loc[out["CompetitionOpenSinceYear"].isna(), "competition_open_months"] = np.nan   # unknown stays unknown
    promo_start = out["Promo2SinceYear"].fillna(9999) * 100 + out["Promo2SinceWeek"].fillna(0)   # yyyyww
    month_name = d.dt.strftime("%b").replace("Sep", "Sept")   # store.csv writes September as "Sept"
    in_interval = [m in str(iv).split(",") for m, iv in zip(month_name, out["PromoInterval"])]   # promo month?
    out["promo2_active"] = ((out["year"] * 100 + out["weekofyear"] >= promo_start) & np.array(in_interval)).astype(int)
    for c in ("StateHoliday", "StoreType", "Assortment"):   # letters -> integer codes for the tree models
        out[c] = out[c].astype(str).astype("category").cat.codes
    out["log_sales"] = np.log1p(out["Sales"])     # model log(1 + sales): stores differ 10x in size, and the
    return out                                    # metric is a PERCENTAGE error, which log space matches


def target_encode(train, other):
    """Each store's average log sales (overall and per weekday), learned from TRAIN rows only, attached to both.

    Computing it on all rows would leak the holdout's sales into its own features.
    """
    s = train.groupby("Store")["log_sales"].mean().rename("store_avg")   # how big the store is
    sd = train.groupby(["Store", "DayOfWeek"])["log_sales"].mean().rename("store_dow_avg")   # its weekly rhythm
    add = lambda d: d.drop(columns=["store_avg", "store_dow_avg"], errors="ignore").join(s, on="Store").join(sd, on=["Store", "DayOfWeek"])
    return add(train), add(other)                 # same mapping for both; unseen stores get NaN (handled natively)


def time_split(df, days=HOLDOUT_DAYS):
    """Everything before the last `days` days for training; the last 6 weeks as the holdout."""
    cut = df["Date"].max() - pd.Timedelta(days=days)   # e.g. 2015-06-19
    return target_encode(df[df["Date"] <= cut], df[df["Date"] > cut])   # (train, holdout), encoded from train only


def rmspe(y, pred):
    """Kaggle's metric: root mean squared PERCENTAGE error (a 10% miss counts the same for small and big stores)."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)   # plain arrays
    return float(np.sqrt(np.mean(((y - pred) / y) ** 2)))   # only open days with sales > 0, so no division by 0


# ========================================================================= gradient boosting from scratch
class ScratchGradientBoosting:
    """Squared-error gradient boosting: start from the mean, then repeatedly fit a small tree to the RESIDUALS
    (= the negative gradient of the squared loss) and add a shrunken copy of it."""

    def __init__(self, n_trees=100, learning_rate=0.1, max_depth=3):
        self.n_trees, self.lr, self.max_depth = n_trees, learning_rate, max_depth   # settings

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)   # plain arrays
        self.init_ = y.mean()                     # stage 0: the best constant prediction
        pred = np.full(len(y), self.init_)        # current predictions for every row
        self.trees, self.train_loss = [], []      # the ensemble, and how the loss falls
        for _ in range(self.n_trees):             # add trees one after another (sequential!)
            residual = y - pred                   # what is still unexplained = -d(loss)/d(pred) for squared loss
            tree = DecisionTreeRegressor(max_depth=self.max_depth, criterion="friedman_mse", random_state=0)
            tree.fit(X, residual)                 # a small tree learns to predict the current errors
            pred = pred + self.lr * tree.predict(X)   # take a small step in that direction (shrinkage)
            self.trees.append(tree)               # keep it
            self.train_loss.append(float(np.mean((y - pred) ** 2)))   # training MSE after this tree
        return self

    def predict(self, X):
        X = np.asarray(X, float)                  # plain array
        return self.init_ + self.lr * np.sum([t.predict(X) for t in self.trees], axis=0)   # start + all steps


# ========================================================================= experiments
def baselines(train, hold):
    """Two forecasts with no machine learning: the store's median, and the store x weekday x promo median."""
    s = train.groupby("Store")["Sales"].median()  # one number per store
    g = train.groupby(["Store", "DayOfWeek", "Promo"])["Sales"].median()   # per store, weekday and promo flag
    p1 = hold["Store"].map(s)                     # the store's typical day
    p2 = pd.Series(list(zip(hold["Store"], hold["DayOfWeek"], hold["Promo"]))).map(g).fillna(p1.reset_index(drop=True))
    return {"store_median": rmspe(hold["Sales"], p1), "store_weekday_promo_median": rmspe(hold["Sales"], p2.values)}


def classic_vs_histogram(train, hold, n=20000):
    """Same idea, two implementations: exact split search vs 255-bin histograms (the LightGBM/XGBoost trick)."""
    sub = train.sample(min(n, len(train)), random_state=0)   # the classic one is too slow for 800k rows
    Xs, Xh = sub[FEATURES].fillna(-1), hold[FEATURES].fillna(-1)   # the classic one can't take NaN (stores with no
                                                  # known competitor); -1 = "missing", for BOTH models to be fair
    out = {}                                      # results per model
    for name, m in (("classic GradientBoostingRegressor", GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=0)),
                    ("HistGradientBoostingRegressor", HistGradientBoostingRegressor(max_iter=200, max_depth=6, learning_rate=0.1, early_stopping=False, random_state=0))):
        t = time.perf_counter()                   # clock
        m.fit(Xs, sub["log_sales"])               # same rows, same features, same number of trees
        out[name] = {"seconds": round(time.perf_counter() - t, 1),
                     "rmspe": round(rmspe(hold["Sales"], np.expm1(m.predict(Xh))), 4)}
    return out


def hgb(lr=0.1, iters=2000, leaves=63, early=True, cat=True):
    """The main model: histogram gradient boosting with native categorical handling and early stopping."""
    return HistGradientBoostingRegressor(
        learning_rate=lr, max_iter=iters, max_leaf_nodes=leaves, min_samples_leaf=40,   # tree size controls
        l2_regularization=1.0,                    # shrink leaf values a little: less overfitting
        categorical_features=[FEATURES.index(c) for c in CATEGORICAL] if cat else None,   # Store 1..1115 as a
        early_stopping=early, validation_fraction=0.1, n_iter_no_change=50,   # category, not a number
        random_state=0)


def learning_rate_sweep(train, hold, rates=(0.3, 0.1, 0.03)):
    """Smaller steps need more trees (early stopping finds how many) and usually generalise better."""
    out = {}                                      # results per learning rate
    for lr in rates:
        t = time.perf_counter()                   # clock
        m = hgb(lr=lr, iters=5000).fit(train[FEATURES], train["log_sales"])   # stops when validation stalls
        out[str(lr)] = {"trees": int(m.n_iter_), "seconds": round(time.perf_counter() - t, 1),
                        "rmspe": round(rmspe(hold["Sales"], np.expm1(m.predict(hold[FEATURES]))), 4)}
        print("  lr", lr, out[str(lr)])
    return out


def leakage_and_targets(train, hold):
    """(a) raw sales instead of log sales; (b) adding Customers, which you would not know for a future day."""
    out = {}                                      # results
    raw = hgb().fit(train[FEATURES], train["Sales"])   # same model, untransformed target
    out["target = raw sales"] = round(rmspe(hold["Sales"], raw.predict(hold[FEATURES])), 4)
    leak = HistGradientBoostingRegressor(max_iter=1000, max_leaf_nodes=63, early_stopping=True, random_state=0,
                                         categorical_features=[FEATURES.index(c) for c in CATEGORICAL])
    leak.fit(train[FEATURES + ["Customers"]], train["log_sales"])   # Customers is only known AFTER the day
    out["with Customers (leak)"] = round(rmspe(hold["Sales"], np.expm1(leak.predict(hold[FEATURES + ["Customers"]]))), 4)
    return out


def random_vs_time_split(df, model_factory):
    """The same model scored on a random 6-week-sized sample vs on the real future 6 weeks."""
    train, hold = time_split(df)                  # honest: the future
    rnd_hold = df.sample(len(hold), random_state=0)   # dishonest: random days from the whole period
    rnd_train = df.drop(rnd_hold.index)           # everything else
    rnd_train, rnd_hold = target_encode(rnd_train, rnd_hold)   # same encoding rule
    honest = model_factory().fit(train[FEATURES], train["log_sales"])
    random = model_factory().fit(rnd_train[FEATURES], rnd_train["log_sales"])
    return {"time split (future 6 weeks)": round(rmspe(hold["Sales"], np.expm1(honest.predict(hold[FEATURES]))), 4),
            "random split": round(rmspe(rnd_hold["Sales"], np.expm1(random.predict(rnd_hold[FEATURES]))), 4)}


# ========================================================================= plots
def plot_store(hold, pred, path, store=1):
    """Actual vs predicted daily sales for one store over the 6-week holdout."""
    h = hold.assign(pred=pred)                    # attach predictions
    s = h[h["Store"] == store].sort_values("Date")   # one store, in date order
    fig, ax = plt.subplots(figsize=(10, 4))       # wide: a time series
    ax.plot(s["Date"], s["Sales"], "o-", label="actual"); ax.plot(s["Date"], s["pred"], "s--", label="predicted")
    ax.set(title=f"Store {store}: last 6 weeks", ylabel="daily sales (EUR)"); ax.legend()
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_lr(sweep, path):
    lrs = list(sweep)                             # learning rates
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))   # two panels
    ax[0].bar(lrs, [sweep[l]["trees"] for l in lrs]); ax[0].set(xlabel="learning rate", ylabel="trees (early stopping)")
    ax[1].bar(lrs, [sweep[l]["rmspe"] for l in lrs]); ax[1].set(xlabel="learning rate", ylabel="holdout RMSPE")
    fig.suptitle("Smaller steps: more trees, lower error"); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ========================================================================= run everything
def main():
    ap = argparse.ArgumentParser()                # options
    ap.add_argument("--quick", action="store_true")   # 30-store sample
    args = ap.parse_args()
    OUTPUTS.mkdir(exist_ok=True)                  # outputs/
    raw = pd.read_csv(SAMPLE_DATA, parse_dates=["Date"], low_memory=False) if args.quick else load()   # merged data
    df = add_features(raw)                        # features, open days only
    train, hold = time_split(df)                  # past -> last 6 weeks
    print(f"{df['Store'].nunique()} stores: train {len(train):,} days to {train['Date'].max().date()}, holdout {len(hold):,}")
    res = {"rows": {"train": len(train), "holdout": len(hold)}}
    res["baselines"] = baselines(train, hold)     # 1. what to beat
    print("baselines", res["baselines"])
    res["classic_vs_histogram"] = classic_vs_histogram(train, hold)   # 2. why histograms
    print("classic vs hist", res["classic_vs_histogram"])
    res["learning_rate"] = learning_rate_sweep(train, hold)   # 3. learning rate x trees
    best_lr = float(min(res["learning_rate"], key=lambda k: res["learning_rate"][k]["rmspe"]))
    final = hgb(lr=best_lr, iters=5000).fit(train[FEATURES], train["log_sales"])   # the chosen model
    pred = np.expm1(final.predict(hold[FEATURES]))   # back from log space
    res["final"] = {"learning_rate": best_lr, "trees": int(final.n_iter_), "rmspe": round(rmspe(hold["Sales"], pred), 4)}
    print("final", res["final"])
    res["targets_and_leak"] = leakage_and_targets(train, hold)   # 4. log target + the Customers leak
    print("targets/leak", res["targets_and_leak"])
    res["splits"] = random_vs_time_split(df, lambda: hgb(lr=0.1))   # 5. why time-based validation
    print("splits", res["splits"])
    small = train.sample(min(5000, len(train)), random_state=0)   # 6. scratch vs sklearn (classic, same settings)
    Xsm, Xho = small[FEATURES].fillna(-1), hold[FEATURES].fillna(-1)   # classic trees need complete data
    ours = ScratchGradientBoosting(n_trees=50, learning_rate=0.1, max_depth=3).fit(Xsm, small["log_sales"])
    ref = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=0).fit(Xsm, small["log_sales"])
    res["scratch_max_abs_diff"] = float(np.max(np.abs(ours.predict(Xho) - ref.predict(Xho))))
    print("scratch vs sklearn max diff", res["scratch_max_abs_diff"])
    plot_store(hold, pred, OUTPUTS / "store1_forecast.png", store=int(hold["Store"].iloc[0]))   # plots
    plot_lr(res["learning_rate"], OUTPUTS / "learning_rate.png")
    (OUTPUTS / "metrics.json").write_text(json.dumps(res, indent=2))
    print("saved to", OUTPUTS)


if __name__ == "__main__":                        # run only when started directly
    main()
