# =====================================================================================================
# Project 3 - Naive Bayes: spam filtering on the original Enron-Spam corpus (6 mailboxes, 30k unique e-mails)
# -----------------------------------------------------------------------------------------------------
# Every line has a comment saying WHY it is there. Read it alongside README.md.
# Data:       ..\.venv\Scripts\python build_dataset.py    (once: downloads the original archives, builds the CSV)
# Tests:      ..\.venv\Scripts\python -m pytest -q
# Full run:   ..\.venv\Scripts\python naive_bayes.py      (about 2 minutes; --quick for the sample)
# =====================================================================================================

import argparse                                   # command-line options
import json                                       # save metrics
from pathlib import Path                          # file paths

import matplotlib                                 # plotting
matplotlib.use("Agg")                             # write images to files
import matplotlib.pyplot as plt                   # plotting functions
import numpy as np                                # arrays and maths
import pandas as pd                               # tables
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer   # text -> word counts / TF-IDF
from sklearn.linear_model import LogisticRegression   # a discriminative model to compare against
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score   # metrics
from sklearn.model_selection import train_test_split   # random stratified split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB   # the NB family

HERE = Path(__file__).resolve().parent                                           # this project's folder
FULL_DATA = HERE.parent / "_datasets" / "enron_aueb" / "enron_spam_aueb.csv"     # built by build_dataset.py
SAMPLE_DATA = HERE / "data" / "enron_sample.csv"                                 # 2,000 e-mails for tests
OUTPUTS = HERE / "outputs"                                                       # plots and metrics


# ========================================================================= data
def load_emails(path):
    """DataFrame with text (subject + body), spam (1/0) and mailbox, exact duplicates removed."""
    df = pd.read_csv(path, keep_default_na=False) # one row per e-mail; empty fields stay "" instead of NaN
    df["text"] = df["subject"] + " " + df["body"] # the subject line carries a lot of the signal
    df["spam"] = (df["label"] == "spam").astype(int)   # the target: 1 = spam, 0 = ham (legitimate)
    df = df.drop_duplicates("text")               # spam campaigns repeat word for word: without this, copies of one
                                                  # e-mail land in train AND test and inflate the score
    return df[["text", "spam", "mailbox"]].reset_index(drop=True)   # only what we use


# ========================================================================= Multinomial NB from scratch
class ScratchMultinomialNB:
    """P(class | words) is proportional to P(class) * product over words of P(word | class)^count.

    Computed in LOG space (sums of logs instead of products of tiny probabilities, which underflow to 0).
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha                        # Laplace smoothing: pretend every word was seen alpha extra times

    def fit(self, X, y):
        self.classes_ = np.unique(y)              # e.g. [0, 1]
        counts = np.vstack([np.asarray(X[y == c].sum(axis=0)).ravel() for c in self.classes_])   # word totals per class
        smoothed = counts + self.alpha            # +alpha: a word never seen in spam still gets a tiny probability
        self.feature_log_prob_ = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))   # log P(word | class)
        self.class_log_prior_ = np.log(np.array([(y == c).mean() for c in self.classes_]))   # log P(class)
        return self                               # sklearn convention: fit returns the model

    def joint_log_likelihood(self, X):
        return np.asarray(X @ self.feature_log_prob_.T) + self.class_log_prior_   # sum of count*log P(w|c), + prior

    def predict_proba(self, X):
        jll = self.joint_log_likelihood(X)        # unnormalised log-probabilities per class
        jll = jll - jll.max(axis=1, keepdims=True)   # subtract the max before exp: no overflow (log-sum-exp trick)
        p = np.exp(jll)                           # back from logs
        return p / p.sum(axis=1, keepdims=True)   # normalise so each row sums to 1

    def predict(self, X):
        return self.classes_[self.joint_log_likelihood(X).argmax(axis=1)]   # the most likely class


# ========================================================================= experiments
def _scores(y, pred):
    """The four numbers we report for every model."""
    return {"accuracy": round(accuracy_score(y, pred), 4),        # share of e-mails classified correctly
            "spam_precision": round(precision_score(y, pred), 4), # of e-mails we binned, how many were really spam
            "spam_recall": round(recall_score(y, pred), 4),       # of all spam, how much we caught
            "ham_lost": int(((pred == 1) & (y == 0)).sum())}      # real e-mails sent to the spam folder (the worst error)


def compare_models(train, test):
    """Fit the NB family (and logistic regression) on bag-of-words / binary / TF-IDF features."""
    counts = CountVectorizer(min_df=2, max_features=50000)   # words seen in at least 2 e-mails, top 50k
    Xc_tr = counts.fit_transform(train["text"])   # learn the vocabulary on TRAIN only
    Xc_te = counts.transform(test["text"])        # re-use it on TEST (unknown words are ignored)
    Xb_tr, Xb_te = (Xc_tr > 0).astype(np.float64), (Xc_te > 0).astype(np.float64)   # present / absent
    tfidf = TfidfVectorizer(min_df=2, max_features=50000, sublinear_tf=True)   # down-weights very common words
    Xt_tr, Xt_te = tfidf.fit_transform(train["text"]), tfidf.transform(test["text"])   # TF-IDF matrices
    ytr, yte = train["spam"].values, test["spam"].values   # labels
    models = {                                    # name: (model, train matrix, test matrix)
        "MultinomialNB (counts)": (MultinomialNB(alpha=0.1), Xc_tr, Xc_te),   # the classic text NB
        "BernoulliNB (present/absent)": (BernoulliNB(alpha=0.1), Xb_tr, Xb_te),   # also uses ABSENT words
        "ComplementNB (TF-IDF)": (ComplementNB(alpha=0.1), Xt_tr, Xt_te),   # fixes NB's bias towards big classes
        "LogisticRegression (TF-IDF)": (LogisticRegression(C=10, max_iter=2000), Xt_tr, Xt_te),   # comparison
    }
    out = {name: _scores(yte, m.fit(A, ytr).predict(B)) for name, (m, A, B) in models.items()}   # fit and score all
    # GaussianNB assumes each word count follows a bell curve: wrong for sparse counts. It needs a dense matrix,
    # so it gets only the 2,000 most common words.
    small = CountVectorizer(max_features=2000).fit(train["text"])   # a vocabulary small enough to densify
    g = GaussianNB().fit(small.transform(train["text"]).toarray(), ytr)   # dense 2,000-column matrix
    out["GaussianNB (2k counts)"] = _scores(yte, g.predict(small.transform(test["text"]).toarray()))
    return out, counts, models["MultinomialNB (counts)"][0], (Xc_te, yte)   # the main model, for analysis


def alpha_sweep(train, test, alphas=(1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0)):
    """Laplace smoothing strength vs accuracy."""
    cv = CountVectorizer(min_df=2, max_features=50000)   # same features as above
    A, B = cv.fit_transform(train["text"]), cv.transform(test["text"])   # train / test matrices
    return {a: round(accuracy_score(test["spam"], MultinomialNB(alpha=a).fit(A, train["spam"]).predict(B)), 4)
            for a in alphas}                      # one accuracy per alpha


def top_words(model, vectorizer, X, k=15, min_count=200):
    """Words with the biggest log P(word|spam) - log P(word|ham): the model's strongest evidence.

    Only words seen at least min_count times: a word seen 3 times, all in spam, gets an extreme ratio by chance.
    """
    vocab = np.array(vectorizer.get_feature_names_out())   # column index -> word
    ratio = model.feature_log_prob_[1] - model.feature_log_prob_[0]   # > 0: more typical of spam
    keep = np.where(np.asarray(X.sum(axis=0)).ravel() >= min_count)[0]   # columns of frequent words only
    order = keep[np.argsort(ratio[keep])]         # frequent words, most ham-like first, most spam-like last
    return {"spam": vocab[order[-k:]][::-1].tolist(), "ham": vocab[order[:k]].tolist()}   # both ends


def mailbox_test(df):
    """The honest test: train on 5 people's mailboxes, test on the 6th (a different person AND spam source)."""
    res = {}                                      # results per held-out mailbox
    for box in sorted(df["mailbox"].unique()):    # enron1 ... enron6
        train, test = df[df["mailbox"] != box], df[df["mailbox"] == box]   # this person's e-mail is never seen
        cv = CountVectorizer(min_df=2, max_features=50000)   # vocabulary from the 5 training mailboxes
        A, B = cv.fit_transform(train["text"]), cv.transform(test["text"])   # word counts
        Ab, Bb = (A > 0).astype(np.float64), (B > 0).astype(np.float64)   # present/absent version
        tf = TfidfVectorizer(min_df=2, max_features=50000, sublinear_tf=True).fit(train["text"])   # TF-IDF version
        y, yt = train["spam"].values, test["spam"].values   # labels
        row = {"emails": len(test)}               # size of this mailbox
        for name, m, X1, X2 in (("MultinomialNB", MultinomialNB(alpha=0.1), A, B),
                                ("BernoulliNB", BernoulliNB(alpha=0.1), Ab, Bb),
                                ("LogisticRegression", LogisticRegression(C=10, max_iter=2000),
                                 tf.transform(train["text"]), tf.transform(test["text"]))):
            row[name] = _scores(yt, m.fit(X1, y).predict(X2))   # train on 5 mailboxes, predict the 6th
        res[box] = row                            # store
    return res


# ========================================================================= plots
def plot_confidence(model, X, y, path):
    """Naive Bayes is wildly over-confident: almost every probability is ~0 or ~1."""
    p = model.predict_proba(X)[:, 1]              # predicted P(spam)
    fig, ax = plt.subplots(figsize=(7, 4))        # one plot
    ax.hist(p[y == 0], bins=50, alpha=0.6, label="ham", log=True)   # log y-axis: the middle bins are nearly empty
    ax.hist(p[y == 1], bins=50, alpha=0.6, label="spam", log=True)  # same for spam
    ax.set(xlabel="predicted P(spam)", ylabel="e-mails (log scale)",
           title="Naive Bayes probabilities pile up at 0 and 1 (over-confidence)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)   # save, free memory


def plot_mailboxes(boxes, path):
    """Accuracy on each unseen mailbox, per model."""
    names = ["MultinomialNB", "BernoulliNB", "LogisticRegression"]   # bars per mailbox
    keys = list(boxes)                            # enron1..6
    x = np.arange(len(keys))                      # bar positions
    fig, ax = plt.subplots(figsize=(9, 4.5))      # wide plot
    for i, n in enumerate(names):                 # one coloured bar per model
        ax.bar(x + (i - 1) * 0.27, [boxes[k][n]["accuracy"] for k in keys], width=0.27, label=n)
    ax.set_xticks(x, keys); ax.set_ylim(0.6, 1.0)  # zoom in on the interesting range
    ax.set(ylabel="accuracy on the unseen mailbox", title="Leave-one-mailbox-out: generalising to a new person")
    ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_confusion(y, pred, path):
    cm = confusion_matrix(y, pred)                # [[ham kept, ham binned], [spam missed, spam caught]]
    fig, ax = plt.subplots(figsize=(4.5, 4))      # small square plot
    ax.imshow(cm, cmap="Blues")                   # colour by count
    for i in range(2):                            # write the counts in the cells
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["ham", "spam"]); ax.set_yticks([0, 1], ["ham", "spam"])   # class names
    ax.set(xlabel="predicted", ylabel="true", title="MultinomialNB, random split")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ========================================================================= run everything
def main():
    ap = argparse.ArgumentParser()                # options
    ap.add_argument("--quick", action="store_true")   # run on the 2,000-e-mail sample
    args = ap.parse_args()
    OUTPUTS.mkdir(exist_ok=True)                  # outputs/ folder
    df = load_emails(SAMPLE_DATA if args.quick else FULL_DATA)   # e-mails, duplicates removed
    train, test = train_test_split(df, test_size=0.2, stratify=df["spam"], random_state=0)   # random 80/20
    print(f"{len(df):,} unique e-mails ({df['spam'].mean():.1%} spam): train {len(train):,}, test {len(test):,}")
    res, counts, mnb, (Xte, yte) = compare_models(train, test)   # 1. the NB family vs logistic regression
    for k, v in res.items():
        print(f" {k:30s} {v}")
    scratch = ScratchMultinomialNB(alpha=0.1).fit(counts.transform(train["text"]), train["spam"].values)   # 2.
    agree = float(np.mean(scratch.predict(Xte) == mnb.predict(Xte)))   # same predictions as sklearn?
    alphas = alpha_sweep(train, test)             # 3. smoothing strength
    words = top_words(mnb, counts, counts.transform(train["text"]))   # 4. what the model relies on
    boxes = mailbox_test(df)                      # 5. the honest test: unseen mailboxes
    print("scratch agreement", agree)
    print("alpha", alphas)
    print("words", words)
    for k, v in boxes.items():
        print(" held out", k, {m: v[m]["accuracy"] for m in ("MultinomialNB", "BernoulliNB", "LogisticRegression")})
    plot_confidence(mnb, Xte, yte, OUTPUTS / "confidence.png")   # plots
    plot_confusion(yte, mnb.predict(Xte), OUTPUTS / "confusion.png")
    plot_mailboxes(boxes, OUTPUTS / "mailboxes.png")
    out = {"random_split": res, "scratch_agreement": agree, "alpha": {str(a): v for a, v in alphas.items()},
           "top_words": words, "leave_one_mailbox_out": boxes}   # everything for the README
    (OUTPUTS / "metrics.json").write_text(json.dumps(out, indent=2))
    print("saved to", OUTPUTS)


if __name__ == "__main__":                        # run only when started directly
    main()
