# Machine Learning Mini Projects

Ten hands-on projects, one per supervised ML algorithm (course sections 30-38 plus a capstone), each on a
**real, large, well-known dataset**: no toy data. The goal is to go from "I know the maths and the intuition" to
"I can write it myself".

**How every project works:** I give a brief (README with the concepts explained), a code skeleton with `TODO`
functions and hints, and tests. **You write the code, with a comment on every line.** A step is done when its
tests pass, and every project includes one algorithm built from scratch in NumPy that must match scikit-learn.

| # | Algorithm | Dataset | Status |
|---|---|---|---|
| 1 | Logistic Regression | Lending Club loans (2.26M) | **Done**: AUC 0.687 → 0.692 tuned; leakage demo 0.995 |
| 2 | Support Vector Machines | Fashion-MNIST (70k images) | **Done**: RBF 88.3% vs linear 84.0% |
| 3 | Naive Bayes | Enron-Spam, original corpus (30k unique e-mails) | **Done**: 98.8%; found the popular Kaggle copy is broken |
| 4 | K-Nearest Neighbours | Forest Covertype (581k) | **Done**: 96.5%; scaling made it worse, and why |
| 5 | Decision Trees (classifier + regressor) | NYC Taxi trips (2.96M) | **Done**: RMSE 5.10 min; tip AUC 0.94 → 0.71 honest |
| 6 | Random Forest | Credit Card Fraud, ULB (284k) | **Done**: PR-AUC 0.80; 72% of fraud € caught at 0.1% review |
| 7 | AdaBoost | HIGGS (11M collisions, UCI) | **Done**: AUC 0.785; physics features 0.757 vs raw 0.620 |
| 8 | Gradient Boosting | Rossmann store sales (1M store-days) | **Done**: RMSPE 0.121 vs 0.145 baseline; histograms 9x faster |
| 9 | XGBoost | Home Credit Default Risk (300k, 7 tables) | Planned |
| 10 | Capstone: all algorithms | IEEE-CIS Fraud Detection | Planned |

## Setup (already done on this machine)

```powershell
cd "D:\AI Projects\ML_Mini_Projects"
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```
Kaggle access: set the `KAGGLE_API_TOKEN` user environment variable. Full datasets live in `_datasets\`; each
project's `data\` folder holds a small sample so the tests run in seconds.
