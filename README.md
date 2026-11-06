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
| 2 | Support Vector Machines | Fashion-MNIST (70k images) | Planned |
| 3 | Naive Bayes | Enron spam corpus (33k emails) | Planned |
| 4 | K-Nearest Neighbours | Forest Covertype (581k) | Planned |
| 5 | Decision Trees (classifier + regressor) | NYC Taxi trips | Planned |
| 6 | Random Forest | Credit Card Fraud, ULB (284k) | Planned |
| 7 | AdaBoost | HIGGS boson, CERN/UCI | Planned |
| 8 | Gradient Boosting | Rossmann store sales (1M+) | Planned |
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
