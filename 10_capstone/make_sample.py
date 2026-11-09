# Builds data/ : 30,000 consecutive transactions (and their identity rows) for fast tests.
# Competition data may not be redistributed, so this sample is rebuilt locally and is not committed.
import pandas as pd                               # tables

import capstone as cp                             # paths

if __name__ == "__main__":
    t = pd.read_csv(cp.DATA_DIR / "train_transaction.csv", nrows=30000)   # the first 30k (consecutive in time)
    cp.SAMPLE_DIR.mkdir(exist_ok=True)
    t.to_csv(cp.SAMPLE_DIR / "train_transaction.csv", index=False)
    i = pd.read_csv(cp.DATA_DIR / "train_identity.csv")
    i[i["TransactionID"].isin(set(t["TransactionID"]))].to_csv(cp.SAMPLE_DIR / "train_identity.csv", index=False)
    print("sample written to", cp.SAMPLE_DIR)
