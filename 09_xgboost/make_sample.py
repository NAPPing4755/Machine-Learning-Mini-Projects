# Builds data/ : the same 7 tables restricted to 10,000 random applicants, for fast tests.
# Competition data may not be redistributed, so this sample is rebuilt locally and is not committed.
import pandas as pd                               # tables

import xgboost_credit as xc                       # paths and table names

if __name__ == "__main__":
    app = pd.read_csv(xc.DATA_DIR / "application_train.csv")   # the main table
    keep = app.sample(10000, random_state=0)      # 10k applicants
    xc.SAMPLE_DIR.mkdir(exist_ok=True)
    keep.to_csv(xc.SAMPLE_DIR / "application_train.csv", index=False)
    ids = set(keep["SK_ID_CURR"])                 # their ids
    bureau = pd.read_csv(xc.DATA_DIR / "bureau.csv"); bureau = bureau[bureau["SK_ID_CURR"].isin(ids)]
    bureau.to_csv(xc.SAMPLE_DIR / "bureau.csv", index=False)
    bb = pd.read_csv(xc.DATA_DIR / "bureau_balance.csv")   # linked through the bureau loan id
    bb[bb["SK_ID_BUREAU"].isin(set(bureau["SK_ID_BUREAU"]))].to_csv(xc.SAMPLE_DIR / "bureau_balance.csv", index=False)
    for t in ("previous_application", "installments_payments", "POS_CASH_balance", "credit_card_balance"):
        d = pd.read_csv(xc.DATA_DIR / f"{t}.csv")
        d[d["SK_ID_CURR"].isin(ids)].to_csv(xc.SAMPLE_DIR / f"{t}.csv", index=False)
    print("sample written to", xc.SAMPLE_DIR)
