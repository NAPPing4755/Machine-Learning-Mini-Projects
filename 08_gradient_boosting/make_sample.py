# Builds data/rossmann_sample.csv (30 stores, every day, merged with store info) from the Kaggle download.
# Competition data may not be redistributed, so the sample is rebuilt locally instead of being committed.
import gradient_boosting as gb                    # reuse the project's loader

if __name__ == "__main__":
    d = gb.load()                                 # needs _datasets/rossmann/train.csv and store.csv
    d[d["Store"].isin(sorted(d["Store"].unique())[:30])].to_csv(gb.SAMPLE_DATA, index=False)   # first 30 stores
    print("wrote", gb.SAMPLE_DATA)
