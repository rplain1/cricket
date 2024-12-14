"""
This is to take the data that is staged, and save files for TRAINING, VALIDATION,
and TESTING. I will be writing these into parquet files. This layer should be free
of buisness logic, and focus on preparing staged data for modeling.

I'm doing a custom split here, with the idea being that the data has a time time
series component, and you need to adjust for data leakage. With more time, the model
might have a time component added, and you could do a rolling time series cross valdiattion.

This is a simple approach, but I also wanted to do something more robust than
sci-kit learns `train_test_split()`

"""

from pathlib import Path
import numpy as np
import polars as pl
import logging
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def split_data(data: pl.DataFrame):
    data = data.sort(by=data.columns).with_row_index()

    train = data.sample(fraction=0.7, seed=427)
    val = data.join(train, on="index", how="anti")
    test = val.sample(fraction=0.5, seed=527)
    val = val.join(test, on="index", how="anti")

    return {"train": train.drop("index"), "val": val.drop("index"), "test": test.drop("index")}


def split_feature_target(data: pl.DataFrame, target="runs") -> tuple[np.ndarray, np.ndarray]:
    """
    Simple function to split X and y

    Args:
        data (pl.DataFrame) model input data

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: tuple of X and y arrays

    """
    assert target in data.columns, f"{target} column not in dataset"
    X = data.select(pl.exclude(target)).to_numpy()
    y = data.select(target).to_numpy().flatten()

    return X, y


def main(data: pl.DataFrame) -> None:
    data = split_data(data)

    for key in data.keys():
        X, y = split_feature_target(data[key])
        logging.info(f"{key} features: {X.shape}")
        logging.info(f"{key} target: {y.shape}")

        out_dir = Path("data/model-prep/")
        out_dir.mkdir(exist_ok=True, parents=True)

        joblib.dump(X, f"data/model-prep/{key}_features.pkl")
        joblib.dump(y, f"data/model-prep/{key}_target.pkl")


if __name__ == "__main__":
    data = pl.read_parquet("data/transformed/transformed_data.parquet")
    main(data)
