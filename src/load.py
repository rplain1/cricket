"""
This part I spent too much time on, and had to come back and trim it down after
better understanding the requirements. Because there was a time component in the
data, I first sought to do a custom split on `matchid` and `dates`, that would prevent
data leakage. I still think that is the appropriate next step for the project, but
given the requirements and recommended time to spend on this, I think this is about
right.

This is to take the data that is staged, and save files for TRAINING, VALIDATION,
and TESTING (would have just done TRAINING/TESTING and used GridSearchCV in `train_model.py`
now that I better understand the requirements). I will be writing these into
parquet files. This layer should be free of sports logic, and focus on preparing
staged data for modeling.å
"""

from pathlib import Path
import numpy as np
import polars as pl
import logging
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def split_data(data: pl.DataFrame) -> dict[pl.DataFrame, pl.DataFrame, pl.dataframe]:
    """
    Split the data into training, testing, and validation sets.

    Args:
        data (pl.DataFrame): total available training data

    Returns:
        dict: dictionary with 3 keys for train, val, and test
    """
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
    X = data.select(pl.exclude(target, "team")).to_numpy()
    y = data.select(target).to_numpy().flatten()

    return X, y


def main(data: pl.DataFrame) -> None:
    """
    Main function to take in the input data, and run the split operations to
    create the features and targets for model training.

    Steps:
        1. split the data into training, validation, testing
        2. for each set, split the features and targets into separate numpy
           arrays
        3. Save the numpy objects to `model-prep/` for training

    Returns:
        None
    """
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
