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


def time_series_split(data: pl.DataFrame, key="dates") -> dict:
    """
    Perform a simple split of the data to prevent data leakage as the model is
    dealing with time series data.

    Args:
        data: (pl.DataFrame): input dataset

    Returns:
        dict: dictionary with train, validation, and test keys and dates
    """

    # without diving too much into how many games are in each time period
    # this was a simple approach to get data split into 70/15/15 split
    date_indices = (
        data.group_by(key)
        .len()
        .sort(key)
        .with_columns(perc=pl.col("len") / pl.col("len").sum())
        .with_columns(perc=pl.col("perc").cum_sum())
    )

    train_idx = date_indices.filter(pl.col("perc") <= 0.7)[key].unique()
    val_idx = date_indices.filter((pl.col("perc") > 0.7) & (pl.col("perc") <= 0.85))[key].unique()
    test_idx = date_indices.filter(pl.col("perc") > 0.85)[key].unique()

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def split_feature_target(data: pl.DataFrame, target="runs") -> tuple[np.ndarray, np.ndarray]:
    """
    Simple function to split X and y

    Args:
        data (pl.DataFrame) model input data

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: tuple of X and y arrays

    """
    X = data.select(pl.exclude(target)).to_numpy()
    y = data.select(target).to_numpy().flatten()

    return X, y


def main(data: pl.DataFrame) -> None:
    date_idx = time_series_split(data)
    logging.info(f"Data was split into: {date_idx.keys()}")

    data = data.select(["dates", "runs", "over", "wickets_remaining"])

    for key in date_idx.keys():
        tmp_data = data.filter(pl.col("dates").is_in(date_idx[key]))

        # removing dates here, although in practice there would likely be
        # some feature engineering to extract datetime components in the model
        drop_col = date_idx[key].name
        tmp_data = tmp_data.drop(drop_col)

        X, y = split_feature_target(tmp_data)
        logging.info(f"{key} features: {X.shape}")
        logging.info(f"{key} target: {y.shape}")

        out_dir = Path("data/model-prep/")
        out_dir.mkdir(exist_ok=True, parents=True)

        joblib.dump(X, f"data/model-prep/{key}_features.pkl")
        joblib.dump(y, f"data/model-prep/{key}_target.pkl")


if __name__ == "__main__":
    data = pl.read_parquet("data/transformed/transformed_data.parquet")
    main(data)
