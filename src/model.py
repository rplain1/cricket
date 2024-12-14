import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from config import MODEL_INPUT_PATH
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def simple_time_series_split(data: pl.DataFrame) -> dict:
    """
    Perform a simple split of the data to prevent data leakage as the model is
    dealing with time series data.

    Args:
        data: (pl.DataFrame): input dataset

    Returns:
        dict: dictionary with train, validation, and test keys and dates
    """

    date_indices = (
        data.group_by("dates")
        .len()
        .sort("dates")
        .with_columns(perc=pl.col("len") / pl.col("len").sum())
        .with_columns(perc=pl.col("perc").cum_sum())
    )

    train_idx = date_indices.filter(pl.col("perc") <= 0.7)["dates"].unique()
    val_idx = date_indices.filter((pl.col("perc") > 0.7) & (pl.col("perc") <= 0.85))["dates"].unique()
    test_idx = date_indices.filter(pl.col("perc") > 0.85)["dates"].unique()

    return {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}


def split_train_test(data: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple function to split X and y

    Args:
        data (pl.DataFrame) model input data

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: tuple of X and y arrays

    """
    X = data.select(pl.exclude("runs")).to_numpy()
    y = data.select("runs").to_numpy().flatten()

    return X, y


def train_model(data):
    date_idx = simple_time_series_split(data)

    train_data = data.filter(pl.col("dates").is_in(date_idx["train_idx"]))
    val_data = data.filter(pl.col("dates").is_in(date_idx["val_idx"]))

    train_data = train_data.drop("dates")
    val_data = val_data.drop("dates")

    X_train, y_train = split_train_test(train_data)
    X_val, y_val = split_train_test(val_data)

    model = LinearRegression()
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, train_pred)

    val_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_pred)

    logging.info(f"Training MSE: {train_mse:.4f}")
    logging.info(f"Validation MSE: {val_mse:.4f}")


if __name__ == "__main__":
    data = pl.read_parquet(MODEL_INPUT_PATH)
    data = data.select(["dates", "over", "ball", "wickets_remaining", "runs"])
    train_model(data)
