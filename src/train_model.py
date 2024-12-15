"""
I wanted to use KNN to do keep the model object as simple as possible, while also
having a meaningful parameter to tune. In development, I looked to see if I could
identify a bias-variance tradeoff to use. Becuase I didn't do any feature engineering,
using features like over that have a numeric representation but nominal meaning, it wasn't
really effective at doing that. I would have to do n=1
"""

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import logging
import numpy as np
import joblib
import os
from pathlib import Path

MODEL_INPUT_PATH = "data/model-prep"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_data(path=MODEL_INPUT_PATH) -> dict:
    datasets = [
        "train_features.pkl",
        "train_target.pkl",
        "val_features.pkl",
        "val_target.pkl",
        "test_features.pkl",
        "test_target.pkl",
    ]

    # Check if all files exist
    missing_files = [file for file in datasets if not Path(os.path.join(path, file)).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing the following files: {', '.join(missing_files)}")

    # Load all datasets into a dictionary
    data = {}
    for file in datasets:
        dataset_name = file.split(".")[0]  # create key for dataset
        data[dataset_name] = joblib.load(os.path.join(path, file))

    return data


def select_model(data: dict):
    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(data["train_features"])
    X_val = scaler.transform(data["val_features"])
    y_train, y_val = data["train_target"], data["val_target"]

    val_metrics = {}

    # Iterate over different values of n_neighbors to find the best value
    for n in range(1, 10):
        model = KNeighborsRegressor(n_neighbors=n)
        model.fit(X_train, y_train)

        # Predict and calculate MSE on validation set
        val_pred = model.predict(X_val)
        val_mse = mean_squared_error(y_val, val_pred)

        logging.info(f"Validation MSE for {n} neighbors: {val_mse:.4f}")
        val_metrics[n] = val_mse

    # Select the best parameter for neighbors
    best_n_neighbors = min(val_metrics, key=val_metrics.get)
    logging.info(f"Best model selected: {best_n_neighbors} neighbors")

    return best_n_neighbors


def train_model(data, best_n_neighbors):
    # Combine training and validation data
    X_train_full = np.concatenate([data["train_features"], data["val_features"]])
    y_train_full = np.concatenate([data["train_target"], data["val_target"]])

    # Standardize the full training set
    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(X_train_full)
    X_test = scaler.transform(data["test_features"])

    # Initialize and train the model with the best hyperparameter
    model = KNeighborsRegressor(n_neighbors=best_n_neighbors)
    model.fit(X_train_full, y_train_full)

    # Evaluate on the training set
    train_pred = model.predict(X_train_full)
    train_mse = mean_squared_error(y_train_full, train_pred)

    # Evaluate on the test set
    test_pred = model.predict(X_test)
    test_mse = mean_squared_error(data["test_target"], test_pred)

    logging.info(f"Final model - {best_n_neighbors} neighbors TRAIN MSE: {train_mse:.4f}")
    logging.info(f"Final model - {best_n_neighbors} neighbors TEST MSE: {test_mse:.4f}")

    out_dir = Path("models/")
    out_dir.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, os.path.join(out_dir, "model.pkl"))


if __name__ == "__main__":
    data = load_data()
    train_model(data, select_model(data))
