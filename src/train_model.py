"""
I wanted to use KNN to do keep the model object as simple as possible, while also
having a meaningful parameter to tune. In development, I looked to see if I could
identify a bias-variance tradeoff and use that range. Becuase I didn't do any
feature engineering, it wasn't really effective at doing that. I don't think KNN
would be appropriate for this, but that is why I chose it.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import logging
import numpy as np
import joblib
import os
from pathlib import Path

MODEL_INPUT_PATH = "data/model-prep"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_data(path=MODEL_INPUT_PATH) -> dict:
    """
    Loads training, validation, and test datasets from the specified path.

    Args:
        path (str): Directory path where dataset files are located.

    Returns:
        dict: A dictionary containing loaded datasets for features and targets.

    Raises:
        FileNotFoundError: If any of the required files are missing.
    """
    datasets = [
        "train_features.pkl",
        "train_target.pkl",
        "val_features.pkl",
        "val_target.pkl",
        "test_features.pkl",
        "test_target.pkl",
    ]

    missing_files = [file for file in datasets if not Path(os.path.join(path, file)).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing the following files: {', '.join(missing_files)}")

    data = {}
    for file in datasets:
        dataset_name = file.split(".")[0]
        data[dataset_name] = joblib.load(os.path.join(path, file))

    return data


def select_model(data: dict):
    """
    Selects the best KNeighborsRegressor model based on cross-validated performance.

    Args:
        data (dict): A dictionary containing training and validation features and targets.

    Returns:
        tuple: The best trained model and its parameters.

    Logs:
        Information about the best model and its performance.
    """
    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(np.vstack([data["train_features"], data["val_features"]]))
    y_train_val = np.concatenate([data["train_target"], data["val_target"]])

    # Define the model and parameter grid
    model = KNeighborsRegressor()
    param_grid = {"n_neighbors": np.arange(5, 30)}

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        verbose=1,
    )
    grid_search.fit(X_train_val, y_train_val)
    best_n_neighbors = grid_search.best_params_["n_neighbors"]

    best_score = -grid_search.best_score_
    logging.info(f"Best model selected: {best_n_neighbors} neighbors with MSE: {best_score:.4f}")
    best_model = grid_search.best_estimator_

    return best_model, grid_search.best_params_


def train_model(data, model, params):
    """
    Trains the model on the full training and validation dataset, evaluates it on the test set,
    and saves the trained model.

    Args:
        data (dict): A dictionary containing training, validation, and test features and targets.
        model: A machine learning model to be trained.
        params (dict): Parameters used to configure the model (e.g., number of neighbors).

    Logs:
        TRAIN and TEST mean squared errors (MSE) of the final model.
    """
    X_train_full = np.concatenate([data["train_features"], data["val_features"]])
    y_train_full = np.concatenate([data["train_target"], data["val_target"]])

    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(X_train_full)
    X_test = scaler.transform(data["test_features"])

    model.fit(X_train_full, y_train_full)

    train_pred = model.predict(X_train_full)
    train_mse = mean_squared_error(y_train_full, train_pred)

    test_pred = model.predict(X_test)
    test_mse = mean_squared_error(data["test_target"], test_pred)

    logging.info(f"Final model - {params[list(params.keys())[0]]} neighbors TRAIN MSE: {train_mse:.4f}")
    logging.info(f"Final model - {params[list(params.keys())[0]]} neighbors TEST MSE: {test_mse:.4f}")

    out_dir = Path("models/")
    out_dir.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, os.path.join(out_dir, "model.pkl"))


if __name__ == "__main__":
    data = load_data()
    model, params = select_model(data)
    train_model(data, model, params)
