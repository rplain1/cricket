import pytest
from unittest.mock import patch
import joblib
from train_model import load_data, select_model, train_model
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor


def test_load_data_all_files_present(tmp_path):
    file_names = [
        "train_features.pkl",
        "train_target.pkl",
        "val_features.pkl",
        "val_target.pkl",
        "test_features.pkl",
        "test_target.pkl",
    ]
    for file_name in file_names:
        joblib.dump([], tmp_path / file_name)

    data = load_data(path=tmp_path)

    assert set(data.keys()) == {
        "train_features",
        "train_target",
        "val_features",
        "val_target",
        "test_features",
        "test_target",
    }


def test_load_data_missing_files(tmp_path):
    joblib.dump([], tmp_path / "train_features.pkl")

    # Expect FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Missing the following files"):
        load_data(path=tmp_path)


def test_select_model():
    X_train, y_train = make_regression(n_samples=100, n_features=5, noise=0.1)
    X_val, y_val = make_regression(n_samples=30, n_features=5, noise=0.1)

    data = {
        "train_features": X_train,
        "train_target": y_train,
        "val_features": X_val,
        "val_target": y_val,
    }

    best_model, best_params = select_model(data)

    assert isinstance(best_model, KNeighborsRegressor)
    assert isinstance(best_params, dict)

    assert "n_neighbors" in best_params.keys()
    assert best_params["n_neighbors"] > 0


@patch("train_model.joblib.dump")
def test_train_model(mock_dump):
    X_train, y_train = make_regression(n_samples=100, n_features=5, noise=0.1)
    X_val, y_val = make_regression(n_samples=30, n_features=5, noise=0.1)
    X_test, y_test = make_regression(n_samples=20, n_features=5, noise=0.1)

    data = {
        "train_features": X_train,
        "train_target": y_train,
        "val_features": X_val,
        "val_target": y_val,
        "test_features": X_test,
        "test_target": y_test,
    }

    train_model(data, KNeighborsRegressor(n_neighbors=5), {"n_neighbors": 5})

    mock_dump.assert_called_once()
