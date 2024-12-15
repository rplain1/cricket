import pytest
import polars as pl
import joblib
from predict import load_data, preprocess_data, predict


@pytest.fixture
def test_data():
    return load_data("tests/data/test_data.csv")


@pytest.fixture
def model():
    return joblib.load("models/model.pkl")


def test_load_data(test_data):
    assert isinstance(test_data, pl.DataFrame)


def test_real_data(test_data, model):
    assert isinstance(test_data, pl.DataFrame)
    df = predict(model, test_data)
    assert isinstance(df, pl.DataFrame)


def test_preprocess_data_normal(test_data):
    data = preprocess_data(test_data)
    isinstance(data, pl.DataFrame)


def test_predictions_valid_output(test_data, model):
    df = predict(model, test_data)

    assert (df["pred_runs"] >= 0).all(), "predicted values are less than 0"
    assert df["pred_runs"].mean() <= df["runs"].max(), "predicted values don't align with the data"
    assert df["pred_runs"].max() <= df["runs"].max(), "predicted valeus are too high"


def test_null_input(test_data, model):
    test_data = test_data.with_columns(innings=pl.lit(None))
    with pytest.raises(ValueError, match="One or more columns contain all null values after preprocessing."):
        predict(model, test_data)


def test_unknown_team(test_data, model):
    test_data[1, "team"] = "NEW TEAM"
    with pytest.raises(ValueError, match="One or more teams is outside league of training data."):
        predict(model, test_data)
