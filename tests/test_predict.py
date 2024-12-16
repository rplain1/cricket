import pytest
import polars as pl
import joblib
from predict import load_data, preprocess_data, predict


@pytest.fixture
def test_data():
    """
    Fixture that loads test data from a CSV file.

    Returns:
        pl.DataFrame: The loaded test data.
    """
    return load_data("tests/data/test_data.csv")


@pytest.fixture
def model():
    """
    Fixture that loads a pre-trained model from a file.

    Returns:
        model: The loaded machine learning model.
    """
    return joblib.load("models/model.pkl")


def test_load_data(test_data):
    """
    Tests if the loaded data is of type pl.DataFrame.

    Args:
        test_data (pl.DataFrame): The test data loaded by the fixture.
    """
    assert isinstance(test_data, pl.DataFrame)


def test_real_data(test_data, model):
    """
    Tests if real data predictions are processed correctly.

    Args:
        test_data (pl.DataFrame): The test data loaded by the fixture.
        model: The pre-trained model loaded by the fixture.

    Asserts:
        - test_data is of type pl.DataFrame.
        - Predictions from the model result in a DataFrame.
    """

    assert isinstance(test_data, pl.DataFrame)
    df = predict(model, test_data)
    assert isinstance(df, pl.DataFrame)


def test_preprocess_data_normal(test_data):
    """
    Tests if data preprocessing returns a DataFrame.

    Args:
        test_data (pl.DataFrame): The input test data.

    Asserts:
        - The output data is of type pl.DataFrame.
    """
    data = preprocess_data(test_data)
    isinstance(data, pl.DataFrame)


def test_predictions_valid_output(test_data, model):
    """
    Tests if the model's predictions are valid and follow sports logic.

    Args:
        test_data (pl.DataFrame): The input test data.
        model: The pre-trained model used for predictions.

    Asserts:
        - Predictions are non-negative.
        - Predictions are within the range of actual values.
        - Predictions do not exceed the maximum value in the actual data.
    """
    df = predict(model, test_data)

    assert (df["pred_runs"] >= 0).all(), "predicted values are less than 0"
    assert df["pred_runs"].mean() <= df["runs"].max(), "predicted values don't align with the data"
    assert df["pred_runs"].max() <= df["runs"].max(), "predicted valeus are too high"


def test_null_input(test_data, model):
    """
    Tests if the model raises an error when input data contains all null values.

    Args:
        test_data (pl.DataFrame): The input test data.
        model: The pre-trained model used for predictions.

    Asserts:
        - Raises a ValueError if any column contains all null values after preprocessing.
    """
    test_data = test_data.with_columns(innings=pl.lit(None))
    with pytest.raises(ValueError, match="One or more columns contain all null values after preprocessing."):
        predict(model, test_data)


def test_unknown_team(test_data, model):
    """
    Tests if the model raises an error when the input data contains an unknown team.

    Args:
        test_data (pl.DataFrame): The input test data.
        model: The pre-trained model used for predictions.

    Asserts:
        - Raises a ValueError if the input data contains a team not present in the training data.
    """
    test_data[1, "team"] = "NEW TEAM"
    with pytest.raises(ValueError, match="One or more teams is outside league of training data."):
        predict(model, test_data)
