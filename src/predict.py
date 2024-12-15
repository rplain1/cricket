import joblib
import polars as pl
import argparse
from sklearn.neighbors import KNeighborsRegressor


def load_model(model_path: str) -> KNeighborsRegressor:
    """
    Load the model object at specified `model_path`.

    Args:
        model_path (str): location of the model object

    Returns:
        KNeighborsRegressor: trained model object
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def load_data(data_path: str) -> pl.DataFrame:
    """
    Reads into a polars DataFrame the specified `data_path`.
    Must be in CSV or JSON format.

    Args:
        data_path (str): location of the data

    Returns:
        pl.DataFrame: data for predictions
    """
    return pl.read_csv(data_path) if data_path.endswith(".csv") else pl.read_json(data_path)


def predict(model: KNeighborsRegressor, data: pl.DataFrame) -> pl.DataFrame:
    """
    Runs the model on the new data for predictions. Before making predictions,
    there is a check to ensure the trained on columns will be in the dataset

    Args:
        model (KNeighborsRegressor): model object
        data (pl.DataFrame): new data to run predictions on

    Returns:
        pl.DataFrame: `data` object with a new column added for predicted values
    """
    required_columns = ["innings", "overs_remaining", "wickets_remaining"]

    try:
        missing_columns = [col for col in required_columns if col not in data.columns]
        assert not missing_columns, f"Missing columns: {missing_columns}"
    except AssertionError as e:
        raise ValueError(f"Dataset validation failed: {e}")

    features = data.select(required_columns)
    predictions = model.predict(features)

    return data.with_columns(pred_runs=predictions)


def main():
    """
    Entry point for running predictions using the cricket model.

    This function parses command-line arguments, loads the prediction model
    and input data, generates predictions, and saves the results to the specified output file.

    Command-Line Arguments:
        --data (str, required): Path to the input dataset file in CSV or JSON format.
        --model (str, required): Path to the model file in PKL (Pickle) format.
        --output (str, required): Path to save the predictions in CSV format.

    Workflow:
        1. Parse command-line arguments for the input data, model, and output paths.
        2. Load the cricket prediction model using the `load_model()` function.
        3. Load the input dataset using the `load_data()` function.
        4. Compute predictions using the `predict()` function.
        5. Save the predictions to the specified output file in CSV format.
        6. Print a confirmation message indicating the saved file's location.

    Raises:
        ValueError: If required arguments are missing or if file paths are invalid.

    Example Usage:
        python script.py --data matches.csv --model cricket_model.pkl --output predictions.csv

    """
    parser = argparse.ArgumentParser(description="Run predictions using the cricket model")
    parser.add_argument("--data", required=True, help="Path to input data (CSV/JSON)")
    parser.add_argument("--model", required=True, help="Path to model file (PKL)")
    parser.add_argument("--output", required=True, help="Path to save predictions (CSV)")

    args = parser.parse_args()

    model = load_model(args.model)
    data = load_data(args.data)

    predictions = predict(model, data)

    predictions.write_csv(args.output)
    print("Predictions saved to:", args.output)


if __name__ == "__main__":
    main()
